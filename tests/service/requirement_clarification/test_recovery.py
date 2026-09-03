# Copyright (c) 2026 Chrys. All rights reserved.

"""Durable P0 recovery and workflow-record tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import chrys.service.session.lifecycle as lifecycle_module
from chrys.foundation.events.bus import EventBus
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.workspace import Workspace
from chrys.kernel import Message
from chrys.service.requirement_clarification.artifacts import (
    ClarificationArtifactStore,
    latest_incomplete_workflow,
)
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshotter
from chrys.service.session.history import SessionHistoryManager
from chrys.service.state.serializers import serialize_state


class _Executor:
    def __init__(self) -> None:
        self.history_state = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
        self.service_session_id = "stale-provider-handle"


@pytest.mark.asyncio
async def test_restore_promotes_matching_p0_and_terminalizes_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _direct(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(lifecycle_module.asyncio, "to_thread", _direct)
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "implementation.py").write_text("P0 = True\n", encoding="utf-8")
    session_dir = tmp_path / "session"
    artifacts = ClarificationArtifactStore(session_dir, 1)
    snapshot = WorkspaceSnapshotter().capture(
        Workspace.from_cwd(str(workspace_root)),
        artifacts.root / "p0",
        snapshot_id="p0",
        include_git_history=False,
    )
    assert WorkspaceSnapshotter.load(Path(snapshot.artifact_root)).manifest_hash == snapshot.manifest_hash
    p0_state = {
        "messages": [Message("user", ["implement"]), Message("assistant", ["baseline"])],
        "compressed_msgs": [],
        "turn_counter": 0,
    }
    artifacts.save_initial_transcript({"history": serialize_state(p0_state), "service_session_id": "p0-handle"})
    artifacts.save_workflow_record(
        {
            "version": 1,
            "workflow_id": "workflow",
            "phase": "clarification",
            "terminal": False,
            "revision": {"number": 1, "messages": ["implement"]},
            "p0": {"manifest_hash": snapshot.manifest_hash},
        }
    )
    executor = _Executor()
    history = SessionHistoryManager()
    history.bind(executor.history_state)
    engine = SimpleNamespace(
        _session_dir=session_dir,
        _session_id="session",
        _executor=executor,
        _history=history,
        event_bus=EventBus(),
    )

    assert await lifecycle_module._recover_incomplete_requirement_workflow(engine) is True

    assert [message.text for message in executor.history_state["messages"][:2]] == ["implement", "baseline"]
    assert executor.history_state["messages"][-1].additional_properties[HistoryMarkerKind.KEY] == HistoryMarkerKind.TURN
    assert executor.service_session_id == ""
    assert not (artifacts.root / "p0").exists()
    assert latest_incomplete_workflow(session_dir) is None
    record = json.loads((artifacts.root / "workflow.json").read_text(encoding="utf-8"))
    assert record["terminal"] is True
    assert record["recovered_after_crash"] is True
