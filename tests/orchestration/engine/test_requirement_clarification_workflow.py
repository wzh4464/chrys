# Copyright (c) 2026 Chrys. All rights reserved.

"""Route-A ordering, fresh history, and P0 fallback tests."""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import chrys.orchestration.engine.run.requirement_clarification as workflow_module
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import AgentMessage, Error, RequirementClarificationPhaseChanged
from chrys.foundation.models.workspace import Workspace
from chrys.kernel import Message
from chrys.orchestration.engine.run.requirement_clarification import (
    RequirementClarificationWorkflow,
    _history_background,
)
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot
from chrys.service.requirement_clarification.types import (
    ClarificationResult,
    ClarificationSelection,
    RequirementWorkflowPhase,
)
from chrys.service.session.history import SessionHistoryManager


@dataclass
class _HistoryCheckpoint:
    state: dict[str, Any]

    @property
    def messages(self):
        return self.state["messages"]


class _Executor:
    def __init__(self, workspace_file: Path, *, repair_fails: bool = False, repair_delay: float = 0) -> None:
        self.history_state = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
        self.service_session_id = "provider-h0"
        self.run_failed = False
        self.was_interrupted = False
        self.last_response_text = ""
        self.is_running = False
        self.phase = ""
        self.workspace_file = workspace_file
        self.repair_fails = repair_fails
        self.repair_delay = repair_delay
        self.run_calls = 0
        self.repair_started_from: list[str] = []
        self.published_finals: list[str] = []

    def snapshot_history(self) -> _HistoryCheckpoint:
        return _HistoryCheckpoint(copy.deepcopy(self.history_state))

    def restore_history(self, snapshot: _HistoryCheckpoint) -> None:
        self.history_state = copy.deepcopy(snapshot.state)

    def set_requirement_phase(self, phase: str) -> None:
        self.phase = phase

    def set_user_messages(self, _messages: list[str]) -> None:
        return

    def reset_counters(self, *, reset_batch_id: bool = True) -> None:
        assert reset_batch_id is True

    async def run(self, contents: list[Any], created_at=None) -> None:
        _ = created_at
        self.run_calls += 1
        if self.repair_delay:
            await asyncio.sleep(self.repair_delay)
        self.repair_started_from = [message.text for message in self.history_state["messages"]]
        assert self.workspace_file.read_text(encoding="utf-8") == "P0\n"
        self.history_state["messages"].extend([Message("user", contents), Message("assistant", ["P1"])])
        self.last_response_text = "P1"
        self.run_failed = self.repair_fails

    def adopt_fallback_success(self, text: str) -> None:
        self.run_failed = False
        self.was_interrupted = False
        self.last_response_text = text

    async def publish_last_response_as_final(self) -> None:
        self.published_finals.append(self.last_response_text)

    async def interrupt(self) -> None:
        self.was_interrupted = True


class _Runner:
    def __init__(self, host, workspace_file: Path) -> None:
        self.host = host
        self.workspace_file = workspace_file
        self.finalized = 0
        self.imported_p0_prepared = 0

    async def _run_fresh_standard(self, text: str, **kwargs) -> None:
        assert kwargs["finalize"] is False
        self.workspace_file.write_text("P0\n", encoding="utf-8")
        self.host._executor.history_state["messages"] = [Message("user", [text]), Message("assistant", ["P0"])]
        self.host._executor.last_response_text = "P0"
        self.host._turn_number = 1

    async def finalize_current_run(self) -> None:
        self.finalized += 1

    async def _prepare_fresh_without_execution(self, _text: str, **_kwargs) -> None:
        assert self.workspace_file.read_text(encoding="utf-8") == "P0\n"
        self.imported_p0_prepared += 1


class _Snapshotter:
    def __init__(self, workspace_file: Path) -> None:
        self.workspace_file = workspace_file
        self.calls: list[str] = []
        self.restored = 0

    def capture(
        self,
        _workspace,
        artifact_root: Path,
        *,
        snapshot_id: str,
        include_git_history: bool,
        committed_git_head_only: bool = False,
    ):
        artifact_root.mkdir(parents=True)
        self.calls.append("s0-head" if committed_git_head_only else "s0" if include_git_history else "p0")
        return WorkspaceSnapshot(
            snapshot_id=snapshot_id,
            artifact_root=str(artifact_root),
            roots=(),
            manifest_hash=snapshot_id,
            total_bytes=0,
            entry_count=0,
        )

    def matches(self, _snapshot) -> bool:
        return True

    def restore(self, _snapshot) -> None:
        self.restored += 1
        self.workspace_file.write_text("P0\n", encoding="utf-8")

    def discard(self, _snapshot) -> None:
        return


class _Reminder:
    def __init__(self) -> None:
        self.values: list[str] = []

    def queue_hook_reminders(self, values: list[str]) -> None:
        self.values.extend(values)


def _result() -> ClarificationResult:
    return ClarificationResult(
        strategy_version="test-v1",
        revision=1,
        delta="Repository implementation guidance:\n- wire the option",
        selection=ClarificationSelection(),
    )


def test_history_background_uses_only_user_and_assistant_text() -> None:
    state = {
        "compressed_msgs": [SimpleNamespace(messages=[Message("user", ["older requirement"])])],
        "messages": [
            Message("tool", ["private tool result"]),
            Message("assistant", ["prior answer"]),
        ],
    }

    assert _history_background(state) == "user: older requirement\nassistant: prior answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_fails", [False, True, "timeout"])
@pytest.mark.parametrize("reuse_workspace_as_p0", [False, True])
@pytest.mark.parametrize("clarification_only", [False, True])
async def test_workflow_orders_p0_before_delta_and_restores_p0_on_repair_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repair_fails: bool | str,
    reuse_workspace_as_p0: bool,
    clarification_only: bool,
) -> None:
    async def _direct(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    order: list[str] = []

    class _Service:
        def __init__(self, _model, **_kwargs) -> None:
            return

        async def clarify(self, **_kwargs):
            order.append("delta")
            return _result()

        async def generate_pact_input(self, **_kwargs):
            clarification_path = tmp_path / "session/requirement_clarification/turn_1/clarification.private.json"
            assert clarification_path.is_file()
            raise RuntimeError("simulated optional PACT failure")

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", _direct)
    monkeypatch.setattr(workflow_module, "ClarificationService", _Service)
    monkeypatch.setattr(workflow_module, "ChrysClarificationModel", lambda **_kwargs: object())

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace_file = workspace_root / "value.txt"
    workspace_file.write_text("P0\n" if reuse_workspace_as_p0 else "S0\n", encoding="utf-8")
    executor = _Executor(
        workspace_file,
        repair_fails=repair_fails is True,
        repair_delay=0.05 if repair_fails == "timeout" else 0,
    )
    history = SessionHistoryManager()
    history.bind(executor.history_state)
    host = SimpleNamespace(
        _active_profile=object(),
        _agent_profile_fingerprint="agent-fp",
        _model_profile_fingerprint="model-fp",
        _bus=EventBus(),
        _consumed_injections=[],
        _executor=executor,
        _history=history,
        _intermediate_texts={},
        _reminder_middleware=_Reminder(),
        _session_id="session",
        _turn_number=0,
        _turn_state=SimpleNamespace(close_injection_admission=lambda _scope: None),
        _workspace=Workspace.from_cwd(str(workspace_root)),
        _session_dir=tmp_path / "session",
        _requirement_clarification_workflow=None,
        _accumulate_side_call_usage=lambda _usage: None,
    )
    runner = _Runner(host, workspace_file)
    snapshotter = _Snapshotter(workspace_file)
    workflow = RequirementClarificationWorkflow(
        host,
        runner,
        reuse_workspace_as_p0=reuse_workspace_as_p0,
        clarification_only=clarification_only,
        repair_timeout_seconds=0.001 if repair_fails == "timeout" else 5400,
    )
    workflow._snapshotter = snapshotter
    phases: list[str] = []
    finals: list[str] = []

    async def _phase(event: RequirementClarificationPhaseChanged) -> None:
        phases.append(event.phase)

    async def _message(event: AgentMessage) -> None:
        if event.is_final:
            finals.append(event.text)

    await host._bus.subscribe(RequirementClarificationPhaseChanged, _phase)
    await host._bus.subscribe(AgentMessage, _message)
    await workflow.run(
        "implement it",
        created_at=None,
        contents=["implement it"],
        run_scope=None,
        injection_window=None,
        admission_preparation=None,
    )

    assert snapshotter.calls == ["s0-head" if reuse_workspace_as_p0 else "s0", "p0"]
    assert runner.imported_p0_prepared == int(reuse_workspace_as_p0)
    assert order == ["delta"]
    assert executor.repair_started_from == []
    assert runner.finalized == 1
    artifact_root = host._session_dir / "requirement_clarification/turn_1"
    assert (artifact_root / "01-input/requirement.md").is_file()
    assert (artifact_root / "01-input/workspace-snapshot.json").is_file()
    assert (artifact_root / "02-initial-trial/response.json").is_file()
    assert (artifact_root / "03-clarification/deliverable/manifest.json").is_file()
    assert (artifact_root / "06-pact-input/generation.private.json").is_file()
    repair_response = artifact_root / "04-repair/attempts/revision-1/response.json"
    assert (artifact_root / "05-outcome/summary.json").is_file()
    assert (artifact_root / "05-outcome/clarified-requirement.md").is_file()
    assert (artifact_root / "05-outcome/clarified-requirement-delta.md").is_file()
    if clarification_only:
        assert executor.run_calls == 0
        assert host._reminder_middleware.values == []
        assert not repair_response.exists()
        expected_p0_text = "Reused the existing workspace implementation as P0." if reuse_workspace_as_p0 else "P0"
        assert executor.published_finals == [expected_p0_text]
        assert phases[-1] == RequirementWorkflowPhase.COMPLETED
        return
    assert executor.run_calls == 1
    assert host._reminder_middleware.values[0].startswith("[REQUIREMENT_CLARIFICATION_REPAIR]")
    assert repair_response.is_file()
    if repair_fails:
        assert snapshotter.restored == 1
        expected_p0_text = "Reused the existing workspace implementation as P0." if reuse_workspace_as_p0 else "P0"
        assert executor.published_finals == [expected_p0_text]
        assert workspace_file.read_text(encoding="utf-8") == "P0\n"
        assert phases[-1] == RequirementWorkflowPhase.DEGRADED
        assert json.loads(repair_response.read_text(encoding="utf-8"))["status"] in {"failed", "timed_out"}
    else:
        assert snapshotter.restored == 0
        assert executor.published_finals == []
        assert phases[-1] == RequirementWorkflowPhase.COMPLETED
        assert json.loads(repair_response.read_text(encoding="utf-8"))["status"] == "succeeded"


async def test_a_failed_rollback_ends_the_turn_instead_of_leaving_it_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A phase event is progress, not an outcome.

    The TUI keeps a turn open until a final answer, an `Error`, or the user's
    own Stop arrives, and headless `chrys run` decides its exit status the same
    way. A repair that failed AND could not roll back delivers no answer at
    all — so without a terminal the spinner runs forever, the input bar stays
    locked, and a script exits 0 on a half-repaired workspace nobody told it
    about.
    """

    async def _direct(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    class _Service:
        def __init__(self, _model, **_kwargs) -> None:
            return

        async def clarify(self, **_kwargs):
            return _result()

        async def generate_pact_input(self, **_kwargs):
            raise RuntimeError("simulated optional PACT failure")

    class _BrokenSnapshotter(_Snapshotter):
        def restore(self, _snapshot) -> None:
            raise OSError("read-only filesystem")

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", _direct)
    monkeypatch.setattr(workflow_module, "ClarificationService", _Service)
    monkeypatch.setattr(workflow_module, "ChrysClarificationModel", lambda **_kwargs: object())

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace_file = workspace_root / "value.txt"
    workspace_file.write_text("S0\n", encoding="utf-8")
    executor = _Executor(workspace_file, repair_fails=True)
    history = SessionHistoryManager()
    history.bind(executor.history_state)
    host = SimpleNamespace(
        _active_profile=object(),
        _agent_profile_fingerprint="agent-fp",
        _model_profile_fingerprint="model-fp",
        _bus=EventBus(),
        _consumed_injections=[],
        _executor=executor,
        _history=history,
        _intermediate_texts={},
        _reminder_middleware=_Reminder(),
        _session_id="session",
        _turn_number=0,
        _turn_state=SimpleNamespace(close_injection_admission=lambda _scope: None),
        _workspace=Workspace.from_cwd(str(workspace_root)),
        _session_dir=tmp_path / "session",
        _requirement_clarification_workflow=None,
        _accumulate_side_call_usage=lambda _usage: None,
    )
    runner = _Runner(host, workspace_file)
    workflow = RequirementClarificationWorkflow(host, runner, reuse_workspace_as_p0=False, clarification_only=False)
    workflow._snapshotter = _BrokenSnapshotter(workspace_file)
    errors: list[Error] = []
    phases: list[str] = []

    async def _error(event: Error) -> None:
        errors.append(event)

    async def _phase(event: RequirementClarificationPhaseChanged) -> None:
        phases.append(event.phase)

    await host._bus.subscribe(Error, _error)
    await host._bus.subscribe(RequirementClarificationPhaseChanged, _phase)

    await workflow.run(
        "implement it",
        created_at=None,
        contents=["implement it"],
        run_scope=None,
        injection_window=None,
        admission_preparation=None,
    )

    assert phases[-1] == RequirementWorkflowPhase.CONFLICTED
    assert executor.published_finals == []
    assert [error.code for error in errors] == ["requirement_clarification_conflicted"]
    assert "rollback failed" in errors[0].message
