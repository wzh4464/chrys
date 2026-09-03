# Copyright (c) 2026 Chrys. All rights reserved.

"""The engine drives ContextGraph writeback and persists the watermark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import MemoryWritebackCompleted
from chrys.foundation.models.workspace import Workspace
from chrys.kernel import Content, Message
from chrys.orchestration.engine import engine as engine_module
from chrys.service.memory.writeback import WritebackOutcome
from chrys.service.state.serializers import serialize_state
from chrys.service.state.store import SESSION_FILE_NAME, JsonFileStateStore

_URI = "bolt://127.0.0.1:7687"


def _tool_turn(index: int) -> list[Message]:
    return [
        Message("user", [Content.from_text(f"Fix failure {index}")]),
        Message("assistant", [Content.from_function_call(f"shell-{index}", "shell", arguments={"command": "ls"})]),
        Message("tool", [Content.from_function_result(f"shell-{index}", result=f"done {index}")]),
        Message("assistant", [Content.from_text(f"Fixed {index}.")]),
    ]


def _seed_session(session_dir: Path, turns: int) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    messages: list[Message] = []
    for index in range(1, turns + 1):
        messages.extend(_tool_turn(index))
    (session_dir / SESSION_FILE_NAME).write_text(
        json.dumps({"state": serialize_state({"messages": messages, "compressed_msgs": [], "turn_counter": turns})}),
        encoding="utf-8",
    )


async def _engine_with_session(agent_engine, tmp_path: Path, settings: Settings, *, turns: int = 2):
    bus = EventBus()
    engine = agent_engine(bus, settings=settings, state_store=JsonFileStateStore(tmp_path))
    engine._session_id = "sess01"
    engine._workspace = Workspace.from_cwd(str(tmp_path / "repo"))
    _seed_session(engine._session_dir_for("sess01"), turns)
    return engine, bus


async def test_flush_deposits_from_the_watermark_and_persists_the_new_mark(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_engine,
) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    seen: list[dict[str, Any]] = []

    def _deposit(session_file: Path, **kwargs: Any) -> WritebackOutcome:
        seen.append({"session_file": session_file, **kwargs})
        return WritebackOutcome(deposited=(1, 2), failed=None, watermark=2)

    monkeypatch.setattr(engine_module, "deposit_pending_turns", _deposit)
    engine, bus = await _engine_with_session(agent_engine, tmp_path, Settings())
    published: list[MemoryWritebackCompleted] = []
    await bus.subscribe(MemoryWritebackCompleted, published.append)

    await engine._flush_memory_writeback("idle")

    assert len(seen) == 1
    assert seen[0]["watermark"] == 0
    assert seen[0]["session_file"].name == SESSION_FILE_NAME
    assert seen[0]["source_prefix"] == "chrys-session:sess01"
    assert seen[0]["repo"] == "repo"
    assert engine._runtime_meta.memory_deposit_watermark == 2
    assert [(event.reason, event.deposited, event.watermark) for event in published] == [("idle", 2, 2)]


async def test_a_second_flush_resumes_from_the_persisted_watermark(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_engine,
) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    watermarks: list[int] = []

    def _deposit(session_file: Path, **kwargs: Any) -> WritebackOutcome:
        watermarks.append(kwargs["watermark"])
        return WritebackOutcome(deposited=(2,), failed=None, watermark=2)

    monkeypatch.setattr(engine_module, "deposit_pending_turns", _deposit)
    engine, _bus = await _engine_with_session(agent_engine, tmp_path, Settings())
    engine._runtime_meta.memory_deposit_watermark = 1

    await engine._flush_memory_writeback("idle")
    await engine._flush_memory_writeback("session_end")

    assert watermarks == [1, 2]


async def test_a_failed_turn_holds_the_watermark_and_is_reported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_engine,
) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    monkeypatch.setattr(
        engine_module,
        "deposit_pending_turns",
        lambda _session_file, **_kwargs: WritebackOutcome(deposited=(1,), failed=2, watermark=1),
    )
    engine, bus = await _engine_with_session(agent_engine, tmp_path, Settings())
    published: list[MemoryWritebackCompleted] = []
    await bus.subscribe(MemoryWritebackCompleted, published.append)

    await engine._flush_memory_writeback("idle")

    assert engine._runtime_meta.memory_deposit_watermark == 1
    assert published[-1].failed_turn == 2


async def test_the_watcher_is_armed_only_when_a_graph_is_configured(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_engine,
) -> None:
    monkeypatch.delenv("CONTEXTGRAPH_NEO4J_URI", raising=False)
    engine, _bus = await _engine_with_session(agent_engine, tmp_path, Settings())

    engine._start_memory_watcher()
    assert engine._memory_watcher is None

    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    engine._start_memory_watcher()
    assert engine._memory_watcher is not None

    await engine._memory_watcher.stop(flush=False, reason="test")


async def test_the_memory_setting_alone_disarms_the_watcher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_engine,
) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    engine, _bus = await _engine_with_session(agent_engine, tmp_path, Settings(memory_mcp_enabled=False))

    engine._start_memory_watcher()

    assert engine._memory_watcher is None


async def test_shutdown_flushes_once_and_honours_the_session_end_setting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_engine,
) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    reasons: list[str] = []

    async def _flush(reason: str) -> None:
        reasons.append(reason)

    engine, _bus = await _engine_with_session(agent_engine, tmp_path, Settings())
    engine._start_memory_watcher()
    assert engine._memory_watcher is not None
    engine._memory_watcher._on_flush = _flush
    engine._memory_watcher.touch()

    await engine.shutdown()

    assert reasons == ["session_end"]
    assert engine._memory_watcher is None


async def test_shutdown_can_be_told_not_to_flush(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_engine,
) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    reasons: list[str] = []

    async def _flush(reason: str) -> None:
        reasons.append(reason)

    engine, _bus = await _engine_with_session(agent_engine, tmp_path, Settings(memory_writeback_on_session_end=False))
    engine._start_memory_watcher()
    assert engine._memory_watcher is not None
    engine._memory_watcher._on_flush = _flush
    engine._memory_watcher.touch()

    await engine.shutdown()

    assert reasons == []


async def test_a_session_without_a_directory_flushes_nothing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_engine,
) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    calls: list[object] = []
    monkeypatch.setattr(engine_module, "deposit_pending_turns", lambda *a, **k: calls.append(a))
    bus = EventBus()
    engine = agent_engine(bus, settings=Settings(), state_store=JsonFileStateStore(tmp_path))

    await engine._flush_memory_writeback("idle")

    assert calls == []


async def test_finalizing_a_turn_restarts_the_idle_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_engine,
) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    engine, _bus = await _engine_with_session(agent_engine, tmp_path, Settings())
    engine._start_memory_watcher()
    watcher = engine._memory_watcher
    assert watcher is not None
    assert watcher._dirty is False

    async def _finalize() -> None:
        return None

    # Finalization itself needs a live run; stub just that call so the test
    # observes the touch the engine performs after it.
    monkeypatch.setattr(engine._turns, "finalize_current_run", _finalize)

    await engine._post_run()

    assert watcher._dirty is True
    assert watcher._last_activity is not None

    await watcher.stop(flush=False, reason="test")
