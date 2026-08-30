# Copyright (c) 2026 Chrys. All rights reserved.

"""Engine integration points for successful-turn callbacks."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.kernel import Message
from chrys.orchestration.engine.engine import AgentEngine
from chrys.orchestration.engine.state.machine import EngineState, Trigger
from chrys.service.agent_middleware.injection import QueuedInjection


class _FakeExecutor:
    run_failed = False
    was_interrupted = False
    last_error = None

    def drain_batch_records(self) -> list[Any]:
        return []

    def drain_approval_decisions(self) -> list[Any]:
        return []


class _FakeInjection:
    def drain_pending(self) -> list[QueuedInjection]:
        return []


def _post_run_ready_engine(on_successful_turn: Callable[[], None] | None = None) -> AgentEngine:
    engine = AgentEngine(EventBus(), settings=Settings(), on_successful_turn=on_successful_turn)
    engine._executor = _FakeExecutor()  # type: ignore[assignment]
    engine._injection = _FakeInjection()  # type: ignore[assignment]
    engine._intermediate_texts = {}
    engine._consumed_injections = []
    engine._history.bind({"messages": [Message("user", ["hi"]), Message("assistant", ["ok"])]})
    engine._fsm.try_transition(Trigger.START)
    engine._fsm.try_transition(Trigger.USER_MESSAGE)
    return engine


@pytest.mark.asyncio
async def test_post_run_calls_successful_turn_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def _on_successful_turn() -> None:
        nonlocal calls
        calls += 1

    engine = _post_run_ready_engine(_on_successful_turn)
    monkeypatch.setattr(engine, "_save_current_session", _noop_save_current_session)

    await engine._post_run()

    assert calls == 1
    assert engine.state is EngineState.IDLE


@pytest.mark.asyncio
async def test_post_run_successful_turn_callback_failure_does_not_skip_session_save(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="chrys.orchestration.engine.run.finalizer")

    def _on_successful_turn() -> None:
        raise PermissionError("callback failed")

    engine = _post_run_ready_engine(_on_successful_turn)

    saved = False

    async def _save_current_session() -> None:
        nonlocal saved
        saved = True

    monkeypatch.setattr(engine, "_save_current_session", _save_current_session)

    await engine._post_run()

    assert saved
    assert engine.state is EngineState.IDLE
    assert "Failed to run successful turn callback" in caplog.text


async def _noop_save_current_session() -> None:
    return None
