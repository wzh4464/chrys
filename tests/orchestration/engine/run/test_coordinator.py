# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for engine-owned turn coordination."""

from __future__ import annotations

import asyncio

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import Error, UserInterrupt, UserMessage, Warning
from chrys.orchestration.engine.engine import AgentEngine
from chrys.orchestration.engine.run.coordinator import TurnCoordinator


def _assert_display_message(event: Error | Warning, key: str) -> None:
    reference = event.display_message
    assert reference is not None
    assert reference.definition.key == key
    assert reference.args == ()


async def _collect[T](events: list[T], event: T) -> None:
    events.append(event)


@pytest.mark.asyncio
async def test_agent_engine_owns_turn_coordinator_state() -> None:
    engine = AgentEngine(EventBus(), settings=Settings())

    assert isinstance(engine._turns, TurnCoordinator)
    assert engine._turns._host is engine

    async def _run() -> None:
        return None

    task = asyncio.create_task(_run())
    engine._turn_state.run_task = task
    try:
        assert engine._turns.run_task is task
        assert engine.turn_lifecycle_task is task
        assert engine.is_turn_lifecycle_active is True
    finally:
        await task
    assert engine.is_turn_lifecycle_active is False


@pytest.mark.asyncio
async def test_final_save_evidence_routes_through_owned_coordinator_task() -> None:
    engine = AgentEngine(EventBus(), settings=Settings())
    owned_task = asyncio.current_task()
    assert owned_task is not None
    engine._turn_state.run_task = owned_task

    engine._turn_state.record_current_run_final_save()

    assert engine._turns.was_run_task_finally_saved(owned_task) is True
    assert engine.was_turn_lifecycle_saved(owned_task) is True


@pytest.mark.asyncio
async def test_interrupt_does_not_target_retry_task_replaced_during_sub_agent_cascade() -> None:
    """A Stop captured for task A must not interrupt replacement retry task B."""
    engine = AgentEngine(EventBus(), settings=Settings())
    release_a = asyncio.Event()
    release_b = asyncio.Event()
    task_b_started = asyncio.Event()

    async def run_a() -> None:
        await release_a.wait()

    async def run_b() -> None:
        task_b_started.set()
        await release_b.wait()

    task_a = asyncio.create_task(run_a())
    await asyncio.sleep(0)
    engine._turn_state.run_task = task_a

    class ReplacementExecutor:
        is_running = False

        def __init__(self) -> None:
            self.interrupt_calls = 0

        async def interrupt(self) -> None:
            self.interrupt_calls += 1

    executor = ReplacementExecutor()
    engine._executor = executor  # type: ignore[assignment]

    class ReplacingSubAgents:
        async def cascade_abort_all(self) -> None:
            release_a.set()
            await task_a
            # Mirrors terminal finalization dispatching a pending retry by
            # replacing the visible run task with B on the same executor.
            engine._turn_state.run_task = asyncio.create_task(run_b())
            executor.is_running = True
            await asyncio.sleep(0)

    engine._sub_agent_tools = ReplacingSubAgents()  # type: ignore[assignment]
    try:
        await engine._turns.on_user_interrupt(UserInterrupt())

        assert engine._turn_state.run_task is not task_a
        assert task_b_started.is_set()
        assert executor.interrupt_calls == 0
    finally:
        release_a.set()
        release_b.set()
        await task_a
        task_b = engine._turn_state.run_task
        if task_b is not None and task_b is not task_a:
            await task_b


@pytest.mark.asyncio
async def test_user_message_without_executor_publishes_semantic_not_ready_error() -> None:
    bus = EventBus()
    errors: list[Error] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = AgentEngine(bus, settings=Settings())
    engine._session_id = "session-1"

    await engine._on_user_message(UserMessage(text="hello"))

    assert len(errors) == 1
    assert (errors[0].code, errors[0].message, errors[0].session_id) == (
        "not_ready",
        "Engine not started",
        "session-1",
    )
    _assert_display_message(errors[0], "coordinator.engine_not_started")


@pytest.mark.asyncio
async def test_interrupt_during_load_publishes_semantic_warning() -> None:
    bus = EventBus()
    warnings: list[Warning] = []
    await bus.subscribe(Warning, lambda event: _collect(warnings, event))
    engine = AgentEngine(bus, settings=Settings())
    engine._session_id = "session-1"
    engine._agent_loading = True

    await engine._on_user_interrupt(UserInterrupt())

    assert len(warnings) == 1
    assert (warnings[0].code, warnings[0].message, warnings[0].session_id) == (
        "agent_loading_interrupt_ignored",
        "Interrupt ignored while agent infrastructure is loading.",
        "session-1",
    )
    _assert_display_message(warnings[0], "coordinator.interrupt_ignored_loading")


@pytest.mark.asyncio
async def test_prompt_admission_conflict_publishes_semantic_error() -> None:
    bus = EventBus()
    errors: list[Error] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = AgentEngine(bus, settings=Settings())
    engine._session_id = "session-1"

    await engine._turns._publish_prompt_admission_conflict()

    assert len(errors) == 1
    assert (errors[0].code, errors[0].message, errors[0].session_id) == (
        "prompt_admission_conflict",
        "Prompt could not be admitted because another turn started.",
        "session-1",
    )
    _assert_display_message(errors[0], "coordinator.prompt_admission_conflict")
