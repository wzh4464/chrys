# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests that retry/resume does not increment the turn number.

When a user interrupts an agent run and then resumes (or retries after an
error), the resumed work belongs to the same logical user turn.  The mutation
tracker should record mutations under the same turn_id, not a new one.

Regression: prior to the fix, pre-run setup always incremented
``_turn_number`` and called ``start_turn()``, so an interrupt+resume
sequence created an extra empty turn and the real mutations landed under
a wrong (incremented) turn_id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import chrys.orchestration.engine.build.builder as builder_module
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    AgentMessage,
    Error,
    SessionReady,
    SessionSaved,
    UsageUpdate,
    UserInterrupt,
    UserMessage,
    UserRetry,
)
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.profiles.agents.schema import AgentProfile, ApprovalConfig, CompactionConfig, ToolsConfig
from chrys.service.state.store import JsonFileStateStore
from tests.support.pipeline_helpers import make_mock_settings_and_registry
from tests.support.waiting import await_run_task_chain, wait_until

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROFILE = AgentProfile(
    name="Code",
    display_name="Code Agent",
    instructions="You are a coding assistant.",
    tools=ToolsConfig(builtins=[]),
    approval=ApprovalConfig(default="auto"),
    compaction=CompactionConfig(enabled=False),
)


def _make_registry():
    from chrys.service.profiles.agents.registry import AgentProfileRegistry

    registry = AgentProfileRegistry()
    registry.register(_PROFILE)
    return registry


async def _collect(events: list, event: object) -> None:
    events.append(event)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_does_not_increment_turn_number(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """Interrupt + resume should keep the same turn number.

    Flow:
    1. User sends message → turn 1 starts
    2. Agent streams slowly, user interrupts
    3. User retries → should still be turn 1 (not turn 2)
    4. Agent completes
    5. Verify: turn_number == 1, mutation tracker has no extra turns
    """
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, AgentMessage, Error, UsageUpdate, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    settings, model_registry = make_mock_settings_and_registry(stream=True)
    state_store = JsonFileStateStore(tmp_path)
    registry = _make_registry()
    clients: list[MockChatClient] = []

    def _mock_create_client(s=None, **kw):
        client = MockChatClient(
            responses=[
                # Run 1: slow stream, will be interrupted
                MockResponse(text="A" * 20, chunk_size=2, chunk_delay=0.02),
                # Run 2 (retry): fast response
                MockResponse(text="Done."),
            ]
        )
        clients.append(client)
        return client

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    engine = None
    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=registry,
        model_registry=model_registry,
        state_store=state_store,
    )
    await engine.start(_PROFILE)

    # Send message → turn 1. Pre-run setup increments ``_turn_number`` to
    # 1 when the run starts; poll for that rather than guessing a settle.
    await bus.publish(UserMessage(text="Hello"))
    assert await wait_until(lambda: engine._turn_number == 1), f"Expected turn 1, got {engine._turn_number}"
    assert await wait_until(lambda: clients[0].call_count >= 1)

    # Interrupt — wait for the interrupted run task to finish rolling back
    # rather than sleeping a fixed duration (flaky on slow Windows CI).
    await bus.publish(UserInterrupt())
    await await_run_task_chain(engine)

    # Turn number should still be 1 after interrupt
    assert engine._turn_number == 1
    picker_revision = engine.conversation_revision
    assert picker_revision == 1

    # Resume/retry → should NOT increment turn number
    await bus.publish(UserRetry())
    assert await wait_until(lambda: engine.conversation_revision == picker_revision + 1)
    await await_run_task_chain(engine)

    assert engine._turn_number == 1, f"Expected turn_number to stay at 1 after retry, got {engine._turn_number}"
    assert engine.conversation_revision == picker_revision + 1

    # If mutation tracker exists, verify no extra turns
    if engine._mutation_tracker is not None:
        all_turns = engine._mutation_tracker.get_all_turns()
        turn_ids = [t.turn_id for t in all_turns]
        assert 1 in turn_ids, f"Turn 1 should exist, got {turn_ids}"
        assert 2 not in turn_ids, f"Turn 2 should NOT exist after retry, got {turn_ids}"

    # Persisted-state invariant: ``turn_counter`` must equal the count
    # of turn markers in history, and marker ``_turn`` values must be
    # sequential starting at 1.  Regression for the bug where
    # ``remove_trailing_markers`` popped a turn marker during retry but
    # left ``turn_counter`` advanced, so the next ``insert_turn_marker``
    # skipped slot 1 and produced ``_turn=2`` / ``turn_counter=2`` for
    # what was actually the first (retried) turn — which then restored
    # as ``Turn 2 (Current)`` in the rollback picker.
    state = engine._history.state
    turn_markers = [
        m for m in state["messages"] if m.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.TURN
    ]
    marker_indices = [m.additional_properties.get("_turn") for m in turn_markers]
    assert state["turn_counter"] == 1, (
        f"turn_counter must be 1 after retry of turn 1, got {state['turn_counter']} (markers: {marker_indices})"
    )
    assert marker_indices == [1], f"Expected exactly one turn marker with _turn=1, got {marker_indices}"
    assert state["turn_counter"] == len(turn_markers), (
        f"turn_counter must equal len(turn_markers): counter={state['turn_counter']} markers={marker_indices}"
    )


@pytest.mark.asyncio
async def test_new_message_after_retry_increments_turn(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """After a retry completes, a new user message should start a new turn.

    Flow:
    1. User sends message → turn 1
    2. Interrupt + retry → still turn 1
    3. New message → turn 2
    """
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, AgentMessage, Error, UsageUpdate, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    settings, model_registry = make_mock_settings_and_registry(stream=True)
    state_store = JsonFileStateStore(tmp_path)
    registry = _make_registry()

    def _mock_create_client(s=None, **kw):
        return MockChatClient(
            responses=[
                # Run 1: slow, will be interrupted
                MockResponse(text="A" * 20, chunk_size=2, chunk_delay=0.02),
                # Run 2 (retry): completes
                MockResponse(text="Done with turn 1."),
                # Run 3 (new message): completes
                MockResponse(text="Done with turn 2."),
            ]
        )

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    engine = None
    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=registry,
        model_registry=model_registry,
        state_store=state_store,
    )
    await engine.start(_PROFILE)

    # Turn 1: send + interrupt + retry. Poll for the turn-start increment
    # and the interrupted run's completion rather than fixed settles
    # (flaky on slow Windows CI).
    await bus.publish(UserMessage(text="First"))
    assert await wait_until(lambda: engine._turn_number == 1)
    await bus.publish(UserInterrupt())
    await await_run_task_chain(engine)
    assert engine._turn_number == 1

    await bus.publish(UserRetry())
    await await_run_task_chain(engine)
    assert engine._turn_number == 1, "Retry should not increment turn"

    # Turn 2: new message
    await bus.publish(UserMessage(text="Second"))
    await await_run_task_chain(engine)
    assert engine._turn_number == 2, f"New message after retry should be turn 2, got {engine._turn_number}"

    # Persisted-state invariant: two turns completed → two turn markers
    # with sequential ``_turn`` indices ``[1, 2]`` and ``turn_counter=2``.
    # This is the exact scenario that reproduced the user-visible
    # "Turn 3 (Current) for a 2-turn session" bug: before the fix the
    # retry path advanced ``turn_counter`` without re-inserting the
    # matching marker, so a fresh turn 2 ended up labelled ``_turn=3``
    # on disk, which restored as ``Turn 3`` in the rollback picker.
    state = engine._history.state
    turn_markers = [
        m for m in state["messages"] if m.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.TURN
    ]
    marker_indices = [m.additional_properties.get("_turn") for m in turn_markers]
    assert marker_indices == [1, 2], f"Expected sequential markers [1, 2], got {marker_indices}"
    assert state["turn_counter"] == 2, f"turn_counter must be 2, got {state['turn_counter']}"
    assert state["turn_counter"] == len(turn_markers)
