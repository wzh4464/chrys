# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for main-screen EventBus subscription ownership."""

from __future__ import annotations

from chrys.app.tui.screens.main.subscriptions import MainScreenSubscriptions
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    ContextPressure,
    RollbackResult,
    SessionReady,
    SettingsReloaded,
    SubAgentToolCallArgsUpdated,
    SubAgentToolCallProgress,
    SubAgentToolCallStatusUpdated,
    TodoListUpdated,
    ToolCallStatusUpdated,
    Warning,
)


class _Handlers:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        async def _handler(_event: object) -> None:
            self.calls.append(name)

        setattr(self, name, _handler)
        return _handler


async def _rollback_handler(_event: object) -> None:
    return


async def test_subscribe_all_registers_each_handler_once() -> None:
    bus = EventBus()
    events = _Handlers()
    sessions = _Handlers()
    subscriptions = MainScreenSubscriptions(
        bus=bus,
        events=events,  # type: ignore[arg-type]
        sessions=sessions,  # type: ignore[arg-type]
        rollback_result_handler=_rollback_handler,
    )

    await subscriptions.subscribe_all()
    await subscriptions.subscribe_all()

    assert len(subscriptions.subscriptions) == len({sub.event_type for sub in subscriptions.subscriptions})
    assert {sub.event_type for sub in subscriptions.subscriptions} >= {
        SessionReady,
        SettingsReloaded,
        Warning,
        ContextPressure,
        RollbackResult,
        TodoListUpdated,
        ToolCallStatusUpdated,
        SubAgentToolCallArgsUpdated,
        SubAgentToolCallStatusUpdated,
        SubAgentToolCallProgress,
    }
    for subscription in subscriptions.subscriptions:
        assert bus._handlers[subscription.event_type].count(subscription.handler) == 1


async def test_unsubscribe_all_is_idempotent_and_remount_does_not_duplicate_handlers() -> None:
    bus = EventBus()
    events = _Handlers()
    sessions = _Handlers()
    subscriptions = MainScreenSubscriptions(
        bus=bus,
        events=events,  # type: ignore[arg-type]
        sessions=sessions,  # type: ignore[arg-type]
        rollback_result_handler=_rollback_handler,
    )

    await subscriptions.subscribe_all()
    await subscriptions.unsubscribe_all()
    await subscriptions.unsubscribe_all()
    await subscriptions.subscribe_all()
    await subscriptions.subscribe_all()

    await bus.publish(Warning(message="heads up"))
    await bus.publish(ContextPressure(reason="round_limit"))
    await bus.publish(SessionReady(session_id="session-1", agent_profile="Code"))
    await bus.publish(TodoListUpdated(session_id="session-1"))

    assert events.calls.count("on_warning") == 1
    assert events.calls.count("on_context_pressure") == 1
    assert events.calls.count("on_session_ready") == 1
    assert events.calls.count("on_todo_list_updated") == 1

    await subscriptions.unsubscribe_all()
    for subscription in subscriptions.subscriptions:
        assert subscription.handler not in bus._handlers.get(subscription.event_type, [])
