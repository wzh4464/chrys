# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for sleep tool middleware."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import SleepSkip, ToolCallResult, UserInterrupt
from chrys.foundation.tool_kinds import KIND_SLEEP
from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY
from chrys.orchestration.engine.executor import Executor
from chrys.service.agent_middleware._metadata_keys import _SLEEP_INTERRUPTED_KEY, _SLEEP_SKIPPED_KEY
from chrys.service.agent_middleware.control.interrupt import InterruptMiddleware
from chrys.service.agent_middleware.control.sleep import SleepMiddleware
from chrys.service.agent_middleware.events.hook_dispatch import set_call_id
from chrys.service.agent_middleware.events.tool_events import ToolEventMiddleware


def _sleep_context(seconds: object, *, call_id: str = "sleep-call") -> SimpleNamespace:
    ctx = SimpleNamespace(
        function=SimpleNamespace(name="sleep", chrys_kind=KIND_SLEEP),
        arguments={"seconds": seconds, "reason": "test wait"},
        result=None,
        metadata={},
    )
    set_call_id(ctx, call_id)
    return ctx


async def test_sleep_middleware_completes_without_calling_tool() -> None:
    bus = EventBus()
    mw = SleepMiddleware(bus)
    ctx = _sleep_context(0)
    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(ctx, _next)

    assert not called
    assert ctx.result == "Slept for 0 seconds."


async def test_sleep_middleware_skip_returns_tool_result() -> None:
    bus = EventBus()
    mw = SleepMiddleware(bus)
    ctx = _sleep_context(30, call_id="c1")

    async def _next() -> None:
        raise AssertionError("sleep middleware should not call the underlying tool")

    task = asyncio.create_task(mw.process(ctx, _next))
    await asyncio.sleep(0)

    await bus.publish(SleepSkip(call_id="c1"))
    await asyncio.wait_for(task, timeout=1)

    assert ctx.result == "Sleep skipped by user after 0 seconds (requested 30 seconds)."
    assert ctx.metadata[_SLEEP_SKIPPED_KEY] is True


async def test_sleep_middleware_user_interrupt_returns_tool_result() -> None:
    bus = EventBus()
    mw = SleepMiddleware(bus)
    ctx = _sleep_context(30, call_id="c1")

    async def _next() -> None:
        raise AssertionError("sleep middleware should not call the underlying tool")

    task = asyncio.create_task(mw.process(ctx, _next))
    await asyncio.sleep(0)

    await bus.publish(UserInterrupt())
    await asyncio.wait_for(task, timeout=1)

    assert ctx.result == "Sleep interrupted after 0 seconds (requested 30 seconds)."
    assert ctx.metadata[_SLEEP_INTERRUPTED_KEY] is True


async def test_executor_interrupt_lets_active_sleep_publish_result_before_cancel() -> None:
    bus = EventBus()
    sleep_mw = SleepMiddleware(bus)
    tool_events = ToolEventMiddleware(bus)
    ctx = _sleep_context(30, call_id="c1")
    results: list[ToolCallResult] = []

    async def _capture(event: ToolCallResult) -> None:
        results.append(event)

    async def _next() -> None:
        async def _underlying() -> None:
            raise AssertionError("sleep middleware should not call the underlying tool")

        await sleep_mw.process(ctx, _underlying)

    await bus.subscribe(ToolCallResult, _capture)
    task = asyncio.create_task(tool_events.process(ctx, _next))
    await asyncio.sleep(0)
    assert sleep_mw.active_call_ids == ("c1",)

    executor = Executor.__new__(Executor)
    executor._interrupt = InterruptMiddleware()
    executor._current_agent_task = task
    executor._sleep = sleep_mw
    executor._bus = bus

    await Executor.interrupt(executor)
    await asyncio.wait_for(task, timeout=1)

    assert results[-1].result == "Sleep interrupted after 0 seconds (requested 30 seconds)."
    assert results[-1].metadata["sleep_interrupted"] is True


async def test_sleep_tool_result_metadata_marks_skip() -> None:
    bus = EventBus()
    tool_events = ToolEventMiddleware(bus)
    sleep_mw = SleepMiddleware(bus)
    ctx = _sleep_context(30, call_id="c1")
    results: list[ToolCallResult] = []

    async def _capture(event: ToolCallResult) -> None:
        results.append(event)

    async def _next() -> None:
        async def _underlying() -> None:
            raise AssertionError("sleep middleware should not call the underlying tool")

        await sleep_mw.process(ctx, _underlying)

    await bus.subscribe(ToolCallResult, _capture)
    task = asyncio.create_task(tool_events.process(ctx, _next))
    await asyncio.sleep(0)

    await bus.publish(SleepSkip(call_id="c1"))
    await asyncio.wait_for(task, timeout=1)

    assert results[-1].metadata["sleep_skipped"] is True
    assert ctx.metadata[TOOL_RESULT_METADATA_KEY] == {"sleep_skipped": True}


async def test_sleep_middleware_rejects_long_sleep() -> None:
    bus = EventBus()
    mw = SleepMiddleware(bus)
    ctx = _sleep_context(3601)

    async def _next() -> None:
        raise AssertionError("invalid sleep should not call the underlying tool")

    await mw.process(ctx, _next)

    assert ctx.result.startswith("Error: sleep duration cannot exceed 3600 seconds.")


async def test_sleep_middleware_cancel_propagates() -> None:
    bus = EventBus()
    mw = SleepMiddleware(bus)
    ctx = _sleep_context(30)

    async def _next() -> None:
        raise AssertionError("sleep middleware should not call the underlying tool")

    task = asyncio.create_task(mw.process(ctx, _next))
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert ctx.result is None


async def test_sleep_middleware_ignores_other_tools() -> None:
    bus = EventBus()
    mw = SleepMiddleware(bus)
    ctx = SimpleNamespace(
        function=SimpleNamespace(name="other", chrys_kind="search"),
        arguments={},
        result=None,
        metadata={},
    )
    called = False

    async def _next() -> None:
        nonlocal called
        called = True
        ctx.result = "ok"

    await mw.process(ctx, _next)

    assert called
    assert ctx.result == "ok"
