# Copyright (c) 2026 Chrys. All rights reserved.

"""Lifecycle coverage for the shared agent_engine fixture's EngineTracker."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.support.ci import CI_LINUX_ONLY
from tests.support.engines import EngineTracker

# Platform-independent test-infra self-checks: the Linux CI job covers them.
pytestmark = CI_LINUX_ONLY


class _FakeEngine:
    def __init__(self) -> None:
        self.shutdown_calls: list[dict[str, bool]] = []
        self.started = 0
        self.fail_next_shutdown = False
        self.cancel_next_shutdown = False

    async def shutdown(self, *, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        if self.cancel_next_shutdown:
            self.cancel_next_shutdown = False
            raise asyncio.CancelledError("mid-shutdown cancellation")
        if self.fail_next_shutdown:
            self.fail_next_shutdown = False
            raise RuntimeError("mid-shutdown failure")
        self.shutdown_calls.append({"release_session_lock": release_session_lock, "close_mcp_cache": close_mcp_cache})

    async def start(self, profile: Any = None) -> None:
        self.started += 1


_FULL = {"release_session_lock": True, "close_mcp_cache": True}


async def test_full_shutdown_is_terminal_and_not_repeated() -> None:
    tracker = EngineTracker()
    engine = tracker.register(_FakeEngine())

    await engine.shutdown()
    await tracker.finalize()

    assert engine.shutdown_calls == [_FULL]


@pytest.mark.parametrize(
    "partial_kwargs",
    [
        {"close_mcp_cache": False},
        {"release_session_lock": False},
        {"release_session_lock": False, "close_mcp_cache": False},
    ],
)
async def test_partial_shutdown_stays_pending_for_final_cleanup(partial_kwargs: dict[str, bool]) -> None:
    tracker = EngineTracker()
    engine = tracker.register(_FakeEngine())

    await engine.shutdown(**partial_kwargs)
    await tracker.finalize()

    assert len(engine.shutdown_calls) == 2
    assert engine.shutdown_calls[-1] == _FULL


async def test_raised_shutdown_is_retried_at_teardown() -> None:
    tracker = EngineTracker()
    engine = tracker.register(_FakeEngine())
    engine.fail_next_shutdown = True

    with pytest.raises(RuntimeError, match="mid-shutdown failure"):
        await engine.shutdown()
    await tracker.finalize()

    assert engine.shutdown_calls == [_FULL]


async def test_restart_after_full_shutdown_rearms_cleanup() -> None:
    """The SessionNew/restore shape: shut down, then start the engine again."""
    tracker = EngineTracker()
    engine = tracker.register(_FakeEngine())

    await engine.shutdown()
    await engine.start()
    await tracker.finalize()

    assert engine.started == 1
    assert len(engine.shutdown_calls) == 2
    assert engine.shutdown_calls[-1] == _FULL


async def test_finalize_attempts_every_engine_despite_a_failure() -> None:
    tracker = EngineTracker()
    first = tracker.register(_FakeEngine())
    failing = tracker.register(_FakeEngine())
    last = tracker.register(_FakeEngine())
    failing.fail_next_shutdown = True

    with pytest.raises(RuntimeError, match="mid-shutdown failure"):
        await tracker.finalize()

    assert first.shutdown_calls == [_FULL], "engines after the failure must still be cleaned up"
    assert last.shutdown_calls == [_FULL]
    assert failing.shutdown_calls == []


async def test_finalize_aggregates_multiple_failures() -> None:
    tracker = EngineTracker()
    engines = [tracker.register(_FakeEngine()) for _ in range(3)]
    engines[0].fail_next_shutdown = True
    engines[2].fail_next_shutdown = True

    with pytest.raises(ExceptionGroup) as info:
        await tracker.finalize()

    assert len(info.value.exceptions) == 2
    assert engines[1].shutdown_calls == [_FULL]


async def test_finalize_continues_past_cancellation() -> None:
    """CancelledError is a BaseException — it must not truncate cleanup."""
    tracker = EngineTracker()
    first = tracker.register(_FakeEngine())
    cancelled = tracker.register(_FakeEngine())
    last = tracker.register(_FakeEngine())
    cancelled.cancel_next_shutdown = True

    with pytest.raises(asyncio.CancelledError):
        await tracker.finalize()

    assert first.shutdown_calls == [_FULL], "engines after the cancellation must still be cleaned up"
    assert last.shutdown_calls == [_FULL]
    assert cancelled.shutdown_calls == []


async def test_finalize_aggregates_cancellation_with_failures() -> None:
    tracker = EngineTracker()
    engines = [tracker.register(_FakeEngine()) for _ in range(3)]
    engines[0].cancel_next_shutdown = True
    engines[2].fail_next_shutdown = True

    with pytest.raises(BaseExceptionGroup) as info:
        await tracker.finalize()

    assert len(info.value.exceptions) == 2
    assert any(isinstance(exc, asyncio.CancelledError) for exc in info.value.exceptions)
    assert engines[1].shutdown_calls == [_FULL]


async def test_finalize_shuts_down_in_reverse_registration_order() -> None:
    tracker = EngineTracker()
    order: list[int] = []

    class _OrderedEngine(_FakeEngine):
        def __init__(self, tag: int) -> None:
            super().__init__()
            self._tag = tag

        async def shutdown(self, **kwargs: bool) -> None:
            order.append(self._tag)
            await super().shutdown(**kwargs)

    tracker.register(_OrderedEngine(1))
    tracker.register(_OrderedEngine(2))
    await tracker.finalize()

    assert order == [2, 1]
