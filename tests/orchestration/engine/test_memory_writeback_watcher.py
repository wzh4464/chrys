# Copyright (c) 2026 Chrys. All rights reserved.

"""The engine-owned idle timer that drives ContextGraph writeback."""

from __future__ import annotations

from chrys.orchestration.engine.memory_writeback import MemoryWritebackWatcher
from tests.support.waiting import wait_for, wait_until

_POLL = 0.01


class FakeClock:
    """A monotonic clock the test advances explicitly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_flushes_once_after_the_idle_window() -> None:
    clock = FakeClock()
    flushed: list[str] = []

    async def _flush(reason: str) -> None:
        flushed.append(reason)

    watcher = MemoryWritebackWatcher(
        idle_seconds=10, on_flush=_flush, is_busy=lambda: False, clock=clock, poll_seconds=_POLL
    )
    watcher.start()
    watcher.touch()
    clock.now = 11

    await wait_for(lambda: flushed == ["idle"], description="idle flush")
    clock.now = 30
    # Nothing new happened, so the watcher must not deposit again.
    assert not await wait_until(lambda: len(flushed) > 1, timeout=0.3)

    await watcher.stop(flush=False, reason="test")


async def test_a_busy_turn_defers_the_flush_until_stop() -> None:
    clock = FakeClock()
    flushed: list[str] = []
    busy = [True]

    async def _flush(reason: str) -> None:
        flushed.append(reason)

    watcher = MemoryWritebackWatcher(
        idle_seconds=10, on_flush=_flush, is_busy=lambda: busy[0], clock=clock, poll_seconds=_POLL
    )
    watcher.start()
    watcher.touch()
    clock.now = 11

    assert not await wait_until(lambda: bool(flushed), timeout=0.3)

    busy[0] = False
    await watcher.stop(flush=True, reason="session_end")

    assert flushed == ["session_end"]


async def test_zero_idle_seconds_disables_the_timer() -> None:
    flushed: list[str] = []

    async def _flush(reason: str) -> None:
        flushed.append(reason)

    watcher = MemoryWritebackWatcher(idle_seconds=0, on_flush=_flush, is_busy=lambda: False, poll_seconds=_POLL)
    watcher.start()
    watcher.touch()

    assert watcher.task is None
    assert not await wait_until(lambda: bool(flushed), timeout=0.3)

    # An explicit flush still works: only the timer is off.
    await watcher.stop(flush=True, reason="session_end")
    assert flushed == ["session_end"]


async def test_nothing_to_deposit_means_no_flush() -> None:
    clock = FakeClock()
    flushed: list[str] = []

    async def _flush(reason: str) -> None:
        flushed.append(reason)

    watcher = MemoryWritebackWatcher(
        idle_seconds=10, on_flush=_flush, is_busy=lambda: False, clock=clock, poll_seconds=_POLL
    )
    watcher.start()
    clock.now = 100

    # touch() was never called, so no turn has happened in this session.
    assert not await wait_until(lambda: bool(flushed), timeout=0.3)

    await watcher.stop(flush=True, reason="session_end")
    assert flushed == []


async def test_a_failing_flush_leaves_the_session_dirty_for_the_next_attempt() -> None:
    clock = FakeClock()
    attempts: list[str] = []

    async def _flush(reason: str) -> None:
        attempts.append(reason)
        raise RuntimeError("neo4j down")

    watcher = MemoryWritebackWatcher(
        idle_seconds=10, on_flush=_flush, is_busy=lambda: False, clock=clock, poll_seconds=_POLL
    )
    watcher.start()
    watcher.touch()
    clock.now = 11

    await wait_for(lambda: attempts == ["idle"], description="first attempt")

    await watcher.stop(flush=True, reason="session_end")

    # The failure must not have cleared the dirty flag, or the turns would
    # never be deposited at all.
    assert attempts == ["idle", "session_end"]


async def test_a_new_turn_after_a_flush_arms_the_timer_again() -> None:
    clock = FakeClock()
    flushed: list[str] = []

    async def _flush(reason: str) -> None:
        flushed.append(reason)

    watcher = MemoryWritebackWatcher(
        idle_seconds=10, on_flush=_flush, is_busy=lambda: False, clock=clock, poll_seconds=_POLL
    )
    watcher.start()
    watcher.touch()
    clock.now = 11
    await wait_for(lambda: flushed == ["idle"], description="first idle flush")

    watcher.touch()
    clock.now = 30
    await wait_for(lambda: flushed == ["idle", "idle"], description="second idle flush")

    await watcher.stop(flush=False, reason="test")


async def test_stop_is_safe_before_start_and_twice() -> None:
    flushed: list[str] = []

    async def _flush(reason: str) -> None:
        flushed.append(reason)

    watcher = MemoryWritebackWatcher(idle_seconds=10, on_flush=_flush, is_busy=lambda: False, poll_seconds=_POLL)

    await watcher.stop(flush=False, reason="test")
    watcher.start()
    watcher.touch()
    await watcher.stop(flush=True, reason="session_end")
    await watcher.stop(flush=True, reason="session_end")

    assert flushed == ["session_end"]


async def test_a_watcher_that_died_does_not_abort_shutdown() -> None:
    """Awaiting a failed task re-raises it; session end must survive that."""
    flushed: list[str] = []

    async def _flush(reason: str) -> None:
        flushed.append(reason)

    def _busy() -> bool:
        raise RuntimeError("host went away")

    watcher = MemoryWritebackWatcher(idle_seconds=1, on_flush=_flush, is_busy=_busy, poll_seconds=_POLL)
    watcher.start()
    watcher.touch()
    await wait_for(lambda: watcher.task is not None and watcher.task.done(), description="watcher task to die")

    await watcher.stop(flush=False, reason="session_end")

    assert watcher.task is None
