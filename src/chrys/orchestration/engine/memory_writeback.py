# Copyright (c) 2026 Chrys. All rights reserved.

"""Idle-triggered ContextGraph writeback owned by the engine.

Chrys has no idle hook event and the user deliberately asked for no static
hook, so the "deposit an hour after the session goes quiet" semantics live in
the process that knows when a turn ended. A cron-only design could not express
it: it sees file mtimes, not turn lifecycles.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class MemoryWritebackWatcher:
    """Flush pending turns once the session has been idle long enough.

    The clock is injectable so tests can cross an hour-long window without
    waiting, and ``poll_seconds`` only shortens the polling cadence for them.
    """

    def __init__(
        self,
        *,
        idle_seconds: float,
        on_flush: Callable[[str], Awaitable[None]],
        is_busy: Callable[[], bool],
        clock: Callable[[], float] = time.monotonic,
        poll_seconds: float = 5.0,
    ) -> None:
        self._idle = max(0.0, idle_seconds)
        self._on_flush = on_flush
        self._is_busy = is_busy
        self._clock = clock
        self._poll = poll_seconds
        self._last_activity: float | None = None
        self._dirty = False
        self._lock = asyncio.Lock()
        self.task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Arm the idle timer, unless it is disabled or already running."""
        if self._idle <= 0 or self.task is not None:
            return
        self.task = asyncio.get_running_loop().create_task(self._run(), name="memory-writeback-watcher")

    def touch(self) -> None:
        """Record that a turn just finished, restarting the idle window."""
        self._last_activity = self._clock()
        self._dirty = True

    async def flush(self, reason: str) -> None:
        """Deposit pending turns, unless nothing is pending or a turn is live.

        A failure leaves the session marked dirty so the next attempt — the next
        idle window, or session end — retries it. Depositing is idempotent, so a
        retry that overlaps a write that actually landed costs nothing.
        """
        async with self._lock:
            if not self._dirty or self._is_busy():
                return
            try:
                await self._on_flush(reason)
            except Exception:
                logger.warning("memory writeback (%s) failed", reason, exc_info=True)
                return
            self._dirty = False

    async def stop(self, *, flush: bool, reason: str) -> None:
        """Cancel the timer and optionally take one last pass. Safe to repeat."""
        if self.task is not None:
            task, self.task = self.task, None
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                # Shutdown must not be aborted by a background timer that
                # already died; awaiting a failed task re-raises its error.
                logger.warning("memory writeback watcher ended with an error", exc_info=True)
        if flush:
            await self.flush(reason)

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._poll)
            if not self._dirty or self._last_activity is None:
                continue
            if self._clock() - self._last_activity >= self._idle:
                await self.flush("idle")
