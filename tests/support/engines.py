# Copyright (c) 2026 Chrys. All rights reserved.

"""Lifecycle-safe AgentEngine factory fixture."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

import pytest

from chrys.orchestration.engine.engine import AgentEngine


class AgentEngineFactory(Protocol):
    """Construct an engine that the fixture will shut down after the test."""

    def __call__(self, *args: Any, **kwargs: Any) -> AgentEngine: ...


class EngineTracker:
    """Track engines and whether each reached a FULL terminal shutdown.

    ``AgentEngine`` exposes no shutdown-completed signal (``_shutting_down``
    is set on entry and ``EngineState`` has no terminal state), so completion
    is tracked by wrapping ``shutdown``. Only a successful shutdown with the
    default ``release_session_lock=True`` and ``close_mcp_cache=True`` counts:
    the session lifecycle legitimately performs PARTIAL shutdowns before
    resetting or restarting an engine, and those must stay pending for final
    cleanup. A later ``start()`` re-arms cleanup after a full shutdown.
    """

    def __init__(self) -> None:
        self._engines: list[Any] = []
        self._fully_shut_down: set[int] = set()

    def register(self, engine: Any) -> Any:
        original_shutdown = engine.shutdown
        original_start = engine.start

        async def _tracked_shutdown(*args: Any, **kwargs: Any) -> None:
            await original_shutdown(*args, **kwargs)
            if kwargs.get("release_session_lock", True) and kwargs.get("close_mcp_cache", True):
                self._fully_shut_down.add(id(engine))

        async def _tracked_start(*args: Any, **kwargs: Any) -> Any:
            self._fully_shut_down.discard(id(engine))
            return await original_start(*args, **kwargs)

        engine.shutdown = _tracked_shutdown
        engine.start = _tracked_start
        self._engines.append(engine)
        return engine

    async def finalize(self) -> None:
        """Fully shut down, in reverse order, every engine still pending.

        Every pending engine is attempted even when one shutdown raises or is
        cancelled — stopping early would leave the remaining engines' session
        locks, background tasks, and MCP caches alive in the xdist worker.
        CancelledError is a BaseException, so the handler must catch
        BaseException; a sole cancellation is re-raised as-is so it still
        propagates as cancellation, and mixed failures become a
        BaseExceptionGroup.
        """
        failures: list[BaseException] = []
        for engine in reversed(self._engines):
            if id(engine) in self._fully_shut_down:
                continue
            try:
                await engine.shutdown()
            except BaseException as exc:
                failures.append(exc)
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("engine teardown failures", failures)


@pytest.fixture
async def agent_engine() -> AsyncIterator[AgentEngineFactory]:
    """Return a factory and shut down every constructed engine in reverse order."""
    tracker = EngineTracker()

    def _factory(*args: Any, **kwargs: Any) -> AgentEngine:
        return tracker.register(AgentEngine(*args, **kwargs))

    yield _factory

    await tracker.finalize()
