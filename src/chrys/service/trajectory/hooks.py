# Copyright (c) 2026 Chrys. All rights reserved.

"""``hook.operation.started`` / ``hook.operation.finished`` recording for the hook manager.

A hook run is itself a wait node on the timeline (a blocking hook holds the
turn; an async one still occupies a subprocess slot), so it gets its own
lifecycle pair and never a ``wait.*`` twin.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from chrys.foundation.trajectory.context import TrajectoryContext, current_trajectory
from chrys.foundation.trajectory.envelope import MeasurementSource, measurement
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.ids import new_analytics_id

logger = logging.getLogger(__name__)

TrajectoryContextProvider = Callable[[], TrajectoryContext | None]
"""Resolves the recording scope for hooks fired outside a model run (session/turn hooks)."""


class HookOutcome:
    """``hook.operation.finished.outcome``."""

    SUCCESS = "success"
    FAILED = "failed"
    LAUNCH_ERROR = "launch_error"
    TIMED_OUT = "timed_out"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    ABANDONED = "abandoned"
    DETACHED = "detached"
    SPAWN_FAILED = "spawn_failed"


class HookOperationTrace:
    """One hook subprocess run under the ambient (or provided) trajectory scope."""

    __slots__ = ("_context", "_finished", "_operation_id", "_start_committed", "_started_ns", "_target_operation_id")

    def __init__(self, context: TrajectoryContext, *, target_operation_id: str | None) -> None:
        self._context = context
        self._operation_id = new_analytics_id()
        self._target_operation_id = target_operation_id
        self._started_ns = time.monotonic_ns()
        self._finished = False
        self._start_committed = False

    @classmethod
    def open(
        cls,
        *,
        target_operation_id: str | None = None,
        provider: TrajectoryContextProvider | None = None,
    ) -> HookOperationTrace | None:
        """Bind to the ambient scope, else to what *provider* resolves; ``None`` when unrecorded."""
        context = current_trajectory()
        if context is None and provider is not None:
            context = provider()
        if context is None:
            return None
        return cls(context, target_operation_id=target_operation_id)

    @property
    def operation_id(self) -> str:
        return self._operation_id

    @property
    def start_committed(self) -> bool:
        """Whether ``hook.operation.started`` acquired a log sequence."""
        return self._start_committed

    def _parent(self) -> str | None:
        return self._target_operation_id or self._context.innermost_model_operation_id

    async def started(
        self,
        *,
        hook_id: str,
        hook_event: str,
        execution_mode: str,
        detach: bool,
        delivery: str,
        scope: str = "turn",
        drain_scope: str | None = "turn",
    ) -> None:
        payload: dict[str, Any] = {
            # Keep schema-v1 readers compatible: unknown ``*_id`` fields are
            # analytics IDs to them, while this user-configured key is opaque.
            "hook_key": hook_id,
            "hook_event": hook_event,
            "execution_mode": execution_mode,
            "detach": detach,
            "delivery": delivery,
            "scope": scope,
        }
        if drain_scope is not None:
            payload["drain_scope"] = drain_scope
        if self._target_operation_id is not None:
            payload["target_operation_id"] = self._target_operation_id

        def _commit(_sequence: int) -> dict[str, Any]:
            # Runs where the writer takes the sequence and queues the line in
            # one locked step: reaching here is what makes this hook real to
            # the log, and nothing below may close a span the log never opened.
            self._start_committed = True
            return payload

        try:
            await self._context.sink.emit(
                self._context.draft(
                    EventType.HOOK_OPERATION_STARTED,
                    operation_id=self._operation_id,
                    parent_operation_id=self._parent(),
                    payload=payload,
                ),
                payload_factory=_commit,
            )
        except Exception:
            logger.debug("Trajectory hook.operation.started emit failed", exc_info=True)

    def _finished_draft(
        self,
        *,
        outcome: str,
        arguments_modified: bool,
        exit_code: int | None,
        timed_out: bool,
    ) -> Any:
        duration_ms = max(0, (time.monotonic_ns() - self._started_ns) // 1_000_000)
        payload: dict[str, Any] = {
            "outcome": outcome,
            "arguments_modified": arguments_modified,
            "duration_ms": duration_ms,
        }
        if exit_code is not None:
            payload["exit_code"] = exit_code
        if timed_out:
            payload["timed_out"] = True
        return self._context.draft(
            EventType.HOOK_OPERATION_FINISHED,
            operation_id=self._operation_id,
            parent_operation_id=self._parent(),
            payload=payload,
            measurements={"/payload/duration_ms": measurement(MeasurementSource.MONOTONIC_CLOCK, method_version=1)},
        )

    async def finished(
        self,
        *,
        outcome: str,
        arguments_modified: bool = False,
        exit_code: int | None = None,
        timed_out: bool = False,
    ) -> None:
        if self._finished:
            return
        self._finished = True
        if not self._start_committed:
            return
        draft = self._finished_draft(
            outcome=outcome, arguments_modified=arguments_modified, exit_code=exit_code, timed_out=timed_out
        )
        try:
            await self._context.sink.emit(draft)
        except Exception:
            logger.debug("Trajectory hook.operation.finished emit failed", exc_info=True)

    def finished_soon(self, *, outcome: str) -> None:
        """Close without awaiting the ack (cancellation paths)."""
        if self._finished:
            return
        self._finished = True
        if not self._start_committed:
            # An interrupt landing in the start marker's ack wait leaves here
            # before the writer took a sequence for it — the very first event
            # of a runtime waits on the activation that opens the log, which
            # is the longest that wait ever is. A terminal against a start
            # nothing recorded is not a shape readers handle; an unclosed
            # span is one they already do, so the hook is dropped instead.
            # This path cannot wait for the start to settle either: it is the
            # cancellation path, and blocking it is what it exists to avoid.
            return
        draft = self._finished_draft(outcome=outcome, arguments_modified=False, exit_code=None, timed_out=False)
        try:
            self._context.sink.emit_soon(draft)
        except Exception:
            logger.debug("Trajectory hook.operation.finished emit failed", exc_info=True)
