# Copyright (c) 2026 Chrys. All rights reserved.

"""Per-invocation sub-agent controller — drives one sub-agent with pause semantics.

Every call to a sub-agent tool creates exactly one :class:`SubAgentController`.
The controller owns:

- the sub-agent ``Agent`` and the kwargs used to invoke it
- a per-invocation ``AgentSession`` carrying ``chrys_history`` state
  across pause/retry within one invocation, so user-clicked Retry can
  resume after the last complete tool exchange instead of restarting
  from the original prompt
- a per-invocation ``LoopRecorder`` recording the completed
  tool-loop iterations of the most recent failed attempt so pause-time
  repair can merge them into the local history
- a :class:`~chrys.foundation.retry.StreamRetryLoop` for transient errors
- the lifecycle state (``SubAgentStatus``)
- a ``pending_decision`` future that gates :meth:`run` while the user decides
  Retry vs. Abort on a paused sub-agent

Because every controller is keyed by ``invocation_id`` in the parent
:class:`~chrys.orchestration.sub_agents.tools.SubAgentTools` registry, multiple sub-agents
can fail independently: a retry event for one invocation never touches
another.  Healthy siblings keep running regardless of what's paused.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict, TypeGuard, cast

from chrys.foundation.errors import clean_error_message, is_retryable
from chrys.foundation.events.types import (
    SubAgentAborted,
    SubAgentCascadeAborted,
    SubAgentPaused,
    SubAgentResumed,
    SubAgentRetryAttempt,
    SubAgentToolCallResult,
)
from chrys.foundation.hosted_tools import HostedToolStatus
from chrys.foundation.models.turns import is_continuation_message
from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.foundation.retry import (
    HistorySnapshot,
    RetryAttemptInfo,
    StreamRetryLoop,
    StreamStall,
    StreamStallExhausted,
    restore_message_properties,
    snapshot_message_properties,
)
from chrys.foundation.trajectory.context import TrajectoryContext
from chrys.foundation.trajectory.event_types import RetryMode, RetryReason
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory.metadata import ensure_analytics_item_id
from chrys.kernel import TOOL_RESULT_CONTENT_TYPES, LoopRecorder, Message, resolve_storage_mode_and_handles
from chrys.service.agent_middleware.events.hosted_tools import adapt_hosted_tool, hosted_replay_status
from chrys.service.agent_middleware.response_validation import (
    RetryableResponseValidationError,
    hosted_commits_from_error,
)
from chrys.service.context.compaction.last_words import LastWordsGenerationError
from chrys.service.context.providers.history import PRE_OUTPUT_HISTORY_LEN_STATE_KEY
from chrys.service.session.history import SessionHistoryManager
from chrys.service.session.sub_agent_logs import preview_text
from chrys.service.tools.result_metadata import record_tool_success, tool_error
from chrys.service.trajectory.retries import RetryBackoffTrace

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Mapping, Sequence

    from chrys.foundation.events.bus import EventBus
    from chrys.kernel import Agent, AgentResponse, AgentResponseUpdate, AgentSession, ResponseStream
    from chrys.kernel.middleware import ChatMiddleware, FunctionMiddleware
    from chrys.service.agent_middleware.control.sleep import SleepMiddleware
    from chrys.service.agent_middleware.events.sub_agent_events import SubAgentEventMiddleware
    from chrys.service.session.sub_agent_logs import SubAgentLogStats, SubAgentSessionLogWriter

logger = logging.getLogger(__name__)


class AgentRunKwargs(TypedDict, total=False):
    """Keyword arguments forwarded unchanged to a sub-agent ``Agent.run``."""

    session: AgentSession
    middleware: Sequence[ChatMiddleware | FunctionMiddleware]
    options: Mapping[str, Any]
    compaction_strategy: Any
    tokenizer: Any
    client_kwargs: Mapping[str, Any]


def _is_string_keyed_dict(value: object) -> TypeGuard[dict[str, Any]]:
    """Narrow controller-owned kwargs dictionaries with string keys."""
    return isinstance(value, dict)


class SubAgentStatus(Enum):
    """Lifecycle state of a single sub-agent invocation."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"
    CASCADE_ABORTED = "cascade_aborted"


class SubAgentFailureReason(StrEnum):
    """Why a sub-agent entered :attr:`SubAgentStatus.PAUSED`.

    Kept small and string-typed so it serialises cleanly for persistence
    (Tier 3) without needing a custom encoder.
    """

    STREAM_STALL = "stream_stall"
    LAST_WORDS = "last_words"
    FRAMEWORK_EXC = "framework_exc"
    ACP_TRANSPORT = "acp_transport"


# Retry/backoff knobs mirror Executor's streaming retry budget so the UX
# feels the same between main-agent and sub-agent failures. The sub-agent
# retry loop uses the same schedule and cap.
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_BACKOFF_SCHEDULE = (3, 7, 15, 30, 60)
_DEFAULT_STREAM_ATTEMPT_TIMEOUT = 300.0

# Decisions resolved into the ``pending_decision`` future. Kept as string
# literals (not an Enum) because futures already pass values by identity —
# the point here is a readable debug repr.
_DECISION_RETRY = "retry"
_DECISION_ABORT = "abort"
_DECISION_CASCADE = "cascade_abort"


def _is_last_words_error(exc: BaseException) -> bool:
    """Detect :class:`LastWordsGenerationError` directly or on the cause chain."""
    return any(isinstance(e, LastWordsGenerationError) for e in (exc, exc.__cause__) if e is not None)


def _validation_retry_exemption(exc: BaseException) -> RetryAttemptInfo | None:
    """Expose the middleware-owned stored validation budget to the retry loop.

    The exemption is bounded because response validation terminally gives up
    when its carried retry budget or identical-reason guard is reached.
    """
    if not isinstance(exc, RetryableResponseValidationError):
        return None
    exemption = exc.exemption
    return RetryAttemptInfo(
        reason=str(exc),
        attempt=exemption.attempt,
        max_attempts=exemption.max_attempts,
        delay_seconds=exemption.delay_seconds,
    )


def _service_retry_reason(exc: BaseException) -> str:
    """Closed trajectory reason for a sub-agent whole-run retry."""
    if isinstance(exc, StreamStall | StreamStallExhausted):
        return RetryReason.STREAM_STALL
    if isinstance(exc, RetryableResponseValidationError):
        return RetryReason.VALIDATION_REJECTED
    return RetryReason.TRANSIENT_ERROR


class SubAgentController:
    """Drive one sub-agent invocation through run → retry → pause → resolve.

    The controller is a single-use object: once :meth:`run` resolves to
    a final string (or raises :class:`asyncio.CancelledError`), the
    instance is terminal. The parent tool call invokes :meth:`run`
    exactly once.
    """

    def __init__(
        self,
        *,
        invocation_id: str,
        tool_name: str,
        agent_name: str,
        agent: Agent,
        session: AgentSession,
        loop_recorder: LoopRecorder,
        prompt: str,
        run_kwargs: AgentRunKwargs,
        event_bus: EventBus | None,
        session_id: str | None = None,
        parent_call_id: str = "",
        parent_event_call_id: str = "",
        sub_agent_log_file: str = "",
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_schedule: tuple[int, ...] = _DEFAULT_BACKOFF_SCHEDULE,
        persist_dir: Path | None = None,
        log_writer: SubAgentSessionLogWriter | None = None,
        log_stats: SubAgentLogStats | None = None,
        pending_record_finalizer: Callable[[Path | None], None] | None = None,
        tool_event_middleware: SubAgentEventMiddleware | None = None,
        stream: bool = False,
        stream_attempt_timeout: float | None = None,
        sleep_middleware: SleepMiddleware | None = None,
        parent_interrupted_result_commit: Callable[[], None] | None = None,
        pass_start_hooks: Sequence[Callable[[], None]] = (),
        hosted_commits_probe: Callable[[], tuple[str, ...]] | None = None,
        trajectory_context: TrajectoryContext | None = None,
        trajectory_boundary_operation_id: str | None = None,
    ) -> None:
        self._invocation_id = invocation_id
        self._tool_name = tool_name
        self._agent_name = agent_name
        self._agent = agent
        self._session = session
        self._loop_recorder = loop_recorder
        self._prompt = prompt
        self._run_kwargs: AgentRunKwargs = {**run_kwargs, "session": session}
        run_options = run_kwargs.get("options")
        try:
            stores_by_default = bool(agent.client.STORES_BY_DEFAULT)
        except AttributeError:
            stores_by_default = False
        try:
            force_stateless = bool(agent.client.FORCES_STATELESS)
        except AttributeError:
            force_stateless = False
        self._service_storage = resolve_storage_mode_and_handles(
            run_options if isinstance(run_options, dict) else None,
            stores_by_default=stores_by_default,
            force_stateless=force_stateless,
        ).service_side
        # The per-invocation compaction strategy rides run_kwargs (see
        # tools.py); the retry loop must roll its exclusion-anchor state
        # back together with history, so keep a direct handle.
        self._compaction_strategy = run_kwargs.get("compaction_strategy")
        self._bus = event_bus
        self._session_id = session_id
        # The framework ``call_id`` of the parent assistant function_call
        # that invoked this sub-agent.  Persisted so reload-recovery can
        # pair a paused record back to its dangling function_call by id
        # rather than by name+appearance-order, which mis-matches when
        # the same sub-agent tool is invoked concurrently.
        self._parent_call_id = parent_call_id
        self._parent_event_call_id = parent_event_call_id
        self._sub_agent_log_file = sub_agent_log_file
        self._max_retries = max_retries
        self._backoff = backoff_schedule
        self._persist_dir = persist_dir
        self._log_writer = log_writer
        self._log_stats = log_stats
        self._pending_record_finalizer = pending_record_finalizer
        self._tool_event_middleware = tool_event_middleware
        self._stream = stream
        self._stream_attempt_timeout = (
            stream_attempt_timeout if stream_attempt_timeout is not None else _DEFAULT_STREAM_ATTEMPT_TIMEOUT
        )
        self._sleep_middleware = sleep_middleware
        self._parent_interrupted_result_commit = parent_interrupted_result_commit
        # Fired at the start of every pass (initial run, or a user Retry
        # decision after a pause).  Components carrying state across a pass's
        # whole-run retry attempts — the validation middleware's retry budget —
        # register here so an aborted pass cannot leak state into the next
        # one; mirrors the main executor's run_cycle_start_hooks.
        self._pass_start_hooks = tuple(pass_start_hooks)
        # Validation-middleware probe for provider-hosted tool executions the
        # loop recorder cannot see; consulted by the whole-run retry gate.
        self._hosted_commits_probe = hosted_commits_probe
        self._trajectory_context = trajectory_context
        self._trajectory_boundary_operation_id = trajectory_boundary_operation_id
        self._service_retry_trace: RetryBackoffTrace | None = None

        self._status: SubAgentStatus = SubAgentStatus.IDLE
        self._pending_decision: asyncio.Future[str] | None = None
        self._last_error: str = ""
        self._failure_reason: SubAgentFailureReason | None = None
        self._retry_attempts_total: int = 0
        self._cascade_requested: bool = False
        # The seed prompt is one persisted context item: it keeps a single
        # analytics identity across prompt replays so the trajectory can
        # account for every model request that re-sent it.
        self._seed_item_id = new_analytics_id()
        self._active_run_input: list[Any] = self._seed_input()
        self._next_run_input: list[Any] = self._seed_input()
        self._pass_start_index = 0
        self._cancellation_finalized = False
        self._cancellation_finalize_lock = asyncio.Lock()
        # Live ``agent.run()`` task for the current attempt — set inside
        # :meth:`_attempt` and cleared when the attempt resolves.  A
        # non-None value means the sub-agent is actively executing.
        # :meth:`cascade_abort` cancels it directly so a user interrupt
        # stops the run immediately instead of waiting for the framework
        # tool loop to reach the next check.
        self._current_task: asyncio.Task[Any] | None = None

    # ── public state ─────────────────────────────────────────────────

    @property
    def invocation_id(self) -> str:
        return self._invocation_id

    @property
    def status(self) -> SubAgentStatus:
        return self._status

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def failure_reason(self) -> SubAgentFailureReason | None:
        return self._failure_reason

    @property
    def max_retries(self) -> int:
        return self._max_retries

    @property
    def backoff_schedule(self) -> tuple[int, ...]:
        return self._backoff

    @property
    def cascade_requested(self) -> bool:
        return self._cascade_requested

    @property
    def is_paused(self) -> bool:
        return self._status == SubAgentStatus.PAUSED

    # ── user-triggered decisions ─────────────────────────────────────

    def request_retry(self) -> bool:
        """User clicked Retry on this paused controller.

        Returns True if a decision was dispatched, False if the
        controller wasn't actually paused (e.g. already completed or
        cascade-aborted mid-click). Synchronous because it only touches
        the future — no awaits required.
        """
        return self._resolve_decision(_DECISION_RETRY)

    def request_abort(self) -> bool:
        """User clicked Abort on this paused controller."""
        return self._resolve_decision(_DECISION_ABORT)

    async def cascade_abort(self) -> None:
        """Global interrupt — tear down this invocation regardless of state.

        If paused, resolves the decision future so :meth:`run` exits its
        wait via the cascade branch. If running, cancels the live
        ``agent.run()`` task directly so the sub-agent stops immediately
        (previously we relied on parent task cancellation propagating
        down the stack; that works for the blocking main-agent path but
        not the streaming path, where no enclosing task wrapped the
        run).  The ``_cascade_requested`` flag is set BEFORE cancelling
        so :meth:`run`'s outer ``CancelledError`` handler can distinguish
        a user-driven cascade from an external cancellation (e.g. stream
        cleanup during retry) — only the former publishes
        :class:`SubAgentCascadeAborted`.
        """
        self._cascade_requested = True
        self._commit_parent_interrupted_result()
        # Resolve BEFORE any await: a stalled log write or bus subscriber
        # must never leave the paused run() blocked on its decision future.
        self._resolve_decision(_DECISION_CASCADE)
        # ``cascade_abort_all`` intentionally snapshots and gathers before the
        # coordinator cancels the parent executor. Publish a paused
        # controller's terminal event here so that subsequent task
        # cancellation cannot win the scheduling race and erase the signal;
        # the status guard makes the woken run loop's own publish a no-op.
        # The run-loop handler remains the once-only fallback for running
        # controllers.
        task = self._current_task
        if task is not None and not task.done():
            sleep_call_ids = self._sleep_middleware.active_call_ids if self._sleep_middleware is not None else ()
            if sleep_call_ids:
                await self._interrupt_active_sleep(set(sleep_call_ids))
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self.finalize_cancellation()

    async def finalize_cancellation(self) -> None:
        """Materialize committed work and durably close a cancelled invocation."""
        self._commit_parent_interrupted_result()
        async with self._cancellation_finalize_lock:
            if self._cancellation_finalized:
                return
            if self._tool_event_middleware is not None:
                await self._tool_event_middleware.reject_hosted_attempt("Sub-agent execution interrupted")
            self._repair_paused_history()
            if self._cascade_requested:
                await self._publish_cascade_aborted()
            else:
                self._status = SubAgentStatus.ABORTED
                await self._write_log(
                    status="cancelled",
                    result="cancelled",
                    last_error="cancelled",
                    ended=True,
                )
            self._cancellation_finalized = True

    def _commit_parent_interrupted_result(self) -> None:
        callback = self._parent_interrupted_result_commit
        if callback is None:
            return
        self._parent_interrupted_result_commit = None
        callback()

    async def _interrupt_active_sleep(self, call_ids: set[str]) -> None:
        """Let an active inner sleep publish SubAgentToolCallResult before cancellation."""
        if not call_ids or self._sleep_middleware is None or self._bus is None:
            return
        pending = set(call_ids)
        loop = asyncio.get_running_loop()
        completed: asyncio.Future[None] = loop.create_future()

        async def _on_tool_result(event: SubAgentToolCallResult) -> None:
            if event.invocation_id != self._invocation_id:
                return
            pending.discard(event.call_id)
            if not pending and not completed.done():
                completed.set_result(None)

        await self._bus.subscribe(SubAgentToolCallResult, _on_tool_result)
        try:
            interrupted = set(self._sleep_middleware.interrupt_active())
            pending.intersection_update(interrupted)
            if not pending:
                return
            # Keep global interrupt responsive: give the inner sleep a
            # small writeback window, then let task cancellation win.
            await asyncio.wait_for(completed, timeout=0.5)
        except TimeoutError:
            return
        finally:
            await self._bus.unsubscribe(SubAgentToolCallResult, _on_tool_result)

    def _resolve_decision(self, decision: str) -> bool:
        fut = self._pending_decision
        if fut is None or fut.done():
            return False
        fut.set_result(decision)
        return True

    # ── main entry ───────────────────────────────────────────────────

    async def run(self) -> str:
        """Execute the sub-agent with auto-retry + pause loop.

        Returns the sub-agent's final text on success, or an ``Error:``
        string when the user aborted after a pause. Propagates
        :class:`asyncio.CancelledError` on cascade abort so the parent
        executor's existing interrupt-handling runs unchanged.

        On terminal exit paths, the on-disk persisted pause record is queued
        for removal after the parent session is durably saved.  If the
        controller is cancelled while still paused, the record is preserved
        for session-reload recovery.
        """
        try:
            return await self._run_loop()
        except asyncio.CancelledError:
            await self.finalize_cancellation()
            raise
        finally:
            self._queue_persisted_cleanup_if_terminal()

    async def _run_loop(self) -> str:
        while True:
            if self._cascade_requested:
                raise asyncio.CancelledError

            self._pass_start_index = len(self._history_messages())
            self._loop_recorder.reset()
            try:
                # Inside the try so a raising hook lands on the normal
                # framework-exception pause path instead of escaping the loop
                # after ``SubAgentResumed`` was already published.
                for hook in self._pass_start_hooks:
                    hook()
                self._status = SubAgentStatus.RUNNING
                await self._write_log(status="running")
                self._active_run_input = list(self._next_run_input)
                response = (
                    await self._build_retry_loop().run(self._attempt)
                    if self._service_storage
                    else await self._attempt()
                )
                if self._tool_event_middleware is not None:
                    await self._tool_event_middleware.reconcile_hosted_response(response.messages)
                self._status = SubAgentStatus.COMPLETED
                text = self._extract_text(response)
                if not text:
                    error_text = tool_error(
                        "sub_agent_empty_output",
                        f"sub-agent '{self._tool_name}' returned no output",
                        details={"tool_name": self._tool_name, "invocation_id": self._invocation_id},
                    )
                    await self._write_log(
                        status="completed",
                        result=error_text,
                        failure_reason="empty_output",
                        last_error=error_text,
                        ended=True,
                    )
                    return error_text
                record_tool_success()
                await self._write_log(status="completed", result=text, ended=True)
                return text

            except StreamStallExhausted as exc:
                # Keep the original stall message (chained via __cause__)
                # so the pause banner shows the underlying reason instead
                # of a generic "retries exhausted" placeholder. Falls back
                # to a static label when the chain is empty (e.g. stalls
                # without a stored cause — defensive only).
                cause_msg = clean_error_message(exc) if exc.__cause__ else ""
                self._last_error = (
                    f"Stream stalled after {self._max_retries} retries: {cause_msg}"
                    if cause_msg
                    else f"Stream stalled after {self._max_retries} retries"
                )
                self._failure_reason = SubAgentFailureReason.STREAM_STALL
                logger.warning(
                    "Sub-agent '%s' (inv=%s) stalled after %d retries — pausing",
                    self._tool_name,
                    self._invocation_id,
                    self._max_retries,
                )

            except StreamStall as exc:
                self._last_error = clean_error_message(exc) or "Stream stalled"
                self._failure_reason = SubAgentFailureReason.STREAM_STALL
                logger.warning(
                    "Sub-agent '%s' (inv=%s) stalled — pausing: %s",
                    self._tool_name,
                    self._invocation_id,
                    self._last_error,
                )

            except asyncio.CancelledError:
                # Task cancel arrived mid-attempt (e.g. cascade from
                # parent interrupt). Let outer handler publish the event.
                raise

            except Exception as e:
                if _is_last_words_error(e):
                    self._failure_reason = SubAgentFailureReason.LAST_WORDS
                else:
                    self._failure_reason = SubAgentFailureReason.FRAMEWORK_EXC
                self._last_error = clean_error_message(e)
                logger.warning(
                    "Sub-agent '%s' (inv=%s) failed with %s — pausing: %s",
                    self._tool_name,
                    self._invocation_id,
                    self._failure_reason.value,
                    self._last_error,
                )

            # Transition to PAUSED and wait for user / cascade.
            if self._tool_event_middleware is not None:
                await self._tool_event_middleware.reject_hosted_attempt(
                    self._last_error or "Sub-agent execution paused"
                )
            self._repair_paused_history()
            decision = await self._pause_and_wait()

            if decision == _DECISION_RETRY:
                self._prepare_retry_input()
                await self._publish_resumed()
                # Loop restarts — status moves back to RUNNING on next iter.
                continue
            if decision == _DECISION_ABORT:
                self._status = SubAgentStatus.ABORTED
                abort_text = tool_error(
                    "sub_agent_aborted",
                    f"sub-agent '{self._tool_name}' aborted by user after failure — {self._last_error}",
                    details={
                        "tool_name": self._tool_name,
                        "invocation_id": self._invocation_id,
                        "failure_reason": self._failure_reason.value if self._failure_reason else "",
                    },
                )
                await self._write_log(
                    status="aborted",
                    result=abort_text,
                    failure_reason=(self._failure_reason.value if self._failure_reason else ""),
                    last_error=self._last_error,
                    ended=True,
                )
                await self._publish_aborted()
                return abort_text
            # Cascade abort
            raise asyncio.CancelledError

    async def _attempt(self) -> AgentResponse[Any]:
        """Single agent.run() attempt — the unit of auto-retry.

        Sub-agents use the resolved ``ModelProfile.stream`` setting. In
        blocking mode, retryable exceptions (e.g.
        ``LastWordsGenerationError``, transient SDK errors) drive retries
        via the loop's catch path. In streaming mode, idle streams raise
        :class:`StreamStall` so the same loop can retry/pause.

        The run is wrapped in a child task stored on ``_current_task`` so
        :meth:`cascade_abort` can cancel it directly even when the
        parent main-agent path has no enclosing task to cancel (the
        streaming executor path).
        """
        if self._cascade_requested:
            raise asyncio.CancelledError
        if self._stream:
            return await self._stream_attempt()
        run = self._agent.run(self._active_run_input, stream=False, **self._run_kwargs)
        # Chrys's non-streaming implementation returns a coroutine; the public
        # overload intentionally promises only the broader Awaitable contract.
        self._current_task = asyncio.create_task(cast("Coroutine[Any, Any, AgentResponse[Any]]", run))
        try:
            return await self._current_task
        finally:
            self._current_task = None

    async def _stream_attempt(self) -> AgentResponse[Any]:
        """Run one streamed sub-agent attempt and return its final response."""
        last_wait_start = _time.monotonic()

        async def _iterate_and_finalize() -> AgentResponse[Any]:
            nonlocal last_wait_start

            stream: ResponseStream[AgentResponseUpdate, AgentResponse[Any]] = self._agent.run(
                self._active_run_input, stream=True, **self._run_kwargs
            )

            try:
                expecting_tool_result = False
                aiter = stream.__aiter__()
                while True:
                    try:
                        if not self._service_storage or expecting_tool_result:
                            update = await aiter.__anext__()
                        else:
                            update = await asyncio.wait_for(
                                aiter.__anext__(),
                                timeout=self._stream_attempt_timeout,
                            )
                    except StopAsyncIteration:
                        break
                    last_wait_start = _time.monotonic()

                    has_text = False
                    has_function_call = False
                    has_function_result = False
                    for content in update.contents or []:
                        if content.type == "text" and content.text:
                            has_text = True
                        elif content.type == "function_call" and not content.informational_only:
                            has_function_call = True
                        elif content.type == "function_result":
                            has_function_result = True

                    if has_function_result:
                        expecting_tool_result = False
                    if has_text:
                        expecting_tool_result = False
                    if has_function_call:
                        # The framework invokes tools inside the next
                        # ``__anext__`` call. Long-running sub-agent tools
                        # must not trip the stream-idle watchdog.
                        expecting_tool_result = True

                    await asyncio.sleep(0)

                last_wait_start = _time.monotonic()
                if not self._service_storage:
                    return await stream.get_final_response()
                return await asyncio.wait_for(stream.get_final_response(), timeout=self._stream_attempt_timeout)
            except TimeoutError:
                if not self._service_storage:
                    raise
                idle = _time.monotonic() - last_wait_start
                if idle + 0.5 < self._stream_attempt_timeout:
                    raise
                try:
                    await stream.aclose()
                except Exception:
                    logger.debug("Failed to close stalled sub-agent ResponseStream", exc_info=True)
                raise StreamStall(f"no streaming updates received for {self._stream_attempt_timeout:g}s") from None

        self._current_task = asyncio.create_task(_iterate_and_finalize())
        try:
            return await self._current_task
        finally:
            self._current_task = None

    async def _pause_and_wait(self) -> str:
        self._status = SubAgentStatus.PAUSED
        # Persist BEFORE publishing the event so that if the app is killed
        # at exactly this instant, the TUI-visible state (driven by the
        # event) matches what will be on disk at next launch.
        self._write_persisted()
        await self._write_log(
            status="paused",
            failure_reason=(self._failure_reason.value if self._failure_reason else ""),
            last_error=self._last_error,
        )
        # ``get_running_loop`` (not ``get_event_loop``) — the latter is
        # deprecated for in-coroutine use and in Python 3.14 emits a
        # warning when no current loop is set.  We're always inside a
        # running loop here (the tool invocation), so the strict variant
        # is both correct and fail-loud.
        loop = asyncio.get_running_loop()
        self._pending_decision = loop.create_future()
        await self._publish_paused()
        try:
            return await self._pending_decision
        finally:
            self._pending_decision = None

    # ── event publishing helpers ─────────────────────────────────────

    async def _publish_paused(self) -> None:
        if self._bus is None:
            return
        await self._bus.publish(
            SubAgentPaused(
                agent_name=self._agent_name,
                invocation_id=self._invocation_id,
                tool_name=self._tool_name,
                reason=(self._failure_reason.value if self._failure_reason else ""),
                last_error=self._last_error,
                retry_attempts=self._retry_attempts_total,
                session_id=self._session_id,
            )
        )

    async def _publish_resumed(self) -> None:
        if self._bus is None:
            return
        await self._bus.publish(
            SubAgentResumed(
                agent_name=self._agent_name,
                invocation_id=self._invocation_id,
                session_id=self._session_id,
            )
        )

    async def _publish_aborted(self) -> None:
        if self._bus is None:
            return
        await self._bus.publish(
            SubAgentAborted(
                agent_name=self._agent_name,
                invocation_id=self._invocation_id,
                last_error=self._last_error,
                session_id=self._session_id,
            )
        )

    async def _publish_cascade_aborted(self) -> None:
        if self._status == SubAgentStatus.CASCADE_ABORTED:
            return  # already published
        self._status = SubAgentStatus.CASCADE_ABORTED
        await self._write_log(
            status="cascade_aborted",
            result="cancelled by parent interrupt",
            failure_reason="cascade_aborted",
            last_error="cancelled by parent interrupt",
            ended=True,
        )
        if self._bus is None:
            return
        await self._bus.publish(
            SubAgentCascadeAborted(
                agent_name=self._agent_name,
                invocation_id=self._invocation_id,
                session_id=self._session_id,
            )
        )

    async def _publish_retry_attempt(
        self,
        message: str,
        attempt: int,
        max_attempts: int,
        delay_seconds: int,
        _exc: BaseException,
    ) -> None:
        self._retry_attempts_total += 1
        if self._tool_event_middleware is not None and not self._has_live_continuation_token():
            await self._tool_event_middleware.reject_hosted_attempt(message)
        if self._bus is None:
            return
        await self._bus.publish(
            SubAgentRetryAttempt(
                agent_name=self._agent_name,
                invocation_id=self._invocation_id,
                message=message,
                attempt=attempt,
                max_attempts=max_attempts,
                delay_seconds=delay_seconds,
                session_id=self._session_id,
            )
        )

    async def _publish_service_retry_attempt(
        self,
        message: str,
        attempt: int,
        max_attempts: int,
        delay_seconds: int,
        exc: BaseException,
    ) -> None:
        await self._publish_retry_attempt(message, attempt, max_attempts, delay_seconds, exc)
        trace = RetryBackoffTrace.open(
            context=self._trajectory_context,
            parent_operation_id=self._trajectory_boundary_operation_id,
            retry_mode=RetryMode.RUN,
        )
        self._service_retry_trace = trace
        if trace is not None:
            await trace.scheduled(reason_code=_service_retry_reason(exc), delay_seconds=delay_seconds)

    async def _sleep_for_service_retry(self, seconds: int) -> bool:
        trace, self._service_retry_trace = self._service_retry_trace, None
        try:
            interrupted = await self._interruptible_sleep(seconds)
        except asyncio.CancelledError:
            raise
        if trace is not None and not interrupted:
            await trace.started()
        return interrupted

    async def publish_wire_retry_attempt(
        self,
        message: str,
        attempt: int,
        max_attempts: int,
        delay_seconds: int,
        exc: BaseException,
    ) -> None:
        """Publish a kernel wire-retry notice through the controller bus."""
        await self._publish_retry_attempt(message, attempt, max_attempts, delay_seconds, exc)

    async def sleep_for_wire_retry(self, seconds: int) -> bool:
        """Use the controller's cascade-aware backoff for kernel retries."""
        return await self._interruptible_sleep(seconds)

    def observe_continuation_token(self, token: Any) -> None:
        """Mirror the kernel's live continuation token into retry-owned options.

        A transient failure while polling a stored background response must
        resume that response on the next attempt, not re-issue the original
        create request. Reads ``_run_kwargs`` at call time so a
        handle-stripping restore that replaces the options dict cannot
        orphan the observer.
        """
        raw_options = self._run_kwargs.get("options")
        options = raw_options if _is_string_keyed_dict(raw_options) else None
        if token is None:
            if options is not None:
                options.pop("continuation_token", None)
            return
        if options is None:
            options = {}
            self._run_kwargs["options"] = options
        options["continuation_token"] = token

    # ── StreamRetryLoop wiring ───────────────────────────────────────

    def _build_retry_loop(self) -> StreamRetryLoop[AgentResponse[Any]]:
        # No subscript on the runtime call: ``AgentResponse`` is a
        # TYPE_CHECKING-only import. The annotation on the signature is
        # evaluated lazily thanks to ``from __future__ import annotations``.
        return StreamRetryLoop(
            max_retries=self._max_retries,
            backoff_schedule=self._backoff,
            is_retryable=is_retryable,
            snapshot_history=self._snapshot_history,
            restore_history=self._restore_history,
            publish_retry_attempt=self._publish_service_retry_attempt,
            is_interrupted=lambda: self._cascade_requested,
            interruptible_sleep=self._sleep_for_service_retry,
            clean_error_message=clean_error_message,
            after_restore=self._restore_service_retry_inputs,
            may_retry=self._may_retry_attempt,
            retry_exemption=_validation_retry_exemption,
        )

    def _may_retry_attempt(self, exc: BaseException) -> bool:
        """Whole-run retry gate: answered tool work must never re-execute.

        Mirrors the main executor's gate: locally answered results are
        counted by the loop recorder, while provider-hosted calls execute
        inside the failed exchange itself and only surface on the raised
        validation error or the middleware's observation probe.  A live
        continuation token exempts the gate — the retry then resumes the
        already-created background response instead of re-creating it.
        """
        if self._loop_recorder.committed_count:
            return False
        hosted = hosted_commits_from_error(exc)
        if not hosted and self._hosted_commits_probe is not None:
            hosted = self._hosted_commits_probe()
        if hosted and not self._has_live_continuation_token():
            logger.warning(
                "Sub-agent not retrying failed attempt: provider-hosted tool call(s) already executed (%s)",
                ", ".join(hosted),
            )
            return False
        return True

    def _has_live_continuation_token(self) -> bool:
        options = self._run_kwargs.get("options")
        return isinstance(options, dict) and options.get("continuation_token") is not None

    def _restore_service_retry_inputs(self) -> None:
        """Discard handles from a failed stored attempt after rollback."""
        self._session.service_session_id = None
        raw_options = self._run_kwargs.get("options")
        options = raw_options if _is_string_keyed_dict(raw_options) else {}
        raw_client_kwargs = self._run_kwargs.get("client_kwargs")
        client_kwargs = raw_client_kwargs if _is_string_keyed_dict(raw_client_kwargs) else {}
        try:
            force_stateless = bool(self._agent.client.FORCES_STATELESS)
        except AttributeError:
            force_stateless = False
        resolution = resolve_storage_mode_and_handles(
            options,
            stores_by_default=True,
            client_kwargs=client_kwargs,
            force_stateless=force_stateless,
        )
        if raw_options is not None:
            self._run_kwargs["options"] = resolution.options_without_handles
        self._run_kwargs["client_kwargs"] = resolution.client_kwargs_without_handles

    def _history_state(self) -> dict[str, Any]:
        """Return the per-invocation ``chrys_history`` state."""
        state = self._session.state.setdefault("chrys_history", {})
        if not isinstance(state, dict):
            state = {}
            self._session.state["chrys_history"] = state
        return state

    def _history_messages(self) -> list[Any]:
        """Return the mutable per-invocation history message list."""
        state = self._history_state()
        messages = state.setdefault("messages", [])
        if not isinstance(messages, list):
            messages = []
            state["messages"] = messages
        return messages

    def _snapshot_history(self) -> HistorySnapshot:
        """Snapshot sub-agent local history for transient retry rollback."""
        history_state = self._history_state()
        messages = list(history_state.get("messages", []))
        return HistorySnapshot(
            messages=messages,
            compressed_count=len(history_state.get("compressed_msgs", [])),
            service_session_id=self._session.service_session_id or "",
            message_properties=snapshot_message_properties(messages),
            pre_output_history_len=history_state.get(PRE_OUTPUT_HISTORY_LEN_STATE_KEY),
            caller_state=(
                self._compaction_strategy.snapshot_retry_state() if self._compaction_strategy is not None else None
            ),
        )

    def _restore_history(self, snapshot: HistorySnapshot) -> None:
        """Restore sub-agent local history from a retry snapshot.

        Mirrors the main executor's exact-restore: ``additional_properties``
        go back to their snapshot values, clearing the rolled-back attempt's
        compaction marks while keeping pre-existing ones.
        """
        history_state = self._history_state()
        restored_messages = list(snapshot.messages)
        history_state["messages"] = restored_messages
        self._session.service_session_id = snapshot.service_session_id or None

        compressed: list = history_state.get("compressed_msgs", [])
        if len(compressed) > snapshot.compressed_count:
            del compressed[snapshot.compressed_count :]

        if snapshot.pre_output_history_len is None:
            history_state.pop(PRE_OUTPUT_HISTORY_LEN_STATE_KEY, None)
        else:
            history_state[PRE_OUTPUT_HISTORY_LEN_STATE_KEY] = snapshot.pre_output_history_len
        restore_message_properties(restored_messages, snapshot.message_properties)
        if self._compaction_strategy is not None and snapshot.caller_state is not None:
            self._compaction_strategy.restore_retry_state(snapshot.caller_state)

    def _history_manager(self) -> SessionHistoryManager:
        """Return a history manager bound to this invocation's history state."""
        manager = SessionHistoryManager()
        manager.bind(self._history_state())
        return manager

    async def _write_log(
        self,
        *,
        status: str,
        result: str = "",
        failure_reason: str = "",
        last_error: str = "",
        ended: bool = False,
    ) -> None:
        """Best-effort audit-log update for this invocation.

        Terminal records (``ended=True``) are the only durable copy of the
        inner history once the controller is removed, so a failed write gets
        one retry and a loud warning instead of the silent debug log.  A
        terminal write is also attempted on a writer whose initial write
        failed (``active`` still False) — success there re-creates the log
        file rather than silently discarding the committed inner history.
        """
        if self._log_writer is None or self._log_writer.path is None:
            return
        if not ended and not self._log_writer.active:
            return
        for _attempt in range(2):
            try:
                if await self._log_writer.write(
                    status=status,
                    state=self._history_state(),
                    result=result,
                    failure_reason=failure_reason,
                    last_error=last_error,
                    ended=ended,
                ):
                    return
            except Exception:
                logger.debug("sub-agent audit log update failed for invocation %s", self._invocation_id, exc_info=True)
            if not ended:
                return
        logger.warning(
            "sub-agent audit log terminal write failed for invocation %s; inner history may be incomplete on reload",
            self._invocation_id,
        )

    def _repair_paused_history(self) -> None:
        """Persist recoverable completed tool-loop work before pausing.

        Also drops ``service_session_id`` so the next ``agent.run`` after a
        user Retry treats the locally repaired ``chrys_history`` as the
        source of truth.  Without this, OpenAI Responses profiles with
        ``store: true`` would see ``CompressibleHistoryProvider.before_run``
        skip the local history (because a service id is still set from the
        failed attempt's partial provider state), leaving the resume on a
        broken server-side conversation.
        Mirrors the main-agent rule at ``run/lifecycle.py`` post_run.
        """
        messages = self._history_messages()
        if self._loop_recorder.loop_messages:
            input_message = self._active_input_user_message()
            if input_message is not None and not messages:
                messages.append(input_message)
        manager = self._history_manager()
        manager.merge_loop_messages(self._loop_recorder, insert_index=self._pass_start_index)
        manager.trim_to_last_complete_tool_results()
        self._session.service_session_id = None

    def _active_input_user_message(self) -> Message | None:
        """Return a clean user message for the current attempt input."""
        if len(self._active_run_input) != 1:
            return None
        item = self._active_run_input[0]
        if isinstance(item, Message):
            return item
        if isinstance(item, str):
            return Message("user", [item])
        return None

    def _prepare_retry_input(self) -> None:
        """Choose empty continuation or prompt replay for a user-triggered retry.

        The anchor scan skips synthetic ``continue`` nudges (a crash-leftover
        flagged nudge with work before it and none after would otherwise
        anchor the decision, flip ``has_work_after`` to False, and restart
        the sub-agent from ``self._prompt`` — redoing completed work).
        """
        messages = self._history_messages()
        user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user" and not is_continuation_message(messages[i]):
                user_idx = i
                break

        if user_idx < 0:
            self._next_run_input = self._seed_input()
            return

        has_work_after = any(messages[j].role in ("assistant", "tool") for j in range(user_idx + 1, len(messages)))
        if has_work_after:
            self._next_run_input = []
            return

        messages.pop(user_idx)
        self._next_run_input = self._seed_input()

    def _seed_input(self) -> list[Any]:
        """Build the run input that starts (or restarts) the sub-agent from its prompt.

        Handing the framework a ready ``Message`` instead of the bare string
        is what lets the analytics item id ride along: the framework would
        otherwise mint an anonymous user message that no context revision
        can identify.
        """
        message = Message("user", [self._prompt])
        ensure_analytics_item_id(message.additional_properties, item_id=self._seed_item_id)
        return [message]

    async def _interruptible_sleep(self, seconds: int) -> bool:
        """Sleep in 1-second ticks, returning True if cascade fired."""
        for _ in range(max(0, seconds)):
            if self._cascade_requested:
                return True
            await asyncio.sleep(1)
        return self._cascade_requested

    # ── persistence ──────────────────────────────────────────────────

    def _persist_path(self) -> Path | None:
        if self._persist_dir is None:
            return None
        return self._persist_dir / f"{self._invocation_id}.json"

    def _serialize_state(self) -> dict[str, Any]:
        """Snapshot the paused state as a plain JSON-safe dict.

        We deliberately persist only pause metadata — not the ``Agent``,
        middleware, run_kwargs, or the prompt-level token counts.  On
        restore we reconstruct the controller as read-only-abort (see
        :meth:`restore_from_data`); there's no safe way to resume the
        original tool call after the parent process died.
        """
        now = datetime.now(UTC).isoformat()
        return {
            "schema_version": 1,
            "record_type": "sub_agent_pending",
            "invocation_id": self._invocation_id,
            "tool_name": self._tool_name,
            "agent_name": self._agent_name,
            "prompt_preview": preview_text(self._prompt),
            "session_id": self._session_id or "",
            # ``parent_call_id`` lets the reload-recovery injector pair
            # this record back to its dangling assistant function_call
            # by framework call_id rather than by tool_name + appearance
            # order.  The injector treats the field as optional so
            # records from future code paths that don't supply it still
            # flow through the name+order fallback.
            "parent_call_id": self._parent_call_id,
            "parent_provider_call_id": self._parent_call_id,
            "parent_event_call_id": self._parent_event_call_id,
            **({"sub_agent_log_file": self._sub_agent_log_file} if self._sub_agent_log_file else {}),
            "created_at": now,
            "paused_at": now,
            "failure_reason": (self._failure_reason.value if self._failure_reason else ""),
            "last_error": self._last_error,
            "retry_attempts_total": self._retry_attempts_total,
        }

    def _write_persisted(self) -> None:
        """Write the paused-state snapshot to disk.  Best-effort — IO errors are logged."""
        path = self._persist_path()
        if path is None:
            return
        try:
            atomic_write_owner_only_text(
                path,
                json.dumps(self._serialize_state(), indent=2, allow_nan=False),
            )
        except OSError as e:
            logger.warning("Failed to persist paused sub-agent %s: %s", self._invocation_id, e)

    def _queue_persisted_cleanup_if_terminal(self) -> None:
        """Queue the pause record for deletion once parent session save succeeds."""
        if self._status == SubAgentStatus.PAUSED:
            return
        path = self._persist_path()
        if path is None or self._pending_record_finalizer is None:
            return
        self._pending_record_finalizer(path)

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_text(response: AgentResponse[Any]) -> str:
        """Extract a parent-visible result from an ``AgentResponse``.

        Text remains authoritative. A hosted image/artifact-only response is
        still a successful structured result, so synthesize a neutral parent
        result after its rich payload has been published by reconciliation.
        """
        if response.text:
            return response.text

        has_images = False
        has_artifacts = False
        for message in response.messages:
            for content in message.contents:
                if content.type not in TOOL_RESULT_CONTENT_TYPES or not content.provider_hosted:
                    continue
                view = adapt_hosted_tool(None, content)
                if hosted_replay_status(view, has_result=True) != HostedToolStatus.COMPLETED:
                    continue
                has_images = has_images or bool(view.image_contents)
                has_artifacts = has_artifacts or bool(view.artifacts)
        if has_images and has_artifacts:
            return "Sub-agent returned image and artifact output."
        if has_images:
            return "Sub-agent returned image output."
        if has_artifacts:
            return "Sub-agent returned artifact output."
        return ""
