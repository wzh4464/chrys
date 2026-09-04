# Copyright (c) 2026 Chrys. All rights reserved.

"""Executor — drives the LLM tool loop with interrupt and inject support.

Wraps the Chrys kernel Agent.run() and coordinates:
- Event publishing via ToolEventMiddleware
- Approval gating via ApprovalMiddleware (inside the tool loop)
- Interrupt via InterruptMiddleware
- User prompt injection via InjectionMiddleware
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypedDict, TypeGuard

from chrys.foundation.errors import clean_error_message as _clean_error_message
from chrys.foundation.errors import is_retryable as _is_retryable
from chrys.foundation.events.types import (
    AgentMessage,
    AgentThinking,
    Error,
    PresentationAttemptAccepted,
    PresentationAttemptRejected,
    ProvisionalPresentation,
    RetryAttempt,
    ToolCallArgsUpdated,
    ToolCallProgress,
    ToolCallResult,
    ToolCallStart,
    ToolCallStatusUpdated,
)
from chrys.foundation.hosted_tools import HOSTED_TOOL_DEFAULT_KIND_BY_FAMILY, HostedToolStatus
from chrys.foundation.i18n import msg
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.turns import (
    UserMessageKind,
    current_turn_start,
    is_continuation_message,
    user_text_matches,
)
from chrys.foundation.retry import (
    HistorySnapshot as _HistorySnapshot,
)
from chrys.foundation.retry import (
    RetryAttemptInfo,
    StreamRetryLoop,
    StreamStallExhausted,
    restore_message_properties,
    snapshot_message_properties,
)
from chrys.foundation.retry import (
    StreamStall as _StreamStall,
)
from chrys.foundation.trajectory.context import TRAJECTORY_CONTEXT_KWARG, trajectory_scope
from chrys.foundation.trajectory.envelope import Link, LinkRelation, MeasurementSource, measurement
from chrys.foundation.trajectory.event_types import EventType as TrajectoryEventType
from chrys.foundation.trajectory.event_types import ModelRunEndReason, RetryMode, RetryReason
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory.metadata import ensure_analytics_item_id, read_analytics_item_id
from chrys.foundation.util.time import parse_created_at
from chrys.kernel import (
    Agent,
    AgentSession,
    Message,
    StallExhaustedAction,
    WireRetryPolicy,
    is_retry_boundary_update,
    resolve_storage_mode_and_handles,
)
from chrys.kernel.loop import ConsumedInjectionMessageProbe, LoopRecorderSnapshot
from chrys.service.agent_middleware import (
    ApprovalMiddleware,
    AskUserMiddleware,
    IntermediateTextBuffer,
    InterruptMiddleware,
    SleepMiddleware,
    ToolEventMiddleware,
)
from chrys.service.agent_middleware.control.approval import ApprovalRetrySnapshot
from chrys.service.agent_middleware.events.hosted_tools import (
    FinalTextOp,
    HostedPresentationBridge,
    HostedToolArgsOp,
    HostedToolProgressOp,
    HostedToolResultOp,
    HostedToolStartOp,
    HostedToolStatusOp,
    IntermediateTextOp,
    PresentationAttemptAcceptedOp,
    PresentationAttemptRejectedOp,
    PresentationSinkOperation,
)
from chrys.service.agent_middleware.events.tool_events import ToolEventRetrySnapshot
from chrys.service.agent_middleware.response_validation import (
    ResponseValidationMiddleware,
    RetryableResponseValidationError,
    hosted_commits_from_error,
)
from chrys.service.context.providers.history import PRE_OUTPUT_HISTORY_LEN_STATE_KEY
from chrys.service.session.message_metadata import (
    LAST_ASSISTANT_CREATED_AT_STATE_KEY,
    MESSAGE_CREATED_AT_KEY,
    stamp_message_created_at,
    try_normalize_created_at,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
    from typing import Protocol

    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.trajectory.context import TrajectoryContext
    from chrys.kernel import AgentResponse, AgentResponseUpdate, LoopRecorder, ResponseStream
    from chrys.kernel.middleware import ChatMiddleware, FunctionMiddleware
    from chrys.service.agent_middleware.events.tool_events import ToolBatchRecord
    from chrys.service.agent_middleware.injection import InjectionMiddleware, QueuedInjection
    from chrys.service.approval.policy import ApprovalMode
    from chrys.service.context.compaction import UnifiedContextStrategy
    from chrys.service.context.compaction.strategy import CompactionRetrySnapshot
    from chrys.service.hooks.manager import HookManager
    from chrys.service.mutations.coordination import MutationCoordinator
    from chrys.service.mutations.tracker import MutationTracker
    from chrys.service.trajectory.preparation import PreparationTrace

    class RecoveryInputRecorder(Protocol):
        """Registers replayed input as recovery current input (``TurnRuntimeState.set_current_input``)."""

        def __call__(
            self,
            text: str,
            contents: list[Any] | None,
            created_at: datetime | str | None,
            kind: UserMessageKind = "opener",
        ) -> None: ...


logger = logging.getLogger(__name__)

_RETRY_STREAM_STALLED = msg(
    "retry.stream_stalled",
    fallback="Stream stalled",
)


class _AssistantMessageEventKwargs(TypedDict, total=False):
    """Optional timestamp forwarded to ``AgentMessage``."""

    timestamp: datetime


class _AgentRunKwargs(TypedDict, total=False):
    """Keyword arguments forwarded unchanged to ``Agent.run``."""

    session: AgentSession
    middleware: Sequence[ChatMiddleware | FunctionMiddleware]
    options: Mapping[str, Any]
    compaction_strategy: Any
    tokenizer: Any
    client_kwargs: Mapping[str, Any]


def _is_string_keyed_dict(value: object) -> TypeGuard[dict[str, Any]]:
    """Narrow internal kwargs dictionaries whose producers use string keys."""
    return isinstance(value, dict)


@dataclass(frozen=True, slots=True)
class _ExecutorRetryState:
    """Auxiliary per-run buffers rolled back with framework history."""

    loop_recorder: LoopRecorderSnapshot | None
    tool_events: ToolEventRetrySnapshot
    approval: ApprovalRetrySnapshot
    compaction: CompactionRetrySnapshot | None


def _make_user_message(
    contents: list[Any], created_at: datetime | str | None = None, *, item_id: str | None = None
) -> Message:
    """Create a user message with Chrys persisted metadata."""
    user_message = Message("user", contents)
    stamp_message_created_at(user_message, created_at)
    ensure_analytics_item_id(user_message.additional_properties, item_id=item_id)
    return user_message


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


def _make_injected_message(
    contents: list[Any], created_at: datetime | str | None = None, *, item_id: str | None = None
) -> Message:
    """Create a mid-turn user message flagged as user-authored input."""
    user_message = _make_user_message(contents, created_at, item_id=item_id)
    user_message.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    return user_message


def _replay_user_message(
    contents: list[Any], created_at: datetime | str | None = None, *, item_id: str | None = None
) -> Message:
    """Recreate an existing user message without inventing missing metadata.

    *item_id* is the popped original's analytics id: the replay is the same
    persisted item, so it keeps the identity the trajectory already refers to.
    """
    user_message = Message("user", contents)
    if created_at:
        stamp_message_created_at(user_message, created_at)
    if item_id is not None:
        ensure_analytics_item_id(user_message.additional_properties, item_id=item_id)
    return user_message


def _existing_message_created_at(message: Message) -> datetime | str | None:
    value = message.additional_properties.get(MESSAGE_CREATED_AT_KEY)
    return try_normalize_created_at(value) if isinstance(value, datetime | str) else None


# ``_HistorySnapshot`` / ``_StreamStall`` / ``_clean_error_message`` keep
# their Executor-private aliases for tests and older internal call sites
# that referenced them under those names.


@dataclass(slots=True)
class _ExecutorWireRetryPolicy:
    """Main-agent policy for retrying one local logical provider call."""

    max_retries: int
    stall_timeout_seconds: float | None
    stall_max_retries: int
    stall_exhausted_action: StallExhaustedAction
    backoff_schedule: tuple[int, ...]
    interrupted: Callable[[], bool]
    interruptible_sleep: Callable[[int], Awaitable[bool]]
    publish_retry: Callable[[str, int, int, int, BaseException], Awaitable[None]]
    replay_injections: Callable[[], None]
    # Consumed duck-typed by the kernel's streaming wire replay: hosted tool
    # calls the current (aborted) wire attempt already executed server-side.
    hosted_commits_in_flight: Callable[[], tuple[str, ...]] | None = None

    def backoff_seconds(self, attempt: int) -> int:
        if not self.backoff_schedule:
            return 0
        return self.backoff_schedule[min(attempt, len(self.backoff_schedule) - 1)]

    def is_retryable(self, exc: BaseException) -> bool:
        return _is_retryable(exc)

    def is_interrupted(self) -> bool:
        return self.interrupted()

    async def sleep(self, seconds: int) -> bool:
        return await self.interruptible_sleep(seconds)

    async def on_retry(
        self,
        message: str,
        attempt: int,
        max_attempts: int,
        delay_seconds: int,
        exc: BaseException,
    ) -> None:
        await self.publish_retry(message, attempt, max_attempts, delay_seconds, exc)

    def before_retry(self) -> None:
        self.replay_injections()


def _has_live_continuation_token(run_kwargs: Mapping[str, object] | None) -> bool:
    """True when the retry-owned options still reference a live background response."""
    if run_kwargs is None:
        return False
    options = run_kwargs.get("options")
    return isinstance(options, dict) and options.get("continuation_token") is not None


def _continuation_token_observer_for(run_kwargs: _AgentRunKwargs) -> Callable[[Any], None]:
    """Mirror the kernel's live continuation token into the retry-owned options.

    A transient failure while polling a stored background response must
    resume that response on the next whole-run attempt — the token has to
    live in the options the retry loop re-issues, not only in the kernel's
    attempt-local copy. Writes go through ``run_kwargs`` at call time so a
    handle-stripping restore that replaces the options dict cannot orphan
    the observer.
    """

    def _observe(token: Any) -> None:
        raw_options = run_kwargs.get("options")
        options = raw_options if _is_string_keyed_dict(raw_options) else None
        if token is None:
            if options is not None:
                options.pop("continuation_token", None)
            return
        if options is None:
            options = {}
            run_kwargs["options"] = options
        options["continuation_token"] = token

    return _observe


def _run_retry_reason(exc: BaseException) -> str:
    """Closed reason code for a whole-run retry decision."""
    if isinstance(exc, _StreamStall | StreamStallExhausted):
        return RetryReason.STREAM_STALL
    if isinstance(exc, RetryableResponseValidationError):
        return RetryReason.VALIDATION_REJECTED
    return RetryReason.TRANSIENT_ERROR


class Executor:
    """Drives the agent execution loop with event-driven extensions.

    Manages a single agent.run() cycle.  Approval is handled by
    ``ApprovalMiddleware`` inside the tool loop — no snapshot-restore
    re-submission needed.
    """

    # Resilience: retry on transient network errors (proxy drops,
    # connection resets, timeouts).  Applies to both blocking and
    # streaming paths.  The classifier lives in :mod:`chrys.foundation.errors`
    # and the loop lives in :mod:`chrys.foundation.retry` so sub-agents share
    # the same policy without reaching back into this class.
    _MAX_RETRIES = 5
    _BACKOFF_SCHEDULE = (3, 7, 15, 30, 60)
    # Fallback when no per-run stream_attempt_timeout is supplied (e.g. tests).
    # Production callers pass ModelProfile.http_read_timeout so stall detection
    # matches the HTTP client's read timeout.
    _DEFAULT_STREAM_ATTEMPT_TIMEOUT = 300.0  # seconds
    # Class-level fallbacks so partially-constructed executors (tests build
    # them via ``object.__new__``) read "no probe wired" instead of raising.
    _hosted_commits_probe: Callable[[], tuple[str, ...]] | None = None
    _hosted_commits_in_flight_probe: Callable[[], tuple[str, ...]] | None = None
    _max_retries_override: int | None = None
    _hosted_bridge: HostedPresentationBridge | None = None
    _requirement_phase = ""
    _last_response_text = ""

    def __init__(
        self,
        *,
        agent: Agent,
        session: AgentSession,
        event_bus: EventBus,
        approval_middleware: ApprovalMiddleware,
        ask_user_middleware: AskUserMiddleware,
        injection_middleware: InjectionMiddleware,
        loop_recorder: LoopRecorder | None = None,
        session_id: str | None = None,
        compaction_strategy: UnifiedContextStrategy | None = None,
        stream: bool = False,
        intermediate_buffer: IntermediateTextBuffer | None = None,
        chat_options: dict[str, Any] | None = None,
        mutation_tracker: MutationTracker | None = None,
        mutation_coordinator: MutationCoordinator | None = None,
        stream_attempt_timeout: float | None = None,
        max_transient_retries: int | None = None,
        hook_manager: HookManager | None = None,
        profile_name: str = "",
        workspace_cwd: str = "",
        serialize_implicit_windows: bool = False,
        extra_function_middleware: Sequence[FunctionMiddleware] = (),
        run_cycle_start_hooks: Sequence[Callable[[], None]] = (),
        hosted_commits_probe: Callable[[], tuple[str, ...]] | None = None,
        hosted_commits_in_flight_probe: Callable[[], tuple[str, ...]] | None = None,
        response_validation_middleware: ResponseValidationMiddleware | None = None,
        publish_intermediate_text: Callable[[str], Awaitable[None]] | None = None,
        commit_intermediate_text: Callable[[str, int], None] | None = None,
        tool_result_ceiling_tokens: int | None = None,
    ) -> None:
        self._agent = agent
        self._session = session
        self._bus = event_bus
        self._approval = approval_middleware
        self._ask_user = ask_user_middleware
        self._sleep = SleepMiddleware(event_bus, session_id=session_id)
        self._injection = injection_middleware
        self._loop_recorder = loop_recorder
        self._session_id = session_id
        self._compaction_strategy = compaction_strategy
        self._stream = stream
        self._chat_options = chat_options
        self._stream_attempt_timeout = (
            stream_attempt_timeout if stream_attempt_timeout is not None else self._DEFAULT_STREAM_ATTEMPT_TIMEOUT
        )
        self._max_retries_override = max_transient_retries

        self._intermediate_buffer = intermediate_buffer
        self._response_validation = response_validation_middleware
        self._publish_intermediate_text = publish_intermediate_text
        self._commit_intermediate_text = commit_intermediate_text
        self._provisional_intermediate_batches: dict[tuple[str, str], int] = {}
        self._hosted_run_generation = 0
        self._hosted_bridge = None
        self._tool_events = ToolEventMiddleware(
            event_bus,
            session_id=session_id,
            intermediate_buffer=intermediate_buffer,
            mutation_tracker=mutation_tracker,
            hook_manager=hook_manager,
            profile_name=profile_name,
            workspace_cwd=workspace_cwd,
            serialize_implicit_windows=serialize_implicit_windows,
            mutation_coordinator=mutation_coordinator,
            on_start_published=self._local_call_start_published,
            tool_result_ceiling_tokens=tool_result_ceiling_tokens,
            workflow_phase_provider=lambda: self._requirement_phase,
        )
        self._interrupt = InterruptMiddleware()
        self._extra_function_middleware = tuple(extra_function_middleware)
        # Fired once at the start of every user-initiated run cycle, before
        # the first attempt.  Components that carry state across the cycle's
        # whole-run retry attempts (e.g. the validation middleware's retry
        # budget) register here so an aborted cycle — interrupt during outer
        # backoff, unrelated exception — cannot leak state into the next
        # independent run.
        self._run_cycle_start_hooks = tuple(run_cycle_start_hooks)
        # Validation-middleware probes for provider-hosted tool executions the
        # loop recorder cannot see (the rejected/aborted exchange never reaches
        # it): run-scoped for the whole-run retry gate, wire-attempt-scoped for
        # the kernel's per-wire replay veto.
        self._hosted_commits_probe = hosted_commits_probe
        self._hosted_commits_in_flight_probe = hosted_commits_in_flight_probe
        self._running = False
        self._was_interrupted = False
        self._run_failed = False
        self._last_error = ""
        # Last continuation token announced by the kernel and not yet
        # resolved to a terminal response.  Run kwargs are rebuilt per run,
        # so without this carry-over a user retry after a gate-blocked poll
        # failure would CREATE a second background response while the
        # announced one keeps running remotely (duplicate hosted work).
        self._pending_continuation_token: Any = None
        self._current_agent_task: asyncio.Task | None = None
        # Crash-recovery channel to the orchestration turn state, wired by the
        # engine after build (``TurnRuntimeState.set_current_input``).  The
        # bare-resume replay branch pops the anchor user message out of state
        # and must register it as recovery current input at pop time — the
        # component that destructively reads is the component that makes it
        # durable.
        self.recovery_input_recorder: RecoveryInputRecorder | None = None
        # Trajectory recording for the next pass, set by the turn runner:
        # the ambient context every model run of the pass binds (None = not
        # recording) and the pre-minted item id of the message opening it.
        self.trajectory_context: TrajectoryContext | None = None
        self._opening_item_id: str | None = None
        # Per-pass model-run bookkeeping: the last run operation a retry
        # links back to, and the pre-minted operation id for a scheduled
        # retry's next run.
        self._trajectory_last_run_id: str | None = None
        self._trajectory_run_attempts = 0
        self._trajectory_retry_run_id: str | None = None
        # Requirement-clarification owns presentation only; execution still
        # uses this one Executor.  The baseline response is captured and
        # exposed as provisional without closing the frontend turn.
        self._requirement_phase = ""
        self._last_response_text = ""

    def set_opening_item_id(self, item_id: str | None) -> None:
        """Pre-assign the analytics item id the next opening user message takes."""
        self._opening_item_id = item_id

    def set_requirement_phase(self, phase: str) -> None:
        """Tag subsequent assistant presentation with a workflow phase."""
        self._requirement_phase = phase

    @property
    def last_response_text(self) -> str:
        """Last accepted assistant text from the current executor pass."""
        return self._last_response_text

    def snapshot_history(self) -> _HistorySnapshot:
        """Capture a complete same-executor checkpoint for orchestration."""
        return self._snapshot_history()

    def restore_history(self, snapshot: _HistorySnapshot) -> None:
        """Restore a checkpoint captured by :meth:`snapshot_history`."""
        self._restore_history(snapshot)

    def adopt_fallback_success(self, text: str) -> None:
        """Clear a failed repair outcome after orchestration restored P0."""
        self._was_interrupted = False
        self._run_failed = False
        self._last_error = ""
        self._last_response_text = text
        self._pending_continuation_token = None

    async def publish_last_response_as_final(self) -> None:
        """Promote the retained response to the terminal answer."""
        await self._bus.publish(
            AgentMessage(
                text=self._last_response_text,
                is_final=True,
                requirement_phase=self._requirement_phase,
                workflow_phase=self._requirement_phase,
                **self._assistant_message_event_kwargs(),
                session_id=self._session_id,
            )
        )

    def _take_opening_item_id(self) -> str | None:
        item_id = self._opening_item_id
        self._opening_item_id = None
        return item_id

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def was_interrupted(self) -> bool:
        """True if the last run was terminated by a user interrupt."""
        return self._was_interrupted

    @property
    def run_failed(self) -> bool:
        """True if the last run ended with an unhandled exception (not interrupt)."""
        return self._run_failed

    @property
    def last_error(self) -> str:
        """The error message from the last failed run, or empty string."""
        return self._last_error

    def record_pre_run_interrupt(self) -> None:
        """Record a task-scoped interrupt that arrived before execution began."""
        self._was_interrupted = True
        self._run_failed = False
        self._last_error = ""
        # A cancelled executor pass abandons any incomplete provider-side
        # response even when Stop prevents ``run()``/``resume()`` from
        # reaching their normal reset paths.
        self._pending_continuation_token = None
        self._interrupt.reset()

    @property
    def history_state(self) -> dict:
        """The ``chrys_history`` provider state dict.

        Returns the live dict — mutations are visible to kernel consumers.
        Uses ``setdefault`` so the same dict is always returned.
        """
        return self._session.state.setdefault("chrys_history", {})

    @property
    def compaction_strategy(self) -> UnifiedContextStrategy | None:
        """Return the strategy whose admission/calibration state this executor owns."""
        return self._compaction_strategy

    @history_state.setter
    def history_state(self, value: dict) -> None:
        """Replace the ``chrys_history`` state (used for session restore)."""
        self._session.state["chrys_history"] = value

    @property
    def service_session_id(self) -> str:
        """Provider-managed conversation/response id for APIs that support it."""
        return self._session.service_session_id or ""

    @service_session_id.setter
    def service_session_id(self, value: str) -> None:
        """Restore a provider-managed session id from Chrys persistence."""
        self._session.service_session_id = value or None

    @property
    def service_session_storage_enabled(self) -> bool:
        """Return whether the first request uses provider-side history."""
        try:
            stores_by_default = bool(self._agent.client.STORES_BY_DEFAULT)
        except AttributeError:
            stores_by_default = False
        try:
            force_stateless = bool(self._agent.client.FORCES_STATELESS)
        except AttributeError:
            force_stateless = False
        return resolve_storage_mode_and_handles(
            self._chat_options,
            stores_by_default=stores_by_default,
            force_stateless=force_stateless,
        ).service_side

    def drain_batch_records(self) -> list[ToolBatchRecord]:
        """Drain batch records recorded by the tool event middleware."""
        return self._tool_events.drain_batch_records()

    async def _local_call_start_published(self, provider_call_id: str) -> None:
        """Release hosted presentation ordered behind a local tool call."""
        if self._hosted_bridge is not None:
            await self._hosted_bridge.local_call_start_published(provider_call_id)

    @staticmethod
    def _artifact_descriptors(operation: HostedToolResultOp) -> list[dict[str, Any]]:
        """Build bounded JSON-safe descriptors for hosted result artifacts."""
        descriptors: list[dict[str, Any]] = []
        for artifact in operation.view.artifacts:
            # "path" must stay a real URI: consumers turn it into links, and a
            # bare hosted filename (OpenAI hosted_file) is not addressable.
            descriptor = {
                "id": artifact.file_id or artifact.vector_store_id or artifact.id or "",
                "name": artifact.name or "",
                "path": artifact.uri or "",
                "mime": artifact.media_type or "",
            }
            size = artifact.additional_properties.get("size")
            if isinstance(size, int) and not isinstance(size, bool):
                descriptor["size"] = size
            descriptors.append({key: value for key, value in descriptor.items() if value != ""})
        return descriptors

    async def _publish_hosted_operation(self, operation: PresentationSinkOperation) -> None:
        """Map one provider-neutral presentation operation onto EventBus events."""
        if isinstance(operation, IntermediateTextOp):
            if operation.provisional:
                segment_id = operation.segment_ids[0] if operation.segment_ids else ""
                if self._intermediate_buffer is not None:
                    self._intermediate_buffer.new_batch()
                    self._provisional_intermediate_batches[(operation.attempt_id, segment_id)] = (
                        self._intermediate_buffer.batch_id
                    )
                await self._bus.publish(
                    AgentMessage(
                        text=operation.text,
                        is_final=False,
                        is_intermediate=True,
                        presentation=ProvisionalPresentation(operation.attempt_id, segment_id),
                        session_id=self._session_id,
                    )
                )
                return
            if self._publish_intermediate_text is not None:
                await self._publish_intermediate_text(operation.text)
            else:
                await self._bus.publish(
                    AgentMessage(
                        text=operation.text,
                        is_final=False,
                        is_intermediate=True,
                        session_id=self._session_id,
                    )
                )
            return
        if isinstance(operation, PresentationAttemptAcceptedOp):
            accepted_ids = tuple(segment_id for segment in operation.segments for segment_id in segment.segment_ids)
            if self._commit_intermediate_text is not None:
                for segment in operation.segments:
                    segment_id = segment.segment_ids[0] if segment.segment_ids else ""
                    batch_id = self._provisional_intermediate_batches.get((operation.attempt_id, segment_id))
                    if batch_id is not None:
                        self._commit_intermediate_text(segment.text, batch_id)
            self._drop_provisional_batches(operation.attempt_id)
            await self._bus.publish(
                PresentationAttemptAccepted(
                    attempt_id=operation.attempt_id,
                    segment_ids=accepted_ids,
                    session_id=self._session_id,
                )
            )
            return
        if isinstance(operation, PresentationAttemptRejectedOp):
            self._drop_provisional_batches(operation.attempt_id)
            await self._bus.publish(
                PresentationAttemptRejected(attempt_id=operation.attempt_id, session_id=self._session_id)
            )
            return
        if isinstance(operation, FinalTextOp):
            self._last_response_text = operation.text
            provisional = self._requirement_phase == "initial_implementation"
            await self._bus.publish(
                AgentMessage(
                    text=operation.text,
                    is_final=not provisional,
                    is_provisional=provisional,
                    requirement_phase=self._requirement_phase,
                    workflow_phase=self._requirement_phase,
                    structured_output_completed=operation.structured_output_completed,
                    **self._assistant_message_event_kwargs(),
                    session_id=self._session_id,
                )
            )
            return

        view = operation.view
        if isinstance(operation, HostedToolStartOp):
            await self._bus.publish(
                ToolCallStart(
                    tool_name=view.tool_name,
                    call_id=operation.presentation_id,
                    provider_hosted=True,
                    hosted_family=view.family,
                    provider=view.provider,
                    provider_item_type=view.provider_item_type,
                    provider_call_id=view.provider_call_id,
                    provider_status=view.provider_status,
                    tool_kind=HOSTED_TOOL_DEFAULT_KIND_BY_FAMILY.get(view.family, ""),
                    args=view.arguments,
                    session_id=self._session_id,
                    workflow_phase=self._requirement_phase,
                )
            )

        elif isinstance(operation, HostedToolArgsOp):
            await self._bus.publish(
                ToolCallArgsUpdated(
                    tool_name=view.tool_name,
                    call_id=operation.presentation_id,
                    provider_hosted=True,
                    hosted_family=view.family,
                    provider=view.provider,
                    provider_item_type=view.provider_item_type,
                    provider_call_id=view.provider_call_id,
                    provider_status=view.provider_status,
                    tool_kind=HOSTED_TOOL_DEFAULT_KIND_BY_FAMILY.get(view.family, ""),
                    args=view.arguments,
                    session_id=self._session_id,
                    workflow_phase=self._requirement_phase,
                )
            )
        elif isinstance(operation, HostedToolProgressOp):
            await self._bus.publish(
                ToolCallProgress(
                    tool_name=view.tool_name,
                    call_id=operation.presentation_id,
                    provider_hosted=True,
                    hosted_family=view.family,
                    provider=view.provider,
                    provider_item_type=view.provider_item_type,
                    provider_call_id=view.provider_call_id,
                    provider_status=view.provider_status,
                    lines=view.result_text.splitlines(),
                    image_contents=view.image_contents,
                    snapshot_metadata=view.metadata,
                    session_id=self._session_id,
                    workflow_phase=self._requirement_phase,
                )
            )
        elif isinstance(operation, HostedToolStatusOp):
            metadata = dict(view.metadata)
            if view.result_text:
                metadata["result_text"] = view.result_text
            if view.provider_item_type:
                metadata["provider_item_type"] = view.provider_item_type
            if view.provider_call_id:
                metadata["provider_call_id"] = view.provider_call_id
            await self._bus.publish(
                ToolCallStatusUpdated(
                    tool_name=view.tool_name,
                    call_id=operation.presentation_id,
                    provider_hosted=True,
                    hosted_family=view.family,
                    provider=view.provider,
                    status=view.status,
                    provider_status=view.provider_status,
                    metadata=metadata,
                    session_id=self._session_id,
                    workflow_phase=self._requirement_phase,
                )
            )
        elif isinstance(operation, HostedToolResultOp):
            await self._bus.publish(
                ToolCallResult(
                    tool_name=view.tool_name,
                    call_id=operation.presentation_id,
                    provider_hosted=True,
                    hosted_family=view.family,
                    provider=view.provider,
                    provider_item_type=view.provider_item_type,
                    provider_call_id=view.provider_call_id,
                    provider_status=view.provider_status,
                    result=view.result_text,
                    image_contents=view.image_contents,
                    metadata=view.metadata,
                    artifacts=self._artifact_descriptors(operation),
                    session_id=self._session_id,
                    workflow_phase=self._requirement_phase,
                )
            )

    def _drop_provisional_batches(self, attempt_id: str) -> None:
        stale = [key for key in self._provisional_intermediate_batches if key[0] == attempt_id]
        for key in stale:
            self._provisional_intermediate_batches.pop(key, None)

    def set_user_message(self, text: str) -> None:
        """Forward user message to the approval middleware for judge context."""
        self._approval.set_user_message(text)

    def set_user_messages(self, messages: list[str]) -> None:
        """Forward current-turn user messages to the approval middleware."""
        self._approval.set_user_messages(messages)

    def append_user_message(self, text: str) -> None:
        """Append a current-turn user message to approval judge context."""
        self._approval.append_user_message(text)

    def remove_user_message(self, text: str) -> None:
        """Remove a withdrawn current-turn user message from approval judge context."""
        self._approval.remove_user_message(text)

    def set_approval_mode(self, mode: ApprovalMode) -> None:
        """Forward approval-mode change to the approval middleware."""
        self._approval.set_approval_mode(mode)

    def drain_approval_decisions(self) -> list[dict[str, str]]:
        """Return and clear collected approval decisions."""
        return self._approval.drain_decisions()

    def _build_run_kwargs(self, wire_retry_policy: WireRetryPolicy | None = None) -> _AgentRunKwargs:
        """Build common kwargs for agent.run()."""
        # ToolEventMiddleware owns hook dispatch too, so hook-modified args
        # are reflected consistently in UI events, mutation tracking, and
        # approval requests.
        # Per-run channel consumed by the chrys ToolLoopLayer (fresh dict per
        # call; the agent merges the session into the same mapping). A None
        # recorder is not injected — absent key and None mean the same thing
        # to the loop, and an absent key keeps the mapping honest.
        client_kwargs: dict[str, object] = {}
        if self._loop_recorder is not None:
            client_kwargs["loop_recorder"] = self._loop_recorder
        if wire_retry_policy is not None:
            client_kwargs["wire_retry_policy"] = wire_retry_policy
        client_kwargs["consumed_injection_message_probe"] = ConsumedInjectionMessageProbe(
            drain_consumed_injection_messages=self._injection.drain_consumed_injection_messages,
            commit_consumed_injections=self._injection.commit_logical_call,
        )
        # Chain order is pinned (first = outermost): extras sit INSIDE
        # tool_events (a short-circuiting extra must still publish
        # ToolCallStart/ToolCallResult) and INSIDE approval (no extra can
        # bypass approval on gated kinds).
        kwargs: _AgentRunKwargs = {
            "session": self._session,
            "middleware": [
                self._tool_events,
                self._ask_user,
                self._approval,
                *self._extra_function_middleware,
                self._sleep,
                self._interrupt,
            ],
            "client_kwargs": client_kwargs,
        }
        if self._compaction_strategy is not None:
            kwargs["compaction_strategy"] = self._compaction_strategy
            # The raw client needs the same tokenizer to estimate live tool
            # definitions before the first calibrated provider response.  The
            # strategy remains the single owner of the tokenizer instance.
            kwargs["tokenizer"] = self._compaction_strategy.tokenizer
        if self._chat_options:
            options_copy = dict(self._chat_options)
            if _is_string_keyed_dict(extra_body := options_copy.get("extra_body")):
                options_copy["extra_body"] = dict(extra_body)
            kwargs["options"] = options_copy
        if self._pending_continuation_token is not None:
            # A previous run of this turn left a background response in
            # flight (poll failed, gate blocked the auto-retry).  Resume it
            # instead of creating a duplicate; the kernel clears the token
            # on any terminal judgment.
            seeded = kwargs.get("options")
            if not _is_string_keyed_dict(seeded):
                seeded = {}
                kwargs["options"] = seeded
            seeded["continuation_token"] = self._pending_continuation_token
        mirror_to_run_kwargs = _continuation_token_observer_for(kwargs)

        def _observe_continuation_token(token: Any) -> None:
            self._pending_continuation_token = token
            mirror_to_run_kwargs(token)

        client_kwargs["continuation_token_observer"] = _observe_continuation_token
        return kwargs

    def _build_wire_retry_policy(self) -> WireRetryPolicy:
        max_retries = self._effective_max_retries()
        return _ExecutorWireRetryPolicy(
            max_retries=max_retries,
            stall_timeout_seconds=self._stream_attempt_timeout,
            stall_max_retries=max_retries,
            stall_exhausted_action=StallExhaustedAction.BLOCKING_FALLBACK,
            backoff_schedule=self._BACKOFF_SCHEDULE,
            interrupted=lambda: self._interrupt.is_interrupted,
            interruptible_sleep=self._interruptible_sleep,
            publish_retry=self._publish_wire_retry_attempt,
            replay_injections=self._injection.restore_for_retry,
            hosted_commits_in_flight=self._hosted_commits_in_flight_probe,
        )

    def _effective_max_retries(self) -> int:
        """Return the injected budget or the lazily resolved class fallback."""
        if self._max_retries_override is not None:
            return self._max_retries_override
        return self._MAX_RETRIES

    def _may_retry_attempt(self, exc: BaseException, run_kwargs: Mapping[str, object] | None = None) -> bool:
        """Whole-run retry gate: answered tool work must never re-execute.

        Locally answered results are counted by the loop recorder.  Provider-
        hosted calls (hosted MCP, hosted shell) execute inside the failed
        exchange itself and never reach the recorder — their evidence rides
        on the raised validation error (the middleware swallows the invalid
        response before the kernel loop sees it) or, for stalls and transport
        drops mid-stream, on the middleware's run-scoped observation probe.
        A live continuation token exempts the gate: the retry then resumes
        the already-created background response instead of re-creating the
        request, so nothing hosted runs twice.
        """
        if self._loop_recorder is not None and self._loop_recorder.committed_count:
            return False
        hosted = hosted_commits_from_error(exc)
        if not hosted and self._hosted_commits_probe is not None:
            hosted = self._hosted_commits_probe()
        if hosted and not _has_live_continuation_token(run_kwargs):
            logger.warning(
                "Not retrying failed attempt: provider-hosted tool call(s) already executed (%s)",
                ", ".join(hosted),
            )
            return False
        return True

    def _restore_service_retry_inputs(self, run_kwargs: _AgentRunKwargs) -> None:
        """Replay failed-call injections and discard every stale service handle."""
        raw_options = run_kwargs.get("options")
        options = raw_options if _is_string_keyed_dict(raw_options) else {}
        if options.get("continuation_token") is None:
            # A live token means the retry resumes the already-created
            # background response: the create that consumed the injections
            # succeeded, so the consumed transaction (and its retained-message
            # mirror for the terminal weave) must stay intact.  Replaying here
            # would strand the batch — polls skip consumption, then the
            # terminal commit would destroy the held replay.
            self._injection.restore_for_retry()
        self._session.service_session_id = None
        raw_client_kwargs = run_kwargs.get("client_kwargs")
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
            run_kwargs["options"] = resolution.options_without_handles
        run_kwargs["client_kwargs"] = resolution.client_kwargs_without_handles

    async def run(self, contents: list, created_at: datetime | str | None = None) -> None:
        """Execute an agent run, publishing events throughout.

        *contents* is a list of ``Content``-compatible items (strings,
        ``Content`` objects) that form a single user message.  Each item
        becomes a separate ``Content`` in the ``Message``, supporting
        multi-modal inputs (text, images, etc.).
        """
        # A fresh user message abandons the failed turn — an in-flight
        # background response from it must not answer the new prompt.
        self._pending_continuation_token = None
        user_message = _make_user_message(contents, created_at, item_id=self._take_opening_item_id())
        await self._execute([user_message])

    async def resume(self, additional_text: str = "", created_at: datetime | str | None = None) -> None:
        """Resume from current conversation state.

        When *additional_text* is non-empty, it is used as the mid-turn
        continuation prompt. The text is sent as a real user message inside
        the current turn, so it appears as a proper user turn in history.

        When *additional_text* is empty:

        * If there are completed tool calls after the last user message,
          continues from the transcript with empty input.
        * If there is no completed work, re-sends the original user
          message so the LLM retries from scratch.

        With the ``SystemReminderMiddleware``, session state messages are
        always clean (no ``<system-reminder>`` tags), so no stripping is
        needed on resume.
        """
        state = self._session.state.get("chrys_history", {})
        messages = state.get("messages", [])

        if additional_text:
            # Mid-turn user note: send as the continuation prompt and
            # preserve it in history.  No pop of the orphan user message
            # either — the original prompt stays, the note is appended
            # as a mid-turn follow-up (flagged ``_injected``: user-authored
            # input that must not open a new turn).  The note needs a fresh
            # create to reach the model — retrieving a pending background
            # response would silently ignore it.
            self._pending_continuation_token = None
            opening_item_id = self._take_opening_item_id()
            await self._execute([_make_injected_message([additional_text], created_at, item_id=opening_item_id)])

            # Belt-and-suspenders: if the agent run errored before
            # persisting the user input, ensure the note survives so the
            # next retry can see it.  Scope dedup to the current turn
            # region (messages after the last ``_chrys_kind='turn'``
            # marker) — mirrors the fallback in the empty-text branch
            # below and :meth:`SessionHistoryManager.ensure_user_message`.
            # The dedup is kind-aware: guidance worded identically to the
            # turn opener must still be appended (flagged), not skipped.
            if self._run_failed or self._was_interrupted:
                messages = state.get("messages", [])
                start = current_turn_start(messages)
                has_note = any(user_text_matches(m, additional_text, kind="injected") for m in messages[start:])
                if not has_note:
                    messages.append(_make_injected_message([additional_text], created_at, item_id=opening_item_id))
            return

        # Find the last real user input. Legacy synthetic nudges stay in
        # persisted history for read compatibility but cannot become anchors.
        user_idx = -1
        user_text = ""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user" and not is_continuation_message(messages[i]):
                user_idx = i
                user_text = messages[i].text or ""
                break

        if user_idx < 0:
            self._run_failed = True
            self._last_error = "Cannot resume without a real user message in history."
            logger.error("Executor resume rejected defensively: %s", self._last_error)
            return

        # Check for completed work (assistant/tool) after the user message
        has_work_after = any(messages[j].role in ("assistant", "tool") for j in range(user_idx + 1, len(messages)))

        if has_work_after or self._pending_continuation_token is not None:
            # Completed tool calls exist — or an announced background
            # response is still in flight with no landed output yet.  Either
            # way the transcript continues: the history provider supplies it,
            # and a pending token retrieves the known response instead of
            # creating a duplicate that would re-run hosted work.
            await self._execute([])
        else:
            # No completed work — re-send the original user message.
            # Session state is clean (no <system-reminder> tags) so we
            # re-send the text as-is.
            original_created_at = _existing_message_created_at(messages[user_idx])
            popped = messages.pop(user_idx)
            popped_item_id = read_analytics_item_id(popped.additional_properties)
            text = user_text or "continue"
            replay_msg = _replay_user_message([text], original_created_at, item_id=popped_item_id)
            # Preserve the popped anchor's mid-turn flags: re-sending a
            # popped injection unflagged would launder it into a
            # turn-splitting opener.
            for key in HistoryMarkerKind.MID_TURN_USER_KEYS:
                if popped.additional_properties.get(key):
                    replay_msg.additional_properties[key] = popped.additional_properties[key]
            replay_kind: UserMessageKind = (
                "injected" if popped.additional_properties.get(HistoryMarkerKind.INJECTED_KEY) else "opener"
            )
            # The pop destructively removed the anchor from state while the
            # runner already cleared the recovery current input — until
            # ``after_run`` stores the input again, the replayed message
            # exists nowhere durable. Register it (kind included) so a
            # crash-recovery checkpoint can re-create it.
            if self.recovery_input_recorder is not None:
                self.recovery_input_recorder(text, None, original_created_at, kind=replay_kind)
            await self._execute([replay_msg])

            # If the run failed, ensure the user message wasn't lost.
            # Scope the dedup to the current turn region (messages after
            # the last ``_chrys_kind='turn'`` marker) — mirrors the fix
            # in :meth:`SessionHistoryManager.ensure_user_message`.  A
            # global scan would be fooled by a same-text user message
            # from an earlier turn (e.g. an injection that got anchored
            # into turn 1 carries the same text the user later types
            # as a fresh turn's prompt) and drop the re-append, losing
            # this turn's prompt on interrupt-before-persist.
            # The dedup and the fallback append are both keyed to the
            # popped anchor's kind: a same-text flagged injection must not
            # suppress a popped opener's re-append, and a popped injection
            # dedups against its persisted flagged copy — and is re-created
            # FLAGGED when missing, never laundered into an opener.
            if (self._run_failed or self._was_interrupted) and user_text:
                messages = state.get("messages", [])
                start = current_turn_start(messages)
                has_user_msg = any(user_text_matches(m, user_text, kind=replay_kind) for m in messages[start:])
                if not has_user_msg:
                    fallback_msg = _replay_user_message([text], original_created_at, item_id=popped_item_id)
                    if replay_kind == "injected":
                        fallback_msg.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
                    messages.append(fallback_msg)

    async def _execute(self, input: list[Message]) -> None:
        """Core execution logic shared by run() and resume()."""
        self._running = True
        self._was_interrupted = False
        self._run_failed = False
        self._last_error = ""
        self._last_response_text = ""
        self._interrupt.reset()
        self._reset_trajectory_runs()

        try:
            # Inside the try so a raising hook lands on the normal executor
            # error path instead of escaping with ``_running`` stuck True.
            self._hosted_run_generation += 1
            self._hosted_bridge = HostedPresentationBridge(
                self._publish_hosted_operation,
                run_generation=self._hosted_run_generation,
                batch_id=self._intermediate_buffer.batch_id if self._intermediate_buffer is not None else 0,
            )
            if self._response_validation is not None:
                self._response_validation.set_observation_hook(self._hosted_bridge)
            else:
                await self._hosted_bridge.begin_response(response_index=0)
            for hook in self._run_cycle_start_hooks:
                hook()
            await self._bus.publish(AgentThinking(session_id=self._session_id, workflow_phase=self._requirement_phase))
            await self._flush_carried_compressions()

            if self._stream:
                await self._run_streaming(input)
            else:
                await self._run_blocking(input)

        except asyncio.CancelledError:
            # Child agent task was cancelled by interrupt() — treat as a
            # normal interrupt so post-processing (session save, etc.) runs.
            self._was_interrupted = True
            return

        except Exception as e:
            # User-requested interrupt — flag it but don't publish an error
            if self._interrupt.is_interrupted:
                self._was_interrupted = True
                return

            self._run_failed = True
            self._last_error = _clean_error_message(e)
            tb = traceback.format_exc()
            err_msg = self._last_error
            logger.error("Executor error: %s", err_msg)
            # Terminalize hosted cards before the run error: event consumers
            # without the TUI's error banner (ACP) would otherwise keep them
            # in-progress forever.
            if self._hosted_bridge is not None:
                await self._hosted_bridge.attempt_rejected(err_msg)
            if self._requirement_phase != "repair":
                await self._bus.publish(
                    Error(
                        code="executor_error",
                        message=err_msg,
                        recoverable=True,
                        session_id=self._session_id,
                    )
                )
            logger.debug("Executor traceback:\n%s", tb)
        finally:
            # If the run completed normally but the interrupt flag was set
            # (user clicked Stop after LLM finished but before TUI processed
            # the final message), still mark as interrupted.
            if self._interrupt.is_interrupted:
                self._was_interrupted = True
                if self._hosted_bridge is not None:
                    await self._hosted_bridge.attempt_rejected(
                        "Execution interrupted",
                        status=HostedToolStatus.INTERRUPTED,
                        preserve_provisional=True,
                    )
            if self._response_validation is not None:
                self._response_validation.set_observation_hook(None)
            # ``was_interrupted`` is the durable outcome after this point.
            # Do not let the middleware flag leak into a later retry/turn,
            # where it would look like a new pre-run interrupt.
            self._interrupt.reset()
            self._running = False

    async def _flush_carried_compressions(self) -> None:
        """Commit a prior failed pass's queued folds before retry snapshots.

        A ``compress_context`` call can validate and queue a fold, then the
        pass can terminate before the history provider reaches its normal
        ``after_run`` flush.  The next user retry must treat that request as
        pre-existing committed work.  If it first runs inside
        ``StreamRetryLoop`` instead, a transient failure restores the
        pre-attempt history (removing the new ``CompressedBlock``) after the
        queue has already been consumed.  The live ``ContextCompressed``
        event then has no durable counterpart and disappears on replay.

        Bind the current provider state explicitly and flush before
        ``_run_blocking``/``_stream_with_retry`` capture their attempt
        snapshot.  The history provider binds the same state again inside
        ``agent.run()``; that later bind is intentionally idempotent.
        """
        strategy = self._compaction_strategy
        if strategy is None:
            return
        history_state = self.history_state
        strategy.bind_state(history_state)
        if not await strategy.flush_pending_compressions():
            return
        messages = history_state.get("messages", [])
        if isinstance(messages, list):
            # TurnRunner.pre_run captured its metadata floor before this
            # carried fold shortened history.  Finalization must start at the
            # post-fold boundary so retry output still receives approval,
            # modified-argument, and timestamp annotations.
            history_state[PRE_OUTPUT_HISTORY_LEN_STATE_KEY] = len(messages)

    # ── trajectory: model runs ───────────────────────────────────────

    def _reset_trajectory_runs(self) -> None:
        self._trajectory_last_run_id = None
        self._trajectory_run_attempts = 0
        self._trajectory_retry_run_id = None

    @contextlib.asynccontextmanager
    async def _trajectory_run(self, run_kwargs: _AgentRunKwargs, *, stream: bool) -> AsyncIterator[None]:
        """Bind one ``model.run`` operation around one ``agent.run`` attempt.

        The narrowed context rides two ways: ambiently (for tool middleware
        and side calls, bound before the attempt task is created so the
        task inherits it) and explicitly in ``client_kwargs`` for the
        kernel loop, whose lazy provider streams resolve outside this
        context.
        """
        base = self.trajectory_context
        if base is None:
            yield
            return
        retry_run_id = self._trajectory_retry_run_id
        run_id = retry_run_id or new_analytics_id()
        context = base.with_run(run_id)
        raw_client_kwargs = run_kwargs.get("client_kwargs")
        run_kwargs["client_kwargs"] = {
            **(raw_client_kwargs if raw_client_kwargs is not None else {}),
            TRAJECTORY_CONTEXT_KWARG: context,
        }
        attempt_index = self._trajectory_run_attempts
        self._trajectory_run_attempts += 1
        previous_run_id = self._trajectory_last_run_id
        started_ns = time.monotonic_ns()
        sink = context.sink
        if retry_run_id is not None:
            self._trajectory_retry_run_id = None
            try:
                await sink.emit(
                    context.draft(
                        TrajectoryEventType.RETRY_STARTED,
                        operation_id=run_id,
                        payload={
                            "retry_mode": RetryMode.RUN,
                            "next_operation_id": run_id,
                            "previous_operation_id": previous_run_id,
                        },
                    )
                )
            except Exception:
                logger.debug("Trajectory retry.started emit failed", exc_info=True)

        def _finished_draft(outcome: str) -> Any:
            return context.draft(
                TrajectoryEventType.MODEL_RUN_FINISHED,
                operation_id=run_id,
                payload={
                    "outcome": outcome,
                    "duration_ms": max(0, (time.monotonic_ns() - started_ns) // 1_000_000),
                },
                measurements={"/payload/duration_ms": measurement(MeasurementSource.MONOTONIC_CLOCK, method_version=1)},
            )

        self._trajectory_last_run_id = run_id
        links = (
            (Link(relation=LinkRelation.CAUSED_BY, target_operation_id=base.turn_preamble_operation_id),)
            if attempt_index == 0 and base.turn_preamble_operation_id is not None
            else ()
        )
        try:
            await sink.emit(
                context.draft(
                    TrajectoryEventType.MODEL_RUN_STARTED,
                    operation_id=run_id,
                    payload={
                        "attempt_index": attempt_index,
                        "stream": stream,
                        "service_side_storage": self.service_session_storage_enabled,
                        "previous_run_operation_id": previous_run_id,
                    },
                    links=links,
                )
            )
        except asyncio.CancelledError:
            # The writer commits the opening line before its shielded ack can
            # be cancelled. Context-manager entry will not reach the body
            # below, so settle the operation here before preserving cancel.
            try:
                sink.emit_soon(_finished_draft(ModelRunEndReason.INTERRUPTED))
            except Exception:
                logger.debug("Trajectory model.run.finished emit failed", exc_info=True)
            raise
        except Exception:
            logger.debug("Trajectory model.run.started emit failed", exc_info=True)
        outcome = ModelRunEndReason.COMPLETED
        with trajectory_scope(context):
            try:
                yield
            except asyncio.CancelledError:
                outcome = ModelRunEndReason.INTERRUPTED
                raise
            except BaseException:
                outcome = ModelRunEndReason.INTERRUPTED if self._interrupt.is_interrupted else ModelRunEndReason.FAILED
                raise
            finally:
                draft = _finished_draft(outcome)
                try:
                    # Queued, never awaited. The model has already answered by
                    # the time this runs and its task is gone, so an ack that
                    # waits out a slow writer holds the answer unpublished for
                    # that long — and a Stop pressed in that window finds
                    # nothing to cancel, marks the turn interrupted, and the
                    # finished answer is dropped. Cancellation cannot await
                    # here either: the scope is already unwinding. The line
                    # takes its sequence here, so this span is only left open
                    # when the process does not outlive the queue — and the
                    # turn still gets its terminal from the recorder's close.
                    sink.emit_soon(draft)
                except Exception:
                    logger.debug("Trajectory model.run.finished emit failed", exc_info=True)

    async def _trajectory_run_retry_scheduled(
        self, *, exc: BaseException, delay_seconds: int, fallback_to_blocking: bool = False
    ) -> None:
        """Record a service-side whole-run retry decision (``retry.scheduled``)."""
        context = self.trajectory_context
        if context is None:
            return
        retry_run_id = new_analytics_id()
        self._trajectory_retry_run_id = retry_run_id
        committed = self._loop_recorder is not None and self._loop_recorder.committed_count > 0
        try:
            await context.sink.emit(
                context.draft(
                    TrajectoryEventType.RETRY_SCHEDULED,
                    operation_id=retry_run_id,
                    payload={
                        "reason_code": _run_retry_reason(exc),
                        "delay_ms": max(0, int(delay_seconds * 1000)),
                        "retry_mode": RetryMode.RUN,
                        "previous_operation_id": self._trajectory_last_run_id,
                        "committed_work_present": committed,
                        "fallback_to_blocking": fallback_to_blocking,
                    },
                )
            )
        except Exception:
            logger.debug("Trajectory retry.scheduled emit failed", exc_info=True)

    # ── blocking path ────────────────────────────────────────────────

    async def _run_agent(self, input: list[Message], run_kwargs: _AgentRunKwargs) -> AgentResponse[Any]:
        """Run agent.run() in a child task so interrupt() can cancel it.

        Using a child task isolates cancellation: the parent task (and its
        post-processing) continues normally even when the child is cancelled.
        """

        # ``.run()`` is called INSIDE the task so the OTel telemetry
        # ContextVars (e.g. ``INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS``) are
        # set and reset in the same asyncio context.  Since P5.5 the chrys
        # AgentTelemetryLayer confines the non-streaming vars to the returned
        # coroutine, so ``create_task(self._agent.run(...))`` would also be
        # safe — this shape is kept as the tidier pattern, not a correctness
        # requirement anymore.
        async def _call() -> AgentResponse[Any]:
            return await self._agent.run(input, stream=False, **run_kwargs)

        async with self._trajectory_run(run_kwargs, stream=False):
            self._current_agent_task = asyncio.create_task(_call())
            try:
                return await self._current_agent_task
            finally:
                self._current_agent_task = None

    async def _run_blocking(self, message: list[Message]) -> None:
        """Run locally with per-wire retry, or stored mode with outer retry."""
        service_side = self.service_session_storage_enabled
        run_kwargs = self._build_run_kwargs(None if service_side else self._build_wire_retry_policy())
        self._injection.begin_retry()
        try:
            if service_side:
                loop = StreamRetryLoop(
                    max_retries=self._effective_max_retries(),
                    backoff_schedule=self._BACKOFF_SCHEDULE,
                    is_retryable=_is_retryable,
                    snapshot_history=self._snapshot_history,
                    restore_history=self._restore_history,
                    publish_retry_attempt=self._publish_retry_attempt,
                    is_interrupted=lambda: self._interrupt.is_interrupted,
                    interruptible_sleep=self._interruptible_sleep,
                    clean_error_message=_clean_error_message,
                    after_restore=lambda: self._restore_service_retry_inputs(run_kwargs),
                    may_retry=lambda exc: self._may_retry_attempt(exc, run_kwargs),
                    retry_exemption=_validation_retry_exemption,
                )
                result = await loop.run(lambda: self._run_agent(message, run_kwargs))
            else:
                result = await self._run_agent(message, run_kwargs)
        finally:
            self._injection.end_retry()

        # Skip publishing if the user interrupted during the final LLM call.
        if not self._interrupt.is_interrupted:
            await self._publish_response_text(result)

    # ── streaming path ───────────────────────────────────────────────

    async def _run_streaming(self, message: list[Message]) -> None:
        """Run locally with kernel-owned wire retry, or stored mode outside."""
        service_side = self.service_session_storage_enabled
        run_kwargs = self._build_run_kwargs(None if service_side else self._build_wire_retry_policy())
        if service_side:
            result = await self._stream_with_retry(message, run_kwargs)
        else:
            self._injection.begin_retry()
            try:
                result = await self._stream_single_attempt(message, run_kwargs, watchdog=False)
            finally:
                self._injection.end_retry()

        if not self._interrupt.is_interrupted:
            await self._publish_response_text(result)

    async def _stream_with_retry(
        self,
        current_input: list[Message],
        run_kwargs: _AgentRunKwargs,
    ) -> AgentResponse[Any]:
        """Execute a streaming agent.run() with retry on transient errors.

        Delegates the generic retry/backoff/rollback machinery to
        :class:`StreamRetryLoop`.  On exhausted stall retries falls back
        to a non-streaming request for this turn.  ``StreamRetryLoop``
        captures its own history snapshot on entry, so callers no longer
        pre-snapshot.
        """
        loop = StreamRetryLoop(
            max_retries=self._effective_max_retries(),
            backoff_schedule=self._BACKOFF_SCHEDULE,
            is_retryable=_is_retryable,
            snapshot_history=self._snapshot_history,
            restore_history=self._restore_history,
            publish_retry_attempt=self._publish_retry_attempt,
            is_interrupted=lambda: self._interrupt.is_interrupted,
            interruptible_sleep=self._interruptible_sleep,
            clean_error_message=_clean_error_message,
            restore_on_stall_exhaustion=True,
            after_restore=lambda: self._restore_service_retry_inputs(run_kwargs),
            may_retry=lambda exc: self._may_retry_attempt(exc, run_kwargs),
            retry_exemption=_validation_retry_exemption,
        )

        async def _attempt() -> AgentResponse[Any]:
            return await self._stream_single_attempt(current_input, run_kwargs)

        self._injection.begin_retry()
        try:
            try:
                return await loop.run(_attempt)
            except StreamStallExhausted as exc:
                # The retry loop restored failed stream state before this
                # blocking replacement, including any consumed injection.
                if self._hosted_bridge is not None:
                    await self._hosted_bridge.attempt_rejected("Streaming stalled; using blocking fallback")
                await self._trajectory_run_retry_scheduled(exc=exc, delay_seconds=0, fallback_to_blocking=True)
                return await self._run_agent(current_input, run_kwargs)
        finally:
            self._injection.end_retry()

    async def _publish_retry_attempt(
        self,
        message: str,
        attempt: int,
        max_attempts: int,
        delay_seconds: int,
        exc: BaseException,
    ) -> None:
        """Publish a :class:`RetryAttempt` event for the main agent, and record the run retry.

        Exposed as a method (not a lambda) so :class:`StreamRetryLoop`
        can bind to it without capturing ``self`` in an awkward closure.
        """
        await self._trajectory_run_retry_scheduled(exc=exc, delay_seconds=delay_seconds)
        await self._publish_retry_notice(message, attempt, max_attempts, delay_seconds, exc)

    async def _publish_wire_retry_attempt(
        self,
        message: str,
        attempt: int,
        max_attempts: int,
        delay_seconds: int,
        exc: BaseException,
    ) -> None:
        """The wire retry policy's notifier: UI only.

        A wire retry is recorded by the loop that performs it, as a new
        exchange under the same run (``retry.scheduled{retry_mode: wire}``).
        Recording a run-level retry here as well would count one retry twice
        and leave a scheduled marker that no ``retry.started`` ever answers.
        """
        await self._publish_retry_notice(message, attempt, max_attempts, delay_seconds, exc)

    async def _publish_retry_notice(
        self,
        message: str,
        attempt: int,
        max_attempts: int,
        delay_seconds: int,
        exc: BaseException,
    ) -> None:
        if self._hosted_bridge is not None and self._pending_continuation_token is None:
            await self._hosted_bridge.attempt_rejected(message)
        await self._bus.publish(
            RetryAttempt(
                message=message,
                attempt=attempt,
                max_attempts=max_attempts,
                delay_seconds=delay_seconds,
                session_id=self._session_id,
                display_message=_RETRY_STREAM_STALLED.bind() if isinstance(exc, _StreamStall) else None,
            )
        )

    async def _stream_single_attempt(
        self,
        current_input: list[Message],
        run_kwargs: _AgentRunKwargs,
        *,
        watchdog: bool = True,
    ) -> AgentResponse[Any]:
        """Run one streaming attempt wrapped in a per-attempt timeout.

        Accumulates text from each ``AgentResponseUpdate`` in a buffer.
        Text belonging to an *intermediate* LLM response (one that also
        contains ``function_call`` content) is discarded — the existing
        intermediate-text handler (via the instrumented client's
        ``result_hook``) publishes it as a single non-streamed message.
        Two signals discard the buffer so ordering within a response
        (text-first vs function_call-first) does not matter:

        1. A ``function_call`` content appearing in any update.
        2. A change in ``IntermediateTextBuffer.batch_id`` between
           iterations — the ``result_hook`` increments ``batch_id`` at the
           end of every tool-calling LLM response, so observing a change
           means a previous response just ended with tools.

        Whatever remains in the buffer once the stream ends is the final
        LLM response's text.  It is emitted as per-line
        ``AgentMessage(is_final=False)`` events so the TUI reveals the
        answer progressively; the trailing (unterminated) line is emitted
        once more so it appears before ``_publish_response_text`` fires
        the matching ``is_final=True``.
        """
        import time as _time

        # Tracks the last point at which our per-chunk watchdog (re)armed
        # its ``wait_for``.  Updated after each successful ``__anext__()``
        # and after the stream is fully drained (before the finalize
        # ``wait_for``).  Used below to distinguish a genuine stall — our
        # timer actually elapsed — from a ``TimeoutError`` bubbling up
        # from inside the stream/transport (e.g. httpx or SDK internals),
        # which would otherwise be mis-labelled "Stream stalled" and
        # retried with the wrong error message.
        last_wait_start = _time.monotonic()

        async def _iterate_and_finalize() -> AgentResponse[Any]:
            nonlocal last_wait_start

            # ``agent.run(stream=True)`` is invoked INSIDE this task so the
            # OTel telemetry ContextVars (e.g.
            # ``INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS``) are set and
            # reset in the same asyncio context — the telemetry cleanup
            # hook fires when the stream is consumed, which happens here.
            # The streaming layer keeps the set-in-call / reset-in-finalizer
            # shape; same-task creation remains the correct pattern here.
            # Matches the blocking path (see ``_run_agent``).
            stream: ResponseStream[AgentResponseUpdate, AgentResponse[Any]] = self._agent.run(
                current_input, stream=True, **run_kwargs
            )

            try:
                buffer = ""
                bridge_owns_text = False
                ibuf = self._intermediate_buffer
                last_batch_id = ibuf.batch_id if ibuf is not None else 0

                # Per-chunk watchdog: each __anext__() gets the stall timeout
                # individually, so the timer resets whenever a chunk arrives.
                # This measures stream-idle time (matching httpx read-timeout
                # semantics) — tool executions between chunks don't count.
                #
                # When a ``function_call`` chunk is observed, the tool loop
                # synchronously invokes the tool inside the NEXT ``__anext__()``
                # call — so that await blocks for the entire tool duration.
                # Tools (especially sub-agents) can legitimately run for
                # minutes, which would trip a short stream-idle timeout.  To
                # avoid a false "Stream stalled" during tool execution we
                # suspend the watchdog until the post-tool chunk arrives.
                # User-initiated cancellation still works — it flows through
                # :meth:`interrupt`, which cancels ``_current_agent_task``.
                expecting_tool_result = False
                aiter = stream.__aiter__()
                while True:
                    try:
                        if not watchdog or expecting_tool_result:
                            update = await aiter.__anext__()
                        else:
                            update = await asyncio.wait_for(
                                aiter.__anext__(),
                                timeout=self._stream_attempt_timeout,
                            )
                    except StopAsyncIteration:
                        break
                    last_wait_start = _time.monotonic()

                    if is_retry_boundary_update(update):
                        buffer = ""
                        if self._hosted_bridge is not None and self._pending_continuation_token is None:
                            await self._hosted_bridge.attempt_rejected("Provider response attempt retried")
                        continue

                    # A batch_id bump means a previous LLM response ended with
                    # tool calls; anything we buffered from it was intermediate.
                    if ibuf is not None and ibuf.batch_id != last_batch_id:
                        buffer = ""
                        last_batch_id = ibuf.batch_id

                    text_chunk = ""
                    has_function_call = False
                    has_function_result = False
                    for content in update.contents or []:
                        if content.type == "text":
                            if content.text:
                                text_chunk += content.text
                        elif content.type == "function_call" and not content.informational_only:
                            has_function_call = True
                        elif content.type == "function_result":
                            has_function_result = True
                        if content.provider_hosted:
                            bridge_owns_text = True

                    if has_function_result:
                        expecting_tool_result = False
                    if text_chunk:
                        buffer += text_chunk
                        # Observing post-tool text means the tool has resolved
                        # and a new LLM response is streaming — re-arm the
                        # watchdog on subsequent idle.
                        expecting_tool_result = False
                    if has_function_call:
                        # This response has tool calls — its text is intermediate.
                        buffer = ""
                        # Suspend the watchdog until a post-tool chunk arrives.
                        # The next ``__anext__`` awaits synchronous tool
                        # invocation, which can legitimately
                        # exceed the stream-idle timeout (especially for
                        # sub-agents).  Intermediate empty chunks
                        # (``finish_reason``-only) won't re-arm the watchdog
                        # because they contain no text.
                        expecting_tool_result = True
                    if bridge_owns_text:
                        buffer = ""

                    await asyncio.sleep(0)  # yield to event loop between chunks

                # Safety net: if a batch boundary fired after our last buffer
                # mutation (e.g., stream ended on an intermediate response),
                # drop any trailing intermediate text too.
                if ibuf is not None and ibuf.batch_id != last_batch_id:
                    buffer = ""

                # Emit the final-response text progressively.  Each completed
                # line fires a streaming event; the final (possibly unterminated)
                # line is emitted once more so it appears before is_final=True.
                if (
                    buffer
                    and not bridge_owns_text
                    and not self._interrupt.is_interrupted
                    and self._requirement_phase != "initial_implementation"
                ):
                    lines = buffer.splitlines(keepends=True)
                    emitted = ""
                    for i, line in enumerate(lines):
                        emitted += line
                        is_last = i == len(lines) - 1
                        if (line and line[-1] in ("\n", "\r")) or is_last:
                            await self._bus.publish(
                                AgentMessage(
                                    text=emitted,
                                    is_final=False,
                                    workflow_phase=self._requirement_phase,
                                    session_id=self._session_id,
                                )
                            )

                # Also guard finalize with the same per-chunk timeout — it
                # shouldn't block once the iterator exhausts, but it's still
                # an await on the same underlying stream.
                last_wait_start = _time.monotonic()
                if not watchdog:
                    return await stream.get_final_response()
                return await asyncio.wait_for(stream.get_final_response(), timeout=self._stream_attempt_timeout)
            except TimeoutError:
                if not watchdog:
                    raise
                # ``asyncio.TimeoutError`` is the builtin ``TimeoutError`` on
                # Python 3.11+, so a ``TimeoutError`` raised from *inside*
                # ``__anext__()`` / ``get_final_response()`` (httpx or SDK
                # internals) would look identical to our own ``wait_for``
                # firing.  Only treat it as a stall when the current watchdog
                # window actually elapsed (with a small margin for scheduling
                # jitter); otherwise re-raise so ``_is_retryable`` classifies
                # the original exception with its real message.
                #
                # Handled inside the task (rather than around the awaited
                # task) so ``_run_cleanup_hooks`` — which invokes OTel
                # ContextVar reset hooks — runs in the same asyncio context
                # as ``.run(stream=True)``.
                idle = _time.monotonic() - last_wait_start
                if idle + 0.5 < self._stream_attempt_timeout:
                    raise
                try:
                    await stream.aclose()
                except Exception:
                    logger.debug("Failed to close stalled service ResponseStream", exc_info=True)
                raise _StreamStall(0) from None

        # Wrap the iteration in a child task so ``interrupt()`` can cancel
        # it.  Without this, the streaming path has no task handle to
        # cancel and user-initiated interrupts cannot stop an in-flight
        # stream/tool — including long-running sub-agents.  (The blocking
        # path already does this in :meth:`_run_agent`.)
        async with self._trajectory_run(run_kwargs, stream=True):
            self._current_agent_task = asyncio.create_task(_iterate_and_finalize())
            try:
                return await self._current_agent_task
            finally:
                self._current_agent_task = None

    async def _publish_response_text(self, result: AgentResponse[Any]) -> None:
        """Publish the final text from a completed AgentResponse.

        Only publishes the *last* message — intermediate text is handled
        by ``IntermediateTextBuffer`` + ``ToolEventMiddleware``.

        The final event is also the frontend's turn-complete signal.  It
        must be emitted even when the final text is empty, for example
        after response validation exhausts retries and drops an empty /
        whitespace-only assistant message.
        """
        if self._response_has_hosted_contents(result) and self._hosted_bridge is not None:
            await self._hosted_bridge.reconcile_accepted(result.messages, final=True)
            return
        final_text = self._extract_final_text(result)
        self._last_response_text = final_text
        event_kwargs = self._assistant_message_event_kwargs()
        provisional = self._requirement_phase == "initial_implementation"
        await self._bus.publish(
            AgentMessage(
                text=final_text,
                is_final=not provisional,
                is_provisional=provisional,
                requirement_phase=self._requirement_phase,
                workflow_phase=self._requirement_phase,
                **event_kwargs,
                session_id=self._session_id,
            )
        )

    # ── control ──────────────────────────────────────────────────────

    async def interrupt(self) -> None:
        """Signal the executor to stop and cancel any running tool.

        Sets the interrupt flag for the middleware boundary check AND
        cancels the current ``agent.run()`` child task so long-running
        tools (sub-agents, scripts) are actually terminated instead of
        completing in the background.
        """
        self._interrupt.set_interrupted()
        if self._hosted_bridge is not None:
            await self._hosted_bridge.attempt_rejected(
                "Execution interrupted",
                status=HostedToolStatus.INTERRUPTED,
                preserve_provisional=True,
            )
        if self._current_agent_task is not None and not self._current_agent_task.done():
            sleep_call_ids = self._sleep.active_call_ids
            if sleep_call_ids:
                await self._interrupt_active_sleep(set(sleep_call_ids))
            if not self._current_agent_task.done():
                self._current_agent_task.cancel()

    async def _interrupt_active_sleep(self, call_ids: set[str]) -> None:
        """Let an active sleep publish its interrupted ToolCallResult before cancellation."""
        if not call_ids:
            return
        pending = set(call_ids)
        loop = asyncio.get_running_loop()
        completed: asyncio.Future[None] = loop.create_future()

        async def _on_tool_result(event: ToolCallResult) -> None:
            pending.discard(event.call_id)
            if not pending and not completed.done():
                completed.set_result(None)

        await self._bus.subscribe(ToolCallResult, _on_tool_result)
        try:
            interrupted = set(self._sleep.interrupt_active())
            pending.intersection_update(interrupted)
            if not pending:
                return
            # Keep Esc responsive: give the interrupted sleep a small
            # writeback window, then let task cancellation win.
            await asyncio.wait_for(completed, timeout=0.5)
        except TimeoutError:
            return
        finally:
            await self._bus.unsubscribe(ToolCallResult, _on_tool_result)

    async def close(self) -> None:
        """Release middleware resources owned by this executor."""
        await self._approval.close()

    async def _interruptible_sleep(self, seconds: int) -> bool:
        """Sleep in 1-second ticks, returning True if interrupted."""
        for _ in range(seconds):
            if self._interrupt.is_interrupted:
                return True
            await asyncio.sleep(1)
        return self._interrupt.is_interrupted

    def inject(
        self,
        text: str,
        created_at: datetime | str | None = None,
        injection_id: str | None = None,
        reminders: tuple[str, ...] = (),
        preparation: PreparationTrace | None = None,
        target_turn_id: str | None = None,
    ) -> None:
        """Queue text for injection before the next model call."""
        self._injection.queue(
            text,
            created_at=created_at,
            injection_id=injection_id,
            reminders=reminders,
            preparation=preparation,
            target_turn_id=target_turn_id,
        )

    def cancel_injection(self, injection_id: str) -> QueuedInjection | None:
        """Remove a still-pending injection; returns it, or None when too late."""
        return self._injection.cancel(injection_id)

    def reset_counters(self, *, reset_batch_id: bool = True) -> None:
        """Reset per-run counters (called by engine before each run).

        Args:
            reset_batch_id: If True, reset batch_id to 0 (for fresh runs).
                On resume, pass False to continue numbering from the
                interrupted run so batch_ids don't collide.
        """
        self._approval.reset()
        self._tool_events.reset_invocation_order()
        if reset_batch_id and self._intermediate_buffer is not None:
            self._intermediate_buffer.batch_id = 0

    def _get_history_messages(self) -> list:
        """Return the message list from the history provider state."""
        history_state = self._session.state.get("chrys_history", {})
        return history_state.get("messages", [])

    def _snapshot_history(self) -> _HistorySnapshot:
        """Snapshot messages and compressed block count for rollback.

        Used before an ``agent.run()`` that may need retry. Captures
        ``compressed_msgs`` length alongside messages so ``_restore_history``
        can undo cross-turn compressions (``_compress_state``) that run
        mid-loop, and each message's ``additional_properties`` so in-place
        annotations from the rolled-back attempt (exclusion flags,
        summarized-by markers) can be reverted exactly.
        """
        history_state = self._session.state.get("chrys_history", {})
        messages = list(history_state.get("messages", []))
        return _HistorySnapshot(
            messages=messages,
            compressed_count=len(history_state.get("compressed_msgs", [])),
            service_session_id=self.service_session_id,
            message_properties=snapshot_message_properties(messages),
            pre_output_history_len=history_state.get(PRE_OUTPUT_HISTORY_LEN_STATE_KEY),
            caller_state=_ExecutorRetryState(
                loop_recorder=self._loop_recorder.snapshot() if self._loop_recorder is not None else None,
                tool_events=self._tool_events.snapshot_retry_state(),
                approval=self._approval.snapshot_retry_state(),
                compaction=(
                    self._compaction_strategy.snapshot_retry_state() if self._compaction_strategy is not None else None
                ),
            ),
        )

    # ── helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _response_has_hosted_contents(result: AgentResponse[Any]) -> bool:
        """Return whether the accepted response contains provider-hosted content."""
        return any(content.provider_hosted for message in result.messages for content in message.contents)

    @staticmethod
    def _extract_final_text(result: AgentResponse[Any]) -> str:
        """Extract only the *last* message's text from an AgentResponse.

        ``AgentResponse.text`` concatenates ALL messages from the tool loop.
        This method returns only the final LLM output after all tool calls.
        """
        if not result.messages:
            return ""
        last_msg = result.messages[-1]
        return "".join(c.text for c in last_msg.contents if c.type == "text" and c.text)

    def _assistant_message_event_kwargs(self) -> _AssistantMessageEventKwargs:
        """Return event kwargs matching the timestamp persisted for assistant output."""
        parsed = parse_created_at(self._session.state.get(LAST_ASSISTANT_CREATED_AT_STATE_KEY))
        return {"timestamp": parsed} if parsed is not None else {}

    def _restore_history(self, snapshot: _HistorySnapshot) -> None:
        """Restore history from a snapshot, including compressed block rollback.

        Undoes ``_compress_state`` list replacements, removes orphaned
        ``CompressedBlock`` entries added during the rolled-back run, and
        restores each message's ``additional_properties`` exactly as captured
        — clearing the rolled-back attempt's compaction marks while keeping
        marks that legitimately predate the attempt.

        Used for retry rollback.
        """
        history_state = self._session.state.get("chrys_history", {})
        restored_messages = list(snapshot.messages)
        history_state["messages"] = restored_messages
        self.service_session_id = snapshot.service_session_id

        # Undo compressed block additions from the rolled-back run.
        compressed: list = history_state.get("compressed_msgs", [])
        if len(compressed) > snapshot.compressed_count:
            del compressed[snapshot.compressed_count :]

        if snapshot.pre_output_history_len is None:
            history_state.pop(PRE_OUTPUT_HISTORY_LEN_STATE_KEY, None)
        else:
            history_state[PRE_OUTPUT_HISTORY_LEN_STATE_KEY] = snapshot.pre_output_history_len
        restore_message_properties(restored_messages, snapshot.message_properties)
        retry_state = snapshot.caller_state
        if not isinstance(retry_state, _ExecutorRetryState):
            raise TypeError("Executor retry snapshot has invalid caller state.")
        if self._loop_recorder is not None and retry_state.loop_recorder is not None:
            self._loop_recorder.restore(retry_state.loop_recorder)
        self._tool_events.restore_retry_state(retry_state.tool_events)
        self._approval.restore_retry_state(retry_state.approval)
        if self._compaction_strategy is not None and retry_state.compaction is not None:
            self._compaction_strategy.restore_retry_state(retry_state.compaction)
