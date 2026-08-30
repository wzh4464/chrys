# Copyright (c) 2026 Chrys. All rights reserved.

"""Event type definitions for frontend ↔ backend communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from chrys.foundation.i18n import MessageRef
from chrys.foundation.models.todos import TodoItem

if TYPE_CHECKING:
    # Typing-only import: avoids a runtime dependency from foundation events
    # into service mutations while still letting
    # :class:`RollbackResult.restore_results` advertise
    # the real dataclass it carries.  ``from __future__ import annotations``
    # (top of file) keeps all annotations as strings, so this import never
    # executes at runtime.
    from chrys.service.mutations.types import RestoreResult

AGENT_LOAD_PHASE_MODEL = "model"
AGENT_LOAD_PHASE_RUNTIME = "runtime"
AGENT_LOAD_PHASE_SESSION = "session"
AGENT_LOAD_PHASE_TOOLS = "tools"
AGENT_LOAD_PHASE_SUB_AGENTS = "sub_agents"
AGENT_LOAD_PHASE_MCP = "mcp"
AGENT_LOAD_PHASE_SKILLS = "skills"
AGENT_LOAD_PHASE_AGENT = "agent"

AGENT_LOAD_STATUS_RUNNING = "running"
AGENT_LOAD_STATUS_DONE = "done"
AGENT_LOAD_STATUS_FAILED = "failed"
AGENT_LOAD_STATUS_SKIPPED = "skipped"

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass
class Event:
    """Base class for all events."""

    event_id: str = field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    session_id: str | None = None


# ---------------------------------------------------------------------------
# Frontend → Backend (user actions)
# ---------------------------------------------------------------------------


@dataclass
class UserMessage(Event):
    """User sends a message to the agent."""

    text: str = ""
    # Filled by the backend after prompt hooks, image validation, and image
    # compression succeed. TUI rendering uses this to preview exactly the
    # multimodal content accepted for the model call.
    prepared_contents: list[Any] | None = field(default=None, repr=False, compare=False)
    # Set by frontends that submit while a run is active so the resulting
    # mid-run injection can be correlated and cancelled (``UserInjectCancel``)
    # before the model sees it. ``None`` for ordinary fresh-turn submits.
    injection_id: str | None = None


@dataclass
class UserInterrupt(Event):
    """User interrupts the current agent execution."""


@dataclass
class UserRetry(Event):
    """User retries the last failed/interrupted run from current state.

    When ``text`` is non-empty, it is used as a mid-turn continuation prompt
    and persisted as real user input inside the current turn. An empty retry
    continues from completed history or replays the prior user opener.
    """

    text: str = ""


@dataclass
class UserInject(Event):
    """User injects a prompt while agent is executing (inserted before next model call)."""

    text: str = ""
    injection_id: str | None = None
    """Frontend-assigned id correlating this injection with cancel/result events."""


@dataclass
class UserInjectCancel(Event):
    """User cancels a queued mid-run injection before the model sees it.

    Effective only while the injection identified by ``injection_id`` is
    still pending (queued or in admission); once consumed, the cancel is a
    no-op and a ``consumed=True`` :class:`UserInjectResult` follows.
    """

    injection_id: str = ""


@dataclass
class UserRollback(Event):
    """User requests rollback to keep the first ``target_turn`` turns.

    ``target_turn = 0`` means "roll back to the empty pre-session
    state" — all conversation history is discarded and (optionally)
    every file mutation is reverted.  ``target_turn = N >= 1`` means
    "keep turns 1..N and discard everything after"; any file
    mutations introduced by the discarded turns are (optionally)
    reverted.
    """

    target_turn: int = 0
    """Number of turns to keep (0 = session start, N = keep turns 1..N)."""

    relative_turns: int | None = None
    """When set, discard this many latest turns under the backend transition fence.

    This is the authoritative selector for relative commands such as
    ``/rollback 1``; ``target_turn`` is ignored in that case.  Deferring the
    calculation prevents a frontend from publishing a stale absolute target
    when another turn finalizes while the request is being admitted.
    """

    expected_current_turn: int | None = None
    """Picker projection turn that must still be current when rollback is fenced.

    Direct commands leave this unset because relative targets are resolved under
    the backend fence and absolute targets are explicit.  Picker confirmations
    set it so file selections projected from an older conversation cannot be
    applied after another turn completes while the modal remains open.
    """

    expected_conversation_revision: int | None = None
    """Fresh/retry lifecycle revision that must still match the picker projection.

    Unlike ``expected_current_turn``, this advances for a retry or continuation
    that changes history and mutation planning without opening a fresh turn.
    """

    expected_build_generation: int | None = None
    """Runtime build generation that produced the picker preview.

    Workspace and settings rebuilds may replace mutation coordination without
    advancing the conversation. Picker confirmations set this so concrete file
    selections cannot be applied against a different runtime build.
    """

    expected_workspace_cwd: str | None = None
    """Normalized primary workspace cwd that produced the picker preview."""

    revert_changes: bool = False
    """When True, also restore file-system changes from the discarded turns."""

    selected_paths: list[str] | None = None
    """Paths to include when ``revert_changes`` is True.

    Either ``None`` ("revert every path in the plan") or a non-empty
    list restricting the revert to those paths.  The rollback modal
    sends a concrete list when the user leaves files checked, and
    ``None`` when ``revert_changes`` is False.  ``[]`` is not a supported
    input — the modal collapses the "no files checked" case into
    ``revert_changes=False, selected_paths=None``; any empty list arriving
    from a server-side caller is treated as ``None`` by the engine.
    Ignored when ``revert_changes`` is False.
    """


@dataclass
class ApprovalResponse(Event):
    """User responds to an approval request."""

    request_id: str = ""
    approved: bool = False
    reason: str = ""
    modified_args: dict[str, Any] | None = None


@dataclass
class ApprovalCancelled(Event):
    """A pending approval request was abandoned by its backend owner."""

    request_id: str = ""


@dataclass
class ApprovalAutoFulfillBlocked(Event):
    """Frontend tells the backend not to auto-fulfil an approved judge verdict."""

    request_id: str = ""


@dataclass
class AskUserResponse(Event):
    """User responds to an ask_user question."""

    request_id: str = ""
    text: str = ""


@dataclass
class SleepSkip(Event):
    """User skips a running sleep tool call."""

    call_id: str = ""


@dataclass
class AgentProfileSwitch(Event):
    """User switches to a different agent profile."""

    profile_name: str = ""


@dataclass
class WorkspaceChange(Event):
    """User requests a workspace/cwd change."""

    primary_cwd: str = ""


@dataclass
class ConfigUpdate(Event):
    """User updates a configuration value."""

    key: str = ""
    value: Any = None


@dataclass
class SettingsReload(Event):
    """User updated .env settings — engine should reload Settings and rebuild agent."""


@dataclass
class SetApprovalMode(Event):
    """User changes the approval mode (manual/auto/bypass).

    The engine updates ``ApprovalMiddleware`` and echoes ``ApprovalModeUpdated``
    so the TUI can refresh the badge from the authoritative backend state.
    ``persist`` controls whether the mode is also written as the global default;
    ACP standard session-mode changes are session-scoped and set this false.
    """

    mode: str = ""  # "manual" | "auto" | "bypass"
    persist: bool = True


@dataclass
class SetModelProfile(Event):
    """User switches the active model profile for this session only.

    The engine swaps ``settings.model_profile`` in-memory and soft-restarts the
    agent — no global ``.env`` write — so each session can run a different model
    without colliding on process-wide environment state.  The engine echoes
    ``ModelProfileSwitched`` once the rebuild completes.
    """

    profile_id: str = ""


# ---------------------------------------------------------------------------
# Backend → Frontend (system events)
# ---------------------------------------------------------------------------


@dataclass
class AgentThinking(Event):
    """Agent is thinking (optional thinking content)."""

    text: str | None = None


@dataclass(frozen=True)
class ProvisionalPresentation:
    """Identity of one retractable text segment from a response attempt."""

    attempt_id: str
    segment_id: str


@dataclass
class AgentMessage(Event):
    """Agent text output (supports streaming via is_final).

    Flags:
        is_final: ``True`` when this is the last message of the agent turn.
        is_intermediate: ``True`` for text returned alongside tool calls
            during the tool loop.  The agent is still running — the TUI
            should render the text but keep the running state active.
    """

    text: str = ""
    is_final: bool = True
    is_intermediate: bool = False
    structured_output_completed: bool = False
    presentation: ProvisionalPresentation | None = None


@dataclass
class PresentationAttemptAccepted(Event):
    """Commit provisional presentation segments from one accepted response attempt."""

    attempt_id: str = ""
    segment_ids: tuple[str, ...] = ()


@dataclass
class PresentationAttemptRejected(Event):
    """Retract provisional presentation segments from one rejected response attempt."""

    attempt_id: str = ""


@dataclass
class ToolCallStart(Event):
    """A tool call has started."""

    tool_name: str = ""
    tool_kind: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    provider_hosted: bool = False
    hosted_family: str = ""
    provider: str = ""
    provider_item_type: str = ""
    provider_call_id: str = ""
    provider_status: str = ""


@dataclass
class ToolCallProgress(Event):
    """Incremental output lines from a running shell tool."""

    tool_name: str = ""
    call_id: str = ""
    lines: list[str] = field(default_factory=list)
    image_contents: list[Any] = field(default_factory=list, repr=False, compare=False)
    snapshot_metadata: dict[str, Any] = field(default_factory=dict)
    provider_hosted: bool = False
    hosted_family: str = ""
    provider: str = ""
    provider_item_type: str = ""
    provider_call_id: str = ""
    provider_status: str = ""


@dataclass
class ToolCallArgsUpdated(Event):
    """A running tool call's arguments were rewritten after its start event."""

    tool_name: str = ""
    tool_kind: str = ""
    call_id: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    provider_hosted: bool = False
    hosted_family: str = ""
    provider: str = ""
    provider_item_type: str = ""
    provider_call_id: str = ""
    provider_status: str = ""


@dataclass
class ToolCallStatusUpdated(Event):
    """A running tool call's structured lifecycle status changed."""

    tool_name: str = ""
    call_id: str = ""
    status: str = ""
    provider_status: str = ""
    provider_hosted: bool = False
    hosted_family: str = ""
    provider: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallResult(Event):
    """A tool call has completed."""

    tool_name: str = ""
    call_id: str = ""
    result: str = ""
    image_contents: list[Any] = field(default_factory=list, repr=False, compare=False)
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    provider_hosted: bool = False
    hosted_family: str = ""
    provider: str = ""
    provider_item_type: str = ""
    provider_call_id: str = ""
    provider_status: str = ""


@dataclass
class SubAgentInvocationStart(Event):
    """A sub-agent invocation has started — binds invocation_id to parent call_id.

    Emitted by ``SubAgentTools._invoke`` immediately after generating the
    ``invocation_id``, before the first LLM call.  Carries both the parent
    tool call's ``call_id`` (generated by :class:`ToolEventMiddleware`) and
    the new ``invocation_id`` so the TUI can deterministically link the
    sub-agent widget that was mounted for the parent call to the sub-agent
    runtime, without relying on first-inner-tool-call arrival order.

    This matters when multiple sub-agents run in parallel: if one stalls
    in a retry backoff while its sibling's inner tool calls start arriving,
    a FIFO linker would assign the wrong widget and route retry banners /
    progress updates to the wrong card.
    """

    agent_name: str = ""
    invocation_id: str = ""
    tool_name: str = ""
    parent_call_id: str = ""


@dataclass
class SubAgentToolCallStart(Event):
    """A tool call started inside a sub-agent."""

    agent_name: str = ""
    invocation_id: str = ""
    tool_name: str = ""
    tool_kind: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""
    provider_hosted: bool = False
    hosted_family: str = ""
    provider: str = ""
    provider_item_type: str = ""
    provider_call_id: str = ""
    provider_status: str = ""


@dataclass
class SubAgentToolCallArgsUpdated(Event):
    """A sub-agent tool call's arguments changed after its start event."""

    agent_name: str = ""
    invocation_id: str = ""
    tool_name: str = ""
    tool_kind: str = ""
    call_id: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    provider_hosted: bool = False
    hosted_family: str = ""
    provider: str = ""
    provider_item_type: str = ""
    provider_call_id: str = ""
    provider_status: str = ""


@dataclass
class SubAgentToolCallStatusUpdated(Event):
    """A sub-agent tool call's structured lifecycle status changed."""

    agent_name: str = ""
    invocation_id: str = ""
    tool_name: str = ""
    call_id: str = ""
    status: str = ""
    provider_status: str = ""
    provider_hosted: bool = False
    hosted_family: str = ""
    provider: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubAgentToolCallProgress(Event):
    """Incremental text or structured progress from a sub-agent tool call."""

    agent_name: str = ""
    invocation_id: str = ""
    tool_name: str = ""
    call_id: str = ""
    lines: list[str] = field(default_factory=list)
    image_contents: list[Any] = field(default_factory=list, repr=False, compare=False)
    snapshot_metadata: dict[str, Any] = field(default_factory=dict)
    provider_hosted: bool = False
    hosted_family: str = ""
    provider: str = ""
    provider_item_type: str = ""
    provider_call_id: str = ""
    provider_status: str = ""


@dataclass
class SubAgentToolCallResult(Event):
    """A tool call completed inside a sub-agent."""

    agent_name: str = ""
    invocation_id: str = ""
    tool_name: str = ""
    call_id: str = ""
    result: str = ""
    image_contents: list[Any] = field(default_factory=list, repr=False, compare=False)
    duration_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    provider_hosted: bool = False
    hosted_family: str = ""
    provider: str = ""
    provider_item_type: str = ""
    provider_call_id: str = ""
    provider_status: str = ""


@dataclass
class SubAgentProgress(Event):
    """Cumulative progress update from a sub-agent invocation."""

    agent_name: str = ""
    invocation_id: str = ""
    tool_call_count: int = 0
    """Completed inner tool calls for this invocation."""

    total_tokens: int = 0
    """Current context-window tokens reported by the most recent sub-agent LLM call."""

    total_usage_tokens: int = 0
    """Cumulative tokens consumed by this sub-agent invocation across LLM calls."""

    usage_unreported_attempts: int = 0
    """ACP attempts whose terminal response did not report spend."""


@dataclass
class SubAgentCompactionStarted(Event):
    """Phase-4 LAST_WORDS compaction began inside a sub-agent invocation.

    Mirrors :class:`CompactionStarted` with ``invocation_id`` as the
    routing key so the TUI can show a live compaction line on the
    owning sub-agent card.
    """

    agent_name: str = ""
    invocation_id: str = ""
    compaction_id: str = ""
    phase: str = "phase4"


@dataclass
class SubAgentCompactionFinished(Event):
    """Phase-4 LAST_WORDS compaction finished inside a sub-agent invocation.

    ``outcome`` matches :class:`CompactionFinished`.  The note text is
    deliberately omitted — sub-agent cards only flip the status line — but
    ``format_violation`` preserves an accepted structured-note violation for
    observers, and ``failure_reason`` carries the safety-limit cause of a
    failed outcome (see :class:`CompactionFinished`).
    """

    agent_name: str = ""
    invocation_id: str = ""
    compaction_id: str = ""
    phase: str = "phase4"
    outcome: str = ""
    duration_ms: int = 0
    format_violation: str = ""
    failure_reason: str = ""


@dataclass
class SubAgentCompactionCommitted(Event):
    """A Phase-4 compaction round was durably applied inside a sub-agent.

    Follows the finished(``outcome="ok"``) signal for the same
    ``compaction_id`` once the strategy has committed the round — spill
    written (or checkpointed), note set, messages excluded.  A finished-ok
    round whose spill write failed never produces this event, so observers
    that count real compactions (the sub-agent card's "Compactions: N"
    subtitle) count this instead of finished-ok.
    """

    agent_name: str = ""
    invocation_id: str = ""
    compaction_id: str = ""
    phase: str = "phase4"


# --- Sub-agent lifecycle (pause / retry / abort) ---------------------------
#
# These events scope retry and abort to a single sub-agent invocation so
# that, when multiple sub-agents run concurrently, a retry click on one
# paused card does not disturb healthy siblings.  Every event carries
# ``invocation_id`` as its routing key.


@dataclass
class SubAgentRetryAttempt(Event):
    """A sub-agent's auto-retry loop is about to sleep before the next attempt.

    Emitted by the per-invocation retry loop (mirrors the parent's
    :class:`RetryAttempt`). Allows the TUI to render the retry banner
    inline on the specific sub-agent card.
    """

    agent_name: str = ""
    invocation_id: str = ""
    message: str = ""
    attempt: int = 0
    """1-based attempt number (the retry about to happen)."""
    max_attempts: int = 0
    delay_seconds: int = 0


@dataclass
class SubAgentPaused(Event):
    """A sub-agent exhausted auto-retry (or hit a non-retryable error) and is
    waiting for the user to decide via Retry or Abort.

    The parent's ``_invoke()`` is still awaiting — the parent tool call
    will only resolve once the user picks a decision (or a global
    interrupt fires).
    """

    agent_name: str = ""
    invocation_id: str = ""
    tool_name: str = ""
    reason: str = ""
    """One of ``stream_stall``, ``last_words``, ``framework_exc``, ``acp_transport``."""
    last_error: str = ""
    retry_attempts: int = 0
    diagnostic_path: str | None = None
    """UI-only diagnostic file path; never interpolate into model-visible errors."""


@dataclass
class SubAgentRetryRequested(Event):
    """User clicked Retry on a paused sub-agent card (frontend → backend)."""

    invocation_id: str = ""


@dataclass
class SubAgentAbortRequested(Event):
    """User clicked Abort on a paused sub-agent card (frontend → backend).

    The controller resolves the pending decision to ``"abort"``; the tool
    call returns an ``Error:`` string so the parent agent sees a normal
    tool-failure result and can decide what to do next.
    """

    invocation_id: str = ""


@dataclass
class SubAgentResumed(Event):
    """The controller restarted the sub-agent after a user Retry (backend → frontend)."""

    agent_name: str = ""
    invocation_id: str = ""


@dataclass
class SubAgentCascadeAborted(Event):
    """A global :class:`UserInterrupt` tore down this paused/running sub-agent.

    Distinct from :class:`SubAgentPaused` with an abort action: here the
    user stopped the whole engine, not just this one sub-agent.
    """

    agent_name: str = ""
    invocation_id: str = ""


@dataclass
class SubAgentAborted(Event):
    """User clicked Abort on a paused sub-agent card (backend → frontend).

    Emitted by the controller after resolving its ``pending_decision``
    with the abort verdict, right before the tool call returns an
    ``Error:`` string to the parent.  Distinct from
    :class:`SubAgentCascadeAborted` — this one is scoped to a single
    invocation, the parent run keeps going.  The engine listens for this
    event to decrement its paused-invocation set and drive the FSM out
    of ``AWAITING_SUB_AGENTS``.
    """

    agent_name: str = ""
    invocation_id: str = ""
    last_error: str = ""


@dataclass
class ApprovalRequest(Event):
    """Engine requests user approval for a tool call."""

    request_id: str = ""
    call_id: str = ""
    caller_name: str = ""
    tool_name: str = ""
    tool_kind: str = ""
    presentation_kind: str = ""
    """Display-only kind hint for bridged remote (ACP) requests.

    Drives the approval dialog's human header ("Run command", …). Never
    consulted by approval policy or the judge — ``tool_kind`` stays ``""``
    for bridged requests precisely so kind-scoped rules cannot match them.
    """
    args: dict[str, Any] = field(default_factory=dict)
    intent_summary: str = ""
    user_message: str = ""
    workspace_roots: list[str] = field(default_factory=list)
    workspace_cwd: str = ""
    judging: bool = False
    """True when an LLM reviewer is concurrently evaluating this request.

    The TUI shows the "Evaluating" spinner on arrival and waits for an
    ``ApprovalReviewed`` event to update the dialog.
    """


@dataclass
class ApprovalReviewed(Event):
    """Automated reviewer has finished evaluating a pending ``ApprovalRequest``.

    Published only for AUTO-mode requests.  An approved outcome normally
    causes the backend to auto-fulfil the approval after the TUI receives this
    event, unless the frontend publishes ``ApprovalAutoFulfillBlocked`` because
    a user decision is already in flight.  A flagged outcome carries the
    concern so the user can decide.
    """

    request_id: str = ""
    approved: bool = False
    reason: str = ""


@dataclass
class ApprovalModeUpdated(Event):
    """Backend confirms the current approval mode (after a ``SetApprovalMode``
    or after session start).  The TUI uses this as the authoritative source
    for the header badge.
    """

    mode: str = ""  # "manual" | "auto" | "bypass"


@dataclass
class QuestionToUser(Event):
    """Agent asks the user a question."""

    question: str = ""
    options: list[str] = field(default_factory=list)
    request_id: str = ""
    call_id: str = ""
    caller_name: str = ""


@dataclass
class AskUserTimedOut(Event):
    """An ask_user question expired before the user responded."""

    request_id: str = ""


@dataclass
class ToolCompacted(Event):
    """Intra-turn tool compaction compressed old tool-call groups."""

    compacted_groups: int = 0
    """Number of tool-call groups that were compacted."""

    phase: str = ""
    """Which phase: ``"phase1"`` to ``"phase4"`` (see compaction module docstring)."""

    turn_numbers: list[int] = field(default_factory=list)
    """1-based turn numbers affected (phase 1 only)."""

    compacted_tool_names: list[str] = field(default_factory=list)
    """Tool names from compacted groups (for debugging)."""

    tokens_before: int = 0
    """Estimated token count before compaction."""

    tokens_after: int = 0
    """Estimated token count after compaction."""

    last_words_generated: bool = False
    """Phase 4 only: True when a LAST_WORDS progress note was produced/updated."""


@dataclass
class CompactionStarted(Event):
    """Phase-4 LAST_WORDS compaction began on the main agent.

    Published immediately before the summarization LLM call so frontends
    can surface a live "Compacting conversation…" indicator during the
    otherwise-silent wait (seconds to minutes with retry backoff).
    ``compaction_id`` correlates with :class:`CompactionFinished`.
    """

    compaction_id: str = ""
    phase: str = "phase4"


@dataclass
class CompactionFinished(Event):
    """Phase-4 LAST_WORDS compaction finished on the main agent.

    ``outcome`` is ``"ok"`` (note generated), ``"failed"`` (generation
    exhausted its retry budget), or ``"canceled"`` (the run was
    interrupted mid-generation).  ``last_words`` carries the generated
    note on success so frontends can render it without re-reading backend
    state. ``format_violation`` records a structured-note violation accepted
    when the bounded corrective retry process stops.  ``failure_reason`` is
    a short human-readable cause set on failed outcomes caused by a known
    safety limit (per-turn round limit, side-call spend budget); frontends
    show it on the failure card in place of the duration.
    """

    compaction_id: str = ""
    phase: str = "phase4"
    outcome: str = ""
    duration_ms: int = 0
    last_words: str = ""
    format_violation: str = ""
    failure_reason: str = ""


@dataclass
class ContextPressure(Event):
    """Phase 4 disabled after its attempt/progress/spend breaker tripped."""

    reason: str = ""
    attempts: int = 0
    side_call_tokens: int = 0
    side_call_token_budget: int = 0
    source: str = "main"
    invocation_id: str = ""


@dataclass
class ContextCompressed(Event):
    """Context was compressed (folded) at a turn marker."""

    compressed_context_id: str = ""
    summary: str = ""
    freed_messages: int = 0
    turn_range: tuple[int, int] = (0, 0)
    source: str = "agent"
    """What initiated: ``"agent"`` (LLM) or ``"auto"`` (platform)."""


@dataclass
class UsageUpdate(Event):
    """Token usage update."""

    agent_profile: str = ""
    """Agent profile that produced this usage update."""

    usage_source_id: str = ""
    """Stable source id for the agent run that produced this update.

    Main-agent usage uses the session id; sub-agent usage uses the sub-agent
    invocation id.  Profile names are display metadata and are not unique
    enough to distinguish main-agent and sub-agent windows.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    pct: float = 0.0
    max_context_tokens: int = 0
    total_session_tokens: int = 0
    """Cumulative tokens consumed across all LLM calls in this session."""

    total_session_input_tokens: int = 0
    """Cumulative input tokens consumed across all LLM calls in this session."""

    total_session_output_tokens: int = 0
    """Cumulative output tokens consumed across all LLM calls in this session."""

    cache_hit_tokens: int | None = None
    """Cache-read tokens reported by the provider for the last LLM call, or
    ``None`` if no provider in this session has reported a cache field."""

    total_session_cache_hit_tokens: int | None = None
    """Cumulative cache-read tokens across the session, or ``None`` until the
    first cache-aware response."""

    local_tokens: int = 0
    """Local tokenizer estimate of included conversation tokens (excludes system prompt/tools)."""

    calibration_ratio: float = 1.0
    """Ratio between API token count and local estimate (> 1.0 = API sees more due to overhead)."""

    system_overhead_tokens: int = 0
    """Estimated fixed overhead (system prompt + tool definitions), not counted in local_tokens."""


@dataclass
class StateChanged(Event):
    """A session state value changed."""

    key: str = ""
    value: Any = None


@dataclass
class TodoListUpdated(Event):
    """The session todo list was replaced (full-list payload, not a diff)."""

    items: list[TodoItem] = field(default_factory=list)
    source: str = "main"
    """Origin of the update; future: sub-agent name / ``"user"``."""


@dataclass
class UserInjectResult(Event):
    """Reports whether a mid-run user message was delivered to the model.

    consumed=True:  The message was injected before the next model call.
                    The TUI should display it in the chat at this point.
    consumed=False: agent.run() ended before the message could be injected.
                    The TUI should keep the text in the input bar for re-use.

    ``created_at`` is the original user-message timestamp, not this result
    event's publish timestamp.
    """

    text: str = ""
    consumed: bool = False
    created_at: datetime | str | None = None
    injection_id: str | None = None
    """Id of the originating injection when it carried one; frontends use it
    to ignore results for injections they already cancelled or replaced."""


@dataclass
class RetryAttempt(Event):
    """Executor is retrying a transient error (published before each retry sleep)."""

    message: str = ""
    """Pre-rendered English retained byte-for-byte for compatibility consumers."""
    attempt: int = 0
    """1-based attempt number (the retry about to happen)."""
    max_attempts: int = 0
    delay_seconds: int = 0
    scope: str = ""
    """What is being retried: ``""`` for the main-agent stream, ``"compaction"``
    for a Phase-4 LAST_WORDS side call.  Frontends render compaction retries
    inside the live compaction card instead of a transcript-level error banner."""
    display_message: MessageRef | None = None
    """Locale-neutral display reference when this retry has migrated prose."""
    detail: str = ""
    """Raw untranslated diagnostic component separated from a fixed wrapper."""


@dataclass
class Error(Event):
    """An error occurred."""

    code: str = ""
    message: str = ""
    """Pre-rendered English retained byte-for-byte for compatibility consumers."""
    recoverable: bool = True
    display_message: MessageRef | None = None


@dataclass
class Warning(Event):
    """A non-fatal warning that should be surfaced to the user without disrupting workflow."""

    code: str = ""
    message: str = ""
    """Pre-rendered English retained byte-for-byte for compatibility consumers."""
    display_message: MessageRef | None = None


@dataclass
class RuntimeModelDetails:
    """Non-sensitive details about the active model profile."""

    profile_id: str = ""
    name: str = ""
    provider: str = ""
    api_style: str = ""
    model_id: str = ""
    max_context_tokens: int = 0
    base_url: str = ""
    stream: bool = False
    vision: bool = False
    selection_source: Literal["override", "agent", "inherited", "active", "default"] = "active"


@dataclass
class RuntimeSkillDetails:
    """Non-sensitive details about one loaded runtime skill."""

    name: str = ""
    description: str = ""
    source: str = ""


@dataclass
class RuntimeHookDetails:
    """Non-sensitive details about one configured runtime hook."""

    id: str = ""
    event: str = ""
    execution_mode: str = ""
    enabled: bool = True
    description: str = ""


@dataclass
class RuntimeHookSourceDetails:
    """One project or global source contributing runtime hooks."""

    scope: Literal["project", "global"] = "global"
    source_path: str = ""
    hooks: list[RuntimeHookDetails] = field(default_factory=list)


@dataclass
class AgentRuntimeDetails:
    """Grouped runtime metadata for the TUI details dialog."""

    model: RuntimeModelDetails = field(default_factory=RuntimeModelDetails)
    builtin_tools: dict[str, list[str]] = field(default_factory=dict)
    sub_agent_tools: list[str] = field(default_factory=list)
    mcp_tools: dict[str, list[str]] = field(default_factory=dict)
    mcp_failures: dict[str, str] = field(default_factory=dict)
    skill_sources: dict[str, list[str]] = field(default_factory=dict)
    skill_details: list[RuntimeSkillDetails] = field(default_factory=list)
    hook_sources: list[RuntimeHookSourceDetails] = field(default_factory=list)
    memory_sources: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class AgentLoadStarted(Event):
    """Agent infrastructure loading has started."""

    operation: str = ""
    from_profile: str = ""
    to_profile: str = ""
    from_display_name: str = ""
    to_display_name: str = ""


@dataclass
class AgentLoadProgress(Event):
    """Incremental progress while building agent tools/context."""

    phase: str = ""
    message: str = ""
    """Pre-rendered English retained byte-for-byte for compatibility consumers."""
    server_name: str = ""
    """MCP server identifier retained for compatibility."""
    current: int = 0
    total: int = 0
    failed: int = 0
    status: str = ""
    """Closed progress status from the ``AGENT_LOAD_STATUS_*`` vocabulary."""
    subject: str = ""
    """Generic per-item identifier, such as an MCP server or sub-agent profile."""
    detail: str = ""
    """Raw untranslated dynamic reason associated with this progress update."""


@dataclass
class AgentLoadFinished(Event):
    """Agent infrastructure loading has finished successfully."""

    operation: str = ""
    agent_profile: str = ""
    display_name: str = ""


@dataclass
class AgentLoadFailed(Event):
    """Agent infrastructure loading failed before the session became usable."""

    operation: str = ""
    agent_profile: str = ""
    display_name: str = ""
    message: str = ""
    """Pre-rendered English retained byte-for-byte for compatibility consumers."""
    display_message: MessageRef | None = None
    """Locale-neutral display reference for future migrated failure producers."""


@dataclass
class ImageAttachmentCompressionStarted(Event):
    """Image attachment compression has started before a user turn is sent."""

    image_count: int = 0


@dataclass
class ImageAttachmentCompressionFinished(Event):
    """Image attachment compression has finished before a user turn is sent."""

    image_count: int = 0


@dataclass
class SessionReady(Event):
    """Session is initialized and ready."""

    agent_profile: str = ""
    display_name: str = ""
    model_profile_id: str = ""
    max_context_tokens: int = 0
    tool_names: list[str] = field(default_factory=list)
    tool_kinds: dict[str, str] = field(default_factory=dict)
    skill_names: list[str] = field(default_factory=list)
    sub_agent_tool_names: list[str] = field(default_factory=list)
    memory_files: list[str] = field(default_factory=list)
    runtime_details: AgentRuntimeDetails = field(default_factory=AgentRuntimeDetails)
    primary_cwd: str = ""
    # All workspace roots (may include the primary cwd) so consumers such as
    # the workspace MRU can record secondary roots without an engine query.
    working_dirs: list[str] = field(default_factory=list)


@dataclass
class AgentRuntimeUpdated(Event):
    """Agent runtime metadata changed without a full agent rebuild.

    ``model_profile_id`` and ``max_context_tokens`` are intentionally
    denormalized from ``runtime_details.model`` for active model and context
    window tracking. ``runtime_details`` carries the complete confirmed
    metadata consumed by runtime confirmation paths.
    """

    model_profile_id: str = ""
    max_context_tokens: int = 0
    tool_names: list[str] = field(default_factory=list)
    skill_names: list[str] = field(default_factory=list)
    memory_files: list[str] = field(default_factory=list)
    runtime_details: AgentRuntimeDetails = field(default_factory=AgentRuntimeDetails)


# ---------------------------------------------------------------------------
# Workspace & history events
# ---------------------------------------------------------------------------


@dataclass
class SessionNew(Event):
    """User requests to start a new session (preserving the current one)."""


@dataclass
class SessionRestore(Event):
    """User requests to restore a saved session."""

    session_id: str = ""
    primary_cwd: str = ""
    profile_name: str = ""
    # Rollback restores have already swapped session.json and must ignore/delete
    # any live crash-recovery sidecar from the pre-rollback state.
    ignore_recovery: bool = False
    apply_saved_model: bool = False
    # Request-time additional working directories (paths), excluding the primary cwd.
    # ``None`` means the caller did not specify roots — keep the saved workspace. A
    # list (possibly empty) is authoritative and replaces the saved additional roots,
    # so ACP clients can narrow, swap, or clear scope on load.
    working_dirs: list[str] | None = None


@dataclass
class SessionDelete(Event):
    """User requests to delete a saved session."""

    session_id: str = ""


@dataclass
class SessionClear(Event):
    """User requests to delete the ACTIVE session and start a fresh one.

    One fenced backend transition: prompt admission is closed for the whole
    operation, the deletion is acknowledged by ``SessionDeleted`` before the
    fresh session starts, and a failed deletion keeps the current session
    intact and reports ``Error(code="session_clear_failed")`` — the fresh
    session is never started in that case.  ``session_id`` must be the
    engine's active session.
    """

    session_id: str = ""


@dataclass
class SessionRestored(Event):
    """A session has been successfully restored (backend → frontend)."""

    session_id: str = ""
    agent_profile: str = ""
    display_name: str = ""
    # Profile shown by the oldest restored messages. This avoids a frontend
    # scan of every saved session merely to recover one history field.
    initial_agent_profile: str = ""
    message_count: int = 0
    cwd_warning: str = ""
    primary_cwd: str = ""
    recovered_from_sidecar: bool = False
    # Restored workspace roots (may include the primary cwd) so consumers
    # such as the workspace MRU can record secondary roots without an
    # engine query.
    working_dirs: list[str] = field(default_factory=list)


@dataclass
class SessionSaved(Event):
    """A session has been auto-saved (backend → frontend, triggers history refresh)."""

    session_id: str = ""


@dataclass
class SessionTitleUpdated(Event):
    """A session's title overlay changed (backend → frontend).

    Published after a title patch lands in ``session.json``: either the
    user saved/cleared a custom title (``custom=True``, ``title`` may be
    empty to fall back to automatic titles) or the post-turn summarizer
    persisted a fresh auto-generated title (``custom=False``).

    ``display_title`` carries the post-update resolved title (custom >
    generated > first-message fallback) so protocol consumers that show a
    single title string (e.g. the ACP bridge) don't clear it while a
    fallback still exists; ``title`` stays the raw patched value for
    consumers that track the overlay fields themselves.
    """

    title: str = ""
    custom: bool = False
    display_title: str = ""


@dataclass
class SessionFork(Event):
    """User requests to fork the current session."""

    session_id: str = ""


@dataclass
class SessionForked(Event):
    """A session has been forked (backend → frontend)."""

    session_id: str = ""
    parent_session_id: str = ""
    new_session_id: str = ""


@dataclass
class RollbackResult(Event):
    """Rollback completed (backend → frontend).

    The TUI should clear its chat view and either replay the restored
    history (for ``target_turn >= 1``) or show the welcome state
    (for ``target_turn == 0``). Frontends may seed their input composer
    from ``rolled_back_user_text`` after the rollback UI refresh.
    """

    session_id: str = ""
    target_turn: int = 0
    """Number of turns kept after the rollback (0 = session start)."""

    rolled_back_user_text: str = ""
    """First user prompt from the discarded turn range, if available."""

    files_reverted: int = 0
    """Count of files that were actually changed on disk by the revert
    (sum of ``r.changed`` across :attr:`restore_results`).  Zero means
    either no revert was requested or every target was already at the
    expected content.  Use ``bool(restore_results)`` to distinguish
    "revert was attempted" from "revert was skipped"."""

    restore_results: list[RestoreResult] = field(default_factory=list)
    """Per-file :class:`chrys.service.mutations.types.RestoreResult`
    entries — one per path the rollback plan targeted.  The import is
    ``TYPE_CHECKING``-only so the events module stays runtime-free of
    a core dependency (see header); consumers import ``RestoreResult``
    directly and use ``r.changed``/``r.ok`` for per-path counts and
    ``r.reason`` for failure text.  The list is never heterogeneous —
    ``MutationTracker.rollback`` is the sole producer."""

    exclusions: list[tuple[str, str]] = field(default_factory=list)
    """``(path, reason)`` pairs for files the rollback plan dropped —
    primitive shapes only (``reason`` is a
    ``RollbackExclusionReason.value`` string like ``"unrestorable"`` /
    ``"move_poisoned"``); the enum lives in the service layer, which
    foundation must not import.  Populated from the pre-built
    ``RollbackPlan`` — the engine builds the plan before executing
    because the rolled-back turns (and with them the exclusions) are
    unreconstructable afterwards.  Empty when no file revert was
    requested."""

    warnings: list[str] = field(default_factory=list)
    """Advisory, repo-level notice strings attached to the rollback
    plan (e.g. another chrys session has an active command in this
    tree).  Per-path hazards are exclusions, never warnings."""


@dataclass
class SessionDeleted(Event):
    """A session has been deleted (backend → frontend)."""

    session_id: str = ""


@dataclass
class ProfileSwitched(Event):
    """A profile switch has completed with history preserved (backend → frontend)."""

    from_profile: str = ""
    to_profile: str = ""
    from_display_name: str = ""
    to_display_name: str = ""
    message_count: int = 0
    model_profile_id: str = ""
    max_context_tokens: int = 0
    tool_names: list[str] = field(default_factory=list)
    skill_names: list[str] = field(default_factory=list)
    sub_agent_tool_names: list[str] = field(default_factory=list)
    memory_files: list[str] = field(default_factory=list)
    runtime_details: AgentRuntimeDetails = field(default_factory=AgentRuntimeDetails)


@dataclass
class ModelProfileSwitched(Event):
    """A model profile switch has completed for this session (backend → frontend)."""

    model_profile_id: str = ""
    max_context_tokens: int = 0
    runtime_details: AgentRuntimeDetails = field(default_factory=AgentRuntimeDetails)


@dataclass
class WorkspaceUpdated(Event):
    """Session workspace has been updated mid-session (backend → frontend)."""

    primary_cwd: str = ""
    working_dirs: list[str] = field(default_factory=list)
    reference_files: list[str] = field(default_factory=list)


@dataclass
class SettingsReloaded(Event):
    """A settings reload (env + registries) has completed (backend → frontend).

    Echoed once the agent has been rebuilt so a caller awaiting the reload can
    distinguish success from an ``AgentLoadFailed`` rebuild failure.
    """

    runtime_details: AgentRuntimeDetails = field(default_factory=AgentRuntimeDetails)
