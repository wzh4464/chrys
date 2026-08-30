# Copyright (c) 2026 Chrys. All rights reserved.

"""EventBus subscription ownership for the main screen."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from chrys.foundation.events.types import (
    AgentLoadFailed,
    AgentLoadFinished,
    AgentLoadProgress,
    AgentLoadStarted,
    AgentMessage,
    AgentRuntimeUpdated,
    AgentThinking,
    ApprovalCancelled,
    ApprovalModeUpdated,
    ApprovalRequest,
    ApprovalReviewed,
    AskUserTimedOut,
    CompactionFinished,
    CompactionStarted,
    ContextCompressed,
    ContextPressure,
    Error,
    Event,
    ImageAttachmentCompressionFinished,
    ImageAttachmentCompressionStarted,
    PresentationAttemptAccepted,
    PresentationAttemptRejected,
    ProfileSwitched,
    QuestionToUser,
    RetryAttempt,
    RollbackResult,
    SessionForked,
    SessionReady,
    SessionRestored,
    SessionSaved,
    SessionTitleUpdated,
    SettingsReloaded,
    SubAgentAborted,
    SubAgentCascadeAborted,
    SubAgentCompactionCommitted,
    SubAgentCompactionFinished,
    SubAgentCompactionStarted,
    SubAgentInvocationStart,
    SubAgentPaused,
    SubAgentProgress,
    SubAgentResumed,
    SubAgentRetryAttempt,
    SubAgentToolCallArgsUpdated,
    SubAgentToolCallProgress,
    SubAgentToolCallResult,
    SubAgentToolCallStart,
    SubAgentToolCallStatusUpdated,
    TodoListUpdated,
    ToolCallArgsUpdated,
    ToolCallProgress,
    ToolCallResult,
    ToolCallStart,
    ToolCallStatusUpdated,
    ToolCompacted,
    UsageUpdate,
    UserInjectResult,
    Warning,
    WorkspaceUpdated,
)

if TYPE_CHECKING:
    from chrys.app.tui.screens.main.event_handlers import BackendEventHandler
    from chrys.app.tui.screens.main.session_handlers import SessionHandler
    from chrys.foundation.events.bus import EventBus


type EventHandler[E: Event] = Callable[[E], Awaitable[None]]


@dataclass(frozen=True)
class EventSubscription[E: Event]:
    """One EventBus subscription owned by the main screen."""

    event_type: type[E]
    handler: EventHandler[E]


def _sub[E: Event](event_type: type[E], handler: EventHandler[E]) -> EventSubscription[Any]:
    """Create a subscription while checking its event/handler pairing."""
    return EventSubscription(event_type, handler)


class MainScreenSubscriptions:
    """Subscribe and unsubscribe the main screen's backend event handlers."""

    def __init__(
        self,
        *,
        bus: EventBus,
        events: BackendEventHandler,
        sessions: SessionHandler,
        rollback_result_handler: EventHandler[RollbackResult],
    ) -> None:
        self._bus = bus
        self._subscriptions: tuple[EventSubscription[Any], ...] = (
            _sub(SessionReady, events.on_session_ready),
            _sub(AgentLoadStarted, events.on_agent_load_started),
            _sub(AgentLoadProgress, events.on_agent_load_progress),
            _sub(AgentLoadFinished, events.on_agent_load_finished),
            _sub(AgentLoadFailed, events.on_agent_load_failed),
            _sub(AgentRuntimeUpdated, events.on_agent_runtime_updated),
            _sub(ImageAttachmentCompressionStarted, events.on_image_attachment_compression_started),
            _sub(ImageAttachmentCompressionFinished, events.on_image_attachment_compression_finished),
            _sub(AgentThinking, events.on_agent_thinking),
            _sub(AgentMessage, events.on_agent_message),
            _sub(PresentationAttemptAccepted, events.on_presentation_attempt_accepted),
            _sub(PresentationAttemptRejected, events.on_presentation_attempt_rejected),
            _sub(ToolCallStart, events.on_tool_start),
            _sub(ToolCallArgsUpdated, events.on_tool_args_updated),
            _sub(ToolCallStatusUpdated, events.on_tool_status_updated),
            _sub(ToolCallProgress, events.on_tool_progress),
            _sub(ToolCallResult, events.on_tool_result),
            _sub(SubAgentInvocationStart, events.on_sub_agent_invocation_start),
            _sub(SubAgentToolCallStart, events.on_sub_agent_tool_start),
            _sub(SubAgentToolCallArgsUpdated, events.on_sub_agent_tool_args_updated),
            _sub(SubAgentToolCallStatusUpdated, events.on_sub_agent_tool_status_updated),
            _sub(SubAgentToolCallProgress, events.on_sub_agent_tool_progress),
            _sub(SubAgentToolCallResult, events.on_sub_agent_tool_result),
            _sub(SubAgentProgress, events.on_sub_agent_progress),
            _sub(SubAgentCompactionStarted, events.on_sub_agent_compaction_started),
            _sub(SubAgentCompactionFinished, events.on_sub_agent_compaction_finished),
            _sub(SubAgentCompactionCommitted, events.on_sub_agent_compaction_committed),
            _sub(SubAgentRetryAttempt, events.on_sub_agent_retry_attempt),
            _sub(SubAgentPaused, events.on_sub_agent_paused),
            _sub(SubAgentResumed, events.on_sub_agent_resumed_after_pause),
            _sub(SubAgentCascadeAborted, events.on_sub_agent_cascade_aborted),
            _sub(SubAgentAborted, events.on_sub_agent_aborted),
            _sub(ApprovalRequest, events.on_approval_request),
            _sub(ApprovalCancelled, events.on_approval_cancelled),
            _sub(ApprovalReviewed, events.on_approval_reviewed),
            _sub(ApprovalModeUpdated, events.on_approval_mode_updated),
            _sub(QuestionToUser, events.on_question_to_user),
            _sub(AskUserTimedOut, events.on_ask_user_timed_out),
            _sub(UsageUpdate, events.on_usage_update),
            _sub(TodoListUpdated, events.on_todo_list_updated),
            _sub(ToolCompacted, events.on_tool_compacted),
            _sub(CompactionStarted, events.on_compaction_started),
            _sub(CompactionFinished, events.on_compaction_finished),
            _sub(ContextCompressed, events.on_context_compressed),
            _sub(ContextPressure, events.on_context_pressure),
            _sub(UserInjectResult, events.on_injection_outcome),
            _sub(Error, events.on_error),
            _sub(RetryAttempt, events.on_retry_attempt),
            _sub(Warning, events.on_warning),
            _sub(SettingsReloaded, events.on_settings_reloaded),
            _sub(SessionRestored, sessions.on_session_restored),
            _sub(SessionSaved, sessions.on_session_saved),
            _sub(SessionTitleUpdated, sessions.on_session_title_updated),
            _sub(SessionForked, sessions.on_session_forked),
            _sub(ProfileSwitched, sessions.on_profile_switched),
            _sub(WorkspaceUpdated, sessions.on_workspace_updated),
            _sub(RollbackResult, rollback_result_handler),
        )
        self._subscribed = False

    @property
    def subscriptions(self) -> tuple[EventSubscription[Any], ...]:
        """Return the owned subscription table."""
        return self._subscriptions

    async def subscribe_all(self) -> None:
        """Register all handlers, once."""
        if self._subscribed:
            return
        for subscription in self._subscriptions:
            await self._bus.subscribe(subscription.event_type, subscription.handler)
        self._subscribed = True

    async def unsubscribe_all(self) -> None:
        """Remove all registered handlers, if currently subscribed."""
        if not self._subscribed:
            return
        for subscription in reversed(self._subscriptions):
            await self._bus.unsubscribe(subscription.event_type, subscription.handler)
        self._subscribed = False
