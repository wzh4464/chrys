# Copyright (c) 2026 Chrys. All rights reserved.

"""Headless session host for running Chrys agents outside the TUI."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from typing import Literal

from chrys.foundation.config.settings import Settings
from chrys.foundation.config.settings_store import LoadedSettings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    AgentLoadFailed,
    AgentLoadFinished,
    AgentLoadProgress,
    AgentLoadStarted,
    AgentMessage,
    AgentRuntimeUpdated,
    AgentThinking,
    ApprovalModeUpdated,
    ApprovalRequest,
    ApprovalReviewed,
    CompactionFinished,
    CompactionStarted,
    ContextCompressed,
    ContextPressure,
    Error,
    Event,
    PresentationAttemptAccepted,
    PresentationAttemptRejected,
    QuestionToUser,
    RequirementClarificationPhaseChanged,
    RetryAttempt,
    SessionRestore,
    SessionRestored,
    SessionSaved,
    StateChanged,
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
    UserInterrupt,
    UserMessage,
    Warning,
)
from chrys.foundation.i18n import DisplayBlock, DisplaySequence, MessageRef, msg
from chrys.foundation.models.workspace import Workspace
from chrys.foundation.util.session_ids import SESSION_SHORT_ID_LEN, session_short_id
from chrys.orchestration.engine.engine import AgentEngine
from chrys.service.approval.policy import ApprovalMode
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import AgentProfile, MCPServerConfig
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.state.store import JsonFileStateStore, StateStore

_AGENT_PROFILE_NOT_FOUND = msg(
    "session_host.agent_profile_not_found",
    fallback="Agent profile not found: {name}",
)
_AGENT_PROFILE_NOT_FOUND_WITH_AVAILABLE = msg(
    "session_host.agent_profile_not_found_with_available",
    fallback="Agent profile not found: {name}. Available profiles: {available}",
)
_SESSION_NOT_FOUND = msg(
    "session_host.session_not_found",
    fallback="Session not found: {session_id}",
)
_SESSION_NOT_FOUND_WITH_RECENT = msg(
    "session_host.session_not_found_with_recent",
    fallback="Session not found: {session_id}. Recent sessions: {recent}",
)
_SESSION_ID_AMBIGUOUS = msg(
    "session_host.session_id_ambiguous",
    fallback="Session id '{session_id}' is ambiguous.",
)
_NO_FINAL_RESPONSE = msg(
    "session_host.no_final_response",
    fallback="Agent run ended without a final response.",
)
_HEADLESS_INTERACTION_REQUIRED = msg(
    "session_host.headless_interaction_required",
    fallback="Agent requested user input in headless mode: {detail}",
    multiline=True,
)

_HEADLESS_RUN_EVENT_TYPES = (
    AgentLoadFailed,
    AgentLoadFinished,
    AgentLoadProgress,
    AgentLoadStarted,
    AgentMessage,
    AgentRuntimeUpdated,
    AgentThinking,
    ApprovalModeUpdated,
    ApprovalRequest,
    ApprovalReviewed,
    CompactionFinished,
    CompactionStarted,
    ContextCompressed,
    ContextPressure,
    Error,
    QuestionToUser,
    RequirementClarificationPhaseChanged,
    PresentationAttemptAccepted,
    PresentationAttemptRejected,
    RetryAttempt,
    SessionRestored,
    SessionSaved,
    StateChanged,
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
)


@dataclass(frozen=True)
class HeadlessRunResult:
    """Final response and events from one headless agent turn."""

    text: str
    session_id: str
    events: list[Event] = field(default_factory=list)


class HeadlessRunError(RuntimeError):
    """Raised when a headless agent turn ends with an engine error."""

    def __init__(self, event: Error, events: list[Event]) -> None:
        self.event = event
        self.events = events
        super().__init__(event.message or event.code or "Headless agent run failed")


@dataclass(frozen=True)
class EndTurn:
    """A turn completed normally."""

    kind: Literal["end_turn"] = "end_turn"
    final_text: str = ""


@dataclass(frozen=True)
class Cancelled:
    """A turn was cancelled by the caller."""

    kind: Literal["cancelled"] = "cancelled"
    reason: str = ""


@dataclass(frozen=True)
class Errored:
    """A turn ended with an engine error."""

    kind: Literal["error"] = "error"
    error: Error = field(default_factory=Error)


TurnOutcome = EndTurn | Cancelled | Errored


class AgentProfileNotFoundError(KeyError):
    """Raised when an agent profile selector can't be resolved against the registry."""

    def __init__(self, message: str, *, display_message: MessageRef | None = None) -> None:
        self.display_message = display_message
        super().__init__(message)


class SessionNotFoundError(KeyError):
    """Raised when a session id (full or short) doesn't match any stored session."""

    def __init__(self, message: str, *, display_message: MessageRef | None = None) -> None:
        self.display_message = display_message
        super().__init__(message)


class AmbiguousSessionIdError(ValueError):
    """Raised when a session-id prefix matches more than one stored session."""

    def __init__(self, message: str, *, display_message: MessageRef | None = None) -> None:
        self.display_message = display_message
        super().__init__(message)


def _normalize_cwd(cwd: str | None) -> str:
    raw = (cwd or "").strip()
    if not raw:
        return ""
    return os.path.abspath(os.path.expanduser(raw))


class ChrysSessionHost:
    """Owns one event bus, one engine, and one session for non-TUI callers.

    ``start()`` is an explicit, idempotent lifecycle boundary.  Callers that
    need startup/restore events should use ``iter_run_events()``, which
    auto-starts before publishing the user turn.  ``run_until_final()`` is only
    a convenience wrapper over that streaming primitive.
    """

    def __init__(
        self,
        *,
        profile_name: str = "",
        session_id: str | None = None,
        settings: Settings | None = None,
        loaded_settings: LoadedSettings | None = None,
        event_bus: EventBus | None = None,
        agent_registry: AgentProfileRegistry | None = None,
        model_registry: ModelProfileRegistry | None = None,
        state_store: StateStore | None = None,
        approval_mode: ApprovalMode = ApprovalMode.BYPASS,
        cwd: str | None = None,
        workspace: Workspace | None = None,
        mcp_overlay: list[MCPServerConfig] | None = None,
        allow_user_interaction: bool = False,
        on_successful_turn: Callable[[], None] | None = None,
        on_turn_started: Callable[[], None] | None = None,
    ) -> None:
        self._profile_name = profile_name.strip()
        self._restore_session_id = (session_id or "").strip() or None
        if not self._profile_name and not self._restore_session_id:
            error_message = "profile_name is required."
            raise ValueError(error_message)
        self._bus = event_bus or EventBus()
        self._agent_registry = agent_registry or AgentProfileRegistry()
        self._model_registry = model_registry or ModelProfileRegistry()
        self._state_store = state_store or JsonFileStateStore()
        self._approval_mode = approval_mode
        self._cwd = _normalize_cwd(cwd)
        self._workspace = workspace
        self._mcp_overlay = list(mcp_overlay or [])
        self._allow_user_interaction = allow_user_interaction
        self._started = False
        self._run_lock = asyncio.Lock()
        self._cancel_requested = False
        self._last_turn_outcome: TurnOutcome | None = None
        self._engine = AgentEngine(
            self._bus,
            settings,
            loaded_settings=loaded_settings,
            agent_registry=self._agent_registry,
            model_registry=self._model_registry,
            state_store=self._state_store,
            initial_approval_mode=approval_mode,
            mcp_overlay=self._mcp_overlay,
            initial_workspace=workspace or (Workspace.from_cwd(self._cwd) if self._cwd else None),
            on_successful_turn=on_successful_turn,
            on_turn_started=on_turn_started,
            allow_user_interaction=allow_user_interaction,
        )

    @property
    def event_bus(self) -> EventBus:
        """Event bus used by this host."""
        return self._bus

    @property
    def engine(self) -> AgentEngine:
        """Engine owned by this host."""
        return self._engine

    @property
    def _settings(self) -> Settings:
        """The engine's current settings, not a copy taken at construction.

        Held nowhere here on purpose: the engine owns the handle, and a second
        field would answer with whatever was true when this host was built —
        stale from the first reload or session restore onwards.
        """
        return self._engine.settings

    @property
    def session_id(self) -> str | None:
        """Current canonical session id, or ``None`` before start."""
        return self._engine.session_id

    @property
    def approval_mode(self) -> ApprovalMode:
        """Initial approval mode used when the engine was built."""
        return self._approval_mode

    @property
    def vision_enabled(self) -> bool:
        """Whether the currently built runtime model accepts image input."""
        return self._engine.runtime_details.model.vision

    @property
    def last_turn_outcome(self) -> TurnOutcome | None:
        """Structured outcome for the most recently completed streamed turn."""
        return self._last_turn_outcome

    async def start(self) -> None:
        """Start a fresh session or restore the configured existing session.

        Idempotent. Callers that need to observe startup/restore events through
        ``iter_run_events()`` should let the first run drive startup so the
        event stream is already subscribed before load events are published.
        """
        if self._started:
            return
        self._load_registries_if_needed()
        if self._restore_session_id:
            await self.restore_session(self._restore_session_id)
        else:
            if not self._profile_name:
                error_message = "profile_name is required to start a new session."
                raise ValueError(error_message)
            profile = self._resolve_profile(self._profile_name)
            self._profile_name = profile.name
            await self._engine.start(profile)
            self._started = True

    async def restore_session(self, session_id: str) -> None:
        """Restore a saved session into this host."""
        target_session_id = await self._resolve_restore_session_id(session_id)
        if not target_session_id:
            error_message = "session_id is required to restore a session."
            raise ValueError(error_message)
        self._load_registries_if_needed()
        if self._profile_name:
            self._profile_name = self._resolve_profile(self._profile_name).name
        await self._engine.prepare()
        # Promote a requested short id to the canonical id used by restore events and future idempotent starts.
        self._restore_session_id = target_session_id
        loop = asyncio.get_running_loop()
        restored: asyncio.Future[SessionRestored] = loop.create_future()

        async def _on_restored(event: SessionRestored) -> None:
            if event.session_id == target_session_id and not restored.done():
                restored.set_result(event)

        async def _on_error(event: Error) -> None:
            if event.session_id == target_session_id and not restored.done():
                restored.set_exception(HeadlessRunError(event, [event]))

        await self._bus.subscribe(SessionRestored, _on_restored)
        await self._bus.subscribe(Error, _on_error)
        try:
            await self._bus.publish(
                SessionRestore(
                    session_id=target_session_id,
                    primary_cwd=self._cwd,
                    profile_name=self._profile_name,
                    working_dirs=(
                        [d.path for d in self._workspace.working_dirs if not d.is_primary]
                        if self._workspace is not None
                        else None
                    ),
                ),
                raise_handler_errors=True,
            )
            await restored
        finally:
            await self._bus.unsubscribe(SessionRestored, _on_restored)
            await self._bus.unsubscribe(Error, _on_error)
            if restored.done() and not restored.cancelled():
                # A failing restore publishes its Error *and* re-raises out of
                # the publish above, so the future's copy of the failure goes
                # unawaited; retrieve it or asyncio logs it as never-retrieved.
                restored.exception()
        self._started = True

    async def iter_run_events(self, message: str | UserMessage) -> AsyncIterator[Event]:
        """Run one user turn and yield backend events until the engine run finishes."""
        async for event in self._iter_turn_events(message, raise_for_outcome=True):
            yield event

    async def iter_turn_events(self, message: str | UserMessage) -> AsyncIterator[Event]:
        """Run one user turn and yield events, recording a structured terminal outcome."""
        async for event in self._iter_turn_events(message, raise_for_outcome=False):
            yield event

    async def _iter_turn_events(self, message: str | UserMessage, *, raise_for_outcome: bool) -> AsyncIterator[Event]:
        if self._run_lock.locked():
            error_message = "Concurrent turns are not supported for a ChrysSessionHost."
            raise RuntimeError(error_message)
        async with self._run_lock:
            self._last_turn_outcome = None
            self._cancel_requested = False
            async for event in self._iter_run_events_locked(message, raise_for_outcome=raise_for_outcome):
                yield event

    async def cancel_current_turn(self) -> None:
        """Request cancellation for the active turn, if any."""
        self._cancel_requested = True
        await self._bus.publish(UserInterrupt(session_id=self.session_id))

    async def _iter_run_events_locked(
        self,
        message: str | UserMessage,
        *,
        raise_for_outcome: bool,
    ) -> AsyncIterator[Event]:
        """Run one user turn while the host run lock is held."""
        events: list[Event] = []
        final: AgentMessage | None = None
        error: Error | None = None
        question: QuestionToUser | None = None
        next_event_task: asyncio.Task[Event] | None = None
        start_task: asyncio.Task[None] | None = None
        run_task: asyncio.Task[None] | None = None

        def _record_event(event: Event) -> None:
            nonlocal final, error, question
            events.append(event)
            if isinstance(event, AgentMessage) and event.is_final and not event.is_intermediate:
                final = event
            elif isinstance(event, Error):
                error = event
            elif isinstance(event, QuestionToUser):
                question = event

        async with self._bus.stream(*_HEADLESS_RUN_EVENT_TYPES) as stream:
            ask_user_interrupted = False

            async def _publish_user_turn() -> None:
                nonlocal run_task
                user_event = self._coerce_user_message(message)
                await self._bus.publish(user_event)
                run_task = asyncio.create_task(self._engine.wait_for_run_task())

            async def _event_task_ready_after_checkpoint() -> bool:
                if next_event_task is not None and next_event_task.done():
                    return True
                # EventBus.publish queues stream events synchronously; this checkpoint lets those tasks win first.
                # Revisit if the bus later grows asynchronous fan-out.
                await asyncio.sleep(0)
                return next_event_task is not None and next_event_task.done()

            next_event_task = asyncio.create_task(stream.__anext__())
            if self._started:
                await _publish_user_turn()
            else:
                start_task = asyncio.create_task(self.start())

            try:
                while True:
                    wait_tasks: set[asyncio.Task[object]] = {next_event_task}
                    if start_task is not None:
                        wait_tasks.add(start_task)
                    if run_task is not None:
                        wait_tasks.add(run_task)
                    done, _pending = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

                    if next_event_task in done:
                        event = next_event_task.result()
                        next_event_task = asyncio.create_task(stream.__anext__())
                        if not self._event_belongs_to_session(event, self.session_id, self._restore_session_id):
                            continue
                        _record_event(event)
                        yield event
                        if (
                            isinstance(event, QuestionToUser)
                            and not self._allow_user_interaction
                            and not ask_user_interrupted
                        ):
                            ask_user_interrupted = True
                            await self._bus.publish(UserInterrupt(session_id=self.session_id))
                        continue

                    if start_task is not None and start_task in done:
                        if await _event_task_ready_after_checkpoint():
                            continue
                        await start_task
                        start_task = None
                        await _publish_user_turn()
                        continue

                    if run_task is not None and run_task in done:
                        if await _event_task_ready_after_checkpoint():
                            continue
                        await run_task
                        outcome = self._resolve_run_outcome(final=final, error=error, question=question)
                        self._last_turn_outcome = outcome
                        if raise_for_outcome:
                            self._raise_for_run_outcome(events, outcome=outcome)
                        return
            finally:
                if start_task is not None:
                    if start_task.done():
                        self._observe_task_exception(start_task)
                    else:
                        start_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await start_task
                if run_task is not None:
                    if run_task.done():
                        self._observe_task_exception(run_task)
                    else:
                        try:
                            self._cancel_requested = True
                            await self._bus.publish(UserInterrupt(session_id=self.session_id))
                            await asyncio.shield(run_task)
                        finally:
                            self._observe_or_defer_task_exception(run_task)
                if next_event_task is not None and not next_event_task.done():
                    next_event_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await next_event_task

    async def run_until_final(self, message: str | UserMessage, *, timeout: float | None = None) -> HeadlessRunResult:
        """Run one user turn and return the final assistant response."""

        async def _run() -> HeadlessRunResult:
            events: list[Event] = []
            final: AgentMessage | None = None
            async for event in self.iter_run_events(message):
                events.append(event)
                if isinstance(event, AgentMessage) and event.is_final and not event.is_intermediate:
                    final = event
            if final is None:
                error_message = "Agent run ended without a final response."
                raise RuntimeError(error_message)
            return HeadlessRunResult(text=final.text, session_id=self.session_id or "", events=events)

        if timeout is None:
            return await _run()
        return await asyncio.wait_for(_run(), timeout=timeout)

    async def shutdown(self) -> None:
        """Shutdown the owned engine and release session resources."""
        await self._engine.shutdown()
        self._started = False

    def _load_registries_if_needed(self) -> None:
        if not self._agent_registry.list_names():
            self._agent_registry.load_all()
        if not self._model_registry.list_ids():
            self._model_registry.load_all()

    def _resolve_profile(self, name: str) -> AgentProfile:
        profile = self._agent_registry.resolve_selector(name)
        if profile is not None:
            return profile
        available_names = sorted(self._agent_registry.list_names())
        available = ", ".join(available_names)
        error_message = f"Agent profile not found: {name}"
        if available_names:
            error_message = f"{error_message}. Available profiles: {available}"
            display_message = _AGENT_PROFILE_NOT_FOUND_WITH_AVAILABLE.bind(
                name=name,
                available=DisplaySequence(available_names),
            )
        else:
            display_message = _AGENT_PROFILE_NOT_FOUND.bind(name=name)
        raise AgentProfileNotFoundError(error_message, display_message=display_message)

    async def _resolve_restore_session_id(self, session_id: str) -> str:
        raw = (session_id or "").strip()
        if not raw:
            return ""
        sessions = await self._state_store.list_sessions()
        normalized = raw.replace("-", "").lower()
        if len(normalized) <= SESSION_SHORT_ID_LEN:
            matches = [meta.session_id for meta in sessions if session_short_id(meta.session_id).lower() == normalized]
        else:
            matches = [
                meta.session_id
                for meta in sessions
                if meta.session_id == raw or meta.session_id.replace("-", "").lower() == normalized
            ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            error_message = f"Session id '{raw}' is ambiguous."
            raise AmbiguousSessionIdError(
                error_message,
                display_message=_SESSION_ID_AMBIGUOUS.bind(session_id=raw),
            )
        recent_ids = [session_short_id(meta.session_id) for meta in sessions[:5]]
        recent = ", ".join(recent_ids)
        error_message = f"Session not found: {raw}"
        if recent_ids:
            error_message = f"{error_message}. Recent sessions: {recent}"
            display_message = _SESSION_NOT_FOUND_WITH_RECENT.bind(
                session_id=raw,
                recent=DisplaySequence(recent_ids),
            )
        else:
            display_message = _SESSION_NOT_FOUND.bind(session_id=raw)
        raise SessionNotFoundError(error_message, display_message=display_message)

    @staticmethod
    def _event_belongs_to_session(
        event: Event,
        session_id: str | None,
        restore_session_id: str | None = None,
    ) -> bool:
        if not event.session_id:
            # Headless streams ignore global/sessionless events so unrelated errors cannot terminate this run.
            return False
        return event.session_id == session_id or (
            restore_session_id is not None and event.session_id == restore_session_id
        )

    @staticmethod
    def _observe_task_exception(task: asyncio.Task[object]) -> None:
        """Mark a completed helper task's exception as retrieved."""
        if task.done() and not task.cancelled():
            task.exception()

    @classmethod
    def _observe_or_defer_task_exception(cls, task: asyncio.Task[object]) -> None:
        """Observe a helper task now, or once it finishes after caller cancellation."""
        if task.done():
            cls._observe_task_exception(task)
            return
        task.add_done_callback(cls._observe_task_exception)

    def _raise_for_run_outcome(
        self,
        events: list[Event],
        *,
        outcome: TurnOutcome,
    ) -> None:
        if isinstance(outcome, Errored):
            raise HeadlessRunError(outcome.error, events)
        if isinstance(outcome, Cancelled):
            error_message = outcome.reason or "Agent run was cancelled."
            raise RuntimeError(error_message)

    def _resolve_run_outcome(
        self,
        *,
        final: AgentMessage | None,
        error: Error | None,
        question: QuestionToUser | None,
    ) -> TurnOutcome:
        if question is not None and not self._allow_user_interaction:
            # Headless runs have no one to answer, so a question terminates the turn.
            # Interactive hosts (e.g. ACP) answer ask_user inline, so the run continues
            # to a real final/error outcome and the question must not fail the turn.
            return Errored(error=self._question_to_error(question))
        if self._cancel_requested:
            return Cancelled(reason="Agent run was cancelled.")
        if error is not None:
            return Errored(error=error)
        if final is not None:
            return EndTurn(final_text=final.text)
        return Errored(
            error=Error(
                code="no_final_response",
                message="Agent run ended without a final response.",
                display_message=_NO_FINAL_RESPONSE.bind(),
                session_id=self.session_id,
            )
        )

    def _coerce_user_message(self, message: str | UserMessage) -> UserMessage:
        if isinstance(message, UserMessage):
            return replace(message, session_id=self.session_id)
        return UserMessage(text=message, session_id=self.session_id)

    @staticmethod
    def _question_to_error(event: QuestionToUser) -> Error:
        detail = event.question.strip() or "The agent requested user input."
        return Error(
            code="headless_interaction_required",
            message=f"Agent requested user input in headless mode: {detail}",
            display_message=_HEADLESS_INTERACTION_REQUIRED.bind(detail=DisplayBlock(detail)),
            session_id=event.session_id,
        )
