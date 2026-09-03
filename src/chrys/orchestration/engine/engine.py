# Copyright (c) 2026 Chrys. All rights reserved.

"""AgentEngine — top-level orchestrator that ties together all core components.

The Engine is the single entry point for the backend. It:
1. Accepts an AgentProfile → configures an Agent with the right tools/prompts/middleware
2. Subscribes to frontend events (UserMessage, UserInterrupt, UserInject, etc.)
3. Delegates execution to Executor
4. Manages session lifecycle
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chrys.foundation.config.settings import Settings, persist_approval_mode
from chrys.foundation.config.settings_store import LoadedSettings, SettingsHandle
from chrys.foundation.events.types import (
    AgentProfileSwitch,
    AgentRuntimeDetails,
    Error,
    MemoryWritebackCompleted,
    ProfileSwitched,
    RouteOverride,
    SessionClear,
    SessionDelete,
    SessionFork,
    SessionForked,
    SessionNew,
    SessionRestore,
    SetApprovalMode,
    SetModelProfile,
    SettingsReload,
    SubAgentAborted,
    SubAgentAbortRequested,
    SubAgentCascadeAborted,
    SubAgentPaused,
    SubAgentResumed,
    SubAgentRetryRequested,
    UserInject,
    UserInjectCancel,
    UserInterrupt,
    UserMessage,
    UserRetry,
    UserRollback,
    WorkspaceChange,
)
from chrys.foundation.i18n import DisplayBlock, MessageRef, msg
from chrys.foundation.models.workspace import Workspace
from chrys.foundation.recovery import RecoveryPersistOutcome
from chrys.foundation.tool_kinds import KIND_SUB_AGENT
from chrys.foundation.trajectory.context import TrajectoryContext
from chrys.foundation.trajectory.event_types import RuntimeFinishReason as TrajectoryRuntimeFinishReason
from chrys.orchestration.engine import rollback as rollback_controller
from chrys.orchestration.engine import trajectory as trajectory_recorder
from chrys.orchestration.engine.build import construction as agent_lifecycle
from chrys.orchestration.engine.build.builder import build_agent
from chrys.orchestration.engine.memory_writeback import MemoryWritebackWatcher
from chrys.orchestration.engine.run import sub_agent_coordination
from chrys.orchestration.engine.run.coordinator import TurnCoordinator
from chrys.orchestration.engine.run.turn_state import (
    CurrentRunInjectionWindow,
    CurrentRunScope,
    RunTaskDrainOutcome,
    TurnRuntimeState,
)
from chrys.orchestration.engine.state import controls as engine_controls
from chrys.orchestration.engine.state.machine import EngineState, EngineStateMachine, Trigger
from chrys.service import usage
from chrys.service.agent_middleware.injection import InjectionMiddleware
from chrys.service.agent_middleware.system_reminder import CATALOG_POINTER_RECORD_COUNT_STATE_KEY
from chrys.service.approval.policy import ApprovalMode
from chrys.service.approval.turn_context import TurnContextHolder
from chrys.service.context.compaction.spill import SpillQuota
from chrys.service.llm.side_call_clients import SideCallClientCache
from chrys.service.mcp.cache import MCPConnectionCache
from chrys.service.memory.overlay import apply_memory_overlay, memory_mcp_server_config
from chrys.service.memory.writeback import deposit_pending_turns
from chrys.service.mutations.coordination import MutationCoordinator
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker
from chrys.service.profiles.models.schema import API_STYLE_RESPONSES
from chrys.service.routing.classifier import RouteDecision
from chrys.service.routing.guard import TiebreakerGuard
from chrys.service.session import checkpoint as session_checkpoint
from chrys.service.session import lifecycle as session_lifecycle
from chrys.service.session.history import SessionHistoryManager
from chrys.service.session.persistence import SessionPersistence, has_real_messages
from chrys.service.session.runtime_metadata import SessionRuntimeMetadata
from chrys.service.state.locks import ActiveSessionGuard
from chrys.service.state.store import (
    SESSION_FILE_NAME,
    SESSION_WRITE_LOCK_TIMEOUT_SECONDS,
    SessionForkError,
    SessionNotFoundError,
    atomic_copy_file,
    session_write_lock_path,
)
from chrys.service.todos.tracker import TodoTracker
from chrys.service.trajectory.items import ensure_history_item_ids
from chrys.service.trajectory.preparation import PreparationOutcome

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import UsageUpdate
    from chrys.foundation.models.session_env import SessionEnvironment
    from chrys.kernel import Agent, LoopRecorder
    from chrys.orchestration.engine.executor import Executor
    from chrys.orchestration.sub_agents.tools import SubAgentTools
    from chrys.service.agent_middleware.injection import ConsumedInjection
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
    from chrys.service.approval.judge import ApprovalJudge
    from chrys.service.context.compaction import CompactionInfo, CompressInfo, PreCompactInfo
    from chrys.service.hooks.manager import HookManager
    from chrys.service.mcp.adapter import MCPAdapter
    from chrys.service.profiles.agents.registry import AgentProfileRegistry
    from chrys.service.profiles.agents.schema import AgentProfile, MCPServerConfig
    from chrys.service.profiles.models.registry import ModelProfileRegistry
    from chrys.service.profiles.models.schema import ModelProfile
    from chrys.service.skills.provider import ChrysSkillsProvider
    from chrys.service.state.store import StateStore
    from chrys.service.trajectory.preparation import PreparationTrace


# Session-keyed engine registry for companion integrations.
_current_engines: dict[str, AgentEngine] = {}
_foreground_engine_session_id: str | None = None
logger = logging.getLogger(__name__)
_SHUTDOWN_POST_RUN_TIMEOUT_SECONDS = 2.0
_SESSION_FORK_AGENT_LOAD_TIMEOUT_SECONDS = 5.0

_FORK_AGENT_LOADING = msg(
    "engine.fork_agent_loading",
    fallback="Cannot fork while the agent is still loading.",
)
_FORK_NOT_READY = msg(
    "engine.fork_not_ready",
    fallback="Cannot fork before a session is ready.",
)
_FORK_SESSION_CHANGED = msg(
    "engine.fork_session_changed",
    fallback="Cannot fork because the active session changed.",
)
_FORK_TURN_ACTIVE = msg(
    "engine.fork_turn_active",
    fallback="Cannot fork while a turn is running.",
)
_FORK_STATE_STORE_MISSING = msg(
    "engine.fork_state_store_missing",
    fallback="State store not configured.",
)
_FORK_RESTORING = msg(
    "engine.fork_restoring",
    fallback="Cannot fork while session state is being restored.",
)
_FORK_LOCK_NOT_OWNED = msg(
    "engine.fork_lock_not_owned",
    fallback="Cannot fork because this window does not own the active session lock.",
)
_FORK_EMPTY_SESSION = msg(
    "engine.fork_empty_session",
    fallback="Cannot fork an empty session.",
)
_FORK_PREPARE_TIMEOUT = msg(
    "engine.fork_prepare_timeout",
    fallback="Timed out preparing session for fork: {detail}",
    multiline=True,
)
_FORK_PREPARE_FAILED = msg(
    "engine.fork_prepare_failed",
    fallback="Failed to prepare session for fork: {detail}",
    multiline=True,
)
_FORK_SESSION_NOT_FOUND = msg(
    "engine.fork_session_not_found",
    fallback="Session '{session_id}' not found.",
)
_FORK_FAILED = msg(
    "engine.fork_failed",
    fallback="Failed to fork session: {detail}",
    multiline=True,
)


def _noop_successful_turn() -> None:
    """Default successful-turn callback."""


def _noop_turn_started() -> None:
    """Default turn-started callback."""


def get_current_engine(session_id: str | None = None) -> AgentEngine | None:
    """Get an active AgentEngine instance.

    Used by buddy module to access conversation history.  A session id
    selects a specific engine; no id preserves the legacy "foreground"
    behavior for single-instance integrations.
    """
    if session_id:
        return _current_engines.get(session_id)
    if _foreground_engine_session_id is not None:
        engine = _current_engines.get(_foreground_engine_session_id)
        if engine is not None:
            return engine
    if len(_current_engines) == 1:
        return next(iter(_current_engines.values()))
    return None


def _set_current_engine(engine: AgentEngine) -> None:
    """Expose an active engine instance for companion integrations."""
    global _foreground_engine_session_id
    if engine.session_id is None:
        return
    _current_engines[engine.session_id] = engine
    _foreground_engine_session_id = engine.session_id


def _unset_current_engine(engine: AgentEngine) -> None:
    """Remove an engine from the companion registry when it shuts down."""
    global _foreground_engine_session_id
    removed = [session_id for session_id, registered in _current_engines.items() if registered is engine]
    for session_id in removed:
        _current_engines.pop(session_id, None)
    if _foreground_engine_session_id in removed:
        _foreground_engine_session_id = next(reversed(_current_engines), None)


class AgentEngine:
    """Top-level orchestrator for the Chrys agent.

    Usage::

        bus = EventBus()
        engine = AgentEngine(bus, settings)
        await engine.start(profile)
        # Frontend publishes UserMessage events → engine handles them
        await engine.shutdown()
    """

    def __init__(
        self,
        event_bus: EventBus,
        settings: Settings | None = None,
        loaded_settings: LoadedSettings | None = None,
        model_registry: ModelProfileRegistry | None = None,
        agent_registry: AgentProfileRegistry | None = None,
        state_store: StateStore | None = None,
        initial_approval_mode: ApprovalMode | None = None,
        mcp_overlay: list[MCPServerConfig] | None = None,
        initial_workspace: Workspace | None = None,
        on_successful_turn: Callable[[], None] | None = None,
        on_turn_started: Callable[[], None] | None = None,
        allow_user_interaction: bool = True,
    ) -> None:
        self._bus = event_bus
        if loaded_settings is not None and settings is not None and settings is not loaded_settings.settings:
            error_message = "Pass either settings or loaded_settings, not two different ones."
            raise ValueError(error_message)
        self._settings_handle = SettingsHandle(
            loaded_settings or LoadedSettings(settings=settings or Settings.from_env(), provenance={})
        )
        self._agent_registry = agent_registry
        self._model_registry = model_registry
        self._persistence = SessionPersistence(state_store, event_bus)
        self._runtime: SessionEnvironment | None = None
        self._agent: Agent | None = None
        self._executor: Executor | None = None
        self._injection: InjectionMiddleware | None = None
        self._agent_profile: AgentProfile | None = None
        self._workspace: Workspace | None = initial_workspace
        self._on_successful_turn: Callable[[], None] = (
            on_successful_turn if on_successful_turn is not None else _noop_successful_turn
        )
        self._on_turn_started: Callable[[], None] = (
            on_turn_started if on_turn_started is not None else _noop_turn_started
        )
        self._allow_user_interaction = allow_user_interaction
        self._session_id: str | None = None
        self._spill_quota = SpillQuota()
        self._subscribed = False
        self._agent_loading = False
        self._agent_load_idle = asyncio.Event()
        self._agent_load_idle.set()
        self._sub_agent_tools: SubAgentTools | None = None
        self._skills_provider: ChrysSkillsProvider | None = None
        self._mcp_cache = MCPConnectionCache()
        self._mcp_adapter: MCPAdapter | None = None
        self._loop_recorder: LoopRecorder | None = None
        self._turn_state = TurnRuntimeState()
        self._turn_context = TurnContextHolder()
        self._turns = TurnCoordinator(self)
        self._shutting_down: bool = False
        self._session_generation: int = 0
        self._build_generation: int = 0
        self._load_generation: int = 0
        self._rebuild_gate_lock = asyncio.Lock()
        self._next_rebuild_permit_id = 1
        self._active_rebuild_permit_id: int | None = None
        self._active_rebuild_permit_owner: str | None = None
        self._active_rebuild_permit_task: asyncio.Task[Any] | None = None
        self._next_session_transition_permit_id = 1
        self._active_session_transition_permit_id: int | None = None
        self._active_session_transition_permit_owner: str | None = None
        self._active_session_transition_permit_task: asyncio.Task[Any] | None = None
        self._active_session_transition_committed = False
        self._runtime_meta = SessionRuntimeMetadata()
        self._memory_watcher: MemoryWritebackWatcher | None = None
        self._route_override: RouteOverride | None = None
        self._last_route: RouteDecision | None = None
        self._route_fingerprint: str = ""
        self._tiebreaker_guard = TiebreakerGuard()
        self._side_call_clients = SideCallClientCache()
        self._long_horizon_campaign: dict[str, Any] | None = None
        self._turn_number: int = 0
        self._active_session_guard = ActiveSessionGuard(state_store)
        self._tool_names: list[str] = []
        self._tool_kinds: dict[str, str] = {}
        self._skill_names: list[str] = []
        self._memory_files: list[str] = []
        self._agent_profile_fingerprint: str = ""
        self._model_profile_fingerprint: str = ""
        self._runtime_details = AgentRuntimeDetails()
        self._trajectory_recorder = trajectory_recorder.TrajectoryRecorder()
        self._fsm = EngineStateMachine()
        self._history = SessionHistoryManager()
        self._mutation_tracker: MutationTracker | None = None
        self._workspace_change_tracker = WorkspaceChangeTracker()
        self._mutation_coordinator: MutationCoordinator | None = None
        self._todo_tracker: TodoTracker | None = None
        self._reminder_middleware: SystemReminderMiddleware | None = None
        self._approval_mode: ApprovalMode = initial_approval_mode or ApprovalMode.MANUAL
        self._approval_judge: ApprovalJudge | None = None
        self._active_profile: ModelProfile | None = None
        # Set only by a per-session ACP model switch (SetModelProfile). When True,
        # SettingsReload keeps the live ``settings.model_profile`` instead of
        # reverting to the global env default. The TUI never sets this (it persists
        # CHRYS_MODEL_PROFILE to .env and relies on SettingsReload re-reading env),
        # so its behavior is unchanged.
        self._model_profile_pinned: bool = False
        # Set via ``pin_ask_user_timeout`` when the caller owns
        # ``ask_user_timeout_seconds`` out-of-band (ACP injects it via
        # dataclasses.replace at launch, not env). When True, SettingsReload
        # preserves the live value instead of reverting to the env default; TUI/CLI
        # leave it False so a changed CHRYS_ASK_USER_TIMEOUT_SECONDS takes effect.
        self._ask_user_timeout_pinned: bool = False
        self._paused_sub_agents: set[str] = set()
        self._intermediate_texts: dict[int, str] = {}
        self._consumed_injections: list[ConsumedInjection] = []
        self._requirement_clarification_workflow: Any | None = None
        self._injection_notify_task: asyncio.Task[None] | None = None
        self._usage_tasks: set[asyncio.Task[None]] = set()
        self._usage_publish_tail: asyncio.Task[None] | None = None
        self._recovered_from_sidecar: bool = False
        # When True, ``_save_current_session`` is a no-op.  Set by the
        # rollback handler immediately before it calls the session
        # restore path, which would otherwise call ``shutdown()`` →
        # ``_save_current_session()`` and clobber the just-swapped
        # ``session.json`` with the in-memory pre-rollback state.
        self._suppress_save: bool = False
        self._shutdown_used_cancel_fallback: bool = False
        # Crash-recovery checkpoint writes are dispatched to a background task
        # so the per-round-trip LLM call never blocks on the sidecar's fsync +
        # file lock (slow on Windows under load).  ``_pending_recovery_state``
        # holds the latest snapshot to write (newest wins); the single in-flight
        # ``_recovery_write_task`` is drained at save/shutdown boundaries so the
        # sidecar stays durable and is never resurrected after a clean turn.
        self._recovery_write_task: asyncio.Task[None] | None = None
        self._pending_recovery_state: tuple[int, dict[str, Any]] | None = None
        self._recovery_persistence_lock = asyncio.Lock()
        self._strict_recovery_write_tasks: set[asyncio.Task[bool]] = set()
        # Monotonic snapshot stamps: a queued background write whose snapshot
        # predates the last persisted one is skipped, so a coalesced writer
        # that froze state before a strict barrier can never downgrade the
        # sidecar to a snapshot missing the barrier's committed exchanges.
        self._recovery_snapshot_seq = 0
        self._recovery_persisted_seq = 0
        # Lifecycle hook orchestrator.  Lazily built on first
        # ``start()`` so unit tests that never start the engine don't
        # pay for filesystem setup (outbox dirs etc.).  Shared across
        # profile switches within a session — hooks are global, not
        # per-profile.
        self._hook_manager: HookManager | None = None
        # ``session_end`` fires once per session: normally from ``shutdown()``,
        # but earlier when the live session is deleted underneath the engine
        # (clear / delete-current), so hooks still see the id and the files.
        self._session_end_fired = False
        # Background outbox-recovery task kicked off in ``start()``.
        # Held here so it isn't garbage-collected mid-run (asyncio only
        # weak-refs tasks created via ``create_task``).
        self._outbox_recovery_task: asyncio.Task[int] | None = None
        self._mcp_overlay = list(mcp_overlay or [])

    @property
    def session_generation(self) -> int:
        """Engine-owned session generation for stale-owner detection."""
        return self._session_generation

    @property
    def conversation_revision(self) -> int:
        """Monotonic fresh/retry lifecycle revision for stale projections."""
        return self._turn_state.conversation_revision

    @property
    def build_generation(self) -> int:
        """Engine-owned successful-build generation."""
        return self._build_generation

    @property
    def workspace_primary_cwd(self) -> str:
        """Current normalized primary cwd for stale projection detection."""
        return self._workspace_cwd()

    @property
    def load_generation(self) -> int:
        """Engine-owned agent-load attempt generation."""
        return self._load_generation

    def _workspace_cwd(self) -> str:
        """Return the current workspace cwd, falling back only before workspace initialization."""
        if self._workspace is not None:
            return self._workspace.primary_cwd
        from chrys.foundation.platform import safe_getcwd

        return safe_getcwd()

    @property
    def _session_dir(self) -> Path | None:
        """Return the session directory path, or ``None`` if no session is active.

        Delegates to the state store when available so that tests using
        ``tmp_path``-based stores write to the temp directory. Without a
        store, fall back to the active sessions directory.
        """
        session_id = self._session_id
        if not session_id:
            return None
        return self._session_dir_for(session_id)

    def _session_dir_for(self, session_id: str) -> Path:
        """Resolve a session directory for an explicit, already-captured id."""
        store = self._persistence.state_store
        if store is not None:
            return store.session_dir(session_id)
        from chrys.foundation.config.settings import resolve_sessions_dir
        from chrys.foundation.util.session_ids import session_short_id

        return resolve_sessions_dir() / session_short_id(session_id)

    def _sessions_root_dir(self, session_id: str) -> Path | None:
        """Return the sessions root used for lock files."""
        store = self._persistence.state_store
        if store is not None:
            return store.session_dir(session_id).parent
        from chrys.foundation.config.settings import resolve_sessions_dir

        return resolve_sessions_dir()

    def _session_write_lock_path(self, session_id: str) -> Path | None:
        sessions_dir = self._sessions_root_dir(session_id)
        if sessions_dir is None:
            return None
        path = session_write_lock_path(sessions_dir, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_running(self) -> bool:
        return self._executor is not None and self._executor.is_running

    @property
    def is_turn_active(self) -> bool:
        """True when a turn is in-flight and can accept an injection.

        Uses the FSM rather than the executor flag so it also covers
        ``PENDING_RETRY`` / ``AWAITING_SUB_AGENTS`` — the same predicate
        ``on_user_message`` uses to inject instead of starting a turn.
        """
        return self._turns.is_turn_active

    @property
    def is_turn_lifecycle_active(self) -> bool:
        """True through execution, session persistence, and after-turn hooks.

        Unlike :attr:`is_turn_active`, this follows the owned run task rather
        than the FSM.  Finalization intentionally transitions the FSM to idle
        before it saves the session and drains hooks.
        """
        return self._turns.is_turn_lifecycle_active

    @property
    def turn_lifecycle_task(self) -> asyncio.Task[None] | None:
        """Return the currently owned execution/finalization task.

        Callers that need a boundary for one captured turn should retain this
        exact task instead of later awaiting the replaceable run-task chain.
        """
        return self._turns.run_task

    def was_turn_lifecycle_saved(self, task: asyncio.Task[None]) -> bool:
        """Return whether *task* completed its final session save successfully."""
        return self._turns.was_run_task_finally_saved(task)

    @property
    def session_id(self) -> str | None:
        """Current canonical session id, or ``None`` before session start."""
        return self._session_id

    @property
    def last_route(self) -> RouteDecision | None:
        """Latest routing decision, or ``None`` before the first classified turn."""
        return self._last_route

    @property
    def runtime_details(self) -> AgentRuntimeDetails:
        """Runtime details for the currently built agent."""
        return self._runtime_details

    # Public read accessors for engine collaborators (buddy, session-title
    # updater, other app-layer features) so they don't reach into privates.

    @property
    def event_bus(self) -> EventBus:
        """Event bus the engine publishes on (constructor-injected)."""
        return self._bus

    @property
    def state(self) -> EngineState:
        """Current lifecycle state of the engine state machine.

        Read-only view; transitions stay engine-internal.
        """
        return self._fsm.state

    @property
    def workspace(self) -> Workspace | None:
        """Active workspace (primary cwd + working dirs), or ``None`` before start.

        Read-only view: rebinding stays with the session-lifecycle host
        contract.
        """
        return self._workspace

    @property
    def _settings(self) -> Settings:
        """Read-only by construction: the handle is where settings change."""
        return self._settings_handle.settings

    @property
    def _loaded_settings(self) -> LoadedSettings:
        return self._settings_handle.loaded

    @property
    def settings(self) -> Settings:
        """Live settings object the engine was built with."""
        return self._settings_handle.settings

    @property
    def loaded_settings(self) -> LoadedSettings:
        """Live settings together with where each value came from.

        Anything that answers "why does this session use this value" — the
        panel, ``settings/options``, the explanation for a key that was sealed
        at its default — needs the provenance that used to be discarded the
        moment a root unpacked ``.settings``.
        """
        return self._settings_handle.loaded

    @property
    def settings_handle(self) -> SettingsHandle:
        """The cell a frontend shares to stay on the same settings as this engine."""
        return self._settings_handle

    @property
    def agent_profile(self) -> AgentProfile | None:
        """Agent profile of the currently built runtime, or ``None`` before build."""
        return self._agent_profile

    @property
    def active_model_profile(self) -> ModelProfile | None:
        """Model profile of the currently built runtime, or ``None`` before build."""
        return self._active_profile

    @property
    def model_registry(self) -> ModelProfileRegistry | None:
        """Model profile registry the engine resolves profiles from, if any."""
        return self._model_registry

    @property
    def session_dir(self) -> Path | None:
        """Session directory path, or ``None`` if no session is active."""
        return self._session_dir

    def trajectory_context(self) -> TrajectoryContext | None:
        """The session's trajectory recording scope (current turn), or ``None`` when unrecorded.

        Side calls made outside a model run (title generation) bind this with
        :func:`chrys.foundation.trajectory.context.side_call_scope` so their
        exchanges are attributed to the session.
        """
        return self._trajectory_recorder.context()

    @property
    def history_messages(self) -> list:
        """Live message list of the bound session history.

        Raises if no history is bound yet (before session start) — callers
        that can run that early should guard accordingly.
        """
        return self._history.messages

    def tool_kind_for_name(self, name: str) -> str:
        """Return the live tool kind for *name*, or an empty string when unknown."""
        kind = self._tool_kinds.get(name, "")
        if kind:
            return kind
        if self._sub_agent_tools is not None and name in self._sub_agent_tools.tool_names():
            return KIND_SUB_AGENT
        return ""

    def current_profile_snapshot(self) -> ProfileSwitched:
        """Build a no-op ``ProfileSwitched`` reflecting the live runtime.

        A frontend reselecting the already-active agent performs no backend
        switch and emits no event, so callers that still owe the client the
        standard runtime envelope read it here instead of fabricating blank
        fields. ``from``/``to`` are identical because nothing changed. Field
        sourcing mirrors the real switch event so the two cannot drift.
        """
        profile = self._agent_profile
        name = profile.name if profile else ""
        display = (profile.display_name or profile.name) if profile else ""
        messages = self._executor.history_state.get("messages", []) if self._executor is not None else []
        return ProfileSwitched(
            from_profile=name,
            to_profile=name,
            from_display_name=display,
            to_display_name=display,
            message_count=len(messages),
            model_profile_id=self._active_profile.id if self._active_profile else "",
            max_context_tokens=self._active_profile.max_context_tokens if self._active_profile else 0,
            session_id=self._session_id,
            tool_names=list(self._tool_names),
            skill_names=list(self._skill_names),
            sub_agent_tool_names=self._sub_agent_tools.tool_names() if self._sub_agent_tools else [],
            memory_files=list(self._memory_files),
            runtime_details=copy.deepcopy(self._runtime_details),
        )

    @property
    def approval_mode(self) -> ApprovalMode:
        """Live approval mode (updated by ``SetApprovalMode``)."""
        return self._approval_mode

    @property
    def recovered_from_sidecar(self) -> bool:
        """Whether the current session restore selected the crash-recovery sidecar."""
        return self._recovered_from_sidecar

    async def wait_for_run_task(self) -> None:
        """Wait for the active run task to finish post-run cleanup and saving."""
        await self._turns.wait_for_run_task()

    async def drain_run_task_chain_for_boundary(self) -> RunTaskDrainOutcome:
        """Observe the active run-task chain for a rebuild/session boundary."""
        return await self._turns.drain_run_task_chain_for_boundary()

    def _begin_agent_load(self) -> None:
        """Mark agent infrastructure as loading and block new user turns."""
        self._load_generation += 1
        self._agent_loading = True
        self._agent_load_idle.clear()

    def _finish_agent_load(self) -> None:
        """Mark the current agent load attempt as settled."""
        self._agent_loading = False
        self._agent_load_idle.set()

    def _advance_build_generation(self) -> None:
        """Record one successful replacement of build-scoped collaborators."""
        self._build_generation += 1

    def _invalidate_turn_runtime_for_session_transition_pre_shutdown(self) -> None:
        """Compatibility seam for state-only tests that exercise pre-shutdown invalidation.

        Real lifecycle transitions use ``_begin_session_transition()``, which also
        closes prompt/retry admission under the serialized gate before invalidating
        session-owned turn state.
        """
        old_generation = self._session_generation
        self._session_generation += 1
        self._turn_state.invalidate_for_session_transition_pre_shutdown(
            old_session_generation=old_generation,
        )

    def _reset_turn_runtime_after_session_shutdown(self) -> None:
        """Clear turn runtime state after shutdown has observed the old task."""
        prompt_admission_owner = (
            self._active_session_transition_permit_owner
            if self._current_task_owns_session_transition_permit()
            else None
        )
        old_scope = self._turn_state.reset_after_session_shutdown(
            prompt_admission_owner=prompt_admission_owner,
        )
        if old_scope is not None and self._reminder_middleware is not None:
            self._reminder_middleware.expire_current_run_scope(old_scope.reminder_scope)

    async def _begin_session_transition(self, operation: str) -> str:
        """Acquire the serialized session-transition boundary and close turn admission."""
        owner = await self._prepare_session_transition(operation)
        if owner is None:  # No owner token was supplied, so validation cannot fail.
            raise RuntimeError("Session transition acquisition unexpectedly failed")
        try:
            self._commit_session_transition(owner)
        except BaseException:
            self._abort_prepared_session_transition(owner)
            raise
        return owner

    async def _prepare_session_transition_if_current(
        self,
        operation: str,
        *,
        session_id: str | None,
        session_generation: int,
    ) -> str | None:
        """Fence new prompts and drain prior admissions without invalidating turn state."""
        return await self._prepare_session_transition(
            operation,
            expected_owner=(session_id, session_generation),
            wait_for_active_admissions=True,
        )

    async def begin_rollback_projection(
        self,
        *,
        session_id: str | None,
        session_generation: int,
    ) -> str | None:
        """Fence prompt admission while the TUI projects live rollback state."""
        if self._session_generation != session_generation:
            return None
        if session_id is not None and self._session_id != session_id:
            return None
        return await self._prepare_session_transition_if_current(
            "rollback_projection",
            session_id=self._session_id,
            session_generation=session_generation,
        )

    def finish_rollback_projection(self, owner: str) -> None:
        """Release a read-only rollback projection fence without committing it."""
        self._finish_session_transition(owner)

    async def _prepare_session_transition(
        self,
        operation: str,
        *,
        expected_owner: tuple[str | None, int] | None = None,
        wait_for_active_admissions: bool = False,
    ) -> str | None:
        """Acquire the shared gate and close new admission without committing."""
        await self._rebuild_gate_lock.acquire()
        if expected_owner is not None and expected_owner != (self._session_id, self._session_generation):
            self._rebuild_gate_lock.release()
            return None
        permit_id = self._next_session_transition_permit_id
        self._next_session_transition_permit_id += 1
        owner = f"session:{operation}:{permit_id}"
        self._active_session_transition_permit_id = permit_id
        self._active_session_transition_permit_owner = owner
        self._active_session_transition_permit_task = asyncio.current_task()
        self._active_session_transition_committed = False
        self._turn_state.close_prompt_admission_for_rebuild(owner)
        try:
            if wait_for_active_admissions:
                await self._turn_state.wait_for_active_admissions_idle()
            await self._turn_state.wait_for_active_injection_commits_idle()
            if expected_owner is not None and expected_owner != (self._session_id, self._session_generation):
                self._abort_prepared_session_transition(owner)
                return None
        except BaseException:
            self._abort_prepared_session_transition(owner)
            raise
        return owner

    def _commit_session_transition(self, owner: str) -> None:
        """Invalidate the old session generation after a prepared transition is accepted."""
        if (
            self._active_session_transition_permit_owner != owner
            or not self._current_task_owns_session_transition_permit()
            or self._active_session_transition_committed
        ):
            raise RuntimeError("Invalid session transition permit")
        self._active_session_transition_committed = True
        old_generation = self._session_generation
        self._session_generation += 1
        self._turn_state.invalidate_for_session_transition_pre_shutdown(
            old_session_generation=old_generation,
            prompt_admission_owner=owner,
        )

    def _abort_prepared_session_transition(self, owner: str) -> None:
        """Release a prepared transition without changing session ownership."""
        self._turn_state.reopen_prompt_admission_after_rebuild(owner)
        self._active_session_transition_permit_id = None
        self._active_session_transition_permit_owner = None
        self._active_session_transition_permit_task = None
        self._active_session_transition_committed = False
        self._rebuild_gate_lock.release()

    def _finish_session_transition(self, owner: str) -> None:
        """Release the serialized session-transition boundary and reopen admission."""
        if self._active_session_transition_permit_owner != owner:
            return
        self._turn_state.reopen_prompt_admission_after_rebuild(owner)
        self._active_session_transition_permit_id = None
        self._active_session_transition_permit_owner = None
        self._active_session_transition_permit_task = None
        self._active_session_transition_committed = False
        self._rebuild_gate_lock.release()

    async def _fire_session_end_hooks(self) -> None:
        """Fire ``session_end`` for the live session and wait for it to finish.

        Fires at most once per session.  ``shutdown()`` is the usual caller;
        deleting the ACTIVE session calls it first, while ``_session_id`` and
        the session files still exist, and the later shutdown then skips the
        duplicate.  ``fire()`` only *spawns* ``async``-mode hooks, so this
        also drains them (without closing the manager) — otherwise a delete
        could remove the files before such a hook ran.  The next ``start()``
        re-arms it for the session it brings up.
        """
        hook_manager = self._hook_manager
        if hook_manager is None or self._session_end_fired:
            return
        from chrys.service.hooks.events import HookEvent

        self._session_end_fired = True
        profile_name = self._agent_profile.name if self._agent_profile is not None else ""
        await hook_manager.fire(
            HookEvent.SESSION_END,
            {"session_id": self._session_id, "profile": profile_name, "cwd": self._workspace_cwd()},
            scope="session",
        )
        await hook_manager.drain_session(close=False)

    async def _close_trajectory_log(self) -> None:
        """Close the trajectory writer so the session directory can be removed.

        Deleting the ACTIVE session removes its folder while this runtime is
        still up. A live writer holds the log's lease, and a leased directory
        is only tombstoned — swept at the next store startup, so the files a
        user asked to delete would outlive the request for the rest of the
        run. Closing first makes the delete physical.

        A delete that then fails (the lock is busy) leaves the session
        running unrecorded until the next runtime resumes its log. That is
        the recorder's standing contract — recording never holds a session
        up — and the log it leaves behind is closed, not torn.
        """
        await self._trajectory_recorder.close(reason=TrajectoryRuntimeFinishReason.SESSION_SWITCH)

    async def _wait_for_agent_load_idle(self) -> None:
        """Wait until no agent build/rebuild is in progress."""
        while self._agent_loading:
            await self._agent_load_idle.wait()

    def capture_rebuild_control_token(self) -> engine_controls.RebuildControlToken:
        """Capture owner clocks for one runtime-control rebuild request."""
        return engine_controls.RebuildControlToken(
            session_id=self._session_id,
            session_generation=self._session_generation,
            build_generation=self._build_generation,
            load_generation=self._load_generation,
        )

    async def acquire_rebuild_permit(
        self,
        token: engine_controls.RebuildControlToken,
    ) -> engine_controls.RebuildPermit | engine_controls.RebuildPermitDenied:
        """Close turn admission and acquire the serialized rebuild boundary."""
        await self._rebuild_gate_lock.acquire()
        permit_id = self._next_rebuild_permit_id
        self._next_rebuild_permit_id += 1
        owner = f"rebuild:{permit_id}"
        self._turn_state.close_prompt_admission_for_rebuild(owner)
        try:
            await self._turn_state.wait_for_active_admissions_idle()
            await self._turn_state.wait_for_active_injection_commits_idle()
            try:
                drain_outcome = await self.drain_run_task_chain_for_boundary()
            except Exception as exc:
                return self._deny_rebuild_permit_locked(
                    owner,
                    reason="busy",
                    message=f"Cannot rebuild because the active run failed: {exc}",
                )
            await self._turn_state.wait_for_active_injection_commits_idle()
            await self._wait_for_agent_load_idle()
            denied = self._validate_rebuild_token_after_boundary(token, drain_cancelled=drain_outcome.cancelled)
            if denied is not None:
                return self._deny_rebuild_permit_locked(owner, denied=denied)
            permit = engine_controls.RebuildPermit(permit_id=permit_id, owner=owner, token=token)
            self._active_rebuild_permit_id = permit_id
            self._active_rebuild_permit_owner = owner
            self._active_rebuild_permit_task = asyncio.current_task()
            return permit
        except BaseException:
            self._turn_state.reopen_prompt_admission_after_rebuild(owner)
            self._rebuild_gate_lock.release()
            raise

    def _validate_rebuild_token_after_boundary(
        self,
        token: engine_controls.RebuildControlToken,
        *,
        drain_cancelled: bool,
    ) -> engine_controls.RebuildPermitDenied | None:
        if self._shutting_down:
            return self._rebuild_denied(
                "shutdown",
                "runtime_mutation_shutdown",
                "Cannot rebuild while the engine is shutting down.",
            )
        if self._session_id != token.session_id or self._session_generation != token.session_generation:
            return self._rebuild_denied(
                "session_changed",
                "runtime_mutation_session_changed",
                "Cannot rebuild because the active session changed.",
            )
        if self._build_generation != token.build_generation:
            return self._rebuild_denied(
                "superseded",
                "runtime_mutation_superseded",
                "Cannot rebuild because a newer runtime is already active.",
            )
        if self._load_generation != token.load_generation:
            return self._rebuild_denied(
                "superseded",
                "runtime_mutation_superseded",
                "Cannot rebuild because a newer load attempt already settled.",
            )
        if self._agent_loading:
            return self._rebuild_denied(
                "load_active",
                "runtime_mutation_load_active",
                "Cannot rebuild while an agent load is still active.",
            )
        if self._turn_state.active_admission_count() > 0:
            return self._rebuild_denied(
                "busy",
                "runtime_mutation_busy",
                "Cannot rebuild while a prompt or retry is being admitted.",
            )
        run_task = self._turn_state.run_task
        if run_task is not None and not run_task.done():
            return self._rebuild_denied(
                "busy",
                "runtime_mutation_busy",
                "Cannot rebuild while a run is active.",
            )
        if drain_cancelled:
            return self._rebuild_denied(
                "busy",
                "runtime_mutation_busy",
                "Cannot rebuild because the active run was cancelled.",
            )
        return None

    def _rebuild_denied(
        self,
        reason: engine_controls.RebuildPermitDeniedReason,
        code: str,
        message: str,
    ) -> engine_controls.RebuildPermitDenied:
        return engine_controls.RebuildPermitDenied(reason=reason, code=code, message=message)

    def _deny_rebuild_permit_locked(
        self,
        owner: str,
        *,
        reason: engine_controls.RebuildPermitDeniedReason | None = None,
        message: str | None = None,
        denied: engine_controls.RebuildPermitDenied | None = None,
    ) -> engine_controls.RebuildPermitDenied:
        self._turn_state.reopen_prompt_admission_after_rebuild(owner)
        self._rebuild_gate_lock.release()
        if denied is not None:
            return denied
        return self._rebuild_denied(
            reason or "busy",
            "runtime_mutation_busy",
            message or "Cannot rebuild while the runtime is busy.",
        )

    def release_rebuild_permit(self, permit: engine_controls.RebuildPermit) -> None:
        """Release a serialized rebuild permit and reopen prompt admission."""
        if self._active_rebuild_permit_id != permit.permit_id:
            return
        self._turn_state.reopen_prompt_admission_after_rebuild(permit.owner)
        self._active_rebuild_permit_id = None
        self._active_rebuild_permit_owner = None
        self._active_rebuild_permit_task = None
        self._rebuild_gate_lock.release()

    def _current_task_owns_rebuild_permit(self) -> bool:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            return False
        return (
            task is not None and self._active_rebuild_permit_id is not None and self._active_rebuild_permit_task is task
        )

    def _current_task_owns_session_transition_permit(self) -> bool:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            return False
        return (
            task is not None
            and self._active_session_transition_permit_id is not None
            and self._active_session_transition_permit_task is task
        )

    def _ensure_rebuild_permit(self, permit: engine_controls.RebuildPermit) -> None:
        if (
            self._active_rebuild_permit_id != permit.permit_id
            or self._active_rebuild_permit_owner != permit.owner
            or not self._current_task_owns_rebuild_permit()
        ):
            raise RuntimeError("Invalid rebuild permit")

    async def start(
        self,
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        """Initialize the engine with an agent profile and wire up event handlers."""
        if self._current_task_owns_rebuild_permit() or self._current_task_owns_session_transition_permit():
            await self._start_without_rebuild_permit(
                profile, operation=operation, staged_loaded=staged_loaded, workspace=workspace
            )
            return
        token = self.capture_rebuild_control_token()
        permit = await self.acquire_rebuild_permit(token)
        if isinstance(permit, engine_controls.RebuildPermitDenied):
            raise RuntimeError(permit.message)
        try:
            await self._start_without_rebuild_permit(
                profile, operation=operation, staged_loaded=staged_loaded, workspace=workspace
            )
        finally:
            self.release_rebuild_permit(permit)

    async def _start_without_rebuild_permit(
        self,
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        """Start the engine while the caller owns the rebuild boundary."""
        # Every session (fresh, restored, reset) starts here: re-arm the
        # once-per-session ``session_end`` hook fired by shutdown/delete.
        self._session_end_fired = False
        self._start_memory_watcher()
        await agent_lifecycle.start(
            self,
            profile,
            operation=operation,
            set_current_engine=lambda: _set_current_engine(self),
            staged_loaded=staged_loaded,
            workspace=workspace,
        )

    async def start_with_rebuild_permit(
        self,
        permit: engine_controls.RebuildPermit,
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        """Start using an already-acquired rebuild permit."""
        self._ensure_rebuild_permit(permit)
        await self.start(profile, operation=operation, staged_loaded=staged_loaded, workspace=workspace)

    async def prepare(self, fallback_profile: AgentProfile | None = None) -> None:
        """Subscribe handlers and optionally seed the profile used by legacy restores."""
        if fallback_profile is not None:
            self._agent_profile = fallback_profile
        await self._subscribe_event_handlers()

    async def _subscribe_event_handlers(self) -> None:
        """Subscribe frontend event handlers once, even if the first build fails.

        Handlers are intentionally available before the initial build completes
        so settings/profile changes can recover from startup failure.  The TUI
        blocks user actions while agent loading is active; if an external caller
        publishes recovery events during a still-running first build, those
        events may re-enter ``start()`` before ``_executor`` exists.
        """
        if self._subscribed:
            return
        self._subscribed = True
        await self._bus.subscribe(UserMessage, self._on_user_message)
        await self._bus.subscribe(UserInterrupt, self._on_user_interrupt)
        await self._bus.subscribe(UserRetry, self._on_user_retry)
        await self._bus.subscribe(UserInject, self._on_user_inject)
        await self._bus.subscribe(UserInjectCancel, self._on_user_inject_cancel)
        await self._bus.subscribe(UserRollback, self._on_user_rollback)
        await self._bus.subscribe(AgentProfileSwitch, self._on_profile_switch)
        await self._bus.subscribe(SessionNew, self._on_new_session)
        await self._bus.subscribe(SessionRestore, self._on_session_restore)
        await self._bus.subscribe(SessionDelete, self._on_session_delete)
        await self._bus.subscribe(SessionClear, self._on_session_clear)
        await self._bus.subscribe(SessionFork, self._on_session_fork)
        await self._bus.subscribe(WorkspaceChange, self._on_workspace_change)
        await self._bus.subscribe(SettingsReload, self._on_settings_reload)
        await self._bus.subscribe(SetApprovalMode, self._on_set_approval_mode)
        await self._bus.subscribe(SetModelProfile, self._on_set_model_profile)
        await self._bus.subscribe(SubAgentRetryRequested, self._on_sub_agent_retry)
        await self._bus.subscribe(RouteOverride, self._on_route_override)
        await self._bus.subscribe(SubAgentAbortRequested, self._on_sub_agent_abort)
        await self._bus.subscribe(SubAgentPaused, self._on_sub_agent_paused)
        await self._bus.subscribe(SubAgentResumed, self._on_sub_agent_unpaused)
        await self._bus.subscribe(SubAgentAborted, self._on_sub_agent_unpaused)
        await self._bus.subscribe(SubAgentCascadeAborted, self._on_sub_agent_unpaused)

    async def _build_agent(
        self,
        profile: AgentProfile,
        staged: agent_lifecycle.StagedBuild,
        *,
        preserved_history: dict | None = None,
    ) -> None:
        """Build Agent, Executor, and all supporting infrastructure."""
        await agent_lifecycle.build_agent(
            self,
            self._profile_with_mcp_overlay(profile),
            staged=staged,
            build_agent_fn=build_agent,
            preserved_history=preserved_history,
        )

    def _profile_with_mcp_overlay(self, profile: AgentProfile) -> AgentProfile:
        """Return a per-session profile copy with memory and ephemeral MCP servers appended."""
        effective = apply_memory_overlay(profile, self._settings)
        if not self._mcp_overlay:
            return effective
        if effective is profile:
            effective = copy.deepcopy(profile)
        effective.tools.mcp.extend(copy.deepcopy(self._mcp_overlay))
        return effective

    async def _cleanup_replaced_build_resources(
        self,
        old_agent: Agent | None,
        old_sub_agent_tools: SubAgentTools | None,
        old_mcp_adapter: MCPAdapter | None,
    ) -> None:
        """Release resources from the previous successful build after replacement."""
        await agent_lifecycle.cleanup_replaced_build_resources(self, old_agent, old_sub_agent_tools, old_mcp_adapter)

    async def _publish_load_progress(
        self,
        *,
        phase: str,
        message: str,
        server_name: str = "",
        current: int = 0,
        total: int = 0,
        failed: int = 0,
        status: str = "",
        subject: str = "",
        detail: str = "",
    ) -> None:
        await agent_lifecycle.publish_load_progress(
            self,
            phase=phase,
            message=message,
            server_name=server_name,
            current=current,
            total=total,
            failed=failed,
            status=status,
            subject=subject,
            detail=detail,
        )

    async def _publish_load_failed(self, *, operation: str, profile: AgentProfile, exc: Exception) -> None:
        await agent_lifecycle.publish_load_failed(self, operation=operation, profile=profile, exc=exc)

    async def _soft_restart(
        self,
        new_profile: AgentProfile,
        workspace: Workspace | None = None,
        *,
        operation: str = "switch",
        staged_loaded: LoadedSettings | None = None,
    ) -> None:
        """Restart the agent with a new profile/workspace while preserving history."""
        if self._current_task_owns_rebuild_permit() or self._current_task_owns_session_transition_permit():
            await self._soft_restart_without_rebuild_permit(
                new_profile, workspace, operation=operation, staged_loaded=staged_loaded
            )
            return
        token = self.capture_rebuild_control_token()
        permit = await self.acquire_rebuild_permit(token)
        if isinstance(permit, engine_controls.RebuildPermitDenied):
            raise RuntimeError(permit.message)
        try:
            await self._soft_restart_without_rebuild_permit(
                new_profile, workspace, operation=operation, staged_loaded=staged_loaded
            )
        finally:
            self.release_rebuild_permit(permit)

    async def _soft_restart_without_rebuild_permit(
        self,
        new_profile: AgentProfile,
        workspace: Workspace | None = None,
        *,
        operation: str = "switch",
        staged_loaded: LoadedSettings | None = None,
    ) -> None:
        """Soft-restart while the caller owns the rebuild boundary."""
        await agent_lifecycle.soft_restart(
            self, new_profile, workspace, operation=operation, staged_loaded=staged_loaded
        )

    async def soft_restart_with_rebuild_permit(
        self,
        permit: engine_controls.RebuildPermit,
        new_profile: AgentProfile,
        workspace: Workspace | None = None,
        *,
        operation: str = "switch",
        staged_loaded: LoadedSettings | None = None,
    ) -> None:
        """Soft-restart using an already-acquired rebuild permit."""
        self._ensure_rebuild_permit(permit)
        await self._soft_restart(new_profile, workspace=workspace, operation=operation, staged_loaded=staged_loaded)

    async def shutdown(self, *, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        """Gracefully shut down the engine."""
        self._shutting_down = True
        self._shutdown_used_cancel_fallback = False
        suppress_trailing_save = False
        _unset_current_engine(self)
        self._fsm.try_transition(Trigger.SHUTDOWN)
        if self._executor is not None and self._executor.is_running:
            await self._executor.interrupt()
        run_task = self._turns.run_task
        if run_task is not None and not run_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(run_task),
                    timeout=_SHUTDOWN_POST_RUN_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                self._shutdown_used_cancel_fallback = True
                suppress_trailing_save = True
                run_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await run_task
            except asyncio.CancelledError:
                pass
        self._turns.clear_run_task()
        # A cancelled/timed-out worker can leave a queued retry or active
        # admission behind. Terminalize their preparations while the trajectory
        # writer is still open; the post-shutdown reset is too late to emit.
        self._turn_state.clear_pending_retry(outcome=PreparationOutcome.DROPPED)
        self._turn_state.clear_active_admissions(outcome=PreparationOutcome.OWNER_CHANGED)
        self._turn_state.clear_pre_admission_preparations()
        if self._injection is not None:
            for injection in self._injection.drain_pending():
                if injection.preparation is not None:
                    injection.preparation.finished_soon(
                        outcome=PreparationOutcome.TARGET_STALE,
                        target_turn_id=injection.target_turn_id,
                    )
        if self._outbox_recovery_task is not None:
            if self._outbox_recovery_task.done():
                try:
                    self._outbox_recovery_task.result()
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("Outbox recovery task failed")
            else:
                self._outbox_recovery_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._outbox_recovery_task
            self._outbox_recovery_task = None
        # Drain any queued UsageUpdate publish tasks before the bus is torn
        # down — orphaned tasks otherwise either deliver a stale UsageUpdate
        # into the next session's subscriber set (engine instance is reused
        # across session restore / new-session) or get garbage-collected mid-
        # await, producing "Task was destroyed but it is pending" warnings on
        # process exit.
        if self._usage_tasks:
            await asyncio.gather(*tuple(self._usage_tasks), return_exceptions=True)
        self._usage_tasks.clear()
        self._usage_publish_tail = None
        # Fire ``session_end`` hooks BEFORE we tear down anything — they
        # may want a live session to inspect.  Then drain any in-flight
        # async hooks within the configured shutdown grace.  Detached
        # hooks survive this call by design.
        if self._hook_manager is not None:
            await self._fire_session_end_hooks()
            await self._hook_manager.drain_session()
            # ``drain_session()`` closes the manager.  The same engine
            # instance is reused for restore/new-session flows, so force
            # the next ``start()`` to reload config and create a fresh
            # manager instead of carrying a closed no-op instance forward.
            self._hook_manager = None
        # Same boundary as ``session_end``: a session that ends normally gets
        # one last deposit, so a short-lived host (a PACT role, `chrys run`)
        # never has to wait out an idle window that will not arrive. The
        # watcher itself flushes only when there is something pending.
        if self._memory_watcher is not None:
            watcher, self._memory_watcher = self._memory_watcher, None
            await watcher.stop(flush=self._settings.memory_writeback_on_session_end, reason="session_end")
        # Side-call clients outlive individual turns but not the session.
        await self._side_call_clients.close()
        # Auto-save before shutdown
        restore_suppress_save = False
        if suppress_trailing_save and not self._suppress_save:
            self._suppress_save = True
            restore_suppress_save = True
        try:
            await self._save_current_session()
        finally:
            if restore_suppress_save:
                self._suppress_save = False
        # Stamp the coordination registry closed (peers stop warning
        # about us); files are kept for peers' late rollbacks — GC only.
        if self._mutation_coordinator is not None:
            try:
                await asyncio.get_running_loop().run_in_executor(None, self._mutation_coordinator.close)
            except Exception:
                logger.debug("Mutation coordinator close failed", exc_info=True)
        # The trajectory writer must be closed before the directory it writes
        # into can be judged empty. ``close_mcp_cache=False`` is how the
        # session-switch flows (new/restore/clear/reset) shut the engine down
        # while the process keeps running; every other caller is exiting.
        await self._trajectory_recorder.close(
            reason=(
                TrajectoryRuntimeFinishReason.GRACEFUL_SHUTDOWN
                if close_mcp_cache
                else TrajectoryRuntimeFinishReason.SESSION_SWITCH
            ),
        )
        # Clean up empty session directory (no messages were sent)
        self._cleanup_empty_session_dir()
        if self._sub_agent_tools is not None:
            await self._sub_agent_tools.cleanup()
            self._sub_agent_tools = None
        self._skills_provider = None
        # Controllers are gone — drop any engine-side tracking so a
        # subsequent ``start()`` (e.g. session restore) begins with an
        # empty paused-invocation set rather than ids pointing at
        # destroyed controllers.
        self._paused_sub_agents.clear()
        if self._mcp_adapter is not None:
            await self._mcp_adapter.disconnect_all()
            self._mcp_adapter = None
        if close_mcp_cache:
            await self._mcp_cache.close_all()
            self._mcp_cache = MCPConnectionCache()
        if self._executor is not None:
            await self._executor.close()
            self._executor = None
        if self._agent is not None:
            await self._agent.__aexit__(None, None, None)
            self._agent = None
        if release_session_lock:
            self._active_session_guard.release()

    # --- Event handlers ---

    @property
    def mutation_tracker(self) -> MutationTracker | None:
        """Live mutation tracker, or ``None`` if no session is active.

        Exposed read-only so the TUI (e.g. the rollback modal) can build
        per-turn diff views directly from in-memory state without a JSON
        round-trip through ``session.json``.  This matters during
        interrupted / failed runs where the tracker has up-to-date turn
        info that hasn't been persisted yet.
        """
        return self._mutation_tracker

    @property
    def mutation_coordinator(self) -> MutationCoordinator | None:
        """Cross-session mutation coordinator, or ``None`` when disabled."""
        return self._mutation_coordinator

    @property
    def todo_tracker(self) -> TodoTracker | None:
        """Live session todo tracker, or ``None`` if no session is active.

        Exposed read-only so hosts (e.g. the ACP plan-update sender) can read
        the current list via the sync ``snapshot()`` without touching the
        private attribute.
        """
        return self._todo_tracker

    async def refresh_mutation_attribution(self, *, force: bool = False) -> bool:
        """Reclassify the live mutation log against peer registry claims.

        The engine-level read-side entry point for cross-session
        coordination: display
        surfaces call it before building net summaries; the rollback
        path calls it with ``force=True`` as the authoritative last
        check.  Returns True when any row changed — the change is
        persisted via the normal session-state save so serialized
        consumers (TUI /diff) observe it too.
        """
        coordinator = self._mutation_coordinator
        tracker = self._mutation_tracker
        if coordinator is None or tracker is None:
            return False
        loop = asyncio.get_running_loop()
        try:
            changed = await loop.run_in_executor(
                None,
                lambda: coordinator.reclassify(tracker, force=force, fallback_root=self._workspace_cwd()),
            )
        except Exception:
            logger.debug("Mutation attribution refresh failed", exc_info=True)
            return False
        # The coordinator's flag, not this call's return value: a
        # finalize-time reclassify may have changed rows earlier with no
        # saver, and its signature update makes this very call report
        # "unchanged".  While a run is active the save must not happen here:
        # a primary save deletes the recovery sidecar, which mid-turn is the
        # only durable copy of committed-but-unmerged tool exchanges.  The
        # flag stays set, and the turn-end save persists the reclassification.
        if (
            not self.is_running
            and self._mutation_coordinator is not None
            and self._mutation_coordinator.consume_unsaved_reclassification()
        ):
            await self._save_current_session()
        return changed

    async def _save_current_session(self, *, raise_on_error: bool = False) -> bool:
        """Auto-save the current session state to disk."""
        # Drain any in-flight recovery write first.  ``save_session`` deletes the
        # sidecar structurally; flushing first guarantees that delete is the last
        # write (no stale sidecar resurrected after a clean turn) and makes the
        # final checkpoint durable even when the trailing save is suppressed.
        await self._flush_recovery_checkpoint()
        if self._executor is None or self._suppress_save:
            return False
        # Serialize mutation tracker into session state
        if self._mutation_tracker is not None:
            self._executor.history_state["chrys_mutations"] = self._mutation_tracker.serialize()
        baseline = self._workspace_change_tracker.serialize()
        if baseline is not None:
            self._executor.history_state["chrys_workspace_baseline"] = baseline
        else:
            self._executor.history_state.pop("chrys_workspace_baseline", None)
        # Session todo list — reads the TRACKER (not prior state): set when
        # non-empty, pop otherwise (empty ≡ absent).
        todos = self._todo_tracker.serialize() if self._todo_tracker is not None else []
        if todos:
            self._executor.history_state["chrys_todos"] = todos
        else:
            self._executor.history_state.pop("chrys_todos", None)
        self._executor.history_state.update(self._runtime_meta.to_state_dict())
        # Persist the Phase 4 LAST_WORDS note so a post-restart resume can
        # re-inject it — the compacted turn's tool-call history is already
        # dropped from ``messages``, and the note is its only replacement.
        if self._reminder_middleware is not None:
            last_words = self._reminder_middleware.get_last_words()
            if last_words:
                self._executor.history_state["last_words"] = last_words
            else:
                self._executor.history_state.pop("last_words", None)
            manifest = self._reminder_middleware.get_last_words_manifest()
            if manifest:
                self._executor.history_state["last_words_manifest"] = manifest
            else:
                self._executor.history_state.pop("last_words_manifest", None)
            breaker = self._reminder_middleware.get_last_words_breaker_state()
            if breaker:
                self._executor.history_state["last_words_breaker"] = breaker
            else:
                self._executor.history_state.pop("last_words_breaker", None)
            catalog_pointer_record_count = self._reminder_middleware.get_catalog_pointer_record_count_state()
            if catalog_pointer_record_count is not None:
                self._executor.history_state[CATALOG_POINTER_RECORD_COUNT_STATE_KEY] = catalog_pointer_record_count
            else:
                self._executor.history_state.pop(CATALOG_POINTER_RECORD_COUNT_STATE_KEY, None)
        else:
            # Note + manifest are erasure-protected while middleware is
            # temporarily unavailable. The breaker is deliberately not:
            # failed-attempt truth must be written unconditionally.
            self._executor.history_state.pop("last_words_breaker", None)
        if self._active_profile is None:
            service_session_id = None
        elif self._active_profile.provider == "openai" and self._active_profile.api_style == API_STYLE_RESPONSES:
            service_session_id = (
                self._executor.service_session_id
                if self._executor.service_session_storage_enabled
                and not self._executor.run_failed
                and not self._executor.was_interrupted
                else ""
            )
        else:
            service_session_id = ""
        # Every persisted item carries its analytics id before the save that
        # first persists it; items created on the normal paths already do.
        ensure_history_item_ids(self._executor.history_state.get("messages", ()))
        saved = await self._persistence.save_session(
            self._session_id,
            self._executor.history_state,
            agent_profile_name=self._agent_profile.name if self._agent_profile else "",
            agent_display_name=self._agent_profile.display_name if self._agent_profile else "",
            agent_profile_id=self._agent_profile.id if self._agent_profile else "",
            agent_profile_fingerprint=self._agent_profile_fingerprint,
            model_profile_fingerprint=self._model_profile_fingerprint,
            workspace=self._workspace,
            model_profile=self._active_profile,
            service_session_id=service_session_id,
            raise_on_error=raise_on_error,
        )
        if saved and self._session_id is not None:
            from chrys.foundation.observability.sink import get_otel_sink

            otel_sink = get_otel_sink()
            if otel_sink is not None:
                otel_sink.flush_pending(self._session_id)
        if saved and self._sub_agent_tools is not None:
            self._sub_agent_tools.finalize_pending_cleanups()
        if self._session_id is not None:
            self._recovered_from_sidecar = await self._persistence.recovery_session_wins(self._session_id)
        return saved

    async def _save_recovery_checkpoint(self) -> None:
        """Snapshot an interrupted-form sidecar for crash recovery at an LLM boundary.

        The snapshot is built synchronously (an atomic view of live state), but the
        disk write is handed to a background task so the agent's LLM round trip never
        waits on the sidecar's fsync + file lock — that I/O is slow on Windows and
        would otherwise stall every tool-loop iteration.
        """
        if self._persistence.state_store is None:
            return
        try:
            state = self._build_recovery_snapshot()
        except Exception:
            logger.warning("Failed to build recovery checkpoint for %s", self._session_id, exc_info=True)
            return
        if state is None or not has_real_messages(state):
            return
        # Coalesce bursts: keep only the newest snapshot and run a single writer.
        self._recovery_snapshot_seq += 1
        self._pending_recovery_state = (self._recovery_snapshot_seq, state)
        if self._recovery_write_task is None or self._recovery_write_task.done():
            self._recovery_write_task = asyncio.create_task(self._drain_recovery_checkpoints())

    def _build_recovery_snapshot(self) -> dict[str, Any] | None:
        """Build the current interrupted-form snapshot from engine-owned state."""
        if self._session_id is None or self._executor is None or self._loop_recorder is None:
            return None
        current_input = self._turns.current_input
        return session_checkpoint.build_recovery_state(
            self._executor.history_state,
            self._loop_recorder,
            mutation_tracker=self._mutation_tracker,
            runtime_meta=self._runtime_meta,
            user_text=current_input.text,
            user_contents=current_input.contents,
            user_created_at=current_input.created_at,
            user_kind=current_input.kind,
            consumed_injections=list(self._consumed_injections),
            insert_index=self._turn_state.history_start_index,
            last_words=self._reminder_middleware.get_last_words() if self._reminder_middleware is not None else None,
            last_words_manifest=(
                self._reminder_middleware.get_last_words_manifest() if self._reminder_middleware is not None else None
            ),
            last_words_breaker=(
                self._reminder_middleware.get_last_words_breaker_state()
                if self._reminder_middleware is not None
                else None
            ),
            catalog_pointer_record_count=(
                self._reminder_middleware.get_catalog_pointer_record_count_state()
                if self._reminder_middleware is not None
                else None
            ),
            todos=self._todo_tracker.serialize() if self._todo_tracker is not None else None,
        )

    async def persist_recovery_now(self) -> bool:
        """Strictly persist the newest recovery snapshot after draining older writes."""
        if self._persistence.state_store is None:
            return False
        await self._flush_recovery_checkpoint()
        state = self._build_recovery_snapshot()
        if state is None or not has_real_messages(state):
            return False
        self._recovery_snapshot_seq += 1
        seq = self._recovery_snapshot_seq
        # Acquire before spawning the writer so no newer background checkpoint
        # can overtake this snapshot. The child task owns release: shielding it
        # keeps the underlying ``to_thread`` write ordered even if the Phase-4
        # caller is cancelled while persistence is in flight.
        await self._recovery_persistence_lock.acquire()
        try:
            task = asyncio.create_task(self._persist_recovery_snapshot_strict(seq, state))
        except BaseException:
            self._recovery_persistence_lock.release()
            raise
        self._strict_recovery_write_tasks.add(task)
        task.add_done_callback(self._strict_recovery_write_done)
        return await asyncio.shield(task)

    async def persist_recovery_barrier(self) -> RecoveryPersistOutcome:
        """Strictly persist the current recovery snapshot with a typed outcome."""
        if self._persistence.state_store is None:
            return RecoveryPersistOutcome.UNCONFIGURED
        try:
            state = self._build_recovery_snapshot()
        except Exception:
            logger.debug("Failed to build strict recovery barrier for %s", self._session_id, exc_info=True)
            return RecoveryPersistOutcome.FAILED
        if state is None or not has_real_messages(state):
            return RecoveryPersistOutcome.NOTHING_TO_PERSIST

        self._recovery_snapshot_seq += 1
        seq = self._recovery_snapshot_seq
        await self._recovery_persistence_lock.acquire()
        try:
            task = asyncio.create_task(self._persist_recovery_snapshot_strict(seq, state))
        except BaseException:
            self._recovery_persistence_lock.release()
            raise
        self._strict_recovery_write_tasks.add(task)
        task.add_done_callback(self._strict_recovery_write_done)
        try:
            persisted = await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Strict recovery barrier failed for %s", self._session_id, exc_info=True)
            return RecoveryPersistOutcome.FAILED
        return RecoveryPersistOutcome.PERSISTED if persisted else RecoveryPersistOutcome.NOTHING_TO_PERSIST

    async def _persist_recovery_snapshot_strict(self, seq: int, state: dict[str, Any]) -> bool:
        """Write one strict snapshot while owning the pre-acquired ordering lock."""
        try:
            if seq <= self._recovery_persisted_seq:
                return True
            persisted = await self._persistence.save_recovery_session_strict(
                self._session_id,
                state,
                agent_profile_name=self._agent_profile.name if self._agent_profile else "",
                agent_display_name=self._agent_profile.display_name if self._agent_profile else "",
                agent_profile_id=self._agent_profile.id if self._agent_profile else "",
                agent_profile_fingerprint=self._agent_profile_fingerprint,
                model_profile_fingerprint=self._model_profile_fingerprint,
                workspace=self._workspace,
                model_profile=self._active_profile,
            )
            if persisted:
                self._recovery_persisted_seq = seq
            return persisted
        finally:
            self._recovery_persistence_lock.release()

    def _strict_recovery_write_done(self, task: asyncio.Task[bool]) -> None:
        """Retire a strict writer while observing failures after caller cancellation."""
        self._strict_recovery_write_tasks.discard(task)
        if task.cancelled():
            return
        with contextlib.suppress(Exception):
            task.exception()

    async def _drain_recovery_checkpoints(self) -> None:
        """Write queued recovery snapshots to disk, newest-wins, one at a time.

        Runs independently of the run task so a cancelled/hung turn (graceful
        shutdown timeout) still flushes its last checkpoint to disk.
        """
        while self._pending_recovery_state is not None:
            seq, state = self._pending_recovery_state
            self._pending_recovery_state = None
            try:
                async with self._recovery_persistence_lock:
                    if seq <= self._recovery_persisted_seq:
                        continue
                    await self._persistence.save_recovery_session(
                        self._session_id,
                        state,
                        agent_profile_name=self._agent_profile.name if self._agent_profile else "",
                        agent_display_name=self._agent_profile.display_name if self._agent_profile else "",
                        agent_profile_id=self._agent_profile.id if self._agent_profile else "",
                        agent_profile_fingerprint=self._agent_profile_fingerprint,
                        model_profile_fingerprint=self._model_profile_fingerprint,
                        workspace=self._workspace,
                        model_profile=self._active_profile,
                    )
                    self._recovery_persisted_seq = seq
            except Exception:
                logger.warning("Failed to save recovery checkpoint for %s", self._session_id, exc_info=True)

    async def _flush_recovery_checkpoint(self) -> None:
        """Await all in-flight recovery writes so the sidecar is durable and ordered.

        Called where the run loop has ended (post-run save, shutdown) and —
        mid-run — after an injection is consumed (its only durable copy until
        finalization is the sidecar).  A checkpoint queued concurrently while
        we drain extends the writer loop, never strands it: the task exits
        only when nothing is pending, and newest-wins coalescing keeps the
        await sound for the injection caller (every later snapshot also
        carries the consumed injection).

        ``asyncio.shield`` keeps run-task cancellation (the graceful-shutdown
        timeout cancels the run task while it is parked here) from propagating
        into the writer and aborting it mid-write: the run task unwinds, but the
        writer survives so the trailing shutdown flush can still drain it.
        """
        while True:
            tasks: list[asyncio.Task[Any]] = [task for task in self._strict_recovery_write_tasks if not task.done()]
            background = self._recovery_write_task
            if background is not None and not background.done():
                tasks.append(background)
            if not tasks:
                return
            await asyncio.shield(asyncio.gather(*tasks, return_exceptions=True))

    def _reset_spill_quota(self) -> None:
        """Start a fresh engine-owned spill ledger for a newly opened session."""
        self._spill_quota = SpillQuota()

    # --- Rollback snapshots -------------------------------------------------
    #
    # Before each new user turn begins, the engine copies the current
    # ``session.json`` to ``{session_dir}/snapshots/turn_{N}.json`` where
    # ``N`` is the turn about to start.  Rolling back to keep turn K means
    # swapping the snapshot whose payload has ``turn_counter == K`` back in
    # as ``session.json`` and reloading.  ``K == 0`` is the pre-session
    # welcome state, so it deletes the session file instead of swapping.

    def _write_rollback_snapshot(self) -> None:
        """Copy the current ``session.json`` to ``snapshots/turn_N.json``."""
        rollback_controller.write_snapshot(self)

    def _capture_rollback_snapshot_writer(self) -> Callable[[], None]:
        """Freeze rollback metadata before dispatching snapshot I/O to a worker."""
        return rollback_controller.capture_snapshot_writer(self)

    def turn_prompt_previews(self) -> dict[int, str]:
        """Return ``{turn_number: first_user_prompt}`` for every known turn."""
        return rollback_controller.turn_prompt_previews(self)

    def first_rolled_back_user_text(self, target_turn: int) -> str:
        """Return the first user prompt discarded by rollback to ``target_turn``."""
        return rollback_controller.first_rolled_back_user_text(self, target_turn)

    def available_rollback_turns(self) -> list[int]:
        """Public accessor for :meth:`_available_rollback_turns`.

        Used by the TUI to decide whether ``/rollback`` should show the
        modal at all, and to populate the turn picker.
        """
        return self._available_rollback_turns()

    @property
    def current_turn_number(self) -> int:
        """Current engine turn counter.

        Tracks the most-recently-started turn (incremented by the turn
        runner at the start of each run, restored from ``turn_counter``
        on session reload).  Equal to the last
        completed turn index while the engine is idle between turns.

        Exposed so the TUI can label the "you are here" entry in the
        rollback picker independently of which snapshots happen to
        exist on disk — otherwise a session with all snapshots
        deleted would show ``Turn 1 (Current)`` instead of the real
        current-turn index.
        """
        return self._turn_number

    def _available_rollback_turns(self) -> list[int]:
        """Return turn-counts that can be targeted by a rollback."""
        return rollback_controller.available_rollback_turns(self)

    async def _on_user_rollback(self, event: UserRollback) -> None:
        """Handle a rollback request."""
        await rollback_controller.on_user_rollback(
            self,
            event,
            atomic_copy_file=atomic_copy_file,
            lock_timeout_seconds=SESSION_WRITE_LOCK_TIMEOUT_SECONDS,
        )

    def _reset_for_restart(self, session_id: str | None) -> None:
        """Reset per-session engine state in preparation for a fresh ``start()``."""
        session_lifecycle.reset_for_restart(self, session_id)

    async def reset_after_failed_startup_restore(self) -> None:
        """Return a partial startup restore to a clean, unlocked baseline."""
        await session_lifecycle.reset_after_failed_startup_restore(self)

    async def _reset_session_to_welcome(
        self,
        session_id: str,
        *,
        write_lock_held: bool = False,
        after_delete: Callable[[], Awaitable[None]] | None = None,
        before_restart: Callable[[], None] | None = None,
    ) -> bool:
        """Delete session.json + snapshots, reload to a welcome state."""
        return await session_lifecycle.reset_session_to_welcome(
            self,
            session_id,
            write_lock_held=write_lock_held,
            after_delete=after_delete,
            before_restart=before_restart,
        )

    async def _on_user_message(self, event: UserMessage) -> None:
        """Handle a user message by running the executor as an async task."""
        await self._turns.on_user_message(event)

    async def _run_and_save(
        self,
        text: str,
        created_at: datetime | str | None = None,
        contents: list[Any] | None = None,
        *,
        run_scope: CurrentRunScope | None = None,
        injection_window: CurrentRunInjectionWindow | None = None,
        admission_preparation: PreparationTrace | None = None,
    ) -> None:
        """Execute the agent and auto-save the session afterward."""
        await self._turns.run_fresh(
            text,
            created_at=created_at,
            contents=contents,
            run_scope=run_scope,
            injection_window=injection_window,
            admission_preparation=admission_preparation,
        )

    async def _retry_and_save(
        self,
        additional_text: str = "",
        created_at: datetime | str | None = None,
        *,
        run_scope: CurrentRunScope | None = None,
        injection_window: CurrentRunInjectionWindow | None = None,
        admission_preparation: PreparationTrace | None = None,
    ) -> None:
        """Resume the agent from current state and auto-save afterward.

        When *additional_text* is non-empty, the executor uses it as the
        mid-turn continuation prompt (instead of the placeholder
        ``"continue"``) and preserves it in history as a real user turn.
        """
        await self._turns.run_retry(
            additional_text,
            created_at=created_at,
            run_scope=run_scope,
            injection_window=injection_window,
            admission_preparation=admission_preparation,
        )

    async def _on_route_override(self, event: RouteOverride) -> None:
        """Hold a routing override for the next message, or downgrade this turn.

        ``/quick`` during a long-horizon turn's preparation means "not this
        one", not "not the next one": a user watching a campaign spin up has no
        other way to stop it, and stopping it is exactly what the workflow's
        own stop path already does — P0 is promoted and nothing is delegated.
        """
        self._route_override = event
        workflow = self._requirement_clarification_workflow
        if event.track == "standard" and workflow is not None and self._last_route is not None:
            self._route_override = None
            await workflow.request_stop()

    async def _post_run(self) -> None:
        """Unified post-execution fixup for both run and retry paths."""
        await self._turns.finalize_current_run()
        if self._memory_watcher is not None:
            self._memory_watcher.touch()

    def _start_memory_watcher(self) -> None:
        """Arm the idle writeback timer, unless memory is unconfigured or off."""
        if self._memory_watcher is not None or not self._memory_configured():
            return
        watcher = MemoryWritebackWatcher(
            idle_seconds=self._settings.memory_writeback_idle_seconds,
            on_flush=self._flush_memory_writeback,
            is_busy=lambda: self.is_turn_lifecycle_active,
        )
        watcher.start()
        self._memory_watcher = watcher

    def _memory_configured(self) -> bool:
        """Return whether this machine has a reachable graph configured at all."""
        return memory_mcp_server_config(self._settings) is not None

    async def _flush_memory_writeback(self, reason: str) -> None:
        """Deposit every turn past the watermark and persist the new mark."""
        session_dir, session_id = self._session_dir, self._session_id
        if session_dir is None or session_id is None:
            return
        # Deposit reads the persisted session, so it has to be current first.
        await self._save_current_session()
        repo = Path(self._workspace.primary_cwd).name if self._workspace is not None else "general"
        outcome = await asyncio.to_thread(
            deposit_pending_turns,
            session_dir / SESSION_FILE_NAME,
            watermark=self._runtime_meta.memory_deposit_watermark,
            repo=repo,
            source_prefix=f"chrys-session:{session_id}",
        )
        if outcome.watermark != self._runtime_meta.memory_deposit_watermark:
            self._runtime_meta.memory_deposit_watermark = outcome.watermark
            await self._save_current_session()
        await self._bus.publish(
            MemoryWritebackCompleted(
                session_id=session_id,
                reason=reason,
                deposited=len(outcome.deposited),
                failed_turn=outcome.failed,
                watermark=outcome.watermark,
            )
        )

    async def _on_user_interrupt(self, _event: UserInterrupt) -> None:
        """Handle user interrupt.

        Cascades to every live sub-agent controller first so paused
        sub-agents resolve via the cascade-abort branch (otherwise their
        ``pending_decision`` future would keep their ``_invoke``
        coroutine pinned forever and the subsequent task cancel below
        would leak).  Then sets the interrupt flag / cancels the parent
        task. It then binds cancellation to the exact pre-executor run task
        or interrupts the active executor; ``_run_and_save`` detects
        ``was_interrupted`` and rolls back history.
        """
        await self._turns.on_user_interrupt(_event)

    async def _on_sub_agent_retry(self, event: SubAgentRetryRequested) -> None:
        """Route a user's per-card Retry click to the owning controller."""
        await sub_agent_coordination.on_sub_agent_retry(self, event)

    async def _on_sub_agent_abort(self, event: SubAgentAbortRequested) -> None:
        """Route a user's per-card Abort click to the owning controller."""
        await sub_agent_coordination.on_sub_agent_abort(self, event)

    async def _on_sub_agent_paused(self, event: SubAgentPaused) -> None:
        """Track a newly paused sub-agent and drive FSM / marker.

        Idempotent on the paused set — if the same id arrives twice
        (controller re-publishes after retry exhaustion, for example)
        the FSM transition only fires on the 0→1 edge.

        Defensive FSM guard: if the parent run has already terminated
        (e.g. the bus is dispatching a stale pause event that was queued
        before the parent's task was cancelled and ``_post_run`` ran),
        drop the event silently rather than re-inserting a marker on top
        of an already-terminal ``interrupted``/``error`` marker.  The
        parent invariant ("parent run ends only after all sub-agents
        resolve") should make this unreachable, but defense in depth
        keeps history well-formed if that invariant ever breaks.
        """
        await sub_agent_coordination.on_sub_agent_paused(self, event)

    async def _on_sub_agent_unpaused(
        self,
        event: SubAgentResumed | SubAgentAborted | SubAgentCascadeAborted,
    ) -> None:
        """Common handler — retry/abort/cascade all remove the invocation from the paused set.

        FSM transitions only on the N→0 edge (last paused sub-agent
        resolved).  Marker is updated or stripped accordingly.
        """
        await sub_agent_coordination.on_sub_agent_unpaused(self, event)

    async def _on_user_retry(self, event: UserRetry) -> None:
        """Handle retry — resume from current state.

        When ``event.text`` is non-empty, it is forwarded to
        ``_retry_and_save`` (or stashed for the pending-retry path) and
        becomes the mid-turn continuation prompt, replacing the
        executor's default ``"continue"`` placeholder.
        """
        await self._turns.on_user_retry(event)

    async def _on_user_inject(self, event: UserInject) -> None:
        """Handle user injection (prompt inserted before next model call)."""
        await self._turns.on_user_inject(event)

    async def _on_user_inject_cancel(self, event: UserInjectCancel) -> None:
        """Handle user cancellation of a still-pending mid-run injection."""
        await self._turns.on_user_inject_cancel(event)

    def make_usage_event(self, *, session_id: str | None = None) -> UsageUpdate:
        """Public accessor for the current usage snapshot (used by frontends)."""
        return usage.make_usage_event(self, session_id=session_id)

    def _make_usage_event(self, *, session_id: str | None = None) -> UsageUpdate:
        """Build a UsageUpdate event — pct always computed from the active profile."""
        return usage.make_usage_event(self, session_id=session_id)

    def _publish_usage(
        self,
        total_tokens: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        local_tokens: int = 0,
        calibration_ratio: float = 1.0,
        system_overhead_tokens: int = 0,
        cache_hit_tokens: int | None = None,
        calibration_initialized: bool = False,
        use_local_context_estimate: bool = False,
    ) -> None:
        """Callback from UsageTrackingMiddleware — publish UsageUpdate event."""
        usage.publish_usage(
            self,
            total_tokens,
            input_tokens,
            output_tokens,
            local_tokens,
            calibration_ratio,
            system_overhead_tokens,
            cache_hit_tokens,
            calibration_initialized,
            use_local_context_estimate,
        )

    def _accumulate_sub_agent_usage(
        self,
        total_tokens: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
        local_tokens: int = 0,
        calibration_ratio: float = 1.0,
        system_overhead_tokens: int = 0,
        cache_hit_tokens: int | None = None,
        max_context_tokens: int | None = None,
        agent_profile: str = "",
        usage_source_id: str = "",
        authoritative_total: int | None = None,
        use_local_context_estimate: bool = False,
    ) -> None:
        """Callback from sub-agent UsageTrackingMiddleware — accumulate and publish live usage."""
        usage.accumulate_sub_agent_usage(
            self,
            total_tokens,
            input_tokens,
            output_tokens,
            local_tokens,
            calibration_ratio,
            system_overhead_tokens,
            cache_hit_tokens,
            max_context_tokens,
            agent_profile,
            usage_source_id,
            authoritative_total,
            use_local_context_estimate,
        )

    def _accumulate_side_call_usage(self, usage_details: Mapping[str, Any]) -> None:
        """Callback from Phase-4 LAST_WORDS generators — accumulate side-call spend."""
        usage.accumulate_side_call_usage(self, usage_details)

    async def _drain_usage_publishes(self) -> None:
        """Await all pending UsageUpdate publish tasks.

        Sub-agent ``_invoke`` calls this in its ``finally`` so the sub-agent's
        trailing UsageUpdate is delivered before the parent's ToolCallResult.
        Without it a slow subscriber can let the parent tool result overtake
        the sub-agent's last usage event.
        """
        tail = self._usage_publish_tail
        if tail is None:
            return
        await asyncio.gather(tail, return_exceptions=True)

    async def _publish_compress(self, info: CompressInfo) -> None:
        """Callback from ContextManagementProvider or force-compress — publish ContextCompressed event."""
        await usage.publish_compress(self, info)

    async def _publish_compaction(self, info: CompactionInfo) -> None:
        """Callback from UnifiedContextStrategy — publish ToolCompacted event."""
        await usage.publish_compaction(self, info)

    async def _publish_pre_compact(self, info: PreCompactInfo) -> None:
        """Callback from UnifiedContextStrategy — fire pre_compact hooks."""
        if self._hook_manager is None:
            return
        from chrys.service.hooks.events import HookEvent

        if not self._hook_manager.has_hooks_for(HookEvent.PRE_COMPACT):
            return
        profile_name = self._agent_profile.name if self._agent_profile is not None else ""
        await self._hook_manager.fire(
            HookEvent.PRE_COMPACT,
            {
                "session_id": self._session_id,
                "profile": profile_name,
                "cwd": self._workspace_cwd(),
                "trigger": info.trigger,
                "usage_pct": info.usage_pct,
                "tokens_before": info.tokens_before,
            },
            target_operation_id=info.trajectory_operation_id,
        )

    async def _on_set_approval_mode(self, event: SetApprovalMode) -> None:
        """Update the active approval mode on the running middleware."""
        await engine_controls.on_set_approval_mode(
            self,
            event,
            persist_approval_mode_fn=persist_approval_mode,
        )

    async def _on_set_model_profile(self, event: SetModelProfile) -> None:
        """Switch the active model profile for this session only."""
        await engine_controls.on_set_model_profile(self, event)

    async def _on_profile_switch(self, event: AgentProfileSwitch) -> None:
        """Handle agent profile switch — preserves conversation history."""
        await engine_controls.on_profile_switch(self, event)

    def pin_ask_user_timeout(self) -> None:
        """Mark ``ask_user_timeout_seconds`` as caller-owned across reloads.

        Callers that inject the timeout out-of-band (ACP sets it to ``None`` at
        launch via ``dataclasses.replace``) call this so a later ``SettingsReload``
        keeps the live value instead of reverting to the env default.
        """
        self._ask_user_timeout_pinned = True

    def pin_model_profile(self) -> None:
        """Mark the model selection as caller-owned across reloads.

        Headless ``--model`` lives only in this host's settings — it never
        parks the choice in the process pointer the way the TUI does — so
        without the pin one ``SettingsReload`` would silently revert the run
        to the global default.
        """
        self._model_profile_pinned = True

    async def _on_settings_reload(self, _event: SettingsReload) -> None:
        """Handle settings reload — recreate Settings from env and rebuild agent."""
        await engine_controls.on_settings_reload(self, _event)

    async def _on_workspace_change(self, event: WorkspaceChange) -> None:
        """Handle workspace/cwd change — rebuild agent with new workspace."""
        await engine_controls.on_workspace_change(self, event)

    def _cleanup_empty_session_dir(self) -> None:
        """Remove the current session directory if it was never saved."""
        session_lifecycle.cleanup_empty_session_dir(self)

    async def _on_new_session(self, _event: SessionNew) -> None:
        """Handle new session request — save current, then start fresh."""
        await session_lifecycle.on_new_session(self, _event)

    async def _on_session_restore(self, event: SessionRestore) -> None:
        """Handle session restore — load saved session state."""
        await session_lifecycle.on_session_restore(self, event)

    async def _on_session_delete(self, event: SessionDelete) -> None:
        """Handle session deletion."""
        await session_lifecycle.on_session_delete(self, event)

    async def _on_session_clear(self, event: SessionClear) -> None:
        """Handle clear: delete the active session and start fresh as one fenced transition."""
        await session_lifecycle.on_session_clear(self, event)

    async def _on_session_fork(self, event: SessionFork) -> None:
        """Handle session fork requests for the active saved session."""
        try:
            await asyncio.wait_for(
                self._wait_for_agent_load_idle(),
                timeout=_SESSION_FORK_AGENT_LOAD_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            await self._publish_session_fork_error(
                "session_fork_busy",
                "Cannot fork while the agent is still loading.",
                session_id=event.session_id or self._session_id or "",
                display_message=_FORK_AGENT_LOADING.bind(),
            )
            return

        active_session_id = self._session_id
        state_store = self._persistence.state_store
        if active_session_id is None or self._executor is None:
            await self._publish_session_fork_error(
                "session_fork_not_ready",
                "Cannot fork before a session is ready.",
                session_id=active_session_id or event.session_id,
                display_message=_FORK_NOT_READY.bind(),
            )
            return
        if event.session_id != active_session_id:
            await self._publish_session_fork_error(
                "session_fork_stale",
                "Cannot fork because the active session changed.",
                session_id=event.session_id or active_session_id,
                display_message=_FORK_SESSION_CHANGED.bind(),
            )
            return
        if self.is_turn_active:
            await self._publish_session_fork_error(
                "session_fork_busy",
                "Cannot fork while a turn is running.",
                session_id=active_session_id,
                display_message=_FORK_TURN_ACTIVE.bind(),
            )
            return
        if state_store is None:
            await self._publish_session_fork_error(
                "session_fork_failed",
                "State store not configured.",
                session_id=active_session_id,
                display_message=_FORK_STATE_STORE_MISSING.bind(),
            )
            return
        if self._suppress_save:
            await self._publish_session_fork_error(
                "session_fork_busy",
                "Cannot fork while session state is being restored.",
                session_id=active_session_id,
                display_message=_FORK_RESTORING.bind(),
            )
            return
        if not self._active_session_guard.owns(active_session_id):
            await self._publish_session_fork_error(
                "session_fork_busy",
                "Cannot fork because this window does not own the active session lock.",
                session_id=active_session_id,
                display_message=_FORK_LOCK_NOT_OWNED.bind(),
            )
            return
        if not has_real_messages(self._executor.history_state):
            await self._publish_session_fork_error(
                "session_fork_empty",
                "Cannot fork an empty session.",
                session_id=active_session_id,
                display_message=_FORK_EMPTY_SESSION.bind(),
            )
            return

        try:
            await self._save_current_session(raise_on_error=True)
            fresh_state = await state_store.load_session(active_session_id)
        except TimeoutError as exc:
            logger.warning("Timed out preparing session %s for fork", active_session_id, exc_info=True)
            await self._publish_session_fork_error(
                "session_fork_busy",
                f"Timed out preparing session for fork: {exc}",
                session_id=active_session_id,
                display_message=_FORK_PREPARE_TIMEOUT.bind(detail=DisplayBlock(str(exc))),
            )
            return
        except Exception as exc:
            logger.warning("Failed to prepare session %s for fork", active_session_id, exc_info=True)
            await self._publish_session_fork_error(
                "session_fork_failed",
                f"Failed to prepare session for fork: {exc}",
                session_id=active_session_id,
                display_message=_FORK_PREPARE_FAILED.bind(detail=DisplayBlock(str(exc))),
            )
            return
        if fresh_state is None or not has_real_messages(fresh_state):
            await self._publish_session_fork_error(
                "session_fork_empty",
                "Cannot fork an empty session.",
                session_id=active_session_id,
                display_message=_FORK_EMPTY_SESSION.bind(),
            )
            return

        try:
            new_session_id = await asyncio.to_thread(state_store.fork_session, active_session_id)
        except SessionNotFoundError:
            await self._publish_session_fork_error(
                "session_fork_not_found",
                f"Session '{active_session_id}' not found.",
                session_id=active_session_id,
                display_message=_FORK_SESSION_NOT_FOUND.bind(session_id=active_session_id),
            )
            return
        except SessionForkError as exc:
            logger.warning("Failed to fork session %s", active_session_id, exc_info=True)
            await self._publish_session_fork_error(
                "session_fork_failed",
                f"Failed to fork session: {exc}",
                session_id=active_session_id,
                display_message=_FORK_FAILED.bind(detail=DisplayBlock(str(exc))),
            )
            return

        # The fork copies the conversation, not the parent's trajectory; it
        # gets its own closed opening runtime pointing back at the parent.
        await self._trajectory_recorder.fork(
            origin_session_id=active_session_id,
            fork_session_id=new_session_id,
            fork_session_dir=self._session_dir_for(new_session_id),
            fork_write_lock_path=self._session_write_lock_path(new_session_id),
            session_start_info=lambda: agent_lifecycle.trajectory_session_start_info(self),
        )
        await self._bus.publish(
            SessionForked(
                session_id=active_session_id,
                parent_session_id=active_session_id,
                new_session_id=new_session_id,
            )
        )

    async def _publish_session_fork_error(
        self, code: str, message: str, *, session_id: str, display_message: MessageRef | None = None
    ) -> None:
        await self._bus.publish(
            Error(code=code, message=message, session_id=session_id, display_message=display_message)
        )
