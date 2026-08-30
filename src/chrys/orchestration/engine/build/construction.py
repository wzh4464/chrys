# Copyright (c) 2026 Chrys. All rights reserved.

"""Engine-internal companion module for ``AgentEngine`` build and restart orchestration."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.errors import clean_error_message
from chrys.foundation.events.types import (
    AgentLoadFailed,
    AgentLoadFinished,
    AgentLoadProgress,
    AgentLoadStarted,
    AgentMessage,
    ApprovalModeUpdated,
    Error,
    ProfileSwitched,
    SessionReady,
    UserInjectResult,
    Warning,
    WorkspaceUpdated,
)
from chrys.foundation.i18n import DisplayBlock, msg
from chrys.foundation.models.workspace import Workspace
from chrys.foundation.platform import get_platform, safe_getcwd
from chrys.foundation.recovery import RecoveryPersistOutcome
from chrys.foundation.trajectory.event_types import ProfileKind
from chrys.orchestration.engine.build.builder import AgentBuildResult
from chrys.orchestration.engine.state.machine import Trigger
from chrys.orchestration.engine.trajectory import TrajectoryRecorder
from chrys.service import usage
from chrys.service.agent_middleware import IntermediateTextBuffer
from chrys.service.agent_middleware.injection import InjectionMiddleware
from chrys.service.agent_middleware.system_reminder import CATALOG_POINTER_RECORD_COUNT_STATE_KEY
from chrys.service.mutations.coordination import ATTRIBUTION_DIR_NAME, MutationCoordinator
from chrys.service.mutations.store import SnapshotPolicy, SnapshotStore
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker
from chrys.service.profiles.models.schema import API_STYLE_RESPONSES
from chrys.service.session.runtime_metadata import SessionRuntimeMetadata
from chrys.service.todos.tracker import TodoTracker
from chrys.service.trajectory.preparation import PreparationOutcome
from chrys.service.trajectory.session import SessionStartInfo

if TYPE_CHECKING:
    from pathlib import Path

    from chrys.foundation.config.settings import Settings
    from chrys.foundation.config.settings_store import LoadedSettings, SettingsHandle
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import AgentRuntimeDetails, UsageUpdate
    from chrys.foundation.models.session_env import SessionEnvironment
    from chrys.kernel import Agent, LoopRecorder
    from chrys.orchestration.engine.executor import Executor
    from chrys.orchestration.engine.run.turn_state import TurnRuntimeState
    from chrys.orchestration.engine.state.machine import EngineStateMachine
    from chrys.orchestration.sub_agents.tools import SubAgentTools
    from chrys.service.agent_middleware.injection import ConsumedInjection
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
    from chrys.service.approval.judge import ApprovalJudge
    from chrys.service.approval.policy import ApprovalMode
    from chrys.service.approval.turn_context import TurnContextHolder
    from chrys.service.context.compaction import CompactionInfo, CompressInfo, PreCompactInfo
    from chrys.service.context.compaction.spill import SpillQuota
    from chrys.service.hooks.manager import HookManager
    from chrys.service.mcp.adapter import MCPAdapter
    from chrys.service.mcp.cache import MCPConnectionCache
    from chrys.service.profiles.agents.registry import AgentProfileRegistry
    from chrys.service.profiles.agents.schema import AgentProfile
    from chrys.service.profiles.models.registry import ModelProfileRegistry
    from chrys.service.profiles.models.schema import ModelProfile
    from chrys.service.session.history import SessionHistoryManager
    from chrys.service.session.persistence import SessionPersistence
    from chrys.service.skills.provider import ChrysSkillsProvider
    from chrys.service.state.locks import ActiveSessionGuard


logger = logging.getLogger(__name__)

_CONSTRUCTION_GLOBAL_HOOKS_INVALID = msg(
    "construction.global_hooks_invalid",
    fallback="Global hooks config could not be loaded: {detail}.  Global hooks disabled.",
    multiline=True,
)
_CONSTRUCTION_PROJECT_HOOKS_INVALID = msg(
    "construction.project_hooks_invalid",
    fallback="Project hooks config could not be loaded: {detail}. Project hooks disabled; global hooks unaffected.",
    multiline=True,
)
_CONSTRUCTION_SERVICE_SESSION_INCOMPATIBLE = msg(
    "construction.service_session_incompatible",
    fallback=(
        "The previous OpenAI Responses service session is not compatible with the active agent profile, workspace, "
        "model profile, service endpoint, or storage is disabled. {app_name} will continue from local history only."
    ),
)
_CONSTRUCTION_TRAJECTORY_ACTIVATION_FAILED = msg(
    "construction.trajectory_activation_failed",
    fallback="Trajectory recording could not start and has been disabled for this session.",
)


BuildAgentFn = Callable[..., Awaitable[AgentBuildResult]]
SetCurrentEngine = Callable[[], None]


@dataclass(frozen=True, slots=True)
class StagedBuild:
    """What a rebuild reads *instead of* live engine state, committed whole.

    A rebuild used to stage its inputs by assigning them to the engine before
    the build and assigning them back on failure. Every await inside the build
    made that a lie: any concurrent task reading the engine saw an uncommitted,
    possibly-about-to-roll-back value. The staged inputs travel here as
    parameters instead, the build reads only these, and
    :func:`build_agent` installs all of them in the same breath as the
    executor they configured — so live state either describes the old build
    entirely or the new one entirely, never a mixture.

    ``loaded`` carries settings and provenance as the one value they already
    are; the agent profile, the workspace, the hook manager and the
    mutation-coordinator target are the other four things a rebuild may
    change. Six commitments, one commit point.
    """

    loaded: LoadedSettings
    agent_profile: AgentProfile
    """The profile the build is configured from, recorded by the commit.

    Recorded there rather than by the caller after the build returns: every
    await between the commit and the caller's next line can be cancelled, and
    a session running the new executor while still recording the old profile
    would save — and later restore — as something it is not.
    """

    workspace: Workspace | None
    hook_manager: HookManager | None
    mutation_coordinator: MutationCoordinator | None
    """The coordinator the new build runs with — a *candidate*, not installed.

    The current instance when it survives, ``None`` when the staged settings
    turn coordination off, or a fresh not-yet-installed instance when they
    turn it on. The old instance is closed only after the commit; a failed
    build closes the candidate instead (see :func:`build_agent`).
    """


def stage_build(
    engine: AgentLifecycleHost,
    *,
    loaded: LoadedSettings,
    agent_profile: AgentProfile,
    workspace: Workspace | None,
    hook_manager: HookManager | None,
) -> StagedBuild:
    """Assemble a rebuild's staged input, deriving the coordinator target.

    Derived here rather than passed by callers because the target is a pure
    function of the staged settings and the session: settings say off →
    ``None`` (the escape hatch must bite without a restart); an instance
    exists and settings say on → keep it; none exists and the session can
    host one → stage a fresh candidate, which stays uninstalled until the
    build commits.
    """
    coordinator = engine._mutation_coordinator
    if not loaded.settings.mutation_coordination:
        coordinator = None
    elif coordinator is None:
        session_dir = engine._session_dir if engine._persistence.state_store is not None else None
        if session_dir is not None and engine._session_id:
            # Registry root sits beside the session dirs so tests with a
            # relocated state store stay sandboxed automatically.
            coordinator = MutationCoordinator(
                registry_root=session_dir.parent / ATTRIBUTION_DIR_NAME,
                session_id=engine._session_id,
            )
    return StagedBuild(
        loaded=loaded,
        agent_profile=agent_profile,
        workspace=workspace,
        hook_manager=hook_manager,
        mutation_coordinator=coordinator,
    )


async def _close_coordinator(coordinator: MutationCoordinator, *, reason: str) -> None:
    """Stamp a coordinator's registry file closed, off-loop and best-effort."""
    try:
        await asyncio.get_running_loop().run_in_executor(None, coordinator.close)
    except Exception:
        logger.debug("Mutation coordinator close failed (%s)", reason, exc_info=True)


def _workspace_cwd(engine: AgentLifecycleHost) -> str:
    """Return the engine workspace cwd, falling back only for legacy unstarted engines."""
    if engine._workspace is not None:
        return engine._workspace.primary_cwd
    return safe_getcwd()


def _workspace_working_dirs(engine: AgentLifecycleHost) -> list[str]:
    """Return all workspace root paths for event payloads."""
    if engine._workspace is None:
        return []
    return [working_dir.path for working_dir in engine._workspace.working_dirs]


def _is_openai_responses_profile(profile: ModelProfile | None) -> bool:
    """Return True for profiles that use OpenAI Responses transport."""
    return profile is not None and profile.provider == "openai" and profile.api_style == API_STYLE_RESPONSES


def _responses_service_session_profiles_match(
    old_profile: ModelProfile | None, new_profile: ModelProfile | None
) -> bool:
    """Return True when two profiles can safely share a Responses service session id."""
    return (
        _is_openai_responses_profile(old_profile)
        and _is_openai_responses_profile(new_profile)
        and old_profile is not None
        and new_profile is not None
        and old_profile.model_id == new_profile.model_id
    )


def _workspace_signature(workspace: Workspace | None) -> tuple[str, tuple[str, ...]]:
    """Return the session-relevant workspace identity for service-session reuse."""
    if workspace is None:
        return ("", ())
    return (workspace.primary_cwd, tuple(working_dir.path for working_dir in workspace.working_dirs))


def trajectory_session_start_info(engine: AgentLifecycleHost) -> SessionStartInfo:
    """Resolved lazily, at the recorder's first event — after the build set the fingerprints."""
    return SessionStartInfo(
        primary_cwd=_workspace_primary_cwd(engine._workspace),
        agent_profile_fingerprint=engine._agent_profile_fingerprint,
        model_profile_fingerprint=engine._model_profile_fingerprint,
    )


def _workspace_primary_cwd(workspace: Workspace | None) -> str:
    """Return the workspace primary cwd used for project-scoped config."""
    return workspace.primary_cwd if workspace is not None else ""


def _can_reuse_responses_service_session(
    *,
    old_agent_profile_fingerprint: str,
    new_agent_profile_fingerprint: str,
    old_model_profile_fingerprint: str,
    new_model_profile_fingerprint: str,
    old_model_profile: ModelProfile | None,
    new_model_profile: ModelProfile | None,
    old_model_base_url: str,
    new_model_base_url: str,
    old_workspace: Workspace | None,
    new_workspace: Workspace | None,
    old_storage_enabled: bool,
    new_storage_enabled: bool,
) -> bool:
    """Return True when a Responses service session can safely survive a rebuild."""
    return bool(
        _responses_service_session_profiles_match(old_model_profile, new_model_profile)
        and old_agent_profile_fingerprint
        and old_agent_profile_fingerprint == new_agent_profile_fingerprint
        and old_model_profile_fingerprint
        and old_model_profile_fingerprint == new_model_profile_fingerprint
        and old_model_base_url
        and old_model_base_url == new_model_base_url
        and _workspace_signature(old_workspace) == _workspace_signature(new_workspace)
        and old_storage_enabled
        and new_storage_enabled
    )


class AgentLifecycleHost(Protocol):
    """Engine state needed by agent build/restart orchestration."""

    _agent_profile: AgentProfile | None
    _workspace: Workspace | None
    _session_id: str | None
    _active_session_guard: ActiveSessionGuard
    _bus: EventBus
    _persistence: SessionPersistence
    _history: SessionHistoryManager
    _executor: Executor | None
    _fsm: EngineStateMachine
    _sub_agent_tools: SubAgentTools | None
    _skills_provider: ChrysSkillsProvider | None
    _active_profile: ModelProfile | None
    _tool_names: list[str]
    _tool_kinds: dict[str, str]
    _skill_names: list[str]
    _memory_files: list[str]
    _agent_profile_fingerprint: str
    _model_profile_fingerprint: str
    _trajectory_recorder: TrajectoryRecorder
    _runtime_details: AgentRuntimeDetails
    _approval_mode: ApprovalMode
    _allow_user_interaction: bool
    _intermediate_texts: dict[int, str]
    _injection_notify_task: asyncio.Task[None] | None
    _mutation_tracker: MutationTracker | None
    _workspace_change_tracker: WorkspaceChangeTracker
    _mutation_coordinator: MutationCoordinator | None
    _todo_tracker: TodoTracker | None
    _model_registry: ModelProfileRegistry | None
    _agent: Agent | None
    _mcp_cache: MCPConnectionCache
    _mcp_adapter: MCPAdapter | None
    _agent_registry: AgentProfileRegistry | None
    _runtime: SessionEnvironment | None
    _injection: InjectionMiddleware | None
    _consumed_injections: list[ConsumedInjection]
    _loop_recorder: LoopRecorder | None
    _reminder_middleware: SystemReminderMiddleware | None
    _approval_judge: ApprovalJudge | None
    _hook_manager: HookManager | None
    _outbox_recovery_task: asyncio.Task[int] | None
    _runtime_meta: SessionRuntimeMetadata
    _turn_state: TurnRuntimeState
    _turn_context: TurnContextHolder
    _spill_quota: SpillQuota
    # UsageHost fields — required for ``usage.enqueue_usage_event`` calls below
    # and the ``_drain_usage_publishes`` hook.
    _usage_tasks: set[asyncio.Task[None]]
    _usage_publish_tail: asyncio.Task[None] | None
    _settings_handle: SettingsHandle

    @property
    def _settings(self) -> Settings:
        """Read-only: settings change through the handle, not the holder."""
        ...

    @property
    def _loaded_settings(self) -> LoadedSettings: ...

    async def _drain_usage_publishes(self) -> None: ...

    # UsageHost declares the public read accessor; required so this host
    # remains forwardable into ``usage.enqueue_usage_event``.
    @property
    def event_bus(self) -> EventBus: ...

    @property
    def _session_dir(self) -> Path | None: ...

    def _session_write_lock_path(self, session_id: str) -> Path | None: ...

    async def _subscribe_event_handlers(self) -> None: ...

    def _begin_agent_load(self) -> None: ...

    async def _build_agent(
        self, profile: AgentProfile, staged: StagedBuild, *, preserved_history: dict | None = None
    ) -> None: ...

    def _finish_agent_load(self) -> None: ...

    def _advance_build_generation(self) -> None: ...

    async def _publish_load_failed(self, *, operation: str, profile: AgentProfile, exc: Exception) -> None: ...

    async def _cleanup_replaced_build_resources(
        self,
        old_agent: Agent | None,
        old_sub_agent_tools: SubAgentTools | None,
        old_mcp_adapter: MCPAdapter | None,
    ) -> None: ...

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
    ) -> None: ...

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
    ) -> None: ...

    def _accumulate_side_call_usage(self, usage_details: Mapping[str, Any]) -> None: ...

    async def _publish_compaction(self, info: CompactionInfo) -> None: ...

    async def _publish_pre_compact(self, info: PreCompactInfo) -> None: ...

    async def _publish_compress(self, info: CompressInfo) -> None: ...

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
    ) -> None: ...

    async def _save_recovery_checkpoint(self) -> None: ...

    async def _flush_recovery_checkpoint(self) -> None: ...

    async def persist_recovery_now(self) -> bool: ...

    async def persist_recovery_barrier(self) -> RecoveryPersistOutcome: ...

    def _make_usage_event(self, *, session_id: str | None = None) -> UsageUpdate: ...


async def _build_hook_manager(
    engine: AgentLifecycleHost,
    *,
    project_root: str,
    project_hooks_enabled: bool = True,
) -> HookManager | None:
    """Load project/global hook configuration for *project_root*.

    The root is a parameter rather than a read of ``engine._workspace``
    because a staged rebuild derives it from the workspace it is about to
    install, which is deliberately not live yet. ``project_hooks_enabled``
    is likewise the staged ``project.hooks_enabled`` value: when off, only
    the global hooks file is consulted.
    """
    from pathlib import Path

    from chrys.service.hooks.loader import (
        HooksConfigError,
        load_hooks_dir,
        load_hooks_project,
        merge_hooks_files,
    )
    from chrys.service.hooks.manager import HookManager
    from chrys.service.hooks.schema import HooksFile

    config_dir = get_platform().config_dir
    root = Path(project_root)

    global_hooks: HooksFile | None = None
    try:
        global_hooks = load_hooks_dir(config_dir)
        if not global_hooks.source:
            global_hooks = None
    except (HooksConfigError, OSError) as exc:
        await engine._bus.publish(
            Warning(
                code="hooks_config_invalid",
                message=f"Global hooks config could not be loaded: {exc}.  Global hooks disabled.",
                display_message=_CONSTRUCTION_GLOBAL_HOOKS_INVALID.bind(detail=DisplayBlock(str(exc))),
                session_id=engine._session_id,
            )
        )
        global_hooks = None

    project_hooks: HooksFile | None = None
    try:
        if project_hooks_enabled:
            project_hooks = load_hooks_project(root)
    except (HooksConfigError, OSError) as exc:
        await engine._bus.publish(
            Warning(
                code="project_hooks_config_invalid",
                message=(
                    f"Project hooks config could not be loaded: {exc}. Project hooks disabled; global hooks unaffected."
                ),
                display_message=_CONSTRUCTION_PROJECT_HOOKS_INVALID.bind(detail=DisplayBlock(str(exc))),
                session_id=engine._session_id,
            )
        )
        project_hooks = None

    merged = merge_hooks_files(project=project_hooks, global_=global_hooks)
    if not merged.sources:
        return None
    return HookManager(
        file=merged,
        hooks_dir=config_dir / "hooks",
    )


def _start_outbox_recovery(engine: AgentLifecycleHost) -> None:
    """Start durable outbox recovery for the current hook manager if needed."""
    if engine._hook_manager is None or engine._outbox_recovery_task is not None:
        return
    # Background-recover the outbox only after the session guard is held.
    # That prevents a failed/conflicting startup from spawning work, and
    # shutdown explicitly observes/cancels this task before draining hooks.
    engine._outbox_recovery_task = asyncio.create_task(engine._hook_manager.recover_outbox())


async def _cancel_outbox_recovery_task(engine: AgentLifecycleHost) -> None:
    """Observe or cancel the active outbox recovery task before replacing managers."""
    task = engine._outbox_recovery_task
    if task is None:
        return
    if task.done():
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Outbox recovery task failed")
    else:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    engine._outbox_recovery_task = None


async def _close_replaced_hook_manager(manager: HookManager | None) -> None:
    """Close a hook manager that no longer belongs to the live executor."""
    if manager is None:
        return
    try:
        await manager.drain_session()
    except Exception:
        logger.exception("Error draining replaced hook manager")


async def _commit_hook_manager_replacement(
    engine: AgentLifecycleHost,
    old_hook_manager: HookManager | None,
) -> None:
    """Retire the old hook manager after a replacement executor is installed."""
    await _cancel_outbox_recovery_task(engine)
    await _close_replaced_hook_manager(old_hook_manager)
    _start_outbox_recovery(engine)


async def _settle_staged_hook_manager(
    engine: AgentLifecycleHost,
    *,
    old: HookManager | None,
    staged: HookManager | None,
) -> None:
    """After a failed build, retire whichever hook manager lost.

    Pre-commit failure: the staged candidate never went live — drain it and
    leave the old manager, its outbox and its recovery task untouched.
    Post-commit failure: the candidate is installed and the build it belongs
    to is what the engine keeps, so retire the old manager exactly as the
    success path would have.
    """
    if staged is old:
        return
    if engine._hook_manager is staged:
        await _commit_hook_manager_replacement(engine, old)
    else:
        await _close_replaced_hook_manager(staged)


async def start(
    engine: AgentLifecycleHost,
    profile: AgentProfile,
    *,
    operation: str = "startup",
    set_current_engine: SetCurrentEngine,
    staged_loaded: LoadedSettings | None = None,
    workspace: Workspace | None = None,
) -> None:
    """Initialize the engine with an agent profile and wire up event handlers.

    ``staged_loaded`` is a settings candidate the build should be configured
    from instead of the live handle — a settings reload or workspace change
    retrying after a failed first build passes the re-derived load here, and
    it goes live only when the build commits. ``workspace`` is the same thing
    for the workspace: a no-executor workspace change passes the new root here
    rather than mutating the live one, so a failed build keeps the workspace
    the live settings and hooks still describe.
    """
    engine._agent_profile = profile
    await engine._subscribe_event_handlers()

    if engine._workspace is None and workspace is None:
        # A fresh default, not a candidate: with no live workspace there is
        # nothing a failed build could corrupt, and error paths after such a
        # failure have always been able to read the ambient root from here.
        engine._workspace = Workspace.from_cwd()
    staged_workspace = workspace if workspace is not None else engine._workspace

    if engine._session_id is None:
        from uuid import uuid4

        engine._session_id = str(uuid4())

    if not engine._active_session_guard.ensure(engine._session_id):
        message = engine._active_session_guard.conflict_message(engine._session_id)
        await engine._bus.publish(Error(code="session_in_use", message=message, session_id=engine._session_id))
        raise RuntimeError(message)

    # Build the hook manager once per engine instance when a hook config
    # file exists.  With no config file, hooks stay a true no-op: no
    # manager, no outbox/log/tmp directories, and no recovery task.
    #
    # The manager survives profile switches within the same session —
    # hooks are global, not profile-scoped. A no-executor workspace-change
    # retry can happen after a startup failure, so reload there because
    # project hooks are primary-cwd scoped. The candidate stays staged: it
    # goes live with the build's commit, and until then the old manager
    # keeps running untouched.
    old_hook_manager = engine._hook_manager
    hook_manager = old_hook_manager
    if old_hook_manager is None or operation == "workspace_change":
        # Project hooks are primary-cwd scoped, so the candidate is built for
        # the root the staged workspace would install, not the live one.
        build_settings = (staged_loaded if staged_loaded is not None else engine._loaded_settings).settings
        hook_manager = await _build_hook_manager(
            engine,
            project_root=_workspace_primary_cwd(staged_workspace),
            project_hooks_enabled=build_settings.project_hooks_enabled,
        )
        if hook_manager is not None:
            # Session / turn lifecycle hooks fire outside any model run, so
            # they record under the recorder's current scope instead.
            hook_manager.trajectory_context_provider = engine._trajectory_recorder.context

    session_dir = engine._session_dir if engine._persistence.state_store is not None else None

    from chrys.foundation.observability.sink import get_otel_sink

    otel_sink = get_otel_sink()
    if otel_sink is not None:
        otel_sink.activate(engine._session_id, session_dir)
    # The trajectory recorder is bound here and activates (opens the log,
    # takes the writer lease) on its first event, so a session that never
    # records anything never creates ``trajectory/``.
    event_loop = asyncio.get_running_loop()
    trajectory_session_id = engine._session_id
    trajectory_warning_tasks: set[asyncio.Task[None]] = set()

    def _report_trajectory_activation_failure(_reason: str) -> None:
        def _publish_warning() -> None:
            task = event_loop.create_task(
                engine._bus.publish(
                    Warning(
                        code="trajectory_activation_failed",
                        message="Trajectory recording could not start and has been disabled for this session.",
                        display_message=_CONSTRUCTION_TRAJECTORY_ACTIVATION_FAILED.bind(),
                        session_id=trajectory_session_id,
                    )
                )
            )
            trajectory_warning_tasks.add(task)
            task.add_done_callback(trajectory_warning_tasks.discard)

        try:
            event_loop.call_soon_threadsafe(_publish_warning)
        except RuntimeError:
            logger.debug("Trajectory activation failure could not be published because the event loop is closed")

    trajectory = engine._trajectory_recorder.bind_session(
        session_id=trajectory_session_id,
        session_dir=session_dir,
        write_lock_path=engine._session_write_lock_path(trajectory_session_id) if session_dir is not None else None,
        session_start_info=lambda: trajectory_session_start_info(engine),
        on_activation_failed=_report_trajectory_activation_failure,
    )

    engine._begin_agent_load()
    try:
        await engine._bus.publish(
            AgentLoadStarted(
                operation=operation,
                to_profile=profile.name,
                to_display_name=profile.display_name or profile.name,
                session_id=engine._session_id,
            ),
        )

        staged = stage_build(
            engine,
            loaded=staged_loaded if staged_loaded is not None else engine._loaded_settings,
            agent_profile=profile,
            workspace=staged_workspace,
            hook_manager=hook_manager,
        )
        await engine._build_agent(profile, staged)
        engine._fsm.try_transition(Trigger.START)
        set_current_engine()
    except Exception as exc:
        await _settle_staged_hook_manager(engine, old=old_hook_manager, staged=hook_manager)
        engine._finish_agent_load()
        await engine._publish_load_failed(operation=operation, profile=profile, exc=exc)
        raise
    except BaseException:
        await _settle_staged_hook_manager(engine, old=old_hook_manager, staged=hook_manager)
        engine._finish_agent_load()
        raise

    # The build committed, so the profiles this session opened with are the
    # ones it keeps: pin them before a switch can rewrite what the recorder
    # reads at its first event.
    trajectory.pin_session_start_info()

    if hook_manager is not old_hook_manager:
        await _commit_hook_manager_replacement(engine, old_hook_manager)
    else:
        _start_outbox_recovery(engine)

    engine._finish_agent_load()
    await engine._bus.publish(
        AgentLoadFinished(
            operation=operation,
            agent_profile=profile.name,
            display_name=profile.display_name or profile.name,
            session_id=engine._session_id,
        ),
    )

    sub_agent_tool_names = engine._sub_agent_tools.tool_names() if engine._sub_agent_tools else []
    await engine._bus.publish(
        SessionReady(
            agent_profile=profile.name,
            display_name=profile.display_name,
            model_profile_id=engine._active_profile.id if engine._active_profile else "",
            max_context_tokens=engine._active_profile.max_context_tokens if engine._active_profile else 0,
            session_id=engine._session_id,
            tool_names=engine._tool_names,
            tool_kinds=engine._tool_kinds,
            skill_names=engine._skill_names,
            sub_agent_tool_names=sub_agent_tool_names,
            memory_files=engine._memory_files,
            runtime_details=engine._runtime_details,
            primary_cwd=_workspace_cwd(engine),
            working_dirs=_workspace_working_dirs(engine),
        ),
    )
    await engine._bus.publish(ApprovalModeUpdated(mode=engine._approval_mode.value, session_id=engine._session_id))

    # Fire ``session_start`` hooks after the engine is fully ready and
    # the UI has its ``SessionReady`` event.  Hooks are observers here:
    # they cannot deny startup or mutate engine state, although a
    # blocking hook still runs inline at this post-ready boundary.
    # Profile-switch/rebuild operations skip this so users don't see a
    # hook fire every time the same session reloads infrastructure.
    if engine._hook_manager is not None and operation in {"startup", "new_session", "reset"}:
        from chrys.service.hooks.events import HookEvent

        await engine._hook_manager.fire(
            HookEvent.SESSION_START,
            {"session_id": engine._session_id, "profile": profile.name, "cwd": _workspace_cwd(engine)},
            scope="session",
        )


async def build_agent(
    engine: AgentLifecycleHost,
    profile: AgentProfile,
    *,
    staged: StagedBuild,
    build_agent_fn: BuildAgentFn,
    preserved_history: dict | None = None,
) -> None:
    """Build Agent, Executor, and supporting infrastructure from *staged* inputs.

    The build reads ``staged`` — never the live engine fields it is about to
    replace — and commits everything it read together with the executor it
    produced in one synchronous block. A failure before that block leaves
    every live field exactly as it was, including the settings handle.
    ``preserved_history``, when given, is installed inside that same block:
    a restart's conversation is commit state, and an executor published
    without it would hand the next save an empty history to persist.
    """
    if profile.acp is not None:
        raise ValueError(f"ACP profile {profile.name!r} is sub-agent-only and cannot be launched as the main agent")

    intermediate_buffer = IntermediateTextBuffer()
    engine._intermediate_texts = {}

    # Both callbacks record ``batch_id → text`` at capture time — list
    # position cannot reconstruct ids after the fact (retries continue
    # numbering, and batch boundaries can fire without a captured text).
    # Shared rule: an empty-text boundary signal still advances the counter
    # but writes no mapping entry; a non-empty text advances the counter
    # FIRST (``new_batch()`` here; ``store()`` already calls it internally
    # in the sync path), then records the post-increment batch_id — the
    # same id the batch's subsequent tool calls are stamped with, so the
    # text attaches to the batch it precedes.

    async def _on_intermediate_async(text: str) -> None:
        intermediate_buffer.new_batch()
        if text:
            engine._intermediate_texts[intermediate_buffer.batch_id] = text
        await engine._bus.publish(
            AgentMessage(
                text=text,
                is_final=False,
                is_intermediate=True,
                session_id=engine._session_id,
            ),
        )

    def _on_intermediate_sync(text: str) -> None:
        intermediate_buffer.store(text)
        if text:
            engine._intermediate_texts[intermediate_buffer.batch_id] = text

    def _commit_intermediate_text(text: str, batch_id: int) -> None:
        if text:
            engine._intermediate_texts[batch_id] = text

    injection = InjectionMiddleware()
    consumed_injections: list[ConsumedInjection] = []

    async def _on_injection_batch_consumed(batch: tuple[ConsumedInjection, ...]) -> None:
        """Register one immutable wire batch before its first durability await."""
        newly_consumed: list[ConsumedInjection] = []
        for consumed in batch:
            existing_index = next(
                (
                    index
                    for index, existing in enumerate(consumed_injections)
                    if existing.consumption_id == consumed.consumption_id
                ),
                None,
            )
            if existing_index is None:
                consumed_injections.append(consumed)
                newly_consumed.append(consumed)
            else:
                # A retry reuses the stable consumption identity but updates
                # the failed attempt's anchor to the successful wire context.
                consumed_injections[existing_index] = consumed

        async def _publish_results() -> None:
            for consumed in newly_consumed:
                await engine._bus.publish(
                    UserInjectResult(
                        text=consumed.text,
                        consumed=True,
                        created_at=consumed.created_at,
                        injection_id=consumed.injection_id,
                        session_id=engine._session_id,
                    )
                )

        if newly_consumed:
            # Consumption is already irrevocably registered. Neither writer
            # backpressure nor task cancellation may prevent its frontend
            # notification from being scheduled.
            for consumed in newly_consumed:
                if consumed.preparation is not None:
                    consumed.preparation.finished_soon(
                        outcome=PreparationOutcome.INJECTED,
                        target_turn_id=consumed.target_turn_id,
                    )
            # Preserve Chrys's established detached notification behavior: a
            # retry updates anchors but does not render the same injection a
            # second time, and Agent.run cancellation cannot tear delivery in
            # half. The complete batch is already registered above.
            engine._injection_notify_task = asyncio.create_task(_publish_results())
        # Crash durability: the wire copy lives only in the tool loop and
        # the persisted copy is written by the finalizer — a hard crash in
        # between loses the injected user message.  Snapshot a recovery
        # checkpoint (which replays ``consumed_injections``) and FLUSH the
        # background writer before returning: ``_save_recovery_checkpoint``
        # alone only queues the snapshot, and every later snapshot also
        # carries the injection (the list is append-only until finalization
        # clears it), so awaiting the newest-wins drain is sound.  Injection
        # consumption is rare and immediately precedes an LLM call — one
        # awaited write here is negligible; plain LLM-boundary checkpoints
        # stay fire-and-forget.
        await engine._save_recovery_checkpoint()
        await engine._flush_recovery_checkpoint()

    injection.set_on_consumed_batch(_on_injection_batch_consumed)

    session_dir = engine._session_dir if engine._persistence.state_store is not None else None
    if engine._mutation_tracker is None and session_dir is not None:
        engine._mutation_tracker = MutationTracker(
            SnapshotStore(session_dir, policy=SnapshotPolicy.from_settings(staged.loaded.settings))
        )
    # Profile-independent (unlike the tool/middleware wiring in build_agent):
    # a conditional tracker would drop ``chrys_todos`` on the first save after
    # switching to a todo-less profile. Restore pre-hydrates; respect it.
    if engine._todo_tracker is None:
        engine._todo_tracker = TodoTracker()

    from chrys.service.approval.judge import ApprovalJudge
    from chrys.service.llm.route_sessions import derive_llm_route_session_id
    from chrys.service.profiles.models.resolver import resolve_active_profile, resolve_judge_profile

    judge_fallback = resolve_active_profile(engine._model_registry, staged.loaded.settings)
    judge_profile = resolve_judge_profile(engine._model_registry, staged.loaded.settings, judge_fallback)
    judge_session_id = derive_llm_route_session_id(
        engine._session_id,
        route_kind="approval-judge",
        model_profile=judge_profile,
    )
    approval_judge = ApprovalJudge(
        judge_profile,
        session_id=judge_session_id,
        parent_session_id=engine._session_id,
        session_dir=session_dir,
    )

    old_agent = engine._agent
    old_executor = engine._executor
    old_sub_agent_tools = engine._sub_agent_tools
    old_mcp_adapter = engine._mcp_adapter
    old_coordinator = engine._mutation_coordinator
    try:
        result = await build_agent_fn(
            profile=profile,
            settings=staged.loaded.settings,
            model_registry=engine._model_registry,
            workspace=staged.workspace,
            session_id=engine._session_id,
            bus=engine._bus,
            agent_registry=engine._agent_registry,
            existing_sub_agent_tools=None,
            existing_mcp_adapter=None,
            injection=injection,
            intermediate_buffer=intermediate_buffer,
            on_intermediate_async=_on_intermediate_async,
            on_intermediate_sync=_on_intermediate_sync,
            commit_intermediate_text=_commit_intermediate_text,
            on_usage=engine._publish_usage,
            on_sub_agent_usage=engine._accumulate_sub_agent_usage,
            on_side_call_usage=engine._accumulate_side_call_usage,
            drain_parent_usage_publishes=engine._drain_usage_publishes,
            on_compaction=engine._publish_compaction,
            on_pre_compact=engine._publish_pre_compact,
            on_compress=engine._publish_compress,
            on_load_progress=engine._publish_load_progress,
            mutation_tracker=engine._mutation_tracker,
            mutation_coordinator=staged.mutation_coordinator,
            todo_tracker=engine._todo_tracker,
            file_change_provider=engine._workspace_change_tracker.take_pending_notice,
            approval_mode=engine._approval_mode,
            approval_judge=approval_judge,
            session_dir=session_dir,
            mcp_cache=engine._mcp_cache,
            hook_manager=staged.hook_manager,
            spill_quota=engine._spill_quota,
            persist_recovery_now=engine.persist_recovery_now,
            turn_context=engine._turn_context,
            allow_user_interaction=engine._allow_user_interaction,
        )
    except BaseException:
        # A fresh candidate coordinator never went live: stamp its registry
        # file closed so peers don't keep seeing a build that never ran. The
        # surviving instance (if any) is still installed and stays untouched.
        if staged.mutation_coordinator is not None and staged.mutation_coordinator is not old_coordinator:
            await _close_coordinator(staged.mutation_coordinator, reason="failed build")
        raise

    # The commit: everything the build was configured from goes live in one
    # synchronous block together with the executor it produced — no await
    # between the first assignment and the last, so a concurrent reader sees
    # the old build entirely or the new one entirely, never a mixture.
    engine._settings_handle.install(staged.loaded)
    engine._agent_profile = staged.agent_profile
    engine._workspace = staged.workspace
    engine._hook_manager = staged.hook_manager
    engine._mutation_coordinator = staged.mutation_coordinator
    engine._agent = result.agent
    engine._executor = result.executor
    engine._runtime = result.runtime
    engine._injection = injection
    engine._consumed_injections = consumed_injections
    engine._loop_recorder = result.loop_recorder
    engine._loop_recorder.on_pre_wire_barrier = engine.persist_recovery_barrier
    engine._loop_recorder.on_result_checkpoint = engine._save_recovery_checkpoint
    engine._executor.recovery_input_recorder = engine._turn_state.set_current_input
    engine._reminder_middleware = result.reminder_middleware
    if preserved_history is not None:
        # Part of the commit, not of the post-commit cleanup below: once the
        # executor is reachable it can be saved, so the conversation must be
        # in it before anything awaits — a cancellation landing in the cleanup
        # steps would otherwise publish an executor whose empty history the
        # next save persists over the real one.
        engine._executor.history_state = preserved_history
        if engine._reminder_middleware is not None:
            engine._reminder_middleware.restore_phase4_state(preserved_history)
    # The history manager re-binds in the same block, for the same reason:
    # orchestration mutates the conversation through it, and a manager left
    # on the replaced executor's dict would fork what the executor
    # accumulates from what the finalizers stamp and the save persists.
    engine._history.bind(engine._executor.history_state)
    engine._approval_judge = approval_judge
    engine._sub_agent_tools = result.sub_agent_tools
    engine._skills_provider = result.skills_provider
    engine._mcp_adapter = result.mcp_adapter
    engine._tool_names = result.tool_names
    engine._tool_kinds = result.tool_kinds
    engine._skill_names = result.skill_names
    engine._memory_files = result.memory_files
    engine._agent_profile_fingerprint = result.agent_profile_fingerprint
    engine._model_profile_fingerprint = result.model_profile_fingerprint
    engine._runtime_details = result.runtime_details
    engine._active_profile = result.active_profile
    # The staged settings are installed above, so this reads the value the
    # rebuilt runtime runs on: with the notice off the roots are not probed.
    engine._workspace_change_tracker.retarget_roots(
        staged.workspace, resolve_scope=engine._settings.workspace_change_notice
    )
    if result.compaction_strategy is not None:
        engine._runtime_meta.restore_context_calibration(
            result.compaction_strategy,
            model_profile_fingerprint=engine._model_profile_fingerprint,
            agent_profile_fingerprint=engine._agent_profile_fingerprint,
        )
    engine._advance_build_generation()
    # Post-commit finalization of the replaced build. The commit above already
    # swapped the live pointers, so a step skipped here is unreachable forever
    # after — shutdown only visits the current build. Cancellation therefore
    # must not strand the remaining steps: each one traps it, the rest still
    # run, and the cancellation resumes once the last has finished.
    cancelled: asyncio.CancelledError | None = None
    if old_executor is not None and old_executor is not engine._executor:
        try:
            await old_executor.close()
        except asyncio.CancelledError as exc:
            cancelled = exc
        except Exception:
            logger.exception("Error closing previous executor")
    if old_coordinator is not None and old_coordinator is not staged.mutation_coordinator:
        # Settings turned coordination off (or the session re-targeted): the
        # replaced registry file is stamped closed only now that the build is
        # committed, so peers keep warning about us until we truly stop.
        try:
            await _close_coordinator(old_coordinator, reason="replaced")
        except asyncio.CancelledError as exc:
            cancelled = exc
    try:
        await engine._cleanup_replaced_build_resources(old_agent, old_sub_agent_tools, old_mcp_adapter)
    except asyncio.CancelledError as exc:
        cancelled = exc
    if cancelled is not None:
        raise cancelled


async def cleanup_replaced_build_resources(
    engine: AgentLifecycleHost,
    old_agent: Agent | None,
    old_sub_agent_tools: SubAgentTools | None,
    old_mcp_adapter: MCPAdapter | None,
) -> None:
    """Release resources from the previous successful build after replacement.

    Cancellation-safe for the same reason as the commit's other post-commit
    steps: the replaced objects are already unreachable from the engine, so a
    step cancellation skips would never run again. Each step traps it and it
    is re-raised after the last one.
    """
    cancelled: asyncio.CancelledError | None = None
    if old_agent is not None and old_agent is not engine._agent:
        try:
            await old_agent.__aexit__(None, None, None)
        except asyncio.CancelledError as exc:
            cancelled = exc
        except Exception:
            logger.exception("Error exiting previous agent")
    if old_sub_agent_tools is not None and old_sub_agent_tools is not engine._sub_agent_tools:
        try:
            await old_sub_agent_tools.cleanup()
        except asyncio.CancelledError as exc:
            cancelled = exc
        except Exception:
            logger.exception("Error cleaning up previous sub-agent tools")
    if old_mcp_adapter is not None and old_mcp_adapter is not engine._mcp_adapter:
        try:
            await old_mcp_adapter.disconnect_all()
        except asyncio.CancelledError as exc:
            cancelled = exc
        except Exception:
            logger.exception("Error disconnecting previous MCP adapter")
    if cancelled is not None:
        raise cancelled


async def publish_load_progress(
    engine: AgentLifecycleHost,
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
    """Publish an agent load progress event."""
    await engine._bus.publish(
        AgentLoadProgress(
            phase=phase,
            message=message,
            server_name=server_name,
            current=current,
            total=total,
            failed=failed,
            status=status,
            subject=subject,
            detail=detail,
            session_id=engine._session_id,
        ),
    )


async def publish_load_failed(
    engine: AgentLifecycleHost,
    *,
    operation: str,
    profile: AgentProfile,
    exc: Exception,
) -> None:
    """Publish an agent load failure event."""
    await engine._bus.publish(
        AgentLoadFailed(
            operation=operation,
            agent_profile=profile.name,
            display_name=profile.display_name or profile.name,
            message=clean_error_message(exc),
            session_id=engine._session_id,
        ),
    )


def _preserved_history_state(raw: dict | None) -> dict | None:
    """Deep copy of the predecessor's history state, installed as the successor's live history.

    The copy owns every layer (messages, contents lists, content objects) so the
    successor's registries and anchors can never alias the predecessor's live
    objects. Falsy/empty history returns ``None`` — no carryover to preserve.
    """
    return copy.deepcopy(raw) if raw else None


async def soft_restart(
    engine: AgentLifecycleHost,
    new_profile: AgentProfile,
    workspace: Workspace | None = None,
    *,
    operation: str = "switch",
    staged_loaded: LoadedSettings | None = None,
) -> None:
    """Restart the agent with a new profile/workspace while preserving history.

    ``staged_loaded`` is a settings candidate the build should be configured
    from instead of the live handle — a settings reload or workspace change
    passes the re-derived load here, and it goes live only when the build
    commits. A pre-commit failure therefore leaves the settings the previous
    executor was built from untouched; a post-commit failure keeps the new
    ones, because they are what the installed executor runs with.
    """
    old_profile_name = engine._agent_profile.name if engine._agent_profile else ""
    old_display_name = engine._agent_profile.display_name if engine._agent_profile else ""
    old_model_profile = engine._active_profile
    old_model_base_url = engine._runtime_details.model.base_url
    old_agent_profile_fingerprint = engine._agent_profile_fingerprint
    old_model_profile_fingerprint = engine._model_profile_fingerprint
    old_service_session_id = engine._executor.service_session_id if engine._executor is not None else ""
    old_service_session_storage_enabled = (
        engine._executor.service_session_storage_enabled if engine._executor is not None else False
    )
    old_workspace = engine._workspace
    old_hook_manager = engine._hook_manager
    staged_hook_manager = old_hook_manager
    new_loaded = staged_loaded if staged_loaded is not None else engine._loaded_settings
    # Project hooks are primary-cwd scoped and gated by ``project.hooks_enabled``:
    # a new primary cwd or a flipped gate rebuilds the manager for the root
    # the build is about to install (the live one when the workspace stays).
    reload_hook_manager = (
        workspace is not None and _workspace_primary_cwd(old_workspace) != _workspace_primary_cwd(workspace)
    ) or new_loaded.settings.project_hooks_enabled != engine._loaded_settings.settings.project_hooks_enabled
    hook_manager_root = _workspace_primary_cwd(workspace if workspace is not None else old_workspace)
    hook_manager_replacement_committed = False

    preserved_state: dict | None = None
    engine._begin_agent_load()
    try:
        await engine._bus.publish(
            AgentLoadStarted(
                operation=operation,
                from_profile=old_profile_name,
                to_profile=new_profile.name,
                from_display_name=old_display_name or old_profile_name,
                to_display_name=new_profile.display_name or new_profile.name,
                session_id=engine._session_id,
            ),
        )

        if engine._executor is not None:
            raw = engine._executor.history_state
            preserved_state = _preserved_history_state(raw)
            if engine._reminder_middleware is not None:
                catalog_pointer_record_count = engine._reminder_middleware.get_catalog_pointer_record_count_state()
                if catalog_pointer_record_count is not None:
                    preserved_state = preserved_state or {}
                    preserved_state[CATALOG_POINTER_RECORD_COUNT_STATE_KEY] = catalog_pointer_record_count

        old_pending_switch = (
            engine._reminder_middleware.snapshot_pending_switch() if engine._reminder_middleware is not None else None
        )
        is_consecutive = old_pending_switch is not None

        if preserved_state is not None and new_profile.name != old_profile_name:
            switches = preserved_state.setdefault("agent_profile_switches", [])
            new_label = new_profile.display_name or new_profile.name
            if is_consecutive and switches:
                switches[-1]["to"] = new_profile.name
                switches[-1]["to_display"] = new_label
                switches[-1]["timestamp"] = datetime.now(UTC).isoformat()
                if switches[-1]["from"] == new_profile.name:
                    switches.pop()
            else:
                switches.append(
                    {
                        "from": old_profile_name,
                        "to": new_profile.name,
                        "from_display": old_display_name or old_profile_name,
                        "to_display": new_label,
                        "at_message_index": len(preserved_state.get("messages", [])),
                        "timestamp": datetime.now(UTC).isoformat(),
                    },
                )

        if reload_hook_manager:
            staged_hook_manager = await _build_hook_manager(
                engine,
                project_root=hook_manager_root,
                project_hooks_enabled=new_loaded.settings.project_hooks_enabled,
            )

        staged = stage_build(
            engine,
            loaded=staged_loaded if staged_loaded is not None else engine._loaded_settings,
            agent_profile=new_profile,
            workspace=workspace if workspace is not None else old_workspace,
            hook_manager=staged_hook_manager,
        )
        # The commit inside ``_build_agent`` records ``new_profile`` with the
        # executor it configured — and installs ``preserved_state`` into that
        # executor in the same synchronous block. Doing either here, after the
        # awaited post-commit steps, would let a cancellation strand the new
        # executor under the old profile's name or publish it with an empty
        # history for the next save to persist.
        await engine._build_agent(new_profile, staged, preserved_history=preserved_state)
        if staged_hook_manager is not old_hook_manager:
            await _commit_hook_manager_replacement(engine, old_hook_manager)
            hook_manager_replacement_committed = True
        if old_service_session_id and engine._executor is not None:
            if _can_reuse_responses_service_session(
                old_agent_profile_fingerprint=old_agent_profile_fingerprint,
                new_agent_profile_fingerprint=engine._agent_profile_fingerprint,
                old_model_profile_fingerprint=old_model_profile_fingerprint,
                new_model_profile_fingerprint=engine._model_profile_fingerprint,
                old_model_profile=old_model_profile,
                new_model_profile=engine._active_profile,
                old_model_base_url=old_model_base_url,
                new_model_base_url=engine._runtime_details.model.base_url,
                old_workspace=old_workspace,
                new_workspace=engine._workspace,
                old_storage_enabled=old_service_session_storage_enabled,
                new_storage_enabled=engine._executor.service_session_storage_enabled,
            ):
                engine._executor.service_session_id = old_service_session_id
            else:
                engine._executor.service_session_id = ""
                await engine._bus.publish(
                    Warning(
                        code="service_session_incompatible",
                        message=(
                            "The previous OpenAI Responses service session is not compatible with "
                            "the active agent profile, workspace, model profile, service endpoint, "
                            "or storage is disabled. "
                            f"{APP_DISPLAY_NAME} will continue from local history only."
                        ),
                        display_message=_CONSTRUCTION_SERVICE_SESSION_INCOMPATIBLE.bind(
                            app_name=APP_DISPLAY_NAME,
                        ),
                        session_id=engine._session_id,
                    )
                )
        if new_profile.name != old_profile_name and engine._reminder_middleware is not None:
            new_label = new_profile.display_name or new_profile.name
            if is_consecutive and old_pending_switch is not None:
                if old_pending_switch["from"] != new_label:
                    engine._reminder_middleware.set_profile_switch(old_pending_switch["from"], new_label)
            else:
                old_label = old_display_name or old_profile_name
                engine._reminder_middleware.set_profile_switch(old_label, new_label)

        if preserved_state is not None:
            last_usage = preserved_state.get("last_usage")
            if isinstance(last_usage, dict):
                preserved_meta = SessionRuntimeMetadata.from_state_dict(preserved_state)
                engine._runtime_meta.last_usage_details = preserved_meta.last_usage_details
                # Route through the ordered chain so a sub-agent UsageUpdate
                # still queued from the prior build can't overtake this
                # post-restart snapshot.
                usage.enqueue_usage_event(engine, engine._make_usage_event())
    except Exception as exc:
        # Nothing was staged onto the engine, so there is nothing to roll
        # back: a pre-commit failure left every live field untouched, and a
        # post-commit failure keeps the committed build. Only the losing
        # hook manager still needs retiring.
        if not hook_manager_replacement_committed:
            await _settle_staged_hook_manager(engine, old=old_hook_manager, staged=staged_hook_manager)
        engine._finish_agent_load()
        await engine._publish_load_failed(operation=operation, profile=new_profile, exc=exc)
        raise
    except BaseException:
        if not hook_manager_replacement_committed:
            await _settle_staged_hook_manager(engine, old=old_hook_manager, staged=staged_hook_manager)
        engine._finish_agent_load()
        raise

    message_count = len(preserved_state.get("messages", [])) if preserved_state else 0
    switched_sub_agent_names = engine._sub_agent_tools.tool_names() if engine._sub_agent_tools else []
    engine._finish_agent_load()
    await engine._bus.publish(
        AgentLoadFinished(
            operation=operation,
            agent_profile=new_profile.name,
            display_name=new_profile.display_name or new_profile.name,
            session_id=engine._session_id,
        ),
    )
    await engine._bus.publish(
        ProfileSwitched(
            from_profile=old_profile_name,
            to_profile=new_profile.name,
            from_display_name=old_display_name or old_profile_name,
            to_display_name=new_profile.display_name or new_profile.name,
            message_count=message_count,
            model_profile_id=engine._active_profile.id if engine._active_profile else "",
            max_context_tokens=engine._active_profile.max_context_tokens if engine._active_profile else 0,
            session_id=engine._session_id,
            tool_names=engine._tool_names,
            skill_names=engine._skill_names,
            sub_agent_tool_names=switched_sub_agent_names,
            memory_files=engine._memory_files,
            runtime_details=engine._runtime_details,
        ),
    )
    await engine._trajectory_recorder.profile_switched(
        kind=ProfileKind.AGENT,
        from_fingerprint=old_agent_profile_fingerprint,
        to_fingerprint=engine._agent_profile_fingerprint,
    )
    await engine._trajectory_recorder.profile_switched(
        kind=ProfileKind.MODEL,
        from_fingerprint=old_model_profile_fingerprint,
        to_fingerprint=engine._model_profile_fingerprint,
    )
    if workspace is not None:
        await engine._bus.publish(
            WorkspaceUpdated(
                primary_cwd=workspace.primary_cwd,
                working_dirs=[d.path for d in workspace.working_dirs],
                reference_files=workspace.reference_files,
                session_id=engine._session_id,
            ),
        )
