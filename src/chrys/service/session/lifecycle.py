# Copyright (c) 2026 Chrys. All rights reserved.

"""Engine-internal companion module for session lifecycle orchestration."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.config.context import EvalContext
from chrys.foundation.config.process_settings import reattribute_command_line, route_restart_settings
from chrys.foundation.config.runtime_pointer import PointerToken, restore_model_pointer, set_model_pointer
from chrys.foundation.config.settings_store import load_settings
from chrys.foundation.config.spec import SettingOrigin, Source
from chrys.foundation.config.warnings import settings_warning_events
from chrys.foundation.events.types import (
    Error,
    SessionClear,
    SessionDelete,
    SessionDeleted,
    SessionNew,
    SessionRestore,
    SessionRestored,
    Warning,
)
from chrys.foundation.i18n import DisplaySequence, msg
from chrys.foundation.i18n.formatting import format_message
from chrys.foundation.models.history_markers import SUB_AGENT_STATE_DISCARDED_MESSAGE, HistoryMarkerKind
from chrys.foundation.models.workspace import WorkingDir, Workspace
from chrys.foundation.platform import safe_getcwd
from chrys.foundation.platform.files import surrogate_safe_text
from chrys.foundation.util.lock import FileLock
from chrys.service import usage
from chrys.service.context.compaction.spill import SpillReconciliationResult, reconcile_spill_storage
from chrys.service.mutations.store import SnapshotPolicy, SnapshotStore
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker
from chrys.service.profiles.models.resolver import loaded_with_active_model_profile
from chrys.service.profiles.models.schema import API_STYLE_RESPONSES, is_model_profile_selectable
from chrys.service.session.runtime_metadata import SessionRuntimeMetadata
from chrys.service.session.sub_agent_logs import SubAgentSessionArtifactService
from chrys.service.state.store import (
    SESSION_BACKUP_FILE_NAME,
    SESSION_FILE_NAME,
    SESSION_RECOVERY_FILE_NAME,
    SESSION_WRITE_LOCK_TIMEOUT_SECONDS,
    SessionMeta,
    parse_snapshot_turn,
)
from chrys.service.todos.tracker import TodoTracker

if TYPE_CHECKING:
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.config.settings_store import LoadedSettings, SettingsHandle
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import UsageUpdate
    from chrys.orchestration.engine.executor import Executor
    from chrys.orchestration.engine.state.machine import EngineStateMachine
    from chrys.orchestration.sub_agents.tools import SubAgentTools
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
    from chrys.service.context.compaction.spill import SpillQuota
    from chrys.service.hooks.manager import HookManager
    from chrys.service.mutations.coordination import MutationCoordinator
    from chrys.service.profiles.agents.registry import AgentProfileRegistry
    from chrys.service.profiles.agents.schema import AgentProfile
    from chrys.service.profiles.models.registry import ModelProfileRegistry
    from chrys.service.profiles.models.schema import ModelProfile
    from chrys.service.session.history import SessionHistoryManager
    from chrys.service.session.persistence import SessionPersistence
    from chrys.service.state.locks import ActiveSessionGuard


logger = logging.getLogger(__name__)

_RESTORE_SERVICE_SESSION_INCOMPATIBLE = msg(
    "restore.service_session_incompatible",
    fallback=(
        "This session was saved with an OpenAI Responses service session. "
        "The active agent/model profile, workspace, service endpoint, or storage mode "
        "is not compatible, so {app} will continue from local history only."
    ),
)
_RESTORE_SUB_AGENTS_DISCARDED = msg(
    "restore.sub_agents_discarded",
    fallback="{discarded} paused sub-agent(s) from a previous session were discarded: {names}",
)
_RESTORE_AGENT_PROFILE_UNRESOLVED_USING_CURRENT = msg(
    "restore.agent_profile_unresolved_using_current",
    fallback=(
        "The saved agent profile {saved} could not be uniquely resolved. Continuing with the current agent {current}."
    ),
)
_RESTORE_AGENT_PROFILE_UNRESOLVED = msg(
    "restore.agent_profile_unresolved",
    fallback="The saved agent profile {saved} could not be uniquely resolved. Session restore was stopped.",
)
_RESTORE_REQUESTED_AGENT_PROFILE_UNRESOLVED = msg(
    "restore.requested_agent_profile_unresolved",
    fallback="The requested agent profile {profile} could not be found. Session restore was stopped.",
)


async def _reconcile_existing_spill_storage(engine: SessionLifecycleHost) -> SpillReconciliationResult:
    """Rebuild and account for retained spill storage without creating a session directory."""
    session_dir = engine._session_dir
    if session_dir is None or not session_dir.is_dir():
        return SpillReconciliationResult(0, 0, frozenset())
    try:
        return await asyncio.to_thread(reconcile_spill_storage, session_dir, engine._spill_quota)
    except OSError, RuntimeError, UnicodeError:
        # Spill records are auxiliary context. Filesystem damage or permissions
        # must not prevent the authoritative session state from hydrating.
        logger.warning("Unable to reconcile spill storage under %s", session_dir, exc_info=True)
        engine._spill_quota.disable_storage()
        return SpillReconciliationResult(0, 0, frozenset())


def _workspace_cwd(engine: SessionLifecycleHost) -> str:
    """Return the engine workspace cwd, falling back only for legacy unstarted engines."""
    if engine._workspace is not None:
        return engine._workspace.primary_cwd
    return safe_getcwd()


def _working_dirs_from_paths(paths: list[str], primary_cwd: str) -> list[WorkingDir]:
    return [WorkingDir(path=path, is_primary=path == primary_cwd) for path in paths]


def _restore_profile_override_switch(
    state: dict | None,
    *,
    from_profile: str,
    from_display: str,
    to_profile: str,
    to_display: str,
) -> bool:
    """Record an explicit restore-time profile override in session state."""
    if state is None or not from_profile or from_profile == to_profile:
        return False
    switches = state.setdefault("agent_profile_switches", [])
    switches.append(
        {
            "from": from_profile,
            "to": to_profile,
            "from_display": from_display or from_profile,
            "to_display": to_display or to_profile,
            "at_message_index": len(state.get("messages", [])),
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
    return True


def _extra_roots(paths: list[str], primary_cwd: str) -> list[str]:
    """Working-dir paths with the primary cwd removed (effective extra roots).

    Sessions persist ``working_dirs`` inconsistently: a no-extra session stores
    ``[]`` while a multi-root session stores ``[primary, *extras]``. Restore also
    always re-inserts the primary for explicit ``additionalDirectories``. Service-
    session identity only cares about the *effective* extra roots, so strip the
    primary from both sides before comparing — otherwise loading a no-extra session
    with ``additionalDirectories: []`` would discard a still-valid service session.
    """
    return [path for path in paths if path != primary_cwd]


def _can_restore_service_session(engine: SessionLifecycleHost, meta: SessionMeta) -> bool:
    """Return True when the active model can continue a saved service session id."""
    profile = engine._active_profile
    if (
        profile is None
        or engine._executor is None
        or meta.model_provider != "openai"
        or meta.model_api_style != API_STYLE_RESPONSES
        or profile.provider != "openai"
        or profile.api_style != API_STYLE_RESPONSES
        or meta.model_id != profile.model_id
        or not meta.model_profile_fingerprint
        or meta.model_profile_fingerprint != engine._model_profile_fingerprint
        or not meta.agent_profile_fingerprint
        or meta.agent_profile_fingerprint != engine._agent_profile_fingerprint
        or engine._workspace is None
        or meta.primary_cwd != engine._workspace.primary_cwd
        or _extra_roots(meta.working_dirs, meta.primary_cwd)
        != _extra_roots(
            [working_dir.path for working_dir in engine._workspace.working_dirs],
            engine._workspace.primary_cwd,
        )
    ):
        return False
    from chrys.service.llm.clients import effective_model_base_url

    if meta.model_base_url != effective_model_base_url(profile):
        return False
    return engine._executor.service_session_storage_enabled


def _restore_terminal_fsm_from_history(engine: SessionLifecycleHost) -> None:
    """Rebuild runtime terminal state from persisted history markers."""
    if not engine._history.is_bound:
        return
    marker = engine._history.trailing_status_marker()
    if marker is None:
        return
    kind, source = marker
    if kind != HistoryMarkerKind.INTERRUPTED:
        return
    engine._fsm.restore_terminal_state(failed=source == "error")


class SessionLifecycleHost(Protocol):
    """Engine state needed by session lifecycle orchestration."""

    _fsm: EngineStateMachine
    _shutting_down: bool
    _workspace: Workspace | None
    _session_id: str | None
    _turn_number: int
    _runtime_meta: SessionRuntimeMetadata
    _reminder_middleware: SystemReminderMiddleware | None
    _mutation_tracker: MutationTracker | None
    _workspace_change_tracker: WorkspaceChangeTracker
    _mutation_coordinator: MutationCoordinator | None
    _todo_tracker: TodoTracker | None
    _settings_handle: SettingsHandle
    _suppress_save: bool
    _persistence: SessionPersistence
    _agent_profile: AgentProfile | None
    _agent_profile_fingerprint: str
    _model_profile_fingerprint: str
    _active_profile: ModelProfile | None
    _active_session_guard: ActiveSessionGuard
    _agent_registry: AgentProfileRegistry | None
    _model_registry: ModelProfileRegistry | None
    _executor: Executor | None
    _history: SessionHistoryManager
    _sub_agent_tools: SubAgentTools | None
    _hook_manager: HookManager | None
    _session_end_fired: bool
    _recovered_from_sidecar: bool
    _ask_user_timeout_pinned: bool
    _model_profile_pinned: bool
    # UsageHost fields — required for ``usage.enqueue_usage_event`` calls below.
    _usage_tasks: set[asyncio.Task[None]]
    _usage_publish_tail: asyncio.Task[None] | None
    _spill_quota: SpillQuota

    @property
    def _settings(self) -> Settings:
        """Read-only: settings change through the handle, not the holder."""
        ...

    @property
    def _loaded_settings(self) -> LoadedSettings: ...

    @property
    def event_bus(self) -> EventBus: ...

    @property
    def _session_dir(self) -> Path | None: ...

    def _session_write_lock_path(self, session_id: str) -> Path | None: ...

    def _reset_for_restart(self, session_id: str | None) -> None: ...

    def _reset_spill_quota(self) -> None: ...

    def _cleanup_empty_session_dir(self) -> None: ...

    def _reset_turn_runtime_after_session_shutdown(self) -> None: ...

    async def _begin_session_transition(self, operation: str) -> str: ...

    async def _prepare_session_transition(self, operation: str) -> str | None: ...

    def _commit_session_transition(self, owner: str) -> None: ...

    def _finish_session_transition(self, owner: str) -> None: ...

    def _current_task_owns_session_transition_permit(self) -> bool: ...

    def _make_usage_event(self, *, session_id: str | None = None) -> UsageUpdate: ...

    async def _save_current_session(self, *, raise_on_error: bool = False) -> bool: ...

    async def _fire_session_end_hooks(self) -> None: ...

    async def _close_trajectory_log(self) -> None: ...

    async def _flush_recovery_checkpoint(self) -> None: ...

    async def shutdown(self, *, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None: ...

    async def start(
        self, profile: AgentProfile, *, operation: str = "startup", staged_loaded: LoadedSettings | None = None
    ) -> None: ...


def _session_pin_overrides(engine: SessionLifecycleHost) -> dict[str, Any]:
    """Snapshot the per-session pins every re-load must carry.

    Duplicated from the engine-controls companion rather than imported — the
    service layer must not import orchestration — and kept to the same
    contract: pins travel only while pinned, so an unpinned session picks up
    a changed env value on its next load.
    """
    overrides: dict[str, Any] = {}
    if engine._ask_user_timeout_pinned:
        overrides["ask_user_timeout_seconds"] = engine._settings.ask_user_timeout_seconds
    if engine._model_profile_pinned:
        overrides["model_profile"] = engine._settings.model_profile
        overrides["model_profile_override"] = engine._settings.model_profile_override
        overrides["model_profile_override_sub_agents"] = engine._settings.model_profile_override_sub_agents
    return overrides


def _reload_eval_context(engine: SessionLifecycleHost) -> EvalContext:
    """The launch mode's retry policy, passed *into* the load (as every re-load must)."""
    return EvalContext(frontend_default_max_transient_retries=engine._settings.frontend_default_max_transient_retries)


@dataclass(frozen=True)
class _ModelRestoreRollbackToken:
    """State needed to undo a model reapply before the build's commit installs it."""

    executor: Executor | None
    pointer: PointerToken


def _reapply_saved_model_profile(
    engine: SessionLifecycleHost,
    meta: SessionMeta | None,
    profile: AgentProfile,
    staged_loaded: LoadedSettings,
) -> tuple[LoadedSettings, _ModelRestoreRollbackToken | None]:
    """Apply a selectable saved model when the restored agent has no live binding.

    Transforms the staged load rather than installing anything: the selection
    goes live with the build's commit, exactly when the executor built from it
    does. Only the process pointer is written eagerly (§7: an intended fork,
    other hosts do see it) — but as SESSION, never ENV: the panel must not
    claim the shell exported what this restore wrote. The rollback token
    carries that one eager write, plus the executor identity that tells a
    pre-commit failure from a post-commit one.
    """
    registry = engine._model_registry
    if meta is None or not meta.model_profile_id or registry is None:
        return staged_loaded, None
    saved = registry.get(meta.model_profile_id)
    if saved is None or not is_model_profile_selectable(saved):
        return staged_loaded, None

    bound_profile_id = profile.model.profile_id
    if bound_profile_id and registry.get(bound_profile_id) is not None:
        return staged_loaded, None
    if saved.id == staged_loaded.settings.model_profile:
        return staged_loaded, None

    token = _ModelRestoreRollbackToken(
        executor=engine._executor,
        pointer=set_model_pointer(saved.id, origin=SettingOrigin(layer=Source.SESSION)),
    )
    # Through the overlay, and as the whole selection, not one field of it.
    # Setting only ``model_profile`` would leave the previous session's pin in
    # ``model_profile_override``, which outranks it — the restore would claim
    # the saved model and resolve the old session's.
    return loaded_with_active_model_profile(staged_loaded, saved, Source.SESSION), token


def _rollback_reapplied_model_profile(token: _ModelRestoreRollbackToken) -> None:
    """Undo the eager pointer write after a pre-commit start failure.

    The settings half needs no undo: the reapplied selection lived only on the
    staged load, and a build that failed before its commit never installed it.
    Value and origin go back together — a bare env write would leave the
    registry saying SESSION about a pointer the rollback just took away.
    A refusal is the registry reporting that someone chose a profile while
    this restore was failing; that choice is newer than the state this token
    describes, so it stands.
    """
    if not restore_model_pointer(token.pointer):
        logger.debug("Model pointer moved during a failed restore; leaving the newer selection in place")


async def reset_after_failed_startup_restore(engine: SessionLifecycleHost) -> None:
    """Discard every partially installed restore field before fresh startup.

    Restore can fail after installing the target session id, workspace, active
    lock, or a partial runtime. Suppress persistence while tearing those down so
    fallback startup cannot overwrite the saved target session.
    """
    engine._suppress_save = True
    try:
        try:
            await engine.shutdown(release_session_lock=True, close_mcp_cache=False)
        finally:
            # ``shutdown`` normally releases this; keep the failure path idempotent
            # if teardown raised after acquiring/installing the restore lock.
            engine._active_session_guard.release()
            engine._reset_turn_runtime_after_session_shutdown()
            engine._workspace = None
            engine._reset_for_restart(None)
    finally:
        engine._suppress_save = False
    # Fallback startup runs in the process cwd, not the failed target's root,
    # so the settings the next build reads must be re-derived for that root —
    # whatever the failed restore left installed described the wrong project
    # trust domain. Best-effort: the reset must never mask the original
    # failure, and the live settings remain a usable baseline without it.
    try:
        old_loaded = engine._loaded_settings
        candidate = await asyncio.to_thread(
            load_settings,
            project_root=Path(safe_getcwd()),
            eval_context=_reload_eval_context(engine),
            **_session_pin_overrides(engine),
        )
        routed, _ = route_restart_settings(reattribute_command_line(candidate, old_loaded), old_loaded)
        engine._settings_handle.install(routed)
    except Exception:
        logger.warning("Settings re-derivation for the fallback startup root failed", exc_info=True)


def reset_for_restart(engine: SessionLifecycleHost, session_id: str | None) -> None:
    """Reset per-session engine state in preparation for a fresh ``start()``."""
    engine._fsm.reset()
    engine._shutting_down = False
    engine._workspace = engine._workspace or Workspace.from_cwd()
    engine._session_id = session_id
    engine._reset_spill_quota()
    engine._turn_number = 0
    engine._runtime_meta = SessionRuntimeMetadata()
    engine._reminder_middleware = None
    engine._mutation_tracker = None
    engine._workspace_change_tracker.reset_for_restart()
    engine._mutation_coordinator = None
    engine._todo_tracker = None
    engine._recovered_from_sidecar = False


def _reset_restore_history_state(history_state: dict) -> dict:
    """Deep copy of the live history captured before a reset deletes session files.

    A failed reset restarts the engine from this copy, so it must own every
    layer (messages, contents lists, content objects) — aliasing the live
    objects of the session being shut down would let the restarted session
    share identities with a torn-down state.
    """
    return copy.deepcopy(history_state)


async def reset_session_to_welcome(
    engine: SessionLifecycleHost,
    session_id: str,
    *,
    write_lock_held: bool = False,
    after_delete: Callable[[], Awaitable[None]] | None = None,
    before_restart: Callable[[], None] | None = None,
) -> bool:
    """Delete ``session.json`` and snapshots, then reload to a welcome state."""
    transition_owner = (
        None
        if engine._current_task_owns_session_transition_permit()
        else await engine._begin_session_transition("reset")
    )
    reset_lock: FileLock | None = None
    try:
        session_dir = None
        lock_path = None
        if engine._persistence.state_store is not None:
            session_dir = engine._persistence.state_store.session_dir(session_id)
        if session_dir is not None and session_dir.is_dir():
            lock_path = engine._session_write_lock_path(session_id)
            if not write_lock_held:
                await engine._flush_recovery_checkpoint()
                if lock_path is not None:
                    reset_lock = _acquire_session_write_lock(lock_path, session_id)
                    if reset_lock is None:
                        return False

        profile = engine._agent_profile
        restore_state = (
            _reset_restore_history_state(engine._executor.history_state) if engine._executor is not None else None
        )
        restore_mutations = (
            copy.deepcopy(engine._mutation_tracker.serialize()) if engine._mutation_tracker is not None else None
        )
        if restore_state is not None and restore_mutations is not None:
            restore_state["chrys_mutations"] = copy.deepcopy(restore_mutations)
        restore_todos = engine._todo_tracker.serialize() if engine._todo_tracker is not None else None
        if restore_state is not None and restore_todos:
            restore_state["chrys_todos"] = copy.deepcopy(restore_todos)
        restore_turn_number = engine._turn_number
        restore_runtime_meta = copy.deepcopy(engine._runtime_meta)
        engine._suppress_save = True
        try:
            await engine.shutdown(release_session_lock=False, close_mcp_cache=False)
            engine._reset_turn_runtime_after_session_shutdown()

            if session_dir is not None and session_dir.is_dir():
                try:
                    _delete_reset_session_files(session_dir)
                    snap_dir = session_dir / "snapshots"
                    if snap_dir.is_dir():
                        shutil.rmtree(snap_dir, ignore_errors=True)
                except OSError:
                    logger.warning("Failed to delete session files for reset %s", session_id, exc_info=True)
                    if reset_lock is not None:
                        reset_lock.release()
                        reset_lock = None
                    if before_restart is not None:
                        before_restart()
                    await _restart_after_failed_reset(
                        engine,
                        session_id=session_id,
                        profile=profile,
                        state=restore_state,
                        mutations=restore_mutations,
                        todos=restore_todos,
                        turn_number=restore_turn_number,
                        runtime_meta=restore_runtime_meta,
                    )
                    return False

            if after_delete is not None:
                await after_delete()
            if session_dir is not None and session_dir.is_dir():
                # The restart below keeps this session id, so the directory is
                # written to again: a delete that cannot finish now must leave
                # no intent naming the path the new run lands in.
                cleanup_empty_session_dir_path(session_dir, path_reused=True)

            # Both the normal directory path and a concurrently removed
            # directory reach the restart below.  The new runtime can activate
            # trajectory recording from a session_start hook, which reacquires
            # this same path-keyed lock, so ownership must end before restart
            # regardless of what happened to the directory entry.
            if reset_lock is not None:
                reset_lock.release()
                reset_lock = None
            if before_restart is not None:
                before_restart()

            engine._reset_for_restart(session_id)
            await _reconcile_existing_spill_storage(engine)
            if profile is None:
                await engine.event_bus.publish(
                    Error(
                        code="no_agent_profile",
                        message="Agent profile not configured.",
                        session_id=session_id,
                    ),
                )
                return True
            await engine.start(profile, operation="reset")
            return True
        finally:
            engine._suppress_save = False
    finally:
        if reset_lock is not None:
            reset_lock.release()
        if transition_owner is not None:
            engine._finish_session_transition(transition_owner)


def cleanup_empty_session_dir(engine: SessionLifecycleHost) -> None:
    """Remove the current session directory if it was never saved."""
    cleanup_empty_session_dir_path(engine._session_dir)


def cleanup_empty_session_dir_path(session_dir: Path | None, *, path_reused: bool = False) -> None:
    """Remove *session_dir* when it has no restorable session files.

    ``path_reused`` is for a caller that restarts on this very directory (a
    reset keeps the session id).
    """
    if session_dir is None or not session_dir.is_dir():
        return
    if (session_dir / SESSION_FILE_NAME).exists():
        return
    if (session_dir / SESSION_BACKUP_FILE_NAME).exists():
        return
    if (session_dir / SESSION_RECOVERY_FILE_NAME).exists():
        return
    if _has_restorable_rollback_snapshots(session_dir):
        return
    if _has_committed_sub_agent_artifacts(session_dir):
        return
    if _has_recorded_trajectory(session_dir):
        return
    from chrys.service.trajectory.tombstone import delete_session_directory

    try:
        # Same disposal as an explicit delete: a trajectory writer that is
        # still alive (a stuck worker, a writer in another process) turns this
        # into a logical delete instead of a half-removed folder.
        delete_session_directory(session_dir, sessions_root=session_dir.parent, path_reused=path_reused)
    except OSError:
        logger.debug("Failed to remove empty session directory %s", session_dir, exc_info=True)


def _delete_reset_session_files(session_dir: Path) -> None:
    for filename in (SESSION_FILE_NAME, SESSION_BACKUP_FILE_NAME, SESSION_RECOVERY_FILE_NAME):
        path = session_dir / filename
        if path.exists():
            path.unlink()


def _acquire_session_write_lock(lock_path: Path, session_id: str) -> FileLock | None:
    lock = FileLock(lock_path, timeout=SESSION_WRITE_LOCK_TIMEOUT_SECONDS)
    try:
        lock.acquire()
        return lock
    except TimeoutError:
        logger.warning("Timed out acquiring session lock for reset %s", session_id, exc_info=True)
        return None
    except OSError:
        logger.warning("Failed to acquire session lock for reset %s", session_id, exc_info=True)
        return None


async def _restart_after_failed_reset(
    engine: SessionLifecycleHost,
    *,
    session_id: str,
    profile: AgentProfile | None,
    state: dict | None,
    mutations: dict[str, Any] | None,
    todos: list[dict[str, str]] | None,
    turn_number: int,
    runtime_meta: SessionRuntimeMetadata,
) -> None:
    engine._reset_for_restart(session_id)
    await _reconcile_existing_spill_storage(engine)
    engine._runtime_meta = runtime_meta
    engine._turn_number = turn_number
    mutation_state = mutations if mutations is not None else state.get("chrys_mutations") if state is not None else None
    if mutation_state is not None:
        session_dir = engine._session_dir
        if session_dir is not None:
            snapshot_store = SnapshotStore(session_dir, policy=SnapshotPolicy.from_settings(engine._settings))
            engine._mutation_tracker = MutationTracker.deserialize(mutation_state, snapshot_store)
    if state is not None:
        engine._workspace_change_tracker.restore(
            state.get("chrys_workspace_baseline"),
            engine._workspace,
            resolve_scope=engine._settings.workspace_change_notice,
        )
    # Rehydrate the todo tracker from the pre-shutdown snapshot: reattaching
    # ``history_state`` alone is not enough — save reads the TRACKER, so an
    # empty one would pop ``chrys_todos`` on the next save.
    todo_state = todos if todos is not None else state.get("chrys_todos") if state is not None else None
    if todo_state:
        engine._todo_tracker = TodoTracker()
        await engine._todo_tracker.restore(todo_state)
    if profile is None:
        return
    try:
        await engine.start(profile, operation="reset_failed")
    except Exception:
        logger.warning("Failed to restart session after reset failure %s", session_id, exc_info=True)
        return
    if state is not None and engine._executor is not None:
        if mutation_state is not None:
            state["chrys_mutations"] = copy.deepcopy(mutation_state)
        if todo_state:
            state["chrys_todos"] = copy.deepcopy(todo_state)
        engine._executor.history_state = state
        if engine._reminder_middleware is not None:
            engine._reminder_middleware.restore_phase4_state(state)
        engine._history.bind(engine._executor.history_state)


def _has_restorable_rollback_snapshots(session_dir: Path) -> bool:
    snap_dir = session_dir / "snapshots"
    if not snap_dir.is_dir():
        return False
    return any(parse_snapshot_turn(path) >= 1 for path in snap_dir.glob("*.json"))


def _has_recorded_trajectory(session_dir: Path) -> bool:
    """Whether this directory holds a log that outlives the conversation in it.

    A rollback only ever appends to the log — including the reset to welcome,
    which discards the session's messages but restarts on this very directory
    and reads its branch from the last line written. Only an explicit clear or
    delete takes a session's trajectory with it, and neither comes through
    here.
    """
    from chrys.service.trajectory.session import trajectory_events_path

    events = trajectory_events_path(session_dir)
    try:
        return events.stat().st_size > 0
    except FileNotFoundError:
        return False
    except OSError:
        # This is the guard that keeps an audit log from being deleted, so a
        # probe that cannot answer keeps the directory: "I could not look" is
        # not "there is nothing there".
        return True


def _has_committed_sub_agent_artifacts(session_dir: Path) -> bool:
    sub_agents = session_dir / "sub_agents"
    if not sub_agents.is_dir():
        return False
    for path in sub_agents.rglob("*.json"):
        if path.name.endswith(".tmp"):
            continue
        if path.is_file():
            return True
    return False


async def on_new_session(engine: SessionLifecycleHost, _event: SessionNew) -> None:
    """Handle new session request: save current session, then start fresh."""
    transition_owner = await engine._begin_session_transition("new_session")
    try:
        # Sample the live profile after the transition fence so a queued
        # profile switch that won the rebuild gate determines the new session.
        profile = engine._agent_profile
        if profile is None:
            return
        await _start_fresh_session(engine, profile)
    finally:
        engine._finish_session_transition(transition_owner)


async def _start_fresh_session(engine: SessionLifecycleHost, profile: AgentProfile) -> None:
    """Shut the current session down and start an empty one under *profile*.

    Callers hold a COMMITTED session transition (``_begin_session_transition``
    or prepare + ``_commit_session_transition``); the shutdown here relies on
    the fence to have already invalidated the old session's turn state.
    """
    await engine.shutdown(close_mcp_cache=False)
    engine._reset_turn_runtime_after_session_shutdown()
    engine._cleanup_empty_session_dir()
    engine._reset_for_restart(None)
    await engine.start(profile, operation="new_session")


async def on_session_clear(engine: SessionLifecycleHost, event: SessionClear) -> None:
    """Delete the ACTIVE session and start a fresh one as one fenced transition.

    Prompt admission is closed before the deletion starts and stays closed
    until the fresh session is ready, so no prompt can be admitted against the
    detached session in between.  The transition is committed only after the
    delete succeeded: a failed delete leaves the current session and its turn
    state intact (admission simply reopens), reports
    ``Error(code="session_clear_failed")``, and never starts a new session.
    """
    transition_owner = await engine._prepare_session_transition("clear")
    if transition_owner is None:  # No owner filter was given, so this cannot happen.
        raise RuntimeError("Session transition acquisition unexpectedly failed")
    failure_message: str | None = None
    try:
        # Sample under the fence: a queued restore/new/switch cannot move the
        # active session while we hold the gate.
        profile = engine._agent_profile
        if not engine._active_session_guard.owns(event.session_id):
            failure_message = "Only the active session can be cleared"
        elif profile is None:
            failure_message = "No agent profile is active"
        else:
            failure = await _delete_session_reporting(engine, event.session_id)
            if failure is not None:
                failure_message = failure.message
            else:
                engine._commit_session_transition(transition_owner)
                await _start_fresh_session(engine, profile)
    finally:
        engine._finish_session_transition(transition_owner)
    # Published after the fence is released — an Error handler must not find
    # the gate still held by the failed clear.
    if failure_message is not None:
        await engine.event_bus.publish(
            Error(code="session_clear_failed", message=failure_message, session_id=event.session_id),
        )


async def on_session_restore(engine: SessionLifecycleHost, event: SessionRestore) -> None:
    """Handle session restore by loading saved state and rebuilding the agent."""
    from chrys.service.state.locks import SESSION_RESTORE_ACTIVE_LOCK_TIMEOUT_SECONDS

    if engine._persistence.state_store is None:
        await engine.event_bus.publish(
            Error(code="no_state_store", message="State store not configured", session_id=event.session_id)
        )
        return

    restoring_current = engine._active_session_guard.owns(event.session_id)
    target_lock: FileLock | None = None
    transition_owner: str | None = None
    if not restoring_current:
        try:
            target_lock = await asyncio.to_thread(
                engine._active_session_guard.acquire_for_restore,
                event.session_id,
                timeout=SESSION_RESTORE_ACTIVE_LOCK_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            await engine.event_bus.publish(
                Error(
                    code="session_in_use",
                    message=engine._active_session_guard.conflict_message(event.session_id),
                    session_id=event.session_id,
                ),
            )
            return

    try:
        if event.ignore_recovery:
            try:
                await asyncio.to_thread(engine._persistence.state_store.delete_recovery_session, event.session_id)
            except Exception:
                logger.debug("Failed to delete ignored recovery sidecar for %s", event.session_id, exc_info=True)

        prefer_recovery = not event.ignore_recovery and (not restoring_current or engine._recovered_from_sidecar)

        state = await engine._persistence.load_session(event.session_id, prefer_recovery=prefer_recovery)
        if state is None:
            if event.ignore_recovery and not restoring_current:
                cleanup_empty_session_dir_path(engine._persistence.state_store.session_dir(event.session_id))
            await engine.event_bus.publish(
                Error(
                    code="session_not_found",
                    message=f"Session '{event.session_id}' not found",
                    session_id=event.session_id,
                )
            )
            return

        # Resolve meta by id directly. ``list_sessions`` relies on
        # ``iterdir`` which can briefly miss a just-written session
        # folder on Windows; a direct envelope read avoids that race.
        meta = await engine._persistence.load_session_meta(event.session_id, prefer_recovery=prefer_recovery)
        recovered_from_sidecar = (
            await engine._persistence.recovery_session_wins(event.session_id) if prefer_recovery else False
        )

        cwd_warning = ""
        saved_cwd = meta.primary_cwd if meta and meta.primary_cwd else ""
        cwd_overridden = bool(event.primary_cwd) and event.primary_cwd != saved_cwd
        target_cwd = event.primary_cwd or saved_cwd or _workspace_cwd(engine)
        target_cwd_exists = os.path.isdir(target_cwd)
        if target_cwd and not target_cwd_exists:
            cwd_warning = f"Working directory no longer exists: {surrogate_safe_text(target_cwd)}"

        profile_name = event.profile_name or (meta.agent_profile if meta else "")
        saved_profile_id = meta.agent_profile_id if meta else ""
        profile_selector = event.profile_name or saved_profile_id or profile_name
        profile: AgentProfile | None = None
        profile_switch: tuple[str, str] | None = None
        profile_resolution_warning: Warning | None = None
        if profile_selector and engine._agent_registry:
            if event.profile_name:
                resolved = engine._agent_registry.resolve_selector(event.profile_name)
            elif saved_profile_id:
                resolved = engine._agent_registry.get_by_id(
                    saved_profile_id,
                    disambiguating_name=profile_name,
                )
            else:
                resolved = engine._agent_registry.get(profile_name)
            if resolved is not None:
                profile = resolved
                profile_name = resolved.name
                if event.profile_name and meta and meta.agent_profile and meta.agent_profile != resolved.name:
                    from_display = meta.agent_display_name or meta.agent_profile
                    to_display = resolved.display_name or resolved.name
                    if _restore_profile_override_switch(
                        state,
                        from_profile=meta.agent_profile,
                        from_display=from_display,
                        to_profile=resolved.name,
                        to_display=to_display,
                    ):
                        profile_switch = (from_display, to_display)
            else:
                saved_profile_name = profile_name
                if event.profile_name:
                    logger.warning(
                        "Requested agent profile '%s' could not be resolved; stopping session restore",
                        event.profile_name,
                    )
                    await engine.event_bus.publish(
                        Error(
                            code="requested_agent_profile_unresolved",
                            message=(
                                f"Requested agent profile '{event.profile_name}' could not be resolved; "
                                "session restore was stopped."
                            ),
                            display_message=_RESTORE_REQUESTED_AGENT_PROFILE_UNRESOLVED.bind(
                                profile=event.profile_name
                            ),
                            session_id=event.session_id,
                        )
                    )
                    return
                unresolved_saved_identity = bool(saved_profile_id)
                saved_display = (meta.agent_display_name if meta else "") or saved_profile_name or saved_profile_id
                if engine._agent_profile is not None:
                    profile_name = engine._agent_profile.name
                    logger.warning(
                        "Agent profile '%s' not found, keeping current profile '%s'",
                        saved_profile_name,
                        profile_name,
                    )
                    if unresolved_saved_identity:
                        current_display = engine._agent_profile.display_name or profile_name
                        profile_resolution_warning = Warning(
                            code="saved_agent_profile_unresolved",
                            message=(
                                f"Saved agent profile id '{saved_profile_id}' for '{saved_profile_name}' "
                                f"could not be uniquely resolved; keeping current profile '{profile_name}'."
                            ),
                            display_message=_RESTORE_AGENT_PROFILE_UNRESOLVED_USING_CURRENT.bind(
                                saved=saved_display,
                                current=current_display,
                            ),
                            session_id=event.session_id,
                        )
                elif unresolved_saved_identity:
                    logger.warning(
                        "Agent profile id '%s' for saved profile '%s' could not be uniquely resolved; "
                        "stopping restore because no current profile is active",
                        saved_profile_id,
                        saved_profile_name,
                    )
                    await engine.event_bus.publish(
                        Error(
                            code="saved_agent_profile_unresolved",
                            message=(
                                f"Saved agent profile id '{saved_profile_id}' for '{saved_profile_name}' "
                                "could not be uniquely resolved, and no current profile is active."
                            ),
                            display_message=_RESTORE_AGENT_PROFILE_UNRESOLVED.bind(saved=saved_display),
                            session_id=event.session_id,
                        )
                    )
                    return
                else:
                    available = engine._agent_registry.list_profiles()
                    if available:
                        profile = available[0]
                        logger.warning(
                            "Agent profile '%s' not found, falling back to '%s'",
                            saved_profile_name,
                            profile.name,
                        )
                        profile_name = profile.name

        if event.working_dirs is not None:
            # Client-supplied additionalDirectories are authoritative (ACP load
            # semantics): the list REPLACES the saved roots so a client can narrow,
            # swap, or clear (empty list) workspace scope. Keep the primary first to
            # mirror new-session storage; service-session compatibility canonicalizes
            # away the primary (see _extra_roots), so its inclusion here is cosmetic.
            merged_dirs = [target_cwd]
            seen_dirs: set[str] = {target_cwd}
            for path in event.working_dirs:
                if path and path not in seen_dirs:
                    seen_dirs.add(path)
                    merged_dirs.append(path)
        else:
            # Caller did not specify roots: keep the saved working_dirs verbatim (it may
            # include the primary cwd, and _can_restore_service_session compares the path
            # list exactly). A moved cwd invalidates the saved layout, so drop it.
            merged_dirs = [] if cwd_overridden else list(meta.working_dirs if meta else [])
        session_ws = Workspace(
            primary_cwd=target_cwd,
            working_dirs=_working_dirs_from_paths(merged_dirs, target_cwd),
        )

        # The transition boundary is taken in two phases. The *prepared* fence
        # comes first: it shares the rebuild gate with settings reloads and
        # model switches, so everything read below — loaded settings, eval
        # context, session pins — is a committed state no concurrent rebuild
        # can move. Snapshotting before the fence would let a reload commit
        # during the load's thread hop and then be silently overwritten by a
        # staged load routed against the stale copy. But the *commit* — which
        # bumps the session generation and invalidates the old session's turn
        # state, retries and injection — waits until the load has succeeded:
        # an unreadable config file must abort the restore with the current
        # session genuinely intact, not admission-open but generation-dead.
        if not engine._current_task_owns_session_transition_permit():
            transition_owner = await engine._prepare_session_transition("restore")
            if transition_owner is None:  # No owner filter was given, so this cannot happen.
                raise RuntimeError("Session transition acquisition unexpectedly failed")

        # The restore crosses into the target session's project trust domain:
        # its settings are re-derived for that root so the build below reads
        # them, exactly as a workspace change would. Loaded here — before
        # anything is torn down — so an unreadable config file aborts the
        # restore with the current session intact.
        old_loaded = engine._loaded_settings
        candidate = await asyncio.to_thread(
            load_settings,
            project_root=Path(target_cwd),
            eval_context=_reload_eval_context(engine),
            **_session_pin_overrides(engine),
        )
        staged_loaded, _ = route_restart_settings(reattribute_command_line(candidate, old_loaded), old_loaded)
        if profile is None:
            profile = engine._agent_profile
            if not profile_name and profile is not None:
                profile_name = profile.name
        if transition_owner is not None:
            engine._commit_session_transition(transition_owner)
        await engine.shutdown(release_session_lock=not restoring_current, close_mcp_cache=False)
        engine._reset_turn_runtime_after_session_shutdown()
        engine._cleanup_empty_session_dir()

        engine._workspace = session_ws

        if target_lock is not None:
            engine._active_session_guard.install(event.session_id, target_lock)
            target_lock = None
    except BaseException as exc:
        if transition_owner is not None:
            engine._finish_session_transition(transition_owner)
            transition_owner = None
        if isinstance(exc, Exception):
            # The interactive callers publish this event with the bus's
            # default swallow-and-log delivery: without a terminal event the
            # restore loading UI has nothing to clear it. Published after the
            # fence is released — an Error handler must not find the gate
            # still held by the failed restore.
            await engine.event_bus.publish(
                Error(code="session_restore_failed", message=str(exc), session_id=event.session_id)
            )
        raise
    finally:
        if target_lock is not None:
            target_lock.release()

    try:
        await _hydrate_restored_session(
            engine,
            event=event,
            state=state,
            meta=meta,
            profile=profile,
            profile_name=profile_name,
            profile_switch=profile_switch,
            profile_resolution_warning=profile_resolution_warning,
            recovered_from_sidecar=recovered_from_sidecar,
            cwd_warning=cwd_warning,
            target_cwd=target_cwd,
            staged_loaded=staged_loaded,
        )
    finally:
        if transition_owner is not None:
            engine._finish_session_transition(transition_owner)


async def _hydrate_restored_session(
    engine: SessionLifecycleHost,
    *,
    event: SessionRestore,
    state: dict[str, Any],
    meta: SessionMeta | None,
    profile: AgentProfile | None,
    profile_name: str,
    profile_switch: tuple[str, str] | None,
    profile_resolution_warning: Warning | None,
    recovered_from_sidecar: bool,
    cwd_warning: str,
    target_cwd: str,
    staged_loaded: LoadedSettings,
) -> None:
    """Hydrate live engine state after restore shutdown while admission remains closed."""
    engine._fsm.reset()
    engine._shutting_down = False
    engine._session_id = event.session_id
    engine._reset_spill_quota()
    engine._recovered_from_sidecar = recovered_from_sidecar
    session_dir = engine._session_dir
    # The staged settings describe the target session; the live ones still
    # belong to the previous session until the build below commits them.
    engine._workspace_change_tracker.restore(
        state.get("chrys_workspace_baseline"),
        engine._workspace,
        resolve_scope=staged_loaded.settings.workspace_change_notice,
    )

    spill_reconciliation = await _reconcile_existing_spill_storage(engine)

    # Installing ``event.session_id`` above makes the engine session directory
    # available for the remainder of successful hydration. The policy reads
    # the staged settings: they are what the build below commits, and the
    # tracker hydrated here outlives that commit.
    snapshot_store = SnapshotStore(
        cast("Path", session_dir), policy=SnapshotPolicy.from_settings(staged_loaded.settings)
    )
    if state and state.get("chrys_mutations"):
        engine._mutation_tracker = MutationTracker.deserialize(state["chrys_mutations"], snapshot_store)
    else:
        engine._mutation_tracker = MutationTracker(snapshot_store)
    # Hydrate todos before ``restore_last_words`` below: the reminder
    # middleware re-captures the todo section from the tracker at restore.
    engine._todo_tracker = TodoTracker()
    if state:
        await engine._todo_tracker.restore(state.get("chrys_todos"))
    # The restored session gets its own coordinator (session id changed;
    # the previous one was closed during shutdown).  Construction builds
    # it during ``start()`` below.
    engine._mutation_coordinator = None

    # Install the target session's runtime metadata BEFORE the build:
    # construction's ``_publish_results`` hydrates the fresh strategy from
    # ``engine._runtime_meta``, so leaving the previous session's metadata in
    # place would leak its calibration record into this session's strategy
    # whenever the build fingerprints happen to match.
    engine._runtime_meta = SessionRuntimeMetadata.from_state_dict(state)

    if profile is not None:
        rollback_token = None
        if event.apply_saved_model:
            staged_loaded, rollback_token = _reapply_saved_model_profile(engine, meta, profile, staged_loaded)
        try:
            await engine.start(profile, operation="restore", staged_loaded=staged_loaded)
        except BaseException:
            if rollback_token is not None and engine._executor is rollback_token.executor:
                _rollback_reapplied_model_profile(rollback_token)
            raise
        if profile_switch is not None and engine._reminder_middleware is not None:
            engine._reminder_middleware.set_profile_switch(*profile_switch)
    else:
        # Nothing to build, so no commit will install the staged load; this
        # degenerate path installs it directly, like a reload with nothing
        # built.
        engine._settings_handle.install(staged_loaded)

    # The verdicts describe the target root's files — the same report a
    # reload or workspace change makes — published only now that the staged
    # load is in force (committed by the build above, or installed by the
    # degenerate path), under the restored session's id.
    restore_warnings = settings_warning_events(staged_loaded)
    if profile_resolution_warning is not None:
        restore_warnings.insert(0, profile_resolution_warning)
    for warning in restore_warnings:
        await engine.event_bus.publish(replace(warning, session_id=engine._session_id))

    engine._turn_number = state.get("turn_counter", 0) if state else 0

    if engine._executor and state:
        engine._executor.history_state = state
        if engine._reminder_middleware is not None:
            engine._reminder_middleware.restore_phase4_state(
                state,
                available_relative_paths=spill_reconciliation.available_relative_paths,
            )
        if meta and meta.service_session_id:
            if _can_restore_service_session(engine, meta):
                engine._executor.service_session_id = meta.service_session_id
            else:
                engine._executor.service_session_id = ""
                await engine.event_bus.publish(
                    Warning(
                        code="service_session_incompatible",
                        message=(
                            "This session was saved with an OpenAI Responses service session. "
                            "The active agent/model profile, workspace, service endpoint, or storage mode "
                            "is not compatible, "
                            f"so {APP_DISPLAY_NAME} will continue from local history only."
                        ),
                        display_message=_RESTORE_SERVICE_SESSION_INCOMPATIBLE.bind(app=APP_DISPLAY_NAME),
                        session_id=event.session_id,
                    )
                )
        engine._history.bind(engine._executor.history_state)

        # ``engine.start`` above already hydrated the fresh strategy through
        # ``_publish_results``; this covers the no-rebuild path (``profile is
        # None``) where the surviving executor's strategy is the only target.
        # The provenance gate makes the repeat call idempotent.
        if engine._runtime_meta.context_calibration is not None and engine._executor.compaction_strategy is not None:
            engine._runtime_meta.restore_context_calibration(
                engine._executor.compaction_strategy,
                model_profile_fingerprint=engine._model_profile_fingerprint,
                agent_profile_fingerprint=engine._agent_profile_fingerprint,
            )

    restored_history_changed = profile_switch is not None
    paused_records: list[dict[str, Any]] = []
    injected_paused_results = 0
    artifact_service = (
        SubAgentSessionArtifactService(engine._session_dir)
        if engine._session_dir is not None and engine._executor is not None
        else None
    )
    if artifact_service is not None and engine._executor is not None:
        paused_records = artifact_service.drain_paused_records()
    # Repair dangling sub-agent function_calls even when drain dropped or
    # quarantined every record (over-cap/corrupt): otherwise the parent
    # assistant call is left with no tool_result and the next turn is
    # provider-invalid. The live tool registry resolves dangling calls whose
    # record did not survive drain — records only enrich the error text.
    registry_sub_agent_names = (
        set(engine._sub_agent_tools.tool_names()) if engine._sub_agent_tools is not None else set()
    )
    record_sub_agent_names = {
        tool_name for record in paused_records if isinstance(tool_name := record.get("tool_name"), str) and tool_name
    }
    # Always attempt injection — never gate it on records/registry being
    # non-empty. A persisted sub-agent function_call self-identifies via its
    # ``_chrys_tool_kind`` marker, so a dangling call must be repaired even
    # when drain lost every record AND the restored profile no longer registers
    # the tool (removed/disabled/depth-skipped). The method self-gates on the
    # history being bound and does nothing when there is no dangling call.
    injected_paused_results = engine._history.inject_error_results_for_sub_agents(
        paused_records, registry_sub_agent_names | record_sub_agent_names
    )
    restored_history_changed = restored_history_changed or injected_paused_results > 0
    if paused_records:
        logger.info(
            "Session restore: loaded %d paused sub-agent record(s), injected %d error tool_result(s)",
            len(paused_records),
            injected_paused_results,
        )
        discarded_tools = sorted({r.get("tool_name", "?") for r in paused_records})
        tool_names = ", ".join(discarded_tools)
        await engine.event_bus.publish(
            Warning(
                code="sub_agents_reload_discarded",
                message=(
                    f"{len(paused_records)} paused sub-agent(s) from a previous session were discarded: {tool_names}"
                ),
                display_message=_RESTORE_SUB_AGENTS_DISCARDED.bind(
                    discarded=len(paused_records),
                    names=DisplaySequence(tuple(discarded_tools)),
                ),
                session_id=event.session_id,
            ),
        )

    if engine._history.is_bound:
        awaiting_count = sum(
            1
            for message in engine._history.messages
            if message.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.AWAITING_SUB_AGENTS
        )
        engine._history.remove_awaiting_sub_agents_marker()
        restored_history_changed = restored_history_changed or awaiting_count > 0
        if (awaiting_count > 0 or injected_paused_results > 0) and engine._history.trailing_status_marker() is None:
            engine._history.insert_interrupted_marker(
                reason=format_message(SUB_AGENT_STATE_DISCARDED_MESSAGE.bind()),
                source="error",
                status_code=HistoryMarkerKind.STATUS_SUB_AGENT_STATE_DISCARDED,
            )
            restored_history_changed = True
        if artifact_service is not None:
            orphaned = artifact_service.reconcile_orphaned_running_logs(engine._history.messages, paused_records)
            if orphaned:
                logger.info("Session restore: marked %d sub-agent audit log(s) orphaned", orphaned)

        _restore_terminal_fsm_from_history(engine)

    if restored_history_changed:
        saved = False
        try:
            saved = await engine._save_current_session(raise_on_error=bool(paused_records))
        except Exception:
            logger.warning("Session restore: failed to save repaired session; preserving paused sub-agent records")
        if saved and paused_records and artifact_service is not None:
            artifact_service.finalize_restored_paused_records(paused_records)
    elif paused_records and artifact_service is not None:
        artifact_service.archive_unconsumed_restored_paused_records(paused_records)

    await engine.event_bus.publish(
        SessionRestored(
            session_id=event.session_id,
            agent_profile=profile_name,
            display_name=profile.display_name if profile else "",
            initial_agent_profile=(
                meta.agent_profile_history[0]
                if meta is not None and meta.agent_profile_history
                else profile.display_name
                if profile
                else profile_name
            ),
            message_count=meta.message_count if meta else 0,
            cwd_warning=cwd_warning,
            primary_cwd=target_cwd,
            recovered_from_sidecar=recovered_from_sidecar,
            working_dirs=(
                [working_dir.path for working_dir in engine._workspace.working_dirs]
                if engine._workspace is not None
                else []
            ),
        ),
    )
    # Restore-time UsageUpdate must follow SessionRestored so the TUI has
    # already bound the new ``_main_usage_source_id`` before classifying it as
    # the session window — otherwise the chat panel keeps stale window tokens
    # from the previous session.  Route through the ordered chain so any
    # pending sub-agent UsageUpdate that was in-flight before the switch can't
    # overtake this restored snapshot.
    usage.enqueue_usage_event(engine, engine._make_usage_event(session_id=event.session_id))
    await _fire_session_restored_hook(engine, restored_session_id=event.session_id, profile_name=profile_name)


async def _fire_session_restored_hook(
    engine: SessionLifecycleHost,
    *,
    restored_session_id: str,
    profile_name: str,
) -> None:
    if engine._hook_manager is None:
        return
    from chrys.service.hooks.events import HookEvent

    if not engine._hook_manager.has_hooks_for(HookEvent.SESSION_RESTORED):
        return
    await engine._hook_manager.fire(
        HookEvent.SESSION_RESTORED,
        {
            "session_id": engine._session_id,
            "profile": profile_name,
            "cwd": _workspace_cwd(engine),
            "restored_session_id": restored_session_id,
        },
        scope="session",
    )


@dataclass(frozen=True, slots=True)
class _SessionDeleteFailure:
    """Why a session delete did not happen; the current session is intact."""

    code: str
    message: str


async def on_session_delete(engine: SessionLifecycleHost, event: SessionDelete) -> None:
    """Handle session deletion."""
    failure = await _delete_session_reporting(engine, event.session_id)
    if failure is not None:
        await engine.event_bus.publish(
            Error(code=failure.code, message=failure.message, session_id=event.session_id),
        )


async def _delete_session_reporting(engine: SessionLifecycleHost, session_id: str) -> _SessionDeleteFailure | None:
    """Delete *session_id* from disk, detaching the engine first when it is the active session.

    Publishes ``SessionDeleted`` on success and returns ``None``; on failure
    returns the reason with the engine's session ownership restored, leaving
    the caller to report it (``on_session_delete`` publishes the raw code,
    ``on_session_clear`` folds it into ``session_clear_failed``).
    """
    if engine._persistence.state_store is None:
        return _SessionDeleteFailure(code="no_state_store", message="State store not configured")

    release_current_lock = engine._active_session_guard.owns(session_id)
    detached_session_id = engine._session_id if release_current_lock else None
    otel_sink = None
    if release_current_lock:
        from chrys.foundation.observability.sink import get_otel_sink

        # Deleting the live session is its end: fire ``session_end`` now and
        # wait for it (async hooks included), while the id and the files still
        # exist, so hooks keep the "before teardown" contract; the shutdown
        # that follows (clear / delete-current -> new) skips the duplicate.  A
        # failed delete re-arms it below — the session then lives on and hooks
        # see one early ``session_end`` plus the real one at shutdown.  This
        # duplicate is accepted by design: exactly-once would need the hook to
        # run only after the delete is known to succeed, i.e. under the store's
        # write lock from a thread (rejected — user hooks may touch the store),
        # and NOT re-arming would instead drop the real ``session_end`` of a
        # session that keeps going, which loses more than a repeat costs.
        await engine._fire_session_end_hooks()
        # The trajectory writer holds the log's lease, and a leased directory
        # is tombstoned instead of removed — close it so the delete below is
        # physical.
        await engine._close_trajectory_log()
        otel_sink = get_otel_sink()
        engine._session_id = None

    delete_succeeded = False
    # Not cancellation-safe on purpose: the delete runs in a thread and keeps
    # going if this task is cancelled, and ``CancelledError`` skips the restore
    # branches below, leaving the engine detached with the guard held.  The
    # only publisher of ``SessionClear``/``SessionDelete`` is a MainScreen
    # worker, which Textual cancels solely on app exit — and there a detached
    # engine is exactly right: ``shutdown()`` must not save (resurrect) a
    # session whose deletion may have just completed.  Restoring ``_session_id``
    # blindly on cancel would do precisely that; shield-and-reconcile would only
    # release a guard and publish an ack inside an app that is going away.
    try:
        await engine._persistence.delete_session(session_id, allow_active=release_current_lock)
        delete_succeeded = True
    except TimeoutError as exc:
        if release_current_lock:
            engine._session_id = detached_session_id
            engine._session_end_fired = False
            return _SessionDeleteFailure(
                code="session_busy",
                message=f"Timed out waiting for session write lock: {exc}",
            )
        return _SessionDeleteFailure(
            code="session_in_use",
            message=engine._active_session_guard.conflict_message(session_id),
        )
    except Exception as exc:
        logger.exception("Failed to delete session %s", session_id)
        if release_current_lock:
            engine._session_id = detached_session_id
            engine._session_end_fired = False
        return _SessionDeleteFailure(code="session_delete_failed", message=f"Failed to delete session: {exc}")
    finally:
        if release_current_lock and delete_succeeded:
            engine._active_session_guard.release()

    # Post-delete cleanup runs only after the delete has committed. Keep it out
    # of the try above: a failure here must not reach the delete-failure handler
    # (which would wrongly restore engine._session_id to a now-deleted session)
    # nor be reported as a delete failure.
    if release_current_lock and otel_sink is not None:
        try:
            otel_sink.deactivate()
        except Exception:
            logger.exception("Failed to deactivate OTel sink after deleting session %s", session_id)
    await engine.event_bus.publish(SessionDeleted(session_id=session_id))
    return None
