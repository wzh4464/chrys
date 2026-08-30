# Copyright (c) 2026 Chrys. All rights reserved.

"""Engine-internal profile, settings, workspace, and approval controls."""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from chrys.foundation.config.context import EvalContext
from chrys.foundation.config.process_settings import reattribute_command_line, route_restart_settings
from chrys.foundation.config.settings import Settings
from chrys.foundation.config.settings_store import LoadedSettings, SettingsHandle, load_settings
from chrys.foundation.config.spec import Source
from chrys.foundation.config.warnings import settings_warning_events
from chrys.foundation.events.types import (
    AgentProfileSwitch,
    AgentRuntimeDetails,
    ApprovalModeUpdated,
    Error,
    ModelProfileSwitched,
    ProfileSwitched,
    SetApprovalMode,
    SetModelProfile,
    SettingsReload,
    SettingsReloaded,
    Warning,
    WorkspaceChange,
    WorkspaceUpdated,
)
from chrys.foundation.i18n import DisplaySequence, msg
from chrys.foundation.i18n.formatting import format_message
from chrys.foundation.models.workspace import Workspace
from chrys.foundation.platform import safe_getcwd
from chrys.service.approval.policy import ApprovalMode

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.orchestration.engine.executor import Executor
    from chrys.orchestration.sub_agents.tools import SubAgentTools
    from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker
    from chrys.service.profiles.agents.registry import AgentProfileRegistry
    from chrys.service.profiles.agents.schema import AgentProfile
    from chrys.service.profiles.models.schema import ModelProfile


logger = logging.getLogger(__name__)

_CONTROLS_NO_REGISTRY = msg(
    "controls.no_registry",
    fallback="No profile registry configured — cannot switch profiles",
)
_CONTROLS_PROFILE_NOT_FOUND = msg(
    "controls.profile_not_found",
    fallback="Profile '{profile_name}' not found",
)
_CONTROLS_MODEL_SWITCH_NOT_READY = msg(
    "controls.model_switch_not_ready",
    fallback="No active agent — cannot switch model",
)
_CONTROLS_WORKSPACE_SWITCH_NOT_READY = msg(
    "controls.workspace_switch_not_ready",
    fallback="No active agent — cannot change workspace",
)
_CONTROLS_SETTINGS_RESTART_REQUIRED = msg(
    "controls.settings_restart_required",
    fallback="Saved; takes effect after restart: {keys}.",
)

PersistApprovalModeFn = Callable[[str], None]
RebuildPermitDeniedReason = Literal["shutdown", "session_changed", "superseded", "load_active", "busy", "not_ready"]


@dataclasses.dataclass(frozen=True)
class RebuildControlToken:
    """Captured owner clocks for one runtime-control rebuild request."""

    session_id: str | None
    session_generation: int
    build_generation: int
    load_generation: int


@dataclasses.dataclass(frozen=True)
class RebuildPermit:
    """Serialized rebuild-admission permit."""

    permit_id: int
    owner: str
    token: RebuildControlToken


@dataclasses.dataclass(frozen=True)
class RebuildPermitDenied:
    """Terminal denial for a rebuild-control request."""

    reason: RebuildPermitDeniedReason
    code: str
    message: str


class EngineControlHost(Protocol):
    """Engine state needed by engine-control event handlers."""

    _session_id: str | None
    _bus: EventBus
    _settings_handle: SettingsHandle
    _approval_mode: ApprovalMode
    _executor: Executor | None
    _sub_agent_tools: SubAgentTools | None
    _agent_registry: AgentProfileRegistry | None
    _agent_profile: AgentProfile | None
    _workspace: Workspace | None
    _active_profile: ModelProfile | None
    _runtime_details: AgentRuntimeDetails
    _model_profile_pinned: bool
    _ask_user_timeout_pinned: bool
    _workspace_change_tracker: WorkspaceChangeTracker

    @property
    def _settings(self) -> Settings:
        """Read-only: settings change through the handle, not the holder."""
        ...

    @property
    def _loaded_settings(self) -> LoadedSettings: ...

    def capture_rebuild_control_token(self) -> RebuildControlToken: ...

    async def acquire_rebuild_permit(self, token: RebuildControlToken) -> RebuildPermit | RebuildPermitDenied: ...

    def release_rebuild_permit(self, permit: RebuildPermit) -> None: ...

    async def start_with_rebuild_permit(
        self,
        permit: RebuildPermit,
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None: ...

    async def soft_restart_with_rebuild_permit(
        self,
        permit: RebuildPermit,
        new_profile: AgentProfile,
        workspace: Workspace | None = None,
        *,
        operation: str = "switch",
        staged_loaded: LoadedSettings | None = None,
    ) -> None: ...

    def current_profile_snapshot(self) -> ProfileSwitched: ...

    async def start(
        self,
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
    ) -> None: ...

    async def _soft_restart(
        self,
        new_profile: AgentProfile,
        workspace: Workspace | None = None,
        *,
        operation: str = "switch",
        staged_loaded: LoadedSettings | None = None,
    ) -> None: ...


async def on_set_approval_mode(
    host: EngineControlHost,
    event: SetApprovalMode,
    *,
    persist_approval_mode_fn: PersistApprovalModeFn,
) -> None:
    """Update the active approval mode on the running middleware."""
    try:
        mode = ApprovalMode(event.mode)
    except ValueError:
        logger.warning("Unknown approval mode: %r", event.mode)
        return
    host._approval_mode = mode
    if host._executor is not None:
        host._executor.set_approval_mode(mode)
    # Propagate to sub-agent tools so each fresh sub-agent invocation
    # constructs its ApprovalMiddleware with the live mode.
    if host._sub_agent_tools is not None:
        host._sub_agent_tools.set_approval_mode(mode)
    if event.persist:
        # Persist the user's choice so it survives restart.  BYPASS is
        # downgraded to AUTO inside ``persist_approval_mode`` to avoid
        # booting into unattended auto-approval on next launch — mirror
        # that same downgrade onto the in-memory Settings so a later
        # ``SettingsReload`` stays consistent with what's on disk.
        #
        # Through the overlay rather than onto the field: this key is
        # ``Risk.DANGEROUS``, so a rejected value seals it at ``manual``, and
        # an in-place write would leave that seal on top of the mode the user
        # just chose.
        persist_approval_mode_fn(mode.value)
        host._settings_handle.override(default_approval_mode="auto" if mode is ApprovalMode.BYPASS else mode.value)
    await host._bus.publish(ApprovalModeUpdated(mode=mode.value, session_id=host._session_id))


async def on_profile_switch(host: EngineControlHost, event: AgentProfileSwitch) -> None:
    """Handle agent profile switch — preserves conversation history."""
    token = host.capture_rebuild_control_token()
    if host._agent_registry is None:
        await host._bus.publish(
            Error(
                code="no_registry",
                message="No profile registry configured — cannot switch profiles",
                display_message=_CONTROLS_NO_REGISTRY.bind(),
                session_id=host._session_id,
            )
        )
        return

    new_profile = host._agent_registry.get(event.profile_name)
    if new_profile is None:
        await host._bus.publish(
            Error(
                code="profile_not_found",
                message=f"Profile '{event.profile_name}' not found",
                display_message=_CONTROLS_PROFILE_NOT_FOUND.bind(profile_name=event.profile_name),
                session_id=host._session_id,
            )
        )
        return

    permit = await host.acquire_rebuild_permit(token)
    if isinstance(permit, RebuildPermitDenied):
        if await _publish_profile_satisfied_or_denied(host, event.profile_name, permit, token.session_id):
            return
        return
    try:
        if host._executor is not None and host._agent_profile and new_profile.name == host._agent_profile.name:
            await host._bus.publish(_profile_switched_snapshot(host, permit.token.session_id))
            return
        if host._executor is None:
            await host.start_with_rebuild_permit(permit, new_profile, operation="switch")
            await host._bus.publish(_profile_switched_snapshot(host, permit.token.session_id))
            return
        await host.soft_restart_with_rebuild_permit(permit, new_profile, operation="switch")
    finally:
        host.release_rebuild_permit(permit)


async def _publish_profile_satisfied_or_denied(
    host: EngineControlHost,
    profile_name: str,
    denied: RebuildPermitDenied,
    session_id: str | None,
) -> bool:
    if _denial_allows_already_satisfied_success(denied) and (
        host._agent_profile is not None and host._agent_profile.name == profile_name
    ):
        await host._bus.publish(_profile_switched_snapshot(host, session_id))
        return True
    await _publish_rebuild_denied(host, denied, session_id)
    return False


def _denial_allows_already_satisfied_success(denied: RebuildPermitDenied) -> bool:
    """Return whether a denial can be converted to an already-satisfied success."""
    return denied.reason == "superseded"


def _profile_switched_snapshot(host: EngineControlHost, session_id: str | None) -> ProfileSwitched:
    return dataclasses.replace(host.current_profile_snapshot(), session_id=session_id)


async def _publish_rebuild_denied(
    host: EngineControlHost,
    denied: RebuildPermitDenied,
    session_id: str | None,
) -> None:
    await host._bus.publish(Error(code=denied.code, message=denied.message, session_id=session_id))


async def _publish_model_switched_snapshot(
    host: EngineControlHost,
    profile_id: str,
    session_id: str | None,
) -> None:
    active = host._active_profile
    await host._bus.publish(
        ModelProfileSwitched(
            model_profile_id=active.id if active else profile_id,
            max_context_tokens=active.max_context_tokens if active else 0,
            runtime_details=host._runtime_details,
            session_id=session_id,
        )
    )


def _pin_session_model_profile(host: EngineControlHost, profile_id: str) -> None:
    host._settings_handle.install(
        host._loaded_settings.overlay(
            Source.SESSION,
            model_profile=profile_id,
            model_profile_override=profile_id,
            model_profile_override_sub_agents=False,
        ),
    )
    host._model_profile_pinned = True


async def _publish_workspace_updated_snapshot(
    host: EngineControlHost,
    session_id: str | None,
    *,
    primary_cwd: str | None = None,
) -> None:
    workspace = host._workspace
    resolved_primary_cwd = primary_cwd if primary_cwd is not None else (workspace.primary_cwd if workspace else "")
    await host._bus.publish(
        WorkspaceUpdated(
            primary_cwd=resolved_primary_cwd,
            working_dirs=[d.path for d in workspace.working_dirs] if workspace is not None else [],
            reference_files=list(workspace.reference_files) if workspace is not None else [],
            session_id=session_id,
        )
    )


async def _publish_model_satisfied_or_denied(
    host: EngineControlHost,
    profile_id: str,
    denied: RebuildPermitDenied,
    session_id: str | None,
) -> None:
    if _denial_allows_already_satisfied_success(denied) and (
        host._active_profile is not None and host._active_profile.id == profile_id
    ):
        _pin_session_model_profile(host, profile_id)
        await _publish_model_switched_snapshot(host, profile_id, session_id)
        return
    await _publish_rebuild_denied(host, denied, session_id)


async def _publish_workspace_satisfied_or_denied(
    host: EngineControlHost,
    primary_cwd: str,
    denied: RebuildPermitDenied,
    session_id: str | None,
) -> None:
    if _denial_allows_already_satisfied_success(denied) and _workspace_primary_matches(host._workspace, primary_cwd):
        await _publish_workspace_updated_snapshot(host, session_id, primary_cwd=primary_cwd)
        return
    await _publish_rebuild_denied(host, denied, session_id)


def _workspace_primary_matches(workspace: Workspace | None, primary_cwd: str) -> bool:
    """Return whether *primary_cwd* names the live workspace primary cwd."""
    if workspace is None:
        return False
    if workspace.primary_cwd == primary_cwd:
        return True
    return workspace.primary_cwd == Workspace.from_cwd(primary_cwd).primary_cwd


async def _acquire_rebuild_permit_or_publish_error(
    host: EngineControlHost,
    token: RebuildControlToken,
) -> RebuildPermit | None:
    permit = await host.acquire_rebuild_permit(token)
    if isinstance(permit, RebuildPermitDenied):
        await _publish_rebuild_denied(host, permit, token.session_id)
        return None
    return permit


async def _start_or_restart_with_permit(
    host: EngineControlHost,
    permit: RebuildPermit,
    profile: AgentProfile,
    *,
    operation: str,
    workspace: Workspace | None = None,
    staged_loaded: LoadedSettings | None = None,
) -> None:
    if host._executor is None:
        # The workspace rides as staged input, exactly like the settings: a
        # build that fails before committing must leave the live root — the
        # one the live settings and hooks were derived from — untouched.
        await host.start_with_rebuild_permit(
            permit, profile, operation=operation, staged_loaded=staged_loaded, workspace=workspace
        )
        return
    await host.soft_restart_with_rebuild_permit(
        permit, profile, workspace=workspace, operation=operation, staged_loaded=staged_loaded
    )


async def on_set_model_profile(host: EngineControlHost, event: SetModelProfile) -> None:
    """Switch the active model profile for this session only (no global .env write).

    Swaps the in-memory model selector and explicit override, then rebuilds;
    the build path resolves the model from ``settings`` against the registry,
    and credentials come from the profile itself, so sessions stay isolated.
    """
    if host._agent_profile is None:
        await host._bus.publish(
            Error(
                code="runtime_mutation_not_ready",
                message="No active agent — cannot switch model",
                display_message=_CONTROLS_MODEL_SWITCH_NOT_READY.bind(),
                session_id=host._session_id,
            )
        )
        return
    token = host.capture_rebuild_control_token()
    permit = await host.acquire_rebuild_permit(token)
    if isinstance(permit, RebuildPermitDenied):
        await _publish_model_satisfied_or_denied(host, event.profile_id, permit, token.session_id)
        return

    old_loaded = host._loaded_settings
    old_pinned = host._model_profile_pinned

    try:
        try:
            _pin_session_model_profile(host, event.profile_id)
            await _start_or_restart_with_permit(host, permit, host._agent_profile, operation="model_switch")
        except Exception:
            # Rebuild failed (AgentLoadFailed already published). Restore the pin/settings
            # so the still-running executor stays consistent with session state.
            host._settings_handle.install(old_loaded)
            host._model_profile_pinned = old_pinned
            raise
        await _publish_model_switched_snapshot(host, event.profile_id, token.session_id)
    finally:
        host.release_rebuild_permit(permit)


def _session_pin_overrides(host: EngineControlHost) -> dict[str, Any]:
    """Snapshot the per-session pins every re-load must carry.

    Per-session overrides injected into ``Settings`` (not the environment) are
    carried across only when pinned, so unpinned sessions still pick up a
    changed env value on the next load:

    * ``model_profile`` / ``model_profile_override`` — pinned by a per-session
      model switch.
    * ``ask_user_timeout_seconds`` — pinned when ACP owns the ask_user
      lifetime (set via dataclasses.replace at launch — see app/cli/acp.py).
    * ``frontend_default_max_transient_retries`` — launch-mode policy; travels
      as the :func:`_reload_eval_context` instead, and the separate
      env-derived override is deliberately re-read.
    """
    overrides: dict[str, Any] = {}
    if host._ask_user_timeout_pinned:
        overrides["ask_user_timeout_seconds"] = host._settings.ask_user_timeout_seconds
    if host._model_profile_pinned:
        overrides["model_profile"] = host._settings.model_profile
        overrides["model_profile_override"] = host._settings.model_profile_override
        overrides["model_profile_override_sub_agents"] = host._settings.model_profile_override_sub_agents
    return overrides


def _reload_eval_context(host: EngineControlHost) -> EvalContext:
    """The launch mode's retry policy, passed *into* the load.

    An input to the load rather than an override: the project layer's
    tighten/loosen verdicts are evaluated against it, and every re-load must
    use the same one the initial load did.
    """
    return EvalContext(frontend_default_max_transient_retries=host._settings.frontend_default_max_transient_retries)


def _session_project_root(host: EngineControlHost) -> Path:
    """The workspace root whose project trust domain this session lives under."""
    workspace = host._workspace
    return Path(workspace.primary_cwd) if workspace is not None else Path(safe_getcwd())


async def on_settings_reload(host: EngineControlHost, _event: SettingsReload) -> None:
    """Handle settings reload — reload Settings from disk and env, then rebuild."""
    token = host.capture_rebuild_control_token()
    # Taken even with nothing built yet. A host that has not built — startup
    # before the first build, a first build that failed, a host that only
    # subscribed — must still genuinely reload; echoing completion without
    # reloading would report success and then hand the first build the old
    # configuration. The permit is also what serializes the load against a
    # build, since ``start`` takes this same boundary: the values installed
    # below cannot land halfway through one.
    permit = await _acquire_rebuild_permit_or_publish_error(host, token)
    if permit is None:
        return

    old_loaded = host._loaded_settings
    try:
        try:
            # Re-read settings from env (matches TUI behavior, which persists
            # changes before triggering reload), for this session's workspace
            # root — the project trust domain is root-derived, so the reload
            # must re-derive it from where the session actually lives.
            #
            # Off-thread because the load reads config files and waits on their
            # lock, and this handler runs inline on the bus: a synchronous load
            # here would stall every other event for as long as the disk takes.
            # The pins are snapshotted before the hand-off, so the load works
            # from the state that asked for it.
            candidate = await asyncio.to_thread(
                load_settings,
                project_root=_session_project_root(host),
                eval_context=_reload_eval_context(host),
                **_session_pin_overrides(host),
            )
            # Routed after the load, not excluded from it: the reload still has
            # to report a RESTART value the user typed wrong, it just must not
            # claim the good ones took effect. The snapshot fields keep the
            # bootstrap values their readers hold; the rest of the RESTART tier
            # is held at the values already in force, so the rebuild below
            # cannot apply what only a restart may.
            loaded, deferred_keys = route_restart_settings(
                reattribute_command_line(candidate, old_loaded),
                old_loaded,
            )
            # A reload is the moment a user finds out their edit did not take.
            # Startup already reports these; dropping them here would make the
            # same bad value silent from the second read onwards.
            for warning in settings_warning_events(loaded):
                await host._bus.publish(dataclasses.replace(warning, session_id=host._session_id))
            if deferred_keys:
                display_message = _CONTROLS_SETTINGS_RESTART_REQUIRED.bind(keys=DisplaySequence(deferred_keys))
                await host._bus.publish(
                    Warning(
                        code="settings_restart_required",
                        message=format_message(display_message),
                        display_message=display_message,
                        session_id=host._session_id,
                    )
                )

            # Pull the latest active profile instance from the registry so edits
            # saved to user YAML (e.g. MCP enabled/disabled) take effect immediately.
            profile_for_restart = host._agent_profile
            if profile_for_restart is not None and host._agent_registry is not None:
                refreshed = host._agent_registry.get(profile_for_restart.name)
                if refreshed is not None:
                    profile_for_restart = refreshed
                else:
                    logger.warning(
                        "Active profile '%s' missing from registry during settings reload; reusing current profile.",
                        profile_for_restart.name,
                    )
        except Exception as exc:
            # A failing load (an unreadable settings file, say) aborts *before*
            # the rebuild publishes any completion event. The bus swallows
            # handler exceptions, so without an explicit failure here a caller
            # awaiting the reload would hang until its timeout. Nothing was
            # installed yet — the candidate stays staged until the rebuild
            # commits it — so surfacing the error is all there is to do.
            #
            # Individually invalid values no longer land here: they are rejected
            # by their coercer and reported as warnings, so one bad variable can
            # no longer take a whole reload down.
            await host._bus.publish(Error(code="settings_reload_failed", message=str(exc), session_id=host._session_id))
            raise

        # Only the rebuild is conditional: with nothing built there is no
        # runtime to replace, so the candidate is installed directly and the
        # first build will read it. With a runtime, the candidate travels as
        # the rebuild's staged input and is committed by the build itself,
        # together with the executor it configured: a rebuild that fails
        # before installing leaves the live settings untouched, and one that
        # fails after keeps the settings its executor was actually built from
        # — restoring the old ones there would desynchronize the two.
        if profile_for_restart is not None:
            await _start_or_restart_with_permit(
                host, permit, profile_for_restart, operation="settings_reload", staged_loaded=loaded
            )
            if old_loaded.settings.workspace_change_notice and not host._settings.workspace_change_notice:
                # Drop the baseline only once the off-state rebuild has committed —
                # a failed rebuild keeps the old enabled settings AND the baseline.
                host._workspace_change_tracker.invalidate()
        else:
            host._settings_handle.install(loaded)
        await _publish_settings_reloaded(host)
    finally:
        host.release_rebuild_permit(permit)


async def _publish_settings_reloaded(host: EngineControlHost) -> None:
    await host._bus.publish(SettingsReloaded(runtime_details=host._runtime_details, session_id=host._session_id))


async def on_workspace_change(host: EngineControlHost, event: WorkspaceChange) -> None:
    """Handle workspace/cwd change — rebuild agent with new workspace."""
    if host._agent_profile is None:
        await host._bus.publish(
            Error(
                code="runtime_mutation_not_ready",
                message="No active agent — cannot change workspace",
                display_message=_CONTROLS_WORKSPACE_SWITCH_NOT_READY.bind(),
                session_id=host._session_id,
            )
        )
        return
    token = host.capture_rebuild_control_token()
    permit = await host.acquire_rebuild_permit(token)
    if isinstance(permit, RebuildPermitDenied):
        await _publish_workspace_satisfied_or_denied(host, event.primary_cwd, permit, token.session_id)
        return

    new_workspace = Workspace.from_cwd(event.primary_cwd)
    starting_without_executor = host._executor is None
    try:
        # A workspace change is a settings reload in disguise: the project
        # trust domain is root-derived, so the new root's layers must be
        # audited and applied — and the old root's dropped — by the same
        # rebuild that installs the new workspace. Same routing as a reload
        # (RESTART values stay in force, the command line keeps its credit),
        # minus the restart-required warning: nothing was edited here, so any
        # deferred value was already reported by the reload that deferred it.
        try:
            old_loaded = host._loaded_settings
            candidate = await asyncio.to_thread(
                load_settings,
                project_root=Path(new_workspace.primary_cwd),
                eval_context=_reload_eval_context(host),
                **_session_pin_overrides(host),
            )
            staged_loaded, _ = route_restart_settings(reattribute_command_line(candidate, old_loaded), old_loaded)
        except Exception as exc:
            # Same shape as a reload's load failure: it aborts before the
            # rebuild publishes any completion event, and the bus swallows
            # handler exceptions — without an explicit terminal event here a
            # caller awaiting the change would hang until its timeout.
            await host._bus.publish(
                Error(code="workspace_change_failed", message=str(exc), session_id=host._session_id)
            )
            raise
        await _start_or_restart_with_permit(
            host,
            permit,
            host._agent_profile,
            operation="workspace_change",
            workspace=new_workspace,
            staged_loaded=staged_loaded,
        )
        # After the commit, not before: these verdicts describe the new
        # root's files, and a failed rebuild keeps the old root's settings.
        for warning in settings_warning_events(staged_loaded):
            await host._bus.publish(dataclasses.replace(warning, session_id=host._session_id))
        if starting_without_executor:
            await _publish_workspace_updated_snapshot(host, token.session_id)
    finally:
        host.release_rebuild_permit(permit)
