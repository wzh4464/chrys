# Copyright (c) 2026 Chrys. All rights reserved.

"""Shared helpers for TUI tests."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING

from chrys.app.tui.screens.main.diff_controller import LiveDiffTracker
from chrys.app.tui.screens.main.event_handlers import BackendEventCallbacks, BackendEventHandler
from chrys.app.tui.screens.main.state import MainScreenServices, MainScreenState
from chrys.app.tui.screens.main.view_adapter import MainScreenViewAdapter
from chrys.foundation.events.bus import EventBus

if TYPE_CHECKING:
    from chrys.app.tui.i18n import LocaleController


def install_trajectory_dashboard_query(screen: object) -> None:
    """Add the dashboard surface expected by the real main-screen view adapter."""
    original_query_one = getattr(screen, "query_one", None)
    if original_query_one is None:
        return
    dashboard = type("_TrajectoryDashboardFake", (), {"foreground": False})()

    def query_one(cls: type):
        if cls.__name__ == "TrajectoryDashboard":
            return dashboard
        return original_query_one(cls)

    screen.query_one = query_one


def make_backend_handler(screen: object, *, locale_controller: LocaleController | None = None) -> BackendEventHandler:
    """Construct a backend event handler around a lightweight screen fake."""
    install_trajectory_dashboard_query(screen)
    state = getattr(screen, "_state", None)
    if not isinstance(state, MainScreenState):
        state = MainScreenState()
    state.run.agent_running = bool(getattr(screen, "_agent_running", state.run.agent_running))
    state.run.agent_loading = bool(getattr(screen, "_agent_loading", state.run.agent_loading))
    state.run.has_messages = bool(getattr(screen, "_has_messages", state.run.has_messages))
    state.session.creating_new_session = bool(
        getattr(screen, "_creating_new_session", state.session.creating_new_session)
    )
    state.session.restoring_session = bool(getattr(screen, "_restoring_session", state.session.restoring_session))
    state.runtime.profile = str(getattr(screen, "_profile", state.runtime.profile))
    state.runtime.details = getattr(screen, "_runtime_details", state.runtime.details)
    state.runtime.main_usage_source_id = str(
        getattr(screen, "_main_usage_source_id", state.runtime.main_usage_source_id)
    )
    state.usage.last_usage_tokens = int(getattr(screen, "_last_usage_tokens", state.usage.last_usage_tokens))
    state.usage.last_total_session_tokens = int(
        getattr(screen, "_last_total_session_tokens", state.usage.last_total_session_tokens)
    )
    state.workspace_marker.original_cwd = getattr(screen, "_chdir_original_cwd", state.workspace_marker.original_cwd)
    state.workspace_marker.current_cwd = str(getattr(screen, "_chdir_current_cwd", state.workspace_marker.current_cwd))
    state.workspace.current_cwd = state.workspace_marker.current_cwd
    state.submit.active = bool(getattr(screen, "_pending_user_submit_active", state.submit.active))
    state.submit.text = str(getattr(screen, "_pending_user_submit_text", state.submit.text))
    state.submit.blocked = bool(getattr(screen, "_pending_user_submit_blocked", state.submit.blocked))
    state.render_gate.active = bool(getattr(screen, "_pending_user_message_render_active", state.render_gate.active))
    screen.context_usage_state = getattr(screen, "context_usage_state", None)

    services = MainScreenServices(
        bus=getattr(screen, "_bus", EventBus()),
        state_store=getattr(screen, "_state_store", None),
        agent_registry=getattr(screen, "_agent_registry", None),
        model_registry=getattr(screen, "_model_registry", None),
        active_model_profile_id=str(getattr(screen, "_active_model_profile_id", "")),
    )
    live_call_paths = getattr(screen, "_live_call_paths", None)
    live_file_mutations = getattr(screen, "_live_file_mutations", None)
    live_diff = LiveDiffTracker(
        call_paths=live_call_paths if isinstance(live_call_paths, MutableMapping) else None,
        file_mutations=live_file_mutations if isinstance(live_file_mutations, MutableMapping) else None,
    )

    def set_agent_running(value: bool) -> None:
        state.run.agent_running = value
        setter = getattr(screen, "_set_agent_running", None)
        if callable(setter):
            setter(value)
        else:
            screen._agent_running = value

    def set_agent_loading(value: bool) -> None:
        state.run.agent_loading = value
        setter = getattr(screen, "_set_agent_loading", None)
        if callable(setter):
            setter(value)
        else:
            screen._agent_loading = value

    def set_has_messages(value: bool) -> None:
        state.run.has_messages = value
        setter = getattr(screen, "_set_has_messages", None)
        if callable(setter):
            setter(value)
        else:
            screen._has_messages = value

    def set_profile_display(value: str) -> None:
        state.runtime.profile = value
        screen._profile = value

    def set_runtime_details(value: object) -> None:
        state.runtime.details = value
        screen._runtime_details = value

    def set_active_model_profile_id(value: str) -> None:
        services.active_model_profile_id = value
        screen._active_model_profile_id = value

    def set_main_usage_source_id(value: str) -> None:
        state.runtime.main_usage_source_id = value
        screen._main_usage_source_id = value

    def set_last_usage_tokens(value: int) -> None:
        state.usage.last_usage_tokens = value
        screen._last_usage_tokens = value

    def set_last_total_session_tokens(value: int) -> None:
        state.usage.last_total_session_tokens = value
        screen._last_total_session_tokens = value

    def set_creating_new_session(value: bool) -> None:
        state.session.creating_new_session = value
        screen._creating_new_session = value

    def set_restoring_session(value: bool) -> None:
        state.session.restoring_session = value
        screen._restoring_session = value

    def set_workspace_cwd(value: str) -> None:
        state.workspace.current_cwd = value
        state.workspace_marker.current_cwd = value
        screen._chdir_current_cwd = value

    def set_workspace_original_cwd(value: str | None) -> None:
        state.workspace_marker.original_cwd = value
        screen._chdir_original_cwd = value

    def update_subtitle() -> None:
        updater = getattr(screen, "_update_subtitle", None)
        if callable(updater):
            updater()

    def update_toc() -> None:
        updater = getattr(screen, "_update_toc", None)
        if callable(updater):
            updater()

    def on_session_fork_error(event: object, message: str, severity: str) -> None:
        sessions = getattr(screen, "_sessions", None)
        if sessions is not None:
            sessions.on_session_fork_error(event, message=message, severity=severity)

    def on_session_clear_error(event: object, message: str) -> None:
        sessions = getattr(screen, "_sessions", None)
        if sessions is not None:
            sessions.on_session_clear_error(event, message=message)

    def block_pending_user_submit() -> None:
        state.submit.block()
        screen._pending_user_submit_blocked = True

    def debug(key: str, message: str = "") -> None:
        debugger = getattr(screen, "_debug", None)
        if callable(debugger):
            debugger(key, message)

    def handle_approval_response(
        request_id: str,
        approved: bool,
        reason: str,
        modified_args: dict[str, object] | None,
    ) -> object | None:
        handler = getattr(screen, "_handle_approval_response", None)
        if callable(handler):
            return handler(request_id, approved, reason, modified_args)
        return None

    def handle_ask_user_response(request_id: str, text: str) -> None:
        handler = getattr(screen, "_handle_ask_user_response", None)
        if callable(handler):
            handler(request_id, text)

    def post_gc_message(message: object) -> None:
        messages = getattr(screen, "_gc_messages", None)
        if isinstance(messages, list):
            messages.append(message)

    def refresh_notification_settings() -> None:
        refresher = getattr(screen, "_refresh_notification_settings", None)
        if callable(refresher):
            refresher()

    def refresh_trajectory_verify_commands() -> None:
        refresher = getattr(screen, "_refresh_trajectory_verify_commands", None)
        if callable(refresher):
            refresher()

    return BackendEventHandler(
        state=state,
        services=services,
        locale_controller=locale_controller,
        view=MainScreenViewAdapter(
            screen,  # type: ignore[arg-type]
            state_store=services.state_store,
            locale_controller=locale_controller,
        ),
        callbacks=BackendEventCallbacks(
            set_agent_running=set_agent_running,
            set_agent_loading=set_agent_loading,
            set_has_messages=set_has_messages,
            set_profile_display=set_profile_display,
            set_runtime_details=set_runtime_details,
            set_active_model_profile_id=set_active_model_profile_id,
            set_main_usage_source_id=set_main_usage_source_id,
            set_last_usage_tokens=set_last_usage_tokens,
            set_last_total_session_tokens=set_last_total_session_tokens,
            set_creating_new_session=set_creating_new_session,
            set_restoring_session=set_restoring_session,
            set_workspace_cwd=set_workspace_cwd,
            set_workspace_original_cwd=set_workspace_original_cwd,
            refresh_git_branch=lambda: None,
            update_subtitle=update_subtitle,
            update_toc=update_toc,
            on_session_fork_error=on_session_fork_error,
            on_session_clear_error=on_session_clear_error,
            block_pending_user_submit=block_pending_user_submit,
            handle_approval_response=handle_approval_response,
            handle_ask_user_response=handle_ask_user_response,
            question_inline_preferred=lambda: False,
            post_gc_message=post_gc_message,
            debug=debug,
            refresh_model_indicator=lambda: None,
            refresh_notification_settings=refresh_notification_settings,
            refresh_trajectory_verify_commands=refresh_trajectory_verify_commands,
            settings_reloaded=lambda: None,
        ),
        live_diff=live_diff,
    )
