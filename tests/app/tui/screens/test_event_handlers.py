# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for BackendEventHandler live-diff accumulation helpers."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from rich.text import Text

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.diff import RollbackProgressModal
from chrys.app.tui.screens.diff.rollback_modal import RollbackLoadCancelled
from chrys.app.tui.screens.main.diff_controller import DiffController, LiveDiffOwner, LiveDiffTracker
from chrys.app.tui.screens.main.event_handlers import (
    _AGENT_FAILED_TO_LOAD,
    _UNKNOWN_ERROR,
    BackendEventHandler,
    _WarningDedupeKey,
)
from chrys.app.tui.screens.main.input_flow import InputFlowController
from chrys.app.tui.screens.main.live_diff import LiveFileMutation, mutation_op_for_live_op
from chrys.app.tui.screens.main.ports import StatusTrail
from chrys.app.tui.screens.main.rollback_controller import RollbackController
from chrys.app.tui.screens.main.screen import MainScreen, _HostRefreshLeaseScreen
from chrys.app.tui.screens.main.session_handlers import (
    _PROFILE_SWITCH_INDICATOR,
    _WORKING_DIRECTORY_INDICATOR,
    SessionCallbacks,
    SessionHandler,
)
from chrys.app.tui.screens.main.shell_mode import ShellModeController
from chrys.app.tui.screens.main.state import MainScreenServices, MainScreenState
from chrys.app.tui.screens.main.tool_action_bridge import ToolActionBridge
from chrys.app.tui.screens.main.view_adapter import MainScreenViewAdapter
from chrys.app.tui.support.gc_freeze import (
    GcAbsorbReason,
    GcAbsorbRequested,
    GcReclaimReason,
    GcReclaimRequested,
)
from chrys.app.tui.util.diff_entries import DiffLoadResult
from chrys.app.tui.widgets.chat.file_snapshot import FileSnapshotRef, file_snapshot_inline_char_limit
from chrys.app.tui.widgets.chrome.file_scanner import ProjectPathSuggestion
from chrys.app.tui.widgets.sidebar.context import ContextUsageState
from chrys.app.tui.widgets.sidebar.tasks import TodoListState
from chrys.foundation.branding import APP_DISPLAY_NAME
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    AgentMessage,
    AgentRuntimeDetails,
    AgentRuntimeUpdated,
    ApprovalModeUpdated,
    CompactionFinished,
    CompactionStarted,
    ContextCompressed,
    ContextPressure,
    Error,
    ImageAttachmentCompressionFinished,
    ImageAttachmentCompressionStarted,
    ProfileSwitched,
    RollbackResult,
    RuntimeHookDetails,
    RuntimeHookSourceDetails,
    RuntimeModelDetails,
    RuntimeSkillDetails,
    SessionClear,
    SessionDeleted,
    SessionFork,
    SessionForked,
    SessionReady,
    SessionRestored,
    SettingsReloaded,
    TodoListUpdated,
    UsageUpdate,
    UserInjectResult,
    UserInterrupt,
    UserMessage,
    UserRetry,
    UserRollback,
    Warning,
    WorkspaceUpdated,
)
from chrys.foundation.i18n import DisplayPath, Localizer, MessageRef
from chrys.foundation.i18n.formatting import format_message
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.todos import TodoItem
from chrys.foundation.util.session_ids import session_short_id
from chrys.kernel import Message
from chrys.service.approval.policy import ApprovalMode
from chrys.service.context.providers.history import CompressedBlock
from chrys.service.mutations.store import SnapshotStore
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.types import FileMutationTextSnapshot, MutationOp, MutationSource
from chrys.service.state.store import JsonFileStateStore
from tests.support.tui_helpers import install_trajectory_dashboard_query, make_backend_handler


def _status_text(value: MessageRef | str) -> str:
    return value if isinstance(value, str) else format_message(value)


def test_event_error_fallbacks_keep_english_and_localize_chinese() -> None:
    assert format_message(_UNKNOWN_ERROR.bind()) == "Unknown error"
    assert format_message(_AGENT_FAILED_TO_LOAD.bind()) == "Agent failed to load."
    chinese = Localizer("zh-Hans")
    assert chinese.render(_UNKNOWN_ERROR.bind()) == "未知错误"
    assert chinese.render(_AGENT_FAILED_TO_LOAD.bind()) == "智能体加载失败。"


def test_session_indicators_keep_english_and_localize_chinese() -> None:
    profile = _PROFILE_SWITCH_INDICATOR.bind(from_label="Code", to_label="QA")
    workspace = _WORKING_DIRECTORY_INDICATOR.bind(path=DisplayPath("/repo"))
    assert format_message(profile) == "Agent profile switched: Code → QA"
    assert format_message(workspace) == "Working directory → /repo"
    chinese = Localizer("zh-Hans")
    assert chinese.render(profile) == "智能体配置已切换：Code → QA"  # noqa: RUF001
    assert chinese.render(workspace) == "工作目录 → /repo"


def _status_trail(value: MessageRef | str | tuple[MessageRef | str, ...]) -> str:
    if isinstance(value, tuple):
        return " · ".join(_status_text(part) for part in value)
    return _status_text(value)


class _RollbackProjectionFenceMixin:
    """Minimal backend projection-fence surface for rollback controller tests."""

    rollback_projection_fenced = False
    build_generation = 1
    workspace_primary_cwd = "/repo/workspace"

    async def begin_rollback_projection(
        self,
        *,
        session_id: str | None,
        session_generation: int,
    ) -> str:
        assert session_id == "session-1"
        assert session_generation in {1, 7}
        self.rollback_projection_fenced = True
        return "test-rollback-projection"

    def finish_rollback_projection(self, owner: str) -> None:
        assert owner == "test-rollback-projection"
        assert self.rollback_projection_fenced
        self.rollback_projection_fenced = False


class _FakeInputFlowView:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.user_contents: list[object] = []
        self.gc_messages: list[object] = []
        self.session_id = "panel-session"

    def set_terminal_title_for_user_message(self, text: str) -> None:
        self.calls.append(("title", text))

    def clear_terminal_title_result(self) -> None:
        self.calls.append("clear_title_result")

    def current_chat_session_id(self) -> str:
        return self.session_id

    def add_input_history(self, text: str, *, session_id: str | None) -> None:
        self.calls.append(f"history:{text}:{session_id}")

    def set_retry_mode(self, enabled: bool, *, label: str = "Retry") -> None:
        self.calls.append(("retry_mode", enabled, label))

    def start_run_status(self, label: MessageRef | str) -> None:
        self.calls.append(f"status:{_status_text(label)}")

    async def render_user_message(self, text: str, *, created_at: object, contents: object) -> None:
        self.calls.append(f"user:{text}")
        self.user_contents.append(contents)

    async def render_user_retry_note(self, text: str, *, created_at: object) -> None:
        self.calls.append(f"retry-note:{text}")

    def hide_trailing_status_action(self) -> None:
        self.calls.append("hide_status_action")

    def clear_inline_questions(self) -> None:
        self.calls.append("clear_questions")

    def clear_ask_user_inline_prompts(self) -> None:
        self.calls.append("clear_inline")

    def lock_input_with_text(self) -> None:
        self.calls.append("lock_input")

    def unlock_input_keep_if_locked(self) -> None:
        self.calls.append("unlock_keep")

    def flash_interrupted(self) -> None:
        self.calls.append("flash_interrupted")

    async def render_interrupted(self) -> None:
        self.calls.append("interrupted")

    def update_toc(self) -> None:
        self.calls.append("toc")


def _make_input_flow_controller(
    bus: EventBus,
    *,
    state: MainScreenState | None = None,
    view: _FakeInputFlowView | None = None,
    handle_agent_message: object | None = None,
    handle_error: object | None = None,
    running: list[bool] | None = None,
    has_messages: list[bool] | None = None,
) -> tuple[InputFlowController, MainScreenState, _FakeInputFlowView]:
    state = state or MainScreenState()
    view = view or _FakeInputFlowView()
    running = running if running is not None else []
    has_messages = has_messages if has_messages is not None else []

    async def _default_handle_agent_message(event: AgentMessage) -> None:
        view.calls.append(f"agent:{event.text}")

    async def _default_handle_error(event: Error) -> None:
        view.calls.append(f"error:{event.message}")

    def _unexpected_start_worker(awaitable: object) -> None:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise AssertionError(f"unexpected worker scheduling: {awaitable!r}")

    def _set_agent_running(value: bool) -> None:
        state.run.agent_running = value
        running.append(value)

    def _set_has_messages(value: bool) -> None:
        state.run.has_messages = value
        has_messages.append(value)

    controller = InputFlowController(
        state=state,
        services=MainScreenServices(bus=bus),
        view=view,
        start_worker=_unexpected_start_worker,
        handle_agent_message=handle_agent_message or _default_handle_agent_message,
        handle_error=handle_error or _default_handle_error,
        set_agent_running=_set_agent_running,
        set_has_messages=_set_has_messages,
        clear_workspace_marker=lambda: setattr(state.workspace_marker, "original_cwd", None),
        clear_pending_questions=view.clear_inline_questions,
        commit_profile_switch_marker=lambda: setattr(state.profile_marker, "from_profile", None),
        post_gc_message=view.gc_messages.append,
        debug=lambda *_args: None,
    )
    return controller, state, view


def _make_session_handler(
    screen: object,
    *,
    locale_controller: LocaleController | None = None,
) -> SessionHandler:
    install_trajectory_dashboard_query(screen)
    state = getattr(screen, "_state", None)
    if not isinstance(state, MainScreenState):
        state = MainScreenState()
    state.run.agent_running = bool(getattr(screen, "_agent_running", state.run.agent_running))
    state.run.agent_loading = bool(getattr(screen, "_agent_loading", state.run.agent_loading))
    state.run.has_messages = bool(getattr(screen, "_has_messages", state.run.has_messages))
    state.session.restoring_session = bool(getattr(screen, "_restoring_session", state.session.restoring_session))
    state.session.creating_new_session = bool(
        getattr(screen, "_creating_new_session", state.session.creating_new_session)
    )
    state.submit.active = bool(getattr(screen, "_pending_user_submit_active", state.submit.active))
    state.runtime.profile = str(getattr(screen, "_profile", state.runtime.profile))
    state.runtime.details = getattr(screen, "_runtime_details", state.runtime.details)
    state.usage.last_usage_tokens = int(getattr(screen, "_last_usage_tokens", state.usage.last_usage_tokens))
    state.usage.last_total_session_tokens = int(
        getattr(screen, "_last_total_session_tokens", state.usage.last_total_session_tokens)
    )
    state.workspace_marker.original_cwd = getattr(screen, "_chdir_original_cwd", state.workspace_marker.original_cwd)
    current_cwd = getattr(screen, "_chdir_current_cwd", state.workspace_marker.current_cwd)
    workspace_cwd = getattr(screen, "_workspace_cwd", None)
    if current_cwd == state.workspace_marker.current_cwd and callable(workspace_cwd):
        current_cwd = workspace_cwd()
    state.workspace_marker.current_cwd = str(current_cwd)
    state.workspace.current_cwd = state.workspace_marker.current_cwd

    services = MainScreenServices(
        bus=getattr(screen, "_bus", EventBus()),
        state_store=getattr(screen, "_state_store", None),
        active_model_profile_id=str(getattr(screen, "_active_model_profile_id", "")),
        apply_saved_model_on_restore=bool(getattr(screen, "_apply_saved_model_on_restore", True)),
    )

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

    def set_creating_new_session(value: bool) -> None:
        state.session.creating_new_session = value
        screen._creating_new_session = value

    def set_restoring_session(value: bool) -> None:
        state.session.restoring_session = value
        screen._restoring_session = value

    def set_profile_display(value: str) -> None:
        state.runtime.profile = value
        screen._profile = value

    def set_active_model_profile_id(value: str) -> None:
        services.active_model_profile_id = value
        screen._active_model_profile_id = value

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

    def clear_suggestion_file_cache() -> None:
        suggestions = getattr(screen, "_suggestions", None)
        if suggestions is not None:
            suggestions.file_cache = None

    def start_session_restore(session_id: str) -> object | None:
        restorer = getattr(screen, "_do_session_restore", None)
        if callable(restorer):
            return restorer(session_id)
        return None

    def debug(key: str, message: str = "") -> None:
        debugger = getattr(screen, "_debug", None)
        if callable(debugger):
            debugger(key, message)

    def post_gc_message(message: object) -> None:
        messages = getattr(screen, "_gc_messages", None)
        if isinstance(messages, list):
            messages.append(message)

    class _AgentLoadPort:
        async def begin_session_restore_load(self, session_id: str) -> None:
            events = getattr(screen, "_events", None)
            handler = getattr(events, "begin_session_restore_load", None)
            if callable(handler):
                result = handler(session_id)
                if inspect.isawaitable(result):
                    await result

        def cancel_agent_load(self) -> None:
            events = getattr(screen, "_events", None)
            handler = getattr(events, "cancel_agent_load", None)
            if callable(handler):
                handler()

        def finish_agent_load(self, message: str = "") -> None:
            events = getattr(screen, "_events", None)
            handler = getattr(events, "finish_agent_load", None)
            if callable(handler):
                handler(message)

    return SessionHandler(
        state=state,
        services=services,
        view=MainScreenViewAdapter(screen, state_store=services.state_store),  # type: ignore[arg-type]
        callbacks=SessionCallbacks(
            set_agent_loading=set_agent_loading,
            set_has_messages=set_has_messages,
            set_creating_new_session=set_creating_new_session,
            set_restoring_session=set_restoring_session,
            set_profile_display=set_profile_display,
            set_active_model_profile_id=set_active_model_profile_id,
            set_workspace_cwd=set_workspace_cwd,
            set_workspace_original_cwd=set_workspace_original_cwd,
            update_subtitle=update_subtitle,
            update_toc=update_toc,
            clear_suggestion_file_cache=clear_suggestion_file_cache,
            start_session_restore=start_session_restore,
            post_gc_message=post_gc_message,
            debug=debug,
            refresh_model_indicator=lambda: None,
        ),
        agent_load=_AgentLoadPort(),
        locale_controller=locale_controller,
    )


class _FakeShellModeView:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def enter_shell_mode(self) -> None:
        self.calls.append("enter")

    def exit_shell_mode(self) -> None:
        self.calls.append("exit")

    async def send_shell_interrupt(self) -> None:
        self.calls.append("interrupt")

    def set_alternate_screen_active(self, active: bool) -> None:
        self.calls.append(("alternate", active))


class _FakeFocusView:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def focus_input(self) -> None:
        self.calls.append("focus_input")

    def insert_paste_payload(self, text: str) -> bool:
        self.calls.append(("paste", text))
        return True


def _make_shell_mode_controller(
    *,
    state: MainScreenState | None = None,
    shell_view: _FakeShellModeView | None = None,
    focus_view: _FakeFocusView | None = None,
    shell_mode_states: list[bool] | None = None,
    dismiss_suggestions: Callable[[], None] | None = None,
) -> tuple[ShellModeController, MainScreenState, _FakeShellModeView, _FakeFocusView, list[bool]]:
    state = state or MainScreenState()
    shell_view = shell_view or _FakeShellModeView()
    focus_view = focus_view or _FakeFocusView()
    shell_mode_states = shell_mode_states if shell_mode_states is not None else []

    def _set_shell_mode(active: bool) -> None:
        state.shell.active = active

    def _set_fullscreen(active: bool) -> None:
        state.shell.fullscreen_terminal = active

    controller = ShellModeController(
        state=state,
        shell_view=shell_view,
        focus_view=focus_view,
        set_shell_mode=_set_shell_mode,
        set_shell_mode_state=shell_mode_states.append,
        set_fullscreen_terminal=_set_fullscreen,
        dismiss_suggestions=dismiss_suggestions or (lambda: None),
        debug=lambda *_args: None,
    )
    return controller, state, shell_view, focus_view, shell_mode_states


def _make_handler() -> tuple[BackendEventHandler, dict[str, LiveFileMutation]]:
    """Create a BackendEventHandler with a minimal mock screen.

    Returns ``(handler, live_file_mutations_dict)`` — the dict is the
    shared mutable mapping on the mock screen.
    """
    live: dict[str, LiveFileMutation] = {}
    screen = SimpleNamespace(_live_file_mutations=live)
    handler = make_backend_handler(screen)
    return handler, live


def test_approval_mode_updated_updates_state_header_and_notification() -> None:
    notifications: list[tuple[str, str, str, float | None]] = []
    debug_log: list[tuple[str, str]] = []
    screen = SimpleNamespace(
        _debug=lambda key, message="": debug_log.append((key, message)),
        header_approval_mode=ApprovalMode.MANUAL,
        notify=lambda message, *, title, severity="information", timeout=3, markup=False: notifications.append(
            (message, title, severity, timeout)
        ),
    )
    handler = make_backend_handler(screen)

    asyncio.run(handler.on_approval_mode_updated(ApprovalModeUpdated(mode=ApprovalMode.AUTO.value)))

    assert handler.approval_mode is ApprovalMode.AUTO
    assert screen.header_approval_mode is ApprovalMode.AUTO
    assert notifications == [("Approval mode: AUTO", "Approval", "information", 2)]
    assert debug_log[-1] == ("ApprovalMode", "AUTO")


def test_settings_reloaded_reprojects_notification_settings() -> None:
    # The delivery service holds a projection taken at app init; a reload that
    # only installed new settings must trigger a re-projection or an external
    # document edit would never reach it.
    calls: list[str] = []
    screen = SimpleNamespace(_refresh_notification_settings=lambda: calls.append("refreshed"))
    handler = make_backend_handler(screen)

    asyncio.run(handler.on_settings_reloaded(SettingsReloaded()))

    assert calls == ["refreshed"]


def test_settings_reloaded_reprojects_the_verify_command_word_list() -> None:
    # Same LIVE-tier routing as the notification projection: the dashboard's
    # classification word list is captured at construction, so a reload that
    # only installed a new value must push it through or actions keep being
    # classified against the previous document until the screen is rebuilt.
    calls: list[str] = []
    screen = SimpleNamespace(_refresh_trajectory_verify_commands=lambda: calls.append("reprojected"))
    handler = make_backend_handler(screen)

    asyncio.run(handler.on_settings_reloaded(SettingsReloaded()))

    assert calls == ["reprojected"]


def _stale_file_cache(*paths: str) -> list[ProjectPathSuggestion]:
    return [ProjectPathSuggestion(path=path, kind="file") for path in paths]


def _live(
    before_text: str,
    after_text: str,
    operation: str,
    *,
    bytes_changed: bool = True,
    before_hash: str | None = None,
    after_hash: str | None = None,
    source: str = "",
) -> LiveFileMutation:
    return LiveFileMutation(
        before_text=before_text,
        after_text=after_text,
        operation=operation,
        bytes_changed=bytes_changed,
        before_hash=before_hash,
        after_hash=after_hash,
        source=source,
    )


def _show_diff_for_test_screen(screen: SimpleNamespace) -> None:
    session_id = screen.query_one(object).session_id
    live_diff = LiveDiffTracker(
        owner=LiveDiffOwner(session_id=session_id, session_generation=0, run_generation=0),
        file_mutations=screen._live_file_mutations,
    )
    controller = DiffController(
        services=MainScreenServices(
            bus=EventBus(),
            state_store=screen._state_store,
        ),
        live_diff=live_diff,
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=screen._workspace_cwd,
        is_agent_running=lambda: screen._agent_running,
        run_generation=lambda: 0,
        session_generation=lambda: 0,
    )
    controller.show_diff()


def _make_rollback_controller_for_test(screen: SimpleNamespace) -> RollbackController:
    def _reset_welcome_workspace_marker(cwd: str) -> None:
        screen._chdir_original_cwd = None
        screen._chdir_current_cwd = cwd

    def _set_has_messages(value: bool) -> None:
        screen._set_has_messages(value)

    def _post_gc_message(message: object) -> None:
        messages = getattr(screen, "_gc_messages", None)
        if isinstance(messages, list):
            messages.append(message)

    return RollbackController(
        services=MainScreenServices(
            bus=EventBus(),
            engine_provider=lambda: screen._engine,
        ),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=screen._workspace_cwd,
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: 1,
        turn_lifecycle_task=lambda: None,
        profile_name=lambda: screen._profile,
        reset_welcome_workspace_marker=_reset_welcome_workspace_marker,
        set_has_messages=_set_has_messages,
        post_gc_message=_post_gc_message,
        debug=screen._debug,
    )


def _shell_snapshot(
    before_text: str,
    after_text: str,
    operation: str,
    *,
    bytes_changed: bool | None = None,
    before_hash: str | None = None,
    after_hash: str | None = None,
    source: str = "",
) -> FileMutationTextSnapshot:
    return FileMutationTextSnapshot(
        before_text=before_text,
        after_text=after_text,
        operation=operation,
        bytes_changed=before_text != after_text if bytes_changed is None else bytes_changed,
        source=source,
        before_hash=before_hash,
        after_hash=after_hash,
    )


# ──────────── _accumulate_live_mutation ────────────────────────────────


def test_accumulate_new_file_create() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "", "content")
    assert live["/tmp/f.py"] == _live("", "content", "create")


def test_accumulate_new_file_modify() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "old", "new")
    assert live["/tmp/f.py"] == _live("old", "new", "modify")


def test_accumulate_preserves_first_before() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "original", "v1")
    handler._accumulate_live_mutation("/tmp/f.py", "v1", "v2")
    assert live["/tmp/f.py"] == _live("original", "v2", "modify")


def test_accumulate_removes_live_mutation_that_reverts_to_original() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "original", "v1")
    handler._accumulate_live_mutation("/tmp/f.py", "v1", "original")
    assert "/tmp/f.py" not in live


def test_accumulate_removes_live_create_then_delete() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "", "generated", "create")
    handler._accumulate_live_mutation("/tmp/f.py", "generated", "", "delete")
    assert "/tmp/f.py" not in live


def test_accumulate_explicit_modify_empty_file_noop_is_ignored() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "", "", "modify")
    assert "/tmp/f.py" not in live


def test_accumulate_explicit_modify_empty_file_roundtrip_is_removed() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "", "content", "modify")
    handler._accumulate_live_mutation("/tmp/f.py", "content", "", "modify")
    assert "/tmp/f.py" not in live


def test_accumulate_with_explicit_op() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "old", "new", "delete")
    assert live["/tmp/f.py"] == _live("old", "new", "delete")


def test_accumulate_explicit_op_ignored_on_update() -> None:
    """Once a path exists, op_str is always preserved from the first entry."""
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/f.py", "", "v1", "create")
    handler._accumulate_live_mutation("/tmp/f.py", "v1", "v2", "modify")
    assert live["/tmp/f.py"] == _live("", "v2", "create")


def test_accumulate_keeps_live_metadata_only_modify() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation(
        "/tmp/f.py",
        "same text\n",
        "same text\n",
        "modify",
        bytes_changed=True,
        before_hash="hash-with-bom",
        after_hash="hash-without-bom",
    )
    assert live["/tmp/f.py"] == _live(
        "same text\n",
        "same text\n",
        "modify",
        bytes_changed=True,
        before_hash="hash-with-bom",
        after_hash="hash-without-bom",
    )


def test_accumulate_keeps_live_move_when_destination_bytes_are_unchanged() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation(
        "/tmp/dst.py",
        "same text\n",
        "same text\n",
        "move",
        bytes_changed=False,
        before_hash="same-hash",
        after_hash="same-hash",
    )
    assert live["/tmp/dst.py"] == _live(
        "same text\n",
        "same text\n",
        "move",
        bytes_changed=False,
        before_hash="same-hash",
        after_hash="same-hash",
    )


def test_accumulate_hashes_remove_live_metadata_roundtrip() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation(
        "/tmp/f.py",
        "same text\n",
        "same text\n",
        "modify",
        bytes_changed=True,
        before_hash="h1",
        after_hash="h2",
    )
    handler._accumulate_live_mutation(
        "/tmp/f.py",
        "same text\n",
        "same text\n",
        "modify",
        bytes_changed=True,
        before_hash="h2",
        after_hash="h1",
    )
    assert "/tmp/f.py" not in live


def test_action_show_diff_opens_when_only_all_entries_survive(monkeypatch) -> None:
    import chrys.app.tui.screens.diff as diff_pkg
    from chrys.app.tui.screens.diff import DiffFileEntry, DiffLoadResult, DiffScreen

    seen_cwds: list[str] = []
    entry = DiffFileEntry(
        path="/repo/Power.yaml",
        rel_path="Power.yaml",
        operation=MutationOp.MODIFY,
        old_path=None,
        before_text="name: Power\n",
        after_text="name: Power\r\n",
        is_binary=False,
        encoding="utf-8",
    )

    def fake_load_diff_entries_by_turn(_state_store, _session_id, cwd):
        seen_cwds.append(cwd)
        return DiffLoadResult(all_entries=[entry], per_turn_entries={}, total_turns=1)

    monkeypatch.setattr(diff_pkg, "load_diff_entries_by_turn", fake_load_diff_entries_by_turn)
    pushed: list[object] = []
    notifications: list[tuple[tuple, dict]] = []
    screen = SimpleNamespace(
        _state_store=object(),
        _agent_running=False,
        _live_file_mutations={},
        app=SimpleNamespace(push_screen=pushed.append),
        query_one=lambda _cls: SimpleNamespace(session_id="session-1"),
        notify=lambda *args, **kwargs: notifications.append((args, kwargs)),
        _workspace_cwd=lambda: "/repo",
    )

    _show_diff_for_test_screen(screen)

    assert notifications == []
    assert len(pushed) == 1
    assert isinstance(pushed[0], DiffScreen)
    loader = pushed[0]._load_data
    assert loader is not None
    result = asyncio.run(loader())

    assert result.per_turn_entries == {}
    assert result.all_entries == [entry]
    assert seen_cwds == ["/repo"]


def test_action_show_diff_keeps_live_metadata_only_change() -> None:
    from chrys.app.tui.screens.diff import DiffScreen

    pushed: list[object] = []
    notifications: list[tuple[tuple, dict]] = []
    screen = SimpleNamespace(
        _state_store=None,
        _agent_running=True,
        _live_file_mutations={
            "/repo/Power.yaml": _live(
                "name: Power\n",
                "name: Power\n",
                "modify",
                bytes_changed=True,
                before_hash="hash-with-bom",
                after_hash="hash-without-bom",
            )
        },
        app=SimpleNamespace(push_screen=pushed.append),
        query_one=lambda _cls: SimpleNamespace(session_id=""),
        notify=lambda *args, **kwargs: notifications.append((args, kwargs)),
        _workspace_cwd=lambda: "/repo",
    )

    _show_diff_for_test_screen(screen)

    assert notifications == []
    assert len(pushed) == 1
    assert isinstance(pushed[0], DiffScreen)
    loader = pushed[0]._load_data
    assert loader is not None
    result = asyncio.run(loader())
    entry = result.per_turn_entries[1][0]
    assert entry.before_text == "name: Power\n"
    assert entry.after_text == "name: Power\n"
    assert entry.bytes_changed is True
    assert entry.before_hash == "hash-with-bom"
    assert entry.after_hash == "hash-without-bom"


def test_action_show_diff_drops_all_entry_when_live_change_restores_original_bytes(monkeypatch) -> None:
    import chrys.app.tui.screens.diff as diff_pkg
    from chrys.app.tui.screens.diff import DiffFileEntry, DiffLoadResult, DiffScreen

    persisted = DiffFileEntry(
        path="/repo/Power.yaml",
        rel_path="Power.yaml",
        operation=MutationOp.MODIFY,
        old_path=None,
        before_text="A\n",
        after_text="B\n",
        is_binary=False,
        encoding="utf-8",
        bytes_changed=True,
        before_hash="hash-A",
        after_hash="hash-B",
    )

    def fake_load_diff_entries_by_turn(_state_store, _session_id, _cwd):
        return DiffLoadResult(all_entries=[persisted], per_turn_entries={1: [persisted]}, total_turns=1)

    monkeypatch.setattr(diff_pkg, "load_diff_entries_by_turn", fake_load_diff_entries_by_turn)
    pushed: list[object] = []
    screen = SimpleNamespace(
        _state_store=object(),
        _agent_running=True,
        _live_file_mutations={
            "/repo/Power.yaml": _live(
                "B\n",
                "A\n",
                "modify",
                bytes_changed=True,
                before_hash="hash-B",
                after_hash="hash-A",
            ),
        },
        app=SimpleNamespace(push_screen=pushed.append),
        query_one=lambda _cls: SimpleNamespace(session_id="session-1"),
        notify=lambda *_args, **_kwargs: None,
        _workspace_cwd=lambda: "/repo",
    )

    _show_diff_for_test_screen(screen)

    assert len(pushed) == 1
    assert isinstance(pushed[0], DiffScreen)
    loader = pushed[0]._load_data
    assert loader is not None
    result = asyncio.run(loader())

    assert result.all_entries == []
    assert result.per_turn_entries[2][0].after_text == "A\n"


def test_open_rollback_modal_uses_workspace_cwd(monkeypatch) -> None:
    import chrys.app.tui.screens.diff as diff_pkg

    seen_cwds: list[str] = []
    state_loaders: list[object] = []
    state_reads: list[str] = []
    pushed: list[object] = []

    class _FakeRollbackModal:
        def __init__(self, *, cwd: str, **kwargs: object) -> None:
            seen_cwds.append(cwd)
            state_loaders.append(kwargs["load_state"])

    class _Coordinator:
        def __init__(self, marker: str) -> None:
            self.marker = marker

        def augment_rollback_plan(self, _tracker, _plan, **_kwargs: object) -> str:
            return self.marker

    class _FakeEngine(_RollbackProjectionFenceMixin):
        mutation_tracker = object()
        mutation_coordinator = _Coordinator("old")
        current_turn_number = 3
        conversation_revision = 7

        async def begin_rollback_projection(
            self,
            *,
            session_id: str | None,
            session_generation: int,
        ) -> str:
            # Model a queued workspace rebuild winning the shared gate before
            # the picker projection acquires it.
            self.build_generation = 2
            self.workspace_primary_cwd = "/repo/rebuilt-workspace"
            self.mutation_coordinator = _Coordinator("rebuilt")
            return await super().begin_rollback_projection(
                session_id=session_id,
                session_generation=session_generation,
            )

        def available_rollback_turns(self) -> list[int]:
            state_reads.append("targets")
            return [1, 2]

        def turn_prompt_previews(self) -> dict[int, str]:
            state_reads.append("prompts")
            return {}

    monkeypatch.setattr(diff_pkg, "RollbackModal", _FakeRollbackModal)
    screen = SimpleNamespace(
        _engine=_FakeEngine(),
        _profile="",
        _set_has_messages=lambda _value: None,
        _debug=lambda *_args: None,
        _workspace_cwd=lambda: "/repo/workspace",
        app=SimpleNamespace(push_screen=lambda modal, _callback=None: pushed.append(modal)),
        notify=lambda *_args, **_kwargs: None,
    )

    _make_rollback_controller_for_test(screen).show_rollback()

    assert seen_cwds == ["/repo/workspace"]
    assert len(pushed) == 1
    # The screen is pushed before snapshot payloads or history are read.
    assert state_reads == []
    assert len(state_loaders) == 1
    state = asyncio.run(state_loaders[0]())
    assert state is not None
    assert state_reads == ["targets", "prompts"]
    assert state.build_generation == 2
    assert state.workspace_cwd == "/repo/rebuilt-workspace"
    assert state.plan_augment is not None
    assert state.plan_augment(object(), state.workspace_cwd) == "rebuilt"


@pytest.mark.parametrize(
    ("arg", "expected_target", "expected_relative"),
    (("1", 0, 1), ("to 1", 1, None)),
)
def test_rollback_commands_preserve_relative_and_absolute_selectors(
    arg: str,
    expected_target: int,
    expected_relative: int | None,
) -> None:
    published: list[UserRollback] = []
    pushed: list[object] = []
    bus = EventBus()

    async def record_rollback(event: UserRollback) -> None:
        published.append(event)

    asyncio.run(bus.subscribe(UserRollback, record_rollback))

    class _FakeEngine(_RollbackProjectionFenceMixin):
        mutation_tracker = object()
        mutation_coordinator = None
        current_turn_number = 3
        conversation_revision = 7

        def available_rollback_turns(self) -> list[int]:
            return [0, 1, 2]

        def turn_prompt_previews(self) -> dict[int, str]:
            return {}

    screen = SimpleNamespace(
        app=SimpleNamespace(push_screen=lambda modal, _callback=None: pushed.append(modal)),
        notify=lambda *_args, **_kwargs: None,
    )
    controller = RollbackController(
        services=MainScreenServices(bus=bus, engine_provider=_FakeEngine),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: 1,
        turn_lifecycle_task=lambda: None,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    controller.show_rollback(arg)

    assert len(pushed) == 1
    progress = pushed[0]
    assert isinstance(progress, RollbackProgressModal)
    assert published == []
    asyncio.run(progress._operation())
    assert len(published) == 1
    assert published[0].target_turn == expected_target
    assert published[0].relative_turns == expected_relative
    assert published[0].revert_changes is True
    assert published[0].session_id == "session-1"


def test_rollback_direct_command_normalizes_empty_chat_session_id() -> None:
    published: list[UserRollback] = []
    pushed: list[object] = []
    bus = EventBus()

    async def record_rollback(event: UserRollback) -> None:
        published.append(event)

    asyncio.run(bus.subscribe(UserRollback, record_rollback))
    screen = SimpleNamespace(
        app=SimpleNamespace(push_screen=lambda modal, _callback=None: pushed.append(modal)),
        notify=lambda *_args, **_kwargs: None,
    )
    controller = RollbackController(
        services=MainScreenServices(bus=bus, engine_provider=object),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=str,
        session_generation=lambda: 1,
        turn_lifecycle_task=lambda: None,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    controller.show_rollback("1")
    progress = pushed[0]
    assert isinstance(progress, RollbackProgressModal)
    asyncio.run(progress._operation())

    assert len(published) == 1
    assert published[0].session_id is None


@pytest.mark.asyncio
async def test_rollback_direct_command_delegates_fencing_and_target_resolution_to_backend() -> None:
    published: list[UserRollback] = []
    pushed: list[object] = []
    target_reads: list[str] = []
    lifecycle_release = asyncio.Event()
    replacement_release = asyncio.Event()
    bus = EventBus()

    async def record_rollback(event: UserRollback) -> None:
        published.append(event)

    async def wait_for_release(release: asyncio.Event) -> None:
        await release.wait()

    await bus.subscribe(UserRollback, record_rollback)
    captured_task = asyncio.create_task(wait_for_release(lifecycle_release))
    replacement_task = asyncio.create_task(wait_for_release(replacement_release))
    lifecycle_ref = [captured_task]

    class _FakeEngine(_RollbackProjectionFenceMixin):
        mutation_tracker = object()
        mutation_coordinator = None
        current_turn_number = 3
        conversation_revision = 7

        def available_rollback_turns(self) -> list[int]:
            target_reads.append("targets")
            return [0, 1, 2]

        def turn_prompt_previews(self) -> dict[int, str]:
            return {}

    screen = SimpleNamespace(
        app=SimpleNamespace(push_screen=lambda modal, _callback=None: pushed.append(modal)),
        notify=lambda *_args, **_kwargs: None,
    )
    controller = RollbackController(
        services=MainScreenServices(bus=bus, engine_provider=_FakeEngine),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: 1,
        turn_lifecycle_task=lambda: lifecycle_ref[0],
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    try:
        controller.show_rollback("1")
        progress = pushed[0]
        assert isinstance(progress, RollbackProgressModal)

        # Direct commands do not scan snapshots or await lifecycle state in
        # the frontend. The backend event handler owns both operations under
        # its transition fence.
        lifecycle_ref[0] = replacement_task
        await progress._operation()
        assert target_reads == []
        assert len(published) == 1
        assert published[0].target_turn == 0
        assert published[0].relative_turns == 1
        assert not captured_task.done()
        assert not replacement_task.done()
    finally:
        lifecycle_release.set()
        await asyncio.gather(captured_task, return_exceptions=True)
        replacement_task.cancel()
        await asyncio.gather(replacement_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_picker", [False, True])
async def test_rollback_publication_propagates_handler_failure_to_progress_modal(
    monkeypatch: pytest.MonkeyPatch,
    use_picker: bool,
) -> None:
    import chrys.app.tui.screens.diff as diff_pkg

    pushed: list[object] = []
    callbacks: list[object] = []
    state_loaders: list[object] = []
    handled: list[UserRollback] = []
    bus = EventBus()

    async def fail_rollback(event: UserRollback) -> None:
        handled.append(event)
        raise RuntimeError("rollback handler failed")

    await bus.subscribe(UserRollback, fail_rollback)

    class _FakeRollbackModal:
        def __init__(self, **kwargs: object) -> None:
            state_loaders.append(kwargs["load_state"])

    class _FakeEngine(_RollbackProjectionFenceMixin):
        mutation_tracker = object()
        mutation_coordinator = None
        current_turn_number = 3
        conversation_revision = 7

        def available_rollback_turns(self) -> list[int]:
            return [0, 1, 2]

        def turn_prompt_previews(self) -> dict[int, str]:
            return {}

    def push_screen(modal: object, callback: object | None = None) -> None:
        pushed.append(modal)
        if callback is not None:
            callbacks.append(callback)

    monkeypatch.setattr(diff_pkg, "RollbackModal", _FakeRollbackModal)
    screen = SimpleNamespace(
        app=SimpleNamespace(push_screen=push_screen),
        notify=lambda *_args, **_kwargs: None,
    )
    controller = RollbackController(
        services=MainScreenServices(bus=bus, engine_provider=_FakeEngine),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: 1,
        turn_lifecycle_task=lambda: None,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    controller.show_rollback("" if use_picker else "1")
    if use_picker:
        state = await state_loaders[0]()  # type: ignore[operator]
        assert state is not None
        callbacks[0]((2, False, None))  # type: ignore[operator]

    progress = pushed[-1]
    assert isinstance(progress, RollbackProgressModal)
    with pytest.raises(RuntimeError, match="rollback handler failed"):
        await progress._operation()
    assert len(handled) == 1


@pytest.mark.asyncio
async def test_rollback_picker_loader_revalidates_owner_after_lifecycle_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chrys.app.tui.screens.diff as diff_pkg

    pushed: list[object] = []
    state_loaders: list[object] = []
    notifications: list[str] = []
    target_reads: list[str] = []
    session_generation_ref = [7]
    lifecycle_release = asyncio.Event()
    bus = EventBus()

    async def wait_for_release() -> None:
        await lifecycle_release.wait()

    lifecycle_task = asyncio.create_task(wait_for_release())

    class _FakeRollbackModal:
        def __init__(self, **kwargs: object) -> None:
            state_loaders.append(kwargs["load_state"])

    class _FakeEngine:
        mutation_tracker = object()
        mutation_coordinator = None
        current_turn_number = 3

        def available_rollback_turns(self) -> list[int]:
            target_reads.append("targets")
            return [0, 1, 2]

        def turn_prompt_previews(self) -> dict[int, str]:
            return {}

    monkeypatch.setattr(diff_pkg, "RollbackModal", _FakeRollbackModal)
    screen = SimpleNamespace(
        notify=lambda message, **_kwargs: notifications.append(message),
        app=SimpleNamespace(push_screen=lambda modal, _callback=None: pushed.append(modal)),
    )
    controller = RollbackController(
        services=MainScreenServices(bus=bus, engine_provider=_FakeEngine),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: session_generation_ref[0],
        turn_lifecycle_task=lambda: lifecycle_task,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    controller.show_rollback()
    assert len(state_loaders) == 1
    operation = asyncio.create_task(state_loaders[0]())  # type: ignore[operator]
    await asyncio.sleep(0)
    assert target_reads == []

    session_generation_ref[0] += 1
    lifecycle_release.set()
    with pytest.raises(RollbackLoadCancelled):
        await asyncio.wait_for(operation, timeout=1)

    assert target_reads == []
    assert notifications == ["Rollback cancelled because the active session changed."]


@pytest.mark.asyncio
async def test_rollback_picker_loader_waits_for_exact_captured_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chrys.app.tui.screens.diff as diff_pkg

    pushed: list[object] = []
    state_loaders: list[object] = []
    state_reads: list[str] = []
    captured_release = asyncio.Event()

    async def wait_for_release(release: asyncio.Event) -> None:
        await release.wait()

    captured_task = asyncio.create_task(wait_for_release(captured_release))
    lifecycle_ref: list[asyncio.Task[None] | None] = [captured_task]

    class _FakeRollbackModal:
        def __init__(self, **kwargs: object) -> None:
            state_loaders.append(kwargs["load_state"])

    class _FakeEngine(_RollbackProjectionFenceMixin):
        mutation_tracker = object()
        mutation_coordinator = None
        current_turn_number = 3
        conversation_revision = 7

        def available_rollback_turns(self) -> list[int]:
            state_reads.append("targets")
            return [0, 1, 2]

        def turn_prompt_previews(self) -> dict[int, str]:
            state_reads.append("prompts")
            return {}

    monkeypatch.setattr(diff_pkg, "RollbackModal", _FakeRollbackModal)
    screen = SimpleNamespace(
        notify=lambda *_args, **_kwargs: None,
        app=SimpleNamespace(push_screen=lambda modal, _callback=None: pushed.append(modal)),
    )
    controller = RollbackController(
        services=MainScreenServices(bus=EventBus(), engine_provider=_FakeEngine),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: 7,
        turn_lifecycle_task=lambda: lifecycle_ref[0],
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    controller.show_rollback()
    assert len(state_loaders) == 1
    # The engine may clear its visible lifecycle slot during final cleanup;
    # the loader must still await the exact task captured when the modal opened.
    lifecycle_ref[0] = None
    operation = asyncio.create_task(state_loaders[0]())  # type: ignore[operator]
    await asyncio.sleep(0)
    assert state_reads == []

    captured_release.set()
    state = await asyncio.wait_for(operation, timeout=1)
    assert state is not None
    assert state_reads == ["targets", "prompts"]


@pytest.mark.asyncio
async def test_rollback_picker_loader_discards_projection_if_run_revision_changes_during_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chrys.app.tui.screens.diff as diff_pkg

    state_loaders: list[object] = []
    notifications: list[str] = []
    conversation_revision = [7]
    scan_started = threading.Event()
    release_scan = threading.Event()

    class _FakeRollbackModal:
        def __init__(self, **kwargs: object) -> None:
            state_loaders.append(kwargs["load_state"])

    class _FakeEngine(_RollbackProjectionFenceMixin):
        mutation_tracker = object()
        mutation_coordinator = None
        current_turn_number = 3

        @property
        def conversation_revision(self) -> int:
            return conversation_revision[0]

        def available_rollback_turns(self) -> list[int]:
            assert self.rollback_projection_fenced
            scan_started.set()
            assert release_scan.wait(timeout=1)
            return [0, 1, 2]

        def turn_prompt_previews(self) -> dict[int, str]:
            return {}

    monkeypatch.setattr(diff_pkg, "RollbackModal", _FakeRollbackModal)
    screen = SimpleNamespace(
        notify=lambda message, **_kwargs: notifications.append(message),
        app=SimpleNamespace(push_screen=lambda *_args, **_kwargs: None),
    )
    controller = RollbackController(
        services=MainScreenServices(bus=EventBus(), engine_provider=_FakeEngine),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: 7,
        turn_lifecycle_task=lambda: None,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    controller.show_rollback()
    operation = asyncio.create_task(state_loaders[0]())  # type: ignore[operator]
    assert await asyncio.to_thread(scan_started.wait, 1)
    conversation_revision[0] = 8
    release_scan.set()

    with pytest.raises(RollbackLoadCancelled):
        await asyncio.wait_for(operation, timeout=1)
    assert notifications == ["Rollback picker cancelled because the conversation changed."]


@pytest.mark.asyncio
async def test_rollback_picker_loader_keeps_projection_fenced_until_cancelled_scan_quiesces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chrys.app.tui.screens.diff as diff_pkg

    state_loaders: list[object] = []
    scan_started = threading.Event()
    release_scan = threading.Event()
    projection_released = asyncio.Event()

    class _FakeRollbackModal:
        def __init__(self, **kwargs: object) -> None:
            state_loaders.append(kwargs["load_state"])

    class _FakeEngine(_RollbackProjectionFenceMixin):
        mutation_tracker = object()
        mutation_coordinator = None
        current_turn_number = 3
        conversation_revision = 7

        def available_rollback_turns(self) -> list[int]:
            assert self.rollback_projection_fenced
            scan_started.set()
            assert release_scan.wait(timeout=2)
            return [0, 1, 2]

        def turn_prompt_previews(self) -> dict[int, str]:
            assert self.rollback_projection_fenced
            return {}

        def finish_rollback_projection(self, owner: str) -> None:
            super().finish_rollback_projection(owner)
            projection_released.set()

    engine = _FakeEngine()
    monkeypatch.setattr(diff_pkg, "RollbackModal", _FakeRollbackModal)
    screen = SimpleNamespace(
        notify=lambda *_args, **_kwargs: None,
        app=SimpleNamespace(push_screen=lambda *_args, **_kwargs: None),
    )
    controller = RollbackController(
        services=MainScreenServices(bus=EventBus(), engine_provider=lambda: engine),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: 7,
        turn_lifecycle_task=lambda: None,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    controller.show_rollback()
    operation = asyncio.create_task(state_loaders[0]())  # type: ignore[operator]
    assert await asyncio.to_thread(scan_started.wait, 1)

    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert engine.rollback_projection_fenced is True
    assert projection_released.is_set() is False

    release_scan.set()
    await asyncio.wait_for(projection_released.wait(), timeout=1)
    assert engine.rollback_projection_fenced is False


@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["build", "workspace"])
async def test_rollback_picker_preview_rejects_runtime_rebuild_drift(
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    import chrys.app.tui.screens.diff as diff_pkg

    state_loaders: list[object] = []
    notifications: list[str] = []

    class _FakeRollbackModal:
        def __init__(self, **kwargs: object) -> None:
            state_loaders.append(kwargs["load_state"])

    class _FakeEngine(_RollbackProjectionFenceMixin):
        mutation_tracker = object()
        mutation_coordinator = None
        current_turn_number = 3
        conversation_revision = 7

        def available_rollback_turns(self) -> list[int]:
            return [0, 1, 2]

        def turn_prompt_previews(self) -> dict[int, str]:
            return {}

    engine = _FakeEngine()
    monkeypatch.setattr(diff_pkg, "RollbackModal", _FakeRollbackModal)
    screen = SimpleNamespace(
        notify=lambda message, **_kwargs: notifications.append(message),
        app=SimpleNamespace(push_screen=lambda *_args, **_kwargs: None),
    )
    controller = RollbackController(
        services=MainScreenServices(bus=EventBus(), engine_provider=lambda: engine),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: 7,
        turn_lifecycle_task=lambda: None,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    controller.show_rollback()
    state = await state_loaders[0]()  # type: ignore[operator]
    assert state is not None
    if drift == "build":
        engine.build_generation += 1
    else:
        engine.workspace_primary_cwd = "/repo/rebuilt-workspace"

    assert state.projection_acquire is not None
    with pytest.raises(RollbackLoadCancelled):
        await state.projection_acquire()

    assert engine.rollback_projection_fenced is False
    assert notifications == ["Rollback picker cancelled because the workspace or runtime changed."]


def test_rollback_command_rejects_removed_revert_suffix() -> None:
    pushed: list[object] = []
    notifications: list[str] = []

    class _FakeEngine:
        current_turn_number = 3

    screen = SimpleNamespace(
        app=SimpleNamespace(push_screen=lambda modal, _callback=None: pushed.append(modal)),
        notify=lambda message, **_kwargs: notifications.append(message),
    )
    controller = RollbackController(
        services=MainScreenServices(bus=EventBus(), engine_provider=_FakeEngine),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: 1,
        turn_lifecycle_task=lambda: None,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    controller.show_rollback("1 revert")

    assert pushed == []
    assert notifications == ["Usage: /rollback, /rollback N, or /rollback to N."]


def test_rollback_confirmation_binds_picker_projection_before_default_file_revert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chrys.app.tui.screens.diff as diff_pkg

    published: list[UserRollback] = []
    pushed: list[object] = []
    modal_callbacks: list[object] = []
    state_loaders: list[object] = []
    conversation_revision = [7]
    bus = EventBus()

    async def record_rollback(event: UserRollback) -> None:
        published.append(event)

    asyncio.run(bus.subscribe(UserRollback, record_rollback))

    class _FakeRollbackModal:
        def __init__(self, **kwargs: object) -> None:
            state_loaders.append(kwargs["load_state"])

    class _FakeEngine(_RollbackProjectionFenceMixin):
        mutation_tracker = object()
        mutation_coordinator = None
        current_turn_number = 3

        @property
        def conversation_revision(self) -> int:
            return conversation_revision[0]

        def available_rollback_turns(self) -> list[int]:
            return [0, 1, 2]

        def turn_prompt_previews(self) -> dict[int, str]:
            return {}

    def push_screen(modal: object, callback: object | None = None) -> None:
        pushed.append(modal)
        if callback is not None:
            modal_callbacks.append(callback)

    monkeypatch.setattr(diff_pkg, "RollbackModal", _FakeRollbackModal)
    screen = SimpleNamespace(
        app=SimpleNamespace(push_screen=push_screen),
        notify=lambda *_args, **_kwargs: None,
    )
    controller = RollbackController(
        services=MainScreenServices(bus=bus, engine_provider=_FakeEngine),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: 1,
        turn_lifecycle_task=lambda: None,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    controller.show_rollback()

    assert len(pushed) == 1
    assert len(modal_callbacks) == 1
    state = asyncio.run(state_loaders[0]())  # type: ignore[operator]
    assert state is not None
    conversation_revision[0] = 8
    modal_callbacks[0]((2, True, ["/repo/workspace/a.py"]))  # type: ignore[operator]

    assert published == []
    assert len(pushed) == 2
    progress = pushed[1]
    assert isinstance(progress, RollbackProgressModal)
    asyncio.run(progress._operation())
    assert len(published) == 1
    assert published[0].target_turn == 2
    assert published[0].expected_current_turn == 3
    assert published[0].expected_conversation_revision == 7
    assert published[0].expected_build_generation == 1
    assert published[0].expected_workspace_cwd == "/repo/workspace"
    assert published[0].revert_changes is True
    assert published[0].selected_paths == ["/repo/workspace/a.py"]
    assert published[0].session_id == "session-1"


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_changed", [False, True])
async def test_rollback_picker_confirmation_rechecks_owner_after_projection(
    monkeypatch: pytest.MonkeyPatch,
    owner_changed: bool,
) -> None:
    import chrys.app.tui.screens.diff as diff_pkg

    published: list[UserRollback] = []
    pushed: list[object] = []
    modal_callbacks: list[object] = []
    state_loaders: list[object] = []
    notifications: list[str] = []
    session_generation_ref = [1]
    lifecycle_release = asyncio.Event()
    bus = EventBus()

    async def record_rollback(event: UserRollback) -> None:
        published.append(event)

    async def wait_for_release() -> None:
        await lifecycle_release.wait()

    await bus.subscribe(UserRollback, record_rollback)
    lifecycle_task = asyncio.create_task(wait_for_release())

    class _FakeRollbackModal:
        def __init__(self, **kwargs: object) -> None:
            state_loaders.append(kwargs["load_state"])

    class _FakeEngine(_RollbackProjectionFenceMixin):
        mutation_tracker = object()
        mutation_coordinator = None
        current_turn_number = 3
        conversation_revision = 7

        def available_rollback_turns(self) -> list[int]:
            return [0, 1, 2]

        def turn_prompt_previews(self) -> dict[int, str]:
            return {}

    def push_screen(modal: object, callback: object | None = None) -> None:
        pushed.append(modal)
        if callback is not None:
            modal_callbacks.append(callback)

    monkeypatch.setattr(diff_pkg, "RollbackModal", _FakeRollbackModal)
    screen = SimpleNamespace(
        app=SimpleNamespace(push_screen=push_screen),
        notify=lambda message, **_kwargs: notifications.append(message),
    )
    controller = RollbackController(
        services=MainScreenServices(bus=bus, engine_provider=_FakeEngine),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: session_generation_ref[0],
        turn_lifecycle_task=lambda: lifecycle_task,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    controller.show_rollback()
    lifecycle_release.set()
    state = await state_loaders[0]()  # type: ignore[operator]
    assert state is not None
    modal_callbacks[0]((2, False, None))  # type: ignore[operator]
    progress = pushed[1]
    assert isinstance(progress, RollbackProgressModal)

    if owner_changed:
        session_generation_ref[0] += 1
    await asyncio.wait_for(progress._operation(), timeout=1)
    if owner_changed:
        assert published == []
        assert notifications == ["Rollback cancelled because the active session changed."]
    else:
        assert len(published) == 1
        assert published[0].target_turn == 2
        assert published[0].expected_current_turn == 3
        assert published[0].expected_conversation_revision == 7
        assert published[0].expected_build_generation == 1
        assert published[0].expected_workspace_cwd == "/repo/workspace"


@pytest.mark.parametrize(
    ("active_session_id", "active_session_generation"),
    (("session-2", 7), ("session-1", 8)),
)
def test_rollback_direct_command_cancels_after_session_identity_changes(
    active_session_id: str,
    active_session_generation: int,
) -> None:
    pushed: list[object] = []
    notifications: list[str] = []
    published: list[UserRollback] = []
    session_id_ref = ["session-1"]
    session_generation_ref = [7]
    bus = EventBus()

    async def record_rollback(event: UserRollback) -> None:
        published.append(event)

    asyncio.run(bus.subscribe(UserRollback, record_rollback))

    class _FakeEngine:
        mutation_tracker = object()
        mutation_coordinator = None
        current_turn_number = 3

        def available_rollback_turns(self) -> list[int]:
            return [0, 1, 2]

        def turn_prompt_previews(self) -> dict[int, str]:
            return {}

    screen = SimpleNamespace(
        notify=lambda message, **_kwargs: notifications.append(message),
        app=SimpleNamespace(push_screen=lambda modal, _callback=None: pushed.append(modal)),
    )
    controller = RollbackController(
        services=MainScreenServices(bus=bus, engine_provider=_FakeEngine),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: session_id_ref[0],
        session_generation=lambda: session_generation_ref[0],
        turn_lifecycle_task=lambda: None,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    controller.show_rollback("1")
    assert len(pushed) == 1

    # The first case switches to another session. The second simulates
    # switching away and back: the visible ID matches again, but the
    # monotonic engine generation still proves this is a different owner.
    session_id_ref[0] = active_session_id
    session_generation_ref[0] = active_session_generation
    progress = pushed[0]
    assert isinstance(progress, RollbackProgressModal)
    asyncio.run(progress._operation())

    assert published == []
    assert notifications == ["Rollback cancelled because the active session changed."]


def test_rollback_direct_command_rechecks_busy_state_before_dispatch() -> None:
    pushed: list[object] = []
    notifications: list[str] = []
    published: list[UserRollback] = []
    agent_busy = False
    bus = EventBus()

    async def record_rollback(event: UserRollback) -> None:
        published.append(event)

    asyncio.run(bus.subscribe(UserRollback, record_rollback))

    class _FakeEngine:
        mutation_tracker = object()
        mutation_coordinator = None
        current_turn_number = 3

        def available_rollback_turns(self) -> list[int]:
            return [0, 1, 2]

        def turn_prompt_previews(self) -> dict[int, str]:
            return {}

    screen = SimpleNamespace(
        notify=lambda message, **_kwargs: notifications.append(message),
        app=SimpleNamespace(push_screen=lambda modal, _callback=None: pushed.append(modal)),
    )
    controller = RollbackController(
        services=MainScreenServices(bus=bus, engine_provider=_FakeEngine),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: agent_busy,
        current_session_id=lambda: "session-1",
        session_generation=lambda: 7,
        turn_lifecycle_task=lambda: None,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    controller.show_rollback("1")
    assert len(pushed) == 1
    agent_busy = True
    progress = pushed[0]
    assert isinstance(progress, RollbackProgressModal)
    asyncio.run(progress._operation())

    assert published == []
    assert notifications == ["Cannot roll back while the agent is running."]


def test_rollback_result_restores_rolled_back_user_text() -> None:
    notifications: list[tuple[str, dict]] = []
    restored: list[str] = []

    class _RollbackView:
        def notify(
            self,
            message: str,
            *,
            title: str,
            severity: str = "information",
            timeout: float | None = 3,
        ) -> None:
            notifications.append((message, {"title": title, "severity": severity, "timeout": timeout}))

        def open_rollback_modal(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("unexpected modal")

        async def restore_welcome_rollback(self, *, session_id: str | None, profile: str, cwd: str) -> None:
            raise AssertionError("unexpected welcome restore")

        def restore_input_text(self, text: str) -> None:
            restored.append(text)

    controller = RollbackController(
        services=MainScreenServices(bus=EventBus(), engine_provider=lambda: None),
        view=_RollbackView(),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: 1,
        turn_lifecycle_task=lambda: None,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    asyncio.run(
        controller.on_result(
            RollbackResult(session_id="session-1", target_turn=1, rolled_back_user_text="second prompt")
        )
    )

    assert restored == ["second prompt"]
    assert notifications[-1][1].get("severity") == "information"


def test_welcome_rollback_restores_input_after_welcome_state() -> None:
    calls: list[tuple[object, ...]] = []

    class _RollbackView:
        def notify(
            self,
            message: str,
            *,
            title: str,
            severity: str = "information",
            timeout: float | None = 3,
        ) -> None:
            calls.append(("notify", message))

        def open_rollback_modal(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("unexpected modal")

        async def restore_welcome_rollback(self, *, session_id: str | None, profile: str, cwd: str) -> None:
            calls.append(("welcome", session_id, profile, cwd))

        def restore_input_text(self, text: str) -> None:
            calls.append(("input", text))

    controller = RollbackController(
        services=MainScreenServices(bus=EventBus(), engine_provider=lambda: None),
        view=_RollbackView(),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: 1,
        turn_lifecycle_task=lambda: None,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda cwd: calls.append(("workspace", cwd)),
        set_has_messages=lambda value: calls.append(("has_messages", value)),
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    asyncio.run(
        controller.on_result(
            RollbackResult(session_id="session-1", target_turn=0, rolled_back_user_text="first prompt")
        )
    )

    assert calls[:4] == [
        ("has_messages", False),
        ("workspace", "/repo/workspace"),
        ("welcome", "session-1", "Code", "/repo/workspace"),
        ("input", "first prompt"),
    ]


def test_rollback_result_surfaces_exclusions_and_warnings() -> None:
    """Plan exclusions/warnings reach the toast even with zero restore results.

    A plan whose every file is excluded restores nothing, so the terminal
    result still needs to report exclusions explicitly (PR1 invariant).
    """
    notifications: list[tuple[str, dict]] = []
    debug_details: list[str] = []
    screen = SimpleNamespace(
        _engine=None,
        _profile="Code",
        _workspace_cwd=lambda: "/repo/workspace",
        notify=lambda msg, **kwargs: notifications.append((msg, kwargs)),
        _debug=lambda _title, detail: debug_details.append(detail),
    )

    asyncio.run(
        _make_rollback_controller_for_test(screen).on_result(
            RollbackResult(
                session_id="session-1",
                target_turn=2,
                restore_results=[],
                exclusions=[
                    ("/ws/shared.py", "contested"),
                    ("/ws/peer[7].txt", "foreign"),
                    ("/ws/logo.png", "move_poisoned"),
                    ("/ws/big.log", "unrestorable"),
                ],
                warnings=["1 other session is writing to this workspace."],
            )
        )
    )

    msg, kwargs = notifications[-1]
    assert kwargs.get("severity") == "warning"
    # Toasts render as plain text, so paths like ``peer[7].txt`` stay literal.
    rendered = msg
    assert "4 files excluded from revert" in rendered
    # Reasons render with the modal's human labels, not raw enum values.
    assert "shared.py (modified internally and externally)" in rendered
    assert "peer[7].txt (changed by another session)" in rendered
    assert "(+1 more)" in rendered
    assert "1 other session is writing" in rendered
    assert any("4 excluded" in detail and "1 warning" in detail for detail in debug_details)


# ──────────── _accumulate_shell_snapshots ─────────────────────────────


def test_accumulate_shell_snapshots_empty_metadata() -> None:
    handler, live = _make_handler()
    handler._accumulate_shell_snapshots({})
    assert live == {}


def test_accumulate_shell_snapshots_none_value() -> None:
    handler, live = _make_handler()
    handler._accumulate_shell_snapshots({"shell_file_snapshots": None})
    assert live == {}


def test_accumulate_shell_snapshots_basic() -> None:
    handler, live = _make_handler()
    handler._accumulate_shell_snapshots(
        {
            "shell_file_snapshots": {
                "/tmp/a.py": _shell_snapshot("old-a", "new-a", "modify"),
                "/tmp/b.py": _shell_snapshot("", "new-b", "create"),
            }
        }
    )
    assert live["/tmp/a.py"] == _live("old-a", "new-a", "modify")
    assert live["/tmp/b.py"] == _live("", "new-b", "create")


def test_accumulate_shell_snapshots_merges_with_existing() -> None:
    handler, live = _make_handler()
    handler._accumulate_live_mutation("/tmp/a.py", "original", "v1")
    handler._accumulate_shell_snapshots(
        {
            "shell_file_snapshots": {
                "/tmp/a.py": _shell_snapshot("v1", "v2", "modify"),
            }
        }
    )
    # Should keep "original" as before_text
    assert live["/tmp/a.py"] == _live("original", "v2", "modify")


def test_accumulate_shell_snapshots_preserves_implicit_source() -> None:
    handler, live = _make_handler()
    handler._accumulate_shell_snapshots(
        {
            "shell_file_snapshots": {
                "/tmp/a.py": _shell_snapshot(
                    "original",
                    "v1",
                    "modify",
                    source=MutationSource.IMPLICIT.value,
                ),
            }
        }
    )
    handler._accumulate_live_mutation("/tmp/a.py", "v1", "v2")

    assert live["/tmp/a.py"] == _live(
        "original",
        "v2",
        "modify",
        source=MutationSource.IMPLICIT.value,
    )


def test_accumulate_shell_snapshots_keeps_metadata_only_change() -> None:
    handler, live = _make_handler()
    handler._accumulate_shell_snapshots(
        {
            "shell_file_snapshots": {
                "/tmp/a.py": _shell_snapshot(
                    "same\n",
                    "same\n",
                    "modify",
                    bytes_changed=True,
                    before_hash="h1",
                    after_hash="h2",
                ),
            }
        }
    )
    assert live["/tmp/a.py"] == _live(
        "same\n",
        "same\n",
        "modify",
        bytes_changed=True,
        before_hash="h1",
        after_hash="h2",
    )


# ──────────── mutation_op_for_live_op ─────────────────────────────────


def test_op_str_map_covers_all_ops() -> None:
    assert mutation_op_for_live_op("create") is MutationOp.CREATE
    assert mutation_op_for_live_op("modify") is MutationOp.MODIFY
    assert mutation_op_for_live_op("delete") is MutationOp.DELETE
    assert mutation_op_for_live_op("move") is MutationOp.MOVE


def test_op_str_map_defaults_to_modify_for_unknown() -> None:
    assert mutation_op_for_live_op("unknown") is MutationOp.MODIFY


# ──────────── on_sub_agent_paused (interrupt-race gate) ─────────────────
#
# During a user interrupt the screen flips ``_agent_running`` to False
# BEFORE the backend cascade tears down live sub-agent controllers.  A
# ``SubAgentPaused`` event that was already in-flight when the interrupt
# fired can therefore arrive at the TUI handler AFTER the UI has already
# considered the run stopped.  Without a gate the card would flicker
# into a paused state carrying Retry/Abort buttons that point at a
# controller the engine has already dropped — see event_handlers.py
# comment for the full rationale.


def _make_pause_handler(agent_running: bool) -> tuple[BackendEventHandler, list[tuple]]:
    """Build a handler whose mock screen records ChatPanel calls.

    When the gate passes, ``query_one(ChatPanel)`` must be called and
    then ``panel.sub_agent_paused(...)`` must record its args in the
    returned list.  When the gate short-circuits, the list stays empty
    and ``query_one`` is never invoked.
    """
    calls: list[tuple] = []

    class _FakePanel:
        def sub_agent_paused(self, *args) -> None:
            calls.append(args)

    def _query_one(_cls):
        return _FakePanel()

    screen = SimpleNamespace(_agent_running=agent_running, query_one=_query_one)
    handler = make_backend_handler(screen)
    return handler, calls


async def _run_pause(handler: BackendEventHandler) -> None:
    from chrys.foundation.events.types import SubAgentPaused

    event = SubAgentPaused(
        invocation_id="inv-1",
        reason="stream_stall",
        last_error="boom",
        retry_attempts=3,
        diagnostic_path="/session/approvals/acp.log",
    )
    await handler.on_sub_agent_paused(event)


def test_on_sub_agent_paused_forwards_when_running() -> None:
    import asyncio

    handler, calls = _make_pause_handler(agent_running=True)
    asyncio.run(_run_pause(handler))
    assert calls == [("inv-1", "stream_stall", "boom", 3, "/session/approvals/acp.log")]


def test_on_sub_agent_paused_gated_after_interrupt() -> None:
    """Late paused event after the interrupt sets ``_agent_running=False``
    is dropped — no ChatPanel lookup, no paused card."""
    import asyncio

    handler, calls = _make_pause_handler(agent_running=False)
    asyncio.run(_run_pause(handler))
    assert calls == []


# ──────────── compaction events: status bar text ────────────────────────
#
# While Phase-4 compaction runs, the status bar must read "Compacting
# conversation..." instead of the stale "Thinking"/"Running: <tool>" text,
# and flip back to "Thinking" once the note is in — but only while the run
# is still live, so a canceled-outcome event arriving during interrupt
# teardown cannot resurrect the hidden status bar (``show`` re-adds
# ``-visible``).


def _make_compaction_handler(agent_running: bool) -> tuple[BackendEventHandler, list[tuple]]:
    """Build a handler whose mock screen records ChatPanel/StatusBar calls."""
    calls: list[tuple] = []

    class _FakeWidget:
        async def add_compaction_start(self, compaction_id: str) -> None:
            calls.append(("add", compaction_id))

        def complete_compaction(self, compaction_id: str, **kwargs: object) -> None:
            calls.append(
                (
                    "complete",
                    compaction_id,
                    kwargs.get("outcome"),
                    kwargs.get("format_violation"),
                    kwargs.get("failure_reason"),
                )
            )

        def show(self, text: MessageRef | str) -> None:
            calls.append(("show", _status_text(text)))

    widget = _FakeWidget()
    screen = SimpleNamespace(_agent_running=agent_running, query_one=lambda _cls: widget)
    handler = make_backend_handler(screen)
    return handler, calls


def test_on_compaction_started_sets_status_bar_to_compacting() -> None:
    handler, calls = _make_compaction_handler(agent_running=True)
    asyncio.run(handler.on_compaction_started(CompactionStarted(compaction_id="c-1")))
    assert calls.index(("add", "c-1")) < calls.index(("show", "Compacting conversation..."))


def test_on_compaction_started_gated_after_interrupt() -> None:
    handler, calls = _make_compaction_handler(agent_running=False)
    asyncio.run(handler.on_compaction_started(CompactionStarted(compaction_id="c-1")))
    assert calls == []


def test_on_compaction_finished_restores_thinking_status_while_running() -> None:
    handler, calls = _make_compaction_handler(agent_running=True)
    violation = 'missing required heading "## Next"'
    event = CompactionFinished(compaction_id="c-1", outcome="ok", duration_ms=5, format_violation=violation)
    asyncio.run(handler.on_compaction_finished(event))
    assert ("complete", "c-1", "ok", violation, "") in calls
    assert ("show", "Thinking") in calls


def test_on_compaction_finished_does_not_resurrect_status_after_interrupt() -> None:
    """The card still gets finalized, but the hidden status bar stays hidden."""
    handler, calls = _make_compaction_handler(agent_running=False)
    event = CompactionFinished(compaction_id="c-1", outcome="canceled")
    asyncio.run(handler.on_compaction_finished(event))
    assert ("complete", "c-1", "canceled", "", "") in calls
    assert not any(call[0] == "show" for call in calls)


def test_on_compaction_finished_canceled_never_restores_thinking_status() -> None:
    """Canceled = interrupted run: during teardown the event can arrive while
    ``agent_running`` is still True, and re-showing "Thinking" would overwrite
    the "Interrupted by user" flash and stick forever."""
    handler, calls = _make_compaction_handler(agent_running=True)
    event = CompactionFinished(compaction_id="c-1", outcome="canceled")
    asyncio.run(handler.on_compaction_finished(event))
    assert ("complete", "c-1", "canceled", "", "") in calls
    assert not any(call[0] == "show" for call in calls)


def test_on_compaction_finished_threads_failure_reason_to_card() -> None:
    handler, calls = _make_compaction_handler(agent_running=True)
    reason = "20 attempts limit exceeded for current turn"
    event = CompactionFinished(compaction_id="c-1", outcome="failed", duration_ms=201, failure_reason=reason)
    asyncio.run(handler.on_compaction_finished(event))
    assert ("complete", "c-1", "failed", "", reason) in calls


def test_on_context_compressed_forwards_turn_range_to_sidebar() -> None:
    calls: list[tuple[object, ...]] = []

    class _FakeChatPanel:
        async def add_context_fold(
            self,
            context_id: str,
            summary: str,
            freed_messages: int,
            turn_range: tuple[int, int],
        ) -> None:
            calls.append(("fold", context_id, summary, freed_messages, turn_range))

        def mark_turn_range_compressed(self, turn_range: tuple[int, int]) -> bool:
            calls.append(("mark", turn_range))
            return True

    class _FakeContextPanel:
        def add_compressed_block(
            self,
            context_id: str,
            summary: str,
            freed_messages: int,
            turn_range: tuple[int, int],
        ) -> None:
            calls.append(("sidebar", context_id, summary, freed_messages, turn_range))

    chat = _FakeChatPanel()
    sidebar = SimpleNamespace(context_panel=_FakeContextPanel())

    def query_one(cls: type) -> object:
        if cls.__name__ == "ChatPanel":
            return chat
        if cls.__name__ == "SidebarPanel":
            return sidebar
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(query_one=query_one, _update_toc=lambda: calls.append(("toc",)))
    handler = make_backend_handler(screen)

    asyncio.run(
        handler.on_context_compressed(
            ContextCompressed(
                compressed_context_id="ctx_930083b2",
                summary="Completed turns",
                freed_messages=18,
                turn_range=(1, 4),
            )
        )
    )

    assert ("fold", "ctx_930083b2", "Completed turns", 18, (1, 4)) in calls
    assert ("sidebar", "ctx_930083b2", "Completed turns", 18, (1, 4)) in calls
    assert ("toc",) in calls


def test_on_context_pressure_surfaces_user_visible_warning() -> None:
    notifications: list[tuple[str, str, str]] = []
    debug_calls: list[tuple[str, str]] = []

    def notify(message: str, *, title: str, severity: str, **_kwargs: object) -> None:
        notifications.append((message, title, severity))

    screen = SimpleNamespace(notify=notify, _debug=lambda *args: debug_calls.append(args))
    handler = make_backend_handler(screen)

    asyncio.run(
        handler.on_context_pressure(
            ContextPressure(
                reason="side_call_budget",
                attempts=2,
                side_call_tokens=300_000,
                side_call_token_budget=300_000,
                source="main",
            )
        )
    )

    assert notifications == [
        (
            (
                "Conversation context compaction stopped because the progress-note token budget was exhausted. "
                "The active task may exceed its model window."
            ),
            "Warning",
            "warning",
        )
    ]
    assert debug_calls[-1] == ("ContextPressure", "main:side_call_budget attempts=2 side_calls=300,000/300,000")


def test_on_context_pressure_does_not_suppress_later_sub_agent_invocations() -> None:
    notifications: list[str] = []

    def notify(message: str, **_kwargs: object) -> None:
        notifications.append(message)

    screen = SimpleNamespace(notify=notify, _debug=lambda *_args: None)
    handler = make_backend_handler(screen)

    for invocation_id in ("inv-1", "inv-2"):
        asyncio.run(
            handler.on_context_pressure(
                ContextPressure(
                    reason="no_progress",
                    source="sub_agent",
                    invocation_id=invocation_id,
                )
            )
        )

    assert len(notifications) == 2
    assert notifications[0] == notifications[1]


# ──────────── on_retry_attempt: prepare_retry wiring ───────────────────
#
# On a main-agent stream stall, ``StreamRetryLoop`` calls
# ``_restore_history`` to roll ``chrys_history`` back to before the failed
# attempt.  The TUI must mirror this by finalising any pending
# intermediate-text / tool-group widgets from the failed run — otherwise
# new tool calls from the retry would mount under the stale assistant
# block.  The wiring contract: ``on_retry_attempt`` calls
# ``panel.prepare_retry()`` BEFORE ``panel.add_retry(...)`` so the retry
# notice lands after the cleanup.


def test_on_retry_attempt_calls_prepare_retry_before_add_retry() -> None:
    """Regression: wiring between the ``RetryAttempt`` event and the
    ``ChatPanel.prepare_retry()`` cleanup hook.  Order matters — the
    retry notice must be mounted AFTER pending widgets are finalised."""
    import asyncio

    from chrys.foundation.events.types import RetryAttempt

    call_log: list[str] = []

    class _FakePanel:
        async def prepare_retry(self) -> None:
            call_log.append("prepare_retry")

        async def add_retry(self, *_args) -> None:
            call_log.append("add_retry")

    class _FakeStatusBar:
        def show(self, _msg: str) -> None:
            call_log.append("status_show")

    # Local sentinel classes so ``query_one`` can dispatch by type
    # without importing the real widgets (which would require a running
    # Textual app for instantiation).
    panel_inst = _FakePanel()
    status_inst = _FakeStatusBar()

    def _query_one(cls):
        # Match by class NAME so the real widget imports inside
        # on_retry_attempt resolve to our fakes.
        if cls.__name__ == "ChatPanel":
            return panel_inst
        if cls.__name__ == "StatusBar":
            return status_inst
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    def _debug(_tag: str, _msg: str) -> None:
        pass

    screen = SimpleNamespace(_agent_running=True, query_one=_query_one, _debug=_debug)
    handler = make_backend_handler(screen)

    event = RetryAttempt(message="Stream stalled", attempt=1, max_attempts=5, delay_seconds=3)
    asyncio.run(handler.on_retry_attempt(event))

    # Critical order: prepare_retry must run BEFORE add_retry so the
    # retry notice mounts after the stale tool group is closed.
    assert call_log[0] == "prepare_retry"
    assert call_log[1] == "add_retry"


def test_on_retry_attempt_compaction_scope_uses_detail_without_parsing_message() -> None:
    """Phase-4 side-call retries land on the live compaction card — they
    must not finalize pending widgets, mount the transcript banner, or
    flip the status bar to "Retrying"."""
    import asyncio

    from chrys.foundation.events.types import RetryAttempt

    call_log: list[object] = []

    class _FakePanel:
        async def prepare_retry(self) -> None:
            call_log.append("prepare_retry")

        async def add_retry(self, *_args) -> None:
            call_log.append("add_retry")

        def show_compaction_retry(self, message: str, attempt: int, max_attempts: int, delay_seconds: int) -> None:
            call_log.append(("compaction_retry", message, attempt, max_attempts, delay_seconds))

    class _FakeStatusBar:
        def show(self, _msg: str) -> None:
            call_log.append("status_show")

    panel_inst = _FakePanel()
    status_inst = _FakeStatusBar()

    def _query_one(cls):
        if cls.__name__ == "ChatPanel":
            return panel_inst
        if cls.__name__ == "StatusBar":
            return status_inst
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(_agent_running=True, query_one=_query_one, _debug=lambda *_: None)
    handler = make_backend_handler(screen)

    event = RetryAttempt(
        message="opaque compatibility diagnostic",
        attempt=2,
        max_attempts=5,
        delay_seconds=7,
        scope="compaction",
        detail="empty response",
    )
    asyncio.run(handler.on_retry_attempt(event))

    # The card gets the semantic detail verbatim; nothing else fires.
    assert call_log == [("compaction_retry", "empty response", 2, 5, 7)]


def test_retry_attempt_display_message_localizes_banner_and_keeps_english_bytes() -> None:
    """The transcript retry banner prefers the producer's display reference;
    the compaction card keeps the raw separated detail regardless."""
    import asyncio

    from chrys.foundation.events.types import RetryAttempt
    from chrys.foundation.i18n import DisplayBlock
    from chrys.orchestration.engine.build.builder import _RETRY_LAST_WORDS_COMPACTION
    from chrys.orchestration.engine.executor import _RETRY_STREAM_STALLED

    def _make(locale_controller: LocaleController | None) -> tuple[object, list[str], list[str]]:
        banners: list[str] = []
        cards: list[str] = []

        class _FakePanel:
            async def prepare_retry(self) -> None:
                pass

            async def add_retry(self, message: str, *_args) -> None:
                banners.append(message)

            def show_compaction_retry(self, message: str, *_args) -> None:
                cards.append(message)

        class _FakeStatusBar:
            def show(self, _msg) -> None:
                pass

        panel = _FakePanel()
        status = _FakeStatusBar()

        def _query_one(cls):
            if cls.__name__ == "ChatPanel":
                return panel
            if cls.__name__ == "StatusBar":
                return status
            raise AssertionError(f"unexpected query_one({cls.__name__})")

        screen = SimpleNamespace(_agent_running=True, query_one=_query_one, _debug=lambda *_: None)
        return make_backend_handler(screen, locale_controller=locale_controller), banners, cards

    stalled = RetryAttempt(
        message="Stream stalled",
        attempt=1,
        max_attempts=5,
        delay_seconds=3,
        display_message=_RETRY_STREAM_STALLED.bind(),
    )

    handler, banners, _cards = _make(LocaleController(Settings(locale="zh-Hans")))
    asyncio.run(handler.on_retry_attempt(stalled))
    assert banners == ["响应流中断"]

    handler, banners, _cards = _make(None)
    asyncio.run(handler.on_retry_attempt(stalled))
    assert banners == [stalled.message]

    compaction = RetryAttempt(
        message="LAST_WORDS compaction: empty response",
        attempt=2,
        max_attempts=5,
        delay_seconds=7,
        scope="compaction",
        detail="empty response",
        display_message=_RETRY_LAST_WORDS_COMPACTION.bind(reason=DisplayBlock("empty response")),
    )
    handler, banners, cards = _make(LocaleController(Settings(locale="zh-Hans")))
    asyncio.run(handler.on_retry_attempt(compaction))
    assert banners == []
    assert cards == ["empty response"]


def test_on_retry_attempt_skipped_when_not_running() -> None:
    """After interrupt the handler drops stale RetryAttempt events —
    including the cleanup call."""
    import asyncio

    from chrys.foundation.events.types import RetryAttempt

    call_log: list[str] = []

    class _FakePanel:
        async def prepare_retry(self) -> None:
            call_log.append("prepare_retry")

        async def add_retry(self, *_args) -> None:
            call_log.append("add_retry")

    def _query_one(_cls):
        raise AssertionError("query_one must not be called when _agent_running is False")

    screen = SimpleNamespace(_agent_running=False, query_one=_query_one, _debug=lambda *_: None)
    handler = make_backend_handler(screen)

    event = RetryAttempt(message="Stream stalled", attempt=1, max_attempts=5, delay_seconds=3)
    asyncio.run(handler.on_retry_attempt(event))

    assert call_log == []


def test_input_bar_retry_hides_trailing_status_action_before_retry() -> None:
    from chrys.app.tui.widgets.chrome.input_bar import InputBar

    calls: list[object] = []

    class _FakeChatPanel:
        def hide_trailing_status_action(self) -> None:
            calls.append("hide_status_action")

    def query_one(cls: type) -> object:
        if cls.__name__ == "ChatPanel":
            return _FakeChatPanel()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    flow = SimpleNamespace(request_retry=lambda text: calls.append(("retry", text)))
    screen = SimpleNamespace(query_one=query_one, _input_flow=flow, _model_unconfigured=lambda: False)

    MainScreen._on_retry_requested(screen, InputBar.RetryRequested("please continue"))

    assert calls == ["hide_status_action", ("retry", "please continue")]


def test_inline_status_retry_resumes_without_consuming_input_draft() -> None:
    from chrys.app.tui.widgets.chat.messages import ConversationStatusAction
    from chrys.app.tui.widgets.chrome.input_bar import InputBar

    calls: list[object] = []

    class _FakeInputBar:
        retry_mode = True

        def consume_retry_text(self) -> str:
            calls.append("consume_retry_text")
            return "typed note"

    def query_one(cls: type) -> object:
        if cls is InputBar:
            return _FakeInputBar()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    flow = SimpleNamespace(request_retry=lambda text: calls.append(("retry", text)))
    screen = SimpleNamespace(
        _agent_loading=False,
        query_one=query_one,
        _input_flow=flow,
        _model_unconfigured=lambda: False,
    )

    MainScreen._on_conversation_status_action_pressed(screen, ConversationStatusAction.Pressed())

    # The inline card resumes plainly: the typed draft is neither consumed
    # nor sent as the continuation prompt (that is the input bar's button).
    assert calls == [("retry", "")]


def test_do_retry_ignores_duplicate_while_retry_submit_is_pending() -> None:
    state = MainScreenState()
    state.submit.active = True
    controller, _state, view = _make_input_flow_controller(EventBus(), state=state)

    asyncio.run(controller.retry("duplicate note"))

    assert view.calls == []


def test_retry_defers_fast_error_until_resuming_status_is_initialized() -> None:
    bus = EventBus()
    state = MainScreenState()
    order: list[str] = []

    class _FlowView(_FakeInputFlowView):
        def start_run_status(self, label: MessageRef | str) -> None:
            super().start_run_status(label)
            order.append(f"status:{_status_text(label)}")

        def hide_trailing_status_action(self) -> None:
            order.append("hide_status_action")

    class _FakeStatusBar:
        def flash(self, message: str, *, error: bool = False) -> None:
            order.append(f"flash:{message}:{error}")

    class _FakeInputBar:
        def __init__(self) -> None:
            self.locked = True
            self._retry_label = ""
            self.retry_mode = False

        def unlock_and_keep(self) -> None:
            self.locked = False
            order.append("unlock")

    class _FakeChatPanel:
        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            order.append(f"error:{message}:{action_label}")

    def query_one(cls: object) -> object:
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ChatPanel":
            return _FakeChatPanel()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    input_bar = _FakeInputBar()
    view = _FlowView()
    screen = SimpleNamespace(
        _state=state,
        _agent_running=False,
        _mark_terminal_title_failed=lambda: None,
        query_one=query_one,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None
    controller, _state, _view = _make_input_flow_controller(
        bus,
        state=state,
        view=view,
        handle_error=handler.on_error,
    )

    async def _fail_retry(_event: UserRetry) -> None:
        await bus.publish(Error(code="executor_error", message="retry failed"))

    async def _run() -> None:
        await bus.subscribe(UserRetry, _fail_retry)
        await bus.subscribe(Error, handler.on_error)
        await controller.retry()

    asyncio.run(_run())

    assert order.index("status:Resuming") < order.index("error:retry failed:Retry")
    assert state.submit.blocked is False
    assert state.run.agent_running is False
    assert input_bar.retry_mode is True


def test_focus_guard_allows_inline_status_action_button_focus() -> None:
    class _FakeStatusButton:
        def __init__(self) -> None:
            self.ancestors_with_self: list[object] = []

        def has_class(self, name: str) -> bool:
            return name == "status-action-btn"

    controller, _state, _shell_view, focus_view, _shell_mode_states = _make_shell_mode_controller()

    controller.on_descendant_focus(_FakeStatusButton())

    assert focus_view.calls == []


def test_focus_guard_still_redirects_display_only_chat_descendants() -> None:
    from chrys.app.tui.widgets.chat.panel import ChatPanel

    class _FakeChatChild:
        def __init__(self) -> None:
            self.ancestors_with_self = [self, ChatPanel()]

        def has_class(self, _name: str) -> bool:
            return False

    controller, _state, _shell_view, focus_view, _shell_mode_states = _make_shell_mode_controller()

    controller.on_descendant_focus(_FakeChatChild())

    assert focus_view.calls == ["focus_input"]


def test_focus_guard_allows_ask_user_inline_controls_to_keep_focus() -> None:
    from chrys.app.tui.widgets.ask_user_controls import AskUserResponseFooter

    controller, _state, _shell_view, focus_view, _shell_mode_states = _make_shell_mode_controller()

    controller.on_descendant_focus(AskUserResponseFooter("req-1"))

    assert focus_view.calls == []


def test_shell_mode_state_watcher_owns_layout() -> None:
    from textual.widgets import Footer

    from chrys.app.tui.terminal.panel import ShellPanel
    from chrys.app.tui.terminal.widget import Terminal
    from chrys.app.tui.widgets.chat.panel import ChatPanel
    from chrys.app.tui.widgets.chat.session_json import SessionJsonPanel
    from chrys.app.tui.widgets.chrome.input_bar import InputBar
    from chrys.app.tui.widgets.chrome.status_bar import StatusBar
    from chrys.app.tui.widgets.sidebar.panel import SidebarPanel
    from chrys.app.tui.widgets.trajectory import TrajectoryDashboard

    calls: list[tuple[str, object]] = []

    class _FakeChat:
        display = True

    class _FakeSessionJson:
        display = False

        def suspend_for_shell_mode(self) -> None:
            calls.append(("session_json_suspend", None))
            self.display = False

        def finish_shell_mode(self, *, restore: bool) -> None:
            calls.append(("session_json_finish", restore))
            self.display = restore

        def hide_session_json(self) -> None:
            self.display = False

    class _FakeDashboard:
        display = False
        session_json_visible = False

        def suspend_for_shell_mode(self) -> bool:
            self.display = False
            return False

        def finish_shell_mode(self) -> bool:
            return False

    class _FakeTerminal:
        def focus(self) -> None:
            calls.append(("terminal_focus", None))

    class _FakeShell:
        def show(self) -> None:
            calls.append(("shell_show", None))

        def hide(self) -> None:
            calls.append(("shell_hide", None))

        def query_one(self, cls: type) -> object:
            if cls is Terminal:
                return _FakeTerminal()
            raise AssertionError(f"unexpected shell query_one({cls.__name__})")

    class _FakeStatus:
        def snapshot(self) -> dict[str, object]:
            calls.append(("status_snapshot", None))
            return {"visible": True, "status": "Thinking"}

        def remove_class(self, class_name: str) -> None:
            calls.append(("status_remove", class_name))

        def flash(self, message: str, **kwargs: object) -> None:
            calls.append(("status_flash", (message, kwargs)))

        def restore(self, state: dict[str, object]) -> None:
            calls.append(("status_restore", state))

    class _FakeInput:
        display = True

        def focus_input(self) -> None:
            calls.append(("input_focus", None))

    class _FakeFooter:
        display = True

    class _FakeSidebar:
        def add_class(self, class_name: str) -> None:
            calls.append(("sidebar_add", class_name))

        def remove_class(self, class_name: str) -> None:
            calls.append(("sidebar_remove", class_name))

    chat = _FakeChat()
    session_json = _FakeSessionJson()
    dashboard = _FakeDashboard()
    shell = _FakeShell()
    status = _FakeStatus()
    input_bar = _FakeInput()
    footer = _FakeFooter()
    sidebar = _FakeSidebar()

    def query_one(cls: type) -> object:
        if cls is ChatPanel:
            return chat
        if cls is SessionJsonPanel:
            return session_json
        if cls is TrajectoryDashboard:
            return dashboard
        if cls is ShellPanel:
            return shell
        if cls is StatusBar:
            return status
        if cls is InputBar:
            return input_bar
        if cls is Footer:
            return footer
        if cls is SidebarPanel:
            return sidebar
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _shell_mode=False,
        _fullscreen_terminal=False,
        _sb_saved={},
        _sidebar_was_visible=False,
        query_one=query_one,
        _debug=lambda *_args: None,
    )
    state = MainScreenState()
    adapter = MainScreenViewAdapter(screen)

    def set_shell_mode(active: bool) -> None:
        screen._shell_mode = active
        state.shell.active = active

    controller = ShellModeController(
        state=state,
        shell_view=adapter,
        focus_view=_FakeFocusView(),
        set_shell_mode=set_shell_mode,
        set_shell_mode_state=lambda active: setattr(screen, "shell_mode_state", active),
        set_fullscreen_terminal=lambda active: setattr(screen, "_fullscreen_terminal", active),
        dismiss_suggestions=lambda: None,
        debug=lambda *_args: None,
    )

    controller.apply(True)

    assert screen._shell_mode is True
    assert chat.display is False
    assert input_bar.display is False
    assert footer.display is False
    assert ("shell_show", None) in calls
    assert ("terminal_focus", None) in calls

    controller.apply(False)

    assert screen._shell_mode is False
    assert chat.display is True
    assert input_bar.display is True
    assert footer.display is True
    assert ("shell_hide", None) in calls
    assert ("session_json_finish", False) in calls
    assert ("input_focus", None) in calls


def test_terminal_exit_events_write_screen_shell_source() -> None:
    state = MainScreenState()
    state.shell.active = True
    controller, _state, _shell_view, _focus_view, shell_mode_states = _make_shell_mode_controller(state=state)

    controller.exit_on_escape()
    assert shell_mode_states == [False]

    shell_mode_states.clear()
    state.shell.active = True
    controller.exit_on_shell_closed()
    assert shell_mode_states == [False]


def test_agent_load_events_lock_input_and_wait_for_switch_event() -> None:
    import asyncio

    from chrys.foundation.events.types import AgentLoadFinished, AgentLoadProgress, AgentLoadStarted

    load_states: list[bool] = []
    pushed: list[object] = []
    status_calls: list[str] = []
    clipboard_dirs: list[object] = []

    class _FakeStatusBar:
        def snapshot(self) -> dict[str, object]:
            return {"visible": False, "flash": None, "status": ""}

        def start_run(self) -> None:
            status_calls.append("start")

        def show(self, msg: MessageRef | str) -> None:
            status_calls.append(_status_text(msg))

    class _FakeApp:
        def push_screen(self, screen: object, _callback=None) -> None:
            pushed.append(screen)

    class _FakeInputBar:
        def set_clipboard_image_dir(self, directory: object) -> None:
            clipboard_dirs.append(directory)

        def focus_input(self) -> None:
            return

    class _FakeStateStore:
        def session_dir(self, session_id: str) -> Path:
            return Path("/sessions") / session_id

    def _query_one(cls):
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return _FakeInputBar()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    def _set_agent_loading(value: bool) -> None:
        load_states.append(value)

    def _debug(*_args: object) -> None:
        return

    screen = SimpleNamespace(
        app=_FakeApp(),
        _state_store=_FakeStateStore(),
        _shell_mode=False,
        _fullscreen_terminal=False,
        _gc_messages=[],
        query_one=_query_one,
        _set_agent_loading=_set_agent_loading,
        _debug=_debug,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None
    handler._agent_load_status_snapshot = None

    asyncio.run(
        handler.on_agent_load_started(
            AgentLoadStarted(
                operation="switch",
                to_profile="Code",
                to_display_name="Code Agent",
                session_id="session-new",
            )
        )
    )
    assert load_states == [True]
    assert status_calls[:2] == ["start", "Switching Agent"]
    assert clipboard_dirs == [Path("/sessions/session-new/attachments/clipboard")]
    assert len(pushed) == 1

    dialog = pushed[0]
    asyncio.run(
        handler.on_agent_load_progress(
            AgentLoadProgress(phase="mcp", message="Connecting MCP server fs", current=0, total=2)
        )
    )
    assert dialog._message == "Connecting MCP server fs"

    asyncio.run(
        handler.on_agent_load_progress(
            AgentLoadProgress(phase="mcp", message="Connecting MCP server fs", current=1, total=2)
        )
    )
    assert dialog._message == "Connecting MCP server fs"
    assert dialog._messages == ["Connecting MCP servers: 1/2"]

    asyncio.run(
        handler.on_agent_load_finished(
            AgentLoadFinished(operation="switch", agent_profile="Code", display_name="Code Agent")
        )
    )
    assert load_states == [True]
    assert dialog._dismiss_pending is False
    assert dialog._message == "Applying agent changes"
    assert dialog._messages[-1] == "Applying agent changes"
    assert len(screen._gc_messages) == 1
    assert isinstance(screen._gc_messages[0], GcReclaimRequested)
    assert screen._gc_messages[0].reason is GcReclaimReason.AGENT_REBUILT
    assert screen._gc_messages[0].prompt is False

    handler.finish_agent_load("Profile switched: QA -> Code")
    assert load_states == [True, False]
    assert dialog._dismiss_pending is True
    assert dialog._messages[-1] == "Profile switched: QA -> Code"
    assert "Applying agent changes" not in dialog._messages


def test_agent_load_dialog_replaces_active_rollback_progress_modal() -> None:
    calls: list[tuple[str, object]] = []
    restore_dialog = object()

    class _RollbackProgress:
        def handoff_to_session_restore(self) -> None:
            calls.append(("rollback-dismiss", self))

    class _FakeApp:
        def push_screen(self, screen: object, _callback=None) -> None:
            calls.append(("restore-push", screen))

    rollback_progress = _RollbackProgress()
    screen = SimpleNamespace(app=_FakeApp(), _shell_mode=False, _fullscreen_terminal=False)
    adapter = MainScreenViewAdapter(screen)  # type: ignore[arg-type]
    adapter._rollback_progress_modal = rollback_progress  # type: ignore[assignment]

    asyncio.run(adapter.push_agent_load_dialog(restore_dialog))

    assert calls == [("rollback-dismiss", rollback_progress), ("restore-push", restore_dialog)]
    assert adapter._rollback_progress_modal is None


def test_rollback_progress_worker_does_not_cancel_prior_handoff_tail() -> None:
    worker_calls: list[tuple[object, bool, str]] = []
    pushed: list[object] = []

    class _FakeScreen:
        app = SimpleNamespace(push_screen=lambda modal, _callback=None: pushed.append(modal))

        def run_worker(self, awaitable: object, *, exclusive: bool, group: str) -> object:
            worker_calls.append((awaitable, exclusive, group))
            return object()

    async def operation() -> None:
        return

    adapter = MainScreenViewAdapter(_FakeScreen())  # type: ignore[arg-type]
    adapter.open_rollback_progress_modal(operation)
    modal = pushed[0]
    assert isinstance(modal, RollbackProgressModal)
    awaitable = modal._run_operation()
    try:
        assert modal._start_worker is not None
        modal._start_worker(awaitable)
    finally:
        awaitable.close()

    assert worker_calls == [(awaitable, False, "rollback-progress")]


def test_nonstartup_agent_load_failure_requests_conservative_idle_reclaim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.foundation.events.types import AgentLoadFailed

    gc_messages: list[object] = []
    screen = SimpleNamespace(_gc_messages=gc_messages, _profile="Code", _debug=lambda *_args: None)
    handler = make_backend_handler(screen)
    monkeypatch.setattr(handler._agent_load(), "on_failed", lambda _event: None)

    asyncio.run(handler.on_agent_load_failed(AgentLoadFailed(operation="startup", agent_profile="Code")))
    assert gc_messages == []

    asyncio.run(handler.on_agent_load_failed(AgentLoadFailed(operation="switch", agent_profile="QA")))
    assert len(gc_messages) == 1
    assert isinstance(gc_messages[0], GcReclaimRequested)
    assert gc_messages[0].reason is GcReclaimReason.AGENT_REBUILD_FAILED
    assert gc_messages[0].prompt is False


def test_image_compression_modal_opens_and_closes() -> None:
    pushed: list[object] = []
    debug_calls: list[tuple[str, str]] = []

    class _FakeApp:
        def push_screen(self, screen: object) -> None:
            pushed.append(screen)

    screen = SimpleNamespace(
        app=_FakeApp(),
        _debug=lambda key, message: debug_calls.append((key, message)),
    )
    handler = make_backend_handler(screen)
    handler._image_compression_dialog = None

    asyncio.run(handler.on_image_attachment_compression_started(ImageAttachmentCompressionStarted(image_count=2)))

    assert len(pushed) == 1
    dialog = pushed[0]
    assert dialog._title == "Preparing Images"
    assert handler._image_compression_dialog is dialog

    asyncio.run(handler.on_image_attachment_compression_finished(ImageAttachmentCompressionFinished(image_count=2)))

    assert handler._image_compression_dialog is None
    assert dialog._dismiss_pending is True
    assert debug_calls == [("ImageCompressionStarted", "2"), ("ImageCompressionFinished", "2")]


def test_finish_agent_load_clears_loading_if_dialog_finish_fails() -> None:
    """A stale successful-load dialog must not prevent input-bar unlock."""
    loading: list[bool] = []

    class _BrokenDialog:
        def finish(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("dialog already gone")

    screen = SimpleNamespace(_set_agent_loading=loading.append)
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = _BrokenDialog()

    handler.finish_agent_load("Session ready: Code")

    assert handler._agent_load_dialog is None
    assert loading == [False]


def test_agent_load_restore_waits_for_session_restored() -> None:
    import asyncio

    from chrys.foundation.events.types import AgentLoadFinished, AgentLoadStarted

    load_states: list[bool] = []
    pushed: list[object] = []
    clipboard_dirs: list[object] = []

    class _FakeStatusBar:
        def snapshot(self) -> dict[str, object]:
            return {"visible": False, "flash": None, "status": ""}

        def start_run(self) -> None:
            return

        def show(self, _msg: str) -> None:
            return

    class _FakeApp:
        def push_screen(self, screen: object, _callback=None) -> None:
            pushed.append(screen)

    class _FakeInputBar:
        def set_clipboard_image_dir(self, directory: object) -> None:
            clipboard_dirs.append(directory)

        def focus_input(self) -> None:
            return

    class _FakeStateStore:
        def session_dir(self, session_id: str) -> Path:
            return Path("/sessions") / session_id

    def _query_one(cls):
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return _FakeInputBar()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        app=_FakeApp(),
        _state_store=_FakeStateStore(),
        _shell_mode=False,
        _fullscreen_terminal=False,
        query_one=_query_one,
        _set_agent_loading=load_states.append,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None
    handler._agent_load_status_snapshot = None

    asyncio.run(
        handler.on_agent_load_started(
            AgentLoadStarted(operation="restore", to_profile="Code", session_id="session-restore")
        )
    )
    dialog = pushed[0]
    assert clipboard_dirs == [Path("/sessions/session-restore/attachments/clipboard")]

    asyncio.run(handler.on_agent_load_finished(AgentLoadFinished(operation="restore", agent_profile="Code")))
    assert load_states == [True]
    assert dialog._message == "Restoring session history"

    handler.finish_agent_load()
    assert load_states == [True, False]
    assert dialog._dismiss_pending is True


def test_do_session_restore_ignores_running_agent() -> None:
    """Direct restore calls should preserve state while a turn is running."""
    begin_calls: list[str] = []
    published: list[object] = []

    async def begin_session_restore_load(session_id: str) -> None:
        begin_calls.append(session_id)

    async def publish(event: object) -> None:
        published.append(event)

    screen = SimpleNamespace(
        _agent_running=True,
        _agent_loading=False,
        _restoring_session=False,
        _events=SimpleNamespace(begin_session_restore_load=begin_session_restore_load),
        _bus=SimpleNamespace(publish=publish),
    )

    asyncio.run(_make_session_handler(screen).do_session_restore("busy"))

    assert screen._restoring_session is False
    assert begin_calls == []
    assert published == []


def test_do_session_restore_can_bypass_loading_guard_for_startup_restore() -> None:
    """Startup --session restore keeps input locked while publishing restore."""
    begin_calls: list[str] = []
    published: list[object] = []

    async def begin_session_restore_load(session_id: str) -> None:
        begin_calls.append(session_id)

    async def publish(event: object) -> None:
        published.append(event)

    screen = SimpleNamespace(
        _agent_running=False,
        _agent_loading=True,
        _restoring_session=False,
        _events=SimpleNamespace(begin_session_restore_load=begin_session_restore_load),
        _bus=SimpleNamespace(publish=publish),
    )

    asyncio.run(_make_session_handler(screen).do_session_restore("startup-session", allow_while_loading=True))

    assert screen._restoring_session is True
    assert begin_calls == ["startup-session"]
    assert len(published) == 1
    assert published[0].session_id == "startup-session"
    assert published[0].apply_saved_model is True


def test_do_session_restore_disables_saved_model_when_startup_model_is_explicit() -> None:
    published: list[object] = []

    async def begin_session_restore_load(_session_id: str) -> None:
        return None

    async def publish(event: object) -> None:
        published.append(event)

    screen = SimpleNamespace(
        _agent_running=False,
        _agent_loading=True,
        _restoring_session=False,
        _apply_saved_model_on_restore=False,
        _events=SimpleNamespace(begin_session_restore_load=begin_session_restore_load),
        _bus=SimpleNamespace(publish=publish),
    )

    asyncio.run(_make_session_handler(screen).do_session_restore("startup-session", allow_while_loading=True))

    assert len(published) == 1
    assert published[0].apply_saved_model is False


def test_do_session_restore_cleans_up_if_loading_ui_fails() -> None:
    """A pre-restore modal failure must not leave the screen stuck restoring."""
    cancel_calls: list[None] = []
    published: list[object] = []
    debug_calls: list[tuple[str, str]] = []

    async def begin_session_restore_load(_session_id: str) -> None:
        raise RuntimeError("push failed")

    async def publish(event: object) -> None:
        published.append(event)

    screen = SimpleNamespace(
        _agent_running=False,
        _agent_loading=False,
        _restoring_session=False,
        _events=SimpleNamespace(
            begin_session_restore_load=begin_session_restore_load,
            cancel_agent_load=lambda: cancel_calls.append(None),
        ),
        _bus=SimpleNamespace(publish=publish),
        _debug=lambda key, msg: debug_calls.append((key, msg)),
    )

    asyncio.run(_make_session_handler(screen).do_session_restore("broken"))

    assert screen._restoring_session is False
    assert cancel_calls == [None]
    assert published == []
    assert debug_calls == [("SessionRestore", "failed to open loading UI: push failed")]


def _resume_screen(
    *,
    latest: object,
    calls: list[tuple[str, object]],
    published: list[object],
    lookup_started: asyncio.Event | None = None,
    lookup_release: asyncio.Event | None = None,
) -> SimpleNamespace:
    async def begin_session_restore_load(session_id: str) -> None:
        calls.append(("lookup_modal" if not session_id else "restore_modal", session_id))

    class _FakeStateStore:
        async def load_latest_session_id(self) -> str | None:
            calls.append(("load_latest", None))
            if lookup_started is not None:
                lookup_started.set()
            if lookup_release is not None:
                await lookup_release.wait()
            if isinstance(latest, BaseException):
                raise latest
            return latest  # type: ignore[return-value]

        async def list_sessions(self) -> list[object]:
            raise AssertionError("resume must not scan list_sessions()")

    async def publish(event: object) -> None:
        published.append(event)

    return SimpleNamespace(
        _agent_running=False,
        _agent_loading=False,
        _restoring_session=False,
        _state_store=_FakeStateStore(),
        _events=SimpleNamespace(
            begin_session_restore_load=begin_session_restore_load,
            cancel_agent_load=lambda: calls.append(("cancel", None)),
        ),
        _bus=SimpleNamespace(publish=publish),
        _debug=lambda key, msg: calls.append(("debug", (key, msg))),
        notify=lambda message, **kwargs: calls.append(("notify", (message, kwargs.get("severity")))),
    )


def test_resume_last_session_opens_modal_before_lookup_and_restores_latest() -> None:
    calls: list[tuple[str, object]] = []
    published: list[object] = []
    screen = _resume_screen(latest="latest-session", calls=calls, published=published)

    asyncio.run(_make_session_handler(screen).resume_last_session())

    assert [name for name, _ in calls] == ["lookup_modal", "load_latest", "restore_modal"]
    assert ("restore_modal", "latest-session") in calls
    assert screen._restoring_session is True
    assert len(published) == 1
    assert published[0].session_id == "latest-session"


def test_resume_last_session_modal_is_open_while_lookup_blocks() -> None:
    calls: list[tuple[str, object]] = []
    published: list[object] = []

    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        screen = _resume_screen(
            latest="slow-session",
            calls=calls,
            published=published,
            lookup_started=started,
            lookup_release=release,
        )
        handler = _make_session_handler(screen)
        task = asyncio.create_task(handler.resume_last_session())
        await started.wait()
        assert calls == [("lookup_modal", ""), ("load_latest", None)]
        assert published == []
        release.set()
        await task

    asyncio.run(scenario())

    assert ("restore_modal", "slow-session") in calls
    assert len(published) == 1


def test_resume_last_session_without_sessions_closes_modal_and_notifies() -> None:
    calls: list[tuple[str, object]] = []
    published: list[object] = []
    screen = _resume_screen(latest=None, calls=calls, published=published)

    asyncio.run(_make_session_handler(screen).resume_last_session())

    assert [name for name, _ in calls] == ["lookup_modal", "load_latest", "cancel", "notify"]
    assert calls[-1][1][1] == "warning"
    assert screen._restoring_session is False
    assert published == []


def test_resume_last_session_lookup_failure_cleans_up() -> None:
    calls: list[tuple[str, object]] = []
    published: list[object] = []
    screen = _resume_screen(latest=RuntimeError("index exploded"), calls=calls, published=published)

    asyncio.run(_make_session_handler(screen).resume_last_session())

    assert ("cancel", None) in calls
    assert ("debug", ("SessionRestore", "failed to look up latest session: index exploded")) in calls
    assert screen._restoring_session is False
    assert screen._agent_loading is False
    assert published == []


def test_resume_last_session_ignores_running_or_loading_agent() -> None:
    calls: list[tuple[str, object]] = []
    published: list[object] = []
    screen = _resume_screen(latest="ignored", calls=calls, published=published)
    screen._agent_running = True

    asyncio.run(_make_session_handler(screen).resume_last_session())

    assert calls == []
    assert published == []


def test_fork_current_session_rejects_empty_session() -> None:
    published: list[object] = []
    notifications: list[tuple[str, str, str]] = []

    async def publish(event: object) -> None:
        published.append(event)

    screen = SimpleNamespace(
        _agent_running=False,
        _agent_loading=False,
        _has_messages=False,
        _bus=SimpleNamespace(publish=publish),
        notify=lambda message, **_kwargs: notifications.append(message),
    )

    asyncio.run(_make_session_handler(screen).fork_current_session())

    assert published == []
    assert notifications == ["Cannot fork an empty session"]


def test_fork_current_session_publishes_session_fork_and_opens_loading_modal(monkeypatch: pytest.MonkeyPatch) -> None:
    from chrys.app.tui.terminal import launcher

    published: list[object] = []
    pushed: list[object] = []
    callbacks: list[object] = []
    loading: list[bool] = []
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setattr(launcher, "can_access_local_desktop", lambda _env=None: True)

    async def publish(event: object) -> None:
        assert screen._agent_loading is True
        published.append(event)

    def query_one(cls: type) -> object:
        if cls.__name__ == "ChatPanel":
            return SimpleNamespace(session_id="session-1")
        raise AssertionError(cls)

    def push_screen(dialog: object, callback=None) -> None:
        pushed.append(dialog)
        callbacks.append(callback)

    def set_agent_loading(value: bool) -> None:
        screen._agent_loading = value
        loading.append(value)

    screen = SimpleNamespace(
        _agent_running=False,
        _agent_loading=False,
        _has_messages=True,
        _bus=SimpleNamespace(publish=publish),
        app=SimpleNamespace(push_screen=push_screen),
        query_one=query_one,
        notify=lambda *_args, **_kwargs: None,
        _set_agent_loading=set_agent_loading,
        _debug=lambda *_args: None,
    )

    asyncio.run(_make_session_handler(screen).fork_current_session())

    assert len(published) == 1
    assert isinstance(published[0], SessionFork)
    assert published[0].session_id == "session-1"
    assert [type(dialog).__name__ for dialog in pushed] == ["ForkSessionDialog"]
    assert pushed[0]._state == "loading"
    assert pushed[0]._show_new_window is True
    assert callbacks[0] is not None
    assert loading == [True]


def _make_clear_screen(bus: EventBus, loading: list[bool]) -> SimpleNamespace:
    def set_agent_loading(value: bool) -> None:
        screen._agent_loading = value
        loading.append(value)

    screen = SimpleNamespace(
        _agent_running=False,
        _agent_loading=False,
        _has_messages=True,
        _creating_new_session=False,
        _bus=bus,
        notify=lambda *_args, **_kwargs: None,
        _set_agent_loading=set_agent_loading,
        _debug=lambda *_args: None,
    )
    return screen


def test_delete_current_and_new_publishes_session_clear_with_input_blocked() -> None:
    """/clear is ONE backend op: input is blocked and the new-session flag set before it is published."""
    bus = EventBus()
    loading: list[bool] = []
    seen: list[tuple[str, bool, bool]] = []
    screen = _make_clear_screen(bus, loading)

    async def fake_backend(event: SessionClear) -> None:
        seen.append((event.session_id, screen._agent_loading, screen._creating_new_session))
        await bus.publish(SessionDeleted(session_id=event.session_id))

    async def run() -> None:
        await bus.subscribe(SessionClear, fake_backend)
        await _make_session_handler(screen).delete_current_and_new("session-1")

    asyncio.run(run())

    assert seen == [("session-1", True, True)]
    # Acknowledged delete: the fresh session's own load/ready events own the
    # flags from here, so nothing is reset optimistically.
    assert loading == [True]
    assert screen._creating_new_session is True


def test_delete_current_and_new_without_ack_resets_new_session_state() -> None:
    """No ``SessionDeleted`` acknowledgement means nothing was deleted: undo the optimistic UI state."""
    bus = EventBus()
    loading: list[bool] = []
    screen = _make_clear_screen(bus, loading)

    async def failing_backend(event: SessionClear) -> None:
        await bus.publish(Error(code="session_clear_failed", message="Failed to delete session: boom"))

    async def run() -> None:
        await bus.subscribe(SessionClear, failing_backend)
        await _make_session_handler(screen).delete_current_and_new("session-1")

    asyncio.run(run())

    assert loading == [True, False]
    assert screen._creating_new_session is False


def test_delete_current_and_new_is_ignored_while_busy() -> None:
    """Running, loading, or a submit still being admitted all veto the worker (re-checked, not just preflight)."""
    published: list[object] = []
    loading_calls: list[bool] = []

    async def publish(event: object) -> None:
        published.append(event)

    for running, loading, submitting in ((True, False, False), (False, True, False), (False, False, True)):
        screen = SimpleNamespace(
            _agent_running=running,
            _agent_loading=loading,
            _pending_user_submit_active=submitting,
            _has_messages=True,
            _bus=SimpleNamespace(publish=publish),
            _set_agent_loading=loading_calls.append,
        )
        handler = _make_session_handler(screen)
        asyncio.run(handler.delete_current_and_new("session-1"))
        assert handler.creating_new_session is False

    assert published == []
    assert loading_calls == []


def test_session_clear_error_toasts_and_keeps_session_without_retry_mode() -> None:
    """``session_clear_failed`` is a toast + state reset, never the generic chat error / retry mode."""
    flashes: list[tuple[str, bool]] = []
    notifications: list[tuple[str, str, str]] = []
    unlocked: list[None] = []
    running: list[bool] = []
    loading: list[bool] = []
    errors_added: list[str] = []

    class _FakeStatusBar:
        def flash(self, message: str, *, error: bool = False) -> None:
            flashes.append((message, error))

    class _FakeInputBar:
        locked = True
        retry_mode = False

        def unlock_and_keep(self) -> None:
            self.locked = False
            unlocked.append(None)

    class _FakeChatPanel:
        async def add_error(self, message: str) -> None:
            errors_added.append(message)

    input_bar = _FakeInputBar()

    def query_one(cls: type) -> object:
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ChatPanel":
            return _FakeChatPanel()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    def notify(message: str, *, title: str, severity: str, **_kwargs: object) -> None:
        notifications.append((title, severity, message))

    def set_agent_loading(value: bool) -> None:
        screen._agent_loading = value
        loading.append(value)

    defaults = _pending_submit_defaults()
    screen = SimpleNamespace(
        _restoring_session=False,
        _agent_loading=True,
        _creating_new_session=True,
        **defaults,
        _set_agent_running=running.append,
        _set_agent_loading=set_agent_loading,
        query_one=query_one,
        notify=notify,
        _debug=lambda *_args: None,
    )
    screen._sessions = _make_session_handler(screen)
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None

    asyncio.run(handler.on_error(Error(code="session_clear_failed", message="Failed to delete session: boom")))

    assert notifications == [("Clear Session", "error", "The current session was kept: Failed to delete session: boom")]
    assert loading == [False]
    assert screen._creating_new_session is False
    assert unlocked == [None]
    assert running == []
    assert flashes == []
    assert errors_added == []
    assert input_bar.retry_mode is False


def test_session_fork_dialog_hides_new_window_over_ssh(monkeypatch: pytest.MonkeyPatch) -> None:
    pushed: list[object] = []
    loading: list[bool] = []
    monkeypatch.setenv("SSH_CONNECTION", "127.0.0.1 50000 127.0.0.1 22")

    def push_screen(dialog: object, callback=None) -> None:
        del callback
        pushed.append(dialog)

    screen = SimpleNamespace(
        app=SimpleNamespace(push_screen=push_screen),
        _set_agent_loading=loading.append,
    )
    handler = _make_session_handler(screen)

    asyncio.run(handler._open_session_fork_dialog("session-1"))

    assert pushed[0]._show_new_window is False
    assert loading == [True]


def test_on_session_forked_routes_dialog_results(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher_calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "chrys.app.tui.terminal.launcher.launch_new_chrys_window",
        lambda session_id, *, cwd=None: launcher_calls.append((session_id, cwd)),
    )

    def run_case(
        result: str, new_session_id: str
    ) -> tuple[list[str], list[str], list[tuple[str, str, str]], list[bool]]:
        pushes: list[object] = []
        callbacks: list[object] = []
        flashes: list[str] = []
        restores: list[str] = []
        notifications: list[tuple[str, str, str]] = []
        loading: list[bool] = []
        debug_calls: list[tuple[str, str]] = []

        class _FakeStatusBar:
            def flash(self, message: str) -> None:
                flashes.append(message)

        def query_one(cls: type) -> object:
            if cls.__name__ == "ChatPanel":
                return SimpleNamespace(session_id="current-session")
            if cls.__name__ == "StatusBar":
                return _FakeStatusBar()
            raise AssertionError(cls)

        def push_screen(dialog: object, callback=None) -> None:
            pushes.append(dialog)
            callbacks.append(callback)

        def set_agent_loading(value: bool) -> None:
            loading.append(value)

        screen = SimpleNamespace(
            app=SimpleNamespace(push_screen=push_screen),
            query_one=query_one,
            _do_session_restore=restores.append,
            notify=lambda message, *, title, severity="information", **_kwargs: notifications.append(
                (title, severity, message)
            ),
            _set_agent_loading=set_agent_loading,
            _workspace_cwd=lambda: "/workspace",
            _debug=lambda key, value: debug_calls.append((key, value)),
        )
        handler = _make_session_handler(screen)

        asyncio.run(handler._open_session_fork_dialog("current-session"))
        assert pushes[0]._state == "loading"
        asyncio.run(
            handler.on_session_forked(
                SessionForked(
                    session_id="current-session",
                    parent_session_id="current-session",
                    new_session_id=new_session_id,
                )
            )
        )
        assert pushes[0]._state == "success"
        assert loading == [True, False]
        callbacks[0](result)
        assert debug_calls == [("SessionForked", session_short_id(new_session_id))]
        return flashes, restores, notifications, loading

    flashes, restores, notifications, loading = run_case("stay", "fork-stay")
    assert [_status_text(message) for message in flashes] == [f"Fork created: {session_short_id('fork-stay')}"]
    assert restores == []
    assert notifications == [("Fork", "information", f"Created fork {session_short_id('fork-stay')}")]
    assert loading == [True, False, False]

    flashes, restores, notifications, loading = run_case("switch", "fork-switch")
    assert flashes == []
    assert restores == ["fork-switch"]
    assert notifications == [("Fork", "information", f"Created fork {session_short_id('fork-switch')}")]
    assert loading == [True, False, False]

    flashes, restores, notifications, loading = run_case("new_window", "fork-window")
    assert [_status_text(message) for message in flashes] == [f"Opened fork: {session_short_id('fork-window')}"]
    assert restores == []
    assert notifications == [("Fork", "information", f"Created fork {session_short_id('fork-window')}")]
    assert loading == [True, False, False]
    assert launcher_calls == [("fork-window", "/workspace")]


def test_on_session_forked_new_window_launcher_failure_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    pushes: list[object] = []
    callbacks: list[object] = []
    notifications: list[tuple[str, str, str]] = []
    loading: list[bool] = []

    class _FakeStatusBar:
        def flash(self, _message: str) -> None:
            raise AssertionError("status flash should not run on launcher failure")

    def query_one(cls: type) -> object:
        if cls.__name__ == "ChatPanel":
            return SimpleNamespace(session_id="current-session")
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        raise AssertionError(cls)

    def push_screen(dialog: object, callback=None) -> None:
        pushes.append(dialog)
        callbacks.append(callback)

    monkeypatch.setenv("SSH_TTY", "/dev/pts/1")
    from chrys.app.tui.terminal import launcher

    with pytest.raises(launcher.TerminalLaunchError) as captured:
        launcher.launch_new_chrys_window("probe")

    def raise_launch_error(_session_id: str, *, cwd: str | None = None) -> None:
        del cwd
        raise captured.value

    monkeypatch.setattr("chrys.app.tui.terminal.launcher.launch_new_chrys_window", raise_launch_error)
    screen = SimpleNamespace(
        app=SimpleNamespace(push_screen=push_screen),
        query_one=query_one,
        _do_session_restore=lambda _session_id: None,
        notify=lambda message, *, title, severity="information", **_kwargs: notifications.append(
            (title, severity, message)
        ),
        _set_agent_loading=loading.append,
        _workspace_cwd=lambda: "/workspace",
        _debug=lambda *_args: None,
    )
    handler = _make_session_handler(screen, locale_controller=LocaleController(Settings(locale="zh-Hans")))

    asyncio.run(handler._open_session_fork_dialog("current-session"))
    asyncio.run(
        handler.on_session_forked(
            SessionForked(
                session_id="current-session",
                parent_session_id="current-session",
                new_session_id="fork-window",
            )
        )
    )
    callbacks[0]("new_window")

    assert notifications == [
        ("Fork", "information", f"Created fork {session_short_id('fork-window')}"),
        ("Fork", "warning", f"当前环境无法打开新的 {APP_DISPLAY_NAME} 窗口。"),
    ]
    assert loading == [True, False, False]


def test_fork_session_dialog_only_allows_dismiss_after_result() -> None:
    from chrys.app.tui.screens.dialogs.fork_session import ForkSessionDialog

    dialog = ForkSessionDialog()
    assert dialog._allow_click_outside_dismiss() is False
    assert dialog._default_dismiss_result() is None

    dialog.set_success("fork-1234")
    assert dialog._allow_click_outside_dismiss() is True
    assert dialog._default_dismiss_result() == "stay"

    dialog = ForkSessionDialog()
    dialog.set_error("Fork failed")
    assert dialog._allow_click_outside_dismiss() is True
    assert dialog._default_dismiss_result() is None


@pytest.mark.asyncio
async def test_fork_session_dialog_omits_new_window_button_when_unavailable() -> None:
    from textual.app import App

    from chrys.app.tui.screens.dialogs.fork_session import ForkSessionDialog

    app = App()
    dialog = ForkSessionDialog("fork-1234", show_new_window=False)

    async with app.run_test():
        await app.push_screen(dialog)

        assert list(dialog.query("#fork-session-new-window")) == []
        assert list(dialog.query("#fork-session-switch"))
        assert list(dialog.query("#fork-session-stay"))


def test_fork_session_focus_skips_buttons_in_hidden_groups() -> None:
    from chrys.app.tui.screens.dialogs.fork_session import ForkSessionDialog

    class _FakeParent:
        def __init__(self, *, display: bool) -> None:
            self.display = display

    class _FakeButton:
        def __init__(self, *, parent: _FakeParent, has_focus: bool = False) -> None:
            self.display = True
            self.parent = parent
            self.has_focus = has_focus
            self.focused = False

        def focus(self) -> None:
            self.focused = True

    visible_group = _FakeParent(display=True)
    hidden_group = _FakeParent(display=False)
    switch = _FakeButton(parent=visible_group)
    stay = _FakeButton(parent=visible_group, has_focus=True)
    ok = _FakeButton(parent=hidden_group)
    dialog = SimpleNamespace(
        query=lambda _cls: [switch, stay, ok],
        _button_is_visible=ForkSessionDialog._button_is_visible,
    )

    ForkSessionDialog._focus_relative(dialog, 1)

    assert switch.focused is True
    assert ok.focused is False
    assert ForkSessionDialog._button_is_visible(ok) is False


def test_session_fork_error_updates_pending_dialog() -> None:
    pushes: list[object] = []
    callbacks: list[object] = []
    flashes: list[tuple[str, bool]] = []
    notifications: list[tuple[str, str, str]] = []
    loading: list[bool] = []
    unlocked: list[None] = []

    class _FakeStatusBar:
        def flash(self, message: str, *, error: bool = False) -> None:
            flashes.append((message, error))

    class _FakeInputBar:
        locked = True

        def unlock_and_keep(self) -> None:
            self.locked = False
            unlocked.append(None)

    input_bar = _FakeInputBar()

    def query_one(cls: type) -> object:
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return input_bar
        raise AssertionError(cls)

    def push_screen(dialog: object, callback=None) -> None:
        pushes.append(dialog)
        callbacks.append(callback)

    def set_agent_loading(value: bool) -> None:
        loading.append(value)

    screen = SimpleNamespace(
        app=SimpleNamespace(push_screen=push_screen),
        query_one=query_one,
        notify=lambda message, *, title, severity, **_kwargs: notifications.append((title, severity, message)),
        _set_agent_loading=set_agent_loading,
    )
    handler = _make_session_handler(screen)

    asyncio.run(handler._open_session_fork_dialog("current-session"))
    assert pushes[0]._state == "loading"

    handler.on_session_fork_error(
        Error(session_id="current-session", code="session_fork_busy", message="Session is busy."),
        message="Session is busy.",
        severity="warning",
    )

    assert pushes[0]._state == "error"
    assert loading == [True, False]
    assert [(_status_text(message), error) for message, error in flashes] == [("Fork: Session is busy.", False)]
    assert notifications == [("Fork", "warning", "Session is busy.")]
    assert unlocked == [None]

    callbacks[0](None)
    assert loading == [True, False, False]


def test_load_file_edit_snapshots_uses_loaded_state_not_primary_session_file(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path / "sessions")
    asyncio.run(store.save_session("sid", {"messages": [], "compressed_msgs": []}))
    session_dir = store.session_dir("sid")
    target = tmp_path / "work.py"
    target.write_text("before", encoding="utf-8")
    tracker = MutationTracker(SnapshotStore(session_dir))
    tracker.start_turn(1)
    mutation = tracker.record(str(target), MutationOp.MODIFY, MutationSource.EDIT_FILE, "runtime-call")
    assert mutation is not None
    target.write_text("after", encoding="utf-8")
    tracker.record_after(mutation)
    recovered_state = {"chrys_mutations": tracker.serialize()}
    messages = [
        {
            "role": "assistant",
            "contents": [{"type": "function_call", "name": "edit_file", "call_id": "fw-call"}],
        }
    ]
    screen = SimpleNamespace(_state_store=store)

    snapshots = _make_session_handler(screen).load_file_edit_snapshots("sid", messages, recovered_state)

    assert snapshots == {"fw-call": [("before", "after")]}


def test_load_file_edit_snapshots_skips_marker_carried_file_calls(tmp_path: Path) -> None:
    # The tracker records snapshots only for executed calls; a marker-carried
    # file call is chrome that replay never renders, so it must not shift the
    # positional call↔snapshot zip away from the visible real call.
    store = JsonFileStateStore(tmp_path / "sessions")
    asyncio.run(store.save_session("sid", {"messages": [], "compressed_msgs": []}))
    session_dir = store.session_dir("sid")
    target = tmp_path / "work.py"
    target.write_text("before", encoding="utf-8")
    tracker = MutationTracker(SnapshotStore(session_dir))
    tracker.start_turn(1)
    mutation = tracker.record(str(target), MutationOp.MODIFY, MutationSource.EDIT_FILE, "runtime-call")
    assert mutation is not None
    target.write_text("after", encoding="utf-8")
    tracker.record_after(mutation)
    recovered_state = {"chrys_mutations": tracker.serialize()}
    messages = [
        {
            "role": "assistant",
            "contents": [{"type": "function_call", "name": "edit_file", "call_id": "hidden"}],
            "additional_properties": {HistoryMarkerKind.KEY: HistoryMarkerKind.INTERRUPTED},
        },
        {
            "role": "assistant",
            "contents": [{"type": "function_call", "name": "edit_file", "call_id": "fw-call"}],
        },
    ]
    screen = SimpleNamespace(_state_store=store)

    snapshots = _make_session_handler(screen).load_file_edit_snapshots("sid", messages, recovered_state)

    assert snapshots == {"fw-call": [("before", "after")]}


def test_load_file_edit_snapshots_externalizes_large_snapshot(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path / "sessions")
    asyncio.run(store.save_session("sid", {"messages": [], "compressed_msgs": []}))
    session_dir = store.session_dir("sid")
    target = tmp_path / "large.py"
    target.write_text("before", encoding="utf-8")
    tracker = MutationTracker(SnapshotStore(session_dir))
    tracker.start_turn(1)
    mutation = tracker.record(str(target), MutationOp.MODIFY, MutationSource.EDIT_FILE, "runtime-call")
    assert mutation is not None
    large_after = "x" * (file_snapshot_inline_char_limit() + 1)
    target.write_text(large_after, encoding="utf-8")
    tracker.record_after(mutation)
    recovered_state = {"chrys_mutations": tracker.serialize()}
    messages = [
        {
            "role": "assistant",
            "contents": [{"type": "function_call", "name": "edit_file", "call_id": "fw-call"}],
        }
    ]
    screen = SimpleNamespace(_state_store=store)

    snapshots = _make_session_handler(screen).load_file_edit_snapshots("sid", messages, recovered_state)

    payload = snapshots["fw-call"][0]
    assert isinstance(payload, FileSnapshotRef)
    assert payload.resolve() == ("before", large_after)


def test_load_file_edit_snapshots_truthy_unhashable_id_discards_all_buckets(tmp_path: Path) -> None:
    """A truthy unhashable snapshot call id aborts the whole load fail-soft:
    every bucket is discarded, including valid ones."""
    store = JsonFileStateStore(tmp_path / "sessions")
    asyncio.run(store.save_session("sid", {"messages": [], "compressed_msgs": []}))
    session_dir = store.session_dir("sid")
    tracker = MutationTracker(SnapshotStore(session_dir))
    tracker.start_turn(1)
    for index in range(2):
        target = tmp_path / f"work{index}.py"
        target.write_text("before", encoding="utf-8")
        mutation = tracker.record(str(target), MutationOp.MODIFY, MutationSource.EDIT_FILE, f"runtime-{index}")
        assert mutation is not None
        target.write_text("after", encoding="utf-8")
        tracker.record_after(mutation)
    recovered_state = {"chrys_mutations": tracker.serialize()}
    messages = [
        {
            "role": "assistant",
            "contents": [
                {"type": "function_call", "name": "edit_file", "call_id": "fw-call"},
                {"type": "function_call", "name": "edit_file", "call_id": ["x"]},
            ],
        }
    ]
    screen = SimpleNamespace(_state_store=store)

    snapshots = _make_session_handler(screen).load_file_edit_snapshots("sid", messages, recovered_state)

    assert snapshots == {}


def test_load_file_edit_snapshots_falsy_unhashable_id_skipped_buckets_kept(tmp_path: Path) -> None:
    """A falsy unhashable snapshot call id fails the truthiness gate and is
    skipped; valid buckets survive."""
    store = JsonFileStateStore(tmp_path / "sessions")
    asyncio.run(store.save_session("sid", {"messages": [], "compressed_msgs": []}))
    session_dir = store.session_dir("sid")
    target = tmp_path / "work.py"
    target.write_text("before", encoding="utf-8")
    tracker = MutationTracker(SnapshotStore(session_dir))
    tracker.start_turn(1)
    mutation = tracker.record(str(target), MutationOp.MODIFY, MutationSource.EDIT_FILE, "runtime-call")
    assert mutation is not None
    target.write_text("after", encoding="utf-8")
    tracker.record_after(mutation)
    recovered_state = {"chrys_mutations": tracker.serialize()}
    messages = [
        {
            "role": "assistant",
            "contents": [
                {"type": "function_call", "name": "edit_file", "call_id": []},
                {"type": "function_call", "name": "edit_file", "call_id": "fw-call"},
            ],
        }
    ]
    screen = SimpleNamespace(_state_store=store)

    snapshots = _make_session_handler(screen).load_file_edit_snapshots("sid", messages, recovered_state)

    assert snapshots == {"fw-call": [("before", "after")]}


def test_begin_session_restore_load_shows_availability_check() -> None:
    import asyncio

    load_states: list[bool] = []
    pushed: list[object] = []
    status_calls: list[str] = []
    clipboard_dirs: list[object] = []

    class _FakeStatusBar:
        def snapshot(self) -> dict[str, object]:
            return {"visible": False, "flash": None, "status": ""}

        def start_run(self) -> None:
            status_calls.append("start")

        def show(self, msg: MessageRef | str) -> None:
            status_calls.append(_status_text(msg))

    class _FakeApp:
        def push_screen(self, screen: object, _callback=None) -> None:
            pushed.append(screen)

    class _FakeInputBar:
        def set_clipboard_image_dir(self, directory: object) -> None:
            clipboard_dirs.append(directory)

        def focus_input(self) -> None:
            return

    class _FakeStateStore:
        def session_dir(self, session_id: str) -> Path:
            return Path("/sessions") / session_id

    def _query_one(cls):
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return _FakeInputBar()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        app=_FakeApp(),
        _state_store=_FakeStateStore(),
        _shell_mode=False,
        _fullscreen_terminal=False,
        query_one=_query_one,
        _set_agent_loading=load_states.append,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None
    handler._agent_load_status_snapshot = None

    asyncio.run(handler.begin_session_restore_load("40d9a048-3e08-4cff-a0e0-ce8c09d3e011"))

    assert load_states == [True]
    assert status_calls[:2] == ["start", "Restoring Session"]
    assert clipboard_dirs == []
    assert len(pushed) == 1
    dialog = pushed[0]
    assert dialog._message == "Checking session availability"
    assert dialog._messages == ["Checking session availability"]


def test_begin_session_restore_load_lookup_then_resolved_id_reuses_dialog() -> None:
    """``/resume`` opens the modal with an empty id, then fills in the resolved id."""
    import asyncio

    pushed: list[object] = []
    snapshots: list[None] = []

    class _FakeStatusBar:
        def snapshot(self) -> dict[str, object]:
            snapshots.append(None)
            return {"visible": False, "flash": None, "status": ""}

        def start_run(self) -> None:
            return

        def show(self, _msg: object) -> None:
            return

    class _FakeApp:
        def push_screen(self, screen: object, _callback=None) -> None:
            pushed.append(screen)

    class _FakeInputBar:
        def set_clipboard_image_dir(self, directory: object) -> None:
            raise AssertionError(f"restore must not touch clipboard dir: {directory}")

        def focus_input(self) -> None:
            return

    def _query_one(cls):
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return _FakeInputBar()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        app=_FakeApp(),
        _state_store=None,
        _shell_mode=False,
        _fullscreen_terminal=False,
        query_one=_query_one,
        _set_agent_loading=lambda _value: None,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None
    handler._agent_load_status_snapshot = None

    asyncio.run(handler.begin_session_restore_load(""))
    assert len(pushed) == 1
    dialog = pushed[0]
    assert dialog._subtitle == ""
    assert dialog._message == "Checking session availability"

    asyncio.run(handler.begin_session_restore_load("40d9a048-3e08-4cff-a0e0-ce8c09d3e011"))

    assert pushed == [dialog]
    assert handler._agent_load_dialog is dialog
    assert dialog._subtitle == "40d9a0483e08"
    assert dialog._message == "Checking session availability"
    assert snapshots == [None]


def test_restore_agent_load_started_reuses_availability_dialog() -> None:
    import asyncio

    from chrys.foundation.events.types import AgentLoadStarted

    pushed: list[object] = []
    clipboard_dirs: list[object] = []

    class _FakeStatusBar:
        def snapshot(self) -> dict[str, object]:
            return {"visible": False, "flash": None, "status": ""}

        def start_run(self) -> None:
            return

        def show(self, _msg: str) -> None:
            return

    class _FakeApp:
        def push_screen(self, screen: object, _callback=None) -> None:
            pushed.append(screen)

    class _FakeInputBar:
        def set_clipboard_image_dir(self, directory: object) -> None:
            clipboard_dirs.append(directory)

        def focus_input(self) -> None:
            return

    class _FakeStateStore:
        def session_dir(self, session_id: str) -> Path:
            return Path("/sessions") / session_id

    def _query_one(cls):
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return _FakeInputBar()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        app=_FakeApp(),
        _state_store=_FakeStateStore(),
        _shell_mode=False,
        _fullscreen_terminal=False,
        query_one=_query_one,
        _set_agent_loading=lambda _value: None,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None
    handler._agent_load_status_snapshot = None

    asyncio.run(handler.begin_session_restore_load("restore"))
    dialog = pushed[0]
    asyncio.run(
        handler.on_agent_load_started(AgentLoadStarted(operation="restore", to_profile="Code", session_id="restore"))
    )

    assert pushed == [dialog]
    assert clipboard_dirs == [Path("/sessions/restore/attachments/clipboard")]
    assert handler._agent_load_dialog is dialog
    assert dialog._message == "Preparing agent"
    assert dialog._messages == ["Session availability checked"]


def test_agent_load_restore_final_message_replaces_pending_message() -> None:
    import asyncio

    from chrys.foundation.events.types import AgentLoadFinished, AgentLoadStarted

    load_states: list[bool] = []
    pushed: list[object] = []
    clipboard_dirs: list[object] = []

    class _FakeStatusBar:
        def snapshot(self) -> dict[str, object]:
            return {"visible": False, "flash": None, "status": ""}

        def start_run(self) -> None:
            return

        def show(self, _msg: str) -> None:
            return

    class _FakeApp:
        def push_screen(self, screen: object, _callback=None) -> None:
            pushed.append(screen)

    class _FakeInputBar:
        def set_clipboard_image_dir(self, directory: object) -> None:
            clipboard_dirs.append(directory)

        def focus_input(self) -> None:
            return

    class _FakeStateStore:
        def session_dir(self, session_id: str) -> Path:
            return Path("/sessions") / session_id

    def _query_one(cls):
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return _FakeInputBar()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        app=_FakeApp(),
        _state_store=_FakeStateStore(),
        _shell_mode=False,
        _fullscreen_terminal=False,
        query_one=_query_one,
        _set_agent_loading=load_states.append,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None
    handler._agent_load_status_snapshot = None

    asyncio.run(
        handler.on_agent_load_started(AgentLoadStarted(operation="restore", to_profile="Code", session_id="restore"))
    )
    dialog = pushed[0]
    assert clipboard_dirs == [Path("/sessions/restore/attachments/clipboard")]

    asyncio.run(handler.on_agent_load_finished(AgentLoadFinished(operation="restore", agent_profile="Code")))
    handler.finish_agent_load("Session restored: abc12345")

    assert dialog._messages[-1] == "Session restored: abc12345"
    assert "Restoring session history" not in dialog._messages


def test_profile_switched_finishes_agent_load_after_final_event() -> None:
    import asyncio

    finish_messages: list[str] = []
    flashes: list[str] = []
    status_clears: list[None] = []
    tool_trails: list[StatusTrail] = []
    welcome_updates: list[tuple[str, str]] = []

    class _FakeEvents:
        def format_tool_info(
            self,
            tool_names: list[str],
            skill_names: list[str],
            *,
            memory_files: list[str] | None = None,
            runtime_details: AgentRuntimeDetails | None = None,
        ) -> str:
            del tool_names, skill_names, memory_files, runtime_details
            return "tools"

        def get_profile_description(self, _profile_name: str) -> str:
            return "description"

        def finish_agent_load(self, message: MessageRef | str = "") -> None:
            finish_messages.append(_status_text(message))

    class _FakePanel:
        def set_profile(self, _profile: str) -> None:
            return

        def update_welcome(self, *, profile: str = "", cwd: str = "") -> None:
            welcome_updates.append((profile, cwd))

    class _FakeInputBar:
        def set_clipboard_image_dir(self, _directory: object) -> None:
            return

    class _FakeStatusBar:
        def set_profile(self, _profile: str, *, description: str = "") -> None:
            return

        def set_tool_info(self, trail: StatusTrail) -> None:
            tool_trails.append(trail)

        def clear_status(self) -> None:
            status_clears.append(None)

        def flash(self, message: str, **_kwargs: object) -> None:
            flashes.append(message)

    panel = _FakePanel()
    input_bar = _FakeInputBar()
    status = _FakeStatusBar()

    def _query_one(cls):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "StatusBar":
            return status
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _events=_FakeEvents(),
        _has_messages=False,
        _profile="Code Agent",
        _gc_messages=[],
        _profile_switch_from=None,
        _profile_switch_to=None,
        _profile_switch_seq=0,
        query_one=_query_one,
        _workspace_cwd=lambda: "/repo/current",
        _update_subtitle=lambda: None,
        _debug=lambda *_args: None,
    )
    handler = _make_session_handler(screen)
    runtime_details = AgentRuntimeDetails(
        hook_sources=[
            RuntimeHookSourceDetails(
                scope="global",
                hooks=[RuntimeHookDetails(id="notify", enabled=True)],
            )
        ]
    )

    asyncio.run(
        handler.on_profile_switched(
            ProfileSwitched(
                from_profile="QA",
                to_profile="Code",
                from_display_name="QA Agent",
                to_display_name="Code Agent",
                runtime_details=runtime_details,
            )
        )
    )

    assert flashes == []
    assert status_clears == [None]
    assert finish_messages == ["Profile switched: QA Agent -> Code Agent"]
    assert [_status_trail(trail) for trail in tool_trails] == ["1 hook"]
    assert welcome_updates == [("Code Agent", "/repo/current")]
    assert len(screen._gc_messages) == 1
    assert isinstance(screen._gc_messages[0], GcAbsorbRequested)
    assert screen._gc_messages[0].reason is GcAbsorbReason.PROFILE_UI_UPDATED
    assert screen._gc_messages[0].terminal_boundary is False


@pytest.mark.parametrize(
    ("selection_source", "expected"),
    [("active", "new-model"), ("agent", "old-model")],
)
def test_profile_switched_syncs_model_cache_only_for_active_selection(
    selection_source: Literal["active", "agent"],
    expected: str,
) -> None:
    tool_trails: list[StatusTrail] = []

    class _FakeEvents:
        def format_tool_info(
            self,
            tool_names: list[str],
            skill_names: list[str],
            *,
            memory_files: list[str] | None = None,
            runtime_details: AgentRuntimeDetails | None = None,
        ) -> str:
            del tool_names, skill_names, memory_files, runtime_details
            return "tools"

        def finish_agent_load(self, _message: MessageRef | str = "") -> None:
            return

    class _FakeStatusBar:
        def set_tool_info(self, trail: StatusTrail) -> None:
            tool_trails.append(trail)

        def clear_status(self) -> None:
            return

        def flash(self, _message: str, **_kwargs: object) -> None:
            return

    def query_one(cls: type):
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _events=_FakeEvents(),
        _active_model_profile_id="old-model",
        _profile="Code",
        _gc_messages=[],
        query_one=query_one,
        _debug=lambda *_args: None,
    )
    handler = _make_session_handler(screen)
    details = AgentRuntimeDetails(
        model=RuntimeModelDetails(profile_id="new-model", selection_source=selection_source),
        hook_sources=[
            RuntimeHookSourceDetails(
                scope="project",
                hooks=[RuntimeHookDetails(id="guard", enabled=True)],
            )
        ],
    )

    asyncio.run(
        handler.on_profile_switched(
            ProfileSwitched(
                from_profile="Code",
                to_profile="Code",
                runtime_details=details,
            )
        )
    )

    assert handler._services.active_model_profile_id == expected
    assert screen._active_model_profile_id == expected
    assert [_status_trail(trail) for trail in tool_trails] == ["1 hook"]


def test_session_ready_during_restore_refreshes_memory_file_status() -> None:
    """Restore-specific SessionReady handling must keep loaded memory metadata.

    During restore the chat panel is rebuilt by SessionRestored, but the
    SessionReady event is still the only event carrying auto-loaded memory
    files.  Dropping it made restored AGENTS.md loads invisible until /chdir
    forced a later ProfileSwitched event.
    """

    tool_info: dict[str, str] = {}
    flash_calls: list[str] = []

    class _FakePanel:
        def set_profile(self, _profile: str) -> None:
            return

        def set_tool_kinds(self, _tool_kinds: dict[str, str]) -> None:
            return

        def update_welcome(self, *_args, **_kwargs) -> None:
            raise AssertionError("restore SessionReady must not rebuild chat panel")

        def set_session_id(self, _session_id: str) -> None:
            raise AssertionError("restore SessionReady must not update chat session id")

    class _FakeInputBar:
        def set_clipboard_image_dir(self, _directory: object) -> None:
            return

    class _FakeStatusBar:
        def set_profile(self, _profile: str, *, description: str = "") -> None:
            return

        def set_tool_info(self, trail: str) -> None:
            tool_info["trail"] = trail

        def flash(self, text: str, **_kwargs) -> None:
            flash_calls.append(text)

    panel = _FakePanel()
    input_bar = _FakeInputBar()
    status = _FakeStatusBar()

    def _query_one(cls):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "StatusBar":
            return status
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _agent_registry=None,
        _model_registry=None,
        _active_model_profile_id="old-model",
        _creating_new_session=False,
        _restoring_session=True,
        _profile="",
        _state_store=None,
        query_one=_query_one,
        _update_subtitle=lambda: None,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)

    asyncio.run(
        handler.on_session_ready(
            SessionReady(
                agent_profile="Code",
                display_name="Code Agent",
                session_id="session-1",
                memory_files=["AGENTS.md"],
                runtime_details=AgentRuntimeDetails(
                    model=RuntimeModelDetails(profile_id="ready-model", selection_source="active"),
                    hook_sources=[
                        RuntimeHookSourceDetails(
                            scope="global",
                            hooks=[RuntimeHookDetails(id="notify", enabled=True)],
                        )
                    ],
                ),
            )
        )
    )

    assert "1 file" in _status_trail(tool_info["trail"])
    assert "1 hook" in _status_trail(tool_info["trail"])
    assert "tooltip" not in tool_info
    assert flash_calls == []
    assert screen._active_model_profile_id == "ready-model"


def _run_existing_session_ready(
    *,
    current_max_context_tokens: int,
    event_max_context_tokens: int,
) -> tuple[SimpleNamespace, list[tuple[str, object]], ContextUsageState]:
    calls: list[tuple[str, object]] = []
    initial_context_usage = ContextUsageState.with_window(
        used_tokens=19_635,
        max_context_tokens=current_max_context_tokens,
        total_session_tokens=253_535,
        total_session_input_tokens=120_000,
        total_session_output_tokens=83_535,
        total_session_cache_hit_tokens=50_000,
    )

    class _FakePanel:
        border_subtitle = None

        def set_profile(self, profile: str) -> None:
            calls.append(("profile", profile))

        def set_tool_kinds(self, _tool_kinds: dict[str, str]) -> None:
            return

        def update_welcome(self, *, profile: str = "", cwd: str = "") -> None:
            calls.append(("welcome", (profile, cwd)))

        def set_session_id(self, session_id: str) -> None:
            calls.append(("session_id", session_id))

        def set_workspace_cwd(self, cwd: str) -> None:
            calls.append(("workspace_cwd", cwd))
            self.border_subtitle = Text(cwd)

    class _FakeInputBar:
        def set_clipboard_image_dir(self, directory: object) -> None:
            calls.append(("clipboard_image_dir", directory))

        def set_paste_cwd(self, cwd: str) -> None:
            calls.append(("paste_cwd", cwd))

    class _FakeStatusBar:
        def set_profile(self, profile: str, *, description: str = "") -> None:
            calls.append(("status_profile", (profile, description)))

        def set_tool_info(self, trail: str) -> None:
            calls.append(("tool_info", trail))

        def clear_status(self) -> None:
            calls.append(("clear_status", None))

        def flash(self, text: str, **_kwargs) -> None:
            calls.append(("flash", text))

    panel = _FakePanel()
    input_bar = _FakeInputBar()
    status = _FakeStatusBar()

    def query_one(cls: type):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "StatusBar":
            return status
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _agent_registry=None,
        _model_registry=None,
        _creating_new_session=False,
        _restoring_session=False,
        _gc_messages=[],
        _profile="",
        _runtime_details=None,
        _state_store=None,
        _last_usage_tokens=19_635,
        _last_total_session_tokens=253_535,
        context_usage_state=initial_context_usage,
        query_one=query_one,
        _update_subtitle=lambda: calls.append(("subtitle", None)),
        _set_agent_loading=lambda value: calls.append(("agent_loading", value)),
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)

    asyncio.run(
        handler.on_session_ready(
            SessionReady(
                agent_profile="Code",
                display_name="Code Agent",
                session_id="session-existing",
                max_context_tokens=event_max_context_tokens,
                primary_cwd="/workspace/existing",
            )
        )
    )
    return screen, calls, initial_context_usage


def test_session_ready_for_existing_session_preserves_context_usage_when_max_unchanged() -> None:
    """A repeated SessionReady must not zero the Context-panel token breakdown."""

    screen, calls, initial_context_usage = _run_existing_session_ready(
        current_max_context_tokens=200_000,
        event_max_context_tokens=200_000,
    )

    assert screen.context_usage_state is initial_context_usage
    assert screen.chat_workspace_cwd == "/workspace/existing"
    assert ("paste_cwd", "/workspace/existing") in calls
    assert ("session_id", "session-existing") in calls
    assert len(screen._gc_messages) == 1
    assert isinstance(screen._gc_messages[0], GcReclaimRequested)
    assert screen._gc_messages[0].reason is GcReclaimReason.SESSION_READY
    assert screen._gc_messages[0].prompt is True


def test_session_ready_for_existing_session_preserves_breakdown_when_max_changes() -> None:
    """A repeated SessionReady max refresh must carry the Context-panel breakdown forward."""

    screen, calls, initial_context_usage = _run_existing_session_ready(
        current_max_context_tokens=200_000,
        event_max_context_tokens=250_000,
    )

    assert screen.context_usage_state is not initial_context_usage
    assert screen.context_usage_state == ContextUsageState.with_window(
        used_tokens=19_635,
        max_context_tokens=250_000,
        total_session_tokens=253_535,
        total_session_input_tokens=120_000,
        total_session_output_tokens=83_535,
        total_session_cache_hit_tokens=50_000,
    )
    assert screen.chat_workspace_cwd == "/workspace/existing"
    assert ("paste_cwd", "/workspace/existing") in calls
    assert ("session_id", "session-existing") in calls


def test_agent_runtime_updated_refreshes_resource_counts_without_model_trail() -> None:
    tool_info: dict[str, str] = {}
    runtime_details = AgentRuntimeDetails(
        model=RuntimeModelDetails(
            profile_id="deepseek",
            name="DeepSeek-V4-Flash",
            model_id="deepseek-v4-flash",
            max_context_tokens=1_000_000,
        ),
        skill_details=[RuntimeSkillDetails(name="unit-converter", description="Convert units")],
        hook_sources=[
            RuntimeHookSourceDetails(
                scope="project",
                hooks=[
                    RuntimeHookDetails(id="guard", enabled=True),
                    RuntimeHookDetails(id="disabled", enabled=False),
                ],
            )
        ],
    )

    class _FakeStatusBar:
        def set_tool_info(self, trail: str) -> None:
            tool_info["trail"] = trail

    def _query_one(cls):
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _runtime_details=None,
        query_one=_query_one,
    )
    handler = make_backend_handler(screen)

    asyncio.run(
        handler.on_agent_runtime_updated(
            AgentRuntimeUpdated(
                model_profile_id="deepseek",
                max_context_tokens=1_000_000,
                tool_names=["read_file", "load_skill"],
                skill_names=["unit-converter"],
                memory_files=["AGENTS.md"],
                runtime_details=runtime_details,
            )
        )
    )

    assert screen._runtime_details is runtime_details
    trail = _status_trail(tool_info["trail"])
    assert trail == "2 tools · 1 skill · 1 hook · 1 file"
    assert "DeepSeek-V4-Flash" not in trail
    assert "deepseek-v4-flash" not in trail
    assert "1m" not in trail


def test_session_ready_for_new_session_resets_terminal_title_to_cwd() -> None:
    """Starting a new session should clear the previous user-message title preview."""

    calls: list[tuple[str, object]] = []
    terminal_title_cwds: list[str] = []

    class _FakePanel:
        border_subtitle = None

        def set_profile(self, profile: str) -> None:
            calls.append(("profile", profile))

        def set_tool_kinds(self, _tool_kinds: dict[str, str]) -> None:
            return

        async def clear(self) -> None:
            calls.append(("clear", None))

        def update_welcome(self, *, profile: str = "", cwd: str = "") -> None:
            calls.append(("welcome", (profile, cwd)))

        def set_session_id(self, session_id: str) -> None:
            calls.append(("session_id", session_id))

        def set_workspace_cwd(self, cwd: str) -> None:
            calls.append(("workspace_cwd", cwd))
            self.border_subtitle = Text(cwd)

    class _FakeInputBar:
        retry_mode = True

        def set_paste_cwd(self, cwd: str) -> None:
            calls.append(("paste_cwd", cwd))

        def set_clipboard_image_dir(self, directory: object) -> None:
            calls.append(("clipboard_image_dir", directory))

    class _FakeStatusBar:
        def set_profile(self, profile: str, *, description: str = "") -> None:
            calls.append(("status_profile", (profile, description)))

        def set_tool_info(self, _trail: str) -> None:
            return

        def clear_status(self) -> None:
            calls.append(("clear_status", None))

        def flash(self, text: str, **_kwargs) -> None:
            calls.append(("flash", text))

    class _FakeContextPanel:
        def reset(self, max_context_tokens: int = 0) -> None:
            calls.append(("context_reset", max_context_tokens))

    class _FakeSidebarPanel:
        context_panel = _FakeContextPanel()

    class _FakeStateStore:
        # Fake skips session_short_id; production JsonFileStateStore.session_dir shortens ids.
        def session_dir(self, session_id: str) -> Path:
            return Path("/sessions") / session_id

    panel = _FakePanel()
    input_bar = _FakeInputBar()
    status = _FakeStatusBar()
    sidebar = _FakeSidebarPanel()

    def record_terminal_title_cwd(cwd: str | None = None) -> None:
        terminal_title_cwds.append(cwd or "<current>")

    def query_one(cls: type):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "StatusBar":
            return status
        if cls.__name__ == "SidebarPanel":
            return sidebar
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _agent_registry=None,
        _model_registry=None,
        _creating_new_session=True,
        _restoring_session=False,
        _profile="",
        _runtime_details=None,
        _state_store=_FakeStateStore(),
        query_one=query_one,
        _set_has_messages=lambda value: calls.append(("has_messages", value)),
        _set_agent_loading=lambda value: calls.append(("agent_loading", value)),
        _set_terminal_title_for_cwd=record_terminal_title_cwd,
        _reset_session_title_state=lambda: None,
        _set_session_title_state=lambda **_kwargs: None,
        _session_custom_title="",
        _update_subtitle=lambda: calls.append(("subtitle", None)),
        _update_toc=lambda: calls.append(("toc", None)),
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None

    asyncio.run(
        handler.on_session_ready(
            SessionReady(
                agent_profile="Code",
                display_name="Code Agent",
                session_id="session-new",
                max_context_tokens=123,
                primary_cwd="/workspace/new",
            )
        )
    )

    assert screen._creating_new_session is False
    assert input_bar.retry_mode is False
    assert terminal_title_cwds == ["/workspace/new"]
    assert ("session_id", "session-new") in calls
    assert ("paste_cwd", "/workspace/new") in calls
    assert ("clipboard_image_dir", Path("/sessions/session-new/attachments/clipboard")) in calls


def test_usage_update_uses_source_id_for_session_window() -> None:
    """Same-profile sub-agent usage must not replace the main session window.

    Sub-agent UsageUpdates only update the session totals row in the sidebar;
    the used/max gauge stays at the main session's value so it does not
    flicker between contexts.  Session totals advance on every event because
    the cumulative figure is global.
    """

    debug_log: list[tuple[str, str]] = []
    context_usage_states: list[ContextUsageState] = []

    screen = SimpleNamespace(
        _main_usage_source_id="session-main",
        _last_usage_tokens=13_614,
        _last_total_session_tokens=13_614,
        context_usage_state=ContextUsageState.with_window(
            used_tokens=13_614,
            max_context_tokens=180_000,
            total_session_tokens=13_614,
        ),
        _debug=lambda event_type, detail="": debug_log.append((event_type, detail)),
    )
    handler = make_backend_handler(screen)

    for idx, total_tokens in enumerate((72_000, 68_500, 71_250), start=1):
        asyncio.run(
            handler.on_usage_update(
                UsageUpdate(
                    agent_profile="Code",
                    usage_source_id=f"sub-agent-{idx}",
                    total_tokens=total_tokens,
                    pct=round(total_tokens / 200_000 * 100, 1),
                    max_context_tokens=200_000,
                    local_tokens=total_tokens - 900,
                    total_session_tokens=233_900 + idx,
                )
            )
        )
        context_usage_states.append(screen.context_usage_state)

    assert debug_log == [
        ("Usage[Code]", "72,000 (36.0%) local=71,100 source=sub-agent-1"),
        ("Usage[Code]", "68,500 (34.2%) local=67,600 source=sub-agent-2"),
        ("Usage[Code]", "71,250 (35.6%) local=70,350 source=sub-agent-3"),
    ]
    # Main window must not move while a sub-agent fires usage updates...
    assert screen._last_usage_tokens == 13_614
    # ...but cumulative session totals always advance.
    assert screen._last_total_session_tokens == 233_903
    # Sidebar used/max gauge is NOT touched by sub-agent events; only the
    # session-totals state advances.
    assert [(state.used_tokens, state.max_context_tokens, state.update_window) for state in context_usage_states] == [
        (13_614, 180_000, False),
        (13_614, 180_000, False),
        (13_614, 180_000, False),
    ]
    assert [state.total_session_tokens for state in context_usage_states] == [233_901, 233_902, 233_903]

    asyncio.run(
        handler.on_usage_update(
            UsageUpdate(
                agent_profile="Code",
                usage_source_id="session-main",
                total_tokens=19_635,
                pct=9.8,
                max_context_tokens=200_000,
                local_tokens=17_960,
                total_session_tokens=253_535,
            )
        )
    )

    assert screen._last_usage_tokens == 19_635
    assert screen._last_total_session_tokens == 253_535
    # Main-session event refreshes the gauge via explicit routed view-state.
    assert screen.context_usage_state == ContextUsageState.with_window(
        used_tokens=19_635,
        max_context_tokens=200_000,
        total_session_tokens=253_535,
    )


def test_restore_session_title_state_honors_prefer_recovery() -> None:
    """Crash-recovered sessions must read title overlays from the same source
    as the replayed history (the recovery sidecar), not the stale primary."""

    calls: list[tuple[str, object]] = []

    class _Store:
        async def load_session_meta(self, session_id: str, *, prefer_recovery: bool = False) -> object:
            calls.append(("meta", (session_id, prefer_recovery)))
            return SimpleNamespace(custom_title="Pinned", generated_title="Auto", title="first msg")

    class _Ui:
        def reset_session_title_state(self) -> None:
            calls.append(("reset", None))

        def set_session_title_state(self, *, custom: str, generated: str, fallback: str) -> None:
            calls.append(("set", (custom, generated, fallback)))

    ui = _Ui()
    host = SimpleNamespace(state_store=_Store(), _ui=lambda: ui)

    asyncio.run(SessionHandler._restore_session_title_state(host, "sess1", prefer_recovery=True))

    assert ("meta", ("sess1", True)) in calls
    assert ("set", ("Pinned", "Auto", "first msg")) in calls


def test_apply_custom_title_clear_publishes_resolved_display_title() -> None:
    """Clearing a custom title falls back to the generated title; the event
    must carry that resolved display so the ACP bridge doesn't clear the
    client's label until the next turn regenerates one."""

    published: list[object] = []

    class _Bus:
        async def publish(self, event: object) -> None:
            published.append(event)

    class _Store:
        async def update_session_titles(
            self,
            session_id: str,
            *,
            custom_title: str | None = None,
            generated_title: str | None = None,
        ) -> object:
            return SimpleNamespace(
                custom_title="",
                generated_title="Auto topic",
                title="first msg",
                display_title="Auto topic",
            )

    class _Ui:
        def chat_session_id(self) -> str:
            return "sess1"

        def set_session_title_state(self, **kwargs: object) -> None:
            pass

        def flash_status(self, message: str) -> None:
            pass

    ui = _Ui()
    host = SimpleNamespace(
        state_store=_Store(),
        _custom_title_apply_lock=asyncio.Lock(),
        bus=_Bus(),
        _ui=lambda: ui,
    )

    asyncio.run(SessionHandler.apply_custom_session_title(host, "", "sess1"))

    [event] = published
    assert event.custom is True
    assert event.title == ""
    assert event.display_title == "Auto topic"


def test_session_restore_resets_terminal_title_to_restored_cwd() -> None:
    """Switching to an old session should clear the previous user-message title preview."""

    calls: list[tuple[str, object]] = []
    terminal_title_cwds: list[str] = []

    class _FakePanel:
        border_subtitle = None

        async def clear(self) -> None:
            calls.append(("clear", None))

        def set_session_id(self, session_id: str) -> None:
            calls.append(("session_id", session_id))

        def set_workspace_cwd(self, cwd: str) -> None:
            calls.append(("workspace_cwd", cwd))
            self.border_subtitle = Text(cwd)

        def update_usage(self, tokens: int, total_session_tokens: int = 0) -> None:
            calls.append(("usage", (tokens, total_session_tokens)))

        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            calls.append(("error", (message, action_label)))

    class _FakeInputBar:
        retry_mode = True

        def set_paste_cwd(self, cwd: str) -> None:
            calls.append(("paste_cwd", cwd))

        def set_clipboard_image_dir(self, directory: object) -> None:
            calls.append(("clipboard_image_dir", directory))

    class _FakeStatusBar:
        def flash(self, text: str) -> None:
            calls.append(("flash", text))

    class _FakeContextPanel:
        def clear_blocks(self) -> None:
            calls.append(("clear_blocks", None))

    class _FakeSidebarPanel:
        context_panel = _FakeContextPanel()

    panel = _FakePanel()
    input_bar = _FakeInputBar()
    status = _FakeStatusBar()
    sidebar = _FakeSidebarPanel()
    context_usage_state = ContextUsageState.with_window(
        used_tokens=7,
        max_context_tokens=55,
        total_session_tokens=11,
    )

    def query_one(cls: type):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "StatusBar":
            return status
        if cls.__name__ == "SidebarPanel":
            return sidebar
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _restoring_session=True,
        _last_usage_tokens=42,
        _last_total_session_tokens=100,
        _gc_messages=[],
        context_usage_state=context_usage_state,
        _state_store=None,
        query_one=query_one,
        _set_has_messages=lambda value: calls.append(("has_messages", value)),
        _set_terminal_title_for_cwd=terminal_title_cwds.append,
        _reset_session_title_state=lambda: None,
        _set_session_title_state=lambda **_kwargs: None,
        _session_custom_title="",
        _events=SimpleNamespace(finish_agent_load=lambda message: calls.append(("finish", message))),
        _debug=lambda *_args: None,
    )

    asyncio.run(
        _make_session_handler(screen).on_session_restored(
            SessionRestored(
                session_id="session-old",
                agent_profile="Code",
                display_name="Code Agent",
                message_count=3,
                primary_cwd="/old/missing/path",
                cwd_warning="Working directory no longer exists: /old/missing/path",
            )
        )
    )

    assert screen._restoring_session is False
    assert input_bar.retry_mode is False
    assert terminal_title_cwds == ["/old/missing/path"]
    assert ("paste_cwd", "/old/missing/path") in calls
    assert ("workspace_cwd", "/old/missing/path") in calls
    assert ("error", ("Working directory no longer exists: /old/missing/path", None)) in calls
    assert ("session_id", "session-old") in calls
    assert screen.context_usage_state == context_usage_state
    assert len(screen._gc_messages) == 1
    assert isinstance(screen._gc_messages[0], GcReclaimRequested)
    assert screen._gc_messages[0].reason is GcReclaimReason.SESSION_RESTORED
    assert screen._gc_messages[0].prompt is True
    flash = next(value for name, value in calls if name == "flash")
    assert isinstance(flash, MessageRef)
    assert flash.definition.key == "tui.status.session_restored"
    assert _status_text(flash) == f"Session restored: {session_short_id('session-old')}"


def test_session_restore_marks_fully_compacted_session_as_having_messages(tmp_path: Path) -> None:
    """A restored compacted session should still be forkable from the TUI."""

    calls: list[tuple[str, object]] = []
    compressed = CompressedBlock(
        compressed_context_id="ctx_1",
        messages=[Message("user", ["old real turn"]), Message("assistant", ["old reply"])],
        summary_text="old turn",
        marker_id="turn_1",
        turn_range=(1, 1),
        created_at="2026-03-17T00:00:00+00:00",
    )
    state = {"messages": [], "compressed_msgs": [compressed]}

    class _FakePanel:
        border_subtitle = None

        async def clear(self) -> None:
            calls.append(("clear", None))

        def set_session_id(self, session_id: str) -> None:
            calls.append(("session_id", session_id))

        def set_workspace_cwd(self, cwd: str) -> None:
            calls.append(("workspace_cwd", cwd))
            self.border_subtitle = Text(cwd)

        def update_usage(self, tokens: int, total_session_tokens: int = 0) -> None:
            calls.append(("usage", (tokens, total_session_tokens)))

        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            calls.append(("error", (message, action_label)))

    class _FakeInputBar:
        retry_mode = True

        def set_paste_cwd(self, cwd: str) -> None:
            calls.append(("paste_cwd", cwd))

        def set_clipboard_image_dir(self, directory: object) -> None:
            calls.append(("clipboard_image_dir", directory))

    class _FakeStatusBar:
        def flash(self, text: str) -> None:
            calls.append(("flash", text))

    class _FakeContextPanel:
        def clear_blocks(self) -> None:
            calls.append(("clear_blocks", None))

        def add_compressed_block(
            self,
            ctx_id: str,
            summary: str,
            freed_messages: int = 0,
            turn_range: tuple[int, int] = (0, 0),
        ) -> None:
            calls.append(("compressed_block", (ctx_id, summary, freed_messages, turn_range)))

    class _FakeSidebarPanel:
        context_panel = _FakeContextPanel()

    class _FakeStateStore:
        def session_dir(self, session_id: str) -> Path:
            return tmp_path / "sessions" / session_id

        async def load_session(self, session_id: str, *, prefer_recovery: bool = False) -> dict[str, object]:
            calls.append(("load_session", (session_id, prefer_recovery)))
            return state

        async def load_session_raw(self, session_id: str, *, prefer_recovery: bool = False) -> list[dict[str, object]]:
            calls.append(("load_session_raw", (session_id, prefer_recovery)))
            return []

        async def list_sessions(self) -> list[object]:
            return []

    panel = _FakePanel()
    input_bar = _FakeInputBar()
    status = _FakeStatusBar()
    sidebar = _FakeSidebarPanel()

    def query_one(cls: type):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "StatusBar":
            return status
        if cls.__name__ == "SidebarPanel":
            return sidebar
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _restoring_session=True,
        _last_usage_tokens=0,
        _last_total_session_tokens=0,
        context_usage_state=ContextUsageState.with_window(
            used_tokens=0,
            max_context_tokens=100_000,
            total_session_tokens=0,
        ),
        _state_store=_FakeStateStore(),
        query_one=query_one,
        _set_has_messages=lambda value: calls.append(("has_messages", value)),
        _set_terminal_title_for_cwd=lambda cwd: calls.append(("title_cwd", cwd)),
        _reset_session_title_state=lambda: None,
        _set_session_title_state=lambda **_kwargs: None,
        _session_custom_title="",
        _events=SimpleNamespace(finish_agent_load=lambda message: calls.append(("finish", message))),
        _debug=lambda *_args: None,
    )

    asyncio.run(
        _make_session_handler(screen).on_session_restored(
            SessionRestored(
                session_id="session-compacted",
                agent_profile="Code",
                display_name="Code Agent",
                message_count=0,
                primary_cwd="/old/missing/path",
            )
        )
    )

    assert ("has_messages", True) in calls
    assert ("has_messages", False) not in calls
    assert ("workspace_cwd", "/old/missing/path") in calls
    assert ("compressed_block", ("ctx_1", "old turn", 2, (1, 1))) in calls
    assert input_bar.retry_mode is False


def test_session_restore_moves_shell_panel_to_existing_restored_cwd(tmp_path: Path) -> None:
    """Restored sessions should keep the embedded terminal aligned with the agent workspace."""

    calls: list[tuple[str, object]] = []
    restored_cwd = tmp_path / "workspace"
    restored_cwd.mkdir()

    class _FakePanel:
        border_subtitle = None

        async def clear(self) -> None:
            calls.append(("clear", None))

        def set_session_id(self, session_id: str) -> None:
            calls.append(("session_id", session_id))

        def set_workspace_cwd(self, cwd: str) -> None:
            calls.append(("workspace_cwd", cwd))
            self.border_subtitle = Text(cwd)

        def update_usage(self, tokens: int, total_session_tokens: int = 0) -> None:
            calls.append(("usage", (tokens, total_session_tokens)))

        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            calls.append(("error", (message, action_label)))

    class _FakeInputBar:
        retry_mode = True

        def set_paste_cwd(self, cwd: str) -> None:
            calls.append(("paste_cwd", cwd))

        def set_clipboard_image_dir(self, directory: object) -> None:
            calls.append(("clipboard_image_dir", directory))

    class _FakeStatusBar:
        def flash(self, text: str) -> None:
            calls.append(("flash", text))

    class _FakeContextPanel:
        def clear_blocks(self) -> None:
            calls.append(("clear_blocks", None))

    class _FakeSidebarPanel:
        context_panel = _FakeContextPanel()

    class _FakeShellPanel:
        async def change_directory(self, cwd: str) -> None:
            calls.append(("shell_cwd", cwd))

    panel = _FakePanel()
    input_bar = _FakeInputBar()
    status = _FakeStatusBar()
    sidebar = _FakeSidebarPanel()
    shell = _FakeShellPanel()

    def query_one(cls: type):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "StatusBar":
            return status
        if cls.__name__ == "SidebarPanel":
            return sidebar
        if cls.__name__ == "ShellPanel":
            return shell
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _restoring_session=True,
        _last_usage_tokens=0,
        _last_total_session_tokens=0,
        context_usage_state=ContextUsageState.with_window(
            used_tokens=0,
            max_context_tokens=100_000,
            total_session_tokens=0,
        ),
        _state_store=None,
        query_one=query_one,
        _set_has_messages=lambda value: calls.append(("has_messages", value)),
        _set_terminal_title_for_cwd=lambda cwd: calls.append(("title_cwd", cwd)),
        _reset_session_title_state=lambda: None,
        _set_session_title_state=lambda **_kwargs: None,
        _session_custom_title="",
        _events=SimpleNamespace(finish_agent_load=lambda message: calls.append(("finish", message))),
        _debug=lambda *_args: None,
    )

    asyncio.run(
        _make_session_handler(screen).on_session_restored(
            SessionRestored(
                session_id="session-old",
                agent_profile="Code",
                display_name="Code Agent",
                message_count=0,
                primary_cwd=str(restored_cwd),
            )
        )
    )

    assert ("shell_cwd", str(restored_cwd)) in calls
    assert ("paste_cwd", str(restored_cwd)) in calls
    assert ("workspace_cwd", str(restored_cwd)) in calls
    assert panel.border_subtitle.plain == str(restored_cwd)


def test_welcome_rollback_keeps_logo_metadata_and_suppresses_chdir_marker() -> None:
    """Rolling back all turns returns to a populated welcome screen, not chat history mode."""

    calls: list[tuple[str, object]] = []
    welcome_updates: list[tuple[str, str]] = []
    system_messages: list[str] = []
    terminal_title_cwds: list[str] = []

    class _FakePanel:
        border_subtitle = None

        async def clear(self) -> None:
            calls.append(("clear", None))

        def set_session_id(self, session_id: str) -> None:
            calls.append(("session_id", session_id))

        def set_workspace_cwd(self, cwd: str) -> None:
            calls.append(("workspace_cwd", cwd))
            self.border_subtitle = Text(cwd)

        def update_welcome(self, *, profile: str = "", cwd: str = "") -> None:
            welcome_updates.append((profile, cwd))

        async def add_system(self, text: str, *, key: str | None = None) -> None:
            system_messages.append(text)

        async def update_system(self, _key: str, new_text: str) -> None:
            system_messages.append(new_text)

        async def remove_system(self, _key: str) -> None:
            calls.append(("remove_system", _key))

    class _FakeInputBar:
        retry_mode = True
        has_messages = True

        def set_paste_cwd(self, cwd: str) -> None:
            calls.append(("paste_cwd", cwd))

        def set_clipboard_image_dir(self, directory: object) -> None:
            calls.append(("clipboard_image_dir", directory))

    class _FakeContextPanel:
        def reset(self) -> None:
            calls.append(("context_reset", None))

    class _FakeSidebarPanel:
        context_panel = _FakeContextPanel()

    class _FakeShellPanel:
        async def change_directory(self, cwd: str) -> None:
            calls.append(("shell_cwd", cwd))

    panel = _FakePanel()
    input_bar = _FakeInputBar()
    sidebar = _FakeSidebarPanel()
    shell = _FakeShellPanel()

    def query_one(cls: type):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "SidebarPanel":
            return sidebar
        if cls.__name__ == "ShellPanel":
            return shell
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _engine=None,
        _profile="Code Agent",
        _has_messages=True,
        _gc_messages=[],
        _chdir_original_cwd="/repo/original",
        _chdir_current_cwd="/repo/current",
        _workspace_cwd=lambda: "/repo/current",
        context_usage_state=ContextUsageState.with_window(
            used_tokens=12_000,
            max_context_tokens=180_000,
            total_session_tokens=64_000,
        ),
        _suggestions=SimpleNamespace(file_cache=_stale_file_cache("stale.py")),
        query_one=query_one,
        _update_toc=lambda: calls.append(("toc", None)),
        _set_terminal_title_for_cwd=terminal_title_cwds.append,
        _reset_session_title_state=lambda: calls.append(("reset_title_state", None)),
        notify=lambda *_args, **_kwargs: None,
        _debug=lambda *_args: None,
    )

    def set_has_messages(value: bool) -> None:
        screen._has_messages = value
        input_bar.has_messages = value

    screen._set_has_messages = set_has_messages

    asyncio.run(
        _make_rollback_controller_for_test(screen).on_result(RollbackResult(session_id="session-1", target_turn=0))
    )
    assert ("reset_title_state", None) in calls
    asyncio.run(_make_session_handler(screen).on_workspace_updated(WorkspaceUpdated(primary_cwd="/repo/next")))

    assert screen._has_messages is False
    assert screen._chdir_original_cwd is None
    assert input_bar.retry_mode is False
    assert screen.context_usage_state == ContextUsageState.with_window(
        used_tokens=0,
        max_context_tokens=180_000,
    )
    assert ("Code Agent", "/repo/current") in welcome_updates
    assert welcome_updates[-1] == ("Code Agent", "/repo/next")
    assert panel.border_subtitle.plain == "/repo/next"
    assert ("workspace_cwd", "/repo/current") in calls
    assert [message.reason for message in screen._gc_messages] == [
        GcReclaimReason.ROLLBACK_WELCOME,
        GcAbsorbReason.WORKSPACE_UI_UPDATED,
    ]
    assert screen._gc_messages[0].prompt is True
    assert ("workspace_cwd", "/repo/next") in calls
    assert ("paste_cwd", "/repo/next") in calls
    assert system_messages == []
    assert terminal_title_cwds == ["/repo/next"]


def test_workspace_update_with_messages_updates_terminal_title_and_chdir_marker(tmp_path: Path) -> None:
    """Changing cwd after chat starts should still update the terminal title."""

    calls: list[tuple[str, object]] = []
    system_messages: list[str] = []
    terminal_title_cwds: list[str] = []
    current_cwd = tmp_path / "current"
    next_cwd = tmp_path / "next"
    current_cwd.mkdir()
    next_cwd.mkdir()

    class _FakePanel:
        border_subtitle = None

        async def add_system(self, text: str, *, key: str | None = None) -> None:
            calls.append(("add_system_key", key))
            system_messages.append(text)

        async def update_system(self, _key: str, new_text: str) -> None:
            system_messages.append(new_text)

        async def remove_system(self, key: str) -> None:
            calls.append(("remove_system", key))

        def set_workspace_cwd(self, cwd: str) -> None:
            calls.append(("workspace_cwd", cwd))
            self.border_subtitle = Text(cwd)

    class _FakeShellPanel:
        async def change_directory(self, cwd: str) -> None:
            calls.append(("shell_cwd", cwd))

    class _FakeInputBar:
        def set_paste_cwd(self, cwd: str) -> None:
            calls.append(("paste_cwd", cwd))

        def set_clipboard_image_dir(self, directory: object) -> None:
            calls.append(("clipboard_image_dir", directory))

    panel = _FakePanel()
    shell = _FakeShellPanel()
    input_bar = _FakeInputBar()

    def query_one(cls: type):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ShellPanel":
            return shell
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _has_messages=True,
        _chdir_original_cwd=None,
        _chdir_current_cwd=str(current_cwd),
        _suggestions=SimpleNamespace(file_cache=_stale_file_cache("stale.py")),
        query_one=query_one,
        _set_terminal_title_for_cwd=terminal_title_cwds.append,
        _debug=lambda *_args: None,
    )

    asyncio.run(_make_session_handler(screen).on_workspace_updated(WorkspaceUpdated(primary_cwd=str(next_cwd))))

    assert screen._chdir_current_cwd == str(next_cwd)
    assert panel.border_subtitle.plain == str(next_cwd)
    assert ("paste_cwd", str(next_cwd)) in calls
    assert ("shell_cwd", str(next_cwd)) in calls
    assert ("remove_system", "chdir") in calls
    assert system_messages == [f"Working directory → {next_cwd}"]
    assert terminal_title_cwds == [str(next_cwd)]


def test_workspace_update_with_missing_cwd_does_not_change_shell_directory(tmp_path: Path) -> None:
    """A restored session can point at a workspace that was deleted between launches."""

    calls: list[tuple[str, object]] = []
    terminal_title_cwds: list[str] = []
    missing_cwd = tmp_path / "deleted-workspace"

    class _FakePanel:
        border_subtitle = None

        def update_welcome(self, *, profile: str = "", cwd: str = "") -> None:
            calls.append(("welcome", (profile, cwd)))

        def set_workspace_cwd(self, cwd: str) -> None:
            calls.append(("workspace_cwd", cwd))
            self.border_subtitle = Text(cwd)

    class _FakeShellPanel:
        async def change_directory(self, cwd: str) -> None:
            raise FileNotFoundError(cwd)

    class _FakeInputBar:
        def set_paste_cwd(self, cwd: str) -> None:
            calls.append(("paste_cwd", cwd))

        def set_clipboard_image_dir(self, directory: object) -> None:
            calls.append(("clipboard_image_dir", directory))

    panel = _FakePanel()
    shell = _FakeShellPanel()
    input_bar = _FakeInputBar()

    def query_one(cls: type):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ShellPanel":
            return shell
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _profile="Code Agent",
        _has_messages=False,
        _chdir_original_cwd=None,
        _chdir_current_cwd=str(tmp_path),
        _suggestions=SimpleNamespace(file_cache=_stale_file_cache("stale.py")),
        query_one=query_one,
        _set_terminal_title_for_cwd=terminal_title_cwds.append,
        _debug=lambda *_args: None,
    )

    asyncio.run(_make_session_handler(screen).on_workspace_updated(WorkspaceUpdated(primary_cwd=str(missing_cwd))))

    assert screen._chdir_current_cwd == str(missing_cwd)
    assert panel.border_subtitle.plain == str(missing_cwd)
    assert ("paste_cwd", str(missing_cwd)) in calls
    assert ("welcome", ("Code Agent", str(missing_cwd))) in calls
    assert terminal_title_cwds == [str(missing_cwd)]


# ──────────── workspace MRU touch scheduling ───────────────────────────


def _capture_mru_touches(captured: list[dict[str, object]]):
    def _capture(paths, *, max_entries, session_id="", used_at=None) -> None:
        captured.append(
            {"paths": list(paths), "max_entries": max_entries, "session_id": session_id, "used_at": used_at}
        )

    return _capture


def _make_mru_workspace_screen(calls: list[tuple[str, object]]) -> SimpleNamespace:
    """Minimal screen fake for the no-messages on_workspace_updated path."""

    class _FakePanel:
        border_subtitle = None

        def update_welcome(self, *, profile: str = "", cwd: str = "") -> None:
            calls.append(("welcome", (profile, cwd)))

        def set_workspace_cwd(self, cwd: str) -> None:
            calls.append(("workspace_cwd", cwd))

    class _FakeShellPanel:
        async def change_directory(self, cwd: str) -> None:
            calls.append(("shell_cwd", cwd))

    class _FakeInputBar:
        def set_paste_cwd(self, cwd: str) -> None:
            calls.append(("paste_cwd", cwd))

        def set_clipboard_image_dir(self, directory: object) -> None:
            calls.append(("clipboard_image_dir", directory))

    panel = _FakePanel()
    shell = _FakeShellPanel()
    input_bar = _FakeInputBar()

    def query_one(cls: type):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ShellPanel":
            return shell
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    return SimpleNamespace(
        _profile="Code Agent",
        _has_messages=False,
        _chdir_original_cwd=None,
        _chdir_current_cwd="/repo/current",
        _suggestions=SimpleNamespace(file_cache=_stale_file_cache("stale.py")),
        query_one=query_one,
        _set_terminal_title_for_cwd=lambda _cwd: None,
        _debug=lambda *_args: None,
    )


async def _drain_workspace_mru_tasks() -> None:
    from chrys.app.tui.support import workspace_mru

    while workspace_mru._BACKGROUND_TASKS:
        await asyncio.gather(*list(workspace_mru._BACKGROUND_TASKS), return_exceptions=True)


def test_session_ready_schedules_mru_touches_for_all_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        "chrys.app.tui.screens.main.event_handlers.schedule_workspace_mru_touches",
        _capture_mru_touches(captured),
    )

    class _FakePanel:
        def set_profile(self, _profile: str) -> None:
            return

        def set_tool_kinds(self, _tool_kinds: dict[str, str]) -> None:
            return

    class _FakeInputBar:
        def set_clipboard_image_dir(self, _directory: object) -> None:
            return

    class _FakeStatusBar:
        def set_profile(self, _profile: str, *, description: str = "") -> None:
            return

        def set_tool_info(self, _trail: str) -> None:
            return

    def _query_one(cls):
        if cls.__name__ == "ChatPanel":
            return _FakePanel()
        if cls.__name__ == "InputBar":
            return _FakeInputBar()
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _agent_registry=None,
        _model_registry=None,
        _creating_new_session=False,
        _restoring_session=True,
        _profile="",
        _state_store=None,
        query_one=_query_one,
        _update_subtitle=lambda: None,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._services.workspace_mru_max_entries = 7
    event = SessionReady(
        agent_profile="Code",
        session_id="session-1",
        primary_cwd="/repo/primary",
        working_dirs=["/repo/primary", "/repo/extra"],
    )

    asyncio.run(handler.on_session_ready(event))

    assert captured == [
        {
            "paths": ["/repo/primary", "/repo/primary", "/repo/extra"],
            "max_entries": 7,
            "session_id": "session-1",
            "used_at": event.timestamp,
        }
    ]


def test_session_restored_schedules_mru_touches_for_all_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        "chrys.app.tui.screens.main.session_handlers.schedule_workspace_mru_touches",
        _capture_mru_touches(captured),
    )
    calls: list[tuple[str, object]] = []

    class _FakePanel:
        border_subtitle = None

        async def clear(self) -> None:
            return

        def set_session_id(self, _session_id: str) -> None:
            return

        def set_workspace_cwd(self, _cwd: str) -> None:
            return

        def update_usage(self, _tokens: int, _total_session_tokens: int = 0) -> None:
            return

    class _FakeInputBar:
        retry_mode = True

        def set_paste_cwd(self, cwd: str) -> None:
            calls.append(("paste_cwd", cwd))

        def set_clipboard_image_dir(self, _directory: object) -> None:
            return

    class _FakeStatusBar:
        def flash(self, _text: str) -> None:
            return

    panel = _FakePanel()
    input_bar = _FakeInputBar()
    status = _FakeStatusBar()

    def query_one(cls: type):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "StatusBar":
            return status
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _restoring_session=True,
        _state_store=None,
        query_one=query_one,
        _set_has_messages=lambda _value: None,
        _set_terminal_title_for_cwd=lambda _cwd: None,
        _reset_session_title_state=lambda: None,
        _set_session_title_state=lambda **_kwargs: None,
        _session_custom_title="",
        _events=SimpleNamespace(finish_agent_load=lambda _message: None),
        _debug=lambda *_args: None,
    )
    event = SessionRestored(
        session_id="session-old",
        agent_profile="Code",
        display_name="Code Agent",
        message_count=3,
        primary_cwd="/repo/restored",
        working_dirs=["/repo/restored", "/repo/extra"],
    )

    asyncio.run(_make_session_handler(screen).on_session_restored(event))

    assert captured == [
        {
            "paths": ["/repo/restored", "/repo/restored", "/repo/extra"],
            "max_entries": 20,
            "session_id": "session-old",
            "used_at": event.timestamp,
        }
    ]


def test_workspace_updated_schedules_mru_touches_with_event_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        "chrys.app.tui.screens.main.session_handlers.schedule_workspace_mru_touches",
        _capture_mru_touches(captured),
    )
    screen = _make_mru_workspace_screen([])
    event = WorkspaceUpdated(
        primary_cwd="/repo/next",
        working_dirs=["/repo/next", "/repo/extra"],
        session_id="session-1",
    )

    asyncio.run(_make_session_handler(screen).on_workspace_updated(event))

    assert captured == [
        {
            "paths": ["/repo/next", "/repo/next", "/repo/extra"],
            "max_entries": 20,
            "session_id": "session-1",
            "used_at": event.timestamp,
        }
    ]


def test_workspace_updated_records_mru_end_to_end(tmp_path: Path) -> None:
    """The detached touch task lands in the index; missing dirs never do."""
    from chrys.app.tui.support.workspace_mru import ensure_workspace_mru_index, load_workspace_mru

    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    primary.mkdir()
    extra.mkdir()
    missing = tmp_path / "deleted"
    ensure_workspace_mru_index([], max_entries=20, root_key="sha256:test-root")
    screen = _make_mru_workspace_screen([])
    event = WorkspaceUpdated(primary_cwd=str(primary), working_dirs=[str(extra), str(missing)])

    async def run() -> None:
        await _make_session_handler(screen).on_workspace_updated(event)
        await _drain_workspace_mru_tasks()

    asyncio.run(run())

    assert load_workspace_mru(20) == [str(primary), str(extra)]


def test_workspace_updated_skips_mru_touch_when_index_missing(tmp_path: Path) -> None:
    """Event touches must not create the index ahead of the one-time backfill."""
    from chrys.app.tui.support.workspace_mru import workspace_mru_exists

    primary = tmp_path / "primary"
    primary.mkdir()
    calls: list[tuple[str, object]] = []
    screen = _make_mru_workspace_screen(calls)

    async def run() -> None:
        await _make_session_handler(screen).on_workspace_updated(WorkspaceUpdated(primary_cwd=str(primary)))
        await _drain_workspace_mru_tasks()

    asyncio.run(run())

    assert not workspace_mru_exists()
    # The skipped touch is not an error — the UI refresh completed normally.
    assert ("workspace_cwd", str(primary)) in calls


def test_workspace_updated_mru_failure_does_not_break_ui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chrys.app.tui.support import workspace_mru
    from chrys.app.tui.support.workspace_mru import ensure_workspace_mru_index

    primary = tmp_path / "primary"
    primary.mkdir()
    ensure_workspace_mru_index([], max_entries=20, root_key="sha256:test-root")

    def _boom(*_args: object, **_kwargs: object) -> bool:
        raise OSError("disk full")

    monkeypatch.setattr(workspace_mru, "record_workspace_mru_uses", _boom)
    calls: list[tuple[str, object]] = []
    screen = _make_mru_workspace_screen(calls)

    async def run() -> None:
        await _make_session_handler(screen).on_workspace_updated(WorkspaceUpdated(primary_cwd=str(primary)))
        await _drain_workspace_mru_tasks()

    asyncio.run(run())  # must not raise

    assert ("workspace_cwd", str(primary)) in calls


# ──────────── approval flow: parallel-judge race ───────────────────────
#
# In AUTO mode with parallel tool calls the engine publishes N
# ``ApprovalRequest`` events concurrently and spawns N judge tasks.  The
# TUI only shows one dialog at a time (the rest sit in ``_approval_queue``
# as raw events), so judges for queued requests can finish and publish
# ``ApprovalReviewed`` *before* their dialog is ever mounted.  The handler
# must:
#
#  1. Cache such verdicts in ``_pending_verdicts`` keyed by request_id.
#  2. When ``_show_next_approval`` pops that request, consume the cached
#     verdict: if approved, skip the dialog entirely; if flagged, push the
#     dialog and deliver the verdict after mount.
#  3. Drop late arrivals silently when the matching request has already
#     been resolved (not in queue, not in live dialogs dict).


class _FakeApprovalDialog:
    """Mock that mimics ``ApprovalDialog`` without touching the Textual runtime.

    Records ``receive_verdict`` calls and ``call_after_refresh`` schedules
    so tests can assert both immediate-delivery and deferred-delivery paths.
    """

    def __init__(
        self,
        *,
        caller_name: str,
        tool_name: str,
        tool_kind: str = "",
        args: dict | None = None,
        judging: bool = False,
        approval_body=None,
        presentation_kind: str = "",
    ) -> None:
        self.caller_name = caller_name
        self._tool_name = tool_name
        self.tool_kind = tool_kind
        self.args = args or {}
        self.judging = judging
        self.approval_body = approval_body
        self.presentation_kind = presentation_kind
        self._dismissed = False
        self._user_decision_submitted = False
        self.received_verdict = None
        self.after_refresh_calls: list[tuple] = []

    @property
    def is_dismissed(self) -> bool:
        return self._dismissed

    @property
    def user_decision_submitted(self) -> bool:
        return self._user_decision_submitted

    def receive_verdict(self, verdict) -> None:
        self.received_verdict = verdict

    def call_after_refresh(self, fn, *args) -> None:
        # Record for assertion; tests that want to simulate mount will
        # invoke the recorded callable themselves.
        self.after_refresh_calls.append((fn, args))


class _FakeNotificationService:
    def __init__(self) -> None:
        self.events: list[object] = []

    def notify(self, event: object) -> bool:
        self.events.append(event)
        return True


class _FakeApp:
    """Mock ``App`` that captures pushed screens and their result callbacks."""

    def __init__(self) -> None:
        self.pushed: list[tuple[object, object]] = []  # [(screen, callback)]
        self.notification_service = _FakeNotificationService()

    def push_screen(self, screen, callback) -> None:
        self.pushed.append((screen, callback))


class _FakeBus:
    """Mock bus that captures frontend → backend events."""

    def __init__(self) -> None:
        self.published: list[object] = []

    async def publish(self, event) -> None:
        self.published.append(event)


class _FakeAskUserDialog:
    def __init__(
        self,
        request_id: str,
        question: str,
        options: list[str] | None = None,
        caller_name: str = "",
        initial_response: str = "",
    ) -> None:
        self.request_id = request_id
        self.question = question
        self.options = options or []
        self.caller_name = caller_name
        self.initial_response = initial_response
        self.dismissed = False
        self.callback = None

    def dismiss_due_to_timeout(self) -> None:
        self.dismissed = True
        if self.callback is not None:
            self.callback(None)

    def submit(self, text: str) -> None:
        self.dismissed = True
        if self.callback is not None:
            self.callback((self.request_id, text))

    def answer_inline(self, draft_text: str = "") -> None:
        from chrys.app.tui.screens.dialogs.ask_user import AskUserInlineResult

        self.dismissed = True
        if self.callback is not None:
            self.callback(AskUserInlineResult(self.request_id, draft_text))


class _FakeAskUserApp:
    def __init__(self) -> None:
        self.notification_service = _FakeNotificationService()
        self.pushed: list[tuple[_FakeAskUserDialog, object]] = []

    def push_screen(self, screen: _FakeAskUserDialog, callback) -> None:
        screen.callback = callback
        self.pushed.append((screen, callback))


class _FakeQuestionHostRefresh:
    def __init__(self, calls: list[tuple[str, str]] | None = None) -> None:
        self.calls = calls if calls is not None else []

    def pause(self) -> None:
        self.calls.append(("pause", ""))

    def resume(self) -> None:
        self.calls.append(("resume", ""))


def test_question_dialog_suspends_background_host_refresh_until_result(monkeypatch: pytest.MonkeyPatch) -> None:
    import chrys.app.tui.screens.dialogs.ask_user as _ask_user_mod
    from chrys.foundation.events.types import QuestionToUser

    monkeypatch.setattr(_ask_user_mod, "AskUserDialog", _FakeAskUserDialog)
    lifecycle_calls: list[tuple[str, str]] = []
    host_refresh = _FakeQuestionHostRefresh(lifecycle_calls)
    results: list[object] = []
    app = _FakeAskUserApp()

    def on_result(result: object) -> None:
        lifecycle_calls.append(("result", str(result)))
        results.append(result)

    screen = SimpleNamespace(
        app=app,
        pause_host_refresh=host_refresh.pause,
        resume_host_refresh=host_refresh.resume,
    )
    adapter = MainScreenViewAdapter(screen)  # type: ignore[arg-type]

    adapter.show_question_dialog(
        QuestionToUser(request_id="req-1", call_id="call-1", question="Proceed?"),
        "",
        on_result,
    )
    dialog, callback = app.pushed[0]

    assert lifecycle_calls == [("pause", "")]
    callback((dialog.request_id, "yes"))
    assert lifecycle_calls == [
        ("pause", ""),
        ("result", "('req-1', 'yes')"),
        ("resume", ""),
    ]
    assert results == [("req-1", "yes")]


def test_question_dialog_restores_host_refresh_when_push_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    import chrys.app.tui.screens.dialogs.ask_user as _ask_user_mod
    from chrys.foundation.events.types import QuestionToUser

    monkeypatch.setattr(_ask_user_mod, "AskUserDialog", _FakeAskUserDialog)
    host_refresh = _FakeQuestionHostRefresh()

    class _FailingApp:
        def push_screen(self, _screen: object, _callback: object) -> None:
            raise RuntimeError("push failed")

    screen = SimpleNamespace(
        app=_FailingApp(),
        pause_host_refresh=host_refresh.pause,
        resume_host_refresh=host_refresh.resume,
    )
    adapter = MainScreenViewAdapter(screen)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="push failed"):
        adapter.show_question_dialog(
            QuestionToUser(request_id="req-1", call_id="call-1", question="Proceed?"),
            "",
            lambda _result: None,
        )

    assert host_refresh.calls == [("pause", ""), ("resume", "")]


@pytest.mark.asyncio
async def test_host_refresh_lease_blocks_every_background_widget_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from textual.app import App, ComposeResult
    from textual.screen import ModalScreen
    from textual.widgets import Static

    class _BackgroundScreen(_HostRefreshLeaseScreen):
        def compose(self) -> ComposeResult:
            yield Static("base", id="base-refresh-probe")

    class _TranslucentModal(ModalScreen[None]):
        DEFAULT_CSS = "_TranslucentModal { background: $background 40%; }"

        def compose(self) -> ComposeResult:
            yield Static("modal", id="modal-refresh-probe")

    class _RefreshProbeApp(App):
        def __init__(self) -> None:
            super().__init__()
            self.background = _BackgroundScreen()

        def on_mount(self) -> None:
            self.push_screen(self.background)

    app = _RefreshProbeApp()
    async with app.run_test(size=(40, 10)) as pilot:
        await pilot.pause()
        app.background.pause_host_refresh()
        app.background.pause_host_refresh()
        modal = _TranslucentModal()
        app.push_screen(modal)
        await pilot.pause()
        assert app.background in app._background_screens

        host_writes: list[object] = []
        original_display = app._display

        def record_display(screen: object, renderable: object) -> None:
            if renderable is not None:
                host_writes.append(screen)
            original_display(screen, renderable)  # type: ignore[arg-type]

        monkeypatch.setattr(app, "_display", record_display)
        modal.query_one("#modal-refresh-probe", Static).update("modal update")
        await pilot.pause()
        assert modal in host_writes
        host_writes.clear()

        probe = app.background.query_one("#base-refresh-probe", Static)
        for update in range(5):
            probe.update(str(update))
            await pilot.pause()

        app.background.resume_host_refresh()
        await pilot.pause()
        assert host_writes == []

        modal.dismiss()
        await pilot.pause()
        assert app.screen is app.background
        host_writes.clear()

        # A leaked lease must not freeze MainScreen after it becomes active again.
        probe.update("visible despite leaked lease")
        await pilot.pause()
        assert host_writes
        assert all(screen is app.background for screen in host_writes)

        app.background.resume_host_refresh()
        await pilot.pause()


def _make_ask_user_handler(monkeypatch) -> tuple[BackendEventHandler, _FakeAskUserApp, list[tuple[str, str]], list]:
    from collections import deque

    import chrys.app.tui.screens.dialogs.ask_user as _ask_user_mod

    monkeypatch.setattr(_ask_user_mod, "AskUserDialog", _FakeAskUserDialog)

    app = _FakeAskUserApp()
    responses: list[tuple[str, str]] = []
    debug_log: list[tuple[str, str]] = []
    host_refresh = _FakeQuestionHostRefresh()

    screen = SimpleNamespace(
        app=app,
        _handle_ask_user_response=lambda request_id, text: responses.append((request_id, text)),
        _debug=lambda event_type, detail="": debug_log.append((event_type, detail)),
        pause_host_refresh=host_refresh.pause,
        resume_host_refresh=host_refresh.resume,
    )
    handler = make_backend_handler(screen)
    handler._question_queue = deque()
    handler._question_dialog_open = False
    handler._open_question_dialogs = {}
    handler._inline_question_call_ids = {}
    handler._inline_question_request_ids = {}
    handler._question_drafts = {}
    return handler, app, responses, debug_log


def test_ask_user_timeout_dismisses_live_dialog(monkeypatch) -> None:
    from chrys.foundation.events.types import AskUserTimedOut, QuestionToUser

    handler, app, responses, debug_log = _make_ask_user_handler(monkeypatch)

    asyncio.run(handler.on_question_to_user(QuestionToUser(request_id="q1", question="Need input?")))
    dialog, _callback = app.pushed[0]

    asyncio.run(handler.on_ask_user_timed_out(AskUserTimedOut(request_id="q1")))

    assert dialog.dismissed is True
    assert responses == []
    assert handler._open_question_dialogs == {}
    assert handler._question_dialog_open is False
    assert debug_log[-1] == ("AskUserTimedOut", "q1")


def test_ask_user_timeout_removes_queued_dialog(monkeypatch) -> None:
    from chrys.foundation.events.types import AskUserTimedOut, QuestionToUser

    handler, app, responses, _debug_log = _make_ask_user_handler(monkeypatch)

    asyncio.run(handler.on_question_to_user(QuestionToUser(request_id="q1", question="First?")))
    asyncio.run(handler.on_question_to_user(QuestionToUser(request_id="q2", question="Second?")))
    assert len(app.pushed) == 1

    asyncio.run(handler.on_ask_user_timed_out(AskUserTimedOut(request_id="q2")))
    first_dialog, _callback = app.pushed[0]
    first_dialog.submit("answer")

    assert responses == [("q1", "answer")]
    assert len(app.pushed) == 1
    assert handler._question_dialog_open is False


def test_ask_user_inline_releases_dialog_slot_for_next_question(monkeypatch) -> None:
    from collections import deque

    import chrys.app.tui.screens.dialogs.ask_user as _ask_user_mod
    from chrys.foundation.events.types import QuestionToUser

    monkeypatch.setattr(_ask_user_mod, "AskUserDialog", _FakeAskUserDialog)

    class _FakeChatPanel:
        def __init__(self) -> None:
            self.inline_calls: list[tuple[str, str, list[str], str]] = []

        def show_ask_user_inline(
            self,
            call_id: str,
            request_id: str,
            options: list[str] | None = None,
            *,
            draft_text: str = "",
        ) -> bool:
            self.inline_calls.append((call_id, request_id, list(options or []), draft_text))
            return True

    app = _FakeAskUserApp()
    panel = _FakeChatPanel()
    debug_log: list[tuple[str, str]] = []
    host_refresh = _FakeQuestionHostRefresh()
    screen = SimpleNamespace(
        app=app,
        _handle_ask_user_response=lambda *_args: None,
        _debug=lambda event_type, detail="": debug_log.append((event_type, detail)),
        query_one=lambda _cls: panel,
        pause_host_refresh=host_refresh.pause,
        resume_host_refresh=host_refresh.resume,
    )
    handler = make_backend_handler(screen)
    handler._question_queue = deque()
    handler._question_dialog_open = False
    handler._open_question_dialogs = {}
    handler._inline_question_call_ids = {}
    handler._inline_question_request_ids = {}
    handler._question_drafts = {}

    asyncio.run(
        handler.on_question_to_user(QuestionToUser(request_id="q1", call_id="c1", question="First?", options=["A"]))
    )
    asyncio.run(handler.on_question_to_user(QuestionToUser(request_id="q2", call_id="c2", question="Second?")))
    first_dialog, _callback = app.pushed[0]

    first_dialog.answer_inline("draft")

    assert panel.inline_calls == [("c1", "q1", ["A"], "draft")]
    assert handler._inline_question_call_ids == {"q1": "c1"}
    assert handler._inline_question_request_ids == {"c1": "q1"}
    assert len(app.pushed) == 2
    second_dialog, _callback = app.pushed[1]
    assert second_dialog.request_id == "q2"
    assert handler._question_dialog_open is True


def test_ask_user_inline_fallback_preserves_draft_when_tool_is_still_running(monkeypatch) -> None:
    from collections import deque

    import chrys.app.tui.screens.dialogs.ask_user as _ask_user_mod
    from chrys.foundation.events.types import QuestionToUser

    monkeypatch.setattr(_ask_user_mod, "AskUserDialog", _FakeAskUserDialog)

    class _FakeChatPanel:
        def show_ask_user_inline(
            self,
            _call_id: str,
            _request_id: str,
            _options: list[str] | None = None,
            *,
            draft_text: str = "",
        ) -> bool:
            assert draft_text == "draft"
            return False

        def is_tool_running(self, call_id: str) -> bool:
            return call_id == "c1"

    app = _FakeAskUserApp()
    host_refresh = _FakeQuestionHostRefresh()
    screen = SimpleNamespace(
        app=app,
        _handle_ask_user_response=lambda *_args: None,
        _debug=lambda *_args: None,
        query_one=lambda _cls: _FakeChatPanel(),
        pause_host_refresh=host_refresh.pause,
        resume_host_refresh=host_refresh.resume,
    )
    handler = make_backend_handler(screen)
    handler._question_queue = deque()
    handler._question_dialog_open = False
    handler._open_question_dialogs = {}
    handler._inline_question_call_ids = {}
    handler._inline_question_request_ids = {}
    handler._question_drafts = {}

    asyncio.run(handler.on_question_to_user(QuestionToUser(request_id="q1", call_id="c1", question="First?")))
    first_dialog, _callback = app.pushed[0]

    first_dialog.answer_inline("draft")

    assert len(app.pushed) == 2
    fallback_dialog, _callback = app.pushed[1]
    assert fallback_dialog.request_id == "q1"
    assert fallback_dialog.initial_response == "draft"


def test_ask_user_inline_fallback_does_not_reopen_for_finished_tool(monkeypatch) -> None:
    from collections import deque

    import chrys.app.tui.screens.dialogs.ask_user as _ask_user_mod
    from chrys.foundation.events.types import QuestionToUser

    monkeypatch.setattr(_ask_user_mod, "AskUserDialog", _FakeAskUserDialog)

    class _FakeChatPanel:
        def show_ask_user_inline(
            self,
            _call_id: str,
            _request_id: str,
            _options: list[str] | None = None,
            *,
            draft_text: str = "",
        ) -> bool:
            return False

        def is_tool_running(self, _call_id: str) -> bool:
            return False

    app = _FakeAskUserApp()
    host_refresh = _FakeQuestionHostRefresh()
    screen = SimpleNamespace(
        app=app,
        _handle_ask_user_response=lambda *_args: None,
        _debug=lambda *_args: None,
        query_one=lambda _cls: _FakeChatPanel(),
        pause_host_refresh=host_refresh.pause,
        resume_host_refresh=host_refresh.resume,
    )
    handler = make_backend_handler(screen)
    handler._question_queue = deque()
    handler._question_dialog_open = False
    handler._open_question_dialogs = {}
    handler._inline_question_call_ids = {}
    handler._inline_question_request_ids = {}
    handler._question_drafts = {}

    asyncio.run(handler.on_question_to_user(QuestionToUser(request_id="q1", call_id="c1", question="First?")))
    first_dialog, _callback = app.pushed[0]

    first_dialog.answer_inline("draft")

    assert len(app.pushed) == 1
    assert handler._question_dialog_open is False
    assert handler._question_queue == deque()
    assert handler._question_drafts == {}


def test_inline_tool_result_restores_input_focus() -> None:
    from chrys.app.tui.widgets.chat.panel import ChatPanel
    from chrys.app.tui.widgets.chrome.input_bar import InputBar
    from chrys.app.tui.widgets.chrome.status_bar import StatusBar
    from chrys.foundation.events.types import ToolCallResult

    calls: list[str] = []

    class _FakeChatPanel:
        session_id = "s1"

        async def add_tool_result(self, *_args, **_kwargs) -> None:
            calls.append("tool_result")

    class _FakeInputBar:
        def focus_input(self) -> None:
            calls.append("focus_input")

    class _FakeStatusBar:
        def show(self, value: MessageRef | str) -> None:
            calls.append(f"status:{_status_text(value)}")

    panel = _FakeChatPanel()
    input_bar = _FakeInputBar()
    status_bar = _FakeStatusBar()

    def query_one(cls: type) -> object:
        if cls is ChatPanel:
            return panel
        if cls is InputBar:
            return input_bar
        if cls is StatusBar:
            return status_bar
        raise AssertionError(f"unexpected query_one({cls})")

    screen = SimpleNamespace(
        _agent_running=True,
        _state_store=None,
        _live_call_paths={},
        _debug=lambda *_args: None,
        query_one=query_one,
    )
    handler = make_backend_handler(screen)
    handler._inline_question_call_ids = {"q1": "c1"}
    handler._inline_question_request_ids = {"c1": "q1"}
    handler._question_drafts = {"q1": "draft"}
    handler._accumulate_shell_snapshots = lambda _metadata: None

    asyncio.run(handler.on_tool_result(ToolCallResult(call_id="c1", tool_name="ask_user", result="User response: yes")))

    assert calls == ["tool_result", "focus_input", "status:Thinking"]
    assert handler._inline_question_call_ids == {}
    assert handler._inline_question_request_ids == {}
    assert handler._question_drafts == {}


def test_ask_user_timeout_clears_inline_pending_map(monkeypatch) -> None:
    from chrys.foundation.events.types import AskUserTimedOut

    handler, _app, _responses, debug_log = _make_ask_user_handler(monkeypatch)
    handler._inline_question_call_ids = {"q1": "c1"}
    handler._inline_question_request_ids = {"c1": "q1"}
    handler._question_drafts = {"q1": "draft"}

    asyncio.run(handler.on_ask_user_timed_out(AskUserTimedOut(request_id="q1")))

    assert handler._inline_question_call_ids == {}
    assert handler._inline_question_request_ids == {}
    assert handler._question_drafts == {}
    assert debug_log[-1] == ("AskUserTimedOut", "q1 (inline)")


def test_clear_pending_questions_dismisses_dialog_and_clears_queue_and_inline_state(monkeypatch) -> None:
    from chrys.foundation.events.types import QuestionToUser

    handler, app, responses, _debug_log = _make_ask_user_handler(monkeypatch)

    asyncio.run(handler.on_question_to_user(QuestionToUser(request_id="q1", question="First?")))
    asyncio.run(handler.on_question_to_user(QuestionToUser(request_id="q2", question="Second?")))
    dialog, _callback = app.pushed[0]
    handler._inline_question_call_ids = {"q-inline": "c-inline"}
    handler._inline_question_request_ids = {"c-inline": "q-inline"}
    handler._question_drafts = {"q-inline": "draft"}

    handler.clear_pending_questions()

    assert dialog.dismissed is True
    assert responses == []
    assert list(handler._question_queue) == []
    assert handler._open_question_dialogs == {}
    assert handler._inline_question_call_ids == {}
    assert handler._inline_question_request_ids == {}
    assert handler._question_drafts == {}
    assert handler._question_dialog_open is False
    assert len(app.pushed) == 1


def _make_approval_handler(monkeypatch) -> tuple[BackendEventHandler, _FakeApp, list, list]:
    """Build a ``BackendEventHandler`` wired to mocks for approval flow.

    Returns ``(handler, fake_app, debug_log, response_log)``:
    - ``debug_log`` — every ``screen._debug(event_type, detail)`` call.
    - ``response_log`` — every ``screen._handle_approval_response`` call.
    """
    from collections import deque

    # Patch the dialog class the handler imports lazily inside
    # ``_show_next_approval`` so our mock is used in its place.
    import chrys.app.tui.screens.dialogs.approval as _approval_mod

    monkeypatch.setattr(_approval_mod, "ApprovalDialog", _FakeApprovalDialog)

    app = _FakeApp()
    bus = _FakeBus()
    debug_log: list[tuple[str, str]] = []
    response_log: list[tuple[str, bool, str]] = []
    worker_tasks: list[asyncio.Task[object]] = []

    def _debug(event_type: str, detail: str = "") -> None:
        debug_log.append((event_type, detail))

    def _handle_approval_response(
        request_id: str,
        approved: bool,
        reason: str = "",
        _modified_args: dict[str, object] | None = None,
    ) -> None:
        response_log.append((request_id, approved, reason))

    def _run_worker(work, **_kwargs) -> SimpleNamespace:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if hasattr(work, "close"):
                work.close()
            return SimpleNamespace()
        task = loop.create_task(work)
        worker_tasks.append(task)
        return SimpleNamespace(task=task)

    screen = SimpleNamespace(
        app=app,
        _bus=bus,
        _debug=_debug,
        _handle_approval_response=_handle_approval_response,
        run_worker=_run_worker,
        worker_tasks=worker_tasks,
    )
    app.screen = screen
    app.bus = bus
    app.worker_tasks = worker_tasks
    handler = make_backend_handler(screen)
    handler._approval_queue = deque()
    handler._approval_request_lock = asyncio.Lock()
    handler._approval_dialog_open = False
    handler._approval_bodies = {}
    handler._open_approval_dialogs = {}
    handler._pending_verdicts = {}
    handler._dismissed_approval_requests = set()
    handler._reviewed_dismissed_approval_requests = set()
    return handler, app, debug_log, response_log


def _make_request(
    request_id: str,
    tool_name: str = "zsh",
    judging: bool = True,
    tool_kind: str = "shell",
    args: dict | None = None,
    call_id: str = "",
):
    from chrys.foundation.events.types import ApprovalRequest

    return ApprovalRequest(
        request_id=request_id,
        call_id=call_id,
        tool_name=tool_name,
        tool_kind=tool_kind,
        args=args or {"command": "ls"},
        judging=judging,
        caller_name="Explore Agent",
    )


def _make_reviewed(request_id: str, approved: bool, reason: str = ""):
    from chrys.foundation.events.types import ApprovalReviewed

    return ApprovalReviewed(request_id=request_id, approved=approved, reason=reason)


def test_modified_approval_args_refresh_visible_tool_card(monkeypatch) -> None:
    handler, app, _debug_log, _response_log = _make_approval_handler(monkeypatch)
    updates: list[tuple[str, dict[str, object]]] = []

    class _FakePanel:
        def update_tool_args(self, call_id: str, args: dict[str, object]) -> None:
            updates.append((call_id, args))

    app.screen.query_one = lambda _cls: _FakePanel()

    asyncio.run(
        handler.on_approval_request(
            _make_request(
                "req-1",
                tool_name="explore_agent",
                tool_kind="sub_agent",
                args={"prompt": "old"},
                call_id="call-1",
                judging=False,
            )
        )
    )
    _dialog, on_result = app.pushed[0]

    on_result((True, "", {"prompt": "new"}))

    assert updates == [("call-1", {"prompt": "new"})]


def test_final_tool_args_update_refreshes_visible_tool_card(monkeypatch) -> None:
    from chrys.foundation.events.types import ToolCallArgsUpdated

    handler, app, debug_log, _response_log = _make_approval_handler(monkeypatch)
    handler.set_agent_running(True)
    updates: list[tuple[str, dict[str, object]]] = []

    class _FakePanel:
        def update_tool_args(self, call_id: str, args: dict[str, object]) -> None:
            updates.append((call_id, args))

    app.screen.query_one = lambda _cls: _FakePanel()

    asyncio.run(
        handler.on_tool_args_updated(
            ToolCallArgsUpdated(
                tool_name="explore_agent",
                tool_kind="sub_agent",
                call_id="call-1",
                args={"prompt": "hook rewritten"},
            )
        )
    )

    assert updates == [("call-1", {"prompt": "hook rewritten"})]
    assert debug_log[-1] == ("ToolCallArgsUpdated", "explore_agent")


# ──────────── baseline: verdict reaches live dialog ────────────────────


def test_approval_verdict_delivered_to_live_dialog(monkeypatch) -> None:
    """Judge finishes *after* its dialog is mounted — verdict goes straight
    to ``dialog.receive_verdict`` (existing happy path, must still work)."""
    import asyncio

    handler, app, _debug, _ = _make_approval_handler(monkeypatch)

    asyncio.run(handler.on_approval_request(_make_request("req-1")))
    assert len(app.pushed) == 1
    dialog, _cb = app.pushed[0]

    asyncio.run(handler.on_approval_reviewed(_make_reviewed("req-1", approved=True)))

    assert dialog.received_verdict is not None
    assert dialog.received_verdict.approved is True
    # Nothing got cached — dialog handled it directly.
    assert handler._pending_verdicts == {}


# ──────────── approved pre-mount → dialog skipped ──────────────────────


def test_approved_pre_mount_verdict_skips_dialog(monkeypatch) -> None:
    """Judge for a queued request finishes before its dialog is pushed.
    When the preceding dialog dismisses, the queued request is dropped
    entirely (tool already ran in the backend)."""
    import asyncio

    handler, app, debug_log, _response_log = _make_approval_handler(monkeypatch)

    # Two parallel requests — first shows a dialog, second queues.
    asyncio.run(handler.on_approval_request(_make_request("req-1", tool_name="zsh")))
    asyncio.run(handler.on_approval_request(_make_request("req-2", tool_name="grep")))
    assert len(app.pushed) == 1
    assert len(handler._approval_queue) == 1

    # Judge approves req-2 before its dialog is shown → verdict cached.
    asyncio.run(handler.on_approval_reviewed(_make_reviewed("req-2", approved=True)))
    assert "req-2" in handler._pending_verdicts

    # User/judge dismisses dialog 1.  The result callback on the first
    # pushed dialog drives the next-draining logic.
    _dialog1, on_result = app.pushed[0]
    on_result((True, "", None))

    # Dialog 2 must NOT have been pushed (approved pre-mount → skipped).
    assert len(app.pushed) == 1
    # Cache drained and queue empty.
    assert handler._pending_verdicts == {}
    assert len(handler._approval_queue) == 0
    # Flag cleared because no dialog remains open.
    assert handler._approval_dialog_open is False
    # Debug log records the pre-mount skip for req-2.
    assert any(evt == "ApprovalJudge" and "pre-mount" in detail and "grep" in detail for evt, detail in debug_log)


# ──────────── flagged pre-mount → dialog shown with verdict ────────────


def test_flagged_pre_mount_verdict_applied_after_mount(monkeypatch) -> None:
    """Judge flags a queued request before its dialog is pushed.  When the
    dialog eventually mounts, the concern is delivered via
    ``call_after_refresh`` (not synchronously — ``query_one`` on an
    unmounted widget would raise)."""
    import asyncio

    handler, app, _debug, _ = _make_approval_handler(monkeypatch)

    asyncio.run(handler.on_approval_request(_make_request("req-1")))
    asyncio.run(handler.on_approval_request(_make_request("req-2", tool_name="rm")))

    # Judge flags req-2 while still queued.
    asyncio.run(handler.on_approval_reviewed(_make_reviewed("req-2", approved=False, reason="rm -rf")))
    assert "req-2" in handler._pending_verdicts

    # Dismiss dialog 1 → drains queue → dialog 2 is pushed.
    _dialog1, on_result_1 = app.pushed[0]
    on_result_1((True, "", None))

    assert len(app.pushed) == 2
    dialog2, _cb2 = app.pushed[1]

    # Verdict is NOT delivered synchronously (would race the mount); it's
    # scheduled via call_after_refresh.
    assert dialog2.received_verdict is None
    assert len(dialog2.after_refresh_calls) == 1
    fn, args = dialog2.after_refresh_calls[0]
    # Simulate mount completing — the scheduled call fires.
    fn(*args)
    assert dialog2.received_verdict is not None
    assert dialog2.received_verdict.approved is False
    assert "rm -rf" in dialog2.received_verdict.reason
    # Pending verdict was consumed.
    assert "req-2" not in handler._pending_verdicts


# ──────────── late arrival after resolution → dropped ──────────────────


def test_reviewed_after_dismissal_is_dropped(monkeypatch) -> None:
    """A verdict that arrives after its request has already been resolved
    (dialog dismissed and not in queue) is silently dropped — not cached."""
    import asyncio

    handler, app, _debug, _ = _make_approval_handler(monkeypatch)

    asyncio.run(handler.on_approval_request(_make_request("req-1")))
    _dialog1, on_result = app.pushed[0]
    # User dismisses before the judge finishes.
    on_result((True, "", None))
    assert handler._open_approval_dialogs == {}
    assert len(handler._approval_queue) == 0

    # Late verdict arrives — should NOT be cached (request is gone).
    asyncio.run(handler.on_approval_reviewed(_make_reviewed("req-1", approved=True)))
    assert handler._pending_verdicts == {}


def test_approved_review_after_user_click_before_result_callback_blocks_auto_fulfill(monkeypatch) -> None:
    """Window A: dialog is dismissed by the user but still registered."""
    import asyncio

    from chrys.foundation.events.types import ApprovalAutoFulfillBlocked

    handler, app, _debug, _response_log = _make_approval_handler(monkeypatch)

    asyncio.run(handler.on_approval_request(_make_request("req-1")))
    dialog, on_result = app.pushed[0]
    dialog._dismissed = True
    dialog._user_decision_submitted = True

    asyncio.run(handler.on_approval_reviewed(_make_reviewed("req-1", approved=True)))

    published = app.bus.published
    assert len(published) == 1
    assert isinstance(published[0], ApprovalAutoFulfillBlocked)
    assert published[0].request_id == "req-1"

    on_result((False, "use safer path", None))
    assert handler._dismissed_approval_requests == set()
    assert handler._reviewed_dismissed_approval_requests == set()


def test_approved_review_after_result_callback_before_response_publish_blocks_auto_fulfill(monkeypatch) -> None:
    """Window B: dialog callback ran, but ApprovalResponse worker may still be pending."""
    import asyncio

    from chrys.foundation.events.types import ApprovalAutoFulfillBlocked

    handler, app, _debug, _response_log = _make_approval_handler(monkeypatch)

    asyncio.run(handler.on_approval_request(_make_request("req-1")))
    dialog, on_result = app.pushed[0]
    dialog._user_decision_submitted = True
    on_result((False, "use safer path", None))

    asyncio.run(handler.on_approval_reviewed(_make_reviewed("req-1", approved=True)))

    published = app.bus.published
    assert len(published) == 1
    assert isinstance(published[0], ApprovalAutoFulfillBlocked)
    assert published[0].request_id == "req-1"
    assert handler._dismissed_approval_requests == set()


def test_manual_user_decision_does_not_track_dismissed_request(monkeypatch) -> None:
    """MANUAL approvals never receive judge reviews, so no race marker is needed."""
    import asyncio

    handler, app, _debug, _response_log = _make_approval_handler(monkeypatch)

    asyncio.run(handler.on_approval_request(_make_request("req-1", judging=False)))
    dialog, on_result = app.pushed[0]
    dialog._user_decision_submitted = True

    on_result((False, "manual decline", None))

    assert handler._dismissed_approval_requests == set()
    assert handler._reviewed_dismissed_approval_requests == set()


def test_auto_user_decision_marker_clears_when_response_worker_finishes(monkeypatch) -> None:
    """AUTO race markers are only kept while the ApprovalResponse worker is pending."""

    async def _run() -> None:
        handler, app, _debug, response_log = _make_approval_handler(monkeypatch)

        class _Worker:
            def __init__(self) -> None:
                self._done = asyncio.Event()

            async def wait(self) -> None:
                await self._done.wait()

            def finish(self) -> None:
                self._done.set()

        response_worker = _Worker()

        def _handle_approval_response(
            request_id: str,
            approved: bool,
            reason: str = "",
            _modified_args: dict[str, object] | None = None,
        ) -> _Worker:
            response_log.append((request_id, approved, reason))
            return response_worker

        app.screen._handle_approval_response = _handle_approval_response

        await handler.on_approval_request(_make_request("req-1", judging=True))
        dialog, on_result = app.pushed[0]
        dialog._user_decision_submitted = True

        on_result((False, "auto decline", None))

        assert handler._dismissed_approval_requests == {"req-1"}
        response_worker.finish()
        await asyncio.gather(*app.worker_tasks)

        assert handler._dismissed_approval_requests == set()

    asyncio.run(_run())


# ──────────── out-of-order parallel judges ─────────────────────────────


def test_parallel_judges_finish_out_of_order(monkeypatch) -> None:
    """Three parallel requests; judges fire for req-2 (approved) and req-3
    (flagged) while req-1's dialog is still visible.  Dismissing req-1
    drains the queue: req-2 is skipped silently, req-3 shows a dialog with
    the flag concern ready to deliver after mount."""
    import asyncio

    handler, app, _debug, _ = _make_approval_handler(monkeypatch)

    asyncio.run(handler.on_approval_request(_make_request("req-1", tool_name="t1")))
    asyncio.run(handler.on_approval_request(_make_request("req-2", tool_name="t2")))
    asyncio.run(handler.on_approval_request(_make_request("req-3", tool_name="t3")))
    assert len(app.pushed) == 1  # only req-1 mounted

    # Judges finish out of order.
    asyncio.run(handler.on_approval_reviewed(_make_reviewed("req-2", approved=True)))
    asyncio.run(handler.on_approval_reviewed(_make_reviewed("req-3", approved=False, reason="danger")))
    assert set(handler._pending_verdicts.keys()) == {"req-2", "req-3"}

    # Dismiss dialog 1 (judge eventually approves it too).
    _d1, on_result_1 = app.pushed[0]
    on_result_1((True, "", None))

    # req-2 skipped, req-3 pushed with flagged verdict pending.
    assert len(app.pushed) == 2
    d3, _ = app.pushed[1]
    assert d3._tool_name == "t3"
    # Flagged verdict delivered after mount (not synchronously).
    assert d3.received_verdict is None
    assert len(d3.after_refresh_calls) == 1
    fn, args = d3.after_refresh_calls[0]
    fn(*args)
    assert d3.received_verdict.approved is False
    assert d3.received_verdict.reason == "danger"

    # Cache fully drained.
    assert handler._pending_verdicts == {}
    # Dialog 3 is the only one still open — flag stays True until user acts.
    assert handler._approval_dialog_open is True

    # Dismiss dialog 3 → queue empty → flag cleared.
    _d3, on_result_3 = app.pushed[1]
    on_result_3((False, "", None))
    assert handler._approval_dialog_open is False


# ──────────── MANUAL mode regression ───────────────────────────────────


def test_manual_mode_shows_dialogs_sequentially(monkeypatch) -> None:
    """MANUAL mode (no judge, no ApprovalReviewed events) still queues
    and shows dialogs one at a time — each user decision drains the next.
    Guards the refactored ``_show_next_approval`` loop against breaking the
    no-cache path."""
    import asyncio

    handler, app, _debug, response_log = _make_approval_handler(monkeypatch)

    # Three MANUAL requests (judging=False) — no judge verdicts will ever
    # arrive; the TUI must still serialize the dialogs.
    asyncio.run(handler.on_approval_request(_make_request("req-1", judging=False)))
    asyncio.run(handler.on_approval_request(_make_request("req-2", judging=False)))
    asyncio.run(handler.on_approval_request(_make_request("req-3", judging=False)))

    assert len(app.pushed) == 1
    assert len(handler._approval_queue) == 2
    assert handler._pending_verdicts == {}

    # User approves each one — every dismissal must push the next queued.
    _d1, cb1 = app.pushed[0]
    cb1((True, "", None))
    assert len(app.pushed) == 2

    _d2, cb2 = app.pushed[1]
    cb2((False, "", None))
    assert len(app.pushed) == 3

    _d3, cb3 = app.pushed[2]
    cb3((True, "", None))
    assert len(app.pushed) == 3
    assert handler._approval_dialog_open is False
    assert len(handler._approval_queue) == 0

    # All three responses were published back to the engine with the
    # correct approve/decline outcomes.
    assert response_log == [
        ("req-1", True, ""),
        ("req-2", False, ""),
        ("req-3", True, ""),
    ]


def test_write_file_conflict_skips_approval_dialog(monkeypatch, tmp_path) -> None:
    import asyncio

    from chrys.foundation.tool_kinds import KIND_FILESYSTEM_WRITE

    target = tmp_path / "exists.txt"
    target.write_text("old", encoding="utf-8")
    handler, app, debug_log, response_log = _make_approval_handler(monkeypatch)

    asyncio.run(
        handler.on_approval_request(
            _make_request(
                "req-1",
                tool_name="write_file",
                judging=False,
                tool_kind=KIND_FILESYSTEM_WRITE,
                args={"path": str(target), "content": "new"},
            )
        )
    )

    assert app.pushed == []
    assert response_log == [("req-1", True, "")]
    assert len(handler._approval_queue) == 0
    assert any("skipped dialog: exists" in detail for event, detail in debug_log if event == "ApprovalRequest")


def test_write_file_conflict_for_non_filesystem_kind_shows_generic_dialog(monkeypatch, tmp_path) -> None:
    import asyncio

    from chrys.foundation.tool_kinds import KIND_MCP

    target = tmp_path / "exists.txt"
    target.write_text("old", encoding="utf-8")
    handler, app, debug_log, response_log = _make_approval_handler(monkeypatch)

    asyncio.run(
        handler.on_approval_request(
            _make_request(
                "req-1",
                tool_name="write_file",
                judging=False,
                tool_kind=KIND_MCP,
                args={"path": str(target), "content": "new"},
            )
        )
    )

    assert response_log == []
    assert len(app.pushed) == 1
    dialog, _callback = app.pushed[0]
    assert dialog.approval_body is None
    assert not any("skipped dialog" in detail for event, detail in debug_log if event == "ApprovalRequest")


def test_edit_file_not_found_skips_approval_dialog(monkeypatch, tmp_path) -> None:
    import asyncio

    from chrys.foundation.tool_kinds import KIND_FILESYSTEM_WRITE

    target = tmp_path / "edit.txt"
    target.write_text("hello\n", encoding="utf-8")
    handler, app, debug_log, response_log = _make_approval_handler(monkeypatch)

    asyncio.run(
        handler.on_approval_request(
            _make_request(
                "req-1",
                tool_name="edit_file",
                judging=False,
                tool_kind=KIND_FILESYSTEM_WRITE,
                args={"path": str(target), "old_string": "missing", "new_string": "replacement"},
            )
        )
    )

    assert app.pushed == []
    assert response_log == [("req-1", True, "")]
    assert target.read_text(encoding="utf-8") == "hello\n"
    assert any("skipped dialog: not_found" in detail for event, detail in debug_log if event == "ApprovalRequest")


def test_write_file_diff_body_is_passed_to_dialog(monkeypatch, tmp_path) -> None:
    import asyncio

    from chrys.foundation.tool_kinds import KIND_FILESYSTEM_WRITE

    target = tmp_path / "new.txt"
    handler, app, _debug_log, _response_log = _make_approval_handler(monkeypatch)

    asyncio.run(
        handler.on_approval_request(
            _make_request(
                "req-1",
                tool_name="write_file",
                judging=False,
                tool_kind=KIND_FILESYSTEM_WRITE,
                args={"path": str(target), "content": "hello\n"},
            )
        )
    )

    assert len(app.pushed) == 1
    dialog, _callback = app.pushed[0]
    assert dialog.approval_body is not None
    assert dialog.approval_body.hidden_arg_keys == frozenset({"content"})
    assert len(dialog.approval_body.widgets) == 1


# ──────────── injection outcome unlock safety ────────────────────────────


def test_consumed_injection_unlocks_even_if_chat_render_fails() -> None:
    """A consumed injection must not leave the input bar locked if rendering fails."""
    calls: list[object] = []

    class _FakeInputBar:
        def __init__(self) -> None:
            self.locked = True

        def unlock_and_clear(self) -> None:
            calls.append("unlock_and_clear")
            self.locked = False

    class _FakeChatPanel:
        async def add_user_message(self, text: str, *, is_injection: bool = False, **_kwargs: object) -> None:
            calls.append(("add_user_message", text, is_injection))
            raise RuntimeError("render failed")

    input_bar = _FakeInputBar()
    panel = _FakeChatPanel()

    def query_one(cls: type):
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ChatPanel":
            return panel
        raise AssertionError(f"unexpected query_one({cls})")

    screen = SimpleNamespace(
        query_one=query_one,
        _update_toc=lambda: calls.append("update_toc"),
        _debug=lambda *_args: calls.append("debug"),
    )
    handler = make_backend_handler(screen)

    async def _run() -> None:
        await handler.on_injection_outcome(UserInjectResult(text="queued text", consumed=True))

    try:
        asyncio.run(_run())
    except RuntimeError as exc:
        assert str(exc) == "render failed"
    else:
        raise AssertionError("render failure should propagate")

    assert input_bar.locked is False
    assert calls == [
        ("add_user_message", "queued text", True),
        "unlock_and_clear",
    ]


def _make_injection_outcome_screen(calls: list[object]) -> SimpleNamespace:
    """Mock screen capturing input-bar and chat-panel effects of injection results."""

    class _FakeInputBar:
        def __init__(self) -> None:
            self.locked = True

        def unlock_and_clear(self) -> None:
            calls.append("unlock_and_clear")
            self.locked = False

        def unlock_and_keep(self) -> None:
            calls.append("unlock_and_keep")
            self.locked = False

    class _FakeChatPanel:
        async def add_user_message(self, text: str, *, is_injection: bool = False, **_kwargs: object) -> None:
            calls.append(("add_user_message", text, is_injection))

    input_bar = _FakeInputBar()
    panel = _FakeChatPanel()

    def query_one(cls: type):
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ChatPanel":
            return panel
        raise AssertionError(f"unexpected query_one({cls})")

    return SimpleNamespace(
        query_one=query_one,
        _state=MainScreenState(),
        _update_toc=lambda: None,
        _debug=lambda *_args: None,
    )


def test_stale_abandoned_injection_result_leaves_input_untouched() -> None:
    """An abandoned result for a cancelled id must not disturb re-editing."""
    calls: list[object] = []
    screen = _make_injection_outcome_screen(calls)
    handler = make_backend_handler(screen)
    # The user Esc-cancelled and requeued: a NEW injection is now pending.
    screen._state.pending_injection.begin("new-id", "edited text")

    asyncio.run(handler.on_injection_outcome(UserInjectResult(text="old text", consumed=False, injection_id="old-id")))

    assert calls == []
    assert screen._state.pending_injection.matches("new-id")


def test_stale_consumed_injection_result_renders_bubble_only() -> None:
    """A consumed result for a cancelled id shows the bubble but keeps the input."""
    calls: list[object] = []
    screen = _make_injection_outcome_screen(calls)
    handler = make_backend_handler(screen)
    # The user Esc-cancelled (pending cleared) before the consumed result landed.

    asyncio.run(
        handler.on_injection_outcome(UserInjectResult(text="delivered anyway", consumed=True, injection_id="old-id"))
    )

    assert calls == [("add_user_message", "delivered anyway", True)]
    assert screen._state.pending_injection.active is False


def test_matching_abandoned_injection_result_unlocks_and_clears_pending() -> None:
    """The tracked injection's abandoned result restores the input for reuse."""
    calls: list[object] = []
    screen = _make_injection_outcome_screen(calls)
    handler = make_backend_handler(screen)
    screen._state.pending_injection.begin("inj-1", "queued text")

    asyncio.run(
        handler.on_injection_outcome(UserInjectResult(text="queued text", consumed=False, injection_id="inj-1"))
    )

    assert calls == ["unlock_and_keep"]
    assert screen._state.pending_injection.active is False


def test_matching_consumed_injection_result_unlocks_and_clears_pending() -> None:
    """The tracked injection's consumed result renders and clears the input."""
    calls: list[object] = []
    screen = _make_injection_outcome_screen(calls)
    handler = make_backend_handler(screen)
    screen._state.pending_injection.begin("inj-1", "queued text")

    asyncio.run(handler.on_injection_outcome(UserInjectResult(text="queued text", consumed=True, injection_id="inj-1")))

    assert calls == [("add_user_message", "queued text", True), "unlock_and_clear"]
    assert screen._state.pending_injection.active is False


# ──────────── _set_agent_running file-cache invalidation ───────────────


def _make_screen_for_running_toggle() -> SimpleNamespace:
    """Mock screen with the attributes ``_set_agent_running`` reads/writes."""
    from chrys.app.tui.widgets.chat.panel import ChatPanel
    from chrys.app.tui.widgets.chrome.input_bar import InputBar

    live_call_paths: dict[str, str] = {}
    live_file_mutations: dict[str, LiveFileMutation] = {}
    input_bar = SimpleNamespace(
        agent_running=False,
        locked=False,
        unlock_and_keep=lambda: None,
    )
    chat_panel = SimpleNamespace(agent_running=False)
    engine = SimpleNamespace(session_generation=1)

    def query_one(cls):
        if cls is InputBar:
            return input_bar
        if cls is ChatPanel:
            return chat_panel
        raise AssertionError(f"unexpected query_one({cls})")

    return SimpleNamespace(
        _state=MainScreenState(),
        _agent_running=False,
        _agent_loading=False,
        _terminal_title_activity_frame=0,
        _terminal_title_result="",
        _sync_terminal_title_activity=lambda: None,
        _live_diff=LiveDiffTracker(call_paths=live_call_paths, file_mutations=live_file_mutations),
        _live_call_paths=live_call_paths,
        _live_file_mutations=live_file_mutations,
        _suggestions=SimpleNamespace(file_cache=None),
        _interrupt_confirm_active=False,
        _dismiss_interrupt_confirm=lambda: None,
        _engine=engine,
        _services=MainScreenServices(bus=EventBus(), engine_provider=lambda: engine),
        _view_adapter=SimpleNamespace(current_chat_session_id=lambda: "session-1"),
        query_one=query_one,
        refresh_bindings=lambda: None,
    )


def test_set_agent_running_false_invalidates_file_cache() -> None:
    """When the agent stops, the ``@`` file cache must be dropped.

    Agent tool calls (``write_file``/``edit_file``/shell) can create or
    delete files during a turn; without invalidation the next ``@``
    trigger would show a stale list.  ``_set_agent_running`` is the
    single chokepoint for all stop transitions (normal completion,
    error, user interrupt).
    """
    from chrys.app.tui.screens.main.screen import MainScreen

    screen = _make_screen_for_running_toggle()
    screen._suggestions.file_cache = _stale_file_cache("src/a.py", "src/b.py")  # prior @ scan
    screen._agent_running = True

    MainScreen._set_agent_running(screen, False)

    assert screen._suggestions.file_cache is None
    assert screen._agent_running is False


def test_set_agent_running_true_preserves_file_cache() -> None:
    """Cache is invalidated only on stop — starting a turn keeps it intact.

    The cache is per-turn staleness: we don't want to rebuild on every
    user prompt, only after the agent has had a chance to mutate the
    filesystem.
    """
    from chrys.app.tui.screens.main.screen import MainScreen

    screen = _make_screen_for_running_toggle()
    cached = _stale_file_cache("src/a.py", "src/b.py")
    screen._suggestions.file_cache = cached

    MainScreen._set_agent_running(screen, True)

    assert screen._suggestions.file_cache is cached
    assert screen._agent_running is True


def test_running_generation_changes_only_when_a_new_turn_starts() -> None:
    from chrys.app.tui.screens.main.screen import MainScreen

    screen = _make_screen_for_running_toggle()
    MainScreen._set_agent_running(screen, True)
    assert screen._state.run.generation == 1
    screen._live_file_mutations["/repo/live.py"] = _live("before", "after", "modify")
    cached = _stale_file_cache("src/warm.py")
    screen._suggestions.file_cache = cached

    MainScreen._set_agent_running(screen, True)
    assert screen._state.run.generation == 1
    assert "/repo/live.py" in screen._live_file_mutations
    assert screen._suggestions.file_cache is cached

    MainScreen._set_agent_running(screen, False)
    MainScreen._set_agent_running(screen, True)
    assert screen._state.run.generation == 2
    assert screen._live_file_mutations == {}


def test_publish_interrupt_publishes_before_marking_agent_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sleep ToolCallResult handlers must still see the screen as running."""
    from textual import _time, events

    bus = EventBus()
    observed_running: list[bool] = []
    order: list[str] = []
    input_events: list[events.Key] = []

    class _RunningStates(list[bool]):
        def append(self, value: bool) -> None:
            order.append("idle")
            super().append(value)

    class _GcMessages(list[object]):
        def append(self, message: object) -> None:
            order.append("gc")
            super().append(message)

    class _OrderedView(_FakeInputFlowView):
        def clear_terminal_title_result(self) -> None:
            order.append("clear-title")
            super().clear_terminal_title_result()

        async def render_interrupted(self) -> None:
            assert state.run.agent_running is True
            input_events.append(events.Key("x", "x"))
            order.append("render")
            await super().render_interrupted()

    running = _RunningStates()
    state = MainScreenState()
    state.run.agent_running = True
    view = _OrderedView()
    view.gc_messages = _GcMessages()
    controller, state, view = _make_input_flow_controller(bus, state=state, view=view, running=running)

    async def _on_interrupt(_event: UserInterrupt) -> None:
        observed_running.append(state.run.agent_running)

    async def _run() -> None:
        await bus.subscribe(UserInterrupt, _on_interrupt)
        await controller.publish_interrupt()

    source_times = iter([10.0, 20.0])
    monkeypatch.setattr(_time, "get_time", lambda: next(source_times))
    asyncio.run(_run())

    assert observed_running == [True]
    assert "interrupted" in view.calls
    assert "clear_questions" in view.calls
    assert running == [False]
    assert state.run.agent_running is False
    assert order == ["render", "clear-title", "idle", "gc"]
    assert "clear_title_result" in view.calls
    assert len(view.gc_messages) == 1
    assert isinstance(view.gc_messages[0], GcAbsorbRequested)
    assert view.gc_messages[0].reason is GcAbsorbReason.TURN_TERMINAL
    assert view.gc_messages[0].terminal_boundary is True
    assert view.gc_messages[0].time == 10.0
    assert input_events[0].time == 20.0


def test_publish_interrupt_reasserts_flash_after_idle_gate_closes() -> None:
    """A teardown event that flips the status bar back to run mode while the
    interrupt awaits (``agent_running`` still True) must be overwritten by a
    second interrupted flash after the gate closes."""
    bus = EventBus()
    order: list[str] = []

    class _RunningStates(list[bool]):
        def append(self, value: bool) -> None:
            order.append("idle")
            super().append(value)

    class _RaceView(_FakeInputFlowView):
        def flash_interrupted(self) -> None:
            order.append("flash")
            super().flash_interrupted()

        async def render_interrupted(self) -> None:
            # Simulate a CompactionFinished/late ToolCallResult handler racing
            # the teardown window and re-showing the run-mode status while
            # ``agent_running`` is still True.
            order.append("raced-show-status")
            await super().render_interrupted()

    state = MainScreenState()
    state.run.agent_running = True
    controller, state, view = _make_input_flow_controller(bus, state=state, view=_RaceView(), running=_RunningStates())

    asyncio.run(controller.publish_interrupt())

    assert order == ["flash", "raced-show-status", "idle", "flash"]
    assert "clear_title_result" in view.calls
    assert view.calls.count("flash_interrupted") == 2


def test_publish_interrupt_render_failure_releases_turn_without_absorb() -> None:
    bus = EventBus()
    running: list[bool] = []

    class _FailingView(_FakeInputFlowView):
        async def render_interrupted(self) -> None:
            self.calls.append("interrupted")
            raise RuntimeError("render failed")

    state = MainScreenState()
    state.run.agent_running = True
    state.pending_injection.begin("pending-id", "queued")
    view = _FailingView()
    controller, state, view = _make_input_flow_controller(bus, state=state, view=view, running=running)

    with pytest.raises(RuntimeError, match="render failed"):
        asyncio.run(controller.publish_interrupt())

    assert running == [False]
    assert state.run.agent_running is False
    assert state.pending_injection.active is False
    assert "unlock_keep" in view.calls
    assert "clear_title_result" in view.calls
    assert view.gc_messages == []
    assert ("retry_mode", True, "Continue") not in view.calls


def test_inline_ask_user_submit_publishes_response() -> None:
    from chrys.app.tui.widgets.chat.renderers.ask_user import AskUserInlineSubmitted
    from chrys.foundation.events.types import AskUserResponse

    bus = EventBus()
    responses: list[AskUserResponse] = []
    screen = object.__new__(MainScreen)
    screen._debug = lambda *_args: None
    screen._tool_actions = ToolActionBridge(publisher=bus, debug=screen._debug)

    async def _collect(event: AskUserResponse) -> None:
        responses.append(event)

    async def _run() -> None:
        await bus.subscribe(AskUserResponse, _collect)
        await screen.on_ask_user_inline_submitted(AskUserInlineSubmitted("c1", "q1", "Python"))

    asyncio.run(_run())

    assert len(responses) == 1
    assert responses[0].request_id == "q1"
    assert responses[0].text == "Python"


def test_agent_loading_does_not_hide_footer_bindings() -> None:
    """Loading modal blocks interaction; footer bindings should stay visually stable."""
    from chrys.app.tui.screens.main.screen import MainScreen

    screen = object.__new__(MainScreen)
    screen._fullscreen_terminal = False
    screen._shell_mode = False
    screen._agent_loading = True
    screen._agent_running = False
    screen._dashboard_visible = lambda: False

    assert MainScreen.check_action(screen, "sessions", ()) is True
    assert MainScreen.check_action(screen, "agents_config", ()) is True
    assert MainScreen.check_action(screen, "models_config", ()) is True
    assert MainScreen.check_action(screen, "show_log_viewer", ()) is True
    assert MainScreen.check_action(screen, "pick_theme", ()) is True
    assert MainScreen.check_action(screen, "settings", ()) is True


def test_history_scope_footer_binding_removed() -> None:
    """Prompt history should not reserve Ctrl+H because it collides with Backspace."""
    from chrys.app.tui.screens.main.screen import MainScreen
    from chrys.app.tui.widgets.chrome.input_bar import _ChatTextArea
    from chrys.foundation.events.bus import EventBus

    screen = MainScreen(EventBus(), engine_provider=None)

    assert all(binding.key != "ctrl+h" for binding in _ChatTextArea.BINDINGS)
    assert "ctrl+h" not in screen._bindings.key_to_bindings


def test_prompt_history_uses_hidden_ctrl_r_binding() -> None:
    """Ctrl+R opens prompt history without adding another footer item."""
    from chrys.app.tui.screens.main.screen import MainScreen
    from chrys.foundation.events.bus import EventBus

    screen = MainScreen(EventBus(), engine_provider=None)
    binding = next(binding for binding in MainScreen.BINDINGS if binding.key == "ctrl+r")

    assert binding.action == "prompt_history"
    assert binding.show is False
    assert binding.priority is True
    assert "ctrl+r" in screen._bindings.key_to_bindings
    assert all(binding.key != "ctrl+t" for binding in MainScreen.BINDINGS)
    assert "ctrl+t" not in screen._bindings.key_to_bindings
    assert "action_toggle_toc" not in MainScreen.__dict__


@pytest.mark.parametrize(
    ("fullscreen_terminal", "shell_mode", "dashboard_visible"),
    [(True, False, False), (False, True, False), (False, False, True)],
    ids=["fullscreen-terminal", "shell-mode", "trajectory-dashboard"],
)
def test_prompt_history_action_enforces_hidden_binding_availability(
    fullscreen_terminal: bool,
    shell_mode: bool,
    dashboard_visible: bool,
) -> None:
    """Hidden bindings still dispatch, so the action must enforce overlay availability."""
    from chrys.app.tui.screens.main.screen import MainScreen

    screen = object.__new__(MainScreen)
    screen._fullscreen_terminal = fullscreen_terminal
    screen._shell_mode = shell_mode
    screen._dashboard_visible = lambda: dashboard_visible
    screen.query_one = lambda _widget: (_ for _ in ()).throw(AssertionError("input bar must not be queried"))

    MainScreen.action_prompt_history(screen)


def test_session_in_use_error_uses_modal_not_chat_or_status() -> None:
    """Session ownership conflicts should be modal-only, not chat/status noise."""
    from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog

    pushed: list[object] = []
    debug_calls: list[tuple[str, str]] = []
    running: list[bool] = []

    def query_one(_cls: object) -> object:
        raise AssertionError("session_in_use should not query chat, status, or input widgets")

    def push_screen(dialog: object) -> None:
        pushed.append(dialog)

    def set_agent_running(value: bool) -> None:
        running.append(value)

    def debug(key: str, msg: str) -> None:
        debug_calls.append((key, msg))

    screen = SimpleNamespace(
        _restoring_session=True,
        _agent_loading=False,
        **_pending_submit_defaults(),
        app=SimpleNamespace(push_screen=push_screen),
        _set_agent_running=set_agent_running,
        query_one=query_one,
        _debug=debug,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None

    message = f"Session '40d9a0483e08' is already open in another {APP_DISPLAY_NAME} instance (pid=47219)."
    asyncio.run(handler.on_error(Error(code="session_in_use", message=message)))

    assert screen._restoring_session is False
    assert running == [False]
    assert len(pushed) == 1
    dialog = pushed[0]
    assert isinstance(dialog, ConfirmDialog)
    assert dialog._title == "Session In Use"
    assert dialog._message.plain == f"Session Already Open\n\n{message}"
    assert dialog._message.spans[0].start == 0
    assert dialog._message.spans[0].end == len("Session Already Open")
    assert str(dialog._message.spans[0].style) == "bold"
    assert dialog._confirm_label == "OK"
    assert dialog._cancel_label is None
    assert dialog._confirm_variant == "warning"
    assert dialog.has_class("-warning-border")
    assert debug_calls and debug_calls[0][0] == "Error"


def test_session_in_use_error_dismisses_active_load_modal() -> None:
    """A restore ownership conflict should close loading UI before showing the standard modal."""
    from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog

    pushed: list[object] = []
    debug_calls: list[tuple[str, str]] = []
    running: list[bool] = []
    loading: list[bool] = []
    status_snapshot = {"visible": True, "flash": None, "status": "Idle"}
    status_restores: list[dict[str, object]] = []

    class _FakeDialog:
        def __init__(self) -> None:
            self.dismissed = False
            self.result_calls: list[tuple[bool, str, bool]] = []

        def dismiss(self, _result: object = None) -> None:
            self.dismissed = True

        def set_result(self, success: bool, message: str, allow_esc: bool = False) -> None:
            self.result_calls.append((success, message, allow_esc))

    class _FakeStatusBar:
        def restore(self, state: dict[str, object]) -> None:
            status_restores.append(state)

    def query_one(cls: object) -> object:
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    def push_screen(dialog: object) -> None:
        pushed.append(dialog)

    def debug(key: str, msg: str) -> None:
        debug_calls.append((key, msg))

    fake_dialog = _FakeDialog()
    screen = SimpleNamespace(
        _restoring_session=True,
        _agent_loading=True,
        **_pending_submit_defaults(),
        app=SimpleNamespace(push_screen=push_screen),
        _set_agent_running=running.append,
        _set_agent_loading=loading.append,
        query_one=query_one,
        _debug=debug,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = fake_dialog
    handler._agent_load_status_snapshot = status_snapshot

    message = f"Session '40d9a0483e08' is already open in another {APP_DISPLAY_NAME} instance (pid=47219)."
    asyncio.run(handler.on_error(Error(code="session_in_use", message=message)))

    assert screen._restoring_session is False
    assert fake_dialog.dismissed is True
    assert fake_dialog.result_calls == []
    assert handler._agent_load_dialog is None
    assert loading == [False]
    assert running == [False]
    assert status_restores == [status_snapshot]
    assert len(pushed) == 1
    assert isinstance(pushed[0], ConfirmDialog)
    assert debug_calls and debug_calls[0][0] == "Error"


def _pending_submit_defaults() -> dict[str, object]:
    return {
        "_agent_running": False,
        "_pending_user_submit_active": False,
        "_pending_user_submit_text": "",
        "_pending_user_submit_blocked": False,
        "_pending_user_message_render_active": False,
        "_deferred_agent_messages": [],
    }


def _attach_pending_submit_helpers(screen: SimpleNamespace) -> None:
    def begin(text: str) -> None:
        screen._pending_user_submit_active = True
        screen._pending_user_submit_text = text
        screen._pending_user_submit_blocked = False

    def clear() -> None:
        screen._pending_user_submit_active = False
        screen._pending_user_submit_text = ""
        screen._pending_user_submit_blocked = False

    screen._begin_pending_submit = begin
    screen._clear_pending_submit = clear


def test_session_fork_error_uses_notification_without_retry_mode() -> None:
    flashes: list[tuple[str, bool]] = []
    notifications: list[tuple[str, str, str]] = []
    unlocked: list[None] = []
    running: list[bool] = []
    loading: list[bool] = []
    debug_calls: list[tuple[str, str]] = []

    class _FakeStatusBar:
        def flash(self, message: str, *, error: bool = False) -> None:
            flashes.append((message, error))

    class _FakeInputBar:
        locked = True
        retry_mode = False

        def unlock_and_keep(self) -> None:
            self.locked = False
            unlocked.append(None)

    input_bar = _FakeInputBar()

    def query_one(cls: type) -> object:
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return input_bar
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    def notify(message: str, *, title: str, severity: str, **_kwargs: object) -> None:
        notifications.append((title, severity, message))

    def set_agent_loading(value: bool) -> None:
        screen._agent_loading = value
        loading.append(value)

    defaults = _pending_submit_defaults()
    defaults["_agent_running"] = True
    screen = SimpleNamespace(
        _restoring_session=True,
        _agent_loading=False,
        **defaults,
        _set_agent_running=running.append,
        _set_agent_loading=set_agent_loading,
        query_one=query_one,
        notify=notify,
        _debug=lambda key, value: debug_calls.append((key, value)),
    )
    screen._sessions = _make_session_handler(screen)
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None

    asyncio.run(handler.on_error(Error(code="session_fork_empty", message="Cannot fork an empty session.")))

    assert screen._restoring_session is False
    assert running == []
    assert loading == [False]
    assert [(_status_text(message), error) for message, error in flashes] == [
        ("Fork: Cannot fork an empty session.", False)
    ]
    assert notifications == [("Fork", "warning", "Cannot fork an empty session.")]
    assert unlocked == [None]
    assert input_bar.retry_mode is False
    assert debug_calls == [("Error", "[session_fork_empty] Cannot fork an empty session.")]


def _make_screen_for_image_rejection(*, text: str) -> tuple[SimpleNamespace, SimpleNamespace, list[object], list[bool]]:
    class _FakeInputBar:
        def __init__(self) -> None:
            self.value = ""
            self.locked = True
            self.unlocked = False

        def unlock_and_keep(self) -> None:
            self.locked = False
            self.unlocked = True

    input_bar = _FakeInputBar()
    pushed: list[object] = []
    callbacks: list[object | None] = []
    running: list[bool] = []

    class _FakeApp:
        def push_screen(self, screen: object, callback: object | None = None) -> None:
            pushed.append(screen)
            callbacks.append(callback)

    def query_one(cls: object) -> object:
        if cls.__name__ == "InputBar":
            return input_bar
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _restoring_session=False,
        _agent_loading=False,
        _agent_running=False,
        _pending_user_submit_active=True,
        _pending_user_submit_text=text,
        _pending_user_submit_blocked=False,
        app=_FakeApp(),
        _pushed_callbacks=callbacks,
        _set_agent_running=running.append,
        query_one=query_one,
        _debug=lambda *_args: None,
    )
    return screen, input_bar, pushed, running


def test_image_attachment_error_from_backend_uses_modal_and_restores_prompt() -> None:
    screen, input_bar, pushed, running = _make_screen_for_image_rejection(text="describe @shot.png")
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None

    asyncio.run(
        handler.on_error(
            Error(
                code="vision_unsupported",
                message='The active model profile "Model" does not support image input.',
            )
        )
    )

    assert screen._pending_user_submit_blocked is True
    assert running == [False]
    assert input_bar.value == "describe @shot.png"
    assert input_bar.unlocked is True
    assert len(pushed) == 1
    dialog = pushed[0]
    assert dialog._title == "Image Input Not Available"
    assert "does not support image input" in dialog._message
    assert screen._pushed_callbacks[0] is not None


def test_image_rejection_dialog_action_rewrites_image_mentions_to_paths(tmp_path: Path) -> None:
    first = tmp_path / "shot.png"
    second = tmp_path / "screen two.jpg"
    text = f'inspect @{first.name} and @"{second}" but keep @notes.txt'
    screen, input_bar, pushed, running = _make_screen_for_image_rejection(text=text)
    screen._chdir_current_cwd = str(tmp_path)
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None

    asyncio.run(
        handler.on_error(
            Error(
                code="vision_unsupported",
                message='The active model profile "Model" does not support image input.',
            )
        )
    )

    from chrys.app.tui.screens.dialogs.vision_unsupported import USE_IMAGE_PATHS_RESULT

    callback = screen._pushed_callbacks[0]
    assert callback is not None
    callback(USE_IMAGE_PATHS_RESULT)

    assert running == [False]
    assert input_bar.value == f"inspect {first} and {second} but keep @notes.txt"
    assert len(pushed) == 1
    dialog = pushed[0]
    assert dialog._show_path_action is True


def test_image_attachment_timeout_error_uses_modal_and_restores_prompt() -> None:
    screen, input_bar, pushed, running = _make_screen_for_image_rejection(text="describe @huge.png")
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None
    message = (
        "We couldn't attach this image.\n\n"
        "- @huge.png: Image preparation took longer than 1 second. "
        "Resize oversized images or send fewer images, then try again."
    )

    asyncio.run(handler.on_error(Error(code="image_attachment_error", message=message)))

    assert screen._pending_user_submit_blocked is True
    assert running == [False]
    assert input_bar.value == "describe @huge.png"
    assert input_bar.unlocked is True
    assert len(pushed) == 1
    dialog = pushed[0]
    assert dialog._title == "Image Not Attached"
    assert "Image preparation took longer than 1 second" in dialog._message


def test_pending_submit_error_restores_prompt_without_inline_retry_action() -> None:
    class _FakeInputBar:
        def __init__(self) -> None:
            self.value = ""
            self.locked = True
            self.unlocked = False

        def unlock_and_keep(self) -> None:
            self.locked = False
            self.unlocked = True

    class _FakeStatusBar:
        def flash(self, _message: str, *, error: bool = False) -> None:
            assert error is True

    class _FakeChatPanel:
        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            chat_errors.append((message, action_label))

    input_bar = _FakeInputBar()
    chat_errors: list[tuple[str, str | None]] = []
    running: list[bool] = []

    def query_one(cls: object) -> object:
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "ChatPanel":
            return _FakeChatPanel()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _restoring_session=False,
        _agent_loading=False,
        _agent_running=False,
        _pending_user_submit_active=True,
        _pending_user_submit_text="blocked prompt",
        _pending_user_submit_blocked=False,
        _pending_user_message_render_active=True,
        _set_agent_running=running.append,
        query_one=query_one,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None

    asyncio.run(handler.on_error(Error(code="hook_blocked", message="Prompt denied")))

    assert screen._pending_user_submit_blocked is True
    assert running == [False]
    assert input_bar.value == "blocked prompt"
    assert input_bar.unlocked is True
    assert chat_errors == [("Prompt denied", None)]


def test_pending_submit_error_display_localizes_chat_but_debug_keeps_protocol_english() -> None:
    from chrys.orchestration.engine.run.turn_hooks import _TURN_HOOKS_PROMPT_BLOCKED

    class _FakeInputBar:
        def __init__(self) -> None:
            self.value = ""
            self.locked = True

        def unlock_and_keep(self) -> None:
            self.locked = False

    class _FakeStatusBar:
        def flash(self, message: MessageRef | str, *, error: bool = False) -> None:
            flashes.append(message)

    class _FakeChatPanel:
        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            chat_errors.append((message, action_label))

    input_bar = _FakeInputBar()
    flashes: list[MessageRef | str] = []
    chat_errors: list[tuple[str, str | None]] = []
    debug_calls: list[tuple[str, str]] = []

    def query_one(cls: object) -> object:
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "ChatPanel":
            return _FakeChatPanel()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _restoring_session=False,
        _agent_loading=False,
        _agent_running=False,
        _pending_user_submit_active=True,
        _pending_user_submit_text="blocked prompt",
        _pending_user_submit_blocked=False,
        _pending_user_message_render_active=True,
        _set_agent_running=lambda _value: None,
        query_one=query_one,
        _debug=lambda key, msg: debug_calls.append((key, msg)),
    )
    handler = make_backend_handler(screen, locale_controller=LocaleController(Settings(locale="zh-Hans")))
    handler._agent_load_dialog = None

    asyncio.run(
        handler.on_error(
            Error(
                code="hook_blocked",
                message="Prompt blocked by hook.",
                display_message=_TURN_HOOKS_PROMPT_BLOCKED.bind(),
            )
        )
    )

    assert chat_errors == [("提示已被钩子阻止。", None)]
    chinese = Localizer("zh-Hans")
    assert [chinese.render(message) for message in flashes] == ["错误：提示已被钩子阻止。"]  # noqa: RUF001
    assert debug_calls == [("Error", "[hook_blocked] Prompt blocked by hook.")]


def test_send_user_message_real_bus_rejection_sets_pending_blocked_before_render() -> None:
    bus = EventBus()
    running: list[bool] = []
    published: list[str] = []
    controller, state, view = _make_input_flow_controller(bus, running=running)

    async def _reject_image_prompt(event: UserMessage) -> None:
        published.append(event.text)
        state.submit.block()

    async def _run() -> None:
        await bus.subscribe(UserMessage, _reject_image_prompt)
        await controller.send_user_message("describe @shot.png")

    asyncio.run(_run())

    assert published == ["describe @shot.png"]
    assert state.submit.active is False
    assert state.submit.blocked is False
    assert state.render_gate.active is False
    assert running == []
    assert "user:describe @shot.png" not in view.calls


def test_send_user_message_defers_fast_agent_message_until_user_bubble_is_rendered() -> None:
    bus = EventBus()
    state = MainScreenState()
    view = _FakeInputFlowView()

    async def _on_agent_message(event: AgentMessage) -> None:
        if state.render_gate.active:
            state.render_gate.defer(event)
            return
        view.calls.append(f"agent:{event.text}")

    controller, _state, view = _make_input_flow_controller(
        bus,
        state=state,
        view=view,
        handle_agent_message=_on_agent_message,
    )

    async def _accept_and_finish(event: UserMessage) -> None:
        event.prepared_contents = ["hello", "prepared-image-content"]
        await bus.publish(AgentMessage(text=f"done:{event.text}", is_final=True, session_id=event.session_id))

    async def _run() -> None:
        await bus.subscribe(UserMessage, _accept_and_finish)
        await bus.subscribe(AgentMessage, _on_agent_message)
        await controller.send_user_message("hello")

    asyncio.run(_run())

    user_index = view.calls.index("user:hello")
    agent_index = view.calls.index("agent:done:hello")
    assert user_index < agent_index
    assert view.user_contents == [["hello", "prepared-image-content"]]


def test_send_user_message_defers_fast_error_until_user_bubble_is_rendered() -> None:
    bus = EventBus()
    state = MainScreenState()
    order: list[str] = []

    class _FlowView(_FakeInputFlowView):
        async def render_user_message(self, text: str, *, created_at: object, contents: object) -> None:
            await super().render_user_message(text, created_at=created_at, contents=contents)
            order.append(f"user:{text}")

    class _FakeStatusBar:
        def flash(self, message: str, *, error: bool = False) -> None:
            order.append(f"status:{message}:{error}")

    class _FakeInputBar:
        def __init__(self) -> None:
            self.locked = True
            self._retry_label = ""
            self.retry_mode = False

        def unlock_and_keep(self) -> None:
            self.locked = False
            order.append("unlock")

    class _FakeChatPanel:
        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            order.append(f"error:{message}:{action_label}")

    def query_one(cls: object) -> object:
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ChatPanel":
            return _FakeChatPanel()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    input_bar = _FakeInputBar()
    view = _FlowView()
    screen = SimpleNamespace(
        _state=state,
        _agent_running=False,
        _mark_terminal_title_failed=lambda: None,
        query_one=query_one,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None

    controller, _state, view = _make_input_flow_controller(
        bus,
        state=state,
        view=view,
        handle_error=handler.on_error,
    )

    async def _accept_and_fail(event: UserMessage) -> None:
        event.prepared_contents = ["hello", "prepared-image-content"]
        await bus.publish(Error(code="executor_error", message=f"failed:{event.text}", session_id=event.session_id))

    async def _run() -> None:
        await bus.subscribe(UserMessage, _accept_and_fail)
        await bus.subscribe(Error, handler.on_error)
        await controller.send_user_message("hello")

    asyncio.run(_run())

    user_index = order.index("user:hello")
    error_index = order.index("error:failed:hello:Retry")
    assert user_index < error_index
    assert view.user_contents == [["hello", "prepared-image-content"]]
    assert state.submit.blocked is False
    assert input_bar.retry_mode is True


def test_backend_handler_defers_agent_message_while_user_bubble_is_rendering() -> None:
    state = MainScreenState()
    state.render_gate.begin()

    def query_one(_cls: object) -> object:
        raise AssertionError("agent message should be deferred")

    screen = SimpleNamespace(
        _state=state,
        query_one=query_one,
    )
    handler = make_backend_handler(screen)
    event = AgentMessage(text="fast", is_final=True)

    asyncio.run(handler.on_agent_message(event))

    assert state.render_gate.consume_deferred() == [event]


def test_final_agent_message_keeps_gate_and_prestamps_terminal_absorb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from textual import _time, events

    order: list[str] = []
    input_events: list[events.Key] = []

    class _GcMessages(list[object]):
        def append(self, message: object) -> None:
            order.append("gc")
            super().append(message)

    class _FakeStatusBar:
        def _format_elapsed(self) -> str:
            return "1s"

        def flash(self, _message: str) -> None:
            order.append("status")

    class _FakeChatPanel:
        async def add_agent_message(self, *_args: object, **_kwargs: object) -> None:
            assert handler._state.run.agent_running is True
            input_events.append(events.Key("x", "x"))
            order.append("render")

    status = _FakeStatusBar()
    panel = _FakeChatPanel()

    def query_one(cls: type) -> object:
        if cls.__name__ == "StatusBar":
            return status
        if cls.__name__ == "ChatPanel":
            return panel
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    gc_messages = _GcMessages()
    screen = SimpleNamespace(
        _agent_running=True,
        _gc_messages=gc_messages,
        _mark_terminal_title_completed=lambda: order.append("completed"),
        _set_agent_running=lambda _value: order.append("idle"),
        query_one=query_one,
        _debug=lambda *_args: None,
    )

    handler = make_backend_handler(screen)
    terminal_event = AgentMessage(text="done", is_final=True)
    source_times = iter([10.0, 20.0])
    monkeypatch.setattr(_time, "get_time", lambda: next(source_times))
    asyncio.run(handler.on_agent_message(terminal_event))

    assert order == ["status", "render", "completed", "idle", "gc"]
    assert len(gc_messages) == 1
    assert isinstance(gc_messages[0], GcAbsorbRequested)
    assert gc_messages[0].reason is GcAbsorbReason.TURN_TERMINAL
    assert gc_messages[0].terminal_boundary is True
    assert gc_messages[0].time == 10.0
    assert input_events[0].time == 20.0


def test_final_agent_message_render_failure_releases_turn_without_absorb() -> None:
    class _FakeStatusBar:
        def _format_elapsed(self) -> str:
            return "1s"

        def flash(self, _message: str) -> None:
            pass

    class _FailingChatPanel:
        async def add_agent_message(self, *_args: object, **_kwargs: object) -> None:
            assert handler._state.run.agent_running is True
            raise RuntimeError("render failed")

    status = _FakeStatusBar()
    panel = _FailingChatPanel()

    def query_one(cls: type) -> object:
        if cls.__name__ == "StatusBar":
            return status
        if cls.__name__ == "ChatPanel":
            return panel
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    running: list[bool] = []
    gc_messages: list[object] = []
    screen = SimpleNamespace(
        _agent_running=True,
        _gc_messages=gc_messages,
        _mark_terminal_title_completed=lambda: None,
        _set_agent_running=running.append,
        query_one=query_one,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)

    with pytest.raises(RuntimeError, match="render failed"):
        asyncio.run(handler.on_agent_message(AgentMessage(text="done", is_final=True)))

    assert running == [False]
    assert handler._state.run.agent_running is False
    assert gc_messages == []


def test_prior_run_terminal_message_cannot_stop_new_retry() -> None:
    started_at = datetime.now(UTC)
    state = MainScreenState()
    state.run.agent_running = True
    state.run.generation = 2
    state.run.started_at = started_at

    def query_one(_cls: object) -> object:
        raise AssertionError("stale terminal event must not touch the current retry UI")

    screen = SimpleNamespace(_state=state, _agent_running=True, query_one=query_one)
    handler = make_backend_handler(screen)

    asyncio.run(
        handler.on_agent_message(
            AgentMessage(
                text="old final",
                is_final=True,
                timestamp=started_at - timedelta(milliseconds=1),
            )
        )
    )

    assert state.run.agent_running is True
    assert state.run.generation == 2


@pytest.mark.asyncio
async def test_terminal_render_completion_cannot_stop_successor_generation() -> None:
    render_started = asyncio.Event()
    release_render = asyncio.Event()
    running: list[bool] = []
    gc_messages: list[object] = []
    state = MainScreenState()
    state.run.agent_running = True
    state.run.generation = 1
    state.run.started_at = datetime.now(UTC)

    class _Status:
        def _format_elapsed(self) -> str:
            return "1s"

        def flash(self, _message: str) -> None:
            return

    class _Panel:
        async def add_agent_message(self, *_args: object, **_kwargs: object) -> None:
            render_started.set()
            await release_render.wait()

    def query_one(cls: type) -> object:
        if cls.__name__ == "StatusBar":
            return _Status()
        if cls.__name__ == "ChatPanel":
            return _Panel()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    completed: list[None] = []
    screen = SimpleNamespace(
        _state=state,
        _agent_running=True,
        _gc_messages=gc_messages,
        _mark_terminal_title_completed=lambda: completed.append(None),
        _set_agent_running=running.append,
        query_one=query_one,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    terminal = asyncio.create_task(
        handler.on_agent_message(
            AgentMessage(text="old final", is_final=True, timestamp=state.run.started_at + timedelta(milliseconds=1))
        )
    )
    await render_started.wait()
    state.run.generation = 2
    state.run.agent_running = True
    release_render.set()
    await terminal

    assert state.run.agent_running is True
    assert state.run.generation == 2
    assert running == []
    assert completed == []
    assert gc_messages == []


def test_backend_handler_defers_error_while_user_bubble_is_rendering() -> None:
    state = MainScreenState()
    state.render_gate.begin()
    state.run.agent_running = True
    state.submit.begin("hello")

    def query_one(_cls: object) -> object:
        raise AssertionError("error should be deferred before querying live widgets")

    screen = SimpleNamespace(
        _state=state,
        query_one=query_one,
    )
    handler = make_backend_handler(screen)
    event = Error(code="executor_error", message="fast failure")

    asyncio.run(handler.on_error(event))

    assert state.run.agent_running is True
    assert state.submit.blocked is False
    assert state.render_gate.consume_deferred() == [event]


def test_live_turn_error_requests_terminal_absorb_after_render() -> None:
    order: list[str] = []

    class _GcMessages(list[object]):
        def append(self, message: object) -> None:
            order.append("gc")
            super().append(message)

    class _FakeStatusBar:
        def flash(self, _message: str, *, error: bool = False) -> None:
            assert error is True
            order.append("status")

    class _FakeInputBar:
        locked = True
        retry_mode = False
        _retry_label = ""

        def unlock_and_keep(self) -> None:
            self.locked = False
            order.append("unlock")

    class _FakeChatPanel:
        async def add_error(self, _message: str, *, action_label: str | None = "Retry") -> None:
            assert action_label == "Retry"
            assert handler._state.run.agent_running is True
            assert input_bar.locked is True
            order.append("render")

    status = _FakeStatusBar()
    input_bar = _FakeInputBar()
    panel = _FakeChatPanel()

    def query_one(cls: type) -> object:
        if cls.__name__ == "StatusBar":
            return status
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ChatPanel":
            return panel
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    gc_messages = _GcMessages()
    screen = SimpleNamespace(
        _agent_running=True,
        _agent_loading=False,
        _restoring_session=False,
        _gc_messages=gc_messages,
        _mark_terminal_title_failed=lambda: order.append("failed"),
        _set_agent_running=lambda _value: order.append("idle"),
        query_one=query_one,
        _debug=lambda *_args: None,
    )

    handler = make_backend_handler(screen)
    asyncio.run(handler.on_error(Error(code="executor_error", message="failed")))

    assert order == ["status", "render", "failed", "idle", "unlock", "gc"]
    assert len(gc_messages) == 1
    assert isinstance(gc_messages[0], GcAbsorbRequested)
    assert gc_messages[0].reason is GcAbsorbReason.TURN_TERMINAL
    assert gc_messages[0].terminal_boundary is True


def test_live_turn_error_render_failure_releases_input_without_absorb() -> None:
    class _FakeStatusBar:
        def flash(self, _message: str, *, error: bool = False) -> None:
            assert error is True

    class _FakeInputBar:
        locked = True
        retry_mode = False
        _retry_label = ""

        def unlock_and_keep(self) -> None:
            self.locked = False

    class _FailingChatPanel:
        async def add_error(self, _message: str, *, action_label: str | None = "Retry") -> None:
            assert action_label == "Retry"
            assert handler._state.run.agent_running is True
            raise RuntimeError("render failed")

    status = _FakeStatusBar()
    input_bar = _FakeInputBar()
    panel = _FailingChatPanel()

    def query_one(cls: type) -> object:
        if cls.__name__ == "StatusBar":
            return status
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ChatPanel":
            return panel
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    running: list[bool] = []
    gc_messages: list[object] = []
    screen = SimpleNamespace(
        _agent_running=True,
        _agent_loading=False,
        _restoring_session=False,
        _gc_messages=gc_messages,
        _mark_terminal_title_failed=lambda: None,
        _set_agent_running=running.append,
        query_one=query_one,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)

    with pytest.raises(RuntimeError, match="render failed"):
        asyncio.run(handler.on_error(Error(code="executor_error", message="failed")))

    assert running == [False]
    assert handler._state.run.agent_running is False
    assert input_bar.locked is False
    assert gc_messages == []
    assert input_bar.retry_mode is False


def test_image_attachment_warning_from_backend_uses_modal_and_restores_prompt() -> None:
    screen, input_bar, pushed, running = _make_screen_for_image_rejection(text="continue with @shot.png")
    handler = make_backend_handler(screen)
    message = "Images cannot be attached to retry or continuation prompts."
    handler._seen_warnings = {
        _WarningDedupeKey(
            session_id=None,
            code="image_attachment_retry_unsupported",
            message=message,
        )
    }

    asyncio.run(
        handler.on_warning(
            Warning(
                code="image_attachment_retry_unsupported",
                message=message,
            )
        )
    )

    assert screen._pending_user_submit_blocked is True
    assert running == []
    assert input_bar.value == "continue with @shot.png"
    assert input_bar.unlocked is True
    assert len(pushed) == 1
    dialog = pushed[0]
    assert dialog._title == "Image Not Attached"
    assert "retry or continuation prompts" in dialog._message


def test_submit_blocking_warning_restores_prompt_without_modal() -> None:
    class _FakeInputBar:
        def __init__(self) -> None:
            self.value = ""
            self.locked = True
            self.unlocked = False

        def unlock_and_keep(self) -> None:
            self.locked = False
            self.unlocked = True

    input_bar = _FakeInputBar()
    notifications: list[tuple[str, str, str]] = []
    pushed: list[object] = []

    def query_one(cls: object) -> object:
        if cls.__name__ == "InputBar":
            return input_bar
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    def notify(message: str, *, title: str, severity: str, **_kwargs: object) -> None:
        notifications.append((message, title, severity))

    screen = SimpleNamespace(
        _pending_user_submit_active=True,
        _pending_user_submit_text="retry after the other agent finishes",
        _pending_user_submit_blocked=False,
        app=SimpleNamespace(push_screen=pushed.append),
        notify=notify,
        query_one=query_one,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._seen_warnings = set()

    asyncio.run(
        handler.on_warning(
            Warning(
                code="sub_agent_paused",
                message="Resolve paused sub-agent(s) first.",
            )
        )
    )

    assert screen._pending_user_submit_blocked is True
    assert input_bar.value == "retry after the other agent finishes"
    assert input_bar.unlocked is True
    assert pushed == []
    assert notifications == [("Resolve paused sub-agent(s) first.", "Warning", "warning")]


def test_error_event_empty_message_uses_code_fallback() -> None:
    """Blank backend error messages should not render as an empty chat/status error."""
    flashes: list[tuple[str, bool]] = []
    chat_errors: list[str] = []
    running: list[bool] = []
    unlocked: list[bool] = []
    debug_calls: list[tuple[str, str]] = []

    class _FakeStatusBar:
        def flash(self, message: str, *, error: bool = False) -> None:
            flashes.append((message, error))

    class _FakeInputBar:
        def __init__(self) -> None:
            self.locked = True
            self._retry_label = ""
            self.retry_mode = False

        def unlock_and_keep(self) -> None:
            unlocked.append(True)
            self.locked = False

    class _FakeChatPanel:
        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            assert action_label == "Retry"
            chat_errors.append(message)

    status = _FakeStatusBar()
    input_bar = _FakeInputBar()
    panel = _FakeChatPanel()

    def query_one(cls: type):
        if cls.__name__ == "StatusBar":
            return status
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ChatPanel":
            return panel
        raise AssertionError(f"unexpected query_one({cls})")

    def set_agent_running(value: bool) -> None:
        running.append(value)

    def debug(key: str, msg: str) -> None:
        debug_calls.append((key, msg))

    screen = SimpleNamespace(
        _restoring_session=True,
        _agent_loading=False,
        **_pending_submit_defaults(),
        _set_agent_running=set_agent_running,
        query_one=query_one,
        _debug=debug,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None

    asyncio.run(handler.on_error(Error(code="executor_error", message="")))

    assert screen._restoring_session is False
    assert running == [False]
    assert [(_status_text(message), error) for message, error in flashes] == [("Error: Executor error", True)]
    assert chat_errors == ["Executor error"]
    assert unlocked == [True]
    assert isinstance(input_bar._retry_label, MessageRef)
    assert _status_text(input_bar._retry_label) == "Retry"
    assert input_bar.retry_mode is True
    assert debug_calls == [("Error", "[executor_error] Executor error")]


def test_error_display_message_localizes_ui_surfaces_and_keeps_raw_debug() -> None:
    """Errors carrying a display reference show it on the status bar and in
    the chat panel; debug lines keep the raw protocol message."""
    flashes: list[tuple[MessageRef | str, bool]] = []
    chat_errors: list[str] = []
    debug_calls: list[tuple[str, str]] = []

    class _FakeStatusBar:
        def flash(self, message: MessageRef | str, *, error: bool = False) -> None:
            flashes.append((message, error))

    class _FakeInputBar:
        def __init__(self) -> None:
            self.locked = True
            self._retry_label = ""
            self.retry_mode = False

        def unlock_and_keep(self) -> None:
            self.locked = False

    class _FakeChatPanel:
        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            chat_errors.append(message)

    status = _FakeStatusBar()
    input_bar = _FakeInputBar()
    panel = _FakeChatPanel()

    def query_one(cls: type):
        if cls.__name__ == "StatusBar":
            return status
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "ChatPanel":
            return panel
        raise AssertionError(f"unexpected query_one({cls})")

    screen = SimpleNamespace(
        _restoring_session=False,
        _agent_loading=False,
        **_pending_submit_defaults(),
        _set_agent_running=lambda _value: None,
        query_one=query_one,
        _debug=lambda key, msg: debug_calls.append((key, msg)),
    )
    handler = make_backend_handler(screen, locale_controller=LocaleController(Settings(locale="zh-Hans")))
    handler._agent_load_dialog = None

    asyncio.run(
        handler.on_error(Error(code="run_failed", message="Unknown error", display_message=_UNKNOWN_ERROR.bind()))
    )

    chinese = Localizer("zh-Hans")
    assert [(chinese.render(message), error) for message, error in flashes] == [("错误：未知错误", True)]  # noqa: RUF001
    assert chat_errors == ["未知错误"]
    assert debug_calls == [("Error", "[run_failed] Unknown error")]


def test_error_event_empty_message_during_agent_loading_uses_load_fallback() -> None:
    """Startup fallback errors should preserve the agent-load specific message."""
    dialog_results: list[tuple[bool, str, bool]] = []
    flashes: list[str] = []
    chat_errors: list[str] = []
    loading: list[bool] = []
    running: list[bool] = []

    class _FakeDialog:
        def set_result(self, success: bool, message: str, *, allow_esc: bool = False) -> None:
            dialog_results.append((success, message, allow_esc))

    class _FakeStatusBar:
        def flash(self, message: str, *, error: bool = False) -> None:
            flashes.append(message)
            assert error is True

    class _FakeInputBar:
        locked = False
        _retry_label = ""
        retry_mode = False

    class _FakeChatPanel:
        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            assert action_label == "Retry"
            chat_errors.append(message)

    def query_one(cls: type):
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return _FakeInputBar()
        if cls.__name__ == "ChatPanel":
            return _FakeChatPanel()
        raise AssertionError(f"unexpected query_one({cls})")

    screen = SimpleNamespace(
        _restoring_session=True,
        _agent_loading=True,
        **_pending_submit_defaults(),
        _set_agent_loading=loading.append,
        _set_agent_running=running.append,
        query_one=query_one,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = _FakeDialog()
    handler._agent_load_status_snapshot = {"visible": True, "flash": None, "status": "Before load"}

    asyncio.run(handler.on_error(Error(code="executor_error", message="   ")))

    assert dialog_results == [(False, "Agent failed to load.", True)]
    assert handler._agent_load_status_snapshot is None
    assert loading == [False]
    assert running == [False]
    assert [_status_text(message) for message in flashes] == ["Error: Agent failed to load."]
    assert chat_errors == ["Agent failed to load."]


def test_error_event_during_agent_loading_clears_loading_if_dialog_update_fails() -> None:
    """A stale loading dialog must not prevent the input bar from unlocking."""
    import asyncio

    flashes: list[str] = []
    chat_errors: list[str] = []
    loading: list[bool] = []
    running: list[bool] = []

    class _BrokenDialog:
        def set_result(self, *_args: object, **_kwargs: object) -> None:
            raise RuntimeError("dialog already gone")

    class _FakeStatusBar:
        def flash(self, message: str, *, error: bool = False) -> None:
            flashes.append(message)
            assert error is True

    class _FakeInputBar:
        locked = False
        _retry_label = ""
        retry_mode = False

    class _FakeChatPanel:
        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            assert action_label == "Retry"
            chat_errors.append(message)

    def query_one(cls: type):
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        if cls.__name__ == "InputBar":
            return _FakeInputBar()
        if cls.__name__ == "ChatPanel":
            return _FakeChatPanel()
        raise AssertionError(f"unexpected query_one({cls})")

    screen = SimpleNamespace(
        _restoring_session=True,
        _agent_loading=True,
        **_pending_submit_defaults(),
        _set_agent_loading=loading.append,
        _set_agent_running=running.append,
        query_one=query_one,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = _BrokenDialog()

    asyncio.run(handler.on_error(Error(code="executor_error", message="   ")))

    assert handler._agent_load_dialog is None
    assert loading == [False]
    assert running == [False]
    assert [_status_text(message) for message in flashes] == ["Error: Agent failed to load."]
    assert chat_errors == ["Agent failed to load."]


def test_show_rollback_defers_attribution_refresh_to_modal(monkeypatch) -> None:
    """The modal opens first and owns its post-paint attribution refresh."""
    import chrys.app.tui.screens.diff as diff_pkg

    pushed: list[object] = []
    refreshed: list[bool] = []
    order: list[str] = []
    state_loaders: list[object] = []

    class _FakeRollbackModal:
        def __init__(self, **kwargs: object) -> None:
            order.append("modal")
            pushed.append(self)
            state_loaders.append(kwargs["load_state"])

    class _FakeEngine(_RollbackProjectionFenceMixin):
        mutation_tracker = object()
        mutation_coordinator = object()
        current_turn_number = 3
        conversation_revision = 7

        def available_rollback_turns(self) -> list[int]:
            return [1, 2]

        def turn_prompt_previews(self) -> dict[int, str]:
            return {}

        async def refresh_mutation_attribution(self, *, force: bool = False) -> bool:
            order.append("refresh")
            refreshed.append(force)
            return False

    monkeypatch.setattr(diff_pkg, "RollbackModal", _FakeRollbackModal)
    screen = SimpleNamespace(
        app=SimpleNamespace(push_screen=lambda modal, _callback=None: None),
        notify=lambda *_args, **_kwargs: None,
    )
    controller = RollbackController(
        services=MainScreenServices(bus=EventBus(), engine_provider=_FakeEngine),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=lambda: "/repo/workspace",
        is_agent_busy=lambda: False,
        current_session_id=lambda: "session-1",
        session_generation=lambda: 1,
        turn_lifecycle_task=lambda: None,
        profile_name=lambda: "Code",
        reset_welcome_workspace_marker=lambda _cwd: None,
        set_has_messages=lambda _value: None,
        post_gc_message=lambda _message: None,
        debug=lambda *_args: None,
    )

    controller.show_rollback()

    assert refreshed == []
    assert order == ["modal"]
    assert len(pushed) == 1
    assert len(state_loaders) == 1
    state = asyncio.run(state_loaders[0]())
    assert state is not None
    assert state.attribution_refresh is not None
    asyncio.run(state.attribution_refresh())
    assert refreshed == [True]


def test_show_diff_refreshes_attribution_when_coordinated() -> None:
    """The loading screen opens before coordinated attribution refresh."""
    refreshed: list[bool] = []
    notified: list[str] = []
    pushed: list[object] = []

    class _FakeEngine:
        mutation_coordinator = object()

        async def refresh_mutation_attribution(self, *, force: bool = False) -> bool:
            refreshed.append(force)
            return False

    screen = SimpleNamespace(
        _state_store=None,
        _live_file_mutations={},
        _workspace_cwd=lambda: "/repo/workspace",
        _agent_running=False,
        query_one=lambda _cls: SimpleNamespace(session_id=""),
        notify=lambda message, **_kwargs: notified.append(message),
        app=SimpleNamespace(push_screen=pushed.append),
    )
    controller = DiffController(
        services=MainScreenServices(bus=EventBus(), engine_provider=_FakeEngine),
        live_diff=LiveDiffTracker(file_mutations=screen._live_file_mutations),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=screen._workspace_cwd,
        is_agent_running=lambda: screen._agent_running,
        run_generation=lambda: 0,
        session_generation=lambda: 0,
    )

    controller.show_diff()

    assert len(pushed) == 1
    assert refreshed == []
    loader = pushed[0]._load_data
    assert loader is not None
    result = asyncio.run(loader())

    assert refreshed == [False]
    assert result == DiffLoadResult.empty()
    assert notified == []


def test_show_diff_defers_load_without_coordinator() -> None:
    """The loading screen also opens immediately without a coordinator."""
    notified: list[str] = []
    pushed: list[object] = []

    class _FakeEngine:
        mutation_coordinator = None

    screen = SimpleNamespace(
        _state_store=None,
        _live_file_mutations={},
        _workspace_cwd=lambda: "/repo/workspace",
        _agent_running=False,
        query_one=lambda _cls: SimpleNamespace(session_id=""),
        notify=lambda message, **_kwargs: notified.append(message),
        app=SimpleNamespace(push_screen=pushed.append),
    )
    controller = DiffController(
        services=MainScreenServices(bus=EventBus(), engine_provider=_FakeEngine),
        live_diff=LiveDiffTracker(file_mutations=screen._live_file_mutations),
        view=MainScreenViewAdapter(screen),  # type: ignore[arg-type]
        workspace_cwd=screen._workspace_cwd,
        is_agent_running=lambda: screen._agent_running,
        run_generation=lambda: 0,
        session_generation=lambda: 0,
    )

    controller.show_diff()

    assert len(pushed) == 1
    loader = pushed[0]._load_data
    assert loader is not None
    result = asyncio.run(loader())

    assert result == DiffLoadResult.empty()
    assert notified == []


# ---------------------------------------------------------------------------
# Todo list (Tasks panel) wiring
# ---------------------------------------------------------------------------


def test_todo_list_updated_sets_todo_state_and_debug_logs() -> None:
    """TodoListUpdated routes the full list into screen.todo_state."""

    debug_calls: list[tuple[str, str]] = []
    screen = SimpleNamespace(
        _debug=lambda key, message="": debug_calls.append((key, message)),
    )
    handler = make_backend_handler(screen)
    items = [
        TodoItem(content="write tests", status="completed"),
        TodoItem(content="run suite", status="in_progress", active_form="Running suite"),
        TodoItem(content="ship it"),
    ]

    asyncio.run(handler.on_todo_list_updated(TodoListUpdated(items=items, session_id="session-1")))

    assert screen.todo_state == TodoListState(items=tuple(items))
    assert ("TodoListUpdated", "1/3 done") in debug_calls


def test_session_ready_for_new_session_clears_todo_state() -> None:
    """Creating a new session resets the Tasks panel at the post-success point."""

    calls: list[tuple[str, object]] = []

    class _FakePanel:
        border_subtitle = None

        def set_profile(self, profile: str) -> None:
            return

        def set_tool_kinds(self, _tool_kinds: dict[str, str]) -> None:
            return

        async def clear(self) -> None:
            calls.append(("clear", None))

        def update_welcome(self, *, profile: str = "", cwd: str = "") -> None:
            return

        def set_session_id(self, session_id: str) -> None:
            return

        def set_workspace_cwd(self, cwd: str) -> None:
            self.border_subtitle = Text(cwd)

    class _FakeInputBar:
        retry_mode = True

        def set_paste_cwd(self, cwd: str) -> None:
            return

        def set_clipboard_image_dir(self, directory: object) -> None:
            return

    class _FakeStatusBar:
        def set_profile(self, _profile: str, *, description: str = "") -> None:
            return

        def set_tool_info(self, _trail: str) -> None:
            return

        def clear_status(self) -> None:
            return

        def flash(self, text: str, **_kwargs) -> None:
            return

    class _FakeContextPanel:
        def reset(self, max_context_tokens: int = 0) -> None:
            return

    class _FakeSidebarPanel:
        context_panel = _FakeContextPanel()

    class _FakeStateStore:
        def session_dir(self, session_id: str) -> Path:
            return Path("/sessions") / session_id

    panel = _FakePanel()
    input_bar = _FakeInputBar()
    status = _FakeStatusBar()
    sidebar = _FakeSidebarPanel()

    def query_one(cls: type):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "StatusBar":
            return status
        if cls.__name__ == "SidebarPanel":
            return sidebar
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _agent_registry=None,
        _model_registry=None,
        _creating_new_session=True,
        _restoring_session=False,
        _profile="",
        _runtime_details=None,
        _state_store=_FakeStateStore(),
        todo_state=TodoListState(items=(TodoItem(content="stale from old session"),)),
        query_one=query_one,
        _set_has_messages=lambda value: calls.append(("has_messages", value)),
        _set_agent_loading=lambda value: calls.append(("agent_loading", value)),
        _set_terminal_title_for_cwd=lambda cwd=None: calls.append(("title", cwd)),
        _reset_session_title_state=lambda: None,
        _set_session_title_state=lambda **_kwargs: None,
        _session_custom_title="",
        _update_subtitle=lambda: None,
        _update_toc=lambda: None,
        _debug=lambda *_args: None,
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None

    asyncio.run(
        handler.on_session_ready(
            SessionReady(
                agent_profile="Code",
                display_name="Code Agent",
                session_id="session-new",
                max_context_tokens=123,
                primary_cwd="/workspace/new",
            )
        )
    )

    assert screen._creating_new_session is False
    assert screen.todo_state == TodoListState()


def test_session_ready_for_existing_session_preserves_todo_state() -> None:
    """A repeated SessionReady (no new-session flow) must not clear the Tasks panel."""

    screen, _calls, _initial = _run_existing_session_ready(
        current_max_context_tokens=200_000,
        event_max_context_tokens=200_000,
    )

    # clear_todos() would have stamped an empty TodoListState onto the screen.
    assert not hasattr(screen, "todo_state")


def test_session_restore_reseeds_todo_state_from_saved_state_tolerantly(tmp_path: Path) -> None:
    """Restoring a session reseeds the Tasks panel, skipping malformed entries."""

    calls: list[tuple[str, object]] = []
    state = {
        "messages": [],
        "chrys_todos": [
            {"content": "restored", "status": "completed", "active_form": ""},
            {"content": "", "status": "pending"},
            {"content": "bad status", "status": "someday"},
            "garbage",
            {"content": "kept"},
        ],
    }

    class _FakePanel:
        border_subtitle = None

        async def clear(self) -> None:
            return

        def set_session_id(self, session_id: str) -> None:
            return

        def set_workspace_cwd(self, cwd: str) -> None:
            self.border_subtitle = Text(cwd)

        def update_usage(self, tokens: int, total_session_tokens: int = 0) -> None:
            return

        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            return

    class _FakeInputBar:
        retry_mode = True

        def set_paste_cwd(self, cwd: str) -> None:
            return

        def set_clipboard_image_dir(self, directory: object) -> None:
            return

    class _FakeStatusBar:
        def flash(self, text: str) -> None:
            return

    class _FakeContextPanel:
        def clear_blocks(self) -> None:
            return

    class _FakeSidebarPanel:
        context_panel = _FakeContextPanel()

    class _FakeStateStore:
        def session_dir(self, session_id: str) -> Path:
            return tmp_path / "sessions" / session_id

        async def load_session(self, session_id: str, *, prefer_recovery: bool = False) -> dict[str, object]:
            calls.append(("load_session", (session_id, prefer_recovery)))
            return state

        async def load_session_raw(self, session_id: str, *, prefer_recovery: bool = False) -> list[dict[str, object]]:
            return []

        async def list_sessions(self) -> list[object]:
            return []

    panel = _FakePanel()
    input_bar = _FakeInputBar()
    status = _FakeStatusBar()
    sidebar = _FakeSidebarPanel()

    def query_one(cls: type):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "StatusBar":
            return status
        if cls.__name__ == "SidebarPanel":
            return sidebar
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _restoring_session=True,
        _last_usage_tokens=0,
        _last_total_session_tokens=0,
        context_usage_state=ContextUsageState.with_window(
            used_tokens=0,
            max_context_tokens=100_000,
            total_session_tokens=0,
        ),
        _state_store=_FakeStateStore(),
        query_one=query_one,
        _set_has_messages=lambda value: calls.append(("has_messages", value)),
        _set_terminal_title_for_cwd=lambda cwd: None,
        _reset_session_title_state=lambda: None,
        _set_session_title_state=lambda **_kwargs: None,
        _session_custom_title="",
        _events=SimpleNamespace(finish_agent_load=lambda message: None),
        _debug=lambda *_args: None,
    )

    asyncio.run(
        _make_session_handler(screen).on_session_restored(
            SessionRestored(
                session_id="session-with-todos",
                agent_profile="Code",
                display_name="Code Agent",
                message_count=0,
                primary_cwd="/old/missing/path",
            )
        )
    )

    assert screen.todo_state == TodoListState(
        items=(
            TodoItem(content="restored", status="completed"),
            TodoItem(content="kept"),
        )
    )


def test_session_restore_without_saved_todos_clears_stale_todo_state() -> None:
    """Switching to a session without todos must clear the previous session's list."""

    class _FakePanel:
        border_subtitle = None

        async def clear(self) -> None:
            return

        def set_session_id(self, session_id: str) -> None:
            return

        def set_workspace_cwd(self, cwd: str) -> None:
            self.border_subtitle = Text(cwd)

        def update_usage(self, tokens: int, total_session_tokens: int = 0) -> None:
            return

        async def add_error(self, message: str, *, action_label: str | None = "Retry") -> None:
            return

    class _FakeInputBar:
        retry_mode = True

        def set_paste_cwd(self, cwd: str) -> None:
            return

        def set_clipboard_image_dir(self, directory: object) -> None:
            return

    class _FakeStatusBar:
        def flash(self, text: str) -> None:
            return

    class _FakeContextPanel:
        def clear_blocks(self) -> None:
            return

    class _FakeSidebarPanel:
        context_panel = _FakeContextPanel()

    panel = _FakePanel()
    input_bar = _FakeInputBar()
    status = _FakeStatusBar()
    sidebar = _FakeSidebarPanel()

    def query_one(cls: type):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "StatusBar":
            return status
        if cls.__name__ == "SidebarPanel":
            return sidebar
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _restoring_session=True,
        _last_usage_tokens=0,
        _last_total_session_tokens=0,
        context_usage_state=None,
        _state_store=None,
        todo_state=TodoListState(items=(TodoItem(content="stale"),)),
        query_one=query_one,
        _set_has_messages=lambda value: None,
        _set_terminal_title_for_cwd=lambda cwd: None,
        _reset_session_title_state=lambda: None,
        _set_session_title_state=lambda **_kwargs: None,
        _session_custom_title="",
        _events=SimpleNamespace(finish_agent_load=lambda message: None),
        _debug=lambda *_args: None,
    )

    asyncio.run(
        _make_session_handler(screen).on_session_restored(
            SessionRestored(
                session_id="session-bare",
                agent_profile="Code",
                display_name="Code Agent",
                message_count=1,
                primary_cwd="/old/missing/path",
            )
        )
    )

    assert screen.todo_state == TodoListState()


def test_welcome_rollback_clears_todo_state() -> None:
    """Rolling back to turn 0 clears the Tasks panel alongside the welcome reset."""

    class _FakePanel:
        border_subtitle = None

        async def clear(self) -> None:
            return

        def set_session_id(self, session_id: str) -> None:
            return

        def set_workspace_cwd(self, cwd: str) -> None:
            self.border_subtitle = Text(cwd)

        def update_welcome(self, *, profile: str = "", cwd: str = "") -> None:
            return

    class _FakeInputBar:
        retry_mode = True
        has_messages = True

        def set_paste_cwd(self, cwd: str) -> None:
            return

        def set_clipboard_image_dir(self, directory: object) -> None:
            return

    class _FakeContextPanel:
        def reset(self) -> None:
            return

    class _FakeSidebarPanel:
        context_panel = _FakeContextPanel()

    panel = _FakePanel()
    input_bar = _FakeInputBar()
    sidebar = _FakeSidebarPanel()

    def query_one(cls: type):
        if cls.__name__ == "ChatPanel":
            return panel
        if cls.__name__ == "InputBar":
            return input_bar
        if cls.__name__ == "SidebarPanel":
            return sidebar
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _engine=None,
        _profile="Code Agent",
        _has_messages=True,
        _chdir_original_cwd="/repo/original",
        _chdir_current_cwd="/repo/current",
        _workspace_cwd=lambda: "/repo/current",
        context_usage_state=None,
        todo_state=TodoListState(items=(TodoItem(content="obsolete", status="in_progress"),)),
        query_one=query_one,
        _update_toc=lambda: None,
        _set_terminal_title_for_cwd=lambda cwd=None: None,
        _reset_session_title_state=lambda: None,
        notify=lambda *_args, **_kwargs: None,
        _debug=lambda *_args: None,
    )
    screen._set_has_messages = lambda value: setattr(screen, "_has_messages", value)

    asyncio.run(
        _make_rollback_controller_for_test(screen).on_result(RollbackResult(session_id="session-1", target_turn=0))
    )

    assert screen.todo_state == TodoListState()
