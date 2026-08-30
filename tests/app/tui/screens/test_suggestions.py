# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for slash command suggestions and dispatch."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

import pytest
from textual import events
from textual.app import App, ComposeResult
from textual.content import Content
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.style import Style
from textual.widgets import Static

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.main import suggestions as suggestions_module
from chrys.app.tui.screens.main.commands import SlashCommandActions
from chrys.app.tui.screens.main.state import MainScreenServices, MainScreenState
from chrys.app.tui.screens.main.suggestions import SuggestionCallbacks, SuggestionHandler
from chrys.app.tui.screens.main.view_adapter import MainScreenViewAdapter
from chrys.app.tui.widgets.chrome.commands import (
    ManPageProseBlock,
    ManPageVerbatimBlock,
    is_slash_command_candidate,
)
from chrys.app.tui.widgets.chrome.file_scanner import ProjectPathScanResult, ProjectPathSuggestion
from chrys.app.tui.widgets.chrome.suggestion_list import SuggestionItem, SuggestionList
from chrys.app.tui.widgets.loading import ChrysLoadingIndicator
from chrys.app.tui.widgets.marquee import OverflowMarquee
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import AgentRuntimeDetails, RuntimeSkillDetails
from chrys.foundation.i18n import MessageRef
from chrys.foundation.i18n.formatting import format_message
from chrys.service.approval.policy import ApprovalMode
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile


class _TaskWorker:
    def __init__(self, work) -> None:
        self._task = asyncio.create_task(work)

    @property
    def is_finished(self) -> bool:
        return self._task.done()

    async def wait(self):
        return await self._task


class _DeferredWorker:
    def __init__(self, work) -> None:
        self._work = work
        self._task: asyncio.Task | None = None

    @property
    def is_finished(self) -> bool:
        return self._task is not None and self._task.done()

    async def wait(self):
        if self._task is None:
            self._task = asyncio.create_task(self._work)
        return await self._task


class _SuggestionListStub:
    def __init__(self) -> None:
        self.last_mode: str | None = None
        self.last_items: list[object] = []
        self.last_title: str | None = None
        self.is_visible = False
        self.is_loading = False
        self.select_result = False

    def show(self, mode: str, items=None, *_args, **_kwargs) -> None:
        self.last_mode = mode
        self.last_items = list(items or [])
        self.last_title = _kwargs.get("title")
        self.is_visible = True
        self.is_loading = False

    def show_loading(self, mode: str, *, title: str = "") -> None:
        self.last_mode = mode
        self.last_items = []
        self.last_title = title
        self.is_visible = True
        self.is_loading = True

    def update(self, items=None, *_args, **_kwargs) -> None:
        self.last_items = list(items or [])
        self.is_loading = False
        title = _kwargs.get("title")
        if title is not None:
            self.last_title = title

    def hide(self) -> None:
        self.last_mode = ""
        self.is_visible = False
        self.is_loading = False
        return

    def select_highlighted(self, *, execute: bool = False) -> bool:
        _ = execute
        return self.select_result

    @property
    def mode(self) -> str:
        return self.last_mode or ""


@dataclass(frozen=True, slots=True)
class _AgentProfileStub:
    name: str
    display_name: str = ""
    description: str = ""


class _AgentRegistryStub:
    def __init__(self, profiles: list[_AgentProfileStub]) -> None:
        self._profiles = profiles

    def list_profiles(self) -> list[_AgentProfileStub]:
        return list(self._profiles)


class _DismissInputBarStub:
    def __init__(self) -> None:
        self.value = ""
        self.replacements: list[tuple[str, str]] = []
        self.prompt_history: list[str] = []
        self.prompt_history_limits: list[int] = []
        self.suggestions_active = False
        self.suggestion_mode: str | None = None

    def set_suggestions_active(self, active: bool, *, mode: str | None = None) -> None:
        self.suggestions_active = active
        self.suggestion_mode = mode if active else None

    def focus_input(self) -> None:
        return

    def replace_trigger_text(self, trigger: str, replacement: str) -> None:
        self.replacements.append((trigger, replacement))

    async def load_prompt_history(self, *, max_entries: int) -> list[str]:
        self.prompt_history_limits.append(max_entries)
        return list(self.prompt_history)


class _SuggestionListApp(App):
    def __init__(self) -> None:
        self.selected: list[tuple[str, str, bool, str]] = []
        super().__init__()

    def compose(self) -> ComposeResult:
        yield SuggestionList()

    def on_suggestion_list_selected(self, event: SuggestionList.Selected) -> None:
        self.selected.append((event.text, event.mode, event.execute, event.kind))


class _SuggestionScreen:
    def __init__(self) -> None:
        self.app = type("_App", (), {"available_themes": ["textual-dark"], "theme": "textual-dark"})()
        self.state = MainScreenState()
        self.services = MainScreenServices(bus=EventBus(), state_store=object())
        self._agent_running = False
        self._profile = "Code"
        self._chdir_current_cwd = ""
        self._runtime_details = AgentRuntimeDetails()
        self.is_attached = True
        self.opened: list[str] = []
        self.notifications: list[str] = []
        self.submitted: list[str] = []
        self.picked_models: list[str] = []
        self.fork_requests = 0
        self.clear_requests = 0
        self.title_editor_requests = 0
        self.applied_titles: list[str] = []
        self.suggestion_list = _SuggestionListStub()
        self.input_bar = _DismissInputBarStub()

    def _debug(self, *_args, **_kwargs) -> None:
        return

    def action_pick_theme(self) -> None:
        return

    def _resume_last_session(self) -> None:
        return

    def _fork_current_session(self) -> None:
        self.fork_requests += 1

    def _open_session_title_editor(self) -> None:
        self.title_editor_requests += 1

    def _apply_session_title_from_command(self, custom_title: str) -> None:
        self.applied_titles.append(custom_title)

    def _create_new_session(self) -> None:
        return

    def _clear_current_session(self) -> None:
        self.clear_requests += 1

    def action_quit(self) -> None:
        return

    def action_sessions(self) -> None:
        return

    def _chdir(self, _arg: str) -> None:
        return

    def _copy_agent_responses(self, _arg: str) -> None:
        return

    def _toggle_fold(self) -> None:
        return

    def action_show_diff(self) -> None:
        return

    def action_show_rollback(self, _arg: str = "") -> None:
        return

    def _set_approval_mode(self, _arg: str) -> None:
        return

    def _open_model_config(self) -> None:
        return

    def _open_agent_config(self) -> None:
        self.opened.append("agent")

    def _open_agent_config_tab(self, tab: str) -> None:
        self.opened.append(tab)

    def action_runtime_details(self) -> None:
        self.opened.append("runtime")

    def _open_settings(self, tab: str) -> None:
        self.opened.append(f"settings:{tab}")

    def _submit_user_text(self, text: str) -> None:
        self.submitted.append(text)

    def notify(self, message: MessageRef | str, **_kwargs) -> None:
        self.notifications.append(format_message(message) if isinstance(message, MessageRef) else message)

    def run_worker(self, work, **_kwargs) -> _TaskWorker:
        return _TaskWorker(work)

    def query_one(self, cls):
        name = getattr(cls, "__name__", "")
        if name == "InputBar":
            return self.input_bar
        return self.suggestion_list

    @property
    def _agent_running(self) -> bool:
        return self.state.run.agent_running

    @_agent_running.setter
    def _agent_running(self, value: bool) -> None:
        self.state.run.agent_running = value

    @property
    def _state_store(self) -> object | None:
        return self.services.state_store

    @_state_store.setter
    def _state_store(self, value: object | None) -> None:
        self.services.state_store = value

    @property
    def _profile(self) -> str:
        return self.state.runtime.profile

    @_profile.setter
    def _profile(self, value: str) -> None:
        self.state.runtime.profile = value

    @property
    def _agent_registry(self) -> object | None:
        return self.services.agent_registry

    @_agent_registry.setter
    def _agent_registry(self, value: object | None) -> None:
        self.services.agent_registry = value

    @property
    def _chdir_current_cwd(self) -> str:
        return self.state.workspace_marker.current_cwd

    @_chdir_current_cwd.setter
    def _chdir_current_cwd(self, value: str) -> None:
        self.state.workspace.current_cwd = value
        self.state.workspace_marker.current_cwd = value

    @property
    def _runtime_details(self) -> AgentRuntimeDetails:
        return self.state.runtime.details

    @_runtime_details.setter
    def _runtime_details(self, value: AgentRuntimeDetails) -> None:
        self.state.runtime.details = value


def _make_slash_actions(screen: _SuggestionScreen) -> SlashCommandActions:
    return SlashCommandActions(
        list_themes=lambda: sorted(screen.app.available_themes),
        get_theme=lambda: screen.app.theme,
        apply_theme=lambda name: setattr(screen.app, "theme", name),
        pick_theme=screen.action_pick_theme,
        list_languages=list,
        get_language=lambda: "system",
        apply_language=lambda _requested_locale: None,
        pick_language=lambda: None,
        render_unknown_language_warning=lambda requested_locale: f"Unknown /language locale: {requested_locale}",
        debug_event=screen._debug,
        new_session=screen._create_new_session,
        clear_session=screen._clear_current_session,
        quit_app=screen.action_quit,
        resume_session=screen._resume_last_session,
        fork_session=screen._fork_current_session,
        browse_session_list=screen.action_sessions,
        edit_session_title=screen._open_session_title_editor,
        apply_session_title=screen._apply_session_title_from_command,
        change_directory=screen._chdir,
        copy_conversation=screen._copy_agent_responses,
        fold_tools=screen._toggle_fold,
        open_diff=screen.action_show_diff,
        open_rollback=screen.action_show_rollback,
        get_approval_mode=lambda: ApprovalMode.MANUAL.value,
        change_approval_mode=screen._set_approval_mode,
        configure_model=screen._open_model_config,
        configure_agent=screen._open_agent_config,
        configure_agent_tab=screen._open_agent_config_tab,
        show_runtime_details=screen.action_runtime_details,
        configure_settings=screen._open_settings,
        show_manual_pages=lambda _pages, _start_index: None,
        warn=lambda message, title, timeout: screen.notify(message, title=title, timeout=timeout),
    )


def _make_handler(
    screen: _SuggestionScreen,
    *,
    locale_controller: LocaleController | None = None,
) -> SuggestionHandler:
    view = MainScreenViewAdapter(screen)  # type: ignore[arg-type]
    return SuggestionHandler(
        state=screen.state,
        services=screen.services,
        view=view,
        command_actions=_make_slash_actions(screen),
        callbacks=SuggestionCallbacks(
            notify_warning=lambda message, title, timeout: screen.notify(message, title=title, timeout=timeout),
            show_file_suggestions=lambda: None,
            submit_user_text=screen._submit_user_text,
            start_agent_profile_switch=lambda _profile: None,
            start_model_profile_switch=screen.picked_models.append,
        ),
        buddy_view=view,
        locale_controller=locale_controller,
    )


def test_view_adapter_suggestions_ignore_teardown_nomatches() -> None:
    screen = _SuggestionScreen()

    def query_one(cls):
        if getattr(cls, "__name__", "") == "SuggestionList":
            raise NoMatches("SuggestionList")
        return screen.input_bar

    screen.query_one = query_one
    view = MainScreenViewAdapter(screen)  # type: ignore[arg-type]

    view.show_suggestions("files", [SuggestionItem(value="a.py", label="a.py", kind="file")])
    view.show_suggestions_loading("files", title="Files")
    view.update_suggestions([SuggestionItem(value="b.py", label="b.py", kind="file")])
    view.hide_suggestions()

    assert screen.input_bar.replacements == []


def _suggestion_values(items: list[object]) -> list[str]:
    values: list[str] = []
    for item in items:
        if isinstance(item, SuggestionItem):
            values.append(item.value)
        else:
            value, _label = item  # type: ignore[misc]
            values.append(value)
    return values


def _scan_result(root: str, paths: list[ProjectPathSuggestion], **kwargs: object) -> ProjectPathScanResult:
    return ProjectPathScanResult.from_suggestions(root=root, paths=paths, **kwargs)


async def _wait_for_file_query(handler: SuggestionHandler) -> None:
    worker = handler._file_query_worker
    if worker is not None:
        await worker.wait()


def test_slash_command_candidate_rejects_code_comments_paths_and_multiline_pastes() -> None:
    rejected = [
        "// comment",
        "/// doc comment",
        "/* block comment */",
        "/** jsdoc */",
        "/",
        "/ path-ish text",
        "/123",
        "/-flag",
        "/path/to/file",
        "/Users/foo",
        "/regex/i",
        "/help\nmore text",
        "  /help",
    ]

    for text in rejected:
        assert not is_slash_command_candidate(text)


def test_slash_command_candidate_accepts_command_shaped_input() -> None:
    assert is_slash_command_candidate("/help")
    assert is_slash_command_candidate("/help arg")
    assert is_slash_command_candidate("/chdir /Users/foo")
    assert is_slash_command_candidate("\uff0fhelp")


@pytest.mark.asyncio
async def test_suggestion_list_groups_items_and_selects_first_enabled_item() -> None:
    async with _SuggestionListApp().run_test() as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        suggestion_list.show(
            "commands",
            [
                SuggestionItem(
                    value="new",
                    label="/new",
                    section="System Commands",
                    kind="command",
                    disabled=True,
                ),
                SuggestionItem(
                    value="review",
                    label="/review",
                    section="Loaded Skills",
                    kind="skill",
                ),
            ],
        )

        assert len(suggestion_list._contents) == 4
        assert str(suggestion_list._contents[0]) == "System Commands"
        assert str(suggestion_list._contents[2]) == "Loaded Skills"
        assert suggestion_list._highlighted == 3
        assert suggestion_list.select_highlighted(execute=True) is True
        await pilot.pause()
        suggestion_list._highlighted = 1
        assert suggestion_list.select_highlighted(execute=True) is False
        await pilot.pause()

    assert pilot.app.selected == [("review", "commands", True, "skill")]


@pytest.mark.asyncio
async def test_suggestion_list_navigation_wraps_and_skips_disabled_items() -> None:
    async with _SuggestionListApp().run_test() as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        suggestion_list.show(
            "commands",
            [
                SuggestionItem(value="new", label="/new", section="System Commands"),
                SuggestionItem(value="exit", label="/exit", section="System Commands", disabled=True),
                SuggestionItem(value="theme", label="/theme", section="System Commands"),
                SuggestionItem(value="review", label="/review", section="Loaded Skills"),
            ],
        )

        assert suggestion_list._highlighted == 1

        suggestion_list.move_cursor_up()
        assert suggestion_list._highlighted == 5

        suggestion_list.move_cursor_down()
        assert suggestion_list._highlighted == 1

        suggestion_list.move_cursor_down()
        assert suggestion_list._highlighted == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["commands", "agents", "models", "history"])
async def test_suggestion_list_marquee_has_one_race_safe_timer_and_resets_static(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    class _FakeTimer:
        def __init__(self, callback) -> None:
            self.callback = callback
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True

        def fire(self) -> None:
            self.stopped = True
            self.callback()

    marquee = OverflowMarquee(start_delay=10.0, step_interval=10.0, end_delay=10.0)
    app = _SuggestionListApp()
    async with app.run_test(size=(40, 20)) as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        suggestion_list._marquee = marquee
        timers: list[_FakeTimer] = []

        def set_timer(_delay: float, callback) -> _FakeTimer:
            timer = _FakeTimer(callback)
            timers.append(timer)
            return timer

        monkeypatch.setattr(suggestion_list, "set_timer", set_timer)
        first_label = Content.assemble("/first  ", ("A description that is far too long for this popup", "dim"))
        second_label = Content.assemble("/second  ", ("Another description that also exceeds the popup", "dim"))
        suggestion_list.show(
            mode,
            [
                SuggestionItem(value="first", label=first_label, marquee_start=len("/first  ")),
                SuggestionItem(value="second", label=second_label, marquee_start=len("/second  ")),
            ],
        )
        await pilot.pause()

        active_timers = [timer for timer in timers if not timer.stopped]
        assert len(active_timers) == 1
        first_generation_timer = active_timers[0]
        assert str(suggestion_list.render()).splitlines()[0] == first_label.plain

        first_generation_timer.fire()
        scrolling_first_row = str(suggestion_list.render()).splitlines()[0]
        assert scrolling_first_row.startswith("/first  ")
        assert scrolling_first_row != first_label.plain
        scrolling_timer = next(timer for timer in timers if not timer.stopped)

        suggestion_list.move_cursor_down()
        assert scrolling_timer.stopped is True
        assert str(suggestion_list.render()).splitlines() == [first_label.plain, second_label.plain]
        replacement_timer = next(timer for timer in timers if not timer.stopped)

        # A callback already queued by Textual before stop() must not advance
        # the replacement selection or disturb its one live timer.
        first_generation_timer.callback()
        assert next(timer for timer in timers if not timer.stopped) is replacement_timer
        assert str(suggestion_list.render()).splitlines() == [first_label.plain, second_label.plain]

        monkeypatch.setattr(suggestion_list, "_is_on_current_screen", lambda: False)
        replacement_timer.fire()
        assert marquee.active is False
        assert [timer for timer in timers if not timer.stopped] == []

        monkeypatch.setattr(suggestion_list, "_is_on_current_screen", lambda: True)
        suggestion_list.resume_marquee()
        resumed_timer = next(timer for timer in timers if not timer.stopped)
        suggestion_list.pause_marquee()
        assert resumed_timer.stopped is True
        assert marquee.active is False
        assert [timer for timer in timers if not timer.stopped] == []

        suggestion_list.resume_marquee()
        assert len([timer for timer in timers if not timer.stopped]) == 1
        suggestion_list.hide()
        assert marquee.active is False
        assert [timer for timer in timers if not timer.stopped] == []


@pytest.mark.asyncio
async def test_suggestion_list_does_not_animate_files_or_fitting_labels() -> None:
    async with _SuggestionListApp().run_test(size=(40, 20)) as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)

        suggestion_list.show("files", [("long", "x" * 100)])
        assert suggestion_list._marquee_timer is None

        suggestion_list.show(
            "commands",
            [SuggestionItem(value="short", label="/short  Fits", marquee_start=len("/short  "))],
        )
        assert suggestion_list._marquee_timer is None


def test_suggestion_list_fired_timer_cancels_after_visibility_is_lost() -> None:
    class _FakeTimer:
        stopped = False

        def stop(self) -> None:
            self.stopped = True

    marquee = OverflowMarquee()
    suggestion_list = SuggestionList(marquee=marquee)
    timer = _FakeTimer()
    suggestion_list._marquee_timer = timer  # type: ignore[assignment]
    suggestion_list.visible = False
    marquee.activate(Content("a long description"), viewport_width=4)

    suggestion_list._advance_marquee(suggestion_list._marquee_generation)

    assert timer.stopped is True
    assert suggestion_list._marquee_timer is None
    assert marquee.active is False


@pytest.mark.asyncio
async def test_suggestion_marquee_reveals_end_then_restores_ellipsized_static_frame() -> None:
    class _OverlayApp(App):
        CSS = "#filler { height: 8; }"

        def compose(self) -> ComposeResult:
            yield Static("filler", id="filler")
            yield SuggestionList(marquee=OverflowMarquee(start_delay=60.0, step_interval=60.0, end_delay=60.0))

    async with _OverlayApp().run_test(size=(40, 10)) as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        label = Content.assemble(
            "/review  ",
            ("Review, refactor, and debug every changed file in the workspace", "dim"),
        )
        suggestion_list.show(
            "commands",
            [SuggestionItem(value="review", label=label, marquee_start=len("/review  "))],
        )
        await pilot.pause()
        await pilot.pause()

        def selected_row() -> str:
            rows = [strip.text for strip in pilot.app.screen._compositor.render_strips()]
            return next(row for row in rows if row.startswith("│") and row.endswith("│"))

        static_row = selected_row()
        assert "/review" in static_row
        assert "…" in static_row

        while suggestion_list._marquee.frame.cell_length > suggestion_list.content_size.width:
            assert suggestion_list._marquee.frame.plain.startswith("/review  ")
            suggestion_list._marquee.advance()
        assert suggestion_list._marquee.frame.plain.startswith("/review  ")
        suggestion_list.refresh()
        await pilot.pause()
        end_row = selected_row()
        assert "/review" in end_row
        assert "the workspace" in end_row
        assert "…" not in end_row

        suggestion_list._marquee.reset()
        suggestion_list.refresh()
        await pilot.pause()
        assert selected_row() == static_row


@pytest.mark.asyncio
async def test_suggestion_marquee_repaints_static_frame_after_screen_resume() -> None:
    class _MarqueeScreen(Screen):
        CSS = "#filler { height: 8; } SuggestionList { offset-y: 0; }"

        def compose(self) -> ComposeResult:
            yield Static("filler", id="filler")
            yield SuggestionList(marquee=OverflowMarquee(start_delay=60.0, step_interval=60.0, end_delay=60.0))

        def on_screen_suspend(self) -> None:
            self.query_one(SuggestionList).pause_marquee()

        def on_screen_resume(self) -> None:
            self.query_one(SuggestionList).resume_marquee()

    class _CoveringScreen(Screen):
        def compose(self) -> ComposeResult:
            yield Static("Covering screen")

    class _ScreenStackApp(App):
        def on_mount(self) -> None:
            self.push_screen(_MarqueeScreen())

    async with _ScreenStackApp().run_test(size=(40, 12)) as pilot:
        suggestion_list = pilot.app.screen.query_one(SuggestionList)
        label = Content.assemble(
            "/review  ",
            ("Review, refactor, and debug every changed file in the workspace", "dim"),
        )
        suggestion_list.show(
            "commands",
            [SuggestionItem(value="review", label=label, marquee_start=len("/review  "))],
        )
        await pilot.pause()
        await pilot.pause()

        def selected_row() -> str:
            rows = [strip.text for strip in pilot.app.screen._compositor.render_strips()]
            return next(row for row in rows if "/review" in row)

        static_row = selected_row()
        suggestion_list._marquee.advance()
        suggestion_list.refresh()
        await pilot.pause()
        assert selected_row() != static_row

        pilot.app.push_screen(_CoveringScreen())
        await pilot.pause()
        await pilot.app.pop_screen()
        await pilot.pause()

        assert selected_row() == static_row


@pytest.mark.asyncio
async def test_suggestion_list_long_window_tracks_wrapped_highlight() -> None:
    async with _SuggestionListApp().run_test() as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        suggestion_list.show(
            "commands",
            [SuggestionItem(value=f"item-{index}", label=f"Item {index}") for index in range(15)],
        )

        assert suggestion_list._highlighted == 0
        assert suggestion_list._window_start == 0

        suggestion_list._hovered_row = 2
        suggestion_list.move_cursor_up()
        assert suggestion_list._highlighted == 14
        assert suggestion_list._window_start == 3
        assert suggestion_list._hovered_row is None
        assert "Item 14" in str(suggestion_list.render())

        suggestion_list._hovered_row = 2
        suggestion_list.move_cursor_down()
        assert suggestion_list._highlighted == 0
        assert suggestion_list._window_start == 0
        assert suggestion_list._hovered_row is None


@pytest.mark.asyncio
async def test_suggestion_list_wrap_to_first_item_keeps_section_header_visible() -> None:
    async with _SuggestionListApp().run_test() as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        suggestion_list.show(
            "commands",
            [
                SuggestionItem(
                    value=f"command-{index}",
                    label=f"/command-{index}",
                    section="System Commands",
                )
                for index in range(15)
            ],
        )

        assert suggestion_list._highlighted == 1
        assert suggestion_list._window_start == 0

        suggestion_list.move_cursor_up()
        assert suggestion_list._highlighted == 15
        assert suggestion_list._window_start == 4

        suggestion_list.move_cursor_down()
        assert suggestion_list._highlighted == 1
        assert suggestion_list._window_start == 0
        assert str(suggestion_list.render()).splitlines()[0] == "System Commands"


def _mouse_event(event_type: type[events.MouseEvent], x: int = 0, y: int = 0) -> events.MouseEvent:
    return event_type(None, x, y, 0, 0, 1, False, False, False)


def _wheel_event(event_type: type[events.MouseEvent]) -> events.MouseEvent:
    return _mouse_event(event_type)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["commands", "files", "agents", "models"])
async def test_suggestion_list_hover_paints_selectable_row_without_layout(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """The shared /, @, #, and $ popup gets TOC-like hover without a reflow."""

    class _HoverSuggestionListApp(_SuggestionListApp):
        CSS = "SuggestionList { offset-y: 0; }"

    async with _HoverSuggestionListApp().run_test() as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        suggestion_list.show(
            mode,
            [
                SuggestionItem(value="first", label="First"),
                SuggestionItem(value="second", label="Second"),
                SuggestionItem(value="disabled", label="Disabled", disabled=True),
            ],
        )
        await pilot.pause()
        layout_refreshes: list[None] = []
        monkeypatch.setattr(pilot.app.screen, "_refresh_layout", lambda: layout_refreshes.append(None))

        content_x = suggestion_list.gutter.left
        second_row_y = suggestion_list.gutter.top + 1
        assert await pilot.hover(suggestion_list, offset=(content_x, second_row_y)) is True
        await pilot.pause()

        assert suggestion_list._hovered_row == 1
        hovered_row = suggestion_list.render().split()[1]
        hover_span_style = hovered_row.spans[-1].style
        assert isinstance(hover_span_style, Style)
        assert (
            hover_span_style.background
            == suggestion_list.get_component_styles("suggestion-list--option-hover").background
        )
        assert hovered_row.cell_length == suggestion_list.content_size.width
        assert layout_refreshes == []

        selected_row_y = suggestion_list.gutter.top
        assert await pilot.hover(suggestion_list, offset=(content_x, selected_row_y)) is True
        await pilot.pause()
        assert suggestion_list._hovered_row == 0
        hovered_selected_row = suggestion_list.render().split()[0]
        selected_hover_style = hovered_selected_row.spans[-1].style
        assert isinstance(selected_hover_style, Style)
        assert (
            selected_hover_style.background
            == suggestion_list.get_component_styles("suggestion-list--option-hover").background
        )
        assert hovered_selected_row.cell_length == suggestion_list.content_size.width
        assert layout_refreshes == []

        disabled_row_y = suggestion_list.gutter.top + 2
        assert await pilot.hover(suggestion_list, offset=(content_x, disabled_row_y)) is True
        await pilot.pause()
        assert suggestion_list._hovered_row is None
        assert layout_refreshes == []

        assert await pilot.hover(suggestion_list, offset=(content_x, second_row_y)) is True
        await pilot.hover(offset=(0, 10))
        await pilot.pause()
        assert suggestion_list._hovered_row is None
        assert layout_refreshes == []


@pytest.mark.asyncio
async def test_suggestion_list_click_maps_rows_through_border() -> None:
    """Click coordinates are widget-relative: border rows AND border columns
    must select nothing while content cells map to their items."""
    app = _SuggestionListApp()
    async with app.run_test() as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        suggestion_list.show(
            "commands",
            [
                SuggestionItem(value="first", label="First"),
                SuggestionItem(value="second", label="Second"),
            ],
        )
        await pilot.pause()

        top = suggestion_list.gutter.top
        suggestion_list.on_click(_mouse_event(events.Click, x=2, y=top - 1))  # border row
        await pilot.pause()
        assert app.selected == []

        suggestion_list.on_click(_mouse_event(events.Click, x=2, y=top + 1))  # second item
        await pilot.pause()
        assert [selected[0] for selected in app.selected] == ["second"]

        # Clicks on the vertical border columns must not select the row.
        left_border_x = suggestion_list.gutter.left - 1
        right_border_x = suggestion_list.size.width - suggestion_list.gutter.right
        suggestion_list.on_click(_mouse_event(events.Click, x=left_border_x, y=top))
        suggestion_list.on_click(_mouse_event(events.Click, x=right_border_x, y=top))
        await pilot.pause()
        assert [selected[0] for selected in app.selected] == ["second"]

        # The first content column still selects.
        suggestion_list.on_click(_mouse_event(events.Click, x=suggestion_list.gutter.left, y=top))
        await pilot.pause()
        assert [selected[0] for selected in app.selected] == ["second", "first"]


@pytest.mark.asyncio
async def test_suggestion_list_wheel_scrolls_window_without_wrapping() -> None:
    """The wheel pans the viewport like the replaced OptionList: the
    highlight stays put and panning clamps at both edges instead of wrapping."""
    async with _SuggestionListApp().run_test() as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        suggestion_list.show(
            "commands",
            [SuggestionItem(value=f"item-{index}", label=f"Item {index}") for index in range(15)],
        )

        assert suggestion_list._highlighted == 0
        assert suggestion_list._window_start == 0

        # Wheel-up at the top edge: no wrap to the last item.
        suggestion_list._hovered_row = 1
        suggestion_list._on_mouse_scroll_up(_wheel_event(events.MouseScrollUp))
        assert suggestion_list._highlighted == 0
        assert suggestion_list._window_start == 0
        assert suggestion_list._hovered_row == 1

        # Wheel-down pans the window and leaves the highlight in place.
        suggestion_list._on_mouse_scroll_down(_wheel_event(events.MouseScrollDown))
        assert suggestion_list._highlighted == 0
        assert suggestion_list._window_start == 1
        assert suggestion_list._hovered_row is None

        # Panning clamps at the bottom edge instead of wrapping.
        for _ in range(20):
            suggestion_list._on_mouse_scroll_down(_wheel_event(events.MouseScrollDown))
        assert suggestion_list._window_start == 3  # 15 items - 12 visible rows
        assert suggestion_list._highlighted == 0
        assert "Item 14" in str(suggestion_list.render())

        # And clamps again at the top on the way back.
        for _ in range(20):
            suggestion_list._on_mouse_scroll_up(_wheel_event(events.MouseScrollUp))
        assert suggestion_list._window_start == 0


@pytest.mark.asyncio
async def test_suggestion_list_flattens_multiline_labels_to_one_row_each() -> None:
    """Window math and click mapping assume one physical row per item."""
    multiline_label = "first line\nvalue [type=missing, input_value={}, input_type=dict])"
    async with _SuggestionListApp().run_test() as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        suggestion_list.show(
            "commands",
            [
                SuggestionItem(value="multi", label=multiline_label),
                SuggestionItem(value="plain", label="plain item", section="Bad\nSection"),
            ],
        )

        rendered = str(suggestion_list.render())
        # One physical row per entry (item, section, item): any newline inside
        # a label would shift every row under the mouse-mapping math.
        assert rendered.count("\n") == 2
        assert multiline_label.replace("\n", " ") in rendered

        assert suggestion_list._values[suggestion_list._index_at_y(0)] == "multi"
        assert suggestion_list._index_at_y(1) is not None  # flattened section row
        assert suggestion_list._values[suggestion_list._index_at_y(2)] == "plain"
        assert suggestion_list._index_at_y(3) is None


@pytest.mark.asyncio
async def test_suggestion_list_height_tracks_row_count() -> None:
    """A short result set must not blank transcript rows above it.

    An oversized overlay paints opaque empty rows over the chat and swallows
    the mouse events aimed there; the box must hug its rendered rows.
    """
    async with _SuggestionListApp().run_test() as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)

        border_rows = suggestion_list.gutter.height  # separator border around the overlay
        suggestion_list.show("commands", [SuggestionItem(value="one", label="only item")])
        await pilot.pause()
        assert suggestion_list.region.height == 1 + border_rows

        suggestion_list.update([SuggestionItem(value=f"i{n}", label=f"Item {n}") for n in range(20)])
        await pilot.pause()
        assert suggestion_list.region.height == 12 + border_rows

        suggestion_list.update([SuggestionItem(value="a", label="A"), SuggestionItem(value="b", label="B")])
        await pilot.pause()
        assert suggestion_list.region.height == 2 + border_rows


@pytest.mark.asyncio
async def test_suggestion_list_loading_state_contains_only_chrys_indicator_until_ready() -> None:
    """Cold async sources must not flash an empty-state row before results arrive."""
    async with _SuggestionListApp().run_test() as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        loading = suggestion_list.query_one(ChrysLoadingIndicator)

        suggestion_list.show_loading("files", title="Files")
        await pilot.pause()

        assert suggestion_list.mode == "files"
        assert suggestion_list.is_loading is True
        assert loading.display is True
        assert suggestion_list._contents == []
        assert suggestion_list.render().plain == ""
        assert suggestion_list.select_highlighted(execute=True) is False
        assert suggestion_list.region.height == 1 + suggestion_list.gutter.height

        suggestion_list.update([SuggestionItem(value="ready.py", label="ready.py")])
        await pilot.pause()

        assert suggestion_list.is_loading is False
        assert loading.display is False
        assert suggestion_list._values == ["ready.py"]
        assert suggestion_list.render().plain == "ready.py"


@pytest.mark.asyncio
async def test_suggestion_list_surgical_height_patch_matches_real_reflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The typing-path in-place map patch must equal stock reflow geometry.

    update() rewrites the overlay's compositor entries directly instead of
    remapping the whole screen per keystroke; this pins that shortcut to the
    ground truth a real reflow computes, so stock compositor drift fails loudly.
    """
    import chrys.app.tui.widgets.chrome.suggestion_list as suggestion_list_module

    async with _SuggestionListApp().run_test() as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        compositor = pilot.app.screen._compositor

        suggestion_list.show(
            "commands",
            [SuggestionItem(value=f"i{n}", label=f"Item {n}") for n in range(12)],
        )
        await pilot.pause()

        # The typing path must take the O(1) patch, never the full resync.
        def _no_resync(_widget: object) -> None:
            raise AssertionError("update() fell back to a compositor-wide remap")

        monkeypatch.setattr(suggestion_list_module, "resync_compositor_regions", _no_resync)
        # Any repaint of a widget outside the compositor's visible set (e.g.
        # a timer tick under Windows xdist load) arms a full-map rebuild,
        # which update() correctly defers to. Consume it here — no awaits
        # follow before update() — so the patch precondition holds and this
        # test pins the typing path, not ambient scheduling noise.
        assert compositor.full_map.get(suggestion_list) is not None
        assert not compositor._full_map_invalidated
        suggestion_list.update([SuggestionItem(value="a", label="A"), SuggestionItem(value="b", label="B")])
        patched = compositor._full_map.get(suggestion_list)
        assert patched is not None
        assert patched.region.height == 2 + suggestion_list.gutter.height

        pilot.app.screen._refresh_layout()
        await pilot.pause()
        ground_truth = compositor._full_map.get(suggestion_list)
        assert patched == ground_truth


def test_show_suggestions_titles_popup_per_mode() -> None:
    """The popup border names what is being suggested; files mode names the
    same root the file scanner is scoped to."""
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    screen.state.workspace_marker.current_cwd = "/tmp/title-root"

    handler._show_suggestions("commands", [])
    assert screen.suggestion_list.last_title == "Commands"

    handler._show_suggestions("agents", [])
    assert screen.suggestion_list.last_title == "Agents"

    handler._show_suggestions("models", [])
    assert screen.suggestion_list.last_title == "Models"

    handler._show_suggestions("files", [])
    assert screen.suggestion_list.last_title == "Files under /tmp/title-root"

    handler._show_suggestions("history", [])
    assert screen.suggestion_list.last_title == "Prompt History"


def test_prompt_history_suggestions_are_global_recent_first_and_single_line() -> None:
    screen = _SuggestionScreen()
    screen.input_bar.value = "current draft"
    screen.input_bar.prompt_history = ["old prompt", "middle\nline", "latest\r\nprompt"]
    handler = _make_handler(screen)

    asyncio.run(handler.show_prompt_history_async())

    assert screen.input_bar.prompt_history_limits == [100]
    assert screen.suggestion_list.last_mode == "history"
    assert screen.suggestion_list.last_title == "Prompt History"
    items = screen.suggestion_list.last_items
    assert [item.value for item in items if isinstance(item, SuggestionItem)] == [
        "latest\r\nprompt",
        "middle\nline",
        "old prompt",
    ]
    assert [item.label for item in items if isinstance(item, SuggestionItem)] == [
        "latest ↵ prompt",
        "middle ↵ line",
        "old prompt",
    ]
    assert all(item.kind == "history" for item in items if isinstance(item, SuggestionItem))
    assert all(item.marquee_start == 0 for item in items if isinstance(item, SuggestionItem))


@pytest.mark.asyncio
async def test_prompt_history_opens_loading_popup_before_history_is_ready() -> None:
    screen = _SuggestionScreen()
    screen.input_bar.value = "draft"
    started = asyncio.Event()
    release = asyncio.Event()

    async def load_prompt_history(*, max_entries: int) -> list[str]:
        assert max_entries == 100
        started.set()
        await release.wait()
        return ["ready prompt"]

    screen.input_bar.load_prompt_history = load_prompt_history  # type: ignore[method-assign]
    handler = _make_handler(screen)
    revision = handler.start_prompt_history()
    assert screen.suggestion_list.last_mode == "history"
    assert screen.suggestion_list.last_title == "Prompt History"
    assert screen.suggestion_list.is_loading is True
    assert screen.suggestion_list.last_items == []

    task = asyncio.create_task(handler.show_prompt_history_async(revision=revision))
    await started.wait()
    release.set()
    await task

    assert screen.suggestion_list.is_loading is False
    assert _suggestion_values(screen.suggestion_list.last_items) == ["ready prompt"]


@pytest.mark.asyncio
async def test_mode_switch_during_prompt_history_load_does_not_replace_new_suggestions() -> None:
    screen = _SuggestionScreen()
    started = asyncio.Event()
    release = asyncio.Event()

    async def load_prompt_history(*, max_entries: int) -> list[str]:
        assert max_entries == 100
        started.set()
        await release.wait()
        return ["stale prompt"]

    screen.input_bar.load_prompt_history = load_prompt_history  # type: ignore[method-assign]
    handler = _make_handler(screen)
    revision = handler.start_prompt_history()
    task = asyncio.create_task(handler.show_prompt_history_async(revision=revision))
    await started.wait()

    handler._show_suggestions("commands", [SuggestionItem(value="new", label="new")])
    release.set()
    await task

    assert screen.suggestion_list.last_mode == "commands"
    assert screen.suggestion_list.is_loading is False
    assert _suggestion_values(screen.suggestion_list.last_items) == ["new"]


def test_prompt_history_selection_restores_original_multiline_prompt() -> None:
    screen = _SuggestionScreen()
    screen.suggestion_list.is_visible = True
    handler = _make_handler(screen)
    handler._suggestion_mode = "history"

    handler.on_suggestion_selected("history", "first line\nsecond line", execute=False, kind="history")

    assert screen.input_bar.value == "first line\nsecond line"
    assert screen.suggestion_list.is_visible is False
    assert screen.submitted == []


def test_typing_dismisses_prompt_history_suggestions() -> None:
    screen = _SuggestionScreen()
    screen.input_bar.value = "draft"
    screen.suggestion_list.is_visible = True
    handler = _make_handler(screen)
    handler._suggestion_mode = "history"
    handler._prompt_history_draft = "draft"

    handler.on_text_changed("draft changed")

    assert screen.suggestion_list.is_visible is False
    assert handler.suggestion_mode is None


def test_update_suggestions_rerenders_border_title_in_active_locale() -> None:
    """Per-keystroke rebuilds re-supply the title so a popup opened before a
    locale switch does not keep its stale-language border."""
    screen = _SuggestionScreen()
    controller = LocaleController(Settings(locale="zh-Hans"))
    handler = _make_handler(screen, locale_controller=controller)
    handler.build_slash_commands()
    handler._suggestion_mode = "commands"

    handler.on_text_changed("/r")

    assert screen.suggestion_list.last_title == "命令"


@pytest.mark.asyncio
async def test_suggestion_list_update_retitles_and_localizes_empty_state() -> None:
    controller = LocaleController(Settings(locale="zh-Hans"))

    class _TitledApp(App):
        CSS = "#filler { height: 15; }"

        def compose(self) -> ComposeResult:
            yield Static("filler", id="filler")
            yield SuggestionList(locale_controller=controller)

    async with _TitledApp().run_test(size=(60, 20)) as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        suggestion_list.show("commands", [SuggestionItem(value="a", label="a")], title="Commands")
        await pilot.pause()
        english_title = suggestion_list.border_title
        assert english_title is not None

        # No title supplied: the current border is kept as-is.
        suggestion_list.update([SuggestionItem(value="b", label="b")])
        assert suggestion_list.border_title is english_title

        # A supplied title re-renders the border; empty results localize.
        suggestion_list.update([], title="命令")
        await pilot.pause()
        await pilot.pause()
        assert suggestion_list._contents[0].plain == "无结果"
        strips = pilot.app.screen._compositor.render_strips()
        frame = "\n".join(strip.text for strip in strips)
        assert "命令" in frame
        assert "无结果" in frame

        suggestion_list.update([], title="")
        assert suggestion_list.border_title is None


@pytest.mark.asyncio
async def test_suggestion_list_paints_border_title() -> None:
    class _TitledApp(App):
        # The -100% offset overlay needs content above it to paint over,
        # like the transcript it covers in the real layout.
        CSS = "#filler { height: 15; }"

        def compose(self) -> ComposeResult:
            yield Static("filler", id="filler")
            yield SuggestionList()

    async with _TitledApp().run_test(size=(60, 20)) as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        suggestion_list.show(
            "files",
            [SuggestionItem(value="a.py", label="a.py")],
            title="Files under /tmp/proj",
        )
        await pilot.pause()
        await pilot.pause()
        strips = pilot.app.screen._compositor.render_strips()
        frame = "\n".join(strip.text for strip in strips)
        assert "Files under /tmp/proj" in frame

        suggestion_list.hide()
        assert suggestion_list.border_title is None


def test_suggestion_labels_style_second_separator_space_dim() -> None:
    """The dim description span starts at the second separator space, so a
    highlighted row's bright-to-gray reverse-video boundary sits between the
    two spaces instead of hugging the description's first letter."""
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.build_slash_commands()

    items = handler._command_suggestion_items(handler._visible_slash_commands(), set())
    label = items[0].label
    assert isinstance(label, Content)
    name_length = len(f"/{items[0].value}")
    assert label.plain[name_length : name_length + 2] == "  "
    assert any(span.start == name_length + 1 and "dim" in str(span.style) for span in label.spans)
    assert items[0].marquee_start == len(f"/{items[0].value}  ")

    skill_label = SuggestionHandler._runtime_skill_label("review", "Review changes")
    assert skill_label.plain == "/review  Review changes"
    assert any(span.start == len("/review ") and "dim" in str(span.style) for span in skill_label.spans)

    screen._agent_registry = _AgentRegistryStub(
        [_AgentProfileStub(name="QA", display_name="Q&A Agent", description="Read-only assistant")]
    )
    agent_items, _disabled = handler._get_agent_items()
    agent_label = agent_items[0].label
    assert isinstance(agent_label, Content)
    assert agent_label.plain == "  Q&A Agent  Read-only assistant"
    assert any(span.start == len("  Q&A Agent ") and "dim" in str(span.style) for span in agent_label.spans)
    assert agent_items[0].marquee_start == len("  Q&A Agent  ")

    model_registry = ModelProfileRegistry()
    model_registry.register(ModelProfile(id="fast", name="Fast Model", model_id="vendor/fast"))
    screen.services.model_registry = model_registry
    model_items, _disabled = handler._get_model_items()
    model_label = model_items[0].label
    assert isinstance(model_label, Content)
    assert model_label.plain == "  Fast Model  vendor/fast"
    assert any(span.start == len("  Fast Model ") and "dim" in str(span.style) for span in model_label.spans)
    assert model_items[0].marquee_start == len("  Fast Model  ")


@pytest.mark.asyncio
async def test_suggestion_list_empty_state_has_no_bullet() -> None:
    async with _SuggestionListApp().run_test() as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        suggestion_list.show("agents", [])
        assert suggestion_list._contents[0].plain == "No results"


def test_slash_suggestions_dismiss_for_non_command_paste() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.build_slash_commands()
    handler._suggestion_mode = "commands"

    handler.on_text_changed("// comment")

    assert handler.suggestion_mode is None


def test_build_slash_commands_uses_agents_and_no_legacy_entries() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)

    commands = handler.build_slash_commands()
    names = [c.name for c in commands]

    assert "agents" in names
    assert "runtime" in names
    assert "settings" in names
    assert "notifications" not in names
    assert "fork" in names
    assert "clear" in names
    assert "rename" in names
    assert "agent" not in names  # "agent" is now an alias, not a primary name
    assert "agent_config" not in names
    assert "mcp" not in names
    assert "skills" not in names
    language = next(command for command in commands if command.name == "language")
    assert language.synopsis == "/language [locale]"


def test_rename_command_opens_session_title_editor() -> None:
    """Bare /rename (or whitespace-only) is a second entry point to the
    border-click title dialog; /rename <title> applies directly."""
    screen = _SuggestionScreen()
    handler = _make_handler(screen)

    commands = handler.build_slash_commands()
    rename = next(command for command in commands if command.name == "rename")

    rename.action("")
    rename.action("   ")
    assert screen.title_editor_requests == 2
    assert screen.applied_titles == []

    rename.action("  Login bug fix  ")
    assert screen.applied_titles == ["Login bug fix"]
    assert screen.title_editor_requests == 2
    assert rename.man_page is not None


def test_runtime_config_commands_stay_available_but_settings_are_disabled_while_agent_runs() -> None:
    screen = _SuggestionScreen()
    screen._agent_running = True
    handler = _make_handler(screen)

    handler.build_slash_commands()

    assert "agents" not in handler._disabled_commands()
    assert "models" not in handler._disabled_commands()
    assert "settings" in handler._disabled_commands()


def test_copy_man_page_documents_role_filters() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)

    commands = handler.build_slash_commands()
    copy_command = next(command for command in commands if command.name == "copy")

    assert format_message(copy_command.description) == "Copy agent, user, or all turns to clipboard"
    assert copy_command.man_page is not None
    assert copy_command.synopsis is not None
    assert copy_command.options_help is not None
    assert isinstance(copy_command.man_page, tuple)
    copy_body = next(
        format_message(segment.message) for segment in copy_command.man_page if isinstance(segment, ManPageProseBlock)
    )
    examples = "\n".join(segment.text for segment in copy_command.man_page if isinstance(segment, ManPageVerbatimBlock))
    assert '"/copy agent N" is equivalent to "/copy N"' in copy_body
    assert '"/copy user N" copies the last N user turns' in copy_body
    assert "/copy agent all" in examples
    assert "/copy user all" in examples
    assert "/copy all" in examples
    assert "/copy agent [N|all]" in copy_command.synopsis
    assert [prefix for prefix, _reference in copy_command.options_help] == [
        "agent [N|all]  ",
        "user [N|all]   ",
        "all            ",
        "N              ",
    ]
    assert "Positive integer count" in format_message(copy_command.options_help[-1][1])


def test_rollback_man_page_documents_direct_relative_and_absolute_forms() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)

    commands = handler.build_slash_commands()
    rollback = next(command for command in commands if command.name == "rollback")

    assert format_message(rollback.description) == "Discard recent turns or return to a specific turn"
    assert rollback.subcommands is None
    assert rollback.synopsis is not None
    assert rollback.options_help is not None
    assert rollback.man_page is not None
    assert isinstance(rollback.man_page, MessageRef)
    man_page = " ".join(format_message(rollback.man_page).split())
    assert "/rollback N" in rollback.synopsis
    assert "/rollback to N" in rollback.synopsis
    assert [prefix for prefix, _reference in rollback.options_help] == ["N       ", "to N    "]
    assert "Positive number of most recent turns" in format_message(rollback.options_help[0][1])
    assert '"/rollback 1" discards the last turn' in man_page
    assert '"/rollback to 1"' in man_page
    assert "valid explicit count executes immediately" in man_page
    assert "restore eligible file changes by default" in man_page
    assert "discard conversation without restoring file changes" in man_page
    assert "non-dismissible loading" in man_page


def test_dispatch_agents_subcommands_route_to_expected_tabs() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.build_slash_commands()

    assert handler.dispatch_slash_command("/agents") is True
    assert handler.dispatch_slash_command("/agents basic") is True
    assert handler.dispatch_slash_command("/agents instructions") is True
    assert handler.dispatch_slash_command("/agents tools") is True
    assert handler.dispatch_slash_command("/agents sub-agents") is True
    assert handler.dispatch_slash_command("/agents subagents") is True
    assert handler.dispatch_slash_command("/agents mcp") is True
    assert handler.dispatch_slash_command("/agents memory") is True
    assert handler.dispatch_slash_command("/agents skill") is True
    assert handler.dispatch_slash_command("/agents skills") is True

    assert screen.opened == [
        "agent",
        "basic",
        "instructions",
        "tools",
        "sub-agents",
        "sub-agents",
        "mcp",
        "memory",
        "skills",
        "skills",
    ]


def test_agents_command_includes_memory_target_in_suggestions_and_man_page() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)

    commands = handler.build_slash_commands()
    agents_command = next(command for command in commands if command.name == "agents")

    assert agents_command.subcommands is not None
    assert ("memory", "Open Memory settings") in agents_command.subcommands()
    assert agents_command.man_page is not None
    assert isinstance(agents_command.man_page, MessageRef)
    assert "memory       - Configure memory files and folders" in format_message(agents_command.man_page)


def test_dispatch_agents_backward_compat_alias() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.build_slash_commands()

    assert handler.dispatch_slash_command("/agent") is True
    assert handler.dispatch_slash_command("/agent mcp") is True
    assert screen.opened == ["agent", "mcp"]


def test_dispatch_runtime_opens_details_for_agent_without_status_trail() -> None:
    from chrys.app.tui.screens.main.runtime_info import RegistryRuntimeInfoProvider

    screen = _SuggestionScreen()
    runtime_info = RegistryRuntimeInfoProvider(screen.services)
    handler = _make_handler(screen)
    handler.build_slash_commands()

    assert runtime_info.format_tool_info([], []) == ()
    assert handler.dispatch_slash_command("/runtime") is True
    assert handler.dispatch_slash_command("/details") is True
    assert screen.opened == ["runtime", "runtime"]


def test_dispatch_fork_requests_session_fork() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.build_slash_commands()

    assert handler.dispatch_slash_command("/fork") is True
    assert screen.fork_requests == 1


def test_dispatch_clear_requests_confirmed_session_clear() -> None:
    """/clear routes to the screen's confirm-then-delete flow, distinct from /new."""
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.build_slash_commands()

    assert handler.dispatch_slash_command("/clear") is True
    assert screen.clear_requests == 1


def test_dispatch_settings_opens_the_panel_on_the_requested_tab() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.build_slash_commands()

    assert handler.dispatch_slash_command("/settings") is True
    assert handler.dispatch_slash_command("/settings sessions") is True
    assert handler.dispatch_slash_command("/settings notifications") is True
    assert screen.opened == ["settings:general", "settings:sessions", "settings:notifications"]

    assert handler.dispatch_slash_command("/settings bogus") is True
    assert screen.opened[-1] == "settings:notifications"
    assert screen.notifications[-1] == "Unknown /settings tab: bogus"


def test_dispatch_settings_is_rejected_while_agent_runs() -> None:
    screen = _SuggestionScreen()
    screen._agent_running = True
    handler = _make_handler(screen)
    handler.build_slash_commands()

    assert handler.dispatch_slash_command("/settings") is True
    assert screen.opened == []
    assert screen.notifications == ["/settings is not available while agent is running"]


def test_removed_notifications_commands_are_not_dispatched() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.build_slash_commands()

    assert handler.dispatch_slash_command("/notifications") is False
    assert handler.dispatch_slash_command("/notify") is False
    assert screen.opened == []


def test_dispatch_unknown_agents_subcommand_notifies_warning() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.build_slash_commands()

    assert handler.dispatch_slash_command("/agents unknown") is True
    assert screen.notifications == ["Unknown /agents target: unknown"]


def test_legacy_mcp_and_skills_commands_are_not_dispatched() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.build_slash_commands()

    assert handler.dispatch_slash_command("/mcp") is False
    assert handler.dispatch_slash_command("/skills") is False


def test_typing_space_switches_to_subcommand_mode_for_theme_and_agents() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.build_slash_commands()

    # Simulate active slash suggestion mode while user types command + space.
    handler._suggestion_mode = "commands"
    handler.on_text_changed("/theme ")
    assert handler.suggestion_mode == "subcommands"
    assert screen.suggestion_list.last_mode == "subcommands"

    handler._suggestion_mode = "commands"
    handler.on_text_changed("/agents ")
    assert handler.suggestion_mode == "subcommands"
    assert screen.suggestion_list.last_mode == "subcommands"


def test_slash_suggestions_include_runtime_skills_in_separate_section() -> None:
    screen = _SuggestionScreen()
    screen._runtime_details = AgentRuntimeDetails(
        skill_details=[RuntimeSkillDetails(name="review", description="Review code and identify issues")]
    )
    handler = _make_handler(screen)
    handler.build_slash_commands()

    handler.on_slash_triggered()

    skill_items = [
        item for item in screen.suggestion_list.last_items if isinstance(item, SuggestionItem) and item.kind == "skill"
    ]
    assert len(skill_items) == 1
    assert skill_items[0].value == "review"
    assert skill_items[0].section == "Loaded Skills"
    assert skill_items[0].marquee_start == len("/review  ")


@pytest.mark.asyncio
async def test_runtime_skill_suggestion_metadata_is_rendered_as_literal_text() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.build_slash_commands()
    items = handler._runtime_skill_suggestion_items(
        [RuntimeSkillDetails(name="review[bad]", description="Review [broken markup")]
    )

    async with _SuggestionListApp().run_test() as pilot:
        suggestion_list = pilot.app.query_one(SuggestionList)
        suggestion_list.show("commands", items)
        await pilot.pause()

        prompt = suggestion_list._contents[1]

    assert str(prompt) == "/review[bad]  Review [broken markup"


def test_slash_filter_matches_runtime_skills() -> None:
    screen = _SuggestionScreen()
    screen._runtime_details = AgentRuntimeDetails(
        skill_details=[RuntimeSkillDetails(name="review", description="Review code")]
    )
    handler = _make_handler(screen)
    handler.build_slash_commands()
    handler._suggestion_mode = "commands"

    handler.on_text_changed("/r")

    assert any(
        isinstance(item, SuggestionItem) and item.kind == "skill" and item.value == "review"
        for item in screen.suggestion_list.last_items
    )


def test_shadowed_runtime_skill_is_visible_but_disabled() -> None:
    screen = _SuggestionScreen()
    screen._runtime_details = AgentRuntimeDetails(
        skill_details=[RuntimeSkillDetails(name="runtime", description="Runtime-like skill")]
    )
    handler = _make_handler(screen)
    handler.build_slash_commands()

    handler.on_slash_triggered()

    skill = next(
        item
        for item in screen.suggestion_list.last_items
        if isinstance(item, SuggestionItem) and item.kind == "skill" and item.value == "runtime"
    )
    assert skill.disabled is True
    assert skill.disabled_reason == "shadowed by /runtime"


def test_suggestion_chrome_renders_chinese_at_show_boundary_without_translating_payloads() -> None:
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo/[literal]"
    screen._agent_registry = _AgentRegistryStub(
        [_AgentProfileStub(name="Explore", display_name="Explorer", description="Profile [red]description")]
    )
    screen._runtime_details = AgentRuntimeDetails(
        skill_details=[RuntimeSkillDetails(name="runtime", description="Skill [blue]description")]
    )
    handler = _make_handler(screen, locale_controller=LocaleController(Settings(locale="zh-Hans")))
    commands = handler.build_slash_commands()

    handler.on_slash_triggered()

    assert screen.suggestion_list.last_title == "命令"
    command = commands[0]
    command_item = next(
        item
        for item in screen.suggestion_list.last_items
        if isinstance(item, SuggestionItem) and item.kind == "command" and item.value == command.name
    )
    assert command_item.section == "系统命令"
    assert handler._render_message(command.description) in command_item.label.plain
    skill = next(
        item
        for item in screen.suggestion_list.last_items
        if isinstance(item, SuggestionItem) and item.kind == "skill" and item.value == "runtime"
    )
    assert skill.section == "已加载技能"
    assert skill.disabled_reason == "被 /runtime 遮蔽"
    assert "Skill [blue]description" in skill.label.plain

    handler.on_agent_triggered()
    assert screen.suggestion_list.last_title == "智能体"
    agent = next(item for item in screen.suggestion_list.last_items if isinstance(item, SuggestionItem))
    assert "Profile [red]description" in agent.label.plain

    model_registry = ModelProfileRegistry()
    model_registry.register(ModelProfile(id="literal", name="Model [green]name", model_id="vendor/[model]"))
    screen.services.model_registry = model_registry
    handler.on_model_triggered()
    assert screen.suggestion_list.last_title == "模型"
    model = next(item for item in screen.suggestion_list.last_items if isinstance(item, SuggestionItem))
    assert model.label.plain == "  Model [green]name  vendor/[model]"

    handler._show_suggestions("files", [])
    assert screen.suggestion_list.last_title == "/repo/[literal] 下的文件"
    scan = _scan_result(
        "/repo/[literal]",
        [ProjectPathSuggestion(path="a.py", kind="file")],
        truncated=True,
        file_budget=1,
        suggestion_budget=1,
        source_truncations={"rg": True},
    )
    index = handler._build_file_index(scan)
    truncation = handler._file_suggestion_items([], index=index)[0]
    assert truncation.label == "此有限索引之外还有更多文件"
    assert truncation.disabled_reason == "已索引 1 个文件 / 1 行"


def test_shadowed_runtime_skill_dispatches_command() -> None:
    screen = _SuggestionScreen()
    screen._runtime_details = AgentRuntimeDetails(
        skill_details=[RuntimeSkillDetails(name="runtime", description="Runtime-like skill")]
    )
    handler = _make_handler(screen)
    handler.build_slash_commands()

    assert handler.dispatch_slash_command("/runtime") is True

    assert screen.opened == ["runtime"]
    assert screen.submitted == []


def test_selecting_runtime_skill_with_enter_submits_slash_reference() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.build_slash_commands()

    handler.on_suggestion_selected("commands", "review", execute=True, kind="skill")

    assert screen.submitted == ["/review"]


def test_stale_command_suggestion_selection_is_ignored() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.build_slash_commands()

    handler.on_suggestion_selected("commands", "missing", execute=True)

    assert screen.opened == []
    assert screen.submitted == []


def test_file_suggestions_rescan_when_cwd_changes_without_explicit_invalidation(monkeypatch) -> None:
    """The @ file cache is scoped by cwd, not just by prior scan existence."""
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    screen._chdir_current_cwd = "/repo-a"
    scans: list[str] = []

    def fake_getcwd() -> str:
        raise AssertionError("tracked workspace cwd should be used before process cwd")

    async def fake_scan_project_paths(root: str) -> ProjectPathScanResult:
        scans.append(root)
        path = "a.py" if root == "/repo-a" else "b.py"
        return _scan_result(root, [ProjectPathSuggestion(path=path, kind="file")])

    monkeypatch.setattr("chrys.foundation.platform.safe_getcwd", fake_getcwd)
    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.scan_project_paths", fake_scan_project_paths)

    handler._suggestion_mode = "files"
    asyncio.run(handler.show_file_suggestions_async())
    assert _suggestion_values(screen.suggestion_list.last_items) == ["a.py"]

    screen._chdir_current_cwd = "/repo-b"
    asyncio.run(handler.show_file_suggestions_async())

    assert scans == ["/repo-a", "/repo-b"]
    assert _suggestion_values(screen.suggestion_list.last_items) == ["b.py"]


@pytest.mark.asyncio
async def test_file_trigger_opens_loading_popup_before_cold_scan_is_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo"
    handler = _make_handler(screen)
    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_scan_project_paths(root: str) -> ProjectPathScanResult:
        assert root == "/repo"
        started.set()
        await release.wait()
        return _scan_result(root, [ProjectPathSuggestion(path="ready.py", kind="file")])

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.scan_project_paths", fake_scan_project_paths)

    handler.on_file_triggered()
    assert screen.suggestion_list.last_mode == "files"
    assert screen.suggestion_list.last_title == "Files under /repo"
    assert screen.suggestion_list.is_loading is True
    assert screen.suggestion_list.last_items == []

    task = asyncio.create_task(handler.show_file_suggestions_async())
    await started.wait()
    assert screen.suggestion_list.is_loading is True

    release.set()
    await task

    assert screen.suggestion_list.is_loading is False
    assert _suggestion_values(screen.suggestion_list.last_items) == ["ready.py"]


def test_file_suggestions_show_disabled_truncation_row(monkeypatch) -> None:
    """A bounded scan advertises truncation instead of silently omitting files."""
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo"
    handler = _make_handler(screen)

    async def fake_scan_project_paths(root: str) -> ProjectPathScanResult:
        return _scan_result(
            root,
            [ProjectPathSuggestion(path="a.py", kind="file")],
            truncated=True,
            file_budget=1,
            suggestion_budget=1,
            source_truncations={"rg": True},
        )

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.scan_project_paths", fake_scan_project_paths)

    handler._suggestion_mode = "files"
    asyncio.run(handler.show_file_suggestions_async())

    assert _suggestion_values(screen.suggestion_list.last_items) == ["a.py", "__chrys_file_results_truncated__"]
    truncation_row = screen.suggestion_list.last_items[-1]
    assert isinstance(truncation_row, SuggestionItem)
    assert truncation_row.disabled is True
    assert truncation_row.kind == "status"
    assert truncation_row.disabled_reason == "1 files / 1 rows indexed"


def test_file_suggestions_show_no_file_rows_for_empty_scan(monkeypatch) -> None:
    """A session from another machine may point at a workspace with no local files."""
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "Z:\\Fake\\MissingWorkspace"
    handler = _make_handler(screen)

    async def fake_scan_project_paths(root: str) -> ProjectPathScanResult:
        return _scan_result(root, [])

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.scan_project_paths", fake_scan_project_paths)

    handler._suggestion_mode = "files"
    asyncio.run(handler.show_file_suggestions_async())

    assert screen.suggestion_list.mode == "files"
    assert screen.suggestion_list.last_items == []
    assert handler.file_cache == []


def test_file_suggestion_enter_without_selection_does_not_submit_text() -> None:
    screen = _SuggestionScreen()
    screen.input_bar.value = "@missing"
    screen.suggestion_list.is_visible = True
    handler = _make_handler(screen)
    handler._suggestion_mode = "files"

    assert handler.on_suggestion_select(execute=True) is True

    assert screen.input_bar.value == "@missing"
    assert screen.submitted == []
    assert screen.suggestion_list.is_visible is True


def test_agent_suggestion_enter_without_selection_submits_unmatched_text() -> None:
    screen = _SuggestionScreen()
    screen._agent_registry = _AgentRegistryStub(
        [
            _AgentProfileStub(name="Code", description="Code agent"),
            _AgentProfileStub(name="QA", description="QA agent"),
        ]
    )
    screen.input_bar.value = "#123"
    screen.suggestion_list.is_visible = True
    handler = _make_handler(screen)
    handler._suggestion_mode = "agents"

    assert handler.on_suggestion_select(execute=True) is True

    assert screen.input_bar.value == ""
    assert screen.submitted == ["#123"]
    assert screen.suggestion_list.is_visible is False


def test_agent_suggestion_enter_on_active_profile_does_not_submit_text() -> None:
    screen = _SuggestionScreen()
    screen._agent_registry = _AgentRegistryStub(
        [
            _AgentProfileStub(name="Code", description="Code agent"),
            _AgentProfileStub(name="QA", description="QA agent"),
        ]
    )
    screen.input_bar.value = "#Code"
    screen.suggestion_list.is_visible = True
    handler = _make_handler(screen)
    handler._suggestion_mode = "agents"

    assert handler.on_suggestion_select(execute=True) is True

    assert screen.input_bar.value == "#Code"
    assert screen.submitted == []
    assert screen.suggestion_list.is_visible is True


def test_model_trigger_filters_selectable_profiles_and_switches_by_profile_id() -> None:
    screen = _SuggestionScreen()
    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="current-id", name="Current", model_id="vendor/current"))
    registry.register(ModelProfile(id="fast-id", name="Fast Model", model_id="vendor/fast"))
    registry.register(ModelProfile(id="incomplete-id", name="Incomplete"))
    screen.services.model_registry = registry
    screen.services.active_model_profile_id = "current-id"
    handler = _make_handler(screen)

    handler.on_model_triggered()

    assert screen.suggestion_list.last_mode == "models"
    assert _suggestion_values(screen.suggestion_list.last_items) == ["current-id", "fast-id"]
    items, disabled = handler._get_model_items()
    assert disabled == {"current-id"}
    assert items[0].label.plain == "◦ Current  vendor/current"

    handler.on_text_changed("$fast")
    assert _suggestion_values(screen.suggestion_list.last_items) == ["fast-id"]

    handler.on_suggestion_selected("models", "fast-id", execute=True)
    assert screen.picked_models == ["fast-id"]
    assert screen.input_bar.value == ""
    assert screen.suggestion_list.is_visible is False


def test_pickers_stay_shut_while_the_agent_loads() -> None:
    """A switch that lands mid-load is dropped after the draft is already gone.

    The status-bar selectors block loading as well as running; the inline
    triggers have to agree, or the popup opens onto a choice that cannot be
    committed.
    """
    screen = _SuggestionScreen()
    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="model-id", name="Model", model_id="vendor/model"))
    screen.services.model_registry = registry
    screen.state.run.agent_loading = True
    handler = _make_handler(screen)

    handler.on_model_triggered()
    assert screen.suggestion_list.is_visible is False

    handler.on_agent_triggered()
    assert screen.suggestion_list.is_visible is False


@pytest.mark.parametrize("selection_source", ["agent", "override", "inherited"])
def test_model_trigger_does_not_bypass_locked_runtime_selection(selection_source: str) -> None:
    screen = _SuggestionScreen()
    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="model-id", name="Model", model_id="vendor/model"))
    screen.services.model_registry = registry
    screen.state.runtime.details_confirmed = True
    screen.state.runtime.details.model.profile_id = "model-id"
    screen.state.runtime.details.model.selection_source = selection_source  # type: ignore[assignment]
    handler = _make_handler(screen)

    handler.on_model_triggered()

    assert screen.suggestion_list.is_visible is False


def test_model_suggestion_enter_without_selection_submits_unmatched_text() -> None:
    screen = _SuggestionScreen()
    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="fast-id", name="Fast Model", model_id="vendor/fast"))
    screen.services.model_registry = registry
    screen.input_bar.value = "$not-a-model"
    screen.suggestion_list.is_visible = True
    handler = _make_handler(screen)
    handler._suggestion_mode = "models"

    assert handler.on_suggestion_select(execute=True) is True

    assert screen.input_bar.value == ""
    assert screen.submitted == ["$not-a-model"]
    assert screen.suggestion_list.is_visible is False


def test_model_suggestion_enter_on_active_profile_does_not_submit_text() -> None:
    screen = _SuggestionScreen()
    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="current-id", name="Current", model_id="vendor/current"))
    screen.services.model_registry = registry
    screen.services.active_model_profile_id = "current-id"
    screen.input_bar.value = "$Current"
    screen.suggestion_list.is_visible = True
    handler = _make_handler(screen)
    handler._suggestion_mode = "models"

    assert handler.on_suggestion_select(execute=True) is True

    assert screen.input_bar.value == "$Current"
    assert screen.submitted == []
    assert screen.suggestion_list.is_visible is True


def test_command_suggestion_enter_without_selection_keeps_submit_fallback() -> None:
    screen = _SuggestionScreen()
    screen.input_bar.value = "/mcp"
    handler = _make_handler(screen)
    handler._suggestion_mode = "commands"
    handler.build_slash_commands()

    assert handler.on_suggestion_select(execute=True) is True

    assert screen.input_bar.value == ""
    assert screen.submitted == ["/mcp"]


def test_file_suggestions_discard_scan_when_cwd_changes_mid_scan(monkeypatch) -> None:
    """A scan result is cached only if its root is still current when it returns."""
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo-a"
    handler = _make_handler(screen)
    scans: list[str] = []

    def fake_getcwd() -> str:
        raise AssertionError("tracked workspace cwd should be used before process cwd")

    async def fake_scan_project_paths(root: str) -> ProjectPathScanResult:
        scans.append(root)
        if root == "/repo-a":
            screen._chdir_current_cwd = "/repo-b"
            return _scan_result(root, [ProjectPathSuggestion(path="stale.py", kind="file")])
        return _scan_result(root, [ProjectPathSuggestion(path="fresh.py", kind="file")])

    monkeypatch.setattr("chrys.foundation.platform.safe_getcwd", fake_getcwd)
    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.scan_project_paths", fake_scan_project_paths)

    handler._suggestion_mode = "files"
    asyncio.run(handler.show_file_suggestions_async())

    assert scans == ["/repo-a", "/repo-b"]
    assert _suggestion_values(screen.suggestion_list.last_items) == ["fresh.py"]


@pytest.mark.asyncio
async def test_file_suggestions_filter_directory_query() -> None:
    """A directory-shaped query should keep the matching directory entry."""
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.file_cache = [
        ProjectPathSuggestion(path="src/chrys/app/tui/", kind="directory"),
        ProjectPathSuggestion(path="src/chrys/app/tui/widgets/chrome/file_scanner.py", kind="file"),
        ProjectPathSuggestion(path="src/chrys/orchestration/engine/engine.py", kind="file"),
    ]
    handler._suggestion_mode = "files"

    handler.on_text_changed("@src/chrys/app/tui/")
    await _wait_for_file_query(handler)

    assert _suggestion_values(screen.suggestion_list.last_items)[0] == "src/chrys/app/tui/"
    assert isinstance(screen.suggestion_list.last_items[0], SuggestionItem)
    assert screen.suggestion_list.last_items[0].kind == "directory"


@pytest.mark.asyncio
async def test_file_suggestions_filter_full_relative_path_from_injected_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct file-cache injection remains a cheap test path for file suggestions."""
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    injected = [
        ProjectPathSuggestion(path="src/chrys/app/tui/widgets/chrome/", kind="directory"),
        ProjectPathSuggestion(path="src/chrys/app/tui/widgets/chrome/file_scanner.py", kind="file"),
        ProjectPathSuggestion(path="src/chrys/orchestration/engine/engine.py", kind="file"),
    ]
    handler.file_cache = injected
    handler._suggestion_mode = "files"

    def fail_fuzzy_filter(*_args, **_kwargs) -> list[str]:
        raise AssertionError("file suggestions should query ProjectPathIndex")

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.fuzzy_filter", fail_fuzzy_filter)

    handler.on_text_changed("@chrome")
    await _wait_for_file_query(handler)

    assert handler.file_cache is injected
    assert _suggestion_values(screen.suggestion_list.last_items) == [
        "src/chrys/app/tui/widgets/chrome/",
        "src/chrys/app/tui/widgets/chrome/file_scanner.py",
    ]
    assert isinstance(screen.suggestion_list.last_items[0], SuggestionItem)
    assert screen.suggestion_list.last_items[0].kind == "directory"


@pytest.mark.asyncio
async def test_file_cache_assignment_invalidates_file_index() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)
    handler.file_cache = [ProjectPathSuggestion(path="src/chrome/file.py", kind="file")]
    handler._suggestion_mode = "files"

    handler.on_text_changed("@chrome")
    await _wait_for_file_query(handler)
    assert handler._file_index is not None

    handler.file_cache = None

    assert handler.file_cache is None
    assert handler._file_index is None


@pytest.mark.asyncio
async def test_dismissed_suggestions_do_not_receive_stale_file_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(suggestions_module, "_FILE_QUERY_DEBOUNCE_SECONDS", 0)
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo"
    handler = _make_handler(screen)
    handler.file_cache = [ProjectPathSuggestion(path="src/chrome/file.py", kind="file")]
    handler._suggestion_mode = "files"

    handler.on_text_changed("@chrome")
    handler.dismiss_suggestions()
    await _wait_for_file_query(handler)

    assert screen.suggestion_list.is_visible is False
    assert screen.suggestion_list.last_items == []


@pytest.mark.asyncio
async def test_switching_modes_invalidates_pending_file_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(suggestions_module, "_FILE_QUERY_DEBOUNCE_SECONDS", 0)
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo"
    handler = _make_handler(screen)
    handler.file_cache = [ProjectPathSuggestion(path="src/chrome/file.py", kind="file")]
    handler._suggestion_mode = "files"

    handler.on_text_changed("@chrome")
    handler._show_suggestions("commands", [])
    await _wait_for_file_query(handler)

    assert screen.suggestion_list.mode == "commands"
    assert _suggestion_values(screen.suggestion_list.last_items) == []


@pytest.mark.asyncio
async def test_rapid_file_typing_keeps_one_active_query_and_runs_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(suggestions_module, "_FILE_QUERY_DEBOUNCE_SECONDS", 0)
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo"
    handler = _make_handler(screen)
    handler._suggestion_mode = "files"
    handler.file_cache = [ProjectPathSuggestion(path="placeholder.py", kind="file")]

    lock = threading.Lock()
    active = 0
    max_active = 0
    calls: list[str] = []

    class SlowIndex:
        truncated = False
        file_count = 0
        suggestion_count = 0

        def query(self, query: str, *, limit: int) -> list[ProjectPathSuggestion]:
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                calls.append(query)
                time.sleep(0.03)
                return [ProjectPathSuggestion(path=f"{query}.py", kind="file")]
            finally:
                with lock:
                    active -= 1

    handler._file_index = SlowIndex()  # type: ignore[assignment]
    handler._file_index_root = "/repo"

    handler.on_text_changed("@alpha")
    await asyncio.sleep(0.01)
    handler.on_text_changed("@beta")
    handler.on_text_changed("@gamma")
    await _wait_for_file_query(handler)

    assert max_active == 1
    assert calls[-1] == "gamma"
    assert "beta" not in calls
    assert _suggestion_values(screen.suggestion_list.last_items) == ["gamma.py"]


@pytest.mark.asyncio
async def test_typing_file_query_before_cold_scan_completion_replays_latest_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(suggestions_module, "_FILE_QUERY_DEBOUNCE_SECONDS", 0)
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo"
    handler = _make_handler(screen)
    scan_started = asyncio.Event()
    finish_scan = asyncio.Event()

    async def fake_scan_project_paths(root: str) -> ProjectPathScanResult:
        assert root == "/repo"
        scan_started.set()
        await finish_scan.wait()
        return _scan_result(
            root,
            [
                ProjectPathSuggestion(path="alpha.py", kind="file"),
                ProjectPathSuggestion(path="src/foo.py", kind="file"),
            ],
        )

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.scan_project_paths", fake_scan_project_paths)

    handler._suggestion_mode = "files"
    warmup = asyncio.create_task(handler.show_file_suggestions_async())
    await scan_started.wait()
    handler.on_text_changed("@foo")
    finish_scan.set()
    await warmup
    await _wait_for_file_query(handler)

    assert _suggestion_values(screen.suggestion_list.last_items) == ["src/foo.py"]


@pytest.mark.asyncio
async def test_cache_invalidation_during_index_build_discards_stale_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo"
    handler = _make_handler(screen)
    scans = 0
    build_started = threading.Event()
    release_build = threading.Event()
    build_calls = 0
    original_build = handler._build_file_index

    async def fake_scan_project_paths(root: str) -> ProjectPathScanResult:
        nonlocal scans
        scans += 1
        path = "stale.py" if scans == 1 else "fresh.py"
        return _scan_result(root, [ProjectPathSuggestion(path=path, kind="file")])

    def fake_build_file_index(scan: ProjectPathScanResult):
        nonlocal build_calls
        build_calls += 1
        if build_calls == 1:
            build_started.set()
            release_build.wait(timeout=1)
        return original_build(scan)

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.scan_project_paths", fake_scan_project_paths)
    monkeypatch.setattr(handler, "_build_file_index", fake_build_file_index)

    handler._suggestion_mode = "files"
    warmup = asyncio.create_task(handler.show_file_suggestions_async())
    await asyncio.to_thread(build_started.wait, 1)
    handler.file_cache = None
    release_build.set()
    await warmup

    assert scans == 2
    assert handler.file_cache == [ProjectPathSuggestion(path="fresh.py", kind="file")]
    assert _suggestion_values(screen.suggestion_list.last_items) == ["fresh.py"]


@pytest.mark.asyncio
async def test_dismiss_during_cold_scan_does_not_reopen_file_suggestions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo"
    handler = _make_handler(screen)
    scan_started = asyncio.Event()
    finish_scan = asyncio.Event()

    async def fake_scan_project_paths(_root: str) -> ProjectPathScanResult:
        scan_started.set()
        await finish_scan.wait()
        return _scan_result(_root, [ProjectPathSuggestion(path="stale.py", kind="file")])

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.scan_project_paths", fake_scan_project_paths)

    handler._suggestion_mode = "files"
    warmup = asyncio.create_task(handler.show_file_suggestions_async())
    await scan_started.wait()
    handler.dismiss_suggestions()
    finish_scan.set()
    await warmup

    assert handler.file_cache is None
    assert screen.suggestion_list.is_visible is False
    assert screen.suggestion_list.last_items == []


@pytest.mark.asyncio
async def test_dismiss_before_warmup_starts_does_not_restore_file_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo"
    handler = _make_handler(screen)
    scans = 0

    async def fake_scan_project_paths(_root: str) -> ProjectPathScanResult:
        nonlocal scans
        scans += 1
        return _scan_result(_root, [ProjectPathSuggestion(path="stale.py", kind="file")])

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.scan_project_paths", fake_scan_project_paths)

    handler._suggestion_mode = "files"
    warmup = asyncio.create_task(handler.show_file_suggestions_async())
    handler.dismiss_suggestions()
    await warmup

    assert scans == 0
    assert handler.suggestion_mode is None
    assert screen.suggestion_list.is_visible is False


@pytest.mark.asyncio
async def test_mode_switch_before_warmup_worker_scans_skips_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo"
    handler = _make_handler(screen)
    scans = 0
    deferred_workers: list[_DeferredWorker] = []

    def run_worker(work, **_kwargs) -> _DeferredWorker:
        worker = _DeferredWorker(work)
        deferred_workers.append(worker)
        return worker

    async def fake_scan_project_paths(_root: str) -> ProjectPathScanResult:
        nonlocal scans
        scans += 1
        return _scan_result(_root, [ProjectPathSuggestion(path="stale.py", kind="file")])

    monkeypatch.setattr(screen, "run_worker", run_worker)
    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.scan_project_paths", fake_scan_project_paths)

    handler._suggestion_mode = "files"
    warmup = asyncio.create_task(handler.show_file_suggestions_async())
    await asyncio.sleep(0)
    assert deferred_workers

    handler._show_suggestions("commands", [])
    await warmup

    assert scans == 0
    assert handler.file_cache is None
    assert screen.suggestion_list.mode == "commands"


@pytest.mark.asyncio
async def test_mode_switch_during_index_build_does_not_reopen_file_suggestions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo"
    handler = _make_handler(screen)
    build_started = threading.Event()
    release_build = threading.Event()
    original_build = handler._build_file_index

    async def fake_scan_project_paths(_root: str) -> ProjectPathScanResult:
        return _scan_result(_root, [ProjectPathSuggestion(path="stale.py", kind="file")])

    def fake_build_file_index(scan: ProjectPathScanResult):
        build_started.set()
        release_build.wait(timeout=1)
        return original_build(scan)

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.scan_project_paths", fake_scan_project_paths)
    monkeypatch.setattr(handler, "_build_file_index", fake_build_file_index)

    handler._suggestion_mode = "files"
    warmup = asyncio.create_task(handler.show_file_suggestions_async())
    await asyncio.to_thread(build_started.wait, 1)
    handler._show_suggestions("commands", [])
    release_build.set()
    await warmup

    assert handler.file_cache is None
    assert screen.suggestion_list.mode == "commands"
    assert screen.suggestion_list.last_items == []


@pytest.mark.asyncio
async def test_detach_during_warmup_does_not_request_followup_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo"
    handler = _make_handler(screen)
    scan_started = asyncio.Event()
    finish_scan = asyncio.Event()
    scans = 0

    async def fake_scan_project_paths(_root: str) -> ProjectPathScanResult:
        nonlocal scans
        scans += 1
        scan_started.set()
        await finish_scan.wait()
        return _scan_result(_root, [ProjectPathSuggestion(path="stale.py", kind="file")])

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.scan_project_paths", fake_scan_project_paths)

    handler._suggestion_mode = "files"
    warmup = asyncio.create_task(handler.show_file_suggestions_async())
    await scan_started.wait()
    screen.is_attached = False
    finish_scan.set()
    await warmup

    assert scans == 1
    assert handler._file_warmup_requested_root is None
    assert handler.file_cache is None


@pytest.mark.asyncio
async def test_detached_screen_does_not_receive_file_query_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(suggestions_module, "_FILE_QUERY_DEBOUNCE_SECONDS", 0)
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo"
    handler = _make_handler(screen)
    handler.file_cache = [ProjectPathSuggestion(path="src/chrome/file.py", kind="file")]
    handler._suggestion_mode = "files"

    handler.on_text_changed("@chrome")
    screen.is_attached = False
    await _wait_for_file_query(handler)

    assert screen.suggestion_list.last_items == []


@pytest.mark.asyncio
async def test_index_warmup_build_does_not_starve_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    screen = _SuggestionScreen()
    screen._chdir_current_cwd = "/repo"
    handler = _make_handler(screen)
    original_build = handler._build_file_index

    async def fake_scan_project_paths(_root: str) -> ProjectPathScanResult:
        return _scan_result(
            _root,
            [ProjectPathSuggestion(path=f"src/file_{index:04d}.py", kind="file") for index in range(1000)],
        )

    loop_kept_ticking = threading.Event()

    def slow_build_file_index(scan: ProjectPathScanResult):
        # Block the worker until the event loop has demonstrably kept
        # ticking underneath the build.  If the build ran on the loop
        # thread instead, this could never be set — the timeout path
        # finishes the warmup with the tick count still at ~0 and the
        # assertion fails.  A handshake instead of a fixed sleep: tick
        # counting against wall-clock flakes on Windows' coarse timer.
        loop_kept_ticking.wait(timeout=5.0)
        return original_build(scan)

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.file_scanner.scan_project_paths", fake_scan_project_paths)
    monkeypatch.setattr(handler, "_build_file_index", slow_build_file_index)

    handler._suggestion_mode = "files"
    warmup = asyncio.create_task(handler.show_file_suggestions_async())
    ticks = 0
    while not warmup.done():
        ticks += 1
        if ticks > 3:
            loop_kept_ticking.set()
        await asyncio.sleep(0.005)
    await warmup

    assert ticks > 3
    assert _suggestion_values(screen.suggestion_list.last_items)[:1] == ["src/file_0000.py"]


def test_selecting_directory_suggestion_inserts_trailing_slash() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)

    handler.on_suggestion_selected("files", "src/chrys", execute=False, kind="directory")

    assert screen.input_bar.replacements == [("@", "@src/chrys/ ")]


def test_selecting_file_status_suggestion_is_ignored() -> None:
    screen = _SuggestionScreen()
    handler = _make_handler(screen)

    handler.on_suggestion_selected("files", "__chrys_file_results_truncated__", execute=False, kind="status")

    assert screen.input_bar.replacements == []
    assert screen.suggestion_list.is_visible is False
