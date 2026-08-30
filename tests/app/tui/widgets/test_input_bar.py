# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the TUI input bar."""

from __future__ import annotations

import asyncio

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.widgets import Button, TextArea

from chrys.app.tui import i18n as tui_i18n
from chrys.app.tui.i18n import LocaleController, LocaleSwitchStatus
from chrys.app.tui.screens.main.screen import MainScreen
from chrys.app.tui.screens.main.view_adapter import MainScreenViewAdapter
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.chrome import input_bar as input_bar_module
from chrys.app.tui.widgets.chrome.input_bar import (
    INPUT_RETRY,
    INPUT_SEND,
    InputBar,
    _ChatTextArea,
    _is_file_trigger_boundary,
)
from chrys.app.tui.widgets.chrome.suggestion_list import SuggestionList
from chrys.app.tui.widgets.loading import ChrysLoadingIndicator
from chrys.app.tui.widgets.sidebar.context import ContextPanel, ContextUsageState
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.i18n import MessageRef


class _InputBarApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.suggestion_directions: list[str] = []
        self.file_trigger_count = 0
        self.model_trigger_count = 0
        self.shell_request_count = 0
        self.editor_request_count = 0
        self.editor_event_order: list[str] = []

    def compose(self) -> ComposeResult:
        yield InputBar()

    def on_input_bar_suggestion_navigate(self, event: InputBar.SuggestionNavigate) -> None:
        self.suggestion_directions.append(event.direction)

    def on_input_bar_file_triggered(self, _event: InputBar.FileTriggered) -> None:
        self.file_trigger_count += 1

    def on_input_bar_model_triggered(self, _event: InputBar.ModelTriggered) -> None:
        self.model_trigger_count += 1

    def on_input_bar_shell_mode_requested(self, _event: InputBar.ShellModeRequested) -> None:
        self.shell_request_count += 1

    def on_input_bar_suggestion_dismiss(self, _event: InputBar.SuggestionDismiss) -> None:
        self.editor_event_order.append("dismiss")

    def on_input_bar_editor_requested(self, _event: InputBar.EditorRequested) -> None:
        self.editor_request_count += 1
        self.editor_event_order.append("editor")


class _LocalizedInputBarApp(App[None]):
    def __init__(self, controller: LocaleController) -> None:
        super().__init__()
        self._controller = controller

    def compose(self) -> ComposeResult:
        yield InputBar(locale_controller=self._controller)


class _MainScreenInputBindingApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.main_screen = MainScreen(EventBus(), engine_provider=None)

    def compose(self) -> ComposeResult:
        yield from ()

    async def on_mount(self) -> None:
        await self.push_screen(self.main_screen)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_loading", [False, True], ids=["idle", "loading"])
async def test_ctrl_r_opens_prompt_history_and_enter_restores_original_prompt(
    monkeypatch: pytest.MonkeyPatch,
    agent_loading: bool,
) -> None:
    markup_shaped_prompt = "failure [type=missing, input_value={}, input_type=dict])"

    async def load_prompt_history(_input_bar: InputBar, *, max_entries: int) -> list[str]:
        assert max_entries == 100
        return ["older", "middle\nwith newline", markup_shaped_prompt]

    monkeypatch.setattr(InputBar, "load_prompt_history", load_prompt_history)
    app = _MainScreenInputBindingApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.main_screen.query_one(InputBar)
        input_bar.focus_input()
        app.main_screen._set_agent_loading(agent_loading)
        await pilot.pause()

        await pilot.press("ctrl+r")
        await pilot.pause()

        suggestions = app.main_screen.query_one(SuggestionList)
        assert suggestions.mode == "history"
        assert suggestions._values == [markup_shaped_prompt, "middle\nwith newline", "older"]
        assert [content.plain for content in suggestions._contents] == [
            markup_shaped_prompt,
            "middle ↵ with newline",
            "older",
        ]
        assert app.focused is input_bar.query_one("#chat-input", _ChatTextArea)

        await pilot.press("enter")
        await pilot.pause()

        assert input_bar.value == markup_shaped_prompt
        assert suggestions.mode == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["shell", "session-json"])
async def test_prompt_history_loading_is_dismissed_before_main_view_transition(
    monkeypatch: pytest.MonkeyPatch,
    transition: str,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def load_prompt_history(_input_bar: InputBar, *, max_entries: int) -> list[str]:
        assert max_entries == 100
        started.set()
        await release.wait()
        return ["stale prompt"]

    monkeypatch.setattr(InputBar, "load_prompt_history", load_prompt_history)
    app = _MainScreenInputBindingApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.main_screen.query_one(InputBar)
        input_bar.focus_input()
        await pilot.press("ctrl+r")
        await started.wait()

        suggestions = app.main_screen.query_one(SuggestionList)
        loading = suggestions.query_one(ChrysLoadingIndicator)
        assert suggestions.mode == "history"
        assert suggestions.is_loading is True
        assert loading.display is True

        try:
            await pilot.press("!" if transition == "shell" else "f12")
            await pilot.pause()

            assert suggestions.mode == ""
            assert suggestions.is_visible is False
            assert suggestions.is_loading is False
            if transition == "shell":
                assert app.main_screen._shell_mode is True
            else:
                assert app.main_screen._dashboard_visible() is True
        finally:
            release.set()

        await pilot.pause()
        assert suggestions.mode == ""
        assert suggestions.is_visible is False


@pytest.mark.asyncio
async def test_prompt_history_selection_resets_up_down_browsing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def load_prompt_history(_input_bar: InputBar, *, max_entries: int) -> list[str]:
        assert max_entries == 100
        return ["selected prompt"]

    monkeypatch.setattr(InputBar, "load_prompt_history", load_prompt_history)
    app = _MainScreenInputBindingApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.main_screen.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        input_bar.add_to_history("instance history")
        input_bar.value = "earlier draft"
        input_bar.focus_input()

        await pilot.press("up")
        assert input_bar.value == "instance history"
        assert text_area._history.index == 0

        await pilot.press("ctrl+r")
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()

        assert input_bar.value == "selected prompt"
        assert text_area._history.index == -1
        assert text_area._history_browsing is False

        await pilot.press("down")
        assert input_bar.value == "selected prompt"


@pytest.mark.asyncio
async def test_main_screen_binds_one_way_child_flags() -> None:
    app = _MainScreenInputBindingApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        chat_panel = app.main_screen.query_one(ChatPanel)
        context_panel = app.main_screen.query_one(ContextPanel)
        input_bar = app.main_screen.query_one(InputBar)

        app.main_screen._set_agent_running(True)
        app.main_screen._set_agent_loading(True)
        app.main_screen._set_has_messages(True)
        app.main_screen.chat_profile_name = "Code"
        app.main_screen.chat_session_id = "12345678-1234-1234-1234-123456789abc"
        app.main_screen.chat_session_title = "Fix login bug"
        app.main_screen.chat_workspace_cwd = "/repo/chrys"
        app.main_screen.context_usage_state = ContextUsageState.with_window(
            used_tokens=57_700,
            max_context_tokens=200_000,
            total_session_tokens=236_100,
        )
        await pilot.pause()

        assert chat_panel.agent_running is True
        assert chat_panel._profile_name == "Code"
        assert chat_panel.session_id == "12345678-1234-1234-1234-123456789abc"
        assert str(chat_panel.border_subtitle) == "/repo/chrys"
        assert str(chat_panel.border_title) == "Session: 123456781234 \u00b7 Fix login bug"
        assert context_panel._current_used == 57_700
        assert context_panel._current_max == 200_000
        assert context_panel._total_session_tokens == 236_100
        assert input_bar.agent_running is True
        assert input_bar.agent_loading is True
        assert input_bar.has_messages is True

        app.main_screen._set_agent_running(False)
        app.main_screen._set_agent_loading(False)
        app.main_screen._set_has_messages(False)
        await pilot.pause()

        assert chat_panel.agent_running is False
        assert input_bar.agent_running is False
        assert input_bar.agent_loading is False
        assert input_bar.has_messages is False


@pytest.mark.asyncio
async def test_main_screen_reactive_sources_apply_existing_side_effects() -> None:
    app = _MainScreenInputBindingApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.main_screen.query_one(InputBar)
        app.main_screen._suggestions.file_cache = {"stale": object()}
        app.main_screen._live_call_paths["call"] = "src/app.py"
        app.main_screen._live_file_mutations["src/app.py"] = object()

        app.main_screen.agent_running_state = True
        await pilot.pause()

        assert app.main_screen._agent_running is True
        assert app.main_screen._live_call_paths == {}
        assert app.main_screen._live_file_mutations == {}

        input_bar.lock_with_text()
        app.main_screen.agent_running_state = False
        app.main_screen.agent_loading_state = False
        app.main_screen.has_messages_state = True
        await pilot.pause()

        assert app.main_screen._agent_running is False
        assert app.main_screen._agent_loading is False
        assert app.main_screen._has_messages is True
        assert app.main_screen._suggestions.file_cache is None
        assert input_bar.locked is False


@pytest.mark.parametrize(
    "text",
    ["@", "some text @", "some text\t@", "some text \uff20", "你好@", "你好\uff20", "こんにちは@", "한글@"],
)
def test_file_trigger_boundary_accepts_start_whitespace_and_cjk(text: str) -> None:
    assert _is_file_trigger_boundary(text, len(text) - 1) is True


@pytest.mark.parametrize("text", ["some text@", "user@web.com", "user\uff20web.com", "path/to@file", "abc_@"])
def test_file_trigger_boundary_rejects_ascii_word_boundaries(text: str) -> None:
    at_pos = next(i for i, char in enumerate(text) if char in ("@", "\uff20"))
    assert _is_file_trigger_boundary(text, at_pos) is False


@pytest.mark.asyncio
async def test_input_bar_requests_shell_mode_without_owning_state() -> None:
    app = _InputBarApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        input_bar.query_one("#chat-input", TextArea).focus()

        await pilot.press("!")
        await pilot.pause()

        assert app.shell_request_count == 1
        assert input_bar.shell_mode is False


@pytest.mark.parametrize(
    ("prefix", "expected_count"),
    [
        ("", 1),
        ("some text ", 1),
        ("你好", 1),
        ("user", 0),
    ],
)
@pytest.mark.asyncio
async def test_input_bar_posts_file_trigger_only_at_file_boundaries(prefix: str, expected_count: int) -> None:
    app = _InputBarApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        input_bar.value = prefix
        input_bar.query_one("#chat-input", TextArea).focus()
        await pilot.press("@")
        await pilot.pause()

    assert app.file_trigger_count == expected_count


@pytest.mark.asyncio
async def test_input_bar_dollar_character_triggers_model_suggestions() -> None:
    app = _InputBarApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        input_bar.query_one("#chat-input", TextArea).focus()

        await pilot.press("$")
        await pilot.pause()

        assert app.model_trigger_count == 1
        assert app.editor_request_count == 0
        assert input_bar.value == "$"


@pytest.mark.parametrize("draft", ["", "existing draft"])
@pytest.mark.asyncio
async def test_input_bar_ctrl_o_opens_editor_without_changing_draft(draft: str) -> None:
    app = _InputBarApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        input_bar.value = draft
        input_bar.query_one("#chat-input", TextArea).focus()

        await pilot.press("ctrl+o")
        await pilot.pause()

        assert app.editor_request_count == 1
        assert input_bar.value == draft


@pytest.mark.asyncio
async def test_input_bar_prompt_button_opens_editor_without_changing_draft() -> None:
    app = _InputBarApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        input_bar.value = "existing draft"
        editor_button = input_bar.query_one("#editor-btn", Button)
        text_area = input_bar.query_one("#chat-input", TextArea)

        assert editor_button.label.plain == ">"
        assert editor_button.disabled is False

        await pilot.click("#editor-btn")
        await pilot.pause()

        assert app.editor_request_count == 1
        assert input_bar.value == "existing draft"
        assert text_area.has_focus


@pytest.mark.asyncio
async def test_input_bar_ctrl_e_keeps_textual_line_end_behavior() -> None:
    app = _InputBarApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", TextArea)
        input_bar.value = "draft"
        text_area.move_cursor((0, 1))
        text_area.focus()

        await pilot.press("ctrl+e")
        await pilot.pause()

        assert text_area.cursor_location == (0, 5)
        assert app.editor_request_count == 0


@pytest.mark.asyncio
async def test_input_bar_ctrl_o_remains_available_while_agent_running() -> None:
    app = _InputBarApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", TextArea)
        input_bar.agent_running = True
        text_area.focus()
        await pilot.pause()

        await pilot.press("ctrl+o")
        await pilot.pause()

        assert app.editor_request_count == 1
        assert text_area.placeholder.plain == "Inject a message  /  Commands  @  Files"


@pytest.mark.parametrize("guard", ["locked", "shell_mode"])
@pytest.mark.asyncio
async def test_input_bar_guards_editor_forwarder(guard: str) -> None:
    app = _InputBarApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", _ChatTextArea)
        if guard == "locked":
            input_bar.locked = True
        else:
            input_bar.shell_mode = True
        await pilot.pause()

        assert input_bar.query_one("#editor-btn", Button).disabled is True

        text_area.post_message(_ChatTextArea.EditorRequested())
        await pilot.pause()

        assert app.editor_request_count == 0


@pytest.mark.asyncio
async def test_input_bar_dismisses_active_suggestions_before_editor_request() -> None:
    app = _InputBarApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        input_bar.set_suggestions_active(True, mode="files")
        input_bar.query_one("#chat-input", TextArea).focus()

        await pilot.press("ctrl+o")
        await pilot.pause()

        assert app.editor_event_order == ["dismiss", "editor"]


@pytest.mark.asyncio
async def test_input_bar_editor_placeholders_are_exact_and_ordered() -> None:
    app = _InputBarApp()

    async with app.run_test(size=(160, 15)) as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", TextArea)

        assert text_area.placeholder.plain == (
            "Type a message (Ctrl+O for Editor)   #  Agents  $  Models  /  Commands  @  Files  !  Shell"
        )
        newline_hint = "Ctrl+J for newline"
        rendered_line = text_area.render_line(0)
        assert rendered_line.text.rstrip().endswith(newline_hint)
        assert rendered_line.text.endswith(" ")
        assert rendered_line.cell_length == text_area.content_size.width

        input_bar.value = "draft"
        await pilot.pause()
        assert newline_hint not in text_area.render_line(0).text
        input_bar.value = ""
        await pilot.pause()

        input_bar.agent_running = True
        await pilot.pause()
        assert text_area.placeholder.plain == "Inject a message  /  Commands  @  Files"


@pytest.mark.asyncio
async def test_input_bar_relocalizes_semantic_state_and_cell_width_in_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    input_bar: InputBar | None = None

    async with _LocalizedInputBarApp(controller).run_test(size=(120, 15)) as pilot:
        input_bar = pilot.app.query_one(InputBar)
        send = input_bar.query_one("#send-btn", Button)
        new = input_bar.query_one("#new-btn", Button)
        text_area = input_bar.query_one("#chat-input", TextArea)
        assert input_bar in controller._surfaces
        assert str(send.label) == "Send"
        assert str(new.label) == "New"

        label_writes: list[tuple[str, str]] = []
        original_set_label = input_bar._set_btn_label

        def record_label(label: str, *, button_id: str = "send-btn", defer_geometry: bool = False) -> None:
            label_writes.append((button_id, label))
            original_set_label(label, button_id=button_id, defer_geometry=defer_geometry)

        monkeypatch.setattr(input_bar, "_set_btn_label", record_label)
        assert controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED

        assert ("send-btn", "发送") in label_writes
        assert str(send.label) == "发送"
        assert str(new.label) == "新建"
        assert send.styles.width is not None
        assert send.styles.width.value == cell_len("发送") + 4
        assert text_area.placeholder.plain == (
            "输入消息（Ctrl+O 打开编辑器）   #  智能体  $  模型  /  命令  @  文件  !  终端"  # noqa: RUF001
        )
        assert text_area.render_line(0).text.rstrip().endswith("Ctrl+J 换行")

        signature_after_switch = input_bar._interaction_signature
        input_bar.agent_running = True
        await pilot.pause()
        assert input_bar._interaction_signature != signature_after_switch
        assert str(send.label) == "中断"
        assert text_area.placeholder.plain == "注入消息  /  命令  @  文件"

        input_bar.agent_running = False
        adapter_screen = type("_AdapterScreen", (), {"query_one": lambda _self, _type: input_bar})()
        adapter = MainScreenViewAdapter(adapter_screen)  # type: ignore[arg-type]
        adapter.set_retry_mode(True, label=INPUT_RETRY.bind())
        assert str(send.label) == "重试"
        assert controller.switch_locale("en").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        assert str(send.label) == "Retry"

        assert controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        input_bar.retry_mode = False
        input_bar.agent_running = True
        input_bar.lock_with_text()
        assert str(send.label) == "已排队"

    assert input_bar is not None
    assert input_bar not in controller._surfaces


@pytest.mark.asyncio
async def test_input_bar_button_labels_treat_translation_markup_as_literal() -> None:
    class MarkupLocalizer:
        effective_locale = "zh-Hans"

        def render(self, reference: MessageRef) -> str:
            if reference.definition is INPUT_SEND:
                return "[red]发送[/red]"
            return reference.definition.fallback

    controller = LocaleController(
        Settings(locale="zh-Hans"),
        localizer=MarkupLocalizer(),  # type: ignore[arg-type]
    )

    async with _LocalizedInputBarApp(controller).run_test(size=(120, 15)) as pilot:
        input_bar = pilot.app.query_one(InputBar)
        send = input_bar.query_one("#send-btn", Button)
        # Markup-looking translation text stays literal (never parsed as
        # Textual markup) and the pinned width matches the literal cells.
        assert send.label.plain == "[red]发送[/red]"
        assert not send.label.spans
        assert send.styles.width is not None
        assert send.styles.width.value == cell_len("[red]发送[/red]") + 4


@pytest.mark.asyncio
async def test_input_bar_snapshot_and_replace_draft_cursor_semantics() -> None:
    app = _InputBarApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", TextArea)
        input_bar.value = "before"
        text_area.move_cursor((0, 2))

        snapshot = input_bar.snapshot_draft()

        assert snapshot.text == "before"
        assert snapshot.cursor_location == (0, 2)
        assert snapshot.revision == input_bar.draft_revision
        assert input_bar.value == "before"
        assert text_area.cursor_location == (0, 2)

        input_bar.replace_draft("first\nsecond")
        await pilot.pause()

        assert input_bar.value == "first\nsecond"
        assert text_area.cursor_location == (1, 6)
        assert input_bar.query_one("#send-btn", Button).disabled is False
        assert input_bar.draft_revision > snapshot.revision


@pytest.mark.asyncio
async def test_input_bar_ctrl_j_keeps_cursor_visible_after_height_cap() -> None:
    app = _InputBarApp()

    async with app.run_test(size=(130, 15)) as pilot:
        await pilot.pause()
        text_area = app.query_one(InputBar).query_one("#chat-input", TextArea)
        text_area.focus()

        await pilot.press("1")
        for line in range(2, 9):
            await pilot.press("ctrl+j")
            await pilot.press(str(line))
        await pilot.pause()

        assert text_area.document.line_count == 8
        assert text_area.content_size.height == 7

        await pilot.press("ctrl+j")
        await pilot.pause()

        cursor_y = text_area.cursor_location[0]
        scroll_y = round(text_area.scroll_y)
        assert scroll_y <= cursor_y < scroll_y + text_area.content_size.height


@pytest.mark.asyncio
async def test_input_bar_keeps_arrow_keys_while_agent_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running drafts should still support normal history and cursor movement."""
    monkeypatch.setattr(
        input_bar_module,
        "append_history",
        lambda text, *, session_id=None, instance_id=None, cwd=None: None,
    )
    app = _InputBarApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", TextArea)
        input_bar.add_to_history("old prompt")
        input_bar.agent_running = True
        input_bar.value = "draft"
        text_area.focus()
        await pilot.pause()

        await pilot.press("up")
        assert input_bar.value == "old prompt"

        text_area.move_cursor((0, 2))
        await pilot.press("left")
        assert text_area.cursor_location == (0, 1)
        await pilot.press("right")
        assert text_area.cursor_location == (0, 2)


@pytest.mark.asyncio
async def test_input_bar_suppresses_arrow_keys_for_queued_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once an injection is queued, disabling the input must block arrow edits."""
    monkeypatch.setattr(
        input_bar_module,
        "append_history",
        lambda text, *, session_id=None, instance_id=None, cwd=None: None,
    )
    app = _InputBarApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        text_area = input_bar.query_one("#chat-input", TextArea)
        input_bar.add_to_history("old prompt")
        input_bar.agent_running = True
        input_bar.value = "draft"
        input_bar.lock_with_text()
        text_area.focus()
        await pilot.pause()

        assert text_area.disabled is True

        await pilot.press("up")
        await pilot.press("down")

        assert input_bar.value == "draft"

        text_area.move_cursor((0, 2))
        await pilot.press("left")
        assert text_area.cursor_location == (0, 2)
        await pilot.press("right")
        assert text_area.cursor_location == (0, 2)


@pytest.mark.asyncio
async def test_input_bar_keeps_running_suggestion_arrow_navigation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unlocked running input should still route up/down to active suggestions."""
    monkeypatch.setattr(input_bar_module, "load_history", list)
    app = _InputBarApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        input_bar = app.query_one(InputBar)
        input_bar.agent_running = True
        input_bar.set_suggestions_active(True, mode="files")
        input_bar.query_one("#chat-input", TextArea).focus()
        await pilot.pause()

        await pilot.press("up")
        await pilot.press("down")

        assert app.suggestion_directions == ["up", "down"]
