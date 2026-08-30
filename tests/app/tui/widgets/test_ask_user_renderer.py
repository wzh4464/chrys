# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the ask_user tool renderer."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable

import pytest
from textual.app import App, ComposeResult
from textual.css.query import NoMatches
from textual.events import Resize
from textual.geometry import Size
from textual.widgets import Button, Static, TextArea

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.widgets.ask_user_controls import (
    _CUSTOM_RESPONSE_PLACEHOLDER,
    ASK_USER_INPUT_MIN_HEIGHT,
    AskUserResponseFooter,
    _AskUserTextArea,
)
from chrys.app.tui.widgets.chat import tool_renderers
from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.chat.renderers.ask_user import (
    _INLINE_BUTTON_ROW_HEIGHT,
    _INLINE_PANEL_FRAME_ROWS,
    AskUserInlineResized,
    AskUserInlineSubmitted,
    AskUserToolCall,
    _answer_from_result,
)
from chrys.app.tui.widgets.chat.tool_call import BaseToolCard, ToolCardHeader
from chrys.app.tui.widgets.chat.tool_renderers import create_tool_widget
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.app.tui.widgets.text_area import EnhancedTextArea
from chrys.foundation.config.settings import Settings
from chrys.foundation.i18n import Localizer
from chrys.foundation.i18n.formatting import format_message
from chrys.foundation.tool_kinds import KIND_ASK_USER
from chrys.foundation.tool_result_metadata import TOOL_FAILED_METADATA_KEY
from tests.support.waiting import wait_until, wait_until_quiet

_RENDERER_MODULES = (
    "chrys.app.tui.widgets.chat.renderers.ask_user",
    "chrys.app.tui.widgets.chat.renderers.execute",
    "chrys.app.tui.widgets.chat.renderers.file_edit",
    "chrys.app.tui.widgets.chat.renderers.read_file",
    "chrys.app.tui.widgets.chat.renderers.search",
    "chrys.app.tui.widgets.chat.renderers.skill",
    "chrys.app.tui.widgets.chat.renderers.sleep",
    "chrys.app.tui.widgets.chat.renderers.sub_agent",
)
_MISSING = object()


# Slow geometry waits must exhaust their own deadline (a clean assert), never
# the global per-test pytest-timeout hard cap — its thread method kills the
# whole xdist worker, surfacing as "worker gwN crashed".
pytestmark = pytest.mark.timeout(120)


def test_custom_response_placeholder_keeps_english_and_localizes_chinese() -> None:
    assert format_message(_CUSTOM_RESPONSE_PLACEHOLDER.bind()) == "Type a custom response..."
    assert Localizer("zh-Hans").render(_CUSTOM_RESPONSE_PLACEHOLDER.bind()) == "输入自定义回复..."


class _LocalizedApp(App):
    locale_controller = LocaleController(Settings(locale="en"))


async def _wait_for_layout(pilot: object, predicate: Callable[[], bool], *, timeout: float = 30.0) -> None:
    # Bare pilot.pause() pumps drain in ~0ms on loaded CI workers before
    # deferred layout lands; poll against a real deadline instead. Loaded CI
    # runners have blown a 5s deadline on these geometry waits (macOS twice,
    # different tests each time) and later a 15s one (Windows), so the long
    # default applies file-wide.
    assert await wait_until(predicate, timeout=timeout, pilot=pilot)


def test_answer_from_result_strips_only_middleware_prefix() -> None:
    assert _answer_from_result("User response: yes\nfull answer") == "yes\nfull answer"
    assert _answer_from_result("User response:   indented") == "  indented"
    assert _answer_from_result("Error: user did not respond") == "Error: user did not respond"


def test_factory_routes_ask_user_tool_name_to_renderer() -> None:
    widget = create_tool_widget("c1", "ask_user", "", args={"question": "Proceed?"})

    assert isinstance(widget, AskUserToolCall)


def test_factory_routes_ask_user_runtime_kind_to_renderer() -> None:
    widget = create_tool_widget("c1", "custom_question", KIND_ASK_USER, args={"question": "Proceed?"})

    assert isinstance(widget, AskUserToolCall)


@pytest.mark.asyncio
async def test_ask_user_text_area_resize_dispatch_skips_duplicate_base_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The auto-growing ask_user textarea should not let Textual re-run base resize."""
    base_resize_calls: list[_AskUserTextArea] = []
    resize_to_content_calls: list[tuple[_AskUserTextArea, bool]] = []

    def base_resize_spy(text_area: _AskUserTextArea) -> None:
        base_resize_calls.append(text_area)

    def resize_to_content_spy(text_area: _AskUserTextArea, *, rewrap: bool = False) -> int:
        resize_to_content_calls.append((text_area, rewrap))
        return 3

    monkeypatch.setattr(TextArea, "_on_resize", base_resize_spy)
    monkeypatch.setattr(_AskUserTextArea, "resize_to_content", resize_to_content_spy)
    text_area = _AskUserTextArea()
    event = Resize(Size(10, 3), Size(10, 3))

    await text_area._on_message(event)

    assert resize_to_content_calls == [(text_area, True)]
    assert base_resize_calls == []
    assert event._no_default_action is True


@pytest.mark.asyncio
async def test_ask_user_text_area_ignores_height_only_resize_after_autogrow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-height feedback should not become a second edit-time resize pass."""
    resize_to_content_calls: list[tuple[_AskUserTextArea, bool]] = []

    def resize_to_content_spy(text_area: _AskUserTextArea, *, rewrap: bool = False) -> int:
        resize_to_content_calls.append((text_area, rewrap))
        text_area._last_wrapped_width = text_area.wrap_width
        return 3

    monkeypatch.setattr(_AskUserTextArea, "resize_to_content", resize_to_content_spy)
    monkeypatch.setattr(_AskUserTextArea, "wrap_width", property(lambda _self: 8))
    text_area = _AskUserTextArea()

    await text_area._on_message(Resize(Size(10, 3), Size(10, 3)))
    await text_area._on_message(Resize(Size(10, 4), Size(10, 4)))

    assert resize_to_content_calls == [(text_area, True)]


@pytest.mark.asyncio
async def test_ask_user_text_area_rewraps_when_scrollbar_changes_wrap_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scrollbar can narrow TextArea.wrap_width without changing widget width."""
    resize_to_content_calls: list[tuple[_AskUserTextArea, bool]] = []
    wrap_width_state = {"value": 8}

    def resize_to_content_spy(text_area: _AskUserTextArea, *, rewrap: bool = False) -> int:
        resize_to_content_calls.append((text_area, rewrap))
        text_area._last_wrapped_width = text_area.wrap_width
        return 7

    monkeypatch.setattr(_AskUserTextArea, "resize_to_content", resize_to_content_spy)
    monkeypatch.setattr(_AskUserTextArea, "wrap_width", property(lambda _self: wrap_width_state["value"]))
    text_area = _AskUserTextArea()

    await text_area._on_message(Resize(Size(10, 5), Size(10, 5)))
    wrap_width_state["value"] = 7
    await text_area._on_message(Resize(Size(10, 7), Size(10, 7)))

    assert resize_to_content_calls == [(text_area, True), (text_area, True)]


def test_ask_user_text_area_measurement_self_heals_a_queued_width_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ancestor measurement must rewrap before the TextArea Resize arrives."""
    wrap_width = {"value": 57}
    monkeypatch.setattr(_AskUserTextArea, "wrap_width", property(lambda _self: wrap_width["value"]))
    text_area = _AskUserTextArea(text="x" * 60, soft_wrap=True)
    text_area.resize_to_content(rewrap=True)

    assert [len(section) for section in text_area.wrapped_document.lines[0]] == [57, 3]

    wrap_width["value"] = 20
    text_area.resize_to_content()

    assert [len(section) for section in text_area.wrapped_document.lines[0]] == [20, 20, 20]
    assert text_area._last_wrapped_width == 20


@pytest.mark.asyncio
async def test_ask_user_text_edit_syncs_response_state_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Text edits and their resize feedback should share one layout sync path."""

    class FooterApp(App):
        def compose(self) -> ComposeResult:
            yield AskUserResponseFooter("req-1")

    async with FooterApp().run_test() as pilot:
        footer = pilot.app.query_one(AskUserResponseFooter)
        text_area = footer.query_one("#askuser-input", _AskUserTextArea)
        await pilot.pause()
        resize_calls: list[tuple[_AskUserTextArea, bool]] = []

        def resize_to_content_spy(area: _AskUserTextArea, *, rewrap: bool = False) -> int:
            resize_calls.append((area, rewrap))
            # Preserve the synchronization side effects that suppress a late
            # same-width Resize message on slower CI workers.
            area._last_wrapped_width = area.wrap_width
            area._last_reported_height = ASK_USER_INPUT_MIN_HEIGHT
            return ASK_USER_INPUT_MIN_HEIGHT

        monkeypatch.setattr(_AskUserTextArea, "resize_to_content", resize_to_content_spy)
        await wait_until_quiet(
            lambda: len(resize_calls),
            description="ask-user resize call count",
            pilot=pilot,
        )
        resize_calls.clear()
        text_area.insert("x")

        assert await wait_until(lambda: bool(resize_calls), pilot=pilot)
        await text_area._on_message(Resize(text_area.size, text_area.size))
        await footer._on_message(Resize(Size(max(1, footer.size.width), footer.size.height + 1), footer.size))

        assert not await wait_until(lambda: len(resize_calls) > 1, timeout=0.2, interval=0.01, pilot=pilot)
        assert resize_calls == [(text_area, False)]


@pytest.mark.asyncio
async def test_ask_user_text_edit_does_not_rewrap_the_full_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Changed handler should consume TextArea's incremental wrap result."""

    class FooterApp(App):
        def compose(self) -> ComposeResult:
            yield AskUserResponseFooter("req-1")

    async with FooterApp().run_test() as pilot:
        text_area = pilot.app.query_one("#askuser-input", _AskUserTextArea)
        await pilot.pause()
        rewrap_calls: list[None] = []
        monkeypatch.setattr(text_area, "_rewrap_and_refresh_virtual_size", lambda: rewrap_calls.append(None))

        text_area.insert("x")
        await pilot.pause()

        assert rewrap_calls == []


@pytest.mark.asyncio
async def test_ask_user_text_area_mounts_with_blink_timer_paused() -> None:
    class FooterApp(App):
        def compose(self) -> ComposeResult:
            yield AskUserResponseFooter("req-1")

    async with FooterApp().run_test() as pilot:
        text_area = pilot.app.query_one("#askuser-input", _AskUserTextArea)
        text_area.focus()
        await pilot.pause()

        assert text_area.cursor_blink is False
        assert text_area.blink_timer._active.is_set() is False


def test_ask_user_renderer_uses_base_tool_card_contract() -> None:
    tool = AskUserToolCall("c1", "ask_user", args={"question": "Proceed?"})

    assert isinstance(tool, BaseToolCard)
    assert tool.call_id == "c1"
    assert tool.tool_name == "ask_user"
    assert tool.status == "running"
    assert tool.result_text == ""
    assert tool.duration_ms == 0
    assert tool.args == {"question": "Proceed?"}


def test_lazy_loader_imports_ask_user_renderer_module() -> None:
    parent_module = sys.modules["chrys.app.tui.widgets.chat.renderers"]
    parent_attrs = vars(parent_module)
    saved_parent_attrs = {
        module_name.rsplit(".", 1)[1]: parent_attrs.get(module_name.rsplit(".", 1)[1], _MISSING)
        for module_name in _RENDERER_MODULES
    }
    saved_registry = dict(tool_renderers._REGISTRY)
    saved_kind_registry = dict(tool_renderers._KIND_REGISTRY)
    saved_loaded = tool_renderers._loaded
    saved_modules = {module_name: sys.modules.pop(module_name, _MISSING) for module_name in _RENDERER_MODULES}
    for attr in saved_parent_attrs:
        parent_attrs.pop(attr, None)
    try:
        tool_renderers._REGISTRY.clear()
        tool_renderers._KIND_REGISTRY.clear()
        tool_renderers._loaded = False

        widget = create_tool_widget("c1", "ask_user", "", args={"question": "Proceed?"})

        assert widget.__class__.__name__ == "AskUserToolCall"
    finally:
        for module_name, module in saved_modules.items():
            if module is _MISSING:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = module
        for attr, value in saved_parent_attrs.items():
            if value is _MISSING:
                parent_attrs.pop(attr, None)
            else:
                parent_attrs[attr] = value
        tool_renderers._REGISTRY.clear()
        tool_renderers._REGISTRY.update(saved_registry)
        tool_renderers._KIND_REGISTRY.clear()
        tool_renderers._KIND_REGISTRY.update(saved_kind_registry)
        tool_renderers._loaded = saved_loaded


@pytest.mark.asyncio
async def test_ask_user_renderer_mounts_question_and_full_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield AskUserToolCall(
                "c1",
                "ask_user",
                args_summary=json.dumps({"question": "**Proceed?**\n\n- yes\n- no"}),
            )

    async with ToolApp().run_test() as pilot:
        copied: list[str] = []
        monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", copied.append)

        tool = pilot.app.query_one(AskUserToolCall)
        question_panel = tool.query_one("#ask-question-panel")
        answer_panel = tool.query_one("#ask-answer-panel")
        assert question_panel.border_title == "Question"
        assert question_panel.border_subtitle is None
        assert answer_panel.border_title == "Answer"
        assert answer_panel.border_subtitle is None
        assert tool.query_one("#ask-question", VirtualizedMarkdown).source == "**Proceed?**\n\n- yes\n- no"

        tool.set_complete("User response: first line\nsecond line", duration_ms=1500)
        await pilot.pause()

        assert tool.status == "complete"
        assert tool.query_one("#ask-answer", Static).render().plain == "first line\nsecond line"
        assert tool.query_one(ToolCardHeader).actions_visible is True

        tool.copy_tool_execution()
        await pilot.pause()

        assert len(copied) == 1
        payload = copied[-1]
        assert "~~~markdown" in payload
        assert "**Proceed?**" in payload
        assert "## Answer" in payload
        assert "User response:" not in payload
        assert "first line\nsecond line" in payload


@pytest.mark.asyncio
async def test_ask_user_renderer_records_rejected_completion_approval() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield AskUserToolCall("c1", "ask_user", args={"question": "Proceed?"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(AskUserToolCall)

        tool.set_complete("Error: rejected", duration_ms=10, approval="user_rejected")
        await pilot.pause()

        assert tool.approval == "user_rejected"
        assert tool.has_class("-rejected")
        assert tool.query_one("#ask-answer", Static).render().plain == "Error: rejected"


@pytest.mark.asyncio
async def test_ask_user_renderer_uses_structured_failure_metadata() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield AskUserToolCall("c1", "ask_user", args={"question": "Proceed?"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(AskUserToolCall)

        tool.set_complete(
            "Error: literal answer text",
            duration_ms=10,
            metadata={TOOL_FAILED_METADATA_KEY: False},
        )
        await pilot.pause()

        assert tool.has_class("-success")
        assert not tool.has_class("-error")


@pytest.mark.asyncio
async def test_ask_user_renderer_sets_error_status_for_failed_results() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield AskUserToolCall("c1", "ask_user", args={"question": "Proceed?"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(AskUserToolCall)

        tool.set_complete("Error: user did not respond", duration_ms=10, metadata={TOOL_FAILED_METADATA_KEY: True})
        await pilot.pause()

        assert tool.status == "error"
        assert tool.has_class("-error")


@pytest.mark.asyncio
async def test_ask_user_renderer_inline_prompt_submits_once_and_clears_on_result() -> None:
    class ToolApp(_LocalizedApp):
        def __init__(self) -> None:
            super().__init__()
            self.submitted: list[tuple[str, str, str]] = []

        def compose(self) -> ComposeResult:
            yield AskUserToolCall("c1", "ask_user", args={"question": "Pick?"})

        def on_ask_user_inline_submitted(self, event: AskUserInlineSubmitted) -> None:
            self.submitted.append((event.call_id, event.request_id, event.text))
            event.stop()

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(AskUserToolCall)

        assert tool.show_inline_prompt("req-1", ["Python", "Go"], draft_text="draft")
        await pilot.pause()

        assert tool.query_one("#askuser-input", EnhancedTextArea).text == "draft"
        tool.query_one("#askuser-opt-0", Button).press()
        tool.query_one("#askuser-opt-0", Button).press()
        await pilot.pause()

        assert pilot.app.submitted == [("c1", "req-1", "Python")]

        tool.set_complete("User response: Python", duration_ms=10)
        await _wait_for_layout(pilot, lambda: len(tool.query("#ask-inline")) == 0)

        with pytest.raises(NoMatches):
            tool.query_one("#ask-inline")
        assert tool.query_one("#ask-answer", Static).render().plain == "Python"


@pytest.mark.asyncio
async def test_chat_panel_inline_ask_user_text_area_accepts_typing_after_click() -> None:
    class PanelApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    async with PanelApp().run_test(size=(100, 30)) as pilot:
        panel = pilot.app.query_one(ChatPanel)

        await panel.add_tool_start("c1", "ask_user", KIND_ASK_USER, args={"question": "Pick?"})
        await pilot.pause()

        assert panel.show_ask_user_inline("c1", "req-1", ["Python"])
        await pilot.pause()

        text_area = panel.query_one("#askuser-input", EnhancedTextArea)
        await pilot.click("#askuser-input")
        await pilot.pause()
        if pilot.app.screen.focused is not text_area:
            text_area.focus()
            await pilot.pause()
        await pilot.press("x", "y")
        await pilot.pause()

        assert text_area.text == "xy"


@pytest.mark.asyncio
async def test_ask_user_renderer_inline_input_keeps_its_frame_without_the_modal_stylesheet() -> None:
    """The compact TextArea drops its frame with an !important rule; the inline
    renderer must restore it on its own, not by riding on the ask_user dialog's
    stylesheet having been loaded earlier in the session."""

    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield AskUserToolCall("c1", "ask_user", args={"question": "Pick?"})

    async with ToolApp().run_test(size=(100, 30)) as pilot:
        tool = pilot.app.query_one(AskUserToolCall)
        assert tool.show_inline_prompt("req-1", ["Python"])
        await pilot.pause()

        input_area = tool.query_one("#askuser-input", EnhancedTextArea)
        assert input_area.has_class("-textual-compact")
        assert input_area.styles.border_top[0] == "round"
        assert input_area.styles.border_left[0] == "round"


@pytest.mark.asyncio
async def test_ask_user_renderer_inline_prompt_without_options_keeps_footer_visible() -> None:
    class ToolApp(_LocalizedApp):
        def __init__(self) -> None:
            super().__init__()
            self.submitted: list[tuple[str, str, str]] = []

        def compose(self) -> ComposeResult:
            yield AskUserToolCall("c1", "ask_user", args={"question": "Pick?"})

        def on_ask_user_inline_submitted(self, event: AskUserInlineSubmitted) -> None:
            self.submitted.append((event.call_id, event.request_id, event.text))
            event.stop()

    async with ToolApp().run_test(size=(100, 30)) as pilot:
        tool = pilot.app.query_one(AskUserToolCall)

        assert tool.show_inline_prompt("req-1", [])
        await pilot.pause()

        with pytest.raises(NoMatches):
            tool.query_one("#askuser-options")
        inline = tool.query_one("#ask-inline")
        footer = tool.query_one("#askuser-footer")
        assert str(inline.styles.min_height) == "6"
        assert str(footer.styles.min_height) == "6"
        assert str(footer.styles.height) == "auto"
        # ASK_USER_INPUT_MIN_HEIGHT counts the round frame, so measure the outer box.
        assert tool.query_one("#askuser-input", EnhancedTextArea).outer_size.height >= 3

        answer_panel = tool.query_one("#ask-answer-panel")
        initial_answer_height = answer_panel.region.height
        input_area = tool.query_one("#askuser-input", EnhancedTextArea)
        input_area.insert("Line one\nLine two\nLine three")
        await _wait_for_layout(
            pilot,
            lambda: input_area.outer_size.height >= 5 and answer_panel.region.height > initial_answer_height,
        )

        assert input_area.outer_size.height >= 5
        assert answer_panel.region.height > initial_answer_height
        tool.query_one("#askuser-submit", Button).press()
        # Button.Pressed → AskUserInlineSubmitted is two message hops; a single
        # pause races on loaded CI workers, so poll the asserted condition.
        await _wait_for_layout(pilot, lambda: bool(pilot.app.submitted))

        assert pilot.app.submitted == [("c1", "req-1", "Line one\nLine two\nLine three")]


@pytest.mark.asyncio
async def test_chat_panel_inline_ask_user_blank_newlines_grow_answer_body() -> None:
    class PanelApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    async with PanelApp().run_test(size=(120, 30)) as pilot:
        panel = pilot.app.query_one(ChatPanel)

        await panel.add_tool_start("c1", "ask_user", KIND_ASK_USER, args={"question": "What next?"})
        await pilot.pause()

        assert panel.show_ask_user_inline("c1", "req-1", [])
        await pilot.pause()

        input_area = panel.query_one("#askuser-input", EnhancedTextArea)
        answer_panel = panel.query_one("#ask-answer-panel")
        tool = panel.query_one(AskUserToolCall)
        expected_initial_answer_height = (
            ASK_USER_INPUT_MIN_HEIGHT + _INLINE_BUTTON_ROW_HEIGHT + _INLINE_PANEL_FRAME_ROWS
        )
        # show_inline_prompt schedules follow-up layout work. A bare pause can
        # return before it lands on loaded CI workers, so establish the known
        # empty-input geometry before taking a baseline for the growth checks.
        await _wait_for_layout(
            pilot,
            lambda: (
                input_area.outer_size.height == ASK_USER_INPUT_MIN_HEIGHT
                and answer_panel.region.height == expected_initial_answer_height
                and tool.region.height > answer_panel.region.height
            ),
        )
        initial_input_height = input_area.outer_size.height
        initial_answer_height = answer_panel.region.height
        initial_tool_height = tool.region.height

        await pilot.click("#askuser-input")
        await pilot.pause()
        # Click-to-focus races layout on loaded CI workers; fall back to
        # focusing directly (same pattern as the option-click test above) so
        # the key presses cannot land nowhere.
        if pilot.app.screen.focused is not input_area:
            input_area.focus()
            await pilot.pause()
        await pilot.press("enter", "enter")
        await _wait_for_layout(
            pilot,
            lambda: (
                input_area.text == "\n\n"
                and input_area.outer_size.height > initial_input_height
                and answer_panel.region.height > initial_answer_height
                and tool.region.height > initial_tool_height
            ),
        )

        assert input_area.text == "\n\n"
        assert input_area.outer_size.height > initial_input_height
        assert answer_panel.region.height > initial_answer_height
        expected_answer_height = (
            int(input_area.styles.height.value) + _INLINE_BUTTON_ROW_HEIGHT + _INLINE_PANEL_FRAME_ROWS
        )
        assert str(answer_panel.styles.height) == str(expected_answer_height)
        assert tool.region.height > initial_tool_height
        assert str(tool.styles.height) == "auto"

        grown_input_height = input_area.outer_size.height
        grown_answer_height = answer_panel.region.height
        grown_tool_height = tool.region.height

        # The shrink assertion exercises the layout response to a document
        # edit, not Textual's focus/keyboard routing. Use the public edit API
        # so a concurrent focus change cannot turn this into an input race.
        input_area.delete((0, 0), input_area.document.end, maintain_selection_offset=False)
        await _wait_for_layout(
            pilot,
            lambda: (
                input_area.text == ""
                and input_area.outer_size.height == ASK_USER_INPUT_MIN_HEIGHT
                and answer_panel.region.height == expected_initial_answer_height
                and tool.region.height < grown_tool_height
            ),
        )

        assert input_area.text == ""
        assert input_area.outer_size.height == ASK_USER_INPUT_MIN_HEIGHT
        assert answer_panel.region.height == expected_initial_answer_height
        assert str(answer_panel.styles.height) == str(expected_initial_answer_height)
        assert input_area.outer_size.height < grown_input_height
        assert answer_panel.region.height < grown_answer_height
        assert tool.region.height < grown_tool_height


@pytest.mark.asyncio
async def test_chat_panel_ask_user_inline_resize_message_reaches_relayout_after_registry_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PanelApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    async with PanelApp().run_test(size=(120, 30)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_tool_start("c1", "ask_user", KIND_ASK_USER, args={"question": "What next?"})
        await pilot.pause()

        assert panel.show_ask_user_inline("c1", "req-1", [])
        await pilot.pause()

        tool = panel.query_one(AskUserToolCall)
        calls: list[tuple[str, bool]] = []
        after_refresh: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

        def record_panel_refresh(*, layout: bool = False, **_kwargs: object) -> None:
            calls.append(("panel_refresh", layout))

        def record_anchor_sync() -> None:
            calls.append(("anchor_sync", True))

        def record_after_refresh(callback: object, *args: object, **kwargs: object) -> bool:
            after_refresh.append((callback, args, kwargs))
            return True

        monkeypatch.setattr(panel, "refresh", record_panel_refresh)
        monkeypatch.setattr(panel, "_schedule_anchor_sync", record_anchor_sync)
        monkeypatch.setattr(panel, "call_after_refresh", record_after_refresh)

        tool.post_message(AskUserInlineResized("c1"))
        await pilot.pause()

        # Textual internals may schedule plain repaints (refresh(layout=False))
        # that land inside this instrumented window on slow CI workers. The
        # resize handler's contract (panel.on_inline_prompt_resized) is exactly
        # one relayout followed by one anchor sync — it never issues a plain
        # repaint, so those are filtered rather than counted.
        contract_calls = [call for call in calls if call != ("panel_refresh", False)]
        assert contract_calls == [
            ("panel_refresh", True),
            ("anchor_sync", True),
        ]
        assert after_refresh == []


@pytest.mark.asyncio
async def test_chat_panel_inline_ask_user_resizes_with_wrapped_question() -> None:
    class PanelApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    async with PanelApp().run_test(size=(140, 35)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        question = "This is a long ask_user question that should wrap after the terminal narrows. " * 8

        await panel.add_tool_start("c1", "ask_user", KIND_ASK_USER, args={"question": question})
        await pilot.pause()

        assert panel.show_ask_user_inline("c1", "req-1", [])
        await pilot.pause()

        tool = panel.query_one(AskUserToolCall)
        question_panel = panel.query_one("#ask-question-panel")
        wide_tool_height = tool.region.height
        wide_question_height = question_panel.region.height

        await pilot.resize_terminal(60, 35)
        await _wait_for_layout(
            pilot,
            lambda: question_panel.region.height > wide_question_height and tool.region.height > wide_tool_height,
        )

        assert str(tool.styles.height) == "auto"
        assert question_panel.region.height > wide_question_height
        assert tool.region.height > wide_tool_height
        assert tool.region.height >= question_panel.region.height + panel.query_one("#ask-answer-panel").region.height


@pytest.mark.asyncio
async def test_chat_panel_inline_ask_user_options_leave_room_before_footer() -> None:
    class PanelApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    async with PanelApp().run_test(size=(120, 35)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        options = [
            "All-purpose assistant",
            "Focused professional tool",
            "Creative brainstorm partner",
            "Casual chat companion",
        ]

        await panel.add_tool_start("c1", "ask_user", KIND_ASK_USER, args={"question": "Pick?"})
        await pilot.pause()

        assert panel.show_ask_user_inline("c1", "req-1", options)
        await _wait_for_layout(pilot, lambda: len(panel.query("#askuser-options")) > 0)

        options_widget = panel.query_one("#askuser-options")
        footer = panel.query_one("#askuser-footer")
        answer_panel = panel.query_one("#ask-answer-panel")
        await _wait_for_layout(
            pilot,
            lambda: (
                footer.region.y - (options_widget.region.y + options_widget.region.height) >= 2
                and answer_panel.region.height >= options_widget.region.height + footer.region.height + 4
            ),
        )

        assert options_widget.styles.margin.bottom == 2
        assert footer.region.y - (options_widget.region.y + options_widget.region.height) >= 2
        assert answer_panel.region.height >= options_widget.region.height + footer.region.height + 4


@pytest.mark.asyncio
async def test_chat_panel_inline_ask_user_wrapped_options_resize_answer_body() -> None:
    class PanelApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    async with PanelApp().run_test(size=(140, 35)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        options = [
            "This is a very long option label that wraps after the terminal narrows significantly",
            "Another long option label that should also wrap and force the inline panel to grow",
        ]

        await panel.add_tool_start("c1", "ask_user", KIND_ASK_USER, args={"question": "Pick?"})
        await pilot.pause()

        assert panel.show_ask_user_inline("c1", "req-1", options)
        await _wait_for_layout(pilot, lambda: len(panel.query("#askuser-options")) > 0)

        options_widget = panel.query_one("#askuser-options")
        footer = panel.query_one("#askuser-footer")
        answer_panel = panel.query_one("#ask-answer-panel")
        option_buttons = list(panel.query("#askuser-options > Button"))

        await pilot.resize_terminal(45, 35)
        # Poll every asserted condition: the container height settles a frame
        # after the buttons wrap, and loaded CI workers need the long deadline.
        await _wait_for_layout(
            pilot,
            lambda: (
                any(button.region.height > 3 for button in option_buttons)
                and options_widget.region.height == sum(button.region.height for button in option_buttons)
                and answer_panel.region.height >= options_widget.region.height + footer.region.height + 4
            ),
        )

        assert any(button.region.height > 3 for button in option_buttons)
        assert options_widget.region.height == sum(button.region.height for button in option_buttons)
        assert answer_panel.region.height >= options_widget.region.height + footer.region.height + 4


@pytest.mark.asyncio
async def test_chat_panel_inline_ask_user_many_options_do_not_collapse_under_multiline_input() -> None:
    class PanelApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    async with PanelApp().run_test(size=(120, 35)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        options = ["写代码", "读书", "运动", "看电影/剧", "睡觉", "outdoors / 户外", "\u200b"]

        await panel.add_tool_start("c1", "ask_user", KIND_ASK_USER, args={"question": "你周末通常喜欢做什么?"})
        await pilot.pause()

        assert panel.show_ask_user_inline("c1", "req-1", options)
        await _wait_for_layout(pilot, lambda: len(panel.query("#askuser-opt-5")) > 0)

        input_area = panel.query_one("#askuser-input", EnhancedTextArea)
        options_widget = panel.query_one("#askuser-options")
        input_area.insert("abca\nasdf\nasdf\nasdf")

        option_buttons = list(panel.query("#askuser-options > Button"))
        footer = panel.query_one("#askuser-footer")
        answer_panel = panel.query_one("#ask-answer-panel")
        await _wait_for_layout(
            pilot,
            lambda: (
                input_area.outer_size.height >= 6
                and str(options_widget.styles.height) == "18"
                and all(str(button.styles.height) == "3" for button in option_buttons)
                and all(button.region.height >= 3 for button in option_buttons)
                and footer.region.y >= option_buttons[-1].region.y + option_buttons[-1].region.height + 2
                and answer_panel.region.height >= 31
            ),
        )

        assert str(options_widget.styles.height) == "18"
        assert all(str(button.styles.height) == "3" for button in option_buttons)
        assert all(button.region.height >= 3 for button in option_buttons)
        assert footer.region.y >= option_buttons[-1].region.y + option_buttons[-1].region.height + 2
        assert [button.label.plain for button in option_buttons] == [
            "写代码",
            "读书",
            "运动",
            "看电影/剧",
            "睡觉",
            "outdoors / 户外",
        ]

        stable_answer_height = answer_panel.region.height
        stable_tool_height = panel.query_one(AskUserToolCall).region.height
        for _ in range(3):
            await pilot.pause()
            assert answer_panel.region.height == stable_answer_height
            assert panel.query_one(AskUserToolCall).region.height == stable_tool_height


@pytest.mark.asyncio
async def test_chat_panel_inline_ask_user_ignores_blank_options() -> None:
    class PanelApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    async with PanelApp().run_test(size=(120, 35)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        options = ["写一段示例代码", "解释一个概念", "审查你的代码", "   ", "\u200b", "\ufeff"]

        await panel.add_tool_start("c1", "ask_user", KIND_ASK_USER, args={"question": "你希望我接下来展示什么?"})
        await pilot.pause()

        assert panel.show_ask_user_inline("c1", "req-1", options)
        await _wait_for_layout(pilot, lambda: len(panel.query("#askuser-opt-2")) > 0)

        input_area = panel.query_one("#askuser-input", EnhancedTextArea)
        options_widget = panel.query_one("#askuser-options")
        option_buttons = list(panel.query("#askuser-options > Button"))

        assert [button.label.plain for button in option_buttons] == ["写一段示例代码", "解释一个概念", "审查你的代码"]
        with pytest.raises(NoMatches):
            panel.query_one("#askuser-opt-3")

        input_area.insert("1111\n2222\n3333")
        await _wait_for_layout(
            pilot,
            lambda: (
                str(options_widget.styles.height) == "9" and all(button.region.height >= 3 for button in option_buttons)
            ),
        )

        assert options_widget.region.height == 9


@pytest.mark.asyncio
async def test_chat_panel_inline_ask_user_expands_and_locks_tool_group_until_result() -> None:
    class PanelApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    async with PanelApp().run_test(size=(100, 30)) as pilot:
        panel = pilot.app.query_one(ChatPanel)

        await panel.add_tool_start("c1", "ask_user", KIND_ASK_USER, args={"question": "Pick?"})
        await pilot.pause()
        group = panel._tool_groups_by_call_id["c1"]
        group.collapsed = True
        await pilot.pause()

        assert panel.show_ask_user_inline("c1", "req-1", ["Python"])
        await pilot.pause()

        assert group.collapsed is False
        group.collapsed = True
        assert group.collapsed is False
        panel.toggle_fold_all()
        assert group.collapsed is False
        group.on_tool_group_title_clicked()
        assert group.collapsed is False

        await panel.add_tool_result("c1", "ask_user", "User response: Python", duration_ms=5)
        await pilot.pause()

        group.on_tool_group_title_clicked()
        assert group.collapsed is True


@pytest.mark.asyncio
async def test_chat_panel_clears_inline_ask_user_prompts_across_tool_groups() -> None:
    class PanelApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    async with PanelApp().run_test(size=(100, 30)) as pilot:
        panel = pilot.app.query_one(ChatPanel)

        await panel.add_tool_start("c1", "ask_user", KIND_ASK_USER, args={"question": "First?"})
        await panel.add_agent_message("between tools")
        await panel.add_tool_start("c2", "ask_user", KIND_ASK_USER, args={"question": "Second?"})
        await pilot.pause()

        first_group = panel._tool_groups_by_call_id["c1"]
        second_group = panel._tool_groups_by_call_id["c2"]
        assert first_group is not second_group

        assert panel.show_ask_user_inline("c1", "req-1", ["Python"])
        await pilot.pause()
        first_group.on_tool_group_title_clicked()
        assert first_group.collapsed is False

        panel.clear_ask_user_inline_prompts()
        await pilot.pause()

        with pytest.raises(NoMatches):
            panel.query_one("#ask-inline")
        first_group.on_tool_group_title_clicked()
        assert first_group.collapsed is True
