# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the ask-user dialog."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalGroup
from textual.css.query import NoMatches
from textual.events import MouseDown, MouseUp
from textual.geometry import Offset
from textual.selection import Selection
from textual.widgets import Button, Static

from chrys.app.tui.screens.dialogs.ask_user import AskUserDialog, AskUserInlineResult
from chrys.app.tui.theme import CHRYS_ANSI_THEME
from chrys.app.tui.widgets import StableAutoHeightScroll
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.app.tui.widgets.text_area import EnhancedTextArea
from tests.support.paths import SRC_ROOT
from tests.support.waiting import wait_for

_CHRYS_CSS = SRC_ROOT / "chrys" / "app" / "tui" / "chrys.tcss"


class _DialogHost(App):
    def compose(self) -> ComposeResult:
        yield Static("placeholder")


class _AnsiDialogHost(App):
    CSS_PATH = str(_CHRYS_CSS)

    def __init__(self) -> None:
        super().__init__()
        self.register_theme(CHRYS_ANSI_THEME)
        self.theme = "chrys-ansi"

    def compose(self) -> ComposeResult:
        yield Static("placeholder")

    def on_mount(self) -> None:
        self.set_class(True, "-chrys-ansi")


@pytest.mark.asyncio
async def test_ask_user_dialog_custom_response_uses_enhanced_text_area() -> None:
    """The custom response field should use the shared textarea input behavior."""
    dialog = AskUserDialog(request_id="ask-1", question="Which path?")
    results: list[tuple[str, str]] = []

    app = _DialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        input_area = dialog.query_one("#askuser-input", EnhancedTextArea)
        assert input_area.has_focus
        input_area.insert("Use src/chrys")
        await pilot.press("ctrl+a")
        await pilot.pause()
        assert input_area.selected_text == "Use src/chrys"

        dialog.query_one("#askuser-submit", Button).press()
        await pilot.pause()

    assert results == [("ask-1", "Use src/chrys")]


@pytest.mark.asyncio
async def test_ask_user_dialog_enter_in_text_area_inserts_newline() -> None:
    """Enter in the custom response field should expand text instead of submitting."""
    dialog = AskUserDialog(request_id="ask-2", question="Proceed?")
    results: list[tuple[str, str]] = []

    app = _DialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        dialog.query_one("#askuser-input", EnhancedTextArea).insert("Yes, continue")
        await pilot.press("enter")
        await pilot.pause()

        assert dialog.query_one("#askuser-input", EnhancedTextArea).text == "Yes, continue\n"
        assert results == []


@pytest.mark.asyncio
async def test_ask_user_dialog_focused_submit_button_enter_submits_custom_response() -> None:
    dialog = AskUserDialog(request_id="ask-7", question="Proceed?")
    results: list[tuple[str, str]] = []

    app = _DialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        dialog.query_one("#askuser-input", EnhancedTextArea).insert("Yes, continue")
        submit = dialog.query_one("#askuser-submit", Button)
        await pilot.pause()
        assert submit.disabled is False
        submit.focus()
        await pilot.press("enter")
        await pilot.pause()

    assert results == [("ask-7", "Yes, continue")]


@pytest.mark.asyncio
async def test_ask_user_dialog_shift_enter_inserts_newline() -> None:
    """Shift+Enter should expand the custom response instead of submitting."""
    dialog = AskUserDialog(request_id="ask-3", question="Explain?")
    results: list[tuple[str, str]] = []

    app = _DialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        input_area = dialog.query_one("#askuser-input", EnhancedTextArea)
        input_area.insert("Line one")
        await pilot.press("shift+enter")
        await pilot.pause()

        assert input_area.text == "Line one\n"
        assert results == []


@pytest.mark.asyncio
async def test_ask_user_dialog_newline_keeps_cursor_visible_after_height_cap() -> None:
    dialog = AskUserDialog(request_id="ask-scroll", question="Explain?")

    app = _DialogHost()
    async with app.run_test(size=(130, 30)) as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        input_area = dialog.query_one("#askuser-input", EnhancedTextArea)
        input_area.focus()
        await pilot.press("1")
        for line in range(2, 7):
            await pilot.press("enter")
            await pilot.press(str(line))
        await pilot.pause()

        assert input_area.document.line_count == 6
        assert input_area.content_size.height == 5

        await pilot.press("enter")
        await pilot.pause()

        cursor_y = input_area.cursor_location[0]
        scroll_y = round(input_area.scroll_y)
        assert scroll_y <= cursor_y < scroll_y + input_area.content_size.height


@pytest.mark.asyncio
async def test_ask_user_dialog_submit_disabled_without_non_whitespace_response() -> None:
    dialog = AskUserDialog(request_id="ask-8", question="Proceed?")
    results: list[tuple[str, str]] = []

    app = _DialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        input_area = dialog.query_one("#askuser-input", EnhancedTextArea)
        submit = dialog.query_one("#askuser-submit", Button)
        assert submit.disabled is True

        input_area.insert("   \n\n")
        await pilot.pause()
        assert submit.disabled is True
        submit.press()
        await pilot.pause()
        assert results == []

        input_area.insert("ok")
        await pilot.pause()
        assert submit.disabled is False

        input_area.clear()
        await pilot.pause()
        assert submit.disabled is True


@pytest.mark.asyncio
async def test_ask_user_dialog_option_labels_are_plain_text() -> None:
    dialog = AskUserDialog(
        request_id="ask-9",
        question="Pick one.",
        options=["[not markup]", "[red]Danger[/red]"],
    )
    results: list[tuple[str, str]] = []

    app = _DialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        first = dialog.query_one("#askuser-opt-0", Button)
        second = dialog.query_one("#askuser-opt-1", Button)
        assert first.label.plain == "[not markup]"
        assert second.label.plain == "[red]Danger[/red]"

        first.press()
        await pilot.pause()

    assert results == [("ask-9", "[not markup]")]


@pytest.mark.asyncio
async def test_ask_user_dialog_ignores_blank_options() -> None:
    dialog = AskUserDialog(request_id="ask-13", question="Pick one.", options=["", "  "])

    app = _DialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        with pytest.raises(NoMatches):
            dialog.query_one("#askuser-options")
        assert dialog.query_one("#askuser-input", EnhancedTextArea).has_focus


@pytest.mark.asyncio
async def test_ask_user_dialog_answer_inline_returns_inline_result_with_draft() -> None:
    dialog = AskUserDialog(request_id="ask-12", question="Need context?")
    results: list[object] = []

    app = _DialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog, callback=results.append)
        await pilot.pause()

        dialog.query_one("#askuser-input", EnhancedTextArea).insert("draft answer")
        await pilot.pause()
        inline = dialog.query_one("#askuser-inline", Button)
        assert inline.variant == "warning"
        assert inline.flat is True
        inline.press()
        await pilot.pause()

    assert results == [AskUserInlineResult("ask-12", "draft answer")]


@pytest.mark.asyncio
async def test_ask_user_dialog_placeholder_is_dim_in_chrys_ansi() -> None:
    dialog = AskUserDialog(request_id="ask-4", question="Proceed?")

    app = _AnsiDialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        input_area = dialog.query_one("#askuser-input", EnhancedTextArea)
        assert input_area.placeholder == "Type a custom response..."
        placeholder_style = input_area.get_visual_style("text-area--placeholder").rich_style
        assert placeholder_style.color is not None
        assert placeholder_style.color.number == 8


@pytest.mark.asyncio
async def test_ask_user_dialog_body_scrolls_question_and_options_together() -> None:
    dialog = AskUserDialog(
        request_id="ask-5",
        question="Review the options and choose the best next step.",
        options=["Use the existing session", "Start a new session"],
    )

    app = _DialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        container = dialog.query_one("#askuser-container", VerticalGroup)
        inner = dialog.query_one("#askuser-inner", StableAutoHeightScroll)
        footer = dialog.query_one("#askuser-footer", VerticalGroup)
        question = dialog.query_one("#askuser-question", VirtualizedMarkdown)
        options = dialog.query_one("#askuser-options", VerticalGroup)
        input_area = dialog.query_one("#askuser-input", EnhancedTextArea)

        assert str(container.styles.width) == "100"
        assert str(container.styles.max_height) == "85h"
        assert str(container.styles.min_height) == "8"
        assert inner.styles.overflow_y == "auto"
        assert inner.styles.padding.left == 1
        assert inner.styles.padding.right == 0
        assert inner.styles.scrollbar_gutter == "stable"
        assert inner.can_focus is False
        assert str(footer.styles.height) == "auto"
        assert options.styles.overflow_y == "hidden"
        assert options.styles.margin.bottom == 0
        assert question.styles.overflow_y == "hidden"
        assert str(input_area.styles.max_height) == "7"
        assert question.parent is inner
        assert options.parent is inner
        assert footer.parent is container
        assert input_area.parent is footer


@pytest.mark.asyncio
async def test_ask_user_dialog_scrollbar_does_not_change_question_width() -> None:
    dialog = AskUserDialog(request_id="ask-stable-scrollbar", question=("word " * 200).strip())

    app = _DialogHost()
    async with app.run_test(size=(80, 40)) as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        inner = dialog.query_one("#askuser-inner", StableAutoHeightScroll)
        question = dialog.query_one("#askuser-question", VirtualizedMarkdown)
        await wait_for(
            lambda: question.size.width > 0 and not inner.show_vertical_scrollbar,
            pilot=pilot,
            description="ask-user body to settle without a scrollbar",
        )
        assert inner.show_vertical_scrollbar is False
        width_without_scrollbar = question.size.width

        await pilot.resize_terminal(80, 30)
        await wait_for(
            lambda: inner.show_vertical_scrollbar,
            pilot=pilot,
            description="ask-user body to show its vertical scrollbar",
        )

        assert inner.show_vertical_scrollbar is True
        assert question.size.width == width_without_scrollbar


@pytest.mark.asyncio
async def test_ask_user_dialog_question_uses_markdown_and_is_left_aligned() -> None:
    dialog = AskUserDialog(request_id="ask-6", question="**What** should happen next?")

    app = _DialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        question = dialog.query_one("#askuser-question", VirtualizedMarkdown)
        assert question.source == "**What** should happen next?"
        assert question.can_focus is False
        assert question.styles.text_align == "left"
        assert question.styles.padding.left == 0


@pytest.mark.asyncio
async def test_ask_user_dialog_markdown_question_expands_to_wrapped_height() -> None:
    wrapped_tail = "择一个方案作为最终态那么过渡期如何最小化对现有 Nginx sidecar 体系的冲击"
    dialog = AskUserDialog(
        request_id="ask-10",
        question=f"## 问题\n\n{wrapped_tail * 8}",
    )

    app = _DialogHost()
    async with app.run_test(size=(80, 30)) as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        question = dialog.query_one("#askuser-question", VirtualizedMarkdown)
        assert question.virtual_size.height > question.source.count("\n") + 1
        assert question.size.height == question.virtual_size.height


@pytest.mark.asyncio
async def test_ask_user_dialog_right_click_copies_selected_question_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = AskUserDialog(request_id="ask-11", question="copy question text")
    copied: list[str] = []
    monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", copied.append)

    app = _DialogHost()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        question = dialog.query_one("#askuser-question", VirtualizedMarkdown)
        dialog.selections = {question: Selection(Offset(0, 0), Offset(len("copy question"), 0))}
        click_x = question.region.x + 1
        click_y = question.region.y

        app.post_message(
            MouseDown(
                question,
                x=click_x,
                y=click_y,
                delta_x=0,
                delta_y=0,
                button=3,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=click_x,
                screen_y=click_y,
            )
        )
        await pilot.pause()
        app.post_message(
            MouseUp(
                question,
                x=click_x,
                y=click_y,
                delta_x=0,
                delta_y=0,
                button=3,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=click_x,
                screen_y=click_y,
            )
        )
        await pilot.pause()

        assert app.clipboard == "copy question"
        assert copied == ["copy question"]
        assert dialog.get_selected_text() is None
