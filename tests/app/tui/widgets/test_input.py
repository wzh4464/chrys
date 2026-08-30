# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the reusable EnhancedInput widget."""

from __future__ import annotations

import pytest
from textual.actions import SkipAction
from textual.app import App, ComposeResult

from chrys.app.tui.widgets.input import EnhancedInput


class _Harness(App[None]):
    def compose(self) -> ComposeResult:
        yield EnhancedInput("hello world", id="input")


@pytest.mark.asyncio
async def test_ctrl_a_selects_all_and_delete_clears_value() -> None:
    app = _Harness()
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input", EnhancedInput)
        input_widget.focus()
        await pilot.pause()

        await pilot.press("ctrl+a")
        await pilot.pause()

        assert input_widget.selected_text == "hello world"

        await pilot.press("backspace")
        await pilot.pause()

        assert input_widget.value == ""


@pytest.mark.asyncio
async def test_regular_keys_still_use_textual_input_handling() -> None:
    app = _Harness()
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input", EnhancedInput)
        input_widget.value = ""
        input_widget.focus()
        await pilot.pause()

        await pilot.press("x")
        await pilot.pause()

        assert input_widget.value == "x"


@pytest.mark.asyncio
async def test_ctrl_c_synchronizes_app_and_os_clipboards(monkeypatch: pytest.MonkeyPatch) -> None:
    os_clipboard: list[str] = []
    monkeypatch.setattr("chrys.app.tui.clipboard.platform_helpers.clipboard_copy", os_clipboard.append)

    app = _Harness()
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input", EnhancedInput)
        input_widget.focus()
        input_widget.select_all()
        await pilot.pause()

        await pilot.press("ctrl+c")
        await pilot.pause()

        assert input_widget.value == "hello world"
        assert app.clipboard == "hello world"
        assert os_clipboard == ["hello world"]


@pytest.mark.asyncio
async def test_ctrl_x_then_ctrl_v_round_trips_through_os_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    os_clipboard = {"text": "older-os-text"}
    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    monkeypatch.setattr(
        "chrys.app.tui.clipboard.platform_helpers.clipboard_copy",
        lambda text: os_clipboard.__setitem__("text", text),
    )
    monkeypatch.setattr(
        "chrys.app.tui.clipboard.platform_helpers.clipboard_paste",
        lambda: os_clipboard["text"],
    )

    app = _Harness()
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input", EnhancedInput)
        input_widget.focus()
        input_widget.select_all()
        await pilot.pause()

        await pilot.press("ctrl+x")
        await pilot.pause()

        assert input_widget.value == ""
        assert os_clipboard["text"] == "hello world"

        app.copy_to_clipboard("stale-app-text")
        await pilot.press("ctrl+v")
        await pilot.pause()

        assert input_widget.value == "hello world"


@pytest.mark.asyncio
async def test_copy_without_selection_preserves_skip_action() -> None:
    app = _Harness()
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input", EnhancedInput)
        input_widget.focus()
        await pilot.pause()
        input_widget.cursor_position = len(input_widget.value)

        with pytest.raises(SkipAction):
            input_widget.action_copy()


@pytest.mark.asyncio
async def test_ctrl_v_prefers_current_os_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    monkeypatch.setattr("chrys.app.tui.clipboard.platform_helpers.clipboard_paste", lambda: "fresh-system-key")

    app = _Harness()
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input", EnhancedInput)
        app.copy_to_clipboard("stale-app-clipboard")
        input_widget.focus()
        input_widget.select_all()
        await pilot.pause()

        await pilot.press("ctrl+v")
        await pilot.pause()

        assert input_widget.value == "fresh-system-key"


@pytest.mark.asyncio
async def test_ctrl_v_falls_back_to_app_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    monkeypatch.setattr("chrys.app.tui.clipboard.platform_helpers.clipboard_paste", lambda: "")

    app = _Harness()
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input", EnhancedInput)
        app.copy_to_clipboard("fallback-key")
        input_widget.focus()
        input_widget.select_all()
        await pilot.pause()

        await pilot.press("ctrl+v")
        await pilot.pause()

        assert input_widget.value == "fallback-key"


@pytest.mark.asyncio
async def test_ctrl_v_keeps_single_line_input_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    monkeypatch.setattr(
        "chrys.app.tui.clipboard.platform_helpers.clipboard_paste",
        lambda: "first-line\nignored-line",
    )

    app = _Harness()
    async with app.run_test() as pilot:
        input_widget = app.query_one("#input", EnhancedInput)
        input_widget.focus()
        input_widget.select_all()
        await pilot.pause()

        await pilot.press("ctrl+v")
        await pilot.pause()

        assert input_widget.value == "first-line"
