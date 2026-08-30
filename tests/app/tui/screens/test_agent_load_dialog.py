# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the agent loading modal."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets import Static

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.dialogs.agent_load import AgentLoadDialog
from chrys.app.tui.util.rich_style import rich_style_from_textual_color
from chrys.app.tui.widgets.loading import ChrysLoadingIndicator
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.types import (
    AGENT_LOAD_STATUS_DONE,
    AGENT_LOAD_STATUS_FAILED,
    AGENT_LOAD_STATUS_RUNNING,
)


def test_agent_load_default_title_keeps_english_and_localizes_chinese() -> None:
    assert AgentLoadDialog()._title == "Loading Agent"
    controller = LocaleController(Settings(locale="zh-Hans"))
    assert AgentLoadDialog(locale_controller=controller)._title == "正在加载智能体"


def _message(dialog: AgentLoadDialog) -> str:
    content = dialog.query_one("#agent-load-message", Static).content
    if isinstance(content, Text):
        return content.plain
    return str(content)


def _line_text(dialog: AgentLoadDialog) -> list[str]:
    return _message(dialog).splitlines()


@pytest.mark.asyncio
async def test_agent_load_dialog_applies_progress_received_before_mount() -> None:
    dialog = AgentLoadDialog(title="Reloading Agent", subtitle="Code")

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

        def on_mount(self) -> None:
            self.push_screen(dialog)
            dialog.update_progress(
                "Connecting MCP server playground-sleepy-stdio",
                subtitle="Code Agent",
                phase="mcp",
                server_name="playground-sleepy-stdio",
                current=0,
                total=2,
            )

    async with TestApp().run_test() as pilot:
        await pilot.pause()

        assert isinstance(dialog.query_one("#agent-load-loading"), ChrysLoadingIndicator)
        assert _line_text(dialog) == ["▸ Connecting MCP servers: 0/2"]
        assert dialog.query_one("#agent-load-container").border_subtitle == "Code Agent"


@pytest.mark.asyncio
async def test_agent_load_dialog_updates_progress_after_mount() -> None:
    dialog = AgentLoadDialog(title="Reloading Agent", subtitle="Code")

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        dialog.update_progress(
            "Connected MCP server playground-sleepy-http",
            phase="mcp",
            server_name="playground-sleepy-http",
            current=1,
            total=2,
        )
        await pilot.pause()

        assert _line_text(dialog) == ["▸ Connecting MCP servers: 1/2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "ansi-dark", "ansi-light"])
async def test_agent_load_dialog_dims_finished_steps(theme: str) -> None:
    dialog = AgentLoadDialog(title="Initializing Agent", subtitle="Code")

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        app.theme = theme
        await app.push_screen(dialog)
        await pilot.pause()

        dialog.update_progress("Resolving model profile", phase="model")
        dialog.update_progress("Loading built-in tools", phase="tools")
        await pilot.pause()

        content = dialog.query_one("#agent-load-message", Static).content
        assert isinstance(content, Text)
        assert content.plain.splitlines() == [
            "✓ Model profile resolved",
            "▸ Loading built-in tools",
        ]
        assert [entry.status for entry in dialog._progress_entries] == ["done", "active"]
        assert any(span.style == "dim" for span in content.spans)
        # Markers match the todo checklist: green ✓ for done, $warning ▸ for active.
        assert any(span.style == "green" and content.plain[span.start] == "✓" for span in content.spans)
        warning = app.theme_variables.get("warning", "yellow")
        expected_style = rich_style_from_textual_color(warning, bold=True)
        assert any(span.style == expected_style and content.plain[span.start] == "▸" for span in content.spans)


def test_agent_load_dialog_disables_text_selection() -> None:
    # Loading chrome is transient status text; the screen-level flag gates
    # selection for every widget on the dialog (widget/screen/app AND-chain).
    assert AgentLoadDialog.ALLOW_SELECT is False
    assert AgentLoadDialog(title="Initializing Agent").allow_select is False


def test_agent_load_dialog_collapses_progress_into_professional_summary() -> None:
    dialog = AgentLoadDialog(title="Initializing Agent", subtitle="Code")

    dialog.update_progress("Resolving model profile", phase="model")
    dialog.update_progress("Capturing workspace context", phase="runtime")
    dialog.update_progress("Loading built-in tools", phase="tools")
    dialog.update_progress("Built-in tools loaded", phase="tools", current=10, total=10, status=AGENT_LOAD_STATUS_DONE)
    dialog.update_progress("Loading sub-agent tools", phase="sub_agents", current=0, total=3)
    dialog.update_progress("Loading sub-agent Explore", phase="sub_agents", current=0, total=3)
    dialog.update_progress("Loaded sub-agent Explore", phase="sub_agents", current=1, total=3)
    dialog.update_progress("Loading sub-agent Plan", phase="sub_agents", current=1, total=3)
    dialog.update_progress("Loaded sub-agent Plan", phase="sub_agents", current=2, total=3)
    dialog.update_progress("Loading sub-agent General", phase="sub_agents", current=2, total=3)
    dialog.update_progress("Loaded sub-agent General", phase="sub_agents", current=3, total=3)
    dialog.update_progress("Connecting MCP servers", phase="mcp", current=0, total=2)
    dialog.update_progress("Connecting MCP server playground-sleepy-stdio", phase="mcp", current=0, total=2)
    dialog.update_progress("Connecting MCP server playground-sleepy-http", phase="mcp", current=0, total=2)
    dialog.update_progress("Connected MCP server playground-sleepy-http", phase="mcp", current=1, total=2)
    dialog.update_progress("Connected MCP server playground-sleepy-stdio", phase="mcp", current=2, total=2)
    dialog.update_progress("Loading skills", phase="skills")
    dialog.update_progress("Skills loaded", phase="skills", current=4, total=4, status=AGENT_LOAD_STATUS_DONE)
    dialog.update_progress("Finalizing agent", phase="agent")

    assert dialog._messages == [
        "Model profile resolved",
        "Built-in tools loaded: 10/10",
        "Sub-agents loaded: 3/3",
        "MCP servers connected: 2/2",
        "Skills loaded: 4/4",
    ]
    assert [entry.status for entry in dialog._progress_entries] == ["done", "done", "done", "done", "done"]


def test_agent_load_dialog_tracks_session_availability_step() -> None:
    dialog = AgentLoadDialog(title="Restoring Session", subtitle="abc123")

    dialog.update_progress("Checking session availability", phase="session")

    assert dialog._messages == ["Checking session availability"]
    assert [entry.status for entry in dialog._progress_entries] == ["active"]

    dialog.update_progress("Session availability checked", phase="session")

    assert dialog._messages == ["Session availability checked"]
    assert [entry.status for entry in dialog._progress_entries] == ["done"]


def test_agent_load_dialog_shows_zero_completed_sub_agents() -> None:
    dialog = AgentLoadDialog(title="Initializing Agent", subtitle="Code")

    dialog.update_progress("Loading sub-agent tools", phase="sub_agents", current=0, total=1)

    assert dialog._messages == ["Loading sub-agents: 0/1"]
    assert [entry.status for entry in dialog._progress_entries] == ["active"]


def test_agent_load_dialog_does_not_report_failed_mcp_as_connected() -> None:
    dialog = AgentLoadDialog(title="Initializing Agent", subtitle="Code")

    dialog.update_progress("Connecting MCP servers", phase="mcp", current=0, total=2)
    dialog.update_progress("Failed MCP server bad", phase="mcp", current=0, total=2, failed=1)
    dialog.update_progress("Connected MCP server good", phase="mcp", current=1, total=2, failed=1)

    assert dialog._messages == ["MCP servers connected: 1/2, failed: 1"]
    assert [entry.status for entry in dialog._progress_entries] == ["error"]


def test_agent_load_dialog_uses_dash_for_zero_total_counts() -> None:
    dialog = AgentLoadDialog(title="Initializing Agent", subtitle="Code")

    dialog.update_progress("Loading sub-agent tools", phase="sub_agents", current=0, total=0)
    dialog.update_progress("Connecting MCP servers", phase="mcp", current=0, total=0)
    dialog.update_progress("Loading skills", phase="skills")
    dialog.update_progress(
        "Skills loaded",
        phase="skills",
        current=0,
        total=0,
        status=AGENT_LOAD_STATUS_DONE,
    )

    assert dialog._messages == [
        "Sub-agents loaded: -",
        "MCP servers connected: -",
        "Skills loaded: -",
    ]
    assert [entry.status for entry in dialog._progress_entries] == ["done", "done", "done"]


def test_agent_load_dialog_uses_semantic_status_for_unstructured_progress() -> None:
    dialog = AgentLoadDialog(title="Initializing Agent", subtitle="Code")

    dialog.update_progress("transport unavailable", phase="mcp", status=AGENT_LOAD_STATUS_FAILED)
    dialog.update_progress("opaque completion", phase="custom_done", status=AGENT_LOAD_STATUS_DONE)
    dialog.update_progress("opaque failure", phase="custom_failed", status=AGENT_LOAD_STATUS_FAILED)
    dialog.update_progress("opaque activity", phase="custom_active", status=AGENT_LOAD_STATUS_RUNNING)

    assert dialog._messages == [
        "MCP servers failed: -",
        "opaque completion",
        "opaque failure",
        "opaque activity",
    ]
    assert [entry.status for entry in dialog._progress_entries] == ["error", "done", "error", "active"]


def test_agent_load_dialog_tools_and_skills_completion_is_status_driven_not_message_matched() -> None:
    dialog = AgentLoadDialog(title="Initializing Agent", subtitle="Code")

    dialog.update_progress("opaque tools progress", phase="tools", status=AGENT_LOAD_STATUS_RUNNING)
    dialog.update_progress("opaque tools completion", phase="tools", current=3, total=3, status=AGENT_LOAD_STATUS_DONE)
    dialog.update_progress("opaque skills completion", phase="skills", status=AGENT_LOAD_STATUS_DONE)

    assert dialog._messages == [
        "Built-in tools loaded: 3/3",
        "Skills loaded: -",
    ]
    assert [entry.status for entry in dialog._progress_entries] == ["done", "done"]


def test_agent_load_dialog_finish_without_message_marks_pending_finish_done() -> None:
    dialog = AgentLoadDialog(title="Initializing Agent", subtitle="Code")

    dialog.update_finish_progress("Preparing session")
    assert dialog._progress_entries[-1].status == "active"

    dialog.finish()

    assert dialog._messages == ["Preparing session"]
    assert dialog._progress_entries[-1].status == "done"


def test_agent_load_dialog_final_message_moves_finish_line_to_end() -> None:
    dialog = AgentLoadDialog(title="Initializing Agent", subtitle="Code")

    dialog.update_finish_progress("Applying agent changes")
    dialog.update_progress("Late progress", phase="late")
    dialog.finish("Profile switched: QA -> Code")

    assert dialog._messages == ["Late progress", "Profile switched: QA -> Code"]


@pytest.mark.asyncio
async def test_agent_load_dialog_waits_before_success_dismiss(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AgentLoadDialog, "FINISH_HOLD_SECONDS", 0.05)
    dialog = AgentLoadDialog(title="Initializing Agent", subtitle="Code")
    scheduled_timers: list[tuple[float, Callable[[], None]]] = []

    def capture_timer(_dialog: AgentLoadDialog, delay: float, callback: Callable[[], None]) -> object:
        scheduled_timers.append((delay, callback))
        return object()

    monkeypatch.setattr(AgentLoadDialog, "set_timer", capture_timer)

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        dialog.update_progress("Skills loaded", phase="skills", current=1, total=1, status=AGENT_LOAD_STATUS_DONE)
        dialog.finish()
        await pilot.pause()

        # ``finish()`` defers the timer setup via ``call_after_refresh``
        # (see agent_load.py:344). On Windows under xdist load a single
        # ``pilot.pause()`` may return before that refresh callback fires —
        # poll until the timer has been recorded.
        for _ in range(20):
            if scheduled_timers:
                break
            await pilot.pause(0.05)

        assert app.screen is dialog
        assert _line_text(dialog) == ["✓ Skills loaded: 1/1"]
        assert list(dialog.query("#agent-load-loading")) == []
        assert scheduled_timers == [(0.05, dialog._dismiss_after_finish)]

        scheduled_timers[0][1]()
        await pilot.pause()
        assert app.screen is not dialog


@pytest.mark.asyncio
async def test_agent_load_dialog_failure_body_has_bold_title() -> None:
    dialog = AgentLoadDialog(title="Initializing Agent", subtitle="Code")

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        message = (
            "Environment variable 'ABCD' is required by model profile 'DeepSeek-V4-Pro' "
            "extra header['test'], but it is not set."
        )
        dialog.set_result(False, message, allow_esc=True)
        await pilot.pause()

        title = dialog.query_one("#agent-load-result-title", Static)
        assert title.display is True
        assert title.styles.text_align == "center"

        title_content = title.content
        assert isinstance(title_content, Text)
        assert title_content.plain == "Unable to Load Agent"
        assert title_content.style == "bold"
        assert str(dialog.query_one("#agent-load-container").border_title) == "Initializing Agent"

        message_content = dialog.query_one("#agent-load-message", Static).content
        assert isinstance(message_content, Text)
        assert message_content.plain == message
        assert message_content.style == ""
        assert message_content.spans == []


@pytest.mark.asyncio
async def test_agent_load_dialog_success_hides_body_title() -> None:
    dialog = AgentLoadDialog(title="Initializing Agent", subtitle="Code")

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield Static("placeholder")

    app = TestApp()
    async with app.run_test() as pilot:
        await app.push_screen(dialog)
        await pilot.pause()

        dialog.set_result(True, "Agent ready.")
        await pilot.pause()

        title = dialog.query_one("#agent-load-result-title", Static)
        assert title.display is False
        assert str(title.content) == ""
        assert str(dialog.query_one("#agent-load-container").border_title) == "Agent Loaded"
