# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the sleep tool renderer."""

from __future__ import annotations

import json

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.chat.renderers.sleep import SleepSkipClicked, SleepToolCall
from chrys.app.tui.widgets.chat.renderers.sub_agent import SubAgentToolCall
from chrys.app.tui.widgets.chat.tool_call import BaseToolCard, ToolGroup
from chrys.foundation.tool_kinds import KIND_SLEEP, KIND_SUB_AGENT
from chrys.foundation.tool_result_metadata import TOOL_FAILED_METADATA_KEY, TOOL_INTERRUPTED_METADATA_KEY
from chrys.service.session.message_metadata import TOOL_RESULT_METADATA_KEY


def test_sleep_renderer_uses_base_tool_card_contract() -> None:
    tool = SleepToolCall(
        "c1",
        "sleep",
        args_summary=json.dumps({"seconds": 3, "reason": "wait"}),
    )

    assert isinstance(tool, BaseToolCard)
    assert tool.call_id == "c1"
    assert tool.tool_name == "sleep"
    assert tool.status == "running"
    assert tool.result_text == ""
    assert tool.duration_ms == 0
    assert tool.args == {"seconds": 3, "reason": "wait"}


def test_sleep_renderer_records_completion_approval() -> None:
    tool = SleepToolCall("c1", "sleep", args={"seconds": 3})

    tool.set_complete("Rejected by user", approval="user_rejected")

    assert tool.approval == "user_rejected"


def test_sleep_renderer_rejected_status_survives_unmounted_query_failure() -> None:
    tool = SleepToolCall("c1", "sleep", args={"seconds": 3})

    tool.set_complete("Rejected by user", approval="user_rejected")

    assert tool.status == "rejected"
    assert tool.has_class("-rejected")


def test_sleep_renderer_error_status_survives_unmounted_query_failure() -> None:
    tool = SleepToolCall("c1", "sleep", args={"seconds": 3})

    tool.set_complete("Error: invalid duration", metadata={TOOL_FAILED_METADATA_KEY: True})

    assert tool.status == "error"
    assert tool.has_class("-error")


def test_sleep_renderer_reads_args_summary_json() -> None:
    tool = SleepToolCall(
        "c1",
        "sleep",
        args_summary=json.dumps({"seconds": 12, "reason": "wait for server boot"}),
    )

    assert tool._seconds == 12
    assert tool._reason == "wait for server boot"


@pytest.mark.parametrize(
    ("metadata", "subtitle"),
    [
        ({"sleep_skipped": True}, "Skipped"),
        ({"sleep_interrupted": True}, "Interrupted"),
    ],
)
@pytest.mark.asyncio
async def test_sleep_renderer_uses_metadata_for_terminal_status(metadata: dict[str, bool], subtitle: str) -> None:
    class ToolApp(App):
        def compose(self) -> ComposeResult:
            yield SleepToolCall("c1", "sleep", args={"seconds": 5})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SleepToolCall)

        tool.set_complete("done by external signal", duration_ms=10, metadata=metadata)
        await pilot.pause()

        assert tool.query_one("#sleep-panel").border_subtitle == subtitle


@pytest.mark.asyncio
async def test_sleep_renderer_places_countdown_next_to_skip_button() -> None:
    class ToolApp(App):
        def compose(self) -> ComposeResult:
            yield SleepToolCall("c1", "sleep", args={"seconds": 5, "reason": "wait"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SleepToolCall)
        countdown = tool.query_one("#sleep-countdown", Static)

        assert countdown.parent is tool.query_one("#sleep-actions")


@pytest.mark.asyncio
async def test_sub_agent_sleep_skip_button_posts_sleep_skip_message() -> None:
    class ToolApp(App):
        def __init__(self) -> None:
            super().__init__()
            self.skipped: list[str] = []

        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("parent", "Explore", args={"prompt": "watch the service"})

        def on_sleep_skip_clicked(self, event: SleepSkipClicked) -> None:
            self.skipped.append(event.call_id)
            event.stop()

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)

        await tool.add_inner_tool_start("sleep1", "pause", {"seconds": 30}, tool_kind=KIND_SLEEP)
        await pilot.pause()
        assert tool.has_class("-sleeping")

        tool.query_one("#sa-skip-sleep-btn", Button).press()
        await pilot.pause()

        assert pilot.app.skipped == ["sleep1"]

        tool.complete_inner_tool(
            "sleep1", "Sleep skipped by user after 0 seconds.", 0, metadata={"sleep_skipped": True}
        )
        await pilot.pause()
        assert not tool.has_class("-sleeping")
        assert "pause skipped" in tool.query_one("#sa-main", Static).render().plain


@pytest.mark.asyncio
async def test_sub_agent_inner_sleep_interrupted_row_uses_metadata() -> None:
    class ToolApp(App):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("parent", "Explore", args={"prompt": "watch the service"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)

        await tool.add_inner_tool_start("sleep1", "sleep", {"seconds": 30}, tool_kind=KIND_SLEEP)
        tool.complete_inner_tool(
            "sleep1",
            "Sleep interrupted after 0 seconds.",
            0,
            metadata={"sleep_interrupted": True},
        )
        await pilot.pause()

        assert "sleep interrupted" in tool.query_one("#sa-main", Static).render().plain


@pytest.mark.parametrize(
    ("metadata", "subtitle"),
    [
        ({"sleep_skipped": True}, "Skipped"),
        ({"sleep_interrupted": True}, "Interrupted"),
    ],
)
@pytest.mark.asyncio
async def test_sleep_replay_recovers_terminal_status_from_result_metadata(
    metadata: dict[str, bool],
    subtitle: str,
) -> None:
    class PanelApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    messages = [
        {"role": "user", "contents": [{"type": "text", "text": "wait"}]},
        {
            "role": "assistant",
            "contents": [
                {
                    "type": "function_call",
                    "name": "sleep",
                    "call_id": "sleep1",
                    "arguments": {"seconds": 30},
                }
            ],
        },
        {
            "role": "tool",
            "contents": [
                {
                    "type": "function_result",
                    "call_id": "sleep1",
                    "result": "Sleep resolved with a reworded result.",
                    "additional_properties": {TOOL_RESULT_METADATA_KEY: metadata},
                }
            ],
        },
    ]

    async with PanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        panel.set_tool_kinds({"sleep": KIND_SLEEP})
        await panel.replay_history(messages)
        await pilot.pause()

        group = panel.query_one(ToolGroup)
        assert group._content_mounted is False
        group.collapsed = False
        await pilot.pause()

        tool = panel.query_one(SleepToolCall)
        assert tool.query_one("#sleep-panel").border_subtitle == subtitle


@pytest.mark.asyncio
async def test_sub_agent_replay_renders_interrupted_parent_result_card() -> None:
    class PanelApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    messages = [
        {"role": "user", "contents": [{"type": "text", "text": "delegate"}]},
        {
            "role": "assistant",
            "contents": [
                {
                    "type": "function_call",
                    "name": "explore",
                    "call_id": "parent-call",
                    "arguments": {"prompt": "inspect"},
                }
            ],
        },
        {
            "role": "tool",
            "contents": [
                {
                    "type": "function_result",
                    "call_id": "parent-call",
                    "result": "Error: Tool execution was interrupted.",
                    "additional_properties": {
                        TOOL_RESULT_METADATA_KEY: {
                            TOOL_INTERRUPTED_METADATA_KEY: True,
                            "sub_agent_invocation_id": "child-invocation",
                            "sub_agent_log_file": "child.json",
                        }
                    },
                }
            ],
        },
    ]

    async with PanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        panel.set_tool_kinds({"explore": KIND_SUB_AGENT})
        await panel.replay_history(messages)
        await pilot.pause()

        group = panel.query_one(ToolGroup)
        group.collapsed = False
        await pilot.pause()

        card = panel.query_one(SubAgentToolCall)
        assert card.status == "interrupted"
        assert card.query_one("#sa-panel").border_subtitle == "Interrupted"
