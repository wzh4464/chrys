# Copyright (c) 2026 Chrys. All rights reserved.

"""Golden checks for live-vs-replayed tool card rendering contracts."""

from __future__ import annotations

import re
from typing import Any

import pytest
from textual.app import App, ComposeResult

from chrys.app.tui.widgets.chat.panel import ChatPanel
from chrys.app.tui.widgets.chat.tool_call import ToolGroup
from chrys.foundation.tool_kinds import (
    KIND_ASK_USER,
    KIND_FILESYSTEM_READ,
    KIND_FILESYSTEM_WRITE,
    KIND_SEARCH,
    KIND_SHELL,
    KIND_SLEEP,
    KIND_SUB_AGENT,
)
from chrys.kernel import Content

_CALL_ID_RE = re.compile(r"^- \*\*Call ID:\*\* `[^`]+`$", re.MULTILINE)
_TINY_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"


class _ChatPanelApp(App):
    def compose(self) -> ComposeResult:
        yield ChatPanel()


class _ToolGroupApp(App):
    def compose(self) -> ComposeResult:
        yield ToolGroup()


def _normalized_copy_payload(tool: object) -> str:
    payload = tool.format_tool_execution_copy()
    return _CALL_ID_RE.sub("- **Call ID:** `<call-id>`", payload)


def _tool_golden(tool: object) -> dict[str, object]:
    return {
        "class": tool.__class__.__name__,
        "status": tool.status,
        "approval": tool.approval,
        "duration_ms": tool.duration_ms,
        "args": tool.args,
        "image_count": len(tool.image_contents),
        "copy_payload": _normalized_copy_payload(tool),
    }


async def _mounted_replay_tool(pilot: object, panel: ChatPanel) -> object:
    group = panel.query_one(ToolGroup)
    group.collapsed = False
    for _ in range(300):
        await pilot.pause()
        if group._content_mounted and group._tools:
            break
    assert group._content_mounted is True
    assert len(group._tools) == 1
    return next(iter(group._tools.values()))


async def _mounted_group_tool(pilot: object, group: ToolGroup) -> object:
    group.collapsed = False
    for _ in range(300):
        await pilot.pause()
        if group._content_mounted and group._tools:
            break
    assert group._content_mounted is True
    assert len(group._tools) == 1
    return next(iter(group._tools.values()))


def _replay_messages(
    *,
    tool_name: str,
    args: dict[str, Any],
    result: str,
    approval: str | None = None,
) -> list[dict[str, Any]]:
    function_call: dict[str, Any] = {
        "type": "function_call",
        "name": tool_name,
        "call_id": "call1",
        "arguments": args,
    }
    if approval is not None:
        function_call["additional_properties"] = {
            "_approval": {"status": approval, "tool_name": tool_name},
        }
    return [
        {"role": "user", "contents": [{"type": "text", "text": f"run {tool_name}"}]},
        {"role": "assistant", "contents": [function_call]},
        {
            "role": "tool",
            "contents": [{"type": "function_result", "call_id": "call1", "result": result}],
        },
    ]


@pytest.mark.parametrize(
    ("tool_name", "tool_kind", "args", "result"),
    [
        ("ask_user", KIND_ASK_USER, {"question": "Pick?", "options": ["Python"]}, "User response: Python"),
        ("glob", KIND_SEARCH, {"pattern": "*.py"}, "Found 4 files matching '*.py' in /repo\nsrc/a.py\nsrc/b.py"),
        ("grep", KIND_SEARCH, {"pattern": "needle"}, "Found 1 match in /repo\nsrc/a.py:1:needle"),
        ("load_skill", "", {"skill_name": "docs"}, "<instructions>\nUse docs.\n</instructions>"),
        ("read_file", KIND_FILESYSTEM_READ, {"path": "README.md"}, "File: README.md\n1: hello"),
        ("sleep", KIND_SLEEP, {"seconds": 1, "reason": "wait for server"}, "Slept for 1 second."),
        ("zsh", KIND_SHELL, {"command": "printf 'hi\\n'"}, "hi\n[exit_code: 0]"),
    ],
)
@pytest.mark.asyncio
async def test_live_and_replayed_simple_tool_cards_match_golden(
    tool_name: str,
    tool_kind: str,
    args: dict[str, Any],
    result: str,
) -> None:
    async with _ChatPanelApp().run_test() as pilot:
        live = pilot.app.query_one(ChatPanel)
        live.set_tool_kinds({tool_name: tool_kind})

        await live.add_tool_start("call1", tool_name, tool_kind, args=args)
        await live.add_tool_result("call1", tool_name, result)
        await pilot.pause()

        live_tool = live.query_one(ToolGroup).get_tool("call1")
        live_golden = _tool_golden(live_tool)

    async with _ChatPanelApp().run_test() as pilot:
        replay = pilot.app.query_one(ChatPanel)
        replay.set_tool_kinds({tool_name: tool_kind})

        await replay.replay_history(_replay_messages(tool_name=tool_name, args=args, result=result))
        await pilot.pause()

        replay_tool = await _mounted_replay_tool(pilot, replay)
        assert _tool_golden(replay_tool) == live_golden


@pytest.mark.asyncio
async def test_live_and_replayed_file_edit_card_match_golden() -> None:
    args = {"path": "README.md"}
    result = "Edited README.md"
    snapshot = ("old\n", "new\n")

    async with _ChatPanelApp().run_test() as pilot:
        live = pilot.app.query_one(ChatPanel)
        live.set_tool_kinds({"edit_file": KIND_FILESYSTEM_WRITE})

        await live.add_tool_start("call1", "edit_file", KIND_FILESYSTEM_WRITE, args=args)
        await live.add_tool_result("call1", "edit_file", result, file_snapshot=snapshot)
        await pilot.pause()

        live_tool = live.query_one(ToolGroup).get_tool("call1")
        live_golden = _tool_golden(live_tool)

    async with _ChatPanelApp().run_test() as pilot:
        replay = pilot.app.query_one(ChatPanel)
        replay.set_tool_kinds({"edit_file": KIND_FILESYSTEM_WRITE})

        await replay.replay_history(
            _replay_messages(tool_name="edit_file", args=args, result=result),
            file_snapshots={"call1": [snapshot]},
        )
        await pilot.pause()

        replay_tool = await _mounted_replay_tool(pilot, replay)
        assert _tool_golden(replay_tool) == live_golden


@pytest.mark.asyncio
async def test_live_and_replayed_write_file_card_match_golden() -> None:
    args = {"path": "README.md", "overwrite": True}
    result = "Successfully wrote README.md (2 lines)"
    snapshot = ("", "hello\nworld\n")

    async with _ChatPanelApp().run_test() as pilot:
        live = pilot.app.query_one(ChatPanel)
        live.set_tool_kinds({"write_file": KIND_FILESYSTEM_WRITE})

        await live.add_tool_start("call1", "write_file", KIND_FILESYSTEM_WRITE, args=args)
        await live.add_tool_result("call1", "write_file", result, file_snapshot=snapshot)
        await pilot.pause()

        live_tool = live.query_one(ToolGroup).get_tool("call1")
        live_golden = _tool_golden(live_tool)

    async with _ChatPanelApp().run_test() as pilot:
        replay = pilot.app.query_one(ChatPanel)
        replay.set_tool_kinds({"write_file": KIND_FILESYSTEM_WRITE})

        await replay.replay_history(
            _replay_messages(tool_name="write_file", args=args, result=result),
            file_snapshots={"call1": [snapshot]},
        )
        await pilot.pause()

        replay_tool = await _mounted_replay_tool(pilot, replay)
        assert _tool_golden(replay_tool) == live_golden


@pytest.mark.asyncio
async def test_live_and_replayed_image_result_tool_card_match_golden() -> None:
    args = {"path": "plot.png"}
    result = "Image: plot\n[image/png image]"
    image = {
        "type": "data",
        "uri": f"data:image/png;base64,{_TINY_PNG}",
        "media_type": "image/png",
        "additional_properties": {"width": 1, "height": 1, "media_type": "image/png"},
    }

    async with _ChatPanelApp().run_test() as pilot:
        live = pilot.app.query_one(ChatPanel)

        await live.add_tool_start("call1", "view_image", "", args=args)
        await live.add_tool_result("call1", "view_image", result, image_contents=[image])
        await pilot.pause()

        live_tool = live.query_one(ToolGroup).get_tool("call1")
        live_golden = _tool_golden(live_tool)

    async with _ChatPanelApp().run_test() as pilot:
        replay = pilot.app.query_one(ChatPanel)

        await replay.replay_history(
            [
                {"role": "user", "contents": [{"type": "text", "text": "show image"}]},
                {
                    "role": "assistant",
                    "contents": [
                        {
                            "type": "function_call",
                            "name": "view_image",
                            "call_id": "call1",
                            "arguments": args,
                        }
                    ],
                },
                {
                    "role": "tool",
                    "contents": [
                        {
                            "type": "function_result",
                            "call_id": "call1",
                            "result": result,
                            "items": [image],
                        }
                    ],
                },
            ]
        )
        await pilot.pause()

        replay_tool = await _mounted_replay_tool(pilot, replay)
        assert _tool_golden(replay_tool) == live_golden


@pytest.mark.asyncio
async def test_live_and_restored_hosted_tool_cards_match_semantic_golden() -> None:
    args = {"query": "Chrys"}
    result = "found"

    async with _ChatPanelApp().run_test() as pilot:
        live = pilot.app.query_one(ChatPanel)
        await live.add_tool_start(
            "hosted:1",
            "web_search",
            KIND_SEARCH,
            args=args,
            provider_hosted=True,
            hosted_family="search",
            provider="openai",
            provider_item_type="web_search_call",
            provider_status="searching",
            provider_call_id="provider-search",
        )
        await live.add_tool_result(
            "hosted:1",
            "web_search",
            result,
            provider_status="completed",
            canonical_status="completed",
        )
        await pilot.pause()

        live_tool = live.query_one(ToolGroup).get_tool("hosted:1")
        live_golden = _tool_golden(live_tool)

    call = Content.from_search_tool_call(
        "provider-search",
        tool_name="web_search",
        arguments=args,
        status="running",
        hosted_provider="openai",
        provider_item_type="web_search_call",
        provider_phase="start",
        provider_status="searching",
    )
    paired_result = Content.from_search_tool_result(
        "provider-search",
        tool_name="web_search",
        result=result,
        status="completed",
        hosted_provider="openai",
        provider_item_type="web_search_call",
        provider_phase="terminal",
        provider_status="completed",
    )
    async with _ChatPanelApp().run_test() as pilot:
        replay = pilot.app.query_one(ChatPanel)
        await replay.replay_history([{"role": "assistant", "contents": [call.to_dict(), paired_result.to_dict()]}])
        await pilot.pause()

        replay_tool = await _mounted_replay_tool(pilot, replay)
        assert _tool_golden(replay_tool) == live_golden


@pytest.mark.parametrize(
    ("family", "tool_name", "tool_kind", "args", "result", "metadata", "artifacts", "image_contents"),
    [
        (
            "search",
            "web_search",
            KIND_SEARCH,
            {"query": "Chrys"},
            '{"results":[{"title":"Docs","url":"https://example.test"}]}',
            {},
            [],
            [],
        ),
        (
            "mcp",
            "lookup",
            "mcp",
            {"server": "docs", "query": "Chrys"},
            '{"answer":"found"}',
            {},
            [],
            [],
        ),
        (
            "code",
            "code_interpreter",
            "",
            {"language": "python", "code": "print('hi')"},
            "",
            {"stdout": "hi", "stderr": ""},
            [{"path": "report.csv", "mime": "text/csv"}],
            [],
        ),
        (
            "image",
            "image_generation",
            "",
            {"prompt": "a chrysanthemum"},
            "created",
            {"quality": "high"},
            [],
            [
                {
                    "type": "data",
                    "uri": f"data:image/png;base64,{_TINY_PNG}",
                    "media_type": "image/png",
                }
            ],
        ),
        (
            "shell",
            "bash",
            KIND_SHELL,
            {"commands": ["printf hi"]},
            "",
            {"stdout": "hi", "stderr": "", "exit_code": 0, "timed_out": False},
            [{"path": "output.txt", "size": 2}],
            [],
        ),
    ],
)
@pytest.mark.asyncio
async def test_live_and_lazy_replayed_hosted_family_cards_match_golden(
    family: str,
    tool_name: str,
    tool_kind: str,
    args: dict[str, Any],
    result: str,
    metadata: dict[str, Any],
    artifacts: list[dict[str, Any]],
    image_contents: list[dict[str, Any]],
) -> None:
    async with _ToolGroupApp().run_test() as pilot:
        live_group = pilot.app.query_one(ToolGroup)
        await live_group.add_tool(
            "hosted:1",
            tool_name,
            tool_kind,
            args=args,
            provider_hosted=True,
            hosted_family=family,
            provider="openai",
            provider_status="running",
        )
        live_group.complete_tool(
            "hosted:1",
            result,
            metadata=metadata,
            artifacts=artifacts,
            image_contents=image_contents,
            provider_status="completed",
            canonical_status="completed",
        )
        await pilot.pause()

        live_golden = _tool_golden(live_group.get_tool("hosted:1"))

    async with _ToolGroupApp().run_test() as pilot:
        replay_group = pilot.app.query_one(ToolGroup)
        await replay_group.add_collapsed_replay_tool(
            "hosted:1",
            tool_name,
            tool_kind,
            args=args,
            result=result,
            metadata=metadata,
            artifacts=artifacts,
            image_contents=image_contents,
            provider_hosted=True,
            hosted_family=family,
            provider="openai",
            provider_status="completed",
            canonical_status="completed",
            lazy=True,
        )

        replay_tool = await _mounted_group_tool(pilot, replay_group)
        assert _tool_golden(replay_tool) == live_golden


@pytest.mark.asyncio
async def test_live_and_replayed_successful_sub_agent_card_match_golden() -> None:
    args = {"prompt": "Inspect the workspace"}
    result = "Found the issue and updated the notes."

    async with _ChatPanelApp().run_test() as pilot:
        live = pilot.app.query_one(ChatPanel)
        live.set_tool_kinds({"explore_agent": KIND_SUB_AGENT})

        await live.add_tool_start("call1", "explore_agent", KIND_SUB_AGENT, args=args)
        await live.add_tool_result("call1", "explore_agent", result)
        await pilot.pause()

        live_tool = live.query_one(ToolGroup).get_tool("call1")
        live_golden = _tool_golden(live_tool)

    async with _ChatPanelApp().run_test() as pilot:
        replay = pilot.app.query_one(ChatPanel)
        replay.set_tool_kinds({"explore_agent": KIND_SUB_AGENT})

        await replay.replay_history(_replay_messages(tool_name="explore_agent", args=args, result=result))
        await pilot.pause()

        replay_tool = await _mounted_replay_tool(pilot, replay)
        assert _tool_golden(replay_tool) == live_golden


@pytest.mark.asyncio
async def test_live_and_replayed_rejected_sub_agent_card_match_golden() -> None:
    args = {"prompt": "Inspect the workspace"}
    result = "Error: Tool execution was rejected by user."

    async with _ChatPanelApp().run_test() as pilot:
        live = pilot.app.query_one(ChatPanel)
        live.set_tool_kinds({"explore_agent": KIND_SUB_AGENT})

        await live.add_tool_start("call1", "explore_agent", KIND_SUB_AGENT, args=args)
        await live.add_tool_result(
            "call1",
            "explore_agent",
            result,
            approval="user_rejected",
        )
        await pilot.pause()

        live_tool = live.query_one(ToolGroup).get_tool("call1")
        live_golden = _tool_golden(live_tool)

    async with _ChatPanelApp().run_test() as pilot:
        replay = pilot.app.query_one(ChatPanel)
        replay.set_tool_kinds({"explore_agent": KIND_SUB_AGENT})

        await replay.replay_history(
            _replay_messages(
                tool_name="explore_agent",
                args=args,
                result=result,
                approval="user_rejected",
            )
        )
        await pilot.pause()

        replay_tool = await _mounted_replay_tool(pilot, replay)
        assert _tool_golden(replay_tool) == live_golden
