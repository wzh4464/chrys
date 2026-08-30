# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for TUI widgets — standalone widget behavior without full app."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from io import BytesIO, StringIO
from typing import ClassVar

import pytest
from PIL import Image
from rich.color import ColorSystem
from rich.console import Console
from rich.text import Text
from textual import events as textual_events
from textual import on
from textual.app import App, ComposeResult
from textual.await_complete import AwaitComplete
from textual.color import Color as TextualColor
from textual.events import Click, Resize
from textual.geometry import Offset, Region, Size
from textual.message import Message
from textual.scrollbar import ScrollTo
from textual.selection import SELECT_ALL, Selection
from textual.widget import Widget
from textual.widgets import Button, RichLog, Static, TextArea, Tree

from chrys.app.tui import i18n as tui_i18n
from chrys.app.tui.i18n import LocaleController, LocaleSwitchStatus
from chrys.app.tui.support.gc_freeze import (
    GcAbsorbReason,
    GcAbsorbRequested,
    GcReclaimReason,
    GcReclaimRequested,
)
from chrys.app.tui.theme import CHRYS_DARK_THEME, CHRYS_DARK_THEME_NAME, TuiVariableDefaultsMixin
from chrys.app.tui.widgets.chat import image_preview as image_preview_module
from chrys.app.tui.widgets.chat import panel as chat_panel_module
from chrys.app.tui.widgets.chat.compaction_card import CompactionCard
from chrys.app.tui.widgets.chat.context_fold import ContextFoldWidget
from chrys.app.tui.widgets.chat.file_snapshot import FileSnapshotRef
from chrys.app.tui.widgets.chat.image_preview import ImagePreviewGrid, extract_image_previews
from chrys.app.tui.widgets.chat.messages import (
    AgentCopyButton,
    AgentMessage,
    ConversationStatusAction,
    ErrorMessage,
    InterruptedMessage,
    RetryMessage,
    SystemMessage,
    UserMessage,
    _UserImagePreview,
    _UserMessageText,
    format_message_created_at,
)
from chrys.app.tui.widgets.chat.panel import ChatPanel, _ChatBottomSpacer, _ScrollToBottomButton
from chrys.app.tui.widgets.chat.ports import TranscriptLocalizationPort
from chrys.app.tui.widgets.chat.renderers.ask_user import AskUserToolCall
from chrys.app.tui.widgets.chat.renderers.sub_agent import SubAgentToolCall
from chrys.app.tui.widgets.chat.session_json import SessionJsonPanel
from chrys.app.tui.widgets.chat.tool_call import (
    BaseToolCard,
    ToolCall,
    ToolCardHeader,
    ToolCopyButton,
    ToolGroup,
    ToolGroupTitle,
    ToolViewButton,
)
from chrys.app.tui.widgets.chat.tool_renderers import register_kind_renderer
from chrys.app.tui.widgets.chat.welcome import WelcomeWidget
from chrys.app.tui.widgets.chrome.app_header import AppHeader
from chrys.app.tui.widgets.chrome.input_bar import InputBar, _ChatTextArea
from chrys.app.tui.widgets.chrome.status_bar import (
    STATUS_COMPLETED,
    STATUS_INTERRUPTED,
    STATUS_SESSION_RESTORED,
    STATUS_THINKING,
    StatusBar,
)
from chrys.app.tui.widgets.click_affordance import ClickAffordance
from chrys.app.tui.widgets.hatch import HATCH_GLYPH, HatchedEmptyState
from chrys.app.tui.widgets.loading import ChrysLoadingIndicator
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.app.tui.widgets.sidebar.context import ContextPanel, ContextUsageState, _CompressedBlock
from chrys.app.tui.widgets.sidebar.debug import DebugPanel, _ClickableLog
from chrys.app.tui.widgets.sidebar.panel import SidebarPanel
from chrys.app.tui.widgets.sidebar.toc import ConversationToc, TocItem, summarize_prompt
from chrys.foundation.branding import format_app_version_title
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.types import AgentRuntimeDetails, RuntimeHookDetails, RuntimeHookSourceDetails
from chrys.foundation.i18n import Localizer, MessageRef
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.patches.textual_tab_selection import apply_runtime_patch
from chrys.foundation.tool_kinds import (
    KIND_ASK_USER,
    KIND_FILESYSTEM_READ,
    KIND_MCP,
    KIND_SEARCH,
    KIND_SHELL,
    KIND_SUB_AGENT,
)
from chrys.foundation.tool_result_metadata import (
    SHELL_EXIT_CODE_METADATA_KEY,
    SHELL_TIMED_OUT_METADATA_KEY,
    TOOL_ERROR_KIND_METADATA_KEY,
    TOOL_FAILED_METADATA_KEY,
)
from chrys.foundation.trajectory_timing import TRAJECTORY_TIMING_KEY
from chrys.kernel import Content
from chrys.service.approval.policy import ApprovalMode
from chrys.service.mutations.store import SnapshotStore
from chrys.service.session.message_metadata import MESSAGE_CREATED_AT_KEY
from tests.support.waiting import wait_for

# ---------------------------------------------------------------------------
# Test apps — minimal harnesses for individual widgets
# ---------------------------------------------------------------------------


class _LocalizedApp(App):
    locale_controller = LocaleController(Settings(locale="en"))


class ChatPanelApp(_LocalizedApp):
    def __init__(self, localization: TranscriptLocalizationPort | None = None) -> None:
        self._localization = localization
        super().__init__()

    def compose(self) -> ComposeResult:
        yield ChatPanel(localization=self._localization)


class GcMessageChatPanelApp(ChatPanelApp):
    def __init__(self) -> None:
        self.gc_messages: list[GcAbsorbRequested | GcReclaimRequested] = []
        super().__init__()

    def on_gc_absorb_requested(self, message: GcAbsorbRequested) -> None:
        self.gc_messages.append(message)

    def on_gc_reclaim_requested(self, message: GcReclaimRequested) -> None:
        self.gc_messages.append(message)


def _chat_content_children(panel: ChatPanel) -> list[object]:
    """Return transcript children, excluding persistent chat-panel infrastructure."""
    return [child for child in panel.children if not isinstance(child, (_ChatBottomSpacer, _ScrollToBottomButton))]


def _install_fake_chat_panel_gc(monkeypatch: pytest.MonkeyPatch):
    """Replace chat-panel GC hooks with a small stateful fake for scroll tests."""
    import chrys.app.tui.widgets.chat.scroll_controller as scroll_controller_module

    class _FakeGC:
        enabled = True
        disable_calls = 0
        enable_calls = 0
        collect_generations: ClassVar[list[int]] = []

        @classmethod
        def isenabled(cls) -> bool:
            return cls.enabled

        @classmethod
        def disable(cls) -> None:
            cls.disable_calls += 1
            cls.enabled = False

        @classmethod
        def enable(cls) -> None:
            cls.enable_calls += 1
            cls.enabled = True

        @classmethod
        def collect(cls, generation: int = 2) -> int:
            cls.collect_generations.append(generation)
            return 0

    monkeypatch.setattr(scroll_controller_module, "gc", _FakeGC)
    monkeypatch.setattr(scroll_controller_module, "_SCROLL_GC_PAUSE_OWNERS", 0)
    monkeypatch.setattr(scroll_controller_module, "_SCROLL_GC_WAS_ENABLED", False)
    return scroll_controller_module, _FakeGC


async def _wait_for_chat_panel_gc_resume(pilot: object, panel: ChatPanel) -> None:
    """Poll until the scroll-GC debounce timer has resumed the panel state."""
    for _ in range(40):
        if not panel._manual_scroll_gc_paused:
            return
        await pilot.pause(0.05)
    assert panel._manual_scroll_gc_paused is False


class InputBarApp(App):
    def compose(self) -> ComposeResult:
        yield InputBar()


def _png_bytes(color: tuple[int, int, int], *, size: tuple[int, int] = (32, 20)) -> bytes:
    image = Image.new("RGB", size, color)
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


class DebugPanelApp(_LocalizedApp):
    def __init__(self, locale: str = "en") -> None:
        self.locale_controller = LocaleController(Settings(locale=locale))
        super().__init__()

    def compose(self) -> ComposeResult:
        yield DebugPanel()


class ContextPanelApp(TuiVariableDefaultsMixin, App):
    def compose(self) -> ComposeResult:
        yield ContextPanel()


class SidebarContextPanelApp(TuiVariableDefaultsMixin, App):
    def compose(self) -> ComposeResult:
        yield SidebarPanel()


class SessionJsonPanelApp(App):
    def compose(self) -> ComposeResult:
        yield SessionJsonPanel()


class AppHeaderApp(App):
    def compose(self) -> ComposeResult:
        yield AppHeader()


class AppHeaderWithoutApprovalApp(App):
    def compose(self) -> ComposeResult:
        yield AppHeader(show_approval_badge=False)


def _header_zone_click(header: ToolCardHeader, zone: str) -> Click:
    """Synthesize a click on a revealed ToolCardHeader action zone."""
    zones = {name: (start, end) for name, start, end in ToolCardHeader._ACTION_ZONES}
    start, end = zones[zone]
    x = header.content_size.width - ToolCardHeader._ACTIONS_WIDTH + (start + end) // 2
    event = Click(
        header,
        x=x,
        y=0,
        delta_x=0,
        delta_y=0,
        button=1,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=header.region.x + x,
        screen_y=header.region.y,
    )
    header.on_click(event)
    return event


def _click_copy_button(header: ToolCardHeader) -> None:
    _header_zone_click(header, "copy")


def _click_widget(widget: Widget) -> Click:
    event = Click(
        widget,
        x=0,
        y=0,
        delta_x=0,
        delta_y=0,
        button=1,
        shift=False,
        meta=False,
        ctrl=False,
        screen_x=widget.region.x,
        screen_y=widget.region.y,
    )
    widget.on_click(event)
    return event


@pytest.mark.asyncio
async def test_click_affordance_buttons_prevent_widget_default_action() -> None:
    seen: list[str] = []

    class ButtonClickApp(App):
        def compose(self) -> ComposeResult:
            yield ToolViewButton()
            yield ToolCopyButton()
            yield ToolGroupTitle("tools")
            yield AgentCopyButton()
            yield _ScrollToBottomButton()

        def on_tool_view_button_clicked(self, _event: object) -> None:
            seen.append("tool-view")

        def on_tool_copy_button_clicked(self, _event: object) -> None:
            seen.append("tool-copy")

        def on_tool_group_title_clicked(self, _event: object) -> None:
            seen.append("tool-group-title")

        def on_agent_copy_button_clicked(self, _event: object) -> None:
            seen.append("agent-copy")

        def on_scroll_to_bottom_requested(self, _event: object) -> None:
            seen.append("scroll-bottom")

    async with ButtonClickApp().run_test() as pilot:
        await pilot.pause()

        widgets = [
            ("tool-view", pilot.app.query_one(ToolViewButton)),
            ("tool-copy", pilot.app.query_one(ToolCopyButton)),
            ("tool-group-title", pilot.app.query_one(ToolGroupTitle)),
            ("agent-copy", pilot.app.query_one(AgentCopyButton)),
            ("scroll-bottom", pilot.app.query_one(_ScrollToBottomButton)),
        ]
        for _name, widget in widgets:
            event = _click_widget(widget)
            assert event._no_default_action is True
            assert event._stop_propagation is True

        await pilot.pause()

    assert set(seen) == {"tool-view", "tool-copy", "tool-group-title", "agent-copy", "scroll-bottom"}
    assert len(seen) == 5


@pytest.mark.asyncio
async def test_click_affordance_real_dispatch_skips_widget_default_action(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    base_clicks: list[Widget] = []

    class ProbeAffordance(ClickAffordance):
        class Clicked(Message):
            """Posted when the probe affordance is clicked."""

        CLICK_MESSAGE = Clicked

        def __init__(self) -> None:
            super().__init__("probe", id="probe-affordance")

    async def record_base_click(self: Widget, _event: Click) -> None:
        base_clicks.append(self)

    monkeypatch.setattr(Widget, "_on_click", record_base_click)

    class DispatchApp(App):
        def compose(self) -> ComposeResult:
            yield ProbeAffordance()

        def on_probe_affordance_clicked(self, _event: ProbeAffordance.Clicked) -> None:
            seen.append("probe")

    async with DispatchApp().run_test() as pilot:
        await pilot.click("#probe-affordance")
        await pilot.pause()

    assert seen == ["probe"]
    assert base_clicks == []


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def test_conversation_toc_default_summary_keeps_longer_prompt() -> None:
    prompt = "x" * 100
    assert summarize_prompt(prompt) == prompt

    long_prompt = "x" * 121
    assert summarize_prompt(long_prompt) == ("x" * 117) + "..."


@pytest.mark.asyncio
async def test_conversation_toc_scrollbar_is_one_cell_and_flush_right() -> None:
    class TocApp(App):
        def compose(self) -> ComposeResult:
            yield ConversationToc()

    async with TocApp().run_test(size=(50, 10)) as pilot:
        toc = pilot.app.query_one(ConversationToc)
        toc.update_items([TocItem(turn_id=f"turn-{i}", summary=f"Turn {i}") for i in range(20)])
        tree = pilot.app.query_one("#toc-tree", Tree)

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            scrollbar_region = tree.vertical_scrollbar.region
            if scrollbar_region.width == 1 and scrollbar_region.right == toc.region.right:
                break
            await pilot.pause()

        assert tree.styles.scrollbar_size_vertical == 1
        assert tree.vertical_scrollbar.region.width == 1
        assert tree.vertical_scrollbar.region.right == toc.region.right


@pytest.mark.asyncio
async def test_conversation_toc_label_keeps_longer_summary() -> None:
    class TocApp(App):
        def compose(self) -> ComposeResult:
            yield ConversationToc()

    summary = "x" * 100
    async with TocApp().run_test() as pilot:
        toc = pilot.app.query_one(ConversationToc)
        toc.update_items([TocItem(turn_id="turn-1", summary=summary)])
        await pilot.pause()

        tree = pilot.app.query_one("#toc-tree", Tree)
        label = tree.root.children[0].label
        assert isinstance(label, Text)
        assert label.plain == f"1. {summary}"


@pytest.mark.asyncio
async def test_conversation_toc_expands_tabs_before_tree_render() -> None:
    class TocApp(App):
        def compose(self) -> ComposeResult:
            yield ConversationToc()

    summary = "样例\t文本 mock\t数据"
    async with TocApp().run_test(size=(70, 10)) as pilot:
        toc = pilot.app.query_one(ConversationToc)
        toc.update_items([TocItem(turn_id="turn-1", summary=summary)])
        await pilot.pause()

        tree = pilot.app.query_one("#toc-tree", Tree)
        label = tree.root.children[0].label
        assert isinstance(label, Text)
        assert label.plain == f"1. {summary}".expandtabs(8)

        rendered = "".join(segment.text for segment in tree.render_line(0)._segments).rstrip()
        assert "\t" not in rendered
        assert rendered == f"1. {summary}".expandtabs(8)


@pytest.mark.asyncio
async def test_app_header_title_uses_product_punctuation() -> None:
    from chrys import __version__

    async with AppHeaderApp().run_test() as pilot:
        title = pilot.app.query_one("#header-title", Static)
        assert title.render().plain == format_app_version_title(__version__)


@pytest.mark.asyncio
async def test_app_header_can_hide_approval_badge() -> None:
    async with AppHeaderWithoutApprovalApp().run_test() as pilot:
        assert list(pilot.app.query("#approval-badge")) == []


@pytest.mark.asyncio
async def test_app_header_reactives_refresh_title_and_approval_badge() -> None:
    async with AppHeaderApp().run_test() as pilot:
        header = pilot.app.query_one(AppHeader)
        header.subtitle_parts = ("Code", "mock-model")
        header.approval_mode = ApprovalMode.AUTO
        await pilot.pause()

        title = pilot.app.query_one("#header-title", Static)
        badge = pilot.app.query_one("#approval-badge", Static)

        assert title.render().plain.endswith("Code \u2502 mock-model")
        assert badge.render().plain == " APPROVAL MODE: AUTO "
        assert badge.has_class("approval-auto")
        assert badge.allow_select is False
        pilot.app.screen.selections = {badge: SELECT_ALL}
        await pilot.pause()
        assert badge.text_selection is None
        assert pilot.app.screen.get_selected_text() == ""


@pytest.mark.asyncio
async def test_app_header_relocalizes_current_approval_mode_and_unregisters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    header: AppHeader | None = None

    class HeaderApp(App):
        def compose(self) -> ComposeResult:
            yield AppHeader(locale_controller=controller)

    async with HeaderApp().run_test() as pilot:
        header = pilot.app.query_one(AppHeader)
        badge = header.query_one("#approval-badge", Static)
        assert header in controller._surfaces
        assert badge.render().plain == " APPROVAL MODE: MANUAL "

        result = controller.switch_locale("zh-Hans")
        assert result.status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        assert badge.render().plain == " 审批模式：手动 "  # noqa: RUF001

        for mode, expected in (
            (ApprovalMode.AUTO, " 审批模式：自动 "),  # noqa: RUF001
            (ApprovalMode.BYPASS, " 审批模式：绕过 "),  # noqa: RUF001
            (ApprovalMode.MANUAL, " 审批模式：手动 "),  # noqa: RUF001
        ):
            header.approval_mode = mode
            await pilot.pause()
            assert badge.render().plain == expected

    assert header is not None
    assert header not in controller._surfaces


@pytest.mark.asyncio
async def test_app_header_approval_badge_click_consumes_event() -> None:
    messages: list[AppHeader.ApprovalBadgeClicked] = []

    class HeaderClickApp(App):
        def compose(self) -> ComposeResult:
            yield AppHeader()

        @on(AppHeader.ApprovalBadgeClicked)
        def on_approval_badge_clicked(self, event: AppHeader.ApprovalBadgeClicked) -> None:
            messages.append(event)

    async with HeaderClickApp().run_test() as pilot:
        header = pilot.app.query_one(AppHeader)
        badge = pilot.app.query_one("#approval-badge", Static)
        event = Click(
            header,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=badge.region.x,
            screen_y=badge.region.y,
        )
        header.on_click(event)
        await pilot.pause()

    assert event._no_default_action is True
    assert event._stop_propagation is True
    assert len(messages) == 1


@pytest.mark.asyncio
async def test_status_bar_status_reactive_refreshes_text() -> None:
    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar()

    async with SBApp().run_test() as pilot:
        status_bar = pilot.app.query_one(StatusBar)
        status_bar.status = "Thinking"
        await pilot.pause()

        assert pilot.app.query_one("#status-text", Static).render().plain == "Thinking"


@pytest.mark.asyncio
async def test_status_bar_relocalizes_status_tool_trail_tooltip_and_literal_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from chrys.app.tui.screens.main.model_indicator import ModelIndicatorState
    from chrys.app.tui.screens.main.runtime_info import RegistryRuntimeInfoProvider
    from chrys.app.tui.widgets.chrome import status_bar as status_bar_module

    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    now = [100.0]
    monkeypatch.setattr(status_bar_module.time, "monotonic", lambda: now[0])
    status_bar: StatusBar | None = None

    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar(locale_controller=controller)

    async with SBApp().run_test() as pilot:
        status_bar = pilot.app.query_one(StatusBar)
        assert status_bar in controller._surfaces
        status_bar.set_profile("Code Agent")
        status_bar.set_model(
            ModelIndicatorState(
                label="Test Model",
                tooltip="",
                mode="select",
                profile_id="test-model",
                visible=True,
            )
        )
        status_bar.start_run()
        now[0] = 161.0
        status_bar.add_tool_call()
        runtime_info = RegistryRuntimeInfoProvider(  # type: ignore[arg-type]
            SimpleNamespace(agent_registry=None)
        )
        status_bar.set_tool_info(
            runtime_info.format_tool_info(
                ["read_file", "write_file"],
                ["review"],
                memory_files=["AGENTS.md"],
                runtime_details=AgentRuntimeDetails(
                    hook_sources=[
                        RuntimeHookSourceDetails(
                            scope="project",
                            hooks=[
                                RuntimeHookDetails(id="guard", enabled=True),
                                RuntimeHookDetails(id="notify", enabled=True),
                                RuntimeHookDetails(id="disabled", enabled=False),
                            ],
                        )
                    ]
                ),
            )
        )
        status_bar.show(STATUS_THINKING.bind())

        assert status_bar.query_one("#agent-label", Static).render().plain == "Agent"
        assert status_bar.query_one("#model-label", Static).render().plain == "Model"
        assert status_bar.query_one("#status-text", Static).render().plain == "Thinking"
        assert status_bar.query_one("#status-trail", Static).render().plain == "  (1m 1s · 1 tool call)"
        assert (
            status_bar.query_one("#status-tool-info", Static).render().plain == "2 tools · 1 skill · 2 hooks · 1 file"
        )

        assert controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        assert status_bar.query_one("#agent-label", Static).render().plain == "智能体"
        assert status_bar.query_one("#model-label", Static).render().plain == "模型"
        assert status_bar.query_one("#status-text", Static).render().plain == "正在思考"
        assert status_bar.query_one("#status-trail", Static).render().plain == "  (1分 1秒 · 1 次工具调用)"
        tool_info = status_bar.query_one("#status-tool-info", Static)
        assert tool_info.render().plain == "2 个工具 · 1 项技能 · 2 个钩子 · 1 个文件"
        assert tool_info.tooltip is not None
        assert tool_info.tooltip.plain == "点击查看详情"

        status_bar.show("provider payload")
        assert controller.switch_locale("en").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        assert status_bar.query_one("#status-text", Static).render().plain == "provider payload"

    assert status_bar is not None
    assert status_bar not in controller._surfaces


@pytest.mark.asyncio
async def test_status_bar_relocalizes_active_flash_and_snapshot_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)

    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar(locale_controller=controller)

    async with SBApp().run_test() as pilot:
        status_bar = pilot.app.query_one(StatusBar)
        status_bar.set_tool_info((STATUS_SESSION_RESTORED.bind(session_id="abc123"),))
        status_bar.show(STATUS_THINKING.bind())
        snapshot = status_bar.snapshot()

        status_bar.flash(STATUS_INTERRUPTED.bind(), caution=True)
        assert status_bar.query_one("#status-flash", Static).render().plain == "Interrupted by user"

        assert controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        assert status_bar.query_one("#status-flash", Static).render().plain == "已由用户中断"

        status_bar.restore(snapshot)
        await pilot.pause()
        assert status_bar.query_one("#status-text", Static).render().plain == "正在思考"
        assert status_bar.query_one("#status-tool-info", Static).render().plain == "会话已恢复：abc123"  # noqa: RUF001


@pytest.mark.asyncio
async def test_status_bar_completed_flash_restores_localized_elapsed_from_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.app.tui.widgets.chrome import status_bar as status_bar_module

    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    now = [100.0]
    monkeypatch.setattr(status_bar_module.time, "monotonic", lambda: now[0])

    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar(locale_controller=controller)

    async with SBApp().run_test() as pilot:
        status_bar = pilot.app.query_one(StatusBar)
        status_bar.start_run()
        now[0] = 161.0
        status_bar.flash(STATUS_COMPLETED.bind(elapsed=status_bar._format_elapsed()))
        snapshot = status_bar.snapshot()
        assert status_bar.query_one("#status-flash", Static).render().plain == "Completed in 1m 1s"

        status_bar.flash("shell payload", warn=True)
        assert controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        status_bar.restore(snapshot)

        assert status_bar.query_one("#status-flash", Static).render().plain == "已在 1分 1秒 内完成"


@pytest.mark.asyncio
async def test_status_bar_localizes_tool_count_choices_and_elapsed_formats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.app.tui.widgets.chrome import status_bar as status_bar_module

    controller = LocaleController(Settings(locale="zh-Hans"))
    now = [117.0]
    monkeypatch.setattr(status_bar_module.time, "monotonic", lambda: now[0])

    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar(locale_controller=controller)

    async with SBApp().run_test() as pilot:
        status_bar = pilot.app.query_one(StatusBar)
        status_bar._start_time = 100.0
        status_bar._tool_count = 1
        status_bar.show(STATUS_THINKING.bind())
        assert status_bar.query_one("#status-trail", Static).render().plain == "  (17秒 · 1 次工具调用)"

        now[0] = 221.0
        status_bar._tool_count = 2
        status_bar.refresh_localization()
        assert status_bar.query_one("#status-trail", Static).render().plain == "  (2分 1秒 · 2 次工具调用)"


@pytest.mark.asyncio
async def test_main_view_status_callers_preserve_message_refs_for_live_retranslation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from textual.widgets import Footer

    from chrys.app.tui.screens.main.view_adapter import MainScreenViewAdapter
    from chrys.app.tui.terminal.panel import ShellPanel
    from chrys.app.tui.widgets.sidebar.panel import SidebarPanel

    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)

    class _RetryPanel:
        async def prepare_retry(self) -> None:
            return

        async def add_retry(self, *_args: object) -> None:
            return

    class _Terminal:
        def focus(self) -> None:
            return

    class _Shell:
        def query_one(self, _widget_type: type) -> _Terminal:
            return _Terminal()

    class _Sidebar:
        is_visible = False

        def toggle(self) -> None:
            return

    class _Footer:
        display = True

    class _Screen:
        _sidebar_was_visible = False
        _shell_mode = True
        _agent_running = True

        def __init__(self, status_bar: StatusBar) -> None:
            self.status_bar = status_bar
            self.retry_panel = _RetryPanel()
            self.shell = _Shell()
            self.sidebar = _Sidebar()
            self.footer = _Footer()
            self.terminal_title_result = ""

        def _mark_terminal_title_completed(self) -> None:
            self.terminal_title_result = "✓"

        def _mark_terminal_title_failed(self) -> None:
            self.terminal_title_result = "✗"

        def query_one(self, widget_type: type):
            if widget_type is StatusBar:
                return self.status_bar
            if widget_type is ChatPanel:
                return self.retry_panel
            if widget_type is ShellPanel:
                return self.shell
            if widget_type is SidebarPanel:
                return self.sidebar
            if widget_type is Footer:
                return self.footer
            raise AssertionError(widget_type)

    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar(locale_controller=controller)

    async with SBApp().run_test() as pilot:
        status_bar = pilot.app.query_one(StatusBar)
        screen = _Screen(status_bar)
        adapter = MainScreenViewAdapter(screen)  # type: ignore[arg-type]

        adapter.flash_interrupted()
        assert status_bar._flash is not None
        assert isinstance(status_bar._flash.text, MessageRef)
        assert controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        assert status_bar.query_one("#status-flash", Static).render().plain == "已由用户中断"

        adapter.flash_turn_complete()
        adapter.mark_terminal_title_completed()
        assert screen.terminal_title_result == "✓"
        assert status_bar._flash is not None
        assert isinstance(status_bar._flash.text, MessageRef)
        assert status_bar.query_one("#status-flash", Static).render().plain.startswith("已在 ")

        adapter.flash_status(status_bar._flash.text, error=True)
        adapter.mark_terminal_title_failed()
        assert screen.terminal_title_result == "✗"

        adapter.start_tool_status("read_file")
        assert isinstance(status_bar.status, MessageRef)
        assert status_bar.query_one("#status-text", Static).render().plain == "正在运行：read_file"  # noqa: RUF001

        await adapter.show_retry_attempt("temporary", 2, 4, 1)
        assert isinstance(status_bar.status, MessageRef)
        assert status_bar.query_one("#status-text", Static).render().plain == "正在重试（2/4）..."  # noqa: RUF001

        adapter.set_alternate_screen_active(False)
        assert status_bar._flash is not None
        assert isinstance(status_bar._flash.text, MessageRef)
        assert status_bar.query_one("#status-flash", Static).render().plain == (
            "终端模式 — 连按两次 Esc 或输入 exit 退出"
        )

        screen._shell_mode = False
        status_bar.set_profile("Code Agent")
        status_bar.flash("Interactive terminal")
        adapter.set_alternate_screen_active(False)
        assert status_bar.visible is True
        assert status_bar._flash is None
        assert status_bar.query_one("#profile-tag", Static).render().plain == "Code Agent"


def test_chat_panel_agent_running_reactive_resets_final_response_gate() -> None:
    panel = ChatPanel()
    panel._final_response_started = True

    panel.agent_running = True

    assert panel._agent_running is True
    assert panel._final_response_started is False


@pytest.mark.asyncio
async def test_chat_panel_clear_resets_defensive_agent_running_cache() -> None:
    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        panel.agent_running = True

        await panel.clear()

        assert panel.agent_running is True
        assert panel._agent_running is False


@pytest.mark.asyncio
async def test_chat_panel_clear_preserves_runtime_metadata_for_replay_and_live_headers() -> None:
    raw_messages = [
        {
            "role": "assistant",
            "contents": [
                {
                    "type": "function_call",
                    "call_id": "ask-1",
                    "name": "ask_user",
                    "arguments": {"question": "Continue?"},
                }
            ],
        },
        {
            "role": "tool",
            "contents": [
                {
                    "type": "function_result",
                    "call_id": "ask-1",
                    "result": "User response: yes",
                }
            ],
        },
    ]

    async with ChatPanelApp().run_test(size=(100, 30)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        panel.set_profile("Code Agent")
        panel.set_tool_kinds({"ask_user": KIND_ASK_USER})
        await panel.add_user_message("before clear")

        await panel.clear()
        await panel.replay_history(raw_messages)
        await pilot.pause()

        group = panel.query_one(ToolGroup)
        assert group._tool_records["ask-1#0"].tool_kind == KIND_ASK_USER

        group.collapsed = False
        for _ in range(300):
            await pilot.pause()
            if group._content_mounted and group._tools:
                break

        assert isinstance(next(iter(group._tools.values())), AskUserToolCall)

        await panel.add_agent_message("live response")
        await pilot.pause()

        assert panel.query_one(AgentMessage)._copy_label() == "Code Agent"


@pytest.mark.asyncio
async def test_user_message_renders() -> None:
    class MsgApp(App):
        def compose(self) -> ComposeResult:
            yield UserMessage("hello world")

    async with MsgApp().run_test() as pilot:
        msg = pilot.app.query_one(UserMessage)
        rendered = msg.query_one(_UserMessageText).render()
        assert "hello world" in rendered.plain


def test_user_message_renders_embedded_image_preview() -> None:
    contents = [
        "look at @shot.png",
        Content.from_data(data=_png_bytes((240, 80, 80)), media_type="image/png"),
    ]
    msg = UserMessage("look at @shot.png", contents=contents)

    console = Console(width=80, record=True, force_terminal=True, color_system="truecolor")
    console.print(_UserMessageText(msg._text, timestamp="", is_injection=False).render())
    console.print(_UserImagePreview(msg._image_previews, is_injection=False).render())
    rendered = console.export_text(styles=False)

    assert "look at @shot.png" in rendered
    assert "\u2580" in rendered
    assert len(msg._image_previews) == 1


@pytest.mark.asyncio
async def test_user_message_with_embedded_image_keeps_text_selectable() -> None:
    text = "look at @shot.png"
    contents = [
        text,
        Content.from_data(data=_png_bytes((240, 80, 80)), media_type="image/png"),
    ]

    class MsgApp(App):
        def compose(self) -> ComposeResult:
            yield UserMessage(text, contents=contents)

    async with MsgApp().run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        msg = pilot.app.query_one(UserMessage)
        text_widget = msg.query_one(_UserMessageText)
        image_widget = msg.query_one(_UserImagePreview)
        widget, offset = pilot.app.screen.get_widget_and_offset_at(text_widget.region.x + 6, text_widget.region.y + 1)
        image_hit, image_offset = pilot.app.screen.get_widget_and_offset_at(
            image_widget.region.x, image_widget.region.y
        )

        assert widget is text_widget
        assert offset == Offset(6, 1)
        assert text_widget.get_selection(Selection(Offset(0, 1), Offset(len(text), 1))) == (text, "\n")
        assert image_hit is image_widget
        assert image_offset is None
        assert image_widget.allow_select is False
        assert image_widget.get_selection(Selection(None, None)) is None


@pytest.mark.asyncio
async def test_user_message_tab_selection_uses_source_offsets() -> None:
    apply_runtime_patch()

    class MsgApp(App):
        def compose(self) -> ComposeResult:
            yield UserMessage("A\tB")

    async with MsgApp().run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        text_widget = pilot.app.query_one(_UserMessageText)
        origin = text_widget.region.offset

        widget, offset = pilot.app.screen.get_widget_and_offset_at(origin.x + 8, origin.y + 1)

        assert widget is text_widget
        assert offset == Offset(2, 1)
        assert text_widget.get_selection(Selection(Offset(2, 1), Offset(3, 1))) == ("B", "\n")


@pytest.mark.asyncio
async def test_user_message_parent_selection_falls_back_to_text() -> None:
    text = "parent-selected text"

    class MsgApp(App):
        def compose(self) -> ComposeResult:
            yield UserMessage(text)

    async with MsgApp().run_test() as pilot:
        await pilot.pause()
        msg = pilot.app.query_one(UserMessage)

        assert msg.get_selection(Selection(Offset(0, 1), Offset(len(text), 1))) == (text, "\n")
        assert msg.get_selection(SELECT_ALL) == (f"[You]\n{text}", "\n")


@pytest.mark.asyncio
async def test_user_message_parent_selection_defers_to_selected_text_child() -> None:
    text = "child-selected text"

    class MsgApp(App):
        def compose(self) -> ComposeResult:
            yield UserMessage(text)

    async with MsgApp().run_test() as pilot:
        await pilot.pause()
        msg = pilot.app.query_one(UserMessage)
        text_widget = msg.query_one(_UserMessageText)
        pilot.app.screen.selections = {
            msg: SELECT_ALL,
            text_widget: Selection(Offset(0, 1), Offset(len(text), 1)),
        }

        assert msg.get_selection(SELECT_ALL) is None
        assert text_widget.get_selection(Selection(Offset(0, 1), Offset(len(text), 1))) == (text, "\n")


def test_image_preview_extracts_bounded_display_copy() -> None:
    previews = extract_image_previews(
        [Content.from_data(data=_png_bytes((240, 80, 80), size=(1200, 800)), media_type="image/png")]
    )

    assert len(previews) == 1
    assert previews[0].image.width <= 512
    assert previews[0].image.height <= 512


def test_user_image_preview_caches_render_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    contents = [
        "look at @shot.png",
        Content.from_data(data=_png_bytes((240, 80, 80)), media_type="image/png"),
    ]
    calls = 0
    original_render_lines = ImagePreviewGrid.render_lines

    def render_lines(grid: ImagePreviewGrid) -> list[Text]:
        nonlocal calls
        calls += 1
        return original_render_lines(grid)

    monkeypatch.setattr("chrys.app.tui.widgets.chat.messages.ImagePreviewGrid.render_lines", render_lines)
    msg = UserMessage("look at @shot.png", contents=contents)
    preview = _UserImagePreview(msg._image_previews, is_injection=False)

    preview.render()
    preview.render()

    assert calls == 1


def test_image_preview_extract_ignores_decompression_bomb(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def raise_decompression_bomb(_data: object) -> None:
        raise Image.DecompressionBombError("too large")

    monkeypatch.setattr(image_preview_module.Image, "open", raise_decompression_bomb)

    with caplog.at_level("DEBUG", logger=image_preview_module.__name__):
        assert extract_image_previews([Content.from_data(data=b"not decoded", media_type="image/png")]) == []
    assert "Skipping invalid chat image preview" in caplog.text


def test_image_preview_grid_renders_multiple_images_in_one_row_and_resizes() -> None:
    contents = [
        Content.from_data(data=_png_bytes((240, 80, 80), size=(48, 30)), media_type="image/png"),
        Content.from_data(data=_png_bytes((80, 180, 120), size=(48, 30)), media_type="image/png"),
    ]
    previews = extract_image_previews(contents)

    wide_lines = ImagePreviewGrid(previews, max_width=42, max_rows=6).render_lines()
    narrow_lines = ImagePreviewGrid(previews, max_width=20, max_rows=6).render_lines()

    assert len(previews) == 2
    assert 0 < len(wide_lines) <= 6
    assert 0 < len(narrow_lines) <= 6
    assert max(line.cell_len for line in wide_lines) <= 42
    assert max(line.cell_len for line in narrow_lines) <= 20
    assert max(line.cell_len for line in narrow_lines) < max(line.cell_len for line in wide_lines)


def test_image_preview_grid_can_upscale_to_requested_width() -> None:
    previews = extract_image_previews(
        [Content.from_data(data=_png_bytes((240, 80, 80), size=(4, 2)), media_type="image/png")]
    )

    lines = ImagePreviewGrid(previews, max_width=20, max_rows=100, allow_upscale=True).render_lines()

    assert max(line.cell_len for line in lines) == 20
    assert len(lines) == 5


@pytest.mark.asyncio
async def test_replay_history_decodes_image_previews_from_base64_contents() -> None:
    raw_messages = [
        {
            "role": "user",
            "contents": [
                {"type": "text", "text": "look at @a.png and @b.png"},
                Content.from_data(data=_png_bytes((240, 80, 80)), media_type="image/png").to_dict(),
                Content.from_data(data=_png_bytes((80, 180, 120)), media_type="image/png").to_dict(),
            ],
        }
    ]

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(raw_messages)
        await pilot.pause()

        msg = panel.query_one(UserMessage)
        assert msg._text == "look at @a.png and @b.png"
        assert len(msg._image_previews) == 2
        # Route to an in-memory buffer: Textual.run_test hijacks sys.stdout,
        # which on Windows resolves to a cp1252 stream that cannot encode the
        # user-bubble glyphs. record=True still captures into Rich's buffer.
        console = Console(file=StringIO(), width=80, record=True, force_terminal=True, color_system="truecolor")
        console.print(msg.query_one(_UserImagePreview).render())
        assert "\u2580" in console.export_text(styles=False)


@pytest.mark.asyncio
async def test_replay_history_decodes_tool_result_images_from_base64_items() -> None:
    image = Content.from_data(
        data=_png_bytes((240, 80, 80), size=(4, 4)),
        media_type="image/png",
        additional_properties={"width": 1977, "height": 1125, "media_type": "image/png"},
    )
    image_dict = image.to_dict()
    image_dict.pop("media_type")
    raw_messages = [
        {"role": "user", "contents": [{"type": "text", "text": "show image"}]},
        {
            "role": "assistant",
            "contents": [
                Content.from_function_call(
                    "tool-image",
                    "view_image",
                    arguments={"path": "plot.png"},
                ).to_dict()
            ],
        },
        {
            "role": "tool",
            "contents": [
                Content.from_function_result(
                    "tool-image",
                    result=[image],
                ).to_dict()
            ],
        },
    ]
    raw_messages[2]["contents"][0]["items"][0] = image_dict

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(raw_messages)
        await pilot.pause()

        group = panel.query_one(ToolGroup)
        assert group.collapsed is True
        assert group._content_mounted is False
        record = next(iter(group._tool_records.values()))
        assert len(record.image_contents) == 1

        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if list(panel.query("#tc-images")):
                break

        tool = panel.query_one(ToolCall)
        body = tool.query_one("#tc-body", Static)
        assert body.render().plain == ""
        image_panel = tool.query_one("#tc-panel")
        assert str(image_panel.border_subtitle) == "Resolution: 1977x1125 · Type: image/png"
        assert len(list(tool.query("#tc-images"))) == 1


def test_format_message_created_at_uses_local_time() -> None:
    created_at = datetime.now(UTC).replace(second=0, microsecond=0)
    local = created_at.astimezone()
    hour = local.hour % 12 or 12
    suffix = "AM" if local.hour < 12 else "PM"
    expected = f"- {hour}:{local.minute:02d} {suffix}"

    assert format_message_created_at(created_at) == expected
    assert format_message_created_at(created_at.isoformat()) == expected
    assert format_message_created_at("") == ""
    assert format_message_created_at("not-a-time") == ""
    assert format_message_created_at(123) == ""
    assert format_message_created_at({}) == ""


def test_message_headers_include_dim_timestamp_when_present() -> None:
    user = _UserMessageText("hello", timestamp="- 10:16 PM", is_injection=False)
    agent = AgentMessage("hi", profile_name="Code Agent", timestamp="- 10:17 PM", duration_ms=2345)
    instant_agent = AgentMessage("hi", profile_name="Code Agent", duration_ms=0)

    assert user.render().plain.splitlines()[0] == "\u276f You - 10:16 PM"
    assert agent._header_text().plain == "\u25c7 Code Agent - 10:17 PM (2s)"
    assert instant_agent._header_text().plain == "\u25c7 Code Agent (0ms)"


@pytest.mark.parametrize(
    ("locale", "agent_label", "think_text", "button_text", "tooltip", "retry_text", "error_text"),
    [
        (
            "en",
            "Agent",
            "Think: *one*\n\n*two*",
            "copy",
            "Copy agent response",
            "✗ Error\ntemporary failure Retrying in 7s (2/4)...",
            "✗ Error\nsomething broke",
        ),
        (
            "zh-Hans",
            "智能体",
            "思考：*one*\n\n*two*",  # noqa: RUF001
            "复制",
            "复制智能体回复",
            "✗ 错误\ntemporary failure 正在重试，等待 7 秒（2/4）...",  # noqa: RUF001
            "✗ 错误\nsomething broke",
        ),
    ],
)
@pytest.mark.asyncio
async def test_agent_and_retry_chrome_render_at_mount_locale(
    locale: str,
    agent_label: str,
    think_text: str,
    button_text: str,
    tooltip: str,
    retry_text: str,
    error_text: str,
) -> None:
    agent = AgentMessage("<think>one\n\ntwo</think>", is_intermediate=True)
    retry = RetryMessage("temporary failure", attempt=2, max_attempts=4, delay_seconds=7)
    error = ErrorMessage("something broke")

    class MessageApp(App):
        def __init__(self) -> None:
            self.locale_controller = LocaleController(Settings(locale=locale))
            super().__init__()

        def compose(self) -> ComposeResult:
            yield agent
            yield retry
            yield error

    async with MessageApp().run_test() as pilot:
        await pilot.pause()

        assert agent._copy_label() == agent_label
        assert agent._header_text().plain == f"◇ {agent_label}"
        assert agent._text == think_text
        copy_button = agent.query_one(AgentCopyButton)
        assert copy_button.render().plain == button_text
        assert copy_button.tooltip == tooltip
        assert retry.render().plain == retry_text
        assert error._render_text().plain == error_text


def test_streaming_agent_header_timestamp_updates_on_final() -> None:
    agent = AgentMessage("partial", is_final=False, profile_name="Code Agent", timestamp="- 10:16 PM")

    agent.stream_update("done", is_final=True, timestamp="- 10:17 PM")

    assert agent._header_text().plain == "\u25c7 Code Agent - 10:17 PM"


def test_agent_message_render_lines_unattached_returns_blank() -> None:
    rendered = AgentMessage("hello").render_lines(Region(0, 0, 20, 2))

    assert [strip.cell_length for strip in rendered] == [20, 20]


@pytest.mark.asyncio
async def test_agent_message_streaming() -> None:
    class MsgApp(App):
        def compose(self) -> ComposeResult:
            yield AgentMessage("partial", is_final=False)

    async with MsgApp().run_test() as pilot:
        msg = pilot.app.query_one(AgentMessage)
        assert msg._text == "partial"
        assert msg._is_final is False
        # Streaming cursor widget should be present
        assert msg.query(".agent-cursor")

        msg.stream_update("full response", is_final=True)
        assert msg._text == "full response"
        assert msg._is_final is True


@pytest.mark.asyncio
async def test_agent_copy_button_sits_next_to_header_timestamp() -> None:
    class MsgApp(App):
        def compose(self) -> ComposeResult:
            yield AgentMessage("hello", profile_name="Code Agent", timestamp="- 1:23 PM")

    async with MsgApp().run_test() as pilot:
        await pilot.pause()
        msg = pilot.app.query_one(AgentMessage)
        header = msg.query_one(".agent-header", Static)
        copy_button = msg.query_one(AgentCopyButton)

        assert copy_button.display is True
        assert copy_button.region.x == header.region.right


@pytest.mark.asyncio
async def test_empty_agent_message_hides_copy_button(monkeypatch: pytest.MonkeyPatch) -> None:
    class MsgApp(App):
        def compose(self) -> ComposeResult:
            yield AgentMessage("", profile_name="Code Agent", timestamp="- 1:23 PM")

    async with MsgApp().run_test() as pilot:
        copied: list[str] = []
        monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", copied.append)
        pilot.app.copy_to_clipboard("existing")
        await pilot.pause()

        msg = pilot.app.query_one(AgentMessage)
        copy_button = msg.query_one(AgentCopyButton)

        assert copy_button.display is False

        msg.copy_agent_response()
        assert pilot.app.clipboard == "existing"
        assert copied == []


@pytest.mark.asyncio
async def test_structured_completion_checkmark_is_styled_and_not_copyable() -> None:
    class MsgApp(App):
        def compose(self) -> ComposeResult:
            yield AgentMessage("✓", profile_name="Code Agent", is_structured_completion=True)

    async with MsgApp().run_test() as pilot:
        await pilot.pause()
        msg = pilot.app.query_one(AgentMessage)

        assert msg.has_class("--structured-completion")
        assert msg.query_one(AgentCopyButton).display is False


@pytest.mark.asyncio
async def test_agent_message_chrome_render_cache_reuses_rows_and_tracks_theme() -> None:
    class MsgApp(App):
        def compose(self) -> ComposeResult:
            yield AgentMessage("hello\nworld")

    async with MsgApp().run_test() as pilot:
        await pilot.pause()
        msg = pilot.app.query_one(AgentMessage)
        crop = Region(0, 0, msg.size.width, 3)

        first = msg.render_lines(crop)
        cached = msg._background_strip_cache
        cached_key = msg._background_strip_cache_key

        assert cached is not None
        assert cached_key is not None
        assert all(strip is cached for strip in first)

        taller = msg.render_lines(Region(0, 0, msg.size.width, 5))
        assert msg._background_strip_cache is cached
        assert all(strip is cached for strip in taller)

        next_color_system = (
            ColorSystem.STANDARD if str(pilot.app.console.color_system) == "256" else ColorSystem.EIGHT_BIT
        )
        pilot.app.console._color_system = next_color_system
        recolored = msg.render_lines(crop)

        assert msg._background_strip_cache_key != cached_key
        assert all(strip is msg._background_strip_cache for strip in recolored)

        theme_key = msg._background_strip_cache_key
        pilot.app.theme = "textual-light"
        await pilot.pause()
        themed = msg.render_lines(crop)

        assert msg._background_strip_cache_key != theme_key
        assert all(strip is msg._background_strip_cache for strip in themed)


def test_hatched_empty_state_render_lines_unattached_returns_blank() -> None:
    rendered = HatchedEmptyState("Empty").render_lines(Region(0, 0, 20, 2))

    assert [strip.cell_length for strip in rendered] == [20, 20]


@pytest.mark.asyncio
async def test_hatched_empty_state_centers_wide_glyph_labels_by_cell_width() -> None:
    """A CJK label is 2 cells per glyph; centring by code points shoves it right."""

    class HatchApp(App):
        def compose(self):
            yield HatchedEmptyState("没有已保存的会话。", id="hatch")

    async with HatchApp().run_test(size=(40, 6)) as pilot:
        await pilot.pause()
        widget = pilot.app.query_one("#hatch", HatchedEmptyState)
        text = widget.render_line(widget.size.height // 2).text
        label = " 没有已保存的会话。 "
        start = text.index(label)
        left = text[:start].count(HATCH_GLYPH)
        right = text[start + len(label) :].count(HATCH_GLYPH)
        assert (left, right) == ((40 - 20) // 2, 40 - 20 - (40 - 20) // 2)


def test_loading_indicator_render_unattached_returns_text() -> None:
    indicator = ChrysLoadingIndicator()
    rendered_lines = indicator.render_lines(Region(0, 0, 20, 2))
    rendered = indicator.render()

    assert [strip.cell_length for strip in rendered_lines] == [20, 20]
    assert rendered.plain == "Loading..."


@pytest.mark.asyncio
async def test_welcome_stays_plain_while_loading_resolves_live_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)

    class PaintApp(App):
        def __init__(self) -> None:
            self.locale_controller = controller
            super().__init__()

        def compose(self) -> ComposeResult:
            yield WelcomeWidget(profile="Code", cwd="/workspace")
            yield ChrysLoadingIndicator()

    async with PaintApp().run_test(size=(60, 20)) as pilot:
        pilot.app.animation_level = "none"
        welcome = pilot.app.query_one(WelcomeWidget)
        loading = pilot.app.query_one(ChrysLoadingIndicator)

        english = "".join(segment.text for segment in pilot.app.console.render(welcome.render()))
        assert "Code" in english
        assert "Agent:" not in english
        assert loading.render().plain == "Loading..."

        assert controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        chinese = "".join(segment.text for segment in pilot.app.console.render(welcome.render()))
        assert "Code" in chinese
        assert "智能体：" not in chinese  # noqa: RUF001
        assert loading.render().plain == "正在加载..."


@pytest.mark.asyncio
async def test_loading_indicator_automatic_refresh_skips_hidden_widget() -> None:
    """The 16/s animation tick must not repaint (nor arrange) while hidden.

    Stock ``DOMNode.automatic_refresh`` gates on ``is_on_screen`` →
    ``Screen.find_widget``, which recomputes the compositor's full map — an
    O(all widgets) arrange per tick on large transcripts even when the
    indicator is invisible.
    """
    from textual.containers import Container

    class _LoadingApp(App):
        def compose(self) -> ComposeResult:
            yield Container(ChrysLoadingIndicator())

    async with _LoadingApp().run_test(size=(40, 10)) as pilot:
        indicator = pilot.app.query_one(ChrysLoadingIndicator)
        container = pilot.app.query_one(Container)

        refreshes: list[bool] = []
        original_refresh = indicator.refresh

        def counting_refresh(*args: object, **kwargs: object) -> None:
            refreshes.append(True)
            original_refresh(*args, **kwargs)

        indicator.refresh = counting_refresh  # type: ignore[method-assign]

        indicator.automatic_refresh()
        assert refreshes, "a visible indicator must keep animating"

        container.display = False
        # visible_widgets is documentedly one-frame stale: a single pause can
        # land before the recomposite that drops the hidden widget from the
        # cut, so wait the compositor state out instead of racing it.
        compositor = pilot.app.screen._compositor
        for _ in range(20):
            await pilot.pause()
            if indicator not in compositor.visible_widgets:
                break
        else:
            pytest.fail("compositor never dropped the display:none indicator from its visible cut")
        # Clear only now: the indicator's live 16/s auto_refresh timer may
        # legitimately repaint once inside the stale window above (that is the
        # accepted one-frame staleness, not the behavior under test). No await
        # between here and the assert, so the timer cannot interleave.
        refreshes.clear()
        indicator.automatic_refresh()
        assert not refreshes, "a hidden indicator must not repaint on its animation tick"


@pytest.mark.asyncio
async def test_is_widget_shown_rejects_offscreen_widgets_without_arranging() -> None:
    """Offscreen widgets in scroll containers must not report as shown.

    The raw compositor maps (``_visible_map``/``_full_map``) retain entries
    for widgets scrolled out of view; only the ``visible_widgets`` cut applies
    the screen/clip predicate. Regression: an offscreen indicator reported
    shown whenever the visible caches were cleared but a stale full map
    remained (the window right after a reflow), reintroducing the repaint
    work the ``auto_refresh`` gating exists to avoid.
    """
    from textual.containers import VerticalScroll

    from chrys.app.tui.util.visibility import is_widget_shown

    class _ScrollApp(App):
        def compose(self) -> ComposeResult:
            with VerticalScroll():
                for index in range(50):
                    yield Static(f"filler {index}")
                yield ChrysLoadingIndicator()

    async with _ScrollApp().run_test(size=(40, 10)) as pilot:
        indicator = pilot.app.query_one(ChrysLoadingIndicator)
        first_static = pilot.app.query(Static).first()
        compositor = pilot.app.screen._compositor

        # Viewport sits at the top; the indicator is 50 rows below it.
        assert not is_widget_shown(indicator)
        assert is_widget_shown(first_static)

        # Post-reflow window: visible caches cleared, stale full map retained
        # (what find_widget/hit-test traffic leaves behind).
        _ = compositor.full_map
        assert indicator in compositor._full_map
        original_arrange_root = compositor._arrange_root
        try:
            compositor._arrange_root = None  # type: ignore[assignment]  # tripwire: must not arrange
            compositor._visible_widgets = None
            compositor._visible_map = None
            assert not is_widget_shown(indicator)

            compositor._visible_widgets = None
            assert is_widget_shown(first_static)
        finally:
            compositor._arrange_root = original_arrange_root  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_is_widget_shown_rejects_visibility_hidden_widgets() -> None:
    """``visibility: hidden`` widgets must not report as shown.

    After a reflow the compositor prunes hidden subtrees from its map, but
    the cached ``visible_widgets`` cut is one frame stale: between a
    visibility flip and the next arrange the widget is still a member (and
    chrys deliberately keeps some visibility flips reflow-free, e.g. the
    scroll-to-bottom button). The helper must consult ``widget.visible`` —
    own rule or inherited — instead of trusting map membership.
    """
    from textual.containers import VerticalScroll

    from chrys.app.tui.util.visibility import is_widget_shown

    class _App(App):
        def compose(self) -> ComposeResult:
            with VerticalScroll():
                yield Static("content")

    async with _App().run_test(size=(40, 10)) as pilot:
        static = pilot.app.query_one(Static)
        container = pilot.app.query_one(VerticalScroll)
        assert is_widget_shown(static)

        # Inherited: the child has no visibility rule of its own. No pause —
        # the stale cached cut still holds the widget (premise assert).
        container.visible = False
        assert static in pilot.app.screen._compositor.visible_widgets
        assert not is_widget_shown(static)
        await pilot.pause()
        assert not is_widget_shown(static)

        container.visible = True
        await pilot.pause()
        assert is_widget_shown(static)

        # Own rule, same stale window.
        static.visible = False
        assert static in pilot.app.screen._compositor.visible_widgets
        assert not is_widget_shown(static)
        await pilot.pause()
        assert not is_widget_shown(static)


@pytest.mark.asyncio
async def test_chat_panel_replay_uses_created_at_metadata() -> None:
    created_at = datetime.now(UTC).replace(second=0, microsecond=0)
    expected = format_message_created_at(created_at)

    raw_messages = [
        {
            "role": "user",
            "contents": [{"type": "text", "text": "hello"}],
            "additional_properties": {
                MESSAGE_CREATED_AT_KEY: created_at.isoformat(),
                TRAJECTORY_TIMING_KEY: {
                    "started_at": created_at.isoformat(),
                    "finished_at": created_at.isoformat(),
                    "duration_ms": 0,
                },
            },
        },
        {
            "role": "assistant",
            "contents": [{"type": "text", "text": "hi"}],
            "additional_properties": {
                MESSAGE_CREATED_AT_KEY: created_at.isoformat(),
                TRAJECTORY_TIMING_KEY: {
                    "started_at": created_at.isoformat(),
                    "finished_at": created_at.isoformat(),
                    "duration_ms": 2345,
                },
            },
        },
    ]

    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        panel.set_profile("Code Agent")
        await panel.replay_history(raw_messages, initial_profile="Code Agent")
        await pilot.pause()

        user = panel.query_one(UserMessage)
        agent = panel.query_one(AgentMessage)
        assert user._ts == expected
        assert "(0ms)" not in user.query_one(_UserMessageText).render().plain
        assert agent._ts == expected
        assert agent._duration_ms == 2345
        assert agent._header_text().plain == f"\u25c7 Code Agent {expected} (2s)"


@pytest.mark.asyncio
async def test_chat_panel_replay_adds_action_only_for_trailing_interruption() -> None:
    raw_messages = [
        {
            "role": "assistant",
            "contents": [{"type": "text", "text": "Execution interrupted"}],
            "additional_properties": {
                HistoryMarkerKind.KEY: HistoryMarkerKind.INTERRUPTED,
                "_interrupted_by": "user",
            },
        },
        {
            "role": "user",
            "contents": [{"type": "text", "text": "later prompt"}],
        },
        {
            "role": "assistant",
            "contents": [{"type": "text", "text": "later answer"}],
        },
    ]

    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(raw_messages)
        await pilot.pause()

        assert not list(panel.query(Button))


@pytest.mark.asyncio
async def test_chat_panel_replay_adds_action_for_current_interruption() -> None:
    raw_messages = [
        {
            "role": "assistant",
            "contents": [{"type": "text", "text": "Execution interrupted"}],
            "additional_properties": {
                HistoryMarkerKind.KEY: HistoryMarkerKind.INTERRUPTED,
                "_interrupted_by": "error",
            },
        }
    ]

    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(raw_messages)
        await pilot.pause()

        action = panel.query_one(InterruptedMessage)
        assert str(action.query_one(Button).label) == "Retry"


@pytest.mark.asyncio
async def test_chat_panel_replay_localizes_recognized_status_and_frame() -> None:
    raw_messages = [
        {
            "role": "assistant",
            "contents": [{"type": "text", "text": "stale English literal"}],
            "additional_properties": {
                HistoryMarkerKind.KEY: HistoryMarkerKind.INTERRUPTED,
                HistoryMarkerKind.STATUS_CODE_KEY: HistoryMarkerKind.STATUS_EXECUTION_INTERRUPTED,
                "_interrupted_by": "user",
            },
        }
    ]

    async with ChatPanelApp(Localizer("zh-Hans")).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(raw_messages)
        await pilot.pause()

        interruption = panel.query_one(InterruptedMessage)
        rendered = interruption._render_text().plain
        assert rendered == "⚠ 已中断\n用户中断：执行已中断"  # noqa: RUF001
        assert "Interrupted" not in rendered
        assert "by user" not in rendered
        assert "stale English literal" not in rendered
        assert str(interruption.query_one(Button).label) == "继续"


@pytest.mark.asyncio
async def test_chat_panel_replay_unknown_status_keeps_literal() -> None:
    raw_messages = [
        {
            "role": "assistant",
            "contents": [{"type": "text", "text": "future marker literal"}],
            "additional_properties": {
                HistoryMarkerKind.KEY: HistoryMarkerKind.INTERRUPTED,
                HistoryMarkerKind.STATUS_CODE_KEY: "future_status",
                "_interrupted_by": "",
            },
        }
    ]

    async with ChatPanelApp(Localizer("zh-Hans")).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(raw_messages)
        await pilot.pause()

        assert panel.query_one(InterruptedMessage)._render_text().plain == "⚠ 已中断\nfuture marker literal"


@pytest.mark.asyncio
async def test_chat_panel_replay_sanitizes_legacy_marker_controls() -> None:
    raw_literal = "first\x1b[31mred\nsecond"
    raw_messages = [
        {
            "role": "assistant",
            "contents": [{"type": "text", "text": raw_literal}],
            "additional_properties": {
                HistoryMarkerKind.KEY: HistoryMarkerKind.INTERRUPTED,
                "_interrupted_by": "",
            },
        }
    ]

    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(raw_messages)
        await pilot.pause()

        rendered = panel.query_one(InterruptedMessage)._render_text().plain
        assert rendered == "⚠ Interrupted\nfirst�[31mred\nsecond"
        assert "\x1b" not in rendered
        assert raw_messages[0]["contents"][0]["text"] == raw_literal


@pytest.mark.asyncio
async def test_chat_panel_replay_awaiting_status_uses_structured_count() -> None:
    raw_messages = [
        {
            "role": "assistant",
            "contents": [{"type": "text", "text": "Awaiting 99 sub-agent(s)"}],
            "additional_properties": {
                HistoryMarkerKind.KEY: HistoryMarkerKind.AWAITING_SUB_AGENTS,
                HistoryMarkerKind.STATUS_CODE_KEY: HistoryMarkerKind.STATUS_AWAITING_SUB_AGENTS,
                "_invocation_ids": ["first", "second"],
            },
        }
    ]

    async with ChatPanelApp(Localizer("zh-Hans")).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(raw_messages)
        await pilot.pause()

        assert panel.query_one(AgentMessage)._text == "正在等待 2 个子智能体"
        assert raw_messages[0]["contents"][0]["text"] == "Awaiting 99 sub-agent(s)"


@pytest.mark.asyncio
async def test_chat_panel_replay_awaiting_status_without_structured_ids_keeps_literal() -> None:
    raw_messages = [
        {
            "role": "assistant",
            "contents": [{"type": "text", "text": "Awaiting 99 sub-agent(s)"}],
            "additional_properties": {
                HistoryMarkerKind.KEY: HistoryMarkerKind.AWAITING_SUB_AGENTS,
                HistoryMarkerKind.STATUS_CODE_KEY: HistoryMarkerKind.STATUS_AWAITING_SUB_AGENTS,
            },
        }
    ]

    async with ChatPanelApp(Localizer("zh-Hans")).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(raw_messages)
        await pilot.pause()

        assert panel.query_one(AgentMessage)._text == "Awaiting 99 sub-agent(s)"


@pytest.mark.asyncio
async def test_chat_panel_replay_downgraded_first_injection_makes_following_user_injection() -> None:
    raw_messages = [
        {
            "role": "user",
            "contents": [{"type": "text", "text": "first"}],
            "additional_properties": {"_injected": True},
        },
        {
            "role": "user",
            "contents": [{"type": "text", "text": "second"}],
        },
    ]

    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(raw_messages)
        await pilot.pause()

        user_messages = list(panel.query(UserMessage))
        assert [(message.id, message.is_injection, message._text) for message in user_messages] == [
            ("turn-1", False, "first"),
            ("inj-1", True, "second"),
        ]


@pytest.mark.asyncio
async def test_chat_panel_timestamps_only_final_agent_response() -> None:
    first_at = datetime(2026, 5, 15, 12, 0, 0, tzinfo=UTC)
    final_at = datetime(2026, 5, 15, 12, 1, 0, tzinfo=UTC)
    expected = format_message_created_at(final_at)

    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        panel.set_profile("Code Agent")
        await panel.add_user_message("hello", created_at=first_at)
        await panel.add_agent_message("working", is_final=True, is_intermediate=True, created_at=first_at)
        await panel.add_agent_message("partial", is_final=False, created_at=first_at)
        await pilot.pause()

        agent_messages = list(panel.query(AgentMessage))
        assert [message._ts for message in agent_messages] == ["", ""]

        await panel.add_agent_message("done", is_final=True, created_at=final_at)
        await pilot.pause()

        agent_messages = list(panel.query(AgentMessage))
        assert [message._ts for message in agent_messages] == ["", expected]


@pytest.mark.asyncio
async def test_chat_panel_toggle_fold_all_only_toggles_tool_groups() -> None:
    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        tool_group = ToolGroup()
        agent_message = AgentMessage("Done.")

        await panel.mount(tool_group)
        await panel.mount(agent_message)
        await pilot.pause()

        tool_group.collapsed = False
        agent_message.collapsed = False

        assert panel.toggle_fold_all() is True
        assert tool_group.collapsed is True
        assert agent_message.collapsed is False

        assert panel.toggle_fold_all() is False
        assert tool_group.collapsed is False
        assert agent_message.collapsed is False


@pytest.mark.asyncio
async def test_chat_panel_toggle_fold_all_affects_replayed_tool_groups() -> None:
    raw_messages = [
        {"role": "user", "contents": [{"type": "text", "text": "run"}]},
        {
            "role": "assistant",
            "contents": [{"type": "function_call", "call_id": "call-1", "name": "zsh", "arguments": "{}"}],
        },
        {"role": "tool", "contents": [{"type": "function_result", "call_id": "call-1", "result": "ok"}]},
    ]

    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(raw_messages)
        await pilot.pause()

        group = panel.query_one(ToolGroup)
        assert group.collapsed is True

        assert panel.toggle_fold_all() is False
        assert group.collapsed is False

        assert panel.toggle_fold_all() is True
        assert group.collapsed is True


@pytest.mark.asyncio
async def test_chat_panel_working_dir_click_posts_compatible_message() -> None:
    messages: list[ChatPanel.WorkingDirClicked] = []

    class WorkingDirApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

        def on_chat_panel_working_dir_clicked(self, event: ChatPanel.WorkingDirClicked) -> None:
            event.stop()
            messages.append(event)

    class _Click:
        def __init__(self, widget: ChatPanel, screen_y: int) -> None:
            self.widget = widget
            self.screen_y = screen_y

        def stop(self) -> None:
            return

        def prevent_default(self) -> None:
            return

    async with WorkingDirApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        panel.border_subtitle = Text("/tmp/project")
        await pilot.pause()

        panel.on_click(_Click(panel, panel.region.y + panel.region.height - 1))  # type: ignore[arg-type]
        await pilot.pause()

    assert len(messages) == 1
    assert isinstance(messages[0], ChatPanel.WorkingDirClicked)


@pytest.mark.asyncio
async def test_chat_panel_replay_omits_intermediate_agent_timestamps() -> None:
    created_at = datetime(2026, 5, 15, 12, 1, 0, tzinfo=UTC)
    expected = format_message_created_at(created_at)

    raw_messages = [
        {
            "role": "assistant",
            "contents": [
                {"type": "text", "text": "I'll inspect the files."},
                {"type": "function_call", "call_id": "call_1", "name": "zsh", "arguments": "{}"},
            ],
            "additional_properties": {MESSAGE_CREATED_AT_KEY: created_at.isoformat()},
        },
        {
            "role": "assistant",
            "contents": [{"type": "text", "text": "Done."}],
            "additional_properties": {MESSAGE_CREATED_AT_KEY: created_at.isoformat()},
        },
    ]

    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        panel.set_profile("Code Agent")
        await panel.replay_history(raw_messages, initial_profile="Code Agent")
        await pilot.pause()

        agent_messages = list(panel.query(AgentMessage))
        assert [message._is_intermediate for message in agent_messages] == [True, False]
        assert [message._ts for message in agent_messages] == ["", expected]


@pytest.mark.asyncio
async def test_error_message_renders() -> None:
    class MsgApp(App):
        def compose(self) -> ComposeResult:
            yield ErrorMessage("something broke")

    async with MsgApp().run_test() as pilot:
        msg = pilot.app.query_one(ErrorMessage)
        rendered = msg._render_text()
        assert "something broke" in rendered.plain
        assert "Error" in rendered.plain


@pytest.mark.asyncio
async def test_chrys_dark_conversation_status_rails_keep_semantic_colors() -> None:
    class MsgApp(TuiVariableDefaultsMixin, App):
        def __init__(self) -> None:
            super().__init__()
            self.register_theme(CHRYS_DARK_THEME)
            self.theme = CHRYS_DARK_THEME_NAME

        def compose(self) -> ComposeResult:
            yield ErrorMessage("something broke")
            yield RetryMessage("temporary failure", attempt=1, max_attempts=3, delay_seconds=1)
            yield InterruptedMessage("Execution interrupted", "user")
            yield InterruptedMessage("Execution interrupted", "error")

    async with MsgApp().run_test() as pilot:
        variables = pilot.app.get_css_variables()
        assert variables["tui-border-status-error"] == variables["error"]
        assert variables["tui-border-status-warning"] == variables["warning"]
        widgets = [
            (pilot.app.query_one(ErrorMessage), "error"),
            (pilot.app.query_one(RetryMessage), "error"),
            *zip(pilot.app.query(InterruptedMessage), ("warning", "error"), strict=True),
        ]
        for widget, semantic_color in widgets:
            border_color = widget.styles.border_left[1]
            expected_color = TextualColor.parse(variables[semantic_color])
            assert border_color.r == pytest.approx(expected_color.r, abs=1)
            assert border_color.g == pytest.approx(expected_color.g, abs=1)
            assert border_color.b == pytest.approx(expected_color.b, abs=1)
            assert border_color.a == 0.8


@pytest.mark.asyncio
async def test_conversation_status_action_posts_pressed() -> None:
    pressed: list[bool] = []

    class MsgApp(App):
        def compose(self) -> ComposeResult:
            yield ErrorMessage("something broke", action_label="Retry")

        @on(ConversationStatusAction.Pressed)
        def on_status_action_pressed(self, event: ConversationStatusAction.Pressed) -> None:
            event.stop()
            pressed.append(True)

    async with MsgApp().run_test() as pilot:
        action = pilot.app.query_one(ErrorMessage)
        button = action.query_one(Button)
        action.on_button_pressed(Button.Pressed(button))
        await pilot.pause()

        assert pressed == [True]
        assert button.parent is not None
        assert button.parent.display is False


@pytest.mark.asyncio
async def test_system_message_renders() -> None:
    class MsgApp(App):
        def compose(self) -> ComposeResult:
            yield SystemMessage("session started")

    async with MsgApp().run_test() as pilot:
        msg = pilot.app.query_one(SystemMessage)
        # SystemMessage passes text to Static constructor
        assert msg is not None


@pytest.mark.asyncio
async def test_status_bar_show_hide() -> None:
    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar()

    async with SBApp().run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        assert sb.styles.padding.left == 0
        assert sb.query_one(".status-selectors").styles.padding.right == 1
        assert sb.query_one(".status-selectors").display is False
        assert sb.query_one(".status-body").styles.padding.left == 0
        assert sb.shown is False
        assert sb.visible is False

        sb.show("thinking...")
        assert sb.shown is True
        assert sb.visible is True
        assert sb.status == "thinking..."
        assert sb.query_one(".status-run").visible is True
        assert sb.query_one(".status-flash-bar").visible is False

        sb.show("running: read_file")
        assert sb.status == "running: read_file"

        sb.flash("done")
        assert sb.query_one(".status-run").visible is False
        assert sb.query_one(".status-flash-bar").visible is True

        sb.hide()
        assert sb.shown is False
        assert sb.visible is False
        assert sb.query_one(".status-run").visible is False
        assert sb.query_one(".status-flash-bar").visible is False


@pytest.mark.asyncio
async def test_status_bar_flash_text_gets_full_row_when_trail_is_empty() -> None:
    """The auto-width trail must leave the whole row to the flash message."""

    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar()

    async with SBApp().run_test(size=(80, 6)) as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.flash("M" * 55 + "TAIL-END-MARKER")
        await pilot.pause()
        await pilot.pause()
        strips = pilot.app.screen._compositor.render_strips()
        frame = "\n".join(strip.text for strip in strips)
        # A half-row flash box (the old 1fr/1fr split) clips this tail.
        assert "TAIL-END-MARKER" in frame


@pytest.mark.asyncio
async def test_status_bar_run_trail_hugs_status_text() -> None:
    """The elapsed/tool-call trail sits right after the status label.

    A flexible status-text width strands the trail mid-row, far from the
    label it annotates; the tool info keeps the right edge of the row.
    """

    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar()

    async with SBApp().run_test(size=(100, 6)) as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.start_run()
        sb.add_tool_call()
        sb.set_tool_info("MODEL-MARKER")
        sb.show("Running: explore_agent")
        await pilot.pause()
        await pilot.pause()

        text_widget = sb.query_one("#status-text", Static)
        trail_widget = sb.query_one("#status-trail", Static)
        assert text_widget.region.width == len("Running: explore_agent")
        assert trail_widget.region.x == text_widget.region.right

        strips = pilot.app.screen._compositor.render_strips()
        row = next(strip.text for strip in strips if "Running:" in strip.text)
        # Trail text is adjacent (its two leading spaces are part of it) and
        # the persistent tool info stays right-aligned.
        assert "Running: explore_agent  (" in row
        assert "tool call)" in row
        assert row.rstrip().endswith("MODEL-MARKER")


@pytest.mark.asyncio
async def test_input_bar_buttons_hug_labels_without_side_gaps() -> None:
    """Localized labels drive button and group width without dead space."""
    async with InputBarApp().run_test(size=(80, 10)) as pilot:
        bar = pilot.app.query_one(InputBar)
        group = bar.query_one("#btn-group")
        editor = bar.query_one("#editor-btn", Button)
        send = bar.query_one("#send-btn", Button)
        new = bar.query_one("#new-btn", Button)
        text_area = bar.query_one("#chat-input", TextArea)
        assert text_area.styles.background.a == pytest.approx(0.04)
        assert editor.region.width == 3
        assert editor.allow_select is False
        assert text_area.allow_select is True
        input_prompt = bar.query_one("#editor-btn", Button)
        pilot.app.screen.selections = {input_prompt: SELECT_ALL}
        await pilot.pause()
        assert input_prompt.text_selection is None
        assert pilot.app.screen.get_selected_text() == ""
        assert send.allow_select is False
        assert new.allow_select is False
        pilot.app.screen.selections = {send: SELECT_ALL, new: SELECT_ALL}
        await pilot.pause()
        assert send.text_selection is None
        assert new.text_selection is None
        assert pilot.app.screen.get_selected_text() == ""

        # Idle without messages: collapsed New slot and label-hugging Send.
        assert new.region.width == 0
        assert send.region.width == len("Send") + 4
        assert send.region.x == group.content_region.x
        assert send.region.right == group.content_region.right

        bar.value = "\n".join(f"line {index}" for index in range(12))
        await pilot.pause()
        assert text_area.show_vertical_scrollbar is True
        assert text_area.styles.scrollbar_background.a == pytest.approx(0.12)
        assert text_area.styles.scrollbar_background_hover.a == pytest.approx(0.16)
        assert text_area.styles.scrollbar_background_active.a == pytest.approx(0.20)
        assert text_area.vertical_scrollbar.region.right + 1 == send.region.x

        # A longer localized-style label expands naturally without side gaps.
        localized_label = "发送消息"
        bar._set_btn_label(localized_label)
        await pilot.pause()
        assert send.region.width == Text(localized_label).cell_len + 4
        assert text_area.vertical_scrollbar.region.right + 1 == send.region.x
        assert send.region.right == group.content_region.right
        bar._set_btn_label("Send")
        await pilot.pause()

        # Running: Interrupt tracks its own label width.
        bar.agent_running = True
        await pilot.pause()
        assert str(send.label) == "Interrupt"
        assert new.region.width == 0
        assert send.region.width == len("Interrupt") + 4
        assert send.region.x == group.content_region.x
        assert send.region.right == group.content_region.right
        strips = pilot.app.screen._compositor.render_strips()
        row = next(strip.text for strip in strips if "Interrupt" in strip.text)
        after_label = row[row.index("Interrupt") + len("Interrupt") :]
        # Only the button's own padding, the group padding, and the border
        # may follow the label — no dead reserved slot.
        assert after_label.strip("│ ") == ""

        # Retry mode also hugs its label.
        bar.agent_running = False
        bar._retry_label = "Continue"
        bar.retry_mode = True
        await pilot.pause()
        assert send.region.width == len("Continue") + 4
        strips = pilot.app.screen._compositor.render_strips()
        row = next(strip.text for strip in strips if "Continue" in strip.text)
        assert "  Continue  " in row

        # Idle with messages: New appears beside Send, flush right.
        bar.retry_mode = False
        bar.has_messages = True
        await pilot.pause()
        assert send.region.width == len("Send") + 4
        assert new.region.width == len("New") + 4
        assert new.region.right == group.content_region.right
        assert new.region.x == send.region.right + 1

        # The reserved group fits the widest real combination plus its
        # inter-button margin and two one-cell edge paddings.
        assert group.region.width == len("Send") + 4 + 1 + len("New") + 4 + 2


@pytest.mark.asyncio
async def test_status_bar_warn_flash_suppresses_tool_trail() -> None:
    """Warn flashes (shell mode instructions) get the row to themselves."""

    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar()

    async with SBApp().run_test(size=(80, 6)) as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_tool_info("TOOLTRAIL-MARKER")

        sb.flash("regular note")
        await pilot.pause()
        await pilot.pause()
        strips = pilot.app.screen._compositor.render_strips()
        frame = "\n".join(strip.text for strip in strips)
        assert "TOOLTRAIL-MARKER" in frame

        sb.flash("shell mode notice", warn=True)
        await pilot.pause()
        await pilot.pause()
        assert sb.query_one("#status-flash").region.width == sb.query_one(".status-body").content_region.width
        strips = pilot.app.screen._compositor.render_strips()
        frame = "\n".join(strip.text for strip in strips)
        assert "TOOLTRAIL-MARKER" not in frame


@pytest.mark.asyncio
async def test_status_bar_snapshot_restore_preserves_run_state() -> None:
    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar()

    async with SBApp().run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_tool_info("3 tools")
        sb.start_run()
        sb.add_tool_call()
        sb.show("Thinking")
        snapshot = sb.snapshot()

        sb.set_tool_info("")
        sb.start_run()
        sb.show("Restoring Session")
        sb.restore(snapshot)
        await pilot.pause()

        assert sb.status == "Thinking"
        assert sb._start_time == snapshot["start_time"]
        assert sb._tool_count == snapshot["tool_count"]
        assert sb._tool_trail == "3 tools"


@pytest.mark.asyncio
async def test_status_bar_idle_restore_clears_shell_flash_before_tool_info_updates() -> None:
    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar()

    async with SBApp().run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_profile("Code Agent")
        sb.set_tool_info("Original runtime")
        sb.clear_status()
        snapshot = sb.snapshot()

        sb.flash("Shell mode", warn=True)
        sb.restore(snapshot)
        sb.set_tool_info("Updated runtime")
        await pilot.pause()

        assert sb._flash is None
        assert not sb.has_class("-warn")
        tool_info = sb.query_one("#status-tool-info", Static)
        assert tool_info.visible is True
        assert tool_info.render().plain == "Updated runtime"


@pytest.mark.asyncio
async def test_status_bar_selectors_only_restore_does_not_poison_next_run_visibility() -> None:
    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar()

    async with SBApp().run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.hide()
        sb.set_profile("Code Agent")
        snapshot = sb.snapshot()
        assert snapshot["content_shown"] is False
        assert snapshot["idle_shown"] is False

        sb.flash("Shell mode", warn=True)
        sb.restore(snapshot)
        sb.show("Thinking")
        await pilot.pause()

        assert sb.query_one("ChrysLoadingIndicator").visible is True
        assert sb.query_one("#status-text", Static).visible is True
        assert sb.query_one("#status-text", Static).render().plain == "Thinking"


@pytest.mark.asyncio
async def test_status_bar_hidden_restore_recovers_configured_idle_chrome() -> None:
    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar()

    async with SBApp().run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_profile("Code Agent")
        sb.hide()
        snapshot = sb.snapshot()

        sb.flash("Shell mode", warn=True)
        sb.restore(snapshot)
        await pilot.pause()

        assert sb.visible is True
        assert sb._flash is None
        assert sb.query_one("#profile-tag", Static).render().plain == "Code Agent"
        assert sb.query_one("#status-text", Static).visible is False


@pytest.mark.asyncio
async def test_status_bar_tooltip_is_click_hint_only() -> None:
    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar()

    async with SBApp().run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_tool_info("3 tools")

        sb.show("Loading")

        tool_info = sb.query_one("#status-tool-info", Static)
        assert tool_info.tooltip is not None
        assert tool_info.tooltip.plain == "Click for details"


@pytest.mark.asyncio
async def test_status_bar_empty_runtime_trail_removes_mouse_details_entry() -> None:
    from types import SimpleNamespace

    from chrys.app.tui.screens.main.runtime_info import RegistryRuntimeInfoProvider

    messages: list[StatusBar.DetailsClicked] = []

    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar()

        @on(StatusBar.DetailsClicked)
        def on_details_clicked(self, event: StatusBar.DetailsClicked) -> None:
            messages.append(event)

    runtime_info = RegistryRuntimeInfoProvider(  # type: ignore[arg-type]
        SimpleNamespace(agent_registry=None)
    )
    trail = runtime_info.format_tool_info(
        [],
        [],
        runtime_details=AgentRuntimeDetails(
            hook_sources=[
                RuntimeHookSourceDetails(
                    scope="global",
                    hooks=[RuntimeHookDetails(id="disabled", enabled=False)],
                )
            ]
        ),
    )
    assert trail == ()

    async with SBApp().run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_tool_info(trail)
        sb.show("Loading")
        await pilot.pause()

        tool_info = sb.query_one("#status-tool-info", Static)
        assert tool_info.render().plain == ""
        assert tool_info.styles.pointer == "default"
        assert tool_info.tooltip is None

        event = Click(
            sb,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=tool_info.region.x,
            screen_y=tool_info.region.y,
        )
        sb.on_click(event)
        await pilot.pause()

    assert event._no_default_action is False
    assert event._stop_propagation is False
    assert messages == []


@pytest.mark.asyncio
async def test_status_bar_set_tool_info_refreshes_visible_flash_trail() -> None:
    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar()

    async with SBApp().run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.flash("Completed in 1s")
        sb.set_tool_info("11 tools · 1 skill")
        await pilot.pause()

        flash_trail = sb.query_one("#status-flash-trail", Static)
        assert flash_trail.render().plain == "11 tools · 1 skill"


@pytest.mark.asyncio
async def test_status_bar_posts_details_clicked_from_tool_info() -> None:
    messages: list[StatusBar.DetailsClicked] = []

    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar()

        @on(StatusBar.DetailsClicked)
        def on_details_clicked(self, event: StatusBar.DetailsClicked) -> None:
            messages.append(event)

    async with SBApp().run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_tool_info("3 tools")
        sb.show("Loading")
        await pilot.pause()

        tool_info = sb.query_one("#status-tool-info", Static)
        event = Click(
            sb,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=tool_info.region.x,
            screen_y=tool_info.region.y,
        )
        sb.on_click(event)
        await pilot.pause()

    assert event._no_default_action is True
    assert event._stop_propagation is True
    assert len(messages) == 1


@pytest.mark.asyncio
async def test_status_bar_posts_details_clicked_from_flash_trail() -> None:
    messages: list[StatusBar.DetailsClicked] = []

    class SBApp(App):
        def compose(self) -> ComposeResult:
            yield StatusBar()

        @on(StatusBar.DetailsClicked)
        def on_details_clicked(self, event: StatusBar.DetailsClicked) -> None:
            messages.append(event)

    async with SBApp().run_test() as pilot:
        sb = pilot.app.query_one(StatusBar)
        sb.set_tool_info("3 tools")
        sb.flash("Profile: Code", trail="3 tools")
        await pilot.pause()

        flash_trail = sb.query_one("#status-flash-trail", Static)
        event = Click(
            sb,
            x=0,
            y=0,
            delta_x=0,
            delta_y=0,
            button=1,
            shift=False,
            meta=False,
            ctrl=False,
            screen_x=flash_trail.region.x,
            screen_y=flash_trail.region.y,
        )
        sb.on_click(event)
        await pilot.pause()

    assert event._no_default_action is True
    assert event._stop_propagation is True
    assert len(messages) == 1


# ---------------------------------------------------------------------------
# ToolCall / ToolGroup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_call_lifecycle() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield ToolCall("c1", "read_file", 'path="/foo"', args={"path": "/foo"})

    async with ToolApp().run_test() as pilot:
        tc = pilot.app.query_one(ToolCall)
        header = tc.query_one(ToolCardHeader)
        assert tc.status == "running"
        assert header.actions_visible is False

        tc.set_complete("file contents", duration_ms=89)
        await pilot.pause()
        assert tc.status == "complete"
        assert tc.duration_ms == 89
        assert header.actions_visible is True

        # Label should contain tool name and duration
        label = tc.query_one("#tc-label", ToolCardHeader).content
        assert "read_file" in label.plain
        assert "89ms" in label.plain


@pytest.mark.asyncio
async def test_tool_card_header_zones_disabled_when_narrower_than_actions_cell() -> None:
    """A header too narrow for the 13-cell actions cell must map no zones.

    Without the clamp, ``offset.x - (width - _ACTIONS_WIDTH)`` shifts clicks
    *into* the zone ranges (e.g. x=0 in a 10-wide header lands in "view"),
    so ordinary label clicks in narrow panes would fire actions.
    """

    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            tc = ToolCall("c1", "read_file", 'path="/foo"', args={"path": "/foo"})
            tc.styles.width = 10
            yield tc

    async with ToolApp().run_test() as pilot:
        tc = pilot.app.query_one(ToolCall)
        tc.set_complete("file contents", duration_ms=89)
        await pilot.pause()
        header = tc.query_one(ToolCardHeader)
        assert header.actions_visible is True
        width = header.content_size.width
        assert 0 < width < ToolCardHeader._ACTIONS_WIDTH
        assert all(header._zone_at(Offset(x, 0)) is None for x in range(width))


@pytest.mark.asyncio
async def test_tool_card_header_renders_plain_string_labels_without_markup() -> None:
    """A str label quoting model output like ``[/]`` must not raise MarkupError.

    All renderers pass Rich ``Text``, but the header's contract is that plain
    strings render literally too — both at construction and via ``update()``.
    """

    class HeaderApp(App):
        def compose(self) -> ComposeResult:
            yield ToolCardHeader("stray close [/] and [bold] unclosed")

    async with HeaderApp().run_test(size=(60, 4)) as pilot:
        header = pilot.app.query_one(ToolCardHeader)
        assert header.render_line(0).text.rstrip() == "stray close [/] and [bold] unclosed"

        header.update("updated [/][red] later")
        await pilot.pause()
        assert header.render_line(0).text.rstrip() == "updated [/][red] later"


@pytest.mark.parametrize(("duration_ms", "expected"), [(0, "(0ms)"), (987, "(987ms)")])
async def test_tool_card_header_appends_replay_timing_omitted_by_renderer(duration_ms: int, expected: str) -> None:
    class HeaderApp(App):
        def compose(self) -> ComposeResult:
            yield ToolCardHeader(Text("• hosted_tool", style="bold"))

    async with HeaderApp().run_test() as pilot:
        header = pilot.app.query_one(ToolCardHeader)
        header.append_replay_timing(
            timestamp="- 9:02 AM",
            duration_ms=duration_ms,
            show_duration=True,
        )
        await pilot.pause()

        assert header._label_renderable().plain == f"• hosted_tool {expected} - 9:02 AM"


@pytest.mark.asyncio
async def test_tool_call_renders_result_images_without_placeholder_text() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield ToolCall("c1", "view_image", args={"path": "/tmp/pixel.png"})

    async with ToolApp().run_test() as pilot:
        tc = pilot.app.query_one(ToolCall)
        image = Content.from_data(
            _png_bytes((255, 0, 0), size=(4, 4)),
            "image/png",
            additional_properties={"width": 4, "height": 4, "media_type": "image/png"},
        )

        tc.set_complete("Image: segmentation mask\n[image/png image]", duration_ms=12, image_contents=[image])
        await pilot.pause()

        output = tc.query_one("#tc-body", Static)
        assert output.render().plain == "Image: segmentation mask"
        assert "data:image" not in output.render().plain
        panel = tc.query_one("#tc-panel")
        assert str(panel.border_subtitle) == "Resolution: 4x4 · Type: image/png"
        assert len(list(tc.query("#tc-images"))) == 1


@pytest.mark.asyncio
async def test_tool_copy_button_copies_full_input_and_output(monkeypatch: pytest.MonkeyPatch) -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield ToolCall("c1", "read_file", args={"path": "/foo", "limit": 3})

    async with ToolApp().run_test() as pilot:
        copied: list[str] = []
        monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", copied.append)

        tc = pilot.app.query_one(ToolCall)
        full_output = "file contents\nwith a ``` fence\nand more text"
        tc.set_complete(full_output, duration_ms=89)
        await pilot.pause()

        _click_copy_button(tc.query_one(ToolCardHeader))
        await pilot.pause()

        assert copied
        assert len(copied) == 1
        payload = copied[-1]
        assert pilot.app.clipboard == payload
        assert "# read_file" in payload
        assert "- **Status:** `completed`" in payload
        assert "- **Duration:** `89ms`" in payload
        assert "- **Call ID:** `c1`" in payload
        assert '"path": "/foo"' in payload
        assert '"limit": 3' in payload
        assert full_output in payload
        assert "~~~text" in payload


@pytest.mark.asyncio
async def test_tool_copy_button_sanitizes_control_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    class RawPayloadToolCall(ToolCall):
        def _tool_copy_input(self) -> tuple[str, str]:
            return "text", "query\tvalue\nnul:\x00 cr:\rend"

        def _tool_copy_sections(self) -> list[tuple[str, str, str]]:
            return [("Output", "text", "color:\x1b[31mred\x1b[0m csi:\x9b31m\nnext\tcell\x7f")]

    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield RawPayloadToolCall("c1", "raw_payload")

    async with ToolApp().run_test() as pilot:
        copied: list[str] = []
        monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", copied.append)

        tc = pilot.app.query_one(RawPayloadToolCall)
        tc.set_complete("ignored", duration_ms=1)
        await pilot.pause()

        _click_copy_button(tc.query_one(ToolCardHeader))
        await pilot.pause()

        assert len(copied) == 1
        payload = copied[-1]
        assert "query\tvalue\nnul:" in payload
        assert "\nnext\tcell" in payload
        assert "\\x00" in payload
        assert "\\x0d" in payload
        assert "\\x1b[31mred\\x1b[0m" in payload
        assert "\\x9b31m" in payload
        assert "\\x7f" in payload
        assert "\x00" not in payload
        assert "\r" not in payload
        assert "\x1b" not in payload
        assert "\x9b" not in payload
        assert "\x7f" not in payload


@pytest.mark.asyncio
async def test_tool_copy_large_payload_skips_terminal_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield ToolCall("c1", "zsh", args={"command": "yes"})

    async with ToolApp().run_test() as pilot:
        copied: list[str] = []
        monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", copied.append)

        tc = pilot.app.query_one(ToolCall)
        tc.set_complete("x" * (70 * 1024))
        await pilot.pause()

        _click_copy_button(tc.query_one(ToolCardHeader))
        await pilot.pause()

        assert copied
        assert len(copied) == 1
        assert pilot.app.clipboard != copied[-1]


@pytest.mark.asyncio
async def test_tool_copy_small_payload_reports_terminal_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    class ToolApp(_LocalizedApp):
        def __init__(self) -> None:
            super().__init__()
            self.notifications: list[str] = []

        def compose(self) -> ComposeResult:
            yield ToolCall("c1", "read_file", args={"path": "/foo"})

        def copy_to_clipboard(self, _text: str) -> None:
            raise RuntimeError("terminal clipboard unavailable")

        def notify(self, message: str, **_kwargs: object) -> None:  # type: ignore[override]
            self.notifications.append(message)

    async with ToolApp().run_test() as pilot:
        copied: list[str] = []
        monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", copied.append)

        tc = pilot.app.query_one(ToolCall)
        tc.set_complete("file contents")
        await pilot.pause()

        _click_copy_button(tc.query_one(ToolCardHeader))
        await pilot.pause()

        assert copied
        assert "terminal clipboard unavailable" in pilot.app.notifications[-1]


@pytest.mark.asyncio
async def test_file_tool_copy_includes_snapshot_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    async with ChatPanelApp().run_test() as pilot:
        copied: list[str] = []
        monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", copied.append)

        cp = pilot.app.query_one(ChatPanel)
        await cp.add_tool_start("c1", "write_file", "filesystem.write", args={"path": "README.md"})
        await pilot.pause()
        header = pilot.app.query_one(ToolCardHeader)
        assert header.actions_visible is False

        await cp.add_tool_result(
            "c1",
            "write_file",
            "Successfully wrote README.md",
            20,
            file_snapshot=("old\n", "new\n"),
        )
        await pilot.pause()
        assert header.actions_visible is True

        _click_copy_button(header)
        await pilot.pause()

        assert len(copied) == 1
        payload = copied[-1]
        assert "## Diff" in payload
        assert "~~~diff" in payload
        assert "--- before/README.md" in payload
        assert "+++ after/README.md" in payload
        assert "-old" in payload
        assert "+new" in payload


@pytest.mark.asyncio
async def test_sub_agent_renders_task_prompt_as_markdown_in_dedicated_panel() -> None:
    prompt = "## Investigate\n\n- check **tests**"

    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": prompt})

    async with ToolApp().run_test() as pilot:
        await pilot.pause()
        tc = pilot.app.query_one(SubAgentToolCall)

        task_panel = tc.query_one("#sa-task-panel")
        border_title = task_panel.border_title
        title = border_title.plain if isinstance(border_title, Text) else str(border_title)
        assert title == "Task"

        task = tc.query_one("#sa-task", VirtualizedMarkdown)
        assert task.source == prompt
        assert tc.query_one("#sa-main", Static).render().plain == ""

        await tc.add_inner_tool_start("inner1", "read_file", {"path": "README.md"})
        await pilot.pause()

        main = tc.query_one("#sa-main", Static).render().plain
        assert "read_file" in main
        assert prompt not in main


def test_sub_agent_renderer_uses_base_tool_card_contract() -> None:
    tool = SubAgentToolCall("c1", "Explore", args={"prompt": "investigate usage"})

    assert isinstance(tool, BaseToolCard)
    assert tool.call_id == "c1"
    assert tool.tool_name == "Explore"
    assert tool.status == "running"
    assert tool.result_text == ""
    assert tool.duration_ms == 0
    assert tool.args == {"prompt": "investigate usage"}


@pytest.mark.asyncio
async def test_sub_agent_inner_tool_start_upserts_for_lifetime_and_ignores_late_terminal_reemit() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Remote", args={"prompt": "delegate"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)
        tool.update_progress(4, 100, 250, 1)

        await tool.add_inner_tool_start("inner-0", "pending", {"value": 1})
        await tool.add_inner_tool_start("inner-0", "renamed", {"value": 2}, tool_kind=KIND_SHELL)
        assert tool._progress_tool_calls == 4
        assert tool._total_inner_calls == 1
        assert tool._inner_tools["inner-0"].tool_name == "renamed"
        assert tool._inner_tools["inner-0"].args_summary == "value: 2"

        for index in range(1, 8):
            await tool.add_inner_tool_start(f"inner-{index}", f"tool-{index}", {})
        assert "inner-0" not in tool._inner_tools
        assert tool._total_inner_calls == 8

        await tool.add_inner_tool_start("inner-0", "terminal-title", {"value": 3})
        assert tool._total_inner_calls == 8
        assert "inner-0" in tool._inner_tools
        tool.complete_inner_tool("inner-0", "done", 15)
        await tool.add_inner_tool_start("inner-new", "new", {})
        assert "inner-0" not in tool._inner_tools

        await tool.add_inner_tool_start("inner-0", "late", {"value": 4})
        assert "inner-0" not in tool._inner_tools
        assert tool._total_inner_calls == 9
        assert "Spend: 250 tokens · 1 unreported attempt" in tool._render_subtitle()


@pytest.mark.asyncio
async def test_sub_agent_inner_hosted_updates_mutate_the_existing_nested_entry() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": "search"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)
        await tool.add_inner_tool_start("hosted:1", "web_search", {"query": "old"}, tool_kind=KIND_SEARCH)

        tool.update_inner_tool_args("hosted:1", {"query": "Chrys"})
        tool.update_inner_tool_progress(
            "hosted:1",
            ["Searching documentation"],
            image_contents=[{"uri": "data:image/png;base64,AAA"}],
        )

        entry = tool._inner_tools["hosted:1"]
        assert entry.args_summary == 'query: "Chrys"'
        assert entry.progress_summary == "Searching documentation"
        assert entry.image_count == 1

        tool.update_inner_tool_status("hosted:1", "interrupted")
        assert entry.status == "interrupted"
        assert "hosted:1" not in tool._terminal_inner_call_ids

        tool.complete_inner_tool(
            "hosted:1",
            "Error: interrupted by provider",
            25,
            image_contents=[{"uri": "data:image/png;base64,BBB"}],
            artifacts=[{"name": "partial.csv"}],
            metadata={TOOL_FAILED_METADATA_KEY: True},
        )
        assert "hosted:1" in tool._terminal_inner_call_ids
        assert entry.status == "error"
        assert entry.duration_ms == 25
        assert entry.image_count == 1
        assert entry.artifact_count == 1
        assert tool._completed_inner_calls == 1


@pytest.mark.asyncio
async def test_sub_agent_inner_completed_snapshot_accepts_late_authoritative_result() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": "generate"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)
        await tool.add_inner_tool_start("hosted:1", "image_generation", {})

        tool.update_inner_tool_status("hosted:1", "completed", metadata={"result_text": "snapshot"})
        entry = tool._inner_tools["hosted:1"]
        assert entry.status == "complete"
        assert "hosted:1" not in tool._terminal_inner_call_ids
        assert tool._completed_inner_calls == 1

        tool.complete_inner_tool(
            "hosted:1",
            "authoritative",
            25,
            image_contents=[{"uri": "data:image/png;base64,AAA"}],
            artifacts=[{"name": "report.csv"}],
            metadata={"provider_item_type": "image_generation_call"},
        )

        assert "hosted:1" in tool._terminal_inner_call_ids
        assert entry.duration_ms == 25
        assert entry.image_count == 1
        assert entry.artifact_count == 1
        assert tool._completed_inner_calls == 1
        assert "[1 artifact]" in tool.query_one("#sa-main", Static).render().plain


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["complete", "error"])
async def test_sub_agent_terminal_card_frees_inner_call_id_tracking(terminal: str) -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Remote", args={"prompt": "delegate"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)
        for index in range(3):
            await tool.add_inner_tool_start(f"inner-{index}", f"tool-{index}", {})
            tool.complete_inner_tool(f"inner-{index}", "done", 5)
        # Completion frees the seen-set slot immediately: the terminal
        # guard short-circuits duplicate starts before the seen-check.
        assert tool._seen_inner_call_ids == set()
        assert tool._terminal_inner_call_ids == {"inner-0", "inner-1", "inner-2"}
        assert tool._total_inner_calls == 3

        if terminal == "complete":
            tool.set_complete("all done")
        else:
            tool.set_error("boom")
        # A finished card stays mounted for the session — it must not pin
        # per-call id sets (thousands per ACP attempt, fresh ids per retry).
        assert tool._seen_inner_call_ids == set()
        assert tool._terminal_inner_call_ids == set()
        assert tool._inner_tools == {}
        assert tool._total_inner_calls == 3


@pytest.mark.asyncio
async def test_sub_agent_resume_after_pause_resets_attempt_local_inner_call_tracking() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Remote", args={"prompt": "delegate"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)
        await tool.add_inner_tool_start("inner-0", "first-attempt", {})
        tool.complete_inner_tool("inner-0", "done", 5)
        await tool.add_inner_tool_start("inner-1", "still-running", {})
        assert tool._total_inner_calls == 2

        tool.set_paused("acp_transport", "connection closed", 1)
        tool.set_resumed_after_pause()
        # Inner-call tracking is attempt-local: the dead attempt's stream is
        # already drained, and a fresh attempt may legitimately reuse raw
        # call ids. Nothing from the old feed may survive the resume.
        assert tool._seen_inner_call_ids == set()
        assert tool._terminal_inner_call_ids == set()
        assert tool._inner_tools == {}

        # A reused id must render as a brand-new running call — the stale
        # terminal guard must not swallow it.
        await tool.add_inner_tool_start("inner-0", "reused-id", {})
        assert tool._inner_tools["inner-0"].tool_name == "reused-id"
        assert tool._inner_tools["inner-0"].status == "running"
        # Spend counters stay cumulative across attempts.
        assert tool._total_inner_calls == 3


@pytest.mark.asyncio
async def test_sub_agent_acp_transport_pause_shows_ui_only_diagnostic_banner() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Remote", args={"prompt": "delegate"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)
        tool.set_paused(
            "acp_transport",
            "connection closed",
            2,
            "/workspace/.chrys/sessions/s1/approvals/acp.log",
        )
        pause_info = tool.query_one("#sa-pause-info", Static)
        await wait_for(
            lambda: "External ACP transport interrupted" in pause_info.render().plain,
            pilot=pilot,
            description="ACP transport pause banner",
        )

        rendered = pause_info.render().plain
        assert "connection closed" in rendered
        assert "after 2 auto-retry attempts" in rendered
        assert "Diagnostics: /workspace/.chrys/sessions/s1/approvals/acp.log" in rendered


@pytest.mark.asyncio
async def test_sub_agent_pause_diagnostic_path_display_copy_is_surrogate_safe() -> None:
    from chrys.foundation.platform.files import surrogate_safe_text

    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Remote", args={"prompt": "delegate"})

    surrogate_path = "/workspace/pro" + chr(0xDCFF) + "ject/acp.log"
    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)
        tool.set_paused("acp_transport", "connection closed", 1, surrogate_path)
        pause_info = tool.query_one("#sa-pause-info", Static)
        await wait_for(
            lambda: "Diagnostics:" in pause_info.render().plain,
            pilot=pilot,
            description="surrogate diagnostic line",
        )

        rendered = pause_info.render().plain
        assert surrogate_path not in rendered
        assert f"Diagnostics: {surrogate_safe_text(surrogate_path)}" in rendered
        rendered.encode("utf-8")  # the display copy must strict-encode


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({SHELL_EXIT_CODE_METADATA_KEY: 1}, "bash"),
        ({SHELL_TIMED_OUT_METADATA_KEY: True}, "bash"),
    ],
)
async def test_sub_agent_inner_shell_failure_rows_use_metadata(metadata: dict[str, object], expected: str) -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": "investigate usage"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)

        await tool.add_inner_tool_start("inner1", "bash", {"command": "false"}, tool_kind=KIND_SHELL)
        tool.complete_inner_tool("inner1", "normal-looking output", 25, metadata=metadata)
        await pilot.pause()

        assert tool._inner_tools["inner1"].status == "error"
        main = tool.query_one("#sa-main", Static).render().plain
        assert f"\u2717 {expected}" in main


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_kind", [KIND_MCP, ""])
async def test_sub_agent_inner_external_error_text_is_not_failure_without_metadata(tool_kind: str) -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": "investigate usage"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)

        await tool.add_inner_tool_start("inner1", "remote_lookup", {"query": "status"}, tool_kind=tool_kind)
        tool.complete_inner_tool("inner1", "Error: expected text from remote tool", 25)
        await pilot.pause()

        assert tool._inner_tools["inner1"].status == "complete"
        main = tool.query_one("#sa-main", Static).render().plain
        assert "\u2713 remote_lookup" in main


@pytest.mark.asyncio
async def test_sub_agent_inner_unkinded_tool_structured_failure_renders_error() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": "investigate usage"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)

        await tool.add_inner_tool_start("inner1", "run_skill_script", {"skill_name": "docs"}, tool_kind="")
        tool.complete_inner_tool("inner1", "Error: script failed", 25, metadata={TOOL_FAILED_METADATA_KEY: True})
        await pilot.pause()

        assert tool._inner_tools["inner1"].status == "error"
        main = tool.query_one("#sa-main", Static).render().plain
        assert "\u2717 run_skill_script" in main


@pytest.mark.asyncio
async def test_sub_agent_inner_legacy_unkinded_chrys_tool_error_text_renders_error() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": "investigate usage"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)

        await tool.add_inner_tool_start("inner1", "run_skill_script", {"skill_name": "docs"}, tool_kind="")
        tool.complete_inner_tool("inner1", "Error: literal remote payload", 25)
        await pilot.pause()

        assert tool._inner_tools["inner1"].status == "error"
        main = tool.query_one("#sa-main", Static).render().plain
        assert "\u2717 run_skill_script" in main


@pytest.mark.asyncio
async def test_sub_agent_structured_success_suppresses_error_text_fallback() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": "explain errno"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)

        tool.set_complete(
            "Error: EACCES is returned when permission is denied.",
            25,
            metadata={TOOL_FAILED_METADATA_KEY: False},
        )
        await pilot.pause()

        assert tool.status == "complete"
        assert tool.has_class("-complete")
        assert not tool.has_class("-error")


@pytest.mark.asyncio
async def test_sub_agent_structured_failure_renders_error_without_error_prefix() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": "investigate"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)

        tool.set_complete(
            "sub-agent failed after retries",
            25,
            metadata={TOOL_FAILED_METADATA_KEY: True},
        )
        await pilot.pause()

        assert tool.status == "error"
        assert tool.has_class("-error")
        assert not tool.has_class("-complete")


@pytest.mark.asyncio
async def test_sub_agent_inner_hook_denied_renders_rejected_not_errored() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": "investigate usage"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(SubAgentToolCall)

        await tool.add_inner_tool_start("inner1", "read_file", {"path": "secret.txt"}, tool_kind=KIND_FILESYSTEM_READ)
        tool.complete_inner_tool(
            "inner1",
            "Error: blocked by policy",
            25,
            metadata={TOOL_FAILED_METADATA_KEY: True, TOOL_ERROR_KIND_METADATA_KEY: "hook_denied"},
        )
        await pilot.pause()

        assert tool._inner_tools["inner1"].status == "rejected"
        main = tool.query_one("#sa-main", Static).render().plain
        assert "\u2717 read_file" in main


@pytest.mark.asyncio
async def test_sub_agent_large_result_defers_markdown_until_expand(monkeypatch: pytest.MonkeyPatch) -> None:
    """Large sub-agent results should not parse markdown if the group auto-collapses."""

    large_result = "# Result\n\n" + "\n".join(f"- item {idx}" for idx in range(5000))
    markdown_updates: list[str] = []
    original_update = VirtualizedMarkdown.update

    def counted_update(self: VirtualizedMarkdown, markdown: str):
        markdown_updates.append(markdown)
        return original_update(self, markdown)

    monkeypatch.setattr(VirtualizedMarkdown, "update", counted_update)
    register_kind_renderer(KIND_SUB_AGENT, SubAgentToolCall)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        cp.set_tool_kinds({"explore_agent": KIND_SUB_AGENT})

        await cp.add_user_message("delegate")
        await cp.add_tool_start("sa1", "explore_agent", KIND_SUB_AGENT, args={"prompt": "investigate"})
        await cp.add_tool_result("sa1", "explore_agent", large_result, 2500)
        await cp.add_agent_message("done", is_final=True)
        await pilot.pause(0.3)

        assert large_result not in markdown_updates

        group = cp.query_one(ToolGroup)
        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if large_result in markdown_updates:
                break

        assert large_result in markdown_updates


@pytest.mark.asyncio
async def test_sub_agent_in_flight_result_render_clears_when_collapsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A large result that finishes rendering after collapse should not stay resident while hidden."""

    large_result = "# Result\n\n" + "\n".join(f"- item {idx}" for idx in range(5000))
    render_started = asyncio.Event()
    release_render = asyncio.Event()
    original_update = VirtualizedMarkdown.update

    def delayed_update(self: VirtualizedMarkdown, markdown: str) -> AwaitComplete:
        if self.id == "sa-result" and markdown == large_result:

            async def await_update() -> None:
                render_started.set()
                await release_render.wait()
                await original_update(self, markdown)

            return AwaitComplete(await_update())
        return original_update(self, markdown)

    monkeypatch.setattr(VirtualizedMarkdown, "update", delayed_update)
    register_kind_renderer(KIND_SUB_AGENT, SubAgentToolCall)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        cp.set_tool_kinds({"explore_agent": KIND_SUB_AGENT})

        await cp.add_user_message("delegate")
        await cp.add_tool_start("sa1", "explore_agent", KIND_SUB_AGENT, args={"prompt": "investigate"})
        await cp.add_tool_result("sa1", "explore_agent", large_result, 2500)
        await cp.add_tool_start("sibling", "plain_tool", "", args={"value": 1})

        group = cp.query_one(ToolGroup)
        tc = cp.query_one(SubAgentToolCall)
        result_markdown = tc.query_one("#sa-result", VirtualizedMarkdown)
        if tc._result_timer is not None:
            tc._result_timer.stop()
            tc._result_timer = None

        render_task = asyncio.create_task(tc.mount_pending_content())
        try:
            # Generous timeouts: the render completes in <1s locally, but a
            # loaded CI worker (-n 8 xdist) can starve the task scheduler well
            # past a tight bound. The 60s per-test timeout is the real backstop
            # for a genuine hang; these only guard against waiting forever.
            await asyncio.wait_for(render_started.wait(), timeout=10)
            group.collapsed = True
            # Do not wait for the whole screen here. Windows CI can starve Pilot's
            # screen-drain wait, and the state we care about is polled below.
            await asyncio.sleep(0)
            release_render.set()
            await asyncio.wait_for(render_task, timeout=30)
        finally:
            release_render.set()
            if not render_task.done():
                render_task.cancel()
                with suppress(asyncio.CancelledError):
                    await render_task

        for _ in range(300):
            await pilot.pause()
            if result_markdown.source == "" and tc._result_pending and not tc.has_class("-done"):
                break

        assert result_markdown.source == ""
        assert not result_markdown._blocks
        assert tc._result_pending is True
        assert not tc.has_class("-done")

        group.collapsed = False
        for _ in range(300):
            await pilot.pause()
            if result_markdown.source == large_result and tc.has_class("-done"):
                break

        assert result_markdown.source == large_result
        assert tc._result_pending is False
        assert tc.has_class("-done")


def test_sub_agent_rejected_large_result_releases_collapsed_content() -> None:
    tc = SubAgentToolCall("c1", "Explore", args={"prompt": "investigate"})
    tc.status = "rejected"
    tc.result_text = "# Rejected\n\n" + "\n".join(f"- reason {idx}" for idx in range(5000))
    tc.add_class("-done")

    tc.release_collapsed_content()

    assert tc._result_pending is True
    assert not tc.has_class("-done")


@pytest.mark.asyncio
async def test_sub_agent_rejected_lazy_completion_clears_inner_tools_and_defers_result() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": "investigate"})

    async with ToolApp().run_test() as pilot:
        tc = pilot.app.query_one(SubAgentToolCall)
        await tc.add_inner_tool_start("inner1", "read_file", {"path": "a.py"}, tool_kind=KIND_FILESYSTEM_READ)
        tc.set_complete(
            "# Rejected\n\n" + "\n".join(f"- reason {idx}" for idx in range(5000)),
            approval="user_rejected",
            lazy=True,
        )
        await pilot.pause()

        assert tc.status == "rejected"
        assert not tc._inner_tools
        assert tc._result_pending is True


@pytest.mark.asyncio
async def test_sub_agent_update_args_refreshes_task_prompt() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": "old prompt"})

    async with ToolApp().run_test() as pilot:
        await pilot.pause()
        tc = pilot.app.query_one(SubAgentToolCall)

        tc.update_args({"prompt": "new prompt"})
        await pilot.pause()

        task = tc.query_one("#sa-task", VirtualizedMarkdown)
        assert task.source == "new prompt"


@pytest.mark.asyncio
async def test_sub_agent_update_args_mounts_task_prompt_when_added() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={})

    async with ToolApp().run_test() as pilot:
        await pilot.pause()
        tc = pilot.app.query_one(SubAgentToolCall)
        assert not list(tc.query("#sa-task-panel"))

        tc.update_args({"prompt": "new prompt"})
        await pilot.pause()

        task_panel = tc.query_one("#sa-task-panel")
        border_title = task_panel.border_title
        assert (border_title.plain if isinstance(border_title, Text) else str(border_title)) == "Task"
        task = tc.query_one("#sa-task", VirtualizedMarkdown)
        assert task.source == "new prompt"


@pytest.mark.asyncio
async def test_sub_agent_update_args_removes_task_prompt_when_cleared() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": "old prompt"})

    async with ToolApp().run_test() as pilot:
        await pilot.pause()
        tc = pilot.app.query_one(SubAgentToolCall)
        assert list(tc.query("#sa-task-panel"))

        tc.update_args({"prompt": ""})
        await pilot.pause()

        assert not list(tc.query("#sa-task-panel"))


@pytest.mark.asyncio
async def test_sub_agent_rejected_title_omits_status_icon() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "explore_agent", args={"prompt": "inspect"})

    async with ToolApp().run_test() as pilot:
        await pilot.pause()
        tc = pilot.app.query_one(SubAgentToolCall)

        tc.set_complete("Error: Tool execution was rejected by user.", approval="user_rejected")
        await pilot.pause()

        assert tc.approval == "user_rejected"
        assert tc.has_class("-rejected")
        panel = tc.query_one("#sa-panel")
        title = panel.border_title
        assert (title.plain if isinstance(title, Text) else str(title)) == "explore_agent"
        assert panel.border_subtitle == "Rejected"


@pytest.mark.asyncio
async def test_sub_agent_omits_task_panel_when_prompt_is_empty() -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={})

    async with ToolApp().run_test() as pilot:
        await pilot.pause()
        tc = pilot.app.query_one(SubAgentToolCall)

        assert not list(tc.query("#sa-task-panel"))
        assert tc.query_one("#sa-main", Static).render().plain == ""


@pytest.mark.asyncio
async def test_sub_agent_copy_uses_prompt_input(monkeypatch: pytest.MonkeyPatch) -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": "investigate the tests"})

    async with ToolApp().run_test() as pilot:
        copied: list[str] = []
        monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", copied.append)

        tc = pilot.app.query_one(SubAgentToolCall)
        header = tc.query_one(ToolCardHeader)
        assert header.actions_visible is False

        tc.set_complete("final **answer**", duration_ms=100)
        await pilot.pause()
        assert header.actions_visible is True

        _click_copy_button(header)
        await pilot.pause()

        assert len(copied) == 1
        payload = copied[-1]
        assert "investigate the tests" in payload
        assert '"prompt"' not in payload
        assert "~~~markdown" in payload
        assert "final **answer**" in payload


def test_sub_agent_running_subtitle_includes_ctx_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 1_000.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    tc = SubAgentToolCall("c1", "Explore", args={"prompt": "investigate usage"})

    tc._progress_tool_calls = 2
    tc._progress_ctx_tokens = 34_000
    tc._progress_total_usage_tokens = 46_000
    tc._start_time = now - 123

    assert tc._render_subtitle() == "Tool calls: 2 · Ctx: 34.0k tokens · Spend: 46.0k tokens · Duration: 2m 3s"


def test_sub_agent_running_subtitle_hides_subsecond_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    now = 1_000.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    tc = SubAgentToolCall("c1", "Explore", args={"prompt": "investigate usage"})

    tc._progress_tool_calls = 1
    tc._progress_ctx_tokens = 900
    tc._progress_total_usage_tokens = 900
    tc._start_time = now - 0.5

    assert tc._render_subtitle() == "Tool calls: 1 · Ctx: 900 tokens · Spend: 900 tokens"


def test_sub_agent_done_subtitle_keeps_ctx_tokens() -> None:
    tc = SubAgentToolCall("c1", "Explore", args={"prompt": "investigate usage"})
    tc.status = "complete"
    tc.duration_ms = 132_000
    tc._total_inner_calls = 21
    tc._progress_ctx_tokens = 57_700
    tc._progress_total_usage_tokens = 236_100

    assert tc._render_subtitle() == "Tool calls: 21 · Ctx: 57.7k tokens · Spend: 236.1k tokens · Duration: 2m 12s"


def test_sub_agent_subtitle_shows_compaction_count_only_after_commit() -> None:
    tc = SubAgentToolCall("c1", "Explore", args={"prompt": "investigate usage"})
    tc._progress_tool_calls = 33
    tc._progress_ctx_tokens = 74_400

    # No compaction yet \u2014 no Compactions segment.
    assert tc._render_subtitle().startswith("Tool calls: 33 \u00b7 Ctx: 74.4k tokens")
    assert "Compactions:" not in tc._render_subtitle()

    tc.add_compaction_start("comp-1")
    assert "Compactions:" not in tc._render_subtitle()

    # finished(ok) alone doesn't count \u2014 the round may still be abandoned
    # by a failed spill write; only the committed signal counts.
    tc.complete_compaction("comp-1", outcome="ok", duration_ms=1_000)
    assert "Compactions:" not in tc._render_subtitle()

    tc.record_compaction_committed("comp-1")
    assert "Compactions: 1" in tc._render_subtitle()

    # Failed / canceled compactions never emit the committed signal.
    tc.add_compaction_start("comp-2")
    tc.complete_compaction("comp-2", outcome="failed", failure_reason="boom")
    tc.add_compaction_start("comp-3")
    tc.complete_compaction("comp-3", outcome="canceled")
    assert "Compactions: 1" in tc._render_subtitle()

    tc.add_compaction_start("comp-4")
    tc.complete_compaction("comp-4", outcome="ok", duration_ms=1_000)
    tc.record_compaction_committed("comp-4")
    assert "Compactions: 2" in tc._render_subtitle()

    # The committed signal is independent of the feed entry \u2014 an evicted
    # or never-seen line still counts.
    tc.record_compaction_committed("comp-unseen")
    assert "Compactions: 3" in tc._render_subtitle()


@pytest.mark.asyncio
async def test_sub_agent_error_copy_preserves_result_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield SubAgentToolCall("c1", "Explore", args={"prompt": "investigate the failure"})

    async with ToolApp().run_test() as pilot:
        copied: list[str] = []
        monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", copied.append)

        tc = pilot.app.query_one(SubAgentToolCall)
        tc.tool_kind = KIND_SUB_AGENT
        tc.set_complete("Error: sub-agent failed", duration_ms=1234)
        await pilot.pause()

        assert tc.duration_ms == 1234
        header = tc.query_one(ToolCardHeader)
        assert header.actions_visible is True

        _click_copy_button(header)
        await pilot.pause()

        assert len(copied) == 1
        payload = copied[-1]
        assert "- **Status:** `error`" in payload
        assert "- **Duration:** `1s`" in payload
        assert "Error: sub-agent failed" in payload


@pytest.mark.asyncio
async def test_tool_group_add_and_complete() -> None:
    class GroupApp(App):
        def compose(self) -> ComposeResult:
            yield ToolGroup()

    async with GroupApp().run_test() as pilot:
        tg = pilot.app.query_one(ToolGroup)
        await tg.add_tool("c1", "read_file", 'path="/a"')
        await tg.add_tool("c2", "grep", 'pattern="foo"')
        assert not tg.all_complete

        tg.complete_tool("c1", "result1", 100)
        tg.complete_tool("c2", "result2", 200)
        assert tg.all_complete


@pytest.mark.parametrize(
    ("tool_name", "tool_kind", "args", "result"),
    [
        ("zsh", "shell", {"command": "echo hi"}, "hi\n[exit_code: 0]"),
        ("read_file", "filesystem.read", {"path": "/tmp/a.txt"}, "File: /tmp/a.txt\n1: hello"),
        ("grep", "search", {"pattern": "hello"}, "Found 1 match in /tmp/a.txt\n/tmp/a.txt:1:hello"),
    ],
)
@pytest.mark.asyncio
async def test_specialized_tool_copy_button_visibility(
    tool_name: str,
    tool_kind: str,
    args: dict[str, str],
    result: str,
) -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_tool_start("c1", tool_name, tool_kind, args=args)
        await pilot.pause()

        header = pilot.app.query_one(ToolCardHeader)
        assert header.actions_visible is False

        await cp.add_tool_result("c1", tool_name, result, 25)
        await pilot.pause()

        assert header.actions_visible is True


# ---------------------------------------------------------------------------
# ChatPanel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("turn_range", "expected_title"),
    [
        ((3, 5), "  ═══ Compressed (Turn 3-5) ═══"),
        ((7, 7), "  ═══ Compressed (Turn 7) ═══"),
        ((0, 0), "  ═══ Compressed (5,000 messages) ═══"),
    ],
)
def test_context_fold_title_includes_turn_range(
    turn_range: tuple[int, int],
    expected_title: str,
) -> None:
    widget = ContextFoldWidget("ctx_abc", "summary", 5000, turn_range)

    assert widget.render().plain.splitlines()[0] == expected_title


@pytest.mark.asyncio
async def test_chat_panel_add_messages() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("hello")
        await cp.add_agent_message("hi there")
        await cp.add_tool_start("c1", "read_file", "path=/foo")
        await cp.add_tool_result("c1", "read_file", "file contents...", 89)
        await cp.add_error("something went wrong")
        await cp.add_system("session started")
        await cp.add_context_fold("rw_abc", "summary of folded messages", 5000, (3, 5))
        # The compaction is retry output, so it removes the stale error even
        # when an informational system row was appended in between.
        assert len(_chat_content_children(cp)) == 5
        assert not list(cp.query(ErrorMessage))
        assert len(list(cp.query(SystemMessage))) == 1
        fold = cp.query_one(ContextFoldWidget)
        assert fold.render().plain.splitlines()[0] == "  ═══ Compressed (Turn 3-5) ═══"


@pytest.mark.asyncio
async def test_chat_panel_removes_failed_prompt_from_toc() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("bad prompt")
        await cp.add_error("failed before output")

        await cp.add_user_message("replacement prompt")
        await pilot.pause()

        items = cp.toc_items
        assert [(item.turn_id, item.summary, item.turn_index) for item in items] == [
            ("turn-1", "replacement prompt", 1)
        ]
        assert [message._text for message in cp.query(UserMessage)] == ["replacement prompt"]


@pytest.mark.asyncio
async def test_chat_panel_marks_compressed_toc_by_backend_turn_index_after_failed_prompt_cleanup() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("bad prompt")
        await cp.add_error("failed before output")
        await cp.add_user_message("first kept")
        await cp.add_user_message("second kept")
        await cp.add_user_message("second kept detail", is_injection=True)
        await cp.add_user_message("third kept")

        changed = cp.mark_turn_range_compressed((2, 2))
        await pilot.pause()

        items = cp.toc_items
        assert changed is True
        assert [item.turn_index for item in items] == [1, 2, 3]
        assert [item.compressed for item in items] == [False, True, False]
        assert len(items[1].children) == 1
        assert items[1].children[0].compressed is True


@pytest.mark.asyncio
async def test_chat_panel_failed_turn_with_output_reuses_backend_turn_index_for_continuation() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("failed with partial output")
        await cp.add_agent_message("partial output")
        await cp.add_error("failed after output")
        await cp.add_user_message("continuation")
        await cp.add_user_message("next fresh turn")

        changed = cp.mark_turn_range_compressed((1, 1))
        await pilot.pause()

        items = cp.toc_items
        assert changed is True
        assert [item.turn_index for item in items] == [1, 1, 2]
        assert [item.compressed for item in items] == [True, True, False]


@pytest.mark.asyncio
async def test_chat_panel_replay_assigns_turn_index_from_trailing_markers() -> None:
    messages = [
        {"role": "user", "contents": [{"type": "text", "text": "first"}], "additional_properties": {}},
        {"role": "assistant", "contents": [{"type": "text", "text": "one"}], "additional_properties": {}},
        {
            "role": "assistant",
            "contents": [""],
            "additional_properties": {HistoryMarkerKind.KEY: HistoryMarkerKind.TURN, "_turn_id": "turn_1", "_turn": 1},
        },
        {"role": "user", "contents": [{"type": "text", "text": "second"}], "additional_properties": {}},
        {"role": "assistant", "contents": [{"type": "text", "text": "two"}], "additional_properties": {}},
        {
            "role": "assistant",
            "contents": [""],
            "additional_properties": {HistoryMarkerKind.KEY: HistoryMarkerKind.TURN, "_turn_id": "turn_2", "_turn": 2},
        },
    ]

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.replay_history(messages)

        changed = cp.mark_turn_range_compressed((2, 2))
        await pilot.pause()

        items = cp.toc_items
        assert changed is True
        assert [item.turn_index for item in items] == [1, 2]
        assert [item.compressed for item in items] == [False, True]


@pytest.mark.asyncio
async def test_chat_panel_adds_inline_retry_action_for_error() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_error("something went wrong")
        await pilot.pause()

        action = cp.query_one(ErrorMessage)
        button = action.query_one(Button)
        assert str(button.label) == "Retry"


@pytest.mark.asyncio
async def test_chat_panel_can_render_error_without_inline_action() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_error("not retryable", action_label=None)
        await pilot.pause()

        assert cp.query_one(ErrorMessage)
        assert not list(cp.query(Button))


@pytest.mark.asyncio
async def test_chat_panel_adds_inline_continue_action_for_interrupt() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_interrupted()
        await pilot.pause()

        action = cp.query_one(InterruptedMessage)
        button = action.query_one(Button)
        assert str(button.label) == "Continue"


def test_interrupted_message_default_copy_is_byte_identical() -> None:
    assert InterruptedMessage("Execution interrupted", "user")._render_text().plain == (
        "⚠ Interrupted\nExecution interrupted by user"
    )
    assert InterruptedMessage("Execution failed", "error")._render_text().plain == "✗ Error\nExecution failed"
    assert InterruptedMessage("Checkpoint restored", "")._render_text().plain == ("⚠ Interrupted\nCheckpoint restored")


@pytest.mark.asyncio
async def test_chat_panel_live_interruption_uses_localized_frame() -> None:
    async with ChatPanelApp(Localizer("zh-Hans")).run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_interrupted("操作\x1b停止", "user")
        await pilot.pause()

        interruption = panel.query_one(InterruptedMessage)
        assert interruption._render_text().plain == "⚠ 已中断\n用户中断：操作�停止"  # noqa: RUF001
        assert str(interruption.query_one(Button).label) == "继续"

        await panel.add_interrupted("失败详情", "error")
        await pilot.pause()

        interruption = panel.query_one(InterruptedMessage)
        assert interruption._render_text().plain == "✗ 错误\n失败详情"
        assert str(interruption.query_one(Button).label) == "重试"


@pytest.mark.asyncio
async def test_chat_panel_removes_inline_status_action_with_status() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_error("something went wrong")
        await cp.add_agent_message("recovered")
        await pilot.pause()

        assert not list(cp.query(ErrorMessage))
        assert not list(cp.query(Button))


@pytest.mark.asyncio
async def test_chat_panel_context_fold_replaces_trailing_error_on_retry() -> None:
    """A retry-time fold is run output, so it supersedes the prior error."""
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_error("first attempt failed")

        await cp.add_context_fold("ctx-retry", "durable summary", 3, (1, 1))
        await pilot.pause()

        assert not list(cp.query(ErrorMessage))
        fold = cp.query_one(ContextFoldWidget)
        assert "Compressed (Turn 1)" in fold.render().plain


@pytest.mark.asyncio
async def test_chat_panel_context_fold_breaks_stale_stream_cursor() -> None:
    """Recovered output mounts after a carried fold, not into failed partial text."""
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_agent_message("failed partial", is_final=False)
        await cp.add_error("stream failed")

        await cp.add_context_fold("ctx-retry", "durable summary", 3, (1, 1))
        await cp.add_agent_message("retry recovered", is_final=True)
        await pilot.pause()

        transcript = _chat_content_children(cp)
        agent_messages = list(cp.query(AgentMessage))
        fold = cp.query_one(ContextFoldWidget)
        assert [message._text for message in agent_messages] == ["failed partial", "retry recovered"]
        assert [message._is_final for message in agent_messages] == [True, True]
        assert [len(list(message.query(".agent-cursor"))) for message in agent_messages] == [0, 0]
        assert transcript.index(agent_messages[0]) < transcript.index(fold) < transcript.index(agent_messages[1])


@pytest.mark.asyncio
async def test_chat_panel_compaction_start_breaks_stale_stream_cursor() -> None:
    """Recovered output mounts after Phase 4, leaving no orphan stream cursor."""
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_agent_message("failed partial", is_final=False)
        await cp.add_error("stream failed")

        await cp.add_compaction_start("comp-retry")
        cp.complete_compaction("comp-retry", outcome="ok", last_words="durable note")
        await cp.add_agent_message("retry recovered", is_final=True)
        await pilot.pause()

        transcript = _chat_content_children(cp)
        agent_messages = list(cp.query(AgentMessage))
        card = cp.query_one(CompactionCard)
        assert [message._text for message in agent_messages] == ["failed partial", "retry recovered"]
        assert [message._is_final for message in agent_messages] == [True, True]
        assert [len(list(message.query(".agent-cursor"))) for message in agent_messages] == [0, 0]
        assert transcript.index(agent_messages[0]) < transcript.index(card) < transcript.index(agent_messages[1])


@pytest.mark.asyncio
async def test_chat_panel_retry_cleanup_finalizes_stream_before_compaction() -> None:
    """The real RetryAttempt sequence cannot orphan a partial stream cursor."""
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_agent_message("failed partial", is_final=False)

        await cp.prepare_retry()
        await cp.add_retry("Stream stalled", 1, 5, 0)
        await cp.add_compaction_start("comp-retry")
        cp.complete_compaction("comp-retry", outcome="ok", last_words="durable note")
        await cp.add_agent_message("retry recovered", is_final=True)
        await pilot.pause()

        transcript = _chat_content_children(cp)
        agent_messages = list(cp.query(AgentMessage))
        card = cp.query_one(CompactionCard)
        assert [message._text for message in agent_messages] == ["failed partial", "retry recovered"]
        assert [message._is_final for message in agent_messages] == [True, True]
        assert [len(list(message.query(".agent-cursor"))) for message in agent_messages] == [0, 0]
        assert transcript.index(agent_messages[0]) < transcript.index(card) < transcript.index(agent_messages[1])


@pytest.mark.asyncio
async def test_chat_panel_tool_group_open_finalizes_failed_partial_stream() -> None:
    """A retry starting with a tool call cannot reuse failed streamed prose."""
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_agent_message("failed partial", is_final=False)
        await cp.add_error("stream failed")

        await cp.add_tool_start("call-retry", "read_file", "filesystem.read", "path=/tmp/retry")
        await cp.add_tool_result("call-retry", "read_file", "recovered tool output")
        await cp.add_agent_message("retry recovered", is_final=True)
        await pilot.pause()

        transcript = _chat_content_children(cp)
        agent_messages = list(cp.query(AgentMessage))
        group = cp.query_one(ToolGroup)
        assert [message._text for message in agent_messages] == ["failed partial", "retry recovered"]
        assert [message._is_final for message in agent_messages] == [True, True]
        assert [len(list(message.query(".agent-cursor"))) for message in agent_messages] == [0, 0]
        assert transcript.index(agent_messages[0]) < transcript.index(group) < transcript.index(agent_messages[1])


@pytest.mark.asyncio
async def test_chat_panel_new_user_finalizes_failed_partial_stream() -> None:
    """Submitting a new prompt after failure leaves no orphan stream cursor."""
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("first prompt")
        await cp.add_agent_message("failed partial", is_final=False)
        await cp.add_error("stream failed")

        await cp.add_user_message("replacement prompt")
        await cp.add_agent_message("replacement response", is_final=True)
        await pilot.pause()

        transcript = _chat_content_children(cp)
        agent_messages = list(cp.query(AgentMessage))
        replacement = list(cp.query(UserMessage))[-1]
        assert [message._text for message in agent_messages] == ["failed partial", "replacement response"]
        assert [message._is_final for message in agent_messages] == [True, True]
        assert [len(list(message.query(".agent-cursor"))) for message in agent_messages] == [0, 0]
        assert transcript.index(agent_messages[0]) < transcript.index(replacement) < transcript.index(agent_messages[1])


@pytest.mark.asyncio
async def test_chat_panel_intermediate_message_finalizes_prior_stream() -> None:
    """Intermediate prose starts after a terminalized prior stream widget."""
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_agent_message("prior partial", is_final=False)

        await cp.add_agent_message("Checking the tools.", is_intermediate=True)
        await cp.add_agent_message("finished", is_final=True)
        await pilot.pause()

        agent_messages = list(cp.query(AgentMessage))
        assert [message._text for message in agent_messages] == ["prior partial", "Checking the tools.", "finished"]
        assert [message._is_final for message in agent_messages] == [True, True, True]
        assert [len(list(message.query(".agent-cursor"))) for message in agent_messages] == [0, 0, 0]


@pytest.mark.asyncio
async def test_chat_panel_retry_output_removes_error_across_system_message() -> None:
    """Profile/workspace indicators do not pin a retried error in history."""
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_error("first attempt failed")
        await cp.add_system("Agent profile switched: Code → QA", key="profile-switch-1")

        await cp.add_agent_message("retry recovered", is_final=True)
        await pilot.pause()

        assert not list(cp.query(ErrorMessage))
        assert len(list(cp.query(SystemMessage))) == 1
        assert cp.query_one(AgentMessage)._text == "retry recovered"


@pytest.mark.asyncio
async def test_chat_panel_removes_status_before_retry_note_on_agent_recovery() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("original prompt")
        await cp.add_error("something went wrong")
        await cp.add_user_message("extra context", is_injection=True)
        await cp.add_agent_message("recovered")
        await pilot.pause()

        assert not list(cp.query(ErrorMessage))
        assert len(list(cp.query(UserMessage))) == 2
        assert len(cp.get_agent_responses()) == 1


@pytest.mark.asyncio
async def test_chat_panel_hides_trailing_status_action_for_input_bar_retry() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_interrupted()
        await pilot.pause()

        action = cp.query_one(InterruptedMessage)
        row = action.query_one(".status-action-row")
        assert row.display is True

        cp.hide_trailing_status_action()
        await pilot.pause()

        assert row.display is False
        assert cp.query_one(InterruptedMessage) is action


@pytest.mark.asyncio
async def test_chat_panel_border_title_click_posts_title_clicked(monkeypatch: pytest.MonkeyPatch) -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        session_id = "12345678-1234-1234-1234-123456789abc"
        posted: list[object] = []
        original_post = cp.post_message

        def capture(message: object) -> bool:
            posted.append(message)
            return original_post(message)

        monkeypatch.setattr(cp, "post_message", capture)

        cp.set_session_id(session_id)
        cp.on_click(
            Click(
                cp,
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=1,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=cp.region.x,
                screen_y=cp.region.y,
            )
        )
        await pilot.pause()

        assert any(isinstance(message, ChatPanel.TitleClicked) for message in posted)


@pytest.mark.asyncio
async def test_chat_panel_passive_click_preserves_existing_input_focus() -> None:
    class PassiveClickApp(App):
        def compose(self) -> ComposeResult:
            yield TextArea(id="input")
            yield ChatPanel()

    async with PassiveClickApp().run_test() as pilot:
        text_area = pilot.app.query_one("#input", TextArea)
        text_area.focus()
        await pilot.pause()
        assert pilot.app.focused is text_area

        await pilot.click(ChatPanel, offset=(1, 1))
        await pilot.pause()

        assert pilot.app.focused is text_area


def test_chat_panel_border_title_shows_session_title() -> None:
    cp = ChatPanel()

    cp.set_session_id("12345678-1234-1234-1234-123456789abc")
    assert str(cp.border_title) == "Session: 123456781234"

    cp.set_session_title("Fix login bug")
    assert str(cp.border_title) == "Session: 123456781234 \u00b7 Fix login bug"

    cp.set_session_title("")
    assert str(cp.border_title) == "Session: 123456781234"


def test_chat_panel_border_title_truncates_long_session_title() -> None:
    cp = ChatPanel()

    cp.set_session_id("12345678-1234-1234-1234-123456789abc")
    cp.set_session_title("t" * 200)

    title = str(cp.border_title)
    assert title.startswith("Session: 123456781234 \u00b7 ")
    assert title.endswith("\u2026")
    assert len(title) <= len("Session: 123456781234 \u00b7 ") + 48


@pytest.mark.asyncio
async def test_chat_and_session_json_chrome_relocalize_without_touching_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = LocaleController(Settings(locale="en"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    chat: ChatPanel | None = None
    session_json: SessionJsonPanel | None = None

    class LocalizedChatChromeApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel(locale_controller=controller)
            yield SessionJsonPanel(locale_controller=controller)

    async with LocalizedChatChromeApp().run_test() as pilot:
        chat = pilot.app.query_one(ChatPanel)
        session_json = pilot.app.query_one(SessionJsonPanel)
        session_id = "12345678-1234-1234-1234-123456789abc"
        chat.set_session_id(session_id)
        chat.set_session_title("Fix login bug")
        monkeypatch.setattr(session_json, "_resolve_session_path", lambda _session_id: None)
        session_json.load_session(session_id)
        transcript = Static(Text("literal transcript [red]payload"))
        await chat.mount(transcript)
        await pilot.pause()
        children_before = tuple(chat.children)
        button = chat.query_one(_ScrollToBottomButton)
        visibility_before = button.styles.visibility
        walk_calls = 0
        original_walk = chat.walk_children

        def record_walk(*args: object, **kwargs: object):
            nonlocal walk_calls
            walk_calls += 1
            return original_walk(*args, **kwargs)

        monkeypatch.setattr(chat, "walk_children", record_walk)
        assert controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED

        assert walk_calls == 0
        assert tuple(chat.children) == children_before
        assert chat.children[-1] is button
        assert transcript.render().plain == "literal transcript [red]payload"
        assert str(chat.border_title) == "会话：123456781234 · Fix login bug"  # noqa: RUF001
        assert button.render().plain == "滚动到底部 ↓"
        assert button.tooltip is not None
        assert button.tooltip.plain == "跳转到对话底部（Ctrl+End）"  # noqa: RUF001
        assert button.styles.visibility == visibility_before
        assert str(session_json.border_title) == "会话 JSON：123456781234"  # noqa: RUF001

        chat.set_session_title("")
        assert str(chat.border_title) == "会话：123456781234"  # noqa: RUF001

    assert chat is not None and session_json is not None
    assert chat not in controller._surfaces
    assert session_json not in controller._surfaces


@pytest.mark.asyncio
async def test_chat_scroll_chrome_reserves_width_for_locale_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = LocaleController(Settings(locale="zh-Hans"))
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)

    class LocalizedChatChromeApp(App):
        def compose(self) -> ComposeResult:
            yield ChatPanel(locale_controller=controller)

    async with LocalizedChatChromeApp().run_test() as pilot:
        button = pilot.app.query_one(_ScrollToBottomButton)
        assert button.render().plain == "滚动到底部 ↓"

        assert controller.switch_locale("en").status is LocaleSwitchStatus.EFFECTIVE_CHANGED

        assert button.render().plain == "Scroll to bottom ↓"
        assert button.styles.min_width is not None
        assert button.styles.min_width.value >= button.render().cell_length + 2


def test_chat_panel_border_subtitle_includes_git_branch() -> None:
    cp = ChatPanel()

    cp.set_workspace_cwd("/tmp/project")
    cp.set_workspace_branch("feat/abc")

    assert str(cp.border_subtitle) == "/tmp/project (feat/abc)"


def test_chat_panel_border_subtitle_omits_empty_git_branch() -> None:
    cp = ChatPanel()

    cp.set_workspace_cwd("/tmp/project")
    cp.set_workspace_branch("main")
    cp.set_workspace_branch("")

    assert str(cp.border_subtitle) == "/tmp/project"


@pytest.mark.asyncio
async def test_context_panel_renders_default_context_window_before_usage_events() -> None:
    async with ContextPanelApp().run_test(size=(46, 24)) as pilot:
        await pilot.pause()

        usage_text = pilot.app.query_one("#ctx-usage-text", Static).render().plain

    assert usage_text == "0 / 200,000 (0.0%)"


def test_context_panel_usage_state_preserves_main_gauge_for_sub_agent_totals() -> None:
    cp = ContextPanel()
    main_state = ContextUsageState.with_window(
        used_tokens=57_700,
        max_context_tokens=200_000,
        total_session_tokens=236_100,
        total_session_input_tokens=100_000,
        total_session_output_tokens=136_100,
        total_session_cache_hit_tokens=4_200,
    )

    cp.watch_usage_state(main_state)

    assert cp._current_used == 57_700
    assert cp._current_max == 200_000
    assert cp._usage_history[-1] == pytest.approx(28.85)
    assert cp._total_session_tokens == 236_100

    cp.watch_usage_state(
        ContextUsageState.session_totals_only(
            main_state,
            fallback_used_tokens=99_999,
            total_session_tokens=253_535,
            total_session_input_tokens=110_000,
            total_session_output_tokens=143_535,
            total_session_cache_hit_tokens=5_000,
        )
    )

    assert cp._current_used == 57_700
    assert cp._current_max == 200_000
    assert len(cp._usage_history) == 2
    assert cp._usage_history[0] == 0.0
    assert cp._usage_history[-1] == pytest.approx(28.85)
    assert cp._total_session_tokens == 253_535
    assert cp._total_session_input_tokens == 110_000
    assert cp._total_session_output_tokens == 143_535
    assert cp._total_session_cache_hit_tokens == 5_000

    remounted = ContextPanel()
    remounted.watch_usage_state(
        ContextUsageState.session_totals_only(
            main_state,
            fallback_used_tokens=99_999,
            total_session_tokens=300_000,
        )
    )

    assert remounted._current_used == 57_700
    assert remounted._current_max == 200_000
    assert remounted._usage_history == [0.0]
    assert remounted._total_session_tokens == 300_000


def test_context_panel_compressed_block_title_includes_turn_range_and_context_id() -> None:
    panel = ContextPanel()

    def title(context_id: str, turn_range: tuple[int, int]) -> str:
        return panel._compressed_block_title(
            _CompressedBlock(context_id=context_id, summary="", freed_messages=0, turn_range=turn_range)
        )

    assert title("ctx_930083b2", (1, 4)) == "Turn 1-4 (ctx_930083b2)"
    assert title("ctx_single", (3, 3)) == "Turn 3 (ctx_single)"
    assert title("ctx_legacy", (0, 0)) == "ctx_legacy"


@pytest.mark.asyncio
async def test_context_panel_compressed_blocks_rerender_in_active_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    controller = LocaleController(Settings(locale="en"))

    class LocalizedContextPanelApp(App):
        def compose(self) -> ComposeResult:
            yield ContextPanel(locale_controller=controller)

    async with LocalizedContextPanelApp().run_test(size=(60, 24)) as pilot:
        panel = pilot.app.query_one(ContextPanel)
        log = pilot.app.query_one("#ctx-snapshots", RichLog)
        panel.add_compressed_block("ctx_a", "summary", 5000, (1, 4))
        panel.add_compressed_block("ctx_b", "solo", 1, (6, 6))
        await wait_for(
            lambda: any("Turn 6 (ctx_b)" in line.text for line in log.lines),
            pilot=pilot,
            description="initial compressed block reflow",
        )

        joined = " ".join(line.text.strip() for line in log.lines)
        assert "Turn 1-4 (ctx_a)  5,000 messages" in joined
        assert "Turn 6 (ctx_b)  1 message" in joined

        assert controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        panel.refresh_localization()
        await wait_for(
            lambda: bool(log.lines and "第" in log.lines[0].text),
            pilot=pilot,
            description="localized compressed block reflow",
        )

        joined = " ".join(line.text.strip() for line in log.lines)
        assert "第 1-4 轮 (ctx_a)  5,000 条消息" in joined
        assert "第 6 轮 (ctx_b)  1 条消息" in joined

        # A block recorded while zh-Hans is already active renders zh directly.
        panel.add_compressed_block("ctx_c", "fresh", 2, (7, 8))
        await wait_for(
            lambda: any("ctx_c" in line.text for line in log.lines),
            pilot=pilot,
            description="new localized compressed block reflow",
        )
        joined = " ".join(line.text.strip() for line in log.lines)
        assert "第 7-8 轮 (ctx_c)  2 条消息" in joined


@pytest.mark.asyncio
async def test_context_panel_wraps_compressed_summary_at_visible_width() -> None:
    summary = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa"
    async with ContextPanelApp().run_test(size=(46, 24)) as pilot:
        panel = pilot.app.query_one(ContextPanel)
        log = pilot.app.query_one("#ctx-snapshots", RichLog)
        assert log.can_focus
        panel.add_compressed_block("ctx_930083b2", summary, turn_range=(1, 4))
        await wait_for(
            lambda: bool(log.lines),
            pilot=pilot,
            description="compressed summary reflow",
        )

        visible_width = log.scrollable_content_region.width
        lines = [line.text.rstrip() for line in log.lines]
        assert visible_width < 78
        assert all(line.cell_length <= visible_width for line in log.lines)
        assert lines[0] == "\u2022 Turn 1-4 (ctx_930083b2)"
        assert " ".join(line.strip() for line in lines[1:]) == summary


@pytest.mark.asyncio
async def test_context_panel_reflows_block_written_while_tab_is_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chrys.app.tui.widgets.sidebar.buddy.get_companion", lambda: None)
    summary = " ".join(f"word{index:03d}" for index in range(80))

    async with SidebarContextPanelApp().run_test(size=(120, 40)) as pilot:
        sidebar = pilot.app.query_one(SidebarPanel)
        log = pilot.app.query_one("#ctx-snapshots", RichLog)

        sidebar.focus_tab("tab-context")
        await pilot.pause()
        assert log.scrollable_content_region.width > 0

        sidebar.focus_tab("tab-toc")
        await pilot.pause()
        assert log.scrollable_content_region.width == 0

        sidebar.context_panel.add_compressed_block("ctx_hidden", summary, turn_range=(5, 7))
        assert log.lines == []

        sidebar.focus_tab("tab-context")
        # The re-shown tab needs a full layout/resize/reflow chain; loaded
        # macOS CI workers have overrun the default window.
        await wait_for(
            lambda: bool(log.scrollable_content_region.width > 0 and log.lines),
            pilot=pilot,
            timeout=15.0,
            description="hidden compressed block reflow",
        )

        visible_width = log.scrollable_content_region.width
        lines = [line.text.rstrip() for line in log.lines]
        assert visible_width > 0
        assert all(line.cell_length <= visible_width for line in log.lines)
        assert lines[0] == "\u2022 Turn 5-7 (ctx_hidden)"
        assert " ".join(line.strip() for line in lines[1:]) == summary


@pytest.mark.asyncio
async def test_context_panel_coalesces_resize_reflows(monkeypatch: pytest.MonkeyPatch) -> None:
    async with ContextPanelApp().run_test(size=(46, 24)) as pilot:
        panel = pilot.app.query_one(ContextPanel)
        log = pilot.app.query_one("#ctx-snapshots", RichLog)
        panel.add_compressed_block("ctx_resize", "summary", turn_range=(8, 8))
        await wait_for(
            lambda: bool(log.lines),
            pilot=pilot,
            description="initial resize fixture reflow",
        )

        reflow_widths: list[int] = []
        log_type = type(log)
        original_reflow = log_type._reflow

        def record_reflow(widget: object, width: int) -> None:
            reflow_widths.append(width)
            original_reflow(widget, width)

        monkeypatch.setattr(log_type, "_reflow", record_reflow)
        log._last_render_width = 0
        resize = Resize(log.size, log.virtual_size)
        log.on_resize(resize)
        log.on_resize(resize)

        assert reflow_widths == []
        # The flush runs via call_after_refresh, i.e. only after the next
        # screen refresh — poll instead of betting on a single pause.
        await wait_for(lambda: reflow_widths, pilot=pilot, description="coalesced reflow")
        assert reflow_widths == [log.scrollable_content_region.width]


@pytest.mark.asyncio
async def test_session_json_border_click_copies_to_terminal_and_os_clipboards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with SessionJsonPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(SessionJsonPanel)
        session_path = "/tmp/chrys/sessions/12345678/session.json"
        copied: list[str] = []
        monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", copied.append)

        panel._session_path = session_path
        panel.on_click(
            Click(
                panel,
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=1,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=panel.region.x,
                screen_y=panel.region.y,
            )
        )

        assert pilot.app.clipboard == session_path
        assert copied == [session_path]


@pytest.mark.asyncio
async def test_session_json_render_line_after_remove_returns_blank() -> None:
    async with SessionJsonPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(SessionJsonPanel)

        await panel.remove()
        await pilot.pause()

        rendered = panel.render_line(0)

    assert rendered.cell_length == 0


@pytest.mark.asyncio
async def test_prepare_retry_cancels_running_tools_and_opens_fresh_group() -> None:
    """Regression: after a main-agent stream retry, new tool calls must
    attach to a NEW tool group under the re-emitted assistant block —
    not the stale group from the failed attempt.

    Without ``prepare_retry()`` the panel would carry ``_current_tool_group``
    across the retry boundary, so the next ``add_tool_start`` after the
    re-emitted intermediate text would merge into the old group, mounting
    new sub-agent cards under the OLD assistant message.
    """
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("count the lines")

        # Attempt 1: intermediate text + two parallel sub-agent tool calls.
        await cp.add_agent_message(
            "I'll launch two sub-agents in parallel.",
            is_intermediate=True,
        )
        await cp.add_tool_start("call_A", "general_agent", "sub_agent")
        await cp.add_tool_start("call_B", "general_agent", "sub_agent")
        old_group = cp._current_tool_group
        assert old_group is not None
        assert len(old_group._tools) == 2
        assert all(tc.status == "running" for tc in old_group._tools.values())

        # Stream stall fires → event handler calls prepare_retry() then add_retry().
        await cp.prepare_retry()
        await cp.add_retry("Stream stalled", 1, 5, 3)

        # All previously-running tools should be marked cancelled so the
        # TUI shows a terminal state for the stale invocations.
        assert all(tc.status != "running" for tc in old_group._tools.values())
        # Group state is cleared so the next ToolCallStart opens a fresh group.
        assert cp._current_tool_group is None
        assert cp._current_agent_msg is None

        # Attempt 2: the re-emitted intermediate text mounts a new assistant
        # widget, and new tool calls open a fresh group beneath it.
        await cp.add_agent_message(
            "I'll launch two sub-agents in parallel.",
            is_intermediate=True,
        )
        await cp.add_tool_start("call_C", "general_agent", "sub_agent")
        await cp.add_tool_start("call_D", "general_agent", "sub_agent")
        new_group = cp._current_tool_group
        assert new_group is not None
        assert new_group is not old_group, "retry-spawned tools must open a fresh group"
        assert len(new_group._tools) == 2
        assert "call_C" in new_group._tools
        assert "call_D" in new_group._tools
        # And the OLD group still holds only the original calls — the new
        # ones did NOT leak into it.
        assert "call_C" not in old_group._tools
        assert "call_D" not in old_group._tools


@pytest.mark.asyncio
async def test_prepare_retry_preserves_completed_tool_results() -> None:
    """Tools that finished BEFORE the retry fires (e.g. one of two parallel
    sub-agents that completed while the other was still running) should
    retain their final results — only the still-running ones transition
    to the cancelled state."""
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("go")
        await cp.add_agent_message("starting", is_intermediate=True)
        await cp.add_tool_start("done_call", "general_agent", "sub_agent")
        await cp.add_tool_start("running_call", "general_agent", "sub_agent")

        # First sub-agent finishes, second still running.
        await cp.add_tool_result("done_call", "general_agent", "all good", 1200)

        group = cp._current_tool_group
        assert group is not None
        assert group._tools["done_call"].status != "running"
        assert group._tools["running_call"].status == "running"

        await cp.prepare_retry()

        # Completed result preserved, running one is now terminal.
        assert group._tools["done_call"].status != "running"
        assert group._tools["running_call"].status != "running"


@pytest.mark.asyncio
async def test_chat_panel_bottom_spacer_lifecycle() -> None:
    """Spacer is hidden during welcome, shown after it's dismissed, and always trails content.

    The spacer's ``height: 1fr`` has an effective ``min-height: 1`` in Textual —
    leaving it visible while the welcome widget (``height: 100%``) is present
    produces a phantom 1-row scrollbar.  After the first user message the
    welcome is removed and the spacer becomes the flexible tail that gives
    ``scroll_visible(top=True)`` room to scroll into.
    """
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        spacer = cp.query_one(_ChatBottomSpacer)
        scroll_button = cp.query_one(_ScrollToBottomButton)

        # Welcome state: spacer exists but is hidden.
        assert spacer.display is False
        # Spacer must trail content so new mounts land before it.
        assert cp.children[-2] is spacer
        assert cp.children[-1] is scroll_button

        await cp.add_user_message("hello")
        # Welcome dismissed → spacer is now visible.
        assert spacer.display is True
        # Spacer is still immediately before the floating affordance.
        assert cp.children[-2] is spacer
        assert cp.children[-1] is scroll_button

        await cp.add_agent_message("hi there")
        await pilot.pause()
        assert cp.children[-2] is spacer
        assert cp.children[-1] is scroll_button

        # clear() recreates the spacer (fresh welcome) and re-hides it.
        await cp.clear()
        spacer2 = cp.query_one(_ChatBottomSpacer)
        scroll_button2 = cp.query_one(_ScrollToBottomButton)
        assert spacer2.display is False
        assert cp.children[-2] is spacer2
        assert cp.children[-1] is scroll_button2


@pytest.mark.asyncio
async def test_chat_panel_scroll_to_bottom_affordance_visibility_and_jump() -> None:
    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        button = cp.query_one(_ScrollToBottomButton)

        await cp._dismiss_welcome()
        for index in range(60):
            await cp.mount(Static(f"line {index}"))
        await pilot.pause()

        assert cp.max_scroll_y > 0

        cp._set_scroll_y_programmatically(cp.max_scroll_y - 0.4)
        cp._sync_scroll_to_bottom_button()
        assert button.visible is False, "sub-cell scroll positions that round to bottom should hide the affordance"

        cp._set_scroll_y_programmatically(cp.max_scroll_y - 1)
        cp._sync_scroll_to_bottom_button()
        assert button.visible is True

        cp._auto_scroll_paused_by_user = True
        cp._anchor_released = True
        cp.jump_to_bottom()
        await pilot.pause()

        assert cp._auto_scroll_paused_by_user is False
        assert cp._anchor_released is False
        assert round(cp.scroll_y) == cp.max_scroll_y
        assert button.visible is False
        # The affordance is toggled via visibility, never display: a display
        # flip would force a full reflow + compositor full-map rebuild on
        # every bottom-boundary crossing while scrolling a large transcript.
        assert button.display is True


@pytest.mark.asyncio
async def test_chat_panel_mount_survives_detached_spacer_reference() -> None:
    """``mount()`` must not crash if ``_bottom_spacer`` points at a detached widget.

    Regression: on Windows session restore a race between ``clear()`` (awaiting
    ``remove_children``) and an incoming event-handler mount caused
    ``MountError: Unable to find relative location of _ChatBottomSpacer``.
    The override now guards on ``spacer.parent is self``; this test forces
    that exact shape by swapping in an unattached ``_ChatBottomSpacer``
    instance and then mounting a normal widget.
    """
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        # Simulate the race: a stale reference to a widget that is not our
        # child.  A freshly constructed spacer has ``parent is None``, which
        # is the same state as a just-removed child during the async yield.
        detached = _ChatBottomSpacer()
        cp._bottom_spacer = detached
        assert detached.parent is None, "test setup precondition"

        survivor = Static("survivor", id="survivor")
        # Must not raise MountError.
        await cp.mount(survivor)

        # The new widget landed (appended) and the detached spacer was
        # correctly ignored rather than used as a ``before=`` target.
        assert survivor.parent is cp
        assert detached.parent is None, "detached spacer must not have been attached by the mount path"


@pytest.mark.asyncio
async def test_chat_panel_bottom_spacer_reshows_after_shrink() -> None:
    """Spacer re-appears when natural content shrinks back below the viewport.

    Regression for a bug where ``watch_virtual_size`` only hid the spacer and
    never restored it: once a tool group auto-collapsed its first expanded
    tool (triggered by a second tool start), the shrunk canvas was aligned
    to the viewport bottom by the bottom-pinning anchor, leaving empty
    padding above the current turn's user message.
    """
    from textual.geometry import Size

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        spacer = cp.query_one(_ChatBottomSpacer)

        # Dismiss welcome so the spacer enters its normal lifecycle.
        await cp.add_user_message("hello")
        await pilot.pause()
        assert spacer.display is True

        viewport_h = cp.size.height
        assert viewport_h > 0, "test harness must give the panel a real viewport"

        # Simulate natural content growing past the viewport: watch_virtual_size
        # sees new_h == natural + spacer_h, and the branch compares the
        # computed natural_h against viewport_h.  With the spacer collapsed
        # to ~0 here (natural alone is already tall), new_h >= viewport_h.
        cp.watch_virtual_size(Size(cp.size.width, 0), Size(cp.size.width, viewport_h + 20))
        assert spacer.display is False, "spacer should hide once natural content fills viewport"

        # Now simulate natural content shrinking back below the viewport
        # (tool-group collapse).  With spacer hidden, new_h == natural_h.
        cp.watch_virtual_size(
            Size(cp.size.width, viewport_h + 20),
            Size(cp.size.width, max(1, viewport_h - 5)),
        )
        assert spacer.display is True, "spacer must re-show so the user message is not bottom-aligned"


@pytest.mark.asyncio
async def test_chat_panel_dismiss_welcome_preserves_hidden_spacer_after_first_turn() -> None:
    """After welcome is gone, dismissing it again must not re-show a hidden spacer."""

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        spacer = cp.query_one(_ChatBottomSpacer)
        await pilot.pause()

        await cp.add_user_message("hello")
        await pilot.pause()
        assert cp._welcome is None
        assert spacer.display is True

        spacer.display = False
        await cp._dismiss_welcome()
        assert spacer.display is False


@pytest.mark.asyncio
async def test_chat_panel_pauses_gc_while_manual_scroll_is_active(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manual scroll pauses cyclic GC and restores it after the debounce window."""
    _, _FakeGC = _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        cp._pause_gc_for_manual_scroll()
        assert _FakeGC.enabled is False
        assert _FakeGC.disable_calls == 1

        cp._pause_gc_for_manual_scroll()
        assert _FakeGC.disable_calls == 1

        await _wait_for_chat_panel_gc_resume(pilot, cp)

        assert _FakeGC.enabled is True
        assert _FakeGC.enable_calls == 1
        assert cp._manual_scroll_gc_paused is False

        _FakeGC.enabled = False
        cp._pause_gc_for_manual_scroll()
        await _wait_for_chat_panel_gc_resume(pilot, cp)

        assert _FakeGC.enabled is False
        assert _FakeGC.enable_calls == 1


def test_scroll_gc_paused_uses_shared_owner_count(monkeypatch: pytest.MonkeyPatch) -> None:
    scroll_controller_module, _FakeGC = _install_fake_chat_panel_gc(monkeypatch)

    assert scroll_controller_module.scroll_gc_paused() is False
    scroll_controller_module._claim_scroll_gc_pause()
    scroll_controller_module._claim_scroll_gc_pause()
    assert scroll_controller_module.scroll_gc_paused() is True
    assert _FakeGC.disable_calls == 1

    scroll_controller_module._release_scroll_gc_pause()
    assert scroll_controller_module.scroll_gc_paused() is True
    scroll_controller_module._release_scroll_gc_pause()
    assert scroll_controller_module.scroll_gc_paused() is False
    assert _FakeGC.enable_calls == 1


@pytest.mark.asyncio
async def test_chat_panel_resume_collects_gen0_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume restores GC and drains the paused-scroll gen0 backlog right away."""
    scroll_controller_module, _FakeGC = _install_fake_chat_panel_gc(monkeypatch)
    monkeypatch.setattr(scroll_controller_module, "_MANUAL_SCROLL_GC_RESUME_SECONDS", 0.01)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        cp._pause_gc_for_manual_scroll()
        assert _FakeGC.collect_generations == []
        await _wait_for_chat_panel_gc_resume(pilot, cp)

        assert _FakeGC.enabled is True
        assert _FakeGC.collect_generations == [0]


@pytest.mark.asyncio
async def test_chat_panel_resume_skips_collect_while_another_owner_holds_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No gen0 collect when GC is still disabled by another panel's pause claim."""
    scroll_controller_module, _FakeGC = _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        scroll_controller_module._claim_scroll_gc_pause()
        cp._pause_gc_for_manual_scroll()

        cp._resume_gc_after_manual_scroll()

        assert cp._manual_scroll_gc_paused is False
        assert _FakeGC.enabled is False
        assert _FakeGC.collect_generations == []

        scroll_controller_module._release_scroll_gc_pause()
        assert _FakeGC.enabled is True


@pytest.mark.asyncio
async def test_chat_panel_scrollbar_grab_holds_gc_pause_without_resume_debounce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grabbed thumb holds the pause open; ticks never arm the resume timer."""
    _, _FakeGC = _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        timers: list[object] = []
        real_set_timer = cp.set_timer

        def set_timer_spy(delay: float, callback: Callable[[], None]) -> object:
            timer = real_set_timer(delay, callback)
            timers.append(timer)
            return timer

        monkeypatch.setattr(cp, "set_timer", set_timer_spy)

        cp._scroll_controller.on_scrollbar_grab()
        assert cp._manual_scroll_gc_paused is True
        assert _FakeGC.enabled is False
        assert cp._manual_scroll_gc_timer is None
        assert timers == []

        cp._pause_gc_for_manual_scroll()
        assert cp._manual_scroll_gc_timer is None
        assert timers == []

        cp._scroll_controller.on_scrollbar_release()
        assert cp._manual_scroll_gc_paused is True
        assert cp._manual_scroll_gc_timer is not None
        assert len(timers) == 1

        await _wait_for_chat_panel_gc_resume(pilot, cp)
        assert _FakeGC.enabled is True
        assert _FakeGC.collect_generations == [0]


@pytest.mark.asyncio
async def test_chat_panel_resume_never_fires_while_thumb_still_grabbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: a stray resume call during an active grab keeps GC paused."""
    _, _FakeGC = _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        cp._scroll_controller.on_scrollbar_grab()
        cp._resume_gc_after_manual_scroll()

        assert cp._manual_scroll_gc_paused is True
        assert _FakeGC.enabled is False
        assert _FakeGC.collect_generations == []


@pytest.mark.asyncio
async def test_chat_panel_scrollbar_grabbed_reactive_drives_gc_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mounted panel watches ScrollBar.grabbed: grab pauses, release re-arms."""
    _, _FakeGC = _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await pilot.pause()

        cp.vertical_scrollbar.grabbed = Offset(0, 0)
        await pilot.pause()
        assert cp._scroll_controller.scrollbar_grabbed is True
        assert cp._manual_scroll_gc_paused is True
        assert _FakeGC.enabled is False
        assert cp._manual_scroll_gc_timer is None

        cp.vertical_scrollbar.grabbed = None
        await pilot.pause()
        assert cp._scroll_controller.scrollbar_grabbed is False
        assert cp._manual_scroll_gc_timer is not None

        await _wait_for_chat_panel_gc_resume(pilot, cp)
        assert _FakeGC.enabled is True


@pytest.mark.asyncio
async def test_chat_panel_mouse_wheel_pauses_gc_before_base_scroll_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wheel input should pause GC before Textual's base scroll handler runs once."""
    _, _FakeGC = _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await pilot.pause()
        scroll_down_calls: list[dict[str, object]] = []
        scroll_down_gc_states: list[bool] = []

        def allow_vertical_scroll(_panel: ChatPanel) -> bool:
            return True

        def scroll_down_spy(**kwargs: object) -> bool:
            scroll_down_calls.append(kwargs)
            scroll_down_gc_states.append(_FakeGC.enabled)
            return True

        monkeypatch.setattr(ChatPanel, "allow_vertical_scroll", property(allow_vertical_scroll))
        monkeypatch.setattr(cp, "_scroll_down_for_pointer", scroll_down_spy)

        await cp._on_message(
            textual_events.MouseScrollDown(
                cp, x=1, y=1, delta_x=0, delta_y=1, button=0, shift=False, meta=False, ctrl=False
            )
        )

        assert scroll_down_calls == [{"animate": False}]
        assert scroll_down_gc_states == [False]
        assert _FakeGC.enabled is False
        assert _FakeGC.disable_calls == 1
        assert cp._manual_scroll_gc_paused is True


@pytest.mark.asyncio
async def test_chat_panel_scrollbar_drag_pauses_gc_before_base_scroll_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scrollbar ScrollTo messages should pause GC before Textual scrolls once."""
    _, _FakeGC = _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await pilot.pause()
        scroll_to_calls: list[tuple[float | None, float | None, object, object, bool]] = []

        def allow_scroll(_panel: ChatPanel) -> bool:
            return True

        def scroll_to_spy(x: float | None = None, y: float | None = None, **kwargs: object) -> None:
            scroll_to_calls.append((x, y, kwargs.get("animate"), kwargs.get("duration"), _FakeGC.enabled))

        monkeypatch.setattr(ChatPanel, "_allow_scroll", property(allow_scroll))
        monkeypatch.setattr(cp, "scroll_to", scroll_to_spy)

        await cp._on_message(ScrollTo(y=10, animate=False))

        assert scroll_to_calls == [(None, 10, False, 0.1, False)]
        assert _FakeGC.enabled is False
        assert _FakeGC.disable_calls == 1
        assert cp._manual_scroll_gc_paused is True


@pytest.mark.asyncio
async def test_chat_panel_clear_and_unmount_release_gc_pause_without_collect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clear and unmount stop a pending resume timer and release the pause, never collecting."""
    _, _FakeGC = _install_fake_chat_panel_gc(monkeypatch)

    class FakeTimer:
        def __init__(self, label: str, callback: Callable[[], None]) -> None:
            self.label = label
            self.callback = callback
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

    timers: list[FakeTimer] = []

    def set_timer_spy(_delay: float, callback: Callable[[], None]) -> FakeTimer:
        timer = FakeTimer(callback.__name__, callback)
        timers.append(timer)
        return timer

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        monkeypatch.setattr(cp, "set_timer", set_timer_spy)

        cp._pause_gc_for_manual_scroll()
        gc_timer_for_clear = timers[-1]
        assert gc_timer_for_clear.label == "resume_gc_after_manual_scroll"

        await cp.clear()

        assert gc_timer_for_clear.stop_calls == 1
        assert cp._manual_scroll_gc_timer is None
        assert cp._manual_scroll_gc_paused is False
        assert _FakeGC.enabled is True
        assert _FakeGC.collect_generations == []

        cp._pause_gc_for_manual_scroll()
        gc_timer_for_unmount = timers[-1]
        assert gc_timer_for_unmount.label == "resume_gc_after_manual_scroll"

        cp.on_unmount()

        assert gc_timer_for_unmount.stop_calls == 1
        assert cp._manual_scroll_gc_timer is None
        assert cp._manual_scroll_gc_paused is False
        assert _FakeGC.enabled is True
        assert _FakeGC.collect_generations == []


@pytest.mark.asyncio
async def test_chat_panel_clear_mid_grab_releases_gesture_hold(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clear during an active thumb grab must still restore GC (restorability parity)."""
    _, _FakeGC = _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        cp._scroll_controller.on_scrollbar_grab()
        assert _FakeGC.enabled is False

        await cp.clear()

        assert cp._scroll_controller.scrollbar_grabbed is False
        assert cp._manual_scroll_gc_paused is False
        assert _FakeGC.enabled is True
        assert _FakeGC.collect_generations == []


@pytest.mark.asyncio
async def test_collapsed_completed_tool_group_prunes_and_rebuilds_file_diff() -> None:
    """Collapsed historical file tools should not keep their diff widget subtree mounted."""
    from chrys.app.tui.widgets.diff_view import DiffView
    from chrys.app.tui.widgets.diff_view.inline import InlineUnifiedDiffLines

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("edit")
        await cp.add_tool_start("edit1", "edit_file", "filesystem.write", args={"path": "src/app.py"})
        await cp.add_tool_result(
            "edit1",
            "edit_file",
            "Edited src/app.py",
            25,
            file_snapshot=("old\n", "new\n"),
        )
        await cp.add_agent_message("done", is_final=True)

        group = cp.query_one(ToolGroup)
        for _ in range(200):
            await pilot.pause()
            if not group._content_mounted:
                break

        assert group.collapsed is True
        assert group.all_complete is True
        assert group._content_mounted is False
        assert len(group._tools) == 0
        assert not list(cp.query(DiffView))

        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if list(cp.query(InlineUnifiedDiffLines)):
                break

        assert group._content_mounted is True
        assert len(group._tools) == 1
        assert not list(cp.query(DiffView))
        assert list(cp.query(InlineUnifiedDiffLines))


@pytest.mark.asyncio
async def test_completed_tool_subtree_posts_idle_reclaim_after_awaited_removal() -> None:
    app = GcMessageChatPanelApp()
    async with app.run_test() as pilot:
        panel = app.query_one(ChatPanel)
        await panel.add_user_message("tools")
        await panel.add_tool_start("tool1", "plain_tool", "", args={"value": 1})
        await panel.add_tool_result("tool1", "plain_tool", "done", 10)
        await panel.add_agent_message("done", is_final=True)
        group = panel.query_one(ToolGroup)

        for _ in range(200):
            await pilot.pause()
            if app.gc_messages:
                break

        assert group._content_mounted is False
        assert not group.query_one("#tg-content").children
        assert len(app.gc_messages) == 1
        message = app.gc_messages[0]
        assert isinstance(message, GcReclaimRequested)
        assert message.reason is GcReclaimReason.STABLE_CONTENT_REMOVED
        assert message.prompt is False

        await group._release_completed_tool_widgets()
        await pilot.pause()
        assert len(app.gc_messages) == 1


@pytest.mark.asyncio
async def test_plain_restored_tool_subtree_posts_stable_content_absorb() -> None:
    app = GcMessageChatPanelApp()
    async with app.run_test() as pilot:
        panel = app.query_one(ChatPanel)
        await panel.add_user_message("tools")
        await panel.add_tool_start("tool1", "plain_tool", "", args={"value": 1})
        await panel.add_tool_result("tool1", "plain_tool", "done", 10)
        await panel.add_agent_message("done", is_final=True)
        group = panel.query_one(ToolGroup)

        for _ in range(200):
            await pilot.pause()
            if not group._content_mounted and app.gc_messages:
                break
        assert isinstance(app.gc_messages[-1], GcReclaimRequested)

        # Model the coordinator having consumed the removal reclaim. The
        # subsequent expansion must independently announce its rebuilt tree.
        app.gc_messages.clear()
        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if app.gc_messages:
                break

        assert group._content_mounted is True
        assert len(group.query_one("#tg-content").children) == 1
        assert len(app.gc_messages) == 1
        message = app.gc_messages[0]
        assert isinstance(message, GcAbsorbRequested)
        assert message.reason is GcAbsorbReason.STABLE_CONTENT_MOUNTED
        assert message.terminal_boundary is False


@pytest.mark.asyncio
async def test_lazy_tool_expand_posts_stable_content_absorb_after_mounts_finish() -> None:
    app = GcMessageChatPanelApp()
    async with app.run_test() as pilot:
        panel = app.query_one(ChatPanel)
        await panel.add_user_message("edit")
        group = ToolGroup()
        await panel.mount(group)
        await group.add_collapsed_replay_tool(
            "edit1",
            "edit_file",
            "filesystem.write",
            args={"path": "src/app.py"},
            result="Edited src/app.py",
            file_snapshot=("old\n", "new\n"),
            lazy=True,
        )
        await pilot.pause()

        group.collapsed = False
        for _ in range(300):
            await pilot.pause()
            if any(isinstance(message, GcAbsorbRequested) for message in app.gc_messages):
                break

        absorbs = [message for message in app.gc_messages if isinstance(message, GcAbsorbRequested)]
        assert len(absorbs) == 1
        assert absorbs[0].reason is GcAbsorbReason.STABLE_CONTENT_MOUNTED
        assert absorbs[0].terminal_boundary is False


@pytest.mark.asyncio
async def test_deferred_live_subagent_markdown_posts_absorb_after_trailing_refresh() -> None:
    register_kind_renderer(KIND_SUB_AGENT, SubAgentToolCall)
    app = GcMessageChatPanelApp()
    large_result = "# Result\n\n" + "\n".join(f"- item {index}" for index in range(2_000))
    async with app.run_test() as pilot:
        panel = app.query_one(ChatPanel)
        panel.set_tool_kinds({"explore_agent": KIND_SUB_AGENT})
        await panel.add_user_message("delegate")
        await panel.add_tool_start("sa1", "explore_agent", KIND_SUB_AGENT, args={"prompt": "investigate"})
        await panel.add_tool_start("sibling", "plain_tool", "", args={"value": 1})
        await panel.add_tool_result("sa1", "explore_agent", large_result, 2500)
        subagent = panel.query_one(SubAgentToolCall)
        if subagent._result_timer is not None:
            subagent._result_timer.stop()
            subagent._result_timer = None

        await subagent._mount_deferred_result()
        for _ in range(5):
            await pilot.pause()
            if app.gc_messages:
                break

        assert len(app.gc_messages) == 1
        message = app.gc_messages[0]
        assert isinstance(message, GcAbsorbRequested)
        assert message.reason is GcAbsorbReason.STABLE_CONTENT_MOUNTED


@pytest.mark.asyncio
async def test_tool_group_expand_restores_cards_before_pending_diff_prepare_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expand should not block cheap card restore on deferred diff preparation."""
    from chrys.app.tui.widgets.diff_view.inline import InlineUnifiedDiffLines

    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()
    original_prepare = InlineUnifiedDiffLines.prepare

    async def delayed_prepare(self: InlineUnifiedDiffLines) -> None:
        prepare_started.set()
        await release_prepare.wait()
        await original_prepare(self)

    monkeypatch.setattr(InlineUnifiedDiffLines, "prepare", delayed_prepare)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("edit")
        await cp.add_tool_start("edit1", "edit_file", "filesystem.write", args={"path": "src/app.py"})
        group = cp.query_one(ToolGroup)
        group.complete_tool(
            "edit1",
            "Edited src/app.py",
            25,
            file_snapshot=("old\n", "new\n"),
            lazy=True,
        )
        await cp.add_agent_message("done", is_final=True)

        for _ in range(200):
            await pilot.pause()
            if not group._content_mounted:
                break
        assert group._content_mounted is False

        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if group._content_mounted and len(group._tools) == 1:
                break

        assert group._content_mounted is True
        assert len(group._tools) == 1
        assert not list(cp.query(InlineUnifiedDiffLines))

        await asyncio.wait_for(prepare_started.wait(), timeout=5)
        assert group._content_mounted is True
        assert not list(cp.query(InlineUnifiedDiffLines))

        release_prepare.set()
        for _ in range(200):
            await pilot.pause()
            if list(cp.query(InlineUnifiedDiffLines)):
                break

        assert list(cp.query(InlineUnifiedDiffLines))


@pytest.mark.asyncio
async def test_rejected_file_tool_stays_text_after_collapse_expand_with_running_sibling() -> None:
    """Rejected file tools are text terminals, not lazy diff candidates."""
    from chrys.app.tui.widgets.diff_view import DiffView

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("tools")
        await cp.add_tool_start("edit1", "edit_file", "filesystem.write", args={"path": "src/app.py"})
        await cp.add_tool_result(
            "edit1",
            "edit_file",
            "Error: Tool execution was rejected by user.",
            25,
            approval="user_rejected",
            file_snapshot=("old\n", "new\n"),
        )
        await cp.add_tool_start("tool2", "plain_tool", "", args={"value": 2})
        await pilot.pause()

        group = cp.query_one(ToolGroup)
        rejected_tool = group._tools["edit1"]
        assert rejected_tool.has_class("-rejected")

        group.collapsed = True
        await pilot.pause()
        assert rejected_tool._diff_pending is False

        group.collapsed = False
        for _ in range(50):
            await pilot.pause()

        assert rejected_tool.has_class("-rejected")
        assert rejected_tool._diff_pending is False
        assert not list(rejected_tool.query(DiffView))
        content = rejected_tool.query_one("#ft-content")
        assert any(isinstance(child, Static) and "rejected" in str(child.content).lower() for child in content.children)


@pytest.mark.asyncio
async def test_shell_tool_streamed_tail_survives_prune_rebuild() -> None:
    """Shell cards should rebuild with the streamed output tail, not the raw result head."""
    from chrys.app.tui.widgets.chat.renderers.execute import ExecuteToolCall

    def _output_lines(tool: ExecuteToolCall) -> list[str]:
        content = tool.query_one("#exec-panel", Static).content
        if isinstance(content, Text):
            return content.plain.splitlines()
        return str(content).splitlines()

    result_lines = [f"line{i:02d}" for i in range(1, 21)]
    expected_tail = result_lines[-10:]

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("run")
        await cp.add_tool_start("sh1", "bash", KIND_SHELL, args={"command": "printf lines"})
        cp.update_tool_progress("sh1", result_lines)
        await cp.add_tool_result("sh1", "bash", "\n".join(result_lines) + "\n[exit_code: 0]", 123)
        await pilot.pause()

        shell = cp.query_one(ExecuteToolCall)
        assert _output_lines(shell) == expected_tail

        await cp.add_agent_message("done", is_final=True)
        group = cp.query_one(ToolGroup)
        for _ in range(200):
            await pilot.pause()
            if not group._content_mounted:
                break

        assert group.collapsed is True
        assert group._content_mounted is False
        assert len(group._tools) == 0

        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if group._content_mounted:
                break

        rebuilt = group._tools["sh1"]
        assert isinstance(rebuilt, ExecuteToolCall)
        assert _output_lines(rebuilt) == expected_tail


@pytest.mark.asyncio
async def test_shell_tool_timeout_result_overrides_streamed_progress_on_rebuild() -> None:
    """Timeout results should render as errors even when live progress was already streamed."""
    from chrys.app.tui.widgets.chat.renderers.execute import ExecuteToolCall

    def _output_text(tool: ExecuteToolCall) -> str:
        content = tool.query_one("#exec-panel", Static).content
        return content.plain if isinstance(content, Text) else str(content)

    result = (
        "Error: command timed out after 180 seconds.\n[partial output]\n1009 tests collected in 0.35s\n[exit_code: 0]"
    )

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("run")
        await cp.add_tool_start("sh1", "bash", KIND_SHELL, args={"command": "pytest", "timeout": 180})
        cp.update_tool_progress("sh1", ["1009 tests collected in 0.35s"])
        await cp.add_tool_result("sh1", "bash", result, 180000, metadata={SHELL_TIMED_OUT_METADATA_KEY: True})
        await pilot.pause()

        shell = cp.query_one(ExecuteToolCall)
        panel = shell.query_one("#exec-panel")
        assert shell.status == "error"
        assert shell.has_class("-error")
        assert not shell.has_class("-success")
        assert str(panel.border_subtitle) == "Errored"
        assert _output_text(shell).startswith("Error: command timed out")

        await cp.add_agent_message("done", is_final=True)
        group = cp.query_one(ToolGroup)
        for _ in range(200):
            await pilot.pause()
            if not group._content_mounted:
                break

        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if group._content_mounted:
                break

        rebuilt = group._tools["sh1"]
        assert isinstance(rebuilt, ExecuteToolCall)
        rebuilt_panel = rebuilt.query_one("#exec-panel")
        assert rebuilt.status == "error"
        assert rebuilt.has_class("-error")
        assert not rebuilt.has_class("-success")
        assert str(rebuilt_panel.border_subtitle) == "Errored"
        assert _output_text(rebuilt).startswith("Error: command timed out")


@pytest.mark.asyncio
async def test_shell_tool_non_timeout_failure_keeps_streamed_tail() -> None:
    """Failing shell commands with exit codes should keep the streamed tail summary."""
    from chrys.app.tui.widgets.chat.renderers.execute import ExecuteToolCall

    def _output_text(tool: ExecuteToolCall) -> str:
        content = tool.query_one("#exec-panel", Static).content
        return content.plain if isinstance(content, Text) else str(content)

    progress_lines = [f"progress {index}" for index in range(12)]
    result = "banner\nsetup\n...\nFAILED tests/test_example.py::test_case\n[exit_code: 1]"

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("run")
        await cp.add_tool_start("sh1", "bash", KIND_SHELL, args={"command": "pytest"})
        cp.update_tool_progress("sh1", progress_lines)
        await cp.add_tool_result("sh1", "bash", result, 1200)
        await pilot.pause()

        shell = cp.query_one(ExecuteToolCall)
        panel = shell.query_one("#exec-panel")
        assert shell.status == "error"
        assert str(panel.border_subtitle) == "Errored [1]"
        assert _output_text(shell) == "\n".join(progress_lines[-10:])


@pytest.mark.asyncio
async def test_shell_tool_exit_code_zero_overrides_error_prefixed_stdout_on_rebuild() -> None:
    """Shell stdout may start with ``Error:``; a trailing zero exit code remains success."""
    from chrys.app.tui.widgets.chat.renderers.execute import ExecuteToolCall

    result = "Error: expected message from stdout\n[exit_code: 0]"

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("run")
        await cp.add_tool_start("sh1", "bash", KIND_SHELL, args={"command": "printf error"})
        await cp.add_tool_result("sh1", "bash", result, 25)
        await pilot.pause()

        shell = cp.query_one(ExecuteToolCall)
        assert shell.status == "complete"
        assert shell.has_class("-success")
        assert not shell.has_class("-error")

        await cp.add_agent_message("done", is_final=True)
        group = cp.query_one(ToolGroup)
        for _ in range(200):
            await pilot.pause()
            if not group._content_mounted:
                break

        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if group._content_mounted:
                break

        rebuilt = group._tools["sh1"]
        assert isinstance(rebuilt, ExecuteToolCall)
        assert rebuilt.status == "complete"
        assert rebuilt.has_class("-success")
        assert not rebuilt.has_class("-error")


@pytest.mark.asyncio
async def test_shell_tool_structured_exit_code_zero_overrides_error_text_without_suffix() -> None:
    """Structured shell metadata is authoritative even when legacy suffix text is absent."""
    from chrys.app.tui.widgets.chat.renderers.execute import ExecuteToolCall

    result = "Error: expected message from stdout"
    metadata = {SHELL_EXIT_CODE_METADATA_KEY: 0}

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("run")
        await cp.add_tool_start("sh1", "bash", KIND_SHELL, args={"command": "printf error"})
        await cp.add_tool_result("sh1", "bash", result, 25, metadata=metadata)
        await pilot.pause()

        shell = cp.query_one(ExecuteToolCall)
        assert shell.status == "complete"
        assert shell.has_class("-success")
        assert not shell.has_class("-error")

        await cp.add_agent_message("done", is_final=True)
        group = cp.query_one(ToolGroup)
        for _ in range(200):
            await pilot.pause()
            if not group._content_mounted:
                break

        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if group._content_mounted:
                break

        rebuilt = group._tools["sh1"]
        assert isinstance(rebuilt, ExecuteToolCall)
        assert rebuilt.status == "complete"
        assert rebuilt.has_class("-success")
        assert not rebuilt.has_class("-error")


@pytest.mark.asyncio
async def test_shell_tool_user_rejection_status_is_rejected_not_error() -> None:
    from chrys.app.tui.widgets.chat.renderers.execute import ExecuteToolCall

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_tool_start("sh1", "bash", KIND_SHELL, args={"command": "rm file"})
        await cp.add_tool_result(
            "sh1",
            "bash",
            "Error: Tool execution was rejected by user.",
            25,
            approval="user_rejected",
        )
        await pilot.pause()

        shell = cp.query_one(ExecuteToolCall)
        assert shell.status == "rejected"
        assert shell.has_class("-rejected")
        assert not shell.has_class("-error")


@pytest.mark.asyncio
async def test_shell_tool_structured_nonzero_exit_code_renders_error_without_suffix() -> None:
    """Structured shell exit-code metadata should not require parsing formatted output text."""
    from chrys.app.tui.widgets.chat.renderers.execute import ExecuteToolCall

    result = "boom"
    metadata = {SHELL_EXIT_CODE_METADATA_KEY: 42}

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("run")
        await cp.add_tool_start("sh1", "bash", KIND_SHELL, args={"command": "false"})
        await cp.add_tool_result("sh1", "bash", result, 1234, metadata=metadata)
        await pilot.pause()

        shell = cp.query_one(ExecuteToolCall)
        panel = shell.query_one("#exec-panel")
        assert shell.status == "error"
        assert shell.has_class("-error")
        assert not shell.has_class("-success")
        assert str(panel.border_subtitle) == "Errored [42]"


@pytest.mark.asyncio
async def test_shell_tool_exit_suffix_removal_preserves_leading_whitespace() -> None:
    """Removing a shell exit-code suffix must not trim indentation or leave fake blank rows."""
    from chrys.app.tui.widgets.chat.renderers.execute import ExecuteToolCall

    result = "  indented\nstill body\n\n[exit_code: 0]"

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("run")
        await cp.add_tool_start("sh1", "bash", KIND_SHELL, args={"command": "printf indent"})
        await cp.add_tool_result("sh1", "bash", result, 25)
        await pilot.pause()

        shell = cp.query_one(ExecuteToolCall)
        content = shell.query_one("#exec-panel", Static).content
        text = content.plain if isinstance(content, Text) else str(content)
        assert text == "  indented\nstill body"
        assert not text.endswith("\n")


@pytest.mark.asyncio
async def test_generic_mcp_tool_error_prefixed_success_stays_complete() -> None:
    """External/MCP output can begin with ``Error:`` without meaning the tool failed."""
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_tool_start("mcp1", "external_tool", KIND_MCP, args={"query": "status"})
        await cp.add_tool_result("mcp1", "external_tool", "Error: expected text from remote tool", 17)
        await pilot.pause()

        tool = cp.query_one(ToolCall)
        assert tool.tool_kind == KIND_MCP
        assert tool.status == "complete"
        assert tool.has_class("-success")
        assert not tool.has_class("-error")


@pytest.mark.asyncio
async def test_unkinded_context_tool_structured_failure_renders_error() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_tool_start("ctx1", "compress_context", "", args={"marker_id": "missing"})
        await cp.add_tool_result(
            "ctx1",
            "compress_context",
            "Error: marker_id not found",
            17,
            metadata={TOOL_FAILED_METADATA_KEY: True},
        )
        await pilot.pause()

        tool = cp.query_one(ToolCall)
        assert tool.tool_kind == ""
        assert tool.status == "error"
        assert tool.has_class("-error")
        assert not tool.has_class("-success")


@pytest.mark.asyncio
async def test_hook_denied_generic_tool_renders_rejected_not_errored() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_tool_start("hook1", "custom_tool", "", args={"value": 1})
        await cp.add_tool_result(
            "hook1",
            "custom_tool",
            "Error: blocked by policy",
            17,
            metadata={TOOL_ERROR_KIND_METADATA_KEY: "hook_denied"},
        )
        await pilot.pause()

        tool = cp.query_one(ToolCall)
        assert tool.status == "rejected"
        assert tool.has_class("-rejected")
        assert not tool.has_class("-error")
        panel = tool.query_one("#tc-panel")
        assert panel.border_subtitle == "Rejected"


@pytest.mark.asyncio
async def test_legacy_unkinded_chrys_tool_error_text_renders_error() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_tool_start("ctx1", "compress_context", "", args={"marker_id": "remote"})
        await cp.add_tool_result("ctx1", "compress_context", "Error: literal remote payload", 17)
        await pilot.pause()

        tool = cp.query_one(ToolCall)
        assert tool.tool_kind == ""
        assert tool.status == "error"
        assert tool.has_class("-error")
        assert not tool.has_class("-success")


@pytest.mark.asyncio
async def test_tool_result_failed_false_suppresses_error_text_fallback() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_tool_start("read1", "custom_read", KIND_FILESYSTEM_READ, args={"path": "a.txt"})
        await cp.add_tool_result(
            "read1",
            "custom_read",
            "Error: expected literal file contents",
            17,
            metadata={TOOL_FAILED_METADATA_KEY: False},
        )
        await pilot.pause()

        tool = cp.query_one(ToolCall)
        assert tool.status == "complete"
        assert tool.has_class("-success")
        assert not tool.has_class("-error")


@pytest.mark.asyncio
async def test_read_file_renderer_shows_parsed_error_even_when_structured_metadata_says_success() -> None:
    from chrys.app.tui.widgets.chat.renderers.read_file import ReadFileToolCall

    class ToolApp(_LocalizedApp):
        def compose(self) -> ComposeResult:
            yield ReadFileToolCall("read1", "read_file", args={"path": "missing.txt"})

    async with ToolApp().run_test() as pilot:
        tool = pilot.app.query_one(ReadFileToolCall)

        tool.set_complete(
            "Error: file not found",
            duration_ms=10,
            metadata={TOOL_FAILED_METADATA_KEY: False},
        )
        await pilot.pause()

        assert tool.status == "error"
        assert tool.has_class("-error")
        assert not tool.has_class("-success")
        assert tool.query_one("#rf-panel", Static).render().plain == "Error: file not found"


@pytest.mark.asyncio
async def test_tool_result_failed_true_marks_external_tool_failed() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_tool_start("mcp1", "external_tool", KIND_MCP, args={"query": "status"})
        await cp.add_tool_result(
            "mcp1",
            "external_tool",
            "completed text",
            17,
            metadata={TOOL_FAILED_METADATA_KEY: True},
        )
        await pilot.pause()

        tool = cp.query_one(ToolCall)
        assert tool.status == "error"
        assert tool.has_class("-error")
        assert not tool.has_class("-success")


@pytest.mark.asyncio
async def test_generic_chrys_tool_error_prefixed_result_renders_error() -> None:
    """Fallback renderers for Chrys-owned tool kinds still use the ``Error:`` convention."""
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_tool_start("read1", "custom_read", KIND_FILESYSTEM_READ, args={"path": "missing.txt"})
        await cp.add_tool_result("read1", "custom_read", "Error: file not found", 17)
        await pilot.pause()

        tool = cp.query_one(ToolCall)
        assert tool.tool_kind == KIND_FILESYSTEM_READ
        assert tool.status == "error"
        assert tool.has_class("-error")
        assert not tool.has_class("-success")


@pytest.mark.asyncio
async def test_shell_tool_nonzero_exit_rebuild_uses_complete_path_metadata() -> None:
    """Non-zero shell results should replay through set_complete, preserving parsed display state."""
    from chrys.app.tui.widgets.chat.renderers.execute import ExecuteToolCall

    def _output_text(tool: ExecuteToolCall) -> str:
        content = tool.query_one("#exec-panel", Static).content
        return content.plain if isinstance(content, Text) else str(content)

    result = "boom\n[exit_code: 42]"

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("run")
        await cp.add_tool_start("sh1", "bash", KIND_SHELL, args={"command": "false"})
        await cp.add_tool_result("sh1", "bash", result, 1234)
        await pilot.pause()

        shell = cp.query_one(ExecuteToolCall)
        panel = shell.query_one("#exec-panel")
        assert shell.status == "error"
        assert shell.duration_ms == 1234
        assert str(panel.border_subtitle) == "Errored [42]"
        assert _output_text(shell) == "boom"

        await cp.add_agent_message("done", is_final=True)
        group = cp.query_one(ToolGroup)
        for _ in range(200):
            await pilot.pause()
            if not group._content_mounted:
                break

        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if group._content_mounted:
                break

        rebuilt = group._tools["sh1"]
        assert isinstance(rebuilt, ExecuteToolCall)
        rebuilt_panel = rebuilt.query_one("#exec-panel")
        assert rebuilt.status == "error"
        assert rebuilt.duration_ms == 1234
        assert str(rebuilt_panel.border_subtitle) == "Errored [42]"
        assert _output_text(rebuilt) == "boom"


@pytest.mark.asyncio
async def test_tool_group_restore_reprunes_when_collapsed_mid_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rapid expand/collapse during async restore must not leave children mounted."""
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("tools")
        await cp.add_tool_start("tool1", "plain_tool", "", args={"value": 1})
        await cp.add_tool_result("tool1", "plain_tool", "done", 10)
        await cp.add_agent_message("done", is_final=True)

        group = cp.query_one(ToolGroup)
        for _ in range(200):
            await pilot.pause()
            if not group._content_mounted:
                break
        assert group.collapsed is True
        assert group._content_mounted is False

        content = group.query_one("#tg-content")
        original_mount = content.mount

        def _drop_call_later(*args, **kwargs):
            return None

        async def _mount_then_collapse(*args, **kwargs):
            result = await original_mount(*args, **kwargs)
            group.collapsed = True
            return result

        monkeypatch.setattr(group, "call_later", _drop_call_later)
        monkeypatch.setattr(content, "mount", _mount_then_collapse)

        group.collapsed = False
        await group._restore_completed_tool_widgets()
        await pilot.pause()

        assert group.collapsed is True
        assert group._content_mounted is False
        assert len(group._tools) == 0
        assert not list(content.children)


@pytest.mark.asyncio
async def test_tool_group_release_restores_when_expand_races_awaited_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-expanding while subtree removal is suspended must not leave an empty group."""
    app = GcMessageChatPanelApp()
    async with app.run_test() as pilot:
        panel = app.query_one(ChatPanel)
        await panel.add_user_message("tools")
        await panel.add_tool_start("tool1", "plain_tool", "", args={"value": 1})
        await panel.add_tool_result("tool1", "plain_tool", "done", 10)
        group = panel.query_one(ToolGroup)
        content = group.query_one("#tg-content")
        original_remove_children = content.remove_children
        removal_started = asyncio.Event()
        release_removal = asyncio.Event()

        def delayed_remove_children():
            pending_remove = original_remove_children()
            removal_started.set()

            async def wait_for_release():
                await release_removal.wait()
                return await pending_remove

            return wait_for_release()

        monkeypatch.setattr(group, "call_later", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(content, "remove_children", delayed_remove_children)
        group.collapsed = True
        release_task = asyncio.create_task(group._release_completed_tool_widgets())
        await removal_started.wait()

        group.collapsed = False
        release_removal.set()
        await release_task
        for _ in range(5):
            await pilot.pause()
            if len(app.gc_messages) == 2:
                break

        assert group.collapsed is False
        assert group._content_mounted is True
        assert list(content.children)
        assert group.get_tool("tool1") is not None
        assert len(app.gc_messages) == 2
        assert isinstance(app.gc_messages[0], GcReclaimRequested)
        assert app.gc_messages[0].reason is GcReclaimReason.STABLE_CONTENT_REMOVED
        assert isinstance(app.gc_messages[1], GcAbsorbRequested)
        assert app.gc_messages[1].reason is GcAbsorbReason.STABLE_CONTENT_MOUNTED


@pytest.mark.asyncio
async def test_tool_group_release_restores_live_tool_added_during_awaited_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool added after release starts must remain mounted and running."""
    app = GcMessageChatPanelApp()
    async with app.run_test() as pilot:
        panel = app.query_one(ChatPanel)
        await panel.add_user_message("tools")
        await panel.add_tool_start("old", "plain_tool", "", args={"value": 1})
        await panel.add_tool_result("old", "plain_tool", "done", 10)
        group = panel.query_one(ToolGroup)
        content = group.query_one("#tg-content")
        original_remove_children = content.remove_children
        removal_started = asyncio.Event()
        release_removal = asyncio.Event()

        def delayed_remove_children():
            pending_remove = original_remove_children()
            removal_started.set()

            async def wait_for_release():
                await release_removal.wait()
                return await pending_remove

            return wait_for_release()

        monkeypatch.setattr(group, "call_later", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(content, "remove_children", delayed_remove_children)
        group.collapsed = True
        release_task = asyncio.create_task(group._release_completed_tool_widgets())
        await removal_started.wait()

        add_task = asyncio.create_task(group.add_tool("new", "plain_tool", "", args={"value": 2}))
        await pilot.pause()
        assert add_task.done() is False
        assert group.is_tool_running("new") is True
        release_removal.set()
        await release_task
        await add_task
        await pilot.pause()

        assert group.collapsed is True
        assert group._content_mounted is True
        assert group.is_tool_running("new") is True
        assert group.get_tool("new") is not None
        assert set(group._tools) == {"old", "new"}
        assert len(content.children) == 2
        assert len(app.gc_messages) == 1
        assert isinstance(app.gc_messages[0], GcReclaimRequested)


@pytest.mark.asyncio
async def test_tool_group_release_normalizes_tool_completed_during_awaited_removal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A racing tool that finishes during prune must leave no orphaned child."""
    app = GcMessageChatPanelApp()
    async with app.run_test() as pilot:
        panel = app.query_one(ChatPanel)
        await panel.add_user_message("tools")
        await panel.add_tool_start("old", "plain_tool", "", args={"value": 1})
        await panel.add_tool_result("old", "plain_tool", "done", 10)
        group = panel.query_one(ToolGroup)
        content = group.query_one("#tg-content")
        original_remove_children = content.remove_children
        removal_started = asyncio.Event()
        release_removal = asyncio.Event()

        def delayed_remove_children():
            pending_remove = original_remove_children()
            removal_started.set()

            async def wait_for_release():
                await release_removal.wait()
                return await pending_remove

            return wait_for_release()

        monkeypatch.setattr(group, "call_later", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(content, "remove_children", delayed_remove_children)
        group.collapsed = True
        release_task = asyncio.create_task(group._release_completed_tool_widgets())
        await removal_started.wait()

        add_task = asyncio.create_task(group.add_tool("new", "plain_tool", "", args={"value": 2}))
        await pilot.pause()
        group.complete_tool("new", "done", 20)
        assert group.all_complete is True
        release_removal.set()
        await release_task
        await add_task
        await pilot.pause()

        assert group._content_mounted is False
        assert group._tools == {}
        assert not content.children

        group.collapsed = False
        await group._restore_completed_tool_widgets()
        await pilot.pause()

        assert set(group._tools) == {"old", "new"}
        assert len(content.children) == 2
        assert all(tool.status != "running" for tool in group._tools.values())


@pytest.mark.asyncio
async def test_tool_group_serializes_add_with_in_progress_full_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent add must not re-enter a full restore or duplicate records."""
    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        group = ToolGroup()
        await panel.mount(group)
        monkeypatch.setattr(group, "call_later", lambda *_args, **_kwargs: None)
        await group.add_collapsed_replay_tool(
            "old",
            "plain_tool",
            "",
            result="done",
        )
        content = group.query_one("#tg-content")
        original_mount = content.mount
        mount_started = asyncio.Event()
        release_mount = asyncio.Event()
        delayed_once = False

        def delayed_mount(*widgets, **kwargs):
            nonlocal delayed_once
            pending_mount = original_mount(*widgets, **kwargs)
            if delayed_once:
                return pending_mount
            delayed_once = True
            mount_started.set()

            async def wait_for_release():
                await release_mount.wait()
                return await pending_mount

            return wait_for_release()

        monkeypatch.setattr(content, "mount", delayed_mount)
        group.collapsed = False
        restore_task = asyncio.create_task(group._restore_completed_tool_widgets())
        await mount_started.wait()

        add_task = asyncio.create_task(group.add_tool("new", "plain_tool", "", args={"value": 2}))
        await pilot.pause()
        assert add_task.done() is False
        release_mount.set()
        await restore_task
        await add_task
        await pilot.pause()

        assert set(group._tools) == {"old", "new"}
        assert len(content.children) == 2
        assert len({id(child) for child in content.children}) == 2
        assert group.get_tool("new") is not None
        assert group.is_tool_running("new") is True


@pytest.mark.asyncio
async def test_collapsed_replay_writer_waits_for_content_structure_lock() -> None:
    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        group = ToolGroup()
        await panel.mount(group)
        await group._content_structure_lock.acquire()
        add_task = asyncio.create_task(
            group.add_collapsed_replay_tool(
                "replay",
                "plain_tool",
                "",
                result="done",
            )
        )
        try:
            await pilot.pause()
            assert add_task.done() is False
            assert "replay" not in group._tool_records
        finally:
            group._content_structure_lock.release()
        await add_task

        assert group._tool_records["replay"].result == "done"
        assert group._content_mounted is False


def test_tool_call_completion_before_compose_is_fail_soft() -> None:
    tool = ToolCall("late", "plain_tool")

    tool.set_complete("done", 10)

    assert tool.status == "complete"
    assert tool.result_text == "done"
    assert tool.has_class("-done")
    assert tool.has_class("-success")


@pytest.mark.asyncio
async def test_tool_group_replay_metadata_approval_renders_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("tools")
        group = ToolGroup()
        await cp.mount(group)
        monkeypatch.setattr(group, "_schedule_pending_content_mounts", lambda **_kwargs: None)
        await group.add_collapsed_replay_tool(
            "reject1",
            "plain_tool",
            "",
            result="Error: rejected",
            duration_ms=10,
            metadata={TOOL_FAILED_METADATA_KEY: True, "approval": "user_rejected"},
        )
        await pilot.pause()

        assert group._tool_records["reject1"].status == "rejected"

        group.collapsed = False
        await group._restore_completed_tool_widgets()
        await pilot.pause()

        tool = group._tools["reject1"]
        assert isinstance(tool, ToolCall)
        assert tool.status == "rejected"
        assert tool.has_class("-rejected")
        assert not tool.has_class("-error")


@pytest.mark.asyncio
async def test_mount_pending_diffs_survives_collapse_during_await() -> None:
    """Pending diff mounting must tolerate collapse pruning the tool map mid-loop."""

    class _PendingTool(Static):
        status = "complete"

        def __init__(self, group: ToolGroup, events: list[str], name: str) -> None:
            super().__init__(name)
            self.group = group
            self.events = events
            self._label = name

        async def mount_diff_if_pending(self) -> bool:
            self.events.append(self._label)
            if self._label == "first":
                self.group.collapsed = True
                await self.group._release_completed_tool_widgets()
            await asyncio.sleep(0)
            return True

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("tools")
        group = ToolGroup()
        await cp.mount(group)
        events: list[str] = []
        first = _PendingTool(group, events, "first")
        second = _PendingTool(group, events, "second")
        group._tool_records["first"] = group._tool_records["second"] = None  # type: ignore[assignment]
        group._tools = {"first": first, "second": second}  # type: ignore[assignment]
        group._done = 2
        await group.query_one("#tg-content").mount(first)
        await group.query_one("#tg-content").mount(second)

        await group._mount_pending_diffs()
        await pilot.pause()

        assert events == ["first"]
        assert group.collapsed is True


@pytest.mark.asyncio
async def test_collapsed_running_tool_group_prunes_when_last_tool_completes() -> None:
    """Manual collapse before completion should prune once the final result arrives."""
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("tools")
        await cp.add_tool_start("tool1", "plain_tool", "", args={"value": 1})
        await cp.add_tool_start("tool2", "plain_tool", "", args={"value": 2})

        group = cp.query_one(ToolGroup)
        group.collapsed = True
        await pilot.pause()
        assert group._content_mounted is True

        await cp.add_tool_result("tool1", "plain_tool", "one", 10)
        await pilot.pause()
        assert group._content_mounted is True

        await cp.add_tool_result("tool2", "plain_tool", "two", 10)
        for _ in range(200):
            await pilot.pause()
            if not group._content_mounted:
                break

        assert group.collapsed is True
        assert group.all_complete is True
        assert group._content_mounted is False
        assert len(group._tools) == 0


@pytest.mark.asyncio
async def test_pruned_open_tool_group_restores_completed_tools_when_new_tool_starts() -> None:
    """A later parallel start must not make earlier pruned tools disappear."""
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("tools")
        await cp.add_tool_start("a", "plain_tool", "", args={"value": "a"})
        await cp.add_tool_result("a", "plain_tool", "a done", 10)

        group = cp.query_one(ToolGroup)
        group.collapsed = True
        for _ in range(200):
            await pilot.pause()
            if not group._content_mounted:
                break
        assert group.collapsed is True
        assert group._content_mounted is False
        assert len(group._tools) == 0
        assert set(group._tool_records) == {"a"}

        await cp.add_tool_start("b", "plain_tool", "", args={"value": "b"})
        await pilot.pause()

        assert group.collapsed is True
        assert group._content_mounted is True
        assert set(group._tools) == {"a", "b"}
        assert group._tools["a"].status == "complete"
        assert group._tools["b"].status == "running"

        group.collapsed = False
        await pilot.pause()

        assert set(group._tools) == {"a", "b"}
        assert [child.call_id for child in group.query(ToolCall)] == ["a", "b"]


@pytest.mark.asyncio
async def test_replay_history_defers_tool_renderer_creation_until_expand(monkeypatch: pytest.MonkeyPatch) -> None:
    from chrys.app.tui.widgets.chat import tool_renderers as tool_renderers_module

    created: list[tuple[str, str, str, dict[str, object] | None]] = []

    def create_tool_widget(
        call_id: str,
        tool_name: str,
        tool_kind: str,
        args_summary: str = "",
        args: dict[str, object] | None = None,
    ) -> ToolCall:
        created.append((call_id, tool_name, tool_kind, args))
        return ToolCall(call_id, tool_name, args_summary, args=args)

    monkeypatch.setattr(tool_renderers_module, "create_tool_widget", create_tool_widget)
    messages = [
        {"role": "user", "contents": [{"type": "text", "text": "run tool"}]},
        {
            "role": "assistant",
            "contents": [
                {
                    "type": "function_call",
                    "name": "plain_tool",
                    "call_id": "tool1",
                    "arguments": {"value": 1},
                    "additional_properties": {
                        TRAJECTORY_TIMING_KEY: {
                            "started_at": "2026-08-19T01:02:03+00:00",
                            "finished_at": "2026-08-19T01:02:04+00:00",
                            "duration_ms": 987,
                        }
                    },
                }
            ],
        },
        {"role": "tool", "contents": [{"type": "function_result", "call_id": "tool1", "result": "done"}]},
    ]

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.replay_history(messages)
        await pilot.pause()

        group = cp.query_one(ToolGroup)
        assert created == []
        assert group.collapsed is True
        assert group._content_mounted is False
        assert len(group._tools) == 0
        assert len(group._tool_records) == 1
        assert group._tool_records["tool1#0"].duration_ms == 987
        assert group._tool_records["tool1#0"].duration_known is True
        expected_tool_timestamp = format_message_created_at("2026-08-19T01:02:03+00:00")
        assert group._tool_records["tool1#0"].timestamp == expected_tool_timestamp
        assert group._elapsed_ms() == 0
        assert group._timer is None

        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if created:
                break

        assert created == [("tool1#0", "plain_tool", "", {"value": 1})]
        assert group._content_mounted is True
        tool = group.get_tool("tool1#0")
        assert isinstance(tool, ToolCall)
        assert tool.query_one(ToolCardHeader)._label_renderable().plain.endswith(f"(987ms) {expected_tool_timestamp}")


@pytest.mark.parametrize(("provider_hosted", "duration_ms"), [(False, 0), (True, 987)])
async def test_replay_tool_timing_is_visible_for_zero_and_hosted_durations(
    provider_hosted: bool,
    duration_ms: int,
) -> None:
    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        group = ToolGroup()
        await panel.mount(group)
        await group.add_collapsed_replay_tool(
            "timed",
            "server_task" if provider_hosted else "plain_tool",
            "",
            result="done",
            duration_ms=duration_ms,
            duration_known=True,
            timestamp="- 9:02 AM",
            provider_hosted=provider_hosted,
            hosted_family="generic" if provider_hosted else "",
            provider="openai" if provider_hosted else "",
            canonical_status="completed",
            lazy=True,
        )

        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if group._content_mounted and group._tools:
                break

        header = next(iter(group._tools.values())).query_one(ToolCardHeader)
        expected_duration = f"({duration_ms}ms)"
        assert header._label_renderable().plain.count(expected_duration) == 1
        assert header._label_renderable().plain.endswith(f"{expected_duration} - 9:02 AM")


@pytest.mark.asyncio
async def test_replay_user_hides_zero_duration_event_suffix() -> None:
    """User timing stays persisted but its definitionally-zero duration is visual noise."""
    messages = [
        {
            "role": "user",
            "contents": [{"type": "text", "text": "hello"}],
            "additional_properties": {
                TRAJECTORY_TIMING_KEY: {
                    "started_at": "2026-08-19T01:02:03+00:00",
                    "finished_at": "2026-08-19T01:02:03+00:00",
                    "duration_ms": 0,
                }
            },
        }
    ]

    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(messages)
        await pilot.pause()

        header = panel.query_one(_UserMessageText).render().plain.splitlines()[0]
        assert "(0ms)" not in header
        assert " - " in header


@pytest.mark.parametrize("provider_hosted", [False, True])
async def test_replay_failed_tool_timing_is_visible(provider_hosted: bool) -> None:
    """Error renderers do not receive set_complete, so replay adds their duration."""
    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        group = ToolGroup()
        await panel.mount(group)
        await group.add_collapsed_replay_tool(
            "failed",
            "server_task" if provider_hosted else "plain_tool",
            "",
            result="Error: failed",
            duration_ms=987,
            duration_known=True,
            timestamp="- 9:02 AM",
            provider_hosted=provider_hosted,
            hosted_family="generic" if provider_hosted else "",
            provider="openai" if provider_hosted else "",
            canonical_status="failed" if provider_hosted else "completed",
            lazy=True,
        )

        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if group._content_mounted and group._tools:
                break

        header = next(iter(group._tools.values())).query_one(ToolCardHeader)
        assert header._label_renderable().plain.count("(987ms)") == 1
        assert header._label_renderable().plain.endswith("(987ms) - 9:02 AM")


@pytest.mark.asyncio
async def test_replay_failed_subagent_uses_persisted_duration() -> None:
    """A rebuilt sub-agent error must not replace persisted timing with widget elapsed time."""
    register_kind_renderer(KIND_SUB_AGENT, SubAgentToolCall)
    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        group = ToolGroup()
        await panel.mount(group)
        await group.add_collapsed_replay_tool(
            "failed-subagent",
            "explore_agent",
            KIND_SUB_AGENT,
            result="Error: sub-agent failed",
            duration_ms=987,
            duration_known=True,
            timestamp="- 9:02 AM",
            metadata={TOOL_FAILED_METADATA_KEY: True},
            lazy=True,
        )

        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if group._content_mounted and group._tools:
                break

        tool = next(iter(group._tools.values()))
        assert isinstance(tool, SubAgentToolCall)
        header = tool.query_one(ToolCardHeader)._label_renderable().plain
        assert header.count("(987ms)") == 1
        assert header.endswith("(987ms) - 9:02 AM")


@pytest.mark.asyncio
async def test_live_zero_duration_tool_does_not_gain_replay_suffix_after_rebuild() -> None:
    """Rebuilding a live card must preserve its original zero-duration label."""
    async with ChatPanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.add_user_message("run")
        await panel.add_tool_start("live", "plain_tool", "", args={})
        await panel.add_tool_result("live", "plain_tool", "done", 0)
        await panel.add_agent_message("finished", is_final=True)

        group = panel.query_one(ToolGroup)
        for _ in range(200):
            await pilot.pause()
            if not group._content_mounted:
                break

        assert group._content_mounted is False
        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if group._content_mounted and group._tools:
                break

        header = group.get_tool("live").query_one(ToolCardHeader)
        assert "(0ms)" not in header._label_renderable().plain


@pytest.mark.asyncio
async def test_replay_history_hatches_batch_until_descendants_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mounted replay root stays hatched while async markdown setup is incomplete."""
    update_started = asyncio.Event()
    release_update = asyncio.Event()
    original_update = VirtualizedMarkdown.update

    def delayed_update(self: VirtualizedMarkdown, markdown: str) -> AwaitComplete:
        async def await_update() -> None:
            update_started.set()
            await release_update.wait()
            await original_update(self, markdown)

        return AwaitComplete(await_update())

    monkeypatch.setattr(VirtualizedMarkdown, "update", delayed_update)
    messages = [
        {
            "role": "assistant",
            "contents": [{"type": "text", "text": "# Restored\n\nPending content"}],
        }
    ]

    async with ChatPanelApp().run_test(size=(80, 24)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        progress: list[tuple[int, int]] = []
        cp.set_replay_progress_callback(lambda current, total: progress.append((current, total)))
        replay = asyncio.create_task(cp.replay_history(messages))

        await asyncio.wait_for(update_started.wait(), timeout=1)
        message = cp.query_one(AgentMessage)
        hatch = message.styles.hatch

        try:
            for _ in range(40):
                if message.size.width > 0 and message.size.height > 0:
                    break
                await asyncio.sleep(0.01)
            assert message.has_class(chat_panel_module._REPLAY_PLACEHOLDER_CLASS)
            assert hatch != "none"
            assert hatch[0] == HATCH_GLYPH
            assert progress == [(0, 1)]
            rendered = message.render_lines(Region(0, 0, message.size.width, min(3, message.size.height)))
            assert any(HATCH_GLYPH in strip.text for strip in rendered)
        finally:
            release_update.set()
        await asyncio.wait_for(replay, timeout=1)
        await pilot.pause()

        assert not message.has_class(chat_panel_module._REPLAY_PLACEHOLDER_CLASS)
        assert not message.styles.has_rule("hatch")
        assert progress == [(0, 1), (1, 1)]


@pytest.mark.asyncio
async def test_replay_history_bounds_each_textual_registration_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Large restores must yield between bounded registration bursts."""
    batch_size = chat_panel_module._REPLAY_MOUNT_BATCH_SIZE
    messages = [
        {"role": "user", "contents": [{"type": "text", "text": f"message {index}"}]} for index in range(batch_size + 8)
    ]

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        real_mount = cp.mount
        batch_sizes: list[int] = []
        progress: list[tuple[int, int]] = []

        def mount_spy(*widgets: Widget, **kwargs: object):
            batch_sizes.append(len(widgets))
            return real_mount(*widgets, **kwargs)

        monkeypatch.setattr(cp, "mount", mount_spy)
        cp.set_replay_progress_callback(lambda current, total: progress.append((current, total)))
        await cp.replay_history(messages)

        total = len(messages)
        assert batch_sizes == [batch_size, 8]
        assert progress == [(0, total), (batch_size, total), (total, total)]


@pytest.mark.asyncio
async def test_collapsed_replay_tool_group_rebuilds_file_diff_from_snapshot_ref(tmp_path) -> None:
    """Large replay snapshots stay disk-backed while collapsed and resolve on expansion."""
    from chrys.app.tui.widgets.diff_view import DiffView
    from chrys.app.tui.widgets.diff_view.inline import InlineUnifiedDiffLines

    store = SnapshotStore(tmp_path)
    before_hash = store.save_data_as_blob(b"old\n").content_hash
    after_hash = store.save_data_as_blob(b"new\n").content_hash
    ref = FileSnapshotRef(store.mutations_dir, before_hash, after_hash)
    messages = [
        {"role": "user", "contents": [{"type": "text", "text": "edit"}]},
        {
            "role": "assistant",
            "contents": [
                {
                    "type": "function_call",
                    "name": "edit_file",
                    "call_id": "edit1",
                    "arguments": {"path": "src/app.py"},
                }
            ],
        },
        {"role": "tool", "contents": [{"type": "function_result", "call_id": "edit1", "result": "Edited"}]},
    ]

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.replay_history(messages, file_snapshots={"edit1": [ref]})
        await pilot.pause()

        group = cp.query_one(ToolGroup)
        assert group._content_mounted is False
        assert len(group._tools) == 0
        record = next(iter(group._tool_records.values()))
        assert record.file_snapshot is ref
        assert not list(cp.query(DiffView))

        group.collapsed = False
        for _ in range(200):
            await pilot.pause()
            if list(cp.query(InlineUnifiedDiffLines)):
                break

        assert not list(cp.query(DiffView))
        assert list(cp.query(InlineUnifiedDiffLines))


@pytest.mark.asyncio
async def test_chat_panel_arrange_defuses_short_canvas_anchor_pin() -> None:
    """When the arranged canvas is shorter than the container, ``arrange`` must
    release the bottom-anchor + reset ``scroll_y`` BEFORE the compositor reads
    the flag at ``_compositor.py:609``.

    Regression for a bug where, whenever the scrollbar wasn't at the top,
    collapsing tool groups left content docked at the viewport bottom with a
    blank band at the top.  Textual's compositor pin writes
    ``scroll_y = total_region.bottom - container_h`` via ``set_reactive``,
    bypassing the [0, max_scroll_y] clamp — a NEGATIVE value when
    virtual_h < container_h, which renders content offset down within the
    viewport.  ``watch_virtual_size`` runs too late to defuse it; the hook
    has to live in ``arrange`` itself.
    """
    from textual.geometry import Size

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        spacer = cp.query_one(_ChatBottomSpacer)

        # Dismiss welcome so the normal spacer lifecycle is in effect.
        await cp.add_user_message("hello")
        await pilot.pause()

        container_h = cp.size.height
        assert container_h > 0

        # Put the panel in the "spacer hidden, canvas short" state we hit
        # after a tool group auto-collapses back below the viewport.
        spacer.display = False
        await pilot.pause()

        # Simulate exactly what _compositor.py:609-619 does: write a
        # negative scroll_y via ``set_reactive`` (which bypasses the
        # validator's [0, max_scroll_y] clamp) while the anchor is still
        # engaged.  A plain ``cp.scroll_y = -5`` would get clamped to 0
        # on write and wouldn't reproduce the bug.
        from textual.widget import Widget

        cp._anchor_released = False
        cp.set_reactive(Widget.scroll_y, -5.0)
        assert cp.scroll_y == -5.0, "set_reactive must bypass validator (sanity check)"

        # Arrange against a container taller than the (now-tiny) content.
        cp.arrange(Size(cp.size.width, container_h))

        assert cp._anchor_released is True, (
            "anchor must be released inside arrange() so _compositor.py:609 skips the pin block on this same pass"
        )
        assert cp.scroll_y == 0, "stale pinned scroll_y must be reset before compositor paints"


@pytest.mark.asyncio
async def test_chat_panel_arrange_scroll_reset_does_not_re_engage_anchor() -> None:
    """The scroll_y reset in ``arrange`` must NOT go through the reactive setter.

    ``watch_scroll_y`` calls ``_check_anchor`` which re-engages the anchor
    whenever ``scroll_y >= max_scroll_y``.  When the canvas fits the
    viewport, both are 0, so the check would trivially re-engage the anchor
    right after we just released it — bringing the negative compositor pin
    straight back on the next layout.  ``arrange`` must use
    ``set_reactive`` to bypass the watcher.
    """
    from textual.geometry import Size
    from textual.widget import Widget

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        spacer = cp.query_one(_ChatBottomSpacer)

        await cp.add_user_message("hello")
        await pilot.pause()

        spacer.display = False
        await pilot.pause()

        container_h = cp.size.height
        cp._anchor_released = False
        cp.set_reactive(Widget.scroll_y, -3.0)

        cp.arrange(Size(cp.size.width, container_h))

        assert cp._anchor_released is True, (
            "anchor must STAY released after scroll_y reset — a reactive setter "
            "would trigger _check_anchor and re-engage with scroll_y >= max_scroll_y (both 0)"
        )


@pytest.mark.asyncio
async def test_chat_panel_manual_scroll_schedules_no_repaint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Manual scrolling must not refresh the panel or arm any repaint timer.

    The old debounced cleanup repaint fired mid-gesture on slow scrollbar
    drags.  The scroll fast path repaints every changed cell on its own
    (pinned by ``test_chat_scroll_fastpath.py``), so the only timer a manual
    scroll tick may arm is the GC resume debounce.
    """
    scroll_controller_module, _FakeGC = _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        for i in range(10):
            await cp.add_user_message(f"message {i}")
        await pilot.pause()

        assert cp.max_scroll_y > 1, "test setup must produce scrollable content"
        cp.scroll_y = 0
        await pilot.pause()

        calls = 0
        original_refresh = cp.refresh

        def refresh_spy(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            return original_refresh(*args, **kwargs)

        monkeypatch.setattr(cp, "refresh", refresh_spy)

        timers: list[tuple[float, object]] = []

        class FakeTimer:
            def stop(self) -> None:
                pass

        def set_timer_spy(delay: float, callback: object) -> FakeTimer:
            timers.append((delay, callback))
            return FakeTimer()

        monkeypatch.setattr(cp, "set_timer", set_timer_spy)

        cp._anchor_released = True
        cp.scroll_y = 1

        assert calls == 0, "released-anchor scroll must not repaint the panel"
        gc_resume = scroll_controller_module._MANUAL_SCROLL_GC_RESUME_SECONDS
        non_gc_timers = [timer for timer in timers if timer[0] != gc_resume]
        assert non_gc_timers == [], "manual scroll must only arm the GC resume debounce"

        cp._anchor_released = False
        cp.scroll_y = 2

        assert calls == 0, "anchored auto-follow must keep Textual's scroll fast path"
        non_gc_timers = [timer for timer in timers if timer[0] != gc_resume]
        assert non_gc_timers == []


def _simulate_chat_panel_user_scroll_y(panel: ChatPanel, y: float) -> None:
    """Move ChatPanel scroll position through the same watcher path as a user scroll."""
    from textual.widget import Widget

    old_y = panel.scroll_y
    panel._anchor_released = True
    panel.set_reactive(Widget.scroll_y, float(y))
    panel.set_reactive(Widget.scroll_target_y, float(y))
    panel.watch_scroll_y(old_y, float(y))


@pytest.mark.asyncio
async def test_chat_panel_user_pause_blocks_growth_reanchor() -> None:
    """New tool/intermediate growth must not override an explicit user scroll-up."""
    from textual.geometry import Size
    from textual.widget import Widget

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        cp.agent_running = True
        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = True
        cp.set_reactive(Widget.scroll_y, 5.0)

        cp.watch_virtual_size(Size(cp.size.width, 20), Size(cp.size.width, 80))

        assert cp._anchor_released is True
        assert cp._auto_scroll_paused_by_user is True


@pytest.mark.asyncio
async def test_chat_panel_soft_release_still_reanchors_on_growth() -> None:
    """The user-message soft-release flow should still reanchor on real growth."""
    from textual.geometry import Size
    from textual.widget import Widget

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        cp.agent_running = True
        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = False
        cp.set_reactive(Widget.scroll_y, 0.0)

        cp.watch_virtual_size(Size(cp.size.width, 20), Size(cp.size.width, 80))

        assert cp._anchor_released is False


@pytest.mark.asyncio
async def test_chat_panel_check_anchor_reengages_at_exact_bottom_only() -> None:
    from textual.widget import Widget

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        for i in range(12):
            await cp.add_user_message(f"message {i}")
        await pilot.pause()
        assert cp.max_scroll_y > 1

        near_bottom = cp.max_scroll_y - 1

        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = False
        cp.set_reactive(Widget.scroll_y, near_bottom)
        cp._check_anchor()
        assert cp._anchor_released is True

        cp._auto_scroll_paused_by_user = True
        cp._check_anchor()
        assert cp._anchor_released is True
        assert cp._auto_scroll_paused_by_user is True

        cp.set_reactive(Widget.scroll_y, cp.max_scroll_y)
        cp._check_anchor()
        assert cp._anchor_released is False
        assert cp._auto_scroll_paused_by_user is False


@pytest.mark.asyncio
async def test_chat_panel_second_upward_scroll_stays_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        body = "\n".join(f"line {i}" for i in range(8))
        for i in range(8):
            await cp.add_user_message(f"message {i}")
            await cp.add_agent_message(body)
        await pilot.pause()

        cp.scroll_end(immediate=True, animate=False)
        await pilot.pause()
        bottom_y = cp.scroll_y
        assert bottom_y == cp.max_scroll_y > 6

        cp.agent_running = True
        _simulate_chat_panel_user_scroll_y(cp, bottom_y - 3)
        after_first_scroll = cp.scroll_y

        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True
        assert bottom_y - after_first_scroll == 3

        _simulate_chat_panel_user_scroll_y(cp, after_first_scroll - 3)

        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True
        assert cp.scroll_y < after_first_scroll
        assert cp.scroll_y < cp.max_scroll_y


@pytest.mark.asyncio
async def test_chat_panel_downward_scroll_to_bottom_resumes_autoscroll(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        body = "\n".join(f"line {i}" for i in range(8))
        for i in range(8):
            await cp.add_user_message(f"message {i}")
            await cp.add_agent_message(body)
        await pilot.pause()

        cp.scroll_end(immediate=True, animate=False)
        await pilot.pause()
        assert cp.max_scroll_y > 6

        cp.agent_running = True
        _simulate_chat_panel_user_scroll_y(cp, cp.max_scroll_y - 6)

        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True
        assert cp.max_scroll_y - cp.scroll_y == 6

        cp.scroll_end(immediate=True, animate=False)
        await pilot.pause()

        assert cp._auto_scroll_paused_by_user is False
        assert cp._anchor_released is False
        assert cp.scroll_y == cp.max_scroll_y


@pytest.mark.asyncio
async def test_chat_panel_tool_growth_follows_after_exact_bottom_resume(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        body = "\n".join(f"line {i}" for i in range(8))
        for i in range(8):
            await cp.add_user_message(f"message {i}")
            await cp.add_agent_message(body)
        await pilot.pause()

        cp.scroll_end(immediate=True, animate=False)
        await pilot.pause()
        assert cp.max_scroll_y > 6

        cp.agent_running = True
        _simulate_chat_panel_user_scroll_y(cp, cp.max_scroll_y - 6)
        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True

        cp.scroll_end(immediate=True, animate=False)
        await pilot.pause()
        assert cp._anchor_released is False
        assert cp._auto_scroll_paused_by_user is False

        await cp.add_tool_start("call-1", "zsh", "shell", '{"cmd":"echo hi"}')
        await pilot.pause()

        assert cp.scroll_y == cp.max_scroll_y


@pytest.mark.asyncio
async def test_chat_panel_tool_growth_stays_detached_after_user_scroll_up(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detached state should block tool anchor-sync and growth reanchor."""
    from textual.geometry import Size

    _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        body = "\n".join(f"line {i}" for i in range(8))
        for i in range(8):
            await cp.add_user_message(f"message {i}")
            await cp.add_agent_message(body)
        await pilot.pause()

        assert cp.max_scroll_y > 6
        cp._set_scroll_y_programmatically(cp.max_scroll_y)
        cp.agent_running = True

        _simulate_chat_panel_user_scroll_y(cp, cp.max_scroll_y - 6)
        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True

        # The setup mounts arm a coalesced anchor-sync via call_after_refresh;
        # whether it has drained by now is refresh-timing dependent (flaky on
        # Windows CI). Reset to a known baseline so this isolates the detached
        # guard: while the anchor is released, _schedule_anchor_sync must not arm.
        cp._anchor_sync_scheduled = False
        cp._schedule_anchor_sync()
        assert cp._anchor_sync_scheduled is False

        cp.watch_virtual_size(Size(cp.size.width, 20), Size(cp.size.width, 80))

        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True


@pytest.mark.asyncio
async def test_chat_panel_anchor_sync_is_coalesced(monkeypatch: pytest.MonkeyPatch) -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        calls: list[object] = []

        def call_after_refresh_spy(callback: object, *args: object, **kwargs: object) -> bool:
            del args, kwargs
            calls.append(callback)
            return True

        monkeypatch.setattr(cp, "call_after_refresh", call_after_refresh_spy)

        cp._anchor_released = False
        cp._schedule_anchor_sync()
        cp._schedule_anchor_sync()
        cp.update_tool_progress("missing", ["ignored"])
        cp.update_tool_args("missing", {"ignored": True})

        assert len(calls) == 1
        assert cp._anchor_sync_scheduled is True

        callback = calls[0]
        assert callable(callback)
        callback()

        assert cp._anchor_sync_scheduled is False
        cp._schedule_anchor_sync()
        assert len(calls) == 2

        cp._anchor_sync_scheduled = False
        cp._auto_scroll_paused_by_user = True
        cp._schedule_anchor_sync()
        assert len(calls) == 2
        cp._auto_scroll_paused_by_user = False

        def call_after_refresh_closed(_callback: object, *args: object, **kwargs: object) -> bool:
            del args, kwargs
            return False

        monkeypatch.setattr(cp, "call_after_refresh", call_after_refresh_closed)
        cp._anchor_sync_scheduled = False
        cp._schedule_anchor_sync()
        assert cp._anchor_sync_scheduled is False


@pytest.mark.asyncio
async def test_chat_panel_bottom_clamp_does_not_mark_user_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    """A layout clamp to exact bottom is not a user scroll-up gesture."""
    from textual.widget import Widget

    _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        body = "\n".join(f"line {i}" for i in range(8))
        for i in range(8):
            await cp.add_user_message(f"message {i}")
            await cp.add_agent_message(body)
        await pilot.pause()

        assert cp.max_scroll_y > 0
        cp.agent_running = True
        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = False
        cp.set_reactive(Widget.scroll_y, float(cp.max_scroll_y))

        cp.watch_scroll_y(float(cp.max_scroll_y + 1), float(cp.max_scroll_y))

        assert cp._anchor_released is False
        assert cp._auto_scroll_paused_by_user is False


@pytest.mark.asyncio
async def test_chat_panel_intermediate_message_does_not_resume_user_pause() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        cp.agent_running = True
        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = True
        cp._final_response_started = False

        await cp.add_agent_message("checking files", is_intermediate=True)

        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True
        assert cp._final_response_started is False


@pytest.mark.asyncio
async def test_chat_panel_final_response_resumes_once() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        cp.agent_running = True
        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = True
        cp._final_response_started = False

        await cp.add_agent_message("final starts", is_final=False)

        # The mount's layout frame can transiently release the anchor with a
        # deferred re-anchor, so wait until the resume state lands.
        await wait_for(
            lambda: cp._auto_scroll_paused_by_user is False and cp._anchor_released is False,
            pilot=pilot,
            description="final-response autoscroll resume",
        )
        assert cp._final_response_started is True

        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = True

        await cp.add_agent_message("final complete", is_final=True)

        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True


@pytest.mark.asyncio
async def test_chat_panel_final_response_scrolls_detached_to_bottom(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        body = "\n".join(f"line {i}" for i in range(8))
        for i in range(8):
            await cp.add_user_message(f"message {i}")
            await cp.add_agent_message(body)
        await pilot.pause()

        cp.scroll_end(immediate=True, animate=False)
        await pilot.pause()
        assert cp.max_scroll_y > 6

        cp.agent_running = True
        _simulate_chat_panel_user_scroll_y(cp, cp.max_scroll_y - 6)
        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True

        await cp.add_agent_message("final starts", is_final=False)

        # Same deferred re-anchor settle as above: one pause is not enough on
        # slow runners, so wait until the yank to the bottom lands.
        await wait_for(
            lambda: (
                cp._auto_scroll_paused_by_user is False
                and cp._anchor_released is False
                and cp.scroll_y == cp.max_scroll_y
            ),
            pilot=pilot,
            description="final-response bottom yank",
        )


@pytest.mark.parametrize(
    ("status_kind", "running_when_status"),
    [
        ("error", False),
        ("interrupted", False),
        ("retry", True),
    ],
)
@pytest.mark.asyncio
async def test_chat_panel_status_messages_scroll_detached_to_bottom(
    monkeypatch: pytest.MonkeyPatch,
    status_kind: str,
    running_when_status: bool,
) -> None:
    _install_fake_chat_panel_gc(monkeypatch)

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        body = "\n".join(f"line {i}" for i in range(8))
        for i in range(8):
            await cp.add_user_message(f"message {i}")
            await cp.add_agent_message(body)
        await pilot.pause()

        cp.scroll_end(immediate=True, animate=False)
        await pilot.pause()
        assert cp.max_scroll_y > 6

        cp.agent_running = True
        _simulate_chat_panel_user_scroll_y(cp, cp.max_scroll_y - 6)
        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True

        cp.agent_running = running_when_status
        if status_kind == "error":
            await cp.add_error("boom")
        elif status_kind == "interrupted":
            await cp.add_interrupted()
        else:
            await cp.add_retry("retrying", 1, 2, 0)

        # The status-message yank re-anchors to the bottom, but on a shrink
        # frame ``arrange`` transiently releases the anchor and defers the
        # re-anchor to ``_reanchor_after_settle`` via ``call_after_refresh``.
        # On slower CI runners (Windows) that settle can span more than one
        # frame, so pump the loop until it lands instead of asserting after a
        # single pause.
        for _ in range(50):
            await pilot.pause()
            if (
                cp._anchor_released is False
                and cp._auto_scroll_paused_by_user is False
                and cp.scroll_y == cp.max_scroll_y
            ):
                break

        assert cp._auto_scroll_paused_by_user is False
        assert cp._anchor_released is False
        assert cp.scroll_y == cp.max_scroll_y


@pytest.mark.asyncio
async def test_chat_panel_agent_running_resets_final_response_autoscroll_gate() -> None:
    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        cp.agent_running = True
        await cp.add_agent_message("first run starts", is_final=False)

        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = True
        await cp.add_agent_message("first run final", is_final=True)
        assert cp._auto_scroll_paused_by_user is True
        assert cp._anchor_released is True

        cp.agent_running = False
        cp.agent_running = True
        await cp.add_agent_message("retry starts", is_final=False)

        # The mount's layout frame can transiently release the anchor and
        # defer the re-anchor via ``call_after_refresh`` (see the settle loop
        # in the sibling test above), so wait until the reset state lands.
        await wait_for(
            lambda: cp._auto_scroll_paused_by_user is False and cp._anchor_released is False,
            pilot=pilot,
            description="rerun autoscroll gate reset",
        )
        assert cp._final_response_started is True


@pytest.mark.asyncio
async def test_chat_panel_agent_running_post_run_disables_running_only_scroll_state() -> None:
    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        for index in range(8):
            await cp.add_user_message(f"message {index}")
            await cp.add_agent_message("\n".join(f"line {line}" for line in range(8)))
        await pilot.pause()
        cp.scroll_end(immediate=True, animate=False)
        await pilot.pause()

        cp.agent_running = True
        cp.agent_running = False
        cp._auto_scroll_paused_by_user = False
        cp._anchor_released = True

        _simulate_chat_panel_user_scroll_y(cp, max(0, cp.scroll_y - 5))

        assert cp._agent_running is False
        assert cp._auto_scroll_paused_by_user is False

        cp._anchor_released = True
        cp._auto_scroll_paused_by_user = False
        cp.watch_virtual_size(Size(80, 10), Size(80, 200))

        assert cp._anchor_released is True


@pytest.mark.asyncio
async def test_chat_panel_programmatic_upward_scroll_does_not_pause() -> None:
    from textual.widget import Widget

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        cp.agent_running = True
        cp.set_reactive(Widget.scroll_y, 10.0)

        cp._set_scroll_y_programmatically(5.0)

        assert cp._auto_scroll_paused_by_user is False


@pytest.mark.asyncio
async def test_chat_panel_scroll_to_turn_pauses_running_autoscroll() -> None:
    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("first")
        await cp.add_agent_message("one")
        await cp.add_user_message("second")
        await pilot.pause()

        cp.agent_running = True
        cp.scroll_to_turn("turn-1")

        assert cp._auto_scroll_paused_by_user is True


@pytest.mark.asyncio
async def test_chat_panel_toc_navigation_running_state_post_run_does_not_pause_autoscroll() -> None:
    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.add_user_message("first")
        await cp.add_agent_message("one")
        await cp.add_user_message("second")
        await pilot.pause()

        cp.agent_running = True
        cp.agent_running = False
        cp._auto_scroll_paused_by_user = False

        cp.scroll_to_turn("turn-1")

        assert cp._agent_running is False
        assert cp._auto_scroll_paused_by_user is False


@pytest.mark.asyncio
async def test_chat_panel_ignores_focus_center_scroll() -> None:
    """Focus restoration must not yank chat back to an old markdown child.

    Textual schedules ``scroll_to_center`` for focused widgets.  After a
    modal dismisses, a stale focused markdown descendant can otherwise center
    the first assistant body and leave the first user message at the top.
    """
    from chrys.app.tui.widgets.markdown import VirtualizedMarkdown

    async with ChatPanelApp().run_test(size=(80, 20)) as pilot:
        cp = pilot.app.query_one(ChatPanel)
        body = "\n".join(f"line {i}" for i in range(8))
        for i in range(8):
            await cp.add_user_message(f"message {i}")
            await cp.add_agent_message(body)
        await pilot.pause()

        cp.scroll_end(immediate=True, animate=False)
        await pilot.pause()
        bottom_y = cp.scroll_y
        assert bottom_y == cp.max_scroll_y > 0, "test setup must produce a bottom-scrolled panel"

        first_markdown = cp.query(VirtualizedMarkdown).first()
        cp.scroll_to_widget(first_markdown, center=True, animate=False, immediate=True)
        await pilot.pause()

        # Tolerate a 1-row drift: on Windows Textual's headless harness can
        # recompute virtual height on the layout pass triggered by
        # ``scroll_to_widget`` and clamp scroll_y down by one even though
        # ``scroll_to_region`` returned Offset() early.  A real yank from
        # focus-centring would move the transcript by many rows.
        assert abs(cp.scroll_y - bottom_y) <= 1, "focus-centering a chat descendant must not move the transcript"
        assert cp._anchor_released is False, "ignored focus scroll must not release bottom-follow"

        cp.scroll_to_widget(first_markdown, top=True, animate=False, immediate=True)
        await pilot.pause()
        assert cp.scroll_y < bottom_y, "explicit top navigation must still work"


@pytest.mark.asyncio
async def test_chat_panel_textual_anchor_contract() -> None:
    """Regression test: guard the private Textual APIs ChatPanel reaches into.

    ``panel.py`` touches four pieces of Textual's internal anchor/scroll state:

    * ``widget._anchored`` — bool flag set by ``widget.anchor()``
    * ``widget._anchor_released`` — bool toggled on user scroll-away / restored on scroll-to-bottom
    * ``widget._container_size`` — ``Size`` mutated inside our ``arrange()`` override so the compositor's anchor pin reads the fresh value
    * ``widget.anchor()`` / ``widget.release_anchor()`` / ``scroll_visible(top=...)``

    All four are either private (``_``-prefixed) or depend on private state, so a
    Textual upgrade could rename / remove them silently.  This test asserts the
    contract ``ChatPanel`` depends on so such a regression surfaces immediately
    rather than producing invisible scroll/flash glitches at runtime.

    If this test fails after a Textual bump, re-audit the corresponding code
    in ``src/chrys/app/tui/widgets/chat/panel.py`` (``arrange`` override +
    ``watch_virtual_size``) — the scroll behaviour likely needs to be
    ported to whatever replaced the private API.
    """
    import inspect

    from textual.geometry import Size

    async with ChatPanelApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)

        # --- Attribute surface ------------------------------------------------
        # Must exist as bool flags we can read/write.
        assert isinstance(cp._anchored, bool), "Textual removed/renamed Widget._anchored"
        assert isinstance(cp._anchor_released, bool), "Textual removed/renamed Widget._anchor_released"
        # _container_size must be a Size (we mutate it with a Size literal).
        assert isinstance(cp._container_size, Size), "Textual changed _container_size type"
        assert hasattr(cp._container_size, "width") and hasattr(cp._container_size, "height")

        # --- Method surface ---------------------------------------------------
        # anchor() / release_anchor() are the public entry points we call.
        assert callable(getattr(cp, "anchor", None)), "Textual removed Widget.anchor()"
        assert callable(getattr(cp, "release_anchor", None)), "Textual removed Widget.release_anchor()"
        # arrange(size, optimal=False) is the method we override.
        arrange_params = list(inspect.signature(cp.arrange).parameters.keys())
        assert arrange_params[:2] == ["size", "optimal"], (
            f"Textual changed Widget.arrange signature (got {arrange_params!r}); our override must match"
        )

        # --- Behavioural contract --------------------------------------------
        # on_mount() calls self.anchor() → _anchored must become True.
        # (ChatPanel.on_mount ran during compose; re-assert for clarity.)
        cp.anchor()
        assert cp._anchored is True, "anchor() must set _anchored=True"

        # release_anchor() flips _anchor_released True — this is what
        # scroll_visible(top=True) triggers, and what watch_virtual_size
        # flips back to False to re-engage auto-follow.
        cp._anchor_released = False
        cp.release_anchor()
        assert cp._anchor_released is True, "release_anchor() must set _anchor_released=True"

        # We manually clear _anchor_released to re-engage — verify that's
        # still a plain attribute and not e.g. a read-only property.
        cp._anchor_released = False
        assert cp._anchor_released is False, (
            "Textual made _anchor_released read-only; watch_virtual_size's manual re-engage will break"
        )

        # _container_size must be writable — we reassign it inside arrange().
        original = cp._container_size
        cp._container_size = Size(original.width, original.height + 1)
        assert cp._container_size.height == original.height + 1, (
            "Textual made _container_size read-only; our arrange() override will break"
        )
        cp._container_size = original  # restore


# ---------------------------------------------------------------------------
# InputBar (enhanced)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_input_bar_submit() -> None:
    messages: list[str] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield InputBar()

        def on_input_bar_user_submitted(self, event: InputBar.UserSubmitted) -> None:
            messages.append(event.text)

    async with TestApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        ib.value = "hello world"
        await ib.action_submit()
        await pilot.pause()
        assert "hello world" in messages
        assert ib.value == ""


@pytest.mark.asyncio
async def test_input_bar_empty_submit() -> None:
    """Empty input should not fire Submitted."""
    messages: list[str] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield InputBar()

        def on_input_bar_user_submitted(self, event: InputBar.UserSubmitted) -> None:
            messages.append(event.text)

    async with TestApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        ib.value = "   "
        await ib.action_submit()
        await pilot.pause()
        assert messages == []


@pytest.mark.asyncio
async def test_input_bar_send_button_disabled_for_empty_input() -> None:
    """Idle Send should only be clickable when there is non-empty text."""
    messages: list[str] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield InputBar()

        def on_input_bar_user_submitted(self, event: InputBar.UserSubmitted) -> None:
            messages.append(event.text)

    async with TestApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        send_btn = ib.query_one("#send-btn", Button)

        assert send_btn.disabled is True
        send_btn.press()
        await pilot.pause()
        assert messages == []

        ib.value = "hello world"
        await pilot.pause()
        assert send_btn.disabled is False

        send_btn.press()
        await pilot.pause()
        assert messages == ["hello world"]
        assert ib.value == ""
        assert send_btn.disabled is True


@pytest.mark.asyncio
async def test_input_bar_loading_disables_submit_but_keeps_input_editable() -> None:
    """Agent loading disables Send without making the draft read-only."""
    messages: list[str] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield InputBar()

        def on_input_bar_user_submitted(self, event: InputBar.UserSubmitted) -> None:
            messages.append(event.text)

    async with TestApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        send_btn = ib.query_one("#send-btn", Button)
        text_area = ib.query_one("#chat-input", TextArea)

        ib.value = "queued text"
        await pilot.pause()
        assert send_btn.disabled is False

        ib.agent_loading = True
        await pilot.pause()
        assert send_btn.disabled is True
        assert str(send_btn.label) == "Send"
        assert ib.query_one("#editor-btn", Button).render().plain == ">"
        assert text_area.read_only is False

        send_btn.press()
        await ib.action_submit()
        await pilot.pause()
        assert messages == []
        assert ib.value == "queued text"

        ib.agent_loading = False
        await pilot.pause()
        assert send_btn.disabled is False


@pytest.mark.asyncio
async def test_input_bar_locked_queue_disables_text_area_until_unlock() -> None:
    """Queued mid-run injection disables text-area interaction until it is consumed or abandoned."""

    async with InputBarApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        send_btn = ib.query_one("#send-btn", Button)
        text_area = ib.query_one("#chat-input", TextArea)

        ib.value = "queued injection"
        ib.lock_with_text()
        await pilot.pause()

        assert ib.locked is True
        assert text_area.read_only is True
        assert text_area.disabled is True
        assert send_btn.disabled is True

        ib.unlock_and_keep()
        await pilot.pause()

        assert ib.locked is False
        assert text_area.read_only is False
        assert text_area.disabled is False


@pytest.mark.asyncio
async def test_input_bar_consume_retry_text_returns_and_clears_note() -> None:
    async with InputBarApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)

        ib.value = "  retry note  "
        await pilot.pause()

        assert ib.consume_retry_text() == "retry note"
        assert ib.value == ""


@pytest.mark.asyncio
async def test_input_bar_up_down_uses_instance_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normal arrows should browse prompts submitted in the current TUI instance."""

    def fake_append_history(
        text: str,
        *,
        session_id: str | None = None,
        instance_id: str | None = None,
        cwd: str | None = None,
    ) -> None:
        del text, session_id, instance_id, cwd

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.append_history", fake_append_history)
    load_calls: list[str | None] = []

    def fake_load_history(*, max_entries: int, instance_id: str | None = None) -> list[str]:
        del max_entries
        load_calls.append(instance_id)
        return ["instance old"]

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.load_history", fake_load_history)

    async with InputBarApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        text_area = ib.query_one("#chat-input", _ChatTextArea)
        ib.add_to_history("instance old", session_id="session-1")
        ib.add_to_history("instance new", session_id="session-2")
        ib.value = "draft"
        text_area.focus()
        await pilot.pause()

        await pilot.press("up")
        await pilot.pause()
        assert ib.value == "instance new"
        assert load_calls == []

        await pilot.press("up")
        await pilot.pause()
        assert ib.value == "instance old"

        await pilot.press("down")
        await pilot.pause()
        assert ib.value == "instance new"

        await pilot.press("down")
        await pilot.pause()
        assert ib.value == "draft"


@pytest.mark.asyncio
async def test_input_bar_enter_submit_keeps_prior_instance_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keyboard Enter submits should reset browsing state without clearing instance entries."""

    monkeypatch.setattr(
        "chrys.app.tui.widgets.chrome.input_bar.append_history",
        lambda text, *, session_id=None, instance_id=None, cwd=None: None,
    )

    submitted: list[str] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield InputBar()

        def on_input_bar_user_submitted(self, event: InputBar.UserSubmitted) -> None:
            submitted.append(event.text)
            self.query_one(InputBar).add_to_history(event.text, session_id="session-1")

    async with TestApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        text_area = ib.query_one("#chat-input", _ChatTextArea)
        text_area.focus()

        ib.value = "first"
        await pilot.press("enter")
        await pilot.pause()
        ib.value = "second"
        await pilot.press("enter")
        await pilot.pause()

        assert submitted == ["first", "second"]

        await pilot.press("up")
        await pilot.pause()
        assert ib.value == "second"
        await pilot.press("up")
        await pilot.pause()
        assert ib.value == "first"


@pytest.mark.asyncio
async def test_instance_history_loads_persisted_records_for_this_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instance scope should read persisted prompt history filtered by the TUI instance id."""
    loop_thread = threading.current_thread()

    def fake_token_hex(nbytes: int) -> str:
        assert nbytes == 8
        return "instance-a"

    monkeypatch.setattr(
        "chrys.app.tui.widgets.chrome.input_bar.token_hex",
        fake_token_hex,
        raising=False,
    )

    calls: list[str | None] = []
    load_threads: list[threading.Thread] = []

    def fake_load_history(
        max_entries: int = 200,
        *,
        instance_id: str | None = None,
    ) -> list[str]:
        del max_entries
        calls.append(instance_id)
        load_threads.append(threading.current_thread())
        if instance_id == "instance-a":
            return ["mine old", "mine new"]
        return ["other"]

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.load_history", fake_load_history)

    async with InputBarApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        text_area = ib.query_one("#chat-input", _ChatTextArea)
        ib.value = "draft"
        text_area.focus()
        await pilot.pause()

        await pilot.press("up")
        await pilot.pause()

        assert ib.value == "mine new"
        assert calls == ["instance-a"]
        assert load_threads[0] is not loop_thread


@pytest.mark.asyncio
async def test_empty_instance_history_is_loaded_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty persisted result should not trigger another disk scan on every Up key."""
    load_threads: list[threading.Thread] = []

    def fake_load_history(
        max_entries: int = 200,
        *,
        instance_id: str | None = None,
    ) -> list[str]:
        del max_entries, instance_id
        load_threads.append(threading.current_thread())
        return []

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.load_history", fake_load_history)

    async with InputBarApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        text_area = ib.query_one("#chat-input", _ChatTextArea)
        text_area.focus()
        await pilot.pause()

        await pilot.press("up")
        await pilot.pause()
        await pilot.press("up")
        await pilot.pause()

        assert len(load_threads) == 1


@pytest.mark.asyncio
async def test_input_bar_plain_page_keys_do_not_browse_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plain PageUp/PageDown should keep the TextArea default cursor-page behavior."""

    def fake_load_history(
        max_entries: int = 200,
        *,
        instance_id: str | None = None,
    ) -> list[str]:
        del max_entries, instance_id
        return ["old", "new"]

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.load_history", fake_load_history)

    async with InputBarApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        text_area = ib.query_one("#chat-input", _ChatTextArea)
        ib.value = "draft"
        text_area.focus()
        await pilot.pause()

        await pilot.press("pageup")
        await pilot.pause()
        assert ib.value == "draft"

        await pilot.press("pagedown")
        await pilot.pause()
        assert ib.value == "draft"


@pytest.mark.asyncio
async def test_instance_history_ignores_records_from_other_instances(monkeypatch: pytest.MonkeyPatch) -> None:
    """Up should ask storage for the current instance id, not the unfiltered global history."""

    def fake_token_hex(nbytes: int) -> str:
        assert nbytes == 8
        return "instance-a"

    monkeypatch.setattr(
        "chrys.app.tui.widgets.chrome.input_bar.token_hex",
        fake_token_hex,
        raising=False,
    )

    def fake_load_history(
        max_entries: int = 200,
        *,
        instance_id: str | None = None,
    ) -> list[str]:
        del max_entries
        if instance_id is not None:
            return ["mine"]
        return ["other"]

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.load_history", fake_load_history)

    async with InputBarApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        text_area = ib.query_one("#chat-input", _ChatTextArea)
        ib.value = "draft"
        text_area.focus()
        await pilot.pause()

        await pilot.press("up")
        await pilot.pause()
        assert ib.value == "mine"


@pytest.mark.asyncio
async def test_instance_history_keeps_latest_1000_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "chrys.app.tui.widgets.chrome.input_bar.append_history",
        lambda text, *, session_id=None, instance_id=None, cwd=None: None,
    )

    async with InputBarApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        text_area = ib.query_one("#chat-input", _ChatTextArea)

        for i in range(1001):
            ib.add_to_history(f"i-{i}")

        text_area.focus()
        await pilot.pause()

        assert text_area._history.entries[0] == "i-1"
        assert len(text_area._history.entries) == 1000

        await pilot.press("up")
        await pilot.pause()
        assert ib.value == "i-1000"


@pytest.mark.asyncio
async def test_down_key_restores_draft_after_instance_browse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Down should walk forward through instance history and then restore the draft."""

    monkeypatch.setattr(
        "chrys.app.tui.widgets.chrome.input_bar.append_history",
        lambda text, *, session_id=None, instance_id=None, cwd=None: None,
    )

    async with InputBarApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        text_area = ib.query_one("#chat-input", _ChatTextArea)
        ib.add_to_history("i-old", session_id="s1")
        ib.add_to_history("i-new", session_id="s1")
        ib.value = "draft"
        text_area.focus()
        await pilot.pause()

        # Browse instance to i-old
        await pilot.press("up")
        await pilot.pause()
        assert ib.value == "i-new"
        await pilot.press("up")
        await pilot.pause()
        assert ib.value == "i-old"

        await pilot.press("down")
        await pilot.pause()
        assert ib.value == "i-new"

        await pilot.press("down")
        await pilot.pause()
        assert ib.value == "draft"


@pytest.mark.asyncio
async def test_send_button_submit_resets_history_browse_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Submitting with the button should not let a later Down restore the pre-browse draft."""

    monkeypatch.setattr(
        "chrys.app.tui.widgets.chrome.input_bar.append_history",
        lambda text, *, session_id=None, instance_id=None, cwd=None: None,
    )
    submitted: list[str] = []

    class TestApp(App):
        def compose(self) -> ComposeResult:
            yield InputBar()

        def on_input_bar_user_submitted(self, event: InputBar.UserSubmitted) -> None:
            submitted.append(event.text)

    async with TestApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        text_area = ib.query_one("#chat-input", _ChatTextArea)
        ib.add_to_history("old")
        ib.value = "draft"
        text_area.focus()
        await pilot.pause()

        await pilot.press("up")
        await pilot.pause()
        assert ib.value == "old"

        ib.query_one("#send-btn", Button).press()
        await pilot.pause()
        assert submitted == ["old"]
        assert ib.value == ""

        await pilot.press("down")
        await pilot.pause()
        assert ib.value == ""


@pytest.mark.asyncio
async def test_input_bar_add_to_history_persists_session_and_instance_id(monkeypatch: pytest.MonkeyPatch) -> None:
    loop_thread = threading.current_thread()
    calls: list[dict[str, str | threading.Thread | None]] = []

    def fake_append_history(
        text: str,
        *,
        session_id: str | None = None,
        instance_id: str | None = None,
        cwd: str | None = None,
    ) -> None:
        calls.append(
            {
                "text": text,
                "session_id": session_id,
                "instance_id": instance_id,
                "cwd": cwd,
                "thread": threading.current_thread(),
            }
        )

    monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.append_history", fake_append_history)
    monkeypatch.setattr("chrys.app.tui.widgets.chrome.input_bar.safe_getcwd", lambda: "/workspace")

    async with InputBarApp().run_test() as pilot:
        ib = pilot.app.query_one(InputBar)
        text_area = ib.query_one("#chat-input", _ChatTextArea)
        ib.set_paste_cwd("/session-workspace")

        ib.add_to_history("hello", session_id="session-1")
        await text_area._history_writer.close(timeout_seconds=None)

    assert len(calls) == 1
    assert calls[0]["text"] == "hello"
    assert calls[0]["session_id"] == "session-1"
    assert isinstance(calls[0]["instance_id"], str)
    assert len(calls[0]["instance_id"]) == 16
    bytes.fromhex(calls[0]["instance_id"])
    assert calls[0]["cwd"] == "/session-workspace"
    assert calls[0]["thread"] is not loop_thread


# ---------------------------------------------------------------------------
# DebugPanel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("locale", "title"), [("en", "Event Stream"), ("zh-Hans", "事件流")])
@pytest.mark.asyncio
async def test_debug_panel_title_uses_mount_locale(locale: str, title: str) -> None:
    async with DebugPanelApp(locale).run_test() as pilot:
        panel = pilot.app.query_one(DebugPanel)
        assert panel.query_one("DebugPanel > Static", Static).render().plain == title


@pytest.mark.asyncio
async def test_debug_panel_log_event() -> None:
    async with DebugPanelApp().run_test() as pilot:
        dp = pilot.app.query_one(DebugPanel)
        dp.log_event("ToolCallStart", "read_file")
        dp.log_event("Usage[Explore]", "4,291")
        dp.log_event("Error", "something broke")
        dp.log_raw("raw debug text")
        log = pilot.app.query_one("#debug-log", _ClickableLog)
        assert any("Usage[Explore]" in line for line in log._plain_lines)
        # No crash = passes


@pytest.mark.asyncio
async def test_debug_log_click_copies_to_terminal_and_os_clipboards(monkeypatch: pytest.MonkeyPatch) -> None:
    async with DebugPanelApp().run_test() as pilot:
        log = pilot.app.query_one("#debug-log", _ClickableLog)
        copied: list[str] = []
        monkeypatch.setattr("chrys.app.tui.clipboard.clipboard_copy", copied.append)

        log.write_with_text(Text("visible event"), "plain event")
        log.on_click(
            Click(
                log,
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=1,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=log.region.x,
                screen_y=log.region.y,
            )
        )

        assert pilot.app.clipboard == "plain event"
        assert copied == ["plain event"]
