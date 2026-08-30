# Copyright (c) 2026 Chrys. All rights reserved.

"""Renderer for sub-agent tool calls.

Shows the sub-agent name, the task prompt, and a live feed of inner tool calls.
On completion the sub-agent's full final response is rendered as markdown
inside the tool/result panel border.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Button, Static

from chrys.app.tui.i18n import render_str, render_text, widget_localizer
from chrys.app.tui.support.gc_freeze import GcAbsorbReason, GcAbsorbRequested
from chrys.app.tui.util.formatting import format_token_count
from chrys.app.tui.widgets.chat.renderers.sleep import SleepSkipClicked
from chrys.app.tui.widgets.chat.tool_call import (
    TOOL_CARD_INTERRUPTED,
    TOOL_CARD_REJECTED,
    TOOL_COPY_INCLUDED_CLASS,
    BaseToolCard,
    ToolCardHeader,
    fmt_duration,
    tool_result_render_status,
)
from chrys.app.tui.widgets.chat.tool_view_builders import (
    TOOL_VIEW_EMPTY,
    TOOL_VIEW_OUTPUT,
    build_code_view,
    build_params_view,
)
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.foundation.i18n import MessageDef, MessageRef, msg
from chrys.foundation.platform.files import surrogate_safe_text
from chrys.foundation.tool_kinds import KIND_SLEEP, KIND_SUB_AGENT
from chrys.foundation.tool_result_metadata import TOOL_INTERRUPTED_METADATA_KEY

if TYPE_CHECKING:
    from textual.app import ComposeResult

_SUB_AGENT_TASK = msg("tui.tool_card.sub_agent.task", fallback="Task")
_SUB_AGENT_TASK_PROMPT = msg("tui.tool_card.sub_agent.task_prompt", fallback="Task Prompt")
_SUB_AGENT_SPEND = msg("tui.tool_card.sub_agent.spend", fallback="Spend: {spend}")
_SUB_AGENT_COMPACTING = msg("tui.tool_card.sub_agent.compacting", fallback="Compacting conversation...")
_SUB_AGENT_COMPACTION_NAME = msg("tui.tool_card.sub_agent.compaction_name", fallback="Compaction")
_SUB_AGENT_TOOL_CALLS = msg("tui.tool_card.sub_agent.tool_calls", fallback="Tool calls: {n}")
_SUB_AGENT_CTX_TOKENS = msg("tui.tool_card.sub_agent.ctx_tokens", fallback="Ctx: {tokens} tokens")
_SUB_AGENT_TOKENS = msg("tui.tool_card.sub_agent.tokens", fallback="{tokens} tokens")
_SUB_AGENT_ZERO_REPORTED_TOKENS = msg(
    "tui.tool_card.sub_agent.zero_reported_tokens",
    fallback="0 reported tokens",
)
_SUB_AGENT_UNREPORTED_ATTEMPTS = msg(
    "tui.tool_card.sub_agent.unreported_attempts",
    fallback="{count} unreported attempt",
    plural_fallback="{count} unreported attempts",
)
_SUB_AGENT_COMPACTIONS = msg("tui.tool_card.sub_agent.compactions", fallback="Compactions: {n}")
_SUB_AGENT_DURATION = msg("tui.tool_card.sub_agent.duration", fallback="Duration: {duration}")
_SUB_AGENT_IMAGES = msg(
    "tui.tool_card.sub_agent.images",
    fallback="{count} image",
    plural_fallback="{count} images",
)
_SUB_AGENT_ARTIFACTS = msg(
    "tui.tool_card.sub_agent.artifacts",
    fallback="{count} artifact",
    plural_fallback="{count} artifacts",
)
_SUB_AGENT_TOOL_SKIPPED = msg("tui.tool_card.sub_agent.tool_skipped", fallback="{tool} skipped")
_SUB_AGENT_TOOL_INTERRUPTED = msg(
    "tui.tool_card.sub_agent.tool_interrupted",
    fallback="{tool} interrupted",
)
_SUB_AGENT_RENDERING_MARKDOWN = msg(
    "tui.tool_card.sub_agent.rendering_markdown",
    fallback="Rendering markdown result...",
)
_SUB_AGENT_COMPACTED_WARNING = msg(
    "tui.tool_card.sub_agent.compacted_warning",
    fallback="Conversation compacted; summary format warning: {violation}.",
)
_SUB_AGENT_COMPACTED = msg(
    "tui.tool_card.sub_agent.compacted",
    fallback="Conversation compacted.",
)
_SUB_AGENT_COMPACTION_FAILED_REASON = msg(
    "tui.tool_card.sub_agent.compaction_failed_reason",
    fallback="Compaction failed ({reason})",
)
_SUB_AGENT_COMPACTION_FAILED = msg(
    "tui.tool_card.sub_agent.compaction_failed",
    fallback="Compaction failed",
)
_SUB_AGENT_RETRYING = msg(
    "tui.tool_card.sub_agent.retrying",
    fallback="↻ Retrying in {delay_seconds}s ({attempt}/{max_attempts}): {message}",
)
_SUB_AGENT_REASON_STREAM_STALLED = msg(
    "tui.tool_card.sub_agent.reason.stream_stalled",
    fallback="Stream stalled",
)
_SUB_AGENT_REASON_COMPACTION_FAILED = msg(
    "tui.tool_card.sub_agent.reason.compaction_failed",
    fallback="Compaction failed",
)
_SUB_AGENT_REASON_FAILED = msg("tui.tool_card.sub_agent.reason.failed", fallback="Sub-agent failed")
_SUB_AGENT_REASON_ACP_INTERRUPTED = msg(
    "tui.tool_card.sub_agent.reason.acp_interrupted",
    fallback="External ACP transport interrupted",
)
_SUB_AGENT_REASON_PAUSED = msg("tui.tool_card.sub_agent.reason.paused", fallback="Sub-agent paused")
_SUB_AGENT_AFTER_RETRIES = msg(
    "tui.tool_card.sub_agent.after_retries",
    fallback="(after {count} auto-retry attempt)",
    plural_fallback="(after {count} auto-retry attempts)",
)
_SUB_AGENT_DIAGNOSTICS = msg("tui.tool_card.sub_agent.diagnostics", fallback="Diagnostics: {path}")
_SUB_AGENT_PAUSED = msg(
    "tui.tool_card.sub_agent.paused",
    fallback="Paused — awaiting user",
)
_SUB_AGENT_SKIP_SLEEP = msg("tui.tool_card.sub_agent.button.skip_sleep", fallback="Skip sleep")
_SUB_AGENT_RETRY = msg("tui.tool_card.sub_agent.button.retry", fallback="Retry")
_SUB_AGENT_ABORT = msg("tui.tool_card.sub_agent.button.abort", fallback="Abort")
_SUB_AGENT_SKIP_SLEEP_TOOLTIP = msg(
    "tui.tool_card.sub_agent.button.skip_sleep_tooltip",
    fallback="Skip the active sleep in this sub-agent",
)

_SUB_AGENT_REASON_MESSAGES: dict[str, MessageDef] = {
    "stream_stall": _SUB_AGENT_REASON_STREAM_STALLED,
    "last_words": _SUB_AGENT_REASON_COMPACTION_FAILED,
    "framework_exc": _SUB_AGENT_REASON_FAILED,
    "acp_transport": _SUB_AGENT_REASON_ACP_INTERRUPTED,
}

_MAX_VISIBLE = 7
"""Maximum inner tool call lines shown at once."""

_ARGS_BRIEF_MAX = 60
"""Maximum characters for inner tool call arguments display."""

_ARG_VALUE_MAX = 60
"""Maximum characters for a single argument value in the compact display."""

_RESULT_MARKDOWN_DEFER_CHARS = 16 * 1024
"""Minimum result size that gets a collapse grace period before markdown rendering."""

_RESULT_MARKDOWN_MOUNT_DELAY_SECONDS = 1.0
"""Delay before mounting large sub-agent markdown so soon-collapsed groups can prune first."""


def _fmt_args(args: dict[str, Any]) -> str:
    """Format tool args as a compact key: value string instead of raw JSON.

    Individual values are truncated to ``_ARG_VALUE_MAX`` characters to
    prevent large fields (e.g. ``write_file`` content) from flooding the
    inner tool call display.
    """
    parts: list[str] = []
    for k, v in args.items():
        if isinstance(v, str):
            display = v if len(v) <= _ARG_VALUE_MAX else v[:_ARG_VALUE_MAX] + "…"
            parts.append(f'{k}: "{display}"')
        else:
            raw = str(v)
            display = raw if len(raw) <= _ARG_VALUE_MAX else raw[:_ARG_VALUE_MAX] + "…"
            parts.append(f"{k}: {display}")
    return " ".join(parts)


_COMPACTION_ENTRY_KIND = "compaction"
"""Synthetic ``tool_kind`` for the Phase-4 compaction progress line."""


@dataclass
class _InnerToolEntry:
    """State for a single inner tool call."""

    call_id: str
    tool_name: str
    tool_kind: str
    args_summary: str
    status: str = "running"
    duration_ms: int = 0
    image_count: int = 0
    artifact_count: int = 0
    progress_summary: str = ""
    started_at: float = 0.0
    """Monotonic start time; when set, the running line shows live elapsed."""
    completion_counted: bool = False


# Custom Textual messages — bubble up to MainScreen so it can publish the
# corresponding bus event.  Keeping a translation layer between widget
# clicks and event bus events means the widget doesn't need to hold a
# reference to the bus.


class SubAgentRetryClicked(Message):
    """User clicked Retry on a paused :class:`SubAgentToolCall` card."""

    def __init__(self, invocation_id: str) -> None:
        super().__init__()
        self.invocation_id = invocation_id


class SubAgentAbortClicked(Message):
    """User clicked Abort on a paused :class:`SubAgentToolCall` card."""

    def __init__(self, invocation_id: str) -> None:
        super().__init__()
        self.invocation_id = invocation_id


class SubAgentToolCall(BaseToolCard):
    """Renderer for sub-agent tool invocations.

    Implements the ToolCall protocol so it can be used in ToolGroup.
    A dedicated :class:`VirtualizedMarkdown` (``#sa-task``) renders the
    original prompt in a ``Task`` panel. While running, a ``Static``
    (``#sa-main``) hosts the animated header+inner-tools feed. On completion,
    ``#sa-main`` is hidden and a sibling :class:`VirtualizedMarkdown`
    (``#sa-result``) renders the sub-agent's full final response as markdown —
    no truncation.
    """

    DEFAULT_CSS = """
    SubAgentToolCall {
        padding: 0 0 0 2;
        height: auto;
        margin: 0 0 1 0;
    }
    SubAgentToolCall #sa-label {
        height: auto;
    }
    SubAgentToolCall > #sa-task-panel {
        height: auto;
        margin: 0 0 0 2;
        border: round $tui-border-neutral-128 $border-opacity;
        border-title-color: $text-muted;
        border-title-style: not bold;
        padding: 0 1 0 1;
    }
    SubAgentToolCall > #sa-panel {
        height: auto;
        margin: 0 0 0 2;
        border: round $tui-border-warning 50%;
        border-title-color: $warning;
        border-title-style: bold;
        border-title-align: left;
        border-subtitle-color: $warning;
        padding: 0 1 0 1;
    }
    SubAgentToolCall.-complete > #sa-panel {
        border: round $tui-border-success 30%;
        border-title-color: $success;
        border-title-style: not bold;
        border-subtitle-color: $success;
    }
    SubAgentToolCall.-error > #sa-panel {
        border: round $tui-border-error 50%;
        border-title-color: $error;
        border-title-style: not bold;
        border-subtitle-color: $error;
    }
    SubAgentToolCall.-rejected > #sa-panel {
        border: round $tui-border-warning 50%;
        border-title-color: $warning;
        border-title-style: not bold;
        border-subtitle-color: $warning;
    }
    SubAgentToolCall.-paused > #sa-panel {
        border: round $tui-border-warning $border-opacity;
        border-title-color: $warning;
        border-title-style: bold;
        border-subtitle-color: $warning;
    }
    SubAgentToolCall #sa-main {
        height: auto;
        min-height: 1;
    }
    SubAgentToolCall #sa-task {
        height: auto;
        min-height: 1;
        padding: 0;
        background: transparent;
    }
    SubAgentToolCall.-done #sa-main {
        display: none;
    }
    SubAgentToolCall #sa-result {
        height: auto;
        min-height: 1;
        padding: 0;
        background: transparent;
        display: none;
    }
    SubAgentToolCall.-done #sa-result {
        display: block;
    }
    SubAgentToolCall #sa-retry-banner {
        height: auto;
        display: none;
        color: $warning;
        text-style: italic;
    }
    SubAgentToolCall.-retrying #sa-retry-banner {
        display: block;
    }
    SubAgentToolCall #sa-pause-info {
        height: auto;
        display: none;
        color: $error;
    }
    SubAgentToolCall.-paused #sa-pause-info {
        display: block;
    }
    SubAgentToolCall #sa-actions {
        height: auto;
        display: none;
        margin: 1 0 0 0;
    }
    SubAgentToolCall #sa-sleep-actions {
        height: auto;
        display: none;
        margin: 1 0 0 0;
    }
    SubAgentToolCall.-sleeping #sa-sleep-actions {
        display: block;
    }
    SubAgentToolCall.-done #sa-sleep-actions {
        display: none;
    }
    SubAgentToolCall.-paused #sa-actions {
        display: block;
    }
    SubAgentToolCall #sa-actions Button {
        margin: 0 1 0 0;
        min-width: 10;
        height: 1;
        border: none;
    }
    SubAgentToolCall #sa-retry-btn {
        background: $warning;
        color: $text;
        text-style: bold;
    }
    SubAgentToolCall #sa-abort-btn {
        background: $error;
        color: $text;
        text-style: bold;
    }
    SubAgentToolCall #sa-skip-sleep-btn {
        min-width: 12;
        height: 1;
        border: none;
        background: $warning;
        color: $text;
        text-style: bold;
    }
    """

    _SPINNERS: ClassVar[str] = "\u25d0\u25d3\u25d1\u25d2"

    def __init__(
        self,
        call_id: str,
        tool_name: str,
        args_summary: str = "",
        args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(call_id, tool_name, args_summary, args=args)
        self._spin_idx = 0

        self._invocation_id: str | None = None
        self._inner_tools: OrderedDict[str, _InnerToolEntry] = OrderedDict()
        self._seen_inner_call_ids: set[str] = set()
        self._terminal_inner_call_ids: set[str] = set()
        self._total_inner_calls = 0
        self._completed_inner_calls = 0
        self._progress_tool_calls = 0
        self._progress_ctx_tokens = 0
        self._progress_total_usage_tokens = 0
        self._usage_unreported_attempts = 0
        # Committed Phase-4 compactions — persistent, unlike the compaction
        # entries in ``_inner_tools`` which the visibility cap can evict.
        # Driven by the committed signal, not finished(ok): a generated note
        # whose spill write fails is abandoned and must not count.
        self._compaction_count = 0

        self._start_time = time.monotonic()
        self._timer: Timer | None = None
        self._result_timer: Timer | None = None
        self._result_pending = False
        self._result_rendering = False
        # NOTE: name must avoid ``_task`` — that attribute is owned by
        # :class:`textual.message_pump.MessagePump`, which assigns the
        # widget's running ``asyncio.Task`` to ``self._task`` before
        # :meth:`compose` runs.  Storing the prompt under that name makes
        # it look like a string in ``__init__`` and an ``asyncio.Task`` by
        # the time ``compose()`` reads it, which crashes ``Text(...)``.
        self._task_prompt = self._extract_prompt(self.args_summary, self.args)

    def _render_message(self, reference: MessageRef) -> str:
        return render_str(widget_localizer(self), reference)

    def _label_text(self, duration_ms: int = 0) -> Text:
        t = Text()
        t.append("• ", style="bold")
        t.append("SubAgent", style="bold")
        if duration_ms:
            t.append(f" ({fmt_duration(duration_ms)})", style="dim")
        return t

    def _running_label_text(self) -> Text:
        """Label with live elapsed time while running."""
        t = Text()
        t.append("• ", style="bold")
        t.append("SubAgent", style="bold")
        elapsed_ms = int((time.monotonic() - self._start_time) * 1000)
        if elapsed_ms >= 1000:
            t.append(f" ({fmt_duration(elapsed_ms)})", style="dim")
        return t

    def compose(self) -> ComposeResult:
        yield ToolCardHeader(self._running_label_text(), id="sa-label")
        if self._task_prompt:
            yield self._build_task_panel()
        with Vertical(id="sa-panel") as panel:
            panel.border_title = Text(self._render_title())
            yield Static("", id="sa-main")
            with Horizontal(id="sa-sleep-actions"):
                skip = Button(
                    render_text(widget_localizer(self), _SUB_AGENT_SKIP_SLEEP.bind()),
                    id="sa-skip-sleep-btn",
                    compact=True,
                )
                skip.tooltip = render_text(widget_localizer(self), _SUB_AGENT_SKIP_SLEEP_TOOLTIP.bind())
                yield skip
            # Final response prose opts back into selection/right-click copy
            # (the card root is copy-excluded chrome).
            yield VirtualizedMarkdown("", id="sa-result", classes=TOOL_COPY_INCLUDED_CLASS)
            yield Static("", id="sa-retry-banner")
            yield Static("", id="sa-pause-info")
            with Horizontal(id="sa-actions"):
                yield Button(
                    render_text(widget_localizer(self), _SUB_AGENT_RETRY.bind()),
                    id="sa-retry-btn",
                    compact=True,
                )
                yield Button(
                    render_text(widget_localizer(self), _SUB_AGENT_ABORT.bind()),
                    id="sa-abort-btn",
                    compact=True,
                )

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.12, self._spin)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._result_timer is not None:
            self._result_timer.stop()
            self._result_timer = None

    @staticmethod
    def _extract_prompt(args_summary: str, args: dict[str, Any]) -> str:
        """Extract prompt string safely from args_summary or args dict."""
        if args_summary:
            try:
                parsed = json.loads(args_summary)
            except Exception:
                parsed = None
            if isinstance(parsed, dict):
                val = parsed.get("prompt", "")
                if isinstance(val, str):
                    return val
        val = args.get("prompt")
        if isinstance(val, str):
            return val
        return ""

    def _build_task_panel(self) -> Widget:
        task_panel = Widget(VirtualizedMarkdown(self._task_prompt, id="sa-task"), id="sa-task-panel")
        task_panel.border_title = render_text(widget_localizer(self), _SUB_AGENT_TASK.bind())
        return task_panel

    def update_args(self, args: dict[str, Any]) -> None:
        """Refresh the task prompt after approval edits the sub-agent handoff."""
        self.args = args
        self.args_summary = json.dumps(args, ensure_ascii=False) if args else ""
        self._task_prompt = self._extract_prompt(self.args_summary, self.args)
        if self._task_prompt:
            if self.query("#sa-task-panel"):
                with suppress(Exception):
                    self.query_one("#sa-task", VirtualizedMarkdown).update(self._task_prompt)
            else:
                with suppress(Exception):
                    self.mount(self._build_task_panel(), before=self.query_one("#sa-panel", Vertical))
            return
        with suppress(Exception):
            self.query_one("#sa-task-panel", Widget).remove()

    def _spin(self) -> None:
        if self.status == "running":
            self._spin_idx = (self._spin_idx + 1) % len(self._SPINNERS)
            with suppress(Exception):
                self.query_one("#sa-label", Static).update(self._running_label_text())
                self._update_title()
                self.query_one("#sa-main", Static).update(self._render_main())

    def _update_title(self) -> None:
        """Update border title (top-left) and subtitle (bottom-right)."""
        panel = self.query_one("#sa-panel", Vertical)
        panel.border_title = Text(self._render_title())
        panel.border_subtitle = Text(self._render_subtitle())

    @staticmethod
    def _fmt_duration(ms: int) -> str:
        return fmt_duration(ms)

    def _render_title(self) -> str:
        if self.status == "running":
            return f"{self._SPINNERS[self._spin_idx]} {self.tool_name}"
        if self.status == "complete":
            return f"\u2713 {self.tool_name}"
        return f"\u2717 {self.tool_name}"

    def _render_subtitle(self) -> str:
        parts: list[str] = []
        tool_calls = self._total_inner_calls if self.status != "running" else self._progress_tool_calls
        if tool_calls > 0:
            parts.append(self._render_message(_SUB_AGENT_TOOL_CALLS.bind(n=tool_calls)))
        if self._progress_ctx_tokens > 0:
            parts.append(
                self._render_message(_SUB_AGENT_CTX_TOKENS.bind(tokens=format_token_count(self._progress_ctx_tokens)))
            )
        if self._progress_total_usage_tokens or self._usage_unreported_attempts:
            spend = (
                self._render_message(
                    _SUB_AGENT_TOKENS.bind(tokens=format_token_count(self._progress_total_usage_tokens))
                )
                if self._progress_total_usage_tokens
                else self._render_message(_SUB_AGENT_ZERO_REPORTED_TOKENS.bind())
            )
            if self._usage_unreported_attempts:
                attempts = self._render_message(
                    _SUB_AGENT_UNREPORTED_ATTEMPTS.bind(count=self._usage_unreported_attempts)
                )
                spend += f" · {attempts}"
            parts.append(self._render_message(_SUB_AGENT_SPEND.bind(spend=spend)))
        if self._compaction_count > 0:
            parts.append(self._render_message(_SUB_AGENT_COMPACTIONS.bind(n=self._compaction_count)))
        duration_ms = self.duration_ms
        if self.status == "running":
            duration_ms = int((time.monotonic() - self._start_time) * 1000)
            if duration_ms < 1000:
                duration_ms = 0
        if duration_ms:
            parts.append(self._render_message(_SUB_AGENT_DURATION.bind(duration=self._fmt_duration(duration_ms))))
        return " \u00b7 ".join(parts)

    def _render_main(self) -> Text:
        """Render inner tool call list (only while running)."""
        t = Text()
        if self.status != "running" or not self._inner_tools:
            return t
        # Estimate available width for args line (panel width minus padding/border/indent).
        # When the enclosing ToolGroup is collapsed, content_size.width is 0, which
        # would make avail negative and send the wrap loop into an infinite loop.
        try:
            avail = self.query_one("#sa-panel", Vertical).content_size.width - 4
        except Exception:
            avail = _ARGS_BRIEF_MAX
        if avail < 20:
            avail = _ARGS_BRIEF_MAX
        visible = list(self._inner_tools.values())[-_MAX_VISIBLE:]
        for i, entry in enumerate(visible):
            if i > 0:
                t.append("\n")
            self._render_inner_tool_line(t, entry, avail)
        return t

    def _render_inner_tool_line(self, t: Text, entry: _InnerToolEntry, avail_width: int) -> None:
        if entry.status == "running":
            t.append(f"{self._SPINNERS[self._spin_idx]} ", style="yellow")
            t.append(entry.tool_name, style="yellow")
            if entry.started_at:
                elapsed_ms = int((time.monotonic() - entry.started_at) * 1000)
                if elapsed_ms >= 1000:
                    t.append(f" ({self._fmt_duration(elapsed_ms)})", style="dim")
        elif entry.status == "complete":
            t.append("\u2713 ", style="green")
            t.append(entry.tool_name, style="green")
            if entry.duration_ms:
                t.append(f" ({self._fmt_duration(entry.duration_ms)})", style="dim")
            if entry.image_count:
                images = self._render_message(_SUB_AGENT_IMAGES.bind(count=entry.image_count))
                t.append(f" [{images}]", style="dim")
            if entry.artifact_count:
                artifacts = self._render_message(_SUB_AGENT_ARTIFACTS.bind(count=entry.artifact_count))
                t.append(f" [{artifacts}]", style="dim")
        elif entry.status == "skipped":
            t.append("\u21b7 ", style="yellow")
            t.append(
                self._render_message(_SUB_AGENT_TOOL_SKIPPED.bind(tool=entry.tool_name)),
                style="yellow",
            )
            if entry.duration_ms:
                t.append(f" ({self._fmt_duration(entry.duration_ms)})", style="dim")
        elif entry.status == "interrupted":
            t.append("\u2717 ", style="yellow")
            t.append(
                self._render_message(_SUB_AGENT_TOOL_INTERRUPTED.bind(tool=entry.tool_name)),
                style="yellow",
            )
            if entry.duration_ms:
                t.append(f" ({self._fmt_duration(entry.duration_ms)})", style="dim")
        elif entry.status == "rejected":
            t.append("\u2717 ", style="yellow")
            t.append(entry.tool_name, style="yellow")
        elif entry.status == "error":
            t.append("\u2717 ", style="red")
            t.append(entry.tool_name, style="red")
        details = [detail for detail in (entry.args_summary, entry.progress_summary) if detail]
        for detail in details:
            indent = "  "
            prefix = "\u2514 "
            line_width = avail_width - len(indent)
            if line_width < 20:
                line_width = avail_width
            if line_width < 1:
                line_width = _ARGS_BRIEF_MAX
            t.append(f"\n{prefix}", style="dim")
            t.append(detail[:line_width], style="dim")
            pos = line_width
            while pos < len(detail):
                t.append(f"\n{indent}", style="dim")
                t.append(detail[pos : pos + line_width], style="dim")
                pos += line_width

    def _running_sleep_call_id(self) -> str:
        """Return the newest running inner sleep call id, if one is visible."""
        for entry in reversed(self._inner_tools.values()):
            is_sleep = entry.tool_kind == KIND_SLEEP or (not entry.tool_kind and entry.tool_name == "sleep")
            if entry.status == "running" and is_sleep:
                return entry.call_id
        return ""

    def _refresh_sleep_action_state(self) -> None:
        if self.status == "running" and self._running_sleep_call_id():
            self.add_class("-sleeping")
        else:
            self.remove_class("-sleeping")

    def _is_in_collapsed_tool_group(self) -> bool:
        """Return true when this tool sits inside a collapsed tool group."""
        from chrys.app.tui.widgets.chat.tool_call import ToolGroup

        return any(isinstance(ancestor, ToolGroup) and ancestor.collapsed for ancestor in self.ancestors)

    def _show_result_placeholder(self) -> None:
        """Show a cheap placeholder while large markdown waits for its grace timer."""
        with suppress(Exception):
            self.query_one("#sa-main", Static).update(
                Text(self._render_message(_SUB_AGENT_RENDERING_MARKDOWN.bind()), style="dim")
            )

    def _clear_result_placeholder(self) -> None:
        with suppress(Exception):
            self.query_one("#sa-main", Static).update("")

    def _clear_result_markdown(self) -> None:
        """Drop rendered markdown blocks from the result widget."""
        with suppress(Exception):
            self.query_one("#sa-result", VirtualizedMarkdown).update("")

    async def _clear_result_markdown_async(self) -> None:
        """Drop rendered markdown blocks and wait for the lightweight clear."""
        with suppress(Exception):
            await self.query_one("#sa-result", VirtualizedMarkdown).update("")

    def _update_result_widget(self, *, lazy: bool = False) -> None:
        """Render, or defer rendering, the full final response as markdown."""
        result = self.result_text or ""
        should_defer = lazy or len(result) >= _RESULT_MARKDOWN_DEFER_CHARS
        if result and should_defer:
            self._result_pending = True
            if self._result_timer is not None:
                self._result_timer.stop()
                self._result_timer = None
            if lazy or self._is_in_collapsed_tool_group():
                self._clear_result_placeholder()
                self._show_tool_copy_button()
                return
            self._show_result_placeholder()
            self._result_timer = self.set_timer(
                _RESULT_MARKDOWN_MOUNT_DELAY_SECONDS,
                self._schedule_result_render,
            )
            self._show_tool_copy_button()
            return

        self._result_pending = False
        if result:
            with suppress(Exception):
                self.query_one("#sa-result", VirtualizedMarkdown).update(result)
        self.add_class("-done")
        self._show_tool_copy_button()

    def _schedule_result_render(self) -> None:
        """Schedule async markdown rendering after the collapse grace period."""
        self._result_timer = None
        if not self.is_attached:
            self._result_pending = True
            return
        self.call_later(self._mount_deferred_result)

    async def _mount_deferred_result(self) -> None:
        """Render deferred live markdown and request absorb after one refresh."""
        if await self.mount_pending_content():

            def request_absorb_if_still_mounted() -> None:
                if self.is_attached and not self._result_pending and not self._is_in_collapsed_tool_group():
                    self.post_message(GcAbsorbRequested(GcAbsorbReason.STABLE_CONTENT_MOUNTED))

            self.call_after_refresh(request_absorb_if_still_mounted)

    async def mount_pending_content(self) -> bool:
        """Render deferred sub-agent markdown after the parent group expands."""
        if not self._result_pending or not self.result_text:
            return False
        if not self.is_attached or self._is_in_collapsed_tool_group():
            self._result_pending = True
            return False
        if self._result_timer is not None:
            self._result_timer.stop()
            self._result_timer = None
        if self._result_rendering:
            return False
        self._result_rendering = True
        mounted = False
        try:
            result = self.result_text
            with suppress(Exception):
                await self.query_one("#sa-result", VirtualizedMarkdown).update(result)
                mounted = True
            if self._is_in_collapsed_tool_group():
                self._result_pending = True
                await self._clear_result_markdown_async()
                self.remove_class("-done")
                return False
            self._result_pending = False
            self.add_class("-done")
            self._show_tool_copy_button()
            return mounted
        finally:
            self._result_rendering = False

    def release_collapsed_content(self) -> None:
        """Cancel deferred markdown work when a completed group collapses."""
        if self.status not in {"complete", "error", "rejected"} or not self.result_text:
            return
        if not self._result_pending and len(self.result_text) < _RESULT_MARKDOWN_DEFER_CHARS:
            return
        self._result_pending = True
        if self._result_timer is not None:
            self._result_timer.stop()
            self._result_timer = None
        self._clear_result_placeholder()
        self._clear_result_markdown()
        self.remove_class("-done")

    def _tool_copy_input(self) -> tuple[str, str]:
        """Copy the sub-agent prompt without duplicating the full args dict."""
        return "markdown", self._task_prompt or self._render_message(TOOL_VIEW_EMPTY.bind())

    def _tool_view_input_widgets(self) -> list[Widget]:
        """Render the sub-agent prompt as markdown in the detail modal."""
        dark = self._view_dark()
        label = Static(
            Text(render_str(widget_localizer(self), _SUB_AGENT_TASK_PROMPT.bind()), style="bold"),
            classes="tool-view-section-title tool-view-section-title-first",
        )
        widgets: list[Widget] = [
            label,
            build_code_view(
                "markdown",
                self._task_prompt or self._render_message(TOOL_VIEW_EMPTY.bind()),
                dark=dark,
                render_message=self._render_message,
            ),
        ]

        args = self._tool_input_args()
        extra = {key: value for key, value in args.items() if key != "prompt"}
        if extra:
            widgets.extend(build_params_view(extra, dark=dark, render_message=self._render_message))
        return widgets

    def _tool_copy_sections(self) -> list[tuple[str, str, str]]:
        """Preserve the sub-agent's markdown final answer in the copy payload."""
        return [
            (
                self._render_message(TOOL_VIEW_OUTPUT.bind()),
                "markdown",
                self.result_text or self._render_message(TOOL_VIEW_EMPTY.bind()),
            )
        ]

    # --- Invocation linking ---

    def claim_invocation(self, invocation_id: str) -> None:
        """Link this widget to a specific sub-agent invocation."""
        self._invocation_id = invocation_id

    def update_progress(
        self,
        tool_call_count: int,
        total_tokens: int,
        total_usage_tokens: int = 0,
        usage_unreported_attempts: int = 0,
    ) -> None:
        """Update cumulative progress stats and refresh the border title."""
        self._progress_tool_calls = tool_call_count
        self._progress_ctx_tokens = total_tokens
        self._progress_total_usage_tokens = total_usage_tokens
        self._usage_unreported_attempts = usage_unreported_attempts
        if self.status == "running":
            self._update_title()

    # --- Inner tool management ---

    async def add_inner_tool_start(
        self,
        call_id: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        tool_kind: str = "",
    ) -> None:
        """Insert or refresh a running inner tool call by its lifetime id."""
        if call_id in self._terminal_inner_call_ids:
            return
        # A new inner tool call means the sub-agent is making forward
        # progress — if a retry banner is still visible from a prior
        # transient failure, the retry has clearly succeeded. Drop it
        # now instead of letting it linger until the parent tool call
        # finally resolves.
        if self.has_class("-retrying"):
            self.remove_class("-retrying")
            with suppress(Exception):
                self.query_one("#sa-retry-banner", Static).update("")
        args_summary = _fmt_args(args) if args else ""
        entry = self._inner_tools.get(call_id)
        if entry is None:
            entry = _InnerToolEntry(
                call_id=call_id,
                tool_name=tool_name,
                tool_kind=tool_kind,
                args_summary=args_summary,
            )
            self._inner_tools[call_id] = entry
        else:
            entry.tool_name = tool_name
            entry.tool_kind = tool_kind
            entry.args_summary = args_summary
        if call_id not in self._seen_inner_call_ids:
            self._seen_inner_call_ids.add(call_id)
            self._total_inner_calls += 1
        self._evict_inner_tools_over_cap()

        with suppress(Exception):
            self.query_one("#sa-main", Static).update(self._render_main())
        self._refresh_sleep_action_state()

    def _evict_inner_tools_over_cap(self) -> None:
        """Evict oldest completed entries (oldest overall as last resort)."""
        while len(self._inner_tools) > _MAX_VISIBLE:
            evicted = False
            for cid, e in self._inner_tools.items():
                if e.status != "running":
                    del self._inner_tools[cid]
                    evicted = True
                    break
            if not evicted:
                cid = next(iter(self._inner_tools))
                del self._inner_tools[cid]
                break

    def update_inner_tool_args(self, call_id: str, args: dict[str, Any]) -> None:
        """Refresh arguments on a running nested tool entry."""
        entry = self._inner_tools.get(call_id)
        if entry is None or call_id in self._terminal_inner_call_ids:
            return
        entry.args_summary = _fmt_args(args) if args else ""
        with suppress(Exception):
            self.query_one("#sa-main", Static).update(self._render_main())

    def update_inner_tool_progress(
        self,
        call_id: str,
        lines: list[str],
        *,
        image_contents: list[Any] | None = None,
    ) -> None:
        """Apply the latest bounded progress snapshot to a nested entry."""
        entry = self._inner_tools.get(call_id)
        if entry is None or call_id in self._terminal_inner_call_ids:
            return
        if lines:
            entry.progress_summary = lines[-1][:_ARGS_BRIEF_MAX]
        if image_contents:
            entry.image_count = len(image_contents)
        with suppress(Exception):
            self.query_one("#sa-main", Static).update(self._render_main())

    def update_inner_tool_status(
        self,
        call_id: str,
        status: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Apply one canonical lifecycle status to a nested entry."""
        entry = self._inner_tools.get(call_id)
        if entry is None or call_id in self._terminal_inner_call_ids:
            return
        if status not in {"completed", "failed", "interrupted"}:
            return
        result_text = str((metadata or {}).get("result_text", ""))
        if status == "completed":
            # A completed provider snapshot can precede output_item.done. Show
            # it as complete now, but leave the id open for one authoritative
            # result carrying final images, duration, and metadata.
            self._apply_inner_tool_completion(
                call_id,
                entry,
                result_text,
                0,
                metadata=metadata,
                authoritative_result=False,
            )
            return
        entry.status = "interrupted" if status == "interrupted" else "error"
        if result_text:
            entry.progress_summary = result_text[:_ARGS_BRIEF_MAX]
        if not entry.completion_counted:
            entry.completion_counted = True
            self._completed_inner_calls += 1
        self._refresh_sleep_action_state()
        with suppress(Exception):
            self.query_one("#sa-main", Static).update(self._render_main())

    def add_compaction_start(self, compaction_id: str) -> None:
        """Show a live "Compacting conversation..." line in the inner feed.

        Phase-4 compaction is not a tool call — the entry rides the inner
        tool list purely for visibility (deliberately not counted in
        ``_total_inner_calls``), with ``started_at`` driving a live elapsed
        display refreshed by the spinner timer.
        """
        entry = _InnerToolEntry(
            call_id=f"compaction:{compaction_id}",
            tool_name=self._render_message(_SUB_AGENT_COMPACTING.bind()),
            tool_kind=_COMPACTION_ENTRY_KIND,
            args_summary="",
            started_at=time.monotonic(),
        )
        self._inner_tools[entry.call_id] = entry
        self._evict_inner_tools_over_cap()
        with suppress(Exception):
            self.query_one("#sa-main", Static).update(self._render_main())

    def complete_compaction(
        self,
        compaction_id: str,
        *,
        outcome: str,
        duration_ms: int = 0,
        format_violation: str = "",
        failure_reason: str = "",
    ) -> None:
        """Flip the compaction line to its terminal state."""
        entry = self._inner_tools.get(f"compaction:{compaction_id}")
        if entry is None:
            return
        entry.started_at = 0.0
        entry.duration_ms = duration_ms
        if outcome == "ok":
            entry.status = "complete"
            if format_violation:
                entry.tool_name = self._render_message(_SUB_AGENT_COMPACTED_WARNING.bind(violation=format_violation))
            else:
                entry.tool_name = self._render_message(_SUB_AGENT_COMPACTED.bind())
            # A successful finish proves the LAST_WORDS retry loop
            # recovered — drop a lingering "Retrying in …" banner, same
            # forward-progress rule as add_inner_tool_start. (Terminal
            # failure pauses the sub-agent, whose flow clears it instead.)
            if self.has_class("-retrying"):
                self.remove_class("-retrying")
                with suppress(Exception):
                    self.query_one("#sa-retry-banner", Static).update("")
        elif outcome == "canceled":
            # The interrupted branch renders "{tool_name} interrupted".
            entry.status = "interrupted"
            entry.tool_name = self._render_message(_SUB_AGENT_COMPACTION_NAME.bind())
        else:
            entry.status = "error"
            entry.tool_name = (
                self._render_message(_SUB_AGENT_COMPACTION_FAILED_REASON.bind(reason=failure_reason))
                if failure_reason
                else self._render_message(_SUB_AGENT_COMPACTION_FAILED.bind())
            )
        with suppress(Exception):
            self.query_one("#sa-main", Static).update(self._render_main())

    def record_compaction_committed(self, compaction_id: str) -> None:
        """Count a durably committed compaction round in the subtitle.

        Fires on the committed signal, which trails ``complete_compaction``'s
        finished(ok) — a successful note generation whose spill write later
        failed never commits, so counting here (not on finished-ok) keeps
        "Compactions: N" honest.  Independent of the feed entry: the line
        may already be evicted by the visibility cap.
        """
        self._compaction_count += 1
        with suppress(Exception):
            self._update_title()

    def complete_inner_tool(
        self,
        call_id: str,
        result: str,
        duration_ms: int,
        image_contents: list[Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        approval: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark an inner tool call as complete."""
        entry = self._inner_tools.get(call_id)
        if not entry or call_id in self._terminal_inner_call_ids:
            return
        self._apply_inner_tool_completion(
            call_id,
            entry,
            result,
            duration_ms,
            image_contents=image_contents,
            artifacts=artifacts,
            approval=approval,
            metadata=metadata,
            authoritative_result=True,
        )

    def _apply_inner_tool_completion(
        self,
        call_id: str,
        entry: _InnerToolEntry,
        result: str,
        duration_ms: int,
        *,
        image_contents: list[Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        approval: str | None = None,
        metadata: dict[str, Any] | None = None,
        authoritative_result: bool,
    ) -> None:
        """Render completion while keeping status-only snapshots replaceable."""
        if authoritative_result:
            self._terminal_inner_call_ids.add(call_id)
            # A terminal id short-circuits add_inner_tool_start before the
            # seen-check, so its _seen membership is dead weight — drop it to
            # keep the dedupe set bounded by in-flight calls.
            self._seen_inner_call_ids.discard(call_id)
        render_status = tool_result_render_status(result, approval, entry.tool_kind, metadata, entry.tool_name)
        if render_status == "rejected":
            entry.status = "rejected"
        elif isinstance(metadata, dict) and metadata.get("sleep_interrupted") is True:
            entry.status = "interrupted"
        elif isinstance(metadata, dict) and metadata.get("sleep_skipped") is True:
            entry.status = "skipped"
        elif render_status == "error":
            entry.status = "error"
        else:
            entry.status = "complete"
        if authoritative_result:
            entry.duration_ms = duration_ms
            entry.image_count = len(image_contents or [])
            entry.artifact_count = len(artifacts or [])
        if not entry.completion_counted:
            entry.completion_counted = True
            self._completed_inner_calls += 1
        self._refresh_sleep_action_state()
        with suppress(Exception):
            self.query_one("#sa-main", Static).update(self._render_main())

    # --- ToolCall protocol ---

    def _drop_inner_call_tracking(self) -> None:
        """Free per-call id bookkeeping once the parent call is terminal.

        Completed cards stay mounted for the session; keeping every inner
        call id (up to thousands per ACP attempt, fresh ids per retry)
        would grow the seen/terminal sets without bound across a long
        session. Late inner events are already inert after a parent
        terminal: completions need a live ``_inner_tools`` entry, and
        session replay rebuilds a fresh card from scratch.
        """
        self._inner_tools.clear()
        self._seen_inner_call_ids.clear()
        self._terminal_inner_call_ids.clear()

    def _clear_transient_state(self) -> None:
        """Drop stale retry/pause CSS classes + banner text before a terminal transition.

        Both ``-retrying`` and ``-paused`` belong to *in-flight* states.
        When the sub-agent finally completes or errors out, the banners
        those classes reveal should not remain visible — otherwise the
        card shows a "Retrying in 7s…" line *after* a successful
        "Recovered" result, or a "Stream stalled" pause-info line *after*
        the user has already aborted.
        """
        self.remove_class("-retrying")
        self.remove_class("-paused")
        self.remove_class("-sleeping")
        with suppress(Exception):
            self.query_one("#sa-retry-banner", Static).update("")
            self.query_one("#sa-pause-info", Static).update("")

    def set_complete(self, result: str, duration_ms: int = 0, **kwargs: Any) -> None:
        """Mark this sub-agent call as complete (parent tool call finished).

        If structured metadata or legacy result text indicates a sub-agent
        error, redirect to ``set_error()`` so the widget renders with the
        error style.
        Rejected tools (approval declined) get a separate warning style.
        """
        self.approval = kwargs.get("approval")
        metadata = kwargs.get("metadata")
        self.metadata = metadata if isinstance(metadata, dict) else {}
        if self.metadata.get(TOOL_INTERRUPTED_METADATA_KEY) is True:
            self.result_text = result
            self.duration_ms = duration_ms
            self.status = "interrupted"
            self._drop_inner_call_tracking()
            self._clear_transient_state()
            self.add_class("-rejected")
            if self._timer is not None:
                self._timer.stop()
            with suppress(Exception):
                panel = self.query_one("#sa-panel")
                panel.border_title = Text(self.tool_name)
                panel.border_subtitle = render_text(widget_localizer(self), TOOL_CARD_INTERRUPTED.bind())
                self.query_one("#sa-label", Static).update(self._label_text(duration_ms))
            self._update_result_widget(lazy=bool(kwargs.get("lazy")))
            return
        render_status = tool_result_render_status(
            result,
            self.approval,
            self.tool_kind or KIND_SUB_AGENT,
            self.metadata,
            self.tool_name,
        )
        if render_status == "rejected":
            self.result_text = result
            self.duration_ms = duration_ms
            self.status = "rejected"
            self._drop_inner_call_tracking()
            self._clear_transient_state()
            self.add_class("-rejected")
            if self._timer is not None:
                self._timer.stop()
            with suppress(Exception):
                panel = self.query_one("#sa-panel")
                panel.border_title = Text(self.tool_name)
                panel.border_subtitle = render_text(widget_localizer(self), TOOL_CARD_REJECTED.bind())
                self.query_one("#sa-label", Static).update(self._label_text(duration_ms))
            self._update_result_widget(lazy=bool(kwargs.get("lazy")))
            return
        if render_status == "error":
            self._set_error(result, duration_ms)
            return
        self.result_text = result
        self.duration_ms = duration_ms
        self._drop_inner_call_tracking()
        self.status = "complete"
        self._clear_transient_state()
        self.add_class("-complete")
        if self._timer is not None:
            self._timer.stop()
        with suppress(Exception):
            self.query_one("#sa-label", Static).update(self._label_text(duration_ms))
            self._update_title()
            self.query_one("#sa-main", Static).update(self._render_main())
        self._update_result_widget(lazy=bool(kwargs.get("lazy")))

    def set_error(self, error: str) -> None:
        """Mark this sub-agent call as failed."""
        self._set_error(error, int((time.monotonic() - self._start_time) * 1000))

    def _set_error(self, error: str, duration_ms: int) -> None:
        """Apply an error with either live elapsed or persisted duration."""
        self.result_text = error
        self.duration_ms = duration_ms
        self._drop_inner_call_tracking()
        self.status = "error"
        self._clear_transient_state()
        self.add_class("-error")
        if self._timer is not None:
            self._timer.stop()
        with suppress(Exception):
            self.query_one("#sa-label", Static).update(self._label_text(duration_ms))
            self._update_title()
            self.query_one("#sa-main", Static).update(self._render_main())
        self._update_result_widget()

    # --- Paused / retry / resumed state ---

    def set_retry_attempt(self, message: str, attempt: int, max_attempts: int, delay_seconds: int) -> None:
        """Show an inline retry banner for an auto-retry attempt.

        Called for :class:`SubAgentRetryAttempt` events — the controller
        is retrying transient failures without user intervention. The
        banner gives the user visibility into the retry cadence.  The
        ``↻`` prefix visually marks the line as a retry state sub-entry,
        distinct from the result line that follows it on recovery.
        """
        self.add_class("-retrying")
        retry_message = self._render_message(
            _SUB_AGENT_RETRYING.bind(
                delay_seconds=delay_seconds,
                attempt=attempt,
                max_attempts=max_attempts,
                message=message,
            )
        )
        with suppress(Exception):
            self.query_one("#sa-retry-banner", Static).update(Text(retry_message, style="italic"))

    def set_paused(
        self,
        reason: str,
        last_error: str,
        retry_attempts: int,
        diagnostic_path: str | None = None,
    ) -> None:
        """Transition the card to a paused state with Retry/Abort buttons.

        Called when :class:`SubAgentPaused` arrives. Auto-retry (if any
        happened first) is now finished; the banner is replaced with the
        pause-info block and the action row is revealed.
        """
        self.status = "paused"
        # Paused supersedes the retrying banner — the retry run completed
        # (unsuccessfully). Keep visual history of the last error only.
        self.remove_class("-retrying")
        self.remove_class("-sleeping")
        self.add_class("-paused")
        self.remove_class("-complete")
        self.remove_class("-error")
        if self._timer is not None:
            self._timer.stop()
        elapsed_ms = int((time.monotonic() - self._start_time) * 1000)
        reason_definition = _SUB_AGENT_REASON_MESSAGES.get(reason, _SUB_AGENT_REASON_PAUSED)
        reason_label = self._render_message(reason_definition.bind())
        info_lines = [reason_label]
        if retry_attempts:
            info_lines.append(self._render_message(_SUB_AGENT_AFTER_RETRIES.bind(count=retry_attempts)))
        if last_error:
            info_lines.append(last_error)
        if diagnostic_path:
            # Display copy only — the operational path stays raw on the event.
            info_lines.append(
                self._render_message(_SUB_AGENT_DIAGNOSTICS.bind(path=surrogate_safe_text(diagnostic_path)))
            )
        info = "\n".join(info_lines)
        with suppress(Exception):
            self.query_one("#sa-label", Static).update(self._label_text(elapsed_ms))
            panel = self.query_one("#sa-panel", Vertical)
            panel.border_title = Text(f"\u25aa {self.tool_name}")  # small square = paused
            panel.border_subtitle = render_text(widget_localizer(self), _SUB_AGENT_PAUSED.bind())
            self.query_one("#sa-main", Static).update(self._render_main())
            self.query_one("#sa-pause-info", Static).update(Text(info, style="red"))
            self.query_one("#sa-retry-banner", Static).update("")

    def set_resumed_after_pause(self) -> None:
        """Clear the paused state and reset the card to a live running view.

        Called when :class:`SubAgentResumed` arrives — the user
        clicked Retry and the controller has re-entered running state.
        """
        if self.status != "paused":
            return
        self.status = "running"
        self.remove_class("-paused")
        self._refresh_sleep_action_state()
        self._start_time = time.monotonic()  # reset elapsed clock for the new attempt
        self._spin_idx = 0
        # Inner-call tracking is attempt-local: the dead attempt's stream is
        # fully drained (the bus delivers synchronously), and a fresh attempt
        # may legitimately reuse raw call ids (the ACP translator scopes ids
        # with an attempt prefix for exactly that reason). Keeping the old
        # feed would pin its ids forever across unlimited manual retries and
        # make the stale terminal-guard swallow a reused id's new call.
        self._drop_inner_call_tracking()
        with suppress(Exception):
            self.query_one("#sa-pause-info", Static).update("")
            self.query_one("#sa-retry-banner", Static).update("")
            self.query_one("#sa-label", Static).update(self._running_label_text())
            self._update_title()
            self.query_one("#sa-main", Static).update(self._render_main())
        self.remove_class("-retrying")
        # Restart the spinner timer.
        if self._timer is not None:
            self._timer.stop()
        self._timer = self.set_interval(0.12, self._spin)

    def set_cascade_aborted(self) -> None:
        """Mark the card as cascade-aborted by a global interrupt."""
        self.remove_class("-paused")
        self.remove_class("-retrying")
        self.set_error("Error: cancelled (global interrupt)")

    def set_aborted(self, last_error: str) -> None:
        """Mark the card as aborted by the user after a pause.

        Fired on :class:`SubAgentAborted`.  Clears the paused banner (so
        the Retry/Abort buttons and pause-info disappear immediately) and
        routes through :meth:`set_error` so the card gets the standard
        error styling + a message indicating the abort was user-driven.
        """
        self.remove_class("-paused")
        self.remove_class("-retrying")
        error_text = (
            f"Error: sub-agent aborted by user after failure — {last_error}"
            if last_error
            else "Error: sub-agent aborted by user"
        )
        self.set_error(error_text)

    # --- Button actions ---

    def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[name-defined]
        """Translate a Retry/Abort click into a bubbled message.

        Guarded by ``_invocation_id`` being set — we should never
        dispatch a retry/abort for a card that hasn't been linked yet
        (that would mean no controller exists to receive it).
        """
        if event.button.id == "sa-skip-sleep-btn":
            if call_id := self._running_sleep_call_id():
                self.post_message(SleepSkipClicked(call_id))
            event.stop()
            return
        if self._invocation_id is None:
            return
        if event.button.id == "sa-retry-btn":
            self.post_message(SubAgentRetryClicked(self._invocation_id))
        elif event.button.id == "sa-abort-btn":
            self.post_message(SubAgentAbortClicked(self._invocation_id))
        event.stop()
