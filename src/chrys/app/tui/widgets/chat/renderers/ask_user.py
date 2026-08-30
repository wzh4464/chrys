# Copyright (c) 2026 Chrys. All rights reserved.

"""Renderer for the ``ask_user`` tool.

Shows the question as Markdown and the user's answer as full plain text in
separate bordered panels.
"""

from __future__ import annotations

import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any, ClassVar

from rich.text import Text
from textual import events, on
from textual.containers import Vertical
from textual.message import Message
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Static

from chrys.app.tui.i18n import render_str, render_text, widget_localizer
from chrys.app.tui.widgets import (
    ASK_USER_INPUT_MIN_HEIGHT,
    AskUserOptions,
    AskUserResponseFooter,
    AskUserResponseResized,
    AskUserSubmitted,
    normalize_ask_user_options,
)
from chrys.app.tui.widgets.chat.tool_call import (
    BaseToolCard,
    ToolCardHeader,
    fmt_duration,
    tool_result_render_status,
)
from chrys.app.tui.widgets.chat.tool_view_builders import TOOL_VIEW_EMPTY, build_code_view, build_params_view
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.foundation.i18n import msg
from chrys.foundation.tool_kinds import KIND_ASK_USER

_ASK_USER_QUESTION = msg("tui.tool_card.ask_user.question", fallback="Question")
_ASK_USER_ANSWER = msg("tui.tool_card.ask_user.answer", fallback="Answer")
_ASK_USER_WAITING = msg("tui.tool_card.ask_user.waiting", fallback="waiting")

if TYPE_CHECKING:
    from textual.app import ComposeResult

_USER_RESPONSE_PREFIX = "User response:"
_INLINE_MIN_CONTENT_HEIGHT = 6
_INLINE_BUTTON_ROW_HEIGHT = 3
_INLINE_PANEL_FRAME_ROWS = 2
_INLINE_OPTIONS_BOTTOM_MARGIN_ROWS = 2


class AskUserInlineSubmitted(Message):
    """User submitted an inline ask_user response from the chat renderer."""

    def __init__(self, call_id: str, request_id: str, text: str) -> None:
        super().__init__()
        self.call_id = call_id
        self.request_id = request_id
        self.text = text


class AskUserInlineResized(Message):
    """Inline ask_user controls changed size inside a tool renderer."""

    def __init__(self, call_id: str) -> None:
        super().__init__()
        self.call_id = call_id


class _AskUserInlinePrompt(Vertical):
    """Inline answer controls mounted inside the ask_user renderer."""

    def __init__(
        self,
        request_id: str,
        options: list[str] | None = None,
        *,
        initial_response: str = "",
    ) -> None:
        super().__init__(id="ask-inline")
        self._request_id = request_id
        self._options = normalize_ask_user_options(options)
        self._initial_response = initial_response

    def compose(self) -> ComposeResult:
        if self._options:
            yield AskUserOptions(self._request_id, self._options)
        yield AskUserResponseFooter(
            self._request_id,
            initial_response=self._initial_response,
            defer_layout_to_parent=True,
        )

    def focus_initial(self) -> None:
        """Focus the first meaningful inline control."""
        if self._options:
            with suppress(Exception):
                self.query_one("#askuser-opt-0").focus()
                return
        with suppress(Exception):
            self.query_one(AskUserResponseFooter).focus_input()

    def disable_controls(self) -> None:
        """Disable all controls in this inline prompt."""
        for options in self.query(AskUserOptions):
            options.disable_controls()
        for footer in self.query(AskUserResponseFooter):
            footer.disable_controls()


def _extract_question(args_summary: str, args: dict[str, Any]) -> str:
    """Extract the question string from normalized args or a JSON args summary."""
    value = args.get("question")
    if isinstance(value, str):
        return value
    if not args_summary:
        return ""
    try:
        parsed = json.loads(args_summary)
    except TypeError, ValueError:
        return ""
    if not isinstance(parsed, dict):
        return ""
    value = parsed.get("question")
    return value if isinstance(value, str) else ""


def _answer_from_result(result: str) -> str:
    """Return the user-facing answer body from the middleware result string."""
    if result.startswith(_USER_RESPONSE_PREFIX):
        return result[len(_USER_RESPONSE_PREFIX) :].removeprefix(" ")
    return result


class AskUserToolCall(BaseToolCard):
    """Rich renderer for ``ask_user`` tool calls."""

    DEFAULT_CSS = """
    AskUserToolCall {
        height: auto;
        padding: 0 0 0 2;
        margin: 0 0 1 0;
    }
    AskUserToolCall #ask-label {
        height: auto;
    }
    AskUserToolCall > #ask-question-panel {
        height: auto;
        margin: 0 0 0 2;
        border: round $tui-border-warning 50%;
        border-title-color: $text;
        border-title-style: not bold;
        padding: 0 1;
    }
    AskUserToolCall > #ask-answer-panel {
        height: auto;
        margin: 0 0 0 2;
        border: round $tui-border-warning 50%;
        border-title-color: $warning;
        border-title-style: bold;
        padding: 0 1;
    }
    AskUserToolCall.-success > #ask-answer-panel {
        border: round $tui-border-success 30%;
        border-title-color: $success;
        border-title-style: not bold;
    }
    AskUserToolCall.-error > #ask-answer-panel {
        border: round $tui-border-error 50%;
        border-title-color: $text-error;
        border-title-style: not bold;
    }
    AskUserToolCall.-rejected > #ask-answer-panel {
        border: round $tui-border-warning 50%;
        border-title-color: $warning;
        border-title-style: not bold;
    }
    AskUserToolCall #ask-question {
        height: auto;
        min-height: 1;
        padding: 0;
        background: transparent;
    }
    AskUserToolCall #ask-spinner {
        height: auto;
    }
    AskUserToolCall #ask-answer {
        height: auto;
        color: $text-muted;
        display: none;
    }
    AskUserToolCall #ask-inline {
        height: auto;
        min-height: 6;
    }
    AskUserToolCall #ask-inline > #askuser-footer {
        min-height: 6;
    }
    AskUserToolCall #askuser-options {
        width: 100%;
        height: auto;
        min-height: 0;
        margin: 0 0 2 0;
        padding: 0;
    }
    AskUserToolCall #askuser-options > Button {
        width: 100%;
        min-height: 3;
        margin-bottom: 0;
    }
    AskUserToolCall #askuser-footer {
        width: 100%;
        height: auto;
    }
    AskUserToolCall #askuser-input {
        width: 100%;
        height: auto;
        /* Keep in sync with ASK_USER_INPUT_MIN_HEIGHT / ASK_USER_INPUT_MAX_HEIGHT. */
        min-height: 3;
        max-height: 7;
        /* The compact TextArea strips its frame with an !important rule of its
           own; this frame must outrank it, and must not depend on the modal
           dialog's stylesheet having been loaded first. */
        border: round $tui-border-accent $border-opacity !important;
        margin: 0;
    }
    AskUserToolCall #askuser-buttons {
        width: 100%;
        align: center top;
        height: 3;
        padding: 0;
    }
    AskUserToolCall #askuser-buttons > Button {
        margin: 0 1;
    }
    AskUserToolCall #askuser-inline {
        min-width: 13;
    }
    AskUserToolCall.-inline #ask-spinner {
        display: none;
    }
    AskUserToolCall.-done #ask-spinner {
        display: none;
    }
    AskUserToolCall.-done #ask-answer {
        display: block;
    }
    """

    _SPINNERS: ClassVar[str] = "◐◓◑◒"

    def __init__(
        self,
        call_id: str,
        tool_name: str,
        args_summary: str = "",
        args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(call_id, tool_name, args_summary, args=args)
        self._question = _extract_question(args_summary, self.args)
        self._spin_idx = 0
        self._timer: Timer | None = None
        self._inline_request_id = ""
        self._inline_submitted = False
        self._inline_resize_sync_pending = False
        self._inline_last_width = 0

    def _label_text(self, duration_ms: int = 0) -> Text:
        t = Text()
        t.append("• ", style="bold")
        t.append(self.tool_name, style="bold")
        if duration_ms:
            t.append(f" ({fmt_duration(duration_ms)})", style="dim")
        return t

    def compose(self) -> ComposeResult:
        yield ToolCardHeader(self._label_text(), id="ask-label")

        question_panel = Widget(VirtualizedMarkdown(self._question, id="ask-question"), id="ask-question-panel")
        question_panel.border_title = render_text(widget_localizer(self), _ASK_USER_QUESTION.bind())
        yield question_panel

        answer_panel = Widget(id="ask-answer-panel")
        answer_panel.border_title = render_text(widget_localizer(self), _ASK_USER_ANSWER.bind())
        yield answer_panel

    def on_mount(self) -> None:
        panel = self.query_one("#ask-answer-panel")
        panel.mount(Static(self._render_spinner(), id="ask-spinner"))
        panel.mount(Static("", id="ask-answer"))
        self._timer = self.set_interval(0.12, self._spin)

    def on_unmount(self) -> None:
        if self._timer is not None:
            self._timer.stop()

    def on_resize(self, event: events.Resize) -> None:
        if self.status != "running" or not self._inline_request_id:
            return
        if event.size.width == self._inline_last_width:
            return
        # Same contract as the options widget below: every width change is
        # recorded and released, even while a sync is in flight — dropping one
        # would freeze the old measurements with nothing left to notice.
        self._inline_last_width = event.size.width
        self._release_inline_option_height()
        if self._inline_resize_sync_pending:
            return
        self._inline_resize_sync_pending = True
        self.call_after_refresh(self._sync_inline_layout_after_resize)

    def _spin(self) -> None:
        if self.status == "running":
            self._spin_idx = (self._spin_idx + 1) % len(self._SPINNERS)
            with suppress(Exception):
                self.query_one("#ask-spinner", Static).update(self._render_spinner(), layout=False)

    def _render_spinner(self) -> Text:
        t = Text()
        t.append(f"{self._SPINNERS[self._spin_idx]} ", style="yellow")
        t.append(render_str(widget_localizer(self), _ASK_USER_WAITING.bind()), style="yellow")
        return t

    def show_inline_prompt(
        self,
        request_id: str,
        options: list[str] | None = None,
        *,
        draft_text: str = "",
    ) -> bool:
        """Mount live answer controls inside the renderer."""
        if self.status != "running" or not request_id:
            return False
        prompt = _AskUserInlinePrompt(request_id, options, initial_response=draft_text)
        try:
            panel = self.query_one("#ask-answer-panel")
            with suppress(Exception):
                self.query_one("#ask-inline").remove()
            panel.mount(prompt)
            if self._sync_inline_layout():
                self.refresh(layout=True)
        except Exception:
            return False

        self._inline_request_id = request_id
        self._inline_submitted = False
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        with suppress(Exception):
            self.query_one("#ask-spinner", Static).display = False

        self.add_class("-inline")
        self.call_after_refresh(self._sync_inline_layout_after_resize)
        self.call_after_refresh(prompt.focus_initial)
        return True

    def clear_inline_prompt(self) -> None:
        """Remove any active inline answer controls."""
        self.remove_class("-inline")
        self._inline_request_id = ""
        self._inline_submitted = False
        self._inline_resize_sync_pending = False
        self._inline_last_width = 0
        with suppress(Exception):
            answer_panel = self.query_one("#ask-answer-panel")
            answer_panel.styles.height = "auto"
            answer_panel.styles.min_height = 0
            self.styles.min_height = 0
        with suppress(Exception):
            self.query_one("#ask-inline").remove()

    def _lock_inline_prompt(self) -> None:
        with suppress(Exception):
            self.query_one("#ask-inline", _AskUserInlinePrompt).disable_controls()

    def _sync_inline_layout_after_resize(self) -> None:
        self._inline_resize_sync_pending = False
        if self._sync_inline_layout():
            self.post_message(AskUserInlineResized(self.call_id))

    def _release_inline_option_height(self) -> None:
        with suppress(Exception):
            # The options widget frees its buttons too: releasing only the
            # container would let a sync measure buttons still frozen at the
            # old width's heights and freeze the container around them.
            self.query_one("#askuser-options", AskUserOptions).release_option_heights()

    def _sync_inline_layout(self, input_height: int | None = None) -> bool:
        """Keep the answer panel height in lockstep with live inline controls."""
        if self.status != "running":
            return False
        try:
            prompt = self.query_one("#ask-inline", _AskUserInlinePrompt)
            answer_panel = self.query_one("#ask-answer-panel")
        except Exception:
            return False

        content_height = _INLINE_MIN_CONTENT_HEIGHT
        with suppress(Exception):
            footer = prompt.query_one("#askuser-footer", AskUserResponseFooter)
            input_rows = (
                max(input_height, ASK_USER_INPUT_MIN_HEIGHT)
                if input_height is not None
                else max(footer.measure_input_height(), ASK_USER_INPUT_MIN_HEIGHT)
            )
            option_height = 0
            if prompt._options:
                option_rows = len(prompt._options) * _INLINE_BUTTON_ROW_HEIGHT
                with suppress(Exception):
                    options = prompt.query_one("#askuser-options", AskUserOptions)
                    option_rows = max(option_rows, options.sync_option_heights())
                option_height = option_rows + _INLINE_OPTIONS_BOTTOM_MARGIN_ROWS
            content_height = max(content_height, option_height + input_rows + _INLINE_BUTTON_ROW_HEIGHT)

        panel_height = content_height + _INLINE_PANEL_FRAME_ROWS
        tool_min_height = panel_height
        for child in self.children:
            if child is answer_panel:
                continue
            tool_min_height += child.region.height
        changed = (
            str(prompt.styles.height) != str(content_height)
            or str(prompt.styles.min_height) != str(content_height)
            or str(answer_panel.styles.height) != str(panel_height)
            or str(answer_panel.styles.min_height) != str(panel_height)
            or str(self.styles.min_height) != str(tool_min_height)
        )
        if not changed:
            return False
        prompt.styles.height = content_height
        prompt.styles.min_height = content_height
        answer_panel.styles.height = panel_height
        answer_panel.styles.min_height = panel_height
        self.styles.min_height = tool_min_height
        return True

    @on(AskUserSubmitted)
    def _on_ask_user_submitted(self, event: AskUserSubmitted) -> None:
        if event.request_id != self._inline_request_id or self.status != "running":
            return
        event.stop()
        if self._inline_submitted:
            return
        self._inline_submitted = True
        self._lock_inline_prompt()
        self.post_message(AskUserInlineSubmitted(self.call_id, event.request_id, event.text))

    @on(AskUserResponseResized)
    def _on_ask_user_response_resized(self, event: AskUserResponseResized) -> None:
        if self.status != "running" or not self._inline_request_id:
            return
        event.stop()
        if self._sync_inline_layout(event.input_height):
            self.post_message(AskUserInlineResized(self.call_id))
        if not self._inline_resize_sync_pending:
            self._inline_resize_sync_pending = True
            self.call_after_refresh(self._sync_inline_layout_after_resize)

    def set_complete(self, result: str, duration_ms: int = 0, **kwargs: Any) -> None:
        """Mark the question as answered."""
        self.clear_inline_prompt()
        self.result_text = result
        self.duration_ms = duration_ms
        self.approval = kwargs.get("approval")
        metadata = kwargs.get("metadata")
        self.metadata = metadata if isinstance(metadata, dict) else {}
        self.status = "complete"
        if self._timer is not None:
            self._timer.stop()

        with suppress(Exception):
            self.query_one("#ask-label", Static).update(self._label_text(duration_ms))

        render_status = tool_result_render_status(result, self.approval, self.tool_kind or KIND_ASK_USER, self.metadata)
        self.status = render_status
        if render_status == "rejected":
            self.add_class("-rejected")
        elif render_status == "error":
            self.add_class("-error")
        else:
            self.add_class("-success")

        with suppress(Exception):
            self.query_one("#ask-answer", Static).update(Text(_answer_from_result(result)))
        self.add_class("-done")
        self._show_tool_copy_button()

    def set_error(self, error: str) -> None:
        """Mark the question as failed."""
        self.clear_inline_prompt()
        self.result_text = error
        self.status = "error"
        if self._timer is not None:
            self._timer.stop()

        self.add_class("-error")
        with suppress(Exception):
            self.query_one("#ask-answer", Static).update(Text(error))
        self.add_class("-done")
        self._show_tool_copy_button()

    def _tool_copy_input(self) -> tuple[str, str]:
        """Preserve the question as markdown in copied transcripts."""
        return "markdown", self._question or self._render_message(TOOL_VIEW_EMPTY.bind())

    def _tool_copy_sections(self) -> list[tuple[str, str, str]]:
        """Copy the plain answer body without the middleware transport prefix."""
        return [
            (
                self._render_message(_ASK_USER_ANSWER.bind()),
                "text",
                _answer_from_result(self.result_text) or self._render_message(TOOL_VIEW_EMPTY.bind()),
            )
        ]

    def _input_extra_args(self) -> dict[str, Any]:
        """Return tool input args other than the question (e.g. ``options``)."""
        args = self._tool_input_args()
        return {key: value for key, value in args.items() if key != "question"}

    def _tool_view_input_widgets(self) -> list[Widget]:
        """Render the question as markdown plus any remaining args as JSON."""
        dark = self._view_dark()
        widgets: list[Widget] = [
            build_code_view(
                "markdown",
                self._question or self._render_message(TOOL_VIEW_EMPTY.bind()),
                dark=dark,
                render_message=self._render_message,
            )
        ]
        extra = self._input_extra_args()
        if extra:
            widgets.extend(build_params_view(extra, dark=dark, render_message=self._render_message))
        return widgets
