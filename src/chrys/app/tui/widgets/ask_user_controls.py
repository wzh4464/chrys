# Copyright (c) 2026 Chrys. All rights reserved.

"""Reusable controls for answering ``ask_user`` prompts."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual import events, on
from textual.containers import HorizontalGroup, VerticalGroup
from textual.content import Content
from textual.css.scalar import Scalar, Unit
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, TextArea

from chrys.app.tui.i18n import render_str, render_text, widget_localizer
from chrys.app.tui.widgets import EnhancedTextArea
from chrys.foundation.i18n import msg

if TYPE_CHECKING:
    from textual.app import ComposeResult


ASK_USER_INTERACTIVE_CLASS = "askuser-interactive"
# Keep these in sync with the #askuser-input CSS in the dialog and inline renderer.
ASK_USER_INPUT_MIN_HEIGHT = 3
ASK_USER_INPUT_MAX_HEIGHT = 7
_ASK_USER_TEXTAREA_FRAME_ROWS = 2
_ASK_USER_LAYOUT_REFRESH_ANCESTORS = 8
_ASK_USER_OPTION_MIN_HEIGHT = 3
_ASK_USER_INVISIBLE_OPTION_CHARS = dict.fromkeys((0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF))

_SUBMIT = msg("tui.ask_user.button.submit", fallback="Submit")
_ANSWER_INLINE = msg("tui.ask_user.button.answer_inline", fallback="Answer Inline")
_CUSTOM_RESPONSE_PLACEHOLDER = msg(
    "tui.ask_user.placeholder.custom_response",
    fallback="Type a custom response...",
)


def normalize_ask_user_options(options: list[str] | None) -> list[str]:
    """Return user-visible ask_user options, dropping blank entries."""
    normalized: list[str] = []
    for option in options or []:
        if not isinstance(option, str):
            continue
        visible = option.translate(_ASK_USER_INVISIBLE_OPTION_CHARS).strip()
        if visible:
            normalized.append(visible)
    return normalized


class AskUserSubmitted(Message):
    """User submitted a response to an ask_user request."""

    def __init__(self, request_id: str, text: str) -> None:
        super().__init__()
        self.request_id = request_id
        self.text = text


class AskUserInlineRequested(Message):
    """User requested to answer the ask_user prompt inline."""

    def __init__(self, request_id: str, draft_text: str = "") -> None:
        super().__init__()
        self.request_id = request_id
        self.draft_text = draft_text


class AskUserResponseResized(Message):
    """Response input height changed and parent layout should be remeasured."""

    def __init__(self, input_height: int) -> None:
        super().__init__()
        self.input_height = input_height


class _AskUserTextArea(EnhancedTextArea):
    """Free-text response input: Enter/Ctrl+J inserts a newline."""

    def __init__(self, *args: Any, defer_layout_to_parent: bool = False, **kwargs: Any) -> None:
        # Terminal IMEs anchor their candidate window to the hardware cursor.
        # A blinking TextArea cursor schedules periodic repaints even while the
        # user is composing, which lets Windows IMEs sample a transient cursor
        # position. Keep the cursor visible without the blink timer instead.
        super().__init__(*args, **kwargs)
        self.cursor_blink = False
        self._defer_layout_to_parent = defer_layout_to_parent
        self._last_reported_height = 0
        self._last_wrapped_width: int | None = None

    def on_mount(self) -> None:
        self.resize_to_content(rewrap=True)

    def _on_resize(self, event: events.Resize) -> None:
        # This path owns width-driven rewrapping; prevent Textual from also
        # dispatching TextArea._on_resize in the MRO handler walk.
        event.prevent_default()
        wrap_width = self.wrap_width
        if wrap_width == self._last_wrapped_width:
            return
        self.resize_to_content(rewrap=True)

    async def _on_key(self, event: events.Key) -> None:
        if event.key in ("enter", "ctrl+j", "shift+enter"):
            event.stop()
            event.prevent_default()
            start, end = self.selection
            if self._replace_via_keyboard("\n", start, end):
                self.scroll_cursor_visible()
            return
        await super()._on_key(event)

    def resize_to_content(self, *, rewrap: bool = False) -> int:
        """Grow the textarea viewport with its wrapped content, up to a cap."""
        # TextArea.edit() has already wrapped the edited range and refreshed
        # virtual_size before its Changed message is dispatched. Repeating the
        # full-document rewrap here creates an extra layout/terminal repaint on
        # every composition update. A caller that observes a new wrap width
        # self-heals even if its own Resize message is still queued.
        wrap_width = self.wrap_width
        if rewrap or wrap_width != self._last_wrapped_width:
            self._last_wrapped_width = wrap_width
            self._rewrap_and_refresh_virtual_size()
        height = min(
            max(self.virtual_size.height + _ASK_USER_TEXTAREA_FRAME_ROWS, ASK_USER_INPUT_MIN_HEIGHT),
            ASK_USER_INPUT_MAX_HEIGHT,
        )
        if height == self._last_reported_height:
            return height
        self._last_reported_height = height
        if self._defer_layout_to_parent:
            # The inline renderer consumes AskUserResponseResized and applies
            # all dependent heights before requesting one ChatPanel layout.
            # Installing the parsed rule directly keeps this first change from
            # racing that parent message with an incomplete global layout.
            self.styles.set_rule("height", Scalar(float(height), Unit.CELLS, Unit.WIDTH))
        else:
            self.styles.height = height
            _refresh_layout_chain(self)
        return height

    def cached_content_height(self) -> int | None:
        """Return the measured height when it still matches the current wrap width."""
        if not self._last_reported_height or self.wrap_width != self._last_wrapped_width:
            return None
        return self._last_reported_height


class AskUserOptions(VerticalGroup):
    """Preset response buttons for an ask_user request."""

    def __init__(self, request_id: str, options: list[str] | None = None) -> None:
        super().__init__(id="askuser-options", classes=ASK_USER_INTERACTIVE_CLASS)
        self._request_id = request_id
        self._options = normalize_ask_user_options(options)
        self._locked = False
        self._last_width = 0
        self._sync_pending = False

    def on_mount(self) -> None:
        self.call_after_refresh(self.sync_option_heights)

    def on_resize(self, event: events.Resize) -> None:
        if event.size.width == self._last_width:
            return
        # Record and release on every width change, even mid-flight: a resize
        # dropped while a sync is already scheduled would leave the buttons
        # frozen at heights measured for a width that no longer exists, with
        # nothing left to notice. One scheduled sync is still enough — it
        # measures whatever layout the last release produced.
        self._last_width = event.size.width
        self.release_option_heights()
        if self._sync_pending:
            return
        self._sync_pending = True
        self.call_after_refresh(self.sync_option_heights)

    def compose(self) -> ComposeResult:
        for i, opt in enumerate(self._options):
            classes = "askuser-option-last" if i == len(self._options) - 1 else ""
            yield Button(Text(opt), id=f"askuser-opt-{i}", variant="primary", flat=True, classes=classes)

    def sync_option_heights(self) -> int:
        """Freeze option heights after Textual has measured wrapped labels."""
        self._sync_pending = False
        total_height = 0
        height_changes: list[tuple[Button, int]] = []
        for button in self.query(Button):
            height = max(_ASK_USER_OPTION_MIN_HEIGHT, button.virtual_size.height)
            if str(button.styles.height) != str(height) or str(button.styles.min_height) != str(height):
                height_changes.append((button, height))
            total_height += height
        container_changed = bool(total_height) and (
            str(self.styles.height) != str(total_height) or str(self.styles.min_height) != str(total_height)
        )
        if height_changes or container_changed:
            for button, height in height_changes:
                button.styles.height = height
                button.styles.min_height = height
            self.styles.height = total_height
            self.styles.min_height = total_height
            _refresh_layout_chain(self)
        return total_height

    def release_option_heights(self) -> None:
        """Return the container and its buttons to auto height for remeasure."""
        self.styles.height = "auto"
        self.styles.min_height = 0
        for button in self.query(Button):
            button.styles.height = "auto"
            button.styles.min_height = _ASK_USER_OPTION_MIN_HEIGHT
        self.refresh(layout=True)

    def disable_controls(self) -> None:
        """Disable controls after a response path has been chosen."""
        self._locked = True
        for button in self.query(Button):
            button.disabled = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if not btn_id.startswith("askuser-opt-"):
            return
        event.stop()
        if self._locked:
            return
        self.disable_controls()
        label = event.button.label
        label_text = label.plain if isinstance(label, (Content, Text)) else label
        self.post_message(AskUserSubmitted(self._request_id, label_text))


class AskUserResponseFooter(VerticalGroup):
    """Custom response input and submit/inline actions for ask_user."""

    def __init__(
        self,
        request_id: str,
        *,
        allow_inline: bool = False,
        initial_response: str = "",
        defer_layout_to_parent: bool = False,
    ) -> None:
        super().__init__(id="askuser-footer", classes=ASK_USER_INTERACTIVE_CLASS)
        self._request_id = request_id
        self._allow_inline = allow_inline
        self._initial_response = initial_response
        self._defer_layout_to_parent = defer_layout_to_parent
        self._locked = False
        self._last_input_height = 0
        self._last_footer_height = 0

    def on_mount(self) -> None:
        self.call_after_refresh(self._capture_initial_input_height)

    def _capture_initial_input_height(self) -> None:
        with suppress(Exception):
            self._last_input_height = self.measure_input_height()

    def on_resize(self, event: events.Resize) -> None:
        if event.size.height == self._last_footer_height:
            return
        self._last_footer_height = event.size.height
        with suppress(Exception):
            text_area = self.query_one("#askuser-input", _AskUserTextArea)
            # The textarea owns wrapping and height calculation. Footer resize
            # is feedback from that layout change, so report its cached result
            # without re-running the edit-time resize path unless the available
            # wrap width genuinely changed.
            input_height = text_area.cached_content_height()
            if input_height is None:
                input_height = text_area.resize_to_content(rewrap=True)
            self.notify_input_resized(input_height)

    def compose(self) -> ComposeResult:
        text_area = _AskUserTextArea(
            text=self._initial_response,
            id="askuser-input",
            compact=True,
            soft_wrap=True,
            show_line_numbers=False,
            defer_layout_to_parent=self._defer_layout_to_parent,
        )
        text_area.placeholder = render_str(widget_localizer(self), _CUSTOM_RESPONSE_PLACEHOLDER.bind())
        yield text_area
        with HorizontalGroup(id="askuser-buttons"):
            localizer = widget_localizer(self)
            submit = Button(
                render_text(localizer, _SUBMIT.bind()),
                id="askuser-submit",
                variant="success",
                flat=True,
            )
            submit.disabled = not bool(self._initial_response.strip())
            yield submit
            if self._allow_inline:
                yield Button(
                    render_text(localizer, _ANSWER_INLINE.bind()),
                    id="askuser-inline",
                    variant="warning",
                    flat=True,
                )

    @on(TextArea.Changed, "#askuser-input")
    def _on_response_changed(self, event: TextArea.Changed) -> None:
        # Single owner for edit-time layout updates. TextArea.edit() posts
        # this message, so doing the same work in _AskUserTextArea.edit()
        # would double rewrap/reflow during IME composition and typing.
        self.sync_response_state(event.text_area)

    def sync_response_state(self, text_area: TextArea | None = None) -> None:
        """Synchronize layout and submit availability with the current response text."""
        if text_area is None:
            text_area = self.query_one("#askuser-input", _AskUserTextArea)
        if isinstance(text_area, _AskUserTextArea):
            input_height = text_area.resize_to_content()
            self.notify_input_resized(input_height)
        if not self._locked:
            self._refresh_submit_disabled(text_area.text)

    def measure_input_height(self) -> int:
        """Measure the response input after TextArea has applied its edit."""
        return self.query_one("#askuser-input", _AskUserTextArea).resize_to_content()

    def notify_input_resized(self, input_height: int) -> None:
        """Notify parent renderers when the input height actually changes."""
        if input_height == self._last_input_height:
            return
        self._last_input_height = input_height
        self.post_message(AskUserResponseResized(input_height))

    def focus_input(self) -> None:
        """Focus the custom response input if it is mounted."""
        with suppress(Exception):
            self.query_one("#askuser-input", _AskUserTextArea).focus()

    def draft_text(self) -> str:
        """Return the current custom-response text without submitting it."""
        with suppress(Exception):
            return self.query_one("#askuser-input", _AskUserTextArea).text
        return ""

    def disable_controls(self) -> None:
        """Disable controls after a response path has been chosen."""
        self._locked = True
        with suppress(Exception):
            self.query_one("#askuser-input", _AskUserTextArea).disabled = True
        for button in self.query(Button):
            button.disabled = True

    def _refresh_submit_disabled(self, raw_response: str) -> None:
        self.query_one("#askuser-submit", Button).disabled = not bool(raw_response.strip())

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id not in {"askuser-submit", "askuser-inline"}:
            return
        event.stop()
        if self._locked:
            return
        if btn_id == "askuser-inline":
            draft = self.draft_text()
            self.disable_controls()
            self.post_message(AskUserInlineRequested(self._request_id, draft))
            return
        text = self.draft_text().strip()
        if not text:
            self._refresh_submit_disabled(text)
            return
        self.disable_controls()
        self.post_message(AskUserSubmitted(self._request_id, text))


def _refresh_layout_chain(widget: Widget) -> None:
    """Refresh the highest relevant ancestor once after an auto-height change."""
    # TextArea virtual-size changes don't always invalidate auto-height containers
    # above the immediate parent. One refresh at the top of the current
    # chat/dialog nesting depth covers the descendants without scheduling the
    # same layout once per ancestor.
    target = widget
    for _ in range(_ASK_USER_LAYOUT_REFRESH_ANCESTORS - 1):
        parent = target.parent
        if not isinstance(parent, Widget):
            break
        target = parent
    target.refresh(layout=True)
