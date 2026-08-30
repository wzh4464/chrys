# Copyright (c) 2026 Chrys. All rights reserved.

"""ApprovalModeScreen — modal for switching approval mode."""

from __future__ import annotations

import contextlib
import textwrap
from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.containers import VerticalGroup
from textual.content import Content
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from chrys.app.tui.binding_display import CLOSE_BINDING, localized_binding
from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.app.tui.widgets.chrome.app_header import APPROVAL_MODE_MESSAGES
from chrys.foundation.i18n import MessageDef, msg
from chrys.service.approval.policy import ApprovalMode

if TYPE_CHECKING:
    from textual.app import ComposeResult

_DESC_INDENT = "    "


def _wrap_description(desc: str, width: int) -> str:
    """Wrap description text so continuation lines align after '  - '."""
    if not desc:
        return ""
    indent = _DESC_INDENT
    first_prefix = "  - "
    text_width = max(width - len(indent), 20)
    lines = textwrap.wrap(desc, width=text_width)
    if not lines:
        return ""
    result = first_prefix + lines[0]
    for line in lines[1:]:
        result += "\n" + indent + line
    return result


_MODE_MANUAL_DESCRIPTION = msg(
    "tui.approval_mode.description.manual",
    fallback="You approve every tool call manually",
)
MODE_AUTO_DESCRIPTION = msg(
    "tui.approval_mode.description.auto",
    fallback="Auto-approves safe calls, flags suspicious ones",
)
MODE_BYPASS_DESCRIPTION = msg(
    "tui.approval_mode.description.bypass",
    fallback="All tool calls run without approval",
)
_APPROVAL_MODE_TITLE = msg("tui.approval_mode.title", fallback="Approval Mode")

_MODE_DESCRIPTIONS: dict[ApprovalMode, MessageDef] = {
    ApprovalMode.MANUAL: _MODE_MANUAL_DESCRIPTION,
    ApprovalMode.AUTO: MODE_AUTO_DESCRIPTION,
    ApprovalMode.BYPASS: MODE_BYPASS_DESCRIPTION,
}


class ApprovalModeScreen(BaseDialog[ApprovalMode | None]):
    """Modal for switching approval mode.

    Single click selects and dismisses. Current mode is dimmed and disabled.
    Escape cancels.
    """

    DEFAULT_CSS = """
    ApprovalModeScreen {
        align: center middle;
    }
    ApprovalModeScreen > #container {
        width: 60;
        max-width: 90%;
        max-height: 90%;
        height: auto;
        background: $surface;
        border: round $tui-border-primary $border-opacity;
        border-title-align: left;
        border-title-color: $primary;
        padding: 0;
        overflow-x: hidden;
    }
    ApprovalModeScreen > #container > OptionList {
        height: auto;
        max-height: 100%;
        border: none;
        padding: 0 0 0 1;
        scrollbar-size: 1 1;
    }
    """

    BINDINGS: ClassVar[list] = [
        localized_binding("escape", "cancel", CLOSE_BINDING),
    ]

    def __init__(self, current_mode: ApprovalMode) -> None:
        self._current_mode = current_mode
        self._dismissed = False
        super().__init__()

    def compose(self) -> ComposeResult:
        with VerticalGroup(id="container") as container:
            container.border_title = Content.from_text(
                render_str(widget_localizer(self), _APPROVAL_MODE_TITLE.bind()),
                markup=False,
            )
            yield OptionList()

    def on_mount(self) -> None:
        ol = self.query_one(OptionList)
        localizer = widget_localizer(self)
        content_width = ol.size.width or 54
        for i, mode in enumerate(ApprovalMode):
            if i > 0:
                ol.add_option(None)
            label = render_str(localizer, APPROVAL_MODE_MESSAGES[mode].bind())
            desc = _wrap_description(render_str(localizer, _MODE_DESCRIPTIONS[mode].bind()), content_width)
            is_current = mode == self._current_mode
            prefix = "◦ " if is_current else ""
            if is_current:
                label_content = Content.from_text(f"{prefix}{label}", markup=False).stylize("bold #888888")
                desc_content = Content.from_text(f"\n{desc}", markup=False).stylize("#888888") if desc else ""
                content = Content.assemble(label_content, desc_content)
            else:
                label_content = Content.from_text(f"{prefix}{label}", markup=False).stylize("bold")
                desc_content = Content.from_text(f"\n{desc}", markup=False) if desc else ""
                content = Content.assemble(label_content, desc_content)
            ol.add_option(Option(content, id=mode.value, disabled=is_current))

    @on(OptionList.OptionSelected)
    def _on_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            event.stop()
            self._safe_dismiss(ApprovalMode(event.option.id))

    def action_cancel(self) -> None:
        self._safe_dismiss(None)

    def _dismiss_clicked_outside(self) -> None:
        self._safe_dismiss(None)

    def _safe_dismiss(self, result: ApprovalMode | None) -> None:
        if self._dismissed:
            return
        # Local guard gates side effects; the mixin guard only protects Textual's dismiss.
        self._dismissed = True
        with contextlib.suppress(Exception):
            self.query_one("#container").display = False
        self.dismiss(result)
