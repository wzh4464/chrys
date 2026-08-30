# Copyright (c) 2026 Chrys. All rights reserved.

"""EnhancedInput - terminal-friendly single-line Input with Ctrl+A and OS clipboard paste."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.actions import SkipAction
from textual.widgets import Input

from chrys.app.tui.clipboard import copy_text_to_clipboards, paste_text_from_clipboards

if TYPE_CHECKING:
    from textual import events


class EnhancedInput(Input):
    """``Input`` variant with conventional select-all and clipboard behavior.

    Textual's built-in ``Input`` maps ``Ctrl+A`` to ``home`` and reserves
    select-all for ``Ctrl+Shift+A``.  Chrys uses ``Home`` for cursor-start,
    so user-editable single-line fields follow the common terminal/editor
    convention and select the field contents with ``Ctrl+A``. Native paste
    actions prefer the current OS clipboard over Textual's process-local cache;
    browser-host sessions stay scoped to their own browser clipboard.
    """

    async def _on_key(self, event: events.Key) -> None:
        if event.key != "ctrl+a":
            return
        event.stop()
        event.prevent_default()
        self.select_all()

    def action_copy(self) -> None:
        """Copy the selection to Textual and host OS clipboards."""
        selected = self.selected_text
        if not selected:
            raise SkipAction
        copy_text_to_clipboards(self.app, selected)

    def action_cut(self) -> None:
        """Cut the selection after synchronizing both clipboards."""
        selected = self.selected_text
        if not selected:
            return
        copy_text_to_clipboards(self.app, selected)
        self.delete_selection()

    def action_paste(self) -> None:
        """Paste the first line from the freshest frontend-safe clipboard."""
        text = paste_text_from_clipboards(self.app)
        if not text:
            return
        line = text.splitlines()[0]
        start, end = self.selection
        self.replace(line, start, end)
