# Copyright (c) 2026 Chrys. All rights reserved.

"""Renderer for the ``todo_write`` tool — inline checklist card."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from rich.text import Text
from textual.widget import Widget
from textual.widgets import Static

from chrys.app.tui.i18n import render_text, widget_localizer
from chrys.app.tui.util.rich_style import rich_style_from_textual_color
from chrys.app.tui.util.todo_format import todo_checklist_renderable, todo_done_count
from chrys.app.tui.widgets.chat.tool_call import ToolCall, ToolCardHeader
from chrys.foundation.i18n import msg
from chrys.foundation.models.todos import parse_todo_items

_TODO_TITLE = msg("tui.tool_card.todo.title", fallback="Todo List")
_TODO_PROGRESS_TITLE = msg(
    "tui.tool_card.todo.progress_title",
    fallback="Todo List ({done}/{total})",
)

if TYPE_CHECKING:
    from rich.table import Table
    from textual.app import ComposeResult

    from chrys.foundation.models.todos import TodoItem


class TodoToolCall(ToolCall):
    """Inline checklist card for whole-list ``todo_write`` updates.

    Inherits the generic card chrome (spinner, status border classes,
    view/copy affordances); replaces the raw-JSON border title with
    ``Todo List (done/total)`` and, on success, renders the submitted
    checklist instead of the result summary line. The raw args stay
    reachable through the header's view/copy affordances, and error or
    rejected results keep the generic text body.
    """

    def _todo_items(self) -> tuple[TodoItem, ...]:
        args = self.args
        if not args and self.args_summary:
            with suppress(Exception):
                parsed = json.loads(self.args_summary)
                if isinstance(parsed, dict):
                    args = parsed
        return parse_todo_items(args.get("todos") if isinstance(args, dict) else None)

    def _title_text(self) -> Text:
        items = self._todo_items()
        if not items:
            return render_text(widget_localizer(self), _TODO_TITLE.bind())
        return render_text(
            widget_localizer(self),
            _TODO_PROGRESS_TITLE.bind(done=todo_done_count(items), total=len(items)),
        )

    def _checklist_renderable(self, items: tuple[TodoItem, ...]) -> Table:
        warning = "yellow"
        with suppress(Exception):
            warning = self.app.theme_variables.get("warning", "yellow")
        return todo_checklist_renderable(
            items,
            in_progress_style=rich_style_from_textual_color(warning, bold=True),
        )

    def compose(self) -> ComposeResult:
        yield ToolCardHeader(self._label_text(), id="tc-label")
        panel = Widget(id="tc-panel")
        panel.border_title = self._title_text()
        yield panel

    def update_args(self, args: dict[str, Any]) -> None:
        """Refresh the displayed arguments after an approval edit."""
        super().update_args(args)
        with suppress(Exception):
            self.query_one("#tc-panel").border_title = self._title_text()

    def set_complete(self, result: str, duration_ms: int = 0, **kwargs: Any) -> None:
        """Render the applied checklist on success; defer to the base otherwise."""
        super().set_complete(result, duration_ms, **kwargs)
        if self.status != "complete":
            return
        items = self._todo_items()
        if not items:
            return
        with suppress(Exception):
            self.query_one("#tc-body", Static).update(self._checklist_renderable(items))
