# Copyright (c) 2026 Chrys. All rights reserved.

"""ConversationToc - Conversation table of contents widget."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.text import Text
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static, Tree

from chrys.app.tui.i18n import render_str
from chrys.app.tui.widgets.sidebar.empty_state import SidebarEmptyStateLabel
from chrys.app.tui.widgets.sidebar.toc_types import _SUMMARY_MAX_LENGTH, TocItem, summarize_prompt
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from chrys.app.tui.i18n import LocaleController


__all__ = ["ConversationToc", "TocItem", "summarize_prompt"]

# Rendered labels include the numeric turn prefix ("12. "), so keep the
# display cap above the default summary cap instead of silently re-clipping it.
_MAX_LABEL = _SUMMARY_MAX_LENGTH + 8
_TREE_TAB_SIZE = 8
_TOC_EMPTY = msg("tui.sidebar.toc.empty", fallback="Your conversation will appear here")


def _tree_label_text(text: str) -> str:
    """Return text safe for Textual Tree's Rich render path."""
    return text.expandtabs(_TREE_TAB_SIZE)


class ConversationToc(Widget, can_focus=False):
    """Conversation table of contents widget."""

    DEFAULT_CSS = """
    ConversationToc {
        height: 1fr;
        padding: 0 0 0 1;
    }
    ConversationToc > Tree {
        background: transparent;
        padding: 0;
        overflow-x: hidden;
        scrollbar-size-vertical: 1;
    }
    ConversationToc Tree > .tree--cursor {
        background: $primary 30%;
    }
    ConversationToc Tree:focus > .tree--cursor {
        background: $primary 50%;
    }
    ConversationToc Tree > .tree--guides {
        color: $primary 40%;
    }
    ConversationToc > .toc-empty {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        color: $text-muted;
    }
    """

    class TurnSelected(Message):
        """User selected a conversation turn."""

        def __init__(self, turn_id: str) -> None:
            super().__init__()
            self.turn_id = turn_id

    def __init__(self, *, locale_controller: LocaleController | None = None) -> None:
        super().__init__()
        self._locale_controller = locale_controller
        self._items: list[TocItem] = []
        self._tree: Tree[str] | None = None
        self._empty_label: Static = SidebarEmptyStateLabel("", classes="toc-empty")

    def compose(self) -> ComposeResult:
        self._tree = Tree[str]("TOC", id="toc-tree")
        self._tree.show_root = False
        self._tree.show_guides = True
        self._tree.display = False
        yield self._tree
        self._empty_label = SidebarEmptyStateLabel(Text(self._render_message(_TOC_EMPTY.bind())), classes="toc-empty")
        yield self._empty_label

    def refresh_localization(self) -> None:
        """Retranslate the current empty-state text without rebuilding the tree."""
        if self.is_mounted:
            self._empty_label.update(Text(self._render_message(_TOC_EMPTY.bind())))

    def _render_message(self, reference: MessageRef) -> str:
        controller = self._locale_controller
        if controller is None:
            return format_message(reference)
        return render_str(controller.localizer, reference)

    def update_items(self, items: list[TocItem]) -> None:
        self._items = items
        self._rebuild_tree()

    def _rebuild_tree(self) -> None:
        if self._tree is None:
            return
        self._tree.clear()
        self._empty_label.display = not self._items
        self._tree.display = bool(self._items)
        for i, item in enumerate(self._items, start=1):
            raw = _tree_label_text(f"{i}. {item.summary}")
            if len(raw) > _MAX_LABEL:
                raw = raw[: _MAX_LABEL - 3] + "..."
            label = Text(raw, style="dim") if item.compressed else Text(raw)
            if item.children:
                node = self._tree.root.add(label, data=item.turn_id)
                node.expand()
                node.allow_expand = False
                for child in item.children:
                    raw_child = _tree_label_text(child.summary)
                    if len(raw_child) > _MAX_LABEL - 2:
                        raw_child = raw_child[: _MAX_LABEL - 5] + "..."
                    child_label = Text(raw_child, style="dim") if child.compressed else Text(raw_child)
                    node.add_leaf(child_label, data=child.turn_id)
            else:
                self._tree.root.add_leaf(label, data=item.turn_id)

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        if event.node.data is not None:
            self.post_message(self.TurnSelected(event.node.data))
