# Copyright (c) 2026 Chrys. All rights reserved.

"""ThemesScreen — modal for browsing and selecting themes."""

from __future__ import annotations

from time import monotonic
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import on
from textual.containers import VerticalGroup
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from chrys.app.tui.binding_display import CLOSE_BINDING, localized_binding
from chrys.app.tui.i18n import render_str, widget_localizer
from chrys.app.tui.screens.dialogs.base import BaseDialog
from chrys.foundation.i18n import msg

if TYPE_CHECKING:
    from textual.app import ComposeResult

_DOUBLE_CLICK_THRESHOLD = 0.4
_THEMES = msg("tui.theme_picker.title", fallback="Themes")


class ThemesScreen(BaseDialog[str | None]):
    """Modal for selecting a theme.

    Single click previews. Double-click/Enter applies and dismisses.
    Escape reverts to the original theme.
    """

    DEFAULT_CSS = """
    ThemesScreen {
        align: center middle;
    }
    ThemesScreen > #container {
        width: 48;
        max-width: 90%;
        max-height: 90%;
        height: auto;
        background: $surface;
        border: round $tui-border-primary $border-opacity;
        border-title-align: left;
        border-title-color: $tui-border-title-primary;
        padding: 0;
        overflow-x: hidden;
    }
    ThemesScreen > #container > OptionList {
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

    def __init__(self, current_theme: str) -> None:
        self._original_theme = current_theme
        self._last_selected: str | None = None
        self._last_selected_time: float = 0
        self._dismissed = False
        super().__init__()

    def compose(self) -> ComposeResult:
        with VerticalGroup(id="container") as container:
            container.border_title = render_str(widget_localizer(self), _THEMES.bind())
            yield OptionList()

    def on_mount(self) -> None:
        ol = self.query_one(OptionList)
        themes = sorted(self.app.available_themes)
        for i, name in enumerate(themes):
            ol.add_option(Option(Text(name), id=name))
            if name == self._original_theme:
                ol.highlighted = i

    @on(OptionList.OptionHighlighted)
    def _on_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Preview theme on highlight."""
        if event.option.id:
            self.app.theme = event.option.id

    @on(OptionList.OptionSelected)
    def _on_selected(self, event: OptionList.OptionSelected) -> None:
        """Single click previews; double-click/Enter applies and dismisses."""
        if not event.option.id:
            return
        now = monotonic()
        if self._last_selected == event.option.id and (now - self._last_selected_time) < _DOUBLE_CLICK_THRESHOLD:
            event.stop()
            self._safe_dismiss(event.option.id)
        else:
            self._last_selected = event.option.id
            self._last_selected_time = now

    def on_key(self, event) -> None:
        """Enter key applies current theme and dismisses."""
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self._safe_dismiss(self.app.theme)

    def action_cancel(self) -> None:
        self._safe_dismiss(None, restore_original=True)

    def _dismiss_clicked_outside(self) -> None:
        # Backdrop click is a cancel: restore the previewed theme and close.
        self.action_cancel()

    def _safe_dismiss(self, result: str | None, *, restore_original: bool = False) -> None:
        if self._dismissed:
            return
        # Local guard gates side effects; the mixin guard only protects Textual's dismiss.
        self._dismissed = True
        if restore_original:
            self.app.theme = self._original_theme
        self.dismiss(result)
