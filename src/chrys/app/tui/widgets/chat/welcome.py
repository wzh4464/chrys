# Copyright (c) 2026 Chrys. All rights reserved.

"""WelcomeWidget — full-panel background with static logo."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.segment import Segment
from rich.style import Style
from textual.widget import Widget

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult

from chrys.app.tui.util.logo import TUI_LOGO

_LOGO_LINES = TUI_LOGO.strip("\n").split("\n")
_LOGO_W = max(len(line) for line in _LOGO_LINES)


class _WelcomeRenderable:
    """Rich renderable that fills the area with centered logo + info."""

    def __init__(
        self,
        profile: str,
        cwd: str,
        width: int,
        height: int,
        logo_color: str = "white",
    ) -> None:
        self.profile = profile
        self.cwd = cwd
        self.width = width
        self.height = height
        self.logo_color = logo_color

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = self.width
        height = self.height
        if width <= 0 or height <= 0:
            return

        logo_style = Style(color=self.logo_color, bold=True)
        bold_style = Style(bold=True)
        dim_style = Style(dim=True)
        nl = Segment.line()
        blank_row = Segment(" " * width)

        logo_h = len(_LOGO_LINES)

        info_lines: list[tuple[str, Style]] = []
        if self.profile:
            info_lines.append((self.profile, bold_style))
        if self.cwd:
            info_lines.append((self.cwd, dim_style))

        content_h = logo_h + (1 + len(info_lines) if info_lines else 0)

        top = max(0, (height - content_h) // 2)
        left = max(0, (width - _LOGO_W) // 2)

        for y in range(height):
            content_y = y - top

            if content_y < 0 or content_y >= content_h:
                yield blank_row
                yield nl
                continue

            if content_y < logo_h:
                logo_line = _LOGO_LINES[content_y]
                line = " " * left + logo_line
                if len(line) < width:
                    line += " " * (width - len(line))
                else:
                    line = line[:width]
                for ch in line:
                    if ch != " ":
                        yield Segment(ch, logo_style)
                    else:
                        yield Segment(" ")
                yield nl
                continue

            info_idx = content_y - logo_h - 1
            if info_idx < 0:
                # Spacer row between logo and info
                yield blank_row
                yield nl
                continue

            if info_idx < len(info_lines):
                text, style = info_lines[info_idx]
                text_left = max(0, (width - len(text)) // 2)
                line = " " * text_left + text
                if len(line) < width:
                    line += " " * (width - len(line))
                else:
                    line = line[:width]
                for ch in line:
                    if ch != " ":
                        yield Segment(ch, style)
                    else:
                        yield Segment(" ")
                yield nl
                continue

            yield blank_row
            yield nl


class WelcomeWidget(Widget):
    """Fills the entire panel with centered static logo + info."""

    DEFAULT_CSS = """
    WelcomeWidget {
        width: 100%;
        height: 100%;
        color: $foreground;
    }
    """

    def __init__(self, profile: str = "", cwd: str = "") -> None:
        self._profile = profile
        self._cwd = cwd
        super().__init__()

    def update_info(self, profile: str = "", cwd: str = "") -> None:
        """Update displayed profile/cwd and re-render."""
        if profile:
            self._profile = profile
        if cwd:
            self._cwd = cwd
        self.refresh()

    def render(self) -> _WelcomeRenderable:
        # Resolve theme foreground color so the logo adapts to theme changes.
        foreground = self.rich_style.color
        logo_color = foreground.name if foreground and foreground.name else "white"
        return _WelcomeRenderable(
            self._profile,
            self._cwd,
            self.size.width,
            self.size.height,
            logo_color=logo_color,
        )
