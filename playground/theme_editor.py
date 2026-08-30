# Copyright (c) 2026 Chrys. All rights reserved.

"""Playground: live theme editor for chrys.

Launches the full chrys TUI with a chrys-aware theme editor docked on
the left.  Extends ``textual_theme_editor`` with the chrys-specific
``theme.variables`` (border, scrollbar, footer, *-muted, etc.),
per-row resets, and an export modal that emits paste-ready Python.

Editor behavior by active theme:

- Built-in ANSI themes (``ansi-dark`` / ``ansi-light``): editing disabled.
- Non-chrys RGB themes: truecolor RGB picker.
- chrys: xterm-256 picker only.
- chrys-ansi: ANSI-token picker + xterm-256 picker.

The four surface slots (background / surface / panel / boost) disable
the transparent option — see ``_XTERM_256_ROWS_BASE`` for the rationale.

Usage:
    uv run python playground/theme_editor.py

    # Force 256-color mode (simulates SSH from Windows Terminal to Linux):
    env -u COLORTERM TERM=xterm-256color uv run python playground/theme_editor.py
"""

from __future__ import annotations

import contextlib
import inspect
import re
from collections.abc import Callable, Iterable
from functools import partial
from itertools import batched
from pathlib import Path
from typing import ClassVar

from rich.color import EIGHT_BIT_PALETTE
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.color import Color
from textual.containers import HorizontalGroup, VerticalGroup, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.screen import ModalScreen
from textual.strip import Strip
from textual.theme import BUILTIN_THEMES, Theme
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Tab, Tabs, TextArea
from textual_colorpicker import ColorPicker
from textual_theme_editor import ThemeEditor

from chrys.app.tui import app as _chrys_app
from chrys.app.tui.screens.main import MainScreen as _ChrysMainScreen
from chrys.app.tui.theme import (
    CHRYS_ANSI_THEME,
    CHRYS_COLOR_NAMES,
    CHRYS_FAMILY_THEMES,
    CHRYS_ONLY_VARIABLES,
    CHRYS_THEME,
    CHRYS_THEMES,
)

_CHRYS_ANSI_NAME = "chrys-ansi"
_SURFACE_COLOR_NAMES = ("background", "surface", "panel", "boost")
_THEME_COLOR_NAMES = (*CHRYS_COLOR_NAMES, *_SURFACE_COLOR_NAMES)
_CHRYS_ANSI_BASE_COLOR_VALUES = {name: getattr(CHRYS_ANSI_THEME, name) for name in CHRYS_COLOR_NAMES}
_CHRYS_BASE_VARIABLE_VALUES = dict(CHRYS_THEME.variables)
_CHRYS_ANSI_BASE_VARIABLE_VALUES = dict(CHRYS_ANSI_THEME.variables)
# Leading letter required so the generated ``<SLUG>_THEME`` is a valid identifier.
_THEME_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

# Variables that textual's ``ColorSystem.generate()`` derives for every
# theme (RGB and ANSI alike) and that chrys / chrys-ansi may or may not
# override.  Listed explicitly so the sidebar shows them as editable
# rows on all themes, with ``(not set)`` for slots the active theme
# leaves to textual's algorithmic default.
_TEXTUAL_EXTRA_VARIABLE_NAMES = (
    "input-cursor-background",
    "input-cursor-foreground",
    "input-selection-background",
    "foreground-muted",
    "text-muted",
    "text-disabled",
    "screen-selection-background",
)

# Truly ANSI-only slots: they only have meaningful values when
# ``Theme.ansi=True``; on RGB themes textual resolves both to
# ``"transparent"`` placeholders.  Shown on every theme anyway — hiding
# two rows on theme switch isn't worth the scrollbar reflow.
_ANSI_VARIABLE_NAMES: tuple[str, ...] = ("ansi-background", "ansi-foreground")

# Every chrys-ansi entry that diverges from base chrys (ansi-only +
# shared-key overrides), in source order from ``_CHRYS_ANSI_VARIABLES``
# in src/chrys/app/tui/theme.py.  Explicit instead of computed because
# ``**_CHRYS_VARIABLES`` spread pins shared keys to base-chrys positions
# in the live dict, losing the source literal order.  A test enforces
# set-equivalence with theme.py so additions/removals can't drift.
_CHRYS_ANSI_OVERRIDE_NAMES: tuple[str, ...] = (
    "ansi-background",
    "ansi-foreground",
    "button-color-foreground",
    "foreground-muted",
    "text-muted",
    "text-disabled",
    "markdown-h1-color",
    "markdown-h2-color",
    "markdown-h3-color",
    "markdown-h4-color",
    "markdown-h5-color",
    "markdown-h6-color",
    "input-selection-background",
    "screen-selection-background",
)

# Source-form metadata for the chrys-ansi update export — mirrors the
# layout of ``_CHRYS_ANSI_VARIABLES`` in src/chrys/app/tui/theme.py so the
# round-trip is byte-identical when the theme is unmodified.
#
# Alias map: keys whose source-form value is ``_CHRYS_COLORS["<key>"]``.
# Export only emits the alias when the current value still matches that
# named color — user edits that diverge fall back to a literal.
_CHRYS_ANSI_ALIAS_KEYS: dict[str, str] = {
    "markdown-h1-color": "primary",
    "markdown-h2-color": "primary",
    "markdown-h3-color": "primary",
    "markdown-h4-color": "foreground",
    "markdown-h5-color": "foreground",
}
# Inline comment block injected immediately before the named key.
_CHRYS_ANSI_INLINE_COMMENTS: dict[str, tuple[str, ...]] = {
    "markdown-h6-color": (
        "# Nearest 256-color approximations of Chrys alpha colors over _DARK_GRAY.",
        "# Keep these covered by tests/app/tui/behaviors/test_app.py so chrys-ansi remains close",
        "# to chrys without forcing the truecolor theme onto flat approximations.",
    ),
}

_CANONICAL_VARIABLE_NAMES = tuple(
    dict.fromkeys(
        (
            *CHRYS_THEME.variables,
            *_ANSI_VARIABLE_NAMES,
            *_TEXTUAL_EXTRA_VARIABLE_NAMES,
        )
    )
)


_PREFIX_BUCKETS = ("border", "scrollbar", "block", "input", "footer")
_BUCKET_ORDER = (*_PREFIX_BUCKETS, "muted", "chrys", "misc", "ansi")
_RESET_BUTTON_LABEL = "↺"
_ANSI_COLOR_TOKENS = (
    "ansi_default",
    "ansi_black",
    "ansi_red",
    "ansi_green",
    "ansi_yellow",
    "ansi_blue",
    "ansi_magenta",
    "ansi_cyan",
    "ansi_white",
    "ansi_bright_black",
    "ansi_bright_red",
    "ansi_bright_green",
    "ansi_bright_yellow",
    "ansi_bright_blue",
    "ansi_bright_magenta",
    "ansi_bright_cyan",
    "ansi_bright_white",
)


def _group_variables(var_names: list[str]) -> list[tuple[str, list[str]]]:
    """Group variable names into ``_BUCKET_ORDER`` buckets for sidebar layout."""
    groups: dict[str, list[str]] = {}
    for name in var_names:
        if name in CHRYS_ONLY_VARIABLES:
            key = "chrys"
        elif name in _ANSI_VARIABLE_NAMES:
            key = "ansi"
        elif name.endswith("-muted"):
            key = "muted"
        else:
            key = next((p for p in _PREFIX_BUCKETS if name.startswith(p)), "misc")
        groups.setdefault(key, []).append(name)
    return [(k, groups[k]) for k in _BUCKET_ORDER if k in groups]


def _try_parse_color(value: str) -> Color | None:
    """Parse ``value`` as a Color, but reject ``transparent``.

    Picker's ``HexInput`` only takes 6-digit hex, so we reject alpha
    even though ``Color.parse`` accepts it.
    """
    if value.strip().lower() == "transparent":
        return None
    with contextlib.suppress(Exception):
        return Color.parse(value)
    return None


def _ansi_base_token(value: str | None) -> str | None:
    """Return the supported ANSI token at the front of ``value``."""
    if not value:
        return None
    parts = value.strip().split(maxsplit=1)
    return parts[0] if parts and parts[0] in _ANSI_COLOR_TOKENS else None


def _ansi_display_label(value: str) -> str | None:
    """Shorten ANSI tokens so they fit in the swatch column."""
    token = _ansi_base_token(value)
    if token is None:
        return None
    suffix = value.strip()[len(token) :].strip()
    if token == "ansi_default":
        label = "default"
    elif token.startswith("ansi_bright_"):
        label = f"br.{token.removeprefix('ansi_bright_')}"
    else:
        label = token.removeprefix("ansi_")
    return f"{label} {suffix}" if suffix else label


def _palette_color(index: int) -> Color:
    """Return xterm-256 palette index as a Textual ``Color``."""
    triplet = EIGHT_BIT_PALETTE[index]
    return Color(triplet.red, triplet.green, triplet.blue)


def _palette_sort_key(index: int) -> tuple[float, float, float]:
    """Group the 216-color cube perceptually instead of by xterm index."""
    h, s, v = _palette_color(index).hsv
    return h, -s, -v


# Sentinel for the dedicated "transparent" cell rendered alongside the
# 256 palette entries.  Negative so it can't collide with any real
# palette index across the ``_XTERM_256_POSITION_BY_INDEX*`` maps.
_TRANSPARENT_SENTINEL = -1

# Three xterm-256 layouts share a base prefix.  The picker picks one in
# ``_Xterm256PalettePicker.__init__`` based on whether transparent is a
# valid choice for the slot being edited:
#   - ``_XTERM_256_ROWS``         : full layout w/ selectable Transparent cell.
#   - ``_XTERM_256_ROWS_BASE``    : no Transparent section at all.
#   - ``_XTERM_256_ROWS_WITH_HINT``: Transparent header + empty placeholder
#                                    row (``(None, ())``); ``render_line``
#                                    paints the disabled hint there.
# The four surface slots (background / surface / panel / boost) use one of
# the two no-transparent layouts because writing ``"transparent"`` to
# ``theme.surface`` crashes Textual's TextArea theme apply — rich's color
# parser rejects the keyword that Textual itself accepts.
_XTERM_ANSI_SECTION = tuple(range(16))
_XTERM_CUBE_SECTION = tuple(sorted(range(16, 232), key=_palette_sort_key))
_XTERM_GRAYSCALE_SECTION = tuple(range(232, 256))
_XTERM_256_ROWS_BASE: tuple[tuple[str | None, tuple[int, ...]], ...] = (
    ("ANSI 16", ()),
    (None, _XTERM_ANSI_SECTION),
    ("", ()),
    ("Color cube (hue grouped)", ()),
    *((None, tuple(row)) for row in batched(_XTERM_CUBE_SECTION, 18, strict=True)),
    ("", ()),
    ("Grayscale", ()),
    *((None, tuple(row)) for row in batched(_XTERM_GRAYSCALE_SECTION, 12, strict=True)),
)
_XTERM_256_ROWS: tuple[tuple[str | None, tuple[int, ...]], ...] = (
    *_XTERM_256_ROWS_BASE,
    ("", ()),
    ("Transparent", ()),
    (None, (_TRANSPARENT_SENTINEL,)),
)
# Placeholder row uses ``indices=()`` so ``_selectable_rows`` excludes it.
_XTERM_256_ROWS_WITH_HINT: tuple[tuple[str | None, tuple[int, ...]], ...] = (
    *_XTERM_256_ROWS_BASE,
    ("", ()),
    ("Transparent", ()),
    (None, ()),
)


def _build_position_map(rows: tuple[tuple[str | None, tuple[int, ...]], ...]) -> dict[int, tuple[int, int]]:
    return {index: (row, column) for row, (_, indices) in enumerate(rows) for column, index in enumerate(indices)}


_XTERM_256_POSITION_BY_INDEX = _build_position_map(_XTERM_256_ROWS)
# Shared by BASE and WITH_HINT: neither carries a selectable sentinel.
_XTERM_256_POSITION_BY_INDEX_BASE = _build_position_map(_XTERM_256_ROWS_BASE)

_XTERM_CELL_STYLES: tuple[Style, ...] = tuple(
    Style.from_color(
        _palette_color(i).get_contrast_text().with_alpha(1.0).rich_color,
        _palette_color(i).rich_color,
    )
    for i in range(256)
)
_ANSI_SWATCH_STYLES: tuple[Style, ...] = tuple(
    Style.from_color(bgcolor=Color.parse(token).rich_color) for token in _ANSI_COLOR_TOKENS
)


class _ResetButton(Button):
    """Per-row reset glyph.

    ``target`` is ``"color:<name>"`` or ``"var:<name>"`` so one handler
    can dispatch on the prefix. ``_refresh_dirty_state`` toggles the
    ``--clean`` class; the CSS uses it to hide the glyph on unedited rows.
    """

    # Mirror of ``_ColorSwatchButton`` bindings: ``left`` hops back to
    # the swatch, ``up``/``down`` move to the neighbouring row's swatch
    # (returning the user to the main navigation column).
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("left", "focus_swatch", show=False),
        Binding("up", "navigate_row(-1)", show=False),
        Binding("down", "navigate_row(1)", show=False),
    ]

    def __init__(self, target: str) -> None:
        super().__init__(_RESET_BUTTON_LABEL, classes="--reset-color --clean")
        self.target = target

    def _find_swatch(self) -> _ColorSwatchButton | None:
        screen = self.screen
        if screen is None:
            return None
        for swatch in screen.query(_ColorSwatchButton):
            if swatch._reset_target() == self.target:
                return swatch
        return None

    def action_focus_swatch(self) -> None:
        if (swatch := self._find_swatch()) is not None:
            swatch.focus()

    def action_navigate_row(self, delta: int) -> None:
        # Up/down on a reset returns to the swatch column, then moves
        # one row in that direction — same end state as if the user had
        # pressed Left first, then Up/Down on the swatch.
        if (swatch := self._find_swatch()) is not None:
            swatch.action_move_focus(delta)


class _ColorSwatchButton(Button):
    """Shared rendering for one-row color swatches.

    Non-hex values (``transparent``, ``ansi_white 40%``, ...) keep the
    button clickable so the picker can replace them with a hex value;
    the ``--non-hex`` class only marks them visually.
    """

    # Keyboard navigation between rows + lateral hop to the row's reset
    # button.  Bindings live on the swatch itself so they only fire when
    # a swatch is focused — no priority/global side effects.
    BINDINGS: ClassVar[list[Binding]] = [
        Binding("up", "move_focus(-1)", show=False),
        Binding("down", "move_focus(1)", show=False),
        Binding("right", "focus_reset", show=False),
    ]

    def __init__(self, value: str | None) -> None:
        super().__init__("")
        self.set_value(value)

    def _reset_target(self) -> str | None:
        """``_ResetButton.target`` for the row's reset glyph.  Subclasses set."""
        return None

    def action_move_focus(self, delta: int) -> None:
        """Focus the previous/next swatch (clamped at the ends)."""
        screen = self.screen
        if screen is None:
            return
        # ``query`` matches subclasses, so this picks up both
        # ``_FlatThemeColorButton`` and ``_VariableSwatchButton`` rows in
        # DOM order — same order the user sees on screen.
        swatches = list(screen.query(_ColorSwatchButton))
        try:
            idx = swatches.index(self)
        except ValueError:
            return
        next_idx = max(0, min(idx + delta, len(swatches) - 1))
        swatches[next_idx].focus()

    def action_focus_reset(self) -> None:
        """Hop to this row's reset glyph, but only when it's visible.

        Clean rows hide their reset via ``--clean`` (visibility: hidden);
        focusing a hidden widget would be a confusing no-op, so we skip.
        """
        target = self._reset_target()
        if target is None:
            return
        screen = self.screen
        if screen is None:
            return
        for reset in screen.query(_ResetButton):
            if reset.target == target and not reset.has_class("--clean"):
                reset.focus()
                return

    def set_value(self, value: str | None) -> None:
        if value is None:
            self.label = "(not set)"
            self.tooltip = None
            self.add_class("--unset")
            self.remove_class("--non-hex")
            self.styles.background = None
            return
        ansi_label = _ansi_display_label(value)
        self.label = ansi_label or value
        self.tooltip = value if ansi_label is not None else None
        self.remove_class("--unset")
        color = _try_parse_color(value)
        if color is None and (token := _ansi_base_token(value)) is not None:
            color = Color.parse(token)
        if color is None:
            self.add_class("--non-hex")
            self.styles.background = None
            return
        self.remove_class("--non-hex")
        # Normalise alpha to opaque so the swatch renders as a solid
        # colour. Alpha is unsupported end-to-end — see
        # ``_PickerModal.compose`` for the constraint.
        if color.a < 1.0:
            color = color.with_alpha(1.0)
        self.styles.background = color


class _FlatThemeColorButton(_ColorSwatchButton):
    """Single-row swatch for a ``Theme.<color>`` attribute."""

    def __init__(self, value: str | None, color_name: str) -> None:
        self.color_name = color_name
        super().__init__(value)

    def _reset_target(self) -> str:
        return f"color:{self.color_name}"


class _VariableSwatchButton(_ColorSwatchButton):
    """Single-row swatch for a ``theme.variables`` entry."""

    def __init__(self, var_name: str, value: str | None) -> None:
        self.var_name = var_name
        super().__init__(value)

    def _reset_target(self) -> str:
        return f"var:{self.var_name}"


class _DismissableModal(ModalScreen):
    """``ModalScreen`` with shared escape-to-close + backdrop click dismiss.

    Both modal screens in this file follow the same "click outside the
    dialog or press escape to dismiss" idiom; this base captures it so
    subclasses only describe their dialog content.
    """

    BINDINGS: ClassVar[list] = [("escape", "close")]

    def action_close(self) -> None:
        self.dismiss()

    def on_click(self, event: events.Click) -> None:
        # ``get_widget_at`` returns the topmost widget under the cursor;
        # if it's the screen itself, the click landed on the backdrop.
        clicked, _ = self.get_widget_at(event.screen_x, event.screen_y)
        if clicked is self:
            self.dismiss()


class _AnsiTokenPicker(Widget):
    """Palette picker for Textual's terminal ANSI color tokens."""

    ALLOW_SELECT = False
    CAN_FOCUS = True

    class Changed(Message):
        """Posted when the selected ANSI token changes."""

        def __init__(self, picker: _AnsiTokenPicker, token: str) -> None:
            super().__init__()
            self.picker = picker
            self.token = token

    def __init__(self, value: str | None) -> None:
        super().__init__()
        token = _ansi_base_token(value) or "ansi_default"
        self._selected_index = _ANSI_COLOR_TOKENS.index(token)

    @property
    def token(self) -> str:
        return _ANSI_COLOR_TOKENS[self._selected_index]

    def render_line(self, y: int) -> Strip:
        if y >= len(_ANSI_COLOR_TOKENS):
            return Strip.blank(self.size.width)
        token = _ANSI_COLOR_TOKENS[y]
        marker = ">" if y == self._selected_index else " "
        label_style = Style(bold=y == self._selected_index)
        return Strip(
            [
                Segment(f"{marker} ", label_style),
                Segment("    ", _ANSI_SWATCH_STYLES[y]),
                Segment(f" {token}", label_style),
            ]
        )

    def render(self) -> Text:
        return Text(self.token)

    def action_move(self, dy: int) -> None:
        self._select_index(min(max(self._selected_index + dy, 0), len(_ANSI_COLOR_TOKENS) - 1), preview=True)

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        offset = event.get_content_offset(self)
        if offset is None or offset.y >= len(_ANSI_COLOR_TOKENS):
            return
        self._select_index(offset.y, preview=True)

    def _select_index(self, index: int, *, preview: bool) -> None:
        # No-op guard avoids a full CSS reparse when the user re-selects
        # the current row (arrow clamped at edge, click on selected row).
        if index == self._selected_index:
            return
        self._selected_index = index
        self.refresh()
        if preview:
            self.post_message(self.Changed(self, self.token))


class _Xterm256PalettePicker(Widget):
    """Palette-only picker for chrys themes.

    ``textual_colorpicker`` is a continuous RGB/HSV picker, so snapping
    after each drag tick feels jittery.  This widget removes the
    continuous controls entirely and lets users pick only real
    xterm-256 entries.
    """

    ALLOW_SELECT = False
    CAN_FOCUS = True

    _CELL_WIDTH = 4

    def __init__(
        self,
        color: Color,
        *,
        allow_transparent: bool = True,
        disabled_hint: str | None = None,
    ) -> None:
        super().__init__()
        # See ``_XTERM_256_ROWS_BASE`` for why callers disable transparent.
        # With ``disabled_hint`` we keep the "Transparent" header visible
        # and paint the hint where the sentinel cell would have been — so
        # the user sees both where the option lives and why it's gone.
        # Without a hint, the section is removed silently.
        self._disabled_hint = disabled_hint if not allow_transparent else None
        if allow_transparent:
            self._rows = _XTERM_256_ROWS
            position_map = _XTERM_256_POSITION_BY_INDEX
        elif self._disabled_hint:
            self._rows = _XTERM_256_ROWS_WITH_HINT
            position_map = _XTERM_256_POSITION_BY_INDEX_BASE
        else:
            self._rows = _XTERM_256_ROWS_BASE
            position_map = _XTERM_256_POSITION_BY_INDEX_BASE
        self._selectable_rows = tuple(row for row, (_, indices) in enumerate(self._rows) if indices)
        self._selected_index = self._match_palette_index(color, allow_transparent=allow_transparent)
        self._selected_row, self._selected_column = position_map[self._selected_index]

    @property
    def color(self) -> Color:
        if self._selected_index == _TRANSPARENT_SENTINEL:
            return Color(0, 0, 0, a=0)
        return _palette_color(self._selected_index)

    def render_line(self, y: int) -> Strip:
        # CSS height matches the full layout; shorter layouts pad with
        # ``rich_style`` so the parent surface doesn't bleed through.
        if y >= len(self._rows):
            return Strip.blank(self.size.width, style=self.rich_style)
        label, indices = self._rows[y]
        # WITH_HINT placeholder row — only emitted when ``_disabled_hint`` is set.
        if label is None and not indices:
            assert self._disabled_hint is not None
            return Strip([Segment(self._disabled_hint, Style(dim=True, italic=True))])
        if label is not None:
            return Strip([Segment(label, Style(dim=not label, bold=bool(label)))])

        segments: list[Segment] = []
        for index in indices:
            if index == _TRANSPARENT_SENTINEL:
                # No background to render; ``▒`` keeps the cell visible at
                # the 4-col width. Selected state overlays a ``✓`` so the
                # marker is content-based (matches how numbered cells show
                # their index when selected) rather than style-only — bold
                # + dim alone reads ambiguous on the checker pattern.
                selected = index == self._selected_index
                text = " ✓ ▒" if selected else "▒" * self._CELL_WIDTH
                style = Style(dim=not selected, bold=selected)
                segments.append(Segment(text, style))
                continue
            text = f"{index:03d} " if index == self._selected_index else " " * self._CELL_WIDTH
            segments.append(Segment(text, _XTERM_CELL_STYLES[index]))
        return Strip(segments)

    def render(self) -> Text:
        if self._selected_index == _TRANSPARENT_SENTINEL:
            return Text("transparent")
        return Text(f"{self._selected_index:03d} {self.color.hex}")

    def action_move(self, dx: int, dy: int) -> None:
        if dy:
            row = self._move_row(dy)
            column = min(self._selected_column, len(self._rows[row][1]) - 1)
            self._select_cell(row, column, preview=True)
            return
        row_indices = self._rows[self._selected_row][1]
        column = min(max(self._selected_column + dx, 0), len(row_indices) - 1)
        self._select_cell(self._selected_row, column, preview=True)

    async def _on_mouse_down(self, event: events.MouseDown) -> None:
        offset = event.get_content_offset(self)
        if offset is None:
            return
        if offset.y >= len(self._rows):
            return
        indices = self._rows[offset.y][1]
        if not indices:
            return
        column = offset.x // self._CELL_WIDTH
        if column >= len(indices):
            return
        self._select_cell(offset.y, column, preview=True)

    def _move_row(self, dy: int) -> int:
        current_position = self._selectable_rows.index(self._selected_row)
        next_position = min(max(current_position + dy, 0), len(self._selectable_rows) - 1)
        return self._selectable_rows[next_position]

    def _select_cell(self, row: int, column: int, *, preview: bool) -> None:
        # See ``_AnsiTokenPicker._select_index`` for the no-op rationale.
        if (row, column) == (self._selected_row, self._selected_column):
            return
        self._selected_row = row
        self._selected_column = column
        self._selected_index = self._rows[row][1][column]
        self.refresh()
        if preview:
            self.post_message(ColorPicker.Changed(self, self.color))

    @staticmethod
    def _match_palette_index(color: Color, *, allow_transparent: bool = True) -> int:
        if color.is_transparent and allow_transparent:
            return _TRANSPARENT_SENTINEL
        # Transparent input with no sentinel cell falls through to the
        # nearest palette entry — BASE map has no sentinel key.
        return EIGHT_BIT_PALETTE.match((color.r, color.g, color.b))


class _PickerModal(_DismissableModal):
    """Generic color picker driven by a caller-supplied ``write`` callback.

    Non-hex ``initial`` values fall back to ``#FFFFFF``.
    ``mutate_reactive`` is fired here so callers don't repeat it.
    """

    def __init__(
        self,
        editor: ResettableThemeEditor,
        *,
        title: str,
        initial: str | None,
        write: Callable[[str], None],
        allow_transparent: bool = True,
        disabled_hint: str | None = None,
    ) -> None:
        super().__init__()
        self._editor = editor
        self._title = title
        self._write = write
        self._allow_transparent = allow_transparent
        # Mirror ``_Xterm256PalettePicker``'s normalization so contradictory
        # caller intent (hint + ``allow_transparent=True``) reads as ``None``
        # at every layer rather than only at the picker.
        self._disabled_hint = disabled_hint if not allow_transparent else None
        self._current_value = initial
        self._picker_mode = self._initial_picker_mode(initial)
        self._picker: Widget | None = None
        # Alpha-bearing values are normalized to opaque because
        # ``textual_colorpicker.HexInput`` validates 6-digit hex only.
        # chrys themes pre-composite their alpha values to opaque
        # equivalents (see ``theme.py``), so this normalization only
        # triggers on non-chrys themes that retain alpha (e.g.
        # gruvbox/monokai ``input-selection-background``).
        parsed = _try_parse_color(initial) if initial else None
        if parsed is not None and parsed.a < 1.0:
            parsed = parsed.with_alpha(1.0)
        self._initial_color = parsed or Color.parse("#FFFFFF")

    def compose(self) -> ComposeResult:
        with VerticalGroup() as dialog:
            dialog.border_title = self._title
            if self._editor._is_chrys_ansi():
                yield Tabs(
                    Tab("ANSI", id="picker-mode-ansi"),
                    Tab("256-color", id="picker-mode-xterm"),
                    active=f"picker-mode-{self._picker_mode}",
                    id="picker-mode-tabs",
                )
            with VerticalGroup(id="picker-host"):
                yield self._make_picker()

    def on_mount(self) -> None:
        self._focus_picker()

    def _initial_picker_mode(self, value: str | None) -> str:
        """Pick the first picker shown when the modal opens."""
        if not self._editor._is_chrys_ansi():
            return "xterm" if self._editor._is_chrys_family() else "rgb"
        return "ansi" if _ansi_base_token(value) is not None else "xterm"

    def _current_color(self) -> Color:
        # Detected before ``_try_parse_color`` because that helper deliberately
        # rejects ``"transparent"`` (alpha would crash ``HexInput``).  Here we
        # do want alpha=0 to flow through so the xterm picker can highlight
        # its transparent cell.
        if self._current_value and self._current_value.strip().lower() == "transparent":
            return Color(0, 0, 0, a=0)
        parsed = _try_parse_color(self._current_value) if self._current_value else None
        if parsed is not None and parsed.a < 1.0:
            return parsed.with_alpha(1.0)
        if parsed is not None:
            return parsed
        token = _ansi_base_token(self._current_value)
        if token is not None:
            return Color.parse(token)
        return Color.parse("#FFFFFF")

    def _make_picker(self) -> Widget:
        if self._picker_mode == "ansi":
            self._picker = _AnsiTokenPicker(self._current_value)
        elif self._picker_mode == "xterm":
            self._picker = _Xterm256PalettePicker(
                self._current_color(),
                allow_transparent=self._allow_transparent,
                disabled_hint=self._disabled_hint,
            )
        else:
            self._picker = ColorPicker(self._current_color())
        return self._picker

    def _focus_picker(self) -> None:
        if self._picker is not None:
            self._picker.focus()

    async def _switch_picker_mode(self, mode: str) -> None:
        if self._picker_mode == mode:
            return
        self._picker_mode = mode
        host = self.query_one("#picker-host", VerticalGroup)
        await host.remove_children()
        await host.mount(self._make_picker())
        self._focus_picker()

    async def on_key(self, event: events.Key) -> None:
        """Drive the palette picker even when focus stays on the modal."""
        # Tab / Shift+Tab toggles between ANSI and 256-color modes on
        # chrys-ansi.  Goes through ``Tabs.active`` so the tab strip's
        # underline bar moves in sync; the ``TabActivated`` handler
        # picks it up and swaps the mounted picker.
        if self._editor._is_chrys_ansi() and event.key in {"tab", "shift+tab"}:
            event.stop()
            event.prevent_default()
            next_mode = "xterm" if self._picker_mode == "ansi" else "ansi"
            self.query_one("#picker-mode-tabs", Tabs).active = f"picker-mode-{next_mode}"
            return
        handler = self._key_handler(event.key)
        if handler is None:
            return
        event.stop()
        event.prevent_default()
        handler()

    _XTERM_KEY_MOVES: ClassVar[dict[str, tuple[int, int]]] = {
        "left": (-1, 0),
        "right": (1, 0),
        "up": (0, -1),
        "down": (0, 1),
    }
    _ANSI_KEY_MOVES: ClassVar[dict[str, int]] = {"up": -1, "down": 1}

    def _key_handler(self, key: str) -> Callable[[], None] | None:
        """Return the action bound to ``key`` for whichever picker is mounted."""
        picker = self._picker
        if isinstance(picker, _Xterm256PalettePicker):
            if (move := self._XTERM_KEY_MOVES.get(key)) is not None:
                return lambda: picker.action_move(*move)
            if key == "enter":
                return self.dismiss
        elif isinstance(picker, _AnsiTokenPicker):
            if (step := self._ANSI_KEY_MOVES.get(key)) is not None:
                return lambda: picker.action_move(step)
            if key == "enter":
                return self.dismiss
        return None

    @on(Tabs.TabActivated, "#picker-mode-tabs")
    async def _on_picker_mode_tab_activated(self, event: Tabs.TabActivated) -> None:
        event.stop()
        # ``Tab.id`` is ``"picker-mode-<mode>"``; strip the prefix.
        await self._switch_picker_mode(event.tab.id.removeprefix("picker-mode-"))

    def _write_value(self, value: str) -> None:
        self._current_value = value
        self._write(value)
        self._editor.mutate_reactive(ThemeEditor._theme)

    def on_color_picker_changed(self, event: ColorPicker.Changed) -> None:
        event.stop()
        # ``textual_colorpicker.ColorPicker`` fires ``Changed`` once at
        # construction; suppress only that widget's initial no-op.  The
        # xterm palette picker emits changes only from user actions, and
        # must be allowed to write the initial color again after a user
        # previews another color then returns to it.
        if not isinstance(event.color_picker, _Xterm256PalettePicker) and event.color == self._initial_color:
            return
        # chrys/chrys-ansi themes use ``_Xterm256PalettePicker``, so
        # ``event.color`` here is guaranteed to be an xterm-256 entry
        # or the dedicated transparent cell.
        if event.color.is_transparent:
            self._write_value("transparent")
        else:
            self._write_value(event.color.hex)

    @on(_AnsiTokenPicker.Changed)
    def _on_ansi_token_picker_changed(self, event: _AnsiTokenPicker.Changed) -> None:
        event.stop()
        self._write_value(event.token)


class _ChrysExportModal(_DismissableModal):
    """Export the current theme as paste-ready Python.

    Two modes selected by the ``Name`` field: ``chrys`` (case-insensitive)
    emits replacement blocks for the existing ``src/chrys/app/tui/theme.py``
    definitions; any other valid name emits a standalone ``<SLUG>_THEME``
    block. Built-in textual theme names are rejected so new exports
    can't shadow ``textual-dark`` / ``nord`` / etc.
    """

    def __init__(self, theme: Theme, snapshot: dict[str, dict[str, str | None]] | None = None) -> None:
        super().__init__()
        self._theme = theme
        self._snapshot = snapshot

    def compose(self) -> ComposeResult:
        with VerticalGroup() as dialog:
            dialog.border_title = "Export theme"
            # Pre-pick a non-conflicting name so builtins show usable
            # code immediately (chrys-family names trigger update mode).
            initial_name = (
                self._theme.name
                if self._theme.name in CHRYS_THEMES
                else self._suggest_unique_name(self._theme.name.strip())
            )
            with HorizontalGroup():
                yield Label("Name:")
                yield Input(value=initial_name, id="theme-name")
            yield Label("", id="export-error", classes="--error-message")
            code_display = TextArea(
                self._render_code(initial_name),
                language="python",
                read_only=True,
                id="export-code",
            )
            code_display.cursor_blink = False
            yield code_display

    def _suggest_unique_name(self, base: str) -> str:
        """Return ``base`` if valid, else ``base-1`` / ``base-2`` / ..."""
        if self._validate_name(base) is None:
            return base
        for i in range(1, 1000):
            candidate = f"{base}-{i}"
            if self._validate_name(candidate) is None:
                return candidate
        return base

    def _validate_name(self, name: str) -> str | None:
        if not name:
            return "Name is required."
        if not _THEME_NAME_RE.fullmatch(name):
            return "Start with a letter; use letters, digits, dashes, or underscores."
        lowered = name.lower()
        # Reject the *other* chrys-family name: update mode renders from
        # ``self._theme``, so a cross-name (e.g. typing "chrys-ansi" while
        # exporting from base chrys) would either silently emit wrong-
        # flavour update output or clobber an existing theme.py block.
        if lowered in CHRYS_THEMES and lowered != self._theme.name:
            return f"To export as '{name}', activate the '{name}' theme first (current: '{self._theme.name}')."
        if lowered in CHRYS_FAMILY_THEMES and lowered not in CHRYS_THEMES:
            return f"'{name}' is reserved by a registered Chrys theme."
        if lowered not in CHRYS_THEMES and lowered in BUILTIN_THEMES:
            return f"'{name}' conflicts with a built-in textual theme."
        return None

    def _render_code(self, name: str) -> str:
        # Update mode only kicks in when the typed name exactly matches
        # the active chrys-family theme; cross-names are rejected by
        # ``_validate_name`` so this branch never sees them in practice.
        if self._theme.name in CHRYS_THEMES and name.lower() == self._theme.name:
            body = self._render_update_chrys()
        else:
            body = self._render_new_theme(name)
        return f"{self._mode_hint(name)}\n\n{body}"

    def _mode_hint(self, name: str) -> str:
        """Python comment block describing the export mode + paste/register
        instructions.  Prepended to the rendered code so it travels with
        the snippet on copy/paste and avoids any sidebar wrapping concerns.
        """
        if self._theme.name in CHRYS_THEMES and name.lower() == self._theme.name:
            return (
                f"# Update '{self._theme.name}' — paste this over the matching\n"
                "# blocks in src/chrys/app/tui/theme.py.  No app.py change needed."
            )
        slug = name.upper().replace("-", "_")
        return (
            f"# Create new theme '{name}' — paste this into src/chrys/app/tui/theme.py,\n"
            "# then register in src/chrys/app/tui/app.py via:\n"
            f"#   self.register_theme({slug}_THEME)"
        )

    @staticmethod
    def _format_dict(
        var_name: str,
        entries: Iterable[tuple[str, str]],
        *,
        changed_for_ansi: set[str] | None = None,
    ) -> list[str]:
        """Render a ``{var_name} = {{ ... }}`` module-level dict literal.

        Keys and values are emitted verbatim inside double quotes; trailing
        comma on every line keeps single-line diffs.  The optional ANSI
        marker annotates shared blocks that carry chrys-ansi-only edits.
        """
        lines = [f"{var_name} = {{"]
        for key, value in entries:
            suffix = "  # changed for chrys-ansi" if changed_for_ansi is not None and key in changed_for_ansi else ""
            lines.append(f'    "{key}": "{value}",{suffix}')
        lines.append("}")
        return lines

    @staticmethod
    def _render_chrys_ansi_variables_block(theme: Theme) -> list[str]:
        """Render ``_CHRYS_ANSI_VARIABLES = {...}`` mirroring theme.py.

        Walks ``_CHRYS_ANSI_OVERRIDE_NAMES`` in source order, emitting
        ``_CHRYS_COLORS["<k>"]`` for entries whose value still matches the
        aliased chrys named color (user edits that diverge fall back to a
        quoted literal), and injecting any registered inline comment block
        before its target key.
        """
        lines = ["_CHRYS_ANSI_VARIABLES = {", "    **_CHRYS_VARIABLES,"]
        for name in _CHRYS_ANSI_OVERRIDE_NAMES:
            lines.extend(f"    {comment}" for comment in _CHRYS_ANSI_INLINE_COMMENTS.get(name, ()))
            value = theme.variables.get(name, CHRYS_ANSI_THEME.variables[name])
            alias_key = _CHRYS_ANSI_ALIAS_KEYS.get(name)
            if alias_key is not None and value == getattr(theme, alias_key):
                lines.append(f'    "{name}": _CHRYS_COLORS["{alias_key}"],')
            else:
                lines.append(f'    "{name}": "{value}",')
        lines.append("}")
        return lines

    @staticmethod
    def _set_color_entries(theme: Theme) -> list[tuple[str, str]]:
        """``CHRYS_COLOR_NAMES`` entries that are actually set.

        Omitting None avoids ``"foreground": "None"`` noise; ``Theme``
        defaults Optional fields to None, so omitting == passing None.
        """
        return [(n, value) for n in CHRYS_COLOR_NAMES if (value := getattr(theme, n)) is not None]

    def _snapshot_color(self, name: str) -> str | None:
        if self._snapshot is not None:
            return self._snapshot["colors"].get(name)
        if self._theme.name == _CHRYS_ANSI_NAME:
            return _CHRYS_ANSI_BASE_COLOR_VALUES.get(name)
        return None

    def _snapshot_variable(self, name: str) -> str | None:
        if self._snapshot is not None:
            return self._snapshot["variables"].get(name)
        if self._theme.name == _CHRYS_ANSI_NAME:
            return _CHRYS_ANSI_BASE_VARIABLE_VALUES.get(name)
        return _CHRYS_BASE_VARIABLE_VALUES.get(name)

    def _render_update_chrys(self) -> str:
        theme = self._theme
        is_ansi = theme.name == _CHRYS_ANSI_NAME
        # ``_CHRYS_VARIABLES`` reflects base chrys; ansi-only and
        # shared-key overrides flow into ``_CHRYS_ANSI_VARIABLES``.
        if is_ansi:
            base_variables = {
                name: (
                    _CHRYS_BASE_VARIABLE_VALUES[name]
                    if name in _CHRYS_ANSI_OVERRIDE_NAMES
                    else theme.variables.get(name, _CHRYS_BASE_VARIABLE_VALUES[name])
                )
                for name in _CHRYS_BASE_VARIABLE_VALUES
            }
            for name, value in theme.variables.items():
                if name not in base_variables and name not in _CHRYS_ANSI_OVERRIDE_NAMES:
                    base_variables[name] = value
        else:
            base_variables = {
                name: theme.variables.get(name, _CHRYS_BASE_VARIABLE_VALUES[name])
                for name in _CHRYS_BASE_VARIABLE_VALUES
            }
            for name, value in theme.variables.items():
                if name not in base_variables:
                    base_variables[name] = value
        chrys_only_entries = [(k, base_variables[k]) for k in CHRYS_ONLY_VARIABLES]
        rest_entries = [(k, v) for k, v in base_variables.items() if k not in CHRYS_ONLY_VARIABLES]
        changed_for_ansi_variables = (
            {
                name
                for name, value in base_variables.items()
                if name not in _CHRYS_ANSI_OVERRIDE_NAMES and self._snapshot_variable(name) != value
            }
            if is_ansi
            else None
        )
        variables_lines = self._format_dict(
            "_CHRYS_VARIABLES",
            rest_entries,
            changed_for_ansi=changed_for_ansi_variables,
        )
        variables_lines = [*variables_lines[:-1], "    **CHRYS_ONLY_VARIABLES,", variables_lines[-1]]
        changed_for_ansi_colors = (
            {name for name in CHRYS_COLOR_NAMES if getattr(theme, name) != self._snapshot_color(name)}
            if is_ansi
            else None
        )
        # Base chrys hoists the four surfaces to _DARK_GRAY; chrys-ansi
        # inlines them per-slot in the constructor instead.
        lines: list[str] = []
        if not is_ansi:
            lines += [f'_DARK_GRAY = "{theme.background}"', ""]
        lines += [
            *self._format_dict(
                "_CHRYS_COLORS",
                self._set_color_entries(theme),
                changed_for_ansi=changed_for_ansi_colors,
            ),
            "",
            *self._format_dict(
                "CHRYS_ONLY_VARIABLES: dict[str, str]",
                chrys_only_entries,
                changed_for_ansi=changed_for_ansi_variables,
            ),
            "",
            *variables_lines,
        ]
        if is_ansi:
            ansi_lines = self._render_chrys_ansi_variables_block(theme)
            constructor_lines = [
                "CHRYS_ANSI_THEME = Theme(",
                f'    name="{_CHRYS_ANSI_NAME}",',
                "    ansi=True,",
                "    **_CHRYS_COLORS,",
                f'    background="{theme.background}",',
                f'    surface="{theme.surface}",',
                f'    panel="{theme.panel}",',
                f'    boost="{theme.boost}",',
                f"    dark={theme.dark},",
                "    variables=_CHRYS_ANSI_VARIABLES,",
                ")",
            ]
            lines.extend(["", *ansi_lines, "", *constructor_lines])
        return "\n".join(lines)

    def _render_new_theme(self, name: str) -> str:
        theme = self._theme
        # ``slug`` is uppercased + dashes-to-underscores so identifiers
        # like ``_MY_THEME_BACKGROUND`` stay valid Python.
        slug = name.upper().replace("-", "_")
        # chrys-family: flat-surface invariant → hoist a single constant.
        # Non-chrys: inline values; skip None (Theme defaults them).
        if theme.name in CHRYS_THEMES:
            surface_decls = [f'_{slug}_BACKGROUND = "{theme.background}"']
            surface_kwargs = [
                f"    background=_{slug}_BACKGROUND,",
                f"    surface=_{slug}_BACKGROUND,",
                f"    panel=_{slug}_BACKGROUND,",
                f"    boost=_{slug}_BACKGROUND,",
            ]
        else:
            surface_decls = []
            surface_kwargs = []
            for attr in _SURFACE_COLOR_NAMES:
                value = getattr(theme, attr)
                if value is None:
                    continue
                surface_kwargs.append(f'    {attr}="{value}",')
        lines: list[str] = []
        if surface_decls:
            lines.extend(surface_decls)
            lines.append("")
        lines.extend(self._format_dict(f"_{slug}_COLORS", self._set_color_entries(theme)))
        lines.append("")
        lines.extend(self._format_dict(f"_{slug}_VARIABLES", theme.variables.items()))
        lines.extend(
            [
                "",
                f"{slug}_THEME = Theme(",
                f'    name="{name}",',
                *(["    ansi=True,"] if theme.ansi else []),
                f"    **_{slug}_COLORS,",
                *surface_kwargs,
                f"    dark={theme.dark},",
                f"    variables=_{slug}_VARIABLES,",
                ")",
            ]
        )
        return "\n".join(lines)

    @on(Input.Changed, "#theme-name")
    def _on_name_changed(self, event: Input.Changed) -> None:
        event.stop()
        name = event.value.strip()
        error_label = self.query_one("#export-error", Label)
        code = self.query_one("#export-code", TextArea)
        error = self._validate_name(name)
        # On error: keep the previous valid render visible for reference,
        # but disable the textarea so it's visually dimmed and the stale
        # text can't be selected/copied while the name is invalid.
        code.disabled = error is not None
        if error is not None:
            error_label.update(error)
            return
        error_label.update("")
        code.load_text(self._render_code(name))


class ResettableThemeEditor(ThemeEditor):
    """Chrys-aware editor: named colors + ``theme.variables`` + snapshot reset.

    Variable rows always cover ``_CANONICAL_VARIABLE_NAMES`` regardless
    of which theme is active; missing entries render as ``(not set)``.
    Snapshots are captured on first sight per theme name; reset restores
    those source-defined values.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Captured on first sight per theme so reset restores
        # source-defined values, not the user's last edit.
        self._snapshots: dict[str, dict[str, dict[str, str | None]]] = {}
        # Per-row widget refs populated during compose so
        # ``_watch__theme`` can repaint without DOM queries.
        self._color_buttons: dict[str, _FlatThemeColorButton] = {}
        self._variable_swatches: dict[str, _VariableSwatchButton] = {}
        self._reset_buttons: dict[str, _ResetButton] = {}
        # Constructed eagerly so ``_refresh_dirty_state`` can toggle
        # its ``disabled`` state without a DOM lookup; yielded by ``compose``.
        self._reset_all_button = Button(
            "Reset all",
            classes="--reset-all",
            flat=True,
            variant="warning",
            disabled=True,
        )
        # Coalesce CSS reparse requests: a ColorPicker drag emits dozens
        # of ``Changed`` events per frame, each triggering ``_watch__theme``;
        # without this flag we'd reparse ~1500 lines of CSS per event.
        self._css_refresh_pending = False

    def on_mount(self) -> None:
        # Parent's ``on_mount`` installs the live-app theme watcher; we
        # don't share that handler chain through Textual's event system
        # because we name-shadow it, so call super explicitly.
        super().on_mount()
        # Defer until after the first refresh so all children (swatches,
        # MainScreen's chat widgets, etc.) have finished their own mount
        # focus dance — otherwise a later mount would steal focus back.
        self.call_after_refresh(self.focus_first_swatch)

    def compose(self) -> ComposeResult:
        # Read through the live app theme — the inherited ``_theme``
        # reactive still holds ``textual-dark`` at this point.
        theme = self.app.current_theme
        self.set_class(theme.ansi and not self._is_chrys_ansi(theme), "--ansi-disabled")

        with VerticalScroll(can_focus=False) as scroll:
            scroll.border_title = "Theme Editor"
            yield Label(self._base_theme_line(theme), id="base-theme", classes="--meta")
            # Empty string on non-chrys themes — row keeps its slot.
            yield Label(self._sync_hint_text(theme), id="surface-sync-hint", classes="--sync-hint")
            yield Label("Colors", classes="--section-header")
            with VerticalGroup(classes="--rows-grid"):
                for color_name in _THEME_COLOR_NAMES:
                    raw = getattr(theme, color_name, None)
                    yield Label(color_name)
                    button = _FlatThemeColorButton(raw, color_name)
                    self._color_buttons[color_name] = button
                    yield button
                    reset = _ResetButton(f"color:{color_name}")
                    self._reset_buttons[reset.target] = reset
                    yield reset

            yield Label("Variables", classes="--section-header")
            for group_name, names in _group_variables(list(_CANONICAL_VARIABLE_NAMES)):
                yield Label(f"----- {group_name} -----", classes="--group-header")
                with VerticalGroup(classes="--rows-grid"):
                    for var_name in names:
                        value = theme.variables.get(var_name)
                        swatch = _VariableSwatchButton(var_name, value)
                        self._variable_swatches[var_name] = swatch
                        yield Label(var_name)
                        yield swatch
                        reset = _ResetButton(f"var:{var_name}")
                        self._reset_buttons[reset.target] = reset
                        yield reset

        with HorizontalGroup():
            yield self._reset_all_button
            yield Button("Export theme", classes="--export-button", flat=True, variant="primary")

        # Sibling of the scroll/action group so ``.--ansi-disabled``
        # can swap them in/out without rearranging the DOM.
        yield Label(
            "Theme editing is not supported on built-in\n"
            "ANSI themes.\n\n"
            "Switch to chrys-ansi, or any RGB theme\n"
            "to continue.",
            classes="--ansi-notice",
        )

    def _is_chrys_family(self, theme: Theme | None = None) -> bool:
        """True for chrys-family themes."""
        if theme is None:
            theme = self._theme
        return theme.name in CHRYS_THEMES

    def _is_chrys_ansi(self, theme: Theme | None = None) -> bool:
        """True only for the chrys-ansi theme (gates the ANSI picker mode)."""
        if theme is None:
            theme = self._theme
        return theme.name == _CHRYS_ANSI_NAME

    def _editing_disabled(self, theme: Theme | None = None) -> bool:
        """True when the active theme can be viewed but not edited."""
        if theme is None:
            if self.has_class("--ansi-disabled"):
                return True
            theme = self._theme
        return theme.ansi and not self._is_chrys_ansi(theme)

    @staticmethod
    def _base_theme_line(theme: Theme) -> str:
        modifiers = (
            "dark" if theme.dark else "light",
            "ansi" if theme.ansi else "rgb",
        )
        return f"{theme.name} ({', '.join(modifiers)})"

    def _sync_hint_text(self, theme: Theme) -> str:
        if self._is_chrys_family(theme):
            return "Note: in chrys themes, background & surface & panel & boost are kept in sync"
        return ""

    def _on_app_theme_change(self, theme_name: str) -> None:
        theme = self.app.get_theme(theme_name)
        assert theme is not None
        editing_disabled = self._editing_disabled(theme)
        # First watcher call fires before ``compose`` mounts the labels.
        with contextlib.suppress(NoMatches):
            self.query_one("#base-theme", Label).update(self._base_theme_line(theme))
            self.query_one("#surface-sync-hint", Label).update(self._sync_hint_text(theme))
        # Built-in ANSI themes have no useful chrys-style derivation —
        # hide the editor and show a notice. chrys-ansi gets full editing.
        self.set_class(editing_disabled, "--ansi-disabled")
        if not editing_disabled and theme.name not in self._snapshots:
            # ``None`` entries record "theme doesn't define this"; reset
            # for such entries pops the key, restoring ``(not set)``.
            self._snapshots[theme.name] = {
                "colors": {n: getattr(theme, n) for n in _THEME_COLOR_NAMES},
                "variables": {n: theme.variables.get(n) for n in _CANONICAL_VARIABLE_NAMES},
            }
        self._theme = theme
        if editing_disabled and self._has_internal_focus() and self.screen is not None:
            self.screen.set_focus(None)

    def _watch__theme(self) -> None:
        # Repaint from raw ``Theme.<color>`` / ``Theme.variables`` — we
        # deliberately don't fall back to ``_color_mapping`` for swatches;
        # only the picker uses that as its initial-color seed.
        for color_name, button in self._color_buttons.items():
            button.set_value(getattr(self._theme, color_name, None))
        for var_name, swatch in self._variable_swatches.items():
            swatch.set_value(self._theme.variables.get(var_name))
        self._refresh_dirty_state()
        # Coalesce CSS reparses to one per frame (see ``__init__``).
        if not self._css_refresh_pending:
            self._css_refresh_pending = True
            self.app.call_after_refresh(self._run_css_refresh)

    def _run_css_refresh(self) -> None:
        self._css_refresh_pending = False
        # Private textual API — no public hook for "vars changed,
        # repaint."  If renamed upstream, the playground fails loud.
        self.app._invalidate_css()
        self.app.refresh_css()

    def _refresh_dirty_state(self) -> None:
        """Show reset button only on rows whose value diverges from the snapshot.

        ``Reset all`` stays visible — only its ``disabled`` state flips
        — so the bottom action bar's layout doesn't jump on edits.
        """
        snap = self._snapshots.get(self._theme.name)
        if snap is None:
            self._reset_all_button.disabled = True
            return
        any_dirty = False
        for target, btn in self._reset_buttons.items():
            kind, _, name = target.partition(":")
            if kind == "color":
                dirty = getattr(self._theme, name) != snap["colors"].get(name)
            else:  # var
                dirty = self._theme.variables.get(name) != snap["variables"].get(name)
            btn.set_class(not dirty, "--clean")
            any_dirty = any_dirty or dirty
        self._reset_all_button.disabled = not any_dirty

    def _restore(
        self,
        snap: dict[str, dict[str, str | None]],
        *,
        colors: Iterable[str] = (),
        variables: Iterable[str] = (),
    ) -> None:
        """Write snapshot values back to the live theme.

        Symmetric None handling: a None snapshot entry means the field
        was unset originally; restoring sets the attr back to None /
        pops the variable key, so reset clears edits made over an
        initially-unset slot (e.g. ``textual-light.boost``).
        """
        for n in colors:
            if n not in snap["colors"]:
                continue
            # ``Theme.<color>`` is ``str | None``; setting None reverts
            # to ColorSystem-derived rendering.
            setattr(self._theme, n, snap["colors"][n])
        for n in variables:
            if n not in snap["variables"]:
                continue
            original = snap["variables"][n]
            if original is None:
                self._theme.variables.pop(n, None)
            else:
                self._theme.variables[n] = original

    def _reset_target(self, target: str) -> None:
        snap = self._snapshots.get(self._theme.name)
        if snap is None:
            return
        kind, _, name = target.partition(":")
        if kind == "color":
            # chrys-family fans surface-group resets out to all four
            # slots — same flat-surface rule as ``_write_color``.
            if self._is_chrys_family() and name in _SURFACE_COLOR_NAMES:
                self._restore(snap, colors=_SURFACE_COLOR_NAMES)
            else:
                self._restore(snap, colors=(name,))
        elif kind == "var":
            self._restore(snap, variables=(name,))
        self.mutate_reactive(ThemeEditor._theme)

    def _reset_all(self) -> None:
        snap = self._snapshots.get(self._theme.name)
        if snap is None:
            return
        self._restore(snap, colors=snap["colors"], variables=snap["variables"])
        self.mutate_reactive(ThemeEditor._theme)

    @on(Button.Pressed, ".--reset-color")
    def _on_reset_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._editing_disabled():
            event.prevent_default()
            return
        assert isinstance(event.button, _ResetButton)
        self._reset_target(event.button.target)
        # Reset hides the glyph again once the row is clean; keep
        # keyboard users anchored on the row they just acted on.
        event.button.action_focus_swatch()

    @on(Button.Pressed, ".--reset-all")
    def _on_reset_all_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._editing_disabled():
            event.prevent_default()
            return
        self._reset_all()

    @on(Button.Pressed, "_VariableSwatchButton")
    def _on_variable_swatch_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._editing_disabled():
            event.prevent_default()
            return
        assert isinstance(event.button, _VariableSwatchButton)
        var_name = event.button.var_name
        # ``_color_mapping`` carries textual's algorithmic defaults
        # (e.g. ``foreground-muted = $foreground 60%``); seed the picker
        # with those first, then fall through to App-level defaults for
        # chrys-only slots so ``(not set)`` rows open at the rendered value.
        initial = self._variable_picker_initial(var_name)
        self.app.push_screen(
            _PickerModal(
                self,
                title=f"Change ${var_name}",
                initial=initial,
                write=partial(self._write_variable, var_name),
            )
        )

    def _variable_picker_initial(self, var_name: str) -> str | None:
        return (
            self._theme.variables.get(var_name)
            or self._color_mapping.get(var_name)
            or self.app.theme_variables.get(var_name)
        )

    @on(Button.Pressed, "_FlatThemeColorButton")
    def _on_theme_color_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if self._editing_disabled():
            event.prevent_default()
            return
        button = event.button
        assert isinstance(button, _FlatThemeColorButton)
        color_name = button.color_name
        # Prefer raw ``Theme.<color>``: ``_color_mapping`` HSL-roundtrips
        # through ``Color.lighten(0)`` and can lose one bit (e.g.
        # ``#FFB86C`` → ``#FEB86C``).  Fall back to the mapping only
        # when the attribute is None.
        initial = getattr(self._theme, color_name, None) or self._color_mapping.get(color_name)
        is_surface = color_name in _SURFACE_COLOR_NAMES
        self.app.push_screen(
            _PickerModal(
                self,
                title=f"Change {color_name}",
                initial=initial,
                write=partial(self._write_color, color_name),
                # See ``_XTERM_256_ROWS_BASE`` for why surface slots
                # disable transparent; the hint explains it in-place.
                allow_transparent=not is_surface,
                disabled_hint=f"{color_name} can't be transparent" if is_surface else None,
            )
        )

    def _write_color(self, name: str, css_color: str) -> None:
        setattr(self._theme, name, css_color)
        # chrys-family flat-surface invariant
        # ($background == $surface == $panel == $boost): fan one edit
        # out to all four slots.  ``_reset_target`` mirrors this.
        if self._is_chrys_family() and name in _SURFACE_COLOR_NAMES:
            for other in _SURFACE_COLOR_NAMES:
                if other != name:
                    setattr(self._theme, other, css_color)

    def _write_variable(self, name: str, css_color: str) -> None:
        self._theme.variables[name] = css_color

    # ``prevent_default()`` suppresses upstream's same-selector handler
    # — ``@on`` handlers collected via MRO both fire otherwise.
    @on(Button.Pressed, ".--export-button")
    def _on_export_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        event.prevent_default()
        if self._editing_disabled():
            return
        self.app.push_screen(_ChrysExportModal(self._theme, self._snapshots.get(self._theme.name)))

    def toggle(self) -> None:
        """F10 entry/exit for the editor sidebar.

        Three-state cycle so a single key reliably gets the user in:

        - Hidden       → show + focus first swatch.
        - Visible, no focus inside → focus first swatch (don't hide).
        - Visible, focus inside    → hide.
        """
        if self.has_class("-hidden"):
            self.remove_class("-hidden")
            self.focus_first_swatch()
            return
        if self._editing_disabled():
            return
        if self._has_internal_focus():
            self.add_class("-hidden")
        else:
            self.focus_first_swatch()

    def focus_first_swatch(self) -> None:
        """Move keyboard focus to the topmost swatch row, if any."""
        if self._color_buttons and not self._editing_disabled():
            next(iter(self._color_buttons.values())).focus()

    def _has_internal_focus(self) -> bool:
        focused = self.screen.focused if self.screen else None
        return focused is not None and self in focused.ancestors_with_self


class _MainScreenWithThemeEditor(_ChrysMainScreen):
    """MainScreen that also yields a permanently docked ThemeEditor.

    Yielding the editor from ``compose`` (rather than ``screen.mount``-ing
    it after the fact) means it participates in the screen's initial
    layout pass, so its ``dock: left`` actually claims the left strip
    instead of getting lost behind already-positioned children.
    """

    # textual resolves ``CSS_PATH`` relative to the leaf class's module,
    # which would point at ``playground/screen.tcss``. Pin to chrys's copy.
    CSS_PATH = str(Path(inspect.getfile(_ChrysMainScreen)).parent / "screen.tcss")

    # F10 picked to avoid keys already bound by chrys.  ``priority=True``
    # matches the sidebar-toggle pattern so it fires even when a child
    # widget is focused.
    BINDINGS = [
        *_ChrysMainScreen.BINDINGS,
        Binding("f10", "toggle_theme_editor", "Theme Editor", priority=True),
    ]

    def compose(self) -> ComposeResult:
        yield ResettableThemeEditor()
        yield from super().compose()

    def action_toggle_theme_editor(self) -> None:
        self.query_one(ResettableThemeEditor).toggle()

    def focus_theme_editor_after_startup(self, deadline: float | None = None) -> None:
        """Focus the editor once startup modals have had time to close.

        Polls every 0.1s for up to ~10s; the cap keeps a stuck
        ``_agent_loading`` from spinning forever.
        """
        import time

        if deadline is None:
            deadline = time.monotonic() + 10.0
        if time.monotonic() > deadline:
            return
        if self._agent_loading or self.app.screen is not self:
            self.set_timer(0.1, lambda: self.focus_theme_editor_after_startup(deadline))
            return
        editor = self.query_one(ResettableThemeEditor)
        if not editor.has_class("-hidden") and not editor.has_class("--ansi-disabled"):
            editor.focus_first_swatch()


class ChrysThemeEditorApp(_chrys_app.ChrysApp):
    """ChrysApp using the MainScreen-with-editor subclass."""

    # textual resolves ``CSS_PATH`` relative to the leaf class's module
    # (would point at ``playground/chrys.tcss``).  Pin to chrys's own
    # copy and load the editor's ``theme_editor.tcss`` alongside.
    CSS_PATH = [
        str(Path(_chrys_app.__file__).parent / "chrys.tcss"),
        str(Path(__file__).parent / "theme_editor.tcss"),
    ]

    def _build_main_screen(self) -> _MainScreenWithThemeEditor:
        # Override the production factory to substitute the screen
        # variant that docks the editor, avoiding any monkey-patching
        # of ``chrys.app.tui.app`` from this playground entry point.
        return _MainScreenWithThemeEditor(
            self._bus,
            state_store=self._state_store,
            agent_registry=self._agent_registry,
            model_registry=self._model_registry,
            engine_provider=lambda: self._engine,
        )

    async def _start_engine(self, profile, screen: _ChrysMainScreen) -> None:
        await super()._start_engine(profile, screen)
        if isinstance(screen, _MainScreenWithThemeEditor):
            screen.set_timer(0.35, screen.focus_theme_editor_after_startup)


def main() -> None:
    # Re-use chrys's own startup wiring (event bus, engine, registries,
    # crash logging, post-exit terminal reset) by passing our App subclass
    # to ``chrys.app.tui.app.main``.  The subclass overrides
    # ``_build_main_screen`` to inject the editor-docked screen variant —
    # no module-global patches needed.
    _chrys_app.main(app_cls=ChrysThemeEditorApp)


if __name__ == "__main__":
    main()
