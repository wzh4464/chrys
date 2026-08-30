# Copyright (c) 2026 Chrys. All rights reserved.
# Adapted from toad (https://github.com/batrachianai/toad).

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, NamedTuple

import rich.repr
from textual.style import Style


class ANSIToken:
    pass


class DEC(NamedTuple):
    slot: int
    character_set: str


class DECInvoke(NamedTuple):
    gl: int | None = None
    gr: int | None = None
    shift: int | None = None


DEC_SLOTS = {"(": 0, ")": 1, "*": 2, "+": 3, "-": 1, ".": 2, "//": 3}


type ClearType = Literal["cursor_to_end", "cursor_to_beginning", "screen", "scrollback"]
ANSI_CLEAR: Mapping[int, ClearType] = {
    0: "cursor_to_end",
    1: "cursor_to_beginning",
    2: "screen",
    3: "scrollback",
}


@rich.repr.auto
class ANSIContent(NamedTuple):
    """Content to be written to the terminal."""

    text: str

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.text


@dataclass(frozen=True, slots=True)
class ANSIBell:
    """BEL — request attention without adding printable content."""


@dataclass(frozen=True, slots=True)
class ANSIHorizontalTab:
    """HT — advance to the next horizontal tab stop."""


@rich.repr.auto
class ANSICursor(NamedTuple):
    """Represents a single operation on the ANSI output.

    All values may be `None` meaning "not set".
    """

    delta_x: int | None = None
    """Relative x change."""
    delta_y: int | None = None
    """Relative y change."""
    absolute_x: int | None = None
    """Replace x."""
    absolute_y: int | None = None
    """Replace y."""
    erase: bool = False
    """Erase (replace with spaces)?"""
    clear_range: tuple[int | None, int | None] | None = None
    """Replace range (slice like)."""
    relative: bool = False
    """Should replace be relative (`False`) or absolute (`True`)"""
    update_background: bool = False
    """Optional style for remaining line."""
    auto_scroll: bool = False
    """Perform a scroll with the movement?"""

    def __rich_repr__(self) -> rich.repr.Result:
        yield "delta_x", self.delta_x, None
        yield "delta_y", self.delta_y, None
        yield "absolute_x", self.absolute_x, None
        yield "absolute_y", self.absolute_y, None
        yield "erase", self.erase, False
        yield "clear_range", self.clear_range, None
        yield "relative", self.relative, False
        yield "update_background", self.update_background, False
        yield "auto_scroll", self.auto_scroll, False

    @lru_cache(maxsize=1024)
    def get_clear_offsets(self, cursor_offset: int, line_length: int) -> tuple[int, int]:
        """Get replace offsets.

        Args:
            cursor_offset: Current cursor offset.
            line_length: Length of line.

        Returns:
            A pair of offsets (inclusive).
        """
        assert self.clear_range is not None, "Only call this if the replace attribute has a value"
        replace_start, replace_end = self.clear_range
        if replace_start is None:
            replace_start = cursor_offset
        if replace_end is None:
            replace_end = cursor_offset
        if replace_start < 0:
            replace_start = line_length + replace_start
        if replace_end < 0:
            replace_end = line_length + replace_end
        if self.relative:
            return (cursor_offset + replace_start, cursor_offset + replace_end)
        return (replace_start, replace_end)


@rich.repr.auto
class ANSINewLine:
    """New line (diffrent in alternate buffer)"""


@rich.repr.auto
class ANSIStyle(NamedTuple):
    """Update style."""

    style: Style

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.style


@rich.repr.auto
class ANSIClear(NamedTuple):
    """Enumeration for clearing the 'screen'."""

    clear: ClearType

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.clear


@rich.repr.auto
class ANSIScrollMargin(NamedTuple):
    """Set the scroll margin."""

    top: int | None = None
    bottom: int | None = None

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.top
        yield self.bottom


@rich.repr.auto
class ANSIScroll(NamedTuple):
    """Scroll buffer."""

    direction: Literal[+1, -1]
    lines: int

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.direction
        yield self.lines


class ANSIFeatures(NamedTuple):
    """Terminal feature flags."""

    show_cursor: bool | None = None
    alternate_screen: bool | None = None
    bracketed_paste: bool | None = None
    cursor_blink: bool | None = None
    cursor_keys: bool | None = None
    replace_mode: bool | None = None
    auto_wrap: bool | None = None


MOUSE_TRACKING_MODES = Literal["button", "drag", "all"]
MOUSE_FORMAT = Literal["normal", "utf8", "sgr", "urxvt"]


class ANSIMouseTracking(NamedTuple):
    """Set mouse tracking."""

    mode: Literal["none"] | MOUSE_TRACKING_MODES | None = None
    format: MOUSE_FORMAT | None = None
    focus_events: bool | None = None
    alternate_scroll: bool | None = None


# Not technically part of the terminal protocol
@rich.repr.auto
class ANSIWorkingDirectory(NamedTuple):
    """Working directory changed"""

    path: str

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.path


@rich.repr.auto
class ANSIShellCommand(NamedTuple):
    """Shell command executed by the user (reported via OSC 2026)."""

    command: str

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.command


@rich.repr.auto
class ANSICharacterSet(NamedTuple):
    """Updated character set state."""

    dec: DEC | None = None
    dec_invoke: DECInvoke | None = None


@rich.repr.auto
class ANSICursorSave(NamedTuple):
    """DECSC — Save cursor position."""


class ANSICursorRestore(NamedTuple):
    """DECRC — Restore cursor position."""


class ANSICursorPositionRequest(NamedTuple):
    pass


@rich.repr.auto
class ANSIDeviceAttributesRequest(NamedTuple):
    """Primary or Secondary Device Attributes request."""

    secondary: bool = False


@rich.repr.auto
class ANSIModeStatusRequest(NamedTuple):
    """DECRQM — Request mode status."""

    mode: int
    private: bool = True


type ANSICommand = (
    ANSIBell
    | ANSIHorizontalTab
    | ANSIStyle
    | ANSIContent
    | ANSICursor
    | ANSINewLine
    | ANSIClear
    | ANSIScrollMargin
    | ANSIScroll
    | ANSIWorkingDirectory
    | ANSIShellCommand
    | ANSICharacterSet
    | ANSIFeatures
    | ANSIMouseTracking
    | ANSICursorSave
    | ANSICursorRestore
    | ANSICursorPositionRequest
    | ANSIDeviceAttributesRequest
    | ANSIModeStatusRequest
)
