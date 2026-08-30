# Copyright (c) 2026 Chrys. All rights reserved.
# Adapted from toad (https://github.com/batrachianai/toad).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import rich.repr
from textual._cells import cell_len
from textual.content import Content
from textual.style import NULL_STYLE, Style

EMPTY_LINE = Content()


class LineFold(NamedTuple):
    """A line from the terminal, folded for presentation."""

    line_no: int
    """The (unfolded) line number."""

    line_offset: int
    """The index of the folded line."""

    offset: int
    """The offset within the original line."""

    content: Content
    """The content."""

    updates: int = 0
    """Integer that increments on update."""


@dataclass
class LineRecord:
    """A single line in the terminal."""

    content: Content
    """The content."""

    style: Style = NULL_STYLE
    """The style for the remaining line."""

    folds: list[LineFold] = field(default_factory=list)
    """Line "folds" for wrapped lines."""

    updates: int = 0
    """An integer used for caching."""


@rich.repr.auto
class ScrollMargin(NamedTuple):
    """Margins at the top and bottom of a window that won't scroll."""

    top: int | None = None
    """Margin at the top (in lines), or `None` for no scroll margin set."""
    bottom: int | None = None
    """Margin at the bottom (in lines), or `None` for no scroll margin set."""

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.top
        yield self.bottom

    def get_line_range(self, height: int) -> tuple[int, int]:
        """Get the scrollable line range (inclusive).

        Args:
            height: terminal height.

        Returns:
            A tuple of the (exclusive) top and bottom line numbers that scroll.
        """
        return (
            self.top or 0,
            height - 1 if self.bottom is None else self.bottom,
        )


def _char_offset_to_cell(content: Content, char_offset: int) -> int:
    """Convert a character offset to a cell (column) offset within *content*."""
    return content[:char_offset].cell_length if char_offset > 0 else 0


def _cell_offset_to_char(content: Content, cell_offset: int) -> int:
    """Convert a cell (column) offset to a character offset within *content*.

    Returns the character index whose left edge is at or just before *cell_offset*.
    If *cell_offset* falls on the right half of a wide character, returns that
    character's index (i.e. snaps to the wide character that spans the column).
    """
    cells = 0
    for i, ch in enumerate(content.plain):
        width = cell_len(ch)
        if cells + width > cell_offset:
            return i
        cells += width
    return len(content)


def _replace_cells(line_content: Content, start_cell: int, content: Content, line_style: Style) -> Content:
    """Overwrite ``line_content`` cells ``[start_cell, start_cell + content.cell_length)`` with ``content``.

    Char-based ``Content`` slicing fails when the line or the new content
    contains wide characters: cells (display columns) and chars (string
    indices) diverge. Walking by cell boundaries here keeps column alignment
    correct, and when the replace region bisects a wide character on either
    edge the orphaned half is padded with blank cells so total cell width is
    preserved.
    """
    if start_cell >= line_content.cell_length:
        return Content.assemble(line_content, content, strip_control_codes=False)

    end_cell = start_cell + content.cell_length
    left_split: int | None = None
    right_split: int | None = None
    left_pad = 0
    right_pad = 0
    cells = 0
    for i, ch in enumerate(line_content.plain):
        ch_start = cells
        ch_end = cells + cell_len(ch)
        cells = ch_end
        if left_split is None and ch_end > start_cell:
            left_split = i
            if ch_start < start_cell:
                left_pad = start_cell - ch_start
        if ch_start >= end_cell:
            right_split = i
            break
        if ch_end > end_cell:
            right_pad = ch_end - end_cell

    if left_split is None:
        left_split = len(line_content)
    if right_split is None:
        right_split = len(line_content)

    before = line_content[:left_split]
    if left_pad > 0:
        before = before + Content.blank(left_pad, line_style)
    after = line_content[right_split:]
    if right_pad > 0:
        after = Content.blank(right_pad, line_style) + after

    return Content.assemble(before, content, after, strip_control_codes=False)


def _fold_start_cell(line_content: Content, folded_line: LineFold) -> int:
    """Return the unfolded cell offset where ``folded_line`` starts."""
    return _char_offset_to_cell(line_content, folded_line.offset)


@dataclass
class Buffer:
    """A terminal buffer (scrollback or alternate)"""

    name: str = "buffer"
    """Name of the buffer (debugging aid)."""
    lines: list[LineRecord] = field(default_factory=list)
    """unfolded lines."""
    line_to_fold: list[int] = field(default_factory=list)
    """An index from folded lines on to unfolded lines."""
    folded_lines: list[LineFold] = field(default_factory=list)
    """Folded lines."""
    scroll_margin: ScrollMargin = ScrollMargin(None, None)
    """Scroll margins"""
    cursor_line: int = 0
    """Folded line index."""
    cursor_offset: int = 0
    """Folded line offset in cells (columns)."""
    max_line_width: int = 0
    """The longest line in the buffer."""
    updates: int = 0
    """Updates count (used in caching)."""
    _updated_lines: set[int] | None = None

    @property
    def line_count(self) -> int:
        """Total number of lines."""
        return len(self.lines)

    @property
    def height(self) -> int:
        """Height of the buffer (number of folded lines)."""
        height = len(self.folded_lines)
        if height == 1 and not self.lines[-1].content.plain and self.name == "scrollback":
            height -= 1
        return height

    @property
    def last_line_no(self) -> int:
        """Index of last lines."""
        return len(self.lines) - 1

    @property
    def cursor(self) -> tuple[int, int]:
        """The cursor character offset within the un-folded lines.

        ``cursor_offset`` is stored as a *cell* (column) offset within the
        current fold.  This property converts it back to a *character* offset
        on the unfolded line so callers can use it for Content slicing.
        """

        if self.cursor_line >= len(self.folded_lines):
            return (len(self.folded_lines), 0)
        cursor_folded_line = self.folded_lines[self.cursor_line]
        cursor_line_offset = cursor_folded_line.line_offset
        line_no = cursor_folded_line.line_no
        line = self.lines[line_no]
        position = 0
        for folded_line_offset, folded_line in enumerate(line.folds):
            if folded_line_offset == cursor_line_offset:
                position += _cell_offset_to_char(folded_line.content, self.cursor_offset)
                break
            position += len(folded_line.content)

        return (line_no, position)

    @property
    def is_blank(self) -> bool:
        """Is this buffer blank (spaces in all lines)?"""
        return not any((line.content.plain.strip() or line.content.spans) for line in self.lines)

    def update_cursor(self, line_no: int, cursor_line_offset: int) -> None:
        """Move the cursor to the given unfolded line and *character* offset.

        Converts *cursor_line_offset* (character-based, for Content slicing)
        into a fold index (``cursor_line``) and a **cell** offset within that
        fold (``cursor_offset``).

        Args:
            line_no: Unfolded line number.
            cursor_line_offset: Character offset within the unfolded line.
        """
        line = self.lines[line_no]
        fold_line_start = self.line_to_fold[line_no]
        position = 0
        fold_offset = 0
        for fold_offset, fold in enumerate(line.folds):
            line_length = len(fold.content)
            if cursor_line_offset >= position and cursor_line_offset < position + line_length:
                self.cursor_line = fold_line_start + fold_offset
                char_in_fold = cursor_line_offset - position
                self.cursor_offset = _char_offset_to_cell(fold.content, char_in_fold)
                break
            position += line_length
        else:
            self.cursor_line = fold_line_start + len(line.folds) - 1
            self.cursor_offset = line.folds[-1].content.cell_length

    def update_line(self, line_no: int) -> None:
        """Record an updated line.

        Args:
            line_no: Line number to update.
        """
        if self._updated_lines is not None:
            self._updated_lines.add(line_no)

    def clear(self, updates: int) -> None:
        """Clear the buffer to its initial state.

        Args:
            updates: the initial updates index.

        """
        del self.lines[:]
        del self.line_to_fold[:]
        del self.folded_lines[:]
        self.cursor_line = 0
        self.cursor_offset = 0
        self.max_line_width = 0
        self.updates = updates

    def remove_last_line(self) -> None:
        if not self.lines:
            return
        last_line_index = len(self.lines) - 1
        del self.lines[-1]
        del self.folded_lines[self.line_to_fold[last_line_index] :]
        del self.line_to_fold[last_line_index]
        self.updates += 1
