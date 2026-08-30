# Copyright (c) 2026 Chrys. All rights reserved.
# Adapted from toad (https://github.com/batrachianai/toad).

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from itertools import accumulate

import rich.repr
from textual import events
from textual.content import EMPTY_CONTENT, Content
from textual.geometry import clamp
from textual.style import NULL_STYLE, Style

from chrys.app.tui.terminal.ansi._buffer import (
    EMPTY_LINE,
    Buffer,
    LineFold,
    LineRecord,
    ScrollMargin,
    _cell_offset_to_char,
    _char_offset_to_cell,
    _fold_start_cell,
    _replace_cells,
)
from chrys.app.tui.terminal.ansi._commands import (
    DEC,
    MOUSE_FORMAT,
    MOUSE_TRACKING_MODES,
    ANSIBell,
    ANSICharacterSet,
    ANSIClear,
    ANSICommand,
    ANSIContent,
    ANSICursor,
    ANSICursorPositionRequest,
    ANSICursorRestore,
    ANSICursorSave,
    ANSIDeviceAttributesRequest,
    ANSIFeatures,
    ANSIHorizontalTab,
    ANSIModeStatusRequest,
    ANSIMouseTracking,
    ANSINewLine,
    ANSIScroll,
    ANSIScrollMargin,
    ANSIShellCommand,
    ANSIStyle,
    ANSIWorkingDirectory,
    ClearType,
    DECInvoke,
)
from chrys.app.tui.terminal.ansi._keys import CURSOR_KEYS_APPLICATION, TERMINAL_KEY_MAP
from chrys.app.tui.terminal.ansi._stream import ANSIStream
from chrys.app.tui.terminal.dec import CHARSET_MAP


def show(obj: object) -> object:
    print(obj)
    return obj


@dataclass
class DECState:
    """The (somewhat bonkers) mechanism for switching characters sets pre-unicode."""

    slots: list[str] = field(default_factory=lambda: ["B", "B", "<", "0"])
    gl_slot: int = 0
    gr_slot: int = 2
    shift: int | None = None

    @property
    def gl(self) -> str:
        return self.slots[self.gl_slot]

    @property
    def gr(self) -> str:
        return self.slots[self.gr_slot]

    def update(self, dec: DEC | None, dec_invoke: DECInvoke | None) -> None:
        if dec is not None:
            self.slots[dec.slot] = dec.character_set
        elif dec_invoke is not None:
            if dec_invoke.shift:
                self.shift = dec_invoke.shift
            else:
                if dec_invoke.gl is not None:
                    self.gl_slot = dec_invoke.gl
                elif dec_invoke.gr is not None:
                    self.gr_slot = dec_invoke.gr

    def translate(self, text: str) -> str:
        translate_table: dict[int, str] | None
        first_character: str | None = None
        if self.shift is not None and (translate_table := CHARSET_MAP.get(self.slots[self.shift], None)):
            first_character = text[0].translate(translate_table)
            self.shift = None

        if translate_table := CHARSET_MAP.get(self.gl, None):
            text = text.translate(translate_table)
        if first_character is None:
            return text
        return f"{first_character}{text}"


@dataclass
class MouseTracking:
    """The mouse tracking state."""

    tracking: MOUSE_TRACKING_MODES = "all"
    format: MOUSE_FORMAT = "normal"
    focus_events: bool = False
    alternate_scroll: bool = False


@rich.repr.auto
class TerminalState:
    """Abstract terminal state."""

    def __init__(
        self,
        write_stdin: Callable[[str], Awaitable],
        *,
        width: int = 80,
        height: int = 24,
    ) -> None:
        """
        Args:
            width: Initial width.
            height: Initial height.
        """
        self._write_stdin = write_stdin

        self._ansi_stream = ANSIStream()
        """ANSI stream processor."""

        self.width = width
        """Width of the terminal."""
        self.height = height
        """Height of the terminal."""
        self.style = NULL_STYLE
        """The current style."""
        self.show_cursor = True
        """Is the cursor visible?"""
        self.alternate_screen = False
        """Is the terminal in the alternate buffer state?"""
        self.bracketed_paste = False
        """Is bracketed pase enabled?"""
        self.cursor_blink = False
        """Should the cursor blink?"""
        self.cursor_keys = False
        """Is cursor keys application mode enabled?"""
        self.replace_mode = True
        """Should content replaces characters (`True`) or insert (`False`)?"""
        self.auto_wrap = True
        """Should content wrap?"""
        self.current_directory: str = ""
        """Current working directory."""
        self.last_command: str = ""
        """Last shell command executed (reported via OSC 2026)."""
        self.scrollback_buffer = Buffer("scrollback")
        """Scrollbar buffer lines."""
        self.alternate_buffer = Buffer("alternate")
        """Alternate buffer lines."""
        self.dec_state = DECState()
        """The DEC (character set) state."""
        self.mouse_tracking: MouseTracking | None = None
        """The mouse tracking state."""

        self._saved_cursor: tuple[int, int] | None = None
        """Saved cursor position (line_no, offset) for DECSC/DECRC."""
        self._screen_start_line_no = 0
        """Top folded line of the visible primary screen.

        The primary screen is not always the last ``height`` lines of
        scrollback.  Windows ConPTY/PSReadLine can keep the prompt on its
        current physical row after a vertical grow, leaving blank rows below
        it.  Absolute cursor sequences are relative to that visible screen,
        so keep the screen origin explicit instead of deriving it from the
        current scrollback height.
        """
        self._pending_primary_cursor_reanchor = False
        """Whether the next primary-screen absolute cursor move may rebase origin."""

        self._updates: int = 0
        """Incrementing integer used in caching."""

    def __rich_repr__(self) -> rich.repr.Result:
        yield "width", self.width
        yield "height", self.height
        yield "style", self.style, NULL_STYLE
        yield "show_cursor", self.show_cursor, True
        yield "alternate_screen", self.alternate_screen, False
        yield "bracketed_paste", self.bracketed_paste, False
        yield "cursor_blink", self.cursor_blink, False
        yield "replace_mode", self.replace_mode, True
        yield "auto_wrap", self.auto_wrap, True
        yield "dec_state", self.dec_state
        yield "mouse_tracking", self.mouse_tracking, None

    async def write_stdin(self, text: str) -> bool:
        if self._write_stdin is not None:
            return await self._write_stdin(text)
            return False
        return True

    @property
    def screen_start_line_no(self) -> int:
        if self.alternate_screen:
            return 0
        return max(0, self._screen_start_line_no)

    @property
    def primary_screen_height(self) -> int:
        """Folded height of the primary screen including blank rows below content."""
        return max(self.scrollback_buffer.height, self.screen_start_line_no + self.height)

    @property
    def screen_end_line_no(self) -> int:
        return self.buffer.height

    @property
    def updates(self) -> int:
        """An integer that advanvces when the state is changed."""
        return self._updates

    @property
    def buffer(self) -> Buffer:
        """The buffer (scrollack or alternate)"""
        if self.alternate_screen:
            return self.alternate_buffer
        return self.scrollback_buffer

    @property
    def max_line_width(self) -> int | None:
        return self.scrollback_buffer.max_line_width

    def advance_updates(self) -> int:
        """Advance the `updates` integer and return it.

        Returns:
            int: Updates.
        """
        self._updates += 1
        return self._updates

    def update_size(
        self,
        width: int | None = None,
        height: int | None = None,
        *,
        force_reflow: bool = False,
        reflow: bool = True,
        external_reflow: bool = False,
    ) -> None:
        """Update the dimensions of the terminal.

        Args:
            width: New width, or `None` for no change.
            height: New height, or `None` for no change.
            force_reflow: Rebuild line folds even if the width was already updated.
            reflow: Rebuild line folds when the width changes. Windows ConPTY
                sends its own reflow sequences, so that path updates geometry
                and screen origin without locally refolding scrollback.
            external_reflow: The backing PTY owns reflow and may emit post-resize
                absolute cursor moves relative to its physical screen.
        """
        previous_width = self.width
        previous_height = self.height
        alternate_size_changed = self.alternate_screen and (
            (width is not None and width != self.width) or (height is not None and height != self.height)
        )
        previous_screen_cursor_line = None
        if not self.alternate_screen and height is not None:
            if self.scrollback_buffer.height <= previous_height:
                previous_screen_cursor_line = self.scrollback_buffer.cursor_line
            else:
                previous_screen_cursor_line = self.scrollback_buffer.cursor_line - self.screen_start_line_no
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height

        if reflow and (force_reflow or previous_width != width):
            self._reflow()

        if alternate_size_changed:
            self.alternate_buffer.clear(self.advance_updates())
            self.alternate_buffer.scroll_margin = ScrollMargin(None, None)

        if height is not None and previous_screen_cursor_line is not None:
            if force_reflow and self.scrollback_buffer.height <= height:
                self._screen_start_line_no = 0
            else:
                visible_cursor_line = clamp(previous_screen_cursor_line, 0, height - 1)
                self._screen_start_line_no = max(0, self.scrollback_buffer.cursor_line - visible_cursor_line)
            self._pending_primary_cursor_reanchor = external_reflow

    def key_event_to_stdin(self, event: events.Key) -> str | None:
        """Get the stdin string for a key event.

        This will depend on the terminal state.

        Args:
            event: Key event.

        Returns:
            A string to be sent to stdin, or `None` if no key was produced.
        """
        if self.cursor_keys and (sequence := CURSOR_KEYS_APPLICATION.get(event.key)) is not None:
            return sequence

        if (mapped_key := TERMINAL_KEY_MAP.get(event.key)) is not None:
            return mapped_key
        if event.character:
            return event.character
        return None

    def key_escape(self) -> str:
        """Generate the escape sequence for the escape key.

        Returns:
            str: ANSI escape sequences.
        """
        return "\x1b"

    def remove_trailing_blank_lines_from_scrollback(self) -> None:
        """Remove blank lines at the end of the scrollback buffer.

        A line is considered blank if it is whitespace and has no color or style applied.

        """
        buffer = self.scrollback_buffer
        while buffer.lines:
            last_line_content = buffer.lines[-1].content
            if last_line_content.spans or last_line_content.plain.rstrip():
                break
            buffer.remove_last_line()

    def _reflow(self) -> None:
        buffer = self.buffer
        if not buffer.lines:
            return

        buffer._updated_lines = None
        # Unfolded cursor position
        cursor_line, cursor_offset = buffer.cursor

        buffer.folded_lines.clear()
        buffer.line_to_fold.clear()
        width = self.width

        for line_no, line_record in enumerate(buffer.lines):
            line_expanded_tabs = line_record.content.expand_tabs(8)
            line_record.folds[:] = self._fold_line(line_no, line_expanded_tabs, width)
            line_record.updates = self.advance_updates()
            buffer.line_to_fold.append(len(buffer.folded_lines))
            buffer.folded_lines.extend(line_record.folds)

        # After reflow, we need to work out where the cursor is within the folded lines
        # cursor_line = min(cursor_line, len(buffer.lines) - 1)
        if cursor_line >= len(buffer.lines):
            buffer.cursor_line = len(buffer.lines)
            buffer.cursor_offset = 0
        else:
            line = buffer.lines[cursor_line]
            fold_cursor_line = buffer.line_to_fold[cursor_line]

            fold_cursor_offset = 0
            for fold in reversed(line.folds):
                if cursor_offset >= fold.offset:
                    fold_cursor_line += fold.line_offset
                    char_in_fold = cursor_offset - fold.offset
                    fold_cursor_offset = _char_offset_to_cell(fold.content, char_in_fold)
                    break

            buffer.cursor_line = fold_cursor_line
            buffer.cursor_offset = fold_cursor_offset

    async def write(self, text: str, *, hide_output: bool = False) -> tuple[set[int] | None, set[int] | None]:
        """Write to the terminal.

        Args:
            text: Text to write.
            hide_output: Hide visible output from buffers.

        Returns:
            A pair of deltas or `None for full refresh, for scrollback and alternate screen.
        """
        alternate_buffer = self.alternate_buffer
        scrollback_buffer = self.scrollback_buffer

        # Reset updated lines delta
        alternate_buffer._updated_lines = set()
        scrollback_buffer._updated_lines = set()
        # Write sequences and update
        if hide_output:
            for ansi_command in self._ansi_stream.feed(text):
                if not isinstance(ansi_command, (ANSIContent, ANSICursor)):
                    await self._handle_ansi_command(ansi_command)
        else:
            for ansi_command in self._ansi_stream.feed(text):
                await self._handle_ansi_command(ansi_command)

        # Get deltas
        scrollback_updates = (
            None if scrollback_buffer._updated_lines is None else scrollback_buffer._updated_lines.copy()
        )
        alternate_updates = None if alternate_buffer._updated_lines is None else alternate_buffer._updated_lines.copy()
        # Reset deltas
        self.alternate_buffer._updated_lines = set()
        self.scrollback_buffer._updated_lines = set()
        self._clamp_primary_screen_to_cursor()
        # Return deltas accumulated during write
        return (scrollback_updates, alternate_updates)

    def _clamp_primary_screen_to_cursor(self) -> None:
        """Keep the primary cursor inside the visible physical screen."""
        if self.alternate_screen:
            return
        max_cursor_line = self.screen_start_line_no + self.height - 1
        if self.scrollback_buffer.cursor_line > max_cursor_line:
            self._screen_start_line_no = self.scrollback_buffer.cursor_line - self.height + 1
        elif self.scrollback_buffer.cursor_line < self.screen_start_line_no:
            self._screen_start_line_no = self.scrollback_buffer.cursor_line

    def _reanchor_primary_cursor_after_resize(self, buffer: Buffer, absolute_y: int) -> None:
        """Align the first post-resize primary CUP with the known cursor row."""
        if not self._pending_primary_cursor_reanchor:
            return
        self._pending_primary_cursor_reanchor = False
        if buffer.name != "scrollback" or buffer.cursor_line >= buffer.height:
            return

        next_cursor_line = self.screen_start_line_no + max(0, absolute_y)
        if next_cursor_line != buffer.cursor_line:
            self._screen_start_line_no = max(0, buffer.cursor_line - max(0, absolute_y))

    def get_cursor_line_offset(self, buffer: Buffer) -> int:
        """The cursor *character* offset within the un-folded line.

        ``buffer.cursor_offset`` is cell-based; this method converts it back
        to a character offset suitable for Content slicing.
        """
        cursor_folded_line = buffer.folded_lines[buffer.cursor_line]
        cursor_line_offset = cursor_folded_line.line_offset
        line_no = cursor_folded_line.line_no
        line = buffer.lines[line_no]
        position = 0
        for folded_line_offset, folded_line in enumerate(line.folds):
            if folded_line_offset == cursor_line_offset:
                position += _cell_offset_to_char(folded_line.content, buffer.cursor_offset)
                break
            position += len(folded_line.content)
        return position

    def clear_buffer(self, clear: ClearType) -> None:
        buffer = self.buffer
        if clear == "screen":
            buffer._updated_lines = None
            buffer.clear(self.advance_updates())
            if buffer.name == "scrollback":
                self._screen_start_line_no = 0
            # for _ in range(self.height):
            #     self.add_line(buffer, EMPTY_CONTENT)
        elif clear == "cursor_to_end":
            buffer._updated_lines = None
            folded_cursor_line = buffer.cursor_line
            cursor_line, cursor_line_offset = buffer.cursor
            while buffer.cursor_line >= len(buffer.folded_lines):
                self.add_line(buffer, EMPTY_LINE)
            line = buffer.lines[cursor_line]
            del buffer.lines[cursor_line + 1 :]
            del buffer.line_to_fold[cursor_line + 1 :]
            del buffer.folded_lines[folded_cursor_line + 1 :]
            self.update_line(buffer, cursor_line, line.content[:cursor_line_offset])
        else:
            # print(f"TODO: clear_buffer({clear!r})")
            buffer._updated_lines = None
            buffer.clear(self.advance_updates())

    def scroll_buffer(self, direction: int, lines: int) -> None:
        """Scroll the buffer.

        Args:
            direction: +1 for down, -1 for up.
            lines: Number of lines.
        """
        buffer = self.buffer
        margin_top, margin_bottom = buffer.scroll_margin.get_line_range(self.height)

        gutter_lines = 0 if buffer.name == "alternate" else self.screen_start_line_no

        if direction == -1:
            # up (first in test)
            for line_no in range(margin_top, margin_bottom + 1):
                copy_line_no = line_no + lines
                copy_content = EMPTY_CONTENT
                copy_style = NULL_STYLE
                if copy_line_no <= margin_bottom:
                    try:
                        copy_line = buffer.lines[copy_line_no + gutter_lines]
                    except IndexError:
                        pass
                    else:
                        copy_content = copy_line.content
                        copy_style = copy_line.style

                self.update_line(buffer, line_no + gutter_lines, copy_content, copy_style)
        else:
            # down
            for line_no in reversed(range(margin_top, margin_bottom + 1)):
                copy_line_no = line_no - lines
                copy_content = EMPTY_CONTENT
                copy_style = NULL_STYLE
                if copy_line_no >= margin_top:
                    try:
                        copy_line = buffer.lines[copy_line_no + gutter_lines]
                    except IndexError:
                        pass
                    else:
                        copy_content = copy_line.content
                        copy_style = copy_line.style
                self.update_line(buffer, line_no + gutter_lines, copy_content, copy_style)

    @classmethod
    def _expand_content(cls, content: Content, offset: int, style: Style) -> Content:
        """Expand content to be at least as long as a given offset.

        Args:
            content: Content to expand.
            offset: Offset within the content.
            style: Style of padding.

        Returns:
            New Content.
        """
        if offset > len(content):
            content += Content.blank(offset - len(content), style)
        return content

    async def _handle_ansi_command(self, ansi_command: ANSICommand) -> None:
        if isinstance(ansi_command, ANSINewLine):
            if self.alternate_screen:
                # New line behaves differently in alternate screen
                ansi_command = ANSICursor(delta_y=+1, auto_scroll=True)
            else:
                ansi_command = ANSICursor(delta_y=+1, absolute_x=0, auto_scroll=True)
        elif isinstance(ansi_command, ANSIHorizontalTab):
            tab_width = 8
            ansi_command = ANSICursor(delta_x=tab_width - self.buffer.cursor_offset % tab_width)

        match ansi_command:
            case ANSIBell():
                # The embedded terminal consumes BEL locally. Forwarding it
                # through Textual would make the host terminal badge/bounce,
                # and storing it as content would replay that side effect.
                pass

            case ANSIStyle(style):
                self.style = style

            case ANSIContent(text):
                self._pending_primary_cursor_reanchor = False
                buffer = self.buffer
                folded_lines = buffer.folded_lines
                while buffer.cursor_line >= len(folded_lines):
                    self.add_line(buffer, EMPTY_LINE)
                folded_line = folded_lines[buffer.cursor_line]
                previous_content = folded_line.content
                line_no = folded_line.line_no
                line = buffer.lines[line_no]

                cursor_line_offset = self.get_cursor_line_offset(buffer)
                line_content = line.content
                cursor_cell = _fold_start_cell(line_content, folded_line) + buffer.cursor_offset
                # Cursor may be past end-of-line in cell space (e.g. CUP into
                # empty columns past current content). _cell_offset_to_char
                # clamps to len(content), so close the gap with blank cells
                # before writing — otherwise the new char overwrites the last
                # existing char and every subsequent column shifts left.
                gap_cells = cursor_cell - line_content.cell_length
                if gap_cells > 0:
                    line_content = line_content + Content.blank(gap_cells, line.style)
                    cursor_line_offset = len(line_content)
                content = Content.styled(
                    self.dec_state.translate(text),
                    self.style,
                    strip_control_codes=False,
                )
                if self.replace_mode:
                    updated_line = _replace_cells(line_content, cursor_cell, content, line.style)
                else:
                    updated_line = Content.assemble(
                        line_content[:cursor_line_offset],
                        content,
                        line_content[cursor_line_offset:],
                        strip_control_codes=False,
                    )
                self.update_line(buffer, line_no, updated_line)
                buffer.update_cursor(line_no, cursor_line_offset + len(content))
                buffer.updates = self.advance_updates()

            case ANSICursor(
                delta_x,
                delta_y,
                absolute_x,
                absolute_y,
                erase,
                clear_range,
                _relative,
                update_background,
                auto_scroll,
            ):
                buffer = self.buffer
                folded_lines = buffer.folded_lines
                while buffer.cursor_line >= len(folded_lines):
                    self.add_line(buffer, EMPTY_LINE)

                if auto_scroll and delta_y is not None:
                    if buffer.name == "scrollback" and buffer.scroll_margin == ScrollMargin(None, None) and delta_y > 0:
                        target_line = buffer.cursor_line + delta_y
                        if target_line >= buffer.height:
                            # Hard newlines at the bottom of buffered content
                            # create new primary rows even when the viewport has
                            # blank space below after a vertical resize.
                            for _ in range(target_line - buffer.height + 1):
                                self.add_line(buffer, EMPTY_LINE)
                        if target_line >= self.screen_start_line_no + self.height:
                            self._screen_start_line_no = target_line - self.height + 1
                    else:
                        margins = buffer.scroll_margin.get_line_range(self.height)
                        margin_top, margin_bottom = margins

                        screen_cursor_line = buffer.cursor_line - self.screen_start_line_no

                        if screen_cursor_line >= margin_top and screen_cursor_line <= margin_bottom:
                            scroll_cursor = screen_cursor_line + delta_y
                            if scroll_cursor > margin_bottom:
                                self.scroll_buffer(-1, 1)
                                if absolute_x is not None:
                                    buffer.cursor_offset = clamp(absolute_x, 0, self.width - 1)
                                    buffer.update_line(buffer.cursor_line)
                                return
                            if scroll_cursor < margin_top:
                                self.scroll_buffer(+1, 1)
                                if absolute_x is not None:
                                    buffer.cursor_offset = clamp(absolute_x, 0, self.width - 1)
                                    buffer.update_line(buffer.cursor_line)
                                return

                folded_line = folded_lines[buffer.cursor_line]
                previous_content = folded_line.content
                line = buffer.lines[folded_line.line_no]
                if update_background:
                    line.style = self.style

                if clear_range is not None:
                    cursor_line_offset = self.get_cursor_line_offset(buffer)

                    line_content = line.content
                    row_start_cell = _fold_start_cell(line_content, folded_line)
                    cursor_cell = row_start_cell + buffer.cursor_offset
                    gap_cells = cursor_cell - line_content.cell_length
                    if gap_cells > 0:
                        line_content = line_content + Content.blank(gap_cells, line.style)
                        cursor_line_offset = len(line_content)
                    row_end_cell = row_start_cell + self.width - 1

                    # Start and end replace are *inclusive* cell offsets.
                    if ansi_command.relative:
                        clear_start, clear_end = ansi_command.get_clear_offsets(cursor_cell, row_end_cell + 1)
                    else:
                        clear_start, clear_end = ansi_command.get_clear_offsets(
                            cursor_cell - row_start_cell,
                            self.width,
                        )
                        clear_start += row_start_cell
                        clear_end += row_start_cell
                    clear_start = clamp(clear_start, row_start_cell, row_end_cell)
                    clear_end = clamp(clear_end, clear_start, row_end_cell)
                    char_start = _cell_offset_to_char(line_content, clear_start)
                    char_end = _cell_offset_to_char(line_content, clear_end + 1)

                    before_clear = line_content[:char_start]
                    after_clear = line_content[char_end:]

                    if erase:
                        updated_line = Content.assemble(
                            before_clear,
                            after_clear,
                            strip_control_codes=False,
                        )
                    else:
                        blank_width = clear_end - clear_start + 1
                        updated_line = Content.assemble(
                            before_clear,
                            Content.blank(blank_width, self.style),
                            after_clear,
                            strip_control_codes=False,
                        )
                    self.update_line(buffer, folded_line.line_no, updated_line)

                if not previous_content.is_same(folded_line.content):
                    buffer.updates = self.advance_updates()

                if delta_x is not None:
                    buffer.cursor_offset = clamp(buffer.cursor_offset + delta_x, 0, self.width - 1)
                    buffer.update_line(buffer.cursor_line)
                if absolute_x is not None:
                    buffer.cursor_offset = clamp(absolute_x, 0, self.width - 1)
                    buffer.update_line(buffer.cursor_line)

                current_cursor_line = buffer.cursor_line
                if delta_y is not None:
                    buffer.update_line(buffer.cursor_line)
                    min_cursor_line = 0 if buffer.name == "alternate" else self.screen_start_line_no
                    max_cursor_line = min_cursor_line + self.height - 1
                    buffer.cursor_line = clamp(buffer.cursor_line + delta_y, min_cursor_line, max_cursor_line)
                    buffer.update_line(buffer.cursor_line)
                if absolute_y is not None:
                    buffer.update_line(buffer.cursor_line)
                    if buffer.name == "scrollback":
                        self._reanchor_primary_cursor_after_resize(buffer, absolute_y)
                        buffer.cursor_line = self.screen_start_line_no + max(0, absolute_y)
                    else:
                        buffer.cursor_line = max(0, absolute_y)
                    buffer.update_line(buffer.cursor_line)

                if current_cursor_line != buffer.cursor_line:
                    # Simplify when the cursor moves away from the current line
                    line.content.simplify()  # Reduce segments
                    self._line_updated(buffer, current_cursor_line)
                    self._line_updated(buffer, buffer.cursor_line)

            case ANSIFeatures() as features:
                if features.show_cursor is not None:
                    if features.show_cursor != self.show_cursor:
                        self._line_updated(self.buffer, self.buffer.cursor_line)
                    self.show_cursor = features.show_cursor
                    self._line_updated(self.buffer, self.buffer.cursor_line)
                if features.alternate_screen is not None:
                    if features.alternate_screen != self.alternate_screen:
                        self.scrollback_buffer._updated_lines = None
                        self.alternate_buffer._updated_lines = None
                        if features.alternate_screen:
                            self.alternate_buffer.clear(self.advance_updates())
                            self.alternate_buffer.scroll_margin = ScrollMargin(None, None)
                        else:
                            self.alternate_buffer.clear(self.advance_updates())
                            self.alternate_buffer.scroll_margin = ScrollMargin(None, None)
                    self.alternate_screen = features.alternate_screen
                if features.bracketed_paste is not None:
                    self.bracketed_paste = features.bracketed_paste
                if features.cursor_blink is not None:
                    self.cursor_blink = features.cursor_blink
                if features.cursor_keys is not None:
                    self.cursor_keys = features.cursor_keys
                if features.replace_mode is not None:
                    self.replace_mode = features.replace_mode
                if features.auto_wrap is not None:
                    self.auto_wrap = features.auto_wrap
                self.advance_updates()

            case ANSIClear(clear):
                self.clear_buffer(clear)

            case ANSIScrollMargin(top, bottom):
                self.buffer.scroll_margin = ScrollMargin(top, bottom)
                # Setting the scroll margins moves the cursor to (1, 1)
                buffer = self.buffer
                self._line_updated(buffer, buffer.cursor_line)
                buffer.cursor_line = 0
                buffer.cursor_offset = 0
                self._line_updated(buffer, buffer.cursor_line)

            case ANSIScroll(direction, lines):
                self.scroll_buffer(direction, lines)

            case ANSICharacterSet(dec, dec_invoke):
                self.dec_state.update(dec, dec_invoke)

            case ANSIWorkingDirectory(path):
                self.current_directory = path

            case ANSIShellCommand(command):
                self.last_command = command

            case ANSIMouseTracking(tracking, format, focus_events, alternate_scroll):
                if tracking == "none":
                    self.mouse_tracking = None
                    return
                # Only create a new MouseTracking if we're actually enabling
                # a tracking mode. Format-only changes (like ESC[?1006l) should
                # not resurrect a disabled mouse_tracking.
                if (mouse_tracking := self.mouse_tracking) is None:
                    if tracking is None:
                        return
                    mouse_tracking = self.mouse_tracking = MouseTracking()
                if tracking is not None:
                    mouse_tracking.tracking = tracking
                if format is not None:
                    mouse_tracking.format = format
                if focus_events is not None:
                    mouse_tracking.focus_events = focus_events
                if alternate_scroll is not None:
                    mouse_tracking.alternate_scroll = alternate_scroll

            case ANSICursorSave():
                buffer = self.buffer
                self._materialize_cursor_line(buffer)
                self._saved_cursor = buffer.cursor

            case ANSICursorRestore():
                if self._saved_cursor is not None:
                    line_no, offset = self._saved_cursor
                    buffer = self.buffer
                    self._line_updated(buffer, buffer.cursor_line)
                    while line_no >= len(buffer.lines):
                        self.add_line(buffer, EMPTY_LINE)
                    buffer.update_cursor(line_no, offset)
                    self._line_updated(buffer, buffer.cursor_line)

            case ANSICursorPositionRequest():
                if self.alternate_screen:
                    row = self.buffer.cursor_line + 1
                else:
                    row = clamp(self.buffer.cursor_line - self.screen_start_line_no + 1, 1, self.height)
                column = self.buffer.cursor_offset + 1
                await self.write_stdin(f"\x1b[{row};{column}R")

            case ANSIDeviceAttributesRequest(secondary=secondary):
                if secondary:
                    # Secondary DA: report as VT220-compatible
                    await self.write_stdin("\x1b[>1;0;0c")
                else:
                    # Primary DA: report VT420 with basic capabilities
                    await self.write_stdin("\x1b[?65;1;9c")

            case ANSIModeStatusRequest(mode=mode, private=True):
                # Report mode status: 1=set, 2=reset, 0=not recognized
                mode_states = {
                    1: self.cursor_keys,  # DECCKM
                    7: self.auto_wrap,  # DECAWM
                    12: self.cursor_blink,  # cursor blink
                    25: self.show_cursor,  # DECTCEM
                    47: self.alternate_screen,  # alternate screen (legacy)
                    1047: self.alternate_screen,  # alternate screen (legacy)
                    1049: self.alternate_screen,  # alternate screen
                    2004: self.bracketed_paste,  # bracketed paste
                }
                state = (1 if mode_states[mode] else 2) if mode in mode_states else 0
                await self.write_stdin(f"\x1b[?{mode};{state}$y")

            case _:
                pass

    def _materialize_cursor_line(self, buffer: Buffer) -> None:
        """Ensure the backing buffer contains the current folded cursor row."""
        while buffer.cursor_line >= len(buffer.folded_lines):
            self.add_line(buffer, EMPTY_LINE)

    def _line_updated(self, buffer: Buffer, line_no: int) -> None:
        """Mark a folded line as having been updated.

        Args:
            buffer: Buffer to use.
            line_no: Folded line number to mark as updated.
        """
        try:
            folded_line = buffer.folded_lines[line_no]
            buffer.lines[folded_line.line_no].updates = self.advance_updates()
            if buffer._updated_lines is not None:
                buffer._updated_lines.add(line_no)
        except IndexError:
            pass

    def _fold_line(self, line_no: int, line: Content, width: int) -> list[LineFold]:
        updates = self._updates
        if not self.auto_wrap:
            return [LineFold(line_no, 0, 0, line, updates)]
        if not width:
            return [LineFold(0, 0, 0, line, updates)]
        line_length = line.cell_length
        if line_length <= width:
            return [LineFold(line_no, 0, 0, line, updates)]

        folded_lines = line.fold(width)
        offsets = [0, *accumulate(len(line) for line in folded_lines)][:-1]
        folds = [
            LineFold(line_no, line_offset, offset, folded_line, updates)
            for line_offset, (offset, folded_line) in enumerate(zip(offsets, folded_lines))
        ]
        assert len(folds)
        return folds

    def add_line(self, buffer: Buffer, content: Content, style: Style = NULL_STYLE) -> None:
        updates = self.advance_updates()
        line_no = buffer.line_count
        width = self.width
        line_record = LineRecord(
            content,
            style,
            self._fold_line(line_no, content, width),
            updates,
        )
        buffer.lines.append(line_record)
        folds = line_record.folds
        buffer.line_to_fold.append(len(buffer.folded_lines))
        fold_count = len(buffer.folded_lines)
        if buffer._updated_lines is not None:
            buffer._updated_lines.update(range(fold_count, fold_count + len(folds)))
        buffer.folded_lines.extend(folds)
        buffer.updates = updates

    def update_line(self, buffer: Buffer, line_index: int, line: Content, style: Style | None = None) -> None:
        """Update a line (potentially refolding and moving subsequencte lines down).

        Args:
            buffer: Buffer.
            line_index: Line index (unfolded).
            line: New line content.
            style: New background style, or `None` not to update.
        """
        while line_index >= len(buffer.lines):
            self.add_line(buffer, EMPTY_LINE)

        line_expanded_tabs = line.expand_tabs(8)
        buffer.max_line_width = max(line_expanded_tabs.cell_length, buffer.max_line_width)
        line_record = buffer.lines[line_index]
        old_fold_count = len(line_record.folds)
        line_record.content = line
        if style is not None:
            line_record.style = style
        new_folds = self._fold_line(line_index, line_expanded_tabs, self.width)
        line_record.folds[:] = new_folds
        line_record.updates = self.advance_updates()

        if (
            buffer.name == "alternate"
            and old_fold_count == 1
            and len(new_folds) == 1
            and line_index < len(buffer.line_to_fold)
        ):
            fold_start = buffer.line_to_fold[line_index]
            if fold_start < len(buffer.folded_lines):
                buffer.folded_lines[fold_start] = new_folds[0]
                if buffer._updated_lines is not None:
                    buffer._updated_lines.add(fold_start)
                return

        if old_fold_count != len(new_folds):
            buffer._updated_lines = None
        elif buffer._updated_lines is not None:
            fold_start = buffer.line_to_fold[line_index]
            buffer._updated_lines.update(range(fold_start, fold_start + len(line_record.folds)))

        fold_line = buffer.line_to_fold[line_index]
        del buffer.line_to_fold[line_index:]
        del buffer.folded_lines[fold_line:]

        for line_no in range(line_index, buffer.line_count):
            line_record = buffer.lines[line_no]
            buffer.line_to_fold.append(len(buffer.folded_lines))
            for fold in line_record.folds:
                buffer.folded_lines.append(fold)
