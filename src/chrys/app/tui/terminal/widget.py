# Copyright (c) 2026 Chrys. All rights reserved.
# Adapted from toad (https://github.com/batrachianai/toad).

import asyncio
import platform
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from time import monotonic

from textual import events, on
from textual.cache import LRUCache
from textual.dom import NoScreen
from textual.geometry import Region, Size
from textual.message import Message
from textual.reactive import reactive
from textual.scroll_view import ScrollView
from textual.selection import Selection
from textual.strip import Strip
from textual.style import Style
from textual.timer import Timer

from chrys.app.tui.clipboard import copy_text_to_clipboards
from chrys.app.tui.support.gc_freeze import detach_lru_cache, renew_lru_cache
from chrys.app.tui.terminal import ansi
from chrys.foundation.branding import APP_DISPLAY_NAME

# Time required to double tab escape
ESCAPE_TAP_DURATION = 400 / 1000

# Debounce interval for render cache clear + refresh during resize.
# On Windows, ConPTY set_size() emits VT reflow sequences that trigger
# additional refresh cycles — the debounce prevents redundant renders.
_RESIZE_REFRESH_DEBOUNCE_S = 0.033 if platform.system() == "Windows" else 0.0
_FOCUS_REPORTS = {"\x1b[I", "\x1b[O"}
_MAX_PENDING_ESCAPE_SEQUENCE = 32
# Numeric CSI reports cover SGR/urxvt-style mouse protocols. X10 is ESC [ M
# followed by three bytes, so it needs a separate arm in the pattern.
_MOUSE_REPORT_RE = re.compile(r"^(?:\x1b\[(?:<)?-?\d+(?:;-?\d+){1,2}[mM]|\x1b\[M.{3})$")
_MOUSE_REPORT_PREFIX_RE = re.compile(r"^\x1b\[(?:<)?[-\d;]*$")
_X10_MOUSE_REPORT_PREFIX = "\x1b[M"
_X10_MOUSE_REPORT_LEN = len(_X10_MOUSE_REPORT_PREFIX) + 3


class Terminal(ScrollView, can_focus=True):
    BINDING_GROUP_TITLE = "Terminal"
    HELP = f"""\
## Terminal

An embedded terminal running within {APP_DISPLAY_NAME}.
When the terminal has focus, it will take over the handling of keys.

Tap escape *twice* to exit.

"""

    CURSOR_STYLE = Style.parse("reverse")

    hide_cursor = reactive(False)

    @dataclass
    class Finalized(Message):
        """Terminal was finalized."""

        terminal: Terminal

        @property
        def control(self) -> Terminal:
            return self.terminal

    @dataclass
    class AlternateScreenChanged(Message):
        """Terminal enabled or disabled alternate screen."""

        terminal: Terminal
        enabled: bool

        @property
        def control(self) -> Terminal:
            return self.terminal

    @dataclass
    class LongRunning(Message):
        """Terminal has been running for a while."""

        terminal: Terminal

        @property
        def control(self) -> Terminal:
            return self.terminal

    @dataclass
    class SizeChanged(Message):
        """Terminal dimensions changed (parent should resize PTY)."""

        terminal: Terminal
        width: int
        height: int

        @property
        def control(self) -> Terminal:
            return self.terminal

    @dataclass
    class EscapeExited(Message):
        """User double-tapped Escape to exit the terminal."""

        terminal: Terminal

        @property
        def control(self) -> Terminal:
            return self.terminal

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        minimum_terminal_width: int = 0,
        size: tuple[int, int] | None = None,
        get_terminal_dimensions: Callable[[], tuple[int, int]] | None = None,
    ):
        super().__init__(
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )
        self.set_reactive(Terminal.auto_links, False)
        self.minimum_terminal_width = minimum_terminal_width
        self._get_terminal_dimensions = get_terminal_dimensions

        self.state = ansi.TerminalState(self.write_process_stdin)

        if size is None:
            self._width = minimum_terminal_width or 80
            self._height: int = 24
        else:
            width, height = size
            self._width = width
            self._height = height

        self.minimum_terminal_width = self._width

        self.max_window_width = 0
        self._escape_time = monotonic()
        self._escaping = False
        self._escape_reset_timer: Timer | None = None
        self._pending_escape_sequence: str | None = None
        self._finalized: bool = False
        self.current_directory: str | None = None
        self.last_command: str | None = None
        self._alternate_screen: bool = False
        self._terminal_render_cache: LRUCache[tuple, Strip] = LRUCache(1024)
        self._write_to_stdin: Callable[[str], Awaitable] | None = None
        self._write_count = 0
        self._long_running_timer: Timer | None = None
        self._external_reflow: bool = False
        """When True, skip the terminal emulator's own _reflow() on resize.
        Set by Shell on Windows where ConPTY sends its own reflow sequences."""
        self._resize_refresh_timer: Timer | None = None
        self._resize_in_progress: bool = False
        self._pending_mouse_move: events.MouseMove | None = None
        self._mouse_flush_task: asyncio.Task[None] | None = None

    @property
    def is_finalized(self) -> bool:
        """Finalized terminals will not accept writes or receive input."""
        return self._finalized

    @property
    def width(self) -> int:
        """Width of the terminal."""
        return self._width

    @property
    def height(self) -> int:
        """Height of the terminal."""
        height = self._height
        return height

    @property
    def size(self) -> Size:
        return Size(self.width, self.height)

    @property
    def alternate_screen(self) -> bool:
        return self._alternate_screen

    @property
    def allow_vertical_scroll(self) -> bool:
        return False if self._alternate_screen else super().allow_vertical_scroll

    @property
    def allow_horizontal_scroll(self) -> bool:
        return False if self._alternate_screen else super().allow_horizontal_scroll

    def notify_style_update(self) -> None:
        """Clear cache when theme chages."""
        self.clear_render_cache()
        super().notify_style_update()

    def clear_render_cache(self) -> None:
        """Clear derived terminal strips without touching ANSI scrollback."""
        self._terminal_render_cache.clear()

    def detach_render_cache(self) -> None:
        """Detach the cyclic render LRU before a GC freeze action."""
        # gc-freeze swaps in an acyclic capacity token; renew restores a real cache before next use.
        self._terminal_render_cache = detach_lru_cache(self._terminal_render_cache)  # ty: ignore[invalid-assignment]

    def renew_render_cache(self) -> None:
        """Recreate the cyclic render LRU after a GC freeze action."""
        self._terminal_render_cache = renew_lru_cache(self._terminal_render_cache)

    def set_state(self, state: ansi.TerminalState) -> None:
        """Set the terminal state, if this terminal is to inherit an existing state.

        Args:
            state: Terminal state object.
        """
        self.state = state

    def set_write_to_stdin(self, write_to_stdin: Callable[[str], Awaitable]) -> None:
        """Set a callable which is invoked with input, to be sent to stdin.

        Args:
            write_to_stdin: Callable which takes a string.
        """
        self._write_to_stdin = write_to_stdin

    def finalize(self) -> None:
        """FInalize the terminal.

        The finalized terminal will reject new writes.
        Adds the TCSS class `-finalize`
        """
        if not self._finalized:
            if self._long_running_timer is not None:
                self._long_running_timer.stop()
            self._finalized = True
            self.state.show_cursor = False
            self.add_class("-finalized")
            self._terminal_render_cache.clear()
            self.refresh()
            self.blur()
            self.post_message(self.Finalized(self))
            if not self.state.buffer.height:
                self.display = False

    def allow_focus(self) -> bool:
        """Prohibit focus when the terminal is finalized and couldn't accept input."""
        return not self.is_finalized

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Get the text under the selection.

        Args:
            selection: Selection information.

        Returns:
            Tuple of extracted text and ending (typically "\n" or " "), or `None` if no text could be extracted.
        """
        text = "\n".join(line_record.content.plain for line_record in self.state.buffer.lines)
        return selection.extract(text), "\n"

    def _on_resize(self, event: events.Resize) -> None:
        if self._get_terminal_dimensions is None:
            width, height = self.scrollable_content_region.size
        else:
            width, height = self._get_terminal_dimensions()
        self.update_size(width, height)

    def update_size(self, width: int, height: int, *, immediate: bool = False, force_reflow: bool = False) -> None:
        # Hidden widgets may report zero-size layout; keep the last real terminal
        # geometry so PTY/ConPTY cursor state does not desync while shell mode is hidden.
        if width <= 0 or height <= 0:
            return

        old_width = self._width
        old_height = self._height
        pin_to_bottom = self._is_following_scrollback_bottom(old_height)

        self._terminal_render_cache.grow(height * 2)
        self._width = width
        self._height = height
        self._width = max(self._width, self.minimum_terminal_width)

        if old_width != self._width or (old_height != self._height and not self.is_finalized):
            self.post_message(self.SizeChanged(self, self._width, self._height))

        if self._external_reflow and not force_reflow:
            # ConPTY sends its own reflow VT sequences on resize — skip
            # the terminal emulator's _reflow() to avoid double-reflow
            # artifacts.  Just update dimensions so incoming sequences are
            # interpreted at the correct width.
            self.state.update_size(self._width, height, reflow=False, external_reflow=True)
        else:
            self.state.update_size(
                self._width,
                height,
                force_reflow=force_reflow,
                external_reflow=self._external_reflow,
            )

        self._sync_viewport_from_state(pin_to_bottom=pin_to_bottom)

        if immediate or not _RESIZE_REFRESH_DEBOUNCE_S:
            # No debounce (Unix, or initial mount) — apply immediately
            self._terminal_render_cache.clear()
            self.refresh()
        else:
            # Debounce the expensive cache clear + refresh on Windows to
            # avoid redundant render cycles during drag-resize / ConPTY
            # reflow storms.
            self._resize_in_progress = True
            if self._resize_refresh_timer is not None:
                self._resize_refresh_timer.stop()
            self._resize_refresh_timer = self.set_timer(_RESIZE_REFRESH_DEBOUNCE_S, self._apply_resize_refresh)

    def _apply_resize_refresh(self) -> None:
        """Deferred cache clear + refresh after resize debounce settles."""
        self._resize_refresh_timer = None
        self._resize_in_progress = False
        self._terminal_render_cache.clear()
        self.refresh()

    def _is_following_scrollback_bottom(self, visible_height: int) -> bool:
        """Return True when scrollback was pinned to the terminal prompt before a resize."""
        if self.state.alternate_screen:
            return False
        bottom_scroll_y = max(0, self.state.scrollback_buffer.height - visible_height)
        return self.scroll_y >= bottom_scroll_y

    def _sync_viewport_from_state(self, *, pin_to_bottom: bool = False) -> None:
        """Keep ScrollView's virtual canvas aligned with the terminal buffers."""
        if self.state.alternate_screen:
            visible_w, visible_h = self.scrollable_content_region.size
            new_virtual_size = Size(visible_w, visible_h)
            if self.virtual_size != new_virtual_size:
                self.virtual_size = new_virtual_size
            if self.scroll_y != 0:
                self.scroll_y = 0
            return

        width = self.state.width
        visible_h = self.scrollable_content_region.size.height
        scrollback_h = self.state.scrollback_buffer.height
        height = (
            scrollback_h
            if scrollback_h <= visible_h
            else max(scrollback_h, self.state.screen_start_line_no + visible_h)
        )
        self.virtual_size = Size(min(self.state.buffer.max_line_width, width), height)
        if pin_to_bottom:
            self.scroll_y = self.max_scroll_y

    def _pin_scroll_to_bottom(self) -> None:
        """Move the scroll view to the current bottom without animation."""
        self._sync_viewport_from_state(pin_to_bottom=True)

    def on_mount(self) -> None:
        self.anchor()
        if self._get_terminal_dimensions is None:
            width, height = self.scrollable_content_region.size
        else:
            width, height = self._get_terminal_dimensions()
        self.update_size(width, height, immediate=True)

    async def write(self, text: str, hide_output: bool = False) -> bool:
        """Write sequences to the terminal.

        Args:
            text: Text with ANSI escape sequences.
            hide_output: Do not update the buffers with visible text.

        Returns:
            `True` if the state visuals changed, `False` if no visual change.
        """
        if self._write_count and self._long_running_timer is None:

            def warn_long_run():
                """Warn about a long running command."""
                self.post_message(self.LongRunning(self))

            self._long_running_timer = self.set_timer(2, warn_long_run)
        self._write_count += 1

        scrollback_delta, alternate_delta = await self.state.write(text, hide_output=hide_output)
        self._update_from_state(scrollback_delta, alternate_delta)
        scrollback_changed = bool(scrollback_delta is None or scrollback_delta)
        alternate_changed = bool(alternate_delta is None or alternate_delta)

        if self._alternate_screen != self.state.alternate_screen:
            self.post_message(self.AlternateScreenChanged(self, enabled=self.state.alternate_screen))
        self._alternate_screen = self.state.alternate_screen
        return scrollback_changed or alternate_changed

    def on_click(self, event: events.Click) -> None:
        self.focus()
        event.stop()

    def _update_from_state(self, scrollback_delta: set[int] | None, alternate_delta: set[int] | None) -> None:
        if self.state.current_directory:
            self.current_directory = self.state.current_directory
            self.state.current_directory = ""
        command_submitted = bool(self.state.last_command)
        if self.state.last_command:
            self.last_command = self.state.last_command
            self.state.last_command = ""
        # During a resize burst on Windows, skip redundant refreshes —
        # the debounced _apply_resize_refresh will trigger a single
        # refresh after the resize settles.
        skip_refresh = self._resize_in_progress

        if self.state.alternate_screen:
            # Alternate screen (htop/vim/nested-TUIs): fullscreen, no
            # scrollback visible.  Use the visible content region size to
            # prevent scrollbars.
            self._sync_viewport_from_state()
            visible_h = self.scrollable_content_region.size.height

            if skip_refresh:
                return

            if alternate_delta is None:
                # Mode transition, bulk clear, or scroll-region shift —
                # repaint everything.
                self.refresh()
            elif alternate_delta:
                # Surgical refresh: re-render only the lines that actually
                # changed during this write batch.  In alt-screen, scroll_y
                # is pinned to 0 and there's no scrollback, so alternate
                # buffer line indices map 1:1 to viewport rows.  This is
                # the hot path when a nested TUI (scrolling in inner chrys
                # on Windows, etc.) streams frequent small redraws through
                # ConPTY — a full refresh here would drive CPU to 100%.
                window_width = self.region.width
                refresh_lines = [Region(0, y, window_width, 1) for y in alternate_delta if 0 <= y < visible_h]
                if refresh_lines:
                    self.refresh(*refresh_lines)
            # If alternate_delta is an empty set, nothing visible changed —
            # skip the refresh entirely.
        else:
            # Check if we're at the bottom BEFORE updating virtual_size
            # (which changes max_scroll_y).  This replaces the unreliable
            # anchor mechanism — ScrollView's anchor state can desync when
            # virtual_size is set outside the normal layout path.
            was_at_bottom = self.scroll_y >= self.max_scroll_y
            self._sync_viewport_from_state(pin_to_bottom=was_at_bottom or command_submitted)

            if skip_refresh:
                return

            scroll_y = self._visible_source_start()
            visible_h = self.scrollable_content_region.size.height
            visible_lines = frozenset(range(scroll_y, scroll_y + visible_h))

            if scrollback_delta is None:
                self.refresh()
            else:
                window_width = self.region.width
                refresh_lines = [
                    Region(0, y - scroll_y, window_width, 1) for y in sorted(scrollback_delta & visible_lines)
                ]
                if refresh_lines:
                    self.refresh(*refresh_lines)

    def render_line(self, y: int) -> Strip:
        if not self.is_attached:
            return Strip.blank(max(0, self._width or self.size.width)).apply_offsets(0, y)

        scroll_x, _scroll_y = self.scroll_offset
        strip = self._render_line(scroll_x, self._visible_source_start() + y, self._width)
        return strip

    def _visible_source_start(self) -> int:
        """Return the buffer row currently mapped to viewport row 0."""
        if self.state.scrollback_buffer.height <= self.scrollable_content_region.size.height:
            return 0
        if not self.state.alternate_screen and self.scroll_y >= self.max_scroll_y:
            return self.state.screen_start_line_no
        return int(self.scroll_y)

    def _render_line(self, x: int, y: int, width: int) -> Strip:
        selection = self._safe_text_selection()
        visual_style = self.visual_style
        rich_style = visual_style.rich_style

        state = self.state
        if state.alternate_screen:
            # Alternate screen: render directly from alternate buffer
            buffer = state.alternate_buffer
            buffer_offset = 0
        else:
            buffer = state.scrollback_buffer
            buffer_offset = 0
        # Get the folded line, which as a one to one relationship with y
        try:
            folded_line_ = buffer.folded_lines[y - buffer_offset]
            line_no, line_offset, offset, line, updates = folded_line_
        except IndexError:
            return Strip.blank(width, rich_style)

        line_record = buffer.lines[line_no]
        cache_key: tuple | None = (
            self.state.alternate_screen,
            y,
            line_record.updates,
            updates,
        )

        # Add in cursor — cursor_offset is cell-based; convert to char for stylize
        if not self.hide_cursor and state.show_cursor and buffer.cursor_line == y - buffer_offset:
            from chrys.app.tui.terminal.ansi._buffer import _cell_offset_to_char

            char_offset = _cell_offset_to_char(line, buffer.cursor_offset)
            if char_offset >= len(line):
                line = line.pad_right(char_offset - len(line) + 1)
            line = line.stylize(self.CURSOR_STYLE, char_offset, char_offset + 1)
            cache_key = None

        # get cached strip if there is no selection
        if not selection and cache_key is not None and (strip := self._terminal_render_cache.get(cache_key)):
            strip = strip.crop(x, x + width)
            strip = strip.adjust_cell_length(width, (visual_style + line_record.style).rich_style)
            strip = strip.apply_offsets(x + offset, line_no)
            return strip

        # Apply selection
        if selection is not None and (select_span := selection.get_span(line_no)):
            unfolded_content = line_record.content.expand_tabs(8)
            start, end = select_span
            if end == -1:
                end = len(unfolded_content)
            selection_style = self.screen.get_visual_style("screen--selection")
            unfolded_content = unfolded_content.stylize(selection_style, start, end)
            try:
                folded_lines = self.state._fold_line(line_no, unfolded_content, width)
                line = folded_lines[line_offset].content
                cache_key = None
            except IndexError:
                pass

        try:
            strip = Strip(line.render_segments(visual_style), cell_length=line.cell_length)
        except Exception:
            # TODO: Is this neccesary?
            strip = Strip.blank(line.cell_length)

        if cache_key is not None:
            self._terminal_render_cache[cache_key] = strip

        strip = strip.crop(x, x + width)
        strip = strip.adjust_cell_length(width, (visual_style + line_record.style).rich_style)
        strip = strip.apply_offsets(x + offset, line_no)

        return strip

    def _safe_text_selection(self) -> Selection | None:
        try:
            return self.text_selection
        except NoScreen:
            return None

    async def _reset_escaping(self) -> None:
        pending = self._pending_escape_sequence or self.state.key_escape()
        write_pending = self._escaping
        self._escaping = False
        self._pending_escape_sequence = None
        self._stop_escape_reset_timer()
        if write_pending:
            await self.write_process_stdin(pending)

    def _clear_pending_escape(self) -> None:
        self._escaping = False
        self._pending_escape_sequence = None
        self._stop_escape_reset_timer()

    def _stop_escape_reset_timer(self) -> None:
        if self._escape_reset_timer is not None:
            self._escape_reset_timer.stop()
            self._escape_reset_timer = None

    def _restart_escape_reset_timer(self) -> None:
        self._stop_escape_reset_timer()
        self._escape_reset_timer = self.set_timer(ESCAPE_TAP_DURATION, self._reset_escaping)

    def _is_outer_terminal_report(self, sequence: str) -> bool:
        """Return True for focus/mouse reports leaked from the host terminal."""
        return sequence in _FOCUS_REPORTS or _MOUSE_REPORT_RE.fullmatch(sequence) is not None

    def _could_be_outer_terminal_report(self, sequence: str) -> bool:
        if sequence == "\x1b[":
            return True
        if "\x1b[I".startswith(sequence) or "\x1b[O".startswith(sequence):
            return True
        if _X10_MOUSE_REPORT_PREFIX.startswith(sequence):
            return True
        if sequence.startswith(_X10_MOUSE_REPORT_PREFIX):
            return len(sequence) < _X10_MOUSE_REPORT_LEN
        return _MOUSE_REPORT_PREFIX_RE.fullmatch(sequence) is not None

    async def _handle_pending_escape_character(self, character: str | None) -> bool:
        """Handle a printable character that arrived after a pending Escape.

        Windows Terminal can emit focus/mouse reports as VT sequences while
        Textual is in application mode. If Textual's parser times out after
        the leading ESC, the focused shell widget receives those bytes as
        ordinary key events and would otherwise pass them through to PowerShell.
        """
        if not self._escaping or self._pending_escape_sequence is None or character is None:
            return False

        sequence = self._pending_escape_sequence + character
        if self._is_outer_terminal_report(sequence):
            self._clear_pending_escape()
            return True
        if len(sequence) > _MAX_PENDING_ESCAPE_SEQUENCE:
            self._clear_pending_escape()
            await self.write_process_stdin(sequence)
            return True
        if self._could_be_outer_terminal_report(sequence):
            self._pending_escape_sequence = sequence
            self._restart_escape_reset_timer()
            return True

        self._clear_pending_escape()
        await self.write_process_stdin(sequence)
        return True

    async def on_key(self, event: events.Key):
        # Ctrl+C: copy selected text to system clipboard, or send \x03 to PTY
        if event.key == "ctrl+c":
            event.prevent_default()
            event.stop()
            selected = self.screen.get_selected_text()
            if selected:
                copy_text_to_clipboards(self.app, selected)
            else:
                await self.write_process_stdin("\x03")
            return

        event.prevent_default()
        event.stop()

        # Escape in alternate screen → forward actual Esc to the child app
        if event.key == "escape" and self._alternate_screen:
            await self.write_process_stdin("\x1b")
            return

        # Normal escape handling (double-tap to blur)
        if event.key == "escape":
            if self._escaping:
                if monotonic() < self._escape_time + ESCAPE_TAP_DURATION:
                    self.blur()
                    self._clear_pending_escape()
                    self.post_message(self.EscapeExited(self))
                    return
                await self._reset_escaping()
            else:
                self._escaping = True
                self._pending_escape_sequence = self.state.key_escape()
                self._escape_time = monotonic()
                self._restart_escape_reset_timer()
                return
        else:
            if await self._handle_pending_escape_character(event.character):
                return
            await self._reset_escaping()

        if (stdin := self.state.key_event_to_stdin(event)) is not None:
            if stdin == "\r":
                self._pin_scroll_to_bottom()
            await self.write_process_stdin(stdin)

    @property
    def allow_select(self) -> bool:
        return self.is_finalized or not self._alternate_screen

    def _encode_mouse_event_sgr(self, event: events.MouseEvent) -> str:
        x = int(event.x)
        y = int(event.y)

        if isinstance(event, events.MouseScrollUp):
            button = 64
        elif isinstance(event, events.MouseScrollDown):
            button = 65
        elif isinstance(event, events.MouseScrollRight):
            button = 66
        elif isinstance(event, events.MouseScrollLeft):
            button = 67
        elif isinstance(event, events.MouseMove):
            button = event.button + 32 if event.button else 35
        else:
            button = event.button - 1

        if event.shift:
            button += 4
        if event.meta:
            button += 8
        if event.ctrl:
            button += 16

        if isinstance(event, events.MouseUp):
            button = 0
            final_character = "m"
        else:
            final_character = "M"
        mouse_stdin = f"\x1b[<{button};{x + 1};{y + 1}{final_character}"
        return mouse_stdin

    @on(events.MouseMove)
    async def on_mouse_move(self, event: events.MouseMove) -> None:
        if self.is_finalized:
            return
        if (mouse_tracking := self.state.mouse_tracking) is None:
            return
        if mouse_tracking.tracking == "all" or (event.button and mouse_tracking.tracking == "drag"):
            self._pending_mouse_move = event
            if self._mouse_flush_task is None or self._mouse_flush_task.done():
                self._mouse_flush_task = asyncio.create_task(self._flush_mouse_move())
            event.prevent_default()
            event.stop()

    async def _flush_mouse_move(self) -> None:
        await asyncio.sleep(1 / 60)
        if (event := self._pending_mouse_move) is None:
            return
        self._pending_mouse_move = None
        await self._handle_mouse_event(event)

    @on(events.MouseDown)
    @on(events.MouseUp)
    async def on_mouse_button(self, event: events.MouseUp | events.MouseDown) -> None:
        if self.is_finalized:
            return
        if self.state.mouse_tracking is None:
            return
        await self._handle_mouse_event(event)
        event.prevent_default()
        event.stop()

    @on(events.MouseScrollUp)
    @on(events.MouseScrollDown)
    @on(events.MouseScrollLeft)
    @on(events.MouseScrollRight)
    async def on_mouse_scroll(
        self,
        event: events.MouseScrollUp | events.MouseScrollDown | events.MouseScrollLeft | events.MouseScrollRight,
    ) -> None:
        if self.is_finalized:
            return
        if self.state.mouse_tracking is not None:
            await self._handle_mouse_event(event)
            event.prevent_default()
            event.stop()
            return
        if self.state.alternate_screen:
            if isinstance(event, events.MouseScrollUp):
                await self.write_process_stdin("\x1b[A")
            elif isinstance(event, events.MouseScrollDown):
                await self.write_process_stdin("\x1b[B")
            event.prevent_default()
            event.stop()

    async def _handle_mouse_event(self, event: events.MouseEvent) -> None:
        if (mouse_tracking := self.state.mouse_tracking) is None:
            return
        # TODO: Other mouse tracking formats
        match mouse_tracking.format:
            case "sgr":
                await self.write_process_stdin(self._encode_mouse_event_sgr(event))

    async def on_paste(self, event: events.Paste) -> None:
        await self.paste(event.text)

    async def paste(self, text: str) -> None:
        if self.state.bracketed_paste:
            await self.write_process_stdin(f"\x1b[200~{text}\x1b[201~")
        else:
            await self.write_process_stdin(text)

    async def write_process_stdin(self, input: str) -> None:
        if self._write_to_stdin is not None:
            await self._write_to_stdin(input)
