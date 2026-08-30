# Copyright (c) 2026 Chrys. All rights reserved.

from __future__ import annotations

import asyncio
import base64
import os
from types import SimpleNamespace

import pytest
from rich.console import Console
from textual.geometry import Region, Size
from textual.strip import Strip

from chrys.app.tui.terminal import ansi
from chrys.app.tui.terminal.ansi._commands import ANSIBell
from chrys.app.tui.terminal.panel import ShellPanel
from chrys.app.tui.terminal.shell import Shell
from chrys.app.tui.terminal.shell_read import shell_read
from chrys.app.tui.terminal.widget import Terminal
from chrys.foundation.i18n import Localizer, MessageRef
from chrys.foundation.i18n.formatting import format_message


class DummyTerminal:
    def __init__(self) -> None:
        self.state = SimpleNamespace(bracketed_paste=False)
        self.size = SimpleNamespace(width=80, height=24)
        self.messages = []

    def post_message(self, message: object) -> None:
        self.messages.append(message)

    def set_write_to_stdin(self, _write: object) -> None:
        pass


def _render_shell_error(error: MessageRef | str) -> str:
    return error if isinstance(error, str) else format_message(error)


def test_terminal_ignores_hidden_zero_size_resize_after_windows_reflow(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = Terminal(size=(20, 6))
    messages: list[object] = []
    refreshes = 0

    def refresh(*_regions: object, **_kwargs: object) -> None:
        nonlocal refreshes
        refreshes += 1

    monkeypatch.setattr(terminal, "post_message", messages.append)
    monkeypatch.setattr(terminal, "refresh", refresh)

    terminal._external_reflow = True
    terminal.update_size(132, 30, immediate=True)
    messages.clear()
    refreshes = 0

    terminal.update_size(0, 0, immediate=True)

    assert terminal.size.width == 132
    assert terminal.size.height == 30
    assert terminal.state.width == 132
    assert terminal.state.height == 30
    assert messages == []
    assert refreshes == 0


def test_terminal_render_line_unattached_returns_blank() -> None:
    terminal = Terminal(size=(20, 4))

    rendered = terminal.render_line(0)

    assert rendered.cell_length == 20


def test_shell_panel_terminal_reserves_scrollbar_gutter() -> None:
    assert "scrollbar-gutter: stable;" in ShellPanel.DEFAULT_CSS


def test_shell_panel_respawns_in_last_confirmed_working_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    path = "/Users/jackil/Repos/demo"
    panel = ShellPanel(working_directory="/Users/jackil/Repos/old")
    starts: list[str | None] = []

    class FinishedShell:
        is_finished = True

    monkeypatch.setattr(panel, "_start_shell", starts.append)
    monkeypatch.setattr(panel, "call_after_refresh", lambda _callback, *args: None)
    monkeypatch.setattr(panel, "post_message", lambda _message: None)
    panel._on_directory_changed(path)
    panel._shell = FinishedShell()  # type: ignore[assignment]

    panel.show()

    assert starts == [path]
    assert panel._working_directory == path


def test_shell_panel_pending_workspace_cwd_overrides_last_shell_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = ShellPanel(working_directory="/Users/jackil/Repos/old")
    starts: list[str | None] = []

    class FinishedShell:
        is_finished = True

    monkeypatch.setattr(panel, "_start_shell", starts.append)
    monkeypatch.setattr(panel, "call_after_refresh", lambda _callback, *args: None)
    panel._shell = FinishedShell()  # type: ignore[assignment]
    panel._pending_cwd = "/Users/jackil/Repos/new"

    panel.show()

    assert starts == ["/Users/jackil/Repos/new"]
    assert panel._working_directory == "/Users/jackil/Repos/new"
    assert panel._pending_cwd is None


@pytest.mark.asyncio
async def test_shell_panel_hidden_cwd_change_restarts_shell_in_new_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = ShellPanel()
    path = "/Users/jackil/Repos/demo"
    closed: list[bool] = []
    commands: list[str] = []
    starts: list[str | None] = []

    class FakeShell:
        is_finished = False

        async def close(self) -> None:
            closed.append(True)
            self.is_finished = True

    async def send_command(command: str) -> None:
        commands.append(command)

    def start_shell(working_directory: str | None = None) -> None:
        starts.append(working_directory)

    monkeypatch.setattr(panel, "send_command", send_command)
    monkeypatch.setattr(panel, "_start_shell", start_shell)
    monkeypatch.setattr(panel, "call_after_refresh", lambda _callback, *args: None)
    panel._shell = FakeShell()  # type: ignore[assignment]

    await panel.change_directory(path)

    assert closed == [True]
    assert panel._shell is None
    assert panel._pending_cwd == path

    panel.show()

    assert starts == [path]
    assert commands == []
    assert panel._pending_cwd is None


@pytest.mark.asyncio
async def test_shell_panel_visible_cwd_change_uses_cd_command(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = ShellPanel(working_directory="/tmp/current")
    commands: list[str] = []

    class FakeShell:
        is_finished = False

    async def send_command(command: str) -> None:
        commands.append(command)

    def start_shell(_working_directory: str | None = None) -> None:
        raise AssertionError("visible cwd changes should use the existing shell")

    monkeypatch.setattr(panel, "send_command", send_command)
    monkeypatch.setattr(panel, "_start_shell", start_shell)
    panel._shell = FakeShell()  # type: ignore[assignment]
    panel.display = True

    await panel.change_directory("/tmp/a b")

    assert commands == [panel._cd_command("/tmp/a b")]
    assert panel._working_directory == "/tmp/current"


@pytest.mark.asyncio
async def test_shell_panel_visible_finished_shell_saves_pending_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = ShellPanel(working_directory="/tmp/current")
    starts: list[str | None] = []

    class FinishedShell:
        is_finished = True

    monkeypatch.setattr(panel, "_start_shell", starts.append)
    monkeypatch.setattr(panel, "call_after_refresh", lambda _callback, *args: None)
    panel._shell = FinishedShell()  # type: ignore[assignment]
    panel.display = True

    await panel.change_directory("/tmp/pending")
    panel.show()

    assert starts == ["/tmp/pending"]
    assert panel._working_directory == "/tmp/pending"
    assert panel._pending_cwd is None


@pytest.mark.asyncio
async def test_terminal_height_resize_preserves_bottom_scroll_with_scrollback(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = Terminal(size=(20, 4))

    monkeypatch.setattr(
        Terminal,
        "scrollable_content_region",
        property(lambda terminal: Region(0, 0, terminal.width, terminal.height)),
    )
    monkeypatch.setattr(terminal, "post_message", lambda _message: None)
    monkeypatch.setattr(terminal, "refresh", lambda *_regions, **_kwargs: None)

    terminal._container_size = Size(20, 4)
    terminal.update_size(20, 4, immediate=True)
    await terminal.write("one\ntwo\nthree\nfour\nPS> ")
    terminal.scroll_y = terminal.max_scroll_y
    assert terminal.max_scroll_y > 0

    terminal._container_size = Size(20, 2)
    terminal.update_size(20, 2, immediate=True)
    await terminal.write("x")

    assert terminal.scroll_y == terminal.max_scroll_y
    assert int(terminal.scroll_y) <= terminal.state.buffer.cursor_line < int(terminal.scroll_y) + 2


@pytest.mark.asyncio
async def test_terminal_command_submission_scrolls_to_bottom(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = Terminal(size=(20, 4))
    stdin_writes: list[str] = []

    async def write_stdin(text: str) -> None:
        stdin_writes.append(text)

    monkeypatch.setattr(
        Terminal,
        "scrollable_content_region",
        property(lambda terminal: Region(0, 0, terminal.width, terminal.height)),
    )
    monkeypatch.setattr(terminal, "post_message", lambda _message: None)
    monkeypatch.setattr(terminal, "refresh", lambda *_regions, **_kwargs: None)
    terminal.set_write_to_stdin(write_stdin)

    terminal._container_size = Size(20, 4)
    terminal.update_size(20, 4, immediate=True)
    await terminal.write("one\ntwo\nthree\nfour\nPS> ")
    terminal.scroll_y = 0
    assert terminal.max_scroll_y > 0

    event = SimpleNamespace(
        key="enter",
        character=None,
        prevent_default=lambda: None,
        stop=lambda: None,
    )
    await terminal.on_key(event)

    assert stdin_writes == ["\r"]
    assert terminal.scroll_y == terminal.max_scroll_y


@pytest.mark.asyncio
async def test_terminal_swallows_leaked_focus_report_after_escape_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A split outer FocusIn report must not be typed into the embedded shell."""
    terminal = Terminal(size=(20, 4))
    stdin_writes: list[str] = []

    async def write_stdin(text: str) -> None:
        stdin_writes.append(text)

    class FakeTimer:
        def stop(self) -> None:
            return

    def key_event(key: str, character: str | None) -> SimpleNamespace:
        return SimpleNamespace(
            key=key,
            character=character,
            prevent_default=lambda: None,
            stop=lambda: None,
        )

    monkeypatch.setattr(terminal, "set_timer", lambda *_args, **_kwargs: FakeTimer())
    terminal.set_write_to_stdin(write_stdin)

    await terminal.on_key(key_event("escape", "\x1b"))
    await terminal.on_key(key_event("[", "["))
    await terminal.on_key(key_event("I", "I"))

    assert stdin_writes == []
    assert terminal._escaping is False
    assert terminal._pending_escape_sequence is None


@pytest.mark.asyncio
async def test_terminal_swallows_leaked_mouse_report_after_escape_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A split outer mouse report must not be typed into the embedded shell prompt."""
    terminal = Terminal(size=(20, 4))
    stdin_writes: list[str] = []

    async def write_stdin(text: str) -> None:
        stdin_writes.append(text)

    class FakeTimer:
        def stop(self) -> None:
            return

    def key_event(key: str, character: str | None) -> SimpleNamespace:
        return SimpleNamespace(
            key=key,
            character=character,
            prevent_default=lambda: None,
            stop=lambda: None,
        )

    monkeypatch.setattr(terminal, "set_timer", lambda *_args, **_kwargs: FakeTimer())
    terminal.set_write_to_stdin(write_stdin)

    await terminal.on_key(key_event("escape", "\x1b"))
    for character in "[555;130;38M":
        await terminal.on_key(key_event(character, character))

    assert stdin_writes == []
    assert terminal._escaping is False
    assert terminal._pending_escape_sequence is None


@pytest.mark.asyncio
async def test_terminal_swallows_leaked_two_field_mouse_tail_after_escape_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A split numeric mouse tail like the reported ``[103;65M`` must not reach the shell."""
    terminal = Terminal(size=(20, 4))
    stdin_writes: list[str] = []

    async def write_stdin(text: str) -> None:
        stdin_writes.append(text)

    class FakeTimer:
        def stop(self) -> None:
            return

    def key_event(key: str, character: str | None) -> SimpleNamespace:
        return SimpleNamespace(
            key=key,
            character=character,
            prevent_default=lambda: None,
            stop=lambda: None,
        )

    monkeypatch.setattr(terminal, "set_timer", lambda *_args, **_kwargs: FakeTimer())
    terminal.set_write_to_stdin(write_stdin)

    await terminal.on_key(key_event("escape", "\x1b"))
    for character in "[103;65M":
        await terminal.on_key(key_event(character, character))

    assert stdin_writes == []
    assert terminal._escaping is False
    assert terminal._pending_escape_sequence is None


@pytest.mark.asyncio
async def test_terminal_swallows_leaked_x10_mouse_report_after_escape_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A split X10 mouse report must not be typed into the embedded shell prompt."""
    terminal = Terminal(size=(20, 4))
    stdin_writes: list[str] = []

    async def write_stdin(text: str) -> None:
        stdin_writes.append(text)

    class FakeTimer:
        def stop(self) -> None:
            return

    def key_event(key: str, character: str | None) -> SimpleNamespace:
        return SimpleNamespace(
            key=key,
            character=character,
            prevent_default=lambda: None,
            stop=lambda: None,
        )

    monkeypatch.setattr(terminal, "set_timer", lambda *_args, **_kwargs: FakeTimer())
    terminal.set_write_to_stdin(write_stdin)

    await terminal.on_key(key_event("escape", "\x1b"))
    for character in "[Mabc":
        await terminal.on_key(key_event(character, character))

    assert stdin_writes == []
    assert terminal._escaping is False
    assert terminal._pending_escape_sequence is None


@pytest.mark.asyncio
async def test_terminal_forwards_non_report_escape_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Filtering leaked reports must not eat ordinary control sequences."""
    terminal = Terminal(size=(20, 4))
    stdin_writes: list[str] = []

    async def write_stdin(text: str) -> None:
        stdin_writes.append(text)

    class FakeTimer:
        def stop(self) -> None:
            return

    def key_event(key: str, character: str | None) -> SimpleNamespace:
        return SimpleNamespace(
            key=key,
            character=character,
            prevent_default=lambda: None,
            stop=lambda: None,
        )

    monkeypatch.setattr(terminal, "set_timer", lambda *_args, **_kwargs: FakeTimer())
    terminal.set_write_to_stdin(write_stdin)

    await terminal.on_key(key_event("escape", "\x1b"))
    await terminal.on_key(key_event("[", "["))
    await terminal.on_key(key_event("A", "A"))

    assert stdin_writes == ["\x1b[A"]
    assert terminal._escaping is False
    assert terminal._pending_escape_sequence is None


@pytest.mark.asyncio
async def test_terminal_caps_pending_escape_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unfinished report-like prefixes should flush instead of growing without bound."""
    terminal = Terminal(size=(20, 4))
    stdin_writes: list[str] = []

    async def write_stdin(text: str) -> None:
        stdin_writes.append(text)

    class FakeTimer:
        def stop(self) -> None:
            return

    def key_event(key: str, character: str | None) -> SimpleNamespace:
        return SimpleNamespace(
            key=key,
            character=character,
            prevent_default=lambda: None,
            stop=lambda: None,
        )

    monkeypatch.setattr(terminal, "set_timer", lambda *_args, **_kwargs: FakeTimer())
    terminal.set_write_to_stdin(write_stdin)

    await terminal.on_key(key_event("escape", "\x1b"))
    for character in "[" + ("1" * 40):
        await terminal.on_key(key_event(character, character))

    assert stdin_writes == ["\x1b[" + ("1" * 31), *("1" for _ in range(9))]
    assert terminal._escaping is False
    assert terminal._pending_escape_sequence is None


@pytest.mark.asyncio
async def test_shell_panel_syncs_visible_size_after_resize_while_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = ShellPanel()
    terminal = Terminal(size=(20, 6))
    shell_sizes: list[tuple[int, int]] = []

    class FakeShell:
        is_finished = False
        pty_size = (132, 30)

        async def wait_for_ready(self) -> None:
            pass

        async def update_size_async(self, width: int, height: int) -> None:
            self.pty_size = (width, height)
            shell_sizes.append((width, height))

    monkeypatch.setattr(
        Terminal,
        "scrollable_content_region",
        property(lambda _terminal: Region(0, 0, 100, 24)),
    )
    monkeypatch.setattr(terminal, "post_message", lambda _message: None)

    terminal.update_size(132, 30, immediate=True)
    terminal.update_size(0, 0, immediate=True)
    panel._terminal = terminal
    panel._shell = FakeShell()  # type: ignore[assignment]
    panel.display = True

    await panel._sync_visible_terminal_size()

    assert terminal.size.width == 100
    assert terminal.size.height == 24
    assert terminal.state.width == 100
    assert terminal.state.height == 24
    assert shell_sizes == [(100, 24)]
    assert panel._pending_resize is None


@pytest.mark.asyncio
async def test_shell_panel_sync_reflows_external_reflow_cursor_after_hidden_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = ShellPanel()
    terminal = Terminal(size=(20, 6))

    monkeypatch.setattr(
        Terminal,
        "scrollable_content_region",
        property(lambda _terminal: Region(0, 0, 100, 6)),
    )
    monkeypatch.setattr(terminal, "post_message", lambda _message: None)
    monkeypatch.setattr(terminal, "refresh", lambda *_regions, **_kwargs: None)

    terminal.update_size(20, 6, immediate=True)
    await terminal.write(f"{'x' * 60}\nPS D:\\Repos\\chrys> ")
    prompt_cursor = terminal.state.buffer.cursor
    assert len(terminal.state.buffer.lines[0].folds) > 1

    terminal._external_reflow = True
    terminal.update_size(100, 6, immediate=True)
    assert terminal.state.buffer.cursor == prompt_cursor
    assert len(terminal.state.buffer.lines[0].folds) > 1

    panel._terminal = terminal
    panel.display = True
    await panel._sync_visible_terminal_size()

    assert terminal.state.buffer.cursor == prompt_cursor
    assert terminal.state.buffer.cursor_line == 1
    assert len(terminal.state.buffer.lines[0].folds) == 1


@pytest.mark.asyncio
async def test_terminal_external_reflow_height_shrink_updates_screen_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = Terminal(size=(80, 24))

    monkeypatch.setattr(
        Terminal,
        "scrollable_content_region",
        property(lambda terminal: Region(0, 0, terminal.width, terminal.height)),
    )
    monkeypatch.setattr(terminal, "post_message", lambda _message: None)
    monkeypatch.setattr(terminal, "refresh", lambda *_regions, **_kwargs: None)

    terminal.update_size(80, 24, immediate=True)
    await terminal.write("row1\nrow2\nrow3\nrow4\nPS> ")
    prompt_line, _prompt_offset = terminal.state.buffer.cursor

    terminal._external_reflow = True
    terminal.update_size(80, 3, immediate=True)
    await terminal.write("\x1b[3;5Htyped")

    assert terminal.state.buffer.lines[prompt_line].content.plain == "PS> typed"
    assert terminal.state.screen_start_line_no == prompt_line - 2


@pytest.mark.asyncio
async def test_shell_panel_sync_resizes_pty_when_terminal_already_at_visible_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = ShellPanel()
    terminal = Terminal()
    shell_sizes: list[tuple[int, int]] = []
    stopped: list[str] = []

    class FakeTimer:
        def stop(self) -> None:
            stopped.append("stop")

    class FakeShell:
        is_finished = False
        pty_size = (80, 24)

        async def wait_for_ready(self) -> None:
            pass

        async def update_size_async(self, width: int, height: int) -> None:
            self.pty_size = (width, height)
            shell_sizes.append((width, height))

    def fail_set_timer(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("matching stale SizeChanged should not debounce another PTY resize")

    monkeypatch.setattr(
        Terminal,
        "scrollable_content_region",
        property(lambda _terminal: Region(0, 0, 100, 24)),
    )
    monkeypatch.setattr(terminal, "post_message", lambda _message: None)
    monkeypatch.setattr(panel, "set_timer", fail_set_timer)

    terminal.update_size(100, 24, immediate=True)
    panel._terminal = terminal
    panel._shell = FakeShell()  # type: ignore[assignment]
    panel._pending_resize = (100, 24)
    panel._resize_timer = FakeTimer()  # type: ignore[assignment]
    panel.display = True

    await panel._sync_visible_terminal_size()
    panel._on_terminal_size_changed(Terminal.SizeChanged(terminal, 100, 24))

    assert stopped == ["stop"]
    assert panel._resize_timer is None
    assert panel._pending_resize is None
    assert shell_sizes == [(100, 24)]


def test_shell_panel_ignores_stale_queued_size_changed(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = ShellPanel()
    terminal = Terminal(size=(100, 20))

    class FakeShell:
        is_finished = False
        pty_size = (80, 24)

    def fail_set_timer(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("stale SizeChanged should not schedule a PTY resize")

    monkeypatch.setattr(panel, "set_timer", fail_set_timer)
    panel._shell = FakeShell()  # type: ignore[assignment]

    panel._on_terminal_size_changed(Terminal.SizeChanged(terminal, 100, 10))

    assert panel._pending_resize is None


@pytest.mark.asyncio
async def test_shell_panel_ignores_stale_pending_resize_when_timer_fires() -> None:
    panel = ShellPanel()
    terminal = Terminal(size=(100, 20))

    class FakeShell:
        is_finished = False

        async def update_size_async(self, _width: int, _height: int) -> None:
            raise AssertionError("stale pending resize should not reach the PTY")

    panel._terminal = terminal
    panel._shell = FakeShell()  # type: ignore[assignment]
    panel._pending_resize = (100, 10)

    await panel._apply_pending_resize()

    assert panel._pending_resize is None
    assert panel._resize_timer is None


def test_shell_panel_does_not_skip_resize_for_unknown_default_pty_size(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = ShellPanel()
    shell = Shell(terminal=DummyTerminal(), working_directory="/")
    timers: list[tuple[float, object]] = []

    class FakeTimer:
        def stop(self) -> None:
            pass

    def set_timer(delay: float, callback: object) -> FakeTimer:
        timers.append((delay, callback))
        return FakeTimer()

    monkeypatch.setattr(panel, "set_timer", set_timer)
    panel._shell = shell

    panel._on_terminal_size_changed(Terminal.SizeChanged(Terminal(), 80, 24))

    assert shell.pty_size is None
    assert panel._pending_resize == (80, 24)
    assert timers


@pytest.mark.asyncio
async def test_shell_resize_before_backend_ready_does_not_mark_pty_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chrys.app.tui.terminal.shell as shell_module

    panel = ShellPanel()
    shell = Shell(terminal=DummyTerminal(), working_directory="/")
    timers: list[tuple[float, object]] = []

    class FakeTimer:
        def stop(self) -> None:
            pass

    def set_timer(delay: float, callback: object) -> FakeTimer:
        timers.append((delay, callback))
        return FakeTimer()

    monkeypatch.setattr(shell_module, "IS_WINDOWS", True)
    monkeypatch.setattr(panel, "set_timer", set_timer)

    await shell.update_size_async(100, 24)
    panel._shell = shell
    panel._on_terminal_size_changed(Terminal.SizeChanged(Terminal(size=(100, 24)), 100, 24))

    assert shell.pty_size is None
    assert panel._pending_resize == (100, 24)
    assert timers


@pytest.mark.asyncio
async def test_shell_read_oserror_does_not_print(capsys: pytest.CaptureFixture[str]) -> None:
    class BrokenReader:
        def __init__(self) -> None:
            self.calls = 0

        async def read(self, _size: int) -> bytes:
            self.calls += 1
            if self.calls == 1:
                return b"x"
            raise OSError("pipe closed")

    data = await shell_read(BrokenReader(), 8)

    captured = capsys.readouterr()
    assert data == b"x"
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.asyncio
async def test_ansi_line_attribute_sequence_does_not_print(capsys: pytest.CaptureFixture[str]) -> None:
    state = ansi.TerminalState(lambda _text: None)

    await state.write("\x1b#8")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.asyncio
async def test_send_input_writes_raw_text(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = DummyTerminal()
    shell = Shell(terminal=terminal, working_directory="/")
    shell._ready_event.set()
    writes = []

    monkeypatch.setattr(shell, "_is_ready", lambda: True)

    async def write(text: str, hide_echo: bool = False) -> int:
        writes.append((text, hide_echo))
        return len(text)

    monkeypatch.setattr(shell, "write", write)

    await shell.send_input("abc")

    assert writes == [("abc", False)]


@pytest.mark.asyncio
async def test_send_input_wraps_bracketed_paste_without_newline(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = DummyTerminal()
    terminal.state.bracketed_paste = True
    shell = Shell(terminal=terminal, working_directory="/")
    shell._ready_event.set()
    writes = []

    monkeypatch.setattr(shell, "_is_ready", lambda: True)

    async def write(text: str, hide_echo: bool = False) -> int:
        writes.append(text)
        return len(text)

    monkeypatch.setattr(shell, "write", write)

    await shell.send_input("abc", paste=True)

    assert writes == ["\x1b[200~abc\x1b[201~"]


@pytest.mark.asyncio
async def test_windows_shell_send_keeps_positive_pty_width(monkeypatch: pytest.MonkeyPatch) -> None:
    import chrys.app.tui.terminal.shell as shell_module

    class FakeWinPTY:
        def __init__(self) -> None:
            self.sizes: list[tuple[int, int]] = []

        def set_size(self, width: int, height: int) -> None:
            self.sizes.append((width, height))

    monkeypatch.setattr(shell_module, "IS_WINDOWS", True)
    shell = Shell(terminal=DummyTerminal(), working_directory="/")
    shell._ready_event.set()
    winpty = FakeWinPTY()
    shell._winpty = winpty  # type: ignore[assignment]
    shell.update_size(132, 30)
    writes: list[str] = []

    async def write(text: str | bytes, hide_echo: bool = False) -> int:
        assert isinstance(text, str)
        writes.append(text)
        return len(text)

    monkeypatch.setattr(shell, "write", write)

    await shell.send("pwd", 0, 1)

    assert winpty.sizes == [(132, 30)]
    assert shell.pty_size == (132, 30)
    assert writes == ["pwd\n"]


@pytest.mark.asyncio
async def test_windows_shell_async_resize_keeps_positive_pty_width(monkeypatch: pytest.MonkeyPatch) -> None:
    import chrys.app.tui.terminal.shell as shell_module

    class FakeWinPTY:
        def __init__(self) -> None:
            self.sizes: list[tuple[int, int]] = []

        def set_size(self, width: int, height: int) -> None:
            self.sizes.append((width, height))

    monkeypatch.setattr(shell_module, "IS_WINDOWS", True)
    shell = Shell(terminal=DummyTerminal(), working_directory="/")
    winpty = FakeWinPTY()
    shell._winpty = winpty  # type: ignore[assignment]
    shell.update_size(120, 40)

    await shell.update_size_async(0, 1)

    assert winpty.sizes == [(120, 40)]
    assert shell.pty_size == (120, 40)


@pytest.mark.asyncio
async def test_windows_shell_async_resize_serializes_matching_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    import chrys.app.tui.terminal.shell as shell_module

    class FakeWinPTY:
        def set_size(self, width: int, height: int) -> None:
            calls.append((width, height))
            time.sleep(0.01)

    monkeypatch.setattr(shell_module, "IS_WINDOWS", True)
    shell = Shell(terminal=DummyTerminal(), working_directory="/")
    shell._winpty = FakeWinPTY()  # type: ignore[assignment]
    calls: list[tuple[int, int]] = []

    await asyncio.gather(
        shell.update_size_async(100, 30),
        shell.update_size_async(100, 30),
    )

    assert calls == [(100, 30)]
    assert shell.pty_size == (100, 30)


@pytest.mark.asyncio
async def test_windows_shell_send_uses_default_pty_size_when_first_measurement_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chrys.app.tui.terminal.shell as shell_module

    class FakeWinPTY:
        def __init__(self) -> None:
            self.sizes: list[tuple[int, int]] = []

        def set_size(self, width: int, height: int) -> None:
            self.sizes.append((width, height))

    monkeypatch.setattr(shell_module, "IS_WINDOWS", True)
    shell = Shell(terminal=DummyTerminal(), working_directory="/")
    shell._ready_event.set()
    winpty = FakeWinPTY()
    shell._winpty = winpty  # type: ignore[assignment]

    async def write(text: str | bytes, hide_echo: bool = False) -> int:
        return len(text)

    monkeypatch.setattr(shell, "write", write)

    await shell.send("pwd", 0, 1)

    assert winpty.sizes == [(80, 24)]
    assert shell.pty_size == (80, 24)


@pytest.mark.asyncio
async def test_windows_shell_resize_suppresses_winpty_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import chrys.app.tui.terminal.shell as shell_module

    class FakeWinptyError(Exception):
        pass

    class FakeWinPTY:
        def set_size(self, _width: int, _height: int) -> None:
            raise FakeWinptyError("closed")

    monkeypatch.setattr(shell_module, "IS_WINDOWS", True)
    monkeypatch.setattr(shell_module, "_PTY_OPERATION_ERRORS", (OSError, RuntimeError, FakeWinptyError))
    shell = Shell(terminal=DummyTerminal(), working_directory="/")
    shell._winpty = FakeWinPTY()  # type: ignore[assignment]

    await shell.update_size_async(120, 40)


@pytest.mark.asyncio
async def test_windows_startup_failure_in_winpty_constructor_wakes_waiters(monkeypatch: pytest.MonkeyPatch) -> None:
    import chrys.app.tui.terminal.shell as shell_module

    class BrokenWinPTY:
        def __init__(self, _cols: int, _rows: int) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(shell_module, "WinPTY", BrokenWinPTY, raising=False)
    terminal = DummyTerminal()
    terminal.size = SimpleNamespace(width=0, height=1)
    errors: list[MessageRef | str] = []
    shell = Shell(terminal=terminal, working_directory="/", on_error=errors.append)

    await shell._run_windows()
    await shell.wait_for_ready()

    assert shell.is_finished
    assert shell._winpty is None
    assert [_render_shell_error(error) for error in errors] == ["Unable to start shell: boom"]
    assert isinstance(errors[0], MessageRef)
    assert Localizer("zh-Hans").render(errors[0]) == "无法启动 Shell：boom"  # noqa: RUF001
    assert terminal.messages


@pytest.mark.asyncio
async def test_write_input_queued_coalesces_pending_input(monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(terminal=DummyTerminal(), working_directory="/")
    writes = []

    monkeypatch.setattr(shell, "_is_ready", lambda: True)

    async def write(text: str | bytes, hide_echo: bool = False) -> int:
        writes.append(text)
        return len(text)

    monkeypatch.setattr(shell, "write", write)

    await shell.write_input_queued("a")
    await shell.write_input_queued("b")
    await shell.write_input_queued("c")
    await shell._input_queue.join()
    shell._cancel_input_writer()

    assert writes == [b"abc"]


@pytest.mark.asyncio
async def test_write_input_queued_applies_current_terminal_size_before_input(monkeypatch: pytest.MonkeyPatch) -> None:
    import chrys.app.tui.terminal.shell as shell_module

    class FakeWinPTY:
        def set_size(self, width: int, height: int) -> None:
            operations.append(("resize", width, height))

    terminal = DummyTerminal()
    terminal.size = SimpleNamespace(width=100, height=20)
    shell = Shell(terminal=terminal, working_directory="/")
    operations: list[tuple[str, int, int] | tuple[str, bytes]] = []

    async def write(text: str | bytes, hide_echo: bool = False) -> int:
        assert isinstance(text, bytes)
        operations.append(("write", text))
        return len(text)

    monkeypatch.setattr(shell_module, "IS_WINDOWS", True)
    monkeypatch.setattr(shell, "write", write)
    shell._winpty = FakeWinPTY()  # type: ignore[assignment]
    shell._mark_pty_size_applied(80, 24)

    await shell.write_input_queued("x")
    await shell._input_queue.join()
    shell._cancel_input_writer()

    assert operations == [("resize", 100, 20), ("write", b"x")]
    assert shell.pty_size == (100, 20)


@pytest.mark.asyncio
async def test_input_size_sync_retries_when_terminal_resizes_during_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    terminal = DummyTerminal()
    terminal.size = SimpleNamespace(width=100, height=20)
    shell = Shell(terminal=terminal, working_directory="/")
    calls: list[tuple[int, int]] = []

    async def update_size_async(width: int, height: int) -> None:
        calls.append((width, height))
        if len(calls) == 1:
            terminal.size = SimpleNamespace(width=120, height=30)

    monkeypatch.setattr(shell, "update_size_async", update_size_async)

    await shell._sync_terminal_size_before_input()

    assert calls == [(100, 20), (120, 30)]


@pytest.mark.asyncio
async def test_shell_interrupt_does_not_resize_before_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(terminal=DummyTerminal(), working_directory="/")
    writes = []

    async def fail_sync() -> None:
        raise AssertionError("explicit interrupt should not wait on a resize")

    async def write(text: str | bytes, hide_echo: bool = False) -> int:
        writes.append(text)
        return len(text)

    monkeypatch.setattr(shell, "_sync_terminal_size_before_input", fail_sync)
    monkeypatch.setattr(shell, "write", write)

    await shell.interrupt()

    assert writes == [b"\x03"]


@pytest.mark.asyncio
async def test_shell_close_terminates_and_waits_for_task(monkeypatch: pytest.MonkeyPatch) -> None:
    shell = Shell(terminal=DummyTerminal(), working_directory="/")
    released = asyncio.Event()
    calls: list[str] = []

    async def runner() -> None:
        try:
            await released.wait()
        finally:
            calls.append("runner_done")

    def terminate() -> None:
        calls.append("terminate")
        shell._mark_finished()
        released.set()

    shell._task = asyncio.create_task(runner())
    monkeypatch.setattr(shell, "terminate", terminate)

    await shell.close()

    assert calls == ["terminate", "runner_done"]
    assert shell._task is None


@pytest.mark.asyncio
async def test_shell_panel_unmount_closes_owned_shell() -> None:
    panel = ShellPanel()
    calls: list[str] = []

    class FakeTimer:
        def stop(self) -> None:
            calls.append("timer_stop")

    class FakeShell:
        async def close(self) -> None:
            calls.append("shell_close")

    panel._resize_timer = FakeTimer()  # type: ignore[assignment]
    panel._shell = FakeShell()  # type: ignore[assignment]

    await panel.on_unmount()

    assert calls == ["timer_stop", "shell_close"]
    assert panel._resize_timer is None
    assert panel._shell is None


@pytest.mark.skipif(os.name == "nt", reason="Unix PTY startup path")
@pytest.mark.asyncio
async def test_closed_shell_run_does_not_spawn_late_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    import chrys.app.tui.terminal.shell as shell_module

    shell = Shell(terminal=DummyTerminal(), working_directory="/")
    shell.terminate()

    async def fail_start(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("closed shell should not spawn a subprocess")

    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_exec", fail_start)

    await shell.run()

    assert shell.is_finished


@pytest.mark.asyncio
async def test_alternate_screen_single_row_update_returns_single_delta() -> None:
    state = ansi.TerminalState(lambda _text: None)

    await state.write("\x1b[?1049h\x1b[1;1Hfirst")
    _scrollback_delta, alternate_delta = await state.write("\x1b[1;1Hsecond")

    assert alternate_delta == {0}


@pytest.mark.asyncio
async def test_ansi_alternate_screen_resize_clears_stale_rows() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=20, height=5)
    await state.write("\x1b[?1049h")
    await state.write("one\ntwo\nthree\nfour\nfive")

    state.update_size(width=10, height=3)

    assert state.alternate_screen
    assert state.alternate_buffer.height == 0
    assert state.alternate_buffer.cursor_line == 0
    assert state.alternate_buffer.cursor_offset == 0


@pytest.mark.asyncio
async def test_ansi_insert_replace_mode_mapping() -> None:
    state = ansi.TerminalState(lambda _text: None)

    await state.write("ab\r\x1b[4hX")
    assert state.buffer.lines[0].content.plain == "Xab"

    await state.write("\r\x1b[4lY")
    assert state.buffer.lines[0].content.plain == "Yab"


@pytest.mark.asyncio
async def test_ansi_csi_save_restore_cursor() -> None:
    state = ansi.TerminalState(lambda _text: None)

    await state.write("ab\x1b[s\rX\x1b[uY")

    assert state.buffer.lines[0].content.plain == "XbY"


@pytest.mark.asyncio
async def test_ansi_save_restore_after_newline_keeps_prompt_line_intact() -> None:
    state = ansi.TerminalState(lambda _text: None)

    await state.write("user@host workspace % mock-tui\r\n")
    await state.write("\x1b7\x1b[r\x1b8 ▐▛███▜▌\x1b[3CMock\x1b[1CTUI\x1b[1Cv1.2.3")

    assert state.buffer.lines[0].content.plain == "user@host workspace % mock-tui"
    assert state.buffer.lines[1].content.plain == " ▐▛███▜▌   Mock TUI v1.2.3"


@pytest.mark.asyncio
async def test_ansi_newline_at_scrollback_bottom_appends_line() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=20, height=3)

    await state.write("line 1\nline 2\nline 3")
    await state.write("\nnext prompt")

    assert [line.content.plain for line in state.buffer.lines] == [
        "line 1",
        "line 2",
        "line 3",
        "next prompt",
    ]
    assert state.screen_start_line_no == 1


@pytest.mark.asyncio
async def test_ansi_vertical_grow_keeps_absolute_cursor_rows_on_same_screen_origin() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=80, height=4)

    await state.write("row1\nrow2\nrow3\nrow4\nPS> ")
    prompt_line, _prompt_offset = state.buffer.cursor
    assert state.screen_start_line_no == 1

    state.update_size(width=80, height=8)
    await state.write("\x1b[4;5Htyped")

    assert state.buffer.lines[prompt_line].content.plain == "PS> typed"
    assert state.screen_start_line_no == 1
    assert state.primary_screen_height == 9


@pytest.mark.asyncio
async def test_ansi_hard_newline_after_wrapped_bottom_line_appends_prompt_line() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=100, height=24)

    for line_no in range(23):
        await state.write(f"old {line_no}\n")
    await state.write("PS D:\\Repos\\chrys> ")

    await state.write(
        "asdfasdf\r\n"
        "\x1b[0m\x1b[0m\x1b[31;1masdfasdf: "
        "\x1b[31;1mThe term 'asdfasdf' is not recognized as a name of a cmdlet, "
        "function, script file, or executable program.\x1b[0m\r\n"
        "\x1b[31;1m\x1b[31;1mCheck the spelling of the name, or if a path was included, "
        "verify that the path is correct and try again.\x1b[0m\r\n"
        "PS D:\\Repos\\chrys> "
    )

    assert state.buffer.lines[-4].content.plain == "PS D:\\Repos\\chrys> asdfasdf"
    assert state.buffer.lines[-3].content.plain.startswith("asdfasdf: The term")
    assert state.buffer.lines[-2].content.plain.startswith("Check the spelling")
    assert state.buffer.lines[-1].content.plain == "PS D:\\Repos\\chrys> "


@pytest.mark.asyncio
async def test_ansi_width_and_height_resize_reanchors_after_reflow() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=100, height=24)

    for line_no in range(23):
        await state.write(f"old {line_no}\n")
    await state.write(
        "PS D:\\Repos\\chrys> asdfasdf\r\n"
        "asdfasdf: The term 'asdfasdf' is not recognized as a name of a cmdlet, "
        "function, script file, or executable program.\r\n"
        "Check the spelling of the name, or if a path was included, verify that the path is correct and try again.\r\n"
        "PS D:\\Repos\\chrys> "
    )
    prompt_line, _prompt_offset = state.buffer.cursor

    state.update_size(width=80, height=12, force_reflow=True)
    await state.write("\x1b[12;20Hsdffasfd")

    assert state.buffer.lines[prompt_line].content.plain == "PS D:\\Repos\\chrys> sdffasfd"


@pytest.mark.asyncio
async def test_ansi_width_and_height_grow_preserves_cursor_physical_row_after_reflow() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=80, height=12)

    for line_no in range(23):
        await state.write(f"old {line_no}\n")
    await state.write(
        "PS D:\\Repos\\chrys> asdfasdf\r\n"
        "asdfasdf: The term 'asdfasdf' is not recognized as a name of a cmdlet, "
        "function, script file, or executable program.\r\n"
        "Check the spelling of the name, or if a path was included, verify that the path is correct and try again.\r\n"
        "PS D:\\Repos\\chrys> \r\n"
        "PS D:\\Repos\\chrys> \r\n"
        "PS D:\\Repos\\chrys> "
    )
    prompt_line, _prompt_offset = state.buffer.cursor
    assert state.buffer.cursor_line - state.screen_start_line_no == 11

    state.update_size(width=100, height=24, force_reflow=True)
    await state.write("\x1b[12;20Hsdfffffffffff")

    assert state.buffer.lines[prompt_line].content.plain == "PS D:\\Repos\\chrys> sdfffffffffff"


@pytest.mark.asyncio
async def test_terminal_bottom_render_keeps_physical_origin_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = Terminal(size=(80, 12))

    monkeypatch.setattr(
        Terminal,
        "scrollable_content_region",
        property(lambda terminal: Region(0, 0, terminal.width, terminal.height)),
    )
    monkeypatch.setattr(terminal, "post_message", lambda _message: None)
    monkeypatch.setattr(terminal, "refresh", lambda *_regions, **_kwargs: None)

    terminal._container_size = Size(80, 12)
    terminal.update_size(80, 12, immediate=True)
    for line_no in range(23):
        await terminal.write(f"old {line_no}\n")
    await terminal.write("PS D:\\Repos\\chrys> ")
    terminal._container_size = Size(100, 24)
    terminal.update_size(100, 24, immediate=True, force_reflow=True)

    assert terminal.virtual_size.height == terminal.state.screen_start_line_no + terminal.height
    assert terminal.max_scroll_y == terminal.state.screen_start_line_no
    assert terminal.scroll_y == terminal.max_scroll_y
    assert terminal._visible_source_start() == terminal.state.screen_start_line_no


@pytest.mark.asyncio
async def test_terminal_hidden_resize_grow_clears_scrollbar_when_scrollback_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = Terminal(size=(80, 3))

    monkeypatch.setattr(
        Terminal,
        "scrollable_content_region",
        property(lambda terminal: Region(0, 0, terminal.width, terminal.height)),
    )
    monkeypatch.setattr(terminal, "post_message", lambda _message: None)
    monkeypatch.setattr(terminal, "refresh", lambda *_regions, **_kwargs: None)

    terminal._container_size = Size(80, 3)
    terminal.update_size(80, 3, immediate=True)
    await terminal.write("one\ntwo\nthree\nPS> ")
    assert terminal.max_scroll_y == 1

    terminal._container_size = Size(80, 10)
    terminal.update_size(80, 10, immediate=True, force_reflow=True)

    assert terminal.state.screen_start_line_no == 0
    assert terminal.max_scroll_y == 0
    assert terminal.scroll_y == 0


@pytest.mark.asyncio
async def test_terminal_live_resize_grow_clears_scrollbar_when_scrollback_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = Terminal(size=(80, 3))

    monkeypatch.setattr(
        Terminal,
        "scrollable_content_region",
        property(lambda terminal: Region(0, 0, terminal.width, terminal.height)),
    )
    monkeypatch.setattr(terminal, "post_message", lambda _message: None)
    monkeypatch.setattr(terminal, "refresh", lambda *_regions, **_kwargs: None)

    terminal._container_size = Size(80, 3)
    terminal.update_size(80, 3, immediate=True)
    await terminal.write("one\ntwo\nthree\nPS> ")
    assert terminal.max_scroll_y == 1

    terminal._container_size = Size(80, 10)
    terminal.update_size(80, 10, immediate=True)

    assert terminal.state.screen_start_line_no == 1
    assert terminal.max_scroll_y == 0
    assert terminal.scroll_y == 0
    assert terminal._visible_source_start() == 0


@pytest.mark.asyncio
async def test_terminal_resize_shrink_after_fit_reanchors_from_visual_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = Terminal(size=(80, 3))

    monkeypatch.setattr(
        Terminal,
        "scrollable_content_region",
        property(lambda terminal: Region(0, 0, terminal.width, terminal.height)),
    )
    monkeypatch.setattr(terminal, "post_message", lambda _message: None)
    monkeypatch.setattr(terminal, "refresh", lambda *_regions, **_kwargs: None)

    terminal._container_size = Size(80, 3)
    terminal.update_size(80, 3, immediate=True)
    await terminal.write("one\ntwo\nthree\nPS> ")
    assert terminal.state.screen_start_line_no == 1

    terminal._container_size = Size(80, 10)
    terminal.update_size(80, 10, immediate=True)
    assert terminal._visible_source_start() == 0
    assert terminal.max_scroll_y == 0

    terminal._container_size = Size(80, 3)
    terminal.update_size(80, 3, immediate=True)
    await terminal.write("\x1b[3;5Htyped")

    assert terminal.state.screen_start_line_no == 1
    assert terminal.max_scroll_y == 1
    assert terminal.state.buffer.lines[-1].content.plain == "PS> typed"


@pytest.mark.asyncio
async def test_terminal_viewport_sync_uses_visible_region_height_for_scroll_canvas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = Terminal(size=(80, 20))

    monkeypatch.setattr(
        Terminal,
        "scrollable_content_region",
        property(lambda _terminal: Region(0, 0, 80, 10)),
    )

    for line_no in range(12):
        await terminal.write(f"old {line_no}\n")
    terminal.state._screen_start_line_no = 3
    terminal.state.height = 20
    terminal._container_size = Size(80, 10)
    terminal._sync_viewport_from_state(pin_to_bottom=True)

    assert terminal.virtual_size.height == 13
    assert terminal.max_scroll_y == 3
    assert terminal.scroll_y == 3


@pytest.mark.asyncio
async def test_ansi_write_clamps_stale_screen_origin_to_cursor() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=100, height=6)

    for line_no in range(10):
        await state.write(f"old {line_no}\n")
    await state.write("PS D:\\Repos\\chrys> ")

    state._screen_start_line_no = state.buffer.cursor_line - state.height
    await state.write("\rPS D:\\Repos\\chrys> ")

    assert state.buffer.cursor_line == state.screen_start_line_no + state.height - 1
    await state.write("\x1b[6;20Htyped")
    assert state.buffer.lines[-1].content.plain == "PS D:\\Repos\\chrys> typed"


@pytest.mark.asyncio
async def test_ansi_first_absolute_cursor_after_resize_reanchors_prompt() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=100, height=24)

    await state.write(
        "PS D:\\Repos\\chrys> ls\r\n"
        "\r\n"
        "    Directory: D:\\Repos\\chrys\r\n"
        "\r\n"
        "Mode                 LastWriteTime         Length Name\r\n"
        "----                 -------------         ------ ----\r\n"
    )
    for line_no in range(17):
        await state.write(f"d----           5/10/2026 12:00 PM                item{line_no}\r\n")
    await state.write("\r\nPS D:\\Repos\\chrys> ")
    prompt_line, _prompt_offset = state.buffer.cursor

    state.update_size(width=80, height=24, force_reflow=True, external_reflow=True)
    state._screen_start_line_no = prompt_line - 4
    await state.write("\x1b[5;20Hsadfasdf")

    assert state.buffer.lines[prompt_line].content.plain == "PS D:\\Repos\\chrys> sadfasdf"
    assert "sadfasdf" not in state.buffer.lines[4].content.plain


@pytest.mark.asyncio
async def test_ansi_local_reflow_resize_does_not_reanchor_absolute_cursor() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=100, height=24)

    await state.write("line0\r\nline1\r\nline2\r\nline3\r\nline4\r\nline5\r\nPS D:\\Repos\\chrys> ")
    prompt_line, _prompt_offset = state.buffer.cursor

    state.update_size(width=80, height=24, force_reflow=True)
    state._screen_start_line_no = 0
    await state.write("\x1b[5;1Htoprow")

    assert state.buffer.lines[4].content.plain == "toprow"
    assert state.buffer.lines[prompt_line].content.plain == "PS D:\\Repos\\chrys> "


@pytest.mark.asyncio
async def test_ansi_reverse_index_at_primary_top_still_scrolls_down() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=20, height=3)

    await state.write("one\ntwo\nthree")
    await state.write("\x1b[1;1H\x1bM")

    assert state.buffer.lines[0].content.plain == ""
    assert state.buffer.lines[1].content.plain == "one"
    assert state.buffer.lines[2].content.plain == "two"


@pytest.mark.asyncio
async def test_ansi_scroll_buffer_uses_explicit_primary_origin() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=20, height=3)

    await state.write("old0\nold1\nscreen0\nscreen1\nscreen2")
    state._screen_start_line_no = 2
    state.buffer.cursor_line = 2
    await state.write("\x1bM")

    assert state.buffer.lines[0].content.plain == "old0"
    assert state.buffer.lines[1].content.plain == "old1"
    assert state.buffer.lines[2].content.plain == ""
    assert state.buffer.lines[3].content.plain == "screen0"
    assert state.buffer.lines[4].content.plain == "screen1"


@pytest.mark.asyncio
async def test_ansi_write_clamps_stale_screen_origin_up_to_cursor() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=20, height=3)

    await state.write("one\ntwo\nthree\nfour")
    state._screen_start_line_no = 3
    state.buffer.cursor_line = 1
    await state.write("\r")

    assert state.screen_start_line_no == 1


@pytest.mark.asyncio
async def test_ansi_primary_cursor_position_request_reports_screen_row() -> None:
    stdin_writes: list[str] = []

    async def write_stdin(text: str) -> bool:
        stdin_writes.append(text)
        return True

    state = ansi.TerminalState(write_stdin)
    state.update_size(width=20, height=3)

    await state.write("one\ntwo\nthree\nfour\nPS> ")
    assert state.screen_start_line_no == 2

    await state.write("\x1b[6n")

    assert stdin_writes == ["\x1b[3;5R"]


@pytest.mark.asyncio
async def test_ansi_alternate_cursor_position_request_reports_buffer_row() -> None:
    stdin_writes: list[str] = []

    async def write_stdin(text: str) -> bool:
        stdin_writes.append(text)
        return True

    state = ansi.TerminalState(write_stdin)
    state.update_size(width=20, height=3)

    await state.write("\x1b[?1049h\x1b[2;4H\x1b[6n")

    assert stdin_writes == ["\x1b[2;4R"]


@pytest.mark.asyncio
async def test_ansi_blank_lines_at_scrollback_bottom_are_preserved() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=20, height=3)

    await state.write("[line 1]\n[line 2]\n[line 3]\n\n[next]")

    assert [line.content.plain for line in state.buffer.lines] == [
        "[line 1]",
        "[line 2]",
        "[line 3]",
        "",
        "[next]",
    ]


@pytest.mark.asyncio
async def test_ansi_newline_scroll_region_keeps_carriage_return() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=20, height=3)

    await state.write("\x1b[1;2r\x1b[2;4H\nX")

    assert state.buffer.lines[1].content.plain == "X"
    assert state.buffer.cursor_offset == 1


@pytest.mark.asyncio
async def test_ansi_cup_past_eol_pads_with_blanks() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=20, height=3)

    await state.write("\x1b[1;1Habcd\x1b[1;10HX")

    assert state.buffer.lines[0].content.plain == "abcd     X"
    assert state.buffer.lines[0].content.cell_length == 10


@pytest.mark.asyncio
async def test_ansi_cup_past_cjk_eol_pads_with_blanks() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=20, height=3)

    await state.write("\x1b[1;1H你是谁?\x1b[1;10HX")

    assert state.buffer.lines[0].content.plain == "你是谁?  X"
    assert state.buffer.lines[0].content.cell_length == 10


@pytest.mark.asyncio
async def test_ansi_replace_cjk_over_ascii_keeps_cell_length() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=20, height=3)

    await state.write("\x1b[1;1HABCDEFGHIJ\x1b[1;1H你好")

    assert state.buffer.lines[0].content.plain == "你好EFGHIJ"
    assert state.buffer.lines[0].content.cell_length == 10


@pytest.mark.asyncio
async def test_ansi_replace_ascii_over_cjk_pads_bisected_wide_char() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=20, height=3)

    await state.write("\x1b[1;1H你好世界AB\x1b[1;1Hxyz")

    assert state.buffer.lines[0].content.plain == "xyz 世界AB"
    assert state.buffer.lines[0].content.cell_length == 10


@pytest.mark.asyncio
async def test_ansi_replace_at_right_half_of_wide_char_blanks_left_half() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=20, height=3)

    await state.write("\x1b[1;1H你BCD\x1b[1;2HX")

    assert state.buffer.lines[0].content.plain == " XBCD"
    assert state.buffer.lines[0].content.cell_length == 5


@pytest.mark.asyncio
async def test_ansi_write_on_wrapped_row_uses_unfolded_cell_offset() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=5, height=3)

    await state.write("abcdefghi\x1b[2;2HX")

    assert state.buffer.lines[0].content.plain == "abcdefXhi"


@pytest.mark.asyncio
async def test_ansi_clear_wrapped_row_uses_unfolded_cell_offset() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=5, height=3)

    await state.write("abcdefghi\x1b[2;2H\x1b[K")

    assert state.buffer.lines[0].content.plain == "abcdef"


@pytest.mark.asyncio
async def test_ansi_erase_wrapped_row_uses_unfolded_cell_offset() -> None:
    state = ansi.TerminalState(lambda _text: None)
    state.update_size(width=5, height=3)

    await state.write("abcdefghi\x1b[2;2H\x1b[X")

    assert state.buffer.lines[0].content.plain == "abcdef hi"


@pytest.mark.skipif(os.name == "nt", reason="Unix PTY startup path")
@pytest.mark.asyncio
async def test_unix_startup_failure_wakes_waiters(monkeypatch: pytest.MonkeyPatch) -> None:
    import chrys.app.tui.terminal.shell as shell_module

    terminal = DummyTerminal()
    errors: list[MessageRef | str] = []
    resize_calls: list[tuple[int, int, int]] = []
    shell = Shell(terminal=terminal, working_directory="/", shell_command="bash", on_error=errors.append)

    def fake_resize_pty(fd: int, cols: int, rows: int) -> bool:
        resize_calls.append((fd, cols, rows))
        return True

    async def fail_start(*_args: object, **_kwargs: object) -> object:
        raise OSError("boom")

    monkeypatch.setattr(shell_module, "resize_pty", fake_resize_pty)
    monkeypatch.setattr(shell_module.asyncio, "create_subprocess_exec", fail_start)

    await shell.run()
    await shell.wait_for_ready()

    assert shell.is_finished
    assert [_render_shell_error(error) for error in errors] == ["Unable to start shell: boom"]
    assert terminal.messages
    assert shell._init_tmpdir is None
    assert resize_calls
    assert resize_calls[0][1:] == (80, 24)
    assert shell.pty_size == (80, 24)


def test_prepare_shell_init_returns_argv_for_bash() -> None:
    shell = Shell(terminal=DummyTerminal(), working_directory="/")
    env = {}

    argv = shell._prepare_shell_init("bash", env)

    assert argv[0] == "bash"
    assert "--rcfile" in argv
    rcfile = argv[argv.index("--rcfile") + 1]
    assert rcfile.endswith("init.bash")
    shell._cleanup_init_tmpdir()


def test_prepare_shell_init_returns_argv_for_fish() -> None:
    shell = Shell(terminal=DummyTerminal(), working_directory="/")

    argv = shell._prepare_shell_init("fish", {})

    assert argv[0] == "fish"
    assert argv[1] == "-C"
    assert "b64:" in argv[2]


def test_prepare_shell_init_returns_argv_for_pwsh() -> None:
    shell = Shell(terminal=DummyTerminal(), working_directory="/")

    argv = shell._prepare_shell_init("pwsh", {})

    assert argv[0] == "pwsh"
    assert argv[1:3] == ["-NoExit", "-Command"]
    assert "b64:" in argv[3]


@pytest.mark.asyncio
async def test_ansi_decodes_b64_working_directory_with_semicolons() -> None:
    state = ansi.TerminalState(lambda _text: None)
    path = "/tmp/a;b c"
    encoded = base64.b64encode(path.encode()).decode()

    await state.write(f"\x1b]2025;b64:{encoded}\x1b\\")

    assert state.current_directory == path


@pytest.mark.asyncio
async def test_ansi_consumes_standalone_and_fragmented_bells() -> None:
    state = ansi.TerminalState(lambda _text: None)

    await state.write("before")
    await state.write("\x07")
    await state.write("\x07\x07after")

    line = state.buffer.lines[0].content
    assert line.plain == "beforeafter"
    assert state.buffer.cursor == (0, len("beforeafter"))


def test_ansi_bell_command_is_truthy() -> None:
    assert ANSIBell()


@pytest.mark.asyncio
async def test_ansi_bell_does_not_reach_rendered_segments() -> None:
    state = ansi.TerminalState(lambda _text: None)

    await state.write("before\x07after")

    line = state.buffer.folded_lines[0].content
    strip = Strip(line.render_segments(), cell_length=line.cell_length)
    rendered = strip.render(Console(force_terminal=True, color_system=None))
    assert "\x07" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("control", [*map(chr, range(0x20)), "\x7f"])
async def test_ansi_never_stores_or_renders_c0_controls(control: str) -> None:
    state = ansi.TerminalState(lambda _text: None)

    await state.write(f"before{control}after")

    for line in state.buffer.lines:
        assert control not in line.content.plain
        for folded_line in line.folds:
            content = folded_line.content
            strip = Strip(content.render_segments(), cell_length=content.cell_length)
            rendered = strip.render(Console(force_terminal=True, color_system=None))
            assert control not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["\x0b", "\x0c"])
async def test_ansi_vertical_tab_and_form_feed_are_line_feeds(control: str) -> None:
    state = ansi.TerminalState(lambda _text: None)

    await state.write(f"first{control}second")

    assert [line.content.plain for line in state.buffer.lines] == ["first", "second"]


@pytest.mark.asyncio
async def test_ansi_shift_out_and_shift_in_invoke_g1_and_g0() -> None:
    state = ansi.TerminalState(lambda _text: None)

    await state.write("\x1b)0q\x0eq\x0fq")

    assert state.buffer.lines[0].content.plain == "q─q"


@pytest.mark.asyncio
async def test_ansi_horizontal_tab_advances_without_storing_control_code() -> None:
    state = ansi.TerminalState(lambda _text: None)

    await state.write("a\tb")

    assert state.buffer.lines[0].content.plain == "a       b"
    assert state.buffer.cursor == (0, 9)


@pytest.mark.asyncio
async def test_ansi_osc_bell_terminator_remains_supported() -> None:
    state = ansi.TerminalState(lambda _text: None)
    path = "/tmp/bell-terminated"
    encoded = base64.b64encode(path.encode()).decode()

    await state.write(f"\x1b]2025;b64:{encoded}")
    await state.write("\x07")
    await state.write("ready")

    assert state.current_directory == path
    assert state.buffer.lines[0].content.plain == "ready"


@pytest.mark.asyncio
async def test_ansi_xterm_title_osc_consumes_bell_terminator() -> None:
    state = ansi.TerminalState(lambda _text: None)

    await state.write("\x1b]0;title")
    await state.write("\x07visible")

    assert state.buffer.lines[0].content.plain == "visible"


@pytest.mark.asyncio
async def test_ansi_decodes_b64_command_with_semicolons() -> None:
    state = ansi.TerminalState(lambda _text: None)
    command = "echo a;b && printf '\\033'"
    encoded = base64.b64encode(command.encode()).decode()

    await state.write(f"\x1b]2026;b64:{encoded}\x1b\\")

    assert state.last_command == command


def test_cd_command_quotes_unix_path() -> None:
    panel = ShellPanel()
    panel._shell_name = "zsh"

    assert panel._cd_command("/tmp/a b/it's") == "cd '/tmp/a b/it'\"'\"'s'"


def test_cd_command_quotes_powershell_path() -> None:
    panel = ShellPanel()
    panel._shell_name = "pwsh"

    assert panel._cd_command("C:\\Users\\O'Brien") == "Set-Location -LiteralPath 'C:\\Users\\O''Brien'"
