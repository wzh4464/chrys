# Copyright (c) 2026 Chrys. All rights reserved.
# Adapted from toad (https://github.com/batrachianai/toad) — decoupled from Conversation,
# writes to a single persistent Terminal instance.

from __future__ import annotations

import asyncio
import codecs
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from time import monotonic, sleep
from typing import TYPE_CHECKING

from textual import log
from textual.message import Message

from chrys.app.tui.terminal._pty_backend import (
    IS_WINDOWS,
    WINPTY_ERRORS,
    WINPTY_FACTORY,
    WinPTYProtocol,
    kill_posix_process,
    open_pty,
    prepare_child_pty,
    resize_pty,
    set_pty_nonblocking,
)
from chrys.app.tui.terminal.shell_read import shell_read
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.platform.runtime_paths import reorder_path_demoting_runtime

if TYPE_CHECKING:
    from chrys.app.tui.terminal.widget import Terminal

WinPTY = WINPTY_FACTORY

# Read buffering for Windows ConPTY — matches shell_read() on Unix.
_WIN_BUFFER_PERIOD: float = 1 / 100  # 10ms pause between batch reads
_WIN_BUFFER_MAX: float = 1 / 60  # ~16.6ms total batching window
_CLOSE_TASK_TIMEOUT: float = 2.0
_DEFAULT_PTY_COLS = 80
_DEFAULT_PTY_ROWS = 24
_PTY_OPERATION_ERRORS = (OSError, RuntimeError, *WINPTY_ERRORS)
_PTY_READ_ERRORS = (EOFError, *_PTY_OPERATION_ERRORS)

_SHELL_FAILED = msg("tui.terminal.shell.failed", fallback="Shell failed: {error}")
_SHELL_START_FAILED = msg("tui.terminal.shell.start_failed", fallback="Unable to start shell: {error}")


@dataclass
class ShellFinished(Message):
    """The shell process exited."""


class Shell:
    """PTY-based shell that writes all output to a single Terminal widget."""

    def __init__(
        self,
        terminal: Terminal,
        working_directory: str,
        shell_command: str = "",
        on_directory_changed: Callable[[str], None] | None = None,
        on_command_executed: Callable[[str], None] | None = None,
        on_error: Callable[[MessageRef | str], None] | None = None,
    ) -> None:
        self.terminal = terminal
        self.working_directory = working_directory
        if IS_WINDOWS:
            self.shell_command = shell_command or "pwsh"
        else:
            self.shell_command = shell_command or os.environ.get("SHELL", "sh")
        self.on_directory_changed = on_directory_changed
        self.on_command_executed = on_command_executed
        self.on_error = on_error

        self.master: int | None = None
        self._winpty: WinPTYProtocol | None = None
        self._transport: asyncio.BaseTransport | None = None
        self._task: asyncio.Task | None = None
        self._finished: bool = False
        self._ready_event: asyncio.Event = asyncio.Event()
        self._hide_echo: set[bytes] = set()
        self._pid: int | None = None
        self._init_tmpdir: str | None = None
        self._win_queue: asyncio.Queue[str] | None = None
        self._win_reader_thread: threading.Thread | None = None
        self._input_queue: asyncio.Queue[bytes] | None = None
        self._input_writer_task: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._resize_lock = asyncio.Lock()
        self._pty_cols = _DEFAULT_PTY_COLS
        self._pty_rows = _DEFAULT_PTY_ROWS
        self._applied_pty_size: tuple[int, int] | None = None

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def pty_size(self) -> tuple[int, int] | None:
        """Last PTY size successfully applied to the shell backend."""
        return self._applied_pty_size

    def _is_ready(self) -> bool:
        """Check if the PTY backend is available for I/O."""
        if IS_WINDOWS:
            return self._winpty is not None
        return self.master is not None

    async def wait_for_ready(self) -> None:
        await self._ready_event.wait()

    def _coerce_pty_size(self, width: int, height: int) -> tuple[int, int]:
        """Return a positive PTY size, preserving the last valid dimension across hidden-layout resizes."""
        if width <= 0 or height <= 0:
            return self._pty_cols, self._pty_rows
        self._pty_cols = width
        self._pty_rows = height
        return self._pty_cols, self._pty_rows

    def _mark_pty_size_applied(self, width: int, height: int) -> None:
        self._applied_pty_size = (width, height)

    def _pty_size_is_applied(self, width: int, height: int) -> bool:
        return self._applied_pty_size == (width, height)

    async def send(self, command: str, width: int, height: int) -> None:
        """Send a command to the shell (programmatic use).

        Shell integration hooks handle directory and command tracking automatically.
        """
        await self._ready_event.wait()
        if not self._is_ready():
            return

        await self.update_size_async(width, height)
        await self.write(f"{command}\n")

    async def send_input(self, text: str, paste: bool = False) -> None:
        """Send raw input to the shell (e.g. for interactive commands)."""
        await self._ready_event.wait()
        if not self._is_ready():
            return
        await self._sync_terminal_size_before_input()
        if paste and self.terminal.state.bracketed_paste:
            text = f"\x1b[200~{text}\x1b[201~"
        await self.write(text)

    async def _sync_terminal_size_before_input(self) -> None:
        """Apply the visible terminal size before bytes can trigger a prompt repaint."""
        while True:
            terminal_size = self.terminal.size
            target = (terminal_size.width, terminal_size.height)
            await self.update_size_async(*target)
            current_size = self.terminal.size
            if (current_size.width, current_size.height) == target:
                return

    def start(self) -> None:
        """Start the shell process."""
        assert self._task is None
        self._task = asyncio.create_task(self.run(), name=repr(self))
        log("shell starting")

    async def interrupt(self) -> None:
        """Send Ctrl+C to the shell."""
        await self.write(b"\x03")

    def update_size(self, width: int, height: int) -> None:
        """Resize the PTY."""
        # Sync resize is only used during startup or under update_size_async's lock.
        # Do not call it concurrently with update_size_async.
        width, height = self._coerce_pty_size(width, height)
        if self._pty_size_is_applied(width, height):
            return
        if IS_WINDOWS:
            if self._winpty is None:
                return
            with suppress(*_PTY_OPERATION_ERRORS):
                self._winpty.set_size(width, height)
                self._mark_pty_size_applied(width, height)
        else:
            if self.master is None:
                return
            if resize_pty(self.master, width, height):
                self._mark_pty_size_applied(width, height)

    async def update_size_async(self, width: int, height: int) -> None:
        """Resize the PTY without blocking the event loop.

        On Windows, ConPTY ``set_size`` triggers a full buffer reflow which
        can be slow.  This offloads the call to a worker thread.
        """
        async with self._resize_lock:
            width, height = self._coerce_pty_size(width, height)
            if self._pty_size_is_applied(width, height):
                return
            if IS_WINDOWS:
                if self._winpty is None:
                    return
                try:
                    await asyncio.to_thread(self._winpty.set_size, width, height)
                except _PTY_OPERATION_ERRORS:
                    pass
                else:
                    self._mark_pty_size_applied(width, height)
            else:
                self.update_size(width, height)

    async def write(self, text: str | bytes, hide_echo: bool = False) -> int:
        """Write to the PTY."""
        if not self._is_ready():
            return 0
        text_bytes = text.encode("utf-8", "ignore") if isinstance(text, str) else text

        if hide_echo:
            for line in text_bytes.split(b"\n"):
                if line:
                    self._hide_echo.add(line)

        async with self._write_lock:
            try:
                if IS_WINDOWS:
                    # pywinpty.write() accepts str
                    winpty = self._winpty
                    if winpty is None:
                        return 0
                    text_str = text_bytes.decode("utf-8", "replace")
                    await asyncio.to_thread(winpty.write, text_str)
                    result = len(text_bytes)
                else:
                    master = self.master
                    if master is None:
                        return 0
                    result = await asyncio.to_thread(os.write, master, text_bytes)
            except _PTY_OPERATION_ERRORS:
                return 0
        return result

    async def write_input_queued(self, text: str | bytes) -> int:
        """Queue terminal input so high-rate keys/mouse events can be coalesced."""
        if not self._is_ready():
            return 0
        text_bytes = text.encode("utf-8", "ignore") if isinstance(text, str) else text
        if not text_bytes:
            return 0
        if self._input_queue is None:
            self._input_queue = asyncio.Queue()
        self._input_queue.put_nowait(text_bytes)
        if self._input_writer_task is None or self._input_writer_task.done():
            self._input_writer_task = asyncio.create_task(self._input_writer_loop())
        return len(text_bytes)

    async def _input_writer_loop(self) -> None:
        queue = self._input_queue
        if queue is None:
            return
        while not self._finished:
            try:
                first = await queue.get()
            except asyncio.CancelledError:
                break
            await asyncio.sleep(0)
            chunks = [first]
            while True:
                try:
                    chunks.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            try:
                await self._sync_terminal_size_before_input()
                await self.write(b"".join(chunks))
            finally:
                for _ in chunks:
                    queue.task_done()

    def _cancel_input_writer(self) -> None:
        if self._input_writer_task is not None:
            self._input_writer_task.cancel()
            self._input_writer_task = None
        self._input_queue = None

    def terminate(self) -> None:
        """Terminate the shell process and close the PTY."""
        self._cancel_input_writer()
        if self._finished:
            return
        if IS_WINDOWS:
            # Kill the process if we have a PID, then drop the handle
            if self._pid is not None:
                with suppress(OSError):
                    os.kill(self._pid, signal.SIGTERM)
            self._winpty = None
            # Ensure the consumer loop exits even if the reader thread
            # hasn't enqueued its EOF sentinel yet.
            if self._win_queue is not None:
                with suppress(Exception):
                    self._win_queue.put_nowait("")
        else:
            # Close via the asyncio transport so the fd isn't double-closed
            if self._transport is not None:
                self._transport.close()
                self._transport = None
            self.master = None
            # Kill the process group to ensure child processes (inner apps) are also killed
            if self._pid is not None:
                kill_posix_process(self._pid, fallback_to_process=True)
        self._pid = None
        self._mark_finished()

    async def close(self) -> None:
        """Terminate the shell and wait briefly for the reader task to unwind."""
        self.terminate()
        task = self._task
        if task is None:
            return
        try:
            if task is not asyncio.current_task() and not task.done():
                await asyncio.wait_for(task, timeout=_CLOSE_TASK_TIMEOUT)
        except TimeoutError:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        except Exception as error:
            log(f"shell async close failed during teardown: {error!r}")
        finally:
            if self._task is task:
                self._task = None

    async def run(self) -> None:
        """Main PTY read loop — routes output to the Terminal widget."""
        try:
            if IS_WINDOWS:
                await self._run_windows()
            else:
                await self._run_unix()
        except Exception as error:
            log(f"shell run failed: {error!r}")
            if self.on_error:
                self.on_error(_SHELL_FAILED.bind(error=str(error)))
        finally:
            if not self._finished:
                self._mark_finished(post_message=True)

    def _cleanup_init_tmpdir(self) -> None:
        if self._init_tmpdir:
            shutil.rmtree(self._init_tmpdir, ignore_errors=True)
            self._init_tmpdir = None

    def _mark_finished(self, post_message: bool = False) -> None:
        was_finished = self._finished
        self._finished = True
        self._ready_event.set()
        self._cleanup_init_tmpdir()
        if post_message and not was_finished:
            self.terminal.post_message(ShellFinished())

    # ── Unix PTY implementation ──────────────────────────────────────

    async def _run_unix(self) -> None:
        """Unix PTY read loop using pty.openpty()."""
        if self._finished:
            return

        current_directory = self.working_directory

        master, slave = open_pty()
        self.master = master

        set_pty_nonblocking(master)
        cols, rows = self._coerce_pty_size(self.terminal.size.width, self.terminal.size.height)
        # Size the slave before spawn so the child inherits termios geometry
        # atomically instead of briefly seeing the kernel default 0x0 size.
        if resize_pty(slave, cols, rows):
            self._mark_pty_size_applied(cols, rows)

        env = os.environ.copy()
        env["FORCE_COLOR"] = "1"
        env["TTY_COMPATIBLE"] = "1"
        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"
        env["CHRYS"] = "1"
        env["CLICOLOR"] = "1"
        env["PYTHONUTF8"] = "1"
        # Prevent macOS "Restored session" message from /etc/zshrc_Apple_Terminal
        env.pop("TERM_SESSION_ID", None)
        env.pop("TERM_PROGRAM", None)
        env.pop("TERM_PROGRAM_VERSION", None)
        reorder_path_demoting_runtime(env)

        argv = self._prepare_shell_init(self.shell_command, env)

        def setup_pty() -> None:
            prepare_child_pty(slave)

        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=slave,
                stdout=slave,
                stderr=slave,
                env=env,
                cwd=current_directory,
                preexec_fn=setup_pty,
            )
        except Exception as error:
            log(f"unable to start shell: {error!r}")
            if self.on_error:
                self.on_error(_SHELL_START_FAILED.bind(error=str(error)))
            with suppress(OSError):
                os.close(slave)
            with suppress(OSError):
                os.close(master)
            self.master = None
            self._mark_finished(post_message=True)
            return
        self._pid = process.pid
        if self._finished:
            kill_posix_process(process.pid, fallback_to_process=True)
            with suppress(OSError):
                os.close(slave)
            with suppress(OSError):
                os.close(master)
            with suppress(Exception):
                await process.wait()
            self.master = None
            self._pid = None
            return

        os.close(slave)
        buffer_size = 64 * 1024
        reader = asyncio.StreamReader(buffer_size)
        protocol = asyncio.StreamReaderProtocol(reader)

        loop = asyncio.get_running_loop()
        try:
            transport, _ = await loop.connect_read_pipe(lambda: protocol, os.fdopen(master, "rb", 0))
        except Exception as error:
            log(f"unable to connect shell pipe: {error!r}")
            if self.on_error:
                self.on_error(_SHELL_START_FAILED.bind(error=str(error)))
            kill_posix_process(process.pid, fallback_to_process=False)
            with suppress(Exception):
                await process.wait()
            self.master = None
            self._mark_finished(post_message=True)
            return
        self._transport = transport

        # Wire terminal input back to PTY
        self.terminal.set_write_to_stdin(self.write_input_queued)

        self._ready_event.set()

        unicode_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        while True:
            try:
                data = await shell_read(reader, buffer_size)
            except Exception:
                break

            for string_bytes in list(self._hide_echo):
                if string_bytes in data:
                    # Strip just the hidden portion, preserving surrounding content
                    data = data.replace(string_bytes, b"", 1)
                    self._hide_echo.discard(string_bytes)

            if line := unicode_decoder.decode(data, final=not data):
                try:
                    await self.terminal.write(line)
                except Exception:
                    pass

                new_directory = self.terminal.current_directory
                if new_directory and new_directory != current_directory:
                    current_directory = new_directory
                    if self.on_directory_changed:
                        self.on_directory_changed(current_directory)

                new_command = self.terminal.last_command
                if new_command:
                    self.terminal.last_command = None
                    if self.on_command_executed:
                        self.on_command_executed(new_command)

            if not data:
                break

        self._cancel_input_writer()
        if self._transport is not None:
            self._transport.close()
            self._transport = None
        self.master = None
        with suppress(Exception):
            await process.wait()
        log("shell finished")
        self._mark_finished(post_message=True)

    # ── Windows ConPTY implementation ────────────────────────────────

    @staticmethod
    def _win_reader_loop(
        win_pty: WinPTYProtocol,
        queue: asyncio.Queue[str],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Persistent reader thread for Windows ConPTY.

        Polls ``win_pty.read(blocking=False)`` then drains any immediately
        available data (check-before-sleep) before pushing the batch to the
        event loop via *queue*.  Runs in a dedicated daemon thread to avoid
        the overhead of creating/tearing-down a thread per read cycle.

        Non-blocking reads are used instead of ``blocking=True`` because the
        ConPTY pipe handle stays open (held by the WinPTY object) even after
        the child process exits.  A blocking read would hang forever in that
        case, preventing ``isalive()`` from ever being checked.
        """

        def _enqueue(data: str) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, data)
            except RuntimeError:
                # Event loop already closed (app shutting down)
                pass

        while True:
            if not win_pty.isalive():
                _enqueue("")
                return

            # Use non-blocking reads with polling so we can reliably
            # detect process exit.  A blocking read on the ConPTY pipe
            # can hang forever after the child exits because the pipe
            # handle stays open (held by the WinPTY object).  The old
            # per-call asyncio.to_thread approach masked this — the
            # natural overhead between calls gave isalive() time to
            # report the process as dead.
            try:
                data = win_pty.read(blocking=False)
            except _PTY_READ_ERRORS:
                _enqueue("")
                return
            if not data:
                sleep(_WIN_BUFFER_PERIOD)
                continue

            # Batch: drain immediately-available data (check before sleep).
            chunks = [data]
            deadline = monotonic() + _WIN_BUFFER_MAX
            while monotonic() < deadline:
                if not win_pty.isalive():
                    break
                try:
                    chunk = win_pty.read(blocking=False)
                except Exception:
                    break
                if chunk:
                    chunks.append(chunk)
                else:
                    # Nothing ready — sleep once then retry
                    sleep(_WIN_BUFFER_PERIOD)
                    if not win_pty.isalive():
                        break
                    try:
                        chunk = win_pty.read(blocking=False)
                    except Exception:
                        break
                    if chunk:
                        chunks.append(chunk)
                    else:
                        break

            _enqueue("".join(chunks))

    @staticmethod
    def _env_to_block(env: dict[str, str]) -> str:
        """Convert an env dict to a Windows environment block string.

        Windows CreateProcess expects a null-separated environment block.
        """
        return "\0".join(f"{k}={v}" for k, v in env.items()) + "\0\0"

    async def _run_windows(self) -> None:
        """Windows PTY read loop using pywinpty (ConPTY)."""
        if self._finished:
            return

        current_directory = self.working_directory

        cols, rows = self._coerce_pty_size(self.terminal.size.width, self.terminal.size.height)

        env = os.environ.copy()
        env["FORCE_COLOR"] = "1"
        env["TTY_COMPATIBLE"] = "1"
        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"
        env["CHRYS"] = "1"
        env["CLICOLOR"] = "1"
        env["PYTHONUTF8"] = "1"
        reorder_path_demoting_runtime(env)

        argv = self._prepare_shell_init(self.shell_command, env)
        shell = subprocess.list2cmdline(argv)
        env_block = self._env_to_block(env)

        try:
            win_pty = WinPTY(cols, rows)
            self._winpty = win_pty
            win_pty.spawn(shell, cwd=current_directory, env=env_block)
            self._mark_pty_size_applied(cols, rows)
        except Exception as error:
            log(f"unable to start shell: {error!r}")
            if self.on_error:
                self.on_error(_SHELL_START_FAILED.bind(error=str(error)))
            self._winpty = None
            self._mark_finished(post_message=True)
            return

        self._pid = win_pty.pid
        if self._finished:
            if self._pid is not None:
                with suppress(OSError):
                    os.kill(self._pid, signal.SIGTERM)
            self._pid = None
            self._winpty = None
            return

        # ConPTY sends its own reflow VT sequences on resize — tell the
        # terminal emulator to skip its internal _reflow() to avoid
        # double-reflow display corruption.
        self.terminal._external_reflow = True

        # Wire terminal input back to PTY
        self.terminal.set_write_to_stdin(self.write_input_queued)
        self._ready_event.set()

        # Start persistent reader thread (avoids per-read thread creation overhead)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str] = asyncio.Queue()
        self._win_queue = queue
        reader_thread = threading.Thread(
            target=self._win_reader_loop,
            args=(win_pty, queue, loop),
            daemon=True,
            name="chrys-winpty-reader",
        )
        self._win_reader_thread = reader_thread
        reader_thread.start()

        # On re-entry to shell mode, ConPTY emits its own cursor-home +
        # full-screen-erase sequences to claim the screen.  We let those
        # through so the widget's scrollback buffer is cleared and stays
        # in sync with ConPTY's internal coordinate system.  This means
        # the previous session's content is wiped on re-entry — the
        # tradeoff for stable absolute-positioning later in the session.
        while True:
            try:
                text = await queue.get()
            except Exception:
                break

            if not text:
                break

            exit_after_write = False
            chunks = [text]
            while True:
                try:
                    chunk = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if not chunk:
                    exit_after_write = True
                    break
                chunks.append(chunk)
            if len(chunks) > 1:
                text = "".join(chunks)

            # Strip hidden echo (convert to str matching for Windows path)
            for string_bytes in list(self._hide_echo):
                hidden_str = string_bytes.decode("utf-8", "ignore")
                if hidden_str and hidden_str in text:
                    text = text.replace(hidden_str, "", 1)
                    self._hide_echo.discard(string_bytes)

            if text:
                try:
                    await self.terminal.write(text)
                except Exception:
                    pass

                new_directory = self.terminal.current_directory
                if new_directory and new_directory != current_directory:
                    current_directory = new_directory
                    if self.on_directory_changed:
                        self.on_directory_changed(current_directory)

                new_command = self.terminal.last_command
                if new_command:
                    self.terminal.last_command = None
                    if self.on_command_executed:
                        self.on_command_executed(new_command)

            if exit_after_write:
                break

        self._cancel_input_writer()
        self._win_queue = None
        self._win_reader_thread = None
        self._winpty = None
        log("shell finished")
        self._mark_finished(post_message=True)

    # ── Shell integration hooks ──────────────────────────────────────

    def _split_shell_command(self, shell_cmd: str) -> list[str]:
        if os.path.exists(shell_cmd):
            return [shell_cmd]
        try:
            argv = shlex.split(shell_cmd, posix=not IS_WINDOWS)
        except ValueError:
            argv = []
        return argv or [shell_cmd]

    def _prepare_shell_init(self, shell_cmd: str, env: dict) -> list[str]:
        """Set up shell integration hooks via init files. Returns the shell argv.

        Hooks are loaded silently during shell startup (no echo).
        Uses OSC 2025 (directory) and OSC 2026 (command) escape sequences.
        """
        argv = self._split_shell_command(shell_cmd)
        shell_name = os.path.basename(argv[0]).lower()
        shell_lower = " ".join([shell_name, *argv[1:]]).lower()
        if "zsh" in shell_lower:
            return self._init_zsh(argv, env)
        if "bash" in shell_lower:
            return self._init_bash(argv, env)
        if "fish" in shell_lower:
            return self._init_fish(argv)
        if "pwsh" in shell_lower or "powershell" in shell_lower:
            return self._init_pwsh(argv)
        return argv

    def _init_zsh(self, argv: list[str], env: dict) -> list[str]:
        """zsh: use ZDOTDIR with a custom .zshrc that sources the original then adds hooks."""
        init_dir = tempfile.mkdtemp(prefix="chrys_zsh_")
        self._init_tmpdir = init_dir
        original_zdotdir = env.get("ZDOTDIR", os.path.expanduser("~"))
        with open(os.path.join(init_dir, ".zshrc"), "w", encoding="utf-8") as f:
            f.write(f"ZDOTDIR={shlex.quote(original_zdotdir)}\n")
            f.write('[[ -f "$ZDOTDIR/.zshrc" ]] && source "$ZDOTDIR/.zshrc"\n')
            f.write('__chrys_b64() { printf "%s" "$1" | base64 | tr -d "\\n"; }\n')
            f.write('__chrys_preexec() { printf "\\e]2026;b64:%s\\e\\\\" "$(__chrys_b64 "$1")"; }\n')
            f.write('__chrys_precmd() { printf "\\e]2025;b64:%s\\e\\\\" "$(__chrys_b64 "$(pwd)")"; }\n')
            f.write("precmd_functions+=(__chrys_precmd)\n")
            f.write("preexec_functions+=(__chrys_preexec)\n")
        env["ZDOTDIR"] = init_dir
        return argv

    def _init_bash(self, argv: list[str], env: dict) -> list[str]:
        """bash: use --rcfile with a temp file that sources .bashrc then adds hooks."""
        init_dir = tempfile.mkdtemp(prefix="chrys_bash_")
        self._init_tmpdir = init_dir
        rcfile = os.path.join(init_dir, "init.bash")
        with open(rcfile, "w", encoding="utf-8") as f:
            f.write("[[ -f ~/.bashrc ]] && source ~/.bashrc\n")
            f.write('__chrys_b64() { printf "%s" "$1" | base64 | tr -d "\\n"; }\n')
            f.write("__chrys_at_prompt=true\n")
            f.write("__chrys_preexec() {\n")
            f.write('  [ "$__chrys_at_prompt" = true ] || return\n')
            f.write("  __chrys_at_prompt=false\n")
            f.write('  printf "\\e]2026;b64:%s\\e\\\\" "$(__chrys_b64 "$BASH_COMMAND")"\n')
            f.write("}\n")
            f.write("trap '__chrys_preexec' DEBUG\n")
            f.write("__chrys_precmd() {\n")
            f.write("  __chrys_at_prompt=true\n")
            f.write('  printf "\\e]2025;b64:%s\\e\\\\" "$(__chrys_b64 "$(pwd)")"\n')
            f.write("}\n")
            f.write('PROMPT_COMMAND="__chrys_precmd${PROMPT_COMMAND:+;$PROMPT_COMMAND}"\n')
        return [*argv, "--rcfile", rcfile]

    def _init_fish(self, argv: list[str]) -> list[str]:
        """fish: use -C flag to run init commands after config."""
        hooks = (
            'function __chrys_b64; printf "%s" "$argv[1]" | base64 | tr -d "\\n"; end; '
            "function __chrys_preexec --on-event fish_preexec; "
            'printf "\\e]2026;b64:%s\\e\\\\" (__chrys_b64 "$argv"); end; '
            "function __chrys_postcmd --on-event fish_postexec; "
            'printf "\\e]2025;b64:%s\\e\\\\" (__chrys_b64 (pwd)); end'
        )
        return [*argv, "-C", hooks]

    def _init_pwsh(self, argv: list[str]) -> list[str]:
        """pwsh: use -NoExit -Command to define hooks before interactive prompt."""
        hooks = (
            "function prompt {"
            " $e = [char]27; $st = [char]92;"
            " $cwd = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes((Get-Location).Path));"
            ' Write-Host -NoNewline "$e]2025;b64:$cwd$e$st";'
            ' return "PS $($executionContext.SessionState.Path.CurrentLocation)> "'
            "};"
            "if (Get-Module PSReadLine) {"
            " Set-PSReadLineOption -AddToHistoryHandler {"
            "  param($cmd)"
            "  $e = [char]27; $st = [char]92;"
            "  $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($cmd));"
            '  [Console]::Write("$e]2026;b64:$encoded$e$st");'
            "  return $true"
            " }"
            "}"
        )
        return [*argv, "-NoExit", "-Command", hooks]
