# Copyright (c) 2026 Chrys. All rights reserved.

"""Lifecycle tests for main-screen cleanup delegation."""

from __future__ import annotations

import asyncio
import gc
from collections.abc import Awaitable, Callable
from types import MethodType, SimpleNamespace

from pytest import WarningsRecorder

from chrys.app.tui.screens.main.screen import MainScreen
from chrys.app.tui.util.git_branch import GIT_BRANCH_POLL_INTERVAL_SECONDS, GitBranchSnapshot


class _Timer:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _queue_screen(
    run_operation: Callable[[str, str], Awaitable[GitBranchSnapshot | None]],
    *,
    current_cwd: str = "",
    displayed_cwd: str | None = None,
) -> SimpleNamespace:
    screen = SimpleNamespace(
        _git_branch_closed=False,
        _git_branch_pending_operation=None,
        _git_branch_task=None,
        _git_branch_monitor=SimpleNamespace(active=True),
        _git_branch_retry_cwd_on_display_sync=None,
        current_cwd=current_cwd,
        chat_workspace_cwd=displayed_cwd if displayed_cwd is not None else current_cwd,
        applied=[],
        sync_count=0,
    )
    screen._workspace_cwd = lambda: screen.current_cwd
    screen._displayed_workspace_cwd = MethodType(MainScreen._displayed_workspace_cwd, screen)
    screen._run_git_branch_operation = run_operation
    screen._apply_git_branch_snapshot = lambda snapshot: screen.applied.append(snapshot.branch)
    screen._sync_git_branch_poll_timer = lambda: setattr(screen, "sync_count", screen.sync_count + 1)
    screen._drain_git_branch_operations = MethodType(MainScreen._drain_git_branch_operations, screen)
    screen._queue_git_branch_operation = MethodType(MainScreen._queue_git_branch_operation, screen)
    screen._queue_git_branch_configure = MethodType(MainScreen._queue_git_branch_configure, screen)
    return screen


def test_unmount_shuts_down_buddy_before_flushing_notifications() -> None:
    calls: list[str] = []

    class _Subscriptions:
        async def unsubscribe_all(self) -> None:
            calls.append("unsubscribe")

    class _BuddyCommand:
        async def shutdown(self) -> None:
            calls.append("buddy_shutdown")

    async def flush_notifications() -> None:
        calls.append("flush_notifications")

    async def stop_git_branch_monitor() -> None:
        pass

    screen = SimpleNamespace(
        _subscriptions=_Subscriptions(),
        _suggestions=SimpleNamespace(buddy_command=_BuddyCommand()),
        _stop_terminal_title_activity_timer=lambda: None,
        _stop_git_branch_refresh_timer=lambda: None,
        _stop_git_branch_poll_timer=lambda: None,
        _stop_git_branch_monitor=stop_git_branch_monitor,
        _flush_settings_save=flush_notifications,
        _locale_controller=None,
    )

    asyncio.run(MainScreen.on_unmount(screen))

    assert calls == ["unsubscribe", "buddy_shutdown", "flush_notifications"]


def test_git_branch_refresh_schedule_debounces_existing_timer() -> None:
    old_timer = _Timer()
    new_timer = _Timer()
    calls: list[tuple[float, object]] = []

    def set_timer(delay: float, callback: object) -> _Timer:
        calls.append((delay, callback))
        return new_timer

    screen = SimpleNamespace(
        _git_branch_monitor=SimpleNamespace(active=True),
        _git_branch_closed=False,
        _git_branch_refresh_timer=old_timer,
        _refresh_git_branch=lambda: None,
        set_timer=set_timer,
    )

    MainScreen._schedule_git_branch_refresh(screen)

    assert old_timer.stopped is True
    assert screen._git_branch_refresh_timer is new_timer
    assert calls == [(0.1, screen._refresh_git_branch)]


def test_git_branch_refresh_timer_queues_async_refresh() -> None:
    queued: list[str] = []
    screen = SimpleNamespace(
        _git_branch_monitor=SimpleNamespace(active=True),
        _git_branch_closed=False,
        _git_branch_refresh_timer=_Timer(),
        _queue_git_branch_refresh=lambda: queued.append("refresh"),
    )

    MainScreen._refresh_git_branch(screen)

    assert screen._git_branch_refresh_timer is None
    assert queued == ["refresh"]


def test_git_branch_poll_timer_stops_when_native_watcher_is_active() -> None:
    poll_timer = _Timer()
    screen = SimpleNamespace(
        _git_branch_monitor=SimpleNamespace(active=True, watching=True),
        _git_branch_closed=False,
        _git_branch_poll_timer=poll_timer,
    )
    screen._stop_git_branch_poll_timer = lambda: MainScreen._stop_git_branch_poll_timer(screen)

    MainScreen._sync_git_branch_poll_timer(screen)

    assert poll_timer.stopped is True
    assert screen._git_branch_poll_timer is None


def test_git_branch_poll_timer_starts_when_native_watcher_is_unavailable() -> None:
    poll_timer = _Timer()
    calls: list[tuple[float, object]] = []

    def set_interval(delay: float, callback: object) -> _Timer:
        calls.append((delay, callback))
        return poll_timer

    screen = SimpleNamespace(
        _git_branch_monitor=SimpleNamespace(active=True, watching=False),
        _git_branch_closed=False,
        _git_branch_poll_timer=None,
        _poll_git_branch=lambda: None,
        set_interval=set_interval,
    )

    MainScreen._sync_git_branch_poll_timer(screen)

    assert screen._git_branch_poll_timer is poll_timer
    assert calls == [(GIT_BRANCH_POLL_INTERVAL_SECONDS, screen._poll_git_branch)]


def test_git_branch_queue_without_running_loop_keeps_pending_operation_without_coroutine_warning(
    recwarn: WarningsRecorder,
) -> None:
    async def run_operation(_operation: str, cwd: str) -> GitBranchSnapshot:
        return GitBranchSnapshot(cwd=cwd, branch="branch")

    screen = _queue_screen(run_operation, current_cwd="/repo")

    MainScreen._queue_git_branch_operation(screen, "configure", "/repo")
    gc.collect()

    assert screen._git_branch_task is None
    assert screen._git_branch_pending_operation == ("configure", "/repo")
    runtime_warnings = [warning for warning in recwarn if issubclass(warning.category, RuntimeWarning)]
    assert runtime_warnings == []


def test_git_branch_configure_same_cwd_does_not_overwrite_pending_start() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []

        async def run_operation(operation: str, cwd: str) -> GitBranchSnapshot:
            calls.append((operation, cwd))
            return GitBranchSnapshot(cwd=cwd, branch=f"{operation}-branch")

        screen = _queue_screen(run_operation, current_cwd="/repo")

        MainScreen._queue_git_branch_operation(screen, "start", "/repo")
        MainScreen._queue_git_branch_operation(screen, "configure", "/repo")
        await screen._git_branch_task

        assert calls == [("start", "/repo")]
        assert screen.applied == ["start-branch"]

    asyncio.run(run())


def test_git_branch_configure_new_cwd_preserves_pending_start_semantics() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []

        async def run_operation(operation: str, cwd: str) -> GitBranchSnapshot:
            calls.append((operation, cwd))
            return GitBranchSnapshot(cwd=cwd, branch=f"{operation}-branch")

        screen = _queue_screen(run_operation, current_cwd="/repo-a")

        MainScreen._queue_git_branch_operation(screen, "start", "/repo-a")
        screen.current_cwd = "/repo-b"
        screen.chat_workspace_cwd = "/repo-b"
        MainScreen._queue_git_branch_operation(screen, "configure", "/repo-b")
        await screen._git_branch_task

        assert calls == [("start", "/repo-b")]
        assert screen.applied == ["start-branch"]

    asyncio.run(run())


def test_git_branch_queue_drops_stale_configure_before_applying_newer_cwd() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def run_operation(operation: str, cwd: str) -> GitBranchSnapshot:
            calls.append((operation, cwd))
            if cwd == "/repo-a":
                first_started.set()
                await release_first.wait()
                return GitBranchSnapshot(cwd=cwd, branch="branch-a")
            return GitBranchSnapshot(cwd=cwd, branch="branch-b")

        screen = _queue_screen(run_operation, current_cwd="/repo-a")

        MainScreen._queue_git_branch_operation(screen, "configure", "/repo-a")
        await first_started.wait()
        screen.current_cwd = "/repo-b"
        screen.chat_workspace_cwd = "/repo-b"
        MainScreen._queue_git_branch_operation(screen, "configure", "/repo-b")
        release_first.set()
        await screen._git_branch_task

        assert calls == [("configure", "/repo-a"), ("configure", "/repo-b")]
        assert screen.applied == ["branch-b"]
        assert screen.sync_count == 1
        assert screen._git_branch_task is None

    asyncio.run(run())


def test_git_branch_queue_skips_refresh_when_configure_is_pending() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []

        async def run_operation(operation: str, cwd: str) -> GitBranchSnapshot:
            calls.append((operation, cwd))
            return GitBranchSnapshot(cwd=cwd, branch=f"{operation}-branch")

        screen = _queue_screen(run_operation, current_cwd="/repo")

        MainScreen._queue_git_branch_operation(screen, "configure", "/repo")
        MainScreen._queue_git_branch_operation(screen, "refresh", "")
        await screen._git_branch_task

        assert calls == [("configure", "/repo")]
        assert screen.applied == ["configure-branch"]
        assert screen.sync_count == 1

    asyncio.run(run())


def test_git_branch_queue_drops_stale_refresh_when_workspace_changes() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []
        refresh_started = asyncio.Event()
        release_refresh = asyncio.Event()

        async def run_operation(operation: str, cwd: str) -> GitBranchSnapshot:
            calls.append((operation, cwd))
            if operation == "refresh":
                refresh_started.set()
                await release_refresh.wait()
                return GitBranchSnapshot(cwd="/repo-a", branch="old-branch")
            return GitBranchSnapshot(cwd=cwd, branch="new-branch")

        screen = _queue_screen(run_operation, current_cwd="/repo-a")

        MainScreen._queue_git_branch_operation(screen, "refresh", "")
        await refresh_started.wait()
        screen.current_cwd = "/repo-b"
        screen.chat_workspace_cwd = "/repo-b"
        MainScreen._queue_git_branch_operation(screen, "configure", "/repo-b")
        release_refresh.set()
        await screen._git_branch_task

        assert calls == [("refresh", ""), ("configure", "/repo-b")]
        assert screen.applied == ["new-branch"]
        assert screen.sync_count == 1

    asyncio.run(run())


def test_git_branch_queue_drops_result_when_displayed_cwd_differs() -> None:
    async def run() -> None:
        async def run_operation(operation: str, cwd: str) -> GitBranchSnapshot:
            return GitBranchSnapshot(cwd=cwd, branch=f"{operation}-branch")

        screen = _queue_screen(run_operation, current_cwd="/real/repo", displayed_cwd="/repo/chrys")

        MainScreen._queue_git_branch_operation(screen, "configure", "/real/repo")
        await screen._git_branch_task

        assert screen.applied == []
        assert screen.sync_count == 0

    asyncio.run(run())


def test_git_branch_queue_retries_when_displayed_cwd_catches_up() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []

        async def run_operation(operation: str, cwd: str) -> GitBranchSnapshot:
            calls.append((operation, cwd))
            return GitBranchSnapshot(cwd=cwd, branch="restored-branch")

        screen = _queue_screen(run_operation, current_cwd="/restored/repo", displayed_cwd="/old/repo")
        cleared: list[str] = []
        screen._set_workspace_git_branch = cleared.append

        MainScreen._queue_git_branch_operation(screen, "configure", "/restored/repo")
        await screen._git_branch_task

        assert calls == [("configure", "/restored/repo")]
        assert screen.applied == []
        assert screen._git_branch_retry_cwd_on_display_sync == "/restored/repo"

        screen.chat_workspace_cwd = "/restored/repo"
        MainScreen.watch_chat_workspace_cwd(screen, "/old/repo", "/restored/repo")
        await screen._git_branch_task

        assert calls == [("configure", "/restored/repo"), ("configure", "/restored/repo")]
        assert cleared == [""]
        assert screen.applied == ["restored-branch"]
        assert screen.sync_count == 1
        assert screen._git_branch_retry_cwd_on_display_sync is None

    asyncio.run(run())


def test_git_branch_queue_does_not_duplicate_configure_when_displayed_cwd_catches_up_before_result() -> None:
    async def run() -> None:
        calls: list[tuple[str, str]] = []
        operation_started = asyncio.Event()
        release_operation = asyncio.Event()

        async def run_operation(operation: str, cwd: str) -> GitBranchSnapshot:
            calls.append((operation, cwd))
            operation_started.set()
            await release_operation.wait()
            return GitBranchSnapshot(cwd=cwd, branch="restored-branch")

        screen = _queue_screen(run_operation, current_cwd="/restored/repo", displayed_cwd="/old/repo")
        cleared: list[str] = []
        screen._set_workspace_git_branch = cleared.append

        MainScreen._queue_git_branch_operation(screen, "configure", "/restored/repo")
        await operation_started.wait()

        screen.chat_workspace_cwd = "/restored/repo"
        MainScreen.watch_chat_workspace_cwd(screen, "/old/repo", "/restored/repo")
        release_operation.set()
        await screen._git_branch_task

        assert calls == [("configure", "/restored/repo")]
        assert cleared == [""]
        assert screen.applied == ["restored-branch"]
        assert screen.sync_count == 1
        assert screen._git_branch_retry_cwd_on_display_sync is None

    asyncio.run(run())


def test_chat_workspace_cwd_change_does_not_reconfigure_for_unmatched_displayed_cwd() -> None:
    queued: list[str] = []
    cleared: list[str] = []
    screen = SimpleNamespace(
        _git_branch_monitor=SimpleNamespace(active=True),
        _git_branch_closed=False,
        _git_branch_retry_cwd_on_display_sync="/repo/chrys",
        _workspace_cwd=lambda: "/real/repo",
        _set_workspace_git_branch=cleared.append,
        _queue_git_branch_configure=queued.append,
    )

    MainScreen.watch_chat_workspace_cwd(screen, "/real/repo", "/repo/chrys")

    assert cleared == [""]
    assert queued == []


def test_chat_workspace_cwd_change_reconfigures_when_monitor_not_yet_active() -> None:
    queued: list[str] = []
    cleared: list[str] = []
    screen = SimpleNamespace(
        _git_branch_monitor=SimpleNamespace(active=False),
        _git_branch_closed=False,
        _git_branch_retry_cwd_on_display_sync="/restored/repo",
        _workspace_cwd=lambda: "/restored/repo",
        _set_workspace_git_branch=cleared.append,
        _queue_git_branch_configure=queued.append,
    )

    MainScreen.watch_chat_workspace_cwd(screen, "/old/repo", "/restored/repo")

    assert cleared == [""]
    assert queued == ["/restored/repo"]


def test_git_branch_queue_drops_result_after_screen_is_closed() -> None:
    async def run() -> None:
        operation_started = asyncio.Event()
        release_operation = asyncio.Event()

        async def run_operation(_operation: str, cwd: str) -> GitBranchSnapshot:
            operation_started.set()
            await release_operation.wait()
            return GitBranchSnapshot(cwd=cwd, branch="late-branch")

        screen = _queue_screen(run_operation, current_cwd="/repo")

        MainScreen._queue_git_branch_operation(screen, "configure", "/repo")
        await operation_started.wait()
        screen._git_branch_closed = True
        release_operation.set()
        await screen._git_branch_task

        assert screen.applied == []
        assert screen.sync_count == 0
        assert screen._git_branch_task is None

    asyncio.run(run())


def test_stop_git_branch_monitor_waits_for_in_flight_task_before_stopping_monitor() -> None:
    async def run() -> None:
        order: list[str] = []
        task_started = asyncio.Event()
        release_task = asyncio.Event()

        async def in_flight() -> None:
            order.append("task-start")
            task_started.set()
            await release_task.wait()
            order.append("task-end")

        class _Monitor:
            def stop(self) -> None:
                order.append("stop")

        task = asyncio.create_task(in_flight())
        await task_started.wait()
        screen = SimpleNamespace(
            _git_branch_closed=False,
            _git_branch_pending_operation=("refresh", ""),
            _git_branch_task=task,
            _git_branch_monitor=_Monitor(),
        )

        stop_task = asyncio.create_task(MainScreen._stop_git_branch_monitor(screen))
        await asyncio.sleep(0)

        assert order == ["task-start"]
        assert screen._git_branch_closed is True
        assert screen._git_branch_pending_operation is None

        release_task.set()
        await stop_task

        assert order == ["task-start", "task-end", "stop"]
        assert screen._git_branch_task is None

    asyncio.run(run())
