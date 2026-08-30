# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the stateful /buddy command controller."""

from __future__ import annotations

import asyncio

import pytest

from chrys.app.tui.screens.main.buddy_command import BuddyCommandController
from chrys.foundation.i18n import MessageRef
from chrys.foundation.i18n.formatting import format_message


class _Companion:
    name = "Biscuit"


class _NotificationHandle:
    def __init__(self, dismissed: list[bool]) -> None:
        self._dismissed = dismissed

    def dismiss(self) -> None:
        self._dismissed.append(True)


class _BuddyView:
    def __init__(self) -> None:
        self.notifications: list[tuple[str, str, float | None]] = []
        self.dismissed: list[bool] = []
        self.refreshes: list[bool] = []

    def notify_buddy(
        self,
        message: MessageRef | str,
        *,
        severity: str = "information",
        timeout: float | None = 10,
    ) -> object | None:
        self.notifications.append((message if isinstance(message, str) else format_message(message), severity, timeout))
        return _NotificationHandle(self.dismissed)

    def dismiss_notification(self, handle: object | None) -> None:
        if isinstance(handle, _NotificationHandle):
            handle.dismiss()

    def refresh_buddy_panel(self, *, focus_tab: bool) -> None:
        self.refreshes.append(focus_tab)

    def call_after_refresh(self, callback) -> None:
        callback()


def test_buddy_pet_without_companion_does_not_start_response(monkeypatch: pytest.MonkeyPatch) -> None:
    view = _BuddyView()
    controller = BuddyCommandController(view)
    began: list[bool] = []

    monkeypatch.setattr("chrys.app.features.buddy.config.is_muted", lambda: False)
    monkeypatch.setattr("chrys.app.features.buddy.notification.get_companion", lambda: None)
    monkeypatch.setattr("chrys.app.features.buddy.notification.try_begin_pet_response", lambda: began.append(True))

    controller.handle("pet")

    assert began == []
    assert controller.pet_task is None
    assert view.notifications == [("You don't have a buddy to pet yet. Try /buddy hatch first!", "warning", 10)]


def test_buddy_pet_muted_companion_does_not_start_response(monkeypatch: pytest.MonkeyPatch) -> None:
    view = _BuddyView()
    controller = BuddyCommandController(view)
    began: list[bool] = []

    monkeypatch.setattr("chrys.app.features.buddy.config.is_muted", lambda: True)
    monkeypatch.setattr("chrys.app.features.buddy.notification.get_companion", _Companion)
    monkeypatch.setattr("chrys.app.features.buddy.notification.try_begin_pet_response", lambda: began.append(True))

    controller.handle("pet")

    assert began == []
    assert controller.pet_task is None
    assert view.notifications == [("Biscuit is nearby but notifications are muted.", "warning", 10)]


@pytest.mark.asyncio
async def test_buddy_pet_generation_error_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    view = _BuddyView()
    controller = BuddyCommandController(view)
    finished: list[bool] = []

    async def generate_response(_companion, _history, notify_callback=None) -> str:
        _ = notify_callback
        raise RuntimeError("generation failed")

    monkeypatch.setattr("chrys.app.features.buddy.config.is_muted", lambda: False)
    monkeypatch.setattr("chrys.app.features.buddy.notification.get_companion", _Companion)
    monkeypatch.setattr("chrys.app.features.buddy.notification.try_begin_pet_response", lambda: True)
    monkeypatch.setattr("chrys.app.features.buddy.notification.finish_pet_response", lambda: finished.append(True))
    monkeypatch.setattr("chrys.app.features.buddy.notification.get_recent_conversation", lambda max_messages=10: [])
    monkeypatch.setattr("chrys.app.features.buddy.notification.generate_ai_pet_response_async", generate_response)

    controller.handle("pet")
    task = controller.pet_task

    assert task is not None
    with pytest.raises(RuntimeError, match="generation failed"):
        await task

    assert controller.pet_task is None
    assert finished == [True]
    assert view.dismissed == [True]


@pytest.mark.asyncio
async def test_buddy_shutdown_cancels_in_flight_pet(monkeypatch: pytest.MonkeyPatch) -> None:
    view = _BuddyView()
    controller = BuddyCommandController(view)
    started = asyncio.Event()
    finished: list[bool] = []

    async def generate_response(_companion, _history, notify_callback=None) -> str:
        _ = notify_callback
        started.set()
        await asyncio.Future()
        return "unreachable"

    monkeypatch.setattr("chrys.app.features.buddy.config.is_muted", lambda: False)
    monkeypatch.setattr("chrys.app.features.buddy.notification.get_companion", _Companion)
    monkeypatch.setattr("chrys.app.features.buddy.notification.try_begin_pet_response", lambda: True)
    monkeypatch.setattr("chrys.app.features.buddy.notification.finish_pet_response", lambda: finished.append(True))
    monkeypatch.setattr("chrys.app.features.buddy.notification.get_recent_conversation", lambda max_messages=10: [])
    monkeypatch.setattr("chrys.app.features.buddy.notification.generate_ai_pet_response_async", generate_response)

    controller.handle("pet")
    task = controller.pet_task
    assert task is not None
    await started.wait()

    await controller.shutdown()

    assert task.cancelled()
    assert controller.pet_task is None
    assert view.dismissed == [True]
    assert finished


@pytest.mark.asyncio
async def test_buddy_shutdown_does_not_finish_peer_owned_pet_response(monkeypatch: pytest.MonkeyPatch) -> None:
    view = _BuddyView()
    controller = BuddyCommandController(view)
    finished: list[bool] = []

    monkeypatch.setattr("chrys.app.features.buddy.notification.finish_pet_response", lambda: finished.append(True))

    await controller.shutdown()

    assert finished == []
