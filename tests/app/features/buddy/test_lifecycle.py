# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for Buddy application-level lifecycle callbacks."""

from __future__ import annotations

import logging

import pytest

from chrys.app.features.buddy import lifecycle


def test_successful_turn_callback_increments_when_hatched(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def _increment_message_count() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr("chrys.app.features.buddy.config.has_hatched", lambda: True)
    monkeypatch.setattr("chrys.app.features.buddy.config.increment_message_count", _increment_message_count)

    lifecycle.on_successful_turn()

    assert calls == 1


def test_successful_turn_callback_skips_when_unhatched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chrys.app.features.buddy.config.has_hatched", lambda: False)
    monkeypatch.setattr(
        "chrys.app.features.buddy.config.increment_message_count",
        lambda: pytest.fail("unhatched buddy should not increment"),
    )

    lifecycle.on_successful_turn()


def test_successful_turn_callback_swallows_buddy_failures(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="chrys.app.features.buddy.lifecycle")
    monkeypatch.setattr("chrys.app.features.buddy.config.has_hatched", lambda: True)
    monkeypatch.setattr(
        "chrys.app.features.buddy.config.increment_message_count",
        lambda: (_ for _ in ()).throw(PermissionError("locked")),
    )

    lifecycle.on_successful_turn()

    assert "Failed to persist buddy message count" in caplog.text
