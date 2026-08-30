# Copyright (c) 2026 Chrys. All rights reserved.

"""Shared fixtures for engine tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _workspace_change_notice_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the workspace change notice out of engine tests by default.

    These tests build real engines whose workspace is the developer
    checkout; the default-enabled notice would capture and diff it with
    Git subprocesses on every turn — pure overhead that pushes slow CI
    hosts past fixed polling timeouts. Notice tests opt back in with an
    explicit ``Settings(workspace_change_notice=True)``.
    """
    monkeypatch.setenv("CHRYS_WORKSPACE_CHANGE_NOTICE", "0")
