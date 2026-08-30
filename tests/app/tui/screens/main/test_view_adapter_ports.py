# Copyright (c) 2026 Chrys. All rights reserved.

"""Port conformance checks for the main-screen Textual adapter."""

from __future__ import annotations

import inspect
from typing import Protocol

import pytest

from chrys.app.tui.screens.main import ports
from chrys.app.tui.screens.main.view_adapter import MainScreenViewAdapter


def _protocol_member_names(protocol: type[Protocol]) -> set[str]:
    names: set[str] = set()
    for base in reversed(protocol.__mro__):
        if base in {Protocol, object}:
            continue
        for name, value in base.__dict__.items():
            if name.startswith("_"):
                continue
            if inspect.isfunction(value) or isinstance(value, property):
                names.add(name)
    return names


@pytest.mark.parametrize(
    "protocol",
    [
        ports.InputFlowView,
        ports.ShellModeView,
        ports.InputFocusView,
        ports.SuggestionPopupView,
        ports.BuddyCommandView,
        ports.SessionLifecycleView,
        ports.RuntimeConfigView,
        ports.WorkspaceView,
        ports.CopyActionView,
        ports.DiffView,
        ports.RollbackView,
        ports.NavigationView,
        ports.ToolActionView,
        ports.DialogGatewayView,
        ports.MainScreenPresentationView,
        ports.BackendEventView,
    ],
)
def test_main_screen_view_adapter_implements_view_ports(protocol: type[Protocol]) -> None:
    missing = sorted(name for name in _protocol_member_names(protocol) if not hasattr(MainScreenViewAdapter, name))

    assert missing == []
