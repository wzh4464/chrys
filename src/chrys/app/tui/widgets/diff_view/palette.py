# Copyright (c) 2026 Chrys. All rights reserved.

"""Shared dark/light palette constants for diff widgets."""

from __future__ import annotations

DARK_NUMBER_STYLES: dict[str, str] = {
    "+": "$text-success on #1f4d2a",
    "-": "$text-error on #5a2628",
    " ": "$foreground 30%",
}
DARK_LINE_STYLES: dict[str, str] = {
    "+": "on #243f30",
    "-": "on #512b35",
    " ": "",
    "/": "",
}
DARK_EDGE_STYLES: dict[str, str] = {
    "+": "$success on #1f4d2a",
    "-": "$error on #5a2628",
    " ": "$foreground 15%",
}
LIGHT_NUMBER_STYLES: dict[str, str] = {
    "+": "#1f7a3a on #D4ECD9",
    "-": "#8f1f25 on #F0D4D8",
    " ": "$foreground 30%",
}
LIGHT_LINE_STYLES: dict[str, str] = {
    "+": "on #d8f0dc",
    "-": "on #F8E8EA",
    " ": "",
    "/": "",
}
LIGHT_EDGE_STYLES: dict[str, str] = {
    "+": "#2E8B57 on #D4ECD9",
    "-": "#C75261 on #F0D4D8",
    " ": "$foreground 15%",
}
