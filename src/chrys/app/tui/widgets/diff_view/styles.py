# Copyright (c) 2026 Chrys. All rights reserved.

"""Shared style helpers for diff view rendering."""

from __future__ import annotations

_DARK_256_LINE_STYLES = {
    "on #243f30": "on #005F00",
    "on #512b35": "on #5F0000",
}


def line_style_for_color_system(line_style: str, color_system: str | None) -> str:
    """Map dark diff row backgrounds to exact xterm colors in 256-color terminals.

    The light-theme backgrounds are intentionally left to Rich's normal
    downgrade path; the SSH/transparent-terminal issue this handles is specific
    to the dark chrys palettes.
    """
    if color_system != "256":
        return line_style
    return _DARK_256_LINE_STYLES.get(line_style.lower(), line_style)
