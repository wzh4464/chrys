# Copyright (c) 2026 Chrys. All rights reserved.

"""User-facing product branding."""

from __future__ import annotations

APP_DISPLAY_NAME = "iCode"


def format_app_version_title(version: str) -> str:
    """Return the user-facing application title with a version suffix."""
    return f"{APP_DISPLAY_NAME} v{version}"
