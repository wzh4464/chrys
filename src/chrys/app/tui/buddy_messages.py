# Copyright (c) 2026 Chrys. All rights reserved.

"""Shared semantic messages for Buddy toast presentation."""

from chrys.foundation.i18n import msg

BUDDY_TITLE = msg("tui.buddy.title", fallback="Buddy")
BUDDY_THINKING = msg("tui.buddy.thinking", fallback="💭 {name} is thinking...")
BUDDY_HUMS = msg("tui.buddy.hums", fallback="💭 {name} hums thoughtfully...")
BUDDY_PONDERS = msg("tui.buddy.ponders", fallback="💭 {name} ponders...")
BUDDY_THINKING_SOUNDS = msg(
    "tui.buddy.thinking_sounds",
    fallback="💭 *{name} makes thinking sounds*...",
)

__all__ = [
    "BUDDY_HUMS",
    "BUDDY_PONDERS",
    "BUDDY_THINKING",
    "BUDDY_THINKING_SOUNDS",
    "BUDDY_TITLE",
]
