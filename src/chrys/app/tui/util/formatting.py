# Copyright (c) 2026 Chrys. All rights reserved.

"""Shared formatting helpers for TUI display text."""

from __future__ import annotations


def format_token_count(n: int) -> str:
    """Format a token count with compact human-friendly units."""
    if n >= 999_950_000:
        return f"{n / 1_000_000_000:.1f}b"
    if n >= 999_950:
        return f"{n / 1_000_000:.1f}m"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def format_byte_size(size_bytes: int) -> str:
    """Format bytes as a human-readable string (e.g. '1.2 MB')."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    kib = size_bytes / 1024
    if kib < 1024:
        return f"{kib:.1f} KB"
    mib = kib / 1024
    if mib < 1024:
        return f"{mib:.1f} MB"
    gib = mib / 1024
    return f"{gib:.1f} GB"
