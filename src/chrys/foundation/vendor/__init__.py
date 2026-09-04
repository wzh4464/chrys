# Copyright (c) 2026 Chrys. All rights reserved.

"""Vendored third-party binaries bundled at build time."""

from __future__ import annotations

import functools
import platform as _platform
import shutil
import subprocess
from pathlib import Path

VENDOR_RIPGREP = Path(__file__).resolve().parent / "ripgrep"

# Platform triple → rg binary filename inside foundation/vendor/ripgrep/.
# Multi-platform builds (--all) use suffixed names; local builds use plain "rg"/"rg.exe".
_TRIPLE_MAP: dict[tuple[str, str], tuple[str, str]] = {
    # (system, machine) → (triple-suffixed name, generic name)
    ("Darwin", "arm64"): ("rg-aarch64-apple-darwin", "rg"),
    ("Darwin", "x86_64"): ("rg-x86_64-apple-darwin", "rg"),
    ("Linux", "x86_64"): ("rg-x86_64-unknown-linux-musl", "rg"),
    ("Linux", "aarch64"): ("rg-aarch64-unknown-linux-gnu", "rg"),
    ("Windows", "AMD64"): ("rg-x86_64-pc-windows-msvc.exe", "rg.exe"),
}


def find_rg() -> str | None:
    """Locate the bundled ``rg`` binary, or return *None* if not found.

    Checks the platform-specific name first (CD multi-platform wheel), then
    the generic name (local single-platform build), then ``PATH``.
    """
    key = (_platform.system(), _platform.machine())
    names = _TRIPLE_MAP.get(key)
    if names:
        for name in names:
            candidate = VENDOR_RIPGREP / name
            if candidate.is_file() and _runs(str(candidate)):
                return str(candidate)
    # Fallback: system PATH
    which = shutil.which("rg")
    if which:
        return which
    return None


@functools.cache
def _runs(path: str) -> bool:
    """Whether this binary actually executes here.

    A source tree copied from another machine carries that machine's generic
    ``rg``; on Linux it fails with ``Exec format error`` on every search, and
    the working ``rg`` further down the list -- or on PATH -- is never tried.
    """
    try:
        completed = subprocess.run(  # noqa: S603
            [path, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return False
    return completed.returncode == 0
