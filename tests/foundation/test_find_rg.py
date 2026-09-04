# Copyright (c) 2026 Chrys. All rights reserved.

"""``find_rg`` must skip a bundled binary that does not run on this machine."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

from chrys.foundation import vendor


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    vendor._runs.cache_clear()
    yield
    vendor._runs.cache_clear()


def _bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, body: str) -> Path:
    ripgrep = tmp_path / "ripgrep"
    ripgrep.mkdir()
    binary = ripgrep / "rg"
    binary.write_text(body, encoding="utf-8")
    binary.chmod(0o755)
    monkeypatch.setattr(vendor, "VENDOR_RIPGREP", ripgrep)
    # Only the generic name is present, whatever the platform.
    monkeypatch.setattr(vendor, "_TRIPLE_MAP", {(vendor._platform.system(), vendor._platform.machine()): ("rg",)})
    return binary


def test_a_foreign_bundled_binary_falls_through_to_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A Mach-O rg copied into a Linux checkout is not a script and not an ELF:
    # exec fails before the program starts, exactly like the binary below.
    _bundle(monkeypatch, tmp_path, body="\x00\x01 not a program\n")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rg" if name == "rg" else None)

    assert vendor.find_rg() == "/usr/bin/rg"


@pytest.mark.skipif(sys.platform == "win32", reason="the probe binary is a POSIX shell script")
def test_a_working_bundled_binary_is_preferred(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    binary = _bundle(monkeypatch, tmp_path, body="#!/bin/sh\necho 'ripgrep 0.0.0-test'\n")
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/rg" if name == "rg" else None)

    assert vendor.find_rg() == os.fspath(binary)


def test_nothing_usable_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _bundle(monkeypatch, tmp_path, body="\x00\x01 not a program\n")
    monkeypatch.setattr(shutil, "which", lambda name: None)

    assert vendor.find_rg() is None
