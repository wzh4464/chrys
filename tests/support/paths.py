# Copyright (c) 2026 Chrys. All rights reserved.

"""Stable path helpers for tests."""

from __future__ import annotations

from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"


def fixture_path(*parts: str) -> Path:
    """Return a path under the test fixture tree."""

    return TESTS_ROOT / "fixtures" / Path(*parts)
