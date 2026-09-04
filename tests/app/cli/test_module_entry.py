# Copyright (c) 2026 Chrys. All rights reserved.

"""`python -m chrys` must be a real entry point: sub-agents launch it that way."""

from __future__ import annotations

import subprocess
import sys


def test_python_dash_m_chrys_runs_the_cli() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "chrys", "--version"],
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert "No module named" not in completed.stderr
