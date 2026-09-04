# Copyright (c) 2026 Chrys. All rights reserved.

"""Semantic-search subprocess ownership tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from chrys.foundation.platform import get_platform
from tests.support.waiting import wait_until


@pytest.mark.asyncio
async def test_bounded_codegraph_process_kills_grandchild_on_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if get_platform().is_windows:
        pytest.skip("POSIX process-group contract")
    scripts = Path(__file__).parents[3] / "src" / "chrys" / "service" / "semantic_search" / "skill" / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    from _common import run_process_bounded

    marker = tmp_path / "grandchild-survived"
    driver = tmp_path / "spawn_grandchild.py"
    driver.write_text(
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', "
        "\"import pathlib,sys,time; time.sleep(3.0); pathlib.Path(sys.argv[1]).write_text('alive')\", "
        "sys.argv[1]])\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )

    # The timeout lands while the grandchild is certainly alive (a loaded CI runner
    # takes a few hundred ms to start the driver and spawn it), well before it writes.
    returncode, _stdout, _stderr, timed_out = run_process_bounded(
        [sys.executable, str(driver), str(marker)],
        cwd=tmp_path,
        timeout=1.0,
        max_chars=1000,
    )

    assert timed_out is True
    assert returncode != 0
    assert not await wait_until(
        marker.exists,
        timeout=3.0,
    )
