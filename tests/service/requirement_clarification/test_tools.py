# Copyright (c) 2026 Chrys. All rights reserved.

"""Frozen-snapshot tool confinement tests."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from chrys.service.requirement_clarification.tools import SnapshotReadTools


@pytest.mark.asyncio
async def test_snapshot_read_tools_reject_absolute_path_outside_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    outside = tmp_path / "live.py"
    outside.write_text("LIVE_SECRET\n", encoding="utf-8")
    tools = SnapshotReadTools(
        SimpleNamespace(cwd=str(snapshot)),
        roots=(snapshot,),
        reference_files=(),
    )

    result = await tools.read_file.invoke(arguments={"path": str(outside)})

    assert result[0].text.startswith("Error: ")
    assert "outside the frozen" in result[0].text
    assert "LIVE_SECRET" not in result[0].text


@pytest.mark.asyncio
async def test_snapshot_read_tools_reject_symlink_escaping_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("TOP_SECRET_SENTINEL\n", encoding="utf-8")
    link = snapshot / "link.py"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")
    tools = SnapshotReadTools(
        SimpleNamespace(cwd=str(snapshot)),
        roots=(snapshot,),
        reference_files=(),
    )

    result = await tools.read_file.invoke(arguments={"path": os.fspath(link)})

    assert result[0].text.startswith("Error: ")
    assert "TOP_SECRET_SENTINEL" not in result[0].text


@pytest.mark.asyncio
async def test_snapshot_read_tools_allow_file_inside_snapshot(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    source = snapshot / "safe.py"
    source.write_text("SAFE_CONTENT\n", encoding="utf-8")
    tools = SnapshotReadTools(
        SimpleNamespace(cwd=str(snapshot)),
        roots=(snapshot,),
        reference_files=(),
    )

    result = await tools.read_file.invoke(arguments={"path": "safe.py"})

    assert "SAFE_CONTENT" in result[0].text
