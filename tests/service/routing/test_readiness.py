# Copyright (c) 2026 Chrys. All rights reserved.

"""Workspace readiness: the PACT veto gate, not a scoring dimension."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from chrys.service.routing.readiness import (
    WorkspaceReadiness,
    probe_git_dirty,
    probe_workspace_readiness,
    workspace_fingerprint,
)


def _probe(cwd: Path, *, verify_command: str = "uv run pytest", tool: bool = True) -> WorkspaceReadiness:
    return probe_workspace_readiness(str(cwd), verify_command=verify_command, pact_tool_available=tool)


def test_pact_is_ready_only_with_both_a_verify_command_and_the_tool(tmp_path: Path) -> None:
    assert _probe(tmp_path).pact_ready is True
    assert _probe(tmp_path, verify_command="").pact_ready is False
    assert _probe(tmp_path, verify_command="   ").pact_ready is False
    assert _probe(tmp_path, tool=False).pact_ready is False


def test_tests_are_detected_by_the_usual_layouts(tmp_path: Path) -> None:
    assert _probe(tmp_path).has_tests is False

    (tmp_path / "tests").mkdir()
    assert _probe(tmp_path).has_tests is True


def test_a_test_suffixed_file_also_counts(tmp_path: Path) -> None:
    (tmp_path / "parser_test.go").write_text("package main\n", encoding="utf-8")

    assert _probe(tmp_path).has_tests is True


def test_git_dirtiness_is_none_outside_a_repository(tmp_path: Path) -> None:
    assert probe_git_dirty(str(tmp_path)) is None


def test_git_dirtiness_is_reported_inside_a_repository(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, stdin=subprocess.DEVNULL)

    assert probe_git_dirty(str(tmp_path)) is False

    (tmp_path / "new.py").write_text("x = 1\n", encoding="utf-8")

    assert probe_git_dirty(str(tmp_path)) is True


def test_the_per_turn_probe_spawns_no_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """It runs inside message admission; a `git status` there delays every turn."""

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("probe_workspace_readiness must not spawn a subprocess")

    monkeypatch.setattr(subprocess, "run", explode)

    assert _probe(tmp_path).has_tests is False


def test_a_missing_directory_probes_as_unready(tmp_path: Path) -> None:
    assert _probe(tmp_path / "absent").has_tests is False


# --------------------------------------------------------------------------
# fingerprint
# --------------------------------------------------------------------------


def test_fingerprint_changes_when_a_top_level_directory_appears(tmp_path: Path) -> None:
    before = workspace_fingerprint(str(tmp_path))
    (tmp_path / "services").mkdir()

    assert workspace_fingerprint(str(tmp_path)) != before


def test_fingerprint_is_stable_when_only_file_contents_change(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'a'\n", encoding="utf-8")
    before = workspace_fingerprint(str(tmp_path))

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'b'\nversion = '2'\n", encoding="utf-8")

    assert workspace_fingerprint(str(tmp_path)) == before


def test_fingerprint_changes_when_the_manifest_set_changes(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    before = workspace_fingerprint(str(tmp_path))

    (tmp_path / "package.json").write_text("{}", encoding="utf-8")

    assert workspace_fingerprint(str(tmp_path)) != before


def test_fingerprint_of_a_missing_directory_is_stable_and_not_a_crash(tmp_path: Path) -> None:
    absent = str(tmp_path / "absent")

    assert workspace_fingerprint(absent) == workspace_fingerprint(absent)
