# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the standalone semantic-localization command."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from chrys.app.cli import locate


def test_locate_bootstraps_and_emits_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    report = tmp_path / "report.md"
    report.write_text("# Report\n", encoding="utf-8")
    bootstrap_calls: list[tuple[bool, bool, Path]] = []

    def fake_bootstrap_runtime(
        *,
        dotenv_override: bool,
        configure_stdio: bool,
        project_root: Path,
    ) -> None:
        bootstrap_calls.append((dotenv_override, configure_stdio, project_root))

    def fake_localize_requirement(
        repo_value: Path,
        requirement: str,
        *,
        artifact_dir: str | None,
        config,
        refresh: bool,
        codegraph_command: str,
    ):
        assert repo_value == repo
        assert requirement == "fix parser"
        assert artifact_dir is None
        assert config.mode.value == "auto"
        assert refresh is False
        assert codegraph_command == ""
        return SimpleNamespace(
            payload={"locations": [{"file": "src/parser.py"}]},
            artifacts=SimpleNamespace(report_markdown=report),
            reused=False,
        )

    monkeypatch.setattr(locate, "bootstrap_runtime", fake_bootstrap_runtime)
    monkeypatch.setattr(locate, "localize_requirement", fake_localize_requirement)

    assert locate.main(["fix parser", "--repo", str(repo), "--json"]) == 0
    assert bootstrap_calls == [(True, True, repo)]
    assert '"file": "src/parser.py"' in capsys.readouterr().out


def test_locate_rejects_disabled_mode() -> None:
    with pytest.raises(SystemExit):
        locate.main(["find code", "--mode", "off"])


def test_locate_allows_dedicated_artifact_directory_outside_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside"
    monkeypatch.setattr(locate, "bootstrap_runtime", lambda **_kwargs: None)

    def fake_localize_requirement(repo_value: Path, requirement: str, *, artifact_dir: Path, **_kwargs):
        assert repo_value == repo
        assert requirement == "find code"
        assert artifact_dir == outside
        return SimpleNamespace(
            payload={},
            artifacts=SimpleNamespace(report_markdown=outside / "code-localization.md"),
            reused=False,
        )

    monkeypatch.setattr(locate, "localize_requirement", fake_localize_requirement)

    assert locate.main(["find code", "--repo", str(repo), "--artifact-dir", str(outside)]) == 0
