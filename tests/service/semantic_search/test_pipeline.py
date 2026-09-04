# Copyright (c) 2026 Chrys. All rights reserved.

"""Production-boundary tests for semantic localization."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import chrys.service.semantic_search.pipeline as pipeline_module
from chrys.foundation.platform import get_platform
from chrys.service.semantic_search import (
    SemanticSearchConfig,
    SemanticSearchError,
    SemanticSearchMode,
    localize_requirement,
    localize_requirement_async,
)


def test_fallback_localization_writes_report_and_reuses_cache(tmp_path: Path) -> None:
    (tmp_path / "parser.py").write_text("def parse_value(value):\n    return value\n", encoding="utf-8")
    config = SemanticSearchConfig(mode=SemanticSearchMode.FALLBACK, timeout_seconds=10)
    first = localize_requirement(tmp_path, "Fix parse_value", config=config)
    assert first.locations
    assert first.artifacts.report_markdown.is_file()
    second = localize_requirement(tmp_path, "Fix parse_value", config=config)
    assert second.reused
    manifest = json.loads(first.artifacts.manifest_json.read_text(encoding="utf-8"))
    assert manifest["format"] == "semantic-search-manifest"
    header = first.artifacts.report_markdown.read_text(encoding="utf-8").split("## Ranked Locations", 1)[0]
    assert str(tmp_path) not in header
    if not get_platform().is_windows:
        artifact_dir = first.artifacts.manifest_json.parent
        assert artifact_dir.stat().st_mode & 0o777 == 0o700
        assert (artifact_dir / "PROMPT.md").stat().st_mode & 0o777 == 0o600
        assert first.artifacts.manifest_json.stat().st_mode & 0o777 == 0o600


def test_cache_is_invalidated_when_output_configuration_changes(tmp_path: Path) -> None:
    (tmp_path / "parser.py").write_text("def parse_value(value):\n    return value\n", encoding="utf-8")
    first = localize_requirement(
        tmp_path,
        "Fix parse_value",
        config=SemanticSearchConfig(mode=SemanticSearchMode.FALLBACK, top_locations=12),
    )
    second = localize_requirement(
        tmp_path,
        "Fix parse_value",
        config=SemanticSearchConfig(mode=SemanticSearchMode.FALLBACK, top_locations=1),
    )

    assert first.reused is False
    assert second.reused is False


def test_cache_is_invalidated_by_same_size_same_mtime_content_change(tmp_path: Path) -> None:
    source = tmp_path / "parser.py"
    source.write_text("def old_name():\n    return 1\n", encoding="utf-8")
    config = SemanticSearchConfig(mode=SemanticSearchMode.FALLBACK)
    localize_requirement(tmp_path, "Find implementation", config=config)
    original_stat = source.stat()
    source.write_text("def new_name():\n    return 2\n", encoding="utf-8")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    result = localize_requirement(tmp_path, "Find implementation", config=config)

    assert result.reused is False


def test_localization_does_not_index_symlink_target_outside_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "secret.py"
    outside.write_text("TOP_SECRET_SENTINEL\n", encoding="utf-8")
    try:
        (repo / "link.py").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    result = localize_requirement(
        repo,
        "Find the secret",
        config=SemanticSearchConfig(mode=SemanticSearchMode.FALLBACK),
    )

    index = result.artifacts.index_json.read_text(encoding="utf-8")
    assert "link.py" not in index
    assert "TOP_SECRET_SENTINEL" not in index


def test_preplanted_artifact_symlink_does_not_overwrite_target(tmp_path: Path) -> None:
    (tmp_path / "parser.py").write_text("def parse_value(value):\n    return value\n", encoding="utf-8")
    victim = tmp_path / "victim.txt"
    victim.write_text("DO_NOT_OVERWRITE\n", encoding="utf-8")
    artifact_dir = tmp_path / ".semantic-search"
    artifact_dir.mkdir()
    try:
        (artifact_dir / "index.json").symlink_to(victim)
    except OSError:
        pytest.skip("symlinks are unavailable")

    result = localize_requirement(
        tmp_path,
        "Fix parse_value",
        config=SemanticSearchConfig(mode=SemanticSearchMode.FALLBACK),
    )

    assert victim.read_text(encoding="utf-8") == "DO_NOT_OVERWRITE\n"
    assert result.artifacts.index_json.is_file()
    assert not result.artifacts.index_json.is_symlink()


def test_custom_artifact_directory_rejects_unrelated_files(tmp_path: Path) -> None:
    (tmp_path / "owner.py").write_text("def owner():\n    pass\n", encoding="utf-8")
    artifact_dir = tmp_path / "source"
    artifact_dir.mkdir()
    (artifact_dir / "important.py").write_text("DO_NOT_OVERWRITE = True\n", encoding="utf-8")

    with pytest.raises(SemanticSearchError, match="contains unrelated files"):
        localize_requirement(
            tmp_path,
            "Find owner",
            artifact_dir=artifact_dir,
            config=SemanticSearchConfig(mode=SemanticSearchMode.FALLBACK),
        )

    assert (artifact_dir / "important.py").read_text(encoding="utf-8") == "DO_NOT_OVERWRITE = True\n"


def test_localization_supports_surrogateescaped_repository_paths(tmp_path: Path) -> None:
    if get_platform().is_windows:
        pytest.skip("surrogateescaped filesystem paths are POSIX-only")
    repo_bytes = os.fsencode(tmp_path) + b"/repo-\xff"
    os.mkdir(repo_bytes)
    source_bytes = repo_bytes + b"/source-\xfe.py"
    Path(os.fsdecode(source_bytes)).write_bytes(b"def locate_me():\n    return 1\n")
    repo = Path(os.fsdecode(repo_bytes))

    result = localize_requirement(
        repo,
        "Find locate_me under \udcff",
        config=SemanticSearchConfig(mode=SemanticSearchMode.FALLBACK),
    )

    assert result.locations
    assert result.artifacts.manifest_json.is_file()


@pytest.mark.asyncio
async def test_failed_codegraph_refresh_does_not_reuse_stale_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "owner.py").write_text("def owner():\n    pass\n", encoding="utf-8")
    artifact_dir = tmp_path / ".semantic-search"
    artifact_dir.mkdir()
    stale = artifact_dir / "codegraph-perception.json"
    stale.write_text('{"available": true, "stale": true}\n', encoding="utf-8")
    localization_arguments: list[str] = []

    async def _run_script(script: str, arguments: list[str], **_kwargs) -> None:
        if script == "build_index.py":
            Path(arguments[arguments.index("--out") + 1]).write_text('{"files": []}\n', encoding="utf-8")
            return
        if script == "codegraph_perception.py":
            raise SemanticSearchError("refresh failed")
        localization_arguments.extend(arguments)
        Path(arguments[arguments.index("--out") + 1]).write_text(
            '{"locations": [{"file_path": "owner.py"}]}\n',
            encoding="utf-8",
        )
        Path(arguments[arguments.index("--markdown") + 1]).write_text("# Result\n", encoding="utf-8")

    monkeypatch.setattr(pipeline_module, "_run_script", _run_script)
    await localize_requirement_async(
        tmp_path,
        "Find owner",
        config=SemanticSearchConfig(mode=SemanticSearchMode.FALLBACK),
        refresh=True,
    )

    assert "--codegraph-perception" not in localization_arguments
    assert not stale.exists()
