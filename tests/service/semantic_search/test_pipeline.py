# Copyright (c) 2026 Chrys. All rights reserved.

"""Production-boundary tests for semantic localization."""

from __future__ import annotations

import json
from pathlib import Path

from chrys.service.semantic_search import SemanticSearchConfig, SemanticSearchMode, localize_requirement


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


def test_a_corrupt_cached_payload_regenerates_instead_of_aborting(tmp_path: Path) -> None:
    """A truncated artifact must not poison every later run in the repository.

    ``json.JSONDecodeError`` is not a ``SemanticSearchError``, so callers that
    degrade gracefully would not have caught it: the run would abort, and keep
    aborting, until the file was deleted by hand.
    """
    (tmp_path / "parser.py").write_text("def parse_value(value):\n    return value\n", encoding="utf-8")
    config = SemanticSearchConfig(mode=SemanticSearchMode.FALLBACK, timeout_seconds=10)
    first = localize_requirement(tmp_path, "Fix parse_value", config=config)
    first.artifacts.result_json.write_text('{"locations": [', encoding="utf-8")

    second = localize_requirement(tmp_path, "Fix parse_value", config=config)

    assert second.reused is False
    assert second.locations
    assert json.loads(second.artifacts.result_json.read_text(encoding="utf-8"))


def test_repo_fingerprint_ignores_dependency_and_build_trees(tmp_path: Path) -> None:
    """Dependency and build trees are not source, and stat-walking them is the cost.

    The fingerprint runs twice per localization, so a populated ``node_modules``
    would add a six-figure stat walk to every turn that enables localization.
    """
    from chrys.service.semantic_search.output import repo_fingerprint

    (tmp_path / "parser.py").write_text("VALUE = 1\n", encoding="utf-8")
    generated = []
    for name in ("node_modules", "target", "dist"):
        directory = tmp_path / name
        directory.mkdir()
        artifact = directory / "artifact.js"
        artifact.write_text("generated", encoding="utf-8")
        generated.append(artifact)
    before = repo_fingerprint(tmp_path)

    for artifact in generated:
        artifact.write_text("regenerated with different length", encoding="utf-8")

    assert repo_fingerprint(tmp_path) == before

    (tmp_path / "parser.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert repo_fingerprint(tmp_path) != before


def test_the_skill_scripts_resolve_through_the_package_not_the_checkout() -> None:
    """An installed wheel has no repository layout to walk up into.

    The previous ``parents[4]`` walk only found the scripts from a source
    checkout, so every localization on an installed Chrys failed with
    "semantic-search script is missing".
    """
    from chrys.service import semantic_search
    from chrys.service.semantic_search.pipeline import _SKILL_SCRIPT_DIR

    package_root = Path(semantic_search.__file__).parent

    assert _SKILL_SCRIPT_DIR.is_relative_to(package_root)
    assert (_SKILL_SCRIPT_DIR / "build_index.py").is_file()
    assert (_SKILL_SCRIPT_DIR / "localize_task.py").is_file()
    assert (package_root / "skill" / "schemas" / "code-localization.schema.json").is_file()
