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
