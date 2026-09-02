# Copyright (c) 2026 Chrys. All rights reserved.

"""Typed views over semantic-search artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LocalizationArtifact:
    """Paths produced by one localization preflight."""

    result_json: Path
    report_markdown: Path
    index_json: Path
    graph_json: Path
    trace_jsonl: Path
    manifest_json: Path
    codegraph_json: Path | None = None


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """Localization payload plus its artifact paths."""

    payload: dict[str, Any]
    artifacts: LocalizationArtifact
    reused: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def locations(self) -> list[dict[str, Any]]:
        """Return normalized candidate locations."""
        value = self.payload.get("locations", [])
        return value if isinstance(value, list) else []
