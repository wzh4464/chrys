# Copyright (c) 2026 Chrys. All rights reserved.

"""Owner-only private artifacts for requirement-clarification turns."""

from __future__ import annotations

import json
from pathlib import Path

from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.foundation.trajectory.keys import ensure_owner_only_directory
from chrys.service.requirement_clarification.types import ClarificationResult

REQUIREMENT_CLARIFICATION_ARTIFACT_DIR = "requirement_clarification"


class ClarificationArtifactStore:
    """Persist private proposals separately from model-visible session history."""

    def __init__(self, session_dir: Path, turn_number: int) -> None:
        self.root = session_dir / REQUIREMENT_CLARIFICATION_ARTIFACT_DIR / f"turn_{turn_number}"
        ensure_owner_only_directory(self.root)

    def save_result(self, result: ClarificationResult) -> None:
        private = {
            "strategy_version": result.strategy_version,
            "revision": result.revision,
            "elapsed_seconds": result.elapsed_seconds,
            "delta": result.delta,
            "selection": result.selection.model_dump(mode="json"),
            "proposals": [proposal.model_dump(mode="json") for proposal in result.proposals],
            "usage_details": list(result.usage_details),
            "warnings": list(result.warnings),
        }
        atomic_write_owner_only_text(
            self.root / "clarification.private.json",
            json.dumps(private, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def save_initial_transcript(self, payload: dict[str, object]) -> None:
        atomic_write_owner_only_text(
            self.root / "initial_implementation.private.json",
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def save_summary(self, payload: dict[str, object]) -> None:
        atomic_write_owner_only_text(
            self.root / "summary.json",
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
