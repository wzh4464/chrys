# Copyright (c) 2026 Chrys. All rights reserved.

"""Owner-only private artifacts for requirement-clarification turns."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.foundation.trajectory.keys import ensure_owner_only_directory
from chrys.service.requirement_clarification.types import ClarificationResult

REQUIREMENT_CLARIFICATION_ARTIFACT_DIR = "requirement_clarification"
_TURN_DIR = re.compile(r"turn_(\d+)\Z")


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

    def save_history_checkpoint(self, payload: dict[str, object]) -> None:
        """Persist H0 privately for phase recovery without model exposure."""
        atomic_write_owner_only_text(
            self.root / "h0.private.json",
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def save_workflow_record(self, payload: dict[str, object]) -> None:
        """Atomically replace the durable phase/revision recovery record."""
        atomic_write_owner_only_text(
            self.root / "workflow.json",
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def save_summary(self, payload: dict[str, object]) -> None:
        atomic_write_owner_only_text(
            self.root / "summary.json",
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )


def prune_workflow_artifacts_after_turn(session_dir: Path, target_turn: int) -> None:
    """Remove workflow artifacts belonging to turns discarded by rollback."""
    root = session_dir / REQUIREMENT_CLARIFICATION_ARTIFACT_DIR
    if not root.is_dir() or root.is_symlink():
        return
    for path in root.iterdir():
        match = _TURN_DIR.fullmatch(path.name)
        if match is None or int(match.group(1)) <= target_turn or path.is_symlink():
            continue
        if path.is_dir():
            shutil.rmtree(path)
