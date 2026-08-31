# Copyright (c) 2026 Chrys. All rights reserved.

"""Owner-only private artifacts for requirement-clarification turns."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.foundation.trajectory.keys import ensure_owner_only_directory
from chrys.service.requirement_clarification.types import ClarificationResult

REQUIREMENT_CLARIFICATION_ARTIFACT_DIR = "requirement_clarification"
_TURN_DIR = re.compile(r"turn_(\d+)\Z")
_WORKFLOW_RECORD_MAX_BYTES = 4 * 1024 * 1024
_PRIVATE_ARTIFACT_MAX_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class IncompleteWorkflowArtifacts:
    """Owner-scoped files needed to recover a completed P0 after a crash."""

    turn_number: int
    root: Path
    record: dict[str, object]


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


def latest_incomplete_workflow(session_dir: Path) -> IncompleteWorkflowArtifacts | None:
    """Return the newest bounded non-terminal workflow record, if any."""
    root = session_dir / REQUIREMENT_CLARIFICATION_ARTIFACT_DIR
    if not root.is_dir() or root.is_symlink():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in root.iterdir():
        match = _TURN_DIR.fullmatch(path.name)
        if match is not None and path.is_dir() and not path.is_symlink():
            candidates.append((int(match.group(1)), path))
    for turn_number, path in sorted(candidates, reverse=True):
        record_path = path / "workflow.json"
        if not record_path.is_file() or record_path.is_symlink():
            continue
        try:
            if record_path.stat().st_size > _WORKFLOW_RECORD_MAX_BYTES:
                continue
            value = json.loads(record_path.read_text(encoding="utf-8"))
        except OSError, UnicodeError, json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            if value.get("terminal") is False:
                return IncompleteWorkflowArtifacts(turn_number=turn_number, root=path, record=value)
            # A newer terminal workflow proves the session progressed beyond
            # any older abandoned record; never resurrect an earlier turn.
            return None
    return None


def load_private_json(path: Path) -> dict[str, object]:
    """Load one bounded, non-symlink owner artifact as a mapping."""
    if not path.is_file() or path.is_symlink() or path.stat().st_size > _PRIVATE_ARTIFACT_MAX_BYTES:
        raise OSError(f"unsafe or oversized workflow artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"workflow artifact is not a mapping: {path}")
    return value


def mark_workflow_recovered(artifacts: IncompleteWorkflowArtifacts, *, detail: str, conflicted: bool) -> None:
    """Terminalize a crash record after recovery or conflict detection."""
    payload = dict(artifacts.record)
    payload.update(
        {
            "phase": "conflicted" if conflicted else "degraded",
            "terminal": True,
            "detail": detail,
            "recovered_after_crash": not conflicted,
        }
    )
    atomic_write_owner_only_text(
        artifacts.root / "workflow.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
