# Copyright (c) 2026 Chrys. All rights reserved.

"""Diff entry data models shared by TUI diff surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from chrys.service.mutations.types import MutationOp


@dataclass
class DiffFileEntry:
    """One changed file to display in the diff viewer."""

    path: str  # absolute path
    rel_path: str  # relative to workspace cwd (for display)
    operation: MutationOp  # CREATE, MODIFY, DELETE, MOVE
    old_path: str | None  # source path for MOVE
    before_text: str  # "" if file didn't exist or binary
    after_text: str  # "" if file was deleted or binary
    is_binary: bool  # True if null bytes detected in content
    encoding: str  # detected encoding of the file content
    bytes_changed: bool = False  # True when before/after snapshot hashes differ
    before_hash: str | None = None
    after_hash: str | None = None
    source: str = ""  # mutation source, e.g. "implicit" for uncertain git-calibrated changes
    # SnapshotSkipReason value ("too_large" / "binary") when the file's
    # content backup was withheld by SnapshotPolicy — the entry is still
    # shown/actionable but has no diff content to render.  "" = content
    # available.
    content_omitted: str = ""
    # Folded provenance badges: a peer
    # session also wrote this path / the net change includes window-diff
    # inference.
    contested: bool = False
    inferred: bool = False


@dataclass
class DiffLoadResult:
    """Result of loading diff data from a persisted session.

    ``all_entries`` — net session-wide changeset (for the "All" tab).
    ``per_turn_entries`` — ``{display_turn_number: [DiffFileEntry, ...]}``
    for every turn that had file changes.  Display numbers are 1-based
    sequential — internal turn IDs may drift due to retries/restores.
    ``total_turns`` — total number of turns recorded in the tracker
    (including empty ones).  Callers use this to compute the correct
    turn number for live mutations when the agent is still running.
    """

    all_entries: list[DiffFileEntry]
    per_turn_entries: dict[int, list[DiffFileEntry]]
    total_turns: int

    @classmethod
    def empty(cls) -> DiffLoadResult:
        return cls(all_entries=[], per_turn_entries={}, total_turns=0)


def entry_has_visible_change(entry: DiffFileEntry) -> bool:
    """Return whether a diff entry should appear in the tree."""
    return entry.bytes_changed or entry.before_text != entry.after_text or entry.operation is MutationOp.MOVE
