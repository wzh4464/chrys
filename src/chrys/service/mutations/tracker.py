# Copyright (c) 2026 Chrys. All rights reserved.

"""MutationTracker — records file mutations and manages per-turn snapshots.

Created once per session.  Passed to tool implementations so they can
record mutations.  Each file is snapshotted at most once per turn
(first mutation wins for that turn).

Typical lifecycle::

    tracker = MutationTracker(SnapshotStore(session_dir))

    # --- turn 1 ---
    tracker.start_turn(1)
    # Called BEFORE the actual write — snapshots the pre-mutation file
    tracker.record("foo.py", MODIFY, EDIT_FILE, "call-001")
    tracker.record("foo.py", MODIFY, EDIT_FILE, "call-002")  # no re-snapshot

    # --- turn 2 (no file changes) ---
    tracker.start_turn(2)   # empty turn, still recorded

    # --- turn 3 ---
    tracker.start_turn(3)
    tracker.record("foo.py", MODIFY, EDIT_FILE, "call-005")  # re-snapshots

    # Multi-step rollback: undo turns 3+2 -> foo.py restored to turn-1 state
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from threading import RLock
from typing import Any, TypeVar

from chrys.service.mutations.store import SnapshotStore
from chrys.service.mutations.types import (
    FileHashDiff,
    FileMutation,
    FileSnapshot,
    MutationLog,
    MutationOp,
    MutationProvenance,
    MutationSource,
    RestoreResult,
    RollbackExclusionReason,
    RollbackPlan,
    SnapshotSkipReason,
    TurnMutations,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class _ProvenanceBadges:
    """Accumulates per-path provenance badge state while folding summaries.

    Shared by the turn and session summary folds so both apply the same
    rules: a ``FOREIGN`` row never contributes endpoints (and a path
    with only foreign rows is absent); a path that mixes foreign rows
    with ours renders contested (disk may differ from "our" after);
    ``inferred`` marks folds whose net change includes window-diff
    inference.  MOVE rows badge both ``path`` and ``old_path``,
    mirroring rollback's two-path treatment.
    """

    def __init__(self) -> None:
        self._foreign: set[str] = set()
        self._contested: set[str] = set()
        self._inferred: set[str] = set()

    def observe(self, m: FileMutation) -> bool:
        """Register one row; True when it is FOREIGN (caller skips folding)."""
        targets = (m.path, m.old_path) if m.old_path else (m.path,)
        if m.provenance is MutationProvenance.FOREIGN:
            self._foreign.update(targets)
            return True
        if m.contested:
            self._contested.update(targets)
        if m.provenance is MutationProvenance.ASSUMED:
            self._inferred.update(targets)
        return False

    def apply(self, result: dict[str, FileHashDiff]) -> dict[str, FileHashDiff]:
        """Set folded badges on entries (foreign-only paths are already absent)."""
        from dataclasses import replace

        for path, diff in result.items():
            contested = path in self._contested or path in self._foreign
            inferred = path in self._inferred
            if contested != diff.contested or inferred != diff.inferred:
                result[path] = replace(diff, contested=contested, inferred=inferred)
        return result


class MutationTracker:
    """Records file mutations and manages per-turn snapshots.

    Two layers of state tracking:

    1. **Turn-level snapshots** (``_ensure_snapshot``): captured once per
       (path, turn) at the first mutation.  Used for rollback — represents
       the file state at the *start* of the turn.

    2. **Last-known hash** (``_last_known_hash``): tracks the most recent
       observed content hash for each path *within the current turn*.
       Updated after every ``record_after()`` and shell ``record()``.
       Used to set accurate ``before_hash`` on each ``FileMutation`` so
       per-mutation DiffView shows the *incremental* change, not a diff
       against the turn-start state.
    """

    def __init__(self, store: SnapshotStore) -> None:
        self._store = store
        self._log = MutationLog()
        # Tracks which (path, turn_id) pairs have been snapshotted
        self._snapshotted: set[str] = set()  # uses MutationLog.snapshot_key
        # Last-known content hash per normalized path within the current turn.
        # Cleared on start_turn().  Enables accurate per-mutation before_hash
        # when the same file is mutated multiple times in one turn.
        self._last_known_hash: dict[str, str | None] = {}
        # Companion to _last_known_hash: why the last-known content backup
        # was withheld (SnapshotPolicy skip), or None when the hash is real
        # or the file was absent.  Keyed/cleared identically.
        self._last_known_skip: dict[str, SnapshotSkipReason | None] = {}
        # Per-file asyncio locks for serializing concurrent mutations to the
        # same file. The Chrys tool loop dispatches parallel tool calls via
        # asyncio.gather(); these locks ensure that record() → call_next() →
        # record_after() for the same file execute sequentially, keeping the
        # _last_known_hash chain accurate.  Acquired/released from the event
        # loop thread only (never from run_in_executor threads).
        self._file_locks: dict[str, asyncio.Lock] = {}
        # Serializes shell / implicit-detection windows inside a session so
        # before/after observers do not attribute another concurrent shell or
        # skill subprocess to the current tool call.
        self._implicit_window_lock = asyncio.Lock()
        # Protects tracker state because mutation finalization still performs
        # some recording work from executor threads.  Keep asyncio locks above
        # event-loop-only; this lock guards plain Python state and blob refs.
        self._state_lock = RLock()

    @property
    def log(self) -> MutationLog:
        """The underlying mutation log (read-only access)."""
        return self._log

    @property
    def store(self) -> SnapshotStore:
        """The snapshot store backing this tracker."""
        return self._store

    def with_locked_log(self, fn: Callable[[MutationLog], _T]) -> _T:
        """Run ``fn`` on the log under the tracker's state lock.

        Generic synchronization seam for components that read or
        re-attribute log rows without the tracker knowing about them
        (cross-session coordination reclassifies provenance this way —
        the tracker itself stays coordination-unaware).  Re-entering
        tracker methods from ``fn`` is safe (RLock), but ``fn`` must
        not block on I/O while holding the lock.
        """
        with self._state_lock:
            return fn(self._log)

    # -- Turn lifecycle --

    def start_turn(self, turn_id: int) -> None:
        """Begin tracking a new agent turn.

        Must be called at the start of each ``agent.run()`` cycle,
        even if no file mutations are expected.  This ensures the
        turn timeline is continuous for multi-step rollback.

        Clears ``_last_known_hash`` so each turn starts fresh —
        the first mutation per file reads from disk.
        """
        with self._state_lock:
            self._log.turns.append(TurnMutations(turn_id=turn_id))
            self._last_known_hash.clear()
            self._last_known_skip.clear()

    def reset_file_cache(self) -> None:
        """Clear per-file hash cache without starting a new turn.

        Used by retry/resume so subsequent mutations read fresh content
        from disk while still recording under the same turn.
        """
        with self._state_lock:
            self._last_known_hash.clear()
            self._last_known_skip.clear()

    def mark_detection_truncated(self) -> None:
        """Record that the active turn's implicit detection was incomplete."""
        with self._state_lock:
            if self._log.turns:
                self._log.turns[-1].detection_truncated = True

    @property
    def current_turn(self) -> TurnMutations | None:
        """The most recent turn, or None if ``start_turn`` hasn't been called."""
        with self._state_lock:
            return self._log.turns[-1] if self._log.turns else None

    def get_file_lock(self, path: str) -> asyncio.Lock:
        """Get an asyncio lock for serializing concurrent mutations to a file.

        The Chrys tool loop dispatches parallel tool calls via
        ``asyncio.gather()``. Callers (middleware) acquire this lock
        before ``record()`` and release after ``record_after()`` to
        ensure the ``_last_known_hash`` chain stays accurate.

        Must be called from the event loop thread, never from
        ``run_in_executor`` threads.
        """
        norm = os.path.normpath(os.path.abspath(path))
        with self._state_lock:
            if norm not in self._file_locks:
                self._file_locks[norm] = asyncio.Lock()
            return self._file_locks[norm]

    def get_implicit_window_lock(self) -> asyncio.Lock:
        """Lock for serializing shell and implicit mutation observation windows."""
        return self._implicit_window_lock

    # -- Recording mutations --

    def record(
        self,
        path: str,
        operation: MutationOp,
        source: MutationSource,
        tool_call_id: str,
        *,
        old_path: str | None = None,
        provenance: MutationProvenance | None = None,
    ) -> FileMutation | None:
        """Record a file mutation.

        **Calling convention**:

        - For file tools (``write_file``, ``edit_file``): call BEFORE
          the actual write, then call :meth:`record_after` when done.
        - For shell commands: call :meth:`pre_snapshot` BEFORE execution,
          then call ``record()`` AFTER execution for each detected change
          (no ``record_after`` needed).

        The turn-level snapshot (for rollback) is taken once per (path,
        turn) via ``_ensure_snapshot``.  The per-mutation ``before_hash``
        reflects the *actual current state* of the file — using
        ``_last_known_hash`` if the file was already mutated earlier in
        this turn, or falling back to the turn-level snapshot hash.

        ``provenance`` refines attribution beyond the source-derived
        default (``None``) — e.g. a trace-confirmed shell write records
        as ``PROVEN``.

        Returns the :class:`FileMutation` or ``None`` if no active turn.
        """
        with self._state_lock:
            turn = self._log.turns[-1] if self._log.turns else None
            if turn is None:
                logger.warning("MutationTracker.record() called with no active turn")
                return None

            norm = os.path.normpath(os.path.abspath(path))

            # Shell mutations call ``record()`` AFTER execution — reading from
            # disk now (via ``_ensure_snapshot``) would capture post-exec state
            # as the turn-start snapshot.  For CREATEs the scanner discovered
            # without a pre-snapshot (e.g. a file inside a newly-created dir),
            # synthesise the correct ``existed=False`` turn-start directly;
            # otherwise the mutation's before_hash == after_hash and the entry
            # drops out of rollback-diff aggregation as net-zero.
            snap_key = MutationLog.snapshot_key(norm, turn.turn_id)
            chain_severed = snap_key in self._snapshotted and norm not in self._last_known_hash
            if source == MutationSource.SHELL and operation == MutationOp.CREATE and snap_key not in self._snapshotted:
                snap = self._register_snapshot(
                    FileSnapshot(path=norm, turn_id=turn.turn_id, existed=False), key=snap_key
                )
            else:
                snap = self._ensure_snapshot(norm, turn.turn_id, key=snap_key)

            # before_hash: use last known hash if the file was already mutated
            # earlier in this turn; otherwise fall back to the turn-level snapshot.
            if norm in self._last_known_hash:
                before_hash = self._last_known_hash[norm]
                before_skip = self._last_known_skip.get(norm)
            elif chain_severed and source != MutationSource.SHELL:
                # Snapshotted this turn but the content chain was cleared
                # (retry/resume) or never completed (an interrupted write):
                # the snapshot may predate earlier same-turn mutations.
                # File-tool records run pre-write, so the true before
                # endpoint is the current disk content.  Shell records run
                # post-execution — too late to observe it — and rely on
                # pre_snapshot having reseeded the chain instead.
                current = self._store.save_blob(norm)
                before_hash = current.content_hash
                before_skip = current.skip_reason
            else:
                before_hash = snap.content_hash if snap else None
                before_skip = snap.skip_reason if snap else None

            # For MOVE, also snapshot the old path
            if old_path is not None:
                old_norm = os.path.normpath(os.path.abspath(old_path))
                self._ensure_snapshot(old_norm, turn.turn_id)
            else:
                old_norm = None

            mutation = FileMutation(
                path=norm,
                operation=operation,
                source=source,
                tool_call_id=tool_call_id,
                timestamp=time.time(),
                old_path=old_norm,
                before_hash=before_hash,
                before_skip=before_skip,
                provenance=provenance,
            )
            turn.mutations.append(mutation)

            # For shell mutations (recorded post-execution): capture the
            # post-execution state as after_hash and update last known hash.
            # File-tool mutations set after_hash via record_after() instead.
            if source == MutationSource.SHELL:
                if operation == MutationOp.DELETE:
                    mutation.after_hash = None
                    self._last_known_hash[norm] = None
                    self._last_known_skip[norm] = None
                else:
                    # CREATE or MODIFY — file now exists with new content
                    after = self._store.save_blob(norm)
                    mutation.after_hash = after.content_hash
                    mutation.after_skip = after.skip_reason
                    self._last_known_hash[norm] = after.content_hash
                    self._last_known_skip[norm] = after.skip_reason

            return mutation

    def record_after(self, mutation: FileMutation) -> None:
        """Snapshot the file AFTER mutation and store the hash.

        Called after a tool (write_file, edit_file) completes to
        capture the post-mutation content for session replay diffs.

        Also updates ``_last_known_hash`` so subsequent mutations to
        the same file within this turn get an accurate ``before_hash``.
        """
        with self._state_lock:
            after = self._store.save_blob(mutation.path)
            mutation.after_hash = after.content_hash
            mutation.after_skip = after.skip_reason
            # Update last known state for this file
            norm = os.path.normpath(os.path.abspath(mutation.path))
            self._last_known_hash[norm] = after.content_hash
            self._last_known_skip[norm] = after.skip_reason

    def pre_snapshot(self, paths: list[str]) -> None:
        """Snapshot files before a shell command that may modify them.

        Called with paths identified by :class:`ShellMutationDetector` or
        :class:`WorkspaceScanner` before shell execution.  The actual
        mutation recording happens after execution via :meth:`record`.

        Also initializes ``_last_known_hash`` for files not yet tracked
        in this turn, so that post-execution ``record()`` calls get
        accurate ``before_hash`` values.
        """
        with self._state_lock:
            turn = self._log.turns[-1] if self._log.turns else None
            if turn is None:
                return
            for path in paths:
                norm = os.path.normpath(os.path.abspath(path))
                snap_key = MutationLog.snapshot_key(norm, turn.turn_id)
                chain_severed = snap_key in self._snapshotted and norm not in self._last_known_hash
                snap = self._ensure_snapshot(norm, turn.turn_id, key=snap_key)
                # Initialize last known hash if this is the first time we see
                # this file in the current turn.  Uses the snapshot hash
                # (avoids a redundant file read since _ensure_snapshot already
                # read and hashed the content).
                if norm not in self._last_known_hash:
                    if chain_severed:
                        # Snapshotted this turn but the content chain was
                        # cleared (retry/resume) or never completed (an
                        # interrupted write): the snapshot may predate
                        # earlier same-turn mutations, so the chain must
                        # restart from what is on disk right now.  The
                        # snapshot stays the rollback anchor untouched.
                        current = self._store.save_blob(norm)
                        self._last_known_hash[norm] = current.content_hash
                        self._last_known_skip[norm] = current.skip_reason
                    else:
                        self._last_known_hash[norm] = snap.content_hash if snap else None
                        self._last_known_skip[norm] = snap.skip_reason if snap else None

    def record_calibrated(
        self,
        path: str,
        operation: MutationOp,
        before_data: bytes | None,
        tool_call_id: str = "git_calibration",
        *,
        provenance: MutationProvenance | None = None,
        before_skip: SnapshotSkipReason | None = None,
        before_size: int | None = None,
        before_symlink: bool = False,
        before_symlink_dir: bool | None = None,
    ) -> FileMutation | None:
        """Record an implicit mutation detected by git calibration at turn end.

        Unlike :meth:`record`, this is called *after* the turn completes
        with explicit before-content (typically from ``git show HEAD:path``).
        The before endpoint is resolved in priority order:

        - the ``_last_known_hash`` chain (content already tracked this
          turn) — same as :meth:`record` — so a path edited by a tool and
          then by an opaque shell command does not re-attribute the tool
          edit to this command;
        - fresh *before_data*: after a retry severs the chain, the old-HEAD
          blob is this mutation's true before endpoint, even when the path
          already has a turn snapshot from the earlier attempt;
        - *before_skip* (policy withheld the oversized old blob): the event
          is recorded with the skip reason, only the backup is missing;
        - the existing turn snapshot as post-hoc fallback; and
        - the nonexistent endpoint of a Git-confirmed CREATE.

        The turn-start snapshot stays the rollback anchor in every branch:
        it is registered only when the (path, turn) key has none yet, never
        rewritten.  Sets both ``before_hash`` and ``after_hash`` on the
        mutation.
        """
        with self._state_lock:
            turn = self._log.turns[-1] if self._log.turns else None
            if turn is None:
                return None

            norm = os.path.normpath(os.path.abspath(path))
            snap_key = MutationLog.snapshot_key(norm, turn.turn_id)

            if norm in self._last_known_hash:
                # Content already tracked this turn: chain from the last
                # known state so this row's diff starts where the previous
                # same-turn mutation ended, not at the turn-start snapshot.
                before_hash = self._last_known_hash[norm]
                before_skip = self._last_known_skip.get(norm)
            elif before_data is not None:
                # Fresh Git evidence beats the turn snapshot: with the chain
                # severed by a retry, the old-HEAD blob is this mutation's
                # true before endpoint.
                before_blob = self._store.save_data_as_blob(before_data)
                before_hash = before_blob.content_hash
                before_skip = before_blob.skip_reason
                if snap_key not in self._snapshotted:
                    self._register_snapshot(
                        FileSnapshot(
                            path=norm,
                            turn_id=turn.turn_id,
                            existed=True,
                            content_hash=before_hash,
                            size=len(before_data),
                            skip_reason=before_skip,
                            is_symlink=before_symlink,
                            # The kind is old-tree evidence supplied by the
                            # calibrator — the live filesystem is post-change
                            # and proves nothing about the old endpoint.
                            symlink_target_is_dir=before_symlink_dir if before_symlink else None,
                        ),
                        key=snap_key,
                    )
            elif before_skip is not None:
                # Policy withheld the before content (oversized old blob):
                # record the event with the skip reason; only the content
                # backup is missing.
                before_hash = None
                if snap_key not in self._snapshotted:
                    self._register_snapshot(
                        FileSnapshot(
                            path=norm,
                            turn_id=turn.turn_id,
                            existed=True,
                            size=before_size or 0,
                            skip_reason=before_skip,
                            is_symlink=before_symlink,
                        ),
                        key=snap_key,
                    )
            elif snap_key in self._snapshotted:
                # Snapshotted this turn but no fresh evidence for this pass:
                # calibration runs post-hoc, so the snapshot is the only
                # before endpoint left.
                snap = self._log.snapshots.get(snap_key)
                before_hash = snap.content_hash if snap else None
                before_skip = snap.skip_reason if snap else None
            else:
                # No before-data and no pre-snapshot (untracked new file)
                before_hash = None
                before_skip = None
                self._register_snapshot(
                    FileSnapshot(path=norm, turn_id=turn.turn_id, existed=False),
                    key=snap_key,
                )

            # Capture current disk state as after
            if operation == MutationOp.DELETE:
                after_hash = None
                after_skip = None
            else:
                after_blob = self._store.save_blob(norm)
                after_hash = after_blob.content_hash
                after_skip = after_blob.skip_reason

            mutation = FileMutation(
                path=norm,
                operation=operation,
                source=MutationSource.IMPLICIT,
                tool_call_id=tool_call_id,
                timestamp=time.time(),
                before_hash=before_hash,
                after_hash=after_hash,
                before_skip=before_skip,
                after_skip=after_skip,
                provenance=provenance,
            )
            turn.mutations.append(mutation)
            self._last_known_hash[norm] = after_hash
            self._last_known_skip[norm] = after_skip
            return mutation

    def get_changed_files_set(self) -> set[str]:
        """Normalized absolute paths of all files mutated in the current turn.

        ``FOREIGN`` rows are excluded: a path demoted to a peer session
        must stay eligible for fresh local detection (git calibration's
        already-tracked skip), or a later real local change on it would
        be silently dropped as "already tracked".
        """
        with self._state_lock:
            turn = self._log.turns[-1] if self._log.turns else None
            if turn is None:
                return set()
            paths: set[str] = set()
            for m in turn.mutations:
                if m.provenance is MutationProvenance.FOREIGN:
                    continue
                paths.add(m.path)
                if m.old_path:
                    paths.add(m.old_path)
            return paths

    def cleanup_unused_snapshots(self) -> int:
        """Remove snapshots from the current turn that have no corresponding mutations.

        Pre-snapshots are taken eagerly before shell commands for all git-dirty
        files, in case the command modifies them.  If no mutation is recorded
        for a file, its snapshot is wasted.  This method removes those orphans
        and deletes any blob files that are no longer referenced.

        Returns the number of snapshots removed.
        """
        with self._state_lock:
            turn = self._log.turns[-1] if self._log.turns else None
            if turn is None:
                return 0

            # Paths that have actual mutations in this turn
            mutated_paths: set[str] = set()
            for m in turn.mutations:
                mutated_paths.add(m.path)
                if m.old_path:
                    mutated_paths.add(m.old_path)

            # Find snapshots for this turn whose paths have no mutations
            orphan_keys: list[str] = []
            orphan_hashes: set[str] = set()
            for key, snap in self._log.snapshots.items():
                if snap.turn_id == turn.turn_id and snap.path not in mutated_paths:
                    orphan_keys.append(key)
                    if snap.content_hash:
                        orphan_hashes.add(snap.content_hash)

            if not orphan_keys:
                return 0

            # Remove orphan snapshot entries
            for key in orphan_keys:
                del self._log.snapshots[key]
                self._snapshotted.discard(key)

            # Only delete blobs not referenced by any remaining snapshot or mutation
            if orphan_hashes:
                still_referenced = self._all_referenced_hashes()
                truly_orphaned = orphan_hashes - still_referenced
                if truly_orphaned:
                    self._store.remove_blobs(truly_orphaned)

            logger.debug(
                "cleanup_unused_snapshots: removed %d orphan snapshots for turn %d", len(orphan_keys), turn.turn_id
            )
            return len(orphan_keys)

    def _ensure_snapshot(self, norm_path: str, turn_id: int, *, key: str | None = None) -> FileSnapshot | None:
        """Snapshot a file if not already done for this (path, turn).

        ``key`` is an optional caller-computed ``MutationLog.snapshot_key`` —
        pass it through to avoid recomputing the same string on the hot
        path (``record()`` already has it in hand).
        """
        if key is None:
            key = MutationLog.snapshot_key(norm_path, turn_id)
        if key not in self._snapshotted:
            return self._register_snapshot(self._store.save(norm_path, turn_id), key=key)
        return self._log.snapshots.get(key)

    def _register_snapshot(self, snapshot: FileSnapshot, *, key: str | None = None) -> FileSnapshot:
        """Insert a snapshot into the log and mark its (path, turn) key seen.

        Single bookkeeping primitive shared by ``_ensure_snapshot`` (disk
        read), the post-exec CREATE guard in ``record()`` (synthetic
        ``existed=False``), and ``record_calibrated`` (git-HEAD blob).
        """
        if key is None:
            key = MutationLog.snapshot_key(snapshot.path, snapshot.turn_id)
        self._log.snapshots[key] = snapshot
        self._snapshotted.add(key)
        return snapshot

    # -- Query methods --

    def get_turn_mutations(self, turn_id: int) -> TurnMutations | None:
        """Get mutations for a specific turn (None if turn not found)."""
        with self._state_lock:
            for turn in self._log.turns:
                if turn.turn_id == turn_id:
                    return turn
            return None

    def get_all_turns(self) -> list[TurnMutations]:
        """All turns in order (including empty ones)."""
        with self._state_lock:
            return list(self._log.turns)

    def get_changed_files(self) -> list[str]:
        """Unique file paths mutated across all turns (ordered by first mutation)."""
        with self._state_lock:
            seen: dict[str, None] = {}
            for turn in self._log.turns:
                for m in turn.mutations:
                    seen.setdefault(m.path, None)
                    if m.old_path:
                        seen.setdefault(m.old_path, None)
            return list(seen)

    def get_original_snapshot(self, path: str) -> FileSnapshot | None:
        """Get the *earliest* snapshot for a file — its pre-session state.

        Used by ``/diff`` to compare original state vs current disk.
        """
        with self._state_lock:
            norm = os.path.normpath(os.path.abspath(path))
            earliest: FileSnapshot | None = None
            for snap in self._log.snapshots.values():
                if snap.path == norm and (earliest is None or snap.turn_id < earliest.turn_id):
                    earliest = snap
            return earliest

    def get_snapshot(self, path: str, turn_id: int) -> FileSnapshot | None:
        """Get the snapshot for a specific file and turn."""
        with self._state_lock:
            key = MutationLog.snapshot_key(path, turn_id)
            return self._log.snapshots.get(key)

    def get_turn_file_summary(self, turn_id: int) -> dict[str, FileHashDiff]:
        """Return per-file ``{path: FileHashDiff}`` for a turn.

        - ``before``: file state at the *start* of the turn (from the
          turn-level snapshot).  ``None`` means the file did not exist.
        - ``after``: file state after the *last* mutation to that file
          in the turn.  ``None`` means the file was deleted or no
          ``record_after()`` was called (shell mutations).

        This gives the true before-turn vs after-turn state for every
        file touched, collapsing multiple intermediate edits.

        ``FOREIGN`` rows are excluded from the fold (a peer's delta must
        not appear inside "our" endpoints); a path with only foreign
        rows is absent entirely.  Folded ``contested``/``inferred``
        badges are set per the rules on :class:`FileHashDiff`.
        """
        with self._state_lock:
            turn = None
            for candidate in self._log.turns:
                if candidate.turn_id == turn_id:
                    turn = candidate
                    break
            if turn is None:
                return {}

            result: dict[str, FileHashDiff] = {}
            badges = _ProvenanceBadges()
            for m in turn.mutations:
                if badges.observe(m):
                    continue  # foreign row — never folds into endpoints
                snap_key = MutationLog.snapshot_key(m.path, turn_id)
                snap = self._log.snapshots.get(snap_key)
                turn_before = snap.content_hash if snap else None
                turn_before_skip = snap.skip_reason if snap else None

                # For multiple mutations to the same file, keep the original
                # before and update after to the latest.
                if m.path in result:
                    prev = result[m.path]
                    result[m.path] = FileHashDiff(
                        before=prev.before,
                        after=m.after_hash,
                        before_skip=prev.before_skip,
                        after_skip=m.after_skip,
                    )
                else:
                    result[m.path] = FileHashDiff(
                        before=turn_before,
                        after=m.after_hash,
                        before_skip=turn_before_skip,
                        after_skip=m.after_skip,
                    )

            return badges.apply(result)

    def get_session_file_summary(self) -> dict[str, FileHashDiff]:
        """Return per-file ``{path: FileHashDiff}`` aggregated across all turns.

        - ``before``: file state at the start of the *first* turn that
          touched the file (i.e. its pre-session state).  ``None`` means
          the file did not exist before the session.
        - ``after``: file state after the *last* mutation to that file
          across the entire session.  ``None`` means the file was deleted
          (or ``record_after()`` was never called, e.g. for shell
          mutations).

        MOVE mutations (``m.old_path`` set) contribute an implicit
        DELETE on the old path — consistent with ``get_changed_files``
        and ``get_rollback_plan``.

        Entries with ``is_net_zero`` (no provable net change — hashes
        match and existence didn't flip) are excluded.  This gives the
        true session-wide before/after state for every file that was
        meaningfully modified.  Skip-marked creates/deletes ARE included
        (existence changed even though both hashes read ``None``); the
        fully-skipped modify is the one skip case that stays excluded.

        Turns are iterated in ``turn_id`` order so "earliest before" and
        "latest after" are chronological, not insertion-order — session
        restore / retry paths can append turns with non-monotonic ids.

        ``FOREIGN`` rows are excluded from the fold — see
        :meth:`get_turn_file_summary` — and folded ``contested`` /
        ``inferred`` badges are set on the surviving entries.
        """
        with self._state_lock:
            result: dict[str, FileHashDiff] = {}
            badges = _ProvenanceBadges()

            def _set_before_if_first(path: str, turn_id: int) -> tuple[str | None, SnapshotSkipReason | None]:
                """Record earliest before_hash/skip for ``path``; return them."""
                if path not in result:
                    snap_key = MutationLog.snapshot_key(path, turn_id)
                    snap = self._log.snapshots.get(snap_key)
                    before = snap.content_hash if snap else None
                    before_skip = snap.skip_reason if snap else None
                    result[path] = FileHashDiff(before=before, after=None, before_skip=before_skip)
                    return before, before_skip
                return result[path].before, result[path].before_skip

            for turn in sorted(self._log.turns, key=lambda t: t.turn_id):
                for m in turn.mutations:
                    if badges.observe(m):
                        continue  # foreign row — never folds into endpoints
                    before, before_skip = _set_before_if_first(m.path, turn.turn_id)
                    result[m.path] = FileHashDiff(
                        before=before,
                        after=m.after_hash,
                        before_skip=before_skip,
                        after_skip=m.after_skip,
                    )

                    # MOVE: old_path is effectively deleted (its content moved away).
                    if m.old_path:
                        old_before, old_before_skip = _set_before_if_first(m.old_path, turn.turn_id)
                        result[m.old_path] = FileHashDiff(before=old_before, after=None, before_skip=old_before_skip)

            result = badges.apply(result)
            return {p: d for p, d in result.items() if not d.is_net_zero}

    def get_file_edit_snapshots(self) -> list[tuple[str, str]]:
        """Return ``(before_text, after_text)`` for all file-tool mutations, in order.

        Only includes mutations from ``write_file`` and ``edit_file``
        (shell mutations are excluded).  Missing blobs resolve to empty
        strings (e.g. ``before_text=""`` for newly created files).
        Used to build the replay mapping for DiffView rendering.
        """
        from chrys.foundation.text.encoding import decode_bytes

        with self._state_lock:
            _file_sources = {MutationSource.WRITE_FILE, MutationSource.EDIT_FILE}
            results: list[tuple[str, str]] = []
            for turn in self._log.turns:
                for m in turn.mutations:
                    if m.source not in _file_sources:
                        continue
                    before_bytes = self._store.read_blob(m.before_hash) if m.before_hash else None
                    after_bytes = self._store.read_blob(m.after_hash) if m.after_hash else None
                    before_text = decode_bytes(before_bytes) if before_bytes else ""
                    after_text = decode_bytes(after_bytes) if after_bytes else ""
                    results.append((before_text, after_text))
            return results

    def get_file_edit_snapshot_refs(self) -> list[FileHashDiff]:
        """Return before/after blob hashes for all file-tool mutations, in order."""
        with self._state_lock:
            _file_sources = {MutationSource.WRITE_FILE, MutationSource.EDIT_FILE}
            results: list[FileHashDiff] = []
            for turn in self._log.turns:
                for m in turn.mutations:
                    if m.source not in _file_sources:
                        continue
                    results.append(
                        FileHashDiff(
                            before=m.before_hash,
                            after=m.after_hash,
                            before_skip=m.before_skip,
                            after_skip=m.after_skip,
                        )
                    )
            return results

    # -- Rollback --

    def _build_rollback_plan(self, rolled_back: list[TurnMutations]) -> RollbackPlan:
        """Earliest-wins restore plan over ``rolled_back`` turns.

        Caller must hold ``_state_lock``.

        Safety filters run AFTER earliest-wins selection; every filtered
        path is reported via :attr:`RollbackPlan.exclusions` (never
        silent):

        1. **Provenance** — a path touched by a ``FOREIGN`` row, or by a
           ``contested`` row, is excluded outright (poisoning-style):
           restoring "our" pre-window snapshot over a path a peer
           session also wrote would revert the peer's work.
        2. **Restorability** — a non-restorable earliest snapshot
           (content withheld by :class:`SnapshotPolicy`) excludes the
           file outright rather than silently restoring a misleading
           mid-window state.
        3. **Move safety** — a MOVE is atomic for rollback: when the
           source's pre-window content is unrecoverable, deleting (or
           overwriting) the destination would destroy the only surviving
           copy, so the destination is excluded too.  Applied through
           the explicit ``old_path`` pairing (with chronological
           propagation across chained moves), plus a conservative
           fallback for scanner-decomposed moves (``mv`` observed as
           independent CREATE + DELETE): if any unrecoverable-origin
           file was vacated in-window, every plan entry whose current
           on-disk content is skip-marked is left untouched — it may be
           that escaped content.
        """
        earliest: dict[str, FileSnapshot] = {}
        foreign_touched: set[str] = set()
        contested: set[str] = set()
        for turn in rolled_back:
            for m in turn.mutations:
                if m.provenance is MutationProvenance.FOREIGN:
                    foreign_touched.add(m.path)
                    if m.old_path:
                        foreign_touched.add(m.old_path)
                if m.contested:
                    contested.add(m.path)
                    if m.old_path:
                        contested.add(m.old_path)
                key = MutationLog.snapshot_key(m.path, turn.turn_id)
                snap = self._log.snapshots.get(key)
                if snap and (m.path not in earliest or turn.turn_id < earliest[m.path].turn_id):
                    earliest[m.path] = snap
                if m.old_path:
                    old_key = MutationLog.snapshot_key(m.old_path, turn.turn_id)
                    old_snap = self._log.snapshots.get(old_key)
                    if old_snap and (m.old_path not in earliest or turn.turn_id < earliest[m.old_path].turn_id):
                        earliest[m.old_path] = old_snap

        poisoned: set[str] = set()
        vacated: set[str] = set()
        last_after_skip: dict[str, SnapshotSkipReason | None] = {}
        for turn in sorted(rolled_back, key=lambda t: t.turn_id):
            for m in turn.mutations:
                if m.operation is MutationOp.DELETE:
                    vacated.add(m.path)
                last_after_skip[m.path] = m.after_skip
                if m.old_path:
                    vacated.add(m.old_path)
                    src = earliest.get(m.old_path)
                    if src is None or not src.restorable or m.old_path in poisoned:
                        poisoned.add(m.path)

        unrecoverable_escape = any((snap := earliest.get(path)) is not None and not snap.restorable for path in vacated)
        if unrecoverable_escape:
            poisoned.update(path for path in earliest if last_after_skip.get(path) is not None)

        # Reason precedence (display-only, one per path): strongest
        # attribution first — not-ours beats ours-but-conflicted beats
        # content-availability mechanics.
        entries: list[tuple[str, FileSnapshot]] = []
        exclusions: list[tuple[str, RollbackExclusionReason]] = []
        for path, snap in earliest.items():
            if path in foreign_touched:
                exclusions.append((path, RollbackExclusionReason.FOREIGN))
            elif path in contested:
                exclusions.append((path, RollbackExclusionReason.CONTESTED))
            elif not snap.restorable:
                exclusions.append((path, RollbackExclusionReason.UNRESTORABLE))
            elif path in poisoned:
                exclusions.append((path, RollbackExclusionReason.MOVE_POISONED))
            else:
                entries.append((path, snap))
        exclusions.sort(key=lambda item: item[0])
        return RollbackPlan(entries=entries, exclusions=exclusions)

    def get_rollback_plan(self, num_turns: int = 1) -> RollbackPlan:
        """Compute which files to restore for rolling back the last N turns.

        For each file mutated in the rolled-back turns, the snapshot
        from the *earliest* rolled-back turn that touched that file is
        used — representing the file state just before those turns began.

        Returns:
            A :class:`RollbackPlan`.  ``entries`` holds ``(path,
            snapshot)`` pairs: ``existed=False`` snapshots mean the file
            should be deleted (it was created during the rolled-back
            turns), ``existed=True`` means restore from blob.  Files
            rollback cannot safely act on are listed in ``exclusions``
            with their reason — see :meth:`_build_rollback_plan` for the
            exact filters.
        """
        with self._state_lock:
            if not self._log.turns or num_turns < 1:
                return RollbackPlan(entries=[])

            n = min(num_turns, len(self._log.turns))
            return self._build_rollback_plan(self._log.turns[-n:])

    def get_rollback_plan_for_turns(self, turn_ids: set[int]) -> RollbackPlan:
        """Compute which files to restore for rolling back explicit turn IDs.

        Unlike :meth:`get_rollback_plan`, this is independent of log
        insertion order.  Rollback controller callers use this when they
        already know the target turn and need to undo every mutation with
        ``turn_id > target_turn``.
        """
        with self._state_lock:
            if not self._log.turns or not turn_ids:
                return RollbackPlan(entries=[])

            return self._build_rollback_plan([turn for turn in self._log.turns if turn.turn_id in turn_ids])

    def rollback(self, num_turns: int = 1, only_paths: set[str] | None = None) -> list[RestoreResult]:
        """Execute a rollback of the last N turns.

        Restores each affected file to its pre-rollback-window state
        using :meth:`SnapshotStore.restore`, then removes the
        rolled-back turns and their snapshots from the log.

        CONTRACT (see :meth:`SnapshotStore.restore` for the full text):
        revert is unconditional — each file in the plan is forced to
        the snapshot's pre-turn state regardless of its current disk
        state.  Users opt out of a file by de-selecting it in the
        rollback modal before pressing confirm; by the time we get
        here the selection IS the user's decision.

        Args:
            num_turns: How many most-recent turns to remove from the log.
            only_paths: If not ``None``, restore only the files whose
                absolute path is in this set.  Paths not in the set keep
                their current on-disk content — the rollback modal's
                per-file checkbox selection passes the checked paths
                here.  ``None`` preserves the historical "restore
                everything in the plan" behaviour for non-UI callers.

        Returns:
            One :class:`RestoreResult` per file in the plan (after
            ``only_paths`` filtering), carrying the per-path outcome.
            Use ``r.changed`` to count files whose on-disk state
            actually changed, ``r.ok`` for "ended up in target state
            (applied or already matched)", or inspect ``r.reason`` on
            FAILED entries for the specific failure mode.
        """
        with self._state_lock:
            entries = self.get_rollback_plan(num_turns).entries
            if only_paths is not None:
                entries = [(path, snapshot) for path, snapshot in entries if path in only_paths]

            results = self._restore_plan(entries)

            # Remove rolled-back turns from the log + clean up
            self._remove_last_turns(min(num_turns, len(self._log.turns)))

            return results

    def rollback_turns(
        self,
        turn_ids: set[int],
        only_paths: set[str] | None = None,
        *,
        plan: RollbackPlan | None = None,
    ) -> list[RestoreResult]:
        """Execute a rollback for explicit turn IDs.

        This is the controller-facing variant used when the target turn
        is known.  It removes all matching turn entries from the mutation
        log, regardless of their insertion order.

        ``plan`` lets the caller execute from a pre-built
        :class:`RollbackPlan` — the engine builds the plan *first* so
        its exclusions/warnings can still be reported after this method
        removes the turns they were derived from (they cannot be
        reconstructed afterwards).  ``None`` builds the plan here.
        """
        with self._state_lock:
            if not turn_ids:
                return []

            if plan is None:
                plan = self.get_rollback_plan_for_turns(turn_ids)
            entries = plan.entries
            if only_paths is not None:
                entries = [(path, snapshot) for path, snapshot in entries if path in only_paths]

            results = self._restore_plan(entries)
            self._remove_turn_ids(turn_ids)
            return results

    def _restore_plan(self, plan: list[tuple[str, FileSnapshot]]) -> list[RestoreResult]:
        """Restore each snapshot in a rollback plan best-effort."""
        from chrys.service.mutations.types import RestoreOutcome

        results: list[RestoreResult] = []
        for _path, snapshot in plan:
            try:
                results.append(self._store.restore(snapshot))
            except Exception as exc:
                logger.exception("Rollback: unexpected error restoring %s", snapshot.path)
                results.append(
                    RestoreResult(path=snapshot.path, outcome=RestoreOutcome.FAILED, reason=str(exc)),
                )
        return results

    # -- Cleanup --

    def remove_turn(self, turn_id: int) -> set[str]:
        """Remove a specific turn and its snapshots from the log.

        Deletes blob files that are no longer referenced by any remaining
        snapshot or mutation.  Returns the set of content hashes whose
        blobs were removed from disk.

        This is a data-management operation — it does NOT restore files.
        Use :meth:`rollback` to undo mutations.
        """
        with self._state_lock:
            # Collect all hashes from this turn: snapshots + mutation after_hashes
            turn_hashes: set[str] = set()
            keys_to_remove: list[str] = []
            for key, snap in self._log.snapshots.items():
                if snap.turn_id == turn_id:
                    if snap.content_hash:
                        turn_hashes.add(snap.content_hash)
                    keys_to_remove.append(key)
                    self._snapshotted.discard(key)

            # Collect after_hash values from this turn's mutations
            removed_turn = None
            for t in self._log.turns:
                if t.turn_id == turn_id:
                    removed_turn = t
                    break
            if removed_turn:
                for m in removed_turn.mutations:
                    if m.before_hash:
                        turn_hashes.add(m.before_hash)
                    if m.after_hash:
                        turn_hashes.add(m.after_hash)

            for key in keys_to_remove:
                del self._log.snapshots[key]

            # Remove the turn entry
            self._log.turns = [t for t in self._log.turns if t.turn_id != turn_id]

            # Find hashes still referenced by remaining snapshots or mutations
            still_referenced = self._all_referenced_hashes()
            orphaned = turn_hashes - still_referenced

            if orphaned:
                self._store.remove_blobs(orphaned)

            return orphaned

    def clear(self) -> None:
        """Remove all turns, snapshots, and blob files.

        Called when a session is deleted or fully reset.
        """
        with self._state_lock:
            self._log.turns.clear()
            self._log.snapshots.clear()
            self._snapshotted.clear()
            self._store.remove_all()

    def _all_referenced_hashes(self) -> set[str]:
        """Collect all content hashes still referenced by remaining data."""
        hashes: set[str] = set()
        for s in self._log.snapshots.values():
            if s.content_hash:
                hashes.add(s.content_hash)
        for t in self._log.turns:
            for m in t.mutations:
                if m.before_hash:
                    hashes.add(m.before_hash)
                if m.after_hash:
                    hashes.add(m.after_hash)
        return hashes

    def _remove_last_turns(self, n: int) -> None:
        """Remove the last *n* turns and clean up orphaned blobs."""
        if n <= 0 or not self._log.turns:
            return

        removed_turns = self._log.turns[-n:]
        self._log.turns = self._log.turns[:-n]

        removed_hashes: set[str] = set()
        for turn in removed_turns:
            for m in turn.mutations:
                # A calibrated row's fresh before blob is neither the turn
                # snapshot nor any after endpoint — collect both sides.
                if m.before_hash:
                    removed_hashes.add(m.before_hash)
                if m.after_hash:
                    removed_hashes.add(m.after_hash)
                # Collect snapshot (before) hashes
                key = MutationLog.snapshot_key(m.path, turn.turn_id)
                snap = self._log.snapshots.pop(key, None)
                self._snapshotted.discard(key)
                if snap and snap.content_hash:
                    removed_hashes.add(snap.content_hash)
                if m.old_path:
                    old_key = MutationLog.snapshot_key(m.old_path, turn.turn_id)
                    old_snap = self._log.snapshots.pop(old_key, None)
                    self._snapshotted.discard(old_key)
                    if old_snap and old_snap.content_hash:
                        removed_hashes.add(old_snap.content_hash)

        # Only delete blobs not referenced by remaining snapshots/mutations
        still_referenced = self._all_referenced_hashes()
        orphaned = removed_hashes - still_referenced
        if orphaned:
            self._store.remove_blobs(orphaned)

    def _remove_turn_ids(self, turn_ids: set[int]) -> None:
        """Remove explicit turn IDs and clean up orphaned blobs."""
        if not turn_ids or not self._log.turns:
            return

        removed_turns = [turn for turn in self._log.turns if turn.turn_id in turn_ids]
        if not removed_turns:
            return
        self._log.turns = [turn for turn in self._log.turns if turn.turn_id not in turn_ids]

        removed_hashes: set[str] = set()
        for turn in removed_turns:
            for m in turn.mutations:
                if m.before_hash:
                    removed_hashes.add(m.before_hash)
                if m.after_hash:
                    removed_hashes.add(m.after_hash)

        keys_to_remove = [key for key, snap in self._log.snapshots.items() if snap.turn_id in turn_ids]
        for key in keys_to_remove:
            snap = self._log.snapshots.pop(key)
            self._snapshotted.discard(key)
            if snap.content_hash:
                removed_hashes.add(snap.content_hash)

        still_referenced = self._all_referenced_hashes()
        orphaned = removed_hashes - still_referenced
        if orphaned:
            self._store.remove_blobs(orphaned)

    # -- Serialization --

    def serialize(self) -> dict[str, Any]:
        """Serialize for session persistence (JSON-compatible dict)."""
        with self._state_lock:
            return {
                "turns": [t.to_dict() for t in self._log.turns],
                "snapshots": {k: s.to_dict() for k, s in self._log.snapshots.items()},
            }

    @classmethod
    def deserialize(cls, data: dict[str, Any], store: SnapshotStore) -> MutationTracker:
        """Restore a MutationTracker from serialized session data."""
        tracker = cls(store)
        for t_data in data.get("turns", []):
            tracker._log.turns.append(TurnMutations.from_dict(t_data))
        for key, s_data in data.get("snapshots", {}).items():
            snap = FileSnapshot.from_dict(s_data)
            tracker._log.snapshots[key] = snap
            tracker._snapshotted.add(key)
        return tracker
