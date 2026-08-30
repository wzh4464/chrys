# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for folder-based sessions, mutation tracking, and middleware capture."""

from __future__ import annotations

import dataclasses
import json
import os
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from chrys.foundation.platform import get_platform
from chrys.kernel import Message
from chrys.service.agent_middleware import ToolEventMiddleware
from chrys.service.mutations import store as mutation_store
from chrys.service.mutations.store import SnapshotStore
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.types import FileSnapshot, MutationOp, MutationSource, RestoreOutcome
from chrys.service.state.store import JsonFileStateStore

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(msgs: list[Message] | None = None) -> dict[str, Any]:
    return {"messages": msgs or [], "compressed_msgs": []}


def _write_legacy_session(tmp_path: Path, session_id: str) -> Path:
    """Write a session in old flat-file format and return its path."""
    safe_id = session_id.replace("/", "_").replace("\\", "_")
    path = tmp_path / f"{safe_id}.json"
    data = {
        "meta": {
            "session_id": session_id,
            "agent_profile": "legacy",
            "display_name": "Legacy Agent",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "message_count": 1,
        },
        "state": {"messages": [{"role": "user", "contents": [{"type": "text", "text": "hello"}]}]},
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ===========================================================================
# Store — folder-based sessions
# ===========================================================================


class TestFolderBasedStore:
    """JsonFileStateStore stores sessions as {id}/session.json folders."""

    @pytest.mark.asyncio
    async def test_save_creates_folder_structure(self, tmp_path: Path) -> None:
        store = JsonFileStateStore(tmp_path)
        await store.save_session("abc123", _make_state([Message("user", ["hi"])]), agent_profile="code")

        session_json = tmp_path / "abc123" / "session.json"
        assert session_json.exists()
        # Old flat-file should NOT exist
        assert not (tmp_path / "abc123.json").exists()


# ===========================================================================
# Store — legacy migration
# ===========================================================================


class TestLegacyMigration:
    """Old {id}.json flat files are migrated into {id}/session.json on access."""

    @pytest.mark.asyncio
    async def test_load_migrates_legacy_file(self, tmp_path: Path) -> None:
        legacy_path = _write_legacy_session(tmp_path, "old_sess")
        assert legacy_path.exists()

        store = JsonFileStateStore(tmp_path)
        loaded = await store.load_session("old_sess")
        assert loaded is not None

        # Legacy file should be gone, folder should exist
        assert not legacy_path.exists()
        assert (tmp_path / "old_sess" / "session.json").exists()

    @pytest.mark.asyncio
    async def test_load_raw_migrates_legacy_file(self, tmp_path: Path) -> None:
        _write_legacy_session(tmp_path, "old_raw")
        store = JsonFileStateStore(tmp_path)
        raw = await store.load_session_raw("old_raw")
        assert raw is not None
        assert len(raw) == 1
        assert raw[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_list_includes_legacy_sessions(self, tmp_path: Path) -> None:
        """list_sessions finds both folder-based and legacy flat-file sessions."""
        store = JsonFileStateStore(tmp_path)
        await store.save_session("new_sess", _make_state(), agent_profile="new")
        _write_legacy_session(tmp_path, "legacy_sess")

        sessions = await store.list_sessions()
        ids = {s.session_id for s in sessions}
        assert "new_sess" in ids
        assert "legacy_sess" in ids

    @pytest.mark.asyncio
    async def test_save_migrates_then_updates(self, tmp_path: Path) -> None:
        """save_session on a legacy session migrates it first, then saves."""
        _write_legacy_session(tmp_path, "migrating")
        store = JsonFileStateStore(tmp_path)
        await store.save_session("migrating", _make_state([Message("user", ["new"])]), agent_profile="updated")

        # Legacy gone, folder exists
        assert not (tmp_path / "migrating.json").exists()
        session_json = tmp_path / "migrating" / "session.json"
        assert session_json.exists()
        raw = json.loads(session_json.read_text(encoding="utf-8"))
        assert raw["meta"]["agent_profile"] == "updated"

    @pytest.mark.asyncio
    async def test_delete_cleans_up_legacy_file(self, tmp_path: Path) -> None:
        _write_legacy_session(tmp_path, "legacy_del")
        store = JsonFileStateStore(tmp_path)
        await store.delete_session("legacy_del")
        assert not (tmp_path / "legacy_del.json").exists()
        assert not (tmp_path / "legacy_del").exists()


# ===========================================================================
# MutationTracker — before/after snapshot capture
# ===========================================================================


class TestMutationTrackerSnapshots:
    """MutationTracker captures before/after content hashes on file mutations."""

    def test_record_sets_before_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("original", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        mutation = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call_1")
        assert mutation is not None
        assert mutation.before_hash is not None

    def test_record_after_sets_after_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("original", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        mutation = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call_1")

        f.write_text("modified", encoding="utf-8")
        tracker.record_after(mutation)

        assert mutation.after_hash is not None
        assert mutation.before_hash != mutation.after_hash

    def test_get_file_edit_snapshots(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("before text", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call_1")
        f.write_text("after text", encoding="utf-8")
        tracker.record_after(m)

        snapshots = tracker.get_file_edit_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0] == ("before text", "after text")

    def test_new_file_has_empty_before(self, tmp_path: Path) -> None:
        f = tmp_path / "new.txt"

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        m = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "call_1")
        f.write_text("created!", encoding="utf-8")
        tracker.record_after(m)

        snapshots = tracker.get_file_edit_snapshots()
        assert snapshots[0] == ("", "created!")

    def test_multiple_edits_ordered(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("a_before", encoding="utf-8")
        f2.write_text("b_before", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        m1 = tracker.record(str(f1), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f1.write_text("a_after", encoding="utf-8")
        tracker.record_after(m1)

        m2 = tracker.record(str(f2), MutationOp.MODIFY, MutationSource.WRITE_FILE, "c2")
        f2.write_text("b_after", encoding="utf-8")
        tracker.record_after(m2)

        snapshots = tracker.get_file_edit_snapshots()
        assert len(snapshots) == 2
        assert snapshots[0] == ("a_before", "a_after")
        assert snapshots[1] == ("b_before", "b_after")

    def test_serialize_deserialize_preserves_hashes(self, tmp_path: Path) -> None:
        f = tmp_path / "test.py"
        f.write_text("v1", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("v2", encoding="utf-8")
        tracker.record_after(m)

        data = tracker.serialize()
        restored = MutationTracker.deserialize(data, SnapshotStore(tmp_path))
        snapshots = restored.get_file_edit_snapshots()
        assert snapshots == [("v1", "v2")]

    def test_shell_mutations_excluded_from_file_edit_snapshots(self, tmp_path: Path) -> None:
        """SHELL mutations are excluded even when both hashes are set."""
        f = tmp_path / "test.py"
        f.write_text("before", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.SHELL, "c1")
        f.write_text("after", encoding="utf-8")
        tracker.record_after(m)

        # Both hashes are set, but source is SHELL -> excluded
        assert m.before_hash is not None
        assert m.after_hash is not None
        assert tracker.get_file_edit_snapshots() == []


# ===========================================================================
# MutationTracker — edge cases and lifecycle
# ===========================================================================


class TestMutationTrackerEdgeCases:
    """Edge cases: deduplication, no-turn guard, pre_snapshot, binary files."""

    def test_record_no_active_turn_returns_none(self, tmp_path: Path) -> None:
        """record() returns None and logs a warning when no turn is active."""
        tracker = MutationTracker(SnapshotStore(tmp_path))
        result = tracker.record("/tmp/x.py", MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        assert result is None

    def test_same_file_multiple_edits_accurate_before_hash(self, tmp_path: Path) -> None:
        """Multiple edits: each mutation's before_hash reflects actual state before that call."""
        f = tmp_path / "test.py"
        f.write_text("original", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("v2", encoding="utf-8")
        tracker.record_after(m1)

        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("v3", encoding="utf-8")
        tracker.record_after(m2)

        # m1 before = original (first mutation, no prior state)
        # m2 before = v2 (the result of m1, not the turn-start state)
        assert m1.before_hash != m2.before_hash
        assert m1.after_hash != m2.after_hash
        # m1.after == m2.before (incremental chain)
        assert m1.after_hash == m2.before_hash

        # DiffView shows incremental diffs, not diffs against turn start
        snapshots = tracker.get_file_edit_snapshots()
        assert len(snapshots) == 2
        assert snapshots[0] == ("original", "v2")
        assert snapshots[1] == ("v2", "v3")

    def test_get_file_lock_returns_same_lock_for_same_path(self, tmp_path: Path) -> None:
        """get_file_lock returns the same asyncio.Lock for the same normalized path."""
        tracker = MutationTracker(SnapshotStore(tmp_path))
        f = tmp_path / "test.py"
        lock1 = tracker.get_file_lock(str(f))
        lock2 = tracker.get_file_lock(str(f))
        assert lock1 is lock2

    def test_get_file_lock_returns_different_lock_for_different_path(self, tmp_path: Path) -> None:
        """get_file_lock returns different locks for different files."""
        tracker = MutationTracker(SnapshotStore(tmp_path))
        lock1 = tracker.get_file_lock(str(tmp_path / "a.py"))
        lock2 = tracker.get_file_lock(str(tmp_path / "b.py"))
        assert lock1 is not lock2

    def test_pre_snapshot_captures_before_state(self, tmp_path: Path) -> None:
        """pre_snapshot creates a snapshot usable by subsequent record()."""
        f = tmp_path / "target.txt"
        f.write_text("pre-snapshot content", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        # Pre-snapshot (as shell tool would do)
        tracker.pre_snapshot([str(f)])

        # Now modify and record
        f.write_text("post content", encoding="utf-8")
        m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.SHELL, "c1")

        # before_hash comes from the pre-snapshot
        assert m is not None
        assert m.before_hash is not None
        before_bytes = tracker.store.read_blob(m.before_hash)
        assert before_bytes == b"pre-snapshot content"

    def test_pre_snapshot_no_active_turn_is_noop(self, tmp_path: Path) -> None:
        """pre_snapshot is a no-op when no turn is active."""
        tracker = MutationTracker(SnapshotStore(tmp_path))
        # Should not raise
        tracker.pre_snapshot(["/tmp/nonexistent"])

    def test_blob_deduplication(self, tmp_path: Path) -> None:
        """Identical files share the same blob on disk."""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("same content", encoding="utf-8")
        f2.write_text("same content", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        m1 = tracker.record(str(f1), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        m2 = tracker.record(str(f2), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")

        assert m1.before_hash == m2.before_hash
        # Only one blob file should exist for that hash
        blobs = list((tmp_path / "mutations").iterdir())
        assert len(blobs) == 1

    def test_nonexistent_file_snapshot(self, tmp_path: Path) -> None:
        """Recording a mutation on a non-existent file sets existed=False, before_hash=None."""
        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        m = tracker.record(str(tmp_path / "ghost.txt"), MutationOp.CREATE, MutationSource.WRITE_FILE, "c1")
        assert m is not None
        assert m.before_hash is None

        snap = tracker.get_snapshot(str(tmp_path / "ghost.txt"), 1)
        assert snap is not None
        assert snap.existed is False

    def test_binary_file_in_get_file_edit_snapshots(self, tmp_path: Path) -> None:
        """Binary content is decoded without raising."""
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x80\x81\xff\xfe binary stuff")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_bytes(b"\x00\x01\x02")
        tracker.record_after(m)

        snapshots = tracker.get_file_edit_snapshots()
        assert len(snapshots) == 1
        # Should not raise; encoding detector may decode as a legacy
        # encoding (e.g. Windows-1251) or fall back to UTF-8 with replacement.
        assert isinstance(snapshots[0][0], str)
        assert "binary stuff" in snapshots[0][0]

    def test_get_changed_files(self, tmp_path: Path) -> None:
        """get_changed_files returns unique paths in first-seen order."""
        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        (tmp_path / "a.py").write_text("a", encoding="utf-8")
        (tmp_path / "b.py").write_text("b", encoding="utf-8")
        tracker.record(str(tmp_path / "a.py"), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        tracker.record(str(tmp_path / "b.py"), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        tracker.record(str(tmp_path / "a.py"), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c3")

        changed = tracker.get_changed_files()
        assert len(changed) == 2
        # a.py first (seen first), b.py second
        assert changed[0].endswith("a.py")
        assert changed[1].endswith("b.py")

    def test_get_original_snapshot(self, tmp_path: Path) -> None:
        """get_original_snapshot returns the earliest snapshot across turns."""
        f = tmp_path / "evolving.py"
        f.write_text("v0", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("v1", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("v2", encoding="utf-8")
        tracker.record_after(m2)

        snap = tracker.get_original_snapshot(str(f))
        assert snap is not None
        assert snap.turn_id == 1
        content = tracker.store.read_content(snap)
        assert content == b"v0"


# ===========================================================================
# MutationTracker — cleanup unused snapshots
# ===========================================================================


class TestCleanupUnusedSnapshots:
    """cleanup_unused_snapshots removes pre-snapshots with no corresponding mutations."""

    def test_removes_orphan_snapshots_and_blobs(self, tmp_path: Path) -> None:
        """Pre-snapshotted files with no mutations get cleaned up."""
        dirty1 = tmp_path / "dirty1.py"
        dirty2 = tmp_path / "dirty2.py"
        target = tmp_path / "target.py"
        dirty1.write_text("unchanged1", encoding="utf-8")
        dirty2.write_text("unchanged2", encoding="utf-8")
        target.write_text("before", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        # Pre-snapshot all three (as GitDiffCalibrator would)
        tracker.pre_snapshot([str(dirty1), str(dirty2), str(target)])

        # Only target.py actually gets a mutation
        target.write_text("after", encoding="utf-8")
        m = tracker.record(str(target), MutationOp.MODIFY, MutationSource.SHELL, "c1")
        assert m is not None

        # Before cleanup: 3 snapshots exist, blobs for all 3 files
        assert len(tracker.log.snapshots) == 3
        mutations_dir = tmp_path / "mutations"
        blobs_before = {f.name for f in mutations_dir.iterdir()} if mutations_dir.exists() else set()

        removed = tracker.cleanup_unused_snapshots()

        assert removed == 2
        assert len(tracker.log.snapshots) == 1
        # Only blobs referenced by the remaining snapshot + mutation survive
        blobs_after = {f.name for f in mutations_dir.iterdir()}
        assert len(blobs_after) < len(blobs_before)

    def test_no_mutations_all_snapshots_removed(self, tmp_path: Path) -> None:
        """When a turn has pre-snapshots but zero mutations, all snapshots are removed."""
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("aaa", encoding="utf-8")
        f2.write_text("bbb", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        tracker.pre_snapshot([str(f1), str(f2)])

        assert len(tracker.log.snapshots) == 2

        removed = tracker.cleanup_unused_snapshots()

        assert removed == 2
        assert len(tracker.log.snapshots) == 0
        # All blobs should be gone
        mutations_dir = tmp_path / "mutations"
        if mutations_dir.exists():
            assert len(list(mutations_dir.iterdir())) == 0

    def test_noop_when_no_orphans(self, tmp_path: Path) -> None:
        """No removal when every snapshot has a corresponding mutation."""
        f = tmp_path / "test.py"
        f.write_text("content", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        tracker.pre_snapshot([str(f)])
        f.write_text("modified", encoding="utf-8")
        tracker.record(str(f), MutationOp.MODIFY, MutationSource.SHELL, "c1")

        removed = tracker.cleanup_unused_snapshots()
        assert removed == 0
        assert len(tracker.log.snapshots) == 1

    def test_noop_when_no_active_turn(self, tmp_path: Path) -> None:
        """Returns 0 when no turn has been started."""
        tracker = MutationTracker(SnapshotStore(tmp_path))
        assert tracker.cleanup_unused_snapshots() == 0

    def test_shared_blob_preserved_when_referenced_elsewhere(self, tmp_path: Path) -> None:
        """A blob shared between an orphan and a used snapshot is NOT deleted."""
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("same content", encoding="utf-8")
        f2.write_text("same content", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        # Both files have the same content → same blob hash
        tracker.pre_snapshot([str(f1), str(f2)])

        # Only f1 gets a mutation
        f1.write_text("changed", encoding="utf-8")
        tracker.record(str(f1), MutationOp.MODIFY, MutationSource.SHELL, "c1")

        # f2's snapshot is orphaned, but its blob hash == f1's snapshot hash
        removed = tracker.cleanup_unused_snapshots()

        assert removed == 1  # f2 snapshot removed
        # Blob is preserved because f1's snapshot still references it
        mutations_dir = tmp_path / "mutations"
        blobs = list(mutations_dir.iterdir())
        assert len(blobs) >= 1  # shared blob survives

    def test_only_cleans_current_turn(self, tmp_path: Path) -> None:
        """Snapshots from prior turns are not touched."""
        f = tmp_path / "dirty.py"
        f.write_text("dirty", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        # Turn 1: pre-snapshot with a mutation
        tracker.start_turn(1)
        tracker.pre_snapshot([str(f)])
        f.write_text("v1", encoding="utf-8")
        tracker.record(str(f), MutationOp.MODIFY, MutationSource.SHELL, "c1")

        # Turn 2: pre-snapshot without mutation (orphan)
        tracker.start_turn(2)
        tracker.pre_snapshot([str(f)])

        assert len(tracker.log.snapshots) == 2  # one per turn

        removed = tracker.cleanup_unused_snapshots()

        assert removed == 1  # only turn 2's orphan
        # Turn 1's snapshot still exists
        remaining_turn_ids = [s.turn_id for s in tracker.log.snapshots.values()]
        assert 1 in remaining_turn_ids
        assert 2 not in remaining_turn_ids


# ===========================================================================
# MutationTracker — cross-tool scenarios within a turn
# ===========================================================================


class TestMutationTrackerCrossToolScenarios:
    """Scenarios: edit→edit, edit→shell rm, write→shell mv, failed edit."""

    def test_edit_then_edit_incremental_before_hash(self, tmp_path: Path) -> None:
        """Two edits to same file: each mutation's before is the previous after."""
        f = tmp_path / "code.py"
        f.write_text("v0", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("v1", encoding="utf-8")
        tracker.record_after(m1)

        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("v2", encoding="utf-8")
        tracker.record_after(m2)

        m3 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c3")
        f.write_text("v3", encoding="utf-8")
        tracker.record_after(m3)

        # Chain: v0→v1→v2→v3
        assert m1.after_hash == m2.before_hash  # v1
        assert m2.after_hash == m3.before_hash  # v2

        snaps = tracker.get_file_edit_snapshots()
        assert snaps == [("v0", "v1"), ("v1", "v2"), ("v2", "v3")]

    def test_edit_then_shell_delete(self, tmp_path: Path) -> None:
        """Edit a file, then shell-delete it: delete's before = edit's after."""
        f = tmp_path / "doomed.py"
        f.write_text("original", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        # 1. edit_file
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("edited", encoding="utf-8")
        tracker.record_after(m1)

        # 2. shell: rm doomed.py (pre_snapshot, then execute, then record)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        m2 = tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c2")

        # delete's before = the edited content, not the turn-start original
        assert m2.before_hash == m1.after_hash
        # DiffView: only edit_file shows (shell excluded)
        assert tracker.get_file_edit_snapshots() == [("original", "edited")]

        # Turn summary: before=original, after=None (deleted)
        summary = tracker.get_turn_file_summary(1)
        norm = os.path.normpath(os.path.abspath(str(f)))
        assert summary[norm].before is not None  # had content before turn
        assert summary[norm].after is None  # deleted (no record_after for shell)

        # Rollback restores to pre-turn state (original)
        f.write_text("should be overwritten", encoding="utf-8")  # simulate something
        tracker.rollback(1)
        assert f.read_text(encoding="utf-8") == "original"

    def test_write_new_file_then_shell_move(self, tmp_path: Path) -> None:
        """Create file via write_file, then mv via shell."""
        src = tmp_path / "src.py"
        dst = tmp_path / "dst.py"

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        # 1. write_file creates src.py
        m1 = tracker.record(str(src), MutationOp.CREATE, MutationSource.WRITE_FILE, "c1")
        assert m1.before_hash is None  # file didn't exist
        src.write_text("new content", encoding="utf-8")
        tracker.record_after(m1)

        # 2. shell: mv src.py dst.py
        tracker.pre_snapshot([str(src), str(dst)])
        src.rename(dst)
        # Scanner would detect: src=DELETE, dst=CREATE
        m_del = tracker.record(str(src), MutationOp.DELETE, MutationSource.SHELL, "c2")
        m_create = tracker.record(str(dst), MutationOp.CREATE, MutationSource.SHELL, "c2")

        # delete's before = hash of "new content" (from write_file)
        assert m_del.before_hash == m1.after_hash
        # create's before = None (dst didn't exist before the mv)
        assert m_create.before_hash is None

        # DiffView: only write_file shows
        assert tracker.get_file_edit_snapshots() == [("", "new content")]

    def test_failed_edit_then_successful_edit(self, tmp_path: Path) -> None:
        """Failed edit doesn't change file; next edit sees the unchanged state."""
        f = tmp_path / "code.py"
        f.write_text("stable", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        # 1. edit_file that fails (file unchanged on disk)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        # Tool raises but finally block still calls record_after
        # File wasn't modified, so after_hash == before_hash
        tracker.record_after(m1)
        assert m1.before_hash == m1.after_hash

        # 2. Successful edit
        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("fixed", encoding="utf-8")
        tracker.record_after(m2)

        # m2's before = stable (unchanged by the failed edit)
        assert m2.before_hash == m1.after_hash

        snaps = tracker.get_file_edit_snapshots()
        assert len(snaps) == 2
        assert snaps[0] == ("stable", "stable")  # no-op edit
        assert snaps[1] == ("stable", "fixed")  # actual change

    def test_shell_modify_then_edit(self, tmp_path: Path) -> None:
        """Shell modifies file, then edit_file: edit sees post-shell state."""
        f = tmp_path / "config.yaml"
        f.write_text("key: old", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        # 1. shell: sed modifies the file
        tracker.pre_snapshot([str(f)])
        f.write_text("key: shell-modified", encoding="utf-8")
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.SHELL, "c1")
        # m1.before_hash = hash("key: old") via _last_known_hash from pre_snapshot

        # 2. edit_file refines it further
        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("key: final", encoding="utf-8")
        tracker.record_after(m2)

        # edit's before = the shell-modified content, not the turn start
        assert m2.before_hash == m1.after_hash

        snaps = tracker.get_file_edit_snapshots()
        assert len(snaps) == 1
        assert snaps[0] == ("key: shell-modified", "key: final")

    def test_turn_file_summary_collapses_multiple_edits(self, tmp_path: Path) -> None:
        """get_turn_file_summary: before=turn-start, after=last mutation."""
        f = tmp_path / "multi.py"
        f.write_text("v0", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("v1", encoding="utf-8")
        tracker.record_after(m1)

        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("v2", encoding="utf-8")
        tracker.record_after(m2)

        summary = tracker.get_turn_file_summary(1)
        norm = os.path.normpath(os.path.abspath(str(f)))
        diff = summary[norm]

        # before = turn start (v0), after = last mutation result (v2)
        assert diff.before == m1.before_hash
        assert diff.after == m2.after_hash
        assert diff.before != diff.after

    def test_turn_file_summary_new_file_created(self, tmp_path: Path) -> None:
        """get_turn_file_summary: new file has before=None."""
        f = tmp_path / "brand_new.py"

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        m = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c1")
        f.write_text("hello", encoding="utf-8")
        tracker.record_after(m)

        summary = tracker.get_turn_file_summary(1)
        norm = os.path.normpath(os.path.abspath(str(f)))
        assert summary[norm].before is None  # didn't exist before turn
        assert summary[norm].after is not None  # exists after

    def test_rollback_after_multiple_edits_restores_pre_turn_state(self, tmp_path: Path) -> None:
        """Rollback with 3 edits in one turn restores to pre-turn state (not intermediate)."""
        f = tmp_path / "many_edits.py"
        f.write_text("v0", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        for i in range(1, 4):
            m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, f"c{i}")
            f.write_text(f"v{i}", encoding="utf-8")
            tracker.record_after(m)

        assert f.read_text(encoding="utf-8") == "v3"
        tracker.rollback(1)
        assert f.read_text(encoding="utf-8") == "v0"


# ===========================================================================
# MutationTracker — session-wide file summary (ALL-tab aggregation)
# ===========================================================================


class TestSessionFileSummary:
    """get_session_file_summary: net before/after across all turns.

    before = earliest turn's snapshot (pre-session state)
    after = latest turn's last mutation's after_hash
    Net-zero churn (before == after, including both None) is filtered out.
    """

    def _mk(self, tmp_path: Path) -> MutationTracker:
        return MutationTracker(SnapshotStore(tmp_path))

    def _key(self, path: Path) -> str:
        return os.path.normpath(os.path.abspath(str(path)))

    def test_simple_modify_across_turns_keeps_pre_session_before(self, tmp_path: Path) -> None:
        """MODIFY v0→v1 (turn 1) then v1→v2 (turn 2): before=v0, after=v2."""
        f = tmp_path / "code.py"
        f.write_text("v0", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("v1", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("v2", encoding="utf-8")
        tracker.record_after(m2)

        summary = tracker.get_session_file_summary()
        diff = summary[self._key(f)]
        assert diff.before == m1.before_hash  # hash of "v0"
        assert diff.after == m2.after_hash  # hash of "v2"

    def test_create_then_delete_in_later_turn_is_net_zero(self, tmp_path: Path) -> None:
        """File didn't exist → CREATE (turn 1) → DELETE (turn 2): filtered."""
        f = tmp_path / "new.py"
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c1")
        f.write_text("hello", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c2")

        summary = tracker.get_session_file_summary()
        assert self._key(f) not in summary

    def test_create_delete_create_ends_as_create(self, tmp_path: Path) -> None:
        """File didn't exist → CREATE (t1) → DELETE (t2) → CREATE (t3): net CREATE."""
        f = tmp_path / "flaky.py"
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c1")
        f.write_text("first", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c2")

        tracker.start_turn(3)
        m3 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c3")
        f.write_text("second", encoding="utf-8")
        tracker.record_after(m3)

        summary = tracker.get_session_file_summary()
        diff = summary[self._key(f)]
        assert diff.before is None  # pre-session: file didn't exist
        assert diff.after == m3.after_hash  # final content = "second"

    def test_create_delete_create_edit_delete_is_net_zero(self, tmp_path: Path) -> None:
        """User-requested worst case: CREATE→DELETE→CREATE→EDIT→DELETE across turns.

        File never existed pre-session and doesn't exist post-session → net-zero → filtered.
        """
        f = tmp_path / "churn.py"
        tracker = self._mk(tmp_path)

        # Turn 1: CREATE
        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c1")
        f.write_text("a", encoding="utf-8")
        tracker.record_after(m1)

        # Turn 2: DELETE
        tracker.start_turn(2)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c2")

        # Turn 3: CREATE again
        tracker.start_turn(3)
        m3 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c3")
        f.write_text("b", encoding="utf-8")
        tracker.record_after(m3)

        # Turn 4: EDIT
        tracker.start_turn(4)
        m4 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c4")
        f.write_text("b-edited", encoding="utf-8")
        tracker.record_after(m4)

        # Turn 5: DELETE
        tracker.start_turn(5)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c5")

        summary = tracker.get_session_file_summary()
        assert self._key(f) not in summary  # net-zero filtered

    def test_existing_file_modified_then_deleted_then_recreated_same_content(self, tmp_path: Path) -> None:
        """Pre-existing X → MODIFY to Y → DELETE → CREATE with X: net-zero (back to X)."""
        f = tmp_path / "roundtrip.py"
        f.write_text("original", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("modified", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c2")

        tracker.start_turn(3)
        m3 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c3")
        f.write_text("original", encoding="utf-8")  # back to original
        tracker.record_after(m3)

        summary = tracker.get_session_file_summary()
        assert self._key(f) not in summary  # same content hash → filtered

    def test_existing_file_modified_then_deleted_then_recreated_different_content(self, tmp_path: Path) -> None:
        """Pre-existing X → MODIFY to Y → DELETE → CREATE with Z: before=X, after=Z."""
        f = tmp_path / "reshape.py"
        f.write_text("X", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("Y", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c2")

        tracker.start_turn(3)
        m3 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c3")
        f.write_text("Z", encoding="utf-8")
        tracker.record_after(m3)

        summary = tracker.get_session_file_summary()
        diff = summary[self._key(f)]
        assert diff.before == m1.before_hash  # hash("X")
        assert diff.after == m3.after_hash  # hash("Z")
        assert diff.before != diff.after

    def test_modify_bounce_back_same_content_is_net_zero(self, tmp_path: Path) -> None:
        """X → Y → X across turns: before == after → filtered."""
        f = tmp_path / "bounce.py"
        f.write_text("X", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("Y", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("X", encoding="utf-8")
        tracker.record_after(m2)

        assert m1.before_hash == m2.after_hash  # sanity: round-trip to same content
        summary = tracker.get_session_file_summary()
        assert self._key(f) not in summary

    def test_multiple_edits_in_same_turn_across_multiple_turns(self, tmp_path: Path) -> None:
        """Chained edits within each turn: earliest-before, latest-after."""
        f = tmp_path / "chained.py"
        f.write_text("v0", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m_a = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "a")
        f.write_text("v1", encoding="utf-8")
        tracker.record_after(m_a)
        m_b = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "b")
        f.write_text("v2", encoding="utf-8")
        tracker.record_after(m_b)

        tracker.start_turn(2)
        m_c = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c")
        f.write_text("v3", encoding="utf-8")
        tracker.record_after(m_c)
        m_d = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "d")
        f.write_text("v4", encoding="utf-8")
        tracker.record_after(m_d)

        summary = tracker.get_session_file_summary()
        diff = summary[self._key(f)]
        assert diff.before == m_a.before_hash  # v0
        assert diff.after == m_d.after_hash  # v4

    def test_multiple_independent_files_tracked_separately(self, tmp_path: Path) -> None:
        """Several files mutated across turns — each gets its own aggregation."""
        a = tmp_path / "a.py"
        b = tmp_path / "b.py"
        c = tmp_path / "c.py"
        a.write_text("a0", encoding="utf-8")
        b.write_text("b0", encoding="utf-8")
        # c doesn't exist initially
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        # a: MODIFY
        ma = tracker.record(str(a), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        a.write_text("a1", encoding="utf-8")
        tracker.record_after(ma)
        # c: CREATE
        mc = tracker.record(str(c), MutationOp.CREATE, MutationSource.WRITE_FILE, "c1")
        c.write_text("c_content", encoding="utf-8")
        tracker.record_after(mc)

        tracker.start_turn(2)
        # b: DELETE
        tracker.pre_snapshot([str(b)])
        b.unlink()
        tracker.record(str(b), MutationOp.DELETE, MutationSource.SHELL, "c2")
        # a: MODIFY back to a0 (net-zero)
        ma2 = tracker.record(str(a), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        a.write_text("a0", encoding="utf-8")
        tracker.record_after(ma2)

        summary = tracker.get_session_file_summary()

        # a: bounced back → net-zero → not in summary
        assert self._key(a) not in summary
        # b: was X, now deleted → before=hash_X, after=None
        b_diff = summary[self._key(b)]
        assert b_diff.before is not None
        assert b_diff.after is None
        # c: didn't exist, now exists → before=None, after=hash
        c_diff = summary[self._key(c)]
        assert c_diff.before is None
        assert c_diff.after == mc.after_hash

    def test_delete_then_create_same_name_same_content_is_net_zero(self, tmp_path: Path) -> None:
        """Pre-existing X → DELETE (t1) → CREATE with same content X (t2): filtered."""
        f = tmp_path / "recreated.py"
        f.write_text("hello", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c1")

        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c2")
        f.write_text("hello", encoding="utf-8")  # same content as original
        tracker.record_after(m2)

        summary = tracker.get_session_file_summary()
        assert self._key(f) not in summary  # hash(hello) == hash(hello) → filtered

    def test_delete_then_create_same_name_different_content_is_modify(self, tmp_path: Path) -> None:
        """Pre-existing X → DELETE (t1) → CREATE with Y (t2): MODIFY, before=X, after=Y."""
        f = tmp_path / "replaced.py"
        f.write_text("old", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        m1 = tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c1")

        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c2")
        f.write_text("new", encoding="utf-8")
        tracker.record_after(m2)

        summary = tracker.get_session_file_summary()
        diff = summary[self._key(f)]
        assert diff.before == m1.before_hash  # hash("old") — captured by turn-1 snapshot
        assert diff.after == m2.after_hash  # hash("new")
        assert diff.before != diff.after

    def test_two_edits_same_turn_netting_to_original_is_filtered(self, tmp_path: Path) -> None:
        """Single turn: A → B → A. Two edits, no net change → filtered."""
        from chrys.app.tui.screens.diff.screen import _build_turn_entries

        f = tmp_path / "reverted.py"
        f.write_text("A", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("B", encoding="utf-8")
        tracker.record_after(m1)

        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("A", encoding="utf-8")  # reverted
        tracker.record_after(m2)

        assert m1.before_hash == m2.after_hash  # sanity: back to original hash
        summary = tracker.get_session_file_summary()
        assert self._key(f) not in summary
        turn = tracker.get_turn_mutations(1)
        assert turn is not None
        assert _build_turn_entries(tracker, turn, str(tmp_path)) == []

    def test_eol_only_modify_is_kept_as_zero_line_diff_entry(self, tmp_path: Path) -> None:
        """Line-ending-only changes have no line-count delta, but still changed bytes."""
        from chrys.app.tui.screens.diff.screen import _build_session_entries, _build_turn_entries

        f = tmp_path / "Power.yaml"
        f.write_bytes(b"name: Power\n")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_bytes(b"name: Power\r\n")
        tracker.record_after(m)

        turn = tracker.get_turn_mutations(1)
        assert turn is not None
        assert m.before_hash != m.after_hash
        turn_entries = _build_turn_entries(tracker, turn, str(tmp_path))
        session_entries = _build_session_entries(tracker, str(tmp_path))
        assert len(turn_entries) == 1
        assert len(session_entries) == 1
        assert turn_entries[0].before_text == "name: Power\n"
        assert turn_entries[0].after_text == "name: Power\r\n"

    def test_bom_only_modify_is_kept_as_metadata_only_diff_entry(self, tmp_path: Path) -> None:
        """Byte-level encoding changes can decode to identical text and still be real changes."""
        from chrys.app.tui.screens.diff.screen import _build_session_entries, _build_turn_entries

        f = tmp_path / "Power.yaml"
        f.write_bytes(b"\xef\xbb\xbfname: Power\n")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_bytes(b"name: Power\n")
        tracker.record_after(m)

        turn = tracker.get_turn_mutations(1)
        assert turn is not None
        assert m.before_hash != m.after_hash
        turn_entries = _build_turn_entries(tracker, turn, str(tmp_path))
        session_entries = _build_session_entries(tracker, str(tmp_path))
        assert len(turn_entries) == 1
        assert len(session_entries) == 1
        assert turn_entries[0].before_text == "name: Power\n"
        assert turn_entries[0].after_text == "name: Power\n"
        assert turn_entries[0].bytes_changed is True

    def test_two_edits_same_turn_with_net_change_is_modify(self, tmp_path: Path) -> None:
        """Single turn: A → B → C. Two edits, net change → MODIFY (before=A, after=C)."""
        f = tmp_path / "stepwise.py"
        f.write_text("A", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("B", encoding="utf-8")
        tracker.record_after(m1)

        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("C", encoding="utf-8")
        tracker.record_after(m2)

        summary = tracker.get_session_file_summary()
        diff = summary[self._key(f)]
        assert diff.before == m1.before_hash  # hash("A")
        assert diff.after == m2.after_hash  # hash("C")

    def test_two_failed_edits_same_content_after_both_is_filtered(self, tmp_path: Path) -> None:
        """Two edits both leaving file untouched (failed tool calls) → filtered."""
        f = tmp_path / "unchanged.py"
        f.write_text("stable", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        # Tool ran but made no change (e.g. edit_file with replacement == original)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        tracker.record_after(m1)  # file bytes unchanged on disk
        assert m1.before_hash == m1.after_hash

        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        tracker.record_after(m2)
        assert m2.before_hash == m2.after_hash

        summary = tracker.get_session_file_summary()
        assert self._key(f) not in summary

    def test_move_mutation_tracks_both_paths(self, tmp_path: Path) -> None:
        """MOVE with old_path: old_path shows as implicit DELETE in summary."""
        src = tmp_path / "src.py"
        dst = tmp_path / "dst.py"
        src.write_text("moved content", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        # Snapshot BOTH paths before the shell command runs (realistic flow
        # from WorkspaceScanner) — otherwise dst's post-move snapshot would
        # equal its after_hash and collapse to net-zero.
        tracker.pre_snapshot([str(src), str(dst)])
        src.rename(dst)
        # Synthesize a MOVE record (old_path is the only code path that
        # produces FileMutation.old_path).
        m = tracker.record(
            str(dst),
            MutationOp.MOVE,
            MutationSource.SHELL,
            "c1",
            old_path=str(src),
        )
        assert m is not None

        summary = tracker.get_session_file_summary()
        # Destination: didn't exist → now exists with moved content.
        dst_diff = summary[self._key(dst)]
        assert dst_diff.before is None
        assert dst_diff.after is not None
        # Source (old_path): implicit DELETE → before=moved content, after=None.
        src_diff = summary[self._key(src)]
        assert src_diff.before is not None  # captured by pre_snapshot
        assert src_diff.after is None  # implicit delete via MOVE

    def test_turn_diff_keeps_move_when_destination_bytes_are_unchanged(self, tmp_path: Path) -> None:
        """A MOVE remains visible even when the destination hash itself is net-zero."""
        from chrys.app.tui.screens.diff.screen import _build_turn_entries

        src = tmp_path / "src.py"
        dst = tmp_path / "dst.py"
        src.write_text("same content", encoding="utf-8")
        dst.write_text("same content", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        tracker.pre_snapshot([str(src), str(dst)])
        src.replace(dst)
        mutation = tracker.record(str(dst), MutationOp.MOVE, MutationSource.SHELL, "c1", old_path=str(src))
        assert mutation is not None

        turn = tracker.get_turn_mutations(1)
        assert turn is not None
        summary = tracker.get_turn_file_summary(1)
        assert summary[self._key(dst)].is_net_zero

        entries = _build_turn_entries(tracker, turn, str(tmp_path))
        assert len(entries) == 1
        assert entries[0].operation is MutationOp.MOVE
        assert entries[0].old_path == self._key(src)
        assert entries[0].bytes_changed is False

    def test_net_modify_when_existing_file_deleted_then_recreated_different(self, tmp_path: Path) -> None:
        """Regression: existing X → DELETE → CREATE(Y) must surface as MODIFY, not CREATE.

        The last recorded mutation op is CREATE, but the *net* effect on
        a pre-existing file is MODIFY (file existed, still exists with
        different content).  This test guards against the
        last-op-fallback pitfall in _build_session_entries.
        """
        from chrys.app.tui.screens.diff.screen import _build_session_entries

        f = tmp_path / "replaced.py"
        f.write_text("X", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        tracker.pre_snapshot([str(f)])
        f.unlink()
        tracker.record(str(f), MutationOp.DELETE, MutationSource.SHELL, "c1")

        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c2")
        f.write_text("Y", encoding="utf-8")
        tracker.record_after(m2)

        entries = _build_session_entries(tracker, str(tmp_path))
        assert len(entries) == 1
        assert entries[0].operation == MutationOp.MODIFY  # NOT CREATE
        assert entries[0].before_text == "X"
        assert entries[0].after_text == "Y"

    def test_empty_tracker_returns_empty(self, tmp_path: Path) -> None:
        """No turns recorded → empty summary."""
        tracker = self._mk(tmp_path)
        assert tracker.get_session_file_summary() == {}

    def test_empty_turns_return_empty(self, tmp_path: Path) -> None:
        """Turns started but no mutations → empty summary."""
        tracker = self._mk(tmp_path)
        tracker.start_turn(1)
        tracker.start_turn(2)
        assert tracker.get_session_file_summary() == {}

    def test_non_monotonic_turn_order_uses_turn_id_not_insertion(self, tmp_path: Path) -> None:
        """Session restore/retry can leave ``_log.turns`` with non-monotonic
        ``turn_id`` values — the summary must aggregate chronologically by
        ``turn_id``, not by list insertion order.

        Regression: without sorting, the reversed list would take turn 3's
        ``before`` as the session start and turn 1's ``after`` as the final
        state, collapsing to net-zero (hash(v1) == hash(v1)) and dropping
        the entry entirely.
        """
        f = tmp_path / "x.py"
        f.write_text("v0", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("v1", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(3)
        m3 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c3")
        f.write_text("v3", encoding="utf-8")
        tracker.record_after(m3)

        # Simulate restore path where the saved turns deserialize in a
        # different order than their turn_ids (the private reverse() is
        # the simplest faithful reproduction of that state).
        tracker._log.turns.reverse()

        summary = tracker.get_session_file_summary()
        diff = summary[self._key(f)]
        assert diff.before == m1.before_hash  # pre-session = v0
        assert diff.after == m3.after_hash  # final = v3

    def test_survives_serialize_roundtrip(self, tmp_path: Path) -> None:
        """Summary is identical before and after tracker serialize/deserialize."""
        f = tmp_path / "persist.py"
        f.write_text("v0", encoding="utf-8")
        tracker = self._mk(tmp_path)

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("v1", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("v2", encoding="utf-8")
        tracker.record_after(m2)

        before_summary = tracker.get_session_file_summary()
        data = tracker.serialize()
        restored = MutationTracker.deserialize(data, SnapshotStore(tmp_path))
        after_summary = restored.get_session_file_summary()

        assert before_summary == after_summary
        diff = after_summary[self._key(f)]
        assert diff.before == m1.before_hash
        assert diff.after == m2.after_hash


# ===========================================================================
# MutationTracker + WorkspaceScanner — failed shell operations
# ===========================================================================


class TestFailedShellOperations:
    """Failed rm/mv/cp: no false mutations when disk is unchanged."""

    def test_failed_rm_records_nothing(self, tmp_path: Path) -> None:
        """rm fails → file still on disk → diff empty → no mutation recorded."""
        from chrys.service.mutations.scanner import WorkspaceScanner

        f = tmp_path / "protected.py"
        f.write_text("important code", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        # Pre-scan (middleware would do this)
        scanner = WorkspaceScanner(str(tmp_path))
        tracker.pre_snapshot([str(f)])
        before = scanner.scan_paths([str(f)])

        # Shell runs "rm protected.py" but fails (permission denied).
        # File is still on disk — we simulate by doing nothing.

        # Post-scan: detect actual changes
        after = scanner.scan_paths([str(f)])
        changes = WorkspaceScanner.diff(before, after)

        # No changes detected → no mutations recorded
        assert changes == []
        turn = tracker.get_turn_mutations(1)
        assert turn is not None
        assert turn.mutations == []

        # last_known_hash correctly reflects the unchanged file
        norm = os.path.normpath(os.path.abspath(str(f)))
        assert tracker._last_known_hash[norm] is not None

    def test_failed_mv_records_nothing(self, tmp_path: Path) -> None:
        """mv fails → both src and dst unchanged → no mutation recorded."""
        from chrys.service.mutations.scanner import WorkspaceScanner

        src = tmp_path / "src.py"
        src.write_text("source", encoding="utf-8")
        # dst doesn't exist

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        scanner = WorkspaceScanner(str(tmp_path))
        tracker.pre_snapshot([str(src), str(tmp_path / "dst.py")])
        before = scanner.scan_paths([str(src), str(tmp_path / "dst.py")])

        # mv fails → nothing changes on disk

        after = scanner.scan_paths([str(src), str(tmp_path / "dst.py")])
        changes = WorkspaceScanner.diff(before, after)

        assert changes == []
        assert tracker.get_turn_mutations(1).mutations == []

    def test_partial_rm_records_only_deleted_files(self, tmp_path: Path) -> None:
        """rm file1 file2: file1 deleted, file2 permission denied."""
        from chrys.service.mutations.scanner import WorkspaceScanner

        f1 = tmp_path / "file1.txt"
        f2 = tmp_path / "file2.txt"
        f1.write_text("data1", encoding="utf-8")
        f2.write_text("data2", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        scanner = WorkspaceScanner(str(tmp_path))
        targets = [str(f1), str(f2)]
        tracker.pre_snapshot(targets)
        before = scanner.scan_paths(targets)

        # Simulate partial success: only file1 gets deleted
        f1.unlink()

        after = scanner.scan_paths(targets)
        changes = WorkspaceScanner.diff(before, after)

        for path, op in changes:
            tracker.record(path, op, MutationSource.SHELL, "c1")

        # Only file1 DELETE recorded
        turn = tracker.get_turn_mutations(1)
        assert len(turn.mutations) == 1
        assert turn.mutations[0].operation == MutationOp.DELETE
        assert turn.mutations[0].path.endswith("file1.txt")

        # file2 state untouched in last_known_hash
        norm_f2 = os.path.normpath(os.path.abspath(str(f2)))
        assert tracker._last_known_hash[norm_f2] is not None

    def test_failed_shell_then_edit_has_correct_before(self, tmp_path: Path) -> None:
        """Failed rm, then edit_file: edit sees the original (not deleted) state."""
        from chrys.service.mutations.scanner import WorkspaceScanner

        f = tmp_path / "code.py"
        f.write_text("original", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        # Shell rm fails
        scanner = WorkspaceScanner(str(tmp_path))
        tracker.pre_snapshot([str(f)])
        before = scanner.scan_paths([str(f)])
        # rm fails → file unchanged
        after = scanner.scan_paths([str(f)])
        changes = WorkspaceScanner.diff(before, after)
        assert changes == []

        # Now edit_file: should see the original content as before
        m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("edited", encoding="utf-8")
        tracker.record_after(m)

        # before_hash = hash("original"), set by pre_snapshot
        assert m.before_hash is not None
        before_bytes = tracker.store.read_blob(m.before_hash)
        assert before_bytes == b"original"

        snaps = tracker.get_file_edit_snapshots()
        assert snaps == [("original", "edited")]

    def test_interrupted_shell_no_false_state(self, tmp_path: Path) -> None:
        """Shell never runs (interrupted) → diff empty → state clean."""
        from chrys.service.mutations.scanner import WorkspaceScanner

        f = tmp_path / "safe.py"
        f.write_text("untouched", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)

        scanner = WorkspaceScanner(str(tmp_path))
        tracker.pre_snapshot([str(f)])
        before = scanner.scan_paths([str(f)])

        # Interrupted before shell executes → disk unchanged
        after = scanner.scan_paths([str(f)])
        changes = WorkspaceScanner.diff(before, after)

        assert changes == []
        assert tracker.get_turn_mutations(1).mutations == []

        # File is still pristine
        assert f.read_text(encoding="utf-8") == "untouched"


# ===========================================================================
# MutationTracker — rollback and cleanup
# ===========================================================================


class TestMutationTrackerRollback:
    """Rollback, remove_turn, and clear operations."""

    def test_rollback_restores_modified_file(self, tmp_path: Path) -> None:
        """Rolling back one turn restores the file to its pre-turn state."""
        f = tmp_path / "code.py"
        f.write_text("original", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("changed", encoding="utf-8")
        tracker.record_after(m)

        assert f.read_text(encoding="utf-8") == "changed"
        restored = tracker.rollback(1)
        assert restored and restored[0].path == str(f) and restored[0].changed
        assert f.read_text(encoding="utf-8") == "original"

    def test_rollback_deletes_created_file(self, tmp_path: Path) -> None:
        """Rolling back a CREATE operation deletes the file."""
        f = tmp_path / "new_file.py"

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        m = tracker.record(str(f), MutationOp.CREATE, MutationSource.WRITE_FILE, "c1")
        f.write_text("created!", encoding="utf-8")
        tracker.record_after(m)

        assert f.exists()
        restored = tracker.rollback(1)
        assert len(restored) == 1
        assert not f.exists()

    def test_rollback_multi_turn(self, tmp_path: Path) -> None:
        """Rolling back multiple turns restores to the pre-window state."""
        f = tmp_path / "multi.py"
        f.write_text("v0", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))

        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("v1", encoding="utf-8")
        tracker.record_after(m1)

        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("v2", encoding="utf-8")
        tracker.record_after(m2)

        # Roll back 2 turns -> restore to v0
        tracker.rollback(2)
        assert f.read_text(encoding="utf-8") == "v0"
        assert len(tracker.get_all_turns()) == 0

    def test_rollback_explicit_turn_ids_ignores_log_insertion_order(self, tmp_path: Path) -> None:
        """Rolling back by turn ID must not treat the log suffix as the target."""
        f_turn_3 = tmp_path / "turn3.py"
        f_turn_2 = tmp_path / "turn2.py"
        f_turn_3.write_text("v0-3", encoding="utf-8")
        f_turn_2.write_text("v0-2", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))

        tracker.start_turn(1)

        tracker.start_turn(3)
        m3 = tracker.record(str(f_turn_3), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c3")
        f_turn_3.write_text("v3", encoding="utf-8")
        tracker.record_after(m3)

        tracker.start_turn(2)
        m2 = tracker.record(str(f_turn_2), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f_turn_2.write_text("v2", encoding="utf-8")
        tracker.record_after(m2)

        restored = tracker.rollback_turns({3})

        assert [result.path for result in restored] == [str(f_turn_3)]
        assert f_turn_3.read_text(encoding="utf-8") == "v0-3"
        assert f_turn_2.read_text(encoding="utf-8") == "v2"
        assert [turn.turn_id for turn in tracker.get_all_turns()] == [1, 2]

    def test_rollback_empty_turns(self, tmp_path: Path) -> None:
        """Rollback with no turns is a no-op."""
        tracker = MutationTracker(SnapshotStore(tmp_path))
        assert tracker.rollback(1) == []

    def test_remove_turn_cleans_orphaned_blobs(self, tmp_path: Path) -> None:
        """remove_turn deletes blobs not referenced by remaining turns."""
        f = tmp_path / "test.py"
        f.write_text("content-a", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("content-b", encoding="utf-8")
        tracker.record_after(m)

        blob_dir = tmp_path / "mutations"
        assert blob_dir.exists()
        blobs_before = set(blob_dir.iterdir())
        assert len(blobs_before) == 2  # before + after blobs

        orphaned = tracker.remove_turn(1)
        assert len(orphaned) == 2

        # Blobs should be deleted
        remaining = set(blob_dir.iterdir()) if blob_dir.exists() else set()
        assert remaining == set()

    def test_remove_turn_preserves_shared_blobs(self, tmp_path: Path) -> None:
        """remove_turn keeps blobs still referenced by other turns."""
        f = tmp_path / "shared.py"
        f.write_text("shared-content", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))

        # Turn 1: snapshot the file
        tracker.start_turn(1)
        m1 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("v1", encoding="utf-8")
        tracker.record_after(m1)

        # Turn 2: snapshot the file again (now "v1")
        tracker.start_turn(2)
        m2 = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c2")
        f.write_text("v2", encoding="utf-8")
        tracker.record_after(m2)

        # Remove turn 1 — "shared-content" blob is only in turn 1
        # but "v1" blob is referenced as turn-2 snapshot AND as turn-1 after_hash
        orphaned = tracker.remove_turn(1)
        # "shared-content" hash should be orphaned
        assert len(orphaned) >= 1

        # Turn 2 snapshots should still be readable
        snap = tracker.get_file_edit_snapshots()
        assert len(snap) == 1
        assert snap[0][0] == "v1"
        assert snap[0][1] == "v2"

    def test_clear_removes_everything(self, tmp_path: Path) -> None:
        """clear() removes all turns, snapshots, and blob files."""
        f = tmp_path / "test.py"
        f.write_text("content", encoding="utf-8")

        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")
        f.write_text("new", encoding="utf-8")
        tracker.record_after(m)

        blob_dir = tmp_path / "mutations"
        assert blob_dir.exists()

        tracker.clear()
        assert tracker.get_all_turns() == []
        assert tracker.get_changed_files() == []
        assert tracker.get_file_edit_snapshots() == []
        assert not blob_dir.exists()

    def test_get_rollback_plan_caps_at_available_turns(self, tmp_path: Path) -> None:
        """Requesting more turns than available rolls back all turns."""
        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        f = tmp_path / "x.py"
        f.write_text("hi", encoding="utf-8")
        tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, "c1")

        plan = tracker.get_rollback_plan(100)  # way more than 1 turn
        assert len(plan.entries) == 1


# ===========================================================================
# SnapshotStore
# ===========================================================================


class TestSnapshotStore:
    """SnapshotStore blob persistence and restore."""

    def test_save_and_read_content(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")

        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), turn_id=1)
        assert snap.existed is True
        assert snap.content_hash is not None
        assert snap.size == len(b"hello world")

        content = store.read_content(snap)
        assert content == b"hello world"

    def test_save_nonexistent_file(self, tmp_path: Path) -> None:
        store = SnapshotStore(tmp_path)
        snap = store.save(str(tmp_path / "nope.txt"), turn_id=1)
        assert snap.existed is False
        assert snap.content_hash is None
        assert store.read_content(snap) is None

    def test_restore_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "restore_me.txt"
        f.write_text("original", encoding="utf-8")

        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), turn_id=1)

        f.write_text("overwritten", encoding="utf-8")
        assert store.restore(snap).ok
        assert f.read_text(encoding="utf-8") == "original"

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_symlink_snapshot_restores_link_not_file(self, tmp_path: Path) -> None:
        """A symlink is snapshotted as itself (target text, the Git blob
        representation) and restored as a link — never materialized as a
        regular file."""
        dest = tmp_path / "dest.txt"
        dest.write_text("dest content", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(dest)

        store = SnapshotStore(tmp_path / "session")
        snap = store.save(str(link), turn_id=1)
        assert snap.existed is True
        assert snap.is_symlink is True
        assert snap.symlink_target_is_dir is False
        assert store.read_content(snap) == os.fsencode(str(dest))
        assert FileSnapshot.from_dict(snap.to_dict()).is_symlink is True

        other = tmp_path / "other.txt"
        other.write_text("other", encoding="utf-8")
        link.unlink()
        link.symlink_to(other)
        assert store.restore(snap).ok
        assert link.is_symlink()
        assert os.readlink(link) == str(dest)
        assert dest.read_text(encoding="utf-8") == "dest content"

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_symlink_snapshot_replaces_regular_file_and_noops_on_match(self, tmp_path: Path) -> None:
        dest = tmp_path / "dest.txt"
        dest.write_text("dest content", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(dest)
        store = SnapshotStore(tmp_path / "session")
        snap = store.save(str(link), turn_id=1)

        link.unlink()
        link.write_text("plain file", encoding="utf-8")
        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.APPLIED
        assert link.is_symlink()
        assert os.readlink(link) == str(dest)

        assert store.restore(snap).outcome is RestoreOutcome.NOOP

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_dangling_symlink_snapshot_is_an_existing_entry(self, tmp_path: Path) -> None:
        link = tmp_path / "dangling"
        link.symlink_to(tmp_path / "missing.txt")
        store = SnapshotStore(tmp_path / "session")
        snap = store.save(str(link), turn_id=1)
        assert snap.existed is True
        assert snap.is_symlink is True

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_save_blob_hashes_symlink_target_text(self, tmp_path: Path) -> None:
        """The entry's identity is the link: its blob is the target text,
        never the destination's content."""
        dest = tmp_path / "dest.txt"
        dest.write_text("dest content", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(dest)
        store = SnapshotStore(tmp_path / "session")
        result = store.save_blob(str(link))
        assert result.content_hash == SnapshotStore.content_hash(os.fsencode(str(dest)))

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_failed_symlink_restore_preserves_existing_entry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refused link creation (Windows without privilege, filesystem
        without symlink support) reports FAILED with whatever sits at the
        path untouched — the replacement is built at a sibling temp path
        and swapped in only once it exists."""
        dest = tmp_path / "dest.txt"
        dest.write_text("dest content", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(dest)
        store = SnapshotStore(tmp_path / "session")
        snap = store.save(str(link), turn_id=1)

        link.unlink()
        link.write_text("survivor", encoding="utf-8")

        def _refuse(*args: object, **kwargs: object) -> None:
            raise OSError("symlink creation refused")

        monkeypatch.setattr(os, "symlink", _refuse)
        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.FAILED
        assert not link.is_symlink()
        assert link.read_text(encoding="utf-8") == "survivor"
        assert not list(tmp_path.glob(".link.*"))

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_directory_symlink_snapshot_persists_target_type(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Windows needs ``target_is_directory`` to recreate a directory
        link; the snapshot persists the kind and restore forwards it."""
        folder = tmp_path / "folder"
        folder.mkdir()
        link = tmp_path / "dirlink"
        link.symlink_to(folder, target_is_directory=True)
        store = SnapshotStore(tmp_path / "session")
        snap = store.save(str(link), turn_id=1)
        assert snap.is_symlink is True
        assert snap.symlink_target_is_dir is True
        assert FileSnapshot.from_dict(snap.to_dict()).symlink_target_is_dir is True

        link.unlink()
        real_symlink = os.symlink
        forwarded: dict[str, bool] = {}

        def _spy(src: str, dst: str, target_is_directory: bool = False) -> None:
            forwarded["target_is_directory"] = target_is_directory
            real_symlink(src, dst)

        monkeypatch.setattr(os, "symlink", _spy)
        assert store.restore(snap).ok
        assert forwarded["target_is_directory"] is True
        assert link.is_symlink()
        assert os.readlink(link) == str(folder)

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_symlink_restore_verifies_recorded_link_kind(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The kind participates in NOOP only where links are typed:
        POSIX links are untyped, so a matching text is NOOP regardless
        of the recorded kind — rebuilding could not change anything.
        Where links are typed (simulated), a mismatched kind rebuilds,
        a matching or unprovable kind NOOPs, and rebuilding with an
        unprovable kind infers it from what the target resolves to."""
        dest = tmp_path / "dest.txt"
        dest.write_text("dest content", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(dest)
        store = SnapshotStore(tmp_path / "session")
        snap = store.save(str(link), turn_id=1)
        assert snap.symlink_target_is_dir is False

        mismatch = dataclasses.replace(snap, symlink_target_is_dir=True)
        assert store.restore(mismatch).outcome is RestoreOutcome.NOOP

        monkeypatch.setattr(mutation_store, "on_disk_link_kind", lambda path: False)
        result = store.restore(mismatch)
        assert result.outcome is RestoreOutcome.APPLIED
        assert link.is_symlink()
        assert os.readlink(link) == str(dest)
        assert store.restore(snap).outcome is RestoreOutcome.NOOP
        unknown = dataclasses.replace(snap, symlink_target_is_dir=None)
        assert store.restore(unknown).outcome is RestoreOutcome.NOOP
        monkeypatch.undo()

        folder = tmp_path / "folder"
        folder.mkdir()
        dlink = tmp_path / "dlink"
        dlink.symlink_to(folder, target_is_directory=True)
        dsnap = dataclasses.replace(store.save(str(dlink), turn_id=1), symlink_target_is_dir=None)
        dlink.unlink()
        real_symlink = os.symlink
        forwarded: dict[str, bool] = {}

        def _spy(src: str, dst: str, target_is_directory: bool = False) -> None:
            forwarded["target_is_directory"] = target_is_directory
            real_symlink(src, dst)

        monkeypatch.setattr(os, "symlink", _spy)
        assert store.restore(dsnap).ok
        assert forwarded["target_is_directory"] is True

    @pytest.mark.skipif(get_platform().is_windows, reason="symlinks require privileges on Windows")
    def test_symlink_restore_spares_unrelated_sibling_entries(self, tmp_path: Path) -> None:
        """The swap must never claim a name it did not create: a user
        entry that happens to sit at a would-be temporary name survives
        the restore untouched."""
        dest = tmp_path / "dest.txt"
        dest.write_text("dest content", encoding="utf-8")
        link = tmp_path / "link"
        link.symlink_to(dest)
        store = SnapshotStore(tmp_path / "session")
        snap = store.save(str(link), turn_id=1)

        bystander = tmp_path / ".link.chrys-link-tmp"
        bystander.write_text("user data", encoding="utf-8")
        other = tmp_path / "other.txt"
        other.write_text("other", encoding="utf-8")
        link.unlink()
        link.symlink_to(other)

        assert store.restore(snap).outcome is RestoreOutcome.APPLIED
        assert os.readlink(link) == str(dest)
        assert bystander.read_text(encoding="utf-8") == "user data"
        assert not list(tmp_path.glob(".link.*.chrys-link-tmp"))

    def test_restore_deletes_created_file(self, tmp_path: Path) -> None:
        f = tmp_path / "created_later.txt"

        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), turn_id=1)  # existed=False
        assert snap.existed is False

        f.write_text("I was created", encoding="utf-8")
        assert store.restore(snap).ok
        assert not f.exists()

    # -------- Revert edge cases (Case 1/2/3/4 + symlink/dir) --------
    #
    # Contract reminder: revert is unconditional — each selected file
    # is forced to the snapshot's pre-turn state.  The user has
    # already reviewed the target state in the rollback modal's diff
    # view and consented by leaving the path checked.  See
    # ``SnapshotStore.restore`` docstring for the full rationale.

    def test_restore_write_target_already_matches_is_noop(self, tmp_path: Path) -> None:
        """Case: disk already matches the snapshot — no write, NOOP outcome."""
        f = tmp_path / "match.txt"
        f.write_text("original", encoding="utf-8")

        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), turn_id=1)

        # Record mtime; ``restore`` must not touch the file.
        mtime_before = f.stat().st_mtime_ns

        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.NOOP
        assert f.stat().st_mtime_ns == mtime_before
        assert f.read_text(encoding="utf-8") == "original"

    def test_restore_delete_when_already_gone_is_noop(self, tmp_path: Path) -> None:
        """Case 3: delete-revert when the file is already gone → NOOP."""
        f = tmp_path / "ghost.txt"
        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), turn_id=1)  # existed=False
        assert not f.exists()

        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.NOOP

    def test_restore_recreates_missing_file(self, tmp_path: Path) -> None:
        """Case 1: snapshot says restore-content but file is missing → APPLIED.

        User explicitly kept this path checked in the modal, so we
        recreate it.  If they wanted it gone, they'd have de-selected.
        """
        f = tmp_path / "was_here.txt"
        f.write_text("original", encoding="utf-8")
        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), turn_id=1)

        f.unlink()
        assert not f.exists()

        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.APPLIED
        assert f.read_text(encoding="utf-8") == "original"

    def test_restore_overwrites_divergent_content(self, tmp_path: Path) -> None:
        """Case 2: file exists with unrelated content → APPLIED, overwritten.

        Matches the agreed "what the diff view shows IS what you get"
        contract — the user's checkbox IS the consent.
        """
        f = tmp_path / "conflict.txt"
        f.write_text("original", encoding="utf-8")
        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), turn_id=1)

        f.write_text("manual edit", encoding="utf-8")
        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.APPLIED
        assert f.read_text(encoding="utf-8") == "original"

    def test_restore_creates_missing_parent_dirs(self, tmp_path: Path) -> None:
        """Case 4: parent directory is gone → mkdir -p then write → APPLIED."""
        nested = tmp_path / "deep" / "nest" / "file.txt"
        nested.parent.mkdir(parents=True)
        nested.write_text("original", encoding="utf-8")

        store = SnapshotStore(tmp_path)
        snap = store.save(str(nested), turn_id=1)

        # Wipe the whole subtree.
        import shutil

        shutil.rmtree(tmp_path / "deep")
        assert not nested.parent.exists()

        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.APPLIED
        assert nested.read_text(encoding="utf-8") == "original"

    def test_restore_write_replaces_symlink_with_real_file(self, tmp_path: Path) -> None:
        """Symlink at target path must be replaced, not written through.

        Without the unlink-first guard, ``write_bytes`` would follow the
        link and corrupt whatever the user pointed it at.
        """
        real = tmp_path / "real.txt"
        real.write_text("original", encoding="utf-8")
        store = SnapshotStore(tmp_path)
        snap = store.save(str(real), turn_id=1)

        # Manually replace with a symlink pointing to unrelated content.
        unrelated = tmp_path / "unrelated.txt"
        unrelated.write_text("do-not-touch", encoding="utf-8")
        real.unlink()
        os.symlink(unrelated, real)

        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.APPLIED
        # The real file at ``real`` now has the snapshot content...
        assert real.is_file() and not real.is_symlink()
        assert real.read_text(encoding="utf-8") == "original"
        # ...and the unrelated file the symlink used to point at is
        # untouched — the whole point of the guard.
        assert unrelated.read_text(encoding="utf-8") == "do-not-touch"

    def test_restore_delete_removes_symlink_without_following(self, tmp_path: Path) -> None:
        """Delete-revert against a symlink removes the link, not its target."""
        target_path = tmp_path / "link_target.txt"
        target_path.write_text("keep-me", encoding="utf-8")
        symlink_path = tmp_path / "created.txt"
        os.symlink(target_path, symlink_path)

        store = SnapshotStore(tmp_path)
        snap = FileSnapshot(path=str(symlink_path), turn_id=1, existed=False)

        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.APPLIED
        assert not symlink_path.exists()
        assert not symlink_path.is_symlink()
        assert target_path.read_text(encoding="utf-8") == "keep-me"

    def test_restore_refuses_to_rmtree_directory_on_delete(self, tmp_path: Path) -> None:
        """Delete-revert when a directory sits at the path → FAILED, dir kept.

        Rmtree would destroy user data far outside the single path they
        consented to revert.  We fail loudly instead.
        """
        p = tmp_path / "was_file.txt"
        store = SnapshotStore(tmp_path)
        snap = FileSnapshot(path=str(p), turn_id=1, existed=False)

        # User replaced the (formerly-planned) file with a directory.
        p.mkdir()
        (p / "precious.txt").write_text("keep-me", encoding="utf-8")

        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.FAILED
        assert "directory" in result.reason
        assert p.is_dir()
        assert (p / "precious.txt").read_text(encoding="utf-8") == "keep-me"

    def test_restore_refuses_to_overwrite_directory_on_write(self, tmp_path: Path) -> None:
        """Write-revert when a directory sits at the path → FAILED, not wiped."""
        f = tmp_path / "was_file.txt"
        f.write_text("original", encoding="utf-8")
        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), turn_id=1)

        f.unlink()
        f.mkdir()
        (f / "precious.txt").write_text("keep-me", encoding="utf-8")

        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.FAILED
        assert "directory" in result.reason
        assert f.is_dir()
        assert (f / "precious.txt").read_text(encoding="utf-8") == "keep-me"

    def test_restore_failed_when_blob_missing(self, tmp_path: Path) -> None:
        """Write-revert with a missing blob → FAILED with a reason string."""
        f = tmp_path / "missing_blob.txt"
        f.write_text("original", encoding="utf-8")
        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), turn_id=1)
        assert snap.content_hash is not None

        # Delete the blob behind the store's back.
        (tmp_path / "mutations" / snap.content_hash).unlink()

        f.write_text("manual", encoding="utf-8")
        result = store.restore(snap)
        assert result.outcome is RestoreOutcome.FAILED
        assert "blob missing" in result.reason
        # Disk untouched — we didn't try to write empty content.
        assert f.read_text(encoding="utf-8") == "manual"

    def test_rollback_continues_past_one_file_failing(self, tmp_path: Path) -> None:
        """Per-file best-effort: one restore raising must not abort the batch.

        ``SnapshotStore.restore`` is already defensive, but we also want
        the tracker's outer loop to survive a truly unexpected exception
        (e.g. a disk IO bug surfacing as something other than OSError).
        This test swaps in a store whose first ``restore()`` raises, then
        confirms the second file still gets attempted and reported.
        """

        class _FlakyStore(SnapshotStore):
            def __init__(self, base: Path) -> None:
                super().__init__(base)
                self.calls = 0

            def restore(self, snapshot):  # type: ignore[override]
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("simulated unexpected failure")
                return super().restore(snapshot)

        store = _FlakyStore(tmp_path)
        tracker = MutationTracker(store)
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_text("a-orig", encoding="utf-8")
        file_b.write_text("b-orig", encoding="utf-8")
        tracker.start_turn(1)
        mut_a = tracker.record(str(file_a), MutationOp.MODIFY, MutationSource.EDIT_FILE, "ca")
        assert mut_a is not None
        file_a.write_text("a-changed", encoding="utf-8")
        tracker.record_after(mut_a)
        mut_b = tracker.record(str(file_b), MutationOp.MODIFY, MutationSource.EDIT_FILE, "cb")
        assert mut_b is not None
        file_b.write_text("b-changed", encoding="utf-8")
        tracker.record_after(mut_b)

        results = tracker.rollback(1)
        # Both files reported, even though the first one raised mid-flight.
        assert len(results) == 2
        outcomes = {r.path: r.outcome for r in results}
        assert any(o is RestoreOutcome.FAILED for o in outcomes.values())
        # One of them succeeded (the non-flaky one) — disk matches pre-turn.
        applied_paths = [r.path for r in results if r.changed]
        assert applied_paths, "expected at least one file to be restored"

    def test_save_blob_and_read_blob(self, tmp_path: Path) -> None:
        f = tmp_path / "blob.txt"
        f.write_text("blob content", encoding="utf-8")

        store = SnapshotStore(tmp_path)
        result = store.save_blob(str(f))
        assert result.content_hash is not None
        assert result.skip_reason is None

        data = store.read_blob(result.content_hash)
        assert data == b"blob content"

    def test_read_blob_missing_returns_none(self, tmp_path: Path) -> None:
        store = SnapshotStore(tmp_path)
        assert store.read_blob("deadbeef") is None

    def test_remove_blobs(self, tmp_path: Path) -> None:
        f = tmp_path / "rm.txt"
        f.write_text("to remove", encoding="utf-8")

        store = SnapshotStore(tmp_path)
        snap = store.save(str(f), turn_id=1)
        assert snap.content_hash is not None
        assert (tmp_path / "mutations" / snap.content_hash).exists()

        store.remove_blobs({snap.content_hash})
        assert not (tmp_path / "mutations" / snap.content_hash).exists()

    def test_remove_all(self, tmp_path: Path) -> None:
        f = tmp_path / "clean.txt"
        f.write_text("data", encoding="utf-8")

        store = SnapshotStore(tmp_path)
        store.save(str(f), turn_id=1)
        assert (tmp_path / "mutations").exists()

        store.remove_all()
        assert not (tmp_path / "mutations").exists()


# ===========================================================================
# ShellMutationDetector
# ===========================================================================


class TestShellMutationDetector:
    """Heuristic shell command parser tests."""

    def test_rm_detects_delete(self, tmp_path: Path) -> None:
        from chrys.service.mutations.detector import ShellMutationDetector

        targets = ShellMutationDetector.detect_targets("rm foo.txt bar.txt", str(tmp_path))
        assert len(targets) == 2
        assert all(op == MutationOp.DELETE for _, op in targets)

    def test_mv_detects_move(self, tmp_path: Path) -> None:
        from chrys.service.mutations.detector import ShellMutationDetector

        targets = ShellMutationDetector.detect_targets("mv old.txt new.txt", str(tmp_path))
        ops = {op for _, op in targets}
        assert MutationOp.MOVE in ops
        assert MutationOp.DELETE in ops

    def test_cp_detects_create(self, tmp_path: Path) -> None:
        from chrys.service.mutations.detector import ShellMutationDetector

        targets = ShellMutationDetector.detect_targets("cp src.txt dst.txt", str(tmp_path))
        assert len(targets) == 1
        assert targets[0][1] == MutationOp.CREATE

    def test_touch_detects_create(self, tmp_path: Path) -> None:
        from chrys.service.mutations.detector import ShellMutationDetector

        targets = ShellMutationDetector.detect_targets("touch newfile.py", str(tmp_path))
        assert len(targets) == 1
        assert targets[0][1] == MutationOp.CREATE

    def test_redirect_detects_modify(self, tmp_path: Path) -> None:
        from chrys.service.mutations.detector import ShellMutationDetector

        targets = ShellMutationDetector.detect_targets("echo hello > output.txt", str(tmp_path))
        assert any(t[0].endswith("output.txt") for t in targets)
        assert any(op == MutationOp.MODIFY for _, op in targets)

    def test_flags_skipped(self, tmp_path: Path) -> None:
        from chrys.service.mutations.detector import ShellMutationDetector

        targets = ShellMutationDetector.detect_targets("rm -rf dir/", str(tmp_path))
        # -rf should be filtered as a flag, dir/ is the target
        paths = [p for p, _ in targets]
        assert not any(p.endswith("-rf") for p in paths)

    def test_empty_command(self) -> None:
        from chrys.service.mutations.detector import ShellMutationDetector

        assert ShellMutationDetector.detect_targets("") == []

    def test_env_var_prefix_skipped(self, tmp_path: Path) -> None:
        from chrys.service.mutations.detector import ShellMutationDetector

        targets = ShellMutationDetector.detect_targets("FOO=bar touch file.txt", str(tmp_path))
        assert len(targets) == 1
        assert targets[0][1] == MutationOp.CREATE


# ===========================================================================
# WorkspaceScanner
# ===========================================================================


class TestWorkspaceScanner:
    """WorkspaceScanner directory comparison and exclusions."""

    def test_scan_finds_files(self, tmp_path: Path) -> None:
        from chrys.service.mutations.scanner import WorkspaceScanner

        (tmp_path / "a.py").write_text("a", encoding="utf-8")
        (tmp_path / "b.py").write_text("b", encoding="utf-8")

        scanner = WorkspaceScanner(str(tmp_path))
        result = scanner.scan()
        assert len(result) == 2

    def test_scan_excludes_default_dirs(self, tmp_path: Path) -> None:
        from chrys.service.mutations.scanner import WorkspaceScanner

        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("code", encoding="utf-8")
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "lib.js").write_text("lib", encoding="utf-8")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "mod.pyc").write_bytes(b"\x00")

        scanner = WorkspaceScanner(str(tmp_path))
        result = scanner.scan()
        paths = list(result.keys())
        assert len(paths) == 1
        assert paths[0].endswith("main.py")

    def test_scan_paths_targeted(self, tmp_path: Path) -> None:
        from chrys.service.mutations.scanner import WorkspaceScanner

        (tmp_path / "target.py").write_text("t", encoding="utf-8")
        (tmp_path / "other.py").write_text("o", encoding="utf-8")

        scanner = WorkspaceScanner(str(tmp_path))
        result = scanner.scan_paths([str(tmp_path / "target.py")])
        assert len(result) == 1
        assert next(iter(result.keys())).endswith("target.py")

    def test_diff_detects_create_modify_delete(self, tmp_path: Path) -> None:
        from chrys.service.mutations.scanner import FileStat, WorkspaceScanner

        before = {
            "/a.py": FileStat(mtime_ns=100, size=10),
            "/b.py": FileStat(mtime_ns=200, size=20),
            "/c.py": FileStat(mtime_ns=300, size=30),
        }
        after = {
            "/a.py": FileStat(mtime_ns=100, size=10),  # unchanged
            "/b.py": FileStat(mtime_ns=999, size=25),  # modified
            "/d.py": FileStat(mtime_ns=400, size=40),  # created
            # /c.py missing -> deleted
        }

        changes = WorkspaceScanner.diff(before, after)
        change_map = dict(changes)
        assert "/a.py" not in change_map
        assert change_map["/b.py"] == MutationOp.MODIFY
        assert change_map["/d.py"] == MutationOp.CREATE
        assert change_map["/c.py"] == MutationOp.DELETE


# ===========================================================================
# GitignoreFilter
# ===========================================================================


class TestGitignoreFilter:
    """GitignoreFilter pattern matching."""

    def test_directory_only_pattern(self) -> None:
        from chrys.service.mutations.scanner import GitignoreFilter

        gi = GitignoreFilter(["build/"])
        assert gi.is_ignored("build", is_dir=True) is True
        assert gi.is_ignored("build", is_dir=False) is False  # files named "build" not matched

    def test_negation(self) -> None:
        from chrys.service.mutations.scanner import GitignoreFilter

        gi = GitignoreFilter(["*.log", "!important.log"])
        assert gi.is_ignored("debug.log") is True
        assert gi.is_ignored("important.log") is False

    def test_comments_and_blanks(self) -> None:
        from chrys.service.mutations.scanner import GitignoreFilter

        gi = GitignoreFilter(["# comment", "", "*.tmp"])
        assert gi.is_ignored("foo.tmp") is True
        assert gi.is_ignored("foo.py") is False

    def test_path_pattern(self) -> None:
        from chrys.service.mutations.scanner import GitignoreFilter

        gi = GitignoreFilter(["docs/generated"])
        assert gi.is_ignored("docs/generated") is True
        assert gi.is_ignored("generated") is False  # no slash in rel path, pattern has slash


# ===========================================================================
# Middleware — mutation tracking via MutationTracker
# ===========================================================================


class TestMiddlewareMutationTracking:
    """ToolEventMiddleware records mutations via MutationTracker."""

    @pytest.mark.asyncio
    async def test_captures_edit_file_mutation(self, tmp_path: Path) -> None:
        bus = AsyncMock()
        bus.publish = AsyncMock()
        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        mw = ToolEventMiddleware(bus, mutation_tracker=tracker)

        target = tmp_path / "test.py"
        target.write_text("original content", encoding="utf-8")

        ctx = MagicMock()
        ctx.function.name = "edit_file"
        ctx.arguments = {"path": str(target)}
        ctx.result = "Success"

        async def fake_call_next():
            target.write_text("modified content", encoding="utf-8")

        await mw.process(ctx, fake_call_next)

        snapshots = tracker.get_file_edit_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0] == ("original content", "modified content")

    @pytest.mark.asyncio
    async def test_captures_write_file_mutation(self, tmp_path: Path) -> None:
        bus = AsyncMock()
        bus.publish = AsyncMock()
        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        mw = ToolEventMiddleware(bus, mutation_tracker=tracker)

        target = tmp_path / "new.txt"

        ctx = MagicMock()
        ctx.function.name = "write_file"
        ctx.arguments = {"path": str(target)}
        ctx.result = "Success"

        async def fake_call_next():
            target.write_text("new file content", encoding="utf-8")

        await mw.process(ctx, fake_call_next)

        snapshots = tracker.get_file_edit_snapshots()
        assert len(snapshots) == 1
        assert snapshots[0] == ("", "new file content")

    @pytest.mark.asyncio
    async def test_no_mutation_for_read_file(self, tmp_path: Path) -> None:
        bus = AsyncMock()
        bus.publish = AsyncMock()
        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        mw = ToolEventMiddleware(bus, mutation_tracker=tracker)

        ctx = MagicMock()
        ctx.function.name = "read_file"
        ctx.arguments = {"path": "/some/file.py"}
        ctx.result = "file content"

        await mw.process(ctx, AsyncMock())

        assert tracker.get_file_edit_snapshots() == []

    @pytest.mark.asyncio
    async def test_no_mutation_for_non_file_tools(self, tmp_path: Path) -> None:
        bus = AsyncMock()
        bus.publish = AsyncMock()
        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        mw = ToolEventMiddleware(bus, mutation_tracker=tracker)

        for tool_name in ("grep", "glob"):
            ctx = MagicMock()
            ctx.function.name = tool_name
            ctx.arguments = {"pattern": "*.py"}
            ctx.result = "output"
            await mw.process(ctx, AsyncMock())

        assert tracker.get_file_edit_snapshots() == []

    @pytest.mark.asyncio
    async def test_shell_tool_triggers_pre_post_scan(self, tmp_path: Path) -> None:
        """Shell tool calls trigger pre/post scanning and record detected mutations."""
        bus = AsyncMock()
        bus.publish = AsyncMock()
        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        mw = ToolEventMiddleware(bus, mutation_tracker=tracker)

        target = tmp_path / "victim.txt"
        target.write_text("before content", encoding="utf-8")

        ctx = MagicMock()
        ctx.function.name = "zsh"
        ctx.function.chrys_kind = "shell"
        ctx.arguments = {"command": "rm victim.txt", "working_dir": str(tmp_path)}
        ctx.result = "ok"

        async def fake_shell():
            target.unlink()

        await mw.process(ctx, fake_shell)

        # The heuristic scanner should have detected the DELETE
        changed = tracker.get_changed_files()
        assert any(p.endswith("victim.txt") for p in changed)

    @pytest.mark.asyncio
    async def test_shell_tool_no_changes_records_nothing(self, tmp_path: Path) -> None:
        """Shell tool that doesn't modify files records no mutations."""
        bus = AsyncMock()
        bus.publish = AsyncMock()
        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        mw = ToolEventMiddleware(bus, mutation_tracker=tracker)

        ctx = MagicMock()
        ctx.function.name = "bash"
        ctx.function.chrys_kind = "shell"
        ctx.arguments = {"command": "echo hello"}
        ctx.result = "hello"

        await mw.process(ctx, AsyncMock())

        # "echo hello" has no file targets — no mutations
        assert tracker.get_changed_files() == []

    @pytest.mark.asyncio
    async def test_shell_tool_modifies_file(self, tmp_path: Path) -> None:
        """Shell tool modifying a file records the mutation with correct before hash."""
        bus = AsyncMock()
        bus.publish = AsyncMock()
        tracker = MutationTracker(SnapshotStore(tmp_path))
        tracker.start_turn(1)
        mw = ToolEventMiddleware(bus, mutation_tracker=tracker)

        target = tmp_path / "output.txt"
        target.write_text("original", encoding="utf-8")

        ctx = MagicMock()
        ctx.function.name = "bash"
        ctx.function.chrys_kind = "shell"
        ctx.arguments = {"command": "echo 'new' > output.txt", "working_dir": str(tmp_path)}
        ctx.result = "ok"

        async def fake_shell():
            target.write_text("new\n", encoding="utf-8")

        await mw.process(ctx, fake_shell)

        changed = tracker.get_changed_files()
        norm = os.path.normpath(os.path.abspath(str(target)))
        assert norm in changed

    @pytest.mark.asyncio
    async def test_no_tracker_shell_runs_without_error(self, tmp_path: Path) -> None:
        """Shell tools work fine when no mutation tracker is configured."""
        bus = AsyncMock()
        bus.publish = AsyncMock()
        mw = ToolEventMiddleware(bus, mutation_tracker=None)

        ctx = MagicMock()
        ctx.function.name = "bash"
        ctx.function.chrys_kind = "shell"
        ctx.arguments = {"command": "echo hello"}
        ctx.result = "hello"

        # Should not raise
        await mw.process(ctx, AsyncMock())


# ===========================================================================
# ToolGroup.complete_tool — file_snapshot kwarg
# ===========================================================================


# ===========================================================================
# ChatPanel.replay_history — file_snapshots threaded to renderers
# ===========================================================================


@pytest.mark.asyncio
async def test_replay_history_with_file_snapshots() -> None:
    """replay_history passes file_snapshots through to tool renderers."""
    from textual.app import App, ComposeResult

    from chrys.app.tui.theme import TuiVariableDefaultsMixin
    from chrys.app.tui.widgets.chat.panel import ChatPanel

    class PanelApp(TuiVariableDefaultsMixin, App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    messages = [
        {"role": "user", "contents": [{"type": "text", "text": "edit file"}]},
        {
            "role": "assistant",
            "contents": [
                {
                    "type": "function_call",
                    "name": "edit_file",
                    "call_id": "call_FW_123",
                    "arguments": {"path": "/tmp/test.py"},
                }
            ],
        },
        {
            "role": "tool",
            "contents": [{"type": "function_result", "call_id": "call_FW_123", "result": "Success"}],
        },
        {
            "role": "assistant",
            "contents": [{"type": "text", "text": "Done editing."}],
        },
    ]
    # The loader returns ``dict[call_id, list[(before, after)]]`` so duplicate
    # call_ids keep distinct snapshots. Snapshot cursors are independent of
    # adjacent tool-result pairing.
    file_snapshots = {"call_FW_123": [("old code", "new code")]}

    async with PanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(messages, file_snapshots=file_snapshots)
        await pilot.pause()

        # Find the ToolGroup and check the edit_file widget got the snapshot
        from chrys.app.tui.widgets.chat.tool_call import ToolGroup

        groups = list(panel.query(ToolGroup))
        assert len(groups) == 1
        tg = groups[0]
        record = tg._tool_records["call_FW_123#0"]
        assert record.file_snapshot == ("old code", "new code")
        tg.collapsed = False
        await pilot.pause()
        # Replay keys ``_tools`` by a disambiguated ``{call_id}#{idx}`` to
        # tolerate LLMs that reuse call_ids across retries.
        tc = tg._tools.get("call_FW_123#0")
        assert tc is not None
        assert tc._before_content == "old code"
        assert tc._after_content == "new code"


@pytest.mark.asyncio
async def test_replay_history_duplicate_call_id_distinct_snapshots() -> None:
    """Duplicate call_ids (LLM reuse after rejection) keep distinct snapshots.

    Regression for a subtle variant of the ``Tools (3/2)`` bug: when an
    LLM reuses a ``call_id`` across a rejected file edit and its retry,
    the old ``dict[call_id, tuple]`` contract collapsed both snapshots
    into one — causing both widgets to render the same diff.  The fix
    buckets snapshots by call_id and consumes that list as a file-tool
    snapshot cursor so each widget gets its own.
    """
    from textual.app import App, ComposeResult

    from chrys.app.tui.theme import TuiVariableDefaultsMixin
    from chrys.app.tui.widgets.chat.panel import ChatPanel
    from chrys.app.tui.widgets.chat.tool_call import ToolGroup

    class PanelApp(TuiVariableDefaultsMixin, App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    messages = [
        {"role": "user", "contents": [{"type": "text", "text": "edit twice"}]},
        {
            "role": "assistant",
            "contents": [
                {
                    "type": "function_call",
                    "name": "edit_file",
                    "call_id": "shared_id",
                    "arguments": {"path": "/tmp/a.py"},
                }
            ],
        },
        {
            "role": "tool",
            "contents": [{"type": "function_result", "call_id": "shared_id", "result": "rejected"}],
        },
        {
            "role": "assistant",
            "contents": [
                {
                    "type": "function_call",
                    "name": "edit_file",
                    "call_id": "shared_id",
                    "arguments": {"path": "/tmp/a.py"},
                }
            ],
        },
        {
            "role": "tool",
            "contents": [{"type": "function_result", "call_id": "shared_id", "result": "ok"}],
        },
    ]
    file_snapshots = {
        "shared_id": [("rejected_before", ""), ("success_before", "success_after")],
    }

    async with PanelApp().run_test() as pilot:
        panel = pilot.app.query_one(ChatPanel)
        await panel.replay_history(messages, file_snapshots=file_snapshots)
        await pilot.pause()

        groups = list(panel.query(ToolGroup))
        assert len(groups) == 1
        tg = groups[0]
        # Both invocations got their own slot via the ``{call_id}#{idx}``
        # disambiguator, and each got its own positional snapshot.
        assert tg._tool_records["shared_id#0"].file_snapshot == ("rejected_before", "")
        assert tg._tool_records["shared_id#1"].file_snapshot == ("success_before", "success_after")
        tg.collapsed = False
        await pilot.pause()
        first = tg._tools.get("shared_id#0")
        second = tg._tools.get("shared_id#1")
        assert first is not None
        assert second is not None
        assert first._before_content == "rejected_before"
        assert first._after_content == ""
        assert second._before_content == "success_before"
        assert second._after_content == "success_after"
