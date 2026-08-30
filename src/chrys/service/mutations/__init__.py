# Copyright (c) 2026 Chrys. All rights reserved.

"""File mutation tracking for diff and rollback support.

Tracks all file system changes made by agent tools (write_file, edit_file,
shell commands) and stores original file snapshots in the session directory
for potential rollback.

**Concepts**

- **Snapshot**: A copy of a file's content taken *before* the first mutation
  within a given turn.  Content-addressable blobs stored under the session
  directory.  A file is snapshotted at most once per turn.  Blob writes are
  gated by a ``SnapshotPolicy`` — oversized (default > 50 MiB) or binary
  file contents are not backed up (the mutation record remains, marked
  with a ``SnapshotSkipReason``; no diff rendering / rollback for it).
- **Mutation**: A record of a single file operation (create / modify / delete
  / move) linked to the originating tool call and turn.
- **Turn**: Mutations are grouped by agent turn (matching ``turn_counter``
  from ``chrys_history``).  Every turn is recorded — even those with zero
  mutations — to support multi-step rollback.

**Storage layout**::

    {session_dir}/
    ├── session.json                  # gains "chrys_mutations" key
    └── mutations/
        └── {sha256_hex}             # content-addressable file blobs

**Integration points** (future implementation):

1. **Engine / Executor**: ``tracker.start_turn(turn_id)`` at the
   start of each ``agent.run()`` cycle.
2. **filesystem.py**: ``tracker.record(path, op, source, call_id)``
   BEFORE performing write_file / edit_file.
3. **shell.py**: ``WorkspaceScanner`` before/after execution, then
   ``tracker.record()`` for each detected change.
4. **SessionPersistence**: ``tracker.serialize()`` / ``deserialize()``
   into ``state["chrys_mutations"]``.
5. **Session deletion**: ``tracker.clear()`` to remove all blobs.
"""

from chrys.service.mutations.detector import ShellMutationDetector
from chrys.service.mutations.git_calibrator import GitDiffCalibrator, ImplicitChange
from chrys.service.mutations.scanner import DEFAULT_EXCLUDES, FileStat, GitignoreFilter, WorkspaceScanner
from chrys.service.mutations.store import BlobSaveResult, SnapshotPolicy, SnapshotStore
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.types import (
    FileMutation,
    FileSnapshot,
    MutationLog,
    MutationOp,
    MutationSource,
    SnapshotSkipReason,
    TurnMutations,
)

__all__ = [
    "DEFAULT_EXCLUDES",
    "BlobSaveResult",
    "FileMutation",
    "FileSnapshot",
    "FileStat",
    "GitDiffCalibrator",
    "GitignoreFilter",
    "ImplicitChange",
    "MutationLog",
    "MutationOp",
    "MutationSource",
    "MutationTracker",
    "ShellMutationDetector",
    "SnapshotPolicy",
    "SnapshotSkipReason",
    "SnapshotStore",
    "TurnMutations",
    "WorkspaceScanner",
]
