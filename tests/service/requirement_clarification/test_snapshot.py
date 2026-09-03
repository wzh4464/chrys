# Copyright (c) 2026 Chrys. All rights reserved.

"""Workspace snapshot isolation and restoration tests."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from chrys.foundation.models.workspace import Workspace
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshotError, WorkspaceSnapshotter


def test_snapshot_hides_binary_and_restores_exact_working_set(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    source = workspace_root / "source.py"
    binary = workspace_root / "asset.bin"
    ignored = workspace_root / "node_modules" / "package.js"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    binary.write_bytes(b"\x00\x01\x02")
    ignored.parent.mkdir()
    ignored.write_text("ignored", encoding="utf-8")
    snapshotter = WorkspaceSnapshotter(max_total_bytes=1024 * 1024)

    snapshot = snapshotter.capture(
        Workspace.from_cwd(str(workspace_root)),
        tmp_path / "artifact",
        snapshot_id="s0",
        include_git_history=False,
    )

    entries = {entry.relative_path: entry for entry in snapshot.roots[0].entries}
    assert entries["source.py"].model_visible is True
    assert entries["asset.bin"].model_visible is False
    assert entries["asset.bin"].metadata_reason == "binary"
    assert "node_modules/package.js" not in entries
    assert (Path(snapshot.roots[0].view_root) / "source.py").is_file()
    assert not (Path(snapshot.roots[0].view_root) / "asset.bin").exists()

    source.write_text("VALUE = 2\n", encoding="utf-8")
    binary.unlink()
    (workspace_root / "created.py").write_text("new", encoding="utf-8")
    ignored.write_text("still outside rollback scope", encoding="utf-8")
    snapshotter.restore(snapshot)

    assert source.read_text(encoding="utf-8") == "VALUE = 1\n"
    assert binary.read_bytes() == b"\x00\x01\x02"
    assert not (workspace_root / "created.py").exists()
    assert ignored.read_text(encoding="utf-8") == "still outside rollback scope"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unsupported")
def test_snapshot_does_not_materialize_external_symlink(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (workspace_root / "external").symlink_to(outside)

    snapshot = WorkspaceSnapshotter().capture(
        Workspace.from_cwd(str(workspace_root)),
        tmp_path / "artifact",
        snapshot_id="s0",
        include_git_history=False,
    )

    entry = snapshot.roots[0].entries[0]
    assert entry.kind == "symlink"
    assert entry.model_visible is False
    assert not (Path(snapshot.roots[0].view_root) / "external").exists()


def test_snapshot_fails_atomically_at_total_size_limit(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "large.txt").write_text("x" * 20, encoding="utf-8")
    artifact = tmp_path / "artifact"

    with pytest.raises(WorkspaceSnapshotError, match="exceeds"):
        WorkspaceSnapshotter(max_total_bytes=10).capture(
            Workspace.from_cwd(str(workspace_root)),
            artifact,
            snapshot_id="s0",
            include_git_history=False,
        )

    assert not artifact.exists()


def test_snapshot_artifact_inside_workspace_is_outside_capture_and_restore_scope(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    artifact = workspace_root / ".chrys" / "snapshots" / "s0"

    snapshotter = WorkspaceSnapshotter()
    snapshot = snapshotter.capture(
        Workspace.from_cwd(str(workspace_root)),
        artifact,
        snapshot_id="s0",
        include_git_history=False,
    )
    assert all(not entry.relative_path.startswith(".chrys") for entry in snapshot.roots[0].entries)

    (workspace_root / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    snapshotter.restore(snapshot)

    assert artifact.is_dir()
    assert (workspace_root / "source.py").read_text(encoding="utf-8") == "VALUE = 1\n"


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_git_snapshot_freezes_head_and_s0_worktree(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace_root)], check=True)
    subprocess.run(["git", "-C", str(workspace_root), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(workspace_root), "config", "user.name", "Test"], check=True)
    source = workspace_root / "source.py"
    source.write_text("COMMITTED = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(workspace_root), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(workspace_root), "commit", "-qm", "initial"], check=True)
    source.write_text("S0_WORKTREE = 2\n", encoding="utf-8")

    snapshot = WorkspaceSnapshotter().capture(
        Workspace.from_cwd(str(workspace_root)),
        tmp_path / "artifact",
        snapshot_id="s0",
        include_git_history=True,
    )
    frozen_root = snapshot.roots[0]
    source.write_text("P0_WORKTREE = 3\n", encoding="utf-8")

    assert frozen_root.git_head
    assert Path(frozen_root.history_bundle).is_file()
    assert (Path(frozen_root.view_root) / "source.py").read_text(encoding="utf-8") == "S0_WORKTREE = 2\n"
    frozen_head = subprocess.run(
        ["git", "-C", frozen_root.view_root, "show", f"{frozen_root.git_head}:source.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert frozen_head.stdout == "COMMITTED = 1\n"


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_committed_head_snapshot_excludes_imported_p0_worktree(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace_root)], check=True)
    subprocess.run(["git", "-C", str(workspace_root), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(workspace_root), "config", "user.name", "Test"], check=True)
    source = workspace_root / "source.py"
    source.write_text("S0 = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(workspace_root), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(workspace_root), "commit", "-qm", "initial"], check=True)
    source.write_text("P0 = 2\n", encoding="utf-8")
    (workspace_root / "p0-only.py").write_text("P0_ONLY = True\n", encoding="utf-8")

    snapshot = WorkspaceSnapshotter().capture(
        Workspace.from_cwd(str(workspace_root)),
        tmp_path / "artifact",
        snapshot_id="s0",
        include_git_history=True,
        committed_git_head_only=True,
    )
    view = Path(snapshot.roots[0].view_root)

    assert source.read_text(encoding="utf-8") == "P0 = 2\n"
    assert (view / "source.py").read_text(encoding="utf-8") == "S0 = 1\n"
    assert not (view / "p0-only.py").exists()


def test_snapshot_freezes_and_restores_explicit_reference_file(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    reference = tmp_path / "contract.md"
    reference.write_text("version: 1\n", encoding="utf-8")
    workspace = Workspace.from_cwd(str(workspace_root))
    workspace.reference_files.append(str(reference))

    snapshotter = WorkspaceSnapshotter()
    snapshot = snapshotter.capture(
        workspace,
        tmp_path / "artifact",
        snapshot_id="s0",
        include_git_history=False,
    )
    frozen = snapshot.references[0]
    assert frozen.managed_by_root is False
    assert Path(frozen.view_path).read_text(encoding="utf-8") == "version: 1\n"
    assert snapshot.clarification_workspace().reference_files == [frozen.view_path]

    reference.write_text("version: 2\n", encoding="utf-8")
    assert snapshotter.matches(snapshot) is False
    snapshotter.restore(snapshot)
    assert reference.read_text(encoding="utf-8") == "version: 1\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory modes")
def test_restore_leaves_live_workspace_directory_modes_alone(tmp_path: Path) -> None:
    """A P0 rollback must not tighten the user's own repository to 0700.

    Snapshot entries record file and symlink modes only, so a directory mode
    changed during restore is never captured, never restored, and invisible to
    ``matches()`` and to ``git status``.
    """
    workspace_root = tmp_path / "workspace"
    nested = workspace_root / "scripts"
    nested.mkdir(parents=True)
    (nested / "build.sh").write_text("echo build\n", encoding="utf-8")
    workspace_root.chmod(0o755)
    nested.chmod(0o755)
    snapshotter = WorkspaceSnapshotter()

    snapshot = snapshotter.capture(
        Workspace.from_cwd(str(workspace_root)),
        tmp_path / "artifact",
        snapshot_id="s0",
        include_git_history=False,
    )
    (nested / "build.sh").write_text("echo repaired\n", encoding="utf-8")
    snapshotter.restore(snapshot)

    assert (nested / "build.sh").read_text(encoding="utf-8") == "echo build\n"
    assert stat.S_IMODE(nested.stat().st_mode) == 0o755
    assert stat.S_IMODE(workspace_root.stat().st_mode) == 0o755


def test_restore_uses_the_snapshot_ignore_rules_not_the_repair_own(tmp_path: Path) -> None:
    """Scanning first asked the repair which of its files should survive rollback.

    A repair that hides its output behind a new `.gitignore` rule kept the
    file, and the `matches()` that follows then reported a conflict over the
    workspace restore had just claimed to put back exactly.
    """
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (workspace_root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    snapshotter = WorkspaceSnapshotter(max_total_bytes=1024 * 1024)

    snapshot = snapshotter.capture(
        Workspace.from_cwd(str(workspace_root)),
        tmp_path / "artifact",
        snapshot_id="s0",
        include_git_history=False,
    )

    # The repair creates a file and then teaches the repository to ignore it.
    # `generated/` rather than `build/`: the latter is excluded from capture
    # outright, so it would be out of rollback scope for a different reason.
    generated = workspace_root / "generated"
    generated.mkdir()
    (generated / "output.txt").write_text("repair output", encoding="utf-8")
    (workspace_root / ".gitignore").write_text("*.log\ngenerated/\n", encoding="utf-8")

    snapshotter.restore(snapshot)

    assert not (generated / "output.txt").exists()
    assert (workspace_root / ".gitignore").read_text(encoding="utf-8") == "*.log\n"
    assert snapshotter.matches(snapshot) is True


def test_restore_prunes_only_the_directories_it_emptied(tmp_path: Path) -> None:
    """Walking for empty directories could not tell whose they were.

    An empty directory is never in the snapshot — `_iter_entries` captures
    files and symlinks — so a pre-existing one looked exactly like one the
    repair left behind, and the same walk descended into excluded trees and
    removed leaves like `.git/refs/tags`.
    """
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    (workspace_root / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    pre_existing = workspace_root / "scratch"
    pre_existing.mkdir()
    git_leaf = workspace_root / ".git" / "refs" / "tags"
    git_leaf.mkdir(parents=True)
    snapshotter = WorkspaceSnapshotter(max_total_bytes=1024 * 1024)

    snapshot = snapshotter.capture(
        Workspace.from_cwd(str(workspace_root)),
        tmp_path / "artifact",
        snapshot_id="s0",
        include_git_history=False,
    )

    repair_dir = workspace_root / "repair_output"
    repair_dir.mkdir()
    (repair_dir / "notes.txt").write_text("scratch", encoding="utf-8")

    snapshotter.restore(snapshot)

    assert not repair_dir.exists()
    assert pre_existing.is_dir()
    assert git_leaf.is_dir()
