# Copyright (c) 2026 Chrys. All rights reserved.

"""Immutable, owner-only workspace snapshots for clarification and repair."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import shutil
import stat
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Literal

from chrys.foundation.models.workspace import WorkingDir, Workspace
from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import atomic_write_owner_only_bytes, atomic_write_owner_only_text
from chrys.foundation.text.encoding import EncodingDetector
from chrys.foundation.trajectory.keys import ensure_owner_only_directory
from chrys.service.mutations.scanner import DEFAULT_EXCLUDES, GitignoreFilter

SNAPSHOT_MAX_ENTRIES_PER_ROOT: Final[int] = 50_000
SNAPSHOT_MAX_FILE_BYTES_FOR_MODEL: Final[int] = 50 * 1024 * 1024
SNAPSHOT_MAX_TOTAL_BYTES: Final[int] = 512 * 1024 * 1024
SNAPSHOT_SAMPLE_BYTES: Final[int] = 8192
SNAPSHOT_MANIFEST_NAME: Final[str] = "manifest.json"
SNAPSHOT_VIEW_MANIFEST_NAME: Final[str] = ".chrys_snapshot_manifest.json"

_PATTERN_EXCLUDES = (".egg-info",)


class WorkspaceSnapshotError(OSError):
    """Raised when a complete, stable snapshot cannot be produced or restored."""


@dataclass(frozen=True, slots=True)
class SnapshotEntry:
    """One source entry captured in the content-addressed recovery layer."""

    relative_path_b64: str
    kind: Literal["file", "symlink"]
    mode: int
    size: int
    content_hash: str
    model_visible: bool
    metadata_reason: str = ""
    symlink_target_is_dir: bool = False

    @property
    def relative_path(self) -> str:
        return os.fsdecode(base64.b64decode(self.relative_path_b64.encode("ascii")))


@dataclass(frozen=True, slots=True)
class SnapshotRoot:
    """Mapping between one live workspace root and its frozen model view."""

    source_root: str
    view_root: str
    label: str
    is_primary: bool
    entries: tuple[SnapshotEntry, ...]
    git_head: str = ""
    history_bundle: str = ""


@dataclass(frozen=True, slots=True)
class SnapshotReference:
    """One explicitly configured reference file and its frozen view."""

    source_path: str
    view_path: str
    entry: SnapshotEntry
    managed_by_root: bool


@dataclass(frozen=True, slots=True)
class WorkspaceSnapshot:
    """A complete recovery snapshot plus a bounded clarification view."""

    snapshot_id: str
    artifact_root: str
    roots: tuple[SnapshotRoot, ...]
    manifest_hash: str
    total_bytes: int
    entry_count: int
    references: tuple[SnapshotReference, ...] = ()

    def clarification_workspace(self) -> Workspace:
        if not self.roots:
            raise WorkspaceSnapshotError("snapshot has no workspace roots")
        working_dirs = [
            WorkingDir(path=root.view_root, label=root.label, is_primary=root.is_primary) for root in self.roots
        ]
        primary = next((root.view_root for root in self.roots if root.is_primary), self.roots[0].view_root)
        return Workspace(
            primary_cwd=primary,
            working_dirs=working_dirs,
            reference_files=[reference.view_path for reference in self.references if reference.entry.model_visible],
        )


def _encoded_relative_path(path: str) -> str:
    return base64.b64encode(os.fsencode(path)).decode("ascii")


def _is_excluded_name(name: str) -> bool:
    return name in DEFAULT_EXCLUDES or any(name.endswith(suffix) for suffix in _PATTERN_EXCLUDES)


def _stable_file_bytes(path: Path, expected: os.stat_result) -> bytes:
    try:
        with path.open("rb") as file:
            data = file.read()
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise WorkspaceSnapshotError(f"failed to capture {path}: {exc}") from exc
    before_identity = (expected.st_dev, expected.st_ino, expected.st_size, expected.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity or len(data) != expected.st_size:
        raise WorkspaceSnapshotError(f"workspace entry changed while being snapshotted: {path}")
    return data


def _write_blob(blob_root: Path, digest: str, data: bytes) -> None:
    path = blob_root / digest
    if path.exists():
        return
    atomic_write_owner_only_bytes(path, data)


def _safe_view_destination(view_root: Path, relative_path: str) -> Path:
    destination = view_root / relative_path
    root_real = os.path.realpath(view_root)
    parent_real = os.path.realpath(destination.parent)
    try:
        common = os.path.commonpath((root_real, parent_real))
    except ValueError as exc:
        raise WorkspaceSnapshotError(f"snapshot path escapes its root: {relative_path}") from exc
    if os.path.normcase(common) != os.path.normcase(root_real):
        raise WorkspaceSnapshotError(f"snapshot path escapes its root: {relative_path}")
    return destination


def _materialize_model_entry(
    *,
    view_root: Path,
    relative_path: str,
    data: bytes,
    mode: int,
) -> None:
    destination = _safe_view_destination(view_root, relative_path)
    ensure_owner_only_directory(destination.parent)
    atomic_write_owner_only_bytes(destination, data)
    if not get_platform().is_windows:
        readable_mode = 0o500 if mode & 0o111 else 0o400
        destination.chmod(readable_mode)


def _git_output(source_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    git = shutil.which("git")
    if git is None:
        raise WorkspaceSnapshotError("git executable is unavailable")
    return subprocess.run(  # noqa: S603 - executable and arguments are code-owned
        [git, "-C", str(source_root), *args],
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=60,
    )


def _freeze_git_history(source_root: Path, view_root: Path, root_artifact: Path) -> tuple[str, str, int]:
    inside = _git_output(source_root, "rev-parse", "--is-inside-work-tree")
    if inside.returncode or inside.stdout.strip() != "true":
        ensure_owner_only_directory(view_root)
        return "", "", 0
    head_result = _git_output(source_root, "rev-parse", "--verify", "HEAD^{commit}")
    if head_result.returncode:
        raise WorkspaceSnapshotError(f"failed to resolve frozen HEAD for {source_root}: {head_result.stderr.strip()}")
    head = head_result.stdout.strip()
    bundle = root_artifact / "history.bundle"
    bundle_result = _git_output(source_root, "bundle", "create", str(bundle), "HEAD")
    if bundle_result.returncode:
        raise WorkspaceSnapshotError(
            f"failed to freeze HEAD ancestors for {source_root}: {bundle_result.stderr.strip()}"
        )
    bundle_size = bundle.stat().st_size
    git = shutil.which("git")
    if git is None:
        raise WorkspaceSnapshotError("git executable is unavailable")
    clone = subprocess.run(  # noqa: S603 - executable and arguments are code-owned
        [git, "clone", "--no-checkout", "--quiet", str(bundle), str(view_root)],
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=120,
    )
    if clone.returncode:
        raise WorkspaceSnapshotError(f"failed to materialize frozen Git history: {clone.stderr.strip()}")
    remove_remote = _git_output(view_root, "remote", "remove", "origin")
    if remove_remote.returncode:
        raise WorkspaceSnapshotError(f"failed to seal frozen Git view: {remove_remote.stderr.strip()}")
    return head, str(bundle), bundle_size


class WorkspaceSnapshotter:
    """Capture and restore deterministic workspace checkpoints."""

    def __init__(
        self,
        *,
        max_entries_per_root: int = SNAPSHOT_MAX_ENTRIES_PER_ROOT,
        max_model_file_bytes: int = SNAPSHOT_MAX_FILE_BYTES_FOR_MODEL,
        max_total_bytes: int = SNAPSHOT_MAX_TOTAL_BYTES,
    ) -> None:
        self._max_entries_per_root = max_entries_per_root
        self._max_model_file_bytes = max_model_file_bytes
        self._max_total_bytes = max_total_bytes

    def capture(
        self,
        workspace: Workspace,
        artifact_root: Path,
        *,
        snapshot_id: str,
        include_git_history: bool,
    ) -> WorkspaceSnapshot:
        """Capture every in-scope workspace entry or fail without a partial result."""
        if artifact_root.exists():
            raise WorkspaceSnapshotError(f"snapshot artifact already exists: {artifact_root}")
        ensure_owner_only_directory(artifact_root)
        blob_root = artifact_root / "blobs"
        roots_root = artifact_root / "roots"
        ensure_owner_only_directory(blob_root)
        ensure_owner_only_directory(roots_root)
        total_bytes = 0
        total_entries = 0
        snapshots: list[SnapshotRoot] = []
        references: list[SnapshotReference] = []
        try:
            roots = self._workspace_roots(workspace)
            for index, working_dir in enumerate(roots):
                source_root = Path(working_dir.path).resolve()
                if not source_root.is_dir():
                    raise WorkspaceSnapshotError(f"workspace root is unavailable: {source_root}")
                root_artifact = roots_root / str(index)
                view_root = root_artifact / "view"
                ensure_owner_only_directory(root_artifact)
                git_head = ""
                bundle_path = ""
                if include_git_history:
                    git_head, bundle_path, history_bytes = _freeze_git_history(source_root, view_root, root_artifact)
                    total_bytes += history_bytes
                    self._check_total_bytes(total_bytes)
                else:
                    ensure_owner_only_directory(view_root)
                entries, root_bytes = self._capture_root(
                    source_root,
                    view_root,
                    blob_root,
                    excluded_roots=(artifact_root,),
                )
                total_bytes += root_bytes
                total_entries += len(entries)
                self._check_total_bytes(total_bytes)
                snapshots.append(
                    SnapshotRoot(
                        source_root=str(source_root),
                        view_root=str(view_root),
                        label=working_dir.label,
                        is_primary=working_dir.is_primary,
                        entries=tuple(entries),
                        git_head=git_head,
                        history_bundle=bundle_path,
                    )
                )
                view_manifest = {
                    "snapshot_id": snapshot_id,
                    "source_root_hash": hashlib.sha256(os.fsencode(str(source_root))).hexdigest(),
                    "git_head": git_head,
                    "entries": [
                        {
                            "path": entry.relative_path,
                            "kind": entry.kind,
                            "size": entry.size,
                            "model_visible": entry.model_visible,
                            "metadata_reason": entry.metadata_reason,
                        }
                        for entry in entries
                    ],
                }
                atomic_write_owner_only_text(
                    view_root / SNAPSHOT_VIEW_MANIFEST_NAME,
                    json.dumps(view_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                )
            references, reference_bytes = self._capture_references(
                workspace.reference_files,
                snapshots,
                roots_root,
                blob_root,
            )
            total_bytes += reference_bytes
            total_entries += sum(not reference.managed_by_root for reference in references)
            self._check_total_bytes(total_bytes)
            payload = {
                "snapshot_id": snapshot_id,
                "total_bytes": total_bytes,
                "entry_count": total_entries,
                "roots": [asdict(root) for root in snapshots],
                "references": [asdict(reference) for reference in references],
            }
            encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            manifest_hash = hashlib.sha256(encoded).hexdigest()
            atomic_write_owner_only_bytes(artifact_root / SNAPSHOT_MANIFEST_NAME, encoded + b"\n")
            return WorkspaceSnapshot(
                snapshot_id=snapshot_id,
                artifact_root=str(artifact_root),
                roots=tuple(snapshots),
                manifest_hash=manifest_hash,
                total_bytes=total_bytes,
                entry_count=total_entries,
                references=tuple(references),
            )
        except BaseException:
            shutil.rmtree(artifact_root, ignore_errors=True)
            raise

    def restore(self, snapshot: WorkspaceSnapshot) -> None:
        """Restore the snapshot's in-scope working set exactly."""
        blob_root = Path(snapshot.artifact_root) / "blobs"
        for root in snapshot.roots:
            source_root = Path(root.source_root)
            if not source_root.is_dir():
                raise WorkspaceSnapshotError(f"workspace root disappeared before restore: {source_root}")
            expected = {entry.relative_path: entry for entry in root.entries}
            current = {
                relative
                for relative, _path, _info in self._iter_entries(
                    source_root,
                    excluded_roots=(Path(snapshot.artifact_root),),
                )
            }
            for relative in sorted(current - expected.keys(), key=lambda item: item.count(os.sep), reverse=True):
                path = source_root / relative
                try:
                    if path.is_symlink() or path.is_file():
                        path.unlink()
                except OSError as exc:
                    raise WorkspaceSnapshotError(f"failed to remove repair-created path {path}: {exc}") from exc
            for relative, entry in expected.items():
                destination = _safe_view_destination(source_root, relative)
                ensure_owner_only_directory(destination.parent)
                data = (blob_root / entry.content_hash).read_bytes()
                if entry.kind == "symlink":
                    if destination.exists() or destination.is_symlink():
                        if destination.is_dir() and not destination.is_symlink():
                            shutil.rmtree(destination)
                        else:
                            destination.unlink()
                    os.symlink(
                        os.fsdecode(data),
                        destination,
                        target_is_directory=entry.symlink_target_is_dir,
                    )
                    continue
                if destination.is_symlink() or (destination.exists() and not destination.is_file()):
                    if destination.is_dir() and not destination.is_symlink():
                        shutil.rmtree(destination)
                    else:
                        destination.unlink()
                atomic_write_owner_only_bytes(destination, data)
                if not get_platform().is_windows:
                    destination.chmod(entry.mode)
            self._prune_empty_in_scope_dirs(source_root)
        for reference in snapshot.references:
            if reference.managed_by_root:
                continue
            self._restore_reference(reference, blob_root)

    def matches(self, snapshot: WorkspaceSnapshot) -> bool:
        """Return whether every in-scope live entry still matches *snapshot*."""
        for root in snapshot.roots:
            source_root = Path(root.source_root)
            if not source_root.is_dir():
                return False
            try:
                live = {
                    relative: (path, info)
                    for relative, path, info in self._iter_entries(
                        source_root,
                        excluded_roots=(Path(snapshot.artifact_root),),
                    )
                }
            except WorkspaceSnapshotError:
                return False
            expected = {entry.relative_path: entry for entry in root.entries}
            if live.keys() != expected.keys():
                return False
            for relative, entry in expected.items():
                path, info = live[relative]
                if stat.S_IMODE(info.st_mode) != entry.mode:
                    return False
                try:
                    data = os.fsencode(os.readlink(path)) if entry.kind == "symlink" else _stable_file_bytes(path, info)
                except WorkspaceSnapshotError, OSError:
                    return False
                if hashlib.sha256(data).hexdigest() != entry.content_hash:
                    return False
        for reference in snapshot.references:
            if reference.managed_by_root:
                continue
            path = Path(reference.source_path)
            try:
                info = path.lstat()
                data = (
                    os.fsencode(os.readlink(path))
                    if reference.entry.kind == "symlink"
                    else _stable_file_bytes(path, info)
                )
            except WorkspaceSnapshotError, OSError:
                return False
            if stat.S_IMODE(info.st_mode) != reference.entry.mode:
                return False
            if hashlib.sha256(data).hexdigest() != reference.entry.content_hash:
                return False
        return True

    @staticmethod
    def discard(snapshot: WorkspaceSnapshot) -> None:
        shutil.rmtree(snapshot.artifact_root, ignore_errors=True)

    def _capture_root(
        self,
        source_root: Path,
        view_root: Path,
        blob_root: Path,
        *,
        excluded_roots: tuple[Path, ...] = (),
    ) -> tuple[list[SnapshotEntry], int]:
        entries: list[SnapshotEntry] = []
        total_bytes = 0
        for relative, path, info in self._iter_entries(source_root, excluded_roots=excluded_roots):
            if len(entries) >= self._max_entries_per_root:
                raise WorkspaceSnapshotError(
                    f"workspace root exceeds {self._max_entries_per_root} snapshot entries: {source_root}"
                )
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                try:
                    data = os.fsencode(os.readlink(path))
                except OSError as exc:
                    raise WorkspaceSnapshotError(f"failed to read workspace symlink {path}: {exc}") from exc
                target_is_dir = path.is_dir()
                visible = self._symlink_stays_inside(source_root, path)
                reason = "" if visible else "symlink_target_outside_snapshot"
                digest = hashlib.sha256(data).hexdigest()
                _write_blob(blob_root, digest, data)
                if visible:
                    destination = _safe_view_destination(view_root, relative)
                    ensure_owner_only_directory(destination.parent)
                    os.symlink(os.fsdecode(data), destination, target_is_directory=target_is_dir)
                entries.append(
                    SnapshotEntry(
                        relative_path_b64=_encoded_relative_path(relative),
                        kind="symlink",
                        mode=mode,
                        size=len(data),
                        content_hash=digest,
                        model_visible=visible,
                        metadata_reason=reason,
                        symlink_target_is_dir=target_is_dir,
                    )
                )
                total_bytes += len(data)
                self._check_total_bytes(total_bytes)
                continue
            data = _stable_file_bytes(path, info)
            total_bytes += len(data)
            self._check_total_bytes(total_bytes)
            digest = hashlib.sha256(data).hexdigest()
            _write_blob(blob_root, digest, data)
            too_large = len(data) > self._max_model_file_bytes
            binary = EncodingDetector.looks_binary(data[:SNAPSHOT_SAMPLE_BYTES])
            visible = not too_large and not binary
            reason = "too_large" if too_large else "binary" if binary else ""
            if visible:
                _materialize_model_entry(view_root=view_root, relative_path=relative, data=data, mode=mode)
            entries.append(
                SnapshotEntry(
                    relative_path_b64=_encoded_relative_path(relative),
                    kind="file",
                    mode=mode,
                    size=len(data),
                    content_hash=digest,
                    model_visible=visible,
                    metadata_reason=reason,
                )
            )
        return entries, total_bytes

    def _capture_references(
        self,
        paths: list[str],
        roots: list[SnapshotRoot],
        roots_root: Path,
        blob_root: Path,
    ) -> tuple[list[SnapshotReference], int]:
        references: list[SnapshotReference] = []
        total_bytes = 0
        for index, raw_path in enumerate(paths):
            source = Path(raw_path).expanduser().absolute()
            if not source.is_file() and not source.is_symlink():
                raise WorkspaceSnapshotError(f"reference file is unavailable: {source}")
            managed = self._managed_reference(source, roots)
            if managed is not None:
                references.append(managed)
                continue
            try:
                info = source.lstat()
            except OSError as exc:
                raise WorkspaceSnapshotError(f"failed to stat reference file {source}: {exc}") from exc
            view_root = roots_root / "references" / str(index)
            ensure_owner_only_directory(view_root)
            view_path = view_root / source.name
            entry, size = self._capture_single_entry(
                source,
                source.name,
                view_root,
                blob_root,
                info,
            )
            total_bytes += size
            self._check_total_bytes(total_bytes)
            references.append(
                SnapshotReference(
                    source_path=str(source),
                    view_path=str(view_path),
                    entry=entry,
                    managed_by_root=False,
                )
            )
        return references, total_bytes

    @staticmethod
    def _managed_reference(source: Path, roots: list[SnapshotRoot]) -> SnapshotReference | None:
        for root in roots:
            source_root = Path(root.source_root)
            try:
                relative = os.path.relpath(source, source_root)
            except ValueError:
                continue
            if relative == os.pardir or relative.startswith(os.pardir + os.sep):
                continue
            entry = next((candidate for candidate in root.entries if candidate.relative_path == relative), None)
            if entry is None:
                continue
            return SnapshotReference(
                source_path=str(source),
                view_path=str(Path(root.view_root) / relative),
                entry=entry,
                managed_by_root=True,
            )
        return None

    def _capture_single_entry(
        self,
        path: Path,
        relative: str,
        view_root: Path,
        blob_root: Path,
        info: os.stat_result,
    ) -> tuple[SnapshotEntry, int]:
        mode = stat.S_IMODE(info.st_mode)
        if stat.S_ISLNK(info.st_mode):
            data = os.fsencode(os.readlink(path))
            # A standalone reference view does not include the symlink's
            # sibling/parent tree, so materializing it could point outside
            # the frozen view even when the live target shares a directory.
            visible = False
            digest = hashlib.sha256(data).hexdigest()
            _write_blob(blob_root, digest, data)
            if visible:
                os.symlink(os.fsdecode(data), view_root / relative, target_is_directory=path.is_dir())
            return (
                SnapshotEntry(
                    relative_path_b64=_encoded_relative_path(relative),
                    kind="symlink",
                    mode=mode,
                    size=len(data),
                    content_hash=digest,
                    model_visible=visible,
                    metadata_reason="" if visible else "symlink_target_outside_snapshot",
                    symlink_target_is_dir=path.is_dir(),
                ),
                len(data),
            )
        if not stat.S_ISREG(info.st_mode):
            raise WorkspaceSnapshotError(f"unsupported reference file type: {path}")
        data = _stable_file_bytes(path, info)
        digest = hashlib.sha256(data).hexdigest()
        _write_blob(blob_root, digest, data)
        too_large = len(data) > self._max_model_file_bytes
        binary = EncodingDetector.looks_binary(data[:SNAPSHOT_SAMPLE_BYTES])
        visible = not too_large and not binary
        if visible:
            _materialize_model_entry(view_root=view_root, relative_path=relative, data=data, mode=mode)
        return (
            SnapshotEntry(
                relative_path_b64=_encoded_relative_path(relative),
                kind="file",
                mode=mode,
                size=len(data),
                content_hash=digest,
                model_visible=visible,
                metadata_reason="too_large" if too_large else "binary" if binary else "",
            ),
            len(data),
        )

    @staticmethod
    def _restore_reference(reference: SnapshotReference, blob_root: Path) -> None:
        destination = Path(reference.source_path)
        entry = reference.entry
        ensure_owner_only_directory(destination.parent)
        data = (blob_root / entry.content_hash).read_bytes()
        if destination.exists() or destination.is_symlink():
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if entry.kind == "symlink":
            os.symlink(os.fsdecode(data), destination, target_is_directory=entry.symlink_target_is_dir)
            return
        atomic_write_owner_only_bytes(destination, data)
        if not get_platform().is_windows:
            destination.chmod(entry.mode)

    def _iter_entries(self, source_root: Path, *, excluded_roots: tuple[Path, ...] = ()):
        excluded = {
            os.path.normcase(os.path.realpath(path)) for path in excluded_roots if self._is_within(source_root, path)
        }
        ignore = GitignoreFilter.from_file(str(source_root / ".gitignore"))
        for current, dirs, files in os.walk(source_root, topdown=True, followlinks=False):
            current_path = Path(current)
            relative_dir = os.path.relpath(current_path, source_root)
            if relative_dir == ".":
                relative_dir = ""
            dirs[:] = [
                name
                for name in dirs
                if os.path.normcase(os.path.realpath(current_path / name)) not in excluded
                and not _is_excluded_name(name)
                and not ignore.is_ignored(os.path.join(relative_dir, name), is_dir=True)
            ]
            names = [*files, *(name for name in dirs if (current_path / name).is_symlink())]
            for name in names:
                if _is_excluded_name(name):
                    continue
                relative = os.path.join(relative_dir, name) if relative_dir else name
                path = source_root / relative
                if ignore.is_ignored(relative, is_dir=False):
                    continue
                try:
                    info = path.lstat()
                except OSError as exc:
                    raise WorkspaceSnapshotError(f"failed to stat workspace entry {path}: {exc}") from exc
                if stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                    yield relative, path, info

    @staticmethod
    def _workspace_roots(workspace: Workspace) -> list[WorkingDir]:
        roots = [WorkingDir(path=workspace.primary_cwd, is_primary=True)]
        roots.extend(workspace.working_dirs)
        unique: list[WorkingDir] = []
        seen: set[str] = set()
        for root in roots:
            normalized = os.path.normcase(os.path.realpath(os.path.abspath(root.path)))
            if normalized in seen:
                if root.is_primary:
                    for index, existing in enumerate(unique):
                        if os.path.normcase(os.path.realpath(os.path.abspath(existing.path))) == normalized:
                            unique[index] = WorkingDir(path=existing.path, label=existing.label, is_primary=True)
                continue
            seen.add(normalized)
            unique.append(WorkingDir(path=root.path, label=root.label, is_primary=root.is_primary))
        if unique and not any(root.is_primary for root in unique):
            unique[0].is_primary = True
        return unique

    @staticmethod
    def _symlink_stays_inside(source_root: Path, path: Path) -> bool:
        try:
            resolved = path.resolve(strict=False)
            common = os.path.commonpath((str(source_root), str(resolved)))
        except OSError, ValueError:
            return False
        return os.path.normcase(common) == os.path.normcase(str(source_root))

    @staticmethod
    def _is_within(root: Path, path: Path) -> bool:
        try:
            common = os.path.commonpath((os.path.realpath(root), os.path.realpath(path)))
        except OSError, ValueError:
            return False
        return os.path.normcase(common) == os.path.normcase(os.path.realpath(root))

    def _check_total_bytes(self, total_bytes: int) -> None:
        if total_bytes > self._max_total_bytes:
            raise WorkspaceSnapshotError(f"workspace snapshot exceeds {self._max_total_bytes} bytes")

    def _prune_empty_in_scope_dirs(self, source_root: Path) -> None:
        for current, dirs, files in os.walk(source_root, topdown=False, followlinks=False):
            path = Path(current)
            if path == source_root or files or dirs or _is_excluded_name(path.name):
                continue
            with contextlib.suppress(OSError):
                path.rmdir()
