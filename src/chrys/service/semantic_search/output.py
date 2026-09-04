# Copyright (c) 2026 Chrys. All rights reserved.

"""Artifact naming, manifest creation, and safe report loading."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.foundation.trajectory.keys import ensure_owner_only_directory
from chrys.service.mutations.scanner import DEFAULT_EXCLUDES

from .models import LocalizationArtifact


def requirement_hash(requirement: str) -> str:
    """Return a stable hash for cache validation."""
    return hashlib.sha256(requirement.encode("utf-8", "surrogateescape")).hexdigest()


_FINGERPRINT_EXCLUDED_DIRS = frozenset(DEFAULT_EXCLUDES) | {".semantic-search", ".semantic-search-tools"}


def repo_fingerprint(
    repo: Path, *, max_files: int = 20_000, max_file_bytes: int = 350_000, exclude: Path | None = None
) -> str:
    """Fingerprint repository contents without following filesystem symlinks.

    *exclude* is the artifact directory. `chrys locate --artifact-dir` documents
    putting it inside the repository, and the run's own outputs land there: the
    manifest is written last, so the next call's fingerprint sees a file the
    stored one could not, and the cache is invalid forever. A run's results are
    not repository source, so they cannot invalidate a result about it.
    """
    digest = hashlib.sha256()
    skip: str | None = None
    if exclude is not None:
        try:
            skip = os.path.realpath(exclude)
        except OSError:
            skip = None
    try:
        stat = repo.stat()
        digest.update(os.fsencode(repo))
        digest.update(f":{stat.st_mtime_ns}:{stat.st_size}".encode())
        head = repo / ".git" / "HEAD"
        if head.is_file() and not head.is_symlink():
            with head.open("rb") as handle:
                digest.update(handle.read(4096))
        index = repo / ".git" / "index"
        if index.is_file() and not index.is_symlink():
            with index.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        visited = 0
        for directory, dirnames, filenames in os.walk(repo):
            dirnames[:] = sorted(
                name
                for name in dirnames
                # Dependency and build trees are not source, so their mtimes
                # cannot invalidate a localization; the fingerprint runs before
                # every localization, so it must not stat a populated
                # node_modules/ or target/ either.
                if name not in _FINGERPRINT_EXCLUDED_DIRS
                and not (Path(directory) / name).is_symlink()
                and (skip is None or os.path.realpath(Path(directory) / name) != skip)
            )
            for name in sorted(filenames):
                if visited >= max_files:
                    break
                path = Path(directory) / name
                try:
                    if path.is_symlink() or not path.is_file():
                        continue
                    visited += 1
                    digest.update(os.fsencode(path.relative_to(repo)))
                    size = path.stat().st_size
                    digest.update(str(size).encode("ascii"))
                    if size <= max_file_bytes:
                        with path.open("rb") as handle:
                            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                                digest.update(chunk)
                except OSError:
                    continue
            if visited >= max_files:
                break
    except OSError:
        digest.update(os.fsencode(repo))
    return digest.hexdigest()


def artifact_paths(artifact_dir: Path) -> LocalizationArtifact:
    """Return the canonical machine artifacts for *artifact_dir*."""
    return LocalizationArtifact(
        result_json=artifact_dir / "code-localization.json",
        report_markdown=artifact_dir / "code-localization.md",
        index_json=artifact_dir / "index.json",
        graph_json=artifact_dir / "localization-graph.json",
        trace_jsonl=artifact_dir / "localization-trace.jsonl",
        manifest_json=artifact_dir / "manifest.json",
        codegraph_json=artifact_dir / "codegraph-perception.json",
    )


def write_manifest(
    path: Path,
    *,
    repo: Path,
    requirement: str,
    artifacts: LocalizationArtifact,
    mode: str,
    config_sha256: str,
    max_files: int,
    max_file_bytes: int,
) -> None:
    """Atomically write a cache manifest using repository-relative display data."""
    ensure_owner_only_directory(path.parent)
    payload: dict[str, Any] = {
        "format": "semantic-search-manifest",
        "schema_version": "semantic-search-manifest.v1",
        "requirement_sha256": requirement_hash(requirement),
        "repo_fingerprint": repo_fingerprint(
            repo, max_files=max_files, max_file_bytes=max_file_bytes, exclude=path.parent
        ),
        "mode": mode,
        "config_sha256": config_sha256,
        "artifacts": {
            "index": artifacts.index_json.name,
            "localization": artifacts.result_json.name,
            "report": artifacts.report_markdown.name,
            "graph": artifacts.graph_json.name,
            "trace": artifacts.trace_jsonl.name,
            "codegraph": artifacts.codegraph_json.name if artifacts.codegraph_json else "",
        },
    }
    atomic_write_owner_only_text(path, json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n")


def load_report(path: Path, *, max_chars: int = 60_000) -> str:
    """Read a Markdown report for context injection, bounded by characters."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars]
