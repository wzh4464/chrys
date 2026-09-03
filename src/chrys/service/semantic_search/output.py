# Copyright (c) 2026 Chrys. All rights reserved.

"""Artifact naming, manifest creation, and safe report loading."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from chrys.service.mutations.scanner import DEFAULT_EXCLUDES

from .models import LocalizationArtifact


def requirement_hash(requirement: str) -> str:
    """Return a stable hash for cache validation."""
    return hashlib.sha256(requirement.encode("utf-8", "surrogateescape")).hexdigest()


_FINGERPRINT_EXCLUDED_DIRS: frozenset[str] = DEFAULT_EXCLUDES | {
    ".semantic-search",
    ".semantic-search-tools",
}


def _is_within(candidate: Path, root: Path) -> bool:
    """Whether *candidate* is *root* or lives under it, resolving symlinks."""
    try:
        resolved = candidate.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def repo_fingerprint(repo: Path, *, exclude: Path | None = None) -> str:
    """Fingerprint repository source metadata for cache invalidation.

    *exclude* is a directory whose contents must not count -- the artifact
    directory. `chrys locate --artifact-dir` documents putting it inside the
    repository, and the run's own outputs land there: the manifest is written
    last, so the next call's fingerprint sees a file the stored one could not,
    and the cache is invalid forever. The run's own results are not repository
    source, so they cannot invalidate a result about it.
    """
    digest = hashlib.sha256()
    skip = None
    if exclude is not None:
        try:
            skip = exclude.resolve()
        except OSError:
            skip = None
    try:
        stat = repo.stat()
        digest.update(f"{repo}:{stat.st_mtime_ns}:{stat.st_size}".encode())
        head = repo / ".git" / "HEAD"
        if head.is_file():
            digest.update(head.read_bytes()[:4096])
        index = repo / ".git" / "index"
        if index.is_file():
            index_stat = index.stat()
            digest.update(f"{index_stat.st_mtime_ns}:{index_stat.st_size}".encode())
        for directory, dirnames, filenames in os.walk(repo):
            # The fingerprint runs before every localization, twice, so it must
            # not stat a populated node_modules/ or target/. Reuse the shared
            # exclude set rather than a local shortlist: dependency and build
            # trees are not source, so their mtimes cannot invalidate a
            # localization cache anyway.
            dirnames[:] = sorted(name for name in dirnames if name not in _FINGERPRINT_EXCLUDED_DIRS)
            if skip is not None:
                dirnames[:] = [name for name in dirnames if not _is_within(Path(directory) / name, skip)]
                if _is_within(Path(directory), skip):
                    continue
            for name in sorted(filenames):
                path = Path(directory) / name
                try:
                    item = path.stat()
                except OSError:
                    continue
                digest.update(f"{path.relative_to(repo).as_posix()}:{item.st_mtime_ns}:{item.st_size}".encode())
    except OSError:
        digest.update(str(repo).encode())
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
) -> None:
    """Atomically write a cache manifest using repository-relative display data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "semantic-search-manifest",
        "schema_version": "semantic-search-manifest.v1",
        "requirement_sha256": requirement_hash(requirement),
        "repo_fingerprint": repo_fingerprint(repo, exclude=path.parent),
        "mode": mode,
        "artifacts": {
            "index": artifacts.index_json.name,
            "localization": artifacts.result_json.name,
            "report": artifacts.report_markdown.name,
            "graph": artifacts.graph_json.name,
            "trace": artifacts.trace_jsonl.name,
            "codegraph": artifacts.codegraph_json.name if artifacts.codegraph_json else "",
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_report(path: Path, *, max_chars: int = 60_000) -> str:
    """Read a Markdown report for context injection, bounded by characters."""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars]
