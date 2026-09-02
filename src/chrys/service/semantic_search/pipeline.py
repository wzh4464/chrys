# Copyright (c) 2026 Chrys. All rights reserved.

"""Semantic localization pipeline backed by the bundled SemLoc-compatible skill.

The skill scripts remain the canonical implementation of indexing, graph
normalization, five-tool DFS/BFS, CodeGraph integration, and deterministic
fallback.  This service wrapper gives Chrys a typed, cache-aware entry point
without importing the scripts into the application process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from .config import SemanticSearchConfig, SemanticSearchMode
from .models import LocalizationResult
from .output import artifact_paths, repo_fingerprint, requirement_hash, write_manifest


class SemanticSearchError(RuntimeError):
    """Raised when a required localization stage fails."""


_SKILL_SCRIPT_DIR = Path(__file__).resolve().parents[4] / ".agents" / "skills" / "semantic-search" / "scripts"


def _resolve_repo(repo: str | Path) -> Path:
    path = Path(repo).expanduser().resolve()
    if not path.is_dir():
        raise SemanticSearchError(f"repository does not exist: {path}")
    return path


def _safe_artifact_dir(repo: Path, artifact_dir: str | Path | None) -> Path:
    path = Path(artifact_dir).expanduser().resolve() if artifact_dir else repo / ".semantic-search"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_script(script: str, arguments: list[str], *, cwd: Path, timeout: float) -> None:
    script_path = _SKILL_SCRIPT_DIR / script
    if not script_path.is_file():
        raise SemanticSearchError(f"semantic-search script is missing: {script_path}")
    try:
        completed = subprocess.run(  # noqa: S603
            [sys.executable, str(script_path), *arguments],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SemanticSearchError(f"semantic-search {script} failed to start or timed out: {exc}") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "").strip()[-4000:]
        raise SemanticSearchError(f"semantic-search {script} failed ({completed.returncode}): {detail}")


def _cache_valid(artifacts, *, repo: Path, requirement: str) -> bool:
    if (
        not artifacts.manifest_json.is_file()
        or not artifacts.result_json.is_file()
        or not artifacts.report_markdown.is_file()
    ):
        return False
    try:
        manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return False
    return manifest.get("requirement_sha256") == requirement_hash(requirement) and manifest.get(
        "repo_fingerprint"
    ) == repo_fingerprint(repo)


def _redact_payload_paths(payload: dict[str, Any], *, artifact_dir: Path) -> dict[str, Any]:
    """Remove host-specific paths before a report is exposed to the agent."""
    redacted = json.loads(json.dumps(payload, ensure_ascii=False))
    inputs = redacted.get("inputs")
    if isinstance(inputs, dict):
        for key in ("repo", "requirement", "index", "trace", "localization_graph"):
            if key in inputs:
                inputs[key] = Path(str(inputs[key])).name
    summary = redacted.get("summary")
    if isinstance(summary, dict):
        for key in ("trace", "localization_graph"):
            if key in summary:
                summary[key] = Path(str(summary[key])).name
    return redacted


def localize_requirement(
    repo: str | Path,
    requirement: str,
    *,
    artifact_dir: str | Path | None = None,
    config: SemanticSearchConfig | None = None,
    refresh: bool = False,
    codegraph_command: str = "",
) -> LocalizationResult:
    """Run or reuse localization for one requirement.

    ``requirement`` is kept in a private artifact prompt file.  The generated
    report only contains repository-relative locations and never absolute
    machine paths.
    """
    cfg = config or SemanticSearchConfig()
    if cfg.mode is SemanticSearchMode.OFF:
        raise SemanticSearchError("semantic localization is disabled")
    root = _resolve_repo(repo)
    artifacts = artifact_paths(_safe_artifact_dir(root, artifact_dir))
    if not refresh and _cache_valid(artifacts, repo=root, requirement=requirement):
        payload = json.loads(artifacts.result_json.read_text(encoding="utf-8"))
        return LocalizationResult(payload=payload, artifacts=artifacts, reused=True)

    prompt_path = artifacts.result_json.parent / "PROMPT.md"
    prompt_path.write_text(requirement, encoding="utf-8")
    _run_script(
        "build_index.py",
        ["--repo", str(root), "--out", str(artifacts.index_json)],
        cwd=root,
        timeout=cfg.timeout_seconds,
    )
    # CodeGraph is an optional accelerator.  Probe an existing command by
    # default and never download a binary implicitly from a normal Chrys run;
    # callers that explicitly opt into installation can set
    # SEMANTIC_SEARCH_CODEGRAPH_INSTALL=auto/force.
    if artifacts.codegraph_json is not None:
        codegraph_args = [
            "--repo",
            str(root),
            "--requirement",
            str(prompt_path),
            "--index",
            str(artifacts.index_json),
            "--out",
            str(artifacts.codegraph_json),
            "--markdown",
            str(artifacts.codegraph_json.with_suffix(".md")),
            "--artifact-dir",
            str(artifacts.result_json.parent),
            "--install-codegraph",
            os.environ.get("SEMANTIC_SEARCH_CODEGRAPH_INSTALL", "never"),
        ]
        selected_codegraph_command = (
            codegraph_command.strip() or os.environ.get("SEMANTIC_SEARCH_CODEGRAPH_CMD", "").strip()
        )
        if selected_codegraph_command:
            codegraph_args.extend(["--codegraph-cmd", selected_codegraph_command])
        with suppress(SemanticSearchError):
            _run_script("codegraph_perception.py", codegraph_args, cwd=root, timeout=cfg.timeout_seconds)
    arguments = [
        "--repo",
        str(root),
        "--requirement",
        str(prompt_path),
        "--index",
        str(artifacts.index_json),
        "--out",
        str(artifacts.result_json),
        "--markdown",
        str(artifacts.report_markdown),
        "--artifact-dir",
        str(artifacts.result_json.parent),
        "--trace",
        str(artifacts.trace_jsonl),
        "--graph-out",
        str(artifacts.graph_json),
        "--mode",
        cfg.mode.value,
        "--max-iterations",
        str(cfg.max_iterations),
        "--top-locations",
        str(cfg.top_locations),
        "--max-tool-results",
        str(cfg.max_tool_results),
    ]
    if cfg.model_profile:
        arguments.extend(["--model-profile", cfg.model_profile])
    if artifacts.codegraph_json is not None and artifacts.codegraph_json.is_file():
        arguments.extend(["--codegraph-perception", str(artifacts.codegraph_json)])
    try:
        _run_script("localize_task.py", arguments, cwd=root, timeout=cfg.timeout_seconds)
    except SemanticSearchError:
        if cfg.mode is SemanticSearchMode.AUTO:
            arguments[arguments.index("--mode") + 1] = SemanticSearchMode.FALLBACK.value
            _run_script("localize_task.py", arguments, cwd=root, timeout=cfg.timeout_seconds)
        else:
            raise
    payload = json.loads(artifacts.result_json.read_text(encoding="utf-8"))
    redacted = _redact_payload_paths(payload, artifact_dir=artifacts.result_json.parent)
    if redacted != payload:
        artifacts.result_json.write_text(
            json.dumps(redacted, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload = redacted
    write_manifest(
        artifacts.manifest_json,
        repo=root,
        requirement=requirement,
        artifacts=artifacts,
        mode=cfg.mode.value,
    )
    return LocalizationResult(payload=payload, artifacts=artifacts)


__all__ = ["SemanticSearchError", "localize_requirement"]
