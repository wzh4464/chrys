# Copyright (c) 2026 Chrys. All rights reserved.

"""Semantic localization pipeline backed by the bundled SemLoc-compatible skill.

The skill scripts remain the canonical implementation of indexing, graph
normalization, five-tool DFS/BFS, CodeGraph integration, and deterministic
fallback.  This service wrapper gives Chrys a typed, cache-aware entry point
without importing the scripts into the application process.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from chrys.foundation.platform.files import atomic_write_owner_only_text, surrogate_safe_text
from chrys.foundation.platform.process import managed_subprocess
from chrys.foundation.trajectory.keys import ensure_owner_only_directory

from .config import SemanticSearchConfig, SemanticSearchMode
from .models import LocalizationResult
from .output import artifact_paths, repo_fingerprint, requirement_hash, write_manifest

_MAX_RESULT_BYTES = 8 * 1024 * 1024
_MAX_INDEX_BYTES = 32 * 1024 * 1024
_MAX_SUBPROCESS_OUTPUT_BYTES = 1024 * 1024
_ARTIFACT_FILENAMES = frozenset(
    {
        "PROMPT.md",
        "code-localization.json",
        "code-localization.md",
        "index.json",
        "localization-graph.json",
        "localization-trace.jsonl",
        "manifest.json",
        "codegraph-perception.json",
        "codegraph-perception.md",
    }
)


class SemanticSearchError(RuntimeError):
    """Raised when a required localization stage fails."""


def _skill_script_dir() -> Path:
    source_checkout = Path(__file__).resolve().parents[4] / ".agents" / "skills" / "semantic-search" / "scripts"
    if source_checkout.is_dir():
        return source_checkout
    return Path(__file__).resolve().parents[2] / "_skills" / "semantic-search" / "scripts"


def _resolve_repo(repo: str | Path) -> Path:
    path = Path(repo).expanduser().resolve()
    if not path.is_dir():
        raise SemanticSearchError(f"repository does not exist: {path}")
    return path


def _safe_artifact_dir(repo: Path, artifact_dir: str | Path | None) -> Path:
    requested = Path(artifact_dir).expanduser() if artifact_dir else repo / ".semantic-search"
    if requested.is_symlink():
        raise SemanticSearchError(f"artifact directory cannot be a symlink: {requested}")
    path = requested.resolve()
    if path == repo:
        raise SemanticSearchError("artifact directory must not be the repository root")
    if path.is_dir():
        unexpected = sorted(child.name for child in path.iterdir() if child.name not in _ARTIFACT_FILENAMES)
        if unexpected:
            names = ", ".join(unexpected[:5])
            raise SemanticSearchError(f"artifact directory contains unrelated files: {names}")
    ensure_owner_only_directory(path)
    return path


async def _run_script(script: str, arguments: list[str], *, cwd: Path, timeout: float) -> None:
    script_path = _skill_script_dir() / script
    if not script_path.is_file():
        raise SemanticSearchError(f"semantic-search script is missing: {script_path}")
    try:
        async with managed_subprocess(
            sys.executable,
            str(script_path),
            *arguments,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8:backslashreplace"},
        ) as process:

            async def _collect_output() -> tuple[bytes, bytes]:
                async with asyncio.TaskGroup() as task_group:
                    stdout_task = task_group.create_task(
                        _read_stream_bounded(process.stdout, _MAX_SUBPROCESS_OUTPUT_BYTES)
                    )
                    stderr_task = task_group.create_task(
                        _read_stream_bounded(process.stderr, _MAX_SUBPROCESS_OUTPUT_BYTES)
                    )
                    await process.wait()
                return await stdout_task, await stderr_task

            stdout, stderr = await asyncio.wait_for(_collect_output(), timeout=timeout)
    except (OSError, TimeoutError) as exc:
        raise SemanticSearchError(f"semantic-search {script} failed to start or timed out: {exc}") from exc
    if process.returncode:
        detail = (stderr or stdout or b"").decode("utf-8", errors="replace").strip()[-4000:]
        raise SemanticSearchError(f"semantic-search {script} failed ({process.returncode}): {detail}")


async def _read_stream_bounded(stream: asyncio.StreamReader, limit: int) -> bytes:
    chunks: list[bytes] = []
    captured = 0
    while chunk := await stream.read(64 * 1024):
        if captured < limit:
            retained = chunk[: limit - captured]
            chunks.append(retained)
            captured += len(retained)
    return b"".join(chunks)


def _config_hash(config: SemanticSearchConfig, *, codegraph_command: str) -> str:
    payload = {
        "mode": config.mode.value,
        "model_profile": config.model_profile,
        "max_iterations": config.max_iterations,
        "top_locations": config.top_locations,
        "max_tool_results": config.max_tool_results,
        "max_files": config.max_files,
        "max_file_bytes": config.max_file_bytes,
        "codegraph_command": codegraph_command,
        "codegraph_runtime": _codegraph_runtime_fingerprint(codegraph_command),
        "codegraph_install": os.environ.get("SEMANTIC_SEARCH_CODEGRAPH_INSTALL", "never"),
    }
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _codegraph_runtime_fingerprint(command: str) -> str:
    executable = ""
    if command:
        try:
            executable = shutil.which(shlex.split(command)[0]) or ""
        except ValueError:
            executable = ""
    else:
        executable = shutil.which("codegraph") or ""
    module = importlib.util.find_spec("codegraph")
    module_origin = module.origin if module is not None and module.origin is not None else ""
    rows = [command, executable, module_origin]
    for value in (executable, module_origin):
        if not value:
            continue
        try:
            stat = Path(value).stat()
        except OSError:
            continue
        rows.append(f"{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(rows)


def _load_json_artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_RESULT_BYTES:
        raise SemanticSearchError(f"unsafe or oversized semantic-search artifact: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise SemanticSearchError(f"failed to load semantic-search artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SemanticSearchError(f"semantic-search artifact is not a mapping: {path}")
    return value


def _validate_artifact_file(path: Path, *, max_bytes: int) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise SemanticSearchError(f"unsafe or oversized semantic-search artifact: {path}")


def _cache_valid(
    artifacts,
    *,
    repo: Path,
    requirement: str,
    config: SemanticSearchConfig,
    config_sha256: str,
) -> bool:
    if (
        not artifacts.manifest_json.is_file()
        or artifacts.manifest_json.is_symlink()
        or not artifacts.result_json.is_file()
        or artifacts.result_json.is_symlink()
        or not artifacts.report_markdown.is_file()
        or artifacts.report_markdown.is_symlink()
    ):
        return False
    try:
        manifest = _load_json_artifact(artifacts.manifest_json)
    except SemanticSearchError:
        return False
    return (
        manifest.get("requirement_sha256") == requirement_hash(requirement)
        and manifest.get("repo_fingerprint")
        == repo_fingerprint(repo, max_files=config.max_files, max_file_bytes=config.max_file_bytes)
        and manifest.get("config_sha256") == config_sha256
    )


def _redact_payload_paths(payload: dict[str, Any], *, artifact_dir: Path) -> dict[str, Any]:
    """Remove host-specific paths before a report is exposed to the agent."""
    redacted = json.loads(json.dumps(payload, ensure_ascii=True))
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


async def localize_requirement_async(
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
    selected_codegraph_command = (
        codegraph_command.strip() or os.environ.get("SEMANTIC_SEARCH_CODEGRAPH_CMD", "").strip()
    )
    config_sha256 = _config_hash(cfg, codegraph_command=selected_codegraph_command)
    if not refresh and _cache_valid(
        artifacts,
        repo=root,
        requirement=requirement,
        config=cfg,
        config_sha256=config_sha256,
    ):
        payload = _load_json_artifact(artifacts.result_json)
        return LocalizationResult(payload=payload, artifacts=artifacts, reused=True)

    prompt_path = artifacts.result_json.parent / "PROMPT.md"
    atomic_write_owner_only_text(prompt_path, surrogate_safe_text(requirement))
    await _run_script(
        "build_index.py",
        [
            "--repo",
            str(root),
            "--out",
            str(artifacts.index_json),
            "--max-files",
            str(cfg.max_files),
            "--max-file-bytes",
            str(cfg.max_file_bytes),
        ],
        cwd=root,
        timeout=cfg.timeout_seconds,
    )
    _validate_artifact_file(artifacts.index_json, max_bytes=_MAX_INDEX_BYTES)
    # CodeGraph is an optional accelerator.  Probe an existing command by
    # default and never download a binary implicitly from a normal Chrys run;
    # callers that explicitly opt into installation can set
    # SEMANTIC_SEARCH_CODEGRAPH_INSTALL=auto/force.
    codegraph_ready = False
    if artifacts.codegraph_json is not None:
        if artifacts.codegraph_json.is_file() or artifacts.codegraph_json.is_symlink():
            artifacts.codegraph_json.unlink()
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
        if selected_codegraph_command:
            codegraph_args.extend(["--codegraph-cmd", selected_codegraph_command])
        with suppress(SemanticSearchError):
            await _run_script("codegraph_perception.py", codegraph_args, cwd=root, timeout=cfg.timeout_seconds)
            _validate_artifact_file(artifacts.codegraph_json, max_bytes=_MAX_RESULT_BYTES)
            codegraph_ready = True
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
    if codegraph_ready and artifacts.codegraph_json is not None:
        arguments.extend(["--codegraph-perception", str(artifacts.codegraph_json)])
    try:
        await _run_script("localize_task.py", arguments, cwd=root, timeout=cfg.timeout_seconds)
    except SemanticSearchError:
        if cfg.mode is SemanticSearchMode.AUTO:
            arguments[arguments.index("--mode") + 1] = SemanticSearchMode.FALLBACK.value
            await _run_script("localize_task.py", arguments, cwd=root, timeout=cfg.timeout_seconds)
        else:
            raise
    payload = _load_json_artifact(artifacts.result_json)
    redacted = _redact_payload_paths(payload, artifact_dir=artifacts.result_json.parent)
    if redacted != payload:
        atomic_write_owner_only_text(
            artifacts.result_json,
            json.dumps(redacted, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        )
        payload = redacted
    write_manifest(
        artifacts.manifest_json,
        repo=root,
        requirement=requirement,
        artifacts=artifacts,
        mode=cfg.mode.value,
        config_sha256=config_sha256,
        max_files=cfg.max_files,
        max_file_bytes=cfg.max_file_bytes,
    )
    return LocalizationResult(payload=payload, artifacts=artifacts)


def localize_requirement(
    repo: str | Path,
    requirement: str,
    *,
    artifact_dir: str | Path | None = None,
    config: SemanticSearchConfig | None = None,
    refresh: bool = False,
    codegraph_command: str = "",
) -> LocalizationResult:
    """Synchronous entry point used by the standalone CLI."""
    return asyncio.run(
        localize_requirement_async(
            repo,
            requirement,
            artifact_dir=artifact_dir,
            config=config,
            refresh=refresh,
            codegraph_command=codegraph_command,
        )
    )


__all__ = ["SemanticSearchError", "localize_requirement", "localize_requirement_async"]
