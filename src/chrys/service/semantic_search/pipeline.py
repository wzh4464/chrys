# Copyright (c) 2026 Chrys. All rights reserved.

"""Semantic localization pipeline backed by the bundled SemLoc-compatible skill.

The skill scripts remain the canonical implementation of indexing, graph
normalization, five-tool DFS/BFS, CodeGraph integration, and deterministic
fallback.  This service wrapper gives Chrys a typed, cache-aware entry point
without importing the scripts into the application process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chrys.service.llm.model_lock import ModelLockError
from chrys.service.profiles.models.schema import ModelProfile

from .config import SemanticSearchConfig, SemanticSearchMode
from .models import LocalizationArtifact, LocalizationResult
from .output import artifact_paths, repo_fingerprint, requirement_hash, write_manifest

logger = logging.getLogger(__name__)


class SemanticSearchError(RuntimeError):
    """Raised when a required localization stage fails."""


# Resolved through the package rather than the repository layout: the previous
# ``parents[4]`` walk only ever found the scripts from a source checkout, so an
# installed wheel raised "script is missing" for every localization.
_SKILL_DIR = Path(__file__).resolve().parent / "skill"
_SKILL_SCRIPT_DIR = _SKILL_DIR / "scripts"


def _resolve_repo(repo: str | Path) -> Path:
    path = Path(repo).expanduser().resolve()
    if not path.is_dir():
        raise SemanticSearchError(f"repository does not exist: {path}")
    return path


def _safe_artifact_dir(repo: Path, artifact_dir: str | Path | None) -> Path:
    path = Path(artifact_dir).expanduser().resolve() if artifact_dir else repo / ".semantic-search"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _run_in_process_localization(
    requirement: str,
    *,
    root: Path,
    artifacts: LocalizationArtifact,
    cfg: SemanticSearchConfig,
    model_profile: ModelProfile | None,
    client: Any | None,
    session_id: str | None,
    parent_session_id: str | None,
    session_dir: Path | None,
    warnings: list[str],
) -> Path | None:
    """Run the model-driven search, or explain why it did not run.

    Returns the path holding its raw locations for the rendering script, or
    ``None`` when the deterministic ranking should stand in.
    """
    if cfg.mode in {SemanticSearchMode.OFF, SemanticSearchMode.FALLBACK}:
        return None
    if model_profile is None:
        warnings.append("model_unavailable: no model profile for semantic localization")
        return None
    from chrys.service.semantic_search.localization_model import ChrysLocalizationModel

    codegraph = artifacts.codegraph_json if artifacts.codegraph_json is not None else None
    if codegraph is not None and not codegraph.is_file():
        codegraph = None
    model = ChrysLocalizationModel(
        model_profile,
        session_id=session_id,
        parent_session_id=parent_session_id,
        session_dir=session_dir,
        client=client,
        on_trace=_trace_writer(artifacts.trace_jsonl),
    )
    try:
        run = asyncio.run(
            model.localize(
                requirement,
                repo=root,
                index_path=artifacts.index_json,
                codegraph_path=codegraph,
                config=cfg,
            )
        )
    except ModelLockError as exc:
        # The lock is a deliberate refusal, not a failure: fall back silently
        # to deterministic ranking rather than making a call it forbade.
        warnings.append(f"model_locked: {exc}")
        return None
    except Exception as exc:
        # Both channels, because neither alone survives: the warning is lost
        # when the pipeline goes on to raise, and a trace line is the only
        # record that lands beside the artifacts a reader will actually open.
        logger.warning("in-process localization failed", exc_info=True)
        _trace_writer(artifacts.trace_jsonl)("agent-failed", {"error": f"{type(exc).__name__}: {exc}"[:800]})
        warnings.append(f"model_unavailable: {type(exc).__name__}: {exc}")
        return None
    if run is None:
        warnings.append("model_unavailable: the search returned no repository locations")
        return None
    destination = artifacts.result_json.parent / "agent-locations.json"
    destination.write_text(json.dumps(run.locations, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def _trace_writer(path: Path) -> Callable[[str, dict[str, Any]], None]:
    """Append in-process search events to the same trace the scripts write."""

    def _write(event: str, data: dict[str, Any]) -> None:
        record = {"created_at": datetime.now(tz=UTC).isoformat(), "event": event, **data}
        with suppress(OSError, TypeError, ValueError):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return _write


def _run_script(script: str, arguments: list[str], *, cwd: Path, timeout: float) -> None:
    script_path = _SKILL_SCRIPT_DIR / script
    if not script_path.is_file():
        raise SemanticSearchError(f"semantic-search script is missing: {script_path}")
    try:
        completed = subprocess.run(  # noqa: S603
            [sys.executable, str(script_path), *arguments],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
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


def _cache_valid(artifacts, *, repo: Path, requirement: str, mode: str) -> bool:
    """Whether the cached run answers the question being asked now.

    The mode is part of the question, not a detail of the answer: a cached
    ``fallback`` run is a deterministic ranking with no model behind it, and
    reusing it for an ``llm`` request hands back the cheap result the caller
    explicitly did not ask for, silently flagged as reused.
    """
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
    return (
        manifest.get("requirement_sha256") == requirement_hash(requirement)
        and manifest.get("repo_fingerprint") == repo_fingerprint(repo, exclude=artifacts.manifest_json.parent)
        and manifest.get("mode") == mode
    )


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


def _read_payload(path: Path) -> dict[str, Any]:
    """Read a localization payload, turning corruption into a typed failure.

    A truncated ``code-localization.json`` (disk full, interrupted write) would
    otherwise raise ``json.JSONDecodeError``, which callers that degrade on
    :class:`SemanticSearchError` do not catch -- so one bad artifact would abort
    every later run in that repository until the file was deleted by hand.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticSearchError(f"semantic-search artifact is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SemanticSearchError(f"semantic-search artifact is not an object: {path}")
    return payload


def localize_requirement(
    repo: str | Path,
    requirement: str,
    *,
    artifact_dir: str | Path | None = None,
    config: SemanticSearchConfig | None = None,
    refresh: bool = False,
    codegraph_command: str = "",
    model_profile: ModelProfile | None = None,
    client: Any | None = None,
    session_id: str | None = None,
    parent_session_id: str | None = None,
    session_dir: Path | None = None,
) -> LocalizationResult:
    """Run or reuse localization for one requirement.

    ``requirement`` is kept in a private artifact prompt file.  The generated
    report only contains repository-relative locations and never absolute
    machine paths.

    The deterministic stages run as subprocesses; the search itself runs here,
    through *model_profile*, so it obeys the same model policy as every other
    Chrys model call. Without a profile — or when the model lock rejects it —
    the run degrades to the deterministic ranking and says so in ``warnings``.
    """
    cfg = config or SemanticSearchConfig()
    warnings: list[str] = []
    if cfg.mode is SemanticSearchMode.OFF:
        raise SemanticSearchError("semantic localization is disabled")
    root = _resolve_repo(repo)
    artifacts = artifact_paths(_safe_artifact_dir(root, artifact_dir))
    if not refresh and _cache_valid(artifacts, repo=root, requirement=requirement, mode=cfg.mode.value):
        # A corrupt cache is not a cache: fall through and regenerate rather
        # than failing a run the user never asked to reuse anything for.
        with suppress(SemanticSearchError):
            return LocalizationResult(payload=_read_payload(artifacts.result_json), artifacts=artifacts, reused=True)

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
    agent_locations = _run_in_process_localization(
        requirement,
        root=root,
        artifacts=artifacts,
        cfg=cfg,
        model_profile=model_profile,
        client=client,
        session_id=session_id,
        parent_session_id=parent_session_id,
        session_dir=session_dir,
        warnings=warnings,
    )
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
    if agent_locations is not None:
        arguments.extend(["--locations", str(agent_locations)])
    elif cfg.mode is SemanticSearchMode.LLM:
        # ``llm`` means the model result is the deliverable; a deterministic
        # ranking silently substituted for it would be a different answer.
        raise SemanticSearchError("semantic localization requires a model, and none produced a result")
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
    payload = _read_payload(artifacts.result_json)
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
    return LocalizationResult(payload=payload, artifacts=artifacts, warnings=warnings)


__all__ = ["SemanticSearchError", "localize_requirement"]
