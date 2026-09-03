# Copyright (c) 2026 Chrys. All rights reserved.

"""Delegate dynamic deposition to a local ContextGraph repository checkout."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chrys.service.memory import contextgraph_mcp as backend

MAX_REPOSITORY_PROBLEM_CHARS = 4000
MAX_REPOSITORY_RESPONSE_CHARS = 2000
MAX_REPOSITORY_STEP_CHARS = 1600
MAX_REPOSITORY_STEPS = 32

_SECRET = re.compile(
    r"(sk-[A-Za-z0-9-]{16,}|Bearer\s+[A-Za-z0-9._-]{12,}|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9._-]{20,}\.[A-Za-z0-9._-]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RepositoryDepositResult:
    """Identity and replay status returned by ContextGraph's writer."""

    trajectory_id: str
    fragment_count: int
    created: bool


# How far past the budget to look before redacting. Every secret pattern has a
# minimum length, so a credential straddling the cut would lose enough
# characters to stop matching and be stored as a readable prefix.
_REDACT_LOOKAHEAD = 512


def _redact(value: object, *, limit: int) -> str:
    """Bound *value*, redacting secrets BEFORE the budget cuts them short."""
    window = backend._sanitize(value, limit=limit + _REDACT_LOOKAHEAD)
    return _SECRET.sub("[REDACTED]", window)[:limit]


def _normalize_steps(
    steps: list[dict[str, Any]] | None,
    *,
    final_response: str,
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for step in (steps or [])[:MAX_REPOSITORY_STEPS]:
        if not isinstance(step, dict):
            continue
        action = _redact(step.get("action", ""), limit=MAX_REPOSITORY_STEP_CHARS)
        observation = _redact(step.get("observation", ""), limit=MAX_REPOSITORY_STEP_CHARS)
        if action or observation:
            normalized.append({"action": action, "observation": observation})
    clean_response = _redact(final_response, limit=MAX_REPOSITORY_RESPONSE_CHARS)
    if normalized and clean_response:
        prior = normalized[-1]["observation"]
        normalized[-1]["observation"] = _redact(
            f"{prior}\nAssistant outcome: {clean_response}" if prior else f"Assistant outcome: {clean_response}",
            limit=MAX_REPOSITORY_STEP_CHARS,
        )
    return normalized


def _repository_path() -> Path:
    configured = os.environ.get("CONTEXTGRAPH_REPO", "").strip()
    repository = Path(configured).expanduser() if configured else Path.home() / "codes" / "ContextGraph"
    repository = repository.resolve()
    if not (repository / "agent_memory" / "memory.py").is_file():
        raise RuntimeError(f"ContextGraph repository not found at {repository}; set CONTEXTGRAPH_REPO")
    return repository


def _repository_python(repository: Path) -> Path:
    configured = os.environ.get("CONTEXTGRAPH_PYTHON", "").strip()
    if configured:
        interpreter = Path(configured).expanduser().resolve()
        if not interpreter.is_file():
            raise RuntimeError(f"CONTEXTGRAPH_PYTHON does not name a file: {interpreter}")
        return interpreter
    for candidate in (repository / ".venv" / "bin" / "python", repository / ".venv" / "Scripts" / "python.exe"):
        if candidate.is_file():
            return candidate
    return Path(sys.executable).resolve()


def _timeout_seconds() -> int:
    try:
        configured = int(os.environ.get("CONTEXTGRAPH_DEPOSIT_TIMEOUT_SECONDS", "300"))
    except ValueError:
        configured = 300
    return max(10, min(configured, 900))


def _worker_environment(repository: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment["CONTEXTGRAPH_REPO"] = str(repository)
    environment["NEO4J_URI"] = backend._neo4j_uri()
    user, password = backend._neo4j_auth()
    environment["NEO4J_USER"] = user
    environment["NEO4J_PASSWORD"] = password

    embedding_key = os.environ.get("CONTEXTGRAPH_EMBEDDING_API_KEY", "").strip()
    embedding_base = os.environ.get("CONTEXTGRAPH_EMBEDDING_BASE_URL", "").strip()
    embedding_model = os.environ.get("CONTEXTGRAPH_EMBEDDING_MODEL", "").strip()
    if embedding_key:
        environment["OPENAI_API_KEY"] = embedding_key
    if embedding_base:
        environment["OPENAI_API_BASE"] = embedding_base
    if embedding_model:
        environment["EMBEDDING_MODEL"] = embedding_model
    return environment


def initialize_schema(*, vector_dimensions: int = 1536) -> None:
    """Create ContextGraph's constraints and indexes through its own checkout."""
    _run_worker({"op": "init_schema", "vector_dimensions": vector_dimensions})


def _run_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one isolated worker request in the ContextGraph interpreter."""
    repository = _repository_path()
    worker = Path(__file__).with_name("_contextgraph_repository_worker.py")
    completed = subprocess.run(  # noqa: S603 — interpreter is an operator-configured ContextGraph runtime
        [str(_repository_python(repository)), str(worker)],
        cwd=repository,
        env=_worker_environment(repository),
        input=json.dumps(payload, ensure_ascii=True),
        text=True,
        capture_output=True,
        timeout=_timeout_seconds(),
        check=False,
    )
    if completed.returncode != 0:
        detail = _redact(completed.stderr or completed.stdout or "unknown worker error", limit=2000)
        raise RuntimeError(f"ContextGraph repository writer failed: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("ContextGraph repository writer returned invalid JSON") from exc
    if not isinstance(result, dict):
        raise RuntimeError("ContextGraph repository writer returned an invalid result")
    return result


def _run_repository_writer(payload: dict[str, Any]) -> RepositoryDepositResult:
    result = _run_worker(payload)
    trajectory_id = result.get("trajectory_id")
    fragment_count = result.get("fragment_count")
    created = result.get("created")
    if (
        not isinstance(trajectory_id, str)
        or not isinstance(fragment_count, int)
        or isinstance(fragment_count, bool)
        or not isinstance(created, bool)
    ):
        raise RuntimeError("ContextGraph repository writer returned an invalid result")
    return RepositoryDepositResult(
        trajectory_id=trajectory_id,
        fragment_count=fragment_count,
        created=created,
    )


def deposit_experience(
    *,
    problem_statement: str,
    success: bool,
    steps: list[dict[str, Any]] | None,
    final_response: str = "",
    repo: str = "general",
    source_id: str = "",
) -> RepositoryDepositResult | None:
    """Normalize one experience and pass it to ContextGraph's ``AgentMemory.learn``."""
    normalized_steps = _normalize_steps(steps, final_response=final_response)
    if not normalized_steps:
        return None
    clean_problem = _redact(problem_statement, limit=MAX_REPOSITORY_PROBLEM_CHARS)
    clean_repo = _redact(repo, limit=200) or "general"
    semantic = {
        "problem_statement": clean_problem,
        "repo": clean_repo,
        "steps": normalized_steps,
        "success": bool(success),
    }
    digest = hashlib.sha256(
        json.dumps(semantic, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    stable_source = _redact(source_id, limit=500) or digest
    identity = hashlib.sha256(f"{stable_source}:{digest}".encode()).hexdigest()[:24]
    return _run_repository_writer(
        {
            **semantic,
            "instance_id": f"chrys:{identity}",
            "trajectory_id": f"traj_chrys_{identity}",
        }
    )


def record_manual(
    *,
    problem_statement: str,
    success: bool,
    steps: list[dict[str, Any]] | None,
    repo: str | None,
) -> str:
    """MCP-facing curated write through ContextGraph's repository implementation."""
    result = deposit_experience(
        problem_statement=problem_statement,
        success=success,
        steps=steps,
        repo=(repo or "").strip() or "general",
    )
    if result is None:
        return "No executable experience steps to record."
    disposition = "Recorded" if result.created else "Already recorded"
    return f"{disposition} ContextGraph trajectory ({result.trajectory_id}; fragments={result.fragment_count})."
