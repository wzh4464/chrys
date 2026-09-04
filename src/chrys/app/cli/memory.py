# Copyright (c) 2026 Chrys. All rights reserved.

"""``chrys memory`` — inspect, repair, and provision the ContextGraph deployment.

Three commands cover the three things that go wrong with a graph nobody looks
at: ``doctor`` says why retrieval is silent, ``sweep`` deposits turns the engine
never got to (a killed process, an offline Neo4j), and ``init`` brings a local
graph up on a machine that has none.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chrys.foundation.config.settings import resolve_sessions_dir
from chrys.orchestration.startup import bootstrap_runtime
from chrys.service.memory.contextgraph_deposit import repo_label
from chrys.service.memory.writeback import WATERMARK_KEY, deposit_pending_turns, pending_turns
from chrys.service.state.store import SESSION_FILE_NAME


@dataclass(frozen=True, slots=True)
class _Check:
    name: str
    ok: bool
    detail: str


def build_parser() -> argparse.ArgumentParser:
    """Return the ``chrys memory`` parser."""
    parser = argparse.ArgumentParser(prog="chrys memory", description="Inspect and maintain ContextGraph memory.")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check the ContextGraph deployment and report what is missing")
    doctor.add_argument("--json", action="store_true", help="Print the machine-readable report")

    sweep = sub.add_parser("sweep", help="Deposit turns the engine never got to")
    sweep.add_argument(
        "--idle-seconds",
        type=float,
        default=3600.0,
        help="Only sweep sessions untouched for at least this long (default: 3600)",
    )
    sweep.add_argument("--dry-run", action="store_true", help="List what would be deposited and change nothing")

    init = sub.add_parser("init", help="Start a local ContextGraph Neo4j and create its indexes")
    init.add_argument(
        "--import",
        dest="import_dump",
        default="",
        help="Optional Neo4j dump to load before creating indexes; the default is an empty graph",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``chrys memory``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    _prepare_runtime()
    if args.command == "doctor":
        return _doctor(as_json=args.json)
    if args.command == "sweep":
        return _sweep(idle_seconds=args.idle_seconds, dry_run=args.dry_run)
    return _init(import_dump=args.import_dump)


def _prepare_runtime() -> None:
    """Load .env and settings so CONTEXTGRAPH_* is visible exactly as the engine sees it."""
    bootstrap_runtime(dotenv_override=True, configure_stdio=True, setup_telemetry=False)


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def _doctor(*, as_json: bool) -> int:
    checks = [*_environment_checks(), _bolt_check(), _checkout_check()]
    ok = all(check.ok for check in checks)
    if as_json:
        payload = {
            "ok": ok,
            "checks": [{"name": check.name, "ok": check.ok, "detail": check.detail} for check in checks],
        }
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return 0 if ok else 2
    for check in checks:
        sys.stdout.write(f"[{'ok' if check.ok else 'FAIL'}] {check.name}: {check.detail}\n")
    return 0 if ok else 2


def _environment_checks() -> list[_Check]:
    uri = os.environ.get("CONTEXTGRAPH_NEO4J_URI", "").strip()
    password = os.environ.get("CONTEXTGRAPH_NEO4J_PASSWORD") or os.environ.get("NEO4J_PASSWORD", "")
    embedding = (
        os.environ.get("CONTEXTGRAPH_EMBEDDING_API_KEY", "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    return [
        _Check("CONTEXTGRAPH_NEO4J_URI", bool(uri), uri or "not set; the memory MCP stays detached without it"),
        _Check("CONTEXTGRAPH_NEO4J_PASSWORD", bool(password), "set" if password else "not set"),
        _Check(
            "CONTEXTGRAPH_EMBEDDING_API_KEY",
            bool(embedding),
            "set" if embedding else "not set; reads degrade to the lexical channel, but every deposit will fail",
        ),
    ]


def _bolt_check() -> _Check:
    """Probe Bolt and the four indexes retrieval depends on."""
    uri = os.environ.get("CONTEXTGRAPH_NEO4J_URI", "").strip()
    if not uri:
        return _Check("neo4j", False, "skipped; CONTEXTGRAPH_NEO4J_URI is not set")
    try:
        from chrys.service.memory.contextgraph_mcp import _do_health, missing_retrieval_indexes

        detail = _do_health()
        missing = missing_retrieval_indexes()
    except Exception as exc:  # the driver raises a wide family of transport errors
        return _Check("neo4j", False, f"unreachable: {exc}")
    if missing:
        return _Check(
            "neo4j",
            False,
            f"reachable, but these indexes are missing: {', '.join(missing)}; run `chrys memory init`",
        )
    return _Check("neo4j", True, detail.strip().splitlines()[0] if detail.strip() else "reachable")


def _checkout_check() -> _Check:
    """The write path needs the ContextGraph checkout and its own interpreter."""
    repo = os.environ.get("CONTEXTGRAPH_REPO", "").strip()
    if not repo:
        return _Check("CONTEXTGRAPH_REPO", False, "not set; experience cannot be deposited")
    repository = Path(repo).expanduser()
    if not repository.is_dir():
        return _Check("CONTEXTGRAPH_REPO", False, f"not a directory: {repository}")
    configured = os.environ.get("CONTEXTGRAPH_PYTHON", "").strip()
    interpreter = Path(configured).expanduser() if configured else repository / ".venv" / "bin" / "python"
    if not interpreter.is_file():
        return _Check("CONTEXTGRAPH_REPO", False, f"no interpreter at {interpreter}; set CONTEXTGRAPH_PYTHON")
    return _Check("CONTEXTGRAPH_REPO", True, f"{repository} ({interpreter.name})")


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------


def _sweep(*, idle_seconds: float, dry_run: bool) -> int:
    sessions_dir = resolve_sessions_dir(create=False)
    if not sessions_dir.is_dir():
        sys.stdout.write("No sessions directory; nothing to sweep.\n")
        return 0
    cutoff = time.time() - max(idle_seconds, 0.0)
    failed = False
    for session_file in sorted(sessions_dir.glob(f"*/{SESSION_FILE_NAME}")):
        session_id = session_file.parent.name
        try:
            if session_file.stat().st_mtime > cutoff:
                continue
            watermark = _stored_watermark(session_file)
            pending = len(pending_turns(session_file, watermark))
        except OSError as exc:
            sys.stdout.write(f"{session_id}: unreadable ({exc})\n")
            failed = True
            continue
        if pending <= 0:
            continue
        if dry_run:
            sys.stdout.write(f"{session_id}: {pending} turn(s) pending after watermark {watermark}\n")
            continue
        outcome = deposit_pending_turns(
            session_file,
            watermark=watermark,
            repo=_repo_label(session_file),
            source_prefix=f"chrys-session:{session_id}",
        )
        if outcome.watermark != watermark:
            _store_watermark(session_file, outcome.watermark)
        if outcome.failed is not None:
            sys.stdout.write(f"{session_id}: stopped at turn {outcome.failed}; watermark held at {outcome.watermark}\n")
            failed = True
            continue
        sys.stdout.write(f"{session_id}: deposited {len(outcome.deposited)} turn(s); watermark {outcome.watermark}\n")
    return 1 if failed else 0


def _repo_label(session_file: Path) -> str:
    """Best-effort repository name for a session Chrys is no longer running."""
    try:
        envelope = json.loads(session_file.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return "general"
    meta = envelope.get("meta") if isinstance(envelope, dict) else None
    # The envelope records the workspace as ``primary_cwd``; reading ``cwd`` labelled
    # every swept session "general", which is why recall by repository found nothing.
    cwd = (meta.get("primary_cwd") or meta.get("cwd")) if isinstance(meta, dict) else None
    # Same label the live hook uses, so a swept session and a deposited one
    # land under one repository name.
    return repo_label(cwd if isinstance(cwd, str) else None)


def _stored_watermark(session_file: Path) -> int:
    envelope = json.loads(session_file.read_text(encoding="utf-8"))
    state = envelope.get("state") if isinstance(envelope, dict) else None
    if not isinstance(state, dict):
        return 0
    value = state.get(WATERMARK_KEY, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _store_watermark(session_file: Path, watermark: int) -> None:
    """Write the mark back in place.

    Only this one key is touched: a swept session may be reopened later, and
    rewriting the whole envelope from a partial read would discard state this
    command never modelled.
    """
    envelope = json.loads(session_file.read_text(encoding="utf-8"))
    state = envelope.get("state") if isinstance(envelope, dict) else None
    if not isinstance(state, dict):
        return
    state[WATERMARK_KEY] = watermark
    temporary = session_file.with_suffix(".sweep.tmp")
    temporary.write_text(json.dumps(envelope), encoding="utf-8")
    temporary.replace(session_file)


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------


def _init(*, import_dump: str) -> int:
    dump_path: Path | None = None
    if import_dump:
        dump_path = Path(import_dump).expanduser()
        if not dump_path.is_file():
            sys.stderr.write(f"Error: dump does not exist: {dump_path}\n")
            return 2
    repo = os.environ.get("CONTEXTGRAPH_REPO", "").strip()
    if not repo:
        sys.stderr.write("Error: CONTEXTGRAPH_REPO is not set; it must point at a ContextGraph checkout\n")
        return 2
    repository = Path(repo).expanduser()
    compose = repository / "docker-compose.yml"
    if not compose.is_file():
        sys.stderr.write(f"Error: no docker-compose.yml at {compose}\n")
        return 2
    if not _compose(repository, ["up", "-d", "neo4j"], "start neo4j"):
        return 1
    if dump_path is not None and not _load_dump(repository, dump_path):
        return 1
    return 0 if _create_indexes(repository) else 1


def _compose(repository: Path, arguments: list[str], description: str) -> bool:
    completed = _run(["docker", "compose", "-f", str(repository / "docker-compose.yml"), *arguments])
    if completed is None or completed.returncode != 0:
        detail = "" if completed is None else (completed.stderr or completed.stdout or "").strip()[:2000]
        sys.stderr.write(f"Error: could not {description}: {detail}\n")
        return False
    return True


def _load_dump(repository: Path, dump: Path) -> bool:
    """Load a user-supplied dump through the compose-managed neo4j container.

    Chrys ships no initial graph: the dump is the user's own, and this is only
    the documented way in.
    """
    return _compose(
        repository,
        [
            "run",
            "--rm",
            "-v",
            f"{dump.parent}:/dumps:ro",
            "neo4j",
            "neo4j-admin",
            "database",
            "load",
            "--from-path=/dumps",
            "--overwrite-destination=true",
            "neo4j",
        ],
        f"load {dump.name}",
    )


def _create_indexes(repository: Path) -> bool:
    """Create ContextGraph's schema through its own checkout.

    The labels and properties belong to ContextGraph; re-declaring them here
    would fork the schema the next time upstream changes one.
    """
    from chrys.service.memory.contextgraph_repository import initialize_schema

    try:
        initialize_schema()
    except Exception as exc:  # worker failures surface as RuntimeError or OSError
        sys.stderr.write(f"Error: could not initialize the ContextGraph schema from {repository}: {exc}\n")
        return False
    sys.stdout.write("ContextGraph schema is ready.\n")
    return True


def _run(argv: list[str]) -> subprocess.CompletedProcess[str] | Any:
    try:
        return subprocess.run(  # noqa: S603
            argv,
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        sys.stderr.write(f"Error: {argv[0]} failed to run: {exc}\n")
        return None


if __name__ == "__main__":
    raise SystemExit(main())
