# Copyright (c) 2026 Chrys. All rights reserved.

"""Reconstruct a fixed-P0 worktree from its input patch and Chrys mutations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git(workspace: Path, *args: str, stdout: Any = None) -> None:
    subprocess.run(  # noqa: S603
        ["/usr/bin/git", "-C", str(workspace), *args],
        check=True,
        stdout=stdout,
    )


def _include_untracked(workspace: Path) -> None:
    _git(workspace, "add", "--intent-to-add", "--all", "--", ".")


def capture_p0(workspace: Path, output: Path) -> None:
    """Capture the complete initial P0, including untracked files."""
    _include_untracked(workspace)
    try:
        with output.open("wb") as stream:
            _git(workspace, "diff", "--binary", "HEAD", stdout=stream)
    finally:
        _git(workspace, "reset", "--mixed", "HEAD")
    if output.stat().st_size == 0:
        raise ValueError("fixed-P0 input patch is empty")


def _latest_session(session_root: Path) -> Path:
    candidates = list(session_root.glob("sessions/*/session.json"))
    if not candidates:
        raise FileNotFoundError(f"no completed Chrys session under {session_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _workspace_path(raw_path: object, workspace: Path) -> Path | None:
    if not isinstance(raw_path, str):
        return None
    path = Path(raw_path)
    try:
        path.relative_to(workspace)
    except ValueError:
        return None
    return path


def _restore_mutations(session_path: Path, workspace: Path) -> int:
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    state = payload.get("state") if isinstance(payload, dict) else None
    mutation_state = state.get("chrys_mutations") if isinstance(state, dict) else None
    turns = mutation_state.get("turns") if isinstance(mutation_state, dict) else None
    if not isinstance(turns, list):
        raise ValueError("completed Chrys session has no mutation log")

    latest: dict[Path, dict[str, Any]] = {}
    for turn in turns:
        mutations = turn.get("mutations") if isinstance(turn, dict) else None
        if not isinstance(mutations, list):
            continue
        for mutation in mutations:
            if not isinstance(mutation, dict):
                continue
            path = _workspace_path(mutation.get("path"), workspace)
            if path is not None:
                latest[path] = mutation

    blobs = session_path.parent / "mutations"
    for path, mutation in latest.items():
        operation = mutation.get("operation")
        after_hash = mutation.get("after_hash")
        if operation == "delete" and not after_hash:
            path.unlink(missing_ok=True)
            continue
        if not isinstance(after_hash, str) or not after_hash:
            raise ValueError(f"mutation for {path} has no final content hash")
        blob = blobs / after_hash
        if not blob.is_file():
            raise FileNotFoundError(f"mutation blob missing for {path}: {after_hash}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob.read_bytes())
    return len(latest)


def reconstruct_patch(session_root: Path, workspace: Path, p0_patch: Path, output: Path) -> int:
    """Restore P0, overlay final Chrys mutations, and emit the complete P1 patch."""
    session_path = _latest_session(session_root)
    _git(workspace, "reset", "--hard", "HEAD")
    _git(workspace, "clean", "-fd")
    _git(workspace, "apply", "--binary", str(p0_patch))
    mutation_count = _restore_mutations(session_path, workspace)
    _include_untracked(workspace)
    with output.open("wb") as stream:
        _git(workspace, "diff", "--binary", "HEAD", stdout=stream)
    if output.stat().st_size == 0:
        raise ValueError("reconstructed fixed-P0 repair patch is empty")
    return mutation_count


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--workspace", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    reconstruct = subparsers.add_parser("reconstruct")
    reconstruct.add_argument("--workspace", type=Path, required=True)
    reconstruct.add_argument("--session-root", type=Path, required=True)
    reconstruct.add_argument("--p0-patch", type=Path, required=True)
    reconstruct.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = args.workspace.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "capture":
        capture_p0(workspace, args.output)
        return 0
    mutation_count = reconstruct_patch(
        args.session_root.resolve(strict=True),
        workspace,
        args.p0_patch.resolve(strict=True),
        args.output,
    )
    sys.stdout.write(f"reconstructed fixed-P0 patch from {mutation_count} mutated paths\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
