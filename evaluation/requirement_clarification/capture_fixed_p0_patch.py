# Copyright (c) 2026 Chrys. All rights reserved.

"""Capture a complete fixed-P0/P1 worktree patch relative to its original base."""

from __future__ import annotations

import argparse
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


def record_base(workspace: Path, output: Path) -> str:
    """Persist the exact commit that predates both P0 and P1."""
    result = subprocess.run(  # noqa: S603
        ["/usr/bin/git", "-C", str(workspace), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if not revision:
        raise ValueError("fixed-P0 workspace has no base revision")
    output.write_text(f"{revision}\n", encoding="utf-8")
    return revision


def capture_patch(workspace: Path, base_revision: str, output: Path) -> None:
    """Capture committed, staged, unstaged, deleted, binary, and untracked changes."""
    revision = base_revision.strip()
    if not revision:
        raise ValueError("base revision is empty")
    _git(workspace, "cat-file", "-e", f"{revision}^{{commit}}")
    _git(workspace, "add", "--intent-to-add", "--all", "--", ".")
    with output.open("wb") as stream:
        _git(workspace, "diff", "--binary", revision, stdout=stream)
    if output.stat().st_size == 0:
        raise ValueError("captured fixed-P0 repair patch is empty")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record-base")
    record.add_argument("--workspace", type=Path, required=True)
    record.add_argument("--output", type=Path, required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--workspace", type=Path, required=True)
    capture.add_argument("--base-file", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workspace = args.workspace.resolve(strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "record-base":
        revision = record_base(workspace, args.output)
        sys.stdout.write(f"recorded fixed-P0 base revision {revision}\n")
        return 0
    base_revision = args.base_file.read_text(encoding="utf-8")
    capture_patch(workspace, base_revision, args.output)
    sys.stdout.write(f"captured fixed-P0 repair patch ({args.output.stat().st_size} bytes)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
