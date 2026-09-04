# Copyright (c) 2026 Chrys. All rights reserved.

"""The verify shim lends a detached worktree the primary checkout's ignored dependencies."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from chrys.pact.verify_shim import link_ignored_dependencies, main, wrap_verify_command

_GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(_GIT is None, reason="git is required")


def _repo(root: Path) -> Path:
    primary = root / "primary"
    primary.mkdir()
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@x",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@x",
    }

    def git(*args: str) -> None:
        subprocess.run([str(_GIT), *args], cwd=primary, check=True, capture_output=True, env=env)

    git("init", "-q")
    (primary / ".gitignore").write_text("node_modules/\n", encoding="utf-8")
    (primary / "package.json").write_text("{}", encoding="utf-8")
    (primary / "packages" / "core").mkdir(parents=True)
    (primary / "packages" / "core" / "index.js").write_text("", encoding="utf-8")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    (primary / "node_modules" / ".bin").mkdir(parents=True)
    (primary / "node_modules" / ".bin" / "marker").write_text("root", encoding="utf-8")
    (primary / "packages" / "core" / "node_modules").mkdir()
    (primary / "packages" / "core" / "node_modules" / "marker").write_text("nested", encoding="utf-8")
    worktree = root / "wt"
    git("worktree", "add", "--detach", "-q", str(worktree), "HEAD")
    return worktree


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_ignored_dependency_directories_are_linked_into_the_worktree(tmp_path: Path) -> None:
    worktree = _repo(tmp_path)
    assert not (worktree / "node_modules").exists()

    linked = link_ignored_dependencies(worktree)

    assert sorted(p.relative_to(worktree).as_posix() for p in linked) == ["node_modules", "packages/core/node_modules"]
    assert (worktree / "node_modules" / ".bin" / "marker").read_text(encoding="utf-8") == "root"
    assert (worktree / "packages" / "core" / "node_modules" / "marker").read_text(encoding="utf-8") == "nested"
    # Idempotent: a second pass links nothing new.
    assert link_ignored_dependencies(worktree) == []


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_main_links_then_runs_the_command_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worktree = _repo(tmp_path)
    monkeypatch.chdir(worktree)

    assert (
        main(
            [
                "--",
                f"{sys.executable} -c \"import pathlib,sys; sys.exit(0 if pathlib.Path('node_modules/.bin/marker').exists() else 7)\"",
            ]
        )
        == 0
    )
    assert main(["--", f'{sys.executable} -c "raise SystemExit(3)"']) == 3


def test_a_primary_checkout_is_left_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worktree = _repo(tmp_path)
    primary = worktree.parent / "primary"
    assert link_ignored_dependencies(primary) == []
    plain = tmp_path / "plain"
    plain.mkdir()
    assert link_ignored_dependencies(plain) == []
    monkeypatch.chdir(plain)
    assert main(["--", f'{sys.executable} -c "raise SystemExit(0)"']) == 0
    assert main([]) == 2


def test_wrap_keeps_the_command_intact_through_a_shell() -> None:
    wrapped = wrap_verify_command("npm test --silent -- --grep 'a b'")
    argv = subprocess.run(
        [sys.executable, "-c", "import shlex,sys; print(shlex.split(sys.argv[1])[-1])", wrapped],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert argv == "npm test --silent -- --grep 'a b'"
    assert "chrys.pact.verify_shim" in wrapped
