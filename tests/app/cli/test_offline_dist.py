# Copyright (c) 2026 Chrys. All rights reserved.

"""Hermetic contracts for the offline distribution's locked Git input."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.paths import REPO_ROOT

_COMMIT = "aa9073bed4970481a035755990e1682e9de486d8"
_REPOSITORY = "https://github.com/SELab-Leibniz/pact.git"
_HELPER = REPO_ROOT / "scripts" / "locked_git_requirement.py"


def _write_lock(path: Path, source: str) -> None:
    path.write_text(
        f'''version = 1

[[package]]
name = "pact-core"
version = "0.2.0.dev0"
source = {{ git = "{source}" }}
''',
        encoding="utf-8",
    )


def _run_helper(lock_path: Path, repository: str = _REPOSITORY) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(_HELPER),
            "--lock",
            str(lock_path),
            "--package",
            "pact-core",
            "--repository",
            repository,
        ],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )


def test_locked_git_requirement_emits_exact_resolved_commit(tmp_path: Path) -> None:
    lock_path = tmp_path / "uv.lock"
    _write_lock(lock_path, f"{_REPOSITORY}?rev={_COMMIT}#{_COMMIT}")

    result = _run_helper(lock_path)

    assert result.returncode == 0
    assert result.stdout.strip() == f"pact-core @ git+{_REPOSITORY}@{_COMMIT}"
    assert result.stderr == ""


def test_committed_lock_has_valid_immutable_pact_requirement() -> None:
    result = _run_helper(REPO_ROOT / "uv.lock")

    assert result.returncode == 0
    assert result.stdout.strip() == f"pact-core @ git+{_REPOSITORY}@{_COMMIT}"
    assert result.stderr == ""


@pytest.mark.parametrize(
    "source",
    [
        f"{_REPOSITORY}?rev=main#{_COMMIT}",
        f"{_REPOSITORY}?rev={'b' * 40}#{_COMMIT}",
        f"https://example.com/pact.git?rev={_COMMIT}#{_COMMIT}",
    ],
)
def test_locked_git_requirement_rejects_mutable_mismatched_or_wrong_source(
    tmp_path: Path,
    source: str,
) -> None:
    lock_path = tmp_path / "uv.lock"
    _write_lock(lock_path, source)

    result = _run_helper(lock_path)

    assert result.returncode != 0
    assert result.stdout == ""


def test_offline_scripts_isolate_git_dependency_from_hashed_install() -> None:
    shell = (REPO_ROOT / "scripts" / "build_offline_dist.sh").read_text(encoding="utf-8")
    powershell = (REPO_ROOT / "scripts" / "build_offline_dist.ps1").read_text(encoding="utf-8")

    assert "--no-emit-package pact-core" in shell
    assert '"$(native_path "$SCRIPT_DIR/locked_git_requirement.py")"' in shell
    assert 'INSTALL_ARGS=(--python "$(native_path "$PY")" --require-hashes --compile-bytecode)' in shell
    assert 'uv pip install --python "$(native_path "$PY")" --no-deps --compile-bytecode' in shell
    assert '"$PACT_REQUIREMENT" --quiet' in shell

    assert '"--no-emit-package", "pact-core"' in powershell
    assert 'Join-Path $ScriptDir "locked_git_requirement.py"' in powershell
    assert "uv pip install --python $Py --require-hashes --compile-bytecode -r $Requirements" in powershell
    assert "uv pip install --python $Py --no-deps --compile-bytecode $PactRequirement" in powershell
