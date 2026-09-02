# Copyright (c) 2026 Chrys. All rights reserved.

"""Hermetic contracts for the vendored pact-core wheel."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.paths import REPO_ROOT

_COMMIT = "aa9073bed4970481a035755990e1682e9de486d8"
_DIGEST = "4f08b2b6ae258823e0e1c824380f58579a6e4174d02b3b1338a32ee9eb76ab2a"
_REPOSITORY = "https://github.com/SELab-Leibniz/pact.git"
_WHEEL_NAME = "pact_core-0.2.0.dev0-py3-none-any.whl"
_HELPER = REPO_ROOT / "scripts" / "vendored_pact_wheel.py"


def _run_helper(project_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(_HELPER), "--project-root", str(project_root)],
        check=False,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )


def _copy_vendored_input(tmp_path: Path) -> tuple[Path, Path]:
    vendor = tmp_path / "vendor"
    wheels = vendor / "wheels"
    wheels.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "vendor" / "pact-core.json", vendor / "pact-core.json")
    wheel = wheels / _WHEEL_NAME
    shutil.copy2(REPO_ROOT / "vendor" / "wheels" / _WHEEL_NAME, wheel)
    return vendor / "pact-core.json", wheel


def test_committed_vendored_pact_wheel_is_valid_and_immutable() -> None:
    result = _run_helper(REPO_ROOT)

    assert result.returncode == 0
    assert Path(result.stdout.strip()) == REPO_ROOT / "vendor" / "wheels" / _WHEEL_NAME
    assert result.stderr == ""
    provenance = json.loads((REPO_ROOT / "vendor" / "pact-core.json").read_text(encoding="utf-8"))
    assert provenance["source_repository"] == _REPOSITORY
    assert provenance["source_commit"] == _COMMIT
    assert provenance["sha256"] == _DIGEST


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_repository", "https://example.com/pact.git"),
        ("source_commit", "main"),
        ("sha256", "0" * 64),
        ("package", "not-pact-core"),
        ("version", "99.0"),
    ],
)
def test_vendored_pact_wheel_rejects_provenance_or_metadata_drift(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    provenance_path, _wheel = _copy_vendored_input(tmp_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance[field] = value
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    result = _run_helper(tmp_path)

    assert result.returncode != 0
    assert result.stdout == ""


def test_vendored_pact_wheel_rejects_tampered_file(tmp_path: Path) -> None:
    _provenance_path, wheel = _copy_vendored_input(tmp_path)
    with wheel.open("ab") as stream:
        stream.write(b"tampered")

    result = _run_helper(tmp_path)

    assert result.returncode != 0
    assert result.stdout == ""


def test_project_uses_vendored_wheel_as_uv_source() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"pact-core==0.2.0.dev0"' in pyproject
    assert 'pact-core = { path = "vendor/wheels/pact_core-0.2.0.dev0-py3-none-any.whl" }' in pyproject
    assert "pact-core @ git+" not in pyproject


def test_offline_scripts_validate_and_install_vendored_wheel() -> None:
    shell = (REPO_ROOT / "scripts" / "build_offline_dist.sh").read_text(encoding="utf-8")
    powershell = (REPO_ROOT / "scripts" / "build_offline_dist.ps1").read_text(encoding="utf-8")

    assert "--no-emit-package pact-core" in shell
    assert '"$(native_path "$SCRIPT_DIR/vendored_pact_wheel.py")"' in shell
    assert 'INSTALL_ARGS=(--python "$(native_path "$PY")" --require-hashes --compile-bytecode)' in shell
    assert '"$(native_path "$PACT_WHEEL")" --quiet' in shell
    assert "locked_git_requirement.py" not in shell

    assert '"--no-emit-package", "pact-core"' in powershell
    assert 'Join-Path $ScriptDir "vendored_pact_wheel.py"' in powershell
    assert "uv pip install --python $Py --require-hashes --compile-bytecode -r $Requirements" in powershell
    assert "uv pip install --python $Py --no-deps --compile-bytecode $PactWheel" in powershell
    assert "locked_git_requirement.py" not in powershell
