# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for delegation to ContextGraph's repository writer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from chrys.service.memory import contextgraph_repository as repository


def test_deposit_normalizes_redacts_and_derives_stable_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_writer(payload: dict[str, object]) -> repository.RepositoryDepositResult:
        captured.append(payload)
        return repository.RepositoryDepositResult(
            trajectory_id=str(payload["trajectory_id"]),
            fragment_count=1,
            created=True,
        )

    monkeypatch.setattr(repository, "_run_repository_writer", fake_writer)
    kwargs = {
        "problem_statement": "Fix login with sk-proj-12345678901234567890",
        "success": True,
        "steps": [{"action": "run tests", "observation": "Bearer abcdefghijklmnop"}],
        "final_response": "all tests pass",
        "repo": "chrys",
        "source_id": "session:turn:digest",
    }

    first = repository.deposit_experience(**kwargs)
    second = repository.deposit_experience(**kwargs)

    assert first is not None and second is not None
    assert captured[0]["trajectory_id"] == captured[1]["trajectory_id"]
    assert str(captured[0]["trajectory_id"]).startswith("traj_chrys_")
    assert str(captured[0]["instance_id"]).startswith("chrys:")
    assert "sk-proj" not in str(captured[0]["problem_statement"])
    assert "Bearer" not in str(captured[0]["steps"])
    assert "Assistant outcome: all tests pass" in str(captured[0]["steps"])


def test_empty_record_is_rejected_before_repository_access(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        repository,
        "_run_repository_writer",
        lambda _payload: (_ for _ in ()).throw(AssertionError("writer must not run")),
    )

    assert repository.deposit_experience(problem_statement="answer only", success=True, steps=[]) is None
    assert (
        repository.record_manual(problem_statement="answer only", success=True, steps=[], repo=None)
        == "No executable experience steps to record."
    )


def test_repository_writer_uses_contextgraph_runtime_and_stdin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "agent_memory").mkdir()
    (tmp_path / "agent_memory" / "memory.py").write_text("", encoding="utf-8")
    interpreter = tmp_path / "python"
    interpreter.write_text("", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["argv"] = argv
        seen.update(kwargs)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps({"trajectory_id": "traj_chrys_test", "fragment_count": 2, "created": True}),
            stderr="",
        )

    monkeypatch.setenv("CONTEXTGRAPH_REPO", str(tmp_path))
    monkeypatch.setenv("CONTEXTGRAPH_PYTHON", str(interpreter))
    monkeypatch.setattr(repository.subprocess, "run", fake_run)

    result = repository._run_repository_writer({"trajectory_id": "traj_chrys_test"})

    assert result == repository.RepositoryDepositResult("traj_chrys_test", 2, True)
    assert seen["argv"] == [
        str(interpreter),
        str(Path(repository.__file__).with_name("_contextgraph_repository_worker.py")),
    ]
    assert seen["cwd"] == tmp_path
    assert seen["input"] == '{"trajectory_id": "traj_chrys_test"}'
    assert seen["check"] is False


def test_repository_writer_surfaces_bounded_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "agent_memory").mkdir()
    (tmp_path / "agent_memory" / "memory.py").write_text("", encoding="utf-8")
    interpreter = tmp_path / "python"
    interpreter.write_text("", encoding="utf-8")
    monkeypatch.setenv("CONTEXTGRAPH_REPO", str(tmp_path))
    monkeypatch.setenv("CONTEXTGRAPH_PYTHON", str(interpreter))
    monkeypatch.setattr(
        repository.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, stdout="", stderr="worker exploded"),
    )

    with pytest.raises(RuntimeError, match="worker exploded"):
        repository._run_repository_writer({"trajectory_id": "traj_chrys_test"})


def test_a_secret_at_the_budget_boundary_is_still_redacted() -> None:
    """Truncating first would cut a key below its pattern's length floor.

    The remaining prefix stops matching and gets written to the graph verbatim,
    which is exactly the outcome redaction exists to prevent.
    """
    from chrys.service.memory.contextgraph_repository import MAX_REPOSITORY_STEP_CHARS, _redact

    key = "sk-proj-" + "a" * 40
    observation = "x" * (MAX_REPOSITORY_STEP_CHARS - 10) + key

    redacted = _redact(observation, limit=MAX_REPOSITORY_STEP_CHARS)

    assert "sk-proj-" not in redacted
    assert len(redacted) <= MAX_REPOSITORY_STEP_CHARS


def test_redaction_still_honours_the_budget() -> None:
    from chrys.service.memory.contextgraph_repository import MAX_REPOSITORY_STEP_CHARS, _redact

    assert len(_redact("y" * 100_000, limit=MAX_REPOSITORY_STEP_CHARS)) == MAX_REPOSITORY_STEP_CHARS
