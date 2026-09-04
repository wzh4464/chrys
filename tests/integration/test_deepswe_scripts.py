# Copyright (c) 2026 Chrys. All rights reserved.

"""Offline contracts for the DeepSWE runner and verifier."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import scripts.deepswe_runner as runner
import scripts.deepswe_verify as verifier


def test_runner_selects_alphabetically_before_offset_and_limit() -> None:
    tasks = [{"task_id": "charlie"}, {"task_id": "alpha"}, {"task_id": "bravo"}]

    selected = runner._select_tasks(tasks, order="alphabetical", offset=1, limit=1)

    assert [task["task_id"] for task in selected] == ["bravo"]
    assert [task["task_id"] for task in tasks] == ["charlie", "alpha", "bravo"]


def test_runner_forwards_localization_timeout_and_supports_native_enrichment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner._run_locate(
        ["chrys"],
        {},
        tmp_path / "repo",
        tmp_path / "PROMPT.md",
        tmp_path / "artifacts",
        mode="llm",
        model_profile="model",
        timeout=900,
    )
    runner._run_agent(
        ["chrys"],
        {},
        tmp_path / "repo",
        tmp_path / "PROMPT.md",
        None,
        agent="DeepSWESmoke",
        model="model",
        timeout=3600,
    )

    locate_command, locate_kwargs = calls[0]
    assert locate_command[locate_command.index("--timeout") + 1] == "900"
    assert locate_kwargs["timeout"] == 930
    assert locate_kwargs["stdin"] is subprocess.DEVNULL
    agent_command, agent_kwargs = calls[1]
    assert "--localization-file" not in agent_command
    assert agent_kwargs["stdin"] is subprocess.DEVNULL


def test_verifier_alphabetical_selection_matches_runner(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "manifest.json").write_text(
        '{"tasks": [{"task_id": "charlie"}, {"task_id": "alpha"}, {"task_id": "bravo"}]}',
        encoding="utf-8",
    )

    selected = verifier._manifest_tasks(tmp_path, limit=1, offset=1, order="alphabetical")

    assert [task["task_id"] for task in selected] == ["bravo"]
