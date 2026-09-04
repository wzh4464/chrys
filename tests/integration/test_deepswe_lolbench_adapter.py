# Copyright (c) 2026 Chrys. All rights reserved.

"""The DeepSWE → LoLBench adapter materializes a benchmark root the engine can run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from evaluation.deepswe.lolbench.gen_instances import image_tag
from evaluation.deepswe.lolbench.gen_instances import main as gen_main
from evaluation.deepswe.verify import verify_command_for


def _task(tasks: Path, task_id: str, *, language: str, image: str = "example.invalid/img:1") -> Path:
    task = tasks / task_id
    (task / "tests").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "task.toml").write_text(
        "[task]\n"
        f'name = "datacurve/{task_id}"\n'
        "[metadata]\n"
        f'task_id = "{task_id}"\n'
        f'language = "{language}"\n'
        'repository_url = "https://example.invalid/r"\n'
        'base_commit_hash = "abc123"\n'
        "[verifier]\n"
        'network_mode = "no-network"\n'
        "timeout_sec = 1800.0\n"
        "[verifier.environment]\n"
        "cpus = 2\n"
        "memory_mb = 8192\n"
        "[agent]\n"
        "timeout_sec = 5400.0\n"
        "[environment]\n"
        f'docker_image = "{image}"\n',
        encoding="utf-8",
    )
    (task / "instruction.md").write_text(f"# do {task_id}\n", encoding="utf-8")
    (task / "tests" / "test.patch").write_text("diff --git a/t b/t\n", encoding="utf-8")
    (task / "solution" / "solution.patch").write_text("diff --git a/s b/s\n", encoding="utf-8")
    return task


def test_the_root_follows_the_manifest_and_carries_a_verify_command(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    _task(tasks, "zeta-go-task", language="go")
    _task(tasks, "alpha-py-task", language="python")
    _task(tasks, "mid-cobol-task", language="cobol")
    (tasks / "manifest.json").write_text(
        json.dumps(
            {"tasks": [{"task_id": "zeta-go-task"}, {"task_id": "mid-cobol-task"}, {"task_id": "alpha-py-task"}]}
        ),
        encoding="utf-8",
    )
    out = tmp_path / "bench"

    assert gen_main(["--tasks-dir", str(tasks), "--out", str(out), "--limit", "2"]) == 0

    with (out / "data" / "lolbench_clean.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # Manifest order, not alphabetical; --limit 2 keeps the first two of it.
    assert [r["instance_id"] for r in rows] == ["zeta-go-task", "mid-cobol-task"]
    assert rows[0] == {
        "instance_id": "zeta-go-task",
        "docker_dir": "zeta-go-task",
        "source_mapping": "zeta-go-task.md",
        "base_commit": "abc123",
    }

    go = (out / "dockers" / "zeta-go-task" / "Dockerfile").read_text(encoding="utf-8")
    assert go.startswith("# DeepSWE task zeta-go-task")
    assert "FROM example.invalid/img:1\n" in go
    assert 'ENV CHRYS_PACT_VERIFY_COMMAND="go test ./..."' in go
    cobol = (out / "dockers" / "mid-cobol-task" / "Dockerfile").read_text(encoding="utf-8")
    assert "CHRYS_PACT_VERIFY_COMMAND" not in cobol  # no verify command for that language: no campaign

    spec = json.loads((out / "dockers" / "zeta-go-task" / "spec.json").read_text(encoding="utf-8"))
    assert spec["verify_command"] == "go test ./..."
    assert spec["timeout_s"] == 5400
    assert spec["verifier_timeout_s"] == 1800
    assert (out / "dockers" / "zeta-go-task" / "eval_tests.patch").read_text(encoding="utf-8") == "diff --git a/t b/t\n"
    assert (out / "instructions" / "original" / "zeta-go-task.md").read_text(encoding="utf-8") == "# do zeta-go-task\n"
    readme = (out / "dockers" / "zeta-go-task" / "README.md").read_text(encoding="utf-8")
    assert "<!-- LOLBENCH_BUILD_BEGIN -->" in readme
    assert f"docker build -t {image_tag('zeta-go-task', go)} ." in readme


def test_the_wrapper_tag_changes_with_its_recipe() -> None:
    one = image_tag("task", "FROM a\n")
    two = image_tag("task", "FROM a\nENV X=1\n")

    assert one != two
    assert one.startswith("lolbench-deepswe-") and one.endswith(":1")
    assert "-base" not in one and "-coverage" not in one


def test_verify_commands_by_language() -> None:
    assert verify_command_for("Go") == "go test ./..."
    assert verify_command_for("python") == "python -m pytest -q -x -p no:cacheprovider"
    assert verify_command_for("") == ""
    assert verify_command_for(None) == ""
    assert verify_command_for("cobol") == ""
