# Copyright (c) 2026 Chrys. All rights reserved.

"""The DeepSWE → LoLBench adapter materializes a benchmark root the engine can run."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from evaluation.deepswe.lolbench.gen_instances import image_tag
from evaluation.deepswe.lolbench.gen_instances import main as gen_main
from evaluation.deepswe.verify import (
    VERIFY_COMMAND,
    hidden_runner_script,
    sanitize_runner,
    verify_command_for,
    verify_from_test_sh,
)


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
    assert f'ENV CHRYS_PACT_VERIFY_COMMAND="{VERIFY_COMMAND}"' in go
    assert "COPY verify_base.sh /opt/deepswe_verify.sh" in go
    script = (out / "dockers" / "zeta-go-task" / "verify_base.sh").read_text(encoding="utf-8")
    assert script.endswith("set -e\ngo test ./...\n")  # no runner in the hidden patch: the language default
    cobol = (out / "dockers" / "mid-cobol-task" / "Dockerfile").read_text(encoding="utf-8")
    assert "CHRYS_PACT_VERIFY_COMMAND" not in cobol  # no verify command for that language: no campaign
    assert not (out / "dockers" / "mid-cobol-task" / "verify_base.sh").exists()

    spec = json.loads((out / "dockers" / "zeta-go-task" / "spec.json").read_text(encoding="utf-8"))
    assert spec["verify_command"] == VERIFY_COMMAND
    assert spec["timeout_s"] == 5400
    assert spec["verifier_timeout_s"] == 1800
    assert (out / "dockers" / "zeta-go-task" / "eval_tests.patch").read_text(encoding="utf-8") == "diff --git a/t b/t\n"
    assert (out / "instructions" / "original" / "zeta-go-task.md").read_text(encoding="utf-8") == "# do zeta-go-task\n"
    readme = (out / "dockers" / "zeta-go-task" / "README.md").read_text(encoding="utf-8")
    assert "<!-- LOLBENCH_BUILD_BEGIN -->" in readme
    assert f"docker build -t {image_tag('zeta-go-task', go, script)} ." in readme


def test_the_wrapper_tag_changes_with_its_recipe() -> None:
    one = image_tag("task", "FROM a\n")
    two = image_tag("task", "FROM a\nENV X=1\n")

    assert one != two
    assert image_tag("task", "FROM a\n", "pytest\n") not in (one, image_tag("task", "FROM a\n", "go test\n"))
    assert one.startswith("lolbench-deepswe-") and one.endswith(":1")
    assert "-base" not in one and "-coverage" not in one


def test_verify_commands_by_language() -> None:
    assert verify_command_for("Go") == "go test ./..."
    assert verify_command_for("python") == "python -m pytest -q -x -p no:cacheprovider"
    assert verify_command_for("") == ""
    assert verify_command_for(None) == ""
    assert verify_command_for("cobol") == ""


_SUPERJSON_PATCH = """diff --git a/src/error-stack.test.ts b/src/error-stack.test.ts
new file mode 100644
--- /dev/null
+++ b/src/error-stack.test.ts
@@ -0,0 +1,2 @@
+import { it } from 'vitest';
+it('works', () => {});
diff --git a/test.sh b/test.sh
new file mode 100755
--- /dev/null
+++ b/test.sh
@@ -0,0 +1,7 @@
+#!/usr/bin/env bash
+set -euo pipefail
+MODE="${1:-base}"
+case "$MODE" in
+  base) npx vitest run -t '^(?!.*perf)' src/index.test.ts ;;
+  new) npx vitest run src/error-stack.test.ts ;;
+esac
"""


def test_the_hidden_runner_is_shipped_in_base_mode_without_the_hidden_test_name() -> None:
    runner = hidden_runner_script(_SUPERJSON_PATCH)
    assert runner is not None
    assert runner.startswith("#!/usr/bin/env bash\n")
    sanitized = sanitize_runner(runner, ["src/error-stack.test.ts", "test.sh"])
    assert "error-stack" not in sanitized
    assert "base) npx vitest run -t '^(?!.*perf)' src/index.test.ts ;;" in sanitized
    assert "new) npx vitest run src/__hidden__.test.ts ;;" in sanitized
    assert hidden_runner_script("diff --git a/t b/t\n") is None


_VERIFIER = """set +e
npm run build > /logs/verifier/build.log 2>&1
gate_rc=$?
PYTEST_ADDOPTS="-p no:cacheprovider --junitxml=/logs/verifier/base.xml" python -m pytest tests/ \\
  --ignore=tests/test_new.py -q > /logs/verifier/base.log 2>&1
python -m pytest tests/test_new.py -q --junitxml=/logs/verifier/new.xml > /logs/verifier/new.log 2>&1
go test -json -count=1 ./pkg/ 2>>"$RUN_LOG" | tee -a "$RUN_LOG" | go-ctrf-json-reporter -output /logs/verifier/base-ctrf.json
junit-to-ctrf /logs/verifier/base.xml -o /logs/verifier/base-ctrf.json
set -e
"""


def test_the_verifier_base_run_is_derived_without_reporters_or_the_hidden_test() -> None:
    patch = "diff --git a/tests/test_new.py b/tests/test_new.py\n+++ b/tests/test_new.py\n"
    command = verify_from_test_sh(_VERIFIER, patch, "python")
    assert command == (
        "npm run build && "
        'PYTEST_ADDOPTS="-p no:cacheprovider" python -m pytest tests/ --ignore=tests/test_new.py -q && '
        "go test -count=1 ./pkg/"
    )
    assert verify_from_test_sh("echo nothing here\n", "", "go") == "go test ./..."
