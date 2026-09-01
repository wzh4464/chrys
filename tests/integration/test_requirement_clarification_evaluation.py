# Copyright (c) 2026 Chrys. All rights reserved.

"""Offline coverage for the requirement-clarification evaluation protocol."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
import yaml
from evaluation.requirement_clarification import build_fixed_p0_images
from evaluation.requirement_clarification.materialize_fixed_p0 import main as materialize_fixed_p0
from evaluation.requirement_clarification.protocol import (
    CANDIDATE_ARM,
    CANDIDATE_PROFILE_NAME,
    CODING_PHASE_TIMEOUT_SECONDS,
    CONTROL_ARM,
    CONTROL_PROFILE_NAME,
    REPAIR_ARM,
    REPAIR_PROFILE_NAME,
    agent_profile_name,
    expected_model_lock,
    read_secrets_env,
    render_fixed_p0_repair_profile,
    render_paired_agent_profiles,
    sha256_file,
)
from evaluation.requirement_clarification.run_pair import _assert_job_is_resumable, _materialize_dataset
from evaluation.requirement_clarification.summarize import compare_jobs, summarize_job

from tests.support.paths import REPO_ROOT


def test_resume_rejects_job_with_running_trial(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "result.json").write_text(
        json.dumps({"stats": {"n_running_trials": 1}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="refusing to resume Harbor job with 1 running trial"):
        _assert_job_is_resumable(job)


def test_resume_allows_job_without_running_trials(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "result.json").write_text(
        json.dumps({"stats": {"n_running_trials": 0, "n_pending_trials": 1}}),
        encoding="utf-8",
    )

    assert _assert_job_is_resumable(job) == []


def test_recover_interrupted_archives_only_trials_without_verifier_results(tmp_path: Path) -> None:
    job = tmp_path / "job"
    recovery = tmp_path / "recoveries"
    job.mkdir()
    (job / "result.json").write_text(
        json.dumps({"stats": {"n_running_trials": 2}}),
        encoding="utf-8",
    )
    incomplete = job / "incomplete__trial"
    no_result = job / "no-result__trial"
    complete = job / "complete__trial"
    incomplete.mkdir()
    no_result.mkdir()
    complete.mkdir()
    (no_result / "config.json").write_text("{}", encoding="utf-8")
    (incomplete / "result.json").write_text(
        json.dumps({"verifier_result": None, "exception_info": {"exception_type": "CancelledError"}}),
        encoding="utf-8",
    )
    (complete / "result.json").write_text(
        json.dumps({"verifier_result": {"rewards": {"reward": 1}}, "exception_info": None}),
        encoding="utf-8",
    )

    archived = _assert_job_is_resumable(
        job,
        recover_interrupted=True,
        recovery_root=recovery,
    )

    assert archived == ["incomplete__trial", "no-result__trial"]
    assert not incomplete.exists()
    assert not no_result.exists()
    assert complete.exists()
    recovery_dirs = [path for path in recovery.iterdir() if path.is_dir()]
    assert len(recovery_dirs) == 1
    assert (recovery_dirs[0] / "incomplete__trial/result.json").is_file()
    recovery_record = json.loads((recovery_dirs[0] / "recovery.json").read_text(encoding="utf-8"))
    assert recovery_record["archived_trials"] == ["incomplete__trial", "no-result__trial"]


def test_materialized_dataset_widens_only_outer_agent_timeout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    task = source / "task-one"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("do it\n", encoding="utf-8")
    (task / "task.toml").write_text(
        "[verifier]\ntimeout_sec = 300.0\n[agent]\nnetwork_mode = \"no-network\"\ntimeout_sec = 5400.0\n",
        encoding="utf-8",
    )

    staged = _materialize_dataset(source, tmp_path / "staged", agent_timeout_seconds=12600)

    rendered = tomllib.loads((staged / "task-one/task.toml").read_text(encoding="utf-8"))
    assert rendered["verifier"]["timeout_sec"] == 300.0
    assert rendered["agent"]["timeout_sec"] == 12600


def test_rendered_profiles_are_a_strict_feature_flag_pair(tmp_path: Path) -> None:
    profiles = render_paired_agent_profiles(
        REPO_ROOT / "src/chrys/service/profiles/agents/builtins/Code.yaml",
        tmp_path,
    )

    control = yaml.safe_load(profiles[CONTROL_ARM].read_text(encoding="utf-8"))
    candidate = yaml.safe_load(profiles[CANDIDATE_ARM].read_text(encoding="utf-8"))

    timeouts = {
        "initial_timeout_seconds": CODING_PHASE_TIMEOUT_SECONDS,
        "repair_timeout_seconds": CODING_PHASE_TIMEOUT_SECONDS,
    }
    assert control["requirement_clarification"] == {"enabled": False, **timeouts}
    assert candidate["requirement_clarification"] == {"enabled": True, **timeouts}
    assert control["instructions"] == candidate["instructions"]
    assert control["tools"] == candidate["tools"]


def test_fixed_p0_repair_profile_is_bounded_and_incremental(tmp_path: Path) -> None:
    path = render_fixed_p0_repair_profile(
        REPO_ROOT / "src/chrys/service/profiles/agents/builtins/Code.yaml",
        tmp_path / "fixed-p0-repair.yaml",
    )

    profile = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert profile["name"] == "DeepSWEFixedP0Repair"
    assert profile["requirement_clarification"]["enabled"] is False
    assert profile["sub_agents"] == {"max_total_concurrency": 1, "agents": []}
    assert profile["tools"]["builtins"] == ["filesystem.write", "filesystem.read", "shell", "search"]
    assert "If P0 already satisfies every ΔR bullet, make no changes and finish immediately" in profile["instructions"]
    assert "Do not run exhaustive manual end-to-end matrices" in profile["instructions"]
    assert "skills" not in profile
    assert "memory" not in profile


@pytest.mark.parametrize(
    ("run_mode", "expected_profile"),
    [
        (CONTROL_ARM, CONTROL_PROFILE_NAME),
        (CANDIDATE_ARM, CANDIDATE_PROFILE_NAME),
        (REPAIR_ARM, REPAIR_PROFILE_NAME),
    ],
)
def test_harbor_adapter_selects_profile_for_each_run_mode(
    run_mode: str,
    expected_profile: str,
) -> None:
    assert agent_profile_name(run_mode) == expected_profile


def test_harbor_adapter_rejects_unknown_run_mode() -> None:
    with pytest.raises(ValueError, match="unsupported run_mode"):
        agent_profile_name("unknown")


def test_secrets_reader_enforces_and_normalizes_model_lock(tmp_path: Path) -> None:
    secrets = tmp_path / ".chrys-secrets.env"
    secrets.write_text(
        f"OPENROUTER_API_KEY=test-only\nCHRYS_MODEL_LOCK='{expected_model_lock()}'\nIGNORED=value\n",
        encoding="utf-8",
    )

    loaded = read_secrets_env(secrets)

    assert loaded == {
        "OPENROUTER_API_KEY": "test-only",
        "CHRYS_MODEL_LOCK": expected_model_lock(),
    }


def _write_trial(job: Path, task: str, *, reward: int, delta: str | None = None) -> Path:
    trial = job / f"{task}__trial"
    (trial / "artifacts").mkdir(parents=True)
    (trial / "artifacts/model.patch").write_text(f"patch for {task}\n", encoding="utf-8")
    result = {
        "task_name": f"datacurve/{task}",
        "trial_name": trial.name,
        "task_id": {"path": f"/dataset/{task}"},
        "agent_result": {"metadata": {"run_mode": "test"}},
        "verifier_result": {"rewards": {"reward": reward}},
        "exception_info": None,
        "finished_at": "2026-08-31T00:00:00Z",
    }
    (trial / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if delta is not None:
        artifact = trial / "agent/chrys-sessions/session/requirement_clarification/turn_1"
        artifact.mkdir(parents=True)
        (artifact / "clarification.private.json").write_text(json.dumps({"delta": delta}), encoding="utf-8")
        (artifact / "summary.json").write_text(json.dumps({"outcome": "repaired"}), encoding="utf-8")
    return trial


def test_summary_reports_strict_paired_flips(tmp_path: Path) -> None:
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    _write_trial(control, "gain", reward=0)
    _write_trial(control, "regression", reward=1)
    _write_trial(candidate, "gain", reward=1, delta="clarify gain")
    _write_trial(candidate, "regression", reward=0, delta="clarify regression")

    summary = summarize_job(candidate)
    comparison = compare_jobs(control, candidate)

    assert summary["solved_count"] == 1
    assert comparison["gains"] == ["gain"]
    assert comparison["regressions"] == ["regression"]
    assert comparison["net_solved_delta"] == 0
    assert comparison["mcnemar_exact_two_sided_p"] == 1.0


def test_fixed_p0_materialization_uses_control_patch_and_candidate_delta(tmp_path: Path) -> None:
    source = tmp_path / "source"
    task = source / "task-one"
    task.mkdir(parents=True)
    (task / "instruction.md").write_text("Original requirement.\n", encoding="utf-8")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n[environment]\ndocker_image = "example/source@sha256:abc"\n'
        '[agent]\nnetwork_mode = "no-network"\n',
        encoding="utf-8",
    )
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    control_trial = _write_trial(control, "task-one", reward=0)
    expected_patch = "fixed P0 patch\n"
    (control_trial / "artifacts/model.patch").write_text(expected_patch, encoding="utf-8")
    _write_trial(candidate, "task-one", reward=1, delta="Require stable compatibility.")
    output = tmp_path / "fixed-p0"

    result = materialize_fixed_p0(
        [
            "--source-dataset",
            str(source),
            "--control-job",
            str(control),
            "--candidate-job",
            str(candidate),
            "--output-dir",
            str(output),
            "--expected-tasks",
            "1",
            "--expected-eligible",
            "1",
        ]
    )

    assert result == 0
    assert (output / "image-contexts/task-one/model.patch").read_text(encoding="utf-8") == expected_patch
    instruction = (output / "dataset/task-one/instruction.md").read_text(encoding="utf-8")
    assert "Original requirement." in instruction
    assert "Require stable compatibility." in instruction
    rendered_task = tomllib.loads((output / "dataset/task-one/task.toml").read_text(encoding="utf-8"))
    assert rendered_task["environment"]["docker_image"] == "chrys/deepswe-fixed-p0:task-one"


def test_fixed_p0_materialization_accepts_collected_patch_and_external_delta(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for task_name in ("task-one", "task-two"):
        task = source / task_name
        task.mkdir(parents=True)
        (task / "instruction.md").write_text(f"Original {task_name}.\n", encoding="utf-8")
        (task / "task.toml").write_text(
            f'schema_version = "1.3"\n[environment]\ndocker_image = "example/{task_name}:base"\n',
            encoding="utf-8",
        )

    control = tmp_path / "control"
    control_trial = _write_trial(control, "task-one", reward=0)
    _write_trial(control, "task-two", reward=1)
    direct_patch = control_trial / "artifacts/model.patch"
    expected_patch = direct_patch.read_text(encoding="utf-8")
    collected_patch = control_trial / "artifacts/logs/artifacts/model.patch"
    collected_patch.parent.mkdir(parents=True)
    direct_patch.replace(collected_patch)

    clarification = tmp_path / "clarification"
    clarification_task = clarification / "task-one"
    clarification_task.mkdir(parents=True)
    (clarification_task / "clarified_requirement.md").write_text(
        "Original task-one.\n\nRepository implementation guidance:\n- Preserve the compatibility path.\n",
        encoding="utf-8",
    )
    output = tmp_path / "fixed-p0"

    result = materialize_fixed_p0(
        [
            "--source-dataset",
            str(source),
            "--control-job",
            str(control),
            "--clarification-root",
            str(clarification),
            "--output-dir",
            str(output),
            "--expected-tasks",
            "2",
            "--expected-eligible",
            "1",
            "--task",
            "task-one",
        ]
    )

    assert result == 0
    assert (output / "image-contexts/task-one/model.patch").read_text(encoding="utf-8") == expected_patch
    assert not (output / "dataset/task-two").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_job"] is None
    assert manifest["clarification_root"] == str(clarification.resolve())


def test_fixed_p0_builder_revalidates_reviewed_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = tmp_path / "image-contexts/task-one"
    context.mkdir(parents=True)
    patch = context / "model.patch"
    patch.write_text("reviewed patch\n", encoding="utf-8")
    (context / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    image = "chrys/deepswe-fixed-p0:task-one"
    base = "example/source@sha256:abc"
    command = ["docker", "build", "--build-arg", f"BASE_IMAGE={base}", "-t", image, str(context)]
    manifest = {
        "protocol": "chrys-deepswe-fixed-p0-repair-v1",
        "tasks": {
            "task-one": {
                "base_image": base,
                "fixed_p0_image": image,
                "control_patch_sha256": sha256_file(patch),
            }
        },
        "docker_commands": {"task-one": command},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    invoked: list[list[str]] = []

    def fake_which(command_name: str) -> str:
        assert command_name == "docker"
        return "/usr/bin/docker"

    def fake_run(command_parts: list[str], *, check: bool) -> None:
        assert check
        invoked.append(command_parts)

    monkeypatch.setattr(build_fixed_p0_images.shutil, "which", fake_which)
    monkeypatch.setattr(build_fixed_p0_images.subprocess, "run", fake_run)

    assert build_fixed_p0_images.main(["--manifest", str(manifest_path)]) == 0
    assert invoked == [["/usr/bin/docker", *command[1:]]]
