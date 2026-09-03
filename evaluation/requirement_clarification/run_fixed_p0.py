# Copyright (c) 2026 Chrys. All rights reserved.

"""Prepare and optionally run the fixed-P0 repair diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluation.requirement_clarification.protocol import (
    CLARIFICATION_STRATEGIES,
    DEFAULT_CLARIFICATION_STRATEGY,
    IMPORTED_P0_CLARIFICATION_ARM,
    MODEL_PROFILE_ID,
    REPAIR_ARM,
    fingerprint_dataset,
    fingerprint_file,
    fingerprints_as_dict,
    read_secrets_env,
    render_fixed_p0_repair_profile,
    render_imported_p0_clarification_profile,
    sha256_file,
    validate_run_id,
    write_json,
)
from evaluation.requirement_clarification.run_pair import (
    _assert_job_is_resumable,
    _git_revision,
    _harbor_binary,
    build_harbor_command,
)
from evaluation.requirement_clarification.summarize import load_selected_attempts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harbor-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset emitted by materialize_fixed_p0")
    parser.add_argument("--materialization-manifest", type=Path, required=True)
    parser.add_argument("--chrys-binary", type=Path, required=True)
    parser.add_argument(
        "--chrys-revision",
        help="Revision used to build --chrys-binary; defaults to the current repository HEAD",
    )
    parser.add_argument("--secrets", type=Path, default=Path(".chrys-secrets.env"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-tasks", type=int)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--recover-interrupted",
        action="store_true",
        help="Archive verifier-incomplete orphan trials before resuming an interrupted Harbor controller",
    )
    parser.add_argument("--clarification-only", action="store_true", help="Persist clarification without repair")
    parser.add_argument(
        "--clarification-strategy",
        choices=CLARIFICATION_STRATEGIES,
        default=DEFAULT_CLARIFICATION_STRATEGY,
        help="Clarification protocol used for imported-P0 runs",
    )
    parser.add_argument("--execute", action="store_true", help="Actually invoke Harbor and OpenRouter")
    return parser


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"materialization manifest {name} must be an object")
    return value


def _validate_materialized_dataset(dataset: Path, manifest_path: Path, task_names: set[str]) -> bool:
    """Reject stale or unsafe fixed-P0 inputs before any model call."""
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "root")
    protocol = manifest.get("protocol")
    if protocol not in {
        "chrys-deepswe-fixed-p0-repair-v1",
        "chrys-deepswe-imported-p0-clarification-v1",
    }:
        raise ValueError("not a fixed-P0 materialization manifest")
    native_clarification = protocol == "chrys-deepswe-imported-p0-clarification-v1"
    manifest_tasks = _mapping(manifest.get("tasks"), "tasks")
    if not task_names <= set(manifest_tasks):
        raise ValueError("materialization manifest does not contain every repair dataset task")
    eligible_count = manifest.get("eligible_count")
    if not isinstance(eligible_count, int) or eligible_count < len(task_names):
        raise ValueError("materialization manifest eligible_count is smaller than the repair dataset")

    contexts_root = manifest_path.parent / "image-contexts"
    for task in sorted(task_names):
        metadata = _mapping(manifest_tasks[task], f"tasks.{task}")
        task_toml_path = dataset / task / "task.toml"
        config = tomllib.loads(task_toml_path.read_text(encoding="utf-8"))
        if config.get("artifacts"):
            raise ValueError(f"fixed-P0 task {task} must not declare top-level artifacts")
        verifier = _mapping(config.get("verifier"), f"dataset.{task}.verifier")
        if verifier.get("environment_mode") != "separate":
            raise ValueError(f"fixed-P0 task {task} must use a separate verifier environment")
        for hook in verifier.get("collect", []):
            command = hook.get("command") if isinstance(hook, dict) else None
            if isinstance(command, str) and "/logs/artifacts/model.patch" in command:
                raise ValueError(f"fixed-P0 task {task} has a collect hook that overwrites model.patch")
        environment = _mapping(config.get("environment"), f"dataset.{task}.environment")
        if environment.get("docker_image") != metadata.get("fixed_p0_image"):
            raise ValueError(f"fixed-P0 task {task} image does not match its materialization manifest")
        control_patch = contexts_root / task / "model.patch"
        if not control_patch.is_file() or sha256_file(control_patch) != metadata.get("control_patch_sha256"):
            raise ValueError(f"fixed-P0 task {task} control patch is missing or changed")
        if native_clarification and metadata.get("delta_sha256") is not None:
            raise ValueError(f"imported-P0 task {task} must not contain a precomputed delta")
    return native_clarification


def _validate_collected_patches(
    job_dir: Path,
    task_names: set[str],
    *,
    allow_empty: bool = False,
) -> dict[str, dict[str, object]]:
    """Require one non-empty convention artifact for every completed repair task."""
    attempts = load_selected_attempts(job_dir)
    if set(attempts) != task_names:
        raise RuntimeError("completed fixed-P0 job tasks do not match the prepared dataset")
    patches: dict[str, dict[str, object]] = {}
    for task in sorted(task_names):
        trial_dir, _, _ = attempts[task]
        candidates = (
            trial_dir / "artifacts/logs/artifacts/model.patch",
            trial_dir / "artifacts/model.patch",
        )
        patch = next(
            (path for path in candidates if path.is_file() and (allow_empty or path.stat().st_size > 0)),
            None,
        )
        if patch is None:
            expected = "model.patch" if allow_empty else "non-empty model.patch"
            raise RuntimeError(f"fixed-P0 task {task} did not publish a {expected}")
        patches[task] = {
            "path": str(patch),
            "size": patch.stat().st_size,
            "sha256": sha256_file(patch),
        }
    return patches


def _clarification_health(job_dir: Path, task_names: set[str]) -> dict[str, object]:
    """Detect missing, degraded, or evidence-free clarification results."""
    attempts = load_selected_attempts(job_dir)
    empty: list[str] = []
    nonempty: list[str] = []
    missing: list[str] = []
    completed: list[str] = []
    degraded: list[str] = []
    pact_generated: list[str] = []
    pact_failed_or_missing: list[str] = []
    empty_reasons: dict[str, str] = {}
    low_coverage_empty: list[str] = []
    empty_after_coverage: list[str] = []
    completed_investigation_count = 0
    failed_investigation_count = 0
    sufficient_coverage_count = 0
    insufficient_coverage_count = 0
    successful_tool_call_count = 0
    inspected_files: set[tuple[str, str]] = set()
    proposer_input_token_counts: list[int] = []
    proposer_output_token_counts: list[int] = []
    for task in sorted(task_names):
        attempt = attempts.get(task)
        if attempt is None:
            missing.append(task)
            continue
        trial_dir, _, _ = attempt
        results = sorted(
            (trial_dir / "agent/chrys-sessions").glob("**/requirement_clarification/turn_*/clarification.private.json")
        )
        if not results:
            missing.append(task)
            continue
        value = _mapping(json.loads(results[-1].read_text(encoding="utf-8")), f"clarification result for {task}")
        investigations = value.get("investigations", [])
        task_sufficient_coverage = 0
        if isinstance(investigations, list):
            for item in investigations:
                if not isinstance(item, dict):
                    insufficient_coverage_count += 1
                    continue
                if item.get("status") == "completed":
                    completed_investigation_count += 1
                else:
                    failed_investigation_count += 1
                input_tokens = item.get("input_token_count")
                if isinstance(input_tokens, int) and not isinstance(input_tokens, bool):
                    proposer_input_token_counts.append(max(0, input_tokens))
                output_tokens = item.get("output_token_count")
                if isinstance(output_tokens, int) and not isinstance(output_tokens, bool):
                    proposer_output_token_counts.append(max(0, output_tokens))
                calls = item.get("tool_calls", [])
                successful_names: set[str] = set()
                if isinstance(calls, list):
                    for call in calls:
                        if not isinstance(call, dict) or call.get("successful") is not True:
                            continue
                        successful_tool_call_count += 1
                        name = call.get("name")
                        if isinstance(name, str):
                            successful_names.add(name)
                        path = call.get("path")
                        if name == "read_file" and isinstance(path, str) and path.strip():
                            inspected_files.add((task, path.strip()))
                coverage_status = item.get("coverage_status")
                if coverage_status is None and {"grep", "read_file"}.issubset(successful_names):
                    coverage_status = "sufficient"
                if coverage_status == "sufficient" and item.get("status") == "completed":
                    sufficient_coverage_count += 1
                    task_sufficient_coverage += 1
                else:
                    insufficient_coverage_count += 1
        pact_generation_path = results[-1].parent / "06-pact-input/generation.private.json"
        if pact_generation_path.is_file():
            pact_generation = _mapping(
                json.loads(pact_generation_path.read_text(encoding="utf-8")),
                f"PACT generation result for {task}",
            )
            if pact_generation.get("status") == "generated":
                pact_generated.append(task)
            else:
                pact_failed_or_missing.append(task)
        else:
            pact_failed_or_missing.append(task)
        clarification_status = value.get("status", "completed")
        if clarification_status == "degraded":
            degraded.append(task)
        else:
            completed.append(task)
        empty_reason = value.get("empty_reason")
        if isinstance(empty_reason, str):
            empty_reasons[task] = empty_reason
        delta = value.get("delta")
        if not isinstance(delta, str):
            missing.append(task)
        elif delta.strip():
            nonempty.append(task)
        else:
            empty.append(task)
            if task_sufficient_coverage:
                empty_after_coverage.append(task)
            else:
                low_coverage_empty.append(task)

    all_empty = len(task_names) > 1 and not missing and len(empty) == len(task_names)
    warnings: list[str] = []
    if all_empty and not low_coverage_empty:
        warnings.append("all tasks produced empty deltas after sufficient investigation coverage")
    if missing:
        status = "invalid_missing_results"
    elif degraded:
        status = "invalid_degraded_results"
    elif pact_failed_or_missing:
        status = "invalid_pact_results"
    elif low_coverage_empty:
        status = "invalid_proposer_collapse"
    else:
        status = "ok"
    return {
        "status": status,
        "task_count": len(task_names),
        "empty_delta_count": len(empty),
        "nonempty_delta_count": len(nonempty),
        "missing_result_count": len(missing),
        "completed_count": len(completed),
        "degraded_count": len(degraded),
        "empty_delta_tasks": empty,
        "nonempty_delta_tasks": nonempty,
        "missing_result_tasks": missing,
        "completed_tasks": completed,
        "degraded_tasks": degraded,
        "pact_generated_count": len(pact_generated),
        "pact_generated_tasks": pact_generated,
        "pact_failed_or_missing_count": len(pact_failed_or_missing),
        "pact_failed_or_missing_tasks": pact_failed_or_missing,
        "empty_reasons": empty_reasons,
        "completed_investigation_count": completed_investigation_count,
        "failed_investigation_count": failed_investigation_count,
        "sufficient_coverage_count": sufficient_coverage_count,
        "insufficient_coverage_count": insufficient_coverage_count,
        "successful_tool_call_count": successful_tool_call_count,
        "unique_inspected_file_count": len(inspected_files),
        "low_coverage_empty_count": len(low_coverage_empty),
        "low_coverage_empty_tasks": low_coverage_empty,
        "empty_after_coverage_count": len(empty_after_coverage),
        "empty_after_coverage_tasks": empty_after_coverage,
        "proposer_input_token_counts": proposer_input_token_counts,
        "proposer_input_token_median": (
            statistics.median(proposer_input_token_counts) if proposer_input_token_counts else None
        ),
        "proposer_output_token_counts": proposer_output_token_counts,
        "proposer_output_token_median": (
            statistics.median(proposer_output_token_counts) if proposer_output_token_counts else None
        ),
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    """Freeze repair inputs; execute only after an explicit flag."""
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    harbor_repo = args.harbor_repo.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    materialization_manifest = args.materialization_manifest.resolve(strict=True)
    chrys_binary = args.chrys_binary.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    run_id = validate_run_id(args.run_id)
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")

    secrets = read_secrets_env(args.secrets.resolve(strict=True))
    tasks = fingerprint_dataset(dataset, expected_tasks=args.expected_tasks)
    native_clarification = _validate_materialized_dataset(
        dataset,
        materialization_manifest,
        {task.name for task in tasks},
    )
    harbor_binary = _harbor_binary(harbor_repo)
    revision = args.chrys_revision.strip() if args.chrys_revision else _git_revision(repo_root)
    if not revision:
        raise ValueError("--chrys-revision cannot be empty")
    code_profile = repo_root / "src/chrys/service/profiles/agents/builtins/Code.yaml"
    if native_clarification:
        arm = IMPORTED_P0_CLARIFICATION_ARM
        agent_profile = render_imported_p0_clarification_profile(
            code_profile,
            output_dir / "config/agents/imported-p0-clarification.yaml",
            clarification_only=args.clarification_only,
            strategy=args.clarification_strategy,
        )
    else:
        arm = REPAIR_ARM
        agent_profile = render_fixed_p0_repair_profile(
            code_profile,
            output_dir / "config/agents/fixed-p0-repair.yaml",
        )
    model_profile = repo_root / "evaluation/requirement_clarification/profiles" / f"{MODEL_PROFILE_ID}.yaml"
    jobs_dir = output_dir / "jobs"
    job_name = f"{run_id}-{arm}"
    command = build_harbor_command(
        harbor_binary=harbor_binary,
        dataset=dataset,
        jobs_dir=jobs_dir,
        job_name=job_name,
        chrys_binary=chrys_binary,
        agent_profile=agent_profile,
        model_profile=model_profile,
        arm=arm,
        revision=revision,
        concurrency=args.concurrency,
    )
    manifest = {
        "schema_version": 1,
        "protocol": (
            "chrys-deepswe-imported-p0-clarification-v1" if native_clarification else "chrys-deepswe-fixed-p0-repair-v1"
        ),
        "run_id": run_id,
        "chrys_revision": revision,
        "dataset": str(dataset),
        "tasks": fingerprints_as_dict(tasks),
        "task_count": len(tasks),
        "concurrency": args.concurrency,
        "clarification_only": args.clarification_only,
        "clarification_strategy": args.clarification_strategy if native_clarification else None,
        "inputs": {
            "materialization_manifest": asdict(fingerprint_file(materialization_manifest)),
            "chrys_binary": asdict(fingerprint_file(chrys_binary)),
            "harbor_binary": asdict(fingerprint_file(harbor_binary)),
            "model_profile": asdict(fingerprint_file(model_profile)),
            "agent_profile": asdict(fingerprint_file(agent_profile)),
        },
        "command": command,
        "secrets": ["OPENROUTER_API_KEY", "CHRYS_MODEL_LOCK"],
    }
    write_json(output_dir / "manifest.json", manifest)
    sys.stdout.write(f"Prepared fixed-P0 repair run with {len(tasks)} tasks in {output_dir}\n")
    if not args.execute:
        sys.stdout.write("Dry run only; pass --execute to invoke Harbor and OpenRouter.\n")
        return 0

    environment = os.environ.copy()
    environment.update(secrets)
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(repo_root) if not python_path else f"{repo_root}{os.pathsep}{python_path}"
    job_dir = jobs_dir / job_name
    if job_dir.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite existing Harbor job: {job_dir}")
        archived = _assert_job_is_resumable(
            job_dir,
            recover_interrupted=args.recover_interrupted,
            recovery_root=output_dir / "recoveries",
        )
        if archived:
            sys.stdout.write(f"Archived {len(archived)} interrupted trial(s) before resume.\n")
        command = [str(harbor_binary), "jobs", "resume", "--job-path", str(job_dir)]
    subprocess.run(command, cwd=harbor_repo, env=environment, check=True)  # noqa: S603
    patch_records = _validate_collected_patches(
        job_dir,
        {task.name for task in tasks},
        allow_empty=native_clarification,
    )
    postflight: dict[str, object] = {"schema_version": 1, "patches": patch_records}
    if native_clarification:
        health = _clarification_health(job_dir, {task.name for task in tasks})
        postflight["clarification_health"] = health
    write_json(output_dir / "postflight.json", postflight)
    if native_clarification and health["status"] != "ok":
        raise RuntimeError(
            f"clarification batch health is {health['status']}; refusing to accept incomplete or collapsed results"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
