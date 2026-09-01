# Copyright (c) 2026 Chrys. All rights reserved.

"""Prepare and optionally run the fixed-P0 repair diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluation.requirement_clarification.protocol import (
    MODEL_PROFILE_ID,
    REPAIR_ARM,
    fingerprint_dataset,
    fingerprint_file,
    fingerprints_as_dict,
    read_secrets_env,
    render_fixed_p0_repair_profile,
    sha256_file,
    validate_run_id,
    write_json,
)
from evaluation.requirement_clarification.run_pair import _git_revision, _harbor_binary, build_harbor_command
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
    parser.add_argument("--execute", action="store_true", help="Actually invoke Harbor and OpenRouter")
    return parser


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"materialization manifest {name} must be an object")
    return value


def _validate_materialized_dataset(dataset: Path, manifest_path: Path, task_names: set[str]) -> None:
    """Reject stale or unsafe fixed-P0 inputs before any model call."""
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), "root")
    if manifest.get("protocol") != "chrys-deepswe-fixed-p0-repair-v1":
        raise ValueError("not a fixed-P0 materialization manifest")
    manifest_tasks = _mapping(manifest.get("tasks"), "tasks")
    if set(manifest_tasks) != task_names:
        raise ValueError("materialization manifest tasks do not match the repair dataset")
    if manifest.get("eligible_count") != len(task_names):
        raise ValueError("materialization manifest eligible_count does not match the repair dataset")

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


def _validate_collected_patches(job_dir: Path, task_names: set[str]) -> dict[str, dict[str, object]]:
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
        patch = next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)
        if patch is None:
            raise RuntimeError(f"fixed-P0 task {task} did not publish a non-empty model.patch")
        patches[task] = {
            "path": str(patch),
            "size": patch.stat().st_size,
            "sha256": sha256_file(patch),
        }
    return patches


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
    _validate_materialized_dataset(dataset, materialization_manifest, {task.name for task in tasks})
    harbor_binary = _harbor_binary(harbor_repo)
    revision = args.chrys_revision.strip() if args.chrys_revision else _git_revision(repo_root)
    if not revision:
        raise ValueError("--chrys-revision cannot be empty")
    agent_profile = render_fixed_p0_repair_profile(
        repo_root / "src/chrys/service/profiles/agents/builtins/Code.yaml",
        output_dir / "config/agents/fixed-p0-repair.yaml",
    )
    model_profile = repo_root / "evaluation/requirement_clarification/profiles" / f"{MODEL_PROFILE_ID}.yaml"
    jobs_dir = output_dir / "jobs"
    job_name = f"{run_id}-{REPAIR_ARM}"
    command = build_harbor_command(
        harbor_binary=harbor_binary,
        dataset=dataset,
        jobs_dir=jobs_dir,
        job_name=job_name,
        chrys_binary=chrys_binary,
        agent_profile=agent_profile,
        model_profile=model_profile,
        arm=REPAIR_ARM,
        revision=revision,
        concurrency=args.concurrency,
    )
    manifest = {
        "schema_version": 1,
        "protocol": "chrys-deepswe-fixed-p0-repair-v1",
        "run_id": run_id,
        "chrys_revision": revision,
        "dataset": str(dataset),
        "tasks": fingerprints_as_dict(tasks),
        "task_count": len(tasks),
        "concurrency": args.concurrency,
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
        command = [str(harbor_binary), "jobs", "resume", "--job-path", str(job_dir)]
    subprocess.run(command, cwd=harbor_repo, env=environment, check=True)  # noqa: S603
    patch_records = _validate_collected_patches(job_dir, {task.name for task in tasks})
    write_json(output_dir / "postflight.json", {"schema_version": 1, "patches": patch_records})
    return 0


if __name__ == "__main__":
    sys.exit(main())
