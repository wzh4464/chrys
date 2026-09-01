# Copyright (c) 2026 Chrys. All rights reserved.

"""Prepare and optionally run the fixed-P0 repair diagnostic."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from evaluation.requirement_clarification.protocol import (
    CONTROL_ARM,
    MODEL_PROFILE_ID,
    REPAIR_ARM,
    fingerprint_dataset,
    fingerprint_file,
    fingerprints_as_dict,
    read_secrets_env,
    render_paired_agent_profiles,
    validate_run_id,
    write_json,
)
from evaluation.requirement_clarification.run_pair import _git_revision, _harbor_binary, build_harbor_command


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
    harbor_binary = _harbor_binary(harbor_repo)
    revision = args.chrys_revision.strip() if args.chrys_revision else _git_revision(repo_root)
    if not revision:
        raise ValueError("--chrys-revision cannot be empty")
    profiles = render_paired_agent_profiles(
        repo_root / "src/chrys/service/profiles/agents/builtins/Code.yaml",
        output_dir / "config/agents",
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
        agent_profile=profiles[CONTROL_ARM],
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
            "agent_profile": asdict(fingerprint_file(profiles[CONTROL_ARM])),
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
