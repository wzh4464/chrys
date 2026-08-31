# Copyright (c) 2026 Chrys. All rights reserved.

"""Prepare and optionally execute paired DeepSWE control/candidate runs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from evaluation.requirement_clarification.protocol import (
    AGENT_IMPORT_PATH,
    ARMS,
    HARBOR_MODEL_NAME,
    MODEL_PROFILE_ID,
    OPENROUTER_HOST,
    fingerprint_dataset,
    fingerprint_file,
    fingerprints_as_dict,
    read_secrets_env,
    render_paired_agent_profiles,
    validate_run_id,
    write_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harbor-repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--chrys-binary", type=Path, required=True)
    parser.add_argument("--secrets", type=Path, default=Path(".chrys-secrets.env"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="Immutable label shared by both arms")
    parser.add_argument("--expected-tasks", type=int, default=113)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--arm", choices=ARMS, action="append", dest="arms")
    parser.add_argument("--resume", action="store_true", help="Resume existing Harbor jobs instead of overwriting")
    parser.add_argument("--execute", action="store_true", help="Actually invoke Harbor and OpenRouter")
    return parser


def _git_revision(repo_root: Path) -> str:
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError("git executable not found")
    result = subprocess.run(  # noqa: S603
        [git, "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _harbor_binary(harbor_repo: Path) -> Path:
    candidates = (
        harbor_repo / "harbor-framework" / ".venv" / "bin" / "harbor",
        harbor_repo / ".venv" / "bin" / "harbor",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Harbor executable not found under {harbor_repo}")


def build_harbor_command(
    *,
    harbor_binary: Path,
    dataset: Path,
    jobs_dir: Path,
    job_name: str,
    chrys_binary: Path,
    agent_profile: Path,
    model_profile: Path,
    arm: str,
    revision: str,
    concurrency: int,
) -> list[str]:
    return [
        str(harbor_binary),
        "run",
        "--path",
        str(dataset),
        "--agent",
        AGENT_IMPORT_PATH,
        "--model",
        HARBOR_MODEL_NAME,
        "--ak",
        f"chrys_binary={chrys_binary}",
        "--ak",
        f"agent_profile={agent_profile}",
        "--ak",
        f"model_profile={model_profile}",
        "--ak",
        f"run_mode={arm}",
        "--ak",
        f"chrys_revision={revision}",
        "--allow-agent-host",
        OPENROUTER_HOST,
        "--agent-include-logs",
        "instruction.md",
        "--agent-include-logs",
        "chrys.stdout.json",
        "--agent-include-logs",
        "chrys.stderr.log",
        "--agent-include-logs",
        "experiment.json",
        "--agent-include-logs",
        "chrys-sessions/**",
        "--verifier-include-logs",
        "reward.txt",
        "--verifier-include-logs",
        "reward.json",
        "--verifier-include-logs",
        "ctrf.json",
        "--verifier-include-logs",
        "run.log",
        "--verifier-include-logs",
        "test-stdout.txt",
        "--verifier-include-logs",
        "reports/**",
        "--n-concurrent",
        str(concurrency),
        "--n-concurrent-agents",
        str(concurrency),
        "--job-name",
        job_name,
        "--jobs-dir",
        str(jobs_dir),
        "--yes",
    ]


def main(argv: list[str] | None = None) -> int:
    """Prepare a frozen manifest; execute only after an explicit flag."""
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    harbor_repo = args.harbor_repo.resolve(strict=True)
    dataset = args.dataset.resolve(strict=True)
    chrys_binary = args.chrys_binary.resolve(strict=True)
    secrets_path = args.secrets.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    run_id = validate_run_id(args.run_id)
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")

    secrets = read_secrets_env(secrets_path)
    tasks = fingerprint_dataset(dataset, expected_tasks=args.expected_tasks)
    harbor_binary = _harbor_binary(harbor_repo)
    revision = _git_revision(repo_root)
    config_dir = output_dir / "config"
    profiles = render_paired_agent_profiles(
        repo_root / "src/chrys/service/profiles/agents/builtins/Code.yaml",
        config_dir / "agents",
    )
    model_profile = repo_root / "evaluation/requirement_clarification/profiles" / f"{MODEL_PROFILE_ID}.yaml"
    jobs_dir = output_dir / "jobs"
    arms = tuple(dict.fromkeys(args.arms or ARMS))
    commands = {
        arm: build_harbor_command(
            harbor_binary=harbor_binary,
            dataset=dataset,
            jobs_dir=jobs_dir,
            job_name=f"{run_id}-{arm}",
            chrys_binary=chrys_binary,
            agent_profile=profiles[arm],
            model_profile=model_profile,
            arm=arm,
            revision=revision,
            concurrency=args.concurrency,
        )
        for arm in arms
    }
    manifest = {
        "schema_version": 1,
        "protocol": "chrys-deepswe-requirement-clarification-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "chrys_revision": revision,
        "dataset": str(dataset),
        "tasks": fingerprints_as_dict(tasks),
        "task_count": len(tasks),
        "concurrency": args.concurrency,
        "arms": list(arms),
        "inputs": {
            "chrys_binary": asdict(fingerprint_file(chrys_binary)),
            "harbor_binary": asdict(fingerprint_file(harbor_binary)),
            "model_profile": asdict(fingerprint_file(model_profile)),
            "agent_profiles": {arm: asdict(fingerprint_file(path)) for arm, path in profiles.items()},
        },
        "commands": commands,
        "secrets": ["OPENROUTER_API_KEY", "CHRYS_MODEL_LOCK"],
    }
    write_json(output_dir / "manifest.json", manifest)
    sys.stdout.write(f"Prepared {len(tasks)} tasks in {output_dir}\n")
    if not args.execute:
        sys.stdout.write("Dry run only; pass --execute to invoke Harbor and OpenRouter.\n")
        return 0

    jobs_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(secrets)
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(repo_root) if not python_path else f"{repo_root}{os.pathsep}{python_path}"
    for arm, command in commands.items():
        job_dir = jobs_dir / f"{run_id}-{arm}"
        if job_dir.exists():
            if not args.resume:
                raise FileExistsError(f"refusing to overwrite existing Harbor job: {job_dir}")
            resume_command = [str(harbor_binary), "jobs", "resume", "--job-path", str(job_dir)]
            sys.stdout.write(f"Resuming {arm} arm: {job_dir.name}\n")
            subprocess.run(resume_command, cwd=harbor_repo, env=environment, check=True)  # noqa: S603
        else:
            sys.stdout.write(f"Starting {arm} arm: {job_dir.name}\n")
            subprocess.run(command, cwd=harbor_repo, env=environment, check=True)  # noqa: S603
    return 0


if __name__ == "__main__":
    sys.exit(main())
