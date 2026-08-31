# Copyright (c) 2026 Chrys. All rights reserved.

"""Materialize a repair-only DeepSWE dataset with the control P0 held fixed."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from evaluation.requirement_clarification.protocol import (
    fingerprint_dataset,
    fingerprints_as_dict,
    sha256_file,
    write_json,
)
from evaluation.requirement_clarification.summarize import load_selected_attempts

_DOCKERFILE = """ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY model.patch /tmp/chrys-p0.patch
RUN cd /app \\
 && git config --global --add safe.directory /app \\
 && if [ -s /tmp/chrys-p0.patch ]; then git apply --binary --whitespace=nowarn /tmp/chrys-p0.patch; fi \\
 && rm -f /tmp/chrys-p0.patch
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--control-job", type=Path, required=True)
    parser.add_argument("--candidate-job", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-repository", default="chrys/deepswe-fixed-p0")
    parser.add_argument("--expected-tasks", type=int, default=113)
    parser.add_argument("--expected-eligible", type=int)
    parser.add_argument("--build-images", action="store_true", help="Actually invoke docker build")
    return parser


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _clarification_delta(trial_dir: Path) -> str | None:
    paths = sorted(trial_dir.glob("agent/chrys-sessions/*/requirement_clarification/turn_*/clarification.private.json"))
    if not paths:
        return None
    delta = _load_mapping(paths[-1]).get("delta")
    return delta.strip() if isinstance(delta, str) and delta.strip() else None


def _image_tag(repository: str, task: str) -> str:
    safe = re.sub(r"[^a-z0-9_.-]+", "-", task.casefold()).strip("-.")
    if not safe:
        raise ValueError(f"task name cannot form a Docker tag: {task!r}")
    return f"{repository}:{safe}"


def _repair_instruction(original: str, delta: str) -> str:
    return (
        f"{original.rstrip()}\n\n"
        "---\n\n"
        "The workspace already contains a provisional implementation (P0) of the requirement above. "
        "Keep correct parts of P0 and repair it using the additional repository-grounded clarification below. "
        "Treat the original requirement as authoritative if there is any conflict.\n\n"
        "Additional requirement clarification (ΔR):\n\n"
        f"{delta.rstrip()}\n"
    )


def _base_image(task_toml: dict[str, Any], task: str) -> str:
    environment = task_toml.get("environment")
    if not isinstance(environment, dict):
        raise ValueError(f"task {task} has no [environment] mapping")
    image = environment.get("docker_image")
    if not isinstance(image, str) or not image:
        raise ValueError(f"task {task} has no environment.docker_image")
    return image


def _replace_environment_image(source: str, image: str) -> str:
    lines = source.splitlines(keepends=True)
    in_environment = False
    replacements = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_environment = stripped == "[environment]"
            continue
        if in_environment and re.match(r"^docker_image\s*=", stripped):
            newline = "\n" if line.endswith("\n") else ""
            lines[index] = f"docker_image = {json.dumps(image)}{newline}"
            replacements += 1
    if replacements != 1:
        raise ValueError(f"expected one environment.docker_image assignment, found {replacements}")
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Create build contexts and a repair dataset; image building is opt-in."""
    args = _parser().parse_args(argv)
    source_dataset = args.source_dataset.resolve(strict=True)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")

    source_tasks = fingerprint_dataset(source_dataset, expected_tasks=args.expected_tasks)
    control = load_selected_attempts(args.control_job)
    candidate = load_selected_attempts(args.candidate_job)
    source_names = {item.name for item in source_tasks}
    if control.keys() != candidate.keys() or set(control) != source_names:
        raise ValueError("source, control, and candidate task sets must match exactly")

    dataset_dir = output_dir / "dataset"
    contexts_dir = output_dir / "image-contexts"
    commands: dict[str, list[str]] = {}
    tasks: dict[str, dict[str, object]] = {}
    excluded: dict[str, str] = {}
    docker = shutil.which("docker")
    if args.build_images and docker is None:
        raise FileNotFoundError("docker executable not found")

    for task in sorted(source_names):
        control_trial, _, _ = control[task]
        candidate_trial, _, _ = candidate[task]
        patch = control_trial / "artifacts/model.patch"
        delta = _clarification_delta(candidate_trial)
        if not patch.is_file():
            excluded[task] = "control model.patch missing"
            continue
        if delta is None:
            excluded[task] = "candidate produced no persisted clarification delta"
            continue

        source_task = source_dataset / task
        task_toml_path = source_task / "task.toml"
        task_toml_text = task_toml_path.read_text(encoding="utf-8")
        task_toml = tomllib.loads(task_toml_text)
        base_image = _base_image(task_toml, task)
        image = _image_tag(args.image_repository, task)
        context = contexts_dir / task
        context.mkdir(parents=True)
        shutil.copy2(patch, context / "model.patch")
        (context / "Dockerfile").write_text(_DOCKERFILE, encoding="utf-8")

        destination = dataset_dir / task
        shutil.copytree(source_task, destination)
        (destination / "task.toml").write_text(
            _replace_environment_image(task_toml_text, image),
            encoding="utf-8",
        )
        original_instruction = (source_task / "instruction.md").read_text(encoding="utf-8")
        (destination / "instruction.md").write_text(_repair_instruction(original_instruction, delta), encoding="utf-8")

        command = [docker or "docker", "build", "--build-arg", f"BASE_IMAGE={base_image}", "-t", image, str(context)]
        commands[task] = command
        tasks[task] = {
            "base_image": base_image,
            "fixed_p0_image": image,
            "control_patch_sha256": sha256_file(patch),
            "delta_sha256": hashlib.sha256(delta.encode()).hexdigest(),
            "source_task_sha256": next(item.sha256 for item in source_tasks if item.name == task),
        }

    if args.expected_eligible is not None and len(tasks) != args.expected_eligible:
        raise ValueError(f"expected {args.expected_eligible} eligible tasks, found {len(tasks)}")
    manifest = {
        "schema_version": 1,
        "protocol": "chrys-deepswe-fixed-p0-repair-v1",
        "source_dataset": str(source_dataset),
        "control_job": str(args.control_job.resolve(strict=True)),
        "candidate_job": str(args.candidate_job.resolve(strict=True)),
        "source_tasks": fingerprints_as_dict(source_tasks),
        "eligible_count": len(tasks),
        "excluded": excluded,
        "tasks": tasks,
        "docker_commands": commands,
    }
    write_json(output_dir / "manifest.json", manifest)

    if args.build_images:
        for task, command in commands.items():
            sys.stdout.write(f"Building fixed P0 image for {task}\n")
            subprocess.run(command, check=True)  # noqa: S603
    else:
        sys.stdout.write(
            f"Materialized {len(tasks)} eligible tasks in {dataset_dir}; pass --build-images to build P0 images.\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
