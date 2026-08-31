# Copyright (c) 2026 Chrys. All rights reserved.

"""Build reviewed fixed-P0 image contexts from a materialization manifest."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from evaluation.requirement_clarification.protocol import sha256_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task", action="append", dest="tasks", help="Build only this eligible task (repeatable)")
    return parser


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"manifest {name} must be an object")
    return value


def main(argv: list[str] | None = None) -> int:
    """Validate generated contexts, then invoke Docker explicitly."""
    args = _parser().parse_args(argv)
    manifest_path = args.manifest.resolve(strict=True)
    manifest = _mapping(json.loads(manifest_path.read_text(encoding="utf-8")), name="root")
    if manifest.get("protocol") != "chrys-deepswe-fixed-p0-repair-v1":
        raise ValueError("not a fixed-P0 materialization manifest")
    tasks = _mapping(manifest.get("tasks"), name="tasks")
    commands = _mapping(manifest.get("docker_commands"), name="docker_commands")
    requested = set(args.tasks or tasks)
    unknown = requested - tasks.keys()
    if unknown:
        raise ValueError(f"unknown or ineligible task(s): {', '.join(sorted(unknown))}")
    docker = shutil.which("docker")
    if docker is None:
        raise FileNotFoundError("docker executable not found")

    contexts_root = manifest_path.parent / "image-contexts"
    for task in sorted(requested):
        task_metadata = _mapping(tasks[task], name=f"tasks.{task}")
        raw_command = commands.get(task)
        if not isinstance(raw_command, list) or not all(isinstance(part, str) for part in raw_command):
            raise ValueError(f"invalid Docker command for {task}")
        context = (contexts_root / task).resolve(strict=True)
        patch = context / "model.patch"
        dockerfile = context / "Dockerfile"
        if not patch.is_file() or not dockerfile.is_file():
            raise ValueError(f"fixed-P0 build context is incomplete for {task}")
        if sha256_file(patch) != task_metadata.get("control_patch_sha256"):
            raise ValueError(f"fixed-P0 patch hash changed for {task}")
        expected_tail = [
            "build",
            "--build-arg",
            f"BASE_IMAGE={task_metadata.get('base_image')}",
            "-t",
            str(task_metadata.get("fixed_p0_image")),
            str(context),
        ]
        if raw_command[1:] != expected_tail:
            raise ValueError(f"Docker command does not match reviewed inputs for {task}")
        sys.stdout.write(f"Building fixed P0 image for {task}\n")
        subprocess.run([docker, *expected_tail], check=True)  # noqa: S603
    return 0


if __name__ == "__main__":
    sys.exit(main())
