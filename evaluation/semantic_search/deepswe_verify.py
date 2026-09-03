# Copyright (c) 2026 Chrys. All rights reserved.

"""Verify Chrys patches with DeepSWE's task-local Docker verifiers.

This is the small subset of the Harbor/Pier execution contract needed after
Chrys has already produced ``model.patch`` files.  Each verifier image is
built only from ``tasks/<id>/tests`` and receives the patch through
``/logs/artifacts/model.patch``.  Agent workspaces never see hidden tests or
reference solutions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def _run(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout)  # noqa: S603


def _manifest_tasks(dataset: Path, limit: int, offset: int) -> list[dict[str, Any]]:
    tasks_root = dataset / "tasks" if (dataset / "tasks" / "manifest.json").is_file() else dataset
    manifest_path = tasks_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = manifest.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError(f"DeepSWE manifest has no tasks list: {manifest_path}")
    start = max(offset, 0)
    selected = [item for item in tasks if isinstance(item, dict)][start : start + max(limit, 0)]
    return [{**item, "task_dir": str(tasks_root / str(item.get("task_id", "")))} for item in selected]


def _next_attempt(verifier_root: Path) -> Path:
    verifier_root.mkdir(parents=True, exist_ok=True)
    existing = [path for path in verifier_root.glob("attempt-*") if path.is_dir()]
    attempt = verifier_root / f"attempt-{len(existing) + 1:03d}"
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "artifacts").mkdir()
    return attempt


def _verify_task(
    task: dict[str, Any],
    *,
    run_dir: Path,
    docker: str,
    build_timeout: float,
    verify_timeout: float,
    rebuild: bool,
) -> dict[str, Any]:
    task_id = str(task.get("task_id", ""))
    task_dir = Path(str(task["task_dir"]))
    result_dir = run_dir / task_id
    patch = result_dir / "model.patch"
    started = time.monotonic()
    record: dict[str, Any] = {"task_id": task_id, "status": "verifier_failed", "resolved": False}
    if not patch.is_file():
        record["error"] = "model.patch is missing"
        return record

    verifier_root = result_dir / "verifier"
    attempt = _next_attempt(verifier_root)
    shutil.copyfile(patch, attempt / "artifacts" / "model.patch")
    image = f"chrys3-deepswe-verifier:{task_id.lower()}"
    tests_dir = task_dir / "tests"
    if rebuild or _run([docker, "image", "inspect", image], timeout=120.0).returncode != 0:
        build = _run([docker, "build", "--tag", image, str(tests_dir)], timeout=build_timeout)
        (attempt / "build.stdout").write_text(build.stdout, encoding="utf-8")
        (attempt / "build.stderr").write_text(build.stderr, encoding="utf-8")
        if build.returncode:
            record.update(
                {"error": (build.stderr or build.stdout).strip()[-4_000:], "build_returncode": build.returncode}
            )
            record["duration_seconds"] = round(time.monotonic() - started, 3)
            return record

    verify = _run(
        [
            docker,
            "run",
            "--rm",
            "--network",
            "none",
            "--volume",
            f"{attempt}:/logs",
            image,
            "/tests/test.sh",
        ],
        timeout=verify_timeout,
    )
    (attempt / "test.stdout").write_text(verify.stdout, encoding="utf-8")
    (attempt / "test.stderr").write_text(verify.stderr, encoding="utf-8")
    reward_path = attempt / "verifier" / "reward.json"
    reward: dict[str, Any] = {}
    if reward_path.is_file():
        try:
            loaded = json.loads(reward_path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            # A verifier killed mid-write leaves a truncated reward file. That
            # is a verifier failure for this task, not a reason to abandon the
            # rest of the run -- the empty reward below reports it as one.
            loaded = None
        reward = loaded if isinstance(loaded, dict) else {}
    resolved = reward.get("reward") == 1 or reward.get("reward") == 1.0
    status = "resolved" if resolved else "unresolved"
    if verify.returncode and not reward:
        status = "verifier_failed"
    record.update(
        {
            "status": status,
            "resolved": resolved,
            "verifier_returncode": verify.returncode,
            "reward": reward,
            "attempt": attempt.name,
            "patch_bytes": patch.stat().st_size,
            "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
            "duration_seconds": round(time.monotonic() - started, 3),
        }
    )
    if status == "verifier_failed":
        record["error"] = (verify.stderr or verify.stdout).strip()[-4_000:]
    (result_dir / "verification.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Local DeepSWE checkout or tasks directory")
    parser.add_argument("--run-dir", required=True, help="Directory produced by deepswe_runner.py")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--build-timeout", type=float, default=3_600.0)
    parser.add_argument("--verify-timeout", type=float, default=1_800.0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = Path(args.dataset).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    tasks = _manifest_tasks(dataset, args.limit, args.offset)
    records: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("task_id", ""))
        cached = run_dir / task_id / "verification.json"
        if args.resume and cached.is_file():
            value = json.loads(cached.read_text(encoding="utf-8"))
            patch = run_dir / task_id / "model.patch"
            cached_size = value.get("patch_bytes") if isinstance(value, dict) else None
            same_patch = patch.is_file() and cached_size == patch.stat().st_size
            cached_sha256 = value.get("patch_sha256") if isinstance(value, dict) else None
            current_sha256 = hashlib.sha256(patch.read_bytes()).hexdigest() if same_patch else ""
            if same_patch and isinstance(cached_sha256, str):
                same_patch = current_sha256 == cached_sha256
            if isinstance(value, dict) and value.get("status") in {"resolved", "unresolved"} and same_patch:
                if not isinstance(cached_sha256, str):
                    value["patch_sha256"] = current_sha256
                    cached.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                records.append(value)
                continue
        record = _verify_task(
            task,
            run_dir=run_dir,
            docker=args.docker,
            build_timeout=args.build_timeout,
            verify_timeout=args.verify_timeout,
            rebuild=args.rebuild,
        )
        (run_dir / task_id / "verification.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        records.append(record)

    resolved = sum(record.get("resolved") is True for record in records)
    verified = sum(record.get("status") in {"resolved", "unresolved"} for record in records)
    summary = {
        "format": "chrys-deepswe-verification.v1",
        "task_count": len(records),
        "verified_count": verified,
        "resolved_count": resolved,
        "resolve_rate": resolved / len(records) if records else 0.0,
        "verified_resolve_rate": resolved / verified if verified else 0.0,
        "records": records,
    }
    (run_dir / "verification-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0 if verified == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
