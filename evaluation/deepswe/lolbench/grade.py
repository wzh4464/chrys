#!/usr/bin/env python3
"""Grade captured patches with each DeepSWE task's own Harbor verifier.

LoLBench's engine captures the agent's working-tree diff as ``solution.patch`` and
mirrors it to ``patches/<agent>/<instance_id>.patch``. DeepSWE grades a submission
by running ``tests/test.sh`` inside the verifier image (the task image with the
hidden tests baked in, built from ``tests/Dockerfile``): the grader resets the files
the patch touches, applies ``/logs/artifacts/model.patch``, applies the hidden
``test.patch``, runs the suites and writes ``/logs/verifier/reward.json``::

    {"reward": 0|1, "f2p_total", "f2p_passed", "p2p_total", "p2p_passed", "f2p", "p2p", ["apply_failed"]}

This script does exactly that for every instance in the benchmark root, with the
task's own limits (cpus, memory, ``verifier.timeout_sec``) and ``--network=none``,
and writes ``results.csv`` + ``summary.json`` next to the run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def verifier_tag(instance_id: str) -> str:
    return "deepswe-verifier-" + hashlib.sha1(instance_id.encode("utf-8")).hexdigest()[:12] + ":1"


def image_exists(tag: str) -> bool:
    return (
        subprocess.run(
            ["docker", "image", "inspect", tag],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def build_verifier(task_dir: Path, tag: str, *, force: bool, log: Path) -> str:
    if image_exists(tag) and not force:
        return "cached"
    with log.open("ab") as fh:
        completed = subprocess.run(
            ["docker", "build", "-t", tag, str(task_dir / "tests")],
            stdin=subprocess.DEVNULL,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return "built" if completed.returncode == 0 else f"build_failed(rc={completed.returncode})"


def grade_one(spec: dict, patch: Path | None, out_dir: Path, *, force_build: bool) -> dict:
    iid = spec["instance_id"]
    task_dir = Path(spec["task_dir"])
    result: dict = {"instance_id": iid, "status": "", "reward": 0, "resolved": False, "elapsed_s": 0.0}
    inst_dir = out_dir / iid
    logs = inst_dir / "logs"
    if logs.exists():
        shutil.rmtree(logs)
    (logs / "artifacts").mkdir(parents=True)
    (logs / "verifier").mkdir(parents=True)
    if patch is None or not patch.is_file() or patch.stat().st_size == 0:
        result["status"] = "no_patch"
        return result
    shutil.copyfile(patch, logs / "artifacts" / "model.patch")
    tag = verifier_tag(iid)
    build = build_verifier(task_dir, tag, force=force_build, log=inst_dir / "build.log")
    if build.startswith("build_failed"):
        result["status"] = build
        return result
    started = time.monotonic()
    timeout = int(spec.get("verifier_timeout_s", 1800)) + 300
    network = "none" if spec.get("verifier_network_mode", "no-network") == "no-network" else "bridge"
    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"deepswe-grade-{tag.split(':')[0][-12:]}-{int(started)}",
        f"--network={network}",
        "--cpus",
        str(spec.get("cpus", 2)),
        "--memory",
        f"{int(spec.get('memory_mb', 8192))}m",
        "-v",
        f"{logs.resolve()}:/logs",
        "--entrypoint",
        "bash",
        tag,
        "/tests/test.sh",
    ]
    with (inst_dir / "verifier.log").open("wb") as fh:
        try:
            completed = subprocess.run(
                cmd, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout, check=False
            )
            rc = completed.returncode
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", cmd[4]], stdin=subprocess.DEVNULL, capture_output=True, check=False)
            rc = -1
            result["status"] = "verifier_timeout"
    result["elapsed_s"] = round(time.monotonic() - started, 1)
    reward_json = logs / "verifier" / "reward.json"
    reward_txt = logs / "verifier" / "reward.txt"
    if reward_json.is_file():
        data = json.loads(reward_json.read_text(encoding="utf-8"))
        result.update({k: data.get(k) for k in ("f2p_total", "f2p_passed", "p2p_total", "p2p_passed", "f2p", "p2p")})
        result["reward"] = data.get("reward", 0)
        result["resolved"] = data.get("reward", 0) == 1
        result["status"] = (
            "apply_failed" if data.get("apply_failed") else ("resolved" if result["resolved"] else "graded")
        )
    elif reward_txt.is_file():
        result["status"] = f"verifier_crash(reward.txt={reward_txt.read_text().strip()})"
    elif not result["status"]:
        result["status"] = f"no_reward(rc={rc})"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-root", type=Path, required=True, help="Benchmark root written by gen_instances.py")
    parser.add_argument(
        "--patches-dir", type=Path, required=True, help="LoLBench patch store for one agent (patches/<agent>)"
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Where verifier logs, results.csv and summary.json go"
    )
    parser.add_argument("--instances", default="", help="Comma-separated instance ids (default: all in the index)")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--force-build", action="store_true")
    args = parser.parse_args(argv)

    only = {s.strip() for s in args.instances.split(",") if s.strip()}
    specs = []
    with (args.repo_root / "data" / "lolbench_clean.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            iid = row["instance_id"]
            if only and iid not in only:
                continue
            spec = json.loads(
                (args.repo_root / "dockers" / row["docker_dir"] / "spec.json").read_text(encoding="utf-8")
            )
            specs.append(spec)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def run(spec: dict) -> dict:
        patch = args.patches_dir / f"{spec['instance_id']}.patch"
        res = grade_one(spec, patch, args.out_dir, force_build=args.force_build)
        print(f"{res['instance_id']}: {res['status']} reward={res['reward']} ({res['elapsed_s']}s)", flush=True)
        return res

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        results = list(pool.map(run, specs))

    fields = [
        "instance_id",
        "status",
        "reward",
        "resolved",
        "f2p_total",
        "f2p_passed",
        "p2p_total",
        "p2p_passed",
        "f2p",
        "p2p",
        "elapsed_s",
    ]
    with (args.out_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    summary = {
        "total": len(results),
        "resolved": sum(1 for r in results if r["resolved"]),
        "with_patch": sum(1 for r in results if r["status"] != "no_patch"),
        "apply_failed": sum(1 for r in results if r["status"] == "apply_failed"),
        "f2p_passed": sum(int(r.get("f2p_passed") or 0) for r in results),
        "f2p_total": sum(int(r.get("f2p_total") or 0) for r in results),
        "p2p_passed": sum(int(r.get("p2p_passed") or 0) for r in results),
        "p2p_total": sum(int(r.get("p2p_total") or 0) for r in results),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
