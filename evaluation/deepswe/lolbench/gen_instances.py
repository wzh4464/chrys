#!/usr/bin/env python3
"""Materialize a LoLBench-compatible benchmark root from DeepSWE Harbor tasks.

DeepSWE (https://deepswe.datacurve.ai/) ships every task in the Harbor format:

    task.toml         metadata (repo, base commit, language, prebuilt image, limits)
    instruction.md    the prompt the agent sees
    tests/            verifier: Dockerfile (FROM the task image), test.sh, grader.py,
                      test.patch (the hidden tests), config.json (f2p / p2p node ids)
    solution/         the reference patch (held out from the agent)

LoLBench's native evaluator (``scripts/lolbench_eval.py``) reads its instance index,
``dockers/`` and ``instructions/`` relative to ``--repo-root``, so a directory laid
out this way is a drop-in benchmark root -- the same trick the SWE-bench Pro
experiment uses. Per instance this writes, under ``--out``:

    data/lolbench_clean.csv               instance_id, docker_dir, source_mapping, base_commit
    dockers/<iid>/Dockerfile              FROM <task image>; /workspace/app -> /app
    dockers/<iid>/README.md               LOLBENCH_BUILD block (docker build -t <tag> .)
    dockers/<iid>/eval_tests.patch        = tests/test.patch (kept out of the captured diff)
    dockers/<iid>/solution.patch          = solution/solution.patch (Jaccard anti-cheat gate)
    dockers/<iid>/spec.json               instance_id, language, timeout_s, verifier limits, task_dir
    instructions/original/<iid>.md        = instruction.md

Generation then runs through the engine unchanged
(``--in-container --skip-grade --no-stdlib-strip --repo-root <out>``); grading is
``grade.py`` here, which runs each task's own Harbor verifier on the captured patch.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tomllib
from pathlib import Path

from evaluation.deepswe.verify import verify_command_for


def image_tag(instance_id: str, dockerfile: str) -> str:
    """A docker tag for the wrapper image.

    Hex only: the evaluator treats any tag containing ``-base`` or ``-coverage`` as a
    helper image and would leave an instance such as ``arcane-drift-detection-baselines``
    without a primary tag. The Dockerfile's content is part of the tag, so a regenerated
    root with a different recipe is rebuilt instead of served from the old image.
    """
    digest = hashlib.sha1((instance_id + "\n" + dockerfile).encode("utf-8")).hexdigest()[:12]
    return f"lolbench-deepswe-{digest}:1"


def load_task(task_dir: Path) -> dict:
    with (task_dir / "task.toml").open("rb") as fh:
        return tomllib.load(fh)


def _manifest_order(tasks_dir: Path) -> list[str]:
    """Task ids in ``tasks/manifest.json`` order -- the order the DeepSWE runner and the
    published benchmark use; the first twenty of it are "the first 20 tasks"."""
    for candidate in (tasks_dir / "manifest.json", tasks_dir.parent / "manifest.json"):
        if candidate.is_file():
            manifest = json.loads(candidate.read_text(encoding="utf-8"))
            entries = manifest.get("tasks", []) if isinstance(manifest, dict) else manifest
            ids = [str(e.get("task_id") or e.get("instance_id") or e.get("id") or "") for e in entries]
            return [i for i in ids if i]
    return []


def select_tasks(tasks_dir: Path, *, instances: list[str], offset: int, limit: int) -> list[Path]:
    """Task directories in the DeepSWE runner's order: the manifest's, else sorted by id."""
    by_name = {d.name: d for d in tasks_dir.iterdir() if d.is_dir() and (d / "task.toml").is_file()}
    ordered = [by_name[i] for i in _manifest_order(tasks_dir) if i in by_name]
    candidates = ordered or sorted(by_name.values())
    if instances:
        missing = [i for i in instances if i not in by_name]
        if missing:
            raise SystemExit(f"unknown instances: {', '.join(missing)}")
        return [by_name[i] for i in instances]
    return candidates[offset : offset + limit] if limit else candidates[offset:]


def write_instance(task_dir: Path, out: Path, *, force: bool) -> dict:
    task = load_task(task_dir)
    meta = task["metadata"]
    iid = meta["task_id"]
    image = task["environment"]["docker_image"]
    base_commit = meta.get("base_commit_hash", "")
    docker_dir = out / "dockers" / iid
    if docker_dir.exists() and not force:
        return {"instance_id": iid, "docker_dir": iid, "source_mapping": f"{iid}.md", "base_commit": base_commit}
    docker_dir.mkdir(parents=True, exist_ok=True)
    language = str(meta.get("language", "")).lower()
    verify = verify_command_for(language)
    dockerfile = (
        f"# DeepSWE task {iid}: the prebuilt Harbor image, repo at base commit {base_commit[:12]}.\n"
        f"FROM {image}\n"
        "# The evaluator's in-container convention is /workspace/<repo>; the Harbor image keeps\n"
        "# the repository at /app. One symlink, discovered by `ls -1 /workspace`.\n"
        "RUN mkdir -p /workspace && ln -sfn /app /workspace/app\n"
    )
    if verify:
        dockerfile += (
            f"# {language}: what a PACT campaign runs to accept a mission (pact.verify_command).\n"
            f'ENV CHRYS_PACT_VERIFY_COMMAND="{verify}"\n'
        )
    tag = image_tag(iid, dockerfile)
    (docker_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    (docker_dir / "README.md").write_text(
        f"# {iid}\n\n"
        f"Wrapper over the DeepSWE task image `{image}`.\n\n"
        "<!-- LOLBENCH_BUILD_BEGIN -->\n"
        f"docker build -t {tag} .\n"
        "<!-- LOLBENCH_BUILD_END -->\n",
        encoding="utf-8",
    )
    test_patch = task_dir / "tests" / "test.patch"
    shutil.copyfile(test_patch, docker_dir / "eval_tests.patch") if test_patch.is_file() else (
        docker_dir / "eval_tests.patch"
    ).write_text("", encoding="utf-8")
    gold = task_dir / "solution" / "solution.patch"
    shutil.copyfile(gold, docker_dir / "solution.patch") if gold.is_file() else (
        docker_dir / "solution.patch"
    ).write_text("", encoding="utf-8")
    verifier = task.get("verifier", {})
    verifier_env = verifier.get("environment", {})
    (docker_dir / "spec.json").write_text(
        json.dumps(
            {
                "instance_id": iid,
                "language": meta.get("language", ""),
                "verify_command": verify,
                "timeout_s": int(task.get("agent", {}).get("timeout_sec", 5400)),
                "verifier_timeout_s": int(verifier.get("timeout_sec", 1800)),
                "verifier_network_mode": verifier.get("network_mode", "no-network"),
                "cpus": verifier_env.get("cpus", 2),
                "memory_mb": verifier_env.get("memory_mb", 8192),
                "docker_image": image,
                "base_commit": base_commit,
                "repository_url": meta.get("repository_url", ""),
                "task_dir": str(task_dir.resolve()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    instructions = out / "instructions" / "original"
    instructions.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(task_dir / "instruction.md", instructions / f"{iid}.md")
    return {"instance_id": iid, "docker_dir": iid, "source_mapping": f"{iid}.md", "base_commit": base_commit}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-dir", type=Path, required=True, help="DeepSWE checkout's tasks/ directory")
    parser.add_argument("--out", type=Path, required=True, help="Benchmark root to write (LoLBench --repo-root)")
    parser.add_argument("--instances", default="", help="Comma-separated task ids (default: --offset/--limit slice)")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20, help="0 = every task from --offset")
    parser.add_argument("--force", action="store_true", help="Rewrite instance directories that already exist")
    args = parser.parse_args(argv)

    instances = [s.strip() for s in args.instances.split(",") if s.strip()]
    tasks = select_tasks(args.tasks_dir, instances=instances, offset=args.offset, limit=args.limit)
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2
    rows = [write_instance(task_dir, args.out, force=args.force) for task_dir in tasks]
    data = args.out / "data"
    data.mkdir(parents=True, exist_ok=True)
    with (data / "lolbench_clean.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["instance_id", "docker_dir", "source_mapping", "base_commit"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} instances under {args.out}")
    for row in rows:
        print(f"  {row['instance_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
