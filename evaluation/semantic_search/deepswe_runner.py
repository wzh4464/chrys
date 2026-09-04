# Copyright (c) 2026 Chrys. All rights reserved.

"""Run Chrys over the first (or selected) tasks in a local DeepSWE checkout.

DeepSWE is a Harbor-compatible directory dataset rather than a JSON/JSONL
file.  This runner reads ``tasks/manifest.json`` in manifest order, checks out
each task's pinned upstream commit into an isolated workspace, and writes all
Chrys/semantic-search artifacts below one directory per task.  The benchmark's
``solution/`` and verifier files are never copied into an agent workspace.

The default operation is localization only. ``--run-agent`` appends that report
to a normal Chrys turn. ``--run-enrichment`` instead runs an agent profile whose
native requirement-enrichment workflow owns clarification and localization.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

from evaluation.deepswe.verify import verify_command_for


def _read_tasks(path: Path) -> list[dict[str, Any]]:
    """Read a DeepSWE checkout or retain the historical JSON/JSONL input."""
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            manifest_path = path / "tasks" / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"DeepSWE manifest.json does not exist below {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        values = manifest.get("tasks", []) if isinstance(manifest, dict) else manifest
        if not isinstance(values, list):
            raise ValueError(f"DeepSWE manifest has no tasks list: {manifest_path}")
        tasks_root = manifest_path.parent
        result: list[dict[str, Any]] = []
        for item in values:
            if not isinstance(item, dict):
                continue
            task = dict(item)
            task_id = str(task.get("task_id") or "").strip()
            if not task_id:
                continue
            task["task_dir"] = str(tasks_root / task_id)
            result.append(task)
        return result

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        value = json.loads(text)
        values = value if isinstance(value, list) else value.get("data", value.get("tasks", []))
    if not isinstance(values, list):
        raise ValueError("dataset must be a DeepSWE directory, JSON array, JSONL file, or object with data/tasks")
    return [item for item in values if isinstance(item, dict)]


def _task_id(task: dict[str, Any], position: int) -> str:
    raw = task.get("instance_id") or task.get("task_id") or task.get("id") or f"task-{position:04d}"
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in str(raw))


def _select_tasks(
    tasks: list[dict[str, Any]],
    *,
    order: str,
    offset: int,
    limit: int,
) -> list[dict[str, Any]]:
    selected = list(tasks)
    if order == "alphabetical":
        selected.sort(key=lambda item: str(item.get("task_id") or item.get("instance_id") or item.get("id") or ""))
    start = max(offset, 0)
    return selected[start : start + max(limit, 0)]


def _requirement(task: dict[str, Any]) -> str:
    task_dir = Path(str(task.get("task_dir", "")))
    instruction = task_dir / "instruction.md"
    if instruction.is_file():
        text = instruction.read_text(encoding="utf-8").strip()
        if text:
            return text
    for key in ("problem_statement", "requirement", "prompt", "issue", "description"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _task_metadata(task: dict[str, Any]) -> dict[str, Any]:
    task_dir = Path(str(task.get("task_dir", "")))
    task_toml = task_dir / "task.toml"
    if not task_toml.is_file():
        return {}
    with task_toml.open("rb") as handle:
        value = tomllib.load(handle)
    metadata = value.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _repo_for(task: dict[str, Any], repo_root: Path) -> Path:
    value = task.get("repo_path") or task.get("workspace")
    if value:
        candidate = Path(str(value)).expanduser().resolve()
        if candidate.is_dir():
            return candidate
    name = str(task.get("repo") or task.get("repository") or "").strip()
    if name:
        candidate = (repo_root / name).resolve()
        if candidate.is_dir():
            return candidate
    return repo_root


def _run_git(args: list[str], *, cwd: Path | None = None, timeout: float = 1_800.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def _write_patch(repo: Path, base_commit: str, destination: Path) -> None:
    """Write committed, tracked, and newly created agent changes as one patch."""
    add = _run_git(["add", "-A"], cwd=repo, timeout=300.0)
    if add.returncode:
        detail = (add.stderr or add.stdout).strip()[-3_000:]
        raise RuntimeError(f"could not stage workspace changes for patch generation: {detail}")
    diff_args = ["diff", "--cached", "--binary"]
    if base_commit:
        diff_args.append(base_commit)
    diff = _run_git(diff_args, cwd=repo, timeout=300.0)
    if diff.returncode:
        detail = (diff.stderr or diff.stdout).strip()[-3_000:]
        raise RuntimeError(f"could not generate model patch: {detail}")
    destination.write_text(diff.stdout, encoding="utf-8")


def _checkout_workspace(
    task: dict[str, Any],
    *,
    workspace: Path,
    repo_root: Path,
    clone: bool,
) -> tuple[Path, str]:
    """Create/reuse a repository workspace at the task's pinned base commit."""
    metadata = _task_metadata(task)
    base_commit = str(metadata.get("base_commit_hash") or task.get("base_commit_hash") or "").strip()
    repository_url = str(
        metadata.get("repository_url") or task.get("repository_url") or task.get("repo_url") or ""
    ).strip()
    existing = _repo_for(task, repo_root)
    if existing != repo_root and (existing / ".git").exists():
        if not (workspace / ".git").exists():
            if not clone:
                raise RuntimeError(f"no owned workspace for {task.get('task_id')} at {workspace}")
            workspace.parent.mkdir(parents=True, exist_ok=True)
            result = _run_git(["clone", str(existing), str(workspace)], timeout=3_600.0)
            if result.returncode:
                detail = (result.stderr or result.stdout).strip()[-3_000:]
                raise RuntimeError(f"local git clone failed for {task.get('task_id')}: {detail}")
    elif not (workspace / ".git").exists():
        if not clone:
            raise RuntimeError(f"no repository workspace for {task.get('task_id')} at {workspace}")
        if not repository_url:
            raise RuntimeError(f"task {task.get('task_id')} does not declare repository_url")
        workspace.parent.mkdir(parents=True, exist_ok=True)
        result = _run_git(["clone", repository_url, str(workspace)], timeout=3_600.0)
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()[-3_000:]
            raise RuntimeError(f"git clone failed for {task.get('task_id')}: {detail}")
    if base_commit:
        # Workspaces are owned by this runner.  Remove any untracked files
        # left by an interrupted/retried agent before restoring the pinned
        # preimage; otherwise a retry could accidentally inherit an old patch.
        clean = _run_git(["clean", "-fdx"], cwd=workspace, timeout=300.0)
        if clean.returncode:
            detail = (clean.stderr or clean.stdout).strip()[-3_000:]
            raise RuntimeError(f"could not clean task workspace {task.get('task_id')}: {detail}")
        result = _run_git(["fetch", "--quiet", "origin", base_commit], cwd=workspace, timeout=1_800.0)
        if result.returncode:
            # A normal clone usually already contains the commit.  Continue to
            # checkout so the error below reports the useful repository state.
            result = _run_git(["cat-file", "-e", f"{base_commit}^{{commit}}"], cwd=workspace, timeout=120.0)
            if result.returncode:
                detail = (result.stderr or result.stdout).strip()[-3_000:]
                raise RuntimeError(f"base commit {base_commit} unavailable for {task.get('task_id')}: {detail}")
        result = _run_git(["checkout", "--force", "--detach", base_commit], cwd=workspace, timeout=300.0)
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()[-3_000:]
            raise RuntimeError(f"git checkout failed for {task.get('task_id')}: {detail}")
    head = _run_git(["rev-parse", "HEAD"], cwd=workspace, timeout=120.0)
    checked_out = head.stdout.strip() if head.returncode == 0 else ""
    return workspace, checked_out


def _chrys_command(
    chrys: str,
    chrys_python: str,
    chrys_src: Path,
    chrys_home: Path,
) -> tuple[list[str], dict[str, str]]:
    chrys_home.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(chrys_src), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["HOME"] = str(chrys_home)
    env["CHRYS_SESSION_ROOT_DIR"] = str(chrys_home / ".chrys")
    if chrys:
        return [chrys], env
    return [chrys_python, "-m", "chrys.app.cli.app"], env


def _run_locate(
    command_prefix: list[str],
    env: dict[str, str],
    repo: Path,
    requirement_file: Path,
    artifact_dir: Path,
    *,
    mode: str,
    model_profile: str,
    timeout: float,
) -> tuple[int, str, str]:
    command = [
        *command_prefix,
        "locate",
        "--repo",
        str(repo),
        "--task",
        str(requirement_file),
        "--artifact-dir",
        str(artifact_dir),
        "--mode",
        mode,
        "--timeout",
        str(timeout),
        "--json",
    ]
    if model_profile:
        command.extend(["--model-profile", model_profile])
    completed = subprocess.run(  # noqa: S603
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout + 30,
        env=env,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _run_agent(
    command_prefix: list[str],
    env: dict[str, str],
    repo: Path,
    requirement_file: Path,
    report: Path | None,
    *,
    agent: str,
    model: str,
    timeout: float,
    route: str = "",
    verify_command: str = "",
) -> tuple[int, str, str]:
    command = [
        *command_prefix,
        "run",
        "--task",
        str(requirement_file),
        "--agent",
        agent,
        "--workdir",
        str(repo),
    ]
    if route:
        # The long-horizon track localizes and clarifies on its own frozen
        # snapshot; forcing the route is what makes a benchmark exercise it
        # regardless of how each task's wording would have scored.
        command.extend(["--route", route])
    if report is not None:
        command.extend(["--localization-file", str(report)])
    # Headless Chrys needs an explicit model selector; otherwise it may use an
    # empty global pointer even when an isolated model profile is present.
    if model:
        command.extend(["--model", model])
    if verify_command:
        # A campaign needs a deterministic verification to accept a mission; the
        # task's language decides what that is (evaluation.deepswe.verify).
        env = {**env, "CHRYS_PACT_VERIFY_COMMAND": verify_command}
    completed = subprocess.run(  # noqa: S603
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=env,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _timeout_output(value: str | bytes | None) -> str:
    """Normalize partial subprocess output captured by ``TimeoutExpired``."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="DeepSWE checkout, tasks directory, or legacy JSON/JSONL file")
    parser.add_argument("--repo-root", default="", help="Optional directory containing pre-cloned task repositories")
    parser.add_argument("--output-dir", required=True, help="Directory for per-task artifacts and workspaces")
    parser.add_argument("--chrys", default="", help="Chrys executable; default runs this checkout as a Python module")
    parser.add_argument("--chrys-python", default=sys.executable, help="Python executable used for the Chrys module")
    parser.add_argument(
        "--chrys-src", default=str(Path(__file__).resolve().parents[1] / "src"), help="Chrys source directory"
    )
    parser.add_argument("--chrys-home", default="", help="Isolated Chrys home (default: OUTPUT_DIR/chrys-home)")
    parser.add_argument("--limit", type=int, default=20, help="Maximum tasks to run (default: 20)")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many manifest tasks before applying --limit")
    parser.add_argument(
        "--order",
        choices=("manifest", "alphabetical"),
        default="manifest",
        help="Task selection order before offset/limit",
    )
    parser.add_argument(
        "--mode", choices=("fallback", "auto", "llm"), default="fallback", help="Semantic localization mode"
    )
    parser.add_argument("--localization-model", default="", help="Model profile used by LLM semantic localization")
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument("--run-agent", action="store_true", help="Run an agent with the standalone localization")
    run_mode.add_argument(
        "--run-enrichment",
        action="store_true",
        help="Run an agent profile whose native workflow performs clarification and localization",
    )
    run_mode.add_argument(
        "--run-long-horizon",
        action="store_true",
        help=(
            "Run each task as one forced long-horizon turn (chrys run --route long-horizon): the track freezes "
            "the workspace, localizes and clarifies on the snapshot, repairs, and delegates a campaign when the "
            "workspace can verify one. No separate localization preflight is run."
        ),
    )
    parser.add_argument("--agent", default="Code", help="Chrys agent profile for an agent run")
    parser.add_argument("--model", default="", help="Optional Chrys model profile for an agent run")
    parser.add_argument(
        "--clone", action=argparse.BooleanOptionalAction, default=True, help="Clone missing task repositories"
    )
    parser.add_argument("--locate-timeout", type=float, default=900.0)
    parser.add_argument("--agent-timeout", type=float, default=3_600.0)
    parser.add_argument(
        "--per-task", action="store_true", help="Run one task at a time (recommended for long DeepSWE jobs)"
    )
    parser.add_argument("--resume", action="store_true", help="Skip tasks with a completed result.json")
    parser.add_argument(
        "--retry-agent",
        action="store_true",
        help="Retry non-completed agents while reusing an existing localization report",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve() if args.repo_root else output_dir / "workspaces"
    if not dataset.exists():
        raise SystemExit(f"Dataset does not exist: {dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = _read_tasks(dataset)
    start = max(args.offset, 0)
    tasks = _select_tasks(all_tasks, order=args.order, offset=start, limit=args.limit)
    agent_requested = args.run_agent or args.run_enrichment or args.run_long_horizon
    native_analysis = args.run_enrichment or args.run_long_horizon
    chrys_home = Path(args.chrys_home).expanduser().resolve() if args.chrys_home else output_dir / "chrys-home"
    command_prefix, command_env = _chrys_command(
        args.chrys,
        args.chrys_python,
        Path(args.chrys_src).expanduser().resolve(),
        chrys_home,
    )
    summary: list[dict[str, Any]] = []
    for local_position, task in enumerate(tasks):
        position = start + local_position + 1
        task_id = _task_id(task, position)
        task_dir = output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        result_path = task_dir / "result.json"
        if args.resume and result_path.is_file():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            completed_statuses = {"completed"} if agent_requested else {"localized", "completed"}
            # A timed-out/terminated agent may still have produced a useful
            # patch before the runner wrote its result.  Treat that durable
            # patch as resumable too; resetting the workspace and invoking the
            # model again would otherwise destroy the partial attempt.
            patch_path = task_dir / "model.patch"
            has_patch = patch_path.is_file() and patch_path.stat().st_size > 0
            patch_is_resumable = agent_requested and has_patch and not args.retry_agent
            if isinstance(previous, dict) and (previous.get("status") in completed_statuses or patch_is_resumable):
                summary.append(previous)
                continue
        requirement = _requirement(task)
        started = time.monotonic()
        record: dict[str, Any] = {"task_id": task_id, "position": position, "status": "invalid"}
        if not requirement:
            record["error"] = "task has no requirement/problem statement"
            result_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            summary.append(record)
            continue
        requirement_file = task_dir / "PROMPT.md"
        requirement_file.write_text(requirement, encoding="utf-8")
        workspace = output_dir / "workspaces" / task_id
        localization_dir = task_dir / "semantic-search"
        repo: Path | None = None
        head = ""
        try:
            repo, head = _checkout_workspace(task, workspace=workspace, repo_root=repo_root, clone=args.clone)
            # On a resumed agent run, localization is already a durable
            # artifact.  Reusing it avoids repeating an expensive semantic
            # search and, importantly, preserves the exact report consumed by
            # the agent.  A missing report still triggers localization.
            report = localization_dir / "code-localization.md"
            reuse_localization = (args.resume or args.retry_agent) and args.run_agent and report.is_file()
            if native_analysis or reuse_localization:
                rc, stdout, stderr = 0, "", ""
            else:
                rc, stdout, stderr = _run_locate(
                    command_prefix,
                    command_env,
                    repo,
                    requirement_file,
                    localization_dir,
                    mode=args.mode,
                    model_profile=args.localization_model,
                    timeout=args.locate_timeout,
                )
                (task_dir / "locate.stdout").write_text(stdout, encoding="utf-8")
                (task_dir / "locate.stderr").write_text(stderr, encoding="utf-8")
            record.update({"repo": str(repo), "base_commit": head})
            if not native_analysis:
                record["localization_returncode"] = rc
            if not native_analysis and (rc != 0 or not report.is_file()):
                record.update({"status": "localization_failed", "error": stderr[-2000:]})
            elif agent_requested:
                try:
                    agent_rc, agent_out, agent_err = _run_agent(
                        command_prefix,
                        command_env,
                        repo,
                        requirement_file,
                        None if native_analysis else report,
                        agent=args.agent,
                        model=args.model,
                        timeout=args.agent_timeout,
                        route="long-horizon" if args.run_long_horizon else "",
                        verify_command=(
                            verify_command_for(_task_metadata(task).get("language")) if args.run_long_horizon else ""
                        ),
                    )
                except subprocess.TimeoutExpired as exc:
                    agent_rc = None
                    agent_out = _timeout_output(exc.stdout)
                    agent_err = _timeout_output(exc.stderr)
                    timeout_message = f"Agent timed out after {args.agent_timeout:.1f} seconds"
                    if agent_err and not agent_err.endswith("\n"):
                        agent_err += "\n"
                    agent_err += timeout_message + "\n"
                (task_dir / "agent.stdout").write_text(agent_out, encoding="utf-8")
                (task_dir / "agent.stderr").write_text(agent_err, encoding="utf-8")
                if agent_rc is None:
                    record.update({"status": "agent_timeout", "agent_returncode": None, "error": timeout_message})
                else:
                    record.update(
                        {"status": "completed" if agent_rc == 0 else "agent_failed", "agent_returncode": agent_rc}
                    )
            else:
                record["status"] = "localized"
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
            record.update({"status": "failed", "error": str(exc)})
        finally:
            # Preserve partial work even when the model process times out or
            # exits non-zero.  A verifier can then distinguish an empty patch
            # from a useful partial attempt.
            if repo is not None and repo.is_dir() and (repo / ".git").exists() and head:
                try:
                    _write_patch(repo, head, task_dir / "model.patch")
                    record["patch_bytes"] = (task_dir / "model.patch").stat().st_size
                except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
                    record.setdefault("error", f"patch extraction failed: {exc}")
        record["duration_seconds"] = round(time.monotonic() - started, 3)
        result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        summary.append(record)
    # Merge with prior one-task/batch invocations so a long run can be safely
    # resumed without losing completed records from earlier invocations.
    summary_path = output_dir / "summary.json"
    merged: dict[str, dict[str, Any]] = {}
    if summary_path.is_file():
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(previous, list):
                merged.update({str(item.get("task_id")): item for item in previous if isinstance(item, dict)})
        except OSError, ValueError:
            pass
    merged.update({str(item.get("task_id")): item for item in summary if isinstance(item, dict)})
    ordered = sorted(merged.values(), key=lambda item: int(item.get("position", 0)))
    summary_path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if all(item.get("status") in {"localized", "completed", "invalid"} for item in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
