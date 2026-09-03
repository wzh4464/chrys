#!/usr/bin/env python3
"""Build the unified code perception package for semantic-search requirement augmentation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from _common import (
    FORMAT_CODE_PERCEPTION,
    ScriptError,
    append_trace,
    bullet_lines,
    ensure_allowed_path,
    load_json,
    now_iso,
    reject_benchmark_answer_path,
    resolve_path,
    sha1_path,
    write_json,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Seeded workspace repo root.")
    parser.add_argument("--requirement", required=True, help="Original task prompt.")
    parser.add_argument("--artifact-dir", required=True, help="Semantic-search artifact directory.")
    parser.add_argument(
        "--out", help="Output code-perception.json path. Defaults to artifact-dir/code-perception.json."
    )
    parser.add_argument("--markdown", help="Output code-perception.md path. Defaults to out with .md suffix.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for component scripts.")
    parser.add_argument("--codegraph-cmd", default="", help="Optional CodeGraph CLI command override.")
    parser.add_argument(
        "--localization-mode",
        choices=("auto", "llm", "fallback"),
        default=os.environ.get("SEMANTIC_SEARCH_LOCALIZATION_MODE", "auto"),
    )
    parser.add_argument(
        "--localization-model-profile",
        default=os.environ.get("SEMANTIC_SEARCH_LOCALIZATION_MODEL_PROFILE")
        or os.environ.get("CHRYS_MODEL_PROFILE", ""),
    )
    parser.add_argument("--localization-max-iterations", type=int, default=20)
    return parser.parse_args(argv)


def load_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    repo = resolve_path(args.repo)
    if not repo.is_dir():
        raise ScriptError(f"repo path does not exist: {repo}")
    artifact_dir = resolve_path(args.artifact_dir)
    requirement = ensure_allowed_path(
        args.requirement,
        allowed_roots=[artifact_dir],
        allowed_files=[resolve_path(args.requirement)],
        purpose="requirement",
    )
    reject_benchmark_answer_path(requirement, purpose="requirement")
    out = resolve_path(args.out or artifact_dir / "code-perception.json")
    markdown = resolve_path(args.markdown or out.with_suffix(".md"))
    out = ensure_allowed_path(out, allowed_roots=[repo, artifact_dir, out.parent], purpose="output")
    markdown = ensure_allowed_path(
        markdown, allowed_roots=[repo, artifact_dir, markdown.parent], purpose="markdown-output"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return repo, requirement, artifact_dir, out, markdown


def run_component(python: str, script_name: str, args: list[str], *, artifact_dir: Path) -> dict[str, Any]:
    stdout_path = artifact_dir / f"{Path(script_name).stem.replace('_', '-')}.stdout"
    stderr_path = artifact_dir / f"{Path(script_name).stem.replace('_', '-')}.stderr"
    argv = [python, str(SCRIPT_DIR / script_name), *args]
    try:
        proc = subprocess.run(argv, text=True, capture_output=True, check=False)
    except OSError as err:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(str(err) + "\n", encoding="utf-8")
        return {
            "script": script_name,
            "argv": argv,
            "returncode": 127,
            "ok": False,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "error": str(err),
        }
    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    return {
        "script": script_name,
        "argv": argv,
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    repo, requirement, artifact_dir, out, markdown = load_inputs(args)
    index = artifact_dir / "index.json"
    global_json = artifact_dir / "global-perception.json"
    global_md = artifact_dir / "global-perception.md"
    codegraph_json = artifact_dir / "codegraph-perception.json"
    codegraph_md = artifact_dir / "codegraph-perception.md"
    repository_json = artifact_dir / "repository-perception.json"
    repository_md = artifact_dir / "repository-perception.md"
    facts_json = artifact_dir / "code-facts.json"
    semantic_json = artifact_dir / "semantic-perception.json"
    semantic_md = artifact_dir / "semantic-perception.md"
    repo_map_json = artifact_dir / "repo-map.json"
    localization_json = artifact_dir / "code-localization.json"
    localization_md = artifact_dir / "code-localization.md"
    localization_trace = artifact_dir / "localization-trace.jsonl"
    localization_graph = artifact_dir / "localization-graph.json"

    steps: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {
        "index": str(index),
        "global_perception": str(global_json),
        "global_perception_markdown": str(global_md),
        "codegraph_perception": str(codegraph_json),
        "codegraph_perception_markdown": str(codegraph_md),
        "repository_perception": str(repository_json),
        "repository_perception_markdown": str(repository_md),
        "facts": str(facts_json),
        "semantic_perception": str(semantic_json),
        "semantic_perception_markdown": str(semantic_md),
        "repo_map": str(repo_map_json),
        "localization": str(localization_json),
        "localization_markdown": str(localization_md),
        "localization_trace": str(localization_trace),
        "localization_graph": str(localization_graph),
        "code_perception": str(out),
        "code_perception_markdown": str(markdown),
    }

    index_step = run_component(
        args.python, "build_index.py", ["--repo", str(repo), "--out", str(index)], artifact_dir=artifact_dir
    )
    steps.append(index_step)
    if not index_step["ok"]:
        payload = render_payload(repo, requirement, artifacts, steps, status="index-failed")
        write_outputs(out, markdown, payload)
        return payload

    global_args: list[str] = []
    global_step = run_component(
        args.python,
        "global_perception.py",
        [
            "--repo",
            str(repo),
            "--index",
            str(index),
            "--out",
            str(global_json),
            "--markdown",
            str(global_md),
            "--artifact-dir",
            str(artifact_dir),
        ],
        artifact_dir=artifact_dir,
    )
    steps.append(global_step)
    if global_step["ok"]:
        global_args = ["--global-perception", str(global_json)]

    codegraph_args: list[str] = []
    codegraph_component_args = [
        "--repo",
        str(repo),
        "--requirement",
        str(requirement),
        "--index",
        str(index),
        "--out",
        str(codegraph_json),
        "--markdown",
        str(codegraph_md),
        "--artifact-dir",
        str(artifact_dir),
    ]
    if args.codegraph_cmd:
        codegraph_component_args.extend(["--codegraph-cmd", args.codegraph_cmd])
    codegraph_step = run_component(
        args.python, "codegraph_perception.py", codegraph_component_args, artifact_dir=artifact_dir
    )
    steps.append(codegraph_step)
    if codegraph_step["ok"]:
        codegraph_args = ["--codegraph-perception", str(codegraph_json)]

    repository_args: list[str] = []
    repository_step = run_component(
        args.python,
        "repository_perception.py",
        [
            "--repo",
            str(repo),
            "--index",
            str(index),
            "--out",
            str(repository_json),
            "--markdown",
            str(repository_md),
            "--artifact-dir",
            str(artifact_dir),
            *global_args,
            *codegraph_args,
        ],
        artifact_dir=artifact_dir,
    )
    steps.append(repository_step)
    if repository_step["ok"]:
        repository_args = ["--repository-perception", str(repository_json)]

    mine_step = run_component(
        args.python,
        "mine_context.py",
        [
            "--repo",
            str(repo),
            "--requirement",
            str(requirement),
            "--index",
            str(index),
            "--out",
            str(facts_json),
            "--semantic-out",
            str(semantic_json),
            "--semantic-markdown",
            str(semantic_md),
            "--artifact-dir",
            str(artifact_dir),
            *global_args,
            *repository_args,
        ],
        artifact_dir=artifact_dir,
    )
    steps.append(mine_step)

    localization_component_args = [
        "--repo",
        str(repo),
        "--requirement",
        str(requirement),
        "--index",
        str(index),
        "--out",
        str(localization_json),
        "--markdown",
        str(localization_md),
        "--artifact-dir",
        str(artifact_dir),
        "--trace",
        str(localization_trace),
        "--graph-out",
        str(localization_graph),
        "--mode",
        args.localization_mode,
        "--max-iterations",
        str(args.localization_max_iterations),
    ]
    if args.localization_model_profile:
        localization_component_args.extend(["--model-profile", args.localization_model_profile])
    if mine_step["ok"]:
        localization_component_args.extend(["--facts", str(facts_json)])
    if codegraph_step["ok"]:
        localization_component_args.extend(["--codegraph-perception", str(codegraph_json)])
    localization_step = run_component(
        args.python,
        "localize_task.py",
        localization_component_args,
        artifact_dir=artifact_dir,
    )
    steps.append(localization_step)

    if args.localization_mode == "llm" and not localization_step["ok"]:
        status = "localization-failed"
    else:
        status = "ok" if mine_step["ok"] and localization_step["ok"] else "partial"
    payload = render_payload(repo, requirement, artifacts, steps, status=status)
    write_outputs(out, markdown, payload)
    append_trace(
        "build-code-perception",
        {
            "out": str(out),
            "status": payload["status"],
            "facts_available": payload["summary"]["facts_available"],
            "codegraph_available": payload["summary"]["codegraph_available"],
            "localization_available": payload["summary"]["localization_available"],
        },
    )
    return payload


def render_payload(
    repo: Path, requirement: Path, artifacts: dict[str, str], steps: list[dict[str, Any]], *, status: str
) -> dict[str, Any]:
    facts = load_optional_json(artifacts["facts"])
    repository = load_optional_json(artifacts["repository_perception"])
    codegraph = load_optional_json(artifacts["codegraph_perception"])
    global_perception = load_optional_json(artifacts["global_perception"])
    localization = load_optional_json(artifacts["localization"])
    return {
        "format": FORMAT_CODE_PERCEPTION,
        "created_at": now_iso(),
        "status": status,
        "inputs": {
            "repo": str(repo),
            "requirement": str(requirement),
            "requirement_sha1": sha1_path(requirement),
        },
        "artifacts": artifacts,
        "steps": steps,
        "summary": {
            "facts_available": bool(facts),
            "repository_perception_available": bool(repository),
            "global_perception_available": bool(global_perception),
            "codegraph_perception_available": bool(codegraph),
            "codegraph_available": bool((codegraph or {}).get("available")),
            "localization_available": bool(localization),
            "localization_count": len(localization.get("locations", [])),
            "localization_mode": localization.get("summary", {}).get("generation_mode", ""),
            "localization_tool_call_count": localization.get("summary", {}).get("tool_call_count", 0),
            "repository_backend": (repository or {}).get("backend", ""),
            "ranked_file_count": len((facts or {}).get("ranked_files", [])),
            "implementation_surface_count": len((facts or {}).get("implementation_surfaces", [])),
        },
    }


def load_optional_json(path: str) -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        return {}
    try:
        return load_json(resolved)
    except ScriptError:
        return {}


def write_outputs(out: Path, markdown: Path, payload: dict[str, Any]) -> None:
    write_json(out, payload)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(payload), encoding="utf-8")


def render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {})
    artifacts = payload.get("artifacts", {})
    lines = [
        "# Code Perception",
        "",
        "This is the unified code perception package used before Augmented Requirement generation.",
        "",
        "## Status",
        "",
        f"- Status: {payload.get('status')}",
        f"- Facts available: {summary.get('facts_available')}",
        f"- Repository perception available: {summary.get('repository_perception_available')}",
        f"- CodeGraph available: {summary.get('codegraph_available')}",
        f"- Code localization available: {summary.get('localization_available')}",
        f"- Code localization mode: {summary.get('localization_mode') or '(unknown)'}",
        f"- Localization tool calls: {summary.get('localization_tool_call_count', 0)}",
        f"- Repository backend: {summary.get('repository_backend') or '(unknown)'}",
        f"- Ranked files: {summary.get('ranked_file_count', 0)}",
        f"- Implementation surfaces: {summary.get('implementation_surface_count', 0)}",
        "",
        "## Perception Layers",
        "",
        "- Base code perception: CodeGraph when available, with builtin static perception as fallback.",
        "- Enhanced code perception: semantic-search links the original requirement to repository evidence.",
        "- Requirement augmentation consumes `code-facts.json`, which includes the merged repository perception "
        "and task-specific semantic perception.",
        "",
        "## Artifact Routes",
        "",
    ]
    lines.extend(bullet_lines([f"{key}: `{value}`" for key, value in artifacts.items()]))
    lines.extend(["", "## Component Steps", ""])
    for step in payload.get("steps", []):
        lines.append(f"- `{step.get('script')}` ok={step.get('ok')} rc={step.get('returncode')}")
        lines.append(f"  stdout: `{step.get('stdout_path')}`")
        lines.append(f"  stderr: `{step.get('stderr_path')}`")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        payload = build(args)
        default_out = resolve_path(args.artifact_dir) / "code-perception.json"
        print(f"Wrote code perception: {resolve_path(args.out or default_out)}")
        print(payload.get("summary", {}))
        return 0 if payload["status"] in {"ok", "partial", "index-failed"} else 1
    except ScriptError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
