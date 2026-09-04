# Copyright (c) 2026 Chrys. All rights reserved.

"""Headless semantic code-localization command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.orchestration.startup import bootstrap_runtime
from chrys.service.semantic_search import SemanticSearchConfig, SemanticSearchMode, localize_requirement


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chrys locate", description="Locate code relevant to a requirement.")
    parser.add_argument("requirement", nargs="?", help="Requirement text")
    parser.add_argument("-t", "--task", help="Read the requirement from a UTF-8 text file")
    parser.add_argument("--repo", "-C", default=".", help="Repository root")
    parser.add_argument(
        "--output",
        "-o",
        help="Markdown report path (defaults to <repo>/.semantic-search/code-localization.md)",
    )
    parser.add_argument("--artifact-dir", help="Dedicated artifact directory (defaults inside the repository)")
    parser.add_argument(
        "--mode",
        choices=[item.value for item in SemanticSearchMode if item is not SemanticSearchMode.OFF],
        default="auto",
    )
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--top-locations", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--model-profile", default="", help="Model profile for LLM localization")
    parser.add_argument("--codegraph-command", default="", help="Optional CodeGraph command override")
    parser.add_argument("--refresh", action="store_true", help="Ignore a matching cached report")
    parser.add_argument("--json", action="store_true", help="Print the machine-readable localization payload")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.requirement is None) == (args.task is None):
        parser.error("provide either REQUIREMENT or --task FILE, not both")
    repo = Path(args.repo).expanduser().resolve()
    bootstrap_runtime(
        dotenv_override=True,
        configure_stdio=True,
        project_root=repo,
    )
    try:
        task_path = Path(args.task).expanduser().resolve() if args.task else None
        requirement = task_path.read_text(encoding="utf-8") if task_path else args.requirement
        artifact_dir = Path(args.artifact_dir).expanduser().resolve() if args.artifact_dir else None
        result = localize_requirement(
            repo,
            requirement,
            artifact_dir=artifact_dir,
            config=SemanticSearchConfig(
                mode=SemanticSearchMode(args.mode),
                max_iterations=args.max_iterations,
                top_locations=args.top_locations,
                timeout_seconds=args.timeout,
                model_profile=args.model_profile,
            ),
            refresh=args.refresh,
            codegraph_command=args.codegraph_command,
        )
        if args.output:
            output = Path(args.output).expanduser().resolve()
            try:
                output.relative_to(repo)
            except ValueError as exc:
                raise ValueError("--output must be inside --repo") from exc
            report = result.artifacts.report_markdown.read_text(encoding="utf-8")
            atomic_write_owner_only_text(output, report)
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    if args.json:
        sys.stdout.write(json.dumps(result.payload, ensure_ascii=True, indent=2) + "\n")
    else:
        sys.stdout.write(f"Wrote code localization: {result.artifacts.report_markdown}\n")
        if result.reused:
            sys.stdout.write("Reused matching localization cache.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
