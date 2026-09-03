# Copyright (c) 2026 Chrys. All rights reserved.

"""Headless semantic code-localization command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from chrys.foundation.config.settings import Settings
from chrys.foundation.config.warnings import settings_warning_events
from chrys.orchestration.startup import bootstrap_runtime
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.resolver import resolve_profile_selector
from chrys.service.profiles.models.schema import ModelProfile
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
    parser.add_argument("--artifact-dir", help="Artifact directory inside the repository")
    parser.add_argument("--mode", choices=[item.value for item in SemanticSearchMode], default="auto")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--top-locations", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--model-profile", default="", help="Model profile for LLM localization")
    parser.add_argument("--codegraph-command", default="", help="Optional CodeGraph command override")
    parser.add_argument("--refresh", action="store_true", help="Ignore a matching cached report")
    parser.add_argument("--json", action="store_true", help="Print the machine-readable localization payload")
    return parser


def _prepare_runtime() -> Settings:
    """Load .env, sanitize proxies, and install settings before localizing."""
    bootstrap = bootstrap_runtime(dotenv_override=True, configure_stdio=True, setup_telemetry=False)
    for warning in (*bootstrap.warnings, *settings_warning_events(bootstrap.loaded)):
        sys.stderr.write(f"Warning: {warning.message}\n")
    return bootstrap.settings


def _resolve_model(settings: Settings, selector: str) -> ModelProfile | None:
    """Resolve the localization model: the flag, then the setting, then the active model."""
    registry = ModelProfileRegistry()
    registry.load_all()
    for candidate in (selector.strip(), settings.semantic_search_model_profile.strip(), settings.model_profile.strip()):
        if not candidate:
            continue
        resolved = resolve_profile_selector(registry, candidate)
        if resolved is not None:
            return resolved
        sys.stderr.write(f"Warning: model profile not found: {candidate}\n")
    return None


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.requirement is None) == (args.task is None):
        parser.error("provide either REQUIREMENT or --task FILE, not both")
    settings = _prepare_runtime()
    model_profile = _resolve_model(settings, args.model_profile)
    try:
        task_path = Path(args.task).expanduser().resolve() if args.task else None
        requirement = task_path.read_text(encoding="utf-8") if task_path else args.requirement
        result = localize_requirement(
            args.repo,
            requirement,
            artifact_dir=args.artifact_dir,
            config=SemanticSearchConfig(
                mode=SemanticSearchMode(args.mode),
                max_iterations=args.max_iterations,
                top_locations=args.top_locations,
                timeout_seconds=args.timeout,
                model_profile=model_profile.id if model_profile is not None else "",
            ),
            refresh=args.refresh,
            codegraph_command=args.codegraph_command,
            model_profile=model_profile,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    if args.output:
        output = Path(args.output).expanduser().resolve()
        try:
            output.relative_to(Path(args.repo).expanduser().resolve())
        except ValueError:
            sys.stderr.write("Error: --output must be inside --repo\n")
            return 1
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.artifacts.report_markdown.read_text(encoding="utf-8"), encoding="utf-8")
    if args.json:
        sys.stdout.write(json.dumps(result.payload, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(f"Wrote code localization: {result.artifacts.report_markdown}\n")
        if result.reused:
            sys.stdout.write("Reused matching localization cache.\n")
        for warning in result.warnings:
            sys.stderr.write(f"Warning: {warning}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
