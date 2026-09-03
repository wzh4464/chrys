# Copyright (c) 2026 Chrys. All rights reserved.

"""``chrys debug router`` — see a routing decision without running a turn.

Routing is the one part of a turn whose reasoning is otherwise invisible: by
the time the user notices a campaign started, it has started. This prints the
signals, the score, the band, and whether the workspace could carry a campaign
at all, for any prompt, without touching an agent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from chrys.orchestration.startup import bootstrap_runtime
from chrys.service.routing.classifier import (
    DEFAULT_BANDS,
    RouteBand,
    band_for,
    extract_prompt_signals,
    plan_for,
    prompt_score,
)
from chrys.service.routing.readiness import probe_workspace_readiness, workspace_fingerprint


def build_parser() -> argparse.ArgumentParser:
    """Return the ``chrys debug router`` parser."""
    parser = argparse.ArgumentParser(
        prog="chrys debug router",
        description="Show how a prompt would be routed, without running an agent.",
    )
    parser.add_argument("prompt", nargs="?", help="Prompt text to classify")
    parser.add_argument("-t", "--task", help="Read the prompt from a UTF-8 text file")
    parser.add_argument("-C", "--workdir", default=".", help="Workspace to probe for readiness")
    parser.add_argument("--json", action="store_true", help="Print the machine-readable report")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run the LLM tiebreaker when the prompt lands in the uncertain band",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``chrys debug router``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.prompt is None) == (args.task is None):
        parser.error("provide either PROMPT or --task FILE, not both")
    try:
        prompt = Path(args.task).expanduser().read_text(encoding="utf-8") if args.task else args.prompt
    except OSError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 2

    bootstrap = bootstrap_runtime(dotenv_override=True, configure_stdio=True, setup_telemetry=False)
    settings = bootstrap.settings
    cwd = str(Path(args.workdir).expanduser().resolve())

    signals = extract_prompt_signals(prompt)
    score, reason = prompt_score(signals)
    band = band_for(score)
    readiness = probe_workspace_readiness(
        cwd,
        verify_command=settings.pact_verify_command,
        # No agent is built here, so the tool cannot be probed; report what the
        # workspace itself decides and say so.
        pact_tool_available=True,
    )
    from chrys.service.profiles.agents.schema import LongHorizonConfig

    plan = plan_for(band, LongHorizonConfig(), readiness)
    would_fire = band is RouteBand.UNCERTAIN and settings.routing_mode != "off"

    report: dict[str, Any] = {
        "prompt_chars": len(prompt),
        "signals": {
            "word_count": signals.word_count,
            "step_markers": signals.step_markers,
            "scope_hits": list(signals.scope_hits),
            "acceptance_hits": list(signals.acceptance_hits),
            "path_mentions": signals.path_mentions,
            "question_like": signals.question_like,
            "archetype": signals.archetype,
        },
        "score": round(score, 4),
        "reason": reason,
        "band": band.value,
        "bands": {
            "strong_standard_max": DEFAULT_BANDS.strong_standard_max,
            "lean_standard_max": DEFAULT_BANDS.lean_standard_max,
            "uncertain_max": DEFAULT_BANDS.uncertain_max,
            "lean_long_horizon_max": DEFAULT_BANDS.lean_long_horizon_max,
        },
        "readiness": {
            "verify_command_configured": readiness.verify_command_configured,
            "has_tests": readiness.has_tests,
            "git_dirty": readiness.git_dirty,
            "pact_ready": readiness.pact_ready,
            "note": "pact_tool_available is assumed; no agent is built for a dry run",
        },
        "plan": {
            "localization": plan.localization,
            "clarification": plan.clarification,
            "pact": plan.pact,
        },
        "routing_mode": settings.routing_mode,
        "workspace_fingerprint": workspace_fingerprint(cwd),
        "tiebreaker": {"would_fire": would_fire},
    }
    if args.full and would_fire:
        report["tiebreaker"].update(_run_tiebreaker(prompt, signals, settings))

    if args.json:
        sys.stdout.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return 0
    _print_report(report)
    return 0


def _run_tiebreaker(prompt: str, signals: Any, settings: Any) -> dict[str, Any]:
    """Actually call the model, reporting a failure rather than raising."""
    import asyncio

    from chrys.service.profiles.models.registry import ModelProfileRegistry
    from chrys.service.profiles.models.resolver import resolve_profile_selector
    from chrys.service.routing.guard import TiebreakerGuard
    from chrys.service.routing.llm import LlmRouteClassifier

    registry = ModelProfileRegistry()
    registry.load_all()
    selector = settings.routing_tiebreaker_model_profile.strip() or settings.model_profile.strip()
    profile = resolve_profile_selector(registry, selector) if selector else None
    if profile is None:
        return {"failure": "unavailable", "detail": "no model profile resolved"}
    classifier = LlmRouteClassifier(
        profile,
        guard=TiebreakerGuard(),
        session_id=None,
        parent_session_id=None,
        session_dir=None,
    )
    verdict = asyncio.run(classifier.classify(prompt, signals))
    return {
        "model_profile": profile.id,
        "long_horizon": verdict.long_horizon,
        "confidence": verdict.confidence,
        "reason": verdict.reason,
        "failure": verdict.failure,
    }


def _print_report(report: dict[str, Any]) -> None:
    signals = report["signals"]
    readiness = report["readiness"]
    plan = report["plan"]
    write = sys.stdout.write
    write(f"band            {report['band']}  (score {report['score']:.2f})\n")
    write(f"reason          {report['reason']}\n")
    write(f"archetype       {signals['archetype']}\n")
    write(
        f"signals         words={signals['word_count']} steps={signals['step_markers']} "
        f"paths={signals['path_mentions']} question={signals['question_like']}\n"
    )
    write(f"scope           {', '.join(signals['scope_hits']) or '-'}\n")
    write(f"acceptance      {', '.join(signals['acceptance_hits']) or '-'}\n")
    write(
        f"readiness       verify_command={readiness['verify_command_configured']} "
        f"tests={readiness['has_tests']} pact_ready={readiness['pact_ready']}\n"
    )
    write(
        f"plan            localization={plan['localization']} clarification={plan['clarification']} "
        f"pact={plan['pact']}\n"
    )
    write(f"routing mode    {report['routing_mode']}\n")
    tiebreaker = report["tiebreaker"]
    write(f"tiebreaker      would_fire={tiebreaker['would_fire']}\n")
    if "failure" in tiebreaker:
        write(f"                failure={tiebreaker['failure'] or 'none'}")
        if not tiebreaker.get("failure"):
            write(f" long_horizon={tiebreaker['long_horizon']} confidence={tiebreaker['confidence']:.2f}")
        write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
