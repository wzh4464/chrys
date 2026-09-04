# Copyright (c) 2026 Chrys. All rights reserved.

"""Generate an Augmented Requirement and routed augmentation sub-documents."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from _common import (
    FORMAT_FACTS,
    FORMAT_LOCALIZATION,
    FORMAT_ROUTES,
    ScriptError,
    append_trace,
    bullet_lines,
    ensure_allowed_path,
    load_json,
    markdown_escape_line,
    now_iso,
    read_text,
    reject_benchmark_answer_path,
    resolve_path,
    sha1_path,
    stable_unique,
    tokenize,
    write_json,
)

TOPICS = [
    ("expected-behavior", "Expected Behavior", "must-read"),
    ("scope-boundary", "Scope Boundary", "must-read"),
    ("task-decomposition", "Task Decomposition", "must-read"),
    ("codebase-context", "Codebase Context", "read-if-needed"),
    ("implementation-surfaces", "Likely Implementation Surfaces", "read-before-editing"),
    ("existing-patterns", "Relevant Existing Patterns", "read-if-needed"),
    ("code-details", "Code Details To Inspect", "read-before-editing"),
    ("validation-plan", "Validation Plan", "must-read"),
    ("anti-patterns", "Anti-patterns and Failure Modes", "must-read"),
    ("open-questions", "Assumptions and Open Questions", "read-if-uncertain"),
]

TASK_PACKAGE_NOTE = (
    "Document role: part of the Augmented Requirement task package. Use this document to guide the current "
    "benchmark task, then verify code claims against the Original Requirement and source before editing. "
    "Candidate files and surfaces guide implementation planning but are not automatic edit mandates. "
    "The document should narrow the task toward a complete, buildable patch rather than expand it into a broad rewrite "
    "or shrink it into an incomplete partial edit."
)

DOC_REQUIRED_HEADINGS = [
    "## High Priority",
    "## Must Implement",
    "## Must Preserve",
    "## Should Inspect",
    "## Do Not Do",
    "## Validation",
    "## Low-confidence Notes",
    "## Evidence Index",
]

SCOPE_GUARD_NOTE = (
    "Scope guard: use this document to clarify and constrain the task. "
    "Treat code surfaces as inspection candidates until direct source inspection proves they are needed for a concrete behavior delta. "
    "Scope control must not omit required new files, generated artifacts, registration points, or build metadata, "
    "but generated/build surfaces should be changed only when source inspection proves they are required."
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirement", required=True)
    parser.add_argument(
        "--facts",
        help="Optional code-facts.json from mine_context.py. If omitted, generate requirement-only augmentation.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--augmentation-dir", required=True)
    parser.add_argument("--artifact-dir", help="Semantic-search artifact directory. Defaults to output parent.")
    parser.add_argument(
        "--localization",
        help="Optional code-localization.json. When supplied, generate the compact localization-guided task brief.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "llm", "fallback"),
        default=os.environ.get("SEMANTIC_SEARCH_AUGMENTATION_MODE", "auto"),
        help="auto tries LLM first and falls back to script-generated docs; llm requires a successful LLM response.",
    )
    parser.add_argument(
        "--model-profile",
        default=os.environ.get("SEMANTIC_SEARCH_MODEL_PROFILE") or os.environ.get("CHRYS_MODEL_PROFILE", ""),
        help="Chrys model profile id/path/name for LLM augmentation. Defaults to CHRYS_MODEL_PROFILE.",
    )
    parser.add_argument("--llm-timeout", type=float, default=float(os.environ.get("SEMANTIC_SEARCH_LLM_TIMEOUT", "240")))
    parser.add_argument("--max-evidence-chars", type=int, default=int(os.environ.get("SEMANTIC_SEARCH_MAX_EVIDENCE_CHARS", "55000")))
    return parser.parse_args(argv)


def load_inputs(args: argparse.Namespace) -> tuple[Path, Path | None, Path, Path, Path, str, dict[str, Any]]:
    out = resolve_path(args.out)
    augmentation_dir = resolve_path(args.augmentation_dir)
    artifact_dir = resolve_path(args.artifact_dir or out.parent)
    requirement = ensure_allowed_path(args.requirement, allowed_roots=[artifact_dir], allowed_files=[resolve_path(args.requirement)], purpose="requirement")
    reject_benchmark_answer_path(requirement, purpose="requirement")
    out = ensure_allowed_path(out, allowed_roots=[artifact_dir, out.parent], purpose="output")
    augmentation_dir = ensure_allowed_path(augmentation_dir, allowed_roots=[artifact_dir, augmentation_dir.parent], purpose="augmentation-dir")
    requirement_text = read_text(requirement)
    facts_path = None
    if args.facts:
        facts_path = ensure_allowed_path(args.facts, allowed_roots=[artifact_dir, out.parent, augmentation_dir], purpose="facts")
        facts = load_json(facts_path)
        if facts.get("format") != FORMAT_FACTS:
            raise ScriptError(f"unsupported facts format: {facts.get('format')}")
    else:
        facts = build_requirement_only_facts(requirement, requirement_text)
    return requirement, facts_path, out, augmentation_dir, artifact_dir, requirement_text, facts


def build_requirement_only_facts(requirement: Path, requirement_text: str) -> dict[str, Any]:
    signals = extract_requirement_signals(requirement_text)
    repo_map = {
        "top_levels": {},
        "generated_files": [],
        "build_files": [],
        "test_roots": [],
        "stats": {},
    }
    general_code_facts = {
        "repo_map": repo_map,
        "capability_requirements": {},
        "repository_perception": {},
        "global_perception": {},
        "index_stats": {},
        "top_levels": {},
        "generated_files": [],
        "build_files": [],
        "test_roots": [],
        "source_of_truth_hints": [],
        "global_risks": [],
    }
    semantic_search_facts = {
        "requirement_signals": signals,
        "ranked_files": [],
        "ranked_file_details": [],
        "implementation_surfaces": [],
        "existing_patterns": [],
        "code_details": [],
        "validation_hints": [],
        "anti_patterns": [
            {
                "claim": "Repository code evidence was not available during requirement augmentation.",
                "source": "uncertain",
                "confidence": "high",
                "action": "Use this task package for requirement clarification, then perform normal source inspection before editing.",
            },
            {
                "claim": "Do not infer required edit files from missing semantic-search code evidence.",
                "source": "inferred",
                "confidence": "high",
                "action": "Treat all implementation surfaces as unknown until Chrys verifies them in the workspace.",
            },
        ],
        "uncertainties": [
            {
                "question": "Code indexing or context mining did not provide repository evidence; identify implementation surfaces with normal Chrys search/read tools.",
                "source": "uncertain",
                "confidence": "low",
                "evidence": [],
            }
        ],
        "global_semantic_links": [],
    }
    return {
        "format": FORMAT_FACTS,
        "created_at": now_iso(),
        "facts_mode": "requirement-only",
        "inputs": {
            "repo": "",
            "requirement": str(requirement),
            "requirement_sha1": sha1_path(requirement),
            "index": "",
            "index_sha1": "",
        },
        "general_code_facts": general_code_facts,
        "semantic_search_facts": semantic_search_facts,
        # Backward-compatible aliases for existing augmentation/rendering code.
        "requirement_signals": signals,
        "repo_map": repo_map,
        "capability_requirements": {},
        "repository_perception": {},
        "global_perception": {},
        "ranked_files": [],
        "ranked_file_details": [],
        "implementation_surfaces": [],
        "existing_patterns": [],
        "code_details": [],
        "validation_hints": [],
        "anti_patterns": semantic_search_facts["anti_patterns"],
        "uncertainties": semantic_search_facts["uncertainties"],
        "global_semantic_links": [],
        "augmentation_routes": [],
    }


def extract_requirement_signals(text: str) -> list[dict[str, Any]]:
    code_terms = re.findall(r"`([^`]{2,80})`", text)
    fenced_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, flags=re.DOTALL)
    file_like = re.findall(r"[\w./-]+\.(?:py|java|scala|rs|c|h|cpp|hpp|toml|xml|gradle|rst|md|gram|asdl|g4|peg|y|l)", text, flags=re.IGNORECASE)
    title_terms = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title_terms.extend(tokenize(stripped))
    behavior_terms = tokenize(" ".join(fenced_blocks + code_terms + [text]), min_len=3)
    signals: list[dict[str, Any]] = []
    for term in stable_unique([*file_like, *code_terms, *title_terms, *behavior_terms])[:160]:
        lowered = str(term).lower().strip()
        if lowered:
            signals.append({"term": lowered, "category": categorize_signal(lowered), "source": "original_requirement"})
    return signals


def categorize_signal(term: str) -> str:
    if "/" in term or ("." in term and not term.endswith(".")):
        return "path"
    if term.startswith("--") or term in {"config", "option", "flag", "setting"}:
        return "config"
    if any(word in term for word in ("syntax", "grammar", "parser", "token", "expr", "statement")):
        return "syntax"
    if any(word in term for word in ("error", "exception", "invalid", "fail")):
        return "error"
    if any(word in term for word in ("test", "expected", "should")):
        return "test"
    if term[:1].isupper() or "_" in term or ":" in term:
        return "api"
    return "behavior"


def generate(args: argparse.Namespace) -> dict[str, Any]:
    requirement, facts_path, out, augmentation_dir, artifact_dir, requirement_text, facts = load_inputs(args)
    if args.localization:
        return generate_localization_task(
            args,
            requirement,
            facts_path,
            out,
            augmentation_dir,
            artifact_dir,
            requirement_text,
            facts,
        )
    augmentation_dir.mkdir(parents=True, exist_ok=True)
    fallback_docs, fallback_summaries = build_fallback_docs(facts)
    evidence_bundle = render_evidence_bundle(requirement_text, facts, max_chars=args.max_evidence_chars)
    prompt = render_llm_prompt(evidence_bundle)
    evidence_path = artifact_dir / "evidence-bundle.md"
    prompt_path = artifact_dir / "augmentation-prompt.md"
    response_path = artifact_dir / "augmentation-llm-response.txt"
    error_path = artifact_dir / "augmentation-llm-error.txt"
    evidence_path.write_text(evidence_bundle, encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")
    for stale_path in (response_path, error_path):
        if stale_path.exists():
            stale_path.unlink()

    generation_mode = "fallback"
    generation_note = "script fallback"
    docs = fallback_docs
    summaries = fallback_summaries
    if args.mode != "fallback":
        try:
            raw_response = call_llm_for_augmentation(args, prompt)
            response_path.write_text(raw_response, encoding="utf-8")
            payload = parse_llm_payload(raw_response)
            docs, summaries, missing = normalize_llm_documents(payload, fallback_docs, fallback_summaries)
            generation_mode = "llm" if not missing else "llm-partial"
            generation_note = "LLM generated all augmentation documents." if not missing else f"LLM response missed {len(missing)} topic(s); script fallback filled them."
            if error_path.exists():
                error_path.unlink()
        except ScriptError as err:
            error_path.write_text(str(err) + "\n", encoding="utf-8")
            if args.mode == "llm":
                raise
            generation_note = f"LLM augmentation failed; script fallback used: {err}"

    routes = []
    for slug, title, priority in TOPICS:
        path = (augmentation_dir / f"{slug}.md").resolve()
        quality = document_quality(docs[slug])
        path.write_text(docs[slug].rstrip() + "\n", encoding="utf-8")
        routes.append(
            {
                "topic": title,
                "slug": slug,
                "path": str(path),
                "sha1": sha1_path(path),
                "priority": priority,
                "summary": summaries[slug],
                "summary_line_count": len(summaries[slug]),
                "quality": quality,
            }
        )

    route_payload = {
        "format": FORMAT_ROUTES,
        "created_at": now_iso(),
        "routes": routes,
        "generation": {
            "mode": generation_mode,
            "note": generation_note,
            "evidence_bundle": str(evidence_path),
            "prompt": str(prompt_path),
            "llm_response": str(response_path) if response_path.exists() else "",
            "llm_error": str(error_path) if error_path.exists() else "",
        },
        "inputs": {
            "requirement": str(requirement),
            "requirement_sha1": sha1_path(requirement),
            "facts": str(facts_path) if facts_path else "",
            "facts_sha1": sha1_path(facts_path) if facts_path else "",
            "facts_source": "general_and_semantic_search_code_facts" if facts_path else "requirement_only_fallback",
        },
    }
    write_json(augmentation_dir.parent / "augmentation_routes.json", route_payload)
    write_manifest(
        augmentation_dir.parent / "manifest.json",
        requirement,
        facts_path,
        out,
        augmentation_dir,
        routes,
        generation={
            "mode": generation_mode,
            "note": generation_note,
            "evidence_bundle": str(evidence_path),
            "prompt": str(prompt_path),
            "llm_response": str(response_path) if response_path.exists() else "",
            "llm_error": str(error_path) if error_path.exists() else "",
        },
    )
    out.write_text(render_main(requirement_text, routes, generation_mode=generation_mode), encoding="utf-8")
    append_trace("augment-requirement", {"out": str(out), "route_count": len(routes), "generation_mode": generation_mode})
    return {"out": str(out), "routes": routes, "generation_mode": generation_mode}


def generate_localization_task(
    args: argparse.Namespace,
    requirement: Path,
    facts_path: Path | None,
    out: Path,
    augmentation_dir: Path,
    artifact_dir: Path,
    requirement_text: str,
    facts: dict[str, Any],
) -> dict[str, Any]:
    """Generate the compact task brief used by the new localization workflow."""
    localization_path = ensure_allowed_path(
        args.localization,
        allowed_roots=[artifact_dir, out.parent, augmentation_dir],
        purpose="localization",
    )
    localization = load_json(localization_path)
    if localization.get("format") != FORMAT_LOCALIZATION:
        raise ScriptError(f"unsupported localization format: {localization.get('format')}")

    augmentation_dir.mkdir(parents=True, exist_ok=True)
    localization_markdown = augmentation_dir / "code-localization.md"
    localization_markdown.write_text(render_localization_markdown(localization), encoding="utf-8")
    evidence_path = artifact_dir / "evidence-bundle.md"
    prompt_path = artifact_dir / "augmentation-prompt.md"
    response_path = artifact_dir / "augmentation-llm-response.txt"
    error_path = artifact_dir / "augmentation-llm-error.txt"
    evidence = render_localization_evidence(requirement_text, localization, facts, max_chars=args.max_evidence_chars)
    prompt = render_localization_prompt(evidence)
    evidence_path.write_text(evidence, encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")
    for stale_path in (response_path, error_path):
        if stale_path.exists():
            stale_path.unlink()

    body = render_localization_fallback(localization)
    generation_mode = "task-fallback"
    generation_note = "deterministic localization-guided task brief"
    if args.mode != "fallback":
        try:
            raw_response = call_llm_for_augmentation(
                args,
                prompt,
                system_prompt=(
                    "You are a senior software engineer preparing a concise task brief for Chrys. "
                    "Use the original requirement and code localization evidence. Return Markdown only, "
                    "preserve the requested behavior, and never write a patch."
                ),
            )
            response_path.write_text(raw_response, encoding="utf-8")
            body = raw_response.strip() or body
            generation_mode = "task-llm"
            generation_note = "LLM generated a localization-guided task brief."
        except ScriptError as err:
            error_path.write_text(str(err) + "\n", encoding="utf-8")
            if args.mode == "llm":
                raise
            generation_note = f"LLM task-brief generation failed; deterministic fallback used: {err}"

    routes = [
        {
            "topic": "Code Localization",
            "slug": "code-localization",
            "path": str(localization_markdown.resolve()),
            "sha1": sha1_path(localization_markdown),
            "priority": "read-before-editing",
            "summary": [
                f"{len(localization.get('locations', []))} ranked code locations were produced.",
                "Every location requires source verification before editing.",
            ],
            "summary_line_count": 2,
        }
    ]
    generation = {
        "mode": generation_mode,
        "note": generation_note,
        "evidence_bundle": str(evidence_path),
        "prompt": str(prompt_path),
        "llm_response": str(response_path) if response_path.exists() else "",
        "llm_error": str(error_path) if error_path.exists() else "",
    }
    route_payload = {
        "format": FORMAT_ROUTES,
        "created_at": now_iso(),
        "routes": routes,
        "generation": generation,
        "inputs": {
            "requirement": str(requirement),
            "requirement_sha1": sha1_path(requirement),
            "facts": str(facts_path) if facts_path else "",
            "facts_sha1": sha1_path(facts_path) if facts_path else "",
            "facts_source": "general_and_semantic_search_code_facts" if facts_path else "requirement_only_fallback",
            "localization": str(localization_path),
            "localization_sha1": sha1_path(localization_path),
        },
    }
    write_json(artifact_dir / "augmentation_routes.json", route_payload)
    write_manifest(artifact_dir / "manifest.json", requirement, facts_path, out, augmentation_dir, routes, generation=generation)
    out.write_text(
        render_localization_task_main(requirement_text, localization_markdown, body, generation_mode),
        encoding="utf-8",
    )
    append_trace("augment-localization-task", {"out": str(out), "generation_mode": generation_mode, "location_count": len(localization.get("locations", []))})
    return {"out": str(out), "routes": routes, "generation_mode": generation_mode}


def render_localization_evidence(requirement_text: str, localization: dict[str, Any], facts: dict[str, Any], *, max_chars: int) -> str:
    lines = [
        "# Localization-Guided Task Evidence",
        "",
        "## Original Requirement",
        "",
        requirement_text.rstrip(),
        "",
        "## Code Localization",
        "",
        render_localization_markdown(localization).rstrip(),
        "",
        "## Repository Validation Hints",
        "",
    ]
    hints = facts.get("validation_hints", [])
    if hints:
        lines.extend(f"- `{item.get('path', '')}`: {item.get('action', '')}" for item in hints[:16])
    else:
        lines.append("- Verify the ranked locations and nearest tests with normal Chrys tools.")
    text = "\n".join(lines).rstrip() + "\n"
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "\n\n[Evidence truncated.]\n"


def render_localization_prompt(evidence: str) -> str:
    return f"""Prepare a concise, execution-oriented task brief for Chrys from the evidence below.

The Original Requirement is authoritative and must remain unchanged in meaning. The Code Localization section contains ranked inspection candidates, not automatic edit mandates. Explain the likely primary and propagation locations, what must be verified, and the narrow validation path. Do not invent requirements and do not generate a patch.

Return Markdown only with these sections:

## Expected Behavior
## Code Localization
## Scope And Constraints
## Validation
## Open Questions

Evidence:

{evidence}
"""


def render_localization_fallback(localization: dict[str, Any]) -> str:
    lines = [
        "## Expected Behavior",
        "",
        "Implement only the behavior required by the Original Requirement; verify all claims against source.",
        "",
        "## Code Localization",
        "",
        "The following locations are inspection candidates produced from repository evidence:",
        "",
    ]
    for rank, item in enumerate(localization.get("locations", []), start=1):
        label = item.get("file", "")
        if item.get("symbol"):
            label += f":{item['symbol']}"
        lines.append(f"{rank}. `{label}` ({item.get('role', '')}, confidence={item.get('confidence', '')}) - {item.get('reason', '')}")
    lines.extend(
        [
            "",
            "## Scope And Constraints",
            "",
            "Treat locations as candidates, preserve unrelated behavior, and include only source-verified consistency updates.",
            "",
            "## Validation",
            "",
            "Read the primary and propagation locations, inspect related tests/configuration, then run the narrowest meaningful checks.",
            "",
            "## Open Questions",
            "",
        ]
    )
    questions = localization.get("unresolved_questions", []) or ["Confirm the final edit surface by reading source."]
    lines.extend(f"- {question}" for question in questions)
    return "\n".join(lines)


def render_localization_markdown(localization: dict[str, Any]) -> str:
    lines = [
        "# Code Localization",
        "",
        "This report is evidence for task preparation, not an automatic edit list.",
        "",
        "## Ranked Locations",
        "",
    ]
    for rank, item in enumerate(localization.get("locations", []), start=1):
        label = item.get("file", "")
        if item.get("symbol"):
            label += f":{item['symbol']}"
        lines.extend(
            [
                f"{rank}. `{label}`",
                f"   Role: {item.get('role', '')}; lines={item.get('start_line')} - {item.get('end_line')}; confidence={item.get('confidence', '')}",
                f"   Reason: {item.get('reason', '')}",
            ]
        )
    lines.extend(["", "## Related Tests And Files", ""])
    lines.extend(f"- Test: `{path}`" for path in localization.get("related_tests", []))
    lines.extend(f"- Related: `{path}`" for path in localization.get("related_files", []))
    if not localization.get("related_tests") and not localization.get("related_files"):
        lines.append("- None identified; use normal Chrys search.")
    return "\n".join(lines).rstrip() + "\n"


def render_localization_task_main(requirement_text: str, localization_path: Path, body: str, generation_mode: str) -> str:
    return "\n".join(
        [
            "# Augmented Requirement",
            "",
            "## Authority",
            "",
            "The Original Requirement below is authoritative and is preserved verbatim.",
            f"Generation mode: {generation_mode}.",
            "",
            "## Original Requirement",
            "",
            requirement_text.rstrip(),
            "",
            "## Code Localization",
            "",
            f"Read the ranked localization report before editing: `{localization_path.resolve()}`",
            "Every location is an inspection candidate and must be verified with source before editing.",
            "",
            body.rstrip(),
            "",
            "## Execution Contract",
            "",
            "- Use the Original Requirement as the source of truth.",
            "- Read and verify primary, propagation, and validation locations before editing.",
            "- Do not edit every candidate automatically or broaden the task beyond the requirement.",
            "- Use Chrys native tools for source reads, edits, tests, and patch generation.",
            "- Keep `.semantic-search/` artifacts out of the final patch.",
        ]
    ).rstrip() + "\n"


def render_main(requirement_text: str, routes: list[dict[str, Any]], *, generation_mode: str) -> str:
    lines = [
        "# Augmented Requirement",
        "",
        "## Authority",
        "",
        "Use this Augmented Requirement and its routed augmentation documents as the primary enhanced task brief for Chrys.",
        "Chrys receives this markdown task brief at run time; JSON facts and evidence bundles are generation inputs, not separate runtime instructions.",
        "The Original Requirement is preserved verbatim inside this document and remains the canonical source when conflicts arise.",
        "The Augmentation documents are task-solving material for the current benchmark task: expectations, scope, decomposition, validation, risks, and repository evidence.",
        "Codebase and semantic-search evidence should guide investigation and implementation planning. Verify code claims against source before editing and do not edit a file solely because it is listed.",
        "If an augmentation item broadens or conflicts with the Original Requirement, treat it as an augmentation defect and resolve it using the Original Requirement plus direct repository evidence.",
        f"Generation mode: {generation_mode}.",
        "",
        "## How To Use This Document",
        "",
        "1. Read this Augmented Requirement as the semantic-search task prompt; the Original Requirement is embedded below for fidelity.",
        "2. Use the Augmentation summaries and linked sub-documents to plan expected behavior, scope, implementation order, risks, and validation.",
        "3. Open code-related Augmentation documents before editing areas they mention, then choose the smallest complete coherent implementation that satisfies the enhanced task brief.",
        "4. Keep `.semantic-search/` artifacts out of the final patch.",
        "",
        "## Execution Contract",
        "",
        "- Goal: satisfy the preserved Original Requirement with the smallest complete coherent source change that is buildable and behavior-preserving.",
        "- Required behavior deltas: derive them from `Expected Behavior` and `Task Decomposition`, then verify each one against the source before editing.",
        "- Completeness rule: small is not the same as incomplete; include required new files, generated artifacts, build/install registration, exports, and metadata only when source inspection shows the requirement cannot work without them.",
        "- Non-goals: do not implement broad background/specification material, unrelated cleanup, or every candidate file listed by code perception.",
        "- Minimal implementation path: read the must-read augmentation documents, inspect candidate and missing surfaces tied to a concrete requirement delta, edit source-of-truth files first, keep only source-verified generated/registered outputs in sync, then validate.",
        "- Build and regression gate: if the patch touches generated, parser, build, packaging, or registration files, preserve executable bits and file metadata, then run or identify a build/import check before expanding scope.",
        "- Validation gate: run the narrowest meaningful tests/build checks available for the touched area, and treat pass-to-pass behavior as a hard constraint unless the Original Requirement explicitly changes it.",
        "- Stop condition: once the required behavior and required consistency surfaces are implemented and validated, stop expanding the patch unless source inspection shows another required consistency update.",
        "",
        "## Completeness Contract",
        "",
        "- Semantic search can only rank files that exist in the current checkout; if the Original Requirement introduces a new module, API, syntax form, generated structure, or registration surface, first verify that requirement in source, then create or update the minimal required surfaces even when they are absent from ranked evidence.",
        "- For parser, grammar, schema, generated-code, build, or packaging work, verify the repository's source-of-truth chain and synchronize checked-in generated artifacts or metadata only when the project expects that for the touched source change.",
        "- Generated/build synchronization must be behavior-preserving: do not drop executable bits, alter unrelated generated output, or change build metadata merely because it is listed as evidence.",
        "- Candidate code evidence is useful context, but the final patch must be complete for the preserved Original Requirement, not merely local to the highest-ranked existing files.",
        "",
        "## Augmentation Reading Order",
        "",
        "1. Read `Expected Behavior`, `Scope Boundary`, `Task Decomposition`, `Validation Plan`, and `Anti-patterns and Failure Modes` before editing.",
        "2. Open `Likely Implementation Surfaces` and `Code Details To Inspect` only to verify concrete source paths and choose a minimal edit set.",
        "3. Use `Codebase Context`, `Relevant Existing Patterns`, and `Assumptions and Open Questions` to resolve uncertainty, not to broaden scope.",
        "",
        "## Benchmark Task Guardrails",
        "",
        "- Do not convert broad background material into additional requirements unless the Original Requirement actually demands it.",
        "- Do not implement every listed implementation surface. Inspect, choose, edit, and validate only what is necessary for the current task.",
        "- Do not add benchmark-specific shortcuts, hidden-answer assumptions, or hard-coded examples.",
        "- If the augmented plan starts to imply a large rewrite, re-check the Original Requirement and narrow the patch to the behavior that must change.",
        "- If the augmented plan starts to imply an artificially tiny patch, re-check whether missing files, generated artifacts, registration, exports, or build metadata are required for correctness.",
        "- If a build or pass-to-pass check fails after generated/build edits, first inspect whether those edits are unnecessary, stale, or metadata-damaging before adding more scope.",
        "",
        "## Original Requirement",
        "",
        requirement_text.rstrip(),
        "",
        "## Augmentation",
        "",
    ]
    for route in routes:
        lines.extend(format_route(route))
        lines.append("")
    lines.extend(
        [
            "## Execution Notes",
            "",
            "- The summaries above are intentionally compact; open the linked documents for evidence and lower-level details.",
            "- High-confidence code details are still not a substitute for reading the actual source before editing.",
            "- If any suggested implementation path seems inconsistent with repository evidence, resolve it using direct source inspection and the embedded Original Requirement.",
            "- A good patch is task-correct, scoped, complete, buildable, and behavior-preserving; the augmentation package exists to help reach that patch, not to enlarge or undercut it.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def format_route(route: dict[str, Any]) -> list[str]:
    quality = route.get("quality", {})
    quality_note = "ok" if quality.get("required_headings_present") else "check headings"
    lines = [
        f"- {route['topic']}",
        f"  Path: {route['path']}",
        f"  Priority: {route['priority']}",
        "  Role: primary augmented task package document",
        "  Confidence: medium",
        f"  Structure: {quality_note}; must-implement bullets={quality.get('must_implement_bullets', 0)}",
        "  Summary:",
    ]
    summary = route.get("summary") or ["No high-confidence summary was generated; open the document if this topic matters."]
    lines.extend(f"    - {markdown_escape_line(item)}" for item in summary[:8])
    return lines


def write_manifest(
    path: Path,
    requirement: Path,
    facts_path: Path | None,
    out: Path,
    augmentation_dir: Path,
    routes: list[dict[str, Any]],
    generation: dict[str, Any],
) -> None:
    inputs = [{"path": str(requirement), "sha1": sha1_path(requirement), "source": "original_requirement"}]
    if facts_path is not None:
        inputs.append({"path": str(facts_path), "sha1": sha1_path(facts_path), "source": "general_and_semantic_search_code_facts"})
    else:
        inputs.append(
            {
                "path": "",
                "sha1": "",
                "source": "requirement_only_fallback",
                "note": "No code-facts.json was provided; augmentation used the Original Requirement without repository code evidence.",
            }
        )
    write_json(
        path,
        {
            "format": "semantic-search-augmentation-manifest",
            "created_at": now_iso(),
            "generation": generation,
            "inputs": inputs,
            "outputs": [
                {"path": str(out), "source": "augmented_requirement"},
                {"path": str(augmentation_dir), "source": "augmentation_dir"},
            ],
            "routes": routes,
        },
    )


def build_fallback_docs(facts: dict[str, Any]) -> tuple[dict[str, str], dict[str, list[str]]]:
    docs: dict[str, str] = {}
    summaries: dict[str, list[str]] = {}
    docs["expected-behavior"], summaries["expected-behavior"] = expected_behavior_doc(facts)
    docs["scope-boundary"], summaries["scope-boundary"] = scope_boundary_doc(facts)
    docs["task-decomposition"], summaries["task-decomposition"] = task_decomposition_doc(facts)
    docs["codebase-context"], summaries["codebase-context"] = codebase_context_doc(facts)
    docs["implementation-surfaces"], summaries["implementation-surfaces"] = implementation_surfaces_doc(facts)
    docs["existing-patterns"], summaries["existing-patterns"] = existing_patterns_doc(facts)
    docs["code-details"], summaries["code-details"] = code_details_doc(facts)
    docs["validation-plan"], summaries["validation-plan"] = validation_plan_doc(facts)
    docs["anti-patterns"], summaries["anti-patterns"] = anti_patterns_doc(facts)
    docs["open-questions"], summaries["open-questions"] = open_questions_doc(facts)
    for slug, title, _priority in TOPICS:
        docs[slug] = ensure_document_shape(docs[slug], title, summaries[slug])
    return docs, summaries


def render_evidence_bundle(requirement_text: str, facts: dict[str, Any], *, max_chars: int) -> str:
    lines = [
        "# Semantic Search Evidence Bundle",
        "",
        "This bundle is runtime evidence for an LLM to generate task-centric requirement augmentation.",
        "It contains the original requirement, repository perception, global codebase perception, and task-specific semantic perception.",
        "Repository perception may merge builtin static evidence with optional CodeGraph CLI evidence. Code facts are split conceptually into General Code Facts (repository-wide structure, topology, graph evidence, and risks) and Semantic Search Facts (task-specific requirement signals, ranked evidence, surfaces, links, and validation hints).",
        "The code evidence is a source for clarifying the current task; it is not a required edit list.",
        "",
        "## Original Requirement",
        "",
        requirement_text.rstrip(),
        "",
        "## General Code Facts",
        "",
        "General Code Facts describe the repository independent of a single ranked edit path.",
        "",
        "## Repository Perception",
        "",
    ]
    repository_perception = facts.get("repository_perception", {})
    if repository_perception:
        lines.extend(render_repository_evidence(repository_perception))
    else:
        lines.append("- No merged repository perception artifact was available; rely on global/task-specific semantic evidence and source inspection.")
    lines.extend(
        [
            "",
            "## Global Codebase Perception",
            "",
        ]
    )
    global_perception = facts.get("global_perception", {})
    if global_perception:
        lines.extend(render_global_evidence(global_perception))
    else:
        lines.append("- No global perception artifact was available; rely on task-specific semantic evidence and source inspection.")
    lines.extend(
        [
            "",
            "## Semantic Search Facts",
            "",
            "Semantic Search Facts connect the Original Requirement to task-specific repository evidence.",
            "",
            "## Semantic Task Perception",
            "",
            "### Requirement Signals",
            "",
        ]
    )
    signals = facts.get("requirement_signals", [])
    lines.extend(bullet_lines([f"`{item.get('term', '')}` ({item.get('category', '')})" for item in signals[:80]]))
    repo = facts.get("repo_map", {})
    lines.extend(["", "### Task-local Repository Map", "", "#### Top-level Areas"])
    for top, counts in list(repo.get("top_levels", {}).items())[:30]:
        lines.append(f"- `{top}`: {counts}")
    lines.extend(["", "#### Generated-looking Files"])
    lines.extend(bullet_lines([f"`{item}`" for item in repo.get("generated_files", [])[:40]]))
    lines.extend(["", "#### Build Files"])
    lines.extend(bullet_lines([f"`{item}`" for item in repo.get("build_files", [])[:40]]))
    lines.extend(["", "#### Test Roots"])
    lines.extend(bullet_lines([f"`{item}`" for item in repo.get("test_roots", [])[:40]]))
    lines.extend(["", "### Global/Semantic Links", ""])
    for link in facts.get("global_semantic_links", [])[:24]:
        target = link.get("cluster") or link.get("path") or "(unknown)"
        lines.append(f"- `{target}` ({link.get('kind', '')})")
        lines.append(f"  Source: {link.get('source', '')}")
        lines.append(f"  Confidence: {link.get('confidence', '')}")
        lines.append(f"  Why: {link.get('why', '')}")
        lines.append(f"  Action: {link.get('action', '')}")
    lines.extend(["", "### Feature Completeness and Missing Surface Hints", ""])
    capabilities = facts.get("capability_requirements", {})
    if capabilities:
        lines.extend(render_capability_evidence(capabilities))
    else:
        lines.append("- No explicit missing-surface or consistency-chain hints were generated.")
    lines.extend(["", "### Inferred Implementation Surfaces", ""])
    for surface in facts.get("implementation_surfaces", [])[:12]:
        lines.append(f"- `{surface.get('name', '')}`")
        lines.append(f"  Source: {surface.get('source', '')}")
        lines.append(f"  Confidence: {surface.get('confidence', '')}")
        lines.append(f"  Why: {surface.get('why', '')}")
        lines.append(f"  Inspect: {', '.join(f'`{item}`' for item in surface.get('inspect', [])[:10]) or '(none)'}")
        lines.append(f"  Risk: {surface.get('risk', '')}")
    lines.extend(["", "### Ranked Code Evidence", ""])
    for detail in facts.get("ranked_file_details", [])[:12]:
        lines.append(f"### `{detail.get('path', '')}`")
        lines.append("")
        lines.append(f"- Kind: {detail.get('kind', '')}")
        lines.append(f"- Language: {detail.get('language', '')}")
        lines.append(f"- Score: {detail.get('score', 0)}")
        symbols = detail.get("symbols", [])
        if symbols:
            lines.append("- Symbols:")
            for symbol in symbols[:12]:
                lines.append(
                    f"  - `{symbol.get('name', '')}` ({symbol.get('kind', '')}) line {symbol.get('line', '')}: {markdown_escape_line(symbol.get('signature', ''))}"
                )
        preview = detail.get("preview", "")
        if preview:
            lines.extend(["", "Preview:", "", "```text", preview[:1800].rstrip(), "```"])
        lines.append("")
    lines.extend(["## Existing Patterns", ""])
    for item in facts.get("existing_patterns", [])[:12]:
        lines.append(f"- Claim: {item.get('pattern', '')}")
        lines.append(f"  Source: {item.get('source', '')}")
        lines.append(f"  Confidence: {item.get('confidence', '')}")
        lines.append(f"  Action: {item.get('action', '')}")
    lines.extend(["", "## Validation Hints", ""])
    for item in facts.get("validation_hints", [])[:18]:
        lines.append(f"- `{item.get('path', '')}` ({item.get('kind', '')})")
        lines.append(f"  Source: {item.get('source', '')}")
        lines.append(f"  Confidence: {item.get('confidence', '')}")
        lines.append(f"  Action: {item.get('action', '')}")
    lines.extend(["", "## Script Uncertainties", ""])
    for item in facts.get("uncertainties", [])[:12]:
        lines.append(f"- {item.get('question', '')}")
        lines.append(f"  Source: {item.get('source', '')}")
        lines.append(f"  Confidence: {item.get('confidence', '')}")
    text = "\n".join(lines).rstrip() + "\n"
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n[Evidence bundle truncated by semantic-search max evidence chars.]\n"


def render_capability_evidence(capabilities: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    missing = capabilities.get("new_or_missing_surfaces", [])
    lines.extend(["#### New or Missing Surfaces"])
    if missing:
        for item in missing[:12]:
            lines.append(f"- `{item.get('name', '')}` ({item.get('kind', '')})")
            lines.append(f"  Source: {item.get('source', '')}")
            lines.append(f"  Confidence: {item.get('confidence', '')}")
            lines.append(f"  Action: {item.get('action', '')}")
    else:
        lines.append("- No named missing surface was detected; still check whether the Original Requirement implies new files or registration.")
    chains = capabilities.get("consistency_chains", [])
    lines.extend(["", "#### Required Consistency Chains"])
    if chains:
        for item in chains[:8]:
            lines.append(f"- `{item.get('name', '')}`")
            lines.append(f"  Source: {item.get('source', '')}")
            lines.append(f"  Confidence: {item.get('confidence', '')}")
            lines.append(f"  Action: {item.get('action', '')}")
    else:
        lines.append("- No specific consistency chain was detected.")
    limits = capabilities.get("localization_limits", [])
    lines.extend(["", "#### Localization Limits"])
    for item in limits[:8]:
        lines.append(f"- Claim: {item.get('claim', '')}")
        lines.append(f"  Source: {item.get('source', '')}")
        lines.append(f"  Confidence: {item.get('confidence', '')}")
        lines.append(f"  Action: {item.get('action', '')}")
    return lines


def render_global_evidence(global_perception: dict[str, Any]) -> list[str]:
    lines = []
    lines.append(f"- Backend: {global_perception.get('backend', '')}")
    overview = global_perception.get("overview", {})
    stats = overview.get("stats", {})
    lines.append(f"- Indexed files: {stats.get('file_count', 0)}")
    lines.append(f"- Indexed symbols: {stats.get('symbol_count', 0)}")
    lines.append(f"- Source/test/generated/build counts: {overview.get('source_count', 0)}/{overview.get('test_count', 0)}/{overview.get('generated_count', 0)}/{overview.get('build_count', 0)}")
    lines.extend(["", "### Module Clusters"])
    for cluster in global_perception.get("module_clusters", [])[:16]:
        lines.append(f"- `{cluster.get('name', '')}`: {cluster.get('file_count', 0)} files, kinds={cluster.get('kind_counts', {})}")
        reps = ", ".join(f"`{item}`" for item in cluster.get("representative_files", [])[:5])
        if reps:
            lines.append(f"  Representative files: {reps}")
    lines.extend(["", "### Global Entrypoints"])
    for entry in global_perception.get("entrypoints", [])[:24]:
        lines.append(f"- `{entry.get('path', '')}` ({entry.get('kind', '')}, score={entry.get('score', 0)}): {entry.get('reason', '')}")
    lines.extend(["", "### Dependency Hints"])
    for edge in global_perception.get("dependency_hints", [])[:32]:
        lines.append(f"- `{edge.get('from', '')}` -> `{edge.get('to', '')}` ({edge.get('kind', '')})")
    topology = global_perception.get("topology", {})
    lines.extend(["", "### Build/Test/Generated Topology"])
    lines.append("Generated-looking files:")
    lines.extend(bullet_lines([f"`{item}`" for item in topology.get("generated_files", [])[:24]]))
    lines.append("Build files:")
    lines.extend(bullet_lines([f"`{item}`" for item in topology.get("build_files", [])[:24]]))
    lines.append("Test roots:")
    lines.extend(bullet_lines([f"`{item.get('root')}` ({item.get('files')} files)" for item in topology.get("test_roots", [])[:24]]))
    lines.extend(["", "### Source-of-truth Hints"])
    lines.extend(
        bullet_lines(
            [
                f"`{item.get('path', '')}` ({item.get('kind', '')}/{item.get('language', '')}): {item.get('why', '')}"
                for item in global_perception.get("source_of_truth_hints", [])[:32]
            ]
        )
    )
    external = global_perception.get("external_graph_summary", {})
    lines.extend(["", "### External Graph"])
    lines.append(f"- Present: {external.get('present', False)}")
    lines.append(f"- Note: {external.get('note', '')}")
    lines.extend(["", "### Global Risks"])
    for risk in global_perception.get("global_risks", [])[:12]:
        lines.append(f"- Claim: {risk.get('claim', '')}")
        lines.append(f"  Source: {risk.get('source', '')}")
        lines.append(f"  Confidence: {risk.get('confidence', '')}")
        lines.append(f"  Action: {risk.get('action', '')}")
    return lines


def render_repository_evidence(repository_perception: dict[str, Any]) -> list[str]:
    lines = []
    lines.append(f"- Backend: {repository_perception.get('backend', '')}")
    overview = repository_perception.get("overview", {})
    stats = overview.get("stats", {})
    lines.append(f"- Indexed files: {stats.get('file_count', 0)}")
    lines.append(f"- Indexed symbols: {stats.get('symbol_count', 0)}")
    lines.append(
        f"- Source/test/generated/build counts: {overview.get('source_count', 0)}/{overview.get('test_count', 0)}/{overview.get('generated_count', 0)}/{overview.get('build_count', 0)}"
    )
    lines.extend(["", "### Repository Module Map"])
    for name, values in list(repository_perception.get("module_map", {}).items())[:18]:
        lines.append(f"- `{name}`: {values.get('file_count', 0)} files, kinds={values.get('kinds', {})}, languages={values.get('languages', {})}")
        reps = ", ".join(f"`{item}`" for item in values.get("representative_files", [])[:5])
        if reps:
            lines.append(f"  Representative files: {reps}")
    codegraph = repository_perception.get("codegraph_layer", {})
    lines.extend(["", "### CodeGraph Evidence"])
    lines.append(f"- Available: {codegraph.get('available', False)}")
    lines.append(f"- Status: {codegraph.get('status', '')}")
    for key, value in codegraph.get("summary", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "#### CodeGraph Query Evidence"])
    for item in codegraph.get("query_evidence", [])[:6]:
        lines.append(f"- Query `{item.get('query', '')}` ok={item.get('ok')} rc={item.get('returncode')}")
        output = item.get("stdout") or item.get("stderr") or ""
        if output:
            lines.extend(["", "```text", output[:1600].rstrip(), "```"])
    lines.extend(["", "#### CodeGraph Symbol Relationship Evidence"])
    for item in codegraph.get("relationship_evidence", [])[:6]:
        lines.append(f"- Symbol `{item.get('symbol', '')}`")
        for key in ("node", "callers", "callees", "impact"):
            result = item.get(key, {})
            lines.append(f"  - {key}: ok={result.get('ok')} rc={result.get('returncode')}")
            output = result.get("stdout") or result.get("stderr") or ""
            if output and key in {"callers", "callees", "impact"}:
                lines.extend(["", "```text", output[:1200].rstrip(), "```"])
    source = repository_perception.get("source_of_truth", {})
    lines.extend(["", "### Source-of-truth and Validation Surface"])
    lines.extend(
        bullet_lines(
            [
                f"`{item.get('path', '')}` ({item.get('kind', '')}/{item.get('language', '')}): {item.get('why', '')}"
                for item in source.get("hints", [])[:24]
            ]
        )
    )
    validation = repository_perception.get("validation_surface", {})
    lines.extend(["", "Validation roots:"])
    lines.extend(bullet_lines([f"`{item.get('root')}` ({item.get('files')} files)" for item in validation.get("test_roots", [])[:20]]))
    lines.extend(["", "### Repository Risks"])
    for risk in repository_perception.get("risks", [])[:12]:
        lines.append(f"- Claim: {risk.get('claim', '')}")
        lines.append(f"  Source: {risk.get('source', '')}")
        lines.append(f"  Confidence: {risk.get('confidence', '')}")
        lines.append(f"  Action: {risk.get('action', '')}")
    return lines


def render_llm_prompt(evidence_bundle: str) -> str:
    topics = "\n".join(f"- {slug}: {title} ({priority})" for slug, title, priority in TOPICS)
    return f"""You generate a task-centric Augmented Requirement for a coding agent.

The script has already collected static repository evidence. Your job is to reason from:
1. the Original Requirement, which is the canonical source;
2. Repository Perception: merged builtin static evidence plus optional CodeGraph graph evidence, including module map, symbol relationships, caller/callee or impact observations when available, generated/build/test topology, and repository risks;
3. Global Codebase Perception: architecture, module clusters, dependency hints, entrypoints, generated/build/test topology, and global risks;
4. Semantic Task Perception: requirement signals, ranked task evidence, implementation surfaces, existing patterns, validation hints, and global/semantic links;
5. normal engineering expectations implied by the requirement and code evidence.

Goal:
- Produce the primary enhanced task brief for the semantic-search Chrys run.
- Strengthen the current benchmark task requirement so the coding agent better understands what to build, what not to build, how to validate it, and where to inspect.
- Code understanding is only one augmentation step. Use repository perception and CodeGraph evidence to strengthen code awareness, while requirement clarification, expectation extraction, scope control, validation planning, and failure-mode analysis remain equally important.
- The output should be richer and more actionable than the Original Requirement alone while still staying faithful to that original task.
- The output should guide the coding agent toward a scoped, complete, buildable patch for the current task. It must not become a broad rewrite plan or an incomplete partial patch.

Do not invent gold patches, hidden tests, target PR diffs, or benchmark-specific answers.
Do not overfit to one example.
Do not turn likely implementation surfaces into mandatory edits. Say "inspect", "verify", "likely", or "candidate" unless the Original Requirement and source evidence make an edit unavoidable.
Do not add new feature requirements beyond the Original Requirement. Broad background material should be converted into task-relevant clarifications and risks, not extra implementation scope.
Do not omit required new files, generated artifacts, build/install registration, exports, metadata, parser/schema consumers, or validation surfaces merely to keep the patch small.
Do not touch generated/build surfaces as a precautionary broadening step; require source evidence, preserve executable bits/file metadata, and keep pass-to-pass behavior stable.
Use the evidence to make the task brief more complete: expectations, scope boundaries, decomposition, code evidence, validation, anti-patterns, and open questions.
When evidence is uncertain, say how to verify it instead of instructing the agent to implement it.
Remember that repository evidence only ranks files that exist in the current checkout. If the Original Requirement introduces a new module, package, API, syntax form, generated artifact, or registration surface, say that missing/new surfaces may be required.

Return JSON only. No Markdown fences around the JSON.
Schema:
{{
  "documents": {{
    "<topic-slug>": {{
      "summary": ["3-8 concise high-priority bullets"],
      "markdown": "# <Topic Title>\\n\\n## High Priority\\n...\\n\\n## Supporting Details\\n...\\n\\n## Low-confidence Notes\\n...\\n\\n## Evidence Index\\n..."
    }}
  }}
}}

Required topic slugs:
{topics}

Each markdown document must include these headings exactly:
- ## High Priority
- ## Must Implement
- ## Must Preserve
- ## Should Inspect
- ## Do Not Do
- ## Validation
- ## Low-confidence Notes
- ## Evidence Index

Evidence style:
- Mark claims with Source: original_requirement | code_evidence | inferred | uncertain.
- Include Evidence path/function names when available.
- Include Confidence and Action for important claims.
- Keep the content useful for solving the current benchmark task, not just locating files.
- For code-related topics, distinguish "inspection candidate" from "required edit".
- Prefer language that narrows ambiguity and controls risk over language that expands scope, while preserving all required cross-file consistency.
- `Must Implement` should contain only behavior deltas directly required by the Original Requirement or verified source evidence. Keep it short.
- `Must Implement` should include completeness obligations only when the behavior cannot work without new files, generated artifacts, registration, exports, metadata, parser/schema consumers, or build updates.
- `Must Preserve` should explicitly protect pass-to-pass behavior, buildability, executable bits, generated/build file metadata, and existing diagnostics when generated/build/parser surfaces are mentioned.
- `Should Inspect` may list candidate files, symbols, graph edges, and patterns. These are not edit mandates.
- `Do Not Do` must explicitly prevent broad rewrites, unrelated feature completion, benchmark-specific shortcuts, and editing every listed candidate surface.
- `Validation` should name concrete validation surfaces, commands, or test roots when evidence supports them; otherwise state the narrow verification strategy.
- Build a chain where possible: requirement signal -> existing code behavior/path -> likely missing delta -> validation point.
- For feature backports and large language/runtime changes, explicitly distinguish broad unrelated scope from required cross-file consistency; broad unrelated scope is a non-goal, required consistency must not be dropped.

Evidence bundle:

{evidence_bundle}
"""


def call_llm_for_augmentation(args: argparse.Namespace, prompt: str, *, system_prompt: str | None = None) -> str:
    mock = os.environ.get("SEMANTIC_SEARCH_LLM_MOCK_RESPONSE")
    if mock:
        return mock
    profile = load_model_profile(args.model_profile)
    provider = str(profile.get("provider", "openai")).lower()
    if provider not in {"openai", "deepseek-openai"}:
        raise ScriptError(f"semantic-search LLM augmentation only supports OpenAI-compatible profiles, got provider={provider!r}")
    model_id = str(profile.get("model_id", "")).strip()
    if not model_id:
        raise ScriptError("model profile does not define model_id")
    base_url = str(profile.get("base_url", "")).strip()
    if not base_url:
        base_url = "https://api.openai.com/v1" if provider == "openai" else "https://api.deepseek.com"
    api_key = resolve_env_templates_simple(str(profile.get("api_key", "")), location="model profile api_key")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    headers.update(load_profile_headers(profile))
    system = system_prompt or (
        "You are a senior software requirements engineer. "
        "Generate rigorous task-centric requirement augmentation from original requirements and code evidence. "
        "Create the primary enhanced task brief for the coding agent while staying faithful to the original task. "
        "Return valid JSON only."
    )
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    }
    max_tokens = os.environ.get("SEMANTIC_SEARCH_LLM_MAX_TOKENS", "").strip()
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    return post_openai_chat(base_url, payload, headers, args.llm_timeout)


def load_model_profile(value: str) -> dict[str, Any]:
    if not value:
        raise ScriptError("no model profile selected for LLM augmentation")
    raw = os.path.expanduser(os.path.expandvars(value))
    candidates: list[Path] = []
    value_path = Path(raw)
    if value_path.suffix in {".yaml", ".yml"} or value_path.is_absolute() or "/" in raw:
        candidates.append(resolve_path(value_path))
    else:
        model_dir = Path.home() / ".chrys" / "models"
        candidates.append(model_dir / f"{raw}.yaml")
        candidates.append(model_dir / f"{raw}.yml")
        if model_dir.is_dir():
            for path in sorted(model_dir.glob("*.y*ml")):
                try:
                    data = read_yaml_mapping(path)
                except ScriptError:
                    continue
                if data.get("name") == value or data.get("id") == value:
                    return data
    for path in candidates:
        if path.is_file():
            return read_yaml_mapping(path)
    raise ScriptError(f"model profile not found for LLM augmentation: {value}")


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        data = read_simple_yaml_mapping(path)
    except Exception as err:
        raise ScriptError(f"failed to read model profile {path}: {err}") from err
    if not isinstance(data, dict):
        raise ScriptError(f"model profile is not a mapping: {path}")
    data.setdefault("id", path.stem)
    return data


def read_simple_yaml_mapping(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as err:
        raise ScriptError(f"failed to read model profile {path}: {err}") from err
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        data[key.strip()] = value
    data.setdefault("id", path.stem)
    return data


def resolve_env_templates_simple(value: str, *, location: str) -> str:
    pattern = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        resolved = os.environ.get(name)
        if not resolved:
            raise ScriptError(f"environment variable {name!r} is required by {location}")
        return resolved

    return os.path.expandvars(pattern.sub(replace, value))


def load_profile_headers(profile: dict[str, Any]) -> dict[str, str]:
    raw = profile.get("http_headers") or ""
    if not raw:
        return {}
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = json.loads(str(raw))
        except ValueError:
            return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): resolve_env_templates_simple(str(value), location=f"model profile header {key}") for key, value in data.items()}


def post_openai_chat(base_url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as err:
        message = err.read().decode("utf-8", errors="replace")[:1200]
        raise ScriptError(f"LLM augmentation HTTP {err.code}: {message}") from err
    except urllib.error.URLError as err:
        raise ScriptError(f"LLM augmentation request failed: {err.reason}") from err
    except TimeoutError as err:
        raise ScriptError(f"LLM augmentation timed out after {timeout:g}s") from err
    try:
        data = json.loads(raw)
    except ValueError as err:
        raise ScriptError(f"LLM augmentation returned non-JSON response: {raw[:1200]}") from err
    choices = data.get("choices")
    if not choices:
        raise ScriptError(f"LLM augmentation response has no choices: {raw[:1200]}")
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ScriptError(f"LLM augmentation response has no text content: {raw[:1200]}")
    return content


def parse_llm_payload(text: str) -> dict[str, Any]:
    stripped = strip_markdown_fence(text.strip())
    try:
        data = json.loads(stripped)
    except ValueError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            raise ScriptError("LLM augmentation response did not contain a JSON object")
        try:
            data = json.loads(match.group(0))
        except ValueError as err:
            raise ScriptError(f"LLM augmentation JSON could not be parsed: {err}") from err
    if not isinstance(data, dict):
        raise ScriptError("LLM augmentation JSON root is not an object")
    return data


def strip_markdown_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def normalize_llm_documents(
    payload: dict[str, Any],
    fallback_docs: dict[str, str],
    fallback_summaries: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, list[str]], list[str]]:
    documents = payload.get("documents")
    if not isinstance(documents, dict):
        raise ScriptError("LLM augmentation JSON missing documents object")
    docs: dict[str, str] = {}
    summaries: dict[str, list[str]] = {}
    missing: list[str] = []
    for slug, title, _priority in TOPICS:
        item = documents.get(slug)
        if item is None:
            item = documents.get(title)
        if not isinstance(item, dict):
            docs[slug] = fallback_docs[slug]
            summaries[slug] = fallback_summaries[slug]
            missing.append(slug)
            continue
        raw_markdown = str(item.get("markdown", "")).strip()
        raw_summary = item.get("summary", [])
        summary = normalize_summary(raw_summary)
        if not raw_markdown:
            docs[slug] = fallback_docs[slug]
            summaries[slug] = fallback_summaries[slug]
            missing.append(slug)
            continue
        if not summary:
            summary = fallback_summaries[slug]
        docs[slug] = ensure_document_shape(raw_markdown, title, summary)
        summaries[slug] = summary[:8]
    return docs, summaries, missing


def normalize_summary(raw: Any) -> list[str]:
    if isinstance(raw, list):
        values = [markdown_escape_line(str(item)) for item in raw]
    elif isinstance(raw, str):
        values = [markdown_escape_line(line.lstrip("-* ").strip()) for line in raw.splitlines()]
    else:
        values = []
    return [item for item in values if item][:8]


def ensure_document_shape(markdown: str, title: str, summary: list[str]) -> str:
    text = markdown.strip()
    if not text.startswith("#"):
        text = f"# {title}\n\n{text}"
    if TASK_PACKAGE_NOTE not in text:
        lines = text.splitlines()
        if lines and lines[0].startswith("#"):
            text = "\n".join([lines[0], "", TASK_PACKAGE_NOTE, "", *lines[1:]]).strip()
        else:
            text = f"{TASK_PACKAGE_NOTE}\n\n{text}"
    if SCOPE_GUARD_NOTE not in text:
        lines = text.splitlines()
        insert_at = 3 if len(lines) >= 3 and lines[0].startswith("#") else 0
        lines[insert_at:insert_at] = [SCOPE_GUARD_NOTE, ""]
        text = "\n".join(lines).strip()
    if "## High Priority" not in text:
        text += "\n\n## High Priority\n\n" + "\n".join(f"- {item}" for item in summary)
    defaults = {
        "## Must Implement": [
            "Implement only behavior deltas that are directly required by the Original Requirement after source verification.",
            "Keep the patch scoped to the smallest coherent source-of-truth change set.",
        ],
        "## Must Preserve": [
            "Preserve existing pass-to-pass behavior and public behavior not mentioned by the Original Requirement.",
            "Preserve generated/build/test topology, executable bits, and file metadata unless direct source inspection proves an update is required.",
        ],
        "## Should Inspect": [
            "Inspect candidate files, symbols, and graph evidence before deciding whether any listed surface belongs in the patch.",
            "Verify each code-evidence claim with direct source reads.",
        ],
        "## Do Not Do": [
            "Do not implement broad background material, unrelated cleanup, or whole-spec completion.",
            "Do not edit every candidate implementation surface just because it is listed.",
            "Do not add benchmark-specific shortcuts or hard-coded examples.",
        ],
        "## Validation": [
            "Run the narrowest meaningful tests or build checks for the touched area.",
            "If validation is expensive or unclear, document the intended narrow check and inspect existing test roots first.",
        ],
        "## Low-confidence Notes": [
            "Treat uncertain augmentation items as verification prompts, not implementation instructions.",
        ],
        "## Evidence Index": [
            "Original Requirement and generated semantic-search evidence bundle.",
        ],
    }
    for heading, bullets in defaults.items():
        if heading not in text:
            text += f"\n\n{heading}\n\n" + "\n".join(f"- {item}" for item in bullets)
    return text.rstrip() + "\n"


def document_quality(markdown: str) -> dict[str, Any]:
    missing = [heading for heading in DOC_REQUIRED_HEADINGS if heading not in markdown]
    return {
        "required_headings_present": not missing,
        "missing_headings": missing,
        "must_implement_bullets": count_section_bullets(markdown, "## Must Implement"),
        "should_inspect_bullets": count_section_bullets(markdown, "## Should Inspect"),
        "do_not_do_bullets": count_section_bullets(markdown, "## Do Not Do"),
        "validation_bullets": count_section_bullets(markdown, "## Validation"),
    }


def count_section_bullets(markdown: str, heading: str) -> int:
    lines = markdown.splitlines()
    try:
        start = lines.index(heading) + 1
    except ValueError:
        return 0
    count = 0
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if line.lstrip().startswith("- "):
            count += 1
    return count


def summary_from(items: list[str], limit: int = 6) -> list[str]:
    return [markdown_escape_line(item) for item in items if markdown_escape_line(item)][:limit]


def evidence_block(item: dict[str, Any]) -> list[str]:
    evidence = item.get("evidence")
    if isinstance(evidence, dict):
        return [f"  Evidence: `{evidence.get('path', '')}` ({evidence.get('kind', '')}, score={evidence.get('score', 0)})"]
    if isinstance(evidence, list):
        values = []
        for entry in evidence[:4]:
            if isinstance(entry, dict):
                values.append(f"`{entry.get('path', '')}`")
        if values:
            return [f"  Evidence: {', '.join(values)}"]
    return ["  Evidence: (not available)"]


def capability_high_priority(facts: dict[str, Any], *, limit: int = 4) -> list[str]:
    capabilities = facts.get("capability_requirements", {})
    items: list[str] = []
    for hint in capabilities.get("new_or_missing_surfaces", [])[:limit]:
        items.append(f"Original Requirement may require a new or absent surface `{hint.get('name', '')}`.")
    for chain in capabilities.get("consistency_chains", [])[:limit]:
        items.append(f"Source-verified completeness chain `{chain.get('name', '')}` may be required: {chain.get('action', '')}")
    return summary_from(items, limit=limit)


def capability_should_inspect_lines(facts: dict[str, Any]) -> list[str]:
    capabilities = facts.get("capability_requirements", {})
    lines: list[str] = []
    missing = capabilities.get("new_or_missing_surfaces", [])
    if missing:
        lines.extend(["### New or Missing Surfaces", ""])
        for item in missing[:10]:
            lines.append(f"- Claim: `{item.get('name', '')}` may be a required new or absent surface.")
            lines.append(f"  Source: {item.get('source', 'original_requirement')}")
            lines.append(f"  Confidence: {item.get('confidence', 'medium')}")
            lines.append(f"  Action: {item.get('action', '')}")
    chains = capabilities.get("consistency_chains", [])
    if chains:
        lines.extend(["", "### Required Consistency Chains", ""])
        for item in chains[:8]:
            lines.append(f"- Claim: `{item.get('name', '')}` may be required for a complete patch.")
            lines.append(f"  Source: {item.get('source', 'inferred')}")
            lines.append(f"  Confidence: {item.get('confidence', 'medium')}")
            lines.append(f"  Action: {item.get('action', '')}")
    return lines


def expected_behavior_doc(facts: dict[str, Any]) -> tuple[str, list[str]]:
    signals = facts.get("requirement_signals", [])
    examples = [signal["term"] for signal in signals if signal.get("category") in {"api", "syntax", "error"}][:10]
    summary = summary_from(
        [
            "Derive acceptance criteria from the preserved Original Requirement before editing.",
            "Preserve existing pass-to-pass behavior while adding only the behavior required for the current task.",
            *[f"Pay attention to requirement signal `{item}`." for item in examples[:4]],
            *capability_high_priority(facts),
        ]
    )
    lines = topic_header("Expected Behavior", summary)
    lines.extend(
        [
            "## Must Implement",
            "",
            "- Derive a compact acceptance checklist from the preserved Original Requirement before editing.",
            "- Implement only the behavior deltas that are directly required by that checklist.",
            "- Include required new files, exports, generated artifacts, registration, metadata, and build updates only when the checklist cannot work without them.",
            "",
            "## Must Preserve",
            "",
            "- Preserve existing behavior for inputs, APIs, and workflows not changed by the Original Requirement.",
            "- Preserve pass-to-pass behavior while adding the requested capability.",
            "- Preserve generated/build file permissions, executable bits, and metadata when those files must be touched.",
            "",
            "## Should Inspect",
            "",
        ]
    )
    lines.extend(bullet_lines([f"Requirement signal: `{item}`" for item in examples], empty="- No specific examples were extracted."))
    capability_lines = capability_should_inspect_lines(facts)
    if capability_lines:
        lines.extend(["", *capability_lines])
    lines.extend(
        [
            "",
            "## Do Not Do",
            "",
            "- Do not implement unrelated background/specification material that is not required by the Original Requirement.",
            "- Do not hard-code benchmark examples as behavior.",
            "",
            "## Validation",
            "",
            "- Validate the specific behavior deltas, required consistency surfaces, buildability when generated/build files are touched, and a nearby existing behavior path when possible.",
            "",
            "## Low-confidence Notes",
            "",
            "- Treat extracted signals as planning hints; verify exact behavior in the Original Requirement.",
            "",
            "## Evidence Index",
            "",
        ]
    )
    lines.extend(bullet_lines([f"`{signal['term']}` from original requirement ({signal['category']})" for signal in signals[:20]]))
    return "\n".join(lines), summary


def scope_boundary_doc(facts: dict[str, Any]) -> tuple[str, list[str]]:
    summary = [
        "Use augmentation to clarify the current task, not to expand it beyond the Original Requirement.",
        "Do not change unrelated public behavior while adding the requested capability.",
        "Do not hard-code benchmark examples; implement the general feature semantics.",
        "Do not shrink required cross-file consistency out of scope, but verify generated/build edits before touching them.",
    ]
    lines = topic_header("Scope Boundary", summary)
    lines.extend(
        [
            "## Must Implement",
            "",
            "- Keep changes tied to concrete behavior required by the Original Requirement.",
            "- Update consistency surfaces only when source inspection shows they are required for the task to work.",
            "- Treat required new modules/files, generated artifacts, registration, exports, and build metadata as in scope when the Original Requirement needs them, not as default edit targets.",
            "",
            "## Must Preserve",
            "",
            "- Preserve unrelated public behavior, compatibility paths, and existing tests unless the requirement explicitly changes them.",
            "- Preserve buildability, generated/build file executable bits, and existing diagnostics while changing only requirement-relevant behavior.",
            "",
            "## Should Inspect",
            "",
            "- Keep implementation changes tied to requirement behavior and repository evidence.",
            "- Treat broad rewrites as risky unless direct source inspection proves they are necessary.",
            "- Treat background/specification material as context unless it maps to required behavior for this task.",
            "- Avoid modifying optional tests as a substitute for source behavior.",
            "- Distinguish unrelated scope expansion from required completeness work.",
            "",
            "## Do Not Do",
            "",
            "- Do not expand the task into a complete implementation of every related spec section.",
            "- Do not over-correct by omitting required generated files, registration points, exports, metadata, or new source files.",
            "- Do not edit files only because they appear in repository perception or graph evidence.",
            "- Do not modify generated/build outputs merely to look complete; require a source-of-truth reason and preserve metadata.",
            "- Do not use benchmark-specific shortcuts, hidden-answer assumptions, or hard-coded examples.",
            "",
            "## Validation",
            "",
            "- Check that the final patch is smaller than the broadest possible interpretation of the requirement.",
            "- Run focused validation for changed code and avoid unrelated churn.",
            "",
            "## Low-confidence Notes",
            "",
            "- If a boundary is unclear, inspect nearby source and existing tests before deciding.",
            "",
            "## Evidence Index",
            "",
            "- Source: original requirement plus generated code facts.",
        ]
    )
    return "\n".join(lines), summary


def task_decomposition_doc(facts: dict[str, Any]) -> tuple[str, list[str]]:
    surfaces = facts.get("implementation_surfaces", [])
    summary = summary_from(
        [
            "Turn the Original Requirement into a small acceptance checklist before choosing files.",
            "Inspect candidate code surfaces only after clarifying expected behavior and scope.",
            "Choose the smallest complete coherent edit set that satisfies the task and preserves existing behavior.",
            *capability_high_priority(facts),
            *[f"Candidate investigation surface `{surface['name']}`: {surface.get('why', '')}" for surface in surfaces[:3]],
        ]
    )
    lines = topic_header("Task Decomposition", summary or ["Clarify behavior, inspect candidates, then choose a scoped implementation."])
    lines.extend(
        [
            "## Must Implement",
            "",
            "- Convert the Original Requirement into a short ordered checklist of required behavior deltas.",
            "- Implement one minimal complete coherent edit path that satisfies that checklist.",
            "- For each required delta, decide whether source-of-truth, generated output, registration/export, metadata, or validation updates are truly required.",
            "",
            "## Must Preserve",
            "",
            "- Preserve existing behavior not named by the checklist.",
            "- Preserve build/test registration semantics, generated/build file metadata, executable bits, and diagnostics unless source inspection proves an update is required.",
            "",
            "## Should Inspect",
            "",
        ]
    )
    capability_lines = capability_should_inspect_lines(facts)
    if capability_lines:
        lines.extend(capability_lines)
        lines.append("")
    for surface in surfaces:
        lines.append(f"- Claim: `{surface['name']}` is a candidate investigation surface.")
        lines.append(f"  Source: {surface.get('source', 'inferred')}")
        lines.append(f"  Confidence: {surface.get('confidence', 'medium')}")
        lines.append("  Action: Inspect this surface only if it helps satisfy a concrete acceptance criterion from the Original Requirement.")
        lines.append(f"  Risk: {surface.get('risk', '')}")
        lines.extend(evidence_block(surface))
    lines.extend(
        [
            "",
            "## Do Not Do",
            "",
            "- Do not treat the decomposition as permission to rewrite all mentioned layers.",
            "- Do not treat scope control as permission to omit required consistency layers.",
            "- Do not edit tests or generated-looking files before identifying the source-of-truth change.",
            "- Do not keep generated/build edits that cause build or pass-to-pass failures unless they are proven required and then fixed.",
            "",
            "## Validation",
            "",
            "- After each meaningful edit, prefer a narrow check before adding more scope.",
            "- Use validation failures to refine the checklist, not to broaden the task blindly.",
            "- If generated/build files are touched, include a build/import sanity check or explicitly inspect why one is unavailable.",
            "",
            "## Low-confidence Notes",
            "",
        ]
    )
    lines.extend(bullet_lines([item["question"] for item in facts.get("uncertainties", [])]))
    lines.extend(["", "## Evidence Index", ""])
    lines.extend(bullet_lines([str(surface.get("inspect", [])) for surface in surfaces]))
    return "\n".join(lines), summary


def codebase_context_doc(facts: dict[str, Any]) -> tuple[str, list[str]]:
    repo = facts.get("repo_map", {})
    repository_perception = facts.get("repository_perception", {})
    codegraph_layer = repository_perception.get("codegraph_layer", {}) if repository_perception else {}
    top_levels = repo.get("top_levels", {})
    summary = summary_from(
        [
            f"Indexed top-level areas: {', '.join(list(top_levels)[:8])}.",
            f"Generated-looking files found: {len(repo.get('generated_files', []))}.",
            f"Build/config entry files found: {len(repo.get('build_files', []))}.",
            f"Test roots found: {', '.join(repo.get('test_roots', [])[:6]) or '(none)'}.",
            f"Repository perception backend: {repository_perception.get('backend', 'unavailable')}.",
            f"CodeGraph available: {codegraph_layer.get('available', False)}.",
        ]
    )
    lines = topic_header("Codebase Context", summary)
    lines.extend(
        [
            "## Must Implement",
            "",
            "- No repository area is automatically required by context alone; implement only source-verified requirement deltas.",
            "",
            "## Must Preserve",
            "",
            "- Preserve repository conventions around generated files, build files, and tests.",
            "- Preserve executable bits and file metadata for generated/build files unless the required source-of-truth change demands otherwise.",
            "",
            "## Should Inspect",
            "",
            "### Top-level Areas",
        ]
    )
    for top, counts in list(top_levels.items())[:20]:
        lines.append(f"- `{top}`: {counts}")
    lines.extend(["", "### Generated-looking Files"])
    lines.extend(bullet_lines([f"`{item}`" for item in repo.get("generated_files", [])[:30]]))
    lines.extend(["", "### Build Files"])
    lines.extend(bullet_lines([f"`{item}`" for item in repo.get("build_files", [])[:30]]))
    lines.extend(["", "### Test Roots"])
    lines.extend(bullet_lines([f"`{item}`" for item in repo.get("test_roots", [])[:30]]))
    if repository_perception:
        lines.extend(["", "### Repository Perception"])
        lines.append(f"- Backend: {repository_perception.get('backend', '')}")
        lines.append(f"- CodeGraph available: {codegraph_layer.get('available', False)}")
        lines.append(f"- CodeGraph status: {codegraph_layer.get('status', '')}")
        for item in repository_perception.get("risks", [])[:8]:
            lines.append(f"- Claim: {item.get('claim', '')}")
            lines.append(f"  Source: {item.get('source', '')}")
            lines.append(f"  Confidence: {item.get('confidence', '')}")
            lines.append(f"  Action: {item.get('action', '')}")
    lines.extend(
        [
            "",
            "## Do Not Do",
            "",
            "- Do not edit generated-looking files unless the repository convention requires checked-in regeneration.",
            "- Do not modify build files simply because they are listed; inspect whether registration is actually needed.",
            "- Do not accept file-mode churn in generated/build files as part of a behavior patch.",
            "",
            "## Validation",
            "",
            "- Prefer validation roots that correspond to the touched module or public entrypoint.",
            "",
            "## Low-confidence Notes",
            "",
            "- Generated-file detection is heuristic; verify source-of-truth relationships before editing.",
            "",
            "## Evidence Index",
            "",
            "- Source: current workspace file index.",
        ]
    )
    return "\n".join(lines), summary


def implementation_surfaces_doc(facts: dict[str, Any]) -> tuple[str, list[str]]:
    surfaces = facts.get("implementation_surfaces", [])
    summary = summary_from(
        [
            f"`{surface['name']}` is an inspection candidate: {', '.join(surface.get('inspect', [])[:3]) or 'repository or requirement evidence'}."
            for surface in surfaces[:6]
        ]
        + capability_high_priority(facts)
    )
    lines = topic_header("Likely Implementation Surfaces", summary or ["No high-confidence implementation surface was inferred."])
    lines.extend(
        [
            "## Must Implement",
            "",
            "- Edit a listed surface only after direct source inspection ties it to a required behavior delta.",
            "- Prefer the smallest complete source-of-truth surface set that can satisfy the task.",
            "- If the requirement names an absent module/API/file, create or register it when source inspection confirms it is required.",
            "",
            "## Must Preserve",
            "",
            "- Preserve unrelated files and sibling modules unless validation proves they must change.",
            "- Preserve buildability, generated/build file modes, and pass-to-pass behavior while selecting the final edit set.",
            "",
            "## Should Inspect",
            "",
        ]
    )
    capability_lines = capability_should_inspect_lines(facts)
    if capability_lines:
        lines.extend(capability_lines)
        lines.append("")
    for surface in surfaces:
        lines.append(f"- Claim: `{surface['name']}` may contain relevant implementation context.")
        lines.append(f"  Source: {surface.get('source', 'inferred')}")
        lines.append(f"  Confidence: {surface.get('confidence', 'medium')}")
        lines.append(f"  Action: Inspect these files as candidates, then edit only if source behavior proves relevance: {', '.join(f'`{item}`' for item in surface.get('inspect', [])[:8]) or '(none)'}")
        lines.append(f"  Risk: {surface.get('risk', '')}")
        lines.extend(evidence_block(surface))
    lines.extend(
        [
            "",
            "## Do Not Do",
            "",
            "- Do not edit every listed surface.",
            "- Do not force new-feature work into unrelated existing surfaces just because they rank highly.",
            "- Do not treat CodeGraph caller/callee/impact output as proof that all related files require changes.",
            "- Do not touch generated/build surfaces without a source-of-truth reason and a build/import validation plan.",
            "",
            "## Validation",
            "",
            "- For each edited surface, identify at least one relevant validation path or explain why source inspection is the only feasible check.",
            "- If an edited surface is generated/build/packaging-related, include a build/import sanity check and verify file modes are preserved.",
            "",
            "## Low-confidence Notes",
            "",
            "- These are inspection surfaces, not mandatory edit lists.",
            "",
            "## Evidence Index",
            "",
        ]
    )
    lines.extend(bullet_lines([str(surface.get("evidence", [])) for surface in surfaces]))
    return "\n".join(lines), summary


def existing_patterns_doc(facts: dict[str, Any]) -> tuple[str, list[str]]:
    patterns = facts.get("existing_patterns", [])
    summary = summary_from([f"{item.get('pattern', '')}: {item.get('action', '')}" for item in patterns[:6]])
    lines = topic_header("Relevant Existing Patterns", summary or ["No strong existing pattern was extracted."])
    lines.extend(
        [
            "## Must Implement",
            "",
            "- Reuse local patterns only when they directly support a required behavior delta.",
            "",
            "## Must Preserve",
            "",
            "- Preserve established style, compatibility behavior, error handling, and diagnostics around touched code.",
            "",
            "## Should Inspect",
            "",
        ]
    )
    for item in patterns:
        lines.append(f"- Claim: {item.get('pattern', '')}")
        lines.append(f"  Source: {item.get('source', 'code_evidence')}")
        lines.append(f"  Confidence: {item.get('confidence', 'medium')}")
        lines.append(f"  Action: {item.get('action', '')}")
        lines.extend(evidence_block(item))
    lines.extend(
        [
            "",
            "## Do Not Do",
            "",
            "- Do not copy a similar implementation without checking whether semantics, registration, and validation match this task.",
            "- Do not perform style-only refactors while following existing patterns.",
            "",
            "## Validation",
            "",
            "- Validate that reused patterns preserve existing behavior and satisfy the new requirement path.",
            "",
            "## Low-confidence Notes",
            "",
            "- Similar files can be misleading; copy local patterns only after reading real source.",
            "",
            "## Evidence Index",
            "",
            "- Source: ranked workspace files and symbols.",
        ]
    )
    return "\n".join(lines), summary


def code_details_doc(facts: dict[str, Any]) -> tuple[str, list[str]]:
    details = facts.get("code_details", [])
    summary = summary_from([f"{item.get('priority', 'useful')} inspection candidate: `{item.get('path', '')}`" for item in details[:8]])
    lines = topic_header("Code Details To Inspect", summary or ["No concrete code details were extracted."])
    groups = {"must inspect": [], "useful": [], "optional": []}
    for item in details:
        groups.setdefault(item.get("priority", "useful"), []).append(item)
    lines.extend(
        [
            "## Must Implement",
            "",
            "- Use code details to verify source-of-truth behavior before editing.",
            "- Edit only details that are necessary for the smallest complete coherent patch.",
            "- Add missing source or generated details when they are required by the Original Requirement but absent from ranked code details.",
            "",
            "## Must Preserve",
            "",
            "- Preserve nearby behavior, error handling, diagnostics, exports, and registration paths unless the task requires them to change.",
            "- Preserve generated/build file permissions and metadata if code details point at those files.",
            "",
            "## Should Inspect",
            "",
        ]
    )
    for group_name in ("must inspect", "useful", "optional"):
        lines.extend([f"### {group_name.title()}", ""])
        group = groups.get(group_name, [])
        if not group:
            lines.append("- (none)")
            continue
        for item in group:
            symbols = ", ".join(symbol.get("name", "") for symbol in item.get("symbols", [])[:6])
            lines.append(f"- Claim: `{item.get('path', '')}` may help explain or implement the task.")
            lines.append("  Source: code_evidence")
            lines.append(f"  Confidence: {item.get('confidence', 'medium')}")
            lines.append(f"  Action: Read this as evidence, then decide whether it is necessary for the minimal task patch. {item.get('why', '')}")
            if symbols:
                lines.append(f"  Evidence: symbols {symbols}")
            else:
                lines.extend(evidence_block(item))
    lines.extend(
        [
            "",
            "## Do Not Do",
            "",
            "- Do not edit a file solely because it is listed here; read it and confirm relevance.",
            "- Do not assume this list is complete when the task introduces new files or generated artifacts.",
            "- Do not broaden edits from one symbol to a whole module without a concrete failing behavior or source requirement.",
            "- Do not let generated/build candidate files become edit targets unless source-of-truth inspection proves they are required.",
            "",
            "## Validation",
            "",
            "- Tie each edited code detail to a validation point, such as a nearby test, parser/runtime smoke check, or build check.",
            "- If generated/build details are edited, check buildability and preserve executable bits/file metadata.",
            "",
            "## Low-confidence Notes",
            "",
            "- Code details are mined heuristically and may be incomplete.",
            "",
            "## Evidence Index",
            "",
        ]
    )
    lines.extend(bullet_lines([f"`{item.get('path', '')}`" for item in details]))
    return "\n".join(lines), summary


def validation_plan_doc(facts: dict[str, Any]) -> tuple[str, list[str]]:
    hints = facts.get("validation_hints", [])
    summary = summary_from([f"{item.get('kind', 'check')}: `{item.get('path', '')}`" for item in hints[:8]] + capability_high_priority(facts))
    if not summary:
        summary = ["Run the narrowest meaningful build/test/smoke checks available in the target repo."]
    lines = topic_header("Validation Plan", summary)
    lines.extend(
        [
            "## Must Implement",
            "",
            "- Select validation that directly exercises the behavior required by the Original Requirement.",
            "- Keep validation proportional to the actual files touched, but include build/import/smoke checks for required new or generated surfaces.",
            "- If generated files, parser/schema artifacts, build scripts, or registration metadata are touched, validate that the project still builds or imports the touched feature and that file modes/metadata remain valid.",
            "",
            "## Must Preserve",
            "",
            "- Include at least one pass-to-pass-sensitive check when a nearby existing behavior path is identifiable.",
            "- Preserve buildability and existing diagnostics; do not trade broad pass-to-pass regressions for narrow feature progress.",
            "",
            "## Should Inspect",
            "",
            "- Validate the narrow behavior implied by the Original Requirement first, then broaden checks only when the edit scope requires it.",
        ]
    )
    capability_lines = capability_should_inspect_lines(facts)
    if capability_lines:
        lines.extend(capability_lines)
    for item in hints:
        lines.append(f"- Claim: `{item.get('path', '')}` may help validate the change.")
        lines.append(f"  Source: {item.get('source', 'code_evidence')}")
        lines.append(f"  Confidence: {item.get('confidence', 'medium')}")
        lines.append(f"  Action: {item.get('action', '')}")
    lines.extend(
        [
            "",
            "## Do Not Do",
            "",
            "- Do not replace implementation work with test-only changes.",
            "- Do not run broad expensive suites as a substitute for understanding the touched area when narrow checks exist.",
            "- Do not skip a build/import check when the patch changes generated, parser, build, or registration files.",
            "- Do not ignore a build failure caused by generated/build file mode or metadata churn.",
            "",
            "## Validation",
            "",
            "- Run or identify the narrowest available check for the modified surface.",
            "- If a command is not obvious, inspect build/config/test roots before choosing one.",
            "- For CPython-style generated/build chains, verify executable scripts such as `configure` remain executable when touched.",
            "",
            "## Low-confidence Notes",
            "",
            "- Validation hints are not guaranteed to be cheap or complete; adapt to repository tooling.",
            "",
            "## Evidence Index",
            "",
            "- Source: workspace tests and build/config files.",
        ]
    )
    return "\n".join(lines), summary


def anti_patterns_doc(facts: dict[str, Any]) -> tuple[str, list[str]]:
    items = facts.get("anti_patterns", [])
    summary = summary_from(
        [
            "Do not turn requirement augmentation into a broad rewrite plan.",
            "Do not treat candidate code surfaces as mandatory edits.",
            "Do not over-narrow feature work by dropping required new files, generated artifacts, or registration.",
            "Do not over-broaden feature work by changing generated/build artifacts without a source-verified reason.",
            *[item.get("claim", "") for item in items[:6]],
        ]
    )
    lines = topic_header("Anti-patterns and Failure Modes", summary)
    lines.extend(
        [
            "## Must Implement",
            "",
            "- Use anti-patterns as hard guardrails while planning and editing.",
            "- Narrow the task whenever the plan starts drifting beyond the Original Requirement.",
            "- Complete every required consistency surface once source inspection proves it is necessary.",
            "",
            "## Must Preserve",
            "",
            "- Preserve buildability, existing behavior, diagnostics, generated/build file modes, and patch locality.",
            "",
            "## Should Inspect",
            "",
            "- Claim: Requirement augmentation should narrow ambiguity and risk, not create extra implementation scope.",
            "  Source: inferred",
            "  Confidence: high",
            "  Action: If a plan becomes too broad, return to the Original Requirement and repository evidence.",
            "  Risk: Broad patches are more likely to fail build or regress pass-to-pass behavior.",
            "",
            "- Claim: Code perception is evidence, not an edit mandate.",
            "  Source: code_evidence",
            "  Confidence: high",
            "  Action: Inspect listed files before deciding whether they belong in the patch.",
            "  Risk: Editing every listed surface can create unrelated failures.",
            "",
            "- Claim: Over-narrowing can be as harmful as broad rewriting for feature backports.",
            "  Source: inferred",
            "  Confidence: high",
            "  Action: If the task requires a new module, generated artifact, registration, export, or build update, include it after source verification.",
            "  Risk: Omitting required consistency surfaces can cause build failures, import errors, or zero F2P/P2P.",
            "",
            "- Claim: Over-broad generated/build synchronization can be as harmful as missing synchronization.",
            "  Source: inferred",
            "  Confidence: high",
            "  Action: Touch generated/build files only when the source-of-truth change requires it; preserve executable bits and validate build/import behavior.",
            "  Risk: Metadata churn or stale generated output can cause build failures or P2P regressions.",
            "",
        ]
    )
    for item in items:
        lines.append(f"- Claim: {item.get('claim', '')}")
        lines.append(f"  Source: {item.get('source', 'inferred')}")
        lines.append(f"  Confidence: {item.get('confidence', 'medium')}")
        lines.append(f"  Action: {item.get('action', '')}")
        lines.append("  Risk: Ignoring this can reduce F2P/P2P or cause build failures.")
    lines.extend(
        [
            "",
            "## Do Not Do",
            "",
            "- Do not turn the augmented requirement into a broad rewrite plan.",
            "- Do not implement the whole PEP/KIP/FLIP/issue background when the current task requires a narrower behavior delta.",
            "- Do not edit every candidate surface, generated artifact, build file, or test root.",
            "- Do not omit required generated artifacts, source-of-truth consumers, new source files, exports, registration, or metadata just to keep the patch tiny.",
            "- Do not keep generated/build edits that are unrelated, stale, or metadata-damaging.",
            "- Do not add benchmark-specific shortcuts or hidden-test guesses.",
            "",
            "## Validation",
            "",
            "- Before finalizing, check that every changed file has a task reason and every task reason maps back to the Original Requirement.",
            "- If build/P2P regresses after generated/build edits, inspect file modes, metadata, and whether those edits were actually required.",
            "",
            "## Low-confidence Notes",
            "",
            "- These are generic risk checks grounded in inferred surfaces; verify which ones apply.",
            "",
            "## Evidence Index",
            "",
            "- Source: implementation surfaces and repository context.",
        ]
    )
    return "\n".join(lines), summary


def open_questions_doc(facts: dict[str, Any]) -> tuple[str, list[str]]:
    items = facts.get("uncertainties", [])
    summary = summary_from([item.get("question", "") for item in items[:8]])
    if not summary:
        summary = ["No major low-confidence questions were generated; still verify key surfaces with source inspection."]
    lines = topic_header("Assumptions and Open Questions", summary)
    lines.extend(
        [
            "## Must Implement",
            "",
            "- Resolve only questions that block choosing a minimal implementation path.",
            "",
            "## Must Preserve",
            "",
            "- Preserve original task fidelity when uncertainty exists; do not make broad assumptions.",
            "",
            "## Should Inspect",
            "",
        ]
    )
    for item in items:
        lines.append(f"- Claim: {item.get('question', '')}")
        lines.append(f"  Source: {item.get('source', 'uncertain')}")
        lines.append(f"  Confidence: {item.get('confidence', 'low')}")
        lines.append("  Action: Resolve this with normal repository inspection before relying on the related augmentation item.")
        lines.extend(evidence_block(item))
    lines.extend(
        [
            "",
            "## Do Not Do",
            "",
            "- Do not treat open questions as new requirements.",
            "- Do not block implementation on uncertainty that can be handled by a small source read or narrow validation check.",
            "",
            "## Validation",
            "",
            "- Use source inspection and narrow checks to resolve uncertainty; record assumptions in reasoning before editing.",
            "",
            "## Low-confidence Notes",
            "",
            "- This section is intentionally cautious; uncertain items are prompts for verification, not edit instructions.",
            "",
            "## Evidence Index",
            "",
            "- Source: low-confidence surfaces and missing matches.",
        ]
    )
    return "\n".join(lines), summary


def topic_header(title: str, summary: list[str]) -> list[str]:
    lines = [f"# {title}", "", TASK_PACKAGE_NOTE, "", "## High Priority", ""]
    lines.extend(bullet_lines(summary, empty="- No high-priority summary generated."))
    lines.append("")
    return lines


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        result = generate(args)
        print(f"Wrote Augmented Requirement: {result['out']}")
        print({"routes": len(result["routes"]), "generation_mode": result.get("generation_mode")})
        return 0
    except ScriptError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
