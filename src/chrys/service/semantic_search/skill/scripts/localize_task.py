#!/usr/bin/env python3
"""Run Chrys-native SemLoc search and write an evidence-backed localization.

The primary path is an LLM-driven DFS/BFS search over five repository tools and
a normalized CodeGraph/source adapter.  A deterministic ranker remains
available when no localization model is configured.  This script never edits
source; final verification and implementation remain Chrys responsibilities.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from _common import (
    FORMAT_CODEGRAPH,
    FORMAT_FACTS,
    FORMAT_INDEX,
    FORMAT_LOCALIZATION,
    FORMAT_LOCALIZATION_GRAPH,
    ScriptError,
    append_trace,
    ensure_allowed_path,
    load_json,
    now_iso,
    path_tokens,
    read_text,
    reject_benchmark_answer_path,
    resolve_path,
    sha1_path,
    stable_unique,
    tokenize,
    write_json,
)
from _localization_agent import (
    TraceWriter,
    normalize_locations,
)
from _localization_graph import LocalizationGraph

SOURCE_SUFFIXES = ("py", "java", "scala", "rs", "c", "h", "cc", "cpp", "hpp")
FILE_RE = re.compile(r"(?P<path>[A-Za-z0-9_./-]+\.(?:" + "|".join(SOURCE_SUFFIXES) + r"))", flags=re.IGNORECASE)
SYMBOL_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./-]+\.(?:" + "|".join(SOURCE_SUFFIXES) + r"))\s*:\s*"
    r"(?P<symbol>[A-Za-z_][A-Za-z0-9_.]*)",
    flags=re.IGNORECASE,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--requirement", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--facts")
    parser.add_argument("--codegraph-perception")
    parser.add_argument("--top-locations", type=int, default=12)
    parser.add_argument(
        "--mode",
        choices=("auto", "llm", "fallback"),
        default=os.environ.get("SEMANTIC_SEARCH_LOCALIZATION_MODE", "auto"),
        help="auto uses the LLM agent when configured, llm requires it, and fallback uses deterministic ranking.",
    )
    parser.add_argument(
        "--model-profile",
        default=os.environ.get("SEMANTIC_SEARCH_LOCALIZATION_MODEL_PROFILE")
        or os.environ.get("CHRYS_MODEL_PROFILE", ""),
    )
    parser.add_argument("--trace", help="Output JSONL search trace. Defaults to artifact-dir/localization-trace.jsonl.")
    parser.add_argument("--graph-out", help="Output normalized localization graph JSON.")
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=float(os.environ.get("SEMANTIC_SEARCH_LOCALIZATION_TIMEOUT", "120")),
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--max-tool-results", type=int, default=20)
    parser.add_argument(
        "--locations",
        help=(
            "JSON array of raw locations produced by the in-process localization agent. "
            "Present means the LLM stage already ran; this script only normalizes and renders."
        ),
    )
    return parser.parse_args(argv)


def load_inputs(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, Path, str, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    repo = resolve_path(args.repo)
    if not repo.is_dir():
        raise ScriptError(f"repo path does not exist: {repo}")
    out = resolve_path(args.out)
    artifact_dir = resolve_path(args.artifact_dir or out.parent)
    requirement = ensure_allowed_path(
        args.requirement,
        allowed_roots=[artifact_dir],
        allowed_files=[resolve_path(args.requirement)],
        purpose="requirement",
    )
    reject_benchmark_answer_path(requirement, purpose="requirement")
    index_path = ensure_allowed_path(args.index, allowed_roots=[repo, artifact_dir], purpose="index")
    out = ensure_allowed_path(out, allowed_roots=[repo, artifact_dir, out.parent], purpose="output")
    markdown = resolve_path(args.markdown or out.with_suffix(".md"))
    markdown = ensure_allowed_path(
        markdown, allowed_roots=[repo, artifact_dir, markdown.parent], purpose="markdown-output"
    )
    index = load_json(index_path)
    if index.get("format") != FORMAT_INDEX:
        raise ScriptError(f"unsupported index format: {index.get('format')}")
    facts = None
    if args.facts:
        facts_path = ensure_allowed_path(args.facts, allowed_roots=[repo, artifact_dir], purpose="facts")
        facts = load_json(facts_path)
        if facts.get("format") != FORMAT_FACTS:
            raise ScriptError(f"unsupported facts format: {facts.get('format')}")
    codegraph = None
    if args.codegraph_perception:
        codegraph_path = ensure_allowed_path(
            args.codegraph_perception, allowed_roots=[repo, artifact_dir], purpose="codegraph-perception"
        )
        codegraph = load_json(codegraph_path)
        if codegraph.get("format") != FORMAT_CODEGRAPH:
            raise ScriptError(f"unsupported codegraph perception format: {codegraph.get('format')}")
    return repo, requirement, index_path, out, markdown, read_text(requirement), index, facts, codegraph


def requirement_terms(text: str) -> list[str]:
    literals = re.findall(r"`([^`]{2,100})`", text)
    paths = re.findall(
        r"[A-Za-z0-9_./-]+\.(?:" + "|".join(SOURCE_SUFFIXES) + r"|toml|yaml|yml|xml)", text, flags=re.IGNORECASE
    )
    return stable_unique([*paths, *literals, *tokenize(" ".join(literals) + " " + text)])[:180]


def codegraph_mentions(codegraph: dict[str, Any] | None) -> tuple[set[str], list[dict[str, Any]]]:
    paths: set[str] = set()
    evidence: list[dict[str, Any]] = []
    if not codegraph:
        return paths, evidence
    for item in codegraph.get("repository_queries", []):
        best = item.get("best", {})
        output = str(best.get("stdout", "") or best.get("stderr", ""))
        paths.update(match.group("path") for match in FILE_RE.finditer(output))
        if output:
            evidence.append({"kind": "query", "query": item.get("query", ""), "text": output[:1200]})
    for item in codegraph.get("symbol_relationships", []):
        for key in ("node", "callers", "callees", "impact"):
            result = item.get(key, {})
            output = str(result.get("stdout", "") or result.get("stderr", ""))
            paths.update(match.group("path") for match in FILE_RE.finditer(output))
            for match in SYMBOL_RE.finditer(output):
                paths.add(match.group("path"))
            if output:
                evidence.append({"kind": key, "symbol": item.get("symbol", ""), "text": output[:1000]})
    return paths, evidence[:24]


def score_file(record: dict[str, Any], terms: set[str], graph_paths: set[str]) -> int:
    path = str(record.get("path", "")).lower()
    record_terms = {str(item).lower() for item in record.get("terms", [])}
    symbol_names = {str(item.get("name", "")).lower() for item in record.get("symbols", [])}
    preview = str(record.get("preview", "")).lower()
    score = 0
    if path in {item.lower() for item in graph_paths}:
        score += 30
    for term in terms:
        lowered = term.lower()
        variants = stable_unique([lowered, *path_tokens(lowered)])
        if any(path == variant or path.endswith("/" + variant) for variant in variants):
            score += 18
        elif any(variant in path for variant in variants):
            score += 7
        if any(variant in record_terms for variant in variants):
            score += 4
        if any(variant in symbol_names for variant in variants):
            score += 10
        if any(variant in preview for variant in variants):
            score += 2
    if record.get("kind") == "source":
        score += 2
    if record.get("is_test"):
        score -= 1
    return score


def rank_files(index: dict[str, Any], terms: list[str], graph_paths: set[str], limit: int) -> list[dict[str, Any]]:
    scored = [
        (score_file(record, set(terms), graph_paths), str(record.get("path", "")), record)
        for record in index.get("files", [])
    ]
    scored = [(score, path, record) for score, path, record in scored if score > 0]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [{**record, "score": score} for score, _path, record in scored[:limit]]


def fallback_locations(
    graph: LocalizationGraph,
    ranked_files: list[dict[str, Any]],
    terms: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Create SemLoc-compatible locations when the LLM agent is unavailable."""
    scored: list[tuple[int, str, int, dict[str, Any]]] = []
    lowered_terms = {term.lower() for term in terms}
    for record in ranked_files:
        relative = str(record.get("path", ""))
        units = graph.units_by_file.get(relative, [])
        reason = "Requirement and repository evidence matched this location."
        if relative in graph.codegraph_paths:
            reason = "CodeGraph evidence and requirement signals matched this location."
        if not units:
            raw = {
                "file_path": relative,
                "start_line": 1,
                "end_line": 1,
                "reason": reason,
                "confidence": "medium",
            }
            scored.append((int(record.get("score", 0)), relative, 1, raw))
            continue
        for unit in units:
            haystack = " ".join(
                [
                    str(unit.get("qualified_name", "")),
                    str(unit.get("signature", "")),
                    str(unit.get("content", ""))[:2500],
                ]
            ).lower()
            unit_score = int(record.get("score", 0))
            unit_score += sum(
                8 for term in lowered_terms if term and term in str(unit.get("qualified_name", "")).lower()
            )
            unit_score += sum(2 for term in lowered_terms if term and term in haystack)
            unit_score += min(len(graph.neighbors(unit)), 8)
            raw = {
                "file_path": relative,
                "class_name": unit.get("class_name", ""),
                "function_name": "" if unit.get("kind") == "class" else unit.get("name", ""),
                "start_line": unit.get("start_line"),
                "end_line": unit.get("end_line"),
                "reason": reason,
                "confidence": "high" if unit_score >= 30 else "medium",
            }
            scored.append((unit_score, relative, int(unit.get("start_line") or 0), raw))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    raw_locations = [item[3] for item in scored[:limit]]
    return normalize_locations(raw_locations, graph, source="deterministic-fallback")


def _load_supplied_locations(path: str | None) -> list[dict[str, Any]]:
    """Read the in-process agent's raw locations, treating absence as none."""
    if not path:
        return []
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise ScriptError(f"could not read supplied localization results: {err}") from err
    if not isinstance(payload, list):
        raise ScriptError("supplied localization results must be a JSON array")
    return [item for item in payload if isinstance(item, dict)]


def localize(args: argparse.Namespace) -> dict[str, Any]:
    repo, requirement, index_path, out, markdown, requirement_text, index, facts, codegraph = load_inputs(args)
    terms = requirement_terms(requirement_text)
    graph_paths, graph_evidence = codegraph_mentions(codegraph)
    graph = LocalizationGraph(repo, index, codegraph)
    artifact_dir = resolve_path(args.artifact_dir or out.parent)
    trace_path = resolve_path(args.trace or artifact_dir / "localization-trace.jsonl")
    trace_path = ensure_allowed_path(
        trace_path,
        allowed_roots=[repo, resolve_path(args.artifact_dir or out.parent), trace_path.parent],
        purpose="localization-trace",
    )
    graph_out = resolve_path(args.graph_out or artifact_dir / "localization-graph.json")
    graph_out = ensure_allowed_path(
        graph_out,
        allowed_roots=[repo, artifact_dir, graph_out.parent],
        purpose="localization-graph-output",
    )
    trace = TraceWriter(trace_path)
    locations: list[dict[str, Any]] = []
    generation_mode = "fallback"
    model = ""
    tool_call_count = 0
    iteration_count = 0
    agent_error = ""
    agent_requested = args.mode != "fallback"
    # The LLM search runs inside the Chrys process, where the model client,
    # the model lock, and usage accounting already live. This script receives
    # its output and owns only the deterministic half: normalization, ranking,
    # the graph export, and the rendered report.
    supplied = _load_supplied_locations(args.locations)
    if supplied:
        locations = normalize_locations(supplied, graph, source="llm-search")[: max(args.top_locations, 1)]
        generation_mode = "llm-agent"
        model = str(args.model_profile or "")
        trace.write("agent-locations-supplied", location_count=len(locations))
        if not locations:
            agent_error = "in-process localization agent returned no valid repository locations"
            trace.write("agent-fallback", reason=agent_error)
    elif args.mode == "llm":
        raise ScriptError("--mode llm requires locations from the in-process localization agent")
    else:
        reason = "no in-process localization result" if agent_requested else "fallback mode requested"
        trace.write("agent-skipped", reason=reason)

    ranked: list[dict[str, Any]] = []
    if not locations:
        ranked = rank_files(index, terms, graph_paths, max(args.top_locations, 1))
        locations = fallback_locations(graph, ranked, terms, max(args.top_locations, 1))
        trace.write("fallback-complete", location_count=len(locations), ranked_file_count=len(ranked))
    else:
        ranked_paths = stable_unique(location.get("file", "") for location in locations)
        ranked = [{**graph.files[path], "score": graph.requirement_file_score(path, terms)} for path in ranked_paths]
    write_json(
        graph_out,
        {
            "format": FORMAT_LOCALIZATION_GRAPH,
            "schema_version": "semantic-search-localization-graph.v1",
            "created_at": now_iso(),
            "inputs": {"repo": str(repo), "index": str(index_path)},
            **graph.export(),
        },
    )
    related_tests = [
        record.get("path", "")
        for record in index.get("files", [])
        if record.get("is_test")
        and any(token in str(record.get("path", "")).lower() for token in path_tokens(" ".join(terms)))
    ][:12]
    related_files = [
        record.get("path", "")
        for record in index.get("files", [])
        if record.get("kind") in {"config", "build", "docs", "generated"}
        and any(token in str(record.get("path", "")).lower() for token in path_tokens(" ".join(terms)))
    ][:12]
    payload = {
        "format": FORMAT_LOCALIZATION,
        "schema_version": "semantic-search-code-localization.v2",
        "created_at": now_iso(),
        "inputs": {
            "repo": str(repo),
            "requirement": str(requirement),
            "requirement_sha1": sha1_path(requirement),
            "index": str(index_path),
            "codegraph_available": bool((codegraph or {}).get("available")),
            "trace": str(trace_path),
            "localization_graph": str(graph_out),
        },
        "entry_points": [{"term": term, "source": "original_requirement"} for term in terms[:40]],
        "locations": locations[: max(args.top_locations, 1)],
        "related_tests": stable_unique(related_tests),
        "related_files": stable_unique(related_files),
        "graph_evidence": graph_evidence,
        "unresolved_questions": []
        if locations
        else ["No repository locations matched the requirement; use normal Chrys search tools."],
        "summary": {
            "location_count": len(locations[: max(args.top_locations, 1)]),
            "file_count": len(ranked),
            "codegraph_available": bool((codegraph or {}).get("available")),
            "facts_available": bool(facts),
            "generation_mode": generation_mode,
            "agent_requested": agent_requested,
            "agent_used": generation_mode == "llm-agent",
            "model": model,
            "tool_call_count": tool_call_count,
            "iteration_count": iteration_count,
            "agent_error": agent_error,
            "trace": str(trace_path),
            "localization_graph": str(graph_out),
            "graph": graph.graph_summary(),
        },
    }
    write_json(out, payload)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    append_trace("localize-task", {"out": str(out), **payload["summary"]})
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Code Localization",
        "",
        (
            "This report ranks inspection candidates for Chrys. It is not an automatic edit list; "
            "verify every location against source."
        ),
        "",
        "## Summary",
        "",
        f"- Files: {payload.get('summary', {}).get('file_count', 0)}",
        f"- Locations: {payload.get('summary', {}).get('location_count', 0)}",
        f"- CodeGraph available: {payload.get('summary', {}).get('codegraph_available', False)}",
        f"- Generation mode: {payload.get('summary', {}).get('generation_mode', '')}",
        f"- Tool calls: {payload.get('summary', {}).get('tool_call_count', 0)}",
        f"- Trace: `{Path(str(payload.get('summary', {}).get('trace', ''))).name}`",
        "",
        "## Ranked Locations",
        "",
    ]
    for rank, location in enumerate(payload.get("locations", []), start=1):
        label = location.get("file", "")
        if location.get("symbol"):
            label += f":{location['symbol']}"
        lines.extend(
            [
                f"### {rank}. `{label}`",
                f"- Role: {location.get('role', '')}",
                f"- Lines: {location.get('start_line')} - {location.get('end_line')}",
                f"- Confidence: {location.get('confidence', '')}",
                f"- Reason: {location.get('reason', '')}",
                "- Source verification required: true",
                "",
            ]
        )
    lines.extend(["## Related Tests, Config, Build, And Docs", ""])
    lines.extend(f"- Test: `{path}`" for path in payload.get("related_tests", []))
    lines.extend(f"- Related: `{path}`" for path in payload.get("related_files", []))
    if not payload.get("related_tests") and not payload.get("related_files"):
        lines.append("- None identified; use normal Chrys search.")
    lines.extend(["", "## Unresolved Questions", ""])
    lines.extend(f"- {question}" for question in payload.get("unresolved_questions", []))
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        payload = localize(args)
        print(f"Wrote code localization: {resolve_path(args.out)}")
        print(payload["summary"])
        return 0
    except ScriptError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
