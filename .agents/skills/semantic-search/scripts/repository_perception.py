#!/usr/bin/env python3
"""Merge builtin, CodeGraph, and index evidence into a repository perception document."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from _common import (
    FORMAT_CODEGRAPH,
    FORMAT_GLOBAL,
    FORMAT_INDEX,
    FORMAT_REPOSITORY,
    ScriptError,
    append_trace,
    bullet_lines,
    ensure_allowed_path,
    load_json,
    markdown_escape_line,
    now_iso,
    resolve_path,
    sha1_path,
    write_json,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Seeded workspace repo root.")
    parser.add_argument("--index", required=True, help="Light semantic-search index from build_index.py.")
    parser.add_argument("--out", required=True, help="Output repository-perception.json path.")
    parser.add_argument("--markdown", help="Output repository-perception.md path. Defaults to out with .md suffix.")
    parser.add_argument("--artifact-dir", help="Semantic-search artifact directory. Defaults to output parent.")
    parser.add_argument("--global-perception", help="Optional global-perception.json.")
    parser.add_argument("--codegraph-perception", help="Optional codegraph-perception.json.")
    return parser.parse_args(argv)


def load_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    repo = resolve_path(args.repo)
    if not repo.is_dir():
        raise ScriptError(f"repo path does not exist: {repo}")
    out = resolve_path(args.out)
    artifact_dir = resolve_path(args.artifact_dir or out.parent)
    index_path = ensure_allowed_path(args.index, allowed_roots=[repo, artifact_dir], purpose="index")
    out = ensure_allowed_path(out, allowed_roots=[repo, artifact_dir, out.parent], purpose="output")
    markdown = resolve_path(args.markdown or out.with_suffix(".md"))
    markdown = ensure_allowed_path(markdown, allowed_roots=[repo, artifact_dir, markdown.parent], purpose="markdown-output")
    index = load_json(index_path)
    if index.get("format") != FORMAT_INDEX:
        raise ScriptError(f"unsupported index format: {index.get('format')}")
    global_perception = None
    if args.global_perception:
        global_path = ensure_allowed_path(args.global_perception, allowed_roots=[repo, artifact_dir], purpose="global-perception")
        global_perception = load_json(global_path)
        if global_perception.get("format") != FORMAT_GLOBAL:
            raise ScriptError(f"unsupported global perception format: {global_perception.get('format')}")
    codegraph_perception = None
    if args.codegraph_perception:
        codegraph_path = ensure_allowed_path(args.codegraph_perception, allowed_roots=[repo, artifact_dir], purpose="codegraph-perception")
        codegraph_perception = load_json(codegraph_path)
        if codegraph_perception.get("format") != FORMAT_CODEGRAPH:
            raise ScriptError(f"unsupported CodeGraph perception format: {codegraph_perception.get('format')}")
    return repo, out, markdown, index_path, index, global_perception, codegraph_perception


def build_repository_perception(args: argparse.Namespace) -> dict[str, Any]:
    repo, out, markdown, index_path, index, global_perception, codegraph_perception = load_inputs(args)
    files = index.get("files", [])
    repo_map = build_repo_map(files, index)
    global_summary = compact_global(global_perception)
    codegraph_summary = compact_codegraph(codegraph_perception)
    risks = repository_risks(repo_map, global_summary, codegraph_summary)
    payload = {
        "format": FORMAT_REPOSITORY,
        "created_at": now_iso(),
        "backend": "builtin+codegraph" if codegraph_summary.get("available") else "builtin",
        "inputs": {
            "repo": str(repo),
            "index": str(index_path),
            "index_sha1": sha1_path(index_path),
            "global_perception_available": bool(global_perception),
            "codegraph_perception_available": bool(codegraph_perception and codegraph_perception.get("available")),
        },
        "overview": repo_map["overview"],
        "module_map": repo_map["module_map"],
        "repository_docs": repo_map["docs"],
        "global_layer": global_summary,
        "codegraph_layer": codegraph_summary,
        "source_of_truth": source_of_truth(global_summary, repo_map),
        "validation_surface": validation_surface(global_summary, repo_map),
        "risks": risks,
    }
    write_json(out, payload)
    markdown.write_text(render_markdown(payload), encoding="utf-8")
    append_trace(
        "repository-perception",
        {
            "out": str(out),
            "backend": payload["backend"],
            "codegraph_available": codegraph_summary.get("available", False),
        },
    )
    return payload


def build_repo_map(files: list[dict[str, Any]], index: dict[str, Any]) -> dict[str, Any]:
    module_map: dict[str, dict[str, Any]] = {}
    docs: list[str] = []
    for record in files:
        top = record.get("top_level") or "."
        bucket = module_map.setdefault(top, {"file_count": 0, "kinds": {}, "languages": {}, "representative_files": []})
        bucket["file_count"] += 1
        kind = record.get("kind", "artifact")
        language = record.get("language") or "text"
        bucket["kinds"][kind] = bucket["kinds"].get(kind, 0) + 1
        bucket["languages"][language] = bucket["languages"].get(language, 0) + 1
        if len(bucket["representative_files"]) < 10 and kind in {"source", "test", "build", "config", "docs", "generated"}:
            bucket["representative_files"].append(record.get("path", ""))
        if kind == "docs":
            docs.append(record.get("path", ""))
    overview = {
        "stats": index.get("stats", {}),
        "top_level_count": len(module_map),
        "source_count": sum(1 for item in files if item.get("kind") == "source"),
        "test_count": sum(1 for item in files if item.get("is_test")),
        "generated_count": sum(1 for item in files if item.get("is_generated")),
        "build_count": sum(1 for item in files if item.get("kind") == "build"),
    }
    return {"overview": overview, "module_map": module_map, "docs": docs[:80]}


def compact_global(global_perception: dict[str, Any] | None) -> dict[str, Any]:
    if not global_perception:
        return {"available": False}
    return {
        "available": True,
        "backend": global_perception.get("backend", ""),
        "module_clusters": global_perception.get("module_clusters", [])[:18],
        "entrypoints": global_perception.get("entrypoints", [])[:30],
        "dependency_hints": global_perception.get("dependency_hints", [])[:50],
        "topology": global_perception.get("topology", {}),
        "source_of_truth_hints": global_perception.get("source_of_truth_hints", [])[:60],
        "global_risks": global_perception.get("global_risks", [])[:16],
    }


def compact_codegraph(codegraph_perception: dict[str, Any] | None) -> dict[str, Any]:
    if not codegraph_perception:
        return {"available": False, "status": "not-run", "note": "CodeGraph perception was not provided."}
    layer = {
        "available": bool(codegraph_perception.get("available")),
        "status": codegraph_perception.get("summary", {}).get("status", "unknown"),
        "summary": codegraph_perception.get("summary", {}),
        "query_evidence": [],
        "relationship_evidence": [],
        "notes": codegraph_perception.get("notes", []),
    }
    for item in codegraph_perception.get("repository_queries", [])[:10]:
        best = item.get("best", {})
        layer["query_evidence"].append(
            {
                "query": item.get("query", ""),
                "ok": best.get("ok", False),
                "returncode": best.get("returncode"),
                "stdout": str(best.get("stdout", ""))[:4000],
                "stderr": str(best.get("stderr", ""))[:1200],
            }
        )
    for item in codegraph_perception.get("symbol_relationships", [])[:10]:
        relationship = {"symbol": item.get("symbol", "")}
        for key in ("node", "callers", "callees", "impact"):
            result = item.get(key, {})
            relationship[key] = {
                "ok": result.get("ok", False),
                "returncode": result.get("returncode"),
                "stdout": str(result.get("stdout", ""))[:2400],
                "stderr": str(result.get("stderr", ""))[:800],
            }
        layer["relationship_evidence"].append(relationship)
    return layer


def source_of_truth(global_summary: dict[str, Any], repo_map: dict[str, Any]) -> dict[str, Any]:
    topology = global_summary.get("topology", {}) if global_summary else {}
    return {
        "hints": global_summary.get("source_of_truth_hints", [])[:60],
        "generated_files": topology.get("generated_files", [])[:60],
        "build_files": topology.get("build_files", [])[:60],
        "doc_files": repo_map.get("docs", [])[:60],
    }


def validation_surface(global_summary: dict[str, Any], repo_map: dict[str, Any]) -> dict[str, Any]:
    topology = global_summary.get("topology", {}) if global_summary else {}
    return {
        "test_roots": topology.get("test_roots", [])[:60],
        "build_files": topology.get("build_files", [])[:60],
        "module_test_counts": {
            name: values.get("kinds", {}).get("test", 0)
            for name, values in repo_map.get("module_map", {}).items()
            if values.get("kinds", {}).get("test", 0)
        },
    }


def repository_risks(repo_map: dict[str, Any], global_summary: dict[str, Any], codegraph_summary: dict[str, Any]) -> list[dict[str, str]]:
    risks: list[dict[str, str]] = []
    if not codegraph_summary.get("available"):
        risks.append(
            {
                "claim": "CodeGraph evidence is unavailable for this run.",
                "source": "tool_status",
                "confidence": "high",
                "action": "Use builtin repository perception and direct Chrys source inspection; do not assume caller/callee coverage.",
            }
        )
    if repo_map.get("overview", {}).get("generated_count", 0):
        risks.append(
            {
                "claim": "Generated-looking artifacts exist and may require source-of-truth updates.",
                "source": "code_evidence",
                "confidence": "medium",
                "action": "Check source-of-truth hints before editing generated-looking files.",
            }
        )
    for risk in global_summary.get("global_risks", [])[:8]:
        risks.append(
            {
                "claim": str(risk.get("claim", "")),
                "source": str(risk.get("source", "code_evidence")),
                "confidence": str(risk.get("confidence", "medium")),
                "action": str(risk.get("action", "")),
            }
        )
    return risks


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Repository Perception",
        "",
        "This document merges builtin static repository evidence with optional CodeGraph CLI evidence. Use it to enrich the Augmented Requirement; verify all code claims before editing.",
        "",
        "## Repository Overview",
        "",
        f"- Backend: {payload.get('backend', '')}",
    ]
    overview = payload.get("overview", {})
    stats = overview.get("stats", {})
    lines.append(f"- Files indexed: {stats.get('file_count', 0)}")
    lines.append(f"- Symbols indexed: {stats.get('symbol_count', 0)}")
    lines.append(f"- Source/test/generated/build counts: {overview.get('source_count', 0)}/{overview.get('test_count', 0)}/{overview.get('generated_count', 0)}/{overview.get('build_count', 0)}")
    lines.extend(["", "## Architecture and Module Map", ""])
    for name, values in sorted(payload.get("module_map", {}).items(), key=lambda item: (-item[1].get("file_count", 0), item[0]))[:30]:
        lines.append(f"- `{name}`: {values.get('file_count', 0)} files, kinds={values.get('kinds', {})}, languages={values.get('languages', {})}")
        reps = ", ".join(f"`{item}`" for item in values.get("representative_files", [])[:6])
        if reps:
            lines.append(f"  Representative files: {reps}")
    global_layer = payload.get("global_layer", {})
    lines.extend(["", "## Entrypoints and Dependency Hints", ""])
    lines.extend(
        bullet_lines(
            [
                f"`{item.get('path', '')}` ({item.get('kind', '')}, score={item.get('score', 0)}): {item.get('reason', '')}"
                for item in global_layer.get("entrypoints", [])[:30]
            ]
        )
    )
    lines.extend(["", "### Dependency Hints"])
    lines.extend(
        bullet_lines(
            [
                f"`{item.get('from', '')}` -> `{item.get('to', '')}` ({item.get('kind', '')})"
                for item in global_layer.get("dependency_hints", [])[:40]
            ]
        )
    )
    codegraph = payload.get("codegraph_layer", {})
    lines.extend(["", "## CodeGraph Layer", ""])
    lines.append(f"- Available: {codegraph.get('available')}")
    lines.append(f"- Status: {codegraph.get('status')}")
    summary = codegraph.get("summary", {})
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "### CodeGraph Query Evidence"])
    for item in codegraph.get("query_evidence", [])[:8]:
        lines.append(f"- Query: `{markdown_escape_line(item.get('query', ''))}` ok={item.get('ok')} rc={item.get('returncode')}")
        output = item.get("stdout") or item.get("stderr") or ""
        if output:
            lines.extend(["", "```text", output[:1600].rstrip(), "```", ""])
    lines.extend(["", "### CodeGraph Symbol Relationships"])
    for item in codegraph.get("relationship_evidence", [])[:8]:
        lines.append(f"- Symbol: `{markdown_escape_line(item.get('symbol', ''))}`")
        for key in ("node", "callers", "callees", "impact"):
            result = item.get(key, {})
            lines.append(f"  - {key}: ok={result.get('ok')} rc={result.get('returncode')}")
    source = payload.get("source_of_truth", {})
    lines.extend(["", "## Source-of-truth and Generated/Build Chain", ""])
    lines.extend(bullet_lines([f"`{item.get('path', '')}` ({item.get('kind', '')}/{item.get('language', '')}): {item.get('why', '')}" for item in source.get("hints", [])[:40]]))
    lines.extend(["", "### Generated-looking Files"])
    lines.extend(bullet_lines([f"`{item}`" for item in source.get("generated_files", [])[:40]]))
    lines.extend(["", "### Build Files"])
    lines.extend(bullet_lines([f"`{item}`" for item in source.get("build_files", [])[:40]]))
    validation = payload.get("validation_surface", {})
    lines.extend(["", "## Validation Surface", ""])
    lines.extend(bullet_lines([f"`{item.get('root')}` ({item.get('files')} files)" for item in validation.get("test_roots", [])[:40]]))
    lines.extend(["", "## Repository Risks for Requirement Augmentation", ""])
    for risk in payload.get("risks", []):
        lines.append(f"- Claim: {risk.get('claim', '')}")
        lines.append(f"  Source: {risk.get('source', '')}")
        lines.append(f"  Confidence: {risk.get('confidence', '')}")
        lines.append(f"  Action: {risk.get('action', '')}")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        payload = build_repository_perception(args)
        print(f"Wrote repository perception: {resolve_path(args.out)}")
        print({"backend": payload["backend"], "codegraph_available": payload["codegraph_layer"].get("available")})
        return 0
    except ScriptError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
