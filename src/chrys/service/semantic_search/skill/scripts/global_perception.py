#!/usr/bin/env python3
"""Build global repository perception for semantic-search requirement augmentation."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _common import (
    FORMAT_GLOBAL,
    FORMAT_INDEX,
    ScriptError,
    append_trace,
    bullet_lines,
    ensure_allowed_path,
    load_json,
    markdown_escape_line,
    now_iso,
    resolve_path,
    sha1_path,
    stable_unique,
    tokenize,
    write_json,
)

IMPORT_PATTERNS = (
    re.compile(r"^\s*from\s+([A-Za-z_][A-Za-z0-9_.]*)\s+import\s+", re.MULTILINE),
    re.compile(r"^\s*import\s+([A-Za-z_][A-Za-z0-9_.]*)", re.MULTILINE),
    re.compile(r"^\s*#\s*include\s+[<\"]([^>\"]+)[>\"]", re.MULTILINE),
    re.compile(r"^\s*import\s+([A-Za-z_][A-Za-z0-9_.]*)\s*;?", re.MULTILINE),
    re.compile(r"^\s*use\s+([A-Za-z_][A-Za-z0-9_:]*)", re.MULTILINE),
)

ENTRYPOINT_TERMS = {
    "__init__",
    "api",
    "app",
    "cli",
    "command",
    "compiler",
    "config",
    "entry",
    "grammar",
    "main",
    "parser",
    "runtime",
    "server",
    "setup",
    "token",
}

SOURCE_OF_TRUTH_TERMS = {
    "asdl",
    "autogen",
    "codegen",
    "generate",
    "generated",
    "grammar",
    "lexer",
    "parser",
    "schema",
    "token",
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Seeded workspace repo root.")
    parser.add_argument("--index", required=True, help="Light semantic-search index from build_index.py.")
    parser.add_argument("--out", required=True, help="Output global-perception.json path.")
    parser.add_argument("--markdown", help="Output global-perception.md path. Defaults to out with .md suffix.")
    parser.add_argument("--artifact-dir", help="Semantic-search artifact directory. Defaults to output parent.")
    parser.add_argument(
        "--external-graph-json",
        default=os.environ.get("SEMANTIC_SEARCH_EXTERNAL_GRAPH_JSON", ""),
        help="Optional CodeGraph/GitNexus-style precomputed graph JSON to summarize and merge.",
    )
    parser.add_argument("--max-clusters", type=int, default=24)
    parser.add_argument("--max-edges", type=int, default=80)
    return parser.parse_args(argv)


def load_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, dict[str, Any], dict[str, Any] | None]:
    repo = resolve_path(args.repo)
    if not repo.is_dir():
        raise ScriptError(f"repo path does not exist: {repo}")
    out = resolve_path(args.out)
    artifact_dir = resolve_path(args.artifact_dir or out.parent)
    index_path = ensure_allowed_path(args.index, allowed_roots=[repo, artifact_dir], purpose="index")
    out = ensure_allowed_path(out, allowed_roots=[repo, artifact_dir, out.parent], purpose="output")
    markdown = resolve_path(args.markdown or out.with_suffix(".md"))
    markdown = ensure_allowed_path(
        markdown, allowed_roots=[repo, artifact_dir, markdown.parent], purpose="markdown-output"
    )
    index = load_json(index_path)
    if index.get("format") != FORMAT_INDEX:
        raise ScriptError(f"unsupported index format: {index.get('format')}")
    external_graph = None
    external_value = args.external_graph_json.strip()
    if external_value:
        external_path = ensure_allowed_path(
            external_value, allowed_roots=[repo, artifact_dir], purpose="external-graph-json"
        )
        external_graph = load_json(external_path)
    return repo, index_path, out, markdown, artifact_dir, index, external_graph


def build_global_perception(args: argparse.Namespace) -> dict[str, Any]:
    repo, index_path, out, markdown, _artifact_dir, index, external_graph = load_inputs(args)
    files = list(index.get("files", []))
    perception = {
        "format": FORMAT_GLOBAL,
        "created_at": now_iso(),
        "backend": "builtin" + ("+external-graph" if external_graph is not None else ""),
        "inputs": {
            "repo": str(repo),
            "index": str(index_path),
            "index_sha1": sha1_path(index_path),
        },
        "overview": overview(index, files),
        "module_clusters": module_clusters(files, args.max_clusters),
        "entrypoints": entrypoints(files),
        "dependency_hints": dependency_hints(files, args.max_edges),
        "topology": topology(files),
        "source_of_truth_hints": source_of_truth_hints(files),
        "external_graph_summary": external_graph_summary(external_graph),
        "global_risks": global_risks(files),
    }
    write_json(out, perception)
    markdown.write_text(render_markdown(perception), encoding="utf-8")
    append_trace("global-perception", {"out": str(out), "markdown": str(markdown), "backend": perception["backend"]})
    return perception


def overview(index: dict[str, Any], files: list[dict[str, Any]]) -> dict[str, Any]:
    top_levels = Counter(record.get("top_level") or "." for record in files)
    source_count = sum(1 for record in files if record.get("kind") == "source")
    test_count = sum(1 for record in files if record.get("is_test"))
    generated_count = sum(1 for record in files if record.get("is_generated"))
    build_count = sum(1 for record in files if record.get("kind") == "build")
    return {
        "stats": index.get("stats", {}),
        "top_levels_by_file_count": [{"name": name, "files": count} for name, count in top_levels.most_common(30)],
        "source_count": source_count,
        "test_count": test_count,
        "generated_count": generated_count,
        "build_count": build_count,
    }


def module_clusters(files: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in files:
        grouped[cluster_key(record)].append(record)
    clusters: list[dict[str, Any]] = []
    for key, records in grouped.items():
        kind_counts = Counter(record.get("kind", "artifact") for record in records)
        language_counts = Counter(record.get("language") or "text" for record in records)
        symbols = []
        for record in records:
            for symbol in record.get("symbols", [])[:4]:
                symbols.append({"name": symbol.get("name"), "kind": symbol.get("kind"), "file": record.get("path")})
        clusters.append(
            {
                "name": key,
                "file_count": len(records),
                "kind_counts": dict(kind_counts),
                "language_counts": dict(language_counts),
                "representative_files": [
                    record["path"] for record in sorted(records, key=representative_sort_key)[:10]
                ],
                "representative_symbols": symbols[:18],
            }
        )
    clusters.sort(key=lambda item: (-item["file_count"], item["name"]))
    return clusters[:limit]


def cluster_key(record: dict[str, Any]) -> str:
    parts = Path(record.get("path", "")).parts
    if len(parts) >= 2 and parts[0] in {"src", "lib", "Lib", "test", "tests"}:
        return "/".join(parts[:2])
    if parts:
        return parts[0]
    return "."


def representative_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    kind_rank = {"source": 0, "generated": 1, "test": 2, "build": 3, "config": 4, "docs": 5}.get(record.get("kind"), 9)
    return (kind_rank, int(record.get("size", 0)), record.get("path", ""))


def entrypoints(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = []
    for record in files:
        path = record.get("path", "")
        lowered = path.lower()
        path_terms = set(tokenize(path))
        score = 0
        reasons = []
        if path_terms & ENTRYPOINT_TERMS:
            score += 5
            reasons.append("path suggests entrypoint or core subsystem")
        if record.get("kind") in {"build", "config"}:
            score += 3
            reasons.append("build/config entry")
        if record.get("language") == "grammar":
            score += 4
            reasons.append("grammar/schema source")
        if lowered.endswith(("main.py", "__main__.py", "setup.py", "pyproject.toml", "makefile", "pom.xml")):
            score += 6
            reasons.append("well-known entrypoint filename")
        if record.get("symbols"):
            names = " ".join(symbol.get("name", "") for symbol in record.get("symbols", [])[:12]).lower()
            if any(term in names for term in ENTRYPOINT_TERMS):
                score += 2
                reasons.append("symbols suggest entrypoint or core subsystem")
        if score <= 0:
            continue
        candidates.append(
            {
                "path": path,
                "kind": record.get("kind"),
                "language": record.get("language"),
                "score": score,
                "reason": "; ".join(stable_unique(reasons)),
                "symbols": [symbol.get("name") for symbol in record.get("symbols", [])[:8]],
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["path"]))
    return candidates[:40]


def dependency_hints(files: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for record in files:
        preview = str(record.get("preview", ""))
        if not preview:
            continue
        for pattern in IMPORT_PATTERNS:
            for match in pattern.finditer(preview):
                target = match.group(1).strip()
                if not target:
                    continue
                edges.append(
                    {
                        "from": record.get("path"),
                        "to": target,
                        "from_cluster": cluster_key(record),
                        "kind": "import/include/use",
                    }
                )
                if len(edges) >= limit:
                    return edges
    return edges


def topology(files: list[dict[str, Any]]) -> dict[str, Any]:
    generated = [record["path"] for record in files if record.get("is_generated")]
    build = [record["path"] for record in files if record.get("kind") == "build"]
    config = [record["path"] for record in files if record.get("kind") == "config"]
    docs = [record["path"] for record in files if record.get("kind") == "docs"]
    tests_by_root: dict[str, int] = defaultdict(int)
    for record in files:
        if record.get("is_test"):
            parts = Path(record.get("path", "")).parts
            root = "/".join(parts[:2]) if len(parts) > 1 else record.get("path", "")
            tests_by_root[root] += 1
    return {
        "generated_files": generated[:80],
        "build_files": build[:80],
        "config_files": config[:80],
        "doc_files": docs[:80],
        "test_roots": [{"root": root, "files": count} for root, count in sorted(tests_by_root.items())[:80]],
    }


def source_of_truth_hints(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hints = []
    for record in files:
        terms = {str(term).lower() for term in record.get("terms", [])}
        path_terms = set(tokenize(record.get("path", "")))
        if not (
            terms & SOURCE_OF_TRUTH_TERMS
            or path_terms & SOURCE_OF_TRUTH_TERMS
            or record.get("language") in {"grammar", "schema"}
        ):
            continue
        hints.append(
            {
                "path": record.get("path"),
                "kind": record.get("kind"),
                "language": record.get("language"),
                "why": "Path/content/suffix suggests grammar, schema, token, generator, or generated-artifact relationship.",
                "is_generated": bool(record.get("is_generated")),
            }
        )
    return hints[:80]


def external_graph_summary(external_graph: dict[str, Any] | None) -> dict[str, Any]:
    if external_graph is None:
        return {
            "present": False,
            "note": "No external CodeGraph/GitNexus JSON was provided; using builtin global perception.",
        }
    keys = sorted(str(key) for key in external_graph)[:40]
    counts = {}
    samples = {}
    for key, value in external_graph.items():
        if isinstance(value, list):
            counts[str(key)] = len(value)
            samples[str(key)] = value[:5]
        elif isinstance(value, dict):
            counts[str(key)] = len(value)
            samples[str(key)] = dict(list(value.items())[:5])
        else:
            samples[str(key)] = str(value)[:500]
    return {
        "present": True,
        "top_level_keys": keys,
        "counts": counts,
        "samples": samples,
        "note": "External graph was summarized as additional global perception evidence.",
    }


def global_risks(files: list[dict[str, Any]]) -> list[dict[str, str]]:
    has_generated = any(record.get("is_generated") for record in files)
    has_build = any(record.get("kind") == "build" for record in files)
    has_tests = any(record.get("is_test") for record in files)
    risks = [
        {
            "claim": "Task-specific matches can miss cross-module registration and caller/callee relationships.",
            "source": "inferred",
            "confidence": "medium",
            "action": "Use global perception to inspect nearby modules, entrypoints, and dependency hints before editing.",
        }
    ]
    if has_generated:
        risks.append(
            {
                "claim": "Generated-looking artifacts exist in the repository.",
                "source": "code_evidence",
                "confidence": "medium",
                "action": "Find the source-of-truth or generation path before editing generated files directly.",
            }
        )
    if has_build:
        risks.append(
            {
                "claim": "Build/config entrypoints exist and may need registration or generation updates.",
                "source": "code_evidence",
                "confidence": "medium",
                "action": "Check build/config topology when source changes add files, features, or generated artifacts.",
            }
        )
    if has_tests:
        risks.append(
            {
                "claim": "Existing tests can reveal local behavior style and narrow validation entrypoints.",
                "source": "code_evidence",
                "confidence": "medium",
                "action": "Use test topology for validation planning, not as a substitute for implementation.",
            }
        )
    return risks


def render_markdown(perception: dict[str, Any]) -> str:
    lines = [
        "# Global Codebase Perception",
        "",
        f"Backend: {perception.get('backend', '')}",
        "",
        "## Overview",
        "",
    ]
    overview_data = perception.get("overview", {})
    stats = overview_data.get("stats", {})
    lines.append(f"- Files indexed: {stats.get('file_count', 0)}")
    lines.append(f"- Symbols indexed: {stats.get('symbol_count', 0)}")
    lines.append(f"- Source files: {overview_data.get('source_count', 0)}")
    lines.append(f"- Tests: {overview_data.get('test_count', 0)}")
    lines.append(f"- Generated-looking files: {overview_data.get('generated_count', 0)}")
    lines.append(f"- Build files: {overview_data.get('build_count', 0)}")
    lines.extend(["", "## Module Clusters", ""])
    for cluster in perception.get("module_clusters", [])[:24]:
        lines.append(
            f"- `{cluster.get('name')}`: {cluster.get('file_count')} files, kinds={cluster.get('kind_counts')}"
        )
        reps = ", ".join(f"`{item}`" for item in cluster.get("representative_files", [])[:5])
        if reps:
            lines.append(f"  Representative files: {reps}")
    lines.extend(["", "## Entrypoints and Core Surfaces", ""])
    lines.extend(
        bullet_lines(
            [
                f"`{item.get('path')}` ({item.get('kind')}, score={item.get('score')}): {item.get('reason')}"
                for item in perception.get("entrypoints", [])[:30]
            ]
        )
    )
    lines.extend(["", "## Dependency Hints", ""])
    lines.extend(
        bullet_lines(
            [
                f"`{item.get('from')}` -> `{item.get('to')}` ({item.get('kind')})"
                for item in perception.get("dependency_hints", [])[:40]
            ]
        )
    )
    topology_data = perception.get("topology", {})
    lines.extend(["", "## Build/Test/Generated Topology", "", "### Generated-looking Files"])
    lines.extend(bullet_lines([f"`{item}`" for item in topology_data.get("generated_files", [])[:40]]))
    lines.extend(["", "### Build Files"])
    lines.extend(bullet_lines([f"`{item}`" for item in topology_data.get("build_files", [])[:40]]))
    lines.extend(["", "### Test Roots"])
    lines.extend(
        bullet_lines(
            [f"`{item.get('root')}` ({item.get('files')} files)" for item in topology_data.get("test_roots", [])[:40]]
        )
    )
    lines.extend(["", "## Source-of-truth Hints", ""])
    lines.extend(
        bullet_lines(
            [
                f"`{item.get('path')}` ({item.get('kind')}/{item.get('language')}): {item.get('why')}"
                for item in perception.get("source_of_truth_hints", [])[:40]
            ]
        )
    )
    external = perception.get("external_graph_summary", {})
    lines.extend(["", "## External Graph Summary", ""])
    lines.append(f"- Present: {external.get('present')}")
    lines.append(f"- Note: {markdown_escape_line(str(external.get('note', '')))}")
    lines.extend(["", "## Global Risks", ""])
    for risk in perception.get("global_risks", []):
        lines.append(f"- Claim: {risk.get('claim')}")
        lines.append(f"  Source: {risk.get('source')}")
        lines.append(f"  Confidence: {risk.get('confidence')}")
        lines.append(f"  Action: {risk.get('action')}")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        perception = build_global_perception(args)
        print(f"Wrote global perception: {resolve_path(args.out)}")
        print(
            {
                "backend": perception["backend"],
                "clusters": len(perception["module_clusters"]),
                "dependency_hints": len(perception["dependency_hints"]),
            }
        )
        return 0
    except ScriptError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
