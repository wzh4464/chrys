#!/usr/bin/env python3
"""Repository graph adapter used by the Chrys-native SemLoc workflow.

The adapter deliberately keeps CodeGraph as the primary graph backend.  It
normalizes CodeGraph observations and supplements them with inexpensive source
analysis so the localization tools have one stable file/function interface.
"""

from __future__ import annotations

import ast
import fnmatch
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from _common import path_tokens, stable_unique

FILE_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./-]+\.(?:py|java|scala|sc|rs|c|h|cc|cpp|cxx|hh|hpp|hxx|g4|proto))",
    flags=re.IGNORECASE,
)
PATH_SYMBOL_RE = re.compile(
    r"(?P<path>[A-Za-z0-9_./-]+\.(?:py|java|scala|sc|rs|c|h|cc|cpp|cxx|hh|hpp|hxx))"
    r"(?::|#)\s*(?P<symbol>[A-Za-z_][A-Za-z0-9_.$:]*)",
    flags=re.IGNORECASE,
)


def normalize_relative_path(value: str, repo: Path) -> str:
    """Normalize a path emitted by CodeGraph without admitting outside files."""
    raw = value.strip().strip("`'\"()[]{}<>,;").replace("\\", "/")
    if not raw:
        return ""
    candidate = Path(raw)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(repo).as_posix()
        except ValueError:
            return ""
    while raw.startswith("./"):
        raw = raw[2:]
    parts = Path(raw).parts
    if ".." in parts:
        return ""
    return Path(raw).as_posix()


def bare_symbol(value: str) -> str:
    return value.rsplit(".", 1)[-1].rsplit("::", 1)[-1].rsplit("$", 1)[-1]


class LocalizationGraph:
    """Normalized repository units and semantic relationships."""

    def __init__(self, repo: Path, index: dict[str, Any], codegraph: dict[str, Any] | None = None):
        self.repo = repo.resolve()
        self.index = index
        self.codegraph = codegraph or {}
        self.files: dict[str, dict[str, Any]] = {
            str(record.get("path", "")): record for record in index.get("files", []) if record.get("path")
        }
        self.units: list[dict[str, Any]] = []
        self.units_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.units_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.units_by_key: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, list[dict[str, str]]] = defaultdict(list)
        self.codegraph_paths: set[str] = set()
        self.dynamic_codegraph_evidence: list[dict[str, Any]] = []
        self._dynamic_expanded: set[str] = set()
        self._python_metadata: dict[str, dict[str, Any]] = {}
        self._load_units()
        self._index_units()
        self._build_source_edges()
        self._load_codegraph_edges()

    @staticmethod
    def unit_key(unit: dict[str, Any]) -> str:
        return f"{unit.get('path', '')}:{unit.get('qualified_name') or unit.get('name', '')}"

    def _read_indexed_file(self, relative: str) -> str:
        if relative not in self.files:
            return ""
        path = (self.repo / relative).resolve()
        try:
            path.relative_to(self.repo)
        except ValueError:
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")
        except OSError:
            return ""

    def read_file(self, relative: str) -> str:
        return self._read_indexed_file(relative)

    def _load_units(self) -> None:
        for relative, record in self.files.items():
            language = record.get("language", "")
            text = self._read_indexed_file(relative) if language and int(record.get("size") or 0) <= 2_000_000 else ""
            if language == "python" and text:
                python_units, metadata = self._python_units(relative, record, text)
                if python_units:
                    self.units.extend(python_units)
                    self._python_metadata[relative] = metadata
                    continue
            self.units.extend(self._index_units_for_file(relative, record, text))

    def _python_units(
        self, relative: str, record: dict[str, Any], text: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return [], {}
        lines = text.splitlines()
        units: list[dict[str, Any]] = []
        aliases: dict[str, str] = {}
        class_bases: dict[str, list[str]] = {}
        call_kinds: dict[str, list[tuple[str, str]]] = defaultdict(list)

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                self._collect_import_aliases(node, aliases)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                self._collect_assignment_alias(node, aliases)

        def visit_body(body: list[ast.stmt], parent_class: str = "", parent_unit: str = "") -> None:
            for node in body:
                if isinstance(node, ast.ClassDef):
                    qualified = f"{parent_class}.{node.name}" if parent_class else node.name
                    unit = self._unit_from_python_node(relative, record, text, lines, node, qualified, parent_class)
                    units.append(unit)
                    class_bases[qualified] = [self._ast_name(base) for base in node.bases if self._ast_name(base)]
                    visit_body(node.body, qualified, self.unit_key(unit))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = f"{parent_class}.{node.name}" if parent_class else node.name
                    unit = self._unit_from_python_node(relative, record, text, lines, node, qualified, parent_class)
                    unit["parent_unit"] = parent_unit
                    units.append(unit)
                    key = self.unit_key(unit)
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            target = self._ast_name(child.func)
                            if target:
                                kind = "async_call" if self._is_async_dispatch(child) else "call"
                                call_kinds[key].append((target, kind))
                                protocol_target = {
                                    "len": "__len__",
                                    "str": "__str__",
                                    "repr": "__repr__",
                                    "bool": "__bool__",
                                    "hash": "__hash__",
                                    "iter": "__iter__",
                                    "next": "__next__",
                                }.get(bare_symbol(target))
                                if protocol_target:
                                    call_kinds[key].append((protocol_target, "implicit_call"))
                                call_kinds[key].extend(self._callback_targets(child))
                        elif isinstance(child, ast.AsyncWith):
                            call_kinds[key].extend([("__aenter__", "implicit_call"), ("__aexit__", "implicit_call")])
                        elif isinstance(child, ast.With):
                            call_kinds[key].extend([("__enter__", "implicit_call"), ("__exit__", "implicit_call")])
                        elif isinstance(child, ast.AsyncFor):
                            call_kinds[key].append(("__aiter__", "implicit_call"))
                        elif isinstance(child, ast.For):
                            call_kinds[key].append(("__iter__", "implicit_call"))
                        elif isinstance(child, ast.Await):
                            target = self._ast_name(child.value.func) if isinstance(child.value, ast.Call) else ""
                            if target:
                                call_kinds[key].append((target, "async_call"))
                    visit_body(node.body, parent_class, key)

        visit_body(tree.body)
        return units, {"aliases": aliases, "class_bases": class_bases, "calls": call_kinds}

    def _unit_from_python_node(
        self,
        relative: str,
        record: dict[str, Any],
        text: str,
        lines: list[str],
        node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
        qualified: str,
        parent_class: str,
    ) -> dict[str, Any]:
        start = int(getattr(node, "lineno", 1))
        end = int(getattr(node, "end_lineno", start))
        signature = lines[start - 1].strip()[:240] if 0 < start <= len(lines) else ""
        content = "\n".join(lines[start - 1 : end])
        kind = "class" if isinstance(node, ast.ClassDef) else ("method" if parent_class else "function")
        return {
            "repo": record.get("repo", ""),
            "path": relative,
            "name": node.name,
            "qualified_name": qualified,
            "class_name": parent_class,
            "kind": kind,
            "start_line": start,
            "end_line": end,
            "signature": signature,
            "content": content,
            "parent_unit": "",
        }

    @staticmethod
    def _collect_import_aliases(node: ast.Import | ast.ImportFrom, aliases: dict[str, str]) -> None:
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = item.name
            return
        module = node.module or ""
        for item in node.names:
            original = f"{module}.{item.name}" if module else item.name
            aliases[item.asname or item.name] = original

    @classmethod
    def _collect_assignment_alias(cls, node: ast.Assign | ast.AnnAssign, aliases: dict[str, str]) -> None:
        value = cls._ast_name(node.value) if node.value is not None else ""
        if not value:
            return
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            name = cls._ast_name(target)
            if name:
                aliases[name] = value

    @staticmethod
    def _ast_name(node: ast.AST | None) -> str:
        if node is None:
            return ""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = LocalizationGraph._ast_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    @classmethod
    def _callback_targets(cls, call: ast.Call) -> list[tuple[str, str]]:
        method = bare_symbol(cls._ast_name(call.func))
        positional = {
            "submit": 0,
            "map": 0,
            "create_task": 0,
            "ensure_future": 0,
            "add_done_callback": 0,
            "call_soon": 0,
            "call_later": 1,
            "register": 0,
            "connect": 0,
            "on": 1,
        }
        index = len(call.args) - 1 if method in {"register", "connect", "on"} and call.args else positional.get(method)
        if index is None or len(call.args) <= index:
            return []
        target_node = call.args[index]
        if isinstance(target_node, ast.Call):
            target_node = target_node.func
        target = cls._ast_name(target_node)
        return [(target, "async_call")] if target else []

    @classmethod
    def _is_async_dispatch(cls, call: ast.Call) -> bool:
        return bare_symbol(cls._ast_name(call.func)) in {
            "submit",
            "create_task",
            "ensure_future",
            "add_done_callback",
            "call_soon",
            "call_later",
            "run_in_executor",
        }

    def _index_units_for_file(self, relative: str, record: dict[str, Any], text: str) -> list[dict[str, Any]]:
        symbols = sorted(record.get("symbols", []), key=lambda item: int(item.get("line") or 0))
        lines = text.splitlines()
        units: list[dict[str, Any]] = []
        for position, symbol in enumerate(symbols):
            start = max(int(symbol.get("line") or 1), 1)
            next_start = (
                int(symbols[position + 1].get("line") or len(lines) + 1)
                if position + 1 < len(symbols)
                else len(lines) + 1
            )
            end = max(start, next_start - 1)
            qualified = str(symbol.get("name", ""))
            class_name = qualified.rsplit(".", 1)[0] if "." in qualified else ""
            name = bare_symbol(qualified)
            units.append(
                {
                    "repo": record.get("repo", ""),
                    "path": relative,
                    "name": name,
                    "qualified_name": qualified,
                    "class_name": class_name,
                    "kind": symbol.get("kind", "symbol"),
                    "start_line": start,
                    "end_line": end,
                    "signature": symbol.get("signature", ""),
                    "content": "\n".join(lines[start - 1 : end]),
                    "parent_unit": "",
                }
            )
        return units

    def _index_units(self) -> None:
        for unit in self.units:
            self.units_by_key[self.unit_key(unit)] = unit
            self.units_by_file[unit["path"]].append(unit)
            for name in stable_unique(
                [unit.get("name"), unit.get("qualified_name"), bare_symbol(unit.get("qualified_name", ""))]
            ):
                if name:
                    self.units_by_name[str(name)].append(unit)
        for values in self.units_by_file.values():
            values.sort(key=lambda item: (item.get("start_line", 0), item.get("qualified_name", "")))

    def _build_source_edges(self) -> None:
        for unit in self.units:
            parent = unit.get("parent_unit", "")
            if parent:
                self._add_edge(parent, self.unit_key(unit), "child")
        for relative, metadata in self._python_metadata.items():
            aliases = metadata.get("aliases", {})
            for source_key, calls in metadata.get("calls", {}).items():
                for target_name, kind in calls:
                    alias_target = aliases.get(target_name) or aliases.get(target_name.split(".", 1)[0], "")
                    resolved_name = alias_target or target_name
                    targets = self._resolve_name(resolved_name, preferred_file=relative)
                    for target in targets[:8]:
                        target_key = self.unit_key(target)
                        edge_kind = kind if target.get("path") == relative else f"cross_file_{kind}"
                        self._add_edge(source_key, target_key, edge_kind)
                        self._add_edge(target_key, source_key, "caller")
                    if alias_target:
                        for target in targets[:8]:
                            self._add_edge(source_key, self.unit_key(target), "alias")
            for class_name, bases in metadata.get("class_bases", {}).items():
                sources = self._resolve_name(class_name, preferred_file=relative)
                for source in sources:
                    for base in bases:
                        for target in self._resolve_name(base)[:8]:
                            self._add_edge(self.unit_key(source), self.unit_key(target), "inherits")
                            source_methods = self._methods_for_class(
                                str(source.get("qualified_name") or source.get("name"))
                            )
                            target_methods = self._methods_for_class(
                                str(target.get("qualified_name") or target.get("name"))
                            )
                            target_by_name = {str(method.get("name")): method for method in target_methods}
                            for method in source_methods:
                                overridden = target_by_name.get(str(method.get("name")))
                                if overridden:
                                    self._add_edge(self.unit_key(method), self.unit_key(overridden), "override")
                                    self._add_edge(self.unit_key(overridden), self.unit_key(method), "overridden_by")

    def _methods_for_class(self, class_name: str) -> list[dict[str, Any]]:
        return [
            unit
            for unit in self.units
            if unit.get("kind") == "method" and unit.get("class_name") in {class_name, bare_symbol(class_name)}
        ]

    def _resolve_name(self, value: str, preferred_file: str = "") -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for name in stable_unique([value, bare_symbol(value)]):
            candidates.extend(self.units_by_name.get(str(name), []))
        candidates = stable_unique(candidates)
        candidates.sort(
            key=lambda item: (item.get("path") != preferred_file, item.get("path", ""), item.get("start_line", 0))
        )
        return candidates

    def _load_codegraph_edges(self) -> None:
        for query in self.codegraph.get("repository_queries", []):
            best = query.get("best", {})
            output = str(best.get("stdout", "") or best.get("stderr", ""))
            self._remember_codegraph_paths(output)
        for relationship in self.codegraph.get("symbol_relationships", []):
            symbol = str(relationship.get("symbol", ""))
            sources = self._resolve_name(symbol)
            for relation_name in ("node", "callers", "callees", "impact"):
                result = relationship.get(relation_name, {})
                output = str(result.get("stdout", "") or result.get("stderr", ""))
                self._remember_codegraph_paths(output)
                targets = self._units_from_output(output)
                for source in sources:
                    for target in targets[:24]:
                        if self.unit_key(source) == self.unit_key(target) and relation_name == "node":
                            continue
                        kind = "codegraph_reference" if relation_name == "node" else relation_name.rstrip("s")
                        self._add_edge(self.unit_key(source), self.unit_key(target), kind)

    def _remember_codegraph_paths(self, output: str) -> None:
        for match in FILE_RE.finditer(output):
            relative = normalize_relative_path(match.group("path"), self.repo)
            if relative in self.files:
                self.codegraph_paths.add(relative)

    def _units_from_output(self, output: str) -> list[dict[str, Any]]:
        units: list[dict[str, Any]] = []
        for match in PATH_SYMBOL_RE.finditer(output):
            relative = normalize_relative_path(match.group("path"), self.repo)
            if relative:
                units.extend(self.resolve_unit(relative, match.group("symbol")))
        for match in FILE_RE.finditer(output):
            relative = normalize_relative_path(match.group("path"), self.repo)
            if relative in self.units_by_file:
                units.extend(self.units_by_file[relative][:8])
        return stable_unique(units)

    def _add_edge(self, source: str, target: str, kind: str) -> None:
        if not source or not target:
            return
        edge = {"target": target, "kind": kind}
        if edge not in self.edges[source]:
            self.edges[source].append(edge)

    def file_matches(self, file_name: str, dir_path: str = ".") -> list[str]:
        normalized_dir = dir_path.strip().strip("/")
        if normalized_dir in {"", "."}:
            normalized_dir = ""
        if ".." in Path(normalized_dir).parts:
            return []
        matches = []
        for relative in self.files:
            if normalized_dir and not (relative == normalized_dir or relative.startswith(normalized_dir + "/")):
                continue
            if fnmatch.fnmatch(relative, file_name) or fnmatch.fnmatch(Path(relative).name, file_name):
                matches.append(relative)
        return sorted(matches)

    def file_pattern_matches(self, relative: str, pattern: str | None) -> bool:
        if not pattern:
            return True
        return fnmatch.fnmatch(relative, pattern) or fnmatch.fnmatch(Path(relative).name, pattern)

    def resolve_unit(self, relative: str, name: str = "") -> list[dict[str, Any]]:
        candidates = self.units_by_file.get(relative, [])
        if not name:
            return list(candidates)
        wanted = {name, bare_symbol(name)}
        return [
            unit
            for unit in candidates
            if unit.get("name") in wanted
            or unit.get("qualified_name") in wanted
            or bare_symbol(str(unit.get("qualified_name", ""))) in wanted
        ]

    def containing_unit(self, relative: str, line: int) -> dict[str, Any] | None:
        candidates = [
            unit
            for unit in self.units_by_file.get(relative, [])
            if int(unit.get("start_line") or 0) <= line <= int(unit.get("end_line") or 0)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda item: int(item.get("end_line") or 0) - int(item.get("start_line") or 0))

    def neighbors(self, unit: dict[str, Any], *, dynamic: bool = False) -> list[dict[str, Any]]:
        if dynamic:
            self.expand_codegraph(unit)
        results = []
        for edge in self.edges.get(self.unit_key(unit), []):
            target = self.unit_by_key(edge["target"])
            if target:
                results.append({"kind": edge["kind"], "unit": target})
        return results

    def expand_codegraph(self, unit: dict[str, Any]) -> None:
        """Ask CodeGraph for relationships discovered after the initial requirement query."""
        source_key = self.unit_key(unit)
        if source_key in self._dynamic_expanded:
            return
        self._dynamic_expanded.add(source_key)
        if len(self.dynamic_codegraph_evidence) >= 24:
            return
        command = self.codegraph.get("inputs", {}).get("command", [])
        if not self.codegraph.get("available") or not isinstance(command, list) or not command:
            return
        if not all(isinstance(item, str) and item for item in command):
            return
        symbol = str(unit.get("qualified_name") or unit.get("name") or "")
        if not symbol:
            return
        timeout = float(os.environ.get("SEMANTIC_SEARCH_CODEGRAPH_DYNAMIC_TIMEOUT", "10"))
        max_chars = int(os.environ.get("SEMANTIC_SEARCH_CODEGRAPH_DYNAMIC_MAX_CHARS", "6000"))
        for relation in ("node", "callers", "callees", "impact"):
            argv = [*command, relation, symbol]
            try:
                proc = subprocess.run(  # noqa: S603
                    argv,
                    cwd=str(self.repo),
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                )
                output = str(proc.stdout or proc.stderr or "")[:max_chars]
                result = {"argv": argv, "returncode": proc.returncode, "ok": proc.returncode == 0, "output": output}
            except (OSError, subprocess.TimeoutExpired) as err:
                output = ""
                result = {"argv": argv, "returncode": 124, "ok": False, "output": "", "error": str(err)}
            self.dynamic_codegraph_evidence.append({"source": source_key, "relation": relation, **result})
            self._remember_codegraph_paths(output)
            for target in self._units_from_output(output)[:24]:
                if self.unit_key(target) == source_key and relation == "node":
                    continue
                kind = "codegraph_reference" if relation == "node" else relation.rstrip("s")
                self._add_edge(source_key, self.unit_key(target), kind)

    def unit_by_key(self, key: str) -> dict[str, Any] | None:
        return self.units_by_key.get(key)

    def graph_summary(self) -> dict[str, Any]:
        kinds = Counter(edge["kind"] for edges in self.edges.values() for edge in edges)
        return {
            "file_count": len(self.files),
            "unit_count": len(self.units),
            "edge_count": sum(kinds.values()),
            "edge_kinds": dict(sorted(kinds.items())),
            "codegraph_available": bool(self.codegraph.get("available")),
            "codegraph_path_count": len(self.codegraph_paths),
            "dynamic_codegraph_query_count": len(self.dynamic_codegraph_evidence),
        }

    def export(self) -> dict[str, Any]:
        """Return a content-free graph artifact suitable for traces and analysis."""
        nodes = [
            {
                "id": self.unit_key(unit),
                "repo": unit.get("repo", ""),
                "file_path": unit.get("path", ""),
                "class_name": unit.get("class_name", ""),
                "name": unit.get("name", ""),
                "qualified_name": unit.get("qualified_name", ""),
                "kind": unit.get("kind", ""),
                "start_line": unit.get("start_line"),
                "end_line": unit.get("end_line"),
                "signature": unit.get("signature", ""),
            }
            for unit in self.units
        ]
        edges = [
            {"source": source, "target": edge["target"], "kind": edge["kind"]}
            for source, values in self.edges.items()
            for edge in values
        ]
        return {
            "summary": self.graph_summary(),
            "nodes": nodes,
            "edges": edges,
            "codegraph_paths": sorted(self.codegraph_paths),
            "dynamic_codegraph_evidence": self.dynamic_codegraph_evidence,
        }

    def requirement_file_score(self, relative: str, terms: list[str]) -> int:
        record = self.files.get(relative, {})
        haystack = " ".join(
            [relative, str(record.get("preview", "")), " ".join(str(item) for item in record.get("terms", []))]
        ).lower()
        score = 30 if relative in self.codegraph_paths else 0
        for term in terms:
            variants = stable_unique([term.lower(), *path_tokens(term.lower())])
            if any(variant and variant in relative.lower() for variant in variants):
                score += 7
            if any(variant and variant in haystack for variant in variants):
                score += 2
        if record.get("kind") == "source":
            score += 2
        if record.get("is_test"):
            score -= 1
        return score
