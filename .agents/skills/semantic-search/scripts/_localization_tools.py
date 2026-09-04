# Copyright (c) 2026 Chrys. All rights reserved.

"""SemLoc-style repository search tools backed by ``LocalizationGraph``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from _common import split_identifier, stable_unique
from _localization_graph import LocalizationGraph, bare_symbol, normalize_relative_path


@dataclass
class ToolResult:
    content: str
    locations: list[dict[str, Any]] = field(default_factory=list)
    finished: bool = False


class LocalizationTools:
    """The five search tools used by the LLM localization loop."""

    definitions: ClassVar[list[dict[str, Any]]] = [
        {
            "type": "function",
            "function": {
                "name": "find_file",
                "description": (
                    "Search files by exact name or glob. Returns file paths and a class/function skeleton."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_name": {"type": "string"},
                        "dir_path": {"type": "string", "default": "."},
                    },
                    "required": ["file_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_code_definition",
                "description": (
                    "Search class/function definitions by exact name, regex, then fuzzy similarity. "
                    "Returns code preview, child units, and graph relationships."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "definition_name": {"type": "string"},
                        "file_path": {"type": "string"},
                    },
                    "required": ["definition_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_code_content",
                "description": (
                    "Search identifiers or exact snippets. Identifier searches include camelCase/snake_case variants. "
                    "Returns matched lines and their containing code units."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "file_path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                    },
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_child_unit",
                "description": (
                    "Look up a code unit by exact file path and definition name, then return its preview "
                    "and graph links."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "definition_name": {"type": "string"},
                        "file_path": {"type": "string"},
                    },
                    "required": ["definition_name", "file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish_search",
                "description": (
                    "Finish repository exploration and proceed to one final structured localization response."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    def __init__(self, graph: LocalizationGraph, *, max_results: int = 20):
        self.graph = graph
        self.max_results = max(max_results, 1)

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        aliases = {
            "find_file": {"name": "file_name", "pattern": "file_name", "query": "file_name", "file_path": "file_name"},
            "find_code_definition": {"name": "definition_name", "query": "definition_name"},
            "find_code_content": {
                "query": "content",
                "search": "content",
                "code": "content",
                "text": "content",
                "keyword": "content",
                "pattern": "content",
            },
            "find_child_unit": {"name": "definition_name"},
        }
        normalized = {aliases.get(name, {}).get(key, key): value for key, value in arguments.items()}
        try:
            if name == "find_file":
                return self.find_file(str(normalized.get("file_name", "")), str(normalized.get("dir_path", ".")))
            if name == "find_code_definition":
                return self.find_code_definition(
                    str(normalized.get("definition_name", "")), self._optional_string(normalized.get("file_path"))
                )
            if name == "find_code_content":
                return self.find_code_content(
                    str(normalized.get("content", "")),
                    self._optional_string(normalized.get("file_path")),
                    self._optional_int(normalized.get("start_line")),
                    self._optional_int(normalized.get("end_line")),
                )
            if name == "find_child_unit":
                return self.find_child_unit(
                    str(normalized.get("definition_name", "")), str(normalized.get("file_path", ""))
                )
            if name == "finish_search":
                return ToolResult(
                    "Search phase complete. Return the final ranked locations as the requested JSON object.",
                    finished=True,
                )
            return ToolResult(f"Unknown localization tool: {name}")
        except (TypeError, ValueError, re.error) as err:
            return ToolResult(f"Tool input error for {name}: {err}")

    @staticmethod
    def _optional_string(value: Any) -> str | None:
        text = str(value or "").strip()
        return text or None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        return int(value)

    def find_file(self, file_name: str, dir_path: str = ".") -> ToolResult:
        if not file_name.strip():
            return ToolResult("find_file requires file_name")
        matches = self.graph.file_matches(file_name.strip(), dir_path)
        if not matches:
            return ToolResult(f'No files found matching "{file_name}" under "{dir_path}"')
        sections: list[str] = []
        locations: list[dict[str, Any]] = []
        for relative in matches[: self.max_results]:
            units = self.graph.units_by_file.get(relative, [])
            lines = [f"File path: {relative}", "File skeleton:"]
            if not units:
                lines.append("  (no definitions found)")
                locations.append(
                    {
                        "file_path": relative,
                        "class_name": "",
                        "function_name": "",
                        "start_line": 1,
                        "end_line": 1,
                        "reason": "file-name match",
                    }
                )
            for unit in units[:40]:
                label = self._unit_label(unit)
                lines.append(
                    f"  {unit.get('kind', 'unit').title()}: {label} (lines {unit['start_line']}-{unit['end_line']})"
                )
                locations.append(self._candidate(unit, "file-name match"))
            sections.append("\n".join(lines))
        if len(matches) > self.max_results:
            sections.append(f"(showing top {self.max_results} of {len(matches)} files)")
        return ToolResult("\n\n".join(sections), stable_unique(locations))

    def find_code_definition(self, definition_name: str, file_path: str | None = None) -> ToolResult:
        query = definition_name.strip()
        if not query:
            return ToolResult("find_code_definition requires definition_name")
        units = [unit for unit in self.graph.units if self.graph.file_pattern_matches(unit["path"], file_path)]
        if not units:
            return ToolResult(f"No code definitions available for file filter {file_path!r}")
        exact = [unit for unit in units if query in {unit.get("name"), unit.get("qualified_name")}]
        match_note = "exact"
        matched = exact
        if not matched and self._looks_regex(query):
            compiled = re.compile(query, flags=re.IGNORECASE)
            matched = [unit for unit in units if compiled.search(str(unit.get("qualified_name", "")))]
            match_note = "regex"
        if not matched:
            scored = sorted(
                ((self._fuzzy_score(query, str(unit.get("qualified_name", ""))), unit) for unit in units),
                key=lambda item: (-item[0], item[1].get("path", ""), item[1].get("start_line", 0)),
            )
            matched = [unit for score, unit in scored if score >= 0.5][:5]
            match_note = "fuzzy"
        if not matched:
            suffix = f" in files matching {file_path!r}" if file_path else ""
            return ToolResult(f"No code definitions found matching {query!r}{suffix}")
        rendered = [
            self._format_unit(unit, prefix=f"Match: {match_note}", dynamic_graph=position < 3)
            for position, unit in enumerate(matched[: self.max_results])
        ]
        locations = [self._candidate(unit, f"{match_note} definition match for {query}") for unit in matched]
        return ToolResult("\n\n".join(rendered), stable_unique(locations))

    def find_code_content(
        self,
        content: str,
        file_path: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> ToolResult:
        query = content.strip()
        if not query:
            return ToolResult("find_code_content requires content")
        terms = self._identifier_variants(query) if self._is_identifier(query) else [query]
        results: list[tuple[int, str, int, str, dict[str, Any] | None]] = []
        for relative, record in self.graph.files.items():
            if not self.graph.file_pattern_matches(relative, file_path):
                continue
            if record.get("kind") not in {"source", "test", "config", "build", "generated"}:
                continue
            if int(record.get("size") or 0) > 2_000_000:
                continue
            text = self.graph.read_file(relative)
            for line_number, line in enumerate(text.splitlines(), start=1):
                if start_line is not None and line_number < start_line:
                    continue
                if end_line is not None and line_number > end_line:
                    continue
                matched = [term for term in terms if term in line]
                if not matched:
                    continue
                priority = 0 if query in line else 1
                results.append(
                    (
                        priority,
                        relative,
                        line_number,
                        line.strip()[:400],
                        self.graph.containing_unit(relative, line_number),
                    )
                )
        if not results:
            suffix = f" in files matching {file_path!r}" if file_path else ""
            return ToolResult(f"No matches found for {query!r}{suffix}")
        results.sort(key=lambda item: (item[0], item[1], item[2]))
        sections = []
        locations = []
        for position, (_priority, relative, line_number, line, unit) in enumerate(results[: self.max_results]):
            if unit:
                sections.append(
                    self._format_unit(
                        unit,
                        prefix=f"Matched line {line_number}: {line}",
                        preview_line=line_number,
                        dynamic_graph=position < 3,
                    )
                )
                locations.append(self._candidate(unit, f"content match for {query} at line {line_number}"))
            else:
                sections.append(f"File path: {relative}\nMatched line {line_number}: {line}")
                locations.append(
                    {
                        "file_path": relative,
                        "class_name": "",
                        "function_name": "",
                        "start_line": line_number,
                        "end_line": line_number,
                        "reason": f"content match for {query}",
                    }
                )
        if len(results) > self.max_results:
            sections.append(f"(showing top {self.max_results} of {len(results)} matches)")
        return ToolResult("\n\n".join(sections), stable_unique(locations))

    def find_child_unit(self, definition_name: str, file_path: str) -> ToolResult:
        if not definition_name.strip() or not file_path.strip():
            return ToolResult("find_child_unit requires definition_name and file_path")
        relative = normalize_relative_path(file_path, self.graph.repo)
        units = self.graph.resolve_unit(relative, definition_name.strip())
        if not units:
            return ToolResult(f"No definition found: {definition_name!r} in {file_path!r}")
        rendered = [self._format_unit(unit) for unit in units[: self.max_results]]
        locations = [self._candidate(unit, "exact child-unit lookup") for unit in units]
        return ToolResult("\n\n".join(rendered), locations)

    def _format_unit(
        self,
        unit: dict[str, Any],
        *,
        prefix: str = "",
        preview_line: int | None = None,
        dynamic_graph: bool = True,
    ) -> str:
        lines = []
        if prefix:
            lines.append(prefix)
        lines.append(f"File path: {unit['path']}")
        if unit.get("class_name"):
            lines.append(f"Class name: {unit['class_name']}")
        if unit.get("kind") == "class":
            lines.append(f"Class name: {unit['name']}")
        else:
            lines.append(f"Function name: {unit['name']}")
        lines.extend([f"Start line: {unit['start_line']}", f"End line: {unit['end_line']}"])
        neighbors = self.graph.neighbors(unit, dynamic=dynamic_graph)
        child_ids = stable_unique(
            f"{item['unit']['path']}:{item['unit']['qualified_name']}"
            for item in neighbors
            if item["kind"] in {"child", "call", "cross_file_call", "async_call", "cross_file_async_call", "callee"}
        )
        lines.append("Child units: " + (", ".join(child_ids[:16]) if child_ids else "None"))
        if neighbors:
            relation_text = stable_unique(
                f"{item['kind']} -> {item['unit']['path']}:{item['unit']['qualified_name']}" for item in neighbors
            )
            lines.append("Graph relationships:")
            lines.extend(f"  {value}" for value in relation_text[:24])
        lines.extend(["Code preview:", self._preview(unit, focus_line=preview_line)])
        return "\n".join(lines)

    @staticmethod
    def _preview(unit: dict[str, Any], *, focus_line: int | None = None) -> str:
        content_lines = str(unit.get("content", "")).splitlines()
        start = int(unit.get("start_line") or 1)
        if not content_lines:
            return f"  {start:>4} | {unit.get('signature', '')}"
        show = set(range(min(5, len(content_lines))))
        if focus_line is not None:
            offset = focus_line - start
            show.update(range(max(offset - 2, 0), min(offset + 3, len(content_lines))))
        output = []
        previous = -1
        for offset in sorted(show):
            if offset > previous + 1:
                output.append("  ...")
            output.append(f"  {start + offset:>4} | {content_lines[offset]}")
            previous = offset
        if previous < len(content_lines) - 1:
            output.append("  ...")
        return "\n".join(output)

    @staticmethod
    def _unit_label(unit: dict[str, Any]) -> str:
        return str(unit.get("qualified_name") or unit.get("name") or "<anonymous>")

    @staticmethod
    def _candidate(unit: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "file_path": unit.get("path", ""),
            "class_name": unit.get("class_name", ""),
            "function_name": "" if unit.get("kind") == "class" else unit.get("name", ""),
            "start_line": unit.get("start_line"),
            "end_line": unit.get("end_line"),
            "reason": reason,
        }

    @staticmethod
    def _looks_regex(value: str) -> bool:
        return bool(set(value) & set(r"[](){}*+?|^$\."))

    @staticmethod
    def _ngrams(value: str, size: int = 2) -> set[str]:
        lowered = value.lower()
        return {lowered[index : index + size] for index in range(max(len(lowered) - size + 1, 0))}

    @classmethod
    def _fuzzy_score(cls, left: str, right: str) -> float:
        left_lower = left.lower()
        right_lower = right.lower()
        if left_lower == right_lower:
            return 1.0
        left_grams = cls._ngrams(left_lower)
        right_grams = cls._ngrams(right_lower)
        ngram = len(left_grams & right_grams) / len(left_grams | right_grams) if left_grams and right_grams else 0.0
        left_tokens = set(split_identifier(left_lower))
        right_tokens = set(split_identifier(right_lower))
        token = (
            len(left_tokens & right_tokens) / len(left_tokens | right_tokens) if left_tokens and right_tokens else 0.0
        )
        prefix = 0
        for a, b in zip(left_lower, right_lower, strict=False):
            if a != b or prefix == 4:
                break
            prefix += 1
        prefix_score = prefix / 4
        containment = 1.0 if left_lower in right_lower or right_lower in left_lower else 0.0
        return 0.45 * ngram + 0.25 * token + 0.15 * prefix_score + 0.15 * containment

    @staticmethod
    def _is_identifier(value: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value))

    @classmethod
    def _identifier_variants(cls, value: str) -> list[str]:
        words = split_identifier(value)
        if not words:
            return [value]
        snake = "_".join(words)
        camel = words[0] + "".join(word[:1].upper() + word[1:] for word in words[1:])
        pascal = "".join(word[:1].upper() + word[1:] for word in words)
        return stable_unique([value, snake, camel, pascal, bare_symbol(value)])
