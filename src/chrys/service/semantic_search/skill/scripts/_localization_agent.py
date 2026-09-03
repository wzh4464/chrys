#!/usr/bin/env python3
"""Deterministic half of SemLoc localization: parsing and normalization.

The DFS/BFS search loop and its model client used to live here. They now run
inside the Chrys process (``service/semantic_search/localization_model.py``)
so localization obeys the same model policy, model lock, and usage accounting
as every other Chrys model call, and this module holds no API key.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _common import now_iso
from _localization_graph import LocalizationGraph, normalize_relative_path

SYSTEM_PROMPT = """You are a specialized repository code-localization agent.

Given an implementation requirement, identify the existing source locations a
developer should inspect or edit. You MUST use the repository tools before
finishing.

Search workflow:
1. Classify the requirement and extract explicit or implicit entry points.
2. Use DFS to follow relevant definitions, child units, callees, and semantic edges.
3. Use BFS to expand to callers, sibling behavior, configuration,
   generated/build surfaces, and validation files when they may require
   coordinated work.
4. Refine the search terms when retrieved code reveals new triggers, state, aliases, or dependency links.
5. Call finish_search only after checking primary logic and plausible propagation paths.

Locations are inspection candidates, not automatic edit mandates. Prefer
functions/methods over entire files. Do not generate a patch and do not access
benchmark answer-side material.
"""

FINAL_PROMPT = """Return the final ranked localization as JSON only. Do not call more tools.

Schema:
{
  "locations": [
    {
      "file_path": "relative/path.py",
      "class_name": "ClassName or empty",
      "function_name": "function_or_method or empty",
      "start_line": 1,
      "end_line": 10,
      "role": "primary | propagation | validation",
      "reason": "why this location is relevant",
      "confidence": "high | medium | low"
    }
  ]
}

Rank highest-confidence primary edit sites first, then propagation/configuration
sites, then validation locations. Include only repository paths observed through
the tools.
"""


@dataclass
class AgentRunResult:
    locations: list[dict[str, Any]]
    observed_candidates: list[dict[str, Any]]
    tool_call_count: int
    iteration_count: int
    model: str
    finished: bool


class TraceWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def write(self, event: str, **data: Any) -> None:
        payload = {"created_at": now_iso(), "event": event, **data}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def parse_locations(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return []
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    payload: Any = None
    try:
        payload = json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}|\[.*\]", text, flags=re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(0))
            except ValueError:
                payload = None
    if isinstance(payload, dict):
        locations = payload.get("locations") or payload.get("pred_locations") or []
        return [item for item in locations if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return parse_semloc_text(text)


def parse_semloc_text(text: str) -> list[dict[str, Any]]:
    blocks = re.findall(r"<code_location>(.*?)</code_location>", text, flags=re.DOTALL) or [text]
    locations: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    fields = {
        "file path": "file_path",
        "class name": "class_name",
        "function name": "function_name",
        "start line": "start_line",
        "end line": "end_line",
        "reason": "reason",
        "role": "role",
        "confidence": "confidence",
    }
    for block in blocks:
        for raw_line in block.splitlines():
            line = re.sub(r"^[\s\-*\d.)]+", "", raw_line).replace("**", "").strip()
            if ":" not in line:
                continue
            label, raw_value = line.split(":", 1)
            key = fields.get(label.strip().lower())
            if not key:
                continue
            if key == "file_path" and current:
                locations.append(current)
                current = {}
            cleaned = raw_value.strip().strip("`'\"")
            if key in {"start_line", "end_line"}:
                try:
                    current[key] = int(cleaned)
                except ValueError:
                    continue
            else:
                current[key] = cleaned
        if current:
            locations.append(current)
            current = {}
    return locations


def normalize_locations(
    locations: list[dict[str, Any]], graph: LocalizationGraph, *, source: str
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in locations:
        relative = normalize_relative_path(
            str(raw.get("file_path") or raw.get("file") or raw.get("path") or ""), graph.repo
        )
        if relative not in graph.files:
            continue
        class_name = str(raw.get("class_name") or raw.get("class") or "").strip()
        function_name = str(
            raw.get("function_name") or raw.get("function") or raw.get("symbol") or raw.get("name") or ""
        ).strip()
        resolved = graph.resolve_unit(relative, function_name or class_name)
        if function_name and not class_name and "." in function_name and not resolved:
            class_name, function_name = function_name.rsplit(".", 1)
            resolved = graph.resolve_unit(relative, function_name or class_name)
        unit = resolved[0] if resolved else None
        if unit:
            class_name = class_name or str(unit.get("class_name") or "")
            if unit.get("kind") == "class" and not function_name:
                class_name = str(unit.get("name") or class_name)
            elif not function_name:
                function_name = str(unit.get("name") or "")
        start_line = _as_int(raw.get("start_line")) or _as_int((unit or {}).get("start_line"))
        end_line = _as_int(raw.get("end_line")) or _as_int((unit or {}).get("end_line")) or start_line
        key = (relative, class_name, function_name, start_line, end_line)
        if key in seen:
            continue
        seen.add(key)
        rank = len(normalized) + 1
        is_test = bool(graph.files[relative].get("is_test"))
        role = str(raw.get("role") or "").lower()
        if role not in {"primary", "propagation", "validation"}:
            role = "validation" if is_test else ("primary" if rank <= 3 else "propagation")
        confidence = str(raw.get("confidence") or "medium").lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        symbol = f"{class_name}.{function_name}" if class_name and function_name else (function_name or class_name)
        normalized.append(
            {
                "rank": rank,
                "role": role,
                "repo": graph.files[relative].get("repo", ""),
                "file": relative,
                "file_path": relative,
                "symbol": symbol,
                "class_name": class_name,
                "function_name": function_name,
                "start_line": start_line,
                "end_line": end_line,
                "reason": str(raw.get("reason") or "Repository search evidence matched this location."),
                "evidence": {"source": source},
                "confidence": confidence,
                "must_verify": True,
            }
        )
    return normalized


def _as_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except TypeError, ValueError:
        return None
    return parsed if parsed > 0 else None
