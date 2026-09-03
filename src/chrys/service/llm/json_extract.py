# Copyright (c) 2026 Chrys. All rights reserved.

"""Recover a JSON object from a model response that is almost JSON.

Models fence their output, prefix it with prose, and truncate it mid-object.
Every caller that asks for a JSON verdict hits the same three failures, so the
candidate walk and the small-truncation repair live here rather than in each
of them.
"""

from __future__ import annotations

__all__ = [
    "balanced_json_objects",
    "json_object_candidates",
    "json_object_start_indices",
    "repair_json_object_candidate",
    "strip_json_fence",
]


def strip_json_fence(text: str) -> str:
    """Strip a surrounding markdown code fence from *text*, if present."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    lines = [line for line in lines if not line.strip().startswith("```")]
    return "\n".join(lines).strip()


def json_object_start_indices(text: str) -> list[int]:
    """Return indices of JSON object starts outside strings."""
    starts: list[int] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if in_string and char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string and char == "{":
            starts.append(index)
    return starts


def balanced_json_objects(text: str) -> list[str]:
    """Return all balanced JSON objects in *text*, ignoring braces in strings."""
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if escaped:
            escaped = False
            continue
        if in_string and char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def json_object_candidates(text: str) -> list[str]:
    """Return possible JSON object substrings from an LLM response."""
    stripped = strip_json_fence(text)
    candidates: list[str] = []

    def add(candidate: str | None) -> None:
        if candidate is None:
            return
        cleaned = candidate.strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    add(stripped)
    for obj in balanced_json_objects(stripped):
        add(obj)

    starts = json_object_start_indices(stripped)
    start = starts[0] if starts else -1
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        add(stripped[start : end + 1])
    for start in starts:
        add(stripped[start:])
    return candidates


def repair_json_object_candidate(candidate: str) -> str:
    """Repair small JSON object truncations common in LLM output."""
    repaired = candidate.strip()
    if not repaired.startswith("{"):
        return repaired

    in_string = False
    escaped = False
    for char in repaired:
        if escaped:
            escaped = False
            continue
        if in_string and char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
    if in_string:
        repaired += '"'

    stack: list[str] = []
    in_string = False
    escaped = False
    for char in repaired:
        if escaped:
            escaped = False
            continue
        if in_string and char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append(char)
        elif char in "}]":
            if not stack:
                return repaired
            opener = stack[-1]
            if (opener, char) not in {("{", "}"), ("[", "]")}:
                return repaired
            stack.pop()

    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired
