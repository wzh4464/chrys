#!/usr/bin/env python3
"""Shared helpers for semantic-search requirement augmentation scripts."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORMAT_INDEX = "semantic-search-light-index"
FORMAT_GLOBAL = "semantic-search-global-perception"
FORMAT_CODEGRAPH = "semantic-search-codegraph-perception"
FORMAT_REPOSITORY = "semantic-search-repository-perception"
FORMAT_CODE_PERCEPTION = "semantic-search-code-perception"
FORMAT_LOCALIZATION = "semantic-search-code-localization"
FORMAT_LOCALIZATION_GRAPH = "semantic-search-localization-graph"
FORMAT_FACTS = "semantic-search-code-facts"
FORMAT_ROUTES = "semantic-search-augmentation-routes"

IGNORE_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".codegraph",
    ".semantic-search",
    ".tox",
    ".venv",
    "build",
    "dist",
    "node_modules",
    "__pycache__",
    "target",
}

BENCHMARK_ANSWER_DIRS = {
    "dockers",
    "patches",
    "requirement_pr_pairs",
}

IGNORE_SUFFIXES = {".diff", ".patch", ".pyc", ".pyo", ".class", ".jar", ".zip", ".tar", ".gz"}

SOURCE_SUFFIXES = {
    ".py": "python",
    ".java": "java",
    ".scala": "scala",
    ".sc": "scala",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".rs": "rust",
    ".asdl": "grammar",
    ".g4": "grammar",
    ".gram": "grammar",
    ".l": "grammar",
    ".peg": "grammar",
    ".proto": "schema",
    ".thrift": "schema",
    ".y": "grammar",
}

DOC_SUFFIXES = {".md", ".rst", ".txt"}
CONFIG_SUFFIXES = {".cfg", ".conf", ".gradle", ".ini", ".properties", ".toml", ".yaml", ".yml", ".xml"}
CONFIG_NAMES = {
    "Cargo.toml",
    "CMakeLists.txt",
    "Makefile",
    "Makefile.pre.in",
    "build.gradle",
    "build.gradle.kts",
    "gradle.properties",
    "meson.build",
    "pom.xml",
    "pyproject.toml",
    "settings.gradle",
    "settings.gradle.kts",
    "setup.cfg",
    "setup.py",
    "tox.ini",
}

BUILD_DIR_NAMES = {"buildSrc", "cmake", "gradle", "m4", "PCbuild", "Tools"}
TEST_PARTS = {"test", "tests", "testing", "src/test"}

STOP_WORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "but",
    "can",
    "cannot",
    "class",
    "code",
    "does",
    "for",
    "from",
    "has",
    "have",
    "into",
    "its",
    "may",
    "must",
    "new",
    "not",
    "one",
    "only",
    "other",
    "should",
    "source",
    "such",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "this",
    "those",
    "use",
    "used",
    "using",
    "when",
    "where",
    "which",
    "will",
    "with",
    "would",
}


class ScriptError(RuntimeError):
    """User-facing script failure."""


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def resolve_path(value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def ensure_allowed_path(
    path: Path,
    *,
    allowed_roots: Iterable[Path] = (),
    allowed_files: Iterable[Path] = (),
    purpose: str,
) -> Path:
    resolved = resolve_path(path)
    roots = [resolve_path(root) for root in allowed_roots]
    files = [resolve_path(item) for item in allowed_files]
    if any(resolved == item for item in files) or any(is_relative_to(resolved, root) for root in roots):
        return resolved
    allowed = [str(root) for root in roots] + [str(item) for item in files]
    raise ScriptError(f"{purpose} path is outside the allowed runtime inputs: {resolved} (allowed: {allowed})")


def reject_benchmark_answer_path(path: Path, *, purpose: str) -> None:
    resolved = resolve_path(path)
    parts = set(resolved.parts)
    blocked = parts & BENCHMARK_ANSWER_DIRS
    if blocked:
        raise ScriptError(f"{purpose} path appears to reference benchmark answer-side material: {resolved}")


def read_text(path: Path, max_chars: int | None = None) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1", errors="replace")
    except OSError as err:
        raise ScriptError(f"failed to read {path}: {err}") from err
    if max_chars is not None:
        return text[:max_chars]
    return text


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise ScriptError(f"failed to load JSON {path}: {err}") from err


def sha1_path(path: Path) -> str:
    digest = hashlib.sha1()  # noqa: S324 — cache key, not a security primitive
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as err:
        raise ScriptError(f"failed to hash {path}: {err}") from err
    return digest.hexdigest()


def stable_unique(values: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        if value is None or value == "":
            continue
        key = json.dumps(value, sort_keys=True, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def tokenize(text: str, *, min_len: int = 3) -> list[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_:.+-]*", text)
    cleaned: list[str] = []
    for token in tokens:
        value = token.strip("._-:+").lower()
        if len(value) < min_len or value in STOP_WORDS or len(value) > 80:
            continue
        cleaned.append(value)
        for piece in re.split(r"[_:.\-+]+", value):
            if len(piece) >= min_len and piece not in STOP_WORDS:
                cleaned.append(piece)
    return stable_unique(cleaned)


def split_identifier(value: str) -> list[str]:
    raw = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return tokenize(raw)


def path_tokens(path: str) -> list[str]:
    tokens: list[str] = []
    for part in Path(path).parts:
        tokens.extend(tokenize(part))
        tokens.extend(split_identifier(part))
    return stable_unique(tokens)


def classify_file(path: Path, relative: str) -> dict[str, Any]:
    suffix = path.suffix.lower()
    name = path.name
    parts = tuple(Path(relative).parts)
    lower_parts = {part.lower() for part in parts}
    language = SOURCE_SUFFIXES.get(suffix, "")
    is_test = is_test_path(relative)
    is_generated = looks_generated(path)
    kind = "source" if language else "artifact"
    if is_test:
        kind = "test"
    elif name in CONFIG_NAMES or suffix in CONFIG_SUFFIXES:
        kind = "build" if name in {"Makefile", "CMakeLists.txt", "pom.xml"} or suffix == ".gradle" else "config"
    elif suffix in DOC_SUFFIXES or "doc" in lower_parts or "docs" in lower_parts:
        kind = "docs"
    elif parts and (set(parts) & BUILD_DIR_NAMES or lower_parts & {item.lower() for item in BUILD_DIR_NAMES}):
        kind = "build"
    elif is_generated:
        kind = "generated" if language else "artifact"
    return {
        "kind": kind,
        "language": language,
        "is_test": is_test,
        "is_generated": is_generated,
        "top_level": parts[0] if parts else "",
    }


def is_test_path(relative: str) -> bool:
    lowered = relative.lower()
    name = Path(relative).name.lower()
    parts = {part.lower() for part in Path(relative).parts}
    return (
        bool(parts & TEST_PARTS)
        or "/test" in lowered
        or name.startswith("test_")
        or name.endswith(("_test.py", "_test.rs", "test.java", "tests.java", "test.scala"))
    )


def looks_generated(path: Path) -> bool:
    lowered = path.as_posix().lower()
    if any(marker in lowered for marker in ("generated", "autogen", "auto_generated")):
        return True
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:6000].lower()
    except OSError:
        return False
    return any(
        phrase in head
        for phrase in (
            "automatically generated",
            "auto-generated",
            "autogenerated",
            "generated by",
            "do not edit",
            "do not modify",
        )
    )


def should_skip_dir(name: str) -> bool:
    return name in IGNORE_DIRS or name in BENCHMARK_ANSWER_DIRS


def should_skip_file(path: Path) -> bool:
    return path.suffix.lower() in IGNORE_SUFFIXES


def append_trace(event: str, payload: dict[str, Any]) -> None:
    trace_log = os.environ.get("SEMANTIC_SEARCH_TRACE_LOG", "").strip()
    if not trace_log:
        return
    try:
        path = resolve_path(trace_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": now_iso(), "event": event, **payload}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        return


def markdown_escape_line(value: str) -> str:
    return value.replace("\n", " ").strip()


def bullet_lines(items: Iterable[str], *, empty: str = "(none)") -> list[str]:
    values = [markdown_escape_line(item) for item in items if markdown_escape_line(item)]
    if not values:
        return [empty]
    return [f"- {item}" for item in values]
