# Copyright (c) 2026 Chrys. All rights reserved.

"""Pure diff computation helpers shared by diff renderers."""

from __future__ import annotations

import difflib
import os
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain

from pygments.lexers import find_lexer_class_for_filename
from textual import highlight
from textual.content import Content, Span

from chrys.app.tui.widgets.diff_view._utils import loop_last
from chrys.app.tui.widgets.diff_view.code import (
    ELLIPSIS_SENTINEL,
    CodeLine,
    DiffAnnotation,
    EllipsisSentinel,
    LineStyle,
)
from chrys.app.tui.widgets.syntax_theme import NoErrorHighlightTheme

type DiffOpcode = tuple[str, int, int, int, int]
type DiffOpcodeGroups = list[list[DiffOpcode]]

_COMMON_LANGUAGE_BY_PATH_KEY = {
    ".arkts": "typescript",
    ".bash": "bash",
    ".bat": "batch",
    ".c": "c",
    ".cjs": "javascript",
    ".cfg": "ini",
    ".cmake": "cmake",
    ".cmd": "batch",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".csproj": "xml",
    ".css": "css",
    ".csv": "text",
    ".csx": "csharp",
    ".cxx": "cpp",
    ".d.ts": "typescript",
    ".dart": "dart",
    ".dockerignore": "text",
    ".editorconfig": "ini",
    ".env": "text",
    ".ets": "typescript",
    ".fish": "fish",
    ".frag": "glsl",
    ".fs": "fsharp",
    ".fsi": "fsharp",
    ".fsproj": "xml",
    ".fsx": "fsharp",
    ".gemspec": "ruby",
    ".gitignore": "text",
    ".gql": "graphql",
    ".go": "go",
    ".gradle": "groovy",
    ".graphql": "graphql",
    ".groovy": "groovy",
    ".h": "c",
    ".hh": "cpp",
    ".htm": "html",
    ".hpp": "cpp",
    ".html": "html",
    ".hxx": "cpp",
    ".ini": "ini",
    ".ipynb": "json",
    ".java": "java",
    ".js": "javascript",
    ".json": "json",
    ".jsonc": "javascript",
    ".jsonl": "json",
    ".jsx": "jsx",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".ksh": "bash",
    ".less": "less",
    ".lua": "lua",
    ".md": "markdown",
    ".mdx": "markdown",
    ".mjs": "javascript",
    ".mm": "objective-c++",
    ".props": "xml",
    ".proto": "protobuf",
    ".ps1": "powershell",
    ".psd1": "powershell",
    ".psm1": "powershell",
    ".php": "php",
    ".py": "python",
    ".pyi": "python",
    ".pyw": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".rst": "restructuredtext",
    ".sass": "sass",
    ".scala": "scala",
    ".scss": "scss",
    ".sh": "bash",
    ".sln": "text",
    ".sql": "sql",
    ".svg": "xml",
    ".svelte": "html",
    ".swift": "swift",
    ".targets": "xml",
    ".tcss": "scss",
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".thrift": "thrift",
    ".tsv": "text",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".txt": "text",
    ".vb": "vb.net",
    ".vbproj": "xml",
    ".vcxproj": "xml",
    ".vert": "glsl",
    ".vue": "vue",
    ".xaml": "xml",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".zsh": "bash",
    "cargo.toml": "toml",
    "cmakelists.txt": "cmake",
    "dockerfile": "docker",
    "gemfile": "ruby",
    "go.mod": "text",
    "go.sum": "text",
    "makefile": "make",
    "pipfile": "toml",
    "rakefile": "ruby",
    "requirements.txt": "text",
}


def _language_cache_key(path: str) -> str:
    """Return a stable cache key for filename-based lexer detection."""
    filename = os.path.basename(path).lower()
    if not filename:
        return ""
    if filename in _COMMON_LANGUAGE_BY_PATH_KEY:
        return filename
    if filename.startswith("dockerfile."):
        return "dockerfile"
    if filename.startswith("makefile."):
        return "makefile"
    if filename.endswith(".d.ts"):
        return ".d.ts"
    if filename.endswith(".gradle.kts"):
        return ".kts"
    _root, ext = os.path.splitext(filename)
    return ext or filename


@lru_cache(maxsize=256)
def _language_from_path_key(key: str) -> str | None:
    """Resolve a Pygments language alias from a path-derived cache key."""
    if not key:
        return None
    if language := _COMMON_LANGUAGE_BY_PATH_KEY.get(key):
        return language
    sample = f"file{key}" if key.startswith(".") else key
    lexer_class = find_lexer_class_for_filename(sample)
    if lexer_class is None:
        return None
    if lexer_class.aliases:
        return lexer_class.aliases[0]
    return lexer_class.name


def _language_from_path(path: str) -> str | None:
    """Resolve a language from the file path without content guessing."""
    return _language_from_path_key(_language_cache_key(path))


def _guess_diff_language(code: str, path: str) -> str:
    """Fast path language detection for diff rendering.

    Diff inputs almost always come from concrete file paths.  Pygments'
    content-based ``guess_lexer_for_filename`` is much more expensive than
    filename matching and is usually unnecessary here, so prefer a cached
    filename/extension lookup and fall back to Textual's heuristic only for
    unknown paths.
    """
    language = _language_from_path(path)
    if language is not None:
        return language
    return highlight.guess_language(code, path)


def _highlight_code_lines(code: str, path: str, language: str) -> list[Content]:
    """Return syntax-highlighted content split into source lines."""
    text_lines = code.splitlines()
    if not text_lines:
        return []
    highlighted = highlight.highlight("\n".join(text_lines), language=language, path=path, theme=NoErrorHighlightTheme)
    return highlighted.split("\n")


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge line ranges for batched sparse highlighting."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if start >= end:
            continue
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def _grouped_line_ranges(grouped_opcodes: DiffOpcodeGroups) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Return before/after line ranges visible in a grouped unified diff."""
    ranges_a: list[tuple[int, int]] = []
    ranges_b: list[tuple[int, int]] = []
    for group in grouped_opcodes:
        starts_a = [i1 for _tag, i1, i2, _j1, _j2 in group if i1 < i2]
        ends_a = [i2 for _tag, i1, i2, _j1, _j2 in group if i1 < i2]
        starts_b = [j1 for _tag, _i1, _i2, j1, j2 in group if j1 < j2]
        ends_b = [j2 for _tag, _i1, _i2, j1, j2 in group if j1 < j2]
        if starts_a and ends_a:
            ranges_a.append((min(starts_a), max(ends_a)))
        if starts_b and ends_b:
            ranges_b.append((min(starts_b), max(ends_b)))
    return _merge_ranges(ranges_a), _merge_ranges(ranges_b)


def _state_preserving_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Expand sparse ranges enough for lexers with multi-line state."""
    merged = _merge_ranges(ranges)
    if not merged:
        return []
    # Keep the range pipeline explicit, but highlight one prefix for now so
    # lexers see the state that earlier offscreen lines establish.
    return [(0, max(end for _start, end in merged))]


def _highlight_code_line_ranges(
    code: str,
    path: str,
    language: str,
    ranges: list[tuple[int, int]],
) -> list[Content]:
    """Highlight sparse diff ranges while preserving original indexes."""
    text_lines = code.splitlines()
    if not text_lines:
        return []
    highlighted_lines = [Content(line) for line in text_lines]
    for start, end in _state_preserving_ranges(ranges):
        bounded_start = max(0, min(start, len(text_lines)))
        bounded_end = max(bounded_start, min(end, len(text_lines)))
        if bounded_start >= bounded_end:
            continue
        range_lines = _highlight_code_lines(
            "\n".join(text_lines[bounded_start:bounded_end]),
            path,
            language,
        )
        count = bounded_end - bounded_start
        if len(range_lines) < count:
            missing_start = bounded_start + len(range_lines)
            range_lines.extend(Content(line) for line in text_lines[missing_start:bounded_end])
        highlighted_lines[bounded_start:bounded_end] = range_lines[:count]
    return highlighted_lines


@dataclass
class UnifiedFlatDiff:
    """Flattened unified-diff row data."""

    line_numbers_a: list[int | EllipsisSentinel | None]
    line_numbers_b: list[int | EllipsisSentinel | None]
    annotations: list[DiffAnnotation | EllipsisSentinel]
    code_lines: list[CodeLine]
    line_styles: list[LineStyle]
    line_number_width: int
    max_code_width: int


def compute_grouped_opcodes(code_before: str, code_after: str) -> DiffOpcodeGroups:
    """Return grouped line opcodes for two text snapshots."""
    text_lines_a = code_before.splitlines()
    text_lines_b = code_after.splitlines()
    sequence_matcher = difflib.SequenceMatcher(
        lambda character: character in {" ", "\t"},
        text_lines_a,
        text_lines_b,
        autojunk=True,
    )
    return list(sequence_matcher.get_grouped_opcodes())


def highlight_diff_lines(lines_a: list[Content], lines_b: list[Content]) -> tuple[list[Content], list[Content]]:
    """Apply character-level diff highlighting for replace operations."""
    code_a = Content("\n").join(content for content in lines_a)
    code_b = Content("\n").join(content for content in lines_b)
    sequence_matcher = difflib.SequenceMatcher(
        lambda character: character in {" ", "\t"},
        code_a.plain,
        code_b.plain,
        autojunk=True,
    )
    spans_a: list[Span] = []
    spans_b: list[Span] = []
    for tag, i1, i2, j1, j2 in sequence_matcher.get_opcodes():
        if tag in {"delete", "replace"}:
            spans_a.append(Span(i1, i2, "on $error 40%"))
        if tag in {"insert", "replace"}:
            spans_b.append(Span(j1, j2, "on $success 40%"))
    diffed_lines_a = code_a.add_spans(spans_a).split("\n")
    diffed_lines_b = code_b.add_spans(spans_b).split("\n")
    return diffed_lines_a, diffed_lines_b


def compute_highlighted_code_lines(
    code_before: str,
    code_after: str,
    path1: str,
    path2: str,
    grouped_opcodes: DiffOpcodeGroups,
    *,
    grouped_only: bool = False,
) -> tuple[list[Content], list[Content]]:
    """Return syntax-highlighted before/after lines with intraline diff spans."""
    path_language1 = _language_from_path(path1)
    language1 = path_language1 or highlight.guess_language(code_before, path1)
    if path1 == path2 and path_language1 is not None:
        language2 = path_language1
    else:
        language2 = _guess_diff_language(code_after, path2)
    if grouped_only:
        ranges_a, ranges_b = _grouped_line_ranges(grouped_opcodes)
        lines_a = _highlight_code_line_ranges(code_before, path1, language1, ranges_a)
        lines_b = _highlight_code_line_ranges(code_after, path2, language2, ranges_b)
    else:
        lines_a = _highlight_code_lines(code_before, path1, language1)
        lines_b = _highlight_code_lines(code_after, path2, language2)

    if code_before:
        for group in grouped_opcodes:
            for tag, i1, i2, j1, j2 in group:
                if tag == "replace" and (j2 - j1) == (i2 - i1):
                    diff_lines_a, diff_lines_b = highlight_diff_lines(lines_a[i1:i2], lines_b[j1:j2])
                    lines_a[i1:i2] = diff_lines_a
                    lines_b[j1:j2] = diff_lines_b

    return lines_a, lines_b


def flatten_unified_diff(
    grouped_opcodes: DiffOpcodeGroups,
    lines_a: list[Content],
    lines_b: list[Content],
    line_styles: dict[str, str],
) -> UnifiedFlatDiff:
    """Flatten grouped opcodes into unified-diff rows."""
    flat_numbers_a: list[int | EllipsisSentinel | None] = []
    flat_numbers_b: list[int | EllipsisSentinel | None] = []
    flat_annotations: list[DiffAnnotation | EllipsisSentinel] = []
    flat_code: list[CodeLine] = []
    flat_styles: list[LineStyle] = []
    max_code_width = 1

    for last, group in loop_last(grouped_opcodes):
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                for line_offset, line in enumerate(lines_a[i1:i2], 1):
                    flat_annotations.append(" ")
                    flat_numbers_a.append(i1 + line_offset)
                    flat_numbers_b.append(j1 + line_offset)
                    flat_code.append(line)
                    flat_styles.append(line_styles[" "])
                    max_code_width = max(max_code_width, line.cell_length)
                continue
            if tag in {"delete", "replace"}:
                for line_offset, line in enumerate(lines_a[i1:i2], 1):
                    flat_annotations.append("-")
                    flat_numbers_a.append(i1 + line_offset)
                    flat_numbers_b.append(None)
                    flat_code.append(line)
                    flat_styles.append(line_styles["-"])
                    max_code_width = max(max_code_width, line.cell_length)
            if tag in {"insert", "replace"}:
                for line_offset, line in enumerate(lines_b[j1:j2], 1):
                    flat_annotations.append("+")
                    flat_numbers_a.append(None)
                    flat_numbers_b.append(j1 + line_offset)
                    flat_code.append(line)
                    flat_styles.append(line_styles["+"])
                    max_code_width = max(max_code_width, line.cell_length)

        if not last:
            flat_numbers_a.append(ELLIPSIS_SENTINEL)
            flat_numbers_b.append(ELLIPSIS_SENTINEL)
            flat_annotations.append(ELLIPSIS_SENTINEL)
            flat_code.append(ELLIPSIS_SENTINEL)
            flat_styles.append(ELLIPSIS_SENTINEL)

    line_number_width = max(
        (len(str(n)) for n in chain(flat_numbers_a, flat_numbers_b) if isinstance(n, int)),
        default=1,
    )
    return UnifiedFlatDiff(
        line_numbers_a=flat_numbers_a,
        line_numbers_b=flat_numbers_b,
        annotations=flat_annotations,
        code_lines=flat_code,
        line_styles=flat_styles,
        line_number_width=line_number_width,
        max_code_width=max_code_width,
    )
