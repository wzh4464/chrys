# Copyright (c) 2026 Chrys. All rights reserved.

"""Code search tools — grep and glob, powered by ripgrep.

Uses the ``rg`` binary for fast, correct file searching with full glob and
regex support.  The binary is bundled as package data in
``chrys/foundation/vendor/ripgrep/`` (downloaded at build time via
``scripts/fetch_rg.sh``).  A system-installed ``rg`` on PATH is also accepted
as a fallback.

When the search pattern contains non-ASCII characters, multiple encoding
passes (UTF-8, GBK, and the system ANSI codepage) are run automatically
to catch files in legacy encodings.
"""

from __future__ import annotations

import asyncio
import json
import locale
import os
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import TYPE_CHECKING, Annotated, Any, cast

from chrys.foundation.platform import get_platform
from chrys.foundation.platform.paths import resolve_workspace_path
from chrys.foundation.platform.process import decode_subprocess_output, managed_subprocess
from chrys.foundation.vendor import find_rg
from chrys.service.tools.kinds import KIND_SEARCH, tool
from chrys.service.tools.result_metadata import record_process_result, record_process_timeout, tool_error

if TYPE_CHECKING:
    from chrys.foundation.models.session_env import SessionEnvironment

_MAX_RESULTS = 50
_MAX_RESULTS_HARD_LIMIT = 100
_TIMEOUT = 30
_MAX_LINE_DISPLAY_CHARS = 2048


@dataclass(slots=True)
class GrepEntry:
    """A single line from ripgrep output (match or context)."""

    rel_path: str
    line_num: int
    marker: str  # ">" for match, " " for context
    text: str


@dataclass(slots=True)
class LongLine:
    """A line that was truncated for exceeding ``_MAX_LINE_DISPLAY_CHARS``."""

    rel_path: str
    line_num: int
    actual_length: int


@dataclass(slots=True)
class GrepParseResult:
    """Parsed output from a single ``rg --json`` invocation."""

    entries: list[GrepEntry] = field(default_factory=list)
    match_count: int = 0
    long_lines: list[LongLine] = field(default_factory=list)


_PLATFORM = get_platform()

# Map Python codec names → WHATWG encoding labels accepted by ``rg -E``.
# Only entries where the names differ need to be listed.
_CODEC_TO_WHATWG: dict[str, str] = {
    "cp932": "shift_jis",
    "cp949": "euc-kr",
    "cp950": "big5",
    "iso8859-1": "windows-1252",
    "latin-1": "windows-1252",
    "euc_kr": "euc-kr",
    "euc_jp": "euc-jp",
    "shift_jis": "shift_jis",
    "gb2312": "gbk",
    "gb18030": "gb18030",
}


def _find_rg() -> str:
    """Locate the ``rg`` binary: bundled vendor copy → PATH → error."""
    path = find_rg()
    if path:
        return path
    raise FileNotFoundError(
        "ripgrep (rg) not found — run scripts/fetch_rg.sh to bundle it, or install rg on your system PATH"
    )


def _to_posix(path: str) -> str:
    """Normalize a path to forward slashes for consistent output across platforms.

    ``PurePath.as_posix()`` is platform-aware: on Windows it converts
    ``\\`` separators; on POSIX it preserves any literal backslash a
    filename may legitimately contain.
    """
    return PurePath(path).as_posix()


def _codec_to_whatwg(codec_name: str) -> str:
    """Convert a Python codec name to a WHATWG encoding label for ``rg -E``."""
    normalized = codec_name.lower().replace("-", "_").replace(" ", "_")
    if normalized in _CODEC_TO_WHATWG:
        return _CODEC_TO_WHATWG[normalized]
    # Many names work directly (utf-8, gbk, big5, etc.)
    return codec_name.lower().replace("_", "-")


def _get_system_ansi_codepage() -> str | None:
    """Detect the system ANSI codepage and return its WHATWG label.

    Returns ``None`` if detection fails or the codepage is already UTF-8.
    """
    try:
        if _PLATFORM.is_windows:
            import ctypes

            ctypes_module = cast(Any, ctypes)
            acp = ctypes_module.windll.kernel32.GetACP()
            codec_name = f"cp{acp}"
        else:
            codec_name = locale.getpreferredencoding(False)

        import codecs

        info = codecs.lookup(codec_name)
        normalized = info.name.lower().replace("-", "_")

        # Skip if it's already UTF-8 or GBK (always included)
        if normalized in ("utf_8", "gbk", "cp936", "gb2312"):
            return None

        return _codec_to_whatwg(info.name)
    except Exception:
        return None


def _get_encodings() -> list[str]:
    """Return the list of WHATWG encoding labels to search with.

    Always includes UTF-8 and GBK.  Adds the system ANSI codepage if it
    differs from both.
    """
    encodings = ["utf-8", "gbk"]
    ansi = _get_system_ansi_codepage()
    if ansi and ansi not in encodings:
        encodings.append(ansi)
    return encodings


def _is_ascii_only(text: str) -> bool:
    """Return ``True`` if *text* contains only ASCII characters."""
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


async def _run_rg(args: list[str], *, timeout: int = _TIMEOUT) -> tuple[str, str, int]:
    """Run ``rg`` asynchronously and return ``(stdout, stderr, returncode)``."""
    rg = _find_rg()
    async with managed_subprocess(
        rg,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    ) as proc:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            decode_subprocess_output(stdout),
            decode_subprocess_output(stderr),
            proc.returncode or 0,
        )


# ---------------------------------------------------------------------------
# grep
# ---------------------------------------------------------------------------


def _parse_grep_jsonl(stdout: str, root: str, max_results: int) -> GrepParseResult:
    """Parse ``rg --json`` output into structured results."""
    result = GrepParseResult()

    for line in stdout.splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        entry_type = entry.get("type")
        if entry_type not in ("match", "context"):
            continue

        data = entry.get("data", {})
        file_path = data.get("path", {}).get("text", "")
        line_num = data.get("line_number", 0)
        line_text = data.get("lines", {}).get("text", "").rstrip("\r\n")
        rel_path = _to_posix(os.path.relpath(file_path, root)) if file_path else ""

        if len(line_text) > _MAX_LINE_DISPLAY_CHARS:
            result.long_lines.append(LongLine(rel_path, line_num, len(line_text)))
            line_text = line_text[:_MAX_LINE_DISPLAY_CHARS] + "... [truncated]"

        if entry_type == "match":
            result.match_count += 1
            if result.match_count > max_results:
                break

        result.entries.append(GrepEntry(rel_path, line_num, ">" if entry_type == "match" else " ", line_text))

    return result


def _format_matches(entries: list[GrepEntry]) -> list[str]:
    """Group entries into context blocks and format for display."""
    matches: list[str] = []
    block: list[GrepEntry] = []

    def _flush_block() -> None:
        if not block:
            return
        first_match = next((b for b in block if b.marker == ">"), block[0])
        header = f"{first_match.rel_path}:{first_match.line_num}"
        lines = [f"  {e.marker} {e.line_num:>4} | {e.text}" for e in block]
        matches.append(header + "\n" + "\n".join(lines))

    for entry in entries:
        if block:
            prev = block[-1]
            if entry.rel_path != prev.rel_path or entry.line_num != prev.line_num + 1:
                _flush_block()
                block = []
        block.append(entry)

    _flush_block()
    return matches


async def _grep_impl(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    context_lines: int = 2,
    max_results: int = _MAX_RESULTS,
    *,
    base_cwd: str | None = None,
) -> str:
    max_results = min(max_results, _MAX_RESULTS_HARD_LIMIT)
    root = resolve_workspace_path(path, base_cwd=base_cwd)
    if not os.path.exists(root):
        return tool_error("path_not_found", f"path not found — {root}", details={"path": path, "resolved_path": root})

    base_args = [
        "--json",
        "--no-messages",
        "-e",
        pattern,
        "-C",
        str(context_lines),
        "--glob",
        "!.git/",
    ]
    # glob="*" is a no-op that overrides .gitignore exclusions — treat as None
    if glob and glob != "*":
        base_args.extend(["--glob", glob])

    # Determine which encodings to search.  For pure-ASCII patterns a single
    # pass suffices since ASCII bytes are identical across all encodings.
    encodings = ["utf-8"] if _is_ascii_only(pattern) else _get_encodings()

    all_entries: list[GrepEntry] = []
    all_long_lines: list[LongLine] = []
    seen: set[tuple[str, int]] = set()  # (rel_path, line_num) for dedup
    seen_long: set[tuple[str, int]] = set()
    total_matches = 0
    last_error: str = ""
    last_error_code: int | None = None

    for enc in encodings:
        args = [*base_args, "-E", enc, root]
        try:
            stdout, stderr, returncode = await _run_rg(args)
        except FileNotFoundError:
            return tool_error("ripgrep_not_found", "ripgrep (rg) not found")
        except TimeoutError:
            record_process_timeout(_TIMEOUT)
            return tool_error("search_timeout", f"search timed out after {_TIMEOUT}s", retryable=True)
        except Exception as e:
            return tool_error("search_failed", f"search failed — {e}", details={"path": path, "pattern": pattern})

        if returncode not in (0, 1):
            last_error = stderr.strip() or f"rg exited with code {returncode}"
            last_error_code = returncode
            continue

        parsed = _parse_grep_jsonl(stdout, root, max_results - total_matches)

        # Deduplicate across encoding passes
        for entry in parsed.entries:
            key = (entry.rel_path, entry.line_num)
            if key not in seen:
                seen.add(key)
                all_entries.append(entry)
                if entry.marker == ">":
                    total_matches += 1
        for ll in parsed.long_lines:
            key = (ll.rel_path, ll.line_num)
            if key not in seen_long:
                seen_long.add(key)
                all_long_lines.append(ll)

        if total_matches >= max_results:
            break

    if not all_entries:
        if last_error:
            if last_error_code is not None:
                record_process_result(last_error_code)
            return tool_error("search_process_failed", last_error, details={"path": path, "pattern": pattern})
        return f"No matches found for /{pattern}/ in {root}"

    # Sort by (file, line_number) for consistent display
    all_entries.sort(key=lambda e: (e.rel_path, e.line_num))

    matches = _format_matches(all_entries)

    if not matches:
        return f"No matches found for /{pattern}/ in {root}"

    header = f"Found {total_matches} match(es) in {root}"
    if total_matches >= max_results:
        header += f" (limited to {max_results})"
    result = header + "\n\n" + "\n\n".join(matches)
    if all_long_lines:
        details = ", ".join(f"{ll.rel_path}:{ll.line_num} ({ll.actual_length} chars)" for ll in all_long_lines)
        result += f"\n\n[Long lines truncated to {_MAX_LINE_DISPLAY_CHARS} chars: {details}]"
    return result


@tool(kind=KIND_SEARCH)
async def grep(
    pattern: Annotated[str, "Regex pattern to search for in file contents."],
    path: Annotated[str, "Directory or file path to search in."] = ".",
    glob: Annotated[
        str | None, "Glob pattern to filter files (e.g. '*.py', '**/*.ts'). Omit to search all files."
    ] = None,
    context_lines: Annotated[int, "Number of context lines before and after each match."] = 2,
    max_results: Annotated[int, "Maximum number of matches to return."] = _MAX_RESULTS,
) -> str:
    """Search file contents using a regex pattern. Returns matching lines with context.

    Powered by ripgrep (rg --json -e PATTERN -C CONTEXT_LINES -E ENCODING PATH).
    The pattern is a Rust-flavor regex (PCRE-like, not Python re). Glob filtering
    uses gitignore-style rules: patterns without '/' match the filename at any
    depth (e.g. '*.py' matches 'src/foo.py'), patterns with '/' match against
    the relative path. max_results is capped at 100 regardless of the value provided.
    """
    return await _grep_impl(
        pattern,
        path=path,
        glob=glob,
        context_lines=context_lines,
        max_results=max_results,
    )


# ---------------------------------------------------------------------------
# glob
# ---------------------------------------------------------------------------


async def _glob_impl(
    pattern: str,
    path: str = ".",
    max_results: int = _MAX_RESULTS,
    *,
    base_cwd: str | None = None,
) -> str:
    max_results = min(max_results, _MAX_RESULTS_HARD_LIMIT)
    root = resolve_workspace_path(path, base_cwd=base_cwd)
    if not os.path.exists(root):
        return tool_error("path_not_found", f"path not found — {root}", details={"path": path, "resolved_path": root})

    args = ["--files", "--glob", "!.git/", "--glob", pattern, root]

    try:
        stdout, stderr, return_code = await _run_rg(args)
    except FileNotFoundError:
        return tool_error("ripgrep_not_found", "ripgrep (rg) not found")
    except TimeoutError:
        record_process_timeout(_TIMEOUT)
        return tool_error("search_timeout", f"search timed out after {_TIMEOUT}s", retryable=True)
    except Exception as e:
        return tool_error("glob_failed", f"glob failed — {e}", details={"path": path, "pattern": pattern})

    if return_code not in (0, 1):
        record_process_result(return_code)
        message = stderr.strip() or f"rg exited with code {return_code}"
        return tool_error("search_process_failed", message, details={"path": path, "pattern": pattern})

    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return f"No files matching '{pattern}' found in {root}"

    # Convert to relative paths and limit results
    results = []
    for line in lines:
        rel = _to_posix(os.path.relpath(line.strip(), root))
        results.append(rel)
        if len(results) >= max_results:
            break

    header = f"Found {len(results)} file(s) matching '{pattern}' in {root}"
    if len(results) >= max_results:
        header += f" (limited to {max_results})"
    return header + "\n" + "\n".join(f"  {r}" for r in results)


@tool(kind=KIND_SEARCH)
async def glob(
    pattern: Annotated[str, "Glob pattern to match file names (e.g. '*.py', 'test_*.py', '**/*.ts')."],
    path: Annotated[str, "Directory to search in."] = ".",
    max_results: Annotated[int, "Maximum number of results to return."] = _MAX_RESULTS,
) -> str:
    """Find files by name pattern. Returns relative paths of matching files.

    Powered by ripgrep (rg --files --glob PATTERN PATH). The pattern uses
    gitignore-style glob rules: patterns without '/' match the filename at any
    depth (e.g. '*.py' matches 'src/foo.py'), patterns with '/' match against
    the relative path. Searches recursively through all subdirectories.
    max_results is capped at 100 regardless of the value provided.
    """
    return await _glob_impl(pattern, path=path, max_results=max_results)


class SearchTools:
    """Runtime-bound search tools that resolve relative paths per session."""

    def __init__(self, runtime: SessionEnvironment) -> None:
        self._runtime = runtime

    def tools(self) -> list:
        """Return search tools for this runtime context."""
        return [self.grep, self.glob]

    @tool(kind=KIND_SEARCH)
    async def grep(
        self,
        pattern: Annotated[str, "Regex pattern to search for in file contents."],
        path: Annotated[str, "Directory or file path to search in."] = ".",
        glob: Annotated[
            str | None, "Glob pattern to filter files (e.g. '*.py', '**/*.ts'). Omit to search all files."
        ] = None,
        context_lines: Annotated[int, "Number of context lines before and after each match."] = 2,
        max_results: Annotated[int, "Maximum number of matches to return."] = _MAX_RESULTS,
    ) -> str:
        """Search file contents using a regex pattern. Returns matching lines with context.

        Powered by ripgrep (rg --json -e PATTERN -C CONTEXT_LINES -E ENCODING PATH).
        The pattern is a Rust-flavor regex (PCRE-like, not Python re). Glob filtering
        uses gitignore-style rules: patterns without '/' match the filename at any
        depth (e.g. '*.py' matches 'src/foo.py'), patterns with '/' match against
        the relative path. max_results is capped at 100 regardless of the value provided.
        """
        return await _grep_impl(
            pattern,
            path=path,
            glob=glob,
            context_lines=context_lines,
            max_results=max_results,
            base_cwd=self._runtime.cwd,
        )

    @tool(kind=KIND_SEARCH)
    async def glob(
        self,
        pattern: Annotated[str, "Glob pattern to match file names (e.g. '*.py', 'test_*.py', '**/*.ts')."],
        path: Annotated[str, "Directory to search in."] = ".",
        max_results: Annotated[int, "Maximum number of results to return."] = _MAX_RESULTS,
    ) -> str:
        """Find files by name pattern. Returns relative paths of matching files.

        Powered by ripgrep (rg --files --glob PATTERN PATH). The pattern uses
        gitignore-style glob rules: patterns without '/' match the filename at any
        depth (e.g. '*.py' matches 'src/foo.py'), patterns with '/' match against
        the relative path. Searches recursively through all subdirectories.
        max_results is capped at 100 regardless of the value provided.
        """
        return await _glob_impl(pattern, path=path, max_results=max_results, base_cwd=self._runtime.cwd)
