# Copyright (c) 2026 Chrys. All rights reserved.

"""Read-only tools confined to one frozen clarification snapshot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from chrys.foundation.platform.paths import resolve_workspace_path
from chrys.kernel import Content
from chrys.service.tools.builtins.filesystem import _read_file_impl, _view_image_impl
from chrys.service.tools.builtins.search import _glob_impl, _grep_impl
from chrys.service.tools.kinds import KIND_FILESYSTEM_READ, KIND_SEARCH, tool
from chrys.service.tools.result_metadata import tool_error

if TYPE_CHECKING:
    from chrys.foundation.models.session_env import SessionEnvironment


class SnapshotReadTools:
    """Expose filesystem reads and searches only inside frozen snapshot paths."""

    def __init__(
        self,
        runtime: SessionEnvironment,
        *,
        roots: tuple[Path, ...],
        reference_files: tuple[Path, ...],
    ) -> None:
        self._runtime = runtime
        self._roots = tuple(_canonical_path(path) for path in roots)
        self._reference_files = frozenset(_canonical_path(path) for path in reference_files)

    def tools(self) -> list:
        """Return the bounded read/search tool set."""
        return [self.read_file, self.view_image, self.grep, self.glob]

    def _resolve_allowed(self, path: str) -> str | None:
        candidate = _canonical_path(Path(resolve_workspace_path(path, base_cwd=self._runtime.cwd)))
        if candidate in self._reference_files:
            return candidate
        if any(_is_within(candidate, root) for root in self._roots):
            return candidate
        return None

    @tool(kind=KIND_FILESYSTEM_READ)
    def read_file(
        self,
        path: Annotated[str, "Absolute or relative path inside the frozen snapshot."],
        max_tokens: Annotated[int, "Token budget for returned content."] = 5000,
        line_range: Annotated[list[int] | None, "Optional [start, end] 1-indexed line range."] = None,
    ) -> str:
        """Read a text file from the frozen snapshot."""
        resolved = self._resolve_allowed(path)
        if resolved is None:
            return _outside_snapshot_error(path)
        return _read_file_impl(resolved, max_tokens=max_tokens, line_range=line_range)

    @tool(kind=KIND_FILESYSTEM_READ)
    def view_image(
        self,
        path: Annotated[str, "Absolute or relative image path inside the frozen snapshot."],
    ) -> list[Content]:
        """Read an image from the frozen snapshot."""
        resolved = self._resolve_allowed(path)
        if resolved is None:
            return [Content.from_text(_outside_snapshot_error(path))]
        return _view_image_impl(resolved)

    @tool(kind=KIND_SEARCH)
    async def grep(
        self,
        pattern: Annotated[str, "Regex pattern to search for."],
        path: Annotated[str, "File or directory inside the frozen snapshot."] = ".",
        glob: Annotated[str | None, "Optional glob filter."] = None,
        context_lines: Annotated[int, "Context lines around each match."] = 2,
        max_results: Annotated[int, "Maximum matches to return."] = 50,
    ) -> str:
        """Search file contents inside the frozen snapshot."""
        resolved = self._resolve_allowed(path)
        if resolved is None:
            return _outside_snapshot_error(path)
        return await _grep_impl(
            pattern,
            path=resolved,
            glob=glob,
            context_lines=context_lines,
            max_results=max_results,
        )

    @tool(kind=KIND_SEARCH)
    async def glob(
        self,
        pattern: Annotated[str, "Glob pattern to match."],
        path: Annotated[str, "Directory inside the frozen snapshot."] = ".",
        max_results: Annotated[int, "Maximum paths to return."] = 50,
    ) -> str:
        """Find paths inside the frozen snapshot."""
        resolved = self._resolve_allowed(path)
        if resolved is None:
            return _outside_snapshot_error(path)
        return await _glob_impl(pattern, path=resolved, max_results=max_results)


def _canonical_path(path: Path) -> str:
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def _is_within(candidate: str, root: str) -> bool:
    try:
        return os.path.commonpath((candidate, root)) == root
    except ValueError:
        return False


def _outside_snapshot_error(path: str) -> str:
    return tool_error(
        "path_outside_snapshot",
        "path is outside the frozen requirement-clarification snapshot",
        details={"path": path},
    )
