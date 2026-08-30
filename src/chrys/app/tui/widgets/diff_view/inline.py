# Copyright (c) 2026 Chrys. All rights reserved.

"""Chat-optimized inline unified diff renderer."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import ClassVar

from textual.content import Content
from textual.geometry import Region
from textual.strip import Strip

from chrys.app.tui.widgets.diff_view import palette as diff_palette
from chrys.app.tui.widgets.diff_view.code import (
    ELLIPSIS_SENTINEL,
    CodeLine,
    DiffAnnotation,
    EllipsisSentinel,
    LineStyle,
    is_ellipsis_sentinel,
)
from chrys.app.tui.widgets.diff_view.compute import (
    DiffOpcodeGroups,
    compute_grouped_opcodes,
    compute_highlighted_code_lines,
    flatten_unified_diff,
)
from chrys.app.tui.widgets.diff_view.styles import line_style_for_color_system
from chrys.app.tui.widgets.diff_view.unified import UnifiedDiffLines


class InlineUnifiedDiffLines(UnifiedDiffLines):
    """A single-widget inline unified diff for chat file-tool previews.

    ``DiffView`` remains the full-featured viewer. Chat inline file cards use
    this direct ``UnifiedDiffLines`` subclass to avoid mounting a wrapper widget
    around the already-flattened no-scrollbar diff surface.
    """

    DARK_NUMBER_STYLES: ClassVar[dict[str, str]] = diff_palette.DARK_NUMBER_STYLES
    DARK_LINE_STYLES: ClassVar[dict[str, str]] = diff_palette.DARK_LINE_STYLES
    DARK_EDGE_STYLES: ClassVar[dict[str, str]] = diff_palette.DARK_EDGE_STYLES
    LIGHT_NUMBER_STYLES: ClassVar[dict[str, str]] = diff_palette.LIGHT_NUMBER_STYLES
    LIGHT_LINE_STYLES: ClassVar[dict[str, str]] = diff_palette.LIGHT_LINE_STYLES
    LIGHT_EDGE_STYLES: ClassVar[dict[str, str]] = diff_palette.LIGHT_EDGE_STYLES

    def __init__(
        self,
        path1: str,
        path2: str,
        code_before: str,
        code_after: str,
        *,
        max_display_lines: int | None = None,
        auto_height: bool = False,
        annotations: bool = True,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = "diff-group",
    ) -> None:
        self.path1 = path1
        self.path2 = path2
        self.code_before = code_before.expandtabs()
        self.code_after = code_after.expandtabs()
        self.max_display_lines = max_display_lines
        self.auto_height = auto_height
        self.annotations = annotations

        self._grouped_opcodes: DiffOpcodeGroups | None = None
        self._highlighted_code_lines: tuple[list[Content], list[Content]] | None = None
        self._flat_numbers_a: list[int | EllipsisSentinel | None] = []
        self._flat_numbers_b: list[int | EllipsisSentinel | None] = []
        self._flat_annotations: list[DiffAnnotation | EllipsisSentinel] = []
        self._flat_styles: list[LineStyle] = []
        self._line_number_width = 1
        self._prepared = False
        self._flat_styles_dark: bool | None = None

        entry_width = self._line_number_width + 2
        super().__init__(
            0,
            entry_width,
            entry_width,
            3 if annotations else 1,
            self._number_factory_a,
            self._number_factory_b,
            self._annotation_factory,
            [],
            [],
            1,
            name=name,
            id=id,
            classes=classes,
        )

    def _is_dark_theme(self) -> bool:
        if not self.is_attached:
            return True
        theme = self.app.current_theme
        return True if theme is None else theme.dark

    def _number_styles(self) -> dict[str, str]:
        return self.DARK_NUMBER_STYLES if self._is_dark_theme() else self.LIGHT_NUMBER_STYLES

    def _line_styles_for_theme(self) -> dict[str, str]:
        return self.DARK_LINE_STYLES if self._is_dark_theme() else self.LIGHT_LINE_STYLES

    def _edge_styles(self) -> dict[str, str]:
        return self.DARK_EDGE_STYLES if self._is_dark_theme() else self.LIGHT_EDGE_STYLES

    async def prepare(self) -> None:
        """Pre-compute syntax highlighting and flattened rows off-thread."""
        if self._prepared:
            return
        line_styles = self._line_styles_for_theme()
        result = await asyncio.to_thread(self._prepare_sync, line_styles)
        (
            self._grouped_opcodes,
            self._highlighted_code_lines,
            self._flat_numbers_a,
            self._flat_numbers_b,
            self._flat_annotations,
            self._code_lines,
            self._line_number_width,
            self._max_code_width,
        ) = result
        entry_width = self._line_number_width + 2
        self._total_lines = len(self._flat_numbers_a)
        self._number_width_a = entry_width
        self._number_width_b = entry_width
        self._annotation_width = 3 if self.annotations else 1
        self._refresh_flat_styles()
        self._apply_height()
        self._prepared = True

    def _prepare_sync(
        self,
        line_styles: dict[str, str],
    ) -> tuple[
        DiffOpcodeGroups,
        tuple[list[Content], list[Content]],
        list[int | EllipsisSentinel | None],
        list[int | EllipsisSentinel | None],
        list[DiffAnnotation | EllipsisSentinel],
        list[CodeLine],
        int,
        int,
    ]:
        grouped_opcodes = compute_grouped_opcodes(self.code_before, self.code_after)
        highlighted_lines = compute_highlighted_code_lines(
            self.code_before,
            self.code_after,
            self.path1,
            self.path2,
            grouped_opcodes,
            grouped_only=True,
        )
        lines_a, lines_b = highlighted_lines
        flat = flatten_unified_diff(grouped_opcodes, lines_a, lines_b, line_styles)
        return (
            grouped_opcodes,
            highlighted_lines,
            flat.line_numbers_a,
            flat.line_numbers_b,
            flat.annotations,
            flat.code_lines,
            flat.line_number_width,
            flat.max_code_width,
        )

    def _apply_height(self) -> None:
        if self.max_display_lines is not None:
            self.styles.height = min(self._total_lines, self.max_display_lines)
        elif self.auto_height:
            self.styles.height = self._total_lines

    def _refresh_flat_styles(self) -> None:
        line_styles = self._line_styles_for_theme()
        self._flat_styles = [
            ELLIPSIS_SENTINEL if is_ellipsis_sentinel(annotation) else line_styles[annotation]
            for annotation in self._flat_annotations
        ]
        self._line_styles = self._flat_styles
        self._flat_styles_dark = self._is_dark_theme()
        self._invalidate_render_cache()

    def _ensure_flat_styles_for_theme(self) -> None:
        if not self._prepared:
            return
        if self._flat_styles_dark != self._is_dark_theme():
            self._refresh_flat_styles()

    def notify_style_update(self) -> None:
        super().notify_style_update()
        self._ensure_flat_styles_for_theme()

    def render_lines(self, crop: Region) -> list[Strip]:
        self._ensure_flat_styles_for_theme()
        return super().render_lines(crop)

    def prewarm_visible_rows(self) -> None:
        """Populate row render caches for the visible inline preview rows."""
        with suppress(Exception):
            if not self._prepared or not self.is_attached:
                return
            width = self.content_region.width or self.size.width
            height = self.size.height
            if width <= 0 or height <= 0:
                return
            self.render_lines(Region(0, 0, width, min(height, self._total_lines)))

    def _number_factory_a(self, index: int) -> Content:
        return self._make_number(index, self._flat_numbers_a, first_column=True)

    def _number_factory_b(self, index: int) -> Content:
        return self._make_number(index, self._flat_numbers_b, first_column=False)

    def _make_number(
        self,
        index: int,
        line_numbers: list[int | EllipsisSentinel | None],
        *,
        first_column: bool,
    ) -> Content:
        number_styles = self._number_styles()
        edge_styles = self._edge_styles()
        line_no = line_numbers[index]
        annotation = self._flat_annotations[index]
        if is_ellipsis_sentinel(line_no) or is_ellipsis_sentinel(annotation):
            return Content(" " * (self._line_number_width + 2))
        if first_column:
            if line_no is None:
                return Content(f"▎{' ' * self._line_number_width} ").stylize(edge_styles[annotation], 0, 1)
            return (
                Content(f"▎{line_no:>{self._line_number_width}} ")
                .stylize(number_styles[annotation], 1)
                .stylize(edge_styles[annotation], 0, 1)
            )
        if line_no is None:
            return Content(f" {' ' * self._line_number_width} ")
        return Content(f" {line_no:>{self._line_number_width}} ").stylize(number_styles[annotation])

    def _annotation_factory(self, index: int) -> Content:
        annotation = self._flat_annotations[index]
        if is_ellipsis_sentinel(annotation):
            return Content("   ")
        line_style = line_style_for_color_system(
            self._line_styles_for_theme()[annotation],
            self.app.console.color_system if self.is_attached else None,
        )
        return Content(f" {annotation} ").stylize(line_style).stylize("bold")
