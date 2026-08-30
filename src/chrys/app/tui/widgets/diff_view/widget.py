# Copyright (c) 2026 Chrys. All rights reserved.

"""DiffView — high-performance split/unified diff widget with virtualized scrolling.

Adapted from toad (https://github.com/willmcgugan/toad) — original author: Will McGugan, License: MIT.

Callers should call ``await dv.prepare()`` (off-thread) before mounting to
pre-compute syntax highlighting and diff opcodes.  The compose methods assume
cached data is available — no on_mount background work or recompose needed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from itertools import chain
from typing import TYPE_CHECKING, ClassVar, Literal

from textual import containers
from textual.content import Content
from textual.reactive import reactive, var

from chrys.app.tui.widgets import HATCH_GLYPH, HATCH_STYLE
from chrys.app.tui.widgets.diff_view import palette as diff_palette
from chrys.app.tui.widgets.diff_view._utils import fill_lists, loop_last
from chrys.app.tui.widgets.diff_view.annotations import AnnotationsColumn
from chrys.app.tui.widgets.diff_view.code import (
    ELLIPSIS_SENTINEL,
    CodeColumn,
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
    highlight_diff_lines,
)
from chrys.app.tui.widgets.diff_view.styles import line_style_for_color_system
from chrys.app.tui.widgets.diff_view.unified import UnifiedDiffLines

if TYPE_CHECKING:
    from textual.app import ComposeResult

type Annotation = DiffAnnotation
type LineNumber = int | EllipsisSentinel | None
type AnnotationLine = Annotation | EllipsisSentinel
type UnifiedFlattened = tuple[
    list[LineNumber],
    list[LineNumber],
    list[AnnotationLine],
    list[CodeLine],
    list[LineStyle],
    int,
    int,
]
type SplitFlattened = tuple[
    list[LineNumber],
    list[LineNumber],
    list[AnnotationLine],
    list[AnnotationLine],
    list[CodeLine],
    list[CodeLine],
    list[LineStyle],
    list[LineStyle],
    int,
    int,
]


class DiffView(containers.VerticalGroup):
    """A formatted diff in unified or split format."""

    code_before: reactive[str] = reactive("")
    code_after: reactive[str] = reactive("")
    path1: reactive[str] = reactive("")
    path2: reactive[str] = reactive("")
    split: reactive[bool] = reactive(True, recompose=True)
    annotations: var[bool] = var(True, toggle_class="-with-annotations")

    DEFAULT_CSS = """
    DiffView {
        width: 1fr;
        height: auto;

        .diff-group {
            background: $foreground 4%;
        }

        .annotations { width: 1; }
        &.-with-annotations {
            .annotations { width: auto; }
        }
        .title {
            border-bottom: dashed $tui-border-foreground 20%;
        }
        .code-left {
            scrollbar-size-vertical: 0;
        }
    }
    """

    DARK_NUMBER_STYLES: ClassVar[dict[str, str]] = diff_palette.DARK_NUMBER_STYLES
    DARK_LINE_STYLES: ClassVar[dict[str, str]] = diff_palette.DARK_LINE_STYLES
    DARK_EDGE_STYLES: ClassVar[dict[str, str]] = diff_palette.DARK_EDGE_STYLES
    LIGHT_NUMBER_STYLES: ClassVar[dict[str, str]] = diff_palette.LIGHT_NUMBER_STYLES
    LIGHT_LINE_STYLES: ClassVar[dict[str, str]] = diff_palette.LIGHT_LINE_STYLES
    LIGHT_EDGE_STYLES: ClassVar[dict[str, str]] = diff_palette.LIGHT_EDGE_STYLES
    NUMBER_STYLES: ClassVar[dict[str, str]] = DARK_NUMBER_STYLES
    LINE_STYLES: ClassVar[dict[str, str]] = DARK_LINE_STYLES
    EDGE_STYLES: ClassVar[dict[str, str]] = DARK_EDGE_STYLES

    def __init__(
        self,
        path1: str,
        path2: str,
        code_before: str,
        code_after: str,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ):
        super().__init__(name=name, id=id, classes=classes)
        self.set_reactive(DiffView.path1, path1)
        self.set_reactive(DiffView.path2, path2)
        self.set_reactive(DiffView.code_before, code_before.expandtabs())
        self.set_reactive(DiffView.code_after, code_after.expandtabs())
        self._grouped_opcodes: DiffOpcodeGroups | None = None
        self._highlighted_code_lines: tuple[list[Content], list[Content]] | None = None
        self._flatten_unified_cache: UnifiedFlattened | None = None
        self._flatten_split_cache: SplitFlattened | None = None
        self._cache_theme_dark: bool | None = None
        self.max_display_lines: int | None = None
        self.auto_height: bool = False
        self.show_scrollbars: bool = True

    def _is_dark_theme(self) -> bool:
        if not self.is_attached:
            return True
        theme = self.app.current_theme
        return True if theme is None else theme.dark

    def _clear_flatten_cache(self) -> None:
        self._flatten_unified_cache = None
        self._flatten_split_cache = None

    def _ensure_palette_cache(self) -> None:
        dark = self._is_dark_theme()
        if self._cache_theme_dark != dark:
            self._clear_flatten_cache()
            self._cache_theme_dark = dark

    def notify_style_update(self) -> None:
        previous_dark = self._cache_theme_dark
        super().notify_style_update()
        self._ensure_palette_cache()
        # Only dark/light flips need a recompose because they change the
        # per-row palette. Same-brightness theme switches keep the same row
        # style strings; CodeColumn/AnnotationsColumn clear rendered strips
        # so CSS variables still re-resolve.
        if previous_dark is not None and previous_dark != self._cache_theme_dark:
            self.refresh(recompose=True)

    def watch_annotations(self, annotations: bool) -> None:
        """Keep the flat unified renderer in sync with the annotation toggle."""
        if not self.is_attached:
            return
        annotation_width = 3 if annotations else 1
        for flat_diff in self.query(UnifiedDiffLines):
            flat_diff.set_annotation_width(annotation_width)

    def _number_styles(self) -> dict[str, str]:
        return self.DARK_NUMBER_STYLES if self._is_dark_theme() else self.LIGHT_NUMBER_STYLES

    def _line_styles(self) -> dict[str, str]:
        return self.DARK_LINE_STYLES if self._is_dark_theme() else self.LIGHT_LINE_STYLES

    def _edge_styles(self) -> dict[str, str]:
        return self.DARK_EDGE_STYLES if self._is_dark_theme() else self.LIGHT_EDGE_STYLES

    async def prepare(self) -> None:
        """Pre-compute diff opcodes and syntax highlighting off-thread.

        Call this before mounting so that compose() finds cached results
        instantly and does not block the event loop.
        """

        def _compute() -> None:
            # Access the cached properties to trigger computation
            self.grouped_opcodes  # noqa: B018
            self.highlighted_code_lines  # noqa: B018

        await asyncio.to_thread(_compute)

    @property
    def grouped_opcodes(self) -> DiffOpcodeGroups:
        if self._grouped_opcodes is None:
            self._grouped_opcodes = compute_grouped_opcodes(self.code_before, self.code_after)
        return self._grouped_opcodes

    @property
    def counts(self) -> tuple[int, int]:
        """Return (additions, removals) counts."""
        additions = 0
        removals = 0
        for group in self.grouped_opcodes:
            for tag, i1, i2, j1, j2 in group:
                if tag == "delete":
                    removals += i2 - i1
                elif tag == "replace":
                    additions += j2 - j1
                    removals += i2 - i1
                elif tag == "insert":
                    additions += j2 - j1
        return additions, removals

    @classmethod
    def _highlight_diff_lines(
        cls, lines_a: list[Content], lines_b: list[Content]
    ) -> tuple[list[Content], list[Content]]:
        """Character-level diff highlighting for replace operations."""
        return highlight_diff_lines(lines_a, lines_b)

    @property
    def highlighted_code_lines(self) -> tuple[list[Content], list[Content]]:
        if self._highlighted_code_lines is None:
            self._highlighted_code_lines = compute_highlighted_code_lines(
                self.code_before,
                self.code_after,
                self.path1,
                self.path2,
                self.grouped_opcodes,
            )
        return self._highlighted_code_lines

    def get_title(self) -> Content:
        additions, removals = self.counts
        title = Content.from_markup(
            "[$text-success][b]+$additions[/b][/], [$text-error][b]-$removals[/b][/]",
            additions=additions,
            removals=removals,
        ).stylize_before("$text")
        return title

    # ── Flattening ────────────────────────────────────────────────────

    def _flatten_unified(
        self,
    ) -> UnifiedFlattened:
        self._ensure_palette_cache()
        if self._flatten_unified_cache is not None:
            return self._flatten_unified_cache

        lines_a, lines_b = self.highlighted_code_lines
        flat = flatten_unified_diff(self.grouped_opcodes, lines_a, lines_b, self._line_styles())
        result = (
            flat.line_numbers_a,
            flat.line_numbers_b,
            flat.annotations,
            flat.code_lines,
            flat.line_styles,
            flat.line_number_width,
            flat.max_code_width,
        )
        self._flatten_unified_cache = result
        return result

    def _flatten_split(
        self,
    ) -> SplitFlattened:
        self._ensure_palette_cache()
        if self._flatten_split_cache is not None:
            return self._flatten_split_cache

        line_styles = self._line_styles()
        lines_a, lines_b = self.highlighted_code_lines

        flat_numbers_a: list[LineNumber] = []
        flat_numbers_b: list[LineNumber] = []
        flat_ann_a: list[AnnotationLine] = []
        flat_ann_b: list[AnnotationLine] = []
        flat_code_a: list[CodeLine] = []
        flat_code_b: list[CodeLine] = []
        flat_styles_a: list[LineStyle] = []
        flat_styles_b: list[LineStyle] = []

        max_code_width = 1

        for last, group in loop_last(self.grouped_opcodes):
            group_numbers_a: list[int | None] = []
            group_numbers_b: list[int | None] = []
            group_ann_a: list[Annotation] = []
            group_ann_b: list[Annotation] = []
            group_code_a: list[Content | None] = []
            group_code_b: list[Content | None] = []

            for tag, i1, i2, j1, j2 in group:
                if tag == "equal":
                    for line_offset, line in enumerate(lines_a[i1:i2], 1):
                        group_ann_a.append(" ")
                        group_ann_b.append(" ")
                        group_numbers_a.append(i1 + line_offset)
                        group_numbers_b.append(j1 + line_offset)
                        group_code_a.append(line)
                        group_code_b.append(line)
                        max_code_width = max(max_code_width, line.cell_length)
                else:
                    if tag in {"delete", "replace"}:
                        for line_number, line in enumerate(lines_a[i1:i2], i1 + 1):
                            group_ann_a.append("-")
                            group_numbers_a.append(line_number)
                            group_code_a.append(line)
                            max_code_width = max(max_code_width, line.cell_length)
                    if tag in {"insert", "replace"}:
                        for line_number, line in enumerate(lines_b[j1:j2], j1 + 1):
                            group_ann_b.append("+")
                            group_numbers_b.append(line_number)
                            group_code_b.append(line)
                            max_code_width = max(max_code_width, line.cell_length)
                    fill_lists(group_code_a, group_code_b, None)
                    fill_lists(group_ann_a, group_ann_b, "/")
                    fill_lists(group_numbers_a, group_numbers_b, None)

            for i, ann in enumerate(group_ann_a):
                flat_numbers_a.append(group_numbers_a[i])
                flat_ann_a.append(ann)
                flat_code_a.append(group_code_a[i])
                flat_styles_a.append(line_styles.get(ann, ""))

            for i, ann in enumerate(group_ann_b):
                flat_numbers_b.append(group_numbers_b[i])
                flat_ann_b.append(ann)
                flat_code_b.append(group_code_b[i])
                flat_styles_b.append(line_styles.get(ann, ""))

            if not last:
                flat_numbers_a.append(ELLIPSIS_SENTINEL)
                flat_numbers_b.append(ELLIPSIS_SENTINEL)
                flat_ann_a.append(ELLIPSIS_SENTINEL)
                flat_ann_b.append(ELLIPSIS_SENTINEL)
                flat_code_a.append(ELLIPSIS_SENTINEL)
                flat_code_b.append(ELLIPSIS_SENTINEL)
                flat_styles_a.append(ELLIPSIS_SENTINEL)
                flat_styles_b.append(ELLIPSIS_SENTINEL)

        line_number_width = max(
            (len(str(n)) for n in chain(flat_numbers_a, flat_numbers_b) if isinstance(n, int)),
            default=1,
        )

        result = (
            flat_numbers_a,
            flat_numbers_b,
            flat_ann_a,
            flat_ann_b,
            flat_code_a,
            flat_code_b,
            flat_styles_a,
            flat_styles_b,
            line_number_width,
            max_code_width,
        )
        self._flatten_split_cache = result
        return result

    # ── Lazy formatting factories ─────────────────────────────────────

    def _make_number_factory_unified(
        self,
        line_numbers: list[LineNumber],
        annotations: list[AnnotationLine],
        line_number_width: int,
        first_column: bool,
    ) -> Callable[[int], Content]:
        NUMBER_STYLES = self._number_styles()
        EDGE_STYLES = self._edge_styles()

        def factory(index: int) -> Content:
            line_no = line_numbers[index]
            annotation = annotations[index]
            if is_ellipsis_sentinel(line_no):
                return Content(" " * (line_number_width + 2))
            if is_ellipsis_sentinel(annotation):
                return Content(" " * (line_number_width + 2))
            if first_column:
                if line_no is None:
                    return Content(f"▎{' ' * line_number_width} ").stylize(EDGE_STYLES[annotation], 0, 1)
                return (
                    Content(f"▎{line_no:>{line_number_width}} ")
                    .stylize(NUMBER_STYLES[annotation], 1)
                    .stylize(EDGE_STYLES[annotation], 0, 1)
                )
            if line_no is None:
                return Content(f" {' ' * line_number_width} ")
            return Content(f" {line_no:>{line_number_width}} ").stylize(NUMBER_STYLES[annotation])

        return factory

    def _make_annotation_factory_unified(
        self,
        annotations: list[AnnotationLine],
    ) -> Callable[[int], Content]:
        LINE_STYLES = self._line_styles()

        def factory(index: int) -> Content:
            annotation = annotations[index]
            if is_ellipsis_sentinel(annotation):
                return Content("   ")
            line_style = line_style_for_color_system(
                LINE_STYLES[annotation],
                self.app.console.color_system if self.is_attached else None,
            )
            # UnifiedDiffLines relies on the leading space so width=1 can
            # collapse annotations without showing the symbol.
            return Content(f" {annotation} ").stylize(line_style).stylize("bold")

        return factory

    def _make_number_factory_split(
        self,
        line_numbers: list[LineNumber],
        annotations_list: list[AnnotationLine],
        line_number_width: int,
    ) -> Callable[[int], Content]:
        NUMBER_STYLES = self._number_styles()
        EDGE_STYLES = self._edge_styles()
        hatch = Content.styled(HATCH_GLYPH * (2 + line_number_width), HATCH_STYLE)

        def factory(index: int) -> Content:
            line_no = line_numbers[index]
            annotation = annotations_list[index]
            if is_ellipsis_sentinel(line_no):
                return Content(" " * (2 + line_number_width))
            if is_ellipsis_sentinel(annotation):
                return Content(" " * (2 + line_number_width))
            if line_no is None:
                return hatch
            return (
                Content(f"▎{line_no:>{line_number_width}} ")
                .stylize(NUMBER_STYLES[annotation], 1)
                .stylize(EDGE_STYLES[annotation], 0, 1)
            )

        return factory

    def _make_annotation_factory_split(
        self,
        annotations_list: list[AnnotationLine],
        highlight_annotation: Literal["+", "-"],
    ) -> Callable[[int], Content]:
        LINE_STYLES = self._line_styles()
        annotation_hatch = Content.styled(HATCH_GLYPH * 3, HATCH_STYLE)
        annotation_blank = Content(" " * 3)

        def factory(index: int) -> Content:
            annotation = annotations_list[index]
            if is_ellipsis_sentinel(annotation):
                return Content("   ")
            if annotation == highlight_annotation:
                line_style = line_style_for_color_system(
                    LINE_STYLES[annotation],
                    self.app.console.color_system if self.is_attached else None,
                )
                return Content(f" {annotation} ").stylize(line_style).stylize("bold")
            if annotation == "/":
                return annotation_hatch
            return annotation_blank

        return factory

    # ── Compose ───────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        if self.split:
            yield from self.compose_split()
        else:
            yield from self.compose_unified()

    def compose_unified(self) -> ComposeResult:
        (
            flat_numbers_a,
            flat_numbers_b,
            flat_annotations,
            flat_code,
            flat_styles,
            line_number_width,
            max_code_width,
        ) = self._flatten_unified()

        total_lines = len(flat_numbers_a)
        entry_width = line_number_width + 2

        num_factory_a = self._make_number_factory_unified(
            flat_numbers_a, flat_annotations, line_number_width, first_column=True
        )
        num_factory_b = self._make_number_factory_unified(
            flat_numbers_b, flat_annotations, line_number_width, first_column=False
        )
        ann_factory = self._make_annotation_factory_unified(flat_annotations)

        if not self.show_scrollbars:
            # Retained for non-chat callers/tests that want the historical
            # flat no-scrollbar DiffView surface. Chat inline previews now use
            # InlineUnifiedDiffLines directly.
            annotation_width = 3 if self.annotations else 1
            flat_diff = UnifiedDiffLines(
                total_lines,
                entry_width,
                entry_width,
                annotation_width,
                num_factory_a,
                num_factory_b,
                ann_factory,
                flat_code,
                flat_styles,
                max_code_width,
                classes="diff-group",
            )
            if self.max_display_lines is not None:
                flat_diff.styles.height = min(total_lines, self.max_display_lines)
            elif self.auto_height:
                flat_diff.styles.height = total_lines
            yield flat_diff
            return

        code_column = CodeColumn(flat_code, flat_styles, max_code_width, classes="code-right")
        if not self.show_scrollbars:
            code_column.styles.overflow_x = "hidden"
            code_column.styles.overflow_y = "hidden"
            code_column.styles.scrollbar_gutter = "stable"

        ann_col_a = AnnotationsColumn(total_lines, entry_width, num_factory_a, code_column)
        ann_col_b = AnnotationsColumn(total_lines, entry_width, num_factory_b, code_column)
        ann_col_sym = AnnotationsColumn(total_lines, 3, ann_factory, code_column, classes="annotations")

        code_column.annotation_columns = [ann_col_a, ann_col_b, ann_col_sym]

        group = containers.HorizontalGroup(classes="diff-group")
        if self.max_display_lines is not None:
            group.styles.height = min(total_lines, self.max_display_lines)
        elif self.auto_height:
            group.styles.height = total_lines + (1 if self.show_scrollbars else 0)
        with group:
            yield ann_col_a
            yield ann_col_b
            yield ann_col_sym
            yield code_column

    def compose_split(self) -> ComposeResult:
        (
            flat_numbers_a,
            flat_numbers_b,
            flat_ann_a,
            flat_ann_b,
            flat_code_a,
            flat_code_b,
            flat_styles_a,
            flat_styles_b,
            line_number_width,
            max_code_width,
        ) = self._flatten_split()

        total_lines = len(flat_numbers_a)
        num_entry_width = line_number_width + 2

        num_factory_a = self._make_number_factory_split(flat_numbers_a, flat_ann_a, line_number_width)
        num_factory_b = self._make_number_factory_split(flat_numbers_b, flat_ann_b, line_number_width)
        sym_factory_a = self._make_annotation_factory_split(flat_ann_a, "-")
        sym_factory_b = self._make_annotation_factory_split(flat_ann_b, "+")

        code_left = CodeColumn(flat_code_a, flat_styles_a, max_code_width, classes="code-left")
        code_right = CodeColumn(flat_code_b, flat_styles_b, max_code_width, classes="code-right")
        if not self.show_scrollbars:
            # Retained for compatibility with the historical no-scrollbar
            # DiffView mode; chat inline previews do not use this path.
            for _col in (code_left, code_right):
                _col.styles.overflow_x = "hidden"
                _col.styles.overflow_y = "hidden"
                _col.styles.scrollbar_gutter = "stable"

        code_left.scroll_sync = code_right
        code_right.scroll_sync = code_left

        ann_num_a = AnnotationsColumn(total_lines, num_entry_width, num_factory_a, code_left)
        ann_sym_a = AnnotationsColumn(total_lines, 3, sym_factory_a, code_left, classes="annotations")
        ann_num_b = AnnotationsColumn(total_lines, num_entry_width, num_factory_b, code_right)
        ann_sym_b = AnnotationsColumn(total_lines, 3, sym_factory_b, code_right, classes="annotations")

        code_left.annotation_columns = [ann_num_a, ann_sym_a]
        code_right.annotation_columns = [ann_num_b, ann_sym_b]

        group = containers.HorizontalGroup(classes="diff-group")
        if self.max_display_lines is not None:
            group.styles.height = min(total_lines, self.max_display_lines)
        elif self.auto_height:
            group.styles.height = total_lines + (1 if self.show_scrollbars else 0)
        with group:
            yield ann_num_a
            yield ann_sym_a
            yield code_left
            yield ann_num_b
            yield ann_sym_b
            yield code_right
