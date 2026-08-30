# Copyright (c) 2026 Chrys. All rights reserved.

"""Virtualized annotation column that syncs scroll position from a linked CodeColumn."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Self

from textual.cache import LRUCache
from textual.content import Content
from textual.geometry import Region, Size
from textual.strip import Strip
from textual.widget import Widget

from chrys.app.tui.support.gc_freeze import detach_lru_cache, renew_lru_cache

if TYPE_CHECKING:
    from chrys.app.tui.widgets.diff_view.code import CodeColumn


class AnnotationsColumn(Widget):
    """Virtualized annotation column that reads scroll position from a linked CodeColumn.

    This widget is NOT scrollable itself — it reads the CodeColumn's
    scroll_offset.y to determine which content lines are visible, then
    renders the matching annotation entries on demand via a factory function.
    """

    DEFAULT_CSS = """
    AnnotationsColumn {
        width: auto;
        height: 1fr;
    }
    """

    def __init__(
        self,
        total_lines: int,
        entry_width: int,
        entry_factory: Callable[[int], Content],
        code_column: CodeColumn,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._total_lines = total_lines
        self._entry_width = entry_width
        self._entry_factory = entry_factory
        self._entry_cache: LRUCache[int, Content] = LRUCache(maxsize=500)
        self._strip_cache: LRUCache[int, Strip] = LRUCache(maxsize=500)
        self.code_column = code_column
        # Per-frame cached values (set in render_lines, used in render_line)
        self._frame_scroll_y: int = 0
        self._frame_height: int = 0
        self._frame_blank: Strip = Strip.blank(0)
        # Full-frame result cache
        self._lines_cache_key: tuple[int, int, int, int, int, int] = (-1, -1, -1, -1, -1, -1)
        self._lines_cache_result: list[Strip] | None = None
        self._cache_theme = ""
        self._cache_color_system = ""

    def prepare_for_gc_freeze(self) -> None:
        """Detach cyclic entry and strip LRUs before freezing."""
        # gc-freeze swaps in an acyclic capacity token; renew restores a real cache before next use.
        self._entry_cache = detach_lru_cache(self._entry_cache)  # ty: ignore[invalid-assignment]
        self._strip_cache = detach_lru_cache(self._strip_cache)  # ty: ignore[invalid-assignment]
        self._lines_cache_result = None

    def after_gc_freeze(self) -> None:
        """Recreate both LRUs after the permanent generation changes."""
        self._entry_cache = renew_lru_cache(self._entry_cache)
        self._strip_cache = renew_lru_cache(self._strip_cache)

    def abort_gc_freeze(self) -> None:
        """Restore both LRUs after an incomplete hook pass."""
        self._entry_cache = renew_lru_cache(self._entry_cache)
        self._strip_cache = renew_lru_cache(self._strip_cache)

    def _get_entry(self, index: int) -> Content:
        """Get or create a Content entry for the given line index."""
        cached = self._entry_cache.get(index)
        if cached is not None:
            return cached
        entry = self._entry_factory(index)
        self._entry_cache[index] = entry
        return entry

    def _invalidate_render_cache(self) -> None:
        self._entry_cache.clear()
        self._strip_cache.clear()
        self._cache_theme = ""
        self._cache_color_system = ""
        self._frame_blank = Strip.blank(0)
        self._lines_cache_key = (-1, -1, -1, -1, -1, -1)
        self._lines_cache_result = None

    def notify_style_update(self) -> None:
        """Clear rendered strips when theme/CSS variables change."""
        super().notify_style_update()
        self._invalidate_render_cache()
        self.refresh()

    def refresh(
        self,
        *regions: Region,
        repaint: bool = True,
        layout: bool = False,
        recompose: bool = False,
    ) -> Self:
        """Invalidate frame cache before refresh to prevent stale data."""
        self._lines_cache_result = None
        return super().refresh(*regions, repaint=repaint, layout=layout, recompose=recompose)

    def get_content_width(self, container: Size, viewport: Size) -> int:
        return self._entry_width

    def render_lines(self, crop: Region) -> list[Strip]:
        if not self.is_attached:
            return [Strip.blank(crop.width) for _ in crop.line_range]

        # Pre-compute expensive values once per frame instead of per row.
        try:
            scroll_y = round(self.code_column.scroll_offset.y)
            height = self.code_column.scrollable_content_region.height
            theme = self.app.theme
            color_system = str(self.app.console.color_system)

            if theme != self._cache_theme or color_system != self._cache_color_system:
                self._invalidate_render_cache()
                self._cache_theme = theme
                self._cache_color_system = color_system

            # Fast path: return cached result if scroll position unchanged.
            lines_key = (scroll_y, height, crop.x, crop.y, crop.width, crop.height)
            if (
                lines_key == self._lines_cache_key
                and self._lines_cache_result is not None
                and crop.height == len(self._lines_cache_result)
            ):
                return self._lines_cache_result

            self._frame_scroll_y = scroll_y
            self._frame_height = height
            self._frame_blank = Strip.blank(self._entry_width, self.visual_style.rich_style)
        except Exception:
            return super().render_lines(crop)

        result = self._render_lines_direct(crop)
        self._lines_cache_key = lines_key
        self._lines_cache_result = result
        return result

    def _render_lines_direct(self, crop: Region) -> list[Strip]:
        """Render visible annotation rows without the generic style wrapper."""
        width = self._entry_width
        line_filters = self.get_line_filters()
        _base_background, background = self.background_colors
        strips: list[Strip] = []
        for y in crop.line_range:
            strip = self.render_line(y)
            for line_filter in line_filters:
                strip = strip.apply_filter(line_filter, background)
            if crop.column_span != (0, width):
                strip = strip.crop(crop.x, crop.x + crop.width)
            strips.append(strip)
        return strips

    def render_line(self, y: int) -> Strip:
        if not self.is_attached:
            return Strip.blank(max(0, self._entry_width)).apply_offsets(0, y)

        width = self._entry_width

        # Only render as many rows as the CodeColumn's scrollable area
        # so annotations stay aligned even when the h-scrollbar is absent.
        if y >= self._frame_height:
            return self._frame_blank

        content_y = self._frame_scroll_y + y

        if content_y < 0 or content_y >= self._total_lines:
            return self._frame_blank

        cached = self._strip_cache.get(content_y)
        if cached is not None:
            return cached

        visual_style = self.visual_style
        rich_style = visual_style.rich_style
        entry = self._get_entry(content_y)
        strip = Strip(entry.render_segments(visual_style), cell_length=entry.cell_length)
        strip = strip.adjust_cell_length(width, rich_style)
        self._strip_cache[content_y] = strip
        return strip
