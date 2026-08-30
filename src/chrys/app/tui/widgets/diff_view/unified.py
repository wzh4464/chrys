# Copyright (c) 2026 Chrys. All rights reserved.

"""Flat unified diff renderer for inline, parent-scrolled diffs."""

from __future__ import annotations

from collections.abc import Callable

from rich.segment import Segment
from textual.cache import LRUCache
from textual.content import Content
from textual.dom import NoScreen
from textual.geometry import Offset, Region, Size
from textual.selection import Selection
from textual.strip import Strip
from textual.widget import Widget

from chrys.app.tui.support.gc_freeze import detach_lru_cache, renew_lru_cache
from chrys.app.tui.widgets import normalize_selection_rich_style
from chrys.app.tui.widgets.diff_view.code import CodeLine, LineStyle, is_ellipsis_sentinel
from chrys.app.tui.widgets.diff_view.styles import line_style_for_color_system


class UnifiedDiffLines(Widget):
    """Single-widget unified diff renderer used when the parent owns scrolling.

    The regular DiffView unified layout uses separate annotation and code
    widgets so the code pane can scroll horizontally and vertically. Chat
    renders diffs with scrollbars hidden and lets the transcript own vertical
    scrolling, so the synchronized child-widget layout only adds compositor
    work. This widget flattens the same data into one Line API surface.
    """

    ALLOW_SELECT = True

    DEFAULT_CSS = """
    UnifiedDiffLines {
        width: 1fr;
        height: auto;
    }
    """

    def __init__(
        self,
        total_lines: int,
        number_width_a: int,
        number_width_b: int,
        annotation_width: int,
        number_factory_a: Callable[[int], Content],
        number_factory_b: Callable[[int], Content],
        annotation_factory: Callable[[int], Content],
        code_lines: list[CodeLine],
        line_styles: list[LineStyle],
        max_code_width: int,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._total_lines = total_lines
        self._number_width_a = number_width_a
        self._number_width_b = number_width_b
        self._annotation_width = annotation_width
        self._number_factory_a = number_factory_a
        self._number_factory_b = number_factory_b
        self._annotation_factory = annotation_factory
        self._code_lines = code_lines
        self._line_styles = line_styles
        self._max_code_width = max_code_width
        self._strip_cache: LRUCache[tuple[int, int], Strip] = LRUCache(maxsize=2000)
        self._plain_code_lines_cache: list[str] | None = None
        self._cache_theme = ""
        self._cache_color_system = ""
        # Hoisted out of ``_render_base_line`` so per-row code doesn't traverse
        # ``self.app.console`` for the line-style downgrade map.
        self._frame_color_system: str = ""
        self._lines_cache_key: tuple[int, int, int, int, int, str, str] = (-1, -1, -1, -1, -1, "", "")
        self._lines_cache_result: list[Strip] | None = None

    def prepare_for_gc_freeze(self) -> None:
        """Detach the cyclic strip LRU before the permanent generation changes."""
        # gc-freeze swaps in an acyclic capacity token; renew restores a real cache before next use.
        self._strip_cache = detach_lru_cache(self._strip_cache)  # ty: ignore[invalid-assignment]
        self._lines_cache_result = None

    def after_gc_freeze(self) -> None:
        """Recreate the strip LRU after the permanent generation changes."""
        self._strip_cache = renew_lru_cache(self._strip_cache)

    def abort_gc_freeze(self) -> None:
        """Restore the strip LRU after an incomplete hook pass."""
        self._strip_cache = renew_lru_cache(self._strip_cache)

    @property
    def _gutter_width(self) -> int:
        return self._number_width_a + self._number_width_b + self._annotation_width

    def set_annotation_width(self, width: int) -> None:
        """Update the rendered annotation column width."""
        if width == self._annotation_width:
            return
        self._annotation_width = width
        self._invalidate_render_cache()
        self.refresh(layout=True)

    def _invalidate_render_cache(self) -> None:
        self._strip_cache.clear()
        self._cache_theme = ""
        self._cache_color_system = ""
        # Reset the frame color system too — otherwise a stale value from the
        # previous theme would mask the live fetch in ``_render_base_line``.
        self._frame_color_system = ""
        self._lines_cache_key = (-1, -1, -1, -1, -1, "", "")
        self._lines_cache_result = None

    def notify_style_update(self) -> None:
        """Clear rendered strips when theme/CSS variables change."""
        super().notify_style_update()
        self._invalidate_render_cache()
        self.refresh()

    def get_content_width(self, container: Size, viewport: Size) -> int:
        return self._gutter_width + self._max_code_width

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return self._total_lines

    def render_lines(self, crop: Region) -> list[Strip]:
        if not self.is_attached:
            return [Strip.blank(crop.width) for _ in crop.line_range]

        theme = self.app.theme
        color_system = str(self.app.console.color_system)
        if theme != self._cache_theme or color_system != self._cache_color_system:
            self._invalidate_render_cache()
            self._cache_theme = theme
            self._cache_color_system = color_system
        self._frame_color_system = color_system

        width = self.content_region.width
        # Line filters and background are safe to omit from the key in Chrys:
        # App.ansi_color is pinned to None, so they change with theme/style
        # invalidation. A runtime ANSI toggle would need to join this key.
        lines_key = (width, crop.x, crop.y, crop.width, crop.height, theme, color_system)
        selection = self._safe_text_selection()
        if (
            selection is None
            and lines_key == self._lines_cache_key
            and self._lines_cache_result is not None
            and crop.height == len(self._lines_cache_result)
        ):
            return self._lines_cache_result

        line_filters = self.get_line_filters()
        _base_background, background = self.background_colors
        strips: list[Strip] = []
        for y in crop.line_range:
            strip = self._render_line_with_selection(y, selection)
            if strip.cell_length != width:
                strip = strip.adjust_cell_length(width, self.visual_style.rich_style)
            for line_filter in line_filters:
                strip = strip.apply_filter(line_filter, background)
            if crop.column_span != (0, width):
                strip = strip.crop(crop.x, crop.x + crop.width)
            strips.append(strip)
        if selection is None:
            self._lines_cache_key = lines_key
            self._lines_cache_result = strips
        else:
            self._lines_cache_result = None
        return strips

    def render_line(self, y: int) -> Strip:
        return self._render_line_with_selection(y, self._safe_text_selection())

    def _safe_text_selection(self) -> Selection | None:
        try:
            return self.text_selection
        except NoScreen:
            return None

    def _render_line_with_selection(self, y: int, selection: Selection | None) -> Strip:
        if not self.is_attached:
            return Strip.blank(max(0, self.size.width)).apply_offsets(0, y)

        width = self.content_region.width
        if width <= 0:
            return Strip.blank(0)

        cache_key = (y, width)
        cached = self._strip_cache.get(cache_key)
        if cached is not None:
            strip = cached
        else:
            strip = self._render_base_line(y, width)
            self._strip_cache[cache_key] = strip

        if selection is not None:
            span = self._code_visual_span(selection, y)
            if span is not None:
                strip = self._apply_selection_highlight(strip, span)

        return strip.apply_offsets(0, y)

    def _code_visual_span(self, selection: Selection, y: int) -> tuple[int, int] | None:
        """Translate visual selection offsets to the code area of a row."""
        span = selection.get_span(y)
        if span is None:
            return None
        start_x, end_x = span
        code_start = self._gutter_width
        if end_x != -1 and end_x <= code_start:
            return None
        return max(start_x, code_start), end_x

    def _render_base_line(self, y: int, width: int) -> Strip:
        """Render a unified diff row without selection or offset metadata."""

        visual_style = self.visual_style
        rich_style = visual_style.rich_style
        if y < 0 or y >= self._total_lines:
            return Strip.blank(width, rich_style)

        code_width = max(1, width - self._gutter_width)
        entry = self._code_lines[y]

        if is_ellipsis_sentinel(entry):
            return self._render_ellipsis(width)

        segments: list[Segment] = []
        for content in (self._number_factory_a(y), self._number_factory_b(y)):
            segments.extend(content.render_segments(visual_style))
        annotation = self._annotation_factory(y)
        # Unified annotation factories emit " {symbol} ". Cropping to width 1
        # keeps the leading spacer and hides the symbol, matching the old
        # CSS-collapsed `.annotations { width: 1; }` behavior.
        segments.extend(
            Strip(annotation.render_segments(visual_style), cell_length=annotation.cell_length)
            .crop(0, self._annotation_width)
            ._segments
        )

        if entry is None:
            segments.append(Segment(" " * code_width, rich_style))
        else:
            line: Content = entry
            line_style_raw = self._line_styles[y] if y < len(self._line_styles) else ""
            # ``render_lines`` sets ``_frame_color_system`` for the hot path;
            # fall back to a live fetch for direct ``render_line`` callers.
            color_system = self._frame_color_system or str(self.app.console.color_system)
            line_style = line_style_for_color_system(
                line_style_raw if isinstance(line_style_raw, str) else "", color_system
            )
            if line.cell_length < code_width:
                line = line.pad_right(code_width - line.cell_length)
            line = line.stylize_before(line_style)
            strip = Strip(line.render_segments(visual_style), cell_length=line.cell_length)
            segments.extend(strip.crop(0, code_width)._segments)

        return Strip(segments).adjust_cell_length(width, rich_style)

    def _apply_selection_highlight(self, strip: Strip, span: tuple[int, int]) -> Strip:
        """Apply the screen selection style to the selected columns in a row."""
        start_x, end_x = span
        selection_style = normalize_selection_rich_style(self.screen.get_component_rich_style("screen--selection"))
        segments: list[Segment] = []
        pos = 0
        for segment in strip._segments:
            text, style, control = segment
            seg_start = pos
            seg_end = pos + len(text)
            actual_end = seg_end if end_x == -1 else min(seg_end, end_x)

            if actual_end > seg_start and start_x < seg_end and seg_start < (seg_end if end_x == -1 else end_x):
                sel_start = max(0, start_x - seg_start)
                sel_end = len(text) if end_x == -1 else min(len(text), end_x - seg_start)
                if sel_start > 0:
                    segments.append(Segment(text[:sel_start], style, control))
                combined = style + selection_style if style else selection_style
                segments.append(Segment(text[sel_start:sel_end], combined, control))
                if sel_end < len(text):
                    segments.append(Segment(text[sel_end:], style, control))
            else:
                segments.append(segment)
            pos = seg_end
        return Strip(segments, strip.cell_length)

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        """Return selected code text from the flattened visual diff rows."""
        text = "\n".join(self._plain_code_lines())
        return self._code_selection(selection).extract(text), "\n"

    def _code_selection(self, selection: Selection) -> Selection:
        """Translate visual selection columns into code-text columns."""

        def offset_to_code(offset: Offset | None) -> Offset | None:
            if offset is None:
                return None
            return Offset(max(0, offset.x - self._gutter_width), offset.y)

        return Selection(offset_to_code(selection.start), offset_to_code(selection.end))

    def _plain_code_lines(self) -> list[str]:
        """Build plain code rows for Textual selection extraction."""
        if self._plain_code_lines_cache is not None:
            return self._plain_code_lines_cache

        lines: list[str] = []
        for entry in self._code_lines:
            # Match CodeColumn selection behavior: non-code rows preserve row
            # boundaries but do not copy visual hatch/context-break glyphs.
            if is_ellipsis_sentinel(entry) or entry is None:
                lines.append("")
            else:
                line: Content = entry
                lines.append(line.plain)
        self._plain_code_lines_cache = lines
        return lines

    def _render_ellipsis(self, width: int) -> Strip:
        visual_style = self.visual_style
        rich_style = visual_style.rich_style
        pad_left = max(0, (width - 1) // 2)
        pad_right = max(0, width - 1 - pad_left)
        line = Content(" " * pad_left) + Content.styled("⋮", "$text-primary bold") + Content(" " * pad_right)
        line = line.stylize_before(visual_style)
        return Strip(line.render_segments(visual_style), cell_length=width).adjust_cell_length(width, rich_style)
