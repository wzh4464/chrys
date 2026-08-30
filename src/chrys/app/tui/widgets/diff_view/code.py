# Copyright (c) 2026 Chrys. All rights reserved.

"""Virtualized code column with horizontal scrolling and LRU strip caching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NewType, TypeIs

from rich.segment import Segment
from rich.style import Style as RichStyle
from textual.cache import LRUCache
from textual.content import Content
from textual.dom import NoScreen
from textual.geometry import Region, Size
from textual.scroll_view import ScrollView
from textual.strip import Strip

from chrys.app.tui.support.gc_freeze import detach_lru_cache, renew_lru_cache
from chrys.app.tui.widgets import HATCH_GLYPH, HATCH_STYLE
from chrys.app.tui.widgets.diff_view.styles import line_style_for_color_system

if TYPE_CHECKING:
    from textual.selection import Selection

    from chrys.app.tui.widgets.diff_view.annotations import AnnotationsColumn

EllipsisSentinel = NewType("EllipsisSentinel", object)
ELLIPSIS_SENTINEL = EllipsisSentinel(object())
type CodeLine = Content | EllipsisSentinel | None
type LineStyle = str | EllipsisSentinel
type DiffAnnotation = Literal["+", "-", "/", " "]


def is_ellipsis_sentinel(value: object) -> TypeIs[EllipsisSentinel]:
    """Narrow the identity-only context-break sentinel."""
    return value is ELLIPSIS_SENTINEL


class CodeColumn(ScrollView):
    """Virtualized code column with horizontal scrollbar.

    Renders one line at a time via render_line(y). Textual only calls
    render_line for visible viewport lines, giving us virtualization.
    """

    ALLOW_SELECT = True

    DEFAULT_CSS = """
    CodeColumn {
        overflow: auto auto;
        scrollbar-size: 1 2;
        scrollbar-gutter: stable;
        width: 1fr;
        height: 1fr;
    }
    """

    def __init__(
        self,
        code_lines: list[CodeLine],
        line_styles: list[LineStyle],
        max_width: int,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.code_lines = code_lines
        self.line_styles = line_styles
        self.max_width = max_width
        self.scroll_sync: CodeColumn | None = None
        self.annotation_columns: list[AnnotationsColumn] = []
        self._syncing = False
        self._strip_cache: LRUCache[int, Strip] = LRUCache(maxsize=2000)
        self._cache_scroll_x: int = -1
        self._cache_width: int = -1
        # Per-frame cached values (set in render_lines, used in render_line)
        self._frame_scroll_x: int = 0
        self._frame_scroll_y: int = 0
        self._frame_width: int = 0
        self._frame_content_width: int = 0
        self._frame_selection: Selection | None = None
        self._frame_blank: Strip = Strip.blank(0)
        # Hoisted out of ``render_line`` so per-row code doesn't traverse
        # ``self.app.console`` for the line-style downgrade map.
        self._frame_color_system: str = ""
        # Full-frame result cache — avoids calling render_line entirely
        # when the integer scroll position hasn't changed between frames
        self._lines_cache_key: tuple[int, int, int, int, int, int, int, int] = (-1, -1, -1, -1, -1, -1, -1, -1)
        self._lines_cache_result: list[Strip] | None = None
        self._cache_theme = ""
        self._cache_color_system = ""

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

    def on_mount(self) -> None:
        self.virtual_size = Size(self.max_width, len(self.code_lines))

    def _invalidate_render_cache(self) -> None:
        self._strip_cache.clear()
        self._cache_scroll_x = -1
        self._cache_width = -1
        self._cache_theme = ""
        self._cache_color_system = ""
        self._frame_blank = Strip.blank(0)
        self._lines_cache_key = (-1, -1, -1, -1, -1, -1, -1, -1)
        self._lines_cache_result = None
        # Reset the frame color system too — otherwise a stale value from the
        # previous theme would mask the live fetch in ``render_line``.
        self._frame_color_system = ""

    def notify_style_update(self) -> None:
        """Clear rendered strips when theme/CSS variables change."""
        super().notify_style_update()
        self._invalidate_render_cache()
        self.refresh()

    def _get_render_width(self) -> int:
        """Width to render strips at.

        With scrollbar-gutter: stable, scrollbar space is always reserved.
        We render into the scrollable area; _styles_cache pads the
        remaining 1-char scrollbar column with the widget background,
        and the scrollbar widget draws on top.
        """
        return self.scrollable_content_region.width

    def render_lines(self, crop: Region) -> list[Strip]:
        if not self.is_attached:
            return [Strip.blank(crop.width) for _ in crop.line_range]

        # Pre-compute expensive values once per frame instead of per row.
        try:
            scroll_x, scroll_y = self.scroll_offset
            frame_sx = round(scroll_x)
            frame_sy = round(scroll_y)
            frame_w = self._get_render_width()
            frame_content_w = self.content_region.width
            frame_sel = self._safe_text_selection()
            frame_theme = self.app.theme
            frame_color_system = str(self.app.console.color_system)

            if frame_theme != self._cache_theme or frame_color_system != self._cache_color_system:
                self._invalidate_render_cache()
                self._cache_theme = frame_theme
                self._cache_color_system = frame_color_system

            # Fast path: if integer scroll position, width, and selection
            # state are all unchanged, the output is identical — return it
            # without calling any render_line.
            lines_key = (
                frame_sx,
                frame_sy,
                frame_w,
                frame_content_w,
                crop.x,
                crop.y,
                crop.width,
                crop.height,
            )
            if (
                lines_key == self._lines_cache_key
                and frame_sel is None
                and self._lines_cache_result is not None
                and crop.height == len(self._lines_cache_result)
            ):
                return self._lines_cache_result

            self._frame_scroll_x = frame_sx
            self._frame_scroll_y = frame_sy
            self._frame_width = frame_w
            self._frame_content_width = frame_content_w
            self._frame_selection = frame_sel
            self._frame_color_system = frame_color_system

            # Reuse blank strip when width hasn't changed
            if frame_w != self._cache_width:
                self._frame_blank = Strip.blank(frame_w, self.visual_style.rich_style)

            # Invalidate strip cache if horizontal scroll or width changed
            if frame_sx != self._cache_scroll_x or frame_w != self._cache_width:
                self._strip_cache.clear()
                self._cache_scroll_x = frame_sx
                self._cache_width = frame_w
        except Exception:
            return super().render_lines(crop)

        result = self._render_lines_direct(crop)

        if frame_sel is None:
            self._lines_cache_key = lines_key
            self._lines_cache_result = result
        else:
            self._lines_cache_result = None

        return result

    def _safe_text_selection(self) -> Selection | None:
        try:
            return self.text_selection
        except NoScreen:
            return None

    def _render_lines_direct(self, crop: Region) -> list[Strip]:
        """Render visible rows without Textual's generic style wrapper.

        CodeColumn has no border or padding; render_line already resolves the
        code row background and text styles.  Avoiding ``StylesCache.render``
        matters when the parent chat panel scrolls over a clipped diff: the
        crop changes every frame, so the generic cache has to walk every
        visible row even though the row-level strip cache is hot.
        """
        content_width = self._frame_content_width
        rich_style = self.visual_style.rich_style
        line_filters = self.get_line_filters()
        _base_background, background = self.background_colors
        strips: list[Strip] = []
        for y in crop.line_range:
            strip = self.render_line(y)
            if strip.cell_length != content_width:
                strip = strip.adjust_cell_length(content_width, rich_style)
            for line_filter in line_filters:
                strip = strip.apply_filter(line_filter, background)
            if crop.column_span != (0, content_width):
                strip = strip.crop(crop.x, crop.x + crop.width)
            strips.append(strip)
        return strips

    def render_line(self, y: int) -> Strip:
        if not self.is_attached:
            return Strip.blank(max(0, self._frame_width or self.size.width)).apply_offsets(0, y)

        content_y = self._frame_scroll_y + y
        width = self._frame_width

        if content_y >= len(self.code_lines):
            return self._frame_blank

        # Check strip cache — key is just content_y (scroll_x/width are
        # validated at the frame level in render_lines)
        selection = self._frame_selection
        use_cache = selection is None
        if use_cache:
            cached = self._strip_cache.get(content_y)
            if cached is not None:
                return cached

        # Cache miss — compute styles and render
        visual_style = self.visual_style
        rich_style = visual_style.rich_style
        scroll_x = self._frame_scroll_x

        entry = self.code_lines[content_y]

        # Ellipsis sentinel — always centered in viewport, no h-scroll
        if is_ellipsis_sentinel(entry):
            ellipsis_content = Content.styled("⋮", "$text-primary bold")
            pad_left = max(0, (width - 1) // 2)
            pad_right = max(0, width - 1 - pad_left)
            line = Content(" " * pad_left) + ellipsis_content + Content(" " * pad_right)
            line = line.stylize_before(visual_style)
            result = Strip(line.render_segments(visual_style), cell_length=width)
            if use_cache:
                self._strip_cache[content_y] = result
            return result

        # None entry (hatch pattern) — fills viewport, no h-scroll
        if entry is None:
            hatch_text = HATCH_GLYPH * max(width, 1)
            line = Content.styled(hatch_text, HATCH_STYLE)
            strip = Strip(line.render_segments(visual_style), cell_length=line.cell_length)
            result = strip.adjust_cell_length(width, rich_style)
            if use_cache:
                self._strip_cache[content_y] = result
            return result

        # Normal code line — render at full width, then crop for h-scroll
        line: Content = entry
        line_style_value = self.line_styles[content_y] if content_y < len(self.line_styles) else ""
        line_style = line_style_value if isinstance(line_style_value, str) else ""
        # ``render_lines`` sets ``_frame_color_system`` for the hot path; fall
        # back to a live fetch when this widget is driven outside of a frame
        # (test helpers, generic Textual fallback render path).
        color_system = self._frame_color_system or str(self.app.console.color_system)
        line_style = line_style_for_color_system(line_style, color_system)

        if selection is not None and (span := selection.get_span(content_y)):
            from textual.style import Style as VisualStyle

            selection_style = VisualStyle.from_rich_style(self.screen.get_component_rich_style("screen--selection"))
            start, end = span
            if end == -1:
                end = len(line)
            line = line.stylize(selection_style, start, end)

        # Pad to fill the full visible area so line style (green/red
        # background) extends to the edge even when viewport > max_width.
        pad_width = max(self.max_width, width)
        if line.cell_length < pad_width:
            line = line.pad_right(pad_width - line.cell_length)

        line = line.stylize_before(line_style).stylize_before(visual_style)

        x = 0
        meta = {"offset": (x, content_y)}
        segments = []
        for text, rs, _ in line.render_segments():
            if rs is not None:
                meta["offset"] = (x, content_y)
                segments.append(Segment(text, rs + RichStyle.from_meta(meta)))
            else:
                segments.append(Segment(text, rs))
            x += len(text)

        strip = Strip(segments, line.cell_length)
        strip = strip.crop(scroll_x, scroll_x + width)
        result = strip.adjust_cell_length(width, rich_style)
        if use_cache:
            self._strip_cache[content_y] = result
        return result

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        if self._syncing:
            return
        self._syncing = True
        try:
            if self.scroll_sync is not None and self.scroll_sync.scroll_offset.y != new_value:
                self.scroll_sync.scroll_y = new_value
            # Only refresh annotations when the visible content actually changes.
            if round(old_value) != round(new_value):
                for col in self.annotation_columns:
                    col.refresh()
        finally:
            self._syncing = False

    def watch_scroll_x(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_x(old_value, new_value)
        if self._syncing:
            return
        self._syncing = True
        try:
            if self.scroll_sync is not None and self.scroll_sync.scroll_offset.x != new_value:
                self.scroll_sync.scroll_x = new_value
        finally:
            self._syncing = False

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        lines = []
        for entry in self.code_lines:
            if is_ellipsis_sentinel(entry) or entry is None:
                lines.append("")
            else:
                lines.append(entry.plain)
        text = "\n".join(lines)
        return selection.extract(text), "\n"
