# Copyright (c) 2026 Chrys. All rights reserved.

"""Golden render tests for :class:`DiffView` (unified flat path).

Locks in the visible output for representative diffs so the Tier 3 hoist of
``self.app.console.color_system`` out of the per-row code path cannot
silently change a row's text or per-segment style.
"""

from __future__ import annotations

import pytest
from rich.color import ColorSystem
from textual.app import App, ComposeResult
from textual.geometry import Region

from chrys.app.tui.widgets.diff_view import DiffView
from chrys.app.tui.widgets.diff_view.code import CodeColumn
from chrys.app.tui.widgets.diff_view.unified import UnifiedDiffLines


def _row_text(strip) -> str:
    return "".join(seg.text for seg in strip._segments).rstrip()


def _render_strips_match(a, b) -> bool:
    """Two strip lists are byte-identical if every segment text and style match."""
    if len(a) != len(b):
        return False
    for sa, sb in zip(a, b, strict=True):
        if sa.cell_length != sb.cell_length:
            return False
        if [(s.text, s.style) for s in sa._segments] != [(s.text, s.style) for s in sb._segments]:
            return False
    return True


class _UnifiedDiffApp(App):
    def __init__(self, before: str, after: str) -> None:
        super().__init__()
        self._before = before
        self._after = after

    def compose(self) -> ComposeResult:
        diff = DiffView("file.py", "file.py", self._before, self._after)
        diff.split = False
        diff.auto_height = True
        diff.show_scrollbars = False
        yield diff


class _SplitDiffApp(App):
    def __init__(self, before: str, after: str) -> None:
        super().__init__()
        self._before = before
        self._after = after

    def compose(self) -> ComposeResult:
        diff = DiffView("file.py", "file.py", self._before, self._after)
        diff.split = True
        diff.auto_height = True
        yield diff


@pytest.mark.asyncio
async def test_unified_diff_row_text_contains_every_changed_line() -> None:
    """Every added line shows up in the unified row stream with a + marker."""
    before = "alpha\nbeta\n"
    after = "alpha\nbeta\ngamma\n"
    async with _UnifiedDiffApp(before, after).run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        flat = pilot.app.query_one(UnifiedDiffLines)
        strips = flat.render_lines(Region(0, 0, flat.size.width, flat.size.height))
        joined = "\n".join(_row_text(s) for s in strips)
        # The added "gamma" line appears with a "+" annotation.
        assert "gamma" in joined
        assert "+" in joined


@pytest.mark.asyncio
async def test_unified_diff_render_is_deterministic_across_repeats() -> None:
    """Rendering the same diff twice produces byte-identical strips."""
    before = "alpha\nbeta\n"
    after = "alpha\nBETA\ngamma\n"
    async with _UnifiedDiffApp(before, after).run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        flat = pilot.app.query_one(UnifiedDiffLines)
        crop = Region(0, 0, flat.size.width, flat.size.height)
        first = flat.render_lines(crop)
        # Drop per-row strip cache to force the full render to re-run.
        flat._invalidate_render_cache()
        second = flat.render_lines(crop)
        assert _render_strips_match(first, second), "two renders of identical diff diverged"


@pytest.mark.asyncio
async def test_unified_diff_color_system_hoist_matches_uncached_path() -> None:
    """Render with ``_frame_color_system`` populated must equal the fallback path.

    Regression test for the Tier 3 hoist: when ``_frame_color_system`` is set
    (normal render frame) versus when it's reset (test helper / generic fallback),
    a row's segments and styles must be identical.

    Both passes are compared at the ``render_line`` level — ``render_lines``
    additionally applies Textual line filters, ``adjust_cell_length``, and
    a crop, which would introduce pipeline asymmetry independent of the
    cached vs. live-fetch color-system behavior under test here.
    """
    before = "same\n"
    after = "added\n"
    async with _UnifiedDiffApp(before, after).run_test(size=(80, 20)) as pilot:
        await pilot.pause()
        # Force 256-color so the line_style_for_color_system map is engaged
        # and the downgrade actually changes output.
        pilot.app.console._color_system = ColorSystem.EIGHT_BIT

        flat = pilot.app.query_one(UnifiedDiffLines)
        flat._invalidate_render_cache()  # clears _frame_color_system back to ""

        crop = Region(0, 0, flat.size.width, flat.size.height)
        row_count = min(crop.height, flat._total_lines)

        # Pass 1: render_lines populates _frame_color_system and the per-row
        # strip cache.  Read row-by-row through ``render_line`` so output is
        # the bare ``_render_base_line`` strip (no filter, no crop).
        flat.render_lines(crop)
        assert flat._frame_color_system == "256"
        cached_rows = [flat.render_line(y) for y in range(row_count)]

        # Pass 2: drop the per-row strip cache *and* the frame color system,
        # then call render_line directly — exercises the live-fetch fallback
        # in ``_render_base_line``.
        flat._invalidate_render_cache()
        fallback_rows = [flat.render_line(y) for y in range(row_count)]

        for y, (a, b) in enumerate(zip(cached_rows, fallback_rows, strict=True)):
            assert a.cell_length == b.cell_length, f"row {y} cell length differs"
            assert [(s.text, s.style) for s in a._segments] == [(s.text, s.style) for s in b._segments], (
                f"row {y} segments differ between cached and fallback color_system paths"
            )


@pytest.mark.asyncio
async def test_split_diff_code_column_renders_both_sides() -> None:
    """Split diff: both CodeColumn widgets emit the expected file content."""
    before = "alpha\nbeta\n"
    after = "alpha\nBETA\n"
    async with _SplitDiffApp(before, after).run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        columns = list(pilot.app.query(CodeColumn))
        assert len(columns) == 2, f"expected 2 code columns in split mode, got {len(columns)}"

        for col in columns:
            col._invalidate_render_cache()
            col._frame_scroll_x = 0
            col._frame_scroll_y = 0
            col._frame_width = col._get_render_width()
            col._frame_selection = None

        left_rows = [_row_text(columns[0].render_line(y)) for y in range(len(columns[0].code_lines))]
        right_rows = [_row_text(columns[1].render_line(y)) for y in range(len(columns[1].code_lines))]

        assert any("alpha" in row for row in left_rows)
        assert any("beta" in row for row in left_rows)
        assert any("alpha" in row for row in right_rows)
        assert any("BETA" in row for row in right_rows)


@pytest.mark.asyncio
async def test_code_column_color_system_hoist_matches_uncached_path() -> None:
    """``CodeColumn.render_line`` must produce the same output via frame-cache
    and via the live-fetch fallback once ``_frame_color_system`` is reset.

    Both passes go through ``render_line`` so the comparison is invariant to
    the line-filter / ``adjust_cell_length`` / crop work that ``render_lines``
    additionally performs (and that would otherwise show a false asymmetry
    under forced ``EIGHT_BIT`` color mode).
    """
    before = "same\n"
    after = "added\n"
    async with _SplitDiffApp(before, after).run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        pilot.app.console._color_system = ColorSystem.EIGHT_BIT

        column = pilot.app.query_one(CodeColumn)
        crop = Region(0, 0, column.size.width, column.size.height)
        row_count = min(crop.height, len(column.code_lines))

        # Pass 1: render_lines populates _frame_color_system, the per-frame
        # geometry fields, and the per-row strip cache.  Read row-by-row
        # via render_line — returns the cached bare-row strip.
        column._invalidate_render_cache()
        column.render_lines(crop)
        assert column._frame_color_system == "256"
        cached_rows = [column.render_line(y) for y in range(row_count)]

        # Pass 2: drop the per-row strip cache and _frame_color_system.
        # ``_invalidate_render_cache`` preserves the geometry fields
        # (_frame_scroll_*, _frame_width, _frame_selection), so render_line
        # still has the per-frame state it needs — but it must now live-fetch
        # ``self.app.console.color_system`` instead of reading the cache.
        column._invalidate_render_cache()
        fallback_rows = [column.render_line(y) for y in range(row_count)]

        for y, (a, b) in enumerate(zip(cached_rows, fallback_rows, strict=True)):
            assert a.cell_length == b.cell_length, f"row {y} cell length differs"
            assert [(s.text, s.style) for s in a._segments] == [(s.text, s.style) for s in b._segments], (
                f"row {y} diverged between cached frame and fallback paths"
            )
