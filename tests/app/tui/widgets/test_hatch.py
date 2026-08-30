# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the shared hatch style helpers."""

from __future__ import annotations

from rich.cells import cell_len
from rich.color import Color as RichColor
from rich.style import Style

from chrys.app.tui.widgets.hatch import hatch_text_style, hatched_text_line


def test_hatch_text_style_blends_truecolor_themes() -> None:
    style = hatch_text_style({"background": "#1C1C1C", "foreground": "#EEEEEE"})

    assert style.color == RichColor.parse("#3b3b3b")
    assert not style.dim


def test_hatch_text_style_blends_over_black_when_background_is_ansi() -> None:
    """An ANSI background must not force the bright dim fallback.

    The stylesheet path resolves ``$foreground 15%`` over an ``ansi_default``
    background to a dark blend; the Rich path has to match it.
    """
    style = hatch_text_style({"background": "ansi_default", "foreground": "#EEEEEE"})

    assert style.color == RichColor.parse("#232323")
    assert not style.dim


def test_hatch_text_style_dims_when_foreground_is_not_blendable() -> None:
    assert hatch_text_style({"background": "#1C1C1C", "foreground": "ansi_white"}) == Style(dim=True)
    assert hatch_text_style({"background": "#1C1C1C", "foreground": "not-a-color"}) == Style(dim=True)
    assert hatch_text_style({}) == Style(dim=True)


def test_hatched_text_line_truncates_a_label_instead_of_dropping_it() -> None:
    line = hatched_text_line(8, "No trajectory data", hatch_style=Style(dim=True), label_style=Style(bold=True))

    assert cell_len(line.plain) == 8
    assert line.plain.endswith("...")
    assert line.plain != "╲" * 8
