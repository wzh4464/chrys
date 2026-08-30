# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the mount-race-tolerant Select widget."""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Select as TextualSelect
from textual.widgets._select import SelectOverlay

from chrys.app.tui.widgets import Select

_OPTIONS = [("Alpha", "a"), ("Beta", "b")]


class _SelectApp(App):
    def compose(self) -> ComposeResult:
        yield Select(_OPTIONS, allow_blank=True, value="a")


@pytest.mark.asyncio
async def test_select_populates_overlay_on_normal_mount() -> None:
    """The defensive overrides must not disturb the ordinary mount path."""
    app = _SelectApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        select = app.query_one(Select)
        overlay = select.query_one(SelectOverlay)

        # allow_blank adds the blank prompt entry ahead of the options.
        assert overlay.option_count == len(_OPTIONS) + 1
        assert select.value == "a"
        assert isinstance(select, TextualSelect)


@pytest.mark.asyncio
async def test_select_survives_children_missing_at_mount_time() -> None:
    """Prune racing a fresh mount must not crash the app.

    ``Widget.mount()`` silently skips mounting children while the widget is
    being pruned, but the already-queued ``Mount`` event still dispatches, so
    ``_on_mount`` runs against a Select with no ``SelectOverlay``. Textual's
    base class crashes the whole app with ``NoMatches`` there; the chrys
    subclass defers instead. Re-running ``_on_mount``'s body after stripping
    the children reproduces that state deterministically.
    """
    app = _SelectApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        select = app.query_one(Select)
        await select.remove_children()

        # Base Select raises NoMatches from _setup_options_renderables here
        # (_watch_value is guarded upstream); ours must survive both.
        select._setup_options_renderables()
        select._watch_value("b")
        await pilot.pause()

        assert select._value == "b"
