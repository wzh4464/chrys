# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for non-selectable sidebar empty-state prompts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.widgets.sidebar import buddy as buddy_module
from chrys.app.tui.widgets.sidebar import context as context_module
from chrys.app.tui.widgets.sidebar import tasks as tasks_module
from chrys.app.tui.widgets.sidebar import toc as toc_module
from chrys.app.tui.widgets.sidebar.buddy import BuddyPanel
from chrys.app.tui.widgets.sidebar.context import ContextPanel
from chrys.app.tui.widgets.sidebar.tasks import TasksPanel
from chrys.app.tui.widgets.sidebar.toc import ConversationToc
from chrys.foundation.config.settings import Settings
from chrys.foundation.i18n import MessageRef


class SidebarEmptyStatesApp(App):
    def compose(self) -> ComposeResult:
        yield ConversationToc()
        yield TasksPanel()
        yield BuddyPanel()


@pytest.mark.asyncio
async def test_sidebar_empty_state_prompts_are_not_selectable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chrys.app.tui.widgets.sidebar.buddy.get_companion", lambda: None)

    async with SidebarEmptyStatesApp().run_test(size=(80, 40)) as pilot:
        await pilot.pause()

        prompts = [
            pilot.app.query_one(".toc-empty", Static),
            pilot.app.query_one("#tasks-empty", Static),
            pilot.app.query_one(".buddy-empty", Static),
        ]
        assert all(not prompt.allow_select for prompt in prompts)
        assert pilot.app.query_one("#tasks-checklist", Static).allow_select


@pytest.mark.asyncio
async def test_buddy_localized_status_treats_translation_markup_as_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MarkupLocalizer:
        effective_locale = "zh-Hans"

        def render(self, reference: MessageRef) -> str:
            if reference.definition is buddy_module._BUDDY_CLICK_TO_PET:
                return "[red]x[/red]"
            return reference.definition.fallback

    controller = LocaleController(
        Settings(locale="zh-Hans"),
        localizer=MarkupLocalizer(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("chrys.app.tui.widgets.sidebar.buddy.get_companion", lambda: None)
    monkeypatch.setattr("chrys.app.features.buddy.config.is_muted", lambda: False)

    class MarkupBuddyApp(App):
        def compose(self) -> ComposeResult:
            yield BuddyPanel(locale_controller=controller)

    async with MarkupBuddyApp().run_test() as pilot:
        panel = pilot.app.query_one(BuddyPanel)
        panel._update_status(buddy_module._BUDDY_CLICK_TO_PET.bind(), style="green")

        assert str(panel.query_one("#buddy-status", Static).render()) == "[red]x[/red]"


@pytest.mark.asyncio
async def test_sidebar_static_chrome_treats_translation_markup_as_literal() -> None:
    markup_definitions = (
        tasks_module._TASKS_EMPTY,
        toc_module._TOC_EMPTY,
        context_module._CONTEXT_USAGE,
    )

    class MarkupLocalizer:
        effective_locale = "zh-Hans"

        def render(self, reference: MessageRef) -> str:
            if reference.definition in markup_definitions:
                return "[red]x[/red]"
            return reference.definition.fallback

    controller = LocaleController(
        Settings(locale="zh-Hans"),
        localizer=MarkupLocalizer(),  # type: ignore[arg-type]
    )

    class MarkupSidebarApp(App):
        def compose(self) -> ComposeResult:
            yield ConversationToc(locale_controller=controller)
            yield TasksPanel(locale_controller=controller)
            yield ContextPanel(locale_controller=controller)

    async with MarkupSidebarApp().run_test(size=(80, 40)) as pilot:
        await pilot.pause()
        for selector in (".toc-empty", "#tasks-empty", "#ctx-usage-label"):
            rendered = pilot.app.query_one(selector, Static).render()
            assert str(rendered) == "[red]x[/red]", selector
            assert not getattr(rendered, "spans", []), selector


@pytest.mark.asyncio
async def test_buddy_level_escapes_translation_markup_but_keeps_accent_styling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MarkupLocalizer:
        effective_locale = "zh-Hans"

        def render(self, reference: MessageRef) -> str:
            if reference.definition is buddy_module._BUDDY_LEVEL:
                return "[red]3 级[/red]"
            return reference.definition.fallback

    controller = LocaleController(
        Settings(locale="zh-Hans"),
        localizer=MarkupLocalizer(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr("chrys.app.tui.widgets.sidebar.buddy.get_companion", lambda: None)
    monkeypatch.setattr("chrys.app.features.buddy.config.is_muted", lambda: False)

    class LevelBuddyApp(App):
        def compose(self) -> ComposeResult:
            yield BuddyPanel(locale_controller=controller)

    async with LevelBuddyApp().run_test() as pilot:
        panel = pilot.app.query_one(BuddyPanel)
        # The stand-in companion below is not renderable as a sprite: stop the
        # animation timer so a tick landing on a slow worker cannot crash the
        # app before the level label is inspected.
        panel._timer.stop()
        panel.set_reactive(
            BuddyPanel.companion,
            SimpleNamespace(
                level=3,
                name="Momo",
                species=SimpleNamespace(value="cat"),
                rarity=SimpleNamespace(value="rare"),
                personality="brave",
                shiny=False,
            ),
        )
        panel._update_info()

        rendered = panel.query_one("#buddy-level", Static).render()
        # Escaped translation renders its markup-looking text literally...
        assert str(rendered) == "[red]3 级[/red]"
        # ...while the code-side bold/accent chrome styling survives as spans.
        assert getattr(rendered, "spans", [])
