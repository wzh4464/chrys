# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the Skills configuration panel."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static

from chrys.app.tui.screens.agents.panels.skills import SkillDirCard, SkillsConfigPanel
from chrys.app.tui.screens.agents.skill_paths import skill_path_duplicate_match_description
from chrys.foundation.i18n import Localizer
from chrys.service.profiles.agents.schema import SkillsConfig


class _SkillsPanelApp(App):
    def compose(self) -> ComposeResult:
        yield Static("placeholder")


async def _wait_for_skill_cards(panel: SkillsConfigPanel, pilot, count: int) -> list[SkillDirCard]:
    cards: list[SkillDirCard] = []
    for _ in range(20):
        cards = list(panel.query(SkillDirCard))
        if len(cards) >= count:
            await pilot.pause()
            return list(panel.query(SkillDirCard))
        await pilot.pause(0.05)
    raise AssertionError(f"Expected at least {count} skill directory cards to be mounted")


async def _add_skill_dir(panel: SkillsConfigPanel, pilot, *, count: int = 1) -> SkillDirCard:
    panel.query_one("#sk-add-btn").press()
    cards = await _wait_for_skill_cards(panel, pilot, count)
    return cards[count - 1]


def _set_fake_platform(monkeypatch: pytest.MonkeyPatch, os_name: str) -> None:
    fake_platform = SimpleNamespace(
        is_macos=os_name == "macos",
        is_windows=os_name == "windows",
        is_linux=os_name == "linux",
    )
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)


def test_skill_path_normalization_hint_keeps_english_and_localizes_chinese(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_fake_platform(monkeypatch, "linux")
    assert skill_path_duplicate_match_description() == "paths are matched after normalization"
    assert skill_path_duplicate_match_description(Localizer("zh-Hans").render) == "路径规范化后进行匹配"


@pytest.mark.asyncio
async def test_add_skill_directory_inserts_new_card_before_existing_paths() -> None:
    """Adding another directory should put the blank row first without wiping existing input."""

    app = _SkillsPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SkillsConfigPanel(SkillsConfig())
        await app.mount(panel)
        await pilot.pause()

        first = await _add_skill_dir(panel, pilot)
        first.query_one("#sk-path-0", Input).value = "/tmp/skills-one"

        await _add_skill_dir(panel, pilot, count=2)

        cards = list(panel.query(SkillDirCard))
        assert cards[0].query_one("#sk-path-0", Input).value == ""
        assert cards[1].query_one("#sk-path-1", Input).value == "/tmp/skills-one"
        assert panel.get_config().paths == ["/tmp/skills-one"]


@pytest.mark.asyncio
async def test_remove_skill_directory_preserves_edited_remaining_paths() -> None:
    """Removing a row should snapshot live edits before rebuilding the remaining cards."""

    app = _SkillsPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SkillsConfigPanel(SkillsConfig(paths=["/tmp/old-one", "/tmp/old-two"]))
        await app.mount(panel)
        await pilot.pause()

        cards = await _wait_for_skill_cards(panel, pilot, 2)
        cards[1].query_one("#sk-path-1", Input).value = "/tmp/new-two"
        cards[0].query_one("#sk-delete-btn-0").press()
        await pilot.pause()

        cards = await _wait_for_skill_cards(panel, pilot, 1)
        assert len(cards) == 1
        assert cards[0].query_one("#sk-path-0", Input).value == "/tmp/new-two"
        assert panel.get_config().paths == ["/tmp/new-two"]


@pytest.mark.parametrize(
    "paths",
    [
        ["/tmp/skills", "/tmp/skills/"],
        ["/tmp/skills", "  /tmp/skills  "],
        ["/tmp/skills", "\\tmp\\skills"],
        [str(Path.home() / "skills"), "~/skills"],
        ["/tmp/skills", "/TMP/Skills"],
    ],
)
@pytest.mark.parametrize("os_name", ["macos", "windows"])
@pytest.mark.asyncio
async def test_validate_rejects_duplicate_skill_directory_paths_on_case_insensitive_platforms(
    paths: list[str],
    os_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Duplicate skill directories should block Save instead of silently repeating discovery."""

    _set_fake_platform(monkeypatch, os_name)
    app = _SkillsPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SkillsConfigPanel(SkillsConfig(paths=paths))
        await app.mount(panel)
        await pilot.pause()

        errors = panel.validate()

    assert (
        f"Skill Directory 2: '{paths[1].strip()}' duplicates Skill Directory 1 "
        "(paths are matched case-insensitively after normalization)."
    ) in errors


@pytest.mark.asyncio
async def test_validate_allows_case_differing_skill_directory_paths_on_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux skill directories should remain case-sensitive during duplicate validation."""

    _set_fake_platform(monkeypatch, "linux")
    app = _SkillsPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SkillsConfigPanel(SkillsConfig(paths=["/tmp/skills", "/TMP/Skills"]))
        await app.mount(panel)
        await pilot.pause()

        errors = panel.validate()

    assert errors == []


@pytest.mark.asyncio
async def test_validate_still_normalizes_duplicate_skill_directory_paths_on_linux(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux duplicate validation should normalize paths without case folding."""

    _set_fake_platform(monkeypatch, "linux")
    app = _SkillsPanelApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SkillsConfigPanel(SkillsConfig(paths=["/tmp/skills", "/tmp/skills/"]))
        await app.mount(panel)
        await pilot.pause()

        errors = panel.validate()

    assert (
        "Skill Directory 2: '/tmp/skills/' duplicates Skill Directory 1 (paths are matched after normalization)."
    ) in errors
