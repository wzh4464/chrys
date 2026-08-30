# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for shared config Add button styling hooks."""

from __future__ import annotations

import asyncio

import pytest
from textual.app import App, ComposeResult
from textual.color import Color
from textual.widgets import Input, Static

from chrys.app.tui.screens.agents.panels.mcp import MCPConfigPanel, MCPConnectionCard
from chrys.app.tui.screens.agents.panels.memory import MemoryConfigPanel
from chrys.app.tui.screens.agents.panels.skills import ExtChip, SkillDirCard, SkillsConfigPanel
from chrys.app.tui.screens.agents.panels.subagents import SubAgentsConfigPanel
from chrys.app.tui.screens.models.screen import ModelConfigScreen
from chrys.app.tui.theme import TUI_VARIABLE_DEFAULTS
from chrys.app.tui.widgets.buttons import ConfigActionButton, ConfigAddButton
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import MCPServerConfig, MemoryConfig, SkillsConfig, SubAgentsConfig
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.paths import SRC_ROOT

_CHRYS_CSS = SRC_ROOT / "chrys" / "app" / "tui" / "chrys.tcss"


class _ConfigAddButtonApp(App):
    def get_theme_variable_defaults(self) -> dict[str, str]:
        return dict(TUI_VARIABLE_DEFAULTS)

    def compose(self) -> ComposeResult:
        yield Static("placeholder")


class _ConfigAddButtonStyleApp(App):
    CSS_PATH = str(_CHRYS_CSS)

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return dict(TUI_VARIABLE_DEFAULTS)

    def on_mount(self) -> None:
        self.theme = "textual-light"

    def compose(self) -> ComposeResult:
        yield ConfigAddButton("+ Add", id="add-btn")


class _FakeClick:
    def __init__(self) -> None:
        self.stopped = False
        self.prevented = False

    def stop(self) -> None:
        self.stopped = True

    def prevent_default(self) -> None:
        self.prevented = True


class _ConfigAddButtonFocusApp(App):
    def get_theme_variable_defaults(self) -> dict[str, str]:
        return dict(TUI_VARIABLE_DEFAULTS)

    def compose(self) -> ComposeResult:
        yield Input(id="focus-target")
        yield ConfigActionButton("+ Add", id="action-btn")


@pytest.fixture(autouse=True)
def _isolate_chrys_config_dir(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_platform = type(
        "P",
        (),
        {"config_dir": tmp_path, "is_macos": True, "is_windows": False, "is_linux": False},
    )()
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)


def _agent_registry() -> AgentProfileRegistry:
    registry = AgentProfileRegistry()
    registry.load_builtins()
    return registry


async def _wait_for_mcp_cards(panel: MCPConfigPanel, pilot, count: int) -> list[MCPConnectionCard]:
    cards: list[MCPConnectionCard] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 20
    while loop.time() <= deadline:
        cards = list(panel.query(MCPConnectionCard))
        if len(cards) >= count:
            await pilot.pause()
            return list(panel.query(MCPConnectionCard))
        await pilot.pause(0.05)
    raise AssertionError(f"Expected at least {count} MCP cards to be mounted")


async def _wait_for_skill_cards(panel: SkillsConfigPanel, pilot, count: int) -> list[SkillDirCard]:
    cards: list[SkillDirCard] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 20
    while loop.time() <= deadline:
        cards = list(panel.query(SkillDirCard))
        if len(cards) >= count:
            await pilot.pause()
            return list(panel.query(SkillDirCard))
        await pilot.pause(0.05)
    raise AssertionError(f"Expected at least {count} skill cards to be mounted")


def _assert_shared_add_button(widget, selector: str) -> None:
    assert widget.query_one(selector, ConfigAddButton).has_class("config-add-button")


def _assert_compact_add_button(widget, selector: str) -> None:
    button = widget.query_one(selector, ConfigAddButton)
    assert button.has_class("config-add-button")
    assert button.has_class("config-add-button-compact")


@pytest.mark.asyncio
async def test_model_config_add_buttons_use_shared_style_class() -> None:
    registry = ModelProfileRegistry()
    profile = ModelProfile(id="model-a", name="Model A", model_id="gpt-test")
    registry.register(profile)

    app = _ConfigAddButtonApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = ModelConfigScreen(registry, global_default_profile_id=profile.id)
        await app.push_screen(screen)
        await pilot.pause()

        _assert_compact_add_button(screen, "#mc-hadd")
        _assert_compact_add_button(screen, "#mc-oadd")


@pytest.mark.asyncio
async def test_agent_config_add_buttons_use_shared_style_class() -> None:
    app = _ConfigAddButtonApp()
    async with app.run_test(size=(120, 40)) as pilot:
        sub_agents = SubAgentsConfigPanel(
            sub_agents_config=SubAgentsConfig(),
            registry=_agent_registry(),
            current_profile_name="Code",
        )
        skills = SkillsConfigPanel(SkillsConfig())
        memory = MemoryConfigPanel(MemoryConfig())
        mcp = MCPConfigPanel([])

        for panel in (sub_agents, skills, memory, mcp):
            await app.mount(panel)
        await pilot.pause()

        _assert_shared_add_button(sub_agents, "#sa-add-btn")
        _assert_shared_add_button(skills, "#sk-add-btn")
        _assert_shared_add_button(memory, "#mem-add-file")
        _assert_shared_add_button(memory, "#mem-add-folder")
        _assert_shared_add_button(mcp, "#mcp-add-btn")


@pytest.mark.asyncio
async def test_mcp_transport_add_buttons_use_shared_style_class() -> None:
    app = _ConfigAddButtonApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = MCPConfigPanel(
            [
                MCPServerConfig(name="stdio-server", transport="stdio", command="python"),
                MCPServerConfig(name="http-server", transport="http", url="https://example.test/mcp"),
            ]
        )
        await app.mount(panel)

        cards = await _wait_for_mcp_cards(panel, pilot, 2)

        _assert_compact_add_button(cards[0], "#mcp-eadd-0")
        _assert_compact_add_button(cards[1], "#mcp-hadd-1")


@pytest.mark.asyncio
async def test_config_action_button_mouse_click_drops_focus_after_press() -> None:
    app = _ConfigAddButtonFocusApp()
    async with app.run_test() as pilot:
        button = app.query_one("#action-btn", ConfigActionButton)
        button.focus()
        button.mouse_hover = True
        await pilot.pause()

        assert button.has_focus is True
        assert button.mouse_hover is True

        click = _FakeClick()
        await button._on_click(click)
        await pilot.pause()

        assert click.stopped is True
        assert click.prevented is True
        assert button.has_focus is False
        assert app.screen.focused is None
        assert button.mouse_hover is False


@pytest.mark.asyncio
async def test_config_action_button_programmatic_press_preserves_focus() -> None:
    app = _ConfigAddButtonFocusApp()
    async with app.run_test() as pilot:
        button = app.query_one("#action-btn", ConfigActionButton)
        button.focus()
        button.mouse_hover = True
        await pilot.pause()

        button.press()
        await pilot.pause()

        assert button.has_focus is True
        assert app.screen.focused is button
        assert button.mouse_hover is True


@pytest.mark.asyncio
async def test_config_add_button_uses_theme_secondary_foreground_on_light_theme() -> None:
    app = _ConfigAddButtonStyleApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        button = app.query_one("#add-btn", ConfigAddButton)

        assert button.styles.color == Color.parse(app.current_theme.secondary)


@pytest.mark.asyncio
async def test_mcp_test_button_uses_shared_add_style() -> None:
    app = _ConfigAddButtonApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = MCPConfigPanel([MCPServerConfig(name="stdio-server", transport="stdio", command="python")])
        await app.mount(panel)

        (card,) = await _wait_for_mcp_cards(panel, pilot, 1)

        _assert_shared_add_button(card, "#mcp-test-btn-0")


@pytest.mark.asyncio
async def test_skill_browse_and_extension_buttons_use_shared_press_behavior() -> None:
    app = _ConfigAddButtonApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SkillsConfigPanel(SkillsConfig(paths=["/tmp/skills"]))
        await app.mount(panel)
        await pilot.pause()

        (card,) = await _wait_for_skill_cards(panel, pilot, 1)

        assert isinstance(card.query_one("#sk-browse-0"), ConfigActionButton)
        assert isinstance(panel.query_one("#sk-ext-py"), ExtChip)
        assert isinstance(panel.query_one("#sk-ext-py"), ConfigActionButton)
