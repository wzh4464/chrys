# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the agent configuration screen."""

from __future__ import annotations

import asyncio
import copy
import logging
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.css.query import NoMatches
from textual.widgets import (
    Button,
    Checkbox,
    ContentSwitcher,
    Input,
    Label,
    OptionList,
    Select,
    Static,
    TabbedContent,
    TextArea,
)
from textual.widgets._select import SelectCurrent, SelectOverlay

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.agents import config as config_mod
from chrys.app.tui.screens.agents.config import AgentsConfigScreen
from chrys.app.tui.screens.agents.panels.basic import BasicConfigPanel
from chrys.app.tui.screens.agents.panels.compaction import CompactionConfigPanel
from chrys.app.tui.screens.agents.panels.mcp import MCPConfigPanel
from chrys.app.tui.screens.agents.panels.tools import ToolsConfigPanel
from chrys.foundation.config.settings import Settings
from chrys.service.profiles.agents.loader import load_profile_from_yaml
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import (
    AgentProfile,
    CompactionConfig,
    MCPServerConfig,
    MemoryConfig,
    SkillsConfig,
    SubAgentRef,
    SubAgentsConfig,
    ToolsConfig,
)
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile
from tests.support.platform_fakes import platform_with_config_dir
from tests.support.tui_helpers import make_backend_handler
from tests.support.waiting import reset_wait_deadline, shared_wait_deadline

# The polling helpers below share one 45s ceiling, below the repository's 60s
# default per-test timeout. This module retains a 120s override for expensive
# screen mounting before the first wait; once polling starts, the helper
# deadline still produces a clean AssertionError before the thread timeout can
# kill the xdist worker and surface only "worker gwN crashed".
pytestmark = pytest.mark.timeout(120)


class _AgentConfigApp(App):
    locale_controller = LocaleController(Settings(locale="en"))

    def compose(self) -> ComposeResult:
        yield Static("placeholder")


# One polling budget is shared by every wait helper in a test (see
# shared_wait_deadline). Heavy multi-tab tests spend it across several
# hydration waits, and Windows CI under xdist load has pushed past 20s
# total — the deadline is a ceiling, not a sleep, so fast tests are
# unaffected by the headroom.
_DEFAULT_WAIT_TIMEOUT = 45.0


async def _wait_for_selectors(screen, pilot, *selectors: str, timeout: float = _DEFAULT_WAIT_TIMEOUT) -> None:
    """Poll until every selector resolves on the screen.

    Re-mounting a config panel (e.g. after Clone) yields its children in
    batches; a single ``pilot.pause()`` is not always enough to drain
    them on slower CI hosts. Waiting on the specific widgets that the
    next interaction needs makes the test deterministic.
    """
    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while True:
        try:
            for sel in selectors:
                screen.query_one(sel)
            return
        except NoMatches:
            if loop.time() > deadline:
                raise
            await pilot.pause(0.05)


async def _wait_for_input_enabled(screen, pilot, selector: str, timeout: float = _DEFAULT_WAIT_TIMEOUT) -> None:
    """Poll until an Input widget matching *selector* exists and is enabled."""
    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while True:
        try:
            widget = screen.query_one(selector, Input)
            if not widget.disabled:
                return
        except NoMatches:
            pass
        if loop.time() > deadline:
            raise AssertionError(f"{selector} did not become enabled within {timeout}s")
        await pilot.pause(0.05)


async def _wait_for_condition(
    predicate, pilot, *, timeout: float = _DEFAULT_WAIT_TIMEOUT, description: str = "condition"
) -> None:
    """Poll *predicate* until it returns True.

    Some panel-state updates (e.g. dirty flag → save-button-enabled)
    cascade through several handlers, and a single ``pilot.pause()``
    is not always enough on slower CI hosts. This helper drains the
    message queue until the assertion would actually hold.
    """
    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError(f"{description} did not become true within {timeout}s")
        await pilot.pause(0.05)


def _agent_config_clone_save_debug(screen: AgentsConfigScreen, registry: AgentProfileRegistry, copied_name: str) -> str:
    """Return a compact snapshot for diagnosing clone-save timing failures."""
    selected = screen._drafts.get(screen._selected_draft_key)
    dirty_names = [draft.profile.name for draft in screen._drafts.values() if draft.dirty]
    registry_names = registry.list_names()
    return (
        f"selected={screen._selected_profile_name!r}; "
        f"selected_draft={selected.profile.name if selected is not None else None!r}; "
        f"hydrating={screen._hydrating!r}; "
        f"save_disabled={screen.query_one('#ac-save', Button).disabled!r}; "
        f"copied_registered={registry.get(copied_name) is not None!r}; "
        f"dirty={dirty_names!r}; "
        f"registry={registry_names!r}"
    )


async def _wait_for_hydrated(screen: AgentsConfigScreen, pilot, timeout: float = _DEFAULT_WAIT_TIMEOUT) -> None:
    """Wait until the agent config screen has finished hydrating its panels.

    ``_finish_hydrating`` clears ``screen._hydrating`` after its retry
    budget is exhausted even when the mounted panels never matched the
    draft (only a warning is logged). On slower Windows CI under xdist
    load that give-up path can fire for profiles with several sub-agent
    rows, so checking the flag alone is a false-positive. Also require
    a fresh panel rebuild to equal the draft before returning.
    """
    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while True:
        last_state = f"hydrating={screen._hydrating!r}"
        if not screen._hydrating:
            draft = screen._drafts.get(screen._selected_draft_key)
            if draft is None:
                return
            try:
                if screen._build_profile_from_mounted_panels(draft) == draft.profile:
                    return
                last_state += "; mounted panels != draft"
            except Exception as exc:  # transient — a panel is still mounting
                last_state += f"; panels not readable: {exc!r}"
        if loop.time() > deadline:
            raise AssertionError(f"agent config screen did not finish hydrating ({last_state})")
        await pilot.pause(0.05)


async def _wait_for_confirm_button(app: App, pilot, timeout: float = _DEFAULT_WAIT_TIMEOUT) -> Button:
    """Poll until a ConfirmDialog is the active screen and its confirm button is mounted.

    ``push_screen`` composes the dialog asynchronously and its buttons live
    inside a nested ``DialogButtonRow``; a single ``pilot.pause()`` after the
    triggering press is not a reliable mount barrier on slower CI hosts.
    """
    from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog

    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while True:
        dialog = app.screen
        if isinstance(dialog, ConfirmDialog):
            try:
                button = dialog.query_one("#confirm-yes", Button)
            except NoMatches:
                pass
            else:
                if button.is_mounted:
                    return button
        if loop.time() > deadline:
            raise AssertionError(f"confirm dialog did not open (active screen: {app.screen!r})")
        await pilot.pause(0.05)


async def _wait_for_active_screen(app: App, pilot, screen, timeout: float = _DEFAULT_WAIT_TIMEOUT) -> None:
    """Poll until *screen* is the active screen again (e.g. a modal was dismissed)."""
    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while app.screen is not screen:
        if loop.time() > deadline:
            raise AssertionError(f"{screen!r} did not become active (active screen: {app.screen!r})")
        await pilot.pause(0.05)


async def _activate_agent_config_tab(screen: AgentsConfigScreen, pilot, tab_id: str) -> None:
    screen.query_one("#ac-tabs", TabbedContent).active = tab_id
    await pilot.pause()
    await _wait_for_hydrated(screen, pilot)


async def _wait_for_panel_display_name(
    screen: AgentsConfigScreen,
    pilot,
    expected: str,
    timeout: float = _DEFAULT_WAIT_TIMEOUT,
) -> None:
    """Wait until the mounted panels are readable and reflect an edited display name.

    Pressing Save while ``_build_profile_from_mounted_panels`` can still raise
    ``_AgentConfigPanelsNotReady`` (a tab transiently not queryable under slow
    Windows CI) makes ``_on_save`` abort with a "Save Error" and silently skip the
    write. Confirm the edit is harvestable before saving so the press is deterministic.
    """
    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while True:
        draft = screen._drafts.get(screen._selected_draft_key)
        if draft is not None and not screen._hydrating:
            try:
                if screen._build_profile_from_mounted_panels(draft).display_name == expected:
                    return
            except Exception:
                pass  # transient — a panel is still mounting
        if loop.time() > deadline:
            raise AssertionError(f"panels did not reflect display name {expected!r}")
        await pilot.pause(0.05)


async def _wait_for_selected_profile_name(
    screen: AgentsConfigScreen,
    pilot,
    expected: str,
    timeout: float = _DEFAULT_WAIT_TIMEOUT,
) -> None:
    """Wait until the selected draft has the expected profile name.

    Some callers use this after editing the mounted Basic tab; others use it
    after a clone button press. In both cases the thing under test is the
    staged draft name, not the full tab hydration lifecycle. Requiring
    ``_hydrating`` to be false made clone-name tests fail under CI load even
    after the clone had been selected correctly.
    """
    loop = asyncio.get_running_loop()
    deadline = shared_wait_deadline(timeout)
    while True:
        draft = screen._drafts.get(screen._selected_draft_key)
        if draft is not None and draft.profile.name == expected:
            return
        if loop.time() > deadline:
            actual = draft.profile.name if draft is not None else None
            raise AssertionError(
                f"selected profile did not become {expected!r}; got {actual!r}; hydrating={screen._hydrating!r}"
            )
        await pilot.pause(0.05)


@pytest.fixture(autouse=True)
def _isolate_chrys_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep profile serializer paths inside pytest temp dirs."""
    reset_wait_deadline()
    fake_platform = platform_with_config_dir(tmp_path)
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)


def _registry() -> AgentProfileRegistry:
    registry = AgentProfileRegistry()
    registry.load_builtins()
    return registry


def _draft_for_profile(screen: AgentsConfigScreen, profile_name: str):
    for draft in screen._drafts.values():
        if draft.profile.name == profile_name:
            return draft
    raise AssertionError(f"Draft for {profile_name!r} not found")


def _draft_for_original(screen: AgentsConfigScreen, original_name: str):
    for draft in screen._drafts.values():
        if draft.original_name == original_name:
            return draft
    raise AssertionError(f"Draft originally named {original_name!r} not found")


def _sidebar_text(screen: AgentsConfigScreen, draft_key: str) -> str:
    option = screen.query_one("#ac-list", OptionList).get_option(draft_key)
    return option.prompt.plain


def test_agent_config_canonicalizes_mcp_always_load_names_without_dropping_progressive_mode() -> None:
    profile = AgentProfile(
        name="Progressive",
        tools=ToolsConfig(
            mcp=[
                MCPServerConfig(
                    name="remote",
                    transport="http",
                    url="https://api.example.com/mcp",
                    use_progressive_disclosure=True,
                    always_load=[" search ", "", "read_file "],
                )
            ]
        ),
    )

    AgentsConfigScreen._canonicalize_profile_for_ui(profile)

    assert profile.tools.mcp[0].use_progressive_disclosure is True
    assert profile.tools.mcp[0].always_load == ["search", "read_file"]


def test_agent_config_clears_initial_subset_for_full_or_empty_loading_policy() -> None:
    full = MCPServerConfig(
        name="full",
        transport="http",
        url="https://api.example.com/full",
        always_load=["stale"],
    )
    empty = MCPServerConfig(
        name="empty",
        transport="http",
        url="https://api.example.com/empty",
        allowed_tools=[],
        use_progressive_disclosure=True,
        always_load=["stale"],
    )
    profile = AgentProfile(name="Policy", tools=ToolsConfig(mcp=[full, empty]))

    AgentsConfigScreen._canonicalize_profile_for_ui(profile)

    assert full.use_progressive_disclosure is False
    assert full.always_load == []
    assert empty.use_progressive_disclosure is False
    assert empty.always_load == []


def test_agent_config_exposes_live_sub_agent_tool_names_to_mcp_diagnostics() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="Parent",
            sub_agents=SubAgentsConfig(
                agents=[
                    SubAgentRef(profile="Explore", tool_name="explore_agent"),
                    SubAgentRef(profile="Plan"),
                ]
            ),
        )
    )
    screen = AgentsConfigScreen(registry, current_profile="Parent")
    screen._initialize_drafts()
    screen._selected_draft_key = screen._existing_draft_key("Parent")

    assert screen._selected_sub_agent_tool_names() == {"explore_agent", "Plan"}

    draft = screen._drafts[screen._selected_draft_key]
    draft.profile.sub_agents.agents[0].tool_name = "research_agent"
    assert screen._selected_sub_agent_tool_names() == {"research_agent", "Plan"}


@pytest.mark.asyncio
async def test_mounted_mcp_panel_receives_selected_draft_sub_agent_names() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="Parent",
            display_name="Parent Agent",
            tools=ToolsConfig(mcp=[]),
            sub_agents=SubAgentsConfig(
                agents=[
                    SubAgentRef(profile="Explore", tool_name="explore_agent"),
                    SubAgentRef(profile="Plan"),
                ]
            ),
        )
    )
    registry.register(AgentProfile(name="Explore", sub_agent_only=True))
    registry.register(AgentProfile(name="Plan", sub_agent_only=True))

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Parent Agent", initial_tab="mcp")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        panel = screen.query_one(MCPConfigPanel)
        assert {"explore_agent", "Plan"} <= panel._current_reserved_tool_names()


@pytest.mark.asyncio
async def test_agent_config_container_default_size_limits() -> None:
    app = _AgentConfigApp()
    async with app.run_test(size=(160, 80)) as pilot:
        screen = AgentsConfigScreen(_registry(), current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()

        container = screen.query_one("#ac-container", Vertical)
        assert str(container.styles.max_width) == "120"
        assert str(container.styles.max_height) == "60"


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_tab", ["basic", "compaction", "instructions"])
async def test_agent_config_builtin_profiles_round_trip_cleanly_through_panels(initial_tab: str) -> None:
    registry = _registry()
    profiles = registry.list_profiles(include_sub_agent_only=True)

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code", initial_tab=initial_tab)
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        for profile in profiles:
            draft = _draft_for_original(screen, profile.name)
            screen._load_profile(draft.key)
            await pilot.pause()
            await _wait_for_hydrated(screen, pilot)

            rebuilt = screen._build_profile_from_mounted_panels(draft)
            expected = copy.deepcopy(profile)
            screen._canonicalize_profile_for_ui(expected)

            assert rebuilt == expected, profile.name


@pytest.mark.asyncio
async def test_agent_config_save_button_and_modified_marker_follow_pending_changes() -> None:
    registry = _registry()

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        save = screen.query_one("#ac-save", Button)
        original_display = screen.query_one("#bc-display-name", Input).value
        selected_key = screen._selected_draft_key
        await _wait_for_condition(lambda: save.disabled, pilot, description="save disabled after initial hydration")
        assert "(modified)" not in _sidebar_text(screen, selected_key)

        screen.query_one("#bc-display-name", Input).focus()
        screen.query_one("#bc-display-name", Input).value = "Code Agent Edited"
        await _wait_for_condition(lambda: not save.disabled, pilot, description="save enabled after first edit")
        assert screen._drafts[selected_key].dirty is True
        assert "(modified)" in _sidebar_text(screen, selected_key)

        screen.query_one("#bc-display-name", Input).value = original_display
        await _wait_for_condition(lambda: save.disabled, pilot, description="save disabled after revert")
        assert screen._drafts[selected_key].dirty is False
        assert "(modified)" not in _sidebar_text(screen, selected_key)

        screen.query_one("#bc-display-name", Input).value = "Code Agent Edited"
        await _wait_for_condition(lambda: not save.disabled, pilot, description="save enabled after re-edit")
        assert screen._drafts[selected_key].dirty is True
        assert "(modified)" in _sidebar_text(screen, selected_key)

        save.press()
        await _wait_for_condition(lambda: save.disabled, pilot, description="save disabled after save")

        saved = registry.get("Code")
        assert saved is not None
        assert saved.display_name == "Code Agent Edited"
        assert "(modified)" not in _sidebar_text(screen, screen._selected_draft_key)


@pytest.mark.asyncio
async def test_agent_config_invalid_mcp_text_change_still_marks_dirty() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="mcp-agent",
            display_name="MCP Agent",
            description="Agent with editable MCP config",
            instructions="Follow the user's instructions.",
            tools=ToolsConfig(mcp=[MCPServerConfig(name="local", transport="stdio", command="python", args=["-V"])]),
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="MCP Agent", initial_tab="mcp")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        save = screen.query_one("#ac-save", Button)
        assert save.disabled is True

        command = screen.query_one("#mcp-cmd-0", TextArea)
        command.focus()
        command.text = '"'
        await _wait_for_condition(
            lambda: not save.disabled and screen._drafts[screen._selected_draft_key].dirty,
            pilot,
            description="save enabled after invalid MCP command edit",
        )

        assert save.disabled is False
        assert screen._drafts[screen._selected_draft_key].dirty is True
        assert "(modified)" in _sidebar_text(screen, screen._selected_draft_key)

        command.text = "python -V"
        await _wait_for_condition(
            lambda: save.disabled and not screen._drafts[screen._selected_draft_key].dirty,
            pilot,
            description="save disabled after MCP command revert",
        )

        assert save.disabled is True
        assert screen._drafts[screen._selected_draft_key].dirty is False
        assert "(modified)" not in _sidebar_text(screen, screen._selected_draft_key)


@pytest.mark.asyncio
async def test_agent_config_delete_mcp_card_marks_draft_dirty() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="mcp-agent",
            display_name="MCP Agent",
            description="Agent with removable MCP config",
            instructions="Follow the user's instructions.",
            tools=ToolsConfig(mcp=[MCPServerConfig(name="local", transport="stdio", command="python", args=["-V"])]),
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="MCP Agent", initial_tab="mcp")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        save = screen.query_one("#ac-save", Button)
        assert save.disabled is True

        screen.query_one("#mcp-delete-btn-0", Button).press()
        await _wait_for_condition(
            lambda: not save.disabled and screen._drafts[screen._selected_draft_key].dirty,
            pilot,
            description="save enabled after MCP card delete",
        )


@pytest.mark.parametrize("row_kind", ["env", "headers"])
@pytest.mark.asyncio
async def test_agent_config_delete_mcp_key_value_row_marks_draft_dirty(row_kind: str) -> None:
    if row_kind == "env":
        server_config = MCPServerConfig(
            name="local",
            transport="stdio",
            command="python",
            env={"TOKEN": "abc", "NO_PROXY": "*"},
        )
        container_selector = "#mcp-env-0"
    else:
        server_config = MCPServerConfig(
            name="remote",
            transport="http",
            url="https://api.example.test/mcp",
            headers={"Authorization": "Bearer token", "X-Team": "platform"},
        )
        container_selector = "#mcp-headers-0"

    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="mcp-agent",
            display_name="MCP Agent",
            description="Agent with removable MCP key-value rows",
            instructions="Follow the user's instructions.",
            tools=ToolsConfig(mcp=[server_config]),
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="MCP Agent", initial_tab="mcp")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        save = screen.query_one("#ac-save", Button)
        assert save.disabled is True

        container = screen.query_one(container_selector)
        remove_buttons = list(container.query(".mcp-remove-btn"))
        assert len(remove_buttons) == 2

        remove_buttons[0].press()
        await _wait_for_condition(
            lambda: (
                len(list(container.query(".mcp-item-row"))) == 1
                and not save.disabled
                and screen._drafts[screen._selected_draft_key].dirty
            ),
            pilot,
            description=f"save enabled after MCP {row_kind} row delete",
        )


@pytest.mark.asyncio
async def test_agent_config_delete_sub_agent_card_marks_draft_dirty() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="Parent",
            display_name="Parent Agent",
            description="Agent with removable sub-agent config",
            instructions="Follow the user's instructions.",
            sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Child")]),
        )
    )
    registry.register(
        AgentProfile(
            name="Child",
            display_name="Child Agent",
            description="Child agent",
            instructions="Follow the user's instructions.",
            sub_agent_only=True,
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Parent Agent", initial_tab="sub-agents")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        save = screen.query_one("#ac-save", Button)
        # Poll rather than assert instantaneously: on slow Windows CI a settle
        # event can momentarily set dirty just after _wait_for_hydrated returns;
        # the clean baseline self-corrects once the queued events drain.
        await _wait_for_condition(
            lambda: save.disabled and not screen._drafts[screen._selected_draft_key].dirty,
            pilot,
            description="clean baseline after sub-agent hydration",
        )

        screen.query_one("#sa-delete-btn-0", Button).press()
        await _wait_for_condition(
            lambda: not save.disabled and screen._drafts[screen._selected_draft_key].dirty,
            pilot,
            description="save enabled after sub-agent card delete",
        )


@pytest.mark.asyncio
async def test_agent_config_transient_panels_not_ready_does_not_force_dirty() -> None:
    """A forced re-evaluation that hits ``_AgentConfigPanelsNotReady`` must not mark dirty.

    Regression: a field-change event firing in the post-hydration settle window drove
    ``_mark_selected_dirty(force=True)`` into the build-failure branch, which force-set
    a sticky spurious dirty and wrongly enabled Save (intermittent CI failures on the
    dirty-tracking tests). A transient "panels not ready" read must leave the flag alone.

    Uses a plain profile and drives the path directly so the assertion does not race
    panel hydration timing.
    """
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="Solo",
            display_name="Solo Agent",
            description="Agent without dynamic panels",
            instructions="Follow the user's instructions.",
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Solo Agent", initial_tab="basic")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        draft = screen._drafts[screen._selected_draft_key]
        screen._set_draft_dirty(draft, False)  # baseline clean, independent of hydration timing

        # Simulate a forced field-change re-evaluation while the panels are
        # transiently unreadable, as they can be during the settle window.
        def _raise_not_ready(_draft: object) -> object:
            raise config_mod._AgentConfigPanelsNotReady("panels still mounting")

        original_build = screen._build_profile_from_mounted_panels
        screen._build_profile_from_mounted_panels = _raise_not_ready  # type: ignore[method-assign]
        try:
            screen._mark_selected_dirty(force=True)
        finally:
            screen._build_profile_from_mounted_panels = original_build  # type: ignore[method-assign]

        assert draft.dirty is False
        assert screen.query_one("#ac-save", Button).disabled is True


@pytest.mark.asyncio
async def test_agent_config_lazy_tab_mount_uses_hydration_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry()

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        mounted_while_hydrating: list[bool] = []
        original_mount_tools = screen._mount_tools_tab

        def mount_tools_with_probe(profile) -> None:
            mounted_while_hydrating.append(screen._hydrating)
            original_mount_tools(profile)

        monkeypatch.setattr(screen, "_mount_tools_tab", mount_tools_with_probe)

        screen.query_one("#ac-tabs", TabbedContent).active = "tools"
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        assert mounted_while_hydrating == [True]
        assert screen.query_one("#ac-save", Button).disabled is True


@pytest.mark.asyncio
async def test_agent_config_stale_hydration_completion_does_not_clear_validation_only_dirty() -> None:
    registry = _registry()

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        save = screen.query_one("#ac-save", Button)
        draft = screen._drafts[screen._selected_draft_key]
        assert draft.original_profile is not None
        assert draft.profile == draft.original_profile
        screen._set_draft_dirty(draft, True)
        assert save.disabled is False

        screen._hydrating = True
        screen._hydration_preserve_dirty = True
        screen._hydrating_generation = 2
        screen._complete_hydration(1)

        assert screen._hydrating is True
        assert screen._hydration_preserve_dirty is True
        assert draft.dirty is True
        assert save.disabled is False

        screen._complete_hydration(2)

        assert screen._hydrating is False
        assert screen._hydration_preserve_dirty is False
        assert draft.dirty is True
        assert save.disabled is False


@pytest.mark.asyncio
async def test_agent_config_delete_memory_file_marks_draft_dirty() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="memory-agent",
            display_name="Memory Agent",
            description="Agent with removable memory files",
            instructions="Follow the user's instructions.",
            memory=MemoryConfig(files=["docs/a.md", "docs/b.md"]),
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Memory Agent", initial_tab="memory")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        save = screen.query_one("#ac-save", Button)
        assert save.disabled is True

        files = screen.query_one("#mem-files")
        assert len(list(files.query(".agent-config-card"))) == 2

        files.query_one("#mem-delete-btn-0", Button).press()
        await _wait_for_condition(
            lambda: (
                len(list(files.query(".agent-config-card"))) == 1
                and not save.disabled
                and screen._drafts[screen._selected_draft_key].dirty
            ),
            pilot,
            description="save enabled after memory file delete",
        )


@pytest.mark.asyncio
async def test_agent_config_delete_skill_path_marks_draft_dirty() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="skills-agent",
            display_name="Skills Agent",
            description="Agent with removable skill paths",
            instructions="Follow the user's instructions.",
            skills=SkillsConfig(paths=["skills/a", "skills/b"]),
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Skills Agent", initial_tab="skills")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        save = screen.query_one("#ac-save", Button)
        assert save.disabled is True

        dirs = screen.query_one("#sk-dirs")
        assert len(list(dirs.query(".agent-config-card"))) == 2

        dirs.query_one("#sk-delete-btn-0", Button).press()
        await _wait_for_condition(
            lambda: (
                len(list(dirs.query(".agent-config-card"))) == 1
                and not save.disabled
                and screen._drafts[screen._selected_draft_key].dirty
            ),
            pilot,
            description="save enabled after skill path delete",
        )


@pytest.mark.asyncio
async def test_agent_config_switch_error_includes_actual_validation_errors() -> None:
    registry = _registry()

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#bc-display-name", Input).focus()
        screen.query_one("#bc-display-name", Input).value = ""
        await pilot.pause()

        with pytest.raises(RuntimeError) as exc_info:
            screen._sync_selected_draft_from_panels()

    lines = str(exc_info.value).splitlines()
    assert lines[0] == "Display name is required."
    assert lines[-1] == "Fix validation errors before switching agents or applying structural changes."


@pytest.mark.asyncio
async def test_agent_config_unsaved_edit_survives_agent_switch_and_close_discards(tmp_path: Path) -> None:
    registry = _registry()
    original = registry.get("Code")
    assert original is not None

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        code_key = screen._selected_draft_key
        explore_key = _draft_for_profile(screen, "Explore").key
        screen.query_one("#bc-display-name", Input).focus()
        screen.query_one("#bc-display-name", Input).value = "Unsaved Code Label"
        await pilot.pause()

        screen._sync_selected_draft_from_panels()
        screen._load_profile(explore_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        screen._sync_selected_draft_from_panels()
        screen._load_profile(code_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        assert screen.query_one("#bc-display-name", Input).value == "Unsaved Code Label"
        screen.action_cancel()
        await pilot.pause()

    current = registry.get("Code")
    assert current is not None
    assert current.display_name == original.display_name
    assert not (tmp_path / "agents" / "Code.yaml").exists()


@pytest.mark.asyncio
async def test_agent_config_reset_preserves_unsaved_memory_and_close_discards(tmp_path: Path) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = _registry()
    template = registry.get_builtin_template("Code")
    assert template is not None
    customized = copy.deepcopy(template)
    customized.instructions = "custom instructions"
    customized.skills = SkillsConfig(paths=["custom-skills"])
    customized.tools.mcp = [MCPServerConfig(name="private", transport="http", url="https://example.test")]
    customized.memory = MemoryConfig(files=["saved.md"])
    canonical_customized = copy.deepcopy(customized)
    AgentsConfigScreen._canonicalize_profile_for_ui(canonical_customized)
    save_profile(customized)
    registry.register(customized)
    original_yaml = (tmp_path / "agents" / "Code.yaml").read_bytes()

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code", initial_tab="memory")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        assert screen.query_one("#ac-reset", Button).display is True
        assert screen.query_one("#ac-delete", Button).display is False
        memory_input = screen.query_one("#mem-file-path-0", Input)
        memory_input.focus()
        memory_input.value = "unsaved.md"
        screen.query_one("#ac-reset", Button).press()
        confirm_button = await _wait_for_confirm_button(app, pilot)
        assert confirm_button.variant == "primary"  # same look as New/Clone, not a destructive action
        confirm_button.press()
        await _wait_for_active_screen(app, pilot, screen)
        await _wait_for_hydrated(screen, pilot)

        draft = screen._drafts[screen._selected_draft_key]
        assert draft.profile.instructions == template.instructions
        assert draft.profile.skills == canonical_customized.skills
        assert draft.profile.tools.mcp == canonical_customized.tools.mcp
        assert draft.profile.memory.files == ["unsaved.md"]
        assert draft.reset_to_builtin is True
        assert draft.dirty is True
        assert screen.query_one("#ac-save", Button).disabled is False
        screen.action_cancel()
        await pilot.pause()

    assert registry.get("Code") == customized
    assert (tmp_path / "agents" / "Code.yaml").read_bytes() == original_yaml


@pytest.mark.asyncio
async def test_agent_config_reset_to_exact_template_deletes_shadow_on_save(tmp_path: Path) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = _registry()
    template = registry.get_builtin_template("Code")
    assert template is not None
    canonical_template = copy.deepcopy(template)
    AgentsConfigScreen._canonicalize_profile_for_ui(canonical_template)
    customized = copy.deepcopy(template)
    customized.instructions = "custom instructions"
    save_profile(customized)
    registry.register(customized)

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen._do_reset(screen._selected_draft_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        screen.query_one("#ac-save", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    assert registry.get("Code") == canonical_template
    assert not (tmp_path / "agents" / "Code.yaml").exists()


@pytest.mark.asyncio
async def test_agent_config_reset_removes_shadow_loaded_from_noncanonical_filename(tmp_path: Path) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    user_dir = tmp_path / "agents"
    template = _registry().get_builtin_template("Code")
    assert template is not None
    customized = copy.deepcopy(template)
    customized.instructions = "custom instructions"
    save_profile(customized)
    (user_dir / "Code.yaml").rename(user_dir / "my-code.yaml")
    registry = AgentProfileRegistry()
    registry.load_all(user_dir=user_dir)
    assert registry.get("Code").instructions == "custom instructions"

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen._do_reset(screen._selected_draft_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        screen.query_one("#ac-save", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    assert registry.get("Code").instructions == template.instructions
    assert sorted(p.name for p in user_dir.glob("*.y*ml")) == []


@pytest.mark.asyncio
async def test_agent_config_reset_with_only_preserved_changes_is_noop(tmp_path: Path) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = _registry()
    template = registry.get_builtin_template("Code")
    assert template is not None
    customized = copy.deepcopy(template)
    customized.skills = SkillsConfig(paths=["custom-skills"])
    canonical_customized = copy.deepcopy(customized)
    AgentsConfigScreen._canonicalize_profile_for_ui(canonical_customized)
    save_profile(customized)
    registry.register(customized)
    shadow_path = tmp_path / "agents" / "Code.yaml"
    original_yaml = shadow_path.read_bytes()

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen._do_reset(screen._selected_draft_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        draft = screen._drafts[screen._selected_draft_key]
        assert draft.profile == canonical_customized
        assert draft.reset_to_builtin is False
        assert draft.dirty is False
        assert screen.query_one("#ac-save", Button).disabled is True

    assert registry.get("Code") == customized
    assert shadow_path.read_bytes() == original_yaml


@pytest.mark.asyncio
async def test_agent_config_reset_with_preserved_settings_keeps_shadow(tmp_path: Path) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = _registry()
    template = registry.get_builtin_template("Code")
    assert template is not None
    customized = copy.deepcopy(template)
    customized.instructions = "custom instructions"
    customized.skills = SkillsConfig(paths=["custom-skills"])
    canonical_customized = copy.deepcopy(customized)
    AgentsConfigScreen._canonicalize_profile_for_ui(canonical_customized)
    save_profile(customized)
    registry.register(customized)

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen._do_reset(screen._selected_draft_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        screen._apply_all()
        await _wait_for_hydrated(screen, pilot)

    saved = load_profile_from_yaml(tmp_path / "agents" / "Code.yaml")
    assert saved.instructions == template.instructions
    assert saved.skills == canonical_customized.skills


@pytest.mark.asyncio
async def test_agent_config_reset_shadow_delete_rolls_back_when_later_save_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = _registry()
    template = registry.get_builtin_template("Code")
    assert template is not None
    customized = copy.deepcopy(template)
    customized.instructions = "custom instructions"
    other = AgentProfile(
        name="Other",
        display_name="Other",
        description="Another agent",
        instructions="Follow the user's instructions.",
    )
    save_profile(customized)
    save_profile(other)
    registry.register(customized)
    registry.register(other)
    code_path = tmp_path / "agents" / "Code.yaml"
    other_path = tmp_path / "agents" / "Other.yaml"
    original_code = code_path.read_bytes()
    original_other = other_path.read_bytes()

    original_save_profile = save_profile

    def fail_on_other(profile, *args, **kwargs):
        if profile.name == "Other":
            raise OSError("simulated save failure")
        return original_save_profile(profile, *args, **kwargs)

    monkeypatch.setattr("chrys.service.profiles.agents.serializer.save_profile", fail_on_other)
    deleted: list[str] = []
    monkeypatch.setattr("chrys.service.profiles.agents.serializer.delete_profile", deleted.append)
    original_code_mode = _file_mode(code_path)

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen._do_reset(screen._selected_draft_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        other_draft = _draft_for_original(screen, "Other")
        other_draft.profile.display_name = "Other edited"
        other_draft.dirty = True

        screen.query_one("#ac-save", Button).press()
        await pilot.pause()

        reset_draft = _draft_for_original(screen, "Code")
        assert reset_draft.reset_to_builtin is True
        assert reset_draft.dirty is True

    restored_code = registry.get("Code")
    restored_other = registry.get("Other")
    assert restored_code is not None
    assert restored_code.instructions == "custom instructions"
    assert restored_other is not None
    assert restored_other.display_name == "Other"
    assert code_path.read_bytes() == original_code
    assert other_path.read_bytes() == original_other
    # Shadow deletes run only after every save succeeded, so a failed save
    # never has to be undone by recreating the shadow.
    assert deleted == []
    assert _file_mode(code_path) == original_code_mode


def _file_mode(path: Path) -> int | None:
    """Return the permission bits of *path* on POSIX; Windows has no comparable mode."""
    if sys.platform == "win32":
        return None
    return stat.S_IMODE(path.stat().st_mode)


@pytest.mark.asyncio
async def test_agent_config_reset_rollback_recreates_shadow_owner_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shadow removed by an exact-template reset comes back 0600 when a later step fails."""
    from chrys.service.profiles.agents.serializer import delete_profile, save_profile

    registry = _registry()
    shadows: dict[str, bytes] = {}
    for name in ("Code", "Explore"):
        template = registry.get_builtin_template(name)
        assert template is not None
        customized = copy.deepcopy(template)
        customized.instructions = f"custom {name}"
        save_profile(customized)
        registry.register(customized)
        shadows[name] = (tmp_path / "agents" / f"{name}.yaml").read_bytes()
    code_path = tmp_path / "agents" / "Code.yaml"
    original_code_mode = _file_mode(code_path)

    original_delete_profile = delete_profile
    deleted: list[str] = []

    def fail_on_explore(name, *args, **kwargs):
        deleted.append(name)
        if name == "Explore":
            raise OSError("simulated delete failure")
        return original_delete_profile(name, *args, **kwargs)

    monkeypatch.setattr("chrys.service.profiles.agents.serializer.delete_profile", fail_on_explore)

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen._do_reset(_draft_for_original(screen, "Code").key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        screen._do_reset(_draft_for_original(screen, "Explore").key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#ac-save", Button).press()
        await pilot.pause()

    # The Code shadow really was removed and then recreated by the rollback.
    assert deleted == ["Code", "Explore"]
    for name, original in shadows.items():
        assert (tmp_path / "agents" / f"{name}.yaml").read_bytes() == original
        assert registry.get(name).instructions == f"custom {name}"
    assert _file_mode(code_path) == original_code_mode


@pytest.mark.asyncio
async def test_agent_config_new_profile_is_staged_and_close_discards(tmp_path: Path) -> None:
    registry = _registry()

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Explore")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#ac-new", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        new_profile = screen._drafts[screen._selected_draft_key].profile

        assert registry.get(new_profile.name) is None
        assert not (tmp_path / "agents" / f"{new_profile.name}.yaml").exists()

        screen.action_cancel()
        await pilot.pause()

    assert registry.get("Explore") is not None
    assert registry.get(new_profile.name) is None
    assert not (tmp_path / "agents" / f"{new_profile.name}.yaml").exists()


@pytest.mark.asyncio
async def test_agent_config_staged_new_agent_can_be_selected_as_subagent_before_save() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="Parent",
            display_name="Parent Agent",
            description="Parent profile",
            instructions="Follow the user's instructions.",
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Parent Agent")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        parent_key = screen._selected_draft_key
        screen.query_one("#ac-new", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        new_profile = screen._drafts[screen._selected_draft_key].profile

        screen._sync_selected_draft_from_panels()
        screen._load_profile(parent_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        await _activate_agent_config_tab(screen, pilot, "sub-agents")
        screen.query_one("#sa-add-btn", Button).press()
        await pilot.pause()
        select = screen.query_one("#sa-profile-0", Select)
        option_values = [value for _prompt, value in select._options]
        assert new_profile.name in option_values

        select.value = new_profile.name
        await pilot.pause()
        screen.query_one("#ac-save", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    saved_parent = registry.get("Parent")
    saved_new = registry.get(new_profile.name)
    assert saved_parent is not None
    assert saved_new is not None
    assert [ref.profile for ref in saved_parent.sub_agents.agents] == [new_profile.name]


@pytest.mark.asyncio
async def test_agent_config_save_ignores_invalid_untouched_draft() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="Good",
            display_name="Good Agent",
            description="Valid profile",
            instructions="Follow the user's instructions.",
        )
    )
    registry.register(
        AgentProfile(
            name="stale",
            display_name="",
            description="Legacy invalid profile",
            instructions="Follow the user's instructions.",
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Good Agent")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#bc-display-name", Input).focus()
        screen.query_one("#bc-display-name", Input).value = "Better Agent"
        await pilot.pause()
        # Wait for Input.Changed → _mark_selected_dirty to flip the draft dirty,
        # otherwise the disabled Save button swallows press() and the save becomes
        # a silent no-op on slower Windows CI under xdist load.
        save = screen.query_one("#ac-save", Button)
        await _wait_for_condition(lambda: not save.disabled, pilot, description="save enabled after display-name edit")
        save.press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    saved = registry.get("Good")
    stale = registry.get("stale")
    assert saved is not None
    assert saved.display_name == "Better Agent"
    assert stale is not None
    assert stale.display_name == ""


@pytest.mark.asyncio
async def test_agent_config_delete_custom_profile_is_immediate_and_leaves_save_disabled(tmp_path: Path) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = AgentProfileRegistry()
    profile = AgentProfile(
        name="custom",
        display_name="Custom",
        description="Custom profile",
        instructions="Follow the user's instructions.",
    )
    fallback = AgentProfile(
        name="fallback",
        display_name="Fallback",
        description="Fallback profile",
        instructions="Follow the user's instructions.",
    )
    save_profile(profile)
    save_profile(fallback)
    registry.register(profile)
    registry.register(fallback)

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Custom")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen._do_delete(screen._selected_draft_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        assert registry.get("custom") is None
        assert not (tmp_path / "agents" / "custom.yaml").exists()
        assert screen.query_one("#ac-save", Button).disabled is True
        assert screen._cancel_result() == "switched"


@pytest.mark.asyncio
async def test_agent_config_delete_active_then_save_promoted_edit_still_switches(tmp_path: Path) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = AgentProfileRegistry()
    active = AgentProfile(
        name="active",
        display_name="Active",
        description="Active profile",
        instructions="Follow the user's instructions.",
    )
    fallback = AgentProfile(
        name="fallback",
        display_name="Fallback",
        description="Fallback profile",
        instructions="Follow the user's instructions.",
    )
    save_profile(active)
    save_profile(fallback)
    registry.register(active)
    registry.register(fallback)
    saved_callbacks: list[tuple[str | None, str | None]] = []

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(
            registry,
            current_profile="Active",
            active_profile_name="active",
            on_saved=lambda display, name: saved_callbacks.append((display, name)),
        )
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen._do_delete(screen._selected_draft_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#bc-display-name", Input).focus()
        screen.query_one("#bc-display-name", Input).value = "Fallback Edited"
        await pilot.pause()
        await _wait_for_panel_display_name(screen, pilot, "Fallback Edited")
        await _wait_for_condition(
            lambda: not screen.query_one("#ac-save", Button).disabled,
            pilot,
            description="save enabled after promoted-profile edit",
        )
        screen.query_one("#ac-save", Button).press()
        await _wait_for_condition(
            lambda: registry.get("fallback") is not None and registry.get("fallback").display_name == "Fallback Edited",
            pilot,
            description="promoted profile edit persisted",
        )
        await _wait_for_hydrated(screen, pilot)

        assert screen._cancel_result() == "switched"

    assert saved_callbacks[0] == (None, "fallback")
    assert registry.get("active") is None
    updated = registry.get("fallback")
    assert updated is not None
    assert updated.display_name == "Fallback Edited"
    assert not (tmp_path / "agents" / "active.yaml").exists()


@pytest.mark.asyncio
async def test_agent_config_saving_active_profile_as_acp_promotes_replacement_main(tmp_path: Path) -> None:
    """Converting the session-active profile to External ACP must hand main
    duty to another profile: the engine refuses an ACP main on reload, and
    the next startup would refuse the persisted config outright."""
    from chrys.service.profiles.agents.serializer import save_profile

    registry = AgentProfileRegistry()
    active = AgentProfile(
        name="active",
        display_name="Active",
        description="Active profile",
        instructions="Follow the user's instructions.",
    )
    fallback = AgentProfile(
        name="fallback",
        display_name="Fallback",
        description="Fallback profile",
        instructions="Follow the user's instructions.",
    )
    save_profile(active)
    save_profile(fallback)
    registry.register(active)
    registry.register(fallback)
    saved_callbacks: list[tuple[str | None, str | None]] = []

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(
            registry,
            current_profile="Active",
            initial_profile="active",
            active_profile_name="active",
            on_saved=lambda display, name: saved_callbacks.append((display, name)),
        )
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#bc-agent-type", Select).value = "acp"
        await _wait_for_condition(
            lambda: (
                screen._drafts.get(screen._selected_draft_key) is not None
                and screen._drafts[screen._selected_draft_key].profile.acp is not None
                and not screen._hydrating
            ),
            pilot,
            description="active profile converted to ACP",
        )
        await _activate_agent_config_tab(screen, pilot, "acp")
        command = screen.query_one("#acp-command", Input)
        command.focus()
        command.value = "external-agent"
        await _wait_for_condition(
            lambda: not screen.query_one("#ac-save", Button).disabled,
            pilot,
            description="save enabled after ACP conversion",
        )
        screen.query_one("#ac-save", Button).press()
        await _wait_for_condition(
            lambda: registry.get("active") is not None and registry.get("active").acp is not None,
            pilot,
            description="ACP conversion persisted",
        )
        await _wait_for_hydrated(screen, pilot)

        assert screen._cancel_result() == "switched"
        assert screen._active_profile_name == "fallback"
        assert screen._current_profile == "Fallback"

    # The queued switch must target the promoted main profile so the caller
    # publishes AgentProfileSwitch instead of reloading into the ACP main.
    assert (None, "fallback") in saved_callbacks
    persisted = registry.get("active")
    assert persisted is not None
    assert persisted.acp is not None
    assert persisted.sub_agent_only is True


@pytest.mark.asyncio
async def test_agent_config_delete_guard_includes_sub_agent_references() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="target",
            display_name="Target",
            description="Referenced profile",
            instructions="Follow the user's instructions.",
        )
    )
    registry.register(
        AgentProfile(
            name="fallback",
            display_name="Fallback",
            description="Fallback profile",
            instructions="Follow the user's instructions.",
        )
    )
    registry.register(
        AgentProfile(
            name="parent",
            display_name="Parent",
            description="Profile that still uses the target",
            instructions="Follow the user's instructions.",
            sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="target")]),
        )
    )
    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Target")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        draft = screen._drafts[screen._selected_draft_key]
        referencing = screen._referencing_profiles_for_delete(draft, {"target"})

    assert referencing == ["Parent"]


@pytest.mark.asyncio
async def test_agent_config_generated_names_skip_existing_profiles() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="Base",
            display_name="Base Agent",
            description="Visible source profile",
            instructions="Follow the user's instructions.",
        )
    )
    registry.register(
        AgentProfile(
            name="new-agent",
            display_name="Existing New Name",
            description="Existing profile using the default new-agent name",
            instructions="Follow the user's instructions.",
        )
    )
    registry.register(
        AgentProfile(
            name="Base-copy",
            display_name="Existing Clone Name",
            description="Existing profile using the default clone name",
            instructions="Follow the user's instructions.",
        )
    )
    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Base Agent")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#ac-new", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        new_profile = screen._drafts[screen._selected_draft_key].profile

        base_key = _draft_for_profile(screen, "Base").key
        screen._sync_selected_draft_from_panels()
        screen._load_profile(base_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#ac-clone", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        clone_profile = screen._drafts[screen._selected_draft_key].profile

    assert new_profile.name == "new-agent-2"
    assert clone_profile.name == "Base-copy-2"


@pytest.mark.asyncio
async def test_agent_config_saves_cross_rename_without_clobbering_profiles(tmp_path: Path) -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="A",
            display_name="Agent A",
            description="First agent",
            instructions="Follow the user's instructions.",
        )
    )
    registry.register(
        AgentProfile(
            name="B",
            display_name="Agent B",
            description="Second agent",
            instructions="Follow the user's instructions.",
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Agent A")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "B"
        await pilot.pause()
        screen._sync_selected_draft_from_panels()

        b_key = _draft_for_original(screen, "B").key
        screen._load_profile(b_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "C"
        await pilot.pause()

        screen.query_one("#ac-save", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    renamed_a = registry.get("B")
    renamed_b = registry.get("C")
    assert renamed_a is not None
    assert renamed_a.display_name == "Agent A"
    assert renamed_b is not None
    assert renamed_b.display_name == "Agent B"
    assert registry.get("A") is None
    assert (tmp_path / "agents" / "B.yaml").is_file()
    assert (tmp_path / "agents" / "C.yaml").is_file()


@pytest.mark.asyncio
async def test_agent_config_cross_rename_retargets_refs_by_profile_identity() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="Parent",
            display_name="Parent Agent",
            description="Uses both agents",
            instructions="Follow the user's instructions.",
            sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="A"), SubAgentRef(profile="B")]),
        )
    )
    registry.register(
        AgentProfile(
            name="A",
            display_name="Agent A",
            description="First agent",
            instructions="Follow the user's instructions.",
        )
    )
    registry.register(
        AgentProfile(
            name="B",
            display_name="Agent B",
            description="Second agent",
            instructions="Follow the user's instructions.",
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Agent A")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "B"
        await pilot.pause()
        screen._sync_selected_draft_from_panels()

        b_key = _draft_for_original(screen, "B").key
        screen._load_profile(b_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "C"
        await pilot.pause()

        screen.query_one("#ac-save", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    parent = registry.get("Parent")
    renamed_a = registry.get("B")
    renamed_b = registry.get("C")
    assert parent is not None
    assert renamed_a is not None
    assert renamed_a.display_name == "Agent A"
    assert renamed_b is not None
    assert renamed_b.display_name == "Agent B"
    assert [ref.profile for ref in parent.sub_agents.agents] == ["B", "C"]


@pytest.mark.asyncio
async def test_agent_config_save_retargets_subagent_refs_after_rename() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="Parent",
            display_name="Parent Agent",
            description="Uses a child agent",
            instructions="Follow the user's instructions.",
            sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Child")]),
        )
    )
    registry.register(
        AgentProfile(
            name="Child",
            display_name="Child Agent",
            description="Child agent",
            instructions="Follow the user's instructions.",
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Child Agent")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "Helper"
        save = screen.query_one("#ac-save", Button)
        await _wait_for_condition(lambda: not save.disabled, pilot, description="save enabled after profile rename")
        save.press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    parent = registry.get("Parent")
    assert parent is not None
    assert [ref.profile for ref in parent.sub_agents.agents] == ["Helper"]
    assert registry.get("Helper") is not None
    assert registry.get("Child") is None


@pytest.mark.asyncio
async def test_agent_config_staged_rename_keeps_original_subagent_ref_selectable_until_save() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="Parent",
            display_name="Parent Agent",
            description="Uses a child agent",
            instructions="Follow the user's instructions.",
            sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Child")]),
        )
    )
    registry.register(
        AgentProfile(
            name="Child",
            display_name="Child Agent",
            description="Child agent",
            instructions="Follow the user's instructions.",
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Child Agent")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "Helper"
        await pilot.pause()
        screen._sync_selected_draft_from_panels()

        parent_key = _draft_for_original(screen, "Parent").key
        screen._load_profile(parent_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        await _activate_agent_config_tab(screen, pilot, "sub-agents")
        await _wait_for_selectors(screen, pilot, "#sa-profile-0")
        select = screen.query_one("#sa-profile-0", Select)
        # SubAgentCard.get_config() falls back to its seed profile name when
        # the Select reactive hasn't settled, so _wait_for_hydrated returns
        # while the widget itself is still on Select.NULL under xdist load on
        # Windows CI.  Poll the widget directly before asserting.
        await _wait_for_condition(
            lambda: select.value == "Child",
            pilot,
            description="sub-agent select reflects staged-rename original name",
        )
        assert select.value == "Child"

        screen.query_one("#ac-save", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    parent = registry.get("Parent")
    assert parent is not None
    assert [ref.profile for ref in parent.sub_agents.agents] == ["Helper"]
    assert registry.get("Helper") is not None
    assert registry.get("Child") is None


@pytest.mark.asyncio
async def test_agent_config_failed_rename_validation_rolls_back_retargeted_draft_refs() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="Parent",
            display_name="Parent Agent",
            description="Uses a child agent",
            instructions="Follow the user's instructions.",
            sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Child")]),
        )
    )
    registry.register(
        AgentProfile(
            name="Child",
            display_name="Child Agent",
            description="Child agent",
            instructions="Follow the user's instructions.",
        )
    )
    registry.register(
        AgentProfile(
            name="Helper",
            display_name="Existing Helper",
            description="Existing helper agent",
            instructions="Follow the user's instructions.",
        )
    )
    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Child Agent")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "Helper"
        await pilot.pause()
        screen.query_one("#ac-save", Button).press()
        await pilot.pause()

        parent_draft = _draft_for_original(screen, "Parent")
        assert [ref.profile for ref in parent_draft.profile.sub_agents.agents] == ["Child"]
        assert registry.get("Child") is not None
        assert registry.get("Helper") is not None

        screen.query_one("#bc-name", Input).value = "Other"
        await pilot.pause()
        screen.query_one("#ac-save", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    parent = registry.get("Parent")
    assert parent is not None
    assert [ref.profile for ref in parent.sub_agents.agents] == ["Other"]
    assert registry.get("Other") is not None
    assert registry.get("Child") is None


@pytest.mark.asyncio
async def test_agent_config_save_persists_retargeted_subagent_refs_after_rename(tmp_path: Path) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = AgentProfileRegistry()
    child = AgentProfile(
        name="Child",
        display_name="Child Agent",
        description="Child agent",
        instructions="Follow the user's instructions.",
    )
    fallback = AgentProfile(
        name="Fallback",
        display_name="Fallback Agent",
        description="Fallback agent",
        instructions="Follow the user's instructions.",
    )
    parent = AgentProfile(
        name="parent",
        display_name="Parent",
        description="Parent profile",
        instructions="Follow the user's instructions.",
        sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Child")]),
    )
    for profile in (child, fallback, parent):
        save_profile(profile)
        registry.register(profile)

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Child Agent")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "Helper"
        save = screen.query_one("#ac-save", Button)
        await _wait_for_condition(lambda: not save.disabled, pilot, description="save enabled after retarget rename")
        save.press()
        await pilot.pause()
        await _wait_for_condition(
            lambda: (
                (saved_parent := registry.get("parent")) is not None
                and [ref.profile for ref in saved_parent.sub_agents.agents] == ["Helper"]
            ),
            pilot,
            description="renamed sub-agent refs persisted",
        )

    parent = registry.get("parent")
    assert parent is not None
    assert [ref.profile for ref in parent.sub_agents.agents] == ["Helper"]
    saved_parent = load_profile_from_yaml(tmp_path / "agents" / "parent.yaml")
    assert [ref.profile for ref in saved_parent.sub_agents.agents] == ["Helper"]


@pytest.mark.asyncio
async def test_agent_config_save_validates_retargeted_profiles_before_writing(tmp_path: Path) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = AgentProfileRegistry()
    child = AgentProfile(
        name="Child",
        display_name="Child Agent",
        description="Child agent",
        instructions="Follow the user's instructions.",
    )
    sibling = AgentProfile(
        name="Sibling",
        display_name="Sibling Agent",
        description="Sibling agent",
        instructions="Follow the user's instructions.",
    )
    parent = AgentProfile(
        name="parent",
        display_name="Parent",
        description="Parent profile",
        instructions="Follow the user's instructions.",
        sub_agents=SubAgentsConfig(
            agents=[
                SubAgentRef(profile="Child"),
                SubAgentRef(profile="Sibling", tool_name="Helper"),
            ]
        ),
    )
    for profile in (child, sibling, parent):
        save_profile(profile)
        registry.register(profile)
    original_parent_yaml = (tmp_path / "agents" / "parent.yaml").read_bytes()

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Child Agent")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "Helper"
        await pilot.pause()
        screen.query_one("#ac-save", Button).press()
        await pilot.pause()

    assert registry.get("Child") is not None
    assert registry.get("Helper") is None
    parent = registry.get("parent")
    assert parent is not None
    assert [ref.profile for ref in parent.sub_agents.agents] == ["Child", "Sibling"]
    assert (tmp_path / "agents" / "parent.yaml").read_bytes() == original_parent_yaml


@pytest.mark.asyncio
async def test_agent_config_save_persists_retargeted_builtin_shadow(tmp_path: Path) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = AgentProfileRegistry()
    registry.load_builtins()
    child = AgentProfile(
        name="Child",
        display_name="Child Agent",
        description="Child agent",
        instructions="Follow the user's instructions.",
    )
    code_shadow = AgentProfile(
        name="Code",
        display_name="Code Agent",
        description="User shadow of built-in Code",
        instructions="Follow the user's instructions.",
        sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Child")]),
    )
    save_profile(code_shadow)
    registry.register(child)
    registry.register(code_shadow)
    assert registry.is_builtin("Code") is True

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Child Agent")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "Helper"
        await pilot.pause()
        # Wait until Input.Changed has been harvested into the draft.
        # On slower Windows CI, pressing Save before the rename lands on
        # the draft causes the retarget pass to find an empty rename_map.
        await _wait_for_selected_profile_name(screen, pilot, "Helper")
        screen.query_one("#ac-save", Button).press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    shadow = registry.get("Code")
    assert shadow is not None
    assert [ref.profile for ref in shadow.sub_agents.agents] == ["Helper"]
    saved_shadow = load_profile_from_yaml(tmp_path / "agents" / "Code.yaml")
    assert [ref.profile for ref in saved_shadow.sub_agents.agents] == ["Helper"]


@pytest.mark.asyncio
async def test_agent_config_case_only_rename_keeps_case_alias_target_file(tmp_path: Path) -> None:
    """Case-only rename keeps the target file; failure rollback is covered by _apply_all snapshots."""
    from chrys.service.profiles.agents.serializer import save_profile

    registry = AgentProfileRegistry()
    profile = AgentProfile(
        name="foo",
        display_name="Foo Agent",
        description="Custom profile",
        instructions="Follow the user's instructions.",
    )
    save_profile(profile)
    registry.register(profile)

    old_path = tmp_path / "agents" / "foo.yaml"
    alias_path = tmp_path / "agents" / "Foo.yaml"
    if not alias_path.exists():
        # A case-only rename is only a real scenario on a case-insensitive
        # filesystem, where ``Foo.yaml`` already resolves to ``foo.yaml``. The
        # former symlink stand-in no longer models this: owner-only writes are
        # ``O_NOFOLLOW`` and would replace the link with a distinct file, so the
        # case-sensitive runner (Linux) is not a valid host for this assertion.
        pytest.skip("case-only rename requires a case-insensitive filesystem")

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Foo Agent")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "Foo"
        save = screen.query_one("#ac-save", Button)
        await _wait_for_selected_profile_name(screen, pilot, "Foo")
        await _wait_for_condition(lambda: not save.disabled, pilot, description="save enabled after case-only rename")
        save.press()
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

    renamed = registry.get("Foo")
    assert renamed is not None
    assert registry.get("foo") is None
    assert old_path.is_file()
    saved = load_profile_from_yaml(alias_path)
    assert saved.name == "Foo"


@pytest.mark.asyncio
async def test_agent_config_save_failure_rolls_back_cross_rename_files_and_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.service.profiles.agents.serializer import save_profile

    registry = AgentProfileRegistry()
    profile_a = AgentProfile(
        name="A",
        display_name="Agent A",
        description="First agent",
        instructions="Follow the user's instructions.",
    )
    profile_b = AgentProfile(
        name="B",
        display_name="Agent B",
        description="Second agent",
        instructions="Follow the user's instructions.",
    )
    save_profile(profile_a)
    save_profile(profile_b)
    registry.register(profile_a)
    registry.register(profile_b)
    path_a = tmp_path / "agents" / "A.yaml"
    path_b = tmp_path / "agents" / "B.yaml"
    original_a = path_a.read_bytes()
    original_b = path_b.read_bytes()

    original_save_profile = save_profile

    def fail_on_c(profile, *args, **kwargs):
        if profile.name == "C":
            raise OSError("simulated save failure")
        return original_save_profile(profile, *args, **kwargs)

    monkeypatch.setattr("chrys.service.profiles.agents.serializer.save_profile", fail_on_c)

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Agent A")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "B"
        await pilot.pause()
        screen._sync_selected_draft_from_panels()

        b_key = _draft_for_original(screen, "B").key
        screen._load_profile(b_key)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)
        screen.query_one("#bc-name", Input).focus()
        screen.query_one("#bc-name", Input).value = "C"
        await pilot.pause()

        screen.query_one("#ac-save", Button).press()
        await pilot.pause()

    restored_a = registry.get("A")
    restored_b = registry.get("B")
    assert restored_a is not None
    assert restored_a.display_name == "Agent A"
    assert restored_b is not None
    assert restored_b.display_name == "Agent B"
    assert registry.get("C") is None
    assert path_a.read_bytes() == original_a
    assert path_b.read_bytes() == original_b
    assert not (tmp_path / "agents" / "C.yaml").exists()


@pytest.mark.asyncio
async def test_agent_config_clone_builtin_saves_editable_custom_copy(
    tmp_path: Path,
) -> None:
    registry = _registry()
    original = registry.get("Code")
    assert original is not None

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#ac-clone", Button).press()
        await pilot.pause()
        await _activate_agent_config_tab(screen, pilot, "tools")
        await _wait_for_selectors(
            screen,
            pilot,
            "#bc-name",
            "#bc-model-use-active",
            "#tc-cat-filesystem-read",
            "#tc-cat-sleep",
        )
        # The input mounts in its builtin (disabled) state then transitions
        # to enabled for the custom clone — wait for that transition.
        await _wait_for_input_enabled(screen, pilot, "#bc-name")
        await _wait_for_hydrated(screen, pilot)

        copied = screen._drafts[screen._selected_draft_key].profile
        assert screen.query_one("#tc-cat-sleep", Checkbox).value is True
        save = screen.query_one("#ac-save", Button)
        await _wait_for_condition(lambda: not save.disabled, pilot, description="save enabled after builtin clone")
        # Exercise the synchronous save transaction directly.  The button
        # handler intentionally swallows validation/save errors into UI
        # notifications, which turns rare CI timing races into opaque waits.
        # This test is about clone persistence, so failures should surface as
        # direct exceptions from the save path.
        screen._apply_all()
        saved = registry.get(copied.name)
        assert saved is not None, _agent_config_clone_save_debug(screen, registry, copied.name)
        await _wait_for_hydrated(screen, pilot)
        await _activate_agent_config_tab(screen, pilot, "basic")
        name_input = screen.query_one("#bc-name", Input)

    assert copied.name == "Code-copy"
    assert copied.id
    assert copied.id != original.id
    assert copied.display_name == f"{original.display_name or original.name} Copy"
    assert copied.description == original.description
    assert copied.instructions == original.instructions
    assert set(copied.tools.builtins) == set(original.tools.builtins)
    assert copied.tools.custom == original.tools.custom
    assert copied.tools.mcp == original.tools.mcp
    assert copied.tools.shell_filter == original.tools.shell_filter
    assert copied.skills.paths == original.skills.paths
    assert copied.skills.inline == original.skills.inline
    assert copied.skills.script_timeout == original.skills.script_timeout
    assert set(copied.skills.script_extensions) == set(original.skills.script_extensions)
    assert copied.skills.auto_load_user_agents_skills == original.skills.auto_load_user_agents_skills
    assert copied.skills.auto_load_cwd_agents_skills == original.skills.auto_load_cwd_agents_skills
    assert copied.approval == original.approval
    assert copied.sub_agents.max_total_concurrency == original.sub_agents.max_total_concurrency
    assert [ref.profile for ref in copied.sub_agents.agents] == [ref.profile for ref in original.sub_agents.agents]
    assert [ref.tool_name for ref in copied.sub_agents.agents] == [ref.tool_name for ref in original.sub_agents.agents]
    assert [ref.max_concurrency for ref in copied.sub_agents.agents] == [
        ref.max_concurrency for ref in original.sub_agents.agents
    ]
    assert copied.model == original.model
    assert saved is not None
    assert set(saved.tools.builtins) == set(original.tools.builtins)
    assert "sleep" in saved.tools.builtins
    assert [ref.profile for ref in saved.sub_agents.agents] == [ref.profile for ref in original.sub_agents.agents]
    assert [ref.tool_name for ref in saved.sub_agents.agents] == [ref.tool_name for ref in original.sub_agents.agents]
    assert not registry.is_builtin(copied.name)
    assert name_input.disabled is False
    assert (tmp_path / "agents" / f"{copied.name}.yaml").is_file()


@pytest.mark.asyncio
async def test_agent_config_save_during_clone_hydration_uses_staged_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#ac-clone", Button).press()
        await pilot.pause()
        copied = screen._drafts[screen._selected_draft_key].profile

        def fail_if_panel_validation_runs() -> list[str]:
            if screen._hydrating:
                raise AssertionError("hydrating save should not read transient panel state")
            return []

        monkeypatch.setattr(screen, "_validate_all", fail_if_panel_validation_runs)
        screen._hydrating = True
        screen.query_one("#ac-save", Button).press()
        await pilot.pause()

    saved = registry.get(copied.name)
    assert saved is not None
    assert saved.name == copied.name


@pytest.mark.asyncio
async def test_agent_config_subagent_add_empty_card_uses_valid_blank_selection() -> None:
    registry = _registry()

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Explore", initial_tab="sub-agents")
        await app.push_screen(screen)
        await pilot.pause()

        screen.query_one("#sa-add-btn", Button).press()
        await pilot.pause()

        select = screen.query_one("#sa-profile-0", Select)

    assert select.value is Select.NULL


@pytest.mark.asyncio
async def test_agent_config_clone_increments_existing_clone_suffix() -> None:
    # Clones Explore (no sub-agents) rather than Code: repeated cloning of a
    # sub-agent-bearing profile mounts a Textual Select per ref, which trips an
    # upstream Mount-time race (SelectOverlay queried before compose) on some
    # platforms. The sub-agent-bearing clone path is exercised separately by
    # test_agent_config_clone_builtin_saves_editable_custom_copy.
    registry = _registry()
    original = registry.get("Explore")
    assert original is not None

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Explore")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        for expected in ("Explore-copy", "Explore-copy-2", "Explore-copy-3"):
            screen.query_one("#ac-clone", Button).press()
            await _wait_for_selected_profile_name(screen, pilot, expected)

        selected = screen._drafts[screen._selected_draft_key].profile

    assert selected.name == "Explore-copy-3"
    assert selected.display_name == f"{original.display_name or original.name} Copy 3"
    assert registry.get("Explore-copy") is None
    assert registry.get("Explore-copy-2") is None
    assert registry.get("Explore-copy-3") is None


@pytest.mark.asyncio
async def test_agent_config_clone_custom_profile_increments_clone_suffix_without_root_profile(
    tmp_path: Path,
) -> None:
    registry = AgentProfileRegistry()
    profile = AgentProfile(
        name="custom-copy-2",
        id="custom-copy-id",
        display_name="Custom copy 2",
        description="Custom profile",
        instructions="Follow the user's instructions.",
    )
    registry.register(profile)

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Custom copy 2")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        screen.query_one("#ac-clone", Button).press()
        await _wait_for_selected_profile_name(screen, pilot, "custom-copy-3")

        selected = screen._drafts[screen._selected_draft_key].profile

    assert selected.name == "custom-copy-3"
    assert selected.display_name == "Custom Copy 3"
    assert selected.id
    assert selected.id != profile.id
    assert not (tmp_path / "agents" / f"{selected.name}.yaml").exists()


@pytest.mark.asyncio
async def test_agent_config_footer_buttons_stay_inside_modal_after_clone() -> None:
    registry = _registry()

    app = _AgentConfigApp()
    async with app.run_test(size=(90, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code")
        await app.push_screen(screen)
        await pilot.pause()

        screen.query_one("#ac-clone", Button).press()
        await pilot.pause()

        container = screen.query_one("#ac-container", Vertical)
        container_left = container.region.x
        container_right = container.region.x + container.region.width
        buttons = [
            screen.query_one(f"#{button_id}", Button)
            for button_id in ("ac-new", "ac-clone", "ac-reset", "ac-delete", "ac-save", "ac-cancel")
        ]

        visible_buttons = [button for button in buttons if button.display]
        assert all(container_left <= button.region.x for button in visible_buttons)
        assert all(button.region.x + button.region.width <= container_right for button in visible_buttons)


@pytest.mark.asyncio
async def test_agent_config_read_only_hides_mutations_and_disables_controls(tmp_path: Path) -> None:
    registry = _registry()

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code", read_only=True)
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_hydrated(screen, pilot)

        assert screen.query_one("#bc-display-name", Input).disabled is True
        assert screen.query_one("#bc-sub-agent-only", Checkbox).disabled is True
        assert screen.query_one("#bc-description", TextArea).read_only is True
        assert screen.query_one("#ac-cancel", Button).display is True
        assert screen.query_one("#ac-cancel", Button).disabled is False
        notice = screen.query_one("#ac-read-only-notice", Static)
        assert notice.display is True
        assert notice.render().plain == "• Agent is running. This page is read-only."
        assert screen.query_one("#ac-buttons-spacer", Static).display is True
        footer = screen.query_one("#ac-footer", Vertical)
        close = screen.query_one("#ac-cancel", Button)
        assert notice.region.y == footer.region.y + footer.region.height - 1
        assert notice.region.y > close.region.y
        assert notice.region.x == footer.region.x + 1
        assert notice.region.width == footer.region.width - 2
        await screen.query_one("#basic").mount(Button("Future", id="future-panel-action"))
        screen._apply_read_only_state()
        future_button = screen.query_one("#future-panel-action", Button)
        assert future_button.display is True
        assert future_button.disabled is True
        for button_id in ("ac-new", "ac-clone", "ac-reset", "ac-delete", "ac-save"):
            button = screen.query_one(f"#{button_id}", Button)
            assert button.display is False
            assert button.disabled is True

        draft_count = len(screen._drafts)
        screen.query_one("#ac-clone", Button).press()
        screen.query_one("#ac-save", Button).press()
        await pilot.pause()

        assert len(screen._drafts) == draft_count
        assert screen._cancel_result() == ""
        assert not (tmp_path / "agents" / "Code.yaml").exists()


@pytest.mark.asyncio
async def test_agent_config_read_only_applies_to_lazy_tabs(tmp_path: Path) -> None:
    from chrys.app.tui.screens.agents.panels.memory import MemoryFileCard, MemoryFolderCard

    registry = _registry()
    registry.register(
        AgentProfile(
            name="memory-agent",
            display_name="Memory Agent",
            description="Agent with memory paths",
            sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Explore")]),
            tools=ToolsConfig(mcp=[MCPServerConfig(name="local", transport="stdio", command="python", args=["-V"])]),
            skills=SkillsConfig(paths=[str(tmp_path / "skills")]),
            memory=MemoryConfig(files=[str(tmp_path / "notes.md")], folders=[str(tmp_path / "docs")]),
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Memory Agent", read_only=True)
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.query_one("#ac-tabs ContentSwitcher", ContentSwitcher).disabled is False

        screen.query_one("#ac-tabs", TabbedContent).active = "tools"
        await pilot.pause()
        assert screen.query_one("#tc-cat-filesystem-read", Checkbox).disabled is True

        screen.query_one("#ac-tabs", TabbedContent).active = "sub-agents"
        await pilot.pause()
        await _wait_for_selectors(screen, pilot, "#sa-profile-0", "#sa-delete-btn-0")
        sub_agent_select = screen.query_one("#sa-profile-0", Select)
        sub_agent_delete = screen.query_one("#sa-delete-btn-0", Button)
        assert sub_agent_select.disabled is True
        assert sub_agent_select.is_disabled is True
        assert sub_agent_delete.display is False
        assert sub_agent_delete.disabled is True

        screen.query_one("#ac-tabs", TabbedContent).active = "mcp"
        await pilot.pause()
        await _wait_for_selectors(screen, pilot, "#mcp-test-btn-0")
        mcp_test = screen.query_one("#mcp-test-btn-0", Button)
        assert mcp_test.display is False
        assert mcp_test.disabled is True
        assert screen.query_one("#mcp-name-0", Input).disabled is True
        assert screen.query_one("#mcp-add-btn", Button).display is False

        screen.query_one("#ac-tabs", TabbedContent).active = "skills"
        await pilot.pause()
        await _wait_for_selectors(screen, pilot, "#sk-add-btn", "#sk-ext-py")
        assert screen.query_one("#sk-add-btn", Button).display is False
        assert screen.query_one("#sk-ext-py", Button).display is True
        assert screen.query_one("#sk-ext-py", Button).disabled is True
        await _wait_for_selectors(screen, pilot, "#sk-path-0", "#sk-browse-0", "#sk-delete-btn-0")
        assert screen.query_one("#sk-path-0", Input).disabled is True
        assert screen.query_one("#sk-browse-0", Button).display is False
        assert screen.query_one("#sk-browse-0", Button).disabled is True
        assert screen.query_one("#sk-delete-btn-0", Button).display is False
        assert screen.query_one("#sk-delete-btn-0", Button).disabled is True

        screen.query_one("#ac-tabs", TabbedContent).active = "memory"
        await pilot.pause()
        await _wait_for_selectors(
            screen,
            pilot,
            "#mem-file-path-0",
            "#mem-file-browse-0",
            "#mem-delete-btn-0",
            "#mem-folder-path-0",
        )
        memory_file_card = screen.query_one(MemoryFileCard)
        memory_folder_card = screen.query_one(MemoryFolderCard)
        memory_file_delete = memory_file_card.query_one(".config-card-delete-btn", Button)
        memory_folder_delete = memory_folder_card.query_one(".config-card-delete-btn", Button)
        assert screen.query_one("#mem-add-file", Button).display is False
        assert screen.query_one("#mem-add-folder", Button).display is False
        assert screen.query_one("#mem-file-path-0", Input).disabled is True
        assert screen.query_one("#mem-file-browse-0", Button).display is False
        assert screen.query_one("#mem-file-browse-0", Button).disabled is True
        assert memory_file_delete.display is False
        assert memory_file_delete.disabled is True
        assert screen.query_one("#mem-folder-path-0", Input).disabled is True
        assert memory_folder_delete.display is False
        assert memory_folder_delete.disabled is True


@pytest.mark.asyncio
async def test_agent_config_subagent_select_mount_race_does_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _registry()
    original_update = SelectCurrent.update
    tripped = False

    def flaky_update(self: SelectCurrent, label: object) -> None:
        nonlocal tripped
        parent = self.parent
        if parent is not None and parent.id == "sa-profile-0" and not tripped:
            tripped = True
            raise NoMatches("No nodes match '#label'")
        original_update(self, label)

    monkeypatch.setattr(SelectCurrent, "update", flaky_update)

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code", initial_tab="sub-agents")
        await app.push_screen(screen)
        await pilot.pause()
        await _wait_for_condition(
            lambda: str(screen.query_one("#sa-profile-0", Select).query_one(SelectCurrent).label) == "Explore Agent",
            pilot,
            description="sub-agent profile select visual sync",
        )

        select = screen.query_one("#sa-profile-0", Select)
        current = select.query_one(SelectCurrent)

    assert tripped is True
    assert select.value == "Explore"
    assert str(current.label) == "Explore Agent"


@pytest.mark.asyncio
async def test_agent_config_subagent_select_overlay_mount_race_does_not_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry()
    original_setup = Select._setup_options_renderables
    tripped = False

    def flaky_setup(self: Select) -> None:
        nonlocal tripped
        if self.id == "sa-profile-0" and not tripped:
            tripped = True
            raise NoMatches("No nodes match 'SelectOverlay'")
        original_setup(self)

    monkeypatch.setattr(Select, "_setup_options_renderables", flaky_setup)

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Code", initial_tab="sub-agents")
        await app.push_screen(screen)
        await pilot.pause()
        await pilot.pause()

        select = screen.query_one("#sa-profile-0", Select)
        current = select.query_one(SelectCurrent)
        overlay = select.query_one(SelectOverlay)

    assert tripped is True
    assert select.value == "Explore"
    assert str(current.label) == "Explore Agent"
    assert overlay.option_count > 1


@pytest.mark.asyncio
async def test_subagent_concurrency_placeholders_match_defaults() -> None:
    from chrys.app.tui.screens.agents.panels.subagents import SubAgentCard, SubAgentsConfigPanel

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SubAgentsConfigPanel(
            sub_agents_config=SubAgentsConfig(agents=[SubAgentRef(profile="Explore")]),
            registry=_registry(),
            current_profile_name="Code",
        )
        await app.mount(panel)
        await _wait_for_condition(
            lambda: len(list(panel.query(SubAgentCard))) == 1,
            pilot,
            description="sub-agent card",
        )

        assert panel.query_one("#sa-max-total", Input).placeholder == "3"
        assert panel.query_one("#sa-max-conc-0", Input).placeholder == "3"


def test_subagents_get_config_falls_back_to_seed_when_cards_have_not_mounted() -> None:
    """Save fired before cards finish mounting must use the seed _config.

    Pins the Windows-CI fix where a single ``pilot.pause()`` after Clone
    returned before ``_rebuild_cards`` had pushed every ``SubAgentCard``
    into the DOM, so ``get_config`` saw zero cards and silent-drop
    persisted an empty sub_agents list over a profile with three refs.
    """
    from chrys.app.tui.screens.agents.panels.subagents import SubAgentCard, SubAgentsConfigPanel
    from chrys.service.profiles.agents.schema import SubAgentRef, SubAgentsConfig

    seed = SubAgentsConfig(
        max_total_concurrency=4,
        agents=[
            SubAgentRef(profile="Explore", tool_name="t_explore", tool_description="d1", max_concurrency=2),
            SubAgentRef(profile="Plan", tool_name="t_plan", tool_description="d2", max_concurrency=1),
            SubAgentRef(profile="", tool_name="blank", tool_description="d3", max_concurrency=1),
        ],
    )
    panel = SubAgentsConfigPanel(sub_agents_config=seed, registry=_registry(), current_profile_name="Code")

    # No cards mounted — simulates the post-Clone window before _rebuild_cards
    # has dispatched all SubAgentCard mounts.
    assert list(panel.query(SubAgentCard)) == []

    cfg = panel.get_config()
    assert [a.profile for a in cfg.agents] == ["Explore", "Plan"]
    assert [a.tool_name for a in cfg.agents] == ["t_explore", "t_plan"]
    assert cfg.max_total_concurrency == 4


@pytest.mark.asyncio
async def test_add_subagent_inserts_new_card_before_existing_refs() -> None:
    from chrys.app.tui.screens.agents.panels.subagents import SubAgentCard, SubAgentsConfigPanel

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SubAgentsConfigPanel(
            sub_agents_config=SubAgentsConfig(agents=[SubAgentRef(profile="Explore")]),
            registry=_registry(),
            current_profile_name="Code",
        )
        await app.mount(panel)
        await _wait_for_condition(
            lambda: len(list(panel.query(SubAgentCard))) == 1,
            pilot,
            description="initial sub-agent card",
        )

        panel.query_one("#sa-add-btn", Button).press()
        await _wait_for_condition(
            lambda: len(list(panel.query(SubAgentCard))) == 2,
            pilot,
            description="new sub-agent card",
        )
        await _wait_for_selectors(panel, pilot, "#sa-profile-0", "#sa-profile-1")

        cards = list(panel.query(SubAgentCard))
        cards[0].query_one("#sa-profile-0", Select).value = "Plan"
        await pilot.pause()

        assert cards[1].query_one("#sa-profile-1", Select).value == "Explore"
        assert [ref.profile for ref in panel.get_config().agents] == ["Plan", "Explore"]


@pytest.mark.asyncio
async def test_add_subagent_preserves_invalid_existing_max_concurrency() -> None:
    from chrys.app.tui.screens.agents.panels.subagents import SubAgentCard, SubAgentsConfigPanel

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SubAgentsConfigPanel(
            sub_agents_config=SubAgentsConfig(agents=[SubAgentRef(profile="Explore", max_concurrency=2)]),
            registry=_registry(),
            current_profile_name="Code",
        )
        await app.mount(panel)
        await _wait_for_condition(
            lambda: len(list(panel.query(SubAgentCard))) == 1,
            pilot,
            description="initial sub-agent card",
        )
        await _wait_for_selectors(panel, pilot, "#sa-profile-0", "#sa-max-conc-0")
        await _wait_for_condition(
            lambda: next(iter(panel.query(SubAgentCard))).has_profile,
            pilot,
            description="initial sub-agent profile selection",
        )

        cards = list(panel.query(SubAgentCard))
        cards[0].query_one("#sa-max-conc-0", Input).value = "abc"
        assert any("max concurrency must be a valid integer" in error for error in cards[0].validate())

        panel.query_one("#sa-add-btn", Button).press()

        await _wait_for_condition(
            lambda: len(list(panel.query(SubAgentCard))) == 2,
            pilot,
            description="new sub-agent card",
        )
        await _wait_for_selectors(panel, pilot, "#sa-profile-0", "#sa-profile-1", "#sa-max-conc-1")
        await _wait_for_condition(
            lambda: len(list(panel.query(SubAgentCard))) == 2 and list(panel.query(SubAgentCard))[1].has_profile,
            pilot,
            description="existing sub-agent profile selection after add",
        )

        cards = list(panel.query(SubAgentCard))
        assert cards[1].query_one("#sa-profile-1", Select).value == "Explore"
        assert cards[1].query_one("#sa-max-conc-1", Input).value == "abc"
        assert any("max concurrency must be a valid integer" in error for error in panel.validate())


@pytest.mark.asyncio
async def test_remove_subagent_preserves_edits_in_remaining_refs() -> None:
    from chrys.app.tui.screens.agents.panels.subagents import SubAgentCard, SubAgentsConfigPanel

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SubAgentsConfigPanel(
            sub_agents_config=SubAgentsConfig(
                agents=[
                    SubAgentRef(profile="Explore"),
                    SubAgentRef(profile="Plan"),
                ]
            ),
            registry=_registry(),
            current_profile_name="Code",
        )
        await app.mount(panel)
        await _wait_for_condition(
            lambda: len(list(panel.query(SubAgentCard))) == 2,
            pilot,
            description="initial sub-agent cards",
        )
        await _wait_for_selectors(panel, pilot, "#sa-profile-0", "#sa-profile-1")
        await _wait_for_condition(
            lambda: list(panel.query(SubAgentCard))[1].query_one("#sa-profile-1", Select).value == "Plan",
            pilot,
            description="surviving sub-agent profile selection",
        )

        cards = list(panel.query(SubAgentCard))
        cards[1].query_one("#sa-tool-name-1", Input).value = "plan_custom"
        cards[1].query_one("#sa-tool-desc-1", TextArea).text = "edited plan description"
        cards[1].query_one("#sa-max-conc-1", Input).value = "5"
        cards[0].query_one("#sa-delete-btn-0", Button).press()

        await _wait_for_condition(
            lambda: len(list(panel.query(SubAgentCard))) == 1,
            pilot,
            description="remaining sub-agent card",
        )
        await _wait_for_selectors(panel, pilot, "#sa-profile-0", "#sa-tool-name-0", "#sa-tool-desc-0")
        await _wait_for_condition(
            lambda: next(iter(panel.query(SubAgentCard))).query_one("#sa-profile-0", Select).value == "Plan",
            pilot,
            description="remaining sub-agent profile selection after remove",
        )

        remaining = next(iter(panel.query(SubAgentCard)))
        assert remaining.query_one("#sa-profile-0", Select).value == "Plan"
        assert remaining.query_one("#sa-tool-name-0", Input).value == "plan_custom"
        assert remaining.query_one("#sa-tool-desc-0", TextArea).text == "edited plan description"
        assert remaining.query_one("#sa-max-conc-0", Input).value == "5"
        assert [
            (ref.profile, ref.tool_name, ref.tool_description, ref.max_concurrency) for ref in panel.get_config().agents
        ] == [("Plan", "plan_custom", "edited plan description", 5)]


@pytest.mark.asyncio
async def test_remove_subagent_preserves_invalid_existing_max_concurrency() -> None:
    from chrys.app.tui.screens.agents.panels.subagents import SubAgentCard, SubAgentsConfigPanel

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        panel = SubAgentsConfigPanel(
            sub_agents_config=SubAgentsConfig(
                agents=[
                    SubAgentRef(profile="Explore"),
                    SubAgentRef(profile="Plan", max_concurrency=2),
                ]
            ),
            registry=_registry(),
            current_profile_name="Code",
        )
        await app.mount(panel)
        await _wait_for_condition(
            lambda: len(list(panel.query(SubAgentCard))) == 2,
            pilot,
            description="initial sub-agent cards",
        )
        await _wait_for_selectors(panel, pilot, "#sa-profile-0", "#sa-profile-1", "#sa-max-conc-1")
        await _wait_for_condition(
            lambda: list(panel.query(SubAgentCard))[1].has_profile,
            pilot,
            description="surviving sub-agent profile selection",
        )

        cards = list(panel.query(SubAgentCard))
        cards[1].query_one("#sa-max-conc-1", Input).value = "abc"
        assert any("max concurrency must be a valid integer" in error for error in cards[1].validate())

        cards[0].query_one("#sa-delete-btn-0", Button).press()

        await _wait_for_condition(
            lambda: len(list(panel.query(SubAgentCard))) == 1,
            pilot,
            description="remaining sub-agent card",
        )
        await _wait_for_selectors(panel, pilot, "#sa-profile-0", "#sa-max-conc-0")
        await _wait_for_condition(
            lambda: next(iter(panel.query(SubAgentCard))).has_profile,
            pilot,
            description="remaining sub-agent profile selection after remove",
        )

        remaining = next(iter(panel.query(SubAgentCard)))
        assert remaining.query_one("#sa-profile-0", Select).value == "Plan"
        assert remaining.query_one("#sa-max-conc-0", Input).value == "abc"
        assert any("max concurrency must be a valid integer" in error for error in panel.validate())


def test_basic_get_config_preserves_seed_before_children_mount() -> None:
    profile = copy.deepcopy(_registry().get("Code"))
    assert profile is not None
    panel = BasicConfigPanel(profile)

    cfg = panel.get_config()

    assert cfg["name"] == profile.name
    assert cfg["display_name"] == profile.display_name
    assert cfg["description"] == profile.description
    assert cfg["sub_agent_only"] == profile.sub_agent_only
    assert cfg["model_profile_id"] == ""


def test_compaction_get_config_preserves_supplement_before_children_mount() -> None:
    panel = CompactionConfigPanel(CompactionConfig(last_words_template="Extra coding emphasis"))

    assert panel.get_config().last_words_template == "Extra coding emphasis"


@pytest.mark.asyncio
@pytest.mark.parametrize("supplement", ["", "Extra coding emphasis"])
async def test_compaction_panel_supplement_round_trip_without_default_prefill(supplement: str) -> None:
    class _CompactionPanelApp(App):
        def compose(self) -> ComposeResult:
            yield CompactionConfigPanel(CompactionConfig(last_words_template=supplement))

    app = _CompactionPanelApp()
    async with app.run_test(size=(80, 40)) as pilot:
        await _wait_for_selectors(app, pilot, "#cc-last-words-template")

        description = app.query_one(".cc-section-desc", Label)
        assert (
            "Compaction triggers automatically at the context window minus the model's maximum output tokens "
            "and a safety margin. Both limits are configured on the model profile." in str(description.render())
        )
        assert str(description.styles.height) == "auto"
        template = app.query_one("#cc-last-words-template", TextArea)
        assert template.text == supplement
        panel = app.query_one(CompactionConfigPanel)
        assert not panel.query("#cc-reserved-context-pct")
        assert panel.validate() == []
        assert panel.get_config().last_words_template == supplement


@pytest.mark.asyncio
async def test_empty_compaction_supplement_dirty_tracking_round_trips_on_revert() -> None:
    registry = AgentProfileRegistry()
    registry.register(
        AgentProfile(
            name="empty-supplement",
            display_name="Empty Supplement",
            description="Compaction supplement test profile",
            instructions="Follow the user's instructions.",
        )
    )

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="empty-supplement", initial_tab="compaction")
        await app.push_screen(screen)
        await _wait_for_hydrated(screen, pilot)

        save = screen.query_one("#ac-save", Button)
        template = screen.query_one("#cc-last-words-template", TextArea)
        selected_key = screen._selected_draft_key
        await _wait_for_condition(lambda: save.disabled, pilot, description="empty supplement starts clean")
        assert template.text == ""

        template.text = "Preserve exact benchmark results."
        await _wait_for_condition(lambda: not save.disabled, pilot, description="supplement edit marks dirty")
        assert screen._drafts[selected_key].dirty is True

        template.text = ""
        await _wait_for_condition(lambda: save.disabled, pilot, description="empty supplement revert clears dirty")
        assert screen._drafts[selected_key].dirty is False
        assert screen.query_one(CompactionConfigPanel).get_config().last_words_template == ""


@pytest.mark.asyncio
async def test_basic_sub_agent_only_description_matches_option_format() -> None:
    profile = copy.deepcopy(_registry().get("Code"))
    assert profile is not None

    class _BasicPanelApp(App):
        def compose(self) -> ComposeResult:
            yield BasicConfigPanel(profile)

    app = _BasicPanelApp()
    async with app.run_test(size=(46, 40)) as pilot:
        await pilot.pause()

        desc = app.query_one(".bc-option-desc", Label)
        await _wait_for_condition(
            lambda: desc.size.height > 1,
            pilot,
            description="sub-agent-only description wrap",
        )

        assert desc.render().plain == (
            "Focused helper agent called by other agents; cannot be selected as the main agent"
        )


@pytest.mark.asyncio
async def test_basic_active_model_profile_name_is_literal_checkbox_text() -> None:
    profile = copy.deepcopy(_registry().get("Code"))
    assert profile is not None
    model_name = "fast [type=missing, input_value={}, input_type=dict])"
    model_registry = ModelProfileRegistry()
    model_registry.register(ModelProfile(id="model-markup", name=model_name, provider="mock", model_id="mock"))

    class _BasicPanelApp(App):
        def compose(self) -> ComposeResult:
            yield BasicConfigPanel(
                profile,
                model_registry=model_registry,
                active_profile_id="model-markup",
            )

    async with _BasicPanelApp().run_test(size=(80, 40)) as pilot:
        checkbox = pilot.app.query_one("#bc-model-use-active", Checkbox)
        assert checkbox.label.plain == f"Use active model profile (Current: {model_name})"

        checkbox.value = False
        await pilot.pause()
        checkbox.value = True
        await pilot.pause()

        assert checkbox.label.plain == f"Use active model profile (Current: {model_name})"


@pytest.mark.asyncio
async def test_tools_descriptions_wrap_under_checkbox_label() -> None:
    class _ToolsPanelApp(App):
        def compose(self) -> ComposeResult:
            yield ToolsConfigPanel()

    app = _ToolsPanelApp()
    async with app.run_test(size=(38, 40)) as pilot:
        await pilot.pause()

        descriptions = list(app.query(".tc-cat-desc").results(Label))
        desc = next(label for label in descriptions if label.render().plain.startswith("Convert PDF"))
        await _wait_for_condition(
            lambda: desc.size.height > 1,
            pilot,
            description="tool category description wrap",
        )

        assert desc.render().plain == "Convert PDF, DOCX, PPTX, XLSX, XLS to Markdown"


@pytest.mark.asyncio
async def test_tools_panel_todo_checkbox_round_trip() -> None:
    """The todo category surfaces as #tc-cat-todo and round-trips get_config()."""

    class _ToolsPanelApp(App):
        def compose(self) -> ComposeResult:
            yield ToolsConfigPanel(ToolsConfig(builtins=["todo"]))

    app = _ToolsPanelApp()
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()

        checkbox = app.query_one("#tc-cat-todo", Checkbox)
        assert checkbox.value is True
        assert checkbox.disabled is False
        assert app.query_one(ToolsConfigPanel).get_config() == ["todo"]

        checkbox.value = False
        await pilot.pause()
        assert "todo" not in app.query_one(ToolsConfigPanel).get_config()


@pytest.mark.asyncio
async def test_tools_panel_todo_checkbox_defaults_off_when_profile_omits_category() -> None:
    class _ToolsPanelApp(App):
        def compose(self) -> ComposeResult:
            yield ToolsConfigPanel(ToolsConfig(builtins=["shell"]))

    app = _ToolsPanelApp()
    async with app.run_test(size=(80, 40)) as pilot:
        await pilot.pause()

        checkbox = app.query_one("#tc-cat-todo", Checkbox)
        assert checkbox.value is False
        assert app.query_one(ToolsConfigPanel).get_config() == ["shell"]


def test_basic_get_config_uses_mounted_model_select_when_checkbox_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = copy.deepcopy(_registry().get("Code"))
    assert profile is not None
    panel = BasicConfigPanel(profile)

    def query_one(selector: str, *_args: object, **_kwargs: object) -> object:
        if selector == "#bc-model-profile":
            return SimpleNamespace(value="live-model-profile")
        raise NoMatches(f"No nodes match {selector!r}")

    monkeypatch.setattr(panel, "query_one", query_one)

    cfg = panel.get_config()

    assert cfg["model_profile_id"] == "live-model-profile"


def test_subagent_card_get_config_preserves_seed_before_children_mount() -> None:
    from rich.text import Text

    from chrys.app.tui.screens.agents.panels.subagents import SubAgentCard

    card = SubAgentCard(
        profile_name="Explore",
        tool_name="explore_custom",
        tool_description="Explore things",
        max_concurrency=3,
        index=0,
        available_profiles=[(Text("Explore Agent"), "Explore")],
    )

    cfg = card.get_config()

    assert cfg == {
        "profile": "Explore",
        "tool_name": "explore_custom",
        "tool_description": "Explore things",
        "max_concurrency": 3,
    }


def test_subagent_card_get_config_allows_cleared_mounted_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rich.text import Text

    from chrys.app.tui.screens.agents.panels.subagents import SubAgentCard

    card = SubAgentCard(
        profile_name="Explore",
        tool_name="explore_custom",
        tool_description="Explore things",
        max_concurrency=3,
        index=0,
        available_profiles=[(Text("Explore Agent"), "Explore")],
    )
    card._profile_user_modified = True

    def query_one(selector: str, *_args: object, **_kwargs: object) -> object:
        if selector == "#sa-profile-0":
            return SimpleNamespace(value=Select.NULL)
        raise NoMatches(f"No nodes match {selector!r}")

    monkeypatch.setattr(card, "query_one", query_one)

    cfg = card.get_config()

    assert cfg["profile"] == ""
    assert cfg["tool_name"] == "explore_custom"


def test_subagent_card_get_config_preserves_seed_for_unmodified_mounting_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rich.text import Text

    from chrys.app.tui.screens.agents.panels.subagents import SubAgentCard

    card = SubAgentCard(
        profile_name="Explore",
        tool_name="explore_custom",
        tool_description="Explore things",
        max_concurrency=3,
        index=0,
        available_profiles=[(Text("Explore Agent"), "Explore")],
    )

    def query_one(selector: str, *_args: object, **_kwargs: object) -> object:
        if selector == "#sa-profile-0":
            return SimpleNamespace(value=Select.NULL)
        raise NoMatches(f"No nodes match {selector!r}")

    monkeypatch.setattr(card, "query_one", query_one)

    cfg = card.get_config()

    assert cfg["profile"] == "Explore"
    assert cfg["tool_name"] == "explore_custom"


def test_memory_get_config_preserves_seed_before_children_mount() -> None:
    from chrys.app.tui.screens.agents.panels.memory import MemoryConfigPanel, MemoryFileCard
    from chrys.service.profiles.agents.schema import MemoryConfig

    panel = MemoryConfigPanel(MemoryConfig(files=["docs/a.md"], folders=["notes"]))
    card = MemoryFileCard("docs/a.md", index=0)

    cfg = panel.get_config()

    assert cfg.files == ["docs/a.md"]
    assert cfg.folders == ["notes"]
    assert card.get_path() == "docs/a.md"


def test_agent_config_missing_child_widgets_are_treated_as_mount_pending(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mounted panel can be queryable before its compose children are."""
    registry = _registry()
    profile = copy.deepcopy(registry.get("Code"))
    assert profile is not None
    screen = AgentsConfigScreen(registry, current_profile="Code")
    draft = config_mod._AgentDraft(
        key="profile:Code",
        original_name="Code",
        profile=profile,
        original_profile=copy.deepcopy(profile),
        is_builtin=True,
    )
    screen._mounted_tabs.add("basic")

    def raise_no_matches(*_args: object, **_kwargs: object) -> object:
        raise NoMatches("No nodes match '#bc-model-use-active'")

    monkeypatch.setattr(screen, "query_one", raise_no_matches)

    with (
        caplog.at_level(logging.DEBUG, logger="chrys.app.tui.screens.agents.config"),
        pytest.raises(config_mod._AgentConfigPanelsNotReady),
    ):
        screen._build_profile_from_mounted_panels(draft)

    assert "Failed to read" not in caplog.text


@pytest.mark.asyncio
async def test_agent_config_empty_initial_profile_loads_first_visible_profile() -> None:
    """Startup failures leave the main screen with no active profile label yet."""
    registry = _registry()
    first = registry.list_profiles(include_sub_agent_only=True)[0]

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="", initial_profile="")
        await app.push_screen(screen)
        await pilot.pause()

        assert screen._selected_profile_name == first.name
        assert len(list(screen.query(BasicConfigPanel))) == 1


@pytest.mark.asyncio
async def test_agent_config_stale_initial_profile_loads_first_visible_profile() -> None:
    registry = _registry()
    first = registry.list_profiles(include_sub_agent_only=True)[0]

    app = _AgentConfigApp()
    async with app.run_test(size=(120, 40)) as pilot:
        screen = AgentsConfigScreen(registry, current_profile="Deleted", initial_profile="Deleted")
        await app.push_screen(screen)
        await pilot.pause()

        assert screen._selected_profile_name == first.name
        assert len(list(screen.query(BasicConfigPanel))) == 1


def test_agent_load_failed_preserves_failed_profile_label() -> None:
    import asyncio

    from chrys.foundation.events.types import AgentLoadFailed
    from chrys.foundation.i18n import MessageRef
    from chrys.foundation.i18n.formatting import format_message

    loading: list[bool] = []
    flashes: list[str] = []
    subtitles: list[str] = []

    class _FakeStatusBar:
        def flash(self, message: MessageRef | str, **_kwargs: object) -> None:
            flashes.append(message if isinstance(message, str) else format_message(message))

    def _query_one(cls):
        if cls.__name__ == "StatusBar":
            return _FakeStatusBar()
        raise AssertionError(f"unexpected query_one({cls.__name__})")

    screen = SimpleNamespace(
        _profile="",
        _set_agent_loading=loading.append,
        query_one=_query_one,
        _debug=lambda *_args: None,
        _update_subtitle=lambda: subtitles.append("updated"),
    )
    handler = make_backend_handler(screen)
    handler._agent_load_dialog = None

    asyncio.run(
        handler.on_agent_load_failed(
            AgentLoadFailed(agent_profile="Code", display_name="Code", message="missing api key")
        )
    )

    assert screen._profile == "Code"
    assert subtitles == ["updated"]
    assert loading == [False]
    assert flashes == ["Agent load failed: missing api key"]
