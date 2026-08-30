# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the Textual app shell."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import re
import sys
from logging.handlers import QueueHandler
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest
from rich.color import EIGHT_BIT_PALETTE, Color, ColorSystem
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Input, Static, Tab, TabbedContent, TextArea

from chrys.app.tui import app as chrys_app
from chrys.app.tui import i18n as tui_i18n
from chrys.app.tui.app import ChrysApp
from chrys.app.tui.i18n import LocaleSwitchStatus
from chrys.app.tui.notifications import NotificationService
from chrys.app.tui.theme import (
    CHRYS_ANSI_THEME,
    CHRYS_DARK_THEME,
    CHRYS_DARK_THEME_NAME,
    CHRYS_THEME,
    TuiVariableDefaultsMixin,
)
from chrys.app.tui.util import debug_log
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.foundation.config.settings import DEFAULT_THEME, Settings
from chrys.foundation.config.settings_store import LoadedSettings, SettingsHandle
from chrys.foundation.config.spec import Source
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import SettingsReload, Warning
from chrys.foundation.i18n import Localizer, MessageRef
from chrys.foundation.i18n.formatting import format_message
from chrys.foundation.util.session_ids import session_short_id
from chrys.orchestration.startup import RuntimeBootstrap
from chrys.service.state.store import JsonFileStateStore
from tests.support.paths import SRC_ROOT
from tests.support.waiting import wait_for, wait_until_quiet

_CHRYS_CSS = SRC_ROOT / "chrys" / "app" / "tui" / "chrys.tcss"
_CHRYS_THEME = SRC_ROOT / "chrys" / "app" / "tui" / "theme.py"


@pytest.mark.parametrize(
    ("theme", "expected_opacity"),
    [
        ("chrys", 0.8),
        ("chrys-dark", 0.8),
        ("chrys-ansi", 0.8),
        ("textual-dark", 0.8),
        ("ansi-dark", 1.0),
        ("ansi-light", 1.0),
    ],
)
@pytest.mark.asyncio
async def test_default_border_opacity_respects_theme_color_mode(
    theme: str,
    expected_opacity: float,
) -> None:
    """Native ANSI tokens remain opaque; RGB-backed themes honor 80% alpha."""

    class BorderApp(TuiVariableDefaultsMixin, App):
        CSS = "Static { border: round $primary $border-opacity; }"

        def __init__(self) -> None:
            super().__init__()
            self.register_theme(CHRYS_THEME)
            self.register_theme(CHRYS_DARK_THEME)
            self.register_theme(CHRYS_ANSI_THEME)
            self.theme = theme

        def compose(self) -> ComposeResult:
            yield Static("border", id="border")

    app = BorderApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        border_color = app.query_one("#border", Static).styles.border_top[1]
        assert border_color.a == expected_opacity
        variables = app.get_css_variables()
        expected_tool_group_title = "#949494" if theme == CHRYS_DARK_THEME_NAME else variables["warning"]
        assert variables["tui-tool-group-title"] == expected_tool_group_title


@pytest.mark.asyncio
async def test_chrys_dark_uses_gray_borders_without_muting_semantic_colors() -> None:
    class BorderApp(TuiVariableDefaultsMixin, App):
        CSS = """
        #primary { border: round $tui-border-primary $border-opacity; }
        #accent { border: round $tui-border-accent $border-opacity; }
        #warning { border: round $tui-border-warning $border-opacity; }
        #literal { border: round $tui-border-neutral-160 $border-opacity; }
        #user-message { border: round $tui-border-user-message $border-opacity; }
        #agent-message { border: round $tui-border-agent-message $border-opacity; }
        #titled {
            border: round $tui-border-primary $border-opacity;
            border-title-color: $tui-border-title-primary;
            border-subtitle-color: $tui-border-title-primary;
        }
        """

        def __init__(self) -> None:
            super().__init__()
            self.register_theme(CHRYS_DARK_THEME)
            self.theme = CHRYS_DARK_THEME_NAME

        def compose(self) -> ComposeResult:
            yield Static("primary", id="primary")
            yield Static("accent", id="accent")
            yield Static("warning", id="warning")
            yield Static("literal", id="literal")
            yield Static("user-message", id="user-message")
            yield Static("agent-message", id="agent-message")
            titled = Static("titled", id="titled")
            titled.border_title = "Title"
            titled.border_subtitle = "Subtitle"
            yield titled

    app = BorderApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        for widget_id in ("primary", "accent", "warning", "literal"):
            border_color = app.query_one(f"#{widget_id}", Static).styles.border_top[1]
            assert border_color.hex6 == "#5F5F5F"
            assert border_color.a == 0.8
        for widget_id, expected_color in (("user-message", "#FF87D7"), ("agent-message", "#5FFF87")):
            border_color = app.query_one(f"#{widget_id}", Static).styles.border_top[1]
            assert border_color.hex6 == expected_color
            assert border_color.a == 0.8
        title_color = app.query_one("#titled", Static).styles.border_title_color
        assert title_color.hex6 == "#AF87FF"
        assert title_color.a == 1.0
        subtitle_color = app.query_one("#titled", Static).styles.border_subtitle_color
        assert subtitle_color.hex6 == "#AF87FF"
        assert subtitle_color.a == 1.0
        variables = app.get_css_variables()
        assert variables["scrollbar"] == "#585858"
        assert variables["scrollbar-hover"] == "#767676"
        assert variables["scrollbar-active"] == "#808080"


def test_all_tui_border_declarations_use_theme_aware_colors() -> None:
    border_declaration = re.compile(
        r"^\s*border(?:-(?:top|right|bottom|left))?\s*:\s*(?P<value>[^;]+);",
        re.MULTILINE,
    )
    failures: list[str] = []
    tui_root = SRC_ROOT / "chrys" / "app" / "tui"
    for path in (*tui_root.rglob("*.py"), *tui_root.rglob("*.tcss")):
        source = path.read_text(encoding="utf-8")
        for match in border_declaration.finditer(source):
            value = match.group("value")
            if value != "none" and "$tui-border-" not in value:
                line = source.count("\n", 0, match.start()) + 1
                failures.append(f"{path.relative_to(tui_root)}:{line} {value}")

    assert failures == []


def test_tui_module_main_is_callable() -> None:
    """The hidden serve subprocess dispatch relies on the TUI module entrypoint."""
    from chrys.app.tui import app as tui_app

    assert callable(tui_app.main)


def test_screen_css_private_fork_version_matches_pinned_textual() -> None:
    """A Textual upgrade must explicitly re-audit the private CSS loader fork."""
    from textual import __version__ as textual_version

    from chrys.app.tui import app as tui_app

    assert textual_version == tui_app._TEXTUAL_LOAD_SCREEN_CSS_FORK_VERSION


def test_chrys_app_defaults_profile_from_settings(tmp_path: Path) -> None:
    class _Engine:
        async def shutdown(self) -> None:
            return

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(default_agent="QA"),
        state_store=JsonFileStateStore(tmp_path),
    )

    assert app._profile_name == "QA"


def test_chrys_app_explicit_profile_overrides_settings_default(tmp_path: Path) -> None:
    class _Engine:
        async def shutdown(self) -> None:
            return

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(default_agent="QA"),
        profile_name="Code",
        state_store=JsonFileStateStore(tmp_path),
    )

    assert app._profile_name == "Code"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested_locale", "expected_message"),
    [
        ("zh-Hans", "Could not load translations for locale zh-Hans; using English."),
        ("system", "Could not load translations for locale system; using English."),
    ],
)
async def test_chrys_app_locale_catalog_failure_follows_startup_warning_order(
    requested_locale: str,
    expected_message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Engine:
        async def shutdown(self) -> None:
            return

    class _TestApp(ChrysApp):
        CSS_PATH = None

        def _build_main_screen(self) -> Screen:
            return Screen()

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    monkeypatch.setenv("CHRYS_LOCALE", requested_locale)
    if requested_locale == "system":
        monkeypatch.setattr("chrys.foundation.i18n.localizer.normalize_locale", lambda _requested: "zh-Hans")

    catalog_root = tmp_path / "catalogs"
    catalog_path = catalog_root / "zh-Hans" / "LC_MESSAGES" / "chrys.mo"
    catalog_path.parent.mkdir(parents=True)
    catalog_path.write_bytes(b"not a catalog")
    localizer = Localizer(requested_locale, catalog_root=catalog_root)
    injected_warning = Warning(code="bootstrap_warning", message="Injected startup warning.")
    bus = EventBus()
    published: list[Warning] = []

    async def _record_warning(event: Warning) -> None:
        published.append(event)

    await bus.subscribe(Warning, _record_warning)
    app = _TestApp(
        bus,
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(locale=requested_locale),
        state_store=JsonFileStateStore(tmp_path / "state"),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        startup_warnings=[injected_warning],
        localizer=localizer,
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        await pilot.pause()

    assert [(warning.code, warning.message) for warning in published] == [
        ("bootstrap_warning", "Injected startup warning."),
        ("i18n_catalog_load_failed", expected_message),
    ]


@pytest.mark.asyncio
async def test_chrys_app_locale_uses_settings_and_healthy_catalog_emits_no_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from chrys.orchestration import session_host as session_host_mod

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _TestApp(ChrysApp):
        CSS_PATH = None

        def _build_main_screen(self) -> Screen:
            return Screen()

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    monkeypatch.setenv("CHRYS_LOCALE", "zh-Hans")
    bus = EventBus()
    published: list[Warning] = []

    async def _record_warning(event: Warning) -> None:
        published.append(event)

    await bus.subscribe(Warning, _record_warning)
    app = _TestApp(
        bus,
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(locale="zh-Hans"),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    assert (
        app.localizer.render(session_host_mod._SESSION_NOT_FOUND.bind(session_id="missing")) == "未找到会话：missing"  # noqa: RUF001
    )

    async with app.run_test() as pilot:
        await pilot.pause()

    assert [warning for warning in published if warning.code == "i18n_catalog_load_failed"] == []


def test_chrys_app_controller_owns_localizer_and_preserves_transcript_seam(tmp_path: Path) -> None:
    class _Engine:
        async def shutdown(self) -> None:
            return

    settings = Settings(locale="en")
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=settings,
        state_store=JsonFileStateStore(tmp_path),
    )

    screen = app._build_main_screen()

    assert app.locale_controller.localizer is app.localizer
    assert app.locale_controller.requested_locale == settings.locale
    assert screen._locale_controller is app.locale_controller
    assert screen._localization is app.localizer


@pytest.mark.asyncio
async def test_locale_switch_avoids_backend_global_ui_transcript_and_gc_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.app.tui.screens.main.screen import MainScreen
    from chrys.app.tui.widgets.chat.panel import ChatPanel

    class _Engine:
        def __init__(self) -> None:
            self.rebuild_calls = 0

        async def shutdown(self) -> None:
            return

        async def rebuild(self) -> None:
            self.rebuild_calls += 1

    engine = _Engine()
    bus = EventBus()
    settings_reloads: list[SettingsReload] = []

    async def record_settings_reload(event: SettingsReload) -> None:
        settings_reloads.append(event)

    await bus.subscribe(SettingsReload, record_settings_reload)
    app = ChrysApp(
        bus,
        engine,  # type: ignore[arg-type]
        settings=Settings(locale="en"),
        state_store=JsonFileStateStore(tmp_path),
        gc_freeze_enabled=False,
    )
    main_screen = app._build_main_screen()
    app._main_screen = main_screen
    forbidden_calls: list[str] = []
    persisted: list[str] = []

    def forbidden(name: str):
        def record(*_args: object, **_kwargs: object) -> None:
            forbidden_calls.append(name)

        return record

    monkeypatch.setattr(tui_i18n, "persist_locale", persisted.append)
    monkeypatch.setattr(app._gc_freeze, "request_absorb", forbidden("gc_absorb"))
    monkeypatch.setattr(app._gc_freeze, "request_reclaim", forbidden("gc_reclaim"))
    monkeypatch.setattr(type(app.stylesheet), "update", forbidden("stylesheet_update"))
    monkeypatch.setattr(MainScreen, "recompose", forbidden("main_recompose"))
    monkeypatch.setattr(MainScreen, "walk_children", forbidden("main_transcript_walk"))
    monkeypatch.setattr(ChatPanel, "walk_children", forbidden("chat_transcript_walk"))
    monkeypatch.setattr(app, "run_worker", forbidden("revision_worker"))

    results = [app.locale_controller.switch_locale("zh-Hans") for _ in range(4)]
    await asyncio.sleep(0)

    assert [result.status for result in results] == [
        LocaleSwitchStatus.EFFECTIVE_CHANGED,
        LocaleSwitchStatus.IDENTICAL_REQUEST,
        LocaleSwitchStatus.IDENTICAL_REQUEST,
        LocaleSwitchStatus.IDENTICAL_REQUEST,
    ]
    assert app.locale_controller.revision == 1
    assert persisted == ["zh-Hans"]
    assert settings_reloads == []
    assert engine.rebuild_calls == 0
    assert forbidden_calls == []


def test_build_main_screen_plumbs_workspace_mru_max_entries(tmp_path: Path) -> None:
    """The startup-only MRU setting must reach the screen services intact."""

    class _Engine:
        async def shutdown(self) -> None:
            return

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(workspace_mru_max_entries=7),
        state_store=JsonFileStateStore(tmp_path),
    )

    screen = app._build_main_screen()

    assert screen._services.workspace_mru_max_entries == 7


@pytest.mark.asyncio
async def test_overlay_close_preserves_keyboard_only_mouse_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Removing an overlay must not invent a pointer at the default position."""
    from textual.screen import Screen

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        underlay = app.screen
        assert app.mouse_over is None

        await app.push_screen(Screen())
        await pilot.pause()
        assert app.mouse_over is None

        hover_hit_tests: list[tuple[int, int]] = []
        get_hover_widgets_at = underlay.get_hover_widgets_at

        def record_hover_hit_test(x: int, y: int):
            hover_hit_tests.append((x, y))
            return get_hover_widgets_at(x, y)

        monkeypatch.setattr(underlay, "get_hover_widgets_at", record_hover_hit_test)

        await app.pop_screen()
        await pilot.pause()

        assert app.screen is underlay
        assert app.mouse_over is None
        assert hover_hit_tests == []


def test_chrys_app_accepts_startup_session_id(tmp_path: Path) -> None:
    class _Engine:
        async def shutdown(self) -> None:
            return

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        state_store=JsonFileStateStore(tmp_path),
        startup_session_id=" session-1 ",
    )

    assert app._startup_session_id == "session-1"


def test_chrys_app_defers_settings_warnings_only_for_a_startup_restore(tmp_path: Path) -> None:
    """Bootstrap settings warnings describe the launch cwd, not a restored session's root."""

    class _Engine:
        async def shutdown(self) -> None:
            return

    bootstrap_warning = Warning(code="bootstrap_warning", message="Env grumbled.")
    settings_warning = Warning(code="settings_layer_warning", message="Project layer grumbled.")

    restoring = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        state_store=JsonFileStateStore(tmp_path / "restoring"),
        startup_warnings=[bootstrap_warning],
        settings_warnings=[settings_warning],
        startup_session_id="session-1",
    )
    fresh = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        state_store=JsonFileStateStore(tmp_path / "fresh"),
        startup_warnings=[bootstrap_warning],
        settings_warnings=[settings_warning],
    )

    assert restoring._deferred_settings_warnings == [settings_warning]
    assert settings_warning not in restoring._startup_warnings
    assert fresh._deferred_settings_warnings == []
    assert fresh._startup_warnings[:2] == [bootstrap_warning, settings_warning]


def _app_with_loaded_settings(
    tmp_path: Path,
    loaded: LoadedSettings,
    handle: SettingsHandle | None = None,
) -> ChrysApp:
    class _Engine:
        async def shutdown(self) -> None:
            return

    return ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings_handle=handle or SettingsHandle(loaded),
        state_store=JsonFileStateStore(tmp_path),
    )


def test_a_live_write_reaches_every_holder_of_the_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """What the shared handle buys over two holders that started equal.

    Freezing ``Settings`` removed the accident that used to carry a live write
    across — the app and the engine held the same instance, so mutating a field
    changed both — and rebinding a private field would have traded a provenance
    bug for a divergence bug. Sharing the cell is what makes the write arrive
    everywhere without anyone having to push it.
    """
    monkeypatch.setattr(chrys_app, "persist_theme", lambda _theme: None)
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    handle = SettingsHandle(LoadedSettings(settings=Settings(theme=DEFAULT_THEME, locale="en"), provenance={}))
    app = _app_with_loaded_settings(tmp_path, handle.loaded, handle=handle)

    app.theme = "chrys-dark"
    app.locale_controller.switch_locale("zh-Hans")

    # Read through the handle, which is what the engine holds too.
    assert handle.settings.theme == "chrys-dark"
    assert handle.settings.locale == "zh-Hans"
    # One object, not two that agree: provenance and seals travel with it.
    assert handle.loaded is app._loaded_settings
    assert handle.settings is app._settings


def test_switching_the_theme_moves_its_provenance_with_the_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The app shares its ``Settings`` with the engine, so a live write must
    go through the overlay or the engine keeps reporting the old layer."""
    monkeypatch.setattr(chrys_app, "persist_theme", lambda _theme: None)
    loaded = LoadedSettings(settings=Settings(theme=DEFAULT_THEME), provenance={})
    app = _app_with_loaded_settings(tmp_path, loaded)

    app.theme = "chrys-dark"

    assert app._settings.theme == "chrys-dark"
    assert app._loaded_settings.settings is app._settings
    assert app._loaded_settings.source_for("ui.theme").layer is Source.RUNTIME


def test_switching_the_locale_moves_its_provenance_with_the_value(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    loaded = LoadedSettings(settings=Settings(locale="en"), provenance={})
    app = _app_with_loaded_settings(tmp_path, loaded)

    app.locale_controller.switch_locale("zh-Hans")

    assert app._settings.locale == "zh-Hans"
    assert app.locale_controller.requested_locale == "zh-Hans"
    assert app._loaded_settings.settings is app._settings
    assert app._loaded_settings.source_for("ui.locale").layer is Source.RUNTIME


def test_chrys_app_rejects_two_different_settings_objects(tmp_path: Path) -> None:
    class _Engine:
        async def shutdown(self) -> None:
            return

    with pytest.raises(ValueError, match="not two different ones"):
        ChrysApp(
            EventBus(),
            _Engine(),  # type: ignore[arg-type]
            settings=Settings(theme="chrys-dark"),
            settings_handle=SettingsHandle(LoadedSettings(settings=Settings(theme=DEFAULT_THEME), provenance={})),
            state_store=JsonFileStateStore(tmp_path),
        )


def test_chrys_app_gc_freeze_argument_overrides_the_module_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Engine:
        async def shutdown(self) -> None:
            return

    monkeypatch.setattr(chrys_app, "GC_FREEZE_ENABLED", False)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        gc_freeze_enabled=True,
    )

    assert app._gc_freeze.enabled is True
    assert app._gc_freeze.started is False


def test_chrys_app_follows_the_module_default_when_no_argument_is_given(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """No setting reaches this: production is on, and only code turns it off."""

    class _Engine:
        async def shutdown(self) -> None:
            return

    monkeypatch.setattr(chrys_app, "GC_FREEZE_ENABLED", True)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
    )

    assert app._gc_freeze.enabled is True
    assert app._startup_warnings == []


@pytest.mark.asyncio
async def test_native_app_focus_preserves_internal_focus_without_layout_or_binding_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Native focus transitions must stay independent of mounted transcript size."""
    from textual import events

    from chrys.app.tui.widgets.chrome.input_bar import InputBar

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        await app.screen.mount(*(Static("") for _ in range(64)))
        app.screen.query_one(InputBar).focus_input()
        await pilot.pause()
        focused = app.screen.focused
        assert focused is not None
        assert len(app.screen.walk_children()) >= 64

        full_restyles: list[bool] = []
        binding_refreshes: list[None] = []
        layout_refreshes: list[None] = []
        monkeypatch.setattr(
            app.screen,
            "update_node_styles",
            lambda animate=True: full_restyles.append(animate),
        )
        monkeypatch.setattr(app.screen, "refresh_bindings", lambda: binding_refreshes.append(None))
        monkeypatch.setattr(app.screen, "_refresh_layout", lambda *_args, **_kwargs: layout_refreshes.append(None))

        # Deferred work from the 64-widget mount can land arbitrarily late on
        # loaded CI workers; drain it so anything recorded below is caused by
        # the focus events under test.
        await wait_until_quiet(
            lambda: (len(full_restyles), len(binding_refreshes), len(layout_refreshes)),
            description="initial focus refresh counters",
            pilot=pilot,
        )
        full_restyles.clear()
        binding_refreshes.clear()
        layout_refreshes.clear()

        await app.on_event(events.AppBlur())
        await pilot.pause()
        assert app.app_focus is False
        assert app.screen.focused is focused
        assert focused.has_focus is True
        assert app._last_focused_on_app_blur is None

        await app.on_event(events.AppFocus())
        await pilot.pause()
        assert app.app_focus is True
        assert app.screen.focused is focused
        assert focused.has_focus is True
        assert app._last_focused_on_app_blur is None
        assert full_restyles == []
        assert binding_refreshes == []
        assert layout_refreshes == []


@pytest.mark.asyncio
async def test_native_app_focus_recovers_deferred_diff_footer_actions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A DiffScreen completed while unfocused must refresh its stock Footer."""
    import asyncio

    from textual import events

    from chrys.app.tui.screens.diff.screen import DiffScreen
    from chrys.app.tui.util.diff_entries import DiffFileEntry, DiffLoadResult
    from chrys.service.mutations.types import MutationOp

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    load_started = asyncio.Event()
    release_load = asyncio.Event()
    entry = DiffFileEntry(
        path=str(tmp_path / "async.py"),
        rel_path="async.py",
        operation=MutationOp.MODIFY,
        old_path=None,
        before_text="before\n",
        after_text="after\n",
        is_binary=False,
        encoding="utf-8",
        bytes_changed=True,
    )

    async def load_data() -> DiffLoadResult:
        load_started.set()
        await release_load.wait()
        return DiffLoadResult(all_entries=[entry], per_turn_entries={1: [entry]}, total_turns=1)

    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        screen = DiffScreen({}, cwd=str(tmp_path), load_data=load_data)
        await app.push_screen(screen)
        # Generous deadline: a loaded CI worker can take >1s just to run
        # the deferred load worker; a short cap here only converts
        # scheduler contention into flakes (no cost on the happy path).
        await asyncio.wait_for(load_started.wait(), timeout=10)
        await app.on_event(events.AppBlur())
        release_load.set()

        await wait_for(
            lambda: screen._content_ready,
            timeout=10.0,
            pilot=pilot,
            description="diff screen content ready",
        )
        await wait_for(
            lambda: "go_back" in {key.action for key in screen.query("FooterKey")},
            pilot=pilot,
            description="blurred diff footer Back binding",
        )
        footer_actions = {key.action for key in screen.query("FooterKey")}
        assert "go_back" in footer_actions
        assert "toggle_view" not in footer_actions
        assert "toggle_change_list" not in footer_actions

        await app.on_event(events.AppFocus())
        await wait_for(
            lambda: (
                {"go_back", "toggle_view", "toggle_change_list"} <= {key.action for key in screen.query("FooterKey")}
            ),
            pilot=pilot,
            description="refocused diff footer full bindings",
        )


@pytest.mark.asyncio
async def test_web_app_focus_retains_textual_full_style_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The web host may visibly depend on Textual's App:focus/App:blur CSS."""
    from textual import events

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    monkeypatch.setenv("TEXTUAL_DRIVER", "textual.drivers.web_driver:WebDriver")
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        full_restyles: list[bool] = []
        monkeypatch.setattr(
            app.screen,
            "update_node_styles",
            lambda animate=True: full_restyles.append(animate),
        )

        await app.on_event(events.AppBlur())
        await app.on_event(events.AppFocus())

        assert full_restyles == [True, True]


@pytest.mark.asyncio
async def test_chrys_footer_composes_bindings_before_deferred_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Startup shortcuts render even if the first deferred refresh is stranded."""
    from chrys.app.tui.widgets.chrome.footer import ChrysFooter

    class _Engine:
        session_generation = 0

        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    monkeypatch.setattr(ChrysFooter, "call_after_refresh", lambda _self, *_args: False)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test():
        assert app._main_screen is not None
        footer = app._main_screen.query_one(ChrysFooter)
        actions = {key.action for key in footer.query("FooterKey")}

        assert {"sessions", "agents_config", "models_config", "show_log_viewer"} <= actions


@pytest.mark.asyncio
async def test_chrys_footer_recomposes_only_for_visible_binding_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Focus and no-op binding signals must not rebuild the MainScreen Footer."""
    from textual import events

    from chrys.app.tui.widgets.chrome.footer import ChrysFooter

    class _Engine:
        session_generation = 0

        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        assert app._main_screen is not None
        footer = app._main_screen.query_one(ChrysFooter)
        await pilot.pause()
        # Exercise the real async recompose before replacing it with a spy.
        # The callback may originate in ChrysApp context, but Footer.compose
        # must bind Footer.compact against the Footer itself.
        app._main_screen._set_agent_running(True)
        for _ in range(20):
            if (
                not footer._binding_recompose_in_progress
                and not footer._binding_recompose_dirty
                and footer._visible_binding_signature == footer._binding_signature(app._main_screen)
            ):
                break
            await pilot.pause()
        running_actions = {key.action for key in footer.query("FooterKey")}
        assert {"pick_theme", "settings"}.isdisjoint(running_actions)

        app._main_screen._set_agent_running(False)
        await pilot.pause()
        # The recompose scheduled by the flip back rides paint scheduling;
        # under load it can land after the spy is installed and pollute the
        # no-change window below. Wait until the footer has committed the
        # settled signature — any callback still queued at that point is
        # generation-stale and no-ops before reaching recompose.
        for _ in range(20):
            if (
                not footer._binding_recompose_in_progress
                and not footer._binding_recompose_dirty
                and footer._visible_binding_signature == footer._binding_signature(app._main_screen)
            ):
                break
            await pilot.pause()
        idle_actions = {key.action for key in footer.query("FooterKey")}
        assert {"pick_theme", "settings"} <= idle_actions
        recomposes: list[None] = []

        async def _record_recompose() -> None:
            recomposes.append(None)

        monkeypatch.setattr(footer, "recompose", _record_recompose)

        app._main_screen.refresh_bindings()
        app._main_screen.refresh_bindings()
        await pilot.pause()
        assert recomposes == []

        app._main_screen._set_agent_running(True)
        # The recompose callback rides screen-paint scheduling; under Windows
        # xdist load a single pause can return before the paint fires. Poll
        # the observable count instead of a fixed wait.
        for _ in range(20):
            await pilot.pause()
            if recomposes:
                break
        assert recomposes == [None]
        app._main_screen._set_agent_running(True)
        await pilot.pause()
        assert recomposes == [None]

        await app.on_event(events.AppBlur())
        app._main_screen._set_agent_running(False)
        await pilot.pause()
        assert recomposes == [None]

        await app.on_event(events.AppFocus())
        for _ in range(20):
            await pilot.pause()
            if len(recomposes) > 1:
                break
        assert recomposes == [None, None]


@pytest.mark.asyncio
async def test_chrys_footer_resume_recovers_stranded_recompose(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A callback lost with an overlay must not commit or suppress its signature."""
    from textual import events
    from textual.screen import Screen

    from chrys.app.tui.widgets.chrome.footer import ChrysFooter

    class _Engine:
        session_generation = 0

        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        assert app._main_screen is not None
        footer = app._main_screen.query_one(ChrysFooter)
        await pilot.pause()
        recomposes = 0

        async def _record_recompose() -> None:
            nonlocal recomposes
            recomposes += 1

        deferred: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

        def _strand_callback(callback: object, *args: object, **kwargs: object) -> bool:
            deferred.append((callback, args, kwargs))
            return True

        monkeypatch.setattr(footer, "recompose", _record_recompose)
        monkeypatch.setattr(footer, "call_after_refresh", _strand_callback)
        rendered_signature = footer._visible_binding_signature
        app._main_screen._set_agent_running(True)
        await pilot.pause()

        assert deferred
        assert footer._visible_binding_signature == rendered_signature
        stranded = tuple(deferred)
        app._main_screen.on_screen_resume(events.ScreenResume())
        assert len(deferred) > len(stranded)

        callback, args, kwargs = deferred[-1]
        await callback(*args, **kwargs)  # type: ignore[operator]
        assert footer._visible_binding_signature != rendered_signature
        assert recomposes == 1

        for callback, args, kwargs in stranded:
            await callback(*args, **kwargs)  # type: ignore[operator]
        assert recomposes == 1

        deferred.clear()
        current_signature = ["A"]
        footer._visible_binding_signature = "A"  # type: ignore[assignment]
        monkeypatch.setattr(footer, "_binding_signature", lambda _screen: current_signature[0])

        current_signature[0] = "B"
        footer.bindings_changed(app._main_screen)
        stale_b_callback = deferred[-1]
        current_signature[0] = "A"
        footer.bindings_changed(app._main_screen)

        callback, args, kwargs = stale_b_callback
        await callback(*args, **kwargs)  # type: ignore[operator]
        assert footer._visible_binding_signature == "A"
        assert recomposes == 1

        current_signature[0] = "B"
        footer.bindings_changed(app._main_screen)
        callback, args, kwargs = deferred[-1]
        await callback(*args, **kwargs)  # type: ignore[operator]
        assert footer._visible_binding_signature == "B"
        assert recomposes == 2

        await app.push_screen(Screen())
        current_signature[0] = "A"
        footer.bindings_changed(app._main_screen)
        callback, args, kwargs = deferred[-1]
        await callback(*args, **kwargs)  # type: ignore[operator]
        assert app.screen is not app._main_screen
        assert footer._visible_binding_signature == "B"
        assert recomposes == 2
        await app.pop_screen()
        await pilot.pause()
        callback, args, kwargs = deferred[-1]
        await callback(*args, **kwargs)  # type: ignore[operator]
        assert footer._visible_binding_signature == "A"
        assert recomposes == 3


@pytest.mark.asyncio
async def test_chrys_footer_retries_binding_publish_during_recompose(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A publish during recompose must reconcile rendered and committed state."""
    from chrys.app.tui.widgets.chrome.footer import ChrysFooter

    class _Engine:
        session_generation = 0

        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        assert app._main_screen is not None
        footer = app._main_screen.query_one(ChrysFooter)
        await pilot.pause()
        current_signature = ["A"]
        rendered: list[str] = []
        first_recompose_started = asyncio.Event()
        release_first_recompose = asyncio.Event()
        deferred: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

        monkeypatch.setattr(footer, "_binding_signature", lambda _screen: current_signature[0])

        async def _record_recompose() -> None:
            rendered.append(current_signature[0])
            if len(rendered) == 1:
                first_recompose_started.set()
                await release_first_recompose.wait()

        def _defer(callback: object, *args: object, **kwargs: object) -> bool:
            deferred.append((callback, args, kwargs))
            return True

        monkeypatch.setattr(footer, "recompose", _record_recompose)
        monkeypatch.setattr(footer, "call_after_refresh", _defer)
        footer._visible_binding_signature = "A"  # type: ignore[assignment]
        current_signature[0] = "B"
        footer.bindings_changed(app._main_screen)
        callback, args, kwargs = deferred.pop()
        first_pass = asyncio.create_task(callback(*args, **kwargs))  # type: ignore[operator]
        await first_recompose_started.wait()

        current_signature[0] = "A"
        footer.bindings_changed(app._main_screen)
        release_first_recompose.set()
        await first_pass

        assert rendered == ["B"]
        assert footer._binding_recompose_dirty is True
        callback, args, kwargs = deferred.pop()
        await callback(*args, **kwargs)  # type: ignore[operator]
        assert rendered == ["B", "A"]
        assert footer._visible_binding_signature == "A"
        assert footer._binding_recompose_dirty is False


@pytest.mark.asyncio
async def test_chrys_footer_localizes_in_place_with_natural_per_locale_widths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A locale switch rewrites FooterKey text in place and re-measures widths."""
    from textual.widgets._footer import FooterKey

    from chrys.app.tui.widgets.chat.panel import ChatPanel
    from chrys.app.tui.widgets.chrome.footer import ChrysFooter

    class _Engine:
        session_generation = 0

        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(locale="en"),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        assert app._main_screen is not None
        footer = app._main_screen.query_one(ChrysFooter)
        chat = app._main_screen.query_one(ChatPanel)
        await pilot.pause()
        for _ in range(20):
            if (
                not footer._binding_recompose_in_progress
                and not footer._binding_recompose_dirty
                and footer._visible_binding_signature == footer._binding_signature(app._main_screen)
            ):
                break
            await pilot.pause()

        keys_before = tuple(footer.query(FooterKey))
        assert {key.action for key in keys_before} == {
            "sessions",
            "agents_config",
            "models_config",
            "show_log_viewer",
            "toggle_trajectory_dashboard",
            "toggle_sidebar",
            "pick_theme",
            "settings",
            "quit",
        }
        widths_before = {key.action: key.region.width for key in keys_before}
        english_descriptions = {key.action: key.description for key in keys_before}
        # Stock auto-width geometry: every key exactly fits its English text.
        assert all(key.content_region.width == key.render().cell_len for key in keys_before)
        committed_signature = footer._visible_binding_signature

        recomposes: list[None] = []
        layout_refreshes: list[None] = []
        chat_refreshes: list[bool] = []
        chat_restyles: list[None] = []

        async def _record_recompose() -> None:
            recomposes.append(None)

        def _record_chat_refresh(
            *_regions: object,
            repaint: bool = True,
            layout: bool = False,
            recompose: bool = False,
        ) -> ChatPanel:
            del repaint, recompose
            chat_refreshes.append(layout)
            return chat

        original_refresh_layout = app._main_screen._refresh_layout

        def _record_layout(*args: object, **kwargs: object) -> None:
            layout_refreshes.append(None)
            original_refresh_layout(*args, **kwargs)

        monkeypatch.setattr(footer, "recompose", _record_recompose)
        monkeypatch.setattr(app._main_screen, "_refresh_layout", _record_layout)
        monkeypatch.setattr(chat, "refresh", _record_chat_refresh)
        monkeypatch.setattr(chat, "update_node_styles", lambda animate=True: chat_restyles.append(None))
        await wait_until_quiet(
            lambda: (len(layout_refreshes), len(chat_refreshes), len(chat_restyles)),
            description="theme refresh counters",
            pilot=pilot,
        )
        layout_refreshes.clear()
        chat_refreshes.clear()
        chat_restyles.clear()

        result = app.locale_controller.switch_locale("zh-Hans")
        await pilot.pause()
        await pilot.pause()

        keys_after = tuple(footer.query(FooterKey))
        assert result.status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        assert len(keys_after) == len(keys_before)
        assert all(before is after for before, after in zip(keys_before, keys_after, strict=True))
        assert recomposes == []
        assert footer._visible_binding_signature is committed_signature
        assert {key.action: key.description for key in keys_after} == {
            "sessions": "会话",
            "agents_config": "智能体",
            "models_config": "模型",
            "show_log_viewer": "日志",
            "toggle_trajectory_dashboard": "轨迹",
            "toggle_sidebar": "侧边栏",
            "pick_theme": "主题",
            "settings": "设置",
            "quit": "退出",
        }
        # Natural per-locale geometry: each key re-measures to exactly fit its
        # zh-Hans text, so the shorter CJK labels shrink their keys.
        assert all(key.content_region.width == key.render().cell_len for key in keys_after)
        widths_after = {key.action: key.region.width for key in keys_after}
        assert widths_after["sessions"] < widths_before["sessions"]
        assert any(key.description != english_descriptions[key.action] for key in keys_after)
        assert layout_refreshes != []
        assert chat_refreshes == []
        assert chat_restyles == []


@pytest.mark.asyncio
async def test_chrys_footer_locale_change_after_compose_before_mount_is_reconciled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A stale compose-node batch is translated once after its async mount."""
    from collections.abc import Iterable

    from textual.widget import Widget
    from textual.widgets._footer import FooterKey

    from chrys.app.tui.widgets.chrome.footer import ChrysFooter

    class _Engine:
        session_generation = 0

        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(locale="en"),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        assert app._main_screen is not None
        footer = app._main_screen.query_one(ChrysFooter)
        await pilot.pause()
        compose_nodes_ready = asyncio.Event()
        release_mount = asyncio.Event()
        recompose_calls: list[None] = []
        original_recompose = footer.recompose
        original_mount_all = footer.mount_all

        async def _record_recompose() -> None:
            recompose_calls.append(None)
            await original_recompose()

        async def _pause_mount_all(
            widgets: Iterable[Widget],
            *,
            before: int | str | Widget | None = None,
            after: int | str | Widget | None = None,
        ) -> None:
            compose_nodes_ready.set()
            await release_mount.wait()
            await original_mount_all(widgets, before=before, after=after)

        monkeypatch.setattr(footer, "recompose", _record_recompose)
        monkeypatch.setattr(footer, "mount_all", _pause_mount_all)
        # An externally published binding change (a late AppFocus refocus
        # recovery, a resume replay) that lands inside the artificially held
        # mount window marks the in-flight compose dirty and forces a second,
        # by-design recompose pass. This test pins the locale reconciliation
        # path, which never routes through bindings_changed, so drop those
        # publishes to keep the recompose count deterministic. The attribute
        # patch alone cannot do that: Footer.on_mount subscribed the ORIGINAL
        # bound method to bindings_updated_signal, so signal-delivered
        # publishes bypass the patched attribute — drop the subscription too.
        stray_binding_publishes: list[None] = []
        monkeypatch.setattr(footer, "bindings_changed", lambda _screen: stray_binding_publishes.append(None))
        app._main_screen.bindings_updated_signal.unsubscribe(footer)
        footer._binding_recompose_generation += 1
        generation = footer._binding_recompose_generation
        task = asyncio.create_task(footer._recompose_bindings(app._main_screen, generation))
        await asyncio.wait_for(compose_nodes_ready.wait(), timeout=10)

        assert footer._rendered_locale_revision == 0
        assert app.locale_controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        assert footer._localization_dirty is True
        release_mount.set()
        await task
        await pilot.pause()

        assert recompose_calls == [None]
        assert footer._localization_dirty is False
        assert footer._rendered_locale_revision == app.locale_controller.revision == 1
        assert {key.description for key in footer.query(FooterKey)} >= {
            "会话",
            "智能体",
            "模型",
            "日志",
            "侧边栏",
            "主题",
            "设置",
            "退出",
        }


@pytest.mark.asyncio
async def test_diff_screen_footer_uses_controller_and_localizes_without_main_transcript_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The production diff navigation wires its active footer into the switch pass."""
    from textual.widgets._footer import FooterKey

    from chrys.app.tui.screens.diff.screen import DiffScreen
    from chrys.app.tui.util.diff_entries import DiffFileEntry
    from chrys.app.tui.widgets.chat.panel import ChatPanel
    from chrys.app.tui.widgets.chrome.footer import ChrysFooter
    from chrys.service.mutations.types import MutationOp

    class _Engine:
        session_generation = 0

        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    entry = DiffFileEntry(
        path=str(tmp_path / "localized.py"),
        rel_path="localized.py",
        operation=MutationOp.MODIFY,
        old_path=None,
        before_text="before\n",
        after_text="after\n",
        is_binary=False,
        encoding="utf-8",
        bytes_changed=True,
    )
    monkeypatch.setattr(tui_i18n, "persist_locale", lambda _locale: None)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(locale="en"),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        assert app._main_screen is not None
        chat = app._main_screen.query_one(ChatPanel)
        app._main_screen._view_adapter.open_diff_screen(
            {1: [entry]},
            cwd=str(tmp_path),
            subtitle_parts=(),
            session_id="session-id",
            all_entries=[entry],
        )
        await wait_for(
            lambda: isinstance(app.screen, DiffScreen) and app.screen._content_ready,
            timeout=10.0,
            pilot=pilot,
            description="diff screen content ready",
        )
        assert isinstance(app.screen, DiffScreen)
        diff_screen = app.screen
        assert diff_screen._locale_controller is app.locale_controller
        footer = diff_screen.query_one(ChrysFooter)
        expected_en_keys = {"Back", "Toggle Split View", "Toggle Change List", "Quit"}
        await wait_for(
            lambda: {key.description for key in footer.query("FooterKey")} == expected_en_keys,
            pilot=pilot,
            description="English footer keys",
        )

        chat_refreshes: list[bool] = []
        chat_restyles: list[None] = []

        def _record_chat_refresh(
            *_regions: object,
            repaint: bool = True,
            layout: bool = False,
            recompose: bool = False,
        ) -> ChatPanel:
            del repaint, recompose
            chat_refreshes.append(layout)
            return chat

        monkeypatch.setattr(chat, "refresh", _record_chat_refresh)
        monkeypatch.setattr(chat, "update_node_styles", lambda animate=True: chat_restyles.append(None))

        assert app.locale_controller.switch_locale("zh-Hans").status is LocaleSwitchStatus.EFFECTIVE_CHANGED
        expected_zh_keys = {"返回", "切换并排视图", "显示/隐藏文件列表", "退出"}
        await wait_for(
            lambda: {key.description for key in footer.query("FooterKey")} == expected_zh_keys,
            pilot=pilot,
            description="Chinese footer keys",
        )
        # One settled pass so the freshly mounted keys gain layout geometry.
        await pilot.pause()
        assert all(key.content_region.width == key.render().cell_len for key in footer.query(FooterKey))
        assert chat_refreshes == []
        assert chat_restyles == []


@pytest.mark.asyncio
async def test_diff_screen_resume_syncs_footer_bindings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resuming the diff screen replays a footer recompose stranded pre-activation."""
    from textual import events

    from chrys.app.tui.screens.diff.screen import DiffScreen
    from chrys.app.tui.util.diff_entries import DiffFileEntry
    from chrys.app.tui.widgets.chrome.footer import ChrysFooter
    from chrys.service.mutations.types import MutationOp

    class _Engine:
        session_generation = 0

        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    entry = DiffFileEntry(
        path=str(tmp_path / "resumed.py"),
        rel_path="resumed.py",
        operation=MutationOp.MODIFY,
        old_path=None,
        before_text="before\n",
        after_text="after\n",
        is_binary=False,
        encoding="utf-8",
        bytes_changed=True,
    )
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(locale="en"),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        assert app._main_screen is not None
        app._main_screen._view_adapter.open_diff_screen(
            {1: [entry]},
            cwd=str(tmp_path),
            subtitle_parts=(),
            session_id="session-id",
            all_entries=[entry],
        )
        await wait_for(
            lambda: isinstance(app.screen, DiffScreen) and app.screen._content_ready,
            timeout=10.0,
            pilot=pilot,
            description="diff screen content ready",
        )
        diff_screen = app.screen
        assert isinstance(diff_screen, DiffScreen)

        footer = diff_screen.query_one(ChrysFooter)
        synced: list[None] = []
        monkeypatch.setattr(footer, "sync_bindings", lambda: synced.append(None))

        diff_screen.post_message(events.ScreenResume())
        await pilot.pause()
        assert synced == [None]

        # A resume that only honors a deferred close must not touch the footer.
        synced.clear()
        diff_screen._close_when_current = True
        monkeypatch.setattr(diff_screen, "_close_if_current", lambda: None)
        diff_screen.post_message(events.ScreenResume())
        await pilot.pause()
        assert synced == []


@pytest.mark.parametrize(
    ("locale", "message", "title"),
    [
        ("en", "Press ctrl+q to quit the app", "Do you want to quit?"),
        ("zh-Hans", "按 ctrl+q 退出应用", "要退出吗？"),  # noqa: RUF001
    ],
)
@pytest.mark.asyncio
async def test_inherited_ctrl_c_help_quit_notifies_in_active_locale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    locale: str,
    message: str,
    title: str,
) -> None:
    """Stock App keeps ctrl+c bound to help_quit; its hint must localize.

    Both lower namespaces that bind ctrl+c to a copy action — the focused
    chat text area and the screen — must defer on an empty selection
    (SkipAction) so the key reaches the app-level hint at all. Press from
    the real post-launch focus to exercise the whole chain.
    """
    from chrys.app.tui.widgets.text_area import EnhancedTextArea

    class _Engine:
        session_generation = 0

        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(locale=locale),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        captured: list[tuple[str, str, bool]] = []

        def _notify(
            notice: str,
            *,
            title: str = "",
            severity: str = "information",
            timeout: float | None = None,
            markup: bool = True,
        ) -> None:
            del severity, timeout
            captured.append((notice, title, markup))

        monkeypatch.setattr(app, "notify", _notify)
        assert isinstance(app.focused, EnhancedTextArea)
        await pilot.press("ctrl+c")
        assert captured == [(message, title, False)]


@pytest.mark.asyncio
async def test_native_modal_css_first_load_updates_only_modal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A new modal stylesheet must not invalidate a populated MainScreen."""
    from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        underlay = VirtualizedMarkdown("# Populated underlay")
        await app.screen.mount(underlay)
        await pilot.pause()

        underlay_style_updates: list[None] = []
        monkeypatch.setattr(underlay, "notify_style_update", lambda: underlay_style_updates.append(None))
        stylesheet_update_roots: list[object] = []
        original_update = app.stylesheet.update

        def _record_update(root: object, animate: bool = False) -> None:
            stylesheet_update_roots.append(root)
            original_update(root, animate=animate)  # type: ignore[arg-type]

        monkeypatch.setattr(app.stylesheet, "update", _record_update)

        first = ConfirmDialog(title="Exit", message="Exit Chrys?", confirm_label="Exit")
        await app.push_screen(first)
        await pilot.pause()

        assert stylesheet_update_roots == [first]
        assert underlay_style_updates == []
        assert first.query_one("#confirm-container").outer_size.width == 43

        first.dismiss(False)
        await pilot.pause()
        stylesheet_update_roots.clear()

        second = ConfirmDialog(title="Exit", message="Exit Chrys?", confirm_label="Exit")
        await app.push_screen(second)
        await pilot.pause()

        assert stylesheet_update_roots == []
        assert second.query_one("#confirm-container").outer_size.width == 43


@pytest.mark.asyncio
async def test_web_modal_css_first_load_retains_full_app_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Web modal CSS keeps Textual's global App:focus/App:blur semantics."""
    from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    monkeypatch.setenv("TEXTUAL_DRIVER", "textual.drivers.web_driver:WebDriver")
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        stylesheet_update_roots: list[object] = []
        original_update = app.stylesheet.update

        def _record_update(root: object, animate: bool = False) -> None:
            stylesheet_update_roots.append(root)
            original_update(root, animate=animate)  # type: ignore[arg-type]

        monkeypatch.setattr(app.stylesheet, "update", _record_update)
        dialog = ConfirmDialog(title="Exit", message="Exit Chrys?", confirm_label="Exit")

        await app.push_screen(dialog)
        await pilot.pause()

        assert stylesheet_update_roots == [app]


@pytest.mark.asyncio
async def test_native_diff_screen_css_first_load_updates_only_diff_screen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A secondary full screen must not restyle the suspended transcript."""
    from chrys.app.tui.screens.diff.screen import DiffScreen

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        stylesheet_update_roots: list[object] = []
        original_update = app.stylesheet.update

        def _record_update(root: object, animate: bool = False) -> None:
            stylesheet_update_roots.append(root)
            original_update(root, animate=animate)  # type: ignore[arg-type]

        monkeypatch.setattr(app.stylesheet, "update", _record_update)
        diff_screen = DiffScreen({}, cwd=str(tmp_path))

        await app.push_screen(diff_screen)
        await pilot.pause()

        assert stylesheet_update_roots == [diff_screen]
        assert diff_screen.query_one("#diff-container").styles.border.top[0] == "round"


@pytest.mark.asyncio
async def test_theme_chrys_classes_join_single_required_global_refresh(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Theme selector classes must not trigger a redundant transcript restyle."""

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    monkeypatch.setattr("chrys.app.tui.app.persist_theme", lambda _theme: None)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(theme="chrys"),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        await app.screen.mount(*(Static("") for _ in range(256)))
        await pilot.pause()
        redundant_screen_updates: list[bool] = []
        monkeypatch.setattr(
            app.screen,
            "update_node_styles",
            lambda animate=True: redundant_screen_updates.append(animate),
        )
        stylesheet_update_roots: list[object] = []
        original_update = app.stylesheet.update

        def _record_update(root: object, animate: bool = False) -> None:
            stylesheet_update_roots.append(root)
            original_update(root, animate=animate)  # type: ignore[arg-type]

        monkeypatch.setattr(app.stylesheet, "update", _record_update)

        app.theme = "textual-dark"

        assert "-chrys" not in app.classes
        assert "-chrys-ansi" not in app.classes
        assert redundant_screen_updates == []

        await pilot.pause()

        assert stylesheet_update_roots[0] is app
        assert sum(root is app for root in stylesheet_update_roots) == 1
        assert redundant_screen_updates == []


@pytest.mark.asyncio
async def test_trajectory_dashboard_visibility_does_not_restyle_chat_descendants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """F12 visibility changes should perform layout without recursive CSS work."""
    from chrys.app.tui.widgets.chat.panel import ChatPanel
    from chrys.app.tui.widgets.chat.session_json import SessionJsonPanel
    from chrys.app.tui.widgets.chrome.input_bar import InputBar
    from chrys.app.tui.widgets.trajectory import DashboardTab, TrajectoryDashboard

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        assert app._main_screen is not None
        chat = app._main_screen.query_one(ChatPanel)
        await chat.mount(*(Static("") for _ in range(256)))
        await pilot.pause()
        recursive_style_updates: list[bool] = []
        monkeypatch.setattr(
            chat,
            "update_node_styles",
            lambda animate=True: recursive_style_updates.append(animate),
        )

        await pilot.press("f12")
        await pilot.pause()
        dashboard = app._main_screen.query_one(TrajectoryDashboard)
        session_json = app._main_screen.query_one(SessionJsonPanel)
        assert chat.display is False
        assert dashboard.foreground is True
        assert dashboard.display is True

        await pilot.click("#session-data")
        await pilot.pause()
        assert dashboard.active_tab is DashboardTab.SESSION_DATA
        assert session_json.display is True
        assert dashboard.display is True

        await pilot.press("escape")
        await pilot.pause()
        assert chat.display is True
        assert dashboard.foreground is False
        assert session_json.display is False
        assert app._main_screen.query_one(InputBar).query_one(TextArea).has_focus

        chat.set_session_id("old-session")
        await pilot.press("f12")
        await pilot.pause()
        assert dashboard.foreground is True
        app._main_screen._view_adapter.set_chat_session_id("new-session")
        assert dashboard.foreground is False
        assert chat.session_id == "new-session"
        assert chat.display is True
        assert recursive_style_updates == []


@pytest.mark.asyncio
async def test_trajectory_dashboard_foreground_shows_only_the_footer_chrome(tmp_path: Path) -> None:
    """A foreground F12 dashboard hides the status and input bars; Esc restores them."""
    from textual.widgets import Footer

    from chrys.app.tui.widgets.chrome.input_bar import InputBar
    from chrys.app.tui.widgets.chrome.status_bar import StatusBar
    from chrys.app.tui.widgets.trajectory import TrajectoryDashboard

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        assert app._main_screen is not None
        status_bar = app._main_screen.query_one(StatusBar)
        input_bar = app._main_screen.query_one(InputBar)
        footer = app._main_screen.query_one(Footer)

        await pilot.press("f12")
        await pilot.pause()
        dashboard = app._main_screen.query_one(TrajectoryDashboard)
        assert dashboard.foreground is True
        assert status_bar.display is False
        assert input_bar.display is False
        assert footer.display is True

        await pilot.press("escape")
        await pilot.pause()
        assert dashboard.foreground is False
        assert status_bar.display is True
        assert input_bar.display is True
        assert footer.display is True


@pytest.mark.asyncio
async def test_visibility_only_widgets_do_not_restyle_descendants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Sidebar, shell, and suggestions should toggle layout without CSS walks."""
    from chrys.app.tui.terminal.panel import ShellPanel
    from chrys.app.tui.widgets.chat.panel import ChatPanel
    from chrys.app.tui.widgets.chrome.input_bar import InputBar
    from chrys.app.tui.widgets.chrome.status_bar import StatusBar
    from chrys.app.tui.widgets.chrome.suggestion_list import SuggestionItem, SuggestionList
    from chrys.app.tui.widgets.sidebar.panel import SidebarPanel

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        sidebar = app.screen.query_one(SidebarPanel)
        shell = app.screen.query_one(ShellPanel)
        suggestions = app.screen.query_one(SuggestionList)
        chat = app.screen.query_one(ChatPanel)
        status = app.screen.query_one(StatusBar)
        input_bar = app.screen.query_one(InputBar)
        recursive_style_updates: list[str] = []
        for name, widget in (("sidebar", sidebar), ("shell", shell), ("suggestions", suggestions)):
            monkeypatch.setattr(
                widget,
                "update_node_styles",
                lambda animate=True, widget_name=name: recursive_style_updates.append(widget_name),
            )
        monkeypatch.setattr(shell, "call_after_refresh", lambda _callback, *args: None)
        monkeypatch.setattr(shell, "_start_shell", lambda _cwd: None)

        sidebar.toggle()
        assert sidebar.is_visible is False
        sidebar.toggle()
        assert sidebar.is_visible is True

        shell.show()
        assert shell.is_visible is True
        shell.hide()
        assert shell.is_visible is False

        status.show("Ready")
        await pilot.pause()
        chat_region = chat.region
        status_region = status.region
        input_region = input_bar.region
        layout_refreshes: list[None] = []
        monkeypatch.setattr(app.screen, "_refresh_layout", lambda *_args, **_kwargs: layout_refreshes.append(None))

        # Drain straggler layout work from setup so anything recorded below is
        # caused by the popup operations under test. Style updates are probed
        # but deliberately NOT cleared: a late restyle from the toggles above
        # is a real contract violation the final assert must still catch.
        await wait_until_quiet(
            lambda: (len(recursive_style_updates), len(layout_refreshes)),
            description="popup style and layout counters",
            pilot=pilot,
        )
        layout_refreshes.clear()

        suggestions.show("commands", [SuggestionItem("/help", "Help")])
        assert suggestions.is_visible is True
        await pilot.pause()
        assert chat.region == chat_region
        assert status.region == status_region
        assert input_bar.region == input_region
        assert suggestions.region.bottom == input_bar.region.y
        assert suggestions.region.overlaps(status.region)

        suggestions.update([SuggestionItem("/help", "Help"), SuggestionItem("/new", "New")])
        await pilot.pause()
        assert chat.region == chat_region
        assert status.region == status_region
        assert input_bar.region == input_region
        assert suggestions.region.bottom == input_bar.region.y
        assert suggestions.region.overlaps(status.region)

        suggestions.hide()
        assert suggestions.is_visible is False
        await pilot.pause()

        assert recursive_style_updates == []
        assert layout_refreshes == []


async def test_escape_collapses_suggestion_popup_before_interrupt_and_exit_prompts(tmp_path: Path) -> None:
    """Esc on an open /, @, # popup collapses it (text kept) — no confirm modal."""
    from chrys.app.tui.screens.dialogs.confirm import ConfirmDialog
    from chrys.app.tui.widgets.chrome.input_bar import InputBar
    from chrys.app.tui.widgets.chrome.suggestion_list import SuggestionList

    class _Engine:
        session_generation = 0

        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        assert app._main_screen is not None
        main_screen = app._main_screen
        suggestions = main_screen.query_one(SuggestionList)
        input_bar = main_screen.query_one(InputBar)
        input_bar.focus_input()
        await pilot.pause()

        await pilot.press("/")
        await pilot.pause()
        assert suggestions.is_visible is True
        assert input_bar.suggestions_active is True

        await pilot.press("escape")
        await pilot.pause()
        assert suggestions.is_visible is False
        assert input_bar.suggestions_active is False
        assert input_bar.query_one("#chat-input", TextArea).text == "/"
        assert app.screen is main_screen  # no interrupt/exit confirm pushed

        # With the popup gone and a run active, Esc prompts Interrupt again.
        main_screen._set_agent_running(True)
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, ConfirmDialog)


@pytest.mark.asyncio
async def test_input_and_status_state_changes_do_not_relayout_large_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Fixed chrome slots must not schedule transcript-wide layouts."""
    from chrys.app.tui.widgets.chrome.input_bar import InputBar
    from chrys.app.tui.widgets.chrome.status_bar import StatusBar

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test() as pilot:
        await app.screen.mount(*(Static("") for _ in range(64)))
        await pilot.pause()
        status = app.screen.query_one(StatusBar)
        input_bar = app.screen.query_one(InputBar)
        status.set_profile("Code Agent")
        status.set_tool_info("3 tools")
        # The 64-widget mount overflows the screen, which grows its vertical
        # scrollbar on a later layout pass; on loaded CI workers that pass can
        # land after any single pause. Snapshot (and patch the layout hook)
        # only once geometry has settled, or the pending scrollbar gutter is
        # applied mid-test by the chrome flips' compositor resync and shows
        # up as a phantom region change.
        await wait_until_quiet(
            lambda: (status.region, input_bar.region),
            description="status and input geometry",
            pilot=pilot,
        )
        status_region = status.region
        input_region = input_bar.region
        layout_refreshes: list[None] = []
        monkeypatch.setattr(app.screen, "_refresh_layout", lambda *_args, **_kwargs: layout_refreshes.append(None))

        # Drain straggler layout work from the 64-widget mount so anything
        # recorded below is caused by the chrome state changes under test.
        await wait_until_quiet(
            lambda: len(layout_refreshes),
            description="post-mount layout refresh count",
            pilot=pilot,
        )
        layout_refreshes.clear()

        status.show("Thinking")
        await pilot.pause()
        status.show("Still thinking")
        await pilot.pause()
        status.flash("Ready", caution=True)
        await pilot.pause()
        status.clear_status()
        await pilot.pause()

        input_bar.agent_running = True
        await pilot.pause()
        input_bar.agent_running = False
        input_bar.has_messages = True
        await pilot.pause()
        input_bar.agent_loading = True
        await pilot.pause()
        input_bar.agent_loading = False
        await pilot.pause()
        input_bar.lock_with_text()
        await pilot.pause()
        input_bar.unlock_and_keep()
        await pilot.pause()

        assert status.region == status_region
        assert input_bar.region == input_region
        assert layout_refreshes == []


@pytest.mark.asyncio
async def test_gc_freeze_opt_out_is_inert(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A disabled coordinator must suppress ownership, timer, and GC mutation."""

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    monkeypatch.setattr(chrys_app, "GC_FREEZE_ENABLED", False)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings.from_env(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._gc_freeze.enabled is False
        assert app._gc_freeze.started is False
        assert app._gc_freeze_watchdog is None
        assert gc.get_freeze_count() == 0


@pytest.mark.asyncio
async def test_enabled_gc_freeze_real_app_lifecycle_returns_two_runs_to_baseline(tmp_path: Path) -> None:
    """Explicit test activation must freeze after mount and fully unwind on every unmount."""
    from chrys.app.tui.support import gc_freeze

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    for run_number in range(2):
        app = ChrysApp(
            EventBus(),
            _Engine(),  # type: ignore[arg-type]
            settings=Settings(),
            state_store=JsonFileStateStore(tmp_path / str(run_number)),
            agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
            gc_freeze_enabled=True,
        )

        async with app.run_test() as pilot:
            for _ in range(5):
                await pilot.pause()
                if app._gc_freeze.frozen:
                    break
            assert app._gc_freeze.started is True
            assert app._gc_freeze.frozen is True
            assert app._gc_freeze_watchdog is not None
            assert app._main_screen is not None
            assert len(app._main_screen._gc_freeze_participants) == 5
            assert gc.get_freeze_count() > 0

        assert gc.get_freeze_count() == 0
        assert gc_freeze._enabled_owner is None


@pytest.mark.asyncio
async def test_gc_freeze_waits_for_backend_turn_finalization_task(tmp_path: Path) -> None:
    """A terminal UI event cannot freeze session-save or after-turn-hook state."""
    from chrys.app.tui.support.gc_freeze import GcAbsorbReason, GcFreezeBlockReason
    from chrys.orchestration.engine.engine import AgentEngine

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    bus = EventBus()
    engine = AgentEngine(bus, settings=Settings(), state_store=JsonFileStateStore(tmp_path / "engine"))
    app = ChrysApp(
        bus,
        engine,
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path / "app"),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=True,
    )

    async with app.run_test() as pilot:
        for _ in range(10):
            await pilot.pause()
            if app._gc_freeze.frozen:
                break
        assert app._gc_freeze.frozen is True
        assert app._gc_freeze_watchdog is not None
        app._gc_freeze_watchdog.pause()

        release_finalization = asyncio.Event()
        run_task = asyncio.create_task(release_finalization.wait())
        engine._turn_state.run_task = run_task
        try:
            # This is the exact interval missed by the UI/FSM gates: the FSM
            # is idle, while the owned task is still saving/draining hooks.
            assert engine.is_turn_active is False
            assert engine.is_turn_lifecycle_active is True
            assert app.freeze_block_reason() is GcFreezeBlockReason.BACKEND_TURN_LIFECYCLE

            previous_metrics = app._gc_freeze.last_action_metrics
            app._gc_freeze.request_absorb(reason=GcAbsorbReason.TURN_TERMINAL, terminal_boundary=True)
            for _ in range(10):
                await pilot.pause()

            assert app._gc_freeze._absorb_pending is True
            assert app._gc_freeze.last_action_metrics is previous_metrics

            release_finalization.set()
            await run_task
            assert engine.is_turn_lifecycle_active is False
            app._gc_freeze.on_tick()
            for _ in range(10):
                await pilot.pause()
                if app._gc_freeze.last_action_metrics is not previous_metrics:
                    break

            assert app._gc_freeze._absorb_pending is False
            assert app._gc_freeze.last_action_metrics is not previous_metrics
            assert app._gc_freeze.last_action_metrics is not None
            assert app._gc_freeze.last_action_metrics.action == "absorb"
        finally:
            release_finalization.set()
            await run_task


@pytest.mark.asyncio
async def test_gc_freeze_prepare_failure_restores_raising_participant_before_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A participant that detaches then raises must be normalized before retry."""
    from chrys.app.tui.support.gc_freeze import DetachedLruCache, GcAbsorbReason
    from chrys.app.tui.widgets.chat.session_json import SessionJsonPanel

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=True,
    )

    async with app.run_test() as pilot:
        for _ in range(10):
            await pilot.pause()
            if app._gc_freeze.frozen:
                break
        assert app._main_screen is not None
        session_json = app._main_screen.query_one(SessionJsonPanel)
        original_prepare = session_json.prepare_for_gc_freeze
        failed = False

        def _fail_once_between_participants() -> None:
            nonlocal failed
            original_prepare()
            assert isinstance(session_json._strip_cache, DetachedLruCache)
            if not failed:
                failed = True
                raise RuntimeError("injected participant prepare failure")

        monkeypatch.setattr(session_json, "prepare_for_gc_freeze", _fail_once_between_participants)
        previous_metrics = app._gc_freeze.last_action_metrics
        app._gc_freeze.request_absorb(reason=GcAbsorbReason.TURN_TERMINAL, terminal_boundary=True)
        for _ in range(10):
            await pilot.pause()
            if app._gc_freeze._prepare_failures == 1:
                break

        assert app._gc_freeze._prepare_failures == 1
        assert not isinstance(session_json._strip_cache, DetachedLruCache)
        session_json._strip_cache.clear()

        app._gc_freeze._prepare_retry_not_before = float("-inf")
        app._gc_freeze.on_tick()
        for _ in range(10):
            await pilot.pause()
            if app._gc_freeze.last_action_metrics is not previous_metrics:
                break

        assert app._gc_freeze.last_action_metrics is not previous_metrics
        assert app._gc_freeze.last_action_metrics is not None
        assert app._gc_freeze.last_action_metrics.action == "absorb"
        assert app._gc_freeze._faulted is False


@pytest.mark.asyncio
async def test_gc_freeze_after_failure_normalizes_mid_screen_renewal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A mid-screen renewal failure must normalize all caches before fail-open returns."""
    from chrys.app.tui.support import gc_freeze
    from chrys.app.tui.support.gc_freeze import DetachedFifoCache, DetachedLruCache, GcAbsorbReason

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=True,
    )

    async with app.run_test() as pilot:
        for _ in range(10):
            await pilot.pause()
            if app._gc_freeze.frozen:
                break
        assert app._main_screen is not None
        original_renew = gc_freeze.renew_lru_cache
        renew_calls = 0

        def _fail_second_screen_renewal(cache):
            nonlocal renew_calls
            renew_calls += 1
            if renew_calls == 2:
                raise RuntimeError("injected screen renewal failure")
            return original_renew(cache)

        monkeypatch.setattr(gc_freeze, "renew_lru_cache", _fail_second_screen_renewal)
        app._gc_freeze.request_absorb(reason=GcAbsorbReason.TURN_TERMINAL, terminal_boundary=True)
        for _ in range(10):
            await pilot.pause()
            if app._gc_freeze._faulted:
                break

        assert renew_calls > 2
        assert app._gc_freeze._faulted is True
        assert app._gc_freeze.frozen is False
        assert gc.get_freeze_count() == 0
        for widget in app._main_screen.walk_children(with_self=True):
            assert not isinstance(widget._box_model_cache, DetachedLruCache)
            assert not isinstance(widget._query_one_cache, DetachedLruCache)
            assert not isinstance(widget._arrangement_cache, DetachedFifoCache)
            widget._box_model_cache.clear()
            widget._query_one_cache.clear()
            widget._arrangement_cache.clear()
        await pilot.pause()


@pytest.mark.asyncio
async def test_foreign_freeze_degradation_publishes_user_visible_startup_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from chrys.app.tui.support import gc_freeze

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    bus = EventBus()
    warnings: list[Warning] = []

    async def _record_warning(event: Warning) -> None:
        warnings.append(event)

    await bus.subscribe(Warning, _record_warning)
    with monkeypatch.context() as patch:
        patch.setattr(gc_freeze.gc, "get_freeze_count", lambda: 5)
        app = ChrysApp(
            bus,
            _Engine(),  # type: ignore[arg-type]
            settings=Settings(),
            state_store=JsonFileStateStore(tmp_path),
            agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
            gc_freeze_enabled=True,
        )

        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._gc_freeze.enabled is False
            assert app._gc_freeze_watchdog is None

    assert [(warning.code, warning.message) for warning in warnings] == [
        (
            "gc_freeze_disabled",
            "GC freeze optimization was disabled because this process already contains 5 externally frozen objects.",
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_hop", [1, 2])
async def test_gc_freeze_recovers_callback_pruned_with_transient_screen(
    lost_hop: int,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A callback accepted by App but stranded on a pruned Screen must be replaced exactly once."""
    from functools import partial

    from textual.screen import Screen

    from chrys.app.tui.support import gc_freeze
    from chrys.app.tui.support.gc_freeze import GcReclaimReason

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    class _PruneOnCallbackScreen(Screen):
        def __init__(self, callback_name: str) -> None:
            self._callback_name = callback_name
            super().__init__()

        def _invoke_later(self, callback, sender) -> None:
            super()._invoke_later(callback, sender)
            callback_repr = repr(callback)
            if self._callback_name not in callback_repr:
                return
            app._transient_callbacks += 1
            app._dropped_callbacks.append(callback_repr)
            self.app.pop_screen()
            # Model Textual pruning the transient pump before its accepted
            # callback queue drains.  App forwarding above is real; only the
            # otherwise timing-sensitive prune is made deterministic.
            self._callbacks.clear()

    class _CallbackLossApp(ChrysApp):
        CSS_PATH = _CHRYS_CSS

        def __init__(self, *args: object, **kwargs: object) -> None:
            self._drop_callback_name: str | None = None
            self._drop_pushes = 0
            self._transient_callbacks = 0
            self._dropped_callbacks: list[str] = []
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]

        def arm_callback_loss(self, hop: int) -> None:
            self._drop_callback_name = "_after_first_refresh" if hop == 1 else "_run"

        def call_after_refresh(self, callback, *args: object, **kwargs: object) -> bool:
            callback_name = callback.func.__name__ if isinstance(callback, partial) else ""
            accepted = super().call_after_refresh(callback, *args, **kwargs)
            if accepted and callback_name == self._drop_callback_name:
                self._drop_pushes += 1
                self._drop_callback_name = None
                self.push_screen(_PruneOnCallbackScreen(callback_name))
            return accepted

    monkeypatch.setattr(gc_freeze, "_SCHEDULE_RECOVERY_SECONDS", 0.0)
    app = _CallbackLossApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=True,
    )

    async with app.run_test() as pilot:
        for _ in range(5):
            await pilot.pause()
            if app._gc_freeze.frozen:
                break
        assert app._gc_freeze.frozen is True
        assert app._gc_freeze_watchdog is not None
        app._gc_freeze_watchdog.pause()

        actions: list[str] = []
        original_log_action = app._gc_freeze._log_action

        def _record_action(action: str) -> None:
            actions.append(action)
            original_log_action(action)

        monkeypatch.setattr(app._gc_freeze, "_log_action", _record_action)
        app.arm_callback_loss(lost_hop)
        app._gc_freeze.request_reclaim(reason=GcReclaimReason.SESSION_READY, prompt=True)

        for _ in range(10):
            await pilot.pause()
            if app.screen is app._main_screen:
                break

        assert app.screen is app._main_screen
        assert app._gc_freeze._prompt_reclaim_pending is True, repr(app._dropped_callbacks)
        assert app._gc_freeze._scheduled is True
        assert app._drop_pushes == 1
        assert app._transient_callbacks == 1
        expected_callback = "_after_first_refresh" if lost_hop == 1 else "_run"
        assert len(app._dropped_callbacks) == 1
        assert expected_callback in app._dropped_callbacks[0]
        assert actions == []

        app._gc_freeze.on_tick()
        for _ in range(10):
            await pilot.pause()
            if not app._gc_freeze._prompt_reclaim_pending:
                break

        assert app._gc_freeze._prompt_reclaim_pending is False
        assert actions == ["full"]
        for _ in range(3):
            await pilot.pause()
        assert actions == ["full"]


@pytest.mark.asyncio
async def test_gc_freeze_app_key_activity_demotes_terminal_before_dispatch(tmp_path: Path) -> None:
    """A real non-forwarded Key must update coordinator chronology before widget dispatch."""
    from textual import events
    from textual.geometry import Size

    from chrys.app.tui.support.gc_freeze import GcAbsorbReason

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=True,
    )

    async with app.run_test() as pilot:
        for _ in range(5):
            await pilot.pause()
            if app._gc_freeze.frozen:
                break
        assert app._gc_freeze.frozen is True

        previous_input_at = app._gc_freeze._last_input_at
        app._gc_freeze._absorb_pending = True
        app._gc_freeze._terminal_boundary_at = previous_input_at
        await pilot.press("x")

        assert app._gc_freeze._last_input_at > previous_input_at
        assert app._gc_freeze._terminal_boundary_at is None
        assert app._gc_freeze._absorb_pending is True

        forwarded = events.Key("y", "y")
        forwarded._set_forwarded()
        last_input_at = app._gc_freeze._last_input_at
        await app.on_event(forwarded)
        assert app._gc_freeze._last_input_at == last_input_at

        passive_move = events.MouseMove(
            None,
            x=1,
            y=1,
            delta_x=1,
            delta_y=0,
            button=0,
            shift=False,
            meta=False,
            ctrl=False,
        )
        await app.on_event(passive_move)
        assert app._gc_freeze._last_input_at == last_input_at

        app._gc_freeze._terminal_boundary_at = last_input_at
        await app.on_event(events.AppFocus())
        assert app._gc_freeze._last_input_at > last_input_at
        assert app._gc_freeze._terminal_boundary_at is None

        app._gc_freeze._absorb_pending = False
        app._gc_freeze._absorb_reasons.clear()
        resize = events.Resize(Size(120, 40), Size(120, 40))
        await app.on_event(resize)
        assert app._gc_freeze._absorb_pending is True
        assert app._gc_freeze._absorb_reasons == {GcAbsorbReason.VIEWPORT_UPDATED}


@pytest.mark.asyncio
async def test_gc_freeze_effective_theme_change_requests_idle_reclaim(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from chrys.app.tui.support.gc_freeze import GcReclaimReason

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    monkeypatch.setattr("chrys.app.tui.app.persist_theme", lambda _theme: None)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(theme="chrys"),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=True,
    )

    async with app.run_test() as pilot:
        for _ in range(5):
            await pilot.pause()
            if app._gc_freeze.frozen:
                break
        assert app._gc_freeze.frozen is True

        app._gc_freeze._absorb_pending = False
        app._gc_freeze._absorb_reasons.clear()
        app._gc_freeze._idle_reclaim_pending = False
        app._gc_freeze._idle_reclaim_reasons.clear()
        app.theme = app.theme
        assert app._gc_freeze._absorb_pending is False
        assert app._gc_freeze._idle_reclaim_pending is False

        app.theme = "textual-dark"
        assert app._gc_freeze._absorb_pending is False
        assert app._gc_freeze._idle_reclaim_pending is True
        assert app._gc_freeze._idle_reclaim_reasons == {GcReclaimReason.THEME_UPDATED}
        assert app._gc_freeze._terminal_boundary_at is None


@pytest.mark.asyncio
async def test_gc_freeze_pointer_hold_blocks_gc_and_lost_button_paths_reconcile(tmp_path: Path) -> None:
    """Held selection gestures are hard gates; release and lost-button paths restore progress."""
    from textual import events

    from chrys.app.tui.support.gc_freeze import GcFreezeBlockReason, GcReclaimReason

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    def _mouse_event(event_type: type[events.MouseEvent], button: int) -> events.MouseEvent:
        return event_type(
            None,
            x=1,
            y=1,
            delta_x=0,
            delta_y=0,
            button=button,
            shift=False,
            meta=False,
            ctrl=False,
        )

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=True,
    )

    async with app.run_test() as pilot:
        for _ in range(5):
            await pilot.pause()
            if app._gc_freeze.frozen:
                break
        assert app._gc_freeze.frozen is True

        actions: list[str] = []
        original_log_action = app._gc_freeze._log_action

        def _record_action(action: str) -> None:
            actions.append(action)
            original_log_action(action)

        app._gc_freeze._log_action = _record_action
        await app.on_event(_mouse_event(events.MouseDown, 1))
        assert app.freeze_block_reason() is GcFreezeBlockReason.POINTER_BUTTON_HELD

        app._gc_freeze.request_reclaim(reason=GcReclaimReason.SESSION_READY, prompt=True)
        for _ in range(5):
            await pilot.pause()
        for _ in range(1_000):
            app._gc_freeze.on_tick()

        assert actions == []
        assert app._gc_freeze._prompt_reclaim_pending is True
        last_input_at = app._gc_freeze._last_input_at
        drag_move = _mouse_event(events.MouseMove, 1)
        drag_move.time = last_input_at + 1.0
        await app.on_event(drag_move)
        assert app._gc_freeze._last_input_at == drag_move.time
        assert app.freeze_block_reason() is GcFreezeBlockReason.POINTER_BUTTON_HELD

        await app.on_event(_mouse_event(events.MouseUp, 1))
        assert app._gc_pointer_buttons_down == set()
        app._gc_freeze.on_tick()
        for _ in range(5):
            await pilot.pause()
            if not app._gc_freeze._prompt_reclaim_pending:
                break
        assert actions == ["full"]

        await app.on_event(_mouse_event(events.MouseDown, 1))
        await app.on_event(_mouse_event(events.MouseMove, 0))
        assert app._gc_pointer_buttons_down == set()
        await app.on_event(_mouse_event(events.MouseDown, 1))
        blur_input_at = app._gc_freeze._last_input_at
        await app.on_event(events.AppBlur())
        assert app._gc_pointer_buttons_down == set()
        assert app._gc_freeze._last_input_at == blur_input_at
        await app.on_event(_mouse_event(events.MouseDown, 1))
        focus_input_at = app._gc_freeze._last_input_at
        focus = events.AppFocus()
        focus.time = focus_input_at + 1.0
        await app.on_event(focus)
        assert app._gc_pointer_buttons_down == set()
        assert app._gc_freeze._last_input_at == focus.time


def test_tui_startup_profile_resolves_preferred_agent() -> None:
    from chrys.app.tui.app import _resolve_startup_profile
    from chrys.service.profiles.agents.registry import AgentProfileRegistry
    from chrys.service.profiles.agents.schema import AgentProfile

    registry = AgentProfileRegistry()
    code = AgentProfile(name="Code")
    qa = AgentProfile(name="QA")
    registry.register(code)
    registry.register(qa)

    assert _resolve_startup_profile(registry, "QA") is qa


def test_tui_startup_profile_accepts_id_and_display_name() -> None:
    from chrys.app.tui.app import _resolve_startup_profile
    from chrys.service.profiles.agents.registry import AgentProfileRegistry
    from chrys.service.profiles.agents.schema import AgentProfile

    registry = AgentProfileRegistry()
    code = AgentProfile(name="Code")
    qa = AgentProfile(name="QA", id="qa-id", display_name="Quality Agent")
    registry.register(code)
    registry.register(qa)

    assert _resolve_startup_profile(registry, "qa-id") is qa
    assert _resolve_startup_profile(registry, "quality agent") is qa


def test_tui_canonical_active_model_profile_id_resolves_explicit_profile_name() -> None:
    from chrys.app.tui.screens.main.screen import _canonical_active_model_profile_id
    from chrys.service.profiles.models.registry import ModelProfileRegistry
    from chrys.service.profiles.models.schema import ModelProfile

    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="model-id", name="Friendly Model"))

    assert _canonical_active_model_profile_id(registry, "friendly model") == "model-id"


def test_tui_canonical_active_model_profile_id_prefers_explicit_active_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.app.tui.screens.main.screen import _canonical_active_model_profile_id
    from chrys.service.profiles.models.registry import ModelProfileRegistry
    from chrys.service.profiles.models.schema import ModelProfile

    registry = ModelProfileRegistry()
    registry.register(ModelProfile(id="env-model", name="Env Model"))
    registry.register(ModelProfile(id="active-model", name="Active Model"))
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "env-model")

    assert _canonical_active_model_profile_id(registry, "active-model") == "active-model"


def test_tui_startup_profile_falls_back_to_code_when_preferred_missing() -> None:
    from chrys.app.tui.app import _resolve_startup_profile
    from chrys.service.profiles.agents.registry import AgentProfileRegistry
    from chrys.service.profiles.agents.schema import AgentProfile

    registry = AgentProfileRegistry()
    qa = AgentProfile(name="QA")
    code = AgentProfile(name="Code")
    registry.register(qa)
    registry.register(code)

    assert _resolve_startup_profile(registry, "Missing") is code


def test_tui_startup_profile_treats_sub_agent_only_preference_as_unusable() -> None:
    from chrys.app.tui.app import _resolve_startup_profile
    from chrys.service.profiles.agents.registry import AgentProfileRegistry
    from chrys.service.profiles.agents.schema import AgentProfile

    registry = AgentProfileRegistry()
    code = AgentProfile(name="Code")
    explore = AgentProfile(name="Explore", sub_agent_only=True)
    registry.register(explore)
    registry.register(code)

    assert _resolve_startup_profile(registry, "Explore") is code


def test_debug_logging_is_bounded_and_queue_backed(tmp_path: Path) -> None:
    """File logging should be capped and moved off the caller thread."""

    root_logger = logging.getLogger()
    previous_handlers = root_logger.handlers[:]
    previous_level = root_logger.level
    for handler in previous_handlers:
        root_logger.removeHandler(handler)

    runtime = None
    log_path = tmp_path / "debug.log"
    try:
        runtime = debug_log.configure_debug_logging(log_path)

        assert root_logger.handlers == [runtime.queue_handler]
        assert isinstance(runtime.queue_handler, QueueHandler)
        assert runtime.queue_handler.queue.maxsize == debug_log._DEBUG_LOG_QUEUE_MAX_RECORDS
        assert runtime.file_handler.maxBytes == debug_log._DEBUG_LOG_MAX_BYTES
        assert runtime.file_handler.backupCount == 1
        assert runtime.file_handler.stream is None

        logging.getLogger("chrys.test").warning("bounded log entry")
    finally:
        if runtime is not None:
            debug_log.stop_debug_logging(runtime)
        root_logger.handlers.clear()
        root_logger.setLevel(previous_level)
        for handler in previous_handlers:
            root_logger.addHandler(handler)

    assert "bounded log entry" in log_path.read_text(encoding="utf-8")


def test_debug_log_handler_closes_shared_file_after_emit(tmp_path: Path) -> None:
    """The shared debug log should not stay open between queued writes."""

    log_path = tmp_path / "debug.log"
    handler = debug_log._DebugRotatingFileHandler(
        log_path,
        maxBytes=debug_log._DEBUG_LOG_MAX_BYTES,
        backupCount=1,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        record = logging.LogRecord("chrys.test", logging.WARNING, __file__, 1, "one line", None, None)
        handler.emit(record)
    finally:
        handler.close()

    assert handler.stream is None
    assert log_path.read_text(encoding="utf-8") == "one line\n"


def test_debug_log_handler_still_rotates_when_unlocked(tmp_path: Path) -> None:
    """The shared debug log handler should retain normal rotation behavior."""

    log_path = tmp_path / "debug.log"
    backup_path = tmp_path / "debug.log.1"
    log_path.write_text("old line\n", encoding="utf-8")
    handler = debug_log._DebugRotatingFileHandler(
        log_path,
        maxBytes=1,
        backupCount=1,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        record = logging.LogRecord("chrys.test", logging.WARNING, __file__, 1, "new line", None, None)
        handler.emit(record)
    finally:
        handler.close()

    assert handler.stream is None
    assert backup_path.read_text(encoding="utf-8") == "old line\n"
    assert log_path.read_text(encoding="utf-8") == "new line\n"


def test_debug_log_rollover_permission_error_keeps_logging(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A denied rollover rename should keep the existing backup and stay quiet."""

    log_path = tmp_path / "debug.log"
    backup_path = tmp_path / "debug.log.1"
    log_path.write_text("existing\n", encoding="utf-8")
    backup_path.write_text("previous backup\n", encoding="utf-8")
    handler = debug_log._DebugRotatingFileHandler(
        log_path,
        maxBytes=1,
        backupCount=1,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    def _blocked_rotate(_source: str, _dest: str) -> None:
        raise PermissionError("locked by another process")

    handler.rotate = _blocked_rotate
    try:
        record = logging.LogRecord("chrys.test", logging.WARNING, __file__, 1, "after lock", None, None)
        handler.emit(record)
    finally:
        handler.close()

    assert "after lock" in log_path.read_text(encoding="utf-8")
    assert backup_path.read_text(encoding="utf-8") == "previous backup\n"
    assert capsys.readouterr().err == ""


def test_debug_log_rollover_restore_temp_when_backup_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If replacing debug.log.1 fails, restore the active log from rollover temp."""

    log_path = tmp_path / "debug.log"
    backup_path = tmp_path / "debug.log.1"
    log_path.write_text("old active\n", encoding="utf-8")
    backup_path.write_text("locked backup\n", encoding="utf-8")
    original_replace = debug_log.os.replace

    def _replace_unless_backup(source: str, dest: str) -> None:
        if Path(dest) == backup_path:
            raise PermissionError("backup locked by another process")
        original_replace(source, dest)

    monkeypatch.setattr(debug_log.os, "replace", _replace_unless_backup)
    handler = debug_log._DebugRotatingFileHandler(
        log_path,
        maxBytes=1,
        backupCount=1,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    try:
        record = logging.LogRecord("chrys.test", logging.WARNING, __file__, 1, "after restore", None, None)
        handler.emit(record)
    finally:
        handler.close()

    assert log_path.read_text(encoding="utf-8") == "old active\nafter restore\n"
    assert backup_path.read_text(encoding="utf-8") == "locked backup\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_debug_log_handler_silences_internal_file_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Debug file logging failures should not leak logging diagnostics to the TUI."""

    log_path = tmp_path / "debug.log"
    handler = debug_log._DebugRotatingFileHandler(
        log_path,
        maxBytes=1,
        backupCount=1,
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))

    def _blocked_open() -> object:
        raise PermissionError("log file is unavailable")

    handler._open = _blocked_open
    try:
        record = logging.LogRecord("chrys.test", logging.WARNING, __file__, 1, "lost", None, None)
        handler.emit(record)
    finally:
        handler.close()

    assert capsys.readouterr().err == ""


def test_debug_log_queue_drops_oldest_when_full() -> None:
    """A stalled writer should not let debug logging grow memory without bound."""

    queue: Queue[logging.LogRecord | None] = Queue(maxsize=1)
    handler = debug_log._BoundedDebugLogHandler(queue)
    old_record = logging.LogRecord("chrys.test", logging.INFO, __file__, 1, "old", None, None)
    new_record = logging.LogRecord("chrys.test", logging.INFO, __file__, 2, "new", None, None)

    handler.enqueue(old_record)
    handler.enqueue(new_record)

    assert queue.get_nowait() is new_record


def test_debug_log_listener_can_stop_with_full_queue() -> None:
    """Shutdown should not fail just because the bounded queue is full."""

    queue: Queue[logging.LogRecord | None] = Queue(maxsize=1)
    old_record = logging.LogRecord("chrys.test", logging.INFO, __file__, 1, "old", None, None)
    queue.put_nowait(old_record)

    listener = debug_log._BoundedDebugLogListener(queue)
    listener.enqueue_sentinel()

    assert queue.get_nowait() is listener._sentinel


def test_debug_log_listener_skips_records_after_stop_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown should stop spending time on debug writes after the stop budget."""

    emitted: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            emitted.append(record.getMessage())

    listener = debug_log._BoundedDebugLogListener(Queue(), _Handler())
    record = logging.LogRecord("chrys.test", logging.INFO, __file__, 1, "late", None, None)

    listener._stop_deadline = 1.0
    monkeypatch.setattr(debug_log.time, "monotonic", lambda: 2.0)
    listener.handle(record)
    assert emitted == []

    listener._stop_deadline = 3.0
    listener.handle(record)
    assert emitted == ["late"]


def test_terminal_restore_skips_textual_web_driver(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """textual-serve captures stderr, so terminal reset escapes must stay native-only."""
    from chrys.app.tui import app as tui_app

    class _App:
        _return_code = 0

    monkeypatch.setenv("TEXTUAL_DRIVER", "textual.drivers.web_driver:WebDriver")

    tui_app._restore_terminal_after_run(_App())  # type: ignore[arg-type]

    assert capsys.readouterr().err == ""


def test_terminal_restore_writes_native_reset(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Native TUI runs keep the terminal reset safety net."""
    from chrys.app.tui import app as tui_app

    class _App:
        _return_code = 0

    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)

    tui_app._restore_terminal_after_run(_App())  # type: ignore[arg-type]

    reset = capsys.readouterr().err

    assert "\x1b[?1049l" in reset
    assert "\x1b[?1002l" in reset
    assert "\x1b[?1004l" in reset
    assert "\x1b[<u" in reset
    assert "\x1b[=0;1u" in reset


def test_main_restores_terminal_when_app_run_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_platform,
) -> None:
    """The terminal reset safety net must run even when Textual exits through an exception."""
    from chrys.app.tui import app as tui_app
    from chrys.app.tui.screens import logs as logs_mod
    from chrys.orchestration import startup as startup_mod

    calls: list[str] = []

    class _App:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self._return_code = 1

        def run(self) -> None:
            calls.append("run")
            raise RuntimeError("boom")

    class _Registry:
        def load_all(self) -> None:
            return

    class _Engine:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.settings_handle = SettingsHandle(LoadedSettings(settings=Settings(), provenance={}))

    monkeypatch.setattr(sys, "argv", ["chrys"])
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform(config_dir=tmp_path))
    monkeypatch.setattr(startup_mod, "configure_utf8_stdio", lambda: None)
    monkeypatch.setattr(
        startup_mod,
        "bootstrap_runtime",
        lambda *, dotenv_override, project_root: RuntimeBootstrap(
            loaded=LoadedSettings(settings=Settings(), provenance={})
        ),
    )
    monkeypatch.setattr(logs_mod, "install_log_handler", lambda: None)
    monkeypatch.setattr(tui_app, "configure_debug_logging", lambda _path: object())
    monkeypatch.setattr(tui_app, "stop_debug_logging", lambda _runtime: calls.append("stop-debug"))
    monkeypatch.setattr(tui_app, "_restore_terminal_after_run", lambda _app: calls.append("restore-terminal"))
    monkeypatch.setattr(tui_app, "_reset_terminal_title_after_run", lambda: calls.append("reset-title"))
    monkeypatch.setattr(tui_app, "EventBus", object)
    monkeypatch.setattr(tui_app, "AgentProfileRegistry", _Registry)
    monkeypatch.setattr(tui_app, "ModelProfileRegistry", _Registry)
    monkeypatch.setattr(tui_app, "JsonFileStateStore", object)
    monkeypatch.setattr(tui_app, "AgentEngine", _Engine)

    with pytest.raises(RuntimeError, match="boom"):
        tui_app.main(app_cls=_App)  # type: ignore[arg-type]

    assert calls == ["run", "restore-terminal", "reset-title", "stop-debug"]


def test_main_passes_startup_args_to_app(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fake_platform,
) -> None:
    """The TUI parser should apply high-priority startup args before app construction."""
    from chrys.app.tui import app as tui_app
    from chrys.app.tui.screens import logs as logs_mod
    from chrys.orchestration import startup as startup_mod
    from chrys.service.profiles.models.schema import ModelProfile

    calls: list[str] = []
    captured_kwargs: dict[str, object] = {}
    captured_engine_kwargs: dict[str, object] = {}
    engine_handle: list[SettingsHandle] = []
    workdir = tmp_path / "workspace"
    workdir.mkdir()

    class _App:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            self._return_code = 0
            captured_kwargs.update(kwargs)

        def run(self) -> None:
            calls.append(f"cwd:{Path.cwd()}")
            calls.append("run")

    class _AgentRegistry:
        def load_all(self) -> None:
            return

    class _ModelRegistry:
        def __init__(self) -> None:
            self.profiles = [ModelProfile(id="model-id", name="Friendly Model")]

        def load_all(self) -> None:
            return

        def get(self, profile_id: str) -> ModelProfile | None:
            for profile in self.profiles:
                if profile.id == profile_id:
                    return profile
            return None

        def list_profiles(self) -> list[ModelProfile]:
            return list(self.profiles)

    class _Engine:
        def __init__(self, *_args: object, **kwargs: object) -> None:
            captured_engine_kwargs.update(kwargs)
            loaded = kwargs["loaded_settings"]
            assert isinstance(loaded, LoadedSettings)
            engine_handle.append(SettingsHandle(loaded))

        @property
        def settings_handle(self) -> SettingsHandle:
            return engine_handle[0]

    monkeypatch.setattr(
        sys,
        "argv",
        ["chrys", "-s", "session-1", "-a", "Code", "-m", "Friendly Model", "-C", str(workdir)],
    )
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "")
    monkeypatch.delenv("CHRYS_MODEL_PROFILE")
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform(config_dir=tmp_path))
    monkeypatch.setattr(startup_mod, "configure_utf8_stdio", lambda: None)
    bootstrap_roots: list[Path] = []

    def _fake_bootstrap(*, dotenv_override: bool, project_root: Path) -> RuntimeBootstrap:
        _ = dotenv_override
        bootstrap_roots.append(project_root)
        return RuntimeBootstrap(loaded=LoadedSettings(settings=Settings(), provenance={}))

    monkeypatch.setattr(startup_mod, "bootstrap_runtime", _fake_bootstrap)
    monkeypatch.setattr(logs_mod, "install_log_handler", lambda: None)
    monkeypatch.setattr(tui_app, "configure_debug_logging", lambda _path: object())
    monkeypatch.setattr(tui_app, "stop_debug_logging", lambda _runtime: calls.append("stop-debug"))
    monkeypatch.setattr(tui_app, "_restore_terminal_after_run", lambda _app: calls.append("restore-terminal"))
    monkeypatch.setattr(tui_app, "_reset_terminal_title_after_run", lambda: calls.append("reset-title"))
    monkeypatch.setattr(tui_app, "EventBus", object)
    monkeypatch.setattr(tui_app, "AgentProfileRegistry", _AgentRegistry)
    monkeypatch.setattr(tui_app, "ModelProfileRegistry", _ModelRegistry)
    monkeypatch.setattr(tui_app, "JsonFileStateStore", object)
    monkeypatch.setattr(tui_app, "AgentEngine", _Engine)

    original_cwd = Path.cwd()
    try:
        tui_app.main(app_cls=_App)  # type: ignore[arg-type]
    finally:
        os.chdir(original_cwd)

    assert captured_kwargs["startup_session_id"] == "session-1"
    assert captured_kwargs["profile_name"] == "Code"
    assert captured_kwargs["apply_saved_model_on_restore"] is False
    # ``-C`` must chdir before bootstrap: the workdir names the project
    # trust domain the settings load reads from.
    assert bootstrap_roots == [workdir]
    loaded = captured_engine_kwargs["loaded_settings"]
    assert isinstance(loaded, LoadedSettings)
    # The app must read through the engine's own handle. Anything else — even
    # an equal LoadedSettings — is a second holder that drifts the moment the
    # user switches theme, so identity is the assertion.
    assert captured_kwargs["settings_handle"] is engine_handle[0]
    assert engine_handle[0].loaded is loaded
    settings = loaded.settings
    assert settings.model_profile == "model-id"
    assert settings.model_profile_override == ""
    assert settings.model_profile_override_sub_agents is False
    assert os.environ["CHRYS_MODEL_PROFILE"] == "model-id"
    # The successful-turn callback composes buddy persistence with the
    # session-title updater; turn-start cancellation goes to the updater.
    title_updater = captured_kwargs["session_title_updater"]
    assert isinstance(title_updater, tui_app.SessionTitleUpdater)
    buddy_calls: list[str] = []
    monkeypatch.setattr(tui_app, "on_buddy_successful_turn", lambda: buddy_calls.append("buddy"))
    monkeypatch.setattr(title_updater, "on_turn_finished", lambda: buddy_calls.append("title"))
    captured_engine_kwargs["on_successful_turn"]()
    assert buddy_calls == ["buddy", "title"]
    assert captured_engine_kwargs["on_turn_started"].__self__ is title_updater
    assert calls == [f"cwd:{workdir}", "run", "restore-terminal", "reset-title", "stop-debug"]


def test_tui_locale_parser_sanitizes_unknown_option_while_usage_stays_english(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from chrys.app.tui import app as tui_app
    from chrys.orchestration import startup as startup_mod

    hostile_option = "--hostile=escape\x1b\nline"
    monkeypatch.setenv("CHRYS_LOCALE", "zh-Hans")
    monkeypatch.setattr(sys, "argv", ["chrys", hostile_option])
    monkeypatch.setattr(startup_mod, "configure_utf8_stdio", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        tui_app.main()

    stderr = capsys.readouterr().err
    error_line = stderr.rstrip("\n").splitlines()[-1]
    assert exc_info.value.code == 2
    assert "usage: chrys" in stderr
    assert "error: unrecognized arguments:" in error_line
    assert "--hostile=escape��line" in error_line
    assert "\x1b" not in stderr


def test_tui_locale_parser_sanitizes_hostile_missing_workdir_while_usage_stays_english(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from chrys.app.tui import app as tui_app
    from chrys.orchestration import startup as startup_mod

    hostile_workdir = f"{tmp_path / 'missing'}\x1b\npayload"
    sanitized_workdir = hostile_workdir.replace("\x1b", "�").replace("\n", "�")
    monkeypatch.setenv("CHRYS_LOCALE", "zh-Hans")
    monkeypatch.setattr(sys, "argv", ["chrys", "--workdir", hostile_workdir])
    monkeypatch.setattr(startup_mod, "configure_utf8_stdio", lambda: None)

    with pytest.raises(SystemExit) as exc_info:
        tui_app.main()

    stderr = capsys.readouterr().err
    error_line = stderr.rstrip("\n").splitlines()[-1]
    assert exc_info.value.code == 2
    assert "usage: chrys" in stderr
    assert "error: workdir does not exist or is not a directory:" in error_line
    assert sanitized_workdir in error_line
    assert "\x1b" not in stderr


@pytest.mark.asyncio
async def test_unmount_drains_title_updater_after_engine_shutdown() -> None:
    """Engine shutdown can finalize a completed run whose success callback
    schedules one last title task; draining the updater afterwards is what
    guarantees that task gets cancelled and awaited."""
    calls: list[str] = []

    class _Updater:
        async def shutdown(self) -> None:
            calls.append("updater")

    class _Engine:
        async def shutdown(self) -> None:
            calls.append("engine")

    class _Freeze:
        def close(self) -> None:
            calls.append("gc-close")

    class _Timer:
        def stop(self) -> None:
            calls.append("timer-stop")

    host = SimpleNamespace(
        _startup_task=None,
        _session_title_updater=_Updater(),
        _engine=_Engine(),
        _gc_freeze=_Freeze(),
        _gc_freeze_watchdog=_Timer(),
    )

    await ChrysApp.on_unmount(host)

    assert calls == ["gc-close", "timer-stop", "engine", "updater"]
    assert host._gc_freeze_watchdog is None


def test_terminal_title_reset_after_run_uses_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    """After Textual exits, the host shell should not keep the last prompt preview."""
    from chrys.app.tui import app as tui_app

    monkeypatch.delenv("TEXTUAL_DRIVER", raising=False)
    monkeypatch.chdir(tmp_path)

    tui_app._reset_terminal_title_after_run()

    title = str(tmp_path)
    assert capsys.readouterr().err == f"\x1b]0;{title}\x07\x1b]2;{title}\x07"


def test_debug_log_queue_preserves_stop_sentinel_when_full() -> None:
    """A late log enqueue must not drop the listener shutdown sentinel."""

    queue: Queue[logging.LogRecord | None] = Queue(maxsize=1)
    handler = debug_log._BoundedDebugLogHandler(queue)
    record = logging.LogRecord("chrys.test", logging.INFO, __file__, 1, "late", None, None)
    queue.put_nowait(None)

    handler.enqueue(record)

    assert queue.get_nowait() is None


def test_debug_log_queue_preserves_stop_sentinel_when_requeue_races() -> None:
    """The shutdown sentinel should survive if another producer refills the freed slot."""

    filler_record = logging.LogRecord("chrys.test", logging.INFO, __file__, 1, "filler", None, None)

    class _RacySentinelQueue(Queue[logging.LogRecord | None]):
        def __init__(self) -> None:
            super().__init__(maxsize=1)
            self._armed = False
            self._filled_once = False

        def arm_race(self) -> None:
            self._armed = True

        def put_nowait(self, item: logging.LogRecord | None) -> None:
            if self._armed and item is None and not self._filled_once and self.empty():
                self._filled_once = True
                super().put_nowait(filler_record)
            super().put_nowait(item)

    queue = _RacySentinelQueue()
    handler = debug_log._BoundedDebugLogHandler(queue)
    record = logging.LogRecord("chrys.test", logging.INFO, __file__, 2, "late", None, None)
    queue.put_nowait(None)
    queue.arm_race()

    handler.enqueue(record)

    assert queue.get_nowait() is None


def test_debug_logging_caps_existing_active_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An oversized pre-existing debug.log should become a capped backup."""

    monkeypatch.setattr(debug_log, "_DEBUG_LOG_MAX_BYTES", 80)

    log_path = tmp_path / "debug.log"
    backup_path = tmp_path / "debug.log.1"
    log_path.write_text(("old active line\n" * 20) + "active tail\n", encoding="utf-8")
    backup_path.write_text("stale backup\n", encoding="utf-8")

    debug_log._normalize_debug_log_files(log_path)

    assert log_path.read_text(encoding="utf-8") == ""
    assert backup_path.stat().st_size <= 80
    assert "active tail" in backup_path.read_text(encoding="utf-8")


def test_debug_logging_caps_existing_backup_log(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An oversized pre-existing debug.log.1 should be capped even when active log is small."""

    monkeypatch.setattr(debug_log, "_DEBUG_LOG_MAX_BYTES", 80)

    log_path = tmp_path / "debug.log"
    backup_path = tmp_path / "debug.log.1"
    log_path.write_text("active\n", encoding="utf-8")
    backup_path.write_text(("old backup line\n" * 20) + "backup tail\n", encoding="utf-8")

    debug_log._normalize_debug_log_files(log_path)

    assert log_path.read_text(encoding="utf-8") == "active\n"
    assert backup_path.stat().st_size <= 80
    assert "backup tail" in backup_path.read_text(encoding="utf-8")


def test_tui_help_does_not_offer_profile(monkeypatch: pytest.MonkeyPatch, capsys, tmp_path, fake_platform) -> None:
    """TUI profile selection should happen inside the app, not through argv."""
    from chrys.app.tui import app as tui_app

    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform(config_dir=tmp_path))
    monkeypatch.setattr("chrys.app.tui.screens.logs.install_log_handler", lambda: None)
    monkeypatch.setattr(sys, "argv", ["chrys", "--help"])
    for key in ("PYTHONUTF8", "PYTHONIOENCODING"):
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)

    with pytest.raises(SystemExit) as exc_info:
        tui_app.main()

    assert exc_info.value.code == 0
    out = capsys.readouterr()
    assert "--profile" not in out.out


def _attach_startup_facade(screen: object) -> object:
    """Add the public MainScreen startup facade to lightweight test doubles."""

    def set_startup_agent_loading(value: bool) -> None:
        screen._set_agent_loading(value)  # type: ignore[attr-defined]

    def is_startup_agent_loading() -> bool:
        return bool(getattr(screen, "_agent_loading", False))

    async def restore_startup_session(session_id: str) -> bool:
        await screen._sessions.do_session_restore(session_id, allow_while_loading=True)  # type: ignore[attr-defined]
        return True

    async def dismiss_startup_load_dialog_before_restore() -> None:
        return

    def cancel_startup_session_restore() -> None:
        return

    screen.set_startup_agent_loading = set_startup_agent_loading  # type: ignore[attr-defined]
    screen.is_startup_agent_loading = is_startup_agent_loading  # type: ignore[attr-defined]
    screen.restore_startup_session = restore_startup_session  # type: ignore[attr-defined]
    screen.dismiss_startup_load_dialog_before_restore = dismiss_startup_load_dialog_before_restore  # type: ignore[attr-defined]
    screen.cancel_startup_session_restore = cancel_startup_session_restore  # type: ignore[attr-defined]
    return screen


@pytest.mark.asyncio
async def test_start_engine_surfaces_early_startup_failure() -> None:
    """Failures before AgentLoadStarted should still be visible to the user."""

    class _Engine:
        async def start(self, _profile: object) -> None:
            raise RuntimeError("startup exploded")

    flashes: list[tuple[str, bool]] = []
    notifications: list[tuple[str, str, str]] = []
    loading_states: list[bool] = []

    class _StatusBar:
        def flash(self, text: MessageRef | str, *, error: bool = False, **_kwargs: object) -> None:
            rendered = text if isinstance(text, str) else format_message(text)
            flashes.append((rendered, error))

    class _Screen:
        _agent_loading = True

        def _set_agent_loading(self, value: bool) -> None:
            self._agent_loading = value
            loading_states.append(value)

        def query_one(self, cls: type) -> _StatusBar:
            if cls.__name__ != "StatusBar":
                raise AssertionError(f"unexpected query_one({cls.__name__})")
            return _StatusBar()

        def notify(self, message: str, *, title: str, severity: str, **_kwargs: object) -> None:
            notifications.append((title, severity, message))

    app = object.__new__(ChrysApp)
    app._engine = _Engine()
    app._locale_controller = tui_i18n.LocaleController(Settings(locale="en"))
    app._startup_session_id = ""
    app._deferred_settings_warnings = []
    screen = _attach_startup_facade(_Screen())

    await app._start_engine(object(), screen)  # type: ignore[arg-type]

    assert loading_states == [False]
    assert flashes == [("Agent startup failed: startup exploded", True)]
    assert notifications == [("Agent startup failed", "error", "startup exploded")]


@pytest.mark.asyncio
async def test_start_engine_restores_startup_session_after_start() -> None:
    calls: list[tuple[str, object]] = []

    class _Engine:
        async def prepare(self, profile: object) -> None:
            calls.append(("prepare", profile))

        async def reset_after_failed_startup_restore(self) -> None:
            calls.append(("reset_restore", None))

        async def start(self, profile: object) -> None:
            calls.append(("start", profile))

    class _Store:
        async def load_session_meta(self, session_id: str) -> object:
            calls.append(("meta", session_id))
            return SimpleNamespace(session_id=session_id)

    class _Sessions:
        async def do_session_restore(self, session_id: str, *, allow_while_loading: bool = False) -> None:
            calls.append(("restore", session_id, allow_while_loading))

    app = object.__new__(ChrysApp)
    app._engine = _Engine()
    app._locale_controller = tui_i18n.LocaleController(Settings(locale="en"))
    app._state_store = _Store()
    app._startup_session_id = "session-1"
    app._deferred_settings_warnings = []
    screen = _attach_startup_facade(
        SimpleNamespace(
            _sessions=_Sessions(),
            _set_agent_loading=lambda value: calls.append(("loading", value)),
        )
    )
    profile = object()

    await app._start_engine(profile, screen)  # type: ignore[arg-type]

    assert calls == [
        ("prepare", profile),
        ("loading", True),
        ("meta", "session-1"),
        ("restore", "session-1", True),
    ]
    assert app._startup_session_id == ""


@pytest.mark.asyncio
async def test_start_engine_restores_canonical_startup_session_id() -> None:
    calls: list[tuple[str, object]] = []

    class _Engine:
        async def prepare(self, profile: object) -> None:
            calls.append(("prepare", profile))

        async def start(self, profile: object) -> None:
            calls.append(("start", profile))

    class _Store:
        async def load_session_meta(self, session_id: str) -> object:
            calls.append(("meta", session_id))
            return SimpleNamespace(session_id="canonical-session-id")

    class _Sessions:
        async def do_session_restore(self, session_id: str, *, allow_while_loading: bool = False) -> None:
            calls.append(("restore", session_id, allow_while_loading))

    app = object.__new__(ChrysApp)
    app._engine = _Engine()
    app._state_store = _Store()
    app._startup_session_id = "canonicalsess"
    app._deferred_settings_warnings = []
    screen = _attach_startup_facade(
        SimpleNamespace(
            _sessions=_Sessions(),
            _set_agent_loading=lambda value: calls.append(("loading", value)),
        )
    )
    profile = object()

    await app._start_engine(profile, screen)  # type: ignore[arg-type]

    assert calls == [
        ("prepare", profile),
        ("loading", True),
        ("meta", "canonicalsess"),
        ("restore", "canonical-session-id", True),
    ]
    assert app._startup_session_id == ""


@pytest.mark.asyncio
async def test_start_engine_falls_back_when_restore_emits_no_success() -> None:
    calls: list[tuple[str, object]] = []

    class _Engine:
        async def prepare(self, profile: object) -> None:
            calls.append(("prepare", profile))

        async def reset_after_failed_startup_restore(self) -> None:
            calls.append(("reset_restore", None))

        async def start(self, profile: object) -> None:
            calls.append(("start", profile))

    class _Store:
        async def load_session_meta(self, session_id: str) -> object:
            calls.append(("meta", session_id))
            return SimpleNamespace(session_id=session_id)

    class _Screen:
        _agent_loading = True

        def set_startup_agent_loading(self, value: bool) -> None:
            self._agent_loading = value
            calls.append(("loading", value))

        def is_startup_agent_loading(self) -> bool:
            return self._agent_loading

        async def dismiss_startup_load_dialog_before_restore(self) -> None:
            return

        async def restore_startup_session(self, session_id: str) -> bool:
            calls.append(("restore", session_id))
            return False

        def cancel_startup_session_restore(self) -> None:
            calls.append(("cancel", None))

        def notify(self, message: str, *, title: str, severity: str, **_kwargs: object) -> None:
            calls.append(("warning", message, title, severity))

    app = object.__new__(ChrysApp)
    app._engine = _Engine()
    app._locale_controller = tui_i18n.LocaleController(Settings(locale="en"))
    app._state_store = _Store()
    app._startup_session_id = "session-1"
    app._deferred_settings_warnings = []
    profile = object()

    await app._start_engine(profile, _Screen())  # type: ignore[arg-type]

    assert calls == [
        ("prepare", profile),
        ("loading", True),
        ("meta", "session-1"),
        ("restore", "session-1"),
        ("cancel", None),
        (
            "warning",
            f"Could not restore session {session_short_id('session-1')}; started a new session instead.",
            "Session",
            "warning",
        ),
        ("reset_restore", None),
        ("start", profile),
    ]


@pytest.mark.asyncio
async def test_main_screen_startup_restore_requires_matching_success_event() -> None:
    from chrys.app.tui.screens.main.screen import MainScreen
    from chrys.foundation.events.types import SessionRestored

    bus = EventBus()

    class _Sessions:
        def __init__(self, *, publish_success: bool) -> None:
            self.publish_success = publish_success

        async def do_session_restore(self, session_id: str, *, allow_while_loading: bool = False) -> None:
            assert allow_while_loading is True
            if self.publish_success:
                await bus.publish(SessionRestored(session_id=session_id))

    screen = object.__new__(MainScreen)
    screen._bus = bus
    screen._sessions = _Sessions(publish_success=False)
    assert await MainScreen.restore_startup_session(screen, "session-1") is False

    screen._sessions = _Sessions(publish_success=True)
    assert await MainScreen.restore_startup_session(screen, "session-1") is True


def test_main_screen_cancel_startup_restore_clears_restoring_state() -> None:
    from chrys.app.tui.screens.main.screen import MainScreen

    calls: list[tuple[str, object]] = []
    screen = object.__new__(MainScreen)
    screen._events = SimpleNamespace(cancel_agent_load=lambda: calls.append(("cancel", None)))
    screen._set_restoring_session = lambda value: calls.append(("restoring", value))

    MainScreen.cancel_startup_session_restore(screen)

    assert calls == [("cancel", None), ("restoring", False)]


@pytest.mark.asyncio
async def test_start_engine_dismisses_startup_modal_before_startup_session_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.app.tui import app as tui_app
    from chrys.app.tui.screens.dialogs.agent_load import AgentLoadDialog

    calls: list[tuple[str, object]] = []

    async def fake_sleep(delay: float) -> None:
        calls.append(("sleep", delay))

    class _Engine:
        async def prepare(self, profile: object) -> None:
            calls.append(("prepare", profile))

        async def reset_after_failed_startup_restore(self) -> None:
            calls.append(("reset_restore", None))

        async def start(self, profile: object) -> None:
            calls.append(("start", profile))

    class _Store:
        async def load_session_meta(self, session_id: str) -> object:
            calls.append(("meta", session_id))
            return SimpleNamespace(session_id=session_id)

    class _Sessions:
        async def do_session_restore(self, session_id: str, *, allow_while_loading: bool = False) -> None:
            calls.append(("restore", session_id, allow_while_loading))

    monkeypatch.setattr(tui_app.asyncio, "sleep", fake_sleep)
    startup_dialog = AgentLoadDialog()
    monkeypatch.setattr(startup_dialog, "dismiss", lambda result=None: calls.append(("dismiss", result)))

    app = object.__new__(ChrysApp)
    app._engine = _Engine()
    app._state_store = _Store()
    app._startup_session_id = "session-1"
    app._deferred_settings_warnings = []
    screen = _attach_startup_facade(
        SimpleNamespace(
            _events=SimpleNamespace(_agent_load_dialog=None),
            app=SimpleNamespace(screen_stack=[startup_dialog]),
            _sessions=_Sessions(),
            _set_agent_loading=lambda value: calls.append(("loading", value)),
        )
    )

    async def dismiss_startup_load_dialog_before_restore() -> None:
        startup_dialog.dismiss(None)
        await tui_app.asyncio.sleep(0)

    screen.dismiss_startup_load_dialog_before_restore = dismiss_startup_load_dialog_before_restore  # type: ignore[attr-defined]
    profile = object()

    await app._start_engine(profile, screen)  # type: ignore[arg-type]

    assert calls == [
        ("prepare", profile),
        ("loading", True),
        ("meta", "session-1"),
        ("dismiss", None),
        ("sleep", 0),
        ("restore", "session-1", True),
    ]


@pytest.mark.asyncio
async def test_start_engine_cancels_active_startup_modal_before_startup_session_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.app.tui import app as tui_app

    calls: list[tuple[str, object]] = []

    async def fake_sleep(delay: float) -> None:
        calls.append(("sleep", delay))

    class _Engine:
        async def prepare(self, profile: object) -> None:
            calls.append(("prepare", profile))

        async def start(self, profile: object) -> None:
            calls.append(("start", profile))

    class _Store:
        async def load_session_meta(self, session_id: str) -> object:
            calls.append(("meta", session_id))
            return SimpleNamespace(session_id=session_id)

    class _Sessions:
        async def do_session_restore(self, session_id: str, *, allow_while_loading: bool = False) -> None:
            calls.append(("restore", session_id, allow_while_loading))

    monkeypatch.setattr(tui_app.asyncio, "sleep", fake_sleep)

    app = object.__new__(ChrysApp)
    app._engine = _Engine()
    app._state_store = _Store()
    app._startup_session_id = "session-1"
    app._deferred_settings_warnings = []
    screen = _attach_startup_facade(
        SimpleNamespace(
            _events=SimpleNamespace(
                _agent_load_dialog=object(),
                cancel_agent_load=lambda: calls.append(("cancel", None)),
            ),
            app=SimpleNamespace(screen_stack=[]),
            _sessions=_Sessions(),
            _set_agent_loading=lambda value: calls.append(("loading", value)),
        )
    )

    async def dismiss_startup_load_dialog_before_restore() -> None:
        screen._events.cancel_agent_load()
        await tui_app.asyncio.sleep(0)

    screen.dismiss_startup_load_dialog_before_restore = dismiss_startup_load_dialog_before_restore  # type: ignore[attr-defined]
    profile = object()

    await app._start_engine(profile, screen)  # type: ignore[arg-type]

    assert calls == [
        ("prepare", profile),
        ("loading", True),
        ("meta", "session-1"),
        ("cancel", None),
        ("sleep", 0),
        ("restore", "session-1", True),
    ]


@pytest.mark.asyncio
async def test_start_engine_missing_startup_session_warns_and_continues() -> None:
    calls: list[tuple[str, object]] = []
    flashes: list[tuple[str, bool]] = []
    notifications: list[tuple[str, str, str]] = []

    class _Engine:
        async def prepare(self, profile: object) -> None:
            calls.append(("prepare", profile))

        async def reset_after_failed_startup_restore(self) -> None:
            calls.append(("reset_restore", None))

        async def start(self, profile: object) -> None:
            calls.append(("start", profile))

    class _Store:
        async def load_session_meta(self, session_id: str) -> object | None:
            calls.append(("meta", session_id))
            return None

    class _StatusBar:
        def flash(self, message: MessageRef | str, *, error: bool = False) -> None:
            flashes.append((format_message(message) if isinstance(message, MessageRef) else message, error))

    class _Screen:
        _agent_loading = False

        def _set_agent_loading(self, value: bool) -> None:
            calls.append(("loading", value))

        def query_one(self, cls: type) -> _StatusBar:
            if cls.__name__ != "StatusBar":
                raise AssertionError(cls)
            return _StatusBar()

        def notify(self, message: str, *, title: str, severity: str, **_kwargs: object) -> None:
            notifications.append((title, severity, message))

    app = object.__new__(ChrysApp)
    app._engine = _Engine()
    app._locale_controller = tui_i18n.LocaleController(Settings(locale="en"))
    app._state_store = _Store()
    app._startup_session_id = "missing-session"
    app._deferred_settings_warnings = []
    profile = object()

    await app._start_engine(profile, _attach_startup_facade(_Screen()))  # type: ignore[arg-type]

    assert calls == [
        ("prepare", profile),
        ("loading", True),
        ("meta", "missing-session"),
        ("loading", False),
        ("reset_restore", None),
        ("start", profile),
    ]
    warning = f"Session {session_short_id('missing-session')} was not found."
    assert flashes == [(warning, True)]
    assert notifications == [("Session", "warning", warning)]
    assert app._startup_session_id == ""


@pytest.mark.asyncio
async def test_start_engine_startup_session_restore_failure_warns_and_continues() -> None:
    calls: list[tuple[str, object]] = []
    flashes: list[tuple[str, bool]] = []
    notifications: list[tuple[str, str, str]] = []

    class _Engine:
        async def prepare(self, profile: object) -> None:
            calls.append(("prepare", profile))

        async def reset_after_failed_startup_restore(self) -> None:
            calls.append(("reset_restore", None))

        async def start(self, profile: object) -> None:
            calls.append(("start", profile))

    class _Store:
        async def load_session_meta(self, session_id: str) -> object:
            calls.append(("meta", session_id))
            return SimpleNamespace(session_id=session_id)

    class _Sessions:
        async def do_session_restore(self, session_id: str, *, allow_while_loading: bool = False) -> None:
            calls.append(("restore", session_id, allow_while_loading))
            raise RuntimeError("restore exploded")

    class _StatusBar:
        def flash(self, message: MessageRef | str, *, error: bool = False) -> None:
            flashes.append((format_message(message) if isinstance(message, MessageRef) else message, error))

    class _Screen:
        _agent_loading = False
        _sessions = _Sessions()

        def _set_agent_loading(self, value: bool) -> None:
            calls.append(("loading", value))

        def query_one(self, cls: type) -> _StatusBar:
            if cls.__name__ != "StatusBar":
                raise AssertionError(cls)
            return _StatusBar()

        def notify(self, message: str, *, title: str, severity: str, **_kwargs: object) -> None:
            notifications.append((title, severity, message))

    app = object.__new__(ChrysApp)
    app._engine = _Engine()
    app._locale_controller = tui_i18n.LocaleController(Settings(locale="en"))
    app._state_store = _Store()
    app._startup_session_id = "session-1"
    app._deferred_settings_warnings = []
    profile = object()

    await app._start_engine(profile, _attach_startup_facade(_Screen()))  # type: ignore[arg-type]

    assert calls == [
        ("prepare", profile),
        ("loading", True),
        ("meta", "session-1"),
        ("restore", "session-1", True),
        ("loading", False),
        ("reset_restore", None),
        ("start", profile),
    ]
    warning = f"Could not restore session {session_short_id('session-1')}: restore exploded"
    assert flashes == [(warning, True)]
    assert notifications == [("Session", "warning", warning)]
    assert app._startup_session_id == ""


@pytest.mark.asyncio
async def test_start_engine_drops_deferred_settings_warnings_after_a_successful_restore() -> None:
    """The restore republishes its own settings load's warnings; the bootstrap ones are stale."""
    calls: list[str] = []

    class _Engine:
        async def prepare(self, profile: object) -> None:
            calls.append("prepare")

        async def start(self, profile: object) -> None:
            calls.append("start")

    class _Store:
        async def load_session_meta(self, session_id: str) -> object:
            return SimpleNamespace(session_id=session_id)

    class _Sessions:
        async def do_session_restore(self, session_id: str, *, allow_while_loading: bool = False) -> None:
            calls.append("restore")

    bus = EventBus()
    published: list[Warning] = []

    async def _record(event: Warning) -> None:
        published.append(event)

    await bus.subscribe(Warning, _record)

    app = object.__new__(ChrysApp)
    app._engine = _Engine()
    app._state_store = _Store()
    app._startup_session_id = "session-1"
    app._bus = bus
    app._deferred_settings_warnings = [Warning(code="settings_layer_warning", message="From the launch cwd.")]
    screen = _attach_startup_facade(
        SimpleNamespace(
            _sessions=_Sessions(),
            _set_agent_loading=lambda value: None,
        )
    )

    await app._start_engine(object(), screen)  # type: ignore[arg-type]

    assert calls == ["prepare", "restore"]
    assert published == []
    assert app._deferred_settings_warnings == []


@pytest.mark.asyncio
async def test_start_engine_flushes_deferred_settings_warnings_when_falling_back() -> None:
    """A failed restore leaves the bootstrap settings in force, so their warnings are due."""
    calls: list[object] = []

    class _Engine:
        async def prepare(self, profile: object) -> None:
            calls.append("prepare")

        async def reset_after_failed_startup_restore(self) -> None:
            calls.append("reset_restore")

        async def start(self, profile: object) -> None:
            calls.append("start")

    class _Store:
        async def load_session_meta(self, session_id: str) -> object:
            return SimpleNamespace(session_id=session_id)

    class _StatusBar:
        def flash(self, message: MessageRef | str, *, error: bool = False) -> None:
            return

    class _Screen:
        _agent_loading = True

        def set_startup_agent_loading(self, value: bool) -> None:
            self._agent_loading = value

        def is_startup_agent_loading(self) -> bool:
            return self._agent_loading

        async def dismiss_startup_load_dialog_before_restore(self) -> None:
            return

        async def restore_startup_session(self, session_id: str) -> bool:
            return False

        def cancel_startup_session_restore(self) -> None:
            return

        def query_one(self, cls: type) -> _StatusBar:
            return _StatusBar()

        def notify(self, message: str, *, title: str, severity: str, **_kwargs: object) -> None:
            return

    bus = EventBus()

    async def _record(event: Warning) -> None:
        calls.append(("flushed", event.code))

    await bus.subscribe(Warning, _record)

    app = object.__new__(ChrysApp)
    app._engine = _Engine()
    app._locale_controller = tui_i18n.LocaleController(Settings(locale="en"))
    app._state_store = _Store()
    app._startup_session_id = "session-1"
    app._bus = bus
    app._deferred_settings_warnings = [Warning(code="settings_layer_warning", message="From the launch cwd.")]

    await app._start_engine(object(), _Screen())  # type: ignore[arg-type]

    assert calls == ["prepare", "reset_restore", ("flushed", "settings_layer_warning"), "start"]
    assert app._deferred_settings_warnings == []


def test_notification_focus_handlers_use_chrys_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Chrys notification state must not collide with Textual's internal notification manager."""
    from textual import events

    class _Engine:
        async def shutdown(self) -> None:
            return

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(theme="chrys"),
        state_store=JsonFileStateStore(tmp_path),
    )

    assert isinstance(app.notification_service, NotificationService)

    app._on_app_blur(events.AppBlur())
    assert app.notification_service._focus_known is True
    assert app.notification_service._focused is False
    app._on_app_focus(events.AppFocus())
    assert app.notification_service._focused is True

    assert isinstance(app.notification_service, NotificationService)


def test_chrys_dark_is_registered_and_selectable(tmp_path: Path) -> None:
    class _Engine:
        async def shutdown(self) -> None:
            return

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(theme=CHRYS_DARK_THEME_NAME),
        state_store=JsonFileStateStore(tmp_path),
    )

    assert CHRYS_DARK_THEME_NAME in app.available_themes
    assert app.theme == CHRYS_DARK_THEME_NAME
    assert "-chrys" in app.classes


def test_chrys_theme_hex_colors_are_256_color_palette_entries() -> None:
    """Chrys/chrys-ansi theme hex literals should stay inside the 256-color palette.

    DiffView's per-row palettes are intentionally excluded: those colors
    are visually tuned truecolor values. test_file_edit_renderer covers the
    weaker terminal-compatibility invariant that dark diff backgrounds do
    not downgrade to xterm black.
    """
    palette = {f"#{red:02X}{green:02X}{blue:02X}" for red, green, blue in EIGHT_BIT_PALETTE}
    hex_color = re.compile(r"#[0-9A-Fa-f]{6}(?![0-9A-Fa-f])")

    failures: list[str] = []
    for path in (_CHRYS_THEME, _CHRYS_CSS):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in hex_color.finditer(line):
                color = match.group(0).upper()
                if color not in palette:
                    failures.append(f"{path.name}:{line_no} {color}")

    assert failures == []


def test_switching_from_ansi_theme_back_to_chrys_restores_truecolor(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """ANSI theme previews should not leave a global ANSI override behind."""

    class _Engine:
        async def shutdown(self) -> None:
            return

    persisted: list[str] = []
    monkeypatch.setattr("chrys.app.tui.app.persist_theme", persisted.append)

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(theme="chrys"),
        state_store=JsonFileStateStore(tmp_path),
    )

    assert app.ansi_color is None
    assert app.native_ansi_color is False

    app.theme = "ansi-dark"

    assert app.ansi_color is None
    assert app.native_ansi_color is True

    app.theme = "chrys"

    assert app.ansi_color is None
    assert app.native_ansi_color is False
    assert "-chrys" in app.classes
    assert "-chrys-ansi" not in app.classes
    assert persisted == ["ansi-dark", "chrys"]


def test_a_saved_theme_that_no_longer_resolves_is_not_written_back(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Falling back is a decision about what to draw, not about what to save.

    A theme renamed between releases, a typo, a value some other writer
    damaged: persisting the fallback would delete the name the user chose and
    the next start would have nothing left to recover from. The user keeps the
    setting and loses only this session's colours — and a real switch
    afterwards still persists.
    """

    class _Engine:
        async def shutdown(self) -> None:
            return

    persisted: list[str] = []
    monkeypatch.setattr("chrys.app.tui.app.persist_theme", persisted.append)

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(theme="retired-theme-name"),
        state_store=JsonFileStateStore(tmp_path),
    )

    assert app.theme == DEFAULT_THEME
    assert persisted == []

    app.theme = "chrys-dark"

    assert persisted == ["chrys-dark"]


def test_apply_theme_setting_persists_a_switch_exactly_once(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The Settings panel routes theme picks through one write point: the
    reactive assignment persists, so the setter itself must not."""

    class _Engine:
        async def shutdown(self) -> None:
            return

    persisted: list[str] = []
    monkeypatch.setattr("chrys.app.tui.app.persist_theme", persisted.append)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(theme="chrys"),
        state_store=JsonFileStateStore(tmp_path),
    )

    app.apply_theme_setting("chrys-dark")
    assert app.theme == "chrys-dark"
    assert persisted == ["chrys-dark"]
    assert app.settings_handle.settings.theme == "chrys-dark"

    # Re-applying the showing-and-saved theme is a no-op.
    app.apply_theme_setting("chrys-dark")
    assert persisted == ["chrys-dark"]

    # A name that is not a registered theme is ignored, not persisted.
    app.apply_theme_setting("no-such-theme")
    assert app.theme == "chrys-dark"
    assert persisted == ["chrys-dark"]


def test_apply_theme_setting_records_the_showing_fallback_when_the_user_picks_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A saved theme that no longer resolves fell back at startup; picking the
    fallback in the panel is a real choice, and the reactive will not fire
    for an equal value — so the setter persists and overlays it directly."""

    class _Engine:
        async def shutdown(self) -> None:
            return

    persisted: list[str] = []
    monkeypatch.setattr("chrys.app.tui.app.persist_theme", persisted.append)
    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(theme="retired-theme-name"),
        state_store=JsonFileStateStore(tmp_path),
    )
    assert app.theme == DEFAULT_THEME
    assert persisted == []

    app.apply_theme_setting(DEFAULT_THEME)

    assert persisted == [DEFAULT_THEME]
    assert app.settings_handle.settings.theme == DEFAULT_THEME


@pytest.mark.asyncio
async def test_ctrl_q_over_the_settings_dialog_commits_a_focused_edit_before_exiting(tmp_path: Path) -> None:
    """Ctrl+Q is an app-level priority binding, so it fires with a modal on
    top; a value being typed in the Settings panel must still land."""
    from textual.widgets import Input

    from chrys.app.tui.screens.settings import SettingsDialog
    from tests.app.tui.screens.settings.support import StubPorts

    class _Engine:
        session_generation = 0

        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )
    ports = StubPorts()

    async with app.run_test() as pilot:
        await pilot.pause()
        dialog = SettingsDialog(ports, initial_tab="sessions")
        await app.push_screen(dialog)
        await pilot.pause()
        row = next(row for row in dialog.rows() if row.spec.key == "rollback.snapshots_keep")
        field = row.query_one(Input)
        field.focus()
        await pilot.pause()
        field.value = "7"
        await pilot.press("ctrl+q")
        await pilot.pause()

    assert app.return_code is not None
    assert ports.persisted == [{"rollback.snapshots_keep": 7}]
    assert ports.closed == 0


def test_chrys_ansi_uses_theme_native_ansi_flag(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The transparent Chrys theme should opt in via Theme.ansi, not App.ansi_color."""

    class _Engine:
        async def shutdown(self) -> None:
            return

    monkeypatch.setattr("chrys.app.tui.app.persist_theme", lambda _theme: None)

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(theme="chrys"),
        state_store=JsonFileStateStore(tmp_path),
    )

    app.theme = "chrys-ansi"

    variables = app.get_css_variables()
    assert app.current_theme.ansi is True
    assert app.ansi_color is None
    assert app.native_ansi_color is True
    assert variables["ansi-background"] == "ansi_black"
    assert variables["ansi-foreground"] == "ansi_white"
    assert variables["button-color-foreground"] == "ansi_black"
    assert variables["text-muted"] == "ansi_white 40%"
    assert variables["markdown-h1-color"] == "#AF87FF"
    assert variables["markdown-h2-color"] == "#AF87FF"
    assert variables["markdown-h3-color"] == "#AF87FF"
    assert variables["markdown-h4-color"] == "#EEEEEE"
    assert variables["markdown-h5-color"] == "#EEEEEE"
    assert variables["markdown-h6-color"] == "#9E9E9E"
    assert variables["input-selection-background"] == "#5F5F87"
    assert variables["screen-selection-background"] == "#5F5F87"
    assert "-chrys" in app.classes
    assert "-chrys-ansi" in app.classes


@pytest.mark.asyncio
async def test_chrys_ansi_theme_defines_textual_ansi_css_variables() -> None:
    """Textual's built-in ANSI-mode CSS needs ansi-background/foreground."""

    class ThemeApp(App):
        def __init__(self) -> None:
            super().__init__()
            self.register_theme(CHRYS_ANSI_THEME)
            self.theme = "chrys-ansi"

        def compose(self) -> ComposeResult:
            yield Static("ok")

    app = ThemeApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.current_theme.name == "chrys-ansi"


@pytest.mark.asyncio
async def test_chrys_ansi_rendered_colors_follow_chrys_without_ansi_fallbacks() -> None:
    """chrys-ansi should stay close to Chrys colors, not Textual ANSI fallbacks."""

    class ThemeParityApp(App):
        CSS_PATH = str(_CHRYS_CSS)

        def __init__(self, theme_name: str) -> None:
            super().__init__()
            self.register_theme(CHRYS_THEME)
            self.register_theme(CHRYS_ANSI_THEME)
            self.theme = theme_name

        def compose(self) -> ComposeResult:
            yield Button("focus target", id="focus-target")
            yield Input("selected text", id="selection-input")
            yield VirtualizedMarkdown(
                "# H1\n\n## H2\n\n### H3\n\n#### H4\n\n##### H5\n\n###### H6",
                id="headings",
            )

        def on_mount(self) -> None:
            self.set_class(True, "-chrys")
            self.set_class(self.theme == "chrys-ansi", "-chrys-ansi")

    def color_rgb(color: Color | None) -> tuple[int, int, int]:
        assert color is not None
        triplet = color.triplet
        assert triplet is not None
        return triplet.red, triplet.green, triplet.blue

    async def collect(theme_name: str) -> dict[str, tuple[int, int, int]]:
        app = ThemeParityApp(theme_name)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#focus-target", Button).focus()
            await pilot.pause()

            markdown = app.query_one("#headings", VirtualizedMarkdown)
            values = {
                "screen-selection": color_rgb(app.screen.get_component_rich_style("screen--selection").bgcolor),
                "input-selection": color_rgb(
                    app.query_one("#selection-input", Input).get_visual_style("input--selection").rich_style.bgcolor
                ),
            }
            for level in range(1, 7):
                values[f"h{level}"] = color_rgb(
                    markdown.get_visual_style(f"virtualized-markdown--h{level}").rich_style.color
                )
            return values

    ansi_values = await collect("chrys-ansi")
    chrys_values = await collect("chrys")

    for key in ("h1", "h2", "h3", "h4", "h5"):
        assert ansi_values[key] == chrys_values[key]

    # These colors are alpha-derived in truecolor chrys but must be flattened
    # to nearby xterm-256 entries in transparent chrys-ansi.
    assert ansi_values["h6"] == (158, 158, 158)
    assert ansi_values["input-selection"] == (95, 95, 135)
    assert ansi_values["screen-selection"] == (95, 95, 135)
    for key in ("h6", "input-selection", "screen-selection"):
        assert max(abs(ansi - chrys) for ansi, chrys in zip(ansi_values[key], chrys_values[key], strict=True)) <= 24


@pytest.mark.asyncio
async def test_chrys_app_runs_with_chrys_ansi_css(tmp_path) -> None:
    """The full Chrys stylesheet should parse with chrys-ansi active."""
    from chrys.app.tui.screens.main.screen import MainScreen
    from chrys.app.tui.widgets.chrome.input_bar import InputBar
    from chrys.app.tui.widgets.chrome.status_bar import StatusBar
    from chrys.app.tui.widgets.sidebar.toc import ConversationToc

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(theme="chrys-ansi"),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.current_theme.name == "chrys-ansi"
        assert app.native_ansi_color is True
        screen = next(screen for screen in app.screen_stack if isinstance(screen, MainScreen))
        chat_input = screen.query_one(InputBar).query_one(TextArea)
        placeholder_style = chat_input.get_visual_style("text-area--placeholder").rich_style
        assert placeholder_style.color is not None
        assert placeholder_style.color.number == 8
        assert chat_input.rich_style.bgcolor is not None
        assert chat_input.rich_style.bgcolor.triplet is not None
        chat_input_background = chat_input.rich_style.bgcolor.triplet
        assert (chat_input_background.red, chat_input_background.green, chat_input_background.blue) == (48, 48, 48)
        selection_style = screen.get_component_rich_style("screen--selection")
        assert selection_style.bgcolor is not None
        assert selection_style.bgcolor.number is None
        assert selection_style.bgcolor.triplet is not None
        triplet = selection_style.bgcolor.triplet
        assert (triplet.red, triplet.green, triplet.blue) == (95, 95, 135)
        toc_empty_style = screen.query_one(ConversationToc).query_one(".toc-empty").rich_style
        assert toc_empty_style.color is not None
        assert toc_empty_style.color.number == 8
        input_bar = screen.query_one(InputBar)
        status_bar = screen.query_one(StatusBar)
        status_bar.set_profile("Code Agent")
        await pilot.pause()
        agent_label_style = status_bar.query_one("#agent-label", Static).rich_style
        assert agent_label_style.bgcolor is not None
        assert agent_label_style.bgcolor.number == 0
        assert agent_label_style.color is not None
        assert agent_label_style.color.number == 7
        profile_tag_style = status_bar.query_one("#profile-tag", Static).rich_style
        assert profile_tag_style.bgcolor is not None
        assert profile_tag_style.bgcolor.triplet is not None
        profile_tag_triplet = profile_tag_style.bgcolor.triplet
        assert (profile_tag_triplet.red, profile_tag_triplet.green, profile_tag_triplet.blue) == (76, 40, 64)
        send_button = input_bar.query_one("#send-btn", Button)
        assert send_button.disabled is True
        assert send_button.styles.text_opacity == pytest.approx(0.9)
        new_button = input_bar.query_one("#new-btn", Button)
        assert new_button.rich_style.bgcolor is not None
        assert new_button.rich_style.bgcolor.downgrade(ColorSystem.EIGHT_BIT).number == 141


@pytest.mark.asyncio
async def test_main_screen_chat_scroll_keys_work_with_input_and_chat_focus(tmp_path: Path) -> None:
    """The transcript should keep ScrollView keyboard scrolling even with chat-input focus."""
    from chrys.app.tui.screens.main.screen import MainScreen
    from chrys.app.tui.widgets.chat.panel import ChatPanel
    from chrys.app.tui.widgets.chrome.input_bar import InputBar

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
    )

    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        screen = next(screen for screen in app.screen_stack if isinstance(screen, MainScreen))
        panel = screen.query_one(ChatPanel)
        input_bar = screen.query_one(InputBar)

        for index in range(30):
            await panel.add_user_message(f"chat line {index}\n" * 3)
        await pilot.pause()
        assert panel.max_scroll_y > 0

        panel.scroll_end(animate=False)
        input_bar.focus_input()
        await pilot.pause()
        assert input_bar.query_one(TextArea).has_focus

        bottom = panel.scroll_y
        await pilot.press("pageup")
        await pilot.pause()
        assert panel.scroll_y < bottom

        page_up_position = panel.scroll_y
        await pilot.press("pagedown")
        await pilot.pause()
        assert panel.scroll_y > page_up_position

        panel.scroll_end(animate=False)
        panel.focus()
        await pilot.pause()
        assert panel.has_focus

        bottom = panel.scroll_y
        await pilot.press("up")
        await pilot.wait_for_scheduled_animations()
        assert panel.scroll_y < bottom

        up_position = panel.scroll_y
        await pilot.press("down")
        await pilot.wait_for_scheduled_animations()
        assert panel.scroll_y > up_position


@pytest.mark.asyncio
async def test_main_screen_chat_page_keys_disabled_for_trajectory_dashboard() -> None:
    from chrys.app.tui.screens.main.screen import MainScreen

    screen = MainScreen(EventBus(), engine_provider=None)
    screen._dashboard_visible = lambda: True  # type: ignore[method-assign]

    assert screen.check_action("chat_page_up", ()) is False
    assert screen.check_action("chat_page_down", ()) is False
    assert screen.check_action("chat_scroll_bottom", ()) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("target_theme", ["textual-dark", "ansi-dark", "ansi-light"])
async def test_theme_switch_to_non_chrys_does_not_crash(tmp_path, target_theme: str) -> None:
    """Switching to reachable non-chrys themes must keep chrys.tcss parseable."""
    from chrys.app.tui.screens.main.screen import MainScreen

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(theme="chrys"),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.current_theme.name == "chrys"
        # Sanity: the screen is mounted and CSS parsed for chrys.
        screen = next(s for s in app.screen_stack if isinstance(s, MainScreen))
        assert screen is not None

        # Switch to a builtin non-chrys theme — CSS re-parse must succeed.
        app.theme = target_theme
        await pilot.pause()
        assert app.current_theme.name == target_theme

        # Switch back to chrys; round trip.
        app.theme = "chrys"
        await pilot.pause()
        assert app.current_theme.name == "chrys"


@pytest.mark.asyncio
async def test_chrys_ansi_config_inner_borders_are_lighter(
    tmp_path, monkeypatch: pytest.MonkeyPatch, fake_platform
) -> None:
    """Config screen inner borders should not disappear in chrys-ansi."""
    from chrys.app.tui.screens.agents.config import AgentsConfigScreen
    from chrys.app.tui.screens.models.screen import ModelConfigScreen
    from chrys.service.profiles.agents.registry import AgentProfileRegistry
    from chrys.service.profiles.models.registry import ModelProfileRegistry
    from chrys.service.profiles.models.schema import ModelProfile

    config_dir = tmp_path / "config"
    model_dir = config_dir / "models"
    existing_model_files = set(model_dir.glob("*.yaml")) if model_dir.exists() else set()
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform(config_dir=config_dir))

    def fail_save_profile(_profile: ModelProfile) -> None:
        raise AssertionError("styling test must not write model profiles")

    monkeypatch.setattr("chrys.service.profiles.models.serializer.save_profile", fail_save_profile)

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(theme="chrys-ansi"),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
    )
    agent_registry = AgentProfileRegistry()
    agent_registry.load_builtins()
    model_registry = ModelProfileRegistry()
    model_profile = ModelProfile(id="model-a", name="Model A", model_id="gpt-test")
    model_registry.register(model_profile)

    try:
        async with app.run_test() as pilot:
            await pilot.pause()

            agents_screen = AgentsConfigScreen(agent_registry, current_profile="Code")
            await app.push_screen(agents_screen)
            await pilot.pause()
            _, agent_sidebar_color = agents_screen.query_one("#ac-sidebar").styles.border_right
            assert agent_sidebar_color.a == pytest.approx(0.25)
            active_agent_tab = next(tab for tab in agents_screen.query(Tab) if tab.has_class("-active"))
            assert active_agent_tab.rich_style.bold is True
            unselected_agent_tab = next(tab for tab in agents_screen.query(Tab) if not tab.has_class("-active"))
            assert unselected_agent_tab.rich_style.dim is not True
            agent_unchecked_checkbox_style = agents_screen.query_one(
                "#bc-sub-agent-only", Checkbox
            ).get_component_rich_style("toggle--button")
            agent_checked_checkbox_style = agents_screen.query_one(
                "#bc-model-use-active", Checkbox
            ).get_component_rich_style("toggle--button")
            agents_screen.query_one("#ac-tabs", TabbedContent).active = "tools"
            await pilot.pause()
            agent_tool_checkbox_style = agents_screen.query_one(
                "#tc-cat-filesystem-read", Checkbox
            ).get_component_rich_style("toggle--button")
            assert agent_tool_checkbox_style == agent_checked_checkbox_style

            await app.pop_screen()
            await pilot.pause()

            models_screen = ModelConfigScreen(model_registry, global_default_profile_id=model_profile.id)
            await app.push_screen(models_screen)
            await pilot.pause()
            _, model_sidebar_color = models_screen.query_one("#mc-sidebar").styles.border_right
            option_section_border = models_screen.query_one("#mc-model-options").styles.border
            model_stream = models_screen.query_one("#mc-stream", Checkbox)
            model_unchecked_checkbox_style = model_stream.get_component_rich_style("toggle--button")
            model_stream.value = True
            await pilot.pause()
            model_checked_checkbox_style = model_stream.get_component_rich_style("toggle--button")
            assert model_sidebar_color.a == pytest.approx(0.25)
            assert option_section_border.top[1].a == pytest.approx(0.25)
            assert model_unchecked_checkbox_style == agent_unchecked_checkbox_style
            assert model_checked_checkbox_style == agent_checked_checkbox_style
    finally:
        if model_dir.exists():
            for path in model_dir.glob("*.yaml"):
                if path not in existing_model_files:
                    path.unlink()


@pytest.mark.asyncio
async def test_chrys_active_config_tab_is_bold(tmp_path) -> None:
    """The base chrys theme should keep selected tabs visually emphasized."""
    from chrys.app.tui.screens.agents.config import AgentsConfigScreen
    from chrys.service.profiles.agents.registry import AgentProfileRegistry

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(theme="chrys"),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
    )
    agent_registry = AgentProfileRegistry()
    agent_registry.load_builtins()

    async with app.run_test() as pilot:
        await pilot.pause()

        agents_screen = AgentsConfigScreen(agent_registry, current_profile="Code")
        await app.push_screen(agents_screen)
        await pilot.pause()

        active_agent_tab = next(tab for tab in agents_screen.query(Tab) if tab.has_class("-active"))
        assert active_agent_tab.rich_style.bold is True


@pytest.mark.asyncio
async def test_startup_enter_spam_does_not_send_or_show_input_loading(tmp_path) -> None:
    """Rapid Enter during startup should leave the empty composer visually idle."""
    import asyncio

    from chrys.app.tui.screens.main.screen import MainScreen
    from chrys.app.tui.widgets.chrome.input_bar import InputBar
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import AgentLoadStarted, SessionReady, UserMessage
    from chrys.service.profiles.agents.schema import AgentProfile
    from chrys.service.profiles.models.registry import ModelProfileRegistry
    from chrys.service.profiles.models.schema import ModelProfile
    from chrys.service.state.store import JsonFileStateStore

    profile = AgentProfile(name="Code", display_name="Code", description="Test profile")

    class _Registry:
        def list_profiles(self) -> list[AgentProfile]:
            return [profile]

        def load_all(self) -> None:
            return

        def get(self, name: str) -> AgentProfile | None:
            return profile if name == profile.name else None

    class _Engine:
        def __init__(self, bus: EventBus) -> None:
            self._bus = bus
            self.session_generation = 0
            self.started = asyncio.Event()
            self.load_started = asyncio.Event()
            self.release = asyncio.Event()
            self.ready = asyncio.Event()

        async def start(self, _profile: AgentProfile) -> None:
            self.started.set()
            await self._bus.publish(
                AgentLoadStarted(operation="startup", to_profile=profile.name, to_display_name=profile.display_name)
            )
            self.load_started.set()
            await self.release.wait()
            await self._bus.publish(
                SessionReady(agent_profile=profile.name, display_name=profile.display_name, session_id="test-session")
            )
            self.ready.set()

        async def shutdown(self) -> None:
            self.release.set()

    bus = EventBus()
    messages: list[str] = []

    async def _record_message(event: UserMessage) -> None:
        messages.append(event.text)

    await bus.subscribe(UserMessage, _record_message)
    engine = _Engine(bus)
    # A selectable model profile keeps the unconfigured-model send guard out
    # of this test's way — its subject is composer state during startup.
    model_registry = ModelProfileRegistry()
    model_registry.register(ModelProfile(id="model-profile", name="Configured Model", model_id="test-model"))
    app = ChrysApp(
        bus,
        engine,  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_Registry(),  # type: ignore[arg-type]
        model_registry=model_registry,
    )

    async with app.run_test() as pilot:
        await asyncio.wait_for(engine.started.wait(), timeout=1)
        await asyncio.wait_for(engine.load_started.wait(), timeout=1)
        await pilot.pause()
        main_screen = next(screen for screen in pilot.app.screen_stack if isinstance(screen, MainScreen))
        ib = main_screen.query_one(InputBar)
        send_btn = ib.query_one("#send-btn", Button)
        text_area = ib.query_one("#chat-input", TextArea)

        assert send_btn.disabled is True
        assert str(send_btn.label) == "Send"
        assert text_area.read_only is False

        for _ in range(5):
            await pilot.press("enter")
        await pilot.pause()

        assert messages == []
        assert send_btn.disabled is True
        assert str(send_btn.label) == "Send"
        assert text_area.read_only is False

        ib.value = "draft while loading"
        await ib.action_submit()
        await pilot.pause()

        assert messages == []
        assert ib.value == "draft while loading"
        assert send_btn.disabled is True
        assert text_area.read_only is False

        engine.release.set()
        await asyncio.wait_for(engine.ready.wait(), timeout=1)
        await pilot.pause()

        assert messages == []
        assert send_btn.disabled is False
        assert str(send_btn.label) == "Send"
        assert text_area.read_only is False

        await pilot.press("enter")
        await pilot.pause()

        assert messages == ["draft while loading"]


@pytest.mark.asyncio
async def test_shell_mode_temporarily_replaces_trajectory_session_data_view(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Shell mode must be the sole main pane and restore the Session Data tab on exit."""
    from chrys.app.tui.terminal.panel import ShellPanel
    from chrys.app.tui.widgets.chat.panel import ChatPanel
    from chrys.app.tui.widgets.chat.session_json import SessionJsonPanel

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    async with app.run_test(size=(120, 32)) as pilot:
        chat = app.screen.query_one(ChatPanel)
        session_json = app.screen.query_one(SessionJsonPanel)
        shell = app.screen.query_one(ShellPanel)
        monkeypatch.setattr(shell, "_start_shell", lambda _cwd: None)

        async def wait_for_shell_state(expected: bool) -> None:
            for _ in range(20):
                if app.screen.shell_mode_state is expected:
                    return
                await pilot.pause()
            raise AssertionError(f"shell_mode_state did not become {expected}")

        await pilot.press("f12")
        await pilot.pause()
        await pilot.click("#session-data")
        await pilot.pause()
        assert session_json.display is True
        assert chat.display is False
        retained_status = session_json._status
        retained_generation = session_json._content_generation
        assert retained_status == "No active session."

        app.screen.shell_mode_state = True
        await wait_for_shell_state(True)
        assert app.screen.shell_mode_state is True
        assert shell.display is True
        assert session_json.display is False
        assert session_json._shell_mode_suspended is True
        assert chat.display is False
        assert session_json._status == retained_status
        assert session_json._content_generation == retained_generation

        app.screen.shell_mode_state = False
        await wait_for_shell_state(False)
        assert shell.display is False
        assert session_json.display is True
        assert session_json._shell_mode_suspended is False
        assert chat.display is False
        assert session_json._status == retained_status
        assert session_json._content_generation == retained_generation


@pytest.mark.asyncio
async def test_shell_mode_transitions_never_paint_half_applied_layout(tmp_path: Path) -> None:
    """Entering/exiting shell mode must never flush a mid-transition frame.

    The status-bar face flip inside enter/exit_shell_mode rebuilds the
    compositor map synchronously; a widget repaint flushed before the screen's
    deferred relayout paints from that map. If the flip runs before the
    display/class changes are complete, the sidebar flashes alone at the left
    edge for one frame.
    """

    from chrys.app.tui.widgets.chrome.status_bar import StatusBar

    class _Engine:
        async def shutdown(self) -> None:
            return

    class _EmptyRegistry:
        def list_profiles(self) -> list[object]:
            return []

        def load_all(self) -> None:
            return

        def get(self, _name: str) -> None:
            return None

    app = ChrysApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
        gc_freeze_enabled=False,
    )

    frames: list[list[str]] = []

    async with app.run_test(size=(120, 32)) as pilot:
        await pilot.pause()
        await pilot.pause()

        orig_display = app._display

        def spy_display(screen_arg, renderable) -> None:
            try:
                frames.append([strip.text for strip in screen_arg._compositor.render_strips()])
            except Exception:
                frames.append([])
            orig_display(screen_arg, renderable)

        app._display = spy_display  # type: ignore[method-assign]
        try:
            await pilot.press("!")
            for _ in range(8):
                await pilot.pause()
            assert app.screen.shell_mode_state is True
            assert app.screen.query_one(StatusBar).query_one(".status-selectors").display is False

            app.screen.shell_mode_state = False
            for _ in range(8):
                await pilot.pause()
            assert app.screen.shell_mode_state is False
            assert app.screen.query_one(StatusBar).query_one(".status-selectors").display is False
        finally:
            del app._display

    assert frames
    for index, frame in enumerate(frames):
        tab_columns = [row.find("Messages") for row in frame if "Messages" in row]
        assert tab_columns, f"sidebar tabs missing from painted frame {index}"
        assert min(tab_columns) > 60, f"frame {index} painted the sidebar in the left half (x={min(tab_columns)})"


def test_editor_keymap_pick_records_a_runtime_override_on_the_shared_handle() -> None:
    """The keymap picker is a write point like theme and locale: the shared
    handle records the choice, and a later reload reapplies it instead of
    un-switching the visible keymap."""
    app = object.__new__(ChrysApp)
    app._settings_handle = SettingsHandle(LoadedSettings(settings=Settings(), provenance={}))

    app._record_editor_keymap_override("vim")

    assert app._settings_handle.settings.editor_keymap == "vim"
    app._settings_handle.install(LoadedSettings(settings=Settings(), provenance={}))
    assert app._settings_handle.settings.editor_keymap == "vim"
