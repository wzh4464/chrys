# Copyright (c) 2026 Chrys. All rights reserved.

"""SettingsDialog behaviour against stub ports."""

from __future__ import annotations

from typing import Any

import pytest
from textual.containers import VerticalGroup, VerticalScroll
from textual.widgets import Button, Checkbox, Input, Select, Static, TabbedContent

from chrys.app.tui.i18n import LocaleController
from chrys.app.tui.screens.settings import GENERAL_TAB_ID, NOTIFICATIONS_TAB_ID, SettingsDialog
from chrys.app.tui.screens.settings.dialog import pane_id
from chrys.app.tui.screens.settings.panes.notifications import NotificationsPane
from chrys.app.tui.screens.settings.rows import SettingRow
from chrys.foundation.config.settings import Settings
from chrys.foundation.config.spec import SettingOrigin, Source
from chrys.foundation.i18n.formatting import format_message
from tests.app.tui.screens.settings.support import Host, StubPorts, env_origin


def _rows(dialog: SettingsDialog) -> dict[str, SettingRow]:
    return {row.spec.key: row for row in dialog.rows()}


def _hint(row: SettingRow) -> str:
    return str(row.query_one(".settings-row-hint", Static).render())


def _badges(row: SettingRow) -> str:
    return str(row.query_one(".settings-row-badges", Static).render())


async def _open(app: Host, ports: StubPorts, *, initial_tab: str = GENERAL_TAB_ID) -> SettingsDialog:
    dialog = SettingsDialog(ports, initial_tab=initial_tab)
    await app.push_screen(dialog)
    return dialog


@pytest.mark.asyncio
async def test_mounting_the_dialog_writes_nothing_and_focuses_a_control() -> None:
    ports = StubPorts()
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports)
        await pilot.pause()
        await pilot.pause()

        assert ports.persisted == []
        assert ports.live == []
        assert ports.confirms == []
        assert isinstance(dialog.focused, Select)
        rows = dialog.rows()
        assert len(rows) == 21
        assert all(row.spec.key != "trajectory.verify_commands" for row in rows)
        assert dialog.query_one(TabbedContent).active == pane_id(GENERAL_TAB_ID)


@pytest.mark.asyncio
async def test_opens_on_the_requested_tab_and_falls_back_for_unknown_ids() -> None:
    app = Host()
    # 85% of 46 rows: the tallest tab still fits without scrolling.
    async with app.run_test(size=(100, 46)) as pilot:
        dialog = await _open(app, StubPorts(), initial_tab=NOTIFICATIONS_TAB_ID)
        await pilot.pause()
        assert dialog.query_one(TabbedContent).active == pane_id(NOTIFICATIONS_TAB_ID)
        assert isinstance(dialog.focused, Checkbox)
        pane = dialog.query_one(NotificationsPane)
        assert pane.query_one("#notifications-enabled", Checkbox).value is True
        scroll = pane.query_ancestor(VerticalScroll)
        assert scroll.virtual_size.height <= scroll.container_size.height
        assert pane.query_one("#notifications-test", Button).region.height == 1
        await dialog.dismiss(None)

        dialog = await _open(app, StubPorts(), initial_tab="does-not-exist")
        await pilot.pause()
        assert dialog.query_one(TabbedContent).active == pane_id(GENERAL_TAB_ID)


@pytest.mark.asyncio
async def test_bool_row_persists_reload_keys_and_live_applies_live_keys() -> None:
    ports = StubPorts()
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports)
        await pilot.pause()
        rows = _rows(dialog)

        rows["session.title.auto"].query_one(Checkbox).value = False
        await pilot.pause()
        assert ports.persisted == [{"session.title.auto": False}]

        rows["ui.theme"].query_one(Select).value = "textual-dark"
        await pilot.pause()
        assert ports.live == [("ui.theme", "textual-dark")]
        # LIVE keys never go through the persistence queue.
        assert ports.persisted == [{"session.title.auto": False}]


@pytest.mark.asyncio
async def test_select_row_injects_the_current_value_when_it_is_not_a_choice() -> None:
    ports = StubPorts(Settings(theme="retired-theme", default_agent="ghost"))
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports)
        await pilot.pause()
        rows = _rows(dialog)

        theme = rows["ui.theme"].query_one(Select)
        assert theme.value == "retired-theme"
        assert [str(prompt) for prompt, _value in theme._options] == [
            "chrys",
            "chrys-dark",
            "textual-dark",
            "retired-theme (current)",
        ]

        agent = rows["agent.default_profile"].query_one(Select)
        assert agent.value == "ghost"
        assert [str(prompt) for prompt, _value in agent._options] == ["(default)", "Code", "Chat", "ghost (current)"]
        assert ports.live == [] and ports.persisted == []

        # Picking a real choice writes it; blank means "use the default".
        agent.value = ""
        await pilot.pause()
        assert ports.persisted == [{"agent.default_profile": ""}]


@pytest.mark.asyncio
async def test_approval_mode_row_hides_bypass_and_confirms_only_the_move_to_auto() -> None:
    ports = StubPorts()
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="security")
        await pilot.pause()
        rows = _rows(dialog)
        select = rows["approval.default_mode"].query_one(Select)
        assert [value for _prompt, value in select._options] == ["manual", "auto"]

        select.value = "auto"
        await pilot.pause()
        assert len(ports.confirms) == 1
        assert format_message(ports.confirms[0]).startswith("Auto mode lets a model approve")
        assert ports.persisted == [{"approval.default_mode": "auto"}]

        select.value = "manual"
        await pilot.pause()
        assert len(ports.confirms) == 1
        assert ports.persisted[-1] == {"approval.default_mode": "manual"}


@pytest.mark.asyncio
async def test_dangerous_bool_confirms_when_enabling_and_reverts_when_declined() -> None:
    ports = StubPorts()
    ports.confirm_answer = False
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="security")
        await pilot.pause()
        rows = _rows(dialog)
        checkbox = rows["log.raw_http_capture"].query_one(Checkbox)

        checkbox.value = True
        await pilot.pause()
        assert len(ports.confirms) == 1
        assert ports.persisted == []
        assert checkbox.value is False

        ports.confirm_answer = True
        checkbox.value = True
        await pilot.pause()
        assert len(ports.confirms) == 2
        assert ports.persisted == [{"log.raw_http_capture": True}]

        # Turning it back off is never a dangerous transition.
        checkbox.value = False
        await pilot.pause()
        assert len(ports.confirms) == 2
        assert ports.persisted[-1] == {"log.raw_http_capture": False}


@pytest.mark.parametrize(
    ("key", "tab", "typed", "expected", "error"),
    [
        ("rollback.snapshots_keep", "sessions", "12345", {"rollback.snapshots_keep": 12345}, None),
        ("rollback.snapshots_keep", "sessions", "abc", None, "Expected an integer."),
        ("rollback.snapshots_keep", "sessions", "", None, "A value is required."),
        # Below the floor is clamped, not rejected.
        ("rollback.snapshots_keep", "sessions", "0", {"rollback.snapshots_keep": 1}, None),
        ("llm.retry.max_transient", "models", "", {"llm.retry.max_transient": None}, None),
        ("llm.retry.max_transient", "models", "3", {"llm.retry.max_transient": 3}, None),
        ("llm.retry.max_transient", "models", "-1", None, "Expected a non-negative integer."),
        # Blank means the built-in default; the ask-user timeout's default is a number.
        ("tools.ask_user.timeout_seconds", "tools", "", {"tools.ask_user.timeout_seconds": 600}, None),
        ("tools.ask_user.timeout_seconds", "tools", "0", {"tools.ask_user.timeout_seconds": None}, None),
        ("tools.ask_user.timeout_seconds", "tools", "90", {"tools.ask_user.timeout_seconds": 90}, None),
    ],
)
@pytest.mark.asyncio
async def test_input_rows_commit_through_the_field_coercer(
    key: str,
    tab: str,
    typed: str,
    expected: dict[str, Any] | None,
    error: str | None,
) -> None:
    # Start away from the defaults so that clearing a field is a real edit.
    ports = StubPorts(Settings(max_transient_retries=9, buddy_model="old-model", ask_user_timeout_seconds=45))
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab=tab)
        await pilot.pause()
        row = _rows(dialog)[key]
        field = row.query_one(Input)

        field.value = typed
        row.commit_pending()
        await pilot.pause()

        if expected is None:
            assert ports.persisted == []
        else:
            assert ports.persisted == [expected]
        if error is None:
            assert not row.query_one(".settings-row-hint", Static).has_class("-error")
        else:
            assert _hint(row) == error
            assert row.query_one(".settings-row-hint", Static).has_class("-error")


@pytest.mark.asyncio
async def test_no_timeout_shows_as_zero_while_an_unset_retry_count_shows_blank() -> None:
    ports = StubPorts(Settings(ask_user_timeout_seconds=None, max_transient_retries=None))
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="tools")
        await pilot.pause()
        assert dialog.query_one(TabbedContent).active == pane_id("tools")
        assert _rows(dialog)["tools.ask_user.timeout_seconds"].query_one(Input).value == "0"

        dialog.query_one(TabbedContent).active = pane_id("models")
        await pilot.pause()
        assert _rows(dialog)["llm.retry.max_transient"].query_one(Input).value == ""


@pytest.mark.asyncio
async def test_input_row_clamps_and_shows_the_clamped_value() -> None:
    ports = StubPorts()
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="models")
        await pilot.pause()
        row = _rows(dialog)["llm.retry.max_transient"]
        field = row.query_one(Input)

        field.value = "999"
        row.commit_pending()
        await pilot.pause()

        assert ports.persisted == [{"llm.retry.max_transient": 50}]
        assert field.value == "50"


@pytest.mark.asyncio
async def test_closing_commits_a_pending_input_edit_then_tells_the_ports() -> None:
    ports = StubPorts()
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="sessions")
        await pilot.pause()
        row = _rows(dialog)["rollback.snapshots_keep"]
        row.query_one(Input).value = "7"

        await pilot.press("escape")
        await pilot.pause()

        assert ports.persisted == [{"rollback.snapshots_keep": 7}]
        assert ports.closed == 1


@pytest.mark.asyncio
async def test_greyed_provenance_disables_the_control_and_explains_why() -> None:
    ports = StubPorts(Settings(theme="chrys-dark"), provenance={"ui.theme": env_origin()})
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports)
        await pilot.pause()
        row = _rows(dialog)["ui.theme"]

        assert row.query_one(Select).disabled is True
        assert "env" in _badges(row)
        assert _hint(row) == "Set by environment variable CHRYS_THEME."
        assert isinstance(dialog.focused, Select)
        assert dialog.focused is not row.query_one(Select)


@pytest.mark.asyncio
async def test_badges_and_status_follow_apply_kind_and_pending_state() -> None:
    ports = StubPorts()
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="security")
        await pilot.pause()
        rows = _rows(dialog)
        container = dialog.query_one("#settings-container", VerticalGroup)

        # Rows carry only the danger mark; when a change takes effect is the
        # dialog's status line, which lives in the frame's bottom edge.
        assert _badges(rows["approval.default_mode"]) == "⚠"
        assert _badges(rows["log.raw_http_capture"]) == "⚠"
        assert _badges(rows["otel.enabled"]) == ""
        assert _badges(rows["project.config_enabled"]) == ""
        assert str(container.border_subtitle) == "Changes are saved as you make them"
        assert not container.has_class("-status-active")

        ports.restart_pending = frozenset({"log.raw_http_capture", "otel.enabled"})
        ports.dirty = True
        dialog.reproject()
        await pilot.pause()
        assert _badges(rows["log.raw_http_capture"]) == "⚠"
        assert str(container.border_subtitle) == "2 changes apply after restart · Changes apply on close (reload)"
        assert container.has_class("-status-active")

        ports.turn_running = True
        dialog.refresh_status()
        assert (
            str(container.border_subtitle) == "2 changes apply after restart · Changes apply when the current turn ends"
        )


@pytest.mark.asyncio
async def test_reproject_repaints_values_without_writing() -> None:
    ports = StubPorts()
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports)
        await pilot.pause()
        rows = _rows(dialog)

        ports.desired["session.title.auto"] = False
        ports.install(Settings(theme="textual-dark"))
        dialog.reproject()
        await pilot.pause()

        assert rows["session.title.auto"].query_one(Checkbox).value is False
        assert rows["ui.theme"].query_one(Select).value == "textual-dark"
        assert ports.persisted == [] and ports.live == []


@pytest.mark.asyncio
async def test_the_buddy_model_is_picked_from_the_registered_model_ids() -> None:
    ports = StubPorts()
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="models")
        await pilot.pause()
        select = _rows(dialog)["model.role.buddy_model_id"].query_one(Select)

        # Blank stores "" and reads "Use active model"; the options are bare model ids.
        assert select.value == ""
        assert [str(prompt) for prompt, _value in select._options] == ["Use active model", "vendor/one", "vendor/two"]
        select.value = "vendor/two"
        await pilot.pause()
        assert ports.persisted == [{"model.role.buddy_model_id": "vendor/two"}]

        # A blank retry cap shows the frontend default it falls back to.
        retries = _rows(dialog)["llm.retry.max_transient"].query_one(Input)
        assert retries.value == "" and retries.placeholder == "7"


@pytest.mark.asyncio
async def test_locale_refresh_repaints_tabs_sections_rows_and_status_in_place() -> None:
    controller = LocaleController(Settings())
    ports = StubPorts()
    app = Host(controller)
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = SettingsDialog(ports, locale_controller=controller)
        await app.push_screen(dialog)
        await pilot.pause()
        rows = _rows(dialog)
        theme_select = rows["ui.theme"].query_one(Select)
        tabs = dialog.query_one(TabbedContent)
        assert str(tabs.get_tab(pane_id(GENERAL_TAB_ID)).label) == "General"

        controller.switch_locale("zh-Hans")
        await pilot.pause()

        assert str(tabs.get_tab(pane_id(GENERAL_TAB_ID)).label) == "通用"
        assert str(tabs.get_tab(pane_id(NOTIFICATIONS_TAB_ID)).label) == "通知"
        assert rows["ui.theme"].query_one(".settings-row-label").render().plain == "主题"
        assert _hint(rows["ui.theme"]) == "用 F9 预览主题。"
        assert str(dialog.query_one("#settings-container", VerticalGroup).border_subtitle) == "修改即时保存"
        assert rows["ui.theme"].query_one(Select) is theme_select
        assert theme_select.value == "chrys"
        assert (
            dialog.query_one(NotificationsPane).query_one("#notifications-enabled", Checkbox).label.plain == "启用通知"
        )
        assert ports.live == [] and ports.persisted == []


@pytest.mark.asyncio
async def test_a_projection_landing_mid_edit_does_not_wipe_the_typed_text() -> None:
    ports = StubPorts()
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="sessions")
        await pilot.pause()
        row = _rows(dialog)["rollback.snapshots_keep"]
        field = row.query_one(Input)
        field.focus()
        await pilot.pause()
        field.value = "7777"

        # Another row's write lands and the coordinator reprojects everything.
        ports.desired["storage.session_root_dir"] = "/elsewhere"
        dialog.reproject()
        await pilot.pause()

        assert field.value == "7777"
        assert _rows(dialog)["storage.session_root_dir"].query_one(Input).value == "/elsewhere"

        # An unfocused row is repainted as usual.
        _rows(dialog)["storage.session_root_dir"].query_one(Input).focus()
        await pilot.pause()
        assert ports.persisted == [{"rollback.snapshots_keep": 7777}]
        ports.desired["rollback.snapshots_keep"] = 999
        dialog.reproject()
        await pilot.pause()
        assert field.value == "999"


@pytest.mark.asyncio
async def test_a_failed_write_snaps_a_focused_input_back_to_the_value_in_force() -> None:
    ports = StubPorts()
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="sessions")
        await pilot.pause()
        row = _rows(dialog)["rollback.snapshots_keep"]
        field = row.query_one(Input)
        in_force = field.value
        field.focus()
        await pilot.pause()
        field.value = " 7777 "
        await pilot.press("enter")
        await pilot.pause()
        assert ports.persisted == [{"rollback.snapshots_keep": 7777}]
        assert field.value == "7777"
        assert field.has_focus

        # The write is rejected: the coordinator reprojects without any
        # ``desired`` for the key, and the field follows even though it kept focus.
        dialog.reproject()
        await pilot.pause()

        assert field.value == in_force
        # Leaving the row must not resubmit the rejected value.
        _rows(dialog)["storage.session_root_dir"].query_one(Input).focus()
        await pilot.pause()
        assert ports.persisted == [{"rollback.snapshots_keep": 7777}]


@pytest.mark.asyncio
async def test_locale_refresh_keeps_focus_and_uncommitted_input() -> None:
    controller = LocaleController(Settings())
    ports = StubPorts()
    app = Host(controller)
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = SettingsDialog(ports, initial_tab="sessions", locale_controller=controller)
        await app.push_screen(dialog)
        await pilot.pause()
        row = _rows(dialog)["rollback.snapshots_keep"]
        field = row.query_one(Input)
        field.focus()
        await pilot.pause()
        field.value = "4242"

        controller.switch_locale("zh-Hans")
        await pilot.pause()

        assert dialog.focused is field
        assert field.value == "4242"
        assert row.query_one(".settings-row-label").render().plain == "保留的回滚快照数"
        assert ports.persisted == []


@pytest.mark.asyncio
async def test_cli_and_sealed_provenance_stay_editable_with_a_badge() -> None:
    from pathlib import Path

    from chrys.foundation.config.coercion import Coerced, CoerceReason, CoerceStatus
    from chrys.foundation.config.settings_store import SettingsWarning

    settings_file = Path("/home/me/.chrys/settings.yaml")
    rejected = SettingsWarning(
        key="log.raw_http_capture",
        origin=SettingOrigin(layer=Source.USER, path=settings_file),
        outcome=Coerced(status=CoerceStatus.INVALID, raw="maybe", reason=CoerceReason.EXPECTED_BOOL),
    )
    ports = StubPorts(
        Settings(default_agent="code", raw_http_capture=False),
        provenance={"agent.default_profile": SettingOrigin(layer=Source.CLI)},
        sealed_keys=frozenset({"log.raw_http_capture"}),
        warnings=(rejected,),
    )
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="models")
        await pilot.pause()
        agent = _rows(dialog)["agent.default_profile"]
        assert agent.query_one(Select).disabled is False
        assert _badges(agent) == "this session"
        assert _hint(agent).startswith("Set for this session by a command-line option")

        dialog.query_one(TabbedContent).active = pane_id("security")
        await pilot.pause()
        raw = _rows(dialog)["log.raw_http_capture"]
        assert raw.query_one(Checkbox).disabled is False
        assert _badges(raw) == "⚠ · sealed"
        # The hint is the loader's own verdict for the key, not a generic sentence.
        # The path renders in the platform's own spelling (backslashes on Windows).
        assert _hint(raw) == f"Ignoring log.raw_http_capture in {settings_file}=maybe: expected a boolean."


@pytest.mark.asyncio
async def test_an_env_pinned_bypass_shows_as_the_current_choice_without_writing() -> None:
    ports = StubPorts(
        Settings(default_approval_mode="bypass"),
        provenance={"approval.default_mode": env_origin()},
    )
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="security")
        await pilot.pause()
        select = _rows(dialog)["approval.default_mode"].query_one(Select)

        assert select.disabled is True
        assert select.value == "bypass"
        assert [str(prompt) for prompt, _value in select._options] == ["manual", "auto", "bypass (current)"]
        assert ports.persisted == [] and ports.confirms == []


@pytest.mark.asyncio
async def test_a_dormant_project_file_swaps_the_project_gate_hint_until_the_gate_is_on() -> None:
    from pathlib import Path

    from chrys.foundation.config.settings_store import DormantProjectConfig, LoadedSettings

    ports = StubPorts()
    ports.handle.install(
        LoadedSettings(
            settings=Settings(),
            provenance={},
            dormant_project=(
                DormantProjectConfig(
                    path=Path("/ws/.chrys/settings.yaml"), keys=("rollback.snapshots_keep", "app.dev_mode")
                ),
            ),
        )
    )
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="security")
        await pilot.pause()
        row = _rows(dialog)["project.config_enabled"]

        assert _hint(row) == "Project settings found (2 keys) — enable to apply them."

        row.query_one(Checkbox).value = True
        await pilot.pause()

        assert ports.persisted == [{"project.config_enabled": True}]
        assert _hint(row).startswith("Let <workspace>/.chrys/settings.yaml")


@pytest.mark.asyncio
async def test_commit_pending_lands_a_focused_edit_without_closing() -> None:
    """The app calls this before exiting with the dialog on top (Ctrl+Q is an app binding)."""
    ports = StubPorts()
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="sessions")
        await pilot.pause()
        field = _rows(dialog)["rollback.snapshots_keep"].query_one(Input)
        field.focus()
        await pilot.pause()
        field.value = "7"

        dialog.commit_pending()
        await pilot.pause()

        assert ports.persisted == [{"rollback.snapshots_keep": 7}]
        assert ports.closed == 0


@pytest.mark.asyncio
async def test_quit_commits_a_settings_dialog_that_sits_under_another_modal() -> None:
    """Ctrl+Q with a confirmation or file picker above the dialog still lands the edit."""
    from types import SimpleNamespace

    from chrys.app.tui.app import ChrysApp

    committed: list[str] = []
    exits: list[bool] = []
    dialog = SettingsDialog.__new__(SettingsDialog)
    dialog.commit_pending = lambda: committed.append("settings")  # type: ignore[method-assign]
    fake_app = SimpleNamespace(
        screen_stack=[object(), dialog, SimpleNamespace(name="confirm-on-top")],
        exit=lambda: exits.append(True),
    )

    await ChrysApp.action_quit(fake_app)  # type: ignore[arg-type]

    assert committed == ["settings"]
    assert exits == [True]


@pytest.mark.asyncio
async def test_a_pane_keeps_its_content_width_when_the_scrollbar_appears() -> None:
    from textual.containers import VerticalScroll

    from tests.support.waiting import wait_for

    ports = StubPorts()
    app = Host()
    async with app.run_test(size=(100, 60)) as pilot:
        dialog = await _open(app, ports, initial_tab="security")
        await pilot.pause()
        pane = dialog.query_one(f"#{pane_id('security')}")
        scroll = pane.query_one(VerticalScroll)
        row = _rows(dialog)["approval.default_mode"]
        assert scroll.styles.scrollbar_gutter == "stable"
        await wait_for(
            lambda: row.size.width > 0 and not scroll.show_vertical_scrollbar,
            pilot=pilot,
            description="the security tab to settle without a scrollbar",
        )
        width_without_scrollbar = row.size.width

        await pilot.resize_terminal(100, 16)
        await wait_for(
            lambda: scroll.show_vertical_scrollbar,
            pilot=pilot,
            description="the security tab to overflow and show its scrollbar",
        )

        assert row.size.width == width_without_scrollbar


@pytest.mark.asyncio
async def test_profile_display_names_are_literal_text_not_markup() -> None:
    from textual.widgets._select import SelectCurrent

    ports = StubPorts(Settings(default_agent="ops"))
    ports.agent_profiles = [("ops", "Ops [/oops]"), ("work", "Code [Work]"), ("plain [odd]", "plain [odd]")]
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="models")
        await pilot.pause()
        select = _rows(dialog)["agent.default_profile"].query_one(Select)

        assert select.value == "ops"
        assert [str(prompt) for prompt, _value in select._options] == [
            "(default)",
            "Ops [/oops]",
            "Code [Work]",
            "plain [odd]",
        ]
        shown = select.query_one(SelectCurrent).query_one("#label", Static)
        assert "Ops [/oops]" in str(shown.render())

        # An unlisted stored value is injected verbatim as well.
        ports.install(Settings(default_agent="ghost [x]"))
        dialog.reproject()
        await pilot.pause()
        assert str(select._options[-1][0]) == "ghost [x] (current)"


@pytest.mark.asyncio
async def test_a_select_shows_its_placeholder_when_the_projected_value_is_blank() -> None:
    """``Select.value`` only notifies on change; a blank projection over new
    options must still repaint the current-value label."""
    from textual.widgets._select import SelectCurrent

    ports = StubPorts(Settings(default_agent=""))
    app = Host()
    async with app.run_test(size=(100, 40)) as pilot:
        dialog = await _open(app, ports, initial_tab="models")
        await pilot.pause()
        rows = _rows(dialog)

        def _label(key: str) -> str:
            return str(rows[key].query_one(Select).query_one(SelectCurrent).query_one("#label", Static).render())

        assert _label("model.role.session_title") == "Use active model"
        assert _label("agent.default_profile") == "(default)"

        ports.install(Settings(default_agent="chat"))
        dialog.reproject()
        await pilot.pause()
        assert _label("agent.default_profile") == "Chat"
        assert ports.persisted == [] and ports.live == []
