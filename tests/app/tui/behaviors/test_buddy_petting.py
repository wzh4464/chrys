# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for Buddy pet de-duplication while AI responses are in flight."""

from __future__ import annotations

import asyncio

import pytest

from chrys.app.features.buddy.notification import finish_pet_response
from chrys.app.tui.screens.main.commands import SlashCommandActions
from chrys.app.tui.screens.main.state import MainScreenServices, MainScreenState
from chrys.app.tui.screens.main.suggestions import SuggestionCallbacks, SuggestionHandler
from chrys.app.tui.screens.main.view_adapter import MainScreenViewAdapter
from chrys.app.tui.widgets.sidebar.buddy import BuddyPanel
from chrys.foundation.events.bus import EventBus
from tests.support.waiting import wait_for


class _Companion:
    name = "Biscuit"


class _SuggestionScreen:
    def __init__(self) -> None:
        self.app = type("_App", (), {"available_themes": ["textual-dark"], "theme": "textual-dark"})()
        self.state = MainScreenState()
        self.services = MainScreenServices(bus=EventBus(), state_store=object())
        self.notifications: list[str] = []

    def _debug(self, *_args, **_kwargs) -> None:
        return

    def action_pick_theme(self) -> None:
        return

    def _resume_last_session(self) -> None:
        return

    def _create_new_session(self) -> None:
        return

    def action_quit(self) -> None:
        return

    def action_sessions(self) -> None:
        return

    def _chdir(self, _arg: str) -> None:
        return

    def _copy_agent_responses(self, _arg: str) -> None:
        return

    def _toggle_fold(self) -> None:
        return

    def action_show_diff(self) -> None:
        return

    def action_show_rollback(self, _arg: str = "") -> None:
        return

    def _set_approval_mode(self, _arg: str) -> None:
        return

    def _open_model_config(self) -> None:
        return

    def _open_agent_config(self) -> None:
        return

    def _open_agent_config_tab(self, _tab: str) -> None:
        return

    def notify(self, message: str, **_kwargs) -> None:
        self.notifications.append(message)

    def query_one(self, _cls):
        raise LookupError("no sidebar in this test")

    def call_after_refresh(self, callback) -> None:
        callback()


def _make_handler(screen: _SuggestionScreen) -> SuggestionHandler:
    view = MainScreenViewAdapter(screen)  # type: ignore[arg-type]
    return SuggestionHandler(
        state=screen.state,
        services=screen.services,
        view=view,
        command_actions=SlashCommandActions(
            list_themes=lambda: sorted(screen.app.available_themes),
            get_theme=lambda: screen.app.theme,
            apply_theme=lambda name: setattr(screen.app, "theme", name),
            pick_theme=screen.action_pick_theme,
            list_languages=list,
            get_language=lambda: "system",
            apply_language=lambda _requested_locale: None,
            pick_language=lambda: None,
            render_unknown_language_warning=lambda requested_locale: f"Unknown /language locale: {requested_locale}",
            debug_event=screen._debug,
            new_session=screen._create_new_session,
            clear_session=lambda: None,
            quit_app=screen.action_quit,
            resume_session=screen._resume_last_session,
            fork_session=lambda: None,
            browse_session_list=screen.action_sessions,
            edit_session_title=lambda: None,
            apply_session_title=lambda _title: None,
            change_directory=screen._chdir,
            copy_conversation=screen._copy_agent_responses,
            fold_tools=screen._toggle_fold,
            open_diff=screen.action_show_diff,
            open_rollback=screen.action_show_rollback,
            apply_route_override=lambda _track, _reroute: None,
            send_prompt=lambda _text: None,
            describe_route=lambda: "global mode: auto",
            get_approval_mode=lambda: "manual",
            change_approval_mode=screen._set_approval_mode,
            configure_model=screen._open_model_config,
            configure_agent=screen._open_agent_config,
            configure_agent_tab=screen._open_agent_config_tab,
            show_runtime_details=lambda: None,
            configure_settings=lambda _tab: None,
            show_manual_pages=lambda _pages, _start_index: None,
            warn=lambda message, title, timeout: screen.notify(message, title=title, timeout=timeout),
        ),
        callbacks=SuggestionCallbacks(
            notify_warning=lambda message, title, timeout: screen.notify(message, title=title, timeout=timeout),
            show_file_suggestions=lambda: None,
            submit_user_text=lambda _text: None,
            start_agent_profile_switch=lambda _profile: None,
            start_model_profile_switch=lambda _profile: None,
        ),
        buddy_view=view,
    )


@pytest.mark.asyncio
async def test_buddy_slash_pet_noops_while_response_is_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    finish_pet_response()
    gate = asyncio.Event()
    calls = 0

    async def generate_response(_companion, _history, notify_callback=None):
        nonlocal calls
        calls += 1
        await gate.wait()
        if notify_callback is not None:
            notify_callback("done")
        return "done"

    companion = _Companion()

    def get_companion():
        return companion

    monkeypatch.setattr("chrys.app.features.buddy.config.is_muted", lambda: False)
    monkeypatch.setattr("chrys.app.features.buddy.notification.get_companion", get_companion)
    monkeypatch.setattr("chrys.app.features.buddy.notification.get_recent_conversation", lambda max_messages=10: [])
    monkeypatch.setattr("chrys.app.features.buddy.notification.generate_ai_pet_response_async", generate_response)

    try:
        handler = _make_handler(_SuggestionScreen())
        handler.build_slash_commands()

        assert handler.dispatch_slash_command("/buddy pet")
        assert handler.dispatch_slash_command("/buddy pet")
        await wait_for(lambda: calls == 1, interval=0, description="first pet response started")

        assert calls == 1

        task = handler.buddy_command.pet_task
        assert task is not None
        gate.set()
        await task

        assert handler.buddy_command.pet_task is None
        assert handler.dispatch_slash_command("/buddy pet")
        await wait_for(lambda: calls == 2, interval=0, description="second pet response started")
        assert calls == 2
    finally:
        finish_pet_response()


def test_buddy_panel_pet_noops_while_response_is_in_flight(monkeypatch: pytest.MonkeyPatch) -> None:
    finish_pet_response()
    pet_count_bumps = 0
    response_starts: list[bool] = []
    panel = BuddyPanel()
    panel.companion = _Companion()  # type: ignore[assignment]

    def increment_pet_count() -> None:
        nonlocal pet_count_bumps
        pet_count_bumps += 1

    monkeypatch.setattr("chrys.app.features.buddy.config.is_muted", lambda: False)
    monkeypatch.setattr("chrys.app.features.buddy.config.increment_pet_count", increment_pet_count)
    monkeypatch.setattr(panel, "_load_companion", lambda: None)
    monkeypatch.setattr(panel, "_render_sprite", lambda: None)
    monkeypatch.setattr(panel, "_update_status", lambda _message, **_kwargs: None)
    monkeypatch.setattr(panel, "set_interval", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        panel,
        "_trigger_ai_response",
        lambda *, pet_response_started=False: response_starts.append(pet_response_started),
    )

    try:
        panel.pet()
        panel.pet()

        assert pet_count_bumps == 1
        assert response_starts == [True]
    finally:
        finish_pet_response()
