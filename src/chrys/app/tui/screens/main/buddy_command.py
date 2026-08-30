# Copyright (c) 2026 Chrys. All rights reserved.

"""Stateful /buddy slash-command handling."""

from __future__ import annotations

import asyncio
import contextlib
import random
from collections.abc import Callable

from chrys.app.features.buddy.notification import BUDDY_MUTED, NO_BUDDY_TO_PET
from chrys.app.tui.buddy_messages import BUDDY_HUMS, BUDDY_PONDERS, BUDDY_THINKING, BUDDY_THINKING_SOUNDS
from chrys.app.tui.screens.main.ports import BuddyCommandView
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message

_SUBCOMMAND_HATCH = msg("tui.buddy.subcommand.hatch", fallback="Hatch a new buddy")
_SUBCOMMAND_INFO = msg("tui.buddy.subcommand.info", fallback="Show buddy info")
_SUBCOMMAND_PET = msg("tui.buddy.subcommand.pet", fallback="Pet your buddy")
_SUBCOMMAND_MUTE = msg("tui.buddy.subcommand.mute", fallback="Toggle buddy notifications")
_SUBCOMMAND_NAME = msg("tui.buddy.subcommand.name", fallback="Rename your buddy")


class BuddyCommandController:
    """Own retained async state for /buddy commands."""

    def __init__(
        self,
        view: BuddyCommandView,
        *,
        render_message: Callable[[MessageRef], str] = format_message,
    ) -> None:
        self._view = view
        self._render_message = render_message
        self._pet_task: asyncio.Task[None] | None = None
        self._pet_response_started = False
        self._thinking_notification: object | None = None

    @property
    def pet_task(self) -> asyncio.Task[None] | None:
        """Return the retained pet-response task for focused lifecycle tests."""
        return self._pet_task

    def subcommands(self) -> list[tuple[str, str]]:
        """Return available /buddy subcommands."""
        from chrys.app.features.buddy.config import has_hatched

        if not has_hatched():
            return [("hatch", self._render_message(_SUBCOMMAND_HATCH.bind()))]
        return [
            ("info", self._render_message(_SUBCOMMAND_INFO.bind())),
            ("pet", self._render_message(_SUBCOMMAND_PET.bind())),
            ("mute", self._render_message(_SUBCOMMAND_MUTE.bind())),
            ("name", self._render_message(_SUBCOMMAND_NAME.bind())),
        ]

    def handle(self, arg: str) -> None:
        """Handle /buddy command text."""
        arg_stripped = arg.strip() if arg else None
        if arg_stripped and arg_stripped.lower() == "pet":
            self._handle_pet()
            return
        self._handle_sync_command(arg_stripped)

    def _handle_pet(self) -> None:
        from chrys.app.features.buddy.config import is_muted
        from chrys.app.features.buddy.notification import (
            finish_pet_response,
            generate_ai_pet_response_async,
            get_companion,
            get_recent_conversation,
            try_begin_pet_response,
        )

        companion = get_companion()
        if companion is None:
            self._view.notify_buddy(
                self._render_message(NO_BUDDY_TO_PET.bind()),
                severity="warning",
                timeout=10,
            )
            return
        if is_muted():
            self._view.notify_buddy(
                self._render_message(BUDDY_MUTED.bind(name=companion.name)),
                severity="warning",
                timeout=10,
            )
            return
        if not try_begin_pet_response():
            return
        self._pet_response_started = True

        try:
            thinking_sounds = [
                definition.bind(name=companion.name)
                for definition in (BUDDY_THINKING, BUDDY_HUMS, BUDDY_PONDERS, BUDDY_THINKING_SOUNDS)
            ]
            self._thinking_notification = self._view.notify_buddy(
                random.choice(thinking_sounds),  # noqa: S311
                timeout=None,
            )
            history = get_recent_conversation(max_messages=10)

            def on_ai_response(final_response: str) -> None:
                self._view.dismiss_notification(self._thinking_notification)
                self._thinking_notification = None
                self._view.notify_buddy(final_response, timeout=10)
                self._view.refresh_buddy_panel(focus_tab=False)

            async def run_ai_call() -> None:
                try:
                    await generate_ai_pet_response_async(companion, history, notify_callback=on_ai_response)
                except Exception:
                    self._view.dismiss_notification(self._thinking_notification)
                    self._thinking_notification = None
                    raise
                finally:
                    if self._pet_response_started:
                        finish_pet_response()
                        self._pet_response_started = False
                    self._pet_task = None

            task = asyncio.create_task(run_ai_call())
            self._pet_task = None if task.done() else task
        except Exception:
            self._view.dismiss_notification(self._thinking_notification)
            self._thinking_notification = None
            if self._pet_response_started:
                finish_pet_response()
                self._pet_response_started = False
            raise

    def _handle_sync_command(self, arg: str | None) -> None:
        from chrys.app.features.buddy.config import is_muted
        from chrys.app.features.buddy.notification import handle_buddy_command

        response, severity = handle_buddy_command(arg)
        show_toast = not is_muted() or (arg and arg.lower() == "mute")
        if response and show_toast:
            rendered = response if isinstance(response, str) else self._render_message(response)
            self._view.notify_buddy(rendered, severity=severity, timeout=10)

        def do_refresh() -> None:
            self._view.refresh_buddy_panel(focus_tab=True)

        self._view.call_after_refresh(do_refresh)

    async def shutdown(self) -> None:
        """Cancel retained /buddy pet work when the owning screen unmounts."""
        from chrys.app.features.buddy.notification import finish_pet_response

        task = self._pet_task
        self._pet_task = None
        self._view.dismiss_notification(self._thinking_notification)
        self._thinking_notification = None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if self._pet_response_started:
            finish_pet_response()
            self._pet_response_started = False
