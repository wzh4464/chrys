# Copyright (c) 2026 Chrys. All rights reserved.

"""Buddy panel widget for the sidebar."""

from __future__ import annotations

import asyncio
import math
from time import time
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.markup import escape as escape_markup
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from chrys.app.features.buddy.companion import get_companion
from chrys.app.features.buddy.sprites import get_frame_count, get_idle_frame, render_sprite
from chrys.app.features.buddy.types import PET_BURST_MS, RARITY_COLORS, Companion
from chrys.app.tui.buddy_messages import (
    BUDDY_HUMS,
    BUDDY_PONDERS,
    BUDDY_THINKING,
    BUDDY_THINKING_SOUNDS,
    BUDDY_TITLE,
)
from chrys.app.tui.i18n import render_str
from chrys.app.tui.util.visibility import is_widget_shown
from chrys.app.tui.widgets.sidebar.empty_state import SidebarEmptyStateLabel
from chrys.foundation.i18n import MessageRef, msg
from chrys.foundation.i18n.formatting import format_message

if TYPE_CHECKING:
    from textual.events import Click

    from chrys.app.tui.i18n import LocaleController

TICK_MS = 500

_BUDDY_EMPTY = msg("tui.sidebar.buddy.empty", fallback="No buddy hatched yet. Try /buddy hatch")
_BUDDY_LEVEL = msg("tui.sidebar.buddy.level", fallback="Level {level}")
_BUDDY_CLICK_TO_PET = msg("tui.sidebar.buddy.click_to_pet", fallback="Click to pet!")
_BUDDY_PETTING = msg("tui.sidebar.buddy.petting", fallback="Petting...")
_BUDDY_SPECIES = msg("tui.sidebar.buddy.species", fallback="Species: {species}")
_BUDDY_RARITY = msg("tui.sidebar.buddy.rarity", fallback="Rarity: {rarity}")
_BUDDY_PERSONALITY = msg("tui.sidebar.buddy.personality", fallback="Personality: {personality}")
_BUDDY_SHINY = msg("tui.sidebar.buddy.shiny", fallback="✨ Shiny!")
_BUDDY_NOTIFICATIONS_MUTED = msg("tui.sidebar.buddy.notifications_muted", fallback="🔇 Notifications muted")


class BuddyPanel(Widget):
    """Buddy companion panel for the sidebar.

    Displays full ASCII sprite animation in a dedicated panel.
    """

    DEFAULT_CSS = """
    BuddyPanel {
        width: 100%;
        height: 100%;
        padding: 1;
    }
    BuddyPanel #buddy-sprite {
        height: 5;
        width: 100%;
        content-align: left top;
        margin: 1 0;
    }
    BuddyPanel #buddy-level {
        height: auto;
        width: 100%;
        color: $accent;
        text-style: bold;
        margin: 0 0 1 0;
    }
    BuddyPanel #buddy-info {
        height: auto;
        width: 100%;
        color: $text-muted;
        margin: 1 0;
    }
    BuddyPanel #buddy-status {
        height: auto;
        width: 100%;
        color: $accent;
        margin: 1 0;
    }
    BuddyPanel.-petting #buddy-sprite {
        color: $accent;
    }
    BuddyPanel .buddy-empty {
        width: 1fr;
        height: 1fr;
        content-align: center middle;
        color: $text-muted;
    }
    """

    companion: reactive[Companion | None] = reactive(None)
    is_petting: reactive[bool] = reactive(False)

    def __init__(self, *, locale_controller: LocaleController | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._locale_controller = locale_controller
        self._tick_count = 0
        self._pet_timer = None
        self._pet_at = 0.0
        self._ai_task: asyncio.Task | None = None
        self._status_message: MessageRef = _BUDDY_CLICK_TO_PET.bind()
        self._status_style: str | None = "green"

    def compose(self) -> ComposeResult:
        yield SidebarEmptyStateLabel(Text(self._render_message(_BUDDY_EMPTY.bind())), classes="buddy-empty")
        yield Static("", id="buddy-sprite")
        yield Static("", id="buddy-level")
        yield Static("", id="buddy-info")
        yield Static("", id="buddy-status")

    def on_mount(self) -> None:
        """Start animation timer and load companion."""
        self._timer = self.set_interval(TICK_MS / 1000, self._on_tick)
        # Load companion and update UI directly
        comp = get_companion()
        self.companion = comp
        self._update_empty_visibility()
        if comp:
            self._update_info()
            self._update_status(_BUDDY_CLICK_TO_PET.bind(), style="green")
            self._render_sprite()

    def refresh_localization(self) -> None:
        """Retranslate the visible empty, level, and interaction status chrome."""
        if not self.is_mounted:
            return
        self.query_one(".buddy-empty", Static).update(Text(self._render_message(_BUDDY_EMPTY.bind())))
        if self.companion is not None:
            self._update_info()
            self._update_status(self._status_message, style=self._status_style)

    def _render_message(self, reference: MessageRef) -> str:
        controller = self._locale_controller
        if controller is None:
            return format_message(reference)
        return render_str(controller.localizer, reference)

    def _load_companion(self) -> None:
        """Load companion from config."""
        comp = get_companion()
        self.companion = comp

    def _on_tick(self) -> None:
        """Animation tick — update sprite frame."""
        self._tick_count += 1

        # Check for pet animation
        if self.is_petting and self._pet_timer:
            elapsed = (time() - self._pet_at) * 1000
            if elapsed >= PET_BURST_MS:
                self.is_petting = False
                if self._pet_timer:
                    self._pet_timer.stop()
                    self._pet_timer = None
                self._update_status(_BUDDY_CLICK_TO_PET.bind(), style="green")

        self._render_sprite()

    def _render_sprite(self) -> None:
        """Render the current sprite frame."""
        if not self.companion:
            return
        if not is_widget_shown(self):
            # Skip animation work while the sidebar/panel is hidden or the
            # sprite is scrolled out; the next tick resumes rendering. Do not
            # use is_on_screen here — it forces a full compositor arrange.
            return

        now = time()

        # 1. Animation frame logic
        if self.is_petting:
            # Faster animation during petting (~100ms per frame)
            frame = int(now * 10) % get_frame_count(self.companion.species)
            blink = False
        else:
            frame, blink = get_idle_frame(self.companion.species, self._tick_count)

        # 2. Horizontal "swimming" logic (sine wave)
        # Slow float by default (3s period), much faster when petting
        offset_x = 2 + int(math.sin(now * 12) * 2) if self.is_petting else 2 + int(math.sin(now * 1.5) * 2)

        # Render sprite lines
        lines = render_sprite(self.companion, frame, blink)

        # 3. Apply horizontal offset by prepending spaces
        offset_str = " " * max(0, offset_x)
        positioned_lines = [f"{offset_str}{line}" for line in lines]
        sprite_text = Text("\n".join(positioned_lines))

        # Apply color based on rarity and evolution
        color = RARITY_COLORS.get(self.companion.rarity, "#ffffff")
        sprite_text.stylize(color)

        if self.companion.is_evolved:
            sprite_text.stylize("bold")

        if self.companion.shiny:
            sprite_text.stylize("blink")

        try:
            sprite = self.query_one("#buddy-sprite", Static)
            # The sprite box is CSS-fixed (height 5, width 100%); frames only
            # repaint. layout=True here would force a full-screen arrange per
            # animation tick, which is O(all transcript widgets) in long chats.
            sprite.update(sprite_text, layout=False)
        except Exception:
            pass

    def _update_info(self) -> None:
        """Update buddy info display."""
        if not self.companion:
            return

        try:
            level = self.query_one("#buddy-level", Static)
            level_label = escape_markup(self._render_message(_BUDDY_LEVEL.bind(level=self.companion.level)))
            level.update(f"[b][accent]{level_label}[/accent][/b]")

            info = self.query_one("#buddy-info", Static)
            info_text = Text()
            info_text.append(self.companion.name, style="bold")
            info_text.append("\n" + self._render_message(_BUDDY_SPECIES.bind(species=self.companion.species.value)))
            info_text.append("\n" + self._render_message(_BUDDY_RARITY.bind(rarity=self.companion.rarity.value)))
            info_text.append(
                "\n" + self._render_message(_BUDDY_PERSONALITY.bind(personality=self.companion.personality))
            )
            if self.companion.shiny:
                info_text.append("\n" + self._render_message(_BUDDY_SHINY.bind()), style="gold1")
            info.update(info_text)
        except Exception:
            pass

    def _update_status(self, message: MessageRef, *, style: str | None) -> None:
        """Update status line."""
        try:
            from chrys.app.features.buddy.config import is_muted

            self._status_message = message
            self._status_style = style
            text = Text(self._render_message(message), style=style or "")
            if is_muted():
                text.append("\n\n" + self._render_message(_BUDDY_NOTIFICATIONS_MUTED.bind()), style="dim")
            status = self.query_one("#buddy-status", Static)
            status.update(text)
        except Exception:
            pass

    def pet(self) -> None:
        """Trigger pet animation and AI response."""
        if not self.companion:
            return

        from chrys.app.features.buddy.config import increment_pet_count, is_muted
        from chrys.app.features.buddy.notification import finish_pet_response, try_begin_pet_response

        pet_response_started = False
        if not is_muted():
            if not try_begin_pet_response():
                return
            pet_response_started = True

        try:
            # If a peer instance is wedged on the lock, skip the persistent bump
            # rather than crashing the TUI handler — the visual pet still plays.
            try:
                increment_pet_count()
            except TimeoutError:
                pass
            else:
                self._load_companion()

            self._pet_at = time()
            self.is_petting = True
            self._pet_timer = self.set_interval(0.1, self._render_sprite)
            self._update_status(_BUDDY_PETTING.bind(), style=None)

            # Trigger AI pet response (async, non-blocking)
            self._trigger_ai_response(pet_response_started=pet_response_started)
        except Exception:
            if pet_response_started:
                finish_pet_response()
            raise

    def _trigger_ai_response(self, *, pet_response_started: bool = False) -> None:
        """Trigger AI pet response in background."""
        from chrys.app.features.buddy.config import is_muted
        from chrys.app.features.buddy.notification import (
            finish_pet_response,
            generate_ai_pet_response_async,
            get_recent_conversation,
        )

        if is_muted():
            if pet_response_started:
                finish_pet_response()
            return
        companion = self.companion
        if companion is None:
            if pet_response_started:
                finish_pet_response()
            return

        import random

        # Show thinking notification
        thinking_sounds = [
            self._render_message(definition.bind(name=companion.name))
            for definition in (BUDDY_THINKING, BUDDY_HUMS, BUDDY_PONDERS, BUDDY_THINKING_SOUNDS)
        ]
        thinking_notification = self.screen.notify(
            random.choice(thinking_sounds),  # noqa: S311
            title=self._render_message(BUDDY_TITLE.bind()),
            timeout=None,
            markup=False,
        )

        # Get conversation history
        history = get_recent_conversation(max_messages=10)

        def on_ai_response(final_response: str) -> None:
            """Callback to show AI response."""
            # Clear thinking notification
            if thinking_notification:
                thinking_notification.dismiss()
            # Show AI response
            self.screen.notify(
                final_response,
                title=self._render_message(BUDDY_TITLE.bind()),
                timeout=10,
                markup=False,
            )
            # Refresh buddy panel
            self.refresh_companion()

        async def run_ai_call():
            try:
                companion = self.companion
                if companion is None:
                    return
                await generate_ai_pet_response_async(companion, history, notify_callback=on_ai_response)
            finally:
                if pet_response_started:
                    finish_pet_response()
                self._ai_task = None

        # Store task reference to prevent GC before completion
        task = asyncio.create_task(run_ai_call())
        self._ai_task = None if task.done() else task

    def on_click(self, event: Click) -> None:
        """Pet the buddy on click."""
        if self.companion:
            self.pet()
            event.stop()

    def _update_empty_visibility(self) -> None:
        """Toggle between empty label and buddy content."""
        has_buddy = self.companion is not None
        try:
            self.query_one(".buddy-empty", Static).display = not has_buddy
            self.query_one("#buddy-sprite", Static).display = has_buddy
            self.query_one("#buddy-level", Static).display = has_buddy
            self.query_one("#buddy-info", Static).display = has_buddy
            self.query_one("#buddy-status", Static).display = has_buddy
        except Exception:
            pass

    def watch_companion(self, companion: Companion | None) -> None:
        """Update display when companion changes."""
        # Only update UI if widget is mounted
        if not self.is_mounted:
            return
        self._update_empty_visibility()
        if companion:
            self._update_info()
            self._update_status(_BUDDY_CLICK_TO_PET.bind(), style="green")
            self._render_sprite()

    def refresh_companion(self) -> None:
        """Reload companion from config and update display."""
        # Load companion directly without using reactive
        comp = get_companion()
        self.companion = comp

        # Only update UI if widget is mounted
        if not self.is_mounted:
            return

        # Force refresh all UI elements
        self._update_info()
        self._update_status(_BUDDY_CLICK_TO_PET.bind(), style="green")
        self._render_sprite()
