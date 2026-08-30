# Copyright (c) 2026 Chrys. All rights reserved.

"""AskUserDialog — modal dialog for agent ask_user questions."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from rich.text import Text
from textual import on
from textual.binding import Binding
from textual.containers import VerticalGroup
from textual.geometry import Size
from textual.screen import ModalScreen

from chrys.app.tui.behaviors.right_click_copy import RightClickScreenCopyMixin
from chrys.app.tui.i18n import render_text, widget_localizer
from chrys.app.tui.widgets import (
    AskUserInlineRequested,
    AskUserOptions,
    AskUserResponseFooter,
    AskUserSubmitted,
    StableAutoHeightScroll,
    normalize_ask_user_options,
)
from chrys.app.tui.widgets.markdown import VirtualizedMarkdown
from chrys.foundation.i18n import msg

if TYPE_CHECKING:
    from textual.app import ComposeResult


_QUESTION_TITLE = msg("tui.ask_user.title", fallback="Question")


class _AskUserQuestionMarkdown(VirtualizedMarkdown, can_focus=False):
    """Markdown question renderer that lets the parent body own scrolling."""

    def watch_virtual_size(self, _old: Size, new: Size) -> None:
        if new.height > 0:
            self.styles.height = new.height


@dataclass(frozen=True)
class AskUserInlineResult:
    """Modal result indicating the prompt should move into the chat transcript."""

    request_id: str
    draft_text: str = ""


AskUserDialogResult = tuple[str, str] | AskUserInlineResult | None


class AskUserDialog(RightClickScreenCopyMixin, ModalScreen[AskUserDialogResult]):
    """Modal dialog for agent questions to the user.

    Displays the question, optional preset option buttons, and a free-text
    input field.  User cannot cancel manually — Esc is disabled — but the
    backend can close the dialog when the pending ask_user request expires.
    Dismisses with ``(request_id, response_text)``, ``AskUserInlineResult``,
    or ``None`` on timeout.
    """

    BINDINGS: ClassVar[list] = [
        Binding("escape", "noop", show=False, priority=True),
    ]

    CSS_PATH = "ask_user.tcss"

    def __init__(
        self,
        request_id: str,
        question: str,
        options: list[str] | None = None,
        caller_name: str = "",
        initial_response: str = "",
    ) -> None:
        self._request_id = request_id
        self._question = question
        self._options = normalize_ask_user_options(options)
        self._caller_name = caller_name
        self._initial_response = initial_response
        self._dismissed = False
        super().__init__()

    def compose(self) -> ComposeResult:
        with VerticalGroup(id="askuser-container") as container:
            container.border_title = render_text(widget_localizer(self), _QUESTION_TITLE.bind())
            if self._caller_name:
                container.border_subtitle = Text(self._caller_name)
            # Scrollable body — question and preset responses move together.
            with StableAutoHeightScroll(id="askuser-inner", can_focus=False):
                yield _AskUserQuestionMarkdown(
                    self._question,
                    id="askuser-question",
                )
                if self._options:
                    yield AskUserOptions(self._request_id, self._options)
            yield AskUserResponseFooter(
                self._request_id,
                allow_inline=True,
                initial_response=self._initial_response,
            )

    def on_mount(self) -> None:
        if self._options:
            self.query_one("#askuser-opt-0").focus()
        else:
            self.query_one(AskUserResponseFooter).focus_input()

    def _safe_dismiss(self, result: AskUserDialogResult) -> None:
        """Dismiss once and hide immediately so late timeout/user clicks do not race."""
        if self._dismissed:
            return
        self._dismissed = True
        with contextlib.suppress(Exception):
            self.query_one("#askuser-container").display = False
        self.dismiss(result)

    def dismiss_due_to_timeout(self) -> None:
        """Close the dialog because the backend ask_user request expired."""
        self._safe_dismiss(None)

    @on(AskUserSubmitted)
    def _on_ask_user_submitted(self, event: AskUserSubmitted) -> None:
        event.stop()
        self._safe_dismiss((event.request_id, event.text))

    @on(AskUserInlineRequested)
    def _on_ask_user_inline_requested(self, event: AskUserInlineRequested) -> None:
        event.stop()
        self._safe_dismiss(AskUserInlineResult(event.request_id, event.draft_text))

    def action_noop(self) -> None:
        """Swallow Esc — user must explicitly respond."""
