# Copyright (c) 2026 Chrys. All rights reserved.

"""Shared base class for dismissable modal dialogs."""

from __future__ import annotations

from typing import Any, TypeVar

from textual.await_complete import AwaitComplete
from textual.screen import ModalScreen

from chrys.app.tui.behaviors.click_outside_dismiss import ClickOutsideDismissMixin
from chrys.app.tui.behaviors.right_click_copy import RightClickScreenCopyMixin

DialogResultT = TypeVar("DialogResultT")


class BaseDialog(RightClickScreenCopyMixin, ClickOutsideDismissMixin, ModalScreen[DialogResultT]):
    """Base for modal dialogs that may dismiss from a backdrop click.

    Keep this under ``screens/dialogs`` because it subclasses
    :class:`textual.screen.ModalScreen`. Non-dismissable dialogs can either
    pass ``dismiss_on_backdrop=False`` or override
    ``_allow_click_outside_dismiss`` for state-dependent gates.
    """

    def __init__(self, *args: Any, dismiss_on_backdrop: bool = True, **kwargs: Any) -> None:
        self._dismiss_on_backdrop = dismiss_on_backdrop
        super().__init__(*args, **kwargs)

    def dismiss(self, result: object | None = None) -> AwaitComplete:
        """Dismiss once, running the pre-dismiss hook before first close."""
        if self._chrys_dismiss_await_complete is None:
            self._before_dismiss(result)
        return super().dismiss(result)

    def _allow_click_outside_dismiss(self) -> bool:
        return self._dismiss_on_backdrop

    def _before_dismiss(self, _result: object | None = None) -> None:
        """Hook for subclasses that need cleanup before the modal closes."""
        return
