# Copyright (c) 2026 Chrys. All rights reserved.

"""``/longrun``, ``/quick``, ``/route`` and the routing announcement."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from chrys.app.tui.screens.main.buddy_command import BuddyCommandController
from chrys.app.tui.screens.main.commands import MainSlashCommandRegistry, SlashCommandDef


class _Actions:
    """Records what a slash command asked for, in order."""

    def __init__(self, *, status: str = "global mode: auto") -> None:
        self.overrides: list[tuple[str, bool]] = []
        self.prompts: list[str] = []
        self.warnings: list[Any] = []
        self._status = status

    def set_route_override(self, track: str, *, reroute: bool = False) -> None:
        self.overrides.append((track, reroute))

    def submit_prompt(self, text: str) -> None:
        self.prompts.append(text)

    def route_status(self) -> str:
        return self._status

    def notify_warning(self, message: Any, *, title: Any = None, timeout: float | None = 3) -> None:
        self.warnings.append(message)

    def __getattr__(self, name: str) -> Any:
        # Every other port member is irrelevant here.
        return MagicMock()


def _command(actions: _Actions, name: str) -> SlashCommandDef:
    registry = MainSlashCommandRegistry(actions=actions, buddy=BuddyCommandController(MagicMock()))
    return next(command for command in registry.build() if command.name == name)


def test_longrun_without_text_only_arms_the_override() -> None:
    actions = _Actions()

    _command(actions, "longrun").action("")

    assert actions.overrides == [("long_horizon", False)]
    assert actions.prompts == []


def test_longrun_with_text_arms_then_submits_in_that_order() -> None:
    actions = _Actions()

    _command(actions, "longrun").action("add OAuth login across api and web")

    assert actions.overrides == [("long_horizon", False)]
    assert actions.prompts == ["add OAuth login across api and web"]


def test_quick_arms_the_standard_track() -> None:
    actions = _Actions()

    _command(actions, "quick").action("just fix the typo")

    assert actions.overrides == [("standard", False)]
    assert actions.prompts == ["just fix the typo"]


def test_quick_is_accepted_while_the_agent_runs() -> None:
    """Its whole purpose is catching a long-horizon turn mid-preparation."""
    actions = _Actions()

    assert _command(actions, "quick").allow_while_running is True


def test_longrun_is_not_accepted_while_the_agent_runs() -> None:
    """Promoting a turn that has already started would have nothing to promote."""
    actions = _Actions()

    assert _command(actions, "longrun").allow_while_running is False


@pytest.mark.parametrize("argument", ["", "show", "  SHOW  "])
def test_route_reports_status(argument: str) -> None:
    actions = _Actions(status="global mode: auto\nlast turn: strong_long_horizon")

    _command(actions, "route").action(argument)

    assert actions.overrides == []
    assert actions.warnings == ["global mode: auto\nlast turn: strong_long_horizon"]


def test_route_reroute_clears_the_inherited_decision() -> None:
    actions = _Actions()

    _command(actions, "route").action("reroute")

    assert actions.overrides == [("", True)]


def test_route_rejects_an_unknown_argument_without_arming_anything() -> None:
    actions = _Actions()

    _command(actions, "route").action("sideways")

    assert actions.overrides == []
    assert actions.warnings
