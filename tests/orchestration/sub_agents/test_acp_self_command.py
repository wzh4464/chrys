# Copyright (c) 2026 Chrys. All rights reserved.

"""``command: chrys`` resolves to this very executable."""

from __future__ import annotations

import sys

import pytest

from chrys.orchestration.sub_agents.tools import _resolve_self_command


def test_a_source_checkout_runs_the_package_through_this_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYAPP", raising=False)
    monkeypatch.delenv("CHRYS_FORCE_FROZEN", raising=False)

    command, args = _resolve_self_command("chrys", ("pact-agent", "--agent", "Code"))

    assert command == sys.executable
    assert args == ("-m", "chrys", "pact-agent", "--agent", "Code")


def test_a_packaged_runtime_runs_its_own_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_FORCE_FROZEN", "1")
    monkeypatch.setattr(sys, "argv", ["/opt/chrys/bin/chrys", "acp"])

    command, args = _resolve_self_command("chrys", ("pact-agent",))

    assert command == "/opt/chrys/bin/chrys"
    assert args == ("pact-agent",)


def test_any_other_command_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the built-in spelling is special; a user's path must not be rewritten."""
    monkeypatch.delenv("PYAPP", raising=False)

    command, args = _resolve_self_command("/usr/local/bin/other-agent", ("--serve",))

    assert command == "/usr/local/bin/other-agent"
    assert args == ("--serve",)


def test_a_packaged_runtime_without_argv_falls_back_to_the_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHRYS_FORCE_FROZEN", "1")
    monkeypatch.setattr(sys, "argv", [])

    command, args = _resolve_self_command("chrys", ("pact-agent",))

    assert command == sys.executable
    assert args == ("-m", "chrys", "pact-agent")


def test_the_builtin_pact_profile_uses_the_self_command() -> None:
    from chrys.service.profiles.agents.registry import AgentProfileRegistry

    registry = AgentProfileRegistry()
    registry.load_all()
    profile = registry.get("ChrysPact")

    assert profile is not None
    assert profile.acp is not None
    assert profile.acp.command == "chrys"
    assert profile.acp.args == ["pact-agent", "--agent", "Code", "--max-rounds", "2", "--verify-from-settings"]
    assert profile.acp.max_depth == 1
    assert profile.acp.idle_timeout_seconds == 0
