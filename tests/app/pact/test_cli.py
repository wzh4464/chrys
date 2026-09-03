# Copyright (c) 2026 Chrys. All rights reserved.

"""Fail-fast command-line configuration tests for Chrys-PACT."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from chrys.pact import cli
from chrys.service.profiles.agents.schema import AcpAgentConfig, AgentProfile


@dataclass
class _FakeRegistry:
    profile: AgentProfile | None
    loaded: bool = False

    def load_all(self) -> int:
        self.loaded = True
        return 1 if self.profile is not None else 0

    def resolve_selector(self, selector: str) -> AgentProfile | None:
        assert self.loaded
        _ = selector
        return self.profile


def test_profile_selector_is_resolved_to_canonical_in_process_name(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = _FakeRegistry(AgentProfile(name="Code", id="profile-code"))
    monkeypatch.setattr(cli, "AgentProfileRegistry", lambda: registry)

    assert cli._resolve_profile_name("code") == "Code"


def test_unknown_profile_fails_before_server_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "AgentProfileRegistry", lambda: _FakeRegistry(None))

    with pytest.raises(ValueError, match="Agent profile not found: typo"):
        cli._resolve_profile_name("typo")


def test_external_acp_profile_cannot_be_used_for_inner_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = AgentProfile(
        name="Remote",
        id="profile-remote",
        acp=AcpAgentConfig(command="remote-agent"),
    )
    monkeypatch.setattr(cli, "AgentProfileRegistry", lambda: _FakeRegistry(profile))

    with pytest.raises(ValueError, match="in-process Chrys profile"):
        cli._resolve_profile_name("Remote")


# ── --verify-from-settings ───────────────────────────────────────────


def test_verify_from_settings_is_mutually_exclusive_with_verify() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--verify", "pytest", "--verify-from-settings"])


def test_a_verification_choice_is_still_required() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


async def test_verify_from_settings_reads_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}
    _patch_run_command(monkeypatch, captured, verify_command="uv run pytest -q")

    code = await cli.run_command(cli.build_parser().parse_args(["--verify-from-settings"]))

    assert code == 0
    assert captured["verify_command"] == "uv run pytest -q"


async def test_verify_from_settings_fails_closed_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """A campaign that cannot verify must not start pretending it can."""
    captured: dict[str, object] = {}
    _patch_run_command(monkeypatch, captured, verify_command="")

    with pytest.raises(ValueError, match=r"pact\.verify_command"):
        await cli.run_command(cli.build_parser().parse_args(["--verify-from-settings"]))

    assert captured == {}


def _patch_run_command(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, object],
    *,
    verify_command: str,
) -> None:
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.config.settings_store import LoadedSettings

    loaded = LoadedSettings(settings=Settings(pact_verify_command=verify_command), provenance={})
    monkeypatch.setattr(cli, "_prepare_runtime", lambda: loaded)
    monkeypatch.setattr(cli, "_resolve_profile_name", lambda selector: selector)

    class _Server:
        async def shutdown(self) -> None:
            return None

    def _default_server(**kwargs: object) -> _Server:
        captured.update(kwargs)
        return _Server()

    async def _run_agent(_server: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(cli, "default_server", _default_server)
    monkeypatch.setattr(cli.acp_sdk, "run_agent", _run_agent)
