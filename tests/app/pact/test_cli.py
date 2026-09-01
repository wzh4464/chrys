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
