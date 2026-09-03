# Copyright (c) 2026 Chrys. All rights reserved.

"""Per-profile ACP nesting depth cap."""

from __future__ import annotations

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import Warning
from chrys.foundation.models.session_env import SessionEnvironment
from chrys.foundation.platform import get_platform
from chrys.orchestration.sub_agents.tools import SubAgentTools
from chrys.service.profiles.agents.loader import AgentProfileLoadError, load_profile_from_yaml
from chrys.service.profiles.agents.schema import AcpAgentConfig, AgentProfile, SubAgentRef


def _profile(max_depth: int = 1) -> AgentProfile:
    return AgentProfile(
        name="ChrysPact",
        sub_agent_only=True,
        acp=AcpAgentConfig(command="python", args=["-c", "pass"], max_depth=max_depth),
    )


async def _register(tmp_path, *, max_depth: int, bus: EventBus) -> SubAgentTools:
    tools = SubAgentTools(max_total_concurrency=3, event_bus=bus, session_id="s", workspace_cwd=str(tmp_path))
    runtime = SessionEnvironment(cwd=str(tmp_path), platform=get_platform())
    await tools.register_acp(SubAgentRef(profile="ChrysPact", tool_name="chrys_pact"), _profile(max_depth), runtime)
    return tools


async def test_registration_succeeds_at_the_top_level(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHRYS_ACP_SUBAGENT_DEPTH", raising=False)

    tools = await _register(tmp_path, max_depth=1, bus=EventBus())

    assert "chrys_pact" in tools.tool_names()


async def test_a_nested_agent_does_not_register_the_tool(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A PACT role must not be able to start another campaign."""
    monkeypatch.setenv("CHRYS_ACP_SUBAGENT_DEPTH", "1")
    bus = EventBus()
    warnings: list[Warning] = []
    await bus.subscribe(Warning, warnings.append)

    tools = await _register(tmp_path, max_depth=1, bus=bus)

    assert "chrys_pact" not in tools.tool_names()
    assert warnings and warnings[-1].code == "acp_sub_agent_depth_exceeded"


async def test_a_profile_may_allow_one_more_level(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_ACP_SUBAGENT_DEPTH", "1")

    tools = await _register(tmp_path, max_depth=2, bus=EventBus())

    assert "chrys_pact" in tools.tool_names()


async def test_the_cap_still_applies_at_the_deeper_level(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_ACP_SUBAGENT_DEPTH", "2")

    tools = await _register(tmp_path, max_depth=2, bus=EventBus())

    assert "chrys_pact" not in tools.tool_names()


def test_max_depth_defaults_to_one(tmp_path) -> None:
    path = tmp_path / "Acp.yaml"
    path.write_text(
        "name: Acp\ndescription: Acp\nsub_agent_only: true\nacp:\n  command: chrys\n  args: [pact-agent]\n",
        encoding="utf-8",
    )

    profile = load_profile_from_yaml(path)

    assert profile.acp is not None
    assert profile.acp.max_depth == 1


@pytest.mark.parametrize("value", ["0", "-1", "1.5", '"two"'])
def test_an_invalid_max_depth_is_rejected(tmp_path, value: str) -> None:
    path = tmp_path / "Acp.yaml"
    path.write_text(
        "name: Acp\ndescription: Acp\nsub_agent_only: true\nacp:\n  command: chrys\n"
        f"  args: [pact-agent]\n  max_depth: {value}\n",
        encoding="utf-8",
    )

    with pytest.raises(AgentProfileLoadError):
        load_profile_from_yaml(path)
