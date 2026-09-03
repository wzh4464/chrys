# Copyright (c) 2026 Chrys. All rights reserved.

"""The memory MCP overlay reaches the main agent build and every native sub-agent."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.orchestration.engine.build import builder as ab
from chrys.orchestration.engine.build import construction as agent_lifecycle
from chrys.service.memory.overlay import MEMORY_MCP_SERVER_NAME
from chrys.service.profiles.agents.schema import (
    AgentProfile,
    MCPServerConfig,
    SubAgentRef,
    SubAgentsConfig,
)
from chrys.service.profiles.models.resolver import ModelSelection
from chrys.service.profiles.models.schema import ModelProfile

_URI = "bolt://127.0.0.1:7687"


def _server_names(profile: AgentProfile) -> list[str]:
    return [server.name for server in profile.tools.mcp]


class _BareExecutor:
    """Enough of an Executor for ``AgentEngine.start`` to reach a built state."""

    def __init__(self) -> None:
        self.agent = MagicMock()

    async def cleanup(self) -> None:
        return None


async def test_main_agent_build_receives_the_memory_mcp(
    monkeypatch: pytest.MonkeyPatch,
    agent_engine,
) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    seen: list[AgentProfile] = []

    async def _capture(_host, profile: AgentProfile, **_kwargs: object) -> None:
        seen.append(profile)

    monkeypatch.setattr(agent_lifecycle, "build_agent", _capture)

    engine = agent_engine(EventBus(), settings=Settings())
    await engine._build_agent(AgentProfile(name="Code"), staged=MagicMock())

    assert seen and _server_names(seen[0]) == [MEMORY_MCP_SERVER_NAME]


async def test_memory_and_ephemeral_overlays_compose(
    monkeypatch: pytest.MonkeyPatch,
    agent_engine,
) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    ephemeral = [MCPServerConfig(name="ephemeral", transport="stdio", command="python")]
    engine = agent_engine(EventBus(), settings=Settings(), mcp_overlay=ephemeral)

    source = AgentProfile(name="Code")
    effective = engine._profile_with_mcp_overlay(source)

    assert _server_names(effective) == [MEMORY_MCP_SERVER_NAME, "ephemeral"]
    assert source.tools.mcp == []


async def test_disabled_setting_leaves_the_build_profile_untouched(
    monkeypatch: pytest.MonkeyPatch,
    agent_engine,
) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    engine = agent_engine(EventBus(), settings=Settings(memory_mcp_enabled=False))

    source = AgentProfile(name="Code")

    assert engine._profile_with_mcp_overlay(source) is source


@contextmanager
def _stub_build_dependencies(register: AsyncMock):
    """Patch every external dependency ``build_agent`` needs to reach registration."""
    tool_registry = MagicMock()
    tool_registry.get_all.return_value = []
    tool_registry.load_builtins = MagicMock()

    context = MagicMock()
    context.providers = []
    context.middleware = []
    context.compaction_strategy = MagicMock()

    sub_agent_tools = MagicMock()
    sub_agent_tools.register = register
    sub_agent_tools.register_acp = AsyncMock()
    sub_agent_tools.get_tools.return_value = []
    sub_agent_tools.cleanup = AsyncMock()

    with (
        patch.object(ab, "Agent", return_value=MagicMock()),
        patch.object(ab, "ContextManager", return_value=context),
        patch.object(ab, "create_client", return_value=MagicMock()),
        patch.object(
            ab,
            "resolve_selection_for_agent",
            return_value=ModelSelection(ModelProfile(id="test-id", name="test"), "override"),
        ),
        patch.object(ab, "effective_chat_options", return_value={}),
        patch.object(ab, "LoopRecorder", return_value=MagicMock()),
        patch.object(ab, "SystemReminderMiddleware", MagicMock(return_value=MagicMock())),
        patch.object(ab, "LastWordsGenerator", MagicMock(return_value=MagicMock())),
        patch("chrys.service.tools.registry.ToolRegistry", return_value=tool_registry),
        patch("chrys.service.skills.adapter.create_skills_provider", new=AsyncMock(return_value=(None, []))),
        patch("chrys.service.mcp.adapter.MCPAdapter"),
        patch("chrys.orchestration.sub_agents.tools.SubAgentTools", return_value=sub_agent_tools),
    ):
        yield


async def _build_with_sub_agent(settings: Settings) -> AsyncMock:
    register = AsyncMock()
    parent = AgentProfile(
        name="Parent",
        sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Helper", tool_name="helper_agent")]),
    )
    registry = MagicMock()
    registry.get.return_value = AgentProfile(name="Helper")

    async def _async_noop(*_a: object, **_k: object) -> None:
        return None

    def _sync_noop(*_a: object, **_k: object) -> None:
        return None

    progress: Callable[..., Awaitable[None]] = _async_noop
    with _stub_build_dependencies(register):
        await ab.build_agent(
            profile=parent,
            settings=settings,
            workspace=None,
            session_id=None,
            bus=EventBus(),
            agent_registry=registry,
            existing_sub_agent_tools=None,
            existing_mcp_adapter=None,
            injection=MagicMock(),
            intermediate_buffer=MagicMock(),
            on_intermediate_async=_async_noop,
            on_intermediate_sync=_sync_noop,
            on_usage=_sync_noop,
            on_load_progress=progress,
            spill_quota=None,
            persist_recovery_now=None,
            on_side_call_usage=None,
            allow_user_interaction=True,
        )
    return register


async def test_native_sub_agent_registration_receives_the_memory_mcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)

    register = await _build_with_sub_agent(Settings())

    assert register.await_count == 1
    registered_profile = register.await_args.args[1]
    assert _server_names(registered_profile) == [MEMORY_MCP_SERVER_NAME]


async def test_sub_agent_registration_is_untouched_when_memory_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)

    register = await _build_with_sub_agent(Settings(memory_mcp_enabled=False))

    assert register.await_count == 1
    assert _server_names(register.await_args.args[1]) == []
