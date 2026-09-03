# Copyright (c) 2026 Chrys. All rights reserved.

"""The code-owned ContextGraph memory MCP overlay applied to every agent build."""

from __future__ import annotations

import sys

from chrys.foundation.config.settings import Settings
from chrys.service.memory.overlay import (
    MEMORY_MCP_SERVER_NAME,
    MEMORY_MCP_TOOL_NAMES,
    apply_memory_overlay,
    memory_mcp_server_config,
)
from chrys.service.profiles.agents.schema import AgentProfile, MCPServerConfig

_URI_ENV = {"CONTEXTGRAPH_NEO4J_URI": "bolt://127.0.0.1:7687"}


def _settings(enabled: bool) -> Settings:
    return Settings(memory_mcp_enabled=enabled)


def test_config_requires_the_graph_uri() -> None:
    assert memory_mcp_server_config(_settings(True), env={}) is None
    assert memory_mcp_server_config(_settings(True), env={"CONTEXTGRAPH_NEO4J_URI": "   "}) is None

    config = memory_mcp_server_config(_settings(True), env=_URI_ENV)

    assert config is not None
    assert config.name == MEMORY_MCP_SERVER_NAME
    assert config.transport == "stdio"
    assert config.command == sys.executable
    assert config.args == ["-m", "chrys.service.memory.contextgraph_mcp"]
    assert config.allowed_tools == list(MEMORY_MCP_TOOL_NAMES)
    assert config.expose_instructions is True
    assert config.max_tool_result_tokens == 2000


def test_disabled_setting_returns_none() -> None:
    assert memory_mcp_server_config(_settings(False), env=_URI_ENV) is None


def test_apply_is_idempotent_and_leaves_the_source_profile_alone() -> None:
    profile = AgentProfile(name="P")

    once = apply_memory_overlay(profile, _settings(True), env=_URI_ENV)
    twice = apply_memory_overlay(once, _settings(True), env=_URI_ENV)

    assert profile.tools.mcp == []
    assert [server.name for server in once.tools.mcp] == [MEMORY_MCP_SERVER_NAME]
    assert [server.name for server in twice.tools.mcp] == [MEMORY_MCP_SERVER_NAME]
    assert twice is once


def test_apply_is_a_no_op_when_unconfigured() -> None:
    profile = AgentProfile(name="P")

    assert apply_memory_overlay(profile, _settings(True), env={}) is profile
    assert apply_memory_overlay(profile, _settings(False), env=_URI_ENV) is profile


def test_an_explicit_profile_server_of_the_same_name_wins() -> None:
    profile = AgentProfile(name="P")
    profile.tools.mcp.append(MCPServerConfig(name=MEMORY_MCP_SERVER_NAME, transport="stdio", command="python"))

    out = apply_memory_overlay(profile, _settings(True), env=_URI_ENV)

    assert out is profile
    assert len(out.tools.mcp) == 1
    assert out.tools.mcp[0].command == "python"


def test_overlay_preserves_servers_the_profile_already_declares() -> None:
    profile = AgentProfile(name="P")
    profile.tools.mcp.append(MCPServerConfig(name="other", transport="stdio", command="python"))

    out = apply_memory_overlay(profile, _settings(True), env=_URI_ENV)

    assert [server.name for server in out.tools.mcp] == ["other", MEMORY_MCP_SERVER_NAME]
