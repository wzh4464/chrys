# Copyright (c) 2026 Chrys. All rights reserved.

"""Code-owned ContextGraph memory MCP server attached to every agent build.

The overlay is the single place that decides whether an agent — main, native
sub-agent, or a PACT role host — can reach the team graph. Keeping it here
rather than in profile YAML means the capability follows the deployment
(``CONTEXTGRAPH_NEO4J_URI`` plus one setting) instead of every profile author
remembering to declare it, and a profile that *does* declare a server of the
same name stays authoritative.
"""

from __future__ import annotations

import copy
import os
import sys
from collections.abc import Mapping

from chrys.foundation.config.settings import Settings
from chrys.service.profiles.agents.schema import AgentProfile, MCPServerConfig

MEMORY_MCP_SERVER_NAME = "contextgraph"
MEMORY_MCP_TOOL_NAMES = ("team_memory_health", "team_memory_query", "team_memory_record")

_NEO4J_URI_ENV = "CONTEXTGRAPH_NEO4J_URI"
_MEMORY_MCP_MODULE = "chrys.service.memory.contextgraph_mcp"

# Retrieval fans out over two Neo4j indexes and fuses them, so a cold call can
# be slow; the cap keeps a chatty graph from eating the model's context.
_REQUEST_TIMEOUT_SECONDS = 300
_MAX_TOOL_RESULT_TOKENS = 2000


def memory_mcp_server_config(settings: Settings, env: Mapping[str, str] | None = None) -> MCPServerConfig | None:
    """Return the memory MCP config, or ``None`` when disabled or unconfigured.

    Args:
        settings: Effective settings; ``memory.mcp.enabled`` is the switch.
        env: Environment to read instead of ``os.environ``. Tests inject here.
    """
    if not settings.memory_mcp_enabled:
        return None
    environ: Mapping[str, str] = os.environ if env is None else env
    if not environ.get(_NEO4J_URI_ENV, "").strip():
        return None
    return MCPServerConfig(
        name=MEMORY_MCP_SERVER_NAME,
        transport="stdio",
        command=sys.executable,
        args=["-m", _MEMORY_MCP_MODULE],
        description="ContextGraph team memory (untrusted reference data)",
        allowed_tools=list(MEMORY_MCP_TOOL_NAMES),
        request_timeout=_REQUEST_TIMEOUT_SECONDS,
        max_tool_result_tokens=_MAX_TOOL_RESULT_TOKENS,
        expose_instructions=True,
    )


def apply_memory_overlay(
    profile: AgentProfile, settings: Settings, env: Mapping[str, str] | None = None
) -> AgentProfile:
    """Return *profile* with the memory MCP appended, or *profile* unchanged.

    The caller's object is never mutated: a profile that gains the server is
    returned as a deep copy, so the shared registry entry stays clean and the
    call is safe to repeat. A profile that already declares a server under
    :data:`MEMORY_MCP_SERVER_NAME` is returned as-is — an explicit declaration
    outranks the overlay.
    """
    config = memory_mcp_server_config(settings, env)
    if config is None or any(server.name == config.name for server in profile.tools.mcp):
        return profile
    effective = copy.deepcopy(profile)
    effective.tools.mcp.append(config)
    return effective
