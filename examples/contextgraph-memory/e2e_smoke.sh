#!/usr/bin/env bash
# Read-only live smoke test for the Chrys -> ContextGraph -> Neo4j path.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CHRYS_REPO="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
SERVER_URL="${CONTEXTGRAPH_SERVER_URL:-http://127.0.0.1:8010}"

cd "$CHRYS_REPO"
CONTEXTGRAPH_SERVER_URL="$SERVER_URL" uv run python - <<'PY'
import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main() -> None:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "chrys.service.memory.contextgraph_mcp"],
        env=dict(os.environ),
    )
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = {tool.name for tool in (await session.list_tools()).tools}
        expected = {"team_memory_health", "team_memory_query"}
        if tools != expected:
            raise SystemExit(f"FAIL: unexpected MCP tool surface: {sorted(tools)}")

        health_result = await session.call_tool("team_memory_health", {})
        health = "\n".join(getattr(block, "text", "") for block in health_result.content)
        if "neo4j_connected=true" not in health:
            raise SystemExit(f"FAIL: {health}")

        query_result = await session.call_tool(
            "team_memory_query",
            {
                "query": "How should I implement a cross-cutting compiler feature safely and verify it?",
                "top_k": 5,
            },
        )
        result = "\n".join(getattr(block, "text", "") for block in query_result.content)
        if "UNTRUSTED DATA" not in result:
            raise SystemExit(f"FAIL: no ContextGraph rules returned: {result}")

        print(health)
        print(result)
        print("PASS: MCP stdio returned bounded read-only ContextGraph advisory data.")


asyncio.run(main())
PY
