#!/usr/bin/env bash
# Closed-loop live smoke test for the Chrys MCP -> Neo4j path.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CHRYS_REPO="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"

cd "$CHRYS_REPO"
uv run python - <<'PY'
import asyncio
import os
import re
import sys
import uuid

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from neo4j import GraphDatabase, WRITE_ACCESS


def cleanup(trajectory_id: str) -> None:
    uri = os.environ.get("CONTEXTGRAPH_NEO4J_URI", "bolt://127.0.0.1:7705")
    user = os.environ.get("CONTEXTGRAPH_NEO4J_USER", os.environ.get("NEO4J_USER", "neo4j"))
    password = os.environ.get("CONTEXTGRAPH_NEO4J_PASSWORD", os.environ.get("NEO4J_PASSWORD", ""))
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(default_access_mode=WRITE_ACCESS) as neo4j:
            neo4j.run(
                "MATCH (:Trajectory {id: $id})-[:HAS_FRAGMENT]->(fragment:Fragment) DETACH DELETE fragment",
                {"id": trajectory_id},
            ).consume()
            neo4j.run("MATCH (trajectory:Trajectory {id: $id}) DETACH DELETE trajectory", {"id": trajectory_id}).consume()
    finally:
        driver.close()


async def main() -> None:
    server_env = dict(os.environ)
    server_env["CONTEXTGRAPH_CONSOLIDATE_EVERY"] = "1000000000"
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "chrys.service.memory.contextgraph_mcp"],
        env=server_env,
    )
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = {tool.name for tool in (await session.list_tools()).tools}
        expected = {"team_memory_health", "team_memory_query", "team_memory_record"}
        if tools != expected:
            raise SystemExit(f"FAIL: unexpected MCP tool surface: {sorted(tools)}")

        health_result = await session.call_tool("team_memory_health", {})
        health = "\n".join(getattr(block, "text", "") for block in health_result.content)
        if "canonical_rules=" not in health:
            raise SystemExit(f"FAIL: {health}")

        trajectory_id = ""
        try:
            marker = f"chrys-dynamic-smoke-{uuid.uuid4().hex}"
            record_result = await session.call_tool(
                "team_memory_record",
                {
                    "problem_statement": marker,
                    "success": True,
                    "steps": [
                        {
                            "action": f"run focused dynamic-memory smoke test {marker}",
                            "observation": "the exact marker was deposited and retrieved",
                        }
                    ],
                    "repo": "chrys-smoke",
                },
            )
            record_text = "\n".join(getattr(block, "text", "") for block in record_result.content)
            match = re.search(r"traj_chrys_[0-9a-f]{24}", record_text)
            if match is None:
                raise SystemExit(f"FAIL: ContextGraph trajectory was not created: {record_text}")
            trajectory_id = match.group(0)
            query_result = await session.call_tool(
                "team_memory_query",
                {
                    "query": marker,
                    "top_k": 20,
                },
            )
            result = "\n".join(getattr(block, "text", "") for block in query_result.content)
            if "UNTRUSTED DATA" not in result or marker not in result:
                raise SystemExit(f"FAIL: ContextGraph trajectory fragment was not recalled: {result}")

            print(health)
            print(record_text)
            print(result)
            print("PASS: MCP stdio completed ContextGraph repository write -> fragment recall.")
        finally:
            if trajectory_id:
                cleanup(trajectory_id)


asyncio.run(main())
PY
