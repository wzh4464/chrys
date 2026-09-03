# Copyright (c) 2026 Chrys. All rights reserved.

"""Isolated worker that deposits one trajectory with ContextGraph's API."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def _fragment_count(store: Any, trajectory_id: str) -> int:
    rows = store.execute_query(
        "MATCH (:Trajectory {id: $trajectory_id})-[:HAS_FRAGMENT]->(fragment:Fragment) RETURN count(fragment) AS count",
        {"trajectory_id": trajectory_id},
    )
    count = rows[0].get("count") if rows else None
    return count if isinstance(count, int) else 0


def _trajectory_count(store: Any) -> int:
    rows = store.execute_query("MATCH (trajectory:Trajectory) RETURN count(trajectory) AS count")
    count = rows[0].get("count") if rows else None
    return count if isinstance(count, int) else 0


def _deposit(payload: dict[str, Any]) -> dict[str, Any]:
    repository = Path(_required_env("CONTEXTGRAPH_REPO")).resolve()
    if not (repository / "agent_memory" / "memory.py").is_file():
        raise RuntimeError(f"CONTEXTGRAPH_REPO is not a ContextGraph checkout: {repository}")
    sys.path.insert(0, str(repository))

    from agent_memory import AgentMemory, RawTrajectory  # ty: ignore[unresolved-import]

    interval = int(os.environ.get("CONTEXTGRAPH_CONSOLIDATE_EVERY", "16"))
    if interval < 1:
        raise ValueError("CONTEXTGRAPH_CONSOLIDATE_EVERY must be positive")
    memory = AgentMemory(
        neo4j_uri=_required_env("NEO4J_URI"),
        neo4j_auth=(_required_env("NEO4J_USER"), os.environ.get("NEO4J_PASSWORD", "")),
        embedding_api_key=_required_env("OPENAI_API_KEY"),
        embedding_base_url=os.environ.get("OPENAI_API_BASE", "").strip() or None,
        embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large"),
        consolidate_every=interval,
    )
    try:
        raw = RawTrajectory(
            instance_id=payload["instance_id"],
            repo=payload["repo"],
            success=payload["success"],
            steps=payload["steps"],
            problem_statement=payload["problem_statement"],
            trajectory_id=payload["trajectory_id"],
        )
        expected_fragments = len(memory.writer._segment_into_fragments(raw))
        existing = memory.store.get_trajectory(raw.trajectory_id)
        if existing is not None:
            actual_fragments = _fragment_count(memory.store, raw.trajectory_id)
            if actual_fragments != expected_fragments:
                raise RuntimeError(
                    f"existing ContextGraph trajectory {raw.trajectory_id} is incomplete: "
                    f"expected {expected_fragments} fragments, found {actual_fragments}"
                )
            return {
                "created": False,
                "fragment_count": actual_fragments,
                "trajectory_id": raw.trajectory_id,
            }

        # AgentMemory normally keeps this counter in a long-lived server. The
        # worker seeds it from the graph so subprocess isolation preserves the
        # repository's configured consolidation cadence.
        memory._trajectory_count = _trajectory_count(memory.store)
        trajectory_id = memory.learn(raw)
        return {
            "created": True,
            "fragment_count": _fragment_count(memory.store, trajectory_id),
            "trajectory_id": trajectory_id,
        }
    finally:
        if memory.store is not None:
            memory.store.close()


def _init_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Create ContextGraph's own constraints and indexes, idempotently.

    The schema belongs to ContextGraph, not to Chrys: re-declaring the labels
    and properties here would fork it silently the next time upstream changes.
    ``init_schema`` is already ``IF NOT EXISTS`` throughout, so this is safe to
    run against a populated graph.
    """
    repository = Path(_required_env("CONTEXTGRAPH_REPO")).resolve()
    if not (repository / "agent_memory" / "neo4j_store.py").is_file():
        raise RuntimeError(f"CONTEXTGRAPH_REPO is not a ContextGraph checkout: {repository}")
    sys.path.insert(0, str(repository))

    from agent_memory.neo4j_store import Neo4jStore  # ty: ignore[unresolved-import]

    dimensions = int(payload.get("vector_dimensions") or 1536)
    store = Neo4jStore(
        uri=_required_env("NEO4J_URI"),
        auth=(_required_env("NEO4J_USER"), os.environ.get("NEO4J_PASSWORD", "")),
    )
    try:
        store.init_schema(vector_dimensions=dimensions)
    finally:
        store.close()
    return {"schema_initialized": True, "vector_dimensions": dimensions}


def main() -> None:
    """Read one JSON request from stdin and write one JSON response."""
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("ContextGraph worker payload must be a JSON object")
    handler = _init_schema if payload.get("op") == "init_schema" else _deposit
    sys.stdout.write(json.dumps(handler(payload), ensure_ascii=True))


if __name__ == "__main__":
    main()
