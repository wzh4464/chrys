# ContextGraph read-only memory integration

This example ports the ContextGraph integration from the earlier Chrys repository
to the current architecture. Chrys owns a small MCP bridge; ContextGraph continues
to own Neo4j, embeddings, and retrieval. The validated graph is queried read-only:
there is no `/learn`, session-end recording, graph construction, or consolidation
in this integration.

The reference selected-Harbor snapshot was built from agent-visible trajectory and
verifier evidence only—no gold patches, reference solutions, or hidden tests. Its
validated inventory is:

| Item | Count |
| --- | ---: |
| Parseable trajectories | 1,463 |
| Fragments | 75,193 |
| Error patterns | 1,076 |
| Strategy trajectories sampled with DeepSeek V4 Pro | 534 |
| Tasks covered | 49 |
| Strategies | 2,607 |
| Canonical rules after deduplication | 2,525 |
| `MERGED_INTO` links | 2,607 |
| `ADDRESSES_ERROR` links | 21,173 |

Every strategy is linked to its source trajectory, a canonical rule, and error
patterns. The 15 source records omitted from the graph had no executable agent
steps. The snapshot is expected at `bolt://127.0.0.1:7705` by default and was built
with ContextGraph's `scripts/capbench/build_selected_harbor_graph.py`.

## Architecture

1. `contextgraph_pro_server.py` runs from a ContextGraph checkout, connects to the
   existing Neo4j graph, embeds the task query, and retrieves `CanonicalRule` nodes.
2. `chrys.service.memory.contextgraph_mcp` calls its loopback
   `/query_memory_items` endpoint and exposes `team_memory_health` and
   `team_memory_query` over MCP stdio.
3. The MCP bridge removes control characters, deduplicates items, caps each item
   and the combined result, and frames all retrieved content as untrusted data.

The query API is read-only. ContextGraph's normal `AgentMemory` initialization may
idempotently ensure indexes exist, but this workflow creates no trajectory,
fragment, strategy, or canonical-rule data.

## Prerequisites

- A ContextGraph checkout with `scripts/baselines/contextgraph_pro_server.py` and
  dependencies installed with `uv sync --extra dev`.
- The validated Neo4j graph already running (default Bolt URI above).
- The same OpenAI-compatible embedding model used during construction:
  `text-embedding-3-large`, dimension 3,072.
- `curl` for the optional POSIX startup hook.

Keep Neo4j and embedding credentials in the ContextGraph checkout's `.env` or
`~/.chrys/.env`; never put real values in the profile or repository.

## Install

1. Copy `env.example` values into `~/.chrys/.env` and replace placeholders.
2. Copy `Memory.yaml` to `~/.chrys/agents/Memory.yaml`. If `python` there does
   not resolve to the Chrys runtime, replace it with that runtime's absolute
   executable path.
3. Either start the query service manually from the ContextGraph checkout:

   ```bash
   NEO4J_URI=bolt://127.0.0.1:7705 \
     uv run python scripts/baselines/contextgraph_pro_server.py \
       --host 127.0.0.1 --port 8010
   ```

   Or merge `hooks/hooks.yaml` into one hooks layer and copy
   `hooks/start_query_service.sh` beside it under `scripts/`. The hook starts
   only the query layer and never manages the Neo4j process.
4. Launch Chrys with `chrys -a Memory`, or run headless with
   `chrys run "..." --agent Memory`.

Install each hook id in exactly one layer. If the same id appears in both
`~/.chrys/hooks/` and `<project>/.chrys/hooks/`, both copies run.

## Verify

Unit tests do not require ContextGraph or Neo4j:

```bash
uv run pytest tests/service/memory/test_contextgraph_mcp.py -n0
```

With the graph and query service running, exercise the complete read path:

```bash
./examples/contextgraph-memory/e2e_smoke.sh
```

The smoke test performs no writes. It requires the health endpoint to report
`neo4j_connected=true` and rejects a response that lacks the MCP bridge's
untrusted-data frame.
