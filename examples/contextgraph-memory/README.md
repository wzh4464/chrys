# ContextGraph read-only memory integration

This example ports the ContextGraph integration from the earlier Chrys repository
to the current architecture. Chrys includes the Neo4j driver and queries the
validated graph directly over Bolt. No ContextGraph checkout or HTTP query service
is required at runtime, and the MCP surface contains no learning or write tool.

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

1. `chrys.service.memory.contextgraph_mcp` connects to Neo4j with the pinned
   `neo4j` dependency and opens every query session with `READ_ACCESS`.
2. The vector channel embeds the task with `text-embedding-3-large` and queries
   `canonical_rule_embedding`. The BM25 channel queries `canonical_rule_text`.
3. Reciprocal-rank fusion combines both result lists. If embeddings are not
   configured or one channel fails, the other channel continues independently.
4. The bridge removes control characters, deduplicates rules, caps each item and
   the combined result, frames it as untrusted data, and exposes only
   `team_memory_health` and `team_memory_query` over MCP stdio.

The code issues only `MATCH`, vector-index, and full-text-index queries. It does
not initialize schemas or create trajectories, fragments, strategies, rules,
relationships, constraints, or indexes.

## Prerequisites

- The validated Neo4j graph already running at the configured Bolt URI.
- Neo4j credentials in `~/.chrys/.env`.
- For vector retrieval, an OpenAI-compatible endpoint serving the same embedding
  model used during construction: `text-embedding-3-large`, dimension 3,072.
  Without it, BM25 retrieval remains available.

The published Chrys wheel now carries the Neo4j driver. Keep database and embedding
credentials out of profiles and repositories.

## Install

1. Copy `env.example` values into `~/.chrys/.env` and replace placeholders.
2. Copy `Memory.yaml` to `~/.chrys/agents/Memory.yaml`. If `python` there does
   not resolve to the Chrys runtime, replace it with that runtime's absolute
   executable path.
3. Launch Chrys with `chrys -a Memory`, or run headless with
   `chrys run "..." --agent Memory`.

## Verify

Unit tests do not require a live Neo4j instance:

```bash
uv run pytest tests/service/memory/test_contextgraph_mcp.py -o addopts=""
```

With the graph running and credentials exported, exercise the complete MCP path:

```bash
./examples/contextgraph-memory/e2e_smoke.sh
```

The smoke test performs no writes. It requires health to report the canonical-rule
inventory and rejects a query response lacking the bridge's untrusted-data frame.
