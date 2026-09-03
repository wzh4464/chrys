# ContextGraph static retrieval + repository-owned dynamic deposition

This example ports the earlier Chrys ContextGraph integration to the current
architecture. Chrys carries the Neo4j dependency and queries the graph directly
over Bolt; dynamic experience is not reimplemented in Chrys. Completed turns are
passed to the configured ContextGraph checkout, whose `AgentMemory.learn` pipeline
owns `Trajectory`, `Fragment`, `ErrorPattern`, entity resolution, community
assignment, and periodic consolidation.

The selected-Harbor graph remains the initial validated knowledge layer:

- 1,463 parseable trajectories, 75,193 fragments, and 1,076 error patterns.
- 534 DeepSeek V4 Pro experience-layer trajectories across all 49 task IDs.
- 2,607 strategies and 2,525 `CanonicalRule` nodes.
- Every strategy is linked to its source trajectory, canonical rule, and error
  pattern; there are no orphan strategies.
- The 15 excluded records have no executable agent steps.

## Runtime flow

Retrieval uses four ContextGraph-owned indexes:

1. `canonical_rule_embedding` and `canonical_rule_text` retrieve the validated
   strategy layer.
2. `fragment_embedding` and `fragment_description` retrieve trajectories deposited
   from Chrys turns.

Reciprocal-rank fusion combines the channels. Results are bounded, stripped of
terminal controls, and framed as untrusted advisory data.

Dynamic deposition has two entry points:

1. The recommended durable `after_turn` hook reads the already-persisted
   `session.json`, resolves the turn with Chrys's canonical turn grammar, pairs tool
   calls/results with the canonical exchange grammar, and submits only turns with
   at least one completed non-memory tool call.
2. `team_memory_record` submits a deliberately curated action/observation list and
   requires approval in the example profile.

Both entry points invoke `_contextgraph_repository_worker.py` with the ContextGraph
checkout's Python environment. The worker imports `agent_memory` from that checkout
and calls `AgentMemory.learn`; Chrys does not define a parallel graph schema. Stable
trajectory IDs make durable replay a no-op. Because the worker is isolated per
write, it seeds ContextGraph's in-memory consolidation counter from the current
trajectory count before calling `learn`, preserving the configured cadence.

## Prerequisites

- The validated Neo4j graph running at the configured Bolt URI.
- A local ContextGraph checkout (default: `~/codes/ContextGraph`).
- The ContextGraph checkout's `.venv`, or an explicit `CONTEXTGRAPH_PYTHON`.
- Neo4j and embedding credentials in `~/.chrys/.env`.
- The same embedding model used by the graph (`text-embedding-3-large` for the
  supplied selected-Harbor graph).

The published Chrys wheel carries `neo4j==6.1.0`; direct retrieval needs no HTTP
query service. The checkout is required only for the repository-owned write path.

## Install

1. Copy the relevant values from `env.example` into `~/.chrys/.env`. Set
   `CONTEXTGRAPH_DYNAMIC_DEPOSIT=1` to opt into automatic writes.
2. Copy `Memory.yaml` to `~/.chrys/agents/Memory.yaml`. If its `python` command does
   not resolve to the Chrys environment, replace it with that interpreter's
   absolute path.
3. Install `hooks/hooks.yaml` in exactly one hook layer: merge it into
   `~/.chrys/hooks/hooks.yaml`, or place it at
   `<project>/.chrys/hooks/hooks.yaml`.
4. Launch `chrys -a Memory`, or run headless with
   `chrys run "..." --agent Memory`.

Do not install the same hook ID globally and per project. If sessions live under a
custom storage root that the hook subprocess cannot resolve, set
`CONTEXTGRAPH_SESSION_ROOT_DIR` to the root containing `sessions/`.

## Safety model

- Canonical-rule retrieval is read-only from Chrys.
- Automatic deposition is explicit opt-in and restricted to the `Memory` profile.
- A normal `after_turn` status is runtime completion, not verifier-confirmed task
  correctness; it is stored as ContextGraph trajectory success/failure, not promoted
  directly into `CanonicalRule`.
- Memory MCP calls are excluded from extracted steps, preventing recursive
  deposition of retrieved memory.
- Secret redaction is best effort. Never place credentials in prompts, tool
  arguments, or curated records.

## Verify

Unit tests do not require live Neo4j or a ContextGraph checkout:

```bash
uv run pytest tests/service/memory -o addopts=""
```

With Neo4j, credentials, the embedding endpoint, and `CONTEXTGRAPH_REPO` available,
run the closed-loop smoke test:

```bash
./examples/contextgraph-memory/e2e_smoke.sh
```

The smoke test creates one ContextGraph `Trajectory`, proves MCP write → fragment
recall, then deletes exactly that trajectory and its fragments.
