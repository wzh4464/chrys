# The long-horizon suite

Chrys normally answers a prompt with one agent loop: read, edit, run, reply. The
long-horizon suite adds a second track for tasks that are too big for that loop to
carry safely: a feature that touches many files, a migration with acceptance
criteria, a change whose requirement is only half written down. On that track a turn
becomes a small campaign — the workspace is frozen, the requirement is clarified
against the code, the relevant code is located, a baseline and a repair pass are run,
and the governed part of the work is handed to a PACT campaign that a separate ACP
agent executes under review. Everything the turn learned is deposited into a team
memory graph, and the next turn on the same repository recalls it.

This document is the map: what each piece is, how a turn flows through them, how
they are invoked (CLI, TUI and ACP), what to configure, and how to evaluate the
result. The design notes and the known gaps live next to it under
`docs/design/`.

- [Pieces](#pieces)
- [How a turn flows](#how-a-turn-flows)
- [The router](#the-router)
- [Invocation: CLI, TUI, ACP](#invocation-cli-tui-acp)
- [PACT delegation over ACP](#pact-delegation-over-acp)
- [ContextGraph memory](#contextgraph-memory)
- [Requirement clarification and code localization](#requirement-clarification-and-code-localization)
- [Configuration reference](#configuration-reference)
- [Artifacts a turn leaves behind](#artifacts-a-turn-leaves-behind)
- [Evaluation](#evaluation)
- [Known gaps](#known-gaps)

## Pieces

| Piece | Where | What it does |
|---|---|---|
| Router | `service/routing/`, `orchestration/engine/run/routing.py` | Classifies each turn as `standard` or `long_horizon`; honours explicit overrides; probes workspace readiness for delegation. |
| Long-horizon extensions | `orchestration/engine/run/long_horizon.py`, `workflow_extensions.py` | The hooks the clarification workflow calls: localize + recall during clarification, write the task brief, run the delegation pass, record the campaign's outcome. |
| Requirement clarification | `service/requirement_clarification/` | Freezes an S0 snapshot, investigates the repository with bounded read tools, proposes and selects guidance, writes the clarified requirement and the PACT inputs. |
| Requirement enrichment | `orchestration/engine/run/runner.py` (`RequirementEnrichmentWorkflow`) | The lighter sibling: clarify ‖ localize as a preflight, then one normal run with the result as a reminder. |
| Semantic search / localization | `service/semantic_search/` | Builds a light index of the workspace, optionally a CodeGraph perception, and asks a model for the code locations a requirement touches. |
| PACT delegation | `pact/`, `orchestration/sub_agents/`, builtin profile `ChrysPact` | Runs a governed campaign (Manager, Planner, Worker, Reviewer roles) on an accepted goal contract and plan, as an ACP sub-agent. |
| ContextGraph memory | `service/memory/` | Recall tools over a Neo4j graph (canonical rules + deposited trajectories); per-repository prior recall for the brief; writeback of turn experience. |
| Evaluation | `evaluation/`, `examples/long-horizon/` | DeepSWE runner, LoLBench adapter, e2e smoke for the whole track. |

## How a turn flows

```
prompt ─► router ──standard──► normal agent loop
            │
            └─long_horizon──► switch to the LongHorizon profile
                                │
                                ├─ S0 snapshot (frozen copy of the workspace)
                                ├─ in parallel:
                                │    ├─ code localization on the S0 view      ─┐
                                │    ├─ prior recall from the team graph        ├─► task brief
                                │    └─ requirement clarification (3 proposals) ─┘
                                ├─ P0: baseline pass on the original requirement
                                ├─ P1: repair pass with the clarified requirement + located code
                                ├─ delegation pass: the model reads the brief and calls chrys_pact
                                │    └─ PACT campaign (separate process, ACP): Manager → Worker → Reviewer …
                                └─ final response; route marker records track / baseline / campaign
                                     └─ writeback: the turn's experience is deposited into the graph
```

Two things are true at every step. The original requirement stays the authority:
clarification adds guidance, it never rewrites acceptance criteria. And every piece
degrades rather than blocks: a localization that times out, a graph that is not
configured, a campaign tool that is not reachable — each is recorded in the artifacts
and the turn still delivers the best answer it has (the repaired baseline).

## The router

The router decides per turn, before the prompt is admitted. Its inputs are the prompt
text, the agent profile's `routing` block, and — only when needed — a readiness probe
of the workspace.

1. **Heuristic score.** `service/routing/classifier.py` extracts signals from the
   prompt (scope words such as "end-to-end", "migrate every caller"; acceptance
   criteria; explicit file lists; multi-step phrasing) and scores them into a band:
   `strong_standard`, `lean_standard`, `lean_long_horizon`, `strong_long_horizon`.
2. **LLM tiebreaker.** A lean band asks a small model once (`routing.tiebreaker_model_profile`)
   for a verdict with a confidence; below `min_confidence` the heuristic band stands.
   An unavailable tiebreaker is recorded as such (`tiebreaker_failure`) and the lean
   band resolves to standard.
3. **Overrides win.** `--route long-horizon|standard` on the CLI, `/longrun` and
   `/quick` in the TUI, and `session/route_override` over ACP publish a one-shot
   `RouteOverride`; the router consumes it for exactly the next turn and reports
   `source: override`. A `/quick` during a long-horizon turn's preparation stops that
   turn's campaign instead of arming the next one.
4. **Readiness gates delegation, not the track.** A long-horizon decision switches
   the profile (`routing.target_profile`, `LongHorizon` for the builtin `Code`), and
   the plan says which stages run (`localization`, `clarification`, `pact`). `pact`
   is only planned when the workspace can carry a campaign: a `pact.verify_command`
   is configured and the profile that will run the turn exposes the `chrys_pact`
   sub-agent tool. Otherwise the turn still runs the track and ends at the repaired
   baseline.

Dry-run any prompt without running an agent:

```bash
chrys debug router -C /path/to/workspace "Implement end-to-end typed parsing …"
chrys debug router --json -C . -t task.md      # band, track, plan, readiness
```

Every routed turn writes a `turn.routed` trajectory event and a `_chrys_route` marker
on the turn's final message (`track`, `band`, `source`, `baseline`, `campaign`).

## Invocation: CLI, TUI, ACP

**CLI (headless).** `chrys run` runs one turn to its final response.

```bash
# let the router decide (the builtin Code profile has routing.mode: auto)
chrys run -a Code -C /path/to/workspace "…requirement…"

# force the track, read the requirement from a file, machine-readable output
chrys run -a Code --route long-horizon -t task.md -C . --json
```

The JSON result carries `session_id`, the final text, and `route` (`track`, `band`,
`source`, `pact`) when a turn was classified. `python -m chrys` is the same entry
point; the PACT sub-agent launches itself through it.

**TUI.** `/route` shows the current mode and the last decision; `/longrun` arms the
long-horizon track for the next message; `/quick` arms standard, or stops a
long-horizon turn that is still preparing. The status bar shows the phase
(`snapshot`, `clarification`, `repair`, `delegation`) while a long-horizon turn runs.

**ACP server.** `chrys acp` exposes Chrys as an Agent Client Protocol stdio server,
which is how editors and other agents drive it:

```bash
chrys acp -a Code -C /path/to/workspace --approval auto
```

Beyond the standard `session/new`, `session/prompt` and `session/cancel`, the server
answers Chrys-specific methods: `session/route_override` (`{sessionId, track:
"long_horizon"|"standard"|"", reroute}`), `session/inject`, `session/mutations`,
`session/diff`, `session/rollback`, `session/switch_agent`. Routed turns stream the
same phase updates the TUI shows, as `tool_call` / `tool_call_update` notifications
titled by phase.

## PACT delegation over ACP

The campaign runs in a separate process so that its roles, its worktrees and its
verification cannot disturb the session that delegated it. The mechanics:

- The `LongHorizon` profile lists `chrys_pact` among its sub-agents. That sub-agent is
  the builtin `ChrysPact` profile, whose `acp` block is:

  ```yaml
  acp:
    command: chrys                     # resolves to THIS executable / interpreter
    args: ["pact-agent", "--agent", "Code", "--max-rounds", "2", "--verify-from-settings"]
    result_mode: last_segment
    idle_timeout_seconds: 0            # a campaign runs as long as the work takes
    max_depth: 1                       # a role can never start another campaign
  ```

  `chrys pact-agent` takes the role profile (`--agent`), the Worker/Reviewer rounds a
  mission may take before the campaign stops (`--max-rounds`, default 3), and exactly
  one verification choice: `--verify '<command>'` names it outright,
  `--verify-from-settings` reads `pact.verify_command` (project or user settings, or
  `CHRYS_PACT_VERIFY_COMMAND`), and `--allow-unverified` runs with none. The builtin
  reads settings because the command is the workspace's, not the profile's; a
  profile of your own can pin one:

  ```bash
  chrys pact-agent --agent Code --max-rounds 2 --verify 'python -m pytest -q'
  ```

  Whichever way it arrives, the command runs through `chrys.pact.verify_shim`:
  pact_core verifies each checkpoint in a fresh `git worktree`, which carries only
  tracked files, so the shim first symlinks the primary checkout's git-ignored
  directories (`node_modules`, `.venv`, `target`, a vendored tree) into the worktree
  where nothing of that name exists, then runs the command in place. Without it,
  `npm test` in the worktree ends in `vitest: not found` while passing in the
  workspace, and every mission fails its gate.

  `command: chrys` is special-cased (`_resolve_self_command`): a source checkout runs
  `sys.executable -m chrys …`, a packaged runtime runs its own binary. The child's
  environment is an allowlist (`HOME`, `PATH`, …) plus the profile's `env`; a self
  child additionally receives `CHRYS_PACT_VERIFY_COMMAND` and `CHRYS_MODEL_PROFILE`
  from the parent so the campaign verifies and models exactly as the session does.

- The delegation pass gives the model one job: read the task brief and call
  `chrys_pact` with the run request `{schema: "chrys-pact/run-request/v1",
  contract_path, plan_path}`. The contract and plan were written by clarification
  under `.pact-io/chrys-pact/<id>/` in the workspace (`goal-contract.json`,
  `initial-plan.json`).

- `chrys pact-agent` (`pact/server.py`) is that ACP agent. Per prompt it starts a
  `CampaignCoordinator` (`pact/campaign.py`) over `pact_core`: the Manager selects a
  frontier mission, the Worker implements it in a dedicated git worktree under
  `~/.pact-core/worktrees/`, the Reviewer grades the diff against the plan and writes
  `reviewer-decision.json`, the Planner replans when asked. Every role is an
  in-process Chrys host (`pact/role_runner.py`) running the `--agent` profile with
  routing and memory switched off. The verification command is what makes a mission
  acceptable; a campaign with no verification is refused unless `--allow-unverified`.

- The campaign reports back as text (`PACT Campaign result / status: … /
  campaign_id: … / artifacts: …`); the delegation pass parses it
  (`parse_campaign_report`) and records `campaign` on the route marker. A pass that
  ends without a report degrades to the repaired baseline with a warning.

Run a campaign by hand for debugging (it speaks ACP on stdio; a client such as
`chrys acp`'s sub-agent runner or any ACP client drives it):

```bash
chrys pact-agent --agent Code --max-rounds 2 --verify 'uv run pytest -q'
```

## ContextGraph memory

The graph is a Neo4j database in ContextGraph's schema: canonical rules distilled
from many trajectories, and trajectories deposited by agents, split into fragments
(`exploration`, `error_recovery`, `failed_attempt`, `successful_fix`) with embeddings
and a full-text index.

**Getting a graph.** Pull the published image, which carries the CAPBench
selected-Harbor graph (2,525 canonical rules) and loads it on first start:

```bash
docker run -d --name contextgraph -p 7705:7687 -p 7495:7474 \
  -e NEO4J_AUTH=neo4j/contextgraph123 wzh4464/contextgraph:capbench-harbor
```

Then tell Chrys about it in `~/.chrys/.env`:

```bash
CONTEXTGRAPH_NEO4J_URI=bolt://127.0.0.1:7705
CONTEXTGRAPH_NEO4J_USER=neo4j
CONTEXTGRAPH_NEO4J_PASSWORD=contextgraph123
CONTEXTGRAPH_EMBEDDING_MODEL=text-embedding-3-large
CONTEXTGRAPH_EMBEDDING_API_KEY=…          # an OpenAI-compatible embeddings endpoint
CONTEXTGRAPH_EMBEDDING_BASE_URL=…
CONTEXTGRAPH_REPO=/path/to/ContextGraph    # checkout of github.com/wzh4464/ContextGraph; deposits run its worker
```

`chrys memory doctor` reports what is missing; `chrys memory init --import
neo4j.dump` provisions a local Neo4j from a dump instead of the image.

**Recall.** With `memory.mcp.enabled` (default on) every session gets the
`contextgraph` MCP server with `team_memory_query` and friends, so any agent can ask
for prior experience. The long-horizon brief does not wait for the model to ask: while
clarification runs, `query_prior` recalls the top canonical rules for the requirement
and the fragments deposited from *this repository*, keeps the two apart, and writes
the result — or the reason there is none — to `long_horizon/turn_N/memory-prior.md`
and into the brief itself.

**Writeback.** After a turn, its experience (problem statement, tool steps, outcome,
final response) is deposited by the `after_turn` hook; a session that is idle for
`memory.writeback.idle_seconds` or ends normally is flushed; `chrys memory sweep`
deposits whatever the engine never got to. A watermark on the session records the
last deposited turn. Deposits are labelled by repository — the main git repository's
directory, so a PACT worktree and the workspace land under one name.

## Requirement clarification and code localization

Clarification (`service/requirement_clarification/`) runs on a frozen S0 snapshot,
never the live workspace. Three investigators, each with a different focus
(ownership/extension, data/control flow, boundary/compatibility), read the snapshot
through bounded tools (`read_file`, `grep`, `glob` — ripgrep-backed) and must
produce at least one successful search and one successful read before their
proposal counts. A selector reviews the proposals and keeps the guidance that is
necessary and evidenced; the result is the clarified requirement (original text plus a
guidance delta), the PACT goal contract and initial plan. Insufficient evidence
degrades the turn to `p0_promoted`: the baseline answer stands, nothing invented is
added.

Localization (`service/semantic_search/`) builds a light index of the snapshot, runs
an agentic search with a model (`semantic_search.model_profile`) under a budget
(`semantic_search.localization_timeout_seconds`), and optionally asks the CodeGraph
CLI for symbol relationships. Its candidates carry a role (`primary`, `propagation`,
`validation`) and a confidence, go into the repair reminder and the brief marked
*untrusted; verify before editing*, and are scored against gold files by the
evaluation.

Both need ripgrep. A source tree carries the platform's `rg` under
`src/chrys/foundation/vendor/ripgrep/`; a binary that does not run on this machine
is skipped in favour of one on `PATH`. CodeGraph is optional: install it with
`curl -fsSL https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh | sh`
and the pipeline picks it up from `PATH`.

## Configuration reference

Settings (`~/.chrys/settings.yaml`, project `.chrys/settings.yaml` when
`project.config_enabled`, or environment):

| Key | Env | Default | Meaning |
|---|---|---|---|
| `routing.mode` | `CHRYS_ROUTING_MODE` | profile's | `off` / `auto` / `always`; `always` forces the track for every turn. |
| `routing.tiebreaker_model_profile` | `CHRYS_ROUTING_TIEBREAKER_MODEL_PROFILE` | active model | Model for the lean-band verdict. |
| `pact.verify_command` | `CHRYS_PACT_VERIFY_COMMAND` | — | Deterministic verification the campaign runs; required for delegation. |
| `semantic_search.model_profile` | `CHRYS_SEMANTIC_SEARCH_MODEL_PROFILE` | active model | Model for localization. |
| `semantic_search.localization_timeout_seconds` | `CHRYS_SEMANTIC_SEARCH_LOCALIZATION_TIMEOUT` | 120 | Localization budget; a reasoning model on a real repository needs 900–1800. |
| `memory.mcp.enabled` | `CHRYS_MEMORY_MCP` | true | Attach the ContextGraph MCP server to sessions. |
| `memory.writeback.idle_seconds` | `CHRYS_MEMORY_WRITEBACK_IDLE_SECONDS` | 3600 | Idle flush. |
| `memory.writeback.on_session_end` | `CHRYS_MEMORY_WRITEBACK_ON_END` | true | Flush at session end. |

Agent profile (`~/.chrys/agents/<Name>.yaml`):

```yaml
routing:
  mode: auto                 # off | auto | always
  target_profile: LongHorizon
  classifier: both           # heuristic | llm | both
  min_confidence: 0.7
  long_horizon:
    localization: true
    clarification: true
    pact_tool: chrys_pact
    require_pact: false      # warn when the delegation pass never calls the tool
requirement_enrichment:      # the lighter sibling; mutually exclusive with requirement_clarification
  enabled: false
```

Model profile for a reasoning model over OpenRouter (`~/.chrys/models/<id>.yaml`):

```yaml
id: deepseek-v4-pro-or
provider: deepseek-openai
api_style: chat_completions
model_id: deepseek/deepseek-v4-pro
base_url: https://openrouter.ai/api/v1
api_key: "{{OPENROUTER_API_KEY}}"
chat_options: '{"extra_body": {"reasoning": {"effort": "high"}}}'
stream: true
```

## Artifacts a turn leaves behind

Under the session directory (`~/.chrys/sessions/<short-id>/`):

```
requirement_clarification/turn_N/
  01-input/            requirement.md, workspace-snapshot.json
  02-initial-trial/    P0 response and transcript
  03-clarification/    candidates/, investigations/, decision/, deliverable/clarified-requirement.md
  04-repair/           repair attempts
  05-outcome/          summary.json, clarified-requirement(-delta).md, final-response.md
  06-pact-input/       goal-contract.json, initial-plan.json, generation.private.json
long_horizon/turn_N/
  brief.md             what the campaign's roles read
  memory-prior.md      the recall's status and content
  semantic-search/     code-localization.{json,md}, codegraph-perception.{json,md}, localization-trace.jsonl
sub_agents/sessions/   chrys_pact_<id>.json (+ .stderr.log): the campaign's update stream and result
trajectory/events.jsonl
```

The workspace keeps `.pact-io/chrys-pact/<id>/` (the inputs) and
`.pact/runtime/campaigns/<id>/` (the campaign's canonical state).

## Evaluation

- `examples/long-horizon/e2e_smoke.sh` runs one real routed turn on a throwaway
  workspace and checks every stage, including writeback to the graph
  (`CHRYS_SMOKE_MODEL=<profile> ./examples/long-horizon/e2e_smoke.sh`).
- `evaluation/semantic_search/deepswe_runner.py` runs DeepSWE tasks through
  localization only (`--per-task`), the enrichment workflow (`--run-enrichment`), or
  the full track (`--run-long-horizon`), and scores localization against gold files.
- `evaluation/deepswe/lolbench/` runs DeepSWE tasks through the LoLBench harness
  (in-container generation with anti-cheat, Harbor verifier grading) — see its README.

## Known gaps

`docs/design/long-horizon-known-gaps.md` is the honest list: no semantic cancel for
a running campaign, no graph across hosts, ripgrep as a hidden hard dependency, role
hosts without memory, and what the first benchmark runs taught. The deviation log in
`docs/superpowers/plans/2026-09-03-long-horizon-suite-checklist.md` records every
fault found by live runs and the commit that closed it.
