# Primary Chrys × chrys_pact × PACT R3 Integration

> Status: **accepted MVP implementation contract**
>
> Date: 2026-09-01
>
> Chrys base: `15c436e7a7b2b3544b60b781124a470410fb99f2`
>
> PACT R3 dependency: `aa9073bed4970481a035755990e1682e9de486d8`

## 1. Outcome

Users start in an ordinary Primary Chrys session. For a long-running governed task, Primary Chrys prepares the
accepted Goal Contract and Initial Plan, then delegates one call to the external ACP agent `chrys_pact`.

```text
Primary Chrys
  -> one external ACP process/session
    -> chrys_pact
      -> PACT R3 CampaignControlPlane
        -> fresh in-process Chrys sessions for Worker, Reviewer, Planner, and Manager
```

There is one process boundary, at the product boundary. A role turn never launches another `chrys acp` child.
PACT artifacts and projections remain the authoritative Work State. Chrys EventBus events and ACP updates are
presentation telemetry only.

## 2. MVP decisions

### 2.1 Use a verified upstream PACT R3 wheel without a fork

The Chrys project dependency is version-pinned and its uv source is the repository wheel:

```text
pact-core==0.2.0.dev0
vendor/wheels/pact_core-0.2.0.dev0-py3-none-any.whl
```

The wheel was built from immutable PACT R3 commit `aa9073bed4970481a035755990e1682e9de486d8`.
`vendor/pact-core.json` records that source and the wheel SHA-256; installation and offline packaging validate the
artifact before use. The source is neither forked nor copied into Chrys, and installation does not require access
to the PACT Git repository. A different local `pact-core` checkout has no effect unless the wheel, provenance,
dependency pin, and lockfile are deliberately updated together.

### 2.2 Run Chrys roles in-process

`chrys_pact` supplies a Chrys-owned PACT adapter. Each Worker, Reviewer, Planner, or Manager turn creates a fresh
`ChrysSessionHost` in the work directory supplied by PACT. All roles use the configured Chrys agent profile and
normal Chrys tools, hooks, skills, MCP configuration, context management, and compaction.

The role runtime is non-interactive:

- `ApprovalMode.BYPASS`;
- `allow_user_interaction=False`;
- one fresh host and EventBus per turn;
- bounded shutdown: cooperative paths are deterministic, while an unresponsive runtime escalates to outer child
  teardown.

`BYPASS` removes interactive approval prompts; it is not an operating-system sandbox.

Planner and Manager use PACT's existing `AdapterPlanningProvider` and `AdapterDecisionProvider`. Reviewer output
uses an adapter-owned `.pact-io/reviewer-decision.json` transport epilogue and PACT's structured decision parser,
so PlanChallenge and governed replan remain PACT behavior rather than Chrys-specific policy.

### 2.3 Keep the Primary runtime and translator unchanged

The MVP uses the existing external ACP sub-agent mechanism. The packaged executable only gains the
`pact-agent` launcher for the external child process; it adds no `/pact` command, dedicated Campaign panel,
private ACP extension, translator change, or PACT state machine to the Primary runtime.

An unmodified Primary receives coarse, standard ACP presentation:

- Campaign and semantic-role stage tool cards;
- inner tool start and terminal updates;
- token/context usage updates;
- one final Campaign summary.

The MVP does not promise live rendering of role prose, hidden thought, intermediate tool progress, or compaction
events in stock Primary Chrys. Richer presentation would require a separate Primary translator/TUI change.

## 3. Public launch contract

### 3.1 Executable

Wheel and source installs expose a dedicated console entry point:

```text
chrys-pact --agent Code --verify "uv run pytest"
chrys-pact --agent Code --allow-unverified
```

The single-file Chrys release contains only the `chrys` executable, so it exposes the same entry point through
the built-in dispatcher:

```text
chrys pact-agent --agent Code --verify "uv run pytest"
chrys pact-agent --agent Code --allow-unverified
```

Exactly one of `--verify` and `--allow-unverified` is required. Verification is configuration for the whole ACP
process; it is not accepted from model-authored prompt JSON. The MVP intentionally does not expose final verifier,
round limits, timeout tuning, or per-role profiles.

### 3.2 Primary-owned input files

Before delegation, Primary Chrys writes both accepted inputs beneath one request directory:

```text
.pact-io/chrys-pact/<request-id>/goal-contract.json
.pact-io/chrys-pact/<request-id>/initial-plan.json
```

The files use the PACT R3 schemas:

```text
pact-runtime/goal-contract/v1
pact-runtime/initial-plan/v1
```

Primary owns their preparation. `chrys_pact` validates and executes them; it does not generate, revise, or ask the
user to confirm the initial contract and plan.

### 3.3 ACP prompt

The only prompt is a single text content block containing exactly this JSON shape:

```json
{
  "schema": "chrys-pact/run-request/v1",
  "contract_path": ".pact-io/chrys-pact/<request-id>/goal-contract.json",
  "plan_path": ".pact-io/chrys-pact/<request-id>/initial-plan.json"
}
```

Validation happens before Campaign creation or a model call:

- reject non-text or multiple prompt blocks, invalid JSON, unknown/missing fields, and the wrong schema;
- require both paths to be workspace-relative regular files;
- require both paths beneath the same `.pact-io/chrys-pact/<request-id>/` directory;
- resolve symlinks and reject any path that leaves the ACP session workspace;
- reject additional workspace directories.

One `chrys-pact` process accepts one ACP session, one prompt, and one Campaign.

## 4. Runtime ownership and flow

| Owner | Responsibility |
|---|---|
| Primary Chrys | user conversation, initial Contract/Plan preparation, external-agent invocation, stock ACP rendering |
| `chrys-pact` ACP shell | ACP lifecycle, launch validation, Campaign task, cancellation signal, terminal response |
| PACT Control Plane | Mission selection, canonical Work State, Evidence, verification, gate, checkpoint, promotion, governed replan |
| in-process Chrys adapter | fresh semantic-role host, prompt execution, structured result normalization, EventBus-to-ACP bridge |

`CampaignControlPlane.run()` is synchronous, so it runs in one owned worker thread. Its synchronous adapter calls
schedule Chrys coroutines onto the ACP event loop with a thread-safe bridge. The ACP event loop remains the sole
owner of role hosts, EventBus callbacks, and ACP sends.

The execution sequence is:

1. validate the ACP session, prompt envelope, paths, and CLI verification mode;
2. generate `campaign_id` and construct the PACT `CampaignRunRequest`;
3. create four semantic-role adapter instances and wrap Planner/Manager with PACT's providers;
4. run the Control Plane in its worker thread;
5. for each role turn, create a fresh host in `TurnRequest.workdir`, stream coarse ACP updates, normalize the result,
   and shut the host down;
6. read the canonical PACT projection at role boundaries and at termination;
7. return a last-segment summary containing `status`, `campaign_id`, `revision`, `next_action`, and the
   workspace-relative Campaign artifact path.

The external ACP profile sets `idle_timeout_seconds: 0`. Deterministic verification and promotion can legitimately
exceed the stock ten-minute idle window without emitting ACP updates, so explicit user cancellation and the Primary
Chrys child-process teardown own that outer lifetime.

Only a canonical PACT `completed` result is reported as successful completion. Agent or Reviewer prose cannot close
the Campaign. A canonical `blocked` or `active` result still ends the ACP transport turn normally so stock Primary
Chrys preserves its final summary, while the Campaign tool card and summary explicitly remain non-completed.

## 5. Event and result mapping

Role-stage and inner tool-call identifiers are namespaced by Campaign/turn before they enter the outer ACP stream;
provider IDs are not assumed globally unique. Chrys Todo-plan updates are not forwarded as PACT plan state.

Role outcomes map to PACT adapter results as follows:

| Chrys turn outcome | PACT adapter result |
|---|---|
| final agent message | `completed` with normalized final text |
| turn ended without a final message | `output_missing` |
| role timeout that cancels and drains cooperatively | `timeout` |
| cancellation, turn drain, or host shutdown exceeds the cleanup grace period | fatal integration abort; Primary must tear down the outer child |
| host construction or execution failure | `spawn_failed` |
| outer cancellation | integration abort; never synthetic success |

ACP/EventBus delivery failure does not mutate canonical Work State. Existing `.pact/` Campaign and loop artifacts
remain available for inspection.

## 6. Cancellation boundary

PACT R3 has no public Campaign cancellation token or canonical `cancelled` Work State. The MVP therefore implements
best-effort invocation cancellation only:

1. ACP `session/cancel` sets an integration-owned abort flag;
2. it calls `cancel_current_turn()` on the active `ChrysSessionHost`;
3. if the role unwinds cooperatively, the outer prompt returns the ACP cancelled stop reason without claiming
   Campaign completion; otherwise Primary force-closes the child and no cancelled response is promised;
4. shutdown closes the active host and owned worker resources as far as the current PACT call boundary permits.

This is not semantic pause/resume. It does not roll back completed Missions, write a canonical cancelled state, or
guarantee preemption if cancellation races with verification or promotion. Primary Chrys may terminate the single
ACP child if it remains unresponsive; that is process cleanup, not a PACT state transition.

## 7. Primary Chrys configuration

Configure the external agent, for example:

```yaml
# ~/.chrys/agents/ChrysPact.yaml
name: ChrysPact
display_name: Chrys PACT
description: Run an accepted long-term plan through a governed PACT Campaign.
acp:
  command: chrys
  args: ["pact-agent", "--agent", "Code", "--verify", "uv run pytest"]
  result_mode: last_segment
  idle_timeout_seconds: 0
```

Then reference it from the Primary agent's `sub_agents.agents` list:

```yaml
- profile: ChrysPact
  tool_name: chrys_pact
  tool_description: Execute an accepted Goal Contract and Initial Plan as a governed long-running Campaign.
  max_concurrency: 1
```

Omit `acp.cwd` so the child inherits the active Primary workspace. Use `--allow-unverified` in the ACP args only
when the user has explicitly accepted unverified execution.

## 8. Acceptance and deferred work

The MVP is accepted when hermetic tests demonstrate strict pre-model input rejection, fresh isolated role hosts,
structured Reviewer/Planner/Manager governance, successful and blocked multi-Mission Campaigns, canonical terminal
summaries, useful stock-Primary stage/tool/usage output, and best-effort cancellation without false completion.

The following remain out of scope:

- a PACT fork or any `pact-core` source change;
- Campaign semantic cancellation, pause/resume, crash recovery, rollback, or R4 continuity;
- a Primary Chrys `/pact` TUI command, dedicated panel, translator change, or private ACP extension;
- nested role ACP processes, interactive role approvals/questions, multiple Campaigns, or per-role model profiles;
- making ACP or EventBus authoritative state.
