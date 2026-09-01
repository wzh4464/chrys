# Chrys × PACT R3 Integration Design

> Status: **proposed, docs-only**
>
> Date: 2026-09-01
>
> Base: `chrys main@15c436e7a7b2b3544b60b781124a470410fb99f2`
>
> PACT evidence reviewed: local R3 candidate `ef8b44bc82e04104487c69633f20d86d011065e0`; input-contract handoff `000c3238d1110fc1d849fdcf8c46becb7bef915a`
>
> Boundary: this document defines an integration direction. No `/pact` command, PACT dependency, ACP role adapter, EventBus event, or TUI panel described below is implemented on this branch yet.

## 1. Outcome

Users start a PACT Campaign from the existing Chrys TUI and keep watching the same interface while PACT runs Worker, Reviewer, Planner, and Manager turns through Chrys:

```text
/pact run --contract goal-contract.json --plan initial-plan.json
```

The TUI should show two distinct kinds of information:

1. PACT control-plane state: Campaign status, current Mission, Frontier, role, Plan revision, acceptance gaps, and terminal outcome;
2. live Chrys execution activity: agent text/thinking, tool calls, tool progress/results, context-window usage, and compaction signals.

PACT remains the authority for Campaign/Work State. Chrys transports agent turns and renders activity. Neither ACP nor EventBus becomes a second Work State store.

## 2. Decisions

### 2.1 Integrate the current PACT R3 line

The integration target is the current `pact-core` R3 candidate, not the older DeepSWE + mini-swe-agent experiment.

R3 already contains the product concepts the Chrys command needs:

- Goal Contract and user-accepted Initial Plan;
- multi-Mission Work State and Frontier;
- Worker/Reviewer execution through the existing Execution Kernel;
- Planner proposal, Manager approval, and governed PlanRevision 2+;
- per-AC Evidence, checkpoint/promotion, and a rebuildable Dashboard projection.

The older experiment remains useful as benchmark and deployment evidence. It is not the product integration base because it binds PACT to one experimental runner rather than the work-centric Runtime contract.

### 2.2 Chrys depends on PACT; PACT does not import Chrys internals

The Chrys integration layer may import PACT's public Runtime and adapter contracts. `pact-core` must continue treating Chrys as a process boundary and must not import `chrys.*` packages.

This direction keeps one PACT implementation. It avoids copying the R3 state machine into Chrys or creating a Chrys-specific fork of Campaign semantics.

### 2.3 EventBus and ACP serve different boundaries

- Chrys `EventBus` is the in-process frontend/backend channel. The TUI subscribes to typed Python events.
- ACP is the cross-process control and streaming protocol. A Chrys ACP client sends session requests and receives serialized agent/tool/context updates from a `chrys acp` child.
- An integration bridge converts ACP updates and PACT projection changes into parent-session EventBus events.

ACP is therefore not a replacement for EventBus. ACP crosses the process boundary; EventBus drives the TUI after data returns to the main process.

### 2.4 PACT artifacts remain canonical

PACT's Goal Contract, PlanRevision, Work State, Evidence, Decision, checkpoints, and promotion receipts remain the recovery and audit source.

EventBus and ACP events are live presentation telemetry. A dropped UI consumer must not change Campaign semantics. Restarting a panel reconstructs its control-plane view from PACT's canonical state/projection, not by replaying the in-memory EventBus.

## 3. Confirmed current behavior

The design relies on these observed boundaries in the two repositories:

| Current surface | Confirmed behavior | Integration consequence |
|---|---|---|
| `EventBus` | In-process async pub/sub; callback delivery is ordered/backpressured and stream queues are lossless but ephemeral | Use it for main-process TUI delivery, not cross-process transport or durable state |
| `chrys run --json` | Writes only `session_id`, final `result`, and `duration` when the turn ends | It cannot drive live PACT role cards |
| `chrys acp` | Streams agent message/thought, tool lifecycle, usage, plan, approval, and Chrys extension notifications | Use it for every PACT semantic role turn that must be visible live |
| `AcpAgentClient` | Existing Chrys-owned client supports spawn, initialize, fresh session, prompt, cancellation, and bounded update handling | Reuse this client instead of implementing a second ACP stack |
| Chrys-in-Chrys test | A Chrys parent already launches a Chrys ACP child and renders nested tool/usage activity | Reuse its transport and translation patterns, without mislabeling PACT roles as ordinary sub-agent decisions |
| `CampaignControlPlane.run()` | Synchronous, single-committer R3 entry point using synchronous `AgentAdapter.run_turn()` | Run it off the Textual event loop and provide a Chrys-owned synchronous ACP adapter |
| current PACT `ChrysAdapter` | Spawns `chrys run --json`, waits for exit, returns final text | Keep it for ordinary headless use; add a separate integration adapter rather than silently changing its contract |

Source anchors:

- [`EventBus`](../../src/chrys/foundation/events/bus.py)
- [ACP frontend contract](../../src/chrys/app/acp/doc/frontend-api.md)
- [ACP event bridge](../../src/chrys/app/acp/bridge.py)
- [ACP client](../../src/chrys/service/acp_client/client.py)
- [Chrys-in-Chrys integration test](../../tests/app/acp/test_chrys_in_chrys.py)
- [slash-command registry](../../src/chrys/app/tui/screens/main/commands.py)

## 4. Proposed architecture

```text
Main Chrys process
┌─────────────────────────────────────────────────────────────────┐
│ Textual TUI                                                     │
│   /pact command                                                 │
│      │                                                          │
│      ▼                                                          │
│ PactIntegrationController                                      │
│   ├── validates Goal Contract + Initial Plan                    │
│   ├── owns one active Campaign task per Chrys session           │
│   ├── watches PACT canonical Dashboard projection              │
│   ├── publishes display-safe PACT events to EventBus            │
│   └── runs synchronous PACT Control Plane in a worker thread    │
│                         │                                       │
│                         ▼                                       │
│                 CampaignControlPlane.run()                      │
│                         │ AgentAdapter.run_turn()                │
│                         ▼                                       │
│                 ChrysPactAcpAdapter                             │
│                   ├── owns one ACP client/child per role turn   │
│                   ├── normalizes final text to TurnResult       │
│                   └── forwards live ACP updates thread-safely   │
│                                  │                              │
│ EventBus ◀──── Pact/ACP presentation bridge ────────────────────┘
│    │
│    ├── Pact Campaign panel/card
│    └── role activity, tool cards, usage/context indicators
└────┼────────────────────────────────────────────────────────────┘
     │ stdio JSON-RPC (ACP)
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ Child `chrys acp --agent <profile> -C <checkpoint-worktree>`    │
│   AgentEngine → hooks/memory/MCP/tools → internal EventBus      │
│                                    → ACP session/update          │
└─────────────────────────────────────────────────────────────────┘
```

### 4.1 Ownership

| Component | Owns | Must not own |
|---|---|---|
| PACT Control Plane | Campaign transitions, Work State commits, Mission selection rules, Evidence, completion | Chrys TUI state or ACP rendering |
| `PactIntegrationController` | command lifecycle, background execution, projection watching, parent-session correlation | Campaign semantic decisions |
| `ChrysPactAcpAdapter` | one role subprocess/session, ACP lifecycle, final-text normalization, live update forwarding | Work State mutation or completion |
| ACP bridge | serialization/translation of execution facts | persistence or recovery authority |
| Chrys EventBus | in-process delivery to TUI subscribers | cross-process transport or durable replay |
| PACT panel/card | presentation and user controls | authoritative Campaign state |

## 5. Command and input contract

The proposed demo command is:

```text
/pact run --contract <goal-contract.json> --plan <initial-plan.json>
```

The command name may change without changing the integration architecture. The first implementation should register one top-level `pact` slash command and parse its argument string independently of the ordinary chat submission path.

### 5.1 Canonical inputs

Both files are required for the Campaign path:

```text
Goal Contract: pact-runtime/goal-contract/v1
  schema
  goal
  acceptance_criteria[{id, text}]
  non_goals[]

Initial Plan: pact-runtime/initial-plan/v1
  schema
  constraints[]
  missions[{
    id,
    objective,
    target_ac_ids[],
    dependencies[],
    verification_intent
  }]
```

Requirement-analysis and code-location Markdown files may accompany the JSON pair, but they are supporting context. Hard completion obligations must be present in the Goal Contract. PACT must not infer hard AC from free-form Markdown.

The command resolves relative paths against the active primary workspace and rejects invalid JSON/schema/cross-file references before starting a model or creating a Campaign.

### 5.2 Demo defaults

To keep the first demo small:

- use the active Chrys model/profile unless an integration setting explicitly overrides it;
- use fresh, separate ACP sessions for Worker, Reviewer, Planner, and Manager turns;
- run one Campaign at a time per parent Chrys session;
- use current PACT Runtime defaults for round counts/timeouts unless exposed later;
- do not add a general plugin configuration system in this slice.

## 6. Runtime flow

### 6.1 Start

1. User submits `/pact run ...` in the TUI.
2. The command handler resolves both files inside the current workspace.
3. PACT validators check Goal Contract, Mission/AC coverage, dependencies, and DAG validity.
4. The controller mounts a PACT Campaign card/panel and starts the synchronous Control Plane outside the Textual loop.
5. The panel reads the first canonical projection rather than inventing an optimistic Work State.

### 6.2 Execute one semantic role turn

1. PACT invokes `AgentAdapter.run_turn()` for Worker, Reviewer, Planner, or Manager.
2. `ChrysPactAcpAdapter` starts a fresh `chrys acp` child in the workdir supplied by PACT.
3. The adapter performs `initialize → session/new → session/prompt`.
4. ACP updates are correlated with `campaign_id`, `mission_id`, `role`, `attempt`, `execution_id`, and ACP `session_id` before reaching the parent EventBus.
5. The TUI renders agent/tool/context activity under the active PACT role.
6. After the ACP response barrier drains, the adapter returns a normal PACT `TurnResult` containing final text and execution metadata.
7. PACT alone decides the next checkpoint, verification, review, gate, promotion, Work State transition, or governed replan.

### 6.3 Complete or block

1. The projection watcher observes a new canonical Work State revision.
2. The controller publishes a display snapshot containing only bounded, non-secret fields and stable artifact references.
3. `completed` is shown only after PACT commits terminal Work State. Worker or Reviewer text never closes the Campaign directly.
4. A blocked/failed Campaign retains its PACT artifacts and last projection for inspection.

## 7. ACP and EventBus presentation contract

### 7.1 Existing ACP data to expose

| ACP update | TUI presentation |
|---|---|
| agent message chunk | streaming role output |
| agent thought chunk | optional thinking section, when provider exposes it |
| tool call start / argument update | tool card with current arguments |
| tool progress / status / result | incremental output and terminal status |
| usage update | context-window gauge for the active role session |
| plan update | agent-local Todo plan; label it as agent-local, not PACT PlanRevision |
| permission request | existing Chrys approval flow |
| `chrys/context_pressure` | role context-pressure warning |
| `chrys/context_compressed` | compaction marker and bounded summary |
| `chrys/tool_compacted` / compaction lifecycle | compacted-tool/context status |

### 7.2 PACT-specific events

The first demo needs only a small display event surface:

- Campaign snapshot updated;
- semantic role started/finished;
- Campaign terminal or blocked.

Every event must include the parent Chrys `session_id` and PACT correlation identifiers. Event payloads are projections, not serialized canonical Work State. The UI uses the artifact reference to inspect authoritative data when needed.

Do not overload these terms:

- an ACP `plan` update is the current agent session's Todo list;
- a PACT `PlanRevision` is the Control Plane's versioned Mission graph;
- a Chrys ACP `session_id` identifies one role execution context;
- PACT `campaign_id` identifies the durable unit of work.

### 7.3 What “context display” means

The demo should display:

- input/output/total tokens and context-window percentage;
- context pressure and compaction events;
- agent text/thinking made available by ACP;
- tool arguments, progress, results, and failures;
- the PACT role/Mission that owns the activity.

It should not expose or persist the complete model context, system prompt, credentials, raw memory database results, or hidden final-verifier content. Current ACP does not provide a full raw-context read API, and the demo does not need to add one.

## 8. Memory, hooks, MCP, and profiles

Running a role through `chrys acp` uses the normal `AgentEngine` construction path. The selected Chrys profile therefore retains its configured hooks, memory providers, MCP servers, skills, and tools.

The PACT integration does not add a second memory lookup path. If the team's memory database is installed as a Chrys hook/MCP integration, role turns reach it through ordinary Chrys tool/hook execution.

PACT supplies the role prompt and isolated workdir. Chrys owns model/profile/tool construction. Separate ACP sessions prevent Worker, Reviewer, Planner, and Manager from silently sharing session-local conversation state.

## 9. Concurrency, cancellation, and failure boundaries

### 9.1 Keep Textual responsive

`CampaignControlPlane.run()` is synchronous. Calling it on the Textual event loop would freeze the TUI. The controller must run it in a dedicated worker thread or equivalent owned background executor and marshal UI events back onto the main loop.

The ACP client used by the synchronous adapter may own an asyncio loop inside that worker thread. Event delivery to the parent loop must use an explicit thread-safe queue or `run_coroutine_threadsafe`; it must not call the parent EventBus from the wrong loop.

### 9.2 Cancellation requires an explicit seam

The current R3 Control Plane has no public cancellation token. Python threads cannot be safely killed. Before `/pact cancel` can be considered complete, the integration needs a small PACT cancellation seam that:

1. stops admission of the next semantic role/round;
2. calls ACP `session/cancel` and terminates the owned child if it does not drain;
3. returns a typed non-complete outcome;
4. leaves source workspace promotion rules and existing PACT artifacts intact.

Until that seam exists, the demo must not claim graceful Campaign cancellation. Process exit remains an emergency stop, not semantic pause/resume.

### 9.3 Failure mapping

| Failure | Owner | Result |
|---|---|---|
| input/schema invalid | Chrys command boundary + PACT validator | reject before model call |
| ACP spawn/handshake failure | `ChrysPactAcpAdapter` | failed `TurnResult`; PACT records non-complete execution evidence |
| ACP idle/turn timeout | adapter | cancel/terminate child, return timeout |
| malformed reviewer/planner/manager output | existing PACT repair/normalization | repair once or fail closed according to R3 policy |
| TUI subscriber/render failure | Chrys presentation | Campaign continues; projection remains recoverable |
| PACT state/projection corruption | PACT Control Plane/reader | fail closed; do not reconstruct authority from EventBus |

## 10. Dependency and version pinning

A local PACT checkout is sufficient for implementation and smoke testing. A shared Chrys branch or CI job must be able to fetch the exact version independently.

Before this integration is pushed for team review:

1. make the selected PACT R3 commit reachable on a remote branch/tag/release;
2. pin Chrys development/CI installation to an immutable commit SHA, wheel hash, or image digest;
3. record the supported Goal Contract and Initial Plan schema IDs;
4. reject an incompatible PACT version before launching a Campaign.

A formal PACT release is not required for the demo. A remote immutable commit is enough. Do not commit an absolute local path such as `/Users/.../pact-core` as the shared dependency.

## 11. Demo scope

### Included

- `/pact run` from the existing Chrys TUI;
- Goal Contract + Initial Plan v1 validation;
- current PACT R3 Campaign execution;
- Worker, Reviewer, Planner, and Manager through fresh Chrys ACP sessions;
- live agent/tool/context activity in the TUI;
- PACT Campaign/Mission/Frontier/Plan revision status from canonical projection;
- blocked/completed result with artifact location;
- hermetic tests plus one explicitly authorized real-model smoke after the fake path passes.

### Explicit non-goals

- a general Chrys plugin marketplace or stable third-party plugin ABI;
- a second PACT implementation inside Chrys;
- raw full-context/session-debug display;
- PACT semantic resume, rollback/fork, or R4 continuity policy;
- distributed scheduling or concurrent Campaigns;
- redesigning Chrys EventBus into a persistent event log;
- duplicating PACT's full browser Dashboard pixel-for-pixel in Textual;
- changing existing `chrys run`, `chrys acp`, or standalone `pact run` behavior.

## 12. Implementation slices

### Slice 1: command and dependency boundary

- pin an exact remotely reachable PACT candidate;
- register `/pact run` and validate/resolve the two JSON inputs;
- add a controller state holder with one-active-Campaign enforcement;
- use FakeAdapter only; render canonical projection snapshots.

Acceptance: invalid input fails before a model call; a fake multi-Mission Campaign reaches the expected PACT terminal state without blocking the TUI.

### Slice 2: Chrys ACP role adapter

- implement the Chrys-owned PACT `AgentAdapter` using `AcpAgentClient`;
- use one child/session per semantic role turn;
- normalize final output, timeout, cancellation, and session provenance;
- forward agent/tool/usage updates with PACT correlation.

Acceptance: an ACP stub drives message, tool start/progress/result, usage, and final output through a PACT role without network or a real model.

### Slice 3: TUI Campaign presentation

- add the minimal PACT event types and parent-session correlation;
- add a Campaign card/panel showing Mission/Frontier/role/Plan revision/gaps;
- route live ACP activity beneath the correct role;
- preserve ordinary chat/tool rendering behavior.

Acceptance: deterministic TUI tests prove two sequential roles cannot cross-route tool calls or context gauges.

### Slice 4: lifecycle closeout

- add the explicit PACT cancellation seam and `/pact cancel` only after it is real;
- cover ACP failure, malformed role result, PACT blocked state, and TUI teardown;
- run one explicitly authorized real Chrys-model smoke.

Acceptance: the TUI remains responsive, cancellation owns all child processes, canonical artifacts remain inspectable, and no non-complete outcome is displayed as success.

## 13. Test plan

The default suite remains hermetic and zero-model:

1. slash-command parsing and workspace-relative input resolution;
2. Goal Contract/Initial Plan invalid-shape and cross-reference cases;
3. fake PACT Campaign projection to EventBus mapping;
4. ACP stub message/thought/tool/usage/result translation;
5. role/campaign/session/call correlation under sequential role turns;
6. slow/failed TUI subscriber does not mutate PACT state;
7. ACP spawn failure, timeout, cancellation, and malformed output;
8. TUI unmount/shutdown drains thread, ACP client, and subprocess ownership;
9. architecture test: `pact-core` never imports `chrys`, and Chrys layer imports follow the existing DAG;
10. one opt-in end-to-end smoke after explicit model-quota authorization.

## 14. Open decisions before implementation

These do not change the architecture and can be settled while preparing Slice 1:

1. final slash syntax and whether a file picker supplements explicit paths;
2. whether the first UI is a chat card, side panel, or both using one projection model;
3. which Chrys profile/model each semantic role uses by default;
4. where the exact PACT dependency pin lives in Chrys packaging before PACT publishes a release;
5. the minimal cancellation callback/token shape to add to PACT without pulling R4 pause/resume into the demo.

## 15. Alternatives rejected for the demo

### Continue using `chrys run --json`

It preserves the existing PACT adapter but yields only final text. Tool calls and context activity cannot be shown live without inventing a second JSONL protocol.

### Import `ChrysSessionHost` directly into PACT

It gives direct EventBus access but breaks PACT's agent-agnostic process boundary and couples PACT releases to Chrys internals.

### Treat EventBus as cross-process transport

EventBus carries Python objects inside one process. Making it remote would recreate serialization, lifecycle, request/response, and capability negotiation that ACP already provides.

### Make ACP or EventBus authoritative Work State

Both are delivery paths. Neither contains the complete versioned Goal/Plan/Evidence/Decision/checkpoint graph required to resume or audit a PACT Campaign.

### Copy the old experimental PACT runner into Chrys

It would produce two diverging PACT implementations and omit the R3 governance model the product integration is meant to expose.
