# Long-horizon tasks

A long-horizon turn is the ordinary coding turn plus three things: a baseline
pass, a clarification with a code search running beside it, and — when the
workspace can verify one — a governed PACT campaign that finishes and checks the
remaining work.

You do not have to ask for it. The router classifies each message and switches
to the built-in `LongHorizon` profile when a message earns the extra work. You
can also force either direction: `/longrun`, `/quick`, `chrys run --route`, or
the ACP `session/route_override` method.

## What a routed turn does

1. **Baseline.** The main agent implements the request once, from the frozen
   workspace. That implementation is P0.
2. **Clarify, and search in parallel.** Read-only side agents derive a
   repository-grounded ΔR from the *frozen* pre-baseline view, while a code
   search over that same frozen view ranks the locations a change is likely to
   touch. The search reads the frozen view on purpose: by now the live tree
   holds the baseline's guesses.
3. **Merge.** The search's candidates are appended to the repair guidance and to
   the campaign's plan prompt, always under an explicit *untrusted* header, and
   a task brief lands at
   `<session>/long_horizon/turn_<n>/brief.md`.
4. **Repair.** The main agent redoes the work from the original history, the
   baseline workspace, and ΔR. That is P1.
5. **Delegate.** With a verify command configured, the accepted Goal Contract
   and Initial Plan are staged into `.pact-io/chrys-pact/<request-id>/` and the
   agent hands them to a PACT campaign with one `chrys_pact` call.

Every stage degrades rather than failing the turn: a failed search costs
evidence, a degraded clarification promotes P0, a failed delegation keeps the
repaired answer.

## Setup

### 1. Tell PACT how to verify

```yaml
# <repo>/.chrys/settings.yaml
pact:
  verify_command: "uv run pytest -q"
```

Without it a campaign cannot tell done from broken, so the router does not
delegate — it still runs the baseline, the search, and the repair.

### 2. Bring up the memory graph (optional)

Memory is not required for the long-horizon track, but it is what lets one
session's experience reach the next.

```bash
chrys memory init          # starts the local Neo4j via the ContextGraph checkout
chrys memory doctor        # says exactly what is still missing
```

Connection details live in `~/.chrys/.env` (`CONTEXTGRAPH_*` — see
`examples/contextgraph-memory/env.example`).

#### Starting from an existing graph

Chrys ships **no initial graph**: a new install starts empty and fills up as
sessions run. To seed one from a dump you already have:

```bash
chrys memory init --import /path/to/neo4j.dump
```

The dump is yours to provide; nothing in this repository distributes one.

### 3. Check what the router would do

```bash
chrys debug router "add OAuth login across the api and the web client"
```

It prints the signals, the score, the band, the workspace readiness and the
resulting plan without running an agent.

## Trying it

```bash
# Let the router decide.
chrys run "Implement end-to-end OAuth login: add the provider abstraction, migrate the
user table, update the API, and write integration tests. Acceptance criteria:
existing sessions keep working and all tests pass."

# Or force it.
chrys run --route long-horizon "refactor the entire auth system"

# Or stay on one pass, whatever the router thinks.
chrys run --route standard "refactor the entire auth system"
```

In the TUI, `/longrun <message>` and `/quick <message>` do the same, and
`/route` reports how the last message was classified.

## What lands on disk

```
<session>/
  requirement_clarification/turn_<n>/   # S0, P0, ΔR, the accepted PACT pair
  long_horizon/turn_<n>/
    brief.md                            # what the campaign's roles read
    semantic-search/                    # the code search's artifacts
<workspace>/.pact-io/chrys-pact/<id>/   # goal-contract.json, initial-plan.json
```

`.pact-io/` is Chrys-owned and excluded from mutation tracking: it holds the
inputs a campaign was launched with, not your edits.

## Verifying an install

```bash
./examples/long-horizon/e2e_smoke.sh
```

It needs a real model and, for the memory half, a running Neo4j.
