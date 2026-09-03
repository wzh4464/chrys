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

A campaign's verify command is what "done" means for your repository. This
one is Chrys's own, in `.chrys/settings.yaml`, ordered so that the cheapest
check fails first:

```yaml
pact:
  verify_command: 'uv run ruff check --no-fix src/ tests/ && uv run ruff format --check src/ tests/ && uv run ty check --error-on-warning src/chrys && LANG=en_US.UTF-8 uv run pytest -q -m "not integration and not gc_calibration" --deselect "tests/service/skills/test_runner.py::test_stopped_script_returns_promptly"'
```

It runs with `shell=True` from the repository root, so `&&` and environment
prefixes work. Make it the same gate your CI applies: a campaign that verifies
with less can report work complete that CI then rejects.

**The project layer is off by default** — cloning a repository must not hand it
configuration authority. To let `<repo>/.chrys/settings.yaml` take effect, you
turn it on yourself, once, in `~/.chrys/settings.yaml`:

```yaml
project:
  config_enabled: true
```

Or scope it to a single run instead, without granting any repository anything:

```bash
CHRYS_PACT_VERIFY_COMMAND='…' chrys run --route long-horizon "…"
```

Without a verify command a campaign cannot tell done from broken, so the router
does not delegate — it still runs the baseline, the search, and the repair.

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

The dump is yours to provide; nothing in this repository distributes one. If
you already have a graph running somewhere, point `CONTEXTGRAPH_NEO4J_URI` at
it instead — that is all "seeding" means here.

**Match the embedding model to the graph.** Retrieval embeds your query and
compares it against vectors already stored, so a different model is a different
space and the vector channel returns noise. Check what is in there:

```bash
cypher-shell "MATCH (r:CanonicalRule) WHERE r.embedding IS NOT NULL RETURN size(r.embedding) LIMIT 1"
# 3072 -> text-embedding-3-large   1536 -> text-embedding-3-small
```

then set `CONTEXTGRAPH_EMBEDDING_MODEL` to match. `chrys memory doctor` cannot
catch this one: a mismatched model still connects, still answers, and quietly
ranks badly.

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

### What was verified on the delivery machine

Both readiness halves are real here, so the router plans the whole chain:

```
$ chrys debug router "Implement end-to-end OAuth login: … Acceptance criteria:
  1) existing sessions keep working 2) … 3) all tests pass. Touch
  src/auth/provider.py, src/api/routes.py and web/src/login.tsx as needed."
band            strong_long_horizon
readiness       verify_command=True tests=True pact_ready=True
plan            localization=True clarification=True pact=True
```

Drop the acceptance criteria and the file mentions from that message and it
lands in `uncertain` instead, where the router spends one tiebreaker call —
worth knowing when you write a prompt.

The memory graph is the CAPBench selected-Harbor corpus on
`bolt://127.0.0.1:7705` (see
[known gaps §3](../../docs/design/long-horizon-known-gaps.md)):

```
$ chrys memory doctor
[ok] CONTEXTGRAPH_NEO4J_URI: bolt://127.0.0.1:7705
[ok] CONTEXTGRAPH_NEO4J_PASSWORD: set
[ok] CONTEXTGRAPH_EMBEDDING_API_KEY: set
[ok] neo4j: ContextGraph Neo4j is healthy (canonical_rules=2525, chrys_trajectories=0).
[ok] CONTEXTGRAPH_REPO: /Users/zihanwu/codes/ContextGraph (python)
```

`chrys_trajectories=0` is the shape of a seeded graph before this machine has
deposited anything: priors to draw on, no local experience yet. A recall
against it returns what a plan can actually use — for "the build fails after I
added a new public method to a Java interface", the top rules are about
outdated mock signatures, symbol-export files per JDK version, and updating the
tests that implement the interface.

The verify command above was run to completion on a clean tree: **exit 0 in
~130 s**, `19373 passed, 11 skipped`. It was also checked to fail — a single
badly formatted file in `src/` makes both ruff stages exit 1, so the gate bites
before a campaign can call unformatted work done.
