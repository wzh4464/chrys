# DeepSWE through LoLBench, with chrys on the long-horizon track

[DeepSWE](https://deepswe.datacurve.ai/) is a benchmark of long-horizon feature tasks
drawn from active open-source repositories, shipped in the Harbor task format (a
prebuilt image at the base commit, an `instruction.md`, hidden tests, a reference
solution). [LoLBench](https://github.com/rshu/LoLBench) is an evaluation harness for
exactly this kind of task: it runs an agent inside the task's image with anti-cheat
(source-hosting and package hosts black-holed, gold stripped, a Jaccard gate against
the reference patch), captures the patch and the full trajectory, keeps a patch store,
and grades hermetically. Its engine reads any benchmark laid out as
`data/ + dockers/ + instructions/` — the same trick its SWE-bench Pro experiment uses —
so this directory turns DeepSWE tasks into such a root and grades them with each task's
own Harbor verifier.

```
DeepSWE tasks/<id>/            gen_instances.py            LoLBench benchmarks/deepswe/
  task.toml, instruction.md  ───────────────────────►      data/lolbench_clean.csv
  tests/{test.sh,test.patch,…}                             dockers/<id>/{Dockerfile,README.md,spec.json,
  solution/solution.patch                                                 eval_tests.patch,solution.patch}
                                                           instructions/original/<id>.md
                                   run_generation.sh
   agent image = task image + uv/Python 3.14 + chrys + rg + CodeGraph + model profile
   chrys run --route long-horizon  (in container, anti-cheat, session store captured)
                                   ───────────────────────►  patches/deepswe/<agent>/<id>.patch
                                                            runs/deepswe/<agent>/<id>/agent_out/chrys/…
                                   grade.py
   verifier image = task tests/Dockerfile;  /tests/test.sh on the patch, --network=none
                                   ───────────────────────►  results.csv, summary.json, reward.json per task
                                   ../report.py  +  ../send_report_email.py
```

## Prerequisites

- Linux host with Docker; the DeepSWE checkout (`tasks/`), a LoLBench checkout.
- This chrys checkout (the source is tarred and served to the image builds over the
  docker bridge; the repository is private, containers cannot clone it).
- `OPENROUTER_API_KEY` for the model; the DeepSeek V4 Pro profile is
  `agent/model.yaml`.
- A ContextGraph Neo4j on the host (see `docs/long-horizon/README.md`; the published
  image is `wzh4464/contextgraph:capbench-harbor`). `run_generation.sh` bridges the
  loopback bolt port onto the docker bridge with `socat` so agents in containers can
  recall; deposits are swept from the host afterwards because the ContextGraph worker
  lives there (`CONTEXTGRAPH_REPO`).
- `socat`, `python3` on the host.

## 1. Materialize the benchmark root

```bash
cd /path/to/LoLBench
python3 /path/to/chrys/evaluation/deepswe/lolbench/gen_instances.py \
  --tasks-dir /data/deepswe/repo/tasks --out benchmarks/deepswe --offset 0 --limit 20
python3 scripts/lolbench_eval.py --repo-root benchmarks/deepswe --plan   # sanity: resolved plan, no Docker
```

The first 20 tasks in `tasks/manifest.json` order are the same 20 the DeepSWE runner
(`evaluation/semantic_search/deepswe_runner.py`) uses, so localization scores and
LoLBench grades line up per task. `--instances a,b` selects explicitly.

## 2. Generate patches (in-container, anti-cheat)

```bash
cd /path/to/LoLBench
set -a; . ./.env; set +a          # OPENROUTER_API_KEY, CONTEXTGRAPH_* secrets
JOBS=4 AGENT_TIMEOUT=5400 /path/to/chrys/evaluation/deepswe/lolbench/run_generation.sh all
# or: run_generation.sh abs-module-cache-flags,abs-stepped-slices
```

Per instance the engine builds `lolbench-deepswe-<sha>:1` (the task image plus a
`/workspace/app` symlink), derives the agent image with `agent/install.sh` (cached by
install-recipe digest, so 20 tasks that share a base image share the layer), runs
`agent/run.sh` — `chrys run --task PROMPT --agent Code --route long-horizon --json` —
with the session store on the captured mount, and stores the diff. Timeouts follow
`task.toml` (`agent.timeout_sec`); `CHRYS_INNER_TIMEOUT` stops chrys ten minutes
earlier so partial edits are still captured. `--agent-retries 1` re-runs an empty
patch once. When the run ends, the captured session stores are swept into the graph
(`chrys memory sweep`), so the next task can recall this one.

What comes back per task under `runs/deepswe/<agent>/<id>/`:

```
solution.patch, agent_out/agent.log, agent_out/agent_rc, agent_out/chrys_run.json (route marker)
agent_out/chrys/<session>/requirement_clarification/turn_1/…   clarification artifacts
agent_out/chrys/<session>/long_horizon/turn_1/{brief.md,memory-prior.md,semantic-search/…}
agent_out/chrys/<session>/sub_agents/sessions/chrys_pact_*      only when the workspace had a verify command
```

A campaign runs only when the workspace has a deterministic verification command, and
DeepSWE tasks expose none to the agent — so the wrapper image ships the task's own
regression runner: every hidden test patch adds a `test.sh` with `base` and `new`
modes, and `gen_instances.py` copies it to `/opt/deepswe_verify.sh` with the hidden
tests' names replaced by placeholder paths (`evaluation/deepswe/verify.py`:
`sanitize_runner`), setting `CHRYS_PACT_VERIFY_COMMAND="bash /opt/deepswe_verify.sh base"`.
A task without such a runner gets the base run parsed out of the verifier's
`tests/test.sh`, else a language default (`go test ./...`, `python -m pytest -q -x`,
`npm test --silent`, `cargo test -q`) — which, measured at the base commit, fails for
half of the first twenty tasks (missing optional dependencies, `go.work` layouts,
snapshot suites), blocking the campaign before it starts. That command is
what the campaign's Worker/Reviewer loop runs to accept each mission (through
`chrys.pact.verify_shim`, which lends pact_core's fresh verification worktrees the
workspace's ignored `node_modules`/`.venv`/`target`). Clarification's goal
contract and initial plan (`06-pact-input/`, copied to `.pact-io/chrys-pact/<id>/` in the
workspace) are the campaign's input. A repository whose own suite fails at the base commit
blocks its campaign; the turn then answers with the repaired baseline.

## 3. Grade with the Harbor verifier

```bash
python3 /path/to/chrys/evaluation/deepswe/lolbench/grade.py \
  --repo-root benchmarks/deepswe --patches-dir patches/deepswe/chrys-lh-deepseek \
  --out-dir runs/deepswe/chrys-lh-deepseek/grades --jobs 2
```

For every instance this builds the task's verifier image from `tests/Dockerfile`
(cached), runs `/tests/test.sh` with the patch at `/logs/artifacts/model.patch`,
`--network=none`, the task's cpu/memory limits and `verifier.timeout_sec`, and reads
`/logs/verifier/reward.json`: `reward` (1 iff every fail-to-pass test passes and no
pass-to-pass test regresses), pass fractions, `apply_failed`. `results.csv` and
`summary.json` aggregate; `verifier.log` and `logs/verifier/` per instance hold the
raw suite output.

## 4. Report and mail

```bash
python3 /path/to/chrys/evaluation/deepswe/report.py \
  --runs-dir runs/deepswe/chrys-lh-deepseek --grades runs/deepswe/chrys-lh-deepseek/grades \
  --out report.md
GMAIL_APP_PASSWORD=… python3 /path/to/chrys/evaluation/deepswe/send_report_email.py \
  --to you@gmail.com --subject "DeepSWE × LoLBench: chrys long-horizon" report.md
```

`report.py` writes the same four sections per task the live checkpoints used —
requirement clarification and localization, ContextGraph recall and deposit, PACT
execution, and the grade — plus totals. `send_report_email.py` sends it over Gmail
SMTP with an app password from the environment.

## Anti-cheat and fidelity notes

- Generation inherits LoLBench's in-container anti-cheat unchanged: github/pypi and
  their mirrors resolve to 127.0.0.1, the reference solution is not in the image, and a
  captured patch that is a near-copy of the gold is rejected. The model endpoint
  (OpenRouter) and the host's graph stay reachable.
- The hidden tests (`tests/test.patch`) are listed as `eval_tests.patch`, so their paths
  are excluded from the captured diff; the verifier applies them itself.
- The agent image carries the task's regression runner in base mode
  (`/opt/deepswe_verify.sh`, see step 2) with the hidden tests' names blanked. DeepSWE's
  own protocol gives the agent no such script; chrys needs a verify command for the
  campaign, and the repository's own suite is the least artificial one.
- The agent runs with network for the model API; DeepSWE's own protocol runs the agent
  with `network_mode = "no-network"` and a Harbor-hosted model. Grading is identical to
  DeepSWE's (`tests/test.sh` in the verifier image, no network).
- Task images are `x86_64` Debian bookworm; run this on an x86_64 host.
- The agent runs as root inside its container, so everything it writes under
  `runs/deepswe/<agent>/<id>/agent_out/` (the captured session store included) is
  root-owned on the host. Reading is fine; to delete a run use
  `docker run --rm -v "$PWD/runs:/r" alpine rm -rf /r/deepswe`.
