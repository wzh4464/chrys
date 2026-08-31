# DeepSWE requirement-clarification evaluation

This directory evaluates the requirement-clarification workflow through the same public `chrys run` path used by a
normal headless Chrys session. It does not reimplement the product workflow in benchmark code.

The primary experiment is a paired comparison:

```text
same task + same Chrys commit + same binary + same model + same Code profile
                          │
              ┌───────────┴───────────┐
              │                       │
       control profile         clarification profile
       enabled: false          enabled: true
              │                       │
          one normal turn       P0 → ΔR → repair
              │                       │
              └──────── DeepSWE verifier ────────┘
```

The generated profiles are checked to differ only in identity fields and
`requirement_clarification.enabled`. Both arms are locked to the committed
`deepseek-v4-pro-0813-openrouter` model profile. Harbor grants agent-phase network access only to `openrouter.ai`;
verification remains offline.

The Harbor runners do not contact OpenRouter unless `--execute` is present. Image construction requires either the
dedicated reviewed-manifest builder or the materializer's explicit `--build-images` flag.

## Prerequisites

- A local DeepSWE Harbor dataset. The production comparison expects all 113 tasks.
- Harbor with its virtual environment installed.
- Docker and the task images when executing, but not for dry-run preparation.
- A self-contained Chrys binary. DeepSWE images do not consistently contain Python 3.14, so using the source checkout
  inside the containers is not supported.
- `.chrys-secrets.env` at the Chrys repository root with `OPENROUTER_API_KEY` and the pinned `CHRYS_MODEL_LOCK`. The file
  is gitignored and is never copied into an experiment directory or container.

Build the self-contained binary from the exact commit being evaluated:

```bash
./scripts/build.sh --offline
```

The resulting input is normally `dist/chrys`. The experiment manifest records its SHA-256 independently of the Git
revision.

## 1. Prepare the control/clarification pair

From the Chrys repository root:

```bash
uv run python -m evaluation.requirement_clarification.run_pair \
  --harbor-repo /path/to/harbor \
  --dataset /path/to/harbor/dataset/deepswe \
  --chrys-binary ./dist/chrys \
  --secrets ./.chrys-secrets.env \
  --output-dir /path/to/experiments/deepswe-rc-0831 \
  --run-id deepswe-rc-0831
```

This is a dry run. It validates and hashes every task, renders the paired profiles, checks the key and model-lock
presence without printing either value, and writes `manifest.json`. Review that manifest before spending tokens.

To run both arms, repeat the command with `--execute`. To run only one arm, add `--arm control` or
`--arm clarification`. Default concurrency is two; change it with `--concurrency`.

If a process stops after the Harbor job directory exists, repeat the same command with `--execute --resume`. Harbor
will continue its recorded job rather than creating a replacement. The runner refuses to overwrite any job unless
resume was explicitly selected.

Each trial retains:

- the original instruction;
- Chrys stdout and stderr;
- a score-free `experiment.json` with binary/profile hashes and the session id;
- the complete Chrys session tree, including private P0/ΔR/workflow artifacts for the clarification arm;
- `model.patch` and the normal DeepSWE verifier output.

The API key is passed as an execution environment value, never a command argument. Project dotenv loading is disabled
inside the task container, so repository-controlled `.env` files cannot replace the key or routing configuration.

## 2. Summarize and compare

Summarize one job:

```bash
uv run python -m evaluation.requirement_clarification.summarize \
  --job /path/to/experiments/deepswe-rc-0831/jobs/deepswe-rc-0831-control \
  --output /path/to/experiments/deepswe-rc-0831/control-summary.json
```

Create the strict paired comparison:

```bash
uv run python -m evaluation.requirement_clarification.summarize \
  --control /path/to/experiments/deepswe-rc-0831/jobs/deepswe-rc-0831-control \
  --candidate /path/to/experiments/deepswe-rc-0831/jobs/deepswe-rc-0831-clarification \
  --output /path/to/experiments/deepswe-rc-0831/comparison.json
```

The comparison fails on non-identical task sets. It reports solved counts, gain/regression task lists, net delta, an
exact two-sided McNemar p-value, selected retry attempts, patch hashes, clarification outcomes, and ΔR hashes. Scoring
is strictly post-run; no verifier result is available to Chrys.

## 3. Fixed-P0 diagnostic

The primary result above measures the complete product, including the extra model work and a separately sampled P0.
The fixed-P0 diagnostic asks a narrower question: given the exact control patch P0, does ΔR-guided repair improve it?

The candidate workflow's clarifier is repository-grounded in S0 and the original requirement/history; it cannot see
P0. Therefore its persisted ΔR can be paired with the control P0 without feeding candidate implementation details into
repair. Tasks without a persisted non-empty ΔR are excluded and listed with a reason.

First materialize immutable task copies and Docker build contexts:

```bash
uv run python -m evaluation.requirement_clarification.materialize_fixed_p0 \
  --source-dataset /path/to/harbor/dataset/deepswe \
  --control-job /path/to/experiments/deepswe-rc-0831/jobs/deepswe-rc-0831-control \
  --candidate-job /path/to/experiments/deepswe-rc-0831/jobs/deepswe-rc-0831-clarification \
  --output-dir /path/to/experiments/deepswe-rc-0831/fixed-p0
```

This step does not run Docker. The materializer is intentionally write-once and refuses to overwrite a directory.
Review `fixed-p0/manifest.json`, then build exactly those hashed contexts:

```bash
uv run python -m evaluation.requirement_clarification.build_fixed_p0_images \
  --manifest /path/to/experiments/deepswe-rc-0831/fixed-p0/manifest.json
```

The builder rejects changed P0 patches and command/input mismatches. A repeated `--task TASK_NAME` limits a pilot to a
reviewed subset.

Run repair with clarification disabled so the supplied ΔR is not recursively clarified:

```bash
uv run python -m evaluation.requirement_clarification.run_fixed_p0 \
  --harbor-repo /path/to/harbor \
  --dataset /path/to/experiments/deepswe-rc-0831/fixed-p0/dataset \
  --materialization-manifest /path/to/experiments/deepswe-rc-0831/fixed-p0/manifest.json \
  --chrys-binary ./dist/chrys \
  --secrets ./.chrys-secrets.env \
  --output-dir /path/to/experiments/deepswe-rc-0831/fixed-p0-run \
  --run-id deepswe-rc-0831
```

Again, this is dry-run only until `--execute` is added. Use the same `summarize` command on its Harbor job. Compare its
task rewards to the control arm only over the eligible subset recorded by the materialization manifest.

## Protocol boundaries

- The primary control/candidate result is the product-level benchmark.
- Fixed-P0 is a causal diagnostic, not a replacement headline score. Its repair prompt is an experimental direct run,
  while the candidate arm exercises the native integrated workflow.
- A failed/degraded native clarification may legitimately return P0. That remains a candidate outcome and is not an
  infrastructure retry by itself.
- Harbor retries are collapsed by task using the latest successful attempt, with attempt counts retained.
- Manifests contain hashes and command lines but no secrets and no verifier scores.
