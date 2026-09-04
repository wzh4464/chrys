---
name: semantic-search
description: Localize task-relevant code with CodeGraph and repository evidence, then generate a compact Augmented Requirement before Chrys implements a code task.
---

# Semantic Search

Use this skill to turn the user's original code-task requirement into a ranked **Code Localization** report, then use that report to create one compact **Augmented Requirement** for Chrys. The original requirement remains authoritative and is preserved verbatim. Locations are inspection candidates, not automatic edit mandates.

Semantic search is not a patch generator and not a second coding agent. It prepares code locations and task context; Chrys remains responsible for source reads, edits, tests, and patch generation.

The public workflow has three phases: build repository perception, localize code, then generate the compact Augmented Requirement. Localization uses a Chrys-native SemLoc loop: an LLM explores a normalized CodeGraph/index through five read-only tools with DFS/BFS, then returns one structured ranked result. There is no separate Stage-1/Stage-2 filtering pass. If the model or CodeGraph is unavailable, deterministic index ranking keeps the workflow runnable.

## Workflow

1. **Build Code Perception**
   - Use `scripts/build_code_perception.py`.
   - For one repo, pass `--repo /abs/path/to/repo`, `--requirement /abs/run/PROMPT.md`, and `--artifact-dir /abs/workspace/.semantic-search`.
   - The script emits `.semantic-search/code-perception.json` and `.semantic-search/code-perception.md`.
   - Internally it builds the lightweight index, builtin global perception, optional CodeGraph perception, merged repository perception, and task-specific semantic perception.
   - CodeGraph is optional but auto-managed. If `codegraph` is not on PATH, semantic-search attempts to download the current CodeGraph release under the Chrys repo at `.semantic-search-tools/codegraph/downloads/`, install it into the active semantic Python environment, and run it from that environment's `bin` directory.
   - Set `SEMANTIC_SEARCH_CODEGRAPH_INSTALL=never` to disable auto-install, `SEMANTIC_SEARCH_CODEGRAPH_CMD=/abs/path/to/codegraph` to force a command, or `CODEGRAPH_VERSION=vX.Y.Z` to pin a release.
   - Set `SEMANTIC_SEARCH_CODEGRAPH_DOWNLOAD_DIR=/abs/path` to override the Chrys-local download cache.
   - If the CodeGraph CLI is unavailable, cannot be installed, or returns partial data, the workflow continues with builtin repository perception.
   - `code-facts.json` is the main machine-readable output consumed by requirement augmentation. It includes repository-wide code facts, CodeGraph availability/evidence when present, and task-specific semantic-search facts.
   - Keep `.semantic-search/` and `.codegraph/` out of benchmark patches.

2. **Generate Code Localization**
   - The first script emits `.semantic-search/code-localization.json` and `.semantic-search/code-localization.md`.
   - It also emits `.semantic-search/localization-graph.json` and `.semantic-search/localization-trace.jsonl`.
   - The normalized graph combines CodeGraph relationships with lightweight source relationships such as child, caller/callee, inheritance, alias, implicit/async, and cross-file edges.
   - The localization model uses `find_file`, `find_code_definition`, `find_code_content`, `find_child_unit`, and `finish_search` to perform adaptive DFS/BFS exploration.
   - After `finish_search`, the model returns one standard JSON result with file, class, function, line range, rank, role, reason, confidence, and evidence fields.
   - The output contract is documented by `schemas/code-localization.schema.json`.
   - Use `--localization-mode llm --localization-model-profile PROFILE` on `build_code_perception.py` to require agentic localization. `auto` uses it when configured and otherwise falls back; `fallback` disables it.
   - The report ranks primary, propagation, and validation locations and records why each location matched.
   - Verify every location with Chrys source tools before editing.

3. **Generate the Augmented Requirement**
   - Use `scripts/augment_requirement.py`.
   - Pass `--localization /abs/workspace/.semantic-search/code-localization.json`.
   - The localization mode generates one compact `.semantic-search/augmented-requirement.md`; it keeps the original requirement and adds ranked locations, scope, and validation guidance.
   - In `auto` mode, the active Chrys model may rewrite the task brief from the original requirement and localization evidence. Fallback mode remains deterministic.
   - It emits `.semantic-search/augmented-requirement.md`.
   - It also writes `augmentation/code-localization.md` and `augmentation_routes.json` for traceability.

4. **Implement from the Augmented Requirement**
   - Read `augmented-requirement.md` and the linked localization report.
   - Treat the embedded Original Requirement as canonical when generated guidance conflicts with it.
   - Verify primary and propagation locations with normal Chrys source tools before editing.
   - Use Chrys native tools for edits, tests, and patch generation.
   - Keep `.semantic-search/` artifacts out of the final patch.
   - Verify claims with normal Chrys source inspection and tests.
   - If augmentation appears to expand the task or imply a broad rewrite, fall back to the Original Requirement plus direct repository evidence.

## Script Examples

Build Code Perception:

```text
run_skill_script(
  skill_name="semantic-search",
  script_name="scripts/build_code_perception.py",
  arguments=[
    "--repo", "/abs/workspace",
    "--requirement", "/abs/run/PROMPT.md",
    "--out", "/abs/workspace/.semantic-search/code-perception.json",
    "--markdown", "/abs/workspace/.semantic-search/code-perception.md",
    "--artifact-dir", "/abs/workspace/.semantic-search",
    "--localization-mode", "auto",
    "--localization-model-profile", "my-model-profile"
  ]
)
```

Generate the Augmented Requirement:

```text
run_skill_script(
  skill_name="semantic-search",
  script_name="scripts/augment_requirement.py",
  arguments=[
    "--requirement", "/abs/run/PROMPT.md",
    "--facts", "/abs/workspace/.semantic-search/code-facts.json",
    "--localization", "/abs/workspace/.semantic-search/code-localization.json",
    "--out", "/abs/workspace/.semantic-search/augmented-requirement.md",
    "--augmentation-dir", "/abs/workspace/.semantic-search/augmentation",
    "--artifact-dir", "/abs/workspace/.semantic-search"
  ]
)
```

## Artifact Layout

```text
.semantic-search/
  index.json
  code-perception.json
  code-perception.md
  code-localization.json
  code-localization.md
  localization-graph.json
  localization-trace.jsonl
  global-perception.json
  global-perception.md
  codegraph-perception.json
  codegraph-perception.md
  repository-perception.json
  repository-perception.md
  repo-map.json
  semantic-perception.json
  semantic-perception.md
  code-facts.json
  evidence-bundle.md
  augmentation-prompt.md
  augmentation-llm-response.txt
  augmentation-llm-error.txt
  augmented-requirement.md
  augmentation_routes.json
  manifest.json
  plan-trace.jsonl
  augmentation/
    code-localization.md
```

## Runtime Safety

- Runtime augmentation may read only the original task prompt, the current workspace checkout, and semantic-search artifacts created in the current run.
- It must not read gold patches, gold tests, `requirement_pr_pairs`, previous benchmark patches, target PR diffs, or answer-side network sources.
- Every important augmentation item should carry source, evidence, confidence, action, and risk in the relevant sub-document.
- `Must Implement` sections should stay short and contain only source-verified behavior deltas required by the Original Requirement.
- Missing/new surfaces are first-class: semantic-search can rank only existing files, so the Augmented Requirement must call out required new modules, generated artifacts, registration, exports, metadata, or build updates when the Original Requirement needs them. These are not default edit targets; they require source verification.
- Generated/build safety is first-class: preserve executable bits, file metadata, buildability, existing diagnostics, and pass-to-pass behavior unless the Original Requirement explicitly requires a change.
- `Should Inspect` sections may be rich and code-aware, but they are not edit lists.
- `Do Not Do` and `Validation` sections are first-class scope controls; use them to avoid broad patches, avoid incomplete tiny patches, and preserve pass-to-pass behavior.
- If augmentation conflicts with the original requirement, treat that as an augmentation defect and re-check the original requirement plus repository evidence before acting.
- Candidate implementation surfaces and code details are inspection hints. They are not permission to edit every listed file.
