#!/usr/bin/env bash
# In-container chrys run wrapper for DeepSWE under LoLBench (installed at
# /opt/chrys_run.sh by agent/install.sh). lolbench_eval.py runs it via
# `bash -c "$AGENT_CMD"` with WORKSPACE + PROMPT_FILE set, cwd=$WORKSPACE, and
# /agent_out mounted back to the host run dir. It edits the repo in $WORKSPACE on the
# long-horizon track; lolbench captures the git diff.
set -u
export HOME=/root
export PATH=/root/.local/bin:$PATH
export CHRYS_MODEL_PROFILE=deepswe-lh
export CHRYS_SEMANTIC_SEARCH_MODEL_PROFILE=deepswe-lh
export CHRYS_SEMANTIC_SEARCH_LOCALIZATION_TIMEOUT="${CHRYS_SEMANTIC_SEARCH_LOCALIZATION_TIMEOUT:-1800}"
# The session store lands in the captured mount: clarification, localization, the
# brief, the prior and the campaign stream all come back to the host with the patch.
export CHRYS_SESSION_ROOT_DIR=/agent_out/chrys
# Deposits need the ContextGraph worker, which lives on the host; the host sweeps the
# captured session store afterwards (run_generation.sh). Recall still works in here.
export CHRYS_MEMORY_WRITEBACK_ON_END=0
# The workspace is /app in every task image; label deposits and recall by the task instead.
export CHRYS_MEMORY_REPO_LABEL="${INSTANCE_ID:-}"
export CHRYS_SESSION_TITLE_AUTO=0
# The container is capped at a few CPUs and 7 GiB, but toolchains size their worker
# pools from the host's core count (jest spawned 95 workers on a 96-core host and
# was OOM-killed). Node reads the CPU affinity mask (the engine's --cpuset-cpus);
# Go, Cargo and make read these.
export GOMAXPROCS="${GOMAXPROCS:-4}" CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}" MAKEFLAGS="${MAKEFLAGS:--j4}"
export VITEST_MAX_THREADS="${VITEST_MAX_THREADS:-4}" VITEST_MAX_FORKS="${VITEST_MAX_FORKS:-4}"
mkdir -p /agent_out/chrys

# chrys run --json prints only the final result; keep lolbench's idle watchdog fed
# with a heartbeat that reflects real session-store activity.
( while true; do
    sleep 120
    echo "[chrys-hb $(date -u +%H:%M:%S)] session_kb=$(du -sk /agent_out/chrys 2>/dev/null | cut -f1)"
  done ) &
HB=$!

cd "$WORKSPACE" || exit 3
# lolbench captures `git add -A && git diff --cached` against HEAD; anything chrys
# committed would vanish from the patch. Remember where we started and unwind
# commits (keeping their changes) before the capture.
START_HEAD="$(git rev-parse HEAD 2>/dev/null || true)"
# Inner timeout strictly shorter than lolbench's --agent-timeout: chrys is stopped
# gracefully and this wrapper returns, so lolbench still captures partial edits.
INNER_T="${CHRYS_INNER_TIMEOUT:-4800}"
timeout --kill-after=90 --signal=TERM "$INNER_T" \
  /opt/chrys/.venv/bin/chrys run --task "$PROMPT_FILE" --agent Code --route long-horizon \
    --workdir "$WORKSPACE" --json > /agent_out/chrys_run.json
rc=$?
[ "$rc" = 124 ] && echo "[chrys] inner timeout (${INNER_T}s) reached — capturing partial edits" >&2
# The campaign's inputs and canonical state live in the workspace; keep them with the
# run for diagnosis and take them out of the captured diff (they are not the solution).
for d in .pact-io .pact; do
  if [ -d "$WORKSPACE/$d" ]; then mkdir -p /agent_out/pact && cp -a "$WORKSPACE/$d" "/agent_out/pact/${d#.}" && rm -rf "${WORKSPACE:?}/$d"; fi
done
if [ -n "$START_HEAD" ] && [ "$(git rev-parse HEAD 2>/dev/null)" != "$START_HEAD" ]; then
  echo "[chrys] unwinding $(git rev-list --count "$START_HEAD"..HEAD) commit(s) into the working tree for capture" >&2
  git reset -q --soft "$START_HEAD"
fi
kill "$HB" 2>/dev/null || true
echo "[chrys] rc=$rc"
exit $rc
