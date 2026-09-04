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
export CHRYS_SESSION_TITLE_AUTO=0
mkdir -p /agent_out/chrys

# chrys run --json prints only the final result; keep lolbench's idle watchdog fed
# with a heartbeat that reflects real session-store activity.
( while true; do
    sleep 120
    echo "[chrys-hb $(date -u +%H:%M:%S)] session_kb=$(du -sk /agent_out/chrys 2>/dev/null | cut -f1)"
  done ) &
HB=$!

cd "$WORKSPACE" || exit 3
# Inner timeout strictly shorter than lolbench's --agent-timeout: chrys is stopped
# gracefully and this wrapper returns, so lolbench still captures partial edits.
INNER_T="${CHRYS_INNER_TIMEOUT:-4800}"
timeout --kill-after=90 --signal=TERM "$INNER_T" \
  /opt/chrys/.venv/bin/chrys run --task "$PROMPT_FILE" --agent Code --route long-horizon \
    --workdir "$WORKSPACE" --json > /agent_out/chrys_run.json
rc=$?
[ "$rc" = 124 ] && echo "[chrys] inner timeout (${INNER_T}s) reached — capturing partial edits" >&2
kill "$HB" 2>/dev/null || true
echo "[chrys] rc=$rc"
exit $rc
