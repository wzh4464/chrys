#!/usr/bin/env bash
# Generate patches for DeepSWE tasks with chrys on the long-horizon track, through
# LoLBench's in-container evaluator (anti-cheat, trajectory capture, patch store).
#
# Run from a LoLBench checkout on a Linux host with Docker:
#
#   evaluation/deepswe/lolbench/run_generation.sh <instances-or-"all"> [extra lolbench args]
#
# Environment:
#   CHRYS_SRC        chrys source tree to ship into the image (default: this checkout)
#   BENCH_ROOT       benchmark root written by gen_instances.py (default: benchmarks/deepswe)
#   AGENT_NAME       run/patch-store label (default: chrys-lh-deepseek)
#   JOBS             instances in flight (default: 4)
#   AGENT_TIMEOUT    seconds per instance (default: 5400 = task.toml agent.timeout_sec)
#   CHRYS_SRV_PORT   host port serving the source tarball to image builds (default: 8731)
#   OPENROUTER_API_KEY, CONTEXTGRAPH_NEO4J_PASSWORD, CONTEXTGRAPH_EMBEDDING_API_KEY,
#   CONTEXTGRAPH_EMBEDDING_BASE_URL, CONTEXTGRAPH_EMBEDDING_MODEL  (secrets; forwarded, never baked)
#   CONTEXTGRAPH_BOLT_PORT  host-side Neo4j bolt port bridged to the containers (default: 7705)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOLBENCH="$(pwd)"
[ -f "$LOLBENCH/scripts/lolbench_eval.py" ] || { echo "run from a LoLBench checkout (scripts/lolbench_eval.py not found)"; exit 2; }
CHRYS_SRC="${CHRYS_SRC:-$(cd "$HERE/../../.." && pwd)}"
BENCH_ROOT="${BENCH_ROOT:-benchmarks/deepswe}"
AGENT_NAME="${AGENT_NAME:-chrys-lh-deepseek}"
JOBS="${JOBS:-4}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-5400}"
CHRYS_SRV_PORT="${CHRYS_SRV_PORT:-8731}"
CONTEXTGRAPH_BOLT_PORT="${CONTEXTGRAPH_BOLT_PORT:-7705}"
: "${OPENROUTER_API_KEY:?set OPENROUTER_API_KEY}"
INSTANCES="${1:-all}"; shift || true

# The docker bridge is how a container reaches this host at build and run time.
BRIDGE_IP="$(ip -4 addr show docker0 2>/dev/null | grep -o 'inet [0-9.]*' | awk '{print $2}')"
BRIDGE_IP="${BRIDGE_IP:-172.17.0.1}"

# --- serve the chrys source over HTTP for the image builds (private repo) ---
SRV_DIR=/tmp/deepswe_chrys_srv; mkdir -p "$SRV_DIR"
PIN="$(git -C "$CHRYS_SRC" rev-parse --short HEAD 2>/dev/null || date +%s)"
tar czf "$SRV_DIR/chrys_src.tgz" --exclude=.git --exclude=.venv --exclude=tests --exclude=playground \
  --exclude='src/chrys/foundation/vendor/ripgrep/*' --exclude='.semantic-search*' -C "$CHRYS_SRC" .
( cd "$SRV_DIR" && exec python3 -m http.server "$CHRYS_SRV_PORT" --bind 0.0.0.0 ) >/tmp/deepswe_chrys_srv.log 2>&1 &
SRV_PID=$!
# --- bridge the host's loopback-only Neo4j to the docker bridge for recall ---
SOCAT_PID=""
if command -v socat >/dev/null && ! ss -ltn 2>/dev/null | grep -q "$BRIDGE_IP:$CONTEXTGRAPH_BOLT_PORT"; then
  socat "TCP-LISTEN:$CONTEXTGRAPH_BOLT_PORT,bind=$BRIDGE_IP,fork,reuseaddr" "TCP:127.0.0.1:$CONTEXTGRAPH_BOLT_PORT" >/dev/null 2>&1 &
  SOCAT_PID=$!
fi
cleanup() { kill "$SRV_PID" 2>/dev/null || true; [ -n "$SOCAT_PID" ] && kill "$SOCAT_PID" 2>/dev/null || true; }
trap cleanup EXIT
sleep 1; kill -0 "$SRV_PID" || { echo "source server failed (port $CHRYS_SRV_PORT busy?)"; cat /tmp/deepswe_chrys_srv.log; exit 1; }

# --- the install recipe, with placeholders filled and base64-wrapped for RUN ---
CG_ENV="$(printf 'CONTEXTGRAPH_NEO4J_URI=bolt://%s:%s\nCONTEXTGRAPH_NEO4J_USER=%s\nCONTEXTGRAPH_EMBEDDING_MODEL=%s\nCONTEXTGRAPH_EMBEDDING_BASE_URL=%s\n' \
  "$BRIDGE_IP" "$CONTEXTGRAPH_BOLT_PORT" "${CONTEXTGRAPH_NEO4J_USER:-neo4j}" "${CONTEXTGRAPH_EMBEDDING_MODEL:-text-embedding-3-large}" "${CONTEXTGRAPH_EMBEDDING_BASE_URL:-}")"
INSTALL="$(sed -e "s|__CHRYS_SRC_URL__|http://$BRIDGE_IP:$CHRYS_SRV_PORT/chrys_src.tgz|" \
               -e "s|__MODEL_B64__|$(base64 < "$HERE/agent/model.yaml" | tr -d '\n')|" \
               -e "s|__RUNSH_B64__|$(base64 < "$HERE/agent/run.sh" | tr -d '\n')|" \
               -e "s|__CG_ENV_B64__|$(printf '%s\n' "$CG_ENV" | base64 | tr -d '\n')|" \
               -e "s|__CHRYS_PIN__|$PIN|" "$HERE/agent/install.sh")"
INSTALL_CMD="printf '%s' '$(printf '%s' "$INSTALL" | base64 | tr -d '\n')' | base64 -d | bash"

if [ "$INSTANCES" = "all" ]; then
  INSTANCES="$(python3 - "$BENCH_ROOT" <<'PY'
import csv, sys
from pathlib import Path
print(",".join(r["instance_id"] for r in csv.DictReader(open(Path(sys.argv[1]) / "data" / "lolbench_clean.csv"))))
PY
)"
fi

export CHRYS_INNER_TIMEOUT=$(( AGENT_TIMEOUT > 700 ? AGENT_TIMEOUT - 600 : AGENT_TIMEOUT - 60 ))
export CONTEXTGRAPH_NEO4J_PASSWORD="${CONTEXTGRAPH_NEO4J_PASSWORD:-}" CONTEXTGRAPH_EMBEDDING_API_KEY="${CONTEXTGRAPH_EMBEDDING_API_KEY:-}"
LOLBENCH_IDLE_LIMIT="${LOLBENCH_IDLE_LIMIT:-3600}" \
python3 scripts/lolbench_eval.py --in-container --skip-grade --no-stdlib-strip \
  --repo-root "$BENCH_ROOT" --out-dir "runs/deepswe" --patches-dir "patches/deepswe" \
  --agent-name "$AGENT_NAME" \
  --agent-install "$INSTALL_CMD" \
  --agent-env OPENROUTER_API_KEY --agent-env CHRYS_INNER_TIMEOUT \
  --agent-env CONTEXTGRAPH_NEO4J_PASSWORD --agent-env CONTEXTGRAPH_EMBEDDING_API_KEY \
  --agent-cmd 'bash /opt/chrys_run.sh' \
  --agent-timeout "$AGENT_TIMEOUT" --jobs "$JOBS" --agent-retries 1 \
  --instances "$INSTANCES" "$@"

# --- deposit the captured sessions into the graph from the host ---
echo "### sweeping captured chrys sessions into ContextGraph ###"
for d in runs/deepswe/"$AGENT_NAME"/*/agent_out/chrys; do
  [ -d "$d" ] || continue
  ( cd "$CHRYS_SRC" && CHRYS_SESSION_ROOT_DIR="$d" uv run chrys memory sweep --idle-seconds 0 ) || true
done
