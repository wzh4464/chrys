#!/usr/bin/env bash
# Start only the ContextGraph HTTP query layer; the validated Neo4j graph is
# externally managed and is never rebuilt or populated by this hook.
set -euo pipefail

[ "${CHRYS_AUTOSTART_CONTEXTGRAPH:-0}" = "1" ] || exit 0

CG_REPO="${CG_REPO:?set CG_REPO in ~/.chrys/.env}"
SERVER_PYTHON="${CONTEXTGRAPH_SERVER_PYTHON:-$CG_REPO/.venv/bin/python}"
SERVER_PORT="${CONTEXTGRAPH_SERVER_PORT:-8010}"
SERVER_URL="${CONTEXTGRAPH_SERVER_URL:-http://127.0.0.1:$SERVER_PORT}"
SERVER_LOG="${CONTEXTGRAPH_SERVER_LOG:-${TMPDIR:-/tmp}/chrys-contextgraph-query.log}"

is_ready() {
  health="$(curl -fsS --noproxy '*' "$SERVER_URL/health" 2>/dev/null)" || return 1
  case "$health" in
    *'"neo4j_connected":true'* | *'"neo4j_connected": true'*) return 0 ;;
    *) return 1 ;;
  esac
}

[ -x "$SERVER_PYTHON" ] || {
  echo "ContextGraph Python is not executable: $SERVER_PYTHON" >&2
  exit 1
}
[ -f "$CG_REPO/scripts/baselines/contextgraph_pro_server.py" ] || {
  echo "ContextGraph query server not found under: $CG_REPO" >&2
  exit 1
}

if is_ready; then
  exit 0
fi

(
  cd "$CG_REPO"
  set -a
  [ ! -f .env ] || . ./.env
  set +a
  export NEO4J_URI="${CONTEXTGRAPH_NEO4J_URI:-bolt://127.0.0.1:7705}"
  export NEO4J_USER="${CONTEXTGRAPH_NEO4J_USER:-${NEO4J_USER:-neo4j}}"
  export NEO4J_PASSWORD="${CONTEXTGRAPH_NEO4J_PASSWORD:-${NEO4J_PASSWORD:-}}"
  export EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-large}"
  export NO_PROXY='*'
  export no_proxy='*'
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
  nohup "$SERVER_PYTHON" scripts/baselines/contextgraph_pro_server.py \
    --host 127.0.0.1 --port "$SERVER_PORT" </dev/null >"$SERVER_LOG" 2>&1 &
)

attempt=0
while [ "$attempt" -lt 60 ]; do
  if is_ready; then
    exit 0
  fi
  attempt=$((attempt + 1))
  sleep 1
done

echo "ContextGraph query service did not become ready; see $SERVER_LOG" >&2
exit 1
