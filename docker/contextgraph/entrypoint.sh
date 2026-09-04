#!/bin/bash
# First start: load the bundled dump into the (empty) data directory, then defer to
# the official Neo4j entrypoint. Later starts, or a volume that already holds a
# database, skip the load.
set -euo pipefail
DATA=/data/databases/neo4j
if [ ! -d "$DATA" ] || [ -z "$(ls -A "$DATA" 2>/dev/null)" ]; then
  echo "[contextgraph] loading the bundled graph into $DATA (first start)"
  mkdir -p /tmp/contextgraph-dump
  ln -sf /contextgraph/neo4j.dump /tmp/contextgraph-dump/neo4j.dump
  neo4j-admin database load neo4j --from-path=/tmp/contextgraph-dump --overwrite-destination=true
  echo "[contextgraph] loaded"
else
  echo "[contextgraph] data volume already holds a database; not loading the bundled dump"
fi
exec /startup/docker-entrypoint.sh "$@"
