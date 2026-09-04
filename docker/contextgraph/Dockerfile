# ContextGraph memory graph for Chrys: Neo4j 5.26 with the CAPBench selected-Harbor
# graph (2,525 canonical rules, their trajectories and fragments, vector + full-text
# indexes) loaded on first start.
#
#   docker build -t wzh4464/contextgraph:capbench-harbor --build-arg DUMP=neo4j.dump docker/contextgraph
#   docker run -d --name contextgraph -p 7705:7687 -p 7495:7474 \
#     -e NEO4J_AUTH=neo4j/contextgraph123 wzh4464/contextgraph:capbench-harbor
#
# The dump is copied, not loaded, at build time, so one image serves every
# architecture the official neo4j image does; the entrypoint loads it into an empty
# data volume the first time the container starts (a minute or two), then hands over
# to Neo4j's own entrypoint. A populated volume is left alone.
FROM neo4j:5.26-community
ARG DUMP=neo4j.dump
COPY ${DUMP} /contextgraph/neo4j.dump
COPY entrypoint.sh /contextgraph/entrypoint.sh
RUN chmod +x /contextgraph/entrypoint.sh
ENV NEO4J_AUTH=neo4j/contextgraph123 \
    NEO4J_server_memory_heap_initial__size=2g \
    NEO4J_server_memory_heap_max__size=4g \
    NEO4J_server_memory_pagecache_size=2g \
    NEO4J_dbms_security_procedures_unrestricted=apoc.* \
    NEO4J_PLUGINS='["apoc"]'
LABEL org.opencontainers.image.title="ContextGraph (CAPBench selected-Harbor)" \
      org.opencontainers.image.description="Neo4j 5.26 preloaded with the ContextGraph memory graph Chrys recalls from and deposits into" \
      org.opencontainers.image.source="https://github.com/wzh4464/ContextGraph"
ENTRYPOINT ["/contextgraph/entrypoint.sh"]
CMD ["neo4j"]
