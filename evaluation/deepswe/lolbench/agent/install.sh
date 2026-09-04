#!/usr/bin/env bash
# Build-time install recipe for the in-container chrys agent image.
#
# lolbench_eval.py runs this as one `RUN` layer on top of the DeepSWE task image
# (Debian bookworm). At build time the container reaches the host, PyPI, astral and
# GitHub; at run time github/pypi are black-holed, so everything the agent needs is
# installed here: uv + Python 3.14, the chrys source (served by run_generation.sh from
# the host over HTTP, since the repository is private), ripgrep, CodeGraph, the model
# profile and the ContextGraph connection settings.
#
# Expanded by run_generation.sh with these placeholders:
#   __CHRYS_SRC_URL__   http://<host bridge ip>:<port>/chrys_src.tgz
#   __MODEL_B64__       base64 of agent/model.yaml
#   __RUNSH_B64__       base64 of agent/run.sh
#   __CG_ENV_B64__      base64 of the non-secret CONTEXTGRAPH_* lines for ~/.chrys/.env
#   __CHRYS_PIN__       short commit of the served source (part of the image cache key)
set -e
echo "chrys pin __CHRYS_PIN__"
export DEBIAN_FRONTEND=noninteractive HOME=/root
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq curl ca-certificates xz-utils git ripgrep >/dev/null 2>&1 || true
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=/root/.local/bin:$PATH
mkdir -p /opt/chrys
curl -fsSL --retry 5 __CHRYS_SRC_URL__ -o /tmp/chrys_src.tgz
tar xzf /tmp/chrys_src.tgz -C /opt/chrys
cd /opt/chrys
uv python install 3.14
uv sync --extra all
/opt/chrys/.venv/bin/python -c "import chrys" || { echo CHRYS_IMPORT_FAILED >&2; exit 1; }
/opt/chrys/.venv/bin/chrys run --help >/dev/null 2>&1 || { echo CHRYS_RUN_HELP_FAILED >&2; exit 1; }
# A vendored rg from the build machine is skipped by find_rg when it cannot run here;
# the apt ripgrep above is the fallback on PATH.
/opt/chrys/.venv/bin/python -c "from chrys.foundation.vendor import find_rg; p = find_rg(); assert p, 'no rg'; print('rg:', p)"
# CodeGraph (optional perception for localization). GitHub is reachable at build time.
CODEGRAPH_BIN_DIR=/root/.local/bin sh -c "$(curl -fsSL https://raw.githubusercontent.com/colbymchenry/codegraph/main/install.sh)" || echo "codegraph install skipped" >&2
CODEGRAPH_TELEMETRY=0 /root/.local/bin/codegraph telemetry off >/dev/null 2>&1 || true
mkdir -p /root/.chrys/models
printf '%s' '__MODEL_B64__' | base64 -d > /root/.chrys/models/deepswe-lh.yaml
printf '%s' '__CG_ENV_B64__' | base64 -d > /root/.chrys/.env
printf '%s' '__RUNSH_B64__' | base64 -d > /opt/chrys_run.sh
chmod +x /opt/chrys_run.sh
echo CHRYS_INSTALL_OK
