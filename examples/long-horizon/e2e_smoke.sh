#!/usr/bin/env bash
# Live end-to-end smoke test for the long-horizon track.
#
# Proves the pieces a unit test cannot: that a routed turn really runs the
# baseline AND the repair, that the brief and the PACT inputs land where the
# campaign expects them, and that idle writeback actually reaches the graph.
#
# Requires a real model. The memory half additionally requires a running Neo4j
# and is skipped, loudly, when one is not configured.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CHRYS_REPO="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }
note() { printf '==> %s\n' "$*"; }

# `Code` routes long-horizon work to the built-in LongHorizon profile, which is
# the switch this test is here to exercise. Override either when running the
# smoke against a different setup.
AGENT="${CHRYS_SMOKE_AGENT:-Code}"
MODEL="${CHRYS_SMOKE_MODEL:-}"
MODEL_ARGS=()
[ -n "$MODEL" ] && MODEL_ARGS=(-m "$MODEL")

# ---------------------------------------------------------------- workspace
note "preparing a throwaway workspace at $WORKDIR"
mkdir -p "$WORKDIR/src" "$WORKDIR/tests" "$WORKDIR/.chrys"
cat > "$WORKDIR/src/parser.py" <<'PY'
def parse_value(value):
    return value
PY
cat > "$WORKDIR/src/types.py" <<'PY'
Value = str
PY
cat > "$WORKDIR/tests/test_parser.py" <<'PY'
from src.parser import parse_value


def test_parse_value():
    assert parse_value("1") == "1"
PY
# The workspace owns its own interpreter and dependencies, through uv. Leaning
# on a system `python3` that happens to have pytest makes the campaign's verify
# depend on the host rather than on the repository under test -- and on a
# Homebrew/uv machine there is often no bare `python` at all.
cat > "$WORKDIR/pyproject.toml" <<'TOML'
[project]
name = "long-horizon-smoke"
version = "0"
requires-python = ">=3.11"

[dependency-groups]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
# The tests import `src.parser`, so the workspace root has to be importable
# regardless of how pytest was invoked.
pythonpath = ["."]
TOML
printf '.venv/\n__pycache__/\n.pytest_cache/\n' > "$WORKDIR/.gitignore"

# A verify command is what makes the workspace able to carry a campaign at all.
VERIFY="uv run --frozen pytest -q"
cat > "$WORKDIR/.chrys/settings.yaml" <<YAML
pact:
  verify_command: "$VERIFY"
YAML
# The project layer is off unless the user turned it on, and a smoke test must
# not need them to grant a throwaway directory configuration authority. Export
# the same command so this runs either way.
export CHRYS_PACT_VERIFY_COMMAND="$VERIFY"

note "provisioning the workspace interpreter with uv"
(cd "$WORKDIR" && uv sync --quiet) || fail "could not provision the smoke workspace with uv"
(cd "$WORKDIR" && uv run --frozen pytest -q >/dev/null) \
  || fail "the workspace verify command does not pass on the untouched baseline"

git -C "$WORKDIR" init -q
git -C "$WORKDIR" add -A
git -C "$WORKDIR" -c user.email=smoke@example.invalid -c user.name=Smoke commit -qm "baseline"

# ------------------------------------------------------------------ routing
note "checking the router classifies the prompt as long-horizon"
PROMPT='Implement end-to-end typed parsing: add the parser abstraction, migrate every caller, and write integration tests. Acceptance criteria: 1) existing behaviour is unchanged 2) all tests pass. Touch src/parser.py, src/types.py and tests/test_parser.py.'
BAND="$(cd "$CHRYS_REPO" && uv run chrys debug router --json -C "$WORKDIR" "$PROMPT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["band"])')"
[ "$BAND" = "strong_long_horizon" ] || fail "expected strong_long_horizon, got $BAND"

# --------------------------------------------------------------------- turn
note "running one routed turn (this calls a real model and takes a while)"
OUT="$WORKDIR/run.json"
(cd "$CHRYS_REPO" && uv run chrys run -a "$AGENT" "${MODEL_ARGS[@]}" --route long-horizon --json \
  -C "$WORKDIR" "$PROMPT") > "$OUT" || fail "the routed run did not complete"
SESSION="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["session_id"])' "$OUT")"
TRACK="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("route",{}).get("track",""))' "$OUT")"
[ "$TRACK" = "long_horizon" ] || fail "the run reported track=$TRACK"

SESSION_DIR="$(cd "$CHRYS_REPO" && uv run python -c '
import sys
from chrys.foundation.config.settings import resolve_sessions_dir
print(resolve_sessions_dir(create=False) / sys.argv[1])
' "$SESSION")"

# ---------------------------------------------------------------- artifacts
note "checking both coding passes really ran"
[ -d "$SESSION_DIR/requirement_clarification/turn_1/02-initial-trial" ] \
  || fail "no baseline pass artifacts: the turn did not run P0"
[ -d "$SESSION_DIR/requirement_clarification/turn_1/04-repair" ] \
  || fail "no repair artifacts: the turn stopped at P0"

note "checking the task brief carries the code search"
BRIEF="$SESSION_DIR/long_horizon/turn_1/brief.md"
[ -f "$BRIEF" ] || fail "no task brief at $BRIEF"
grep -q "Code localization" "$BRIEF" || fail "the brief has no localization section"

note "checking the campaign inputs landed in the workspace"
CONTRACT="$(find "$WORKDIR/.pact-io/chrys-pact" -name goal-contract.json -print -quit 2>/dev/null || true)"
[ -n "$CONTRACT" ] || fail "no goal-contract.json under .pact-io/"
python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$CONTRACT" || fail "the goal contract is not valid JSON"

# ------------------------------------------------------------------- memory
if [ -z "${CONTEXTGRAPH_NEO4J_URI:-}" ]; then
  note "SKIP: CONTEXTGRAPH_NEO4J_URI is not set, so the writeback half is not exercised"
  note "PASS: the long-horizon track ran end to end"
  exit 0
fi

note "checking idle writeback reaches the graph"
(cd "$CHRYS_REPO" && uv run chrys memory doctor) || fail "chrys memory doctor reported a problem"
BEFORE="$(cd "$CHRYS_REPO" && uv run chrys memory sweep --dry-run --idle-seconds 0 | grep -c "$SESSION" || true)"
[ "$BEFORE" != "0" ] || fail "the swept session has no pending turns; writeback would deposit nothing"
(cd "$CHRYS_REPO" && uv run chrys memory sweep --idle-seconds 0) || fail "the sweep failed"
AFTER="$(cd "$CHRYS_REPO" && uv run chrys memory sweep --dry-run --idle-seconds 0 | grep -c "$SESSION" || true)"
[ "$AFTER" = "0" ] || fail "the watermark did not advance: the turn was not deposited"

note "PASS: the long-horizon track ran end to end and its turn reached the graph"
