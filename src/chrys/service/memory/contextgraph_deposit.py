# Copyright (c) 2026 Chrys. All rights reserved.

"""Deposit one persisted Chrys turn through the ContextGraph repository.

This module is intended for an ``after_turn`` hook. That event fires after the
session save, so extraction observes a durable transcript revision. Replays are
idempotent because the source identity includes the selected turn's digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chrys.foundation.config.settings import resolve_sessions_dir
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.turns import turn_slices
from chrys.kernel.exchanges import (
    DictAccessor,
    EmptyIdPolicy,
    NoneIdPolicy,
    PairingPolicy,
    iter_exchanges,
    pair_results,
)
from chrys.service.context.providers.history import TURN_INDEX_KEY
from chrys.service.memory.contextgraph_repository import (
    MAX_REPOSITORY_STEPS,
    RepositoryDepositResult,
    deposit_experience,
)
from chrys.service.state.serializers import deserialize_state

MAX_SESSION_FILE_BYTES = 64 * 1024 * 1024
MAX_ARGUMENT_SUMMARY_CHARS = 800
MAX_RESULT_SUMMARY_CHARS = 1600

_SESSION_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_SENSITIVE_ARGUMENT = re.compile(r"(?:api.?key|authorization|bearer|credential|password|secret|token)", re.IGNORECASE)
_USEFUL_ARGUMENT_KEYS = (
    "command",
    "path",
    "file_path",
    "cwd",
    "query",
    "pattern",
    "name",
    "skill_name",
    "prompt",
    "url",
)
_MEMORY_TOOL_NAMES = frozenset({"team_memory_health", "team_memory_query", "team_memory_record"})
_FAILED_STATUS_CODES = frozenset(
    {
        HistoryMarkerKind.STATUS_EXECUTION_INTERRUPTED,
        HistoryMarkerKind.STATUS_EXECUTION_FAILED,
        HistoryMarkerKind.STATUS_SESSION_CLOSED,
    }
)


@dataclass(frozen=True)
class TurnExperience:
    """A selected persisted turn ready for dynamic deposition."""

    problem_statement: str
    final_response: str
    steps: tuple[dict[str, str], ...]
    turn_digest: str
    success: bool = True
    """Whether the turn reached a normal end, or was verifiably completed.

    A standard turn is successful when it carries no interrupted/failed marker
    -- runtime completion, not verified correctness. A long-horizon turn that
    delegated a campaign is held to the campaign's own ``completed`` status
    instead: the campaign ran the repository's verify command, so its verdict
    is evidence where a clean exit is only an absence of errors.
    """
    route: str = ""
    """``standard`` or ``long_horizon``, from the turn's own route marker.

    Not stored in the graph: ContextGraph's ``RawTrajectory`` has no field for
    it, and inventing one here would fork a schema this repository does not
    own. It is what decides :attr:`success`, which does reach the graph, and it
    is available to anything reading a persisted session.
    """
    campaign_status: str = ""
    """The campaign's reported status, when this turn delegated one."""


_PAIRING_POLICY = PairingPolicy(
    call_types=DictAccessor().call_types(),
    include_informational_calls=False,
    result_types=DictAccessor().result_types(),
    none_id=NoneIdPolicy.POSITIONAL,
    empty_id=EmptyIdPolicy.POSITIONAL,
    malformed_id="stringify",
)


def _json_text(value: object, *, limit: int) -> str:
    if isinstance(value, str):
        return value[:limit]
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)[:limit]
    except TypeError, ValueError:
        return str(value)[:limit]


def _decode_arguments(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except ValueError, RecursionError:
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _argument_summary(call: Mapping[str, Any]) -> str:
    decoded = _decode_arguments(call.get("arguments"))
    if decoded is None:
        raw = call.get("arguments", "")
        return _json_text(raw, limit=MAX_ARGUMENT_SUMMARY_CHARS)
    selected = {
        key: decoded[key] for key in _USEFUL_ARGUMENT_KEYS if key in decoded and not _SENSITIVE_ARGUMENT.search(key)
    }
    return _json_text(selected, limit=MAX_ARGUMENT_SUMMARY_CHARS) if selected else ""


def _result_summary(result: Mapping[str, Any]) -> str:
    for key in ("result", "output", "text", "error", "items"):
        value = result.get(key)
        if value not in (None, "", []):
            return _json_text(value, limit=MAX_RESULT_SUMMARY_CHARS)
    return ""


def _paired_steps(messages: list[dict[str, Any]]) -> tuple[dict[str, str], ...]:
    accessor = DictAccessor()
    pairs = []
    for exchange in iter_exchanges(messages, accessor):
        pairing = pair_results(messages, exchange, accessor, _PAIRING_POLICY)
        for assignments in (*pairing.truthy_assignments.values(), *pairing.falsy_assignments.values()):
            pairs.extend((call, result) for call, result in assignments if result is not None)
    pairs.sort(key=lambda pair: (pair[0].message_index, pair[0].content_index))

    steps: list[dict[str, str]] = []
    for call_occurrence, result_occurrence in pairs:
        if len(steps) >= MAX_REPOSITORY_STEPS:
            break
        call = accessor.contents(messages[call_occurrence.message_index])[call_occurrence.content_index]
        result = accessor.contents(messages[result_occurrence.message_index])[result_occurrence.content_index]
        if not isinstance(call, Mapping) or not isinstance(result, Mapping):
            continue
        name = call.get("name")
        tool_name = name if isinstance(name, str) and name else call_occurrence.content_type
        if tool_name in _MEMORY_TOOL_NAMES:
            continue
        arguments = _argument_summary(call)
        action = f"{tool_name} {arguments}".strip()
        observation = _result_summary(result)
        if action or observation:
            steps.append({"action": action, "observation": observation})
    return tuple(steps)


def _marker_turn_number(selected: Sequence[Any]) -> int | None:
    """Return the global turn number this span's own turn marker carries."""
    for message in reversed(list(selected)):
        properties = getattr(message, "additional_properties", None)
        if not isinstance(properties, Mapping):
            continue
        number = properties.get(TURN_INDEX_KEY)
        if isinstance(number, int) and not isinstance(number, bool):
            return number
    return None


def live_turn_numbers(live_messages: Sequence[Any], *, folded: bool) -> list[int]:
    """Return the global numbers of the finalized turns still in the live list.

    Identity comes from each turn's own marker, never from its position.
    Compaction folds completed turns out of ``state["messages"]`` and leaves an
    assistant summary that opens no turn, so the Nth slice stops being turn N
    the first time a session compacts -- and a positional watermark then marks
    turns as deposited that never were.

    A session with nothing folded is the one case where position is provably
    equal to identity, so an unmarked span there still counts. Once anything
    has been folded, an unmarked span is skipped: refusing to guess is what
    keeps a turn from being silently written off.
    """
    numbers: list[int] = []
    for index, (start, end) in enumerate(turn_slices(live_messages), start=1):
        number = _marker_turn_number(live_messages[start:end])
        if number is None and not folded:
            number = index
        if number is not None:
            numbers.append(number)
    return sorted(set(numbers))


def _turn_span(live_messages: Sequence[Any], *, folded: bool, turn: int) -> tuple[int, int] | None:
    """Resolve a global turn number to its span in the live message list."""
    slices = turn_slices(live_messages)
    for start, end in slices:
        if _marker_turn_number(live_messages[start:end]) == turn:
            return start, end
    if folded or not 1 <= turn <= len(slices):
        return None
    return slices[turn - 1]


def _folded(state: Mapping[str, Any]) -> bool:
    """Whether this session has ever folded a turn out of its live history."""
    blocks = state.get("compressed_msgs")
    return isinstance(blocks, list) and bool(blocks)


def extract_turn_experience(session_file: Path, turn: int) -> TurnExperience | None:
    """Extract one real turn using Chrys's canonical turn/exchange grammar.

    ``turn`` is the session's global turn number -- the one the turn marker and
    the artifact directories use -- not an index into the live message list.
    """
    if turn < 1 or session_file.is_symlink() or not session_file.is_file():
        return None
    if session_file.stat().st_size > MAX_SESSION_FILE_BYTES:
        raise ValueError(f"session file exceeds {MAX_SESSION_FILE_BYTES} bytes")
    envelope = json.loads(session_file.read_text(encoding="utf-8"))
    state = envelope.get("state") if isinstance(envelope, dict) else None
    if not isinstance(state, dict):
        return None
    deserialized = deserialize_state(state)
    live_messages = deserialized.get("messages", [])
    span = _turn_span(live_messages, folded=_folded(state), turn=turn)
    if span is None:
        return None
    start, end = span
    selected = live_messages[start:end]
    if not selected:
        return None
    serialized = [message.to_dict() for message in selected]
    steps = _paired_steps(serialized)
    if not steps:
        return None
    problem_statement = selected[0].text or ""
    final_response = next(
        (message.text or "" for message in reversed(selected) if message.role == "assistant" and message.text),
        "",
    )
    canonical = json.dumps(serialized, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)
    turn_digest = hashlib.sha256(canonical.encode()).hexdigest()
    route = _route_record(live_messages[start:end])
    campaign = route.get("campaign") if isinstance(route.get("campaign"), Mapping) else None
    campaign_status = str(campaign.get("status", "")) if campaign is not None else ""
    marker_success = turn_succeeded(serialized)
    return TurnExperience(
        problem_statement=_clarified_problem_statement(session_file, turn) or problem_statement,
        final_response=final_response,
        steps=steps,
        turn_digest=turn_digest,
        # A delegated campaign verified the work; nothing else here did.
        success=(campaign_status == "completed") if campaign_status else marker_success,
        route=str(route.get("track", "")),
        campaign_status=campaign_status,
    )


def _route_record(messages: Sequence[Any]) -> Mapping[str, Any]:
    """Return this turn's ``_chrys_route`` marker payload, or an empty mapping."""
    for message in reversed(list(messages)):
        properties = getattr(message, "additional_properties", None)
        if not isinstance(properties, Mapping):
            continue
        record = properties.get("_chrys_route")
        if isinstance(record, Mapping):
            return record
    return {}


def _clarified_problem_statement(session_file: Path, turn: int) -> str:
    """Prefer the clarified requirement over the raw prompt, when one exists.

    The clarified requirement is what the repair actually implemented, so it
    describes the problem the recorded steps solve.
    """
    path = (
        session_file.parent / "requirement_clarification" / f"turn_{turn}" / "05-outcome" / "clarified-requirement.md"
    )
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def turn_succeeded(serialized_messages: Sequence[Mapping[str, Any]]) -> bool:
    """Return whether a turn's persisted messages carry no failure marker."""
    for message in serialized_messages:
        properties = message.get("additional_properties")
        if not isinstance(properties, Mapping):
            continue
        if properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.INTERRUPTED:
            return False
        if properties.get(HistoryMarkerKind.STATUS_CODE_KEY) in _FAILED_STATUS_CODES:
            return False
    return True


def _session_file(session_id: str) -> Path:
    override = os.environ.get("CONTEXTGRAPH_SESSION_ROOT_DIR", "").strip()
    sessions_dir = Path(override).expanduser() / "sessions" if override else resolve_sessions_dir(create=False)
    return sessions_dir / session_id / "session.json"


def repo_label(cwd: str | None) -> str:
    """Name the repository a session worked in, the same way for every session of it.

    A PACT worktree, a clarification snapshot and the workspace itself are one
    repository to the graph; naming each by its own directory scattered one
    campaign's experience over "worker", "view" and the task, and recall by
    repository found none of it. Inside a git checkout the label is the main
    repository's directory (worktrees share it); elsewhere, the directory.
    """
    if not cwd:
        return "general"
    path = Path(cwd)
    git = shutil.which("git")
    if git is None:
        return path.name or "general"
    try:
        completed = subprocess.run(  # noqa: S603
            [git, "-C", cwd, "rev-parse", "--git-common-dir"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        completed = None
    if completed is not None and completed.returncode == 0:
        common = completed.stdout.strip()
        if common:
            common_dir = Path(common) if Path(common).is_absolute() else path / common
            name = common_dir.resolve().parent.name
            if name:
                return name
    return path.name or "general"


def deposit_hook_payload(payload: Mapping[str, Any]) -> RepositoryDepositResult | None:
    """Extract and deposit the completed turn named by an ``after_turn`` payload."""
    session_id = payload.get("session_id")
    turn = payload.get("turn")
    if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
        raise ValueError("after_turn payload has an invalid session_id")
    if not isinstance(turn, int) or isinstance(turn, bool):
        raise ValueError("after_turn payload has an invalid turn")
    extracted = extract_turn_experience(_session_file(session_id), turn)
    if extracted is None:
        return None
    cwd = payload.get("cwd")
    repo = repo_label(cwd if isinstance(cwd, str) else None)
    status = payload.get("status")
    source_id = f"chrys-after-turn:{session_id}:{turn}:{extracted.turn_digest}"
    return deposit_experience(
        problem_statement=extracted.problem_statement,
        success=(status == "ok") if isinstance(status, str) else extracted.success,
        steps=list(extracted.steps),
        final_response=extracted.final_response,
        repo=repo,
        source_id=source_id,
    )


def main() -> None:
    """Run as a durable ``after_turn`` hook worker."""
    if os.environ.get("CONTEXTGRAPH_DYNAMIC_DEPOSIT", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    payload_file = os.environ.get("CHRYS_HOOK_PAYLOAD_FILE", "").strip()
    if not payload_file:
        raise RuntimeError("CHRYS_HOOK_PAYLOAD_FILE is not set")
    payload = json.loads(Path(payload_file).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("hook payload must be a JSON object")
    deposit_hook_payload(payload)


if __name__ == "__main__":
    main()
