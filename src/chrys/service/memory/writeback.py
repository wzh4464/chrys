# Copyright (c) 2026 Chrys. All rights reserved.

"""Watermark-driven deposition of completed turns into ContextGraph.

The watermark is a high-water mark stored in the session's runtime metadata, so
depositing is resumable and replay-safe: a crashed or offline write leaves the
mark where it was and the next pass retries from there. Stable ``source_id``
values make a retry that already landed a no-op on the ContextGraph side.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chrys.service.memory.contextgraph_deposit import (
    deposit_experience,
    extract_turn_experience,
    live_turn_numbers,
)
from chrys.service.session.runtime_metadata import MEMORY_DEPOSIT_WATERMARK_KEY
from chrys.service.state.serializers import deserialize_state

logger = logging.getLogger(__name__)

# Re-exported so a caller reading or writing the mark in persisted state does
# not have to know it is declared alongside the other session-state keys.
WATERMARK_KEY = MEMORY_DEPOSIT_WATERMARK_KEY


@dataclass(frozen=True, slots=True)
class WritebackOutcome:
    """What one writeback pass accomplished."""

    deposited: tuple[int, ...]
    failed: int | None
    watermark: int


def session_turn_numbers(session_file: Path) -> list[int]:
    """Return the global numbers of the turns *session_file* still holds live.

    Global, not positional: compaction folds completed turns out of the live
    list, so counting slices would make the watermark mean a different turn
    after every fold.
    """
    try:
        envelope = json.loads(session_file.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return []
    state = envelope.get("state") if isinstance(envelope, dict) else None
    if not isinstance(state, dict):
        return []
    blocks = state.get("compressed_msgs")
    try:
        messages = deserialize_state(state).get("messages", [])
    except TypeError, ValueError, KeyError:
        return []
    return live_turn_numbers(messages, folded=isinstance(blocks, list) and bool(blocks))


def pending_turns(session_file: Path, watermark: int) -> list[int]:
    """Return the turns after *watermark* that are still available to deposit."""
    return [turn for turn in session_turn_numbers(session_file) if turn > watermark]


def deposit_pending_turns(
    session_file: Path,
    *,
    watermark: int,
    repo: str,
    source_prefix: str,
    deposit: Callable[..., Any] = deposit_experience,
) -> WritebackOutcome:
    """Deposit every turn after *watermark*, stopping at the first failure.

    Stopping matters: the returned watermark is the last turn *known* to have
    been handled, so skipping past a failure would silently drop that turn from
    the graph forever. A turn with no tool-backed work yields nothing to deposit
    but still advances the mark — there is nothing to retry.

    The mark only ever moves forward. Compaction removes completed turns from
    the live list, and a mark that followed it down would re-point at turns
    that were never deposited and call them done.
    """
    mark = max(watermark, 0)
    deposited: list[int] = []
    for turn in pending_turns(session_file, mark):
        try:
            extracted = extract_turn_experience(session_file, turn)
            if extracted is not None:
                deposit(
                    problem_statement=extracted.problem_statement,
                    success=extracted.success,
                    steps=list(extracted.steps),
                    final_response=extracted.final_response,
                    repo=repo,
                    source_id=f"{source_prefix}:{turn}:{extracted.turn_digest}",
                )
                deposited.append(turn)
        except Exception:
            logger.warning("ContextGraph deposit failed for turn %d", turn, exc_info=True)
            return WritebackOutcome(deposited=tuple(deposited), failed=turn, watermark=mark)
        mark = turn
    return WritebackOutcome(deposited=tuple(deposited), failed=None, watermark=mark)
