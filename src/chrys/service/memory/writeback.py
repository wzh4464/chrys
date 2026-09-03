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

from chrys.foundation.models.turns import turn_slices
from chrys.service.memory.contextgraph_deposit import deposit_experience, extract_turn_experience
from chrys.service.state.serializers import deserialize_state

logger = logging.getLogger(__name__)

WATERMARK_KEY = "memory_deposit_watermark"


@dataclass(frozen=True, slots=True)
class WritebackOutcome:
    """What one writeback pass accomplished."""

    deposited: tuple[int, ...]
    failed: int | None
    watermark: int


def count_turns(session_file: Path) -> int:
    """Return how many complete turns *session_file* holds, or ``0`` if unreadable."""
    try:
        envelope = json.loads(session_file.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return 0
    state = envelope.get("state") if isinstance(envelope, dict) else None
    if not isinstance(state, dict):
        return 0
    try:
        messages = deserialize_state(state).get("messages", [])
    except TypeError, ValueError, KeyError:
        return 0
    return len(turn_slices(messages))


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
    """
    total = count_turns(session_file)
    deposited: list[int] = []
    for turn in range(max(watermark, 0) + 1, total + 1):
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
            return WritebackOutcome(deposited=tuple(deposited), failed=turn, watermark=turn - 1)
    return WritebackOutcome(deposited=tuple(deposited), failed=None, watermark=total)
