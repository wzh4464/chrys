# Copyright (c) 2026 Chrys. All rights reserved.

"""Three-proposal/one-selector requirement clarification service."""

from __future__ import annotations

import asyncio
import re
import time

from chrys.service.requirement_clarification.evidence import collect_base_evidence
from chrys.service.requirement_clarification.model import ClarificationModelRunner
from chrys.service.requirement_clarification.prompts import (
    MAX_FINAL_CHARS,
    MAX_FINAL_POINTS,
    PROPOSAL_COUNT,
    STRATEGY_VERSION,
    build_proposal_prompt,
    build_selector_prompt,
)
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot
from chrys.service.requirement_clarification.types import (
    ClarificationProposal,
    ClarificationResult,
    ClarificationSelection,
    RequirementRevision,
    SelectedGuidancePoint,
)

_RENDER_CONFIDENCE_THRESHOLD = 0.75
_PROCESS_MARKERS = ("confidence:", "source:", "evidence:", "gate:")


class ClarificationService:
    """Generate a small validated ΔR without observing the baseline patch."""

    def __init__(self, model: ClarificationModelRunner) -> None:
        self._model = model

    async def clarify(
        self,
        *,
        revision: RequirementRevision,
        background: str,
        snapshot: WorkspaceSnapshot,
    ) -> ClarificationResult:
        started = time.monotonic()
        # Evidence collection is deliberately bounded (both search volume and
        # subprocess timeouts).  Keeping it in this task also avoids retaining
        # a process-wide default executor beyond the clarification lifecycle.
        base_evidence = collect_base_evidence(snapshot, revision.rendered)
        proposal_calls = [
            self._model.propose(
                build_proposal_prompt(revision.rendered, background, base_evidence, sample_index),
                sample_index=sample_index,
            )
            for sample_index in range(1, PROPOSAL_COUNT + 1)
        ]
        proposal_results = await asyncio.gather(*proposal_calls)
        proposals = [proposal for proposal, _usage in proposal_results]
        if len(proposals) != PROPOSAL_COUNT or not all(isinstance(item, ClarificationProposal) for item in proposals):
            raise ValueError(f"expected {PROPOSAL_COUNT} valid clarification proposals")
        selection_prompt = build_selector_prompt(revision.rendered, background, base_evidence, proposals)
        selection, selector_usage = await self._model.select(selection_prompt)
        cleaned = sanitize_selection(revision.rendered, selection)
        delta = render_delta(cleaned)
        usage = (*(dict(item) for _proposal, item in proposal_results), dict(selector_usage))
        return ClarificationResult(
            strategy_version=STRATEGY_VERSION,
            revision=revision.number,
            delta=delta,
            selection=cleaned,
            proposals=tuple(proposals),
            elapsed_seconds=time.monotonic() - started,
            usage_details=usage,
        )


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.casefold())


def sanitize_selection(requirement: str, selection: ClarificationSelection) -> ClarificationSelection:
    """Apply deterministic, generic cleanup without adding model judgments."""
    requirement_text = " ".join(_normalized_words(requirement))
    rows: list[SelectedGuidancePoint] = []
    seen: set[str] = set()
    for row in selection.guidance_points:
        statement = " ".join(row.statement.split())
        normalized = " ".join(_normalized_words(statement))
        lowered = statement.casefold()
        if (
            not normalized
            or normalized in seen
            or normalized in requirement_text
            or "<" in statement
            or ">" in statement
            or any(marker in lowered for marker in _PROCESS_MARKERS)
        ):
            continue
        seen.add(normalized)
        rows.append(row.model_copy(update={"statement": statement}))
    rows.sort(key=lambda item: (-item.confidence, item.category, item.statement))
    return ClarificationSelection(guidance_points=rows[:MAX_FINAL_POINTS])


def render_delta(selection: ClarificationSelection) -> str:
    """Render only high-confidence selected statements within the v1 budget."""
    selected: list[str] = []
    used = 0
    for row in selection.guidance_points:
        if row.confidence < _RENDER_CONFIDENCE_THRESHOLD:
            continue
        statement = " ".join(row.statement.split())
        line_size = len(statement) + 3
        if len(selected) >= MAX_FINAL_POINTS or used + line_size > MAX_FINAL_CHARS:
            continue
        selected.append(statement)
        used += line_size
    if not selected:
        return ""
    return "Repository implementation guidance:\n" + "\n".join(f"- {statement}" for statement in selected)
