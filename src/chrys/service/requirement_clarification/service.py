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
    build_pact_goal_contract_prompt,
    build_pact_initial_plan_prompt,
    build_proposal_prompt,
    build_selector_prompt,
)
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot
from chrys.service.requirement_clarification.types import (
    ClarificationProposal,
    ClarificationResult,
    ClarificationSelection,
    PactGoalContract,
    PactInitialPlan,
    PactRuntimeInput,
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
        usage = [*(dict(item) for _proposal, item in proposal_results), dict(selector_usage)]
        return ClarificationResult(
            strategy_version=STRATEGY_VERSION,
            revision=revision.number,
            delta=delta,
            selection=cleaned,
            raw_selection=selection,
            proposals=tuple(proposals),
            elapsed_seconds=time.monotonic() - started,
            usage_details=tuple(usage),
        )

    async def generate_pact_input(
        self,
        *,
        result: ClarificationResult,
        revision: RequirementRevision,
        background: str,
        snapshot: WorkspaceSnapshot,
    ) -> tuple[PactRuntimeInput, tuple[dict[str, object], ...]]:
        """Generate and validate the optional PACT pair after ΔR is already safe."""
        goal_contract, goal_usage = await self._model.generate_pact_goal_contract(
            build_pact_goal_contract_prompt(revision.rendered, background)
        )
        base_evidence = collect_base_evidence(snapshot, revision.rendered)
        initial_plan, plan_usage = await self._model.generate_pact_initial_plan(
            build_pact_initial_plan_prompt(goal_contract, base_evidence, list(result.proposals), result.selection)
        )
        validate_pact_runtime_input(goal_contract, initial_plan)
        pact_input = PactRuntimeInput(goal_contract=goal_contract, initial_plan=initial_plan)
        return pact_input, (dict(goal_usage), dict(plan_usage))


def validate_pact_runtime_input(goal_contract: PactGoalContract, initial_plan: PactInitialPlan) -> None:
    """Validate invariants spanning the two closed PACT Runtime v1 payloads."""
    strings = [goal_contract.goal, *goal_contract.non_goals, *initial_plan.constraints]
    strings.extend(criterion.text for criterion in goal_contract.acceptance_criteria)
    for mission in initial_plan.missions:
        strings.extend((mission.objective, mission.verification_intent))
    if any(not value.strip() for value in strings):
        raise ValueError("PACT input contains a blank string")

    ac_ids = [criterion.id for criterion in goal_contract.acceptance_criteria]
    if len(ac_ids) != len(set(ac_ids)):
        raise ValueError("PACT Goal Contract acceptance criterion ids must be unique")
    mission_ids = [mission.id for mission in initial_plan.missions]
    if len(mission_ids) != len(set(mission_ids)):
        raise ValueError("PACT Initial Plan mission ids must be unique")

    known_ac_ids = set(ac_ids)
    known_mission_ids = set(mission_ids)
    covered_ac_ids: set[str] = set()
    dependency_counts: dict[str, int] = {}
    successors = {mission_id: [] for mission_id in mission_ids}
    for mission in initial_plan.missions:
        if len(mission.target_ac_ids) != len(set(mission.target_ac_ids)):
            raise ValueError(f"PACT mission {mission.id!r} contains duplicate target_ac_ids")
        unknown_ac_ids = set(mission.target_ac_ids) - known_ac_ids
        if unknown_ac_ids:
            raise ValueError(f"PACT mission {mission.id!r} references unknown acceptance criteria")
        covered_ac_ids.update(mission.target_ac_ids)
        if len(mission.dependencies) != len(set(mission.dependencies)):
            raise ValueError(f"PACT mission {mission.id!r} contains duplicate dependencies")
        if mission.id in mission.dependencies:
            raise ValueError(f"PACT mission {mission.id!r} depends on itself")
        unknown_dependencies = set(mission.dependencies) - known_mission_ids
        if unknown_dependencies:
            raise ValueError(f"PACT mission {mission.id!r} references unknown dependencies")
        dependency_counts[mission.id] = len(mission.dependencies)
        for dependency in mission.dependencies:
            successors[dependency].append(mission.id)
    if covered_ac_ids != known_ac_ids:
        raise ValueError("PACT Initial Plan does not cover every acceptance criterion")

    frontier = [mission_id for mission_id, count in dependency_counts.items() if count == 0]
    visited = 0
    while frontier:
        mission_id = frontier.pop()
        visited += 1
        for successor in successors[mission_id]:
            dependency_counts[successor] -= 1
            if dependency_counts[successor] == 0:
                frontier.append(successor)
    if visited != len(mission_ids):
        raise ValueError("PACT Initial Plan dependency graph contains a cycle")


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
