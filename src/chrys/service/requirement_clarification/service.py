# Copyright (c) 2026 Chrys. All rights reserved.

"""Three-proposal/one-selector requirement clarification service."""

from __future__ import annotations

import asyncio
import re
import time

from pydantic import ValidationError

from chrys.service.requirement_clarification.evidence import collect_base_evidence
from chrys.service.requirement_clarification.model import ClarificationModelRunner
from chrys.service.requirement_clarification.prompts import (
    LEGACY_V1_STRATEGY_VERSION,
    MAX_FINAL_CHARS,
    MAX_FINAL_POINTS,
    MIN_VALID_PROPOSALS,
    PROPOSAL_COUNT,
    STRATEGY_VERSION,
    build_legacy_v1_proposal_prompt,
    build_legacy_v1_selector_prompt,
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
    ClarificationSelectorDecision,
    ClarificationStrategy,
    LegacyV1ClarificationProposal,
    PactGoalContract,
    PactInitialPlan,
    PactMission,
    PactRuntimeInput,
    ProposalGuidancePoint,
    ProposalInvestigation,
    ProposalModelResult,
    RequirementRevision,
    SelectedGuidancePoint,
    SelectorCandidateReview,
)

_RENDER_CONFIDENCE_THRESHOLD = 0.75
_PROCESS_MARKERS = ("confidence:", "source:", "evidence:", "gate:")
_SELECTOR_REVIEW_BATCH_SIZE = 6
_NOT_REVIEWED = "candidate ids were not reviewed: "


class ClarificationService:
    """Generate a small validated ΔR without observing the baseline patch."""

    def __init__(
        self,
        model: ClarificationModelRunner,
        *,
        strategy: ClarificationStrategy = "legacy-v1-stabilized",
    ) -> None:
        self._model = model
        self._strategy = strategy

    async def clarify(
        self,
        *,
        revision: RequirementRevision,
        background: str,
        snapshot: WorkspaceSnapshot,
    ) -> ClarificationResult:
        if self._strategy == "legacy-v1-exact":
            return await self._clarify_legacy_v1(
                revision=revision,
                background=background,
                snapshot=snapshot,
            )
        started = time.monotonic()
        base_evidence = await asyncio.to_thread(collect_base_evidence, snapshot, revision.rendered)
        proposal_calls = [
            self._model.propose(
                build_proposal_prompt(revision.rendered, background, base_evidence, sample_index),
                sample_index=sample_index,
            )
            for sample_index in range(1, PROPOSAL_COUNT + 1)
        ]
        raw_proposal_results = await asyncio.gather(*proposal_calls, return_exceptions=True)
        proposal_results: list[ClarificationProposal] = []
        proposal_indices: list[int] = []
        investigations: list[ProposalInvestigation] = []
        usage: list[dict[str, object]] = []
        warnings: list[str] = []
        for sample_index, item in enumerate(raw_proposal_results, start=1):
            if isinstance(item, BaseException):
                detail = f"{type(item).__name__}: {item}"[:600]
                warnings.append(f"clarification proposer {sample_index} failed: {detail}")
                investigations.append(
                    ProposalInvestigation(
                        sample_index=sample_index,
                        status="failed",
                        investigation_attempts=1,
                        synthesis_attempts=0,
                        validation_errors=[detail],
                    )
                )
                continue
            if not isinstance(item, ProposalModelResult):
                detail = "returned an invalid result type"
                warnings.append(f"clarification proposer {sample_index} {detail}")
                investigations.append(
                    ProposalInvestigation(
                        sample_index=sample_index,
                        status="failed",
                        investigation_attempts=1,
                        synthesis_attempts=0,
                        validation_errors=[detail],
                    )
                )
                continue
            investigations.append(item.investigation)
            usage.extend(dict(details) for details in item.usage_details)
            if item.proposal is None:
                detail = item.error or "investigation or synthesis did not produce a valid proposal"
                warnings.append(f"clarification proposer {sample_index} failed: {detail}"[:700])
                continue
            proposal_results.append(item.proposal)
            proposal_indices.append(sample_index)
        proposals = proposal_results
        if len(proposals) < MIN_VALID_PROPOSALS:
            return ClarificationResult(
                strategy_version=STRATEGY_VERSION,
                revision=revision.number,
                delta="",
                selection=ClarificationSelection(),
                status="degraded",
                empty_reason="insufficient_valid_proposals",
                proposals=tuple(proposals),
                proposal_sample_indices=tuple(proposal_indices),
                investigations=tuple(investigations),
                elapsed_seconds=time.monotonic() - started,
                usage_details=tuple(usage),
                warnings=tuple(warnings),
            )
        candidate_map = _candidate_map(proposals)
        if not candidate_map:
            return ClarificationResult(
                strategy_version=STRATEGY_VERSION,
                revision=revision.number,
                delta="",
                selection=ClarificationSelection(),
                status="completed",
                empty_reason="requirement_complete",
                proposals=tuple(proposals),
                proposal_sample_indices=tuple(proposal_indices),
                investigations=tuple(investigations),
                elapsed_seconds=time.monotonic() - started,
                usage_details=tuple(usage),
                warnings=tuple(warnings),
            )
        raw_selection: ClarificationSelectorDecision | None = None
        selector_errors: list[str] = []
        reviews: list[SelectorCandidateReview] = []
        candidate_ids = tuple(candidate_map)
        for offset in range(0, len(candidate_ids), _SELECTOR_REVIEW_BATCH_SIZE):
            batch_ids = candidate_ids[offset : offset + _SELECTOR_REVIEW_BATCH_SIZE]
            batch_candidates = {candidate_id: candidate_map[candidate_id] for candidate_id in batch_ids}
            selection_prompt = build_selector_prompt(
                revision.rendered,
                background,
                base_evidence,
                proposals,
                candidate_ids=set(batch_ids),
            )
            decision: ClarificationSelectorDecision | None = None
            for selector_attempt in (1, 2):
                prompt = selection_prompt
                if selector_attempt == 2:
                    prompt += (
                        "\n\nThe prior selection needs a second-pass audit: "
                        + "; ".join(selector_errors)
                        + ". Return exactly one review per unique id from this closed candidate packet."
                    )
                try:
                    decision, selector_usage = await self._model.select(prompt)
                    usage.append(dict(selector_usage))
                    selector_errors = _selector_errors(decision, batch_candidates)
                    if (
                        not selector_errors
                        and len(candidate_ids) <= _SELECTOR_REVIEW_BATCH_SIZE
                        and not any(review.decision == "select" for review in decision.reviews)
                        and selector_attempt == 1
                    ):
                        selector_errors = ["the first-pass selector rejected every candidate after review"]
                        continue
                    if not selector_errors:
                        break
                except Exception as exc:
                    selector_errors = [f"{type(exc).__name__}: {exc}"[:600]]
            if decision is not None and selector_errors and all(e.startswith(_NOT_REVIEWED) for e in selector_errors):
                # Two passes and the selector still skipped some ids. Its verdicts on
                # the rest are sound; an unreviewed candidate is simply not selected.
                # Failing the whole selection here promoted P0 on a benchmark task
                # whose selector had reviewed twenty-four of twenty-nine candidates.
                reviewed = {review.candidate_id for review in decision.reviews}
                missing = [candidate_id for candidate_id in batch_ids if candidate_id not in reviewed]
                warnings.append(
                    "clarification selector left candidates unreviewed after two passes; treated as rejected: "
                    + ", ".join(missing)
                )
                decision = ClarificationSelectorDecision(
                    reviews=[
                        *decision.reviews,
                        *(
                            SelectorCandidateReview(
                                candidate_id=candidate_id,
                                decision="reject",
                                rationale="Not reviewed by the selector in two passes; not selected.",
                            )
                            for candidate_id in missing
                        ),
                    ]
                )
                selector_errors = []
            if decision is None or selector_errors:
                break
            reviews.extend(decision.reviews)
        if not selector_errors and len(reviews) == len(candidate_map):
            raw_selection = (
                decision
                if len(candidate_ids) <= _SELECTOR_REVIEW_BATCH_SIZE and decision is not None
                else ClarificationSelectorDecision(reviews=reviews)
            )
        else:
            raw_selection = None
        if raw_selection is None or selector_errors:
            warnings.append("clarification selector failed: " + "; ".join(selector_errors))
            return ClarificationResult(
                strategy_version=STRATEGY_VERSION,
                revision=revision.number,
                delta="",
                selection=ClarificationSelection(),
                raw_selection=raw_selection,
                status="degraded",
                empty_reason="selector_failed",
                proposals=tuple(proposals),
                proposal_sample_indices=tuple(proposal_indices),
                investigations=tuple(investigations),
                elapsed_seconds=time.monotonic() - started,
                usage_details=tuple(usage),
                warnings=tuple(warnings),
            )
        selection = _materialize_selection(raw_selection, candidate_map)
        cleaned = sanitize_selection(revision.rendered, selection)
        delta = render_delta(cleaned)
        return ClarificationResult(
            strategy_version=STRATEGY_VERSION,
            revision=revision.number,
            delta=delta,
            selection=cleaned,
            raw_selection=raw_selection,
            status="completed",
            empty_reason=None if delta else "selector_rejected",
            proposals=tuple(proposals),
            proposal_sample_indices=tuple(proposal_indices),
            investigations=tuple(investigations),
            elapsed_seconds=time.monotonic() - started,
            usage_details=tuple(usage),
            warnings=tuple(warnings),
        )

    async def _clarify_legacy_v1(
        self,
        *,
        revision: RequirementRevision,
        background: str,
        snapshot: WorkspaceSnapshot,
    ) -> ClarificationResult:
        """Execute the source-equivalent proposal/selector path from the historical +1 run."""
        started = time.monotonic()
        base_evidence = await asyncio.to_thread(collect_base_evidence, snapshot, revision.rendered)
        proposal_calls = [
            self._model.propose_legacy_v1(
                build_legacy_v1_proposal_prompt(revision.rendered, background, base_evidence, sample_index),
                sample_index=sample_index,
            )
            for sample_index in range(1, PROPOSAL_COUNT + 1)
        ]
        proposal_results = await asyncio.gather(*proposal_calls)
        proposals = [proposal for proposal, _usage in proposal_results]
        if len(proposals) != PROPOSAL_COUNT or not all(
            isinstance(item, LegacyV1ClarificationProposal) for item in proposals
        ):
            raise ValueError(f"expected {PROPOSAL_COUNT} valid historical v1 clarification proposals")
        selection, selector_usage = await self._model.select_legacy_v1(
            build_legacy_v1_selector_prompt(
                revision.rendered,
                background,
                base_evidence,
                proposals,
            )
        )
        cleaned = sanitize_selection(revision.rendered, selection)
        delta = render_delta(cleaned)
        usage = (*(dict(item) for _proposal, item in proposal_results), dict(selector_usage))
        candidate_count = sum(len(proposal.guidance_points) for proposal in proposals)
        return ClarificationResult(
            strategy_version=LEGACY_V1_STRATEGY_VERSION,
            revision=revision.number,
            delta=delta,
            selection=cleaned,
            raw_selection=selection,
            status="completed",
            empty_reason=None if delta else ("selector_rejected" if candidate_count else "requirement_complete"),
            proposals=tuple(proposals),
            proposal_sample_indices=tuple(range(1, len(proposals) + 1)),
            elapsed_seconds=time.monotonic() - started,
            usage_details=usage,
        )

    async def generate_pact_input(
        self,
        *,
        result: ClarificationResult,
        revision: RequirementRevision,
        background: str,
        snapshot: WorkspaceSnapshot,
        localization_hints: str = "",
    ) -> tuple[PactRuntimeInput, tuple[dict[str, object], ...]]:
        """Generate and validate the optional PACT pair after ΔR is already safe.

        *localization_hints* is untrusted evidence -- a code search's ranked
        guesses -- and reaches the Initial Plan prompt only. The Goal Contract
        stays derived from user authority alone: a search result must never
        be able to widen what the campaign is allowed to do.
        """
        goal_contract, goal_usage = await self._model.generate_pact_goal_contract(
            build_pact_goal_contract_prompt(revision.rendered, background)
        )
        base_evidence = await asyncio.to_thread(collect_base_evidence, snapshot, revision.rendered)
        plan_prompt = build_pact_initial_plan_prompt(
            goal_contract,
            base_evidence,
            list(result.proposals),
            result.selection,
        )
        if localization_hints.strip():
            plan_prompt += (
                "\n\nCandidate code locations from a repository search (untrusted; verify before relying on them):\n"
                + localization_hints.strip()
            )
        usage = [dict(goal_usage)]
        validation_error = ""
        for plan_attempt in (1, 2):
            prompt = plan_prompt
            if plan_attempt == 2:
                prompt += (
                    "\n\nThe prior Initial Plan failed deterministic validation: "
                    + validation_error
                    + ". Regenerate the complete Initial Plan. Cover every Goal Contract acceptance criterion, "
                    "reference only declared criterion and mission ids, and keep mission dependencies acyclic."
                )
            initial_plan: PactInitialPlan | None = None
            try:
                initial_plan, plan_usage = await self._model.generate_pact_initial_plan(prompt)
                usage.append(dict(plan_usage))
                initial_plan = _sanitize_pact_target_ac_ids(goal_contract, initial_plan)
                validate_pact_runtime_input(goal_contract, initial_plan)
            except (ValidationError, ValueError) as exc:
                validation_error = str(exc)
                if plan_attempt == 1:
                    continue
                if initial_plan is None:
                    raise
                initial_plan = _complete_missing_pact_coverage(goal_contract, initial_plan)
                validate_pact_runtime_input(goal_contract, initial_plan)
            pact_input = PactRuntimeInput(goal_contract=goal_contract, initial_plan=initial_plan)
            return pact_input, tuple(usage)
        raise RuntimeError("PACT Initial Plan generation exhausted without a result")


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
        missing = ", ".join(sorted(known_ac_ids - covered_ac_ids))
        raise ValueError(f"PACT Initial Plan does not cover acceptance criteria: {missing}")

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


def _candidate_map(proposals: list[ClarificationProposal]) -> dict[str, ProposalGuidancePoint]:
    candidates: dict[str, ProposalGuidancePoint] = {}
    for proposal_index, proposal in enumerate(proposals, start=1):
        for guidance_index, point in enumerate(proposal.guidance_points, start=1):
            candidates[f"p{proposal_index}-g{guidance_index}"] = point
    return candidates


def _selector_errors(
    decision: ClarificationSelectorDecision,
    candidates: dict[str, ProposalGuidancePoint],
) -> list[str]:
    ids = [review.candidate_id for review in decision.reviews]
    errors: list[str] = []
    unknown = sorted(set(ids) - set(candidates))
    if unknown:
        errors.append("unknown candidate ids: " + ", ".join(unknown))
    if len(ids) != len(set(ids)):
        errors.append("duplicate candidate ids are not allowed")
    missing = sorted(set(candidates) - set(ids))
    if missing:
        errors.append(_NOT_REVIEWED + ", ".join(missing))
    return errors


def _materialize_selection(
    decision: ClarificationSelectorDecision,
    candidates: dict[str, ProposalGuidancePoint],
) -> ClarificationSelection:
    rows: list[SelectedGuidancePoint] = []
    for review in decision.reviews:
        if review.decision != "select":
            continue
        candidate = candidates[review.candidate_id]
        rows.append(
            SelectedGuidancePoint(
                category=candidate.category,
                statement=candidate.statement,
                confidence=candidate.confidence,
                basis=candidate.basis,
            )
        )
    rows.sort(key=lambda item: (-item.confidence, item.category, item.statement))
    return ClarificationSelection(guidance_points=rows[:MAX_FINAL_POINTS])


def _complete_missing_pact_coverage(
    goal_contract: PactGoalContract,
    initial_plan: PactInitialPlan,
) -> PactInitialPlan:
    """Add authority-preserving missions only for acceptance criteria omitted by the model.

    Unknown target acceptance-criterion ids are removed before this helper runs. Other invalid
    plans, including duplicate ids and cycles, still fail closed. The generated mission repeats
    the Goal Contract criterion verbatim and therefore cannot introduce a new completion
    obligation.
    """
    covered = {ac_id for mission in initial_plan.missions for ac_id in mission.target_ac_ids}
    existing_ids = {mission.id for mission in initial_plan.missions}
    additions: list[PactMission] = []
    for index, criterion in enumerate(goal_contract.acceptance_criteria, start=1):
        if criterion.id in covered:
            continue
        base_id = f"cover-missing-ac-{index}"
        mission_id = base_id
        suffix = 2
        while mission_id in existing_ids:
            mission_id = f"{base_id}-{suffix}"
            suffix += 1
        existing_ids.add(mission_id)
        additions.append(
            PactMission(
                id=mission_id,
                objective=criterion.text,
                target_ac_ids=[criterion.id],
                dependencies=[],
                verification_intent=(
                    f"Collect observable evidence for Goal Contract acceptance criterion {criterion.id}: "
                    f"{criterion.text}"
                ),
            )
        )
    return initial_plan.model_copy(update={"missions": [*initial_plan.missions, *additions]})


def _sanitize_pact_target_ac_ids(
    goal_contract: PactGoalContract,
    initial_plan: PactInitialPlan,
) -> PactInitialPlan:
    """Remove model-invented acceptance-criterion references from a PACT plan."""
    known_ac_ids = {criterion.id for criterion in goal_contract.acceptance_criteria}
    missions = []
    for mission in initial_plan.missions:
        target_ac_ids = [ac_id for ac_id in mission.target_ac_ids if ac_id in known_ac_ids]
        if target_ac_ids:
            missions.append(mission.model_copy(update={"target_ac_ids": target_ac_ids}))
    return PactInitialPlan.model_validate(
        {
            **initial_plan.model_dump(),
            "missions": [mission.model_dump() for mission in missions],
        }
    )


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
