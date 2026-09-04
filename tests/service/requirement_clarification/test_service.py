# Copyright (c) 2026 Chrys. All rights reserved.

"""Three-proposal/one-selector clarification behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from chrys.service.requirement_clarification.service import (
    ClarificationService,
    _materialize_selection,
    render_delta,
    validate_pact_runtime_input,
)
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot
from chrys.service.requirement_clarification.types import (
    ClarificationProposal,
    ClarificationSelection,
    ClarificationSelectorDecision,
    EvidenceAnchor,
    InvestigationCoverageFinding,
    LegacyV1ClarificationProposal,
    PactAcceptanceCriterion,
    PactGoalContract,
    PactInitialPlan,
    PactMission,
    ProposalGuidancePoint,
    ProposalInvestigation,
    ProposalModelResult,
    RequirementRevision,
    SelectedGuidancePoint,
    SelectorCandidateReview,
)


def _coverage() -> list[InvestigationCoverageFinding]:
    return [
        InvestigationCoverageFinding(
            target="primary",
            status="found",
            summary="The runtime owner was inspected.",
            evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
        ),
        InvestigationCoverageFinding(
            target="adjacent",
            status="found",
            summary="The adjacent consumer was inspected.",
            evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:20")],
        ),
    ]


def _proposal(sample_index: int) -> ClarificationProposal:
    return ClarificationProposal(
        verdict="clarification_needed",
        rationale="The repository exposes an integration seam not fully described by the requirement.",
        coverage=_coverage(),
        evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
        guidance_points=[
            ProposalGuidancePoint(
                category="integration",
                statement=f"Candidate {sample_index} connects the option to its runtime consumer.",
                confidence=0.8,
                basis="current_repo",
                contract_cell="transport_integration",
                decision_impact="The option otherwise stops before reaching the consumer.",
                evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
                risk="The selector must confirm that this is the authoritative consumer.",
            )
        ],
    )


def _decision(
    *,
    candidate_ids: tuple[str, ...] = ("p1-g1", "p2-g1", "p3-g1"),
    selected_ids: tuple[str, ...] = (),
) -> ClarificationSelectorDecision:
    return ClarificationSelectorDecision(
        reviews=[
            SelectorCandidateReview(
                candidate_id=candidate_id,
                decision="select" if candidate_id in selected_ids else "reject",
                rationale=(
                    "This is a necessary repository mapping."
                    if candidate_id in selected_ids
                    else "This does not add a necessary repository mapping."
                ),
            )
            for candidate_id in candidate_ids
        ]
    )


class _FakeModel:
    def __init__(self, selection: ClarificationSelectorDecision) -> None:
        self.selection = selection
        self.proposal_calls: list[int] = []
        self.selector_calls = 0

    async def propose(self, _prompt: str, *, sample_index: int):
        self.proposal_calls.append(sample_index)
        return ProposalModelResult(
            proposal=_proposal(sample_index),
            investigation=ProposalInvestigation(
                sample_index=sample_index,
                status="completed",
                investigation_attempts=1,
                synthesis_attempts=1,
            ),
            usage_details=({"input_tokens": sample_index},),
        )

    async def select(self, _prompt: str):
        self.selector_calls += 1
        return self.selection, {"output_tokens": 5}

    async def generate_pact_goal_contract(self, _prompt: str):
        return (
            PactGoalContract(
                schema="pact-runtime/goal-contract/v1",
                goal="Add the option end to end.",
                acceptance_criteria=[PactAcceptanceCriterion(id="ac-option", text="The option takes effect.")],
                non_goals=[],
            ),
            {"output_tokens": 6},
        )

    async def generate_pact_initial_plan(self, _prompt: str):
        return (
            PactInitialPlan(
                schema="pact-runtime/initial-plan/v1",
                constraints=[],
                missions=[
                    PactMission(
                        id="wire-option",
                        objective="Wire the option to its runtime consumer.",
                        target_ac_ids=["ac-option"],
                        dependencies=[],
                        verification_intent="Exercise the public option through the runtime path.",
                    )
                ],
            ),
            {"output_tokens": 7},
        )


def _snapshot(tmp_path: Path) -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        snapshot_id="s0",
        artifact_root=str(tmp_path),
        roots=(),
        manifest_hash="abc",
        total_bytes=0,
        entry_count=0,
    )


def test_proposal_schema_allows_only_evidenced_empty_candidates() -> None:
    with pytest.raises(ValidationError):
        ClarificationProposal.model_validate({})
    complete = ClarificationProposal(
        verdict="requirement_complete",
        rationale="The requirement and repository contract already cover the requested behavior.",
        coverage=_coverage(),
        evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
        guidance_points=[],
    )

    assert complete.guidance_points == []

    with pytest.raises(ValidationError, match="clarification_needed requires"):
        ClarificationProposal(
            verdict="clarification_needed",
            rationale="The repository was inspected and no candidate has been supplied.",
            coverage=_coverage(),
            evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
            guidance_points=[],
        )

    with pytest.raises(ValidationError):
        ClarificationProposal(
            verdict="requirement_complete",
            rationale="   ",
            coverage=_coverage(),
            evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
            guidance_points=[],
        )

    with pytest.raises(ValidationError):
        ProposalGuidancePoint(
            category="integration",
            statement=" \n ",
            confidence=0.9,
            basis="current_repo",
            contract_cell="transport_integration",
            decision_impact="The consumer would otherwise not receive the value.",
            evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
            risk="Confirm the consumer is authoritative.",
        )


@pytest.mark.asyncio
async def test_service_runs_three_proposals_then_one_selector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection = _decision(selected_ids=("p1-g1",))
    model = _FakeModel(selection)
    monkeypatch.setattr(
        "chrys.service.requirement_clarification.service.collect_base_evidence",
        lambda _snapshot, _requirement: "packet",
    )

    service = ClarificationService(model)
    revision = RequirementRevision(number=1, messages=("Add the option.",))
    snapshot = _snapshot(tmp_path)
    result = await service.clarify(
        revision=revision,
        background="Earlier context",
        snapshot=snapshot,
    )
    pact_input, pact_usage = await service.generate_pact_input(
        result=result,
        revision=revision,
        background="Earlier context",
        snapshot=snapshot,
    )

    assert sorted(model.proposal_calls) == [1, 2, 3]
    assert model.selector_calls == 1
    assert result.delta == (
        "Repository implementation guidance:\n- Candidate 1 connects the option to its runtime consumer."
    )
    assert result.raw_selection is selection
    assert pact_input.goal_contract.acceptance_criteria[0].id == "ac-option"
    assert len(result.usage_details) == 4
    assert len(pact_usage) == 2


@pytest.mark.asyncio
async def test_pact_initial_plan_retries_after_cross_payload_validation_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeModel(_decision(selected_ids=("p1-g1",)))
    original_generate_plan = model.generate_pact_initial_plan
    plan_prompts: list[str] = []

    async def generate_plan_after_feedback(prompt: str):
        plan_prompts.append(prompt)
        if len(plan_prompts) == 1:
            return (
                PactInitialPlan(
                    schema="pact-runtime/initial-plan/v1",
                    constraints=[],
                    missions=[
                        PactMission(
                            id="wrong-plan",
                            objective="Cover an unknown criterion.",
                            target_ac_ids=["ac-unknown"],
                            dependencies=[],
                            verification_intent="Verify the unknown criterion.",
                        )
                    ],
                ),
                {"output_tokens": 1},
            )
        return await original_generate_plan(prompt)

    model.generate_pact_initial_plan = generate_plan_after_feedback  # type: ignore[method-assign]
    monkeypatch.setattr(
        "chrys.service.requirement_clarification.service.collect_base_evidence",
        lambda _snapshot, _requirement: "packet",
    )
    service = ClarificationService(model)
    revision = RequirementRevision(number=1, messages=("Add the option.",))
    snapshot = _snapshot(tmp_path)
    result = await service.clarify(revision=revision, background="", snapshot=snapshot)

    pact_input, pact_usage = await service.generate_pact_input(
        result=result,
        revision=revision,
        background="",
        snapshot=snapshot,
    )

    assert len(plan_prompts) == 2
    assert "failed deterministic validation" in plan_prompts[1]
    assert pact_input.initial_plan.missions[0].target_ac_ids == ["ac-option"]
    assert len(pact_usage) == 3


@pytest.mark.asyncio
async def test_pact_initial_plan_retries_after_structured_output_parse_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeModel(_decision(selected_ids=("p1-g1",)))
    original_generate_plan = model.generate_pact_initial_plan
    plan_prompts: list[str] = []

    async def generate_plan_after_parse_failure(prompt: str):
        plan_prompts.append(prompt)
        if len(plan_prompts) == 1:
            raise ValidationError.from_exception_data(
                "PactInitialPlan",
                [{"type": "json_invalid", "loc": (), "input": "", "ctx": {"error": "EOF"}}],
            )
        return await original_generate_plan(prompt)

    model.generate_pact_initial_plan = generate_plan_after_parse_failure  # type: ignore[method-assign]
    monkeypatch.setattr(
        "chrys.service.requirement_clarification.service.collect_base_evidence",
        lambda _snapshot, _requirement: "packet",
    )
    service = ClarificationService(model)
    revision = RequirementRevision(number=1, messages=("Add the option.",))
    snapshot = _snapshot(tmp_path)
    result = await service.clarify(revision=revision, background="", snapshot=snapshot)

    pact_input, pact_usage = await service.generate_pact_input(
        result=result,
        revision=revision,
        background="",
        snapshot=snapshot,
    )

    assert len(plan_prompts) == 2
    assert "Invalid JSON" in plan_prompts[1]
    assert pact_input.initial_plan.missions[0].target_ac_ids == ["ac-option"]
    assert len(pact_usage) == 2


@pytest.mark.asyncio
async def test_pact_initial_plan_completes_only_missing_acceptance_criterion_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeModel(_decision(selected_ids=("p1-g1",)))

    async def generate_goal_with_two_criteria(_prompt: str):
        return (
            PactGoalContract(
                schema="pact-runtime/goal-contract/v1",
                goal="Add the option end to end.",
                acceptance_criteria=[
                    PactAcceptanceCriterion(id="ac-wired", text="The option reaches its consumer."),
                    PactAcceptanceCriterion(id="ac-visible", text="The option is publicly visible."),
                ],
                non_goals=[],
            ),
            {"output_tokens": 1},
        )

    async def generate_plan_missing_one(_prompt: str):
        return (
            PactInitialPlan(
                schema="pact-runtime/initial-plan/v1",
                constraints=[],
                missions=[
                    PactMission(
                        id="wire-option",
                        objective="Wire the option to its consumer.",
                        target_ac_ids=["ac-wired"],
                        dependencies=[],
                        verification_intent="Exercise the consumer path.",
                    )
                ],
            ),
            {"output_tokens": 1},
        )

    model.generate_pact_goal_contract = generate_goal_with_two_criteria  # type: ignore[method-assign]
    model.generate_pact_initial_plan = generate_plan_missing_one  # type: ignore[method-assign]
    monkeypatch.setattr(
        "chrys.service.requirement_clarification.service.collect_base_evidence",
        lambda _snapshot, _requirement: "packet",
    )
    service = ClarificationService(model)
    revision = RequirementRevision(number=1, messages=("Add the option.",))
    snapshot = _snapshot(tmp_path)
    result = await service.clarify(revision=revision, background="", snapshot=snapshot)

    pact_input, pact_usage = await service.generate_pact_input(
        result=result,
        revision=revision,
        background="",
        snapshot=snapshot,
    )

    assert len(pact_usage) == 3
    assert [mission.id for mission in pact_input.initial_plan.missions] == [
        "wire-option",
        "cover-missing-ac-2",
    ]
    assert pact_input.initial_plan.missions[-1].objective == "The option is publicly visible."
    validate_pact_runtime_input(pact_input.goal_contract, pact_input.initial_plan)


@pytest.mark.asyncio
async def test_service_continues_with_two_valid_proposers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel(_decision(candidate_ids=("p1-g1", "p2-g1")))
    original_propose = model.propose

    async def propose_with_one_failure(prompt: str, *, sample_index: int):
        if sample_index == 2:
            raise RuntimeError("invalid structured proposal")
        return await original_propose(prompt, sample_index=sample_index)

    model.propose = propose_with_one_failure  # type: ignore[method-assign]
    monkeypatch.setattr(
        "chrys.service.requirement_clarification.service.collect_base_evidence",
        lambda _snapshot, _requirement: "packet",
    )

    result = await ClarificationService(model).clarify(
        revision=RequirementRevision(number=1, messages=("Add the option.",)),
        background="",
        snapshot=_snapshot(tmp_path),
    )

    assert len(result.proposals) == 2
    assert len(result.warnings) == 1
    assert "proposer 2 failed" in result.warnings[0]
    assert model.selector_calls == 2
    assert result.empty_reason == "selector_rejected"


@pytest.mark.asyncio
async def test_service_retries_an_empty_selector_decision_with_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeModel(_decision())

    async def select_after_review(prompt: str):
        model.selector_calls += 1
        if model.selector_calls == 1:
            return _decision(), {"output_tokens": 1}
        assert "second-pass audit" in prompt
        return (
            _decision(selected_ids=("p1-g1",)),
            {"output_tokens": 2},
        )

    model.select = select_after_review  # type: ignore[method-assign]
    monkeypatch.setattr(
        "chrys.service.requirement_clarification.service.collect_base_evidence",
        lambda _snapshot, _requirement: "packet",
    )

    result = await ClarificationService(model).clarify(
        revision=RequirementRevision(number=1, messages=("Add the option.",)),
        background="",
        snapshot=_snapshot(tmp_path),
    )

    assert model.selector_calls == 2
    assert result.empty_reason is None
    assert "Candidate 1" in result.delta


@pytest.mark.asyncio
async def test_service_continues_with_one_valid_proposer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel(_decision(candidate_ids=("p1-g1",), selected_ids=("p1-g1",)))
    original_propose = model.propose

    async def propose_with_two_failures(prompt: str, *, sample_index: int):
        if sample_index != 1:
            raise RuntimeError("invalid structured proposal")
        return await original_propose(prompt, sample_index=sample_index)

    model.propose = propose_with_two_failures  # type: ignore[method-assign]
    monkeypatch.setattr(
        "chrys.service.requirement_clarification.service.collect_base_evidence",
        lambda _snapshot, _requirement: "packet",
    )

    result = await ClarificationService(model).clarify(
        revision=RequirementRevision(number=1, messages=("Add the option.",)),
        background="",
        snapshot=_snapshot(tmp_path),
    )

    assert result.status == "completed"
    assert result.empty_reason is None
    assert "Candidate 1" in result.delta


@pytest.mark.asyncio
async def test_legacy_v1_exact_preserves_two_empty_proposals_and_one_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _LegacyModel(_FakeModel):
        def __init__(self) -> None:
            super().__init__(_decision())
            self.legacy_selector_prompt = ""

        async def propose_legacy_v1(self, _prompt: str, *, sample_index: int):
            self.proposal_calls.append(sample_index)
            points = [] if sample_index < 3 else _proposal(sample_index).guidance_points
            return LegacyV1ClarificationProposal(guidance_points=points), {"input_tokens": sample_index}

        async def select_legacy_v1(self, prompt: str):
            self.selector_calls += 1
            self.legacy_selector_prompt = prompt
            return (
                ClarificationSelection(
                    guidance_points=[
                        SelectedGuidancePoint(
                            category="integration",
                            statement="Candidate 3 connects the option to its runtime consumer.",
                            confidence=0.8,
                            basis="current_repo",
                        )
                    ]
                ),
                {"output_tokens": 5},
            )

    model = _LegacyModel()
    monkeypatch.setattr(
        "chrys.service.requirement_clarification.service.collect_base_evidence",
        lambda _snapshot, _requirement: "packet",
    )

    result = await ClarificationService(model, strategy="legacy-v1-exact").clarify(
        revision=RequirementRevision(number=1, messages=("Add the option.",)),
        background="",
        snapshot=_snapshot(tmp_path),
    )

    assert result.strategy_version == "chrys-requirement-clarification-v1"
    assert [len(proposal.guidance_points) for proposal in result.proposals] == [0, 0, 1]
    assert "Proposal 3" in model.legacy_selector_prompt
    assert result.delta.endswith("Candidate 3 connects the option to its runtime consumer.")
    assert len(result.usage_details) == 4


@pytest.mark.asyncio
async def test_service_accepts_evidenced_complete_proposals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel(_decision())

    async def propose_complete(_prompt: str, *, sample_index: int):
        model.proposal_calls.append(sample_index)
        return ProposalModelResult(
            proposal=ClarificationProposal(
                verdict="requirement_complete",
                rationale="The original requirement is complete for this focus.",
                coverage=_coverage(),
                evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
                guidance_points=[],
            ),
            investigation=ProposalInvestigation(
                sample_index=sample_index,
                status="completed",
                investigation_attempts=1,
                synthesis_attempts=1,
            ),
            usage_details=({"input_tokens": sample_index},),
        )

    model.propose = propose_complete  # type: ignore[method-assign]
    monkeypatch.setattr(
        "chrys.service.requirement_clarification.service.collect_base_evidence",
        lambda _snapshot, _requirement: "packet",
    )

    result = await ClarificationService(model).clarify(
        revision=RequirementRevision(number=1, messages=("Add the option.",)),
        background="",
        snapshot=_snapshot(tmp_path),
    )

    assert result.is_empty
    assert result.status == "completed"
    assert result.empty_reason == "requirement_complete"
    assert all(not proposal.guidance_points for proposal in result.proposals)


@pytest.mark.asyncio
async def test_service_degrades_after_two_unknown_selector_decisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _FakeModel(_decision(candidate_ids=("p3-g6",)))
    monkeypatch.setattr(
        "chrys.service.requirement_clarification.service.collect_base_evidence",
        lambda _snapshot, _requirement: "packet",
    )

    result = await ClarificationService(model).clarify(
        revision=RequirementRevision(number=1, messages=("Add the option.",)),
        background="",
        snapshot=_snapshot(tmp_path),
    )

    assert result.status == "degraded"
    assert result.empty_reason == "selector_failed"
    assert model.selector_calls == 2


def test_render_delta_applies_confidence_and_character_budgets() -> None:
    selection = ClarificationSelection(
        guidance_points=[
            SelectedGuidancePoint(
                category="integration",
                statement="keep",
                confidence=0.75,
                basis="current_repo",
            ),
            SelectedGuidancePoint(
                category="testing",
                statement="drop",
                confidence=0.749,
                basis="current_repo",
            ),
        ]
    )

    assert render_delta(selection) == "Repository implementation guidance:\n- keep"


def test_materialize_selection_caps_model_overselection_before_schema_validation() -> None:
    candidates = {
        f"p1-g{index}": ProposalGuidancePoint(
            category="integration",
            statement=f"Candidate {index} maps the request to a repository seam.",
            confidence=confidence,
            basis="current_repo",
            contract_cell="transport_integration",
            decision_impact="The value otherwise does not reach its consumer.",
            evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
            risk="Confirm the consumer is authoritative.",
        )
        for index, confidence in enumerate((0.76, 0.99, 0.8, 0.95, 0.85, 0.9), start=1)
    }
    decision = _decision(
        candidate_ids=tuple(candidates),
        selected_ids=tuple(candidates),
    )

    selection = _materialize_selection(decision, candidates)

    assert len(selection.guidance_points) == 5
    assert [point.confidence for point in selection.guidance_points] == [0.99, 0.95, 0.9, 0.85, 0.8]


def test_pact_pair_validation_rejects_cycles() -> None:
    goal_contract = PactGoalContract(
        schema="pact-runtime/goal-contract/v1",
        goal="Deliver both behaviors.",
        acceptance_criteria=[PactAcceptanceCriterion(id="ac-both", text="Both behaviors are observable.")],
        non_goals=[],
    )
    initial_plan = PactInitialPlan(
        schema="pact-runtime/initial-plan/v1",
        constraints=[],
        missions=[
            PactMission(
                id="first",
                objective="Deliver the first behavior.",
                target_ac_ids=["ac-both"],
                dependencies=["second"],
                verification_intent="Observe the first behavior.",
            ),
            PactMission(
                id="second",
                objective="Deliver the second behavior.",
                target_ac_ids=["ac-both"],
                dependencies=["first"],
                verification_intent="Observe the second behavior.",
            ),
        ],
    )

    with pytest.raises(ValueError, match="contains a cycle"):
        validate_pact_runtime_input(goal_contract, initial_plan)
