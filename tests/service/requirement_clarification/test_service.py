# Copyright (c) 2026 Chrys. All rights reserved.

"""Three-proposal/one-selector clarification behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.service.requirement_clarification.service import (
    ClarificationService,
    render_delta,
    validate_pact_runtime_input,
)
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot
from chrys.service.requirement_clarification.types import (
    ClarificationProposal,
    ClarificationSelection,
    PactAcceptanceCriterion,
    PactGoalContract,
    PactInitialPlan,
    PactMission,
    RequirementRevision,
    SelectedGuidancePoint,
)


class _FakeModel:
    def __init__(self, selection: ClarificationSelection) -> None:
        self.selection = selection
        self.proposal_calls: list[int] = []
        self.selector_calls = 0

    async def propose(self, _prompt: str, *, sample_index: int):
        self.proposal_calls.append(sample_index)
        return ClarificationProposal(), {"input_tokens": sample_index}

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


@pytest.mark.asyncio
async def test_service_runs_three_proposals_then_one_selector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection = ClarificationSelection(
        guidance_points=[
            SelectedGuidancePoint(
                category="integration",
                statement="Wire the parsed value into the existing runtime consumer.",
                confidence=0.91,
                basis="current_repo",
            )
        ]
    )
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
        "Repository implementation guidance:\n- Wire the parsed value into the existing runtime consumer."
    )
    assert result.raw_selection is selection
    assert pact_input.goal_contract.acceptance_criteria[0].id == "ac-option"
    assert len(result.usage_details) == 4
    assert len(pact_usage) == 2


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
