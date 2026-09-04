# Copyright (c) 2026 Chrys. All rights reserved.

"""Phase-oriented requirement-clarification artifact tests."""

from __future__ import annotations

import json
from pathlib import Path

from chrys.service.requirement_clarification.artifacts import ClarificationArtifactStore
from chrys.service.requirement_clarification.types import (
    ClarificationProposal,
    ClarificationResult,
    ClarificationSelection,
    ClarificationSelectorDecision,
    EvidenceAnchor,
    InvestigationCoverageFinding,
    InvestigationToolCall,
    PactAcceptanceCriterion,
    PactGoalContract,
    PactInitialPlan,
    PactMission,
    PactRuntimeInput,
    ProposalGuidancePoint,
    ProposalInvestigation,
    SelectedGuidancePoint,
    SelectorCandidateReview,
)


def _selection(statement: str) -> ClarificationSelection:
    return ClarificationSelection(
        guidance_points=[
            SelectedGuidancePoint(
                category="integration",
                statement=statement,
                confidence=0.9,
                basis="current_repo",
            )
        ]
    )


def test_artifacts_escape_surrogateescaped_identity_text(tmp_path: Path) -> None:
    store = ClarificationArtifactStore(tmp_path, 1)

    store.save_requirement_input(revision=1, messages=("Inspect \udcff/path",))
    store.save_snapshot_metadata({"source_root": "/repo/\udcff"})

    requirement = (store.root / "01-input" / "requirement.md").read_text(encoding="utf-8")
    metadata = json.loads((store.root / "01-input" / "workspace-snapshot.json").read_text(encoding="utf-8"))
    assert "\\udcff" in requirement
    assert metadata["source_root"] == "/repo/\udcff"


def _proposal(statement: str) -> ClarificationProposal:
    return ClarificationProposal(
        verdict="clarification_needed",
        rationale="The repository exposes a runtime integration seam.",
        coverage=[
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
        ],
        evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
        guidance_points=[
            ProposalGuidancePoint(
                category="integration",
                statement=statement,
                confidence=0.8,
                basis="current_repo",
                contract_cell="transport_integration",
                decision_impact="The value otherwise does not reach its consumer.",
                evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
                risk="Confirm the consumer before accepting the candidate.",
            )
        ],
    )


def test_store_writes_phase_oriented_outputs_and_legacy_artifacts(tmp_path: Path) -> None:
    store = ClarificationArtifactStore(tmp_path / "session", 2)
    requirement_messages = ("Original requirement.", "Preserve compatibility.")
    raw_selection = ClarificationSelectorDecision(
        reviews=[
            SelectorCandidateReview(
                candidate_id="p1-g1",
                decision="select",
                rationale="This is the strongest necessary repository mapping.",
            )
        ]
    )
    cleaned_selection = _selection("cleaned guidance")
    result = ClarificationResult(
        strategy_version="test-v1",
        revision=3,
        delta="Repository implementation guidance:\n- cleaned guidance",
        selection=cleaned_selection,
        raw_selection=raw_selection,
        pact_input=PactRuntimeInput(
            goal_contract=PactGoalContract(
                schema="pact-runtime/goal-contract/v1",
                goal="Add the option.",
                acceptance_criteria=[PactAcceptanceCriterion(id="ac-option", text="The option takes effect.")],
                non_goals=[],
            ),
            initial_plan=PactInitialPlan(
                schema="pact-runtime/initial-plan/v1",
                constraints=[],
                missions=[
                    PactMission(
                        id="wire-option",
                        objective="Wire the option end to end.",
                        target_ac_ids=["ac-option"],
                        dependencies=[],
                        verification_intent="Exercise the option through its public path.",
                    )
                ],
            ),
        ),
        proposals=(_proposal("candidate one"), _proposal("candidate two"), _proposal("candidate three")),
        investigations=tuple(
            ProposalInvestigation(
                sample_index=index,
                focus=focus,
                status="completed",
                investigation_attempts=1,
                synthesis_attempts=1,
                coverage_status="sufficient",
                required_coverage=["Inspect the owner.", "Inspect the adjacent surface."],
                search_queries=["runtime"],
                inspected_paths=["src/runtime.py"],
                input_token_count=12000,
                output_token_count=800,
                total_token_count=12800,
                tool_calls=[
                    InvestigationToolCall(
                        call_id=f"read-{index}",
                        name="read_file",
                        path="src/runtime.py",
                        successful=True,
                        result_chars=100,
                        result_sha256="a" * 64,
                    )
                ],
            )
            for index, focus in enumerate(
                ("ownership_extension", "data_control_flow", "boundary_compatibility"),
                start=1,
            )
        ),
        elapsed_seconds=1.5,
    )

    store.save_requirement_input(revision=3, messages=requirement_messages)
    store.save_snapshot_metadata({"workflow_id": "workflow", "s0": {"snapshot_id": "s0"}, "p0": None})
    store.save_initial_response(revision=3, response="P0")
    store.save_initial_transcript({"history": {"messages": []}, "service_session_id": "provider-p0"})
    store.save_result(result, requirement_messages=requirement_messages)
    store.save_pact_generation(result)
    store.save_repair_attempt(
        revision=3,
        status="succeeded",
        response="P1",
        transcript={"history": {"messages": []}, "service_session_id": "provider-p1"},
    )
    store.save_summary(
        {
            "workflow_id": "workflow",
            "revision": 3,
            "outcome": "repaired",
            "accepted_phase": "repair",
            "final_response": "P1",
        },
        requirement_messages=requirement_messages,
        delta=result.delta,
    )

    assert (store.root / "clarification.private.json").is_file()
    assert (store.root / "initial_implementation.private.json").is_file()
    assert (store.root / "summary.json").is_file()
    assert (store.root / "01-input/requirement.md").is_file()
    assert (store.root / "01-input/workspace-snapshot.json").is_file()
    assert (store.root / "02-initial-trial/response.json").is_file()
    assert (store.root / "02-initial-trial/transcript.private.json").is_file()
    assert len(list((store.root / "03-clarification/candidates").glob("proposal-*.private.json"))) == 3
    assert len(list((store.root / "03-clarification/investigations").glob("proposal-*.private.json"))) == 3
    investigation = json.loads(
        (store.root / "03-clarification/investigations/proposal-1.private.json").read_text(encoding="utf-8")
    )
    assert investigation["schema"] == "chrys/requirement-clarification/investigation/v2"
    assert investigation["investigation"]["coverage_status"] == "sufficient"
    assert investigation["investigation"]["tool_calls"][0]["result_sha256"] == "a" * 64
    assert (store.root / "03-clarification/sources/delta.md").read_text(encoding="utf-8").endswith("\n")
    assert (store.root / "04-repair/attempts/revision-3/response.json").is_file()
    goal_contract = json.loads((store.root / "06-pact-input/goal-contract.json").read_text(encoding="utf-8"))
    assert goal_contract["schema"] == "pact-runtime/goal-contract/v1"
    assert "schema_id" not in goal_contract
    assert set(goal_contract) == {"schema", "goal", "acceptance_criteria", "non_goals"}
    initial_plan = json.loads((store.root / "06-pact-input/initial-plan.json").read_text(encoding="utf-8"))
    assert set(initial_plan) == {"schema", "constraints", "missions"}
    assert set(initial_plan["missions"][0]) == {
        "id",
        "objective",
        "target_ac_ids",
        "dependencies",
        "verification_intent",
    }
    generation = json.loads((store.root / "06-pact-input/generation.private.json").read_text(encoding="utf-8"))
    assert generation["status"] == "generated"

    selection = json.loads(
        (store.root / "03-clarification/decision/selection.private.json").read_text(encoding="utf-8")
    )
    assert selection["raw"]["reviews"][0]["candidate_id"] == "p1-g1"
    assert selection["raw"]["reviews"][0]["decision"] == "select"
    assert selection["cleaned"]["guidance_points"][0]["statement"] == "cleaned guidance"
    deliverable = (store.root / "03-clarification/deliverable/clarified-requirement.md").read_text(encoding="utf-8")
    outcome_requirement = (store.root / "05-outcome/clarified-requirement.md").read_text(encoding="utf-8")
    assert outcome_requirement == deliverable
    assert "Original requirement." in outcome_requirement
    assert "Preserve compatibility." in outcome_requirement
    assert result.delta in outcome_requirement
    outcome_delta = (store.root / "05-outcome/clarified-requirement-delta.md").read_text(encoding="utf-8")
    assert "Original requirement." in outcome_delta
    assert result.delta in outcome_delta
    assert "Preserve compatibility." not in outcome_delta
    assert (store.root / "05-outcome/final-response.md").read_text(encoding="utf-8") == "P1\n"
    summary = json.loads((store.root / "05-outcome/summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "chrys/requirement-clarification/final-summary/v1"
    assert summary["accepted_phase"] == "repair"


def test_store_distinguishes_degraded_clarification_from_legitimate_empty(tmp_path: Path) -> None:
    store = ClarificationArtifactStore(tmp_path / "session", 1)
    result = ClarificationResult(
        strategy_version="test-v7",
        revision=1,
        delta="",
        selection=ClarificationSelection(),
        status="degraded",
        empty_reason="insufficient_valid_proposals",
        investigations=(
            ProposalInvestigation(
                sample_index=1,
                status="failed",
                investigation_attempts=2,
                synthesis_attempts=0,
                validation_errors=["investigation has no successful read_file"],
            ),
        ),
    )

    store.save_result(result, requirement_messages=("Original requirement.",))
    store.save_pact_generation(result)

    private = json.loads((store.root / "clarification.private.json").read_text(encoding="utf-8"))
    assert private["status"] == "degraded"
    assert private["empty_reason"] == "insufficient_valid_proposals"
    clarified = (store.root / "03-clarification/deliverable/clarified-requirement.md").read_text(encoding="utf-8")
    assert "failed or degraded" in clarified
    generation = json.loads((store.root / "06-pact-input/generation.private.json").read_text(encoding="utf-8"))
    assert generation["status"] == "skipped"
