# Copyright (c) 2026 Chrys. All rights reserved.

"""Clarification side-agent option contracts."""

from chrys.kernel import AgentResponse, Content, Message
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.requirement_clarification.model import (
    _investigation_errors,
    _investigation_tool_calls,
    _proposal_errors,
    _stateless_options,
)
from chrys.service.requirement_clarification.types import (
    ClarificationProposal,
    ClarificationSelectorDecision,
    EvidenceAnchor,
    ProposalGuidancePoint,
)


def _profile() -> ModelProfile:
    return ModelProfile(id="model", name="Model", provider="mock", model_id="mock")


def test_proposer_options_require_the_first_tool_call() -> None:
    options = _stateless_options(_profile(), ClarificationProposal, required_tool_name="grep")

    assert options["tool_choice"] == {
        "mode": "required",
        "required_function_name": "grep",
    }


def test_other_side_call_options_leave_tool_choice_automatic() -> None:
    options = _stateless_options(_profile(), ClarificationSelectorDecision)

    assert "tool_choice" not in options


def test_investigation_options_do_not_request_structured_output() -> None:
    options = _stateless_options(_profile(), None, required_tool_name="grep")

    assert "response_format" not in options
    assert options["tool_choice"]["required_function_name"] == "grep"


def _tool_response(call_id: str, name: str, path: str, result: str) -> AgentResponse[object]:
    return AgentResponse(
        messages=[
            Message("assistant", [Content.from_function_call(call_id, name, arguments={"path": path})]),
            Message("tool", [Content.from_function_result(call_id, result=result)]),
        ]
    )


def test_investigation_requires_successful_search_and_file_read() -> None:
    calls = _investigation_tool_calls(
        [
            _tool_response("grep-1", "grep", ".", "src/runtime.py:10: consume(option)"),
            _tool_response("read-1", "read_file", "src/runtime.py", "10 consume(option)"),
        ]
    )

    assert _investigation_errors(calls) == []
    assert [(call.name, call.successful) for call in calls] == [("grep", True), ("read_file", True)]


def test_proposal_semantics_reject_placeholder_even_with_valid_tool_trace() -> None:
    calls = _investigation_tool_calls(
        [
            _tool_response("grep-1", "grep", ".", "src/runtime.py:10: consume(option)"),
            _tool_response("read-1", "read_file", "src/runtime.py", "10 consume(option)"),
        ]
    )
    proposal = ClarificationProposal(
        verdict="requirement_complete",
        rationale="PLACEHOLDER - will refine after inspecting repository",
        evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
        guidance_points=[],
    )

    assert _proposal_errors(proposal, calls) == ["proposal contains placeholder or unfinished investigation text"]


def test_proposal_semantics_reject_unresolved_confirmation_note() -> None:
    calls = _investigation_tool_calls(
        [
            _tool_response("grep-1", "grep", ".", "src/runtime.py:10: consume(option)"),
            _tool_response("read-1", "read_file", "src/runtime.py", "10 consume(option)"),
        ]
    )
    proposal = ClarificationProposal(
        verdict="clarification_needed",
        rationale="The repository exposes an implementation-specific registration seam.",
        evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
        guidance_points=[
            ProposalGuidancePoint(
                category="integration",
                statement="Need to confirm that the option is registered in setup.cfg before implementation.",
                confidence=0.9,
                basis="current_repo",
                contract_cell="transport_integration",
                decision_impact="Registration controls whether the option reaches its runtime consumer.",
                evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
                risk="The registration owner must be established from inspected repository evidence.",
            )
        ],
    )

    assert _proposal_errors(proposal, calls) == ["proposal contains placeholder or unfinished investigation text"]
