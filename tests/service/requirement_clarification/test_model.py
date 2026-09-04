# Copyright (c) 2026 Chrys. All rights reserved.

"""Clarification side-agent option contracts."""

import hashlib
import json

import pytest
from pydantic import BaseModel

from chrys.kernel import AgentResponse, Content, Message
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.requirement_clarification.model import (
    ChrysClarificationModel,
    _investigation_errors,
    _investigation_tool_calls,
    _proposal_coverage_errors,
    _proposal_errors,
    _stateless_options,
)
from chrys.service.requirement_clarification.prompts import (
    legacy_v1_proposal_instructions,
    legacy_v1_selector_instructions,
)
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot
from chrys.service.requirement_clarification.types import (
    ClarificationProposal,
    ClarificationSelectorDecision,
    EvidenceAnchor,
    InvestigationCoverageFinding,
    LegacyV1ClarificationProposal,
    ProposalGuidancePoint,
)


def test_legacy_v1_wire_contract_matches_historical_7619a94() -> None:
    schema = json.dumps(
        LegacyV1ClarificationProposal.model_json_schema(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )

    assert LegacyV1ClarificationProposal.__name__ == "ClarificationProposal"
    assert hashlib.sha256(legacy_v1_proposal_instructions().encode()).hexdigest() == (
        "615e67f1033cc3b33557ec581d9050905d3b34be940bd84147209109c1a8b78f"
    )
    assert hashlib.sha256(legacy_v1_selector_instructions().encode()).hexdigest() == (
        "6aa165e4d1317523a3ccb4646103545a0d08452e0ab86403e7eb32e35a822447"
    )
    assert hashlib.sha256(schema.encode()).hexdigest() == (
        "6bbb903bdfc5380160196b7b58977687169e3f1bb7068edd8cb156bdce3f8541"
    )


def _profile() -> ModelProfile:
    return ModelProfile(id="model", name="Model", provider="mock", model_id="mock")


def _coverage(
    primary: str = "src/runtime.py:10",
    adjacent: str = "src/runtime.py:20",
) -> list[InvestigationCoverageFinding]:
    return [
        InvestigationCoverageFinding(
            target="primary",
            status="found",
            summary="The authoritative implementation owner was inspected.",
            evidence=[EvidenceAnchor(kind="current_repo", anchor=primary)],
        ),
        InvestigationCoverageFinding(
            target="adjacent",
            status="found",
            summary="The adjacent integration or consumer surface was inspected.",
            evidence=[EvidenceAnchor(kind="current_repo", anchor=adjacent)],
        ),
    ]


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
    arguments = {"path": path}
    if name in {"grep", "glob"}:
        arguments["pattern"] = "option"
    return AgentResponse(
        messages=[
            Message("assistant", [Content.from_function_call(call_id, name, arguments=arguments)]),
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
    assert calls[0].query == "option"
    assert calls[0].result_sha256


def test_investigation_pairs_reused_call_ids_within_each_exchange() -> None:
    calls = _investigation_tool_calls(
        [
            _tool_response("call-0", "grep", ".", "src/runtime.py:10: consume(option)"),
            _tool_response("call-0", "read_file", "src/runtime.py", "10 consume(option)"),
        ]
    )

    assert [(call.name, call.path, call.successful) for call in calls] == [
        ("grep", ".", True),
        ("read_file", "src/runtime.py", True),
    ]


def test_investigation_records_query_line_range_and_result_fingerprint() -> None:
    response = AgentResponse(
        messages=[
            Message(
                "assistant",
                [
                    Content.from_function_call(
                        "read-1",
                        "read_file",
                        arguments={"path": "src/runtime.py", "line_range": [10, 28]},
                    )
                ],
            ),
            Message("tool", [Content.from_function_result("read-1", result="10 consume(option)")]),
        ]
    )

    call = _investigation_tool_calls([response])[0]

    assert call.call_id == "read-1"
    assert call.line_range == (10, 28)
    assert call.result_sha256 == hashlib.sha256(b"10 consume(option)").hexdigest()


def test_proposal_semantics_reject_placeholder_even_with_valid_tool_trace() -> None:
    calls = _investigation_tool_calls(
        [
            _tool_response("grep-1", "grep", ".", "src/runtime.py:10: consume(option)"),
            AgentResponse(
                messages=[
                    Message(
                        "assistant",
                        [Content.from_function_call("grep-2", "grep", arguments={"pattern": "dispatch", "path": "."})],
                    ),
                    Message("tool", [Content.from_function_result("grep-2", result="src/runtime.py:20: dispatch")]),
                ]
            ),
            _tool_response("read-1", "read_file", "src/runtime.py", "10 consume(option)"),
        ]
    )
    proposal = ClarificationProposal(
        verdict="requirement_complete",
        rationale="PLACEHOLDER - will refine after inspecting repository",
        coverage=_coverage(),
        evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
        guidance_points=[],
    )

    assert _proposal_errors(proposal, calls) == ["proposal contains placeholder or unfinished investigation text"]


def test_proposal_coverage_rejects_uninspected_repository_anchor() -> None:
    calls = _investigation_tool_calls(
        [
            AgentResponse(
                messages=[
                    Message(
                        "assistant",
                        [Content.from_function_call("grep-1", "grep", arguments={"pattern": "option", "path": "."})],
                    ),
                    Message("tool", [Content.from_function_result("grep-1", result="src/runtime.py:10: option")]),
                ]
            ),
            _tool_response("read-1", "read_file", "src/runtime.py", "10 consume(option)"),
        ]
    )
    proposal = ClarificationProposal(
        verdict="clarification_needed",
        rationale="Registration is owned by a separate repository surface.",
        coverage=_coverage(),
        evidence=[EvidenceAnchor(kind="current_repo", anchor="src/registry.py:12")],
        guidance_points=[
            ProposalGuidancePoint(
                category="integration",
                statement="Register the option with the runtime consumer.",
                confidence=0.9,
                basis="current_repo",
                contract_cell="transport_integration",
                decision_impact="The consumer otherwise cannot observe the option.",
                evidence=[EvidenceAnchor(kind="current_repo", anchor="src/registry.py:12")],
                risk="The registry must remain the authoritative integration surface.",
            )
        ],
    )

    assert _proposal_coverage_errors(proposal, calls) == [
        "current-repository evidence cites a file not inspected with read_file: src/registry.py:12"
    ]


def test_evidenced_empty_requires_breadth_beyond_one_search_and_one_file() -> None:
    calls = _investigation_tool_calls(
        [
            AgentResponse(
                messages=[
                    Message(
                        "assistant",
                        [Content.from_function_call("grep-1", "grep", arguments={"pattern": "option", "path": "."})],
                    ),
                    Message("tool", [Content.from_function_result("grep-1", result="src/runtime.py:10: option")]),
                ]
            ),
            _tool_response("read-1", "read_file", "src/runtime.py", "10 consume(option)"),
        ]
    )
    proposal = ClarificationProposal(
        verdict="requirement_complete",
        rationale="The owner and consumer are fully specified by the requirement.",
        coverage=_coverage(),
        evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
        guidance_points=[],
    )

    assert _proposal_coverage_errors(proposal, calls) == [
        "requirement_complete lacks a second inspected surface or targeted search"
    ]


class _SequenceAgent:
    def __init__(self, responses: list[AgentResponse[object]]) -> None:
        self._responses = iter(responses)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create_session(self, *, session_id: str | None = None) -> object:
        return {"session_id": session_id}

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_value, _traceback) -> None:
        return None

    async def run(self, message: str, *, stream: bool, session: object, options: dict[str, object]):
        assert stream is False
        assert session
        self.calls.append((message, options))
        return next(self._responses)


async def test_coverage_failure_returns_to_investigation_before_resynthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_investigation = AgentResponse(
        messages=[
            Message(
                "assistant",
                [Content.from_function_call("grep-1", "grep", arguments={"pattern": "option", "path": "."})],
            ),
            Message("tool", [Content.from_function_result("grep-1", result="src/runtime.py:10: option")]),
            Message(
                "assistant",
                [Content.from_function_call("read-1", "read_file", arguments={"path": "src/runtime.py"})],
            ),
            Message("tool", [Content.from_function_result("read-1", result="10 consume(option)")]),
        ]
    )
    shallow_empty = ClarificationProposal(
        verdict="requirement_complete",
        rationale="The requirement appears complete for the owner.",
        coverage=_coverage(),
        evidence=[EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10")],
        guidance_points=[],
    )
    second_investigation = AgentResponse(
        messages=[
            Message(
                "assistant",
                [Content.from_function_call("grep-2", "grep", arguments={"pattern": "register", "path": "."})],
            ),
            Message("tool", [Content.from_function_result("grep-2", result="src/registry.py:20: register")]),
            Message(
                "assistant",
                [Content.from_function_call("read-2", "read_file", arguments={"path": "src/registry.py"})],
            ),
            Message("tool", [Content.from_function_result("read-2", result="20 register(option)")]),
        ]
    )
    evidenced_empty = ClarificationProposal(
        verdict="requirement_complete",
        rationale="The owner and registration surfaces add no unstated constraint.",
        coverage=_coverage("src/runtime.py:10", "src/registry.py:20"),
        evidence=[
            EvidenceAnchor(kind="current_repo", anchor="src/runtime.py:10"),
            EvidenceAnchor(kind="current_repo", anchor="src/registry.py:20"),
        ],
        guidance_points=[],
    )
    agent = _SequenceAgent(
        [
            first_investigation,
            AgentResponse(value=shallow_empty),
            second_investigation,
            AgentResponse(value=evidenced_empty),
        ]
    )
    model = ChrysClarificationModel(
        profile=_profile(),
        snapshot=WorkspaceSnapshot(
            snapshot_id="s0",
            artifact_root="/snapshot",
            roots=(),
            manifest_hash="abc",
            total_bytes=0,
            entry_count=0,
        ),
        session_id="parent",
        session_dir=None,
    )
    monkeypatch.setattr(model, "_create_agent", lambda **_kwargs: agent)

    result = await model.propose("Investigate the option.", sample_index=1)

    assert result.proposal == evidenced_empty
    assert result.investigation.status == "completed"
    assert result.investigation.coverage_status == "sufficient"
    assert result.investigation.investigation_attempts == 2
    assert result.investigation.inspected_paths == ["src/runtime.py", "src/registry.py"]
    assert {reference.path for reference in result.investigation.verified_evidence} == {
        "src/runtime.py",
        "src/registry.py",
    }
    assert all(reference.result_sha256 for reference in result.investigation.verified_evidence)
    assert all(reference.tool_call_id.startswith("read-") for reference in result.investigation.verified_evidence)
    assert len(agent.calls) == 4
    assert "response_format" not in agent.calls[0][1]
    assert agent.calls[1][1]["tool_choice"] == "none"
    assert "controller rejected" in agent.calls[2][0].lower()
    assert "response_format" not in agent.calls[2][1]


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
        coverage=_coverage(),
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


class _Reply(BaseModel):
    ok: bool


class _RunAgent:
    """An agent for ``_run``: replies in order, records prompts."""

    def __init__(self, responses: list[object]) -> None:
        self._responses = iter(responses)
        self.prompts: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, _exc_type, _exc_value, _traceback) -> None:
        return None

    async def run(self, message: str, *, stream: bool, options: dict[str, object]):
        assert stream is False
        self.prompts.append(message)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


def _model_for_run(monkeypatch: pytest.MonkeyPatch, agent: _RunAgent) -> ChrysClarificationModel:
    model = ChrysClarificationModel(
        profile=_profile(),
        snapshot=WorkspaceSnapshot(
            snapshot_id="s0", artifact_root="/snapshot", roots=(), manifest_hash="abc", total_bytes=0, entry_count=0
        ),
        session_id="parent",
        session_dir=None,
    )
    monkeypatch.setattr(model, "_create_agent", lambda **_kwargs: agent)
    return model


@pytest.mark.asyncio
async def test_a_structured_side_call_that_returns_no_object_is_asked_once_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import ValidationError as _ValidationError

    try:
        _Reply.model_validate_json("Based on the requirement, here is the contract")
    except _ValidationError as error:
        invalid = error
    agent = _RunAgent([invalid, AgentResponse(value=_Reply(ok=True))])
    model = _model_for_run(monkeypatch, agent)

    value, _usage = await model._run(
        "Produce the object.", response_format=_Reply, instructions="i", route_kind="k", route_part="1"
    )

    assert value == _Reply(ok=True)
    assert len(agent.prompts) == 2
    assert agent.prompts[1].startswith("Produce the object.")
    assert "Reply with exactly one JSON object" in agent.prompts[1]
    assert "The error was:" in agent.prompts[1]


@pytest.mark.asyncio
async def test_a_structured_side_call_gives_up_after_the_second_miss(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _RunAgent([AgentResponse(value="prose"), AgentResponse(value="still prose")])
    model = _model_for_run(monkeypatch, agent)

    with pytest.raises(ValueError, match="returned no _Reply"):
        await model._run(
            "Produce the object.", response_format=_Reply, instructions="i", route_kind="k", route_part="1"
        )
    assert len(agent.prompts) == 2
