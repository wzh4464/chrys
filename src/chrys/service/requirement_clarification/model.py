# Copyright (c) 2026 Chrys. All rights reserved.

"""Fresh, read-only Chrys agents used by proposal and selector passes."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from chrys.foundation.models.session_env import SessionEnvironment
from chrys.kernel import CONVERSATION_HANDLE_KEYS, Agent
from chrys.service.llm.clients import create_client
from chrys.service.llm.route_sessions import derive_llm_route_session_id
from chrys.service.profiles.models.options import effective_chat_options
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.requirement_clarification.prompts import (
    pact_goal_contract_instructions,
    pact_initial_plan_instructions,
    proposal_instructions,
    selector_instructions,
)
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot
from chrys.service.requirement_clarification.types import (
    ClarificationProposal,
    ClarificationSelectorDecision,
    InvestigationToolCall,
    PactGoalContract,
    PactInitialPlan,
    ProposalInvestigation,
    ProposalModelResult,
)
from chrys.service.tools.registry import ToolRegistry

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class ClarificationModelRunner(Protocol):
    """Model boundary consumed by :class:`ClarificationService`."""

    async def propose(self, prompt: str, *, sample_index: int) -> ProposalModelResult: ...

    async def select(self, prompt: str) -> tuple[ClarificationSelectorDecision, dict[str, object]]: ...

    async def generate_pact_goal_contract(self, prompt: str) -> tuple[PactGoalContract, dict[str, object]]: ...

    async def generate_pact_initial_plan(self, prompt: str) -> tuple[PactInitialPlan, dict[str, object]]: ...


def _stateless_options[ResponseT: BaseModel](
    profile: ModelProfile,
    response_format: type[ResponseT] | None,
    *,
    required_tool_name: str | None = None,
) -> dict[str, Any]:
    options = dict(effective_chat_options(profile) or {})
    if response_format is not None:
        options["response_format"] = response_format
    if required_tool_name is not None:
        options["tool_choice"] = {
            "mode": "required",
            "required_function_name": required_tool_name,
        }
    if "store" in options or profile.provider == "openai":
        options["store"] = False
    for key in CONVERSATION_HANDLE_KEYS:
        options.pop(key, None)
    options.pop("continuation_token", None)
    options.pop("background", None)
    extra_body = options.get("extra_body")
    if isinstance(extra_body, Mapping):
        clean_extra = {
            key: value
            for key, value in extra_body.items()
            if key not in {*CONVERSATION_HANDLE_KEYS, "continuation_token", "background"}
        }
        if "store" in clean_extra:
            clean_extra["store"] = False
        options["extra_body"] = clean_extra
    return options


_PLACEHOLDER_PATTERNS = (
    re.compile(
        r"^\s*(?:placeholder|todo|pending|test|no|n/?a|tbd|beginning|probe_start|initial_exploration)\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"\bwill\s+(?:refine|investigate|inspect)\b", re.IGNORECASE),
    re.compile(r"\bneed(?:s|ed)?\s+to\s+(?:inspect|investigate)\b", re.IGNORECASE),
    re.compile(r"^\s*need(?:s|ed)?\s+to\s+(?:confirm|determine|verify|check)\b", re.IGNORECASE),
    re.compile(r"\binvestigat(?:e|ing|ion)\b.*\b(?:first|pending|before)\b", re.IGNORECASE),
    re.compile(r"\bno\s+guidance\s+yet\b", re.IGNORECASE),
    re.compile(r"\bto\s+be\s+replaced\b", re.IGNORECASE),
    re.compile(r"\b(?:schema|validation)\b.*\b(?:not\s+provided|switch\s+to)\b", re.IGNORECASE),
    re.compile(r"\bsynthetic\s+investiga(?:t)?ion\b", re.IGNORECASE),
)
_INVESTIGATION_MAX_ITERATIONS = 10
_INVESTIGATION_MAX_FUNCTION_CALLS = 10


def _contains_placeholder(value: str) -> bool:
    return any(pattern.search(value) for pattern in _PLACEHOLDER_PATTERNS)


def _call_arguments(content: Any) -> dict[str, object]:
    arguments = getattr(content, "arguments", None)
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        if isinstance(value, dict):
            return value
    return {}


def _investigation_tool_calls(responses: list[Any]) -> list[InvestigationToolCall]:
    calls: dict[str, tuple[str, str]] = {}
    results: dict[str, tuple[bool, int]] = {}
    for response in responses:
        for message in getattr(response, "messages", ()):
            for content in getattr(message, "contents", ()):
                call_id = getattr(content, "call_id", None)
                if not isinstance(call_id, str) or not call_id:
                    continue
                if getattr(content, "type", None) == "function_call":
                    name = getattr(content, "name", "")
                    if name not in {"grep", "glob", "read_file", "view_image"}:
                        continue
                    arguments = _call_arguments(content)
                    path = arguments.get("path", "")
                    calls[call_id] = (name, path if isinstance(path, str) else "")
                elif getattr(content, "type", None) == "function_result":
                    result = getattr(content, "result", "")
                    text = result if isinstance(result, str) else str(result or "")
                    exception = getattr(content, "exception", None)
                    results[call_id] = (
                        not exception and bool(text.strip()) and not text.lstrip().startswith("Error:"),
                        len(text),
                    )
    observations: list[InvestigationToolCall] = []
    for call_id, (name, path) in calls.items():
        successful, result_chars = results.get(call_id, (False, 0))
        observations.append(
            InvestigationToolCall(name=name, path=path, successful=successful, result_chars=result_chars)
        )
    return observations


def _investigation_errors(calls: list[InvestigationToolCall]) -> list[str]:
    successful = [call for call in calls if call.successful]
    errors: list[str] = []
    if not any(call.name == "grep" for call in successful):
        errors.append("investigation has no successful grep")
    if not any(call.name == "read_file" for call in successful):
        errors.append("investigation has no successful read_file")
    return errors


def _proposal_errors(proposal: ClarificationProposal, calls: list[InvestigationToolCall]) -> list[str]:
    errors: list[str] = []
    text_fields = [proposal.rationale]
    text_fields.extend(anchor.anchor for anchor in proposal.evidence)
    for point in proposal.guidance_points:
        text_fields.extend((point.statement, point.decision_impact, point.risk))
        text_fields.extend(anchor.anchor for anchor in point.evidence)
    if any(_contains_placeholder(value) for value in text_fields):
        errors.append("proposal contains placeholder or unfinished investigation text")
    read_paths = {
        Path(call.path).as_posix() for call in calls if call.successful and call.name == "read_file" and call.path
    }
    anchors = [anchor.anchor.replace("\\", "/") for anchor in proposal.evidence]
    anchors.extend(anchor.anchor.replace("\\", "/") for point in proposal.guidance_points for anchor in point.evidence)
    if read_paths and not any(
        any(path in anchor or Path(path).name in anchor for path in read_paths) for anchor in anchors
    ):
        errors.append("proposal evidence does not cite a file inspected with read_file")
    return errors


class ChrysClarificationModel:
    """Run each clarification pass as a fresh, tool-bounded Chrys agent."""

    def __init__(
        self,
        *,
        profile: ModelProfile,
        snapshot: WorkspaceSnapshot,
        session_id: str | None,
        session_dir: Path | None,
        report_usage: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self._profile = profile
        self._snapshot = snapshot
        self._session_id = session_id
        self._session_dir = session_dir
        self._report_usage = report_usage

    async def propose(self, prompt: str, *, sample_index: int) -> ProposalModelResult:
        route_session_id = derive_llm_route_session_id(
            self._session_id,
            route_kind="requirement-clarification-proposal",
            route_parts=(self._snapshot.snapshot_id, str(sample_index)),
            model_profile=self._profile,
        )
        agent = self._create_agent(
            route_session_id=route_session_id,
            instructions=proposal_instructions(),
        )
        session = agent.create_session(session_id=route_session_id)
        responses: list[Any] = []
        usage_details: list[dict[str, object]] = []
        validation_errors: list[str] = []
        investigation_attempts = 1
        synthesis_attempts = 0

        async def run_stage(
            message: str,
            *,
            response_format: type[BaseModel] | None = None,
            required_tool_name: str | None = None,
            tool_choice_none: bool = False,
        ) -> Any:
            options = _stateless_options(
                self._profile,
                response_format,
                required_tool_name=required_tool_name,
            )
            if response_format is None:
                options["allow_multiple_tool_calls"] = False
            if tool_choice_none:
                options["tool_choice"] = "none"
            response = await agent.run(
                message,
                stream=False,
                session=session,
                options=options,
            )
            responses.append(response)
            usage = dict(response.usage_details or {})
            if usage:
                usage_details.append(usage)
                if self._report_usage is not None:
                    self._report_usage(usage)
            return response

        await agent.__aenter__()
        try:
            try:
                await run_stage(
                    "Investigate the frozen repository for this focus. Use the tools to trace concrete ownership and "
                    "integration seams. Return concise investigation notes only; do not emit proposal JSON yet.\n\n"
                    + prompt,
                    required_tool_name="grep",
                )
                calls = _investigation_tool_calls(responses)
                investigation_errors = _investigation_errors(calls)
                if investigation_errors:
                    investigation_attempts = 2
                    validation_errors.extend(investigation_errors)
                    await run_stage(
                        "The investigation is incomplete: "
                        + "; ".join(investigation_errors)
                        + ". Continue from the existing evidence, inspect the most relevant source file, and return "
                        "concrete repository findings only. Do not emit proposal JSON yet.",
                        required_tool_name="read_file",
                    )
                    calls = _investigation_tool_calls(responses)
                    investigation_errors = _investigation_errors(calls)
                if investigation_errors:
                    validation_errors.extend(investigation_errors)
                    investigation = ProposalInvestigation(
                        sample_index=sample_index,
                        status="failed",
                        investigation_attempts=investigation_attempts,
                        synthesis_attempts=0,
                        tool_calls=calls,
                        validation_errors=list(dict.fromkeys(validation_errors)),
                    )
                    return ProposalModelResult(
                        proposal=None,
                        investigation=investigation,
                        usage_details=tuple(usage_details),
                        error="; ".join(investigation_errors),
                    )

                proposal: ClarificationProposal | None = None
                proposal_errors: list[str] = []
                for synthesis_attempts in (1, 2):
                    message = (
                        "Investigation is complete. Synthesize the final proposal now. Return schema JSON only. "
                        "A requirement_complete verdict is valid only when the investigation found no useful "
                        "repository-specific supplement."
                        if synthesis_attempts == 1
                        else "The previous proposal was rejected: "
                        + "; ".join(proposal_errors)
                        + ". Correct those semantic defects using the repository evidence already inspected and "
                        "return the complete proposal schema again."
                    )
                    response = await run_stage(
                        message,
                        response_format=ClarificationProposal,
                        tool_choice_none=True,
                    )
                    try:
                        value = response.value
                    except Exception as exc:
                        proposal_errors = [f"{type(exc).__name__}: {exc}"[:600]]
                        continue
                    if not isinstance(value, ClarificationProposal):
                        proposal_errors = ["proposal synthesis returned no ClarificationProposal"]
                        continue
                    proposal = value
                    proposal_errors = _proposal_errors(proposal, calls)
                    if not proposal_errors:
                        break
                if proposal is None or proposal_errors:
                    validation_errors.extend(proposal_errors)
                    investigation = ProposalInvestigation(
                        sample_index=sample_index,
                        status="failed",
                        investigation_attempts=investigation_attempts,
                        synthesis_attempts=synthesis_attempts,
                        tool_calls=calls,
                        validation_errors=list(dict.fromkeys(validation_errors)),
                    )
                    return ProposalModelResult(
                        proposal=None,
                        investigation=investigation,
                        usage_details=tuple(usage_details),
                        error="; ".join(proposal_errors),
                    )
                investigation = ProposalInvestigation(
                    sample_index=sample_index,
                    status="completed",
                    investigation_attempts=investigation_attempts,
                    synthesis_attempts=synthesis_attempts,
                    tool_calls=calls,
                    validation_errors=list(dict.fromkeys(validation_errors)),
                )
                return ProposalModelResult(
                    proposal=proposal,
                    investigation=investigation,
                    usage_details=tuple(usage_details),
                )
            except Exception as exc:
                calls = _investigation_tool_calls(responses)
                detail = f"{type(exc).__name__}: {exc}"[:600]
                investigation = ProposalInvestigation(
                    sample_index=sample_index,
                    status="failed",
                    investigation_attempts=investigation_attempts,
                    synthesis_attempts=synthesis_attempts,
                    tool_calls=calls,
                    validation_errors=[*validation_errors, detail],
                )
                return ProposalModelResult(
                    proposal=None,
                    investigation=investigation,
                    usage_details=tuple(usage_details),
                    error=detail,
                )
        finally:
            await agent.__aexit__(None, None, None)

    async def select(self, prompt: str) -> tuple[ClarificationSelectorDecision, dict[str, object]]:
        return await self._run(
            prompt,
            response_format=ClarificationSelectorDecision,
            instructions=selector_instructions(),
            route_kind="requirement-clarification-selector",
            route_part="1",
        )

    async def generate_pact_goal_contract(self, prompt: str) -> tuple[PactGoalContract, dict[str, object]]:
        return await self._run(
            prompt,
            response_format=PactGoalContract,
            instructions=pact_goal_contract_instructions(),
            route_kind="requirement-clarification-pact-goal-contract",
            route_part="1",
        )

    async def generate_pact_initial_plan(self, prompt: str) -> tuple[PactInitialPlan, dict[str, object]]:
        return await self._run(
            prompt,
            response_format=PactInitialPlan,
            instructions=pact_initial_plan_instructions(),
            route_kind="requirement-clarification-pact-initial-plan",
            route_part="1",
        )

    async def _run(
        self,
        prompt: str,
        *,
        response_format: type[ResponseT],
        instructions: str,
        route_kind: str,
        route_part: str,
        required_tool_name: str | None = None,
    ) -> tuple[ResponseT, dict[str, object]]:
        route_session_id = derive_llm_route_session_id(
            self._session_id,
            route_kind=route_kind,
            route_parts=(self._snapshot.snapshot_id, route_part),
            model_profile=self._profile,
        )
        agent = self._create_agent(route_session_id=route_session_id, instructions=instructions)
        await agent.__aenter__()
        try:
            response = await agent.run(
                prompt,
                stream=False,
                options=_stateless_options(
                    self._profile,
                    response_format,
                    required_tool_name=required_tool_name,
                ),
            )
            value = response.value
            if not isinstance(value, response_format):
                raise ValueError(f"clarification side call returned no {response_format.__name__}")
            usage = dict(response.usage_details or {})
            if usage and self._report_usage is not None:
                self._report_usage(usage)
            return value, usage
        finally:
            await agent.__aexit__(None, None, None)

    def _create_agent(self, *, route_session_id: str | None, instructions: str) -> Agent:
        client = create_client(
            self._profile,
            session_id=route_session_id,
            parent_session_id=self._session_id,
            session_dir=self._session_dir,
        )
        # Clarification side agents are intentionally bounded. ``create_client``
        # returns the ToolLoopLayer at the top of the client stack; these knobs
        # apply to every run made by this short-lived agent. Structured synthesis
        # disables tools, while investigation gets enough calls for a focused
        # repository trace without expanding into an autonomous coding session.
        client.max_iterations = _INVESTIGATION_MAX_ITERATIONS
        client.max_function_calls = _INVESTIGATION_MAX_FUNCTION_CALLS
        runtime = SessionEnvironment.capture(
            session_id=route_session_id,
            workspace=self._snapshot.clarification_workspace(),
        )
        registry = ToolRegistry(vision_enabled=False)
        tools = registry.load_builtins(
            ["filesystem.read", "search"],
            runtime=runtime,
            session_id=route_session_id,
            session_dir=self._session_dir,
        )
        roots = "\n".join(
            f"- {root.view_root} ({'primary' if root.is_primary else 'additional'})" for root in self._snapshot.roots
        )
        references = "\n".join(
            f"- {reference.view_path}: {reference.entry.size} bytes"
            + (f" ({reference.entry.metadata_reason})" if reference.entry.metadata_reason else "")
            for reference in self._snapshot.references
        )
        return Agent(
            client=client,
            name="ChrysRequirementClarifier",
            instructions=(
                instructions
                + "\n\nFrozen workspace roots:\n"
                + roots
                + ("\nFrozen explicit reference files:\n" + references if references else "")
                + "\nUse only read_file, view_image, grep, and glob. Never address a path outside these roots."
            ),
            tools=tools,
        )
