# Copyright (c) 2026 Chrys. All rights reserved.

"""Fresh, read-only Chrys agents used by proposal and selector passes."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel, ValidationError

from chrys.foundation.models.session_env import SessionEnvironment
from chrys.kernel import CONVERSATION_HANDLE_KEYS, Agent
from chrys.kernel.exchanges import (
    EmptyIdPolicy,
    LiveAccessor,
    NoneIdPolicy,
    PairingPolicy,
    iter_exchanges,
    pair_results,
)
from chrys.service.llm.clients import create_client
from chrys.service.llm.route_sessions import derive_llm_route_session_id
from chrys.service.profiles.models.options import effective_chat_options
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.requirement_clarification.prompts import (
    legacy_v1_proposal_instructions,
    legacy_v1_selector_instructions,
    pact_goal_contract_instructions,
    pact_initial_plan_instructions,
    proposal_focus,
    proposal_instructions,
    proposal_required_coverage,
    selector_instructions,
)
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot
from chrys.service.requirement_clarification.tools import SnapshotReadTools
from chrys.service.requirement_clarification.types import (
    ClarificationProposal,
    ClarificationSelection,
    ClarificationSelectorDecision,
    InvestigationToolCall,
    LegacyV1ClarificationProposal,
    PactGoalContract,
    PactInitialPlan,
    ProposalInvestigation,
    ProposalModelResult,
    VerifiedEvidenceReference,
)

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class ClarificationModelRunner(Protocol):
    """Model boundary consumed by :class:`ClarificationService`."""

    async def propose(self, prompt: str, *, sample_index: int) -> ProposalModelResult: ...

    async def select(self, prompt: str) -> tuple[ClarificationSelectorDecision, dict[str, object]]: ...

    async def propose_legacy_v1(
        self, prompt: str, *, sample_index: int
    ) -> tuple[LegacyV1ClarificationProposal, dict[str, object]]: ...

    async def select_legacy_v1(self, prompt: str) -> tuple[ClarificationSelection, dict[str, object]]: ...

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


logger = logging.getLogger(__name__)

_STRUCTURED_REPLY_ATTEMPTS = 2
_STRUCTURED_REPLY_REMINDER = (
    "Your previous reply was not the required JSON object (prose, a fence, or tool-call "
    "markup instead of the object). This turn has no tools to call. Reply with exactly one "
    "JSON object matching the schema, and nothing else."
)


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
MAX_RECORDED_TOOL_CALLS = 100  # ProposalInvestigation.tool_calls max_length

_INVESTIGATION_PAIRING_POLICY = PairingPolicy(
    call_types=frozenset({"function_call"}),
    include_informational_calls=False,
    result_types=frozenset({"function_result"}),
    none_id=NoneIdPolicy.IGNORE,
    empty_id=EmptyIdPolicy.IGNORE,
    malformed_id="treat_as_none",
)


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


def _argument_text(arguments: Mapping[str, object], key: str) -> str:
    value = arguments.get(key)
    return value[:1200] if isinstance(value, str) else ""


def _line_range(arguments: Mapping[str, object]) -> tuple[int, int] | None:
    value = arguments.get("line_range")
    if (
        isinstance(value, list | tuple)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        return value[0], value[1]
    return None


def _tool_result_text(content: Any) -> str:
    result = content.result
    if isinstance(result, str):
        return result
    return str(result or "")


def _investigation_tool_calls(responses: list[Any]) -> list[InvestigationToolCall]:
    observations: list[InvestigationToolCall] = []
    accessor = LiveAccessor()
    for response in responses:
        messages = list(getattr(response, "messages", ()))
        for exchange in iter_exchanges(messages, accessor):
            pairing = pair_results(messages, exchange, accessor, _INVESTIGATION_PAIRING_POLICY)
            for assignments in pairing.truthy_assignments.values():
                for call_occurrence, result_occurrence in assignments:
                    call = messages[call_occurrence.message_index].contents[call_occurrence.content_index]
                    name = call.name
                    if name not in {"grep", "glob", "read_file", "view_image"}:
                        continue
                    arguments = _call_arguments(call)
                    text = ""
                    exception: str | None = None
                    if result_occurrence is not None:
                        result = messages[result_occurrence.message_index].contents[result_occurrence.content_index]
                        text = _tool_result_text(result)
                        exception = result.exception
                    observations.append(
                        InvestigationToolCall(
                            call_id=call.call_id,
                            name=cast("Literal['grep', 'glob', 'read_file', 'view_image']", name),
                            query=_argument_text(arguments, "pattern") if name in {"grep", "glob"} else "",
                            path=_argument_text(arguments, "path"),
                            line_range=_line_range(arguments),
                            successful=(
                                not exception and bool(text.strip()) and not text.lstrip().startswith("Error:")
                            ),
                            result_chars=len(text),
                            result_sha256=(
                                hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest() if text else ""
                            ),
                        )
                    )
    return observations


def _investigation_errors(calls: list[InvestigationToolCall]) -> list[str]:
    successful = [call for call in calls if call.successful]
    errors: list[str] = []
    if not any(call.name == "grep" for call in successful):
        errors.append("investigation has no successful grep")
    elif not any(call.name == "grep" and call.query.strip() for call in successful):
        errors.append("investigation has no recorded search query")
    if not any(call.name == "read_file" for call in successful):
        errors.append("investigation has no successful read_file")
    return errors


def _normalized_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.rstrip("/")


def _path_aliases(value: str) -> tuple[str, ...]:
    normalized = _normalized_path(value)
    if not normalized:
        return ()
    parts = [part for part in normalized.split("/") if part]
    aliases = [normalized]
    aliases.extend("/".join(parts[index:]) for index in range(1, max(1, len(parts) - 1)))
    if len(parts) == 1:
        aliases.append(parts[0])
    return tuple(dict.fromkeys(aliases))


def _anchor_read_paths(anchor: str, read_paths: tuple[str, ...]) -> tuple[str, ...]:
    normalized_anchor = anchor.replace("\\", "/")
    matched = tuple(
        path for path in read_paths if any(alias and alias in normalized_anchor for alias in _path_aliases(path))
    )
    if matched:
        return matched
    # A file at the repository root has no directory component to cite, and the
    # suffix aliases above never go down to a single segment -- so `pyproject.toml`
    # was uncitable no matter how carefully it had been read. Fall back to the
    # bare name, but only for an anchor that names no directory at all, and only
    # when exactly one inspected path carries it: an anchor that DID name a
    # directory and still missed is a real miss, and a name shared by several
    # inspected files is the ambiguity the suffix rule exists to prevent.
    cited = normalized_anchor.split(":", 1)[0].strip()
    if not cited or "/" in cited:
        return ()
    by_name = tuple(path for path in read_paths if path.rsplit("/", 1)[-1] == cited)
    return by_name if len(by_name) == 1 else ()


def _proposal_current_repo_anchors(proposal: ClarificationProposal) -> list[str]:
    anchors = [*proposal.evidence]
    anchors.extend(anchor for finding in proposal.coverage for anchor in finding.evidence)
    anchors.extend(anchor for point in proposal.guidance_points for anchor in point.evidence)
    return list(dict.fromkeys(anchor.anchor for anchor in anchors if anchor.kind == "current_repo"))


def _anchor_line_range(anchor: str, path: str) -> tuple[int | None, int | None]:
    normalized_anchor = anchor.replace("\\", "/")
    for alias in sorted(_path_aliases(path), key=len, reverse=True):
        match = re.search(rf"{re.escape(alias)}:(\d+)(?:-(\d+))?", normalized_anchor)
        if match is None:
            continue
        start = int(match.group(1))
        return start, int(match.group(2)) if match.group(2) else start
    return None, None


def _verified_evidence(
    proposal: ClarificationProposal,
    calls: list[InvestigationToolCall],
) -> list[VerifiedEvidenceReference]:
    successful_reads = [
        call
        for call in calls
        if call.successful and call.name == "read_file" and call.path and call.call_id and call.result_sha256
    ]
    verified: list[VerifiedEvidenceReference] = []
    seen: set[tuple[str, str]] = set()
    for anchor in _proposal_current_repo_anchors(proposal):
        matched = next(
            (call for call in successful_reads if _anchor_read_paths(anchor, (_normalized_path(call.path),))),
            None,
        )
        if matched is None:
            continue
        path = _normalized_path(matched.path)
        key = path, anchor
        if key in seen:
            continue
        seen.add(key)
        line_start, line_end = _anchor_line_range(anchor, path)
        verified.append(
            VerifiedEvidenceReference(
                path=path,
                line_start=line_start,
                line_end=line_end,
                claim=anchor,
                tool_call_id=matched.call_id,
                result_sha256=matched.result_sha256,
            )
        )
    return verified


def _successful_search_queries(calls: list[InvestigationToolCall]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            call.query.strip()
            for call in calls
            if call.successful and call.name in {"grep", "glob"} and call.query.strip()
        )
    )


def _successful_read_paths(calls: list[InvestigationToolCall]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            _normalized_path(call.path)
            for call in calls
            if call.successful and call.name == "read_file" and _normalized_path(call.path)
        )
    )


def _proposal_semantic_errors(proposal: ClarificationProposal) -> list[str]:
    errors: list[str] = []
    text_fields = [proposal.rationale]
    text_fields.extend(anchor.anchor for anchor in proposal.evidence)
    for finding in proposal.coverage:
        text_fields.append(finding.summary)
        text_fields.extend(anchor.anchor for anchor in finding.evidence)
    for point in proposal.guidance_points:
        text_fields.extend((point.statement, point.decision_impact, point.risk))
        text_fields.extend(anchor.anchor for anchor in point.evidence)
    if any(_contains_placeholder(value) for value in text_fields):
        errors.append("proposal contains placeholder or unfinished investigation text")
    return errors


def _proposal_coverage_errors(
    proposal: ClarificationProposal,
    calls: list[InvestigationToolCall],
) -> list[str]:
    errors: list[str] = []
    read_paths = _successful_read_paths(calls)
    search_queries = _successful_search_queries(calls)
    all_anchors = [
        *proposal.evidence,
        *(anchor for finding in proposal.coverage for anchor in finding.evidence),
        *(anchor for point in proposal.guidance_points for anchor in point.evidence),
    ]
    current_repo_anchors = [anchor for anchor in all_anchors if anchor.kind == "current_repo"]
    if not current_repo_anchors:
        errors.append("proposal has no current-repository evidence")
    uncited = [anchor.anchor for anchor in current_repo_anchors if not _anchor_read_paths(anchor.anchor, read_paths)]
    if uncited:
        # Name them: the rejected proposal is not persisted, so without the
        # offending anchors the artifact records that a gate fired but not
        # what tripped it -- and a bare file name looks identical to an
        # invented one in the record.
        errors.append(
            "current-repository evidence cites a file not inspected with read_file: "
            + ", ".join(sorted(dict.fromkeys(uncited))[:5])
        )
    for point in proposal.guidance_points:
        if not any(anchor.kind in {"current_repo", "exact_ancestor"} for anchor in point.evidence):
            errors.append("guidance point has no current-repository or exact-ancestor evidence")
            break
    errors.extend(
        f"{finding.target} coverage has no current-repository evidence"
        for finding in proposal.coverage
        if not any(anchor.kind == "current_repo" for anchor in finding.evidence)
    )
    if proposal.verdict == "requirement_complete":
        if len(read_paths) < 2 and len(search_queries) < 2:
            errors.append("requirement_complete lacks a second inspected surface or targeted search")
        cited_paths = {
            path
            for anchor in proposal.evidence
            if anchor.kind == "current_repo"
            for path in _anchor_read_paths(anchor.anchor, read_paths)
        }
        if len(read_paths) >= 2 and len(cited_paths) < 2:
            errors.append("requirement_complete does not cite two inspected repository surfaces")
    return list(dict.fromkeys(errors))


def _proposal_errors(proposal: ClarificationProposal, calls: list[InvestigationToolCall]) -> list[str]:
    return [*_proposal_semantic_errors(proposal), *_proposal_coverage_errors(proposal, calls)]


def _required_tool_for_retry(errors: list[str]) -> str:
    if any("grep" in error or "search query" in error for error in errors):
        return "grep"
    if any("read_file" in error for error in errors):
        return "read_file"
    return "grep"


def _usage_total(usage_details: list[dict[str, object]], *keys: str) -> int:
    total = 0
    for details in usage_details:
        for key in keys:
            value = details.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                total += max(0, value)
                break
    return total


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
        investigation_attempts = 0
        synthesis_attempts = 0
        focus = proposal_focus(sample_index)
        required_coverage = list(proposal_required_coverage(sample_index))

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

        def investigation_record(
            *,
            status: Literal["completed", "failed"],
            calls: list[InvestigationToolCall],
            coverage_errors: list[str],
            proposal: ClarificationProposal | None = None,
        ) -> ProposalInvestigation:
            return ProposalInvestigation(
                sample_index=sample_index,
                focus=focus,
                status=status,
                investigation_attempts=max(1, investigation_attempts),
                synthesis_attempts=synthesis_attempts,
                coverage_status="sufficient" if not coverage_errors else "insufficient",
                required_coverage=required_coverage,
                search_queries=list(_successful_search_queries(calls)),
                inspected_paths=list(_successful_read_paths(calls)),
                coverage_errors=list(dict.fromkeys(coverage_errors))[:20],
                input_token_count=_usage_total(usage_details, "input_token_count", "input_tokens"),
                output_token_count=_usage_total(usage_details, "output_token_count", "output_tokens"),
                total_token_count=_usage_total(usage_details, "total_token_count", "total_tokens"),
                # The record keeps the tail of a long trace; the schema caps it at
                # 100 and a 117-call investigation on a real repository was thrown
                # away whole for exceeding that, coverage and evidence included.
                tool_calls=calls[-MAX_RECORDED_TOOL_CALLS:],
                verified_evidence=_verified_evidence(proposal, calls) if proposal is not None else [],
                validation_errors=list(dict.fromkeys(validation_errors))[:20],
            )

        await agent.__aenter__()
        try:
            try:
                pending_investigation_feedback: list[str] = []
                calls: list[InvestigationToolCall] = []
                for investigation_attempts in (1, 2):
                    if investigation_attempts == 1:
                        investigation_message = (
                            "Investigate the frozen repository for this focus. The evidence seed is incomplete and "
                            "cannot justify a final verdict by itself. Start with the required search, then keep using "
                            "read-only tools until both focus-specific coverage targets are addressed. Inspect concrete "
                            "owners and an adjacent flow, registration, compatibility, or test surface. Return concise "
                            "investigation notes only; do not emit proposal JSON yet.\n\n" + prompt
                        )
                        required_tool_name = "grep"
                    else:
                        investigation_message = (
                            "The controller rejected the investigation as insufficient: "
                            + "; ".join(pending_investigation_feedback)
                            + ". Continue in the same frozen repository session. Inspect the missing adjacent surface "
                            "and collect evidence that directly addresses both focus-specific coverage targets. Do not "
                            "emit proposal JSON yet."
                        )
                        required_tool_name = _required_tool_for_retry(pending_investigation_feedback)
                    await run_stage(
                        investigation_message,
                        required_tool_name=required_tool_name,
                    )
                    calls = _investigation_tool_calls(responses)
                    investigation_errors = _investigation_errors(calls)
                    if investigation_errors:
                        validation_errors.extend(investigation_errors)
                        pending_investigation_feedback = investigation_errors
                        if investigation_attempts == 1:
                            continue
                        investigation = investigation_record(
                            status="failed",
                            calls=calls,
                            coverage_errors=investigation_errors,
                        )
                        return ProposalModelResult(
                            proposal=None,
                            investigation=investigation,
                            usage_details=tuple(usage_details),
                            error="; ".join(investigation_errors),
                        )

                    proposal: ClarificationProposal | None = None
                    proposal_errors: list[str] = []
                    coverage_errors: list[str] = []
                    for synthesis_pass in (1, 2):
                        synthesis_attempts += 1
                        message = (
                            "Investigation is complete. Synthesize the final proposal now from inspected evidence. "
                            "Return schema JSON only. Every current_repo anchor must cite a file actually read, as a "
                            "repository-relative path with at least one directory component (e.g. "
                            "'src/parser.py:12-20'); a bare file name is rejected. A "
                            "requirement_complete verdict must summarize coverage and is valid only after checking a "
                            "second relevant surface or running a second targeted search."
                            if synthesis_pass == 1
                            else "The previous proposal was rejected: "
                            + "; ".join(proposal_errors)
                            + ". Correct only those synthesis defects using repository evidence already inspected and "
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
                        coverage_errors = _proposal_coverage_errors(value, calls)
                        if coverage_errors:
                            validation_errors.extend(coverage_errors)
                            pending_investigation_feedback = coverage_errors
                            break
                        proposal = value
                        proposal_errors = _proposal_semantic_errors(proposal)
                        if not proposal_errors:
                            investigation = investigation_record(
                                status="completed",
                                calls=calls,
                                coverage_errors=[],
                                proposal=proposal,
                            )
                            return ProposalModelResult(
                                proposal=proposal,
                                investigation=investigation,
                                usage_details=tuple(usage_details),
                            )
                    if coverage_errors:
                        if investigation_attempts == 1:
                            continue
                        investigation = investigation_record(
                            status="failed",
                            calls=calls,
                            coverage_errors=coverage_errors,
                        )
                        return ProposalModelResult(
                            proposal=None,
                            investigation=investigation,
                            usage_details=tuple(usage_details),
                            error="; ".join(coverage_errors),
                        )
                    if proposal_errors:
                        validation_errors.extend(proposal_errors)
                        investigation = investigation_record(
                            status="failed",
                            calls=calls,
                            coverage_errors=[],
                        )
                        return ProposalModelResult(
                            proposal=None,
                            investigation=investigation,
                            usage_details=tuple(usage_details),
                            error="; ".join(proposal_errors),
                        )

                investigation_errors = pending_investigation_feedback or ["investigation did not reach synthesis"]
                validation_errors.extend(investigation_errors)
                investigation = investigation_record(
                    status="failed",
                    calls=calls,
                    coverage_errors=investigation_errors,
                )
                return ProposalModelResult(
                    proposal=None,
                    investigation=investigation,
                    usage_details=tuple(usage_details),
                    error="; ".join(investigation_errors),
                )
            except Exception as exc:
                calls = _investigation_tool_calls(responses)
                detail = f"{type(exc).__name__}: {exc}"[:600]
                validation_errors.append(detail)
                coverage_errors = _investigation_errors(calls)
                investigation = investigation_record(
                    status="failed",
                    calls=calls,
                    coverage_errors=coverage_errors,
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

    async def propose_legacy_v1(
        self,
        prompt: str,
        *,
        sample_index: int,
    ) -> tuple[LegacyV1ClarificationProposal, dict[str, object]]:
        """Run the historical v1 single-turn proposer without forced tools or a shared session."""
        return await self._run(
            prompt,
            response_format=LegacyV1ClarificationProposal,
            instructions=legacy_v1_proposal_instructions(),
            route_kind="requirement-clarification-proposal",
            route_part=str(sample_index),
            bounded=False,
        )

    async def select_legacy_v1(self, prompt: str) -> tuple[ClarificationSelection, dict[str, object]]:
        """Run the historical v1 selector over complete proposal packets."""
        return await self._run(
            prompt,
            response_format=ClarificationSelection,
            instructions=legacy_v1_selector_instructions(),
            route_kind="requirement-clarification-selector",
            route_part="1",
            bounded=False,
        )

    async def generate_pact_goal_contract(self, prompt: str) -> tuple[PactGoalContract, dict[str, object]]:
        return await self._run(
            prompt,
            response_format=PactGoalContract,
            instructions=pact_goal_contract_instructions(),
            route_kind="requirement-clarification-pact-goal-contract",
            route_part="1",
            tool_choice_none=True,
        )

    async def generate_pact_initial_plan(self, prompt: str) -> tuple[PactInitialPlan, dict[str, object]]:
        return await self._run(
            prompt,
            response_format=PactInitialPlan,
            instructions=pact_initial_plan_instructions(),
            route_kind="requirement-clarification-pact-initial-plan",
            route_part="1",
            tool_choice_none=True,
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
        bounded: bool = True,
        tool_choice_none: bool = False,
    ) -> tuple[ResponseT, dict[str, object]]:
        route_session_id = derive_llm_route_session_id(
            self._session_id,
            route_kind=route_kind,
            route_parts=(self._snapshot.snapshot_id, route_part),
            model_profile=self._profile,
        )
        agent = self._create_agent(
            route_session_id=route_session_id,
            instructions=instructions,
            bounded=bounded,
        )
        await agent.__aenter__()
        try:
            options = _stateless_options(self._profile, response_format, required_tool_name=required_tool_name)
            if tool_choice_none:
                # Pure synthesis: the prompt carries every input. Telling the provider
                # so keeps a model from "calling" a tool as raw markup in its text
                # (DeepSeek's <|DSML|tool_calls> blocks arrived instead of the object).
                options["tool_choice"] = "none"
            attempts = 0
            while True:
                attempts += 1
                try:
                    response = await agent.run(prompt, stream=False, options=options)
                    value = response.value
                    if not isinstance(value, response_format):
                        raise ValueError(f"clarification side call returned no {response_format.__name__}")
                except (ValidationError, ValueError) as exc:
                    # A reply that is not the object -- prose, a fenced block the adapter
                    # could not unwrap, or a provider's raw tool-call markup in a turn
                    # that has no tools -- gets exactly one more chance with the format
                    # spelled out; anything else is the caller's failure to report.
                    if attempts > _STRUCTURED_REPLY_ATTEMPTS - 1:
                        raise
                    logger.warning(
                        "Clarification side call %s returned no %s; asking once more",
                        route_kind,
                        response_format.__name__,
                    )
                    prompt = f"{prompt}\n\n{_STRUCTURED_REPLY_REMINDER}\nThe error was: {str(exc)[:600]}"
                    continue
                usage = dict(response.usage_details or {})
                if usage and self._report_usage is not None:
                    self._report_usage(usage)
                return value, usage
        finally:
            await agent.__aexit__(None, None, None)

    def _create_agent(
        self,
        *,
        route_session_id: str | None,
        instructions: str,
        bounded: bool = True,
    ) -> Agent:
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
        if bounded:
            client.max_iterations = _INVESTIGATION_MAX_ITERATIONS
            client.max_function_calls = _INVESTIGATION_MAX_FUNCTION_CALLS
        runtime = SessionEnvironment.capture(
            session_id=route_session_id or "",
            workspace=self._snapshot.clarification_workspace(),
        )
        tools = SnapshotReadTools(
            runtime,
            roots=tuple(Path(root.view_root) for root in self._snapshot.roots),
            reference_files=tuple(Path(reference.view_path) for reference in self._snapshot.references),
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
            tools=tools.tools(),
        )
