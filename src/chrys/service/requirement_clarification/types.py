# Copyright (c) 2026 Chrys. All rights reserved.

"""Typed contracts for the requirement-clarification workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

ClarificationCategory = Literal[
    "interface",
    "behavior",
    "data_format",
    "error",
    "integration",
    "compatibility",
    "testing",
]
ClarificationBasis = Literal[
    "original_requirement",
    "current_repo",
    "exact_ancestor",
    "related_ancestor",
    "ecosystem_prior",
    "speculative",
]
ClarificationContractCell = Literal[
    "owner_signature",
    "values_defaults",
    "transport_integration",
    "branch_matrix",
    "error_result",
    "compatibility_scope",
]
ClarificationVerdict = Literal["clarification_needed", "requirement_complete"]
ClarificationStatus = Literal["completed", "degraded"]
ClarificationEmptyReason = Literal[
    "requirement_complete",
    "selector_rejected",
    "insufficient_valid_proposals",
    "selector_failed",
    "clarification_failed",
]


class RequirementWorkflowPhase(StrEnum):
    """Durable phases of one baseline/clarification/repair turn."""

    SNAPSHOT = "snapshot"
    INITIAL_IMPLEMENTATION = "initial_implementation"
    CLARIFICATION = "clarification"
    REPAIR = "repair"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    INTERRUPTED = "interrupted"
    CONFLICTED = "conflicted"


@dataclass(frozen=True, slots=True)
class RequirementRevision:
    """Verbatim user-authored requirement messages in authority order."""

    number: int
    messages: tuple[str, ...]

    @property
    def rendered(self) -> str:
        if len(self.messages) == 1:
            return self.messages[0]
        sections = [f"Requirement message {index}:\n{text}" for index, text in enumerate(self.messages, start=1)]
        return "\n\n".join(sections)

    def append(self, text: str) -> RequirementRevision:
        return RequirementRevision(number=self.number + 1, messages=(*self.messages, text))


class EvidenceAnchor(BaseModel):
    """Private evidence attached to one proposal candidate."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "original_requirement",
        "current_repo",
        "exact_ancestor",
        "related_ancestor",
        "ecosystem_prior",
    ]
    anchor: NonBlankText = Field(max_length=600)


class ProposalGuidancePoint(BaseModel):
    """One evidence-bearing private proposal packet."""

    model_config = ConfigDict(extra="forbid")

    category: ClarificationCategory
    statement: NonBlankText = Field(max_length=1500)
    confidence: float = Field(ge=0.0, le=1.0)
    basis: ClarificationBasis
    contract_cell: ClarificationContractCell
    decision_impact: NonBlankText = Field(max_length=800)
    evidence: list[EvidenceAnchor] = Field(min_length=1, max_length=4)
    risk: NonBlankText = Field(max_length=800)


class ClarificationProposal(BaseModel):
    """Structured response from one independent proposal pass."""

    model_config = ConfigDict(extra="forbid")

    verdict: ClarificationVerdict
    rationale: NonBlankText = Field(max_length=1200)
    evidence: list[EvidenceAnchor] = Field(min_length=1, max_length=4)
    guidance_points: list[ProposalGuidancePoint] = Field(max_length=6)

    @model_validator(mode="after")
    def _validate_verdict_matches_guidance(self) -> ClarificationProposal:
        if self.verdict == "clarification_needed" and not self.guidance_points:
            raise ValueError("clarification_needed requires at least one guidance point")
        if self.verdict == "requirement_complete" and self.guidance_points:
            raise ValueError("requirement_complete requires an empty guidance_points list")
        return self


class InvestigationToolCall(BaseModel):
    """Auditable read-only tool activity from one proposal investigation."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["grep", "glob", "read_file", "view_image"]
    path: str = Field(default="", max_length=1200)
    successful: bool
    result_chars: int = Field(ge=0)


class ProposalInvestigation(BaseModel):
    """Private diagnostics proving that proposal synthesis followed investigation."""

    model_config = ConfigDict(extra="forbid")

    sample_index: int = Field(ge=1, le=3)
    status: Literal["completed", "failed"]
    investigation_attempts: int = Field(ge=1, le=2)
    synthesis_attempts: int = Field(ge=0, le=2)
    tool_calls: list[InvestigationToolCall] = Field(default_factory=list, max_length=100)
    validation_errors: list[str] = Field(default_factory=list, max_length=20)


@dataclass(frozen=True, slots=True)
class ProposalModelResult:
    """One proposal side-agent result, including auditable investigation state."""

    proposal: ClarificationProposal | None
    investigation: ProposalInvestigation
    usage_details: tuple[dict[str, object], ...] = ()
    error: str = ""


class SelectedGuidancePoint(BaseModel):
    """Compact selector output safe to render for the repair agent."""

    model_config = ConfigDict(extra="forbid")

    category: ClarificationCategory
    statement: NonBlankText = Field(max_length=1500)
    confidence: float = Field(ge=0.0, le=1.0)
    basis: ClarificationBasis


class ClarificationSelection(BaseModel):
    """Structured response from the single selector pass."""

    model_config = ConfigDict(extra="forbid")

    guidance_points: list[SelectedGuidancePoint] = Field(default_factory=list, max_length=5)


class SelectorCandidateReview(BaseModel):
    """One auditable selector judgment over a closed proposal candidate id."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(pattern=r"^p[1-3]-g[1-6]$")
    decision: Literal["select", "reject"]
    rationale: NonBlankText = Field(max_length=800)


class ClarificationSelectorDecision(BaseModel):
    """Closed selector output with one private judgment per candidate id."""

    model_config = ConfigDict(extra="forbid")

    reviews: list[SelectorCandidateReview] = Field(min_length=1, max_length=18)


class PactAcceptanceCriterion(BaseModel):
    """One stable, user-authoritative PACT completion obligation."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$", max_length=120)
    text: str = Field(min_length=1, max_length=2000)


class PactGoalContract(BaseModel):
    """Closed PACT Runtime Goal Contract v1 payload."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_id: Literal["pact-runtime/goal-contract/v1"] = Field(alias="schema")
    goal: str = Field(min_length=1, max_length=4000)
    acceptance_criteria: list[PactAcceptanceCriterion] = Field(min_length=1, max_length=100)
    non_goals: list[str] = Field(max_length=100)


class PactMission(BaseModel):
    """One mission in a closed PACT Runtime Initial Plan v1 payload."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$", max_length=120)
    objective: str = Field(min_length=1, max_length=4000)
    target_ac_ids: list[str] = Field(min_length=1, max_length=100)
    dependencies: list[str] = Field(max_length=100)
    verification_intent: str = Field(min_length=1, max_length=4000)


class PactInitialPlan(BaseModel):
    """Closed PACT Runtime Initial Plan v1 payload."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_id: Literal["pact-runtime/initial-plan/v1"] = Field(alias="schema")
    constraints: list[str] = Field(max_length=100)
    missions: list[PactMission] = Field(min_length=1, max_length=100)


@dataclass(frozen=True, slots=True)
class PactRuntimeInput:
    """Validated pair written as the two canonical PACT input files."""

    goal_contract: PactGoalContract
    initial_plan: PactInitialPlan


@dataclass(frozen=True, slots=True)
class ClarificationResult:
    """Validated public delta plus private evidence-bearing model outputs."""

    strategy_version: str
    revision: int
    delta: str
    selection: ClarificationSelection
    raw_selection: ClarificationSelectorDecision | None = None
    status: ClarificationStatus = "completed"
    empty_reason: ClarificationEmptyReason | None = None
    pact_input: PactRuntimeInput | None = None
    pact_generation_error: str = ""
    proposals: tuple[ClarificationProposal, ...] = ()
    investigations: tuple[ProposalInvestigation, ...] = ()
    elapsed_seconds: float = 0.0
    usage_details: tuple[dict[str, object], ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.delta.strip()


@dataclass(slots=True)
class WorkflowUsage:
    """Mutable side-call usage accumulator owned by one workflow."""

    details: list[dict[str, object]] = field(default_factory=list)

    def record(self, details: dict[str, object]) -> None:
        if details:
            self.details.append(dict(details))
