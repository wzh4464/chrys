# Copyright (c) 2026 Chrys. All rights reserved.

"""Typed contracts for the requirement-clarification workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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
    anchor: str = Field(min_length=1, max_length=600)


class ProposalGuidancePoint(BaseModel):
    """One evidence-bearing private proposal packet."""

    model_config = ConfigDict(extra="forbid")

    category: ClarificationCategory
    statement: str = Field(min_length=1, max_length=1500)
    confidence: float = Field(ge=0.0, le=1.0)
    basis: ClarificationBasis
    contract_cell: ClarificationContractCell
    decision_impact: str = Field(min_length=1, max_length=800)
    evidence: list[EvidenceAnchor] = Field(min_length=1, max_length=4)
    risk: str = Field(min_length=1, max_length=800)


class ClarificationProposal(BaseModel):
    """Structured response from one independent proposal pass."""

    model_config = ConfigDict(extra="forbid")

    guidance_points: list[ProposalGuidancePoint] = Field(default_factory=list, max_length=6)


class SelectedGuidancePoint(BaseModel):
    """Compact selector output safe to render for the repair agent."""

    model_config = ConfigDict(extra="forbid")

    category: ClarificationCategory
    statement: str = Field(min_length=1, max_length=1500)
    confidence: float = Field(ge=0.0, le=1.0)
    basis: ClarificationBasis


class ClarificationSelection(BaseModel):
    """Structured response from the single selector pass."""

    model_config = ConfigDict(extra="forbid")

    guidance_points: list[SelectedGuidancePoint] = Field(default_factory=list, max_length=5)


@dataclass(frozen=True, slots=True)
class ClarificationResult:
    """Validated public delta plus private evidence-bearing model outputs."""

    strategy_version: str
    revision: int
    delta: str
    selection: ClarificationSelection
    proposals: tuple[ClarificationProposal, ...] = ()
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
