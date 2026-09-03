# Copyright (c) 2026 Chrys. All rights reserved.

"""Versioned prompts for repository-grounded requirement clarification."""

from __future__ import annotations

import json
from typing import cast

from chrys.service.requirement_clarification.types import (
    ClarificationProposal,
    ClarificationSelection,
    ClarificationSelectorDecision,
    InvestigationFocus,
    LegacyV1ClarificationProposal,
    PactGoalContract,
)

STRATEGY_VERSION = "chrys-requirement-clarification-v10"
LEGACY_V1_STRATEGY_VERSION = "chrys-requirement-clarification-v1"
PROPOSAL_COUNT = 3
MIN_VALID_PROPOSALS = 1
SELECTOR_COUNT = 1
MAX_FINAL_POINTS = 5
MAX_FINAL_CHARS = 1400

_PROPOSAL_INSTRUCTIONS = """You are an internal Chrys requirement-clarification agent.
Work read-only in the frozen pre-implementation repository snapshot. Investigation and final
structured synthesis are separate turns; follow the current turn's requested output shape exactly.

Allowed evidence is limited to the authoritative requirement, bounded prior user/assistant text,
the frozen current repository, and commits reachable from the frozen HEAD. Never inspect parent
directories, later workspace state, the initial implementation, remote refs, generated evaluation
data, the web, or network services. Do not edit files or run code or tests.

Find only repository-specific guidance whose addition is likely to help a fresh repair agent. Do
not invent a new observable requirement: the user's exact text owns the contract. Prefer concrete
ownership, entry points, existing abstractions, caller/consumer paths, state lifecycle, validation,
and compatibility connections that a surface-only implementation could miss. Omit generic advice.
Inspect relevant repository files with the available read/search tools before final synthesis.

Exact identifiers require direct requirement/repository/ancestor support. Keep each statement concise
and actionable. Final statements must be declarative or imperative findings, never unresolved questions
or "need to confirm/determine/verify" notes. Put residual uncertainty in the private risk field. Always
return a concrete investigation rationale and direct evidence. Return 1-6
private guidance points with verdict clarification_needed when a repository-specific supplement
could help; use verdict requirement_complete and an empty guidance_points list when the original
requirement already covers the relevant behavior completely.
An empty list never replaces investigation. When the strongest repository-grounded lead is uncertain,
preserve it with calibrated confidence and risk so the selector can reject it. Evidence,
decision_impact, confidence, and risk are private selection metadata.

During final synthesis, enumerate the distinct high-value gaps found within your assigned focus;
do not collapse unrelated ownership, lifecycle, transport, or compatibility seams into one vague
theme. Prefer 2-4 candidates when the evidence supports them, but never manufacture candidates to
meet a quota. Each statement must name the concrete repository surface or existing abstraction it
connects and the implementation consequence. Avoid slogans such as "focus on", "handle correctly",
or "ensure support" without that concrete connection.

The coverage array is a private investigation certificate. Return exactly one primary finding for
the first coverage target and one adjacent finding for the second. Mark a target not_found only after
searching and inspecting the most relevant repository surface; its evidence must still cite inspected
repository files. Coverage findings prove investigation breadth but are never rendered into the public
clarification delta.
"""

_SELECTOR_INSTRUCTIONS = """You are the internal Chrys requirement-clarification selector.
Work read-only in the same frozen pre-implementation snapshot and return schema JSON only.

Select a very small repository-grounded supplement for a fresh repair agent. The user's exact
requirement remains highest authority. Reject new requirements, paraphrases, generic coding advice,
speculative exact names, optional nice-to-haves, documentation work, and facts about unchanged
behavior. Agreement based on the same mismatched symbol is still one bad source.

Judge whether each candidate adds a concrete repository mapping, not whether the user's observable
requirement is incomplete. A candidate may be valuable precisely because the requirement states the
behavior but does not identify the existing owner, data representation, dispatch lifecycle, or
compatibility seam needed to implement it. Do not reject such a mapping merely because its intended
observable result already appears in the requirement. When candidates overlap, select the strongest
repository-supported version. Reject all only when every candidate is generic, speculative,
duplicative of the requirement without a repository connection, or optional.

Return exactly one review for every candidate id in the packet, in packet order. Prefer at most five
select decisions and mark the rest reject; deterministic rendering will retain at most five. Never
omit an id, repeat an id, rewrite a candidate, merge candidates, or invent an id. Give a short private
rationale tied to the selection rules for every judgment. Candidate confidence remains owned by the
evidence-bearing proposal and is not re-estimated here. An all-reject result is valid only after
explicitly reviewing every candidate.
"""

_LEGACY_V1_PROPOSAL_INSTRUCTIONS = """You are an internal Chrys requirement-clarification agent.
Work read-only in the frozen pre-implementation repository snapshot and return schema JSON only.

Allowed evidence is limited to the authoritative requirement, bounded prior user/assistant text,
the frozen current repository, and commits reachable from the frozen HEAD. Never inspect parent
directories, later workspace state, the initial implementation, remote refs, generated evaluation
data, the web, or network services. Do not edit files or run code or tests.

Find only repository-specific guidance whose addition is likely to help a fresh repair agent. Do
not invent a new observable requirement: the user's exact text owns the contract. Prefer concrete
ownership, entry points, existing abstractions, caller/consumer paths, state lifecycle, validation,
and compatibility connections that a surface-only implementation could miss. Omit generic advice.

Exact identifiers require direct requirement/repository/ancestor support. Keep each statement
concise and actionable. Return 1-6 private candidates, or none when no high-value clarification is
supported. Evidence, decision_impact, confidence, and risk are private selection metadata.
"""

_LEGACY_V1_SELECTOR_INSTRUCTIONS = """You are the internal Chrys requirement-clarification selector.
Work read-only in the same frozen pre-implementation snapshot and return schema JSON only.

Select a very small repository-grounded supplement for a fresh repair agent. The user's exact
requirement remains highest authority. Reject new requirements, paraphrases, generic coding advice,
speculative exact names, optional nice-to-haves, documentation work, and facts about unchanged
behavior. Agreement based on the same mismatched symbol is still one bad source.

Return at most five short statements. Every statement must give a task-specific implementation
step connecting ownership/declaration or data/control flow to an integration, lifecycle,
validation, error, or compatibility consequence. Do not expose evidence, confidence labels,
selection gates, XML, or process commentary inside statement text.
"""

_PACT_GOAL_CONTRACT_INSTRUCTIONS = """You generate a PACT Runtime Goal Contract v1 as schema JSON only.
The user's requirement messages are the sole authority for completion obligations. Produce one concise outcome goal,
atomic externally observable acceptance criteria with stable descriptive ids, and only explicitly supported non-goals.
Do not add repository implementation details, file names, functions, missions, test commands, hidden grader details, or
requirements inferred only from repository conventions. Use an empty non_goals array when none are stated or clearly
bounded by the user's request. Return exactly the closed pact-runtime/goal-contract/v1 shape.
"""

_PACT_INITIAL_PLAN_INSTRUCTIONS = """You generate a PACT Runtime Initial Plan v1 as schema JSON only.
Treat the supplied Goal Contract as immutable authority. Use the frozen-repository evidence and clarification results
to create a small end-to-end mission graph. Every mission must cover at least one existing acceptance criterion, and
every acceptance criterion must be covered. Dependencies must reference mission ids and form a DAG. Put cross-mission
implementation invariants in constraints. verification_intent describes public evidence to collect, never hidden
grader commands. Do not emit runtime-owned state or extra fields. Return exactly the closed
pact-runtime/initial-plan/v1 shape.
"""

_FOCUSES = (
    (
        "ownership_extension",
        "Map repository ownership and extension seams for the requested public surfaces.",
        (
            "Locate the authoritative definition or owner.",
            "Trace at least one registration, dispatch, caller, or extension seam adjacent to that owner.",
        ),
    ),
    (
        "data_control_flow",
        "Trace the requested value or operation from declaration through transport/state to consumers.",
        (
            "Locate the source or entry point for the requested value or operation.",
            "Trace its state/transform path to an authoritative consumer or sink.",
        ),
    ),
    (
        "boundary_compatibility",
        "Find compatibility, error, boundary, and integration traps a surface-only change could miss.",
        (
            "Locate the implementation boundary or error/compatibility decision point.",
            "Inspect an adjacent integration, compatibility, or test surface that constrains it.",
        ),
    ),
)


def proposal_instructions() -> str:
    return _PROPOSAL_INSTRUCTIONS


def selector_instructions() -> str:
    return _SELECTOR_INSTRUCTIONS


def legacy_v1_proposal_instructions() -> str:
    """Return the exact proposal instructions from the historical v1 strategy."""
    return _LEGACY_V1_PROPOSAL_INSTRUCTIONS


def legacy_v1_selector_instructions() -> str:
    """Return the exact selector instructions from the historical v1 strategy."""
    return _LEGACY_V1_SELECTOR_INSTRUCTIONS


def pact_goal_contract_instructions() -> str:
    return _PACT_GOAL_CONTRACT_INSTRUCTIONS


def pact_initial_plan_instructions() -> str:
    return _PACT_INITIAL_PLAN_INSTRUCTIONS


def build_proposal_prompt(
    requirement: str,
    background: str,
    base_evidence: str,
    sample_index: int,
) -> str:
    """Build one independent proposal prompt."""
    _focus_key, focus, coverage = _FOCUSES[sample_index - 1]
    rendered_coverage = "\n".join(f"- {item}" for item in coverage)
    return (
        f"Authoritative requirement:\n{requirement}\n\n"
        f"Bounded prior conversation background (non-authoritative):\n{background or '[none]'}\n\n"
        "Deterministic frozen-repository evidence seed (incomplete; matches are leads, not proof):\n"
        f"{base_evidence}\n\n"
        f"Independent focus:\n{focus}\n\n"
        f"Coverage target before synthesis:\n{rendered_coverage}"
    )


def build_legacy_v1_proposal_prompt(
    requirement: str,
    background: str,
    base_evidence: str,
    sample_index: int,
) -> str:
    """Build the proposal prompt used by the historical v1 strategy."""
    _focus_key, focus, _coverage = _FOCUSES[sample_index - 1]
    return (
        f"Authoritative requirement:\n{requirement}\n\n"
        f"Bounded prior conversation background (non-authoritative):\n{background or '[none]'}\n\n"
        f"Deterministic frozen-repository evidence packet:\n{base_evidence}\n\n"
        f"Independent focus:\n{focus}"
    )


def proposal_focus(sample_index: int) -> InvestigationFocus:
    """Return the stable focus id for one proposal slot."""
    return cast(InvestigationFocus, _FOCUSES[sample_index - 1][0])


def proposal_required_coverage(sample_index: int) -> tuple[str, ...]:
    """Return the focus-specific investigation obligations."""
    return _FOCUSES[sample_index - 1][2]


def build_selector_prompt(
    requirement: str,
    background: str,
    base_evidence: str,
    proposals: list[ClarificationProposal],
) -> str:
    """Build the single selector prompt from three private proposals."""
    candidates = []
    for proposal_index, proposal in enumerate(proposals, start=1):
        for guidance_index, point in enumerate(proposal.guidance_points, start=1):
            candidates.append(
                {
                    "candidate_id": f"p{proposal_index}-g{guidance_index}",
                    "category": point.category,
                    "statement": point.statement,
                    "confidence": point.confidence,
                    "basis": point.basis,
                    "decision_impact": point.decision_impact,
                    "evidence": [anchor.model_dump(mode="json") for anchor in point.evidence],
                    "risk": point.risk,
                }
            )
    encoded = json.dumps(candidates, ensure_ascii=False, indent=2)
    return (
        f"Authoritative requirement:\n{requirement}\n\n"
        f"Bounded prior conversation background (non-authoritative):\n{background or '[none]'}\n\n"
        f"Deterministic frozen-repository evidence packet:\n{base_evidence}\n\n"
        f"Candidate packet with closed ids:\n{encoded or '[]'}"
    )


def build_legacy_v1_selector_prompt(
    requirement: str,
    background: str,
    base_evidence: str,
    proposals: list[LegacyV1ClarificationProposal],
) -> str:
    """Build the exact selector packet used by the historical v1 strategy."""
    encoded = "\n\n".join(
        f"Proposal {index}:\n{proposal.model_dump_json(indent=2)}" for index, proposal in enumerate(proposals, start=1)
    )
    return (
        f"Authoritative requirement:\n{requirement}\n\n"
        f"Bounded prior conversation background (non-authoritative):\n{background or '[none]'}\n\n"
        f"Deterministic frozen-repository evidence packet:\n{base_evidence}\n\n"
        f"Candidate packets:\n{encoded}"
    )


def build_pact_goal_contract_prompt(requirement: str, background: str) -> str:
    """Build the user-authority-only Goal Contract prompt."""
    return (
        f"Authoritative requirement messages:\n{requirement}\n\n"
        f"Bounded prior conversation background (non-authoritative):\n{background or '[none]'}"
    )


def build_pact_initial_plan_prompt(
    goal_contract: PactGoalContract,
    base_evidence: str,
    proposals: list[ClarificationProposal | LegacyV1ClarificationProposal],
    selection: ClarificationSelection,
) -> str:
    """Build the repository-grounded Initial Plan prompt."""
    encoded_proposals = "\n\n".join(
        f"Proposal {index}:\n{proposal.model_dump_json(indent=2)}" for index, proposal in enumerate(proposals, start=1)
    )
    return (
        "Validated Goal Contract:\n"
        + goal_contract.model_dump_json(indent=2, by_alias=True)
        + f"\n\nDeterministic frozen-repository evidence packet:\n{base_evidence}\n\n"
        + f"Private clarification proposals:\n{encoded_proposals}\n\n"
        + "Cleaned clarification selection:\n"
        + selection.model_dump_json(indent=2)
    )


def proposal_schema_text() -> str:
    """Return the proposal schema for logs and non-native structured-output adapters."""
    return json.dumps(ClarificationProposal.model_json_schema(), ensure_ascii=False, sort_keys=True)


def selector_schema_text() -> str:
    """Return the closed candidate-reference selector schema."""
    return json.dumps(ClarificationSelectorDecision.model_json_schema(), ensure_ascii=False, sort_keys=True)


def legacy_v1_proposal_schema_text() -> str:
    """Return the exact historical v1 proposal schema."""
    return json.dumps(LegacyV1ClarificationProposal.model_json_schema(), ensure_ascii=False, sort_keys=True)
