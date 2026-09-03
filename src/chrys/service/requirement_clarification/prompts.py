# Copyright (c) 2026 Chrys. All rights reserved.

"""Versioned prompts for repository-grounded requirement clarification."""

from __future__ import annotations

import json

from chrys.service.requirement_clarification.types import ClarificationProposal

STRATEGY_VERSION = "chrys-requirement-clarification-v1"
PROPOSAL_COUNT = 3
SELECTOR_COUNT = 1
MAX_FINAL_POINTS = 5
MAX_FINAL_CHARS = 1400

_PROPOSAL_INSTRUCTIONS = """You are an internal Chrys requirement-clarification agent.
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

_SELECTOR_INSTRUCTIONS = """You are the internal Chrys requirement-clarification selector.
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

_FOCUSES = (
    "Map repository ownership and extension seams for the requested public surfaces.",
    "Trace the requested value or operation from declaration through transport/state to consumers.",
    "Find compatibility, error, boundary, and integration traps a surface-only change could miss.",
)


def proposal_instructions() -> str:
    return _PROPOSAL_INSTRUCTIONS


def selector_instructions() -> str:
    return _SELECTOR_INSTRUCTIONS


def build_proposal_prompt(
    requirement: str,
    background: str,
    base_evidence: str,
    sample_index: int,
) -> str:
    """Build one independent proposal prompt."""
    focus = _FOCUSES[sample_index - 1]
    return (
        f"Authoritative requirement:\n{requirement}\n\n"
        f"Bounded prior conversation background (non-authoritative):\n{background or '[none]'}\n\n"
        f"Deterministic frozen-repository evidence packet:\n{base_evidence}\n\n"
        f"Independent focus:\n{focus}"
    )


def build_selector_prompt(
    requirement: str,
    background: str,
    base_evidence: str,
    proposals: list[ClarificationProposal],
) -> str:
    """Build the single selector prompt from three private proposals."""
    encoded = "\n\n".join(
        f"Proposal {index}:\n{proposal.model_dump_json(indent=2)}" for index, proposal in enumerate(proposals, start=1)
    )
    return (
        f"Authoritative requirement:\n{requirement}\n\n"
        f"Bounded prior conversation background (non-authoritative):\n{background or '[none]'}\n\n"
        f"Deterministic frozen-repository evidence packet:\n{base_evidence}\n\n"
        f"Candidate packets:\n{encoded}"
    )


def proposal_schema_text() -> str:
    """Return the proposal schema for logs and non-native structured-output adapters."""
    return json.dumps(ClarificationProposal.model_json_schema(), ensure_ascii=False, sort_keys=True)
