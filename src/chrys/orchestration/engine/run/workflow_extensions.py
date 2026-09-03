# Copyright (c) 2026 Chrys. All rights reserved.

"""Extension points on the requirement-clarification workflow.

The long-horizon track is the clarification workflow plus three things: a code
search running beside the clarification, that search folded into the repair
reminder and the PACT inputs, and a delegation pass after the repair. None of
those change how clarification itself decides anything, so they attach here
rather than forking the workflow — one implementation keeps working, and the
default is a no-op that leaves every existing behaviour byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot
    from chrys.service.requirement_clarification.types import RequirementRevision

RepairStatus = Literal["succeeded", "promoted_p0", "failed", "interrupted"]


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    """What the repair pass left behind for anything that runs after it."""

    status: RepairStatus
    final_text: str
    baseline: Literal["p1", "p0", "none"]
    """Which implementation the workspace actually holds now."""
    pact_input_dir: Path | None = None
    """``06-pact-input/`` when a validated Goal Contract and plan were written."""


class RequirementWorkflowExtensions(Protocol):
    """Optional work the clarification workflow invites in at six points."""

    def wants_delegation_pass(self) -> bool:
        """Whether a pass after the repair will produce this turn's final answer.

        When true, the workflow shows the repaired text as an intermediate
        message instead of closing the turn on it.
        """
        ...

    async def on_clarification_start(self, revision: RequirementRevision, s0: WorkspaceSnapshot) -> None:
        """Start work that runs beside clarification, against the frozen S0 view."""
        ...

    def augment_repair_reminder(self, delta_text: str) -> str:
        """Return the repair reminder, optionally extended with more evidence."""
        ...

    def pact_input_hints(self) -> str:
        """Return untrusted evidence to attach to the Initial Plan prompt."""
        ...

    async def after_repair(self, outcome: RepairOutcome) -> None:
        """React once the workspace holds its final baseline for this turn."""
        ...

    async def on_revision(self, revision: RequirementRevision) -> None:
        """React to an amendment: the previous revision's work is now stale."""
        ...

    async def cancel(self) -> None:
        """Abandon anything still running."""
        ...


class NoopExtensions:
    """The default: the clarification workflow exactly as it was."""

    def wants_delegation_pass(self) -> bool:
        return False

    async def on_clarification_start(self, revision: RequirementRevision, s0: WorkspaceSnapshot) -> None:
        return None

    def augment_repair_reminder(self, delta_text: str) -> str:
        return delta_text

    def pact_input_hints(self) -> str:
        return ""

    async def after_repair(self, outcome: RepairOutcome) -> None:
        return None

    async def on_revision(self, revision: RequirementRevision) -> None:
        return None

    async def cancel(self) -> None:
        return None
