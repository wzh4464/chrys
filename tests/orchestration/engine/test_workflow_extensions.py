# Copyright (c) 2026 Chrys. All rights reserved.

"""The clarification workflow's extension points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chrys.orchestration.engine.run.workflow_extensions import (
    NoopExtensions,
    RepairOutcome,
    RequirementWorkflowExtensions,
)


class _Recorder:
    """Records the hooks the workflow calls, in order."""

    def __init__(self, *, delegation: bool = False) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._delegation = delegation

    def wants_delegation_pass(self) -> bool:
        return self._delegation

    async def on_clarification_start(self, revision: Any, s0: Any) -> None:
        self.calls.append(("on_clarification_start", revision))

    def augment_repair_reminder(self, delta_text: str) -> str:
        self.calls.append(("augment_repair_reminder", delta_text))
        return f"{delta_text}\n\n[extension evidence]"

    def pact_input_hints(self) -> str:
        self.calls.append(("pact_input_hints", None))
        return "src/a.py:1 primary"

    async def after_repair(self, outcome: RepairOutcome) -> None:
        self.calls.append(("after_repair", outcome))

    async def on_revision(self, revision: Any) -> None:
        self.calls.append(("on_revision", revision))

    async def cancel(self) -> None:
        self.calls.append(("cancel", None))


def test_the_recorder_satisfies_the_protocol() -> None:
    extensions: RequirementWorkflowExtensions = _Recorder()

    assert extensions.wants_delegation_pass() is False


def test_the_noop_default_changes_nothing() -> None:
    """Every existing clarification behaviour has to stay byte-identical."""
    extensions = NoopExtensions()

    assert extensions.wants_delegation_pass() is False
    assert extensions.augment_repair_reminder("delta text") == "delta text"
    assert extensions.pact_input_hints() == ""


async def test_the_noop_hooks_are_awaitable_and_return_nothing() -> None:
    extensions = NoopExtensions()

    assert await extensions.on_clarification_start(object(), object()) is None
    assert await extensions.after_repair(RepairOutcome("succeeded", "text", "p1")) is None
    assert await extensions.on_revision(object()) is None
    assert await extensions.cancel() is None


def test_the_noop_default_satisfies_the_protocol() -> None:
    extensions: RequirementWorkflowExtensions = NoopExtensions()

    assert extensions.pact_input_hints() == ""


@pytest.mark.parametrize(
    ("status", "baseline"),
    [("succeeded", "p1"), ("promoted_p0", "p0"), ("failed", "none"), ("interrupted", "none")],
)
def test_the_outcome_names_what_the_workspace_holds(status: str, baseline: str) -> None:
    """Anything running after the repair has to know which implementation is live."""
    outcome = RepairOutcome(status=status, final_text="text", baseline=baseline)  # type: ignore[arg-type]

    assert outcome.status == status
    assert outcome.baseline == baseline
    assert outcome.pact_input_dir is None


def test_the_outcome_carries_the_pact_input_directory(tmp_path: Path) -> None:
    outcome = RepairOutcome("succeeded", "text", "p1", pact_input_dir=tmp_path)

    assert outcome.pact_input_dir == tmp_path
