# Copyright (c) 2026 Chrys. All rights reserved.

"""Three-proposal/one-selector clarification behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from chrys.service.requirement_clarification.service import ClarificationService, render_delta
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot
from chrys.service.requirement_clarification.types import (
    ClarificationProposal,
    ClarificationSelection,
    RequirementRevision,
    SelectedGuidancePoint,
)


class _FakeModel:
    def __init__(self, selection: ClarificationSelection) -> None:
        self.selection = selection
        self.proposal_calls: list[int] = []
        self.selector_calls = 0

    async def propose(self, _prompt: str, *, sample_index: int):
        self.proposal_calls.append(sample_index)
        return ClarificationProposal(), {"input_tokens": sample_index}

    async def select(self, _prompt: str):
        self.selector_calls += 1
        return self.selection, {"output_tokens": 5}


def _snapshot(tmp_path: Path) -> WorkspaceSnapshot:
    return WorkspaceSnapshot(
        snapshot_id="s0",
        artifact_root=str(tmp_path),
        roots=(),
        manifest_hash="abc",
        total_bytes=0,
        entry_count=0,
    )


@pytest.mark.asyncio
async def test_service_runs_three_proposals_then_one_selector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    selection = ClarificationSelection(
        guidance_points=[
            SelectedGuidancePoint(
                category="integration",
                statement="Wire the parsed value into the existing runtime consumer.",
                confidence=0.91,
                basis="current_repo",
            )
        ]
    )
    model = _FakeModel(selection)
    monkeypatch.setattr(
        "chrys.service.requirement_clarification.service.collect_base_evidence",
        lambda _snapshot, _requirement: "packet",
    )

    result = await ClarificationService(model).clarify(
        revision=RequirementRevision(number=1, messages=("Add the option.",)),
        background="Earlier context",
        snapshot=_snapshot(tmp_path),
    )

    assert sorted(model.proposal_calls) == [1, 2, 3]
    assert model.selector_calls == 1
    assert result.delta == (
        "Repository implementation guidance:\n- Wire the parsed value into the existing runtime consumer."
    )
    assert len(result.usage_details) == 4


def test_render_delta_applies_confidence_and_character_budgets() -> None:
    selection = ClarificationSelection(
        guidance_points=[
            SelectedGuidancePoint(
                category="integration",
                statement="keep",
                confidence=0.75,
                basis="current_repo",
            ),
            SelectedGuidancePoint(
                category="testing",
                statement="drop",
                confidence=0.749,
                basis="current_repo",
            ),
        ]
    )

    assert render_delta(selection) == "Repository implementation guidance:\n- keep"
