# Copyright (c) 2026 Chrys. All rights reserved.

"""Hermetic Chrys coordinator integration with the real PACT R3 Control Plane."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from pact_core.adapters.fake import FakeAdapter, FakeTurn
from pact_core.runtime.planner import PlanningRequest, parse_plan_revision_proposal
from pact_core.runtime.planning import validate_initial_plan
from pact_core.runtime.schemas import MANAGER_DECISION_PROPOSAL_SCHEMA
from pact_core.schemas import PlanChallenge, ReviewDecisionCapture

from chrys.app.acp.bridge import SessionUpdate
from chrys.foundation.config.settings_store import LoadedSettings
from chrys.pact.campaign import CampaignCoordinator, SemanticRole, UpdateSender

# PACT keeps every mission in its own git worktree named by campaign, mission, plan and
# attempt; under pytest's temp directory on Windows that path exceeds what git accepts
# ("fatal: '$GIT_DIR' too big"). The runtime behaves the same on every platform; only
# the path budget differs, so these run on POSIX.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="git worktree paths exceed Windows limits under pytest tmp"
)

_WORKER_RESULT = """## Changed files
- greeting.txt: added

## Verification evidence
- deterministic artifact checks passed

## Known gaps
- none

## Next step
- none
"""
_REVIEW_COMPLETE = "All target criteria are satisfied.\n\nPACT_COMPLETE\n"
_WORKER_FOUNDATION = _WORKER_RESULT.replace("greeting.txt", "foundation.txt")
_REVIEW_CHALLENGE = "The original foundation route is invalid and should be replaced.\n"


def _default_scripts() -> dict[SemanticRole, list[FakeTurn]]:
    return {
        "worker": [FakeTurn(final_text=_WORKER_RESULT, write_files={"greeting.txt": "hello\n"})],
        "reviewer": [FakeTurn(final_text=_REVIEW_COMPLETE)],
        "planner": [],
        "manager": [],
    }


class _CancellableFakeAdapter(FakeAdapter):
    """PACT's hermetic adapter with the Chrys-owned cancel extension."""

    def __init__(self, role: SemanticRole, turns: list[FakeTurn]) -> None:
        super().__init__(turns)
        self.id = f"scripted-{role}"
        self.cancelled = False

    async def cancel_current_turn(self) -> None:
        self.cancelled = True


class _ScriptedAdapterFactory:
    def __init__(self, scripts: dict[SemanticRole, list[FakeTurn]] | None = None) -> None:
        scripts = _default_scripts() if scripts is None else scripts
        self.adapters: dict[SemanticRole, _CancellableFakeAdapter] = {
            role: _CancellableFakeAdapter(role, scripts[role]) for role in ("worker", "reviewer", "planner", "manager")
        }

    def __call__(
        self,
        *,
        semantic_role: SemanticRole,
        profile_name: str,
        loaded_settings: LoadedSettings,
        outer_loop: asyncio.AbstractEventLoop,
        campaign_id: str,
        send_update: UpdateSender,
        abort_event: threading.Event,
    ) -> _CancellableFakeAdapter:
        _ = profile_name, loaded_settings, outer_loop, campaign_id, send_update, abort_event
        return self.adapters[semantic_role]


class _UpdateCollector:
    def __init__(self) -> None:
        self.updates: list[SessionUpdate] = []

    async def __call__(self, update: SessionUpdate) -> None:
        self.updates.append(update)


def _write_inputs(workspace: Path) -> tuple[Path, Path]:
    request_dir = workspace / ".pact-io" / "chrys-pact" / "integration"
    request_dir.mkdir(parents=True)
    contract = request_dir / "goal-contract.json"
    plan = request_dir / "initial-plan.json"
    contract.write_text(
        json.dumps(
            {
                "schema": "pact-runtime/goal-contract/v1",
                "goal": "Create a greeting artifact.",
                "acceptance_criteria": [{"id": "ac-1", "text": "greeting.txt contains hello."}],
                "non_goals": ["No unrelated refactor."],
            }
        ),
        encoding="utf-8",
    )
    plan.write_text(
        json.dumps(
            {
                "schema": "pact-runtime/initial-plan/v1",
                "constraints": ["Keep the change scoped to greeting.txt."],
                "missions": [
                    {
                        "id": "greeting",
                        "objective": "Create the greeting.",
                        "target_ac_ids": ["ac-1"],
                        "dependencies": [],
                        "verification_intent": "Confirm greeting.txt contains hello.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return contract, plan


def _replan_initial_plan() -> dict[str, object]:
    return {
        "schema": "pact-runtime/initial-plan/v1",
        "constraints": ["Keep changes scoped."],
        "missions": [
            {
                "id": "foundation",
                "objective": "Create the foundation marker.",
                "target_ac_ids": ["ac-1"],
                "dependencies": [],
                "verification_intent": "Confirm foundation.txt exists.",
            },
            {
                "id": "integration",
                "objective": "Create the greeting.",
                "target_ac_ids": ["ac-2"],
                "dependencies": ["foundation"],
                "verification_intent": "Confirm greeting.txt contains hello.",
            },
        ],
    }


def _write_replan_inputs(workspace: Path) -> tuple[Path, Path]:
    request_dir = workspace / ".pact-io" / "chrys-pact" / "governed-replan"
    request_dir.mkdir(parents=True)
    contract = request_dir / "goal-contract.json"
    plan = request_dir / "initial-plan.json"
    contract.write_text(
        json.dumps(
            {
                "schema": "pact-runtime/goal-contract/v1",
                "goal": "Create a governed two-stage greeting.",
                "acceptance_criteria": [
                    {"id": "ac-1", "text": "foundation.txt records ready."},
                    {"id": "ac-2", "text": "greeting.txt contains hello."},
                ],
                "non_goals": ["No unrelated work."],
            }
        ),
        encoding="utf-8",
    )
    plan.write_text(json.dumps(_replan_initial_plan()), encoding="utf-8")
    return contract, plan


def _replan_proposal_payload() -> dict[str, object]:
    return {
        "schema": "pact-runtime/plan-revision-proposal/v1",
        "parent_plan_revision": 1,
        "input_work_state_revision": 1,
        "reason": "Reviewer found the original foundation route invalid.",
        "rationale": "Use replacement Missions with explicit evidence lanes.",
        "constraints": ["Keep changes scoped."],
        "missions": [
            {
                "id": "foundation",
                "objective": "Create the foundation marker.",
                "target_ac_ids": ["ac-1"],
                "dependencies": [],
                "supersedes": [],
                "verification_intent": "Confirm foundation.txt exists.",
            },
            {
                "id": "integration",
                "objective": "Create the greeting.",
                "target_ac_ids": ["ac-2"],
                "dependencies": ["foundation"],
                "supersedes": [],
                "verification_intent": "Confirm greeting.txt contains hello.",
            },
            {
                "id": "replacement-foundation",
                "objective": "Create a verified replacement foundation.",
                "target_ac_ids": ["ac-1"],
                "dependencies": [],
                "supersedes": ["foundation"],
                "verification_intent": "Confirm foundation.txt records ready.",
            },
            {
                "id": "replacement-integration",
                "objective": "Create the greeting on the replacement foundation.",
                "target_ac_ids": ["ac-2"],
                "dependencies": ["replacement-foundation"],
                "supersedes": ["integration"],
                "verification_intent": "Confirm greeting.txt contains hello.",
            },
        ],
        "operations": [
            {"op": "add_mission", "mission_id": "replacement-foundation"},
            {"op": "add_mission", "mission_id": "replacement-integration"},
            {
                "op": "supersede_mission",
                "mission_id": "foundation",
                "replacement_mission_ids": ["replacement-foundation"],
            },
            {
                "op": "supersede_mission",
                "mission_id": "integration",
                "replacement_mission_ids": ["replacement-integration"],
            },
        ],
        "affected_mission_ids": [
            "foundation",
            "integration",
            "replacement-foundation",
            "replacement-integration",
        ],
        "affected_ac_ids": ["ac-1", "ac-2"],
    }


async def test_real_control_plane_promotes_and_terminal_matches_canonical_projection(
    tmp_path: Path,
    git_repo_factory: Callable[[Path], Path],
) -> None:
    workspace = git_repo_factory(tmp_path / "repo")
    contract, plan = _write_inputs(workspace)
    adapter_factory = _ScriptedAdapterFactory()
    collector = _UpdateCollector()
    coordinator = CampaignCoordinator(
        profile_name="Code",
        loaded_settings=cast(LoadedSettings, object()),
        verify_command=None,
        allow_unverified=True,
        adapter_factory=adapter_factory,
        worktree_root=tmp_path / "pact-worktrees",
    )

    terminal = await coordinator.run(
        workspace=workspace,
        contract_file=contract,
        plan_file=plan,
        send_update=collector,
    )

    assert terminal.status == "completed"
    assert terminal.revision == 1
    assert terminal.next_action == "none"
    assert terminal.artifact_ref == f".pact/runtime/campaigns/{terminal.campaign_id}"
    assert (workspace / "greeting.txt").read_text(encoding="utf-8") == "hello\n"

    campaign_dir = workspace / terminal.artifact_ref
    work_state = json.loads((campaign_dir / "work-state.json").read_text(encoding="utf-8"))
    dashboard = json.loads((campaign_dir / "dashboard.json").read_text(encoding="utf-8"))
    assert work_state["status"] == terminal.status
    assert work_state["revision"] == terminal.revision
    assert work_state["next_action"] == terminal.next_action
    assert work_state["acceptance"]["gaps"] == []
    assert dashboard["overview"]["status"] == terminal.status
    assert dashboard["source"]["work_state_revision"] == terminal.revision
    assert collector.updates[0].status == "in_progress"
    assert collector.updates[-1].status == "completed"


async def test_real_control_plane_blocks_failed_later_mission_without_false_promotion(
    tmp_path: Path,
    git_repo_factory: Callable[[Path], Path],
) -> None:
    workspace = git_repo_factory(tmp_path / "repo")
    contract, plan = _write_replan_inputs(workspace)
    verify_command = (
        f'"{sys.executable}" -c "from pathlib import Path; raise SystemExit(Path(\'greeting.txt\').exists())"'
    )
    adapter_factory = _ScriptedAdapterFactory(
        {
            "worker": [
                FakeTurn(
                    final_text=_WORKER_FOUNDATION,
                    write_files={"foundation.txt": "ready\n"},
                ),
                *[
                    FakeTurn(
                        final_text=_WORKER_RESULT,
                        write_files={"greeting.txt": "must-not-promote\n"},
                    )
                    for _ in range(3)
                ],
            ],
            "reviewer": [FakeTurn(final_text=_REVIEW_COMPLETE) for _ in range(4)],
            "planner": [],
            "manager": [
                FakeTurn(
                    final_text=json.dumps(
                        {
                            "schema": MANAGER_DECISION_PROPOSAL_SCHEMA,
                            "phase": "route",
                            "action": "block",
                            "expected_plan_revision": 1,
                            "expected_work_state_revision": 2,
                            "reason": "Deterministic verification failed for the later Mission.",
                        }
                    )
                )
            ],
        }
    )
    collector = _UpdateCollector()
    coordinator = CampaignCoordinator(
        profile_name="Code",
        loaded_settings=cast(LoadedSettings, object()),
        verify_command=verify_command,
        allow_unverified=False,
        adapter_factory=adapter_factory,
        worktree_root=tmp_path / "pact-worktrees",
    )

    terminal = await coordinator.run(
        workspace=workspace,
        contract_file=contract,
        plan_file=plan,
        send_update=collector,
    )

    assert terminal.status == "blocked"
    assert terminal.revision == 3
    assert terminal.next_action == "manager_blocked"
    assert (workspace / "foundation.txt").read_text(encoding="utf-8") == "ready\n"
    assert not (workspace / "greeting.txt").exists()

    campaign_dir = workspace / terminal.artifact_ref
    work_state = json.loads((campaign_dir / "work-state.json").read_text(encoding="utf-8"))
    dashboard = json.loads((campaign_dir / "dashboard.json").read_text(encoding="utf-8"))
    evidence = [json.loads((workspace / ref).read_text(encoding="utf-8")) for ref in work_state["evidence_refs"]]
    assert work_state["status"] == terminal.status
    assert work_state["revision"] == terminal.revision
    assert work_state["next_action"] == terminal.next_action
    assert [mission["status"] for mission in work_state["missions"]] == ["completed", "blocked"]
    assert work_state["acceptance"]["gaps"] == ["ac-2"]
    assert work_state["closure"]["completed"] is False
    assert [item["promotion_status"] for item in evidence] == ["applied", "not_attempted"]
    assert dashboard["overview"]["status"] == terminal.status
    assert dashboard["overview"]["next_action"] == terminal.next_action
    assert dashboard["source"]["work_state_revision"] == terminal.revision
    assert collector.updates[-1].status == "failed"
    assert collector.updates[-1].raw_output == terminal.summary_text()


async def test_real_control_plane_governed_replan_reaches_plan_revision_two(
    tmp_path: Path,
    git_repo_factory: Callable[[Path], Path],
) -> None:
    workspace = git_repo_factory(tmp_path / "repo")
    contract, plan = _write_replan_inputs(workspace)
    proposal_payload = _replan_proposal_payload()
    parent_plan = validate_initial_plan(_replan_initial_plan(), acceptance_criterion_ids=("ac-1", "ac-2"))
    proposal_request = PlanningRequest(
        campaign_id="hash-only",
        contract_revision=1,
        plan_revision=1,
        work_state_revision=1,
        trigger="reviewer_plan_challenge",
        manager_constraints=("Replace the challenged route.",),
        acceptance_criterion_ids=("ac-1", "ac-2"),
        parent_plan=parent_plan,
        projection={},
    )
    proposal = parse_plan_revision_proposal(json.dumps(proposal_payload), request=proposal_request)
    challenge = ReviewDecisionCapture(
        verdict_status="valid",
        verdict="continue",
        plan_challenge_status="valid",
        plan_challenge=PlanChallenge(
            reason="The foundation route is invalid.",
            gap_signature="foundation:invalid-route",
            recommended_action="replan",
        ),
    )
    adapter_factory = _ScriptedAdapterFactory(
        {
            "worker": [
                FakeTurn(
                    final_text=_WORKER_FOUNDATION,
                    write_files={"foundation.txt": "untrusted\n"},
                ),
                FakeTurn(
                    final_text=_WORKER_FOUNDATION,
                    write_files={"foundation.txt": "ready\n"},
                ),
                FakeTurn(
                    final_text=_WORKER_RESULT,
                    write_files={"greeting.txt": "hello\n"},
                ),
            ],
            "reviewer": [
                FakeTurn(final_text=_REVIEW_CHALLENGE, review_decision=challenge),
                FakeTurn(final_text=_REVIEW_COMPLETE),
                FakeTurn(final_text=_REVIEW_COMPLETE),
            ],
            "planner": [FakeTurn(final_text=json.dumps(proposal_payload))],
            "manager": [
                FakeTurn(
                    final_text=json.dumps(
                        {
                            "schema": MANAGER_DECISION_PROPOSAL_SCHEMA,
                            "phase": "route",
                            "action": "request_replan",
                            "expected_plan_revision": 1,
                            "expected_work_state_revision": 1,
                            "reason": "The typed challenge requires a new route.",
                            "constraints": ["Replace the challenged route."],
                        }
                    )
                ),
                FakeTurn(
                    final_text=json.dumps(
                        {
                            "schema": MANAGER_DECISION_PROPOSAL_SCHEMA,
                            "phase": "review_plan",
                            "action": "approve_plan",
                            "expected_plan_revision": 1,
                            "expected_work_state_revision": 1,
                            "reason": "The proposal preserves the contract.",
                            "proposal_sha256": proposal.sha256,
                        }
                    )
                ),
            ],
        }
    )
    collector = _UpdateCollector()
    coordinator = CampaignCoordinator(
        profile_name="Code",
        loaded_settings=cast(LoadedSettings, object()),
        verify_command=None,
        allow_unverified=True,
        adapter_factory=adapter_factory,
        worktree_root=tmp_path / "pact-worktrees",
    )

    terminal = await coordinator.run(
        workspace=workspace,
        contract_file=contract,
        plan_file=plan,
        send_update=collector,
    )

    assert terminal.status == "completed"
    assert terminal.revision == 4
    assert (workspace / "foundation.txt").read_text(encoding="utf-8") == "ready\n"
    assert (workspace / "greeting.txt").read_text(encoding="utf-8") == "hello\n"
    campaign_dir = workspace / terminal.artifact_ref
    work_state = json.loads((campaign_dir / "work-state.json").read_text(encoding="utf-8"))
    dashboard = json.loads((campaign_dir / "dashboard.json").read_text(encoding="utf-8"))
    plan_revision = json.loads((campaign_dir / "plan-revisions" / "rev-0002.json").read_text(encoding="utf-8"))
    assert work_state["revision"] == terminal.revision
    assert work_state["plan_revision"] == 2
    assert [mission["status"] for mission in work_state["missions"]] == [
        "superseded",
        "superseded",
        "completed",
        "completed",
    ]
    assert plan_revision["planner_proposal_ref"]
    assert plan_revision["manager_approval_ref"]
    assert dashboard["overview"]["plan_revision"] == 2
    assert dashboard["source"]["work_state_revision"] == terminal.revision
    assert f"revision: {terminal.revision}" in cast(str, collector.updates[-2].raw_output)
    assert collector.updates[-1].status == "completed"
