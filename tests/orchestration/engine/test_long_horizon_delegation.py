# Copyright (c) 2026 Chrys. All rights reserved.

"""The PACT delegation pass that runs after the repair."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import LongHorizonPhaseChanged, Warning
from chrys.orchestration.engine.run.long_horizon import LongHorizonExtensions, LongHorizonPhase
from chrys.orchestration.engine.run.workflow_extensions import RepairOutcome
from chrys.service.llm.side_call_clients import SideCallClientCache
from chrys.service.profiles.agents.schema import AgentProfile, RoutingConfig
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.routing.classifier import RouteBand, RouteDecision, RouteTrack, TurnPlan


@dataclass
class _Executor:
    adopted: list[str] = field(default_factory=list)
    history_state: dict[str, Any] = field(default_factory=dict)

    def adopt_fallback_success(self, text: str) -> None:
        self.adopted.append(text)


@dataclass
class _Reminder:
    queued: list[list[str]] = field(default_factory=list)

    def queue_hook_reminders(self, reminders: list[str]) -> None:
        self.queued.append(reminders)


@dataclass
class _Workspace:
    primary_cwd: str


@dataclass
class _Runner:
    passes: list[str] = field(default_factory=list)
    fail: bool = False

    async def _run_fresh_standard(self, text: str, **_kwargs: Any) -> None:
        if self.fail:
            msg = "executor blew up"
            raise RuntimeError(msg)
        self.passes.append(text)


@dataclass
class _Host:
    _bus: EventBus = field(default_factory=EventBus)
    _session_id: str | None = "sess"
    _session_dir: Path | None = None
    _turn_number: int = 0
    _workspace: _Workspace | None = None
    _executor: _Executor = field(default_factory=_Executor)
    _reminder_middleware: _Reminder = field(default_factory=_Reminder)
    _agent_profile: AgentProfile | None = None
    _active_profile: ModelProfile | None = field(default_factory=lambda: ModelProfile(id="m", name="M"))
    _side_call_clients: SideCallClientCache = field(default_factory=SideCallClientCache)
    settings: Settings = field(default_factory=Settings)

    @property
    def _settings(self) -> Settings:
        return self.settings


def _decision(*, pact: bool = True) -> RouteDecision:
    return RouteDecision(
        track=RouteTrack.LONG_HORIZON,
        band=RouteBand.STRONG_LONG_HORIZON if pact else RouteBand.LEAN_LONG_HORIZON,
        plan=TurnPlan(True, True, pact),
        reason="scope",
        confidence=0.9,
        source="heuristic",
    )


@pytest.fixture
def staged(tmp_path: Path) -> tuple[LongHorizonExtensions, _Host, _Runner, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pact_input = tmp_path / "session" / "requirement_clarification" / "turn_1" / "06-pact-input"
    pact_input.mkdir(parents=True)
    (pact_input / "goal-contract.json").write_text('{"goal": "add oauth"}', encoding="utf-8")
    (pact_input / "initial-plan.json").write_text('{"steps": []}', encoding="utf-8")
    host = _Host(
        _session_dir=tmp_path / "session",
        _workspace=_Workspace(primary_cwd=str(workspace)),
        _agent_profile=AgentProfile(name="LongHorizon", routing=RoutingConfig(mode="auto")),
    )
    runner = _Runner()
    extensions = LongHorizonExtensions(host, _decision(), runner=runner)
    extensions._requirement = "Add OAuth login"
    return extensions, host, runner, pact_input


async def test_a_third_pass_runs_with_the_run_request_reminder(
    staged: tuple[LongHorizonExtensions, _Host, _Runner, Path],
) -> None:
    extensions, host, runner, pact_input = staged

    await extensions.after_repair(RepairOutcome("succeeded", "P1 text", "p1", pact_input_dir=pact_input))

    assert runner.passes == ["Add OAuth login"]
    (reminders,) = host._reminder_middleware.queued
    assert "chrys-pact/run-request/v1" in reminders[0]
    assert "chrys_pact" in reminders[0]
    assert "verbatim" in reminders[0]


async def test_the_inputs_are_copied_into_the_workspace(
    staged: tuple[LongHorizonExtensions, _Host, _Runner, Path],
) -> None:
    """The campaign runs in the workspace, so its inputs have to be there."""
    extensions, host, _runner, pact_input = staged

    await extensions.after_repair(RepairOutcome("succeeded", "P1", "p1", pact_input_dir=pact_input))

    assert extensions.request is not None
    workspace = Path(host._workspace.primary_cwd)
    contract = workspace / extensions.request.contract_path
    assert json.loads(contract.read_text(encoding="utf-8")) == {"goal": "add oauth"}


async def test_no_pact_input_means_no_delegation(
    staged: tuple[LongHorizonExtensions, _Host, _Runner, Path],
) -> None:
    """Without an accepted contract there is nothing to hand over."""
    extensions, host, runner, _pact_input = staged

    await extensions.after_repair(RepairOutcome("succeeded", "P1", "p1", pact_input_dir=None))

    assert runner.passes == []
    assert host._reminder_middleware.queued == []


async def test_a_lean_plan_never_delegates(
    staged: tuple[LongHorizonExtensions, _Host, _Runner, Path],
) -> None:
    extensions, _host, runner, pact_input = staged
    extensions._decision = _decision(pact=False)

    await extensions.after_repair(RepairOutcome("succeeded", "P1", "p1", pact_input_dir=pact_input))

    assert runner.passes == []


async def test_a_failed_delegation_keeps_the_repaired_answer(
    staged: tuple[LongHorizonExtensions, _Host, _Runner, Path],
) -> None:
    """The repaired baseline is already in the workspace; do not discard it."""
    extensions, host, runner, pact_input = staged
    runner.fail = True
    warnings: list[Warning] = []
    await host._bus.subscribe(Warning, warnings.append)

    await extensions.after_repair(RepairOutcome("succeeded", "P1 text", "p1", pact_input_dir=pact_input))

    assert host._executor.adopted == ["P1 text"]
    assert warnings and warnings[-1].code == "long_horizon_delegation_failed"


async def test_an_interrupted_delegation_reports_interrupted(
    staged: tuple[LongHorizonExtensions, _Host, _Runner, Path],
) -> None:
    import asyncio

    extensions, host, runner, pact_input = staged
    phases: list[LongHorizonPhaseChanged] = []
    await host._bus.subscribe(LongHorizonPhaseChanged, phases.append)

    async def _cancelled(*_args: Any, **_kwargs: Any) -> None:
        raise asyncio.CancelledError

    runner._run_fresh_standard = _cancelled  # type: ignore[method-assign]

    with pytest.raises(asyncio.CancelledError):
        await extensions.after_repair(RepairOutcome("succeeded", "P1", "p1", pact_input_dir=pact_input))

    assert phases[-1].phase == LongHorizonPhase.INTERRUPTED
    assert phases[-1].terminal is True


async def test_the_delegating_phase_is_announced(
    staged: tuple[LongHorizonExtensions, _Host, _Runner, Path],
) -> None:
    extensions, host, _runner, pact_input = staged
    phases: list[LongHorizonPhaseChanged] = []
    await host._bus.subscribe(LongHorizonPhaseChanged, phases.append)

    await extensions.after_repair(RepairOutcome("succeeded", "P1", "p1", pact_input_dir=pact_input))

    names = [event.phase for event in phases]
    assert LongHorizonPhase.DELEGATING in names
    assert names[-1] == LongHorizonPhase.COMPLETED
    assert phases[-1].terminal is True


async def test_the_profile_can_rename_the_delegation_tool(
    staged: tuple[LongHorizonExtensions, _Host, _Runner, Path],
) -> None:
    from chrys.service.profiles.agents.schema import LongHorizonConfig

    extensions, host, _runner, pact_input = staged
    host._agent_profile = AgentProfile(
        name="LongHorizon",
        routing=RoutingConfig(mode="auto", long_horizon=LongHorizonConfig(pact_tool="my_campaign")),
    )

    await extensions.after_repair(RepairOutcome("succeeded", "P1", "p1", pact_input_dir=pact_input))

    (reminders,) = host._reminder_middleware.queued
    assert "my_campaign" in reminders[0]
