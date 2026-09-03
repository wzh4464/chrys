# Copyright (c) 2026 Chrys. All rights reserved.

"""Amendments, downgrades, and cancellation on the long-horizon track."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import RouteOverride
from chrys.service.routing.classifier import RouteBand, RouteDecision, RouteTrack, TurnPlan
from chrys.service.state.store import JsonFileStateStore


@dataclass
class _Workflow:
    stops: int = 0
    revisions: list[Any] = field(default_factory=list)

    async def request_stop(self) -> None:
        self.stops += 1


def _decision() -> RouteDecision:
    return RouteDecision(
        track=RouteTrack.LONG_HORIZON,
        band=RouteBand.STRONG_LONG_HORIZON,
        plan=TurnPlan(True, True, True),
        reason="scope",
        confidence=0.9,
        source="heuristic",
    )


async def test_quick_during_preparation_stops_the_live_workflow(agent_engine, tmp_path) -> None:
    """A user watching a campaign spin up has no other way to stop it."""
    engine = agent_engine(EventBus(), settings=Settings(), state_store=JsonFileStateStore(tmp_path))
    workflow = _Workflow()
    engine._requirement_clarification_workflow = workflow
    engine._last_route = _decision()

    await engine._on_route_override(RouteOverride(track="standard"))

    assert workflow.stops == 1
    # Consumed by the downgrade: it was about this turn, and this turn heard it.
    assert engine._route_override is None


async def test_quick_with_no_live_workflow_arms_the_next_turn(agent_engine, tmp_path) -> None:
    engine = agent_engine(EventBus(), settings=Settings(), state_store=JsonFileStateStore(tmp_path))

    await engine._on_route_override(RouteOverride(track="standard"))

    assert engine._route_override is not None
    assert engine._route_override.track == "standard"


async def test_longrun_never_stops_a_live_workflow(agent_engine, tmp_path) -> None:
    """Promoting a turn that is already long-horizon has nothing to promote."""
    engine = agent_engine(EventBus(), settings=Settings(), state_store=JsonFileStateStore(tmp_path))
    workflow = _Workflow()
    engine._requirement_clarification_workflow = workflow
    engine._last_route = _decision()

    await engine._on_route_override(RouteOverride(track="long_horizon"))

    assert workflow.stops == 0
    assert engine._route_override is not None


async def test_a_reroute_request_never_stops_a_live_workflow(agent_engine, tmp_path) -> None:
    engine = agent_engine(EventBus(), settings=Settings(), state_store=JsonFileStateStore(tmp_path))
    workflow = _Workflow()
    engine._requirement_clarification_workflow = workflow
    engine._last_route = _decision()

    await engine._on_route_override(RouteOverride(track="", reroute=True))

    assert workflow.stops == 0


async def test_quick_on_an_unrouted_clarification_turn_is_left_alone(agent_engine, tmp_path) -> None:
    """A profile-configured clarification turn is not a routing decision to undo."""
    engine = agent_engine(EventBus(), settings=Settings(), state_store=JsonFileStateStore(tmp_path))
    workflow = _Workflow()
    engine._requirement_clarification_workflow = workflow
    engine._last_route = None

    await engine._on_route_override(RouteOverride(track="standard"))

    assert workflow.stops == 0


async def test_an_amendment_cancels_and_clears_the_search(tmp_path) -> None:
    from pathlib import Path

    from chrys.orchestration.engine.run.long_horizon import LocalizationOutcome, LongHorizonExtensions
    from chrys.service.llm.side_call_clients import SideCallClientCache
    from chrys.service.profiles.models.schema import ModelProfile

    @dataclass
    class _Host:
        _bus: EventBus = field(default_factory=EventBus)
        _session_id: str | None = "sess"
        _session_dir: Path | None = None
        _turn_number: int = 0
        _active_profile: ModelProfile | None = None
        _side_call_clients: SideCallClientCache = field(default_factory=SideCallClientCache)
        settings: Settings = field(default_factory=Settings)

        @property
        def _settings(self) -> Settings:
            return self.settings

    @dataclass
    class _Revision:
        rendered: str
        number: int = 2

    extensions = LongHorizonExtensions(_Host(), _decision())
    extensions.localization = LocalizationOutcome(locations=[{"file": "a.py"}])
    extensions._memory_prior = "Strategy: recalled for the original requirement"

    await extensions.on_revision(_Revision(rendered="Add OAuth login, and also SAML"))

    assert extensions.localization.available is False
    # Both hints were gathered for text that no longer stands.
    assert extensions._memory_prior == ""
    # The next search runs against the amended requirement, not the original.
    assert extensions._requirement == "Add OAuth login, and also SAML"
