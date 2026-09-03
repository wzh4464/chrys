# Copyright (c) 2026 Chrys. All rights reserved.

"""Ordering, inheritance, and escapes in the admission-time turn router."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import RouteOverride, TurnRouted, Warning
from chrys.orchestration.engine.run.routing import TurnRouter
from chrys.service.profiles.agents.schema import AgentProfile, RoutingConfig
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.routing.classifier import RouteBand, RouteDecision, RouteTrack
from chrys.service.routing.guard import TiebreakerGuard
from chrys.service.routing.llm import TiebreakerVerdict

_STRONG = (
    "Implement end-to-end OAuth login: add the provider abstraction, migrate the user table, "
    "update the API, write integration tests, and document the flow. Acceptance criteria: "
    "1) existing sessions keep working 2) new users can sign up with Google 3) all tests pass. "
    "Touch src/auth/provider.py, src/api/routes.py and web/src/login.tsx as needed."
)
_UNCERTAIN = "refactor the entire auth system"
_TRIVIAL = "thanks"
_FOLLOW_UP = "also rename the helper in the payments module and update its callers"


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Tools:
    def __init__(self, names: set[str]) -> None:
        self._names = sorted(names)

    def tool_names(self) -> list[str]:
        return list(self._names)


@dataclass
class _Host:
    """The narrow slice of the engine the router reads."""

    _bus: EventBus = field(default_factory=EventBus)
    _session_id: str | None = "sess"
    _session_dir: Path | None = None
    _agent_profile: AgentProfile | None = None
    _active_profile: ModelProfile | None = None
    _route_override: RouteOverride | None = None
    _last_route: RouteDecision | None = None
    _route_fingerprint: str = ""
    _tiebreaker_guard: TiebreakerGuard = field(default_factory=TiebreakerGuard)
    _sub_agent_tools: Any = field(default_factory=lambda: _Tools({"chrys_pact"}))
    settings: Settings = field(default_factory=lambda: Settings(pact_verify_command="uv run pytest"))
    switched: list[str] = field(default_factory=list)
    switch_ok: bool = True

    @property
    def _settings(self) -> Settings:
        return self.settings

    @property
    def _workspace(self) -> Any:
        return None


def _profile(**routing: Any) -> AgentProfile:
    return AgentProfile(name="Code", routing=RoutingConfig(**routing))


def _router(host: _Host, *, cwd: Path, clock: _Clock | None = None, verdict: TiebreakerVerdict | None = None):
    async def _switch(target: str) -> bool:
        host.switched.append(target)
        if host.switch_ok:
            host._agent_profile = _profile(mode="auto")
            host._agent_profile.name = target
        return host.switch_ok

    async def _tiebreak(_text: str, _signals: Any) -> TiebreakerVerdict:
        if verdict is None:
            raise AssertionError("tiebreaker must not be consulted")
        return verdict

    return TurnRouter(
        host,  # type: ignore[arg-type]
        workspace_cwd=str(cwd),
        clock=clock or _Clock(),
        switch_profile=_switch,
        tiebreak=_tiebreak,
    )


@pytest.fixture
def ready_workspace(tmp_path: Path) -> Path:
    (tmp_path / "tests").mkdir()
    return tmp_path


# --------------------------------------------------------------------------
# the probe is lazy: this runs on the admission path
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "settings", "profile"),
    [
        (_STRONG, Settings(routing_mode="off"), None),
        (_TRIVIAL, Settings(pact_verify_command="uv run pytest"), None),
        (_STRONG, Settings(pact_verify_command="uv run pytest"), {"mode": "off"}),
    ],
    ids=["routing_off", "trivial_follow_up", "profile_off"],
)
async def test_a_standard_decision_never_probes_the_workspace(
    ready_workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    prompt: str,
    settings: Settings,
    profile: dict[str, Any] | None,
) -> None:
    """Readiness vetoes only the campaign, and this runs before the turn starts."""

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("a standard-track decision must not probe the workspace")

    monkeypatch.setattr("chrys.orchestration.engine.run.routing.probe_workspace_readiness", explode)
    host = _Host(_agent_profile=_profile(**(profile or {})), settings=settings)

    decision = await _router(host, cwd=ready_workspace).decide(prompt, turn=1)

    assert decision.track is RouteTrack.STANDARD


async def test_a_long_horizon_decision_probes_exactly_once(
    ready_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The demotion check and the plan both need it; one probe answers both."""
    from chrys.service.routing import readiness as readiness_module

    calls = 0
    real = readiness_module.probe_workspace_readiness

    def counted(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr("chrys.orchestration.engine.run.routing.probe_workspace_readiness", counted)
    host = _Host(_agent_profile=_profile(mode="always"))

    decision = await _router(host, cwd=ready_workspace).decide(_STRONG, turn=1)

    assert decision.track is RouteTrack.LONG_HORIZON
    assert calls == 1


# --------------------------------------------------------------------------
# guards and overrides come first
# --------------------------------------------------------------------------


async def test_a_nested_acp_sub_agent_never_routes(ready_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_ACP_SUBAGENT_DEPTH", "1")
    host = _Host(_agent_profile=_profile(mode="always"))

    decision = await _router(host, cwd=ready_workspace).decide(_STRONG, turn=1)

    assert decision.track is RouteTrack.STANDARD
    assert decision.source == "guard"


async def test_the_global_setting_can_disable_routing_entirely(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="always"), settings=Settings(routing_mode="off"))

    decision = await _router(host, cwd=ready_workspace).decide(_STRONG, turn=1)

    assert decision.track is RouteTrack.STANDARD
    assert decision.source == "guard"


async def test_an_override_beats_the_heuristic_and_is_consumed_once(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="auto"), _route_override=RouteOverride(track="long_horizon"))
    router = _router(host, cwd=ready_workspace)

    first = await router.decide(_TRIVIAL, turn=1)
    second = await router.decide(_TRIVIAL, turn=2)

    assert first.track is RouteTrack.LONG_HORIZON
    assert first.source == "override"
    assert second.track is RouteTrack.STANDARD
    assert host._route_override is None


async def test_a_standard_override_downgrades_a_strong_prompt(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="auto"), _route_override=RouteOverride(track="standard"))

    decision = await _router(host, cwd=ready_workspace).decide(_STRONG, turn=1)

    assert decision.track is RouteTrack.STANDARD
    assert decision.source == "override"


async def test_the_guard_outranks_even_an_override(ready_workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_ACP_SUBAGENT_DEPTH", "2")
    host = _Host(_agent_profile=_profile(mode="auto"), _route_override=RouteOverride(track="long_horizon"))

    decision = await _router(host, cwd=ready_workspace).decide(_STRONG, turn=1)

    assert decision.source == "guard"
    # Still consumed: it was about this turn, and this turn is over.
    assert host._route_override is None


async def test_profile_mode_off_short_circuits_classification(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="off"))

    decision = await _router(host, cwd=ready_workspace).decide(_STRONG, turn=1)

    assert decision.track is RouteTrack.STANDARD
    assert decision.source == "profile"


async def test_profile_mode_always_routes_everything(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="always"))

    decision = await _router(host, cwd=ready_workspace).decide(_TRIVIAL, turn=1)

    assert decision.track is RouteTrack.LONG_HORIZON
    assert decision.source == "profile"


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


async def test_a_strong_prompt_routes_without_a_model_call(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="auto"))

    decision = await _router(host, cwd=ready_workspace).decide(_STRONG, turn=1)

    assert decision.track is RouteTrack.LONG_HORIZON
    assert decision.band is RouteBand.STRONG_LONG_HORIZON
    assert decision.source == "heuristic"
    assert host._tiebreaker_guard.calls == 0


async def test_a_trivial_follow_up_never_reaches_the_tiebreaker(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="auto"))

    decision = await _router(host, cwd=ready_workspace).decide(_TRIVIAL, turn=1)

    assert decision.track is RouteTrack.STANDARD
    assert decision.source == "heuristic"


@pytest.mark.parametrize(
    ("confidence", "expected_band"),
    [(0.9, RouteBand.STRONG_LONG_HORIZON), (0.72, RouteBand.LEAN_LONG_HORIZON)],
)
async def test_the_tiebreaker_promotes_by_confidence(
    ready_workspace: Path, confidence: float, expected_band: RouteBand
) -> None:
    host = _Host(_agent_profile=_profile(mode="auto"))
    verdict = TiebreakerVerdict(long_horizon=True, confidence=confidence, reason="multi-module")

    decision = await _router(host, cwd=ready_workspace, verdict=verdict).decide(_UNCERTAIN, turn=1)

    assert decision.track is RouteTrack.LONG_HORIZON
    assert decision.band is expected_band
    assert decision.source == "llm"


async def test_a_low_confidence_verdict_stays_standard(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="auto"))
    verdict = TiebreakerVerdict(long_horizon=True, confidence=0.4, reason="unsure")

    decision = await _router(host, cwd=ready_workspace, verdict=verdict).decide(_UNCERTAIN, turn=1)

    assert decision.track is RouteTrack.STANDARD


async def test_a_tiebreaker_failure_falls_back_and_is_reported(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="auto"))
    verdict = TiebreakerVerdict(long_horizon=False, confidence=0.0, reason="", failure="timeout")

    decision = await _router(host, cwd=ready_workspace, verdict=verdict).decide(_UNCERTAIN, turn=1)

    assert decision.track is RouteTrack.STANDARD
    assert decision.tiebreaker_failure == "timeout"


async def test_a_heuristic_only_profile_never_calls_a_model(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="auto", classifier="heuristic"))

    decision = await _router(host, cwd=ready_workspace).decide(_UNCERTAIN, turn=1)

    assert decision.source == "heuristic"
    assert decision.track is RouteTrack.STANDARD


# --------------------------------------------------------------------------
# readiness veto
# --------------------------------------------------------------------------


async def test_a_workspace_without_a_verify_command_cannot_delegate(tmp_path: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="auto"), settings=Settings())

    decision = await _router(host, cwd=tmp_path).decide(_STRONG, turn=1)

    assert decision.track is RouteTrack.LONG_HORIZON
    assert decision.plan.pact is False
    assert decision.band is RouteBand.LEAN_LONG_HORIZON
    assert "pact_not_ready" in decision.reason


async def test_a_ready_workspace_delegates(tmp_path: Path) -> None:
    host = _Host(
        _agent_profile=_profile(mode="auto"),
        settings=Settings(pact_verify_command="uv run pytest"),
        _sub_agent_tools=_Tools({"chrys_pact"}),
    )

    decision = await _router(host, cwd=tmp_path).decide(_STRONG, turn=1)

    assert decision.plan.pact is True
    assert decision.band is RouteBand.STRONG_LONG_HORIZON


# --------------------------------------------------------------------------
# inheritance and escapes
# --------------------------------------------------------------------------


def _lean(clock: _Clock, *, band: RouteBand = RouteBand.LEAN_LONG_HORIZON) -> RouteDecision:
    from chrys.service.routing.classifier import TurnPlan

    return RouteDecision(
        track=RouteTrack.LONG_HORIZON,
        band=band,
        plan=TurnPlan(True, True, False),
        reason="prior",
        confidence=0.75,
        source="heuristic",
        decided_at=clock.now,
        archetype="mutating_broad",
    )


async def test_a_lean_decision_is_inherited_by_a_follow_up(ready_workspace: Path) -> None:
    clock = _Clock()
    host = _Host(_agent_profile=_profile(mode="auto"), _route_fingerprint="fp")
    router = _router(host, cwd=ready_workspace, clock=clock)
    host._route_fingerprint = router.fingerprint()
    host._last_route = _lean(clock)

    decision = await router.decide(_FOLLOW_UP, turn=2)

    assert decision.source == "inherited"
    assert decision.band is RouteBand.LEAN_LONG_HORIZON


async def test_a_strong_decision_is_never_inherited(ready_workspace: Path) -> None:
    """Every PACT delegation is re-decided; none is carried by momentum."""
    clock = _Clock()
    host = _Host(_agent_profile=_profile(mode="auto"))
    router = _router(host, cwd=ready_workspace, clock=clock)
    host._route_fingerprint = router.fingerprint()
    host._last_route = _lean(clock, band=RouteBand.STRONG_LONG_HORIZON)

    decision = await router.decide(_FOLLOW_UP, turn=2)

    assert decision.source != "inherited"


async def test_inheritance_expires(ready_workspace: Path) -> None:
    clock = _Clock()
    host = _Host(_agent_profile=_profile(mode="auto", stale_after_seconds=60))
    router = _router(host, cwd=ready_workspace, clock=clock)
    host._route_fingerprint = router.fingerprint()
    host._last_route = _lean(clock)
    clock.now += 61

    decision = await router.decide(_FOLLOW_UP, turn=2)

    assert decision.source != "inherited"


async def test_a_changed_workspace_shape_breaks_inheritance(ready_workspace: Path) -> None:
    clock = _Clock()
    host = _Host(_agent_profile=_profile(mode="auto"))
    router = _router(host, cwd=ready_workspace, clock=clock)
    host._route_fingerprint = router.fingerprint()
    host._last_route = _lean(clock)
    (ready_workspace / "services").mkdir()

    decision = await router.decide(_FOLLOW_UP, turn=2)

    assert decision.source != "inherited"


async def test_an_archetype_flip_to_a_question_breaks_inheritance(ready_workspace: Path) -> None:
    clock = _Clock()
    host = _Host(_agent_profile=_profile(mode="auto"))
    router = _router(host, cwd=ready_workspace, clock=clock)
    host._route_fingerprint = router.fingerprint()
    host._last_route = _lean(clock)

    decision = await router.decide("explain what you just did", turn=2)

    assert decision.track is RouteTrack.STANDARD
    assert decision.source != "inherited"


async def test_a_trivial_acknowledgement_short_circuits_before_inheritance(ready_workspace: Path) -> None:
    clock = _Clock()
    host = _Host(_agent_profile=_profile(mode="auto"))
    router = _router(host, cwd=ready_workspace, clock=clock)
    host._route_fingerprint = router.fingerprint()
    host._last_route = _lean(clock)

    decision = await router.decide("thanks", turn=2)

    assert decision.track is RouteTrack.STANDARD
    assert decision.source == "heuristic"


async def test_reroute_abandons_inheritance(ready_workspace: Path) -> None:
    clock = _Clock()
    host = _Host(_agent_profile=_profile(mode="auto"), _route_override=RouteOverride(reroute=True))
    router = _router(host, cwd=ready_workspace, clock=clock)
    host._route_fingerprint = router.fingerprint()
    host._last_route = _lean(clock)

    decision = await router.decide(_FOLLOW_UP, turn=2)

    assert decision.source != "inherited"


async def test_inheritance_can_be_switched_off_per_profile(ready_workspace: Path) -> None:
    clock = _Clock()
    host = _Host(_agent_profile=_profile(mode="auto", inherit=False))
    router = _router(host, cwd=ready_workspace, clock=clock)
    host._route_fingerprint = router.fingerprint()
    host._last_route = _lean(clock)

    decision = await router.decide(_FOLLOW_UP, turn=2)

    assert decision.source != "inherited"


# --------------------------------------------------------------------------
# applying the decision
# --------------------------------------------------------------------------


async def test_a_long_horizon_decision_switches_to_the_target_profile(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="auto", target_profile="LongHorizon"))
    router = _router(host, cwd=ready_workspace)
    decision = await router.decide(_STRONG, turn=1)

    applied = await router.apply(decision)

    assert host.switched == ["LongHorizon"]
    assert applied.switched_to == "LongHorizon"
    assert applied.track is RouteTrack.LONG_HORIZON


async def test_a_failed_switch_downgrades_the_turn_and_warns(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="auto", target_profile="Missing"), switch_ok=False)
    warnings: list[Warning] = []
    await host._bus.subscribe(Warning, warnings.append)
    router = _router(host, cwd=ready_workspace)
    decision = await router.decide(_STRONG, turn=1)

    applied = await router.apply(decision)

    assert applied.track is RouteTrack.STANDARD
    assert applied.switched_to == ""
    assert warnings and warnings[-1].code == "route_profile_switch_failed"


async def test_no_switch_when_already_on_the_target(ready_workspace: Path) -> None:
    profile = _profile(mode="auto", target_profile="Code")
    host = _Host(_agent_profile=profile)
    router = _router(host, cwd=ready_workspace)
    decision = await router.decide(_STRONG, turn=1)

    applied = await router.apply(decision)

    assert host.switched == []
    assert applied.switched_to == ""


async def test_a_standard_decision_never_switches(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="auto", target_profile="LongHorizon"))
    router = _router(host, cwd=ready_workspace)
    decision = await router.decide(_TRIVIAL, turn=1)

    await router.apply(decision)

    assert host.switched == []


# --------------------------------------------------------------------------
# publication
# --------------------------------------------------------------------------


async def test_publishing_records_the_decision_for_the_next_turn(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="auto"))
    routed: list[TurnRouted] = []
    await host._bus.subscribe(TurnRouted, routed.append)
    router = _router(host, cwd=ready_workspace)
    decision = await router.decide(_STRONG, turn=1)

    await router.publish(decision, turn=1)

    assert host._last_route is decision
    assert host._route_fingerprint == router.fingerprint()
    assert routed[-1].track == "long_horizon"
    assert routed[-1].can_downgrade is True
    assert routed[-1].turn == 1


async def test_a_standard_turn_cannot_be_downgraded_further(ready_workspace: Path) -> None:
    host = _Host(_agent_profile=_profile(mode="auto"))
    routed: list[TurnRouted] = []
    await host._bus.subscribe(TurnRouted, routed.append)
    router = _router(host, cwd=ready_workspace)

    await router.publish(await router.decide(_TRIVIAL, turn=1), turn=1)

    assert routed[-1].can_downgrade is False
