# Copyright (c) 2026 Chrys. All rights reserved.

"""Decide a turn's track while it is still being admitted.

Admission is the last moment where switching profiles is free: the FSM is not
running, no run task exists, and the reserved admission slot has not been taken
yet. One step later the decision would have to unwind a started turn.
"""

from __future__ import annotations

import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from chrys.foundation.events.types import RouteOverride, TurnRouted, Warning
from chrys.service.profiles.agents.schema import RoutingConfig
from chrys.service.routing.classifier import (
    PromptSignals,
    RouteBand,
    RouteDecision,
    RouteTrack,
    TurnPlan,
    band_for,
    extract_prompt_signals,
    plan_for,
    prompt_score,
)
from chrys.service.routing.guard import TiebreakerGuard
from chrys.service.routing.llm import LlmRouteClassifier, TiebreakerVerdict
from chrys.service.routing.readiness import (
    WorkspaceReadiness,
    probe_workspace_readiness,
    workspace_fingerprint,
)

if TYPE_CHECKING:
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.events.bus import EventBus
    from chrys.service.profiles.agents.schema import AgentProfile
    from chrys.service.profiles.models.schema import ModelProfile

_ACP_DEPTH_ENV = "CHRYS_ACP_SUBAGENT_DEPTH"
_STRONG_CONFIDENCE = 0.85
_TRIVIAL_FOLLOW_UP_WORDS = 6
_SWITCH_TIMEOUT_SECONDS = 60.0


class TurnRouterHost(Protocol):
    """The slice of the engine the router reads and records into."""

    _bus: EventBus
    _session_id: str | None
    _agent_profile: AgentProfile | None
    _active_profile: ModelProfile | None
    _route_override: RouteOverride | None
    _last_route: RouteDecision | None
    _route_fingerprint: str
    _tiebreaker_guard: TiebreakerGuard
    _sub_agent_tools: Any | None
    _agent_registry: Any | None

    @property
    def _settings(self) -> Settings: ...

    @property
    def _session_dir(self) -> Path | None: ...


class TurnRouter:
    """Classify one turn, switch profiles for it, and record the verdict."""

    def __init__(
        self,
        host: TurnRouterHost,
        *,
        workspace_cwd: str,
        clock: Callable[[], float] = time.monotonic,
        switch_profile: Callable[[str], Awaitable[bool]] | None = None,
        tiebreak: Callable[[str, PromptSignals], Awaitable[TiebreakerVerdict]] | None = None,
    ) -> None:
        self._host = host
        self._cwd = workspace_cwd
        self._clock = clock
        self._switch_profile = switch_profile
        self._tiebreak = tiebreak

    def fingerprint(self) -> str:
        """Return the workspace shape this decision is valid for."""
        return workspace_fingerprint(self._cwd)

    async def decide(self, text: str, *, turn: int) -> RouteDecision:
        """Return the track for this turn, consuming any pending override."""
        host = self._host
        override, host._route_override = host._route_override, None
        config = host._agent_profile.routing if host._agent_profile is not None else RoutingConfig()
        signals = extract_prompt_signals(text)
        score, heuristic_reason = prompt_score(signals)
        probed: WorkspaceReadiness | None = None

        def readiness() -> WorkspaceReadiness:
            """Probe the workspace at most once, and only if a branch asks.

            Most turns are standard, and readiness vetoes only the campaign, so
            the common path must not pay for a filesystem walk it will not read.
            """
            nonlocal probed
            if probed is None:
                probed = probe_workspace_readiness(
                    self._cwd,
                    verify_command=host._settings.pact_verify_command,
                    pact_tool_available=self._pact_tool_available(config),
                )
            return probed

        def decide(
            track: RouteTrack,
            band: RouteBand,
            *,
            reason: str,
            confidence: float,
            source: str,
            inherited_from: int | None = None,
            tiebreaker_failure: str = "",
        ) -> RouteDecision:
            effective_band, effective_reason, plan = band, reason, TurnPlan()
            if track is RouteTrack.LONG_HORIZON:
                # Only this branch reads readiness, so only this branch pays
                # for the probe -- the standard track is the common case.
                if band is RouteBand.STRONG_LONG_HORIZON and not readiness().pact_ready:
                    # Still worth the full clarification pass; only the campaign
                    # is impossible, and saying so makes the band explicable.
                    effective_band = RouteBand.LEAN_LONG_HORIZON
                    effective_reason = f"{reason}; pact_not_ready"
                plan = plan_for(effective_band, config.long_horizon, readiness())
            return RouteDecision(
                track=track,
                band=effective_band,
                plan=plan,
                reason=effective_reason,
                confidence=confidence,
                source=source,
                prompt_score=score,
                decided_at=self._clock(),
                archetype=signals.archetype,
                inherited_from_turn=inherited_from,
                tiebreaker_failure=tiebreaker_failure,
            )

        if _acp_depth() > 0 or host._settings.routing_mode == "off":
            return decide(
                RouteTrack.STANDARD,
                RouteBand.STRONG_STANDARD,
                reason="routing disabled in this context",
                confidence=1.0,
                source="guard",
            )
        if override is not None and override.track == RouteTrack.LONG_HORIZON.value:
            return decide(
                RouteTrack.LONG_HORIZON,
                RouteBand.STRONG_LONG_HORIZON,
                reason="user override",
                confidence=1.0,
                source="override",
            )
        if override is not None and override.track == RouteTrack.STANDARD.value:
            return decide(
                RouteTrack.STANDARD,
                RouteBand.STRONG_STANDARD,
                reason="user override",
                confidence=1.0,
                source="override",
            )
        mode = "always" if host._settings.routing_mode == "always" else config.mode
        if mode == "off":
            return decide(
                RouteTrack.STANDARD,
                RouteBand.STRONG_STANDARD,
                reason="profile routing off",
                confidence=1.0,
                source="profile",
            )
        if mode == "always":
            return decide(
                RouteTrack.LONG_HORIZON,
                RouteBand.STRONG_LONG_HORIZON,
                reason="profile routing always",
                confidence=1.0,
                source="profile",
            )
        # "thanks" / "继续" after a long-horizon turn is an acknowledgement, not
        # a continuation of it, and inheriting there is the expensive mistake.
        if signals.archetype in {"trivial", "read_only"} and signals.word_count < _TRIVIAL_FOLLOW_UP_WORDS:
            return decide(
                RouteTrack.STANDARD,
                RouteBand.STRONG_STANDARD,
                reason="trivial follow-up",
                confidence=1.0,
                source="heuristic",
            )
        previous = host._last_route
        reroute = override is not None and override.reroute
        if config.inherit and previous is not None and not reroute and self._may_inherit(previous, signals, config):
            inherited_from = previous.inherited_from_turn or turn - 1
            return decide(
                previous.track,
                previous.band,
                reason=f"inherited from turn {inherited_from}",
                confidence=previous.confidence,
                source="inherited",
                inherited_from=inherited_from,
            )
        band = RouteBand.UNCERTAIN if config.classifier == "llm" else band_for(score)
        if band is not RouteBand.UNCERTAIN or config.classifier == "heuristic":
            long_horizon = band in {RouteBand.LEAN_LONG_HORIZON, RouteBand.STRONG_LONG_HORIZON}
            return decide(
                RouteTrack.LONG_HORIZON if long_horizon else RouteTrack.STANDARD,
                band if long_horizon else RouteBand.STRONG_STANDARD,
                reason=heuristic_reason,
                confidence=score if long_horizon else 1.0 - score,
                source="heuristic",
            )
        verdict = await self._verdict(text, signals)
        if verdict.failure:
            return decide(
                RouteTrack.STANDARD,
                RouteBand.LEAN_STANDARD,
                reason=f"tiebreaker {verdict.failure}",
                confidence=0.5,
                source="llm",
                tiebreaker_failure=verdict.failure,
            )
        if verdict.long_horizon and verdict.confidence >= _STRONG_CONFIDENCE:
            return decide(
                RouteTrack.LONG_HORIZON,
                RouteBand.STRONG_LONG_HORIZON,
                reason=verdict.reason,
                confidence=verdict.confidence,
                source="llm",
            )
        if verdict.long_horizon and verdict.confidence >= config.min_confidence:
            return decide(
                RouteTrack.LONG_HORIZON,
                RouteBand.LEAN_LONG_HORIZON,
                reason=verdict.reason,
                confidence=verdict.confidence,
                source="llm",
            )
        return decide(
            RouteTrack.STANDARD,
            RouteBand.LEAN_STANDARD,
            reason=verdict.reason or "tiebreaker says short task",
            confidence=1.0 - verdict.confidence,
            source="llm",
        )

    async def apply(self, decision: RouteDecision) -> RouteDecision:
        """Switch profiles for a long-horizon turn, downgrading if that fails."""
        host = self._host
        if decision.track is not RouteTrack.LONG_HORIZON:
            return decision
        config = host._agent_profile.routing if host._agent_profile is not None else RoutingConfig()
        target = config.target_profile.strip()
        current = host._agent_profile.name if host._agent_profile is not None else ""
        if not target or target == current:
            return decision
        if self._switch_profile is None or not await self._switch_profile(target):
            await host._bus.publish(
                Warning(
                    code="route_profile_switch_failed",
                    message=f"Could not switch to routing target profile {target!r}; running the standard pass",
                    session_id=host._session_id,
                )
            )
            return _as_standard(decision)
        return _with_switch(decision, target)

    async def publish(self, decision: RouteDecision, *, turn: int) -> None:
        """Record the decision as this session's latest and announce it."""
        host = self._host
        host._last_route = decision
        host._route_fingerprint = self.fingerprint()
        await host._bus.publish(
            TurnRouted(
                session_id=host._session_id,
                turn=turn,
                track=decision.track.value,
                band=decision.band.value,
                reason=decision.reason,
                confidence=decision.confidence,
                source=decision.source,
                inherited=decision.source == "inherited",
                prompt_score=decision.prompt_score,
                plan_localization=decision.plan.localization,
                plan_clarification=decision.plan.clarification,
                plan_pact=decision.plan.pact,
                pact_ready=decision.plan.pact,
                tiebreaker_failure=decision.tiebreaker_failure,
                switched_to=decision.switched_to,
                can_downgrade=decision.track is RouteTrack.LONG_HORIZON,
            )
        )

    # ------------------------------------------------------------------

    def _pact_tool_available(self, config: RoutingConfig) -> bool:
        """Whether the profile that will RUN this turn can hand work to a campaign.

        Not the current profile: routing a long-horizon turn switches to
        ``target_profile``, and the campaign tool deliberately lives only
        there — ``Code`` does not carry ``chrys_pact``. Asking the profile we
        are about to switch away from made every campaign unreachable from the
        default profile, which is the entire path this track exists to take.
        """
        pact_tool = config.long_horizon.pact_tool
        target = config.target_profile.strip()
        if target:
            registry = self._host._agent_registry
            profile = registry.get(target) if registry is not None else None
            if profile is not None:
                return any((ref.tool_name or ref.profile) == pact_tool for ref in profile.sub_agents.agents)
        tools = self._host._sub_agent_tools
        return tools is not None and pact_tool in tools.tool_names()

    def _may_inherit(self, previous: RouteDecision, signals: PromptSignals, config: RoutingConfig) -> bool:
        """Whether the previous turn's verdict still describes this one."""
        if previous.band is RouteBand.STRONG_LONG_HORIZON:
            # A decision that delegates a campaign is re-earned every time.
            return False
        if self._host._route_fingerprint != self.fingerprint():
            return False
        if self._clock() - previous.decided_at > config.stale_after_seconds:
            return False
        if previous.track is RouteTrack.LONG_HORIZON and signals.archetype == "read_only":
            return False
        return not (previous.track is RouteTrack.STANDARD and signals.archetype == "mutating_broad")

    async def _verdict(self, text: str, signals: PromptSignals) -> TiebreakerVerdict:
        if self._tiebreak is not None:
            return await self._tiebreak(text, signals)
        host = self._host
        profile = self._tiebreaker_profile()
        if profile is None:
            return TiebreakerVerdict(long_horizon=False, confidence=0.0, reason="", failure="unavailable")
        return await LlmRouteClassifier(
            profile,
            guard=host._tiebreaker_guard,
            session_id=host._session_id,
            parent_session_id=host._session_id,
            session_dir=host._session_dir,
        ).classify(text, signals)

    def _tiebreaker_profile(self) -> ModelProfile | None:
        """Prefer the configured cheap profile, else the session's active model."""
        host = self._host
        configured = host._settings.routing_tiebreaker_model_profile.strip()
        if configured:
            from chrys.service.profiles.models.registry import ModelProfileRegistry
            from chrys.service.profiles.models.resolver import resolve_profile_selector

            registry = ModelProfileRegistry()
            registry.load_all()
            resolved = resolve_profile_selector(registry, configured)
            if resolved is not None:
                return resolved
        return host._active_profile


def _as_standard(decision: RouteDecision) -> RouteDecision:
    from dataclasses import replace

    return replace(
        decision,
        track=RouteTrack.STANDARD,
        band=RouteBand.STRONG_STANDARD,
        plan=TurnPlan(),
        switched_to="",
    )


def _with_switch(decision: RouteDecision, target: str) -> RouteDecision:
    from dataclasses import replace

    return replace(decision, switched_to=target)


def _acp_depth() -> int:
    """Return the ACP nesting depth; anything above zero must not route."""
    try:
        return int(os.environ.get(_ACP_DEPTH_ENV, "0").strip() or "0")
    except ValueError:
        return 0
