# Copyright (c) 2026 Chrys. All rights reserved.

"""The route marker a finished long-horizon turn leaves in history."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from chrys.orchestration.engine.run.finalizer import _route_marker
from chrys.service.routing.classifier import RouteBand, RouteDecision, RouteTrack, TurnPlan


def _decision(*, pact: bool = True) -> RouteDecision:
    return RouteDecision(
        track=RouteTrack.LONG_HORIZON,
        band=RouteBand.STRONG_LONG_HORIZON if pact else RouteBand.LEAN_LONG_HORIZON,
        plan=TurnPlan(True, True, pact),
        reason="scope=entire; acceptance=acceptance criteria",
        confidence=0.9,
        source="heuristic",
    )


def _host(decision: RouteDecision | None, campaign: dict[str, Any] | None = None) -> Any:
    return SimpleNamespace(_last_route=decision, _long_horizon_campaign=campaign)


def test_an_unrouted_turn_leaves_no_route_marker() -> None:
    assert _route_marker(_host(None), failed=False, interrupted=False) is None


def test_a_completed_campaign_is_recorded_verbatim() -> None:
    """Only the campaign knows whether it completed; nothing else may claim it."""
    campaign = {"status": "completed", "campaign_id": "c1", "artifact": ".pact/campaigns/c1"}

    marker = _route_marker(_host(_decision(), campaign), failed=False, interrupted=False)

    assert marker is not None
    route = marker["_chrys_route"]
    assert route["campaign"] == campaign
    assert route["baseline"] == "p1"
    assert route["track"] == "long_horizon"
    assert route["band"] == "strong_long_horizon"


def test_a_blocked_campaign_is_recorded_as_blocked() -> None:
    campaign = {"status": "blocked", "campaign_id": "c2", "artifact": ""}

    marker = _route_marker(_host(_decision(), campaign), failed=False, interrupted=False)

    assert marker["_chrys_route"]["campaign"]["status"] == "blocked"


def test_a_turn_without_a_campaign_records_none() -> None:
    marker = _route_marker(_host(_decision(pact=False)), failed=False, interrupted=False)

    assert marker["_chrys_route"]["campaign"] is None
    assert marker["_chrys_route"]["baseline"] == "p1"


@pytest.mark.parametrize(("failed", "interrupted"), [(True, False), (False, True)])
def test_a_failed_or_interrupted_turn_holds_no_baseline(failed: bool, interrupted: bool) -> None:
    marker = _route_marker(_host(_decision(pact=False)), failed=failed, interrupted=interrupted)

    assert marker["_chrys_route"]["baseline"] == "none"


def test_a_standard_turn_is_still_recorded() -> None:
    """Routing telemetry needs the negatives too, or precision cannot be measured."""
    decision = RouteDecision(
        track=RouteTrack.STANDARD,
        band=RouteBand.STRONG_STANDARD,
        plan=TurnPlan(),
        reason="trivial follow-up",
        confidence=1.0,
        source="heuristic",
    )

    marker = _route_marker(_host(decision), failed=False, interrupted=False)

    assert marker["_chrys_route"]["track"] == "standard"
    assert marker["_chrys_route"]["baseline"] == "none"


def test_the_reason_is_bounded() -> None:
    decision = RouteDecision(
        track=RouteTrack.STANDARD,
        band=RouteBand.STRONG_STANDARD,
        plan=TurnPlan(),
        reason="x" * 5000,
        confidence=1.0,
        source="heuristic",
    )

    marker = _route_marker(_host(decision), failed=False, interrupted=False)

    assert len(marker["_chrys_route"]["reason"]) == 400


def test_reserved_marker_keys_cannot_be_overwritten() -> None:
    """Turn identity is the marker's own; an annotation must not corrupt it."""
    from chrys.foundation.models.history_markers import HistoryMarkerKind
    from chrys.kernel import Content, Message
    from chrys.service.context.providers.history import CompressibleHistoryProvider

    state: dict[str, Any] = {"messages": [Message("user", [Content.from_text("hi")])], "turn_counter": 0}

    CompressibleHistoryProvider.insert_marker(
        state, 1, {"_chrys_route": {"track": "standard"}, HistoryMarkerKind.KEY: "not-a-turn"}
    )

    marker = state["messages"][-1]
    assert marker.additional_properties[HistoryMarkerKind.KEY] == HistoryMarkerKind.TURN
    assert marker.additional_properties["_chrys_route"] == {"track": "standard"}
