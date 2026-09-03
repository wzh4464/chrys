# Copyright (c) 2026 Chrys. All rights reserved.

"""``session/route_override``, the runtime route payload, and turn_routed."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from acp import RequestError

from chrys.foundation.events.types import LongHorizonPhaseChanged, RouteOverride, TurnRouted
from chrys.service.routing.classifier import RouteBand, RouteDecision, RouteTrack, TurnPlan


def _decision() -> RouteDecision:
    return RouteDecision(
        track=RouteTrack.LONG_HORIZON,
        band=RouteBand.STRONG_LONG_HORIZON,
        plan=TurnPlan(True, True, True),
        reason="scope=entire",
        confidence=0.9,
        source="heuristic",
    )


async def test_route_override_reaches_the_session_bus(acp_server) -> None:
    server, manager = acp_server

    await server.ext_method("session/route_override", {"sessionId": "s1", "track": "long_horizon"})

    manager.route_override.assert_awaited_once()
    assert manager.route_override.await_args.kwargs == {"track": "long_horizon", "reroute": False}


async def test_route_override_accepts_a_bare_reroute(acp_server) -> None:
    server, manager = acp_server

    await server.ext_method("session/route_override", {"sessionId": "s1", "reroute": True})

    assert manager.route_override.await_args.kwargs == {"track": "", "reroute": True}


@pytest.mark.parametrize("track", ["sideways", "LONG_HORIZON", "quick"])
async def test_an_unknown_track_is_rejected(acp_server, track: str) -> None:
    server, manager = acp_server

    with pytest.raises(RequestError):
        await server.ext_method("session/route_override", {"sessionId": "s1", "track": track})

    manager.route_override.assert_not_awaited()


async def test_the_runtime_payload_reports_the_last_decision(acp_server) -> None:
    server, manager = acp_server
    manager.get.return_value.host.engine.last_route = _decision()
    manager.get.return_value.host.engine.settings.routing_mode = "auto"

    payload = server._runtime_payload("s1")

    assert payload["route"]["mode"] == "auto"
    assert payload["route"]["last"]["track"] == "long_horizon"
    assert payload["route"]["last"]["canDowngrade"] is True


async def test_an_unrouted_session_reports_no_last_decision(acp_server) -> None:
    server, manager = acp_server
    manager.get.return_value.host.engine.last_route = None

    payload = server._runtime_payload("s1")

    # None, not a fabricated "standard": the client must be able to tell
    # "not classified yet" from "classified as standard".
    assert payload["route"]["last"] is None


async def test_turn_routed_is_forwarded_as_an_extension_notification(acp_server) -> None:
    server, _manager = acp_server
    client = server._client_or_error()

    handled = await server._handle_chrys_extension_event(
        "s1",
        TurnRouted(track="long_horizon", band="strong_long_horizon", reason="scope", can_downgrade=True),
    )

    assert handled is True
    method, payload = client.ext_notification.await_args.args
    assert method == "chrys/turn_routed"
    assert payload["track"] == "long_horizon"
    assert payload["canDowngrade"] is True


async def test_long_horizon_phase_is_forwarded(acp_server) -> None:
    server, _manager = acp_server
    client = server._client_or_error()

    handled = await server._handle_chrys_extension_event(
        "s1", LongHorizonPhaseChanged(phase="localizing", detail="12 files", terminal=False)
    )

    assert handled is True
    method, payload = client.ext_notification.await_args.args
    assert method == "chrys/long_horizon_phase"
    assert payload["phase"] == "localizing"


def test_the_override_event_is_the_engine_facing_shape() -> None:
    """The manager publishes the same event the TUI does."""
    event = RouteOverride(track="standard", reroute=False, session_id="s1")

    assert event.one_shot is True
    assert event.plan_localization is None


@pytest.fixture
def acp_server(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    from chrys.app.acp import server as server_module

    manager = MagicMock()
    manager.route_override = AsyncMock()
    session = MagicMock()
    session.profile_name = "Code"
    engine = session.host.engine
    engine.last_route = None
    engine.settings.routing_mode = "auto"
    manager.get.return_value = session

    instance = object.__new__(server_module.ChrysAcpServer)
    instance._manager = manager
    client = MagicMock()
    client.ext_notification = AsyncMock()
    instance._client_or_error = lambda: client
    return instance, manager
