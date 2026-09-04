# Copyright (c) 2026 Chrys. All rights reserved.

"""Localization runs beside clarification, against the frozen S0 view."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import Warning
from chrys.orchestration.engine.run import long_horizon as module
from chrys.orchestration.engine.run.long_horizon import LongHorizonExtensions
from chrys.service.llm.side_call_clients import SideCallClientCache
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.routing.classifier import RouteBand, RouteDecision, RouteTrack, TurnPlan


@dataclass
class _Root:
    view_root: str
    is_primary: bool = True


@dataclass
class _Snapshot:
    roots: list[_Root] = field(default_factory=list)


@dataclass
class _Revision:
    rendered: str = "add oauth login"
    number: int = 1


@dataclass
class _Host:
    _bus: EventBus = field(default_factory=EventBus)
    _session_id: str | None = "sess"
    _session_dir: Path | None = None
    _turn_number: int = 0
    _active_profile: ModelProfile | None = field(default_factory=lambda: ModelProfile(id="m", name="M"))
    _side_call_clients: SideCallClientCache = field(default_factory=SideCallClientCache)
    settings: Settings = field(default_factory=Settings)

    @property
    def _settings(self) -> Settings:
        return self.settings


def _decision(*, localization: bool = True, pact: bool = False) -> RouteDecision:
    return RouteDecision(
        track=RouteTrack.LONG_HORIZON,
        band=RouteBand.STRONG_LONG_HORIZON if pact else RouteBand.LEAN_LONG_HORIZON,
        plan=TurnPlan(localization=localization, clarification=True, pact=pact),
        reason="scope",
        confidence=0.9,
        source="heuristic",
    )


class _Result:
    def __init__(self, locations: list[dict[str, Any]]) -> None:
        self.locations = locations
        self.warnings: list[str] = []


@pytest.fixture
def frozen(tmp_path: Path) -> tuple[_Host, _Snapshot]:
    view = tmp_path / "view"
    view.mkdir()
    host = _Host(_session_dir=tmp_path / "session")
    return host, _Snapshot(roots=[_Root(view_root=str(view))])


async def test_localization_reads_the_frozen_view_not_the_live_workspace(
    frozen: tuple[_Host, _Snapshot], monkeypatch: pytest.MonkeyPatch
) -> None:
    """By clarification time the baseline pass has already edited the live tree."""
    host, snapshot = frozen
    seen: dict[str, Any] = {}

    async def _localize(repo: Any, requirement: str, **kwargs: Any) -> _Result:
        seen["repo"] = repo
        seen["requirement"] = requirement
        seen.update(kwargs)
        return _Result([{"file": "src/auth.py", "role": "primary"}])

    monkeypatch.setattr(module, "localize_requirement_async", _localize)
    extensions = LongHorizonExtensions(host, _decision())

    await extensions.on_clarification_start(_Revision(), snapshot)

    assert str(seen["repo"]) == snapshot.roots[0].view_root
    assert seen["requirement"] == "add oauth login"
    assert extensions.localization.locations == [{"file": "src/auth.py", "role": "primary"}]


async def test_the_search_is_told_which_model_to_use(
    frozen: tuple[_Host, _Snapshot], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline resolves the profile itself; the track only names it."""
    host, snapshot = frozen
    seen: dict[str, Any] = {}

    async def _localize(*_args: Any, **kwargs: Any) -> _Result:
        seen.update(kwargs)
        return _Result([])

    monkeypatch.setattr(module, "localize_requirement_async", _localize)
    host.settings = Settings(semantic_search_model_profile="cheap-model")

    await LongHorizonExtensions(host, _decision()).on_clarification_start(_Revision(), snapshot)

    assert seen["config"].model_profile == "cheap-model"
    assert seen["config"].timeout_seconds == module.LOCALIZATION_TIMEOUT_SECONDS


async def test_a_failing_search_warns_and_leaves_the_turn_running(
    frozen: tuple[_Host, _Snapshot], monkeypatch: pytest.MonkeyPatch
) -> None:
    host, snapshot = frozen
    warnings: list[Warning] = []
    await host._bus.subscribe(Warning, warnings.append)

    async def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("index build failed")

    monkeypatch.setattr(module, "localize_requirement_async", _boom)
    extensions = LongHorizonExtensions(host, _decision())

    await extensions.on_clarification_start(_Revision(), snapshot)

    assert extensions.localization.available is False
    assert warnings and warnings[-1].code == "long_horizon_localization_failed"


async def test_a_plan_without_localization_never_searches(
    frozen: tuple[_Host, _Snapshot], monkeypatch: pytest.MonkeyPatch
) -> None:
    host, snapshot = frozen
    called: list[object] = []

    async def _localize(*args: Any, **_kwargs: Any) -> Any:
        called.append(args)

    monkeypatch.setattr(module, "localize_requirement_async", _localize)

    await LongHorizonExtensions(host, _decision(localization=False)).on_clarification_start(_Revision(), snapshot)

    assert called == []


async def test_a_slow_search_is_bounded_and_degrades(
    frozen: tuple[_Host, _Snapshot], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parallel work may not become the turn's critical path."""
    host, snapshot = frozen

    async def _hang(*_args: Any, **_kwargs: Any) -> Any:
        await asyncio.sleep(5)
        return _Result([])

    monkeypatch.setattr(module, "localize_requirement_async", _hang)
    monkeypatch.setattr(module, "LOCALIZATION_TIMEOUT_SECONDS", 0.05)
    extensions = LongHorizonExtensions(host, _decision())

    await extensions.on_clarification_start(_Revision(), snapshot)

    assert extensions.localization.available is False
    assert "exceeded" in extensions.localization.warning
