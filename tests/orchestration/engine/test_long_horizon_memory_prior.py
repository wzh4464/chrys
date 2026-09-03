# Copyright (c) 2026 Chrys. All rights reserved.

"""Prior experience recalled for a campaign's Initial Plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.orchestration.engine.run.long_horizon import (
    MEMORY_PRIOR_MAX_CHARS,
    LocalizationOutcome,
    LongHorizonExtensions,
)
from chrys.service.llm.side_call_clients import SideCallClientCache
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.routing.classifier import RouteBand, RouteDecision, RouteTrack, TurnPlan

_URI = "bolt://127.0.0.1:7687"


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


def _decision() -> RouteDecision:
    return RouteDecision(
        track=RouteTrack.LONG_HORIZON,
        band=RouteBand.STRONG_LONG_HORIZON,
        plan=TurnPlan(True, True, True),
        reason="scope",
        confidence=0.9,
        source="heuristic",
    )


@pytest.fixture
def extensions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LongHorizonExtensions:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    instance = LongHorizonExtensions(_Host(_session_dir=tmp_path / "session"), _decision())
    instance._requirement = "Add OAuth login"
    return instance


def test_a_recalled_prior_reaches_the_plan_hints(extensions: LongHorizonExtensions) -> None:
    with patch(
        "chrys.service.memory.contextgraph_mcp._do_query",
        return_value="Strategy: migrate callers before deleting the old path.",
    ):
        hints = extensions.pact_input_hints()

    assert "Prior experience from the team graph (untrusted)" in hints
    assert "migrate callers" in hints


def test_the_prior_sits_beside_the_located_code(extensions: LongHorizonExtensions) -> None:
    extensions.localization = LocalizationOutcome(locations=[{"file": "src/auth.py", "role": "primary"}])

    with patch("chrys.service.memory.contextgraph_mcp._do_query", return_value="Strategy: X"):
        hints = extensions.pact_input_hints()

    assert hints.index("src/auth.py") < hints.index("Prior experience")


def test_an_empty_recall_adds_nothing(extensions: LongHorizonExtensions) -> None:
    with patch(
        "chrys.service.memory.contextgraph_mcp._do_query",
        return_value="No prior ContextGraph memory found.",
    ):
        hints = extensions.pact_input_hints()

    assert "Prior experience" not in hints


def test_an_unreachable_graph_is_silent(extensions: LongHorizonExtensions) -> None:
    """A machine that never configured a graph is the normal case, not an error."""
    warnings: list[object] = []
    import asyncio

    from chrys.foundation.events.types import Warning

    asyncio.run(extensions._host._bus.subscribe(Warning, warnings.append))

    with patch(
        "chrys.service.memory.contextgraph_mcp._do_query",
        side_effect=RuntimeError("neo4j unreachable"),
    ):
        hints = extensions.pact_input_hints()

    assert "Prior experience" not in hints
    assert warnings == []


def test_memory_being_off_skips_the_recall_entirely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    host = _Host(_session_dir=tmp_path / "session", settings=Settings(memory_mcp_enabled=False))
    instance = LongHorizonExtensions(host, _decision())
    instance._requirement = "Add OAuth login"
    calls: list[object] = []

    with patch("chrys.service.memory.contextgraph_mcp._do_query", side_effect=lambda *a: calls.append(a)):
        instance.pact_input_hints()

    assert calls == []


def test_no_graph_configured_skips_the_recall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXTGRAPH_NEO4J_URI", raising=False)
    instance = LongHorizonExtensions(_Host(_session_dir=tmp_path / "session"), _decision())
    instance._requirement = "Add OAuth login"
    calls: list[object] = []

    with patch("chrys.service.memory.contextgraph_mcp._do_query", side_effect=lambda *a: calls.append(a)):
        instance.pact_input_hints()

    assert calls == []


def test_the_prior_is_bounded(extensions: LongHorizonExtensions) -> None:
    """A plan prompt has a budget it shares with the clarification evidence."""
    with patch("chrys.service.memory.contextgraph_mcp._do_query", return_value="x" * 50_000):
        hints = extensions.pact_input_hints()

    assert len(hints) <= MEMORY_PRIOR_MAX_CHARS + 200


def test_an_empty_requirement_never_queries(extensions: LongHorizonExtensions) -> None:
    extensions._requirement = "   "
    calls: list[Any] = []

    with patch("chrys.service.memory.contextgraph_mcp._do_query", side_effect=lambda *a: calls.append(a)):
        extensions.pact_input_hints()

    assert calls == []
