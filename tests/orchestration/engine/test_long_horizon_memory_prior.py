# Copyright (c) 2026 Chrys. All rights reserved.

"""Prior experience recalled for a campaign's Initial Plan."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import Warning
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
    _workspace: object | None = None

    @property
    def _settings(self) -> Settings:
        return self.settings


def _decision() -> RouteDecision:
    return RouteDecision(
        track=RouteTrack.LONG_HORIZON,
        band=RouteBand.STRONG_LONG_HORIZON,
        plan=TurnPlan(localization=False, clarification=True, pact=True),
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


async def _hints(instance: LongHorizonExtensions) -> str:
    """Drive the real ordering: recall during clarification, render afterwards."""
    await instance.on_clarification_start(_revision(instance._requirement), None)  # type: ignore[arg-type]
    return instance.pact_input_hints()


def _revision(rendered: str) -> Any:
    return SimpleNamespace(rendered=rendered)


async def test_a_recalled_prior_reaches_the_plan_hints(extensions: LongHorizonExtensions) -> None:
    with patch(
        "chrys.service.memory.contextgraph_mcp.query_prior",
        return_value="Strategy: migrate callers before deleting the old path.",
    ):
        hints = await _hints(extensions)

    assert "Prior experience from the team graph (untrusted)" in hints
    assert "migrate callers" in hints


async def test_the_prior_sits_beside_the_located_code(extensions: LongHorizonExtensions) -> None:
    with patch("chrys.service.memory.contextgraph_mcp.query_prior", return_value="Strategy: X"):
        await extensions.on_clarification_start(_revision(extensions._requirement), None)  # type: ignore[arg-type]
    extensions.localization = LocalizationOutcome(locations=[{"file": "src/auth.py", "role": "primary"}])

    hints = extensions.pact_input_hints()

    assert hints.index("src/auth.py") < hints.index("Prior experience")


async def test_the_recall_happens_before_the_hints_are_rendered(extensions: LongHorizonExtensions) -> None:
    """Rendering must be pure: a blocking Bolt query there would stall the session."""
    with patch("chrys.service.memory.contextgraph_mcp.query_prior", side_effect=AssertionError("queried too late")):
        assert extensions.pact_input_hints() == ""


async def test_an_empty_recall_adds_nothing(extensions: LongHorizonExtensions) -> None:
    with patch(
        "chrys.service.memory.contextgraph_mcp.query_prior",
        return_value="No prior ContextGraph memory found.",
    ):
        hints = await _hints(extensions)

    assert "Prior experience" not in hints


async def test_an_unreachable_graph_is_silent(extensions: LongHorizonExtensions) -> None:
    """A machine that never configured a graph is the normal case, not an error."""
    warnings: list[object] = []

    async def _collect(event: Warning) -> None:
        warnings.append(event)

    await extensions._host._bus.subscribe(Warning, _collect)

    with patch(
        "chrys.service.memory.contextgraph_mcp.query_prior",
        side_effect=RuntimeError("neo4j unreachable"),
    ):
        hints = await _hints(extensions)

    assert "Prior experience" not in hints
    assert warnings == []


async def test_a_hanging_graph_does_not_hold_up_the_turn(extensions: LongHorizonExtensions) -> None:
    """The query is blocking, so only a thread plus a timeout can bound it."""

    release = threading.Event()

    def _hang(*_args: object) -> str:
        release.wait(30)
        return "never read"

    try:
        with (
            patch("chrys.orchestration.engine.run.long_horizon.MEMORY_PRIOR_TIMEOUT_SECONDS", 0.05),
            patch("chrys.service.memory.contextgraph_mcp.query_prior", side_effect=_hang),
        ):
            hints = await _hints(extensions)
    finally:
        # Abandoning the thread is the production behaviour; leaving it asleep
        # would make the loop's executor shutdown wait it out.
        release.set()

    assert "Prior experience" not in hints


async def test_memory_being_off_skips_the_recall_entirely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", _URI)
    host = _Host(_session_dir=tmp_path / "session", settings=Settings(memory_mcp_enabled=False))
    instance = LongHorizonExtensions(host, _decision())
    instance._requirement = "Add OAuth login"
    calls: list[object] = []

    with patch("chrys.service.memory.contextgraph_mcp.query_prior", side_effect=lambda *a, **k: calls.append((a, k))):
        await _hints(instance)

    assert calls == []


async def test_no_graph_configured_skips_the_recall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTEXTGRAPH_NEO4J_URI", raising=False)
    instance = LongHorizonExtensions(_Host(_session_dir=tmp_path / "session"), _decision())
    instance._requirement = "Add OAuth login"
    calls: list[object] = []

    with patch("chrys.service.memory.contextgraph_mcp.query_prior", side_effect=lambda *a, **k: calls.append((a, k))):
        await _hints(instance)

    assert calls == []


async def test_the_prior_is_bounded(extensions: LongHorizonExtensions) -> None:
    """A plan prompt has a budget it shares with the clarification evidence."""
    with patch("chrys.service.memory.contextgraph_mcp.query_prior", return_value="x" * 50_000):
        hints = await _hints(extensions)

    assert len(hints) <= MEMORY_PRIOR_MAX_CHARS + 200


async def test_an_empty_requirement_never_queries(extensions: LongHorizonExtensions) -> None:
    extensions._requirement = "   "
    calls: list[Any] = []

    with patch("chrys.service.memory.contextgraph_mcp.query_prior", side_effect=lambda *a, **k: calls.append((a, k))):
        await _hints(extensions)

    assert calls == []


async def test_the_recall_is_scoped_to_the_workspaces_repository(
    tmp_path: Path, extensions: LongHorizonExtensions
) -> None:
    repo = tmp_path / "parser-kit"
    repo.mkdir()
    # Outside git the label is the workspace directory itself.
    extensions._host._workspace = SimpleNamespace(primary_cwd=str(repo))
    calls: list[tuple[tuple, dict]] = []

    def fake_query(*args, **kwargs):
        calls.append((args, kwargs))
        return "Canonical rules:\n- migrate callers"

    with patch("chrys.service.memory.contextgraph_mcp.query_prior", side_effect=fake_query):
        await extensions.on_clarification_start(_revision("add typed parsing"), None)  # type: ignore[arg-type]

    assert [kwargs["repo"] for _args, kwargs in calls] == ["parser-kit"]
    assert extensions._memory_prior_status.startswith("recalled ")
    assert "'parser-kit'" in extensions._memory_prior_status
