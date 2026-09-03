# Copyright (c) 2026 Chrys. All rights reserved.

"""The search's results merged into the repair reminder, brief, and plan hints."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.orchestration.engine.run.long_horizon import LocalizationOutcome, LongHorizonExtensions
from chrys.orchestration.engine.run.workflow_extensions import RepairOutcome
from chrys.service.llm.side_call_clients import SideCallClientCache
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.routing.classifier import RouteBand, RouteDecision, RouteTrack, TurnPlan


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


def _decision(*, pact: bool = True) -> RouteDecision:
    return RouteDecision(
        track=RouteTrack.LONG_HORIZON,
        band=RouteBand.STRONG_LONG_HORIZON,
        plan=TurnPlan(True, True, pact),
        reason="scope",
        confidence=0.9,
        source="heuristic",
    )


def _located() -> list[dict[str, Any]]:
    return [
        {"file": "src/auth/provider.py", "symbol": "exchange", "role": "primary", "reason": "token exchange"},
        {"file": "src/api/routes.py", "symbol": "login", "role": "propagation", "reason": "entry point"},
    ]


@pytest.fixture
def extensions(tmp_path: Path) -> LongHorizonExtensions:
    host = _Host(_session_dir=tmp_path / "session")
    instance = LongHorizonExtensions(host, _decision())
    instance._requirement = "Add OAuth login"
    return instance


def test_the_repair_reminder_keeps_the_delta_then_adds_the_table(
    extensions: LongHorizonExtensions,
) -> None:
    extensions.localization = LocalizationOutcome(locations=_located())

    reminder = extensions.augment_repair_reminder("ΔR: use the provider abstraction.")

    assert reminder.startswith("ΔR: use the provider abstraction.")
    assert "src/auth/provider.py" in reminder
    assert "untrusted" in reminder.lower()


def test_the_plan_hints_carry_the_same_locations(extensions: LongHorizonExtensions) -> None:
    extensions.localization = LocalizationOutcome(locations=_located())

    hints = extensions.pact_input_hints()

    assert "src/auth/provider.py" in hints
    assert "src/api/routes.py" in hints


def test_asking_for_plan_hints_writes_the_brief(extensions: LongHorizonExtensions, tmp_path: Path) -> None:
    """The plan may reference the brief, so it has to exist by then."""
    extensions.localization = LocalizationOutcome(locations=_located())

    extensions.pact_input_hints()

    brief = tmp_path / "session" / "long_horizon" / "turn_1" / "brief.md"
    assert brief.is_file()
    assert "Add OAuth login" in brief.read_text(encoding="utf-8")
    assert "src/auth/provider.py" in brief.read_text(encoding="utf-8")


def test_the_brief_records_the_baseline_after_the_repair(extensions: LongHorizonExtensions, tmp_path: Path) -> None:
    extensions.localization = LocalizationOutcome(locations=_located())
    extensions.pact_input_hints()

    import asyncio

    asyncio.run(extensions.after_repair(RepairOutcome("succeeded", "final", "p1")))

    brief = (tmp_path / "session" / "long_horizon" / "turn_1" / "brief.md").read_text(encoding="utf-8")
    assert "The workspace holds: p1" in brief


def test_the_brief_names_a_degraded_search(extensions: LongHorizonExtensions, tmp_path: Path) -> None:
    extensions.localization = LocalizationOutcome(warning="code localization exceeded 120 seconds")

    extensions.write_brief(baseline="p0")

    brief = (tmp_path / "session" / "long_horizon" / "turn_1" / "brief.md").read_text(encoding="utf-8")
    assert "exceeded 120 seconds" in brief
    assert "(no candidate locations)" in brief


def test_the_brief_folds_in_the_clarified_requirement(extensions: LongHorizonExtensions, tmp_path: Path) -> None:
    outcome = tmp_path / "session" / "requirement_clarification" / "turn_1" / "05-outcome"
    outcome.mkdir(parents=True)
    (outcome / "clarified-requirement.md").write_text("## Clarified\nSupport Google only.", encoding="utf-8")

    extensions.write_brief(baseline="p1")

    brief = (tmp_path / "session" / "long_horizon" / "turn_1" / "brief.md").read_text(encoding="utf-8")
    assert "Support Google only." in brief


def test_no_session_directory_means_no_brief_and_no_crash() -> None:
    extensions = LongHorizonExtensions(_Host(_session_dir=None), _decision())

    assert extensions.write_brief(baseline="p1") is None
