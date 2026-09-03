# Copyright (c) 2026 Chrys. All rights reserved.

"""The long-horizon track: clarification with a code search running beside it.

Localization reads the frozen S0 view rather than the live workspace. By the
time clarification starts, the baseline pass has already edited the workspace,
so a search over the live tree would be describing the baseline's guesses back
to the repair that is supposed to correct them.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chrys.foundation.events.types import LongHorizonPhaseChanged, Warning
from chrys.orchestration.engine.run.workflow_extensions import RepairOutcome
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.routing.delegation import augment_delta_with_locations, localization_hints
from chrys.service.semantic_search import SemanticSearchConfig, SemanticSearchMode, localize_requirement
from chrys.service.semantic_search.localization_model import resolve_localization_model_profile

if TYPE_CHECKING:
    from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot
    from chrys.service.requirement_clarification.types import RequirementRevision
    from chrys.service.routing.classifier import RouteDecision

logger = logging.getLogger(__name__)

# Localization runs beside clarification, so it may not outlast it. Two minutes
# is generous for a bounded graph walk and short enough that a stuck search
# never becomes the turn's critical path.
LOCALIZATION_TIMEOUT_SECONDS = 120.0


class LongHorizonPhase:
    """Phases the long-horizon track adds to the clarification workflow's own."""

    LOCALIZING = "localizing"
    MERGING = "merging"
    DELEGATING = "delegating"
    COMPLETED = "completed"
    DEGRADED = "degraded"
    INTERRUPTED = "interrupted"


@dataclass
class LocalizationOutcome:
    """What the parallel search produced, or why it produced nothing."""

    locations: list[dict[str, Any]] = field(default_factory=list)
    warning: str = ""

    @property
    def available(self) -> bool:
        return bool(self.locations)


class LongHorizonExtensions:
    """Clarification-workflow extensions for a routed long-horizon turn."""

    def __init__(self, host: Any, decision: RouteDecision) -> None:
        self._host = host
        self._decision = decision
        self._workflow_id = ""
        self.localization = LocalizationOutcome()
        self._task: asyncio.Task[None] | None = None

    # -- RequirementWorkflowExtensions ---------------------------------

    def wants_delegation_pass(self) -> bool:
        """Whether a PACT delegation pass will produce this turn's final answer."""
        return self._decision.plan.pact

    async def on_clarification_start(self, revision: RequirementRevision, s0: WorkspaceSnapshot) -> None:
        """Search the frozen S0 view while clarification runs against it."""
        if not self._decision.plan.localization:
            return
        await self._phase(LongHorizonPhase.LOCALIZING, "searching the frozen workspace")
        self._task = asyncio.create_task(self._localize(revision, s0), name="long-horizon-localization")
        try:
            await asyncio.wait_for(asyncio.shield(self._task), LOCALIZATION_TIMEOUT_SECONDS)
        except TimeoutError:
            self._task.cancel()
            await self._degrade(f"code localization exceeded {LOCALIZATION_TIMEOUT_SECONDS:g} seconds")
        except asyncio.CancelledError:
            self._task.cancel()
            raise
        finally:
            self._task = None

    def augment_repair_reminder(self, delta_text: str) -> str:
        """Extend the repair guidance with the search's candidate locations."""
        if not self.localization.available:
            return delta_text
        return augment_delta_with_locations(delta_text, self.localization.locations)

    def pact_input_hints(self) -> str:
        """Return the search's candidates as untrusted evidence for the plan."""
        if not self.localization.available:
            return ""
        return localization_hints(self.localization.locations)

    async def after_repair(self, outcome: RepairOutcome) -> None:
        """Hand the repaired baseline to a PACT campaign when one is warranted."""
        await self._phase(
            LongHorizonPhase.COMPLETED if outcome.status == "succeeded" else LongHorizonPhase.DEGRADED,
            f"baseline={outcome.baseline}",
            terminal=True,
        )

    async def on_revision(self, revision: RequirementRevision) -> None:
        """An amendment invalidates the search: the requirement it ran on changed."""
        await self.cancel()
        self.localization = LocalizationOutcome()

    async def cancel(self) -> None:
        """Abandon a search still in flight."""
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()

    # -- internals -----------------------------------------------------

    async def _localize(self, revision: RequirementRevision, s0: WorkspaceSnapshot) -> None:
        host = self._host
        model_profile = self._localization_profile()
        if model_profile is None:
            await self._degrade("no model profile for code localization")
            return
        artifact_dir = self._artifact_dir()
        if artifact_dir is None:
            await self._degrade("no session directory for localization artifacts")
            return
        view_root = _primary_view_root(s0)
        if view_root is None:
            await self._degrade("the frozen workspace view is unavailable")
            return
        try:
            result = await asyncio.to_thread(
                localize_requirement,
                view_root,
                revision.rendered,
                artifact_dir=artifact_dir,
                config=SemanticSearchConfig(mode=SemanticSearchMode.AUTO),
                model_profile=model_profile,
                client=self._client(model_profile),
                session_id=host._session_id,
                parent_session_id=host._session_id,
                session_dir=host._session_dir,
            )
        except Exception as exc:
            await self._degrade(f"code localization failed: {exc}")
            return
        self.localization = LocalizationOutcome(locations=list(result.locations))
        for warning in result.warnings:
            logger.info("localization degraded: %s", warning)
        await self._phase(
            LongHorizonPhase.MERGING,
            f"{len(self.localization.locations)} candidate location(s)",
        )

    def _localization_profile(self) -> Any:
        registry = ModelProfileRegistry()
        registry.load_all()
        return resolve_localization_model_profile(self._host._settings, registry, self._host._active_profile)

    def _client(self, profile: Any) -> Any:
        host = self._host
        cache = host._side_call_clients
        return cache.get(
            profile,
            session_id=host._session_id,
            parent_session_id=host._session_id,
            session_dir=host._session_dir,
        )

    def _artifact_dir(self) -> Path | None:
        session_dir = self._host._session_dir
        if session_dir is None:
            return None
        return session_dir / "long_horizon" / f"turn_{self._host._turn_number + 1}" / "semantic-search"

    async def _degrade(self, detail: str) -> None:
        """Record a localization failure without failing the turn.

        The repair still runs; it just runs on the clarification alone, which
        is exactly what the standard clarification track already does.
        """
        self.localization = LocalizationOutcome(warning=detail)
        logger.info("long-horizon localization degraded: %s", detail)
        await self._host._bus.publish(
            Warning(
                code="long_horizon_localization_failed",
                message=detail,
                session_id=self._host._session_id,
            )
        )

    async def _phase(self, phase: str, detail: str = "", *, terminal: bool = False) -> None:
        await self._host._bus.publish(
            LongHorizonPhaseChanged(
                workflow_id=self._workflow_id,
                phase=phase,
                detail=detail,
                terminal=terminal,
                session_id=self._host._session_id,
            )
        )


def _primary_view_root(snapshot: WorkspaceSnapshot) -> Path | None:
    """Return the frozen view of the primary workspace root."""
    for root in snapshot.roots:
        if root.is_primary:
            return Path(root.view_root)
    return Path(snapshot.roots[0].view_root) if snapshot.roots else None
