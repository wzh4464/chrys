# Copyright (c) 2026 Chrys. All rights reserved.

"""The long-horizon track: clarification with a code search running beside it.

Localization reads the frozen S0 view rather than the live workspace. By the
time clarification starts, the baseline pass has already edited the workspace,
so a search over the live tree would be describing the baseline's guesses back
to the repair that is supposed to correct them.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from chrys.foundation.events.types import LongHorizonPhaseChanged, Warning
from chrys.foundation.platform import safe_getcwd
from chrys.kernel.exchanges import (
    EmptyIdPolicy,
    LiveAccessor,
    NoneIdPolicy,
    PairingPolicy,
    iter_exchanges,
    pair_results,
)
from chrys.orchestration.engine.run.workflow_extensions import RepairOutcome
from chrys.service.llm.json_extract import json_object_candidates, repair_json_object_candidate
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.routing.delegation import (
    PactRunRequest,
    augment_delta_with_locations,
    build_delegation_reminder,
    build_task_brief,
    localization_hints,
    materialize_pact_request,
)
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

# A prior is a hint, not a plan. Three strategies is enough to be useful and
# small enough that it cannot crowd out the clarification evidence beside it.
MEMORY_PRIOR_TOP_K = 3
MEMORY_PRIOR_MAX_CHARS = 2000
# A graph that has not answered in fifteen seconds is not going to make this
# plan better. The recall runs beside clarification, so this budget is spent in
# parallel with work that would have happened anyway.
MEMORY_PRIOR_TIMEOUT_SECONDS = 15.0


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

    def __init__(self, host: Any, decision: RouteDecision, *, runner: Any = None) -> None:
        self._host = host
        self._decision = decision
        # The workflow's own runner: the delegation pass is another executor
        # pass of the same turn, not a new turn with its own runner.
        self._turn_runner = runner
        self._workflow_id = ""
        self.localization = LocalizationOutcome()
        self.brief_path: Path | None = None
        self.request: PactRunRequest | None = None
        self._requirement = ""
        self._memory_prior = ""
        self._task: asyncio.Task[None] | None = None

    # -- RequirementWorkflowExtensions ---------------------------------

    def wants_delegation_pass(self) -> bool:
        """Whether a PACT delegation pass will produce this turn's final answer."""
        return self._decision.plan.pact

    async def on_clarification_start(self, revision: RequirementRevision, s0: WorkspaceSnapshot) -> None:
        """Search the frozen S0 view and recall priors while clarification runs.

        Both are I/O against something outside this process, and both belong
        here rather than at the point of use: this hook is already one leg of
        the workflow's gather, so their latency overlaps work that had to
        happen anyway.
        """
        self._requirement = revision.rendered
        self._memory_prior = await self._recall_prior()
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
        """Return untrusted evidence for the plan: located code, then prior experience.

        Pure: both halves were gathered during clarification. Also the moment
        the task brief first lands, because the plan may reference it and a
        role reading the brief needs it on disk before the campaign runs.
        """
        self.write_brief(baseline="none")
        sections: list[str] = []
        if self.localization.available:
            sections.append(localization_hints(self.localization.locations))
        if self._memory_prior:
            sections.append(f"Prior experience from the team graph (untrusted):\n{self._memory_prior}")
        return "\n\n".join(section for section in sections if section)

    async def _recall_prior(self) -> str:
        """Recall prior experience for this requirement, or nothing at all.

        Off the event loop and on a timeout: the query is a synchronous Bolt
        round trip plus an embedding call, so running it where it is consumed
        would stall the whole session -- and ``asyncio.timeout`` cannot
        interrupt a blocking call, only an await. A timeout abandons the thread
        rather than killing it (nothing can kill a thread), which is safe
        because it only holds a pool slot until the driver gives up.

        Silent on every failure: an unreachable graph is the normal case on a
        machine that never configured one, and a plan is perfectly valid
        without a prior.
        """
        if not self._requirement.strip():
            return ""
        try:
            async with asyncio.timeout(MEMORY_PRIOR_TIMEOUT_SECONDS):
                return await asyncio.to_thread(self._query_prior)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("memory prior unavailable", exc_info=True)
            return ""

    def _query_prior(self) -> str:
        """Ask the graph, bounded. Runs in a worker thread."""
        from chrys.service.memory.contextgraph_mcp import _do_query
        from chrys.service.memory.overlay import memory_mcp_server_config

        if memory_mcp_server_config(self._host._settings) is None:
            return ""
        recalled = _do_query(self._requirement, MEMORY_PRIOR_TOP_K)
        if not isinstance(recalled, str) or "No prior ContextGraph memory found." in recalled:
            return ""
        return recalled.strip()[:MEMORY_PRIOR_MAX_CHARS]

    def write_brief(self, *, baseline: str) -> Path | None:
        """Write the brief the campaign's roles read, and return its path.

        Written even when a stage degraded: a brief that names what is missing
        is more use to a role than no brief at all.
        """
        directory = self._turn_dir()
        if directory is None:
            return None
        warnings = [self.localization.warning] if self.localization.warning else []
        brief = build_task_brief(
            original_requirement=self._requirement,
            clarified_requirement_md=self._clarified_requirement(),
            locations=self.localization.locations,
            baseline=baseline,
            warnings=warnings,
        )
        try:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "brief.md"
            path.write_text(brief, encoding="utf-8")
        except OSError:
            logger.warning("could not write the long-horizon task brief", exc_info=True)
            return None
        self.brief_path = path
        return path

    def _clarified_requirement(self) -> str | None:
        """Read the clarification's own canonical output, when it produced one."""
        session_dir = self._host._session_dir
        if session_dir is None:
            return None
        path = (
            session_dir
            / "requirement_clarification"
            / f"turn_{self._host._turn_number + 1}"
            / "05-outcome"
            / "clarified-requirement.md"
        )
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def _turn_dir(self) -> Path | None:
        session_dir = self._host._session_dir
        if session_dir is None:
            return None
        return session_dir / "long_horizon" / f"turn_{self._host._turn_number + 1}"

    async def after_repair(self, outcome: RepairOutcome) -> None:
        """Hand the repaired baseline to a PACT campaign when one is warranted."""
        brief_path = self.write_brief(baseline=outcome.baseline)
        if not self._can_delegate(outcome):
            await self._phase(
                LongHorizonPhase.COMPLETED if outcome.status == "succeeded" else LongHorizonPhase.DEGRADED,
                f"baseline={outcome.baseline}",
                terminal=True,
            )
            return
        assert outcome.pact_input_dir is not None
        try:
            request = materialize_pact_request(Path(self._workspace_cwd()), outcome.pact_input_dir, uuid4().hex[:12])
        except OSError as exc:
            await self._degrade_delegation(f"could not stage the PACT inputs: {exc}", outcome)
            return
        self.request = request
        reminder = build_delegation_reminder(
            brief_path=brief_path or Path("brief.md"),
            brief_summary=self._brief_summary(),
            baseline=outcome.baseline,
            request=request,
            pact_tool=self._pact_tool(),
        )
        reminder_middleware = self._host._reminder_middleware
        if reminder_middleware is not None:
            reminder_middleware.queue_hook_reminders([reminder])
        await self._phase(LongHorizonPhase.DELEGATING, f"request {request.request_id}")
        if self._turn_runner is None:
            await self._degrade_delegation("no turn runner is available for the delegation pass", outcome)
            return
        try:
            await self._turn_runner._run_fresh_standard(
                self._requirement,
                created_at=None,
                contents=None,
                run_scope=None,
                injection_window=None,
                admission_preparation=None,
                finalize=False,
            )
        except asyncio.CancelledError:
            await self._phase(LongHorizonPhase.INTERRUPTED, "delegation interrupted", terminal=True)
            raise
        except Exception as exc:
            # The repaired baseline is already in the workspace and already
            # answered; a failed hand-off must not discard it.
            await self._degrade_delegation(f"delegation pass failed: {exc}", outcome)
            return
        campaign = self._campaign_result()
        self._host._long_horizon_campaign = campaign
        if campaign is None and self._require_pact():
            # The reminder asked for one call and the model made none. Say so
            # rather than reporting the turn as a governed campaign.
            await self._host._bus.publish(
                Warning(
                    code="long_horizon_delegation_skipped",
                    message="the delegation pass finished without calling the campaign tool",
                    session_id=self._host._session_id,
                )
            )
        await self._phase(
            LongHorizonPhase.COMPLETED,
            f"delegated request {request.request_id}",
            terminal=True,
        )

    def _require_pact(self) -> bool:
        profile = self._host._agent_profile
        return profile is not None and profile.routing.long_horizon.require_pact

    def _campaign_result(self) -> dict[str, Any] | None:
        """Read the campaign's own reported outcome from the delegation pass.

        Parsed from the tool result rather than inferred: only the campaign
        knows whether it completed, and reporting anything else as completed is
        exactly the failure the governance layer exists to prevent.
        """
        executor = self._host._executor
        messages = executor.history_state.get("messages") if executor.history_state else None
        if not isinstance(messages, list):
            return None
        tool_name = self._pact_tool()
        for text in reversed(list(_tool_results(messages, tool_name))):
            payload = _loads(text)
            if payload is not None and isinstance(payload.get("status"), str):
                return {
                    "status": payload["status"],
                    "campaign_id": str(payload.get("campaign_id") or ""),
                    "artifact": str(payload.get("artifact") or ""),
                }
        return None

    def _can_delegate(self, outcome: RepairOutcome) -> bool:
        """Whether there is both an accepted PACT pair and a plan that wants one."""
        return self.wants_delegation_pass() and outcome.pact_input_dir is not None

    async def _degrade_delegation(self, detail: str, outcome: RepairOutcome) -> None:
        """Fall back to the repaired text as this turn's answer."""
        logger.warning("long-horizon delegation degraded: %s", detail)
        await self._host._bus.publish(
            Warning(
                code="long_horizon_delegation_failed",
                message=detail,
                session_id=self._host._session_id,
            )
        )
        self._host._executor.adopt_fallback_success(outcome.final_text)
        await self._phase(LongHorizonPhase.DEGRADED, detail, terminal=True)

    def _pact_tool(self) -> str:
        profile = self._host._agent_profile
        return profile.routing.long_horizon.pact_tool if profile is not None else "chrys_pact"

    def _workspace_cwd(self) -> str:
        workspace = self._host._workspace
        return workspace.primary_cwd if workspace is not None else safe_getcwd()

    def _brief_summary(self, *, max_chars: int = 1200) -> str:
        """Return the first part of the brief, for a reminder that cannot hold it all."""
        if self.brief_path is None:
            return "(no task brief was written)"
        try:
            return self.brief_path.read_text(encoding="utf-8")[:max_chars]
        except OSError:
            return "(the task brief could not be read)"

    async def on_revision(self, revision: RequirementRevision) -> None:
        """An amendment invalidates both the search and the prior.

        Each was gathered for the requirement text that just changed. The
        re-run's ``on_clarification_start`` refills them; clearing here is what
        makes sure a superseded hint cannot outlive that.
        """
        await self.cancel()
        self.localization = LocalizationOutcome()
        self._memory_prior = ""
        self._requirement = revision.rendered

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


_PAIRING_POLICY = PairingPolicy(
    call_types=LiveAccessor().call_types(),
    include_informational_calls=False,
    result_types=LiveAccessor().result_types(),
    none_id=NoneIdPolicy.POSITIONAL,
    empty_id=EmptyIdPolicy.POSITIONAL,
    malformed_id="stringify",
)


def _tool_results(messages: list[Any], tool_name: str) -> Iterator[str]:
    """Yield the text of every result answering a call to *tool_name*.

    Walks the canonical exchange grammar rather than scanning for call ids:
    ids repeat across exchanges, so a global scan pairs the wrong ones.
    """
    accessor = LiveAccessor()
    for exchange in iter_exchanges(messages, accessor):
        pairing = pair_results(messages, exchange, accessor, _PAIRING_POLICY)
        for assignments in (*pairing.truthy_assignments.values(), *pairing.falsy_assignments.values()):
            for call_occurrence, result_occurrence in assignments:
                if result_occurrence is None:
                    continue
                call = accessor.contents(messages[call_occurrence.message_index])[call_occurrence.content_index]
                if getattr(call, "name", "") != tool_name:
                    continue
                result = accessor.contents(messages[result_occurrence.message_index])[result_occurrence.content_index]
                yield str(getattr(result, "result", "") or "")


def _loads(text: str) -> dict[str, Any] | None:
    for candidate in json_object_candidates(text):
        try:
            payload = json.loads(repair_json_object_candidate(candidate))
        except ValueError, RecursionError:
            continue
        if isinstance(payload, dict):
            return payload
    return None
