# Copyright (c) 2026 Chrys. All rights reserved.

"""Run one PACT Campaign with fresh in-process Chrys role adapters."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from acp import schema as acp_schema
from acp.helpers import start_tool_call, update_tool_call
from pact_core.adapters.base import AgentAdapter
from pact_core.runtime import (
    AdapterDecisionProvider,
    AdapterPlanningProvider,
    CampaignControlPlane,
    CampaignRunRequest,
    CampaignRunResult,
)
from pact_core.runtime.dashboard import load_dashboard_projection

from chrys.app.acp.bridge import SessionUpdate

if TYPE_CHECKING:
    from chrys.foundation.config.settings_store import LoadedSettings

UpdateSender = Callable[[SessionUpdate], Awaitable[None]]
ProjectionLoader = Callable[[Path, str], dict[str, object]]
SemanticRole = Literal["worker", "reviewer", "planner", "manager"]
_SEMANTIC_ROLES: tuple[SemanticRole, ...] = ("worker", "reviewer", "planner", "manager")


class CancellableAgentAdapter(AgentAdapter, Protocol):
    """PACT adapter extended with the invocation-level cancellation hook."""

    async def cancel_current_turn(self) -> None:
        """Interrupt the currently active Chrys role turn, if any."""
        ...


class RoleAdapterFactory(Protocol):
    """Construct a role-bound in-process Chrys adapter."""

    def __call__(
        self,
        *,
        semantic_role: SemanticRole,
        profile_name: str,
        loaded_settings: LoadedSettings,
        outer_loop: asyncio.AbstractEventLoop,
        campaign_id: str,
        send_update: UpdateSender,
        abort_event: threading.Event,
    ) -> CancellableAgentAdapter: ...


class CampaignRunner(Protocol):
    """Narrow injectable surface of the PACT Control Plane."""

    def run(self, request: CampaignRunRequest) -> CampaignRunResult: ...


class CampaignCancelled(Exception):
    """The outer ACP invocation was cancelled without changing canonical status."""


class _ProjectionReporter:
    """Project canonical Work State revisions onto the existing Campaign card."""

    def __init__(
        self,
        *,
        workspace: Path,
        campaign_id: str,
        campaign_tool_id: str,
        send_update: UpdateSender,
        loader: ProjectionLoader,
    ) -> None:
        self._workspace = workspace
        self._campaign_id = campaign_id
        self._campaign_tool_id = campaign_tool_id
        self._send_update = send_update
        self._loader = loader
        self._last_revision: int | None = None

    async def refresh(self) -> None:
        """Emit at most one update for each canonical Work State revision."""
        try:
            projection = await asyncio.to_thread(self._loader, self._workspace, self._campaign_id)
        except Exception:
            # The dashboard is a rebuildable display projection, never a control signal.
            return
        if projection.get("availability") != "ready":
            return
        source = projection.get("source")
        if not isinstance(source, dict):
            return
        revision = source.get("work_state_revision")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision == self._last_revision:
            return

        overview = projection.get("overview")
        frontier = projection.get("frontier")
        overview = overview if isinstance(overview, dict) else {}
        frontier = frontier if isinstance(frontier, dict) else {}
        status = overview.get("status", "unknown")
        plan_revision = overview.get("plan_revision", "unknown")
        next_action = overview.get("next_action", "unknown")
        selected = frontier.get("selected") or "none"
        summary = (
            f"PACT status: {status}; revision: {revision}; plan: {plan_revision}; "
            f"next_action: {next_action}; mission: {selected}"
        )
        await self._send_update(
            update_tool_call(
                self._campaign_tool_id,
                status="in_progress",
                raw_output=summary,
            )
        )
        self._last_revision = revision


def _is_role_boundary(update: SessionUpdate) -> bool:
    return isinstance(update, acp_schema.ToolCallStart | acp_schema.ToolCallProgress) and update.tool_call_id.endswith(
        ":role"
    )


@dataclass(frozen=True, slots=True)
class CampaignTerminal:
    """Bounded terminal projection returned to the Primary Chrys client."""

    status: str
    campaign_id: str
    revision: int
    next_action: str
    artifact_ref: str

    @property
    def completed(self) -> bool:
        """Whether PACT's canonical Work State reached completion."""
        return self.status == "completed"

    def summary_text(self) -> str:
        """Render the stable last ACP message segment."""
        return "\n".join(
            (
                "PACT Campaign result",
                f"status: {self.status}",
                f"campaign_id: {self.campaign_id}",
                f"revision: {self.revision}",
                f"next_action: {self.next_action}",
                f"artifacts: {self.artifact_ref}",
            )
        )


class CampaignCoordinator:
    """Own the single PACT thread and the four fresh Chrys role adapters."""

    def __init__(
        self,
        *,
        profile_name: str,
        loaded_settings: LoadedSettings,
        verify_command: str | None,
        allow_unverified: bool,
        max_rounds: int = 3,
        control_plane: CampaignRunner | None = None,
        adapter_factory: RoleAdapterFactory | None = None,
        projection_loader: ProjectionLoader = load_dashboard_projection,
        worktree_root: Path | None = None,
    ) -> None:
        self._profile_name = profile_name
        self._loaded_settings = loaded_settings
        self._verify_command = verify_command
        self._allow_unverified = allow_unverified
        self._max_rounds = max_rounds
        self._control_plane = control_plane or CampaignControlPlane()
        self._adapter_factory = adapter_factory or _default_adapter_factory
        self._projection_loader = projection_loader
        self._worktree_root = worktree_root
        self._abort_event = threading.Event()
        self._cancel_requested = threading.Event()
        self._adapters: tuple[CancellableAgentAdapter, ...] = ()
        self._thread: threading.Thread | None = None
        self._started = False

    async def run(
        self,
        *,
        workspace: Path,
        contract_file: Path,
        plan_file: Path,
        send_update: UpdateSender,
    ) -> CampaignTerminal:
        """Run exactly one Campaign without blocking the ACP event loop."""
        if self._started:
            raise RuntimeError("CampaignCoordinator accepts exactly one run.")
        self._started = True
        if self._cancel_requested.is_set():
            raise CampaignCancelled("PACT invocation was cancelled before it started.")

        workspace = workspace.resolve(strict=True)
        campaign_id = f"chrys-pact-{uuid.uuid4().hex}"
        campaign_tool_id = f"{campaign_id}/campaign"
        await send_update(
            start_tool_call(
                campaign_tool_id,
                "Run PACT Campaign",
                kind="execute",
                status="in_progress",
                raw_input={"campaign_id": campaign_id},
            )
        )
        outer_loop = asyncio.get_running_loop()
        projection_reporter = _ProjectionReporter(
            workspace=workspace,
            campaign_id=campaign_id,
            campaign_tool_id=campaign_tool_id,
            send_update=send_update,
            loader=self._projection_loader,
        )

        async def send_role_update(update: SessionUpdate) -> None:
            if _is_role_boundary(update):
                await projection_reporter.refresh()
            await send_update(update)

        adapters = tuple(
            self._adapter_factory(
                semantic_role=role,
                profile_name=self._profile_name,
                loaded_settings=self._loaded_settings,
                outer_loop=outer_loop,
                campaign_id=campaign_id,
                send_update=send_role_update,
                abort_event=self._abort_event,
            )
            for role in _SEMANTIC_ROLES
        )
        self._adapters = adapters
        worker, reviewer, planner, manager = adapters
        request = CampaignRunRequest(
            workspace=workspace,
            contract_file=contract_file,
            plan_file=plan_file,
            worker=worker,
            reviewer=reviewer,
            verify_command=self._verify_command,
            allow_unverified=self._allow_unverified,
            max_rounds=self._max_rounds,
            campaign_id=campaign_id,
            worktree_root=self._worktree_root,
            planning_provider=AdapterPlanningProvider(planner),
            decision_provider=AdapterDecisionProvider(manager),
        )

        try:
            result = await self._run_control_plane(request)
        except asyncio.CancelledError:
            # Preserve structured task cancellation from the ACP SDK/transport.
            # Invocation-level session/cancel reaches the Exception path via
            # the role adapter and is mapped to CampaignCancelled below.
            raise
        except Exception:
            message = (
                "Invocation cancelled; canonical PACT artifacts, if any, were preserved."
                if self._cancel_requested.is_set()
                else "PACT Campaign failed; inspect canonical artifacts for details."
            )
            with contextlib.suppress(Exception):
                await send_update(update_tool_call(campaign_tool_id, status="failed", raw_output=message))
            if self._cancel_requested.is_set():
                raise CampaignCancelled(message) from None
            raise

        if self._cancel_requested.is_set():
            message = "Invocation cancelled; canonical PACT artifacts, if any, were preserved."
            # Suppressed like the failure path above: a broken channel is the
            # usual companion of a cancel, and letting the notification's own
            # error escape reports the run as a refusal instead of a cancel.
            with contextlib.suppress(Exception):
                await send_update(update_tool_call(campaign_tool_id, status="failed", raw_output=message))
            raise CampaignCancelled(message)

        await projection_reporter.refresh()
        terminal = _terminal_from_result(result, workspace=workspace)
        await send_update(
            update_tool_call(
                campaign_tool_id,
                status="completed" if terminal.completed else "failed",
                raw_output=terminal.summary_text(),
            )
        )
        return terminal

    async def cancel(self) -> None:
        """Best-effort abort of the active invocation; not a Work State transition."""
        self._cancel_requested.set()
        self._abort_event.set()
        for adapter in self._adapters:
            with contextlib.suppress(Exception):
                await adapter.cancel_current_turn()

    async def wait_closed(self, timeout: float | None = None) -> bool:
        """Wait for the owned Control Plane thread; report whether it settled.

        The default is unbounded, which is what a caller waiting for a campaign
        to deliver its result wants. A caller whose transport has already gone
        away passes a bound instead: ``cancel()`` only sets a flag that
        pact_core's deterministic verify subprocess never reads, so an
        unbounded wait there holds the process open for the whole verify
        timeout with nobody left to receive the answer.
        """
        thread = self._thread
        if thread is None or not thread.is_alive():
            return True
        await asyncio.to_thread(thread.join, timeout)
        return not thread.is_alive()

    async def _run_control_plane(self, request: CampaignRunRequest) -> CampaignRunResult:
        loop = asyncio.get_running_loop()
        result_future: asyncio.Future[CampaignRunResult] = loop.create_future()

        def consume_result_exception(future: asyncio.Future[CampaignRunResult]) -> None:
            # Keep the cross-thread completion future observed even if its
            # original waiter is cancelled by the ACP transport.
            if not future.cancelled():
                future.exception()

        result_future.add_done_callback(consume_result_exception)

        def settle_result(result: CampaignRunResult | None, error: BaseException | None) -> None:
            if result_future.done():
                return
            if error is not None:
                result_future.set_exception(error)
            elif result is not None:
                result_future.set_result(result)
            else:
                result_future.set_exception(RuntimeError("PACT Control Plane ended without a result."))

        def run_control_plane() -> None:
            try:
                result = self._control_plane.run(request)
            except BaseException as exc:
                loop.call_soon_threadsafe(settle_result, None, exc)
            else:
                loop.call_soon_threadsafe(settle_result, result, None)

        thread = threading.Thread(
            target=run_control_plane,
            name=f"chrys-pact-{request.campaign_id}",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        try:
            return await asyncio.shield(result_future)
        except asyncio.CancelledError:
            # Python 3.14's shield installs an exception-reporting callback
            # when its outer waiter is cancelled. This future is only a
            # transport for the owned thread result, so cancel it before the
            # thread settles and let wait_closed() own thread termination.
            result_future.cancel()
            await self.cancel()
            raise
        finally:
            if not thread.is_alive():
                thread.join()


def _default_adapter_factory(
    *,
    semantic_role: SemanticRole,
    profile_name: str,
    loaded_settings: LoadedSettings,
    outer_loop: asyncio.AbstractEventLoop,
    campaign_id: str,
    send_update: UpdateSender,
    abort_event: threading.Event,
) -> CancellableAgentAdapter:
    """Delay the Chrys runtime import until CLI bootstrap has completed."""
    from chrys.pact.role_runner import InProcessChrysAdapter

    return InProcessChrysAdapter(
        semantic_role=semantic_role,
        profile_name=profile_name,
        loaded_settings=loaded_settings,
        outer_loop=outer_loop,
        campaign_id=campaign_id,
        send_update=send_update,
        abort_event=abort_event,
    )


def _terminal_from_result(result: CampaignRunResult, *, workspace: Path) -> CampaignTerminal:
    try:
        artifact_ref = result.campaign_dir.resolve().relative_to(workspace).as_posix()
    except ValueError as exc:
        raise RuntimeError("PACT Campaign artifact directory escaped the workspace.") from exc
    return CampaignTerminal(
        status=result.status,
        campaign_id=result.campaign_id,
        revision=result.revision,
        next_action=result.next_action,
        artifact_ref=artifact_ref,
    )
