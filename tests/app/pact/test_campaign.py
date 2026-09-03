# Copyright (c) 2026 Chrys. All rights reserved.

"""Campaign thread, role wiring, progress, and cancellation tests."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from pact_core.runtime import CampaignRunRequest, CampaignRunResult
from pact_core.schemas import AdapterProbe, TurnRequest, TurnResult

from chrys.app.acp.bridge import SessionUpdate
from chrys.foundation.config.settings_store import LoadedSettings
from chrys.pact.campaign import (
    CampaignCancelled,
    CampaignCoordinator,
    SemanticRole,
    UpdateSender,
    _ProjectionReporter,
)


class _FakeAdapter:
    def __init__(self, semantic_role: str) -> None:
        self.id = f"fake-{semantic_role}"
        self.semantic_role = semantic_role
        self.cancelled = 0

    def probe(self) -> AdapterProbe:
        return AdapterProbe(adapter_id=self.id, available=True, version="test")

    def run_turn(self, request: TurnRequest) -> TurnResult:
        raise AssertionError(f"unexpected role turn: {request.role}")

    async def cancel_current_turn(self) -> None:
        self.cancelled += 1


class _FakeControlPlane:
    def __init__(self, workspace: Path, *, status: str = "completed") -> None:
        self.workspace = workspace
        self.status = status
        self.request: CampaignRunRequest | None = None
        self.thread: threading.Thread | None = None

    def run(self, request: CampaignRunRequest) -> CampaignRunResult:
        self.request = request
        self.thread = threading.current_thread()
        return CampaignRunResult(
            campaign_id=request.campaign_id or "missing",
            campaign_dir=self.workspace / ".pact" / "runtime" / "campaigns" / (request.campaign_id or "missing"),
            status=cast(Any, self.status),
            revision=3,
            next_action="none" if self.status == "completed" else "manager_decide",
            execution_outcomes=(),
            promotion_receipts=(),
            evidence_refs=(),
        )


def _inputs(workspace: Path) -> tuple[Path, Path]:
    contract = workspace / "goal.json"
    plan = workspace / "plan.json"
    contract.write_text("{}", encoding="utf-8")
    plan.write_text("{}", encoding="utf-8")
    return contract, plan


class _AdapterFactory:
    def __init__(self, *, observe_abort: Callable[[threading.Event], None] | None = None) -> None:
        self.roles: list[str] = []
        self.adapters: list[_FakeAdapter] = []
        self._observe_abort = observe_abort

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
    ) -> _FakeAdapter:
        _ = profile_name, loaded_settings, outer_loop, campaign_id, send_update
        if self._observe_abort is not None:
            self._observe_abort(abort_event)
        self.roles.append(semantic_role)
        adapter = _FakeAdapter(semantic_role)
        self.adapters.append(adapter)
        return adapter


class _UpdateCollector:
    def __init__(self) -> None:
        self.updates: list[SessionUpdate] = []

    async def __call__(self, update: SessionUpdate) -> None:
        self.updates.append(update)


async def test_projection_reporter_deduplicates_canonical_work_state_revision(tmp_path: Path) -> None:
    collector = _UpdateCollector()
    projection: dict[str, object] = {
        "availability": "ready",
        "source": {"work_state_revision": 2},
        "overview": {
            "status": "active",
            "plan_revision": 1,
            "next_action": "worker_execute",
        },
        "frontier": {"selected": "mission-1"},
    }
    reporter = _ProjectionReporter(
        workspace=tmp_path,
        campaign_id="campaign-test",
        campaign_tool_id="campaign-test/campaign",
        send_update=collector,
        loader=lambda _workspace, _campaign_id: projection,
    )

    await reporter.refresh()
    await reporter.refresh()
    cast(dict[str, object], projection["source"])["work_state_revision"] = 3
    await reporter.refresh()

    assert len(collector.updates) == 2
    assert collector.updates[0].status == "in_progress"
    assert "revision: 2" in cast(str, collector.updates[0].raw_output)
    assert "revision: 3" in cast(str, collector.updates[1].raw_output)


async def test_coordinator_runs_control_plane_on_daemon_thread_and_wires_four_roles(tmp_path: Path) -> None:
    contract, plan = _inputs(tmp_path)
    control_plane = _FakeControlPlane(tmp_path)
    adapter_factory = _AdapterFactory()
    collector = _UpdateCollector()

    coordinator = CampaignCoordinator(
        profile_name="Code",
        loaded_settings=cast(LoadedSettings, object()),
        verify_command="pytest",
        allow_unverified=False,
        control_plane=control_plane,
        adapter_factory=adapter_factory,
        worktree_root=tmp_path / "pact-worktrees",
    )

    terminal = await coordinator.run(
        workspace=tmp_path,
        contract_file=contract,
        plan_file=plan,
        send_update=collector,
    )

    assert adapter_factory.roles == ["worker", "reviewer", "planner", "manager"]
    assert control_plane.thread is not threading.current_thread()
    assert control_plane.thread is not None and control_plane.thread.daemon
    assert control_plane.request is not None
    assert control_plane.request.worker is adapter_factory.adapters[0]
    assert control_plane.request.reviewer is adapter_factory.adapters[1]
    assert control_plane.request.planning_provider is not None
    assert control_plane.request.decision_provider is not None
    assert control_plane.request.verify_command == "pytest"
    assert control_plane.request.worktree_root == tmp_path / "pact-worktrees"
    assert terminal.completed
    assert collector.updates[0].status == "in_progress"
    assert collector.updates[-1].status == "completed"


class _BlockingControlPlane(_FakeControlPlane):
    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace, status="active")
        self.started = threading.Event()
        self.abort_event: threading.Event | None = None

    def run(self, request: CampaignRunRequest) -> CampaignRunResult:
        self.started.set()
        assert self.abort_event is not None
        assert self.abort_event.wait(timeout=5)
        return super().run(request)


class _FailingAfterAbortControlPlane(_BlockingControlPlane):
    def run(self, request: CampaignRunRequest) -> CampaignRunResult:
        self.started.set()
        assert self.abort_event is not None
        assert self.abort_event.wait(timeout=5)
        raise RuntimeError("Control Plane failed after invocation cancellation.")


class _FailingAfterInternalAbortControlPlane(_FakeControlPlane):
    def __init__(self, workspace: Path) -> None:
        super().__init__(workspace, status="active")
        self.abort_event: threading.Event | None = None

    def run(self, request: CampaignRunRequest) -> CampaignRunResult:
        assert self.abort_event is not None
        self.abort_event.set()
        raise RuntimeError("Role runtime became unresponsive.")


async def test_cancel_interrupts_all_adapters_and_never_reports_completion(tmp_path: Path) -> None:
    contract, plan = _inputs(tmp_path)
    control_plane = _BlockingControlPlane(tmp_path)

    def observe_abort(abort_event: threading.Event) -> None:
        control_plane.abort_event = abort_event

    adapter_factory = _AdapterFactory(observe_abort=observe_abort)
    collector = _UpdateCollector()

    coordinator = CampaignCoordinator(
        profile_name="Code",
        loaded_settings=cast(LoadedSettings, object()),
        verify_command=None,
        allow_unverified=True,
        control_plane=control_plane,
        adapter_factory=adapter_factory,
    )
    run_task = asyncio.create_task(
        coordinator.run(
            workspace=tmp_path,
            contract_file=contract,
            plan_file=plan,
            send_update=collector,
        )
    )
    assert await asyncio.to_thread(control_plane.started.wait, 5)

    await coordinator.cancel()

    with pytest.raises(CampaignCancelled):
        await run_task
    assert all(adapter.cancelled == 1 for adapter in adapter_factory.adapters)
    assert collector.updates[-1].status == "failed"


async def test_explicit_cancel_maps_control_plane_failure_to_campaign_cancelled(tmp_path: Path) -> None:
    contract, plan = _inputs(tmp_path)
    control_plane = _FailingAfterAbortControlPlane(tmp_path)

    def observe_abort(abort_event: threading.Event) -> None:
        control_plane.abort_event = abort_event

    adapter_factory = _AdapterFactory(observe_abort=observe_abort)
    collector = _UpdateCollector()
    coordinator = CampaignCoordinator(
        profile_name="Code",
        loaded_settings=cast(LoadedSettings, object()),
        verify_command=None,
        allow_unverified=True,
        control_plane=control_plane,
        adapter_factory=adapter_factory,
    )
    run_task = asyncio.create_task(
        coordinator.run(
            workspace=tmp_path,
            contract_file=contract,
            plan_file=plan,
            send_update=collector,
        )
    )
    assert await asyncio.to_thread(control_plane.started.wait, 5)

    await coordinator.cancel()

    with pytest.raises(CampaignCancelled, match="Invocation cancelled"):
        await run_task
    assert all(adapter.cancelled == 1 for adapter in adapter_factory.adapters)
    assert collector.updates[-1].status == "failed"
    assert str(collector.updates[-1].raw_output).startswith("Invocation cancelled")


async def test_internal_abort_preserves_control_plane_failure_diagnostic(tmp_path: Path) -> None:
    contract, plan = _inputs(tmp_path)
    control_plane = _FailingAfterInternalAbortControlPlane(tmp_path)

    def observe_abort(abort_event: threading.Event) -> None:
        control_plane.abort_event = abort_event

    collector = _UpdateCollector()
    coordinator = CampaignCoordinator(
        profile_name="Code",
        loaded_settings=cast(LoadedSettings, object()),
        verify_command=None,
        allow_unverified=True,
        control_plane=control_plane,
        adapter_factory=_AdapterFactory(observe_abort=observe_abort),
    )

    with pytest.raises(RuntimeError, match="Role runtime became unresponsive"):
        await coordinator.run(
            workspace=tmp_path,
            contract_file=contract,
            plan_file=plan,
            send_update=collector,
        )

    assert collector.updates[-1].status == "failed"
    assert collector.updates[-1].raw_output == "PACT Campaign failed; inspect canonical artifacts for details."


async def test_task_cancellation_remains_structured_asyncio_cancellation(tmp_path: Path) -> None:
    contract, plan = _inputs(tmp_path)
    control_plane = _BlockingControlPlane(tmp_path)

    def observe_abort(abort_event: threading.Event) -> None:
        control_plane.abort_event = abort_event

    adapter_factory = _AdapterFactory(observe_abort=observe_abort)
    coordinator = CampaignCoordinator(
        profile_name="Code",
        loaded_settings=cast(LoadedSettings, object()),
        verify_command=None,
        allow_unverified=True,
        control_plane=control_plane,
        adapter_factory=adapter_factory,
    )
    run_task = asyncio.create_task(
        coordinator.run(
            workspace=tmp_path,
            contract_file=contract,
            plan_file=plan,
            send_update=_UpdateCollector(),
        )
    )
    assert await asyncio.to_thread(control_plane.started.wait, 5)

    run_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await run_task
    await coordinator.wait_closed()
    assert all(adapter.cancelled == 1 for adapter in adapter_factory.adapters)
    assert coordinator._thread is not None and not coordinator._thread.is_alive()


async def test_task_cancellation_consumes_late_control_plane_failure(tmp_path: Path) -> None:
    contract, plan = _inputs(tmp_path)
    control_plane = _FailingAfterAbortControlPlane(tmp_path)

    def observe_abort(abort_event: threading.Event) -> None:
        control_plane.abort_event = abort_event

    coordinator = CampaignCoordinator(
        profile_name="Code",
        loaded_settings=cast(LoadedSettings, object()),
        verify_command=None,
        allow_unverified=True,
        control_plane=control_plane,
        adapter_factory=_AdapterFactory(observe_abort=observe_abort),
    )
    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    run_task: asyncio.Task[Any] | None = None
    try:
        run_task = asyncio.create_task(
            coordinator.run(
                workspace=tmp_path,
                contract_file=contract,
                plan_file=plan,
                send_update=_UpdateCollector(),
            )
        )
        assert await asyncio.to_thread(control_plane.started.wait, 5)

        run_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await run_task
    finally:
        await coordinator.cancel()
        if run_task is not None and not run_task.done():
            run_task.cancel()
            await asyncio.gather(run_task, return_exceptions=True)
        await coordinator.wait_closed()
        callbacks_drained = asyncio.Event()
        loop.call_soon(callbacks_drained.set)
        await callbacks_drained.wait()
        loop.set_exception_handler(previous_handler)

    assert loop_errors == []
    assert coordinator._thread is not None and not coordinator._thread.is_alive()


async def test_wait_closed_can_be_bounded_when_the_client_is_already_gone() -> None:
    """`cancel()` only sets a flag pact_core's verify subprocess never reads.

    An unbounded wait there holds the process open for the whole verify
    timeout with nobody left to receive the answer.
    """
    import threading

    coordinator = CampaignCoordinator(
        profile_name="Code",
        loaded_settings=None,
        verify_command="true",
        allow_unverified=True,
    )
    release = threading.Event()
    thread = threading.Thread(target=lambda: release.wait(30), daemon=True)
    thread.start()
    coordinator._thread = thread
    try:
        assert await coordinator.wait_closed(0.05) is False
    finally:
        release.set()
        thread.join(timeout=5)

    assert await coordinator.wait_closed(5) is True
