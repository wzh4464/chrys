# Copyright (c) 2026 Chrys. All rights reserved.

"""Opt-in baseline, clarification, and fresh-repair turn orchestration."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from chrys.foundation.events.types import RequirementClarificationPhaseChanged, UserInjectResult, Warning
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.kernel import Message
from chrys.service.agent_middleware.injection import ConsumedInjection, InjectionAnchor
from chrys.service.requirement_clarification.artifacts import ClarificationArtifactStore
from chrys.service.requirement_clarification.model import ChrysClarificationModel
from chrys.service.requirement_clarification.service import ClarificationService
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot, WorkspaceSnapshotter
from chrys.service.requirement_clarification.types import RequirementRevision, RequirementWorkflowPhase
from chrys.service.state.serializers import serialize_state

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.models.workspace import Workspace
    from chrys.orchestration.engine.executor import Executor
    from chrys.orchestration.engine.run.runner import TurnRunner
    from chrys.orchestration.engine.run.turn_state import CurrentRunInjectionWindow, CurrentRunScope, TurnRuntimeState
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
    from chrys.service.profiles.models.schema import ModelProfile
    from chrys.service.session.history import SessionHistoryManager
    from chrys.service.trajectory.preparation import PreparationTrace


logger = logging.getLogger(__name__)


class RequirementClarificationHost(Protocol):
    """Engine surface used by one requirement-clarification workflow."""

    _active_profile: ModelProfile | None
    _bus: EventBus
    _consumed_injections: list[ConsumedInjection]
    _executor: Executor
    _history: SessionHistoryManager
    _intermediate_texts: dict[int, str]
    _agent_profile_fingerprint: str
    _model_profile_fingerprint: str
    _reminder_middleware: SystemReminderMiddleware | None
    _session_id: str | None
    _turn_number: int
    _turn_state: TurnRuntimeState
    _workspace: Workspace | None
    _requirement_clarification_workflow: RequirementClarificationWorkflow | None

    @property
    def _session_dir(self) -> Path | None: ...

    def _accumulate_side_call_usage(self, usage_details: dict[str, Any]) -> None: ...


class RequirementClarificationWorkflow:
    """Keep P0 deliverable while deriving ΔR exclusively from frozen S0."""

    def __init__(
        self,
        host: RequirementClarificationHost,
        runner: TurnRunner,
        *,
        initial_timeout_seconds: float = 5400.0,
        repair_timeout_seconds: float = 5400.0,
    ) -> None:
        self._host = host
        self._runner = runner
        self._workflow_id = uuid4().hex
        self._snapshotter = WorkspaceSnapshotter()
        self._phase_name = RequirementWorkflowPhase.SNAPSHOT
        self._revision = RequirementRevision(number=1, messages=())
        self._late_injections: list[ConsumedInjection] = []
        self._stop_requested = False
        self._artifacts: ClarificationArtifactStore | None = None
        self._s0: WorkspaceSnapshot | None = None
        self._p0: WorkspaceSnapshot | None = None
        self._initial_timeout_seconds = initial_timeout_seconds
        self._repair_timeout_seconds = repair_timeout_seconds

    @property
    def accepts_amendments(self) -> bool:
        """Return whether a user message can revise this live workflow."""
        return self._phase_name in {
            RequirementWorkflowPhase.CLARIFICATION,
            RequirementWorkflowPhase.REPAIR,
        }

    async def accept_amendment(
        self,
        text: str,
        *,
        created_at: datetime | str | None,
        injection_id: str | None,
    ) -> bool:
        """Append user authority verbatim and invalidate downstream work."""
        if not self.accepts_amendments or not text:
            return False
        self._revision = self._revision.append(text)
        self._late_injections.append(
            ConsumedInjection(
                text=text,
                anchor=InjectionAnchor(),
                created_at=created_at,
                injection_id=injection_id,
                consumption_id=injection_id or uuid4().hex,
                analytics_item_id=new_analytics_id(),
            )
        )
        if self._phase_name == RequirementWorkflowPhase.REPAIR and self._host._executor.is_running:
            await self._host._executor.interrupt()
        await self._host._bus.publish(
            UserInjectResult(
                text=text,
                consumed=True,
                created_at=created_at,
                injection_id=injection_id,
                session_id=self._host._session_id,
            )
        )
        return True

    async def request_stop(self) -> None:
        """Stop the active phase and retain P0 whenever it already exists."""
        self._stop_requested = True
        if self._host._executor.is_running:
            await self._host._executor.interrupt()

    async def run(
        self,
        text: str,
        *,
        created_at: datetime | str | None,
        contents: list[Any],
        run_scope: CurrentRunScope | None,
        injection_window: CurrentRunInjectionWindow | None,
        admission_preparation: PreparationTrace | None,
    ) -> None:
        host = self._host
        executor = host._executor
        workspace = host._workspace
        model_profile = host._active_profile
        session_dir = host._session_dir
        if workspace is None or model_profile is None or session_dir is None:
            await self._degraded("clarification runtime is unavailable")
            executor.set_requirement_phase("")
            await self._runner._run_fresh_standard(
                text,
                created_at=created_at,
                contents=contents,
                run_scope=run_scope,
                injection_window=injection_window,
                admission_preparation=admission_preparation,
            )
            return

        revision = RequirementRevision(number=1, messages=(text,))
        self._revision = revision
        try:
            artifacts = ClarificationArtifactStore(session_dir, host._turn_number + 1)
            self._artifacts = artifacts
            h0 = executor.snapshot_history()
            history_background = _history_background(executor.history_state)
            artifacts.save_history_checkpoint(
                {
                    "history": serialize_state(executor.history_state),
                    "service_session_id": executor.service_session_id,
                }
            )
            await self._phase(RequirementWorkflowPhase.SNAPSHOT, revision.number)
        except Exception as exc:
            logger.warning("Requirement clarification checkpoint setup failed", exc_info=True)
            await host._bus.publish(
                Warning(
                    code="requirement_clarification_degraded",
                    message=f"workflow checkpoint setup failed: {exc}",
                    session_id=host._session_id,
                )
            )
            executor.set_requirement_phase("")
            await self._runner._run_fresh_standard(
                text,
                created_at=created_at,
                contents=contents,
                run_scope=run_scope,
                injection_window=injection_window,
                admission_preparation=admission_preparation,
            )
            return
        s0: WorkspaceSnapshot | None = None
        p0: WorkspaceSnapshot | None = None
        try:
            s0 = await asyncio.to_thread(
                self._snapshotter.capture,
                workspace,
                artifacts.root / "s0",
                snapshot_id=f"{self._workflow_id}-s0",
                include_git_history=True,
            )
            self._s0 = s0
        except Exception as exc:
            logger.warning("Requirement clarification S0 capture failed", exc_info=True)
            await self._degraded(f"S0 capture failed: {exc}")
            executor.set_requirement_phase("")
            await self._runner._run_fresh_standard(
                text,
                created_at=created_at,
                contents=contents,
                run_scope=run_scope,
                injection_window=injection_window,
                admission_preparation=admission_preparation,
            )
            return

        host._requirement_clarification_workflow = self
        try:
            await self._phase(RequirementWorkflowPhase.INITIAL_IMPLEMENTATION, revision.number)
            executor.set_requirement_phase(RequirementWorkflowPhase.INITIAL_IMPLEMENTATION)
            try:
                async with asyncio.timeout(self._initial_timeout_seconds):
                    await self._runner._run_fresh_standard(
                        text,
                        created_at=created_at,
                        contents=contents,
                        run_scope=run_scope,
                        injection_window=injection_window,
                        admission_preparation=admission_preparation,
                        finalize=False,
                    )
            except TimeoutError:
                if executor.is_running:
                    await executor.interrupt()
                executor.set_requirement_phase("")
                await self._phase(
                    RequirementWorkflowPhase.INTERRUPTED,
                    revision.number,
                    detail=f"initial implementation exceeded {self._initial_timeout_seconds:g} seconds",
                    terminal=True,
                )
                await self._runner.finalize_current_run()
                return
            if executor.run_failed or executor.was_interrupted:
                executor.set_requirement_phase("")
                await self._phase(
                    RequirementWorkflowPhase.INTERRUPTED
                    if executor.was_interrupted
                    else RequirementWorkflowPhase.DEGRADED,
                    revision.number,
                    terminal=True,
                )
                await self._runner.finalize_current_run()
                return

            p0_text = executor.last_response_text
            p0_history = executor.snapshot_history()
            baseline_injections = list(host._consumed_injections)
            for injection in baseline_injections:
                revision = revision.append(injection.text)
            self._revision = revision
            try:
                artifacts.save_initial_transcript(
                    {
                        "history": serialize_state(executor.history_state),
                        "service_session_id": executor.service_session_id,
                    }
                )
            except OSError as exc:
                logger.warning("Requirement clarification P0 transcript persistence failed", exc_info=True)
                await self._deliver_p0(p0_text, revision, detail=f"P0 transcript persistence failed: {exc}")
                return
            try:
                p0 = await asyncio.to_thread(
                    self._snapshotter.capture,
                    workspace,
                    artifacts.root / "p0",
                    snapshot_id=f"{self._workflow_id}-p0",
                    include_git_history=False,
                )
                self._p0 = p0
            except Exception as exc:
                logger.warning("Requirement clarification P0 capture failed", exc_info=True)
                await self._deliver_p0(p0_text, revision, detail=f"P0 checkpoint failed: {exc}")
                return

            while True:
                repair_timed_out = False
                revision = self._revision
                await self._phase(RequirementWorkflowPhase.CLARIFICATION, revision.number)
                try:
                    model = ChrysClarificationModel(
                        profile=model_profile,
                        snapshot=s0,
                        session_id=host._session_id,
                        session_dir=session_dir,
                        report_usage=host._accumulate_side_call_usage,
                    )
                    result = await ClarificationService(model).clarify(
                        revision=revision,
                        background=history_background,
                        snapshot=s0,
                    )
                except Exception as exc:
                    logger.warning("Requirement clarification side calls failed", exc_info=True)
                    await self._deliver_p0(p0_text, self._revision, detail=f"clarification failed: {exc}")
                    return
                if revision.number != self._revision.number:
                    continue
                try:
                    artifacts.save_result(result)
                except OSError as exc:
                    logger.warning("Requirement clarification result persistence failed", exc_info=True)
                    await self._deliver_p0(
                        p0_text,
                        revision,
                        detail=f"clarification result persistence failed: {exc}",
                    )
                    return
                if self._stop_requested:
                    await self._deliver_p0(p0_text, revision, detail="workflow stopped after P0")
                    return
                if result.is_empty:
                    await self._deliver_p0(
                        p0_text,
                        revision,
                        detail="clarification produced no additional guidance",
                    )
                    return
                if not await asyncio.to_thread(self._snapshotter.matches, p0):
                    await self._deliver_p0(
                        p0_text,
                        revision,
                        detail="workspace changed outside the workflow; repair was not started",
                        phase=RequirementWorkflowPhase.CONFLICTED,
                    )
                    return

                await self._phase(RequirementWorkflowPhase.REPAIR, revision.number)
                executor.restore_history(h0)
                host._history.bind(executor.history_state)
                host._intermediate_texts.clear()
                host._consumed_injections.clear()
                executor.reset_counters(reset_batch_id=True)
                executor.set_requirement_phase(RequirementWorkflowPhase.REPAIR)
                reminder = _repair_reminder(revision, result.delta)
                if host._reminder_middleware is not None:
                    host._reminder_middleware.queue_hook_reminders([reminder])
                executor.set_user_messages(list(revision.messages))
                try:
                    async with asyncio.timeout(self._repair_timeout_seconds):
                        await executor.run(contents, created_at=created_at)
                except TimeoutError:
                    if executor.is_running:
                        await executor.interrupt()
                    executor.run_failed = True
                    repair_timed_out = True
                if revision.number != self._revision.number:
                    try:
                        await asyncio.to_thread(self._snapshotter.restore, p0)
                    except Exception as exc:
                        logger.error("Requirement clarification amendment rollback failed", exc_info=True)
                        await self._phase(
                            RequirementWorkflowPhase.CONFLICTED,
                            self._revision.number,
                            detail=f"amendment invalidated repair and P0 rollback failed: {exc}",
                            terminal=True,
                        )
                        await self._runner.finalize_current_run()
                        return
                    executor.restore_history(p0_history)
                    host._history.bind(executor.history_state)
                    executor.adopt_fallback_success(p0_text)
                    continue
                if executor.run_failed or executor.was_interrupted:
                    try:
                        await asyncio.to_thread(self._snapshotter.restore, p0)
                    except Exception as exc:
                        logger.error("Requirement clarification P0 rollback failed", exc_info=True)
                        await self._phase(
                            RequirementWorkflowPhase.CONFLICTED,
                            revision.number,
                            detail=f"repair failed and P0 rollback failed: {exc}",
                            terminal=True,
                        )
                        await self._runner.finalize_current_run()
                        return
                    executor.restore_history(p0_history)
                    host._history.bind(executor.history_state)
                    all_injections = [*baseline_injections, *self._late_injections]
                    host._consumed_injections[:] = _reanchor_injections(all_injections, executor.history_state)
                    executor.adopt_fallback_success(p0_text)
                    detail = (
                        f"repair exceeded {self._repair_timeout_seconds:g} seconds; restored P0"
                        if repair_timed_out
                        else "repair failed; restored P0"
                    )
                    await self._deliver_p0(p0_text, revision, detail=detail)
                    return
                break

            all_injections = [*baseline_injections, *self._late_injections]
            host._consumed_injections[:] = _reanchor_injections(all_injections, executor.history_state)
            await self._phase(RequirementWorkflowPhase.FINALIZING, revision.number)
            await self._runner.finalize_current_run()
            try:
                artifacts.save_summary(
                    {
                        "workflow_id": self._workflow_id,
                        "revision": revision.number,
                        "outcome": "repaired",
                        "strategy_version": result.strategy_version,
                    }
                )
            except OSError:
                logger.warning("Failed to persist requirement-clarification summary", exc_info=True)
            await self._phase(RequirementWorkflowPhase.COMPLETED, revision.number, terminal=True)
        finally:
            executor.set_requirement_phase("")
            if run_scope is not None:
                host._turn_state.close_injection_admission(run_scope)
            if host._requirement_clarification_workflow is self:
                host._requirement_clarification_workflow = None
            if s0 is not None:
                await asyncio.to_thread(self._snapshotter.discard, s0)
            if p0 is not None:
                await asyncio.to_thread(self._snapshotter.discard, p0)

    async def _deliver_p0(
        self,
        text: str,
        revision: RequirementRevision,
        *,
        detail: str,
        phase: RequirementWorkflowPhase = RequirementWorkflowPhase.DEGRADED,
    ) -> None:
        executor = self._host._executor
        injections = _unique_injections([*self._host._consumed_injections, *self._late_injections])
        self._host._consumed_injections[:] = _reanchor_injections(injections, executor.history_state)
        executor.set_requirement_phase(RequirementWorkflowPhase.INITIAL_IMPLEMENTATION)
        executor.adopt_fallback_success(text)
        await self._host._bus.publish(
            Warning(
                code="requirement_clarification_fallback",
                message=detail,
                session_id=self._host._session_id,
            )
        )
        await executor.publish_last_response_as_final()
        await self._phase(RequirementWorkflowPhase.FINALIZING, revision.number, detail=detail)
        await self._runner.finalize_current_run()
        await self._phase(phase, revision.number, detail=detail, terminal=True)

    async def _degraded(self, detail: str) -> None:
        await self._host._bus.publish(
            Warning(
                code="requirement_clarification_degraded",
                message=detail,
                session_id=self._host._session_id,
            )
        )
        await self._phase(RequirementWorkflowPhase.DEGRADED, 1, detail=detail, terminal=True)

    async def _phase(
        self,
        phase: RequirementWorkflowPhase,
        revision: int,
        *,
        detail: str = "",
        terminal: bool = False,
    ) -> None:
        self._phase_name = phase
        if self._artifacts is not None:
            try:
                self._artifacts.save_workflow_record(
                    {
                        "version": 1,
                        "workflow_id": self._workflow_id,
                        "phase": phase,
                        "terminal": terminal,
                        "detail": detail,
                        "revision": {
                            "number": self._revision.number,
                            "messages": list(self._revision.messages),
                        },
                        "agent_profile_fingerprint": self._host._agent_profile_fingerprint,
                        "model_profile_fingerprint": self._host._model_profile_fingerprint,
                        "s0": _snapshot_record(self._s0),
                        "p0": _snapshot_record(self._p0),
                    }
                )
            except OSError:
                logger.warning("Failed to persist requirement-clarification workflow phase", exc_info=True)
        await self._host._bus.publish(
            RequirementClarificationPhaseChanged(
                workflow_id=self._workflow_id,
                phase=phase,
                revision=revision,
                detail=detail,
                terminal=terminal,
                session_id=self._host._session_id,
                workflow_phase=phase,
            )
        )


def _history_background(history_state: dict[str, Any]) -> str:
    messages: list[Message] = []
    for block in history_state.get("compressed_msgs", []):
        block_messages = getattr(block, "messages", None)
        if isinstance(block_messages, list):
            messages.extend(message for message in block_messages if isinstance(message, Message))
    live = history_state.get("messages", [])
    if isinstance(live, list):
        messages.extend(message for message in live if isinstance(message, Message))
    rows = [
        f"{message.role}: {message.text}"
        for message in messages[-40:]
        if message.role in {"user", "assistant"}
        and message.text
        and not message.additional_properties.get("_chrys_kind")
    ]
    return "\n".join(rows)[-12_000:]


def _repair_reminder(revision: RequirementRevision, delta: str) -> str:
    amendments = ""
    if len(revision.messages) > 1:
        amendments = "\n\nUser amendments, in authority order:\n" + "\n".join(
            f"{index}. {message}" for index, message in enumerate(revision.messages[1:], start=1)
        )
    return (
        "[REQUIREMENT_CLARIFICATION_REPAIR]\n"
        "A baseline implementation already exists in the workspace. Inspect it and make only the changes needed "
        "to satisfy the original request plus the repository-grounded guidance below. Do not assume the baseline "
        "transcript or rationale; verify the files directly.\n\n"
        f"{delta}{amendments}"
    )


def _reanchor_injections(
    injections: list[ConsumedInjection],
    history_state: dict[str, Any],
) -> list[ConsumedInjection]:
    messages = history_state.get("messages", [])
    opener = next((message for message in reversed(messages) if message.role == "user"), None)
    anchor = InjectionAnchor.from_message(opener)
    return [replace(injection, anchor=anchor) for injection in injections]


def _unique_injections(injections: list[ConsumedInjection]) -> list[ConsumedInjection]:
    seen: set[str] = set()
    unique: list[ConsumedInjection] = []
    for injection in injections:
        key = injection.consumption_id or f"legacy:{len(unique)}:{injection.text}"
        if key in seen:
            continue
        seen.add(key)
        unique.append(injection)
    return unique


def _snapshot_record(snapshot: WorkspaceSnapshot | None) -> dict[str, object] | None:
    if snapshot is None:
        return None
    return {
        "snapshot_id": snapshot.snapshot_id,
        "manifest_hash": snapshot.manifest_hash,
        "total_bytes": snapshot.total_bytes,
        "entry_count": snapshot.entry_count,
        "artifact_root": snapshot.artifact_root,
    }
