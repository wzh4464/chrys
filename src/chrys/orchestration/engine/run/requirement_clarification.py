# Copyright (c) 2026 Chrys. All rights reserved.

"""Opt-in baseline, clarification, and fresh-repair turn orchestration."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from chrys.foundation.events.types import Error, RequirementClarificationPhaseChanged, UserInjectResult, Warning
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.kernel import Message
from chrys.orchestration.engine.run.workflow_extensions import (
    NoopExtensions,
    RepairOutcome,
    RequirementWorkflowExtensions,
)
from chrys.service.agent_middleware.injection import ConsumedInjection, InjectionAnchor
from chrys.service.requirement_clarification.artifacts import ClarificationArtifactStore
from chrys.service.requirement_clarification.model import ChrysClarificationModel
from chrys.service.requirement_clarification.prompts import LEGACY_V1_STRATEGY_VERSION, STRATEGY_VERSION
from chrys.service.requirement_clarification.service import ClarificationService
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot, WorkspaceSnapshotter
from chrys.service.requirement_clarification.types import (
    ClarificationResult,
    ClarificationSelection,
    ClarificationStrategy,
    RequirementRevision,
    RequirementWorkflowPhase,
)
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

    def _accumulate_side_call_usage(self, usage_details: Mapping[str, Any]) -> None: ...


class RequirementClarificationWorkflow:
    """Keep P0 deliverable while deriving ΔR exclusively from frozen S0."""

    def __init__(
        self,
        host: RequirementClarificationHost,
        runner: TurnRunner,
        *,
        strategy: ClarificationStrategy = "legacy-v1-stabilized",
        reuse_workspace_as_p0: bool = False,
        clarification_only: bool = False,
        clarification_timeout_seconds: float = 1800.0,
        initial_timeout_seconds: float = 5400.0,
        repair_timeout_seconds: float = 5400.0,
        extensions: RequirementWorkflowExtensions | None = None,
    ) -> None:
        self._host = host
        self._runner = runner
        # The long-horizon track is this same workflow plus six hooks; the
        # default keeps clarification byte-identical to running without them.
        self._extensions: RequirementWorkflowExtensions = extensions or NoopExtensions()
        self._workflow_id = uuid4().hex
        self._snapshotter = WorkspaceSnapshotter()
        self._phase_name = RequirementWorkflowPhase.SNAPSHOT
        self._revision = RequirementRevision(number=1, messages=())
        self._late_injections: list[ConsumedInjection] = []
        self._latest_delta = ""
        self._latest_clarification_status = "completed"
        self._latest_empty_reason: str | None = None
        self._stop_requested = False
        self._side_task: asyncio.Task[Any] | None = None
        self._artifacts: ClarificationArtifactStore | None = None
        self._s0: WorkspaceSnapshot | None = None
        self._p0: WorkspaceSnapshot | None = None
        self._strategy = strategy
        self._reuse_workspace_as_p0 = reuse_workspace_as_p0
        self._clarification_only = clarification_only
        self._clarification_timeout_seconds = clarification_timeout_seconds
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
        self._latest_delta = ""
        self._latest_clarification_status = "completed"
        self._latest_empty_reason = None
        if self._side_task is not None:
            self._side_task.cancel()
        if self._artifacts is not None:
            try:
                self._artifacts.save_requirement_input(
                    revision=self._revision.number,
                    messages=self._revision.messages,
                )
            except OSError:
                logger.warning("Failed to persist amended requirement input", exc_info=True)
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
        await self._extensions.on_revision(self._revision)
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
        if self._side_task is not None:
            self._side_task.cancel()
        await self._extensions.cancel()
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
            artifacts.save_requirement_input(revision=revision.number, messages=revision.messages)
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
            s0_capture_options: dict[str, Any] = {
                "snapshot_id": f"{self._workflow_id}-s0",
                "include_git_history": True,
            }
            if self._reuse_workspace_as_p0:
                s0_capture_options["committed_git_head_only"] = True
            s0 = await asyncio.to_thread(
                self._snapshotter.capture,
                workspace,
                artifacts.root / "s0",
                **s0_capture_options,
            )
            self._s0 = s0
            try:
                artifacts.save_snapshot_metadata(
                    {
                        "workflow_id": self._workflow_id,
                        "revision": revision.number,
                        "s0": _snapshot_record(s0),
                        "p0": None,
                    }
                )
            except OSError:
                logger.warning("Failed to persist requirement-clarification snapshot metadata", exc_info=True)
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
            if self._reuse_workspace_as_p0:
                await self._runner._prepare_fresh_without_execution(
                    text,
                    created_at=created_at,
                    contents=contents,
                    run_scope=run_scope,
                    admission_preparation=admission_preparation,
                )
                p0_text = "Reused the existing workspace implementation as P0."
                p0_history = h0
                baseline_injections: list[ConsumedInjection] = []
            else:
                p0_timed_out = False
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
                    # The budget bounds the baseline pass; it does not abandon
                    # the turn. The workspace holds whatever P0 managed, and
                    # clarification and repair exist to improve exactly that --
                    # ending here threw away ninety minutes of edits on two
                    # benchmark tasks and answered with nothing at all.
                    p0_timed_out = True
                    if executor.is_running:
                        await executor.interrupt()
                    await _settle_executor(executor)
                if executor.was_interrupted and not p0_timed_out:
                    executor.set_requirement_phase("")
                    await self._phase(RequirementWorkflowPhase.INTERRUPTED, revision.number, terminal=True)
                    await self._runner.finalize_current_run()
                    return

                p0_failed = executor.run_failed
                if p0_timed_out or p0_failed:
                    if p0_failed:
                        # A baseline pass that ended in a provider error (a reply with
                        # reasoning but no text, a dropped connection) is no reason to
                        # abandon the turn: the workspace holds whatever it wrote, and
                        # clarification and repair exist to improve exactly that. One
                        # benchmark task answered with nothing at all after an hour.
                        code = "requirement_clarification_p0_failed"
                        detail = f"initial implementation failed ({executor.last_error or 'unknown error'}); continuing with the partial baseline"
                        fallback = "The baseline pass failed; the workspace holds whatever changes it had made."
                    else:
                        code = "requirement_clarification_p0_timeout"
                        detail = (
                            f"initial implementation exceeded {self._initial_timeout_seconds:g} seconds; "
                            "continuing with the partial baseline"
                        )
                        fallback = (
                            "The baseline pass was stopped at its time budget; the workspace holds its partial changes."
                        )
                    executor.adopt_fallback_success(executor.last_response_text.strip() or fallback)
                    await host._bus.publish(Warning(code=code, message=detail, session_id=host._session_id))
                p0_text = executor.last_response_text
                p0_history = executor.snapshot_history()
                baseline_injections = list(host._consumed_injections)
                for injection in baseline_injections:
                    revision = revision.append(injection.text)
                self._revision = revision
            try:
                artifacts.save_requirement_input(revision=revision.number, messages=revision.messages)
                artifacts.save_initial_response(revision=revision.number, response=p0_text)
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
                try:
                    artifacts.save_snapshot_metadata(
                        {
                            "workflow_id": self._workflow_id,
                            "revision": revision.number,
                            "s0": _snapshot_record(s0),
                            "p0": _snapshot_record(p0),
                        }
                    )
                except OSError:
                    logger.warning("Failed to update requirement-clarification snapshot metadata", exc_info=True)
            except Exception as exc:
                logger.warning("Requirement clarification P0 capture failed", exc_info=True)
                await self._deliver_p0(p0_text, revision, detail=f"P0 checkpoint failed: {exc}")
                return

            while True:
                repair_timed_out = False
                revision = self._revision
                await self._phase(RequirementWorkflowPhase.CLARIFICATION, revision.number)
                clarification_service = ClarificationService(
                    ChrysClarificationModel(
                        profile=model_profile,
                        snapshot=s0,
                        session_id=host._session_id,
                        session_dir=session_dir,
                        report_usage=host._accumulate_side_call_usage,
                    ),
                    strategy=self._strategy,
                )
                try:
                    async with asyncio.timeout(self._clarification_timeout_seconds):
                        side_task = asyncio.create_task(
                            clarification_service.clarify(
                                revision=revision,
                                background=history_background,
                                snapshot=s0,
                            )
                        )
                        self._side_task = side_task
                        # Parallel work is additive: losing it costs evidence,
                        # never the clarification the turn actually needs.
                        extension_task = asyncio.create_task(self._extensions.on_clarification_start(revision, s0))
                        try:
                            result = await side_task
                        finally:
                            extension_outcome = await _settle(extension_task)
                        if isinstance(extension_outcome, BaseException):
                            logger.warning("Clarification-parallel extension failed", exc_info=extension_outcome)
                            await host._bus.publish(
                                Warning(
                                    code="requirement_clarification_extension_failed",
                                    message=f"parallel clarification work failed: {extension_outcome}",
                                    session_id=host._session_id,
                                )
                            )
                except asyncio.CancelledError:
                    if self._stop_requested:
                        await self._deliver_p0(p0_text, revision, detail="workflow stopped after P0")
                        return
                    if revision.number != self._revision.number:
                        continue
                    raise
                except Exception as exc:
                    logger.warning("Requirement clarification side calls failed", exc_info=True)
                    detail = f"{type(exc).__name__}: {exc}"[:1000]
                    result = ClarificationResult(
                        strategy_version=(
                            LEGACY_V1_STRATEGY_VERSION if self._strategy == "legacy-v1-exact" else STRATEGY_VERSION
                        ),
                        revision=revision.number,
                        delta="",
                        selection=ClarificationSelection(),
                        status="degraded",
                        empty_reason="clarification_failed",
                        warnings=(f"clarification failed: {detail}",),
                    )
                finally:
                    if self._side_task is side_task:
                        self._side_task = None
                if revision.number != self._revision.number:
                    continue
                self._latest_delta = result.delta
                self._latest_clarification_status = result.status
                self._latest_empty_reason = result.empty_reason
                try:
                    # Persist the clarification decision before starting optional PACT calls.
                    # This makes PACT a downstream artifact producer rather than a gate on ΔR.
                    artifacts.save_result(result, requirement_messages=revision.messages)
                except OSError as exc:
                    logger.warning("Requirement clarification result persistence failed", exc_info=True)
                    await self._deliver_p0(
                        p0_text,
                        revision,
                        detail=f"clarification result persistence failed: {exc}",
                    )
                    return
                # A degraded clarification (no valid proposal, a failed selector) still
                # leaves the requirement itself: the Goal Contract derives from it alone,
                # and the Initial Plan can be built without proposals. Without this, the
                # long-horizon turn ends at the baseline and the campaign never starts.
                if result.status == "degraded":
                    result = replace(
                        result,
                        warnings=(
                            *result.warnings,
                            "clarification degraded; PACT input generated from the requirement alone",
                        ),
                    )
                if result.status in ("completed", "degraded"):
                    pact_service = ClarificationService(
                        ChrysClarificationModel(
                            profile=model_profile,
                            snapshot=s0,
                            session_id=host._session_id,
                            session_dir=session_dir,
                            report_usage=host._accumulate_side_call_usage,
                        )
                    )
                    try:
                        async with asyncio.timeout(self._clarification_timeout_seconds):
                            side_task = asyncio.create_task(
                                pact_service.generate_pact_input(
                                    result=result,
                                    revision=revision,
                                    background=history_background,
                                    snapshot=s0,
                                    localization_hints=self._extensions.pact_input_hints(),
                                )
                            )
                            self._side_task = side_task
                            pact_input, pact_usage = await side_task
                        result = replace(
                            result,
                            pact_input=pact_input,
                            usage_details=(*result.usage_details, *pact_usage),
                        )
                    except asyncio.CancelledError:
                        if self._stop_requested:
                            await self._deliver_p0(p0_text, revision, detail="workflow stopped after P0")
                            return
                        if revision.number != self._revision.number:
                            continue
                        raise
                    except Exception as exc:
                        pact_generation_error = f"{type(exc).__name__}: {exc}"[:1000]
                        result = replace(
                            result,
                            pact_generation_error=pact_generation_error,
                            warnings=(*result.warnings, f"PACT input generation failed: {pact_generation_error}"),
                        )
                    finally:
                        if self._side_task is side_task:
                            self._side_task = None
                else:
                    result = replace(
                        result,
                        pact_generation_error="clarification did not finish before PACT generation",
                    )
                if revision.number != self._revision.number:
                    continue
                try:
                    artifacts.save_result(result, requirement_messages=revision.messages)
                except OSError:
                    # The authority/delta result was already stored before optional PACT
                    # generation. A failure to refresh its PACT metadata cannot invalidate it.
                    logger.warning("Failed to refresh clarification result after PACT generation", exc_info=True)
                try:
                    artifacts.save_pact_generation(result)
                except OSError:
                    logger.warning("Failed to persist generated PACT inputs", exc_info=True)
                if self._stop_requested:
                    await self._deliver_p0(p0_text, revision, detail="workflow stopped after P0")
                    return
                if self._clarification_only:
                    await self._deliver_p0(
                        p0_text,
                        revision,
                        detail=(
                            "clarification-only mode completed; repair was not started"
                            if result.status == "completed"
                            else "clarification degraded; repair and PACT were not started"
                        ),
                        phase=(
                            RequirementWorkflowPhase.COMPLETED
                            if result.status == "completed"
                            else RequirementWorkflowPhase.DEGRADED
                        ),
                        warn=result.status == "degraded",
                    )
                    return
                if result.status == "degraded":
                    await self._deliver_p0(
                        p0_text,
                        revision,
                        detail=f"clarification degraded: {result.empty_reason or 'unknown reason'}",
                    )
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
                reminder = _repair_reminder(revision, self._extensions.augment_repair_reminder(result.delta))
                if host._reminder_middleware is not None:
                    host._reminder_middleware.queue_hook_reminders([reminder])
                executor.set_user_messages(list(revision.messages))
                try:
                    async with asyncio.timeout(self._repair_timeout_seconds):
                        await executor.run(contents, created_at=created_at)
                except TimeoutError:
                    if executor.is_running:
                        await executor.interrupt()
                    repair_timed_out = True
                if revision.number != self._revision.number:
                    repair_status = "invalidated_by_amendment"
                elif repair_timed_out:
                    repair_status = "timed_out"
                elif executor.was_interrupted:
                    repair_status = "interrupted"
                elif executor.run_failed:
                    repair_status = "failed"
                else:
                    repair_status = "succeeded"
                try:
                    artifacts.save_repair_attempt(
                        revision=revision.number,
                        status=repair_status,
                        response=executor.last_response_text,
                        transcript={
                            "history": serialize_state(executor.history_state),
                            "service_session_id": executor.service_session_id,
                        },
                    )
                except OSError:
                    logger.warning("Failed to persist requirement-clarification repair attempt", exc_info=True)
                if revision.number != self._revision.number:
                    try:
                        await asyncio.to_thread(self._snapshotter.restore, p0)
                    except Exception as exc:
                        logger.error("Requirement clarification amendment rollback failed", exc_info=True)
                        detail = f"amendment invalidated repair and P0 rollback failed: {exc}"
                        await self._phase(
                            RequirementWorkflowPhase.CONFLICTED,
                            self._revision.number,
                            detail=detail,
                            terminal=True,
                        )
                        await self._end_turn_with_error("requirement_clarification_conflicted", detail)
                        await self._runner.finalize_current_run()
                        return
                    executor.restore_history(p0_history)
                    host._history.bind(executor.history_state)
                    executor.adopt_fallback_success(p0_text)
                    continue
                if repair_timed_out or executor.run_failed or executor.was_interrupted:
                    try:
                        await asyncio.to_thread(self._snapshotter.restore, p0)
                    except Exception as exc:
                        logger.error("Requirement clarification P0 rollback failed", exc_info=True)
                        detail = f"repair failed and P0 rollback failed: {exc}"
                        await self._phase(
                            RequirementWorkflowPhase.CONFLICTED,
                            revision.number,
                            detail=detail,
                            terminal=True,
                        )
                        await self._end_turn_with_error("requirement_clarification_conflicted", detail)
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
            await self._extensions.after_repair(
                RepairOutcome(
                    status="succeeded",
                    final_text=executor.last_response_text,
                    baseline="p1",
                    pact_input_dir=self._pact_input_dir(),
                )
            )
            await self._runner.finalize_current_run()
            try:
                artifacts.save_summary(
                    {
                        "workflow_id": self._workflow_id,
                        "revision": revision.number,
                        "outcome": "repaired",
                        "accepted_phase": "repair",
                        "final_response": executor.last_response_text,
                        "strategy_version": result.strategy_version,
                        "clarification_status": result.status,
                        "clarification_empty_reason": result.empty_reason,
                    },
                    requirement_messages=revision.messages,
                    delta=result.delta,
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

    def _pact_input_dir(self) -> Path | None:
        """Where the accepted PACT pair landed, or None when none was generated."""
        if self._artifacts is None:
            return None
        directory = self._artifacts.pact_input_dir
        return directory if (directory / "goal-contract.json").is_file() else None

    async def _end_turn_with_error(self, code: str, detail: str) -> None:
        """Publish the terminal an aborted workflow otherwise never delivers.

        A phase event is progress, not an outcome: the TUI keeps a turn open
        until a final answer, an ``Error``, or the user's own Stop arrives, and
        headless ``chrys run`` decides its exit status the same way. These
        paths end the turn with no answer at all -- a rollback that failed and
        left the workspace half repaired -- so without this the spinner runs
        forever and a script exits 0 on a workspace nobody has told it is
        inconsistent.
        """
        await self._host._bus.publish(
            Error(
                code=code,
                message=f"Requirement clarification stopped: {detail}",
                recoverable=True,
                session_id=self._host._session_id,
            )
        )

    async def _deliver_p0(
        self,
        text: str,
        revision: RequirementRevision,
        *,
        detail: str,
        phase: RequirementWorkflowPhase = RequirementWorkflowPhase.DEGRADED,
        warn: bool = True,
    ) -> None:
        executor = self._host._executor
        injections = _unique_injections([*self._host._consumed_injections, *self._late_injections])
        self._host._consumed_injections[:] = _reanchor_injections(injections, executor.history_state)
        executor.set_requirement_phase(RequirementWorkflowPhase.INITIAL_IMPLEMENTATION)
        executor.adopt_fallback_success(text)
        if warn:
            await self._host._bus.publish(
                Warning(
                    code="requirement_clarification_fallback",
                    message=detail,
                    session_id=self._host._session_id,
                )
            )
        # A reused workspace synthesises its baseline text without running a
        # P0 pass, so nothing showed it yet; every other path already did.
        await executor.publish_last_response_as_final(repeats_provisional=not self._reuse_workspace_as_p0)
        await self._phase(RequirementWorkflowPhase.FINALIZING, revision.number, detail=detail)
        await self._extensions.after_repair(
            RepairOutcome(
                status="promoted_p0",
                final_text=text,
                baseline="p0",
                pact_input_dir=self._pact_input_dir(),
            )
        )
        await self._runner.finalize_current_run()
        if self._artifacts is not None:
            try:
                self._artifacts.save_summary(
                    {
                        "workflow_id": self._workflow_id,
                        "revision": revision.number,
                        "outcome": "p0_promoted",
                        "accepted_phase": "initial_trial",
                        "final_response": text,
                        "detail": detail,
                        "clarification_status": self._latest_clarification_status,
                        "clarification_empty_reason": self._latest_empty_reason,
                    },
                    requirement_messages=revision.messages,
                    delta=self._latest_delta,
                )
            except OSError:
                logger.warning("Failed to persist requirement-clarification fallback summary", exc_info=True)
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
                        "schema": "chrys/requirement-clarification/workflow/v1",
                        "artifact_version": 1,
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


async def _settle(task: asyncio.Task[Any]) -> BaseException | None:
    """Wait for a parallel task and return its exception instead of raising it."""
    try:
        await task
    except asyncio.CancelledError:
        raise
    except BaseException as exc:
        return exc
    return None


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


async def _settle_executor(executor: Any, *, timeout_seconds: float = 30.0) -> None:
    """Wait for an interrupted executor pass to actually stop.

    ``interrupt`` signals the pass and cancels its tool; the history is only
    safe to snapshot once the pass has observed that and returned.
    """
    # The executor exposes no "stopped" event, only ``is_running``; a bounded
    # number of short waits is the whole of what can be done with that.
    interval = 0.05
    for _ in range(int(timeout_seconds / interval)):
        if not getattr(executor, "is_running", False):
            return
        await asyncio.sleep(interval)


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
