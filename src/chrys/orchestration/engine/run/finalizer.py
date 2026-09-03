# Copyright (c) 2026 Chrys. All rights reserved.

"""Post-run finalization for main-agent turns."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from chrys.foundation.events.types import UserInjectResult
from chrys.foundation.i18n.formatting import format_message
from chrys.foundation.models.history_markers import EXECUTION_FAILED_MESSAGE, HistoryMarkerKind
from chrys.foundation.trajectory.context import trajectory_scope
from chrys.foundation.trajectory.event_types import TurnEndReason
from chrys.orchestration.engine.run import sub_agent_coordination as sub_agents
from chrys.orchestration.engine.run.active_injection import withdraw_committed_injection_reminders
from chrys.orchestration.engine.run.turn_hooks import TurnHookDispatcher
from chrys.orchestration.engine.run.turn_state import CurrentRunScope
from chrys.orchestration.engine.state.machine import Trigger
from chrys.service.context.providers.history import PRE_OUTPUT_HISTORY_LEN_STATE_KEY
from chrys.service.routing.classifier import RouteDecision
from chrys.service.trajectory.preparation import PreparationOutcome

if TYPE_CHECKING:
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.models.workspace import Workspace
    from chrys.kernel import LoopRecorder
    from chrys.orchestration.engine.executor import Executor
    from chrys.orchestration.engine.run.turn_state import TurnRuntimeState
    from chrys.orchestration.engine.state.machine import EngineStateMachine
    from chrys.orchestration.engine.trajectory import TrajectoryRecorder
    from chrys.orchestration.sub_agents.tools import SubAgentTools
    from chrys.service.agent_middleware.injection import ConsumedInjection, InjectionMiddleware, QueuedInjection
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
    from chrys.service.hooks.manager import HookManager
    from chrys.service.mutations.tracker import MutationTracker
    from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker
    from chrys.service.profiles.agents.schema import AgentProfile
    from chrys.service.session.history import SessionHistoryManager
    from chrys.service.session.persistence import SessionPersistence


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostRunOutcome:
    """Finalization result used by the runner's terminal-boundary cleanup."""

    failed: bool
    interrupted: bool
    completed_scope: CurrentRunScope | None


def _route_marker(host: Any, *, failed: bool, interrupted: bool) -> dict[str, Any] | None:
    """Describe how this turn was routed, for later deposition and review.

    ``baseline`` says what the workspace actually holds, and ``campaign`` is
    the campaign's own reported status -- never inferred, because only the
    campaign knows whether it completed.
    """
    decision = host._last_route
    if decision is None:
        return None
    campaign = host._long_horizon_campaign
    baseline = "none" if interrupted or failed else ("p1" if decision.plan.clarification else "none")
    if campaign is not None:
        baseline = "p1"
    return {
        "_chrys_route": {
            "track": decision.track.value,
            "band": decision.band.value,
            "source": decision.source,
            "reason": decision.reason[:400],
            "baseline": baseline,
            "campaign": campaign,
        }
    }


class TurnFinalizerHost(Protocol):
    """Engine host contract needed for post-run finalization."""

    _turn_state: TurnRuntimeState
    _bus: EventBus
    _session_id: str | None
    _executor: Executor
    _history: SessionHistoryManager
    _loop_recorder: LoopRecorder | None
    _injection: InjectionMiddleware
    _shutting_down: bool
    _intermediate_texts: dict[int, str]
    _consumed_injections: list[ConsumedInjection]
    _paused_sub_agents: set[str]
    _sub_agent_tools: SubAgentTools | None
    _fsm: EngineStateMachine
    _mutation_tracker: MutationTracker | None
    _workspace_change_tracker: WorkspaceChangeTracker
    _reminder_middleware: SystemReminderMiddleware | None
    _hook_manager: HookManager | None
    _agent_profile: AgentProfile | None
    _workspace: Workspace | None
    _turn_number: int
    _last_route: RouteDecision | None
    _long_horizon_campaign: dict[str, Any] | None
    _on_successful_turn: Callable[[], None]
    _persistence: SessionPersistence
    _trajectory_recorder: TrajectoryRecorder

    @property
    def _settings(self) -> Settings:
        """Read-only: settings change through the handle, not the holder."""
        ...

    async def _save_current_session(self) -> bool: ...


class TurnFinalizer:
    """Finalize one just-ended executor pass."""

    def __init__(self, host: TurnFinalizerHost) -> None:
        self._host = host

    async def finalize(self) -> PostRunOutcome:
        """Run post-execution fixup without starting pending retry dispatch."""
        host = self._host
        completed_scope = host._turn_state.current_run_scope
        failed = host._executor.run_failed or host._executor.was_interrupted
        interrupted = host._executor.was_interrupted
        approval_decisions = host._executor.drain_approval_decisions()
        metadata_start_index = _post_compaction_history_start_index(host)

        if failed:
            # The pass-start boundary keeps a retried/resumed pass's recovered
            # messages AFTER retained work from an earlier same-turn pass.
            # A failed pass can finalize only after construction installs its recorder.
            loop_recorder = cast("LoopRecorder", host._loop_recorder)
            host._history.merge_loop_messages(
                loop_recorder,
                insert_index=metadata_start_index,
            )
            host._history.persist_approval_decisions(
                approval_decisions,
                start_index=metadata_start_index,
            )
            approval_decisions = []
            host._history.trim_to_last_complete_tool_results()
            if interrupted:
                host._history.remove_trailing_agent_text()

        await self._drain_abandoned_injections()
        self._persist_history_metadata(
            approval_decisions=approval_decisions,
            metadata_start_index=metadata_start_index,
            failed=failed,
        )
        self._clear_parent_sub_agent_state()
        self._apply_terminal_history_state(failed=failed, interrupted=interrupted)
        turn_close_cancelled = await self._record_trajectory_turn_finished(failed=failed, interrupted=interrupted)
        response_outcome = "partial"
        drained_scopes: list[str] = []
        waited_hook_operation_ids: list[str] = []
        degraded = False
        try:
            if turn_close_cancelled:
                # The capture behind it is advisory and would hold shutdown for its
                # own timeout on an already-cancelled task, so it is skipped — and
                # skipped means there is no baseline to keep.
                host._workspace_change_tracker.capture_cancelled()
                baseline_cancelled = True
            else:
                baseline_cancelled, baseline_degraded = await self._capture_workspace_baseline()
                degraded = degraded or baseline_degraded
            if turn_close_cancelled or baseline_cancelled:
                # A cancellation (shutdown's post-run fallback) landed on the
                # advisory capture. Shutdown suppresses its own trailing save
                # after that fallback, so the critical save must still run here.
                try:
                    saved = await host._save_current_session()
                    degraded = degraded or not saved
                    if saved:
                        host._turn_state.record_current_run_final_save()
                    await self._record_trajectory_after_save()
                finally:
                    degraded = not self._run_success_callback(failed=failed) or degraded
                response_outcome = "cancelled"
                raise asyncio.CancelledError
            try:
                save_degraded = await self._save_and_drain_hooks(
                    failed=failed,
                    drained_scopes=drained_scopes,
                    waited_hook_operation_ids=waited_hook_operation_ids,
                )
                degraded = degraded or save_degraded
            finally:
                # After the save so callbacks observe the persisted turn — a
                # brand-new session's first save must exist before side effects
                # keyed on session.json (e.g. the session-title updater) run.
                degraded = not self._run_success_callback(failed=failed) or degraded
            response_outcome = "partial" if degraded else "settled"
        except asyncio.CancelledError:
            response_outcome = "cancelled"
            raise
        finally:
            if response_outcome == "cancelled":
                host._trajectory_recorder.turn_response_settled_soon(
                    outcome=response_outcome,
                    drained_scopes=drained_scopes,
                    waited_hook_operation_ids=waited_hook_operation_ids,
                )
            else:
                await host._trajectory_recorder.turn_response_settled(
                    outcome=response_outcome,
                    drained_scopes=drained_scopes,
                    waited_hook_operation_ids=waited_hook_operation_ids,
                )

        return PostRunOutcome(failed=failed, interrupted=interrupted, completed_scope=completed_scope)

    async def _capture_workspace_baseline(self) -> tuple[bool, bool]:
        """Capture after the terminal transition and before the persisted save.

        Returns ``(cancelled, degraded)``. The capture is advisory but the
        save behind it is not, so cancellation is absorbed long enough for
        the caller to finish the critical save.
        """
        host = self._host
        tracker = host._workspace_change_tracker
        if not host._settings.workspace_change_notice:
            tracker.invalidate()
            return False, False
        try:
            await asyncio.wait_for(
                asyncio.to_thread(tracker.capture_baseline, host._turn_number),
                timeout=15.0,
            )
        except asyncio.CancelledError:
            tracker.capture_cancelled()
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
            return True, False
        except TimeoutError:
            tracker.capture_timed_out(host._turn_number)
            logger.debug("Workspace baseline capture exceeded its deadline")
            return False, True
        except Exception:
            tracker.invalidate()
            logger.debug("Workspace baseline capture failed", exc_info=True)
            return False, True
        return False, False

    async def _drain_abandoned_injections(self) -> None:
        """Close and publish unconsumed injections unless shutdown owns notification."""
        host = self._host
        abandoned: list[QueuedInjection] = host._injection.drain_pending()
        for injection in abandoned:
            # The text never reached the model, so its commit-time reminders
            # must not either — a pending retry preserves turn reminders and
            # would otherwise carry them into the next model call.
            withdraw_committed_injection_reminders(
                host._reminder_middleware,
                host._turn_state.current_run_scope,
                injection,
            )
            if injection.preparation is not None:
                injection.preparation.finished_soon(
                    outcome=PreparationOutcome.TARGET_STALE,
                    target_turn_id=injection.target_turn_id,
                )
        if host._shutting_down:
            return
        for injection in abandoned:
            await host._bus.publish(
                UserInjectResult(
                    text=injection.text,
                    consumed=False,
                    created_at=injection.created_at,
                    injection_id=injection.injection_id,
                    session_id=host._session_id,
                )
            )

    def _persist_history_metadata(
        self,
        *,
        approval_decisions: list[Any],
        metadata_start_index: int,
        failed: bool,
    ) -> None:
        """Attach post-run metadata after any failed-run history repair."""
        host = self._host
        batch_records = host._executor.drain_batch_records()
        tool_call_count = sum(
            1
            for message in host._history.messages
            if message.role == "assistant"
            and any(content.type == "function_call" and not content.informational_only for content in message.contents)
        )
        logger.debug(
            "_post_run: batch_records=%d tool_call_msgs=%d total_msgs=%d failed=%s",
            len(batch_records),
            tool_call_count,
            len(host._history.messages),
            failed,
        )
        batch_anchors = host._history.persist_batch_ids(batch_records)
        if host._intermediate_texts:
            host._history.persist_intermediate_texts(dict(host._intermediate_texts), batch_anchors)
            host._intermediate_texts.clear()
        host._history.persist_approval_decisions(
            approval_decisions,
            start_index=metadata_start_index,
        )
        if host._consumed_injections:
            host._history.persist_consumed_injections(host._consumed_injections)
            host._consumed_injections.clear()
        host._history.backfill_missing_created_at(start_index=metadata_start_index)
        _clear_post_compaction_history_start_index(host)

    def _clear_parent_sub_agent_state(self) -> None:
        """Clear parent sub-agent pause state before terminal markers are inserted."""
        sub_agents.clear_parent_paused_state(self._host)

    def _apply_terminal_history_state(self, *, failed: bool, interrupted: bool) -> None:
        """Apply FSM transitions and terminal history markers."""
        host = self._host
        if failed:
            if interrupted:
                host._history.insert_interrupted_marker()
                host._fsm.try_transition(Trigger.RUN_INTERRUPTED)
            else:
                last_error = host._executor.last_error
                if last_error:
                    host._history.insert_interrupted_marker(reason=last_error, source="error")
                else:
                    host._history.insert_interrupted_marker(
                        reason=format_message(EXECUTION_FAILED_MESSAGE.bind()),
                        source="error",
                        status_code=HistoryMarkerKind.STATUS_EXECUTION_FAILED,
                    )
                host._fsm.try_transition(Trigger.RUN_FAILED)
            # A failed/interrupted Responses turn can leave the framework session
            # holding a service id from an incomplete provider-side state. Drop it
            # so retry/resume replays local recovery history instead of skipping it.
            host._executor.service_session_id = ""
        else:
            host._history.remove_all_status_markers()
            host._fsm.try_transition(Trigger.RUN_COMPLETED)

        host._history.insert_turn_marker(_route_marker(host, failed=failed, interrupted=interrupted))
        if host._mutation_tracker is not None:
            host._mutation_tracker.cleanup_unused_snapshots()

    def _run_success_callback(self, *, failed: bool) -> bool:
        """Run the successful-turn callback after markers and cleanup."""
        if failed:
            return True
        try:
            self._host._on_successful_turn()
        except Exception:
            logger.debug("Failed to run successful turn callback", exc_info=True)
            return False
        return True

    async def _save_and_drain_hooks(
        self,
        *,
        failed: bool,
        drained_scopes: list[str],
        waited_hook_operation_ids: list[str],
    ) -> bool:
        """Save the session, fire after-turn hooks, then drain turn hooks."""
        host = self._host
        saved = await host._save_current_session()
        if saved:
            host._turn_state.record_current_run_final_save()
        await self._record_trajectory_after_save()
        with trajectory_scope(host._trajectory_recorder.finished_turn_context()):
            await TurnHookDispatcher(host).fire_after_turn(failed=failed)
        if host._hook_manager is not None:
            drained_scopes.append("turn")
            try:
                await host._hook_manager.drain_turn()
            finally:
                waited_hook_operation_ids.extend(host._hook_manager.active_turn_drain_operation_ids)
        return not saved

    async def _record_trajectory_turn_finished(self, *, failed: bool, interrupted: bool) -> bool:
        """Close the pass's trajectory turn right after its terminal marker landed.

        Returns True when a cancellation landed on the write acknowledgement —
        a slow backend can hold it until shutdown gives up on the run task.
        The terminal's line is committed by then, so the turn is finished in
        the log while the save behind this call is the only thing that would
        make it finished in the session. The cancellation is absorbed here
        (and the task uncancelled) exactly as the workspace capture does, so
        the caller can complete that save before completing it.
        """
        if interrupted:
            end_reason = TurnEndReason.INTERRUPTED
        elif failed:
            end_reason = TurnEndReason.ERROR
        else:
            end_reason = TurnEndReason.COMPLETED
        try:
            await self._host._trajectory_recorder.turn_finished(end_reason=end_reason)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()
            return True
        return False

    async def _record_trajectory_after_save(self) -> None:
        """Derive the turn's mutation summary from the saved state, then checkpoint the log."""
        host = self._host
        recorder = host._trajectory_recorder
        tracker = host._mutation_tracker
        if tracker is not None:
            try:
                summary = tracker.get_turn_file_summary(host._turn_number)
            except Exception:
                logger.debug("Mutation summary unavailable for turn %d", host._turn_number, exc_info=True)
                summary = {}
            await recorder.mutation_summary(summary, checkpoint=host._persistence.checkpoint_for(host._session_id))
        await recorder.checkpoint()


def _post_compaction_history_start_index(host: TurnFinalizerHost) -> int:
    """Return the current-run metadata floor after request-time history compression."""
    start_index = host._turn_state.history_start_index
    try:
        history_state = host._executor.history_state
    except AttributeError:
        return start_index
    pre_output_len = history_state.get(PRE_OUTPUT_HISTORY_LEN_STATE_KEY)
    return pre_output_len if isinstance(pre_output_len, int) else start_index


def _clear_post_compaction_history_start_index(host: TurnFinalizerHost) -> None:
    """Clear the transient request-time compression metadata floor."""
    try:
        history_state = host._executor.history_state
    except AttributeError:
        return
    history_state.pop(PRE_OUTPUT_HISTORY_LEN_STATE_KEY, None)


def _expire_current_run_scope(host: TurnFinalizerHost, scope: CurrentRunScope | None) -> None:
    """Clear a completed current-run scope and expire its service-owned reminder scope."""
    if scope is None:
        return
    host._turn_state.clear_current_run_scope(scope)
    if host._reminder_middleware is not None:
        host._reminder_middleware.expire_current_run_scope(scope.reminder_scope)
