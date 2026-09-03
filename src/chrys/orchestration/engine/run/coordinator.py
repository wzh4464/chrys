# Copyright (c) 2026 Chrys. All rights reserved.

"""Event-facing turn coordination for main-agent turns."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from chrys.foundation.events.types import (
    AgentProfileSwitch,
    Error,
    ProfileSwitched,
    RouteOverride,
    UserInject,
    UserInjectCancel,
    UserInjectResult,
    UserInterrupt,
    UserMessage,
    UserRetry,
    Warning,
)
from chrys.foundation.i18n import msg
from chrys.foundation.platform import safe_getcwd
from chrys.orchestration.engine.run.active_injection import (
    ActiveInjectionHost,
    ActiveTurnInjector,
    withdraw_committed_injection_reminders,
)
from chrys.orchestration.engine.run.attachments import (
    AttachmentDiscoveryResult,
    discover_image_mentions,
    discover_image_references,
    load_image_attachments,
)
from chrys.orchestration.engine.run.finalizer import TurnFinalizerHost, _expire_current_run_scope
from chrys.orchestration.engine.run.prompt_content import PromptContentHost, PromptContentPreparer
from chrys.orchestration.engine.run.retry import RetryCoordinator, RetryHost
from chrys.orchestration.engine.run.routing import TurnRouter
from chrys.orchestration.engine.run.runner import TurnRunnerHost
from chrys.orchestration.engine.run.turn_hooks import PromptSubmitGate, TurnHookDispatcher, TurnHookHost
from chrys.orchestration.engine.run.turn_state import (
    CurrentRunInjectionWindow,
    CurrentRunScope,
    CurrentTurnInput,
    PreAdmissionPreparationEntry,
    PreAdmissionPreparationTracker,
    PromptAdmissionScope,
    RunTaskDrainOutcome,
)
from chrys.orchestration.engine.state.machine import EngineState, Trigger
from chrys.service.hooks.schema import HookDecision
from chrys.service.routing.classifier import RouteDecision
from chrys.service.routing.guard import TiebreakerGuard
from chrys.service.trajectory.preparation import (
    PreparationOutcome,
    PreparationScope,
    PreparationTrace,
    input_admission_wait,
)

if TYPE_CHECKING:
    from chrys.orchestration.engine.executor import Executor


_IMAGE_COMPRESSION_TIMEOUT_SECONDS = 30.0

_COORDINATOR_ENGINE_NOT_STARTED = msg(
    "coordinator.engine_not_started",
    fallback="Engine not started",
)
_COORDINATOR_INTERRUPT_IGNORED_LOADING = msg(
    "coordinator.interrupt_ignored_loading",
    fallback="Interrupt ignored while agent infrastructure is loading.",
)
_COORDINATOR_PROMPT_ADMISSION_CONFLICT = msg(
    "coordinator.prompt_admission_conflict",
    fallback="Prompt could not be admitted because another turn started.",
)


_log = logging.getLogger(__name__)

# A soft restart rebuilds tools, skills and MCP connections; past this the
# switch is not coming and the turn is better run on the current profile.
_ROUTE_SWITCH_TIMEOUT_SECONDS = 60.0


class TurnCoordinatorHost(
    TurnRunnerHost,
    RetryHost,
    ActiveInjectionHost,
    TurnHookHost,
    PromptContentHost,
    TurnFinalizerHost,
    Protocol,
):
    """Engine host contract for event-facing turn orchestration.

    The coordinator fans its host out to every phase service, so this
    contract is the union of the downstream host contracts plus the few
    members only the coordinator itself touches.
    """

    _agent_loading: bool
    # TurnRunnerHost sees the executor mid-run and declares it non-Optional;
    # the coordinator admits prompts before the first build, when the
    # executor can still be None, so the union type is pinned here.
    _executor: Executor | None
    _route_override: RouteOverride | None
    _last_route: RouteDecision | None
    _route_fingerprint: str
    _tiebreaker_guard: TiebreakerGuard
    _agent_registry: Any | None

    async def _run_and_save(
        self,
        text: str,
        created_at: datetime | str | None = None,
        contents: list[Any] | None = None,
        *,
        run_scope: CurrentRunScope | None = None,
        injection_window: CurrentRunInjectionWindow | None = None,
        admission_preparation: PreparationTrace | None = None,
    ) -> None: ...


PromptContentPreparerFactory = Callable[[TurnRunnerHost], PromptContentPreparer]


class TurnCoordinator:
    """Event-facing facade for one main-agent turn lifecycle.

    The host is the engine itself, passed directly (no forwarding adapter).
    Rebuilds replace engine attributes in place — ``_executor``,
    ``_reminder_middleware``, ``_workspace``, etc. — so phase services must
    read them through the host at each use and never cache them across an
    ``await``.
    """

    def __init__(
        self,
        host: TurnCoordinatorHost,
        *,
        prompt_content_preparer_factory: PromptContentPreparerFactory | None = None,
    ) -> None:
        self._host = host
        self._prompt_content_preparer_factory = prompt_content_preparer_factory or prompt_content_preparer

    @property
    def run_task(self) -> asyncio.Task[None] | None:
        """Current run task, including a retry task installed during finalization."""
        return self._host._turn_state.run_task

    @property
    def current_input(self) -> CurrentTurnInput:
        """Current prompt data used by crash-recovery checkpoint writes."""
        return self._host._turn_state.current_input

    @property
    def is_turn_active(self) -> bool:
        """Return whether the FSM currently admits active-turn injection."""
        return self._host._fsm.is_running()

    @property
    def is_turn_lifecycle_active(self) -> bool:
        """Return whether execution or post-run finalization is still in flight."""
        task = self._host._turn_state.run_task
        return task is not None and not task.done()

    def was_run_task_finally_saved(self, task: asyncio.Task[None]) -> bool:
        """Return whether *task* completed the durable final session save."""
        return self._host._turn_state.was_run_task_finally_saved(task)

    def clear_run_task(self) -> None:
        """Clear the visible run task after shutdown has drained or cancelled it."""
        self._host._turn_state.run_task = None

    async def wait_for_run_task(self) -> None:
        """Wait for the active run-task chain, preserving task failure semantics."""
        await self._observe_run_task_chain(propagate_inner_cancel=True)

    async def drain_run_task_chain_for_boundary(self) -> RunTaskDrainOutcome:
        """Observe the active run-task chain for a rebuild/session boundary."""
        return await self._observe_run_task_chain(propagate_inner_cancel=False)

    async def _observe_run_task_chain(self, *, propagate_inner_cancel: bool) -> RunTaskDrainOutcome:
        """Observe each installed run task, including tasks installed during finalization."""
        outcome = RunTaskDrainOutcome()
        observed: set[asyncio.Task[None]] = set()
        while True:
            task = self._host._turn_state.run_task
            if task is None or task in observed:
                return outcome
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                waiter = asyncio.current_task()
                caller_cancelled = waiter is not None and waiter.cancelling() > 0
                if task.cancelled() and not caller_cancelled:
                    outcome = RunTaskDrainOutcome(cancelled=True)
                    observed.add(task)
                    if propagate_inner_cancel:
                        raise
                else:
                    raise
            else:
                observed.add(task)

    async def run_fresh(
        self,
        text: str,
        created_at: datetime | str | None = None,
        contents: list[Any] | None = None,
        *,
        run_scope: CurrentRunScope | None = None,
        injection_window: CurrentRunInjectionWindow | None = None,
        admission_preparation: PreparationTrace | None = None,
    ) -> None:
        """Execute a fresh agent pass against the current engine state."""
        from chrys.orchestration.engine.run.runner import TurnRunner

        await TurnRunner(self._host, prompt_content_preparer_factory=self._prompt_content_preparer_factory).run_fresh(
            text,
            created_at=created_at,
            contents=contents,
            run_scope=run_scope,
            injection_window=injection_window,
            admission_preparation=admission_preparation,
        )

    async def run_retry(
        self,
        additional_text: str = "",
        created_at: datetime | str | None = None,
        *,
        run_scope: CurrentRunScope | None = None,
        injection_window: CurrentRunInjectionWindow | None = None,
        admission_preparation: PreparationTrace | None = None,
    ) -> None:
        """Execute a retry/resume pass against the current engine state."""
        from chrys.orchestration.engine.run.runner import TurnRunner

        await TurnRunner(self._host).run_retry(
            additional_text,
            created_at=created_at,
            run_scope=run_scope,
            injection_window=injection_window,
            admission_preparation=admission_preparation,
        )

    async def finalize_current_run(self) -> None:
        """Finalize the current executor pass against the current engine state."""
        from chrys.orchestration.engine.run.runner import TurnRunner

        await TurnRunner(self._host).finalize_current_run()

    async def on_user_message(self, event: UserMessage) -> None:
        """Handle a user message by running the executor as an async task.

        On accepted fresh turns, mutates ``event.prepared_contents`` before
        returning so sequential ``EventBus.publish`` callers can render the exact
        multimodal payload that will be sent to the model.
        """
        # Track injection-identified submits so a concurrent cancel knows an
        # admission is still in flight to observe its mark (no-op for None).
        turn_state = self._host._turn_state
        turn_state.begin_inflight_injection(event.injection_id)
        preparation_tracker = PreAdmissionPreparationTracker()
        handed_off = False
        try:
            await self._start_pre_admission_preparation(preparation_tracker)
            handed_off = await self._admit_user_message(event, preparation_tracker=preparation_tracker)
        except asyncio.CancelledError:
            preparation = preparation_tracker.preparation
            if (
                preparation is not None
                and not preparation_tracker.preparation_handed_off
                and not preparation.finished_state
            ):
                preparation.finished_soon(outcome=PreparationOutcome.CANCELLED)
            raise
        except BaseException:
            preparation = preparation_tracker.preparation
            if (
                preparation is not None
                and not preparation_tracker.preparation_handed_off
                and not preparation.finished_state
            ):
                preparation.finished_soon(outcome=PreparationOutcome.PREPARATION_FAILED)
            raise
        finally:
            turn_state.deregister_pre_admission_preparation(preparation_tracker.current)
            turn_state.finish_inflight_injection(event.injection_id)
            preparation = preparation_tracker.preparation
            if (
                preparation is not None
                and not handed_off
                and not preparation_tracker.preparation_handed_off
                and not preparation.finished_state
            ):
                await preparation.finished(outcome=PreparationOutcome.PREPARATION_FAILED)

    async def _admit_user_message(
        self,
        event: UserMessage,
        *,
        preparation_tracker: PreAdmissionPreparationTracker,
    ) -> bool:
        """Admit one message and report whether its preparation changed owners."""
        host = self._host
        admission: PromptAdmissionScope | None = None
        admission_released = False
        routed = False
        while True:
            await self._wait_for_pre_admission_gate(host._wait_for_agent_load_idle, preparation_tracker)
            preparation = preparation_tracker.preparation
            if host._executor is None:
                if host._turn_state.prompt_admission_closed:
                    await self._wait_for_pre_admission_gate(
                        host._turn_state.wait_for_prompt_admission_open,
                        preparation_tracker,
                    )
                    continue
                await host._bus.publish(
                    Error(
                        code="not_ready",
                        message="Engine not started",
                        display_message=_COORDINATOR_ENGINE_NOT_STARTED.bind(),
                        session_id=host._session_id,
                    )
                )
                if preparation is not None:
                    await preparation.finished(outcome=PreparationOutcome.NOT_READY)
                return False

            # A run is in-flight — FSM covers RUNNING, PENDING_RETRY, and
            # AWAITING_SUB_AGENTS (parent task pinned on a sub-agent's
            # ``pending_decision`` future). Gate on the FSM so the intent
            # travels with the state contract, not with the executor's
            # ``_running`` implementation detail — a new state that should
            # also inject only has to be added to
            # :meth:`EngineStateMachine.is_running` to be covered here.
            if host._fsm.is_running():
                workflow = getattr(host, "_requirement_clarification_workflow", None)
                if workflow is not None and workflow.accepts_amendments:
                    return await self._admit_requirement_amendment(
                        workflow,
                        event,
                        preparation=preparation,
                    )
                return await ActiveTurnInjector(host).inject(
                    event.text,
                    created_at=event.timestamp,
                    route="fsm_active",
                    reject_images_without_target=True,
                    injection_id=event.injection_id,
                    preparation=preparation,
                    preparation_tracker=preparation_tracker,
                )

            # If a previous run task is still cleaning up (e.g., interrupt teardown),
            # wait for it to finish before starting a new run. Without this, the
            # message would be injected into the dying run and then drained/lost.
            if host._turn_state.run_task is not None and not host._turn_state.run_task.done():
                if host._executor.is_running:
                    # Executor is actively running (not just cleaning up) — inject
                    return await ActiveTurnInjector(host).inject(
                        event.text,
                        created_at=event.timestamp,
                        route="executor_fallback",
                        reject_images_without_target=True,
                        injection_id=event.injection_id,
                        preparation=preparation,
                        preparation_tracker=preparation_tracker,
                    )
                # Executor finished but _post_run() is still running (session
                # save, FSM transition). Wait for cleanup to complete so the
                # new run starts from a clean state.
                await self._wait_for_pre_admission_gate(self._wait_for_existing_run_task, preparation_tracker)
                preparation = preparation_tracker.preparation

            if host._turn_state.prompt_admission_closed:
                await self._wait_for_pre_admission_gate(
                    host._turn_state.wait_for_prompt_admission_open,
                    preparation_tracker,
                )
                continue
            if host._turn_state.active_admission_count() > 0:
                await self._wait_for_pre_admission_gate(
                    host._turn_state.wait_for_active_admissions_idle,
                    preparation_tracker,
                )
                continue
            if not routed:
                routed = True
                if await self._route_turn(event):
                    # The profile switch rebuilt the agent, so every gate above
                    # has to be re-checked against the new build before a slot
                    # is reserved against its generation.
                    continue
            admission = host._turn_state.reserve_prompt_admission(
                kind="fresh",
                session_generation=host.session_generation,
                build_generation=host.build_generation,
                preparation_trace=preparation,
            )
            if admission is None:
                await self._wait_for_pre_admission_gate(
                    host._turn_state.wait_for_prompt_admission_open,
                    preparation_tracker,
                )
                continue
            host._turn_state.deregister_pre_admission_preparation(preparation_tracker.current)
            break

        try:
            # A locked-input submit can outlive its run and arrive here as a
            # fresh turn; honor a cancel that landed while it waited so Esc
            # never lets the withdrawn text start a new turn.
            if await self._abandon_cancelled_injection_submit(event):
                if preparation is not None:
                    await preparation.finished(outcome=PreparationOutcome.CANCELLED)
                return False
            admission_session_id = host._session_id
            decision = await self._evaluate_user_prompt_submit(
                event.text,
                injected=False,
                target_operation_id=preparation.committed_operation_id if preparation is not None else None,
            )
            if not self._admission_owner_is_current(admission):
                if preparation is not None:
                    await preparation.finished(outcome=PreparationOutcome.OWNER_CHANGED)
                return False
            if await self._handle_user_prompt_submit_decision(decision, injected=False):
                if preparation is not None:
                    await preparation.finished(outcome=PreparationOutcome.REJECTED)
                return False

            if event.prepared_contents is not None:
                contents = list(event.prepared_contents)
                if await self.reject_text_only_prepared_contents(
                    event.text,
                    contents,
                    admission=admission,
                    event_session_id=admission_session_id,
                ):
                    if preparation is not None:
                        await preparation.finished(outcome=PreparationOutcome.IMAGE_REJECTED)
                    return False
            else:
                contents = await self.prepare_user_contents(
                    event.text,
                    admission=admission,
                    event_session_id=admission_session_id,
                )
                if contents is None:
                    if preparation is not None:
                        await preparation.finished(outcome=PreparationOutcome.PREPARATION_FAILED)
                    return False
                event.prepared_contents = contents
            if not self._admission_owner_is_current(admission):
                if preparation is not None:
                    await preparation.finished(outcome=PreparationOutcome.OWNER_CHANGED)
                return False

            # Final cancel check after all awaited preparation; task creation
            # below is synchronous, so a cancel can no longer interleave.
            if await self._abandon_cancelled_injection_submit(event):
                if preparation is not None:
                    await preparation.finished(outcome=PreparationOutcome.CANCELLED)
                return False
            if not self._fresh_prompt_fsm_accepts():
                await self._publish_prompt_admission_conflict()
                if preparation is not None:
                    await preparation.finished(outcome=PreparationOutcome.CONFLICT)
                return False
            if host._turn_state.run_task is not None and not host._turn_state.run_task.done():
                await self._publish_prompt_admission_conflict()
                if preparation is not None:
                    await preparation.finished(outcome=PreparationOutcome.CONFLICT)
                return False
            _expire_current_run_scope(host, host._turn_state.current_run_scope)

            # After a failed/interrupted run, remove status markers (interrupted,
            # error, turn_marker) so the new message appends cleanly. Then remove
            # any orphaned user message that got no model response (e.g. immediate
            # 401 error) — this matches the live UX where the failed message
            # disappeared when the user typed a new one. If the model DID produce
            # output (tool calls, partial text) the user message is preserved.
            #
            # Also check the history itself for trailing error markers — after a
            # session restore the FSM is reset to IDLE, but the saved history
            # still carries the error state from the previous run.
            if (
                host._fsm.state in (EngineState.INTERRUPTED, EngineState.FAILED)
                or host._history.has_trailing_error_markers()
            ):
                host._history.remove_trailing_markers()
                host._history.remove_orphaned_user_message()

            reminder_middleware = host._reminder_middleware
            if reminder_middleware is None:
                host._turn_state.release_prompt_admission(admission)
                admission_released = True
                host._fsm.try_transition(Trigger.USER_MESSAGE)
                host._turn_state.run_task = asyncio.create_task(
                    host._run_and_save(
                        event.text,
                        created_at=event.timestamp,
                        contents=contents,
                        admission_preparation=preparation,
                    )
                )
                self._queue_prompt_hook_reminders(decision, injected=False)
                return True

            reminder_scope = reminder_middleware.create_current_run_scope()
            promotion = host._turn_state.promote_fresh_admission_to_run(
                admission,
                reminder_scope=reminder_scope,
                make_task=lambda scope, window: asyncio.create_task(
                    host._run_and_save(
                        event.text,
                        created_at=event.timestamp,
                        contents=contents,
                        run_scope=scope,
                        injection_window=window,
                        admission_preparation=preparation,
                    )
                ),
            )
            if not promotion.promoted:
                reminder_middleware.expire_current_run_scope(reminder_scope)
                if promotion.conflict:
                    await self._publish_prompt_admission_conflict()
                if preparation is not None:
                    await preparation.finished(
                        outcome=PreparationOutcome.CONFLICT if promotion.conflict else PreparationOutcome.OWNER_CHANGED
                    )
                return False
            admission_released = True
            host._fsm.try_transition(Trigger.USER_MESSAGE)
            self._queue_prompt_hook_reminders(decision, injected=False)
            return True
        finally:
            if admission is not None and not admission_released:
                host._turn_state.release_prompt_admission(admission)

    async def _admit_requirement_amendment(
        self,
        workflow: Any,
        event: UserMessage,
        *,
        preparation: PreparationTrace | None,
    ) -> bool:
        """Route late user authority into the live clarification revision."""
        if await self._abandon_cancelled_injection_submit(event):
            if preparation is not None:
                await preparation.finished(outcome=PreparationOutcome.CANCELLED)
            return False
        if await PromptContentPreparer(self._host).reject_injected_images(
            event.text,
            session_id=self._host._session_id,
        ):
            if preparation is not None:
                await preparation.finished(outcome=PreparationOutcome.IMAGE_REJECTED)
            return False
        decision = await self._evaluate_user_prompt_submit(
            event.text,
            injected=True,
            target_operation_id=preparation.committed_operation_id if preparation is not None else None,
        )
        if await self._handle_user_prompt_submit_decision(decision, injected=True):
            if preparation is not None:
                await preparation.finished(outcome=PreparationOutcome.REJECTED)
            return False
        accepted = await workflow.accept_amendment(
            event.text,
            created_at=event.timestamp,
            injection_id=event.injection_id,
        )
        if accepted:
            self._queue_prompt_hook_reminders(decision, injected=True)
        if preparation is not None:
            await preparation.finished(
                outcome=PreparationOutcome.INJECTED if accepted else PreparationOutcome.TARGET_STALE
            )
        return accepted

    async def on_user_interrupt(self, _event: UserInterrupt) -> None:
        """Handle user interrupt.

        Cascades to every live sub-agent controller first so paused
        sub-agents resolve via the cascade-abort branch (otherwise their
        ``pending_decision`` future would keep their ``_invoke``
        coroutine pinned forever and the subsequent task cancel below
        would leak). Then sets the interrupt flag / cancels the parent
        task. It then either binds cancellation to the exact pre-executor
        run task or interrupts the active executor; ``_run_and_save``
        detects ``was_interrupted`` and rolls back history.
        """
        host = self._host
        if host._agent_loading:
            await host._bus.publish(
                Warning(
                    code="agent_loading_interrupt_ignored",
                    message="Interrupt ignored while agent infrastructure is loading.",
                    display_message=_COORDINATOR_INTERRUPT_IGNORED_LOADING.bind(),
                    session_id=host._session_id,
                )
            )
            return
        executor = host._executor
        interrupt_task = host._turn_state.run_task
        host._trajectory_recorder.interrupt_requested_soon()
        workflow = getattr(host, "_requirement_clarification_workflow", None)
        if workflow is not None:
            await workflow.request_stop()
        if executor is not None and not executor.is_running:
            host._turn_state.request_pre_executor_interrupt(interrupt_task)
        if host._sub_agent_tools is not None:
            await host._sub_agent_tools.cascade_abort_all()
        if executor is not None and interrupt_task is not None and host._turn_state.run_task is interrupt_task:
            if executor.is_running:
                await executor.interrupt()
            else:
                host._turn_state.request_pre_executor_interrupt(interrupt_task)
        self.schedule_user_interrupt_hook()

    async def on_user_retry(self, event: UserRetry) -> None:
        """Handle retry requests and route them to immediate or pending execution."""
        await RetryCoordinator(self._host).handle_user_retry(event)

    async def on_user_inject(self, event: UserInject) -> None:
        """Handle user injection (prompt inserted before next model call)."""
        host = self._host
        host._turn_state.begin_inflight_injection(event.injection_id)
        preparation_tracker = PreAdmissionPreparationTracker()
        handed_off = False
        try:
            await self._start_pre_admission_preparation(preparation_tracker)
            await self._wait_for_pre_admission_gate(host._wait_for_agent_load_idle, preparation_tracker)
            preparation = preparation_tracker.preparation
            if host._executor is None:
                if preparation is not None:
                    await preparation.finished(outcome=PreparationOutcome.NOT_READY)
                return
            handed_off = await ActiveTurnInjector(host).inject(
                event.text,
                created_at=event.timestamp,
                route="fsm_active",
                reject_images_without_target=False,
                injection_id=event.injection_id,
                preparation=preparation,
                preparation_tracker=preparation_tracker,
            )
        except asyncio.CancelledError:
            preparation = preparation_tracker.preparation
            if (
                preparation is not None
                and not preparation_tracker.preparation_handed_off
                and not preparation.finished_state
            ):
                preparation.finished_soon(outcome=PreparationOutcome.CANCELLED)
            raise
        except BaseException:
            preparation = preparation_tracker.preparation
            if (
                preparation is not None
                and not preparation_tracker.preparation_handed_off
                and not preparation.finished_state
            ):
                preparation.finished_soon(outcome=PreparationOutcome.PREPARATION_FAILED)
            raise
        finally:
            host._turn_state.deregister_pre_admission_preparation(preparation_tracker.current)
            host._turn_state.finish_inflight_injection(event.injection_id)
            preparation = preparation_tracker.preparation
            if (
                preparation is not None
                and not handed_off
                and not preparation_tracker.preparation_handed_off
                and not preparation.finished_state
            ):
                await preparation.finished(outcome=PreparationOutcome.PREPARATION_FAILED)

    async def on_user_inject_cancel(self, event: UserInjectCancel) -> None:
        """Withdraw a queued mid-run injection before the model sees it.

        Covers both pending phases: an injection already queued on the
        middleware is removed here directly (including its pre-appended
        approval judge context), and an injection still in awaited admission
        aborts via a cancel mark it observes before committing. Everything
        below is synchronous, so the injection cannot move between phases
        mid-cancel. A cancel that matches neither phase is a no-op — the
        injection already resolved (for a consumed one, the ``consumed=True``
        result tells frontends the text reached the model) and recording a
        mark would leave it dangling for the rest of the session.
        """
        host = self._host
        injection_id = event.injection_id
        if not injection_id:
            return
        executor = host._executor
        if executor is None:
            if host._turn_state.is_injection_inflight(injection_id):
                host._turn_state.mark_injection_cancelled(injection_id)
            return
        removed = executor.cancel_injection(injection_id)
        if removed is None:
            if host._turn_state.is_injection_inflight(injection_id):
                host._turn_state.mark_injection_cancelled(injection_id)
            return
        executor.remove_user_message(removed.text)
        withdraw_committed_injection_reminders(
            host._reminder_middleware,
            host._turn_state.current_run_scope,
            removed,
        )
        if removed.preparation is not None:
            removed.preparation.finished_soon(
                outcome=PreparationOutcome.CANCELLED,
                target_turn_id=removed.target_turn_id,
            )
        await host._bus.publish(
            UserInjectResult(
                text=removed.text,
                consumed=False,
                created_at=removed.created_at,
                injection_id=injection_id,
                session_id=host._session_id,
            )
        )

    async def _abandon_cancelled_injection_submit(self, event: UserMessage) -> bool:
        """Publish an abandoned result and return True for a cancelled locked submit."""
        host = self._host
        if not host._turn_state.discard_injection_cancellation(event.injection_id):
            return False
        await host._bus.publish(
            UserInjectResult(
                text=event.text,
                consumed=False,
                created_at=event.timestamp,
                injection_id=event.injection_id,
                session_id=host._session_id,
            )
        )
        return True

    def check_pending_retry(self) -> None:
        """If a retry was queued while the executor was running, start it now."""
        RetryCoordinator(self._host).start_pending_retry_if_due()

    async def prepare_user_contents(
        self,
        text: str,
        *,
        discovered: AttachmentDiscoveryResult | None = None,
        admission: PromptAdmissionScope | None = None,
        event_session_id: str | None = None,
    ) -> list[Any] | None:
        """Return user-message contents or publish a recoverable attachment error."""
        result = await self._prompt_content_preparer_factory(self._host).prepare_fresh(
            text,
            discovered=discovered,
            event_session_id=event_session_id,
            should_publish=self._publication_guard_for_admission(admission),
        )
        return None if result is None else result.contents

    async def reject_text_only_prepared_contents(
        self,
        text: str,
        contents: list[Any],
        *,
        admission: PromptAdmissionScope,
        event_session_id: str | None,
    ) -> bool:
        """Reject prebuilt image contents before a fresh prompt is promoted."""
        return await self._prompt_content_preparer_factory(self._host).reject_text_only_prepared_contents(
            text,
            contents,
            event_session_id=event_session_id,
            should_publish=self._publication_guard_for_admission(admission),
        )

    async def _wait_for_existing_run_task(self) -> None:
        """Wait for current run-task cleanup, if any."""
        task = self._host._turn_state.run_task
        if task is not None and not task.done():
            await task

    async def _route_turn(self, event: UserMessage) -> bool:
        """Classify this turn and report whether a profile switch rebuilt the agent.

        Routing is an optimization layer over a turn that is already valid, so
        a failure here must cost the classification and nothing else: losing
        the user's message to a router bug would be far worse than running it
        on the standard pass.
        """
        try:
            return await self._route_turn_unguarded(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.warning("turn routing failed; running the standard pass", exc_info=True)
            self._host._last_route = None
            return False

    async def _route_turn_unguarded(self, event: UserMessage) -> bool:
        host = self._host
        router = TurnRouter(
            host,
            workspace_cwd=host._workspace.primary_cwd if host._workspace is not None else safe_getcwd(),
            switch_profile=self._switch_routing_profile,
        )
        decision = await router.apply(await router.decide(event.text, turn=host._turn_number + 1))
        await router.publish(decision, turn=host._turn_number + 1)
        await host._trajectory_recorder.turn_routed(
            track=decision.track.value,
            band=decision.band.value,
            source=decision.source,
            confidence=decision.confidence,
            prompt_score=decision.prompt_score,
            plan_pact=decision.plan.pact,
            switched_to=decision.switched_to,
            tiebreaker_failure=decision.tiebreaker_failure,
        )
        return bool(decision.switched_to)

    async def _switch_routing_profile(self, target: str) -> bool:
        """Soft-restart onto *target*, keeping history, and report success.

        The same path as a manual ``#Profile`` switch, so history preservation
        and the model's switch reminder come for free rather than being
        reimplemented for routing.
        """
        host = self._host
        registry = host._agent_registry
        if registry is None or registry.get(target) is None:
            return False
        async with host._bus.stream(ProfileSwitched, Error) as events:
            await host._bus.publish(AgentProfileSwitch(profile_name=target, session_id=host._session_id))
            try:
                async with asyncio.timeout(_ROUTE_SWITCH_TIMEOUT_SECONDS):
                    async for observed in events:
                        if isinstance(observed, ProfileSwitched):
                            return observed.to_profile == target
                        return False
            except TimeoutError:
                return False
        return False

    def _admission_owner_is_current(self, admission: PromptAdmissionScope) -> bool:
        """Return whether *admission* still belongs to the live session/build owner."""
        host = self._host
        return (
            host.session_generation == admission.session_generation
            and host.build_generation == admission.build_generation
        )

    def _publication_guard_for_admission(
        self,
        admission: PromptAdmissionScope | None,
    ) -> Callable[[], bool] | None:
        """Return a prompt-preparation side-effect guard for an exact admission."""
        if admission is None:
            return None
        return lambda: self._admission_owner_is_current(admission)

    def _fresh_prompt_fsm_accepts(self) -> bool:
        """Return whether the current FSM state can start a fresh user turn."""
        return self._host._fsm.state in (
            EngineState.UNINITIALIZED,
            EngineState.IDLE,
            EngineState.INTERRUPTED,
            EngineState.FAILED,
        )

    async def _publish_prompt_admission_conflict(self) -> None:
        """Publish a deterministic conflict when a fresh prompt cannot promote."""
        host = self._host
        await host._bus.publish(
            Error(
                code="prompt_admission_conflict",
                message="Prompt could not be admitted because another turn started.",
                display_message=_COORDINATOR_PROMPT_ADMISSION_CONFLICT.bind(),
                session_id=host._session_id,
            )
        )

    async def _evaluate_user_prompt_submit(
        self,
        text: str,
        *,
        injected: bool | None,
        target_operation_id: str | None = None,
    ) -> HookDecision | None:
        """Run ``user_prompt_submit`` hooks without applying reminder side effects."""
        return await PromptSubmitGate(self._host).evaluate(
            text,
            injected=injected,
            target_operation_id=target_operation_id,
        )

    def _open_pre_turn_preparation(self) -> PreparationTrace | None:
        """Open a session-root preparation scope before prompt admission awaits."""
        context = self._host._trajectory_recorder.context()
        if context is not None:
            context = context.with_turn(None).with_run(None)
        return PreparationTrace.open(
            scope=PreparationScope.PRE_TURN,
            phase="input_admission",
            context=context,
        )

    async def _start_pre_admission_preparation(self, tracker: PreAdmissionPreparationTracker) -> None:
        """Start and register a preparation against the current session context."""
        preparation = self._open_pre_turn_preparation()
        if preparation is None:
            tracker.current = None
            return
        entry = PreAdmissionPreparationEntry(preparation=preparation)
        tracker.current = entry
        await preparation.started()
        self._host._turn_state.register_pre_admission_preparation(entry)

    async def _wait_for_pre_admission_gate(
        self,
        wait_for_gate: Callable[[], Awaitable[Any]],
        tracker: PreAdmissionPreparationTracker,
    ) -> None:
        """Wait at one gate and rebind a preparation swept by a session boundary."""
        entry = tracker.current
        preparation = tracker.preparation
        await input_admission_wait(wait_for_gate, preparation, entry)
        if preparation is None or not preparation.finished_state:
            return
        self._host._turn_state.deregister_pre_admission_preparation(entry)
        await self._start_pre_admission_preparation(tracker)

    async def _handle_user_prompt_submit_decision(
        self,
        decision: HookDecision | None,
        *,
        injected: bool | None,
        session_id: str | None = None,
    ) -> bool:
        """Return True after publishing when a prompt-submit hook blocked."""
        return await PromptSubmitGate(self._host).handle_decision(
            decision,
            injected=injected,
            session_id=session_id,
        )

    def _queue_prompt_hook_reminders(
        self,
        decision: HookDecision | None,
        *,
        injected: bool | None,
    ) -> None:
        """Apply non-blocking prompt-submit hook reminders after prompt validation passes."""
        PromptSubmitGate(self._host).queue_reminders(decision, injected=injected)

    def schedule_user_interrupt_hook(self) -> None:
        """Schedule the asynchronous user-interrupt lifecycle hook."""
        TurnHookDispatcher(self._host).schedule_user_interrupt()


def prompt_content_preparer(host: TurnRunnerHost) -> PromptContentPreparer:
    """Return a prompt content preparer using coordinator compatibility hooks."""
    return PromptContentPreparer(
        host,
        discover_mentions=discover_image_mentions,
        discover_references=discover_image_references,
        load_attachments=load_image_attachments,
        compression_timeout_seconds=_IMAGE_COMPRESSION_TIMEOUT_SECONDS,
    )
