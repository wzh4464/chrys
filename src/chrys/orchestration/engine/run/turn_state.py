# Copyright (c) 2026 Chrys. All rights reserved.

"""Owned mutable runtime state for main-agent turn orchestration."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from weakref import WeakSet

from chrys.foundation.models.turns import UserMessageKind
from chrys.service.trajectory.preparation import PreparationOutcome, PreparationTrace
from chrys.service.trajectory.waits import WaitOutcome, WaitTrace

if TYPE_CHECKING:
    from chrys.orchestration.engine.executor import Executor
    from chrys.service.agent_middleware.system_reminder import (
        CurrentRunReminderScope,
        CurrentRunReminderTarget,
        SystemReminderMiddleware,
    )


@dataclass(frozen=True)
class RunTaskDrainOutcome:
    """Outcome from observing a run-task chain for a rebuild/session boundary."""

    cancelled: bool = False


@dataclass
class PreAdmissionPreparationEntry:
    """Registered pre-turn preparation that has not reached an admission record."""

    preparation: PreparationTrace
    current_wait: WaitTrace | None = field(default=None, repr=False, compare=False)


@dataclass
class PreAdmissionPreparationTracker:
    """Mutable worker-local pointer to the current pre-admission preparation."""

    current: PreAdmissionPreparationEntry | None = None
    preparation_handed_off: bool = False
    """Whether the current trace moved to a longer-lived owner."""

    @property
    def preparation(self) -> PreparationTrace | None:
        """Return the worker's current preparation trace, if recording is enabled."""
        return self.current.preparation if self.current is not None else None


@dataclass(frozen=True)
class PromptAdmissionRecord:
    """Visible prompt/retry admission while awaited validation is in progress."""

    admission_id: int
    kind: Literal["fresh", "retry"]
    session_generation: int
    build_generation: int
    created_at_monotonic: float
    preparation_trace: PreparationTrace | None = field(default=None, repr=False, compare=False)


@dataclass
class PendingRetry:
    """Retry request queued to run after the current executor call returns."""

    text: str = ""
    created_at: datetime | str | None = None
    session_generation: int = 0
    run_generation: int = 0
    updated_by_admission_id: int | None = None
    owner_admission_id: int | None = None
    dispatch_disabled: bool = False
    preparation_trace: PreparationTrace | None = field(default=None, repr=False, compare=False)


@dataclass
class CurrentTurnInput:
    """Current prompt data used by crash-recovery checkpoint writes.

    ``kind`` records how a recovery checkpoint must re-create the message
    (``"opener"`` unflagged, ``"injected"`` flagged mid-turn input) — see
    :func:`chrys.foundation.models.turns.user_text_matches`. Empty-input
    resumes clear this field; opener replay registers the popped anchor.
    """

    text: str = ""
    contents: list[Any] | None = None
    created_at: datetime | str | None = None
    kind: UserMessageKind = "opener"


@dataclass(frozen=True)
class PromptAdmissionScope:
    """Exact prompt/retry admission token."""

    admission_id: int
    kind: Literal["fresh", "retry"]
    session_generation: int
    build_generation: int
    preparation_trace: PreparationTrace | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class CurrentRunScope:
    """Orchestration-owned identity for one logical main-agent run."""

    run_generation: int
    session_generation: int
    build_generation: int
    owner_admission_id: int
    reminder_scope: CurrentRunReminderScope


@dataclass(frozen=True)
class CurrentRunInjectionWindow:
    """Injectable executor-pass window for a current run."""

    run_generation: int
    window_generation: int


@dataclass(frozen=True)
class ActiveInjectionTarget:
    """Operation-local target captured before awaited active-injection work."""

    route: Literal["fsm_active", "executor_fallback"]
    session_id: str | None
    session_generation: int
    build_generation: int
    current_run_scope: CurrentRunScope
    run_task: asyncio.Task[None]
    executor: Executor
    reminder_middleware: SystemReminderMiddleware
    reminder_target: CurrentRunReminderTarget
    injection_window: CurrentRunInjectionWindow
    trajectory_turn_id: str | None = None


@dataclass(frozen=True)
class PromptPromotionResult:
    """Result from promoting a fresh prompt admission into a run task."""

    promoted: bool
    conflict: bool = False
    stale: bool = False


@dataclass(frozen=True)
class RetryPromotionResult:
    """Result from promoting a retry admission into a task or pending retry."""

    outcome: Literal["task", "pending_installed", "pending_updated", "stale", "invalid"]
    task: asyncio.Task[None] | None = None


@dataclass
class TurnRuntimeState:
    """Single owner for mutable per-turn state during the migration."""

    next_admission_id: int = 1
    active_admissions: dict[int, PromptAdmissionRecord] = field(default_factory=dict)
    pre_admission_preparations: dict[str, PreAdmissionPreparationEntry] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    active_admissions_idle: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)
    prompt_admission_closed: bool = False
    prompt_admission_close_owner: str | None = None
    prompt_admission_open: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)
    run_task: asyncio.Task[None] | None = None
    current_run_scope: CurrentRunScope | None = None
    next_run_generation: int = 1
    conversation_revision: int = 0
    injection_window_generation: int = 0
    injection_admission_open: bool = False
    active_injection_commits: int = 0
    active_injection_commits_idle: asyncio.Event = field(default_factory=asyncio.Event, repr=False, compare=False)
    pending_retry: PendingRetry = field(default_factory=PendingRetry)
    pending_retry_dispatch_disabled_for_session_generation: int | None = None
    current_input: CurrentTurnInput = field(default_factory=CurrentTurnInput)
    history_start_index: int = 0
    cancelled_injection_ids: set[str] = field(default_factory=set)
    inflight_injection_ids: dict[str, int] = field(default_factory=dict)
    _successfully_saved_run_tasks: WeakSet[asyncio.Task[Any]] = field(
        default_factory=WeakSet,
        repr=False,
        compare=False,
    )
    _pre_executor_interrupt_tasks: WeakSet[asyncio.Task[Any]] = field(
        default_factory=WeakSet,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Initialize waitable idle markers for freshly constructed state."""
        if not self.active_admissions:
            self.active_admissions_idle.set()
        if not self.prompt_admission_closed:
            self.prompt_admission_open.set()
        if self.active_injection_commits == 0:
            self.active_injection_commits_idle.set()

    def record_current_run_final_save(self) -> None:
        """Record that the current lifecycle task completed its final save."""
        task = asyncio.current_task()
        if task is not None:
            self._successfully_saved_run_tasks.add(task)

    def was_run_task_finally_saved(self, task: asyncio.Task[Any]) -> bool:
        """Return whether *task* completed a successful final session save."""
        return task in self._successfully_saved_run_tasks

    def request_pre_executor_interrupt(self, task: asyncio.Task[Any] | None) -> bool:
        """Bind a pre-executor interrupt to an exact captured run task."""
        if task is None or task.done():
            return False
        self._pre_executor_interrupt_tasks.add(task)
        return True

    def consume_pre_executor_interrupt(self) -> bool:
        """Consume an interrupt only when it belongs to the calling run task."""
        task = asyncio.current_task()
        if task is None or task not in self._pre_executor_interrupt_tasks:
            return False
        self._pre_executor_interrupt_tasks.discard(task)
        return True

    def discard_pre_executor_interrupt(self, task: asyncio.Task[Any] | None) -> None:
        """Drop a task-scoped interrupt at that task's terminal boundary."""
        if task is not None:
            self._pre_executor_interrupt_tasks.discard(task)

    def advance_conversation_revision(self) -> int:
        """Advance the monotonic revision for one fresh or retry lifecycle."""
        self.conversation_revision += 1
        return self.conversation_revision

    def register_pre_admission_preparation(self, entry: PreAdmissionPreparationEntry) -> None:
        """Expose a started preparation until it reaches durable admission state."""
        self.pre_admission_preparations[entry.preparation.operation_id] = entry

    def deregister_pre_admission_preparation(self, entry: PreAdmissionPreparationEntry | None) -> None:
        """Remove *entry* when it is still the registered owner."""
        if entry is None:
            return
        operation_id = entry.preparation.operation_id
        if self.pre_admission_preparations.get(operation_id) is entry:
            del self.pre_admission_preparations[operation_id]

    def clear_pre_admission_preparations(self) -> None:
        """Settle preparations and straddling waits that have not reached admission."""
        entries = tuple(self.pre_admission_preparations.values())
        self.pre_admission_preparations.clear()
        for entry in entries:
            wait = entry.current_wait
            if wait is not None:
                wait.finished_soon(outcome=WaitOutcome.CANCELLED)
                entry.current_wait = None
            entry.preparation.finished_soon(outcome=PreparationOutcome.OWNER_CHANGED)

    def reserve_prompt_admission(
        self,
        *,
        kind: Literal["fresh", "retry"],
        session_generation: int,
        build_generation: int,
        preparation_trace: PreparationTrace | None = None,
    ) -> PromptAdmissionScope | None:
        """Reserve an exact prompt/retry admission unless admission is closed."""
        if self.prompt_admission_closed:
            return None
        admission_id = self.next_admission_id
        self.next_admission_id += 1
        self.active_admissions_idle.clear()
        self.active_admissions[admission_id] = PromptAdmissionRecord(
            admission_id=admission_id,
            kind=kind,
            session_generation=session_generation,
            build_generation=build_generation,
            preparation_trace=preparation_trace,
            created_at_monotonic=time.monotonic(),
        )
        return PromptAdmissionScope(
            admission_id=admission_id,
            kind=kind,
            session_generation=session_generation,
            build_generation=build_generation,
            preparation_trace=preparation_trace,
        )

    def active_admission_count(self) -> int:
        """Return the number of exact active admissions."""
        return len(self.active_admissions)

    def has_active_admission_kind(self, kind: Literal["fresh", "retry"]) -> bool:
        """Return whether an active admission of *kind* is currently preparing."""
        return any(record.kind == kind for record in self.active_admissions.values())

    def release_prompt_admission(self, admission: PromptAdmissionScope) -> bool:
        """Release an exact admission; duplicate releases are no-ops."""
        record = self.active_admissions.get(admission.admission_id)
        if record is None:
            return False
        if (
            record.kind != admission.kind
            or record.session_generation != admission.session_generation
            or record.build_generation != admission.build_generation
        ):
            return False
        del self.active_admissions[admission.admission_id]
        if not self.active_admissions:
            self.active_admissions_idle.set()
        return True

    async def wait_for_active_admissions_idle(self) -> None:
        """Wait until no exact prompt/retry admission is active."""
        while self.active_admissions:
            await self.active_admissions_idle.wait()

    def begin_current_run_scope(
        self,
        *,
        owner_admission_id: int,
        session_generation: int,
        build_generation: int,
        reminder_scope: CurrentRunReminderScope,
    ) -> CurrentRunScope:
        """Create and install a current-run scope."""
        scope = CurrentRunScope(
            run_generation=self.next_run_generation,
            session_generation=session_generation,
            build_generation=build_generation,
            owner_admission_id=owner_admission_id,
            reminder_scope=reminder_scope,
        )
        self.next_run_generation += 1
        self.current_run_scope = scope
        return scope

    def open_injection_admission(self, scope: CurrentRunScope) -> CurrentRunInjectionWindow:
        """Open injection admission for the current executor pass."""
        self.injection_window_generation += 1
        self.injection_admission_open = self.current_run_scope == scope
        return CurrentRunInjectionWindow(
            run_generation=scope.run_generation,
            window_generation=self.injection_window_generation,
        )

    def close_injection_admission(self, scope: CurrentRunScope) -> None:
        """Close injection admission for *scope* if it is current."""
        if self.current_run_scope == scope:
            self.injection_admission_open = False

    def capture_current_injection_window(self, scope: CurrentRunScope) -> CurrentRunInjectionWindow | None:
        """Capture the currently open injection window for *scope*, if any."""
        if self.current_run_scope != scope or not self.injection_admission_open:
            return None
        return CurrentRunInjectionWindow(
            run_generation=scope.run_generation,
            window_generation=self.injection_window_generation,
        )

    def is_injection_admission_current(self, window: CurrentRunInjectionWindow) -> bool:
        """Return whether *window* still admits active-turn injection."""
        scope = self.current_run_scope
        return (
            self.injection_admission_open
            and scope is not None
            and scope.run_generation == window.run_generation
            and self.injection_window_generation == window.window_generation
        )

    def begin_inflight_injection(self, injection_id: str | None) -> None:
        """Track one in-flight admission of a frontend-identified injection.

        While an id is tracked, a user cancel for it is recorded as a mark
        (:meth:`mark_injection_cancelled`) that the admission observes at its
        cancellation checkpoints. No-op for id-less submits.
        """
        if not injection_id:
            return
        self.inflight_injection_ids[injection_id] = self.inflight_injection_ids.get(injection_id, 0) + 1

    def finish_inflight_injection(self, injection_id: str | None) -> None:
        """Stop tracking one in-flight admission for *injection_id*.

        When the last admission for the id exits, any cancel mark it did not
        observe (e.g. the admission aborted for an unrelated reason first) is
        dropped with it — nothing can observe the mark afterwards.
        """
        if not injection_id:
            return
        count = self.inflight_injection_ids.get(injection_id, 0)
        if count > 1:
            self.inflight_injection_ids[injection_id] = count - 1
            return
        self.inflight_injection_ids.pop(injection_id, None)
        self.cancelled_injection_ids.discard(injection_id)

    def is_injection_inflight(self, injection_id: str) -> bool:
        """Return whether an admission for *injection_id* is currently in flight."""
        return injection_id in self.inflight_injection_ids

    def mark_injection_cancelled(self, injection_id: str) -> None:
        """Record a user cancel for an injection whose admission is in flight.

        Callers must only mark ids that something can still observe (an
        in-flight admission — see :meth:`is_injection_inflight`); the mark is
        consumed at the admission's next cancellation checkpoint and dropped
        at admission end otherwise, so it can never outlive its injection.
        """
        self.cancelled_injection_ids.add(injection_id)

    def is_injection_cancelled(self, injection_id: str | None) -> bool:
        """Return whether *injection_id* carries an unconsumed cancel mark."""
        return injection_id is not None and injection_id in self.cancelled_injection_ids

    def discard_injection_cancellation(self, injection_id: str | None) -> bool:
        """Consume the cancel mark for *injection_id*; returns whether it was set."""
        if injection_id is None or injection_id not in self.cancelled_injection_ids:
            return False
        self.cancelled_injection_ids.discard(injection_id)
        return True

    def begin_active_injection_commit(self, target: ActiveInjectionTarget) -> bool:
        """Begin a commit guard when the captured active-injection target is current."""
        if target.current_run_scope != self.current_run_scope:
            return False
        if not self.is_injection_admission_current(target.injection_window):
            return False
        self.active_injection_commits += 1
        self.active_injection_commits_idle.clear()
        return True

    def finish_active_injection_commit(self) -> None:
        """Release one active-injection commit guard."""
        if self.active_injection_commits > 0:
            self.active_injection_commits -= 1
        if self.active_injection_commits == 0:
            self.active_injection_commits_idle.set()

    async def wait_for_active_injection_commits_idle(self) -> None:
        """Wait until active-injection commit guards have drained."""
        while self.active_injection_commits > 0:
            await self.active_injection_commits_idle.wait()

    def promote_fresh_admission_to_run(
        self,
        admission: PromptAdmissionScope,
        *,
        reminder_scope: CurrentRunReminderScope,
        make_task: Callable[[CurrentRunScope, CurrentRunInjectionWindow], asyncio.Task[None]],
    ) -> PromptPromotionResult:
        """Promote an exact fresh admission into a visible run task."""
        if not self._admission_matches(admission, expected_kind="fresh"):
            return PromptPromotionResult(promoted=False, stale=True)
        if self.current_run_scope is not None or (self.run_task is not None and not self.run_task.done()):
            return PromptPromotionResult(promoted=False, conflict=True)
        scope = self.begin_current_run_scope(
            owner_admission_id=admission.admission_id,
            session_generation=admission.session_generation,
            build_generation=admission.build_generation,
            reminder_scope=reminder_scope,
        )
        window = self.open_injection_admission(scope)
        self.run_task = make_task(scope, window)
        self.release_prompt_admission(admission)
        return PromptPromotionResult(promoted=True)

    def promote_retry_admission(
        self,
        admission: PromptAdmissionScope,
        *,
        reminder_scope: CurrentRunReminderScope,
        make_task: Callable[[CurrentRunScope, CurrentRunInjectionWindow], asyncio.Task[None]],
        text: str,
        created_at: datetime | str | None,
        pending: bool = False,
    ) -> RetryPromotionResult:
        """Promote an exact retry admission into a task or pending retry."""
        if not self._admission_matches(admission, expected_kind="retry"):
            return RetryPromotionResult(outcome="stale")
        if pending:
            updated = self.upsert_pending_retry_from_admission(admission, text, created_at)
            self.release_prompt_admission(admission)
            return RetryPromotionResult(outcome="pending_updated" if updated else "pending_installed")
        if self.run_task is not None and not self.run_task.done():
            return RetryPromotionResult(outcome="invalid")
        scope = self._retry_scope_for_admission(admission, reminder_scope=reminder_scope)
        window = self.open_injection_admission(scope)
        task = make_task(scope, window)
        self.run_task = task
        self.release_prompt_admission(admission)
        return RetryPromotionResult(outcome="task", task=task)

    def close_prompt_admission_for_rebuild(self, owner: str = "rebuild") -> None:
        """Close fresh prompt/retry admission for a future rebuild fence."""
        self.prompt_admission_closed = True
        self.prompt_admission_close_owner = owner
        self.prompt_admission_open.clear()

    def reopen_prompt_admission_after_rebuild(self, owner: str = "rebuild") -> None:
        """Reopen prompt/retry admission if *owner* owns the close."""
        if self.prompt_admission_close_owner in (None, owner):
            self.prompt_admission_closed = False
            self.prompt_admission_close_owner = None
            self.prompt_admission_open.set()

    async def wait_for_prompt_admission_open(self) -> None:
        """Wait until fresh prompt/retry admission is open."""
        while self.prompt_admission_closed:
            await self.prompt_admission_open.wait()

    def clear_current_run_scope(self, scope: CurrentRunScope) -> None:
        """Clear the current-run scope if *scope* is still installed."""
        if self.current_run_scope == scope:
            self.current_run_scope = None
            self.injection_admission_open = False

    def clear_pending_retry(self, *, outcome: Literal["dropped", "retry_turn"]) -> PendingRetry:
        """Clear pending retry state and terminalize its preparation scope."""
        pending = self.pending_retry
        if pending.preparation_trace is not None:
            pending.preparation_trace.finished_soon(outcome=outcome)
        self.pending_retry = PendingRetry()
        return pending

    def clear_active_admissions(self, *, outcome: str) -> None:
        """Clear active admissions and terminalize their preparation scopes."""
        for admission in self.active_admissions.values():
            if admission.preparation_trace is not None:
                admission.preparation_trace.finished_soon(outcome=outcome)
        self.active_admissions.clear()
        self.active_admissions_idle.set()

    def upsert_pending_retry_from_admission(
        self,
        admission: PromptAdmissionScope,
        text: str,
        created_at: datetime | str | None,
    ) -> bool:
        """Install or update pending retry for the current run from *admission*."""
        run_generation = self.current_run_scope.run_generation if self.current_run_scope is not None else 0
        existing_same_run = (
            self.pending_retry.owner_admission_id is not None
            and self.pending_retry.session_generation == admission.session_generation
            and self.pending_retry.run_generation == run_generation
        )
        if (
            existing_same_run
            and self.pending_retry.updated_by_admission_id is not None
            and admission.admission_id < self.pending_retry.updated_by_admission_id
        ):
            if admission.preparation_trace is not None:
                admission.preparation_trace.finished_soon(outcome=PreparationOutcome.SUPERSEDED)
            return True
        if existing_same_run and self.pending_retry.preparation_trace is not None:
            self.pending_retry.preparation_trace.finished_soon(outcome=PreparationOutcome.SUPERSEDED)
        if admission.preparation_trace is not None:
            admission.preparation_trace.state_soon("requeued" if existing_same_run else "queued")
        self.pending_retry = PendingRetry(
            text=text,
            created_at=created_at,
            session_generation=admission.session_generation,
            run_generation=run_generation,
            updated_by_admission_id=admission.admission_id,
            owner_admission_id=self.pending_retry.owner_admission_id if existing_same_run else admission.admission_id,
            dispatch_disabled=False,
            preparation_trace=admission.preparation_trace,
        )
        return existing_same_run

    def disable_pending_retry_dispatch_for_session_transition(self, session_generation: int) -> None:
        """Disable pending retry dispatch for an abandoned session generation."""
        self.pending_retry_dispatch_disabled_for_session_generation = session_generation
        if self.pending_retry.session_generation == session_generation:
            self.pending_retry.dispatch_disabled = True

    def set_current_input(
        self,
        text: str,
        contents: list[Any] | None,
        created_at: datetime | str | None,
        kind: UserMessageKind = "opener",
    ) -> None:
        """Set current prompt data for recovery checkpointing."""
        self.current_input = CurrentTurnInput(text=text, contents=contents, created_at=created_at, kind=kind)

    def clear_current_input(self) -> None:
        """Clear current prompt data after run finalization."""
        self.current_input = CurrentTurnInput()

    def invalidate_for_session_transition_pre_shutdown(
        self,
        *,
        old_session_generation: int,
        prompt_admission_owner: str | None = None,
    ) -> None:
        """Invalidate old-session admission/injection/pending-retry state before shutdown."""
        if prompt_admission_owner is not None:
            self.close_prompt_admission_for_rebuild(prompt_admission_owner)
            self.clear_active_admissions(outcome=PreparationOutcome.OWNER_CHANGED)
        self.clear_pre_admission_preparations()
        self.injection_admission_open = False
        self.disable_pending_retry_dispatch_for_session_transition(old_session_generation)

    def reset_after_session_shutdown(
        self,
        *,
        prompt_admission_owner: str | None = None,
    ) -> CurrentRunScope | None:
        """Clear turn-owned state after shutdown has observed the old task."""
        old_scope = self.current_run_scope
        self.run_task = None
        self.current_run_scope = None
        self.injection_admission_open = False
        # ``AgentEngine.shutdown()`` terminalizes pending-retry and admission
        # preparations before it closes the trajectory recorder. This reset
        # runs after shutdown and must never try to emit against the closed writer.
        self.pending_retry = PendingRetry()
        self.current_input = CurrentTurnInput()
        self.history_start_index = 0
        self._pre_executor_interrupt_tasks.clear()
        self.cancelled_injection_ids.clear()
        self.inflight_injection_ids.clear()
        self.active_admissions.clear()
        self.pre_admission_preparations.clear()
        self.active_admissions_idle.set()
        self.active_injection_commits = 0
        self.active_injection_commits_idle.set()
        if prompt_admission_owner is None:
            self.prompt_admission_closed = False
            self.prompt_admission_close_owner = None
            self.prompt_admission_open.set()
        else:
            self.prompt_admission_closed = True
            self.prompt_admission_close_owner = prompt_admission_owner
            self.prompt_admission_open.clear()
        return old_scope

    def _admission_matches(
        self,
        admission: PromptAdmissionScope,
        *,
        expected_kind: Literal["fresh", "retry"],
    ) -> bool:
        record = self.active_admissions.get(admission.admission_id)
        return (
            record is not None
            and admission.kind == expected_kind
            and record.kind == expected_kind
            and record.session_generation == admission.session_generation
            and record.build_generation == admission.build_generation
        )

    def _retry_scope_for_admission(
        self,
        admission: PromptAdmissionScope,
        *,
        reminder_scope: CurrentRunReminderScope,
    ) -> CurrentRunScope:
        """Return the retry scope for *admission*, rebinding an existing logical retry scope."""
        current = self.current_run_scope
        if (
            current is not None
            and current.session_generation == admission.session_generation
            and current.build_generation == admission.build_generation
        ):
            scope = CurrentRunScope(
                run_generation=current.run_generation,
                session_generation=current.session_generation,
                build_generation=current.build_generation,
                owner_admission_id=admission.admission_id,
                reminder_scope=current.reminder_scope,
            )
            self.current_run_scope = scope
            return scope
        return self.begin_current_run_scope(
            owner_admission_id=admission.admission_id,
            session_generation=admission.session_generation,
            build_generation=admission.build_generation,
            reminder_scope=reminder_scope,
        )
