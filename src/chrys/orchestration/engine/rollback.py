# Copyright (c) 2026 Chrys. All rights reserved.

"""Engine-internal companion module for rollback orchestration.

Pure rollback-file helpers live in :mod:`chrys.service.mutations.snapshot_files`.
This module owns the engine-facing workflow: validate the requested
target, optionally revert file mutations, promote a saved snapshot, and
reload the session while suppressing stale in-memory saves.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from chrys.foundation.events.types import Error, RollbackResult, SessionRestore, UserRollback, Warning
from chrys.foundation.i18n import DisplayBlock, msg
from chrys.foundation.util.async_tasks import await_task_quiescence
from chrys.foundation.util.lock import FileLock
from chrys.orchestration.engine.state.machine import EngineState
from chrys.service.mutations.snapshot_files import (
    rollback_snapshot_for_target,
    rollback_snapshot_paths,
    snapshot_target_turn,
    snapshot_target_turns,
    write_rollback_snapshot,
)
from chrys.service.mutations.snapshot_files import (
    turn_prompt_previews as collect_turn_prompt_previews,
)
from chrys.service.mutations.workspace_changes import format_partial_revert_notice, format_retained_changes_notice
from chrys.service.requirement_clarification.artifacts import prune_workflow_artifacts_after_turn
from chrys.service.state.store import SESSION_BACKUP_FILE_NAME

if TYPE_CHECKING:
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.events.bus import EventBus
    from chrys.orchestration.engine.state.machine import EngineStateMachine
    from chrys.orchestration.engine.trajectory import TrajectoryRecorder
    from chrys.service.mutations.coordination import MutationCoordinator
    from chrys.service.mutations.tracker import MutationTracker
    from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker
    from chrys.service.session.history import SessionHistoryManager


logger = logging.getLogger(__name__)

_ROLLBACK_NO_SESSION = msg(
    "rollback.no_session",
    fallback="No active session to roll back.",
)
_ROLLBACK_CONVERSATION_CHANGED = msg(
    "rollback.conversation_changed",
    fallback="Rollback cancelled because the conversation changed after the picker was loaded.",
)
_ROLLBACK_CONVERSATION_ADVANCED = msg(
    "rollback.conversation_advanced",
    fallback="Rollback cancelled because the conversation advanced from turn {expected_turn} to turn {current_turn}.",
)
_ROLLBACK_RUNTIME_CHANGED = msg(
    "rollback.runtime_changed",
    fallback="Rollback cancelled because the workspace or runtime changed after the picker was loaded.",
)
_ROLLBACK_RELATIVE_TURNS_INVALID = msg(
    "rollback.relative_turns_invalid",
    fallback="relative_turns must be positive.",
)
_ROLLBACK_TURNS_UNAVAILABLE = msg(
    "rollback.turns_unavailable",
    fallback="Cannot roll back {requested_turns} turns; the session currently has {current_turns}.",
)
_ROLLBACK_TARGET_TURN_INVALID = msg(
    "rollback.target_turn_invalid",
    fallback="target_turn must be >= 0.",
)
_ROLLBACK_TURN_UNAVAILABLE = msg(
    "rollback.turn_unavailable",
    fallback="Cannot roll back to turn {target_turn}; available turns: {available}",
)
_ROLLBACK_RESET_FAILED = msg(
    "rollback.reset_failed",
    fallback="Rollback to welcome could not reset the session because the session state is busy.",
)
_ROLLBACK_SNAPSHOT_MISSING = msg(
    "rollback.snapshot_missing",
    fallback="Snapshot for turn {target_turn} is missing.",
)
_ROLLBACK_SWAP_LOCKED = msg(
    "rollback.swap_locked",
    fallback="Timed out waiting for session lock: {detail}",
    multiline=True,
)
_ROLLBACK_SWAP_FAILED = msg(
    "rollback.swap_failed",
    fallback="Failed to restore snapshot: {detail}",
    multiline=True,
)
_ROLLBACK_SESSION_CHANGED = msg(
    "rollback.session_changed",
    fallback="Rollback cancelled because the active session changed.",
)
_ROLLBACK_REFUSED = msg(
    "rollback.refused",
    fallback="Rollback is not allowed in state {state}.",
)


@dataclass(slots=True)
class _RollbackLockLease:
    """Idempotent owner for a pre-commit rollback write lock."""

    lock: FileLock | None = None

    @property
    def held(self) -> bool:
        """Return whether the lease still owns its lock."""
        return self.lock is not None

    def release(self) -> None:
        """Release the lock once, tolerating defensive cleanup paths."""
        lock = self.lock
        self.lock = None
        if lock is not None:
            lock.release()


def _read_title_overlays(session_file: Path) -> dict[str, str]:
    """Capture title overlays before a snapshot overwrites ``session.json``.

    Only ``custom_title`` is session-scoped: the user's pin (or explicit
    clear — an empty string must survive too, or a pre-clear snapshot would
    resurrect the pin) wins over whatever the snapshot carried.
    ``generated_title`` is deliberately NOT preserved: it summarizes the
    conversation, and after a rollback the snapshot's own value is the one
    that describes the restored turns — the newer title describes history
    that no longer exists.  The key is captured verbatim; only an
    unreadable envelope (or one whose meta never had it) yields nothing.
    """
    try:
        envelope = json.loads(session_file.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return {}
    meta = envelope.get("meta") if isinstance(envelope, dict) else None
    if not isinstance(meta, dict):
        return {}
    overlays: dict[str, str] = {}
    value = meta.get("custom_title")
    if isinstance(value, str):
        overlays["custom_title"] = value
    return overlays


def _reapply_title_overlays(session_file: Path, overlays: dict[str, str]) -> None:
    """Merge preserved title overlays back into the restored snapshot, best-effort."""
    if not overlays:
        return
    from chrys.foundation.platform.files import atomic_write_text

    try:
        envelope = json.loads(session_file.read_text(encoding="utf-8"))
        meta = envelope.get("meta") if isinstance(envelope, dict) else None
        if not isinstance(meta, dict):
            return
        meta.update(overlays)
        atomic_write_text(session_file, json.dumps(envelope, indent=2, ensure_ascii=False))
    except OSError, ValueError:
        logger.warning("Failed to preserve session titles across rollback for %s", session_file, exc_info=True)


class RollbackHost(Protocol):
    """Engine state needed by rollback orchestration."""

    _session_id: str | None
    _turn_number: int
    _history: SessionHistoryManager
    _mutation_tracker: MutationTracker | None
    _mutation_coordinator: MutationCoordinator | None
    _workspace_change_tracker: WorkspaceChangeTracker
    _fsm: EngineStateMachine
    _bus: EventBus
    _suppress_save: bool
    _trajectory_recorder: TrajectoryRecorder

    @property
    def _settings(self) -> Settings:
        """Read-only: settings change through the handle, not the holder."""
        ...

    @property
    def _session_dir(self) -> Path | None: ...

    @property
    def session_generation(self) -> int: ...

    @property
    def conversation_revision(self) -> int: ...

    @property
    def build_generation(self) -> int: ...

    @property
    def turn_lifecycle_task(self) -> asyncio.Task[None] | None: ...

    def _session_write_lock_path(self, session_id: str) -> Path | None: ...

    def _session_dir_for(self, session_id: str) -> Path: ...

    def _workspace_cwd(self) -> str: ...

    async def _wait_for_agent_load_idle(self) -> None: ...

    async def _prepare_session_transition_if_current(
        self,
        operation: str,
        *,
        session_id: str | None,
        session_generation: int,
    ) -> str | None: ...

    def _commit_session_transition(self, owner: str) -> None: ...

    def _finish_session_transition(self, owner: str) -> None: ...

    async def _reset_session_to_welcome(
        self,
        session_id: str,
        *,
        write_lock_held: bool = False,
        after_delete: Callable[[], Awaitable[None]] | None = None,
        before_restart: Callable[[], None] | None = None,
    ) -> bool: ...

    async def _flush_recovery_checkpoint(self) -> None: ...

    async def _on_session_restore(self, event: SessionRestore) -> None: ...

    async def _save_current_session(self, *, raise_on_error: bool = False) -> bool: ...


def capture_snapshot_writer(engine: RollbackHost) -> Callable[[], None]:
    """Freeze snapshot metadata while deferring path resolution and file I/O."""
    session_id = engine._session_id
    turn_number = engine._turn_number
    keep = engine._settings.rollback_snapshots_keep
    session_dir_for = engine._session_dir_for
    lock_path_for = engine._session_write_lock_path

    def write() -> None:
        session_dir = session_dir_for(session_id) if session_id else None
        lock_path = lock_path_for(session_id) if session_id else None
        write_rollback_snapshot(
            session_dir=session_dir,
            session_id=session_id,
            turn_number=turn_number,
            keep=keep,
            lock_path=lock_path,
        )

    return write


def write_snapshot(engine: RollbackHost) -> None:
    """Copy the current ``session.json`` to ``snapshots/turn_N.json``."""
    capture_snapshot_writer(engine)()


def turn_prompt_previews(engine: RollbackHost) -> dict[int, str]:
    """Return ``{turn_number: first_user_prompt}`` for every known turn."""
    if not engine._history.is_bound:
        return {}
    blocks = engine._history.state.get("compressed_msgs", []) or []
    return collect_turn_prompt_previews(list(engine._history.messages), blocks)


def first_rolled_back_user_text(engine: RollbackHost, target_turn: int) -> str:
    """Return the first user prompt that rollback to ``target_turn`` discards."""
    previews = turn_prompt_previews(engine)
    discarded_turns = [turn for turn in previews if turn > target_turn]
    return previews[min(discarded_turns)] if discarded_turns else ""


def available_rollback_turns(engine: RollbackHost) -> list[int]:
    """Return turn-counts that can be targeted by a rollback."""
    session_dir = engine._session_dir
    if session_dir is None:
        return []
    turns = snapshot_target_turns(session_dir)
    current_turn = engine._turn_number
    if current_turn > 0:
        turns = {turn for turn in turns if turn < current_turn}
    if (engine._mutation_tracker is not None and engine._mutation_tracker.get_all_turns()) or (
        engine._history.is_bound and engine._history.messages
    ):
        turns.add(0)
    if 0 not in turns:
        # No welcome anchor means every remaining snapshot target would be
        # orphaned in the picker.
        return []
    return sorted(turns)


async def on_user_rollback(
    engine: RollbackHost,
    event: UserRollback,
    *,
    atomic_copy_file: Callable[[Path, Path], None],
    lock_timeout_seconds: float,
) -> None:
    """Handle rollback under an exclusive prompt/session-transition boundary."""
    sid = engine._session_id
    session_generation = engine.session_generation
    if sid is None:
        await engine._bus.publish(
            Warning(
                code="rollback_no_session",
                message="No active session to roll back.",
                display_message=_ROLLBACK_NO_SESSION.bind(),
            ),
        )
        return
    if await _cancel_if_rollback_owner_changed(engine, event, sid, session_generation):
        return
    if await _refuse_if_rollback_state_invalid(engine, sid):
        return

    transition_owner = await engine._prepare_session_transition_if_current(
        "rollback",
        session_id=sid,
        session_generation=session_generation,
    )
    if transition_owner is None:
        await _cancel_if_rollback_owner_changed(engine, event, sid, session_generation)
        return
    try:
        prepared = await _prepare_user_rollback_with_permit(
            engine,
            event,
            sid=sid,
            session_generation=session_generation,
        )
        if prepared is None:
            return
        target_turn, rolled_back_user_text = prepared
        await _execute_user_rollback_with_permit(
            engine,
            event,
            sid=sid,
            session_generation=session_generation,
            transition_owner=transition_owner,
            target_turn=target_turn,
            rolled_back_user_text=rolled_back_user_text,
            atomic_copy_file=atomic_copy_file,
            lock_timeout_seconds=lock_timeout_seconds,
        )
    finally:
        engine._finish_session_transition(transition_owner)


async def _prepare_user_rollback_with_permit(
    engine: RollbackHost,
    event: UserRollback,
    *,
    sid: str,
    session_generation: int,
) -> tuple[int, str] | None:
    """Quiesce and validate rollback without invalidating the current generation."""
    lifecycle_task = engine.turn_lifecycle_task
    await engine._wait_for_agent_load_idle()
    target_turn = event.target_turn
    if await _cancel_if_rollback_owner_changed(engine, event, sid, session_generation):
        return None
    if await _refuse_if_rollback_state_invalid(engine, sid):
        return None

    # IDLE is reached before final persistence and after-turn hooks finish.
    # New admission is fenced and older reserved admissions have drained, so
    # this task cannot be replaced while rollback waits for durable quiescence.
    await await_task_quiescence(lifecycle_task)
    if await _cancel_if_rollback_owner_changed(engine, event, sid, session_generation):
        return None
    if await _refuse_if_rollback_state_invalid(engine, sid):
        return None

    if (
        event.expected_conversation_revision is not None
        and engine.conversation_revision != event.expected_conversation_revision
    ):
        await engine._bus.publish(
            Warning(
                code="rollback_conversation_changed",
                message="Rollback cancelled because the conversation changed after the picker was loaded.",
                display_message=_ROLLBACK_CONVERSATION_CHANGED.bind(),
                session_id=sid,
            ),
        )
        return None

    if event.expected_current_turn is not None and engine._turn_number != event.expected_current_turn:
        await engine._bus.publish(
            Warning(
                code="rollback_conversation_changed",
                message=(
                    "Rollback cancelled because the conversation advanced "
                    f"from turn {event.expected_current_turn} to turn {engine._turn_number}."
                ),
                display_message=_ROLLBACK_CONVERSATION_ADVANCED.bind(
                    expected_turn=event.expected_current_turn,
                    current_turn=engine._turn_number,
                ),
                session_id=sid,
            ),
        )
        return None

    if (event.expected_build_generation is not None and engine.build_generation != event.expected_build_generation) or (
        event.expected_workspace_cwd is not None and engine._workspace_cwd() != event.expected_workspace_cwd
    ):
        await engine._bus.publish(
            Warning(
                code="rollback_runtime_changed",
                message="Rollback cancelled because the workspace or runtime changed after the picker was loaded.",
                display_message=_ROLLBACK_RUNTIME_CHANGED.bind(),
                session_id=sid,
            ),
        )
        return None

    if event.relative_turns is not None:
        if event.relative_turns <= 0:
            await engine._bus.publish(
                Warning(
                    code="rollback_invalid_turn",
                    message="relative_turns must be positive.",
                    display_message=_ROLLBACK_RELATIVE_TURNS_INVALID.bind(),
                    session_id=sid,
                ),
            )
            return None
        target_turn = engine._turn_number - event.relative_turns
        if target_turn < 0:
            await engine._bus.publish(
                Warning(
                    code="rollback_unavailable",
                    message=(
                        f"Cannot roll back {event.relative_turns} turns; "
                        f"the session currently has {engine._turn_number}."
                    ),
                    display_message=_ROLLBACK_TURNS_UNAVAILABLE.bind(
                        requested_turns=event.relative_turns,
                        current_turns=engine._turn_number,
                    ),
                    session_id=sid,
                ),
            )
            return None

    if target_turn < 0:
        await engine._bus.publish(
            Warning(
                code="rollback_invalid_turn",
                message="target_turn must be >= 0.",
                display_message=_ROLLBACK_TARGET_TURN_INVALID.bind(),
                session_id=sid,
            ),
        )
        return None

    available = available_rollback_turns(engine)
    if target_turn not in available:
        await engine._bus.publish(
            Warning(
                code="rollback_unavailable",
                message=f"Cannot roll back to turn {target_turn}; available turns: {available}",
                display_message=_ROLLBACK_TURN_UNAVAILABLE.bind(
                    target_turn=target_turn,
                    available=str(available),
                ),
                session_id=sid,
            ),
        )
        return None

    return target_turn, first_rolled_back_user_text(engine, target_turn)


async def _execute_user_rollback_with_permit(
    engine: RollbackHost,
    event: UserRollback,
    *,
    sid: str,
    session_generation: int,
    transition_owner: str,
    target_turn: int,
    rolled_back_user_text: str,
    atomic_copy_file: Callable[[Path, Path], None],
    lock_timeout_seconds: float,
) -> None:
    """Secure rollback resources, then commit immediately before mutation."""

    welcome_reset_lock: FileLock | None = None
    if target_turn == 0:
        await engine._flush_recovery_checkpoint()
        reset_lock_acquired, welcome_reset_lock = _acquire_welcome_reset_write_lock(engine, sid, lock_timeout_seconds)
        if not reset_lock_acquired:
            await engine._bus.publish(
                Error(
                    code="rollback_reset_failed",
                    message="Rollback to welcome could not reset the session because the session state is busy.",
                    display_message=_ROLLBACK_RESET_FAILED.bind(),
                    session_id=sid,
                ),
            )
            return

    welcome_lock_lease = _RollbackLockLease(welcome_reset_lock)
    try:
        await _execute_user_rollback_after_resource_preflight(
            engine,
            event,
            sid=sid,
            session_generation=session_generation,
            transition_owner=transition_owner,
            target_turn=target_turn,
            rolled_back_user_text=rolled_back_user_text,
            atomic_copy_file=atomic_copy_file,
            lock_timeout_seconds=lock_timeout_seconds,
            welcome_lock_lease=welcome_lock_lease,
        )
    finally:
        welcome_lock_lease.release()


async def _execute_user_rollback_after_resource_preflight(
    engine: RollbackHost,
    event: UserRollback,
    *,
    sid: str,
    session_generation: int,
    transition_owner: str,
    target_turn: int,
    rolled_back_user_text: str,
    atomic_copy_file: Callable[[Path, Path], None],
    lock_timeout_seconds: float,
    welcome_lock_lease: _RollbackLockLease,
) -> None:
    """Build the rollback plan and mutate only after secured preflight resources."""

    tracker = engine._mutation_tracker
    tracker_turn_ids: list[int] = sorted(t.turn_id for t in tracker.get_all_turns()) if tracker is not None else []
    retained_notice = (
        format_retained_changes_notice(
            tracker,
            cwd=engine._workspace_cwd(),
            max_entries=engine._settings.workspace_change_notice_max_entries,
        )
        if target_turn == 0 and not event.revert_changes
        else None
    )

    restore_results: list = []
    plan_exclusions: list[tuple[str, str]] = []
    plan_warnings: list[str] = []
    plan_candidates: list[str] = []
    revert_detection_truncated = False
    welcome_file_rollback: Callable[[], Awaitable[None]] | None = None
    post_swap_file_rollback: Callable[[], None] | None = None
    if event.revert_changes and tracker is not None and tracker_turn_ids:
        rollback_turn_ids = {tid for tid in tracker_turn_ids if tid > target_turn}
        only_paths: set[str] | None = set(event.selected_paths) if event.selected_paths else None
        if rollback_turn_ids:
            # Truncated detection means the rolled-back turns may hold
            # writes that never became plan candidates; the flag lives on
            # the turns rollback_turns removes, so capture it here.
            revert_detection_truncated = any(
                turn.detection_truncated for turn in tracker.get_all_turns() if turn.turn_id in rollback_turn_ids
            )
            try:
                # Authoritative peer re-check before the only destructive
                # consumer acts: force-reclassify against the coordination
                # registry (a peer's claim may have landed after our
                # finalize), then let the coordinator exclude paths a peer
                # modified since and warn about in-flight peer commands.
                # Coordination failures must not break rollback itself.
                coordinator = engine._mutation_coordinator
                loop = asyncio.get_running_loop()
                # The workspace cwd doubles as the registry fallback root:
                # outside a git repo, claims were published under it, and
                # the destructive rollback-time check must rediscover the
                # same peer files.
                workspace_cwd = engine._workspace_cwd()
                if coordinator is not None:
                    try:
                        await loop.run_in_executor(
                            None,
                            lambda: coordinator.reclassify(tracker, force=True, fallback_root=workspace_cwd),
                        )
                    except Exception:
                        logger.debug("Pre-rollback attribution re-check failed", exc_info=True)
                # Build the plan FIRST: rollback_turns removes the turns
                # it rolls back, so exclusions/warnings cannot be
                # reconstructed afterwards — execute from the pre-built
                # plan and thread its report into the result event.
                plan = tracker.get_rollback_plan_for_turns(rollback_turn_ids)
                if coordinator is not None:
                    try:
                        plan = await loop.run_in_executor(
                            None,
                            lambda: coordinator.augment_rollback_plan(
                                tracker,
                                plan,
                                scope_paths=[workspace_cwd],
                                fallback_root=workspace_cwd,
                            ),
                        )
                    except Exception:
                        logger.debug("Rollback plan peer augmentation failed", exc_info=True)
                plan_exclusions = [(path, reason.value) for path, reason in plan.exclusions]
                plan_warnings = [w.message for w in plan.warnings]
                # Every path the rolled-back turns touched — entries the
                # restore will attempt plus exclusions it never will. The
                # turns are removed during execution, so the retained set
                # can only be derived from this pre-built list afterwards.
                plan_candidates = [path for path, _ in plan.entries] + [path for path, _ in plan.exclusions]

                def _apply_file_rollback() -> None:
                    nonlocal restore_results
                    try:
                        restore_results = tracker.rollback_turns(
                            rollback_turn_ids,
                            only_paths=only_paths,
                            plan=plan,
                        )
                    except Exception:
                        logger.exception("Rollback file-restore failed (target_turn=%d)", target_turn)
                        restore_results = []

                if target_turn == 0:

                    async def _apply_welcome_file_rollback() -> None:
                        _apply_file_rollback()

                    welcome_file_rollback = _apply_welcome_file_rollback
                else:
                    post_swap_file_rollback = _apply_file_rollback
            except Exception:
                logger.exception("Rollback file-restore failed (target_turn=%d)", target_turn)
                restore_results = []

    if await _cancel_if_rollback_owner_changed(engine, event, sid, session_generation):
        return

    session_dir = engine._session_dir
    if target_turn == 0:
        # Keep the old registry for the audit record: a successful reset
        # replaces the live history before the rollback event is committed.
        rollback_history_state = engine._history.state if engine._history.is_bound else None
        engine._commit_session_transition(transition_owner)

        try:
            reset_succeeded = await engine._reset_session_to_welcome(
                sid,
                write_lock_held=welcome_lock_lease.held,
                after_delete=welcome_file_rollback,
                before_restart=welcome_lock_lease.release,
            )
        finally:
            welcome_lock_lease.release()
        if not reset_succeeded:
            await engine._bus.publish(
                Error(
                    code="rollback_reset_failed",
                    message="Rollback to welcome could not reset the session because the session state is busy.",
                    display_message=_ROLLBACK_RESET_FAILED.bind(),
                    session_id=sid,
                ),
            )
            return
        # The reset has committed and released the welcome write lock. Its
        # restarted recorder is bound lazily to the same log, so activation
        # recovers the old branch and this event atomically opens the new one.
        # A failed reset never reaches here and therefore never supersedes the
        # branch of the session that was restored.
        await engine._trajectory_recorder.rollback(
            target_turn=0,
            history_state=rollback_history_state,
        )
        if retained_notice:
            engine._workspace_change_tracker.queue_safety_notice(retained_notice, cwd=engine._workspace_cwd())
        _queue_partial_revert_notice(
            engine, plan_candidates, restore_results, detection_truncated=revert_detection_truncated
        )
        files_reverted = sum(1 for r in restore_results if r.changed)
        await engine._bus.publish(
            RollbackResult(
                session_id=sid,
                target_turn=0,
                rolled_back_user_text=rolled_back_user_text,
                files_reverted=files_reverted,
                restore_results=restore_results,
                exclusions=plan_exclusions,
                warnings=plan_warnings,
            ),
        )
        return

    snapshot_path = rollback_snapshot_for_target(session_dir, target_turn)
    session_file = session_dir / "session.json" if session_dir else None
    if snapshot_path is None or session_file is None or not snapshot_path.exists():
        await engine._bus.publish(
            Warning(
                code="rollback_snapshot_missing",
                message=f"Snapshot for turn {target_turn} is missing.",
                display_message=_ROLLBACK_SNAPSHOT_MISSING.bind(target_turn=target_turn),
                session_id=sid,
            ),
        )
        return

    snapshot_missing_after_lock = False
    try:
        lock_path = engine._session_write_lock_path(sid)
        if lock_path is None:
            raise OSError("session state store is unavailable")
        with FileLock(lock_path, timeout=lock_timeout_seconds):
            if not snapshot_path.exists():
                snapshot_missing_after_lock = True
            else:
                title_overlays = _read_title_overlays(session_file)
                engine._commit_session_transition(transition_owner)
                atomic_copy_file(snapshot_path, session_file)
                _reapply_title_overlays(session_file, title_overlays)
                backup_file = session_file.with_name(SESSION_BACKUP_FILE_NAME)
                try:
                    # Copy the restored (title-patched) primary so both files
                    # stay identical.
                    atomic_copy_file(session_file, backup_file)
                except OSError:
                    with contextlib.suppress(OSError):
                        backup_file.unlink()
                    logger.warning("Failed to update rollback backup %s", backup_file, exc_info=True)
    except TimeoutError as exc:
        await engine._bus.publish(
            Error(
                code="rollback_swap_locked",
                message=f"Timed out waiting for session lock: {exc}",
                display_message=_ROLLBACK_SWAP_LOCKED.bind(detail=DisplayBlock(str(exc))),
                session_id=sid,
            ),
        )
        return
    except OSError as exc:
        await engine._bus.publish(
            Error(
                code="rollback_swap_failed",
                message=f"Failed to restore snapshot: {exc}",
                display_message=_ROLLBACK_SWAP_FAILED.bind(detail=DisplayBlock(str(exc))),
                session_id=sid,
            ),
        )
        return

    if snapshot_missing_after_lock:
        await engine._bus.publish(
            Warning(
                code="rollback_snapshot_missing",
                message=f"Snapshot for turn {target_turn} is missing.",
                display_message=_ROLLBACK_SNAPSHOT_MISSING.bind(target_turn=target_turn),
                session_id=sid,
            ),
        )
        return

    if post_swap_file_rollback is not None:
        post_swap_file_rollback()

    if session_dir is not None:
        snap_dir = session_dir / "snapshots"
        if snap_dir.is_dir():
            for path in rollback_snapshot_paths(session_dir):
                turn = snapshot_target_turn(path)
                if turn is None:
                    continue
                if turn <= target_turn:
                    continue
                try:
                    path.unlink()
                except OSError:
                    logger.debug("Failed to prune stale snapshot %s", path, exc_info=True)
        try:
            prune_workflow_artifacts_after_turn(session_dir, target_turn)
        except OSError:
            logger.debug("Failed to prune stale requirement-clarification artifacts", exc_info=True)

    # The rollback opens a new trajectory branch: recorded on the outgoing
    # runtime (the restore below closes it and the restored session resumes
    # on the new branch), after the swap lock is released — activation
    # takes the same lock — and while the live registry still names the
    # first superseded turn.
    record_cancelled = False
    try:
        await engine._trajectory_recorder.rollback(
            target_turn=target_turn,
            history_state=engine._history.state if engine._history.is_bound else None,
        )
    except asyncio.CancelledError:
        # The swap has committed but the engine still holds the pre-rollback
        # history: leaving here would let the next save write that history
        # back over the file the rollback just restored. The audit record is
        # the only part that can be dropped, so the cancellation waits for the
        # restore below and completes right after it.
        task = asyncio.current_task()
        if task is not None:
            task.uncancel()
        record_cancelled = True

    engine._suppress_save = True
    try:
        await engine._on_session_restore(SessionRestore(session_id=sid, ignore_recovery=True))
    finally:
        engine._suppress_save = False
    if record_cancelled:
        raise asyncio.CancelledError

    if event.revert_changes:
        engine._workspace_change_tracker.invalidate()
        # After the invalidate: the stale baseline can never rediscover
        # files the revert retained, so they must be reported directly.
        _queue_partial_revert_notice(
            engine, plan_candidates, restore_results, detection_truncated=revert_detection_truncated
        )
        await engine._save_current_session()

    files_reverted = sum(1 for r in restore_results if r.changed)
    await engine._bus.publish(
        RollbackResult(
            session_id=sid,
            target_turn=target_turn,
            rolled_back_user_text=rolled_back_user_text,
            files_reverted=files_reverted,
            restore_results=restore_results,
            exclusions=plan_exclusions,
            warnings=plan_warnings,
        ),
    )


def _queue_partial_revert_notice(
    engine: RollbackHost,
    plan_candidates: list[str],
    restore_results: list,
    *,
    detection_truncated: bool = False,
) -> None:
    """Report rollback candidates the revert did not actually restore.

    Covers a partial ``selected_paths`` selection (unselected entries are
    never attempted), plan exclusions, per-file restore failures, and a
    restore pass that failed wholesale (empty results). Truncated
    detection on a rolled-back turn means writes may exist that never
    became candidates, so it warns even when every known candidate
    restored. A fully-applied revert with complete detection queues
    nothing.
    """
    if not plan_candidates and not detection_truncated:
        return
    restored = {result.path for result in restore_results if result.ok}
    retained = [path for path in plan_candidates if path not in restored]
    cwd = engine._workspace_cwd()
    notice = format_partial_revert_notice(
        retained,
        cwd=cwd,
        max_entries=engine._settings.workspace_change_notice_max_entries,
        detection_incomplete=detection_truncated,
    )
    if notice:
        engine._workspace_change_tracker.queue_safety_notice(notice, cwd=cwd)


async def _cancel_if_rollback_owner_changed(
    engine: RollbackHost,
    event: UserRollback,
    session_id: str,
    session_generation: int,
) -> bool:
    """Reject a deferred rollback that no longer belongs to its requester."""
    if (
        engine._session_id == session_id
        and engine.session_generation == session_generation
        and (event.session_id is None or event.session_id == session_id)
    ):
        return False
    await engine._bus.publish(
        Warning(
            code="rollback_session_changed",
            message="Rollback cancelled because the active session changed.",
            display_message=_ROLLBACK_SESSION_CHANGED.bind(),
            session_id=session_id,
        )
    )
    return True


async def _refuse_if_rollback_state_invalid(engine: RollbackHost, session_id: str) -> bool:
    """Preserve the immediate refusal contract for a genuinely active turn."""
    if engine._fsm.state in (EngineState.IDLE, EngineState.INTERRUPTED, EngineState.FAILED):
        return False
    await engine._bus.publish(
        Warning(
            code="rollback_refused",
            message=f"Rollback is not allowed in state {engine._fsm.state.name}.",
            display_message=_ROLLBACK_REFUSED.bind(state=engine._fsm.state.name),
            session_id=session_id,
        ),
    )
    return True


def _acquire_welcome_reset_write_lock(
    engine: RollbackHost,
    session_id: str,
    timeout: float,
) -> tuple[bool, FileLock | None]:
    lock_path = engine._session_write_lock_path(session_id)
    if lock_path is None:
        return True, None
    lock = FileLock(lock_path, timeout=timeout)
    try:
        lock.acquire()
        return True, lock
    except TimeoutError:
        logger.warning("Timed out acquiring session lock for welcome rollback %s", session_id, exc_info=True)
        return False, None
    except OSError:
        logger.warning("Failed to acquire session lock for welcome rollback %s", session_id, exc_info=True)
        return False, None
