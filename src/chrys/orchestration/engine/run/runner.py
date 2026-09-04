# Copyright (c) 2026 Chrys. All rights reserved.

"""Fresh and retry executor passes for main-agent turns."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.turns import current_turn_start, is_continuation_message
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.orchestration.engine import trajectory as trajectory_recorder
from chrys.orchestration.engine.run.finalizer import PostRunOutcome, TurnFinalizer, _expire_current_run_scope
from chrys.orchestration.engine.run.input_refs import format_skill_reference_reminder, parse_skill_reference
from chrys.orchestration.engine.run.prompt_content import PromptContentPreparer
from chrys.orchestration.engine.run.retry import RetryCoordinator
from chrys.orchestration.engine.run.runtime_skills import RuntimeSkillRefresher
from chrys.orchestration.engine.run.turn_hooks import TurnHookDispatcher
from chrys.service.routing.classifier import RouteDecision, RouteTrack
from chrys.service.trajectory.preparation import PreparationOutcome, PreparationScope, PreparationTrace

if TYPE_CHECKING:
    from pathlib import Path

    from chrys.foundation.config.settings import Settings
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import AgentRuntimeDetails, RuntimeSkillDetails
    from chrys.foundation.models.workspace import Workspace
    from chrys.kernel import LoopRecorder
    from chrys.orchestration.engine.executor import Executor
    from chrys.orchestration.engine.run.turn_state import (
        CurrentRunInjectionWindow,
        CurrentRunScope,
        TurnRuntimeState,
    )
    from chrys.orchestration.engine.state.machine import EngineStateMachine
    from chrys.orchestration.engine.trajectory import TrajectoryRecorder
    from chrys.orchestration.sub_agents.tools import SubAgentTools
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
    from chrys.service.hooks.manager import HookManager
    from chrys.service.mutations.tracker import MutationTracker
    from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker
    from chrys.service.profiles.agents.schema import AgentProfile
    from chrys.service.profiles.models.schema import ModelProfile
    from chrys.service.session.history import SessionHistoryManager
    from chrys.service.session.persistence import SessionPersistence
    from chrys.service.session.runtime_metadata import SessionRuntimeMetadata
    from chrys.service.skills.provider import ChrysSkillsProvider


logger = logging.getLogger(__name__)

PromptContentPreparerFactory = Callable[["TurnRunnerHost"], PromptContentPreparer]


class TurnRunnerHost(Protocol):
    """Engine host contract needed for fresh and retry executor passes."""

    _bus: EventBus
    _session_id: str | None
    _executor: Executor
    _history: SessionHistoryManager
    _reminder_middleware: SystemReminderMiddleware | None
    _runtime_meta: SessionRuntimeMetadata
    _turn_state: TurnRuntimeState
    _runtime_details: AgentRuntimeDetails
    _skills_provider: ChrysSkillsProvider | None
    _tool_names: list[str]
    _skill_names: list[str]
    _memory_files: list[str]
    _workspace: Workspace | None
    _hook_manager: HookManager | None
    _agent_profile: AgentProfile | None
    _fsm: EngineStateMachine
    _consumed_injections: list[Any]
    _intermediate_texts: dict[int, str]
    _turn_number: int
    _loop_recorder: LoopRecorder | None
    _mutation_tracker: MutationTracker | None
    _workspace_change_tracker: WorkspaceChangeTracker
    _agent_profile_fingerprint: str
    _model_profile_fingerprint: str
    _active_profile: ModelProfile | None
    _trajectory_recorder: TrajectoryRecorder
    _persistence: SessionPersistence
    _recovered_from_sidecar: bool
    _injection: Any
    _shutting_down: bool
    _paused_sub_agents: set[str]
    _sub_agent_tools: SubAgentTools | None
    _on_successful_turn: Callable[[], None]
    _on_turn_started: Callable[[], None]
    _requirement_clarification_workflow: Any | None
    _requirement_enrichment_workflow: Any | None

    async def persist_recovery_now(self) -> bool: ...

    _last_route: RouteDecision | None
    _long_horizon_campaign: dict[str, Any] | None

    def _accumulate_side_call_usage(self, usage_details: Mapping[str, Any]) -> None: ...

    @property
    def _settings(self) -> Settings:
        """Read-only: settings change through the handle, not the holder."""
        ...

    @property
    def _session_dir(self) -> Path | None: ...

    @property
    def session_generation(self) -> int: ...

    @property
    def build_generation(self) -> int: ...

    def _capture_rollback_snapshot_writer(self) -> Callable[[], None]: ...

    async def _save_current_session(self) -> bool: ...

    async def _wait_for_agent_load_idle(self) -> None: ...

    async def _retry_and_save(
        self,
        additional_text: str = "",
        created_at: datetime | str | None = None,
        *,
        run_scope: CurrentRunScope | None = None,
        injection_window: CurrentRunInjectionWindow | None = None,
        admission_preparation: PreparationTrace | None = None,
    ) -> None: ...


class TurnRunner:
    """Execute fresh and retry passes while preserving turn-layer ordering."""

    def __init__(
        self,
        host: TurnRunnerHost,
        *,
        prompt_content_preparer_factory: PromptContentPreparerFactory | None = None,
    ) -> None:
        self._host = host
        self._prompt_content_preparer_factory = prompt_content_preparer_factory or PromptContentPreparer
        # Item id pre-minted for the message that opens this pass, so the
        # ``turn.started`` event can name it before the message exists.
        self._opening_item_id: str | None = None

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
        """Execute a fresh agent turn, optionally through clarification."""
        host = self._host
        profile = host._agent_profile
        decision = host._last_route
        long_horizon = decision is not None and decision.track is RouteTrack.LONG_HORIZON and profile is not None
        # A long-horizon turn runs the full clarification workflow whatever the
        # profile's own switch says: the router, not the profile, decided this
        # turn earns it.
        clarification_enabled = long_horizon or bool(profile is not None and profile.requirement_clarification.enabled)
        enrichment_enabled = not long_horizon and profile is not None and profile.requirement_enrichment.enabled
        if not clarification_enabled and not enrichment_enabled:
            await self._run_fresh_standard(
                text,
                created_at=created_at,
                contents=contents,
                run_scope=run_scope,
                injection_window=injection_window,
                admission_preparation=admission_preparation,
            )
            return
        if contents is None:
            contents = await self._prepare_user_contents(text)
            if contents is None:
                if admission_preparation is not None:
                    await admission_preparation.finished(outcome=PreparationOutcome.PREPARATION_FAILED)
                return
        if enrichment_enabled and profile is not None:
            from chrys.orchestration.engine.run.requirement_enrichment import RequirementEnrichmentWorkflow
            from chrys.service.semantic_search import SemanticSearchMode

            config = profile.requirement_enrichment
            await RequirementEnrichmentWorkflow(
                self._host,
                self,
                strategy=config.clarification_strategy,
                clarification_timeout_seconds=config.clarification_timeout_seconds,
                localization_mode=SemanticSearchMode(config.localization_mode),
                localization_timeout_seconds=config.localization_timeout_seconds,
                localization_model_profile=config.localization_model_profile,
            ).run(
                text,
                created_at=created_at,
                contents=contents,
                run_scope=run_scope,
                injection_window=injection_window,
                admission_preparation=admission_preparation,
            )
            return
        from chrys.orchestration.engine.run.requirement_clarification import RequirementClarificationWorkflow

        extensions = None
        if long_horizon and decision is not None:
            from chrys.orchestration.engine.run.long_horizon import LongHorizonExtensions

            extensions = LongHorizonExtensions(host, decision, runner=self)
        await RequirementClarificationWorkflow(
            self._host,
            self,
            strategy=profile.requirement_clarification.strategy,
            # A routed turn always starts from a fresh baseline and always
            # continues past clarification, whatever the profile configured for
            # the standard track.
            reuse_workspace_as_p0=(False if long_horizon else profile.requirement_clarification.reuse_workspace_as_p0),
            clarification_only=(False if long_horizon else profile.requirement_clarification.clarification_only),
            clarification_timeout_seconds=profile.requirement_clarification.clarification_timeout_seconds,
            initial_timeout_seconds=profile.requirement_clarification.initial_timeout_seconds,
            repair_timeout_seconds=profile.requirement_clarification.repair_timeout_seconds,
            extensions=extensions,
        ).run(
            text,
            created_at=created_at,
            contents=contents,
            run_scope=run_scope,
            injection_window=injection_window,
            admission_preparation=admission_preparation,
        )

    async def _run_fresh_standard(
        self,
        text: str,
        created_at: datetime | str | None = None,
        contents: list[Any] | None = None,
        *,
        run_scope: CurrentRunScope | None = None,
        injection_window: CurrentRunInjectionWindow | None = None,
        admission_preparation: PreparationTrace | None = None,
        finalize: bool = True,
        before_execution: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Execute the ordinary fresh pass, with optional deferred finalization.

        *before_execution* runs after the pass is otherwise prepared and just
        before the reminder middleware assembles the turn: the enrichment
        preflight uses it to queue advisory guidance only once the snapshot it
        was derived from is confirmed to still match the live workspace.
        """
        _ = injection_window
        host = self._host
        if contents is None:
            contents = await self._prepare_user_contents(text)
            if contents is None:
                if admission_preparation is not None:
                    await admission_preparation.finished(outcome=PreparationOutcome.PREPARATION_FAILED)
                return

        try:
            await self.pre_run(
                reset_batch_id=True,
                preparation_scope_operation_id=self._preparation_operation_id(admission_preparation),
            )
            if admission_preparation is not None:
                await admission_preparation.finished(outcome=PreparationOutcome.FRESH_TURN)
        except asyncio.CancelledError:
            if admission_preparation is not None:
                admission_preparation.finished_soon(outcome=PreparationOutcome.CANCELLED)
            raise
        except BaseException:
            if admission_preparation is not None:
                admission_preparation.finished_soon(outcome=PreparationOutcome.PREPARATION_FAILED)
            raise
        preamble = self._open_turn_preamble()
        try:
            if preamble is not None:
                await preamble.started()
                self._bind_turn_preamble(preamble)
            await self._fire_before_turn(
                text,
                target_operation_id=preamble.operation_id
                if preamble is not None and preamble.start_committed
                else None,
            )
            await self._compute_workspace_notice(is_retry=False)
            write_rollback_snapshot = host._capture_rollback_snapshot_writer()
            await asyncio.to_thread(write_rollback_snapshot)
            await self._refresh_runtime_skills(update_active_turn=False)
            self._queue_skill_reference_reminder(text, for_next_turn=True)
            if before_execution is not None:
                await before_execution()
            if host._reminder_middleware is not None:
                if run_scope is not None:
                    host._reminder_middleware.prepare_turn(
                        reminder_scope=run_scope.reminder_scope,
                        usage=host._runtime_meta.last_usage_details or None,
                    )
                else:
                    host._reminder_middleware.prepare_turn(usage=host._runtime_meta.last_usage_details or None)
            host._executor.set_user_messages([text] if text else [])
            host._turn_state.set_current_input(text, contents, created_at)
            if preamble is not None:
                await preamble.finished(outcome=PreparationOutcome.HANDOFF)
        except asyncio.CancelledError:
            if preamble is not None:
                preamble.finished_soon(outcome=PreparationOutcome.INTERRUPTED)
            raise
        except BaseException:
            if preamble is not None:
                preamble.finished_soon(outcome=PreparationOutcome.FAILED)
            raise
        try:
            # Stop may arrive while pre-executor work (notably the rollback
            # snapshot) is awaiting worker I/O.  Consume only a cancellation
            # bound to this exact run task so a late Stop from an older task
            # cannot suppress the next turn.
            if host._turn_state.consume_pre_executor_interrupt():
                host._executor.record_pre_run_interrupt()
            else:
                await host._executor.run(contents, created_at=created_at)
        finally:
            if run_scope is not None and finalize:
                host._turn_state.close_injection_admission(run_scope)
            # The tracker was drained at prepare_turn; any pass ending before
            # a model request carried the notice (Stop, load/hook failure,
            # cancellation) must hand it back for the next turn.
            self._requeue_undelivered_file_change()
        if host._executor.run_failed or host._executor.was_interrupted:
            host._history.ensure_user_message(
                text, created_at=created_at, contents=contents, item_id=self._opening_item_id
            )
        self._tag_consumed_profile_switch()
        if finalize:
            await self.finalize_current_run()

    async def _prepare_fresh_without_execution(
        self,
        text: str,
        *,
        created_at: datetime | str | None,
        contents: list[Any],
        run_scope: CurrentRunScope | None,
        admission_preparation: PreparationTrace | None,
    ) -> None:
        """Open a fresh turn while leaving an imported workspace P0 untouched."""
        host = self._host
        try:
            await self.pre_run(
                reset_batch_id=True,
                preparation_scope_operation_id=self._preparation_operation_id(admission_preparation),
            )
            if admission_preparation is not None:
                await admission_preparation.finished(outcome=PreparationOutcome.FRESH_TURN)
        except asyncio.CancelledError:
            if admission_preparation is not None:
                admission_preparation.finished_soon(outcome=PreparationOutcome.CANCELLED)
            raise
        except BaseException:
            if admission_preparation is not None:
                admission_preparation.finished_soon(outcome=PreparationOutcome.PREPARATION_FAILED)
            raise
        preamble = self._open_turn_preamble()
        try:
            if preamble is not None:
                await preamble.started()
                self._bind_turn_preamble(preamble)
            await self._fire_before_turn(
                text,
                target_operation_id=(
                    preamble.operation_id if preamble is not None and preamble.start_committed else None
                ),
            )
            await self._compute_workspace_notice(is_retry=False)
            write_rollback_snapshot = host._capture_rollback_snapshot_writer()
            await asyncio.to_thread(write_rollback_snapshot)
            await self._refresh_runtime_skills(update_active_turn=False)
            self._queue_skill_reference_reminder(text, for_next_turn=True)
            if host._reminder_middleware is not None:
                if run_scope is not None:
                    host._reminder_middleware.prepare_turn(
                        reminder_scope=run_scope.reminder_scope,
                        usage=host._runtime_meta.last_usage_details or None,
                    )
                else:
                    host._reminder_middleware.prepare_turn(usage=host._runtime_meta.last_usage_details or None)
            host._executor.set_user_messages([text] if text else [])
            host._turn_state.set_current_input(text, contents, created_at)
            if preamble is not None:
                await preamble.finished(outcome=PreparationOutcome.HANDOFF)
        except asyncio.CancelledError:
            if preamble is not None:
                preamble.finished_soon(outcome=PreparationOutcome.INTERRUPTED)
            raise
        except BaseException:
            if preamble is not None:
                preamble.finished_soon(outcome=PreparationOutcome.FAILED)
            raise

    async def run_retry(
        self,
        additional_text: str = "",
        created_at: datetime | str | None = None,
        *,
        run_scope: CurrentRunScope | None = None,
        injection_window: CurrentRunInjectionWindow | None = None,
        admission_preparation: PreparationTrace | None = None,
    ) -> None:
        """Resume the agent from current state and finalize it."""
        _ = injection_window
        host = self._host
        try:
            await self.pre_run(
                reset_batch_id=False,
                is_retry=True,
                has_opening_input=bool(additional_text),
                preparation_scope_operation_id=self._preparation_operation_id(admission_preparation),
            )
            if admission_preparation is not None:
                await admission_preparation.finished(outcome=PreparationOutcome.RETRY_TURN)
        except asyncio.CancelledError:
            if admission_preparation is not None:
                admission_preparation.finished_soon(outcome=PreparationOutcome.CANCELLED)
            raise
        except BaseException:
            if admission_preparation is not None:
                admission_preparation.finished_soon(outcome=PreparationOutcome.PREPARATION_FAILED)
            raise
        preamble = self._open_turn_preamble()
        try:
            if preamble is not None:
                await preamble.started()
                self._bind_turn_preamble(preamble)
            await self._fire_before_turn(
                additional_text,
                is_retry=True,
                target_operation_id=preamble.operation_id
                if preamble is not None and preamble.start_committed
                else None,
            )
            await self._compute_workspace_notice(is_retry=True)
            if host._reminder_middleware is not None:
                if run_scope is not None:
                    host._reminder_middleware.prepare_turn(
                        reminder_scope=run_scope.reminder_scope,
                        usage=host._runtime_meta.last_usage_details or None,
                        preserve_last_words=True,
                        preserve_turn_reminders=True,
                    )
                else:
                    host._reminder_middleware.prepare_turn(
                        usage=host._runtime_meta.last_usage_details or None,
                        preserve_last_words=True,
                        preserve_turn_reminders=True,
                    )
            approval_context = self._retry_approval_context_messages(additional_text)
            if approval_context:
                host._executor.set_user_messages(approval_context)
            if additional_text:
                # Retry guidance is user-authored MID-TURN input: a recovery
                # checkpoint must re-create it flagged (kind-aware), or a crash
                # during a guided retry restores unflagged guidance that opens a
                # pseudo-turn on reload.
                host._turn_state.set_current_input(additional_text, None, created_at, kind="injected")
            else:
                # Empty-input continuation has no current input. The opener-replay
                # branch re-registers its popped anchor at pop time.
                host._turn_state.clear_current_input()
            if preamble is not None:
                await preamble.finished(outcome=PreparationOutcome.HANDOFF)
        except asyncio.CancelledError:
            if preamble is not None:
                preamble.finished_soon(outcome=PreparationOutcome.INTERRUPTED)
            raise
        except BaseException:
            if preamble is not None:
                preamble.finished_soon(outcome=PreparationOutcome.FAILED)
            raise
        try:
            if host._turn_state.consume_pre_executor_interrupt():
                host._executor.record_pre_run_interrupt()
                if additional_text:
                    host._history.ensure_user_message(
                        additional_text, created_at=created_at, kind="injected", item_id=self._opening_item_id
                    )
            else:
                await host._executor.resume(additional_text=additional_text, created_at=created_at)
        finally:
            if run_scope is not None:
                host._turn_state.close_injection_admission(run_scope)
            self._requeue_undelivered_file_change()
        self._tag_consumed_profile_switch()
        await self.finalize_current_run()

    async def finalize_current_run(self) -> PostRunOutcome:
        """Finalize the just-ended executor pass and close the terminal boundary."""
        try:
            outcome = await TurnFinalizer(self._host).finalize()
            self._complete_finalized_run(outcome)
        except BaseException:
            self._clear_current_input()
            raise
        return outcome

    async def pre_run(
        self,
        *,
        reset_batch_id: bool,
        is_retry: bool = False,
        has_opening_input: bool = True,
        preparation_scope_operation_id: str | None = None,
    ) -> None:
        """Common setup before ``Executor.run()`` or ``Executor.resume()``.

        ``has_opening_input`` says whether this pass sends a user-authored
        message (a fresh prompt or retry guidance) — only then does the pass
        have an opening item to name in its ``turn.started`` event.
        """
        host = self._host
        host._turn_state.advance_conversation_revision()
        self._fire_turn_started()
        host._consumed_injections.clear()
        host._intermediate_texts.clear()
        if not is_retry:
            host._turn_number += 1
        turn_counter = host._executor.history_state.get("turn_counter", 0)
        host._turn_state.history_start_index = len(host._history.messages)
        logger.debug(
            "pre_run: is_retry=%s turn_number=%d turn_counter=%d fsm=%s",
            is_retry,
            host._turn_number,
            turn_counter,
            host._fsm.state.name,
        )
        host._executor.reset_counters(reset_batch_id=reset_batch_id)
        if host._loop_recorder is not None:
            host._loop_recorder.reset()
        if host._mutation_tracker is not None:
            if is_retry and host._mutation_tracker.current_turn is not None:
                host._mutation_tracker.reset_file_cache()
            else:
                host._mutation_tracker.start_turn(host._turn_number)
        await self._record_trajectory_turn_started(
            is_retry=is_retry,
            has_opening_input=has_opening_input,
            preparation_scope_operation_id=preparation_scope_operation_id,
        )

    async def _record_trajectory_turn_started(
        self,
        *,
        is_retry: bool,
        has_opening_input: bool,
        preparation_scope_operation_id: str | None,
    ) -> None:
        """Open the pass's trajectory turn and bind the run context on the executor."""
        host = self._host
        recorder = host._trajectory_recorder
        self._opening_item_id = new_analytics_id() if has_opening_input else None
        await recorder.turn_started(
            turn_number=host._turn_number,
            is_retry=is_retry,
            agent_profile_fingerprint=host._agent_profile_fingerprint,
            model_profile_fingerprint=host._model_profile_fingerprint,
            primary_cwd=host._workspace.primary_cwd if host._workspace is not None else "",
            history_state=host._executor.history_state,
            opening_item_id=self._opening_item_id,
            preparation_scope_operation_id=preparation_scope_operation_id,
        )
        context = recorder.context()
        if context is not None:
            context = context.with_exchange_facts(
                trajectory_recorder.exchange_facts(
                    agent_profile_name=host._agent_profile.name if host._agent_profile is not None else "",
                    agent_profile_fingerprint=host._agent_profile_fingerprint,
                    model_profile=host._active_profile,
                    model_profile_fingerprint=host._model_profile_fingerprint,
                )
            )
        host._executor.trajectory_context = context
        host._executor.set_opening_item_id(self._opening_item_id)

    @staticmethod
    def _preparation_operation_id(preparation: PreparationTrace | None) -> str | None:
        """Return a committed pre-turn operation id for the turn-start join."""
        if preparation is None or not preparation.start_committed:
            return None
        return preparation.operation_id

    async def _prepare_user_contents(self, text: str) -> list[Any] | None:
        """Return prepared user-message contents or publish a recoverable attachment error."""
        result = await self._prompt_content_preparer_factory(self._host).prepare_fresh(text)
        return None if result is None else result.contents

    def _fire_turn_started(self) -> None:
        """Run the turn-started callback; failures never block the run."""
        try:
            self._host._on_turn_started()
        except Exception:
            logger.debug("Failed to run turn started callback", exc_info=True)

    def _open_turn_preamble(self) -> PreparationTrace | None:
        return PreparationTrace.open(
            scope=PreparationScope.TURN_PREAMBLE,
            phase="turn_dispatch",
            context=self._host._executor.trajectory_context,
        )

    def _bind_turn_preamble(self, preamble: PreparationTrace) -> None:
        if not preamble.start_committed:
            return
        context = self._host._executor.trajectory_context
        if context is not None:
            self._host._executor.trajectory_context = context.with_turn_preamble(preamble.operation_id)

    async def _fire_before_turn(
        self,
        user_text: str,
        *,
        is_retry: bool = False,
        target_operation_id: str | None = None,
    ) -> None:
        """Publish a ``before_turn`` hook event."""
        await TurnHookDispatcher(self._host).fire_before_turn(
            user_text,
            is_retry=is_retry,
            target_operation_id=target_operation_id,
        )

    async def _compute_workspace_notice(self, *, is_retry: bool) -> None:
        """Compute advisory boundary state before reminder preparation."""
        host = self._host
        if not host._settings.workspace_change_notice:
            return
        turn_id = host._turn_number
        previous_turn = None if is_retry else turn_id - 1
        overlap_turn = turn_id if is_retry else turn_id - 1
        latest_agent_turn = turn_id if is_retry else turn_id - 1
        cwd = host._workspace.primary_cwd if host._workspace is not None else ""
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    host._workspace_change_tracker.compute_turn_notice,
                    turn_id=turn_id,
                    mutation_tracker=host._mutation_tracker,
                    agent_turn_id=previous_turn,
                    overlap_turn_id=overlap_turn,
                    cwd=cwd,
                    max_entries=host._settings.workspace_change_notice_max_entries,
                    recovered=host._recovered_from_sidecar,
                    latest_agent_turn=latest_agent_turn,
                ),
                timeout=15.0,
            )
        except asyncio.CancelledError:
            host._workspace_change_tracker.notice_cancelled()
            raise
        except TimeoutError:
            # The turn continues to finalization, whose fresh baseline
            # capture absorbs whatever this comparison would have
            # reported — leave a persistent caveat, never absorb silently.
            host._workspace_change_tracker.notice_timed_out()
            logger.debug("Workspace notice computation exceeded its deadline")
        except Exception:
            host._workspace_change_tracker.clear_boundary_notice()
            logger.debug("Workspace notice computation failed", exc_info=True)

    def _requeue_undelivered_file_change(self) -> None:
        """Return a drained notice that no model request received."""
        host = self._host
        if host._reminder_middleware is None:
            return
        notice = host._reminder_middleware.take_undelivered_file_change()
        if notice:
            host._workspace_change_tracker.requeue_notice(
                notice,
                cwd=host._workspace.primary_cwd if host._workspace is not None else None,
            )

    async def _refresh_runtime_skills(self, *, update_active_turn: bool = False) -> None:
        """Refresh runtime skills and optionally update the active reminder snapshot."""
        await RuntimeSkillRefresher(self._host).refresh(update_active_turn=update_active_turn)

    def _retry_approval_context_messages(self, additional_text: str) -> list[str]:
        """Return current-turn user texts approval should see during a retry."""
        messages = self._current_turn_user_messages()
        stripped = additional_text.strip()
        if stripped:
            messages.append(stripped)
        if messages:
            return messages
        latest = self._latest_user_message()
        return [latest] if latest else []

    def _current_turn_user_messages(self) -> list[str]:
        """Return user messages after the last turn marker.

        Feeds the approval-judge context: synthetic ``continue`` nudges are
        skipped — they are orchestration placeholders, not user input, and must
        not be presented to the judge as such.  Injections and guidance
        stay: they ARE user input.
        """
        history_messages = self._host._history.messages
        start = current_turn_start(history_messages)

        messages: list[str] = []
        for message in history_messages[start:]:
            if message.role == "user" and not is_continuation_message(message):
                text = (message.text or "").strip()
                if text:
                    messages.append(text)
        return messages

    def _latest_user_message(self) -> str:
        """Return the latest REAL user message in the full history.

        Fallback leg of the approval-judge context: skips synthetic
        ``continue`` nudges so a synthetic-only current region surfaces the
        last real user message (typically the interrupted turn's opener)
        instead of a fabricated ``continue`` request.
        """
        for message in reversed(self._host._history.messages):
            if message.role == "user" and not is_continuation_message(message):
                text = (message.text or "").strip()
                if text:
                    return text
        return ""

    def _tag_consumed_profile_switch(self) -> None:
        """Tag the last user message when the reminder middleware consumed a profile switch."""
        host = self._host
        if host._reminder_middleware is None or not host._reminder_middleware.consumed_switch_to:
            return
        host._history.tag_last_user_message(
            HistoryMarkerKind.PROFILE_SWITCH_TO_KEY,
            host._reminder_middleware.consumed_switch_to,
        )

    def _queue_skill_reference_reminder(self, text: str, *, for_next_turn: bool) -> None:
        """Queue a system reminder when *text* starts with a loaded skill reference."""
        host = self._host
        if host._reminder_middleware is None:
            return
        reminder = self._skill_reference_reminder(text)
        if reminder is None:
            return
        host._reminder_middleware.queue_hook_reminders(
            [reminder],
            for_next_turn=for_next_turn,
        )

    def _skill_reference_reminder(
        self,
        text: str,
        *,
        skill_details: list[RuntimeSkillDetails] | None = None,
    ) -> str | None:
        """Return the skill-reference reminder for *text*, if it names a loaded skill."""
        reference = parse_skill_reference(text, skill_details or self._host._runtime_details.skill_details)
        if reference is None:
            return None
        return format_skill_reference_reminder(reference)

    def _complete_finalized_run(self, outcome: PostRunOutcome) -> None:
        """Synchronously finish retry dispatch, scope expiry, and recovery cleanup."""
        host = self._host
        task_before_pending_retry = host._turn_state.run_task
        RetryCoordinator(host).start_pending_retry_if_due()
        host._turn_state.discard_pre_executor_interrupt(task_before_pending_retry)
        task_after_pending_retry = host._turn_state.run_task
        retry_dispatched = (
            task_after_pending_retry is not None
            and task_after_pending_retry is not task_before_pending_retry
            and not task_after_pending_retry.done()
        )
        if not retry_dispatched and not outcome.failed:
            _expire_current_run_scope(host, outcome.completed_scope)
        self._clear_current_input()

    def _clear_current_input(self) -> None:
        """Clear current-turn recovery input after run finalization."""
        self._host._turn_state.clear_current_input()
