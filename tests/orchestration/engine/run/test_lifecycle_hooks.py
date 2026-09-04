# Copyright (c) 2026 Chrys. All rights reserved.

"""Regression tests for run lifecycle hook gates."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import chrys.orchestration.engine.run.attachments as attachment_helpers
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    AgentRuntimeDetails,
    AgentRuntimeUpdated,
    Error,
    RuntimeModelDetails,
    RuntimeSkillDetails,
    UserInject,
    UserInjectResult,
    UserMessage,
    UserRetry,
    Warning,
)
from chrys.foundation.i18n import DisplayBlock
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.workspace import Workspace
from chrys.foundation.trajectory.event_types import EventType
from chrys.kernel import Content, Message
from chrys.orchestration.engine.run.coordinator import TurnCoordinator
from chrys.orchestration.engine.run.finalizer import _expire_current_run_scope
from chrys.orchestration.engine.run.retry import RetryCoordinator
from chrys.orchestration.engine.run.runtime_skills import RuntimeSkillRefresher
from chrys.orchestration.engine.run.turn_hooks import PromptSubmitGate, TurnHookDispatcher
from chrys.orchestration.engine.run.turn_state import PendingRetry, TurnRuntimeState
from chrys.orchestration.engine.state.machine import EngineState, Trigger
from chrys.service.agent_middleware.system_reminder import CurrentRunReminderScope, CurrentRunReminderTarget
from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.manager import HookManager
from chrys.service.hooks.runner import HookResult
from chrys.service.hooks.schema import HookConfig, HookDecision, HookExecution, HookRun, HooksFile
from chrys.service.skills.model import SkillProviderWarning
from chrys.service.trajectory.preparation import PreparationOutcome
from chrys.service.trajectory.waits import WaitOutcome
from tests.service.trajectory._fakes import FakeSink, make_context


async def on_user_message(host: object, event: UserMessage) -> None:
    await TurnCoordinator(host).on_user_message(event)  # type: ignore[arg-type]


async def on_user_retry(host: object, event: UserRetry) -> None:
    await TurnCoordinator(host).on_user_retry(event)  # type: ignore[arg-type]


async def on_user_inject(host: object, event: UserInject) -> None:
    await TurnCoordinator(host).on_user_inject(event)  # type: ignore[arg-type]


class _PromptHookManager:
    def __init__(self, decision: HookDecision) -> None:
        self.decision = decision
        self.payloads: list[dict[str, Any]] = []
        self.target_operation_ids: list[object] = []

    def has_hooks_for(self, event: HookEvent) -> bool:
        return event == HookEvent.USER_PROMPT_SUBMIT

    async def fire(self, _event: HookEvent, payload: dict[str, Any], **kwargs: object) -> HookDecision:
        self.payloads.append(payload)
        self.target_operation_ids.append(kwargs.get("target_operation_id"))
        return self.decision


class _BlockingPromptHookManager:
    def __init__(self, decision: HookDecision, *, blocked_calls: int = 1) -> None:
        self.decision = decision
        self.payloads: list[dict[str, Any]] = []
        self.entered = asyncio.Event()
        self.second_entered = asyncio.Event()
        self.release = asyncio.Event()
        self._blocked_calls = blocked_calls

    def has_hooks_for(self, event: HookEvent) -> bool:
        return event == HookEvent.USER_PROMPT_SUBMIT

    async def fire(self, _event: HookEvent, payload: dict[str, Any], **_kwargs: object) -> HookDecision:
        self.payloads.append(payload)
        if len(self.payloads) == 2:
            self.second_entered.set()
        if self._blocked_calls > 0:
            self._blocked_calls -= 1
            self.entered.set()
            await self.release.wait()
        return self.decision


class _Executor:
    def __init__(self, *, running: bool = False) -> None:
        self.is_running = running
        self.trajectory_context = None
        self.injected: list[str] = []
        self.approval_context: list[str] = []

    def inject(self, text: str, **_kwargs: object) -> None:
        self.injected.append(text)

    def append_user_message(self, text: str) -> None:
        self.approval_context.append(text)


class _SkillsProvider:
    def __init__(self, skills: list[RuntimeSkillDetails]) -> None:
        self._skills = skills
        self.refresh_calls = 0

    async def refresh_context(self) -> list[Warning]:
        self.refresh_calls += 1
        return []

    def skill_names(self) -> list[str]:
        return [skill.name for skill in self._skills]

    def skill_sources(self) -> dict[str, list[str]]:
        sources: dict[str, list[str]] = {}
        for skill in self._skills:
            sources.setdefault(skill.source, []).append(skill.name)
        return sources

    def skill_details(self) -> list[RuntimeSkillDetails]:
        return list(self._skills)


class _StagedRefresh:
    def __init__(self, skills: list[RuntimeSkillDetails]) -> None:
        self._skills = skills
        self.warnings: list[Warning] = []

    def skill_names(self) -> list[str]:
        return [skill.name for skill in self._skills]

    def skill_sources(self) -> dict[str, list[str]]:
        sources: dict[str, list[str]] = {}
        for skill in self._skills:
            sources.setdefault(skill.source, []).append(skill.name)
        return sources

    def skill_details(self) -> list[RuntimeSkillDetails]:
        return list(self._skills)

    def render_catalog_reminder(self) -> str:
        return "staged catalog"


class _StagedSkillsProvider(_SkillsProvider):
    def __init__(
        self,
        skills: list[RuntimeSkillDetails],
        *,
        commit_warnings: list[SkillProviderWarning] | None = None,
    ) -> None:
        super().__init__(skills)
        self._commit_warnings = list(commit_warnings or [])
        self.stage_calls = 0
        self.commit_calls = 0

    async def stage_context_refresh(self) -> _StagedRefresh:
        self.stage_calls += 1
        return _StagedRefresh(self._skills)

    def commit_context_refresh(self, _staged: _StagedRefresh) -> list[SkillProviderWarning]:
        self.commit_calls += 1
        return list(self._commit_warnings)


class _Fsm:
    def __init__(self, *, state: EngineState = EngineState.INTERRUPTED, awaiting_sub_agents: bool = False) -> None:
        self.state = state
        self.awaiting_sub_agents = awaiting_sub_agents
        self.transitions: list[Trigger] = []

    def is_running(self) -> bool:
        return self.state in (EngineState.RUNNING, EngineState.PENDING_RETRY, EngineState.AWAITING_SUB_AGENTS)

    def is_awaiting_sub_agents(self) -> bool:
        return self.awaiting_sub_agents

    def try_transition(self, trigger: Trigger) -> None:
        self.transitions.append(trigger)
        if trigger == Trigger.RETRY_REQUESTED:
            self.state = EngineState.PENDING_RETRY
        elif trigger == Trigger.RETRY_STARTED or trigger == Trigger.USER_MESSAGE:
            self.state = EngineState.RUNNING
        elif trigger == Trigger.RUN_FAILED:
            self.state = EngineState.FAILED


class _History:
    def __init__(self) -> None:
        self.removed_trailing_markers = 0
        self.removed_orphaned_user_messages = 0
        self.messages: list[Message] = []

    def remove_trailing_markers(self) -> None:
        self.removed_trailing_markers += 1

    def remove_orphaned_user_message(self) -> None:
        self.removed_orphaned_user_messages += 1

    def has_trailing_error_markers(self) -> bool:
        return False


class _Reminder:
    def __init__(self) -> None:
        self.queued: list[tuple[list[str], bool]] = []
        self._next_scope_id = 1
        self._scopes: set[CurrentRunReminderScope] = set()
        self.valid = True

    def queue_hook_reminders(self, reminders: list[str], *, for_next_turn: bool = False) -> None:
        self.queued.append((reminders, for_next_turn))

    def create_current_run_scope(self) -> CurrentRunReminderScope:
        scope = CurrentRunReminderScope(self._next_scope_id)
        self._next_scope_id += 1
        self._scopes.add(scope)
        return scope

    def capture_current_run_target(self, reminder_scope: CurrentRunReminderScope) -> CurrentRunReminderTarget | None:
        if reminder_scope not in self._scopes:
            return None
        return CurrentRunReminderTarget(reminder_scope, "pre_prepare", 0)

    def queue_hook_reminders_for_current_run(
        self,
        _target: CurrentRunReminderTarget,
        reminders: list[str],
    ) -> bool:
        if not self.valid:
            return False
        if reminders:
            self.queued.append((reminders, False))
        return True

    def update_skill_catalog_for_current_run(self, _target: CurrentRunReminderTarget) -> bool:
        return self.valid

    def set_skill_catalog_for_current_run(
        self,
        _target: CurrentRunReminderTarget,
        _skill_catalog: str | None,
    ) -> bool:
        return self.valid

    def is_current_run_target_valid(self, _target: CurrentRunReminderTarget) -> bool:
        return self.valid

    def expire_current_run_scope(self, reminder_scope: CurrentRunReminderScope) -> None:
        self._scopes.discard(reminder_scope)

    def update_skill_catalog_for_active_turn(self) -> None:
        return None


class _Host:
    def __init__(self, *, decision: HookDecision, executor_running: bool = False) -> None:
        self._agent_loading = False
        self._bus = EventBus()
        self._session_id = "s1"
        self._executor = _Executor(running=executor_running)
        self._fsm = _Fsm(state=EngineState.RUNNING if executor_running else EngineState.INTERRUPTED)
        self._turn_state = TurnRuntimeState()
        self._history = _History()
        self._history.messages.append(Message("user", ["original request"]))
        self._reminder_middleware = _Reminder()
        self._runtime_meta = None
        self._runtime_details = AgentRuntimeDetails(model=RuntimeModelDetails(vision=True))
        self._consumed_injections = []
        self._intermediate_texts = {}
        self._turn_number = 1
        self._loop_recorder = None
        self._mutation_tracker = None
        self._mutation_coordinator = None
        self._injection = None
        self._shutting_down = False
        self._paused_sub_agents = set()
        self._sub_agent_tools = None
        self._skills_provider = None
        self._tool_names = ["read_file", "load_skill"]
        self._skill_names = []
        self.session_generation = 0
        self.build_generation = 0
        self._turn_state.pending_retry = PendingRetry(text="stale", session_generation=self.session_generation)
        self._hook_manager = _PromptHookManager(decision)
        self._agent_profile = None
        self._requirement_clarification_workflow = None
        self._requirement_enrichment_workflow = None
        self._memory_files = ["AGENTS.md"]
        self._workspace = None
        self._trajectory_recorder = _TrajectoryRecorder()
        self.retry_texts: list[str] = []
        self.run_texts: list[str] = []
        self.run_started = asyncio.Event()
        self.finish_runs = asyncio.Event()

    async def _wait_for_agent_load_idle(self) -> None:
        return None

    async def _run_and_save(self, text: str, **_kwargs: object) -> None:
        self.run_texts.append(text)
        self.run_started.set()
        await self.finish_runs.wait()

    async def _retry_and_save(self, additional_text: str = "", **_kwargs: object) -> None:
        self.retry_texts.append(additional_text)


class _TrajectoryRecorder:
    def __init__(self, context: Any = None) -> None:
        self._context = context

    def context(self) -> Any:
        return self._context


async def _never_finishes() -> None:
    await asyncio.Event().wait()


def _install_active_run(host: _Host) -> None:
    scope = host._turn_state.begin_current_run_scope(
        owner_admission_id=1,
        session_generation=host.session_generation,
        build_generation=host.build_generation,
        reminder_scope=host._reminder_middleware.create_current_run_scope(),
    )
    host._turn_state.open_injection_admission(scope)
    if host._turn_state.run_task is None:
        host._turn_state.run_task = asyncio.create_task(_never_finishes())


async def _cancel_active_run(host: _Host) -> None:
    task = host._turn_state.run_task
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _collect_errors(bus: EventBus) -> list[Error]:
    errors: list[Error] = []

    async def _on_error(event: Error) -> None:
        errors.append(event)

    await bus.subscribe(Error, _on_error)
    return errors


async def _collect_warnings(bus: EventBus) -> list[Warning]:
    warnings: list[Warning] = []

    async def _on_warning(event: Warning) -> None:
        warnings.append(event)

    await bus.subscribe(Warning, _on_warning)
    return warnings


def _status_and_turn_markers() -> tuple[Message, Message]:
    interrupted = Message("assistant", ["Execution interrupted"])
    interrupted.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.INTERRUPTED
    turn = Message("assistant", [""])
    turn.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    return interrupted, turn


@pytest.mark.asyncio
async def test_user_interrupt_hook_is_detached_from_turn_drain(tmp_path: Path) -> None:
    host = _Host(decision=HookDecision())
    hook = HookConfig(
        id="interrupt-observer",
        event=HookEvent.USER_INTERRUPT,
        run=HookRun(type="command", argv=["unused"]),
        execution=HookExecution(mode="async"),
    )
    manager = HookManager(file=HooksFile(hooks=[hook]), hooks_dir=tmp_path)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def _run_and_wait(_hook: HookConfig, _payload: dict[str, Any]) -> HookResult:
        entered.set()
        await release.wait()
        return HookResult(hook_id=hook.id, exit_code=0)

    manager._runner.run_and_wait = _run_and_wait  # type: ignore[method-assign]
    sink = FakeSink()
    manager.trajectory_context_provider = lambda: make_context(sink)
    host._hook_manager = manager

    try:
        await TurnHookDispatcher(host).fire_user_interrupt()
        await asyncio.wait_for(entered.wait(), timeout=5.0)

        started = sink.only(EventType.HOOK_OPERATION_STARTED)
        assert started.payload["drain_scope"] == "detached"
        assert await manager.drain_turn() == ()
        assert manager.active_turn_drain_operation_ids == ()
        assert any(not tracked.task.done() for tracked in manager._inflight)
    finally:
        release.set()
        await manager._drain(scopes={"detached"}, timeout=None)

    sink.assert_operations_settled()


@pytest.mark.parametrize(
    "logical_messages",
    [
        [],
        [Message("assistant", ["completed work without a user anchor"])],
    ],
    ids=["no-work", "with-work"],
)
@pytest.mark.asyncio
async def test_retry_admission_rejects_anchorless_terminal_history_without_mutation(
    logical_messages: list[Message],
) -> None:
    """Anchorless states are rejected before admission, marker, or FSM mutation."""
    host = _Host(decision=HookDecision())
    interrupted, turn = _status_and_turn_markers()
    messages = [*logical_messages, interrupted, turn]
    host._history.messages = messages
    errors = await _collect_errors(host._bus)

    await RetryCoordinator(host).handle_user_retry(UserRetry())

    assert len(errors) == 1
    assert errors[0].code == "executor_error"
    assert errors[0].message == "Cannot retry without a real user message in history."
    assert errors[0].display_message is not None
    assert errors[0].display_message.definition.key == "retry.missing_user_anchor"
    assert dict(errors[0].display_message.args) == {}
    assert errors[0].recoverable is True
    assert errors[0].session_id == "s1"
    assert host._history.messages == messages
    assert host._history.removed_trailing_markers == 0
    assert host._fsm.state is EngineState.INTERRUPTED
    assert host._fsm.transitions == []
    assert host._turn_state.next_admission_id == 1
    assert host._turn_state.active_admission_count() == 0
    assert host._turn_state.run_task is None


@pytest.mark.parametrize(
    ("reason", "expected_message", "expected_key"),
    [
        ("hook 'guard' blocked: denied", "hook 'guard' blocked: denied", None),
        ("", "Prompt blocked by hook.", "turn_hooks.prompt_blocked"),
    ],
    ids=["hook-authored-reason", "fixed-fallback"],
)
@pytest.mark.asyncio
async def test_prompt_submit_blocked_error_preserves_reason_ownership(
    reason: str,
    expected_message: str,
    expected_key: str | None,
) -> None:
    host = _Host(decision=HookDecision())
    errors = await _collect_errors(host._bus)

    blocked = await PromptSubmitGate(host).handle_decision(
        HookDecision(blocked=True, block_reason=reason),
        injected=None,
    )

    assert blocked is True
    assert len(errors) == 1
    error = errors[0]
    assert (error.code, error.message, error.session_id) == ("hook_blocked", expected_message, "s1")
    if expected_key is None:
        assert error.display_message is None
    else:
        assert error.display_message is not None
        assert error.display_message.definition.key == expected_key
        assert dict(error.display_message.args) == {}


@pytest.mark.asyncio
async def test_retry_admission_accepts_finalized_failed_turn_with_real_user() -> None:
    """Trailing status/turn markers do not hide the preceding turn's user anchor."""
    host = _Host(decision=HookDecision())
    interrupted, turn = _status_and_turn_markers()
    host._history.messages = [
        Message("user", ["original request"]),
        Message("assistant", ["completed work"]),
        interrupted,
        turn,
    ]
    errors = await _collect_errors(host._bus)

    await RetryCoordinator(host).handle_user_retry(UserRetry())
    assert host._turn_state.run_task is not None
    await host._turn_state.run_task

    assert errors == []
    assert host.retry_texts == [""]
    assert host._history.removed_trailing_markers == 1
    assert host._fsm.state is EngineState.RUNNING


async def _collect_runtime_updates(bus: EventBus) -> list[AgentRuntimeUpdated]:
    updates: list[AgentRuntimeUpdated] = []

    async def _on_update(event: AgentRuntimeUpdated) -> None:
        updates.append(event)

    await bus.subscribe(AgentRuntimeUpdated, _on_update)
    return updates


@pytest.mark.asyncio
async def test_refresh_runtime_skills_publishes_runtime_update() -> None:
    skill = RuntimeSkillDetails(name="unit-converter", description="Convert units", source="/skills/unit")
    host = _Host(decision=HookDecision())
    host._runtime_details.model.profile_id = "deepseek"
    host._runtime_details.model.max_context_tokens = 1_000_000
    host._skills_provider = _SkillsProvider([skill])
    updates = await _collect_runtime_updates(host._bus)

    await RuntimeSkillRefresher(host).refresh()

    assert host._skill_names == ["unit-converter"]
    assert host._runtime_details.skill_sources == {"/skills/unit": ["unit-converter"]}
    assert host._runtime_details.skill_details == [skill]
    assert len(updates) == 1
    assert updates[0].model_profile_id == "deepseek"
    assert updates[0].max_context_tokens == 1_000_000
    assert updates[0].tool_names == ["read_file", "load_skill"]
    assert updates[0].skill_names == ["unit-converter"]
    assert updates[0].memory_files == ["AGENTS.md"]
    assert updates[0].runtime_details.skill_details == [skill]
    assert updates[0].runtime_details is not host._runtime_details


@pytest.mark.asyncio
async def test_user_retry_text_blocked_by_user_prompt_submit_hook() -> None:
    host = _Host(
        decision=HookDecision(blocked=True, block_reason="hook 'guard' blocked: denied"),
        executor_running=True,
    )
    errors = await _collect_errors(host._bus)

    await on_user_retry(host, UserRetry(text="continue with secret"))

    assert host._turn_state.pending_retry.text == ""
    assert host._fsm.transitions == []
    assert host.retry_texts == []
    assert errors[0].code == "hook_blocked"
    assert host._hook_manager.payloads[0]["text"] == "continue with secret"
    assert host._hook_manager.payloads[0]["injected"] is True


@pytest.mark.asyncio
async def test_user_retry_text_with_image_mention_is_rejected_after_prompt_hook(tmp_path: Path) -> None:
    (tmp_path / "shot.png").write_bytes(b"abc")
    host = _Host(decision=HookDecision(system_reminders=["do not queue"]))
    host._workspace = Workspace(primary_cwd=str(tmp_path))
    warnings = await _collect_warnings(host._bus)

    await on_user_retry(host, UserRetry(text="continue with @shot.png"))

    assert len(warnings) == 1
    warning = warnings[0]
    image_path = tmp_path / "shot.png"
    assert (warning.code, warning.message, warning.session_id) == (
        "image_attachment_retry_unsupported",
        (
            "Images cannot be attached to retry or continuation prompts.\n\n"
            "Send the image in a new message after this retry or continuation finishes.\n\n"
            f"Image not attached:\n- {image_path}"
        ),
        "s1",
    )
    assert warning.display_message is not None
    assert warning.display_message.definition.key == "attachments.retry_image_unsupported"
    assert dict(warning.display_message.args) == {"files": DisplayBlock(f"- {image_path}")}
    assert warning.display_message.count == 1
    assert host._hook_manager.payloads[0]["text"] == "continue with @shot.png"
    assert host._turn_state.pending_retry.text == ""
    assert host._fsm.transitions == []
    assert host.retry_texts == []
    assert host._reminder_middleware.queued == []


@pytest.mark.asyncio
async def test_user_retry_invalid_image_publishes_legacy_and_localized_warning(tmp_path: Path) -> None:
    host = _Host(decision=HookDecision())
    host._workspace = Workspace(primary_cwd=str(tmp_path))
    warnings = await _collect_warnings(host._bus)
    discovered = attachment_helpers.discover_image_mentions("continue with @missing.png", tmp_path)

    await on_user_retry(host, UserRetry(text="continue with @missing.png"))

    assert len(warnings) == 1
    warning = warnings[0]
    items = DisplayBlock("\n".join(f"- {item}" for item in discovered.errors))
    assert (warning.code, warning.message, warning.session_id) == (
        "image_attachment_error",
        attachment_helpers.format_attachment_error_message(discovered.errors),
        "s1",
    )
    assert warning.display_message is not None
    assert warning.display_message.definition.key == "attachments.attachment_error"
    assert dict(warning.display_message.args) == {"items": items}
    assert warning.display_message.count == 1
    assert host._turn_state.pending_retry.text == ""
    assert host._fsm.transitions == []
    assert host.retry_texts == []


@pytest.mark.asyncio
async def test_empty_user_retry_after_image_prompt_is_rejected() -> None:
    host = _Host(decision=HookDecision())
    host._history.messages.append(
        Message(
            "user",
            [
                "describe @shot.png",
                Content.from_data(data=b"abc", media_type="image/png"),
            ],
        )
    )
    warnings = await _collect_warnings(host._bus)

    await on_user_retry(host, UserRetry())

    assert len(warnings) == 1
    warning = warnings[0]
    assert (warning.code, warning.message, warning.session_id) == (
        "image_attachment_retry_unsupported",
        (
            "This retry cannot include the original image attachment.\n\n"
            "The retry path would resend only the text from the original prompt. "
            "Send the image again in a new message instead."
        ),
        "s1",
    )
    assert warning.display_message is not None
    assert warning.display_message.definition.key == "attachments.retry_history_image_unsupported"
    assert dict(warning.display_message.args) == {}
    assert warning.display_message.count is None
    assert host._fsm.transitions == []
    assert host.retry_texts == []


@pytest.mark.asyncio
async def test_empty_user_retry_after_failed_image_prompt_ignores_status_markers() -> None:
    host = _Host(decision=HookDecision())
    interrupted = Message("assistant", ["Execution interrupted"])
    interrupted.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.INTERRUPTED
    turn = Message("assistant", [""])
    turn.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    host._history.messages.extend(
        [
            Message(
                "user",
                [
                    "describe @shot.png",
                    Content.from_data(data=b"abc", media_type="image/png"),
                ],
            ),
            interrupted,
            turn,
        ]
    )
    warnings = await _collect_warnings(host._bus)

    await on_user_retry(host, UserRetry())

    assert len(warnings) == 1
    assert warnings[0].code == "image_attachment_retry_unsupported"
    assert host._history.removed_trailing_markers == 0
    assert host._fsm.transitions == []
    assert host.retry_texts == []


@pytest.mark.asyncio
async def test_user_retry_text_allowed_by_user_prompt_submit_hook() -> None:
    host = _Host(decision=HookDecision(system_reminders=["remember this"]))
    errors = await _collect_errors(host._bus)

    await on_user_retry(host, UserRetry(text="continue with note"))
    assert host._turn_state.run_task is not None
    await host._turn_state.run_task

    assert errors == []
    assert host.retry_texts == ["continue with note"]
    assert host._history.removed_trailing_markers == 1
    assert host._fsm.transitions == [Trigger.RETRY_STARTED]
    assert host._reminder_middleware.queued == [(["remember this"], False)]
    assert host._hook_manager.payloads[0]["injected"] is True


@pytest.mark.asyncio
async def test_uncommitted_preparation_is_never_exposed_as_a_hook_target() -> None:
    fresh = _Host(decision=HookDecision())
    fresh_sink = FakeSink()
    fresh_sink.fail_next = True
    fresh._trajectory_recorder = _TrajectoryRecorder(make_context(fresh_sink))
    await on_user_message(fresh, UserMessage(text="fresh prompt"))
    fresh.finish_runs.set()
    assert fresh._turn_state.run_task is not None
    await fresh._turn_state.run_task

    active = _Host(decision=HookDecision(), executor_running=True)
    active_sink = FakeSink()
    active_sink.fail_next = True
    active._trajectory_recorder = _TrajectoryRecorder(make_context(active_sink))
    _install_active_run(active)
    try:
        await on_user_message(active, UserMessage(text="active injection"))
    finally:
        await _cancel_active_run(active)

    retry = _Host(decision=HookDecision())
    retry_sink = FakeSink()
    retry_sink.fail_next = True
    retry._trajectory_recorder = _TrajectoryRecorder(make_context(retry_sink))
    await on_user_retry(retry, UserRetry(text="retry note"))
    assert retry._turn_state.run_task is not None
    await retry._turn_state.run_task

    assert fresh._hook_manager.target_operation_ids == [None]
    assert active._hook_manager.target_operation_ids == [None]
    assert retry._hook_manager.target_operation_ids == [None]


@pytest.mark.asyncio
async def test_user_retry_text_refreshes_runtime_skills_without_active_target() -> None:
    skill = RuntimeSkillDetails(name="unit-converter", description="Convert units", source="/skills/unit")
    provider = _StagedSkillsProvider([skill])
    host = _Host(decision=HookDecision())
    host._skills_provider = provider

    await on_user_retry(host, UserRetry(text="continue with ordinary note"))
    assert host._turn_state.run_task is not None
    await host._turn_state.run_task

    assert provider.stage_calls == 1
    assert provider.commit_calls == 1
    assert host._skill_names == ["unit-converter"]
    assert host._runtime_details.skill_details == [skill]
    assert host.retry_texts == ["continue with ordinary note"]


@pytest.mark.asyncio
async def test_user_retry_discards_promoted_task_if_deferred_retry_note_commit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_deferred_commit(
        _coordinator: RetryCoordinator,
        _side_effects: object,
        _scope: object,
    ) -> bool:
        return False

    monkeypatch.setattr(RetryCoordinator, "_commit_deferred_retry_note_side_effects", fail_deferred_commit)
    host = _Host(decision=HookDecision(system_reminders=["deferred retry note"]))
    sink = FakeSink()
    host._trajectory_recorder = _TrajectoryRecorder(make_context(sink))

    await on_user_retry(host, UserRetry(text="continue with note"))

    assert host._turn_state.run_task is None
    assert host._turn_state.run_task is None
    assert host._turn_state.current_run_scope is None
    assert host._fsm.state is EngineState.FAILED
    assert host._fsm.transitions == [Trigger.RETRY_STARTED, Trigger.RUN_FAILED]
    assert host.retry_texts == []
    assert host._reminder_middleware.queued == []
    assert sink.only(EventType.PREPARATION_FINISHED).payload["outcome"] == PreparationOutcome.PREPARATION_FAILED
    sink.assert_operations_settled()


@pytest.mark.asyncio
async def test_user_retry_text_aborts_when_ordinary_owner_changes_during_hook() -> None:
    host = _Host(decision=HookDecision())
    host._turn_state.clear_pending_retry(outcome=PreparationOutcome.DROPPED)
    host._hook_manager = _BlockingPromptHookManager(
        HookDecision(blocked=True, block_reason="hook 'guard' blocked: denied")
    )
    errors = await _collect_errors(host._bus)

    task = asyncio.create_task(on_user_retry(host, UserRetry(text="continue after restore")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        host.session_generation += 1
        host._session_id = "s2"
        host._hook_manager.release.set()
        await task

        assert host._turn_state.run_task is None
        assert host._turn_state.pending_retry.text == ""
        assert host._turn_state.pending_retry.created_at is None
        assert host.retry_texts == []
        assert host._fsm.transitions == []
        assert host._reminder_middleware.queued == []
        assert errors == []
    finally:
        host._hook_manager.release.set()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_active_user_retry_text_uses_scoped_current_run_reminders() -> None:
    host = _Host(decision=HookDecision(system_reminders=["remember this"]), executor_running=True)
    _install_active_run(host)

    try:
        await on_user_retry(host, UserRetry(text="continue with note"))
    finally:
        await _cancel_active_run(host)

    assert host._turn_state.pending_retry.text == "continue with note"
    assert host._fsm.transitions == [Trigger.RETRY_REQUESTED]
    assert host._reminder_middleware.queued == [(["remember this"], False)]
    assert host._hook_manager.payloads[0]["injected"] is True


@pytest.mark.asyncio
async def test_post_install_pending_retry_validation_keeps_specific_terminal_outcome() -> None:
    host = _Host(decision=HookDecision(system_reminders=["remember this"]), executor_running=True)
    sink = FakeSink()
    host._trajectory_recorder = _TrajectoryRecorder(make_context(sink))
    _install_active_run(host)
    validity_checks = 0

    def valid_once(_target: CurrentRunReminderTarget) -> bool:
        nonlocal validity_checks
        validity_checks += 1
        return validity_checks == 1

    host._reminder_middleware.is_current_run_target_valid = valid_once
    try:
        await on_user_retry(host, UserRetry(text="continue with note"))
    finally:
        await _cancel_active_run(host)

    assert host._turn_state.pending_retry == PendingRetry()
    assert sink.only(EventType.PREPARATION_FINISHED).payload["outcome"] == PreparationOutcome.PREPARATION_FAILED
    sink.assert_operations_settled()


@pytest.mark.asyncio
async def test_active_user_retry_text_refreshes_runtime_skills_without_hook_reminders() -> None:
    skill = RuntimeSkillDetails(name="unit-converter", description="Convert units", source="/skills/unit")
    provider = _StagedSkillsProvider([skill])
    host = _Host(decision=HookDecision(), executor_running=True)
    host._skills_provider = provider
    _install_active_run(host)

    try:
        await on_user_retry(host, UserRetry(text="continue with ordinary note"))
    finally:
        await _cancel_active_run(host)

    assert provider.stage_calls == 1
    assert provider.commit_calls == 1
    assert host._skill_names == ["unit-converter"]
    assert host._runtime_details.skill_details == [skill]
    assert host._turn_state.pending_retry.text == "continue with ordinary note"


@pytest.mark.asyncio
async def test_user_retry_text_when_executor_finishes_during_hook_keeps_scoped_reminders() -> None:
    host = _Host(decision=HookDecision(), executor_running=True)
    release_run = asyncio.Event()
    host._turn_state.run_task = asyncio.create_task(release_run.wait())
    _install_active_run(host)
    host._hook_manager = _BlockingPromptHookManager(HookDecision(system_reminders=["stale retry note"]))

    task = asyncio.create_task(on_user_retry(host, UserRetry(text="continue after stale")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        host._executor.is_running = False
        release_run.set()
        host._hook_manager.release.set()
        await task
        assert host.retry_texts == ["continue after stale"]
        assert host._reminder_middleware.queued == [(["stale retry note"], False)]
    finally:
        host._hook_manager.release.set()
        await _cancel_active_run(host)
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_user_retry_text_after_finalization_expires_scope_keeps_retry_reminders() -> None:
    host = _Host(decision=HookDecision(), executor_running=True)
    release_run = asyncio.Event()
    finalized = asyncio.Event()

    async def _finalize_and_expire_scope() -> None:
        await release_run.wait()
        host._executor.is_running = False
        _expire_current_run_scope(host, host._turn_state.current_run_scope)
        host._fsm.state = EngineState.INTERRUPTED
        finalized.set()

    host._turn_state.run_task = asyncio.create_task(_finalize_and_expire_scope())
    _install_active_run(host)
    host._hook_manager = _BlockingPromptHookManager(HookDecision(system_reminders=["finalized retry note"]))

    task = asyncio.create_task(on_user_retry(host, UserRetry(text="continue after finalized")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        release_run.set()
        await asyncio.wait_for(finalized.wait(), timeout=5.0)
        host._hook_manager.release.set()
        await task
        assert host._turn_state.run_task is not None
        await host._turn_state.run_task

        assert host.retry_texts == ["continue after finalized"]
        assert host._reminder_middleware.queued == [(["finalized retry note"], False)]
        assert host._fsm.transitions == [Trigger.RETRY_STARTED]
    finally:
        release_run.set()
        host._hook_manager.release.set()
        await _cancel_active_run(host)
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_user_retry_text_aborts_when_active_owner_changes_during_hook() -> None:
    host = _Host(decision=HookDecision(), executor_running=True)
    host._turn_state.clear_pending_retry(outcome=PreparationOutcome.DROPPED)
    _install_active_run(host)
    host._hook_manager = _BlockingPromptHookManager(
        HookDecision(blocked=True, block_reason="hook 'guard' blocked: denied")
    )
    errors = await _collect_errors(host._bus)

    task = asyncio.create_task(on_user_retry(host, UserRetry(text="continue after restore")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        host.session_generation += 1
        host._session_id = "s2"
        host._hook_manager.release.set()
        await task

        assert host._turn_state.pending_retry.text == ""
        assert host._turn_state.pending_retry.created_at is None
        assert host.retry_texts == []
        assert host._fsm.transitions == []
        assert host._reminder_middleware.queued == []
        assert errors == []
    finally:
        host._hook_manager.release.set()
        await _cancel_active_run(host)
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_user_inject_blocked_by_user_prompt_submit_hook() -> None:
    host = _Host(
        decision=HookDecision(blocked=True, block_reason="hook 'guard' blocked: denied"),
        executor_running=True,
    )
    _install_active_run(host)
    errors = await _collect_errors(host._bus)

    try:
        await on_user_inject(host, UserInject(text="inject secret"))
    finally:
        await _cancel_active_run(host)

    assert host._executor.injected == []
    assert host._executor.approval_context == []
    assert errors[0].code == "hook_blocked"
    assert host._hook_manager.payloads[0]["text"] == "inject secret"
    assert host._hook_manager.payloads[0]["injected"] is True


@pytest.mark.asyncio
async def test_user_message_active_injection_stale_owner_after_blocked_hook_suppresses_error() -> None:
    host = _Host(decision=HookDecision(), executor_running=True)
    _install_active_run(host)
    host._hook_manager = _BlockingPromptHookManager(
        HookDecision(blocked=True, block_reason="hook 'guard' blocked: denied")
    )
    errors = await _collect_errors(host._bus)

    task = asyncio.create_task(on_user_message(host, UserMessage(text="inject after restore")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        host.session_generation += 1
        host._session_id = "s2"
        host._hook_manager.release.set()
        await task
    finally:
        host._hook_manager.release.set()
        await _cancel_active_run(host)
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert errors == []
    assert host._executor.injected == []
    assert host._reminder_middleware.queued == []


@pytest.mark.asyncio
async def test_user_inject_stale_owner_after_blocked_hook_suppresses_error() -> None:
    host = _Host(decision=HookDecision(), executor_running=True)
    _install_active_run(host)
    host._hook_manager = _BlockingPromptHookManager(
        HookDecision(blocked=True, block_reason="hook 'guard' blocked: denied")
    )
    errors = await _collect_errors(host._bus)

    task = asyncio.create_task(on_user_inject(host, UserInject(text="inject after restore")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        host.session_generation += 1
        host._session_id = "s2"
        host._hook_manager.release.set()
        await task
    finally:
        host._hook_manager.release.set()
        await _cancel_active_run(host)
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert errors == []
    assert host._executor.injected == []
    assert host._reminder_middleware.queued == []


@pytest.mark.asyncio
async def test_user_inject_stale_owner_after_hook_suppresses_image_warning(tmp_path: Path) -> None:
    (tmp_path / "shot.png").write_bytes(b"abc")
    host = _Host(decision=HookDecision(), executor_running=True)
    host._workspace = Workspace(primary_cwd=str(tmp_path))
    _install_active_run(host)
    host._hook_manager = _BlockingPromptHookManager(HookDecision())
    warnings = await _collect_warnings(host._bus)

    task = asyncio.create_task(on_user_inject(host, UserInject(text="inject @shot.png after restore")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        host.session_generation += 1
        host._session_id = "s2"
        host._hook_manager.release.set()
        await task
    finally:
        host._hook_manager.release.set()
        await _cancel_active_run(host)
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert warnings == []
    assert host._executor.injected == []
    assert host._reminder_middleware.queued == []


@pytest.mark.asyncio
async def test_user_message_active_injection_stale_owner_after_hook_suppresses_image_warning(tmp_path: Path) -> None:
    (tmp_path / "shot.png").write_bytes(b"abc")
    host = _Host(decision=HookDecision(), executor_running=True)
    host._workspace = Workspace(primary_cwd=str(tmp_path))
    _install_active_run(host)
    host._hook_manager = _BlockingPromptHookManager(HookDecision())
    warnings = await _collect_warnings(host._bus)

    task = asyncio.create_task(on_user_message(host, UserMessage(text="inject @shot.png after restore")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        host.session_generation += 1
        host._session_id = "s2"
        host._hook_manager.release.set()
        await task
    finally:
        host._hook_manager.release.set()
        await _cancel_active_run(host)
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert warnings == []
    assert host._executor.injected == []
    assert host._reminder_middleware.queued == []


@pytest.mark.asyncio
async def test_user_inject_allowed_by_user_prompt_submit_hook() -> None:
    host = _Host(decision=HookDecision(system_reminders=["current turn note"]), executor_running=True)
    _install_active_run(host)
    errors = await _collect_errors(host._bus)

    try:
        await on_user_inject(host, UserInject(text="inject context"))
    finally:
        await _cancel_active_run(host)

    assert errors == []
    assert host._executor.injected == ["inject context"]
    assert host._executor.approval_context == ["inject context"]
    assert host._reminder_middleware.queued == [(["current turn note"], False)]
    assert host._hook_manager.payloads[0]["injected"] is True


@pytest.mark.asyncio
async def test_user_inject_abandoned_when_turn_ends_before_queue() -> None:
    # FSM not running by the time we reach the queue call (the turn finished during
    # the hook/skill awaits). The text must be dropped, not queued for the next turn.
    host = _Host(decision=HookDecision())
    results: list[UserInjectResult] = []

    async def _on_result(event: UserInjectResult) -> None:
        results.append(event)

    await host._bus.subscribe(UserInjectResult, _on_result)

    await on_user_inject(host, UserInject(text="late inject"))

    assert host._executor.injected == []
    assert [r.text for r in results] == ["late inject"]
    assert results[0].consumed is False
    assert results[0].session_id == "s1"


@pytest.mark.asyncio
async def test_user_inject_invalidated_after_hook_does_not_queue_current_turn_reminders() -> None:
    host = _Host(decision=HookDecision(), executor_running=True)
    _install_active_run(host)
    host._hook_manager = _BlockingPromptHookManager(HookDecision(system_reminders=["stale current-turn note"]))
    results: list[UserInjectResult] = []

    async def _on_result(event: UserInjectResult) -> None:
        results.append(event)

    await host._bus.subscribe(UserInjectResult, _on_result)

    task = asyncio.create_task(on_user_inject(host, UserInject(text="late inject")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        host._fsm.state = EngineState.IDLE
        host._hook_manager.release.set()
        await task
    finally:
        host._hook_manager.release.set()
        await _cancel_active_run(host)

    assert host._executor.injected == []
    assert [result.text for result in results] == ["late inject"]
    assert host._reminder_middleware.queued == []


@pytest.mark.asyncio
async def test_user_inject_invalid_scoped_reminder_target_does_not_commit_provider_or_inject() -> None:
    skill = RuntimeSkillDetails(name="unit-converter", description="Convert units", source="/skills/unit")
    host = _Host(decision=HookDecision(), executor_running=True)
    provider = _StagedSkillsProvider([skill])
    host._skills_provider = provider
    _install_active_run(host)
    host._hook_manager = _BlockingPromptHookManager(HookDecision(system_reminders=["stale current-turn note"]))
    results: list[UserInjectResult] = []

    async def _on_result(event: UserInjectResult) -> None:
        results.append(event)

    await host._bus.subscribe(UserInjectResult, _on_result)

    task = asyncio.create_task(on_user_inject(host, UserInject(text="late inject")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        host._reminder_middleware.valid = False
        host._hook_manager.release.set()
        await task
    finally:
        host._hook_manager.release.set()
        await _cancel_active_run(host)

    assert provider.stage_calls == 1
    assert provider.commit_calls == 0
    assert host._executor.injected == []
    assert [result.text for result in results] == ["late inject"]
    assert host._skill_names == []
    assert host._runtime_details.skill_details == []


@pytest.mark.asyncio
async def test_user_inject_publishes_warnings_returned_by_staged_refresh_commit() -> None:
    warning = SkillProviderWarning(code="skill_load_error", message="Skill 'demo' failed to load")
    host = _Host(decision=HookDecision(), executor_running=True)
    host._skills_provider = _StagedSkillsProvider([], commit_warnings=[warning])
    _install_active_run(host)
    warnings = await _collect_warnings(host._bus)

    try:
        await on_user_inject(host, UserInject(text="inject context"))
    finally:
        await _cancel_active_run(host)

    assert [event.code for event in warnings] == ["skill_load_error"]
    assert warnings[0].message == "Skill 'demo' failed to load"


@pytest.mark.asyncio
async def test_user_message_fallback_injection_invalidated_after_hook_does_not_inject_or_queue_reminders() -> None:
    host = _Host(decision=HookDecision(), executor_running=True)
    host._fsm.state = EngineState.IDLE
    host._hook_manager = _BlockingPromptHookManager(HookDecision(system_reminders=["fallback stale note"]))

    async def _never_finishes() -> None:
        await asyncio.Event().wait()

    host._turn_state.run_task = asyncio.create_task(_never_finishes())
    _install_active_run(host)
    task = asyncio.create_task(on_user_message(host, UserMessage(text="fallback text")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        host._executor.is_running = False
        host._hook_manager.release.set()
        await task

        assert host._executor.injected == []
        assert host._reminder_middleware.queued == []
    finally:
        host._hook_manager.release.set()
        await _cancel_active_run(host)
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_user_message_fallback_injection_stale_owner_after_hook_suppresses_error() -> None:
    host = _Host(decision=HookDecision(), executor_running=True)
    host._fsm.state = EngineState.IDLE
    host._hook_manager = _BlockingPromptHookManager(
        HookDecision(blocked=True, block_reason="hook 'guard' blocked: denied")
    )
    errors = await _collect_errors(host._bus)

    async def _never_finishes() -> None:
        await asyncio.Event().wait()

    host._turn_state.run_task = asyncio.create_task(_never_finishes())
    _install_active_run(host)
    task = asyncio.create_task(on_user_message(host, UserMessage(text="fallback after restore")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        host.session_generation += 1
        host._session_id = "s2"
        host._hook_manager.release.set()
        await task

        assert errors == []
        assert host._executor.injected == []
        assert host._reminder_middleware.queued == []
    finally:
        host._hook_manager.release.set()
        await _cancel_active_run(host)
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_fresh_user_message_waits_while_prompt_admission_closed() -> None:
    host = _Host(decision=HookDecision())
    host._fsm.state = EngineState.IDLE
    host._turn_state.close_prompt_admission_for_rebuild("rebuild")

    task = asyncio.create_task(on_user_message(host, UserMessage(text="fresh after rebuild")))
    try:
        await asyncio.sleep(0)
        assert not task.done()
        assert host.run_texts == []
        assert host._hook_manager.payloads == []

        host._turn_state.reopen_prompt_admission_after_rebuild("rebuild")
        await asyncio.wait_for(task, timeout=5.0)
        await asyncio.wait_for(host.run_started.wait(), timeout=5.0)

        assert [payload["text"] for payload in host._hook_manager.payloads] == ["fresh after rebuild"]
        assert host.run_texts == ["fresh after rebuild"]
    finally:
        host._turn_state.reopen_prompt_admission_after_rebuild("rebuild")
        host.finish_runs.set()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        current_task = host._turn_state.run_task
        if current_task is not None and not current_task.done():
            current_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await current_task


@pytest.mark.asyncio
async def test_fresh_user_message_reopens_preparation_in_new_session_after_transition_sweep() -> None:
    host = _Host(decision=HookDecision(blocked=True, block_reason="hook 'guard' blocked: inspected after transition"))
    host._fsm.state = EngineState.IDLE
    host._hook_manager = _BlockingPromptHookManager(host._hook_manager.decision)
    old_sink = FakeSink()
    new_sink = FakeSink()
    host._trajectory_recorder = _TrajectoryRecorder(make_context(old_sink))
    owner = "session:test-resume:1"
    host._turn_state.close_prompt_admission_for_rebuild(owner)
    gate_entered = asyncio.Event()
    wait_for_prompt_admission_open = host._turn_state.wait_for_prompt_admission_open

    async def observed_prompt_gate() -> None:
        gate_entered.set()
        await wait_for_prompt_admission_open()

    host._turn_state.wait_for_prompt_admission_open = observed_prompt_gate  # type: ignore[method-assign]
    task = asyncio.create_task(on_user_message(host, UserMessage(text="fresh in the new session")))
    try:
        await asyncio.wait_for(gate_entered.wait(), timeout=5.0)
        old_entry = next(iter(host._turn_state.pre_admission_preparations.values()))
        old_preparation = old_entry.preparation
        assert old_entry.current_wait is not None

        host._turn_state.invalidate_for_session_transition_pre_shutdown(
            old_session_generation=host.session_generation,
            prompt_admission_owner=owner,
        )
        assert old_preparation.finished_state is True
        assert old_sink.only(EventType.PREPARATION_FINISHED).payload["outcome"] == PreparationOutcome.OWNER_CHANGED
        assert WaitOutcome.CANCELLED in [
            draft.payload["outcome"] for draft in old_sink.of_type(EventType.WAIT_FINISHED)
        ]
        old_sink.assert_operations_settled()

        host.session_generation += 1
        host._session_id = "s2"
        host._trajectory_recorder = _TrajectoryRecorder(make_context(new_sink))
        host._turn_state.reopen_prompt_admission_after_rebuild(owner)
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)

        admission = next(iter(host._turn_state.active_admissions.values()))
        new_preparation = admission.preparation_trace
        assert new_preparation is not None
        assert new_preparation is not old_preparation
        assert new_preparation.finished_state is False
        assert new_preparation.context.sink is new_sink
        assert new_sink.only(EventType.PREPARATION_STARTED).operation_id == new_preparation.operation_id
        assert old_preparation.operation_id not in {draft.operation_id for draft in new_sink.drafts}
        assert admission.session_generation == host.session_generation

        host._hook_manager.release.set()
        await asyncio.wait_for(task, timeout=5.0)
        assert new_sink.only(EventType.PREPARATION_FINISHED).payload["outcome"] == PreparationOutcome.REJECTED
        new_sink.assert_operations_settled()
    finally:
        host._turn_state.reopen_prompt_admission_after_rebuild(owner)
        host._hook_manager.release.set()
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_fresh_user_message_waits_while_prompt_admission_closed_before_executor_ready() -> None:
    host = _Host(decision=HookDecision())
    host._fsm.state = EngineState.IDLE
    host._executor = None
    host._turn_state.close_prompt_admission_for_rebuild("rebuild")
    errors = await _collect_errors(host._bus)

    task = asyncio.create_task(on_user_message(host, UserMessage(text="fresh after startup rebuild")))
    try:
        await asyncio.sleep(0)
        assert not task.done()
        assert errors == []
        assert host.run_texts == []
        assert host._hook_manager.payloads == []

        host._executor = _Executor()
        host._turn_state.reopen_prompt_admission_after_rebuild("rebuild")
        await asyncio.wait_for(task, timeout=5.0)
        await asyncio.wait_for(host.run_started.wait(), timeout=5.0)

        assert errors == []
        assert [payload["text"] for payload in host._hook_manager.payloads] == ["fresh after startup rebuild"]
        assert host.run_texts == ["fresh after startup rebuild"]
    finally:
        host._turn_state.reopen_prompt_admission_after_rebuild("rebuild")
        host.finish_runs.set()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        current_task = host._turn_state.run_task
        if current_task is not None and not current_task.done():
            current_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await current_task


@pytest.mark.asyncio
async def test_user_retry_waits_while_prompt_admission_closed() -> None:
    host = _Host(decision=HookDecision())
    host._turn_state.close_prompt_admission_for_rebuild("rebuild")

    task = asyncio.create_task(on_user_retry(host, UserRetry(text="retry after rebuild")))
    try:
        await asyncio.sleep(0)
        assert not task.done()
        assert host.retry_texts == []
        assert host._hook_manager.payloads == []

        host._turn_state.reopen_prompt_admission_after_rebuild("rebuild")
        await asyncio.wait_for(task, timeout=5.0)
        current_task = host._turn_state.run_task
        if current_task is not None:
            await asyncio.wait_for(current_task, timeout=5.0)

        assert [payload["text"] for payload in host._hook_manager.payloads] == ["retry after rebuild"]
        assert host.retry_texts == ["retry after rebuild"]
    finally:
        host._turn_state.reopen_prompt_admission_after_rebuild("rebuild")
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_user_retry_waits_while_prompt_admission_closed_before_executor_ready() -> None:
    host = _Host(decision=HookDecision())
    host._executor = None
    host._turn_state.close_prompt_admission_for_rebuild("rebuild")

    task = asyncio.create_task(on_user_retry(host, UserRetry(text="retry after startup rebuild")))
    try:
        await asyncio.sleep(0)
        assert not task.done()
        assert host.retry_texts == []
        assert host._hook_manager.payloads == []

        host._executor = _Executor()
        host._turn_state.reopen_prompt_admission_after_rebuild("rebuild")
        await asyncio.wait_for(task, timeout=5.0)
        current_task = host._turn_state.run_task
        if current_task is not None:
            await asyncio.wait_for(current_task, timeout=5.0)

        assert [payload["text"] for payload in host._hook_manager.payloads] == ["retry after startup rebuild"]
        assert host.retry_texts == ["retry after startup rebuild"]
    finally:
        host._turn_state.reopen_prompt_admission_after_rebuild("rebuild")
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_prompt_admission_concurrent_fresh_user_message_waits_before_second_hook_then_routes_once() -> None:
    host = _Host(decision=HookDecision())
    host._fsm.state = EngineState.IDLE
    host._hook_manager = _BlockingPromptHookManager(HookDecision())
    first = asyncio.create_task(on_user_message(host, UserMessage(text="first fresh")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        second = asyncio.create_task(on_user_message(host, UserMessage(text="second fresh")))
        try:
            await asyncio.sleep(0)
            assert [payload["text"] for payload in host._hook_manager.payloads] == ["first fresh"]
        finally:
            host._hook_manager.release.set()
            await asyncio.wait_for(asyncio.gather(first, second), timeout=5.0)
            host.finish_runs.set()

        assert [payload["injected"] for payload in host._hook_manager.payloads] == [False, True]
        assert host.run_texts == ["first fresh"]
        assert host._executor.injected == ["second fresh"]
        assert host._turn_state.current_run_scope is not None
        assert host._turn_state.current_run_scope.owner_admission_id == 1
    finally:
        host._hook_manager.release.set()
        host.finish_runs.set()
        with contextlib.suppress(asyncio.CancelledError):
            await first


@pytest.mark.asyncio
async def test_prompt_admission_post_hook_promotion_conflict_rejects_without_reroute() -> None:
    host = _Host(decision=HookDecision())
    host._fsm.state = EngineState.IDLE
    host._hook_manager = _BlockingPromptHookManager(HookDecision())
    errors = await _collect_errors(host._bus)

    async def _never_finishes() -> None:
        await asyncio.Event().wait()

    message_task = asyncio.create_task(on_user_message(host, UserMessage(text="conflict fresh")))
    conflict_task: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        conflict_task = asyncio.create_task(_never_finishes())
        host._turn_state.run_task = conflict_task
        host._hook_manager.release.set()
        await message_task

        assert host._turn_state.run_task is conflict_task
        assert host.run_texts == []
        assert [error.code for error in errors] == ["prompt_admission_conflict"]
        assert [payload["injected"] for payload in host._hook_manager.payloads] == [False]
    finally:
        host._hook_manager.release.set()
        host.finish_runs.set()
        if conflict_task is not None:
            conflict_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await conflict_task
        current_task = host._turn_state.run_task
        if current_task is not None and not current_task.done():
            current_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await current_task
        with contextlib.suppress(asyncio.CancelledError):
            await message_task


@pytest.mark.asyncio
async def test_concurrent_user_retry_uses_latest_pending_retry_note_after_blocked_hook() -> None:
    host = _Host(decision=HookDecision(), executor_running=True)
    host._hook_manager = _BlockingPromptHookManager(HookDecision(), blocked_calls=1)
    first_timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    second_timestamp = datetime(2026, 1, 2, tzinfo=UTC)

    first = asyncio.create_task(on_user_retry(host, UserRetry(text="first note", timestamp=first_timestamp)))
    await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)

    await on_user_retry(host, UserRetry(text="second note", timestamp=second_timestamp))
    assert host._turn_state.pending_retry.text == "second note"
    assert host._turn_state.pending_retry.created_at == second_timestamp

    host._hook_manager.release.set()
    await first

    assert [payload["text"] for payload in host._hook_manager.payloads] == ["first note", "second note"]
    assert host._turn_state.pending_retry.text == "second note"
    assert host._turn_state.pending_retry.created_at == second_timestamp
    assert host._turn_state.pending_retry.owner_admission_id == 2
    assert host._turn_state.pending_retry.updated_by_admission_id == 2


@pytest.mark.asyncio
async def test_user_retry_waits_behind_active_fresh_admission_before_prompt_hook() -> None:
    host = _Host(decision=HookDecision())
    host._fsm.state = EngineState.IDLE
    host._hook_manager = _BlockingPromptHookManager(HookDecision(), blocked_calls=1)

    fresh = asyncio.create_task(on_user_message(host, UserMessage(text="fresh prompt")))
    retry = None
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        retry = asyncio.create_task(on_user_retry(host, UserRetry(text="retry note")))
        await asyncio.sleep(0)

        assert [payload["text"] for payload in host._hook_manager.payloads] == ["fresh prompt"]

        host._hook_manager.release.set()
        await asyncio.wait_for(host._hook_manager.second_entered.wait(), timeout=5.0)

        assert [payload["text"] for payload in host._hook_manager.payloads] == ["fresh prompt", "retry note"]
        assert [payload["injected"] for payload in host._hook_manager.payloads] == [False, True]
        assert host.run_texts == ["fresh prompt"]
        assert retry.done() is False

        host.finish_runs.set()
        await asyncio.wait_for(asyncio.gather(fresh, retry), timeout=5.0)
        current_task = host._turn_state.run_task
        if current_task is not None:
            await asyncio.wait_for(current_task, timeout=5.0)

        assert host.retry_texts == ["retry note"]
    finally:
        host._hook_manager.release.set()
        host.finish_runs.set()
        with contextlib.suppress(asyncio.CancelledError):
            await fresh
        if retry is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await retry


@pytest.mark.asyncio
async def test_immediate_retry_reuses_existing_reminder_scope_without_allocating_unused_scope() -> None:
    host = _Host(decision=HookDecision())
    scope = host._turn_state.begin_current_run_scope(
        owner_admission_id=1,
        session_generation=host.session_generation,
        build_generation=host.build_generation,
        reminder_scope=host._reminder_middleware.create_current_run_scope(),
    )
    scopes_before = set(host._reminder_middleware._scopes)

    await on_user_retry(host, UserRetry(text="reuse scope"))
    assert host._turn_state.run_task is not None
    await host._turn_state.run_task

    assert host.retry_texts == ["reuse scope"]
    assert set(host._reminder_middleware._scopes) == scopes_before
    assert host._turn_state.current_run_scope is not None
    assert host._turn_state.current_run_scope.reminder_scope == scope.reminder_scope
    assert host._turn_state.current_run_scope.owner_admission_id == 1


@pytest.mark.asyncio
async def test_immediate_retry_stale_shutdown_state_does_not_install_task_after_hook() -> None:
    host = _Host(decision=HookDecision())
    host._hook_manager = _BlockingPromptHookManager(HookDecision())

    task = asyncio.create_task(on_user_retry(host, UserRetry(text="retry during shutdown")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        host._shutting_down = True
        host._fsm.state = EngineState.SHUTTING_DOWN
        host._hook_manager.release.set()
        await asyncio.wait_for(task, timeout=5.0)

        assert host._turn_state.run_task is None
        assert host.retry_texts == []
        assert host._fsm.transitions == []
        assert host._reminder_middleware.queued == []
    finally:
        host._hook_manager.release.set()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_active_retry_stale_shutdown_after_hook_does_not_queue_pending_retry_or_reminders() -> None:
    host = _Host(decision=HookDecision(), executor_running=True)
    host._hook_manager = _BlockingPromptHookManager(HookDecision(system_reminders=["do not queue"]))
    host._turn_state.clear_pending_retry(outcome=PreparationOutcome.DROPPED)
    host._turn_state.clear_pending_retry(outcome=PreparationOutcome.DROPPED)
    _install_active_run(host)

    task = asyncio.create_task(on_user_retry(host, UserRetry(text="retry during shutdown")))
    try:
        await asyncio.wait_for(host._hook_manager.entered.wait(), timeout=5.0)
        host._shutting_down = True
        host._fsm.state = EngineState.SHUTTING_DOWN
        host._hook_manager.release.set()
        await asyncio.wait_for(task, timeout=5.0)

        assert host._turn_state.pending_retry.text == ""
        assert host._turn_state.pending_retry.created_at is None
        assert host._turn_state.pending_retry == PendingRetry()
        assert host._reminder_middleware.queued == []
        assert host._fsm.transitions == []
        assert host.retry_texts == []
    finally:
        host._hook_manager.release.set()
        await _cancel_active_run(host)
        with contextlib.suppress(asyncio.CancelledError):
            await task
