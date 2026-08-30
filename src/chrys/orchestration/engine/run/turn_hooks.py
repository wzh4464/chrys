# Copyright (c) 2026 Chrys. All rights reserved.

"""Lifecycle hook dispatch helpers for main-agent turns."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Protocol

from chrys.foundation.events.types import Error
from chrys.foundation.i18n import msg
from chrys.foundation.platform import safe_getcwd
from chrys.service.hooks.events import HookEvent
from chrys.service.hooks.schema import HookDecision

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.models.workspace import Workspace
    from chrys.orchestration.engine.executor import Executor
    from chrys.orchestration.engine.state.machine import EngineStateMachine
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
    from chrys.service.hooks.manager import HookManager
    from chrys.service.profiles.agents.schema import AgentProfile


logger = logging.getLogger(__name__)

_TURN_HOOKS_PROMPT_BLOCKED = msg(
    "turn_hooks.prompt_blocked",
    fallback="Prompt blocked by hook.",
)


class TurnHookHost(Protocol):
    """Engine host contract needed for turn lifecycle hooks."""

    _bus: EventBus
    _session_id: str | None
    _hook_manager: HookManager | None
    _agent_profile: AgentProfile | None
    _workspace: Workspace | None
    _turn_number: int

    @property
    def _executor(self) -> Executor | None: ...

    _fsm: EngineStateMachine
    _reminder_middleware: SystemReminderMiddleware | None


class TurnHookDispatcher:
    """Dispatch non-prompt turn lifecycle hooks."""

    def __init__(self, host: TurnHookHost) -> None:
        self._host = host

    async def fire_before_turn(
        self,
        user_text: str,
        *,
        is_retry: bool = False,
        target_operation_id: str | None = None,
    ) -> None:
        """Publish a ``before_turn`` hook event."""
        host = self._host
        if host._hook_manager is None or not host._hook_manager.has_hooks_for(HookEvent.BEFORE_TURN):
            return
        profile_name = host._agent_profile.name if host._agent_profile is not None else ""
        await host._hook_manager.fire(
            HookEvent.BEFORE_TURN,
            {
                "session_id": host._session_id,
                "profile": profile_name,
                "cwd": _workspace_cwd(host),
                "turn": host._turn_number,
                "user_text": user_text,
                "is_retry": is_retry,
            },
            target_operation_id=target_operation_id,
        )

    async def fire_after_turn(self, *, failed: bool) -> None:
        """Publish an ``after_turn`` hook event."""
        host = self._host
        if host._hook_manager is None or not host._hook_manager.has_hooks_for(HookEvent.AFTER_TURN):
            return
        profile_name = host._agent_profile.name if host._agent_profile is not None else ""
        status = (
            "failed"
            if (host._executor and host._executor.run_failed)
            else ("interrupted" if (host._executor and host._executor.was_interrupted) else "ok")
        )
        await host._hook_manager.fire(
            HookEvent.AFTER_TURN,
            {
                "session_id": host._session_id,
                "profile": profile_name,
                "cwd": _workspace_cwd(host),
                "turn": host._turn_number,
                "status": status,
                "failed": failed,
            },
        )

    def schedule_user_interrupt(self) -> None:
        """Schedule observer dispatch for a ``user_interrupt`` hook."""
        host = self._host
        if host._hook_manager is None or not host._hook_manager.has_hooks_for(HookEvent.USER_INTERRUPT):
            return
        task = asyncio.create_task(self.fire_user_interrupt())
        task.add_done_callback(self.log_user_interrupt_hook_error)

    async def fire_user_interrupt(self) -> None:
        """Publish a ``user_interrupt`` hook event."""
        host = self._host
        manager = host._hook_manager
        if manager is None:
            return
        profile_name = host._agent_profile.name if host._agent_profile is not None else ""
        await manager.fire(
            HookEvent.USER_INTERRUPT,
            {
                "session_id": host._session_id,
                "profile": profile_name,
                "cwd": _workspace_cwd(host),
            },
            scope="detached",
        )

    @staticmethod
    def log_user_interrupt_hook_error(task: asyncio.Task[None]) -> None:
        """Log a failed background interrupt-hook dispatch."""
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("user_interrupt hook dispatch failed")


class PromptSubmitGate:
    """Evaluate ``user_prompt_submit`` hooks and apply accepted reminders."""

    def __init__(self, host: TurnHookHost) -> None:
        self._host = host

    async def fire(self, text: str, *, injected: bool | None = None) -> bool:
        """Return True when a blocking hook denied the prompt."""
        decision = await self.evaluate(text, injected=injected)
        if await self.handle_decision(decision, injected=injected):
            return True
        self.queue_reminders(decision, injected=injected)
        return False

    async def evaluate(
        self,
        text: str,
        *,
        injected: bool | None,
        target_operation_id: str | None = None,
    ) -> HookDecision | None:
        """Run ``user_prompt_submit`` hooks without applying reminder side effects."""
        host = self._host
        if host._hook_manager is None or not host._hook_manager.has_hooks_for(HookEvent.USER_PROMPT_SUBMIT):
            return None
        profile_name = host._agent_profile.name if host._agent_profile is not None else ""
        is_injected = host._fsm.is_running() if injected is None else injected
        return await host._hook_manager.fire(
            HookEvent.USER_PROMPT_SUBMIT,
            {
                "session_id": host._session_id,
                "profile": profile_name,
                "cwd": _workspace_cwd(host),
                "text": text,
                "injected": is_injected,
            },
            target_operation_id=target_operation_id,
        )

    async def handle_decision(
        self,
        decision: HookDecision | None,
        *,
        injected: bool | None,
        session_id: str | None = None,
    ) -> bool:
        """Return True after publishing when a prompt-submit hook blocked."""
        _ = injected
        if decision is None:
            return False
        if not decision.blocked:
            return False
        host = self._host
        await host._bus.publish(
            Error(
                code="hook_blocked",
                message=decision.block_reason or "Prompt blocked by hook.",
                display_message=None if decision.block_reason else _TURN_HOOKS_PROMPT_BLOCKED.bind(),
                session_id=host._session_id if session_id is None else session_id,
            )
        )
        return True

    def queue_reminders(self, decision: HookDecision | None, *, injected: bool | None) -> None:
        """Apply non-blocking prompt-submit hook reminders after prompt validation passes."""
        host = self._host
        if decision is None or not decision.system_reminders or host._reminder_middleware is None:
            return
        is_injected = host._fsm.is_running() if injected is None else injected
        host._reminder_middleware.queue_hook_reminders(
            decision.system_reminders,
            for_next_turn=not is_injected,
        )

    @staticmethod
    def reminder_texts(decision: HookDecision | None) -> list[str]:
        """Return non-empty prompt-hook reminders from an accepted decision."""
        if decision is None:
            return []
        return [reminder for reminder in decision.system_reminders if reminder]


def _workspace_cwd(host: TurnHookHost) -> str:
    """Return the current workspace cwd, falling back only before workspace initialization."""
    if host._workspace is not None:
        return host._workspace.primary_cwd
    return safe_getcwd()
