# Copyright (c) 2026 Chrys. All rights reserved.

"""Bridge inline tool widget actions to backend events."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from chrys.app.tui.screens.main.ports import EventPublisher
from chrys.foundation.events.types import (
    ApprovalResponse,
    AskUserResponse,
    SleepSkip,
    SubAgentAbortRequested,
    SubAgentRetryRequested,
)


class ToolActionBridge:
    """Publish backend events for inline tool action controls."""

    def __init__(self, *, publisher: EventPublisher, debug: Callable[[str, str], None]) -> None:
        self._publisher = publisher
        self._debug = debug

    async def request_sub_agent_retry(self, invocation_id: str) -> None:
        """Publish a sub-agent retry request."""
        await self._publisher.publish(SubAgentRetryRequested(invocation_id=invocation_id))
        self._log(f"SubAgentRetryRequested inv={invocation_id}")

    async def request_sub_agent_abort(self, invocation_id: str) -> None:
        """Publish a sub-agent abort request."""
        await self._publisher.publish(SubAgentAbortRequested(invocation_id=invocation_id))
        self._log(f"SubAgentAbortRequested inv={invocation_id}")

    async def skip_sleep(self, call_id: str) -> None:
        """Publish a sleep-skip request."""
        await self._publisher.publish(SleepSkip(call_id=call_id))
        self._log(f"SleepSkip call={call_id}")

    async def submit_ask_user_inline(self, request_id: str, text: str) -> None:
        """Publish an inline ask-user response."""
        await self._publisher.publish(AskUserResponse(request_id=request_id, text=text))
        self._log("AskUserResponse", text[:60])

    async def publish_approval_response(
        self,
        request_id: str,
        approved: bool,
        reason: str = "",
        modified_args: dict[str, Any] | None = None,
    ) -> None:
        """Publish an approval response."""
        await self._publisher.publish(
            ApprovalResponse(
                request_id=request_id,
                approved=approved,
                reason=reason,
                modified_args=modified_args,
            )
        )
        self._log("ApprovalResponse", "approved" if approved else "declined")

    async def publish_ask_user_response(self, request_id: str, text: str) -> None:
        """Publish a modal ask-user response."""
        await self._publisher.publish(AskUserResponse(request_id=request_id, text=text))
        self._log("AskUserResponse", text[:60])

    def _log(self, event_type: str, detail: str = "") -> None:
        self._debug(event_type, detail)
