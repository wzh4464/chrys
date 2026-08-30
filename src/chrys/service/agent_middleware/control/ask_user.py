# Copyright (c) 2026 Chrys. All rights reserved.

"""Ask-user middleware — intercepts ask_user tool calls to show a dialog."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import uuid4

from chrys.foundation.events.types import AskUserResponse, AskUserTimedOut, QuestionToUser
from chrys.foundation.tool_kinds import KIND_ASK_USER, get_tool_kind
from chrys.foundation.trajectory.event_types import WaitCategory
from chrys.kernel.middleware import FunctionMiddleware
from chrys.service.agent_middleware._metadata_keys import _SHORT_ID_LEN
from chrys.service.agent_middleware.events.hook_dispatch import get_call_id
from chrys.service.tools.result_metadata import tool_error
from chrys.service.trajectory.tools import tool_operation_id
from chrys.service.trajectory.waits import WaitOutcome, WaitTrace

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from chrys.foundation.events.bus import EventBus
    from chrys.kernel.middleware import FunctionInvocationContext


_DEFAULT_ASK_USER_TIMEOUT_SECONDS = 600


def _format_timeout(seconds: int) -> str:
    """Render a timeout as whole minutes when divisible, else seconds."""
    if seconds % 60 == 0:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return f"{seconds} second{'s' if seconds != 1 else ''}"


class AskUserMiddleware(FunctionMiddleware):
    """Intercepts ``ask_user`` tool calls to show a dialog and wait for a response.

    Follows the same pattern as ``ApprovalMiddleware``: publishes a
    ``QuestionToUser`` event, awaits the ``AskUserResponse``, and writes
    the user's answer into ``context.result`` — the underlying tool function
    is never invoked.

    ``timeout_seconds`` bounds how long to wait for the reply.  ``None`` waits
    indefinitely (no ``AskUserTimedOut``) — used by frontends that own the
    interaction lifetime themselves (e.g. ACP clients).
    """

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str | None = None,
        caller_name: str = "",
        *,
        timeout_seconds: int | None = _DEFAULT_ASK_USER_TIMEOUT_SECONDS,
    ) -> None:
        self._bus = event_bus
        self._session_id = session_id
        self._caller_name = caller_name
        self._timeout_seconds = timeout_seconds

    async def process(
        self,
        context: FunctionInvocationContext,
        call_next: Callable[[], Awaitable[None]],
    ) -> None:
        if get_tool_kind(context.function) != KIND_ASK_USER:
            await call_next()
            return

        # Extract question and options from tool arguments
        args = context.arguments if isinstance(context.arguments, dict) else {}
        question = args.get("question", "")
        options = args.get("options") or []

        request_id = uuid4().hex[:_SHORT_ID_LEN]

        # Subscribe before publishing (same pattern as ApprovalMiddleware)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()

        async def _handler(event: AskUserResponse) -> None:
            if event.request_id == request_id and not future.done():
                future.set_result(event.text)

        await self._bus.subscribe(AskUserResponse, _handler)

        await self._bus.publish(
            QuestionToUser(
                request_id=request_id,
                question=question,
                options=options,
                session_id=self._session_id,
                call_id=get_call_id(context),
                caller_name=self._caller_name,
            )
        )

        timeout = self._timeout_seconds
        wait = WaitTrace.open(WaitCategory.USER_INPUT, target_operation_id=tool_operation_id(context.metadata))
        try:
            # Inside the block that closes the wait and drops the subscription:
            # the start marker awaits its write ack, and an interrupt landing
            # there is exactly the case the cancellation branch below exists for.
            if wait is not None:
                await wait.started()
            response = await (future if timeout is None else asyncio.wait_for(future, timeout=timeout))
            context.result = f"User response: {response}"
            if wait is not None:
                await wait.finished()
        except TimeoutError:
            if wait is not None:
                await wait.finished(outcome=WaitOutcome.TIMED_OUT)
            context.result = tool_error(
                "ask_user_timeout",
                f"user did not respond within {_format_timeout(timeout or 0)}.",
                retryable=True,
                details={"timeout_seconds": timeout or 0},
            )
            await self._bus.publish(AskUserTimedOut(request_id=request_id, session_id=self._session_id))
        except asyncio.CancelledError:
            if wait is not None:
                wait.finished_soon(outcome=WaitOutcome.CANCELLED)
            raise
        finally:
            await self._bus.unsubscribe(AskUserResponse, _handler)
