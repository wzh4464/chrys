# Copyright (c) 2026 Chrys. All rights reserved.

"""Decide for paused sub-agents when no one is there to click Retry or Abort.

A sub-agent whose ACP transport fails pauses and waits for a decision from
the person at the screen. A headless run has no such person: on the DeepSWE
benchmark a PACT campaign sat paused for hours after one malformed
``session/update`` frame, the parent's tool call never returned, and the
task was lost. This policy retries a bounded number of times and then
aborts, so the parent agent sees a tool error instead of silence.
"""

from __future__ import annotations

import logging

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import SubAgentAbortRequested, SubAgentPaused, SubAgentRetryRequested

logger = logging.getLogger(__name__)

HEADLESS_SUB_AGENT_RETRIES = 2


async def install_headless_sub_agent_policy(
    bus: EventBus,
    session_id: str,
    *,
    retries: int = HEADLESS_SUB_AGENT_RETRIES,
) -> None:
    """Answer every ``SubAgentPaused`` with Retry, ``retries`` times per invocation, then Abort."""
    subscribe = getattr(bus, "subscribe", None)
    if subscribe is None:
        # A publish-only bus stand-in (the CLI tests' host double) has nothing to pause.
        return
    attempts: dict[str, int] = {}

    async def _decide(event: SubAgentPaused) -> None:
        used = attempts.get(event.invocation_id, 0)
        if used < retries:
            attempts[event.invocation_id] = used + 1
            logger.warning(
                "Headless run: sub-agent %s paused (%s: %s); retrying (%d/%d)",
                event.invocation_id,
                event.reason,
                event.last_error,
                used + 1,
                retries,
            )
            await bus.publish(SubAgentRetryRequested(invocation_id=event.invocation_id, session_id=session_id))
            return
        logger.error(
            "Headless run: sub-agent %s paused again (%s: %s) after %d retries; aborting it",
            event.invocation_id,
            event.reason,
            event.last_error,
            retries,
        )
        await bus.publish(SubAgentAbortRequested(invocation_id=event.invocation_id, session_id=session_id))

    await subscribe(SubAgentPaused, _decide)
