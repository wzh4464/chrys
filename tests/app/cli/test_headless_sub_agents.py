# Copyright (c) 2026 Chrys. All rights reserved.

from __future__ import annotations

import pytest

from chrys.app.cli.headless_sub_agents import install_headless_sub_agent_policy
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import SubAgentAbortRequested, SubAgentPaused, SubAgentRetryRequested


@pytest.mark.asyncio
async def test_a_paused_sub_agent_is_retried_then_aborted_without_a_person() -> None:
    bus = EventBus()
    decisions: list[str] = []

    async def _retry(event: SubAgentRetryRequested) -> None:
        decisions.append(f"retry:{event.invocation_id}")

    async def _abort(event: SubAgentAbortRequested) -> None:
        decisions.append(f"abort:{event.invocation_id}")

    await bus.subscribe(SubAgentRetryRequested, _retry)
    await bus.subscribe(SubAgentAbortRequested, _abort)
    await install_headless_sub_agent_policy(bus, "s1", retries=2)

    for _ in range(3):
        await bus.publish(SubAgentPaused(invocation_id="inv1", reason="acp_transport", last_error="x", session_id="s1"))
    await bus.publish(SubAgentPaused(invocation_id="inv2", reason="acp_transport", session_id="s1"))

    assert decisions == ["retry:inv1", "retry:inv1", "abort:inv1", "retry:inv2"]
