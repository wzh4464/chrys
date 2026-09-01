# Copyright (c) 2026 Chrys. All rights reserved.

"""Hermetic stdio host for exercising the production Chrys-PACT ACP shell."""

from __future__ import annotations

import asyncio
import os

import acp
from acp.helpers import start_tool_call, update_agent_message_text

from chrys.pact.campaign import CampaignCancelled, CampaignTerminal, UpdateSender
from chrys.pact.server import ChrysPactServer

_SCENARIO_ENV = "CHRYS_PACT_STUB_SCENARIO"


class _StubCoordinator:
    """Replace only the model-bearing Campaign body, not the ACP server."""

    def __init__(self, scenario: str) -> None:
        self._scenario = scenario
        self._cancelled = asyncio.Event()

    async def run(
        self,
        *,
        workspace,
        contract_file,
        plan_file,
        send_update: UpdateSender,
    ) -> CampaignTerminal:
        _ = workspace, contract_file, plan_file
        await send_update(
            start_tool_call(
                "campaign-stub/reviewer-role",
                "PACT Reviewer turn",
                kind="think",
                status="in_progress",
            )
        )
        if self._scenario == "cancel":
            await self._cancelled.wait()
            raise CampaignCancelled("Invocation cancelled; canonical PACT artifacts were preserved.")

        await send_update(update_agent_message_text("reviewer prose that is not the Campaign result"))
        return CampaignTerminal(
            status="completed",
            campaign_id="campaign-stub",
            revision=7,
            next_action="none",
            artifact_ref=".pact/runtime/campaigns/campaign-stub",
        )

    async def cancel(self) -> None:
        self._cancelled.set()

    async def wait_closed(self) -> None:
        return


async def _main() -> None:
    scenario = os.environ.get(_SCENARIO_ENV, "normal")
    server = ChrysPactServer(lambda: _StubCoordinator(scenario))
    try:
        await acp.run_agent(server, use_unstable_protocol=True)
    finally:
        await server.shutdown()


if __name__ == "__main__":
    asyncio.run(_main())
