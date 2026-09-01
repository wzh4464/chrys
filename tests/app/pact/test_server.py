# Copyright (c) 2026 Chrys. All rights reserved.

"""Strict launch-contract and lifecycle tests for the Chrys-PACT ACP shell."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from acp import schema as acp_schema
from acp.exceptions import RequestError
from acp.helpers import update_agent_message_text

from chrys.orchestration.sub_agents.acp_controller import AcpUpdateTranslator
from chrys.pact.campaign import CampaignTerminal, UpdateSender
from chrys.pact.server import ChrysPactServer, LaunchContractError, parse_launch_request


def _write_launch_files(workspace: Path, request_id: str = "request-1") -> tuple[str, str]:
    request_dir = workspace / ".pact-io" / "chrys-pact" / request_id
    request_dir.mkdir(parents=True)
    contract = request_dir / "goal-contract.json"
    plan = request_dir / "initial-plan.json"
    contract.write_text("{}", encoding="utf-8")
    plan.write_text("{}", encoding="utf-8")
    return (
        contract.relative_to(workspace).as_posix(),
        plan.relative_to(workspace).as_posix(),
    )


def _prompt(contract_path: str, plan_path: str, **extra: object) -> list[Any]:
    payload = {
        "schema": "chrys-pact/run-request/v1",
        "contract_path": contract_path,
        "plan_path": plan_path,
        **extra,
    }
    return [acp_schema.TextContentBlock(type="text", text=json.dumps(payload))]


def test_parse_launch_request_resolves_two_contained_regular_files(tmp_path: Path) -> None:
    contract, plan = _write_launch_files(tmp_path)

    result = parse_launch_request(_prompt(contract, plan), workspace=tmp_path)

    assert result.contract_file == (tmp_path / contract).resolve()
    assert result.plan_file == (tmp_path / plan).resolve()


@pytest.mark.parametrize(
    ("prompt", "message"),
    [
        ([], "exactly one text block"),
        ([acp_schema.TextContentBlock(type="text", text="[]")], "JSON object"),
        ([acp_schema.TextContentBlock(type="text", text="not-json")], "valid JSON object"),
        (
            [
                acp_schema.TextContentBlock(
                    type="text",
                    text=json.dumps(
                        {
                            "schema": "chrys-pact/run-request/v2",
                            "contract_path": "x",
                            "plan_path": "y",
                        }
                    ),
                )
            ],
            "run-request/v1",
        ),
    ],
)
def test_parse_launch_request_rejects_non_contract_inputs(tmp_path: Path, prompt: list[Any], message: str) -> None:
    with pytest.raises(LaunchContractError, match=message):
        parse_launch_request(prompt, workspace=tmp_path)


def test_parse_launch_request_rejects_unknown_fields(tmp_path: Path) -> None:
    contract, plan = _write_launch_files(tmp_path)

    with pytest.raises(LaunchContractError, match="unknown fields: model"):
        parse_launch_request(_prompt(contract, plan, model="other"), workspace=tmp_path)


def test_parse_launch_request_rejects_files_from_different_request_directories(tmp_path: Path) -> None:
    contract, _ = _write_launch_files(tmp_path, "one")
    _, plan = _write_launch_files(tmp_path, "two")

    with pytest.raises(LaunchContractError, match="same request directory"):
        parse_launch_request(_prompt(contract, plan), workspace=tmp_path)


def test_parse_launch_request_rejects_symlink_escape(tmp_path: Path) -> None:
    contract, plan = _write_launch_files(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text("{}", encoding="utf-8")
    contract_file = tmp_path / contract
    contract_file.unlink()
    try:
        contract_file.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this filesystem")

    with pytest.raises(LaunchContractError, match="outside the session workspace"):
        parse_launch_request(_prompt(contract, plan), workspace=tmp_path)


class _FakeClient:
    def __init__(self) -> None:
        self.updates: list[Any] = []

    async def session_update(self, *, session_id: str, update: Any) -> None:
        self.updates.append((session_id, update))


class _FakeCoordinator:
    def __init__(self, *, status: str = "completed", emit_role_text: bool = False) -> None:
        self.runs = 0
        self.cancelled = 0
        self.status = status
        self.emit_role_text = emit_role_text

    async def run(
        self,
        *,
        workspace: Path,
        contract_file: Path,
        plan_file: Path,
        send_update: UpdateSender,
    ) -> CampaignTerminal:
        _ = workspace, contract_file, plan_file, send_update
        self.runs += 1
        if self.emit_role_text:
            await send_update(update_agent_message_text("reviewer prose that must not leak into the result"))
        return CampaignTerminal(
            status=self.status,
            campaign_id="campaign-test",
            revision=2,
            next_action="none",
            artifact_ref=".pact/runtime/campaigns/campaign-test",
        )

    async def cancel(self) -> None:
        self.cancelled += 1

    async def wait_closed(self) -> None:
        return


async def test_server_accepts_one_session_and_one_prompt_and_sends_summary_last(tmp_path: Path) -> None:
    contract, plan = _write_launch_files(tmp_path)
    coordinator = _FakeCoordinator()
    client = _FakeClient()
    server = ChrysPactServer(lambda: coordinator)
    server.on_connect(client)  # type: ignore[arg-type]
    session = await server.new_session(cwd=str(tmp_path))

    response = await server.prompt(_prompt(contract, plan), session.session_id, message_id="message-1")

    assert response.stop_reason == "end_turn"
    assert response.user_message_id == "message-1"
    assert coordinator.runs == 1
    assert client.updates[-1][1].content.text.startswith("PACT Campaign result\nstatus: completed")
    with pytest.raises(RequestError):
        await server.prompt(_prompt(contract, plan), session.session_id)
    with pytest.raises(RequestError):
        await server.new_session(cwd=str(tmp_path))


async def test_blocked_campaign_returns_normal_transport_with_incomplete_summary(tmp_path: Path) -> None:
    contract, plan = _write_launch_files(tmp_path)
    coordinator = _FakeCoordinator(status="blocked")
    client = _FakeClient()
    server = ChrysPactServer(lambda: coordinator)
    server.on_connect(client)  # type: ignore[arg-type]
    session = await server.new_session(cwd=str(tmp_path))

    response = await server.prompt(_prompt(contract, plan), session.session_id)

    assert response.stop_reason == "end_turn"
    assert client.updates[-1][1].content.text.startswith("PACT Campaign result\nstatus: blocked")


async def test_stock_primary_last_segment_is_exactly_campaign_summary(tmp_path: Path) -> None:
    contract, plan = _write_launch_files(tmp_path)
    coordinator = _FakeCoordinator(emit_role_text=True)
    client = _FakeClient()
    server = ChrysPactServer(lambda: coordinator)
    server.on_connect(client)  # type: ignore[arg-type]
    session = await server.new_session(cwd=str(tmp_path))
    translator = AcpUpdateTranslator(
        event_bus=None,
        session_id=session.session_id,
        agent_name="ChrysPact",
        invocation_id="invocation-test",
        attempt=1,
        result_mode="last_segment",
    )

    await server.prompt(_prompt(contract, plan), session.session_id)
    for sequence, (_session_id, update) in enumerate(client.updates, start=1):
        await translator.on_update(
            sequence,
            acp_schema.SessionNotification(sessionId=session.session_id, update=update),
        )

    assert (
        translator.result_text()
        == CampaignTerminal(
            status="completed",
            campaign_id="campaign-test",
            revision=2,
            next_action="none",
            artifact_ref=".pact/runtime/campaigns/campaign-test",
        ).summary_text()
    )


async def test_invalid_prompt_refuses_without_starting_campaign(tmp_path: Path) -> None:
    coordinator = _FakeCoordinator()
    client = _FakeClient()
    server = ChrysPactServer(lambda: coordinator)
    server.on_connect(client)  # type: ignore[arg-type]
    session = await server.new_session(cwd=str(tmp_path))

    response = await server.prompt([acp_schema.TextContentBlock(type="text", text="{}")], session.session_id)

    assert response.stop_reason == "refusal"
    assert coordinator.runs == 0
    assert client.updates[-1][1].content.text.startswith("PACT launch refused:")


async def test_cancel_and_close_delegate_to_single_coordinator(tmp_path: Path) -> None:
    coordinator = _FakeCoordinator()
    server = ChrysPactServer(lambda: coordinator)
    session = await server.new_session(cwd=str(tmp_path))

    await server.cancel(session.session_id)
    await server.close_session(session.session_id)

    assert coordinator.cancelled == 2
