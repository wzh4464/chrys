# Copyright (c) 2026 Chrys. All rights reserved.

"""Hermetic stdio E2E coverage for the external Chrys-PACT agent."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import TextIO

import pytest

from chrys.orchestration.sub_agents.acp_controller import AcpUpdateTranslator
from chrys.pact.campaign import CampaignTerminal
from chrys.service.acp_client import AcpAgentClient, AcpAgentSpec
from tests.service.acp_client.helpers import CallbackRecorder, residual_client_tasks
from tests.support.paths import REPO_ROOT, SRC_ROOT
from tests.support.waiting import wait_for

_STUB_AGENT = (REPO_ROOT / "tests" / "support" / "chrys_pact_stub_agent.py").resolve()
_HANDSHAKE_TIMEOUT_SECONDS = 45


def _write_launch_files(workspace: Path) -> str:
    request_dir = workspace / ".pact-io" / "chrys-pact" / "request-e2e"
    request_dir.mkdir(parents=True)
    contract = request_dir / "goal-contract.json"
    plan = request_dir / "initial-plan.json"
    contract.write_text("{}", encoding="utf-8")
    plan.write_text("{}", encoding="utf-8")
    return json.dumps(
        {
            "schema": "chrys-pact/run-request/v1",
            "contract_path": contract.relative_to(workspace).as_posix(),
            "plan_path": plan.relative_to(workspace).as_posix(),
        }
    )


def _spec(tmp_path: Path, workspace: Path, *, scenario: str) -> AcpAgentSpec:
    return AcpAgentSpec(
        command=sys.executable,
        args=(str(_STUB_AGENT),),
        env={
            "PYTHONPATH": str(SRC_ROOT),
            "CHRYS_PACT_STUB_SCENARIO": scenario,
        },
        cwd=str(workspace.resolve()),
        stderr_log_path=(tmp_path / f"chrys-pact-{scenario}.stderr.log").resolve(),
        handshake_timeout_seconds=20,
        idle_timeout_seconds=0,
    )


async def _assert_client_reaped(
    client: AcpAgentClient,
    *,
    tasks_before: set[asyncio.Task[object]],
) -> None:
    spawn = client._spawn
    await client.aclose()
    assert spawn is not None and spawn.process.returncode is not None
    await wait_for(
        lambda: not residual_client_tasks(tasks_before),
        description="Chrys-PACT ACP client task reaping",
    )


async def test_stdio_prompt_returns_exact_stock_primary_last_segment_and_reaps_child(tmp_path: Path) -> None:
    tasks_before = asyncio.all_tasks()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prompt = _write_launch_files(workspace)
    callbacks = CallbackRecorder()
    translator = AcpUpdateTranslator(
        event_bus=None,
        session_id=None,
        agent_name="ChrysPact",
        invocation_id="invocation-e2e",
        attempt=1,
        result_mode="last_segment",
    )
    client = AcpAgentClient(
        _spec(tmp_path, workspace, scenario="normal"),
        callbacks,
        update_sink=translator,
    )
    try:
        await client.connect()
        handshake = await client.open_session()
        outcome = await client.prompt(prompt)

        assert handshake.agent_info is not None and handshake.agent_info.name == "chrys-pact"
        assert outcome.stop_reason == "end_turn"
        assert (
            translator.result_text()
            == CampaignTerminal(
                status="completed",
                campaign_id="campaign-stub",
                revision=7,
                next_action="none",
                artifact_ref=".pact/runtime/campaigns/campaign-stub",
            ).summary_text()
        )
        with pytest.raises(RuntimeError, match="already been started"):
            await client.prompt(prompt)
    finally:
        await _assert_client_reaped(client, tasks_before=tasks_before)


async def test_stdio_cancel_returns_cancelled_and_reaps_child(tmp_path: Path) -> None:
    tasks_before = asyncio.all_tasks()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    prompt = _write_launch_files(workspace)
    callbacks = CallbackRecorder()
    client = AcpAgentClient(_spec(tmp_path, workspace, scenario="cancel"), callbacks)
    prompt_task: asyncio.Task[object] | None = None
    try:
        await client.connect()
        await client.open_session()
        prompt_task = asyncio.create_task(client.prompt(prompt))
        await wait_for(lambda: bool(callbacks.updates), description="Chrys-PACT cancel-ready stage update")

        await client.cancel()
        outcome = await asyncio.wait_for(prompt_task, timeout=5)

        assert outcome.stop_reason == "cancelled"
        assert callbacks.updates[-1][1].update.content.text.startswith("Invocation cancelled")
    finally:
        if prompt_task is not None and not prompt_task.done():
            prompt_task.cancel()
            await asyncio.gather(prompt_task, return_exceptions=True)
        await _assert_client_reaped(client, tasks_before=tasks_before)


def _drain_lines(stream: TextIO, sink: queue.Queue[str]) -> None:
    try:
        for line in stream:
            sink.put(line)
    finally:
        sink.put("")


def test_top_level_cli_dispatch_initialize_keeps_stdout_json_rpc_only(tmp_path: Path) -> None:
    isolated_home = tmp_path / "home"
    env = os.environ.copy()
    env["HOME"] = os.fspath(isolated_home)
    env["USERPROFILE"] = os.fspath(isolated_home)
    env["APPDATA"] = os.fspath(tmp_path / "appdata")
    process = subprocess.Popen(
        [sys.executable, "-m", "chrys.app.cli.app", "pact-agent", "--allow-unverified"],
        cwd=REPO_ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stdout_lines: queue.Queue[str] = queue.Queue()
    stdout_reader = threading.Thread(target=_drain_lines, args=(process.stdout, stdout_lines), daemon=True)
    stderr_chunks: list[str] = []
    stderr_reader = threading.Thread(target=lambda: stderr_chunks.append(process.stderr.read()), daemon=True)
    stdout_reader.start()
    stderr_reader.start()
    try:
        process.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": 1,
                        "clientCapabilities": {},
                        "clientInfo": {"name": "pytest", "version": "0"},
                    },
                }
            )
            + "\n"
        )
        process.stdin.flush()
        try:
            line = stdout_lines.get(timeout=_HANDSHAKE_TIMEOUT_SECONDS)
        except queue.Empty:
            raise AssertionError("chrys-pact did not return an initialize response") from None
        assert line, (
            "chrys-pact closed stdout before initialize completed; "
            f"returncode={process.poll()}, stderr={''.join(stderr_chunks)[-1_000:]}"
        )
        response = json.loads(line)
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response["result"]["agentInfo"]["name"] == "chrys-pact"

        process.stdin.close()
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        stdout_reader.join(timeout=5)
        stderr_reader.join(timeout=5)

    assert process.returncode == 0
    assert [line for line in iter(stdout_lines.get_nowait, "") if line.strip()] == []
