# Copyright (c) 2026 Chrys. All rights reserved.

"""Engine-driven requirement-enrichment production-boundary tests."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest

import chrys.orchestration.engine.build.builder as builder_module
import chrys.orchestration.engine.run.requirement_enrichment as workflow_module
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import UserInterrupt, UserMessage
from chrys.foundation.models.workspace import Workspace
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import (
    AgentProfile,
    ApprovalConfig,
    CompactionConfig,
    RequirementEnrichmentConfig,
    ToolsConfig,
)
from chrys.service.requirement_clarification.types import ClarificationResult, ClarificationSelection
from chrys.service.semantic_search import LocalizationArtifact, LocalizationResult
from chrys.service.state.store import JsonFileStateStore
from tests.support.pipeline_helpers import make_mock_settings_and_registry
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for


def _profile() -> AgentProfile:
    return AgentProfile(
        name="Code",
        display_name="Code Agent",
        instructions="You are a coding assistant.",
        tools=ToolsConfig(builtins=[]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
        requirement_enrichment=RequirementEnrichmentConfig(
            enabled=True,
            localization_mode="fallback",
        ),
    )


def _localization(tmp_path: Path) -> LocalizationResult:
    return LocalizationResult(
        payload={"locations": [{"file_path": "src/owner.py", "role": "primary"}]},
        artifacts=LocalizationArtifact(
            result_json=tmp_path / "result.json",
            report_markdown=tmp_path / "report.md",
            index_json=tmp_path / "index.json",
            graph_json=tmp_path / "graph.json",
            trace_jsonl=tmp_path / "trace.jsonl",
            manifest_json=tmp_path / "manifest.json",
        ),
    )


def _clarification() -> ClarificationResult:
    return ClarificationResult(
        strategy_version="test-v1",
        revision=1,
        delta="Repository implementation guidance:\n- preserve compatibility",
        selection=ClarificationSelection(),
    )


def _engine_setup(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch, client: MockChatClient):
    profile = _profile()
    registry = AgentProfileRegistry()
    registry.register(profile)
    settings, model_registry = make_mock_settings_and_registry(stream=False)
    settings = replace(settings, workspace_change_notice=False)
    monkeypatch.setattr(builder_module, "create_client", lambda *_args, **_kwargs: client)
    engine = agent_engine(
        EventBus(),
        settings=settings,
        agent_registry=registry,
        model_registry=model_registry,
        state_store=JsonFileStateStore(tmp_path / "state"),
        initial_workspace=Workspace.from_cwd(str(tmp_path / "workspace")),
    )
    return engine, profile


@pytest.mark.asyncio
async def test_engine_enrichment_injects_current_turn_wire_only(
    tmp_path: Path,
    agent_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "owner.py").write_text("def owner():\n    pass\n", encoding="utf-8")
    client = MockChatClient(responses=[MockResponse(text="implemented")])
    engine, profile = _engine_setup(tmp_path, agent_engine, monkeypatch, client)

    async def _clarify(self, **_kwargs):
        return _clarification()

    async def _localize(self, **_kwargs):
        return _localization(tmp_path)

    monkeypatch.setattr(workflow_module.RequirementEnrichmentWorkflow, "_clarify", _clarify)
    monkeypatch.setattr(workflow_module.RequirementEnrichmentWorkflow, "_localize", _localize)
    await engine.start(profile)

    await engine.event_bus.publish(UserMessage(text="implement owner"))
    await engine.wait_for_run_task()

    assert client.call_count == 1
    wire_messages, _options = client.call_history[0]
    assert "[REQUIREMENT_ENRICHMENT]" in wire_messages[-1].text
    assert engine._executor is not None
    persisted_users = [message.text for message in engine._executor.history_state["messages"] if message.role == "user"]
    assert persisted_users == ["implement owner"]


@pytest.mark.asyncio
async def test_engine_interrupt_during_enrichment_never_calls_main_model(
    tmp_path: Path,
    agent_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = MockChatClient(responses=[MockResponse(text="must not run")])
    engine, profile = _engine_setup(tmp_path, agent_engine, monkeypatch, client)
    both_started = asyncio.Event()
    started: set[str] = set()

    async def _wait_for_interrupt(name: str):
        started.add(name)
        if len(started) == 2:
            both_started.set()
        await asyncio.Event().wait()

    async def _clarify(self, **_kwargs):
        await _wait_for_interrupt("clarification")

    async def _localize(self, **_kwargs):
        await _wait_for_interrupt("localization")

    monkeypatch.setattr(workflow_module.RequirementEnrichmentWorkflow, "_clarify", _clarify)
    monkeypatch.setattr(workflow_module.RequirementEnrichmentWorkflow, "_localize", _localize)
    await engine.start(profile)

    await engine.event_bus.publish(UserMessage(text="implement owner"))
    await wait_for(
        both_started.is_set,
        timeout=ENGINE_TURN_TIMEOUT,
        description="both requirement enrichment side calls started",
    )
    await engine.event_bus.publish(UserInterrupt())
    await engine.wait_for_run_task()

    assert client.call_count == 0
