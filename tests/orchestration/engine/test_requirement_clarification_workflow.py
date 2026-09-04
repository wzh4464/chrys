# Copyright (c) 2026 Chrys. All rights reserved.

"""Route-A ordering, fresh history, and P0 fallback tests."""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import chrys.orchestration.engine.run.requirement_clarification as workflow_module
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import AgentMessage, RequirementClarificationPhaseChanged, Warning
from chrys.foundation.models.workspace import Workspace
from chrys.kernel import Message
from chrys.orchestration.engine.run.requirement_clarification import (
    RequirementClarificationWorkflow,
    _history_background,
)
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot
from chrys.service.requirement_clarification.types import (
    ClarificationResult,
    ClarificationSelection,
    RequirementWorkflowPhase,
)
from chrys.service.session.history import SessionHistoryManager


@dataclass
class _HistoryCheckpoint:
    state: dict[str, Any]

    @property
    def messages(self):
        return self.state["messages"]


class _Executor:
    def __init__(self, workspace_file: Path, *, repair_fails: bool = False, repair_delay: float = 0) -> None:
        self.history_state = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
        self.service_session_id = "provider-h0"
        self.run_failed = False
        self.was_interrupted = False
        self.last_response_text = ""
        self.is_running = False
        self.phase = ""
        self.workspace_file = workspace_file
        self.repair_fails = repair_fails
        self.repair_delay = repair_delay
        self.run_calls = 0
        self.repair_started_from: list[str] = []
        self.published_finals: list[str] = []

    def snapshot_history(self) -> _HistoryCheckpoint:
        return _HistoryCheckpoint(copy.deepcopy(self.history_state))

    def restore_history(self, snapshot: _HistoryCheckpoint) -> None:
        self.history_state = copy.deepcopy(snapshot.state)

    def set_requirement_phase(self, phase: str) -> None:
        self.phase = phase

    def set_user_messages(self, _messages: list[str]) -> None:
        return

    def reset_counters(self, *, reset_batch_id: bool = True) -> None:
        assert reset_batch_id is True

    async def run(self, contents: list[Any], created_at=None) -> None:
        _ = created_at
        self.run_calls += 1
        if self.repair_delay:
            await asyncio.sleep(self.repair_delay)
        self.repair_started_from = [message.text for message in self.history_state["messages"]]
        assert self.workspace_file.read_text(encoding="utf-8") == "P0\n"
        self.history_state["messages"].extend([Message("user", contents), Message("assistant", ["P1"])])
        self.last_response_text = "P1"
        self.run_failed = self.repair_fails

    def adopt_fallback_success(self, text: str) -> None:
        self.run_failed = False
        self.was_interrupted = False
        self.last_response_text = text

    async def publish_last_response_as_final(self, *, repeats_provisional: bool = False) -> None:
        self.published_finals.append(self.last_response_text)

    async def interrupt(self) -> None:
        self.was_interrupted = True


class _Runner:
    def __init__(self, host, workspace_file: Path) -> None:
        self.host = host
        self.workspace_file = workspace_file
        self.finalized = 0
        self.imported_p0_prepared = 0

    async def _run_fresh_standard(self, text: str, **kwargs) -> None:
        assert kwargs["finalize"] is False
        self.workspace_file.write_text("P0\n", encoding="utf-8")
        self.host._executor.history_state["messages"] = [Message("user", [text]), Message("assistant", ["P0"])]
        self.host._executor.last_response_text = "P0"
        self.host._turn_number = 1

    async def finalize_current_run(self) -> None:
        self.finalized += 1

    async def _prepare_fresh_without_execution(self, _text: str, **_kwargs) -> None:
        assert self.workspace_file.read_text(encoding="utf-8") == "P0\n"
        self.imported_p0_prepared += 1


class _Snapshotter:
    def __init__(self, workspace_file: Path) -> None:
        self.workspace_file = workspace_file
        self.calls: list[str] = []
        self.restored = 0

    def capture(
        self,
        _workspace,
        artifact_root: Path,
        *,
        snapshot_id: str,
        include_git_history: bool,
        committed_git_head_only: bool = False,
    ):
        artifact_root.mkdir(parents=True)
        self.calls.append("s0-head" if committed_git_head_only else "s0" if include_git_history else "p0")
        return WorkspaceSnapshot(
            snapshot_id=snapshot_id,
            artifact_root=str(artifact_root),
            roots=(),
            manifest_hash=snapshot_id,
            total_bytes=0,
            entry_count=0,
        )

    def matches(self, _snapshot) -> bool:
        return True

    def restore(self, _snapshot) -> None:
        self.restored += 1
        self.workspace_file.write_text("P0\n", encoding="utf-8")

    def discard(self, _snapshot) -> None:
        return


class _Reminder:
    def __init__(self) -> None:
        self.values: list[str] = []

    def queue_hook_reminders(self, values: list[str]) -> None:
        self.values.extend(values)


def _result() -> ClarificationResult:
    return ClarificationResult(
        strategy_version="test-v1",
        revision=1,
        delta="Repository implementation guidance:\n- wire the option",
        selection=ClarificationSelection(),
    )


def test_history_background_uses_only_user_and_assistant_text() -> None:
    state = {
        "compressed_msgs": [SimpleNamespace(messages=[Message("user", ["older requirement"])])],
        "messages": [
            Message("tool", ["private tool result"]),
            Message("assistant", ["prior answer"]),
        ],
    }

    assert _history_background(state) == "user: older requirement\nassistant: prior answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_fails", [False, True, "timeout"])
@pytest.mark.parametrize("reuse_workspace_as_p0", [False, True])
@pytest.mark.parametrize("clarification_only", [False, True])
async def test_workflow_orders_p0_before_delta_and_restores_p0_on_repair_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repair_fails: bool | str,
    reuse_workspace_as_p0: bool,
    clarification_only: bool,
) -> None:
    async def _direct(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    order: list[str] = []

    class _Service:
        def __init__(self, _model, **_kwargs) -> None:
            return

        async def clarify(self, **_kwargs):
            order.append("delta")
            return _result()

        async def generate_pact_input(self, **_kwargs):
            clarification_path = tmp_path / "session/requirement_clarification/turn_1/clarification.private.json"
            assert clarification_path.is_file()
            raise RuntimeError("simulated optional PACT failure")

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", _direct)
    monkeypatch.setattr(workflow_module, "ClarificationService", _Service)
    monkeypatch.setattr(workflow_module, "ChrysClarificationModel", lambda **_kwargs: object())

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace_file = workspace_root / "value.txt"
    workspace_file.write_text("P0\n" if reuse_workspace_as_p0 else "S0\n", encoding="utf-8")
    executor = _Executor(
        workspace_file,
        repair_fails=repair_fails is True,
        repair_delay=0.05 if repair_fails == "timeout" else 0,
    )
    history = SessionHistoryManager()
    history.bind(executor.history_state)
    host = SimpleNamespace(
        _active_profile=object(),
        _agent_profile_fingerprint="agent-fp",
        _model_profile_fingerprint="model-fp",
        _bus=EventBus(),
        _consumed_injections=[],
        _executor=executor,
        _history=history,
        _intermediate_texts={},
        _reminder_middleware=_Reminder(),
        _session_id="session",
        _turn_number=0,
        _turn_state=SimpleNamespace(close_injection_admission=lambda _scope: None),
        _workspace=Workspace.from_cwd(str(workspace_root)),
        _session_dir=tmp_path / "session",
        _requirement_clarification_workflow=None,
        _accumulate_side_call_usage=lambda _usage: None,
    )
    runner = _Runner(host, workspace_file)
    snapshotter = _Snapshotter(workspace_file)
    workflow = RequirementClarificationWorkflow(
        host,
        runner,
        reuse_workspace_as_p0=reuse_workspace_as_p0,
        clarification_only=clarification_only,
        repair_timeout_seconds=0.001 if repair_fails == "timeout" else 5400,
    )
    workflow._snapshotter = snapshotter
    phases: list[str] = []
    finals: list[str] = []

    async def _phase(event: RequirementClarificationPhaseChanged) -> None:
        phases.append(event.phase)

    async def _message(event: AgentMessage) -> None:
        if event.is_final:
            finals.append(event.text)

    await host._bus.subscribe(RequirementClarificationPhaseChanged, _phase)
    await host._bus.subscribe(AgentMessage, _message)
    await workflow.run(
        "implement it",
        created_at=None,
        contents=["implement it"],
        run_scope=None,
        injection_window=None,
        admission_preparation=None,
    )

    assert snapshotter.calls == ["s0-head" if reuse_workspace_as_p0 else "s0", "p0"]
    assert runner.imported_p0_prepared == int(reuse_workspace_as_p0)
    assert order == ["delta"]
    assert executor.repair_started_from == []
    assert runner.finalized == 1
    artifact_root = host._session_dir / "requirement_clarification/turn_1"
    assert (artifact_root / "01-input/requirement.md").is_file()
    assert (artifact_root / "01-input/workspace-snapshot.json").is_file()
    assert (artifact_root / "02-initial-trial/response.json").is_file()
    assert (artifact_root / "03-clarification/deliverable/manifest.json").is_file()
    assert (artifact_root / "06-pact-input/generation.private.json").is_file()
    repair_response = artifact_root / "04-repair/attempts/revision-1/response.json"
    assert (artifact_root / "05-outcome/summary.json").is_file()
    assert (artifact_root / "05-outcome/clarified-requirement.md").is_file()
    assert (artifact_root / "05-outcome/clarified-requirement-delta.md").is_file()
    if clarification_only:
        assert executor.run_calls == 0
        assert host._reminder_middleware.values == []
        assert not repair_response.exists()
        expected_p0_text = "Reused the existing workspace implementation as P0." if reuse_workspace_as_p0 else "P0"
        assert executor.published_finals == [expected_p0_text]
        assert phases[-1] == RequirementWorkflowPhase.COMPLETED
        return
    assert executor.run_calls == 1
    assert host._reminder_middleware.values[0].startswith("[REQUIREMENT_CLARIFICATION_REPAIR]")
    assert repair_response.is_file()
    if repair_fails:
        assert snapshotter.restored == 1
        expected_p0_text = "Reused the existing workspace implementation as P0." if reuse_workspace_as_p0 else "P0"
        assert executor.published_finals == [expected_p0_text]
        assert workspace_file.read_text(encoding="utf-8") == "P0\n"
        assert phases[-1] == RequirementWorkflowPhase.DEGRADED
        assert json.loads(repair_response.read_text(encoding="utf-8"))["status"] in {"failed", "timed_out"}
    else:
        assert snapshotter.restored == 0
        assert executor.published_finals == []
        assert phases[-1] == RequirementWorkflowPhase.COMPLETED
        assert json.loads(repair_response.read_text(encoding="utf-8"))["status"] == "succeeded"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["amend", "stop"])
async def test_live_clarification_side_call_is_cancelled_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
) -> None:
    async def _direct(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    clarify_started = asyncio.Event()
    clarify_cancelled = asyncio.Event()
    clarification_calls = 0

    class _Service:
        def __init__(self, _model, **_kwargs) -> None:
            return

        async def clarify(self, **_kwargs):
            nonlocal clarification_calls
            clarification_calls += 1
            if clarification_calls > 1:
                return replace(_result(), revision=2)
            clarify_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                clarify_cancelled.set()
                raise

        async def generate_pact_input(self, **_kwargs):
            raise RuntimeError("optional PACT unavailable")

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", _direct)
    monkeypatch.setattr(workflow_module, "ClarificationService", _Service)
    monkeypatch.setattr(workflow_module, "ChrysClarificationModel", lambda **_kwargs: object())
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace_file = workspace_root / "value.txt"
    workspace_file.write_text("S0\n", encoding="utf-8")
    executor = _Executor(workspace_file)
    history = SessionHistoryManager()
    history.bind(executor.history_state)
    host = SimpleNamespace(
        _active_profile=object(),
        _agent_profile_fingerprint="agent-fp",
        _model_profile_fingerprint="model-fp",
        _bus=EventBus(),
        _consumed_injections=[],
        _executor=executor,
        _history=history,
        _intermediate_texts={},
        _reminder_middleware=_Reminder(),
        _session_id="session",
        _turn_number=0,
        _turn_state=SimpleNamespace(close_injection_admission=lambda _scope: None),
        _workspace=Workspace.from_cwd(str(workspace_root)),
        _session_dir=tmp_path / "session",
        _requirement_clarification_workflow=None,
        _accumulate_side_call_usage=lambda _usage: None,
    )
    runner = _Runner(host, workspace_file)
    workflow = RequirementClarificationWorkflow(host, runner, clarification_only=True)
    workflow._snapshotter = _Snapshotter(workspace_file)
    run_task = asyncio.create_task(
        workflow.run(
            "implement it",
            created_at=None,
            contents=["implement it"],
            run_scope=None,
            injection_window=None,
            admission_preparation=None,
        )
    )
    async with asyncio.timeout(10):
        await clarify_started.wait()
    if action == "amend":
        assert await workflow.accept_amendment(
            "preserve compatibility",
            created_at=None,
            injection_id="amendment",
        )
    else:
        await workflow.request_stop()
    async with asyncio.timeout(1):
        await clarify_cancelled.wait()
        await run_task

    if action == "amend":
        assert clarification_calls == 2
    else:
        assert clarification_calls == 1
        assert executor.published_finals == ["P0"]


async def test_a_baseline_pass_that_runs_out_of_budget_keeps_its_partial_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P0 hitting its time budget is a bounded baseline, not an abandoned turn.

    Two benchmark tasks spent ninety minutes editing and were then reported with
    no final response because the budget ended the workflow. The budget now ends
    the pass: the partial baseline is snapshotted, clarification and repair run on
    it, and the turn delivers an answer.
    """

    async def _direct(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    class _Service:
        def __init__(self, _model, **_kwargs) -> None:
            return

        async def clarify(self, **_kwargs):
            return _result()

        async def generate_pact_input(self, **_kwargs):
            raise RuntimeError("no PACT input in this test")

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", _direct)
    monkeypatch.setattr(workflow_module, "ClarificationService", _Service)
    monkeypatch.setattr(workflow_module, "ChrysClarificationModel", lambda **_kwargs: object())

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace_file = workspace_root / "value.txt"
    workspace_file.write_text("S0\n", encoding="utf-8")
    executor = _Executor(workspace_file)
    history = SessionHistoryManager()
    history.bind(executor.history_state)
    warnings: list[str] = []
    bus = EventBus()
    host = SimpleNamespace(
        _active_profile=object(),
        _agent_profile_fingerprint="agent-fp",
        _model_profile_fingerprint="model-fp",
        _bus=bus,
        _consumed_injections=[],
        _executor=executor,
        _history=history,
        _intermediate_texts={},
        _reminder_middleware=_Reminder(),
        _session_id="session",
        _turn_number=0,
        _turn_state=SimpleNamespace(close_injection_admission=lambda _scope: None),
        _workspace=Workspace.from_cwd(str(workspace_root)),
        _session_dir=tmp_path / "session",
        _requirement_clarification_workflow=None,
        _accumulate_side_call_usage=lambda _usage: None,
    )

    class _SlowRunner(_Runner):
        async def _run_fresh_standard(self, text: str, **kwargs) -> None:
            # Half the work lands, then the pass overruns its budget.
            self.workspace_file.write_text("P0\n", encoding="utf-8")
            self.host._executor.history_state["messages"] = [Message("user", [text]), Message("assistant", ["half"])]
            self.host._executor.last_response_text = "half"
            self.host._executor.is_running = True
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                self.host._executor.is_running = False
                raise

    runner = _SlowRunner(host, workspace_file)
    snapshotter = _Snapshotter(workspace_file)
    workflow = RequirementClarificationWorkflow(host, runner, initial_timeout_seconds=0.05)
    workflow._snapshotter = snapshotter

    async def _warning(event: Warning) -> None:
        warnings.append(event.code)

    await bus.subscribe(Warning, _warning)
    await workflow.run(
        "do the thing",
        created_at=None,
        contents=["do the thing"],
        run_scope=None,
        injection_window=None,
        admission_preparation=None,
    )

    assert "requirement_clarification_p0_timeout" in warnings
    assert "p0" in snapshotter.calls  # the partial baseline was checkpointed
    assert executor.run_calls == 1  # and the repair pass ran on it
    assert executor.last_response_text == "P1"  # the turn's answer is the repair, not nothing
    assert executor.was_interrupted is False
    assert runner.finalized == 1


@pytest.mark.asyncio
async def test_a_degraded_clarification_still_generates_the_pact_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No valid proposal is no reason to skip the campaign: the requirement alone carries the contract."""

    async def _direct(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    generation_calls: list[str] = []

    class _Service:
        def __init__(self, _model, **_kwargs) -> None:
            return

        async def clarify(self, **_kwargs):
            return ClarificationResult(
                strategy_version="test-v1",
                revision=1,
                delta="",
                selection=ClarificationSelection(),
                status="degraded",
                empty_reason="insufficient_valid_proposals",
            )

        async def generate_pact_input(self, *, result, **_kwargs):
            generation_calls.append(result.status)
            raise RuntimeError("simulated PACT failure after degradation")

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", _direct)
    monkeypatch.setattr(workflow_module, "ClarificationService", _Service)
    monkeypatch.setattr(workflow_module, "ChrysClarificationModel", lambda **_kwargs: object())

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    workspace_file = workspace_root / "value.txt"
    workspace_file.write_text("S0\n", encoding="utf-8")
    executor = _Executor(workspace_file)
    history = SessionHistoryManager()
    history.bind(executor.history_state)
    host = SimpleNamespace(
        _active_profile=object(),
        _agent_profile_fingerprint="agent-fp",
        _model_profile_fingerprint="model-fp",
        _bus=EventBus(),
        _consumed_injections=[],
        _executor=executor,
        _history=history,
        _intermediate_texts={},
        _reminder_middleware=_Reminder(),
        _session_id="session",
        _turn_number=0,
        _turn_state=SimpleNamespace(close_injection_admission=lambda _scope: None),
        _workspace=Workspace.from_cwd(str(workspace_root)),
        _session_dir=tmp_path / "session",
        _requirement_clarification_workflow=None,
        _accumulate_side_call_usage=lambda _usage: None,
    )
    runner = _Runner(host, workspace_file)
    workflow = RequirementClarificationWorkflow(host, runner, clarification_only=True)
    workflow._snapshotter = _Snapshotter(workspace_file)

    await workflow.run(
        "implement it",
        created_at=None,
        contents=["implement it"],
        run_scope=None,
        injection_window=None,
        admission_preparation=None,
    )

    assert generation_calls == ["degraded"]
    artifact_root = host._session_dir / "requirement_clarification/turn_1"
    generation = json.loads((artifact_root / "06-pact-input/generation.private.json").read_text(encoding="utf-8"))
    assert generation["status"] == "failed"
    assert "simulated PACT failure after degradation" in generation["error"]
    assert generation["clarification_status"] == "degraded"
    assert any("generated from the requirement alone" in warning for warning in generation["warnings"])
