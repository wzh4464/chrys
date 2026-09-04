# Copyright (c) 2026 Chrys. All rights reserved.

"""Parallel requirement-enrichment workflow tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import chrys.orchestration.engine.run.requirement_enrichment as workflow_module
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import RequirementEnrichmentPhaseChanged, UserInterrupt, Warning
from chrys.foundation.models.workspace import Workspace
from chrys.kernel import Content
from chrys.orchestration.engine.run.coordinator import TurnCoordinator
from chrys.orchestration.engine.run.requirement_enrichment import RequirementEnrichmentWorkflow
from chrys.orchestration.engine.run.turn_state import TurnRuntimeState
from chrys.service.requirement_clarification.artifacts import ClarificationArtifactStore
from chrys.service.requirement_clarification.snapshot import SnapshotRoot, WorkspaceSnapshot
from chrys.service.requirement_clarification.types import (
    ClarificationResult,
    ClarificationSelection,
    RequirementRevision,
)
from chrys.service.semantic_search import LocalizationArtifact, LocalizationResult, SemanticSearchMode


class _Snapshotter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.discarded = False
        self.matches_current = True

    def capture(self, _workspace, artifact_root: Path, **_kwargs) -> WorkspaceSnapshot:
        artifact_root.mkdir(parents=True)
        return WorkspaceSnapshot(
            snapshot_id="s0",
            artifact_root=str(artifact_root),
            roots=(),
            manifest_hash="hash",
            total_bytes=0,
            entry_count=0,
        )

    def matches(self, _snapshot: WorkspaceSnapshot) -> bool:
        return self.matches_current

    def discard(self, _snapshot: WorkspaceSnapshot) -> None:
        self.discarded = True


class _Reminder:
    def __init__(self) -> None:
        self.values: list[str] = []

    def queue_hook_reminders(self, values: list[str], *, for_next_turn: bool = False) -> None:
        assert for_next_turn is True
        self.values.extend(values)


class _Runner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Any]]] = []

    async def _run_fresh_standard(self, text: str, **kwargs: Any) -> None:
        before_execution = kwargs.pop("before_execution", None)
        if before_execution is not None:
            await before_execution()
        self.calls.append((text, kwargs["contents"]))


def _workflow_fixture(
    tmp_path: Path,
) -> tuple[RequirementEnrichmentWorkflow, _Runner, _Reminder, _Snapshotter]:
    async def _persist_recovery_now() -> bool:
        return True

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    reminder = _Reminder()
    executor = SimpleNamespace(history_state={"messages": [], "compressed_msgs": []}, is_running=False)
    host = SimpleNamespace(
        _active_profile=object(),
        _bus=EventBus(),
        _executor=executor,
        _history=SimpleNamespace(messages=[]),
        _reminder_middleware=reminder,
        _session_id="session",
        _session_dir=tmp_path / "session",
        _turn_number=0,
        _turn_state=TurnRuntimeState(),
        _workspace=Workspace.from_cwd(str(workspace_root)),
        _requirement_enrichment_workflow=None,
        _accumulate_side_call_usage=lambda _usage: None,
        persist_recovery_now=_persist_recovery_now,
    )
    runner = _Runner()
    workflow = RequirementEnrichmentWorkflow(
        host,
        runner,
        strategy="legacy-v1-stabilized",
        clarification_timeout_seconds=10,
        localization_mode=SemanticSearchMode.FALLBACK,
        localization_timeout_seconds=10,
        localization_model_profile="",
    )
    snapshotter = _Snapshotter(workspace_root)
    workflow._snapshotter = snapshotter
    return workflow, runner, reminder, snapshotter


def _clarification() -> ClarificationResult:
    return ClarificationResult(
        strategy_version="test-v1",
        revision=1,
        delta="Repository implementation guidance:\n- preserve the public contract",
        selection=ClarificationSelection(),
    )


def _localization(tmp_path: Path) -> LocalizationResult:
    artifact = LocalizationArtifact(
        result_json=tmp_path / "result.json",
        report_markdown=tmp_path / "report.md",
        index_json=tmp_path / "index.json",
        graph_json=tmp_path / "graph.json",
        trace_jsonl=tmp_path / "trace.jsonl",
        manifest_json=tmp_path / "manifest.json",
    )
    return LocalizationResult(
        payload={
            "locations": [
                {
                    "role": "primary",
                    "file_path": "src/example.py",
                    "symbol": "run",
                    "start_line": 10,
                    "end_line": 20,
                    "reason": "Owns the requested behavior.",
                }
            ],
            "related_tests": ["tests/test_example.py"],
            "related_files": ["pyproject.toml"],
            "unresolved_questions": ["Does the compatibility path need the same behavior?"],
        },
        artifacts=artifact,
    )


@pytest.mark.asyncio
async def test_workflow_runs_analyses_concurrently_and_executes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _direct(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", _direct)
    workflow, runner, reminder, snapshotter = _workflow_fixture(tmp_path)
    both_started = asyncio.Event()
    release = asyncio.Event()
    started: set[str] = set()

    async def _wait_for_peer(name: str) -> None:
        started.add(name)
        if len(started) == 2:
            both_started.set()
        await release.wait()

    async def _clarify(**_kwargs):
        await _wait_for_peer("clarification")
        return _clarification()

    async def _localize(**_kwargs):
        await _wait_for_peer("localization")
        return _localization(tmp_path)

    monkeypatch.setattr(workflow, "_clarify", _clarify)
    monkeypatch.setattr(workflow, "_localize", _localize)
    task = asyncio.create_task(
        workflow.run(
            "implement it",
            created_at=None,
            contents=[Content.from_text("implement it")],
            run_scope=None,
            injection_window=None,
            admission_preparation=None,
        )
    )
    await asyncio.wait_for(both_started.wait(), timeout=1)
    assert started == {"clarification", "localization"}
    release.set()
    await task

    assert len(runner.calls) == 1
    assert runner.calls[0][0] == "implement it"
    assert "## Requirement clarification" in reminder.values[0]
    assert "## Requirement localization" in reminder.values[0]
    assert "src/example.py:10-20 (run)" in reminder.values[0]
    assert "tests/test_example.py" in reminder.values[0]
    assert "pyproject.toml" in reminder.values[0]
    assert "Does the compatibility path need the same behavior?" in reminder.values[0]
    assert snapshotter.discarded is True
    turn_artifacts = tmp_path / "session" / "requirement_enrichment" / "turn_1"
    assert (turn_artifacts / "clarification").is_dir()
    bundle = turn_artifacts / "bundle.json"
    assert bundle.is_file()


def test_render_bundle_uses_available_side_when_the_other_failed(tmp_path: Path) -> None:
    clarification_only = workflow_module._render_bundle(_clarification(), None)
    localization_only = workflow_module._render_bundle(None, _localization(tmp_path))

    assert "Requirement clarification" in clarification_only
    assert "Requirement localization" not in clarification_only
    assert "Requirement localization" in localization_only
    assert "Requirement clarification" not in localization_only


@pytest.mark.asyncio
async def test_localization_searches_every_frozen_workspace_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, _runner, _reminder, _snapshotter = _workflow_fixture(tmp_path)
    primary = tmp_path / "snapshot-primary"
    secondary = tmp_path / "snapshot-secondary"
    primary.mkdir()
    secondary.mkdir()
    snapshot = WorkspaceSnapshot(
        snapshot_id="s0",
        artifact_root=str(tmp_path / "snapshot"),
        roots=(
            SnapshotRoot(str(tmp_path / "live-primary"), str(primary), "main", True, ()),
            SnapshotRoot(str(tmp_path / "live-secondary"), str(secondary), "plugin", False, ()),
        ),
        manifest_hash="hash",
        total_bytes=0,
        entry_count=0,
    )
    localized_roots: list[Path] = []

    async def _localize(repo, _requirement, *, artifact_dir, config):
        root = Path(repo)
        localized_roots.append(root)
        return LocalizationResult(
            payload={"locations": [{"file_path": f"{root.name}.py"}]},
            artifacts=_localization(Path(artifact_dir)).artifacts,
        )

    monkeypatch.setattr(workflow_module, "localize_requirement_async", _localize)
    result = await workflow._localize(
        revision=RequirementRevision(number=1, messages=("implement it",)),
        snapshot=snapshot,
        artifacts=ClarificationArtifactStore(
            tmp_path / "session",
            1,
            artifact_dir_name="requirement_enrichment",
            artifact_subdir="clarification",
        ),
    )

    assert localized_roots == [primary, secondary]
    assert [(item["workspace_root"], item["file_path"]) for item in result.locations] == [
        ("main", "snapshot-primary.py"),
        ("plugin", "snapshot-secondary.py"),
    ]


@pytest.mark.asyncio
async def test_workflow_restarts_both_analyses_for_an_amendment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _direct(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", _direct)
    workflow, runner, reminder, _snapshotter = _workflow_fixture(tmp_path)
    first_pair_started = asyncio.Event()
    revision_calls: dict[str, list[int]] = {"clarification": [], "localization": []}

    async def _analysis(name: str, revision) -> None:
        revision_calls[name].append(revision.number)
        if sum(calls.count(1) for calls in revision_calls.values()) == 2:
            first_pair_started.set()
        if revision.number == 1:
            await asyncio.Event().wait()

    async def _clarify(*, revision, **_kwargs):
        await _analysis("clarification", revision)
        return _clarification()

    async def _localize(*, revision, **_kwargs):
        await _analysis("localization", revision)
        return _localization(tmp_path)

    monkeypatch.setattr(workflow, "_clarify", _clarify)
    monkeypatch.setattr(workflow, "_localize", _localize)
    task = asyncio.create_task(
        workflow.run(
            "original requirement",
            created_at=None,
            contents=[Content.from_text("original requirement")],
            run_scope=None,
            injection_window=None,
            admission_preparation=None,
        )
    )
    await asyncio.wait_for(first_pair_started.wait(), timeout=1)
    assert await workflow.accept_amendment(
        "preserve compatibility",
        created_at=None,
        injection_id="amendment-1",
    )
    await task

    assert revision_calls == {"clarification": [1, 2], "localization": [1, 2]}
    assert len(runner.calls) == 1
    assert "original requirement" in runner.calls[0][0]
    assert "preserve compatibility" in runner.calls[0][0]
    assert len(runner.calls[0][1]) == 2
    assert reminder.values


@pytest.mark.asyncio
async def test_workflow_accepts_amendment_while_snapshot_is_being_captured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, runner, _reminder, _snapshotter = _workflow_fixture(tmp_path)
    snapshot_started = asyncio.Event()
    release_snapshot = asyncio.Event()
    original_to_thread = asyncio.to_thread

    async def _controlled_to_thread(function, /, *args, **kwargs):
        if function == workflow._snapshotter.capture:
            snapshot_started.set()
            await release_snapshot.wait()
        return await original_to_thread(function, *args, **kwargs)

    async def _clarify(**_kwargs):
        return _clarification()

    async def _localize(**_kwargs):
        return _localization(tmp_path)

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", _controlled_to_thread)
    monkeypatch.setattr(workflow, "_clarify", _clarify)
    monkeypatch.setattr(workflow, "_localize", _localize)
    task = asyncio.create_task(
        workflow.run(
            "original requirement",
            created_at=None,
            contents=[Content.from_text("original requirement")],
            run_scope=None,
            injection_window=None,
            admission_preparation=None,
        )
    )
    await asyncio.wait_for(snapshot_started.wait(), timeout=1)

    assert await workflow.accept_amendment(
        "include the compatibility path",
        created_at=None,
        injection_id="snapshot-amendment",
    )
    release_snapshot.set()
    await task

    assert len(runner.calls) == 1
    assert "include the compatibility path" in runner.calls[0][0]


@pytest.mark.asyncio
async def test_workflow_discards_both_results_when_workspace_snapshot_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _direct(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", _direct)
    workflow, runner, reminder, snapshotter = _workflow_fixture(tmp_path)
    snapshotter.matches_current = False

    async def _clarify(**_kwargs):
        return _clarification()

    async def _localize(**_kwargs):
        return _localization(tmp_path)

    monkeypatch.setattr(workflow, "_clarify", _clarify)
    monkeypatch.setattr(workflow, "_localize", _localize)
    await workflow.run(
        "implement it",
        created_at=None,
        contents=[Content.from_text("implement it")],
        run_scope=None,
        injection_window=None,
        admission_preparation=None,
    )

    assert len(runner.calls) == 1
    assert runner.calls[0][0] == "implement it"
    assert reminder.values == []


@pytest.mark.asyncio
async def test_workflow_treats_artifact_write_failure_as_advisory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _direct(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", _direct)
    workflow, runner, reminder, _snapshotter = _workflow_fixture(tmp_path)
    warnings: list[Warning] = []

    async def _capture_warning(event: Warning) -> None:
        warnings.append(event)

    await workflow._host._bus.subscribe(Warning, _capture_warning)

    async def _clarify(**_kwargs):
        return _clarification()

    async def _localize(**_kwargs):
        return _localization(tmp_path)

    def _fail_bundle(*_args, **_kwargs) -> None:
        raise OSError("artifact filesystem unavailable")

    monkeypatch.setattr(workflow, "_clarify", _clarify)
    monkeypatch.setattr(workflow, "_localize", _localize)
    monkeypatch.setattr(workflow_module, "_save_bundle", _fail_bundle)
    await workflow.run(
        "implement it",
        created_at=None,
        contents=[Content.from_text("implement it")],
        run_scope=None,
        injection_window=None,
        admission_preparation=None,
    )

    assert len(runner.calls) == 1
    assert reminder.values
    assert any("artifact write failed" in warning.message for warning in warnings)


@pytest.mark.asyncio
async def test_workflow_reports_partial_success_as_degraded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _direct(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", _direct)
    workflow, _runner, reminder, _snapshotter = _workflow_fixture(tmp_path)
    phases: list[RequirementEnrichmentPhaseChanged] = []

    async def _capture_phase(event: RequirementEnrichmentPhaseChanged) -> None:
        phases.append(event)

    await workflow._host._bus.subscribe(RequirementEnrichmentPhaseChanged, _capture_phase)

    async def _clarify(**_kwargs):
        return _clarification()

    async def _fail_localization(**_kwargs):
        raise RuntimeError("localization unavailable")

    monkeypatch.setattr(workflow, "_clarify", _clarify)
    monkeypatch.setattr(workflow, "_localize", _fail_localization)
    await workflow.run(
        "implement it",
        created_at=None,
        contents=[Content.from_text("implement it")],
        run_scope=None,
        injection_window=None,
        admission_preparation=None,
    )

    assert reminder.values
    assert phases[-1].terminal is True
    assert phases[-1].phase == "degraded"


@pytest.mark.asyncio
async def test_workflow_reports_successful_empty_results_as_completed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _direct(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", _direct)
    workflow, _runner, reminder, _snapshotter = _workflow_fixture(tmp_path)
    phases: list[RequirementEnrichmentPhaseChanged] = []

    async def _capture_phase(event: RequirementEnrichmentPhaseChanged) -> None:
        phases.append(event)

    await workflow._host._bus.subscribe(RequirementEnrichmentPhaseChanged, _capture_phase)
    empty_clarification = ClarificationResult(
        strategy_version="test-v1",
        revision=1,
        delta="",
        selection=ClarificationSelection(),
        empty_reason="already_specific",
    )
    empty_localization = LocalizationResult(payload={"locations": []}, artifacts=_localization(tmp_path).artifacts)

    async def _clarify(**_kwargs):
        return empty_clarification

    async def _localize(**_kwargs):
        return empty_localization

    monkeypatch.setattr(workflow, "_clarify", _clarify)
    monkeypatch.setattr(workflow, "_localize", _localize)
    await workflow.run(
        "implement it",
        created_at=None,
        contents=[Content.from_text("implement it")],
        run_scope=None,
        injection_window=None,
        admission_preparation=None,
    )

    assert reminder.values == []
    assert phases[-1].terminal is True
    assert phases[-1].phase == "completed"


@pytest.mark.asyncio
async def test_workflow_executes_original_requirement_when_both_analyses_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _direct(function, /, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(workflow_module.asyncio, "to_thread", _direct)
    workflow, runner, reminder, _snapshotter = _workflow_fixture(tmp_path)

    async def _fail(**_kwargs):
        raise RuntimeError("analysis unavailable")

    monkeypatch.setattr(workflow, "_clarify", _fail)
    monkeypatch.setattr(workflow, "_localize", _fail)
    await workflow.run(
        "implement it",
        created_at=None,
        contents=[Content.from_text("implement it")],
        run_scope=None,
        injection_window=None,
        admission_preparation=None,
    )

    assert len(runner.calls) == 1
    assert runner.calls[0][0] == "implement it"
    assert reminder.values == []


@pytest.mark.asyncio
async def test_coordinator_interrupt_stops_enrichment_and_arms_pre_executor_interrupt() -> None:
    class _ActiveWorkflow:
        def __init__(self) -> None:
            self.stopped = False

        async def request_stop(self) -> None:
            self.stopped = True

    workflow = _ActiveWorkflow()
    turn_state = TurnRuntimeState()
    current = asyncio.current_task()
    assert current is not None
    turn_state.run_task = current
    host = SimpleNamespace(
        _agent_loading=False,
        _agent_profile=None,
        _executor=SimpleNamespace(is_running=False),
        _hook_manager=None,
        _requirement_clarification_workflow=None,
        _requirement_enrichment_workflow=workflow,
        _session_id="session",
        _sub_agent_tools=None,
        _trajectory_recorder=SimpleNamespace(interrupt_requested_soon=lambda: None),
        _turn_state=turn_state,
        _workspace=None,
    )

    await TurnCoordinator(host).on_user_interrupt(UserInterrupt())

    assert workflow.stopped is True
    assert turn_state.consume_pre_executor_interrupt() is True
