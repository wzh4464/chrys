# Copyright (c) 2026 Chrys. All rights reserved.

"""Focused engine coverage for recovery and session-safety edge paths."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Event as ThreadEvent
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import chrys.orchestration.engine.build.builder as agent_builder_module
import chrys.orchestration.engine.engine as engine_module
import chrys.service.session.lifecycle as session_lifecycle
from chrys.foundation.config.process_settings import install_process_settings, process_settings
from chrys.foundation.config.runtime_pointer import set_model_pointer
from chrys.foundation.config.settings import (
    DEFAULT_ROLLBACK_SNAPSHOTS_KEEP,
    SESSION_ROOT_DIR_ENV_VAR,
    Settings,
)
from chrys.foundation.config.settings_store import LoadedSettings, load_settings
from chrys.foundation.config.spec import SettingOrigin, Source
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    AgentProfileSwitch,
    ApprovalModeUpdated,
    Error,
    ModelProfileSwitched,
    ProfileSwitched,
    RollbackResult,
    SessionNew,
    SessionRestore,
    SessionRestored,
    SetApprovalMode,
    SetModelProfile,
    SettingsReload,
    SettingsReloaded,
    UserRollback,
    Warning,
    WorkspaceChange,
    WorkspaceUpdated,
)
from chrys.foundation.models.workspace import Workspace
from chrys.foundation.observability.sink import OtelSessionSink, get_otel_sink, set_otel_sink
from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.foundation.recovery import RecoveryPersistOutcome
from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY
from chrys.foundation.util.lock import FileLock
from chrys.foundation.util.session_ids import session_short_id
from chrys.kernel import Content, FunctionTool, LoopRecorder, Message
from chrys.orchestration.engine.build.construction import StagedBuild
from chrys.orchestration.engine.engine import AgentEngine
from chrys.orchestration.engine.state import controls as engine_controls
from chrys.orchestration.engine.state.machine import EngineState, Trigger
from chrys.orchestration.sub_agents.tools import SubAgentTools
from chrys.service.agent_middleware.control.approval import ApprovalMiddleware
from chrys.service.agent_middleware.system_reminder import CATALOG_POINTER_RECORD_COUNT_STATE_KEY
from chrys.service.approval.policy import ApprovalMode, ApprovalPolicy
from chrys.service.context.compaction.spill import CATALOG_RELATIVE_PATH
from chrys.service.hooks.schema import HookDecision
from chrys.service.mutations import workspace_changes
from chrys.service.mutations.store import SnapshotStore
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.types import RollbackPlan
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import (
    AgentProfile,
    ApprovalConfig,
    CompactionConfig,
    MCPServerConfig,
    ModelConfig,
    ToolsConfig,
)
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.resolver import ModelSelection, loaded_with_active_model_profile
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.session.message_metadata import stamp_message_created_at
from chrys.service.state.store import SESSION_RECOVERY_FILE_NAME, JsonFileStateStore, SessionMeta
from chrys.service.todos.tracker import TodoTracker
from chrys.service.trajectory.tombstone import pending_delete_intents
from tests.support.waiting import wait_until


def _profile(name: str = "Code", display_name: str = "") -> AgentProfile:
    return AgentProfile(
        name=name,
        display_name=display_name,
        instructions=f"{name} instructions.",
        tools=ToolsConfig(builtins=[]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
    )


def _registry(*profiles: AgentProfile) -> AgentProfileRegistry:
    registry = AgentProfileRegistry()
    for profile in profiles:
        registry.register(profile)
    return registry


def _model_registry(*profiles: ModelProfile) -> ModelProfileRegistry:
    registry = ModelProfileRegistry()
    for profile in profiles:
        registry.register(profile)
    return registry


def _session_meta(*, model_profile_id: str) -> SessionMeta:
    now = datetime.now(UTC)
    return SessionMeta(
        session_id="restore_me",
        agent_profile="Code",
        agent_display_name="Code",
        created_at=now,
        updated_at=now,
        message_count=1,
        model_profile_id=model_profile_id,
    )


def _restore_model_profile_environment(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("CHRYS_MODEL_PROFILE", None)
    else:
        os.environ["CHRYS_MODEL_PROFILE"] = previous


async def _collect(events: list[Any], event: Any) -> None:
    events.append(event)


def _assert_display_message(event: Error | Warning, key: str, args: dict[str, str | int] | None = None) -> None:
    reference = event.display_message
    assert reference is not None
    assert reference.definition.key == key
    assert dict(reference.args) == (args or {})


@dataclass
class _FakeTurn:
    turn_id: int
    detection_truncated: bool = False


@dataclass
class _FakeRestoreResult:
    changed: bool


class _FakeMutationTracker:
    def __init__(self, turn_ids: list[int]) -> None:
        self._turns = [_FakeTurn(turn_id) for turn_id in turn_ids]
        self.rollback_calls: list[tuple[set[int], set[str] | None]] = []

    def get_all_turns(self) -> list[_FakeTurn]:
        return self._turns

    def serialize(self) -> dict[str, Any]:
        return {
            "turns": [{"turn_id": turn.turn_id, "mutations": []} for turn in self._turns],
            "snapshots": {},
        }

    def get_rollback_plan_for_turns(self, turn_ids: set[int]) -> RollbackPlan:
        return RollbackPlan(entries=[])

    def rollback_turns(
        self,
        turn_ids: set[int],
        *,
        only_paths: set[str] | None = None,
        plan: RollbackPlan | None = None,
    ) -> list[_FakeRestoreResult]:
        self.rollback_calls.append((turn_ids, only_paths))
        return [_FakeRestoreResult(changed=True), _FakeRestoreResult(changed=False)]


class _ApprovalTarget:
    def __init__(self) -> None:
        self.modes: list[ApprovalMode] = []

    def set_approval_mode(self, mode: ApprovalMode) -> None:
        self.modes.append(mode)


class _SessionEndProbeHookManager:
    def __init__(self, session_file: Path) -> None:
        self._session_file = session_file
        self.exists_during_fire: list[bool] = []
        self.payloads: list[dict[str, Any]] = []

    async def fire(self, _event: object, payload: dict[str, Any], **_kwargs: object) -> HookDecision:
        self.exists_during_fire.append(self._session_file.exists())
        self.payloads.append(payload)
        return HookDecision()

    async def drain_session(self, *, close: bool = True) -> None:
        return None


@pytest.mark.asyncio
async def test_profile_switch_reports_missing_registry() -> None:
    events: list[Error] = []
    bus = EventBus()
    await bus.subscribe(Error, lambda event: _collect(events, event))
    engine = AgentEngine(bus, settings=Settings())
    engine._session_id = "sid"

    await engine._on_profile_switch(AgentProfileSwitch(profile_name="Explore"))

    assert [event.code for event in events] == ["no_registry"]
    assert events[0].message == "No profile registry configured — cannot switch profiles"
    _assert_display_message(events[0], "controls.no_registry")
    assert events[0].session_id == "sid"


@pytest.mark.asyncio
async def test_profile_switch_reports_missing_profile() -> None:
    events: list[Error] = []
    bus = EventBus()
    await bus.subscribe(Error, lambda event: _collect(events, event))
    engine = AgentEngine(bus, settings=Settings(), agent_registry=_registry(_profile()))
    engine._session_id = "sid"

    await engine._on_profile_switch(AgentProfileSwitch(profile_name="Explore"))

    assert [event.code for event in events] == ["profile_not_found"]
    assert events[0].message == "Profile 'Explore' not found"
    _assert_display_message(events[0], "controls.profile_not_found", {"profile_name": "Explore"})


@pytest.mark.asyncio
async def test_profile_switch_retries_start_when_no_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    explore = _profile("Explore", "Explore Agent")
    events: list[ProfileSwitched] = []
    bus = EventBus()
    await bus.subscribe(ProfileSwitched, lambda event: _collect(events, event))
    engine = AgentEngine(bus, settings=Settings(), agent_registry=_registry(explore))
    engine._session_id = "sid"
    calls: list[tuple[AgentProfile, str]] = []

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        calls.append((profile, operation))
        engine._agent_profile = profile
        executor = MagicMock()
        executor.history_state = {"messages": []}
        engine._executor = executor

    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_profile_switch(AgentProfileSwitch(profile_name="Explore"))

    assert calls == [(explore, "switch")]
    assert [(event.from_profile, event.to_profile, event.session_id) for event in events] == [
        ("Explore", "Explore", "sid")
    ]


@pytest.mark.asyncio
async def test_profile_switch_waits_for_active_run_before_soft_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    code = _profile("Code", "Code Agent")
    explore = _profile("Explore", "Explore Agent")
    engine = AgentEngine(EventBus(), settings=Settings(), agent_registry=_registry(code, explore))
    engine._agent_profile = code
    engine._executor = object()  # type: ignore[assignment]
    release_run = asyncio.Event()
    calls: list[tuple[AgentProfile, str]] = []

    async def active_run() -> None:
        await release_run.wait()

    async def fake_soft_restart(profile: AgentProfile, **kwargs: Any) -> None:
        calls.append((profile, kwargs["operation"]))

    engine._turn_state.run_task = asyncio.create_task(active_run())
    monkeypatch.setattr(engine, "_soft_restart", fake_soft_restart)

    switch_task = asyncio.create_task(engine._on_profile_switch(AgentProfileSwitch(profile_name="Explore")))
    try:
        await asyncio.sleep(0)
        assert calls == []
    finally:
        release_run.set()
        await asyncio.gather(switch_task, engine._turn_state.run_task, return_exceptions=True)

    assert calls == [(explore, "switch")]


@pytest.mark.asyncio
async def test_settings_reload_with_no_executor_uses_refreshed_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    stale = _profile("Code", "Old Code")
    refreshed = _profile("Code", "Fresh Code")
    replacement_settings = Settings(default_approval_mode="auto")
    engine = AgentEngine(EventBus(), settings=Settings(), agent_registry=_registry(refreshed))
    engine._agent_profile = stale
    calls: list[tuple[AgentProfile, str]] = []

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        calls.append((profile, operation))
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(
        "chrys.orchestration.engine.state.controls.load_settings",
        lambda **kwargs: LoadedSettings(settings=replacement_settings, provenance={}),
    )
    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_settings_reload(SettingsReload())

    assert engine._settings is replacement_settings
    assert calls == [(refreshed, "settings_reload")]


@pytest.mark.asyncio
async def test_settings_reload_disables_baseline_but_preserves_safety_notice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    engine = AgentEngine(EventBus(), settings=Settings(workspace_change_notice=True))
    engine._agent_profile = profile
    engine._workspace = Workspace.from_cwd(str(workspace_root))
    tracker = engine._workspace_change_tracker
    tracker.retarget_roots(engine._workspace)
    tracker.capture_baseline(1)
    tracker.queue_safety_notice("retained files")

    async def fake_start(
        _profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "settings_reload"
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    # The reload's real load reads the engine conftest's
    # ``CHRYS_WORKSPACE_CHANGE_NOTICE=0``, which is exactly the disabling
    # re-read this test needs.
    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_settings_reload(SettingsReload())

    assert tracker.baseline is None
    assert tracker.take_pending_notice() == "retained files"
    assert tracker.take_pending_notice() is None


@pytest.mark.asyncio
async def test_failed_settings_reload_keeps_workspace_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    engine = AgentEngine(EventBus(), settings=Settings(workspace_change_notice=True))
    engine._agent_profile = profile
    engine._workspace = Workspace.from_cwd(str(workspace_root))
    tracker = engine._workspace_change_tracker
    tracker.retarget_roots(engine._workspace)
    baseline = tracker.capture_baseline(1)

    async def failing_start(
        _profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = operation
        raise RuntimeError("rebuild failed")

    monkeypatch.setattr(engine, "start", failing_start)

    with pytest.raises(RuntimeError, match="rebuild failed"):
        await engine._on_settings_reload(SettingsReload())

    # The reload rolled back to the enabled settings; the live baseline must survive.
    assert engine._settings.workspace_change_notice is True
    assert tracker.baseline == baseline


@pytest.mark.asyncio
async def test_model_switch_waits_for_startup_load_before_mutating_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    active = _profile("Code", "Code Agent")
    engine = AgentEngine(EventBus(), settings=Settings(model_profile="old-model"), agent_registry=_registry(active))
    engine._agent_profile = active
    engine._begin_agent_load()
    old_settings = engine._settings
    calls: list[tuple[AgentProfile, str]] = []

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        calls.append((profile, operation))

    monkeypatch.setattr(engine, "start", fake_start)
    task = asyncio.create_task(engine._on_set_model_profile(SetModelProfile(profile_id="new-model")))
    try:
        await asyncio.sleep(0)
        assert calls == []
        assert engine._settings is old_settings
    finally:
        engine._finish_agent_load()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_settings_reload_waits_for_startup_load_before_mutating_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = _profile("Code", "Code Agent")
    replacement_settings = Settings(default_approval_mode="auto")
    engine = AgentEngine(
        EventBus(), settings=Settings(default_approval_mode="manual"), agent_registry=_registry(active)
    )
    engine._agent_profile = active
    engine._begin_agent_load()
    old_settings = engine._settings
    calls: list[tuple[AgentProfile, str]] = []

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        calls.append((profile, operation))

    monkeypatch.setattr(
        "chrys.orchestration.engine.state.controls.load_settings",
        lambda **kwargs: LoadedSettings(settings=replacement_settings, provenance={}),
    )
    monkeypatch.setattr(engine, "start", fake_start)
    task = asyncio.create_task(engine._on_settings_reload(SettingsReload()))
    try:
        await asyncio.sleep(0)
        assert calls == []
        assert engine._settings is old_settings
    finally:
        engine._finish_agent_load()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_settings_reload_with_missing_registry_entry_reuses_active_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = _profile("Code", "Code Agent")
    replacement_settings = Settings(default_approval_mode="auto")
    engine = AgentEngine(EventBus(), settings=Settings(), agent_registry=_registry())
    engine._agent_profile = active
    engine._executor = object()  # type: ignore[assignment]
    calls: list[tuple[AgentProfile, str]] = []

    async def fake_soft_restart(profile: AgentProfile, **kwargs: Any) -> None:
        calls.append((profile, kwargs["operation"]))
        if kwargs.get("staged_loaded") is not None:
            engine._settings_handle.install(kwargs["staged_loaded"])

    monkeypatch.setattr(
        "chrys.orchestration.engine.state.controls.load_settings",
        lambda **kwargs: LoadedSettings(settings=replacement_settings, provenance={}),
    )
    monkeypatch.setattr(engine, "_soft_restart", fake_soft_restart)

    await engine._on_settings_reload(SettingsReload())

    assert engine._settings is replacement_settings
    assert calls == [(active, "settings_reload")]


@pytest.mark.asyncio
async def test_workspace_change_starts_from_failed_build_without_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    events: list[WorkspaceUpdated] = []
    bus = EventBus()
    await bus.subscribe(WorkspaceUpdated, lambda event: _collect(events, event))
    engine = AgentEngine(bus, settings=Settings())
    engine._session_id = "sid"
    engine._agent_profile = profile
    calls: list[tuple[AgentProfile, str]] = []

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        calls.append((start_profile, operation))
        executor = MagicMock()
        executor.history_state = {"messages": []}
        engine._executor = executor
        # A successful start commits the staged workspace; the double honors
        # the same contract now that nothing pre-assigns it.
        if workspace is not None:
            engine._workspace = workspace

    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_workspace_change(WorkspaceChange(primary_cwd=str(tmp_path)))

    assert engine.workspace is not None
    assert engine.workspace.primary_cwd == str(tmp_path)
    assert calls == [(profile, "workspace_change")]
    assert [(event.primary_cwd, event.session_id) for event in events] == [(str(tmp_path), "sid")]


@pytest.mark.asyncio
async def test_failed_workspace_change_without_executor_keeps_the_old_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new root rides the build as staged input: when the no-executor
    rebuild fails, both the live workspace and the live settings must still
    describe the old root."""
    profile = _profile()
    old_root = tmp_path / "old"
    old_root.mkdir()
    new_root = tmp_path / "new"
    new_root.mkdir()
    replacement_settings = Settings(default_approval_mode="auto")
    engine = AgentEngine(EventBus(), settings=Settings())
    engine._session_id = "sid"
    engine._agent_profile = profile
    engine._workspace = Workspace.from_cwd(str(old_root))
    old_settings = engine._settings

    async def failing_build(_profile: AgentProfile, _staged: Any, **_kwargs: object) -> None:
        raise RuntimeError("build failed")

    monkeypatch.setattr(
        "chrys.orchestration.engine.state.controls.load_settings",
        lambda **kwargs: LoadedSettings(settings=replacement_settings, provenance={}),
    )
    monkeypatch.setattr(engine, "_build_agent", failing_build)

    with pytest.raises(RuntimeError, match="build failed"):
        await engine._on_workspace_change(WorkspaceChange(primary_cwd=str(new_root)))

    assert engine.workspace is not None
    assert engine.workspace.primary_cwd == Workspace.from_cwd(str(old_root)).primary_cwd
    assert engine._settings is old_settings


@pytest.mark.asyncio
async def test_failed_workspace_change_load_publishes_a_terminal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A load failure aborts before the rebuild publishes anything, and the
    bus swallows handler exceptions: without an explicit Error a caller
    awaiting the change (ACP's 60s wait, the TUI's loading state) hangs."""
    profile = _profile()
    new_root = tmp_path / "new"
    new_root.mkdir()
    errors: list[Error] = []
    bus = EventBus()
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = AgentEngine(bus, settings=Settings())
    engine._session_id = "sid"
    engine._agent_profile = profile
    old_settings = engine._settings

    def failing_load(**kwargs: Any) -> LoadedSettings:
        raise RuntimeError("unreadable config")

    monkeypatch.setattr("chrys.orchestration.engine.state.controls.load_settings", failing_load)

    with pytest.raises(RuntimeError, match="unreadable config"):
        await engine._on_workspace_change(WorkspaceChange(primary_cwd=str(new_root)))

    assert engine._settings is old_settings
    assert [error.code for error in errors] == ["workspace_change_failed"]
    assert errors[0].session_id == "sid"


@pytest.mark.asyncio
async def test_workspace_change_waits_for_startup_load_before_mutating_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    engine = AgentEngine(EventBus(), settings=Settings())
    engine._agent_profile = profile
    engine._begin_agent_load()
    calls: list[tuple[AgentProfile, str]] = []

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        calls.append((start_profile, operation))

    monkeypatch.setattr(engine, "start", fake_start)
    task = asyncio.create_task(engine._on_workspace_change(WorkspaceChange(primary_cwd=str(tmp_path))))
    try:
        await asyncio.sleep(0)
        assert calls == []
        assert engine.workspace is None
    finally:
        engine._finish_agent_load()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_workspace_change_soft_restarts_live_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    engine = AgentEngine(EventBus(), settings=Settings())
    engine._agent_profile = profile
    engine._executor = object()  # type: ignore[assignment]
    calls: list[tuple[AgentProfile, str, str]] = []

    async def fake_soft_restart(start_profile: AgentProfile, **kwargs: Any) -> None:
        workspace = kwargs["workspace"]
        calls.append((start_profile, workspace.primary_cwd, kwargs["operation"]))

    monkeypatch.setattr(engine, "_soft_restart", fake_soft_restart)

    await engine._on_workspace_change(WorkspaceChange(primary_cwd=str(tmp_path)))

    assert calls == [(profile, str(tmp_path), "workspace_change")]


@pytest.mark.asyncio
async def test_settings_reload_derives_the_project_root_from_the_session_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The project trust domain is root-derived, and the pinned model plus the
    launch mode's retry policy travel *into* the load, not after it."""
    profile = _profile()
    root = tmp_path / "workspace"
    root.mkdir()
    engine = AgentEngine(
        EventBus(),
        settings=Settings(model_profile="pinned-model", frontend_default_max_transient_retries=15),
    )
    engine._agent_profile = profile
    engine._workspace = Workspace.from_cwd(str(root))
    engine.pin_model_profile()
    load_kwargs: dict[str, Any] = {}

    def fake_load(**kwargs: Any) -> LoadedSettings:
        load_kwargs.update(kwargs)
        return LoadedSettings(settings=Settings(), provenance={})

    async def fake_start(
        _profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr("chrys.orchestration.engine.state.controls.load_settings", fake_load)
    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_settings_reload(SettingsReload())

    assert load_kwargs["project_root"] == Path(engine._workspace.primary_cwd)
    assert load_kwargs["eval_context"].frontend_default_max_transient_retries == 15
    assert load_kwargs["model_profile"] == "pinned-model"


@pytest.mark.asyncio
async def test_workspace_change_derives_settings_from_the_new_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A workspace change is a settings reload in disguise: the new root's
    project layer rides the same rebuild that installs the new workspace."""
    profile = _profile()
    old_root = tmp_path / "old"
    old_root.mkdir()
    new_root = tmp_path / "new"
    new_root.mkdir()
    replacement_settings = Settings(default_approval_mode="auto")
    engine = AgentEngine(EventBus(), settings=Settings())
    engine._agent_profile = profile
    engine._workspace = Workspace.from_cwd(str(old_root))
    load_kwargs: dict[str, Any] = {}

    def fake_load(**kwargs: Any) -> LoadedSettings:
        load_kwargs.update(kwargs)
        return LoadedSettings(settings=replacement_settings, provenance={})

    async def fake_start(
        _profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr("chrys.orchestration.engine.state.controls.load_settings", fake_load)
    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_workspace_change(WorkspaceChange(primary_cwd=str(new_root)))

    assert load_kwargs["project_root"] == Path(Workspace.from_cwd(str(new_root)).primary_cwd)
    assert engine._settings is replacement_settings


@pytest.mark.asyncio
async def test_set_approval_mode_updates_runtime_targets_and_persists_bypass_as_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[ApprovalModeUpdated] = []
    bus = EventBus()
    await bus.subscribe(ApprovalModeUpdated, lambda event: _collect(events, event))
    settings = Settings(default_approval_mode="manual")
    engine = AgentEngine(bus, settings=settings)
    engine._session_id = "sid"
    executor = _ApprovalTarget()
    sub_agents = SubAgentTools(event_bus=bus, approval_mode=ApprovalMode.MANUAL)
    live_approval = ApprovalMiddleware(
        approval_policy=ApprovalPolicy(ApprovalConfig(default="require")),
        event_bus=bus,
        approval_mode=ApprovalMode.MANUAL,
    )
    sub_agents._live_approvals.append(live_approval)
    engine._executor = executor  # type: ignore[assignment]
    engine._sub_agent_tools = sub_agents
    persisted: list[str] = []

    def persist_mode(mode: str) -> None:
        persisted.append(mode)

    monkeypatch.setattr(engine_module, "persist_approval_mode", persist_mode)

    await engine._on_set_approval_mode(SetApprovalMode(mode="bypass"))

    assert engine._approval_mode is ApprovalMode.BYPASS
    assert executor.modes == [ApprovalMode.BYPASS]
    assert sub_agents._approval_mode is ApprovalMode.BYPASS
    assert live_approval.approval_mode is ApprovalMode.BYPASS
    assert persisted == ["bypass"]
    assert engine.settings.default_approval_mode == "auto"
    assert [(event.mode, event.session_id) for event in events] == [("bypass", "sid")]


@pytest.mark.asyncio
async def test_setting_the_approval_mode_moves_its_provenance_with_the_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live write is still a write: the layer that won has to change too."""
    bus = EventBus()
    loaded = LoadedSettings(settings=Settings(default_approval_mode="manual"), provenance={})
    engine = AgentEngine(bus, loaded_settings=loaded)
    engine._session_id = "sid"
    monkeypatch.setattr(engine_module, "persist_approval_mode", lambda _mode: None)

    await engine._on_set_approval_mode(SetApprovalMode(mode="auto"))

    assert engine.settings.default_approval_mode == "auto"
    assert engine._loaded_settings.settings is engine.settings
    assert engine._loaded_settings.source_for("approval.default_mode").layer is Source.RUNTIME


@pytest.mark.asyncio
async def test_setting_the_approval_mode_breaks_the_seal_it_no_longer_describes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``approval.default_mode`` is DANGEROUS, so a bad env value seals it.

    The seal means "we refused every layer and fell back to the built-in
    default". Once the user picks a mode at runtime that sentence is false,
    and leaving the seal on reproduces the contradiction the freeze fix
    removed: sealed, yet holding a value nobody defaulted to.
    """
    monkeypatch.setenv("CHRYS_DEFAULT_APPROVAL_MODE", "garbage")
    loaded = load_settings()
    assert loaded.settings.default_approval_mode == "manual"
    assert "approval.default_mode" in loaded.sealed_keys

    bus = EventBus()
    engine = AgentEngine(bus, loaded_settings=loaded)
    engine._session_id = "sid"
    monkeypatch.setattr(engine_module, "persist_approval_mode", lambda _mode: None)

    await engine._on_set_approval_mode(SetApprovalMode(mode="auto"))

    assert engine.settings.default_approval_mode == "auto"
    assert "approval.default_mode" not in engine._loaded_settings.sealed_keys


@pytest.mark.asyncio
async def test_set_approval_mode_can_skip_global_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[ApprovalModeUpdated] = []
    bus = EventBus()
    await bus.subscribe(ApprovalModeUpdated, lambda event: _collect(events, event))
    settings = Settings(default_approval_mode="manual")
    engine = AgentEngine(bus, settings=settings)
    engine._session_id = "sid"
    executor = _ApprovalTarget()
    sub_agents = _ApprovalTarget()
    engine._executor = executor  # type: ignore[assignment]
    engine._sub_agent_tools = sub_agents  # type: ignore[assignment]
    persisted: list[str] = []

    def persist_mode(mode: str) -> None:
        persisted.append(mode)

    monkeypatch.setattr(engine_module, "persist_approval_mode", persist_mode)

    await engine._on_set_approval_mode(SetApprovalMode(mode="bypass", persist=False))

    assert engine._approval_mode is ApprovalMode.BYPASS
    assert executor.modes == [ApprovalMode.BYPASS]
    assert sub_agents.modes == [ApprovalMode.BYPASS]
    assert persisted == []
    assert settings.default_approval_mode == "manual"
    assert [(event.mode, event.session_id) for event in events] == [("bypass", "sid")]


def test_current_profile_snapshot_reflects_live_runtime() -> None:
    engine = AgentEngine(EventBus(), settings=Settings())
    engine._session_id = "sid"
    engine._agent_profile = _profile(name="Code", display_name="Coder")
    engine._active_profile = ModelProfile(
        id="model-1", name="GPT", provider="openai", model_id="gpt-5", max_context_tokens=200000
    )
    engine._tool_names = ["shell", "read_file"]
    engine._skill_names = ["search"]
    engine._memory_files = ["AGENTS.md"]
    engine._sub_agent_tools = MagicMock()
    engine._sub_agent_tools.tool_names.return_value = ["explore"]
    executor = MagicMock()
    executor.history_state = {"messages": [1, 2, 3]}
    engine._executor = executor

    snapshot = engine.current_profile_snapshot()

    # No-op switch: from and to are identical and reflect the live agent.
    assert snapshot.from_profile == "Code"
    assert snapshot.to_profile == "Code"
    assert snapshot.from_display_name == "Coder"
    assert snapshot.to_display_name == "Coder"
    assert snapshot.session_id == "sid"
    assert snapshot.message_count == 3
    assert snapshot.model_profile_id == "model-1"
    assert snapshot.max_context_tokens == 200000
    assert snapshot.tool_names == ["shell", "read_file"]
    assert snapshot.skill_names == ["search"]
    assert snapshot.sub_agent_tool_names == ["explore"]
    assert snapshot.memory_files == ["AGENTS.md"]
    # Mutating the snapshot's lists must not bleed back into engine state.
    snapshot.tool_names.append("rm")
    assert engine._tool_names == ["shell", "read_file"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("shutdown", "runtime_mutation_shutdown"),
        ("session_changed", "runtime_mutation_session_changed"),
        ("superseded", "runtime_mutation_superseded"),
        ("load_generation", "runtime_mutation_superseded"),
        ("load_active", "runtime_mutation_load_active"),
        ("active_admission", "runtime_mutation_busy"),
        ("active_run", "runtime_mutation_busy"),
        ("drain_cancelled", "runtime_mutation_busy"),
    ],
)
async def test_rebuild_permit_validator_denial_codes(case: str, expected_code: str) -> None:
    engine = AgentEngine(EventBus(), settings=Settings())
    engine._session_id = "sid"
    token = engine.capture_rebuild_control_token()
    release_run = asyncio.Event()
    run_task: asyncio.Task[None] | None = None

    if case == "shutdown":
        engine._shutting_down = True
    elif case == "session_changed":
        engine._session_generation += 1
    elif case == "superseded":
        engine._advance_build_generation()
    elif case == "load_generation":
        engine._load_generation += 1
    elif case == "load_active":
        engine._agent_loading = True
    elif case == "active_admission":
        assert engine._turn_state.reserve_prompt_admission(
            kind="fresh",
            session_generation=token.session_generation,
            build_generation=token.build_generation,
        )
    elif case == "active_run":

        async def _run() -> None:
            await release_run.wait()

        run_task = asyncio.create_task(_run())
        engine._turn_state.run_task = run_task

    try:
        denied = engine._validate_rebuild_token_after_boundary(token, drain_cancelled=case == "drain_cancelled")
    finally:
        release_run.set()
        if run_task is not None:
            await asyncio.wait_for(run_task, timeout=5.0)

    assert denied is not None
    assert denied.code == expected_code


@pytest.mark.asyncio
async def test_runtime_control_denials_publish_captured_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = EventBus()
    errors: list[Error] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = AgentEngine(bus, settings=Settings(), agent_registry=_registry(_profile(), _profile("Explore")))
    engine._session_id = "old-session"
    engine._agent_profile = _profile()
    engine._executor = object()  # type: ignore[assignment]
    denied = engine_controls.RebuildPermitDenied(
        reason="session_changed",
        code="runtime_mutation_session_changed",
        message="session moved",
    )

    async def deny_with_new_live_session(
        _token: engine_controls.RebuildControlToken,
    ) -> engine_controls.RebuildPermitDenied:
        engine._session_id = "new-session"
        return denied

    monkeypatch.setattr(engine, "acquire_rebuild_permit", deny_with_new_live_session)

    engine._session_id = "old-session"
    await engine._on_profile_switch(AgentProfileSwitch(profile_name="Explore"))
    engine._session_id = "old-session"
    await engine._on_set_model_profile(SetModelProfile(profile_id="new-model"))
    engine._session_id = "old-session"
    await engine._on_settings_reload(SettingsReload())
    engine._session_id = "old-session"
    await engine._on_workspace_change(WorkspaceChange(primary_cwd="/tmp/chrys-new"))

    assert [event.code for event in errors] == [
        "runtime_mutation_session_changed",
        "runtime_mutation_session_changed",
        "runtime_mutation_session_changed",
        "runtime_mutation_session_changed",
    ]
    assert {event.session_id for event in errors} == {"old-session"}


@pytest.mark.asyncio
async def test_already_satisfied_denials_publish_typed_success(monkeypatch: pytest.MonkeyPatch) -> None:
    bus = EventBus()
    profile_events: list[ProfileSwitched] = []
    model_events: list[ModelProfileSwitched] = []
    workspace_events: list[WorkspaceUpdated] = []
    await bus.subscribe(ProfileSwitched, lambda event: _collect(profile_events, event))
    await bus.subscribe(ModelProfileSwitched, lambda event: _collect(model_events, event))
    await bus.subscribe(WorkspaceUpdated, lambda event: _collect(workspace_events, event))

    profile = _profile()
    engine = AgentEngine(bus, settings=Settings(), agent_registry=_registry(profile))
    engine._session_id = "sid"
    engine._agent_profile = profile
    executor = MagicMock()
    executor.history_state = {"messages": []}
    engine._executor = executor
    engine._active_profile = ModelProfile(id="model-1", name="Model", provider="openai", model_id="gpt-5")
    engine._workspace = Workspace.from_cwd("/tmp/chrys-current")
    denied = engine_controls.RebuildPermitDenied(
        reason="superseded",
        code="runtime_mutation_superseded",
        message="newer runtime",
    )
    acquire_calls = 0

    async def deny_after_boundary(_token: engine_controls.RebuildControlToken) -> engine_controls.RebuildPermitDenied:
        nonlocal acquire_calls
        acquire_calls += 1
        return denied

    monkeypatch.setattr(engine, "acquire_rebuild_permit", deny_after_boundary)

    await engine._on_profile_switch(AgentProfileSwitch(profile_name="Code"))
    await engine._on_set_model_profile(SetModelProfile(profile_id="model-1"))
    await engine._on_workspace_change(WorkspaceChange(primary_cwd="/tmp/chrys-current"))

    assert acquire_calls == 3
    assert [(event.from_profile, event.to_profile, event.session_id) for event in profile_events] == [
        ("Code", "Code", "sid")
    ]
    assert [(event.model_profile_id, event.session_id) for event in model_events] == [("model-1", "sid")]
    assert engine._model_profile_pinned is True
    assert engine._settings.model_profile == "model-1"
    assert engine._settings.model_profile_override == "model-1"
    assert engine._settings.model_profile_override_sub_agents is False
    assert [(event.primary_cwd, event.session_id) for event in workspace_events] == [("/tmp/chrys-current", "sid")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "code"),
    [
        ("session_changed", "runtime_mutation_session_changed"),
        ("shutdown", "runtime_mutation_shutdown"),
    ],
)
async def test_terminal_denials_are_not_masked_by_live_satisfied_state(
    monkeypatch: pytest.MonkeyPatch,
    reason: engine_controls.RebuildPermitDeniedReason,
    code: str,
) -> None:
    bus = EventBus()
    errors: list[Error] = []
    profile_events: list[ProfileSwitched] = []
    model_events: list[ModelProfileSwitched] = []
    workspace_events: list[WorkspaceUpdated] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    await bus.subscribe(ProfileSwitched, lambda event: _collect(profile_events, event))
    await bus.subscribe(ModelProfileSwitched, lambda event: _collect(model_events, event))
    await bus.subscribe(WorkspaceUpdated, lambda event: _collect(workspace_events, event))

    profile = _profile()
    engine = AgentEngine(bus, settings=Settings(), agent_registry=_registry(profile))
    engine._session_id = "old-session"
    engine._agent_profile = profile
    executor = MagicMock()
    executor.history_state = {"messages": []}
    engine._executor = executor
    engine._active_profile = ModelProfile(id="model-1", name="Model", provider="openai", model_id="gpt-5")
    engine._workspace = Workspace.from_cwd("/tmp/chrys-current")
    denied = engine_controls.RebuildPermitDenied(reason=reason, code=code, message=reason)

    async def deny_after_boundary(_token: engine_controls.RebuildControlToken) -> engine_controls.RebuildPermitDenied:
        if reason == "session_changed":
            engine._session_id = "new-session"
        return denied

    monkeypatch.setattr(engine, "acquire_rebuild_permit", deny_after_boundary)

    engine._session_id = "old-session"
    await engine._on_profile_switch(AgentProfileSwitch(profile_name="Code"))
    engine._session_id = "old-session"
    await engine._on_set_model_profile(SetModelProfile(profile_id="model-1"))
    engine._session_id = "old-session"
    await engine._on_workspace_change(WorkspaceChange(primary_cwd="/tmp/chrys-current"))

    assert [(event.code, event.session_id) for event in errors] == [
        (code, "old-session"),
        (code, "old-session"),
        (code, "old-session"),
    ]
    assert profile_events == []
    assert model_events == []
    assert workspace_events == []


@pytest.mark.asyncio
async def test_concurrent_profile_switch_waiters_supersede_after_first_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    code = _profile("Code")
    explore = _profile("Explore")
    docs = _profile("docs")
    bus = EventBus()
    errors: list[Error] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = AgentEngine(bus, settings=Settings(), agent_registry=_registry(code, explore, docs))
    engine._session_id = "sid"
    engine._agent_profile = code
    engine._executor = object()  # type: ignore[assignment]
    restart_entered = asyncio.Event()
    release_restart = asyncio.Event()
    calls: list[str] = []

    async def fake_soft_restart_with_permit(
        _permit: engine_controls.RebuildPermit,
        profile: AgentProfile,
        workspace: Workspace | None = None,
        *,
        operation: str = "switch",
        staged_loaded: LoadedSettings | None = None,
    ) -> None:
        _ = workspace, operation
        calls.append(profile.name)
        restart_entered.set()
        await release_restart.wait()
        engine._agent_profile = profile
        engine._advance_build_generation()

    monkeypatch.setattr(engine, "soft_restart_with_rebuild_permit", fake_soft_restart_with_permit)

    first = asyncio.create_task(engine._on_profile_switch(AgentProfileSwitch(profile_name="Explore")))
    await asyncio.wait_for(restart_entered.wait(), timeout=5.0)
    second = asyncio.create_task(engine._on_profile_switch(AgentProfileSwitch(profile_name="docs")))
    await asyncio.sleep(0)
    release_restart.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=5.0)

    assert calls == ["Explore"]
    assert [(event.code, event.session_id) for event in errors] == [("runtime_mutation_superseded", "sid")]


@pytest.mark.asyncio
async def test_workspace_change_without_agent_publishes_not_ready_error(tmp_path: Path) -> None:
    bus = EventBus()
    errors: list[Error] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = AgentEngine(bus, settings=Settings())
    engine._session_id = "sid"

    await engine._on_workspace_change(WorkspaceChange(primary_cwd=str(tmp_path)))

    assert [(event.code, event.message, event.session_id) for event in errors] == [
        ("runtime_mutation_not_ready", "No active agent — cannot change workspace", "sid")
    ]
    _assert_display_message(errors[0], "controls.workspace_switch_not_ready")


@pytest.mark.asyncio
async def test_model_switch_without_agent_publishes_not_ready_error() -> None:
    bus = EventBus()
    errors: list[Error] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = AgentEngine(bus, settings=Settings())
    engine._session_id = "sid"

    await engine._on_set_model_profile(SetModelProfile(profile_id="model-1"))

    assert [(event.code, event.message, event.session_id) for event in errors] == [
        ("runtime_mutation_not_ready", "No active agent — cannot switch model", "sid")
    ]
    _assert_display_message(errors[0], "controls.model_switch_not_ready")


@pytest.mark.asyncio
async def test_model_switch_success_publishes_before_rebuild_permit_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = EventBus()
    release_seen = False
    event_release_states: list[bool] = []
    await bus.subscribe(ModelProfileSwitched, lambda _event: _collect(event_release_states, release_seen))
    profile = _profile()
    engine = AgentEngine(bus, settings=Settings(), agent_registry=_registry(profile))
    engine._session_id = "sid"
    engine._agent_profile = profile
    engine._executor = MagicMock()
    engine._active_profile = ModelProfile(id="old-model", name="Old", provider="openai", model_id="gpt-4")

    async def fake_soft_restart(
        _profile: AgentProfile,
        workspace: Workspace | None = None,
        *,
        operation: str = "switch",
        staged_loaded: LoadedSettings | None = None,
    ) -> None:
        _ = workspace, operation
        engine._active_profile = ModelProfile(id="new-model", name="New", provider="openai", model_id="gpt-5")

    original_release = engine.release_rebuild_permit

    def release_with_marker(permit: engine_controls.RebuildPermit) -> None:
        nonlocal release_seen
        release_seen = True
        original_release(permit)

    monkeypatch.setattr(engine, "_soft_restart", fake_soft_restart)
    monkeypatch.setattr(engine, "release_rebuild_permit", release_with_marker)

    await engine._on_set_model_profile(SetModelProfile(profile_id="new-model"))

    assert event_release_states == [False]


@pytest.mark.asyncio
async def test_settings_reload_success_publishes_before_rebuild_permit_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = EventBus()
    release_seen = False
    event_release_states: list[bool] = []
    await bus.subscribe(SettingsReloaded, lambda _event: _collect(event_release_states, release_seen))
    profile = _profile()
    replacement_settings = Settings(default_approval_mode="auto")
    engine = AgentEngine(bus, settings=Settings(), agent_registry=_registry(profile))
    engine._session_id = "sid"
    engine._agent_profile = profile
    engine._executor = MagicMock()

    async def fake_soft_restart(
        _profile: AgentProfile,
        workspace: Workspace | None = None,
        *,
        operation: str = "switch",
        staged_loaded: LoadedSettings | None = None,
    ) -> None:
        _ = workspace, operation

    original_release = engine.release_rebuild_permit

    def release_with_marker(permit: engine_controls.RebuildPermit) -> None:
        nonlocal release_seen
        release_seen = True
        original_release(permit)

    monkeypatch.setattr(
        "chrys.orchestration.engine.state.controls.load_settings",
        lambda **kwargs: LoadedSettings(settings=replacement_settings, provenance={}),
    )
    monkeypatch.setattr(engine, "_soft_restart", fake_soft_restart)
    monkeypatch.setattr(engine, "release_rebuild_permit", release_with_marker)

    await engine._on_settings_reload(SettingsReload())

    assert event_release_states == [False]


@pytest.mark.asyncio
async def test_set_approval_mode_ignores_invalid_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[ApprovalModeUpdated] = []
    bus = EventBus()
    await bus.subscribe(ApprovalModeUpdated, lambda event: _collect(events, event))
    engine = AgentEngine(bus, settings=Settings())
    persisted: list[str] = []

    def persist_mode(mode: str) -> None:
        persisted.append(mode)

    monkeypatch.setattr(engine_module, "persist_approval_mode", persist_mode)

    await engine._on_set_approval_mode(SetApprovalMode(mode="unknown"))

    assert engine._approval_mode is ApprovalMode.MANUAL
    assert persisted == []
    assert events == []


@pytest.mark.asyncio
async def test_settings_reload_follows_env_without_per_session_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TUI model switch path: writes CHRYS_MODEL_PROFILE to env, then reloads.
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "env-model")
    engine = AgentEngine(EventBus(), settings=Settings(model_profile="old-model"))
    engine._session_id = "sid"
    engine._agent_profile = _profile()
    engine._executor = None

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_settings_reload(SettingsReload())

    assert engine._settings.model_profile == "env-model"


@pytest.mark.asyncio
async def test_settings_reload_preserves_frontend_retry_default_and_rereads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHRYS_MAX_TRANSIENT_RETRIES", "7")
    engine = AgentEngine(
        EventBus(),
        settings=Settings(
            max_transient_retries=None,
            frontend_default_max_transient_retries=10,
        ),
    )
    engine._session_id = "sid"
    engine._agent_profile = _profile()
    engine._executor = None

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_settings_reload(SettingsReload())

    assert engine._settings.frontend_default_max_transient_retries == 10
    assert engine._settings.max_transient_retries == 7
    assert engine._settings.effective_max_transient_retries() == 7


@pytest.mark.asyncio
async def test_settings_reload_reports_a_value_it_had_to_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reload is when a user finds out their edit did not take.

    The reload used to truncate ``LoadedSettings`` to ``.settings``, so the
    same bad value went silent from the second read onwards — exactly when the
    user is looking for feedback.
    """
    monkeypatch.setenv("CHRYS_SESSION_TITLE_AUTO", "nonsense")
    bus = EventBus()
    warnings: list[Warning] = []
    await bus.subscribe(Warning, warnings.append)
    engine = AgentEngine(bus, settings=Settings())
    engine._session_id = "sid"
    engine._agent_profile = _profile()
    engine._executor = None

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_settings_reload(SettingsReload())

    assert [warning.code for warning in warnings] == ["setting_rejected"]
    assert "CHRYS_SESSION_TITLE_AUTO" in warnings[0].message
    assert warnings[0].session_id == "sid"


@pytest.mark.asyncio
async def test_settings_reload_does_not_claim_a_restart_value_took_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reload re-reads every key; the process re-reads none of these six.

    Raw HTTP capture is decided once, at bootstrap, and its consumers hold that
    answer for the process. A reload that wrote the new value into the live
    settings would have the engine — and the panel reading it — report capture
    as on while nothing was capturing.
    """
    monkeypatch.delenv("CHRYS_DEBUG_LLM_RAW_HTTP_LOG", raising=False)
    install_process_settings(load_settings())
    engine = AgentEngine(EventBus(), settings=Settings())
    engine._agent_profile = _profile()
    engine._executor = None

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)
    monkeypatch.setenv("CHRYS_DEBUG_LLM_RAW_HTTP_LOG", "1")

    await engine._on_settings_reload(SettingsReload())

    assert engine._settings.raw_http_capture is False
    assert process_settings().raw_http_capture is False


@pytest.mark.asyncio
async def test_settings_reload_still_applies_a_reload_scoped_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other side of the freeze: it must not stop a reload from reloading."""
    monkeypatch.delenv("CHRYS_SESSION_TITLE_AUTO", raising=False)
    install_process_settings(load_settings())
    engine = AgentEngine(EventBus(), settings=Settings())
    engine._agent_profile = _profile()
    engine._executor = None

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)
    monkeypatch.setenv("CHRYS_SESSION_TITLE_AUTO", "0")

    await engine._on_settings_reload(SettingsReload())

    assert engine._settings.session_title_auto is False


@pytest.mark.asyncio
async def test_settings_reload_holds_a_routed_restart_field_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otel has no snapshot slot; the routing itself holds it — and says so.

    Its readers decided at bootstrap whether telemetry exists, so a reload
    writing the new value into the live settings would only change what the
    process *reports*, not what it does. The user still deserves to hear that
    the edit was saved and what it is waiting on.
    """
    monkeypatch.delenv("CHRYS_OTEL", raising=False)
    install_process_settings(load_settings())
    bus = EventBus()
    warnings: list[Warning] = []
    await bus.subscribe(Warning, warnings.append)
    engine = AgentEngine(bus, settings=Settings())
    engine._session_id = "sid"
    engine._agent_profile = _profile()
    engine._executor = None

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)
    monkeypatch.setenv("CHRYS_OTEL", "1")

    await engine._on_settings_reload(SettingsReload())

    assert engine._settings.otel_enabled is False
    assert [warning.code for warning in warnings] == ["settings_restart_required"]
    assert "otel.enabled" in warnings[0].message
    assert warnings[0].session_id == "sid"


@pytest.mark.asyncio
async def test_settings_reload_applies_a_dev_mode_change_without_restart_noise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dev_mode's one consumer reads it during the rebuild a reload performs.

    So the reload genuinely applies it — and must not warn that a restart is
    needed for a value that just took effect.
    """
    monkeypatch.delenv("CHRYS_DEV_MODE", raising=False)
    install_process_settings(load_settings())
    bus = EventBus()
    warnings: list[Warning] = []
    await bus.subscribe(Warning, warnings.append)
    engine = AgentEngine(bus, settings=Settings())
    engine._agent_profile = _profile()
    engine._executor = None

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)
    monkeypatch.setenv("CHRYS_DEV_MODE", "1")

    await engine._on_settings_reload(SettingsReload())

    assert engine._settings.dev_mode is True
    assert warnings == []


@pytest.mark.asyncio
async def test_settings_reload_without_an_agent_reloads_instead_of_echoing_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A host with nothing built must reload for real, not report one it skipped.

    Startup before the first build, a first build that failed, and a host that
    only subscribed all reach this path. Echoing completion left the next build
    reading the configuration the user had just changed.
    """
    monkeypatch.delenv("CHRYS_SESSION_TITLE_AUTO", raising=False)
    install_process_settings(load_settings())
    bus = EventBus()
    reloaded: list[SettingsReloaded] = []
    await bus.subscribe(SettingsReloaded, lambda event: _collect(reloaded, event))
    engine = AgentEngine(bus, settings=Settings())
    engine._session_id = "sid"
    engine._agent_profile = None
    engine._executor = None
    starts: list[str] = []

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile
        starts.append(operation)

    monkeypatch.setattr(engine, "start", fake_start)
    monkeypatch.setenv("CHRYS_SESSION_TITLE_AUTO", "0")

    await engine._on_settings_reload(SettingsReload())

    assert engine._settings.session_title_auto is False
    # Only the rebuild is skipped — there is no runtime to replace.
    assert starts == []
    assert len(reloaded) == 1


@pytest.mark.asyncio
async def test_settings_reload_without_an_agent_reports_a_failed_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same path must not report success when the load itself failed."""
    bus = EventBus()
    errors: list[Error] = []
    reloaded: list[SettingsReloaded] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    await bus.subscribe(SettingsReloaded, lambda event: _collect(reloaded, event))
    original = Settings(ask_user_timeout_seconds=123)
    engine = AgentEngine(bus, settings=original)
    engine._session_id = "sid"
    engine._agent_profile = None
    engine._executor = None

    def explode(**_kwargs: object) -> object:
        raise ValueError("settings store unavailable")

    monkeypatch.setattr("chrys.orchestration.engine.state.controls.load_settings", explode)

    with pytest.raises(ValueError):
        await engine._on_settings_reload(SettingsReload())

    assert engine._settings is original
    assert [e.code for e in errors] == ["settings_reload_failed"]
    assert reloaded == []


@pytest.mark.asyncio
async def test_settings_reload_loads_off_the_event_loop_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load reads config files and waits on their lock.

    This handler runs inline on the bus, so loading on the event-loop thread
    would stall every other event for as long as the disk takes.
    """
    load_threads: list[int] = []
    real_load = engine_controls.load_settings

    def recording_load(**kwargs: Any) -> LoadedSettings:
        load_threads.append(threading.get_ident())
        return real_load(**kwargs)

    monkeypatch.setattr(engine_controls, "load_settings", recording_load)
    engine = AgentEngine(EventBus(), settings=Settings())
    engine._agent_profile = _profile()
    engine._executor = None

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_settings_reload(SettingsReload())

    assert load_threads and threading.get_ident() not in load_threads


@pytest.mark.asyncio
async def test_settings_reload_preserves_per_session_model_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ACP set_session_model path: in-memory override must survive reload, not
    # revert to the global env default.
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "env-model")
    engine = AgentEngine(
        EventBus(),
        settings=Settings(
            model_profile="session-model",
            model_profile_override="session-model",
            model_profile_override_sub_agents=True,
        ),
    )
    engine._session_id = "sid"
    engine._agent_profile = _profile()
    engine._executor = None
    engine._model_profile_pinned = True

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_settings_reload(SettingsReload())

    assert engine._settings.model_profile == "session-model"
    assert engine._settings.model_profile_override == "session-model"
    assert engine._settings.model_profile_override_sub_agents is True


@pytest.mark.asyncio
async def test_settings_reload_returns_the_model_label_to_the_command_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--model`` parks its value in the environment so it survives reload.

    The reload reads it back from there, so without re-attribution the panel
    would claim the user configured ``CHRYS_MODEL_PROFILE`` — on the first
    reload and every one after.
    """
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "cli-model")
    cli_model = ModelProfile(id="cli-model", name="Cli", model_id="gpt-cli")
    loaded = loaded_with_active_model_profile(load_settings(), cli_model, Source.CLI)
    engine = AgentEngine(EventBus(), loaded_settings=loaded)
    engine._session_id = "sid"
    engine._agent_profile = _profile()
    engine._executor = None

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)

    for _ in range(2):
        await engine._on_settings_reload(SettingsReload())

        assert engine._settings.model_profile == "cli-model"
        assert engine._loaded_settings.source_for("model.profile.active").layer is Source.CLI


@pytest.mark.asyncio
async def test_settings_reload_does_not_relabel_a_model_the_user_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-attribution only, never re-imposition.

    The model config screen replaces the parked environment value on purpose;
    the reload must let the new value win and credit the environment, not stamp
    the command line's label (or worse, its stale value) back on top.
    """
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "cli-model")
    cli_model = ModelProfile(id="cli-model", name="Cli", model_id="gpt-cli")
    loaded = loaded_with_active_model_profile(load_settings(), cli_model, Source.CLI)
    engine = AgentEngine(EventBus(), loaded_settings=loaded)
    engine._session_id = "sid"
    engine._agent_profile = _profile()
    engine._executor = None

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "screen-model")

    await engine._on_settings_reload(SettingsReload())

    assert engine._settings.model_profile == "screen-model"
    assert engine._loaded_settings.source_for("model.profile.active").layer is Source.ENV


@pytest.mark.asyncio
async def test_settings_reload_does_not_relabel_a_runtime_pick_of_the_same_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Choosing the profile the flag already named is still the user choosing it.

    Nothing about the value distinguishes the two, so it is the carrier that
    decides: this one arrives registered as ``PROCESS_RUNTIME`` rather than
    read back out of the parked variable. Relabelling it would credit the flag
    for a live choice — for the rest of the session, since this reload's output
    is the next reload's ``previous``.
    """
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "cli-model")
    cli_model = ModelProfile(id="cli-model", name="Cli", model_id="gpt-cli")
    loaded = loaded_with_active_model_profile(load_settings(), cli_model, Source.CLI)
    engine = AgentEngine(EventBus(), loaded_settings=loaded)
    engine._session_id = "sid"
    engine._agent_profile = _profile()
    engine._executor = None

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)
    set_model_pointer("cli-model", origin=SettingOrigin(layer=Source.PROCESS_RUNTIME))

    for _ in range(2):
        await engine._on_settings_reload(SettingsReload())

        assert engine._settings.model_profile == "cli-model"
        assert engine._loaded_settings.source_for("model.profile.active").layer is Source.PROCESS_RUNTIME


@pytest.mark.asyncio
async def test_settings_reload_preserves_pinned_ask_user_timeout_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ACP injects ask_user_timeout_seconds via Settings (not env) and pins it;
    # reload must keep it (None = client owns timing) instead of reverting to env.
    monkeypatch.setenv("CHRYS_ASK_USER_TIMEOUT_SECONDS", "600")
    engine = AgentEngine(EventBus(), settings=Settings(ask_user_timeout_seconds=None))
    engine.pin_ask_user_timeout()
    engine._session_id = "sid"
    engine._agent_profile = _profile()
    engine._executor = None

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_settings_reload(SettingsReload())

    assert engine._settings.ask_user_timeout_seconds is None


@pytest.mark.asyncio
async def test_settings_reload_unpinned_ask_user_timeout_follows_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # TUI/CLI never pin the timeout: a changed CHRYS_ASK_USER_TIMEOUT_SECONDS must
    # take effect on reload instead of being frozen at the live in-memory value.
    monkeypatch.setenv("CHRYS_ASK_USER_TIMEOUT_SECONDS", "42")
    engine = AgentEngine(EventBus(), settings=Settings(ask_user_timeout_seconds=999))
    engine._session_id = "sid"
    engine._agent_profile = _profile()
    engine._executor = None

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_settings_reload(SettingsReload())

    assert engine._settings.ask_user_timeout_seconds == 42


@pytest.mark.asyncio
async def test_settings_reload_publishes_error_when_the_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A load that raises aborts before any rebuild publishes a completion event.
    # The handler must emit a failure event so a caller awaiting the reload
    # resolves instead of hanging, and must restore the previous live settings.
    bus = EventBus()
    errors: list[Error] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    original = Settings(ask_user_timeout_seconds=123)
    engine = AgentEngine(bus, settings=original)
    engine._session_id = "sid"
    engine._agent_profile = _profile()
    engine._executor = None

    def explode(**_kwargs: object) -> object:
        raise ValueError("settings store unavailable")

    monkeypatch.setattr("chrys.orchestration.engine.state.controls.load_settings", explode)

    with pytest.raises(ValueError):
        await engine._on_settings_reload(SettingsReload())

    assert engine._settings is original
    assert [e.code for e in errors] == ["settings_reload_failed"]
    assert errors[0].message == "settings store unavailable"
    assert errors[0].display_message is None
    assert errors[0].session_id == "sid"


@pytest.mark.asyncio
async def test_settings_reload_survives_one_invalid_env_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed variable is rejected on its own; the reload still completes.

    ``CHRYS_ROLLBACK_SNAPSHOTS_KEEP=abc`` used to raise out of the loader and
    take the whole reload with it.
    """
    bus = EventBus()
    engine = AgentEngine(bus, settings=Settings(ask_user_timeout_seconds=123))
    engine._session_id = "sid"
    engine._agent_profile = _profile()
    engine._executor = None

    async def fake_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr(engine, "start", fake_start)
    monkeypatch.setenv("CHRYS_ROLLBACK_SNAPSHOTS_KEEP", "abc")

    await engine._on_settings_reload(SettingsReload())

    assert engine._settings.rollback_snapshots_keep == DEFAULT_ROLLBACK_SNAPSHOTS_KEEP


@pytest.mark.asyncio
async def test_rollback_welcome_reverts_selected_paths_and_reports_changed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[RollbackResult] = []
    bus = EventBus()
    await bus.subscribe(RollbackResult, lambda event: _collect(events, event))
    engine = AgentEngine(bus, settings=Settings(), state_store=JsonFileStateStore(tmp_path))
    engine._session_id = "rb_test"
    engine._fsm.try_transition(Trigger.START)
    tracker = _FakeMutationTracker([1, 3])
    engine._mutation_tracker = tracker  # type: ignore[assignment]
    reset_calls: list[str] = []

    async def fake_reset(
        session_id: str,
        *,
        write_lock_held: bool = False,
        after_delete: Any = None,
        before_restart: Any = None,
    ) -> bool:
        _ = write_lock_held, before_restart
        reset_calls.append(session_id)
        if after_delete is not None:
            await after_delete()
        return True

    monkeypatch.setattr(engine, "_reset_session_to_welcome", fake_reset)

    await engine._on_user_rollback(
        UserRollback(target_turn=0, revert_changes=True, selected_paths=["src/a.py", "src/b.py"])
    )

    assert tracker.rollback_calls == [({1, 3}, {"src/a.py", "src/b.py"})]
    assert reset_calls == ["rb_test"]
    assert len(events) == 1
    assert events[0].target_turn == 0
    assert events[0].files_reverted == 1


@pytest.mark.asyncio
async def test_rollback_welcome_lock_failure_does_not_revert_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[RollbackResult] = []
    errors: list[Error] = []
    bus = EventBus()
    await bus.subscribe(RollbackResult, lambda event: _collect(events, event))
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = AgentEngine(bus, settings=Settings(), state_store=JsonFileStateStore(tmp_path))
    engine._session_id = "rb_test"
    engine._fsm.try_transition(Trigger.START)
    tracker = _FakeMutationTracker([1, 3])
    engine._mutation_tracker = tracker  # type: ignore[assignment]
    lock_path = engine._session_write_lock_path("rb_test")
    assert lock_path is not None
    monkeypatch.setattr(engine_module, "SESSION_WRITE_LOCK_TIMEOUT_SECONDS", 0.0)

    with FileLock(lock_path):
        await engine._on_user_rollback(UserRollback(target_turn=0, revert_changes=True, selected_paths=["src/a.py"]))

    assert tracker.rollback_calls == []
    assert events == []
    assert [event.code for event in errors] == ["rollback_reset_failed"]
    assert errors[0].message == ("Rollback to welcome could not reset the session because the session state is busy.")
    _assert_display_message(errors[0], "rollback.reset_failed")
    assert engine.session_generation == 0
    assert engine._turn_state.prompt_admission_closed is False


@pytest.mark.asyncio
async def test_rollback_welcome_reset_failure_does_not_revert_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[RollbackResult] = []
    errors: list[Error] = []
    bus = EventBus()
    await bus.subscribe(RollbackResult, lambda event: _collect(events, event))
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = AgentEngine(bus, settings=Settings(), state_store=JsonFileStateStore(tmp_path))
    engine._session_id = "rb_test"
    engine._fsm.try_transition(Trigger.START)
    tracker = _FakeMutationTracker([1, 3])
    engine._mutation_tracker = tracker  # type: ignore[assignment]

    async def fake_reset(
        _session_id: str,
        *,
        write_lock_held: bool = False,
        after_delete: Any = None,
        before_restart: Any = None,
    ) -> bool:
        _ = write_lock_held, after_delete, before_restart
        return False

    monkeypatch.setattr(engine, "_reset_session_to_welcome", fake_reset)

    await engine._on_user_rollback(UserRollback(target_turn=0, revert_changes=True, selected_paths=["src/a.py"]))

    assert tracker.rollback_calls == []
    assert events == []
    assert [event.code for event in errors] == ["rollback_reset_failed"]
    assert errors[0].message == ("Rollback to welcome could not reset the session because the session state is busy.")
    _assert_display_message(errors[0], "rollback.reset_failed")


@pytest.mark.asyncio
async def test_rollback_welcome_unmaterialized_session_dir_still_reverts_files(tmp_path: Path) -> None:
    events: list[RollbackResult] = []
    bus = EventBus()
    await bus.subscribe(RollbackResult, lambda event: _collect(events, event))
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(bus, settings=Settings(), state_store=store)
    engine._session_id = "rb_test"
    engine._fsm.try_transition(Trigger.START)
    tracker = _FakeMutationTracker([1, 3])
    engine._mutation_tracker = tracker  # type: ignore[assignment]

    session_dir = store.session_dir("rb_test")
    assert not session_dir.exists()

    await engine._on_user_rollback(UserRollback(target_turn=0, revert_changes=True, selected_paths=["src/a.py"]))

    assert tracker.rollback_calls == [({1, 3}, {"src/a.py"})]
    assert len(events) == 1
    assert events[0].files_reverted == 1
    assert not session_dir.exists()


@pytest.mark.asyncio
async def test_rollback_reverts_explicit_turn_ids_above_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bus = EventBus()
    engine = AgentEngine(bus, settings=Settings(), state_store=JsonFileStateStore(tmp_path))
    engine._session_id = "rb_test"
    engine._fsm.try_transition(Trigger.START)
    tracker = _FakeMutationTracker([1, 3, 2])
    engine._mutation_tracker = tracker  # type: ignore[assignment]

    session_dir = tmp_path / "rb_test"
    snap_dir = session_dir / "snapshots"
    snap_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")
    (snap_dir / "turn_3.json").write_text("{}", encoding="utf-8")

    restore_calls: list[tuple[str, bool]] = []

    async def fake_restore(event) -> None:
        restore_calls.append((event.session_id, event.apply_saved_model))

    monkeypatch.setattr(engine, "_on_session_restore", fake_restore)

    await engine._on_user_rollback(UserRollback(target_turn=2, revert_changes=True))

    assert tracker.rollback_calls == [({3}, None)]
    assert restore_calls == [("rb_test", False)]


@pytest.mark.asyncio
async def test_restore_loads_winning_recovery_sidecar_after_acquiring_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["primary"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=profile.name,
    )
    await asyncio.to_thread(
        store.save_recovery_session,
        "restore_me",
        {"messages": [Message("user", ["recovery"])], "compressed_msgs": [], "turn_counter": 9},
        agent_profile=profile.name,
    )
    events: list[SessionRestored] = []
    bus = EventBus()
    await bus.subscribe(SessionRestored, lambda event: _collect(events, event))
    engine = AgentEngine(bus, settings=Settings(), state_store=store, agent_registry=_registry(profile))

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = start_profile, operation

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me"))
    finally:
        engine._active_session_guard.release()

    assert engine._turn_number == 9
    assert engine.recovered_from_sidecar is True
    assert len(events) == 1
    assert events[0].recovered_from_sidecar is True


@pytest.mark.asyncio
async def test_restore_snapshots_settings_inside_the_transition_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A settings reload racing the restore must not be silently clobbered:
    the restore takes the transition boundary (shared with the rebuild gate)
    before it snapshots and derives, so a concurrent reload either lands
    before the snapshot or waits and is *audibly denied* — never committed
    and then overwritten by a staged load routed against a stale copy."""
    profile = _profile()
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=profile.name,
    )
    original_settings = Settings()
    restore_settings = Settings(default_approval_mode="manual")
    replacement_settings = Settings(default_approval_mode="auto")
    errors: list[Error] = []
    bus = EventBus()
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = AgentEngine(bus, settings=original_settings, state_store=store, agent_registry=_registry(profile))
    engine._agent_profile = profile

    loop = asyncio.get_running_loop()
    release_restore_load = threading.Event()
    restore_load_entered = asyncio.Event()

    def hanging_restore_load(**kwargs: Any) -> LoadedSettings:
        loop.call_soon_threadsafe(restore_load_entered.set)
        release_restore_load.wait(5)
        return LoadedSettings(settings=restore_settings, provenance={})

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = start_profile, operation
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)
        if workspace is not None:
            engine._workspace = workspace

    monkeypatch.setattr("chrys.service.session.lifecycle.load_settings", hanging_restore_load)
    monkeypatch.setattr(
        "chrys.orchestration.engine.state.controls.load_settings",
        lambda **kwargs: LoadedSettings(settings=replacement_settings, provenance={}),
    )
    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        restore = asyncio.create_task(engine._on_session_restore(SessionRestore(session_id="restore_me")))
        await restore_load_entered.wait()
        reload = asyncio.create_task(engine._on_settings_reload(SettingsReload()))
        for _ in range(5):
            await asyncio.sleep(0)
        # The reload is parked at the shared gate: nothing it does may land
        # between the restore's snapshot and the restore's install.
        assert engine._settings is original_settings
        release_restore_load.set()
        await asyncio.gather(restore, reload)
    finally:
        release_restore_load.set()
        engine._active_session_guard.release()

    # The restore's derivation is in force, the parked reload's never was —
    # its token predates the transition, so it was denied out loud instead of
    # committing first and being silently overwritten.
    assert engine._settings is restore_settings
    assert errors != []


@pytest.mark.asyncio
async def test_failed_restore_load_leaves_the_old_session_generation_intact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transition fence is *prepared* before the load but *committed* only
    after it succeeds: the generation bump and the turn-state invalidation
    (pending retries, injection) live in that one commit, so a load failure
    must release the fence with the generation unchanged — and a reload that
    was parked behind the failed restore proceeds instead of being denied."""
    profile = _profile()
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=profile.name,
    )
    original_settings = Settings()
    replacement_settings = Settings(default_approval_mode="auto")
    errors: list[Error] = []
    bus = EventBus()
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = AgentEngine(bus, settings=original_settings, state_store=store, agent_registry=_registry(profile))
    generation_before = engine._session_generation

    loop = asyncio.get_running_loop()
    release_restore_load = threading.Event()
    restore_load_entered = asyncio.Event()

    def failing_restore_load(**kwargs: Any) -> LoadedSettings:
        loop.call_soon_threadsafe(restore_load_entered.set)
        release_restore_load.wait(5)
        raise RuntimeError("unreadable config")

    monkeypatch.setattr("chrys.service.session.lifecycle.load_settings", failing_restore_load)
    monkeypatch.setattr(
        "chrys.orchestration.engine.state.controls.load_settings",
        lambda **kwargs: LoadedSettings(settings=replacement_settings, provenance={}),
    )

    try:
        restore = asyncio.create_task(engine._on_session_restore(SessionRestore(session_id="restore_me")))
        await restore_load_entered.wait()
        reload = asyncio.create_task(engine._on_settings_reload(SettingsReload()))
        for _ in range(5):
            await asyncio.sleep(0)
        assert engine._settings is original_settings
        release_restore_load.set()
        results = await asyncio.gather(restore, reload, return_exceptions=True)
    finally:
        release_restore_load.set()
        engine._active_session_guard.release()

    # The load failure aborted the restore out loud with the old session intact.
    assert isinstance(results[0], RuntimeError)
    assert results[1] is None
    assert engine._session_generation == generation_before
    # The parked reload went through: an uncommitted fence leaves its token
    # valid, so nothing denied it and its derivation is now in force.
    assert engine._settings is replacement_settings
    # The failure is a bus event, not just a raised exception: interactive
    # callers publish with swallow-and-log delivery, and the Error is what
    # clears their restore loading state.
    assert [error.code for error in errors] == ["session_restore_failed"]
    assert errors[0].session_id == "restore_me"


@pytest.mark.asyncio
async def test_failed_startup_restore_reset_releases_target_without_saving(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved target"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile="Code",
    )
    session_file = store.session_dir("restore_me") / "session.json"
    original_payload = session_file.read_bytes()
    target_cwd = tmp_path / "restored-workspace"
    target_cwd.mkdir()

    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "restore_me"
    engine._workspace = Workspace.from_cwd(str(target_cwd))
    target_lock = engine._active_session_guard.acquire_for_restore("restore_me")
    engine._active_session_guard.install("restore_me", target_lock)

    await engine.reset_after_failed_startup_restore()

    assert engine._session_id is None
    assert engine._workspace == Workspace.from_cwd()
    assert engine._executor is None
    assert engine._mutation_tracker is None
    assert engine._suppress_save is False
    assert not engine._active_session_guard.owns("restore_me")
    assert session_file.read_bytes() == original_payload


@pytest.mark.asyncio
async def test_failed_startup_restore_reset_restores_save_flag_when_cleanup_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=JsonFileStateStore(tmp_path))

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    def fail_cleanup() -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "_reset_turn_runtime_after_session_shutdown", fail_cleanup)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        await engine.reset_after_failed_startup_restore()

    assert engine._suppress_save is False


@pytest.mark.asyncio
async def test_failed_startup_restore_reset_rederives_settings_for_the_fallback_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback startup runs in the process cwd, so whatever the failed target
    left installed describes the wrong project trust domain."""
    replacement_settings = Settings(default_approval_mode="auto")
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=JsonFileStateStore(tmp_path))
    load_kwargs: dict[str, Any] = {}

    def fake_load(**kwargs: Any) -> LoadedSettings:
        load_kwargs.update(kwargs)
        return LoadedSettings(settings=replacement_settings, provenance={})

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    monkeypatch.setattr("chrys.service.session.lifecycle.load_settings", fake_load)
    monkeypatch.setattr(engine, "shutdown", fake_shutdown)

    await engine.reset_after_failed_startup_restore()

    assert load_kwargs["project_root"] == Path(os.getcwd())
    assert engine._settings is replacement_settings


@pytest.mark.asyncio
async def test_failed_startup_restore_rederivation_failure_keeps_the_installed_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best-effort: the reset must never mask the original failure, and the
    live settings remain a usable baseline without the re-derivation."""
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=JsonFileStateStore(tmp_path))
    installed = engine._loaded_settings

    def fail_load(**_kwargs: Any) -> LoadedSettings:
        raise OSError("config unreadable")

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    monkeypatch.setattr("chrys.service.session.lifecycle.load_settings", fail_load)
    monkeypatch.setattr(engine, "shutdown", fake_shutdown)

    await engine.reset_after_failed_startup_restore()

    assert engine._loaded_settings is installed


@pytest.mark.asyncio
async def test_ignore_recovery_restore_discards_recovery_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rollback-style restores must not consume a live/stale checkpoint sidecar."""
    profile = _profile()
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["primary"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=profile.name,
    )
    await asyncio.to_thread(
        store.save_recovery_session,
        "restore_me",
        {"messages": [Message("user", ["recovery"])], "compressed_msgs": [], "turn_counter": 9},
        agent_profile=profile.name,
    )

    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store, agent_registry=_registry(profile))
    active_lock = engine._active_session_guard.acquire_for_restore("restore_me")
    engine._active_session_guard.install("restore_me", active_lock)

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = start_profile, operation

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me", ignore_recovery=True))
    finally:
        engine._active_session_guard.release()

    assert engine._turn_number == 1
    assert not (store.session_dir("restore_me") / SESSION_RECOVERY_FILE_NAME).exists()


@pytest.mark.asyncio
async def test_ignore_recovery_restore_removes_recovery_only_session_dir(tmp_path: Path) -> None:
    bus = EventBus()
    errors: list[Error] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    store = JsonFileStateStore(tmp_path)
    await asyncio.to_thread(
        store.save_recovery_session,
        "restore_me",
        {"messages": [Message("user", ["recovery"])], "compressed_msgs": [], "turn_counter": 9},
        agent_profile="Code",
    )
    session_dir = store.session_dir("restore_me")
    assert session_dir.is_dir()

    engine = AgentEngine(bus, settings=Settings(), state_store=store)
    await engine._on_session_restore(SessionRestore(session_id="restore_me", ignore_recovery=True))

    assert [event.code for event in errors] == ["session_not_found"]
    assert not session_dir.exists()


@pytest.mark.asyncio
async def test_ignore_recovery_restore_keeps_current_recovery_only_session_dir(tmp_path: Path) -> None:
    bus = EventBus()
    errors: list[Error] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    store = JsonFileStateStore(tmp_path)
    await asyncio.to_thread(
        store.save_recovery_session,
        "restore_me",
        {"messages": [Message("user", ["recovery"])], "compressed_msgs": [], "turn_counter": 9},
        agent_profile="Code",
    )
    session_dir = store.session_dir("restore_me")

    engine = AgentEngine(bus, settings=Settings(), state_store=store)
    active_lock = engine._active_session_guard.acquire_for_restore("restore_me")
    engine._active_session_guard.install("restore_me", active_lock)
    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me", ignore_recovery=True))
    finally:
        engine._active_session_guard.release()

    assert [event.code for event in errors] == ["session_not_found"]
    assert not (session_dir / SESSION_RECOVERY_FILE_NAME).exists()
    assert session_dir.exists()


@pytest.mark.asyncio
async def test_current_recovered_session_restore_keeps_recovery_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["primary"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=profile.name,
    )
    await asyncio.to_thread(
        store.save_recovery_session,
        "restore_me",
        {"messages": [Message("user", ["recovery"])], "compressed_msgs": [], "turn_counter": 9},
        agent_profile=profile.name,
    )

    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store, agent_registry=_registry(profile))
    active_lock = engine._active_session_guard.acquire_for_restore("restore_me")
    engine._active_session_guard.install("restore_me", active_lock)
    engine._recovered_from_sidecar = True

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = start_profile, operation

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me"))
    finally:
        engine._active_session_guard.release()

    assert engine._turn_number == 9
    assert engine.recovered_from_sidecar is True
    assert (store.session_dir("restore_me") / SESSION_RECOVERY_FILE_NAME).exists()


@pytest.mark.asyncio
async def test_session_restore_keeps_admission_closed_through_restored_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = _profile()
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["primary"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=profile.name,
    )
    bus = EventBus()
    engine = AgentEngine(bus, settings=Settings(), state_store=store, agent_registry=_registry(profile))
    restored_event_closed_states: list[bool] = []
    start_closed_states: list[bool] = []
    shutdown_closed_states: list[bool] = []
    await bus.subscribe(
        SessionRestored,
        lambda _event: _collect(restored_event_closed_states, engine._turn_state.prompt_admission_closed),
    )

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache
        shutdown_closed_states.append(engine._turn_state.prompt_admission_closed)

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = start_profile, operation
        start_closed_states.append(engine._turn_state.prompt_admission_closed)

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me"))
    finally:
        engine._active_session_guard.release()

    assert shutdown_closed_states == [True]
    assert start_closed_states == [True]
    assert restored_event_closed_states == [True]
    assert engine._turn_state.prompt_admission_closed is False


@pytest.mark.asyncio
async def test_new_session_keeps_admission_closed_through_restart(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _profile()
    engine = AgentEngine(EventBus(), settings=Settings())
    engine._agent_profile = profile
    states: list[tuple[str, bool]] = []

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache
        states.append(("shutdown", engine._turn_state.prompt_admission_closed))

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = start_profile, operation
        states.append(("start", engine._turn_state.prompt_admission_closed))

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_new_session(SessionNew())

    assert states == [("shutdown", True), ("start", True)]
    assert engine.session_generation == 1
    assert engine._turn_state.prompt_admission_closed is False


@pytest.mark.asyncio
async def test_new_session_uses_profile_after_transition_permit(monkeypatch: pytest.MonkeyPatch) -> None:
    old_profile = _profile("Code")
    new_profile = _profile("Explore")
    engine = AgentEngine(EventBus(), settings=Settings())
    engine._agent_profile = old_profile
    started_profiles: list[str] = []

    async def fake_begin_session_transition(operation: str) -> str:
        assert operation == "new_session"
        engine._agent_profile = new_profile
        return "session:new_session:test"

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "new_session"
        started_profiles.append(start_profile.name)

    monkeypatch.setattr(engine, "_begin_session_transition", fake_begin_session_transition)
    monkeypatch.setattr(engine, "_finish_session_transition", lambda _owner: None)
    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    await engine._on_new_session(SessionNew())

    assert started_profiles == ["Explore"]


@pytest.mark.asyncio
async def test_prepared_startup_profile_is_blank_session_restore_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fallback_profile = _profile("Code")
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["primary"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile="",
    )
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store, agent_registry=_registry())
    await engine.prepare(fallback_profile)
    started_profiles: list[str] = []

    async def fake_begin_session_transition(operation: str) -> str:
        assert operation == "restore"
        return "session:restore:test"

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "restore"
        started_profiles.append(start_profile.name)

    monkeypatch.setattr(engine, "_begin_session_transition", fake_begin_session_transition)
    monkeypatch.setattr(engine, "_finish_session_transition", lambda _owner: None)
    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me"))
    finally:
        engine._active_session_guard.release()

    assert started_profiles == ["Code"]


@pytest.mark.asyncio
async def test_session_restore_live_profile_fallback_is_sampled_after_transition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_profile = _profile("Code")
    new_profile = _profile("Explore")
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["primary"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile="",
    )
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store, agent_registry=_registry())
    engine._agent_profile = old_profile
    started_profiles: list[str] = []

    async def fake_prepare_session_transition(operation: str) -> str:
        # The boundary acquisition is the exclusion point: a profile switch
        # landing just before it must be what the fallback then samples.
        assert operation == "restore"
        engine._agent_profile = new_profile
        return "session:restore:test"

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "restore"
        started_profiles.append(start_profile.name)

    monkeypatch.setattr(engine, "_prepare_session_transition", fake_prepare_session_transition)
    monkeypatch.setattr(engine, "_commit_session_transition", lambda _owner: None)
    monkeypatch.setattr(engine, "_finish_session_transition", lambda _owner: None)
    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me"))
    finally:
        engine._active_session_guard.release()

    assert started_profiles == ["Explore"]


@pytest.mark.asyncio
async def test_session_restore_resolves_saved_agent_by_id_after_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    renamed_profile = _profile("Renamed")
    renamed_profile.id = "stable-agent-id"
    stale_name_collision = _profile("OldName")
    stale_name_collision.id = "different-agent-id"
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile="OldName",
        agent_profile_id="stable-agent-id",
    )
    engine = AgentEngine(
        EventBus(),
        settings=Settings(),
        state_store=store,
        agent_registry=_registry(stale_name_collision, renamed_profile),
    )
    started_profiles: list[AgentProfile] = []

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "restore"
        started_profiles.append(start_profile)

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me"))
    finally:
        engine._active_session_guard.release()

    assert started_profiles == [renamed_profile]


@pytest.mark.asyncio
async def test_session_restore_unresolved_explicit_agent_stops_instead_of_falling_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_profile = _profile("Current")
    first_available = _profile("First")
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile="Saved",
    )
    bus = EventBus()
    errors: list[Error] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = AgentEngine(
        bus,
        settings=Settings(),
        state_store=store,
        agent_registry=_registry(first_available),
    )
    engine._agent_profile = current_profile
    start = AsyncMock()
    monkeypatch.setattr(engine, "start", start)

    await engine._on_session_restore(SessionRestore(session_id="restore_me", profile_name="Missing"))

    start.assert_not_awaited()
    resolution_errors = [event for event in errors if event.code == "requested_agent_profile_unresolved"]
    assert len(resolution_errors) == 1
    _assert_display_message(
        resolution_errors[0],
        "restore.requested_agent_profile_unresolved",
        {"profile": "Missing"},
    )
    assert not engine._active_session_guard.owns("restore_me")


@pytest.mark.asyncio
async def test_session_restore_stale_agent_id_does_not_match_reused_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_profile = _profile("Current")
    replacement_profile = _profile("Legacy")
    replacement_profile.id = "new-profile-id"
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=replacement_profile.name,
        # Deliberately equals the replacement's name: an id-only lookup must
        # not reinterpret this stale identity as a generic name selector.
        agent_profile_id=replacement_profile.name,
    )
    bus = EventBus()
    warnings: list[Warning] = []
    await bus.subscribe(Warning, lambda event: _collect(warnings, event))
    engine = AgentEngine(
        bus,
        settings=Settings(),
        state_store=store,
        agent_registry=_registry(replacement_profile),
    )
    engine._agent_profile = current_profile
    started_profiles: list[AgentProfile] = []

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "restore"
        started_profiles.append(start_profile)

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me"))
    finally:
        engine._active_session_guard.release()

    assert started_profiles == [current_profile]
    resolution_warnings = [event for event in warnings if event.code == "saved_agent_profile_unresolved"]
    assert len(resolution_warnings) == 1
    _assert_display_message(
        resolution_warnings[0],
        "restore.agent_profile_unresolved_using_current",
        {"saved": "Legacy", "current": "Current"},
    )


@pytest.mark.asyncio
async def test_session_restore_unresolved_agent_id_without_current_agent_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement_profile = _profile("Legacy")
    replacement_profile.id = "new-profile-id"
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=replacement_profile.name,
        agent_profile_id="deleted-profile-id",
    )
    bus = EventBus()
    errors: list[Error] = []
    await bus.subscribe(Error, lambda event: _collect(errors, event))
    engine = AgentEngine(
        bus,
        settings=Settings(),
        state_store=store,
        agent_registry=_registry(replacement_profile),
    )
    start = AsyncMock()
    monkeypatch.setattr(engine, "start", start)

    await engine._on_session_restore(SessionRestore(session_id="restore_me"))

    start.assert_not_awaited()
    resolution_errors = [event for event in errors if event.code == "saved_agent_profile_unresolved"]
    assert len(resolution_errors) == 1
    _assert_display_message(
        resolution_errors[0],
        "restore.agent_profile_unresolved",
        {"saved": "Legacy"},
    )
    assert not engine._active_session_guard.owns("restore_me")


@pytest.mark.asyncio
async def test_session_restore_ambiguous_agent_id_uses_saved_name_without_current_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_match = _profile("First")
    first_match.id = "shared-agent-id"
    saved_profile = _profile("Second")
    saved_profile.id = "shared-agent-id"
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=saved_profile.name,
        agent_profile_id=saved_profile.id,
    )
    engine = AgentEngine(
        EventBus(),
        settings=Settings(),
        state_store=store,
        agent_registry=_registry(first_match, saved_profile),
    )
    started_profiles: list[AgentProfile] = []

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "restore"
        started_profiles.append(start_profile)

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me"))
    finally:
        engine._active_session_guard.release()

    assert started_profiles == [saved_profile]


@pytest.mark.asyncio
async def test_session_restore_missing_saved_profile_keeps_current_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current_profile = _profile("Current")
    first_available = _profile("First")
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile="Deleted",
    )
    engine = AgentEngine(
        EventBus(),
        settings=Settings(),
        state_store=store,
        agent_registry=_registry(first_available),
    )
    engine._agent_profile = current_profile
    started_profiles: list[AgentProfile] = []

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "restore"
        started_profiles.append(start_profile)

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me"))
    finally:
        engine._active_session_guard.release()

    assert started_profiles == [current_profile]


@pytest.mark.asyncio
async def test_session_restore_missing_saved_profile_without_current_agent_uses_first_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_available = _profile("First")
    second_available = _profile("Second")
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile="Deleted",
    )
    engine = AgentEngine(
        EventBus(),
        settings=Settings(),
        state_store=store,
        agent_registry=_registry(first_available, second_available),
    )
    started_profiles: list[AgentProfile] = []

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "restore"
        started_profiles.append(start_profile)

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me"))
    finally:
        engine._active_session_guard.release()

    assert started_profiles == [first_available]


@pytest.mark.asyncio
async def test_tui_session_restore_reapplies_saved_model_without_touching_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_model_environment = os.environ.get("CHRYS_MODEL_PROFILE")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    dotenv = config_dir / ".env"
    original_dotenv = b'CHRYS_MODEL_PROFILE="global-model"\nUNCHANGED="yes"\n'
    dotenv.write_bytes(original_dotenv)
    platform = replace(get_platform(), config_dir=config_dir, data_dir=config_dir)
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: platform)
    monkeypatch.delenv("CHRYS_MODEL_PROFILE", raising=False)

    saved_model = ModelProfile(id="saved-model", name="Saved", model_id="gpt-saved")
    agent_profile = _profile("Code")
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=agent_profile.name,
        model_profile_id=saved_model.id,
    )
    settings = Settings(model_profile="global-model")
    engine = AgentEngine(
        EventBus(),
        settings=settings,
        state_store=store,
        agent_registry=_registry(agent_profile),
        model_registry=_model_registry(saved_model),
    )
    started_model_ids: list[str] = []

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert start_profile is agent_profile
        assert operation == "restore"
        # The reapplied model travels on the staged load; a successful build
        # commits it, so a fake simulating one must install it.
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)
        started_model_ids.append(engine._settings.model_profile)

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me", apply_saved_model=True))
    finally:
        engine._active_session_guard.release()

    assert started_model_ids == [saved_model.id]
    assert engine._settings.model_profile == saved_model.id
    assert engine._settings.model_profile_override == ""
    assert os.environ["CHRYS_MODEL_PROFILE"] == saved_model.id
    assert Settings.from_env().model_profile == saved_model.id
    assert dotenv.read_bytes() == original_dotenv
    _restore_model_profile_environment(previous_model_environment)


@pytest.mark.asyncio
async def test_session_restore_derives_settings_from_the_target_sessions_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The restore crosses into the target's project trust domain: its root —
    not the current session's — is what the staged load is derived from."""
    agent_profile = _profile("Code")
    target_cwd = tmp_path / "restored-workspace"
    target_cwd.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=agent_profile.name,
        primary_cwd=str(target_cwd),
    )
    replacement_settings = Settings(default_approval_mode="auto")
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store, agent_registry=_registry(agent_profile))
    load_kwargs: dict[str, Any] = {}

    def fake_load(**kwargs: Any) -> LoadedSettings:
        load_kwargs.update(kwargs)
        return LoadedSettings(settings=replacement_settings, provenance={})

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        _profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "restore"
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    monkeypatch.setattr("chrys.service.session.lifecycle.load_settings", fake_load)
    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me"))
    finally:
        engine._active_session_guard.release()

    assert load_kwargs["project_root"] == Path(str(target_cwd))
    assert engine._settings is replacement_settings


@pytest.mark.asyncio
async def test_session_restore_hydrates_the_workspace_baseline_with_the_target_sessions_notice_setting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The baseline restore runs before the build commits the staged settings:
    with the notice on in the live session but off for the restored target,
    it must read the target's value and skip the root probes."""
    agent_profile = _profile("Code")
    target_cwd = tmp_path / "restored-workspace"
    target_cwd.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=agent_profile.name,
        primary_cwd=str(target_cwd),
    )
    engine = AgentEngine(
        EventBus(),
        settings=Settings(workspace_change_notice=True),
        state_store=store,
        agent_registry=_registry(agent_profile),
    )
    assert engine._settings.workspace_change_notice is True

    def fake_load(**_kwargs: Any) -> LoadedSettings:
        return LoadedSettings(settings=Settings(workspace_change_notice=False), provenance={})

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        _profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "restore"
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    probed: list[Path] = []

    def _record(path: Path, *_args: Any, **_kwargs: Any) -> Path | None:
        probed.append(path)
        return None

    monkeypatch.setattr("chrys.service.session.lifecycle.load_settings", fake_load)
    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)
    # Installed after construction: the live engine's own retarget legitimately probes.
    monkeypatch.setattr(workspace_changes, "resolve_git_root", _record)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me"))
    finally:
        engine._active_session_guard.release()

    assert engine._settings.workspace_change_notice is False
    assert probed == []


@pytest.mark.asyncio
async def test_session_restore_aborts_before_teardown_when_the_settings_load_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loaded before anything is torn down: an unreadable config file must
    abort the restore with the current session intact."""
    agent_profile = _profile("Code")
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=agent_profile.name,
    )
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store, agent_registry=_registry(agent_profile))
    engine._session_id = "current-session"
    installed = engine._loaded_settings
    shutdowns: list[bool] = []

    def fail_load(**_kwargs: Any) -> LoadedSettings:
        raise OSError("config unreadable")

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache
        shutdowns.append(True)

    monkeypatch.setattr("chrys.service.session.lifecycle.load_settings", fail_load)
    monkeypatch.setattr(engine, "shutdown", fake_shutdown)

    with pytest.raises(OSError, match="config unreadable"):
        await engine._on_session_restore(SessionRestore(session_id="restore_me"))

    assert shutdowns == []
    assert engine._session_id == "current-session"
    assert engine._loaded_settings is installed


def test_saved_model_restore_clears_the_previous_sessions_pin() -> None:
    """The pin outranks the plain selection, so leaving it standing defeats the restore.

    Switching models pins the whole selection — ``model_profile_override``
    included, and the resolver consults that field first. A restore that writes
    only ``model_profile`` claims the saved model while the rebuild goes on
    resolving the previous session's.
    """
    previous_model_environment = os.environ.get("CHRYS_MODEL_PROFILE")
    saved_model = ModelProfile(id="saved-model", name="Saved", model_id="gpt-saved")
    engine = AgentEngine(
        EventBus(),
        settings=Settings(
            model_profile="pinned-model",
            model_profile_override="pinned-model",
            model_profile_override_sub_agents=True,
        ),
        model_registry=_model_registry(saved_model),
    )

    staged, token = session_lifecycle._reapply_saved_model_profile(
        engine,
        _session_meta(model_profile_id=saved_model.id),
        _profile("Code"),
        engine._loaded_settings,
    )

    assert token is not None
    assert staged.settings.model_profile == saved_model.id
    assert staged.settings.model_profile_override == ""
    assert staged.settings.model_profile_override_sub_agents is False
    # Nothing installed: the selection goes live with the build's commit.
    assert engine._settings.model_profile == "pinned-model"
    _restore_model_profile_environment(previous_model_environment)


def test_saved_model_restore_moves_settings_and_provenance_together() -> None:
    """The restored model is a session-scoped decision and has to say so.

    A bare ``replace`` on the settings would leave the staged load's two
    halves disagreeing, which the build's commit turns into a live
    inconsistency.
    """
    previous_model_environment = os.environ.get("CHRYS_MODEL_PROFILE")
    saved_model = ModelProfile(id="saved-model", name="Saved", model_id="gpt-saved")
    engine = AgentEngine(
        EventBus(),
        settings=Settings(model_profile="global-model"),
        model_registry=_model_registry(saved_model),
    )

    staged, token = session_lifecycle._reapply_saved_model_profile(
        engine,
        _session_meta(model_profile_id=saved_model.id),
        _profile("Code"),
        engine._loaded_settings,
    )

    assert token is not None
    assert staged.settings.model_profile == saved_model.id
    assert staged.source_for("model.profile.active").layer is Source.SESSION
    # The transform stays staged: the live settings have not moved.
    assert engine._settings.model_profile == "global-model"
    _restore_model_profile_environment(previous_model_environment)


@pytest.mark.asyncio
async def test_a_failed_reload_after_a_saved_model_restore_keeps_the_restored_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rollback restores ``_loaded_settings``, so it had to have been updated.

    Otherwise a reload that fails puts the pre-restore model back into
    ``_settings`` while the executor the failure kept running is still bound to
    the saved one — the two disagreeing about which model is in use.
    """
    previous_model_environment = os.environ.get("CHRYS_MODEL_PROFILE")
    saved_model = ModelProfile(id="saved-model", name="Saved", model_id="gpt-saved")
    engine = AgentEngine(
        EventBus(),
        settings=Settings(model_profile="global-model"),
        model_registry=_model_registry(saved_model),
    )
    engine._agent_profile = _profile("Code")
    engine._executor = None
    staged, _token = session_lifecycle._reapply_saved_model_profile(
        engine,
        _session_meta(model_profile_id=saved_model.id),
        _profile("Code"),
        engine._loaded_settings,
    )
    # The restore build's commit, minimally: the reapplied selection goes live.
    engine._settings_handle.install(staged)
    engine._model_profile_pinned = True

    async def failing_start(
        profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = profile, operation
        raise RuntimeError("rebuild failed")

    monkeypatch.setattr(engine, "start", failing_start)

    with pytest.raises(RuntimeError):
        await engine._on_settings_reload(SettingsReload())

    assert engine._settings.model_profile == saved_model.id
    assert engine._loaded_settings.settings.model_profile == saved_model.id
    _restore_model_profile_environment(previous_model_environment)


@pytest.mark.parametrize("case", ["empty", "missing", "unselectable", "current"])
def test_saved_model_restore_short_circuits_without_side_effects(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_model_environment = os.environ.get("CHRYS_MODEL_PROFILE")
    saved_model = ModelProfile(id="saved-model", name="Saved", model_id="gpt-saved")
    profiles: tuple[ModelProfile, ...]
    settings_model = "current-model"
    saved_model_id = saved_model.id
    if case == "empty":
        profiles = (saved_model,)
        saved_model_id = ""
    elif case == "missing":
        profiles = ()
    elif case == "unselectable":
        profiles = (replace(saved_model, model_id=""),)
    else:
        profiles = (saved_model,)
        settings_model = saved_model.id

    settings = Settings(model_profile=settings_model, model_profile_override="existing-override")
    engine = AgentEngine(EventBus(), settings=settings, model_registry=_model_registry(*profiles))
    os.environ["CHRYS_MODEL_PROFILE"] = "environment-before"

    staged, token = session_lifecycle._reapply_saved_model_profile(
        engine,
        _session_meta(model_profile_id=saved_model_id),
        _profile("Code"),
        engine._loaded_settings,
    )

    assert token is None
    assert staged is engine._loaded_settings
    assert engine._settings is settings
    assert engine._settings.model_profile_override == "existing-override"
    assert os.environ["CHRYS_MODEL_PROFILE"] == "environment-before"
    _restore_model_profile_environment(previous_model_environment)


def test_saved_model_restore_skips_registered_agent_binding_without_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_model_environment = os.environ.get("CHRYS_MODEL_PROFILE")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    dotenv = config_dir / ".env"
    original_dotenv = b'CHRYS_MODEL_PROFILE="global-model"\n'
    dotenv.write_bytes(original_dotenv)
    platform = replace(get_platform(), config_dir=config_dir, data_dir=config_dir)
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: platform)
    saved_model = ModelProfile(id="saved-model", name="Saved", model_id="gpt-saved")
    bound_model = ModelProfile(id="bound-model", name="Bound", model_id="gpt-bound")
    agent_profile = replace(_profile("Code"), model=ModelConfig(profile_id=bound_model.id))
    settings = Settings(model_profile="global-model", model_profile_override="existing-override")
    engine = AgentEngine(
        EventBus(),
        settings=settings,
        model_registry=_model_registry(saved_model, bound_model),
    )
    os.environ["CHRYS_MODEL_PROFILE"] = "environment-before"

    staged, token = session_lifecycle._reapply_saved_model_profile(
        engine,
        _session_meta(model_profile_id=saved_model.id),
        agent_profile,
        engine._loaded_settings,
    )

    assert token is None
    assert staged is engine._loaded_settings
    assert engine._settings is settings
    assert engine._settings.model_profile_override == "existing-override"
    assert os.environ["CHRYS_MODEL_PROFILE"] == "environment-before"
    assert dotenv.read_bytes() == original_dotenv
    _restore_model_profile_environment(previous_model_environment)


@pytest.mark.asyncio
async def test_session_restore_default_gate_does_not_apply_saved_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_model_environment = os.environ.get("CHRYS_MODEL_PROFILE")
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "environment-before")
    saved_model = ModelProfile(id="saved-model", name="Saved", model_id="gpt-saved")
    agent_profile = _profile("Code")
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=agent_profile.name,
        model_profile_id=saved_model.id,
    )
    settings = Settings(model_profile="current-model", model_profile_override="existing-override")
    engine = AgentEngine(
        EventBus(),
        settings=settings,
        state_store=store,
        agent_registry=_registry(agent_profile),
        model_registry=_model_registry(saved_model),
    )
    started_model_ids: list[str] = []

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        _profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "restore"
        started_model_ids.append(engine._settings.model_profile)

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_me"))
    finally:
        engine._active_session_guard.release()

    assert started_model_ids == ["current-model"]
    assert engine._settings is settings
    assert engine._settings.model_profile_override == "existing-override"
    assert os.environ["CHRYS_MODEL_PROFILE"] == "environment-before"
    _restore_model_profile_environment(previous_model_environment)


@pytest.mark.asyncio
async def test_saved_model_restore_rolls_back_when_start_fails_before_executor_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_model_environment = os.environ.get("CHRYS_MODEL_PROFILE")
    monkeypatch.delenv("CHRYS_MODEL_PROFILE", raising=False)
    saved_model = ModelProfile(id="saved-model", name="Saved", model_id="gpt-saved")
    agent_profile = _profile("Code")
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=agent_profile.name,
        model_profile_id=saved_model.id,
    )
    settings = Settings(model_profile="old-model", model_profile_override="existing-override")
    engine = AgentEngine(
        EventBus(),
        settings=settings,
        state_store=store,
        agent_registry=_registry(agent_profile),
        model_registry=_model_registry(saved_model),
    )

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fail_start(
        _profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "restore"
        # The reapplied model arrives staged; a build that fails before its
        # commit never installs it, so the live settings must not have moved.
        assert staged_loaded is not None
        assert staged_loaded.settings.model_profile == saved_model.id
        assert engine._settings.model_profile == "old-model"
        raise RuntimeError("start failed")

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fail_start)

    try:
        with pytest.raises(RuntimeError, match="start failed"):
            await engine._on_session_restore(SessionRestore(session_id="restore_me", apply_saved_model=True))
    finally:
        engine._active_session_guard.release()

    assert engine._settings is settings
    assert engine._settings.model_profile_override == "existing-override"
    assert "CHRYS_MODEL_PROFILE" not in os.environ
    _restore_model_profile_environment(previous_model_environment)


@pytest.mark.asyncio
async def test_saved_model_restore_does_not_roll_back_after_executor_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_model_environment = os.environ.get("CHRYS_MODEL_PROFILE")
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "environment-before")
    saved_model = ModelProfile(id="saved-model", name="Saved", model_id="gpt-saved")
    agent_profile = _profile("Code")
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=agent_profile.name,
        model_profile_id=saved_model.id,
    )
    settings = Settings(model_profile="old-model", model_profile_override="existing-override")
    engine = AgentEngine(
        EventBus(),
        settings=settings,
        state_store=store,
        agent_registry=_registry(agent_profile),
        model_registry=_model_registry(saved_model),
    )
    replacement_executor = MagicMock()

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fail_after_install(
        _profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "restore"
        # The build's commit: settings and executor go live together, and the
        # failure lands after it.
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)
        engine._executor = replacement_executor
        raise RuntimeError("post-install failure")

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fail_after_install)

    try:
        with pytest.raises(RuntimeError, match="post-install failure"):
            await engine._on_session_restore(SessionRestore(session_id="restore_me", apply_saved_model=True))
    finally:
        engine._active_session_guard.release()

    assert engine._executor is replacement_executor
    # The commit installs the whole selection, so the stale override went
    # with it — and a post-install failure must not resurrect either field.
    assert engine._settings.model_profile == saved_model.id
    assert engine._settings.model_profile_override == ""
    assert os.environ["CHRYS_MODEL_PROFILE"] == saved_model.id
    _restore_model_profile_environment(previous_model_environment)


@pytest.mark.asyncio
async def test_saved_model_restore_stays_applied_after_successful_start_and_later_hydrate_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_model_environment = os.environ.get("CHRYS_MODEL_PROFILE")
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "environment-before")
    saved_model = ModelProfile(id="saved-model", name="Saved", model_id="gpt-saved")
    agent_profile = _profile("Code")
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_me",
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=agent_profile.name,
        model_profile_id=saved_model.id,
    )
    settings = Settings(model_profile="old-model", model_profile_override="existing-override")
    engine = AgentEngine(
        EventBus(),
        settings=settings,
        state_store=store,
        agent_registry=_registry(agent_profile),
        model_registry=_model_registry(saved_model),
    )

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        _profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        assert operation == "restore"
        # A successful build commits the staged load.
        if staged_loaded is not None:
            engine._settings_handle.install(staged_loaded)

    def fail_usage_event(*, session_id: str | None = None) -> None:
        _ = session_id
        raise RuntimeError("late hydrate failure")

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)
    monkeypatch.setattr(engine, "_make_usage_event", fail_usage_event)

    try:
        with pytest.raises(RuntimeError, match="late hydrate failure"):
            await engine._on_session_restore(SessionRestore(session_id="restore_me", apply_saved_model=True))
    finally:
        engine._active_session_guard.release()

    # The commit installs the whole selection, so the stale override went
    # with it — and a failure after a successful start must not roll it back.
    assert engine._settings.model_profile == saved_model.id
    assert engine._settings.model_profile_override == ""
    assert os.environ["CHRYS_MODEL_PROFILE"] == saved_model.id
    _restore_model_profile_environment(previous_model_environment)


@pytest.mark.asyncio
async def test_successful_primary_save_clears_recovered_sidecar_marker(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "restore_me"
    engine._recovered_from_sidecar = True

    class _Executor:
        def __init__(self) -> None:
            self.history_state = {
                "messages": [Message("user", ["saved"])],
                "compressed_msgs": [],
                "turn_counter": 10,
            }

    engine._executor = _Executor()  # type: ignore[assignment]
    await asyncio.to_thread(
        store.save_recovery_session,
        "restore_me",
        {"messages": [Message("user", ["recovery"])], "compressed_msgs": [], "turn_counter": 9},
    )

    await engine._save_current_session()

    assert engine.recovered_from_sidecar is False
    assert not (store.session_dir("restore_me") / SESSION_RECOVERY_FILE_NAME).exists()


@pytest.mark.asyncio
async def test_save_current_session_finalizes_pending_records_only_after_written(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "pending-save"
    session_dir = store.session_dir("pending-save")
    tools = SubAgentTools(event_bus=EventBus(), session_id="pending-save", session_dir=session_dir)
    engine._sub_agent_tools = tools

    class _Executor:
        def __init__(self, messages: list[Message]) -> None:
            self.history_state = {"messages": messages, "compressed_msgs": [], "turn_counter": 1}

    pending_file = session_dir / "sub_agents" / "pending" / "done.json"
    pending_file.parent.mkdir(parents=True)
    # Write the record the way production does (owner-only). finalize_pending_cleanups
    # deletes it through secure_unlink_owner_verified, which requires an owner-verified
    # file; a plain write_text on an elevated Windows runner is owned by Administrators
    # (the token's default owner), so the secure unlink would correctly refuse it.
    atomic_write_owner_only_text(pending_file, "{}")
    tools.queue_pending_cleanup(pending_file)
    engine._executor = _Executor([Message("user", ["saved"])])  # type: ignore[assignment]

    assert await engine._save_current_session() is True
    assert not pending_file.exists()

    skipped_file = session_dir / "sub_agents" / "pending" / "skipped.json"
    atomic_write_owner_only_text(skipped_file, "{}")
    tools.queue_pending_cleanup(skipped_file)
    engine._executor = _Executor([])  # type: ignore[assignment]

    assert await engine._save_current_session() is False
    assert skipped_file.exists()


@pytest.mark.asyncio
async def test_save_current_session_flushes_buffered_otel_lines_after_materializing_root(tmp_path: Path) -> None:
    previous_sink = get_otel_sink()
    sink = OtelSessionSink(write_files=True)
    set_otel_sink(sink)
    try:
        store = JsonFileStateStore(tmp_path)
        engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
        engine._session_id = "otel-save"
        session_dir = store.session_dir("otel-save")

        class _Executor:
            def __init__(self) -> None:
                self.history_state = {
                    "messages": [Message("user", ["saved"])],
                    "compressed_msgs": [],
                    "turn_counter": 1,
                }

        engine._executor = _Executor()  # type: ignore[assignment]
        sink.activate("otel-save", session_dir=session_dir)
        sink.write_logs(['{"before_save": true}\n'])

        assert not session_dir.exists()

        assert await engine._save_current_session() is True

        logs = (session_dir / "otel" / "logs.jsonl").read_text(encoding="utf-8")
        assert "before_save" in logs
    finally:
        set_otel_sink(previous_sink)


@pytest.mark.asyncio
async def test_shutdown_timeout_preserves_recovery_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "shutdown_timeout"
    sidecar = store.session_dir("shutdown_timeout") / SESSION_RECOVERY_FILE_NAME
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(engine_module, "_SHUTDOWN_POST_RUN_TIMEOUT_SECONDS", 0.01)

    class _HangingExecutor:
        is_running = True

        def __init__(self) -> None:
            self.interrupt_called = False
            self.close_called = False

        async def interrupt(self) -> None:
            self.interrupt_called = True

        async def close(self) -> None:
            self.close_called = True

    executor = _HangingExecutor()
    engine._executor = executor  # type: ignore[assignment]
    save_suppress_values: list[bool] = []

    async def fake_save_current_session() -> None:
        save_suppress_values.append(engine._suppress_save)
        if not engine._suppress_save:
            sidecar.unlink(missing_ok=True)

    async def never_finishes() -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr(engine, "_save_current_session", fake_save_current_session)
    engine._turn_state.run_task = asyncio.create_task(never_finishes())

    await engine.shutdown(close_mcp_cache=False)

    assert engine._shutdown_used_cancel_fallback is True
    assert save_suppress_values == [True]
    assert sidecar.exists()
    assert engine._suppress_save is False
    assert executor.interrupt_called is True
    assert executor.close_called is True


@pytest.mark.asyncio
async def test_reset_session_to_welcome_deletes_state_and_reports_missing_profile(tmp_path: Path) -> None:
    events: list[Error] = []
    bus = EventBus()
    await bus.subscribe(Error, lambda event: _collect(events, event))
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(bus, settings=Settings(), state_store=store)
    engine._session_id = "reset_me"
    engine._turn_number = 7
    engine._runtime_meta.total_session_tokens = 11
    engine._runtime_meta.total_session_input_tokens = 5
    engine._runtime_meta.total_session_output_tokens = 6
    engine._runtime_meta.last_usage_details = {"total_token_count": 11}
    engine._fsm.try_transition(Trigger.START)
    engine._fsm.try_transition(Trigger.USER_MESSAGE)
    engine._fsm.try_transition(Trigger.RUN_FAILED)
    engine._shutting_down = True

    session_dir = store.session_dir("reset_me")
    snapshots_dir = session_dir / "snapshots"
    snapshots_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")
    (session_dir / "session.json.bak").write_text("{}", encoding="utf-8")
    (session_dir / SESSION_RECOVERY_FILE_NAME).write_text("{}", encoding="utf-8")
    (snapshots_dir / "turn_2.json").write_text("{}", encoding="utf-8")

    reset_succeeded = await engine._reset_session_to_welcome("reset_me")

    assert reset_succeeded is True
    assert not (session_dir / "session.json").exists()
    assert not (session_dir / "session.json.bak").exists()
    assert not (session_dir / SESSION_RECOVERY_FILE_NAME).exists()
    assert not snapshots_dir.exists()
    assert not session_dir.exists()
    assert engine._session_id == "reset_me"
    assert engine._turn_number == 0
    assert engine._runtime_meta.total_session_tokens == 0
    assert engine._runtime_meta.total_session_input_tokens == 0
    assert engine._runtime_meta.total_session_output_tokens == 0
    assert engine._runtime_meta.last_usage_details == {}
    assert engine._shutting_down is False
    assert engine.state is EngineState.UNINITIALIZED
    assert [event.code for event in events] == ["no_agent_profile"]


@pytest.mark.asyncio
async def test_reset_session_to_welcome_leaves_no_delete_intent_for_the_id_it_restarts_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A writer still pinning the log turns the reset's cleanup into a logical
    delete. The restart keeps the session id, so nothing may be left naming
    that directory for a later sweep."""
    import chrys.service.trajectory.tombstone as tombstone_module
    from chrys.foundation.trajectory.keys import ensure_owner_only_directory
    from chrys.foundation.trajectory.lease import WriterLease

    class _RefusesRename:
        """The module's own ``os`` name — patching the stdlib one is global."""

        def __getattr__(self, name: str) -> object:
            return getattr(os, name)

        def rename(self, *_args: object, **_kwargs: object) -> None:
            raise OSError("rename refused")

    monkeypatch.setattr(tombstone_module, "os", _RefusesRename())

    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "reset_me"
    session_dir = store.session_dir("reset_me")
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")
    lease_path = tombstone_module.session_lease_path(session_dir)
    ensure_owner_only_directory(lease_path.parent)
    # The stuck writer nobody can pull the directory from under.
    lease = WriterLease.try_acquire(lease_path)
    assert lease is not None
    try:
        assert await engine._reset_session_to_welcome("reset_me") is True
    finally:
        lease.release()

    assert engine._session_id == "reset_me"
    assert pending_delete_intents(session_dir.parent) == frozenset()


@pytest.mark.asyncio
async def test_reset_session_to_welcome_reconciles_retained_spill_quota(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "reset_with_spill"
    session_dir = store.session_dir("reset_with_spill")
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")
    sub_agent_artifact = session_dir / "sub_agents" / "sessions" / "Explore_saved.json"
    sub_agent_artifact.parent.mkdir(parents=True)
    sub_agent_artifact.write_text("{}", encoding="utf-8")

    relative_path = "compactions/dropped/turn001/001_read_file_11111111.md"
    record_path = session_dir / relative_path
    record_path.parent.mkdir(parents=True)
    record_path.write_text("retained record\n<!-- end of record -->\n", encoding="utf-8")
    catalog = session_dir / CATALOG_RELATIVE_PATH
    catalog.write_text(
        json.dumps(
            {
                "record_id": "1" * 8,
                "relative_path": relative_path,
                "turn": 1,
                "round": 1,
                "tool": "read_file",
                "bytes": 1,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    engine._spill_quota.initialize(1)

    reset_succeeded = await engine._reset_session_to_welcome("reset_with_spill")

    assert reset_succeeded is True
    assert session_dir.is_dir()
    assert sub_agent_artifact.is_file()
    assert not (session_dir / "session.json").exists()
    assert engine._spill_quota.spent_bytes == record_path.stat().st_size


@pytest.mark.asyncio
async def test_spill_reconciliation_failure_does_not_abort_session_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "restore_with_broken_spill"
    session_dir = store.session_dir("restore_with_broken_spill")
    session_dir.mkdir(parents=True)
    engine._spill_quota.initialize(17)

    def fail_reconciliation(_session_dir: Path, _quota: object) -> None:
        raise OSError("spill catalog is unreadable")

    monkeypatch.setattr(session_lifecycle, "reconcile_spill_storage", fail_reconciliation)

    result = await session_lifecycle._reconcile_existing_spill_storage(engine)

    assert result == session_lifecycle.SpillReconciliationResult(0, 0, frozenset())
    assert not engine._spill_quota.storage_available
    assert engine._spill_quota.spent_bytes == 0
    assert engine._spill_quota.try_reserve(1) is False


@pytest.mark.asyncio
async def test_malformed_spill_catalog_text_does_not_abort_session_lifecycle(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "restore_with_malformed_spill"
    session_dir = store.session_dir("restore_with_malformed_spill")
    catalog = session_dir / CATALOG_RELATIVE_PATH
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        '{"record_id":"1","relative_path":"compactions/dropped/turn001/'
        '001_read_file_11111111.md","turn":1,"round":1,'
        '"tool":"\\ud800","bytes":1,"created_at":"2026-01-01T00:00:00+00:00"}\n',
        encoding="utf-8",
    )
    engine._spill_quota.initialize(17)

    result = await session_lifecycle._reconcile_existing_spill_storage(engine)

    assert result == session_lifecycle.SpillReconciliationResult(0, 0, frozenset())
    assert engine._spill_quota.spent_bytes == 0


@pytest.mark.asyncio
async def test_reset_session_to_welcome_no_lock_path_deletes_backup_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "reset_me"
    session_dir = store.session_dir("reset_me")
    session_dir.mkdir(parents=True)
    (session_dir / "session.json.bak").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(engine, "_session_write_lock_path", lambda _session_id: None)

    reset_succeeded = await engine._reset_session_to_welcome("reset_me")

    assert reset_succeeded is True
    assert not (session_dir / "session.json.bak").exists()
    assert not session_dir.exists()


@pytest.mark.asyncio
async def test_reset_session_to_welcome_preserves_snapshots_when_state_delete_lock_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "reset_me"
    engine._turn_number = 5

    session_dir = store.session_dir("reset_me")
    snapshots_dir = session_dir / "snapshots"
    snapshots_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")
    snapshot_file = snapshots_dir / "turn_2.json"
    snapshot_file.write_text("{}", encoding="utf-8")
    lock_path = engine._session_write_lock_path("reset_me")
    assert lock_path is not None

    monkeypatch.setattr(session_lifecycle, "SESSION_WRITE_LOCK_TIMEOUT_SECONDS", 0.0)
    with FileLock(lock_path):
        reset_succeeded = await engine._reset_session_to_welcome("reset_me")

    assert reset_succeeded is False
    assert (session_dir / "session.json").exists()
    assert snapshot_file.exists()
    assert engine._turn_number == 5
    assert engine._suppress_save is False


@pytest.mark.asyncio
async def test_reset_session_to_welcome_session_end_hooks_see_session_file(tmp_path: Path) -> None:
    events: list[Error] = []
    bus = EventBus()
    await bus.subscribe(Error, lambda event: _collect(events, event))
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(bus, settings=Settings(), state_store=store)
    engine._session_id = "reset_me"
    session_dir = store.session_dir("reset_me")
    session_dir.mkdir(parents=True)
    session_file = session_dir / "session.json"
    session_file.write_text("{}", encoding="utf-8")
    hook_manager = _SessionEndProbeHookManager(session_file)
    engine._hook_manager = hook_manager  # type: ignore[assignment]

    reset_succeeded = await engine._reset_session_to_welcome("reset_me")

    assert reset_succeeded is True
    assert hook_manager.exists_during_fire == [True]
    assert hook_manager.payloads[0]["session_id"] == "reset_me"
    assert not session_dir.exists()
    assert [event.code for event in events] == ["no_agent_profile"]


@pytest.mark.asyncio
async def test_reset_session_to_welcome_delete_failure_restores_engine_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "reset_me"
    engine._turn_number = 5
    session_dir = store.session_dir("reset_me")
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")

    def fail_delete(_session_dir: Path) -> None:
        raise OSError("delete failed")

    monkeypatch.setattr(session_lifecycle, "_delete_reset_session_files", fail_delete)

    reset_succeeded = await engine._reset_session_to_welcome("reset_me")

    assert reset_succeeded is False
    assert engine._session_id == "reset_me"
    assert engine._turn_number == 5
    assert engine._shutting_down is False
    assert engine.state is EngineState.UNINITIALIZED
    assert engine._suppress_save is False


@pytest.mark.asyncio
async def test_failed_reset_recovery_unlink_does_not_refresh_created_at_on_next_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    message = Message("user", ["old history"])
    stamp_message_created_at(message, "2020-01-01T00:00:00+00:00")
    state = {"messages": [message], "compressed_msgs": []}
    await store.save_session("reset_me", state)
    await asyncio.to_thread(store.save_recovery_session, "reset_me", state)
    session_dir = store.session_dir("reset_me")
    primary_file = session_dir / "session.json"
    backup_file = session_dir / "session.json.bak"
    recovery_file = session_dir / SESSION_RECOVERY_FILE_NAME
    snapshots_dir = session_dir / "snapshots"
    snapshots_dir.mkdir()
    snapshot_file = snapshots_dir / "turn_1.json"
    snapshot_file.write_bytes(primary_file.read_bytes())
    recovery = json.loads(recovery_file.read_text(encoding="utf-8"))
    drifted = datetime.now(UTC).isoformat()
    recovery["meta"]["created_at"] = drifted
    recovery["meta"]["updated_at"] = drifted
    recovery_file.write_text(json.dumps(recovery), encoding="utf-8")
    real_unlink = Path.unlink

    def fail_recovery_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == recovery_file:
            raise OSError("simulated recovery unlink failure")
        real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "unlink", fail_recovery_unlink)
        with pytest.raises(OSError, match="simulated recovery unlink failure"):
            session_lifecycle._delete_reset_session_files(session_dir)

    assert not primary_file.exists()
    assert not backup_file.exists()
    assert recovery_file.exists()
    assert snapshot_file.exists()

    await store.save_session("reset_me", state)

    saved = json.loads(primary_file.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(saved["meta"]["created_at"]) == datetime(2020, 1, 1, tzinfo=UTC)
    assert saved["state"]["messages"][0]["contents"][0]["text"] == "old history"
    assert snapshot_file.exists()


@pytest.mark.asyncio
async def test_reset_session_to_welcome_delete_failure_releases_external_lock_before_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "reset_me"
    engine._agent_profile = _profile()
    session_dir = store.session_dir("reset_me")
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")
    lock_path = engine._session_write_lock_path("reset_me")
    assert lock_path is not None
    external_lock = FileLock(lock_path, timeout=0.0)
    external_lock.acquire()
    lifecycle: list[str] = []

    def release_external_lock() -> None:
        lifecycle.append("released")
        external_lock.release()

    async def restart(_profile: AgentProfile, *, operation: str = "startup", **_kwargs: object) -> None:
        lifecycle.append(f"restart:{operation}")
        probe = FileLock(lock_path, timeout=0.0)
        probe.acquire()
        probe.release()

    def fail_delete(_session_dir: Path) -> None:
        raise OSError("delete failed")

    monkeypatch.setattr(engine, "start", restart)
    monkeypatch.setattr(session_lifecycle, "_delete_reset_session_files", fail_delete)
    try:
        reset_succeeded = await engine._reset_session_to_welcome(
            "reset_me",
            write_lock_held=True,
            before_restart=release_external_lock,
        )
    finally:
        external_lock.release()

    assert reset_succeeded is False
    assert lifecycle == ["released", "restart:reset_failed"]


@pytest.mark.asyncio
async def test_reset_session_to_welcome_success_releases_external_lock_after_cleanup_before_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "reset_me"
    engine._agent_profile = _profile()
    session_dir = store.session_dir("reset_me")
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")
    lock_path = engine._session_write_lock_path("reset_me")
    assert lock_path is not None
    external_lock = FileLock(lock_path, timeout=0.0)
    external_lock.acquire()
    lifecycle: list[str] = []
    real_cleanup = session_lifecycle.cleanup_empty_session_dir_path

    def assert_lock_is_held(stage: str) -> None:
        with pytest.raises(TimeoutError), FileLock(lock_path, timeout=0.0):
            pass
        lifecycle.append(stage)

    async def after_delete() -> None:
        assert_lock_is_held("after_delete")

    def cleanup(path: Path | None, *, path_reused: bool = False) -> None:
        if path is not None and not (path / "session.json").exists():
            assert_lock_is_held("cleanup")
        real_cleanup(path, path_reused=path_reused)

    def release_external_lock() -> None:
        lifecycle.append("released")
        external_lock.release()

    async def restart(_profile: AgentProfile, *, operation: str = "startup", **_kwargs: object) -> None:
        lifecycle.append(f"restart:{operation}")
        probe = FileLock(lock_path, timeout=0.0)
        probe.acquire()
        probe.release()

    monkeypatch.setattr(engine, "start", restart)
    monkeypatch.setattr(session_lifecycle, "cleanup_empty_session_dir_path", cleanup)
    try:
        reset_succeeded = await engine._reset_session_to_welcome(
            "reset_me",
            write_lock_held=True,
            after_delete=after_delete,
            before_restart=release_external_lock,
        )
    finally:
        external_lock.release()

    assert reset_succeeded is True
    assert lifecycle == ["after_delete", "cleanup", "released", "restart:reset"]


@pytest.mark.asyncio
async def test_reset_session_to_welcome_delete_failure_restores_mutation_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "reset_me"
    engine._agent_profile = _profile()
    engine._turn_number = 5
    session_dir = store.session_dir("reset_me")
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")
    tracker = MutationTracker(SnapshotStore(session_dir))
    tracker.start_turn(3)
    engine._mutation_tracker = tracker

    class _ResetExecutor(_HistoryStateExecutor):
        is_running = False

        async def interrupt(self) -> None:
            return None

        async def close(self) -> None:
            return None

    engine._executor = _ResetExecutor(  # type: ignore[assignment]
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 5}
    )

    async def fake_build_agent(_profile: AgentProfile, _staged: StagedBuild) -> None:
        engine._executor = _ResetExecutor({"messages": [], "compressed_msgs": [], "turn_counter": 0})  # type: ignore[assignment]

    def fail_delete(_session_dir: Path) -> None:
        raise OSError("delete failed")

    monkeypatch.setattr(engine, "_build_agent", fake_build_agent)
    monkeypatch.setattr(session_lifecycle, "_delete_reset_session_files", fail_delete)

    try:
        reset_succeeded = await engine._reset_session_to_welcome("reset_me")

        assert reset_succeeded is False
        assert engine._mutation_tracker is not None
        assert [turn.turn_id for turn in engine._mutation_tracker.get_all_turns()] == [3]
        assert engine._executor is not None
        assert engine._executor.history_state["chrys_mutations"]["turns"][0]["turn_id"] == 3
        assert engine.state is EngineState.IDLE
        assert engine._suppress_save is False
    finally:
        engine._active_session_guard.release()


@pytest.mark.asyncio
async def test_new_session_reuses_mcp_cache_and_final_shutdown_closes_it(tmp_path: Path, agent_engine) -> None:
    """``/new`` should release leases but keep the engine-owned MCP cache warm."""
    profile = AgentProfile(
        name="Code",
        instructions="Code instructions.",
        tools=ToolsConfig(mcp=[MCPServerConfig(name="srv", transport="stdio", command="python")]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
    )
    bus = EventBus()
    store = JsonFileStateStore(tmp_path)
    engine = agent_engine(bus, settings=Settings(), state_store=store)

    async def _remote(ctx, value: str = "") -> str:
        return value or "ok"

    remote_tool = FunctionTool(
        func=_remote,
        name="remote",
        input_model={"type": "object", "properties": {"value": {"type": "string"}}},
    )

    class _FakeMCPTool:
        def __init__(self) -> None:
            self.functions = [remote_tool]
            self.request_timeout = 30
            self.dropped_banner_lines: list[str] = []
            self.exit_count = 0

        async def __aenter__(self) -> _FakeMCPTool:
            return self

        async def __aexit__(self, *_args: object) -> None:
            self.exit_count += 1

    fake_mcp = _FakeMCPTool()
    fake_tool_registry = MagicMock()
    fake_tool_registry.get_all.return_value = []
    fake_tool_registry.load_builtins = MagicMock()
    fake_ctx = MagicMock()
    fake_ctx.providers = []
    fake_ctx.middleware = []
    fake_ctx.compaction_strategy = MagicMock()
    agent_mock = MagicMock()
    agent_mock.__aenter__ = AsyncMock(return_value=agent_mock)
    agent_mock.__aexit__ = AsyncMock()
    agent_mock.create_session = MagicMock(return_value=MagicMock())
    executor_mock = MagicMock()
    executor_mock.is_running = False
    executor_mock.history_state = {}
    executor_mock.close = AsyncMock()

    with (
        patch.object(agent_builder_module, "Agent", return_value=agent_mock),
        patch.object(agent_builder_module, "ContextManager", return_value=fake_ctx),
        patch.object(agent_builder_module, "create_client", return_value=MagicMock()),
        patch.object(
            agent_builder_module,
            "resolve_selection_for_agent",
            return_value=ModelSelection(ModelProfile(id="test-id", name="test-model"), "active"),
        ),
        patch.object(agent_builder_module, "effective_chat_options", return_value={}),
        patch.object(agent_builder_module, "LoopRecorder", return_value=MagicMock()),
        patch.object(agent_builder_module, "SystemReminderMiddleware", return_value=MagicMock()),
        patch.object(agent_builder_module, "LastWordsGenerator", return_value=MagicMock()),
        patch.object(agent_builder_module, "Executor", return_value=executor_mock),
        patch.object(agent_builder_module, "ApprovalMiddleware", return_value=MagicMock()),
        patch.object(agent_builder_module, "AskUserMiddleware", return_value=MagicMock()),
        patch.object(agent_builder_module, "ApprovalPolicy", return_value=MagicMock()),
        patch("chrys.service.tools.registry.ToolRegistry", return_value=fake_tool_registry),
        patch("chrys.service.skills.adapter.create_skills_provider", new=AsyncMock(return_value=(None, []))),
        patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=fake_mcp) as factory,
    ):
        await engine.start(profile)
        await engine._on_new_session(SessionNew())
        assert factory.call_count == 1
        assert fake_mcp.exit_count == 0

        await engine.shutdown()

    assert fake_mcp.exit_count == 1


def test_cleanup_empty_session_dir_removes_only_unsaved_session(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "empty"
    empty_dir = store.session_dir("empty")
    empty_dir.mkdir(parents=True)

    engine._cleanup_empty_session_dir()

    assert not empty_dir.exists()

    engine._session_id = "recovery"
    recovery_dir = store.session_dir("recovery")
    recovery_dir.mkdir(parents=True)
    (recovery_dir / SESSION_RECOVERY_FILE_NAME).write_text("{}", encoding="utf-8")

    engine._cleanup_empty_session_dir()

    assert recovery_dir.exists()

    engine._session_id = "saved"
    saved_dir = store.session_dir("saved")
    saved_dir.mkdir(parents=True)
    (saved_dir / "session.json").write_text("{}", encoding="utf-8")

    engine._cleanup_empty_session_dir()

    assert saved_dir.exists()


def test_cleanup_empty_session_dir_preserves_restorable_rollback_snapshots(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "snapshot-only"
    session_dir = store.session_dir("snapshot-only")
    snap_dir = session_dir / "snapshots"
    snap_dir.mkdir(parents=True)
    (snap_dir / "turn_1.json").write_text("{}", encoding="utf-8")

    engine._cleanup_empty_session_dir()

    assert session_dir.exists()


@pytest.mark.parametrize(
    "relative_path",
    [
        "sub_agents/legacy.json",
        "sub_agents/pending/active.json",
        "sub_agents/pending/unmatched/old.json",
        "sub_agents/pending/corrupt/bad.json",
        "sub_agents/sessions/Explore_a1b2c3d4e5f6.json",
    ],
)
def test_cleanup_empty_session_dir_preserves_sub_agent_artifacts(tmp_path: Path, relative_path: str) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "sub-agent-only"
    session_dir = store.session_dir("sub-agent-only")
    artifact = session_dir / relative_path
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")

    engine._cleanup_empty_session_dir()

    assert session_dir.exists()


def test_cleanup_empty_session_dir_preserves_a_recorded_trajectory(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "log-only"
    session_dir = store.session_dir("log-only")
    events = session_dir / "trajectory" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text("{}\n", encoding="utf-8")

    engine._cleanup_empty_session_dir()

    # Only an explicit clear or delete takes a session's log with it.
    assert session_dir.exists()

    events.write_text("", encoding="utf-8")
    engine._cleanup_empty_session_dir()

    # A file the writer opened and never wrote to records nothing.
    assert not session_dir.exists()


def test_cleanup_empty_session_dir_keeps_a_trajectory_it_cannot_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "unreadable"
    session_dir = store.session_dir("unreadable")
    (session_dir / "trajectory").mkdir(parents=True)

    class _Unreadable:
        def stat(self) -> None:
            raise PermissionError("cannot stat")

    monkeypatch.setattr("chrys.service.trajectory.session.trajectory_events_path", lambda _dir: _Unreadable())

    engine._cleanup_empty_session_dir()

    # "I could not look" is not "there is nothing there".
    assert session_dir.exists()


def test_cleanup_empty_session_dir_ignores_sub_agent_tmp_files(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "tmp-only"
    session_dir = store.session_dir("tmp-only")
    tmp_file = session_dir / "sub_agents" / "sessions" / ".Explore_a1b2c3d4e5f6.json.tmp"
    tmp_file.parent.mkdir(parents=True)
    tmp_file.write_text("{}", encoding="utf-8")

    engine._cleanup_empty_session_dir()

    assert not session_dir.exists()


def test_no_store_write_lock_path_uses_resolved_sessions_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_root = tmp_path / "custom-root"
    monkeypatch.setenv(SESSION_ROOT_DIR_ENV_VAR, str(custom_root))
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=None)

    lock_path = engine._session_write_lock_path("lock-session")

    assert lock_path == custom_root / "sessions" / ".locks" / f"{session_short_id('lock-session')}.write.lock"
    assert lock_path.parent.is_dir()


def _seed_checkpoint_engine(store: JsonFileStateStore, session_id: str) -> AgentEngine:
    """Build an engine wired enough to produce a real recovery checkpoint."""
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = session_id
    user = Message("user", ["do work"])
    assistant = Message("assistant", [Content.from_function_call("c1", "read_file", arguments={})])
    tool = Message("tool", [Content.from_function_result("c1", result="done")])

    class _Executor:
        def __init__(self) -> None:
            self.history_state = {"messages": [user], "compressed_msgs": [], "turn_counter": 1}

    engine._executor = _Executor()  # type: ignore[assignment]
    capture = LoopRecorder()
    capture._initial_count = 1
    capture._captured = [user, assistant, tool]
    engine._loop_recorder = capture
    engine._turn_state.set_current_input("do work", None, None)
    return engine


@pytest.mark.asyncio
async def test_recovery_checkpoint_write_is_backgrounded_then_flushed(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "bg")
    sidecar = store.session_dir("bg") / SESSION_RECOVERY_FILE_NAME

    await engine._save_recovery_checkpoint()

    # The disk write is dispatched to a background task, not awaited inline, so
    # the LLM round trip never blocks on the sidecar's fsync/lock.
    assert engine._recovery_write_task is not None

    await engine._flush_recovery_checkpoint()
    assert sidecar.exists()


@pytest.mark.asyncio
async def test_persist_recovery_now_strictly_writes_current_snapshot(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "strict-now")

    assert await engine.persist_recovery_now() is True

    restored = await store.load_recovery_session("strict-now")
    assert restored is not None
    assert any(
        content.type == "function_result" and content.result == "done"
        for message in restored["messages"]
        for content in message.contents
    )


@pytest.mark.asyncio
async def test_typed_recovery_barrier_persists_committed_journal_exchange(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "journal-barrier"
    user = Message("user", ["do work"])

    class _Executor:
        def __init__(self) -> None:
            self.history_state = {"messages": [user], "compressed_msgs": [], "turn_counter": 1}

    engine._executor = _Executor()  # type: ignore[assignment]
    recorder = LoopRecorder(on_pre_wire_barrier=engine.persist_recovery_barrier)
    engine._loop_recorder = recorder
    engine._turn_state.set_current_input("do work", None, None)
    first_call = Content.from_function_call("committed-call-1", "write_file", arguments={})
    first_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    first_commit = recorder.stage_exchange(
        [Message("assistant", [first_call])], [first_call], result_carrier_item_id="a" * 32
    )[0]
    first_result = Content.from_function_result("committed-call-1", result="written-1")
    first_commit.commit_final(first_result)
    recorder.seal_exchange(Message("tool", [first_result]))

    second_call = Content.from_function_call("committed-call-2", "write_file", arguments={})
    second_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 1
    second_commit = recorder.stage_exchange(
        [Message("assistant", [second_call])], [second_call], result_carrier_item_id="b" * 32
    )[0]
    second_commit.commit_final(Content.from_function_result("committed-call-2", result="written-2"))

    await recorder.record_pre_call([user])

    restored = await store.load_recovery_session("journal-barrier")
    assert restored is not None
    assert [
        (content.call_id, content.result)
        for message in restored["messages"]
        for content in message.contents
        if content.type == "function_result"
    ] == [
        ("committed-call-1", "written-1"),
        ("committed-call-2", "written-2"),
    ]


@pytest.mark.asyncio
async def test_stale_background_checkpoint_cannot_downgrade_barrier_snapshot(tmp_path: Path) -> None:
    """A coalesced writer that froze state before a strict barrier must not
    overwrite the sidecar with its older snapshot after the barrier ran."""
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "stale-guard"
    user = Message("user", ["do work"])

    class _Executor:
        def __init__(self) -> None:
            self.history_state = {"messages": [user], "compressed_msgs": [], "turn_counter": 1}

    engine._executor = _Executor()  # type: ignore[assignment]
    recorder = LoopRecorder()
    engine._loop_recorder = recorder
    engine._turn_state.set_current_input("do work", None, None)

    first_call = Content.from_function_call("stale-call-1", "write_file", arguments={})
    first_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 0
    first_commit = recorder.stage_exchange(
        [Message("assistant", [first_call])], [first_call], result_carrier_item_id="a" * 32
    )[0]
    first_result = Content.from_function_result("stale-call-1", result="written-1")
    first_commit.commit_final(first_result)
    recorder.seal_exchange(Message("tool", [first_result]))

    engine._recovery_snapshot_seq += 1
    stale = (engine._recovery_snapshot_seq, engine._build_recovery_snapshot())

    second_call = Content.from_function_call("stale-call-2", "write_file", arguments={})
    second_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 1
    second_commit = recorder.stage_exchange(
        [Message("assistant", [second_call])], [second_call], result_carrier_item_id="b" * 32
    )[0]
    second_result = Content.from_function_result("stale-call-2", result="written-2")
    second_commit.commit_final(second_result)
    recorder.seal_exchange(Message("tool", [second_result]))

    assert await engine.persist_recovery_barrier() is RecoveryPersistOutcome.PERSISTED

    engine._pending_recovery_state = stale
    await engine._drain_recovery_checkpoints()

    restored = await store.load_recovery_session("stale-guard")
    assert restored is not None
    results = [
        content.result
        for message in restored["messages"]
        for content in message.contents
        if content.type == "function_result"
    ]
    assert results == ["written-1", "written-2"]


@pytest.mark.asyncio
async def test_recovery_persistence_apis_keep_typed_and_bool_contracts_distinct(tmp_path: Path) -> None:
    configured = _seed_checkpoint_engine(JsonFileStateStore(tmp_path), "typed-outcomes")
    assert await configured.persist_recovery_barrier() is RecoveryPersistOutcome.PERSISTED
    assert await configured.persist_recovery_now() is True

    empty = AgentEngine(EventBus(), settings=Settings(), state_store=JsonFileStateStore(tmp_path / "empty"))
    assert await empty.persist_recovery_barrier() is RecoveryPersistOutcome.NOTHING_TO_PERSIST
    assert await empty.persist_recovery_now() is False

    unconfigured = AgentEngine(EventBus(), settings=Settings(), state_store=None)
    assert await unconfigured.persist_recovery_barrier() is RecoveryPersistOutcome.UNCONFIGURED
    assert await unconfigured.persist_recovery_now() is False


@pytest.mark.asyncio
async def test_persist_recovery_now_propagates_strict_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "strict-failure")

    async def _fail_write(*_args: object, **_kwargs: object) -> bool:
        raise OSError("recovery fsync failed")

    monkeypatch.setattr(engine._persistence, "save_recovery_session_strict", _fail_write)

    assert await engine.persist_recovery_barrier() is RecoveryPersistOutcome.FAILED
    with pytest.raises(OSError, match="recovery fsync failed"):
        await engine.persist_recovery_now()


@pytest.mark.asyncio
async def test_persist_recovery_now_drains_stale_background_write_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "strict-order")
    background_started = asyncio.Event()
    release_background = asyncio.Event()
    order: list[str] = []
    original_background = engine._persistence.save_recovery_session
    original_strict = engine._persistence.save_recovery_session_strict

    async def _parked_background(*args: Any, **kwargs: Any) -> None:
        order.append("background-start")
        background_started.set()
        await release_background.wait()
        await original_background(*args, **kwargs)
        order.append("background-end")

    async def _ordered_strict(*args: Any, **kwargs: Any) -> bool:
        order.append(f"strict-{sum(item.startswith('strict-') for item in order) + 1}")
        return await original_strict(*args, **kwargs)

    monkeypatch.setattr(engine._persistence, "save_recovery_session", _parked_background)
    monkeypatch.setattr(engine._persistence, "save_recovery_session_strict", _ordered_strict)

    await engine._save_recovery_checkpoint()
    await background_started.wait()
    assert engine._loop_recorder is not None
    assert engine._loop_recorder._captured is not None
    engine._loop_recorder._captured[-1] = Message("tool", [Content.from_function_result("c1", result="newer")])

    strict_task = asyncio.create_task(engine.persist_recovery_now())
    release_background.set()

    assert await strict_task is True
    assert order == ["background-start", "strict-1", "background-end", "strict-2"]
    restored = await store.load_recovery_session("strict-order")
    assert restored is not None
    assert any(
        content.type == "function_result" and content.result == "newer"
        for message in restored["messages"]
        for content in message.contents
    )


@pytest.mark.asyncio
async def test_persist_recovery_now_cancellation_keeps_clean_save_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "strict-cancel")
    sidecar = store.session_dir("strict-cancel") / SESSION_RECOVERY_FILE_NAME
    strict_started = ThreadEvent()
    release_strict = ThreadEvent()
    clean_started = ThreadEvent()
    original_recovery = store.save_recovery_session
    original_primary = store._save_session_sync

    def _parked_recovery(*args: Any, **kwargs: Any) -> None:
        strict_started.set()
        release_strict.wait()
        original_recovery(*args, **kwargs)

    def _observed_primary(*args: Any, **kwargs: Any) -> None:
        clean_started.set()
        original_primary(*args, **kwargs)

    monkeypatch.setattr(store, "save_recovery_session", _parked_recovery)
    monkeypatch.setattr(store, "_save_session_sync", _observed_primary)

    strict_task = asyncio.create_task(engine.persist_recovery_now())
    assert await wait_until(strict_started.is_set, timeout=1.0, interval=0.01)
    strict_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await strict_task

    clean_task = asyncio.create_task(engine._save_current_session())
    clean_started_early = await wait_until(clean_started.is_set, timeout=0.2, interval=0.01)
    release_strict.set()
    assert await clean_task is True

    assert not clean_started_early
    assert clean_started.is_set()
    assert not sidecar.exists()


@pytest.mark.asyncio
async def test_llm_boundary_checkpoint_never_awaits_the_writer(tmp_path: Path) -> None:
    """The plain ``on_checkpoint`` path stays fire-and-forget (§2.5).

    The consumed-injection callback awaits the sidecar write, but that flush
    must not creep into the per-LLM-boundary hot path: with the disk writer
    parked on an event nobody has set yet, ``_save_recovery_checkpoint`` (the
    exact callback wired to ``LoopRecorder.on_checkpoint``) still returns —
    an implementation that awaited the writer would deadlock here.
    """
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "hot")
    sidecar = store.session_dir("hot") / SESSION_RECOVERY_FILE_NAME

    release = asyncio.Event()
    original = engine._persistence.save_recovery_session

    async def parked_save(*args: Any, **kwargs: Any) -> None:
        await release.wait()
        await original(*args, **kwargs)

    engine._persistence.save_recovery_session = parked_save  # type: ignore[method-assign]

    # Two boundary checkpoints while the writer is parked: both return, the
    # snapshots coalesce newest-wins, and nothing has reached disk yet.
    await engine._save_recovery_checkpoint()
    await engine._save_recovery_checkpoint()

    assert engine._recovery_write_task is not None
    assert not engine._recovery_write_task.done()
    assert not sidecar.exists()

    release.set()
    await engine._flush_recovery_checkpoint()
    assert sidecar.exists()


@pytest.mark.asyncio
async def test_primary_save_flushes_pending_checkpoint_then_deletes_sidecar(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "ord")
    sidecar = store.session_dir("ord") / SESSION_RECOVERY_FILE_NAME

    await engine._save_recovery_checkpoint()
    assert engine._recovery_write_task is not None

    # A clean primary save must flush the in-flight checkpoint first, so its
    # structural sidecar delete is the last write — no stale sidecar survives.
    await engine._save_current_session()

    assert not sidecar.exists()
    assert engine._pending_recovery_state is None


@pytest.mark.asyncio
async def test_flush_recovery_checkpoint_survives_outer_cancellation(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "cancel")
    sidecar = store.session_dir("cancel") / SESSION_RECOVERY_FILE_NAME

    started = asyncio.Event()
    release = asyncio.Event()
    original = engine._persistence.save_recovery_session

    async def slow_save(*args: Any, **kwargs: Any) -> None:
        started.set()
        await release.wait()
        await original(*args, **kwargs)

    engine._persistence.save_recovery_session = slow_save  # type: ignore[method-assign]

    await engine._save_recovery_checkpoint()
    assert engine._recovery_write_task is not None

    # Mimic the run task parking in the post-run flush, then being cancelled by
    # the graceful-shutdown timeout while the writer is mid-write.
    waiter = asyncio.create_task(engine._flush_recovery_checkpoint())
    await started.wait()
    waiter.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await waiter

    # The writer must survive the outer cancellation (shield), not be aborted,
    # so the trailing shutdown flush can still drain it to disk.
    assert not engine._recovery_write_task.cancelled()

    release.set()
    await engine._flush_recovery_checkpoint()
    assert sidecar.exists()


# ---------------------------------------------------------------------------
# Phase 4 LAST_WORDS note persistence across save / recovery / restore
# ---------------------------------------------------------------------------


class _HistoryStateExecutor:
    """Minimal executor surface for session state save/restore plumbing."""

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self.history_state: dict[str, Any] = state if state is not None else {}


@pytest.mark.asyncio
async def test_save_session_sets_and_pops_workspace_baseline(tmp_path: Path) -> None:
    """The live history and disk cannot retain a stale truthy baseline."""
    from chrys.foundation.models.workspace import Workspace

    store = JsonFileStateStore(tmp_path / "sessions")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "workspace_baseline"
    engine._workspace = Workspace.from_cwd(str(workspace))
    engine._workspace_change_tracker.retarget_roots(engine._workspace)
    engine._workspace_change_tracker.capture_baseline(1)
    engine._executor = _HistoryStateExecutor(  # type: ignore[assignment]
        {"messages": [Message("user", ["hi"])], "compressed_msgs": [], "turn_counter": 1}
    )

    assert await engine._save_current_session() is True
    assert "chrys_workspace_baseline" in engine._executor.history_state
    loaded = await store.load_session("workspace_baseline")
    assert loaded is not None and "chrys_workspace_baseline" in loaded

    engine._workspace_change_tracker.invalidate()
    assert await engine._save_current_session() is True
    assert "chrys_workspace_baseline" not in engine._executor.history_state
    loaded = await store.load_session("workspace_baseline")
    assert loaded is not None and "chrys_workspace_baseline" not in loaded


@pytest.mark.asyncio
async def test_save_session_persists_last_words_note(tmp_path: Path) -> None:
    """The Phase 4 note lands in the saved session state — it is the only
    replacement for the compacted turn's dropped tool-call history — and a
    cleared note removes the stale key on the next save."""
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware

    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "lw_save"
    engine._executor = _HistoryStateExecutor(  # type: ignore[assignment]
        {"messages": [Message("user", ["hi"])], "compressed_msgs": [], "turn_counter": 1}
    )
    engine._reminder_middleware = SystemReminderMiddleware()
    engine._reminder_middleware.set_last_words("[LAST_WORDS] resume from step 3")

    assert await engine._save_current_session() is True
    loaded = await store.load_session("lw_save")
    assert loaded is not None
    assert loaded["last_words"] == "[LAST_WORDS] resume from step 3"

    engine._reminder_middleware.set_last_words(None)
    assert await engine._save_current_session() is True
    loaded = await store.load_session("lw_save")
    assert loaded is not None
    assert "last_words" not in loaded


@pytest.mark.asyncio
async def test_save_session_persists_manifest_and_breaker_without_note(tmp_path: Path) -> None:
    from chrys.service.agent_middleware.system_reminder import (
        DropRoundBreakerState,
        ManifestEntry,
        SystemReminderMiddleware,
    )

    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "lw_family_save"
    engine._executor = _HistoryStateExecutor(  # type: ignore[assignment]
        {"messages": [Message("user", ["hi"])], "compressed_msgs": [], "turn_counter": 1}
    )
    engine._reminder_middleware = SystemReminderMiddleware()
    engine._reminder_middleware.append_manifest(
        [
            ManifestEntry(
                record_id="r1",
                group_id="g1",
                record_dir="compactions/dropped/turn001",
                relative_path="compactions/dropped/turn001/001_tool_r1.md",
                turn=1,
                round=1,
                sequence=1,
                tool="tool",
                display_argument="",
                outcome="ok",
                size_chars=10,
            )
        ]
    )
    breaker = DropRoundBreakerState(attempts=1, consecutive_no_progress=1, tail_override=True, side_call_tokens=42)
    engine._reminder_middleware.set_drop_round_breaker(breaker)
    engine._reminder_middleware.restore_catalog_pointer_record_count(0)

    assert await engine._save_current_session() is True

    loaded = await store.load_session("lw_family_save")
    assert loaded is not None
    assert loaded["last_words_manifest"][0]["record_id"] == "r1"
    assert loaded["last_words_breaker"] == breaker.to_state()
    assert loaded[CATALOG_POINTER_RECORD_COUNT_STATE_KEY] == 0
    assert "last_words" not in loaded


@pytest.mark.asyncio
async def test_save_erasure_protects_note_and_manifest_but_not_breaker(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "lw_erasure"
    manifest = [{"record_id": "r1", "relative_path": "compactions/dropped/turn001/a.md"}]
    engine._executor = _HistoryStateExecutor(  # type: ignore[assignment]
        {
            "messages": [Message("user", ["hi"])],
            "compressed_msgs": [],
            "turn_counter": 1,
            "last_words": "keep note",
            "last_words_manifest": manifest,
            "last_words_breaker": {"version": 1, "attempts": 7},
        }
    )
    engine._reminder_middleware = None

    assert await engine._save_current_session() is True

    loaded = await store.load_session("lw_erasure")
    assert loaded is not None
    assert loaded["last_words"] == "keep note"
    assert loaded["last_words_manifest"] == manifest
    assert "last_words_breaker" not in loaded


@pytest.mark.asyncio
async def test_recovery_checkpoint_persists_last_words_note(tmp_path: Path) -> None:
    """The crash-recovery sidecar carries the Phase 4 note for the in-flight turn."""
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware

    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "lw_sidecar")
    engine._reminder_middleware = SystemReminderMiddleware()
    engine._reminder_middleware.set_last_words("[LAST_WORDS] mid-turn progress")

    await engine._save_recovery_checkpoint()
    await engine._flush_recovery_checkpoint()

    loaded = await store.load_session("lw_sidecar", prefer_recovery=True)
    assert loaded is not None
    assert loaded["last_words"] == "[LAST_WORDS] mid-turn progress"


@pytest.mark.asyncio
async def test_recovery_checkpoint_carries_manifest_and_breaker(tmp_path: Path) -> None:
    from chrys.service.agent_middleware.system_reminder import (
        DropRoundBreakerState,
        ManifestEntry,
        SystemReminderMiddleware,
    )

    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "lw_family_sidecar")
    engine._reminder_middleware = SystemReminderMiddleware()
    entry = ManifestEntry(
        record_id="r1",
        group_id="g1",
        record_dir="compactions/dropped/turn001",
        relative_path="compactions/dropped/turn001/001_tool_r1.md",
        turn=1,
        round=1,
        sequence=1,
        tool="tool",
        display_argument="",
        outcome="unknown",
        size_chars=10,
    )
    engine._reminder_middleware.append_manifest([entry])
    breaker = DropRoundBreakerState(attempts=2, side_call_tokens=333)
    engine._reminder_middleware.set_drop_round_breaker(breaker)
    engine._reminder_middleware.restore_catalog_pointer_record_count(0)

    await engine._save_recovery_checkpoint()
    await engine._flush_recovery_checkpoint()

    loaded = await store.load_session("lw_family_sidecar", prefer_recovery=True)
    assert loaded is not None
    assert loaded["last_words_manifest"] == [entry.to_state()]
    assert loaded["last_words_breaker"] == breaker.to_state()
    assert loaded[CATALOG_POINTER_RECORD_COUNT_STATE_KEY] == 0


@pytest.mark.asyncio
async def test_recovery_sidecar_keeps_new_same_text_injection_alongside_earlier_one(tmp_path: Path) -> None:
    """Identity-dedup end-to-end: sidecar keeps a new injection repeating old text.

    Resumed-turn shape: the turn region already holds a persisted injected
    "same note" (own ``_injection_id``); the user injects the same text again
    mid-run and the process would crash before finalization.  The recovery
    checkpoint replays the consumed-injection mirror — region-wide TEXT dedup
    would match the earlier note and drop the new one (user-input loss); the
    ``_injection_id`` stamp must land BOTH copies in the sidecar.
    """
    from chrys.foundation.models.history_markers import HistoryMarkerKind
    from chrys.service.agent_middleware.injection import ConsumedInjection, InjectionAnchor

    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "inj-dup"
    opener = Message("user", ["task"])
    earlier = Message("user", ["same note"])
    earlier.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    earlier.additional_properties[HistoryMarkerKind.INJECTION_ID_KEY] = "id-old"
    done = Message("assistant", ["done"])

    engine._executor = _HistoryStateExecutor(  # type: ignore[assignment]
        {"messages": [opener, earlier, done], "compressed_msgs": [], "turn_counter": 1}
    )
    capture = LoopRecorder()
    capture._initial_count = 3
    capture._captured = [opener, earlier, done]
    engine._loop_recorder = capture
    engine._turn_state.set_current_input("task", None, None)
    engine._consumed_injections.append(
        ConsumedInjection(text="same note", anchor=InjectionAnchor.from_message(done), consumption_id="id-new")
    )

    await engine._save_recovery_checkpoint()
    await engine._flush_recovery_checkpoint()

    loaded = await store.load_session("inj-dup", prefer_recovery=True)
    assert loaded is not None
    notes = [m for m in loaded["messages"] if m.role == "user" and m.text == "same note"]
    assert len(notes) == 2
    assert [m.additional_properties.get(HistoryMarkerKind.INJECTION_ID_KEY) for m in notes] == ["id-old", "id-new"]
    assert all(m.additional_properties.get(HistoryMarkerKind.INJECTED_KEY) for m in notes)


@pytest.mark.asyncio
async def test_restore_rearms_persisted_last_words_for_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persisted LAST_WORDS note is re-armed on session restore: resuming the
    interrupted turn re-injects it into the next request, exactly like an
    in-process retry; it also survives a restore that is closed without a run."""
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware

    profile = _profile()
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_lw",
        {
            "messages": [Message("user", ["primary"])],
            "compressed_msgs": [],
            "turn_counter": 1,
            "last_words": "[LAST_WORDS] resume from step 3",
        },
        agent_profile=profile.name,
    )
    events: list[SessionRestored] = []
    bus = EventBus()
    await bus.subscribe(SessionRestored, lambda event: _collect(events, event))
    engine = AgentEngine(bus, settings=Settings(), state_store=store, agent_registry=_registry(profile))

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = start_profile, operation
        engine._executor = _HistoryStateExecutor()  # type: ignore[assignment]
        engine._reminder_middleware = SystemReminderMiddleware()

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_lw"))
    finally:
        engine._active_session_guard.release()

    assert len(events) == 1
    mw = engine._reminder_middleware
    assert mw is not None
    # The note is visible to an idle save (restore → quit must not erase it)...
    assert mw.get_last_words() == "[LAST_WORDS] resume from step 3"
    # ...and a post-restart retry (Continue) injects it into the next request.
    mw.prepare_turn(usage={}, preserve_last_words=True)
    appended = mw._build_last_words_reminders()
    assert len(appended) == 1
    assert "[LAST_WORDS] resume from step 3" in appended[0]


@pytest.mark.asyncio
async def test_restore_reconciles_spill_quota_manifest_and_availability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chrys.service.agent_middleware.system_reminder import (
        DropRoundBreakerState,
        ManifestEntry,
        SystemReminderMiddleware,
    )
    from chrys.service.context.compaction.spill import CATALOG_RELATIVE_PATH

    profile = _profile()
    store = JsonFileStateStore(tmp_path)
    session_id = "restore_lw_family"
    session_dir = store.session_dir(session_id)
    live_relative = "compactions/dropped/turn001/001_tool_11111111.md"
    missing_relative = "compactions/dropped/turn001/002_tool_22222222.md"
    live_path = session_dir / live_relative
    live_path.parent.mkdir(parents=True, exist_ok=True)
    live_path.write_text("record\n<!-- end of record -->\n", encoding="utf-8")
    orphan = live_path.with_name("orphan.md")
    orphan.write_text("partial", encoding="utf-8")
    manifest_projection = live_path.with_name("manifest.md")
    manifest_projection.write_text("stale projection", encoding="utf-8")
    catalog = session_dir / CATALOG_RELATIVE_PATH
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        "\n".join(
            json.dumps(
                {
                    "record_id": record_id,
                    "relative_path": relative_path,
                    "turn": 1,
                    "round": 1,
                    "tool": "tool",
                    "bytes": 999,
                    "created_at": "2026-01-01T00:00:00+00:00",
                }
            )
            for record_id, relative_path in (("r-live", live_relative), ("r-missing", missing_relative))
        )
        + "\n",
        encoding="utf-8",
    )

    def entry(record_id: str, relative_path: str, sequence: int) -> dict[str, Any]:
        return ManifestEntry(
            record_id=record_id,
            group_id=f"g-{sequence}",
            record_dir="compactions/dropped/turn001",
            relative_path=relative_path,
            turn=1,
            round=1,
            sequence=sequence,
            tool="tool",
            display_argument="",
            outcome="unknown",
            size_chars=10,
        ).to_state()

    breaker = DropRoundBreakerState(attempts=3, consecutive_no_progress=1, tail_override=True, side_call_tokens=777)
    await store.save_session(
        session_id,
        {
            "messages": [Message("user", ["primary"])],
            "compressed_msgs": [],
            "turn_counter": 1,
            "last_words_manifest": [entry("r-live", live_relative, 1), entry("r-missing", missing_relative, 2)],
            "last_words_breaker": breaker.to_state(),
        },
        agent_profile=profile.name,
    )
    engine = AgentEngine(
        EventBus(),
        settings=Settings(),
        state_store=store,
        agent_registry=_registry(profile),
    )

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = start_profile, operation
        engine._executor = _HistoryStateExecutor()  # type: ignore[assignment]
        engine._reminder_middleware = SystemReminderMiddleware(
            session_root=engine._session_dir,
            file_read_available=True,
        )

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id=session_id))
    finally:
        engine._active_session_guard.release()

    assert not orphan.exists()
    assert engine._spill_quota.spent_bytes == live_path.stat().st_size
    assert "stale projection" not in manifest_projection.read_text(encoding="utf-8")
    assert live_path.name in manifest_projection.read_text(encoding="utf-8")
    mw = engine._reminder_middleware
    assert mw is not None
    mw.prepare_turn(usage={}, preserve_last_words=True)
    assert mw.get_drop_round_breaker() == breaker
    rendered = mw._build_last_words_reminders()[0]
    assert live_path.name in rendered
    assert "002_tool_22222222.md (record missing)" in rendered


# ---------------------------------------------------------------------------
# Session todo list persistence across save / recovery / restore / reset
# ---------------------------------------------------------------------------


_TODOS = [
    {"content": "read the plan", "status": "completed", "active_form": "Reading the plan"},
    {"content": "implement", "status": "in_progress", "active_form": "Implementing"},
]


async def _tracker_with_todos(todos: list[dict[str, str]]) -> TodoTracker:
    tracker = TodoTracker()
    await tracker.restore(todos)
    return tracker


def test_todo_tracker_property_is_read_only() -> None:
    """``todo_tracker`` mirrors ``mutation_tracker``: a read-only view of the
    private attribute for hosts (e.g. the ACP plan-update sender)."""
    engine = AgentEngine(EventBus(), settings=Settings())
    assert engine.todo_tracker is None
    tracker = TodoTracker()
    engine._todo_tracker = tracker
    assert engine.todo_tracker is tracker
    with pytest.raises(AttributeError):
        engine.todo_tracker = TodoTracker()  # type: ignore[misc]


@pytest.mark.asyncio
async def test_reset_for_restart_nulls_todo_tracker(tmp_path: Path) -> None:
    """No cross-session leak: a fresh start must not inherit the old list."""
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._todo_tracker = await _tracker_with_todos(_TODOS)

    engine._reset_for_restart(None)

    assert engine._todo_tracker is None


def test_reset_for_restart_preserves_but_clears_workspace_tracker(tmp_path: Path) -> None:
    engine = AgentEngine(EventBus(), settings=Settings())
    workspace = Workspace.from_cwd(str(tmp_path))
    tracker = engine._workspace_change_tracker
    tracker.retarget_roots(workspace)
    tracker.capture_baseline(1)
    tracker.queue_safety_notice("old safety")
    engine._mutation_tracker = MutationTracker(SnapshotStore(tmp_path / "mutations"))
    engine._todo_tracker = TodoTracker()

    engine._reset_for_restart(None)

    assert engine._workspace_change_tracker is tracker
    assert tracker.baseline is None
    assert tracker.take_pending_notice() is None
    assert engine._mutation_tracker is None
    assert engine._todo_tracker is None


@pytest.mark.asyncio
async def test_save_session_persists_todo_list(tmp_path: Path) -> None:
    """Save reads the TRACKER (not prior state): set when non-empty, and a
    cleared tracker pops the stale key on the next save (empty ≡ absent)."""
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "todo_save"
    engine._executor = _HistoryStateExecutor(  # type: ignore[assignment]
        {"messages": [Message("user", ["hi"])], "compressed_msgs": [], "turn_counter": 1}
    )
    engine._todo_tracker = await _tracker_with_todos(_TODOS)

    assert await engine._save_current_session() is True
    loaded = await store.load_session("todo_save")
    assert loaded is not None
    assert loaded["chrys_todos"] == _TODOS

    await engine._todo_tracker.clear()
    assert await engine._save_current_session() is True
    loaded = await store.load_session("todo_save")
    assert loaded is not None
    assert "chrys_todos" not in loaded


@pytest.mark.asyncio
async def test_recovery_checkpoint_persists_todo_list(tmp_path: Path) -> None:
    """The crash-recovery sidecar carries the todo list for the in-flight turn."""
    store = JsonFileStateStore(tmp_path)
    engine = _seed_checkpoint_engine(store, "todo_sidecar")
    engine._todo_tracker = await _tracker_with_todos(_TODOS)

    await engine._save_recovery_checkpoint()
    await engine._flush_recovery_checkpoint()

    loaded = await store.load_session("todo_sidecar", prefer_recovery=True)
    assert loaded is not None
    assert loaded["chrys_todos"] == _TODOS


@pytest.mark.asyncio
async def test_restore_hydrates_todo_tracker_before_last_words(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore hydrates ``chrys_todos`` into the tracker, and does so BEFORE
    ``restore_last_words`` — the restored note re-captures its todo section
    from the already-hydrated tracker."""
    from functools import partial

    from chrys.orchestration.engine.build.builder import _render_todo_reminder
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware

    profile = _profile()
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_todo",
        {
            "messages": [Message("user", ["primary"])],
            "compressed_msgs": [],
            "turn_counter": 1,
            "chrys_todos": _TODOS,
            "last_words": "[LAST_WORDS] resume from step 3",
        },
        agent_profile=profile.name,
    )
    events: list[SessionRestored] = []
    bus = EventBus()
    await bus.subscribe(SessionRestored, lambda event: _collect(events, event))
    engine = AgentEngine(bus, settings=Settings(), state_store=store, agent_registry=_registry(profile))

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = start_profile, operation
        engine._executor = _HistoryStateExecutor()  # type: ignore[assignment]
        # Mirror the builder wiring: the provider reads the engine's tracker,
        # which _hydrate_restored_session must have populated already.
        engine._reminder_middleware = SystemReminderMiddleware(
            todo_state_provider=partial(_render_todo_reminder, engine._todo_tracker),
        )

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_todo"))
    finally:
        engine._active_session_guard.release()

    assert len(events) == 1
    tracker = engine.todo_tracker
    assert tracker is not None
    assert tracker.serialize() == _TODOS
    mw = engine._reminder_middleware
    assert mw is not None
    mw.prepare_turn(usage={}, preserve_last_words=True)
    appended = mw._build_last_words_reminders()
    assert len(appended) == 1
    assert "[LAST_WORDS] resume from step 3" in appended[0]
    assert "- [>] implement" in appended[0]


@pytest.mark.asyncio
async def test_restore_without_todos_leaves_tracker_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restoring a session without ``chrys_todos`` yields a fresh empty
    tracker (never ``None`` — the tracker is profile/state independent)."""
    profile = _profile()
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "restore_no_todo",
        {"messages": [Message("user", ["primary"])], "compressed_msgs": [], "turn_counter": 1},
        agent_profile=profile.name,
    )
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store, agent_registry=_registry(profile))
    engine._todo_tracker = await _tracker_with_todos(_TODOS)  # stale previous-session state

    async def fake_shutdown(*, release_session_lock: bool = True, close_mcp_cache: bool = True) -> None:
        _ = release_session_lock, close_mcp_cache

    async def fake_start(
        start_profile: AgentProfile,
        *,
        operation: str = "startup",
        staged_loaded: LoadedSettings | None = None,
        workspace: Workspace | None = None,
    ) -> None:
        _ = start_profile, operation
        engine._executor = _HistoryStateExecutor()  # type: ignore[assignment]

    monkeypatch.setattr(engine, "shutdown", fake_shutdown)
    monkeypatch.setattr(engine, "start", fake_start)

    try:
        await engine._on_session_restore(SessionRestore(session_id="restore_no_todo"))
    finally:
        engine._active_session_guard.release()

    tracker = engine.todo_tracker
    assert tracker is not None
    assert tracker.snapshot() == ()


@pytest.mark.asyncio
async def test_reset_session_to_welcome_delete_failure_restores_todo_tracker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed reset rehydrates the TRACKER, not just ``history_state`` —
    save reads the tracker, so an empty one would pop ``chrys_todos`` on the
    next save even with the key still present in the reattached state."""
    store = JsonFileStateStore(tmp_path)
    engine = AgentEngine(EventBus(), settings=Settings(), state_store=store)
    engine._session_id = "reset_me"
    engine._agent_profile = _profile()
    engine._turn_number = 5
    session_dir = store.session_dir("reset_me")
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}", encoding="utf-8")
    engine._todo_tracker = await _tracker_with_todos(_TODOS)

    class _ResetExecutor(_HistoryStateExecutor):
        is_running = False

        async def interrupt(self) -> None:
            return None

        async def close(self) -> None:
            return None

    engine._executor = _ResetExecutor(  # type: ignore[assignment]
        {"messages": [Message("user", ["saved"])], "compressed_msgs": [], "turn_counter": 5}
    )

    async def fake_build_agent(_profile: AgentProfile, _staged: StagedBuild) -> None:
        engine._executor = _ResetExecutor({"messages": [], "compressed_msgs": [], "turn_counter": 0})  # type: ignore[assignment]

    def fail_delete(_session_dir: Path) -> None:
        raise OSError("delete failed")

    monkeypatch.setattr(engine, "_build_agent", fake_build_agent)
    monkeypatch.setattr(session_lifecycle, "_delete_reset_session_files", fail_delete)

    try:
        reset_succeeded = await engine._reset_session_to_welcome("reset_me")

        assert reset_succeeded is False
        tracker = engine.todo_tracker
        assert tracker is not None
        assert tracker.serialize() == _TODOS
        assert engine._executor is not None
        assert engine._executor.history_state["chrys_todos"] == _TODOS
    finally:
        engine._active_session_guard.release()
