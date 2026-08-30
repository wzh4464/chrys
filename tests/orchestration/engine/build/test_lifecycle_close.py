# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for agent lifecycle cleanup of executor-owned resources."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.config.settings_store import LoadedSettings, SettingsHandle
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import AgentRuntimeDetails, ApprovalAutoFulfillBlocked
from chrys.foundation.models.workspace import Workspace
from chrys.foundation.recovery import RecoveryPersistOutcome
from chrys.orchestration.engine.build import construction as agent_lifecycle
from chrys.orchestration.engine.build.builder import AgentBuildResult
from chrys.orchestration.engine.run.turn_state import TurnRuntimeState
from chrys.service.agent_middleware.control.approval import ApprovalMiddleware
from chrys.service.approval.judge import JudgeVerdict
from chrys.service.approval.policy import ApprovalMode, ApprovalPolicy
from chrys.service.approval.turn_context import TurnContextHolder
from chrys.service.context.compaction.spill import SpillQuota
from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker
from chrys.service.profiles.agents.schema import AgentProfile, ApprovalConfig
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.session.runtime_metadata import SessionRuntimeMetadata


def _approval_context() -> MagicMock:
    ctx = MagicMock()
    ctx.function = SimpleNamespace(name="write_file", chrys_kind="filesystem.write")
    ctx.arguments = {"path": "/tmp/file.txt"}
    ctx.metadata = {}
    ctx.result = None
    return ctx


def _approval_policy() -> ApprovalPolicy:
    tool = MagicMock()
    tool.name = "write_file"
    tool.chrys_kind = "filesystem.write"
    return ApprovalPolicy(ApprovalConfig(default="auto", overrides={"filesystem.write": "require"}), tools=[tool])


class _FakeJudge:
    async def evaluate(self, **_kwargs: object) -> JudgeVerdict:
        return JudgeVerdict(approved=True, reason="safe")


class _FakeExecutor:
    def __init__(self, approval_middleware: ApprovalMiddleware) -> None:
        self.approval_middleware = approval_middleware
        self.history_state: dict = {}
        self.is_running = False
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        await self.approval_middleware.close()


class _FakeEngine:
    def __init__(self, bus: EventBus, executor: _FakeExecutor) -> None:
        self._agent_profile = AgentProfile(name="Code")
        self._workspace = None
        self._session_id = "session-1"
        self._bus = bus
        self._persistence = SimpleNamespace(state_store=None)
        self._history = MagicMock()
        self._executor = executor
        self._sub_agent_tools = None
        self._active_profile = None
        self._tool_names = []
        self._tool_kinds = {}
        self._skill_names = []
        self._memory_files = []
        self._runtime_meta = SessionRuntimeMetadata()
        self._runtime_details = AgentRuntimeDetails()
        self._approval_mode = ApprovalMode.AUTO
        self._allow_user_interaction = True
        self._intermediate_texts = {}
        self._injection_notify_task = None
        self._mutation_tracker = None
        self._workspace_change_tracker = WorkspaceChangeTracker()
        self._mutation_coordinator = None
        self._model_registry = None
        self._settings_handle = SettingsHandle(LoadedSettings(settings=Settings(), provenance={}))
        self._agent = None
        self._mcp_cache = MagicMock()
        self._mcp_adapter = None
        self._agent_registry = None
        self._runtime = None
        self._injection = None
        self._consumed_injections = []
        self._loop_recorder = None
        self._reminder_middleware = None
        self._approval_judge = None
        self._hook_manager = None
        self._outbox_recovery_task = None
        self._todo_tracker = None
        self._turn_state = TurnRuntimeState()
        self._turn_context = TurnContextHolder()
        self._spill_quota = SpillQuota()

    @property
    def _settings(self) -> Settings:
        return self._settings_handle.settings

    @property
    def _loaded_settings(self) -> LoadedSettings:
        return self._settings_handle.loaded

    @property
    def _session_dir(self) -> None:
        return None

    async def _cleanup_replaced_build_resources(self, *_args: object) -> None:
        return None

    def _advance_build_generation(self) -> None:
        return None

    def _publish_usage(self, *_args: object, **_kwargs: object) -> None:
        return None

    def _accumulate_sub_agent_usage(self, *_args: object, **_kwargs: object) -> None:
        return None

    def _accumulate_side_call_usage(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def _drain_usage_publishes(self) -> None:
        return None

    async def _publish_compaction(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def _publish_pre_compact(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def _publish_compress(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def _publish_load_progress(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def _save_recovery_checkpoint(self) -> None:
        return None

    async def _flush_recovery_checkpoint(self) -> None:
        return None

    async def persist_recovery_now(self) -> bool:
        return True

    async def persist_recovery_barrier(self) -> RecoveryPersistOutcome:
        return RecoveryPersistOutcome.PERSISTED


def _stage(engine: _FakeEngine) -> agent_lifecycle.StagedBuild:
    """Stage the engine's own live state, as start/soft_restart do for a rebuild."""
    return agent_lifecycle.stage_build(
        engine,  # type: ignore[arg-type]
        loaded=engine._loaded_settings,
        agent_profile=AgentProfile(name="Code"),
        workspace=engine._workspace,
        hook_manager=engine._hook_manager,
    )


async def _drive_auto_approval(approval_middleware: ApprovalMiddleware) -> None:
    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await approval_middleware.process(_approval_context(), _next)
    assert called


@pytest.mark.asyncio
async def test_agent_rebuild_closes_previous_executor_approval_handler() -> None:
    bus = EventBus()
    old_approval = ApprovalMiddleware(
        approval_policy=_approval_policy(),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=_FakeJudge(),
    )
    await _drive_auto_approval(old_approval)
    old_handler = old_approval._on_auto_fulfill_blocked
    old_executor = _FakeExecutor(old_approval)
    engine = _FakeEngine(bus, old_executor)

    new_approvals: list[ApprovalMiddleware] = []

    async def _build_agent_fn(**_kwargs: object) -> AgentBuildResult:
        new_approval = ApprovalMiddleware(
            approval_policy=_approval_policy(),
            event_bus=bus,
            approval_mode=ApprovalMode.AUTO,
            approval_judge=_FakeJudge(),
        )
        await _drive_auto_approval(new_approval)
        new_approvals.append(new_approval)
        return AgentBuildResult(
            agent=MagicMock(),
            executor=_FakeExecutor(new_approval),
            runtime=MagicMock(),
            loop_recorder=MagicMock(),
            reminder_middleware=MagicMock(),
            sub_agent_tools=None,
            mcp_adapter=None,
            skills_provider=None,
            tool_names=[],
            tool_kinds={},
            skill_names=[],
            memory_files=[],
            agent_profile_fingerprint="agent-fp",
            model_profile_fingerprint="model-fp",
            runtime_details=AgentRuntimeDetails(),
            compaction_strategy=MagicMock(),
            active_profile=ModelProfile(id="test", name="test"),
        )

    await agent_lifecycle.build_agent(
        engine, AgentProfile(name="Code"), staged=_stage(engine), build_agent_fn=_build_agent_fn
    )

    new_handler = new_approvals[0]._on_auto_fulfill_blocked
    handlers = bus._handlers[ApprovalAutoFulfillBlocked]
    assert old_executor.closed is True
    assert old_handler not in handlers
    assert new_handler in handlers
    assert old_handler != new_handler


@pytest.mark.asyncio
async def test_agent_rebuild_retargets_workspace_after_install_before_awaited_cleanup(tmp_path: Path) -> None:
    bus = EventBus()
    old_approval = _fresh_approval(bus)
    old_executor = _FakeExecutor(old_approval)
    engine = _FakeEngine(bus, old_executor)
    engine._workspace = Workspace.from_cwd(str(tmp_path))
    order: list[str] = []

    class _Tracker:
        def take_pending_notice(self) -> None:
            return None

        def retarget_roots(self, workspace: Workspace | None, *, resolve_scope: bool = True) -> None:
            assert workspace is engine._workspace
            assert engine._executor is not old_executor
            order.append("retarget")

    tracker = _Tracker()
    engine._workspace_change_tracker = tracker  # type: ignore[assignment]
    original_close = old_executor.close

    async def _close() -> None:
        order.append("close")
        await original_close()

    old_executor.close = _close  # type: ignore[method-assign]

    async def _build_agent_fn(**_kwargs: object) -> AgentBuildResult:
        return _make_build_result(_fresh_approval(bus))

    await agent_lifecycle.build_agent(
        engine, AgentProfile(name="Code"), staged=_stage(engine), build_agent_fn=_build_agent_fn
    )

    assert order == ["retarget", "close"]


@pytest.mark.anyio
@pytest.mark.parametrize("notice_enabled", [True, False])
async def test_agent_build_retargets_with_the_installed_notice_setting(tmp_path: Path, notice_enabled: bool) -> None:
    """The retarget reads the settings the build just installed: with the
    notice off the tracker is told not to probe the roots at all."""
    bus = EventBus()
    engine = _FakeEngine(bus, _FakeExecutor(_fresh_approval(bus)))
    engine._workspace = Workspace.from_cwd(str(tmp_path))
    engine._settings_handle.install(
        LoadedSettings(settings=Settings(workspace_change_notice=notice_enabled), provenance={})
    )
    seen: list[bool] = []

    class _Tracker:
        def take_pending_notice(self) -> None:
            return None

        def retarget_roots(self, workspace: Workspace | None, *, resolve_scope: bool = True) -> None:
            seen.append(resolve_scope)

    engine._workspace_change_tracker = _Tracker()  # type: ignore[assignment]

    async def _build_agent_fn(**_kwargs: object) -> AgentBuildResult:
        return _make_build_result(_fresh_approval(bus))

    await agent_lifecycle.build_agent(
        engine, AgentProfile(name="Code"), staged=_stage(engine), build_agent_fn=_build_agent_fn
    )

    assert seen == [notice_enabled]


def _make_build_result(approval: ApprovalMiddleware) -> AgentBuildResult:
    return AgentBuildResult(
        agent=MagicMock(),
        executor=_FakeExecutor(approval),
        runtime=MagicMock(),
        loop_recorder=MagicMock(),
        reminder_middleware=MagicMock(),
        sub_agent_tools=None,
        mcp_adapter=None,
        skills_provider=None,
        tool_names=[],
        tool_kinds={},
        skill_names=[],
        memory_files=[],
        agent_profile_fingerprint="agent-fp",
        model_profile_fingerprint="model-fp",
        runtime_details=AgentRuntimeDetails(),
        compaction_strategy=MagicMock(),
        active_profile=ModelProfile(id="test", name="test"),
    )


def _fresh_approval(bus: EventBus) -> ApprovalMiddleware:
    return ApprovalMiddleware(
        approval_policy=_approval_policy(),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=_FakeJudge(),
    )


@pytest.mark.asyncio
async def test_agent_rebuild_closes_coordinator_when_coordination_disabled() -> None:
    """``CHRYS_MUTATION_COORDINATION=0`` must bite on settings reload,
    not only at process start: the rebuild stamps + drops an existing
    coordinator instead of threading it into the new executor.
    """
    bus = EventBus()
    old_approval = _fresh_approval(bus)
    await _drive_auto_approval(old_approval)
    engine = _FakeEngine(bus, _FakeExecutor(old_approval))
    coordinator = MagicMock()
    engine._mutation_coordinator = coordinator
    engine._settings_handle.install(LoadedSettings(settings=Settings(mutation_coordination=False), provenance={}))

    captured: dict = {}

    async def _build_agent_fn(**kwargs: object) -> AgentBuildResult:
        captured.update(kwargs)
        new_approval = _fresh_approval(bus)
        await _drive_auto_approval(new_approval)
        return _make_build_result(new_approval)

    await agent_lifecycle.build_agent(
        engine, AgentProfile(name="Code"), staged=_stage(engine), build_agent_fn=_build_agent_fn
    )

    coordinator.close.assert_called_once_with()
    assert engine._mutation_coordinator is None
    assert captured["mutation_coordinator"] is None


@pytest.mark.asyncio
async def test_agent_rebuild_keeps_coordinator_when_coordination_enabled() -> None:
    """The teardown branch must not touch a coordinator while the
    setting stays on — same instance threads into the rebuilt executor."""
    bus = EventBus()
    old_approval = _fresh_approval(bus)
    await _drive_auto_approval(old_approval)
    engine = _FakeEngine(bus, _FakeExecutor(old_approval))
    coordinator = MagicMock()
    engine._mutation_coordinator = coordinator
    engine._settings_handle.install(LoadedSettings(settings=Settings(mutation_coordination=True), provenance={}))

    captured: dict = {}

    async def _build_agent_fn(**kwargs: object) -> AgentBuildResult:
        captured.update(kwargs)
        new_approval = _fresh_approval(bus)
        await _drive_auto_approval(new_approval)
        return _make_build_result(new_approval)

    await agent_lifecycle.build_agent(
        engine, AgentProfile(name="Code"), staged=_stage(engine), build_agent_fn=_build_agent_fn
    )

    coordinator.close.assert_not_called()
    assert engine._mutation_coordinator is coordinator
    assert captured["mutation_coordinator"] is coordinator
    assert captured["spill_quota"] is engine._spill_quota
    assert captured["persist_recovery_now"] == engine.persist_recovery_now


# ──────────────── the staged-build transaction ─────────────────────────


@pytest.mark.asyncio
async def test_a_failed_build_leaves_the_live_state_untouched_and_drains_the_candidate(tmp_path: Path) -> None:
    """A failure before the commit is a failure that never happened, live-wise:
    settings handle, workspace, hooks, coordinator and executor all stay — only
    the never-went-live coordinator candidate is stamped closed."""
    bus = EventBus()
    old_approval = _fresh_approval(bus)
    old_executor = _FakeExecutor(old_approval)
    engine = _FakeEngine(bus, old_executor)
    engine._workspace = Workspace.from_cwd(str(tmp_path / "old"))
    old_loaded = engine._loaded_settings
    old_workspace = engine._workspace
    old_hook_manager = MagicMock()
    engine._hook_manager = old_hook_manager
    old_coordinator = MagicMock()
    engine._mutation_coordinator = old_coordinator

    staged_coordinator = MagicMock()
    staged = agent_lifecycle.StagedBuild(
        loaded=LoadedSettings(settings=Settings(default_approval_mode="auto"), provenance={}),
        agent_profile=AgentProfile(name="Code"),
        workspace=Workspace.from_cwd(str(tmp_path / "new")),
        hook_manager=MagicMock(),
        mutation_coordinator=staged_coordinator,
    )

    async def _failing_build(**_kwargs: object) -> AgentBuildResult:
        raise RuntimeError("build failed")

    with pytest.raises(RuntimeError, match="build failed"):
        await agent_lifecycle.build_agent(
            engine, AgentProfile(name="Code"), staged=staged, build_agent_fn=_failing_build
        )

    assert engine._loaded_settings is old_loaded
    assert engine._workspace is old_workspace
    assert engine._hook_manager is old_hook_manager
    assert engine._mutation_coordinator is old_coordinator
    assert engine._executor is old_executor
    assert old_executor.closed is False
    old_coordinator.close.assert_not_called()
    staged_coordinator.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_the_build_commit_installs_the_staged_state_together(tmp_path: Path) -> None:
    """Six commitments, one commit point: what the build was configured from
    is exactly what goes live with the executor it produced — and the replaced
    coordinator is stamped closed only after that commit."""
    bus = EventBus()
    old_approval = _fresh_approval(bus)
    await _drive_auto_approval(old_approval)
    old_executor = _FakeExecutor(old_approval)
    engine = _FakeEngine(bus, old_executor)
    engine._workspace = Workspace.from_cwd(str(tmp_path / "old"))
    old_coordinator = MagicMock()
    engine._mutation_coordinator = old_coordinator

    staged = agent_lifecycle.StagedBuild(
        loaded=LoadedSettings(settings=Settings(default_approval_mode="auto"), provenance={}),
        agent_profile=AgentProfile(name="Code"),
        workspace=Workspace.from_cwd(str(tmp_path / "new")),
        hook_manager=MagicMock(),
        mutation_coordinator=MagicMock(),
    )
    captured: dict = {}

    async def _build_agent_fn(**kwargs: object) -> AgentBuildResult:
        captured.update(kwargs)
        new_approval = _fresh_approval(bus)
        await _drive_auto_approval(new_approval)
        return _make_build_result(new_approval)

    await agent_lifecycle.build_agent(engine, staged.agent_profile, staged=staged, build_agent_fn=_build_agent_fn)

    # The build read staged inputs, never the live fields it replaced.
    assert captured["settings"] is staged.loaded.settings
    assert captured["workspace"] is staged.workspace
    assert captured["hook_manager"] is staged.hook_manager
    assert captured["mutation_coordinator"] is staged.mutation_coordinator
    # The commit installed the same staged state, whole.
    assert engine._loaded_settings is staged.loaded
    assert engine._agent_profile is staged.agent_profile
    assert engine._workspace is staged.workspace
    assert engine._hook_manager is staged.hook_manager
    assert engine._mutation_coordinator is staged.mutation_coordinator
    assert engine._executor is not old_executor
    assert old_executor.closed is True
    old_coordinator.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_a_cancelled_post_commit_cleanup_still_leaves_the_committed_profile() -> None:
    """The awaited cleanups between the commit and the return can be
    cancelled; the profile went live inside the commit, so the session never
    runs the new executor under the old profile's name — and the remaining
    finalization steps still run before the cancellation resumes, because the
    commit already unhooked the replaced objects from every later cleanup."""
    bus = EventBus()
    old_approval = _fresh_approval(bus)
    await _drive_auto_approval(old_approval)

    class _CancelledOnClose(_FakeExecutor):
        async def close(self) -> None:
            raise asyncio.CancelledError

    old_executor = _CancelledOnClose(old_approval)
    engine = _FakeEngine(bus, old_executor)
    old_profile = engine._agent_profile
    old_coordinator = MagicMock()
    engine._mutation_coordinator = old_coordinator
    replaced_cleanups: list[tuple[object, ...]] = []

    async def _record_replaced_cleanup(*args: object) -> None:
        replaced_cleanups.append(args)

    engine._cleanup_replaced_build_resources = _record_replaced_cleanup  # type: ignore[method-assign]
    staged = agent_lifecycle.StagedBuild(
        loaded=LoadedSettings(settings=Settings(), provenance={}),
        agent_profile=AgentProfile(name="Explore"),
        workspace=None,
        hook_manager=None,
        mutation_coordinator=None,
    )

    async def _build_agent_fn(**kwargs: object) -> AgentBuildResult:
        _ = kwargs
        new_approval = _fresh_approval(bus)
        await _drive_auto_approval(new_approval)
        return _make_build_result(new_approval)

    with pytest.raises(asyncio.CancelledError):
        await agent_lifecycle.build_agent(engine, staged.agent_profile, staged=staged, build_agent_fn=_build_agent_fn)

    assert engine._agent_profile is staged.agent_profile
    assert engine._agent_profile is not old_profile
    # The cancelled executor close did not strand the later steps.
    old_coordinator.close.assert_called_once_with()
    assert replaced_cleanups != []


@pytest.mark.asyncio
async def test_a_cancelled_post_commit_cleanup_still_carries_the_preserved_history() -> None:
    """The conversation rides the commit, not the finalization after it: once
    the executor is reachable it can be saved, so a cancellation landing in
    the awaited cleanup steps must never publish an executor whose empty
    history the next save would persist over the real one."""
    bus = EventBus()
    old_approval = _fresh_approval(bus)
    await _drive_auto_approval(old_approval)

    class _CancelledOnClose(_FakeExecutor):
        async def close(self) -> None:
            raise asyncio.CancelledError

    old_executor = _CancelledOnClose(old_approval)
    engine = _FakeEngine(bus, old_executor)
    staged = agent_lifecycle.StagedBuild(
        loaded=LoadedSettings(settings=Settings(), provenance={}),
        agent_profile=AgentProfile(name="Explore"),
        workspace=None,
        hook_manager=None,
        mutation_coordinator=None,
    )
    preserved = {"messages": [{"role": "user", "content": "kept"}], "turn_counter": 3}

    async def _build_agent_fn(**kwargs: object) -> AgentBuildResult:
        _ = kwargs
        new_approval = _fresh_approval(bus)
        await _drive_auto_approval(new_approval)
        return _make_build_result(new_approval)

    with pytest.raises(asyncio.CancelledError):
        await agent_lifecycle.build_agent(
            engine,
            staged.agent_profile,
            staged=staged,
            build_agent_fn=_build_agent_fn,
            preserved_history=preserved,
        )

    assert engine._executor is not old_executor
    assert engine._executor.history_state == preserved
    engine._reminder_middleware.restore_phase4_state.assert_called_once_with(preserved)
    # The history manager rode the same commit: bound to the very dict the
    # new executor holds, not left on the replaced executor's history.
    engine._history.bind.assert_called_with(engine._executor.history_state)
    assert engine._history.bind.call_args[0][0] is engine._executor.history_state


@pytest.mark.asyncio
async def test_replaced_resource_cleanup_survives_a_cancelled_step() -> None:
    """A cancellation inside one release step must not skip the steps after
    it: the commit already swapped the live pointers, so whatever is skipped
    here is unreachable and never runs again."""

    class _CancelledAgent:
        async def __aexit__(self, *args: object) -> None:
            raise asyncio.CancelledError

    cleaned: list[str] = []

    class _Tools:
        async def cleanup(self) -> None:
            cleaned.append("sub_agent_tools")

    class _Mcp:
        async def disconnect_all(self) -> None:
            cleaned.append("mcp_adapter")

    engine = SimpleNamespace(_agent=object(), _sub_agent_tools=object(), _mcp_adapter=object())

    with pytest.raises(asyncio.CancelledError):
        await agent_lifecycle.cleanup_replaced_build_resources(engine, _CancelledAgent(), _Tools(), _Mcp())  # type: ignore[arg-type]

    assert cleaned == ["sub_agent_tools", "mcp_adapter"]


# ──────────────── intermediate-text capture callbacks (§2.1.1) ─────────


async def _build_with_captured_callbacks() -> tuple[_FakeEngine, dict]:
    """Build via the real ``build_agent`` and capture its callback kwargs."""
    bus = EventBus()
    old_approval = _fresh_approval(bus)
    await _drive_auto_approval(old_approval)
    engine = _FakeEngine(bus, _FakeExecutor(old_approval))

    captured: dict = {}

    async def _build_agent_fn(**kwargs: object) -> AgentBuildResult:
        captured.update(kwargs)
        new_approval = _fresh_approval(bus)
        await _drive_auto_approval(new_approval)
        return _make_build_result(new_approval)

    await agent_lifecycle.build_agent(
        engine, AgentProfile(name="Code"), staged=_stage(engine), build_agent_fn=_build_agent_fn
    )
    return engine, captured


@pytest.mark.asyncio
async def test_intermediate_capture_async_records_batch_id_at_capture_time() -> None:
    """The async callback maps the POST-increment batch id — the same id the
    batch's subsequent tool records are stamped with — to the captured text.
    An empty-text boundary advances the counter but writes no mapping entry.
    """
    engine, captured = await _build_with_captured_callbacks()
    on_async = captured["on_intermediate_async"]
    buffer = captured["intermediate_buffer"]

    await on_async("")  # tool-only boundary
    assert buffer.batch_id == 1
    assert engine._intermediate_texts == {}

    await on_async("first text")
    assert buffer.batch_id == 2
    assert engine._intermediate_texts == {2: "first text"}

    await on_async("")  # another boundary — mapping unchanged
    assert buffer.batch_id == 3
    assert engine._intermediate_texts == {2: "first text"}


@pytest.mark.asyncio
async def test_intermediate_capture_sync_and_async_paths_yield_identical_mappings() -> None:
    """The same event sequence — including a second pass continuing the counter
    (``reset_batch_id=False`` retry semantics: nothing resets the buffer) —
    produces the same ``batch_id → text`` mapping on both paths.
    """
    engine_a, captured_a = await _build_with_captured_callbacks()
    engine_s, captured_s = await _build_with_captured_callbacks()
    on_async = captured_a["on_intermediate_async"]
    on_sync = captured_s["on_intermediate_sync"]

    sequence = ["", "first text", "", "second text", ""]
    for text in sequence:  # pass 1 + retry continuation, uninterrupted counter
        await on_async(text)
    for text in sequence:
        on_sync(text)

    expected = {2: "first text", 4: "second text"}
    assert engine_a._intermediate_texts == expected
    assert engine_s._intermediate_texts == expected
    assert captured_a["intermediate_buffer"].batch_id == len(sequence)
    assert captured_s["intermediate_buffer"].batch_id == len(sequence)
