# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for ACP session manager workspace scoping."""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import os
import threading
from datetime import UTC, datetime, timedelta
from typing import ClassVar

import pytest
import yaml
from acp import schema as acp_schema

from chrys.app.acp import session_manager as session_manager_module
from chrys.app.acp.session_manager import AcpSessionError, AcpSessionManager, ManagedSession
from chrys.foundation.config.env_layers import freeze_process_env
from chrys.foundation.config.runtime_pointer import MODEL_POINTER_ENV, get_model_pointer
from chrys.foundation.config.settings import Settings
from chrys.foundation.config.settings_store import LoadedSettings, PersistResult, load_settings, persist
from chrys.foundation.config.spec import Source, specs_by_key
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import ProfileSwitched, Warning
from chrys.foundation.util.lock import FileLock
from chrys.kernel import Message
from chrys.service.approval.policy import ApprovalMode
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import (
    AcpAgentConfig,
    AgentProfile,
    MCPServerConfig,
    SubAgentRef,
    SubAgentsConfig,
    ToolsConfig,
)
from chrys.service.profiles.agents.serializer import save_profile as save_agent_profile
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.state.store import JsonFileStateStore, SessionMeta, StateStore
from tests.support.platform_fakes import platform_with_config_dir


class _PinnableEngine:
    """Stand-in for the engine the manager pins the ask_user timeout on."""

    def __init__(self) -> None:
        self.timeout_pinned = False
        self.snapshot: ProfileSwitched | None = None
        self.recovered_from_sidecar = False
        self.loaded_settings = LoadedSettings(settings=Settings(), provenance={})

    def pin_ask_user_timeout(self) -> None:
        self.timeout_pinned = True

    def current_profile_snapshot(self) -> ProfileSwitched:
        return self.snapshot if self.snapshot is not None else ProfileSwitched()


class _CloseHost:
    def __init__(self) -> None:
        self.shutdown_called = False
        self.engine = _PinnableEngine()

    async def shutdown(self) -> None:
        self.shutdown_called = True


class _FailingStartHost:
    instances: ClassVar[list[_FailingStartHost]] = []

    def __init__(self, **kwargs) -> None:
        self.session_id = "failed-session"
        self.shutdown_called = False
        self.engine = _PinnableEngine()
        self.event_bus = EventBus()
        _ = kwargs
        self.instances.append(self)

    async def start(self) -> None:
        raise RuntimeError("boom")

    async def shutdown(self) -> None:
        self.shutdown_called = True


class _StartedHost:
    instances: ClassVar[list[_StartedHost]] = []

    def __init__(self, **kwargs) -> None:
        self.session_id = f"started-session-{len(self.instances) + 1}"
        self.shutdown_called = False
        self.engine = _PinnableEngine()
        if "loaded_settings" in kwargs:
            self.engine.loaded_settings = kwargs["loaded_settings"]
        self.event_bus = EventBus()
        self.kwargs = kwargs
        self.instances.append(self)

    async def start(self) -> None:
        return None

    async def shutdown(self) -> None:
        self.shutdown_called = True


class _StaticListStore:
    def __init__(self, sessions: list[SessionMeta]) -> None:
        self._sessions = sessions

    async def list_sessions(self) -> list[SessionMeta]:
        return list(self._sessions)

    def session_dir(self, _session_id: str):
        raise NotImplementedError

    async def save_session(self, *_args, **_kwargs) -> None:
        raise NotImplementedError

    async def load_session(self, _session_id: str, *, prefer_recovery: bool = False):
        _ = prefer_recovery
        raise NotImplementedError

    async def load_session_raw(self, _session_id: str, *, prefer_recovery: bool = False):
        _ = prefer_recovery
        raise NotImplementedError

    async def delete_session(self, _session_id: str, *, allow_active: bool = False) -> None:
        _ = allow_active
        raise NotImplementedError


def _registries() -> tuple[AgentProfileRegistry, ModelProfileRegistry]:
    agent_registry = AgentProfileRegistry()
    agent_registry.register(AgentProfile(name="Code"))
    model_registry = ModelProfileRegistry()
    model_registry.register(ModelProfile(id="model", name="Mock"))
    return agent_registry, model_registry


def _manager(process_cwd: str | None, store: StateStore, *, profile_name: str = "Code") -> AcpSessionManager:
    agent_registry, model_registry = _registries()
    return AcpSessionManager(
        loaded_settings=LoadedSettings(settings=Settings(), provenance={}),
        profile_name=profile_name,
        approval_mode=ApprovalMode.MANUAL,
        process_cwd=process_cwd,
        state_store=store,
        agent_registry=agent_registry,
        model_registry=model_registry,
    )


def _unsupported_sse_mcp() -> list[acp_schema.SseMcpServer]:
    return [
        acp_schema.SseMcpServer(
            type="sse",
            name="events",
            url="https://example.test/sse",
            headers=[],
        )
    ]


@pytest.mark.asyncio
async def test_list_sessions_filters_to_process_cwd(tmp_path) -> None:
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["hello project"])]},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    await store.save_session(
        "other-session",
        {"messages": [Message("user", ["hello other"])]},
        agent_profile="Code",
        primary_cwd=str(other),
    )
    manager = _manager(str(project), store)

    sessions, next_cursor = await manager.list_sessions(cwd=str(project), cursor=None)

    assert next_cursor is None
    assert [session.session_id for session in sessions] == ["project-session"]
    assert sessions[0].cwd == str(project)
    assert sessions[0].field_meta == {
        "agent_profile": "Code",
        "agent_display_name": "",
        "message_count": 1,
        "model_provider": "",
        "model_api_style": "",
        "model_id": "",
    }


@pytest.mark.asyncio
async def test_list_sessions_excludes_primary_cwd_from_additional_directories(tmp_path) -> None:
    project = tmp_path / "project"
    extra = tmp_path / "extra"
    project.mkdir()
    extra.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    # ACP multi-dir sessions persist the primary cwd inside working_dirs.
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["hello project"])]},
        agent_profile="Code",
        primary_cwd=str(project),
        working_dirs=[str(project), str(extra)],
    )
    manager = _manager(str(project), store)

    sessions, _ = await manager.list_sessions(cwd=str(project), cursor=None)

    assert sessions[0].cwd == str(project)
    # The primary root is reported via cwd, not duplicated in additionalDirectories.
    assert sessions[0].additionalDirectories == [str(extra)]


@pytest.mark.asyncio
async def test_list_sessions_filters_by_additional_directories(tmp_path) -> None:
    project = tmp_path / "project"
    extra_a = tmp_path / "extra-a"
    extra_b = tmp_path / "extra-b"
    for d in (project, extra_a, extra_b):
        d.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "session-a",
        {"messages": [Message("user", ["a"])]},
        agent_profile="Code",
        primary_cwd=str(project),
        working_dirs=[str(project), str(extra_a)],
    )
    await store.save_session(
        "session-b",
        {"messages": [Message("user", ["b"])]},
        agent_profile="Code",
        primary_cwd=str(project),
        working_dirs=[str(project), str(extra_b)],
    )
    await store.save_session(
        "session-plain",
        {"messages": [Message("user", ["plain"])]},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    manager = _manager(str(project), store)

    # additionalDirectories is an exact additional-root filter: asking for [extra-a]
    # returns only the session scoped to exactly that root.
    scoped, _ = await manager.list_sessions(cwd=str(project), cursor=None, additional_directories=[str(extra_a)])
    assert [s.session_id for s in scoped] == ["session-a"]

    # Per the ACP schema, an empty list is equivalent to omitting the filter.
    empty_filter, _ = await manager.list_sessions(cwd=str(project), cursor=None, additional_directories=[])
    assert {s.session_id for s in empty_filter} == {"session-a", "session-b", "session-plain"}

    # Omitting the filter returns all sessions for the cwd (unchanged behavior).
    everything, _ = await manager.list_sessions(cwd=str(project), cursor=None)
    assert {s.session_id for s in everything} == {"session-a", "session-b", "session-plain"}


@pytest.mark.asyncio
async def test_list_sessions_rejects_relative_additional_directory(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    manager = _manager(str(project), store)

    # Filter roots must be absolute (ACP contract); a relative root would otherwise
    # resolve against the server process cwd instead of being rejected.
    with pytest.raises(AcpSessionError, match="absolute"):
        await manager.list_sessions(cwd=str(project), cursor=None, additional_directories=["../extra"])


@pytest.mark.asyncio
async def test_new_session_rejects_relative_cwd(tmp_path) -> None:
    manager = _manager(str(tmp_path), _StaticListStore([]))

    with pytest.raises(AcpSessionError, match="absolute"):
        await manager.new_session(cwd="relative/dir", mcp_servers=None)


@pytest.mark.asyncio
async def test_new_session_rejects_relative_additional_directory(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = _manager(str(project), _StaticListStore([]))

    with pytest.raises(AcpSessionError, match="absolute"):
        await manager.new_session(
            cwd=str(project),
            additional_directories=["../sibling"],
            mcp_servers=None,
        )


@pytest.mark.asyncio
async def test_list_sessions_without_cwd_before_dynamic_binding_returns_empty_page(tmp_path) -> None:
    manager = _manager(None, JsonFileStateStore(tmp_path / "sessions"))

    sessions, next_cursor = await manager.list_sessions(cwd=None, cursor=None)

    assert sessions == []
    assert next_cursor is None
    assert manager.process_cwd is None


@pytest.mark.asyncio
async def test_list_sessions_uses_explicit_cwd_without_binding_process_default(tmp_path) -> None:
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["hello project"])]},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    await store.save_session(
        "other-session",
        {"messages": [Message("user", ["hello other"])]},
        agent_profile="Code",
        primary_cwd=str(other),
    )
    manager = _manager(None, store)

    sessions, next_cursor = await manager.list_sessions(cwd=str(project), cursor=None)

    assert next_cursor is None
    assert manager.process_cwd is None
    assert [session.session_id for session in sessions] == ["project-session"]
    sessions, next_cursor = await manager.list_sessions(cwd=None, cursor=None)
    assert next_cursor is None
    assert sessions == []
    sessions, next_cursor = await manager.list_sessions(cwd=str(other), cursor=None)
    assert next_cursor is None
    assert [session.session_id for session in sessions] == ["other-session"]


@pytest.mark.asyncio
async def test_list_sessions_sorts_by_updated_at_descending(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    now = datetime(2026, 5, 18, tzinfo=UTC)
    store = _StaticListStore(
        [
            SessionMeta(
                session_id="older",
                agent_profile="Code",
                agent_display_name="",
                created_at=now - timedelta(days=2),
                updated_at=now - timedelta(days=1),
                message_count=1,
                primary_cwd=str(project),
            ),
            SessionMeta(
                session_id="newer",
                agent_profile="Code",
                agent_display_name="",
                created_at=now - timedelta(days=1),
                updated_at=now,
                message_count=1,
                primary_cwd=str(project),
            ),
        ]
    )
    manager = _manager(str(project), store)

    sessions, next_cursor = await manager.list_sessions(cwd=str(project), cursor=None)

    assert next_cursor is None
    assert [session.session_id for session in sessions] == ["newer", "older"]


@pytest.mark.asyncio
async def test_list_sessions_accepts_explicit_cwd_different_from_default(tmp_path) -> None:
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "other-session",
        {"messages": [Message("user", ["hello other"])]},
        agent_profile="Code",
        primary_cwd=str(other),
    )
    manager = _manager(str(project), store)

    sessions, next_cursor = await manager.list_sessions(cwd=str(other), cursor=None)

    assert next_cursor is None
    assert [session.session_id for session in sessions] == ["other-session"]


@pytest.mark.asyncio
async def test_load_session_rejects_saved_session_from_other_cwd(tmp_path) -> None:
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "other-session",
        {"messages": [Message("user", ["hello other"])]},
        agent_profile="Code",
        primary_cwd=str(other),
    )
    manager = _manager(str(project), store)

    with pytest.raises(AcpSessionError, match="belongs to"):
        await manager.load_session(cwd=str(project), session_id="other-session", mcp_servers=None)


@pytest.mark.asyncio
async def test_failed_load_does_not_bind_unbound_process_cwd(tmp_path) -> None:
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "other-session",
        {"messages": [Message("user", ["hello other"])]},
        agent_profile="Code",
        primary_cwd=str(other),
    )
    manager = _manager(None, store)

    with pytest.raises(AcpSessionError, match="belongs to"):
        await manager.load_session(cwd=str(project), session_id="other-session", mcp_servers=None)

    assert manager.process_cwd is None


@pytest.mark.asyncio
async def test_load_session_rejects_saved_session_without_cwd_metadata(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "legacy-session",
        {"messages": [Message("user", ["hello legacy"])]},
        agent_profile="Code",
    )
    manager = _manager(str(project), store)

    with pytest.raises(AcpSessionError, match="no saved cwd metadata"):
        await manager.load_session(cwd=str(project), session_id="legacy-session", mcp_servers=None)


@pytest.mark.asyncio
async def test_load_session_returns_existing_active_session(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["hello project"])]},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    manager = _manager(str(project), store)
    existing_host = _CloseHost()
    existing = ManagedSession(  # type: ignore[arg-type]
        session_id="project-session",
        cwd=str(project),
        profile_name="Code",
        host=existing_host,
    )
    manager._sessions["project-session"] = existing  # type: ignore[assignment]
    _FailingStartHost.instances = []
    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _FailingStartHost)

    loaded = await manager.load_session(cwd=str(project), session_id="project-session", mcp_servers=None)

    assert loaded.session is existing
    assert loaded.reused_existing is True
    assert _FailingStartHost.instances == []


@pytest.mark.asyncio
async def test_load_session_checks_active_sessions_before_persisted_list(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await asyncio.to_thread(
        store.save_recovery_session,
        "project-session",
        {"messages": [Message("user", ["recovered"])], "compressed_msgs": []},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    active_lock = FileLock(store.active_lock_path("project-session"), timeout=1.0)
    active_lock.acquire()
    manager = _manager(str(project), store)
    existing_host = _CloseHost()
    existing_host.engine.recovered_from_sidecar = True
    existing = ManagedSession(  # type: ignore[arg-type]
        session_id="project-session",
        cwd=str(project),
        profile_name="Code",
        host=existing_host,
    )
    manager._sessions["project-session"] = existing  # type: ignore[assignment]
    _FailingStartHost.instances = []
    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _FailingStartHost)

    try:
        loaded = await manager.load_session(cwd=str(project), session_id="project-session", mcp_servers=None)
    finally:
        active_lock.release()

    assert loaded.session is existing
    assert loaded.reused_existing is True
    assert loaded.recovered_from_sidecar is True
    assert _FailingStartHost.instances == []


@pytest.mark.asyncio
async def test_load_session_tracks_profile_resolved_by_restore(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["hello project"])]},
        agent_profile="OldName",
        agent_profile_id="stable-id",
        primary_cwd=str(project),
    )
    manager = _manager(str(project), store)
    _StartedHost.instances = []

    class _RenamedProfileHost(_StartedHost):
        async def start(self) -> None:
            self.engine.snapshot = ProfileSwitched(to_profile="Renamed")

    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _RenamedProfileHost)

    loaded = await manager.load_session(cwd=str(project), session_id="project-session", mcp_servers=None)

    assert loaded.session.profile_name == "Renamed"


@pytest.mark.asyncio
async def test_session_history_uses_active_recovery_source(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await asyncio.to_thread(
        store.save_recovery_session,
        "project-session",
        {"messages": [Message("user", ["recovered"])], "compressed_msgs": []},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    active_lock = FileLock(store.active_lock_path("project-session"), timeout=1.0)
    active_lock.acquire()
    manager = _manager(str(project), store)
    existing_host = _CloseHost()
    existing_host.engine.recovered_from_sidecar = True
    manager._sessions["project-session"] = ManagedSession(  # type: ignore[assignment]
        session_id="project-session",
        cwd=str(project),
        profile_name="Code",
        host=existing_host,
    )

    try:
        canonical_id, messages = await manager.session_history(cwd=str(project), session_id="project-session")
    finally:
        active_lock.release()

    assert canonical_id == "project-session"
    assert messages[0]["contents"][0]["text"] == "recovered"


@pytest.mark.asyncio
async def test_session_history_uses_inactive_recovery_source_when_it_wins(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["primary"])], "compressed_msgs": []},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    await asyncio.to_thread(
        store.save_recovery_session,
        "project-session",
        {"messages": [Message("user", ["recovery"])], "compressed_msgs": []},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    manager = _manager(str(project), store)

    canonical_id, messages = await manager.session_history(cwd=str(project), session_id="project-session")

    assert canonical_id == "project-session"
    assert messages[0]["contents"][0]["text"] == "recovery"


@pytest.mark.asyncio
async def test_session_history_ignores_external_active_recovery_sidecar(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["primary"])], "compressed_msgs": []},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    await asyncio.to_thread(
        store.save_recovery_session,
        "project-session",
        {"messages": [Message("user", ["recovery"])], "compressed_msgs": []},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    active_lock = FileLock(store.active_lock_path("project-session"), timeout=1.0)
    active_lock.acquire()
    manager = _manager(str(project), store)

    try:
        canonical_id, messages = await manager.session_history(cwd=str(project), session_id="project-session")
    finally:
        active_lock.release()

    assert canonical_id == "project-session"
    assert messages[0]["contents"][0]["text"] == "primary"


@pytest.mark.asyncio
async def test_begin_delete_session_canonicalizes_short_ids_and_scopes_by_cwd(tmp_path) -> None:
    from chrys.foundation.util.session_ids import session_short_id

    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "saved-session",
        {"messages": [Message("user", ["hello"])]},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    manager = _manager(str(project), store)

    canonical = await manager.begin_delete_session(cwd=str(project), session_id="saved-session")
    from_short = await manager.begin_delete_session(cwd=str(project), session_id=session_short_id("saved-session"))

    assert canonical == "saved-session"
    assert from_short == "saved-session"
    with pytest.raises(AcpSessionError):
        await manager.begin_delete_session(cwd=str(other), session_id="saved-session")
    with pytest.raises(AcpSessionError):
        await manager.begin_delete_session(cwd=str(project), session_id="missing-session")


@pytest.mark.asyncio
async def test_begin_delete_session_marks_an_active_target_closing_but_not_on_rejection(tmp_path) -> None:
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "saved-session",
        {"messages": [Message("user", ["hello"])]},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    manager = _manager(str(project), store)
    host = _CloseHost()
    managed = ManagedSession(  # type: ignore[arg-type]
        session_id="saved-session",
        cwd=str(project),
        profile_name="Code",
        host=host,
    )
    manager._sessions["saved-session"] = managed

    with pytest.raises(AcpSessionError):
        await manager.begin_delete_session(cwd=str(other), session_id="saved-session")
    assert managed.closing is False

    canonical = await manager.begin_delete_session(cwd=str(project), session_id="saved-session")

    assert canonical == "saved-session"
    assert managed.closing is True
    with pytest.raises(AcpSessionError, match="not active"):
        manager.get("saved-session")

    await manager.finish_delete_session(canonical)
    assert host.shutdown_called is True
    assert "saved-session" not in manager._sessions
    assert all(meta.session_id != "saved-session" for meta in await store.list_sessions())


@pytest.mark.asyncio
async def test_load_session_rejects_a_closing_session_instead_of_reusing_or_reloading_it(tmp_path, monkeypatch) -> None:
    """A session marked closing stays in the map while close/delete drain waits.

    Handing it out as ``reused_existing`` would report success for a session
    about to shut down, and falling through to a fresh load would overwrite
    the map entry the in-flight teardown is about to pop — so load must
    reject, and must not start a replacement host.
    """
    from chrys.foundation.util.session_ids import session_short_id

    project = tmp_path / "project"
    project.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["hello project"])]},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    manager = _manager(str(project), store)
    existing = ManagedSession(  # type: ignore[arg-type]
        session_id="project-session",
        cwd=str(project),
        profile_name="Code",
        host=_CloseHost(),
    )
    manager._sessions["project-session"] = existing  # type: ignore[assignment]
    await manager.begin_close("project-session")
    _FailingStartHost.instances = []
    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _FailingStartHost)

    with pytest.raises(AcpSessionError, match="closing"):
        await manager.load_session(cwd=str(project), session_id="project-session", mcp_servers=None)
    with pytest.raises(AcpSessionError, match="closing"):
        await manager.load_session(
            cwd=str(project),
            session_id=session_short_id("project-session"),
            mcp_servers=None,
        )
    with pytest.raises(AcpSessionError, match="closing"):
        await manager.session_history(cwd=str(project), session_id="project-session")

    assert _FailingStartHost.instances == []
    assert manager._sessions["project-session"] is existing


@pytest.mark.asyncio
async def test_close_waits_for_session_prompt_lock(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = _manager(str(project), JsonFileStateStore(tmp_path / "sessions"))
    host = _CloseHost()
    managed = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(project),
        profile_name="Code",
        host=host,
    )
    manager._sessions["s1"] = managed

    async with managed.prompt_lock:
        task = asyncio.create_task(manager.close("s1"))
        await asyncio.sleep(0)
        assert not task.done()
        assert host.shutdown_called is False
        with pytest.raises(AcpSessionError, match="not active"):
            manager.get("s1")

    await asyncio.wait_for(task, timeout=1)
    assert host.shutdown_called is True


@pytest.mark.asyncio
async def test_delete_session_removes_saved_session_and_shuts_down_active_host(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["hello project"])]},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    manager = _manager(str(project), store)
    host = _CloseHost()
    manager._sessions["project-session"] = ManagedSession(  # type: ignore[assignment]
        session_id="project-session",
        cwd=str(project),
        profile_name="Code",
        host=host,
    )

    await manager.delete_session(cwd=str(project), session_id="project-session")

    assert host.shutdown_called is True
    assert "project-session" not in manager._sessions
    remaining = await store.list_sessions()
    assert [session.session_id for session in remaining] == []


@pytest.mark.asyncio
async def test_new_session_wires_per_session_title_updater(tmp_path, monkeypatch) -> None:
    """Each ACP session gets its own auto-title updater whose turn callbacks
    are composed with the process-level successful-turn callback, and close()
    drains it after the host."""
    project = tmp_path / "project"
    project.mkdir()
    agent_registry, model_registry = _registries()
    process_turns: list[bool] = []
    manager = AcpSessionManager(
        loaded_settings=LoadedSettings(settings=Settings(), provenance={}),
        profile_name="Code",
        approval_mode=ApprovalMode.MANUAL,
        process_cwd=None,
        state_store=JsonFileStateStore(tmp_path / "sessions"),
        agent_registry=agent_registry,
        model_registry=model_registry,
        on_successful_turn=lambda: process_turns.append(True),
    )
    _StartedHost.instances = []
    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _StartedHost)

    managed = await manager.new_session(cwd=str(project), mcp_servers=None)
    updater = managed.title_updater
    assert updater is not None

    host = _StartedHost.instances[0]
    finished_turns: list[bool] = []
    started_turns: list[bool] = []
    monkeypatch.setattr(updater, "on_turn_finished", lambda: finished_turns.append(True))
    monkeypatch.setattr(updater, "on_turn_started", lambda: started_turns.append(True))
    host.kwargs["on_successful_turn"]()
    host.kwargs["on_turn_started"]()
    assert process_turns == [True]
    assert finished_turns == [True]
    assert started_turns == [True]

    await manager.close(managed.session_id)
    assert host.shutdown_called is True
    assert updater._closed is True


@pytest.mark.asyncio
async def test_new_session_shuts_down_host_when_start_fails(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = _manager(None, JsonFileStateStore(tmp_path / "sessions"))
    _FailingStartHost.instances = []
    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _FailingStartHost)
    original_cwd = os.getcwd()

    try:
        with pytest.raises(RuntimeError, match="boom"):
            await manager.new_session(cwd=str(project), mcp_servers=None)
        assert os.getcwd() == original_cwd
        assert manager.process_cwd is None
    finally:
        os.chdir(original_cwd)

    assert len(_FailingStartHost.instances) == 1
    assert _FailingStartHost.instances[0].shutdown_called is True


@pytest.mark.asyncio
async def test_new_session_start_failure_restores_cwd_for_prebound_manager(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = _manager(str(project), JsonFileStateStore(tmp_path / "sessions"))
    _FailingStartHost.instances = []
    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _FailingStartHost)
    original_cwd = os.getcwd()

    try:
        with pytest.raises(RuntimeError, match="boom"):
            await manager.new_session(cwd=str(project), mcp_servers=None)
        assert os.getcwd() == original_cwd
        assert manager.process_cwd == str(project)
    finally:
        os.chdir(original_cwd)

    assert len(_FailingStartHost.instances) == 1
    assert _FailingStartHost.instances[0].shutdown_called is True


@pytest.mark.asyncio
async def test_load_session_shuts_down_host_when_start_fails(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["hello project"])]},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    manager = _manager(None, store)
    _FailingStartHost.instances = []
    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _FailingStartHost)
    original_cwd = os.getcwd()

    try:
        with pytest.raises(RuntimeError, match="boom"):
            await manager.load_session(cwd=str(project), session_id="project-session", mcp_servers=None)
        assert os.getcwd() == original_cwd
        assert manager.process_cwd is None
    finally:
        os.chdir(original_cwd)

    assert len(_FailingStartHost.instances) == 1
    assert _FailingStartHost.instances[0].shutdown_called is True


@pytest.mark.asyncio
async def test_load_session_start_failure_restores_cwd_for_prebound_manager(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["hello project"])]},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    manager = _manager(str(project), store)
    _FailingStartHost.instances = []
    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _FailingStartHost)
    original_cwd = os.getcwd()

    try:
        with pytest.raises(RuntimeError, match="boom"):
            await manager.load_session(cwd=str(project), session_id="project-session", mcp_servers=None)
        assert os.getcwd() == original_cwd
        assert manager.process_cwd == str(project)
    finally:
        os.chdir(original_cwd)

    assert len(_FailingStartHost.instances) == 1
    assert _FailingStartHost.instances[0].shutdown_called is True


@pytest.mark.asyncio
async def test_new_session_allows_multiple_cwds_in_one_manager(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    manager = _manager(None, JsonFileStateStore(tmp_path / "sessions"))
    _StartedHost.instances = []
    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _StartedHost)
    original_cwd = os.getcwd()

    try:
        first = await manager.new_session(cwd=str(project), mcp_servers=None)
        second = await manager.new_session(cwd=str(other), mcp_servers=None)
    finally:
        await manager.shutdown()
        os.chdir(original_cwd)

    assert manager.process_cwd is None
    assert first.cwd == str(project)
    assert second.cwd == str(other)
    assert len(_StartedHost.instances) == 2
    assert all(instance.shutdown_called for instance in _StartedHost.instances)


@pytest.mark.asyncio
async def test_new_session_uses_process_cwd_as_default_when_request_omits_cwd(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = _manager(str(project), JsonFileStateStore(tmp_path / "sessions"))
    _StartedHost.instances = []
    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _StartedHost)

    try:
        session = await manager.new_session(cwd=None, mcp_servers=None)
    finally:
        await manager.shutdown()

    assert session.cwd == str(project)
    assert manager.process_cwd == str(project)


@pytest.mark.asyncio
async def test_new_session_profile_not_found_does_not_bind_cwd(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = _manager(None, JsonFileStateStore(tmp_path / "sessions"), profile_name="Bogus")

    with pytest.raises(AcpSessionError, match="Agent profile not found"):
        await manager.new_session(cwd=str(project), mcp_servers=None)

    assert manager.process_cwd is None


@pytest.mark.asyncio
async def test_new_session_unsupported_mcp_overlay_does_not_bind_cwd(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    manager = _manager(None, JsonFileStateStore(tmp_path / "sessions"))

    with pytest.raises(AcpSessionError, match="SSE MCP servers are not supported"):
        await manager.new_session(cwd=str(project), mcp_servers=_unsupported_sse_mcp())

    assert manager.process_cwd is None


@pytest.mark.asyncio
async def test_load_session_unsupported_mcp_overlay_does_not_bind_cwd(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["hello project"])]},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    manager = _manager(None, store)

    with pytest.raises(AcpSessionError, match="SSE MCP servers are not supported"):
        await manager.load_session(cwd=str(project), session_id="project-session", mcp_servers=_unsupported_sse_mcp())

    assert manager.process_cwd is None


def test_acp_http_mcp_overlay_uses_literal_headers() -> None:
    configs = session_manager_module._mcp_overlay(
        [
            acp_schema.HttpMcpServer(
                type="http",
                name="remote",
                url="https://example.test/mcp",
                headers=[acp_schema.HttpHeader(name="Authorization", value="Bearer {{OPENAI_API_KEY}}")],
            )
        ]
    )

    assert configs[0].headers == {"Authorization": "Bearer {{OPENAI_API_KEY}}"}
    assert configs[0].resolve_header_templates is False


def test_mcp_test_config_rejects_client_supplied_stdio() -> None:
    with pytest.raises(AcpSessionError, match="stdio MCP servers are not supported"):
        session_manager_module._mcp_config_from_data(
            {"name": "local", "transport": "stdio", "command": "dangerous-server"}
        )


def test_mcp_test_http_config_keeps_headers_literal() -> None:
    config = session_manager_module._mcp_config_from_data(
        {
            "name": "remote",
            "transport": "http",
            "url": "https://example.test/mcp",
            "headers": {"Authorization": "Bearer {{OPENAI_API_KEY}}"},
        }
    )

    assert config.transport == "http"
    assert config.headers == {"Authorization": "Bearer {{OPENAI_API_KEY}}"}
    assert config.resolve_header_templates is False


def _mcp_profile() -> AgentProfile:
    return AgentProfile(
        name="WithMcp",
        tools=ToolsConfig(
            mcp=[
                MCPServerConfig(
                    name="remote",
                    transport="http",
                    url="https://example.test/mcp",
                    headers={"Authorization": "Bearer real-token", "X-Empty": ""},
                ),
                MCPServerConfig(
                    name="local",
                    transport="stdio",
                    command="run",
                    env={"API_KEY": "real-secret"},
                ),
            ]
        ),
    )


def _acp_profile() -> AgentProfile:
    return AgentProfile(
        name="WithAcp",
        sub_agent_only=True,
        acp=AcpAgentConfig(
            command="remote-agent",
            args=["--token", "real-token", "***"],
            env={"API_KEY": "real-secret", "EMPTY": ""},
            config_options={"channel": "private", "telemetry": False},
        ),
    )


def test_read_agent_profile_masks_mcp_secrets() -> None:
    agent_registry = AgentProfileRegistry()
    agent_registry.register(_mcp_profile())
    manager = AcpSessionManager(
        loaded_settings=LoadedSettings(settings=Settings(), provenance={}),
        profile_name="WithMcp",
        approval_mode=ApprovalMode.MANUAL,
        process_cwd=None,
        state_store=_StaticListStore([]),
        agent_registry=agent_registry,
        model_registry=ModelProfileRegistry(),
    )

    data = manager.read_agent_profile("WithMcp")

    servers = {server["name"]: server for server in data["tools"]["mcp"]}
    # Non-empty secret values are masked; empty stays empty; keys preserved.
    assert servers["remote"]["headers"] == {"Authorization": "***", "X-Empty": ""}
    assert servers["local"]["env"] == {"API_KEY": "***"}


def test_read_agent_profile_masks_acp_args_env_and_string_config_options() -> None:
    agent_registry = AgentProfileRegistry()
    agent_registry.register(_acp_profile())
    manager = AcpSessionManager(
        loaded_settings=LoadedSettings(settings=Settings(), provenance={}),
        profile_name="WithAcp",
        approval_mode=ApprovalMode.MANUAL,
        process_cwd=None,
        state_store=_StaticListStore([]),
        agent_registry=agent_registry,
        model_registry=ModelProfileRegistry(),
    )

    data = manager.read_agent_profile("WithAcp")

    assert data["acp"]["args"] == ["***", "***", "***"]
    assert data["acp"]["env"] == {"API_KEY": "***", "EMPTY": "***"}
    assert data["acp"]["config_options"] == {"channel": "***", "telemetry": False}


def test_reset_agent_profile_preserves_integrations_without_cascade(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake_platform = platform_with_config_dir(tmp_path)
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)
    user_dir = tmp_path / "agents"
    agent_registry = AgentProfileRegistry()
    agent_registry.load_builtins()
    customized = agent_registry.get_builtin_template("Code")
    assert customized is not None
    customized.instructions = "custom shadow"
    customized.skills.paths = ["private-skills"]
    customized.tools.mcp = [
        MCPServerConfig(
            name="private",
            transport="http",
            url="https://example.test",
            headers={"Authorization": "Bearer secret"},
        )
    ]
    customized.memory.files = ["private.md"]
    save_agent_profile(customized, target_dir=user_dir)

    agent_registry = AgentProfileRegistry()
    agent_registry.load_all(user_dir=user_dir)
    parent = AgentProfile(
        name="Parent",
        sub_agents=SubAgentsConfig(agents=[SubAgentRef(profile="Code")]),
    )
    agent_registry.register(parent)
    manager = AcpSessionManager(
        loaded_settings=LoadedSettings(settings=Settings(), provenance={}),
        profile_name="Code",
        approval_mode=ApprovalMode.MANUAL,
        process_cwd=None,
        state_store=_StaticListStore([]),
        agent_registry=agent_registry,
        model_registry=ModelProfileRegistry(),
    )

    result = manager.reset_agent_profile("Code")

    restored = agent_registry.get("Code")
    assert result["changed"] is True
    assert result["profile"]["builtin"] is True
    assert restored is not None
    assert restored.instructions != "custom shadow"
    assert restored.skills.paths == ["private-skills"]
    assert restored.tools.mcp[0].headers == {"Authorization": "Bearer secret"}
    assert restored.memory.files == ["private.md"]
    assert (user_dir / "Code.yaml").exists()
    assert [ref.profile for ref in parent.sub_agents.agents] == ["Code"]

    assert manager.reset_agent_profile("Code")["changed"] is False


def test_reset_agent_profile_without_shadow_is_noop(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake_platform = platform_with_config_dir(tmp_path)
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)
    agent_registry = AgentProfileRegistry()
    agent_registry.load_all(user_dir=tmp_path / "agents")
    manager = _acp_manager(agent_registry)

    result = manager.reset_agent_profile("Code")

    assert result["changed"] is False
    assert result["profile"]["builtin"] is True
    profiles = {profile["name"]: profile for profile in manager.list_agent_profiles()}
    assert profiles["Code"]["builtin"] is True


def test_reset_agent_profile_exact_template_deletes_canonicalized_shadow(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fake_platform = platform_with_config_dir(tmp_path)
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)
    user_dir = tmp_path / "agents"
    source_registry = AgentProfileRegistry()
    source_registry.load_builtins()
    customized = source_registry.get_builtin_template("Code")
    assert customized is not None
    customized.instructions = "custom shadow"
    customized.skills.script_extensions = sorted(customized.skills.script_extensions)
    save_agent_profile(customized, target_dir=user_dir)
    agent_registry = AgentProfileRegistry()
    agent_registry.load_all(user_dir=user_dir)
    manager = _acp_manager(agent_registry)

    result = manager.reset_agent_profile("Code")

    assert result["changed"] is True
    assert not (user_dir / "Code.yaml").exists()


def test_reset_agent_profile_removes_noncanonical_shadow_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A shadow stored as ``my-code.yaml`` must not resurrect the customization after restart."""
    fake_platform = platform_with_config_dir(tmp_path)
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)
    user_dir = tmp_path / "agents"
    source_registry = AgentProfileRegistry()
    source_registry.load_builtins()
    customized = source_registry.get_builtin_template("Code")
    assert customized is not None
    customized.instructions = "custom shadow"
    save_agent_profile(customized, target_dir=user_dir)
    (user_dir / "Code.yaml").rename(user_dir / "my-code.yaml")
    agent_registry = AgentProfileRegistry()
    agent_registry.load_all(user_dir=user_dir)
    manager = _acp_manager(agent_registry)
    assert agent_registry.get("Code").instructions == "custom shadow"

    result = manager.reset_agent_profile("Code")

    assert result["changed"] is True
    assert not (user_dir / "my-code.yaml").exists()
    assert sorted(p.name for p in user_dir.glob("*.y*ml")) == []
    fresh = AgentProfileRegistry()
    fresh.load_all(user_dir=user_dir)
    assert fresh.get("Code").instructions != "custom shadow"


def test_delete_agent_profile_rejects_builtin(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    fake_platform = platform_with_config_dir(tmp_path)
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)
    agent_registry = AgentProfileRegistry()
    agent_registry.load_all(user_dir=tmp_path / "agents")
    manager = _acp_manager(agent_registry)

    with pytest.raises(AcpSessionError, match="cannot be deleted"):
        manager.delete_agent_profile("Code")


def test_restore_masked_mcp_secrets_recovers_originals() -> None:
    existing = _mcp_profile()
    incoming = {
        "name": "WithMcp",
        "tools": {
            "mcp": [
                {"name": "remote", "headers": {"Authorization": "***", "X-New": "added"}},
                {"name": "local", "env": {"API_KEY": "***"}},
                {"name": "unknown", "headers": {"Authorization": "***"}},
            ]
        },
    }

    session_manager_module._restore_masked_mcp_secrets(incoming, existing)

    servers = {server["name"]: server for server in incoming["tools"]["mcp"]}
    assert servers["remote"]["headers"] == {"Authorization": "Bearer real-token", "X-New": "added"}
    assert servers["local"]["env"] == {"API_KEY": "real-secret"}
    # No matching prior server -> masked sentinel left untouched (not inventable).
    assert servers["unknown"]["headers"] == {"Authorization": "***"}


def test_restore_masked_acp_secrets_requires_unchanged_all_masked_argument_shape() -> None:
    existing = _acp_profile()
    round_trip = {
        "acp": {
            "args": ["***", "***", "***"],
            "env": {"API_KEY": "***"},
            "config_options": {"channel": "***", "telemetry": False},
        }
    }

    restoration = session_manager_module._restore_masked_acp_secrets(round_trip, existing)

    assert restoration.args_restored is True
    assert round_trip["acp"]["args"] == ["--token", "real-token", "***"]
    assert round_trip["acp"]["env"] == {"API_KEY": "real-secret"}
    assert round_trip["acp"]["config_options"] == {"channel": "private", "telemetry": False}

    edited = {"acp": {"args": ["--token", "***"], "env": {}, "config_options": {}}}
    assert session_manager_module._restore_masked_acp_secrets(edited, existing).args_restored is False
    assert edited["acp"]["args"] == ["--token", "***"]

    literal = _acp_profile()
    assert literal.acp is not None
    literal.acp.args = ["***"]
    literal_round_trip = {"acp": {"args": ["***"], "env": {}, "config_options": {}}}
    literal_restored = session_manager_module._restore_masked_acp_secrets(literal_round_trip, literal)
    session_manager_module._reject_unresolved_masked_acp_secrets(
        literal_round_trip,
        restoration=literal_restored,
    )


def test_restore_masked_acp_secrets_round_trips_literal_mask_values_in_mappings() -> None:
    existing = _acp_profile()
    assert existing.acp is not None
    existing.acp.env = {"API_KEY": "***"}
    existing.acp.config_options = {"channel": "***", "telemetry": False}
    round_trip = {
        "acp": {
            "args": [],
            "env": {"API_KEY": "***"},
            "config_options": {"channel": "***", "telemetry": False},
        }
    }

    restoration = session_manager_module._restore_masked_acp_secrets(round_trip, existing)

    assert restoration.env_keys == {"API_KEY"}
    assert restoration.option_keys == {"channel"}
    # A stored secret that IS the literal mask must survive read → write.
    session_manager_module._reject_unresolved_masked_acp_secrets(round_trip, restoration=restoration)
    assert round_trip["acp"]["env"] == {"API_KEY": "***"}
    assert round_trip["acp"]["config_options"] == {"channel": "***", "telemetry": False}

    added = {
        "acp": {
            "args": [],
            "env": {"API_KEY": "***", "NEW_SECRET": "***"},
            "config_options": {},
        }
    }
    added_restoration = session_manager_module._restore_masked_acp_secrets(added, existing)
    with pytest.raises(AcpSessionError, match="still masked"):
        session_manager_module._reject_unresolved_masked_acp_secrets(added, restoration=added_restoration)


def test_reject_unresolved_acp_arg_masks_but_accept_literal_star_in_full_list() -> None:
    with pytest.raises(AcpSessionError, match="complete unmasked argument list"):
        session_manager_module._reject_unresolved_masked_acp_secrets(
            {"acp": {"args": ["***", "***"], "env": {}, "config_options": {}}}
        )

    session_manager_module._reject_unresolved_masked_acp_secrets(
        {"acp": {"args": ["--literal", "***"], "env": {}, "config_options": {}}}
    )


def test_reject_partially_masked_acp_args_after_masked_read_edit() -> None:
    existing = _acp_profile()
    # After a masked read of a non-empty stored list, positional identity is
    # lost the moment the list is edited: ANY leftover mask is a placeholder
    # whose acceptance would silently persist the literal "***" over the
    # stored secrets.
    for edited_args in (
        ["***", "***", "--verbose"],  # same length, partially replaced
        ["***", "--verbose"],  # shortened, mask left behind
        ["***", "***", "***", "--verbose"],  # extended, masks left behind
    ):
        edited = {"acp": {"args": list(edited_args), "env": {}, "config_options": {}}}
        restoration = session_manager_module._restore_masked_acp_secrets(edited, existing)
        assert restoration.args_restored is False
        assert restoration.args_were_masked is True
        with pytest.raises(AcpSessionError, match="complete unmasked argument list"):
            session_manager_module._reject_unresolved_masked_acp_secrets(edited, restoration=restoration)

    # A fully re-entered unmasked list is the documented way out.
    clean = {"acp": {"args": ["--token", "new-token", "--verbose"], "env": {}, "config_options": {}}}
    clean_restoration = session_manager_module._restore_masked_acp_secrets(clean, existing)
    session_manager_module._reject_unresolved_masked_acp_secrets(clean, restoration=clean_restoration)
    assert clean["acp"]["args"] == ["--token", "new-token", "--verbose"]

    # Literal "***" for an existing profile: clear the stored list in one
    # write, then re-add it — with an empty stored list the mask can only be
    # literal input, so the mixed shape is accepted (two-step escape hatch).
    cleared = _acp_profile()
    assert cleared.acp is not None
    cleared.acp.args = []
    literal = {"acp": {"args": ["--literal", "***"], "env": {}, "config_options": {}}}
    literal_restoration = session_manager_module._restore_masked_acp_secrets(literal, cleared)
    assert literal_restoration.args_were_masked is False
    session_manager_module._reject_unresolved_masked_acp_secrets(literal, restoration=literal_restoration)


@pytest.mark.parametrize(
    "acp",
    [
        {"args": [], "env": {"NEW_SECRET": "***"}, "config_options": {}},
        {"args": [], "env": {}, "config_options": {"new-option": "***"}},
    ],
)
def test_reject_unresolved_acp_mapping_masks(acp: dict[str, object]) -> None:
    with pytest.raises(AcpSessionError, match="still masked"):
        session_manager_module._reject_unresolved_masked_acp_secrets({"acp": acp})


def test_write_agent_profile_rejects_unresolved_masked_mcp_secret_after_rename() -> None:
    agent_registry = AgentProfileRegistry()
    agent_registry.register(_mcp_profile())
    manager = AcpSessionManager(
        loaded_settings=LoadedSettings(settings=Settings(), provenance={}),
        profile_name="WithMcp",
        approval_mode=ApprovalMode.MANUAL,
        process_cwd=None,
        state_store=_StaticListStore([]),
        agent_registry=agent_registry,
        model_registry=ModelProfileRegistry(),
    )

    with pytest.raises(AcpSessionError, match="still masked"):
        manager.write_agent_profile(
            {
                "name": "WithMcp",
                "tools": {
                    "mcp": [
                        {
                            "name": "renamed",
                            "transport": "http",
                            "url": "https://example.test/mcp",
                            "headers": {"Authorization": "***"},
                        }
                    ]
                },
            }
        )


def _acp_manager(agent_registry: AgentProfileRegistry) -> AcpSessionManager:
    return AcpSessionManager(
        loaded_settings=LoadedSettings(settings=Settings(), provenance={}),
        profile_name="WithAcp",
        approval_mode=ApprovalMode.MANUAL,
        process_cwd=None,
        state_store=_StaticListStore([]),
        agent_registry=agent_registry,
        model_registry=ModelProfileRegistry(),
    )


def test_write_agent_profile_rename_restores_masked_secrets_via_stable_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    fake_platform = platform_with_config_dir(tmp_path)
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)
    original = _acp_profile()
    original.id = "acp-stable-id"
    agent_registry = AgentProfileRegistry()
    agent_registry.register(original)
    manager = _acp_manager(agent_registry)

    data = manager.read_agent_profile("WithAcp")
    assert data["id"] == "acp-stable-id"
    assert data["acp"]["args"] == ["***", "***", "***"]
    data["name"] = "RenamedAcp"

    manager.write_agent_profile(data)

    renamed = agent_registry.get("RenamedAcp")
    assert renamed is not None
    assert renamed.acp is not None
    # ``id`` is the identity that survives renames: the all-masked
    # round-trip restores the stored secrets instead of persisting "***".
    assert renamed.acp.args == ["--token", "real-token", "***"]
    assert renamed.acp.env == {"API_KEY": "real-secret", "EMPTY": ""}
    assert renamed.acp.config_options == {"channel": "private", "telemetry": False}


def test_write_agent_profile_rename_rejects_partially_masked_acp_args() -> None:
    original = _acp_profile()
    original.id = "acp-stable-id"
    agent_registry = AgentProfileRegistry()
    agent_registry.register(original)
    manager = _acp_manager(agent_registry)

    data = manager.read_agent_profile("WithAcp")
    data["name"] = "RenamedAcp"
    # Positional identity was lost when the masked list was edited — a
    # leftover mask is a placeholder even across a rename, and accepting it
    # would persist literal "***" into the renamed profile.
    data["acp"]["args"] = ["***", "***", "--verbose"]

    with pytest.raises(AcpSessionError, match="complete unmasked argument list"):
        manager.write_agent_profile(data)
    assert agent_registry.get("RenamedAcp") is None


def test_write_agent_profile_duplicate_id_twins_fail_closed_on_masked_args() -> None:
    twin_a = _acp_profile()
    twin_a.id = "twin-id"
    twin_b = _acp_profile()
    twin_b.name = "WithAcpCopy"
    twin_b.id = "twin-id"
    agent_registry = AgentProfileRegistry()
    agent_registry.register(twin_a)
    agent_registry.register(twin_b)
    manager = _acp_manager(agent_registry)

    # Two stored profiles share the id, so there is no single source of
    # truth for restoration — leftover masks must fail closed, not guess.
    payload = {
        "name": "RenamedAgain",
        "id": "twin-id",
        "acp": {"command": "remote-agent", "args": ["***", "--verbose"], "env": {}, "config_options": {}},
    }

    with pytest.raises(AcpSessionError, match="complete unmasked argument list"):
        manager.write_agent_profile(payload)
    assert agent_registry.get("RenamedAgain") is None


@pytest.mark.parametrize(
    ("method", "payload"),
    [
        ("write_agent_profile", {"name": "../escape"}),
        ("delete_agent_profile", "../escape"),
        ("reset_agent_profile", "../escape"),
        ("write_model_profile", {"id": "../escape", "name": "Bad"}),
        ("delete_model_profile", "../escape"),
        # Windows-reserved names share the loader predicate.
        ("write_agent_profile", {"name": "CON"}),
        ("delete_agent_profile", "foo:bar"),
        ("reset_agent_profile", "nul.txt"),
        ("write_model_profile", {"id": "aux", "name": "Bad"}),
    ],
)
def test_profile_write_and_delete_reject_path_values(method: str, payload: object) -> None:
    manager = _manager(None, _StaticListStore([]))

    with pytest.raises(AcpSessionError, match="not a path"):
        getattr(manager, method)(payload)


def test_write_model_profile_rejects_masked_api_key_for_new_profile() -> None:
    manager = _manager(None, _StaticListStore([]))

    # Read-copy-write with a new id: no existing profile to restore the secret from,
    # so a still-masked api_key must be rejected rather than persisted literally.
    with pytest.raises(AcpSessionError, match="still masked"):
        manager.write_model_profile({"id": "copy-of-model", "name": "Copy", "api_key": "***"})


def test_write_model_profile_clears_emptied_fields_but_keeps_api_key(monkeypatch) -> None:
    agent_registry = AgentProfileRegistry()
    agent_registry.register(AgentProfile(name="Code"))
    model_registry = ModelProfileRegistry()
    model_registry.register(
        ModelProfile(id="model", name="Mock", api_key="real-secret", base_url="https://old.example/v1")
    )
    manager = AcpSessionManager(
        loaded_settings=LoadedSettings(settings=Settings(), provenance={}),
        profile_name="Code",
        approval_mode=ApprovalMode.MANUAL,
        process_cwd=None,
        state_store=_StaticListStore([]),
        agent_registry=agent_registry,
        model_registry=model_registry,
    )
    saved: dict[str, ModelProfile] = {}

    def _fake_save(profile: ModelProfile):
        saved["profile"] = profile
        return "/tmp/fake.yaml"

    monkeypatch.setattr(session_manager_module, "save_model_profile", _fake_save)

    manager.write_model_profile({"id": "model", "name": "Mock", "base_url": "", "api_key": "***"})

    # Explicit "" clears base_url; masked api_key preserves the stored secret.
    assert saved["profile"].base_url == ""
    assert saved["profile"].api_key == "real-secret"


@pytest.mark.asyncio
async def test_set_workspace_rejects_nonexistent_dir(tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import WorkspaceChange

    manager = _manager(str(tmp_path), _StaticListStore([]))
    bus = EventBus()
    published: list[WorkspaceChange] = []

    async def _capture(event: WorkspaceChange) -> None:
        published.append(event)

    await bus.subscribe(WorkspaceChange, _capture)
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_InjectHost(bus, is_turn_active=True),  # type: ignore[arg-type]
    )

    with pytest.raises((AcpSessionError, FileNotFoundError)):
        await manager.set_workspace("s1", str(tmp_path / "does-not-exist"))

    # Validation happens before publishing, so no broken workspace is rebuilt/persisted.
    assert published == []


def _redirect_config_dir(monkeypatch: pytest.MonkeyPatch, config_dir) -> None:
    """Point the settings document at a tmp dir and clear ambient mirrors."""
    fake_platform = platform_with_config_dir(config_dir)
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)
    for key in ("CHRYS_DEFAULT_APPROVAL_MODE", "CHRYS_ROLLBACK_SNAPSHOTS_KEEP", "CHRYS_THEME"):
        monkeypatch.delenv(key, raising=False)


def _stored_setting(config_dir, dotted: str) -> object | None:
    settings_path = config_dir / "settings.yaml"
    if not settings_path.exists():
        return None
    node: object = yaml.safe_load(settings_path.read_text(encoding="utf-8"))
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def test_set_config_option_stores_newline_value_without_injection(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The dotenv-injection vector is gone: YAML stores the newline verbatim."""
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)

    manager.set_config_option("theme", "dark\nCHRYS_INJECTED=evil")

    assert _stored_setting(tmp_path, "ui.theme") == "dark\nCHRYS_INJECTED=evil"
    assert "CHRYS_INJECTED" not in os.environ
    assert not (tmp_path / ".env").exists()


def test_set_config_option_maps_empty_value_to_removal(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)
    env_path = tmp_path / ".env"
    env_original = "CHRYS_THEME=dark\nKEEP=value\n"
    env_path.write_text(env_original, encoding="utf-8")
    manager.set_config_option("theme", "dark")

    result = manager.set_config_option("theme", None)

    assert result["value"] == ""
    assert _stored_setting(tmp_path, "ui.theme") is None
    # The user's dotenv is not this write path's to clean up — migration owns it.
    assert env_path.read_text(encoding="utf-8") == env_original


def test_two_concurrent_config_writes_both_reach_the_base_settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Neither client's value may be lost from the settings new sessions start on."""
    # Two clients configuring two sessions run this transaction in separate
    # worker threads, and each one writes the document and then re-reads it.
    # A refresh that publishes a snapshot taken before the other client's write
    # leaves the manager handing every session created afterwards a value that
    # is already on disk.
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)
    # Without a frozen snapshot the file layers are not read at all, and both
    # refreshes would agree on the defaults for the wrong reason.
    freeze_process_env()
    first_read = threading.Event()
    second_published = threading.Event()
    reads = itertools.count()

    def load_with_a_gap(*args: object, **kwargs: object) -> LoadedSettings:
        first = next(reads) == 0
        loaded = load_settings(*args, **kwargs)  # type: ignore[arg-type]
        if first:
            # Hold the first reader between its read and its publish: that gap
            # is the only window in which the other write can be lost.
            first_read.set()
            second_published.wait(0.3)
        return loaded

    monkeypatch.setattr("chrys.app.acp.session_manager.load_settings", load_with_a_gap)

    def write_theme() -> None:
        manager.set_config_option("theme", "dark")

    def write_default_agent() -> None:
        first_read.wait(10)
        manager.set_config_option("default_agent", "Reviewer")
        second_published.set()

    writers = [threading.Thread(target=write_theme), threading.Thread(target=write_default_agent)]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(30)

    assert _stored_setting(tmp_path, "ui.theme") == "dark"
    assert _stored_setting(tmp_path, "agent.default_profile") == "Reviewer"
    base = {entry["key"]: entry["baseValue"] for entry in manager.get_config_options()["options"]}
    assert (base["theme"], base["default_agent"]) == ("dark", "Reviewer")


def test_session_loaded_settings_derives_each_sessions_own_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Two sessions in two roots each get their own project trust domain,
    and neither leaks into the manager's deliberately project-free base."""
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path / "config")
    monkeypatch.delenv("CHRYS_SESSION_TITLE_AUTO", raising=False)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "settings.yaml").write_text("project:\n  config_enabled: true\n", encoding="utf-8")
    quiet = tmp_path / "quiet"
    (quiet / ".chrys").mkdir(parents=True)
    (quiet / ".chrys" / "settings.yaml").write_text("session:\n  title:\n    auto: false\n", encoding="utf-8")
    loud = tmp_path / "loud"
    loud.mkdir()
    freeze_process_env()

    quiet_loaded = manager._session_loaded_settings(str(quiet))
    loud_loaded = manager._session_loaded_settings(str(loud))

    assert quiet_loaded.settings.session_title_auto is False
    assert quiet_loaded.source_for("session.title.auto").layer is Source.PROJECT
    assert loud_loaded.settings.session_title_auto is True
    assert manager._loaded_settings.settings.session_title_auto is True
    # The launch-time timeout stays the CLI layer's, not a re-read's.
    assert quiet_loaded.source_for("tools.ask_user.timeout_seconds").layer is Source.CLI


@pytest.mark.asyncio
async def test_new_session_collects_project_settings_warnings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Session creation runs outside any prompt turn, so the load's verdicts
    reach the caller through the collector — here, a dormant project file."""
    _redirect_config_dir(monkeypatch, tmp_path / "config")
    project = tmp_path / "project"
    (project / ".chrys").mkdir(parents=True)
    (project / ".chrys" / "settings.yaml").write_text("session:\n  title:\n    auto: false\n", encoding="utf-8")
    freeze_process_env()
    manager = _manager(None, JsonFileStateStore(tmp_path / "sessions"))
    _StartedHost.instances = []
    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _StartedHost)
    warnings: list[Warning] = []

    await manager.new_session(cwd=str(project), mcp_servers=None, warnings=warnings)

    assert [event.code for event in warnings] == ["project_config_dormant"]


@pytest.mark.asyncio
async def test_load_session_collects_project_settings_warnings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The committed restore load's verdicts reach the caller through the
    same collector new_session uses."""
    _redirect_config_dir(monkeypatch, tmp_path / "config")
    project = tmp_path / "project"
    (project / ".chrys").mkdir(parents=True)
    (project / ".chrys" / "settings.yaml").write_text("session:\n  title:\n    auto: false\n", encoding="utf-8")
    freeze_process_env()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["hello project"])]},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    manager = _manager(None, store)
    _StartedHost.instances = []
    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _StartedHost)
    warnings: list[Warning] = []

    loaded = await manager.load_session(
        cwd=str(project), session_id="project-session", mcp_servers=None, warnings=warnings
    )

    assert loaded.reused_existing is False
    assert [event.code for event in warnings] == ["project_config_dormant"]


@pytest.mark.asyncio
async def test_load_session_reports_the_committed_restore_loads_warnings(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The restore re-reads settings; an edit landing between the manager's
    pre-restore snapshot and that committed load must be reported as the
    session will actually run — the committed verdicts, not the snapshot's."""
    _redirect_config_dir(monkeypatch, tmp_path / "config")
    project = tmp_path / "project"
    (project / ".chrys").mkdir(parents=True)
    (project / ".chrys" / "settings.yaml").write_text("session:\n  title:\n    auto: false\n", encoding="utf-8")
    freeze_process_env()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["hello project"])]},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    manager = _manager(None, store)
    _StartedHost.instances = []

    class _RestoringHost(_StartedHost):
        async def start(self) -> None:
            # The restore's own load saw one more problem than the snapshot.
            self.engine.loaded_settings = dataclasses.replace(
                self.kwargs["loaded_settings"], unknown_keys=("mystery.key",)
            )

    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _RestoringHost)
    warnings: list[Warning] = []

    await manager.load_session(cwd=str(project), session_id="project-session", mcp_servers=None, warnings=warnings)

    assert [event.code for event in warnings] == ["setting_unknown_keys", "project_config_dormant"]


def test_two_concurrent_model_writes_leave_the_pointer_agreeing_with_the_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The live model and the stored model must not be two different profiles."""
    # The pointer outranks the document it was written from, so an interleave
    # that commits one profile to disk and installs the other as the pointer is
    # not merely a lost update: this process runs one model and would come back
    # on the other after a restart, with no write in between to explain it.
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)
    freeze_process_env()
    # Claim the carrier so an installed pointer never outlives the test.
    monkeypatch.setenv(MODEL_POINTER_ENV, "claimed")
    monkeypatch.delenv(MODEL_POINTER_ENV)
    first_stored = threading.Event()
    second_done = threading.Event()
    writes = itertools.count()

    def persist_with_a_gap(*args: object, **kwargs: object) -> PersistResult:
        result = persist(*args, **kwargs)  # type: ignore[arg-type]
        if next(writes) == 0:
            # Hold the first writer between its document commit and its pointer
            # write — the window where the two can end up disagreeing.
            first_stored.set()
            second_done.wait(0.3)
        return result

    monkeypatch.setattr("chrys.app.acp.session_manager.persist", persist_with_a_gap)

    def write_model_a() -> None:
        manager.set_config_option("model_profile", "model-a")

    def write_model_b() -> None:
        first_stored.wait(10)
        manager.set_config_option("model_profile", "model-b")
        second_done.set()

    writers = [threading.Thread(target=write_model_a), threading.Thread(target=write_model_b)]
    for writer in writers:
        writer.start()
    for writer in writers:
        writer.join(30)

    stored = _stored_setting(tmp_path, "model.profile.active")
    base = {entry["key"]: entry["baseValue"] for entry in manager.get_config_options()["options"]}
    assert stored in {"model-a", "model-b"}
    assert (get_model_pointer()[0], base["model_profile"]) == (stored, stored)


def test_a_config_read_cannot_land_between_a_write_and_its_republish(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """``value`` and ``baseValue`` must describe one state, or the reply is a lie."""
    # The document's own lock is released at the commit; the base settings are
    # only republished at the end of the transaction. A read in that gap pairs
    # the newly stored value with the superseded effective one — two answers to
    # "what is this option" that contradict each other.
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)
    freeze_process_env()
    committed = threading.Event()
    released = threading.Event()

    def persist_then_stall(*args: object, **kwargs: object) -> PersistResult:
        result = persist(*args, **kwargs)  # type: ignore[arg-type]
        committed.set()
        released.wait(10)
        return result

    monkeypatch.setattr("chrys.app.acp.session_manager.persist", persist_then_stall)

    payloads: list[dict[str, object]] = []
    writer = threading.Thread(target=lambda: manager.set_config_option("theme", "dark"))
    reader = threading.Thread(target=lambda: payloads.append(manager.get_config_options()))

    writer.start()
    assert committed.wait(10)
    reader.start()
    reader.join(0.3)
    assert reader.is_alive(), "the read entered the writer's transaction"
    released.set()
    for thread in (writer, reader):
        thread.join(30)

    options = {entry["key"]: entry for entry in payloads[0]["options"]}  # type: ignore[union-attr]
    assert (options["theme"]["value"], options["theme"]["baseValue"]) == ("dark", "dark")


def test_set_config_option_downgrades_bypass_approval_mode(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)

    result = manager.set_config_option("default_approval_mode", "bypass")

    assert _stored_setting(tmp_path, "approval.default_mode") == "auto"
    assert result["value"] == "auto"
    # No mirror: an environment copy would come back as the ENV layer and
    # outrank the document this just wrote.
    assert "CHRYS_DEFAULT_APPROVAL_MODE" not in os.environ


def test_set_config_option_keeps_non_bypass_approval_mode(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)

    manager.set_config_option("default_approval_mode", "manual")

    assert _stored_setting(tmp_path, "approval.default_mode") == "manual"


def test_set_config_option_rejects_non_int_rollback_snapshots(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)

    # A non-integer would be persisted then break the next settings load,
    # so it must be rejected before any document write.
    with pytest.raises(AcpSessionError, match="rollback_snapshots_keep"):
        manager.set_config_option("rollback_snapshots_keep", "abc")

    assert not (tmp_path / "settings.yaml").exists()


def test_set_config_option_accepts_int_rollback_snapshots(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)

    result = manager.set_config_option("rollback_snapshots_keep", "5")

    # Stored canonical (an int), rendered on the wire as text.
    assert _stored_setting(tmp_path, "rollback.snapshots_keep") == 5
    assert result["value"] == "5"


def test_set_config_option_accepts_the_legacy_env_spelling(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Old clients send ``CHRYS_*`` names; both grammars land on the same key."""
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)

    result = manager.set_config_option("CHRYS_THEME", "dark")

    assert _stored_setting(tmp_path, "ui.theme") == "dark"
    assert result["key"] == "CHRYS_THEME"
    assert result["envKey"] == "CHRYS_THEME"
    assert result["settingKey"] == "ui.theme"


def test_set_config_option_rejects_unknown_keys_in_either_grammar(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)

    with pytest.raises(AcpSessionError, match="Unsupported config option"):
        manager.set_config_option("mystery", "x")
    with pytest.raises(AcpSessionError, match="Unsupported config option"):
        manager.set_config_option("CHRYS_MYSTERY", "x")
    assert not (tmp_path / "settings.yaml").exists()


def test_config_option_descriptors_agree_with_the_settings_specs() -> None:
    """Three names, one key: the wire aliases must track the spec declarations."""
    specs = specs_by_key(Settings)
    for option in session_manager_module._SUPPORTED_CONFIG_OPTIONS:
        assert specs[option.setting_key].env == option.env_alias, option.logical_key
        assert specs[option.setting_key].persist, option.logical_key


def test_get_config_options_reports_document_value_and_base_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)
    freeze_process_env()
    manager.set_config_option("theme", "dark")

    result = manager.get_config_options()

    assert "sessionId" not in result
    options = {entry["key"]: entry for entry in result["options"]}
    assert options["theme"] == {
        "key": "theme",
        "envKey": "CHRYS_THEME",
        "settingKey": "ui.theme",
        "value": "dark",
        "baseValue": "dark",
        "baseSource": "user",
    }
    # A key the document does not hold has no durable value, but the base
    # pair still answers with the built-in default.
    assert options["rollback_snapshots_keep"]["value"] == ""
    assert options["rollback_snapshots_keep"]["baseValue"] == str(Settings().rollback_snapshots_keep)
    assert options["rollback_snapshots_keep"]["baseSource"] == "default"


class _SettingsScopeEngine:
    def __init__(self, loaded: LoadedSettings) -> None:
        self.loaded_settings = loaded


class _SettingsScopeHost:
    def __init__(self, loaded: LoadedSettings) -> None:
        self.engine = _SettingsScopeEngine(loaded)


def test_get_config_options_with_a_session_id_adds_that_sessions_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)
    session_loaded = LoadedSettings(settings=Settings(), provenance={}).overlay(Source.CLI, theme="session-dark")
    manager._sessions["s1"] = ManagedSession(
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_SettingsScopeHost(session_loaded),  # type: ignore[arg-type]
    )

    result = manager.get_config_options("s1")

    assert result["sessionId"] == "s1"
    options = {entry["key"]: entry for entry in result["options"]}
    assert options["theme"]["sessionValue"] == "session-dark"
    assert options["theme"]["sessionSource"] == "cli"
    # The base pair keeps answering for the manager, never renamed to look
    # like the session's own view.
    assert options["theme"]["baseValue"] == Settings().theme
    assert options["theme"]["baseSource"] == "default"


def test_get_config_options_rejects_an_unknown_session(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manager = _manager(None, _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)

    with pytest.raises(AcpSessionError, match="not active"):
        manager.get_config_options("ghost")


@pytest.mark.asyncio
async def test_set_config_option_refreshes_settings_for_future_sessions(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _StartedHost.instances.clear()
    manager = _manager(str(tmp_path), _StaticListStore([]))
    manager._loaded_settings = LoadedSettings(
        settings=Settings(model_profile="old-model", ask_user_timeout_seconds=None),
        provenance={},
    )
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "old-model")
    _redirect_config_dir(monkeypatch, tmp_path)
    monkeypatch.setattr(session_manager_module, "ChrysSessionHost", _StartedHost)

    manager.set_config_option("model_profile", "new-model")
    await manager.new_session(cwd=str(tmp_path), mcp_servers=None)

    loaded = _StartedHost.instances[-1].kwargs["loaded_settings"]
    assert manager._loaded_settings.settings.model_profile == "new-model"
    assert loaded.settings.model_profile == "new-model"
    # The pointer write registered this process as the writer, so the refresh
    # attributes it to the runtime instead of blaming the environment.
    assert loaded.source_for("model.profile.active").layer is Source.PROCESS_RUNTIME
    assert loaded.settings.ask_user_timeout_seconds is None
    # The launch-time --ask-user-timeout is a CLI value; a re-read of the
    # environment cannot produce it, so it has to be carried across as one.
    assert loaded.source_for("tools.ask_user.timeout_seconds").layer is Source.CLI


@pytest.mark.asyncio
async def test_apply_config_option_rejects_inactive_session_without_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    manager = _manager(str(tmp_path), _StaticListStore([]))
    _redirect_config_dir(monkeypatch, tmp_path)

    # A missing/stale/inactive session must fail BEFORE the global document
    # write, so a failed RPC never leaves persisted config mutated.
    with pytest.raises(AcpSessionError, match="not active"):
        await manager.apply_config_option("ghost", "model_profile", "new-model")

    assert not (tmp_path / "settings.yaml").exists()


@pytest.mark.asyncio
async def test_switch_agent_rejects_unknown_profile(tmp_path) -> None:
    manager = _manager(str(tmp_path), _StaticListStore([]))
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_CloseHost(),
    )

    with pytest.raises(AcpSessionError, match="Agent profile not found: Ghost"):
        await manager.switch_agent("s1", "Ghost")


@pytest.mark.asyncio
async def test_switch_agent_same_profile_is_noop(tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import AgentProfileSwitch

    manager = _manager(str(tmp_path), _StaticListStore([]))
    host = _CloseHost()
    host.session_id = "s1"  # type: ignore[attr-defined]
    host.event_bus = EventBus()  # type: ignore[attr-defined]
    # The live engine snapshot carries the current runtime so the no-op response
    # does not blank out the client's model/tool/skill state.
    host.engine.snapshot = ProfileSwitched(
        from_profile="Code",
        to_profile="Code",
        from_display_name="Code",
        to_display_name="Code",
        message_count=4,
        model_profile_id="gpt-5",
        max_context_tokens=200000,
        session_id="s1",
        tool_names=["shell", "read_file"],
        skill_names=["search"],
        sub_agent_tool_names=["explore"],
        memory_files=["AGENTS.md"],
    )
    seen: list[AgentProfileSwitch] = []

    async def _unexpected_backend_switch(event: AgentProfileSwitch) -> None:
        seen.append(event)

    await host.event_bus.subscribe(AgentProfileSwitch, _unexpected_backend_switch)  # type: ignore[attr-defined]
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=host,
    )

    result = await asyncio.wait_for(manager.switch_agent("s1", "Code"), timeout=0.5)

    assert result.from_profile == "Code"
    assert result.to_profile == "Code"
    assert result.session_id == "s1"
    # Runtime fields must reflect the live engine, not dataclass blanks.
    assert result.model_profile_id == "gpt-5"
    assert result.max_context_tokens == 200000
    assert result.tool_names == ["shell", "read_file"]
    assert result.skill_names == ["search"]
    assert result.sub_agent_tool_names == ["explore"]
    assert result.memory_files == ["AGENTS.md"]
    assert seen == []
    assert result.message_count == 4


@pytest.mark.asyncio
async def test_switch_agent_surfaces_rebuild_failure(tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import AgentLoadFailed, AgentProfileSwitch

    manager = _manager(str(tmp_path), _StaticListStore([]))
    bus = EventBus()

    class _BusHost:
        session_id = "s1"
        event_bus = bus

    async def _fail_rebuild(event: AgentProfileSwitch) -> None:
        # Mirror soft_restart: a valid profile whose rebuild fails publishes
        # AgentLoadFailed (then raises inside the handler, which the bus swallows).
        await bus.publish(AgentLoadFailed(session_id="s1", agent_profile=event.profile_name, message="rebuild boom"))

    await bus.subscribe(AgentProfileSwitch, _fail_rebuild)
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Bootstrap",
        host=_BusHost(),  # type: ignore[arg-type]
    )

    with pytest.raises(AcpSessionError, match="rebuild boom"):
        await manager.switch_agent("s1", "Code")


class _EngineStub:
    def __init__(self, *, is_turn_active: bool) -> None:
        self.is_turn_active = is_turn_active


class _InjectHost:
    def __init__(self, bus, *, is_turn_active: bool) -> None:
        self.session_id = "s1"
        self.event_bus = bus
        self.engine = _EngineStub(is_turn_active=is_turn_active)


@pytest.mark.asyncio
async def test_inject_rejects_when_no_active_turn(tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus

    manager = _manager(str(tmp_path), _StaticListStore([]))
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_InjectHost(EventBus(), is_turn_active=False),  # type: ignore[arg-type]
    )

    with pytest.raises(AcpSessionError, match="No active turn"):
        await manager.inject("s1", "more context")


@pytest.mark.asyncio
async def test_inject_publishes_user_inject_when_active(tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import UserInject, UserMessage

    manager = _manager(str(tmp_path), _StaticListStore([]))
    bus = EventBus()
    injects: list[UserInject] = []
    messages: list[UserMessage] = []

    async def _capture_inject(event: UserInject) -> None:
        injects.append(event)

    async def _capture_message(event: UserMessage) -> None:
        messages.append(event)

    await bus.subscribe(UserInject, _capture_inject)
    await bus.subscribe(UserMessage, _capture_message)
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_InjectHost(bus, is_turn_active=True),  # type: ignore[arg-type]
    )

    await manager.inject("s1", "more context")

    # UserInject (never starts a turn), not UserMessage (would start a stray turn).
    assert [e.text for e in injects] == ["more context"]
    assert injects[0].session_id == "s1"
    assert messages == []


@pytest.mark.asyncio
async def test_rollback_ignores_non_fatal_restore_warnings(tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import RollbackResult, UserRollback, Warning

    manager = _manager(str(tmp_path), _StaticListStore([]))
    bus = EventBus()

    async def _on_rollback(event: UserRollback) -> None:
        # A successful rollback restores the session before publishing its result,
        # and restore can emit unrelated non-fatal warnings. These must not be
        # mistaken for a rollback refusal.
        await bus.publish(Warning(code="service_session_incompatible", message="local only", session_id="s1"))
        await bus.publish(Warning(code="sub_agents_reload_discarded", message="discarded", session_id="s1"))
        await bus.publish(RollbackResult(session_id="s1", target_turn=event.target_turn))

    await bus.subscribe(UserRollback, _on_rollback)
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_InjectHost(bus, is_turn_active=False),  # type: ignore[arg-type]
    )

    result = await manager.rollback("s1", target_turn=2, revert_changes=False, selected_paths=None)

    assert result.target_turn == 2


@pytest.mark.asyncio
async def test_rollback_refusal_warning_fails_request(tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import UserRollback, Warning

    manager = _manager(str(tmp_path), _StaticListStore([]))
    bus = EventBus()

    async def _on_rollback(_event: UserRollback) -> None:
        # A genuine refusal (rollback_* code) emits no RollbackResult and must
        # surface as an error rather than hanging until the timeout.
        await bus.publish(Warning(code="rollback_refused", message="cannot roll back", session_id="s1"))

    await bus.subscribe(UserRollback, _on_rollback)
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_InjectHost(bus, is_turn_active=False),  # type: ignore[arg-type]
    )

    with pytest.raises(AcpSessionError, match="cannot roll back"):
        await manager.rollback("s1", target_turn=1, revert_changes=False, selected_paths=None)


@pytest.mark.asyncio
async def test_set_approval_mode_publishes_session_scoped_event(tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ApprovalModeUpdated, SetApprovalMode

    manager = _manager(str(tmp_path), _StaticListStore([]))
    bus = EventBus()
    seen: list[SetApprovalMode] = []

    async def _echo(event: SetApprovalMode) -> None:
        seen.append(event)
        await bus.publish(ApprovalModeUpdated(mode=event.mode, session_id="s1"))

    await bus.subscribe(SetApprovalMode, _echo)
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_InjectHost(bus, is_turn_active=True),  # type: ignore[arg-type]
    )

    result = await manager.set_approval_mode("s1", "auto")

    assert result.mode == "auto"
    assert len(seen) == 1
    assert seen[0].mode == "auto"
    assert seen[0].persist is False
    assert seen[0].session_id == "s1"


@pytest.mark.asyncio
async def test_concurrent_approval_mode_updates_resolve_independently(tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ApprovalModeUpdated, SetApprovalMode

    manager = _manager(str(tmp_path), _StaticListStore([]))
    bus = EventBus()
    seen: list[str] = []

    async def _echo(event: SetApprovalMode) -> None:
        await asyncio.sleep(0)
        seen.append(event.mode)
        await bus.publish(ApprovalModeUpdated(mode=event.mode, session_id="s1"))

    await bus.subscribe(SetApprovalMode, _echo)
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_InjectHost(bus, is_turn_active=True),  # type: ignore[arg-type]
    )

    results = await asyncio.gather(
        manager.set_approval_mode("s1", "auto"),
        manager.set_approval_mode("s1", "bypass"),
    )

    assert [result.mode for result in results] == ["auto", "bypass"]
    assert seen == ["auto", "bypass"]


@pytest.mark.asyncio
async def test_set_model_profile_surfaces_rebuild_failure(tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import AgentLoadFailed, SetModelProfile

    manager = _manager(str(tmp_path), _StaticListStore([]))
    bus = EventBus()

    async def _fail(event: SetModelProfile) -> None:
        await bus.publish(AgentLoadFailed(session_id="s1", message="model boom"))

    await bus.subscribe(SetModelProfile, _fail)
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_InjectHost(bus, is_turn_active=True),  # type: ignore[arg-type]
    )

    with pytest.raises(AcpSessionError, match="model boom"):
        await manager.set_model_profile("s1", "model")


@pytest.mark.asyncio
async def test_set_workspace_surfaces_rebuild_failure(tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import AgentLoadFailed, WorkspaceChange

    manager = _manager(str(tmp_path), _StaticListStore([]))
    bus = EventBus()

    async def _fail(event: WorkspaceChange) -> None:
        await bus.publish(AgentLoadFailed(session_id="s1", message="workspace boom"))

    await bus.subscribe(WorkspaceChange, _fail)
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_InjectHost(bus, is_turn_active=True),  # type: ignore[arg-type]
    )

    with pytest.raises(AcpSessionError, match="workspace boom"):
        await manager.set_workspace("s1", str(tmp_path))


@pytest.mark.asyncio
async def test_reload_settings_surfaces_rebuild_failure(tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import AgentLoadFailed, SettingsReload

    manager = _manager(str(tmp_path), _StaticListStore([]))
    bus = EventBus()

    async def _fail(_event: SettingsReload) -> None:
        # A reload whose soft-restart fails publishes AgentLoadFailed and then
        # raises inside the bus handler (swallowed) — the await must surface it
        # rather than report a silent success.
        await bus.publish(AgentLoadFailed(session_id="s1", message="reload boom"))

    await bus.subscribe(SettingsReload, _fail)
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_InjectHost(bus, is_turn_active=True),  # type: ignore[arg-type]
    )

    with pytest.raises(AcpSessionError, match="reload boom"):
        await manager.reload_settings("s1")


@pytest.mark.asyncio
async def test_reload_settings_surfaces_handler_error(tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import Error, SettingsReload

    manager = _manager(str(tmp_path), _StaticListStore([]))
    bus = EventBus()

    async def _fail(_event: SettingsReload) -> None:
        # Mirror a bad-env reload: the handler reports an Error then raises before
        # publishing any completion event. The bus swallows the raise, so the await
        # must resolve on the Error rather than hanging until the timeout.
        await bus.publish(Error(code="settings_reload_failed", message="bad env", session_id="s1"))
        raise ValueError("bad env")

    await bus.subscribe(SettingsReload, _fail)
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_InjectHost(bus, is_turn_active=True),  # type: ignore[arg-type]
    )

    with pytest.raises(AcpSessionError, match="bad env"):
        await manager.reload_settings("s1")


@pytest.mark.asyncio
async def test_reload_settings_awaits_completion_event(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import SettingsReload, SettingsReloaded

    manager = _manager(str(tmp_path), _StaticListStore([]))
    manager._loaded_settings = LoadedSettings(
        settings=Settings(model_profile="old-model", ask_user_timeout_seconds=None),
        provenance={},
    )
    monkeypatch.setenv("CHRYS_MODEL_PROFILE", "new-model")
    bus = EventBus()

    async def _succeed(_event: SettingsReload) -> None:
        await bus.publish(SettingsReloaded(session_id="s1"))

    await bus.subscribe(SettingsReload, _succeed)
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_InjectHost(bus, is_turn_active=True),  # type: ignore[arg-type]
    )

    result = await manager.reload_settings("s1")

    assert isinstance(result, SettingsReloaded)
    assert result.session_id == "s1"
    assert manager._loaded_settings.settings.model_profile == "new-model"
    assert manager._loaded_settings.settings.ask_user_timeout_seconds is None


@pytest.mark.asyncio
async def test_concurrent_model_switches_resolve_independently(tmp_path) -> None:
    # Two overlapping mutations of the same result type must not cross-resolve:
    # the per-session lock serializes them so each awaits its own completion.
    import asyncio

    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ModelProfileSwitched, SetModelProfile

    manager = _manager(str(tmp_path), _StaticListStore([]))
    bus = EventBus()
    seen: list[str] = []

    async def _echo(event: SetModelProfile) -> None:
        # Yield so a second request can interleave if the lock were absent, then
        # echo a result tagged with the profile that drove this rebuild.
        await asyncio.sleep(0)
        seen.append(event.profile_id)
        await bus.publish(ModelProfileSwitched(session_id="s1", model_profile_id=event.profile_id))

    await bus.subscribe(SetModelProfile, _echo)
    manager._sessions["s1"] = ManagedSession(  # type: ignore[assignment]
        session_id="s1",
        cwd=str(tmp_path),
        profile_name="Code",
        host=_InjectHost(bus, is_turn_active=True),  # type: ignore[arg-type]
    )
    manager._model_registry.register(ModelProfile(id="a", name="A"))
    manager._model_registry.register(ModelProfile(id="b", name="B"))

    results = await asyncio.gather(
        manager.set_model_profile("s1", "a"),
        manager.set_model_profile("s1", "b"),
    )

    # Each call resolved from its own echo, not whichever fired first.
    assert {r.model_profile_id for r in results} == {"a", "b"}
    assert sorted(seen) == ["a", "b"]


@pytest.mark.asyncio
async def test_load_session_rejects_additional_directories_for_active_session(tmp_path) -> None:
    project = tmp_path / "project"
    extra = tmp_path / "extra"
    project.mkdir()
    extra.mkdir()
    store = JsonFileStateStore(tmp_path / "sessions")
    await store.save_session(
        "project-session",
        {"messages": [Message("user", ["hello project"])]},
        agent_profile="Code",
        primary_cwd=str(project),
    )
    manager = _manager(str(project), store)
    manager._sessions["project-session"] = ManagedSession(  # type: ignore[assignment]
        session_id="project-session",
        cwd=str(project),
        profile_name="Code",
        host=_CloseHost(),
    )

    with pytest.raises(AcpSessionError, match="additional directories"):
        await manager.load_session(
            cwd=str(project),
            session_id="project-session",
            additional_directories=[str(extra)],
            mcp_servers=None,
        )


def test_session_info_prefers_title_overlays() -> None:
    """ACP session listings must surface custom/generated titles, not just the first message."""
    from datetime import UTC, datetime

    from chrys.service.state.store import SessionMeta

    now = datetime.now(UTC)
    meta = SessionMeta(
        session_id="sess1",
        agent_profile="code",
        agent_display_name="Code",
        created_at=now,
        updated_at=now,
        message_count=1,
        title="fix the login bug",
        generated_title="Login bug fix",
    )
    info = session_manager_module._session_info(meta)
    assert info.title == "Login bug fix"

    meta.custom_title = "My session"
    assert session_manager_module._session_info(meta).title == "My session"
