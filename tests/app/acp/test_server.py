# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for ACP server request handling glue."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest
from acp import PROTOCOL_VERSION, RequestError
from acp import schema as acp_schema
from acp.helpers import image_block, text_block

from chrys.app.acp import server as server_module
from chrys.app.acp.bridge import AcpEventBridge
from chrys.app.acp.server import ChrysAcpServer
from chrys.app.acp.session_manager import AcpSessionError
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    AgentLoadProgress,
    AgentMessage,
    AgentRuntimeDetails,
    ApprovalCancelled,
    ApprovalModeUpdated,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalReviewed,
    AskUserResponse,
    AskUserTimedOut,
    CompactionFinished,
    CompactionStarted,
    ContextPressure,
    Error,
    Event,
    ModelProfileSwitched,
    ProfileSwitched,
    QuestionToUser,
    RollbackResult,
    RuntimeHookDetails,
    RuntimeHookSourceDetails,
    RuntimeModelDetails,
    RuntimeSkillDetails,
    SessionTitleUpdated,
    SubAgentCompactionCommitted,
    SubAgentCompactionFinished,
    SubAgentCompactionStarted,
    SubAgentInvocationStart,
    SubAgentPaused,
    SubAgentProgress,
    TodoListUpdated,
    UsageUpdate,
    UserInjectResult,
    Warning,
    WorkspaceUpdated,
)
from chrys.foundation.models.todos import TodoItem
from chrys.orchestration.session_host import Cancelled, EndTurn, Errored
from chrys.service.approval.policy import ApprovalMode
from chrys.service.mutations.types import (
    FileHashDiff,
    FileMutation,
    MutationOp,
    MutationSource,
    RestoreOutcome,
    RestoreResult,
    TurnMutations,
)
from chrys.service.routing.classifier import RouteDecision
from chrys.service.todos.tracker import TodoTracker


class _FakeClient:
    def __init__(
        self,
        *,
        option_id: str = "allow",
        permission_outcome: acp_schema.AllowedOutcome | acp_schema.DeniedOutcome | None = None,
        permission_exc: Exception | None = None,
        permission_responder: Any = None,
        input_responder: Any = None,
    ) -> None:
        self.option_id = option_id
        self.permission_outcome = permission_outcome
        self.permission_exc = permission_exc
        self.permission_responder = permission_responder
        self.input_responder = input_responder
        self.permission_requests: list[acp_schema.RequestPermissionRequest] = []
        self.input_requests: list[tuple[str, dict[str, Any]]] = []
        self.updates: list[acp_schema.SessionNotification] = []
        self.ext_notifications: list[tuple[str, dict[str, Any]]] = []

    async def request_permission(
        self,
        options: list[acp_schema.PermissionOption],
        session_id: str,
        tool_call: acp_schema.ToolCallUpdate,
        **kwargs: Any,
    ) -> acp_schema.RequestPermissionResponse:
        self.permission_requests.append(
            acp_schema.RequestPermissionRequest(options=options, sessionId=session_id, toolCall=tool_call, **kwargs)
        )
        if self.permission_exc is not None:
            raise self.permission_exc
        if self.permission_responder is not None:
            return await self.permission_responder(session_id, tool_call)
        if self.permission_outcome is not None:
            return acp_schema.RequestPermissionResponse(outcome=self.permission_outcome)
        return acp_schema.RequestPermissionResponse(
            outcome=acp_schema.AllowedOutcome(outcome="selected", optionId=self.option_id)
        )

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append(acp_schema.SessionNotification(sessionId=session_id, update=update, **kwargs))

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        self.ext_notifications.append((method, params))

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.input_requests.append((method, params))
        if self.input_responder is not None:
            return await self.input_responder(method, params)
        return {"text": "ok"}


@dataclass
class _FakeBlobStore:
    blobs: dict[str, bytes]

    def read_blob(self, blob_hash: str) -> bytes:
        return self.blobs[blob_hash]


@dataclass
class _FakeMutationTracker:
    store: _FakeBlobStore
    turns: list[TurnMutations] = field(default_factory=list)
    session_summary: dict[str, FileHashDiff] = field(default_factory=dict)
    turn_summaries: dict[int, dict[str, FileHashDiff]] = field(default_factory=dict)

    def get_all_turns(self) -> list[TurnMutations]:
        return self.turns

    def get_session_file_summary(self) -> dict[str, FileHashDiff]:
        return self.session_summary

    def get_turn_file_summary(self, turn_id: int) -> dict[str, FileHashDiff]:
        return self.turn_summaries.get(turn_id, {})


@dataclass
class _FakeEngine:
    runtime_details: AgentRuntimeDetails = field(default_factory=AgentRuntimeDetails)
    mutation_tracker: _FakeMutationTracker | None = None
    current_turn_number: int = 0
    usage: UsageUpdate = field(default_factory=UsageUpdate)
    rollback_turns: list[int] = field(default_factory=list)
    approval_mode: ApprovalMode = ApprovalMode.MANUAL
    todo_tracker: TodoTracker | None = None
    # The runtime payload reports routing; nothing here has been classified.
    last_route: RouteDecision | None = None
    settings: Settings = field(default_factory=Settings)

    def make_usage_event(self, *, session_id: str | None = None) -> UsageUpdate:
        return UsageUpdate(
            session_id=session_id or self.usage.session_id,
            agent_profile=self.usage.agent_profile,
            usage_source_id=self.usage.usage_source_id,
            input_tokens=self.usage.input_tokens,
            output_tokens=self.usage.output_tokens,
            total_tokens=self.usage.total_tokens,
            pct=self.usage.pct,
            max_context_tokens=self.usage.max_context_tokens,
            total_session_tokens=self.usage.total_session_tokens,
            total_session_input_tokens=self.usage.total_session_input_tokens,
            total_session_output_tokens=self.usage.total_session_output_tokens,
            cache_hit_tokens=self.usage.cache_hit_tokens,
            total_session_cache_hit_tokens=self.usage.total_session_cache_hit_tokens,
            local_tokens=self.usage.local_tokens,
            calibration_ratio=self.usage.calibration_ratio,
            system_overhead_tokens=self.usage.system_overhead_tokens,
        )

    def available_rollback_turns(self) -> list[int]:
        return list(self.rollback_turns)


@dataclass
class _FakeHost:
    event_bus: EventBus
    events: list[Event] = field(default_factory=list)
    outcome: Any = None
    vision_enabled: bool = False
    last_turn_outcome: Any = None
    engine: _FakeEngine = field(default_factory=_FakeEngine)

    async def iter_turn_events(self, _message: Any):
        for event in self.events:
            yield event
        self.last_turn_outcome = self.outcome


@dataclass
class _FakeSession:
    host: _FakeHost
    profile_name: str = "Code"
    session_id: str = "s1"
    prompt_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closing: bool = False


class _FakeStateStore:
    async def load_session_raw(
        self,
        session_id: str,
        *,
        prefer_recovery: bool = False,
    ) -> list[dict[str, Any]]:
        _ = prefer_recovery
        return [{"role": "user", "contents": [{"type": "text", "text": f"hello {session_id}"}]}]


@dataclass
class _LoadedSession:
    session_id: str
    profile_name: str = "Code"
    host: _FakeHost = field(default_factory=lambda: _FakeHost(event_bus=EventBus()))
    prompt_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class _LoadResult:
    session: _LoadedSession
    reused_existing: bool = False
    recovered_from_sidecar: bool = False


class _FakeManager:
    def __init__(self, host: _FakeHost) -> None:
        self._session = _FakeSession(host=host)
        self.injected: list[tuple[str, str]] = []
        self.rollbacks: list[dict[str, Any]] = []
        self.sub_agent_retries: list[tuple[str, str]] = []
        self.sub_agent_aborts: list[tuple[str, str]] = []
        self.approval_modes: list[tuple[str, str]] = []
        self.agent_switches: list[tuple[str, str]] = []
        self.settings_reloads: list[str] = []
        self.reload_warnings: list[Warning] = []
        self.session_warnings: list[Warning] = []
        self.workspace_updates: list[tuple[str, str]] = []
        self.model_switches: list[tuple[str, str]] = []
        self.mcp_tests: list[dict[str, object]] = []
        self.config_updates: list[tuple[str, object]] = []
        self.config_option_queries: list[str | None] = []
        self.deleted_sessions: list[tuple[str | None, str]] = []
        self.last_delete_cwd: str | None = None
        self.history_reads: list[tuple[str | None, str]] = []
        self.cancelled_sessions: list[str] = []
        self.closed_sessions: list[str] = []
        self.lifecycle_calls: list[str] = []
        self.state_store = _FakeStateStore()

    def get(self, session_id: str) -> _FakeSession:
        assert session_id == "s1"
        if self._session.closing:
            raise AcpSessionError(f"ACP session is not active: {session_id}")
        return self._session

    async def new_session(
        self,
        *,
        cwd: str | None,
        additional_directories: list[str] | None,
        mcp_servers: Any,
        warnings: list[Warning] | None = None,
    ) -> _FakeSession:
        _ = cwd, additional_directories, mcp_servers
        if warnings is not None:
            warnings.extend(self.session_warnings)
        return self._session

    def tool_kind_resolver(self, session_id: str) -> Any:
        _ = session_id
        return None

    async def inject(self, session_id: str, text: str) -> None:
        self.injected.append((session_id, text))

    async def rollback(
        self,
        session_id: str,
        *,
        target_turn: int,
        revert_changes: bool,
        selected_paths: list[str] | None,
    ) -> Any:
        self.rollbacks.append(
            {
                "session_id": session_id,
                "target_turn": target_turn,
                "revert_changes": revert_changes,
                "selected_paths": selected_paths,
            }
        )
        return RollbackResult(
            session_id=session_id,
            target_turn=target_turn,
            rolled_back_user_text="discarded prompt",
            files_reverted=1,
            restore_results=[RestoreResult(path="/workspace/a.py", outcome=RestoreOutcome.APPLIED)],
        )

    async def retry_sub_agent(self, session_id: str, invocation_id: str) -> None:
        self.sub_agent_retries.append((session_id, invocation_id))

    async def abort_sub_agent(self, session_id: str, invocation_id: str) -> None:
        self.sub_agent_aborts.append((session_id, invocation_id))

    async def set_approval_mode(self, session_id: str, mode: str) -> ApprovalModeUpdated:
        self.approval_modes.append((session_id, mode))
        return ApprovalModeUpdated(mode=mode)

    async def switch_agent(self, session_id: str, profile_name: str) -> ProfileSwitched:
        self.agent_switches.append((session_id, profile_name))
        return ProfileSwitched(from_profile="Code", to_profile=profile_name, to_display_name=profile_name)

    async def reload_settings(self, session_id: str, *, warnings: list[Warning] | None = None) -> None:
        self.settings_reloads.append(session_id)
        if warnings is not None:
            warnings.extend(self.reload_warnings)

    async def session_history(self, *, cwd: str | None, session_id: str) -> tuple[str, list[dict[str, Any]]]:
        self.history_reads.append((cwd, session_id))
        return session_id, await self.state_store.load_session_raw(session_id)

    async def set_workspace(self, session_id: str, primary_cwd: str) -> WorkspaceUpdated:
        self.workspace_updates.append((session_id, primary_cwd))
        return WorkspaceUpdated(primary_cwd=primary_cwd, working_dirs=[primary_cwd], reference_files=[])

    def list_agent_profiles(self) -> list[dict[str, object]]:
        return [{"name": "Code", "displayName": "Code"}]

    def read_agent_profile(self, name: str) -> dict[str, object]:
        return {"name": name, "description": "agent"}

    def write_agent_profile(self, data: dict[str, object]) -> dict[str, object]:
        return {"profile": {"name": data["name"]}, "path": "/agents/test.yaml"}

    def delete_agent_profile(self, name: str) -> dict[str, object]:
        return {"name": name, "deleted": True}

    def reset_agent_profile(self, name: str) -> dict[str, object]:
        return {"profile": {"name": name, "builtin": True}, "changed": True}

    def list_model_profiles(self) -> list[dict[str, object]]:
        return [{"id": "m1", "name": "Model", "modelId": "gpt"}]

    def read_model_profile(self, profile_id: str) -> dict[str, object]:
        return {"id": profile_id, "api_key": ""}

    def write_model_profile(self, data: dict[str, object]) -> dict[str, object]:
        return {"profile": {"id": data.get("id", "m2"), "name": data["name"]}, "path": "/models/m2.yaml"}

    def delete_model_profile(self, profile_id: str) -> dict[str, object]:
        return {"id": profile_id, "deleted": True}

    async def set_model_profile(self, session_id: str, profile_id: str) -> ModelProfileSwitched:
        self.model_switches.append((session_id, profile_id))
        return ModelProfileSwitched(model_profile_id=profile_id, max_context_tokens=128000)

    async def test_mcp_server(self, data: dict[str, object]) -> dict[str, object]:
        self.mcp_tests.append(data)
        return {"ok": True, "name": data.get("name", "server"), "message": "Connected."}

    def get_config_options(self, session_id: str | None = None) -> dict[str, object]:
        self.config_option_queries.append(session_id)
        return {"options": [{"key": "theme", "envKey": "CHRYS_THEME", "settingKey": "ui.theme", "value": "chrys"}]}

    def set_config_option(self, key: str, value: object) -> dict[str, object]:
        self.config_updates.append((key, value))
        return {"key": key, "envKey": "CHRYS_THEME", "settingKey": "ui.theme", "value": value}

    async def apply_config_option(
        self,
        session_id: str,
        key: str,
        value: object,
        *,
        warnings: list[Warning] | None = None,
    ) -> dict[str, object]:
        result = self.set_config_option(key, value)
        await self.reload_settings(session_id, warnings=warnings)
        return result

    async def begin_delete_session(self, *, cwd: str | None, session_id: str) -> str:
        self.lifecycle_calls.append(f"begin_delete:{session_id}")
        # The real manager resolves saved metadata from disk before it can
        # validate and mark the session closing; model that async gap so the
        # regressions exercise the real interleaving.
        await asyncio.sleep(0)
        if session_id not in ("s1", "short1"):
            raise AcpSessionError(f"Session not found: {session_id}")
        if cwd is not None and cwd != "/tmp/project":
            raise AcpSessionError(f"Saved session is not in workspace: {session_id}")
        self.last_delete_cwd = cwd
        self._session.closing = True
        return "s1"

    async def finish_delete_session(self, canonical_id: str) -> None:
        self.deleted_sessions.append((self.last_delete_cwd, canonical_id))
        self.lifecycle_calls.append(f"delete:{canonical_id}")
        # Generous async gap before the teardown reaches the prompt lock, like
        # the real manager's lock and state-store awaits: if the session is not
        # already closing when the server releases waits, a queued prompt gets
        # admitted here and this acquisition wedges behind its new turn.
        for _ in range(50):
            await asyncio.sleep(0)
        self._session.closing = True
        async with self._session.prompt_lock:
            pass

    async def cancel(self, session_id: str) -> None:
        self.cancelled_sessions.append(session_id)
        self.lifecycle_calls.append(f"cancel:{session_id}")

    async def begin_close(self, session_id: str) -> None:
        self.lifecycle_calls.append(f"begin_close:{session_id}")
        self._session.closing = True

    async def close(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)
        self.lifecycle_calls.append(f"close:{session_id}")
        async with self._session.prompt_lock:
            pass


class _FakeLoadManager:
    def __init__(self, *, reused_existing: bool = False) -> None:
        self.state_store = object()
        self.closed: list[str] = []
        self.reused_existing = reused_existing
        self.session = _LoadedSession(session_id="s1")
        self.session_warnings: list[Warning] = []

    async def load_session(
        self,
        *,
        cwd: str,
        session_id: str,
        additional_directories: list[str] | None = None,
        mcp_servers: Any,
        warnings: list[Warning] | None = None,
    ) -> _LoadResult:
        _ = cwd, additional_directories, mcp_servers
        assert session_id == self.session.session_id
        if warnings is not None:
            warnings.extend(self.session_warnings)
        return _LoadResult(session=self.session, reused_existing=self.reused_existing)

    def get(self, session_id: str) -> _LoadedSession:
        assert session_id == self.session.session_id
        return self.session

    def tool_kind_resolver(self, session_id: str) -> Any:
        _ = session_id
        return None

    def list_model_profiles(self) -> list[dict[str, object]]:
        return [{"id": "m1", "name": "Model", "modelId": "gpt"}]

    async def close(self, session_id: str) -> None:
        self.closed.append(session_id)


@pytest.mark.anyio
async def test_manual_approval_request_bridges_to_acp_permission() -> None:
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)
    responses: list[ApprovalResponse] = []

    async def _collect(event: ApprovalResponse) -> None:
        responses.append(event)

    await host.event_bus.subscribe(ApprovalResponse, _collect)

    await server._handle_event(
        "s1",
        ApprovalRequest(
            request_id="r1",
            tool_name="bash",
            tool_kind="shell",
            args={"command": "rm file"},
            intent_summary="Run shell command",
            session_id="s1",
        ),
        AcpEventBridge(),
        {},
    )

    assert len(client.permission_requests) == 1
    tool_call = client.permission_requests[0].tool_call
    assert tool_call.tool_call_id == "r1"
    assert tool_call.title == "Run shell command"
    assert tool_call.kind == "execute"
    assert tool_call.field_meta == {"chrys": {"tool_name": "bash", "tool_kind": "shell"}}
    assert len(responses) == 1
    assert responses[0].request_id == "r1"
    assert responses[0].approved is True
    assert responses[0].reason == ""
    assert responses[0].session_id == "s1"


@pytest.mark.anyio
async def test_permission_title_falls_back_to_shell_command() -> None:
    """Without an intent summary the title is the command itself, not
    "Execute zsh" — the shell tool's name is the bare shell binary."""
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    await server._handle_event(
        "s1",
        ApprovalRequest(
            request_id="r1",
            tool_name="zsh",
            tool_kind="shell",
            args={"command": "cloc --vcs=git ."},
            session_id="s1",
        ),
        AcpEventBridge(),
        {},
    )

    assert len(client.permission_requests) == 1
    assert client.permission_requests[0].tool_call.title == "cloc --vcs=git ."


@pytest.mark.anyio
async def test_session_title_watcher_forwards_post_turn_updates() -> None:
    """Generated titles land after the prompt event stream has closed, so
    they must reach the client via the long-lived per-session subscription."""
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    await server._watch_session_titles(_FakeSession(host=host))  # type: ignore[arg-type]

    await host.event_bus.publish(
        SessionTitleUpdated(session_id="s1", title="Login bug fix", custom=False, display_title="Login bug fix")
    )
    # Cross-session events on the same bus are not this session's updates.
    await host.event_bus.publish(SessionTitleUpdated(session_id="s2", title="Other", display_title="Other"))

    info_updates = [n for n in client.updates if isinstance(n.update, acp_schema.SessionInfoUpdate)]
    assert len(info_updates) == 1
    assert info_updates[0].session_id == "s1"
    assert info_updates[0].update.title == "Login bug fix"
    assert info_updates[0].update.updated_at is not None


@pytest.mark.anyio
async def test_auto_approval_judging_request_waits_for_flagged_review() -> None:
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient(option_id="reject")
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)
    pending: dict[str, ApprovalRequest] = {}
    responses: list[ApprovalResponse] = []

    async def _collect(event: ApprovalResponse) -> None:
        responses.append(event)

    await host.event_bus.subscribe(ApprovalResponse, _collect)
    request = ApprovalRequest(
        request_id="r1",
        tool_name="write_file",
        args={"path": "a.py"},
        intent_summary="Write file",
        session_id="s1",
        judging=True,
    )

    await server._handle_event("s1", request, AcpEventBridge(), pending)
    assert client.permission_requests == []
    assert pending == {"r1": request}

    await server._handle_event(
        "s1",
        ApprovalReviewed(request_id="r1", approved=False, reason="Risky", session_id="s1"),
        AcpEventBridge(),
        pending,
    )

    assert len(client.permission_requests) == 1
    assert len(responses) == 1
    assert responses[0].request_id == "r1"
    assert responses[0].approved is False
    assert responses[0].reason == "Rejected by ACP client."
    assert responses[0].session_id == "s1"


@pytest.mark.anyio
async def test_prompt_returns_cancelled_stop_reason() -> None:
    host = _FakeHost(
        event_bus=EventBus(),
        events=[AgentMessage(text="working", is_final=False, session_id="s1")],
        outcome=Cancelled(),
    )
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    response = await server.prompt([text_block("stop")], session_id="s1", message_id="m1")

    assert response.stop_reason == "cancelled"
    assert response.user_message_id == "m1"
    assert len(client.updates) == 1
    assert client.updates[0].update.session_update == "agent_message_chunk"


@pytest.mark.anyio
async def test_prompt_snapshots_cumulative_usage_in_finally_while_holding_prompt_lock() -> None:
    usage = UsageUpdate(
        total_session_input_tokens=70,
        total_session_output_tokens=20,
        total_session_tokens=100,
        total_session_cache_hit_tokens=11,
    )
    host = _FakeHost(event_bus=EventBus(), outcome=EndTurn())
    manager = _FakeManager(host)
    snapshot_lock_states: list[bool] = []

    class _Engine:
        def make_usage_event(self, *, session_id: str | None = None) -> UsageUpdate:
            assert session_id == "s1"
            snapshot_lock_states.append(manager._session.prompt_lock.locked())
            return usage

    host.engine = _Engine()  # type: ignore[assignment]
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient())

    response = await server.prompt([text_block("run")], session_id="s1")

    assert snapshot_lock_states == [True]
    assert response.usage is not None
    assert response.usage.input_tokens == 70
    assert response.usage.output_tokens == 20
    assert response.usage.total_tokens == 100
    assert response.usage.cached_read_tokens == 11


@pytest.mark.anyio
async def test_prompt_errored_outcome_carries_usage_in_request_error_data() -> None:
    host = _FakeHost(
        event_bus=EventBus(),
        outcome=Errored(error=Error(code="model_failed", message="provider stopped")),
        engine=_FakeEngine(
            usage=UsageUpdate(
                total_session_input_tokens=30,
                total_session_output_tokens=5,
                total_session_tokens=35,
            )
        ),
    )
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient())

    with pytest.raises(RequestError) as exc_info:
        await server.prompt([text_block("run")], session_id="s1")

    assert exc_info.value.data["code"] == "model_failed"
    assert exc_info.value.data["usage"] == {
        "inputTokens": 30,
        "outputTokens": 5,
        "totalTokens": 35,
    }


@pytest.mark.anyio
async def test_prompt_execution_exception_carries_usage_but_pre_execution_validation_does_not() -> None:
    class _ExplodingHost(_FakeHost):
        async def iter_turn_events(self, _message: Any):
            if False:
                yield Event()
            raise RuntimeError("event bridge failed")

    host = _ExplodingHost(
        event_bus=EventBus(),
        engine=_FakeEngine(
            usage=UsageUpdate(
                total_session_input_tokens=9,
                total_session_output_tokens=4,
                total_session_tokens=13,
            )
        ),
    )
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient())

    with pytest.raises(RequestError) as executed:
        await server.prompt([text_block("run")], session_id="s1")
    assert executed.value.data["usage"] == {
        "inputTokens": 9,
        "outputTokens": 4,
        "totalTokens": 13,
    }

    with pytest.raises(RequestError) as validation:
        await server.prompt([image_block("not-image-data", "image/png")], session_id="s1")
    assert "usage" not in validation.value.data


@pytest.mark.anyio
async def test_cancel_resolves_pending_request_input() -> None:
    started = asyncio.Event()

    async def _block_input(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
        started.set()
        await asyncio.Future()
        return {"text": "too late"}

    host = _FakeHost(
        event_bus=EventBus(),
        events=[QuestionToUser(request_id="q1", question="Continue?", session_id="s1")],
        outcome=Cancelled(),
    )
    manager = _FakeManager(host)
    client = _FakeClient(input_responder=_block_input)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)
    responses: list[AskUserResponse] = []

    async def _collect(event: AskUserResponse) -> None:
        responses.append(event)

    await host.event_bus.subscribe(AskUserResponse, _collect)
    prompt_task = asyncio.create_task(server.prompt([text_block("ask")], session_id="s1", message_id="m1"))
    await asyncio.wait_for(started.wait(), timeout=1)

    await server.cancel("s1")
    response = await asyncio.wait_for(prompt_task, timeout=1)

    assert response.stop_reason == "cancelled"
    assert manager.cancelled_sessions == ["s1"]
    assert client.input_requests[0][0] == "chrys/request_input"
    assert len(responses) == 1
    assert responses[0].request_id == "q1"
    assert responses[0].text.startswith("Error:")


@pytest.mark.anyio
async def test_cancel_interrupts_before_rejecting_pending_permission_and_allows_following_turn() -> None:
    started = asyncio.Event()

    async def _block_permission(_session_id: str, _tool_call: Any) -> acp_schema.RequestPermissionResponse:
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable after cancellation")

    host = _FakeHost(
        event_bus=EventBus(),
        events=[ApprovalRequest(request_id="r1", tool_name="bash", args={}, session_id="s1")],
        outcome=Cancelled(),
    )
    manager = _FakeManager(host)
    client = _FakeClient(permission_responder=_block_permission)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)
    responses: list[ApprovalResponse] = []

    async def _collect(event: ApprovalResponse) -> None:
        assert manager.cancelled_sessions == ["s1"], "engine interruption must precede bridge rejection"
        responses.append(event)

    await host.event_bus.subscribe(ApprovalResponse, _collect)
    prompt_task = asyncio.create_task(server.prompt([text_block("run")], session_id="s1", message_id="m1"))
    await asyncio.wait_for(started.wait(), timeout=1)

    await server.cancel("s1")
    response = await asyncio.wait_for(prompt_task, timeout=1)

    assert response.stop_reason == "cancelled"
    assert len(responses) == 1
    assert responses[0].approved is False
    assert responses[0].reason == "ACP permission request was cancelled."
    assert server._pending_permission_cancels == {}
    assert server._pending_permission_tasks == {}

    host.events = []
    host.outcome = EndTurn()
    following = await asyncio.wait_for(
        server.prompt([text_block("next")], session_id="s1", message_id="m2"),
        timeout=1,
    )
    assert following.stop_reason == "end_turn"


@pytest.mark.anyio
async def test_late_permission_allow_after_cancel_is_ignored() -> None:
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release_late_reply = asyncio.Event()
    late_reply_returned = asyncio.Event()

    async def _late_allow(_session_id: str, _tool_call: Any) -> acp_schema.RequestPermissionResponse:
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_late_reply.wait()
        late_reply_returned.set()
        return acp_schema.RequestPermissionResponse(
            outcome=acp_schema.AllowedOutcome(outcome="selected", optionId="allow")
        )

    host = _FakeHost(
        event_bus=EventBus(),
        events=[ApprovalRequest(request_id="r1", tool_name="bash", args={}, session_id="s1")],
        outcome=Cancelled(),
    )
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient(permission_responder=_late_allow))
    responses: list[ApprovalResponse] = []

    async def _collect(event: ApprovalResponse) -> None:
        responses.append(event)

    await host.event_bus.subscribe(ApprovalResponse, _collect)
    prompt_task = asyncio.create_task(server.prompt([text_block("run")], session_id="s1"))
    await asyncio.wait_for(started.wait(), timeout=1)

    await server.cancel("s1")
    await asyncio.wait_for(prompt_task, timeout=1)
    await asyncio.wait_for(cancellation_seen.wait(), timeout=1)
    release_late_reply.set()
    await asyncio.wait_for(late_reply_returned.wait(), timeout=1)
    await asyncio.sleep(0)

    assert len(responses) == 1
    assert responses[0].approved is False


@pytest.mark.anyio
async def test_wedged_permission_client_times_out_server_side() -> None:
    async def _never_reply(_session_id: str, _tool_call: Any) -> acp_schema.RequestPermissionResponse:
        await asyncio.Future()
        raise AssertionError("unreachable after timeout")

    host = _FakeHost(
        event_bus=EventBus(),
        events=[ApprovalRequest(request_id="r1", tool_name="bash", args={}, session_id="s1")],
        outcome=EndTurn(),
    )
    server = ChrysAcpServer(  # type: ignore[arg-type]
        _FakeManager(host),
        initial_vision=False,
        permission_timeout_seconds=0.01,
    )
    server.on_connect(_FakeClient(permission_responder=_never_reply))
    responses: list[ApprovalResponse] = []

    async def _collect(event: ApprovalResponse) -> None:
        responses.append(event)

    await host.event_bus.subscribe(ApprovalResponse, _collect)

    response = await asyncio.wait_for(server.prompt([text_block("run")], session_id="s1"), timeout=1)

    assert response.stop_reason == "end_turn"
    assert len(responses) == 1
    assert responses[0].approved is False
    assert responses[0].reason == "ACP permission request timed out."
    assert server._pending_permission_cancels == {}
    assert server._pending_permission_tasks == {}


@pytest.mark.anyio
async def test_wedged_permission_in_one_session_does_not_block_another_session_prompt() -> None:
    permission_started = asyncio.Event()

    async def _block_permission(_session_id: str, _tool_call: Any) -> acp_schema.RequestPermissionResponse:
        permission_started.set()
        await asyncio.Future()
        raise AssertionError("unreachable after cancellation")

    first_host = _FakeHost(
        event_bus=EventBus(),
        events=[ApprovalRequest(request_id="r1", tool_name="bash", args={}, session_id="s1")],
        outcome=Cancelled(),
    )
    second_host = _FakeHost(event_bus=EventBus(), events=[], outcome=EndTurn())
    sessions = {
        "s1": _FakeSession(host=first_host, session_id="s1"),
        "s2": _FakeSession(host=second_host, session_id="s2"),
    }

    class _MultiSessionManager:
        def __init__(self) -> None:
            self.cancelled: list[str] = []

        def get(self, session_id: str) -> _FakeSession:
            return sessions[session_id]

        async def cancel(self, session_id: str) -> None:
            self.cancelled.append(session_id)

    manager = _MultiSessionManager()
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient(permission_responder=_block_permission))
    first_prompt = asyncio.create_task(server.prompt([text_block("block")], session_id="s1"))
    await asyncio.wait_for(permission_started.wait(), timeout=1)

    second_response = await asyncio.wait_for(
        server.prompt([text_block("independent")], session_id="s2"),
        timeout=0.2,
    )

    assert second_response.stop_reason == "end_turn"
    await server.cancel("s1")
    first_response = await asyncio.wait_for(first_prompt, timeout=1)
    assert first_response.stop_reason == "cancelled"
    assert manager.cancelled == ["s1"]


@pytest.mark.anyio
async def test_permission_cancel_tracking_is_scoped_by_session_and_request_id() -> None:
    server = ChrysAcpServer(  # type: ignore[arg-type]
        _FakeManager(_FakeHost(event_bus=EventBus())),
        initial_vision=False,
    )
    loop = asyncio.get_running_loop()
    first = loop.create_future()
    second = loop.create_future()
    server._pending_permission_cancels[("s1", "same-request")] = first
    server._pending_permission_cancels[("s2", "same-request")] = second

    server._cancel_pending_waits("s1")

    assert first.done()
    assert not second.done()
    second.cancel()


@pytest.mark.anyio
async def test_out_of_band_approval_cancellation_tombstone_prevents_lost_early_cancel() -> None:
    host = _FakeHost(event_bus=EventBus())
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    client = _FakeClient()
    server.on_connect(client)
    responses: list[ApprovalResponse] = []

    async def _collect(event: ApprovalResponse) -> None:
        responses.append(event)

    await host.event_bus.subscribe(ApprovalResponse, _collect)
    await server._watch_nested_wait_cancellations(_FakeSession(host=host))  # type: ignore[arg-type]
    await host.event_bus.publish(ApprovalCancelled(request_id="early", session_id="s1"))
    assert ("s1", "early") in server._permission_cancel_tombstones

    await asyncio.wait_for(
        server._request_permission(
            "s1",
            ApprovalRequest(
                request_id="early",
                tool_name="zsh",
                tool_kind="shell",
                args={"cmd": "echo"},
                session_id="s1",
            ),
        ),
        timeout=1,
    )

    assert ("s1", "early") not in server._permission_cancel_tombstones
    assert server._pending_permission_cancels == {}
    assert len(responses) == 1
    assert responses[0].approved is False
    assert responses[0].reason == "ACP permission request was cancelled."
    # The wait was already dead — the client must never see a request for it
    # (a sent request would linger as a stale dialog; nothing revokes it).
    assert client.permission_requests == []


@pytest.mark.anyio
async def test_out_of_band_ask_user_clear_tombstone_prevents_lost_early_cancel() -> None:
    host = _FakeHost(event_bus=EventBus())
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    client = _FakeClient()
    server.on_connect(client)
    responses: list[AskUserResponse] = []

    async def _collect(event: AskUserResponse) -> None:
        responses.append(event)

    await host.event_bus.subscribe(AskUserResponse, _collect)
    await server._watch_nested_wait_cancellations(_FakeSession(host=host))  # type: ignore[arg-type]
    await host.event_bus.publish(AskUserTimedOut(request_id="early-input", session_id="s1"))
    assert ("s1", "early-input") in server._input_cancel_tombstones

    await asyncio.wait_for(
        server._request_input(
            "s1",
            QuestionToUser(
                request_id="early-input",
                question="Continue?",
                session_id="s1",
            ),
        ),
        timeout=1,
    )

    assert ("s1", "early-input") not in server._input_cancel_tombstones
    assert server._pending_input_cancels == {}
    assert len(responses) == 1
    assert responses[0].text.startswith("Error:")
    # The wait was already dead — the client must never see a request for it
    # (a sent request would linger as a stale dialog; nothing revokes it).
    assert client.input_requests == []


@pytest.mark.anyio
async def test_prompt_turn_end_sweeps_tombstone_of_judge_aborted_approval() -> None:
    host = _FakeHost(event_bus=EventBus(), outcome=Cancelled())
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    client = _FakeClient()
    server.on_connect(client)
    await server._watch_nested_wait_cancellations(_FakeSession(host=host))  # type: ignore[arg-type]

    async def _judged_then_cancelled(_message: Any):
        # A judging approval installs no cancel waiter (it waits in
        # pending_approvals for ApprovalReviewed). Cancellation aborts the
        # judge, so the out-of-band ApprovalCancelled has nothing to resolve
        # and lands as a tombstone no later request will ever consume.
        yield ApprovalRequest(
            request_id="judged",
            tool_name="zsh",
            tool_kind="shell",
            args={"cmd": "echo"},
            judging=True,
            session_id="s1",
        )
        await host.event_bus.publish(ApprovalCancelled(request_id="judged", session_id="s1"))
        assert ("s1", "judged") in server._permission_cancel_tombstones
        host.last_turn_outcome = host.outcome

    host.iter_turn_events = _judged_then_cancelled  # type: ignore[method-assign]

    response = await asyncio.wait_for(server.prompt([text_block("run")], session_id="s1"), timeout=1)

    assert response.stop_reason == "cancelled"
    assert client.permission_requests == []
    assert server._permission_cancel_tombstones == set()
    assert server._input_cancel_tombstones == set()


@pytest.mark.anyio
async def test_session_teardown_clears_nested_wait_cancellation_tombstones() -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server._permission_cancel_tombstones.update({("s1", "p"), ("s2", "keep")})
    server._input_cancel_tombstones.update({("s1", "q"), ("s2", "keep")})

    await server.close_session("s1")

    assert server._permission_cancel_tombstones == {("s2", "keep")}
    assert server._input_cancel_tombstones == {("s2", "keep")}


@pytest.mark.anyio
async def test_close_session_cancelled_during_wait_release_still_clears_cancellation_state() -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)

    async def _cancel_and_die(session_id: str) -> None:
        _ = session_id
        raise asyncio.CancelledError

    manager.cancel = _cancel_and_die  # type: ignore[method-assign]
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient())
    server._permission_cancel_tombstones.add(("s1", "p"))
    server._input_cancel_tombstones.add(("s1", "q"))

    with pytest.raises(asyncio.CancelledError):
        await server.close_session("s1")

    assert server._permission_cancel_tombstones == set()
    assert server._input_cancel_tombstones == set()
    assert manager.closed_sessions == []


@pytest.mark.anyio
async def test_delete_session_cancelled_during_wait_release_still_clears_cancellation_state() -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)

    async def _cancel_and_die(session_id: str) -> None:
        _ = session_id
        raise asyncio.CancelledError

    manager.cancel = _cancel_and_die  # type: ignore[method-assign]
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient())
    server._permission_cancel_tombstones.add(("s1", "p"))

    with pytest.raises(asyncio.CancelledError):
        await server.delete_session("s1")

    assert server._permission_cancel_tombstones == set()
    assert manager.deleted_sessions == []


@pytest.mark.anyio
async def test_close_releases_pending_ask_user_before_waiting_for_prompt_lock() -> None:
    started = asyncio.Event()

    async def _block_input(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
        started.set()
        await asyncio.Future()
        return {"text": "too late"}

    host = _FakeHost(
        event_bus=EventBus(),
        events=[QuestionToUser(request_id="q1", question="Continue?", session_id="s1")],
        outcome=Cancelled(),
    )
    manager = _FakeManager(host)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient(input_responder=_block_input))
    prompt_task = asyncio.create_task(server.prompt([text_block("ask")], session_id="s1"))
    await asyncio.wait_for(started.wait(), timeout=1)

    await asyncio.wait_for(server.close_session("s1"), timeout=1)
    response = await asyncio.wait_for(prompt_task, timeout=1)

    assert response.stop_reason == "cancelled"
    assert manager.lifecycle_calls == ["begin_close:s1", "cancel:s1", "close:s1"]
    assert server._pending_input_cancels == {}


@pytest.mark.anyio
async def test_close_releases_pending_permission_before_waiting_for_prompt_lock() -> None:
    started = asyncio.Event()

    async def _block_permission(_session_id: str, _tool_call: Any) -> acp_schema.RequestPermissionResponse:
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable after cancellation")

    host = _FakeHost(
        event_bus=EventBus(),
        events=[ApprovalRequest(request_id="r1", tool_name="bash", args={}, session_id="s1")],
        outcome=Cancelled(),
    )
    manager = _FakeManager(host)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient(permission_responder=_block_permission))
    responses: list[ApprovalResponse] = []

    async def _collect(event: ApprovalResponse) -> None:
        responses.append(event)

    await host.event_bus.subscribe(ApprovalResponse, _collect)
    prompt_task = asyncio.create_task(server.prompt([text_block("run")], session_id="s1"))
    await asyncio.wait_for(started.wait(), timeout=1)

    await asyncio.wait_for(server.close_session("s1"), timeout=1)
    response = await asyncio.wait_for(prompt_task, timeout=1)

    assert response.stop_reason == "cancelled"
    assert manager.lifecycle_calls == ["begin_close:s1", "cancel:s1", "close:s1"]
    assert len(responses) == 1
    assert responses[0].approved is False
    assert server._pending_permission_cancels == {}
    assert server._pending_permission_tasks == {}


@pytest.mark.anyio
async def test_delete_releases_pending_permission_before_waiting_for_prompt_lock() -> None:
    started = asyncio.Event()

    async def _block_permission(_session_id: str, _tool_call: Any) -> acp_schema.RequestPermissionResponse:
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable after cancellation")

    host = _FakeHost(
        event_bus=EventBus(),
        events=[ApprovalRequest(request_id="r1", tool_name="bash", args={}, session_id="s1")],
        outcome=Cancelled(),
    )
    manager = _FakeManager(host)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient(permission_responder=_block_permission))
    responses: list[ApprovalResponse] = []

    async def _collect(event: ApprovalResponse) -> None:
        responses.append(event)

    await host.event_bus.subscribe(ApprovalResponse, _collect)
    prompt_task = asyncio.create_task(server.prompt([text_block("run")], session_id="s1"))
    await asyncio.wait_for(started.wait(), timeout=1)

    await asyncio.wait_for(server.delete_session("s1", cwd="/tmp/project"), timeout=1)
    response = await asyncio.wait_for(prompt_task, timeout=1)

    assert response.stop_reason == "cancelled"
    assert manager.lifecycle_calls == ["begin_delete:s1", "cancel:s1", "delete:s1"]
    assert manager.deleted_sessions == [("/tmp/project", "s1")]
    assert len(responses) == 1
    assert responses[0].approved is False
    assert server._pending_permission_cancels == {}
    assert server._pending_permission_tasks == {}


@pytest.mark.anyio
async def test_delete_by_short_id_releases_waits_under_the_canonical_session_id() -> None:
    started = asyncio.Event()

    async def _block_permission(_session_id: str, _tool_call: Any) -> acp_schema.RequestPermissionResponse:
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable after cancellation")

    host = _FakeHost(
        event_bus=EventBus(),
        events=[ApprovalRequest(request_id="r1", tool_name="bash", args={}, session_id="s1")],
        outcome=Cancelled(),
    )
    manager = _FakeManager(host)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient(permission_responder=_block_permission))
    prompt_task = asyncio.create_task(server.prompt([text_block("run")], session_id="s1"))
    await asyncio.wait_for(started.wait(), timeout=1)

    await asyncio.wait_for(server.delete_session("short1", cwd="/tmp/project"), timeout=1)
    response = await asyncio.wait_for(prompt_task, timeout=1)

    assert response.stop_reason == "cancelled"
    assert manager.lifecycle_calls == ["begin_delete:short1", "cancel:s1", "delete:s1"]
    assert manager.deleted_sessions == [("/tmp/project", "s1")]
    assert server._pending_permission_cancels == {}
    assert server._pending_permission_tasks == {}


@pytest.mark.anyio
async def test_delete_rejected_for_workspace_scope_does_not_interrupt_the_active_turn() -> None:
    started = asyncio.Event()

    async def _block_permission(_session_id: str, _tool_call: Any) -> acp_schema.RequestPermissionResponse:
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable after cancellation")

    host = _FakeHost(
        event_bus=EventBus(),
        events=[ApprovalRequest(request_id="r1", tool_name="bash", args={}, session_id="s1")],
        outcome=Cancelled(),
    )
    manager = _FakeManager(host)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient(permission_responder=_block_permission))
    prompt_task = asyncio.create_task(server.prompt([text_block("run")], session_id="s1"))
    await asyncio.wait_for(started.wait(), timeout=1)

    with pytest.raises(RequestError):
        await server.delete_session("s1", cwd="/somewhere/else")

    assert manager.lifecycle_calls == ["begin_delete:s1"]
    assert manager.cancelled_sessions == []
    assert manager.deleted_sessions == []
    assert manager._session.closing is False
    assert list(server._pending_permission_cancels) == [("s1", "r1")]

    await server.cancel("s1")
    response = await asyncio.wait_for(prompt_task, timeout=1)
    assert response.stop_reason == "cancelled"


@pytest.mark.anyio
async def test_delete_rejects_a_queued_prompt_instead_of_admitting_it_into_a_new_turn() -> None:
    started = asyncio.Event()

    async def _block_permission(_session_id: str, _tool_call: Any) -> acp_schema.RequestPermissionResponse:
        started.set()
        await asyncio.Future()
        raise AssertionError("unreachable after cancellation")

    host = _FakeHost(
        event_bus=EventBus(),
        events=[ApprovalRequest(request_id="r1", tool_name="bash", args={}, session_id="s1")],
        outcome=Cancelled(),
    )
    manager = _FakeManager(host)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient(permission_responder=_block_permission))
    first_prompt = asyncio.create_task(server.prompt([text_block("one")], session_id="s1"))
    await asyncio.wait_for(started.wait(), timeout=1)
    second_prompt = asyncio.create_task(server.prompt([text_block("two")], session_id="s1"))
    # Pure scheduler ticks (no wall-clock) so the second prompt reaches its
    # prompt-lock wait before the delete starts.
    for _ in range(5):
        await asyncio.sleep(0)
    assert not second_prompt.done()

    # Deleting must release the first prompt's permission wait AND reject the
    # queued second prompt: without the closing mark set during validation,
    # the second prompt would be admitted the moment the first drains, and
    # the teardown would wedge behind its brand-new turn.
    await asyncio.wait_for(server.delete_session("s1", cwd="/tmp/project"), timeout=1)

    first_response = await asyncio.wait_for(first_prompt, timeout=1)
    assert first_response.stop_reason == "cancelled"
    with pytest.raises(RequestError):
        await asyncio.wait_for(second_prompt, timeout=1)
    assert manager.lifecycle_calls == ["begin_delete:s1", "cancel:s1", "delete:s1"]
    assert manager.deleted_sessions == [("/tmp/project", "s1")]
    assert server._pending_permission_cancels == {}
    assert server._pending_permission_tasks == {}


@pytest.mark.anyio
async def test_settings_reload_tells_the_client_which_values_were_dropped() -> None:
    """A reload that answers "done" and never says what it refused misinforms.

    ``_handle_event`` only runs inside a prompt turn, and the bus does not replay
    events to later subscribers — so the reload's warnings had nowhere to go.
    """
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient()
    manager = _FakeManager(host)
    manager.reload_warnings.append(Warning(code="setting_rejected", message="Ignoring CHRYS_THEME=x", session_id="s1"))
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    await server.ext_method("settings/reload", {"sessionId": "s1"})

    assert ("chrys/warning", {"sessionId": "s1", "code": "setting_rejected", "message": "Ignoring CHRYS_THEME=x"}) in [
        (method, params) for method, params in client.ext_notifications
    ]


@pytest.mark.anyio
async def test_setting_a_config_option_tells_the_client_what_the_reload_refused() -> None:
    """This route persists the value *before* reloading it, so silence is worse here.

    ``rollback_snapshots_keep=0`` is written to disk, clamped to 1 at load, and
    the client was told only that the call succeeded.
    """
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient()
    manager = _FakeManager(host)
    manager.reload_warnings.append(
        Warning(code="setting_clamped", message="Raising CHRYS_ROLLBACK_SNAPSHOTS_KEEP=0 to the minimum of 1.")
    )
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    await server.ext_method(
        "session/set_config_option",
        {"sessionId": "s1", "key": "rollback_snapshots_keep", "value": "0"},
    )

    assert (
        "chrys/warning",
        {
            "sessionId": "s1",
            "code": "setting_clamped",
            "message": "Raising CHRYS_ROLLBACK_SNAPSHOTS_KEEP=0 to the minimum of 1.",
        },
    ) in [(method, params) for method, params in client.ext_notifications]


@pytest.mark.anyio
async def test_error_and_warning_events_are_exposed_as_chrys_extension_notifications() -> None:
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    await server._handle_event(
        "s1",
        Error(code="boom", message="Something failed", recoverable=False, session_id="s1"),
        AcpEventBridge(),
        {},
    )
    await server._handle_event(
        "s1",
        Warning(code="heads_up", message="Be careful", session_id="s1"),
        AcpEventBridge(),
        {},
    )

    assert client.ext_notifications == [
        (
            "chrys/error",
            {
                "sessionId": "s1",
                "code": "boom",
                "message": "Something failed",
                "recoverable": False,
            },
        ),
        (
            "chrys/warning",
            {
                "sessionId": "s1",
                "code": "heads_up",
                "message": "Be careful",
            },
        ),
    ]


@pytest.mark.anyio
async def test_extension_inject_request_publishes_to_active_session() -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]

    await server.ext_method("session/inject", {"sessionId": "s1", "text": "one more thing"})

    assert manager.injected == [("s1", "one more thing")]


@pytest.mark.anyio
async def test_user_inject_result_is_exposed_as_extension_notification() -> None:
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    await server._handle_event(
        "s1",
        UserInjectResult(text="extra context", consumed=True, created_at="now", session_id="s1"),
        AcpEventBridge(),
        {},
    )

    assert client.ext_notifications == [
        (
            "chrys/user_inject_result",
            {
                "sessionId": "s1",
                "text": "extra context",
                "consumed": True,
                "createdAt": "now",
                "injectionId": None,
            },
        )
    ]


@pytest.mark.anyio
async def test_extension_rollback_request_returns_result_and_notifies_client() -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    client = _FakeClient()
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    result = await server.ext_method(
        "session/rollback",
        {
            "sessionId": "s1",
            "targetTurn": 0,
            "revertChanges": True,
            "selectedPaths": ["/workspace/a.py"],
        },
    )

    assert manager.rollbacks == [
        {
            "session_id": "s1",
            "target_turn": 0,
            "revert_changes": True,
            "selected_paths": ["/workspace/a.py"],
        }
    ]
    assert result["targetTurn"] == 0
    assert result["rolledBackUserText"] == "discarded prompt"
    assert result["filesReverted"] == 1
    assert result["restoreResults"][0]["outcome"] == "applied"
    assert client.ext_notifications[0][0] == "chrys/rollback_result"
    assert client.ext_notifications[0][1]["rolledBackUserText"] == "discarded prompt"


def _plan_updates(client: _FakeClient) -> list[Any]:
    return [notification.update for notification in client.updates if notification.update.session_update == "plan"]


@pytest.mark.anyio
async def test_prompt_forwards_todo_list_updates_as_plan_updates() -> None:
    items = [TodoItem(content="write tests", status="in_progress", active_form="writing tests")]
    host = _FakeHost(
        event_bus=EventBus(),
        events=[TodoListUpdated(items=items, session_id="s1")],
        outcome=Cancelled(),
    )
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    await server.prompt([text_block("go")], session_id="s1", message_id="m1")

    plans = _plan_updates(client)
    assert len(plans) == 1
    assert [(entry.content, entry.status, entry.priority) for entry in plans[0].entries] == [
        ("write tests", "in_progress", "medium")
    ]


@pytest.mark.anyio
async def test_new_session_seeds_empty_plan_update() -> None:
    """A fresh session clears any stale plan panel left by a previous session."""
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    response = await server.new_session(cwd="/workspace")

    assert response.session_id == "s1"
    plans = _plan_updates(client)
    assert len(plans) == 1
    assert plans[0].entries == []


@pytest.mark.anyio
async def test_load_session_seeds_plan_from_todo_tracker(monkeypatch) -> None:
    manager = _FakeLoadManager()
    tracker = TodoTracker()
    await tracker.replace((TodoItem(content="restored step", status="in_progress", active_form="restoring"),))
    manager.session.host.engine.todo_tracker = tracker
    client = _FakeClient()
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    async def noop_replay(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(server_module, "replay_session_history", noop_replay)

    await server.load_session(cwd="/workspace", session_id="s1")

    plans = _plan_updates(client)
    assert len(plans) == 1
    assert [(entry.content, entry.status) for entry in plans[0].entries] == [("restored step", "in_progress")]


@pytest.mark.anyio
async def test_load_session_without_todos_seeds_empty_plan(monkeypatch) -> None:
    """A session with no tracker (or an empty one) still sends a clearing plan."""
    manager = _FakeLoadManager()
    assert manager.session.host.engine.todo_tracker is None
    client = _FakeClient()
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    async def noop_replay(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(server_module, "replay_session_history", noop_replay)

    await server.load_session(cwd="/workspace", session_id="s1")

    plans = _plan_updates(client)
    assert len(plans) == 1
    assert plans[0].entries == []


@pytest.mark.anyio
async def test_new_session_forwards_the_loads_settings_warnings() -> None:
    """Session creation runs outside any prompt turn, so the load's verdicts
    only reach the client if the handler forwards what it collected."""
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient()
    manager = _FakeManager(host)
    manager.session_warnings.append(Warning(code="project_config_dormant", message="Project settings found but idle"))
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    await server.new_session(cwd="/workspace")

    assert (
        "chrys/warning",
        {"sessionId": "s1", "code": "project_config_dormant", "message": "Project settings found but idle"},
    ) in [(method, params) for method, params in client.ext_notifications]


@pytest.mark.anyio
async def test_load_session_forwards_the_loads_settings_warnings(monkeypatch) -> None:
    manager = _FakeLoadManager()
    manager.session_warnings.append(Warning(code="project_config_dormant", message="Project settings found but idle"))
    client = _FakeClient()
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    async def noop_replay(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(server_module, "replay_session_history", noop_replay)

    await server.load_session(cwd="/workspace", session_id="s1")

    assert (
        "chrys/warning",
        {"sessionId": "s1", "code": "project_config_dormant", "message": "Project settings found but idle"},
    ) in [(method, params) for method, params in client.ext_notifications]


class _WarningRejectingClient(_FakeClient):
    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        if method == "chrys/warning":
            raise RuntimeError("warning send boom")
        await super().ext_notification(method, params)


@pytest.mark.anyio
async def test_load_session_closes_loaded_host_when_warning_send_fails() -> None:
    """The warning forwarding sits in the same cleanup scope as the replay:
    a send failure must not leave the freshly loaded session in the map."""
    manager = _FakeLoadManager()
    manager.session_warnings.append(Warning(code="project_config_dormant", message="Project settings found but idle"))
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_WarningRejectingClient())

    with pytest.raises(RequestError):
        await server.load_session(cwd="/workspace", session_id="s1")

    assert manager.closed == ["s1"]


@pytest.mark.anyio
async def test_load_session_keeps_existing_host_when_warning_send_fails() -> None:
    manager = _FakeLoadManager(reused_existing=True)
    manager.session_warnings.append(Warning(code="project_config_dormant", message="Project settings found but idle"))
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_WarningRejectingClient())

    with pytest.raises(RequestError):
        await server.load_session(cwd="/workspace", session_id="s1")

    assert manager.closed == []


@pytest.mark.anyio
async def test_new_session_closes_created_host_when_warning_send_fails() -> None:
    """new_session has the same establishment window as load_session: the
    session is in the active map but its id never reached the caller, so a
    failed send must close it rather than strand it."""
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    manager.session_warnings.append(Warning(code="project_config_dormant", message="Project settings found but idle"))
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_WarningRejectingClient())

    with pytest.raises(RuntimeError):
        await server.new_session(cwd="/workspace")

    assert manager.closed_sessions == ["s1"]


@pytest.mark.anyio
async def test_load_session_closes_loaded_host_when_cancelled_mid_establishment(monkeypatch) -> None:
    """Cancellation is a BaseException: it must run the same close as any
    other establishment failure, then keep propagating."""
    manager = _FakeLoadManager()
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient())

    async def cancelled_replay(*_args: Any, **_kwargs: Any) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(server_module, "replay_session_history", cancelled_replay)

    with pytest.raises(asyncio.CancelledError):
        await server.load_session(cwd="/workspace", session_id="s1")

    assert manager.closed == ["s1"]


@pytest.mark.anyio
async def test_extension_rollback_to_turn_sends_plan_update_after_replay(monkeypatch) -> None:
    host = _FakeHost(event_bus=EventBus())
    tracker = TodoTracker()
    await tracker.replace((TodoItem(content="turn two step", status="pending"),))
    host.engine.todo_tracker = tracker
    manager = _FakeManager(host)
    client = _FakeClient()
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)
    replays: list[str] = []

    async def record_replay(*_args: Any, **_kwargs: Any) -> None:
        replays.append("replayed")

    monkeypatch.setattr(server_module, "replay_session_history", record_replay)

    await server.ext_method("session/rollback", {"sessionId": "s1", "targetTurn": 2})

    assert replays == ["replayed"]
    plans = _plan_updates(client)
    assert len(plans) == 1
    assert [(entry.content, entry.status) for entry in plans[0].entries] == [("turn two step", "pending")]


@pytest.mark.anyio
async def test_extension_rollback_to_welcome_sends_empty_plan_update() -> None:
    """targetTurn == 0 takes the no-replay branch but must still clear the plan."""
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    client = _FakeClient()
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    await server.ext_method("session/rollback", {"sessionId": "s1", "targetTurn": 0})

    plans = _plan_updates(client)
    assert len(plans) == 1
    assert plans[0].entries == []


@pytest.mark.anyio
async def test_sub_agent_events_are_exposed_as_extension_notifications() -> None:
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    await server._handle_event(
        "s1",
        SubAgentProgress(
            agent_name="Explore", invocation_id="inv1", tool_call_count=2, total_tokens=42, session_id="s1"
        ),
        AcpEventBridge(),
        {},
    )
    await server._handle_event(
        "s1",
        SubAgentPaused(
            agent_name="Explore",
            invocation_id="inv1",
            tool_name="explore",
            reason="stream_stall",
            last_error="stalled",
            retry_attempts=3,
            session_id="s1",
        ),
        AcpEventBridge(),
        {},
    )

    assert client.ext_notifications == [
        (
            "chrys/sub_agent_progress",
            {
                "sessionId": "s1",
                "agentName": "Explore",
                "invocationId": "inv1",
                "toolCallCount": 2,
                "totalTokens": 42,
                "totalUsageTokens": 0,
                "usageUnreportedAttempts": 0,
            },
        ),
        (
            "chrys/sub_agent_paused",
            {
                "sessionId": "s1",
                "agentName": "Explore",
                "invocationId": "inv1",
                "toolName": "explore",
                "reason": "stream_stall",
                "lastError": "stalled",
                "retryAttempts": 3,
            },
        ),
    ]


@pytest.mark.anyio
async def test_session_history_forwards_cwd_for_workspace_scoping() -> None:
    """session/history is scoped by cwd (like load/delete), not a raw store read."""
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]

    result = await server.ext_method("session/history", {"sessionId": "s1", "cwd": "/tmp/project"})

    assert manager.history_reads == [("/tmp/project", "s1")]
    assert result["sessionId"] == "s1"
    assert result["messages"][0]["contents"][0]["text"] == "hello s1"


@pytest.mark.anyio
async def test_sub_agent_events_also_emit_standard_parent_tool_call_updates() -> None:
    """Sub-agent events are dual-path: ext notification AND standard bridge updates.

    A standard-only ACP client relies on `session/update` tool-call progress on
    the parent sub_agent call; the Chrys extension handler must not suppress it.
    """
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)
    bridge = AcpEventBridge()

    await server._handle_event(
        "s1",
        SubAgentInvocationStart(
            agent_name="Explore", invocation_id="inv1", tool_name="explore", parent_call_id="call-1", session_id="s1"
        ),
        bridge,
        {},
    )
    await server._handle_event(
        "s1",
        SubAgentProgress(
            agent_name="Explore", invocation_id="inv1", tool_call_count=2, total_tokens=42, session_id="s1"
        ),
        bridge,
        {},
    )

    # Extension notifications still flow for Chrys-aware clients.
    assert [method for method, _ in client.ext_notifications] == [
        "chrys/sub_agent_invocation_start",
        "chrys/sub_agent_progress",
    ]
    # And standard tool-call updates on the parent call are no longer suppressed.
    assert len(client.updates) == 2
    assert all(update.update.tool_call_id == "call-1" for update in client.updates)


@pytest.mark.anyio
async def test_compaction_events_emit_extension_notifications() -> None:
    """Main-agent Phase-4 compaction events are ext-only (like ContextCompressed)."""
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)
    bridge = AcpEventBridge()

    await server._handle_event("s1", CompactionStarted(compaction_id="c-1", session_id="s1"), bridge, {})
    await server._handle_event(
        "s1",
        CompactionFinished(
            compaction_id="c-1",
            outcome="ok",
            duration_ms=68_000,
            last_words="## Note",
            format_violation='missing required heading "## Next"',
            session_id="s1",
        ),
        bridge,
        {},
    )
    await server._handle_event(
        "s1",
        ContextPressure(
            reason="side_call_budget",
            attempts=2,
            side_call_tokens=300_000,
            side_call_token_budget=300_000,
            source="sub_agent",
            invocation_id="inv-1",
            session_id="s1",
        ),
        bridge,
        {},
    )

    assert client.ext_notifications == [
        ("chrys/compaction_started", {"sessionId": "s1", "compactionId": "c-1", "phase": "phase4"}),
        (
            "chrys/compaction_finished",
            {
                "sessionId": "s1",
                "compactionId": "c-1",
                "outcome": "ok",
                "durationMs": 68_000,
                "lastWords": "## Note",
                "formatViolation": 'missing required heading "## Next"',
                "failureReason": "",
            },
        ),
        (
            "chrys/context_pressure",
            {
                "sessionId": "s1",
                "reason": "side_call_budget",
                "attempts": 2,
                "sideCallTokens": 300_000,
                "sideCallTokenBudget": 300_000,
                "source": "sub_agent",
                "invocationId": "inv-1",
            },
        ),
    ]
    assert client.updates == []


@pytest.mark.anyio
async def test_sub_agent_compaction_events_are_dual_path() -> None:
    """Sub-agent compaction events emit ext notifications AND standard parent tool-call updates."""
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)
    bridge = AcpEventBridge()

    await server._handle_event(
        "s1",
        SubAgentInvocationStart(
            agent_name="Explore", invocation_id="inv1", tool_name="explore", parent_call_id="call-1", session_id="s1"
        ),
        bridge,
        {},
    )
    await server._handle_event(
        "s1",
        SubAgentCompactionStarted(agent_name="Explore", invocation_id="inv1", compaction_id="c-1", session_id="s1"),
        bridge,
        {},
    )
    await server._handle_event(
        "s1",
        SubAgentCompactionFinished(
            agent_name="Explore",
            invocation_id="inv1",
            compaction_id="c-1",
            outcome="ok",
            duration_ms=2_500,
            format_violation='missing required heading "## Next"',
            session_id="s1",
        ),
        bridge,
        {},
    )

    await server._handle_event(
        "s1",
        SubAgentCompactionCommitted(agent_name="Explore", invocation_id="inv1", compaction_id="c-1", session_id="s1"),
        bridge,
        {},
    )

    assert [method for method, _ in client.ext_notifications] == [
        "chrys/sub_agent_invocation_start",
        "chrys/sub_agent_compaction_started",
        "chrys/sub_agent_compaction_finished",
        "chrys/sub_agent_compaction_committed",
    ]
    started_payload = client.ext_notifications[1][1]
    assert started_payload["invocationId"] == "inv1"
    assert started_payload["compactionId"] == "c-1"
    finished_payload = client.ext_notifications[2][1]
    assert finished_payload["outcome"] == "ok"
    assert finished_payload["durationMs"] == 2_500
    assert finished_payload["formatViolation"] == 'missing required heading "## Next"'
    assert finished_payload["failureReason"] == ""
    committed_payload = client.ext_notifications[3][1]
    assert committed_payload["invocationId"] == "inv1"
    assert committed_payload["compactionId"] == "c-1"
    # Standard-only clients get parent tool-call progress updates too —
    # but the committed signal is ext-only (no human-facing note).
    assert len(client.updates) == 3
    assert all(update.update.tool_call_id == "call-1" for update in client.updates)


@pytest.mark.anyio
async def test_sub_agent_retry_and_abort_extension_requests_route_to_manager() -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]

    await server.ext_method("sub_agent/retry", {"sessionId": "s1", "invocationId": "inv1"})
    await server.ext_method("sub_agent/abort", {"sessionId": "s1", "invocationId": "inv2"})

    assert manager.sub_agent_retries == [("s1", "inv1")]
    assert manager.sub_agent_aborts == [("s1", "inv2")]


@pytest.mark.anyio
async def test_set_session_mode_routes_to_manager_and_emits_current_mode_update() -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    client = _FakeClient()
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    result = await server.set_session_mode(mode_id="auto", session_id="s1")

    assert manager.approval_modes == [("s1", "auto")]
    assert isinstance(result, acp_schema.SetSessionModeResponse)
    assert len(client.updates) == 1
    assert client.updates[0].update.current_mode_id == "auto"


@pytest.mark.anyio
async def test_set_session_mode_rejects_unknown_mode() -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient())

    with pytest.raises(RequestError):
        await server.set_session_mode(mode_id="yolo", session_id="s1")
    assert manager.approval_modes == []


@pytest.mark.anyio
async def test_set_session_model_routes_to_manager_and_sends_runtime_update() -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    client = _FakeClient()
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    result = await server.set_session_model(model_id="m1", session_id="s1")

    assert manager.model_switches == [("s1", "m1")]
    assert isinstance(result, acp_schema.SetSessionModelResponse)
    method, payload = client.ext_notifications[0]
    assert method == "chrys/runtime_update"
    # Unified envelope: {sessionId, runtime: {...}} — same shape as the
    # event-bridged senders, so a client handler reads one shape.
    assert payload["sessionId"] == "s1"
    assert "modelProfileId" in payload["runtime"]
    assert payload["runtime"]["sessionId"] == "s1"


@pytest.mark.anyio
async def test_set_workspace_sends_runtime_update() -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    client = _FakeClient()
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    result = await server.ext_method("session/set_workspace", {"sessionId": "s1", "primaryCwd": "/tmp/project"})

    assert manager.workspace_updates == [("s1", "/tmp/project")]
    assert result["primaryCwd"] == "/tmp/project"
    # A workspace soft-restart can change skills/MCP/memory, so the client must
    # get a refreshed runtime envelope.
    method, payload = client.ext_notifications[0]
    assert method == "chrys/runtime_update"
    assert payload["sessionId"] == "s1"
    assert payload["runtime"]["sessionId"] == "s1"


@pytest.mark.anyio
async def test_session_mode_and_model_states_advertise_available_options() -> None:
    engine = _FakeEngine(
        runtime_details=AgentRuntimeDetails(model=RuntimeModelDetails(profile_id="m1", name="Mock")),
        approval_mode=ApprovalMode.AUTO,
    )
    host = _FakeHost(event_bus=EventBus(), engine=engine)
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]

    modes = server._session_mode_state("s1")
    models = server._session_model_state("s1")

    assert modes.current_mode_id == "auto"
    assert [mode.id for mode in modes.available_modes] == ["manual", "auto", "bypass"]
    assert models.current_model_id == "m1"
    assert [model.model_id for model in models.available_models] == ["m1"]


@pytest.mark.anyio
async def test_session_runtime_extension_returns_runtime_payload() -> None:
    runtime_details = AgentRuntimeDetails(
        model=RuntimeModelDetails(profile_id="m1", name="Mock", max_context_tokens=128000),
        mcp_tools={"server1": ["tool_a"]},
        skill_sources={"builtin": ["skill1"]},
        hook_sources=[
            RuntimeHookSourceDetails(
                scope="project",
                source_path="/repo/.chrys/hooks/hooks.yaml",
                hooks=[
                    RuntimeHookDetails(
                        id="guard",
                        event="before_tool_call",
                        execution_mode="blocking",
                        enabled=True,
                        description="Guard writes",
                    )
                ],
            )
        ],
    )
    engine = _FakeEngine(
        runtime_details=runtime_details,
        usage=UsageUpdate(
            input_tokens=60,
            output_tokens=40,
            total_tokens=100,
            pct=12.5,
            max_context_tokens=128000,
        ),
    )
    host = _FakeHost(event_bus=EventBus(), engine=engine)
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]

    result = await server.ext_method("chrys/session_runtime", {"sessionId": "s1"})

    assert result["sessionId"] == "s1"
    assert result["agentProfile"] == "Code"
    assert result["modelProfileId"] == "m1"
    assert result["maxContextTokens"] == 128000
    assert result["inputTokens"] == 60
    assert result["outputTokens"] == 40
    assert result["totalTokens"] == 100
    assert result["pct"] == 12.5
    assert result["runtimeDetails"]["mcp_tools"] == {"server1": ["tool_a"]}
    assert result["runtimeDetails"]["model"]["selection_source"] == "active"
    assert result["runtimeDetails"]["hook_sources"] == [
        {
            "scope": "project",
            "source_path": "/repo/.chrys/hooks/hooks.yaml",
            "hooks": [
                {
                    "id": "guard",
                    "event": "before_tool_call",
                    "execution_mode": "blocking",
                    "enabled": True,
                    "description": "Guard writes",
                }
            ],
        }
    ]


@pytest.mark.anyio
async def test_session_mutations_extension_returns_tracker_summary() -> None:
    mutation = FileMutation(
        path="/workspace/a.py",
        operation=MutationOp.MODIFY,
        source=MutationSource.WRITE_FILE,
        tool_call_id="tc1",
        timestamp=1.0,
        before_hash="before",
        after_hash="after",
    )
    tracker = _FakeMutationTracker(
        store=_FakeBlobStore(blobs={}),
        turns=[TurnMutations(turn_id=1, mutations=[mutation])],
        session_summary={"/workspace/a.py": FileHashDiff(before="before", after="after")},
    )
    engine = _FakeEngine(mutation_tracker=tracker, current_turn_number=2, rollback_turns=[0, 1])
    host = _FakeHost(event_bus=EventBus(), engine=engine)
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]

    result = await server.ext_method("session/mutations", {"sessionId": "s1"})

    assert result["sessionId"] == "s1"
    assert result["currentTurn"] == 2
    assert result["availableRollbackTurns"] == [0, 1]
    assert result["turns"] == [
        {
            "turnId": 1,
            "mutationCount": 1,
            "mutations": [
                {
                    "path": "/workspace/a.py",
                    "operation": "modify",
                    "source": "write_file",
                    "toolCallId": "tc1",
                    "timestamp": 1.0,
                    "oldPath": None,
                    "beforeHash": "before",
                    "afterHash": "after",
                    "beforeSkip": None,
                    "afterSkip": None,
                    "provenance": "proven",
                    "contested": False,
                }
            ],
        }
    ]
    assert result["files"] == [
        {
            "path": "/workspace/a.py",
            "beforeHash": "before",
            "afterHash": "after",
            "operation": "modify",
            "contested": False,
            "inferred": False,
        }
    ]


@pytest.mark.anyio
async def test_session_mutations_extension_without_tracker_returns_empty_summary() -> None:
    host = _FakeHost(event_bus=EventBus(), engine=_FakeEngine(mutation_tracker=None, current_turn_number=3))
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]

    result = await server.ext_method("session/mutations", {"sessionId": "s1"})

    assert result == {
        "sessionId": "s1",
        "currentTurn": 3,
        "availableRollbackTurns": [],
        "turns": [],
        "files": [],
    }


@pytest.mark.anyio
async def test_session_diff_extension_returns_text_entries_and_filters() -> None:
    tracker = _FakeMutationTracker(
        store=_FakeBlobStore(blobs={"before": b"old", "after": b"new"}),
        session_summary={"/workspace/a.py": FileHashDiff(before="before", after="after", contested=True)},
        turn_summaries={1: {"/workspace/b.py": FileHashDiff(before=None, after="after", inferred=True)}},
    )
    host = _FakeHost(event_bus=EventBus(), engine=_FakeEngine(mutation_tracker=tracker))
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]

    session_diff = await server.ext_method("session/diff", {"sessionId": "s1"})
    path_diff = await server.ext_method("session/diff", {"sessionId": "s1", "path": "/workspace/a.py"})
    turn_diff = await server.ext_method("session/diff", {"sessionId": "s1", "turn": 1})

    assert session_diff["entries"] == [
        {
            "path": "/workspace/a.py",
            "operation": "modify",
            "beforeHash": "before",
            "afterHash": "after",
            "beforeText": "old",
            "afterText": "new",
            "isBinary": False,
            "bytesChanged": True,
            "contested": True,
            "inferred": False,
        }
    ]
    assert path_diff["entries"][0]["path"] == "/workspace/a.py"
    assert turn_diff == {
        "sessionId": "s1",
        "turn": 1,
        "entries": [
            {
                "path": "/workspace/b.py",
                "operation": "create",
                "beforeHash": None,
                "afterHash": "after",
                "beforeText": "",
                "afterText": "new",
                "isBinary": False,
                "bytesChanged": True,
                "contested": False,
                "inferred": True,
            }
        ],
    }


@pytest.mark.anyio
async def test_integer_extension_params_reject_bool_values() -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]

    with pytest.raises(RequestError):
        await server.ext_method("session/rollback", {"sessionId": "s1", "targetTurn": True})
    with pytest.raises(RequestError):
        await server.ext_method("session/diff", {"sessionId": "s1", "turn": False})

    assert manager.rollbacks == []


@pytest.mark.anyio
async def test_session_delete_extension_routes_to_manager() -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]

    await server.ext_method("session/delete", {"sessionId": "s1", "cwd": "/tmp/project"})

    assert manager.deleted_sessions == [("/tmp/project", "s1")]


@pytest.mark.anyio
async def test_mcp_list_and_skills_list_extension_read_runtime_details() -> None:
    runtime_details = AgentRuntimeDetails(
        mcp_tools={"server1": ["tool_a"]},
        mcp_failures={"server2": "timeout"},
        skill_sources={"builtin": ["skill1"]},
        skill_details=[RuntimeSkillDetails(name="skill1", description="desc", source="builtin")],
    )
    host = _FakeHost(event_bus=EventBus(), engine=_FakeEngine(runtime_details=runtime_details))
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]

    mcp = await server.ext_method("mcp/list", {"sessionId": "s1"})
    skills = await server.ext_method("skills/list", {"sessionId": "s1"})

    assert mcp == {
        "sessionId": "s1",
        "mcpTools": {"server1": ["tool_a"]},
        "mcpFailures": {"server2": "timeout"},
    }
    assert skills == {
        "sessionId": "s1",
        "skillSources": {"builtin": ["skill1"]},
        "skillDetails": [{"name": "skill1", "description": "desc", "source": "builtin"}],
    }


@pytest.mark.anyio
async def test_initialize_advertises_additional_directories_capability() -> None:
    server = ChrysAcpServer(_FakeManager(_FakeHost(event_bus=EventBus())), initial_vision=False)  # type: ignore[arg-type]

    response = await server.initialize(protocol_version=PROTOCOL_VERSION)

    caps = response.agentCapabilities.sessionCapabilities
    # Multi-root support must be discoverable so conforming clients send the option.
    assert caps.additional_directories is not None
    assert caps.close is not None
    assert caps.list is not None


@pytest.mark.anyio
async def test_remaining_extension_requests_route_to_manager() -> None:
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    client = _FakeClient()
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    switch = await server.ext_method("session/switch_agent", {"sessionId": "s1", "agentProfile": "QA"})
    await server.ext_method("settings/reload", {"sessionId": "s1"})
    workspace = await server.ext_method("session/set_workspace", {"sessionId": "s1", "primaryCwd": "/tmp/project"})
    history = await server.ext_method("session/history", {"sessionId": "s1"})
    agents = await server.ext_method("profiles/agents/list", {})
    agent = await server.ext_method("profiles/agents/read", {"name": "Code"})
    agent_write = await server.ext_method("profiles/agents/write", {"profile": {"name": "Custom"}})
    agent_delete = await server.ext_method("profiles/agents/delete", {"name": "Custom"})
    agent_reset = await server.ext_method("profiles/agents/reset", {"name": "Code"})
    models = await server.ext_method("profiles/models/list", {})
    model = await server.ext_method("profiles/models/read", {"id": "m1"})
    model_write = await server.ext_method("profiles/models/write", {"profile": {"id": "m2", "name": "Other"}})
    model_delete = await server.ext_method("profiles/models/delete", {"id": "m2"})
    mcp_test = await server.ext_method("mcp/test", {"server": {"name": "test", "transport": "stdio"}})
    config_options = await server.ext_method("settings/options", {})
    config_options_scoped = await server.ext_method("settings/options", {"sessionId": "s1"})
    config_set = await server.ext_method(
        "session/set_config_option",
        {"sessionId": "s1", "key": "theme", "value": "dark"},
    )

    assert manager.agent_switches == [("s1", "QA")]
    assert manager.workspace_updates == [("s1", "/tmp/project")]
    assert switch["toProfile"] == "QA"
    assert workspace["primaryCwd"] == "/tmp/project"
    assert history["messages"][0]["contents"][0]["text"] == "hello s1"
    assert agents["agents"][0]["name"] == "Code"
    assert agent["profile"]["name"] == "Code"
    assert agent_write["profile"]["name"] == "Custom"
    assert agent_delete["deleted"] is True
    assert agent_reset == {"profile": {"name": "Code", "builtin": True}, "changed": True}
    assert models["models"][0]["id"] == "m1"
    assert model["profile"]["id"] == "m1"
    assert model_write["profile"]["name"] == "Other"
    assert model_delete["deleted"] is True
    assert manager.settings_reloads == ["s1", "s1"]
    # settings/reload must push a runtime_update so clients refresh after a reload.
    assert ("chrys/runtime_update", {"sessionId": "s1"}) in [
        (method, {"sessionId": params.get("sessionId")}) for method, params in client.ext_notifications
    ]
    assert mcp_test["ok"] is True
    assert manager.mcp_tests == [{"name": "test", "transport": "stdio"}]
    assert config_options["options"][0]["key"] == "theme"
    assert config_options_scoped["options"][0]["settingKey"] == "ui.theme"
    # The scope the manager answers for is the server's to forward verbatim:
    # no sessionId means the base query, never a guessed session.
    assert manager.config_option_queries == [None, "s1"]
    assert config_set == {
        "sessionId": "s1",
        "key": "theme",
        "envKey": "CHRYS_THEME",
        "settingKey": "ui.theme",
        "value": "dark",
    }
    assert manager.config_updates == [("theme", "dark")]


@pytest.mark.anyio
async def test_runtime_extension_events_are_exposed() -> None:
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient()
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)

    await server._handle_event(
        "s1",
        AgentLoadProgress(phase="mcp", message="Loading tools", current=1, total=2),
        AcpEventBridge(),
        {},
    )
    await server._handle_event(
        "s1",
        UsageUpdate(total_tokens=100, pct=10.0, max_context_tokens=1000),
        AcpEventBridge(),
        {},
    )
    await server._handle_event(
        "s1",
        WorkspaceUpdated(primary_cwd="/tmp/project", working_dirs=["/tmp/project"], reference_files=[]),
        AcpEventBridge(),
        {},
    )

    assert client.ext_notifications == [
        (
            "chrys/agent_load_progress",
            {
                "sessionId": "s1",
                "phase": "mcp",
                "message": "Loading tools",
                "serverName": "",
                "current": 1,
                "total": 2,
                "failed": 0,
            },
        ),
        (
            "chrys/usage_update",
            {
                "sessionId": "s1",
                "agentProfile": "",
                "usageSourceId": "",
                "inputTokens": 0,
                "outputTokens": 0,
                "totalTokens": 100,
                "pct": 10.0,
                "maxContextTokens": 1000,
                "totalSessionTokens": 0,
                "totalSessionInputTokens": 0,
                "totalSessionOutputTokens": 0,
                "cacheHitTokens": None,
                "totalSessionCacheHitTokens": None,
                "localTokens": 0,
                "calibrationRatio": 1.0,
                "systemOverheadTokens": 0,
            },
        ),
        (
            "chrys/workspace_updated",
            {
                "sessionId": "s1",
                "primaryCwd": "/tmp/project",
                "workingDirs": ["/tmp/project"],
                "referenceFiles": [],
            },
        ),
    ]


@pytest.mark.anyio
async def test_cancelled_permission_request_rejects_tool_call() -> None:
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient(permission_outcome=acp_schema.DeniedOutcome(outcome="cancelled"))
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)
    responses: list[ApprovalResponse] = []

    async def _collect(event: ApprovalResponse) -> None:
        responses.append(event)

    await host.event_bus.subscribe(ApprovalResponse, _collect)

    await server._handle_event(
        "s1",
        ApprovalRequest(
            request_id="r1", tool_name="bash", args={}, intent_summary="Run shell command", session_id="s1"
        ),
        AcpEventBridge(),
        {},
    )

    assert len(responses) == 1
    assert responses[0].approved is False
    assert responses[0].reason == "ACP permission request was cancelled."


@pytest.mark.anyio
async def test_failed_permission_request_rejects_tool_call() -> None:
    host = _FakeHost(event_bus=EventBus())
    client = _FakeClient(permission_exc=RuntimeError("client gone"))
    server = ChrysAcpServer(_FakeManager(host), initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(client)
    responses: list[ApprovalResponse] = []

    async def _collect(event: ApprovalResponse) -> None:
        responses.append(event)

    await host.event_bus.subscribe(ApprovalResponse, _collect)

    await server._handle_event(
        "s1",
        ApprovalRequest(
            request_id="r1", tool_name="bash", args={}, intent_summary="Run shell command", session_id="s1"
        ),
        AcpEventBridge(),
        {},
    )

    assert len(responses) == 1
    assert responses[0].approved is False
    assert responses[0].reason == "ACP permission request failed or was cancelled."


@pytest.mark.anyio
async def test_load_session_closes_loaded_host_when_history_replay_fails(monkeypatch) -> None:
    manager = _FakeLoadManager()
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient())

    async def fail_replay(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("replay boom")

    monkeypatch.setattr(server_module, "replay_session_history", fail_replay)

    with pytest.raises(RequestError):
        await server.load_session(cwd="/workspace", session_id="s1")

    assert manager.closed == ["s1"]


@pytest.mark.anyio
async def test_load_session_keeps_existing_host_when_history_replay_fails(monkeypatch) -> None:
    manager = _FakeLoadManager(reused_existing=True)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient())

    async def fail_replay(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("replay boom")

    monkeypatch.setattr(server_module, "replay_session_history", fail_replay)

    with pytest.raises(RequestError):
        await server.load_session(cwd="/workspace", session_id="s1")

    assert manager.closed == []


@pytest.mark.anyio
async def test_settings_options_is_read_off_the_event_loop_thread() -> None:
    # The read takes the settings document's file lock, whose timeout is ten
    # seconds. Inline, that wait belongs to the ACP loop, so one held lock
    # stalls every session on the connection rather than just this request.
    host = _FakeHost(event_bus=EventBus())
    manager = _FakeManager(host)
    server = ChrysAcpServer(manager, initial_vision=False)  # type: ignore[arg-type]
    server.on_connect(_FakeClient())

    read_threads: list[int] = []
    inner = manager.get_config_options

    def recording(session_id: str | None = None) -> dict[str, object]:
        read_threads.append(threading.get_ident())
        return inner(session_id)

    manager.get_config_options = recording  # type: ignore[method-assign]

    await server.ext_method("settings/options", {})

    assert read_threads and threading.get_ident() not in read_threads
