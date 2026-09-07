# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the in-process Chrys PACT role adapter."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest
from acp import schema as acp_schema
from pact_core.schemas import Role, TurnRequest

from chrys.foundation.config.settings_store import LoadedSettings, load_settings
from chrys.foundation.events.types import (
    AgentMessage,
    Error,
    Event,
    TodoListUpdated,
    ToolCallResult,
    ToolCallStart,
    UsageUpdate,
)
from chrys.foundation.models.todos import TodoItem
from chrys.foundation.tool_kinds import KIND_FILESYSTEM_READ
from chrys.orchestration.session_host import Cancelled, EndTurn, Errored, TurnOutcome
from chrys.pact.role_runner import (
    InProcessChrysAdapter,
    RoleRuntimeUnresponsive,
    RoleTurnCancelled,
    RoleUpdateError,
    SemanticRole,
)
from chrys.service.approval.policy import ApprovalMode


@dataclass
class _HostScript:
    outcome: TurnOutcome | None
    events: list[Event] = field(default_factory=list)
    decision_text: str | None = None
    decision_bytes: bytes | None = None
    block_until_cancel: bool = False
    never_finishes: bool = False
    shield_cancel_cleanup: bool = False
    session_id: str = "inner-session"
    followup_outcome: TurnOutcome | None = None


class _FakeHost:
    def __init__(self, script: _HostScript, cwd: Path) -> None:
        self._script = script
        self._cwd = cwd
        self._last_turn_outcome: TurnOutcome | None = None
        self.prompts: list[str] = []
        self.iter_started = asyncio.Event()
        self._release = asyncio.Event()
        self.turn_cleanup_started = asyncio.Event()
        self.turn_cleanup_finished = asyncio.Event()
        self.cancelled = False
        self.shutdown_called = False
        self._turn_task: asyncio.Task[None] | None = None

    @property
    def session_id(self) -> str | None:
        return self._script.session_id

    @property
    def last_turn_outcome(self) -> TurnOutcome | None:
        return self._last_turn_outcome

    async def iter_turn_events(self, message: str):
        self.prompts.append(message)
        self.iter_started.set()
        self._turn_task = asyncio.current_task()
        if self._script.block_until_cancel:
            await self._release.wait()
        if self._script.never_finishes:
            try:
                await asyncio.Future()
            finally:
                if self._script.shield_cancel_cleanup:
                    self.turn_cleanup_started.set()
                    await asyncio.shield(self._release.wait())
                    self.turn_cleanup_finished.set()
        if self._script.decision_text is not None:
            decision_path = self._cwd / ".pact-io" / "reviewer-decision.json"
            decision_path.parent.mkdir(parents=True, exist_ok=True)
            decision_path.write_text(self._script.decision_text, encoding="utf-8")
        elif self._script.decision_bytes is not None:
            decision_path = self._cwd / ".pact-io" / "reviewer-decision.json"
            decision_path.parent.mkdir(parents=True, exist_ok=True)
            decision_path.write_bytes(self._script.decision_bytes)
        for event in self._script.events:
            yield event
        self._last_turn_outcome = self._script.outcome
        if len(self.prompts) > 1 and self._script.followup_outcome is not None:
            self._last_turn_outcome = self._script.followup_outcome

    async def cancel_current_turn(self) -> None:
        self.cancelled = True
        if self._script.never_finishes and self._turn_task is not None:
            self._turn_task.cancel()
        else:
            self._release.set()

    async def shutdown(self) -> None:
        self.shutdown_called = True


@dataclass(frozen=True)
class _FactoryCall:
    profile_name: str
    loaded_settings: LoadedSettings
    approval_mode: ApprovalMode
    cwd: str
    allow_user_interaction: bool


class _FakeHostFactory:
    def __init__(self, scripts: list[_HostScript]) -> None:
        self._scripts = list(scripts)
        self.calls: list[_FactoryCall] = []
        self.hosts: list[_FakeHost] = []
        self.created = asyncio.Event()

    def __call__(
        self,
        *,
        profile_name: str,
        loaded_settings: LoadedSettings,
        approval_mode: ApprovalMode,
        cwd: str,
        allow_user_interaction: bool,
    ) -> _FakeHost:
        if not self._scripts:
            raise AssertionError("fake host script exhausted")
        self.calls.append(
            _FactoryCall(
                profile_name=profile_name,
                loaded_settings=loaded_settings,
                approval_mode=approval_mode,
                cwd=cwd,
                allow_user_interaction=allow_user_interaction,
            )
        )
        host = _FakeHost(self._scripts.pop(0), Path(cwd))
        self.hosts.append(host)
        self.created.set()
        return host


def _base_settings() -> LoadedSettings:
    return load_settings(env={})


def _adapter(
    *,
    semantic_role: SemanticRole,
    base_settings: LoadedSettings,
    factory: _FakeHostFactory,
    updates: list[object],
    abort_event: threading.Event | None = None,
    cleanup_grace_seconds: float = 5.0,
) -> InProcessChrysAdapter:
    async def _send_update(update: object) -> None:
        updates.append(update)

    return InProcessChrysAdapter(
        semantic_role=semantic_role,
        profile_name="Code",
        outer_loop=asyncio.get_running_loop(),
        campaign_id="campaign-1",
        send_update=_send_update,
        abort_event=abort_event or threading.Event(),
        loaded_settings=base_settings,
        host_factory=factory,
        cleanup_grace_seconds=cleanup_grace_seconds,
    )


def _request(
    workdir: Path,
    *,
    role: Role = "worker",
    artifact_dir: Path | None = None,
    timeout_seconds: float = 1.0,
) -> TurnRequest:
    workdir.mkdir(parents=True, exist_ok=True)
    return TurnRequest(
        role=role,
        prompt="Follow the PACT role prompt.",
        workdir=workdir,
        timeout_seconds=timeout_seconds,
        artifact_dir=artifact_dir,
    )


@pytest.mark.asyncio
async def test_fresh_hosts_use_turn_workdirs_namespace_tools_and_filter_todos(tmp_path: Path) -> None:
    first_session = "inner-1"
    factory = _FakeHostFactory(
        [
            _HostScript(
                outcome=EndTurn(final_text="first final"),
                session_id=first_session,
                events=[
                    ToolCallStart(
                        tool_name="read_file",
                        tool_kind=KIND_FILESYSTEM_READ,
                        call_id="call-1",
                        args={"path": "README.md"},
                        session_id=first_session,
                    ),
                    TodoListUpdated(
                        items=[TodoItem(content="inner todo", status="in_progress")],
                        session_id=first_session,
                    ),
                    ToolCallResult(
                        tool_name="read_file",
                        call_id="call-1",
                        result="contents",
                        session_id=first_session,
                    ),
                    UsageUpdate(
                        total_tokens=5,
                        max_context_tokens=100,
                        usage_source_id=first_session,
                        session_id=first_session,
                    ),
                    AgentMessage(text="first final", is_final=True, session_id=first_session),
                ],
            ),
            _HostScript(
                outcome=EndTurn(final_text="second final"),
                session_id="inner-2",
                events=[AgentMessage(text="second final", is_final=True, session_id="inner-2")],
            ),
        ]
    )
    base = _base_settings()
    updates: list[object] = []
    adapter = _adapter(semantic_role="worker", base_settings=base, factory=factory, updates=updates)
    first_workdir = tmp_path / "first"
    second_workdir = tmp_path / "second"

    with patch("chrys.pact.role_runner.load_settings", autospec=True, return_value=base) as settings_load:
        first = await asyncio.to_thread(adapter.run_turn, _request(first_workdir))
        second = await asyncio.to_thread(adapter.run_turn, _request(second_workdir))

    assert first.status == "completed"
    assert first.final_text == "first final"
    assert first.session_id == first_session
    assert first.review_decision.verdict_status == "not_applicable"
    assert second.status == "completed"
    assert second.final_text == "second final"
    assert len(factory.hosts) == 2
    assert all(host.shutdown_called for host in factory.hosts)
    assert [call.cwd for call in factory.calls] == [str(first_workdir), str(second_workdir)]
    assert all(call.profile_name == "Code" for call in factory.calls)
    assert all(call.approval_mode is ApprovalMode.BYPASS for call in factory.calls)
    assert all(not call.allow_user_interaction for call in factory.calls)
    assert [call.kwargs["project_root"] for call in settings_load.call_args_list] == [
        first_workdir,
        second_workdir,
    ]

    tool_starts = [update for update in updates if isinstance(update, acp_schema.ToolCallStart)]
    tool_progress = [update for update in updates if isinstance(update, acp_schema.ToolCallProgress)]
    assert any(update.tool_call_id == "pact:campaign-1:worker:1:role" for update in tool_starts)
    assert any(update.tool_call_id == "pact:campaign-1:worker:1:inner:call-1" for update in tool_starts)
    assert any(update.tool_call_id == "pact:campaign-1:worker:1:inner:call-1" for update in tool_progress)
    assert any(
        update.tool_call_id == "pact:campaign-1:worker:2:role" and update.status == "completed"
        for update in tool_progress
    )
    assert any(isinstance(update, acp_schema.UsageUpdate) and update.used == 5 for update in updates)
    assert not any(isinstance(update, acp_schema.AgentPlanUpdate) for update in updates)


@pytest.mark.asyncio
async def test_reviewer_captures_typed_plan_challenge_and_clears_transport(tmp_path: Path) -> None:
    decision = {
        "verdict": "continue",
        "plan_challenge": {
            "reason": "dependency unavailable",
            "gap_signature": "dependency:unavailable",
            "recommended_action": "replan",
        },
    }
    raw_decision = json.dumps(decision)
    factory = _FakeHostFactory([_HostScript(outcome=EndTurn(final_text="review evidence"), decision_text=raw_decision)])
    base = _base_settings()
    updates: list[object] = []
    adapter = _adapter(semantic_role="reviewer", base_settings=base, factory=factory, updates=updates)
    workdir = tmp_path / "workdir"
    artifact_dir = tmp_path / "artifacts"
    stale_path = workdir / ".pact-io" / "reviewer-decision.json"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text('{"verdict":"failed"}', encoding="utf-8")

    with patch("chrys.pact.role_runner.load_settings", autospec=True, return_value=base):
        result = await asyncio.to_thread(
            adapter.run_turn,
            _request(workdir, role="reviewer", artifact_dir=artifact_dir),
        )

    assert result.status == "completed"
    assert result.final_text == "review evidence"
    assert result.review_decision.verdict_status == "valid"
    assert result.review_decision.verdict == "continue"
    assert result.review_decision.plan_challenge_status == "valid"
    challenge = result.review_decision.plan_challenge
    assert challenge is not None
    assert challenge.reason == "dependency unavailable"
    assert challenge.gap_signature == "dependency:unavailable"
    assert challenge.recommended_action == "replan"
    assert not stale_path.exists()
    assert (artifact_dir / "reviewer-decision-raw.txt").read_text(encoding="utf-8") == raw_decision
    assert ".pact-io/reviewer-decision.json" in factory.hosts[0].prompts[0]
    assert "plan_challenge" in factory.hosts[0].prompts[0]
    assert factory.hosts[0].shutdown_called


@pytest.mark.parametrize(
    ("decision_text", "expected_status"),
    [
        (None, "missing"),
        ("{", "malformed"),
    ],
)
@pytest.mark.asyncio
async def test_reviewer_reports_missing_and_malformed_decisions(
    tmp_path: Path,
    decision_text: str | None,
    expected_status: str,
) -> None:
    factory = _FakeHostFactory(
        [_HostScript(outcome=EndTurn(final_text="review evidence"), decision_text=decision_text)]
    )
    base = _base_settings()
    adapter = _adapter(semantic_role="reviewer", base_settings=base, factory=factory, updates=[])
    workdir = tmp_path / expected_status
    artifact_dir = tmp_path / f"{expected_status}-artifacts"

    with patch("chrys.pact.role_runner.load_settings", autospec=True, return_value=base):
        result = await asyncio.to_thread(
            adapter.run_turn,
            _request(workdir, role="reviewer", artifact_dir=artifact_dir),
        )

    assert result.status == "completed"
    assert result.review_decision.verdict_status == expected_status
    assert not (workdir / ".pact-io" / "reviewer-decision.json").exists()
    raw_artifact = artifact_dir / "reviewer-decision-raw.txt"
    assert raw_artifact.exists() is (decision_text is not None)


@pytest.mark.asyncio
async def test_reviewer_invalid_utf8_remains_completed_and_triggers_decision_repair(tmp_path: Path) -> None:
    factory = _FakeHostFactory([_HostScript(outcome=EndTurn(final_text="review evidence"), decision_bytes=b"\xff")])
    base = _base_settings()
    adapter = _adapter(semantic_role="reviewer", base_settings=base, factory=factory, updates=[])
    workdir = tmp_path / "invalid-utf8"
    artifact_dir = tmp_path / "invalid-utf8-artifacts"

    with patch("chrys.pact.role_runner.load_settings", autospec=True, return_value=base):
        result = await asyncio.to_thread(
            adapter.run_turn,
            _request(workdir, role="reviewer", artifact_dir=artifact_dir),
        )

    assert result.status == "completed"
    assert result.final_text == "review evidence"
    assert result.review_decision.verdict_status == "malformed"
    assert not (workdir / ".pact-io" / "reviewer-decision.json").exists()
    assert (artifact_dir / "reviewer-decision-raw.txt").read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    ("outcome", "expected_status"),
    [
        (EndTurn(final_text=""), "output_missing"),
        (
            Errored(error=Error(code="no_final_response", message="no final response")),
            "output_missing",
        ),
        (
            Errored(error=Error(code="provider_failed", message="provider failed")),
            "spawn_failed",
        ),
    ],
)
@pytest.mark.asyncio
async def test_terminal_outcomes_map_to_pact_statuses(
    tmp_path: Path,
    outcome: TurnOutcome,
    expected_status: str,
) -> None:
    factory = _FakeHostFactory([_HostScript(outcome=outcome)])
    base = _base_settings()
    updates: list[object] = []
    adapter = _adapter(semantic_role="worker", base_settings=base, factory=factory, updates=updates)

    with patch("chrys.pact.role_runner.load_settings", autospec=True, return_value=base):
        result = await asyncio.to_thread(adapter.run_turn, _request(tmp_path / expected_status))

    assert result.status == expected_status
    assert result.final_text == ""
    assert result.exit_code is None
    assert result.stderr_tail
    assert factory.hosts[0].shutdown_called
    terminal = [
        update
        for update in updates
        if isinstance(update, acp_schema.ToolCallProgress) and update.tool_call_id.endswith(":role")
    ][-1]
    assert terminal.status == "failed"


@pytest.mark.asyncio
async def test_timeout_interrupts_and_shuts_down_host(tmp_path: Path) -> None:
    factory = _FakeHostFactory([_HostScript(outcome=None, never_finishes=True)])
    base = _base_settings()
    updates: list[object] = []
    adapter = _adapter(semantic_role="worker", base_settings=base, factory=factory, updates=updates)

    with patch("chrys.pact.role_runner.load_settings", autospec=True, return_value=base):
        result = await asyncio.to_thread(
            adapter.run_turn,
            _request(tmp_path / "timeout", timeout_seconds=0.01),
        )

    assert result.status == "timeout"
    assert result.stderr_tail == "TimeoutError"
    assert factory.hosts[0].cancelled
    assert factory.hosts[0].shutdown_called


@pytest.mark.asyncio
async def test_timeout_fails_fatally_when_generator_cleanup_ignores_cancellation(tmp_path: Path) -> None:
    factory = _FakeHostFactory(
        [
            _HostScript(
                outcome=None,
                never_finishes=True,
                shield_cancel_cleanup=True,
            )
        ]
    )
    base = _base_settings()
    abort_event = threading.Event()
    adapter = _adapter(
        semantic_role="worker",
        base_settings=base,
        factory=factory,
        updates=[],
        abort_event=abort_event,
        cleanup_grace_seconds=0.05,
    )
    host: _FakeHost | None = None

    try:
        with (
            patch("chrys.pact.role_runner.load_settings", autospec=True, return_value=base),
            pytest.raises(RoleRuntimeUnresponsive, match="did not stop"),
        ):
            await asyncio.wait_for(
                asyncio.to_thread(
                    adapter.run_turn,
                    _request(tmp_path / "unresponsive-timeout", timeout_seconds=0.01),
                ),
                timeout=0.5,
            )

        host = factory.hosts[0]
        assert abort_event.is_set()
        assert host.cancelled
        assert not host.shutdown_called
        assert adapter._active_host is host
        assert adapter._active_turn_task is not None
        assert not adapter._active_turn_task.done()
    finally:
        if host is None and factory.hosts:
            host = factory.hosts[0]
        if host is not None:
            host._release.set()
            await asyncio.wait_for(host.turn_cleanup_finished.wait(), timeout=0.5)
            await asyncio.sleep(0)
            assert adapter._active_turn_task is None
            assert not adapter._retained_tasks
            await host.shutdown()


@pytest.mark.asyncio
async def test_cancel_aborts_active_turn_without_returning_pact_success(tmp_path: Path) -> None:
    factory = _FakeHostFactory(
        [
            _HostScript(
                outcome=Cancelled(reason="outer ACP cancelled"),
                block_until_cancel=True,
            )
        ]
    )
    base = _base_settings()
    updates: list[object] = []
    abort_event = threading.Event()
    adapter = _adapter(
        semantic_role="worker",
        base_settings=base,
        factory=factory,
        updates=updates,
        abort_event=abort_event,
    )

    with patch("chrys.pact.role_runner.load_settings", autospec=True, return_value=base):
        run_task = asyncio.create_task(asyncio.to_thread(adapter.run_turn, _request(tmp_path / "cancel")))
        await factory.created.wait()
        host = factory.hosts[0]
        await host.iter_started.wait()
        await adapter.cancel_current_turn()
        with pytest.raises(RoleTurnCancelled, match="cancelled"):
            await run_task

    assert abort_event.is_set()
    assert host.cancelled
    assert host.shutdown_called
    terminal = [
        update
        for update in updates
        if isinstance(update, acp_schema.ToolCallProgress) and update.tool_call_id.endswith(":role")
    ][-1]
    assert terminal.status == "failed"
    assert "cancelled" in str(terminal.raw_output)


@pytest.mark.parametrize(("fail_at_update", "expected_hosts"), [(1, 0), (2, 1)])
@pytest.mark.asyncio
async def test_acp_update_failure_unwinds_adapter_and_shuts_down_active_host(
    tmp_path: Path,
    fail_at_update: int,
    expected_hosts: int,
) -> None:
    factory = _FakeHostFactory(
        [
            _HostScript(
                outcome=EndTurn(final_text="must not become a PACT result"),
                events=[
                    ToolCallStart(
                        tool_name="read_file",
                        tool_kind=KIND_FILESYSTEM_READ,
                        call_id="call-1",
                        args={"path": "README.md"},
                    )
                ],
            )
        ]
    )
    base = _base_settings()
    update_count = 0

    async def _send_update(_update: object) -> None:
        nonlocal update_count
        update_count += 1
        if update_count == fail_at_update:
            raise ConnectionError("outer ACP disconnected")

    adapter = InProcessChrysAdapter(
        semantic_role="worker",
        profile_name="Code",
        outer_loop=asyncio.get_running_loop(),
        campaign_id="campaign-1",
        send_update=_send_update,
        abort_event=threading.Event(),
        loaded_settings=base,
        host_factory=factory,
    )

    with (
        patch("chrys.pact.role_runner.load_settings", autospec=True, return_value=base),
        pytest.raises(RoleUpdateError, match="over ACP"),
    ):
        await asyncio.to_thread(adapter.run_turn, _request(tmp_path / f"failure-{fail_at_update}"))

    assert len(factory.hosts) == expected_hosts
    assert all(host.shutdown_called for host in factory.hosts)


def test_role_host_settings_never_route(tmp_path) -> None:
    """A role already runs inside a campaign; routing here would start another."""
    from chrys.foundation.config.settings import Settings
    from chrys.foundation.config.settings_store import LoadedSettings
    from chrys.pact.role_runner import _derive_turn_settings

    base = LoadedSettings(
        settings=Settings(routing_mode="always", memory_mcp_enabled=True, memory_writeback_on_session_end=True),
        provenance={},
    )

    derived = _derive_turn_settings(tmp_path, base)

    assert derived.settings.routing_mode == "off"
    # The delegating session deposits the campaign; a role host carries no
    # memory server to tear down and no deposit to flush at shutdown.
    assert derived.settings.memory_mcp_enabled is False
    assert derived.settings.memory_writeback_on_session_end is False


async def test_a_directory_where_the_decision_belongs_does_not_fail_the_turn(tmp_path: Path) -> None:
    """The reviewer owns that directory, so it can put anything at that path.

    Raising there was caught as a spawn failure and threw away the final text
    of a turn that had actually completed — over a misplaced `mkdir`.
    """
    factory = _FakeHostFactory([_HostScript(outcome=EndTurn(final_text="review evidence"))])
    base = _base_settings()
    adapter = _adapter(semantic_role="reviewer", base_settings=base, factory=factory, updates=[])
    workdir = tmp_path / "workdir"
    misplaced = workdir / ".pact-io" / "reviewer-decision.json"
    misplaced.mkdir(parents=True)

    with patch("chrys.pact.role_runner.load_settings", autospec=True, return_value=base):
        result = await asyncio.to_thread(
            adapter.run_turn,
            _request(workdir, role="reviewer"),
        )

    assert result.status == "completed"
    assert result.final_text == "review evidence"
    assert result.review_decision.verdict_status == "malformed"
    assert misplaced.is_dir()


# ── fenced role replies ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"schema": "pact-runtime/manager-decision-proposal/v1"}\n```',
        '```\n{"schema": "pact-runtime/manager-decision-proposal/v1"}\n```\n',
        '  ```json  \r\n{"schema": "pact-runtime/manager-decision-proposal/v1"}\r\n```  ',
    ],
)
def test_a_reply_that_is_exactly_one_fence_is_unwrapped(text: str) -> None:
    from chrys.pact.role_runner import _unfenced

    assert _unfenced(text) == '{"schema": "pact-runtime/manager-decision-proposal/v1"}'


@pytest.mark.parametrize(
    "text",
    [
        '{"schema": "bare"}',
        'Here you go:\n```json\n{"schema": "x"}\n```',
        '```json\n{"schema": "x"}\n```\nDone.',
        "prose with ``` inside ``` it",
    ],
)
def test_anything_else_is_left_alone(text: str) -> None:
    from chrys.pact.role_runner import _unfenced

    assert _unfenced(text) == text


def test_map_outcome_unfences_a_completed_reply() -> None:
    from chrys.pact.role_runner import InProcessChrysAdapter

    status, text, diagnostic = InProcessChrysAdapter._map_outcome(EndTurn(final_text='```json\n{"a": 1}\n```'))

    assert (status, text, diagnostic) == ("completed", '{"a": 1}', "")


@pytest.mark.parametrize(
    "text",
    [
        'The campaign is blocked: deps missing.\n\n```json\n{"schema": "x", "action": "request_replan"}\n```',
        'Reasoning first. {"schema": "x", "action": "request_replan"} And a closing remark.',
        '```json\n{"schema": "x", "action": "request_replan"}\n```',
    ],
)
def test_a_json_protocol_reply_is_reduced_to_its_object(text: str) -> None:
    from chrys.pact.role_runner import _protocol_payload

    assert json.loads(_protocol_payload(text)) == {"schema": "x", "action": "request_replan"}


def test_a_reply_without_any_object_is_left_for_the_repair_pass() -> None:
    from chrys.pact.role_runner import _protocol_payload

    assert _protocol_payload("Removed the stray fields; the plan is otherwise unchanged.") == (
        "Removed the stray fields; the plan is otherwise unchanged."
    )


def test_planner_and_manager_prompts_carry_the_protocol_constraints() -> None:
    from chrys.pact.role_runner import _ROLE_PROTOCOL_REMINDERS

    assert "Never delete, rename or edit an existing mission" in _ROLE_PROTOCOL_REMINDERS["planner"]
    assert "supersedes" in _ROLE_PROTOCOL_REMINDERS["planner"]
    assert "`constraints` are copied from the current plan unchanged" in _ROLE_PROTOCOL_REMINDERS["planner"]
    assert (
        "`affected_mission_ids` is exactly the set of mission ids your operations name"
        in _ROLE_PROTOCOL_REMINDERS["planner"]
    )
    assert "JSON decision object as the text of your message" in _ROLE_PROTOCOL_REMINDERS["manager"]
    assert "`selected_mission_id` belongs to the `select` action alone" in _ROLE_PROTOCOL_REMINDERS["manager"]
    assert "change ONLY what the error names" in _ROLE_PROTOCOL_REMINDERS["planner"]
    assert set(_ROLE_PROTOCOL_REMINDERS) == {"planner", "manager"}


@pytest.mark.asyncio
async def test_planner_turn_without_json_text_is_asked_once_more_in_the_same_session(tmp_path: Path) -> None:
    from chrys.pact.role_runner import _JSON_FOLLOWUP_PROMPT

    factory = _FakeHostFactory(
        [
            _HostScript(
                outcome=EndTurn(final_text=""),
                followup_outcome=EndTurn(final_text='Here it is:\n```json\n{"schema": "plan", "missions": []}\n```'),
            )
        ]
    )
    base = _base_settings()
    updates: list[object] = []
    adapter = _adapter(semantic_role="planner", base_settings=base, factory=factory, updates=updates)

    with patch("chrys.pact.role_runner.load_settings", autospec=True, return_value=base):
        result = await asyncio.to_thread(adapter.run_turn, _request(tmp_path / "planner"))

    assert result.status == "completed"
    assert json.loads(result.final_text) == {"schema": "plan", "missions": []}
    host = factory.hosts[0]
    assert len(factory.hosts) == 1
    assert host.prompts[1] == _JSON_FOLLOWUP_PROMPT
    assert host.shutdown_called


@pytest.mark.asyncio
async def test_worker_turn_without_text_is_not_asked_again(tmp_path: Path) -> None:
    factory = _FakeHostFactory(
        [_HostScript(outcome=EndTurn(final_text=""), followup_outcome=EndTurn(final_text="late"))]
    )
    base = _base_settings()
    updates: list[object] = []
    adapter = _adapter(semantic_role="worker", base_settings=base, factory=factory, updates=updates)

    with patch("chrys.pact.role_runner.load_settings", autospec=True, return_value=base):
        result = await asyncio.to_thread(adapter.run_turn, _request(tmp_path / "worker"))

    assert result.status == "output_missing"
    assert len(factory.hosts[0].prompts) == 1


def test_a_planner_format_repair_keeps_the_original_semantics() -> None:
    from chrys.pact.role_runner import _preserve_repair_semantics

    original = {
        "schema": "pact-runtime/plan-revision-proposal/v1",
        "parent_plan_revision": 2,
        "input_work_state_revision": 5,
        "reason": "m1 blocked",
        "rationale": "split it",
        "constraints": ["keep API"],
        "missions": [
            {
                "id": "m1a",
                "objective": "half",
                "target_ac_ids": ["ac-1"],
                "dependencies": [],
                "supersedes": ["m1"],
                "verification_intent": "run",
            }
        ],
        "operations": [{"op": "supersede_mission", "mission_id": "m1", "successor": "m1a"}],
        "affected_mission_ids": ["m1", "m1a"],
        "affected_ac_ids": ["ac-1"],
    }
    prompt = (
        "# Planner Structured Output Repair\n<validation-error>bad op</validation-error>\n<invalid-output>\n"
        + json.dumps(original)
        + "\n</invalid-output>"
    )
    reply = dict(
        original,
        reason="rewritten",
        missions=[],
        operations=[{"op": "supersede_mission", "mission_id": "m1", "replacement_mission_ids": ["m1a"]}],
    )

    merged = json.loads(_preserve_repair_semantics(prompt, json.dumps(reply)))

    assert merged["reason"] == "m1 blocked"
    assert merged["missions"] == original["missions"]
    assert merged["operations"] == reply["operations"]
    assert _preserve_repair_semantics("# Runtime Planner Proposal", json.dumps(reply)) == json.dumps(reply)


@pytest.mark.asyncio
async def test_worker_prompts_carry_the_staged_requirement_verbatim(tmp_path: Path) -> None:
    factory = _FakeHostFactory([_HostScript(outcome=EndTurn(final_text="done"))])
    base = _base_settings()
    updates: list[object] = []
    adapter = _adapter(semantic_role="worker", base_settings=base, factory=factory, updates=updates)
    workdir = tmp_path / "ws"
    staged = workdir / ".pact-io" / "chrys-pact" / "req1"
    staged.mkdir(parents=True)
    (staged / "requirement.md").write_text("Add `errorStack` with modes off, string, frames.\n", encoding="utf-8")
    (staged / "baseline.md").write_text(
        "Baseline pass: p1.\n### Recent commits\nabc123 add errorStack\n", encoding="utf-8"
    )

    with patch("chrys.pact.role_runner.load_settings", autospec=True, return_value=base):
        await asyncio.to_thread(adapter.run_turn, _request(workdir))

    prompt = factory.hosts[0].prompts[0]
    assert "## Authoritative requirement (verbatim)" in prompt
    assert "modes off, string, frames" in prompt
    assert "## Existing baseline implementation" in prompt
    assert "abc123 add errorStack" in prompt


def test_reviewers_judge_against_the_clarification_and_the_contract(tmp_path: Path) -> None:
    from chrys.pact.role_runner import _campaign_context

    workdir = tmp_path / "ws"
    staged = workdir / ".pact-io" / "chrys-pact" / "req1"
    staged.mkdir(parents=True)
    (staged / "requirement.md").write_text("Add `errorStack` with modes off, string, frames.\n", encoding="utf-8")
    (staged / "clarification.md").write_text(
        "# Clarified Requirement\n## Original Requirement\nAdd `errorStack`.\n## Clarification Delta\n"
        "- frames means the parsed stack array\n",
        encoding="utf-8",
    )
    (staged / "goal-contract.json").write_text(
        '{"goal": "errorStack option", "acceptance_criteria": []}', encoding="utf-8"
    )
    (staged / "baseline.md").write_text("Baseline pass: p1.\n", encoding="utf-8")

    worker = _campaign_context(workdir, "worker")
    reviewer = _campaign_context(workdir, "reviewer")

    assert "## Authoritative requirement (verbatim)" in worker
    assert "modes off, string, frames" in worker
    assert "Clarified requirement" not in worker
    assert "Goal Contract (review authority)" not in worker
    assert "Baseline pass: p1." in worker

    assert "## Clarified requirement (review authority)" in reviewer
    assert "frames means the parsed stack array" in reviewer
    assert "## Goal Contract (review authority)" in reviewer
    assert '"goal": "errorStack option"' in reviewer
    assert "## Authoritative requirement (verbatim)" not in reviewer
    assert "Baseline pass: p1." in reviewer


def test_planner_proposals_get_the_current_plan_missions_and_constraints_back(tmp_path: Path) -> None:
    from chrys.pact.role_runner import _restore_existing_missions

    workdir = tmp_path / "ws"
    campaign = workdir / ".pact" / "runtime" / "campaigns" / "c1"
    (campaign / "plan-revisions").mkdir(parents=True)
    m1 = {
        "id": "m1",
        "objective": "Thread the signature",
        "target_ac_ids": ["ac1"],
        "dependencies": [],
        "supersedes": [],
        "verification_intent": "run the unit tests",
    }
    (campaign / "plan-revisions" / "rev-0001.json").write_text(
        json.dumps({"revision": 1, "constraints": ["keep the public API"], "missions": [m1]}), encoding="utf-8"
    )
    (campaign / "work-state.json").write_text(
        json.dumps({"plan_ref": ".pact/runtime/campaigns/c1/plan-revisions/rev-0001.json"}), encoding="utf-8"
    )
    proposal = {
        "constraints": ["keep the public API", "and also this new one"],
        "missions": [
            {**m1, "objective": "Thread the signature (rewritten)", "dependencies": ["m2"]},
            {
                "id": "m2",
                "objective": "New work",
                "target_ac_ids": ["ac1"],
                "dependencies": [],
                "supersedes": [],
                "verification_intent": "tests",
            },
        ],
        "operations": [{"op": "add_mission", "mission_id": "m2"}],
    }

    restored = json.loads(_restore_existing_missions(workdir, json.dumps(proposal)))

    assert restored["constraints"] == ["keep the public API"]
    assert restored["missions"][0] == m1
    assert restored["missions"][1]["id"] == "m2"
    assert restored["operations"] == proposal["operations"]
    assert _restore_existing_missions(tmp_path / "nowhere", "not json") == "not json"


def test_a_manager_retry_on_repeated_no_progress_becomes_a_replan() -> None:
    from chrys.pact.role_runner import _replan_instead_of_retry

    prompt = 'Governance: {"trigger": "repeated_no_progress", "consecutive_no_progress": 2}'
    decision = {
        "schema": "pact-runtime/manager-decision-proposal/v1",
        "phase": "route",
        "action": "retry",
        "expected_plan_revision": 1,
        "expected_work_state_revision": 1,
        "reason": "one more try",
    }

    rewritten = json.loads(_replan_instead_of_retry(prompt, json.dumps(decision)))

    assert rewritten["action"] == "request_replan"
    assert rewritten["selected_mission_id"] is None
    assert rewritten["constraints"]
    assert rewritten["reason"].startswith("one more try")
    assert _replan_instead_of_retry("trigger: stall", json.dumps(decision)) == json.dumps(decision)
    select = {**decision, "action": "select", "selected_mission_id": "m1"}
    assert _replan_instead_of_retry(prompt, json.dumps(select)) == json.dumps(select)
