# Copyright (c) 2026 Chrys. All rights reserved.

"""ACP controller, translator, permission, and usage-chain behavior."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from acp.schema import (
    AgentMessageChunk,
    ContentToolCallContent,
    FileEditToolCallContent,
    ImageContentBlock,
    PermissionOption,
    ResourceContentBlock,
    SessionNotification,
    TerminalToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallUpdate,
    UsageUpdate,
)

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    ApprovalAutoFulfillBlocked,
    ApprovalCancelled,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalReviewed,
    AskUserResponse,
    AskUserTimedOut,
    QuestionToUser,
    SubAgentAborted,
    SubAgentCascadeAborted,
    SubAgentPaused,
    SubAgentProgress,
    SubAgentResumed,
    SubAgentRetryAttempt,
    SubAgentToolCallResult,
    SubAgentToolCallStart,
)
from chrys.foundation.text.images import MAX_IMAGE_BYTES
from chrys.foundation.tool_result_metadata import TOOL_FAILED_METADATA_KEY
from chrys.foundation.trajectory.context import TrajectoryContext
from chrys.foundation.trajectory.event_types import EventType, RetryMode, RetryReason, WaitCategory
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.orchestration.sub_agents import acp_controller as acp_module
from chrys.orchestration.sub_agents.acp_controller import (
    AcpPermissionBroker,
    AcpSubAgentController,
    AcpUpdateTranslator,
)
from chrys.orchestration.sub_agents.controller import SubAgentStatus
from chrys.orchestration.sub_agents.tools import SubAgentTools
from chrys.service.acp_client import AcpAgentSpec, AcpPromptOutcome, AcpPromptUsage
from chrys.service.acp_client import client as acp_client_module
from chrys.service.acp_client.errors import AcpConnectError, AcpRefusalError, AcpTransportError
from chrys.service.agent_middleware.control.approval import ApprovalMiddleware
from chrys.service.approval.judge import JudgeVerdict
from chrys.service.approval.policy import ApprovalMode, ApprovalPolicy
from chrys.service.approval.turn_context import TurnContextHolder, TurnContextReader
from chrys.service.profiles.agents.schema import ApprovalConfig
from chrys.service.session.runtime_metadata import SessionRuntimeMetadata
from chrys.service.trajectory.approvals import ApprovalDecider
from chrys.service.trajectory.waits import WaitOutcome
from tests.service.trajectory._fakes import CancelAckSink, FakeSink, make_context
from tests.support.waiting import wait_for


async def _capture(bus: EventBus, event_type: type) -> list[Any]:
    events: list[Any] = []

    async def append(event: Any) -> None:
        events.append(event)

    await bus.subscribe(event_type, append)
    return events


def _notification(update: Any) -> SessionNotification:
    return SessionNotification(sessionId="remote", update=update)


@pytest.mark.asyncio
async def test_translator_synthesizes_progress_before_start_and_prefixes_attempt() -> None:
    bus = EventBus()
    starts = await _capture(bus, SubAgentToolCallStart)
    results = await _capture(bus, SubAgentToolCallResult)
    translator = AcpUpdateTranslator(
        event_bus=bus,
        session_id="parent",
        agent_name="External",
        invocation_id="inv",
        attempt=2,
    )
    update = ToolCallProgress(
        sessionUpdate="tool_call_update",
        toolCallId="remote-call",
        title="Read",
        kind="read",
        rawInput="README.md",
        rawOutput={"ok": True},
        status="failed",
    )
    await translator.put(1, _notification(update))
    await translator.put(2, _notification(update))

    assert starts[0].call_id == "a2:remote-call"
    assert starts[0].tool_kind == "filesystem.read"
    assert starts[0].args == {"input": "README.md"}
    assert len(results) == 1
    assert results[0].metadata == {TOOL_FAILED_METADATA_KEY: True}


@pytest.mark.asyncio
async def test_translator_forwards_only_explicit_remote_hosted_metadata_without_wrapping() -> None:
    bus = EventBus()
    starts = await _capture(bus, SubAgentToolCallStart)
    results = await _capture(bus, SubAgentToolCallResult)
    translator = AcpUpdateTranslator(
        event_bus=bus,
        session_id="parent",
        agent_name="External",
        invocation_id="inv",
        attempt=1,
    )
    await translator.put(
        1,
        _notification(
            ToolCallProgress(
                sessionUpdate="tool_call_update",
                toolCallId="remote-search",
                title="Search",
                kind="search",
                rawOutput="found",
                status="completed",
            )
        ),
    )
    await translator.put(
        2,
        _notification(
            ToolCallProgress(
                sessionUpdate="tool_call_update",
                toolCallId="remote-hosted",
                title="Provider search",
                kind="search",
                rawOutput="found",
                status="completed",
                _meta={
                    "chrys": {
                        "provider_hosted": True,
                        "hosted_family": "search",
                        "provider": "openai",
                        "provider_item_type": "web_search_call",
                        "provider_call_id": "provider-search",
                        "provider_status": "completed",
                    }
                },
            )
        ),
    )

    assert len(results) == 2
    start_by_id = {event.call_id: event for event in starts}
    assert set(start_by_id) == {"a1:remote-search", "a1:remote-hosted"}
    assert start_by_id["a1:remote-search"].provider_hosted is False
    assert start_by_id["a1:remote-search"].hosted_family == ""
    assert start_by_id["a1:remote-hosted"].provider_hosted is True
    assert start_by_id["a1:remote-hosted"].hosted_family == "search"
    assert start_by_id["a1:remote-hosted"].provider_call_id == "provider-search"
    assert results[1].provider_item_type == "web_search_call"


@pytest.mark.asyncio
async def test_translator_extracts_every_tool_content_variant_and_flushes_interrupted() -> None:
    bus = EventBus()
    results = await _capture(bus, SubAgentToolCallResult)
    translator = AcpUpdateTranslator(
        event_bus=bus,
        session_id=None,
        agent_name="External",
        invocation_id="inv",
        attempt=1,
    )
    await translator.put(
        1,
        _notification(
            ToolCallProgress(
                sessionUpdate="tool_call_update",
                toolCallId="content",
                title="Mixed",
                content=[
                    ContentToolCallContent(
                        type="content",
                        content=TextContentBlock(type="text", text="hello"),
                    ),
                    ContentToolCallContent(
                        type="content",
                        content=ResourceContentBlock(type="resource_link", uri="file:///a", name="a"),
                    ),
                    FileEditToolCallContent(type="diff", path="a.py", newText="new"),
                    TerminalToolCallContent(type="terminal", terminalId="term-1"),
                ],
                status="completed",
            )
        ),
    )
    await translator.put(
        2,
        _notification(
            ToolCallProgress(
                sessionUpdate="tool_call_update",
                toolCallId="running",
                title="Running",
                status="in_progress",
            )
        ),
    )
    await translator.flush_interrupted()
    assert "hello" in results[0].result
    assert "[resource: file:///a]" in results[0].result
    assert "edited a.py" in results[0].result
    assert "[terminal term-1]" in results[0].result
    assert results[1].result == "(interrupted)"
    assert results[1].metadata[TOOL_FAILED_METADATA_KEY] is True


@pytest.mark.asyncio
async def test_translator_accepts_maximum_supported_image_size() -> None:
    bus = EventBus()
    results = await _capture(bus, SubAgentToolCallResult)
    translator = AcpUpdateTranslator(
        event_bus=bus,
        session_id=None,
        agent_name="External",
        invocation_id="inv",
        attempt=1,
    )
    encoded = base64.b64encode(b"x" * MAX_IMAGE_BYTES).decode("ascii")

    notification = _notification(
        ToolCallProgress(
            sessionUpdate="tool_call_update",
            toolCallId="image",
            title="Image",
            content=[
                ContentToolCallContent(
                    type="content",
                    content=ImageContentBlock(type="image", data=encoded, mimeType="image/png"),
                )
            ],
            status="completed",
        )
    )
    params = notification.model_dump(mode="json", by_alias=True)
    observer_buffer = acp_client_module._ObserverBuffer()

    observer_buffer.put_update(params)
    buffered = await observer_buffer.get()
    assert isinstance(buffered, dict)
    await translator.put(1, SessionNotification.model_validate(buffered))

    assert len(results) == 1
    assert len(results[0].image_contents) == 1
    assert results[0].image_contents[0].uri.endswith(encoded)


@pytest.mark.asyncio
async def test_translator_charges_repeated_image_snapshot_once() -> None:
    translator = AcpUpdateTranslator(
        event_bus=None,
        session_id=None,
        agent_name="External",
        invocation_id="inv",
        attempt=1,
    )
    encoded = base64.b64encode(b"x" * MAX_IMAGE_BYTES).decode("ascii")

    for seq, status in enumerate(("in_progress", "completed"), start=1):
        await translator.put(
            seq,
            _notification(
                ToolCallProgress(
                    sessionUpdate="tool_call_update",
                    toolCallId="image",
                    title="Image",
                    content=[
                        ContentToolCallContent(
                            type="content",
                            content=ImageContentBlock(type="image", data=encoded, mimeType="image/png"),
                        )
                    ],
                    status=status,
                )
            ),
        )

    assert translator.completed_count == 1


@pytest.mark.asyncio
async def test_translator_replaces_partial_image_charge_with_final_snapshot() -> None:
    translator = AcpUpdateTranslator(
        event_bus=None,
        session_id=None,
        agent_name="External",
        invocation_id="inv",
        attempt=1,
    )
    snapshots = (
        (base64.b64encode(b"p" * (1024 * 1024)).decode("ascii"), "in_progress"),
        (base64.b64encode(b"f" * MAX_IMAGE_BYTES).decode("ascii"), "completed"),
    )

    for seq, (encoded, status) in enumerate(snapshots, start=1):
        await translator.put(
            seq,
            _notification(
                ToolCallProgress(
                    sessionUpdate="tool_call_update",
                    toolCallId="image",
                    title="Image",
                    content=[
                        ContentToolCallContent(
                            type="content",
                            content=ImageContentBlock(type="image", data=encoded, mimeType="image/png"),
                        )
                    ],
                    status=status,
                )
            ),
        )

    assert translator.completed_count == 1


@pytest.mark.asyncio
async def test_translator_charges_images_retained_by_separate_calls() -> None:
    translator = AcpUpdateTranslator(
        event_bus=None,
        session_id=None,
        agent_name="External",
        invocation_id="inv",
        attempt=1,
    )

    for seq, byte in enumerate((b"a", b"b"), start=1):
        encoded = base64.b64encode(byte * MAX_IMAGE_BYTES).decode("ascii")
        notification = _notification(
            ToolCallProgress(
                sessionUpdate="tool_call_update",
                toolCallId=f"image-{seq}",
                title="Image",
                content=[
                    ContentToolCallContent(
                        type="content",
                        content=ImageContentBlock(type="image", data=encoded, mimeType="image/png"),
                    )
                ],
                status="in_progress",
            )
        )
        if seq == 1:
            await translator.put(seq, notification)
        else:
            with pytest.raises(AcpTransportError, match="translated update budget"):
                await translator.put(seq, notification)


@pytest.mark.asyncio
async def test_translator_message_boundaries_result_modes_and_usage_gauge() -> None:
    bus = EventBus()
    progress = await _capture(bus, SubAgentProgress)
    translator = AcpUpdateTranslator(
        event_bus=bus,
        session_id=None,
        agent_name="External",
        invocation_id="inv",
        attempt=1,
        result_mode="transcript",
        spend_before=21,
        unreported_before=1,
    )
    for seq, (message_id, text) in enumerate((("a", "first"), ("a", " chunk"), ("b", "second")), start=1):
        await translator.put(
            seq,
            _notification(
                AgentMessageChunk(
                    sessionUpdate="agent_message_chunk",
                    messageId=message_id,
                    content=TextContentBlock(type="text", text=text),
                )
            ),
        )
    await translator.put(
        4,
        _notification(UsageUpdate(sessionUpdate="usage_update", used=120, size=1_000)),
    )
    assert translator.result_text() == "first chunk\n\nsecond"
    assert progress[-1].total_tokens == 120
    assert progress[-1].total_usage_tokens == 21
    assert progress[-1].usage_unreported_attempts == 1
    assert translator.latest_context_size == 1_000


def _broker(
    bus: EventBus,
    mode: list[ApprovalMode],
    *,
    timeout: float | None = 1,
    allow_user_interaction: bool = True,
    trajectory_context: TrajectoryContext | None = None,
    trajectory_boundary_operation_id: str | None = None,
) -> AcpPermissionBroker:
    context = TurnContextHolder()
    context.replace(["original prompt"])
    return AcpPermissionBroker(
        event_bus=bus,
        session_id="parent",
        caller_name="External",
        mode_getter=lambda: mode[0],
        turn_context=context,
        workspace_roots=["/workspace"],
        workspace_cwd="/workspace",
        approval_judge=None,
        ask_user_timeout_seconds=timeout,
        allow_user_interaction=allow_user_interaction,
        trajectory_context=trajectory_context,
        trajectory_boundary_operation_id=trajectory_boundary_operation_id,
    )


@pytest.mark.asyncio
async def test_permission_polarity_preflight_runs_before_bypass() -> None:
    bus = EventBus()
    requests = await _capture(bus, ApprovalRequest)
    broker = _broker(bus, [ApprovalMode.BYPASS])
    tool_call = ToolCallUpdate(toolCallId="call", title="Spoof", kind="execute", rawInput={"command": "x"})
    reject = PermissionOption(optionId="reject", name="Reject", kind="reject_once")

    decision = await broker.on_permission_request(tool_call, [reject])
    empty_decision = await broker.on_permission_request(tool_call, [])

    assert decision.action == "deny"
    assert decision.option_id == "reject"
    assert empty_decision.action == "cancelled"
    assert requests == []


@pytest.mark.asyncio
async def test_permission_request_uses_spoof_proof_presentation_and_user_decision_wins() -> None:
    bus = EventBus()
    broker = _broker(bus, [ApprovalMode.MANUAL])
    requests = await _capture(bus, ApprovalRequest)

    async def respond(event: ApprovalRequest) -> None:
        await bus.publish(
            ApprovalResponse(
                request_id=event.request_id,
                approved=False,
                reason="no",
                session_id=event.session_id,
            )
        )

    await bus.subscribe(ApprovalRequest, respond)
    decision = await broker.on_permission_request(
        ToolCallUpdate(toolCallId="call", title="write_file", kind="edit", rawInput={"path": "x"}),
        [
            PermissionOption(optionId="allow", name="Allow", kind="allow_once"),
            PermissionOption(optionId="reject", name="Reject", kind="reject_once"),
        ],
    )
    assert decision.action == "deny"
    assert requests[0].tool_name == "acp:write_file"
    assert requests[0].tool_kind == ""
    assert requests[0].presentation_kind == "filesystem.write"
    assert requests[0].user_message == "original prompt"

    # A kind chrys does not recognize still yields a non-empty presentation
    # hint so the dialog can tell bridged requests from local ones.
    await broker.on_permission_request(
        ToolCallUpdate(toolCallId="call2", title="mystery", kind="think", rawInput={}),
        [
            PermissionOption(optionId="allow", name="Allow", kind="allow_once"),
            PermissionOption(optionId="reject", name="Reject", kind="reject_once"),
        ],
    )
    assert requests[1].presentation_kind == "remote"


@pytest.mark.asyncio
async def test_auto_permission_reuses_judge_race_with_separate_judge_and_presentation_fields() -> None:
    bus = EventBus()
    context = TurnContextHolder()
    context.replace(["review this"])

    class Judge:
        called_with: dict[str, Any] | None = None

        async def evaluate(self, **kwargs: Any) -> JudgeVerdict:
            self.called_with = kwargs
            return JudgeVerdict(approved=True, reason="safe")

    judge = Judge()
    broker = AcpPermissionBroker(
        event_bus=bus,
        session_id="parent",
        caller_name="External",
        mode_getter=lambda: ApprovalMode.AUTO,
        turn_context=context,
        workspace_roots=["/workspace"],
        workspace_cwd="/workspace",
        approval_judge=judge,
        ask_user_timeout_seconds=1,
    )
    requests = await _capture(bus, ApprovalRequest)

    async def reject_after_review(event: ApprovalReviewed) -> None:
        await bus.publish(ApprovalAutoFulfillBlocked(request_id=event.request_id))
        await bus.publish(
            ApprovalResponse(
                request_id=event.request_id,
                approved=False,
                reason="user wins",
                session_id=event.session_id,
            )
        )

    await bus.subscribe(ApprovalReviewed, reject_after_review)
    decision = await broker.on_permission_request(
        ToolCallUpdate(
            toolCallId="call",
            title="outer intent",
            kind="other",
            rawInput={"command": "echo ok"},
            _meta={"chrys": {"tool_name": "zsh", "tool_kind": "shell"}},
        ),
        [
            PermissionOption(optionId="allow", name="Allow", kind="allow_once"),
            PermissionOption(optionId="deny", name="Deny", kind="reject_once"),
        ],
    )
    assert decision.action == "deny"
    assert requests[0].tool_name == "acp:outer intent"
    assert requests[0].tool_kind == ""
    # Presentation rides the _meta-enhanced judge kind, never tool_kind.
    assert requests[0].presentation_kind == "shell"
    assert judge.called_with is not None
    assert judge.called_with["tool_name"] == "zsh"
    assert judge.called_with["tool_kind"] == "shell"
    assert judge.called_with["args"] == {"command": "echo ok"}
    assert judge.called_with["log_dir"] is None


@pytest.mark.asyncio
async def test_auto_permission_records_user_when_user_beats_enabled_judge() -> None:
    bus = EventBus()
    sink = FakeSink()
    context = TurnContextHolder()
    context.replace(["review this"])

    class Judge:
        async def evaluate(self, **_kwargs: Any) -> JudgeVerdict:
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    broker = AcpPermissionBroker(
        event_bus=bus,
        session_id="parent",
        caller_name="External",
        mode_getter=lambda: ApprovalMode.AUTO,
        turn_context=context,
        workspace_roots=["/workspace"],
        workspace_cwd="/workspace",
        approval_judge=Judge(),
        ask_user_timeout_seconds=1,
        trajectory_context=make_context(sink),
        trajectory_boundary_operation_id=new_analytics_id(),
    )

    async def approve(event: ApprovalRequest) -> None:
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=True, session_id=event.session_id))

    await bus.subscribe(ApprovalRequest, approve)
    decision = await broker.on_permission_request(
        ToolCallUpdate(toolCallId="call", title="Run", rawInput={}),
        [PermissionOption(optionId="allow", name="Allow", kind="allow_once")],
    )

    assert decision.action == "allow"
    assert sink.only(EventType.APPROVAL_RESOLVED).payload["decider"] == ApprovalDecider.USER


@pytest.mark.asyncio
async def test_judge_and_dialog_receive_untruncated_raw_arguments() -> None:
    """A destructive suffix beyond the preview cap must stay visible (§9.2).

    The judge and the published approval both authorize what the remote will
    execute; a truncating canonicalization here would let AUTO approve the
    benign prefix of a command whose tail it never saw.
    """
    bus = EventBus()
    context = TurnContextHolder()
    context.replace(["review this"])

    class Judge:
        called_with: dict[str, Any] | None = None

        async def evaluate(self, **kwargs: Any) -> JudgeVerdict:
            self.called_with = kwargs
            return JudgeVerdict(approved=True, reason="safe")

    judge = Judge()
    broker = AcpPermissionBroker(
        event_bus=bus,
        session_id="parent",
        caller_name="External",
        mode_getter=lambda: ApprovalMode.AUTO,
        turn_context=context,
        workspace_roots=["/workspace"],
        workspace_cwd="/workspace",
        approval_judge=judge,
        ask_user_timeout_seconds=1,
    )
    requests = await _capture(bus, ApprovalRequest)
    command = "echo " + "x" * 5_000 + " && rm -rf target"
    decision = await broker.on_permission_request(
        ToolCallUpdate(toolCallId="call", title="run", kind="execute", rawInput={"command": command}),
        [PermissionOption(optionId="allow", name="Allow", kind="allow_once")],
    )
    assert decision.action == "allow"
    assert judge.called_with is not None
    assert judge.called_with["args"] == {"command": command}
    assert requests[0].args == {"command": command}


@pytest.mark.asyncio
async def test_standalone_permission_requests_enter_the_audit_trail() -> None:
    """Permission requests with no surrounding ToolCallStart still get audited.

    ACP permits ``session/request_permission`` before any tool-call update;
    judge logging is deliberately ``log_dir=None``, so the translator audit
    ring is the only durable home for the request's rawInput and outcome.
    """
    bus = EventBus()
    broker = _broker(bus, [ApprovalMode.MANUAL])
    translator = AcpUpdateTranslator(
        event_bus=None,
        session_id="parent",
        agent_name="External",
        invocation_id="inv",
        attempt=1,
    )
    broker.set_translator(translator)

    async def respond(event: ApprovalRequest) -> None:
        await bus.publish(
            ApprovalResponse(
                request_id=event.request_id,
                approved=False,
                reason="no",
                session_id=event.session_id,
            )
        )

    await bus.subscribe(ApprovalRequest, respond)
    decision = await broker.on_permission_request(
        ToolCallUpdate(toolCallId="call", title="write_file", kind="edit", rawInput={"path": "secrets.txt"}),
        [
            PermissionOption(optionId="allow", name="Allow", kind="allow_once"),
            PermissionOption(optionId="reject", name="Reject", kind="reject_once"),
        ],
    )
    assert decision.action == "deny"
    entries = translator.translated_updates
    assert entries
    recorded = entries[-1]["update"]
    assert recorded["sessionUpdate"] == "permission_request"
    assert recorded["title"] == "write_file"
    assert recorded["kind"] == "filesystem.write"
    assert recorded["rawInput"] == {"path": "secrets.txt"}
    assert recorded["outcome"] == "denied"


@pytest.mark.asyncio
async def test_permission_abandonment_and_ask_user_timeout_publish_clear_events() -> None:
    bus = EventBus()
    broker = _broker(bus, [ApprovalMode.MANUAL], timeout=0.01)
    cancelled = await _capture(bus, ApprovalCancelled)
    timed_out = await _capture(bus, AskUserTimedOut)
    questions = await _capture(bus, QuestionToUser)
    allow = PermissionOption(optionId="allow", name="Allow", kind="allow_once")
    permission_task = asyncio.create_task(
        broker.on_permission_request(
            ToolCallUpdate(toolCallId="call", title="Run", rawInput={}),
            [allow],
        )
    )
    await wait_for(lambda: bool(broker._permission_waits), description="permission wait")
    await broker.cancel_pending_waits()
    assert (await permission_task).action == "cancelled"
    assert cancelled

    result = await broker.on_ext_method(
        "chrys/request_input",
        {
            "sessionId": "remote",
            "requestId": "remote-request",
            "question": "Continue?",
            "options": ["yes", "no"],
            "callerName": "Remote",
        },
    )
    assert result == {"text": ""}
    assert questions
    assert timed_out


@pytest.mark.asyncio
async def test_acp_permission_and_input_waits_target_the_boundary() -> None:
    bus = EventBus()
    sink = FakeSink()
    boundary = new_analytics_id()
    broker = _broker(
        bus,
        [ApprovalMode.MANUAL],
        trajectory_context=make_context(sink),
        trajectory_boundary_operation_id=boundary,
    )

    async def approve(event: ApprovalRequest) -> None:
        await bus.publish(ApprovalResponse(request_id=event.request_id, approved=True, session_id=event.session_id))

    async def answer(event: QuestionToUser) -> None:
        await bus.publish(AskUserResponse(request_id=event.request_id, text="yes", session_id=event.session_id))

    await bus.subscribe(ApprovalRequest, approve)
    await bus.subscribe(QuestionToUser, answer)
    decision = await broker.on_permission_request(
        ToolCallUpdate(toolCallId="call", title="Run", rawInput={}),
        [PermissionOption(optionId="allow", name="Allow", kind="allow_once")],
    )
    response = await broker.on_ext_method(
        "chrys/request_input",
        {
            "sessionId": "remote",
            "requestId": "remote-request",
            "question": "Continue?",
            "options": ["yes", "no"],
        },
    )

    assert decision.action == "allow"
    assert response == {"text": "yes"}
    approval_events = [
        event
        for event in sink.drafts
        if event.event_type in {EventType.APPROVAL_REQUESTED, EventType.APPROVAL_RESOLVED}
    ]
    assert len(approval_events) == 2
    assert all(event.parent_operation_id == boundary for event in approval_events)
    assert all(event.payload["target_operation_id"] == boundary for event in approval_events)
    assert all(event.operation_id == event.payload["approval_request_id"] for event in approval_events)
    assert all(event.operation_id != event.parent_operation_id for event in approval_events)
    wait_events = [
        event for event in sink.drafts if event.event_type in {EventType.WAIT_STARTED, EventType.WAIT_FINISHED}
    ]
    assert len(wait_events) == 2
    assert all(event.payload["category"] == WaitCategory.USER_INPUT for event in wait_events)
    assert all(event.parent_operation_id == boundary for event in wait_events)
    sink.assert_operations_settled()


@pytest.mark.asyncio
async def test_permission_start_ack_cancellation_clears_registered_future() -> None:
    bus = EventBus()
    sink = CancelAckSink(at=1)
    broker = _broker(
        bus,
        [ApprovalMode.MANUAL],
        trajectory_context=make_context(sink),
        trajectory_boundary_operation_id=new_analytics_id(),
    )

    decision = await broker.on_permission_request(
        ToolCallUpdate(toolCallId="call", title="Run", rawInput={}),
        [PermissionOption(optionId="allow", name="Allow", kind="allow_once")],
    )

    assert decision.action == "cancelled"
    assert broker._permission_waits == {}
    assert sink.event_types == [EventType.APPROVAL_REQUESTED, EventType.APPROVAL_RESOLVED]
    sink.assert_operations_settled()


@pytest.mark.asyncio
async def test_input_start_ack_cancellation_clears_registered_future() -> None:
    bus = EventBus()
    sink = CancelAckSink(at=1)
    broker = _broker(
        bus,
        [ApprovalMode.MANUAL],
        trajectory_context=make_context(sink),
        trajectory_boundary_operation_id=new_analytics_id(),
    )

    response = await broker.on_ext_method(
        "chrys/request_input",
        {
            "sessionId": "remote",
            "requestId": "remote-request",
            "question": "Continue?",
            "options": ["yes", "no"],
        },
    )

    assert response == {"text": ""}
    assert broker._ask_waits == {}
    assert sink.event_types == [EventType.WAIT_STARTED, EventType.WAIT_FINISHED]
    sink.assert_operations_settled()


@pytest.mark.asyncio
async def test_permission_request_publish_cancellation_closes_opened_dialog() -> None:
    bus = EventBus()
    broker = _broker(bus, [ApprovalMode.MANUAL])
    opened = await _capture(bus, ApprovalRequest)
    cancelled = await _capture(bus, ApprovalCancelled)
    publication_blocked = asyncio.Event()
    never = asyncio.Event()

    async def block_later_handler(_event: ApprovalRequest) -> None:
        publication_blocked.set()
        await never.wait()

    await bus.subscribe(ApprovalRequest, block_later_handler)
    task = asyncio.create_task(
        broker.on_permission_request(
            ToolCallUpdate(toolCallId="call", title="Run", rawInput={}),
            [PermissionOption(optionId="allow", name="Allow", kind="allow_once")],
        )
    )
    await publication_blocked.wait()
    task.cancel()
    decision = await task

    assert decision.action == "cancelled"
    assert len(opened) == 1
    assert [event.request_id for event in cancelled] == [opened[0].request_id]
    assert broker._permission_waits == {}


@pytest.mark.asyncio
async def test_question_publish_cancellation_closes_opened_dialog() -> None:
    bus = EventBus()
    broker = _broker(bus, [ApprovalMode.MANUAL], timeout=None)
    opened = await _capture(bus, QuestionToUser)
    timed_out = await _capture(bus, AskUserTimedOut)
    publication_blocked = asyncio.Event()
    never = asyncio.Event()

    async def block_later_handler(_event: QuestionToUser) -> None:
        publication_blocked.set()
        await never.wait()

    await bus.subscribe(QuestionToUser, block_later_handler)
    task = asyncio.create_task(
        broker.on_ext_method(
            "chrys/request_input",
            {
                "sessionId": "remote",
                "requestId": "remote-request",
                "question": "Continue?",
                "options": [],
            },
        )
    )
    await publication_blocked.wait()
    task.cancel()
    response = await task

    assert response == {"text": ""}
    assert len(opened) == 1
    assert [event.request_id for event in timed_out] == [opened[0].request_id]
    assert broker._ask_waits == {}


@pytest.mark.asyncio
async def test_permission_notification_cancellation_cannot_leave_approval_open() -> None:
    bus = EventBus()
    sink = FakeSink()
    broker = _broker(
        bus,
        [ApprovalMode.MANUAL],
        trajectory_context=make_context(sink),
        trajectory_boundary_operation_id=new_analytics_id(),
    )

    async def cancel_notification(_event: ApprovalCancelled) -> None:
        raise asyncio.CancelledError

    await bus.subscribe(ApprovalCancelled, cancel_notification)
    task = asyncio.create_task(
        broker.on_permission_request(
            ToolCallUpdate(toolCallId="call", title="Run", rawInput={}),
            [PermissionOption(optionId="allow", name="Allow", kind="allow_once")],
        )
    )
    await wait_for(lambda: bool(broker._permission_waits), description="permission wait")
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert sink.only(EventType.APPROVAL_RESOLVED).payload["decider"] == ApprovalDecider.NONE
    sink.assert_operations_settled()


@pytest.mark.asyncio
async def test_ask_timeout_notification_cancellation_cannot_leave_wait_open() -> None:
    bus = EventBus()
    sink = FakeSink()
    broker = _broker(
        bus,
        [ApprovalMode.MANUAL],
        timeout=0,
        trajectory_context=make_context(sink),
        trajectory_boundary_operation_id=new_analytics_id(),
    )

    async def cancel_notification(_event: AskUserTimedOut) -> None:
        raise asyncio.CancelledError

    await bus.subscribe(AskUserTimedOut, cancel_notification)
    with pytest.raises(asyncio.CancelledError):
        await broker.on_ext_method(
            "chrys/request_input",
            {
                "sessionId": "remote",
                "requestId": "remote-request",
                "question": "Continue?",
                "options": [],
            },
        )

    assert sink.only(EventType.WAIT_FINISHED).payload["outcome"] == WaitOutcome.TIMED_OUT
    sink.assert_operations_settled()


@pytest.mark.asyncio
async def test_ask_cancel_notification_cancellation_cannot_leave_wait_open() -> None:
    bus = EventBus()
    sink = FakeSink()
    broker = _broker(
        bus,
        [ApprovalMode.MANUAL],
        timeout=None,
        trajectory_context=make_context(sink),
        trajectory_boundary_operation_id=new_analytics_id(),
    )

    async def cancel_notification(_event: AskUserTimedOut) -> None:
        raise asyncio.CancelledError

    await bus.subscribe(AskUserTimedOut, cancel_notification)
    task = asyncio.create_task(
        broker.on_ext_method(
            "chrys/request_input",
            {
                "sessionId": "remote",
                "requestId": "remote-request",
                "question": "Continue?",
                "options": [],
            },
        )
    )
    await wait_for(lambda: bool(broker._ask_waits), description="ask-user wait")
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert sink.only(EventType.WAIT_FINISHED).payload["outcome"] == WaitOutcome.CANCELLED
    sink.assert_operations_settled()


@pytest.mark.asyncio
async def test_cascade_abort_denies_permission_future_already_settled_to_allow() -> None:
    """A settled "allow" that races ahead of the backgrounded cancel must not win.

    cascade_abort latches broker.mark_aborted() synchronously; a permission
    whose ApprovalResponse future was already resolved to approved — and whose
    resolving continuation is queued ahead of cancel_pending_waits — must still
    resolve to cancelled, else the remote runs a tool after the interrupt.
    """
    bus = EventBus()
    broker = _broker(bus, [ApprovalMode.MANUAL])
    allow = PermissionOption(optionId="allow", name="Allow", kind="allow_once")
    permission_task = asyncio.create_task(
        broker.on_permission_request(
            ToolCallUpdate(toolCallId="call", title="Run", rawInput={}),
            [allow],
        )
    )
    await wait_for(lambda: bool(broker._permission_waits), description="permission wait")
    request_id = next(iter(broker._permission_waits))

    # The user's approval settles the future BEFORE the abort lands...
    await bus.publish(ApprovalResponse(request_id=request_id, approved=True, session_id="parent"))
    # ...but the abort latches before the awaiter's continuation resumes.
    broker.mark_aborted()

    assert (await permission_task).action == "cancelled"


@pytest.mark.asyncio
async def test_broker_refuses_new_permission_after_abort_latched() -> None:
    """Once aborted, even a BYPASS-mode auto-allow must be refused, unpublished."""
    bus = EventBus()
    requests = await _capture(bus, ApprovalRequest)
    broker = _broker(bus, [ApprovalMode.BYPASS])
    broker.mark_aborted()
    allow = PermissionOption(optionId="allow", name="Allow", kind="allow_once")
    decision = await broker.on_permission_request(
        ToolCallUpdate(toolCallId="call", title="Run", rawInput={}),
        [allow],
    )
    assert decision.action == "cancelled"
    assert requests == []


@pytest.mark.asyncio
async def test_ask_user_after_abort_returns_empty_without_publishing() -> None:
    """Once aborted, a fresh chrys/request_input must not open a QuestionToUser
    (symmetric with the permission entry-guard) — the remote gets no dialog."""
    bus = EventBus()
    questions = await _capture(bus, QuestionToUser)
    # Finite timeout so that if the entry-guard regresses the call still returns
    # (via timeout) instead of hanging — the published-question assertion is what
    # actually pins the invariant.
    broker = _broker(bus, [ApprovalMode.MANUAL], timeout=0.1)
    broker.mark_aborted()
    result = await broker.on_ext_method(
        "chrys/request_input",
        {
            "sessionId": "remote",
            "requestId": "remote-request",
            "question": "Continue?",
            "options": [],
            "callerName": "Remote",
        },
    )
    assert result == {"text": ""}
    assert questions == []


@pytest.mark.asyncio
async def test_ask_user_is_suppressed_when_owner_is_noninteractive() -> None:
    bus = EventBus()
    questions = await _capture(bus, QuestionToUser)
    broker = _broker(
        bus,
        [ApprovalMode.MANUAL],
        timeout=None,
        allow_user_interaction=False,
    )

    result = await broker.on_ext_method(
        "chrys/request_input",
        {
            "sessionId": "remote",
            "requestId": "remote-request",
            "question": "Continue?",
            "options": ["yes", "no"],
            "callerName": "Remote",
        },
    )

    assert result == {"text": ""}
    assert questions == []
    assert broker._ask_waits == {}


@pytest.mark.asyncio
async def test_ask_user_answer_dropped_when_abort_latches_before_resume() -> None:
    """An answer future already settled to a real text — whose continuation races
    ahead of cancel_pending_waits — must yield empty after cascade, not the
    answer, or the remote proceeds past the global interrupt."""
    bus = EventBus()
    broker = _broker(bus, [ApprovalMode.MANUAL], timeout=None)
    ask_task = asyncio.create_task(
        broker.on_ext_method(
            "chrys/request_input",
            {
                "sessionId": "remote",
                "requestId": "remote-request",
                "question": "Continue?",
                "options": [],
                "callerName": "Remote",
            },
        )
    )
    await wait_for(lambda: bool(broker._ask_waits), description="ask wait")
    request_id = next(iter(broker._ask_waits))

    # The answer settles the future BEFORE the abort lands...
    await bus.publish(AskUserResponse(request_id=request_id, text="proceed"))
    # ...but the abort latches before the awaiter's continuation resumes.
    broker.mark_aborted()

    assert await ask_task == {"text": ""}


@pytest.mark.asyncio
async def test_ask_user_none_timeout_preserves_wait_until_response() -> None:
    bus = EventBus()
    broker = _broker(bus, [ApprovalMode.MANUAL], timeout=None)

    async def respond(event: QuestionToUser) -> None:
        await bus.publish(AskUserResponse(request_id=event.request_id, text="yes"))

    await bus.subscribe(QuestionToUser, respond)
    result = await broker.on_ext_method(
        "chrys/request_input",
        {
            "sessionId": "remote",
            "requestId": "remote-request",
            "question": "Continue?",
            "options": [],
            "callerName": "Remote",
        },
    )
    assert result == {"text": "yes"}


class _FakeClient:
    scripts: ClassVar[list[dict[str, Any]]] = []
    force_closes = 0

    def __init__(self, spec: AcpAgentSpec, callbacks: Any, **kwargs: Any) -> None:
        self.spec = spec
        self.script = self.scripts.pop(0)
        self.callbacks = callbacks
        # Mirrors the real client: flips only once session/new is sent, so
        # scripted connect/open failures stay in the stateless retry window.
        self.stateful_phase_started = False

    async def connect(self) -> None:
        outcome = self.script.get("connect")
        if isinstance(outcome, BaseException):
            raise outcome

    async def open_session(self) -> Any:
        outcome = self.script.get("open")
        if isinstance(outcome, BaseException):
            raise outcome
        self.stateful_phase_started = True
        return SimpleNamespace(session_id="remote", agent_info=None)

    async def prompt(self, _prompt: str) -> AcpPromptOutcome:
        outcome = self.script["prompt"]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def cancel(self) -> None:
        return

    async def force_close(self) -> None:
        type(self).force_closes += 1


def _spec(tmp_path: Path) -> AcpAgentSpec:
    return AcpAgentSpec(command="stub", cwd=str(tmp_path), stderr_log_path=tmp_path / "stderr.log")


@pytest.mark.asyncio
async def test_connect_retry_cancelled_during_ui_publish_does_not_open_backoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bus = EventBus()
    sink = FakeSink()
    boundary = new_analytics_id()
    _FakeClient.force_closes = 0
    _FakeClient.scripts = [{"connect": AcpConnectError("temporary")}]
    monkeypatch.setattr(acp_module, "AcpAgentClient", _FakeClient)

    async def cancel_publish(_event: SubAgentRetryAttempt) -> None:
        raise asyncio.CancelledError

    await bus.subscribe(SubAgentRetryAttempt, cancel_publish)
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
        trajectory_context=make_context(sink),
        trajectory_boundary_operation_id=boundary,
    )

    with pytest.raises(asyncio.CancelledError):
        await controller.run()

    assert sink.event_types == []
    assert _FakeClient.force_closes == 1
    sink.assert_operations_settled()


@pytest.mark.asyncio
async def test_connect_retry_cancelled_during_schedule_ack_does_not_start_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bus = EventBus()
    sink = CancelAckSink(at=1)
    _FakeClient.force_closes = 0
    _FakeClient.scripts = [{"connect": AcpConnectError("temporary")}]
    monkeypatch.setattr(acp_module, "AcpAgentClient", _FakeClient)
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
        trajectory_context=make_context(sink),
        trajectory_boundary_operation_id=new_analytics_id(),
    )

    with pytest.raises(asyncio.CancelledError):
        await controller.run()

    assert sink.event_types == [EventType.RETRY_SCHEDULED]
    scheduled = sink.only(EventType.RETRY_SCHEDULED)
    assert scheduled.payload["reason_code"] == RetryReason.CONNECTION
    assert scheduled.payload["retry_mode"] == RetryMode.CONNECTION
    assert _FakeClient.force_closes == 1


@pytest.mark.asyncio
async def test_controller_retries_only_connect_and_accounts_authoritative_usage_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bus = EventBus()
    retries = await _capture(bus, SubAgentRetryAttempt)
    _FakeClient.force_closes = 0
    _FakeClient.scripts = [
        {"connect": AcpConnectError("temporary")},
        {
            "prompt": AcpPromptOutcome(
                stop_reason="end_turn",
                usage=AcpPromptUsage(
                    input_tokens=2,
                    output_tokens=3,
                    total_tokens=11,
                    cached_read_tokens=4,
                ),
            )
        },
    ]
    monkeypatch.setattr(acp_module, "AcpAgentClient", _FakeClient)

    async def no_wait(_delay: float) -> None:
        # The failed process must be torn down before entering backoff.
        assert _FakeClient.force_closes == 1
        return

    monkeypatch.setattr(acp_module.asyncio, "sleep", no_wait)
    usage_calls: list[tuple[Any, ...]] = []
    broker = _broker(bus, [ApprovalMode.BYPASS])
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=broker,
        event_bus=bus,
        usage_callback=lambda *args: usage_calls.append(args),
    )
    # The fake sends no chunks, so success correctly becomes empty-output.
    result = await controller.run()
    assert result.startswith("Error:")
    assert len(retries) == 1
    assert _FakeClient.force_closes == 2
    assert controller.total_usage_tokens == 11
    assert len(usage_calls) == 1
    assert usage_calls[0][6] == 4
    assert usage_calls[0][-1] == 11


@pytest.mark.asyncio
async def test_stateless_open_session_failure_retries_instead_of_pausing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A failure classified stateless by open_session stays in the retry window.

    The client deliberately classifies failures that settled before
    session/new was sent as retryable AcpConnectError; the controller must
    honor that instead of pausing on its own connected-flag bookkeeping.
    """
    bus = EventBus()
    retries = await _capture(bus, SubAgentRetryAttempt)
    paused = await _capture(bus, SubAgentPaused)
    _FakeClient.force_closes = 0
    _FakeClient.scripts = [
        {"open": AcpConnectError("agent exited after initialize")},
        {"prompt": AcpPromptOutcome(stop_reason="end_turn", usage=None)},
    ]
    monkeypatch.setattr(acp_module, "AcpAgentClient", _FakeClient)

    async def no_wait(_delay: float) -> None:
        return

    monkeypatch.setattr(acp_module.asyncio, "sleep", no_wait)
    broker = _broker(bus, [ApprovalMode.BYPASS])
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=broker,
        event_bus=bus,
    )
    result = await controller.run()
    # The fake sends no chunks, so the successful retry is empty-output.
    assert result.startswith("Error:")
    assert len(retries) == 1
    assert paused == []


@pytest.mark.asyncio
async def test_translator_callback_adopts_translator_before_transport_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The audit-trail adoption seam must fire before connect().

    A permission decision recorded during open_session (ACP allows requests
    the moment session/new is announced) must already be part of the durable
    trail when a pre-handshake failure writes the pause audit.
    """
    bus = EventBus()
    order: list[str] = []

    class OrderClient(_FakeClient):
        async def connect(self) -> None:
            order.append("connect")
            await super().connect()

    _FakeClient.scripts = [{"prompt": AcpPromptOutcome(stop_reason="end_turn", usage=None)}]
    monkeypatch.setattr(acp_module, "AcpAgentClient", OrderClient)

    async def adopt(translator: Any) -> None:
        order.append("adopt")
        assert translator is not None

    broker = _broker(bus, [ApprovalMode.BYPASS])
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=broker,
        event_bus=bus,
        translator_callback=adopt,
    )
    await controller.run()
    assert order[:2] == ["adopt", "connect"]


@pytest.mark.asyncio
async def test_controller_post_session_transport_pauses_then_abort_never_leaks_diagnostic_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bus = EventBus()
    pauses = await _capture(bus, SubAgentPaused)
    _FakeClient.scripts = [{"prompt": AcpTransportError("wire closed")}]
    monkeypatch.setattr(acp_module, "AcpAgentClient", _FakeClient)
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
    )
    task = asyncio.create_task(controller.run())
    await wait_for(lambda: bool(pauses), description="ACP pause")
    assert pauses[0].diagnostic_path == str(tmp_path / "stderr.log")
    assert controller.request_abort()
    result = await task
    assert result.startswith("Error:")
    assert str(tmp_path / "stderr.log") not in result


@pytest.mark.asyncio
async def test_controller_cascade_abort_during_prompt_is_non_blocking_and_cancels_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bus = EventBus()
    cascade_events = await _capture(bus, SubAgentCascadeAborted)

    class HangingClient(_FakeClient):
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        def __init__(self, spec: AcpAgentSpec, callbacks: Any, **kwargs: Any) -> None:
            self.spec = spec
            self.callbacks = callbacks

        async def connect(self) -> None:
            return

        async def open_session(self) -> Any:
            return SimpleNamespace(session_id="remote", agent_info=None)

        async def prompt(self, _prompt: str) -> AcpPromptOutcome:
            type(self).entered.set()
            await type(self).cancelled.wait()
            return AcpPromptOutcome(stop_reason="cancelled", usage=None)

        async def cancel(self) -> None:
            type(self).cancelled.set()

    monkeypatch.setattr(acp_module, "AcpAgentClient", HangingClient)
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
    )
    task = asyncio.create_task(controller.run())
    await wait_for(HangingClient.entered.is_set, description="ACP prompt")
    await controller.cascade_abort()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cascade_events


@pytest.mark.asyncio
async def test_acp_cascade_commits_parent_before_controller_finalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bus = EventBus()
    order: list[str] = []

    class HangingClient(_FakeClient):
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        def __init__(self, spec: AcpAgentSpec, callbacks: Any, **kwargs: Any) -> None:
            self.spec = spec
            self.callbacks = callbacks

        async def connect(self) -> None:
            return

        async def open_session(self) -> Any:
            return SimpleNamespace(session_id="remote", agent_info=None)

        async def prompt(self, _prompt: str) -> AcpPromptOutcome:
            type(self).entered.set()
            await type(self).cancelled.wait()
            return AcpPromptOutcome(stop_reason="cancelled", usage=None)

        async def cancel(self) -> None:
            type(self).cancelled.set()

    async def finalize() -> None:
        order.append("finalized")

    monkeypatch.setattr(acp_module, "AcpAgentClient", HangingClient)
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
        parent_interrupted_result_commit=lambda: order.append("parent-committed"),
        cancellation_finalizer=finalize,
    )
    task = asyncio.create_task(controller.run())
    await wait_for(HangingClient.entered.is_set, description="ACP prompt")

    await controller.cascade_abort()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The parent slot commits synchronously at cascade time; the durable log
    # finalization is deferred to the invocation wrapper, which runs it only
    # after ``run()`` has fully unwound (teardown + terminal flush included).
    assert order == ["parent-committed"]

    await controller.finalize_cancellation()
    assert order == ["parent-committed", "finalized"]


@pytest.mark.asyncio
async def test_cascade_during_pause_callback_await_does_not_deadlock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A cascade that lands while the pause callback is awaiting must not hang run().

    _pause_and_wait installs _pending_decision only AFTER awaiting _pause_callback.
    A cascade during that await finds no future to resolve (its one-shot
    _resolve_decision is a no-op) and no active client to cancel; without the
    post-callback re-check, a fresh unresolved future is then created and awaited
    forever. Reproduce with a gated pause callback.
    """
    bus = EventBus()
    cascade_events = await _capture(bus, SubAgentCascadeAborted)
    pauses = await _capture(bus, SubAgentPaused)
    _FakeClient.scripts = [{"prompt": AcpTransportError("wire closed")}]
    monkeypatch.setattr(acp_module, "AcpAgentClient", _FakeClient)

    in_pause_cb = asyncio.Event()
    release_pause_cb = asyncio.Event()

    async def gated_pause_callback() -> None:
        in_pause_cb.set()
        await release_pause_cb.wait()

    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
        pause_callback=gated_pause_callback,
    )
    task = asyncio.create_task(controller.run())
    # Inside the pause-callback await, with _pending_decision still None — the
    # exact window where a cascade would otherwise be lost.
    await wait_for(in_pause_cb.is_set, description="pause callback entered")
    await controller.cascade_abort()
    release_pause_cb.set()
    try:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert cascade_events
    assert controller.status is SubAgentStatus.CASCADE_ABORTED
    # No pause was announced for an invocation that was already being torn down.
    assert not pauses


@pytest.mark.asyncio
async def test_post_session_refusal_flushes_interrupted_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A refusal after the remote streamed a tool call must finalize that call.

    Config/auth/refusal handlers previously skipped translator.flush_interrupted(),
    so a SubAgentToolCallStart published before the rejection had no matching
    result — stale running ownership after the parent invocation terminated.
    """
    bus = EventBus()
    results = await _capture(bus, SubAgentToolCallResult)

    class RefusingClient(_FakeClient):
        def __init__(self, spec: AcpAgentSpec, callbacks: Any, **kwargs: Any) -> None:
            self.spec = spec
            self.callbacks = callbacks
            self.stateful_phase_started = False
            self._sink = kwargs["update_sink"]

        async def connect(self) -> None:
            return

        async def open_session(self) -> Any:
            self.stateful_phase_started = True
            return SimpleNamespace(session_id="remote", agent_info=None)

        async def prompt(self, _prompt: str) -> AcpPromptOutcome:
            await self._sink.put(
                1,
                _notification(
                    ToolCallProgress(
                        sessionUpdate="tool_call_update",
                        toolCallId="running",
                        title="Running",
                        status="in_progress",
                    )
                ),
            )
            raise AcpRefusalError("policy refusal")

    _FakeClient.scripts = []
    monkeypatch.setattr(acp_module, "AcpAgentClient", RefusingClient)
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
    )
    result = await controller.run()

    assert result.startswith("Error:")
    assert controller.status is SubAgentStatus.COMPLETED
    # The started tool call was flushed as interrupted — no dangling running call.
    assert any(r.result == "(interrupted)" for r in results)


@pytest.mark.asyncio
async def test_tool_call_started_during_force_close_is_flushed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A tool-call start the still-alive update consumer processes DURING the
    force-close ladder must be finalized by the post-teardown sweep.

    The client cancels the update consumer only at the very end of force_close,
    so a buffered update dispatched to the translator mid-teardown lands AFTER
    the per-handler flush. Without the post-close sweep it leaves a
    SubAgentToolCallStart with no result and stale running ownership.
    """
    bus = EventBus()
    starts = await _capture(bus, SubAgentToolCallStart)
    results = await _capture(bus, SubAgentToolCallResult)

    class LateCloseClient(_FakeClient):
        def __init__(self, spec: AcpAgentSpec, callbacks: Any, **kwargs: Any) -> None:
            self.spec = spec
            self.callbacks = callbacks
            self.stateful_phase_started = False
            self._sink = kwargs["update_sink"]

        async def connect(self) -> None:
            return

        async def open_session(self) -> Any:
            self.stateful_phase_started = True
            return SimpleNamespace(session_id="remote", agent_info=None)

        async def prompt(self, _prompt: str) -> AcpPromptOutcome:
            return AcpPromptOutcome(stop_reason="end_turn", usage=None)

        async def force_close(self) -> None:
            type(self).force_closes += 1
            await self._sink.put(
                1,
                _notification(
                    ToolCallProgress(
                        sessionUpdate="tool_call_update",
                        toolCallId="late",
                        title="Late",
                        status="in_progress",
                    )
                ),
            )

    _FakeClient.scripts = []
    monkeypatch.setattr(acp_module, "AcpAgentClient", LateCloseClient)
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
    )
    await controller.run()

    assert any(s.call_id.endswith("late") for s in starts)
    assert any(r.result == "(interrupted)" for r in results)


@pytest.mark.asyncio
async def test_tool_call_started_before_prompt_is_flushed_on_cancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A tool call streamed after session/new but before the prompt (prompt_started
    still False) must be finalized when the attempt is cancelled — the
    CancelledError handler flushes unconditionally like its siblings."""
    bus = EventBus()
    results = await _capture(bus, SubAgentToolCallResult)

    class PrePromptClient(_FakeClient):
        def __init__(self, spec: AcpAgentSpec, callbacks: Any, **kwargs: Any) -> None:
            self.spec = spec
            self.callbacks = callbacks
            self.stateful_phase_started = False
            self._sink = kwargs["update_sink"]

        async def connect(self) -> None:
            return

        async def open_session(self) -> Any:
            self.stateful_phase_started = True
            await self._sink.put(
                1,
                _notification(
                    ToolCallProgress(
                        sessionUpdate="tool_call_update",
                        toolCallId="pre",
                        title="Pre",
                        status="in_progress",
                    )
                ),
            )
            return SimpleNamespace(session_id="remote", agent_info=None)

        async def prompt(self, _prompt: str) -> AcpPromptOutcome:
            raise AssertionError("prompt must not run — cancelled first")

    async def cancel_before_prompt(_attempt: int, _handshake: Any, _translator: Any) -> None:
        raise asyncio.CancelledError

    _FakeClient.scripts = []
    monkeypatch.setattr(acp_module, "AcpAgentClient", PrePromptClient)
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
        attempt_callback=cancel_before_prompt,
    )
    with pytest.raises(asyncio.CancelledError):
        await controller.run()

    assert any(r.result == "(interrupted)" for r in results)


@pytest.mark.asyncio
async def test_completed_count_synced_after_cancel_flush(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A tool call finalized on the cancel path must reach completed_tool_calls.

    Every sibling error handler syncs ``self._completed_calls`` after its flush;
    the cancel handler and the post-teardown sweep must too, or a call the cancel
    flush terminalized publishes its terminal event yet stays absent from
    ``completed_tool_calls`` — the value tools.py persists into the audit log.
    """
    bus = EventBus()
    results = await _capture(bus, SubAgentToolCallResult)

    class PrePromptClient(_FakeClient):
        def __init__(self, spec: AcpAgentSpec, callbacks: Any, **kwargs: Any) -> None:
            self.spec = spec
            self.callbacks = callbacks
            self.stateful_phase_started = False
            self._sink = kwargs["update_sink"]

        async def connect(self) -> None:
            return

        async def open_session(self) -> Any:
            self.stateful_phase_started = True
            await self._sink.put(
                1,
                _notification(
                    ToolCallProgress(
                        sessionUpdate="tool_call_update",
                        toolCallId="pre",
                        title="Pre",
                        status="in_progress",
                    )
                ),
            )
            return SimpleNamespace(session_id="remote", agent_info=None)

        async def prompt(self, _prompt: str) -> AcpPromptOutcome:
            raise AssertionError("prompt must not run — cancelled first")

    async def cancel_before_prompt(_attempt: int, _handshake: Any, _translator: Any) -> None:
        raise asyncio.CancelledError

    _FakeClient.scripts = []
    monkeypatch.setattr(acp_module, "AcpAgentClient", PrePromptClient)
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
        attempt_callback=cancel_before_prompt,
    )
    with pytest.raises(asyncio.CancelledError):
        await controller.run()

    assert any(r.result == "(interrupted)" for r in results)
    # The flushed call is counted for the audit log, not silently dropped.
    assert controller.completed_tool_calls == 1


@pytest.mark.asyncio
async def test_terminal_publish_is_single_delivery_under_concurrent_flush() -> None:
    """The live consumer completing a call and a cancel-flush must not both
    terminalize it. The terminal claim happens before any await, so whichever
    task wins the guard delivers exactly one SubAgentToolCallResult and the
    other returns at the guard — no duplicate result, no double count."""
    bus = EventBus()
    results = await _capture(bus, SubAgentToolCallResult)
    translator = AcpUpdateTranslator(
        event_bus=bus,
        session_id=None,
        agent_name="External",
        invocation_id="inv",
        attempt=1,
    )
    await translator.put(
        1,
        _notification(
            ToolCallProgress(
                sessionUpdate="tool_call_update",
                toolCallId="dup",
                title="Dup",
                status="in_progress",
            )
        ),
    )

    async def slow_dup_start(event: SubAgentToolCallStart) -> None:
        # Yield inside the terminal's ensure-start so a second terminalizer can
        # reach the guard while the first is mid-publish — the exact race window.
        if event.call_id.endswith("dup"):
            await asyncio.sleep(0)

    await bus.subscribe(SubAgentToolCallStart, slow_dup_start)

    # The consumer processing a completed update and the cancel-path flush race
    # to terminalize the same record.
    await asyncio.gather(
        translator.put(
            2,
            _notification(
                ToolCallProgress(
                    sessionUpdate="tool_call_update",
                    toolCallId="dup",
                    title="Dup",
                    status="completed",
                )
            ),
        ),
        translator.flush_interrupted(),
    )

    dup_results = [r for r in results if r.call_id.endswith("dup")]
    assert len(dup_results) == 1
    assert translator.completed_count == 1


@pytest.mark.asyncio
async def test_post_close_sweep_is_cancellation_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A second interrupt landing DURING the post-teardown sweep must not tear
    it. The sweep is shielded like force_close, so a call streamed at close time
    is still finalized even though the run is being cancelled a second time."""
    bus = EventBus()
    results = await _capture(bus, SubAgentToolCallResult)

    class LateCloseClient(_FakeClient):
        def __init__(self, spec: AcpAgentSpec, callbacks: Any, **kwargs: Any) -> None:
            self.spec = spec
            self.callbacks = callbacks
            self.stateful_phase_started = False
            self._sink = kwargs["update_sink"]

        async def connect(self) -> None:
            return

        async def open_session(self) -> Any:
            self.stateful_phase_started = True
            return SimpleNamespace(session_id="remote", agent_info=None)

        async def prompt(self, _prompt: str) -> AcpPromptOutcome:
            return AcpPromptOutcome(stop_reason="end_turn", usage=None)

        async def force_close(self) -> None:
            type(self).force_closes += 1
            await self._sink.put(
                1,
                _notification(
                    ToolCallProgress(
                        sessionUpdate="tool_call_update",
                        toolCallId="late",
                        title="Late",
                        status="in_progress",
                    )
                ),
            )

    run_task: asyncio.Task[Any] | None = None
    late_starts = 0

    async def cancel_during_sweep(event: SubAgentToolCallStart) -> None:
        nonlocal late_starts
        # The first "late" start is the consumer's; the second is the sweep's
        # ensure-start — cancel the run there, mid-sweep, to model a re-interrupt.
        if event.call_id.endswith("late"):
            late_starts += 1
            if late_starts == 2 and run_task is not None:
                run_task.cancel()
                await asyncio.sleep(0)

    await bus.subscribe(SubAgentToolCallStart, cancel_during_sweep)

    _FakeClient.scripts = []
    monkeypatch.setattr(acp_module, "AcpAgentClient", LateCloseClient)
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
    )
    run_task = asyncio.create_task(controller.run())
    with pytest.raises(asyncio.CancelledError):
        await run_task

    # The re-interrupt did not strand the late call: the sweep still finalized it.
    assert any(r.result == "(interrupted)" for r in results)


@pytest.mark.asyncio
async def test_cascade_during_abort_publish_wins_over_normal_return(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A cascade_abort landing DURING the SubAgentAborted publish must win.

    The user Abort decision passes the pre-branch cascade recheck, then awaits
    SubAgentAborted publication; a concurrent cascade sets _cascade_requested in
    that window. Without a post-publish recheck run() returns the abort error
    normally while SubAgentCascadeAborted is already published — a contradiction.
    """
    bus = EventBus()
    cascades = await _capture(bus, SubAgentCascadeAborted)
    _FakeClient.scripts = [{"prompt": AcpTransportError("drain failed", usage=None)}]
    monkeypatch.setattr(acp_module, "AcpAgentClient", _FakeClient)
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
    )

    async def cascade_mid_abort_publish(_event: SubAgentAborted) -> None:
        # Runs inline during the SubAgentAborted publish (the exact window).
        await controller.cascade_abort()

    await bus.subscribe(SubAgentAborted, cascade_mid_abort_publish)

    task = asyncio.create_task(controller.run())
    await wait_for(lambda: controller.status.value == "paused", description="abort-vs-cascade pause")
    assert controller.request_abort()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The cascade won: status is CASCADE_ABORTED and its terminal event fired.
    assert controller.status is SubAgentStatus.CASCADE_ABORTED
    assert len(cascades) == 1


@pytest.mark.asyncio
async def test_drain_task_cancellation_safe_survives_repeated_awaiter_cancel() -> None:
    """The shield-LOOP the cascade-publish / teardown / flush paths funnel through
    must not abandon its task when the AWAITER is cancelled more than once.

    ``asyncio.shield`` only protects the shielded task from an external cancel —
    the awaiter still receives ``CancelledError``. A single ``await
    asyncio.shield(task)`` therefore returns (cancelled) the instant a second
    interrupt / shutdown lands, abandoning e.g. the in-flight
    SubAgentCascadeAborted publish. The loop re-awaits until the task genuinely
    completes, then re-raises ``CancelledError``.
    """
    released = asyncio.Event()

    async def slow() -> str:
        await released.wait()
        return "done"

    inner = asyncio.create_task(slow())
    drain = asyncio.create_task(acp_module._drain_task_cancellation_safe(inner))
    await asyncio.sleep(0)  # let drain reach `await asyncio.shield(inner)`

    for _ in range(2):  # repeated interrupt / shutdown cancels of the awaiter
        drain.cancel()
        await asyncio.sleep(0)
        # A naive single-shield drain would already be done (cancelled) here,
        # having abandoned `inner`; the loop keeps it waiting.
        assert not drain.done()
        assert not inner.done()

    released.set()
    with pytest.raises(asyncio.CancelledError):
        await drain
    # The shielded task ran to completion despite the repeated cancels.
    assert inner.done() and inner.result() == "done"


@pytest.mark.asyncio
async def test_transport_usage_survives_drain_failure_and_is_charged_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bus = EventBus()
    usage = AcpPromptUsage(input_tokens=8, output_tokens=13, total_tokens=25, cached_read_tokens=3)
    _FakeClient.scripts = [{"prompt": AcpTransportError("drain failed", usage=usage)}]
    monkeypatch.setattr(acp_module, "AcpAgentClient", _FakeClient)
    calls: list[tuple[Any, ...]] = []
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
        usage_callback=lambda *args: calls.append(args),
    )
    task = asyncio.create_task(controller.run())
    await wait_for(lambda: controller.status.value == "paused", description="usage-bearing transport pause")
    assert controller.request_abort()
    await task
    assert controller.total_usage_tokens == 25
    assert controller.usage_unreported_attempts == 0
    assert len(calls) == 1
    assert calls[0][6] == 3
    assert calls[0][-1] == 25


@pytest.mark.asyncio
async def test_absent_usage_is_unreported_and_authoritative_total_reaches_runtime_metadata() -> None:
    metadata = SessionRuntimeMetadata()
    metadata.accumulate_sub_agent_usage(
        5,
        input_tokens=2,
        output_tokens=3,
        cache_hit_tokens=4,
        authoritative_total=11,
    )
    assert metadata.total_session_tokens == 11
    assert metadata.total_session_input_tokens == 2
    assert metadata.total_session_output_tokens == 3
    assert metadata.total_session_cache_hit_tokens == 4


@pytest.mark.asyncio
async def test_cascade_abort_all_snapshots_and_gathers_failures() -> None:
    tools = SubAgentTools()

    class Controller:
        status = "running"

        def request_retry(self) -> bool:
            return False

        def request_abort(self) -> bool:
            return False

        async def cascade_abort(self) -> None:
            raise RuntimeError("boom")

    tools._controllers = {"a": Controller(), "b": Controller()}
    await tools.cascade_abort_all()


def test_turn_context_consumers_share_one_protocol_compliant_holder() -> None:
    holder = TurnContextHolder()
    bus = EventBus()
    middleware = ApprovalMiddleware(
        approval_policy=ApprovalPolicy(ApprovalConfig(), tools=[]),
        event_bus=bus,
        turn_context=holder,
    )
    broker = AcpPermissionBroker(
        event_bus=bus,
        session_id=None,
        caller_name="External",
        mode_getter=lambda: ApprovalMode.MANUAL,
        turn_context=holder,
        workspace_roots=[],
        workspace_cwd="",
        approval_judge=None,
        ask_user_timeout_seconds=None,
    )
    assert isinstance(holder, TurnContextReader)
    assert middleware._turn_context is holder
    assert broker._turn_context is holder


@pytest.mark.asyncio
async def test_cascade_abort_wins_over_remote_end_turn_race(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A remote finishing before our cancel lands must not turn an interrupt into success."""
    bus = EventBus()
    aborted = await _capture(bus, SubAgentCascadeAborted)
    controller_box: dict[str, AcpSubAgentController] = {}

    class _RacingClient(_FakeClient):
        async def prompt(self, _prompt: str) -> AcpPromptOutcome:
            await controller_box["controller"].cascade_abort()
            return AcpPromptOutcome(stop_reason="end_turn", usage=None)

    _FakeClient.scripts = [{}]
    monkeypatch.setattr(acp_module, "AcpAgentClient", _RacingClient)
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
    )
    controller_box["controller"] = controller

    with pytest.raises(asyncio.CancelledError):
        await controller.run()

    assert len(aborted) == 1


@pytest.mark.asyncio
async def test_cascade_during_terminal_force_close_overrides_completed_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A cascade_abort landing during a terminal handler's finally force-close
    (a window r8's flush/close awaits widened) must win over the COMPLETED
    result the handler already computed: the in-prompt guard sees a clean flag,
    so only run()'s re-check before `return result` can honor the cascade."""
    bus = EventBus()
    aborted = await _capture(bus, SubAgentCascadeAborted)
    controller_box: dict[str, AcpSubAgentController] = {}

    class _CascadeOnCloseClient(_FakeClient):
        async def prompt(self, _prompt: str) -> AcpPromptOutcome:
            # No cascade yet — the in-prompt guard passes and the success path
            # computes a terminal (empty-output) result.
            return AcpPromptOutcome(stop_reason="end_turn", usage=None)

        async def force_close(self) -> None:
            # Cascade arrives only during teardown, AFTER the result exists.
            await controller_box["controller"].cascade_abort()
            await super().force_close()

    _FakeClient.scripts = [{}]
    monkeypatch.setattr(acp_module, "AcpAgentClient", _CascadeOnCloseClient)
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
    )
    controller_box["controller"] = controller

    with pytest.raises(asyncio.CancelledError):
        await controller.run()

    assert controller.status is SubAgentStatus.CASCADE_ABORTED
    assert len(aborted) == 1


@pytest.mark.asyncio
async def test_cascade_after_resolved_pause_decision_overrides_abort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A cascade_abort that lands AFTER the user's Abort already resolved the
    pause decision must still win. The settled `_pending_decision` future makes
    cascade's one-shot `_resolve_decision` a no-op, so `decision` stays "abort";
    run() must re-check `_cascade_requested` before acting on it, else it
    publishes a contradictory SubAgentAborted and returns normally instead of
    cascade-cancelling."""
    bus = EventBus()
    aborted = await _capture(bus, SubAgentAborted)
    resumed = await _capture(bus, SubAgentResumed)
    cascaded = await _capture(bus, SubAgentCascadeAborted)
    controller_box: dict[str, AcpSubAgentController] = {}

    async def on_paused(_event: SubAgentPaused) -> None:
        # This runs inline during _pause_and_wait's SubAgentPaused publish,
        # BEFORE it awaits the decision future: resolve to "abort" first, then
        # cascade (its resolve no-ops the settled future) — exactly the race.
        controller = controller_box["controller"]
        assert controller.request_abort()
        await controller.cascade_abort()

    await bus.subscribe(SubAgentPaused, on_paused)

    _FakeClient.scripts = [{"prompt": AcpTransportError("wire closed")}]
    monkeypatch.setattr(acp_module, "AcpAgentClient", _FakeClient)
    controller = AcpSubAgentController(
        invocation_id="inv",
        tool_name="external",
        agent_name="External",
        prompt="work",
        spec_factory=lambda _attempt: _spec(tmp_path),
        broker=_broker(bus, [ApprovalMode.BYPASS]),
        event_bus=bus,
    )
    controller_box["controller"] = controller

    with pytest.raises(asyncio.CancelledError):
        await controller.run()

    assert controller.status is SubAgentStatus.CASCADE_ABORTED
    # The stale abort/retry decision must NOT surface a lifecycle event.
    assert aborted == []
    assert resumed == []
    assert len(cascaded) == 1


@pytest.mark.asyncio
async def test_progress_updates_do_not_split_streaming_message_segments() -> None:
    """Only ToolCallStart seals the open segment (§7.5); progress merges tool state."""
    from acp.schema import ToolCallStart

    translator = AcpUpdateTranslator(
        event_bus=None,
        session_id=None,
        agent_name="External",
        invocation_id="inv",
        attempt=1,
    )

    def _chunk(text: str) -> SessionNotification:
        return _notification(
            AgentMessageChunk(
                sessionUpdate="agent_message_chunk",
                messageId="a",
                content=TextContentBlock(type="text", text=text),
            )
        )

    await translator.put(1, _chunk("prefix "))
    await translator.put(
        2,
        _notification(
            ToolCallProgress(
                sessionUpdate="tool_call_update",
                toolCallId="call",
                title="Read",
                kind="read",
                status="in_progress",
            )
        ),
    )
    await translator.put(3, _chunk("suffix"))
    assert translator.result_text() == "prefix suffix"

    sealing = AcpUpdateTranslator(
        event_bus=None,
        session_id=None,
        agent_name="External",
        invocation_id="inv",
        attempt=1,
    )
    await sealing.put(
        1,
        _notification(
            AgentMessageChunk(
                sessionUpdate="agent_message_chunk",
                messageId="a",
                content=TextContentBlock(type="text", text="before tool"),
            )
        ),
    )
    await sealing.put(
        2,
        _notification(
            ToolCallStart(
                sessionUpdate="tool_call",
                toolCallId="call",
                title="Read",
                kind="read",
                status="in_progress",
            )
        ),
    )
    await sealing.put(
        3,
        _notification(
            AgentMessageChunk(
                sessionUpdate="agent_message_chunk",
                messageId="a",
                content=TextContentBlock(type="text", text="after tool"),
            )
        ),
    )
    assert sealing.result_text() == "after tool"
