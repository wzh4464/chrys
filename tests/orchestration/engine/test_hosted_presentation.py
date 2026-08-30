# Copyright (c) 2026 Chrys. All rights reserved.

"""Executor wiring tests for provider-hosted presentation events."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    AgentMessage,
    Error,
    Event,
    PresentationAttemptAccepted,
    PresentationAttemptRejected,
    ProvisionalPresentation,
    ToolCallArgsUpdated,
    ToolCallResult,
    ToolCallStart,
    ToolCallStatusUpdated,
)
from chrys.foundation.tool_result_metadata import TOOL_FAILURE_TEXT_SYNTHESIZED_METADATA_KEY
from chrys.kernel import (
    AgentResponse,
    AgentSession,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)
from chrys.kernel.middleware import ChatContext
from chrys.orchestration.engine.executor import Executor
from chrys.service.agent_middleware import IntermediateTextBuffer
from chrys.service.agent_middleware.events.hosted_tools import (
    FinalTextOp,
    HostedToolResultOp,
    IntermediateTextOp,
    PresentationAttemptAcceptedOp,
    PresentationAttemptRejectedOp,
    adapt_hosted_tool,
)
from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware
from chrys.service.llm.mock import MockResponse
from chrys.service.llm.openai_responses import RawOpenAIChatClient
from tests.support.pipeline_helpers import create_test_engine


class _FakeAsyncOpenAI:
    base_url = "https://api.openai.test"


def _responses_events(*, before: str = "", after: str = "", status: str = "completed") -> list[SimpleNamespace]:
    """Build a small Responses event fixture for the production normalizer."""
    item = SimpleNamespace(
        type="web_search_call",
        id="ws_1",
        status=status,
        action=SimpleNamespace(type="search", query="Chrys"),
    )
    events: list[SimpleNamespace] = []
    if before:
        events.append(
            SimpleNamespace(
                type="response.output_text.delta",
                delta=before,
                item_id="msg_before",
                output_index=0,
                content_index=0,
            )
        )
    events.extend(
        [
            SimpleNamespace(
                type="response.output_item.added",
                output_index=1,
                item=SimpleNamespace(type="web_search_call", id="ws_1", status="in_progress", action=None),
            ),
            SimpleNamespace(type="response.output_item.done", output_index=1, item=item),
        ]
    )
    if after:
        events.append(
            SimpleNamespace(
                type="response.output_text.delta",
                delta=after,
                item_id="msg_after",
                output_index=2,
                content_index=0,
            )
        )
    return events


def _response_completed_with_usage() -> SimpleNamespace:
    return SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            id="resp_1",
            conversation=None,
            model="gpt-test",
            created_at=0,
            status="completed",
            incomplete_details=None,
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=4,
                total_tokens=14,
                input_tokens_details=None,
                output_tokens_details=None,
            ),
        ),
    )


def _plain_text_then_reasoning_events() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            type="response.output_text.delta",
            delta="Final answer.",
            item_id="msg_final",
            output_index=0,
            content_index=0,
        ),
        SimpleNamespace(
            type="response.output_item.added",
            output_index=1,
            item=SimpleNamespace(
                type="reasoning",
                id="reasoning_1",
                content=[SimpleNamespace(text="internal reasoning")],
                summary=[],
                encrypted_content=None,
            ),
        ),
    ]


class _ValidatedAgent:
    """Minimal agent surface that runs parsed updates through validation."""

    def __init__(
        self,
        validation: ResponseValidationMiddleware,
        events: Sequence[SimpleNamespace],
        *,
        wait_after_updates: bool = False,
        explode_after_updates: bool = False,
    ) -> None:
        self.client = SimpleNamespace(STORES_BY_DEFAULT=False, FORCES_STATELESS=False)
        self._validation = validation
        self._events = list(events)
        self._wait_after_updates = wait_after_updates
        self._explode_after_updates = explode_after_updates
        self.wire_calls = 0

    async def run(self, messages: Sequence[Message], *, stream: bool = False, **_kwargs: Any) -> AgentResponse[Any]:
        assert not stream
        context = ChatContext(
            client=self.client,
            messages=list(messages),
            options={},
            stream=True,
            kwargs={"client_kwargs": {}},
        )

        async def _raw_updates() -> AsyncIterator[ChatResponseUpdate]:
            client = RawOpenAIChatClient(model="gpt-test", async_client=_FakeAsyncOpenAI())
            function_call_ids: dict[int, tuple[str, str]] = {}
            calls: dict[int, Content] = {}
            results: dict[int, Content] = {}
            for event in self._events:
                yield client._parse_chunk_from_openai(
                    event,
                    {},
                    function_call_ids,
                    hosted_call_contents=calls,
                    hosted_result_contents=results,
                )
            if self._explode_after_updates:
                raise RuntimeError("wire exploded")
            if self._wait_after_updates:
                await asyncio.Event().wait()

        async def _call_next() -> None:
            self.wire_calls += 1
            context.result = ResponseStream(_raw_updates(), finalizer=ChatResponse.from_updates)

        await self._validation.process(context, _call_next)
        assert isinstance(context.result, ResponseStream)
        _ = [update async for update in context.result]
        final = await context.result.get_final_response()
        return AgentResponse(messages=final.messages, finish_reason=final.finish_reason)


async def _executor_fixture(
    raw_events: Sequence[SimpleNamespace],
    *,
    wait_after_updates: bool = False,
    explode_after_updates: bool = False,
) -> tuple[Executor, _ValidatedAgent, list[Event]]:
    bus = EventBus()
    events: list[Event] = []

    async def _collect(event: Event) -> None:
        events.append(event)

    for event_type in (
        AgentMessage,
        Error,
        PresentationAttemptAccepted,
        PresentationAttemptRejected,
        ToolCallStart,
        ToolCallArgsUpdated,
        ToolCallStatusUpdated,
        ToolCallResult,
    ):
        await bus.subscribe(event_type, _collect)

    validation = ResponseValidationMiddleware(backoff_schedule=[0.0])
    agent = _ValidatedAgent(
        validation,
        raw_events,
        wait_after_updates=wait_after_updates,
        explode_after_updates=explode_after_updates,
    )
    executor = Executor(
        agent=agent,  # type: ignore[arg-type]
        session=AgentSession(),
        event_bus=bus,
        approval_middleware=MagicMock(),
        ask_user_middleware=MagicMock(),
        injection_middleware=MagicMock(),
        response_validation_middleware=validation,
    )
    return executor, agent, events


@pytest.mark.asyncio
async def test_failed_hosted_sequence_publishes_text_call_result_then_final() -> None:
    executor, _agent, events = await _executor_fixture(
        _responses_events(before="Checking sources.", after="No result was available.", status="failed")
    )

    await executor._execute([Message("user", ["search"])])

    intermediate = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, AgentMessage) and event.is_intermediate and event.text == "Checking sources."
    )
    start = next(index for index, event in enumerate(events) if isinstance(event, ToolCallStart))
    result = next(index for index, event in enumerate(events) if isinstance(event, ToolCallResult))
    final = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, AgentMessage) and event.is_final and event.text == "No result was available."
    )
    failed = events[result]
    assert isinstance(failed, ToolCallResult)
    assert intermediate < start < result < final
    assert failed.provider_status == "failed"
    assert failed.result.startswith("Error: ")
    accepted = next(event for event in events if isinstance(event, PresentationAttemptAccepted))
    assert accepted.segment_ids == ("item:msg_before:content:0",)


@pytest.mark.parametrize(
    ("provider_text", "expected_result", "expected_synthesized"),
    [
        ("", "Error: Provider-hosted tool failed.", True),
        ("provider failed", "Error: provider failed", False),
    ],
    ids=["synthesized", "provider-originated"],
)
@pytest.mark.asyncio
async def test_published_hosted_result_carries_synthesized_failure_signal_only_for_fallback(
    provider_text: str,
    expected_result: str,
    expected_synthesized: bool,
) -> None:
    executor, _agent, events = await _executor_fixture([])
    result = Content.from_search_tool_result(
        "search_semantic",
        tool_name="web_search",
        result=provider_text,
        status="",
        hosted_family="search",
        hosted_provider="openai",
        provider_status="",
        additional_properties={"is_error": True},
    )
    view = adapt_hosted_tool(None, result)

    await executor._publish_hosted_operation(HostedToolResultOp(0, "hosted:semantic", view, (0, 0, 0)))

    published = next(event for event in events if isinstance(event, ToolCallResult))
    assert published.result == expected_result
    assert (published.metadata.get(TOOL_FAILURE_TEXT_SYNTHESIZED_METADATA_KEY) is True) is expected_synthesized


@pytest.mark.asyncio
async def test_executor_reserves_provisional_batch_and_persists_only_on_acceptance() -> None:
    bus = EventBus()
    events: list[Event] = []

    async def _collect(event: Event) -> None:
        events.append(event)

    for event_type in (AgentMessage, PresentationAttemptAccepted, PresentationAttemptRejected):
        await bus.subscribe(event_type, _collect)
    buffer = IntermediateTextBuffer()
    committed: list[tuple[str, int]] = []
    executor = Executor(
        agent=MagicMock(),
        session=AgentSession(),
        event_bus=bus,
        approval_middleware=MagicMock(),
        ask_user_middleware=MagicMock(),
        injection_middleware=MagicMock(),
        intermediate_buffer=buffer,
        commit_intermediate_text=lambda text, batch_id: committed.append((text, batch_id)),
    )
    segment = IntermediateTextOp(
        "Checking sources.",
        (0, 0, 0),
        attempt_id="presentation:1",
        segment_ids=("segment:1",),
        provisional=True,
    )

    await executor._publish_hosted_operation(segment)
    assert committed == []
    assert buffer.batch_id == 1
    provisional = events[-1]
    assert isinstance(provisional, AgentMessage)
    assert provisional.presentation == ProvisionalPresentation("presentation:1", "segment:1")

    await executor._publish_hosted_operation(PresentationAttemptAcceptedOp("presentation:1", (segment,)))
    assert committed == [("Checking sources.", 1)]
    assert isinstance(events[-1], PresentationAttemptAccepted)

    await executor._publish_hosted_operation(FinalTextOp("", structured_output_completed=True))
    terminal = events[-1]
    assert isinstance(terminal, AgentMessage)
    assert terminal.text == ""
    assert terminal.structured_output_completed is True

    rejected_segment = IntermediateTextOp(
        "Stale.",
        (0, 0, 0),
        attempt_id="presentation:2",
        segment_ids=("segment:2",),
        provisional=True,
    )
    await executor._publish_hosted_operation(rejected_segment)
    await executor._publish_hosted_operation(PresentationAttemptRejectedOp("presentation:2"))
    assert committed == [("Checking sources.", 1)]
    assert isinstance(events[-1], PresentationAttemptRejected)


@pytest.mark.asyncio
async def test_output_item_added_and_done_publish_one_start_through_hook_and_bridge() -> None:
    executor, _agent, events = await _executor_fixture(_responses_events(after="Done."))

    await executor._execute([Message("user", ["search"])])

    starts = [event for event in events if isinstance(event, ToolCallStart)]
    results = [event for event in events if isinstance(event, ToolCallResult)]
    assert len(starts) == len(results) == 1
    assert starts[0].call_id == results[0].call_id
    assert starts[0].provider_call_id == "ws_1"


@pytest.mark.asyncio
async def test_response_completed_usage_does_not_publish_final_text_as_provisional() -> None:
    raw_events = [
        *_responses_events(before="Checking sources.", after="Final answer."),
        _response_completed_with_usage(),
    ]
    executor, _agent, events = await _executor_fixture(raw_events)

    await executor._execute([Message("user", ["search"])])

    intermediate = [event.text for event in events if isinstance(event, AgentMessage) and event.is_intermediate]
    final = [event.text for event in events if isinstance(event, AgentMessage) and event.is_final]
    assert intermediate == ["Checking sources."]
    assert final == ["Final answer."]


@pytest.mark.asyncio
async def test_plain_reasoning_does_not_publish_final_text_as_provisional() -> None:
    executor, _agent, events = await _executor_fixture(_plain_text_then_reasoning_events())

    await executor._execute([Message("user", ["answer"])])

    intermediate = [event.text for event in events if isinstance(event, AgentMessage) and event.is_intermediate]
    final = [event.text for event in events if isinstance(event, AgentMessage) and event.is_final]
    assert intermediate == []
    assert final == ["Final answer."]


@pytest.mark.asyncio
async def test_search_without_final_text_retries_then_fails_without_blank_agent_message() -> None:
    executor, agent, events = await _executor_fixture(_responses_events())

    await executor._execute([Message("user", ["search"])])

    assert agent.wire_calls == 2
    assert executor.run_failed
    assert [event for event in events if isinstance(event, AgentMessage) and event.is_final] == []
    assert len([event for event in events if isinstance(event, Error)]) == 1


@pytest.mark.asyncio
async def test_interrupt_terminalizes_running_hosted_card() -> None:
    raw_events = _responses_events(before="Checking sources.")[:2]
    executor, _agent, events = await _executor_fixture(raw_events, wait_after_updates=True)
    task = asyncio.create_task(executor._execute([Message("user", ["search"])]))
    for _ in range(100):
        if any(isinstance(event, ToolCallStart) for event in events):
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("hosted start was not published")

    await executor.interrupt()
    await task

    interrupted = [
        event for event in events if isinstance(event, ToolCallStatusUpdated) and event.status == "interrupted"
    ]
    assert len(interrupted) == 1
    assert executor.was_interrupted
    provisional = [event for event in events if isinstance(event, AgentMessage) and event.presentation is not None]
    assert [event.text for event in provisional] == ["Checking sources."]
    accepted = [event for event in events if isinstance(event, PresentationAttemptAccepted)]
    assert len(accepted) == 1
    assert accepted[0].segment_ids == ("item:msg_before:content:0",)
    assert not any(isinstance(event, PresentationAttemptRejected) for event in events)


@pytest.mark.asyncio
async def test_terminal_executor_error_terminalizes_running_hosted_card() -> None:
    # A wire failure mid-search must fail the visible card before the run
    # error goes out: ACP clients have no error banner tied to the card and
    # would otherwise show it in-progress forever.
    raw_events = _responses_events()[:1]
    executor, _agent, events = await _executor_fixture(raw_events, explode_after_updates=True)

    await executor._execute([Message("user", ["search"])])

    assert executor.run_failed
    failed = next(
        index
        for index, event in enumerate(events)
        if isinstance(event, ToolCallStatusUpdated) and event.status == "failed"
    )
    error = next(index for index, event in enumerate(events) if isinstance(event, Error))
    start = next(index for index, event in enumerate(events) if isinstance(event, ToolCallStart))
    assert start < failed < error


@pytest.mark.asyncio
async def test_local_function_tools_keep_tool_event_middleware_as_single_publisher(tmp_path: Any) -> None:
    context = await create_test_engine(
        [MockResponse(tool_calls=[("echo", "local_1", {"message": "hello"})]), MockResponse(text="done")],
        tmp_path,
    )
    try:
        await context.send_message("go")

        starts = [event for event in context.events if isinstance(event, ToolCallStart)]
        results = [event for event in context.events if isinstance(event, ToolCallResult)]
        assert len(starts) == len(results) == 1
        assert starts[0].provider_hosted is False
        assert results[0].provider_hosted is False
    finally:
        await context.cleanup()


@pytest.mark.asyncio
async def test_non_hosted_response_keeps_existing_final_text_path(tmp_path: Any) -> None:
    context = await create_test_engine([MockResponse(text="unchanged final text")], tmp_path)
    try:
        await context.send_message("go")

        finals = [event.text for event in context.events if isinstance(event, AgentMessage) and event.is_final]
        assert finals == ["unchanged final text"]
        assert not any(isinstance(event, ToolCallStart) for event in context.events)
    finally:
        await context.cleanup()
