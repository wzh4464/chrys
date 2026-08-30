# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the instrumented LLM client helpers."""

from __future__ import annotations

import asyncio
from copy import copy
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from chrys.foundation.trajectory.context import (
    TRAJECTORY_EXCHANGE_KWARG,
    ExchangeTrace,
    side_call_scope,
    trajectory_scope,
)
from chrys.foundation.trajectory.envelope import ActorRole
from chrys.foundation.trajectory.event_types import EventType, ExchangeOutcome
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory_timing import TRAJECTORY_TIMING_KEY, build_trajectory_timing
from chrys.foundation.util.chrys_headers import (
    MODEL_ID_HEADER,
    PARENT_SESSION_ID_HEADER,
    SESSION_ID_HEADER,
    X_PARENT_SESSION_ID_HEADER,
    X_SESSION_ID_HEADER,
)
from chrys.kernel import (
    AgentResponse,
    AgentSession,
    BaseChatClient,
    ChatClientException,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FunctionTool,
    Message,
    ResponseStream,
    SessionContext,
    internal_side_call_scope,
)
from chrys.kernel.exceptions import ChatClientContentFilterException, ChatClientInvalidRequestException
from chrys.service.context.providers.history import CompressibleHistoryProvider
from chrys.service.llm.instrumented import (
    _compose_client_stack,
    _count_function_calls,
    _ensure_openai_response_has_choices,
    _extract_intermediate_text,
    _IntermediateTextMixin,
    _set_chrys_request_headers,
    create_instrumented_anthropic_client,
    create_instrumented_openai_client,
    create_instrumented_openai_responses_client,
)
from chrys.service.llm.route_sessions import llm_parent_session_id, llm_route_session_id
from chrys.service.session.message_metadata import MESSAGE_CREATED_AT_KEY, stamp_message_response_timing
from tests.service.trajectory._fakes import FakeSink, make_context


def _make_response(*messages: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(messages=list(messages))


def _make_msg(*contents: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(contents=list(contents))


def _text(t: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=t, provider_hosted=False)


def _fn_call(name: str = "tool", *, informational_only: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        type="function_call",
        name=name,
        informational_only=informational_only,
        provider_hosted=False,
    )


def _fn_result() -> SimpleNamespace:
    return SimpleNamespace(type="function_result", provider_hosted=False)


# ──────────────── _extract_intermediate_text ────────────────────────────


def test_extract_text_with_function_call() -> None:
    """Text alongside function_call should be extracted."""
    resp = _make_response(_make_msg(_text("Let me check"), _fn_call()))
    assert _extract_intermediate_text(resp) == "Let me check"


def test_extract_multiple_text_parts() -> None:
    """Multiple text parts should be concatenated."""
    resp = _make_response(_make_msg(_text("A"), _text("B"), _fn_call()))
    assert _extract_intermediate_text(resp) == "AB"


def test_extract_no_function_call_returns_none() -> None:
    """Text-only response (no function_call) should return None."""
    resp = _make_response(_make_msg(_text("Hello")))
    assert _extract_intermediate_text(resp) is None


def test_extract_function_call_only_returns_none() -> None:
    """Function call without text should return None."""
    resp = _make_response(_make_msg(_fn_call()))
    assert _extract_intermediate_text(resp) is None


def test_extract_empty_text_ignored() -> None:
    """Empty text parts should not count as intermediate text."""
    resp = _make_response(_make_msg(_text(""), _fn_call()))
    assert _extract_intermediate_text(resp) is None


def test_extract_across_multiple_messages() -> None:
    """Text in one message, function_call in another."""
    resp = _make_response(
        _make_msg(_text("thinking")),
        _make_msg(_fn_call()),
    )
    assert _extract_intermediate_text(resp) == "thinking"


def test_extract_ignores_informational_function_call() -> None:
    resp = _make_response(_make_msg(_text("visible"), _fn_call(informational_only=True)))

    assert _extract_intermediate_text(resp) is None


def test_extract_defers_hosted_response_text_to_presentation_bridge() -> None:
    hosted = SimpleNamespace(type="search_tool_call", provider_hosted=True)
    resp = _make_response(_make_msg(_text("checking"), hosted, _text("answer")))

    assert _extract_intermediate_text(resp) is None


# ──────────────── _count_function_calls ─────────────────────────────────


def test_count_function_calls_single() -> None:
    resp = _make_response(_make_msg(_fn_call()))
    assert _count_function_calls(resp) == 1


def test_count_function_calls_multiple() -> None:
    resp = _make_response(_make_msg(_fn_call("a"), _fn_call("b"), _fn_call("c")))
    assert _count_function_calls(resp) == 3


def test_count_function_calls_ignores_informational_calls() -> None:
    resp = _make_response(_make_msg(_fn_call("hosted", informational_only=True)))

    assert _count_function_calls(resp) == 0


def test_count_function_calls_text_only() -> None:
    resp = _make_response(_make_msg(_text("hello")))
    assert _count_function_calls(resp) == 0


def test_count_function_calls_empty() -> None:
    resp = _make_response(_make_msg())
    assert _count_function_calls(resp) == 0


def test_count_function_calls_ignores_function_result() -> None:
    """function_result is NOT a function_call."""
    resp = _make_response(_make_msg(_fn_result()))
    assert _count_function_calls(resp) == 0


# ──────────────── _ensure_openai_response_has_choices ───────────────────


def test_ensure_choices_passes_with_populated_choices() -> None:
    resp = SimpleNamespace(choices=[SimpleNamespace()])
    _ensure_openai_response_has_choices(resp)


def test_ensure_choices_passes_with_empty_list() -> None:
    """Empty choices is a valid (no-completion) response and must not raise."""
    resp = SimpleNamespace(choices=[])
    _ensure_openai_response_has_choices(resp)


def test_ensure_choices_raises_when_none_and_includes_body() -> None:
    body = '{"error": "rate limit exceeded"}'
    resp = SimpleNamespace(choices=None, model_dump_json=lambda: body)
    with pytest.raises(ChatClientException) as exc_info:
        _ensure_openai_response_has_choices(resp)
    msg = str(exc_info.value)
    assert "no 'choices' field" in msg
    assert "rate limit exceeded" in msg


def test_ensure_choices_truncates_long_body() -> None:
    big = '{"error": "' + ("x" * 5000) + '"}'
    resp = SimpleNamespace(choices=None, model_dump_json=lambda: big)
    with pytest.raises(ChatClientException) as exc_info:
        _ensure_openai_response_has_choices(resp)
    assert "[truncated]" in str(exc_info.value)


def test_ensure_choices_falls_back_to_repr_on_dump_failure() -> None:
    def _bad_dump() -> str:
        raise RuntimeError("pydantic broken")

    resp = SimpleNamespace(choices=None, model_dump_json=_bad_dump)
    with pytest.raises(ChatClientException):
        _ensure_openai_response_has_choices(resp)


# ──────────────── integration: instrumented OpenAI subclass ─────────────
#
# These tests build the actual ``_InstrumentedOpenAIChatCompletionClient``
# subclass via the factory and feed it real ``openai.types`` ``ChatCompletion``
# objects, verifying both that the override is wired in and that valid
# responses still flow through to the base parser.


def _make_chat_client(
    session_id: str | None = None,
    chat_client_cls: type[Any] | None = None,
    parent_session_id: str | None = None,
    use_route_session_context: bool = False,
) -> object:
    """Construct the real instrumented OpenAI client with a fake API key."""
    from openai import AsyncOpenAI

    from chrys.service.llm.instrumented import create_instrumented_openai_client

    return create_instrumented_openai_client(
        model_id="gpt-test",
        session_id=session_id,
        parent_session_id=parent_session_id,
        use_route_session_context=use_route_session_context,
        client=AsyncOpenAI(api_key="sk-fake"),
        chat_client_cls=chat_client_cls,
    )


def _make_responses_chat_client(
    session_id: str | None = None,
    parent_session_id: str | None = None,
    use_route_session_context: bool = False,
    chat_client_cls: type[Any] | None = None,
) -> object:
    """Construct the real instrumented OpenAI Responses client with a fake API key."""
    from openai import AsyncOpenAI

    from chrys.service.llm.instrumented import create_instrumented_openai_responses_client

    return create_instrumented_openai_responses_client(
        model_id="gpt-test",
        session_id=session_id,
        parent_session_id=parent_session_id,
        use_route_session_context=use_route_session_context,
        client=AsyncOpenAI(api_key="sk-fake"),
        chat_client_cls=chat_client_cls,
    )


@pytest.mark.asyncio
async def test_instrumented_responses_factory_preserves_deepseek_subclass_callbacks_and_headers() -> None:
    from chrys.service.llm.deepseek import DeepSeekResponsesClient

    async_calls: list[str] = []
    sync_calls: list[str] = []

    async def _async_callback(text: str) -> None:
        async_calls.append(text)

    def _sync_callback(text: str) -> None:
        sync_calls.append(text)

    from openai import AsyncOpenAI

    from chrys.service.llm.instrumented import create_instrumented_openai_responses_client

    client = create_instrumented_openai_responses_client(
        model_id="deepseek-test",
        session_id="session-1",
        parent_session_id="parent-1",
        client=AsyncOpenAI(api_key="sk-fake"),
        chat_client_cls=DeepSeekResponsesClient,
        on_intermediate_text_async=_async_callback,
        on_intermediate_text_sync=_sync_callback,
    )
    raw = client.inner.inner

    prepared = await raw._prepare_options([Message("user", ["hi"])], {})

    assert DeepSeekResponsesClient in type(raw).__mro__
    assert raw._on_intermediate_text_async is _async_callback
    assert raw._on_intermediate_text_sync is _sync_callback
    assert prepared["extra_headers"][SESSION_ID_HEADER] == "session-1"
    assert prepared["extra_headers"][PARENT_SESSION_ID_HEADER] == "parent-1"
    assert async_calls == []
    assert sync_calls == []


def test_raw_clients_require_preconfigured_sdk_clients() -> None:
    from chrys.service.llm.anthropic_chat import RawAnthropicClient
    from chrys.service.llm.openai_chat_completion import RawOpenAIChatCompletionClient
    from chrys.service.llm.openai_responses import RawOpenAIChatClient

    with pytest.raises(ValueError, match="pre-configured async_client"):
        RawOpenAIChatCompletionClient(model="gpt-test")
    with pytest.raises(ValueError, match="pre-configured async_client"):
        RawOpenAIChatClient(model="gpt-test")
    with pytest.raises(ValueError, match="pre-configured anthropic_client"):
        RawAnthropicClient(model="claude-test")


def test_instrumented_factories_require_preconfigured_sdk_clients() -> None:
    with pytest.raises(TypeError, match="client"):
        create_instrumented_openai_client(model_id="gpt-test")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="client"):
        create_instrumented_openai_responses_client(model_id="gpt-test")  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="anthropic_client"):
        create_instrumented_anthropic_client(model_id="claude-test")  # type: ignore[call-arg]


class _FailingWireClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def _inner_get_response(
        self,
        *,
        messages: Any,
        options: Any,
        stream: bool = False,
        **kwargs: Any,
    ) -> Any:
        del messages, options, kwargs
        if stream:

            async def _updates() -> Any:
                raise self.exc
                yield ChatResponseUpdate(contents=[Content.from_text("unused")])

            return ResponseStream(
                _updates(),
                finalizer=ChatResponse.from_updates,
            )

        async def _response() -> Any:
            raise self.exc

        return _response()


class _InstrumentedFailingWireClient(_IntermediateTextMixin, _FailingWireClient):
    pass


def _make_instrumented_wire_client(
    *,
    response: Any = None,
    stream_text: str = "streamed",
    delay: float = 0,
) -> Any:
    class _RawWireClient(BaseChatClient):
        def __init__(self) -> None:
            super().__init__()
            self.calls: list[dict[str, Any]] = []

        def _inner_get_response(
            self,
            *,
            messages: Any,
            options: Any,
            stream: bool = False,
            **kwargs: Any,
        ) -> Any:
            self.calls.append(
                {"messages": list(messages), "options": dict(options), "stream": stream, "kwargs": kwargs}
            )
            if stream:

                async def _updates() -> Any:
                    if delay > 0:
                        await asyncio.sleep(delay)
                    yield ChatResponseUpdate(
                        contents=[Content.from_text(stream_text)],
                        role="assistant",
                    )

                return ResponseStream(
                    _updates(),
                    finalizer=lambda updates: ChatResponse.from_updates(
                        updates,
                        output_format_type=options.get("response_format"),
                    ),
                )

            async def _response() -> Any:
                if delay > 0:
                    await asyncio.sleep(delay)
                if response is not None:
                    return response
                return ChatResponse(
                    messages=[
                        Message(
                            "assistant",
                            [Content.from_text(stream_text)],
                        )
                    ],
                    response_format=options.get("response_format"),
                )

            return _response()

    class _InstrumentedWireClient(_IntermediateTextMixin, _RawWireClient):
        pass

    return _InstrumentedWireClient()


class _NoopCompaction:
    async def __call__(self, messages: list[Any], context: Any = None) -> bool:
        self.messages = messages
        self.context = context
        return False


@pytest.mark.asyncio
async def test_chrys_chat_client_exception_propagates_non_streaming() -> None:
    inner = ValueError("root cause")
    chrys_exc = ChatClientInvalidRequestException(
        "provider rejected request",
        inner_exception=inner,
        log_level=None,
    )
    client = _InstrumentedFailingWireClient(chrys_exc)

    with pytest.raises(ChatClientInvalidRequestException) as exc_info:
        await client._inner_get_response(messages=[Message("user", ["hi"])], options={})

    assert exc_info.value is chrys_exc
    assert exc_info.value.args == ("provider rejected request", inner)


@pytest.mark.asyncio
async def test_chrys_chat_client_exception_propagates_streaming() -> None:
    inner = RuntimeError("filter details")
    chrys_exc = ChatClientContentFilterException(
        "provider content filter",
        inner_exception=inner,
        log_level=None,
    )
    client = _InstrumentedFailingWireClient(chrys_exc)

    stream = client._inner_get_response(messages=[Message("user", ["hi"])], options={}, stream=True)

    assert isinstance(stream, ResponseStream)
    with pytest.raises(ChatClientContentFilterException) as exc_info:
        async for _update in stream:
            pass
    assert exc_info.value is chrys_exc
    assert exc_info.value.args == ("provider content filter", inner)


@pytest.mark.asyncio
async def test_streaming_get_response_with_compaction_returns_chrys_stream() -> None:
    client = _make_instrumented_wire_client(stream_text="compacted stream")
    compaction = _NoopCompaction()

    stream = client.get_response(
        [Message("user", ["hi"])],
        stream=True,
        options={},
        compaction_strategy=compaction,
    )

    assert isinstance(stream, ResponseStream)
    updates = [update async for update in stream]
    assert [update.text for update in updates] == ["compacted stream"]
    final = await stream.get_final_response()
    assert final.text == "compacted stream"
    assert client.calls[0]["stream"] is True
    assert compaction.messages


@pytest.mark.asyncio
async def test_native_response_preserves_lazy_value_parse() -> None:
    class StructuredPayload(BaseModel):
        answer: int

    native_response = ChatResponse(
        messages=[Message("assistant", [Content.from_text("not-json")])],
        response_format=StructuredPayload,
    )
    client = _make_instrumented_wire_client(response=native_response)

    response = await client._inner_get_response(messages=[Message("user", ["hi"])], options={})

    assert response._value_parsed is False
    with pytest.raises(ValidationError):
        _ = response.value


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_instrumented_response_persists_measured_trajectory_timing(stream: bool) -> None:
    client = _make_instrumented_wire_client(stream_text="timed")

    response_or_stream = client._inner_get_response(
        messages=[Message("user", ["hi"])],
        options={},
        stream=stream,
    )
    if stream:
        async for _update in response_or_stream:
            pass
        response = await response_or_stream.get_final_response()
    else:
        response = await response_or_stream

    message = response.messages[-1]
    timing = message.additional_properties[TRAJECTORY_TIMING_KEY]
    assert timing["started_at"] <= timing["finished_at"]
    assert timing["finished_at"] == message.additional_properties[MESSAGE_CREATED_AT_KEY]
    assert isinstance(timing["duration_ms"], int)
    assert timing["duration_ms"] >= 0


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_measured_timing_survives_tool_loop_and_history_persistence(stream: bool) -> None:
    """The production loop and history provider preserve the wire-stamped message."""
    wire_client = _make_instrumented_wire_client(stream_text="persisted", delay=0.01)
    client = _compose_client_stack(
        wire_client,
        max_iterations=None,
        max_consecutive_errors=None,
    )
    user = Message("user", [Content.from_text("hi")])
    provider = CompressibleHistoryProvider()
    session = AgentSession(session_id="timing-survival")
    context = SessionContext(session_id="timing-survival", input_messages=[user])
    state: dict[str, Any] = {"messages": [], "compressed_msgs": []}

    await provider.before_run(agent=object(), session=session, context=context, state=state)
    wire_messages = context.get_messages(include_input=True)
    if stream:
        response_stream = client.get_response(wire_messages, stream=True, options={})
        assert isinstance(response_stream, ResponseStream)
        _ = [update async for update in response_stream]
        response = await response_stream.get_final_response()
    else:
        pending_response = client.get_response(wire_messages, stream=False, options={})
        assert not isinstance(pending_response, ResponseStream)
        response = await pending_response
    context._response = AgentResponse(messages=response.messages)
    await provider.after_run(agent=object(), session=session, context=context, state=state)

    persisted = state["messages"][-1]
    assert persisted is response.messages[-1]
    timing = persisted.additional_properties[TRAJECTORY_TIMING_KEY]
    assert timing["finished_at"] == persisted.additional_properties[MESSAGE_CREATED_AT_KEY]
    assert timing["duration_ms"] >= 1


@pytest.mark.asyncio
async def test_instrumented_response_stamps_provider_hosted_tool_contents() -> None:
    hosted_call = Content.from_search_tool_call(
        "search-1",
        tool_name="web_search",
        arguments={"query": "timing"},
    )
    native_response = ChatResponse(messages=[Message("assistant", [hosted_call])])
    client = _make_instrumented_wire_client(response=native_response)

    response = await client._inner_get_response(messages=[Message("user", ["hi"])], options={})

    message_timing = response.messages[-1].additional_properties[TRAJECTORY_TIMING_KEY]
    assert hosted_call.additional_properties[TRAJECTORY_TIMING_KEY] == message_timing


@pytest.mark.asyncio
async def test_instrumented_response_timing_does_not_mutate_echoed_request_objects() -> None:
    old_started_at = "2000-01-01T01:02:03+00:00"
    old_finished_at = "2000-01-01T01:02:04+00:00"
    hosted_call = Content.from_search_tool_call(
        "search-old",
        tool_name="web_search",
        arguments={"query": "old"},
    )
    old_timing = build_trajectory_timing(
        started_at=old_started_at,
        finished_at=old_finished_at,
        duration_ms=1_000,
    )
    hosted_call.additional_properties[TRAJECTORY_TIMING_KEY] = dict(old_timing)
    echoed = Message("assistant", [hosted_call])
    stamp_message_response_timing(
        echoed,
        started_at=old_started_at,
        finished_at=old_finished_at,
        duration_ms=1_000,
    )
    shallow_hosted_echo = copy(hosted_call)
    assert shallow_hosted_echo.additional_properties is hosted_call.additional_properties
    shallow_echo = Message("assistant", [shallow_hosted_echo])
    fresh = Message("assistant", [Content.from_text("fresh")])
    client = _make_instrumented_wire_client(response=ChatResponse(messages=[echoed, shallow_echo, fresh]))

    response = await client._inner_get_response(messages=[echoed], options={})

    assert echoed.additional_properties[TRAJECTORY_TIMING_KEY] == old_timing
    assert echoed.additional_properties[MESSAGE_CREATED_AT_KEY] == old_timing["finished_at"]
    assert hosted_call.additional_properties[TRAJECTORY_TIMING_KEY] == old_timing
    assert shallow_hosted_echo.additional_properties[TRAJECTORY_TIMING_KEY] == old_timing
    assert response.messages[-1].additional_properties[TRAJECTORY_TIMING_KEY] != old_timing


@pytest.mark.asyncio
async def test_native_stream_final_response_preserves_response_format() -> None:
    class StructuredPayload(BaseModel):
        answer: str

    client = _make_instrumented_wire_client(stream_text='{"answer":"ok"}')

    stream = client._inner_get_response(
        messages=[Message("user", ["hi"])],
        options={"response_format": StructuredPayload},
        stream=True,
    )
    updates = [update async for update in stream]
    final = await stream.get_final_response()

    assert [update.text for update in updates] == ['{"answer":"ok"}']
    assert final.value == StructuredPayload(answer="ok")


# ──────────────── internal side-call suppression ─────────────────────────
#
# LAST_WORDS side calls go through ``_inner_get_response`` inside
# ``internal_side_call_scope()``.  If the model ignores the no-tools
# instruction and returns text alongside a function_call, the mixin must NOT
# publish that text (or a batch-boundary signal) — the throwaway side-call
# response never joins the conversation.


def _make_tool_call_wire_client() -> Any:
    """Instrumented wire client whose responses carry text + function_call."""

    def _contents() -> list[Content]:
        return [Content.from_text("Let me check"), Content.from_function_call("call-1", "tool")]

    class _RawToolCallWireClient(BaseChatClient):
        def _inner_get_response(
            self,
            *,
            messages: Any,
            options: Any,
            stream: bool = False,
            **kwargs: Any,
        ) -> Any:
            del messages, options, kwargs
            if stream:

                async def _updates() -> Any:
                    yield ChatResponseUpdate(contents=_contents(), role="assistant")

                return ResponseStream(_updates(), finalizer=ChatResponse.from_updates)

            async def _response() -> Any:
                return ChatResponse(messages=[Message("assistant", _contents())])

            return _response()

    class _InstrumentedToolCallWireClient(_IntermediateTextMixin, _RawToolCallWireClient):
        pass

    return _InstrumentedToolCallWireClient()


@pytest.mark.asyncio
async def test_intermediate_text_suppressed_in_internal_side_call_non_streaming() -> None:
    client = _make_tool_call_wire_client()
    fired: list[str] = []

    async def _cb(text: str) -> None:
        fired.append(text)

    client._on_intermediate_text_async = _cb

    with internal_side_call_scope():
        response = await client._inner_get_response(messages=[Message("user", ["hi"])], options={})

    assert response.text == "Let me check"
    assert fired == []

    # Control: the same response outside the scope does fire the callback.
    await client._inner_get_response(messages=[Message("user", ["hi"])], options={})
    assert fired == ["Let me check"]


@pytest.mark.asyncio
async def test_intermediate_text_suppressed_in_internal_side_call_streaming() -> None:
    client = _make_tool_call_wire_client()
    fired: list[str] = []
    client._on_intermediate_text_sync = fired.append

    with internal_side_call_scope():
        stream = client._inner_get_response(messages=[Message("user", ["hi"])], options={}, stream=True)
        final = await stream.get_final_response()

    assert final.text == "Let me check"
    assert fired == []

    # Control: outside the scope the result hook publishes on finalization.
    stream = client._inner_get_response(messages=[Message("user", ["hi"])], options={}, stream=True)
    await stream.get_final_response()
    assert fired == ["Let me check"]


# ──────────────── side-call exchange closure ─────────────────────────────
#
# A side call below the kernel opens its own exchange trace; nothing above it
# holds the handle, so a stream that never reaches a final response has to
# report its own end or the acquisition reads as one still in flight.


def _make_stream_wire_client(fail_with: type[BaseException] | None) -> Any:
    """Instrumented wire client whose stream ends in *fail_with* (or normally)."""

    class _RawStreamWireClient(BaseChatClient):
        def _inner_get_response(self, *, messages: Any, options: Any, stream: bool = False, **kwargs: Any) -> Any:
            del messages, options, stream, kwargs

            async def _updates() -> Any:
                yield ChatResponseUpdate(contents=[Content.from_text("partial")], role="assistant")
                if fail_with is not None:
                    raise fail_with()

            return ResponseStream(_updates(), finalizer=ChatResponse.from_updates)

    class _InstrumentedStreamWireClient(_IntermediateTextMixin, _RawStreamWireClient):
        pass

    return _InstrumentedStreamWireClient()


async def _drain_side_call_stream(client: Any, sink: FakeSink) -> None:
    with trajectory_scope(make_context(sink)), internal_side_call_scope(), side_call_scope(ActorRole.COMPLETER):
        stream = client._inner_get_response(messages=[Message("user", ["hi"])], options={}, stream=True)
        await stream.get_final_response()


@pytest.mark.asyncio
async def test_a_side_call_stream_that_errors_closes_its_own_exchange() -> None:
    sink = FakeSink()
    with pytest.raises(RuntimeError):
        await _drain_side_call_stream(_make_stream_wire_client(RuntimeError), sink)

    finished = sink.only(EventType.MODEL_EXCHANGE_FINISHED)
    assert finished.payload["outcome"] == ExchangeOutcome.ERROR
    assert finished.payload["error_code"] == "RuntimeError"
    assert finished.operation_id == sink.only(EventType.MODEL_EXCHANGE_STARTED).operation_id


@pytest.mark.asyncio
async def test_a_side_call_stream_dropped_mid_flight_closes_its_own_exchange() -> None:
    sink = FakeSink()
    with pytest.raises(asyncio.CancelledError):
        await _drain_side_call_stream(_make_stream_wire_client(asyncio.CancelledError), sink)

    assert sink.only(EventType.MODEL_EXCHANGE_FINISHED).payload["outcome"] == ExchangeOutcome.ABANDONED


@pytest.mark.asyncio
async def test_a_side_call_stream_that_finishes_reports_success_once() -> None:
    sink = FakeSink()
    await _drain_side_call_stream(_make_stream_wire_client(None), sink)

    assert sink.only(EventType.MODEL_EXCHANGE_FINISHED).payload["outcome"] == ExchangeOutcome.SUCCESS


@pytest.mark.asyncio
async def test_a_forwarded_exchange_is_left_to_the_loop_that_owns_it() -> None:
    """The loop closes its own exchanges with the outcome it knows (stalled,
    interrupted), so a failing stream must not close them first."""
    sink = FakeSink()
    context = make_context(sink).with_cycle(new_analytics_id()).with_exchange(new_analytics_id())
    client = _make_stream_wire_client(RuntimeError)
    with trajectory_scope(context), pytest.raises(RuntimeError):
        stream = client._inner_get_response(
            messages=[Message("user", ["hi"])],
            options={},
            stream=True,
            **{TRAJECTORY_EXCHANGE_KWARG: ExchangeTrace(context)},
        )
        await stream.get_final_response()

    assert sink.of_type(EventType.MODEL_EXCHANGE_STARTED)
    assert not sink.of_type(EventType.MODEL_EXCHANGE_FINISHED)


@pytest.mark.asyncio
async def test_a_per_request_model_override_cannot_grow_past_the_line_budget() -> None:
    """A profile's chat options are unrestricted, and the per-request model
    override is the one request fact only the start marker carries: one long
    enough to make that line unwritable would leave the terminal closing a
    start that became a gap."""
    sink = FakeSink()
    client = _make_stream_wire_client(None)
    with trajectory_scope(make_context(sink)), internal_side_call_scope(), side_call_scope(ActorRole.COMPLETER):
        stream = client._inner_get_response(
            messages=[Message("user", ["hi"])], options={"model": "m" * 200_000}, stream=True
        )
        await stream.get_final_response()

    # The sink applies the writer's own checks, so an unbounded override fails
    # here as the over-budget line it would have been.
    assert sink.only(EventType.MODEL_EXCHANGE_STARTED).payload["request_model"] == "m" * 256
    assert sink.only(EventType.MODEL_EXCHANGE_FINISHED).payload["outcome"] == ExchangeOutcome.SUCCESS


def test_openai_parse_response_accepts_millisecond_created_timestamp() -> None:
    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage

    from chrys.service.llm.openai_timestamps import openai_created_at_iso

    created_ms = 1_717_171_717_123
    chat_client = _make_chat_client()
    response = ChatCompletion(
        id="resp-1",
        object="chat.completion",
        created=created_ms,
        model="gpt-test",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hi"),
                finish_reason="stop",
            )
        ],
    )

    parsed = chat_client._parse_response_from_openai(response, {})

    assert type(parsed) is ChatResponse
    assert parsed.created_at == openai_created_at_iso(created_ms)
    assert response.created == created_ms


def test_openai_parse_response_update_accepts_millisecond_created_timestamp() -> None:
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
    from openai.types.chat.chat_completion_chunk import ChoiceDelta as ChunkChoiceDelta

    from chrys.service.llm.openai_timestamps import openai_created_at_iso

    created_ms = 1_717_171_717_123
    chat_client = _make_chat_client()
    chunk = ChatCompletionChunk(
        id="chunk-1",
        object="chat.completion.chunk",
        created=created_ms,
        model="gpt-test",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChunkChoiceDelta(role="assistant", content="hi"),
                finish_reason=None,
            )
        ],
    )

    parsed = chat_client._parse_response_update_from_openai(chunk)

    assert type(parsed) is ChatResponseUpdate
    assert parsed.created_at == openai_created_at_iso(created_ms)
    assert chunk.created == created_ms


def test_openai_prepare_options_sets_model_header_from_effective_model() -> None:
    chat_client = _make_chat_client()

    prepared = chat_client._prepare_options(
        [Message("user", ["hi"])],
        {
            "model": "gpt-final",
            "extra_headers": {
                "X-Team": "platform",
                "chrys-debug": "drop-me",
                MODEL_ID_HEADER: "wrong",
            },
        },
    )

    assert prepared["model"] == "gpt-final"
    assert prepared["extra_headers"][MODEL_ID_HEADER] == "gpt-final"
    assert prepared["extra_headers"]["X-Team"] == "platform"
    assert "chrys-debug" not in prepared["extra_headers"]


def test_openai_prepare_options_sets_model_header_from_default_model() -> None:
    chat_client = _make_chat_client()

    prepared = chat_client._prepare_options([Message("user", ["hi"])], {})

    assert prepared["model"] == "gpt-test"
    assert prepared["extra_headers"][MODEL_ID_HEADER] == "gpt-test"


def test_openai_prepare_options_sets_session_headers_from_session_id() -> None:
    chat_client = _make_chat_client(session_id="sess-123", parent_session_id="parent-123")

    prepared = chat_client._prepare_options(
        [Message("user", ["hi"])],
        {
            "extra_headers": {
                X_SESSION_ID_HEADER: "wrong",
                X_PARENT_SESSION_ID_HEADER: "wrong-parent",
                "X-Session-Id": "wrong-mixed-case",
                SESSION_ID_HEADER: "wrong",
                PARENT_SESSION_ID_HEADER: "wrong-parent",
            },
        },
    )

    assert prepared["extra_headers"][X_SESSION_ID_HEADER] == "sess-123"
    assert prepared["extra_headers"][SESSION_ID_HEADER] == "sess-123"
    assert prepared["extra_headers"][X_PARENT_SESSION_ID_HEADER] == "parent-123"
    assert prepared["extra_headers"][PARENT_SESSION_ID_HEADER] == "parent-123"
    assert "X-Session-Id" not in prepared["extra_headers"]


def test_openai_prepare_options_prefers_context_route_session_headers() -> None:
    chat_client = _make_chat_client(
        session_id="default-session",
        parent_session_id="default-parent",
        use_route_session_context=True,
    )
    session_token = llm_route_session_id.set("invocation-session")
    parent_token = llm_parent_session_id.set("root-session")
    try:
        prepared = chat_client._prepare_options([Message("user", ["hi"])], {})
    finally:
        llm_parent_session_id.reset(parent_token)
        llm_route_session_id.reset(session_token)

    assert prepared["extra_headers"][X_SESSION_ID_HEADER] == "invocation-session"
    assert prepared["extra_headers"][SESSION_ID_HEADER] == "invocation-session"
    assert prepared["extra_headers"][X_PARENT_SESSION_ID_HEADER] == "root-session"
    assert prepared["extra_headers"][PARENT_SESSION_ID_HEADER] == "root-session"


def test_openai_prepare_options_ignores_context_route_session_headers_by_default() -> None:
    chat_client = _make_chat_client(session_id="default-session", parent_session_id="default-parent")
    session_token = llm_route_session_id.set("invocation-session")
    parent_token = llm_parent_session_id.set("root-session")
    try:
        prepared = chat_client._prepare_options([Message("user", ["hi"])], {})
    finally:
        llm_parent_session_id.reset(parent_token)
        llm_route_session_id.reset(session_token)

    assert prepared["extra_headers"][X_SESSION_ID_HEADER] == "default-session"
    assert prepared["extra_headers"][SESSION_ID_HEADER] == "default-session"
    assert prepared["extra_headers"][X_PARENT_SESSION_ID_HEADER] == "default-parent"
    assert prepared["extra_headers"][PARENT_SESSION_ID_HEADER] == "default-parent"


def test_openai_prepare_options_rejects_non_ascii_extra_header_value() -> None:
    """Resolved chat_options.extra_headers hit the wire-charset gate at request time."""
    chat_client = _make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError) as info:
        chat_client._prepare_options(
            [Message("user", ["hi"])],
            {"extra_headers": {"X-Test": "秘密token"}},
        )

    message = str(info.value)
    assert "'X-Test'" in message
    assert "position 1" in message
    assert "秘密" not in message


def test_openai_prepare_options_rejects_non_ascii_model_override() -> None:
    chat_client = _make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError) as info:
        chat_client._prepare_options([Message("user", ["hi"])], {"model": "模型"})

    message = str(info.value)
    assert "Model ID" in message
    assert "U+6A21" in message


def test_openai_prepare_options_rejects_outer_space_header_value() -> None:
    chat_client = _make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError) as info:
        chat_client._prepare_options(
            [Message("user", ["hi"])],
            {"extra_headers": {"X-Test": "token "}},
        )

    message = str(info.value)
    assert "'X-Test'" in message
    assert "ends with a space" in message
    assert "token" not in message


def test_openai_prepare_options_rejects_non_string_extra_header_value() -> None:
    chat_client = _make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError) as info:
        chat_client._prepare_options(
            [Message("user", ["hi"])],
            {"extra_headers": {"X-Test": ["secret-value"]}},
        )

    message = str(info.value)
    assert "Header 'X-Test' value must be a string" in message
    assert "secret-value" not in message


def test_openai_prepare_options_rejects_non_string_extra_header_name() -> None:
    chat_client = _make_chat_client(session_id="sess-123")

    with pytest.raises(ValueError, match="Header name at position 1 must be a string"):
        chat_client._prepare_options(
            [Message("user", ["hi"])],
            {"extra_headers": {123: "value"}},
        )


def test_openai_prepare_options_allows_dropped_managed_header_with_unsafe_value() -> None:
    """A Chrys-managed header never reaches the wire, so its value is not validated."""
    chat_client = _make_chat_client(session_id="sess-123")

    prepared = chat_client._prepare_options(
        [Message("user", ["hi"])],
        {"extra_headers": {"chrys-debug": "值"}},
    )

    assert "chrys-debug" not in prepared["extra_headers"]


def test_set_chrys_request_headers_validates_on_early_return_path() -> None:
    """Even with no Chrys metadata to merge, caller headers still get the gate."""
    options: dict[str, Any] = {"extra_headers": {"X-Test": "值"}}

    with pytest.raises(ValueError) as info:
        _set_chrys_request_headers(options, session_id=None)

    message = str(info.value)
    assert "'X-Test'" in message
    assert "值" not in message


@pytest.mark.asyncio
async def test_openai_responses_prepare_options_sets_chrys_headers() -> None:
    chat_client = _make_responses_chat_client(session_id="sess-123", parent_session_id="parent-123")

    prepared = await chat_client._prepare_options(
        [Message("user", ["hi"])],
        {
            "model": "gpt-final",
            "extra_headers": {
                "X-Team": "platform",
                MODEL_ID_HEADER: "wrong",
                SESSION_ID_HEADER: "wrong",
                PARENT_SESSION_ID_HEADER: "wrong-parent",
                X_SESSION_ID_HEADER: "wrong",
                X_PARENT_SESSION_ID_HEADER: "wrong-parent",
            },
        },
    )

    assert prepared["model"] == "gpt-final"
    assert prepared["extra_headers"][MODEL_ID_HEADER] == "gpt-final"
    assert prepared["extra_headers"][SESSION_ID_HEADER] == "sess-123"
    assert prepared["extra_headers"][X_SESSION_ID_HEADER] == "sess-123"
    assert prepared["extra_headers"][PARENT_SESSION_ID_HEADER] == "parent-123"
    assert prepared["extra_headers"][X_PARENT_SESSION_ID_HEADER] == "parent-123"
    assert prepared["extra_headers"]["X-Team"] == "platform"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "required"])
async def test_openai_responses_preserves_allowed_tools_mode(mode: str) -> None:
    chat_client = _make_responses_chat_client()
    search_tool = FunctionTool(func=lambda query: query, name="search_docs", description="Search documentation")

    prepared = await chat_client._prepare_options(
        [Message("user", ["hi"])],
        {
            "tools": [search_tool],
            "tool_choice": {"mode": mode, "allowed_tools": ["search_docs"]},
        },
    )

    assert prepared["tool_choice"] == {
        "type": "allowed_tools",
        "mode": mode,
        "tools": [{"type": "function", "name": "search_docs"}],
    }


@pytest.mark.asyncio
async def test_openai_responses_required_without_allowlist_stays_plain_required() -> None:
    chat_client = _make_responses_chat_client()
    search_tool = FunctionTool(func=lambda query: query, name="search_docs", description="Search documentation")

    prepared = await chat_client._prepare_options(
        [Message("user", ["hi"])],
        {"tools": [search_tool], "tool_choice": {"mode": "required"}},
    )

    assert prepared["tool_choice"] == "required"


def test_openai_prepare_message_merges_text_and_tool_calls_for_vllm_replay() -> None:
    """Text + tool call from one kernel message must stay one wire message."""
    chat_client = _make_chat_client()

    prepared = chat_client._prepare_message_for_openai(
        Message(
            "assistant",
            [
                Content.from_text("I'll read it."),
                Content.from_function_call(call_id="call_bad", name="read_file", arguments="{}"),
            ],
        )
    )

    assert len(prepared) == 1
    assert prepared[0]["role"] == "assistant"
    assert prepared[0]["content"] == "I'll read it."
    assert prepared[0]["tool_calls"] == [
        {
            "id": "call_bad",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }
    ]


def test_openai_prepare_message_function_call_only_includes_empty_content() -> None:
    """Strict OpenAI-compatible servers reject assistant tool_calls without content."""
    chat_client = _make_chat_client()

    prepared = chat_client._prepare_message_for_openai(
        Message(
            "assistant",
            [Content.from_function_call(call_id="call_bad", name="read_file", arguments="{}")],
        )
    )

    assert prepared == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
            "content": "",
        }
    ]


def test_openai_prepare_message_repairs_malformed_tool_call_arguments() -> None:
    """Strict OpenAI-compatible servers reject malformed historical arguments."""
    chat_client = _make_chat_client()

    prepared = chat_client._prepare_message_for_openai(
        Message(
            "assistant",
            [Content.from_function_call(call_id="call_bad", name="glob", arguments="{")],
        )
    )

    assert prepared == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {"name": "glob", "arguments": "{}"},
                }
            ],
            "content": "",
        }
    ]


@pytest.mark.parametrize("arguments", [None, [], 123])
def test_openai_prepare_message_repairs_non_string_tool_call_arguments(arguments: object) -> None:
    """OpenAI Chat Completions requires function arguments to be a JSON string."""
    chat_client = _make_chat_client()

    prepared = chat_client._prepare_message_for_openai(
        Message(
            "assistant",
            [Content.from_function_call(call_id="call_bad", name="glob", arguments=arguments)],
        )
    )

    assert prepared[0]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_openai_prepare_message_repairs_only_bad_arguments_in_parallel_batch() -> None:
    """One malformed parallel call must not poison the whole vLLM replay."""
    chat_client = _make_chat_client()

    prepared = chat_client._prepare_message_for_openai(
        Message(
            "assistant",
            [
                Content.from_text("I'll search these."),
                Content.from_function_call(
                    call_id="call_a",
                    name="grep",
                    arguments='{"pattern": "needle", "path": "."}',
                ),
                Content.from_function_call(
                    call_id="call_b",
                    name="glob",
                    arguments="{",
                ),
                Content.from_function_call(
                    call_id="call_c",
                    name="read_file",
                    arguments='{"path": "README.md"}',
                ),
            ],
        )
    )

    assert len(prepared) == 1
    assert prepared[0]["role"] == "assistant"
    assert prepared[0]["content"] == "I'll search these."
    assert prepared[0]["tool_calls"] == [
        {
            "id": "call_a",
            "type": "function",
            "function": {"name": "grep", "arguments": '{"pattern": "needle", "path": "."}'},
        },
        {
            "id": "call_b",
            "type": "function",
            "function": {"name": "glob", "arguments": "{}"},
        },
        {
            "id": "call_c",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "README.md"}'},
        },
    ]


@pytest.mark.parametrize("text_first", [True, False])
def test_openai_canonicalizer_keeps_reasoning_aggregate_and_multimodal_fragment(text_first: bool) -> None:
    """The reasoning aggregate is complete by construction; the canonicalizer
    must not absorb the excluded image fragment into it (or vice versa)."""
    chat_client = _make_chat_client()
    text = Content.from_text("Look at this.")
    image = Content.from_uri(uri="https://example.com/img.png", media_type="image/png")
    reasoning = Content.from_text_reasoning(
        text="chain",
        additional_properties={"openai_reasoning_format": "reasoning_content"},
    )
    function_call = Content.from_function_call(call_id="call_1", name="lookup", arguments="{}")
    contents = [text, image, reasoning, function_call] if text_first else [image, text, reasoning, function_call]

    prepared = chat_client._prepare_message_for_openai(Message("assistant", contents))

    assert len(prepared) == 2
    image_message, aggregate = prepared
    assert image_message["content"][0]["type"] == "image_url"
    assert "reasoning_content" not in image_message
    assert "tool_calls" not in image_message
    assert aggregate["content"] == "Look at this."
    assert aggregate["reasoning_content"] == "chain"
    assert aggregate["tool_calls"][0]["function"]["name"] == "lookup"


def test_openai_canonicalizer_zero_text_reasoning_aggregate_not_merged_with_image() -> None:
    chat_client = _make_chat_client()

    prepared = chat_client._prepare_message_for_openai(
        Message(
            "assistant",
            [
                Content.from_uri(uri="https://example.com/img.png", media_type="image/png"),
                Content.from_text_reasoning(
                    text="chain",
                    additional_properties={"openai_reasoning_format": "reasoning_content"},
                ),
                Content.from_function_call(call_id="call_1", name="lookup", arguments="{}"),
            ],
        )
    )

    assert len(prepared) == 2
    image_message, aggregate = prepared
    assert image_message["content"][0]["type"] == "image_url"
    assert aggregate["content"] == ""
    assert aggregate["reasoning_content"] == "chain"
    assert aggregate["tool_calls"][0]["function"]["name"] == "lookup"


def test_openai_canonicalizer_keeps_vllm_reasoning_aggregate_separate_from_image() -> None:
    chat_client = _make_chat_client()

    prepared = chat_client._prepare_message_for_openai(
        Message(
            "assistant",
            [
                Content.from_uri(uri="https://example.com/img.png", media_type="image/png"),
                Content.from_text("Look at this."),
                Content.from_text_reasoning(
                    text="chain",
                    additional_properties={"openai_reasoning_format": "reasoning"},
                ),
                Content.from_function_call(call_id="call_1", name="lookup", arguments="{}"),
            ],
        )
    )

    assert len(prepared) == 2
    image_message, aggregate = prepared
    assert image_message["content"][0]["type"] == "image_url"
    assert "reasoning" not in image_message
    assert aggregate["content"] == "Look at this."
    assert aggregate["reasoning"] == "chain"
    assert aggregate["tool_calls"][0]["function"]["name"] == "lookup"


def test_openai_canonicalizer_preserves_multimodal_content_next_to_tool_calls() -> None:
    """A no-reasoning text+image+tool_calls history must not drop the image:
    non-empty str and list contents cannot combine, so the fragment moves
    ahead of the carrier, keeping the carrier adjacent to its tool results."""
    chat_client = _make_chat_client()

    prepared = chat_client._prepare_message_for_openai(
        Message(
            "assistant",
            [
                Content.from_text("Preface"),
                Content.from_function_call(call_id="call_1", name="lookup", arguments="{}"),
                Content.from_uri(uri="https://example.com/img.png", media_type="image/png"),
            ],
        )
    )

    assert len(prepared) == 2
    image_message, tool_call_message = prepared
    assert tool_call_message["content"] == "Preface"
    assert tool_call_message["tool_calls"][0]["function"]["name"] == "lookup"
    assert image_message["content"][0]["type"] == "image_url"
    assert "tool_calls" not in image_message


def test_openai_instrumented_no_reasoning_multimodal_keeps_carrier_adjacent_to_tool_result() -> None:
    """The tool result must directly follow the tool_calls carrier: a trailing
    image fragment may not strand between them in a full replayed history."""
    chat_client = _make_chat_client()

    prepared = chat_client._prepare_messages_for_openai(
        [
            Message("user", ["Q"]),
            Message(
                "assistant",
                [
                    Content.from_text("Preface"),
                    Content.from_function_call(call_id="call_1", name="lookup", arguments="{}"),
                    Content.from_uri(uri="https://example.com/img.png", media_type="image/png"),
                ],
            ),
            Message("tool", [Content.from_function_result(call_id="call_1", result="found")]),
        ]
    )

    assert [message["role"] for message in prepared] == ["user", "assistant", "assistant", "tool"]
    image_message, carrier, tool_result = prepared[1], prepared[2], prepared[3]
    assert image_message["content"][0]["type"] == "image_url"
    assert carrier["content"] == "Preface"
    assert carrier["tool_calls"][0]["function"]["name"] == "lookup"
    assert tool_result["tool_call_id"] == "call_1"


@pytest.mark.parametrize("image_first", [True, False])
def test_openai_instrumented_plural_keeps_aggregate_adjacent_to_tool_result(image_first: bool) -> None:
    chat_client = _make_chat_client()
    text = Content.from_text("Preface")
    function_call = Content.from_function_call(call_id="call_1", name="lookup", arguments="{}")
    image = Content.from_uri(uri="https://example.com/img.png", media_type="image/png")
    reasoning = Content.from_text_reasoning(
        text="chain",
        additional_properties={"openai_reasoning_format": "reasoning_content"},
    )
    contents = [image, text, function_call, reasoning] if image_first else [text, function_call, image, reasoning]

    prepared = chat_client._prepare_messages_for_openai(
        [
            Message("user", ["Q"]),
            Message("assistant", contents),
            Message("tool", [Content.from_function_result(call_id="call_1", result="found")]),
        ]
    )

    assert [message["role"] for message in prepared] == ["user", "assistant", "assistant", "tool"]
    image_message, aggregate, tool_result = prepared[1], prepared[2], prepared[3]
    assert image_message["content"][0]["type"] == "image_url"
    assert aggregate["content"] == "Preface"
    assert aggregate["reasoning_content"] == "chain"
    assert aggregate["tool_calls"][0]["function"]["name"] == "lookup"
    assert tool_result["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_openai_tool_loop_replays_argument_error_with_canonical_tool_call_message() -> None:
    """Malformed tool args should not poison the next vLLM/OpenAI-compatible request."""

    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function

    from chrys.kernel import FunctionTool
    from chrys.service.llm.instrumented import create_instrumented_openai_client

    captured_requests: list[dict[str, Any]] = []

    class _FakeCompletions:
        def __init__(self) -> None:
            self._call_count = 0

        async def create(self, stream: bool = False, **kwargs: Any) -> ChatCompletion:
            captured_requests.append(kwargs)
            self._call_count += 1
            if self._call_count == 1:
                return ChatCompletion(
                    id="resp-1",
                    object="chat.completion",
                    created=1234567890,
                    model="vllm-model",
                    choices=[
                        Choice(
                            index=0,
                            message=ChatCompletionMessage(
                                role="assistant",
                                content="I'll read it.",
                                tool_calls=[
                                    ChatCompletionMessageToolCall(
                                        id="call_bad",
                                        type="function",
                                        function=Function(name="read_file", arguments="{}"),
                                    )
                                ],
                            ),
                            finish_reason="tool_calls",
                        )
                    ],
                )
            return ChatCompletion(
                id="resp-2",
                object="chat.completion",
                created=1234567891,
                model="vllm-model",
                choices=[
                    Choice(
                        index=0,
                        message=ChatCompletionMessage(role="assistant", content="Recovered."),
                        finish_reason="stop",
                    )
                ],
            )

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeAsyncOpenAI:
        def __init__(self) -> None:
            self.chat = _FakeChat()

    def read_file(path: str) -> str:
        return path

    client = create_instrumented_openai_client(model_id="vllm-model", client=_FakeAsyncOpenAI())
    tool = FunctionTool(name="read_file", description="Read a file", func=read_file)

    await client.get_response([Message("user", ["read x"])], options={"tools": [tool]})

    assert len(captured_requests) == 2
    assert captured_requests[1]["messages"] == [
        {"role": "user", "content": "read x"},
        {
            "role": "assistant",
            "content": "I'll read it.",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_bad",
            "content": ("Error: Invalid arguments for 'read_file': missing 'path'. Expected: {path: string}."),
        },
    ]


def test_openai_canonicalizes_messages_after_session_json_round_trip() -> None:
    """Reloaded session.json messages must produce the same canonical wire shape.

    Chrys persists kernel Messages via ``Message.to_dict()`` and restores
    them via ``Message.from_dict()``. The per-Message canonicalization scope
    only holds if a single restored Message still bundles text and the
    matching function_call as separate Contents on one Message — the same
    shape AF emits when parsing a live OpenAI choice.  Round-tripping the
    poisoned pair through serialization here pins that invariant: a new
    user message replayed against the reloaded history must still produce
    the merged, ``content``-bearing assistant tool-call wire message that
    vLLM accepts.
    """
    chat_client = _make_chat_client()

    poisoned_assistant = Message(
        "assistant",
        [
            Content.from_text("I'll read it."),
            Content.from_function_call(call_id="call_bad", name="read_file", arguments="{}"),
        ],
    )
    tool_error = Message(
        "tool",
        [Content.from_function_result(call_id="call_bad", result="Error: Argument parsing failed.")],
    )

    # Mimic chrys's persistence pipeline (serializers.py:serialize_message /
    # deserialize_message): Message -> dict -> Message.
    persisted = [
        Message("user", ["read x"]).to_dict(),
        poisoned_assistant.to_dict(),
        tool_error.to_dict(),
    ]
    reloaded = [Message.from_dict(d) for d in persisted]

    # Replay: append a new user message to the reloaded history and prep.
    history_for_replay = [*reloaded, Message("user", ["please retry"])]
    prepared = chat_client._prepare_messages_for_openai(history_for_replay)

    assert prepared == [
        {"role": "user", "content": "read x"},
        {
            "role": "assistant",
            "content": "I'll read it.",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_bad", "content": "Error: Argument parsing failed."},
        {"role": "user", "content": "please retry"},
    ]


def test_openai_canonicalizes_malformed_arguments_after_session_json_round_trip() -> None:
    """Reloaded bad tool-call arguments must not make the next vLLM request 400."""
    chat_client = _make_chat_client()

    poisoned_assistant = Message(
        "assistant",
        [
            Content.from_text("I'll search."),
            Content.from_function_call(call_id="call_bad", name="glob", arguments="{"),
        ],
    )
    tool_error = Message(
        "tool",
        [Content.from_function_result(call_id="call_bad", result="Error: Argument parsing failed.")],
    )

    persisted = [
        Message("user", ["find files"]).to_dict(),
        poisoned_assistant.to_dict(),
        tool_error.to_dict(),
    ]
    assert persisted[1] == {
        "type": "message",
        "role": "assistant",
        "contents": [
            {"type": "text", "text": "I'll search.", "additional_properties": {}},
            {
                "type": "function_call",
                "call_id": "call_bad",
                "name": "glob",
                "arguments": "{",
                "additional_properties": {},
            },
        ],
        "additional_properties": {},
    }
    reloaded = [Message.from_dict(d) for d in persisted]

    history_for_replay = [*reloaded, Message("user", ["please continue"])]
    prepared = chat_client._prepare_messages_for_openai(history_for_replay)

    assert prepared == [
        {"role": "user", "content": "find files"},
        {
            "role": "assistant",
            "content": "I'll search.",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {"name": "glob", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_bad", "content": "Error: Argument parsing failed."},
        {"role": "user", "content": "please continue"},
    ]


def test_openai_repairs_malformed_arguments_with_deepseek_client_cls() -> None:
    """The instrumented wrapper must repair arguments after DeepSeek-specific prep."""
    from chrys.service.llm.deepseek import DeepSeekChatCompletionClient

    chat_client = _make_chat_client(chat_client_cls=DeepSeekChatCompletionClient)

    prepared = chat_client._prepare_message_for_openai(
        Message(
            "assistant",
            [
                Content.from_text("I'll search."),
                Content.from_function_call(call_id="call_bad", name="glob", arguments="{"),
                Content.from_text_reasoning(
                    text=None,
                    protected_data='"reasoning"',
                    additional_properties={"openai_reasoning_format": "reasoning_content"},
                ),
            ],
        )
    )

    assert prepared == [
        {
            "role": "assistant",
            "content": "I'll search.",
            "tool_calls": [
                {
                    "id": "call_bad",
                    "type": "function",
                    "function": {"name": "glob", "arguments": "{}"},
                }
            ],
            "reasoning_content": "reasoning",
        }
    ]


def test_anthropic_prepare_options_sets_model_header_from_effective_model() -> None:
    from chrys.service.llm.instrumented import create_instrumented_anthropic_client

    chat_client = create_instrumented_anthropic_client(
        model_id="claude-default",
        anthropic_client=object(),
    )

    prepared = chat_client._prepare_options(
        [Message("user", ["hi"])],
        {
            "model": "claude-final",
            "extra_headers": {
                "chrys-debug": "drop-me",
                MODEL_ID_HEADER: "wrong",
            },
        },
    )

    assert prepared["model"] == "claude-final"
    assert prepared["extra_headers"][MODEL_ID_HEADER] == "claude-final"
    assert "chrys-debug" not in prepared["extra_headers"]


def test_anthropic_response_format_uses_ga_output_config() -> None:
    class StructuredPayload(BaseModel):
        answer: str

    chat_client = create_instrumented_anthropic_client(
        model_id="claude-default",
        anthropic_client=object(),
    )

    prepared = chat_client._prepare_options(
        [Message("user", ["hi"])],
        {"response_format": StructuredPayload},
    )

    assert "output_format" not in prepared
    assert prepared["output_config"]["format"] == {
        "type": "json_schema",
        "schema": {
            "properties": {"answer": {"title": "Answer", "type": "string"}},
            "required": ["answer"],
            "title": "StructuredPayload",
            "type": "object",
            "additionalProperties": False,
        },
    }
    assert "structured-outputs-2025-11-13" not in prepared["betas"]


def test_anthropic_response_format_preserves_output_config_without_mutating_caller() -> None:
    class StructuredPayload(BaseModel):
        answer: str

    chat_client = create_instrumented_anthropic_client(
        model_id="claude-default",
        anthropic_client=object(),
    )
    caller_output_config = {"effort": "high"}

    prepared = chat_client._prepare_options(
        [Message("user", ["hi"])],
        {"response_format": StructuredPayload, "output_config": caller_output_config},
    )

    assert prepared["output_config"]["effort"] == "high"
    assert prepared["output_config"]["format"]["type"] == "json_schema"
    assert caller_output_config == {"effort": "high"}


def test_anthropic_response_format_conflicts_with_explicit_output_config_format() -> None:
    class StructuredPayload(BaseModel):
        answer: str

    chat_client = create_instrumented_anthropic_client(
        model_id="claude-default",
        anthropic_client=object(),
    )

    with pytest.raises(ChatClientInvalidRequestException, match="cannot be combined"):
        chat_client._prepare_options(
            [Message("user", ["hi"])],
            {
                "response_format": StructuredPayload,
                "output_config": {"format": {"type": "json_schema", "schema": {}}},
            },
        )


def test_anthropic_without_response_format_does_not_add_output_config() -> None:
    chat_client = create_instrumented_anthropic_client(
        model_id="claude-default",
        anthropic_client=object(),
    )

    prepared = chat_client._prepare_options([Message("user", ["hi"])], {})

    assert "output_config" not in prepared
    assert "output_format" not in prepared


def test_anthropic_prepare_options_sets_session_headers_from_session_id() -> None:
    from chrys.service.llm.instrumented import create_instrumented_anthropic_client

    chat_client = create_instrumented_anthropic_client(
        model_id="claude-default",
        session_id="sess-456",
        parent_session_id="parent-456",
        anthropic_client=object(),
    )

    prepared = chat_client._prepare_options(
        [Message("user", ["hi"])],
        {
            "extra_headers": {
                X_SESSION_ID_HEADER: "wrong",
                X_PARENT_SESSION_ID_HEADER: "wrong-parent",
                "X-Session-Id": "wrong-mixed-case",
                SESSION_ID_HEADER: "wrong",
                PARENT_SESSION_ID_HEADER: "wrong-parent",
            },
        },
    )

    assert prepared["extra_headers"][X_SESSION_ID_HEADER] == "sess-456"
    assert prepared["extra_headers"][SESSION_ID_HEADER] == "sess-456"
    assert prepared["extra_headers"][X_PARENT_SESSION_ID_HEADER] == "parent-456"
    assert prepared["extra_headers"][PARENT_SESSION_ID_HEADER] == "parent-456"
    assert "X-Session-Id" not in prepared["extra_headers"]


def test_anthropic_parse_usage_counts_cache_tokens_as_context_input() -> None:
    from chrys.service.llm.instrumented import create_instrumented_anthropic_client

    chat_client = create_instrumented_anthropic_client(
        model_id="claude-default",
        anthropic_client=object(),
    )
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=20,
        cache_read_input_tokens=30,
    )

    details = chat_client._parse_usage_from_anthropic(usage)

    assert details is not None
    assert details["input_token_count"] == 150
    assert details["output_token_count"] == 50
    assert details["anthropic.cache_creation_input_tokens"] == 20
    assert details["anthropic.cache_read_input_tokens"] == 30
    assert details["cache_creation_input_token_count"] == 20
    assert details["cache_read_input_token_count"] == 30
    assert details["context_input_token_floor"] == 120


def test_anthropic_parse_usage_preserves_uncached_input_tokens() -> None:
    from chrys.service.llm.instrumented import create_instrumented_anthropic_client

    chat_client = create_instrumented_anthropic_client(
        model_id="claude-default",
        anthropic_client=object(),
    )
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )

    details = chat_client._parse_usage_from_anthropic(usage)

    assert details is not None
    assert details["input_token_count"] == 100
    assert details["output_token_count"] == 50


def test_anthropic_parse_usage_does_not_invent_input_for_sparse_delta() -> None:
    """Sparse stream deltas must not overwrite message_start input counts."""
    from chrys.service.llm.instrumented import create_instrumented_anthropic_client

    chat_client = create_instrumented_anthropic_client(
        model_id="claude-default",
        anthropic_client=object(),
    )
    usage = SimpleNamespace(
        input_tokens=None,
        output_tokens=50,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=30,
    )

    details = chat_client._parse_usage_from_anthropic(usage)

    assert details is not None
    assert "input_token_count" not in details
    assert details["output_token_count"] == 50
    assert details["anthropic.cache_read_input_tokens"] == 30
    assert details["cache_read_input_token_count"] == 30


def test_anthropic_stream_usage_exposes_final_context_occupancy() -> None:
    """Terminal hosted-loop usage excludes cumulative internal cache reads from context."""
    from chrys.service.llm.anthropic_chat import _AnthropicStreamState
    from chrys.service.llm.instrumented import create_instrumented_anthropic_client

    chat_client = create_instrumented_anthropic_client(
        model_id="claude-default",
        anthropic_client=object(),
    )
    state = _AnthropicStreamState(
        pending_function_calls={},
        hosted_tool_indices=set(),
        hosted_tool_calls={},
        hosted_argument_deltas={},
        deferred_updates={},
        defer_from_index=None,
    )
    start_usage = SimpleNamespace(
        input_tokens=6,
        output_tokens=1,
        cache_creation_input_tokens=36_301,
        cache_read_input_tokens=100,
    )
    chat_client._process_stream_event(
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=start_usage,
                id="msg_1",
                content=[],
                model="claude-default",
                stop_reason=None,
            ),
        ),
        state,
    )
    final_usage = SimpleNamespace(
        input_tokens=9,
        output_tokens=1_657,
        cache_creation_input_tokens=41_173,
        cache_read_input_tokens=111_175,
    )

    update = chat_client._process_stream_event(
        SimpleNamespace(
            type="message_delta",
            usage=final_usage,
            delta=SimpleNamespace(stop_reason="end_turn"),
        ),
        state,
    )

    assert update is not None
    assert len(update.contents) == 1
    details = update.contents[0].usage_details
    assert details is not None
    assert details["input_token_count"] == 152_357
    assert details["context_input_token_floor"] == 41_182
    assert details["context_input_token_count"] == 41_282


def test_anthropic_prepare_options_drops_unsigned_thinking_blocks() -> None:
    from chrys.service.llm.instrumented import create_instrumented_anthropic_client

    chat_client = create_instrumented_anthropic_client(
        model_id="claude-default",
        anthropic_client=object(),
    )

    prepared = chat_client._prepare_options(
        [
            Message("user", ["describe @one.png"]),
            Message(
                "assistant",
                [
                    Content.from_text_reasoning(text="unsigned private reasoning"),
                    Content.from_text("visible answer"),
                ],
            ),
            Message(
                "user",
                [
                    "compare these",
                    Content.from_data(data=b"one", media_type="image/png"),
                    Content.from_data(data=b"two", media_type="image/jpeg"),
                ],
            ),
        ],
        {},
    )

    assistant = prepared["messages"][1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == [{"type": "text", "text": "visible answer"}]
    image_blocks = [block for block in prepared["messages"][2]["content"] if block["type"] == "image"]
    assert [block["source"]["media_type"] for block in image_blocks] == ["image/png", "image/jpeg"]


def test_anthropic_prepare_options_drops_empty_messages_after_unsigned_thinking_filter() -> None:
    from chrys.service.llm.instrumented import create_instrumented_anthropic_client

    chat_client = create_instrumented_anthropic_client(
        model_id="claude-default",
        anthropic_client=object(),
    )

    prepared = chat_client._prepare_options(
        [
            Message("user", ["before"]),
            Message("assistant", [Content.from_text_reasoning(text="unsigned private reasoning")]),
            Message("assistant", [""]),
            Message("user", ["after"]),
        ],
        {},
    )

    assert prepared["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "before"}]},
        {"role": "user", "content": [{"type": "text", "text": "after"}]},
    ]


def test_anthropic_prepare_options_preserves_signed_thinking_blocks() -> None:
    from chrys.service.llm.instrumented import create_instrumented_anthropic_client

    chat_client = create_instrumented_anthropic_client(
        model_id="claude-default",
        anthropic_client=object(),
    )

    prepared = chat_client._prepare_options(
        [
            Message(
                "assistant",
                [
                    Content.from_text_reasoning(text="signed private reasoning", protected_data="sig-123"),
                    Content.from_text("visible answer"),
                ],
            )
        ],
        {},
    )

    assert prepared["messages"][0]["content"][0] == {
        "type": "thinking",
        "thinking": "signed private reasoning",
        "signature": "sig-123",
    }
    assert prepared["messages"][0]["content"][1] == {"type": "text", "text": "visible answer"}


def test_anthropic_prepare_options_keeps_redacted_and_drops_unsigned_thinking() -> None:
    from chrys.service.llm.instrumented import create_instrumented_anthropic_client

    chat_client = create_instrumented_anthropic_client(
        model_id="claude-default",
        anthropic_client=object(),
    )
    prepared = chat_client._prepare_options(
        [
            Message(
                "assistant",
                [
                    Content.from_text_reasoning(
                        protected_data="opaque-redacted",
                        additional_properties={"anthropic_redacted_thinking": True},
                    ),
                    Content.from_text_reasoning(text="unsigned private reasoning"),
                ],
            )
        ],
        {},
    )

    assert prepared["messages"][0]["content"] == [{"type": "redacted_thinking", "data": "opaque-redacted"}]


def test_integration_choices_none_raises_with_gateway_error_body() -> None:
    """A 200 response carrying a gateway error envelope surfaces as ChatClientException."""
    from openai.types.chat.chat_completion import ChatCompletion

    chat_client = _make_chat_client()
    bad = ChatCompletion.model_construct(
        id="resp-bad",
        choices=None,
        created=0,
        model="gpt-test",
        object="chat.completion",
        error={"message": "rate limit exceeded", "code": 429},
    )

    with pytest.raises(ChatClientException) as exc_info:
        chat_client._parse_response_from_openai(bad, {})

    msg = str(exc_info.value)
    assert "no 'choices' field" in msg
    assert "rate limit exceeded" in msg
    assert "429" in msg


def test_integration_valid_response_delegates_to_super() -> None:
    """A valid (empty-choices) response should pass through and produce a ChatResponse."""
    from openai.types.chat.chat_completion import ChatCompletion

    chat_client = _make_chat_client()
    valid = ChatCompletion.model_construct(
        id="resp-ok",
        choices=[],
        created=0,
        model="gpt-test",
        object="chat.completion",
        usage=None,
    )

    result = chat_client._parse_response_from_openai(valid, {})
    assert result.response_id == "resp-ok"
    assert result.messages == []


def test_integration_class_uses_overridden_method() -> None:
    """Sanity check that the subclass — not the parent — defines the active method."""
    chat_client = _make_chat_client()
    cls = type(chat_client.inner.inner)
    assert "_parse_response_from_openai" in cls.__dict__
    assert cls.__name__ == "_InstrumentedOpenAIChatCompletionClient"


# ---------------------------------------------------------------------------
# Cache-token preservation
# ---------------------------------------------------------------------------


def test_parse_usage_preserves_cached_tokens_zero() -> None:
    """OpenAI ``cached_tokens=0`` must survive parsing so the UI shows ``0``,
    not ``-``. A naive ``if tokens := ...:`` truthiness check would drop it."""
    from openai.types.completion_usage import CompletionUsage

    chat_client = _make_chat_client()
    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    )

    details = chat_client._parse_usage_from_openai(usage)
    assert details["prompt/cached_tokens"] == 0
    assert details["cache_read_input_token_count"] == 0
    assert details["completion/reasoning_tokens"] == 0
    assert details["reasoning_output_token_count"] == 0


def test_responses_parse_usage_preserves_cached_tokens_zero() -> None:
    """Responses ``cached_tokens=0`` must survive parsing."""
    from openai.types.responses import ResponseUsage

    chat_client = _make_responses_chat_client()
    usage = ResponseUsage.model_validate(
        {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 50,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 1050,
        }
    )

    details = chat_client._parse_usage_from_openai(usage)
    assert details["openai.cached_input_tokens"] == 0
    assert details["cache_read_input_token_count"] == 0
    assert details["openai.reasoning_tokens"] == 0
    assert details["reasoning_output_token_count"] == 0


def test_parse_usage_preserves_cached_tokens_nonzero() -> None:
    """Override must not overwrite a non-zero value the base parser already set."""
    from openai.types.completion_usage import CompletionUsage

    chat_client = _make_chat_client()
    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
            "prompt_tokens_details": {"cached_tokens": 256},
        }
    )

    details = chat_client._parse_usage_from_openai(usage)
    assert details["prompt/cached_tokens"] == 256
    assert details["cache_read_input_token_count"] == 256


def test_parse_usage_omits_cache_key_when_provider_does_not_report() -> None:
    """Absent ``prompt_tokens_details`` must stay absent — ``None`` semantics."""
    from openai.types.completion_usage import CompletionUsage

    chat_client = _make_chat_client()
    usage = CompletionUsage.model_validate(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
        }
    )

    details = chat_client._parse_usage_from_openai(usage)
    assert "prompt/cached_tokens" not in details
    assert "cache_read_input_token_count" not in details
