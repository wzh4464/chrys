# Copyright (c) 2026 Chrys. All rights reserved.

"""Acceptance tests for the chrys-owned tool loop.

Pins the loop invariants in ``chrys.kernel.loop`` against the production composition
``ToolLoopLayer(ChatMiddlewareLayer(inner))``, plus drift pins for the private
surface the loop depends on (validation helpers, loop defaults,
``FunctionTool`` internals).
"""

from __future__ import annotations

import asyncio
import datetime
import decimal
import gc
import inspect
import json
import logging
import random
from threading import Event as ThreadEvent
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

import pytest
from pydantic import AliasGenerator, BaseModel, ConfigDict, Field, computed_field, field_serializer, model_validator

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.observability.gate import TELEMETRY_GATE
from chrys.foundation.retry import HistorySnapshot, StreamRetryLoop
from chrys.foundation.tool_execution_stamp import EXECUTION_STAMP_KEY, write_execution_stamp
from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY
from chrys.foundation.tool_result_metadata import (
    TOOL_ERROR_KIND_METADATA_KEY,
    TOOL_ERROR_MESSAGE_METADATA_KEY,
    TOOL_FAILED_METADATA_KEY,
    TOOL_INTERRUPTED_METADATA_KEY,
    TOOL_POST_PROCESSING_INTERRUPTED_METADATA_KEY,
    TOOL_RESULT_METADATA_KEY,
)
from chrys.foundation.trajectory.context import TRAJECTORY_CONTEXT_KWARG
from chrys.foundation.trajectory.event_types import EventType as TrajectoryEventType
from chrys.foundation.trajectory.event_types import ToolOutcome
from chrys.foundation.trajectory.ids import is_valid_analytics_id
from chrys.foundation.trajectory.metadata import (
    ANALYTICS_ITEM_ID_KEY,
    OPERATION_ID_KEY,
    TOOL_RESULT_ITEM_ID_METADATA_KEY,
)
from chrys.foundation.trajectory_timing import TRAJECTORY_TIMING_KEY
from chrys.foundation.util.sub_agent_context import SUB_AGENT_RESULT_COMMIT_CALLBACK_KEY
from chrys.kernel import SKIP_PARSING, AgentSession, FunctionTool, tool
from chrys.kernel.identity import WeakIdentityRegistry
from chrys.kernel.loop import (
    _MAX_ITERATIONS_FALLBACK_TEXT,
    DEFAULT_MAX_CONSECUTIVE_ERRORS,
    DEFAULT_MAX_ITERATIONS,
    ConsumedInjectionMessageProbe,
    LoopRecorder,
    ToolLoopLayer,
    _extract_function_calls,
    _middleware_arguments_equal,
    _record_wire_request_content_identities,
    _strip_unexecutable_calls_from_update,
)
from chrys.kernel.middleware import (
    ChatContext,
    ChatMiddleware,
    ChatMiddlewareLayer,
    FunctionInvocationContext,
    FunctionMiddleware,
    MiddlewareTermination,
)
from chrys.kernel.tools import (
    SyncToolCancelledAfterCompletion,
    _matches_json_schema_type,
    _validate_arguments_against_schema,
)
from chrys.kernel.types import (
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)
from chrys.service.session.history import SessionHistoryManager
from chrys.service.state.serializers import serialized_message_payload
from tests.service.trajectory._fakes import FakeSink, make_context
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user(text: str = "hi") -> Message:
    return Message(role="user", contents=[text])


def _text_response(text: str = "done", **kwargs: Any) -> ChatResponse:
    return ChatResponse(messages=[Message(role="assistant", contents=[text])], **kwargs)


def _call_response(*calls: tuple[str, str, dict[str, Any]], **kwargs: Any) -> ChatResponse:
    """Assistant response containing function_call contents (call_id, name, args)."""
    contents = [Content.from_function_call(call_id=cid, name=name, arguments=args) for cid, name, args in calls]
    return ChatResponse(messages=[Message(role="assistant", contents=contents)], **kwargs)


def _text_update(text: str) -> ChatResponseUpdate:
    return ChatResponseUpdate(contents=[{"type": "text", "text": text}], role="assistant")


def _call_update(cid: str, name: str, args: dict[str, Any]) -> ChatResponseUpdate:
    return ChatResponseUpdate(
        contents=[Content.from_function_call(call_id=cid, name=name, arguments=args)], role="assistant"
    )


class _ScriptedClient:
    """Innermost fake wire client.

    Non-stream turns are ``ChatResponse`` objects; stream turns are lists of
    ``ChatResponseUpdate``. Records every call's messages/options/kwargs.
    """

    def __init__(self, turns: list[Any], *, result_hook: Callable[[ChatResponse], ChatResponse] | None = None) -> None:
        self.turns = list(turns)
        self.calls: list[dict[str, Any]] = []
        self.result_hook = result_hook

    def get_response(
        self,
        messages: Any,
        *,
        stream: bool = False,
        options: Any = None,
        function_invocation_kwargs: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.calls.append(
            {
                "messages": list(messages),
                "options": dict(options or {}),
                "kwargs": dict(kwargs),
                "stream": stream,
            }
        )
        turn = self.turns.pop(0)
        if not stream:

            async def _resolve() -> ChatResponse:
                return turn

            return _resolve()

        async def _gen() -> Any:
            for update in turn:
                yield update

        rs: ResponseStream[ChatResponseUpdate, ChatResponse] = ResponseStream(
            _gen(), finalizer=ChatResponse.from_updates
        )
        if self.result_hook is not None:
            rs.with_result_hook(self.result_hook)
        return rs


def _stack(
    turns: list[Any],
    *,
    result_hook: Callable[[ChatResponse], ChatResponse] | None = None,
    **layer_kwargs: Any,
) -> tuple[ToolLoopLayer, _ScriptedClient]:
    """The production composition: ToolLoopLayer(ChatMiddlewareLayer(wire)).

    The returned layer transparently checks the transcript invariants on
    every final response (see ``InvariantCheckedToolLoopLayer``).
    """
    wire = _ScriptedClient(turns, result_hook=result_hook)
    return InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire), **layer_kwargs), wire


def _make_tool(events: list[str] | None = None, *, name: str = "echo") -> FunctionTool:
    @tool(name=name)
    async def echo(text: str) -> str:
        if events is not None:
            events.append(f"tool:{name}:{text}")
        return f"echo:{text}"

    return echo


class _NullableValue(BaseModel):
    value: str | None


class _DivergentDefaults(BaseModel):
    count: int = 7


class _SerializedNullable(BaseModel):
    """Field serializer owns one field while a safe root null remains restorable."""

    safe: str | None
    value: str

    @field_serializer("value")
    def _serialize_value(self, value: str) -> dict[str, Any]:
        return {"serialized": value}


class _ComputedGhost(BaseModel):
    """Computed field appears in every dump but is never caller-supplied."""

    count: int = 1

    @computed_field
    @property
    def ghost(self) -> int | None:
        return None


class _OptionalChild(BaseModel):
    value: str | None = None


class _ReorderingItems(BaseModel):
    """Same-shape field serializer: equal length no longer proves alignment."""

    items: list[_OptionalChild]

    @field_serializer("items")
    def _reverse(self, items: list[_OptionalChild]) -> list[_OptionalChild]:
        return list(reversed(items))


class _TupleItems(BaseModel):
    items: tuple[_NullableValue, ...]


class _AliasedNullable(BaseModel):
    """``serialize_by_alias``: dumps are keyed by the wire alias, not the field name."""

    model_config = ConfigDict(serialize_by_alias=True)

    value: str | None = Field(alias="wire")


class _Round15SplitAlias(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    value: str | None = Field(validation_alias="inputKey", serialization_alias="callableKey")


class _Round15ValidationAliasOnly(BaseModel):
    value: str | None = Field(validation_alias="inputKey")


class _Round15GeneratedValidationAlias(BaseModel):
    model_config = ConfigDict(
        alias_generator=AliasGenerator(validation_alias=str.upper),
        serialize_by_alias=True,
        validate_by_name=True,
    )

    value: str | None = Field(serialization_alias="callableKey")


class _Round15PlainArguments(BaseModel):
    value: str | None


class _Round16RewriteArguments(BaseModel):
    value: Any


class _Round17ConsumingArguments(BaseModel):
    x: int

    @model_validator(mode="before")
    @classmethod
    def _consume(cls, data: Any) -> dict[str, Any]:
        return {"x": data.pop("x")}


class _Round17NestedConsumingArguments(BaseModel):
    items: list[dict[str, int]]

    @model_validator(mode="before")
    @classmethod
    def _consume_nested(cls, data: Any) -> dict[str, Any]:
        item = data["items"].pop(0)
        return {"items": [{"x": item.pop("x")}]}


class _Round17OrderFastPathArguments(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    first: int = Field(validation_alias="inputFirst", serialization_alias="callableFirst")
    second: int


class _MemberNullable(TypedDict):
    value: str | None


class _MemberNullableHolder(BaseModel):
    inner: _MemberNullable


def _result_contents(response: ChatResponse) -> list[Content]:
    return [item for msg in response.messages for item in msg.contents if item.type == "function_result"]


def _assert_actionable_argument_error(result: Content, tool_name: str, *expected_parts: str) -> str:
    result_text = str(result.result)
    assert result_text.startswith(f"Error: Invalid arguments for '{tool_name}':")
    assert "Expected:" in result_text
    for expected in expected_parts:
        assert expected in result_text
    assert result.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_parsing"
    assert result.additional_properties[TOOL_ERROR_MESSAGE_METADATA_KEY] == result_text.removeprefix("Error: ")
    return result_text


class _ProbeFunction(FunctionMiddleware):
    def __init__(self) -> None:
        self.contexts: list[FunctionInvocationContext] = []

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        self.contexts.append(context)
        await call_next()


class _ExecutionStampFunction(FunctionMiddleware):
    def __init__(self, completion_order: list[str]) -> None:
        self._completion_order = completion_order

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        outcome: Literal["ok", "error"] = "ok"
        error_kind: str | None = None
        try:
            await call_next()
        except Exception as exc:
            outcome = "error"
            error_kind = type(exc).__name__
            raise
        finally:
            value = str(context.arguments["value"])
            self._completion_order.append(value)
            write_execution_stamp(context.metadata, context.arguments, outcome=outcome, error_kind=error_kind)


class _ProbeChat(ChatMiddleware):
    def __init__(self) -> None:
        self.calls = 0

    async def process(self, context: Any, call_next: Callable[[], Awaitable[None]]) -> None:
        self.calls += 1
        await call_next()


class _TerminateFunction(FunctionMiddleware):
    def __init__(self, *, result: Any = None, exc_result: Any = None) -> None:
        self._result = result
        self._exc_result = exc_result

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        if self._result is not None:
            context.result = self._result
        raise MiddlewareTermination("stop", result=self._exc_result)


@pytest.mark.asyncio
async def test_tool_result_ceiling_bounds_transcript_and_raw_journal() -> None:
    @tool(name="large")
    async def large() -> str:
        return "HEAD" + "x" * 10_000 + "TAIL"

    recorder = LoopRecorder()
    layer, _wire = _stack(
        [_call_response(("c1", "large", {})), _text_response()],
        tool_result_ceiling_tokens=100,
    )

    response = await layer.get_response(
        [_user()],
        options={"tools": [large]},
        client_kwargs={"loop_recorder": recorder},
    )

    result = _result_contents(response)[0]
    assert isinstance(result.result, str)
    assert "truncated" in result.result
    assert len(result.result) < 1_000
    journal_results = [
        content
        for message in recorder.loop_messages or []
        for content in message.contents
        if content.type == "function_result"
    ]
    assert len(journal_results) == 1
    assert journal_results[0].result == result.result


@pytest.mark.asyncio
async def test_tool_result_ceiling_bounds_prebuilt_middleware_termination() -> None:
    prebuilt = Content.from_function_result("cached", result="x" * 10_000)
    layer, _wire = _stack(
        [_call_response(("c1", "echo", {"text": "unused"}))],
        tool_result_ceiling_tokens=100,
    )

    response = await layer.get_response(
        [_user()],
        options={"tools": [_make_tool()]},
        middleware=[_TerminateFunction(exc_result=prebuilt)],
    )

    result = _result_contents(response)[0]
    assert result is not prebuilt
    assert result.call_id == "c1"
    assert isinstance(result.result, str)
    assert "truncated" in result.result
    assert len(result.result) < 1_000


# ---------------------------------------------------------------------------
# Stack composition and delegation
# ---------------------------------------------------------------------------


class TestStackAndDelegation:
    @pytest.mark.asyncio
    async def test_loop_drives_chat_layer_then_wire(self) -> None:
        """Production stack order: loop → chat middleware → wire client."""
        probe = _ProbeChat()
        layer, wire = _stack([_call_response(("c1", "echo", {"text": "a"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]}, middleware=[probe])
        assert len(wire.calls) == 2
        assert probe.calls == 2, "per-call chat middleware must run on every loop iteration"
        assert response.messages[-1].contents[0].text == "done"

    @pytest.mark.asyncio
    async def test_explicit_none_client_middleware_merges_with_per_call_middleware(self) -> None:
        probe = _ProbeChat()
        layer, _wire = _stack([_text_response()])

        response = await layer.get_response(
            [_user()],
            middleware=[probe],
            client_kwargs={"middleware": None},
        )

        assert probe.calls == 1
        assert response.messages[-1].text == "done"

    def test_getattr_two_hop_delegation(self) -> None:
        wire = _ScriptedClient([])
        wire.model = "wire-model"  # type: ignore[attr-defined]
        wire.STORES_BY_DEFAULT = True  # type: ignore[attr-defined]
        layer = ToolLoopLayer(ChatMiddlewareLayer(wire))
        assert layer.model == "wire-model"
        assert layer.STORES_BY_DEFAULT is True

    def test_getattr_attribute_error_passthrough(self) -> None:
        layer = ToolLoopLayer(ChatMiddlewareLayer(_ScriptedClient([])))
        with pytest.raises(AttributeError):
            _ = layer.does_not_exist
        with pytest.raises(AttributeError):
            _ = ChatMiddlewareLayer(_ScriptedClient([])).also_missing

    def test_ctor_rejects_chat_middleware(self) -> None:
        with pytest.raises(TypeError, match="chat middleware belongs to the inner"):
            ToolLoopLayer(_ScriptedClient([]), middleware=[_ProbeChat()])

    def test_ctor_defaults_mirror_framework_values(self) -> None:
        layer = ToolLoopLayer(_ScriptedClient([]))
        assert layer.max_iterations == DEFAULT_MAX_ITERATIONS
        assert layer.max_consecutive_errors == DEFAULT_MAX_CONSECUTIVE_ERRORS
        assert layer.max_function_calls is None


# ---------------------------------------------------------------------------
# Non-streaming loop
# ---------------------------------------------------------------------------


class TestNonStreamingLoop:
    @pytest.mark.asyncio
    async def test_informational_function_call_never_executes_same_named_local_tool(self) -> None:
        events: list[str] = []
        hosted_call = Content.from_function_call(
            "hosted-1",
            "echo",
            arguments={"text": "must stay remote"},
            informational_only=True,
        )
        response = ChatResponse(messages=[Message(role="assistant", contents=[hosted_call])])
        layer, wire = _stack([response])

        result = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert result is response
        assert events == []
        assert len(wire.calls) == 1
        assert result.messages[0].contents == [hosted_call]

    @pytest.mark.asyncio
    async def test_single_turn_no_tool_calls(self) -> None:
        layer, wire = _stack([_text_response("plain")])
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        assert len(wire.calls) == 1
        assert response.messages[-1].contents[0].text == "plain"

    @pytest.mark.parametrize("raw_seconds", [True, False])
    @pytest.mark.asyncio
    async def test_auto_model_rejects_bool_for_int(self, raw_seconds: bool) -> None:
        received: list[int] = []

        @tool(name="sleep_like")
        async def sleep_like(seconds: Annotated[int, "Seconds to wait."]) -> str:
            received.append(seconds)
            return f"slept:{seconds}"

        layer, _wire = _stack(
            [_call_response(("c1", "sleep_like", {"seconds": raw_seconds})), _text_response("recovered")]
        )
        response = await layer.get_response([_user()], options={"tools": [sleep_like]})

        assert received == []
        result = _result_contents(response)[0]
        _assert_actionable_argument_error(result, "sleep_like", "seconds: expected integer")

    @pytest.mark.parametrize("raw_mode", [True, False])
    @pytest.mark.asyncio
    async def test_auto_model_rejects_bool_for_int_literal(self, raw_mode: bool) -> None:
        received: list[int] = []

        @tool(name="mode_like")
        async def mode_like(mode: Annotated[Literal[0, 1], "Mode selector."]) -> str:
            received.append(mode)
            return f"mode:{mode}"

        layer, _wire = _stack([_call_response(("c1", "mode_like", {"mode": raw_mode})), _text_response("recovered")])
        response = await layer.get_response([_user()], options={"tools": [mode_like]})

        assert received == []
        result = _result_contents(response)[0]
        _assert_actionable_argument_error(result, "mode_like", "mode: expected 0 | 1")

    @pytest.mark.parametrize(
        ("literal_kind", "raw_mode"),
        [
            ("zero_or_true", False),
            ("zero_or_true", 1),
            ("one_or_false", True),
            ("one_or_false", 0),
            ("pure_bool", 1),
        ],
    )
    @pytest.mark.asyncio
    async def test_auto_model_rejects_cross_type_literal_match(self, literal_kind: str, raw_mode: bool | int) -> None:
        received: list[object] = []

        @tool(name="zero_or_true")
        async def zero_or_true(mode: Annotated[Literal[0, True], "Mixed mode."]) -> str:
            received.append(mode)
            return f"mode:{mode}"

        @tool(name="one_or_false")
        async def one_or_false(mode: Annotated[Literal[1, False], "Mixed mode."]) -> str:
            received.append(mode)
            return f"mode:{mode}"

        @tool(name="pure_bool")
        async def pure_bool(mode: Annotated[Literal[True], "Must be literal true."]) -> str:
            received.append(mode)
            return f"mode:{mode}"

        tools = {"zero_or_true": zero_or_true, "one_or_false": one_or_false, "pure_bool": pure_bool}
        layer, _wire = _stack([_call_response(("c1", literal_kind, {"mode": raw_mode})), _text_response("recovered")])
        response = await layer.get_response([_user()], options={"tools": [tools[literal_kind]]})

        assert received == []
        result = _result_contents(response)[0]
        _assert_actionable_argument_error(result, literal_kind, "mode: expected")

    @pytest.mark.asyncio
    async def test_auto_model_keeps_exact_mixed_literal_matches(self) -> None:
        received: list[object] = []

        @tool(name="mixed_like")
        async def mixed_like(mode: Annotated[Literal[0, True], "Mixed mode."]) -> str:
            received.append(mode)
            return f"mode:{mode}"

        layer, _wire = _stack(
            [
                _call_response(("c1", "mixed_like", {"mode": 0}), ("c2", "mixed_like", {"mode": True})),
                _text_response("done"),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [mixed_like]})

        assert received == [0, True]
        assert [type(value) for value in received] == [int, bool]
        results = _result_contents(response)
        assert results[0].exception is None
        assert results[1].exception is None

    @pytest.mark.asyncio
    async def test_auto_model_keeps_int_literal_and_bool_literal_semantics(self) -> None:
        received_mode: list[int] = []
        received_flag: list[bool] = []

        @tool(name="mode_like")
        async def mode_like(mode: Annotated[Literal[0, 1], "Mode selector."]) -> str:
            received_mode.append(mode)
            return f"mode:{mode}"

        @tool(name="flag_like")
        async def flag_like(flag: Annotated[Literal[True], "Must be literal true."]) -> str:
            received_flag.append(flag)
            return "flag:ok"

        layer, _wire = _stack(
            [
                _call_response(
                    ("c1", "mode_like", {"mode": 1}),
                    ("c2", "flag_like", {"flag": True}),
                ),
                _text_response("done"),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [mode_like, flag_like]})

        assert received_mode == [1]
        assert received_flag == [True]
        results = _result_contents(response)
        assert results[0].exception is None
        assert results[1].exception is None

    @pytest.mark.parametrize(("raw_seconds", "expected"), [("3", 3), (3.0, 3)])
    @pytest.mark.asyncio
    async def test_auto_model_keeps_non_bool_int_coercion(self, raw_seconds: str | float, expected: int) -> None:
        received: list[int] = []

        @tool(name="sleep_like")
        async def sleep_like(seconds: Annotated[int, "Seconds to wait."]) -> str:
            received.append(seconds)
            return f"slept:{seconds}"

        layer, _wire = _stack([_call_response(("c1", "sleep_like", {"seconds": raw_seconds})), _text_response("done")])
        response = await layer.get_response([_user()], options={"tools": [sleep_like]})

        assert received == [expected]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_auto_model_rejects_bool_nested_in_int_list(self) -> None:
        received: list[list[int]] = []

        @tool(name="read_like")
        async def read_like(line_range: Annotated[list[int], "Inclusive line range."]) -> str:
            received.append(line_range)
            return f"read:{line_range}"

        layer, _wire = _stack(
            [_call_response(("c1", "read_like", {"line_range": [True, True]})), _text_response("recovered")]
        )
        response = await layer.get_response([_user()], options={"tools": [read_like]})

        assert received == []
        result = _result_contents(response)[0]
        _assert_actionable_argument_error(result, "read_like", "line_range[0]: expected integer")

    @pytest.mark.asyncio
    async def test_auto_model_guards_optional_int_without_changing_bool_params(self) -> None:
        received_optional: list[int | None] = []
        received_bool: list[bool] = []

        @tool(name="optional_int")
        async def optional_int(value: Annotated[int | None, "Optional integer."]) -> str:
            received_optional.append(value)
            return f"optional:{value}"

        @tool(name="boolean")
        async def boolean(value: Annotated[bool, "Boolean flag."]) -> str:
            received_bool.append(value)
            return f"boolean:{value}"

        layer, _wire = _stack(
            [
                _call_response(
                    ("c1", "optional_int", {"value": True}),
                    ("c2", "boolean", {"value": True}),
                ),
                _text_response("recovered"),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [optional_int, boolean]})

        assert received_optional == []
        assert received_bool == [True]
        results = _result_contents(response)
        _assert_actionable_argument_error(results[0], "optional_int", "value: expected integer | null")
        assert results[1].exception is None

    @pytest.mark.asyncio
    async def test_explicit_null_for_required_nullable_argument_reaches_tool(self) -> None:
        received: list[str | None] = []

        @tool(name="nullable")
        async def nullable(value: str | None) -> str:
            received.append(value)
            return "ok"

        layer, _wire = _stack([_call_response(("c1", "nullable", {"value": None})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [nullable]})

        assert received == [None]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_nested_explicit_null_reaches_tool(self) -> None:
        received: list[Any] = []

        @tool(name="nested_nullable")
        async def nested_nullable(inner: _NullableValue) -> str:
            received.append(inner)
            return "ok"

        layer, _wire = _stack([_call_response(("c1", "nested_nullable", {"inner": {"value": None}})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [nested_nullable]})

        assert received == [{"value": None}]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_list_of_models_preserves_explicit_null_by_index(self) -> None:
        received: list[Any] = []

        @tool(name="listed_nullable")
        async def listed_nullable(items: list[_NullableValue]) -> str:
            received.append(items)
            return "ok"

        layer, _wire = _stack(
            [
                _call_response(("c1", "listed_nullable", {"items": [{"value": None}, {"value": "x"}]})),
                _text_response(),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [listed_nullable]})

        assert received == [[{"value": None}, {"value": "x"}]]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_omitted_optional_argument_keeps_callable_default(self) -> None:
        received: list[str] = []

        @tool(name="optional_default")
        async def optional_default(value: str = "fallback") -> str:
            received.append(value)
            return "ok"

        layer, _wire = _stack([_call_response(("c1", "optional_default", {})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [optional_default]})

        assert received == ["fallback"]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_custom_input_model_keeps_model_default_when_argument_is_omitted(self) -> None:
        received: list[int] = []

        async def custom_default(count: int = 3) -> str:
            received.append(count)
            return "ok"

        custom_tool = FunctionTool(
            name="custom_default",
            description="Use a custom input model.",
            func=custom_default,
            input_model=_DivergentDefaults,
        )
        layer, _wire = _stack([_call_response(("c1", "custom_default", {})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [custom_tool]})

        assert received == [7]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_field_serializer_does_not_corrupt_explicit_null(self) -> None:
        received: list[dict[str, Any]] = []

        async def serialized(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        serialized_tool = FunctionTool(
            name="serialized",
            description="Custom input model with a field serializer.",
            func=serialized,
            input_model=_SerializedNullable,
        )
        layer, _wire = _stack([_call_response(("c1", "serialized", {"safe": None, "value": "x"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [serialized_tool]})

        assert received == [{"safe": None, "value": {"serialized": "x"}}]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_computed_field_is_not_injected_into_arguments(self) -> None:
        received: list[int] = []

        async def computed(count: int) -> str:
            received.append(count)
            return "ok"

        computed_tool = FunctionTool(
            name="computed",
            description="Custom input model with a computed field.",
            func=computed,
            input_model=_ComputedGhost,
        )
        layer, _wire = _stack([_call_response(("c1", "computed", {"count": 5})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [computed_tool]})

        assert received == [5]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_reordering_field_serializer_subtree_is_left_untouched(self) -> None:
        received: list[Any] = []

        async def reordered(items: list[Any]) -> str:
            received.append(items)
            return "ok"

        reordering_tool = FunctionTool(
            name="reordered",
            description="Field serializer reverses the list.",
            func=reordered,
            input_model=_ReorderingItems,
        )
        layer, _wire = _stack(
            [
                _call_response(("c1", "reordered", {"items": [{"value": None}, {"value": "KEEP"}]})),
                _text_response(),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [reordering_tool]})

        # The serializer's output is authoritative: the explicit null must NOT
        # be re-applied by original index onto the reordered dump. The loop's
        # early serialization is a discarded validation transform; invoke
        # receives the original validation-keyed payload, so its serializer's
        # single callable-facing reversal remains authoritative.
        assert received == [[{"value": "KEEP"}, {}]]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_tuple_of_models_preserves_explicit_null_by_index(self) -> None:
        received: list[Any] = []

        async def tupled(items: Any) -> str:
            received.append(items)
            return "ok"

        tuple_tool = FunctionTool(
            name="tupled",
            description="Tuple-typed field dumps stay tuples in Python mode.",
            func=tupled,
            input_model=_TupleItems,
        )
        layer, _wire = _stack(
            [
                _call_response(("c1", "tupled", {"items": [{"value": None}, {"value": "x"}]})),
                _text_response(),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [tuple_tool]})

        assert received == [({"value": None}, {"value": "x"})]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_serialize_by_alias_restores_null_under_the_wire_key(self) -> None:
        received: list[Any] = []

        async def aliased(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        aliased_tool = FunctionTool(
            name="aliased_null",
            description="serialize_by_alias keys the dump by the wire alias.",
            func=aliased,
            input_model=_AliasedNullable,
        )
        layer, _wire = _stack([_call_response(("c1", "aliased_null", {"wire": None})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [aliased_tool]})

        # The restore must land under the alias the schema requires — a
        # field-name key fails validation as "'wire' missing" and the call
        # errors before reaching the tool.
        assert received == [{"wire": None}]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.parametrize(
        ("input_model", "arguments", "expected"),
        [
            (_Round15SplitAlias, {"inputKey": None}, {"callableKey": None}),
            (_Round15SplitAlias, {"inputKey": "x"}, {"callableKey": "x"}),
            (_Round15ValidationAliasOnly, {"inputKey": None}, {"value": None}),
            (_Round15GeneratedValidationAlias, {"VALUE": None}, {"callableKey": None}),
            (_Round15PlainArguments, {"value": None}, {"value": None}),
        ],
        ids=["split-null", "split-non-null", "validation-only", "generated-validation", "plain"],
    )
    @pytest.mark.asyncio
    async def test_loop_preserves_validation_input_until_invoke(
        self, input_model: type[BaseModel], arguments: dict[str, Any], expected: dict[str, Any]
    ) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        alias_tool = FunctionTool(
            name="round15_aliases",
            description="Loop validation never round-trips serialized keys through validation aliases.",
            func=capture,
            input_model=input_model,
        )
        layer, _wire = _stack([_call_response(("c1", "round15_aliases", arguments)), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [alias_tool]})

        assert received == [expected]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.parametrize("replacement", [True, 1.0], ids=["bool", "float"])
    @pytest.mark.asyncio
    async def test_equal_cross_type_middleware_rewrite_is_preserved(self, replacement: Any) -> None:
        received: list[Any] = []

        class _Rewrite(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                assert isinstance(context.arguments, dict)
                context.arguments["value"] = replacement
                await call_next()

        async def capture(value: Any) -> str:
            received.append(value)
            return "ok"

        rewrite_tool = FunctionTool(
            name="equal_rewrite",
            description="Type-changing middleware rewrites remain observable.",
            func=capture,
            input_model=_Round16RewriteArguments,
        )
        layer, _wire = _stack([_call_response(("c1", "equal_rewrite", {"value": 1})), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [rewrite_tool]}, middleware=[_Rewrite()])

        assert len(received) == 1
        assert type(received[0]) is type(replacement)
        assert received[0] == replacement
        assert _result_contents(response)[0].exception is None

    @pytest.mark.parametrize("replacement", ["after", None], ids=["non-null", "null"])
    @pytest.mark.asyncio
    async def test_mutated_split_alias_arguments_use_callable_keyspace(self, replacement: Any) -> None:
        received: list[dict[str, Any]] = []

        class _Rewrite(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                assert context.arguments == {"callableKey": "before"}
                context.arguments["callableKey"] = replacement
                await call_next()

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        alias_tool = FunctionTool(
            name="mutated_split_alias",
            description="Mutated middleware arguments already use callable-facing keys.",
            func=capture,
            input_model=_Round15SplitAlias,
        )
        layer, _wire = _stack([_call_response(("c1", "mutated_split_alias", {"inputKey": "before"})), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [alias_tool]}, middleware=[_Rewrite()])

        assert received == [{"callableKey": replacement}]
        assert _result_contents(response)[0].exception is None

    def test_middleware_compare_is_representation_sensitive(self) -> None:
        utc = datetime.UTC
        named_utc = datetime.timezone(datetime.timedelta(0), "named")

        assert not _middleware_arguments_equal({"value": -0.0}, {"value": 0.0})
        assert not _middleware_arguments_equal(
            {"value": decimal.Decimal("1.0")},
            {"value": decimal.Decimal("1.00")},
        )
        assert not _middleware_arguments_equal(
            {"value": datetime.datetime(2026, 1, 1, tzinfo=utc)},
            {"value": datetime.datetime(2026, 1, 1, tzinfo=named_utc)},
        )

        assert _middleware_arguments_equal({"value": 1.5}, {"value": 1.5})
        assert _middleware_arguments_equal(
            {"value": decimal.Decimal("1.0")},
            {"value": decimal.Decimal("1.0")},
        )
        assert _middleware_arguments_equal(
            {"value": datetime.datetime(2026, 1, 1, tzinfo=utc)},
            {"value": datetime.datetime(2026, 1, 1, tzinfo=utc)},
        )

    @pytest.mark.asyncio
    async def test_signed_zero_middleware_rewrite_reaches_callable(self) -> None:
        received: list[float] = []

        class _Rewrite(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                assert context.arguments == {"value": -0.0}
                context.arguments["value"] = 0.0
                await call_next()

        async def capture(value: Any) -> str:
            received.append(value)
            return "ok"

        rewrite_tool = FunctionTool(
            name="signed_zero_rewrite",
            description="Representation-changing numeric rewrites remain observable.",
            func=capture,
            input_model=_Round16RewriteArguments,
        )
        layer, _wire = _stack([_call_response(("c1", "signed_zero_rewrite", {"value": -0.0})), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [rewrite_tool]}, middleware=[_Rewrite()])

        assert received == [0.0]
        assert repr(received[0]) == "0.0"
        assert _result_contents(response)[0].exception is None

    @pytest.mark.parametrize(
        ("input_model", "arguments", "expected"),
        [
            (_Round17ConsumingArguments, {"x": 1}, {"x": 1}),
            (_Round17NestedConsumingArguments, {"items": [{"x": 1}]}, {"items": [{"x": 1}]}),
        ],
        ids=["top-level-pop", "nested-list-dict-pop"],
    )
    @pytest.mark.asyncio
    async def test_throwaway_validation_cannot_mutate_forwarded_original(
        self, input_model: type[BaseModel], arguments: dict[str, Any], expected: dict[str, Any]
    ) -> None:
        received: list[dict[str, Any]] = []

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        consuming_tool = FunctionTool(
            name="consuming_validator",
            description="Throwaway validation receives an isolated container copy.",
            func=capture,
            input_model=input_model,
        )
        layer, _wire = _stack([_call_response(("c1", "consuming_validator", arguments)), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [consuming_tool]})

        assert received == [expected]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_invoke_validation_cannot_mutate_persisted_call_arguments(self) -> None:
        received: list[dict[str, Any]] = []
        arguments = {"items": [{"x": 1}]}
        function_call = Content.from_function_call(
            call_id="c1",
            name="persisted_arguments",
            arguments=arguments,
        )

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        consuming_tool = FunctionTool(
            name="persisted_arguments",
            description="Invoke validation cannot mutate persisted tool-call input.",
            func=capture,
            input_model=_Round17NestedConsumingArguments,
        )
        first_turn = ChatResponse(messages=[Message(role="assistant", contents=[function_call])])
        layer, _wire = _stack([first_turn, _text_response()])

        response = await layer.get_response([_user()], options={"tools": [consuming_tool]})

        assert received == [{"items": [{"x": 1}]}]
        assert arguments == {"items": [{"x": 1}]}
        assert function_call.arguments == {"items": [{"x": 1}]}
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_cyclic_arguments_degrade_to_legacy_path(self) -> None:
        received: list[Any] = []
        cycle: dict[str, Any] = {}
        cycle["self"] = cycle

        async def capture(payload: Any) -> str:
            received.append(payload)
            return "ok"

        cyclic_tool = FunctionTool(
            name="cyclic_schema_arguments",
            description="Cyclic raw arguments cannot escape snapshot handling.",
            func=capture,
            input_model={
                "type": "object",
                "properties": {"payload": {"type": "object"}},
                "required": ["payload"],
                "additionalProperties": False,
            },
        )
        layer, _wire = _stack([_call_response(("c1", "cyclic_schema_arguments", {"payload": cycle})), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [cyclic_tool]})

        assert len(received) == 1
        assert received[0]["self"] is received[0]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_mapping_order_only_rewrite_may_use_untouched_fast_path(self) -> None:
        received_order: list[list[str]] = []

        class _Reorder(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                assert isinstance(context.arguments, dict)
                first = context.arguments.pop("first")
                context.arguments["first"] = first
                await call_next()

        async def capture(**kwargs: Any) -> str:
            received_order.append(list(kwargs))
            return "ok"

        ordered_tool = FunctionTool(
            name="ordered_schema_arguments",
            description="Mapping insertion order is part of middleware output.",
            func=capture,
            input_model={
                "type": "object",
                "properties": {"first": {"type": "integer"}, "second": {"type": "integer"}},
                "required": ["first", "second"],
                "additionalProperties": False,
            },
        )
        layer, _wire = _stack(
            [_call_response(("c1", "ordered_schema_arguments", {"first": 1, "second": 2})), _text_response()]
        )

        response = await layer.get_response([_user()], options={"tools": [ordered_tool]}, middleware=[_Reorder()])

        # Order-insensitive dict equality is intentional for trusted middleware.
        assert received_order == [["first", "second"]]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_same_order_reinsert_keeps_split_alias_fast_path(self) -> None:
        received: list[dict[str, Any]] = []

        class _ReinsertLast(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                assert isinstance(context.arguments, dict)
                second = context.arguments.pop("second")
                context.arguments["second"] = second
                await call_next()

        async def capture(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        alias_tool = FunctionTool(
            name="same_order_alias",
            description="A same-position reinsert remains structurally untouched.",
            func=capture,
            input_model=_Round17OrderFastPathArguments,
        )
        layer, _wire = _stack(
            [_call_response(("c1", "same_order_alias", {"inputFirst": 1, "second": 2})), _text_response()]
        )

        response = await layer.get_response([_user()], options={"tools": [alias_tool]}, middleware=[_ReinsertLast()])

        assert received == [{"callableFirst": 1, "second": 2}]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_typed_dict_member_explicit_null_survives_loop_path(self) -> None:
        received: list[Any] = []

        async def td_nullable(**kwargs: Any) -> str:
            received.append(kwargs)
            return "ok"

        td_tool = FunctionTool(
            name="td_nullable",
            description="exclude_none drops TypedDict member nulls like model fields.",
            func=td_nullable,
            input_model=_MemberNullableHolder,
        )
        layer, _wire = _stack(
            [
                _call_response(("c1", "td_nullable", {"inner": {"value": None}})),
                _text_response(),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [td_tool]})

        assert received == [{"inner": {"value": None}}]
        assert _result_contents(response)[0].exception is None

    @pytest.mark.asyncio
    async def test_two_iteration_loop_extends_prepped_and_prepends_fcc(self) -> None:
        events: list[str] = []
        layer, wire = _stack([_call_response(("c1", "echo", {"text": "a"})), _text_response("final")])
        response = await layer.get_response([_user("q")], options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:a"]
        # Second wire call sees user + assistant(call) + tool(result).
        second = wire.calls[1]["messages"]
        assert [m.role for m in second] == ["user", "assistant", "tool"]
        # Final response carries the accumulated loop transcript (fcc prepend).
        assert [m.role for m in response.messages] == ["assistant", "tool", "assistant"]
        results = _result_contents(response)
        assert len(results) == 1
        assert results[0].call_id == "c1"
        assert "echo:a" in str(results[0].result)

    @pytest.mark.asyncio
    async def test_successful_tool_result_preserves_function_call_metadata(self) -> None:
        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        function_call.additional_properties["provider_marker"] = "kept"
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response("final")])

        response = await layer.get_response([_user("q")], options={"tools": [_make_tool()]})

        results = _result_contents(response)
        assert len(results) == 1
        assert results[0].additional_properties["provider_marker"] == "kept"
        assert TOOL_INVOCATION_ORDER_KEY not in results[0].additional_properties
        timing = function_call.additional_properties[TRAJECTORY_TIMING_KEY]
        assert timing == results[0].additional_properties[TRAJECTORY_TIMING_KEY]
        assert timing["duration_ms"] == 0
        call_props = function_call.additional_properties
        assert {
            key: value
            for key, value in call_props.items()
            if key not in (TRAJECTORY_TIMING_KEY, ANALYTICS_ITEM_ID_KEY, OPERATION_ID_KEY)
        } == {"provider_marker": "kept", TOOL_INVOCATION_ORDER_KEY: 0}
        # Landing minted the call's item id and tool operation id; the result
        # shares the operation and owns a distinct item id.
        assert is_valid_analytics_id(call_props[ANALYTICS_ITEM_ID_KEY])
        assert is_valid_analytics_id(call_props[OPERATION_ID_KEY])
        result_props = results[0].additional_properties
        assert result_props[OPERATION_ID_KEY] == call_props[OPERATION_ID_KEY]
        assert is_valid_analytics_id(result_props[ANALYTICS_ITEM_ID_KEY])
        assert result_props[ANALYTICS_ITEM_ID_KEY] != call_props[ANALYTICS_ITEM_ID_KEY]

    @pytest.mark.asyncio
    async def test_caller_containers_not_mutated(self) -> None:
        tools = [_make_tool()]
        options = {"tools": tools}
        caller_messages = [_user("q")]
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "a"})), _text_response()])
        await layer.get_response(caller_messages, options=options)
        assert options == {"tools": tools}, "caller options must not gain loop-internal keys"
        assert len(tools) == 1
        assert len(caller_messages) == 1

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_result_without_raising(self) -> None:
        function_call = Content.from_function_call("c1", "ghost", arguments={})
        function_call.additional_properties["existing"] = "kept"
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        results = _result_contents(response)
        assert len(results) == 1
        assert 'Requested function "ghost" not found' in str(results[0].result)
        assert results[0].exception is not None
        assert results[0].additional_properties["existing"] == "kept"
        assert results[0].additional_properties[TOOL_FAILED_METADATA_KEY] is True
        assert results[0].additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "tool_not_found"
        assert results[0].additional_properties[TOOL_ERROR_MESSAGE_METADATA_KEY] == (
            'Requested function "ghost" not found.'
        )
        assert EXECUTION_STAMP_KEY not in results[0].additional_properties
        assert TOOL_INVOCATION_ORDER_KEY not in results[0].additional_properties
        timing = function_call.additional_properties[TRAJECTORY_TIMING_KEY]
        assert timing == results[0].additional_properties[TRAJECTORY_TIMING_KEY]
        assert timing["duration_ms"] == 0
        assert timing["started_at"] == timing["finished_at"]
        assert {
            key: value
            for key, value in function_call.additional_properties.items()
            if key not in (TRAJECTORY_TIMING_KEY, ANALYTICS_ITEM_ID_KEY, OPERATION_ID_KEY)
        } == {"existing": "kept", TOOL_INVOCATION_ORDER_KEY: 0}

    @pytest.mark.asyncio
    async def test_prevalidation_failure_skips_function_pipeline(self) -> None:
        probe = _ProbeFunction()
        function_call = Content.from_function_call("c1", "echo", arguments={"wrong_arg": 1})
        function_call.additional_properties["existing"] = "kept"
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]}, middleware=[probe])
        results = _result_contents(response)
        assert len(results) == 1
        message = _assert_actionable_argument_error(
            results[0],
            "echo",
            "unknown 'wrong_arg'",
            "missing 'text'",
            "Expected: {text: string}",
        )
        assert results[0].additional_properties["existing"] == "kept"
        assert results[0].additional_properties[TOOL_FAILED_METADATA_KEY] is True
        assert results[0].additional_properties[TOOL_ERROR_MESSAGE_METADATA_KEY] == message.removeprefix("Error: ")
        assert EXECUTION_STAMP_KEY not in results[0].additional_properties
        assert TOOL_INVOCATION_ORDER_KEY not in results[0].additional_properties
        timing = function_call.additional_properties[TRAJECTORY_TIMING_KEY]
        assert timing == results[0].additional_properties[TRAJECTORY_TIMING_KEY]
        assert timing["duration_ms"] == 0
        assert timing["started_at"] == timing["finished_at"]
        assert {
            key: value
            for key, value in function_call.additional_properties.items()
            if key not in (TRAJECTORY_TIMING_KEY, ANALYTICS_ITEM_ID_KEY, OPERATION_ID_KEY)
        } == {"existing": "kept", TOOL_INVOCATION_ORDER_KEY: 0}
        assert probe.contexts == [], "bad arguments must fail before the middleware pipeline runs"

    @pytest.mark.asyncio
    async def test_typed_tool_rejects_unknown_argument_and_suggests_valid_names(self) -> None:
        received: list[str] = []

        @tool(name="read_file")
        async def read_file(
            path: Annotated[str, "Absolute or relative path to the file to read."],
            max_tokens: int = 5000,
            line_range: list[int] | None = None,
        ) -> str:
            received.append(path)
            return path

        layer, _wire = _stack(
            [
                _call_response(
                    ("c1", "read_file", {"file_path": "wrong.py", "max_tokens": 4000}),
                    ("c2", "read_file", {"path": "right.py", "file_path": "wrong.py"}),
                ),
                _text_response("recovered"),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [read_file]})

        assert received == []
        missing_path, extra_only = _result_contents(response)
        missing_path_text = _assert_actionable_argument_error(
            missing_path,
            "read_file",
            "unknown 'file_path' (did you mean 'path'?)",
            "missing 'path'",
            "Expected: {path: string, max_tokens?: integer, line_range?: [integer] | null}",
        )
        assert len(missing_path_text) < 220
        extra_only_text = _assert_actionable_argument_error(
            extra_only,
            "read_file",
            "unknown 'file_path' (did you mean 'path'?)",
        )
        assert "missing '" not in extra_only_text

    @pytest.mark.asyncio
    async def test_before_validator_model_keeps_rewritten_argument_names(self) -> None:
        # A model-level before-validator may accept and rewrite top-level keys
        # the schema and field spellings can't advertise; the unknown-argument
        # pre-check must stand down for such models.
        received: list[int] = []

        class UnwrapArguments(BaseModel):
            x: int

            @model_validator(mode="before")
            @classmethod
            def _unwrap(cls, data: object) -> object:
                if isinstance(data, dict) and "payload" in data:
                    return {"x": data["payload"]}
                return data

        async def unwrap_like(x: int) -> str:
            received.append(x)
            return str(x)

        unwrap_tool = FunctionTool(
            name="unwrap_like",
            description="Unwrap a payload.",
            func=unwrap_like,
            input_model=UnwrapArguments,
        )
        layer, _wire = _stack(
            [
                _call_response(("c1", "unwrap_like", {"payload": 7})),
                _text_response("done"),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [unwrap_tool]})

        assert received == [7]
        results = _result_contents(response)
        assert results[0].exception is None

    @pytest.mark.asyncio
    async def test_parallel_mixed_completion_stamps_each_matching_result(self) -> None:
        release_slow = asyncio.Event()
        fast_finished = asyncio.Event()
        completion_order: list[str] = []

        @tool(name="mixed")
        async def mixed(value: str) -> str:
            if value == "slow-ok":
                await release_slow.wait()
                return "slow result"
            fast_finished.set()
            raise RuntimeError("fast failure")

        async def _release_after_fast() -> None:
            await fast_finished.wait()
            release_slow.set()

        releaser = asyncio.create_task(_release_after_fast())
        layer, _wire = _stack(
            [
                _call_response(
                    ("slow-call", "mixed", {"value": "slow-ok"}),
                    ("fast-call", "mixed", {"value": "fast-error"}),
                ),
                _text_response(),
            ]
        )

        response = await layer.get_response(
            [_user()],
            options={"tools": [mixed]},
            middleware=[_ExecutionStampFunction(completion_order)],
        )
        await releaser

        results = _result_contents(response)
        assert completion_order == ["fast-error", "slow-ok"]
        assert [result.call_id for result in results] == ["slow-call", "fast-call"]
        slow_stamp = results[0].additional_properties[EXECUTION_STAMP_KEY]
        fast_stamp = results[1].additional_properties[EXECUTION_STAMP_KEY]
        assert slow_stamp["effective_args"] == '{"value":"slow-ok"}'
        assert slow_stamp["outcome"] == "ok"
        assert fast_stamp["effective_args"] == '{"value":"fast-error"}'
        assert fast_stamp["outcome"] == "error"
        assert fast_stamp["error_kind"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_truncated_response_surfaces_output_limit_message(self) -> None:
        # A response cut off at the output token limit (finish_reason="length")
        # leaves the tool arguments as incomplete/unparseable JSON. The error
        # must name the token-limit cause instead of the opaque generic message
        # so the user knows to raise max_tokens and the model can split the call.
        function_call = Content.from_function_call("c1", "echo", arguments='{"text": "hello wor')
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [function_call])], finish_reason="length"),
                _text_response(),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        results = _result_contents(response)
        assert len(results) == 1
        assert "max_tokens" in str(results[0].result)
        assert "cut off" in str(results[0].result)
        assert results[0].additional_properties[TOOL_FAILED_METADATA_KEY] is True
        assert results[0].additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_truncated"

    @pytest.mark.asyncio
    async def test_truncated_response_with_empty_string_args_surfaces_output_limit_message(self) -> None:
        # Some providers can surface a length-cutoff tool call with an empty raw
        # argument string.  Content.parse_arguments() turns that into {}, so the
        # truncation classification must catch it before it degrades to generic
        # schema-validation failure.
        function_call = Content.from_function_call("c1", "echo", arguments="")
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [function_call])], finish_reason="length"),
                _text_response(),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        results = _result_contents(response)
        assert len(results) == 1
        assert "max_tokens" in str(results[0].result)
        assert results[0].additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_truncated"

    @pytest.mark.asyncio
    async def test_truncated_final_mapping_args_surfaces_output_limit_message(self) -> None:
        # Anthropic's blocking SDK path delivers tool_use.input as a parsed dict,
        # even when a length cutoff likely omitted required fields.  An empty
        # final call mapping under finish_reason="length" should still get the
        # actionable truncation error instead of generic argument_parsing.
        function_call = Content.from_function_call("c1", "echo", arguments={})
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [function_call])], finish_reason="length"),
                _text_response(),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        results = _result_contents(response)
        assert len(results) == 1
        assert "max_tokens" in str(results[0].result)
        assert results[0].additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_truncated"

    @pytest.mark.asyncio
    async def test_truncated_final_partial_mapping_args_surfaces_output_limit_message(self) -> None:
        # Cover the non-empty parsed-mapping branch: Anthropic can return a
        # partial dict for a final tool call that has one required field present
        # and another omitted by the output-token cutoff.
        @tool(name="write_file")
        async def write_file(path: str, content: str) -> str:
            return f"{path}:{content}"

        function_call = Content.from_function_call("c1", "write_file", arguments={"path": "out.txt"})
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [function_call])], finish_reason="length"),
                _text_response(),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [write_file]})
        results = _result_contents(response)
        assert len(results) == 1
        assert "max_tokens" in str(results[0].result)
        assert results[0].additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_truncated"

    @pytest.mark.asyncio
    async def test_truncated_response_with_parseable_but_invalid_args_is_not_truncation(self) -> None:
        # finish_reason="length" is response-wide. A call whose arguments parsed
        # cleanly but fail schema validation is a real argument error, not a
        # cutoff — it must keep the actionable "argument_parsing" classification
        # so the model fixes the arguments instead of being told to raise max_tokens.
        function_call = Content.from_function_call("c1", "echo", arguments={"wrong_arg": 1})
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [function_call])], finish_reason="length"),
                _text_response(),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        results = _result_contents(response)
        assert len(results) == 1
        _assert_actionable_argument_error(
            results[0],
            "echo",
            "unknown 'wrong_arg'",
            "missing 'text'",
        )

    @pytest.mark.asyncio
    async def test_truncated_non_final_mapping_args_stay_argument_parsing(self) -> None:
        # The parsed-mapping heuristic exists for the final block that was most
        # likely cut off.  Earlier calls in the same length-truncated response
        # keep actionable argument_parsing unless their raw JSON is unparseable.
        bad_first = Content.from_function_call("c1", "echo", arguments={})
        good_final = Content.from_function_call("c2", "echo", arguments={"text": "ok"})
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [bad_first, good_final])], finish_reason="length"),
                _text_response(),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        results = _result_contents(response)
        assert len(results) == 2
        _assert_actionable_argument_error(results[0], "echo", "missing 'text'")
        assert results[1].exception is None

    @pytest.mark.asyncio
    async def test_truncated_response_with_parseable_json_string_invalid_args_is_not_truncation(self) -> None:
        # Same invariant as above, but with a JSON string so the
        # _arguments_unparseable json.loads branch is actually exercised.
        function_call = Content.from_function_call("c1", "echo", arguments='{"wrong_arg": 1}')
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [function_call])], finish_reason="length"),
                _text_response(),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        results = _result_contents(response)
        assert len(results) == 1
        _assert_actionable_argument_error(
            results[0],
            "echo",
            "unknown 'wrong_arg'",
            "missing 'text'",
        )

    @pytest.mark.asyncio
    async def test_truncated_finish_reason_with_valid_args_runs_normally(self) -> None:
        # finish_reason="length" alone must not fabricate an error: when the
        # arguments actually parse, the tool runs (no false positive).
        events: list[str] = []
        function_call = Content.from_function_call("c1", "echo", arguments={"text": "hello"})
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [function_call])], finish_reason="length"),
                _text_response(),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})
        results = _result_contents(response)
        assert len(results) == 1
        assert results[0].exception is None
        assert "echo:hello" in str(results[0].result)
        assert events == ["tool:echo:hello"]

    @pytest.mark.asyncio
    async def test_tool_exception_returns_short_error_result(self) -> None:
        @tool(name="boom")
        async def boom(text: str) -> str:
            raise RuntimeError("kaput")

        layer, _wire = _stack([_call_response(("c1", "boom", {"text": "x"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [boom]})
        results = _result_contents(response)
        assert str(results[0].result) == "Error: Function failed."
        assert "kaput" in str(results[0].exception)

    @pytest.mark.asyncio
    async def test_consecutive_error_cap_forces_tool_choice_none(self) -> None:
        @tool(name="boom")
        async def boom(text: str) -> str:
            raise RuntimeError("kaput")

        layer, wire = _stack(
            [_call_response(("c1", "boom", {"text": "x"})), _text_response("recovered")],
            max_consecutive_errors=1,
        )
        response = await layer.get_response([_user()], options={"tools": [boom]})
        assert len(wire.calls) == 2
        assert wire.calls[1]["options"].get("tool_choice") == "none"
        # The error result is still submitted before the final turn.
        assert "Error: Function failed." in str(_result_contents(response)[0].result)
        assert response.messages[-1].contents[0].text == "recovered"

    @pytest.mark.asyncio
    async def test_ceiling_preserves_unknown_tool_failure_for_consecutive_error_cap(self) -> None:
        missing_name = "missing-" + "x" * 10_000
        layer, wire = _stack(
            [_call_response(("c1", missing_name, {})), _text_response("recovered")],
            max_consecutive_errors=1,
            tool_result_ceiling_tokens=100,
        )

        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})

        result = _result_contents(response)[0]
        assert result.exception is not None
        assert result.additional_properties[TOOL_FAILED_METADATA_KEY] is True
        assert wire.calls[1]["options"].get("tool_choice") == "none"

    @pytest.mark.asyncio
    async def test_parallel_multi_failure_batch_counts_one_error_increment(self) -> None:
        """A batch failing several calls at once advances the cap by ONE, not per call."""
        layer, wire = _stack(
            [
                _call_response(("c1", "echo", {"wrong_arg": 1}), ("c2", "ghost", {})),
                _call_response(("c3", "echo", {"wrong_arg": 2})),
                _text_response("recovered"),
            ],
            max_consecutive_errors=2,
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        assert len(wire.calls) == 3
        # Two same-batch failures = one increment: still under the cap.
        assert wire.calls[1]["options"].get("tool_choice") is None
        # The second failing batch reaches the cap: final submit disables tools.
        assert wire.calls[2]["options"].get("tool_choice") == "none"
        assert response.messages[-1].contents[0].text == "recovered"

    @pytest.mark.asyncio
    async def test_max_iterations_exhaustion_final_turn_without_tools(self) -> None:
        layer, wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"})), _text_response("forced final")],
            max_iterations=1,
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        assert len(wire.calls) == 2
        assert wire.calls[1]["options"].get("tool_choice") == "none"
        # Exhaustion final turn prepends the accumulated transcript.
        assert [m.role for m in response.messages] == ["assistant", "tool", "assistant"]

    @pytest.mark.asyncio
    async def test_required_tool_choice_resets_after_one_iteration(self) -> None:
        layer, wire = _stack([_call_response(("c1", "echo", {"text": "a"})), _text_response()])
        await layer.get_response([_user()], options={"tools": [_make_tool()], "tool_choice": "required"})
        assert wire.calls[0]["options"]["tool_choice"] == "required"
        assert wire.calls[1]["options"]["tool_choice"] is None

    @pytest.mark.asyncio
    async def test_usage_aggregates_across_iterations(self) -> None:
        layer, _wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"}), usage_details={"input_token_count": 3}),
                _text_response("final", usage_details={"input_token_count": 4}),
            ]
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        assert response.usage_details["input_token_count"] == 7
        assert response.latest_usage_details == {"input_token_count": 4}

    @pytest.mark.asyncio
    async def test_termination_returns_current_response_without_fcc_prepend(self) -> None:
        """Framework shape ``_tools.py:2472-2474``: terminate returns the current
        iteration's response (tool results appended), with no fcc prepend."""
        layer, wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"})),
                _call_response(("c2", "echo", {"text": "b"})),
                _text_response("never"),
            ]
        )
        terminator = _TerminateFunction(result="interrupted")
        # First iteration runs normally; second iteration's middleware terminates.
        flip = {"first": True}

        class _SecondCallTerminates(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                if flip["first"]:
                    flip["first"] = False
                    await call_next()
                    return
                await terminator.process(context, call_next)

        response = await layer.get_response(
            [_user()], options={"tools": [_make_tool()]}, middleware=[_SecondCallTerminates()]
        )
        assert len(wire.calls) == 2, "loop must stop right after the terminated batch"
        # Only the second iteration's messages — no fcc prepend on terminate.
        assert [m.role for m in response.messages] == ["assistant", "tool"]
        assert str(_result_contents(response)[0].result) == "interrupted"

    @pytest.mark.asyncio
    async def test_termination_exc_result_content_is_owned_by_invocation(self) -> None:
        canned = Content.from_function_result(call_id="c1", result="canned")
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "a"}))])
        response = await layer.get_response(
            [_user()],
            options={"tools": [_make_tool()]},
            middleware=[_TerminateFunction(exc_result=canned)],
        )
        result = _result_contents(response)[0]
        assert result is not canned
        assert result.call_id == "c1"
        assert result.result == "canned"
        assert canned.call_id == "c1"

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_execute_concurrently(self) -> None:
        barrier = asyncio.Barrier(2)

        @tool(name="alpha")
        async def alpha(text: str) -> str:
            await asyncio.wait_for(barrier.wait(), timeout=2)
            return "alpha-done"

        @tool(name="beta")
        async def beta(text: str) -> str:
            await asyncio.wait_for(barrier.wait(), timeout=2)
            return "beta-done"

        layer, _wire = _stack(
            [_call_response(("c1", "alpha", {"text": "x"}), ("c2", "beta", {"text": "y"})), _text_response()]
        )
        response = await layer.get_response([_user()], options={"tools": [alpha, beta]})
        results = _result_contents(response)
        assert [r.call_id for r in results] == ["c1", "c2"], "batch order is preserved"

    @pytest.mark.asyncio
    async def test_duplicate_and_already_resolved_calls_are_skipped(self) -> None:
        events: list[str] = []
        call_a = Content.from_function_call(call_id="c1", name="echo", arguments={"text": "a"})
        dup_a = Content.from_function_call(call_id="c1", name="echo", arguments={"text": "a"})
        resolved_call = Content.from_function_call(call_id="c2", name="echo", arguments={"text": "b"})
        resolved_result = Content.from_function_result(call_id="c2", result="cached")
        first = ChatResponse(
            messages=[
                Message(role="assistant", contents=[call_a, dup_a, resolved_call]),
                Message(role="tool", contents=[resolved_result]),
            ]
        )
        layer, _wire = _stack([first, _text_response()])
        await layer.get_response([_user()], options={"tools": [_make_tool(events)]})
        assert events == ["tool:echo:a"], "duplicate call_id and already-resolved calls must not execute"

    @pytest.mark.asyncio
    async def test_runtime_kwargs_merge_and_filtering(self) -> None:
        probe = _ProbeFunction()
        session = AgentSession()
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "a"})), _text_response()])
        await layer.get_response(
            [_user()],
            options={
                "tools": [_make_tool()],
                "additional_function_arguments": {"from_options": 2},
            },
            middleware=[probe],
            function_invocation_kwargs={"from_fik": 1, "middleware": "smuggled", "conversation_id": "nope"},
            client_kwargs={"session": session},
        )
        ctx = probe.contexts[0]
        assert ctx.kwargs["from_fik"] == 1
        assert ctx.kwargs["from_options"] == 2
        assert "middleware" not in ctx.kwargs
        assert "conversation_id" not in ctx.kwargs
        assert ctx.kwargs["session"] is session
        assert ctx.session is session
        assert ctx.metadata["call_id"] == "c1"

    @pytest.mark.asyncio
    async def test_additional_function_arguments_not_sent_to_wire(self) -> None:
        layer, wire = _stack([_text_response()])
        await layer.get_response(
            [_user()],
            options={"tools": [_make_tool()], "additional_function_arguments": {"x": 1}},
        )
        assert "additional_function_arguments" not in wire.calls[0]["options"]

    @pytest.mark.asyncio
    async def test_middleware_argument_rewrite_reaches_tool(self) -> None:
        class _Rewrite(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                assert isinstance(context.arguments, dict)
                context.arguments["text"] = "rewritten"
                await call_next()

        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "original"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]}, middleware=[_Rewrite()])
        assert "echo:rewritten" in str(_result_contents(response)[0].result)

    @pytest.mark.asyncio
    async def test_progressive_tool_exposure_via_live_tools(self) -> None:
        events: list[str] = []
        late_tool = _make_tool(events, name="late")

        class _AddTool(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                assert context.tools is not None
                if late_tool not in context.tools:
                    context.tools.append(late_tool)
                await call_next()

        layer, wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"})),
                _call_response(("c2", "late", {"text": "b"})),
                _text_response(),
            ]
        )
        await layer.get_response([_user()], options={"tools": [_make_tool(events)]}, middleware=[_AddTool()])
        assert "tool:late:b" in events
        # The model also sees the grown list (same run-local list in options).
        assert len(wire.calls[1]["options"]["tools"]) == 2

    @pytest.mark.asyncio
    async def test_conversation_id_continuation(self) -> None:
        session = AgentSession()
        layer, wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"}), conversation_id="conv-1"),
                _text_response("final", conversation_id="conv-1"),
            ]
        )
        await layer.get_response(
            [_user()],
            options={"tools": [_make_tool()]},
            client_kwargs={"session": session, "x-extra": "kept"},
        )
        # Only the new tool-result message is resent; the service holds the rest.
        second = wire.calls[1]["messages"]
        assert [m.role for m in second] == ["tool"]
        # Continuation id written back into the shared client_kwargs dict + options.
        inner_client_kwargs = wire.calls[1]["kwargs"]["client_kwargs"]
        assert inner_client_kwargs["conversation_id"] == "conv-1"
        assert inner_client_kwargs["x-extra"] == "kept"
        assert "session" not in inner_client_kwargs
        assert "loop_recorder" not in inner_client_kwargs
        assert wire.calls[1]["options"]["conversation_id"] == "conv-1"
        assert session.service_session_id == "conv-1"


# ---------------------------------------------------------------------------
# same_tool_calls_in_batch stamping
# ---------------------------------------------------------------------------


class TestSameToolCallsInBatch:
    """Per-name batch counts stamped onto every FunctionInvocationContext.

    Middleware with singleton semantics (whole-list replacement tools like
    ``todo_write``) reads the stamped count to reject same-batch duplicates
    deterministically instead of racing the gather.
    """

    @pytest.mark.asyncio
    async def test_two_same_name_calls_stamp_two_on_both(self) -> None:
        probe = _ProbeFunction()
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"}), ("c2", "echo", {"text": "b"})), _text_response()]
        )
        await layer.get_response([_user()], options={"tools": [_make_tool()]}, middleware=[probe])
        assert [ctx.same_tool_calls_in_batch for ctx in probe.contexts] == [2, 2]

    @pytest.mark.asyncio
    async def test_mixed_names_stamp_one_each(self) -> None:
        probe = _ProbeFunction()
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"}), ("c2", "other", {"text": "b"})), _text_response()]
        )
        await layer.get_response(
            [_user()],
            options={"tools": [_make_tool(), _make_tool(name="other")]},
            middleware=[probe],
        )
        assert {ctx.function.name: ctx.same_tool_calls_in_batch for ctx in probe.contexts} == {
            "echo": 1,
            "other": 1,
        }

    @pytest.mark.asyncio
    async def test_count_recomputed_per_batch(self) -> None:
        """A singleton batch after a duplicate batch stamps 1, not a stale 2."""
        probe = _ProbeFunction()
        layer, _wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"}), ("c2", "echo", {"text": "b"})),
                _call_response(("c3", "echo", {"text": "c"})),
                _text_response(),
            ]
        )
        await layer.get_response([_user()], options={"tools": [_make_tool()]}, middleware=[probe])
        assert [ctx.same_tool_calls_in_batch for ctx in probe.contexts] == [2, 2, 1]

    def test_default_is_one_outside_the_loop(self) -> None:
        context = FunctionInvocationContext(function=_make_tool(), arguments={})
        assert context.same_tool_calls_in_batch == 1


# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------


class TestStreamingLoop:
    @pytest.mark.asyncio
    async def test_stream_usage_normalizes_each_call_before_aggregating_iterations(self) -> None:
        first_call = [
            _call_update("c1", "echo", {"text": "a"}),
            ChatResponseUpdate(
                contents=[
                    Content.from_usage(
                        usage_details={"input_token_count": 20, "output_token_count": 1, "total_token_count": 21}
                    )
                ],
                role="assistant",
            ),
            ChatResponseUpdate(
                contents=[
                    Content.from_usage(
                        usage_details={"input_token_count": 20, "output_token_count": 2, "total_token_count": 22}
                    )
                ],
                role="assistant",
            ),
        ]
        final_call = [
            _text_update("done"),
            ChatResponseUpdate(
                contents=[
                    Content.from_usage(
                        usage_details={"input_token_count": 22, "output_token_count": 1, "total_token_count": 23}
                    )
                ],
                role="assistant",
            ),
            ChatResponseUpdate(
                contents=[
                    Content.from_usage(
                        usage_details={"input_token_count": 22, "output_token_count": 3, "total_token_count": 25}
                    )
                ],
                role="assistant",
            ),
        ]
        layer, wire = _stack([first_call, final_call])

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        _ = [update async for update in stream]
        response = await stream.get_final_response()

        assert len(wire.calls) == 2
        assert response.usage_details == {
            "input_token_count": 42,
            "output_token_count": 5,
            "total_token_count": 47,
        }
        assert response.latest_usage_details == {
            "input_token_count": 22,
            "output_token_count": 3,
            "total_token_count": 25,
        }

    @pytest.mark.asyncio
    async def test_stream_usage_identity_can_repeat_across_model_calls(self) -> None:
        usage = Content.from_usage(
            usage_details={"input_token_count": 3, "output_token_count": 1, "total_token_count": 4}
        )
        first_call = [
            _call_update("c1", "echo", {"text": "a"}),
            ChatResponseUpdate(contents=[usage], role="assistant"),
        ]
        final_call = [
            _text_update("done"),
            ChatResponseUpdate(contents=[usage], role="assistant"),
        ]
        layer, wire = _stack([first_call, final_call])

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        updates = [update async for update in stream]
        response = await stream.get_final_response()
        streamed_usage = [content for update in updates for content in update.contents if content.type == "usage"]

        assert len(wire.calls) == 2
        assert len(streamed_usage) == 2
        assert all(content is usage for content in streamed_usage)
        assert response.usage_details == {
            "input_token_count": 6,
            "output_token_count": 2,
            "total_token_count": 8,
        }
        assert response.latest_usage_details == {
            "input_token_count": 3,
            "output_token_count": 1,
            "total_token_count": 4,
        }

    @pytest.mark.asyncio
    async def test_streaming_informational_function_call_does_not_continue_local_loop(self) -> None:
        events: list[str] = []
        hosted_call = Content.from_function_call(
            "hosted-1",
            "echo",
            arguments={"text": "must stay remote"},
            informational_only=True,
        )
        update = ChatResponseUpdate(contents=[hosted_call], role="assistant")
        layer, wire = _stack([[update]])

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        observed = [item async for item in stream]
        result = await stream.get_final_response()

        assert observed == [update]
        assert events == []
        assert len(wire.calls) == 1
        assert result.messages[0].contents == [hosted_call]

    def test_stream_returns_synchronously_without_executing(self) -> None:
        layer, wire = _stack([[_text_update("hi")]])
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        assert isinstance(stream, ResponseStream)
        assert wire.calls == [], "stream construction must execute nothing"

    @pytest.mark.asyncio
    async def test_stream_single_turn_passthrough(self) -> None:
        layer, wire = _stack([[_text_update("hello")]])
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        updates = [u async for u in stream]
        assert len(updates) == 1
        final = await stream.get_final_response()
        assert final.messages[-1].contents[-1].text == "hello"
        assert len(wire.calls) == 1

    @pytest.mark.asyncio
    async def test_stream_tool_loop_yields_synthetic_tool_update(self) -> None:
        events: list[str] = []
        layer, wire = _stack([[_call_update("c1", "echo", {"text": "a"})], [_text_update("final")]])
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        updates = [u async for u in stream]
        assert events == ["tool:echo:a"]
        assert len(wire.calls) == 2
        roles = [u.role for u in updates]
        assert roles == ["assistant", "tool", "assistant"]
        tool_update = updates[1]
        assert tool_update.contents[0].type == "function_result"
        # from_updates rebuilds the full transcript including the tool results.
        final = await stream.get_final_response()
        kinds = [item.type for msg in final.messages for item in msg.contents]
        assert "function_call" in kinds
        assert "function_result" in kinds

    @pytest.mark.asyncio
    async def test_stream_truncated_response_surfaces_output_limit_message(self) -> None:
        # Streaming aggregates finish_reason via ``from_updates``; a "length"
        # cutoff must reach the truncation-aware error message on the stream path.
        bad_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c1", name="echo", arguments='{"text": "hello wor')],
            role="assistant",
            finish_reason="length",
        )
        layer, _wire = _stack([[bad_update], [_text_update("final")]])
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()
        results = _result_contents(final)
        assert len(results) == 1
        assert "max_tokens" in str(results[0].result)
        assert results[0].additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_truncated"

    @pytest.mark.asyncio
    async def test_stream_inner_result_hook_runs_before_tool_execution(self) -> None:
        events: list[str] = []

        def hook(response: ChatResponse) -> ChatResponse:
            events.append("hook")
            return response

        layer, _wire = _stack([[_call_update("c1", "echo", {"text": "a"})], [_text_update("final")]], result_hook=hook)
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        async for _ in stream:
            pass
        await stream.get_final_response()
        assert events == ["hook", "tool:echo:a", "hook"], (
            "the inner stream's result hooks must fire (via get_final_response) before tool extraction, "
            "and the outer finalizer must not re-run them"
        )

    @pytest.mark.asyncio
    async def test_stream_termination_stops_after_synthetic_update(self) -> None:
        layer, wire = _stack([[_call_update("c1", "echo", {"text": "a"})], [_text_update("never")]])
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()]},
            middleware=[_TerminateFunction(result="stopped")],
        )
        updates = [u async for u in stream]
        assert len(wire.calls) == 1, "termination must stop the loop before another wire call"
        assert updates[-1].role == "tool"
        assert str(updates[-1].contents[0].result) == "stopped"

    @pytest.mark.asyncio
    async def test_stream_conversation_id_continuation(self) -> None:
        session = AgentSession()
        call_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c1", name="echo", arguments={"text": "a"})],
            role="assistant",
            conversation_id="conv-1",
        )
        layer, wire = _stack([[call_update], [_text_update("final")]])
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()]},
            client_kwargs={"session": session, "x-extra": "kept"},
        )
        async for _ in stream:
            pass
        # Only the new tool-result message is resent; the service holds the rest.
        second = wire.calls[1]["messages"]
        assert [m.role for m in second] == ["tool"]
        # Continuation id written back into the shared client_kwargs dict + options.
        inner_client_kwargs = wire.calls[1]["kwargs"]["client_kwargs"]
        assert inner_client_kwargs["conversation_id"] == "conv-1"
        assert inner_client_kwargs["x-extra"] == "kept"
        assert "session" not in inner_client_kwargs
        assert "loop_recorder" not in inner_client_kwargs
        assert wire.calls[1]["options"]["conversation_id"] == "conv-1"
        assert session.service_session_id == "conv-1"

    @pytest.mark.asyncio
    async def test_stream_exhaustion_final_turn_without_tools(self) -> None:
        layer, wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [_text_update("forced")]],
            max_iterations=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        updates = [u async for u in stream]
        assert len(wire.calls) == 2
        assert wire.calls[1]["options"].get("tool_choice") == "none"
        assert updates[-1].contents[-1].text == "forced"


# ---------------------------------------------------------------------------
# Exhaustion-tail strip of unexecutable function calls
# ---------------------------------------------------------------------------


class TestExhaustionTailStrip:
    """The final ``tool_choice="none"`` turn never persists unexecutable calls.

    A provider that ignores the disabled tools would otherwise leave a
    dangling function call on the success path — a shape no later repair
    revisits and that rejects every subsequent turn of a stored conversation.
    """

    @pytest.mark.asyncio
    async def test_call_only_final_is_stripped_with_fallback(self) -> None:
        recorder = LoopRecorder()
        layer, wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"})),
                _call_response(("c2", "echo", {"text": "tool_choice ignored"})),
            ],
            max_iterations=1,
        )
        response = await layer.get_response(
            [_user()],
            options={"tools": [_make_tool()]},
            client_kwargs={"loop_recorder": recorder},
        )
        assert wire.calls[1]["options"].get("tool_choice") == "none"
        # The executed exchange survives; the misbehaving final call is gone
        # and the fallback text takes its place.
        assert [m.role for m in response.messages] == ["assistant", "tool", "assistant"]
        assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT
        call_ids = [c.call_id for m in response.messages for c in m.contents if c.type == "function_call"]
        assert call_ids == ["c1"]
        recorded_call_ids = [
            content.call_id
            for message in recorder.loop_messages or []
            for content in message.contents
            if content.type == "function_call"
        ]
        assert "c2" not in recorded_call_ids
        assert response._chrys_service_state_invalidated is False

    @pytest.mark.asyncio
    async def test_a_stripped_call_closes_its_operation_as_filtered(self) -> None:
        """The strip takes the call off the transcript; the operation it opened still has to end."""
        ignored = Content.from_function_call("c2", "echo", arguments={"text": "tool_choice ignored"})
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"})), ChatResponse(messages=[Message("assistant", [ignored])])],
            max_iterations=1,
        )
        sink = FakeSink()

        await layer.get_response(
            [_user()],
            options={"tools": [_make_tool()]},
            client_kwargs={TRAJECTORY_CONTEXT_KWARG: make_context(sink)},
        )

        filtered = [
            event
            for event in sink.of_type(TrajectoryEventType.TOOL_OPERATION_FINISHED)
            if event.payload["outcome"] == ToolOutcome.FILTERED
        ]
        assert len(filtered) == 1
        assert filtered[0].operation_id == ignored.additional_properties[OPERATION_ID_KEY]

    @pytest.mark.asyncio
    async def test_a_stripped_streamed_call_closes_its_operation_as_filtered(self) -> None:
        ignored = Content.from_function_call(call_id="c2", name="echo", arguments={"text": "tool_choice ignored"})
        layer, _wire = _stack(
            [
                [_call_update("c1", "echo", {"text": "a"})],
                [ChatResponseUpdate(contents=[ignored], role="assistant", finish_reason="tool_calls")],
            ],
            max_iterations=1,
        )
        sink = FakeSink()

        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()]},
            client_kwargs={TRAJECTORY_CONTEXT_KWARG: make_context(sink)},
        )
        _ = [u async for u in stream]
        await stream.get_final_response()

        filtered = [
            event
            for event in sink.of_type(TrajectoryEventType.TOOL_OPERATION_FINISHED)
            if event.payload["outcome"] == ToolOutcome.FILTERED
        ]
        assert len(filtered) == 1
        assert filtered[0].operation_id == ignored.additional_properties[OPERATION_ID_KEY]

    @pytest.mark.asyncio
    async def test_informational_call_survives_the_strip(self) -> None:
        informational = Content.from_function_call(
            "hosted-1",
            "remote_search",
            arguments={},
            informational_only=True,
        )
        final = ChatResponse(messages=[Message("assistant", [informational, Content.from_text("summarized")])])
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"})), final],
            max_iterations=1,
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        final_contents = response.messages[-1].contents
        assert [c.type for c in final_contents] == ["function_call", "text"]
        assert final_contents[0] is informational
        assert final_contents[1].text == "summarized"

    @pytest.mark.asyncio
    async def test_mixed_text_and_call_strip_normalizes_finish_reason(self) -> None:
        # Visible text survives the strip (no fallback), but the response
        # must stop advertising tool work that no longer exists.
        mixed_final = ChatResponse(
            messages=[
                Message(
                    "assistant",
                    [
                        Content.from_text("half an answer"),
                        Content.from_function_call("c2", "echo", arguments={"text": "b"}),
                    ],
                )
            ],
            finish_reason="tool_calls",
        )
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"})), mixed_final],
            max_iterations=1,
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        assert response.messages[-1].text == "half an answer"
        assert _MAX_ITERATIONS_FALLBACK_TEXT not in (response.text or "")
        call_ids = [c.call_id for m in response.messages for c in m.contents if c.type == "function_call"]
        assert call_ids == ["c1"]
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_mixed_text_and_call_strip_normalizes_finish_reason(self) -> None:
        mixed_update = ChatResponseUpdate(
            contents=[
                Content.from_text("half an answer"),
                Content.from_function_call(call_id="c2", name="echo", arguments={"text": "b"}),
            ],
            role="assistant",
            finish_reason="tool_calls",
        )
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [mixed_update]],
            max_iterations=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        updates = [u async for u in stream]
        response = await stream.get_final_response()
        assert response.messages[-1].text == "half an answer"
        assert _MAX_ITERATIONS_FALLBACK_TEXT not in (response.text or "")
        # The corrective metadata update carries the normalized reason so the
        # latest-non-null assembly and streamed consumers both see "stop".
        assert updates[-1].contents == []
        assert updates[-1].finish_reason == "stop"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_blank_final_synthesizes_fallback(self) -> None:
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"})), ChatResponse(messages=[])],
            max_iterations=1,
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        assert response.messages[-1].role == "assistant"
        assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT

    @pytest.mark.asyncio
    async def test_reasoning_only_final_synthesizes_fallback(self) -> None:
        reasoning_final = ChatResponse(messages=[Message("assistant", [Content.from_text_reasoning(text="pondering")])])
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"})), reasoning_final],
            max_iterations=1,
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        # Nothing was stripped; the reasoning survives with the fallback beside it.
        assert any(c.type == "text_reasoning" for m in response.messages for c in m.contents)
        assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT

    @pytest.mark.asyncio
    async def test_stream_call_only_final_strips_updates_and_yields_fallback(self) -> None:
        exhaustion_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c2", name="echo", arguments={"text": "b"})],
            role="assistant",
            finish_reason="tool_calls",
        )
        layer, wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [exhaustion_update]],
            max_iterations=1,
        )
        sink = FakeSink()
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()]},
            client_kwargs={TRAJECTORY_CONTEXT_KWARG: make_context(sink)},
        )
        updates = [u async for u in stream]
        response = await stream.get_final_response()
        assert wire.calls[1]["options"].get("tool_choice") == "none"
        # The call the tail asked not to receive is stripped from the updates
        # but still on the response the cycle is counted from, so the log says
        # the provider ignored ``tool_choice="none"`` rather than complied.
        counted = [
            event.payload["function_call_count"] for event in sink.of_type(TrajectoryEventType.MODEL_CYCLE_FINISHED)
        ]
        assert counted == [1, 1]
        streamed_call_ids = [c.call_id for u in updates for c in u.contents if c.type == "function_call"]
        assert streamed_call_ids == ["c1"], "the exhaustion turn's call must not reach the stream"
        assert updates[-1].contents[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT
        # The fallback ends the run: its "stop" must override the stripped
        # update's "tool_calls" in the latest-non-null assembly.
        assert updates[-1].finish_reason == "stop"
        assert response.finish_reason == "stop"
        assert [m.role for m in response.messages] == ["assistant", "tool", "assistant"]
        assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT
        call_ids = [c.call_id for m in response.messages for c in m.contents if c.type == "function_call"]
        assert call_ids == ["c1"]

    @pytest.mark.asyncio
    async def test_stream_emptied_update_with_metadata_still_yields(self) -> None:
        final_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c2", name="echo", arguments={"text": "b"})],
            role="assistant",
            response_id="resp-2",
            finish_reason="stop",
        )
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [final_update]],
            max_iterations=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        updates = [u async for u in stream]
        response = await stream.get_final_response()
        emptied = [u for u in updates if u.response_id == "resp-2" and not u.contents]
        assert len(emptied) == 1
        assert emptied[0] is not final_update, "yielded update is a copy"
        # The stripped copy must not carry the provider's terminal reason:
        # the tail re-emits the corrected one after every stripped update,
        # and a consumer keying on the first terminal reason would otherwise
        # treat the run as complete on withheld calls.
        assert emptied[0].finish_reason is None
        # Copy-on-write: the retry stream's recorded update keeps the call
        # and its reason, so the post-assembly strip still observes both.
        assert final_update.contents[0].call_id == "c2"
        assert final_update.finish_reason == "stop"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_terminal_reason_arrives_once_and_last(self) -> None:
        exhaustion_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c2", name="echo", arguments={"text": "b"})],
            role="assistant",
            finish_reason="tool_calls",
        )
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [exhaustion_update]],
            max_iterations=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        updates = [u async for u in stream]
        response = await stream.get_final_response()
        # A reason-only stripped update has nothing left once the terminal
        # reason is cleared, so it drops from the stream entirely; the
        # provider's "tool_calls" verdict on withheld work must never reach
        # a streaming consumer.
        assert all(u.finish_reason != "tool_calls" for u in updates)
        terminal = [i for i, u in enumerate(updates) if u.finish_reason in ("tool_calls", "stop")]
        assert len(terminal) == 1, "exactly one terminal reason crosses the stream"
        assert terminal[0] == len(updates) - 1, "the corrected terminal reason arrives last"
        assert updates[-1].finish_reason == "stop"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_separate_terminal_chunk_suppressed_after_strip(self) -> None:
        # Providers commonly emit the call delta and the finish-reason chunk
        # separately. The stateless per-update strip cannot see across
        # chunks, so the tail must remember the strip and suppress the later
        # reason-only chunk — otherwise it crosses untouched and a
        # first-terminal consumer stops before the corrective update.
        tail_turn = [
            _call_update("c2", "echo", {"text": "b"}),
            ChatResponseUpdate(contents=[], role="assistant", finish_reason="tool_calls"),
        ]
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], tail_turn],
            max_iterations=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        updates = [u async for u in stream]
        response = await stream.get_final_response()
        reasons = [u.finish_reason for u in updates if u.finish_reason is not None]
        assert reasons == ["stop"], "only the corrective terminal reason may cross the stream"
        assert updates[-1].finish_reason == "stop"
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stream_stripped_length_reason_waits_for_corrective_update(self) -> None:
        # "length" on a stripped update is the provider's verdict on a turn
        # that ended in withheld calls; letting it cross ahead of the
        # corrective update makes a first-terminal consumer stop before the
        # fallback text arrives.
        exhaustion_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c2", name="echo", arguments={"text": "b"})],
            role="assistant",
            finish_reason="length",
        )
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [exhaustion_update]],
            max_iterations=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        updates = [u async for u in stream]
        await stream.get_final_response()
        reasons = [u.finish_reason for u in updates if u.finish_reason is not None]
        assert reasons == ["stop"], "the fallback's terminal reason must be the only one crossing"
        assert updates[-1].contents and updates[-1].contents[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT

    @pytest.mark.asyncio
    async def test_stream_corrective_reemits_non_stop_reason_after_suppression(self) -> None:
        # Visible text in the tail means no fallback update; the suppressed
        # provider reason ("length" here, via a separate reason-only chunk)
        # must still reach the consumer once — re-emitted by the corrective
        # update after every stripped update, matching the assembled
        # response's reason.
        tail_turn = [
            _text_update("partial answer"),
            _call_update("c2", "echo", {"text": "b"}),
            ChatResponseUpdate(contents=[], role="assistant", finish_reason="length"),
        ]
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], tail_turn],
            max_iterations=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        updates = [u async for u in stream]
        response = await stream.get_final_response()
        reasons = [u.finish_reason for u in updates if u.finish_reason is not None]
        assert reasons == ["length"], "the corrective update re-emits the assembled response's reason"
        assert updates[-1].finish_reason == "length"
        assert updates[-1].contents == []
        assert response.finish_reason == "length"
        assert any(c.text == "partial answer" for m in response.messages for c in m.contents if c.type == "text")

    def test_strip_update_helper_metadata_exemptions(self) -> None:
        def call_update(**kwargs: Any) -> ChatResponseUpdate:
            return ChatResponseUpdate(
                contents=[Content.from_function_call(call_id="c9", name="echo", arguments={})],
                role="assistant",
                **kwargs,
            )

        assert _strip_unexecutable_calls_from_update(call_update()) is None
        for exempting in (
            {"response_id": "resp-1"},
            {"conversation_id": "conv-1"},
            {"continuation_token": {"response_id": "bg-1"}},
            {"model": "m-1"},
            {"created_at": "2026-01-01"},
            {"message_id": "msg-1"},
            {"author_name": "agent"},
            {"additional_properties": {"provider": "meta"}},
        ):
            kept = _strip_unexecutable_calls_from_update(call_update(**exempting))
            assert kept is not None, f"metadata {exempting} must exempt the emptied update"
            assert kept.contents == []
        # finish_reason never exempts and every stripped copy sheds it — the
        # provider concluded the turn on the withheld calls, so any reason it
        # attached (terminal or "length") must wait for the corrective final
        # update. The original stays untouched for the post-assembly strip.
        for reason in ("stop", "tool_calls", "length"):
            assert _strip_unexecutable_calls_from_update(call_update(finish_reason=reason)) is None
            original = call_update(response_id="resp-1", finish_reason=reason)
            kept = _strip_unexecutable_calls_from_update(original)
            assert kept is not None
            assert kept.finish_reason is None
            assert original.finish_reason == reason
        passthrough = _text_update("hello")
        assert _strip_unexecutable_calls_from_update(passthrough) is passthrough

    def test_strip_update_helper_suppresses_later_reason_only_chunks(self) -> None:
        # Providers commonly emit the call delta and the finish-reason chunk
        # separately; once the tail has stripped a call, the caller raises
        # suppress_finish_reason and the later chunk must not carry the
        # provider's verdict across the stream.
        reason_only = ChatResponseUpdate(contents=[], role="assistant", finish_reason="tool_calls")
        assert _strip_unexecutable_calls_from_update(reason_only) is reason_only, (
            "without the flag a call-less update passes through untouched"
        )
        assert _strip_unexecutable_calls_from_update(reason_only, suppress_finish_reason=True) is None
        assert reason_only.finish_reason == "tool_calls", "the recorded original stays untouched"
        with_metadata = ChatResponseUpdate(contents=[], role="assistant", response_id="resp-1", finish_reason="length")
        kept = _strip_unexecutable_calls_from_update(with_metadata, suppress_finish_reason=True)
        assert kept is not None
        assert kept.finish_reason is None
        assert kept.response_id == "resp-1"
        reasonless = _text_update("hello")
        assert _strip_unexecutable_calls_from_update(reasonless, suppress_finish_reason=True) is reasonless

    @pytest.mark.asyncio
    async def test_strip_invalidates_service_continuation_blocking(self) -> None:
        session = AgentSession()
        final = ChatResponse(
            messages=[Message("assistant", [Content.from_function_call("c2", "echo", arguments={"text": "b"})])],
            conversation_id="conv-svc",
            response_id="resp-svc",
        )
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"})), final],
            max_iterations=1,
        )
        response = await layer.get_response(
            [_user()],
            options={"tools": [_make_tool()], "store": True},
            client_kwargs={"session": session},
        )
        # The service-side transcript still holds the stripped call unanswered;
        # its continuation handle must not survive anywhere.
        assert response.conversation_id is None
        assert response.response_id is None
        assert response._chrys_service_state_invalidated is True
        assert session.service_session_id is None

    @pytest.mark.asyncio
    async def test_strip_without_service_storage_preserves_response_metadata(self) -> None:
        # Client-side storage: the service holds nothing, so the strip must
        # not erase ids a local-storage client attached as plain metadata.
        final = ChatResponse(
            messages=[Message("assistant", [Content.from_function_call("c2", "echo", arguments={"text": "b"})])],
            conversation_id="conv-local",
            response_id="resp-local",
        )
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"})), final],
            max_iterations=1,
        )
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        assert response.conversation_id == "conv-local"
        assert response.response_id == "resp-local"
        assert response._chrys_service_state_invalidated is False
        assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT

    @pytest.mark.asyncio
    async def test_stream_strip_invalidates_service_continuation(self) -> None:
        session = AgentSession()
        final_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c2", name="echo", arguments={"text": "b"})],
            role="assistant",
            conversation_id="conv-svc",
            response_id="resp-svc",
        )
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [final_update]],
            max_iterations=1,
        )
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()], "store": True},
            client_kwargs={"session": session},
        )
        updates = [u async for u in stream]
        response = await stream.get_final_response()
        # The emptied copy still yields (it carries conversation metadata) …
        emptied = [u for u in updates if u.conversation_id == "conv-svc"]
        assert len(emptied) == 1
        assert emptied[0].contents == []
        # … but the finalized response withholds the stale handles even though
        # ``from_updates`` would restore them from that update.
        assert response.conversation_id is None
        assert response.response_id is None
        assert response._chrys_service_state_invalidated is True
        assert session.service_session_id is None
        assert response.messages[-1].text == _MAX_ITERATIONS_FALLBACK_TEXT

    @pytest.mark.asyncio
    async def test_service_tail_restores_fresh_handle_after_finalization(self) -> None:
        # Entering the service-stored tail clears the consumed handle; a
        # successfully finalized, non-invalidated tail restores the fresh one.
        session = AgentSession()
        session.service_session_id = "conv-old"
        tail_updates = [
            ChatResponseUpdate(contents=[Content.from_text("done")], role="assistant"),
            ChatResponseUpdate(contents=[], role="assistant", conversation_id="conv-new"),
        ]
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], tail_updates],
            max_iterations=1,
        )
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()], "store": True},
            client_kwargs={"session": session},
        )
        [u async for u in stream]
        response = await stream.get_final_response()
        assert response.messages[-1].text == "done"
        assert response._chrys_service_state_invalidated is False
        assert session.service_session_id == "conv-new"

    @pytest.mark.asyncio
    async def test_blocking_middleware_termination_invalidates_service_state(self) -> None:
        # The termination return issues no further request, so the service
        # transcript behind the mirrored handle keeps this batch's calls
        # unanswered — the early return must carry the same invalidation
        # verdict as the exhaustion strip.
        class _Terminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                raise MiddlewareTermination("stop")

        session = AgentSession()
        layer, _wire = _stack(
            [_call_response(("c1", "echo", {"text": "a"}), conversation_id="conv-old", response_id="resp-old")],
            max_iterations=3,
        )
        response = await layer.get_response(
            [_user()],
            options={"tools": [_make_tool()], "store": True},
            middleware=[_Terminate()],
            client_kwargs={"session": session},
        )
        assert response._chrys_service_state_invalidated is True
        assert response.conversation_id is None
        assert response.response_id is None
        assert session.service_session_id is None
        assert {"conv-old", "resp-old"} <= session.invalidated_service_session_ids

    @pytest.mark.asyncio
    async def test_stream_middleware_termination_invalidates_service_state(self) -> None:
        # Termination on a NON-last iteration: the verdict must not lean on
        # the last-iteration pre-clear, and ``_finalize`` must withhold the
        # handles ``from_updates`` would restore from the raw updates.
        class _Terminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                raise MiddlewareTermination("stop")

        session = AgentSession()
        metadata_update = ChatResponseUpdate(
            contents=[],
            role="assistant",
            conversation_id="conv-old",
            response_id="resp-old",
        )
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"}), metadata_update]],
            max_iterations=3,
        )
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()], "store": True},
            middleware=[_Terminate()],
            client_kwargs={"session": session},
        )
        [u async for u in stream]
        response = await stream.get_final_response()
        assert response._chrys_service_state_invalidated is True
        assert response.conversation_id is None
        assert response.response_id is None
        assert session.service_session_id is None
        assert {"conv-old", "resp-old"} <= session.invalidated_service_session_ids

    @pytest.mark.asyncio
    async def test_stream_last_iteration_termination_still_records_session_handle(self) -> None:
        # On the LAST iteration the pre-clear ahead of the results yield
        # would drop the session's mirrored handle unrecorded; the
        # termination verdict must win that ordering. The round response
        # carries no ids, so the session field is the only handle source.
        class _Terminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                raise MiddlewareTermination("stop")

        session = AgentSession()
        session.service_session_id = "conv-old"
        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})]],
            max_iterations=1,
        )
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()], "store": True},
            middleware=[_Terminate()],
            client_kwargs={"session": session},
        )
        [u async for u in stream]
        response = await stream.get_final_response()
        assert response._chrys_service_state_invalidated is True
        assert session.service_session_id is None
        assert "conv-old" in session.invalidated_service_session_ids


# ---------------------------------------------------------------------------
# LoopRecorder
# ---------------------------------------------------------------------------


class TestLoopRecorder:
    @pytest.mark.asyncio
    async def test_pre_call_snapshot_and_slice_semantics(self) -> None:
        recorder = LoopRecorder()
        first = [_user("a")]
        assert recorder.loop_messages is None
        await recorder.record_pre_call(first)
        assert recorder.initial_count == 1
        assert recorder.captured_count == 1
        assert recorder.loop_messages is None, "single iteration → no loop messages"
        grown = [*first, Message(role="assistant", contents=["x"]), Message(role="tool", contents=["y"])]
        await recorder.record_pre_call(grown)
        assert recorder.initial_count == 1, "initial count locks on first call"
        tail = recorder.loop_messages
        assert tail is not None
        assert [m.role for m in tail] == ["assistant", "tool"]
        assert tail[0] is grown[1], "snapshot preserves message object identity"

    @pytest.mark.asyncio
    async def test_checkpoint_fires_on_change_and_suppresses_on_identical_prefix(self) -> None:
        fired: list[int] = []

        async def on_checkpoint() -> None:
            fired.append(1)

        hashed: list[str] = []

        def hasher(message: Message) -> str:
            payload = f"hashed:{message.role}"
            hashed.append(payload)
            return payload

        recorder = LoopRecorder(on_checkpoint=on_checkpoint, message_hasher=hasher)
        messages = [_user("a")]
        await recorder.record_pre_call(messages)
        await recorder.record_pre_call(list(messages))
        assert len(fired) == 1, "identical prefix must suppress the second checkpoint"
        assert hashed, "the injected hasher must feed the suppression key"
        await recorder.record_pre_call([*messages, Message(role="assistant", contents=["x"])])
        assert len(fired) == 2

    @pytest.mark.asyncio
    async def test_checkpoint_hash_accepts_surrogates_with_production_message_hasher(self) -> None:
        fired: list[int] = []

        async def on_checkpoint() -> None:
            fired.append(1)

        recorder = LoopRecorder(on_checkpoint=on_checkpoint, message_hasher=serialized_message_payload)
        messages = [Message(role="tool", contents=["result with split pair \ud83c\udf0d and lone \ud83c"])]

        await recorder.record_pre_call(messages)
        await recorder.record_pre_call(list(messages))

        assert len(fired) == 1, "the unchanged production payload must still suppress duplicate checkpoints"

    @pytest.mark.asyncio
    async def test_checkpoint_hash_does_not_alias_surrogate_to_literal_escape(self) -> None:
        fired: list[int] = []

        async def on_checkpoint() -> None:
            fired.append(1)

        def text_hasher(message: Message) -> str:
            return message.contents[0].text or ""

        recorder = LoopRecorder(on_checkpoint=on_checkpoint, message_hasher=text_hasher)
        await recorder.record_pre_call([Message(role="tool", contents=["\ud83c"])])
        await recorder.record_pre_call([Message(role="tool", contents=[r"\ud83c"])])

        assert len(fired) == 2, "a surrogate and the literal escape that spells it must hash differently"

    @pytest.mark.asyncio
    async def test_service_mode_capture_paths(self) -> None:
        recorder = LoopRecorder(capture_service_loop_messages=True)
        await recorder.record_pre_call([_user("a")])
        call_msg = Message(
            role="assistant", contents=[Content.from_function_call(call_id="c1", name="t", arguments={})]
        )
        recorder.record_response(ChatResponse(messages=[call_msg]))
        # Seeded by the response capture; later pre-calls collect assistant/tool messages.
        tool_msg = Message(role="tool", contents=[Content.from_function_result(call_id="c1", result="r")])
        await recorder.record_pre_call([tool_msg])
        loop_messages = recorder.loop_messages
        assert loop_messages is not None
        assert loop_messages[0] is call_msg
        assert loop_messages[1] is tool_msg
        # Identity dedup: recording the same objects again must not duplicate.
        recorder.record_response(ChatResponse(messages=[call_msg]))
        await recorder.record_pre_call([tool_msg])
        assert len(recorder.loop_messages or []) == 2

    @pytest.mark.asyncio
    async def test_non_service_mode_ignores_record_response(self) -> None:
        recorder = LoopRecorder()
        call_msg = Message(
            role="assistant", contents=[Content.from_function_call(call_id="c1", name="t", arguments={})]
        )
        recorder.record_response(ChatResponse(messages=[call_msg]))
        await recorder.record_pre_call([_user()])
        assert recorder.loop_messages is None

    @pytest.mark.asyncio
    async def test_reset_clears_all_state(self) -> None:
        recorder = LoopRecorder(capture_service_loop_messages=True)
        await recorder.record_pre_call([_user()])
        recorder.record_response(
            ChatResponse(
                messages=[
                    Message(
                        role="assistant", contents=[Content.from_function_call(call_id="c", name="t", arguments={})]
                    )
                ]
            )
        )
        recorder.reset()
        assert recorder.initial_count is None
        assert recorder.captured_count is None
        assert recorder.loop_messages is None

    @pytest.mark.asyncio
    async def test_retry_snapshot_restores_every_mutable_recorder_field(self) -> None:
        recorder = LoopRecorder(capture_service_loop_messages=True)
        user = _user("start")
        await recorder.record_pre_call([user])
        call = Message(role="assistant", contents=[Content.from_function_call(call_id="c1", name="echo", arguments={})])
        recorder.record_response(ChatResponse(messages=[call]))
        snapshot = recorder.snapshot()

        await recorder.record_pre_call([user, call, Message(role="tool", contents=["failed"])])
        recorder.record_response(_call_response(("c2", "echo", {"text": "failed"})))
        recorder.restore(snapshot)

        assert recorder.initial_count == 1
        assert recorder.captured_count == 1
        assert recorder.loop_messages == [call]

    @pytest.mark.asyncio
    async def test_loop_feeds_recorder_via_client_kwargs(self) -> None:
        recorder = LoopRecorder()
        layer, wire = _stack([_call_response(("c1", "echo", {"text": "a"})), _text_response()])
        await layer.get_response(
            [_user()],
            options={"tools": [_make_tool()]},
            client_kwargs={"loop_recorder": recorder},
        )
        assert recorder.initial_count == 1
        tail = recorder.loop_messages
        assert tail is not None
        assert [m.role for m in tail] == ["assistant", "tool"]
        # The recorder key never reaches the wire.
        for call in wire.calls:
            assert "loop_recorder" not in call["kwargs"].get("client_kwargs", {})

    @pytest.mark.asyncio
    async def test_concurrent_runs_use_isolated_recorders(self) -> None:
        rec_a = LoopRecorder()
        rec_b = LoopRecorder()
        gate = asyncio.Barrier(2)

        @tool(name="sync")
        async def sync_tool(text: str) -> str:
            await asyncio.wait_for(gate.wait(), timeout=2)
            return text

        wire = _ScriptedClient(
            [
                _call_response(("c1", "sync", {"text": "a"})),
                _call_response(("c2", "sync", {"text": "b"})),
                _text_response("a-final"),
                _text_response("b-final"),
            ]
        )
        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))
        run_a = layer.get_response([_user("a")], options={"tools": [sync_tool]}, client_kwargs={"loop_recorder": rec_a})
        run_b = layer.get_response(
            [_user("b"), _user("b2")], options={"tools": [sync_tool]}, client_kwargs={"loop_recorder": rec_b}
        )
        await asyncio.gather(run_a, run_b)
        assert rec_a.initial_count == 1
        assert rec_b.initial_count == 2
        assert len(rec_a.loop_messages or []) == 2
        assert len(rec_b.loop_messages or []) == 2

    @pytest.mark.asyncio
    async def test_concurrent_runs_on_shared_stack_strip_only_their_own_echoes(self) -> None:
        """Two simultaneous runs share one client stack; the echo memo is
        PER-RUN. A content object from run A's history echoed by the shared
        client is stripped in A (echo of A's conversation) but lands in B
        (fresh work there) — a stack-shared memo would strip B's copy too."""
        shared_x = Content.from_text("A-history")
        history_a = [_user("a"), Message("assistant", [shared_x]), _user("go")]
        history_b = [_user("b")]
        gate = asyncio.Barrier(2)

        @tool(name="sync")
        async def sync_tool(text: str) -> str:
            await asyncio.wait_for(gate.wait(), timeout=2)
            return text

        # Both second-round turns carry the SAME object from A's history, so
        # the pin holds regardless of which run receives which turn.
        wire = _ScriptedClient(
            [
                _call_response(("c1", "sync", {"text": "a"})),
                _call_response(("c2", "sync", {"text": "b"})),
                ChatResponse(messages=[Message("assistant", [shared_x, Content.from_text("fresh-1")])]),
                ChatResponse(messages=[Message("assistant", [shared_x, Content.from_text("fresh-2")])]),
            ]
        )
        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))

        response_a, response_b = await asyncio.gather(
            layer.get_response(history_a, options={"tools": [sync_tool]}),
            layer.get_response(history_b, options={"tools": [sync_tool]}),
        )

        contents_a = [c for m in response_a.messages for c in m.contents]
        contents_b = [c for m in response_b.messages for c in m.contents]
        assert all(c is not shared_x for c in contents_a), "A strips its own history echo"
        assert sum(1 for c in contents_b if c is shared_x) == 1, "B lands A's object as fresh work"
        assert any(getattr(c, "text", None) in {"fresh-1", "fresh-2"} for c in contents_a)


# ---------------------------------------------------------------------------
# Per-invocation logging (Phase 2 N1) — chrys-owned tool call log lines
# ---------------------------------------------------------------------------


_LOOP_LOGGER = "chrys.kernel.loop"


def _loop_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _LOOP_LOGGER]


class TestInvocationLogging:
    @pytest.mark.asyncio
    async def test_gate_off_logs_summaries_under_chrys_logger(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", False)
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "hello"})), _text_response()])
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            await layer.get_response([_user()], options={"tools": [_make_tool()]})

        records = _loop_records(caplog)
        by_level = {(r.levelno, r.getMessage()) for r in records}
        assert (logging.INFO, "Function name: echo") in by_level
        assert (logging.DEBUG, "Function arguments: 1 key(s) (text)") in by_level
        assert any(
            lvl == logging.INFO and msg.startswith("Function echo succeeded in") and msg.endswith("s.")
            for lvl, msg in by_level
        )
        assert (logging.DEBUG, "Function result: 1 item(s) (text)") in by_level
        # Raw argument / result text must not leak with the gate off.
        assert all("hello" not in r.getMessage() for r in records)

    @pytest.mark.asyncio
    async def test_gate_off_does_not_disclose_unrecognized_argument_keys(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Schemas rarely pin ``additionalProperties: false``, so model-supplied
        keys survive validation — a key name carrying secret text must not
        reach the gate-off log line."""
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", False)

        async def lookup(**kwargs: Any) -> str:
            return "ok"

        schema_tool = FunctionTool(
            name="lookup",
            description="schema-supplied tool",
            func=lookup,
            input_model={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
        layer, _wire = _stack(
            [_call_response(("c1", "lookup", {"query": "x", "hunter2-secret": "y"})), _text_response()]
        )
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            await layer.get_response([_user()], options={"tools": [schema_tool]})

        messages = [r.getMessage() for r in _loop_records(caplog)]
        assert "Function arguments: 2 key(s) (query, +1 unrecognized)" in messages
        assert all("hunter2" not in message for message in messages)

    @pytest.mark.asyncio
    async def test_gate_off_handles_basemodel_argument_rewrite(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The middleware contract admits ``BaseModel`` rewrites of
        ``context.arguments`` — the describe helper must not crash on one
        (the invoke site has no try/except, so a describe error would be
        converted into a tool error result)."""
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", False)
        echo_tool = _make_tool()
        input_model = echo_tool.input_model
        assert input_model is not None

        class _RewriteToModel(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                context.arguments = input_model(text="hello")
                await call_next()

        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "hi"})), _text_response()])
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            response = await layer.get_response(
                [_user()], options={"tools": [echo_tool]}, middleware=[_RewriteToModel()]
            )

        messages = [r.getMessage() for r in _loop_records(caplog)]
        assert "Function arguments: 1 key(s) (text)" in messages
        assert all("hello" not in message for message in messages)
        assert str(_result_contents(response)[0].result) == "echo:hello"

    @pytest.mark.asyncio
    async def test_gate_on_logs_raw_arguments_and_result(
        self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", True)
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "hello"})), _text_response()])
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            await layer.get_response([_user()], options={"tools": [_make_tool()]})

        messages = [r.getMessage() for r in _loop_records(caplog)]
        assert "Function arguments: {'text': 'hello'}" in messages
        assert "Function result: echo:hello" in messages

    @pytest.mark.parametrize(("sensitive", "expected"), [(False, "list"), (True, "[1, 2, 3]")])
    @pytest.mark.asyncio
    async def test_gate_off_skip_parsing_raw_list_result_does_not_crash_describe(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
        *,
        sensitive: bool,
        expected: str,
    ) -> None:
        """A skip-parsing tool returns the raw value — possibly a non-Content
        ``list`` — so the result describe helper must render it scalar-style
        (framework ``:697`` / ``:756``) instead of iterating ``.type``.  The
        invoke site has no try/except, so a describe crash would convert a
        successful call into a tool error result."""
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", sensitive)

        @tool(name="rawlist", result_parser=SKIP_PARSING)
        async def rawlist(n: int) -> list[int]:
            return [1, 2, 3]

        layer, _wire = _stack([_call_response(("c1", "rawlist", {"n": 1})), _text_response()])
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            response = await layer.get_response([_user()], options={"tools": [rawlist]})

        results = _result_contents(response)
        assert results and results[0].exception is None
        assert str(results[0].result) == "[1, 2, 3]"
        assert f"Function result: {expected}" in [r.getMessage() for r in _loop_records(caplog)]

    @pytest.mark.asyncio
    async def test_failure_logs_single_warning_no_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Failure reporting is owned by the loop's outer except (one WARNING).
        The chrys invoke is logging-free and the loop-owned per-tool span
        already records the exception (N5), so an extra ERROR at the invoke
        site would only duplicate that single WARNING."""

        @tool(name="boom")
        async def boom(text: str) -> str:
            raise RuntimeError("kaput")

        layer, _wire = _stack([_call_response(("c1", "boom", {"text": "x"})), _text_response()])
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            response = await layer.get_response([_user()], options={"tools": [boom]})

        records = _loop_records(caplog)
        assert [r for r in records if r.levelno >= logging.ERROR] == []
        warnings = [r for r in records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "Function 'boom' raised an exception" in message
        assert "kaput" in message
        assert not any("succeeded" in r.getMessage() for r in records)
        # The loop still converts the failure into an error result for the model.
        assert str(_result_contents(response)[0].result) == "Error: Function failed."

    @pytest.mark.asyncio
    async def test_termination_through_invoke_logs_nothing_above_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """A termination raised inside the tool (e.g. an interrupt propagating
        out of a nested sub-agent run) is control flow, not a failure — no
        WARNING/ERROR record may be emitted for it."""
        canned = Content.from_function_result(call_id="c1", result="interrupted")

        @tool(name="interruptible")
        async def interruptible(text: str) -> str:
            raise MiddlewareTermination("stop", result=canned)

        layer, _wire = _stack([_call_response(("c1", "interruptible", {"text": "x"}))])
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            await layer.get_response([_user()], options={"tools": [interruptible]})

        records = _loop_records(caplog)
        assert all(r.levelno <= logging.INFO for r in records)
        assert any(r.getMessage() == "Function name: interruptible" for r in records)
        assert not any("succeeded" in r.getMessage() for r in records)

    @pytest.mark.asyncio
    async def test_blocked_invocation_emits_no_log_lines(self, caplog: pytest.LogCaptureFixture) -> None:
        """Middleware termination before the final handler must not log an invocation."""
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "a"}))])
        with caplog.at_level(logging.DEBUG, logger=_LOOP_LOGGER):
            await layer.get_response(
                [_user()],
                options={"tools": [_make_tool()]},
                middleware=[_TerminateFunction(result="interrupted")],
            )

        assert _loop_records(caplog) == []


# ---------------------------------------------------------------------------
# Drift contracts — private upstream surface the mirror depends on
# ---------------------------------------------------------------------------


class TestDriftContracts:
    def test_loop_defaults_stay_stable(self) -> None:
        assert DEFAULT_MAX_ITERATIONS == 40
        assert DEFAULT_MAX_CONSECUTIVE_ERRORS == 3

    def test_function_tool_private_surface(self) -> None:
        ft = _make_tool()
        assert hasattr(ft, "_schema_supplied")
        assert hasattr(ft, "_context_parameter_name")
        params = inspect.signature(FunctionTool.invoke).parameters
        for name in ("arguments", "context", "tool_call_id"):
            assert name in params, f"FunctionTool.invoke lost the {name!r} parameter"
        assert callable(Content.from_function_call(call_id="c", name="n", arguments={}).parse_arguments)

    @pytest.mark.parametrize(
        "arguments,schema,expected",
        [
            ({}, {"required": ["a"], "properties": {"a": {"type": "string"}}}, "type_error"),
            (
                {"a": "x", "b": 1},
                {"properties": {"a": {"type": "string"}}, "additionalProperties": False},
                "type_error",
            ),
            ({"a": "x"}, {"properties": {"a": {"type": "string", "enum": ["y", "z"]}}}, "type_error"),
            ({"a": 1}, {"properties": {"a": {"type": "string"}}}, "type_error"),
            ({"a": True}, {"properties": {"a": {"type": "integer"}}}, "type_error"),
            ({"a": "x"}, {"properties": {"a": {"type": ["integer", "boolean"]}}}, "type_error"),
            ({"a": "x"}, {"properties": {"a": {"type": "string"}}}, "ok"),
            ({"a": [1]}, {"properties": {"a": {"type": "array"}}}, "ok"),
            ({"a": {"k": 1}}, {"properties": {"a": {"type": "object"}}}, "ok"),
            ({"a": None}, {"properties": {"a": {"type": "null"}}}, "ok"),
            ({"a": "x"}, {"properties": {"a": {"type": "made-up-type"}}}, "ok"),
            ({"a": 2}, {"properties": {"a": {"type": ["integer", "boolean"]}}}, "ok"),
        ],
    )
    def test_validation_contract(self, arguments: dict[str, Any], schema: dict[str, Any], expected: str) -> None:
        def outcome(fn: Any) -> tuple[str, Any]:
            try:
                return ("ok", fn(arguments=arguments, schema=schema, tool_name="t"))
            except TypeError:
                return ("type_error", None)

        assert outcome(_validate_arguments_against_schema)[0] == expected

    @pytest.mark.parametrize(
        "value,schema_type,expected",
        [
            ("s", "string", True),
            (1, "integer", True),
            (True, "integer", False),
            (1.5, "number", True),
            (True, "number", False),
            (True, "boolean", True),
            ([1], "array", True),
            ({"a": 1}, "object", True),
            (None, "null", True),
            ("anything", "unknown-type", True),
        ],
    )
    def test_type_matcher_contract(self, value: Any, schema_type: str, expected: bool) -> None:
        assert _matches_json_schema_type(value, schema_type) is expected


# ---------------------------------------------------------------------------
# Chrys per-tool span (N5) — the loop-side mirror of the framework's
# instrumented-invoke block
# ---------------------------------------------------------------------------


class _SpanProbe:
    """Recording stand-ins for the telemetry seams the span path touches."""

    class _Span:
        def __init__(self, name: str, attributes: dict[Any, Any]) -> None:
            self.name = name
            self.attributes = dict(attributes)
            self.set_attrs: dict[Any, Any] = {}
            self.exceptions: list[BaseException] = []
            self.status: Any = None

        def set_attribute(self, key: Any, value: Any) -> None:
            self.set_attrs[key] = value

        def record_exception(self, exception: BaseException, timestamp: int | None = None) -> None:
            self.exceptions.append(exception)

        def set_status(self, status: Any, description: str | None = None) -> None:
            self.status = (status, description)

    class _Histogram:
        def __init__(self) -> None:
            self.records: list[tuple[float, dict[Any, Any]]] = []

        def record(self, value: float, attributes: Any = None) -> None:
            self.records.append((value, dict(attributes or {})))

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import contextlib

        from chrys.kernel import loop as loop_module

        self.spans: list[_SpanProbe._Span] = []
        self.histogram = self._Histogram()
        probe = self

        @contextlib.contextmanager
        def fake_span(attributes: dict[Any, Any]) -> Any:
            span = probe._Span(
                f"{attributes[loop_module.OtelAttr.OPERATION]} {attributes[loop_module.OtelAttr.TOOL_NAME]}",
                attributes,
            )
            probe.spans.append(span)
            yield span

        monkeypatch.setattr(loop_module, "get_function_span", lambda *, attributes: fake_span(attributes))
        monkeypatch.setattr(loop_module, "_FUNCTION_DURATION_HISTOGRAM", self.histogram)


@pytest.fixture
def span_probe(monkeypatch: pytest.MonkeyPatch) -> _SpanProbe:
    return _SpanProbe(monkeypatch)


class TestPerToolSpan:
    @pytest.mark.asyncio
    async def test_gate_off_emits_no_span(self, span_probe: _SpanProbe, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(TELEMETRY_GATE, "enabled", False)
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "hello"})), _text_response()])
        await layer.get_response([_user()], options={"tools": [_make_tool()]})
        assert span_probe.spans == []
        assert span_probe.histogram.records == []

    @pytest.mark.asyncio
    async def test_gate_on_emits_framework_shaped_span(
        self, span_probe: _SpanProbe, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chrys.kernel.instrumentation import OtelAttr

        monkeypatch.setattr(TELEMETRY_GATE, "enabled", True)
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", False)
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "hello"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})
        assert response.messages[-1].contents[0].text == "done"

        assert len(span_probe.spans) == 1, "exactly one span per invocation — no double emission"
        span = span_probe.spans[0]
        assert span.name == "execute_tool echo"
        assert span.attributes[OtelAttr.OPERATION] == OtelAttr.TOOL_EXECUTION_OPERATION
        assert span.attributes[OtelAttr.TOOL_NAME] == "echo"
        assert span.attributes[OtelAttr.TOOL_CALL_ID] == "c1"
        assert span.attributes[OtelAttr.TOOL_TYPE] == "function"
        # Gate-off-sensitive: raw arguments/results must not reach the span.
        assert OtelAttr.TOOL_ARGUMENTS not in span.attributes
        assert OtelAttr.TOOL_RESULT not in span.set_attrs
        assert OtelAttr.MEASUREMENT_FUNCTION_INVOCATION_DURATION in span.set_attrs

        assert len(span_probe.histogram.records) == 1
        duration, attrs = span_probe.histogram.records[0]
        assert duration >= 0
        assert attrs[OtelAttr.MEASUREMENT_FUNCTION_TAG_NAME] == "echo"

    @pytest.mark.asyncio
    async def test_sensitive_on_captures_arguments_and_result(
        self, span_probe: _SpanProbe, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chrys.kernel.instrumentation import OtelAttr

        monkeypatch.setattr(TELEMETRY_GATE, "enabled", True)
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", True)
        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "hello"})), _text_response()])
        await layer.get_response([_user()], options={"tools": [_make_tool()]})

        span = span_probe.spans[0]
        assert json.loads(span.attributes[OtelAttr.TOOL_ARGUMENTS]) == {"text": "hello"}
        assert span.set_attrs[OtelAttr.TOOL_RESULT] == "echo:hello"

    @pytest.mark.asyncio
    async def test_sensitive_basemodel_argument_rewrite_is_captured(
        self, span_probe: _SpanProbe, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Middleware may rewrite ``context.arguments`` to a ``BaseModel`` (the
        same shape ``invoke`` and the logging path accept).  The sensitive span
        capture must normalize it via ``model_dump`` instead of recording
        ``"None"`` for the whole argument set."""
        from chrys.kernel.instrumentation import OtelAttr

        monkeypatch.setattr(TELEMETRY_GATE, "enabled", True)
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", True)

        echo_tool = _make_tool()
        input_model = echo_tool.input_model
        assert input_model is not None

        class _RewriteToModel(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                context.arguments = input_model(text="rewritten")
                await call_next()

        layer, _wire = _stack([_call_response(("c1", "echo", {"text": "orig"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [echo_tool]}, middleware=[_RewriteToModel()])

        assert str(_result_contents(response)[0].result) == "echo:rewritten"
        span = span_probe.spans[0]
        assert json.loads(span.attributes[OtelAttr.TOOL_ARGUMENTS]) == {"text": "rewritten"}
        assert span.set_attrs[OtelAttr.TOOL_RESULT] == "echo:rewritten"

    @pytest.mark.asyncio
    async def test_failure_records_error_type_and_exception(
        self, span_probe: _SpanProbe, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chrys.kernel.instrumentation import OtelAttr

        monkeypatch.setattr(TELEMETRY_GATE, "enabled", True)
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", False)

        @tool(name="boom")
        async def boom(text: str) -> str:
            raise RuntimeError("tool broke")

        layer, _wire = _stack([_call_response(("c1", "boom", {"text": "x"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [boom]})
        # The loop still converts the failure into an error function_result.
        results = _result_contents(response)
        assert results and "Error" in str(results[0].result)

        span = span_probe.spans[0]
        assert len(span.exceptions) == 1
        assert isinstance(span.exceptions[0], RuntimeError)
        assert span.set_attrs[OtelAttr.ERROR_TYPE] == "RuntimeError"
        (_duration, attrs) = span_probe.histogram.records[0]
        assert attrs[OtelAttr.ERROR_TYPE] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_skip_parsing_tool_sensitive_capture_uses_str(
        self, span_probe: _SpanProbe, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Skip-parsing tools return the raw value, not ``list[Content]`` —
        the sensitive capture must use the framework's ``str(result)`` form
        (``_tools.py:753-759``), not the parsed text-join."""
        from chrys.kernel.instrumentation import OtelAttr

        monkeypatch.setattr(TELEMETRY_GATE, "enabled", True)
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", True)

        @tool(name="raw", result_parser=SKIP_PARSING)
        async def raw(text: str) -> str:
            return f"raw:{text}"

        layer, _wire = _stack([_call_response(("c1", "raw", {"text": "x"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [raw]})

        # The call must succeed — telemetry-on must not turn a successful
        # invocation into an error function_result.
        results = _result_contents(response)
        assert results and results[0].exception is None
        assert str(results[0].result) == "raw:x"

        span = span_probe.spans[0]
        assert span.exceptions == []
        assert span.set_attrs[OtelAttr.TOOL_RESULT] == "raw:x"

    @pytest.mark.asyncio
    async def test_compatible_skip_parsing_sentinel_set_late_sensitive_capture_uses_str(
        self, span_probe: _SpanProbe, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from chrys.kernel.instrumentation import OtelAttr

        class _CompatibleSkipParsing:
            def __repr__(self) -> str:
                return "SKIP_PARSING"

        monkeypatch.setattr(TELEMETRY_GATE, "enabled", True)
        monkeypatch.setattr(TELEMETRY_GATE, "sensitive_data", True)

        @tool(name="raw")
        async def raw(text: str) -> str:
            return f"raw:{text}"

        raw.result_parser = _CompatibleSkipParsing()

        layer, _wire = _stack([_call_response(("c1", "raw", {"text": "x"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": [raw]})

        results = _result_contents(response)
        assert results and results[0].exception is None
        assert str(results[0].result) == "raw:x"

        span = span_probe.spans[0]
        assert span.exceptions == []
        assert span.set_attrs[OtelAttr.TOOL_RESULT] == "raw:x"


class TestExecutionBoundaryUpcast:
    """The loop executes only chrys-owned FunctionTools."""

    @pytest.mark.asyncio
    async def test_direct_caller_single_callable_tool_is_normalized(self) -> None:
        def echo(text: str) -> str:
            return f"echo:{text}"

        layer, wire = _stack([_call_response(("c1", "echo", {"text": "hello"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": echo})

        assert str(_result_contents(response)[0].result) == "echo:hello"
        assert isinstance(wire.calls[0]["options"]["tools"], list)
        assert all(type(tool_item) is FunctionTool for tool_item in wire.calls[0]["options"]["tools"])

    @pytest.mark.asyncio
    async def test_direct_caller_tuple_callable_tool_is_normalized_before_execution(self) -> None:
        def echo(text: str) -> str:
            return f"echo:{text}"

        layer, wire = _stack([_call_response(("c1", "echo", {"text": "hello"})), _text_response()])
        response = await layer.get_response([_user()], options={"tools": (echo,)})

        result_content = _result_contents(response)[0]
        assert result_content.items and all(isinstance(item, Content) for item in result_content.items)
        assert str(result_content.result) == "echo:hello"
        assert isinstance(wire.calls[1]["options"]["tools"], list)
        assert all(type(tool_item) is FunctionTool for tool_item in wire.calls[1]["options"]["tools"])


# ---------------------------------------------------------------------------
# Pre-pipeline call-provenance stamp (kind + static context on the function_call)
# ---------------------------------------------------------------------------


def _provenance_tool(*, name: str = "echo", kind: str = "probe.kind", static: dict | None = None) -> FunctionTool:
    from chrys.foundation.tool_call_context import set_tool_context
    from chrys.foundation.tool_kinds import set_tool_kind

    t = _make_tool(name=name)
    set_tool_kind(t, kind)
    if static is not None:
        set_tool_context(t, static)
    return t


class TestCallProvenanceStamp:
    """The loop stamps kind + static context on the call, and results never carry them."""

    @pytest.mark.asyncio
    async def test_success_path_stamps_call_and_filters_result(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        function_call.additional_properties["provider_marker"] = "kept"
        tool_obj = _provenance_tool(static={"server_name": "probe"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [tool_obj]})

        # Same-object invariant: the loop mutates the caller's Content in place.
        assert function_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] == "probe.kind"
        assert function_call.additional_properties[TOOL_CALL_CONTEXT_METADATA_KEY] == {"server_name": "probe"}
        result = _result_contents(response)[0]
        assert result.additional_properties["provider_marker"] == "kept"
        assert TOOL_CALL_KIND_METADATA_KEY not in result.additional_properties
        assert TOOL_CALL_CONTEXT_METADATA_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_prevalidation_failure_still_stamps_call(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY

        probe = _ProbeFunction()
        function_call = Content.from_function_call("c1", "echo", arguments={"wrong_arg": 1})
        tool_obj = _provenance_tool(static={"server_name": "probe"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [tool_obj]}, middleware=[probe])

        assert probe.contexts == [], "bad arguments must still fail before the pipeline"
        assert function_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] == "probe.kind"
        assert function_call.additional_properties[TOOL_CALL_CONTEXT_METADATA_KEY] == {"server_name": "probe"}
        result = _result_contents(response)[0]
        assert result.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_parsing"
        assert TOOL_CALL_KIND_METADATA_KEY not in result.additional_properties
        assert TOOL_CALL_CONTEXT_METADATA_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_tool_not_found_stays_unclassified(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY

        function_call = Content.from_function_call("c1", "ghost", arguments={})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [_provenance_tool()]})

        assert TOOL_CALL_KIND_METADATA_KEY not in function_call.additional_properties
        result = _result_contents(response)[0]
        assert result.additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "tool_not_found"
        assert TOOL_CALL_KIND_METADATA_KEY not in result.additional_properties
        assert TOOL_CALL_CONTEXT_METADATA_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_stamp_is_first_write_wins(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        function_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] = "persisted.kind"
        function_call.additional_properties[TOOL_CALL_CONTEXT_METADATA_KEY] = {"server_name": "persisted"}
        tool_obj = _provenance_tool(static={"server_name": "live"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        await layer.get_response([_user()], options={"tools": [tool_obj]})

        assert function_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] == "persisted.kind"
        assert function_call.additional_properties[TOOL_CALL_CONTEXT_METADATA_KEY] == {"server_name": "persisted"}

    @pytest.mark.asyncio
    async def test_tool_exception_result_carries_no_provenance(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY, set_tool_context
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY, set_tool_kind

        @tool(name="boom")
        async def boom(text: str) -> str:
            raise RuntimeError("kaboom")

        set_tool_kind(boom, "probe.kind")
        set_tool_context(boom, {"server_name": "probe"})
        function_call = Content.from_function_call("c1", "boom", arguments={"text": "a"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [boom]})

        assert function_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] == "probe.kind"
        result = _result_contents(response)[0]
        assert "Error: Function failed." in str(result.result)
        assert TOOL_CALL_KIND_METADATA_KEY not in result.additional_properties
        assert TOOL_CALL_CONTEXT_METADATA_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_middleware_termination_result_carries_no_provenance(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        tool_obj = _provenance_tool(static={"server_name": "probe"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])])])

        response = await layer.get_response(
            [_user()],
            options={"tools": [tool_obj]},
            middleware=[_TerminateFunction(result="interrupted")],
        )

        assert function_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] == "probe.kind"
        result = _result_contents(response)[0]
        assert str(result.result) == "interrupted"
        assert TOOL_CALL_KIND_METADATA_KEY not in result.additional_properties
        assert TOOL_CALL_CONTEXT_METADATA_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_result_props_do_not_share_the_call_context_dict(self) -> None:
        """The shallow-copy hazard: the result must never alias the call's nested context."""
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        tool_obj = _provenance_tool(static={"server_name": "probe"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [tool_obj]})

        stamped = function_call.additional_properties[TOOL_CALL_CONTEXT_METADATA_KEY]
        result = _result_contents(response)[0]
        assert all(value is not stamped for value in result.additional_properties.values())


# ---------------------------------------------------------------------------
# Invocation-order stamping + result-metadata carriage (loop authority)
# ---------------------------------------------------------------------------


def _call_contents(response: ChatResponse) -> list[Content]:
    return [item for msg in response.messages for item in msg.contents if item.type == "function_call"]


def _ordinals(response: ChatResponse) -> list[int | None]:
    from chrys.foundation.tool_invocation_order import read_tool_invocation_order

    return [read_tool_invocation_order(c.additional_properties) for c in _call_contents(response)]


class _CarriageFunction(FunctionMiddleware):
    """Writes result metadata + a builder context subkey, like the event middleware."""

    def __init__(self, *, result_metadata: dict[str, Any] | None = None, tool_context: dict[str, Any] | None = None):
        self._result_metadata = result_metadata
        self._tool_context = tool_context

    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        await call_next()
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY

        if self._result_metadata is not None:
            context.metadata[TOOL_RESULT_METADATA_KEY] = dict(self._result_metadata)
        if self._tool_context is not None:
            context.metadata[TOOL_CALL_CONTEXT_METADATA_KEY] = dict(self._tool_context)


class TestInvocationOrderStamping:
    """The loop is the single ordinal producer: stamped at response arrival, pre-dispatch."""

    @pytest.mark.asyncio
    async def test_ordinals_accumulate_across_iterations(self) -> None:
        layer, _wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "a"}), ("c2", "echo", {"text": "b"})),
                _call_response(("c3", "echo", {"text": "c"})),
                _text_response("done"),
            ]
        )

        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})

        assert _ordinals(response) == [0, 1, 2]
        for result in _result_contents(response):
            assert TOOL_INVOCATION_ORDER_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_prevalidation_failure_and_unknown_tool_consume_ordinals(self) -> None:
        """The issue-589 shape: calls that never enter the pipeline still hold their slot."""
        layer, _wire = _stack(
            [
                _call_response(
                    ("c1", "echo", {"wrong_arg": "x"}),
                    ("c2", "missing_tool", {}),
                    ("c3", "echo", {"text": "ok"}),
                ),
                _text_response("done"),
            ]
        )

        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})

        assert _ordinals(response) == [0, 1, 2]
        results = {r.call_id: r for r in _result_contents(response)}
        assert results["c1"].additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_parsing"
        assert results["c2"].additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "tool_not_found"
        assert str(results["c3"].result) == "echo:ok"
        for result in results.values():
            assert TOOL_INVOCATION_ORDER_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_client_mutating_its_retained_message_cannot_rewrite_landed_history(self) -> None:
        """A stateful client appends a fresh call to a Message it already returned.

        Landing retains loop-owned snapshots, so the in-place append changes
        only the client's object: the first exchange keeps exactly the calls
        it landed with, and the late call lands once, in the later response.
        """
        first_message = Message("assistant", [Content.from_function_call("c1", "echo", arguments={"text": "a"})])
        served = {"count": 0}

        class _MutatingStatefulClient:
            def get_response(self, messages: Any, *, stream: bool = False, **kwargs: Any) -> Any:
                served["count"] += 1
                if served["count"] == 1:
                    turn = ChatResponse(messages=[first_message])
                elif served["count"] == 2:
                    first_message.contents.append(Content.from_function_call("c2", "echo", arguments={"text": "b"}))
                    turn = ChatResponse(messages=[first_message])
                else:
                    turn = _text_response()

                async def _resolve() -> ChatResponse:
                    return turn

                return _resolve()

        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(_MutatingStatefulClient()))
        events: list[str] = []
        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:a", "tool:echo:b"], "each call executes exactly once"
        assert _ordinals(response) == [0, 1]
        # The first exchange's landed snapshot still holds ONLY c1; c2 lives
        # in the client's object and in the second exchange it landed with.
        assert [c.call_id for c in response.messages[0].contents] == ["c1"]
        assert all(m is not first_message for m in response.messages)

    @pytest.mark.asyncio
    async def test_client_mutating_next_turn_input_cannot_rewrite_landed_history(self) -> None:
        """The reverse aliasing direction: the client mutates a message it RECEIVED.

        The wire call hands the client per-call snapshot views, never the
        loop's retained transcript wrappers — so appending a call to an input
        history message (and re-sending that object as the response) changes
        only the throwaway view: the first exchange keeps exactly the calls
        it landed with, and the appended call lands once, in its own response.
        """
        served = {"count": 0}

        class _InputMutatingClient:
            def get_response(self, messages: Any, *, stream: bool = False, **kwargs: Any) -> Any:
                served["count"] += 1
                if served["count"] == 1:
                    turn = ChatResponse(
                        messages=[
                            Message("assistant", [Content.from_function_call("c1", "echo", arguments={"text": "a"})])
                        ]
                    )
                elif served["count"] == 2:
                    victim = next(
                        m
                        for m in messages
                        if m.role == "assistant" and any(c.type == "function_call" for c in m.contents)
                    )
                    victim.contents.append(Content.from_function_call("c2", "echo", arguments={"text": "b"}))
                    turn = ChatResponse(messages=[victim])
                else:
                    turn = _text_response()

                async def _resolve() -> ChatResponse:
                    return turn

                return _resolve()

        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(_InputMutatingClient()))
        events: list[str] = []
        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:a", "tool:echo:b"], "each call executes exactly once"
        assert _ordinals(response) == [0, 1]
        assert [c.call_id for c in response.messages[0].contents] == ["c1"]

    @pytest.mark.asyncio
    async def test_client_echoing_historical_call_object_does_not_reexecute(self) -> None:
        """The echo memo is seeded from caller history.

        A client returning the SAME Content object as a historical function
        call (a bridged or echoing client replaying the conversation) must
        not re-execute the past side-effecting call: the memo covers every
        content the conversation holds, not just what this run landed.
        """
        events: list[str] = []
        hist_call = Content.from_function_call("h1", "echo", arguments={"text": "once"})
        history = [
            _user("do it"),
            Message("assistant", [hist_call]),
            Message("tool", [Content.from_function_result("h1", result="echo:once")]),
            _user("again?"),
        ]
        served = {"count": 0}

        class _HistoryEchoingClient:
            def get_response(self, messages: Any, *, stream: bool = False, **kwargs: Any) -> Any:
                served["count"] += 1
                turn = (
                    ChatResponse(messages=[Message("assistant", [hist_call])])
                    if served["count"] == 1
                    else _text_response()
                )

                async def _resolve() -> ChatResponse:
                    return turn

                return _resolve()

        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(_HistoryEchoingClient()))
        response = await layer.get_response(history, options={"tools": [_make_tool(events)]})

        assert events == [], "historical call must not re-execute"
        assert TOOL_INVOCATION_ORDER_KEY not in hist_call.additional_properties
        assert _call_contents(response) == []

    def test_wire_request_identity_registration_registers_contents_and_skips_usage(self) -> None:
        """Wire-only contents enter the echo registry; usage stays out via the
        registry's construction-time policy. No retention side: weakref
        liveness makes a collected wire-only content's entry evict, so its
        recycled id can never strip a genuinely fresh content."""
        injected = Content.from_text("wire-only")
        usage = Content.from_usage(
            usage_details={"input_token_count": 1, "output_token_count": 0, "total_token_count": 1}
        )
        echo_registry = WeakIdentityRegistry(ignore_usage=True)

        _record_wire_request_content_identities(
            [Message("user", [injected, usage])],
            echo_registry,
        )

        assert injected in echo_registry
        assert usage not in echo_registry
        assert len(echo_registry) == 1

        del injected
        gc.collect()
        assert len(echo_registry) == 0, "no retention: a dead wire-only content's entry evicts"

    @pytest.mark.asyncio
    async def test_blocking_usage_identity_can_repeat_across_history_and_model_calls(self) -> None:
        """Request-scoped usage never enters the blocking echo memo."""
        usage = Content.from_usage(
            usage_details={"input_token_count": 3, "output_token_count": 1, "total_token_count": 4}
        )
        history = [
            _user("old"),
            Message("assistant", [usage]),
            _user("continue"),
        ]
        first = ChatResponse(
            messages=[
                Message(
                    "assistant",
                    [
                        Content.from_function_call("c1", "echo", arguments={"text": "a"}),
                        usage,
                    ],
                )
            ]
        )
        final = ChatResponse(messages=[Message("assistant", [Content.from_text("done"), usage])])
        layer, wire = _stack([first, final])

        response = await layer.get_response(history, options={"tools": [_make_tool()]})
        landed_usage = [
            content for message in response.messages for content in message.contents if content.type == "usage"
        ]

        assert len(wire.calls) == 2
        assert landed_usage == [usage, usage]
        assert all(content is usage for content in landed_usage)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
    async def test_post_middleware_request_content_echo_is_removed(self, stream: bool) -> None:
        """A wire-only injected object cannot land back as assistant text."""
        injected_message = Message("user", ["INJECTED"])
        injected_content = injected_message.contents[0]

        class _InjectWireOnlyMessage(ChatMiddleware):
            async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
                context.messages = [*context.messages, injected_message]
                await call_next()

        class _EchoInjectedContentClient:
            def get_response(self, messages: Any, *, stream: bool = False, **kwargs: Any) -> Any:
                assert any(item is injected_content for message in messages for item in message.contents)
                fresh = Content.from_text("done")
                if not stream:

                    async def _resolve() -> ChatResponse:
                        return ChatResponse(messages=[Message("assistant", [injected_content, fresh])])

                    return _resolve()

                async def _gen() -> Any:
                    yield ChatResponseUpdate(contents=[injected_content], role="assistant")
                    yield ChatResponseUpdate(contents=[fresh], role="assistant")

                return ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        layer = InvariantCheckedToolLoopLayer(
            ChatMiddlewareLayer(_EchoInjectedContentClient(), middleware=[_InjectWireOnlyMessage()])
        )

        if stream:
            response_stream = layer.get_response([_user()], stream=True)
            yielded = [update async for update in response_stream]
            response = await response_stream.get_final_response()
            assert [content.text for update in yielded for content in update.contents] == ["done"]
        else:
            response = await layer.get_response([_user()])

        assert response.text == "done"
        assert all(item is not injected_content for message in response.messages for item in message.contents)

    @pytest.mark.asyncio
    async def test_client_mutating_received_history_wrapper_cannot_corrupt_session_state(self) -> None:
        """Caller-history wrappers reach the client as per-call views too.

        A client appending a call to a received historical assistant message
        (and re-sending that object as its response) mutates only the
        throwaway view: the caller's live history object is untouched, the
        echoed historical call does not re-execute, and only the genuinely
        fresh call lands and runs.
        """
        events: list[str] = []
        hist_asst = Message("assistant", [Content.from_function_call("h1", "echo", arguments={"text": "old"})])
        history = [
            _user("do it"),
            hist_asst,
            Message("tool", [Content.from_function_result("h1", result="echo:old")]),
            _user("again?"),
        ]
        served = {"count": 0}

        class _HistoryMutatingClient:
            def get_response(self, messages: Any, *, stream: bool = False, **kwargs: Any) -> Any:
                served["count"] += 1
                if served["count"] == 1:
                    victim = next(
                        m
                        for m in messages
                        if m.role == "assistant" and any(c.type == "function_call" for c in m.contents)
                    )
                    victim.contents.append(Content.from_function_call("c9", "echo", arguments={"text": "new"}))
                    turn = ChatResponse(messages=[victim])
                else:
                    turn = _text_response()

                async def _resolve() -> ChatResponse:
                    return turn

                return _resolve()

        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(_HistoryMutatingClient()))
        response = await layer.get_response(history, options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:new"], "only the fresh call executes"
        assert [c.call_id for c in hist_asst.contents] == ["h1"], "live session-state history stays intact"
        assert [c.call_id for c in _call_contents(response)] == ["c9"]

    @pytest.mark.asyncio
    async def test_client_mutating_received_injection_cannot_corrupt_retained_copy(self) -> None:
        """Consumed injection wrappers the loop retains go out as views too.

        The probe hands the loop a middleware-owned user message between
        iterations, and the loop re-sends it on every later call. A client
        appending to the received wrapper corrupts only its per-call view:
        the retained wrapper — what later calls re-send and what persistence
        captures — keeps exactly its original content.
        """
        injected = Message("user", ["injected note"])
        injected.additional_properties["_injection_id"] = "inj-1"
        pending = [injected]
        probe = ConsumedInjectionMessageProbe(lambda: [pending.pop()] if pending else [])
        received: list[Message] = []
        final_wire: list[Message] = []

        class _InjectionMutatingClient:
            def __init__(self) -> None:
                self.count = 0

            def get_response(self, messages: Any, *, stream: bool = False, **kwargs: Any) -> Any:
                self.count += 1
                if self.count == 1:
                    turn = _call_response(("c1", "echo", {"text": "a"}))
                elif self.count == 2:
                    victim = next(m for m in messages if m.role == "user" and m.contents[0].text == "injected note")
                    received.append(victim)
                    victim.contents.append(Content.from_text("CORRUPTED"))
                    turn = _call_response(("c2", "echo", {"text": "b"}))
                else:
                    final_wire.extend(messages)
                    turn = _text_response()

                async def _resolve() -> ChatResponse:
                    return turn

                return _resolve()

        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(_InjectionMutatingClient()))
        await layer.get_response(
            [_user()],
            options={"tools": [_make_tool()]},
            client_kwargs={"consumed_injection_message_probe": probe},
        )

        assert [c.text for c in injected.contents] == ["injected note"]
        view = received[0]
        assert view is not injected
        assert view.contents[0] is injected.contents[0]
        assert view.additional_properties is injected.additional_properties
        # The third call's wire input carries the clean injection, not the mutation.
        third_call_copies = [m for m in final_wire if m.role == "user" and m.contents[0].text == "injected note"]
        assert len(third_call_copies) == 1
        assert [c.text for c in third_call_copies[0].contents] == ["injected note"]

    @pytest.mark.asyncio
    async def test_streaming_client_mutating_received_injection_cannot_corrupt_retained_copy(self) -> None:
        injected = Message("user", ["injected note"])
        pending = [injected]
        probe = ConsumedInjectionMessageProbe(lambda: [pending.pop()] if pending else [])
        received: list[Message] = []

        class _StreamInjectionMutatingClient:
            def __init__(self) -> None:
                self.count = 0

            def get_response(self, messages: Any, *, stream: bool = False, **kwargs: Any) -> Any:
                self.count += 1
                if self.count == 2:
                    victim = next(m for m in messages if m.role == "user" and m.contents[0].text == "injected note")
                    received.append(victim)
                    victim.contents.append(Content.from_text("CORRUPTED"))
                    updates = [_text_update("final")]
                else:
                    updates = [_call_update("c1", "echo", {"text": "a"})]

                async def _gen() -> Any:
                    for update in updates:
                        yield update

                stream_response: ResponseStream[ChatResponseUpdate, ChatResponse] = ResponseStream(
                    _gen(), finalizer=ChatResponse.from_updates
                )
                return stream_response

        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(_StreamInjectionMutatingClient()))
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()]},
            client_kwargs={"consumed_injection_message_probe": probe},
        )
        _ = [u async for u in stream]
        await stream.get_final_response()

        assert [c.text for c in injected.contents] == ["injected note"]
        assert received[0] is not injected
        assert received[0].contents[0] is injected.contents[0]

    @pytest.mark.asyncio
    async def test_duplicate_and_already_answered_calls_are_stamped_inert(self) -> None:
        """Calls the dispatch filter drops still consume ordinals but never execute."""
        events: list[str] = []
        answered_call = Content.from_function_call("done-1", "echo", arguments={"text": "past"})
        answered_result = Content.from_function_result("done-1", result="already")
        live_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        duplicate_call = Content.from_function_call("c1", "echo", arguments={"text": "b"})
        first_turn = ChatResponse(
            messages=[
                Message("assistant", [answered_call, live_call, duplicate_call]),
                Message("tool", [answered_result]),
            ]
        )
        layer, _wire = _stack([first_turn, _text_response("done")])
        sink = FakeSink()

        response = await layer.get_response(
            [_user()],
            options={"tools": [_make_tool(events)]},
            client_kwargs={TRAJECTORY_CONTEXT_KWARG: make_context(sink)},
        )

        assert events == ["tool:echo:a"], "answered + duplicate calls must not execute"
        assert answered_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert live_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 1
        assert duplicate_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 2
        assert OPERATION_ID_KEY not in answered_call.additional_properties
        assert (
            duplicate_call.additional_properties[OPERATION_ID_KEY] == live_call.additional_properties[OPERATION_ID_KEY]
        )
        # Neither dropped call opens an operation of its own: the answered one
        # is provider-owned history and the duplicate rides the dispatched
        # call's operation, so there is nothing here for a terminal to close.
        # (The executed call's own pair comes from the tool-events middleware,
        # which this kernel-only stack does not install.)
        assert sink.of_type(TrajectoryEventType.TOOL_OPERATION_STARTED) == []
        assert sink.of_type(TrajectoryEventType.TOOL_OPERATION_FINISHED) == []
        new_results = [r for r in _result_contents(response) if str(r.result) != "already"]
        assert [r.call_id for r in new_results] == ["c1"]

    @pytest.mark.asyncio
    async def test_stale_foreign_stamp_is_overwritten(self) -> None:
        """A pre-stamped content on a landed response is re-numbered — sole authority.

        Bridged or cached clients can replay content that carries another
        run's ordinal; preserving it while the counter advances would let the
        stale value collide with a fresh assignment.
        """
        stale_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        stale_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 7
        fresh_call = Content.from_function_call("c2", "echo", arguments={"text": "b"})
        first_turn = ChatResponse(messages=[Message("assistant", [stale_call, fresh_call])])
        layer, _wire = _stack([first_turn, _text_response("done")])

        response = await layer.get_response([_user()], options={"tools": [_make_tool()]})

        assert stale_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert fresh_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 1
        assert _ordinals(response) == [0, 1]

    @pytest.mark.asyncio
    async def test_stale_foreign_stamp_is_overwritten_streaming(self) -> None:
        stale_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        stale_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = 7
        stale_update = ChatResponseUpdate(contents=[stale_call], role="assistant")
        layer, _wire = _stack([[stale_update], [_text_update("done")]])

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        calls = _call_contents(final)
        assert len(calls) == 1
        assert calls[0].additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0

    @pytest.mark.asyncio
    async def test_revisited_content_keeps_ordinal_and_is_not_dispatched_again(self) -> None:
        """A content this run already numbered is an echo: same ordinal, no re-run.

        An echoing client can re-embed a prior iteration's call object in a
        later response. Re-numbering it would corrupt decisions recorded
        against the first value, re-dispatching it would execute the tool
        twice under one ordinal — collapsing two results onto one approval
        slot — and leaving it in the response would persist a dangling
        duplicate call, so landing removes the echo outright.
        """
        events: list[str] = []
        first_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        second_call = Content.from_function_call("c2", "echo", arguments={"text": "b"})
        third_call = Content.from_function_call("c3", "echo", arguments={"text": "c"})
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [first_call, second_call])]),
                ChatResponse(messages=[Message("assistant", [first_call, third_call])]),
                _text_response("done"),
            ]
        )

        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert first_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert second_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 1
        assert third_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 2
        assert events == ["tool:echo:a", "tool:echo:b", "tool:echo:c"], "the echoed call must execute exactly once"
        assert [r.call_id for r in _result_contents(response)].count("c1") == 1
        assert [c.call_id for c in _call_contents(response)] == ["c1", "c2", "c3"], "the echo must leave no duplicate"

    @pytest.mark.asyncio
    async def test_response_of_only_revisited_calls_ends_the_loop(self) -> None:
        """An echo-only response carries no new work and terminates cleanly.

        The echo is removed from the landed response, so the final transcript
        must end call/result-paired — no dangling duplicate function call
        after its result (providers reject such histories on the next turn).
        """
        events: list[str] = []
        first_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [first_call])]),
                ChatResponse(messages=[Message("assistant", [first_call])]),
            ]
        )

        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:a"]
        assert first_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert [r.call_id for r in _result_contents(response)] == ["c1"]
        assert len(_call_contents(response)) == 1, "the echo must not survive as a dangling duplicate call"
        assert response.messages[-1].role == "tool"

    @pytest.mark.asyncio
    async def test_fresh_call_reusing_an_echoed_call_id_still_executes(self) -> None:
        """Echo removal is by object identity; a fresh same-id call is new work.

        Providers mint per-response counter ids, so a later response can
        legitimately pair an echoed old object with a NEW content reusing the
        same call id. The fresh call must land, stamp, and execute.
        """
        events: list[str] = []
        old_call = Content.from_function_call("call_00", "echo", arguments={"text": "old"})
        fresh_call = Content.from_function_call("call_00", "echo", arguments={"text": "new"})
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [old_call])]),
                ChatResponse(messages=[Message("assistant", [old_call, fresh_call])]),
                _text_response("done"),
            ]
        )

        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:old", "tool:echo:new"]
        assert old_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert fresh_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 1
        assert [r.call_id for r in _result_contents(response)] == ["call_00", "call_00"]

    @pytest.mark.asyncio
    async def test_same_object_twice_in_one_response_executes_once(self) -> None:
        """An identity duplicate within one response is normalized to one copy."""
        events: list[str] = []
        call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack(
            [
                ChatResponse(messages=[Message("assistant", [call, call])]),
                _text_response("done"),
            ]
        )

        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:a"], "the duplicated object must execute exactly once"
        assert call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert [c.call_id for c in _call_contents(response)] == ["c1"]
        assert [r.call_id for r in _result_contents(response)] == ["c1"]

    @pytest.mark.asyncio
    async def test_echoed_message_object_preserves_prior_transcript(self) -> None:
        """Echo removal must not mutate a Message the transcript already holds.

        A client can echo the whole prior Message object, not just its
        contents; clearing that message in place would also clear the
        accumulated transcript's copy, leaving an orphaned result whose call
        vanished from history.
        """
        events: list[str] = []
        call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        shared_message = Message("assistant", [call])
        layer, _wire = _stack(
            [
                ChatResponse(messages=[shared_message]),
                ChatResponse(messages=[shared_message]),
            ]
        )

        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:a"]
        assert shared_message.contents == [call], "the shared message must keep its call"
        assert [c.call_id for c in _call_contents(response)] == ["c1"]
        assert [r.call_id for r in _result_contents(response)] == ["c1"]
        assert response.messages[-1].role == "tool"

    @pytest.mark.asyncio
    async def test_full_exchange_echo_is_removed(self) -> None:
        """Echoing a completed call/result exchange lands as pure history.

        The echoed result must vanish with its call: removing only the call
        would leave one call answered by two results — an invalid transcript.
        """
        events: list[str] = []
        call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        first_response = ChatResponse(messages=[Message("assistant", [call])])

        class _EchoingClient(_ScriptedClient):
            """Second turn echoes the first exchange — including the loop-made result — in new Message wrappers."""

            def get_response(self, messages: Any, **kwargs: Any) -> Any:
                if not self.turns:
                    self.turns.append(
                        ChatResponse(messages=[Message(m.role, list(m.contents)) for m in first_response.messages])
                    )
                return super().get_response(messages, **kwargs)

        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(_EchoingClient([first_response])))

        response = await layer.get_response([_user()], options={"tools": [_make_tool(events)]})

        assert events == ["tool:echo:a"]
        assert [c.call_id for c in _call_contents(response)] == ["c1"]
        assert [r.call_id for r in _result_contents(response)] == ["c1"], "the echoed result must not duplicate"
        assert [m.role for m in response.messages] == ["assistant", "tool"]

    @pytest.mark.asyncio
    async def test_streaming_echoed_result_content_is_removed(self) -> None:
        """A streamed echo of an already-landed result does not duplicate it."""
        events: list[str] = []
        answered_call = Content.from_function_call("done-1", "echo", arguments={"text": "past"})
        answered_result = Content.from_function_result("done-1", result="already")
        live_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack(
            [
                [
                    ChatResponseUpdate(contents=[answered_call], role="assistant"),
                    ChatResponseUpdate(contents=[answered_result], role="tool"),
                    ChatResponseUpdate(contents=[live_call], role="assistant"),
                ],
                [
                    ChatResponseUpdate(contents=[answered_call], role="assistant"),
                    ChatResponseUpdate(contents=[answered_result], role="tool"),
                ],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:a"]
        assert [c.call_id for c in _call_contents(final)].count("done-1") == 1
        assert [r.call_id for r in _result_contents(final)].count("done-1") == 1

    @pytest.mark.asyncio
    async def test_revisited_content_is_removed_from_streaming_final_response(self) -> None:
        """A streamed echo is removed from the assembled final response."""
        events: list[str] = []
        first_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack(
            [
                [ChatResponseUpdate(contents=[first_call], role="assistant")],
                [ChatResponseUpdate(contents=[first_call], role="assistant")],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:a"]
        assert len(_call_contents(final)) == 1, "the echo must not survive as a dangling duplicate call"
        assert [r.call_id for r in _result_contents(final)] == ["c1"]

    @pytest.mark.asyncio
    async def test_no_tools_response_calls_are_stamped(self) -> None:
        """A call batch returned without a tools option is stamped before the early exit."""
        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])])])

        response = await layer.get_response([_user()], options={})

        assert function_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert _result_contents(response) == []

    @pytest.mark.asyncio
    async def test_a_call_with_no_tools_to_run_it_closes_as_filtered(self) -> None:
        """The stamp is handed out at landing; the operation it opens has to end somewhere."""
        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])])])
        sink = FakeSink()

        await layer.get_response([_user()], options={}, client_kwargs={TRAJECTORY_CONTEXT_KWARG: make_context(sink)})

        started = sink.only(TrajectoryEventType.TOOL_OPERATION_STARTED)
        finished = sink.only(TrajectoryEventType.TOOL_OPERATION_FINISHED)
        assert finished.operation_id == started.operation_id
        assert finished.payload["outcome"] == ToolOutcome.FILTERED
        assert "result_item_id" not in finished.payload  # nothing ran, so there is no result to name

    @pytest.mark.asyncio
    async def test_a_streamed_call_with_no_tools_to_run_it_closes_as_filtered(self) -> None:
        layer, _wire = _stack([[_call_update("c1", "echo", {"text": "a"})]])
        sink = FakeSink()

        stream = layer.get_response(
            [_user()],
            stream=True,
            options={},
            client_kwargs={TRAJECTORY_CONTEXT_KWARG: make_context(sink)},
        )
        _ = [u async for u in stream]
        await stream.get_final_response()

        assert sink.only(TrajectoryEventType.TOOL_OPERATION_FINISHED).payload["outcome"] == ToolOutcome.FILTERED

    @pytest.mark.asyncio
    async def test_informational_calls_are_not_stamped(self) -> None:
        hosted = Content.from_function_call("h1", "web_search", arguments={}, informational_only=True)
        local = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [hosted, local])]), _text_response("done")])

        await layer.get_response([_user()], options={"tools": [_make_tool()]})

        assert TOOL_INVOCATION_ORDER_KEY not in hosted.additional_properties
        assert local.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0

    @pytest.mark.asyncio
    async def test_result_metadata_and_context_carriage_fold_at_result_construction(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY
        from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        tool_obj = _provenance_tool(static={"server_name": "static"})
        carrier = _CarriageFunction(result_metadata={"shell_exit_code": 1}, tool_context={"built": "filled"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        response = await layer.get_response([_user()], options={"tools": [tool_obj]}, middleware=[carrier])

        # Call side: kind + static context stamped pre-validation, builder
        # subkeys backfilled post-execution without overwriting static keys.
        assert function_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] == "probe.kind"
        assert function_call.additional_properties[TOOL_CALL_CONTEXT_METADATA_KEY] == {
            "server_name": "static",
            "built": "filled",
        }
        assert TOOL_RESULT_METADATA_KEY not in function_call.additional_properties
        # Result side: carried metadata folded at construction, nothing else.
        result = _result_contents(response)[0]
        assert result.additional_properties[TOOL_RESULT_METADATA_KEY] == {"shell_exit_code": 1}
        assert TOOL_CALL_CONTEXT_METADATA_KEY not in result.additional_properties
        assert TOOL_INVOCATION_ORDER_KEY not in result.additional_properties

    @pytest.mark.asyncio
    async def test_middleware_termination_prebuilt_result_metadata_first_write_wins(self) -> None:
        """A middleware-built result Content owns its own metadata; carriage never overwrites."""
        from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY

        prebuilt = Content.from_function_result("c1", result="stopped")
        prebuilt.additional_properties[TOOL_RESULT_METADATA_KEY] = {"owned": True}

        class _CarryThenTerminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                context.metadata[TOOL_RESULT_METADATA_KEY] = {"carried": True}
                raise MiddlewareTermination("stop", result=prebuilt)

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])])])

        response = await layer.get_response(
            [_user()], options={"tools": [_make_tool()]}, middleware=[_CarryThenTerminate()]
        )

        result = _result_contents(response)[0]
        assert result is not prebuilt
        assert result.call_id == "c1"
        assert result.additional_properties[TOOL_RESULT_METADATA_KEY] == {"owned": True}
        assert prebuilt.additional_properties[TOOL_RESULT_METADATA_KEY] == {"owned": True}

    @pytest.mark.asyncio
    async def test_middleware_termination_prebuilt_result_carries_this_calls_identity(self) -> None:
        """The trajectory names the result item before it exists; the saved one has to be it."""
        prebuilt = Content.from_function_result("stale", result="stopped")
        prebuilt.additional_properties[ANALYTICS_ITEM_ID_KEY] = "an-id-from-another-call"
        promised: dict[str, str] = {}

        class _Terminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                promised["result_item_id"] = str(context.metadata[TOOL_RESULT_ITEM_ID_METADATA_KEY])
                promised["operation_id"] = str(context.metadata[OPERATION_ID_KEY])
                raise MiddlewareTermination("stop", result=prebuilt)

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])])])

        response = await layer.get_response([_user()], options={"tools": [_make_tool()]}, middleware=[_Terminate()])

        result = _result_contents(response)[0]
        props = result.additional_properties
        assert props[OPERATION_ID_KEY] == promised["operation_id"]
        assert props[ANALYTICS_ITEM_ID_KEY] == promised["result_item_id"]
        # The cached result keeps the identity it came with; this call's went
        # onto the copy.
        assert prebuilt.additional_properties[ANALYTICS_ITEM_ID_KEY] == "an-id-from-another-call"

    @pytest.mark.asyncio
    async def test_middleware_termination_prebuilt_result_without_metadata_gets_carriage(self) -> None:
        from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY

        prebuilt = Content.from_function_result("c1", result="stopped")

        class _CarryThenTerminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                context.metadata[TOOL_RESULT_METADATA_KEY] = {"carried": True}
                raise MiddlewareTermination("stop", result=prebuilt)

        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])])])

        response = await layer.get_response(
            [_user()], options={"tools": [_make_tool()]}, middleware=[_CarryThenTerminate()]
        )

        result = _result_contents(response)[0]
        assert result is not prebuilt
        assert result.call_id == "c1"
        assert result.additional_properties[TOOL_RESULT_METADATA_KEY] == {"carried": True}
        assert TOOL_RESULT_METADATA_KEY not in prebuilt.additional_properties

    @pytest.mark.asyncio
    async def test_middleware_termination_shared_prebuilt_result_is_owned_per_parallel_call(self) -> None:
        """Concurrent terminations cannot alias one cached result or retain its stale call id."""
        from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY

        prebuilt = Content.from_function_result("stale", result="stopped")

        class _CarryThenTerminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                text = str(context.arguments["text"])
                context.metadata[TOOL_RESULT_METADATA_KEY] = {"text": text}
                write_execution_stamp(context.metadata, context.arguments, outcome="ok")
                raise MiddlewareTermination("stop", result=prebuilt)

        layer, _wire = _stack(
            [
                _call_response(
                    ("c1", "echo", {"text": "a"}),
                    ("c2", "echo", {"text": "b"}),
                )
            ]
        )

        response = await layer.get_response(
            [_user()], options={"tools": [_make_tool()]}, middleware=[_CarryThenTerminate()]
        )

        first, second = _result_contents(response)
        assert first is not second
        assert first is not prebuilt
        assert second is not prebuilt
        assert [first.call_id, second.call_id] == ["c1", "c2"]
        assert first.additional_properties[TOOL_RESULT_METADATA_KEY] == {"text": "a"}
        assert second.additional_properties[TOOL_RESULT_METADATA_KEY] == {"text": "b"}
        assert first.additional_properties[EXECUTION_STAMP_KEY]["effective_args"] == '{"text":"a"}'
        assert second.additional_properties[EXECUTION_STAMP_KEY]["effective_args"] == '{"text":"b"}'
        assert prebuilt.call_id == "stale"
        assert prebuilt.additional_properties == {}

    @pytest.mark.asyncio
    async def test_middleware_termination_shared_prebuilt_result_is_owned_per_run(self) -> None:
        """A cached result reused by later runs receives fresh call-scoped metadata."""
        from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY

        prebuilt = Content.from_function_result("stale", result="stopped")

        class _CarryThenTerminate(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                text = str(context.arguments["text"])
                context.metadata[TOOL_RESULT_METADATA_KEY] = {"text": text}
                write_execution_stamp(context.metadata, context.arguments, outcome="ok")
                raise MiddlewareTermination("stop", result=prebuilt)

        middleware = _CarryThenTerminate()
        layer, _wire = _stack(
            [
                _call_response(("c1", "echo", {"text": "first"})),
                _call_response(("c2", "echo", {"text": "second"})),
            ]
        )

        first_response = await layer.get_response([_user()], options={"tools": [_make_tool()]}, middleware=[middleware])
        second_response = await layer.get_response(
            [_user()], options={"tools": [_make_tool()]}, middleware=[middleware]
        )

        first = _result_contents(first_response)[0]
        second = _result_contents(second_response)[0]
        assert first is not second
        assert [first.call_id, second.call_id] == ["c1", "c2"]
        assert first.additional_properties[TOOL_RESULT_METADATA_KEY] == {"text": "first"}
        assert second.additional_properties[TOOL_RESULT_METADATA_KEY] == {"text": "second"}
        assert first.additional_properties[EXECUTION_STAMP_KEY]["effective_args"] == '{"text":"first"}'
        assert second.additional_properties[EXECUTION_STAMP_KEY]["effective_args"] == '{"text":"second"}'
        assert prebuilt.call_id == "stale"
        assert prebuilt.additional_properties == {}

    @pytest.mark.asyncio
    async def test_cancellation_after_call_next_propagates_without_result(self) -> None:
        """A cancelled invocation constructs no result and fabricates no metadata."""

        class _CancelAfter(FunctionMiddleware):
            async def process(
                self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]
            ) -> None:
                await call_next()
                raise asyncio.CancelledError

        events: list[str] = []
        function_call = Content.from_function_call("c1", "echo", arguments={"text": "a"})
        layer, wire = _stack([ChatResponse(messages=[Message("assistant", [function_call])]), _text_response()])

        with pytest.raises(asyncio.CancelledError):
            await layer.get_response([_user()], options={"tools": [_make_tool(events)]}, middleware=[_CancelAfter()])

        assert events == ["tool:echo:a"], "cancellation hit after the tool body ran"
        assert len(wire.calls) == 1, "the loop must not continue past a cancelled batch"
        # The stamp landed at response arrival; nothing result-shaped was built.
        assert function_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0


# ---------------------------------------------------------------------------
# Streaming reconstruction (single reconstruction authority)
# ---------------------------------------------------------------------------


class _StructuredAnswer(BaseModel):
    answer: int


class _AdversarialEchoClient:
    """Wire double that mixes fresh work with every known echo shape.

    Each tool turn emits 1-2 FRESH calls (unique ids within the turn; ids may
    deliberately REUSE a previous turn's id — providers mint per-response
    counter ids, so reuse is legitimate new work; arguments are randomly
    dict- or string-typed — the string form is the wire representation,
    where a duplicate object self-merging in assembly corrupts by
    concatenation instead of merging silently) and then, driven by the
    seeded rng, injects adversarial shapes the landing normalization must
    neutralize:

    - an echo of a prior function_call content (same object, new wrapper);
    - an echo of a prior function_result content (same object);
    - a whole prior Message object re-sent verbatim (assistant or tool);
    - one of this turn's fresh call objects duplicated within the response;
    - a previously returned assistant Message mutated IN PLACE (a fresh call
      appended) and re-sent — the stateful-client shape (non-streamed turns
      only: streaming assembly mints fresh containers, so the client's
      wrappers never enter the transcript there);
    - an assistant Message from this call's INPUT mutated in place and
      re-sent — the reverse aliasing direction (the wire hands the client
      per-call snapshot views, so the mutation must change only the view;
      non-streamed turns only, same reason as above);
    - the same landed call object twice ADJACENTLY in a streamed turn —
      assembly would merge the pair into a NEW object, so the echo
      fragments must be stripped from assembly input before a laundered
      copy can be minted;
    - that merged double echo followed, behind a text separator, by a
      FRESH call reusing the same call id — the poison combo: the
      laundered copy must not re-execute the old arguments, and the fresh
      call must not be swallowed by duplicate-id dispatch filtering;
    - a fresh call split into raw string-argument fragments across
      adjacent updates (streamed turns) — assembly merges them into ONE
      call that executes once;
    - a raw fragment from a COMPLETED turn replayed verbatim on a later
      streamed turn — the merge minted a NEW assembled object, so only
      post-landing fragment-identity recording can recognize the replay;
      it must not re-execute and must not corrupt adjacent fresh work.

    Echo candidates are harvested from the conversation the loop sends back
    to the wire (``prepped_messages`` carries the full exchange, including
    loop-produced result messages), exactly where a real echoing or bridged
    client would get them — including CALLER HISTORY when the run is seeded
    with one, whose objects the memo must treat as landed from the start.
    After ``tool_turns`` turns it returns a plain text response so the loop
    terminates.

    The oracle contract for tests: every fresh call executes exactly once and
    yields exactly one result; echoes execute nothing and leave no trace in
    the final transcript (enforced by ``assert_transcript_invariants`` via
    ``InvariantCheckedToolLoopLayer``).
    """

    def __init__(self, rng: random.Random, tool_turns: int) -> None:
        self.rng = rng
        self.remaining_tool_turns = tool_turns
        self.fresh_calls: list[Content] = []
        self.minted_ids: list[str] = []
        self.returned_messages: list[Message] = []
        self._serial = 0
        # Raw fragments emitted on the turn in flight vs. fragments whose
        # turn has landed — only the latter are fair game for replay (a
        # same-turn duplicate object is a different shape: within one
        # response, assembly merges it instead of the memo stripping it).
        self._pending_fragments: list[Content] = []
        self._replayable_fragments: list[Content] = []

    def _fresh_call(self, response_ids: set[str]) -> Content:
        reusable = [i for i in self.minted_ids if i not in response_ids]
        if reusable and self.rng.random() < 0.4:
            call_id = self.rng.choice(reusable)
        else:
            call_id = f"call_{self._serial}"
            self.minted_ids.append(call_id)
        arguments: Any = {"text": f"t{self._serial}"}
        if self.rng.random() < 0.3:
            # String-typed arguments (the wire representation): a repeated
            # object corrupts by self-concatenation if assembly ever
            # merges it with itself, where dict arguments merge silently.
            arguments = json.dumps(arguments)
        call = Content.from_function_call(call_id, "echo", arguments=arguments)
        self._serial += 1
        self.fresh_calls.append(call)
        response_ids.add(call_id)
        return call

    def _build_turn_messages(self, conversation: list[Message], *, distinct_call_ids: bool) -> list[Message]:
        # ``distinct_call_ids`` keeps a streamed turn's function-call ids
        # unique: streaming assembly coalesces same-id call contents within
        # one response as fragments of a single call, so an echoed call
        # sharing a fresh call's id would be MERGED with it — a shape real
        # providers never emit inside one response (ids are unique per
        # response; reuse happens across responses, which both fuzz modes
        # still exercise). The non-streaming mode leaves collisions in to
        # also cover echo-beside-same-id-fresh-call within one response.
        prior_messages = [m for m in conversation if m.role in ("assistant", "tool")]
        prior_calls = [c for m in prior_messages for c in m.contents if c.type == "function_call"]
        prior_results = [c for m in prior_messages for c in m.contents if c.type == "function_result"]

        response_ids: set[str] = set()
        contents: list[Content] = [self._fresh_call(response_ids) for _ in range(self.rng.randint(1, 2))]
        if prior_calls and self.rng.random() < 0.6:
            echoed_call = self.rng.choice(prior_calls)
            if not (distinct_call_ids and echoed_call.call_id in response_ids):
                response_ids.add(echoed_call.call_id or "")
                contents.insert(self.rng.randrange(len(contents) + 1), echoed_call)
        if prior_calls and distinct_call_ids and self.rng.random() < 0.35:
            # Laundering shape (streamed turns): the SAME landed call object
            # twice ADJACENTLY — assembly would merge the pair into a NEW
            # object, so the echo fragments must be stripped before assembly.
            echoed_call = self.rng.choice(prior_calls)
            if echoed_call.call_id not in response_ids:
                response_ids.add(echoed_call.call_id or "")
                contents.extend([echoed_call, echoed_call])
                if self.rng.random() < 0.5:
                    # Poison combo: a FRESH call reusing the laundered id
                    # later in the same turn (text keeps it out of the
                    # merge). The old arguments must not run again, and the
                    # fresh call must not fall to duplicate-id dispatch.
                    contents.append(Content.from_text("separator"))
                    fresh_reuse = Content.from_function_call(
                        echoed_call.call_id, "echo", arguments={"text": f"t{self._serial}"}
                    )
                    self._serial += 1
                    self.fresh_calls.append(fresh_reuse)
                    contents.append(fresh_reuse)
        if self.rng.random() < 0.4:
            # Same fresh object twice in one response — regardless of how the
            # landing/assembly path normalizes the duplicate, the oracle
            # holds it to one execution and one transcript copy.
            contents.append(self.rng.choice([c for c in contents if c.type == "function_call"]))

        hosted_continuation: Content | None = None
        if self.rng.random() < 0.35:
            # Hosted-tool shapes widen the fuzz beyond function calls:
            # provider-executed calls arrive answered within the same
            # response — embedded in the call-carrying message itself
            # (either content order: exchange pairing is message-phased) or
            # riding a result-only assistant continuation message. Ids are
            # minted unique, hosted calls are never dispatched or stamped,
            # and nothing here enters the echo pools, which harvest
            # function shapes only.
            hosted_id = f"hosted_{self._serial}"
            self._serial += 1
            if self.rng.random() < 0.5:
                hosted_call = Content.from_mcp_server_tool_call(hosted_id, "search")
                hosted_result = Content.from_mcp_server_tool_result(hosted_id, output="ok")
            else:
                hosted_call = Content.from_image_generation_tool_call(image_id=hosted_id)
                hosted_result = Content.from_image_generation_tool_result(image_id=hosted_id)
            placement = self.rng.random()
            if placement < 0.4:
                contents.extend([hosted_call, hosted_result])
            elif placement < 0.7:
                contents.extend([hosted_result, hosted_call])
            else:
                contents.append(hosted_call)
                hosted_continuation = hosted_result

        minted = Message("assistant", contents)
        self.returned_messages.append(minted)
        messages = [minted]
        if prior_results and self.rng.random() < 0.6:
            messages.append(Message("tool", [self.rng.choice(prior_results)]))
        if prior_messages and self.rng.random() < 0.5:
            echoed_message = self.rng.choice(prior_messages)
            echoed_ids = {c.call_id for c in echoed_message.contents if c.type == "function_call" and c.call_id}
            if not (distinct_call_ids and echoed_ids & response_ids):
                response_ids.update(echoed_ids)
                messages.insert(self.rng.randrange(len(messages) + 1), echoed_message)
        input_assistants = [m for m in prior_messages if m.role == "assistant"]
        if input_assistants and not distinct_call_ids and self.rng.random() < 0.3:
            # Reverse aliasing: mutate a history message received as INPUT
            # and re-send that same object. The wire view makes the mutation
            # land on a throwaway wrapper; at landing the old contents drop
            # as echoes and only the appended call lands. Always re-sent so
            # the execution oracle's fresh-call count stays exact.
            victim = self.rng.choice(input_assistants)
            victim.contents.append(self._fresh_call(response_ids))
            if all(m is not victim for m in messages):
                messages.insert(self.rng.randrange(len(messages) + 1), victim)
        earlier_returned = [m for m in self.returned_messages if m is not minted]
        if earlier_returned and not distinct_call_ids and self.rng.random() < 0.35:
            # Stateful-client mutation: append a fresh call INTO a message
            # this client returned on an earlier turn, then re-send that same
            # object. The append must change nothing the loop already landed
            # (landing snapshots wrappers); at landing the old contents drop
            # as echoes and only the appended call lands. Always re-sent so
            # the execution oracle's fresh-call count stays exact.
            victim = self.rng.choice(earlier_returned)
            victim.contents.append(self._fresh_call(response_ids))
            if all(m is not victim for m in messages):
                messages.insert(self.rng.randrange(len(messages) + 1), victim)
        if hosted_continuation is not None:
            # The continuation goes LAST: a call-carrying assistant behind a
            # result-only assistant would start a new exchange, while the
            # loop folds every executed result after the whole response —
            # real hosted results either share the call's message or close
            # the turn, so the fuzz keeps them turn-closing too.
            messages.append(Message("assistant", [hosted_continuation]))
        return messages

    def get_response(
        self,
        messages: Any,
        *,
        stream: bool = False,
        options: Any = None,
        **kwargs: Any,
    ) -> Any:
        conversation = list(messages)
        # The previous turn has landed by the time the loop calls again, so
        # its fragments graduate to the replayable pool.
        self._replayable_fragments.extend(self._pending_fragments)
        self._pending_fragments = []
        fresh_before = len(self.fresh_calls)
        if self.remaining_tool_turns > 0:
            self.remaining_tool_turns -= 1
            turn_messages = self._build_turn_messages(conversation, distinct_call_ids=stream)
        else:
            turn_messages = [Message("assistant", [Content.from_text("done")])]
        turn_fresh_ids = {id(c) for c in self.fresh_calls[fresh_before:]}

        if not stream:
            turn_response = ChatResponse(messages=turn_messages)

            async def _resolve() -> ChatResponse:
                return turn_response

            return _resolve()

        async def _gen() -> Any:
            replay = (
                self.rng.choice(self._replayable_fragments)
                if self._replayable_fragments and self.rng.random() < 0.35
                else None
            )
            if replay is not None and self.rng.random() < 0.5:
                yield ChatResponseUpdate(contents=[replay], role="assistant")
                replay = None
            all_contents = [c for m in turn_messages for c in m.contents]
            for message in turn_messages:
                whole: list[Content] = []
                for content in message.contents:
                    # Only calls minted THIS turn and appearing once may be
                    # fragmented: fragmenting an echoed prior-turn object (or
                    # one leg of a duplicated pair) would mint new objects
                    # carrying old data — a data copy, which the identity
                    # model deliberately treats as fresh work.
                    fragment = (
                        content.type == "function_call"
                        and id(content) in turn_fresh_ids
                        and sum(1 for c in all_contents if c is content) == 1
                        and self.rng.random() < 0.35
                    )
                    if not fragment:
                        whole.append(content)
                        continue
                    if whole:
                        yield ChatResponseUpdate(contents=whole, role=message.role)
                        whole = []
                    args_json = (
                        content.arguments if isinstance(content.arguments, str) else json.dumps(content.arguments)
                    )
                    split_at = self.rng.randint(1, len(args_json) - 1)
                    frag_pair = [
                        Content.from_function_call(content.call_id, content.name, arguments=args_json[:split_at]),
                        Content.from_function_call(content.call_id, content.name, arguments=args_json[split_at:]),
                    ]
                    self._pending_fragments.extend(frag_pair)
                    for frag in frag_pair:
                        yield ChatResponseUpdate(contents=[frag], role=message.role)
                if whole:
                    yield ChatResponseUpdate(contents=whole, role=message.role)
            if replay is not None:
                yield ChatResponseUpdate(contents=[replay], role="assistant")

        rs: ResponseStream[ChatResponseUpdate, ChatResponse] = ResponseStream(
            _gen(), finalizer=ChatResponse.from_updates
        )
        return rs


def _fresh_call_text(call: Content) -> str:
    """The ``text`` argument of a fuzz fresh call, dict- or string-typed."""
    args = call.arguments
    if isinstance(args, str):
        args = json.loads(args)
    return args["text"]


def _fuzz_history(rng: random.Random) -> list[Message]:
    """Seeded caller history: 0-2 past tool exchanges the client can echo.

    Historical call/result objects reach the client through the wire views
    (shared content objects), so every echo branch of the adversarial client
    naturally exercises the history-seeded memo as well.
    """
    history: list[Message] = [_user()]
    for i in range(rng.randint(0, 2)):
        history.append(
            Message("assistant", [Content.from_function_call(f"hist_{i}", "echo", arguments={"text": f"h{i}"})])
        )
        history.append(Message("tool", [Content.from_function_result(f"hist_{i}", result=f"echo:h{i}")]))
    history.append(_user("again"))
    return history


class TestTranscriptInvariantFuzz:
    """Seeded adversarial fuzz of the landing/dispatch/fold transcript contract.

    Rationale: the echo-normalization defects were each found one
    counterexample at a time (dangling duplicate call, suppressed same-id
    fresh call, in-place mutation of a shared message, double-recorded
    result). This harness generates the whole input space those shapes came
    from — random compositions of fresh work, id reuse, and object echoes —
    and holds the loop to the structural oracle plus an exact execution
    count. A regression in ANY of the normalization rules surfaces here
    without anyone having to imagine the specific failing shape first.

    Seeds are fixed and each seed is its own parametrized case, so any
    failure — including one raised by the structural oracle inside the
    checked layer — names its seed in the failing test id; rerun exactly that
    case with ``-k`` while debugging. Do not replace the seeded rng with
    entropy — flaky guards get deleted, deterministic ones get fixed.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("seed", range(30))
    async def test_adversarial_echo_clients_never_break_transcript_invariants(self, seed: int) -> None:
        rng = random.Random(seed)
        wire = _AdversarialEchoClient(rng, tool_turns=rng.randint(1, 3))
        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))
        events: list[str] = []
        history = _fuzz_history(rng)
        history_shapes = [[c.call_id for c in m.contents] for m in history]

        response = await layer.get_response(history, options={"tools": [_make_tool(events)]})

        # The structural invariants already ran inside the checked layer.
        # The execution oracle pins behavior the structure alone cannot:
        # every fresh call ran exactly once, echoes ran zero times, and the
        # caller's history objects came through structurally untouched.
        expected = sorted(f"tool:echo:{_fresh_call_text(c)}" for c in wire.fresh_calls)
        assert sorted(events) == expected, "fresh-call executions diverged"
        assert len(_call_contents(response)) == len(wire.fresh_calls), "echoed call leaked"
        assert len(_result_contents(response)) == len(wire.fresh_calls), "result count diverged"
        assert [[c.call_id for c in m.contents] for m in history] == history_shapes, "caller history corrupted"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("seed", range(20))
    async def test_adversarial_echo_clients_never_break_streaming_invariants(self, seed: int) -> None:
        # Streaming assembly preserves object identity for whole-content
        # updates, so the same echo shapes must normalize identically; whole
        # echoed Messages degrade to their contents (assembly builds fresh
        # containers), which the content-identity memo still covers — except
        # the adjacent double echo, stripped from assembly input before its
        # merge can launder the identity, and replayed raw fragments, caught
        # by post-landing fragment-identity recording.
        rng = random.Random(seed)
        wire = _AdversarialEchoClient(rng, tool_turns=rng.randint(1, 3))
        layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))
        events: list[str] = []
        history = _fuzz_history(rng)
        history_shapes = [[c.call_id for c in m.contents] for m in history]

        stream = layer.get_response(history, stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        expected = sorted(f"tool:echo:{_fresh_call_text(c)}" for c in wire.fresh_calls)
        assert sorted(events) == expected, "fresh-call executions diverged"
        assert len(_call_contents(final)) == len(wire.fresh_calls), "echoed call leaked"
        assert len(_result_contents(final)) == len(wire.fresh_calls), "result count diverged"
        assert [[c.call_id for c in m.contents] for m in history] == history_shapes, "caller history corrupted"


class TestStreamingReconstruction:
    """The final streamed response reuses the loop's assembled messages, never a re-merge."""

    @pytest.mark.asyncio
    async def test_streamed_double_echo_of_landed_call_does_not_reexecute(self) -> None:
        """Assembly must not launder an echoed call's identity past the memo.

        Consecutive same-call fragments merge into a NEW Content object, so
        an already-landed call object echoed twice adjacently would dodge the
        identity memo. Echo fragments are stripped from assembly input, so
        the merge copy never exists and nothing re-executes.
        """
        events: list[str] = []
        c1_obj = Content.from_function_call("c1", "echo", arguments={"text": "once"})
        layer, _wire = _stack(
            [
                [ChatResponseUpdate(contents=[c1_obj], role="assistant")],
                [
                    ChatResponseUpdate(contents=[c1_obj], role="assistant"),
                    ChatResponseUpdate(contents=[c1_obj], role="assistant"),
                ],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:once"], "the landed call executes exactly once"
        assert [c.call_id for c in _call_contents(final)] == ["c1"]
        assert [r.call_id for r in _result_contents(final)] == ["c1"]

    @pytest.mark.asyncio
    async def test_streamed_same_object_twice_in_one_response_executes_once(self) -> None:
        """Streaming counterpart of the blocking identity-duplicate collapse.

        The echo filter only knows PREVIOUSLY landed identities, so a fresh
        object emitted twice within one streamed response reaches assembly
        twice — and adjacent copies would SELF-MERGE, concatenating the
        call's own argument string into unparseable JSON and turning a
        valid call into an argument-parsing failure. Assembly skips a
        content object it already incorporated in the same pass, matching
        the blocking path's one-copy-one-execution collapse at landing.
        """
        events: list[str] = []
        call = Content.from_function_call(call_id="c1", name="echo", arguments='{"text": "once"}')
        layer, _wire = _stack(
            [
                [
                    ChatResponseUpdate(contents=[call], role="assistant"),
                    ChatResponseUpdate(contents=[call], role="assistant"),
                ],
                [_text_update("done")],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:once"], "the duplicated object must execute exactly once"
        assert [c.call_id for c in _call_contents(final)] == ["c1"]
        assert [r.call_id for r in _result_contents(final)] == ["c1"]

    @pytest.mark.asyncio
    async def test_streamed_same_object_twice_nonadjacent_executes_once(self) -> None:
        """A non-adjacent within-response duplicate collapses to one copy too.

        With text between the two emissions there is no self-merge hazard;
        assembly-pass dedup (and, failing that, landing) must still keep a
        single copy — parity with the blocking path.
        """
        events: list[str] = []
        call = Content.from_function_call(call_id="c1", name="echo", arguments='{"text": "once"}')
        layer, _wire = _stack(
            [
                [
                    ChatResponseUpdate(contents=[call], role="assistant"),
                    ChatResponseUpdate(contents=[Content.from_text("thinking")], role="assistant"),
                    ChatResponseUpdate(contents=[call], role="assistant"),
                ],
                [_text_update("done")],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:once"]
        assert [c.call_id for c in _call_contents(final)] == ["c1"]
        assert [r.call_id for r in _result_contents(final)] == ["c1"]

    @pytest.mark.asyncio
    async def test_streamed_fragment_replay_after_merge_does_not_reexecute(self) -> None:
        """Raw stream fragments are conversation history too.

        A multi-fragment call assembles into a NEW merged Content, so only
        the merged copy reaches the transcript — if the memo held nothing
        else, the raw fragment objects would stay unknown and a stateful
        client replaying its own buffered fragments next turn would
        re-execute the call. Accepted update-content identities are
        recorded after landing, so a replayed fragment strips as an echo
        while genuinely fresh work still lands.
        """
        events: list[str] = []
        frag_a = Content.from_function_call(call_id="c1", name="echo", arguments='{"text": "on')
        frag_b = Content.from_function_call(call_id="c1", name="echo", arguments='ce"}')
        fresh = Content.from_function_call("c2", "echo", arguments={"text": "fresh"})
        layer, _wire = _stack(
            [
                [
                    ChatResponseUpdate(contents=[frag_a], role="assistant"),
                    ChatResponseUpdate(contents=[frag_b], role="assistant"),
                ],
                [
                    ChatResponseUpdate(contents=[frag_a], role="assistant"),
                    ChatResponseUpdate(contents=[frag_b], role="assistant"),
                    ChatResponseUpdate(contents=[fresh], role="assistant"),
                ],
                [_text_update("done")],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:once", "tool:echo:fresh"], "replayed fragments must not re-merge and re-execute"
        assert [c.call_id for c in _call_contents(final)] == ["c1", "c2"]
        assert [r.call_id for r in _result_contents(final)] == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_echo_stripping_survives_response_validation_proxy(self) -> None:
        """The strip filter must reach BENEATH a draining stream proxy.

        ``ResponseValidationMiddleware`` returns its own ``ResponseStream``
        whose generator drains the provider stream, preview-finalizes, and
        replays updates — ``with_update_filter`` push-down cannot cross that
        semantic boundary (the proxy's source is a plain generator), so the
        filter is delivered on the request path and the middleware pipeline
        attaches it to the stream its final handler resolves. Without that,
        the proxy's cached final response is assembled UNFILTERED and the
        laundering P1 returns in the production topology.
        """
        from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware

        events: list[str] = []
        landed = Content.from_function_call("c1", "echo", arguments={"text": "old"})
        fresh_same_id = Content.from_function_call("c1", "echo", arguments={"text": "new"})
        wire = _ScriptedClient(
            [
                [ChatResponseUpdate(contents=[landed], role="assistant")],
                [
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(contents=[Content.from_text("thinking")], role="assistant"),
                    ChatResponseUpdate(contents=[fresh_same_id], role="assistant"),
                ],
                [_text_update("done")],
            ]
        )
        layer = InvariantCheckedToolLoopLayer(
            ChatMiddlewareLayer(wire, middleware=[ResponseValidationMiddleware(backoff_schedule=[0.0])])
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:old", "tool:echo:new"], "no re-execution, no swallowed fresh call"
        assert [c.arguments["text"] for c in _call_contents(final)] == ["old", "new"]

    @pytest.mark.asyncio
    async def test_echo_stripping_reattaches_on_validation_retry(self) -> None:
        """Every validation retry's fresh provider stream gets the filter.

        Attempt one of the second loop turn returns whitespace-only text
        (invalid, retried inside the validation proxy); the retry attempt
        carries the echo-laundering shape. The filter must cover the RETRY
        attempt's inner stream, not just the first.
        """
        from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware

        events: list[str] = []
        landed = Content.from_function_call("c1", "echo", arguments={"text": "old"})
        fresh_same_id = Content.from_function_call("c1", "echo", arguments={"text": "new"})
        wire = _ScriptedClient(
            [
                [ChatResponseUpdate(contents=[landed], role="assistant")],
                [_text_update("   \n  ")],
                [
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(contents=[Content.from_text("thinking")], role="assistant"),
                    ChatResponseUpdate(contents=[fresh_same_id], role="assistant"),
                ],
                [_text_update("done")],
            ]
        )
        layer = InvariantCheckedToolLoopLayer(
            ChatMiddlewareLayer(wire, middleware=[ResponseValidationMiddleware(backoff_schedule=[0.0])])
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:old", "tool:echo:new"]
        assert [c.arguments["text"] for c in _call_contents(final)] == ["old", "new"]

    @pytest.mark.asyncio
    async def test_echo_stripping_drops_only_the_message_shell_it_emptied(self) -> None:
        """An echo in one message must not erase a different empty message.

        A genuine empty shell precedes an echo-only shell and fresh content.
        Streaming echo removal must discard only the middle shell; a
        call-wide "anything was stripped" flag would erase both empty
        messages.
        """
        historical = Content.from_text("historical")
        layer, _wire = _stack(
            [
                [
                    ChatResponseUpdate(contents=[], role="assistant", message_id="legitimate-empty"),
                    ChatResponseUpdate(contents=[historical], role="assistant", message_id="echo-only"),
                    ChatResponseUpdate(
                        contents=[Content.from_text("fresh")],
                        role="assistant",
                        message_id="fresh",
                    ),
                ]
            ]
        )

        stream = layer.get_response([Message("user", [historical])], stream=True)
        _ = [update async for update in stream]
        final = await stream.get_final_response()

        assert [(message.message_id, message.text) for message in final.messages] == [
            ("legitimate-empty", ""),
            ("fresh", "fresh"),
        ]

    @pytest.mark.asyncio
    async def test_echo_emptied_marker_isolated_from_validation_retry_message_id_collision(self) -> None:
        """A rejected attempt cannot mark a same-id accepted shell as echo.

        Providers may restart positional message IDs on every response.
        The rejected attempt strips an echo from ``msg_0``; the accepted
        attempt legitimately emits an empty ``msg_0`` tool message. Cleanup
        must follow accepted update identity, not an ID set shared across
        validation retries.
        """
        from chrys.service.agent_middleware.response_validation import ResponseValidationMiddleware

        historical = Content.from_text("historical")
        wire = _ScriptedClient(
            [
                [ChatResponseUpdate(contents=[historical], role="assistant", message_id="msg_0")],
                [
                    ChatResponseUpdate(contents=[], role="tool", message_id="msg_0"),
                    ChatResponseUpdate(
                        contents=[Content.from_text("fresh")],
                        role="assistant",
                        message_id="msg_1",
                    ),
                ],
            ]
        )
        layer = InvariantCheckedToolLoopLayer(
            ChatMiddlewareLayer(wire, middleware=[ResponseValidationMiddleware(backoff_schedule=[0.0])])
        )

        stream = layer.get_response([Message("user", [historical])], stream=True)
        _ = [update async for update in stream]
        final = await stream.get_final_response()

        assert len(wire.calls) == 2
        assert [(message.message_id, message.role, message.text) for message in final.messages] == [
            ("msg_0", "tool", ""),
            ("msg_1", "assistant", "fresh"),
        ]

    @pytest.mark.asyncio
    async def test_blocking_and_streaming_echo_cleanup_preserve_the_same_message_shape(self) -> None:
        """Blocking and streaming converge for empty, echo, fresh messages."""
        blocking_historical = Content.from_text("historical")
        blocking_layer, _wire = _stack(
            [
                ChatResponse(
                    messages=[
                        Message("assistant", [], message_id="legitimate-empty"),
                        Message("assistant", [blocking_historical], message_id="echo-only"),
                        Message("assistant", ["fresh"], message_id="fresh"),
                    ]
                )
            ]
        )
        blocking = await blocking_layer.get_response([Message("user", [blocking_historical])])

        streaming_historical = Content.from_text("historical")
        streaming_layer, _wire = _stack(
            [
                [
                    ChatResponseUpdate(contents=[], role="assistant", message_id="legitimate-empty"),
                    ChatResponseUpdate(contents=[streaming_historical], role="assistant", message_id="echo-only"),
                    ChatResponseUpdate(
                        contents=[Content.from_text("fresh")],
                        role="assistant",
                        message_id="fresh",
                    ),
                ]
            ]
        )
        stream = streaming_layer.get_response([Message("user", [streaming_historical])], stream=True)
        _ = [update async for update in stream]
        streaming = await stream.get_final_response()

        def shape(response: ChatResponse) -> list[tuple[str | None, str, str]]:
            return [(message.message_id, message.role, message.text) for message in response.messages]

        assert (
            shape(streaming)
            == shape(blocking)
            == [
                ("legitimate-empty", "assistant", ""),
                ("fresh", "assistant", "fresh"),
            ]
        )

    @pytest.mark.asyncio
    async def test_echo_stripping_does_not_resurrect_hook_removed_calls(self) -> None:
        """Echo handling must compose with result hooks, never race them.

        ``get_final_response`` runs the provider finalizer and result hooks;
        a hook deleting a call from the assembled response is a safety gate
        the loop promises runs before tool extraction. When the same turn
        also carries echo fragments, echo handling must not re-assemble the
        response behind the hook's back and resurrect the deleted call —
        echo fragments are stripped BEFORE assembly (update filter), so the
        hook operates on the one and only assembly.
        """
        events: list[str] = []
        landed = Content.from_function_call("c1", "echo", arguments={"text": "old"})

        def drop_danger(response: ChatResponse) -> ChatResponse:
            for message in response.messages:
                message.contents = [
                    c for c in message.contents if not (c.type == "function_call" and c.name == "danger")
                ]
            return response

        layer, _wire = _stack(
            [
                [ChatResponseUpdate(contents=[landed], role="assistant")],
                [
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(
                        contents=[Content.from_function_call("c9", "danger", arguments={"text": "boom"})],
                        role="assistant",
                    ),
                    ChatResponseUpdate(
                        contents=[Content.from_function_call("c2", "echo", arguments={"text": "new"})],
                        role="assistant",
                    ),
                ],
                [_text_update("done")],
            ],
            result_hook=drop_danger,
        )
        tools = [_make_tool(events), _make_tool(events, name="danger")]

        stream = layer.get_response([_user()], stream=True, options={"tools": tools})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:old", "tool:echo:new"], "hook-removed danger never runs; echo never re-runs"
        assert [c.call_id for c in _call_contents(final)] == ["c1", "c2"]
        assert [r.call_id for r in _result_contents(final)] == ["c1", "c2"]

    @pytest.mark.asyncio
    async def test_streamed_double_echo_merge_beside_same_id_fresh_call_executes_fresh_only(self) -> None:
        """A laundered echo merge must not shadow fresh same-id work.

        The poison combo: an already-landed call object echoed twice
        adjacently (assembly merges the fragments into a NEW object) AND a
        fresh call reusing the same call id later in the same stream turn.
        Response-level per-id provenance cannot express this — the fresh
        fragment would exonerate the id, letting the merged echo copy land,
        execute the OLD arguments again, and shadow the fresh call via
        duplicate-id dispatch filtering. Echo fragments must be removed from
        assembly input instead, so the merge copy never exists.
        """
        events: list[str] = []
        landed = Content.from_function_call("c1", "echo", arguments={"text": "old"})
        fresh_same_id = Content.from_function_call("c1", "echo", arguments={"text": "new"})
        layer, _wire = _stack(
            [
                [ChatResponseUpdate(contents=[landed], role="assistant")],
                [
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(contents=[Content.from_text("thinking")], role="assistant"),
                    ChatResponseUpdate(contents=[fresh_same_id], role="assistant"),
                ],
                [_text_update("done")],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:old", "tool:echo:new"], "old ran once in turn 1; the fresh call still runs"
        assert [c.arguments["text"] for c in _call_contents(final)] == ["old", "new"]

    @pytest.mark.asyncio
    async def test_streamed_fresh_same_id_call_beside_echo_stays_fresh_work(self) -> None:
        """Echo removal must not over-remove: id reuse by a fresh object stays fresh.

        One echoed landed object plus a DIFFERENT fresh call object reusing
        the same call id in the same stream turn: the echo is stripped from
        assembly input, and the fresh call must still land and execute —
        providers legitimately reuse ids across responses.
        """
        events: list[str] = []
        landed = Content.from_function_call("c1", "echo", arguments={"text": "old"})
        fresh_same_id = Content.from_function_call("c1", "echo", arguments={"text": "new"})
        layer, _wire = _stack(
            [
                [ChatResponseUpdate(contents=[landed], role="assistant")],
                [
                    ChatResponseUpdate(contents=[landed], role="assistant"),
                    ChatResponseUpdate(contents=[Content.from_text("thinking")], role="assistant"),
                    ChatResponseUpdate(contents=[fresh_same_id], role="assistant"),
                ],
                [_text_update("done")],
            ]
        )

        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool(events)]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert events == ["tool:echo:old", "tool:echo:new"], "fresh same-id work still executes"
        assert [c.arguments["text"] for c in _call_contents(final)] == ["old", "new"]
        assert all(not message._chrys_echo_content_stripped for message in final.messages)

    @pytest.mark.asyncio
    async def test_fragmented_call_keeps_stamps_and_metadata_through_final_response(self) -> None:
        from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY
        from chrys.foundation.tool_kinds import TOOL_CALL_KIND_METADATA_KEY
        from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY

        frag1 = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c1", name="echo", arguments='{"text": "he')],
            role="assistant",
        )
        frag2 = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c1", name="echo", arguments='llo"}')],
            role="assistant",
        )
        tool_obj = _provenance_tool(static={"server_name": "static"})
        carrier = _CarriageFunction(result_metadata={"shell_exit_code": 0}, tool_context={"built": "filled"})
        layer, _wire = _stack([[frag1, frag2], [_text_update("done")]])

        stream = layer.get_response([_user()], stream=True, options={"tools": [tool_obj]}, middleware=[carrier])
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        calls = _call_contents(final)
        assert len(calls) == 1
        merged_call = calls[0]
        assert merged_call.parse_arguments() == {"text": "hello"}
        assert merged_call.additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert merged_call.additional_properties[TOOL_CALL_KIND_METADATA_KEY] == "probe.kind"
        assert merged_call.additional_properties[TOOL_CALL_CONTEXT_METADATA_KEY] == {
            "server_name": "static",
            "built": "filled",
        }
        result = _result_contents(final)[0]
        assert str(result.result) == "echo:hello"
        assert result.additional_properties[TOOL_RESULT_METADATA_KEY] == {"shell_exit_code": 0}

    @pytest.mark.asyncio
    async def test_final_response_snapshots_wrappers_and_reuses_content_objects(self) -> None:
        # The result hook runs before landing rebinds ``response.messages``,
        # so copying the list here captures the assembled pre-landing wrappers.
        assembled_wrappers: list[list[Message]] = []

        def capture(response: ChatResponse) -> ChatResponse:
            assembled_wrappers.append(list(response.messages))
            return response

        layer, _wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [_text_update("final")]], result_hook=capture
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert len(assembled_wrappers) == 2
        # Landing retains loop-owned Message snapshots (a client mutating a
        # wrapper it returned must not rewrite landed history); the CONTENT
        # objects are reused by identity — the echo memo depends on it.
        assert final.messages[0] is not assembled_wrappers[0][0]
        assert final.messages[0].contents[0] is assembled_wrappers[0][0].contents[0]
        assert final.messages[-1] is not assembled_wrappers[1][-1]
        assert final.messages[-1].contents[0] is assembled_wrappers[1][-1].contents[0]
        assert [m.role for m in final.messages] == ["assistant", "tool", "assistant"]

    @pytest.mark.asyncio
    async def test_no_call_single_turn_snapshots_wrapper_and_reuses_content(self) -> None:
        assembled_wrappers: list[list[Message]] = []

        def capture(response: ChatResponse) -> ChatResponse:
            assembled_wrappers.append(list(response.messages))
            return response

        layer, _wire = _stack([[_text_update("hello")]], result_hook=capture)
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert final.messages[0] is not assembled_wrappers[0][0]
        assert final.messages[0].contents[0] is assembled_wrappers[0][0].contents[0]

    @pytest.mark.asyncio
    async def test_termination_exit_keeps_accumulated_transcript_with_stamps(self) -> None:
        layer, _wire = _stack([[_call_update("c1", "echo", {"text": "a"})], [_text_update("never")]])
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()]},
            middleware=[_TerminateFunction(result="stopped")],
        )
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        # Streaming termination keeps the accumulated transcript (deliberate
        # asymmetry with the non-streaming termination return).
        kinds = [item.type for msg in final.messages for item in msg.contents]
        assert "function_call" in kinds
        assert "function_result" in kinds
        assert _call_contents(final)[0].additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert str(_result_contents(final)[0].result) == "stopped"

    @pytest.mark.asyncio
    async def test_error_cap_exit_keeps_stamps(self) -> None:
        @tool(name="boom")
        async def boom(text: str) -> str:
            raise RuntimeError("kaboom")

        layer, wire = _stack(
            [[_call_update("c1", "boom", {"text": "a"})], [_text_update("forced")]],
            max_consecutive_errors=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [boom]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert wire.calls[1]["options"].get("tool_choice") == "none"
        assert _call_contents(final)[0].additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert final.messages[-1].contents[-1].text == "forced"

    @pytest.mark.asyncio
    async def test_exhaustion_tail_keeps_stamps(self) -> None:
        layer, wire = _stack(
            [[_call_update("c1", "echo", {"text": "a"})], [_text_update("forced")]],
            max_iterations=1,
        )
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert wire.calls[1]["options"].get("tool_choice") == "none"
        assert _call_contents(final)[0].additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert final.messages[-1].contents[-1].text == "forced"

    @pytest.mark.asyncio
    async def test_degenerate_stream_falls_back_to_from_updates(self) -> None:
        """No loop exit completed: _finalize keeps the raw-update merge as a fallback."""

        class _RaisingClient:
            def get_response(self, messages: Any, *, stream: bool = False, options: Any = None, **kwargs: Any) -> Any:
                async def _gen() -> Any:
                    yield _call_update("c1", "echo", {"text": "a"})
                    raise RuntimeError("wire died")

                return ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        # Bare ToolLoopLayer on purpose (hygiene-allowlisted): the wire dies
        # mid-stream, no loop iteration ever lands, and the fallback re-merge
        # below is NOT a loop-landed transcript — the invariant oracle's
        # precondition does not hold for it.
        layer = ToolLoopLayer(ChatMiddlewareLayer(_RaisingClient()))
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        with pytest.raises(RuntimeError, match="wire died"):
            async for _ in stream:
                pass
        final = await stream.get_final_response()

        # Fallback reconstruction: raw updates re-merged (no loop stamps —
        # the iteration never landed), but the transcript is not lost.
        calls = _call_contents(final)
        assert len(calls) == 1
        assert TOOL_INVOCATION_ORDER_KEY not in calls[0].additional_properties

    @pytest.mark.asyncio
    async def test_boundary_delta_starts_fresh_assistant_message(self) -> None:
        """A first final-turn delta with no role and no message_id must not glue onto the tool message."""
        call_update = ChatResponseUpdate(
            contents=[Content.from_function_call(call_id="c1", name="echo", arguments={"text": "a"})],
            role="assistant",
            message_id="m1",
        )
        boundary_delta = ChatResponseUpdate(
            contents=[Content.from_text("final")],
            role=None,
            message_id=None,
        )
        usage_update = ChatResponseUpdate(
            contents=[
                Content.from_usage(
                    usage_details={"input_token_count": 1, "output_token_count": 2, "total_token_count": 3}
                )
            ],
            role=None,
            message_id=None,
        )
        layer, _wire = _stack([[call_update], [boundary_delta, usage_update]])
        stream = layer.get_response([_user()], stream=True, options={"tools": [_make_tool()]})
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert [m.role for m in final.messages] == ["assistant", "tool", "assistant"]
        assert final.messages[0].message_id == "m1"
        assert final.messages[-1].text == "final"
        assert all(item.type != "usage" for msg in final.messages for item in msg.contents)
        assert final.usage_details == {"input_token_count": 1, "output_token_count": 2, "total_token_count": 3}

    @pytest.mark.asyncio
    async def test_response_format_parses_structured_value_from_swapped_messages(self) -> None:
        layer, _wire = _stack([[_call_update("c1", "echo", {"text": "a"})], [_text_update('{"answer": 42}')]])
        stream = layer.get_response(
            [_user()],
            stream=True,
            options={"tools": [_make_tool()], "response_format": _StructuredAnswer},
        )
        _ = [u async for u in stream]
        final = await stream.get_final_response()

        assert _call_contents(final)[0].additional_properties[TOOL_INVOCATION_ORDER_KEY] == 0
        assert isinstance(final.value, _StructuredAnswer)
        assert final.value.answer == 42


def test_extract_function_calls_dedups_within_the_whole_response() -> None:
    """Response-scoped on purpose, not exchange-scoped: a call answered
    anywhere in the same response is resolved, and a repeated call_id is
    collected once — landing already removed echoed calls before this
    filter sees the response."""
    answered = Message(role="assistant", contents=[Content.from_function_call("c1", "first", arguments={})])
    repeated_first = Message(role="assistant", contents=[Content.from_function_call("c2", "second", arguments={})])
    repeated_second = Message(role="assistant", contents=[Content.from_function_call("c2", "second", arguments={})])
    response = ChatResponse(
        messages=[
            answered,
            Message(role="tool", contents=[Content.from_function_result("c1", result="done")]),
            repeated_first,
            repeated_second,
        ]
    )

    calls = _extract_function_calls(response)

    assert calls == [repeated_first.contents[0]]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
async def test_post_commit_wire_failure_preserves_exchange_and_blocks_outer_retry(stream: bool) -> None:
    tool_runs = 0
    wire_calls = 0

    @tool(name="commit_once")
    async def commit_once() -> str:
        nonlocal tool_runs
        tool_runs += 1
        return "committed"

    class _FailAfterToolClient:
        def get_response(self, _messages: Any, *, stream: bool = False, **_kwargs: Any) -> Any:
            nonlocal wire_calls
            wire_calls += 1
            if wire_calls <= 2:
                call_id = f"commit-{wire_calls}"
                if not stream:

                    async def _call() -> ChatResponse:
                        return _call_response((call_id, "commit_once", {}))

                    return _call()

                async def _call_updates() -> Any:
                    yield _call_update(call_id, "commit_once", {})

                return ResponseStream(_call_updates(), finalizer=ChatResponse.from_updates)
            if not stream:

                async def _fail() -> ChatResponse:
                    raise ConnectionError("wire failed after commit")

                return _fail()

            async def _fail_updates() -> Any:
                raise ConnectionError("wire failed after commit")
                yield  # pragma: no cover

            return ResponseStream(_fail_updates(), finalizer=ChatResponse.from_updates)

    recorder = LoopRecorder(capture_service_loop_messages=True)
    recorder_snapshot = recorder.snapshot()
    retry = StreamRetryLoop(
        max_retries=2,
        backoff_schedule=(0, 0),
        is_retryable=lambda _exc: True,
        snapshot_history=lambda: HistorySnapshot(messages=[], compressed_count=0),
        restore_history=lambda _snapshot: recorder.restore(recorder_snapshot),
        publish_retry_attempt=lambda *_args: asyncio.sleep(0),
        is_interrupted=lambda: False,
        interruptible_sleep=lambda _seconds: asyncio.sleep(0, result=False),
        clean_error_message=str,
        may_retry=lambda _exc: recorder.committed_count == 0,
    )
    layer = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(_FailAfterToolClient()))
    attempts = 0

    async def _attempt() -> ChatResponse:
        nonlocal attempts
        attempts += 1
        response = layer.get_response(
            [_user()],
            stream=stream,
            options={"tools": [commit_once], "store": True},
            client_kwargs={"loop_recorder": recorder},
        )
        if not stream:
            return await response
        assert isinstance(response, ResponseStream)
        async for _update in response:
            pass
        return await response.get_final_response()

    with pytest.raises(ConnectionError, match="wire failed after commit"):
        await retry.run(_attempt)

    assert attempts == 1
    assert tool_runs == 2
    assert recorder.committed_count == 2
    projected = recorder.loop_messages
    assert projected is not None
    assert [message.role for message in projected] == ["assistant", "tool", "assistant", "tool"]
    assert [projected[index].contents[0].result for index in (1, 3)] == ["committed", "committed"]

    state = {"messages": [_user()]}
    history = SessionHistoryManager()
    history.bind(state)
    history.merge_loop_messages(recorder, insert_index=1)
    history.trim_to_last_complete_tool_results()
    history.insert_interrupted_marker(reason="wire failed after commit", source="error")
    assert [
        content.call_id
        for message in history.messages
        for content in message.contents
        if content.type == "function_result"
    ] == ["commit-1", "commit-2"]
    assert history.messages[-1].additional_properties[HistoryMarkerKind.KEY] == HistoryMarkerKind.INTERRUPTED
    assert history.messages[-1].additional_properties["_interrupted_by"] == "error"


@pytest.mark.asyncio
async def test_post_execution_middleware_cancel_keeps_raw_result_and_marks_processing_interrupted() -> None:
    tool_runs = 0
    tool_returned = asyncio.Event()
    hold_post_processing = asyncio.Event()

    @tool(name="returned_tool")
    async def returned_tool() -> str:
        nonlocal tool_runs
        tool_runs += 1
        return "raw-result"

    class _HoldAfterReturn(FunctionMiddleware):
        async def process(
            self,
            _context: FunctionInvocationContext,
            call_next: Callable[[], Awaitable[None]],
        ) -> None:
            await call_next()
            tool_returned.set()
            await hold_post_processing.wait()

    measured_timing = {
        "started_at": "2026-08-19T01:02:03.000000+00:00",
        "finished_at": "2026-08-19T01:02:03.025000+00:00",
        "duration_ms": 25,
    }

    class _StampTimingOnExit(FunctionMiddleware):
        async def process(
            self,
            context: FunctionInvocationContext,
            call_next: Callable[[], Awaitable[None]],
        ) -> None:
            try:
                await call_next()
            finally:
                context.metadata[TRAJECTORY_TIMING_KEY] = dict(measured_timing)

    recorder = LoopRecorder()
    layer, _wire = _stack([_call_response(("c1", "returned_tool", {}))])
    task = asyncio.create_task(
        layer.get_response(
            [_user()],
            options={"tools": [returned_tool]},
            middleware=[_StampTimingOnExit(), _HoldAfterReturn()],
            client_kwargs={"loop_recorder": recorder},
        )
    )
    await tool_returned.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    projected = recorder.loop_messages
    assert projected is not None
    result = projected[-1].contents[0]
    call = projected[0].contents[0]
    metadata = result.additional_properties[TOOL_RESULT_METADATA_KEY]
    assert result.result == "raw-result"
    assert metadata[TOOL_INTERRUPTED_METADATA_KEY] is True
    assert metadata[TOOL_POST_PROCESSING_INTERRUPTED_METADATA_KEY] is True
    assert call.additional_properties[TRAJECTORY_TIMING_KEY] == measured_timing
    assert result.additional_properties[TRAJECTORY_TIMING_KEY] == measured_timing
    assert tool_runs == 1


@pytest.mark.asyncio
async def test_early_interrupted_slot_upgrades_timing_after_middleware_unwinds() -> None:
    """A cascade-style early fill accepts measured timing without replacing its result."""
    measured_timing = {
        "started_at": "2026-08-19T01:02:03.000000+00:00",
        "finished_at": "2026-08-19T01:02:03.025000+00:00",
        "duration_ms": 25,
    }

    class _StampTimingOnExit(FunctionMiddleware):
        async def process(
            self,
            context: FunctionInvocationContext,
            call_next: Callable[[], Awaitable[None]],
        ) -> None:
            try:
                await call_next()
            finally:
                context.metadata[TRAJECTORY_TIMING_KEY] = dict(measured_timing)

    class _CommitThenCancel(FunctionMiddleware):
        async def process(
            self,
            context: FunctionInvocationContext,
            _call_next: Callable[[], Awaitable[None]],
        ) -> None:
            callback = context.metadata[SUB_AGENT_RESULT_COMMIT_CALLBACK_KEY]
            assert callable(callback)
            callback({"sub_agent_invocation_id": "inv-1"})
            raise asyncio.CancelledError

    recorder = LoopRecorder()
    layer, _wire = _stack([_call_response(("c1", "echo", {"text": "unused"}))])

    with pytest.raises(asyncio.CancelledError):
        await layer.get_response(
            [_user()],
            options={"tools": [_make_tool()]},
            middleware=[_StampTimingOnExit(), _CommitThenCancel()],
            client_kwargs={"loop_recorder": recorder},
        )

    projected = recorder.loop_messages
    assert projected is not None
    call = projected[0].contents[0]
    result = projected[1].contents[0]
    assert result.additional_properties[TOOL_RESULT_METADATA_KEY]["sub_agent_invocation_id"] == "inv-1"
    assert call.additional_properties[TRAJECTORY_TIMING_KEY] == measured_timing
    assert result.additional_properties[TRAJECTORY_TIMING_KEY] == measured_timing


@pytest.mark.asyncio
async def test_batch_failure_waits_for_slower_sibling_commit_before_propagating() -> None:
    slow_started = asyncio.Event()
    release_slow = asyncio.Event()
    slow_finished = False

    @tool(name="slow_sibling")
    async def slow_sibling() -> str:
        nonlocal slow_finished
        slow_started.set()
        await release_slow.wait()
        slow_finished = True
        return "slow-done"

    missing_id = Content.from_function_call(None, "slow_sibling", arguments={})
    valid = Content.from_function_call("slow", "slow_sibling", arguments={})
    recorder = LoopRecorder()
    layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [missing_id, valid])])])
    task = asyncio.create_task(
        layer.get_response(
            [_user()],
            options={"tools": [slow_sibling]},
            client_kwargs={"loop_recorder": recorder},
        )
    )
    await slow_started.wait()
    await asyncio.sleep(0)
    assert not task.done()

    release_slow.set()
    with pytest.raises(KeyError, match="missing call_id"):
        await task

    assert slow_finished is True
    projected = recorder.loop_messages
    assert projected is not None
    assert [content.call_id for content in projected[0].contents if content.type == "function_call"] == ["slow"]
    assert projected[1].contents[0].result == "slow-done"


@pytest.mark.asyncio
async def test_sync_worker_cancel_commits_interrupted_slot_and_keeps_falsy_sibling_alignment() -> None:
    slow_started = ThreadEvent()
    release_slow = ThreadEvent()
    fast_finished = ThreadEvent()

    @tool(name="threaded_batch")
    def threaded_batch(kind: str) -> str:
        if kind == "slow":
            slow_started.set()
            # Generous timeout: an early expiry would let the worker finish
            # while cancellation is in flight, where draining it as a commit
            # is legitimate — and the slot assertions below would flip.
            release_slow.wait(timeout=30)
            return "slow-returned-after-cancel"
        fast_finished.set()
        return "fast-result"

    slow_call = Content.from_function_call("", "threaded_batch", arguments={"kind": "slow"})
    fast_call = Content.from_function_call("", "threaded_batch", arguments={"kind": "fast"})
    recorder = LoopRecorder()
    layer, _wire = _stack([ChatResponse(messages=[Message("assistant", [slow_call, fast_call])])])
    task = asyncio.create_task(
        layer.get_response(
            [_user()],
            options={"tools": [threaded_batch]},
            client_kwargs={"loop_recorder": recorder},
        )
    )
    assert await asyncio.to_thread(slow_started.wait, 2)
    assert await asyncio.to_thread(fast_finished.wait, 2)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # Release only after the run settled: a worker completing while
    # cancellation is still in flight is legitimately drained as a commit,
    # which would make the slow slot's outcome interleaving-dependent.
    release_slow.set()

    projected = recorder.loop_messages
    assert projected is not None
    assert projected[0].contents == [slow_call, fast_call]
    assert len(projected[1].contents) == 2
    slow_metadata = projected[1].contents[0].additional_properties[TOOL_RESULT_METADATA_KEY]
    assert slow_metadata[TOOL_INTERRUPTED_METADATA_KEY] is True
    assert projected[1].contents[1].result == "fast-result"
    assert recorder.committed_count == 1


@pytest.mark.asyncio
async def test_sync_worker_result_drained_on_cancel_commits_raw_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    """A drained completed value reaches the journal as a raw commit before cancellation proceeds."""

    @tool(name="threaded_tool")
    def threaded_tool() -> str:
        return "unused"

    async def _cancelled_after_completion(self: FunctionTool, call_kwargs: Any) -> Any:
        raise SyncToolCancelledAfterCompletion("fast-result")

    monkeypatch.setattr(FunctionTool, "_invoke_function", _cancelled_after_completion)
    recorder = LoopRecorder()
    layer, _wire = _stack([_call_response(("fast", "threaded_tool", {}))])
    with pytest.raises(asyncio.CancelledError):
        await layer.get_response(
            [_user()],
            options={"tools": [threaded_tool]},
            client_kwargs={"loop_recorder": recorder},
        )

    projected = recorder.loop_messages
    assert projected is not None
    result = projected[-1].contents[0]
    metadata = result.additional_properties[TOOL_RESULT_METADATA_KEY]
    assert result.result == "fast-result"
    assert metadata[TOOL_INTERRUPTED_METADATA_KEY] is True
    assert metadata[TOOL_POST_PROCESSING_INTERRUPTED_METADATA_KEY] is True
    assert recorder.committed_count == 1
