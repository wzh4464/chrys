# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for ask_user tool and AskUserMiddleware."""

from __future__ import annotations

import asyncio

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import AskUserResponse, AskUserTimedOut, QuestionToUser
from chrys.foundation.tool_result_metadata import TOOL_ERROR_KIND_METADATA_KEY, TOOL_FAILED_METADATA_KEY
from chrys.kernel import ChatResponse, Content, Message
from chrys.kernel.middleware import ChatMiddlewareLayer
from chrys.service.agent_middleware import AskUserMiddleware
from chrys.service.tools.builtins.ask_user import ask_user
from chrys.service.tools.result_metadata import tool_result_metadata
from tests.support.cpu_guard import cpu_bounded
from tests.support.transcript_invariants import InvariantCheckedToolLoopLayer


class _ScriptedAskUserClient:
    def __init__(self, first_arguments: dict[str, object] | None = None) -> None:
        self.calls = 0
        self.messages: list[list[Message]] = []
        self._first_arguments = (
            first_arguments
            if first_arguments is not None
            else {
                "question": "Pick?",
                "options": [
                    {"label": "One"},
                    {"description": "Two", "item": {"label": "Three"}},
                ],
            }
        )

    def get_response(
        self,
        messages: list[Message],
        *,
        stream: bool = False,
        options: dict[str, object] | None = None,
        **_kwargs: object,
    ) -> ChatResponse:
        self.calls += 1
        self.messages.append(list(messages))
        assert not stream
        assert options is not None
        if self.calls == 1:
            return ChatResponse(
                messages=[
                    Message(
                        role="assistant",
                        contents=[
                            Content.from_function_call(
                                call_id="call-ask",
                                name="ask_user",
                                arguments=self._first_arguments,
                            )
                        ],
                    )
                ]
            )
        return ChatResponse(messages=[Message(role="assistant", contents=["done"])])


@pytest.mark.asyncio
async def test_ask_user_without_middleware() -> None:
    """Without middleware, the tool returns the fallback message."""
    result = await ask_user("What language?")
    assert "[Question for user]" in result
    assert "What language?" in result


def test_ask_user_question_schema_describes_markdown_support() -> None:
    schema = ask_user.parameters()
    description = schema["properties"]["question"]["description"]
    assert "Plain text and Markdown are both supported" in description
    assert "Put the complete question here" in description
    assert "Ask exactly one question in each call" in description
    assert "wait for the response" in description
    assert "do not send the question as intermediate assistant text" in description
    options_schema = schema["properties"]["options"]
    assert options_schema["anyOf"] == [{"items": {"type": "string"}, "type": "array"}, {"type": "null"}]
    assert schema["required"] == ["question"]
    options_description = options_schema["description"]
    assert "flat array of strings" in options_description
    assert "Pass the array directly" in options_description
    assert "do not JSON-encode it" in options_description
    assert "encoded array in one item" in options_description
    assert "Omit when there are no useful choices" in options_description
    tool_description = ask_user.to_json_schema_spec()["function"]["description"]
    assert "use Markdown to structure longer questions" in tool_description
    assert "Always put the complete question in the ``question`` argument" in tool_description
    assert "user's request is to ask them a question" in tool_description
    assert "current task should" in tool_description
    assert "do not leave ``options`` empty" in tool_description
    assert "stop and wait" in tool_description
    assert "final response" in tool_description
    assert "flat list of string" in tool_description
    assert "Do not bundle several independent questions" in tool_description
    assert "ask the most blocking question first" in tool_description
    assert "multiSelect" in tool_description
    assert "objects" in tool_description


def _validated_ask_user_args(arguments: dict[str, object]) -> dict[str, object]:
    assert ask_user.input_model is not None
    return ask_user.input_model.model_validate(arguments).model_dump(exclude_none=True)


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        (["x", "y"], ["x", "y"]),
        (
            [
                {"content": "晚饭吃什么?", "label": "晚饭吃什么"},
                {"content": "今天要不要加点加班?", "label": "今天要不要加班"},
            ],
            ["晚饭吃什么", "今天要不要加班"],
        ),
        ([{"label": "One"}, {"label": "Two"}], ["One", "Two"]),
        (
            [
                {
                    "description": "阅读、修改或重构 Chrys 代码",
                    "item": {
                        "description": "运行 uv run chrys 体验 CLI / TUI / ACP",
                        "item": {
                            "description": "整理 AGENTS.md / README / 文档",
                            "item": {"description": "只是随便聊聊"},
                        },
                    },
                }
            ],
            [
                "阅读、修改或重构 Chrys 代码",
                "运行 uv run chrys 体验 CLI / TUI / ACP",
                "整理 AGENTS.md / README / 文档",
                "只是随便聊聊",
            ],
        ),
        ([{"choice": "A"}], ["A"]),
        ([{}, {"label": "ok"}], ["ok"]),
        ("Yes", ["Yes"]),
        ({"options": [{"label": "A"}, {"label": "B"}]}, ["A", "B"]),
        ('["One", "Two"]', ["One", "Two"]),
        (['["One", "Two"]'], ["One", "Two"]),
        (['["One", {"label": "Two"}]'], ["One", "Two"]),
        (["[not JSON]"], ["[not JSON]"]),
        (["[1, 100]", "[100, 200]"], ["[1, 100]", "[100, 200]"]),
        ([{"label": '["One", "Two"]'}], ["One", "Two"]),
        ([{"note": '["A", "B"]'}], ["A", "B"]),
        ([{"label": "[1, 100]"}], ["[1, 100]"]),
        # json.loads raises plain ValueError (int digit limit), not
        # JSONDecodeError; the literal string must survive as an option.
        (["[" + "1" * 5000 + "]"], ["[" + "1" * 5000 + "]"]),
    ],
)
def test_ask_user_options_normalizes_common_model_shapes(options: object, expected: list[str]) -> None:
    args = _validated_ask_user_args({"question": "Pick?", "options": options})
    assert args["options"] == expected


def test_ask_user_options_normalization_preserves_absent_and_null_options() -> None:
    assert _validated_ask_user_args({"question": "Pick?"}) == {"question": "Pick?"}
    assert _validated_ask_user_args({"question": "Pick?", "options": None}) == {"question": "Pick?"}
    assert _validated_ask_user_args({"question": "Pick?", "options": [{}]}) == {"question": "Pick?"}
    assert _validated_ask_user_args({"question": "Pick?", "options": "[]"}) == {"question": "Pick?"}
    assert _validated_ask_user_args({"question": "Pick?", "options": ["[]"]}) == {"question": "Pick?"}


def test_ask_user_options_normalization_bounds_shared_reference_graphs() -> None:
    # Eight shared levels of seven references each are only eight list
    # objects, but without a total-visit budget the depth-capped walk would
    # traverse ~7^8 paths. The budget must keep validation fast; flattening
    # degrades instead of running away.
    node: object = ["opt"]
    for _ in range(8):
        node = [node] * 7
    args = cpu_bounded(lambda: _validated_ask_user_args({"question": "Pick?", "options": node}))
    options = args.get("options")
    assert options is None or all(option == "opt" for option in options)

    # The budget must also stop ITERATION, not just deeper visits — a flat
    # exact list far past the visit budget must not be walked to the end.
    cpu_bounded(lambda: _validated_ask_user_args({"question": "Pick?", "options": [None] * 100_000}))

    # Container subclasses are never iterated by the flatten itself — their
    # hooks are user code that can raise or block mid-validation. A
    # top-level list subclass passes through to pydantic's C-level list
    # validation; a nested one degrades to a dropped option.
    class BoomList(list):
        def __iter__(self):  # type: ignore[override]
            raise RuntimeError("flatten must not iterate a list subclass")

    args = _validated_ask_user_args({"question": "Pick?", "options": BoomList(["x"])})
    assert args["options"] == ["x"]
    args = _validated_ask_user_args({"question": "Pick?", "options": [BoomList(["x"])]})
    assert "options" not in args

    # ``json.loads`` runs before the visit budget can meter its output, so a
    # giant stringified array must be length-gated up front — the literal
    # text survives as an option instead of materializing a huge decoded
    # array the budget would immediately discard.
    giant = "[" + "0," * 100_000 + "0]"
    args = cpu_bounded(lambda: _validated_ask_user_args({"question": "Pick?", "options": [giant, "keep-me"]}))
    assert args["options"] == [giant, "keep-me"]

    # Under the gate, decode cost is charged to the budget by SIZE: a stack
    # of near-gate arrays exhausts it after a few parses instead of each
    # paying full parse price. Shared references hit the expansion memo, and
    # a memo hit still charges for the options it emits — repeated refs to
    # one decoded array must not multiply its options past the budget.
    near_gate = "[" + ",".join(f'"o{index}"' for index in range(9_000)) + "]"
    args = cpu_bounded(lambda: _validated_ask_user_args({"question": "Pick?", "options": [near_gate] * 5_000}))
    assert len(args.get("options") or []) < 20_000

    # Stripping is O(len) and must be memoized by object identity: a small
    # list of many references to one huge whitespace string would otherwise
    # pay a full strip pass per reference (round 26: 0.67s -> ~1ms).
    shared_whitespace = " " * 2_000_000
    args = cpu_bounded(lambda: _validated_ask_user_args({"question": "Pick?", "options": [shared_whitespace] * 5_000}))
    assert "options" not in args
    args = cpu_bounded(
        lambda: _validated_ask_user_args(
            {"question": "Pick?", "options": [{"label": shared_whitespace, "zzz": shared_whitespace}] * 1_000}
        )
    )
    assert "options" not in args

    # The memos key by id() but must RETAIN their source objects: a freed
    # temporary (a decoded array element) can otherwise hand its reused
    # address — and its stale cached expansion — to a later, distinct string
    # (round 27: alternating decodes left arrays un-expanded).
    aliasing_options = []
    for index in range(20):
        aliasing_options.append('["          "]')
        aliasing_options.append(f'["value-{index}"]')
    args = _validated_ask_user_args({"question": "Pick?", "options": aliasing_options})
    assert [option for option in args["options"] if option.startswith("value-")] == [
        f"value-{index}" for index in range(20)
    ]
    assert not any(option.startswith('["value-') for option in args["options"])


@pytest.mark.parametrize(
    ("first_arguments", "expected_options"),
    [
        (
            {
                "question": "Pick?",
                "options": [
                    {"label": "One"},
                    {"description": "Two", "item": {"label": "Three"}},
                ],
            },
            ["One", "Two", "Three"],
        ),
        (
            {"question": "Pick?", "options": ['["One", "Two"]']},
            ["One", "Two"],
        ),
    ],
)
@pytest.mark.asyncio
async def test_ask_user_loop_validation_passes_coerced_options_to_middleware(
    first_arguments: dict[str, object],
    expected_options: list[str],
) -> None:
    bus = EventBus()
    questions: list[QuestionToUser] = []

    async def reply(event: QuestionToUser) -> None:
        questions.append(event)
        await bus.publish(AskUserResponse(request_id=event.request_id, text="One"))

    await bus.subscribe(QuestionToUser, reply)

    wire = _ScriptedAskUserClient(first_arguments)
    client = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))
    response = await client.get_response(
        [Message(role="user", contents=["go"])],
        options={"tools": [ask_user]},
        middleware=[AskUserMiddleware(bus)],
    )

    assert response.text == "done"
    assert wire.calls == 2
    assert len(questions) == 1
    assert questions[0].question == "Pick?"
    assert questions[0].options == expected_options


@pytest.mark.asyncio
async def test_ask_user_invalid_options_shape_returns_model_actionable_schema_guidance() -> None:
    wire = _ScriptedAskUserClient({"question": "Pick?", "options": 7})
    client = InvariantCheckedToolLoopLayer(ChatMiddlewareLayer(wire))

    response = await client.get_response(
        [Message(role="user", contents=["go"])],
        options={"tools": [ask_user]},
    )

    assert response.text == "done"
    assert wire.calls == 2
    results = [
        content for message in wire.messages[1] for content in message.contents if content.type == "function_result"
    ]
    assert len(results) == 1
    result_text = str(results[0].result)
    assert result_text == (
        "Error: Invalid arguments for 'ask_user': options: expected [string] | null, got integer. "
        "Expected: {question: string, options?: [string] | null}."
    )
    assert len(result_text) < 160
    assert results[0].additional_properties[TOOL_ERROR_KIND_METADATA_KEY] == "argument_parsing"


@pytest.mark.asyncio
async def test_ask_user_middleware_publishes_and_waits() -> None:
    """AskUserMiddleware publishes QuestionToUser and waits for AskUserResponse."""
    from unittest.mock import AsyncMock, MagicMock

    from chrys.service.agent_middleware import AskUserMiddleware

    bus = EventBus()
    mw = AskUserMiddleware(event_bus=bus, session_id="test")

    questions: list[QuestionToUser] = []

    async def capture_and_reply(event: QuestionToUser) -> None:
        questions.append(event)
        await bus.publish(AskUserResponse(request_id=event.request_id, text="Python"))

    await bus.subscribe(QuestionToUser, capture_and_reply)

    # Build a mock FunctionInvocationContext
    mock_function = MagicMock()
    mock_function.chrys_kind = "ask_user"
    mock_function.name = "ask_user"

    context = MagicMock()
    context.function = mock_function
    context.arguments = {"question": "What language?", "options": ["Python", "Go"]}
    context.metadata = {}
    context.result = None
    from chrys.service.agent_middleware.events.hook_dispatch import set_call_id

    set_call_id(context, "call-123")

    call_next = AsyncMock()
    await mw.process(context, call_next)

    assert "Python" in context.result
    assert len(questions) == 1
    assert questions[0].question == "What language?"
    assert questions[0].options == ["Python", "Go"]
    assert questions[0].request_id  # should be set
    assert questions[0].call_id == "call-123"
    call_next.assert_not_called()  # middleware handles entirely, never calls through


@pytest.mark.asyncio
async def test_ask_user_middleware_passes_through_non_ask_user() -> None:
    """Non-ask_user tools are passed through to call_next."""
    from unittest.mock import AsyncMock, MagicMock

    from chrys.service.agent_middleware import AskUserMiddleware

    bus = EventBus()
    mw = AskUserMiddleware(event_bus=bus, session_id="test")

    mock_function = MagicMock()
    mock_function.chrys_kind = "shell"

    context = MagicMock()
    context.function = mock_function

    call_next = AsyncMock()
    await mw.process(context, call_next)

    call_next.assert_called_once()


@pytest.mark.asyncio
async def test_ask_user_middleware_ignores_mismatched_request_id() -> None:
    """Responses with wrong request_id are ignored."""
    from unittest.mock import AsyncMock, MagicMock

    from chrys.service.agent_middleware import AskUserMiddleware

    bus = EventBus()
    mw = AskUserMiddleware(event_bus=bus, session_id="test")

    async def reply_wrong_then_right(event: QuestionToUser) -> None:
        # ``bus.publish`` awaits each subscriber inline and in order, so the
        # wrong-id response is fully delivered to (and rejected by) the
        # middleware before the next publish runs — no settle sleep needed.
        await bus.publish(AskUserResponse(request_id="wrong_id", text="bad"))
        await bus.publish(AskUserResponse(request_id=event.request_id, text="good"))

    await bus.subscribe(QuestionToUser, reply_wrong_then_right)

    mock_function = MagicMock()
    mock_function.chrys_kind = "ask_user"
    mock_function.name = "ask_user"

    context = MagicMock()
    context.function = mock_function
    context.arguments = {"question": "Pick?"}
    context.result = None

    await mw.process(context, AsyncMock())
    assert "good" in context.result


@pytest.mark.asyncio
async def test_ask_user_middleware_timeout_defaults_to_ten_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ask_user response wait defaults to 10 minutes without sleeping in the test."""
    from unittest.mock import AsyncMock, MagicMock

    import chrys.service.agent_middleware.control.ask_user as ask_user_middleware_module
    from chrys.service.agent_middleware import AskUserMiddleware

    bus = EventBus()
    mw = AskUserMiddleware(event_bus=bus, session_id="test")
    seen_timeouts: list[float] = []
    questions: list[QuestionToUser] = []
    timed_out: list[AskUserTimedOut] = []

    async def collect_question(event: QuestionToUser) -> None:
        questions.append(event)

    async def collect_timeout(event: AskUserTimedOut) -> None:
        timed_out.append(event)

    await bus.subscribe(QuestionToUser, collect_question)
    await bus.subscribe(AskUserTimedOut, collect_timeout)

    async def fake_wait_for(_future: asyncio.Future[str], *, timeout: float) -> str:
        seen_timeouts.append(timeout)
        raise TimeoutError

    monkeypatch.setattr(ask_user_middleware_module.asyncio, "wait_for", fake_wait_for)

    mock_function = MagicMock()
    mock_function.chrys_kind = "ask_user"
    mock_function.name = "ask_user"

    context = MagicMock()
    context.function = mock_function
    context.arguments = {"question": "Pick?"}
    context.result = None

    metadata: dict[str, object] = {}
    token = tool_result_metadata.set(metadata)
    try:
        await mw.process(context, AsyncMock())
    finally:
        tool_result_metadata.reset(token)

    assert seen_timeouts == [600]
    assert context.result == "Error: user did not respond within 10 minutes."
    assert metadata[TOOL_FAILED_METADATA_KEY] is True
    assert metadata[TOOL_ERROR_KIND_METADATA_KEY] == "ask_user_timeout"
    assert len(timed_out) == 1
    assert timed_out[0].request_id == questions[0].request_id
    assert timed_out[0].session_id == "test"


@pytest.mark.asyncio
async def test_ask_user_middleware_custom_timeout_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-default timeout is honored and reported in seconds when not whole minutes."""
    from unittest.mock import AsyncMock, MagicMock

    import chrys.service.agent_middleware.control.ask_user as ask_user_middleware_module
    from chrys.service.agent_middleware import AskUserMiddleware

    bus = EventBus()
    mw = AskUserMiddleware(event_bus=bus, session_id="test", timeout_seconds=90)
    seen_timeouts: list[float] = []

    async def fake_wait_for(_future: asyncio.Future[str], *, timeout: float) -> str:
        seen_timeouts.append(timeout)
        raise TimeoutError

    monkeypatch.setattr(ask_user_middleware_module.asyncio, "wait_for", fake_wait_for)

    mock_function = MagicMock()
    mock_function.chrys_kind = "ask_user"
    mock_function.name = "ask_user"
    context = MagicMock()
    context.function = mock_function
    context.arguments = {"question": "Pick?"}
    context.result = None

    await mw.process(context, AsyncMock())

    assert seen_timeouts == [90]
    assert context.result == "Error: user did not respond within 90 seconds."


@pytest.mark.asyncio
async def test_ask_user_middleware_no_timeout_waits_indefinitely(monkeypatch: pytest.MonkeyPatch) -> None:
    """timeout_seconds=None waits on the response without arming asyncio.wait_for."""
    from unittest.mock import AsyncMock, MagicMock

    import chrys.service.agent_middleware.control.ask_user as ask_user_middleware_module
    from chrys.service.agent_middleware import AskUserMiddleware

    bus = EventBus()
    mw = AskUserMiddleware(event_bus=bus, session_id="test", timeout_seconds=None)
    timed_out: list[AskUserTimedOut] = []

    async def reply(event: QuestionToUser) -> None:
        await bus.publish(AskUserResponse(request_id=event.request_id, text="later"))

    async def collect_timeout(event: AskUserTimedOut) -> None:
        timed_out.append(event)

    async def boom_wait_for(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("wait_for must not be used when timeout is disabled")

    await bus.subscribe(QuestionToUser, reply)
    await bus.subscribe(AskUserTimedOut, collect_timeout)
    monkeypatch.setattr(ask_user_middleware_module.asyncio, "wait_for", boom_wait_for)

    mock_function = MagicMock()
    mock_function.chrys_kind = "ask_user"
    mock_function.name = "ask_user"
    context = MagicMock()
    context.function = mock_function
    context.arguments = {"question": "Pick?"}
    context.result = None

    await mw.process(context, AsyncMock())

    assert context.result == "User response: later"
    assert timed_out == []
