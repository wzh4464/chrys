# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the response-validation middleware + built-in validators.

Covers the three malformed-response shapes the user explicitly called out:

1. ``contents == []`` — provider returned a message with no content items.
2. Whitespace-only text — ``text=""`` or ``text="\\n\\n\\n"``.
3. Leaked ``minimax:tool_call ... </minimax:tool_call>`` markers in text
   (with text before, after, or surrounding the marker).

Each scenario is exercised in **both** streaming and non-streaming modes,
and the retry-exhaustion path (5 retries → return the bad response with
``function_call`` contents stripped on the final attempt) is asserted
explicitly so we know the middleware never raises on bad content and
never lets an exhausted-but-still-malformed function call slip through
to the Chrys tool loop.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from chrys.foundation.hosted_tools import HostedRetrySafety, HostedToolFamily, HostedToolPhase
from chrys.foundation.retry import RetryAttemptInfo
from chrys.foundation.trajectory.context import TRAJECTORY_EXCHANGE_KWARG, ExchangeTrace
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.kernel import (
    ChatResponse,
    ChatResponseUpdate,
    Content,
    Message,
    ResponseStream,
)
from chrys.kernel.middleware import ChatContext
from chrys.service.agent_middleware.response_validation import (
    MAX_RETRIES,
    ResponseValidationMiddleware,
    RetryableResponseValidationError,
    TerminalResponseValidationError,
    ValidationRetryExemption,
    hosted_commits_from_error,
)
from chrys.service.agent_middleware.validators import (
    HOSTED_EVIDENCE_MISSING_FINAL_TEXT_REASON,
    NO_VISIBLE_OUTPUT_REASON,
    OUTPUT_TRUNCATED_REASON,
    REASONING_EXHAUSTED_OUTPUT_REASON,
    DefaultResponseValidator,
    ValidationResult,
)
from chrys.service.context.middleware.usage import UsageTrackingMiddleware
from tests.service.trajectory._fakes import FakeSink, make_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assistant(contents: list[Any]) -> ChatResponse:
    """Build a minimal ChatResponse with a single assistant message."""
    return ChatResponse(messages=[Message(role="assistant", contents=contents)], finish_reason="stop")


def _assistant_truncated(contents: list[Any]) -> ChatResponse:
    """Assistant response that hit the output token limit (finish_reason='length')."""
    return ChatResponse(messages=[Message(role="assistant", contents=contents)], finish_reason="length")


# The three distinct invalid shapes the default validator catches.  Used
# to construct exhaustion sequences that walk through different
# ``ValidationResult.reason`` strings on consecutive attempts so the
# middleware's fail-fast (identical-reason short-circuit) does not fire
# and the loop actually reaches MAX_RETRIES.
def _bad_empty() -> ChatResponse:
    return _assistant([])  # reason: "empty contents"


def _bad_whitespace() -> ChatResponse:
    return _assistant([Content.from_text("\n\n  \t\n")])  # reason: "empty or whitespace-only text response"


def _bad_leaked() -> ChatResponse:
    return _assistant(
        [Content.from_text("minimax:tool_call {} </minimax:tool_call>")]
    )  # reason: "leaked tool-call marker..."


def _search_without_final_text(*, intermediate_text: str = "Checking sources.") -> ChatResponse:
    return _assistant(
        [
            Content.from_text(intermediate_text),
            Content.from_search_tool_call(
                "ws_1",
                tool_name="web_search",
                status="completed",
                provider_phase=HostedToolPhase.TERMINAL,
                provider_status="completed",
            ),
            Content.from_search_tool_result(
                "ws_1",
                tool_name="web_search",
                status="completed",
                provider_phase=HostedToolPhase.TERMINAL,
                provider_status="completed",
                result={"query": "Chrys"},
            ),
        ]
    )


def _varied_bads(n: int, *, last: ChatResponse | None = None) -> list[ChatResponse]:
    """Build n bad responses cycling through distinct reasons.

    Each consecutive pair has a different ``ValidationResult.reason`` so
    fail-fast does not short-circuit the retry loop.  When ``last`` is
    given, it overrides the final response — useful for tests that need
    the give-up path to scrub a specific shape (e.g. drop an empty
    assistant message).
    """
    factories = [_bad_empty, _bad_whitespace, _bad_leaked]
    out = [factories[i % len(factories)]() for i in range(n)]
    if last is not None and out:
        out[-1] = last
    return out


def _make_context(stream: bool = False) -> ChatContext:
    return ChatContext(
        client=None,  # type: ignore[arg-type]
        messages=[Message(role="user", contents=[Content.from_text("hi")])],
        options=None,
        stream=stream,
    )


def _response_stream_from(response: ChatResponse) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
    """Build a fake ResponseStream that yields one update per content and finalizes to *response*."""
    msg = response.messages[-1] if response.messages else Message(role="assistant", contents=[])
    updates = [ChatResponseUpdate(contents=msg.contents or [], role="assistant")]

    async def _gen() -> AsyncIterator[ChatResponseUpdate]:
        for u in updates:
            yield u

    return ResponseStream(_gen(), finalizer=lambda _u: response)


def _semantic_updates(updates: list[ChatResponseUpdate]) -> list[ChatResponseUpdate]:
    """Exclude raw-only transport heartbeats from response-content assertions."""
    return [
        update
        for update in updates
        if update.contents
        or update.role is not None
        or update.author_name is not None
        or update.response_id is not None
        or update.message_id is not None
        or update.conversation_id is not None
        or update.model is not None
        or update.created_at is not None
        or update.finish_reason is not None
        or update.continuation_token is not None
        or update.additional_properties
    ]


class _FakeCallNext:
    """Simulates a middleware chain where each ``call_next()`` sets ``context.result``
    to the next scripted response — either a ``ChatResponse`` (non-stream) or
    a fresh ``ResponseStream`` (stream).
    """

    def __init__(self, responses: list[ChatResponse], *, stream: bool) -> None:
        self._responses = list(responses)
        self._stream = stream
        self._call_count = 0
        self.context: ChatContext | None = None

    @property
    def call_count(self) -> int:
        return self._call_count

    def bind(self, context: ChatContext) -> None:
        self.context = context

    async def __call__(self) -> None:
        assert self.context is not None
        idx = min(self._call_count, len(self._responses) - 1)
        resp = self._responses[idx]
        self._call_count += 1
        if self._stream:
            self.context.result = _response_stream_from(resp)
        else:
            self.context.result = resp


class _ObservationHook:
    """Records the optional validation observation contract in call order."""

    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    async def begin_response(self, *, response_index: int | None = None, batch_id: int | None = None) -> None:
        self.events.append(("response", response_index, batch_id))

    async def observe_contents(self, contents: list[Any], *, is_final: bool = False) -> None:
        self.events.append(("contents", is_final, tuple(content.type for content in contents)))

    async def attempt_started(self, *, continuation: bool = False) -> None:
        self.events.append(("started", continuation))

    async def attempt_rejected(self, reason: str = "") -> None:
        self.events.append(("rejected", reason))

    async def attempt_accepted(self, messages: Sequence[Message]) -> None:
        self.events.append(
            (
                "accepted",
                tuple(tuple(content.type for content in message.contents) for message in messages),
            )
        )


# ---------------------------------------------------------------------------
# DefaultResponseValidator — rule-level tests
# ---------------------------------------------------------------------------


class TestDefaultValidatorRules:
    def test_empty_contents_is_invalid(self) -> None:
        v = DefaultResponseValidator()
        result = v.validate(_assistant([]))
        assert result == ValidationResult.invalid("empty contents")

    def test_text_only_empty_string_is_invalid(self) -> None:
        v = DefaultResponseValidator()
        result = v.validate(_assistant([Content.from_text("")]))
        assert not result.ok
        assert "empty or whitespace" in result.reason

    def test_text_only_whitespace_newlines_is_invalid(self) -> None:
        """contents=[{type=text, text='\\n\\n\\n'}] must be flagged."""
        v = DefaultResponseValidator()
        result = v.validate(_assistant([Content.from_text("\n\n\n")]))
        assert not result.ok
        assert "whitespace" in result.reason

    def test_text_only_spaces_and_tabs_is_invalid(self) -> None:
        v = DefaultResponseValidator()
        result = v.validate(_assistant([Content.from_text("   \t  \n ")]))
        assert not result.ok

    def test_real_text_is_valid(self) -> None:
        v = DefaultResponseValidator()
        result = v.validate(_assistant([Content.from_text("Here is the answer.")]))
        assert result.ok

    def test_tool_call_only_is_valid(self) -> None:
        """A pure tool-calling turn with NO text is perfectly legal."""
        v = DefaultResponseValidator()
        tool_call = Content.from_function_call("call_1", "read_file", arguments={"path": "x"})
        result = v.validate(_assistant([tool_call]))
        assert result.ok

    def test_tool_call_with_empty_text_is_valid(self) -> None:
        """Don't flag empty text when the message has a tool call."""
        v = DefaultResponseValidator()
        tool_call = Content.from_function_call("call_1", "read_file", arguments={"path": "x"})
        result = v.validate(_assistant([Content.from_text(""), tool_call]))
        assert result.ok

    def test_informational_custom_tool_call_only_is_valid(self) -> None:
        """A preserved non-executable Responses custom call is still output."""
        call = Content.from_function_call(
            "call_custom_1",
            "python",
            arguments="print('hi')",
            informational_only=True,
            additional_properties={"item_type": "custom_tool_call"},
        )

        assert DefaultResponseValidator().validate(_assistant([call])).ok

    def test_terminal_hosted_only_response_is_valid(self) -> None:
        """A provider-hosted terminal item is usable output without text."""
        v = DefaultResponseValidator()
        hosted = Content.from_hosted_tool_result(
            "hosted_1",
            tool_name="server_task",
            status="completed",
            provider_phase=HostedToolPhase.TERMINAL,
            provider_status="completed",
            result="done",
        )

        assert v.validate(_assistant([hosted])).ok

    def test_running_hosted_only_response_is_not_terminal_output(self) -> None:
        v = DefaultResponseValidator()
        hosted = Content.from_hosted_tool_call(
            "hosted_1",
            tool_name="server_task",
            status="running",
            provider_phase=HostedToolPhase.START,
            provider_status="running",
        )

        result = v.validate(_assistant([hosted]))

        assert not result.ok

    def test_terminal_search_after_intermediate_text_requires_final_answer(self) -> None:
        result = DefaultResponseValidator().validate(_search_without_final_text())

        assert result == ValidationResult.invalid(
            HOSTED_EVIDENCE_MISSING_FINAL_TEXT_REASON,
            terminal_on_giveup=True,
        )

    @pytest.mark.parametrize(
        "family",
        [HostedToolFamily.SEARCH, HostedToolFamily.FETCH, HostedToolFamily.TOOL_DISCOVERY],
    )
    @pytest.mark.parametrize("content_type", ["call", "result"])
    def test_terminal_evidence_only_hosted_content_requires_final_answer(
        self,
        family: HostedToolFamily,
        content_type: str,
    ) -> None:
        if content_type == "call":
            content = Content.from_hosted_tool_call(
                "evidence_1",
                tool_name="provider_tool",
                hosted_family=family,
                status="completed",
                provider_phase=HostedToolPhase.TERMINAL,
                provider_status="completed",
            )
        else:
            content = Content.from_hosted_tool_result(
                "evidence_1",
                tool_name="provider_tool",
                hosted_family=family,
                status="completed",
                provider_phase=HostedToolPhase.TERMINAL,
                provider_status="completed",
                result={"evidence": "found"},
            )

        assert DefaultResponseValidator().validate(_assistant([content])) == ValidationResult.invalid(
            HOSTED_EVIDENCE_MISSING_FINAL_TEXT_REASON,
            terminal_on_giveup=True,
        )

    def test_terminal_tool_discovery_followed_by_final_text_is_valid(self) -> None:
        discovery = Content.from_hosted_tool_result(
            "discovery_1",
            tool_name="tool_search",
            hosted_family=HostedToolFamily.TOOL_DISCOVERY,
            status="completed",
            provider_phase=HostedToolPhase.TERMINAL,
            provider_status="completed",
            result={"tools": [{"name": "get_weather"}]},
        )

        assert (
            DefaultResponseValidator()
            .validate(_assistant([discovery, Content.from_text("I found the appropriate tool.")]))
            .ok
        )

    def test_terminal_search_followed_by_final_text_is_valid(self) -> None:
        response = _search_without_final_text()
        response.messages[0].contents.append(Content.from_text("Here is the answer."))

        assert DefaultResponseValidator().validate(response).ok

    def test_terminal_search_followed_by_image_result_is_valid(self) -> None:
        response = _search_without_final_text()
        response.messages[0].contents.append(
            Content.from_image_generation_tool_result(
                image_id="image_1",
                outputs=["data:image/png;base64,AA=="],
                provider_phase=HostedToolPhase.TERMINAL,
                provider_status="completed",
            )
        )

        assert DefaultResponseValidator().validate(response).ok

    def test_local_function_call_before_terminal_search_is_valid(self) -> None:
        response = _search_without_final_text()
        response.messages[0].contents[0] = Content.from_function_call(
            "call_1", "read_file", arguments={"path": "README.md"}
        )

        assert DefaultResponseValidator().validate(response).ok

    # -- Leaked tool-call markers -----------------------------------------

    def test_leaked_minimax_tool_call_plain(self) -> None:
        v = DefaultResponseValidator()
        text = 'minimax:tool_call {"name":"read_file"} </minimax:tool_call>'
        result = v.validate(_assistant([Content.from_text(text)]))
        assert not result.ok
        assert "leaked tool-call" in result.reason

    def test_leaked_minimax_tool_call_with_prefix_text(self) -> None:
        """Prefix text should not hide the leak."""
        v = DefaultResponseValidator()
        text = "Sure, I will check the file.\nminimax:tool_call {} </minimax:tool_call>"
        result = v.validate(_assistant([Content.from_text(text)]))
        assert not result.ok

    def test_leaked_minimax_tool_call_with_suffix_text(self) -> None:
        """Suffix text should not hide the leak."""
        v = DefaultResponseValidator()
        text = "minimax:tool_call {} </minimax:tool_call>\nI will do that now."
        result = v.validate(_assistant([Content.from_text(text)]))
        assert not result.ok

    def test_leaked_minimax_tool_call_surrounded(self) -> None:
        v = DefaultResponseValidator()
        text = "Here is my plan.\nminimax:tool_call {} </minimax:tool_call>\nDone."
        result = v.validate(_assistant([Content.from_text(text)]))
        assert not result.ok

    def test_leaked_tool_use_tag(self) -> None:
        v = DefaultResponseValidator()
        text = "<tool_use name='read_file'>...</tool_use>"
        result = v.validate(_assistant([Content.from_text(text)]))
        assert not result.ok

    def test_leaked_function_call_tag(self) -> None:
        v = DefaultResponseValidator()
        text = "<function_call>...</function_call>"
        result = v.validate(_assistant([Content.from_text(text)]))
        assert not result.ok

    def test_word_minimax_in_prose_is_not_flagged(self) -> None:
        """Only the tool-call token patterns are flagged — prose is fine."""
        v = DefaultResponseValidator()
        text = "The MiniMax model was released in 2024."
        result = v.validate(_assistant([Content.from_text(text)]))
        assert result.ok

    def test_empty_response_no_messages(self) -> None:
        """ChatResponse with no messages at all."""
        v = DefaultResponseValidator()
        result = v.validate(ChatResponse(messages=[]))
        assert not result.ok

    def test_length_finish_reason_empty_contents_is_terminal(self) -> None:
        # Empty output + finish_reason="length" → no room to reply; terminal.
        v = DefaultResponseValidator()
        result = v.validate(_assistant_truncated([]))
        assert result == ValidationResult.invalid(OUTPUT_TRUNCATED_REASON, retryable=False)
        assert result.retryable is False

    def test_length_finish_reason_whitespace_text_is_terminal(self) -> None:
        v = DefaultResponseValidator()
        result = v.validate(_assistant_truncated([Content.from_text("\n\n  \t")]))
        assert not result.ok
        assert result.retryable is False
        assert result.reason == OUTPUT_TRUNCATED_REASON

    def test_length_finish_reason_with_real_text_is_valid(self) -> None:
        # A response truncated mid-sentence that still has content is valid —
        # "length" alone must not fabricate a failure (no false positive).
        v = DefaultResponseValidator()
        result = v.validate(_assistant_truncated([Content.from_text("partial answer")]))
        assert result.ok

    def test_length_finish_reason_with_reasoning_text_is_retryable(self) -> None:
        # Reasoning-only output cut off by the output budget means the model
        # over-thought — the generated reasoning proves the input did NOT fill
        # the context window, so a re-roll plausibly recovers.
        # Give-up must still fail through the Error path (nothing visible to
        # return), hence terminal_on_giveup.
        v = DefaultResponseValidator()
        result = v.validate(_assistant_truncated([Content.from_text_reasoning(text="partial thought")]))
        assert result == ValidationResult.invalid(REASONING_EXHAUSTED_OUTPUT_REASON, terminal_on_giveup=True)
        assert result.retryable is True

    def test_length_finish_reason_with_protected_reasoning_is_retryable(self) -> None:
        # Signed/private reasoning is generated content too: the length cutoff
        # came from over-thinking, not a full context window.
        v = DefaultResponseValidator()
        result = v.validate(_assistant_truncated([Content.from_text_reasoning(protected_data="sig-123")]))
        assert result == ValidationResult.invalid(REASONING_EXHAUSTED_OUTPUT_REASON, terminal_on_giveup=True)

    def test_stop_finish_reason_with_reasoning_text_is_retryable(self) -> None:
        # Reasoning-only stop has no user-visible answer, but a fresh sample
        # plausibly produces one — retry instead of failing on first sight.
        # Exhausted retries still raise (terminal_on_giveup)
        # rather than allowing a blank final AgentMessage.
        v = DefaultResponseValidator()
        result = v.validate(_assistant([Content.from_text_reasoning(text="complete thought")]))
        assert result == ValidationResult.invalid(NO_VISIBLE_OUTPUT_REASON, terminal_on_giveup=True)
        assert result.retryable is True

    def test_stop_finish_reason_with_protected_reasoning_is_retryable(self) -> None:
        v = DefaultResponseValidator()
        result = v.validate(_assistant([Content.from_text_reasoning(protected_data="sig-123")]))
        assert result == ValidationResult.invalid(NO_VISIBLE_OUTPUT_REASON, terminal_on_giveup=True)

    def test_reasoning_with_visible_text_is_valid(self) -> None:
        v = DefaultResponseValidator()
        result = v.validate(
            _assistant(
                [
                    Content.from_text_reasoning(text="complete thought"),
                    Content.from_text("visible answer"),
                ]
            )
        )
        assert result.ok


# ---------------------------------------------------------------------------
# Middleware — non-streaming retries
# ---------------------------------------------------------------------------


class TestMiddlewareNonStreaming:
    @pytest.mark.asyncio
    async def test_valid_first_try_no_retry(self) -> None:
        good = _assistant([Content.from_text("all good")])
        fake = _FakeCallNext([good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, fake)

        assert fake.call_count == 1
        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages[0].contents[0].text == "all good"

    @pytest.mark.asyncio
    async def test_empty_contents_retried_then_valid(self) -> None:
        bad = _assistant([])
        good = _assistant([Content.from_text("recovered")])
        fake = _FakeCallNext([bad, good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[RetryAttemptInfo] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 2
        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages[0].contents[0].text == "recovered"
        assert len(retries) == 1
        assert "empty contents" in retries[0].reason

    @pytest.mark.asyncio
    async def test_search_without_final_text_retried_then_valid(self) -> None:
        bad = _search_without_final_text()
        good = _assistant([Content.from_text("recovered answer")])
        fake = _FakeCallNext([bad, good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[RetryAttemptInfo] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 2
        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages[0].contents[0].text == "recovered answer"
        assert [retry.reason for retry in retries] == [HOSTED_EVIDENCE_MISSING_FINAL_TEXT_REASON]

    @pytest.mark.asyncio
    async def test_truncated_empty_response_raises_terminal_without_retry(self) -> None:
        # finish_reason="length" + empty output is terminal: give up on the first
        # attempt (no retry, no backoff) by raising into the executor error path.
        # This avoids the retry channel and avoids returning an empty assistant
        # response that the UI would render as a blank Code Agent block.
        truncated = _assistant_truncated([])
        truncated.usage_details = {"input_token_count": 11, "total_token_count": 11}
        fake = _FakeCallNext([truncated], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[RetryAttemptInfo] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info)

        mw = ResponseValidationMiddleware(
            publish_retry=on_retry,
            backoff_schedule=[0.0],
        )
        with pytest.raises(TerminalResponseValidationError, match="output token limit") as exc_info:
            await mw.process(ctx, fake)

        assert fake.call_count == 1, "terminal failure must not retry"
        assert retries == [], "terminal failure must not use the retry channel"
        assert exc_info.value.usage_details == {"input_token_count": 11, "total_token_count": 11}

    @pytest.mark.asyncio
    async def test_reasoning_only_response_retries_then_succeeds(self) -> None:
        # A reasoning-only final response is a transient over-thinking
        # failure — retry it like the other malformed shapes instead of
        # terminating the run on first sight.
        reasoning_only = _assistant([Content.from_text_reasoning(text="private thought")])
        good = _assistant([Content.from_text("recovered answer")])
        fake = _FakeCallNext([reasoning_only, good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[RetryAttemptInfo] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 2
        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages[0].contents[0].text == "recovered answer"
        assert [r.reason for r in retries] == [NO_VISIBLE_OUTPUT_REASON]

    @pytest.mark.asyncio
    async def test_reasoning_only_repeat_failure_raises_terminal_after_retry(self) -> None:
        # Two identical reasoning-only failures trip the deterministic-stuck
        # short-circuit; the give-up must RAISE (terminal_on_giveup) instead of
        # returning a blank response the caller would treat as a success.
        reasoning_only = _assistant([Content.from_text_reasoning(text="private thought")])
        fake = _FakeCallNext([reasoning_only, reasoning_only], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[RetryAttemptInfo] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        with pytest.raises(TerminalResponseValidationError, match="no visible answer"):
            await mw.process(ctx, fake)

        assert fake.call_count == 2, "reasoning-only must retry at least once before giving up"
        assert len(retries) == 1

    @pytest.mark.asyncio
    async def test_reasoning_only_exhaustion_raises_terminal(self) -> None:
        # Alternating stop/length reasoning-only responses carry distinct
        # reasons, so the short-circuit never fires and the loop runs the full
        # MAX_RETRIES retries before the terminal give-up raise.
        stop_flavor = _assistant([Content.from_text_reasoning(text="thinking")])
        length_flavor = _assistant_truncated([Content.from_text_reasoning(text="thinking harder")])
        responses = [stop_flavor if i % 2 == 0 else length_flavor for i in range(MAX_RETRIES + 1)]
        fake = _FakeCallNext(responses, stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[RetryAttemptInfo] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        with pytest.raises(TerminalResponseValidationError):
            await mw.process(ctx, fake)

        assert fake.call_count == MAX_RETRIES + 1
        assert len(retries) == MAX_RETRIES
        assert retries[0].reason == NO_VISIBLE_OUTPUT_REASON
        assert retries[1].reason == REASONING_EXHAUSTED_OUTPUT_REASON

    @pytest.mark.asyncio
    async def test_whitespace_text_retried(self) -> None:
        bad = _assistant([Content.from_text("\n\n\n")])
        good = _assistant([Content.from_text("OK")])
        fake = _FakeCallNext([bad, good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 2
        assert ctx.result.messages[0].contents[0].text == "OK"

    @pytest.mark.asyncio
    async def test_leaked_minimax_tool_call_retried(self) -> None:
        bad = _assistant([Content.from_text("Here goes:\nminimax:tool_call {} </minimax:tool_call>\nend")])
        good = _assistant([Content.from_text("clean reply")])
        fake = _FakeCallNext([bad, good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 2
        assert ctx.result.messages[0].contents[0].text == "clean reply"

    @pytest.mark.asyncio
    async def test_exhaustion_drops_empty_assistant_message(self) -> None:
        """After MAX_RETRIES retries of (varied) bad responses, the bad
        assistant message is dropped on the give-up path so it does not
        land in persisted history next to the engine's turn marker.

        Uses ``_varied_bads`` so consecutive attempts have distinct
        validator reasons — fail-fast does not short-circuit and the
        loop reaches MAX_RETRIES exhaustion as intended.  ``last``
        forces the final attempt's response to be empty so the give-up
        cleanup demonstrably drops it.
        """
        responses = _varied_bads(MAX_RETRIES + 1, last=_bad_empty())
        fake = _FakeCallNext(responses, stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[str] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info.reason)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        # Must NOT raise — per spec, on exhaustion we return without re-raising.
        await mw.process(ctx, fake)

        # One initial + MAX_RETRIES retries = MAX_RETRIES + 1 total call_next invocations.
        assert fake.call_count == MAX_RETRIES + 1
        # Retry events fire only on the retries themselves, not on the final attempt.
        assert len(retries) == MAX_RETRIES
        # The give-up path scrubs the empty assistant message so the
        # framework's after_run never appends it to history.
        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages == []

    @pytest.mark.asyncio
    async def test_publish_retry_receives_correct_metadata(self) -> None:
        # Distinct reasons on each bad attempt so fail-fast does not
        # short-circuit the loop before we observe two retry events.
        good = _assistant([Content.from_text("ok")])
        fake = _FakeCallNext([_bad_empty(), _bad_whitespace(), good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        recorded: list[RetryAttemptInfo] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            recorded.append(info)

        mw = ResponseValidationMiddleware(
            publish_retry=on_retry,
            max_retries=3,
            backoff_schedule=[0.0, 0.0, 0.0],
        )
        await mw.process(ctx, fake)

        assert [r.attempt for r in recorded] == [1, 2]  # attempt numbers are 1-based
        assert all(r.max_attempts == 3 for r in recorded)

    @pytest.mark.asyncio
    async def test_publish_retry_exception_does_not_break_loop(self) -> None:
        """A raising publish_retry must not prevent retry from happening."""
        bad = _assistant([])
        good = _assistant([Content.from_text("ok")])
        fake = _FakeCallNext([bad, good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        async def bad_publish(_info: RetryAttemptInfo) -> None:
            raise RuntimeError("telemetry broke")

        mw = ResponseValidationMiddleware(publish_retry=bad_publish, backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 2
        assert ctx.result.messages[0].contents[0].text == "ok"


# ---------------------------------------------------------------------------
# Middleware — streaming retries (consume, replay, retry)
# ---------------------------------------------------------------------------


class TestMiddlewareStreaming:
    @pytest.mark.asyncio
    async def test_valid_first_try_replay_stream_works(self) -> None:
        good = _assistant([Content.from_text("streaming good")])
        fake = _FakeCallNext([good], stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, fake)

        # Caller sees a ResponseStream — must be iterable and finalizable.
        assert isinstance(ctx.result, ResponseStream)
        replayed_updates = [u async for u in ctx.result]
        assert len(_semantic_updates(replayed_updates)) == 1
        final = await ctx.result.get_final_response()
        assert final.messages[0].contents[0].text == "streaming good"

    @pytest.mark.asyncio
    async def test_rejected_provider_raw_payload_never_leaks_through_heartbeats(self) -> None:
        """Only opaque markers cross validation before an attempt is accepted."""
        rejected_raw = {"generated": "rejected secret"}
        accepted_raw = {"event": "accepted provider metadata"}
        attempt_updates = [
            [
                ChatResponseUpdate(
                    contents=[Content.from_text("   ")],
                    role="assistant",
                    raw_representation=rejected_raw,
                )
            ],
            [
                ChatResponseUpdate(contents=[], raw_representation=accepted_raw),
                ChatResponseUpdate(contents=[Content.from_text("accepted")], role="assistant"),
            ],
        ]
        attempt = 0
        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            nonlocal attempt
            updates = attempt_updates[attempt]
            attempt += 1

            async def _gen() -> AsyncIterator[ChatResponseUpdate]:
                for update in updates:
                    yield update

            ctx.result = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        await ResponseValidationMiddleware(backoff_schedule=[0.0]).process(ctx, _call_next)
        assert isinstance(ctx.result, ResponseStream)
        visible = [update async for update in ctx.result]

        assert attempt == 2
        assert visible[0].is_transport_heartbeat
        assert all(update.raw_representation is not rejected_raw for update in visible)
        accepted_raw_index = next(
            index for index, update in enumerate(visible) if update.raw_representation is accepted_raw
        )
        assert all(update.is_transport_heartbeat for update in visible[:accepted_raw_index])

    @pytest.mark.asyncio
    async def test_empty_contents_stream_retried(self) -> None:
        bad = _assistant([])
        good = _assistant([Content.from_text("stream recovered")])
        fake = _FakeCallNext([bad, good], stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ResponseStream)
        final = await ctx.result.get_final_response()
        assert fake.call_count == 2
        assert final.messages[0].contents[0].text == "stream recovered"

    @pytest.mark.asyncio
    async def test_truncated_empty_stream_normalizes_usage_before_terminal_error(self) -> None:
        usage_start = ChatResponseUpdate(
            contents=[
                Content.from_usage(usage_details={"input_token_count": 100, "output_token_count": 1}),
            ],
            role="assistant",
        )
        usage_final = ChatResponseUpdate(
            contents=[
                Content.from_usage(usage_details={"output_token_count": 7}),
            ],
            role="assistant",
        )
        terminal = ChatResponseUpdate(contents=[], role="assistant", finish_reason="length")

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield usage_start
            yield usage_final
            yield terminal

        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, _call_next)
        assert isinstance(ctx.result, ResponseStream)
        stream = ctx.result
        with pytest.raises(TerminalResponseValidationError, match="output token limit") as exc_info:
            await stream.get_final_response()

        assert exc_info.value.usage_details == {"input_token_count": 100, "output_token_count": 7}

    @pytest.mark.asyncio
    async def test_usage_cleanup_failure_does_not_mask_terminal_stream_error(self) -> None:
        usage_update = ChatResponseUpdate(
            contents=[
                Content.from_usage(usage_details={"input_token_count": 100, "output_token_count": 1}),
            ],
            role="assistant",
        )
        terminal = ChatResponseUpdate(contents=[], role="assistant", finish_reason="length")

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield usage_update
            yield terminal

        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        validation = ResponseValidationMiddleware(backoff_schedule=[0.0])

        async def _validation_next() -> None:
            await validation.process(ctx, _call_next)

        usage_calls: list[tuple[Any, ...]] = []

        def _on_usage(*args: Any) -> None:
            usage_calls.append(args)
            raise RuntimeError("usage callback failed")

        await UsageTrackingMiddleware(on_usage=_on_usage).process(ctx, _validation_next)

        assert isinstance(ctx.result, ResponseStream)
        with pytest.raises(TerminalResponseValidationError, match="output token limit") as exc_info:
            await ctx.result.get_final_response()

        assert exc_info.value.usage_details == {"input_token_count": 100, "output_token_count": 1}
        assert usage_calls
        assert usage_calls[0][1] == 100
        assert usage_calls[0][2] == 1

    @pytest.mark.asyncio
    async def test_service_retry_usage_counted_before_reraise(self) -> None:
        """A service-side retryable failure bills its rejected attempt like a terminal one."""
        err = RetryableResponseValidationError(
            "stored response invalid",
            exemption=ValidationRetryExemption(attempt=1, max_attempts=3, delay_seconds=1),
            usage_details={"input_token_count": 17, "output_token_count": 4},
        )
        ctx = _make_context(stream=False)

        async def _failing_next() -> None:
            raise err

        usage_calls: list[tuple[Any, ...]] = []

        def _on_usage(*args: Any) -> None:
            usage_calls.append(args)

        with pytest.raises(RetryableResponseValidationError) as exc_info:
            await UsageTrackingMiddleware(on_usage=_on_usage).process(ctx, _failing_next)

        assert exc_info.value is err
        assert usage_calls
        assert usage_calls[0][1] == 17
        assert usage_calls[0][2] == 4

    @pytest.mark.asyncio
    async def test_service_retry_stream_usage_counted_during_cleanup(self) -> None:
        """Streaming service-side failures bill the rejected attempt from the stream error."""
        usage_update = ChatResponseUpdate(
            contents=[
                Content.from_usage(usage_details={"input_token_count": 17, "output_token_count": 4}),
            ],
            role="assistant",
        )
        empty = ChatResponseUpdate(contents=[], role="assistant")

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield usage_update
            yield empty

        ctx = ChatContext(
            client=SimpleNamespace(STORES_BY_DEFAULT=False),
            messages=[Message("user", ["hi"])],
            options={"extra_body": {"store": True}},
            stream=True,
            kwargs={"client_kwargs": {}},
        )

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        validation = ResponseValidationMiddleware(backoff_schedule=[0.0])

        async def _validation_next() -> None:
            await validation.process(ctx, _call_next)

        usage_calls: list[tuple[Any, ...]] = []

        def _on_usage(*args: Any) -> None:
            usage_calls.append(args)

        await UsageTrackingMiddleware(on_usage=_on_usage).process(ctx, _validation_next)
        assert isinstance(ctx.result, ResponseStream)
        with pytest.raises(RetryableResponseValidationError) as exc_info:
            await ctx.result.get_final_response()

        assert exc_info.value.usage_details == {"input_token_count": 17, "output_token_count": 4}
        assert usage_calls
        assert usage_calls[0][1] == 17
        assert usage_calls[0][2] == 4

    @pytest.mark.asyncio
    async def test_reasoning_only_stream_retried_then_succeeds(self) -> None:
        # Streaming flavor: reasoning-only final responses retry inside the
        # lazy validating proxy instead of failing the stream.
        reasoning_only = _assistant([Content.from_text_reasoning(text="private thought")])
        good = _assistant([Content.from_text("stream recovered")])
        fake = _FakeCallNext([reasoning_only, good], stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ResponseStream)
        final = await ctx.result.get_final_response()
        assert fake.call_count == 2
        assert final.messages[0].contents[0].text == "stream recovered"

    @pytest.mark.asyncio
    async def test_reasoning_only_stream_repeat_failure_raises_terminal(self) -> None:
        # Streaming give-up on a reasoning-only verdict raises out of the proxy
        # (terminal_on_giveup) after the deterministic-stuck short-circuit, so
        # the executor records an error instead of a blank final message.
        reasoning_only = _assistant_truncated([Content.from_text_reasoning(text="private thought")])
        fake = _FakeCallNext([reasoning_only, reasoning_only], stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ResponseStream)
        with pytest.raises(TerminalResponseValidationError, match="output token limit while reasoning"):
            await ctx.result.get_final_response()
        assert fake.call_count == 2, "reasoning-only must retry at least once before giving up"

    @pytest.mark.asyncio
    async def test_whitespace_text_stream_retried(self) -> None:
        bad = _assistant([Content.from_text("   \n  ")])
        good = _assistant([Content.from_text("real text")])
        fake = _FakeCallNext([bad, good], stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ResponseStream)
        final = await ctx.result.get_final_response()
        assert fake.call_count == 2
        assert final.messages[0].contents[0].text == "real text"

    @pytest.mark.asyncio
    async def test_leaked_minimax_tool_call_stream_retried(self) -> None:
        bad = _assistant([Content.from_text('preface\nminimax:tool_call {"x":1} </minimax:tool_call>')])
        good = _assistant([Content.from_text("proper final answer")])
        fake = _FakeCallNext([bad, good], stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ResponseStream)
        final = await ctx.result.get_final_response()
        assert fake.call_count == 2
        assert final.messages[0].contents[0].text == "proper final answer"

    @pytest.mark.asyncio
    async def test_stream_exhaustion_drops_empty_assistant_message(self) -> None:
        """Streaming exhaustion: replay stream still iterable, but both
        the finalised response AND the scrubbed updates are empty so
        the framework's outer ``ChatResponse.from_updates`` cannot
        rebuild the bad message.

        Without filtering updates, the streaming wrapper would replay
        the original empty-content update and then ``from_updates``
        would re-create the same ``content: []`` assistant message we
        just dropped from ``response.messages``.

        Uses ``_varied_bads`` to walk distinct validator reasons so the
        fail-fast short-circuit does not fire and the loop reaches
        MAX_RETRIES exhaustion.  ``last`` pins the final attempt to an
        empty response so the give-up cleanup demonstrably drops it.
        """
        responses = _varied_bads(MAX_RETRIES + 1, last=_bad_empty())
        fake = _FakeCallNext(responses, stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)  # must NOT raise

        assert isinstance(ctx.result, ResponseStream)
        # Empty semantic updates were filtered out — only transport heartbeats
        # remain. That's correct: the framework's outer finaliser uses
        # ``ChatResponse.from_updates`` to rebuild the persisted
        # response and would re-create the bad message from any
        # leftover empty update.
        updates = [u async for u in ctx.result]
        assert fake.call_count == MAX_RETRIES + 1
        assert _semantic_updates(updates) == []
        # And the finalised response is empty too.
        final = await ctx.result.get_final_response()
        assert final.messages == []

    @pytest.mark.asyncio
    async def test_inner_result_hook_fires_only_on_caller_finalize(self) -> None:
        """The inner stream's result_hook MUST NOT fire while the middleware
        is draining + validating.  It must fire only when the caller finalises
        the replay — matching the original (no-middleware) ordering.

        Regression guard: without this, provider-installed hooks (e.g. the
        mock client's intermediate-text sync callback, which bumps
        ``IntermediateTextBuffer.batch_id``) fire too early and break
        chrys's streaming intermediate-text detection.
        """
        # Build a ChatResponse we expect to survive validation.
        good = _assistant([Content.from_text("once")])

        # Track exactly when the inner's result_hook fires.
        hook_fired_count = 0

        def _inner_hook(response: ChatResponse) -> ChatResponse:
            nonlocal hook_fired_count
            hook_fired_count += 1
            return response

        # Build an inner stream with a result_hook attached, just like
        # real providers / the MockChatClient do.
        updates = [ChatResponseUpdate(contents=good.messages[0].contents, role="assistant")]

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            for u in updates:
                yield u

        inner_stream = ResponseStream(_gen(), finalizer=lambda _u: good)
        inner_stream.with_result_hook(_inner_hook)

        # Custom call_next that returns this specific inner stream.
        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = inner_stream

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, _call_next)

        # Hook must NOT have fired yet — validation uses a preview finalize
        # that bypasses hooks.
        assert hook_fired_count == 0

        # Iterate + finalise the replay.  NOW the inner's hook should fire,
        # since the replay inherits it.
        assert isinstance(ctx.result, ResponseStream)
        _ = [u async for u in ctx.result]
        await ctx.result.get_final_response()

        assert hook_fired_count == 1

    @pytest.mark.asyncio
    async def test_inner_result_hook_runs_before_outer_result_hooks(self) -> None:
        """Provider finalization must populate usage before UsageTracking runs."""
        good = _assistant([Content.from_text("once")])
        updates = [ChatResponseUpdate(contents=good.messages[0].contents, role="assistant")]
        hook_order: list[str] = []
        outer_usage: list[tuple[int, int]] = []

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            for update in updates:
                yield update

        def _provider_hook(response: ChatResponse) -> ChatResponse:
            hook_order.append("provider")
            response.usage_details = {"input_token_count": 11, "output_token_count": 3}
            return response

        inner_stream = ResponseStream(_gen(), finalizer=lambda _updates: good)
        inner_stream.with_result_hook(_provider_hook)
        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = inner_stream

        validation = ResponseValidationMiddleware()

        async def _validation_next() -> None:
            await validation.process(ctx, _call_next)

        def _on_usage(_total: int, input_tokens: int, output_tokens: int, *_args: Any) -> None:
            hook_order.append("usage")
            outer_usage.append((input_tokens, output_tokens))

        await UsageTrackingMiddleware(on_usage=_on_usage).process(ctx, _validation_next)
        assert isinstance(ctx.result, ResponseStream)
        proxy = ctx.result
        await proxy.get_final_response()

        assert hook_order == ["provider", "usage"]
        assert outer_usage == [(11, 3)]

    @pytest.mark.asyncio
    async def test_inner_hook_does_not_fire_for_dropped_bad_attempts(self) -> None:
        """A failed validation attempt must NOT run the inner's result_hooks.

        Running them would double-fire telemetry / intermediate-text
        capture for responses the caller never sees.
        """
        bad = _assistant([])  # empty contents — invalid
        good = _assistant([Content.from_text("recovered")])

        hook_fired = 0

        def _inner_hook(response: ChatResponse) -> ChatResponse:
            nonlocal hook_fired
            hook_fired += 1
            return response

        responses = [bad, good]
        stream_idx = 0

        async def _call_next() -> None:
            nonlocal stream_idx
            resp = responses[min(stream_idx, len(responses) - 1)]
            stream_idx += 1
            updates = [ChatResponseUpdate(contents=resp.messages[0].contents, role="assistant")]

            async def _gen() -> AsyncIterator[ChatResponseUpdate]:
                for u in updates:
                    yield u

            inner = ResponseStream(_gen(), finalizer=lambda _u, r=resp: r)
            inner.with_result_hook(_inner_hook)
            ctx.result = inner

        ctx = _make_context(stream=True)
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, _call_next)

        # Before caller finalises the replay, no inner hook has fired.
        assert hook_fired == 0

        _ = [u async for u in ctx.result]  # type: ignore[union-attr]
        await ctx.result.get_final_response()  # type: ignore[union-attr]

        # Only the SUCCESSFUL (final, good) attempt's hook fires — not
        # the dropped bad attempt's.
        assert hook_fired == 1


# ---------------------------------------------------------------------------
# Rule-interaction edge cases
# ---------------------------------------------------------------------------


class TestRuleInteractions:
    @pytest.mark.asyncio
    async def test_multiple_text_fragments_concat_for_whitespace_check(self) -> None:
        """contents=[text='', text='\\n', text='   '] — all empty, must retry."""
        bad = _assistant([Content.from_text(""), Content.from_text("\n"), Content.from_text("   ")])
        good = _assistant([Content.from_text("something")])
        fake = _FakeCallNext([bad, good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)
        assert fake.call_count == 2

    @pytest.mark.asyncio
    async def test_leaked_marker_in_second_text_fragment(self) -> None:
        """Leak in ANY text fragment flags the response."""
        bad = _assistant(
            [
                Content.from_text("Thinking...\n"),
                Content.from_text("minimax:tool_call {} </minimax:tool_call>"),
            ]
        )
        good = _assistant([Content.from_text("OK")])
        fake = _FakeCallNext([bad, good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)
        assert fake.call_count == 2


# ---------------------------------------------------------------------------
# Give-up path strips function_call contents
# ---------------------------------------------------------------------------


class TestExhaustionStripsFunctionCalls:
    """The give-up path must strip ``function_call`` contents from the
    exhausted response so the Chrys tool loop sees no calls to
    execute and exits — without this the loop would iterate, re-trigger
    the full validation cycle on the next call, and never terminate
    when the model is stuck in a leaked-marker + function_call state.
    """

    @pytest.mark.asyncio
    async def test_non_stream_exhaustion_strips_function_call(self) -> None:
        # Use varied reasons across the leading attempts so the fail-fast
        # short-circuit does not fire — we want to actually exhaust
        # MAX_RETRIES.  The final attempt is the leaked-marker +
        # function_call shape we want to verify gets stripped.
        bad_with_fc = _assistant(
            [
                Content.from_text("<function_call>garbage</function_call>"),
                Content.from_function_call(call_id="c1", name="echo", arguments={"message": "x"}),
            ]
        )
        fake = _FakeCallNext(_varied_bads(MAX_RETRIES + 1, last=bad_with_fc), stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == MAX_RETRIES + 1
        assert isinstance(ctx.result, ChatResponse)
        # function_call stripped, text preserved.
        types = [c.type for c in ctx.result.messages[0].contents]
        assert "function_call" not in types
        assert "text" in types

    @pytest.mark.asyncio
    async def test_exhaustion_strips_actionable_call_but_preserves_informational_transcript(self) -> None:
        hosted_call = Content.from_function_call(
            call_id="hosted-1",
            name="web_search",
            arguments={"query": "chrys"},
            informational_only=True,
        )
        bad = _assistant(
            [
                Content.from_text("<function_call>garbage</function_call>"),
                Content.from_function_call(call_id="local-1", name="echo", arguments={"message": "x"}),
                hosted_call,
            ]
        )
        fake = _FakeCallNext(_varied_bads(MAX_RETRIES + 1, last=bad), stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        await ResponseValidationMiddleware(backoff_schedule=[0.0]).process(ctx, fake)

        assert isinstance(ctx.result, ChatResponse)
        calls = [content for content in ctx.result.messages[0].contents if content.type == "function_call"]
        assert calls == [hosted_call]

    @pytest.mark.asyncio
    async def test_stream_exhaustion_strips_function_call(self) -> None:
        bad_with_fc = _assistant(
            [
                Content.from_text("<function_call>garbage</function_call>"),
                Content.from_function_call(call_id="c1", name="echo", arguments={"message": "x"}),
            ]
        )
        fake = _FakeCallNext(_varied_bads(MAX_RETRIES + 1, last=bad_with_fc), stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ResponseStream)

        # Drain the replay — function_call must NOT appear in any update.
        updates = [u async for u in ctx.result]
        assert fake.call_count == MAX_RETRIES + 1
        for u in updates:
            for c in u.contents or []:
                assert c.type != "function_call", "Replay must not yield function_call updates"

        # Finalised response must also be stripped.
        final = await ctx.result.get_final_response()
        for msg in final.messages:
            for c in msg.contents:
                assert c.type != "function_call", "Final response must not carry function_call"

    @pytest.mark.asyncio
    async def test_non_stream_exhaustion_drops_whitespace_only_message(self) -> None:
        """Whitespace-only text exhaustion drops the message — same root
        cause as ``content: []``: a bad message would otherwise land in
        history next to the turn marker as ``content: [{text: ""}]``.
        """
        bad = _assistant([Content.from_text("\n\n  \t\n")])
        fake = _FakeCallNext([bad] * (MAX_RETRIES + 1), stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages == []

    @pytest.mark.asyncio
    async def test_stream_exhaustion_drops_whitespace_only_message(self) -> None:
        bad = _assistant([Content.from_text("   ")])
        fake = _FakeCallNext([bad] * (MAX_RETRIES + 1), stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ResponseStream)
        _ = [u async for u in ctx.result]
        final = await ctx.result.get_final_response()
        assert final.messages == []

    @pytest.mark.asyncio
    async def test_exhaustion_text_with_leaked_marker_keeps_message(self) -> None:
        """A leaked-marker exhaustion preserves the (still-bad) text so
        the user can see what the model emitted.  Only function_call
        contents are stripped; the text message stays in history."""
        bad = _assistant(
            [
                Content.from_text("Sure, calling it: minimax:tool_call {} </minimax:tool_call>"),
                Content.from_function_call(call_id="c1", name="echo", arguments={"message": "x"}),
            ]
        )
        fake = _FakeCallNext([bad] * (MAX_RETRIES + 1), stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ChatResponse)
        # Message preserved, function_call gone, text intact.
        assert len(ctx.result.messages) == 1
        types = [c.type for c in ctx.result.messages[0].contents]
        assert "function_call" not in types
        assert "text" in types
        assert "minimax" in (ctx.result.messages[0].contents[0].text or "")

    @pytest.mark.asyncio
    async def test_exhaustion_drops_only_empty_message_in_multi_message(self) -> None:
        """Multi-message response: empty assistant message dropped,
        valid one preserved.  Catches the rare provider/streaming case
        where ``_process_update`` creates a placeholder
        ``Message("assistant", [])`` that never gets populated."""
        from chrys.kernel import Message

        empty = Message(role="assistant", contents=[])
        valid = Message(role="assistant", contents=[Content.from_text("real")])
        # Build a response where validation flags the LAST message
        # (empty contents wins) — so the give-up path runs.
        bad = ChatResponse(messages=[valid, empty], finish_reason="stop")
        fake = _FakeCallNext([bad] * (MAX_RETRIES + 1), stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ChatResponse)
        # Empty message dropped, valid message preserved.
        assert len(ctx.result.messages) == 1
        assert ctx.result.messages[0].contents[0].text == "real"

    @pytest.mark.asyncio
    async def test_exhaustion_preserves_non_assistant_messages(self) -> None:
        """Tool / user messages embedded in a malformed response stay put —
        only assistant outputs are scrubbed."""
        from chrys.kernel import Message

        tool_msg = Message(
            role="tool",
            contents=[Content.from_function_result(call_id="c1", result="result text")],
        )
        bad_assistant = Message(role="assistant", contents=[])
        bad = ChatResponse(messages=[tool_msg, bad_assistant], finish_reason="stop")
        fake = _FakeCallNext([bad] * (MAX_RETRIES + 1), stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ChatResponse)
        roles = [m.role for m in ctx.result.messages]
        assert roles == ["tool"]

    @pytest.mark.asyncio
    async def test_valid_path_drops_leading_empty_assistant_non_stream(self) -> None:
        """[empty, valid] response shape — validation passes (only the
        last assistant message is checked) BUT the leading empty message
        must still be scrubbed before persistence.

        Without scrubbing on the valid path, the framework's after_run
        would persist BOTH messages: a stray ``content: []`` followed by
        the real reply.
        """
        empty = Message(role="assistant", contents=[])
        valid = Message(role="assistant", contents=[Content.from_text("real answer")])
        good = ChatResponse(messages=[empty, valid], finish_reason="stop")
        fake = _FakeCallNext([good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, fake)

        # Validation passed first try — no retries.
        assert fake.call_count == 1
        assert isinstance(ctx.result, ChatResponse)
        # Empty dropped, valid preserved.
        assert len(ctx.result.messages) == 1
        assert ctx.result.messages[0].contents[0].text == "real answer"

    @pytest.mark.asyncio
    async def test_valid_path_drops_leading_whitespace_assistant_non_stream(self) -> None:
        """Same shape but with whitespace-only text instead of empty contents."""
        ws = Message(role="assistant", contents=[Content.from_text("\n\n")])
        valid = Message(role="assistant", contents=[Content.from_text("done")])
        good = ChatResponse(messages=[ws, valid], finish_reason="stop")
        fake = _FakeCallNext([good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ChatResponse)
        assert len(ctx.result.messages) == 1
        assert ctx.result.messages[0].contents[0].text == "done"

    @pytest.mark.asyncio
    async def test_valid_path_drops_leading_empty_assistant_stream(self) -> None:
        """Streaming counterpart of the [empty, valid] non-stream case.

        The framework's outer wrapper rebuilds the persisted response
        with ``ChatResponse.from_updates(updates)``.  Without filtering
        updates on the valid path, the rebuild would re-create the
        empty leading assistant message even though we cleaned
        ``final.messages``.
        """
        # Build a stream whose updates produce [empty assistant, valid assistant].
        # Two distinct message_ids force ``_process_update`` to create
        # two messages.
        empty_upd = ChatResponseUpdate(role="assistant", message_id="m1", contents=[])
        valid_upd = ChatResponseUpdate(role="assistant", message_id="m2", contents=[Content.from_text("real")])

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield empty_upd
            yield valid_upd

        # Finaliser mirrors what ``ChatResponse.from_updates`` would produce.
        final_response = ChatResponse(
            messages=[
                Message(role="assistant", message_id="m1", contents=[]),
                Message(role="assistant", message_id="m2", contents=[Content.from_text("real")]),
            ],
            finish_reason="stop",
        )
        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=lambda _u: final_response)

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, _call_next)

        assert isinstance(ctx.result, ResponseStream)
        # Replayed updates: empty filtered out, only valid yielded so the
        # framework's outer ``from_updates`` rebuild stays clean.
        replayed = [u async for u in ctx.result]
        semantic = _semantic_updates(replayed)
        assert len(semantic) == 1
        assert semantic[0].contents[0].text == "real"
        # Final (the inner replay's finaliser path) is also clean.
        rebuilt = await ctx.result.get_final_response()
        assert len(rebuilt.messages) == 1
        assert rebuilt.messages[0].contents[0].text == "real"

    @pytest.mark.asyncio
    async def test_valid_path_keeps_metadata_terminal_update_in_filled_group(self) -> None:
        """Real providers emit a terminal "stop" update: empty contents +
        ``finish_reason="stop"``.  When the same ``(role, message_id)``
        group already received text, that group is non-empty and ALL
        its updates — including the terminal one — are kept so the
        rebuilt response retains ``finish_reason`` / ``response_id``.
        """
        text_upd = ChatResponseUpdate(
            role="assistant",
            message_id="m1",
            contents=[Content.from_text("hello")],
        )
        terminal_upd = ChatResponseUpdate(
            role="assistant",
            message_id="m1",
            contents=[],
            finish_reason="stop",
            response_id="resp-123",
        )

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield text_upd
            yield terminal_upd

        final_response = ChatResponse(
            messages=[Message(role="assistant", message_id="m1", contents=[Content.from_text("hello")])],
            finish_reason="stop",
            response_id="resp-123",
        )
        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=lambda _u: final_response)

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, _call_next)

        assert isinstance(ctx.result, ResponseStream)
        replayed = [u async for u in ctx.result]
        # Both updates kept — the terminal one preserves response-level
        # metadata that ``_process_update`` propagates onto the rebuild.
        semantic = _semantic_updates(replayed)
        assert len(semantic) == 2
        assert semantic[1].finish_reason == "stop"
        assert semantic[1].response_id == "resp-123"

    @pytest.mark.asyncio
    async def test_valid_path_drops_roleless_empty_then_tool_role_starts_new_group(self) -> None:
        """Regression: kernel default role for a placeholder message is
        ``"assistant"``.  A role-less empty leading update therefore
        creates an empty assistant message; a following ``role="tool"``
        update crosses a role boundary and starts a new tool message.

        The grouping logic must initialise its tracked role to
        ``"assistant"`` to mirror this — otherwise the empty leading
        update gets bucketed with the tool group, the assistant group is
        never identified as empty, and the framework's rebuild reproduces
        the stray ``Message("assistant", [])`` we tried to drop.
        """
        # Build the failing shape: empty role-less update + tool update.
        empty_upd = ChatResponseUpdate(contents=[])
        tool_upd = ChatResponseUpdate(
            role="tool",
            contents=[Content.from_function_result(call_id="c1", result="result text")],
        )

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield empty_upd
            yield tool_upd

        # Use a no-op validator so the test isolates the scrub path.
        # ``DefaultResponseValidator`` would actually flag this shape
        # (``_final_assistant_message`` walks back from the end and
        # returns the empty assistant message, then Rule 1 fires for
        # ``empty contents``), forcing the give-up branch.  That branch
        # also runs the scrub, but we want this test to fail loudly if
        # the *valid* branch's scrub regresses — hence the bypass.
        from chrys.service.agent_middleware.validators import DefaultResponseValidator as _DRV

        no_op = _DRV(disable_empty_contents=True, disable_whitespace_text=True, disable_leaked_tool_call=True)

        final_response = ChatResponse(
            messages=[
                Message(role="assistant", contents=[]),
                Message(
                    role="tool",
                    contents=[Content.from_function_result(call_id="c1", result="result text")],
                ),
            ],
            finish_reason="stop",
        )
        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=lambda _u: final_response)

        mw = ResponseValidationMiddleware(validator=no_op)
        await mw.process(ctx, _call_next)

        assert isinstance(ctx.result, ResponseStream)
        replayed = [u async for u in ctx.result]
        # Empty assistant update dropped, tool update kept — confirms the
        # role boundary is recognised even when the leading update has
        # no explicit role.
        semantic = _semantic_updates(replayed)
        assert len(semantic) == 1
        assert semantic[0].role == "tool"
        # Final response also has the empty assistant message dropped.
        rebuilt = await ctx.result.get_final_response()
        roles = [m.role for m in rebuilt.messages]
        assert roles == ["tool"]

    @pytest.mark.asyncio
    async def test_valid_path_drops_empty_assistant_after_tool_via_message_id_boundary(self) -> None:
        """Regression: a new group caused by a ``message_id`` change must
        treat the new placeholder as ``Message("assistant", [])`` per
        the framework, NOT inherit the previous group's role.

        Without the reset, a sequence like::

            Update(role="tool", message_id="tool-1", contents=[fr])
            Update(message_id="assistant-1", contents=[])

        would be bucketed as ``[tool, tool]`` (role inherited), so the
        empty trailing group is treated as non-assistant and kept —
        letting the framework rebuild reproduce the empty assistant
        placeholder we tried to drop.

        Expected: tool update kept, role-less empty new-message_id
        update dropped.
        """
        from chrys.service.agent_middleware.validators import DefaultResponseValidator as _DRV

        no_op = _DRV(disable_empty_contents=True, disable_whitespace_text=True, disable_leaked_tool_call=True)

        tool_upd = ChatResponseUpdate(
            role="tool",
            message_id="tool-1",
            contents=[Content.from_function_result(call_id="c1", result="ok")],
        )
        empty_assistant_upd = ChatResponseUpdate(
            message_id="assistant-1",
            contents=[],
        )

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield tool_upd
            yield empty_assistant_upd

        # Mirror what ``ChatResponse.from_updates`` would build: tool
        # message + empty assistant placeholder (role defaults to
        # "assistant" for the new message_id boundary).
        final_response = ChatResponse(
            messages=[
                Message(
                    role="tool",
                    message_id="tool-1",
                    contents=[Content.from_function_result(call_id="c1", result="ok")],
                ),
                Message(role="assistant", message_id="assistant-1", contents=[]),
            ],
            finish_reason="stop",
        )
        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=lambda _u: final_response)

        mw = ResponseValidationMiddleware(validator=no_op)
        await mw.process(ctx, _call_next)

        assert isinstance(ctx.result, ResponseStream)
        replayed = [u async for u in ctx.result]
        # Tool update kept, empty assistant placeholder dropped.
        semantic = _semantic_updates(replayed)
        assert len(semantic) == 1
        assert semantic[0].role == "tool"
        rebuilt = await ctx.result.get_final_response()
        roles = [m.role for m in rebuilt.messages]
        assert roles == ["tool"]

    @pytest.mark.asyncio
    async def test_valid_path_does_not_drop_message_with_real_text(self) -> None:
        """Sanity check: the valid-path scrub MUST NOT drop messages
        carrying real text or non-text payloads."""
        msg1 = Message(role="assistant", contents=[Content.from_text("first part")])
        msg2 = Message(role="assistant", contents=[Content.from_text("second part")])
        good = ChatResponse(messages=[msg1, msg2], finish_reason="stop")
        fake = _FakeCallNext([good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware()
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ChatResponse)
        # Both real-text messages preserved.
        assert len(ctx.result.messages) == 2
        assert ctx.result.messages[0].contents[0].text == "first part"
        assert ctx.result.messages[1].contents[0].text == "second part"

    @pytest.mark.asyncio
    async def test_valid_response_with_function_call_not_stripped(self) -> None:
        """Stripping must only happen on the give-up path, never on success.

        A first-try valid response with function_call must pass through
        unchanged — otherwise the tool loop would never execute any calls.
        """
        good_with_tool = _assistant(
            [
                Content.from_text("calling tool"),
                Content.from_function_call(call_id="c1", name="echo", arguments={"message": "x"}),
            ]
        )
        fake = _FakeCallNext([good_with_tool], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 1
        assert isinstance(ctx.result, ChatResponse)
        types = [c.type for c in ctx.result.messages[0].contents]
        assert "function_call" in types  # still present


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_service_storage_validation_failure_surfaces_without_internal_replay(stream: bool) -> None:
    bad = _bad_empty()
    fake = _FakeCallNext([bad, _assistant([Content.from_text("must stay unused")])], stream=stream)
    ctx = ChatContext(
        client=SimpleNamespace(STORES_BY_DEFAULT=False),
        messages=[Message("user", ["hi"])],
        options={"extra_body": {"store": True}},
        stream=stream,
        kwargs={"client_kwargs": {}},
    )
    fake.bind(ctx)
    retries: list[RetryAttemptInfo] = []

    async def _on_retry(info: RetryAttemptInfo) -> None:
        retries.append(info)

    middleware = ResponseValidationMiddleware(
        publish_retry=_on_retry,
        backoff_schedule=[0.0],
    )

    if stream:
        await middleware.process(ctx, fake)
        assert isinstance(ctx.result, ResponseStream)
        with pytest.raises(RetryableResponseValidationError, match="empty contents"):
            _ = [update async for update in ctx.result]
    else:
        with pytest.raises(RetryableResponseValidationError, match="empty contents"):
            await middleware.process(ctx, fake)

    assert fake.call_count == 1
    assert retries == []


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.asyncio
async def test_forced_stateless_store_true_uses_client_side_validation_retry(stream: bool) -> None:
    fake = _FakeCallNext([_bad_empty(), _assistant([Content.from_text("recovered")])], stream=stream)
    ctx = ChatContext(
        client=SimpleNamespace(STORES_BY_DEFAULT=True, FORCES_STATELESS=True),
        messages=[Message("user", ["hi"])],
        options={"store": True, "previous_response_id": "resp_1"},
        stream=stream,
        kwargs={"client_kwargs": {"store": True}},
    )
    fake.bind(ctx)
    middleware = ResponseValidationMiddleware(backoff_schedule=[0.0])

    await middleware.process(ctx, fake)
    if stream:
        assert isinstance(ctx.result, ResponseStream)
        final = await ctx.result.get_final_response()
    else:
        assert isinstance(ctx.result, ChatResponse)
        final = ctx.result

    assert final.text == "recovered"
    assert fake.call_count == 2


class TestHostedCommitEvidence:
    """Service-storage failures carry the hosted tool calls the rejected
    response already executed server-side.  The invalid response never
    reaches the kernel loop recorder, so the error is the only carrier by
    which the whole-run retry owner can honour hosted work as commit
    points instead of re-creating the request and re-running the side
    effects."""

    @staticmethod
    def _service_context(stream: bool) -> ChatContext:
        return ChatContext(
            client=SimpleNamespace(STORES_BY_DEFAULT=False),
            messages=[Message("user", ["hi"])],
            options={"extra_body": {"store": True}},
            stream=stream,
            kwargs={"client_kwargs": {}},
        )

    async def _failing_attempt(
        self, response: ChatResponse, *, stream: bool = False
    ) -> RetryableResponseValidationError:
        fake = _FakeCallNext([response], stream=stream)
        ctx = self._service_context(stream)
        fake.bind(ctx)
        middleware = ResponseValidationMiddleware(backoff_schedule=[0.0])
        if stream:
            await middleware.process(ctx, fake)
            assert isinstance(ctx.result, ResponseStream)
            with pytest.raises(RetryableResponseValidationError) as exc_info:
                _ = [update async for update in ctx.result]
        else:
            with pytest.raises(RetryableResponseValidationError) as exc_info:
                await middleware.process(ctx, fake)
        return exc_info.value

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.asyncio
    async def test_hosted_mcp_exchange_rides_on_the_raised_error(self, stream: bool) -> None:
        response = _assistant(
            [
                Content.from_mcp_server_tool_call("mc1", "create_issue", server_name="github"),
                Content.from_mcp_server_tool_result("mc1", output=[Content.from_text("created #42")]),
                Content.from_text("<tool_use>malformed</tool_use>"),
            ]
        )
        err = await self._failing_attempt(response, stream=stream)
        assert err.hosted_commits == ("create_issue",)
        assert hosted_commits_from_error(err) == ("create_issue",)

    @pytest.mark.asyncio
    async def test_hosted_shell_counts_as_commit(self) -> None:
        response = _assistant(
            [
                Content.from_shell_tool_call(call_id="sh1", commands=["curl -X POST https://x.test"]),
                Content.from_shell_tool_result(call_id="sh1", outputs=[]),
                Content.from_text("<tool_use>malformed</tool_use>"),
            ]
        )
        err = await self._failing_attempt(response)
        assert err.hosted_commits == ("shell",)

    @pytest.mark.asyncio
    async def test_hosted_free_failure_carries_no_commits(self) -> None:
        err = await self._failing_attempt(_bad_empty())
        assert err.hosted_commits == ()

    @pytest.mark.asyncio
    async def test_sandboxed_hosted_work_is_not_a_commit(self) -> None:
        # Web/file search and code interpreter cannot reach outside the
        # provider sandbox — re-running them is wasteful but safe, so they
        # must not turn a recoverable blank response into a hard failure.
        response = _assistant(
            [
                Content.from_search_tool_call(call_id="ws1", tool_name="web_search", arguments={}),
                Content.from_search_tool_result(call_id="ws1", tool_name="web_search", result={}),
                Content.from_code_interpreter_tool_call(call_id="ci1", inputs=[Content.from_text("1+1")]),
                Content.from_code_interpreter_tool_result(call_id="ci1", outputs=[]),
                Content.from_text("<tool_use>malformed</tool_use>"),
            ]
        )
        err = await self._failing_attempt(response)
        assert err.hosted_commits == ()

    @pytest.mark.asyncio
    async def test_retry_safety_side_effectful_start_is_a_commit(self) -> None:
        response = _assistant(
            [
                Content.from_hosted_tool_call(
                    "write_1",
                    tool_name="write_workspace",
                    status="running",
                    provider_phase=HostedToolPhase.START,
                    provider_status="running",
                    retry_safety=HostedRetrySafety.SIDE_EFFECTFUL,
                ),
                Content.from_text("<tool_use>malformed</tool_use>"),
            ]
        )

        err = await self._failing_attempt(response)

        assert err.hosted_commits == ("write_workspace",)

    @pytest.mark.asyncio
    async def test_retry_safety_read_only_start_is_exempt(self) -> None:
        response = _assistant(
            [
                Content.from_hosted_tool_call(
                    "read_1",
                    tool_name="read_workspace",
                    status="running",
                    provider_phase=HostedToolPhase.START,
                    provider_status="running",
                    retry_safety=HostedRetrySafety.READ_ONLY,
                ),
                Content.from_text("<tool_use>malformed</tool_use>"),
            ]
        )

        err = await self._failing_attempt(response)

        assert err.hosted_commits == ()

    def test_unrelated_errors_have_no_commits(self) -> None:
        assert hosted_commits_from_error(ConnectionError("transient")) == ()


def _bad_hosted_mcp() -> ChatResponse:
    return _assistant(
        [
            Content.from_mcp_server_tool_call("mc1", "create_issue", server_name="github"),
            Content.from_mcp_server_tool_result("mc1", output=[Content.from_text("created #42")]),
            Content.from_text("<tool_use>malformed</tool_use>"),
        ]
    )


class TestHostedCommitLocalReplayVeto:
    """Client-mode (in-place) validation retries re-send the request, so an
    invalid response that already executed hosted tool calls must give up
    immediately — the scrubbed response keeps the hosted transcript in
    history instead of re-rolling the side effects."""

    @pytest.mark.asyncio
    async def test_blocking_hosted_failure_is_not_replayed(self) -> None:
        fake = _FakeCallNext([_bad_hosted_mcp(), _assistant([Content.from_text("must stay unused")])], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 1
        assert isinstance(ctx.result, ChatResponse)
        types = [c.type for m in ctx.result.messages for c in m.contents]
        assert "mcp_server_tool_call" in types
        assert "mcp_server_tool_result" in types

    @pytest.mark.asyncio
    async def test_streaming_hosted_failure_is_not_replayed(self) -> None:
        fake = _FakeCallNext([_bad_hosted_mcp(), _assistant([Content.from_text("must stay unused")])], stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)
        assert isinstance(ctx.result, ResponseStream)
        updates = [update async for update in ctx.result]

        assert fake.call_count == 1
        types = [c.type for u in updates for c in u.contents or []]
        assert "mcp_server_tool_call" in types

    @pytest.mark.asyncio
    async def test_hosted_free_failure_still_replays_in_place(self) -> None:
        fake = _FakeCallNext([_bad_empty(), _assistant([Content.from_text("recovered")])], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 2
        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages[-1].contents[0].text == "recovered"


class TestHostedCommitObservationProbes:
    """The middleware records hosted executions as they land so retry owners
    can consult them even when the failure (stall, transport drop) carries
    no evidence of its own."""

    @pytest.mark.asyncio
    async def test_valid_hosted_response_registers_in_both_scopes(self) -> None:
        response = _assistant(
            [
                Content.from_mcp_server_tool_call("mc1", "create_issue", server_name="github"),
                Content.from_mcp_server_tool_result("mc1", output=[Content.from_text("done")]),
                Content.from_text("issue filed"),
            ]
        )
        fake = _FakeCallNext([response], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert mw.hosted_commits_observed() == ("create_issue",)
        assert mw.hosted_commits_in_flight() == ("create_issue",)

    @pytest.mark.asyncio
    async def test_streaming_updates_register_before_validation_concludes(self) -> None:
        # Evidence must exist the moment the update lands — a stall after the
        # hosted result but before stream completion consults these probes.
        hosted_update = ChatResponseUpdate(
            contents=[Content.from_mcp_server_tool_result("mc1", output=[Content.from_text("done")])],
            role="assistant",
        )
        seen_during_stream: list[tuple[str, ...]] = []

        async def _gen() -> AsyncIterator[ChatResponseUpdate]:
            yield ChatResponseUpdate(
                contents=[Content.from_mcp_server_tool_call("mc1", "create_issue")],
                role="assistant",
            )
            yield hosted_update
            seen_during_stream.append(mw.hosted_commits_in_flight())
            yield ChatResponseUpdate(contents=[Content.from_text("answer")], role="assistant")

        ctx = _make_context(stream=True)

        async def _call_next() -> None:
            ctx.result = ResponseStream(_gen(), finalizer=ChatResponse.from_updates)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, _call_next)
        assert isinstance(ctx.result, ResponseStream)
        _ = [update async for update in ctx.result]

        assert seen_during_stream == [("create_issue",)]

    @pytest.mark.asyncio
    async def test_new_wire_attempt_resets_in_flight_but_not_run_scope(self) -> None:
        hosted = _assistant(
            [
                Content.from_mcp_server_tool_call("mc1", "create_issue", server_name="github"),
                Content.from_text("issue filed"),
            ]
        )
        plain = _assistant([Content.from_text("plain answer")])
        fake = _FakeCallNext([hosted, plain], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)
        assert mw.hosted_commits_in_flight() == ("create_issue",)

        fake.bind(ctx)
        await mw.process(ctx, fake)
        # The second wire call carried no hosted work: an in-place replay of
        # THAT request is safe, but a whole-run retry (which would restore
        # pre-run history and re-create the hosted exchange) is not.
        assert mw.hosted_commits_in_flight() == ()
        assert mw.hosted_commits_observed() == ("create_issue",)

    def test_reset_clears_both_scopes(self) -> None:
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        mw._observe_hosted_contents([Content.from_mcp_server_tool_call("mc1", "create_issue")])
        assert mw.hosted_commits_observed() == ("create_issue",)
        mw.reset_hosted_commit_observations()
        assert mw.hosted_commits_observed() == ()
        assert mw.hosted_commits_in_flight() == ()


class TestServiceStorageRetryBudget:
    """Service-storage failures retry at the whole-run boundary, so the
    middleware carries its budget and last failure reason on the instance
    across outer attempts.  Each ``process()`` call below simulates one
    outer whole-run attempt against the same middleware instance."""

    @staticmethod
    def _service_context(stream: bool, exchange: ExchangeTrace | None = None) -> ChatContext:
        client_kwargs: dict[str, Any] = {} if exchange is None else {TRAJECTORY_EXCHANGE_KWARG: exchange}
        return ChatContext(
            client=SimpleNamespace(STORES_BY_DEFAULT=False),
            messages=[Message("user", ["hi"])],
            options={"extra_body": {"store": True}},
            stream=stream,
            kwargs={"client_kwargs": client_kwargs},
        )

    async def _outer_attempt(
        self,
        mw: ResponseValidationMiddleware,
        response: ChatResponse,
        *,
        stream: bool = False,
        exchange: ExchangeTrace | None = None,
    ) -> ChatContext:
        fake = _FakeCallNext([response], stream=stream)
        ctx = self._service_context(stream, exchange)
        fake.bind(ctx)
        await mw.process(ctx, fake)
        if stream:
            assert isinstance(ctx.result, ResponseStream)
            _ = [update async for update in ctx.result]
        return ctx

    @pytest.mark.asyncio
    async def test_identical_reason_second_outer_attempt_is_terminal(self) -> None:
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError, match="empty contents"):
            await self._outer_attempt(mw, _bad_empty())
        with pytest.raises(TerminalResponseValidationError, match="empty contents"):
            await self._outer_attempt(mw, _bad_empty())

    @pytest.mark.asyncio
    async def test_identical_reason_second_outer_attempt_is_terminal_stream(self) -> None:
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError, match="empty contents"):
            await self._outer_attempt(mw, _bad_empty(), stream=True)
        with pytest.raises(TerminalResponseValidationError, match="empty contents"):
            await self._outer_attempt(mw, _bad_empty(), stream=True)

    @pytest.mark.asyncio
    async def test_budget_exhaustion_with_distinct_reasons_is_terminal(self) -> None:
        """Distinct reasons dodge the short-circuit; the count budget still trips."""
        mw = ResponseValidationMiddleware(max_retries=2, backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_whitespace())
        with pytest.raises(TerminalResponseValidationError):
            await self._outer_attempt(mw, _bad_leaked())

    @pytest.mark.asyncio
    async def test_retry_errors_stamp_actual_budget_attempt_and_ceiled_delay(self) -> None:
        mw = ResponseValidationMiddleware(max_retries=2, backoff_schedule=[0.2, 1.2])

        with pytest.raises(RetryableResponseValidationError) as first:
            await self._outer_attempt(mw, _bad_empty())
        with pytest.raises(RetryableResponseValidationError) as second:
            await self._outer_attempt(mw, _bad_whitespace())

        assert first.value.exemption == ValidationRetryExemption(attempt=1, max_attempts=2, delay_seconds=1)
        assert second.value.exemption == ValidationRetryExemption(attempt=2, max_attempts=2, delay_seconds=2)

    @pytest.mark.asyncio
    async def test_success_resets_carried_budget(self) -> None:
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())
        ctx = await self._outer_attempt(mw, _assistant([Content.from_text("recovered")]))
        assert isinstance(ctx.result, ChatResponse)
        # The next identical failure starts a fresh cycle, not a short-circuit.
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())

    @pytest.mark.asyncio
    async def test_cycle_start_reset_gives_fresh_budget(self) -> None:
        """An aborted outer cycle (interrupt during backoff, unrelated
        exception) concludes through none of the internal reset points; the
        executor calls ``reset_service_retry_state`` at the next cycle start
        so the leftover count/reason cannot judge an independent run."""
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())
        mw.reset_service_retry_state()
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())

    @pytest.mark.asyncio
    async def test_client_mode_giveup_resets_carried_state(self) -> None:
        """A concluded client-mode cycle must not leave stale carry-over
        behind a later storage-mode flip back to service storage."""
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())
        # Client-mode cycle: identical failures short-circuit to scrub-accept.
        fake = _FakeCallNext([_bad_empty(), _bad_empty()], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)
        await mw.process(ctx, fake)
        assert isinstance(ctx.result, ChatResponse)
        # Back in service mode, the same reason starts a fresh cycle.
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_the_recorded_verdict_states_the_service_side_give_up(self, stream: bool) -> None:
        """The give-up decision belongs in the event that reports the failure.

        Service-storage mode takes it against the carried count/reason instead
        of a loop attempt index, and the terminal raise below is what an
        analyst reading the log has to see as the cycle's give-up."""
        sink = FakeSink()
        exchange = ExchangeTrace(make_context(sink).with_cycle(new_analytics_id()).with_exchange(new_analytics_id()))
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty(), stream=stream, exchange=exchange)
        with pytest.raises(TerminalResponseValidationError):
            await self._outer_attempt(mw, _bad_empty(), stream=stream, exchange=exchange)

        retried, gave_up = sink.of_type(EventType.MODEL_VALIDATION_FINISHED)
        assert "gave_up" not in retried.payload
        assert gave_up.payload["gave_up"] is True

    @pytest.mark.asyncio
    async def test_terminal_giveup_resets_carried_budget(self) -> None:
        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())
        with pytest.raises(TerminalResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())
        # A later user-initiated run gets a fresh budget after the terminal raise.
        with pytest.raises(RetryableResponseValidationError):
            await self._outer_attempt(mw, _bad_empty())


# ---------------------------------------------------------------------------
# Fail-fast — identical-reason short-circuit
# ---------------------------------------------------------------------------


class TestFailFastOnIdenticalReason:
    """Two consecutive attempts with the same ``ValidationResult.reason``
    almost always indicate a deterministic-stuck producer (KV-cache /
    chat-template / fine-tuning quirk).  Re-rolling with identical input
    will produce identical output, so the middleware short-circuits to
    the give-up path instead of paying the full MAX_RETRIES latency tax.
    """

    @pytest.mark.asyncio
    async def test_two_identical_empty_responses_short_circuit_non_stream(self) -> None:
        """Same reason on attempts 0 and 1 → give up at attempt 1 without
        consuming the remaining MAX_RETRIES - 1 attempts."""
        # Three available bads; we expect only 2 to be consumed before fail-fast.
        fake = _FakeCallNext([_bad_empty(), _bad_empty(), _bad_empty()], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[str] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info.reason)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        # Initial + one retry = 2 attempts.  Remaining retries are skipped.
        assert fake.call_count == 2
        # publish_retry fires only on the first (real) retry, not on the
        # fail-fast short-circuit.
        assert len(retries) == 1
        # Give-up cleanup still runs: empty assistant message dropped.
        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages == []

    @pytest.mark.asyncio
    async def test_two_identical_responses_short_circuit_stream(self) -> None:
        fake = _FakeCallNext([_bad_empty(), _bad_empty(), _bad_empty()], stream=True)
        ctx = _make_context(stream=True)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert isinstance(ctx.result, ResponseStream)
        # Replay must still drop the empty updates so the framework's
        # outer ``ChatResponse.from_updates`` cannot rebuild the bad msg.
        updates = [u async for u in ctx.result]
        assert fake.call_count == 2
        assert _semantic_updates(updates) == []
        final = await ctx.result.get_final_response()
        assert final.messages == []

    @pytest.mark.asyncio
    async def test_fail_fast_strips_function_call_on_repeat(self) -> None:
        """Give-up cleanup must run even when triggered by fail-fast,
        not only on MAX_RETRIES exhaustion.  Otherwise a stuck producer
        would leak ``function_call`` contents into the tool loop after
        the early give-up and re-trigger the validation cycle every
        iteration — defeating the whole point of give-up scrubbing.
        """
        bad_with_fc = _assistant(
            [
                Content.from_text("<function_call>garbage</function_call>"),
                Content.from_function_call(call_id="c1", name="echo", arguments={"message": "x"}),
            ]
        )
        # Two identical bads → fail-fast at attempt 1 (well before MAX_RETRIES).
        fake = _FakeCallNext([bad_with_fc, bad_with_fc, bad_with_fc, bad_with_fc], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 2  # short-circuit, not MAX_RETRIES + 1
        assert isinstance(ctx.result, ChatResponse)
        types = [c.type for c in ctx.result.messages[0].contents]
        assert "function_call" not in types  # stripped on the early give-up
        assert "text" in types

    @pytest.mark.asyncio
    async def test_first_attempt_invalid_always_retries_at_least_once(self) -> None:
        """Fail-fast needs a *previous* reason to compare against, so the
        very first invalid attempt must always be retried — even if the
        producer ends up deterministic.  Without this, a single transient
        hiccup would skip recovery entirely.
        """
        good = _assistant([Content.from_text("recovered")])
        fake = _FakeCallNext([_bad_empty(), good], stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[str] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info.reason)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        assert fake.call_count == 2
        assert len(retries) == 1  # the first invalid attempt did get a retry
        assert ctx.result.messages[0].contents[0].text == "recovered"

    @pytest.mark.asyncio
    async def test_alternating_reasons_do_not_trigger_fail_fast(self) -> None:
        """When consecutive attempts have *different* reasons, the producer
        is plausibly stochastic — keep retrying until either a valid
        response arrives or MAX_RETRIES is reached.  Verifies fail-fast
        does not misfire on a sequence like empty → whitespace → empty.
        """
        good = _assistant([Content.from_text("ok")])
        # empty / whitespace / empty / good — three distinct reason
        # transitions, none repeating consecutively.
        fake = _FakeCallNext(
            [_bad_empty(), _bad_whitespace(), _bad_empty(), good],
            stream=False,
        )
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        retries: list[str] = []

        async def on_retry(info: RetryAttemptInfo) -> None:
            retries.append(info.reason)

        mw = ResponseValidationMiddleware(publish_retry=on_retry, backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        # All 3 bads consumed + 1 good = 4 calls.  Three retry events,
        # one for each invalid attempt that preceded the next call.
        assert fake.call_count == 4
        assert len(retries) == 3
        assert ctx.result.messages[0].contents[0].text == "ok"

    @pytest.mark.asyncio
    async def test_final_attempt_takes_priority_over_fail_fast(self) -> None:
        """When the final attempt's reason matches the previous, the
        regular exhaustion give-up path runs (not the early fail-fast
        log).  Both lead to the same cleanup, but the log message
        differs — verify cleanup correctness here regardless.
        """
        # MAX_RETRIES + 1 attempts: varied for the leading ones, then
        # the same reason on the final two.
        responses = [_bad_empty(), _bad_whitespace(), _bad_empty(), _bad_empty()]
        assert len(responses) == MAX_RETRIES + 1
        fake = _FakeCallNext(responses, stream=False)
        ctx = _make_context(stream=False)
        fake.bind(ctx)

        mw = ResponseValidationMiddleware(backoff_schedule=[0.0])
        await mw.process(ctx, fake)

        # All MAX_RETRIES + 1 attempts consumed because the repeat only
        # appears on the final attempt — there were no earlier
        # consecutive matches to short-circuit on.
        assert fake.call_count == MAX_RETRIES + 1
        assert isinstance(ctx.result, ChatResponse)
        assert ctx.result.messages == []  # give-up cleanup ran


class TestValidationObservationHook:
    @pytest.mark.asyncio
    async def test_blocking_accepted_attempt_callbacks(self) -> None:
        good = _assistant([Content.from_text("accepted")])
        fake = _FakeCallNext([good], stream=False)
        context = _make_context(stream=False)
        fake.bind(context)
        hook = _ObservationHook()

        await ResponseValidationMiddleware(observation_hook=hook).process(context, fake)

        assert hook.events == [
            ("response", None, None),
            ("started", False),
            ("contents", True, ("text",)),
            ("accepted", (("text",),)),
        ]

    @pytest.mark.asyncio
    async def test_blocking_rejected_then_accepted_callbacks(self) -> None:
        bad = _assistant([])
        good = _assistant([Content.from_text("accepted")])
        fake = _FakeCallNext([bad, good], stream=False)
        context = _make_context(stream=False)
        fake.bind(context)
        hook = _ObservationHook()

        await ResponseValidationMiddleware(
            observation_hook=hook,
            backoff_schedule=[0.0],
        ).process(context, fake)

        assert hook.events == [
            ("response", None, None),
            ("started", False),
            ("contents", True, ()),
            ("rejected", "empty contents"),
            ("started", False),
            ("contents", True, ("text",)),
            ("accepted", (("text",),)),
        ]

    @pytest.mark.asyncio
    async def test_streaming_observes_updates_and_acceptance(self) -> None:
        good = _assistant([Content.from_text("accepted")])
        fake = _FakeCallNext([good], stream=True)
        context = _make_context(stream=True)
        fake.bind(context)
        hook = _ObservationHook()
        middleware = ResponseValidationMiddleware(observation_hook=hook)

        await middleware.process(context, fake)
        assert isinstance(context.result, ResponseStream)
        await context.result.get_final_response()

        assert hook.events == [
            ("response", None, None),
            ("started", False),
            ("contents", False, ("text",)),
            ("contents", True, ("text",)),
            ("accepted", (("text",),)),
        ]
