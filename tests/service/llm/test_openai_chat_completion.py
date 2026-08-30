# Copyright (c) 2026 Chrys. All rights reserved.

"""Chat Completions wire-prep pins: author-name sanitizing, response-format
normalization, and cache-write usage extraction.

The ``name`` field on wire messages accepts CJK and hyphens but rejects
whitespace and ``< | \\ / >`` (the API's documented charset); the
response-format schema ``name`` is stricter (``[A-Za-z0-9_-]``, ≤64). Both
sanitizers run at the wire boundary so persisted or caller-supplied values
can never 400 a request Chrys itself constructs.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pytest
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, ChoiceDelta
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
from pydantic import BaseModel

from chrys.kernel import ChatResponse, Content, Message
from chrys.service.llm.deepseek import DeepSeekChatCompletionClient
from chrys.service.llm.openai_chat_completion import (
    RawOpenAIChatCompletionClient,
    _sanitize_author_name,
)


class _UnusedCompletions:
    async def create(self, *_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("wire-prep tests should not call the SDK")


class _UnusedChat:
    def __init__(self) -> None:
        self.completions = _UnusedCompletions()


class _UnusedAsyncOpenAI:
    base_url = "https://api.test"

    def __init__(self) -> None:
        self.chat = _UnusedChat()


def _client() -> RawOpenAIChatCompletionClient:
    return RawOpenAIChatCompletionClient(model="glm-5.2", async_client=_UnusedAsyncOpenAI())


def _deepseek() -> DeepSeekChatCompletionClient:
    return DeepSeekChatCompletionClient(model="deepseek-reasoner", async_client=_UnusedAsyncOpenAI())


def _user(text: str = "hi", **kwargs: Any) -> Message:
    return Message(role="user", contents=[text], **kwargs)


# ───────────────────────── author_name sanitizing ─────────────────────────


def test_sanitize_author_name_value_rules() -> None:
    assert _sanitize_author_name("my agent") == "myagent"
    assert _sanitize_author_name("小助手") == "小助手", "CJK is valid for the wire name field"
    assert _sanitize_author_name("my-agent_2") == "my-agent_2"
    assert _sanitize_author_name("a<b|c\\d/e>f") == "abcdef"
    assert _sanitize_author_name("< | / >") is None, "empty after sanitize omits the key"
    assert _sanitize_author_name(None) is None
    assert _sanitize_author_name("") is None
    truncated = _sanitize_author_name("a b" + "c" * 80)
    assert truncated is not None
    assert len(truncated) == 64
    assert truncated == ("ab" + "c" * 80)[:64]


def test_system_message_author_name_sanitized() -> None:
    message = Message(role="system", contents=["be nice"], author_name="sys helper")
    (prepared,) = _client()._prepare_message_for_openai(message)
    assert prepared["name"] == "syshelper"


def test_user_message_author_name_sanitized() -> None:
    (prepared,) = _client()._prepare_message_for_openai(_user(author_name="user one"))
    assert prepared["name"] == "userone"


def test_unsanitizable_author_name_omits_the_key() -> None:
    (prepared,) = _client()._prepare_message_for_openai(_user(author_name="< / >"))
    assert "name" not in prepared


def test_tool_result_message_never_carries_a_name() -> None:
    message = Message(
        role="tool",
        contents=[Content.from_function_result(call_id="c1", result="ok")],
        author_name="tool helper",
    )
    (prepared,) = _client()._prepare_message_for_openai(message)
    assert "name" not in prepared


def test_reasoning_coalescer_aggregate_sanitizes_name_and_keeps_reasoning() -> None:
    # The run-coalescer aggregate path builds its own wire message; the
    # sanitize must touch only ``name``, never the reasoning keys.
    reasoning = Content.from_text_reasoning(
        text="thinking",
        additional_properties={"openai_reasoning_format": "reasoning_content"},
    )
    message = Message(
        role="assistant",
        contents=[reasoning, Content.from_text("answer")],
        author_name="agent one",
    )
    (aggregate,) = _client()._prepare_message_for_openai(message)
    assert aggregate["name"] == "agentone"
    assert aggregate["reasoning_content"] == "thinking"
    assert aggregate["content"] == "answer"


def test_reasoning_coalescer_standalone_non_text_content_sanitizes_name() -> None:
    # A non-text fragment (e.g. an image) keeps its standalone emission ahead
    # of the run's aggregate; that standalone message carries the sanitized
    # name too.
    reasoning = Content.from_text_reasoning(
        text="thinking",
        additional_properties={"openai_reasoning_format": "reasoning_content"},
    )
    message = Message(
        role="assistant",
        contents=[
            reasoning,
            Content.from_text("answer"),
            Content.from_uri(uri="https://example.com/img.png", media_type="image/png"),
        ],
        author_name="agent one",
    )
    standalone, aggregate = _client()._prepare_message_for_openai(message)
    assert standalone["content"][0]["type"] == "image_url"
    assert standalone["name"] == "agentone"
    assert aggregate["name"] == "agentone"
    assert aggregate["reasoning_content"] == "thinking"


def test_reasoning_props_carrier_message_sanitizes_name() -> None:
    # Message-level reasoning props with no wire carrier synthesize an
    # empty-content assistant message; it carries the sanitized name.
    message = Message(
        role="assistant",
        contents=[],
        additional_properties={"reasoning_content": "props chain", "openai_reasoning_format": "reasoning_content"},
        author_name="agent one",
    )
    (carrier,) = _client()._prepare_message_for_openai(message)
    assert carrier == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "props chain",
        "name": "agentone",
    }


def test_restored_history_author_name_sanitized_on_wire() -> None:
    # A deserialized message carrying an invalid persisted author_name must
    # still serialize to the wire sanitized.
    restored = Message.from_dict(_user(author_name="restored user").to_dict())
    assert restored.author_name == "restored user"
    (prepared,) = _client()._prepare_message_for_openai(restored)
    assert prepared["name"] == "restoreduser"


def test_deepseek_sites_sanitize_author_name() -> None:
    deepseek = _deepseek()
    (sys_prepared,) = deepseek._prepare_message_for_openai(
        Message(role="system", contents=["s"], author_name="sys helper")
    )
    assert sys_prepared["name"] == "syshelper"
    (user_prepared,) = deepseek._prepare_message_for_openai(_user(author_name="user one"))
    assert user_prepared["name"] == "userone"


# ───────────────────────── response_format normalization ─────────────────────────


def _prepared_response_format(response_format: Any) -> Any:
    run_options = _client()._prepare_options([_user()], {"response_format": response_format})
    return run_options["response_format"]


def test_raw_object_schema_is_wrapped() -> None:
    raw = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "title": "Weather Digest",
    }
    original = {**raw, "properties": dict(raw["properties"])}
    wrapped = _prepared_response_format(raw)
    assert wrapped["type"] == "json_schema"
    envelope = wrapped["json_schema"]
    assert envelope["name"] == "WeatherDigest"
    assert envelope["strict"] is True
    assert envelope["schema"]["additionalProperties"] is False
    # Strict mode rejects schemas whose objects omit ``required``.
    assert envelope["schema"]["required"] == ["answer"]
    assert "title" not in envelope["schema"]
    assert raw == original, "input dict must not be mutated"


def test_keyword_only_schema_is_wrapped() -> None:
    # A type-less root violates the strict-mode root contract (the root must
    # be an object schema), so the wrap goes out non-strict, verbatim.
    wrapped = _prepared_response_format({"properties": {"a": {"type": "integer"}}})
    assert wrapped["type"] == "json_schema"
    envelope = wrapped["json_schema"]
    assert envelope["name"] == "response"
    assert "strict" not in envelope
    assert envelope["schema"] == {"properties": {"a": {"type": "integer"}}}


def test_raw_schema_strictified_recursively() -> None:
    raw = {
        "type": "object",
        "properties": {
            "digest": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            }
        },
    }
    envelope = _prepared_response_format(raw)["json_schema"]
    assert envelope["strict"] is True
    nested = envelope["schema"]["properties"]["digest"]
    assert nested["additionalProperties"] is False
    assert nested["required"] == ["answer"]
    assert "additionalProperties" not in raw["properties"]["digest"], "input dict must not be mutated"


def test_unstrictifiable_raw_schema_sent_non_strict_verbatim() -> None:
    # Boolean subschemas are valid JSON Schema but the SDK strictifier
    # rejects them; forcing strict anyway would 400 on the API side, so the
    # schema goes to the wire non-strict exactly as the caller wrote it.
    raw = {
        "type": "object",
        "properties": {"anything": True},
        "title": "Loose",
    }
    wrapped = _prepared_response_format(raw)
    envelope = wrapped["json_schema"]
    assert envelope["name"] == "Loose"
    assert "strict" not in envelope
    assert envelope["schema"] == {"type": "object", "properties": {"anything": True}}


@pytest.mark.parametrize(
    "raw",
    [
        {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": True},
        {
            "type": "object",
            "properties": {"inner": {"type": "object", "properties": {}, "additionalProperties": True}},
        },
        {"type": "object", "properties": {}, "additionalProperties": {"type": "string"}},
    ],
)
def test_explicitly_open_schema_sent_non_strict_verbatim(raw: dict[str, Any]) -> None:
    # The strictifier preserves an explicit non-false additionalProperties
    # anywhere in the tree; strict mode rejects exactly that, so the schema
    # must go to the wire non-strict, verbatim.
    envelope = _prepared_response_format({**raw, "title": "Open"})["json_schema"]
    assert envelope["name"] == "Open"
    assert "strict" not in envelope
    assert envelope["schema"] == raw


@pytest.mark.parametrize(
    "raw",
    [
        {
            "type": "object",
            "properties": {"choice": {"oneOf": [{"type": "object", "properties": {"value": {"type": "string"}}}]}},
        },
        {
            "type": "object",
            "properties": {
                "x": {"allOf": [{"type": "object", "properties": {}}, {"type": "object", "properties": {}}]}
            },
        },
        {"type": "object", "properties": {}, "patternProperties": {"^x": {"type": "object", "properties": {}}}},
        {"type": "object", "properties": {"a": {"type": "string"}}, "dependentRequired": {"a": ["b"]}},
        {"type": "object", "properties": {"a": {"type": "string"}}, "dependentSchemas": {"a": {"type": "string"}}},
        {
            "type": "object",
            "properties": {},
            "patternProperties": {
                "^x": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "additionalProperties": False,
                }
            },
        },
        {
            "type": "object",
            "properties": {},
            "patternProperties": {
                "^x": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "additionalProperties": False,
                    "required": ["value", "value"],
                }
            },
        },
        {
            "type": "object",
            "properties": {},
            "patternProperties": {
                "^x": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "additionalProperties": False,
                    "required": [["value"]],
                }
            },
        },
    ],
)
def test_unstrictified_branches_sent_non_strict_verbatim(raw: dict[str, Any]) -> None:
    # The strictifier does not traverse oneOf or patternProperties (their
    # objects stay open) and leaves multi-entry allOf and the dependent*
    # keywords in place; strict mode rejects all of them, so none may go out
    # strict. The dependent* values hold property names or primitive
    # subschemas, so only the keyword gate can catch them. The last three
    # shapes exercise the required-completeness leg on a pre-closed object:
    # missing required, duplicate names that a bare set comparison would
    # collapse, and an unhashable entry that must fall back instead of
    # raising from set().
    envelope = _prepared_response_format({**raw, "title": "Branch"})["json_schema"]
    assert envelope["name"] == "Branch"
    assert "strict" not in envelope
    assert envelope["schema"] == raw


@pytest.mark.parametrize(
    "raw",
    [
        {"type": "string"},
        {"type": "object", "properties": {}, "anyOf": [{"type": "object", "properties": {}}]},
        {"anyOf": [{"type": "object", "properties": {}}, {"type": "object", "properties": {}}]},
    ],
)
def test_strict_root_contract_violations_sent_non_strict_verbatim(raw: dict[str, Any]) -> None:
    # Structured Outputs requires the root to be an object schema and
    # forbids a root-level anyOf; the strictifier passes such roots through
    # unchanged, so they must go to the wire non-strict, verbatim.
    envelope = _prepared_response_format({**raw, "title": "Root"})["json_schema"]
    assert envelope["name"] == "Root"
    assert "strict" not in envelope
    assert envelope["schema"] == raw


def test_mapping_proxy_response_format_supported() -> None:
    # The response_format contract is Mapping, which admits read-only views;
    # a dict-only gate would misroute them to the SDK's model-class helper.
    raw = MappingProxyType({"type": "object", "properties": {"answer": {"type": "string"}}, "title": "Prox"})
    envelope = _prepared_response_format(raw)["json_schema"]
    assert envelope["name"] == "Prox"
    assert envelope["strict"] is True
    assert envelope["schema"]["required"] == ["answer"]
    assert "title" in raw, "read-only input must stay untouched"


def test_nested_mapping_proxy_response_format_supported() -> None:
    # Read-only views can sit at any depth, not just the root — deepcopy
    # falls back to pickling them and raises; the materializing copy must
    # convert the whole tree to plain containers.
    raw = MappingProxyType(
        {
            "type": "object",
            "properties": MappingProxyType({"answer": MappingProxyType({"type": "string", "enum": ("a", "b")})}),
            "title": "DeepProx",
        }
    )
    envelope = _prepared_response_format(raw)["json_schema"]
    assert envelope["name"] == "DeepProx"
    assert envelope["strict"] is True
    assert isinstance(envelope["schema"]["properties"], dict)
    assert envelope["schema"]["properties"]["answer"]["enum"] == ["a", "b"]
    assert envelope["schema"]["required"] == ["answer"]


_FT_MODEL = "ft:gpt-4o-mini:acme::abc123"

_CONSTRAINED_RAW_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string", "minLength": 2}},
}


def _prepared_response_format_for_model(model: str, response_format: Any) -> Any:
    run_options = _client()._prepare_options([_user()], {"model": model, "response_format": response_format})
    return run_options["response_format"]


@pytest.mark.parametrize(
    "constrained_property",
    [
        {"type": "string", "minLength": 2},
        {"type": "integer", "maximum": 10},
        {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    ],
    ids=["string-bound", "number-bound", "array-bound"],
)
def test_fine_tuned_model_constraint_keywords_sent_non_strict(constrained_property: dict[str, Any]) -> None:
    # Structured Outputs rejects constraint keywords on fine-tuned models
    # only — strict:true would 400 there, so the schema goes to the wire
    # non-strict, verbatim.
    raw = {"type": "object", "properties": {"code": constrained_property}, "title": "Constrained"}
    envelope = _prepared_response_format_for_model(_FT_MODEL, raw)["json_schema"]
    assert envelope["name"] == "Constrained"
    assert "strict" not in envelope
    assert envelope["schema"] == {"type": "object", "properties": {"code": constrained_property}}


def test_base_model_keeps_strict_with_constraint_keywords() -> None:
    # Base models support the constraint keywords under strict mode; the
    # fine-tune gate must not cost them the strict guarantee.
    envelope = _prepared_response_format_for_model("gpt-4o", _CONSTRAINED_RAW_SCHEMA)["json_schema"]
    assert envelope["strict"] is True
    assert envelope["schema"]["properties"]["code"]["minLength"] == 2


def test_fine_tuned_model_without_constraint_keywords_stays_strict() -> None:
    raw = {"type": "object", "properties": {"code": {"type": "string"}}}
    envelope = _prepared_response_format_for_model(_FT_MODEL, raw)["json_schema"]
    assert envelope["strict"] is True


@pytest.mark.parametrize(
    "raw",
    [
        {"type": ["object", "null"], "properties": {"a": {"type": "string"}}},
        {"enum": ["red", "green"]},
    ],
)
def test_unusual_raw_schema_shapes_wrapped_non_strict(raw: dict[str, Any]) -> None:
    # JSON Schema admits array-valued and absent ``type`` (enum-only), so
    # both classify as raw schemas — not a crash, a passthrough, or a
    # rejection. Neither satisfies the strict root contract, so they wrap
    # non-strict verbatim.
    envelope = _prepared_response_format({**raw, "title": "Odd"})["json_schema"]
    assert envelope["name"] == "Odd"
    assert "strict" not in envelope
    assert envelope["schema"] == raw


def test_empty_raw_schema_wrapped_non_strict() -> None:
    # An empty mapping is a valid type-less JSON Schema; a truthiness gate
    # would leave it riding the seeded options copy unwrapped.
    envelope = _prepared_response_format({})["json_schema"]
    assert envelope["name"] == "response"
    assert "strict" not in envelope
    assert envelope["schema"] == {}


def test_json_object_and_text_pass_through_unchanged() -> None:
    for passthrough in ({"type": "json_object"}, {"type": "text"}):
        assert _prepared_response_format(passthrough) is passthrough


def test_cjk_title_falls_back_to_response() -> None:
    wrapped = _prepared_response_format({"type": "object", "properties": {}, "title": "天气摘要"})
    assert wrapped["json_schema"]["name"] == "response"


def test_long_title_truncates_to_64() -> None:
    wrapped = _prepared_response_format({"type": "object", "properties": {}, "title": "T" * 80})
    assert wrapped["json_schema"]["name"] == "T" * 64


def test_enveloped_dict_with_invalid_name_sanitized_copy_on_write() -> None:
    envelope = {
        "type": "json_schema",
        "json_schema": {"name": "bad name!", "schema": {"type": "object"}},
    }
    prepared = _prepared_response_format(envelope)
    assert prepared["json_schema"]["name"] == "badname"
    assert envelope["json_schema"]["name"] == "bad name!", "input not mutated"


def test_enveloped_dict_with_valid_name_passes_identically() -> None:
    envelope = {
        "type": "json_schema",
        "json_schema": {"name": "Good_Name-1", "schema": {"type": "object"}},
    }
    assert _prepared_response_format(envelope) is envelope


def test_model_class_name_is_sanitized() -> None:
    # The SDK helper copies the class __name__ into json_schema.name
    # unchecked; a PEP 3131 identifier would 400 without the sanitize.
    class _Digest(BaseModel):
        answer: str

    _Digest.__name__ = "天气 Digest"
    prepared = _prepared_response_format(_Digest)
    assert prepared["json_schema"]["name"] == "Digest"


# ───────────────────────── cache_write_tokens extraction ─────────────────────────


def _usage(**detail_kwargs: Any) -> CompletionUsage:
    return CompletionUsage(
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=3, **detail_kwargs),
    )


def test_cache_write_tokens_extracted() -> None:
    details = _client()._parse_usage_from_openai(_usage(cache_write_tokens=7))
    assert details["prompt/cache_write_tokens"] == 7
    assert details["cache_creation_input_token_count"] == 7


def test_cache_write_tokens_zero_preserved() -> None:
    details = _client()._parse_usage_from_openai(_usage(cache_write_tokens=0))
    assert details["prompt/cache_write_tokens"] == 0
    assert details["cache_creation_input_token_count"] == 0


def test_cache_write_tokens_absent_omits_both_keys() -> None:
    details = _client()._parse_usage_from_openai(_usage())
    assert "prompt/cache_write_tokens" not in details
    assert "cache_creation_input_token_count" not in details
    assert details["cache_read_input_token_count"] == 3


def _completion_choice(*, content: Any, refusal: Any = None) -> Choice:
    return Choice.model_construct(
        index=0,
        finish_reason="stop",
        logprobs=None,
        message=ChatCompletionMessage.model_construct(role="assistant", content=content, refusal=refusal),
    )


def _chunk_choice(*, content: Any = None, reasoning_content: Any = None, refusal: Any = None) -> ChunkChoice:
    return ChunkChoice.model_construct(
        index=0,
        finish_reason=None,
        logprobs=None,
        delta=ChoiceDelta.model_construct(
            role="assistant",
            content=content,
            refusal=refusal,
            reasoning_content=reasoning_content,
        ),
    )


def test_non_string_blocking_content_is_not_stored_as_text() -> None:
    response = ChatCompletion.model_construct(
        id="resp-1",
        choices=[_completion_choice(content=[{"type": "text", "text": "hi"}])],
        created=1,
        model="gateway-model",
        object="chat.completion",
    )

    parsed = _client()._parse_response_from_openai(response, {})

    assert parsed.messages[0].contents == []
    assert parsed.messages[0].text == ""


def test_non_string_stream_content_is_not_stored_as_text() -> None:
    chunk = ChatCompletionChunk.model_construct(
        id="chunk-1",
        choices=[_chunk_choice(content=[{"type": "text", "text": "hi"}])],
        created=1,
        model="gateway-model",
        object="chat.completion.chunk",
        usage=None,
    )

    parsed = _client()._parse_response_update_from_openai(chunk)

    assert parsed.contents == []
    assert parsed.text == ""


def test_non_string_stream_content_does_not_fall_through_to_refusal() -> None:
    choice = _chunk_choice(content=[{"type": "thinking", "text": "private"}], refusal="blocked")

    assert _client()._parse_text_from_openai(choice) is None


def test_non_string_stream_reasoning_delta_is_not_stored_as_reasoning_text() -> None:
    chunk = ChatCompletionChunk.model_construct(
        id="chunk-1",
        choices=[_chunk_choice(reasoning_content=[{"type": "thinking", "text": "private"}])],
        created=1,
        model="gateway-model",
        object="chat.completion.chunk",
        usage=None,
    )

    update = _client()._parse_response_update_from_openai(chunk)
    response = ChatResponse.from_updates([update])

    assert update.contents == []
    assert response.raw_text == ""


def test_anthropic_redacted_reasoning_is_not_replayed_as_chat_completions_reasoning() -> None:
    redacted = Content.from_text_reasoning(
        protected_data='{"private":"opaque"}',
        additional_properties={"anthropic_redacted_thinking": True},
    )

    prepared = _client()._prepare_message_for_openai(Message("assistant", [redacted, Content.from_text("answer")]))

    assert prepared == [{"role": "assistant", "content": "answer"}]
