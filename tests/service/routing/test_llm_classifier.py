# Copyright (c) 2026 Chrys. All rights reserved.

"""The one guarded model call the router is allowed to make."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.routing.classifier import extract_prompt_signals
from chrys.service.routing.guard import TiebreakerGuard
from chrys.service.routing.llm import LlmRouteClassifier, TiebreakerVerdict

_PROMPT = "refactor the entire auth system"


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.messages: list[Any] = []


class _FakeClient:
    """Return one scripted response, or raise, or hang."""

    def __init__(self, *, text: str = "", error: Exception | None = None, hang: bool = False) -> None:
        self._text = text
        self._error = error
        self._hang = hang
        self.calls: list[list[Any]] = []
        self.options: list[Any] = []

    async def get_response(self, messages: list[Any], stream: bool = False, options: Any = None, **_: Any) -> Any:
        self.calls.append(list(messages))
        self.options.append(options)
        if self._hang:
            await asyncio.sleep(3600)
        if self._error is not None:
            raise self._error
        return _FakeResponse(self._text)


def _profile() -> ModelProfile:
    return ModelProfile(id="cheap", name="Cheap", stream=False)


def _classifier(client: _FakeClient, *, guard: TiebreakerGuard | None = None, timeout: float = 5.0):
    return LlmRouteClassifier(
        _profile(),
        guard=guard or TiebreakerGuard(),
        session_id="s",
        parent_session_id="m",
        session_dir=None,
        client=client,
        timeout_seconds=timeout,
    )


async def _classify(classifier) -> TiebreakerVerdict:
    return await classifier.classify(_PROMPT, extract_prompt_signals(_PROMPT))


async def test_a_json_verdict_is_parsed() -> None:
    client = _FakeClient(text='{"long_horizon": true, "confidence": 0.9, "reason": "multi-module"}')

    verdict = await _classify(_classifier(client))

    assert verdict == TiebreakerVerdict(long_horizon=True, confidence=0.9, reason="multi-module")


async def test_a_fenced_verdict_is_parsed() -> None:
    client = _FakeClient(
        text='Sure!\n```json\n{"long_horizon": false, "confidence": 0.8, "reason": "single file"}\n```'
    )

    verdict = await _classify(_classifier(client))

    assert verdict.long_horizon is False
    assert verdict.confidence == pytest.approx(0.8)
    assert verdict.failure == ""


async def test_prose_without_json_is_a_malformed_failure() -> None:
    client = _FakeClient(text="I think this is a big task, honestly.")

    verdict = await _classify(_classifier(client))

    assert verdict.failure == "malformed"
    assert verdict.long_horizon is False
    assert verdict.confidence == 0.0


async def test_a_hanging_model_is_a_timeout_failure() -> None:
    client = _FakeClient(hang=True)

    verdict = await _classify(_classifier(client, timeout=0.05))

    assert verdict.failure == "timeout"
    assert verdict.long_horizon is False


async def test_a_raising_client_is_an_unavailable_failure() -> None:
    client = _FakeClient(error=RuntimeError("model lock mismatch"))

    verdict = await _classify(_classifier(client))

    assert verdict.failure == "unavailable"
    assert verdict.long_horizon is False


async def test_the_guard_short_circuits_before_any_call() -> None:
    guard = TiebreakerGuard(max_calls=0)
    client = _FakeClient(text='{"long_horizon": true, "confidence": 0.9, "reason": "x"}')

    verdict = await _classify(_classifier(client, guard=guard))

    assert verdict.failure == "rate_limited"
    assert client.calls == []


async def test_an_open_breaker_short_circuits_before_any_call() -> None:
    guard = TiebreakerGuard(trip_after=1)
    guard.record_failure()
    client = _FakeClient(text='{"long_horizon": true, "confidence": 0.9, "reason": "x"}')

    verdict = await _classify(_classifier(client, guard=guard))

    assert verdict.failure == "circuit_open"
    assert client.calls == []


async def test_a_successful_verdict_is_recorded_on_the_guard() -> None:
    guard = TiebreakerGuard()
    client = _FakeClient(text='{"long_horizon": true, "confidence": 0.9, "reason": "x"}')

    await _classify(_classifier(client, guard=guard))

    assert guard.calls == 1
    assert guard.allow() == (True, "")


async def test_a_failed_verdict_is_recorded_as_a_failure() -> None:
    guard = TiebreakerGuard(trip_after=1)
    client = _FakeClient(error=RuntimeError("boom"))

    await _classify(_classifier(client, guard=guard))

    assert guard.allow() == (False, "circuit_open")


async def test_a_malformed_verdict_counts_as_a_failure() -> None:
    """A model that cannot produce JSON will not start on the next turn."""
    guard = TiebreakerGuard(trip_after=1)
    client = _FakeClient(text="no json here")

    await _classify(_classifier(client, guard=guard))

    assert guard.allow() == (False, "circuit_open")


async def test_confidence_is_clamped_and_non_numeric_is_malformed() -> None:
    high = await _classify(_classifier(_FakeClient(text='{"long_horizon": true, "confidence": 5, "reason": "x"}')))
    assert high.confidence == pytest.approx(1.0)

    low = await _classify(_classifier(_FakeClient(text='{"long_horizon": true, "confidence": -2, "reason": "x"}')))
    assert low.confidence == pytest.approx(0.0)

    bad = await _classify(_classifier(_FakeClient(text='{"long_horizon": true, "confidence": "high"}')))
    assert bad.failure == "malformed"


async def test_the_prompt_carries_the_heuristic_signals() -> None:
    client = _FakeClient(text='{"long_horizon": false, "confidence": 0.9, "reason": "x"}')

    await _classify(_classifier(client))

    rendered = "\n".join(str(content) for message in client.calls[0] for content in message.contents)
    assert "mutating_broad" in rendered
    assert _PROMPT in rendered


async def test_output_tokens_are_capped_for_a_one_line_verdict() -> None:
    client = _FakeClient(text='{"long_horizon": false, "confidence": 0.9, "reason": "x"}')

    await _classify(_classifier(client))

    assert client.options[0]["max_output_tokens"] == 150


def test_a_client_is_created_lazily_with_the_route_session_context() -> None:
    classifier = LlmRouteClassifier(
        _profile(),
        guard=TiebreakerGuard(),
        session_id="sess-route",
        parent_session_id="parent-route",
        session_dir=None,
    )

    with patch("chrys.service.llm.clients.create_client", return_value=MagicMock()) as create_client:
        classifier._get_client()
        classifier._get_client()

    create_client.assert_called_once()
    assert create_client.call_args.kwargs["session_id"] == "sess-route"
    assert create_client.call_args.kwargs["parent_session_id"] == "parent-route"
