# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the fail-closed outbound model lock."""

from __future__ import annotations

import json

import pytest

from chrys.service.llm.clients import create_client
from chrys.service.llm.model_lock import MODEL_LOCK_ENV, ModelLockError, enforce_model_lock
from chrys.service.profiles.models.schema import ModelProfile


def _profile(**overrides: str) -> ModelProfile:
    values = {
        "id": "deepseek-v4-pro-0813-openrouter",
        "name": "DeepSeek V4 Pro 0813 (OpenRouter)",
        "provider": "openai",
        "api_style": "chat_completions",
        "model_id": "deepseek/deepseek-v4-pro-0813",
        "base_url": "https://openrouter.ai/api/v1",
    }
    values.update(overrides)
    return ModelProfile(**values)  # type: ignore[arg-type]


def _lock(**overrides: str) -> str:
    values = {
        "provider": "openai",
        "api_style": "chat_completions",
        "base_url": "https://openrouter.ai/api/v1",
        "model_id": "deepseek/deepseek-v4-pro-0813",
    }
    values.update(overrides)
    return json.dumps(values)


def test_absent_model_lock_allows_normal_profile_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_LOCK_ENV, raising=False)

    enforce_model_lock(_profile(model_id="another/model"), effective_base_url="https://example.com/v1")


def test_matching_model_lock_allows_exact_wire_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_LOCK_ENV, _lock(base_url="https://openrouter.ai/api/v1/"))

    enforce_model_lock(_profile(), effective_base_url="https://openrouter.ai/api/v1")


def test_client_factory_enforces_lock_before_credentials_or_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MODEL_LOCK_ENV, _lock())

    with pytest.raises(ModelLockError, match="blocked"):
        create_client(_profile(model_id="another/model", api_key="{{MISSING_API_KEY}}"))


@pytest.mark.parametrize(
    ("profile_override", "effective_base_url"),
    [
        ({"provider": "anthropic"}, "https://openrouter.ai/api/v1"),
        ({"api_style": "responses"}, "https://openrouter.ai/api/v1"),
        ({"model_id": "deepseek/deepseek-v4-flash"}, "https://openrouter.ai/api/v1"),
        ({}, "https://api.openai.com/v1"),
    ],
)
def test_model_lock_rejects_every_wire_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    profile_override: dict[str, str],
    effective_base_url: str,
) -> None:
    monkeypatch.setenv(MODEL_LOCK_ENV, _lock())

    with pytest.raises(ModelLockError, match="blocked"):
        enforce_model_lock(_profile(**profile_override), effective_base_url=effective_base_url)


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not-json",
        "[]",
        json.dumps({"model_id": "deepseek/deepseek-v4-pro-0813"}),
        _lock(extra="unexpected"),
    ],
)
def test_present_but_invalid_model_lock_fails_closed(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(MODEL_LOCK_ENV, value)

    with pytest.raises(ModelLockError):
        enforce_model_lock(_profile(), effective_base_url="https://openrouter.ai/api/v1")
