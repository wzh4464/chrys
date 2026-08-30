# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for Buddy notification LLM calls."""

from __future__ import annotations

import pytest

from chrys.app.features.buddy.notification import generate_ai_pet_response_async
from chrys.app.features.buddy.types import Species
from chrys.foundation.config.settings import Settings
from chrys.service.llm.route_sessions import derive_llm_route_session_id
from chrys.service.profiles.models.resolver import default_profile


class _Companion:
    name = "Biscuit"
    species = Species.DUCK
    personality = "gentle"


class _Response:
    text = "hello"


class _Stream:
    def __init__(self) -> None:
        self._done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._done:
            raise StopAsyncIteration
        self._done = True
        return object()

    async def get_final_response(self):
        return _Response()


class _Client:
    def __init__(self) -> None:
        self.streams: list[bool] = []

    async def get_response(self, _messages, *, stream=False, **_kwargs):
        self.streams.append(stream)
        if stream:
            return _Stream()
        return _Response()


@pytest.mark.asyncio
async def test_buddy_response_uses_model_profile_stream_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = default_profile()
    profile.stream = True
    client = _Client()

    monkeypatch.setattr("chrys.orchestration.engine.engine.get_current_engine", lambda: None)
    monkeypatch.setattr(
        "chrys.service.profiles.models.resolver.resolve_active_profile", lambda _registry, _settings: profile
    )
    monkeypatch.setattr("chrys.service.llm.clients.create_client", lambda _profile, **_kwargs: client)

    response = await generate_ai_pet_response_async(_Companion(), [])

    assert response.endswith("hello")
    assert client.streams == [True]


@pytest.mark.asyncio
async def test_buddy_response_honours_the_pet_model_override_without_an_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The override is configuration, not session state.

    The buddy fires before any session exists (hatch, idle nudges), and that is
    precisely when there is no engine to read it from — so resolving it only
    from the engine would make the setting silently inert exactly where it is
    most visible.
    """
    profile = default_profile()
    client = _Client()
    captured: dict[str, object] = {}

    async def _get_response(_messages, *, stream=False, options=None, **_kwargs):
        captured["options"] = options
        return _Response()

    client.get_response = _get_response  # type: ignore[method-assign]
    monkeypatch.setenv("CHRYS_PET_MODEL", "pet-mini")
    monkeypatch.setattr("chrys.orchestration.engine.engine.get_current_engine", lambda: None)
    monkeypatch.setattr(
        "chrys.service.profiles.models.resolver.resolve_active_profile", lambda _registry, _settings: profile
    )
    monkeypatch.setattr("chrys.service.llm.clients.create_client", lambda _profile, **_kwargs: client)

    response = await generate_ai_pet_response_async(_Companion(), [])

    assert response.endswith("hello")
    assert captured["options"]["model"] == "pet-mini"  # type: ignore[index]


@pytest.mark.asyncio
async def test_buddy_response_passes_current_session_context_to_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    profile = default_profile()
    client = _Client()
    captured: dict[str, object] = {}

    class _Engine:
        active_model_profile = profile
        session_id = "sess-buddy"
        session_dir = tmp_path
        model_registry = None
        settings = Settings()

    def _create_client(_profile, *, session_id=None, parent_session_id=None, session_dir=None):
        captured["session_id"] = session_id
        captured["parent_session_id"] = parent_session_id
        captured["session_dir"] = session_dir
        return client

    def _get_current_engine():
        return _Engine()

    monkeypatch.setattr("chrys.orchestration.engine.engine.get_current_engine", _get_current_engine)
    monkeypatch.setattr("chrys.service.llm.clients.create_client", _create_client)

    response = await generate_ai_pet_response_async(_Companion(), [])

    assert response.endswith("hello")
    assert captured["session_id"] == derive_llm_route_session_id(
        "sess-buddy",
        route_kind="buddy-notify",
        model_profile=profile,
    )
    assert captured["parent_session_id"] == "sess-buddy"
    assert captured["session_dir"] == tmp_path
