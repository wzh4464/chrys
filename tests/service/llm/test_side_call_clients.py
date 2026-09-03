# Copyright (c) 2026 Chrys. All rights reserved.

"""Session-scoped reuse of side-call model clients."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from chrys.service.llm.side_call_clients import SideCallClientCache
from chrys.service.profiles.models.schema import ModelProfile


def _profile(identifier: str) -> ModelProfile:
    return ModelProfile(id=identifier, name=identifier)


def test_the_same_profile_is_created_once() -> None:
    cache = SideCallClientCache()

    with patch(
        "chrys.service.llm.side_call_clients.create_client", side_effect=lambda *a, **k: MagicMock()
    ) as create_client:
        first = cache.get(_profile("cheap"))
        second = cache.get(_profile("cheap"))

    assert first is second
    assert create_client.call_count == 1


def test_different_profiles_get_different_clients() -> None:
    cache = SideCallClientCache()

    with patch("chrys.service.llm.side_call_clients.create_client", side_effect=lambda *a, **k: MagicMock()):
        cheap = cache.get(_profile("cheap"))
        main = cache.get(_profile("main"))

    assert cheap is not main


def test_the_route_session_context_is_requested() -> None:
    cache = SideCallClientCache()

    with patch("chrys.service.llm.side_call_clients.create_client", return_value=MagicMock()) as create_client:
        cache.get(_profile("cheap"), session_id="s", parent_session_id="p")

    assert create_client.call_args.kwargs["use_route_session_context"] is True
    assert create_client.call_args.kwargs["session_id"] == "s"
    assert create_client.call_args.kwargs["parent_session_id"] == "p"


class _Client:
    """A close-able client double.

    Deliberately not a MagicMock: a mock auto-creates ``aclose``, which the
    cache prefers, so a mock would never exercise the synchronous path.
    """

    def __init__(self, identifier: str, closed: list[str], *, fail: bool = False) -> None:
        self._identifier = identifier
        self._closed = closed
        self._fail = fail

    def close(self) -> None:
        if self._fail:
            msg = "connection already gone"
            raise RuntimeError(msg)
        self._closed.append(self._identifier)


class _AsyncClient:
    def __init__(self, identifier: str, closed: list[str]) -> None:
        self._identifier = identifier
        self._closed = closed

    async def aclose(self) -> None:
        self._closed.append(self._identifier)


async def test_close_releases_every_client_and_empties_the_cache() -> None:
    cache = SideCallClientCache()
    closed: list[str] = []

    with patch(
        "chrys.service.llm.side_call_clients.create_client",
        side_effect=lambda profile, **_kwargs: _Client(profile.id, closed),
    ) as create_client:
        cache.get(_profile("a"))
        cache.get(_profile("b"))
        await cache.close()
        # A get after close creates a fresh client rather than reviving a closed one.
        cache.get(_profile("a"))

    assert sorted(closed) == ["a", "b"]
    assert create_client.call_count == 3


async def test_an_async_close_is_awaited() -> None:
    cache = SideCallClientCache()
    closed: list[str] = []

    with patch(
        "chrys.service.llm.side_call_clients.create_client",
        side_effect=lambda profile, **_kwargs: _AsyncClient(profile.id, closed),
    ):
        cache.get(_profile("a"))
        await cache.close()

    assert closed == ["a"]


async def test_one_failing_close_does_not_strand_the_others() -> None:
    """Shutdown must release every client even when a provider is unreachable."""
    cache = SideCallClientCache()
    closed: list[str] = []

    def _make(profile: ModelProfile, **_kwargs: Any) -> Any:
        return _Client(profile.id, closed, fail=profile.id == "broken")

    with patch("chrys.service.llm.side_call_clients.create_client", side_effect=_make):
        cache.get(_profile("broken"))
        cache.get(_profile("healthy"))
        await cache.close()

    assert closed == ["healthy"]


async def test_a_client_without_a_close_method_is_skipped() -> None:
    cache = SideCallClientCache()

    class _Bare:
        pass

    with patch("chrys.service.llm.side_call_clients.create_client", return_value=_Bare()):
        cache.get(_profile("bare"))
        await cache.close()
