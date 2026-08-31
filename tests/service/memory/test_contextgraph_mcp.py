# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for the read-only ContextGraph MCP bridge."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from chrys.service.memory import contextgraph_mcp as memory
from chrys.service.profiles.agents.loader import load_profile_from_yaml


@pytest.fixture(autouse=True)
def _reset_client() -> Iterator[None]:
    previous = memory._CLIENT
    memory._CLIENT = None
    try:
        yield
    finally:
        if memory._CLIENT is not None:
            memory._CLIENT.close()
        memory._CLIENT = previous


def _set_client(handler) -> None:
    memory._CLIENT = httpx.Client(transport=httpx.MockTransport(handler), base_url=memory.DEFAULT_BASE_URL)


def test_health_reports_neo4j_connected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"neo4j_connected": True})

    _set_client(handler)

    assert memory._do_health() == "ContextGraph query service is healthy (neo4j_connected=true)."


def test_health_reports_neo4j_disconnected() -> None:
    _set_client(lambda _request: httpx.Response(200, json={"neo4j_connected": False}))

    assert memory._do_health() == "ContextGraph query service is reachable, but Neo4j is not connected."


def test_query_maps_to_items_endpoint_and_frames_untrusted_data() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "items": [
                    {"id": "rule-1", "text": "Inspect the existing tests before editing."},
                    {"id": "rule-2", "text": "Run focused tests after each logical change."},
                ]
            },
        )

    _set_client(handler)
    result = memory._do_query("port a memory integration", top_k=7)

    assert seen == {
        "path": "/query_memory_items",
        "body": {
            "query": "port a memory integration",
            "task_description": "port a memory integration",
            "top_k": 7,
        },
    }
    assert "UNTRUSTED DATA" in result
    assert "Inspect the existing tests" in result
    assert "Run focused tests" in result


def test_query_sanitizes_controls_and_deduplicates_items() -> None:
    _set_client(
        lambda _request: httpx.Response(
            200,
            json={
                "items": [
                    {"text": "\x1b[31mcheck tests\x00"},
                    {"text": "[31mcheck tests"},
                    {"text": ""},
                    "not-a-mapping",
                ]
            },
        )
    )

    result = memory._do_query("task")

    assert "\x1b" not in result and "\x00" not in result
    assert result.count("[31mcheck tests") == 1


def test_query_empty_and_failures_fail_open() -> None:
    _set_client(lambda _request: httpx.Response(500))

    assert memory._do_query("") == "No prior ContextGraph memory found."
    assert memory._do_query("anything") == "No prior ContextGraph memory found."


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("not-an-int", memory.DEFAULT_TOP_K),
        (0, 1),
        (memory.MAX_TOP_K + 10, memory.MAX_TOP_K),
    ],
)
def test_query_clamps_top_k(requested: int | str, expected: int) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"items": []})

    _set_client(handler)

    assert memory._do_query("anything", top_k=requested) == "No prior ContextGraph memory found."
    assert seen["top_k"] == expected


def test_query_result_respects_note_cap() -> None:
    _set_client(
        lambda _request: httpx.Response(
            200,
            json={"items": [{"text": f"rule-{index}: " + "x" * 1000} for index in range(20)]},
        )
    )

    result = memory._do_query("task", top_k=20)

    assert len(result) <= memory.MAX_NOTE_CHARS
    assert "rule-0" in result
    assert "rule-19" not in result


def test_example_profile_loads_with_read_only_memory_tools() -> None:
    profile_path = Path(__file__).parents[3] / "examples" / "contextgraph-memory" / "Memory.yaml"

    profile = load_profile_from_yaml(profile_path)

    assert len(profile.tools.mcp) == 1
    assert profile.tools.mcp[0].allowed_tools == ["team_memory_health", "team_memory_query"]
    assert "team_memory_record" not in profile.approval.overrides
