# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for direct read-only ContextGraph Neo4j retrieval."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from neo4j import READ_ACCESS

from chrys.service.memory import contextgraph_mcp as memory
from chrys.service.profiles.agents.loader import load_profile_from_yaml


@pytest.fixture(autouse=True)
def _reset_clients() -> Iterator[None]:
    previous_driver = memory._DRIVER
    previous_embedding_client = memory._EMBEDDING_CLIENT
    memory._DRIVER = None
    memory._EMBEDDING_CLIENT = None
    try:
        yield
    finally:
        memory._DRIVER = previous_driver
        memory._EMBEDDING_CLIENT = previous_embedding_client


def test_run_read_uses_read_access_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeRecord:
        def data(self) -> dict[str, object]:
            return {"value": 1}

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def run(self, cypher: object, parameters: dict[str, object]):
            seen["cypher"] = str(cypher)
            seen["parameters"] = parameters
            return [FakeRecord()]

    class FakeDriver:
        def session(self, **kwargs: object) -> FakeSession:
            seen.update(kwargs)
            return FakeSession()

    fake_driver = FakeDriver()
    monkeypatch.setattr(memory, "_driver", lambda: fake_driver)

    assert memory._run_read("RETURN $value", {"value": 1}) == [{"value": 1}]
    assert seen == {
        "default_access_mode": READ_ACCESS,
        "cypher": "RETURN $value",
        "parameters": {"value": 1},
    }


def test_health_reports_canonical_rule_count(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDriver:
        connected = False

        def verify_connectivity(self) -> None:
            self.connected = True

    driver = FakeDriver()
    monkeypatch.setattr(memory, "_driver", lambda: driver)
    monkeypatch.setattr(memory, "_run_read", lambda *_args, **_kwargs: [{"count": 2525}])

    assert memory._do_health() == "ContextGraph Neo4j is healthy (canonical_rules=2525)."
    assert driver.connected


def test_query_fuses_vector_and_fulltext_results(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, dict[str, object]]] = []

    def fake_run(cypher: str, parameters: dict[str, object]):
        seen.append((cypher, parameters))
        if "vector.queryNodes" in cypher:
            return [
                {"id": "rule-a", "text": "Inspect tests before editing."},
                {"id": "rule-b", "text": "Run focused tests after each change."},
            ]
        return [
            {"id": "rule-b", "text": "Run focused tests after each change."},
            {"id": "rule-c", "text": "Check every related call site."},
        ]

    monkeypatch.setattr(memory, "_embed", lambda _query: [0.1, 0.2])
    monkeypatch.setattr(memory, "_run_read", fake_run)

    result = memory._do_query("port a memory integration", top_k=3)

    assert "UNTRUSTED DATA" in result
    assert result.index("Run focused tests") < result.index("Inspect tests")
    assert "Check every related call site" in result
    assert len(seen) == 2
    assert seen[0][1] == {"limit": 15, "embedding": [0.1, 0.2]}
    assert seen[1][1] == {"limit": 15, "query": "port memory integration"}


def test_query_uses_fulltext_when_embedding_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_run(cypher: str, _parameters: dict[str, object]):
        seen.append(cypher)
        return [{"id": "rule-a", "text": "Use repository-native tests."}]

    monkeypatch.setattr(memory, "_embed", lambda _query: None)
    monkeypatch.setattr(memory, "_run_read", fake_run)

    result = memory._do_query("repository tests")

    assert "Use repository-native tests" in result
    assert len(seen) == 1 and "fulltext.queryNodes" in seen[0]


def test_query_sanitizes_controls_and_deduplicates_items(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "_embed", lambda _query: None)
    monkeypatch.setattr(
        memory,
        "_run_read",
        lambda *_args, **_kwargs: [
            {"id": "rule-a", "text": "\x1b[31mcheck tests\x00"},
            {"id": "rule-b", "text": "[31mcheck tests"},
            {"id": "rule-empty", "text": ""},
        ],
    )

    result = memory._do_query("task")

    assert "\x1b" not in result and "\x00" not in result
    assert result.count("[31mcheck tests") == 1


def test_query_empty_and_channel_failures_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(memory, "_embed", lambda _query: (_ for _ in ()).throw(RuntimeError("embedding down")))
    monkeypatch.setattr(memory, "_run_read", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("db down")))

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
def test_clamp_top_k(requested: int | str, expected: int) -> None:
    assert memory._clamp_top_k(requested) == expected


def test_query_result_respects_note_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        memory,
        "_search_vector",
        lambda *_args: [memory._Hit(key=f"rule-{index}", text=f"rule-{index}: " + "x" * 1000) for index in range(20)],
    )
    monkeypatch.setattr(memory, "_search_fulltext", lambda *_args: [])

    result = memory._do_query("task", top_k=20)

    assert len(result) <= memory.MAX_NOTE_CHARS
    assert "rule-0" in result
    assert "rule-19" not in result


def test_environment_prefers_contextgraph_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://fallback:7687")
    monkeypatch.setenv("NEO4J_USER", "fallback-user")
    monkeypatch.setenv("NEO4J_PASSWORD", "fallback-password")
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_URI", "bolt://contextgraph:7705")
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_USER", "contextgraph-user")
    monkeypatch.setenv("CONTEXTGRAPH_NEO4J_PASSWORD", "contextgraph-password")

    assert memory._neo4j_uri() == "bolt://contextgraph:7705"
    assert memory._neo4j_auth() == ("contextgraph-user", "contextgraph-password")


def test_example_profile_loads_with_read_only_memory_tools() -> None:
    profile_path = Path(__file__).parents[3] / "examples" / "contextgraph-memory" / "Memory.yaml"

    profile = load_profile_from_yaml(profile_path)

    assert len(profile.tools.mcp) == 1
    assert profile.tools.mcp[0].allowed_tools == ["team_memory_health", "team_memory_query"]
    assert "team_memory_record" not in profile.approval.overrides
