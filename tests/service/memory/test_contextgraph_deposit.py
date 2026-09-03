# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for persisted-turn extraction and ContextGraph deposition."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chrys.kernel import Content, Message
from chrys.service.memory import contextgraph_deposit as deposit
from chrys.service.memory.contextgraph_repository import RepositoryDepositResult
from chrys.service.state.serializers import serialize_state


def _write_session(path: Path, messages: list[Message]) -> None:
    path.write_text(
        json.dumps({"state": serialize_state({"messages": messages, "compressed_msgs": [], "turn_counter": 1})}),
        encoding="utf-8",
    )


def _tool_turn() -> list[Message]:
    return [
        Message("user", [Content.from_text("Fix the failing parser tests")]),
        Message(
            "assistant",
            [Content.from_function_call("memory-1", "team_memory_query", arguments={"query": "parser tests"})],
        ),
        Message("tool", [Content.from_function_result("memory-1", result="untrusted prior memory")]),
        Message(
            "assistant",
            [
                Content.from_function_call(
                    "shell-1",
                    "shell",
                    arguments={"command": "uv run pytest tests/parser.py", "api_key": "must-not-copy"},
                )
            ],
        ),
        Message("tool", [Content.from_function_result("shell-1", result="3 passed")]),
        Message("assistant", [Content.from_text("Implemented the parser fix and verified it.")]),
    ]


def test_extract_turn_pairs_calls_and_excludes_memory_recursion(tmp_path: Path) -> None:
    session_file = tmp_path / "session.json"
    _write_session(session_file, _tool_turn())

    extracted = deposit.extract_turn_experience(session_file, 1)

    assert extracted is not None
    assert extracted.problem_statement == "Fix the failing parser tests"
    assert extracted.final_response == "Implemented the parser fix and verified it."
    assert extracted.steps == (
        {
            "action": 'shell {"command": "uv run pytest tests/parser.py"}',
            "observation": "3 passed",
        },
    )
    assert len(extracted.turn_digest) == 64
    assert "untrusted prior memory" not in str(extracted.steps)
    assert "must-not-copy" not in str(extracted.steps)


def test_extract_turn_rejects_answer_only_record(tmp_path: Path) -> None:
    session_file = tmp_path / "session.json"
    _write_session(
        session_file,
        [
            Message("user", [Content.from_text("Explain this code")]),
            Message("assistant", [Content.from_text("Here is the explanation")]),
        ],
    )

    assert deposit.extract_turn_experience(session_file, 1) is None


def test_deposit_hook_payload_binds_session_turn_and_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session_file = tmp_path / "session.json"
    _write_session(session_file, _tool_turn())
    seen: dict[str, object] = {}

    def fake_record(**kwargs):
        seen.update(kwargs)
        return RepositoryDepositResult(trajectory_id="traj_chrys_test", fragment_count=1, created=True)

    monkeypatch.setattr(deposit, "_session_file", lambda _session_id: session_file)
    monkeypatch.setattr(deposit, "deposit_experience", fake_record)

    result = deposit.deposit_hook_payload(
        {
            "session_id": "abc123",
            "turn": 1,
            "cwd": str(tmp_path / "chrys"),
            "status": "ok",
        }
    )

    assert result == RepositoryDepositResult(trajectory_id="traj_chrys_test", fragment_count=1, created=True)
    assert seen["repo"] == "chrys"
    assert seen["success"] is True
    assert seen["final_response"] == "Implemented the parser fix and verified it."
    assert str(seen["source_id"]).startswith("chrys-after-turn:abc123:1:")


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "../escape", "turn": 1},
        {"session_id": "abc", "turn": True},
        {"session_id": "abc", "turn": "1"},
    ],
)
def test_deposit_hook_payload_rejects_invalid_identity(payload: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="invalid"):
        deposit.deposit_hook_payload(payload)


# ── route and campaign semantics ─────────────────────────────────────


def _route_marker(record: dict[str, object]) -> Message:
    marker = Message("assistant", [Content.from_text("")])
    marker.additional_properties["_chrys_route"] = record
    return marker


def test_a_standard_turn_keeps_marker_derived_success(tmp_path: Path) -> None:
    session_file = tmp_path / "session.json"
    _write_session(
        session_file,
        [*_tool_turn(), _route_marker({"track": "standard", "baseline": "none", "campaign": None})],
    )

    extracted = deposit.extract_turn_experience(session_file, 1)

    assert extracted is not None
    assert extracted.route == "standard"
    assert extracted.campaign_status == ""
    assert extracted.success is True


def test_a_completed_campaign_is_a_verified_success(tmp_path: Path) -> None:
    """The campaign ran the repository's verify command; a clean exit did not."""
    session_file = tmp_path / "session.json"
    _write_session(
        session_file,
        [
            *_tool_turn(),
            _route_marker(
                {
                    "track": "long_horizon",
                    "baseline": "p1",
                    "campaign": {"status": "completed", "campaign_id": "c1"},
                }
            ),
        ],
    )

    extracted = deposit.extract_turn_experience(session_file, 1)

    assert extracted is not None
    assert extracted.route == "long_horizon"
    assert extracted.campaign_status == "completed"
    assert extracted.success is True


def test_a_blocked_campaign_is_recorded_as_a_failure(tmp_path: Path) -> None:
    """A turn that exited cleanly but did not finish the work is not a success."""
    session_file = tmp_path / "session.json"
    _write_session(
        session_file,
        [
            *_tool_turn(),
            _route_marker({"track": "long_horizon", "baseline": "p1", "campaign": {"status": "blocked"}}),
        ],
    )

    extracted = deposit.extract_turn_experience(session_file, 1)

    assert extracted is not None
    assert extracted.campaign_status == "blocked"
    assert extracted.success is False


def test_a_long_horizon_turn_without_a_campaign_uses_the_markers(tmp_path: Path) -> None:
    session_file = tmp_path / "session.json"
    _write_session(
        session_file,
        [*_tool_turn(), _route_marker({"track": "long_horizon", "baseline": "p1", "campaign": None})],
    )

    extracted = deposit.extract_turn_experience(session_file, 1)

    assert extracted is not None
    assert extracted.campaign_status == ""
    assert extracted.success is True


def test_the_clarified_requirement_replaces_the_raw_prompt(tmp_path: Path) -> None:
    """The clarified requirement is what the recorded steps actually solved."""
    session_file = tmp_path / "session.json"
    _write_session(session_file, _tool_turn())
    outcome = tmp_path / "requirement_clarification" / "turn_1" / "05-outcome"
    outcome.mkdir(parents=True)
    (outcome / "clarified-requirement.md").write_text("Fix the parser so it accepts trailing commas.", encoding="utf-8")

    extracted = deposit.extract_turn_experience(session_file, 1)

    assert extracted is not None
    assert extracted.problem_statement == "Fix the parser so it accepts trailing commas."


def test_a_missing_clarified_requirement_keeps_the_prompt(tmp_path: Path) -> None:
    session_file = tmp_path / "session.json"
    _write_session(session_file, _tool_turn())

    extracted = deposit.extract_turn_experience(session_file, 1)

    assert extracted is not None
    assert extracted.problem_statement == "Fix the failing parser tests"
