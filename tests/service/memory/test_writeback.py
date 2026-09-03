# Copyright (c) 2026 Chrys. All rights reserved.

"""Watermark-driven deposition of completed turns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.kernel import Content, Message
from chrys.service.memory.writeback import WritebackOutcome, count_turns, deposit_pending_turns
from chrys.service.state.serializers import serialize_state


def _write_session(path: Path, messages: list[Message]) -> None:
    path.write_text(
        json.dumps({"state": serialize_state({"messages": messages, "compressed_msgs": [], "turn_counter": 1})}),
        encoding="utf-8",
    )


def _tool_turn(index: int) -> list[Message]:
    return [
        Message("user", [Content.from_text(f"Fix failure {index}")]),
        Message("assistant", [Content.from_function_call(f"shell-{index}", "shell", arguments={"command": "ls"})]),
        Message("tool", [Content.from_function_result(f"shell-{index}", result=f"done {index}")]),
        Message("assistant", [Content.from_text(f"Fixed {index}.")]),
    ]


@pytest.fixture
def three_turns(tmp_path: Path) -> Path:
    session_file = tmp_path / "session.json"
    messages: list[Message] = []
    for index in (1, 2, 3):
        messages.extend(_tool_turn(index))
    _write_session(session_file, messages)
    return session_file


def test_counts_turns(three_turns: Path) -> None:
    assert count_turns(three_turns) == 3


def test_deposits_only_the_turns_after_the_watermark(three_turns: Path) -> None:
    seen: list[str] = []

    def _deposit(**kwargs: Any) -> object:
        seen.append(kwargs["source_id"])
        return object()

    outcome = deposit_pending_turns(three_turns, watermark=1, repo="r", source_prefix="s", deposit=_deposit)

    assert outcome == WritebackOutcome(deposited=(2, 3), failed=None, watermark=3)
    assert [item.split(":")[1] for item in seen] == ["2", "3"]
    assert all(item.startswith("s:") for item in seen)


def test_a_second_pass_at_the_new_watermark_deposits_nothing(three_turns: Path) -> None:
    calls: list[str] = []

    def _deposit(**kwargs: Any) -> object:
        calls.append(kwargs["source_id"])
        return object()

    outcome = deposit_pending_turns(three_turns, watermark=3, repo="r", source_prefix="s", deposit=_deposit)

    assert outcome == WritebackOutcome(deposited=(), failed=None, watermark=3)
    assert calls == []


def test_stops_at_the_first_failure_and_holds_the_watermark(three_turns: Path) -> None:
    def _deposit(**kwargs: Any) -> object:
        if kwargs["source_id"].split(":")[1] == "2":
            raise RuntimeError("neo4j down")
        return object()

    outcome = deposit_pending_turns(three_turns, watermark=0, repo="r", source_prefix="s", deposit=_deposit)

    # Turn 3 must not be deposited past a hole: the watermark is a high-water
    # mark, so advancing over turn 2 would lose it forever.
    assert outcome == WritebackOutcome(deposited=(1,), failed=2, watermark=1)


def test_source_ids_are_stable_so_replay_is_idempotent(three_turns: Path) -> None:
    def _collect(**kwargs: Any) -> object:
        collected.append(kwargs["source_id"])
        return object()

    collected: list[str] = []
    deposit_pending_turns(three_turns, watermark=0, repo="r", source_prefix="s", deposit=_collect)
    first = list(collected)
    collected.clear()
    deposit_pending_turns(three_turns, watermark=0, repo="r", source_prefix="s", deposit=_collect)

    assert collected == first


def test_a_turn_with_no_tool_work_advances_the_watermark_without_depositing(tmp_path: Path) -> None:
    session_file = tmp_path / "session.json"
    _write_session(
        session_file,
        [
            Message("user", [Content.from_text("hello")]),
            Message("assistant", [Content.from_text("hi")]),
        ],
    )
    calls: list[str] = []

    def _deposit(**kwargs: Any) -> object:
        calls.append(kwargs["source_id"])
        return object()

    outcome = deposit_pending_turns(session_file, watermark=0, repo="r", source_prefix="s", deposit=_deposit)

    # Nothing was written, so nothing is reported as deposited -- but the
    # watermark still advances: there is nothing here to retry.
    assert calls == []
    assert outcome == WritebackOutcome(deposited=(), failed=None, watermark=1)


def test_an_interrupted_turn_is_deposited_as_a_failure(tmp_path: Path) -> None:
    session_file = tmp_path / "session.json"
    interrupted = Message("assistant", [Content.from_text("Execution interrupted")])
    interrupted.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.INTERRUPTED
    interrupted.additional_properties[HistoryMarkerKind.STATUS_CODE_KEY] = (
        HistoryMarkerKind.STATUS_EXECUTION_INTERRUPTED
    )
    _write_session(session_file, [*_tool_turn(1), interrupted])
    seen: list[bool] = []

    def _deposit(**kwargs: Any) -> object:
        seen.append(kwargs["success"])
        return object()

    deposit_pending_turns(session_file, watermark=0, repo="r", source_prefix="s", deposit=_deposit)

    assert seen == [False]


def test_a_clean_turn_is_deposited_as_a_success(three_turns: Path) -> None:
    seen: list[bool] = []

    def _deposit(**kwargs: Any) -> object:
        seen.append(kwargs["success"])
        return object()

    deposit_pending_turns(three_turns, watermark=2, repo="r", source_prefix="s", deposit=_deposit)

    assert seen == [True]


def test_a_missing_session_file_counts_as_no_turns(tmp_path: Path) -> None:
    assert count_turns(tmp_path / "absent.json") == 0
    outcome = deposit_pending_turns(
        tmp_path / "absent.json", watermark=0, repo="r", source_prefix="s", deposit=lambda **_: object()
    )
    assert outcome == WritebackOutcome(deposited=(), failed=None, watermark=0)
