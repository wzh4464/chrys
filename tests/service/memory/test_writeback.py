# Copyright (c) 2026 Chrys. All rights reserved.

"""Watermark-driven deposition of completed turns."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.kernel import Content, Message
from chrys.service.context.providers.history import CompressedBlock
from chrys.service.memory.writeback import (
    WritebackOutcome,
    deposit_pending_turns,
    session_turn_numbers,
)
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
    assert session_turn_numbers(three_turns) == [1, 2, 3]


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
    assert session_turn_numbers(tmp_path / "absent.json") == []
    outcome = deposit_pending_turns(
        tmp_path / "absent.json", watermark=0, repo="r", source_prefix="s", deposit=lambda **_: object()
    )
    assert outcome == WritebackOutcome(deposited=(), failed=None, watermark=0)


# --------------------------------------------------------------------------
# compaction: the live list stops being an index of turns
# --------------------------------------------------------------------------


def _marked(index: int) -> list[Message]:
    """A finalized turn, carrying the turn marker the engine writes."""
    marker = Message("assistant", [Content.from_text("")])
    marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    marker.additional_properties["_turn_id"] = f"turn_{index}"
    marker.additional_properties["_turn"] = index
    return [*_tool_turn(index), marker]


def _write_compacted(path: Path, messages: list[Message], *, folded_through: int) -> None:
    """A session whose first *folded_through* turns are only in a compressed block."""
    summary = Message("assistant", [Content.from_text("[Compressed context: c1]\nSummary: earlier work")])
    summary.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.SUMMARY
    path.write_text(
        json.dumps(
            {
                "state": serialize_state(
                    {
                        "messages": [summary, *messages],
                        "compressed_msgs": [
                            CompressedBlock(
                                compressed_context_id="c1",
                                summary_text="earlier work",
                                marker_id=f"turn_{folded_through}",
                                turn_range=(1, folded_through),
                            )
                        ],
                        "turn_counter": folded_through + len(messages),
                    }
                )
            }
        ),
        encoding="utf-8",
    )


def test_turns_are_identified_by_their_marker_not_their_position(tmp_path: Path) -> None:
    """After a fold the fourth slice is turn 9, and the graph has to agree."""
    session_file = tmp_path / "session.json"
    _write_compacted(session_file, [*_marked(6), *_marked(7), *_marked(8), *_marked(9)], folded_through=5)

    assert session_turn_numbers(session_file) == [6, 7, 8, 9]


def test_compaction_never_re_deposits_turns_as_new(tmp_path: Path) -> None:
    """The mark counts turns, so a shrinking live list must not pull it back."""
    session_file = tmp_path / "session.json"
    _write_compacted(session_file, [*_marked(6), *_marked(7)], folded_through=5)
    seen: list[str] = []

    outcome = deposit_pending_turns(
        session_file,
        watermark=7,
        repo="r",
        source_prefix="s",
        deposit=lambda **kwargs: seen.append(kwargs["source_id"]),
    )

    assert seen == []
    assert outcome == WritebackOutcome(deposited=(), failed=None, watermark=7)


def test_a_backlog_survives_compaction_without_being_written_off(tmp_path: Path) -> None:
    """The undeposited turns that are still here must still be deposited.

    A positional mark would have called turns 6 and 7 done — they are the first
    two slices, and the mark said two — and deposited 8 and 9 in their place.
    """
    session_file = tmp_path / "session.json"
    _write_compacted(session_file, [*_marked(6), *_marked(7), *_marked(8), *_marked(9)], folded_through=5)
    seen: list[int] = []

    outcome = deposit_pending_turns(
        session_file,
        watermark=2,
        repo="r",
        source_prefix="s",
        deposit=lambda **kwargs: seen.append(int(kwargs["source_id"].split(":")[1])),
    )

    assert seen == [6, 7, 8, 9]
    assert outcome.watermark == 9


def test_an_unmarked_span_after_a_fold_is_skipped_rather_than_guessed(tmp_path: Path) -> None:
    """Only a marker can say which turn a span is once positions have shifted."""
    session_file = tmp_path / "session.json"
    _write_compacted(session_file, [*_marked(6), *_tool_turn(99)], folded_through=5)

    assert session_turn_numbers(session_file) == [6]
