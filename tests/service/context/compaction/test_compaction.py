# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for UnifiedContextStrategy (turn-aware four-phase algorithm)."""

from __future__ import annotations

import ast
import asyncio
import gc
import inspect
import os
import textwrap
import weakref
from collections.abc import Callable, Mapping
from io import BytesIO
from pathlib import Path
from threading import Event
from typing import Any

import pytest
from PIL import Image

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.retry import restore_message_properties, snapshot_message_properties
from chrys.foundation.trajectory.context import trajectory_scope
from chrys.foundation.trajectory.envelope import SEGMENT_EVENT_TYPE, EventDraft
from chrys.foundation.trajectory.event_types import EventType as TrajectoryEventType
from chrys.foundation.trajectory.writer import EmitResult
from chrys.kernel import (
    EXCLUDE_REASON_KEY,
    EXCLUDED_KEY,
    Agent,
    AgentSession,
    ChatResponse,
    Content,
    FunctionTool,
    Message,
    SessionContext,
    annotate_message_groups,
    annotate_token_counts,
    included_token_count,
    project_included_messages,
    set_excluded,
)
from chrys.kernel import compaction as chrys_compaction
from chrys.kernel.client import _clamp_output_cap_for_context
from chrys.kernel.loop import _wire_message_view
from chrys.service.agent_middleware.system_reminder import ManifestEntry, SystemReminderMiddleware
from chrys.service.context import compaction as compaction_mod
from chrys.service.context.compaction import (
    _REASON_COMPRESSION,
    _REASON_CURRENT_TURN_DROP,
    CompactionInfo,
    CompressInfo,
    MixedLanguageTokenizer,
    PreCompactInfo,
    UnifiedContextStrategy,
    _build_summary,
    _content_signature,
    _dedup_message_ids,
    _ExclusionAnchor,
    _group_id,
    _group_kind_map,
    _group_messages_by_id,
    _ordered_group_ids,
    _resolve_turns,
    _set_summarized,
)
from chrys.service.context.compaction.current_turn_drop import CurrentTurnDropRound
from chrys.service.context.compaction.last_words import LastWordsSpendBudgetExceeded
from chrys.service.context.compaction.scoped import DEGRADED_SCOPED_PREAMBLE
from chrys.service.context.compaction.spill import (
    SpillBatchResult,
    SpillManifestItem,
    SpillQuota,
    catalog_live_records,
)
from chrys.service.context.compaction.summaries import _SUMMARY_TOTAL_MAX
from chrys.service.context.compaction.turns import _state_correspondence
from chrys.service.context.providers.history import (
    PRE_OUTPUT_HISTORY_LEN_STATE_KEY,
    CompressedBlock,
    CompressibleHistoryProvider,
)
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.session.message_metadata import LAST_ASSISTANT_CREATED_AT_STATE_KEY, MESSAGE_CREATED_AT_KEY
from chrys.service.session.runtime_metadata import SessionRuntimeMetadata
from tests.service.trajectory._fakes import FakeSink, make_context
from tests.support.phase4_stubs import StubLastWordsGenerator as _StubLastWordsGenerator
from tests.support.phase4_stubs import StubReminderMiddleware as _StubReminderMiddleware
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for, wait_until

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_tokenizer = MixedLanguageTokenizer()


def _async_appender(target: list) -> object:
    """Create an async callback that appends to *target*."""

    async def _cb(info: CompactionInfo) -> None:
        target.append(info)

    return _cb


def _user(text: str) -> Message:
    return Message(role="user", contents=[Content.from_text(text)])


def _assistant_text(text: str) -> Message:
    return Message(role="assistant", contents=[Content.from_text(text)])


def _assistant_tool_call(call_id: str, name: str, args: dict | None = None) -> Message:
    return Message(
        role="assistant",
        contents=[Content.from_function_call(call_id, name, arguments=args or {})],
    )


def _tool_result(call_id: str, result: str) -> Message:
    return Message(
        role="tool",
        contents=[Content.from_function_result(call_id, result=result)],
    )


def _build_tool_group(call_id: str, name: str, result: str, args: dict | None = None) -> list[Message]:
    """Build a tool-call group: assistant function_call + tool result."""
    return [
        _assistant_tool_call(call_id, name, args=args),
        _tool_result(call_id, result),
    ]


def _has_call_id(msg: Message, call_id: str) -> bool:
    return any(c.call_id == call_id for c in msg.contents if getattr(c, "call_id", None))


def _estimate_tokens(messages: list[Message]) -> int:
    annotate_message_groups(messages)
    annotate_token_counts(messages, tokenizer=_tokenizer)
    return included_token_count(messages)


def _generated_png_bytes(size: tuple[int, int] = (512, 512)) -> bytes:
    image = Image.effect_noise(size, 100).convert("RGB")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _build_single_turn(num_groups: int, result_size: int = 2000) -> list[Message]:
    """Build a single-turn message list: user + N tool groups + assistant text."""
    messages = [_user("start")]
    for i in range(num_groups):
        messages.extend(_build_tool_group(f"call_{i}", f"tool_{i}", "x" * result_size))
    messages.append(_assistant_text("done"))
    return messages


def _build_multi_turn(turns: int, groups_per_turn: int = 3, result_size: int = 2000) -> list[Message]:
    """Build a multi-turn message list.

    Each turn: user + N tool groups + assistant text.
    """
    messages: list[Message] = []
    for t in range(turns):
        messages.append(_user(f"Turn {t + 1} request"))
        for g in range(groups_per_turn):
            cid = f"t{t}_call_{g}"
            messages.extend(_build_tool_group(cid, f"tool_{g}", "x" * result_size))
        messages.append(_assistant_text(f"Turn {t + 1} response"))
    return messages


def _make_strategy(
    *,
    last_words_generator: object | None = None,
    reminder_middleware: object | None = None,
    **kwargs,
) -> UnifiedContextStrategy:
    """Create a strategy with Phase 4 collaborators stubbed by default.

    Tests that want to observe the LAST_WORDS pipeline can pass their own
    stub instances; tests that don't care still get drop-all Phase 4
    behaviour out of the box.
    """
    defaults = {
        "max_context_tokens": 100_000,
        "trigger_pct": 0.85,
        "target_pct": 0.50,
    }
    defaults.update(kwargs)
    strategy = UnifiedContextStrategy(**defaults)
    strategy.set_last_words_generator(last_words_generator or _StubLastWordsGenerator())
    strategy.set_reminder_middleware(reminder_middleware or _StubReminderMiddleware())
    return strategy


def _failed_spill_result(reason: str = "I/O failure") -> SpillBatchResult:
    return SpillBatchResult(
        entries=(
            SpillManifestItem(
                record_id="",
                group_id="failed-group",
                record_dir="compactions/dropped/turn001",
                relative_path="",
                turn=1,
                round=1,
                sequence=1,
                tool="tool",
                display_argument="",
                outcome="unknown",
                size_chars=0,
                available=False,
                no_record_reason=reason,
            ),
        ),
        unexpected_io_failure=True,
    )


def _scoped_messages(call: dict) -> list[Message]:
    return [message for group in call["scoped_groups"] for message in group.messages]


def _scoped_user_texts(call: dict) -> list[str]:
    return [message.text for group in call["scoped_groups"] if group.kind == "user" for message in group.messages]


def test_phase4_default_side_call_budget_is_unlimited() -> None:
    strategy = UnifiedContextStrategy(max_context_tokens=12_345)

    assert strategy._phase4_side_call_token_budget == -1
    # None (legacy call sites) normalizes to unlimited too.
    legacy = UnifiedContextStrategy(max_context_tokens=100, phase4_side_call_token_budget=None)
    assert legacy._phase4_side_call_token_budget == -1


# ---------------------------------------------------------------------------
# Turn resolution tests (§3.2)
# ---------------------------------------------------------------------------


def _injected(text: str) -> Message:
    msg = _user(text)
    msg.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    return msg


def _nudge(text: str = "continue") -> Message:
    msg = _user(text)
    msg.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
    return msg


def _turn_marker(turn_index: int) -> Message:
    msg = _assistant_text("")
    msg.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    msg.additional_properties["_turn_id"] = f"turn_{turn_index}"
    msg.additional_properties["_turn"] = turn_index
    return msg


def _status_marker(kind: str) -> Message:
    msg = _assistant_text("")
    msg.additional_properties[HistoryMarkerKind.KEY] = kind
    return msg


def _wire_view(state: dict) -> list[Message]:
    """Marker-less wire view, mirroring ``CompressibleHistoryProvider.get_messages``."""
    return [
        m
        for m in state["messages"]
        if m.additional_properties.get(HistoryMarkerKind.KEY) != HistoryMarkerKind.TURN
        and not m.additional_properties.get(EXCLUDED_KEY, False)
        and not (m.role == "user" and m.additional_properties.get(HistoryMarkerKind.CONTINUATION_KEY))
    ]


def _spans(resolved) -> list[tuple[int, int]]:
    return [(s.start, s.end) for s in resolved.spans]


class TestResolveTurnsStrict:
    """Stateless (unbound-strategy) resolution keeps legacy strict slicing."""

    def test_empty_messages(self) -> None:
        resolved = _resolve_turns([], None)
        assert resolved.spans == []
        assert not resolved.degraded

    def test_single_turn(self) -> None:
        resolved = _resolve_turns([_user("hello"), _assistant_text("hi")], None)
        assert _spans(resolved) == [(0, 2)]
        assert resolved.current.absolute_number == 1

    def test_multiple_turns(self) -> None:
        msgs = [
            _user("q1"),
            _assistant_text("a1"),
            _user("q2"),
            _assistant_text("a2"),
            _user("q3"),
            _assistant_text("a3"),
        ]
        resolved = _resolve_turns(msgs, None)
        assert _spans(resolved) == [(0, 2), (2, 4), (4, 6)]
        assert [s.absolute_number for s in resolved.spans] == [1, 2, 3]
        assert not resolved.degraded

    def test_prefix_before_first_opener(self) -> None:
        """Compressed summaries before the first opener are not part of any turn."""
        summary = _assistant_text("[Compressed context: ctx_abc]")
        resolved = _resolve_turns([summary, _user("q1"), _assistant_text("a1")], None)
        assert _spans(resolved) == [(1, 3)]

    def test_no_user_messages_rule5(self) -> None:
        resolved = _resolve_turns([_assistant_text("hello"), _assistant_text("world")], None)
        assert resolved.spans == []
        assert not resolved.degraded

    def test_injected_and_nudge_do_not_split(self) -> None:
        """An injected turn is ONE slice — the §2.1 bug family fix."""
        msgs = [
            _user("q1"),
            _assistant_text("a1"),
            _user("q2"),
            _assistant_text("working"),
            _injected("mid-turn note"),
            _assistant_text("more work"),
            _nudge(),
            _assistant_text("done"),
        ]
        resolved = _resolve_turns(msgs, None)
        assert _spans(resolved) == [(0, 2), (2, 8)]
        assert not resolved.degraded

    def test_legacy_unflagged_guidance_still_splits(self) -> None:
        """Documented degradation (§6): pre-flag histories keep the old split."""
        msgs = [_user("q1"), _assistant_text("a1"), _user("continue"), _assistant_text("a2")]
        resolved = _resolve_turns(msgs, None)
        assert _spans(resolved) == [(0, 2), (2, 4)]

    def test_excluded_opener_does_not_reopen_turn(self) -> None:
        """A mid-call fold (Step 0 / P3) excludes a completed turn in place;
        its opener must not keep opening a turn the state no longer holds."""
        folded_opener = _user("q1")
        folded_opener.additional_properties[EXCLUDED_KEY] = True
        folded_answer = _assistant_text("a1")
        folded_answer.additional_properties[EXCLUDED_KEY] = True
        msgs = [folded_opener, folded_answer, _user("q2"), _assistant_text("a2")]
        resolved = _resolve_turns(msgs, None)
        assert _spans(resolved) == [(2, 4)]
        assert not resolved.degraded

    def test_excluded_opener_degrades_to_flagged_tail(self) -> None:
        """Post-fold, a flagged-only visible tail degrades — the note quotes
        the guidance, never the folded opener."""
        folded_opener = _user("q1")
        folded_opener.additional_properties[EXCLUDED_KEY] = True
        folded_answer = _assistant_text("a1")
        folded_answer.additional_properties[EXCLUDED_KEY] = True
        guidance = _injected("resume with flag X")
        msgs = [folded_opener, folded_answer, guidance, _assistant_text("work")]
        resolved = _resolve_turns(msgs, None)
        assert resolved.degraded
        assert _spans(resolved) == [(0, 4)]
        assert resolved.synthetic_opener_idx == 2

    def test_legacy_split_coexists_with_flagged_turns(self) -> None:
        """A restored pre-change session still splits at its unflagged resume
        guidance while later flagged turns resolve as single slices (§6)."""
        msgs = [
            _user("q1"),
            _assistant_text("a1"),
            _user("continue"),  # legacy unflagged resume guidance
            _assistant_text("a2"),
            _user("q2"),
            _assistant_text("working"),
            _injected("mid-turn note"),
            _nudge(),
            _assistant_text("done"),
        ]
        resolved = _resolve_turns(msgs, None)
        assert _spans(resolved) == [(0, 2), (2, 4), (4, 9)]
        assert not resolved.degraded


class TestResolveTurnsWithState:
    """State-backed resolution: history-segment clip, marker regions, numbering."""

    def test_provider_prefix_user_opens_no_span(self) -> None:
        """§3.2 rule 1: a provider prefix user message never opens a turn."""
        opener1 = _user("q1")
        answer1 = _assistant_text("a1")
        opener2 = _user("q2")
        state = {"messages": [opener1, answer1, _turn_marker(1), opener2]}
        prefix_user = _user("provider context prompt")  # context_management.py shape
        wire = [prefix_user, *_wire_view(state)]

        resolved = _resolve_turns(wire, state)

        assert _spans(resolved) == [(1, 3), (3, 4)]
        assert [s.absolute_number for s in resolved.spans] == [1, 2]

    def test_mixed_shape_degrades_per_marker_region(self) -> None:
        """opener → work → marker → flagged tail: completed turn stays previous."""
        opener1 = _user("q1")
        answer1 = _assistant_text("a1")
        guidance = _injected("try again with flag X")
        state = {"messages": [opener1, answer1, _turn_marker(1), guidance]}
        wire = _wire_view(state)

        resolved = _resolve_turns(wire, state)

        assert resolved.degraded
        assert resolved.region_start == 2
        assert _spans(resolved) == [(0, 2), (2, 3)]
        assert resolved.synthetic_opener_idx == 2
        assert [s.absolute_number for s in resolved.spans] == [1, 2]

    def test_crash_cluster_keeps_turn_current(self) -> None:
        """§3.2 rule 1: an unhealthy cluster does not close a completed turn."""
        opener1 = _user("q1")
        answer1 = _assistant_text("a1")
        opener2 = _user("q2")
        work2 = _assistant_text("work2")
        guidance = _injected("guidance after crash")
        state = {
            "messages": [
                opener1,
                answer1,
                _turn_marker(1),
                opener2,
                work2,
                _status_marker(HistoryMarkerKind.INTERRUPTED),
                _turn_marker(2),
                guidance,
            ]
        }
        wire = _wire_view(state)

        resolved = _resolve_turns(wire, state)

        # ONE current turn spanning opener2 through the guidance — the
        # boundary walk skips the crash-terminated cluster, so P1/P2 never
        # see work2 as a previous turn.  (The INTERRUPTED status message is
        # not a turn marker and stays on the wire view.)
        assert _spans(resolved) == [(0, 2), (2, 6)]
        assert not resolved.degraded
        # turn_counter would say 3; the resolver refuses to close turn 2.
        assert resolved.current.absolute_number == 2

    def test_awaiting_sub_agents_cluster_keeps_turn_current(self) -> None:
        """Health = STATUS_MARKERS, not an interrupted/error enumeration."""
        opener1 = _user("q1")
        work1 = _assistant_text("work1")
        state = {
            "messages": [
                opener1,
                work1,
                _status_marker(HistoryMarkerKind.AWAITING_SUB_AGENTS),
                _turn_marker(1),
            ]
        }
        wire = _wire_view(state)

        resolved = _resolve_turns(wire, state)

        assert _spans(resolved) == [(0, 3)]
        assert resolved.current.absolute_number == 1

    def test_all_flagged_marker_less_history_degrades(self) -> None:
        """The no-prior-user resume shape: degraded slice spans the whole wire."""
        work = _assistant_text("interrupted tool work")
        nudge = _nudge()
        # A live nudge travels as the run's input prefix — on the wire but
        # not yet in state.
        state = {"messages": [work]}
        wire = [work, nudge]

        resolved = _resolve_turns(wire, state)

        assert resolved.degraded
        assert _spans(resolved) == [(0, 2)]
        assert resolved.synthetic_opener_idx == 1
        assert resolved.current.absolute_number == 1

    def test_work_only_region_behind_completed_turns(self) -> None:
        """§3.2 rule 4: assistant/tool-only region degrades with no synthetic opener."""
        opener1 = _user("q1")
        answer1 = _assistant_text("a1")
        work2 = _assistant_text("work2")
        state = {"messages": [opener1, answer1, _turn_marker(1), work2]}
        wire = _wire_view(state)

        resolved = _resolve_turns(wire, state)

        assert resolved.degraded
        assert _spans(resolved) == [(0, 2), (2, 3)]
        assert resolved.synthetic_opener_idx is None

    def test_hidden_duplicate_id_does_not_move_region(self) -> None:
        """§3.2 rule 1: the id fallback never lands on a pre-marker EXCLUDED
        state message that reuses a visible current-turn message's id."""
        opener1 = _user("q1")
        excluded_old = _assistant_text("old excluded work")
        excluded_old.message_id = "msg_1"
        excluded_old.additional_properties[EXCLUDED_KEY] = True
        opener2 = _user("q2")
        current_work = _assistant_text("current work")
        current_work.message_id = "msg_1"  # cross-run id reuse
        state = {"messages": [opener1, excluded_old, _turn_marker(1), opener2, current_work]}
        # The current-turn work reaches the wire as a REBUILT object
        # (streaming finalizer) — identity misses, the id fallback engages.
        rebuilt_work = _assistant_text("current work")
        rebuilt_work.message_id = "msg_1"
        wire = [opener1, opener2, rebuilt_work]

        resolved = _resolve_turns(wire, state)

        # A naive id lookup would map the rebuilt work onto the excluded
        # pre-marker message (state idx 1 ≤ marker) and jump region_start
        # past live work; the guarded fallback maps it to its state twin.
        assert _spans(resolved) == [(0, 1), (1, 3)]
        assert resolved.region_start == 1
        assert resolved.current.state_boundary == 3

    def test_falsy_marker_is_excluded_from_message_id_correspondence(self) -> None:
        """A malformed marker cannot shadow a real message with the same positional id."""
        real = _assistant_text("real")
        real.message_id = "msg_1"
        marker = _assistant_text("marker")
        marker.message_id = "msg_1"
        marker.additional_properties[HistoryMarkerKind.KEY] = ""
        rebuilt = _assistant_text("real")
        rebuilt.message_id = "msg_1"

        assert _state_correspondence([rebuilt], [real, marker]) == {0: 0}

    def test_identity_broken_twin_uses_guarded_id_fallback(self) -> None:
        """Rebuilt wire objects resolve via message_id with role agreement."""
        opener1 = _user("q1")
        opener1.message_id = "msg_a"
        answer1 = _assistant_text("a1")
        answer1.message_id = "msg_b"
        opener2 = _user("q2")
        opener2.message_id = "msg_c"
        state = {"messages": [opener1, answer1, _turn_marker(1), opener2]}
        # The wire carries REBUILT copies (fresh objects, same ids).
        wire_opener1 = _user("q1")
        wire_opener1.message_id = "msg_a"
        wire_answer1 = _assistant_text("a1")
        wire_answer1.message_id = "msg_b"
        wire_opener2 = _user("q2")
        wire_opener2.message_id = "msg_c"
        wire = [wire_opener1, wire_answer1, wire_opener2]

        resolved = _resolve_turns(wire, state)

        assert _spans(resolved) == [(0, 2), (2, 3)]
        assert resolved.region_start == 2
        assert resolved.current.state_boundary == 3

    def test_wire_view_wrappers_resolve_completed_turns(self) -> None:
        """The kernel wire hands per-call message VIEWS — fresh wrapper +
        fresh contents list, shared content objects.  With no message_ids
        anywhere, correspondence must still hold via shared content-object
        identity, or every completed turn silently drops out of resolution."""
        opener1 = _user("q1")
        answer1 = _assistant_text("a1")
        opener2 = _user("q2")
        state = {"messages": [opener1, answer1, _turn_marker(1), opener2]}
        wire = [_wire_message_view(m) for m in _wire_view(state)]
        assert all(
            w is not s and w.contents is not s.contents for w, s in zip(wire, [opener1, answer1, opener2], strict=True)
        )

        resolved = _resolve_turns(wire, state)

        assert _spans(resolved) == [(0, 2), (2, 3)]
        assert resolved.region_start == 2
        assert resolved.previous[0].state_boundary == 0
        assert resolved.current.state_boundary == 3

    def test_fresh_prompt_boundary_absent_from_state(self) -> None:
        """A fresh prompt's opener is not stored yet: state_boundary is None."""
        opener1 = _user("q1")
        answer1 = _assistant_text("a1")
        state = {"messages": [opener1, answer1, _turn_marker(1)]}
        fresh_opener = _user("q2")  # in-flight input, unsaved
        wire = [*_wire_view(state), fresh_opener]

        resolved = _resolve_turns(wire, state)

        assert _spans(resolved) == [(0, 2), (2, 3)]
        assert resolved.current.state_boundary is None
        assert resolved.previous[0].state_boundary == 0

    def test_numbering_seeds_from_folded_turn_range(self) -> None:
        """A current turn behind a fully folded prefix numbers max_folded + 1."""
        summary = _assistant_text("[Compressed context: ctx_old]")
        summary.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.SUMMARY
        opener = _user("current question")
        state = {
            "messages": [summary, opener],
            "compressed_msgs": [
                CompressedBlock(compressed_context_id="ctx_sentinel", turn_range=(0, 0)),
                CompressedBlock(compressed_context_id="ctx_old", turn_range=(1, 4)),
            ],
        }
        wire = _wire_view(state)

        resolved = _resolve_turns(wire, state)

        assert _spans(resolved) == [(1, 2)]
        assert resolved.current.absolute_number == 5


@pytest.mark.asyncio
async def test_compaction_reports_absolute_turn_numbers():
    """After compress_context removes early turns, Phase 1 must report
    absolute turn numbers, not relative enumeration indices.

    The compacted list is the marker-LESS wire view, so the numbers come
    from the resolver's state-side marker scan seeded with the folded
    ``turn_range`` (§4.1) — never from wire-embedded markers.
    """
    # Simulate: turns 1-2 folded into a compressed block, turns 3-4-5 active.
    compressed_summary = _assistant_text("[Compressed context: ctx_abc]")
    compressed_summary.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.SUMMARY

    state_messages: list[Message] = [compressed_summary]
    # Turn 3 (first visible)
    state_messages.append(_user("Turn 3 question"))
    state_messages.extend(_build_tool_group("t3_c0", "search", "x" * 3000))
    state_messages.extend(_build_tool_group("t3_c1", "read_file", "x" * 3000))
    state_messages.append(_assistant_text("Turn 3 answer"))
    state_messages.append(_turn_marker(3))
    # Turn 4
    state_messages.append(_user("Turn 4 question"))
    state_messages.extend(_build_tool_group("t4_c0", "grep", "x" * 3000))
    state_messages.extend(_build_tool_group("t4_c1", "read_file", "x" * 3000))
    state_messages.append(_assistant_text("Turn 4 answer"))
    state_messages.append(_turn_marker(4))
    # Turn 5 (current)
    state_messages.append(_user("Turn 5 question"))
    state_messages.extend(_build_tool_group("t5_c0", "glob", "x" * 1000))
    state_messages.append(_assistant_text("Turn 5 answer"))

    state = {
        "messages": state_messages,
        "compressed_msgs": [CompressedBlock(compressed_context_id="ctx_abc", turn_range=(1, 2))],
    }
    messages = _wire_view(state)

    total = _estimate_tokens(messages)
    events: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.90,
        target_pct=0.50,
        on_compaction=_async_appender(events),
    )
    strategy.bind_state(state)
    changed = await strategy(messages)
    assert changed

    p1_events = [e for e in events if e.phase == "phase1"]
    assert p1_events, "Phase 1 should have fired"
    # Must report absolute turn numbers 3, 4 — NOT relative 1, 2
    assert 3 in p1_events[0].turn_numbers
    assert 1 not in p1_events[0].turn_numbers, (
        f"Reported relative turn 1 instead of absolute; got {p1_events[0].turn_numbers}"
    )


# ---------------------------------------------------------------------------
# Basic strategy behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_compaction_under_trigger():
    """No changes when usage is under trigger_pct."""
    messages = [_user("hello"), _assistant_text("hi")]
    strategy = _make_strategy(max_context_tokens=1_000_000)
    changed = await strategy(messages)
    assert not changed


@pytest.mark.asyncio
async def test_png_attachment_does_not_trigger_auto_compaction_at_low_usage():
    """A large PNG payload is budgeted as image tokens, not base64 text."""
    data = _generated_png_bytes()
    assert len(data) > 300_000
    messages = [
        _user("x" * 40_000),
        _assistant_text("answered"),
        Message(
            role="user",
            contents=[
                Content.from_text("describe this @clipboard-image.png"),
                Content.from_data(data, "image/png"),
            ],
        ),
    ]
    assert chrys_compaction.estimate_message_image_tokens(messages[-1]) == 255
    events: list[CompactionInfo] = []
    strategy = _make_strategy(max_context_tokens=100_000, on_compaction=_async_appender(events))

    changed = await strategy(messages)

    assert not changed
    assert events == []
    assert strategy.last_included_tokens < 20_000


@pytest.mark.asyncio
async def test_multiple_png_attachments_do_not_trigger_auto_compaction_at_low_usage():
    """Multiple PNG attachments are summed as image tokens, not base64 text."""
    data = _generated_png_bytes()
    assert len(data) > 300_000
    messages = [
        _user("x" * 40_000),
        _assistant_text("answered"),
        Message(
            role="user",
            contents=[
                Content.from_text("compare @first.png and @second.png"),
                Content.from_data(data, "image/png"),
                Content.from_data(data, "image/png"),
            ],
        ),
    ]
    assert chrys_compaction.estimate_message_image_tokens(messages[-1]) == 510
    events: list[CompactionInfo] = []
    strategy = _make_strategy(max_context_tokens=100_000, on_compaction=_async_appender(events))

    changed = await strategy(messages)

    assert not changed
    assert events == []
    assert strategy.last_included_tokens < 25_000


@pytest.mark.asyncio
async def test_no_tool_groups_no_change():
    """No-op when there are no tool-call groups, even if over trigger."""
    messages = [_user("hello"), _assistant_text("world " * 500)]
    strategy = _make_strategy(max_context_tokens=100, trigger_pct=0.01, target_pct=0.005)
    changed = await strategy(messages)
    assert not changed


@pytest.mark.asyncio
async def test_text_only_history_emergency_compresses_completed_turns_before_request():
    """Text-heavy history with markers must be folded before the next model call.

    The legacy behaviour is still covered by ``test_no_tool_groups_no_change``:
    without bound history state/markers there is nothing safe to fold.  This
    test exercises the real session shape, where completed turns are present
    in provider state but filtered out of the wire list.
    """
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    for idx in range(5):
        state["messages"].append(_user(f"Turn {idx + 1} request " + ("x " * 2000)))
        state["messages"].append(_assistant_text(f"Turn {idx + 1} answer " + ("y " * 2000)))
        CompressibleHistoryProvider.insert_marker(state, idx + 1)

    messages = [
        msg
        for msg in state["messages"]
        if msg.additional_properties.get(HistoryMarkerKind.KEY) != HistoryMarkerKind.TURN
    ]
    messages.append(_user("Current turn"))
    tokens_before = _estimate_tokens(messages)
    compressed_events = []
    precompact_events: list[PreCompactInfo] = []

    async def _on_compress(info) -> None:
        compressed_events.append(info)

    strategy = _make_strategy(
        max_context_tokens=tokens_before + 100,
        trigger_pct=0.85,
        target_pct=0.50,
        on_compress=_on_compress,
        on_pre_compact=_async_appender(precompact_events),
    )
    strategy.bind_state(state)

    changed = await strategy(messages)

    assert changed
    assert precompact_events
    triggers = [info.trigger for info in precompact_events]
    assert "phase3" in triggers
    assert "force" not in triggers
    assert compressed_events
    assert all(info.source == "auto" for info in compressed_events)
    assert state["compressed_msgs"]
    projected = project_included_messages(messages)
    assert any(msg.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.SUMMARY for msg in projected)
    assert included_token_count(messages) < tokens_before
    assert strategy._usage_pct(included_token_count(messages)) <= strategy.target_pct

    second_call_messages = [
        msg for msg in messages if msg.additional_properties.get(HistoryMarkerKind.KEY) != HistoryMarkerKind.SUMMARY
    ]
    assert not any(
        msg.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.SUMMARY
        for msg in second_call_messages
    )

    await strategy(second_call_messages)

    assert any(
        msg.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.SUMMARY
        for msg in project_included_messages(second_call_messages)
    )


@pytest.mark.asyncio
async def test_text_only_history_emergency_compresses_completed_turns_with_wire_views():
    """Same emergency shape, but through the kernel's real per-call VIEWS.

    ``_wire_message_view`` mints a fresh wrapper + fresh contents list for
    every outgoing message, so wrapper identity and ``id(msg.contents)``
    both miss state; only the content OBJECTS are shared.  Phase 3 must
    still fold completed turns, and same-run reinjection must recognise
    brand-new views of the folded originals on the next call.
    """
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    for idx in range(5):
        state["messages"].append(_user(f"Turn {idx + 1} request " + ("x " * 2000)))
        state["messages"].append(_assistant_text(f"Turn {idx + 1} answer " + ("y " * 2000)))
        CompressibleHistoryProvider.insert_marker(state, idx + 1)

    wire_source = [
        msg
        for msg in state["messages"]
        if msg.additional_properties.get(HistoryMarkerKind.KEY) != HistoryMarkerKind.TURN
    ]
    wire_source.append(_user("Current turn"))
    messages = [_wire_message_view(msg) for msg in wire_source]
    tokens_before = _estimate_tokens(messages)
    precompact_events: list[PreCompactInfo] = []

    strategy = _make_strategy(
        max_context_tokens=tokens_before + 100,
        trigger_pct=0.85,
        target_pct=0.50,
        on_pre_compact=_async_appender(precompact_events),
    )
    strategy.bind_state(state)

    changed = await strategy(messages)

    assert changed
    assert "phase3" in [info.trigger for info in precompact_events]
    assert state["compressed_msgs"]
    projected = project_included_messages(messages)
    assert any(msg.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.SUMMARY for msg in projected)
    assert included_token_count(messages) < tokens_before
    assert strategy._usage_pct(included_token_count(messages)) <= strategy.target_pct

    # The next call gets FRESH views of the same originals (per-call wire
    # semantics): the folded ones carry their exclusion through the shared
    # props dict, and the cached summary must reinject against pure view
    # copies that share nothing but content objects with call one.
    second_call_messages = [_wire_message_view(msg) for msg in wire_source]
    assert not any(
        msg.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.SUMMARY
        for msg in second_call_messages
    )

    await strategy(second_call_messages)

    assert any(
        msg.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.SUMMARY
        for msg in project_included_messages(second_call_messages)
    )
    assert strategy._compressed_context_cache
    assert all(len(entry.folded_contents) > 0 for entry in strategy._compressed_context_cache.values())

    # The next bind is a run boundary and clears the cache outright.
    strategy.bind_state(state)
    assert not strategy._compressed_context_cache


@pytest.mark.asyncio
async def test_retry_restore_silently_replays_published_phase3_folds_in_order() -> None:
    """Published completed-turn folds survive attempt rollback with stable blocks."""
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    for idx in range(5):
        state["messages"].append(_user(f"Turn {idx + 1} request " + ("x " * 2_000)))
        state["messages"].append(_assistant_text(f"Turn {idx + 1} answer " + ("y " * 2_000)))
        CompressibleHistoryProvider.insert_marker(state, idx + 1)

    baseline_messages = list(state["messages"])
    baseline_properties = snapshot_message_properties(baseline_messages)
    current_user = _user("Current turn")
    wire = [
        _wire_message_view(message)
        for message in [
            *(
                message
                for message in baseline_messages
                if message.additional_properties.get(HistoryMarkerKind.KEY) != HistoryMarkerKind.TURN
            ),
            current_user,
        ]
    ]
    tokens_before = _estimate_tokens(wire)
    compressed_events: list[CompressInfo] = []
    strategy = _make_strategy(
        max_context_tokens=tokens_before + 100,
        trigger_pct=0.85,
        target_pct=0.20,
        on_compress=_async_appender(compressed_events),
    )
    strategy.bind_state(state)
    retry_snapshot = strategy.snapshot_retry_state()

    assert await strategy(wire)
    published_blocks = list(state["compressed_msgs"])
    published_ids = [block.compressed_context_id for block in published_blocks]
    assert len(published_ids) >= 2  # Exercise ordered replay, not only one fold.
    assert [event.compressed_context_id for event in compressed_events] == published_ids

    # Mirror Executor._restore_history: restore the pre-input baseline first,
    # then let the strategy silently preserve committed completed-turn folds.
    state["messages"] = list(baseline_messages)
    state["compressed_msgs"].clear()
    state.pop(PRE_OUTPUT_HISTORY_LEN_STATE_KEY, None)
    restore_message_properties(state["messages"], baseline_properties)
    strategy.restore_retry_state(retry_snapshot)

    restored_blocks = state["compressed_msgs"]
    assert restored_blocks == published_blocks
    assert all(restored is published for restored, published in zip(restored_blocks, published_blocks, strict=True))
    assert [block.compressed_context_id for block in restored_blocks] == published_ids
    assert len(compressed_events) == len(published_ids)  # Replay emits no second ContextCompressed event.
    assert state[PRE_OUTPUT_HISTORY_LEN_STATE_KEY] == len(state["messages"])


@pytest.mark.asyncio
async def test_queued_compression_inserts_and_caches_summary_once_for_same_run_retry():
    first_user = _user("Turn one request")
    first_answer = _assistant_text("Turn one answer")
    state: dict = {
        "messages": [first_user, first_answer],
        "compressed_msgs": [],
        "turn_counter": 0,
    }
    marker_id = CompressibleHistoryProvider.insert_marker(state, 1)
    current_user = _user("Current turn")
    wire_source = [first_user, first_answer, current_user]
    first_wire = [_wire_message_view(message) for message in wire_source]
    strategy = _make_strategy(compaction_enabled=False)
    strategy.bind_state(state)
    compressed_context_id, _ = strategy.queue_compression(marker_id, "Turn one summary")

    assert await strategy(first_wire)

    first_projected = project_included_messages(first_wire)
    first_summaries = [
        message
        for message in first_projected
        if message.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.SUMMARY
    ]
    assert len(first_summaries) == 1
    assert len(state["compressed_msgs"]) == 1
    assert compressed_context_id in strategy._compressed_context_cache
    assert all(
        message.additional_properties.get(EXCLUDE_REASON_KEY) == _REASON_COMPRESSION
        for message in (first_user, first_answer)
    )

    retry_wire = [_wire_message_view(message) for message in wire_source]
    assert not any(
        message.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.SUMMARY for message in retry_wire
    )

    await strategy(retry_wire)

    retry_summaries = [
        message
        for message in project_included_messages(retry_wire)
        if message.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.SUMMARY
    ]
    assert retry_summaries == first_summaries
    assert len(state["compressed_msgs"]) == 1


@pytest.mark.asyncio
async def test_compressed_summary_cache_does_not_retain_folded_originals():
    """The summary cache holds WEAK identities: it never keeps the folded
    originals alive, and once they die its identity tiers match nothing —
    stale folded evidence can never misplace a summary into new work."""
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    for idx in range(3):
        state["messages"].append(_user(f"Turn {idx + 1} request " + ("x " * 2000)))
        state["messages"].append(_assistant_text(f"Turn {idx + 1} answer " + ("y " * 2000)))
        CompressibleHistoryProvider.insert_marker(state, idx + 1)

    messages = [
        msg
        for msg in state["messages"]
        if msg.additional_properties.get(HistoryMarkerKind.KEY) != HistoryMarkerKind.TURN
    ]
    messages.append(_user("Current turn"))
    tokens_before = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=tokens_before + 100,
        trigger_pct=0.85,
        target_pct=0.50,
    )
    strategy.bind_state(state)

    changed = await strategy(messages)

    assert changed
    assert strategy._compressed_context_cache
    folded = [msg for msg in messages if msg.additional_properties.get(EXCLUDE_REASON_KEY) == "cross_turn_compression"]
    assert folded
    message_refs = [weakref.ref(msg) for msg in folded]
    contents_refs = [weakref.ref(msg.contents) for msg in folded]

    del folded
    messages.clear()
    gc.collect()

    assert all(ref() is None for ref in message_refs), "cache must not retain folded originals"
    assert all(ref() is None for ref in contents_refs)

    # Dead tiers match nothing: a fresh per-call list of brand-new work
    # yields no insertion anchor, so the summary is not misplaced into it.
    fresh = [_user("brand new work")]
    strategy._reinject_compressed_context_summaries(fresh)
    assert all(msg.additional_properties.get(HistoryMarkerKind.KEY) != HistoryMarkerKind.SUMMARY for msg in fresh)


@pytest.mark.asyncio
async def test_text_only_history_emergency_does_not_persist_folded_anchor_onto_repeated_current_turn():
    """Folded old text must not structurally match and exclude repeated current input."""
    repeated_text = "repeat me " * 2000
    old_user = _user(repeated_text)
    old_answer = _assistant_text("old answer " + ("y " * 2000))
    current_user = _user(repeated_text)
    state: dict = {"messages": [old_user, old_answer], "compressed_msgs": [], "turn_counter": 0}
    CompressibleHistoryProvider.insert_marker(state, 1)

    messages = [old_user, old_answer, current_user]
    tokens_before = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=tokens_before + 100,
        trigger_pct=0.85,
        target_pct=0.50,
    )
    strategy.bind_state(state)

    changed = await strategy(messages)

    assert changed
    assert old_user.additional_properties.get(EXCLUDE_REASON_KEY) == "cross_turn_compression"
    assert current_user.additional_properties.get(EXCLUDED_KEY) is not True
    assert all(anchor.reason != _REASON_COMPRESSION for anchor in strategy._excluded_anchors)

    final = _assistant_text("final")
    state["messages"].extend([current_user, final])
    strategy.persist_exclusions_to_state(state["messages"])

    assert current_user.additional_properties.get(EXCLUDED_KEY) is not True
    assert final.additional_properties.get(EXCLUDED_KEY) is not True


@pytest.mark.asyncio
async def test_text_only_history_emergency_skips_empty_markers_before_visible_turns():
    """Phase 3 must keep scanning when an old marker has no visible wire messages."""
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    CompressibleHistoryProvider.insert_marker(state, 1)
    state["messages"].append(_user("Large completed request " + ("x " * 2000)))
    state["messages"].append(_assistant_text("Large completed answer " + ("y " * 2000)))
    CompressibleHistoryProvider.insert_marker(state, 2)

    messages = [
        msg
        for msg in state["messages"]
        if msg.additional_properties.get(HistoryMarkerKind.KEY) != HistoryMarkerKind.TURN
    ]
    messages.append(_user("Current turn"))
    tokens_before = _estimate_tokens(messages)
    compressed_events = []
    precompact_events: list[PreCompactInfo] = []

    async def _on_compress(info) -> None:
        compressed_events.append(info)

    strategy = _make_strategy(
        max_context_tokens=tokens_before + 100,
        trigger_pct=0.85,
        target_pct=0.50,
        on_compress=_on_compress,
        on_pre_compact=_async_appender(precompact_events),
    )
    strategy.bind_state(state)

    changed = await strategy(messages)

    assert changed
    assert [info.trigger for info in precompact_events].count("phase3") == 1
    assert len(compressed_events) == 1
    assert compressed_events[0].summary != "Earlier conversation"
    assert not any("Earlier conversation" in (msg.text or "") for msg in state["messages"])
    assert included_token_count(messages) < tokens_before
    assert strategy._usage_pct(included_token_count(messages)) <= strategy.target_pct


@pytest.mark.asyncio
async def test_text_only_history_emergency_preserves_nonvisible_state_markers():
    """Phase 3 must not delete real history markers when local history is omitted."""
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    state["messages"].append(_user("old real context"))
    state["messages"].append(_assistant_text("old real answer"))
    CompressibleHistoryProvider.insert_marker(state, 1)

    messages = [_user("Current service-side continuation " + ("x " * 2000))]
    tokens_before = _estimate_tokens(messages)
    compressed_events = []
    precompact_events: list[PreCompactInfo] = []

    async def _on_compress(info) -> None:
        compressed_events.append(info)

    strategy = _make_strategy(
        max_context_tokens=tokens_before + 100,
        trigger_pct=0.85,
        target_pct=0.50,
        on_compress=_on_compress,
        on_pre_compact=_async_appender(precompact_events),
    )
    strategy.bind_state(state)

    changed = await strategy(messages)

    assert not changed
    assert not compressed_events
    assert not precompact_events
    assert not state["compressed_msgs"]
    assert any(
        msg.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.TURN for msg in state["messages"]
    )


@pytest.mark.asyncio
async def test_text_only_history_emergency_preserves_after_run_timestamp_window():
    """Phase 3 state compression must not hide current assistant outputs from after_run stamping."""
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    for idx in range(5):
        state["messages"].append(_user(f"Turn {idx + 1} request " + ("x " * 2000)))
        state["messages"].append(_assistant_text(f"Turn {idx + 1} answer " + ("y " * 2000)))
        CompressibleHistoryProvider.insert_marker(state, idx + 1)

    current_user = _user("Current turn")
    wire_messages = [
        msg
        for msg in state["messages"]
        if msg.additional_properties.get(HistoryMarkerKind.KEY) != HistoryMarkerKind.TURN
    ]
    wire_messages.append(current_user)
    tokens_before = _estimate_tokens(wire_messages)

    strategy = _make_strategy(
        max_context_tokens=tokens_before + 100,
        trigger_pct=0.85,
        target_pct=0.50,
    )
    provider = CompressibleHistoryProvider(compaction_strategy=strategy)
    session = AgentSession(session_id="timestamps")
    context = SessionContext(session_id="timestamps", input_messages=[current_user])

    await provider.before_run(agent=object(), session=session, context=context, state=state)
    changed = await strategy(wire_messages)
    assert changed

    final_assistant = _assistant_text("final answer")
    context._response = ChatResponse(messages=[final_assistant])
    await provider.after_run(agent=object(), session=session, context=context, state=state)

    assert final_assistant.additional_properties[MESSAGE_CREATED_AT_KEY]
    assert (
        session.state[LAST_ASSISTANT_CREATED_AT_STATE_KEY]
        == final_assistant.additional_properties[MESSAGE_CREATED_AT_KEY]
    )


@pytest.mark.asyncio
async def test_text_only_history_emergency_records_floor_after_exhausting_markers():
    """Phase 3 must leave a metadata floor even when no completed marker survives."""
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    for idx in range(2):
        state["messages"].append(_user(f"Turn {idx + 1} request " + ("x " * 1000)))
        state["messages"].append(_assistant_text(f"Turn {idx + 1} answer " + ("y " * 1000)))
        CompressibleHistoryProvider.insert_marker(state, idx + 1)

    messages = [
        msg
        for msg in state["messages"]
        if msg.additional_properties.get(HistoryMarkerKind.KEY) != HistoryMarkerKind.TURN
    ]
    messages.append(_user("Current oversized turn " + ("z " * 8000)))
    tokens_before = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=tokens_before + 100,
        trigger_pct=0.85,
        target_pct=0.001,
    )
    strategy.bind_state(state)

    changed = await strategy(messages)

    assert changed
    assert not any(
        msg.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.TURN for msg in state["messages"]
    )
    assert state[PRE_OUTPUT_HISTORY_LEN_STATE_KEY] == len(state["messages"])


@pytest.mark.asyncio
async def test_text_only_history_emergency_prunes_cache_after_retry_rollback():
    """Rolled-back Phase 3 summaries must not be reinserted on retry."""
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    for idx in range(5):
        state["messages"].append(_user(f"Turn {idx + 1} request " + ("x " * 2000)))
        state["messages"].append(_assistant_text(f"Turn {idx + 1} answer " + ("y " * 2000)))
        CompressibleHistoryProvider.insert_marker(state, idx + 1)

    snapshot_messages = list(state["messages"])
    snapshot_properties = snapshot_message_properties(snapshot_messages)
    snapshot_compressed_count = len(state["compressed_msgs"])

    first_messages = [
        msg
        for msg in state["messages"]
        if msg.additional_properties.get(HistoryMarkerKind.KEY) != HistoryMarkerKind.TURN
    ]
    first_messages.append(_user("Current turn"))
    tokens_before = _estimate_tokens(first_messages)

    strategy = _make_strategy(
        max_context_tokens=tokens_before + 100,
        trigger_pct=0.85,
        target_pct=0.50,
    )
    strategy.bind_state(state)

    assert await strategy(first_messages)
    first_block_count = len(state["compressed_msgs"])
    assert first_block_count > 0
    assert strategy._compressed_context_cache

    state["messages"] = snapshot_messages
    del state["compressed_msgs"][snapshot_compressed_count:]
    restore_message_properties(snapshot_messages, snapshot_properties)

    retry_messages = [
        msg
        for msg in state["messages"]
        if msg.additional_properties.get(HistoryMarkerKind.KEY) != HistoryMarkerKind.TURN
    ]
    retry_messages.append(_user("Current turn retry"))

    assert await strategy(retry_messages)
    projected_summaries = [
        msg
        for msg in project_included_messages(retry_messages)
        if msg.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.SUMMARY
    ]
    assert len(projected_summaries) == len(state["compressed_msgs"]) == first_block_count


@pytest.mark.asyncio
async def test_single_turn_compaction():
    """With only one turn, Phase 4 (current turn removal) handles compaction."""
    messages = _build_single_turn(8, result_size=3000)
    total_before = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total_before + 100,  # just over total → trigger fires
        trigger_pct=0.90,
        target_pct=0.50,
    )
    changed = await strategy(messages)
    assert changed

    # Some messages should be excluded (Phase 4 removes groups entirely)
    excluded = [m for m in messages if m.additional_properties.get(EXCLUDED_KEY, False)]
    assert len(excluded) > 0

    # Token count should decrease
    total_after = included_token_count(messages)
    assert total_after < total_before


@pytest.mark.asyncio
async def test_phase4_drops_all_current_turn_groups():
    """Phase 4 drops every current-turn tool-call group in one shot."""
    messages = _build_single_turn(6, result_size=1000)
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,  # impossibly low → force Phase 4 drop-all
    )
    changed = await strategy(messages)
    assert changed

    # With drop-all semantics every tool_call/result message in the current
    # turn is excluded — only the user message survives.
    excluded = [m for m in messages if m.additional_properties.get(EXCLUDED_KEY, False) and _group_id(m) is not None]
    assert len(excluded) >= 6  # at least one per tool-call group (call + result)


@pytest.mark.asyncio
async def test_phase4_skipped_when_current_turn_has_no_tool_calls():
    """Phase 4 does nothing if the current turn has only a user message."""
    messages = [_user("hi")]
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
    )
    changed = await strategy(messages)
    assert not changed


# ---------------------------------------------------------------------------
# Phase 1: Old turn compaction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase1_compacts_old_turns_first():
    """Old turns are compacted before touching the current turn."""
    messages = _build_multi_turn(3, groups_per_turn=3, result_size=2000)
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.90,
        target_pct=0.50,
    )
    changed = await strategy(messages)
    assert changed

    # Current turn's tool groups (the last 3) should NOT be compacted
    # Find the last user message (start of current turn)
    last_user_idx = max(i for i, m in enumerate(messages) if m.role == "user")

    # Check that no tool results AFTER the last user message are excluded
    current_turn_excluded = [
        m
        for m in messages[last_user_idx:]
        if m.additional_properties.get(EXCLUDED_KEY, False) and any(c.type == "function_result" for c in m.contents)
    ]
    assert len(current_turn_excluded) == 0, "Current turn tool results should not be compacted in phase 1"


@pytest.mark.asyncio
async def test_phase1_never_splits_a_call_from_its_result_block():
    """A previous turn whose response interleaves text between the call and
    its result block must compact atomically. The call and the result block
    grouped apart lets Phase 1 exclude the call, hit the token target, and
    stop — projecting a function_result whose call is gone, which providers
    reject outright."""
    messages = [
        _user("Turn 1 request"),
        _assistant_tool_call("call_a", "big_tool", args={"payload": "x" * 8000}),
        _assistant_text("narration between call and result"),
        _tool_result("call_a", "ok"),
        _assistant_text("Turn 1 answer"),
        _user("Turn 2 request"),
        _assistant_text("Turn 2 answer"),
    ]
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.90,
        target_pct=0.50,
    )
    changed = await strategy(messages)
    assert changed

    projected = project_included_messages(messages)
    projected_calls = {c.call_id for m in projected for c in m.contents if c.type == "function_call"}
    projected_results = {c.call_id for m in projected for c in m.contents if c.type == "function_result"}
    assert projected_results <= projected_calls, "projected a result whose call was compacted away"
    assert projected_calls <= projected_results, "projected a call whose result was compacted away"
    assert "call_a" not in projected_calls, "the oversized previous-turn group should have been compacted"


@pytest.mark.asyncio
async def test_phase1_summary_carries_absorbed_narration():
    """Assistant narration fused between a call and its result block is
    excluded with the rest of the group; the summary must carry its text
    instead of silently dropping it."""
    messages = [
        _user("Turn 1 request"),
        _assistant_tool_call("call_a", "big_tool", args={"payload": "x" * 8000}),
        _assistant_text("NARRATION checking the parser config"),
        _tool_result("call_a", "ok"),
        _assistant_text("Turn 1 answer"),
        _user("Turn 2 request"),
        _assistant_text("Turn 2 answer"),
    ]
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.90,
        target_pct=0.50,
    )
    assert await strategy(messages)

    summary = next(m for m in messages if m.text and m.text.startswith("[Tool call:"))
    assert "NARRATION checking the parser config" in summary.text


@pytest.mark.asyncio
async def test_phase1_stops_at_target():
    """Phase 1 stops as soon as usage drops to target_pct."""
    # 5 turns with enough content that compacting 1-2 old turns reaches target
    messages = _build_multi_turn(5, groups_per_turn=2, result_size=1500)
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.90,
        target_pct=0.80,  # high target → only 1-2 old turns need compaction
    )
    changed = await strategy(messages)
    assert changed

    # Not all old turns should be compacted (target reached early)
    summaries = [m for m in messages if m.text and m.text.startswith("[Tool call:")]
    total_old_groups = 4 * 2  # 4 old turns * 2 groups each
    assert len(summaries) < total_old_groups


@pytest.mark.asyncio
async def test_phase1_then_phase4_fallthrough():
    """When old turns are exhausted, falls through to Phase 4 (current turn removal)."""
    # 2 turns: small old turn, large current turn
    messages = [
        _user("Turn 1"),
        *_build_tool_group("t0_c0", "tool_0", "short"),
        _assistant_text("R1"),
        _user("Turn 2"),
    ]
    # Add many large tool calls in current turn
    for i in range(8):
        messages.extend(_build_tool_group(f"t1_c{i}", f"tool_{i}", "x" * 3000))
    messages.append(_assistant_text("R2"))

    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.90,
        target_pct=0.30,  # low target forces current turn compaction
    )
    changed = await strategy(messages)
    assert changed

    # Old turn groups should have summaries (Phase 1), current turn groups
    # should be excluded (Phase 4 removes them entirely)
    excluded = [m for m in messages if m.additional_properties.get(EXCLUDED_KEY, False)]
    assert len(excluded) > 0  # old turn + some current turn groups


# ---------------------------------------------------------------------------
# Summary format
# ---------------------------------------------------------------------------


def test_summary_preserves_args():
    """Summary text includes function name and arguments."""
    group = _build_tool_group("c1", "grep", "47 matches found", args={"pattern": "class Foo", "path": "src/"})
    summary = _build_summary(group)
    assert "grep" in summary
    assert "class Foo" in summary
    assert "src/" in summary
    assert "\u2192" in summary  # → arrow


def test_summary_truncates_long_results():
    """Long results are truncated with char count."""
    group = _build_tool_group("c1", "read_file", "x" * 500, args={"path": "foo.py"})
    summary = _build_summary(group)
    assert "500 chars" in summary


def test_summary_multiple_tools():
    """Summary handles multiple tool calls in one group."""
    msgs = [
        _assistant_tool_call("c1", "read_file", args={"path": "a.py"}),
        _tool_result("c1", "content A"),
        _assistant_tool_call("c2", "read_file", args={"path": "b.py"}),
        _tool_result("c2", "content B"),
    ]
    summary = _build_summary(msgs)
    assert "a.py" in summary
    assert "b.py" in summary


@pytest.mark.parametrize("restored", [False, True], ids=["live", "restored"])
def test_summary_preserves_hosted_mcp_name_and_result_text(restored: bool) -> None:
    messages = [
        Message(
            "assistant",
            [Content.from_mcp_server_tool_call("mcp-1", "search_docs", arguments={"query": "Chrys"})],
        ),
        Message(
            "assistant",
            [Content.from_mcp_server_tool_result("mcp-1", output=[Content.from_text("found the guide")])],
        ),
    ]
    if restored:
        messages = [Message.from_dict(message.to_dict()) for message in messages]

    summary = _build_summary(messages)

    assert "search_docs" in summary
    assert 'query="Chrys"' in summary
    assert "found the guide" in summary


def test_summary_prefers_hosted_items_over_duplicate_result_mirror() -> None:
    messages = [
        Message(
            "assistant",
            [Content.from_hosted_tool_call("hosted-1", tool_name="remote_task", arguments={"task": "inspect"})],
        ),
        Message(
            "assistant",
            [
                Content.from_hosted_tool_result(
                    "hosted-1",
                    tool_name="remote_task",
                    result="authoritative output",
                    items=[Content.from_text("authoritative output")],
                )
            ],
        ),
    ]

    summary = _build_summary(messages)

    assert summary.count("authoritative output") == 1


@pytest.mark.parametrize("restored", [False, True], ids=["live", "restored"])
def test_summary_preserves_specialized_hosted_calls_and_results(restored: bool) -> None:
    image = Content.from_uri("data:image/png;base64,QUJD", media_type="image/png")
    messages = [
        Message(
            "assistant",
            [
                Content.from_search_tool_call("search-1", tool_name="web_search", arguments={"query": "Chrys"}),
                Content.from_code_interpreter_tool_call(
                    call_id="code-1",
                    inputs=[Content.from_text("print('code input')")],
                ),
                Content.from_image_generation_tool_call(image_id="image-1"),
                Content.from_shell_tool_call(call_id="shell-1", commands=["printf shell-input"]),
            ],
        ),
        Message(
            "tool",
            [
                Content.from_search_tool_result("search-1", tool_name="web_search", result="search result"),
                Content.from_code_interpreter_tool_result(
                    call_id="code-1",
                    outputs=[Content.from_text("code result")],
                ),
                Content.from_image_generation_tool_result(image_id="image-1", outputs=[image]),
                Content.from_shell_tool_result(
                    call_id="shell-1",
                    outputs=[Content.from_shell_command_output(stdout="shell result", exit_code=0)],
                ),
            ],
        ),
    ]
    if restored:
        messages = [Message.from_dict(message.to_dict()) for message in messages]

    summary = _build_summary(messages)

    for fragment in (
        "web_search",
        'query="Chrys"',
        "search result",
        "code_interpreter",
        "code input",
        "code result",
        "image_generation",
        'image_id="image-1"',
        "image/png image",
        "shell",
        "shell-input",
        "shell result",
        "exit code: 0",
    ):
        assert fragment in summary


def test_summary_preserves_absorbed_narration():
    """Fused groups may hold assistant narration between the call and its
    results; the summary carries that text in transcript order."""
    msgs = [
        _assistant_tool_call("c1", "read_file", args={"path": "a.py"}),
        _assistant_text("Now checking the parser config"),
        _tool_result("c1", "content A"),
    ]
    summary = _build_summary(msgs)
    assert 'assistant: "Now checking the parser config"' in summary
    assert "read_file" in summary
    assert summary.index("Now checking the parser config") < summary.index("content A")


def test_summary_truncates_long_narration():
    """Absorbed narration is trimmed with a char count, like results."""
    msgs = [
        _assistant_tool_call("c1", "read_file", args={"path": "a.py"}),
        _assistant_text("n" * 500),
        _tool_result("c1", "content A"),
    ]
    summary = _build_summary(msgs)
    assert "(500 chars)" in summary
    assert "n" * 500 not in summary


def test_compact_group_skips_group_with_excluded_result() -> None:
    call, result = _build_tool_group("c1", "read_file", "excluded payload", args={"path": "secret.txt"})
    set_excluded(result, excluded=True, reason="cross_turn_compression")
    messages = [call, result]
    before = _estimate_tokens(messages)
    grouped = _group_messages_by_id(messages)
    group_id = _tool_group_id_of(messages, "c1")
    strategy = _make_strategy()

    result = strategy._compact_group(messages, group_id, grouped[group_id])
    assert included_token_count(messages) == before
    assert result is None
    assert strategy._summary_cache == {}


def test_compact_group_does_not_build_summary_from_excluded_call(monkeypatch: pytest.MonkeyPatch) -> None:
    call, result = _build_tool_group("c1", "sensitive_tool_name", "visible result", args={"secret": "value"})
    set_excluded(call, excluded=True, reason="cross_turn_compression")
    messages = [call, result]
    _estimate_tokens(messages)
    grouped = _group_messages_by_id(messages)
    group_id = _tool_group_id_of(messages, "c1")
    strategy = _make_strategy()

    def fail_build_summary(_group_msgs: list[Message]) -> str:
        pytest.fail("excluded group metadata reached _build_summary")

    monkeypatch.setattr("chrys.service.context.compaction.summaries._build_summary", fail_build_summary)

    assert strategy._compact_group(messages, group_id, grouped[group_id]) is None
    assert strategy._summary_cache == {}


def test_summary_bounds_total_length() -> None:
    calls = [
        Content.from_function_call(f"call_{index}", f"tool_{index}", arguments={"value": index}) for index in range(300)
    ]
    results = [
        Content.from_function_result(f"call_{index}", result=f"result-{index}-" + "x" * 100) for index in range(300)
    ]

    summary = _build_summary([Message("assistant", calls), Message("tool", results)])

    assert len(summary) <= _SUMMARY_TOTAL_MAX
    assert "tool_0" in summary
    assert "tool_299" not in summary
    assert "... [truncated]" in summary


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_invalid_params():
    """Constructor rejects invalid parameters."""
    with pytest.raises(ValueError, match="max_context_tokens"):
        UnifiedContextStrategy(max_context_tokens=0)
    with pytest.raises(ValueError, match="trigger_pct"):
        UnifiedContextStrategy(trigger_pct=0.0)
    with pytest.raises(ValueError, match="target_pct"):
        UnifiedContextStrategy(trigger_pct=0.85, target_pct=0.90)


# ---------------------------------------------------------------------------
# MixedLanguageTokenizer tests
# ---------------------------------------------------------------------------


def test_mixed_language_tokenizer_returns_positive():
    """The built-in estimator always returns at least one token."""
    t = MixedLanguageTokenizer()
    assert t.count_tokens("") == 1
    assert t.count_tokens("hello") >= 1
    assert t.count_tokens("你好") >= 1


def test_mixed_language_tokenizer_weights_cjk_more_than_ascii():
    tokenizer = MixedLanguageTokenizer()
    assert tokenizer.count_tokens("你" * 100) > tokenizer.count_tokens("a" * 100)


def test_content_signature_distinguishes_function_result_images():
    first = Message(
        "tool",
        [
            Content.from_function_result(
                "call_1",
                result=[Content.from_text("same"), Content.from_data(b"first-image", "image/png")],
            )
        ],
    )
    second = Message(
        "tool",
        [
            Content.from_function_result(
                "call_1",
                result=[Content.from_text("same"), Content.from_data(b"second-image", "image/png")],
            )
        ],
    )

    assert _content_signature(first) != _content_signature(second)


def test_content_signature_ignores_unknown_function_result_uri_without_crashing():
    message = Message(
        "tool",
        [
            Content.from_function_result(
                "call_1",
                result=[Content.from_text("same"), Content.from_uri("https://example.com/blob")],
            )
        ],
    )

    assert _content_signature(message)


# ---------------------------------------------------------------------------
# Message ID deduplication tests
# ---------------------------------------------------------------------------


def test_dedup_message_ids_unique():
    """Unique IDs are left unchanged."""
    msgs = [
        Message("user", ["hi"], message_id="msg_1"),
        Message("assistant", ["hey"], message_id="msg_2"),
    ]
    _dedup_message_ids(msgs)
    assert msgs[0].message_id == "msg_1"
    assert msgs[1].message_id == "msg_2"


def test_dedup_message_ids_duplicates():
    """Duplicate IDs get positional suffixes."""
    msgs = [
        Message("user", ["hi"], message_id="msg_1"),
        Message("assistant", ["a"], message_id="msg_2"),
        Message("user", ["q2"], message_id="msg_1"),
        Message("assistant", ["b"], message_id="msg_2"),
    ]
    _dedup_message_ids(msgs)
    assert msgs[0].message_id == "msg_1"
    assert msgs[1].message_id == "msg_2"
    assert msgs[2].message_id == "msg_1_i2"
    assert msgs[3].message_id == "msg_2_i3"


def test_dedup_message_ids_no_id():
    """Messages without IDs are left untouched."""
    msgs = [Message("user", ["hi"]), Message("user", ["bye"])]
    _dedup_message_ids(msgs)
    assert msgs[0].message_id is None
    assert msgs[1].message_id is None


@pytest.mark.asyncio
async def test_dedup_renames_persist_on_state_messages():
    """Dedup renames land on stored history (P5.4 aliasing pin).

    The kernel ``SessionContext.extend_messages`` does not copy, so the wire
    list the strategy dedups holds the very state objects — renamed
    ``message_id`` values persist across runs (and to disk). Code doing
    message_id-based cross-run accounting must not assume ids are immutable.
    """
    state_messages = [
        Message("user", ["hi"], message_id="dup"),
        Message("user", ["again"], message_id="dup"),
    ]
    strategy = _make_strategy()  # budget far above usage: no compaction fires

    await strategy(state_messages)

    assert state_messages[0].message_id == "dup"
    assert state_messages[1].message_id == "dup_i1"


# ---------------------------------------------------------------------------
# Regression: duplicate message IDs cause mega-group merging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_ids_merge_groups_without_dedup():
    """Reproduce mega-group bug: duplicate IDs cause cross-run group merging."""
    messages = [
        Message("user", ["What does file A do?"], message_id="msg_1"),
        _assistant_tool_call("call_r1", "read_file"),
        _tool_result("call_r1", "file A content: " + "a" * 3000),
        _assistant_text("File A does X."),
        Message("user", ["Now read file B."], message_id="msg_5"),
        _assistant_tool_call("call_r2", "read_file"),
        _tool_result("call_r2", "file B content: " + "b" * 3000),
        _assistant_text("File B does Y."),
    ]
    messages[1].message_id = "msg_2"
    messages[2].message_id = "msg_3"
    messages[5].message_id = "msg_2"  # duplicate!
    messages[6].message_id = "msg_3"  # duplicate!

    # Without dedup: annotate_message_groups merges the two read_file groups
    annotate_message_groups(messages)
    group_ids = set()
    for m in messages:
        ga = m.additional_properties.get("_group", {})
        if ga.get("kind") == "tool_call":
            group_ids.add(ga.get("id"))
    assert len(group_ids) == 1, f"BUG NOT REPRODUCED: expected 1 merged group, got {len(group_ids)}"

    # Now verify the fix: with dedup, they become separate groups
    for m in messages:
        if "_group" in m.additional_properties:
            del m.additional_properties["_group"]
    messages[1].message_id = "msg_2"
    messages[2].message_id = "msg_3"
    messages[5].message_id = "msg_2"
    messages[6].message_id = "msg_3"

    _dedup_message_ids(messages)
    annotate_message_groups(messages)
    group_ids_fixed = set()
    for m in messages:
        ga = m.additional_properties.get("_group", {})
        if ga.get("kind") == "tool_call":
            group_ids_fixed.add(ga.get("id"))
    assert len(group_ids_fixed) == 2, f"Fix failed: expected 2 separate groups, got {len(group_ids_fixed)}"


@pytest.mark.asyncio
async def test_duplicate_ids_compaction_with_dedup():
    """With dedup, compaction correctly treats cross-run tool calls as separate groups."""
    messages = [Message("user", ["start"], message_id="u1")]
    for run in range(3):
        messages.append(
            Message(
                "assistant",
                [Content.from_function_call(f"call_run{run}", "read_file", arguments={})],
                message_id="msg_2",
            )
        )
        messages.append(
            Message(
                "tool",
                [Content.from_function_result(f"call_run{run}", result=f"run {run}: " + "x" * 2000)],
                message_id="msg_3",
            )
        )

    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
    )
    changed = await strategy(messages)
    assert changed

    # With dedup: 3 separate groups. Phase 4 drops ALL current-turn groups
    # once LAST_WORDS is produced — so all 3 should be excluded.
    excluded_groups = {
        _group_id(m) for m in messages if m.additional_properties.get(EXCLUDED_KEY, False) and _group_id(m) is not None
    }
    assert len(excluded_groups) == 3, f"Expected 3 excluded groups (drop-all), got {len(excluded_groups)}"


# ---------------------------------------------------------------------------
# on_compaction callback tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_compaction_called_with_correct_info():
    """Callback receives CompactionInfo with accurate group count, tool names, and token deltas."""
    messages = _build_single_turn(6, result_size=2000)
    total = _estimate_tokens(messages)

    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        on_compaction=_async_appender(received),
    )
    changed = await strategy(messages)
    assert changed

    # Phase 4 drops all 6 tool-call groups + the trailing assistant_text group.
    assert len(received) >= 1
    p4 = [r for r in received if r.phase == "phase4"]
    assert len(p4) == 1
    assert p4[0].compacted_groups == 7  # 6 tool-call groups + 1 assistant_text
    assert p4[0].tool_names == [f"tool_{i}" for i in range(6)]
    assert p4[0].tokens_before > p4[0].tokens_after
    assert p4[0].tokens_after > 0


@pytest.mark.asyncio
async def test_on_pre_compact_called_before_phase4():
    messages = _build_single_turn(4, result_size=2000)
    total = _estimate_tokens(messages)

    received: list[PreCompactInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        on_pre_compact=_async_appender(received),
    )

    await strategy(messages)

    assert [info.trigger for info in received] == ["phase4"]
    assert received[0].usage_pct > 0
    assert received[0].tokens_before > 0


@pytest.mark.asyncio
async def test_on_compaction_multiple_groups():
    """Callback reports all compacted groups and their tool names."""
    messages = _build_single_turn(8, result_size=2000)
    total = _estimate_tokens(messages)

    received: list[CompactionInfo] = []

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        on_compaction=_async_appender(received),
    )
    await strategy(messages)

    assert len(received) >= 1
    # Each callback has consistent token deltas and tool names
    for info in received:
        assert info.compacted_groups >= 1
        # Phase 4 drops tool_call AND inline assistant_text groups; tool_names
        # only lists function names from tool_call groups.
        assert len(info.tool_names) <= info.compacted_groups
        assert info.tokens_before > info.tokens_after


@pytest.mark.asyncio
async def test_on_compaction_not_called_under_trigger():
    """Callback is NOT called when no compaction occurs (under trigger)."""
    messages = [_user("hello"), _assistant_text("hi")]

    called = False

    async def on_compact(info: CompactionInfo) -> None:
        nonlocal called
        called = True

    strategy = _make_strategy(max_context_tokens=1_000_000, on_compaction=on_compact)
    changed = await strategy(messages)
    assert not changed
    assert not called


@pytest.mark.asyncio
async def test_on_compaction_token_deltas_are_consistent():
    """tokens_before and tokens_after match independent measurement."""
    messages = _build_single_turn(8, result_size=3000)
    measured_before = _estimate_tokens(messages)

    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=measured_before + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        on_compaction=_async_appender(received),
    )
    await strategy(messages)

    assert len(received) >= 1
    # First phase's tokens_before == original measurement
    assert received[0].tokens_before == measured_before
    # Last phase's tokens_after == final included count
    assert received[-1].tokens_after == included_token_count(messages)


# ---------------------------------------------------------------------------
# Bug reproductions: EXCLUDED flags persist, shared dict mutation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repro_excluded_lost_on_copies():
    """REPRO: compaction on shallow copies does NOT persist _excluded to originals."""
    import copy

    originals = _build_single_turn(6, result_size=2000)
    total = _estimate_tokens(originals)

    copies = []
    for msg in originals:
        c = copy.copy(msg)
        c.additional_properties = dict(msg.additional_properties)
        copies.append(c)

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
    )
    changed = await strategy(copies)
    assert changed

    excluded_copies = [m for m in copies if m.additional_properties.get(EXCLUDED_KEY, False)]
    assert len(excluded_copies) > 0

    excluded_originals = [m for m in originals if m.additional_properties.get(EXCLUDED_KEY, False)]
    assert len(excluded_originals) == 0


def test_repro_set_summarized_leaks_via_shared_dict():
    """REPRO: without the fix, _set_summarized mutates the shared _group dict."""
    import copy

    from chrys.service.context.compaction import GROUP_ANNOTATION_KEY, SUMMARIZED_BY_SUMMARY_ID_KEY

    original = _assistant_tool_call("c1", "read_file")
    original.additional_properties[GROUP_ANNOTATION_KEY] = {
        "_group_id": "group_c1",
        "_group_kind": "tool_call",
    }

    msg_copy = copy.copy(original)
    msg_copy.additional_properties = dict(original.additional_properties)

    shared_group = msg_copy.additional_properties[GROUP_ANNOTATION_KEY]
    assert shared_group is original.additional_properties[GROUP_ANNOTATION_KEY]

    shared_group[SUMMARIZED_BY_SUMMARY_ID_KEY] = "leak_test"
    assert SUMMARIZED_BY_SUMMARY_ID_KEY in original.additional_properties[GROUP_ANNOTATION_KEY]

    del shared_group[SUMMARIZED_BY_SUMMARY_ID_KEY]


def test_set_summarized_does_not_mutate_shared_group_dict():
    """_set_summarized creates a new _group dict, never mutating the original."""
    import copy

    original = _assistant_tool_call("c1", "read_file")
    original.additional_properties["_group"] = {"_group_id": "group_c1", "_group_kind": "tool_call"}

    msg_copy = copy.copy(original)
    msg_copy.additional_properties = dict(original.additional_properties)

    assert original.additional_properties["_group"] is msg_copy.additional_properties["_group"]

    _set_summarized(msg_copy, "tool_summary_group_c1")

    assert "_summarized_by_summary_id" in msg_copy.additional_properties["_group"]
    assert "_summarized_by_summary_id" not in original.additional_properties["_group"]


# ---------------------------------------------------------------------------
# Excluded flags persist across agent.run() calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_excluded_flags_persist_on_originals():
    """Running the strategy on originals persists _excluded flags."""
    messages = _build_single_turn(6, result_size=2000)
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
    )

    changed = await strategy(messages)
    assert changed

    excluded_ids_run1 = {m.message_id for m in messages if m.additional_properties.get(EXCLUDED_KEY, False)}
    assert len(excluded_ids_run1) > 0

    await strategy(messages)
    excluded_ids_run2 = {m.message_id for m in messages if m.additional_properties.get(EXCLUDED_KEY, False)}
    assert excluded_ids_run1 == excluded_ids_run2


@pytest.mark.asyncio
async def test_after_run_compaction_prevents_recompaction():
    """Simulates the full cross-run cycle with after_run compaction."""
    messages = _build_single_turn(6, result_size=2000)
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
    )
    provider = CompressibleHistoryProvider(compaction_strategy=strategy)

    state: dict = {"messages": messages}

    await strategy(state["messages"])

    excluded_count = sum(1 for m in state["messages"] if m.additional_properties.get(EXCLUDED_KEY, False))
    assert excluded_count > 0

    visible = await provider.get_messages(None, state=state)
    assert len(visible) < len(state["messages"])
    # Excluded messages must not appear in visible output
    for m in visible:
        assert not m.additional_properties.get(EXCLUDED_KEY, False)


async def _run_tool_loop_with_current_turn_compaction(*, stream: bool) -> list[Message]:
    """Run the real Agent tool loop and return persisted history messages."""

    def _big_tool() -> str:
        return "tool output\n" * 5000

    strategy = _make_strategy(
        max_context_tokens=1000,
        trigger_pct=0.01,
        target_pct=0.001,
    )
    history = CompressibleHistoryProvider(compaction_strategy=strategy)
    state: dict[str, object] = {}
    session = AgentSession()
    session.state[history.source_id] = state

    mock = MockChatClient(
        responses=[
            MockResponse(tool_calls=[("_big_tool", "c1", {})]),
            MockResponse(text="done"),
        ]
    )
    tool = FunctionTool(func=_big_tool, name="_big_tool", description="Big tool")
    agent = Agent(
        client=mock,
        instructions="test",
        tools=[tool],
        context_providers=[history],
    )

    async with agent:
        if stream:
            response_stream = agent.run("q", session=session, stream=True, compaction_strategy=strategy)
            async for _ in response_stream:
                pass
            await response_stream.get_final_response()
        else:
            await agent.run("q", session=session, compaction_strategy=strategy)

    messages = state["messages"]
    assert isinstance(messages, list)
    return messages


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
async def test_current_turn_exclusions_persist_for_streaming_and_non_streaming(stream: bool) -> None:
    """Current-turn Phase 4 exclusions must survive both response paths.

    Non-streaming stores response messages directly enough that identity-based
    anchors work.  Streaming rebuilds the same messages from updates before
    ``after_run`` persists them, so this also covers the structural fallback.
    """
    messages = await _run_tool_loop_with_current_turn_compaction(stream=stream)

    assistant_call = next(
        m
        for m in messages
        if m.role == "assistant" and any(c.type == "function_call" for c in m.contents) and _has_call_id(m, "c1")
    )
    tool_result = next(
        m
        for m in messages
        if m.role == "tool" and any(c.type == "function_result" for c in m.contents) and _has_call_id(m, "c1")
    )

    assert assistant_call.additional_properties.get(EXCLUDED_KEY) is True
    assert assistant_call.additional_properties.get(EXCLUDE_REASON_KEY) == _REASON_CURRENT_TURN_DROP
    assert tool_result.additional_properties.get(EXCLUDED_KEY) is True
    assert tool_result.additional_properties.get(EXCLUDE_REASON_KEY) == _REASON_CURRENT_TURN_DROP


def test_exclusion_anchor_ignores_contents_list_hit_with_wrong_shape() -> None:
    """A list-identity hit must not beat a valid structural match."""
    wrong_id_hit = _assistant_text("wrong message")
    structural_match = _tool_result("c1", "current")
    anchor = _ExclusionAnchor(
        reason=_REASON_CURRENT_TURN_DROP,
        role="tool",
        contents_ref=weakref.ref(wrong_id_hit.contents),
        call_ids=("c1",),
    )

    assert anchor.find_in([wrong_id_hit, structural_match], set()) == 1


def test_exclusion_anchor_ignores_message_id_hit_with_wrong_shape() -> None:
    """A duplicate message_id must not beat a valid structural match."""
    wrong_id_hit = _assistant_tool_call("old", "zsh")
    wrong_id_hit.message_id = "dup"
    structural_match = _tool_result("c1", "current")
    structural_match.message_id = "dup"
    anchor = _ExclusionAnchor(
        reason=_REASON_CURRENT_TURN_DROP,
        role="tool",
        contents_ref=None,
        call_ids=("c1",),
        message_id="dup",
    )

    assert anchor.find_in([wrong_id_hit, structural_match], set()) == 1


def test_exclusion_anchor_from_wire_view_binds_state_original_not_structural_twin() -> None:
    """Shared-content-ids tier: a view-snapshotted anchor keeps identity precision.

    The wire hands clients per-call views (fresh wrapper + fresh contents
    list, SAME content objects). An anchor built from the view must bind the
    state original through the shared content ids — not fall through to the
    newest-first structural scan, which would bind a byte-equal twin sitting
    later in state.
    """
    from copy import copy

    original = _tool_result("c1", "same payload")
    twin = _tool_result("c1", "same payload")
    view = copy(original)
    view.contents = list(original.contents)
    anchor = _ExclusionAnchor.from_message(view)

    assert anchor.find_in([original, twin], set()) == 0


def test_exclusion_anchor_does_not_retain_its_source() -> None:
    """Identity tiers hold WEAK references: an anchor never keeps its source
    message alive, and a dead tier simply stops matching — the structural
    fallback still lands the exclusion on an equal-shaped rebuild, exactly
    the streaming-rebuild path."""
    strategy = _make_strategy()
    source = _tool_result("c1", "same payload")
    source.additional_properties[EXCLUDED_KEY] = True
    source.additional_properties[EXCLUDE_REASON_KEY] = _REASON_CURRENT_TURN_DROP
    source_ref = weakref.ref(source)
    content_ref = weakref.ref(source.contents[0])

    strategy._snapshot_excluded_anchors([source])
    snapshot = strategy.snapshot_retry_state()

    strategy._excluded_anchors = []
    del source
    gc.collect()
    assert source_ref() is None, "anchors must not pin their source"
    assert content_ref() is None

    strategy.restore_retry_state(snapshot)
    target = _tool_result("c1", "same payload")
    strategy.persist_exclusions_to_state([target])

    assert target.additional_properties[EXCLUDED_KEY] is True
    assert strategy._excluded_anchors == []


def test_exclusion_anchor_replaced_contents_list_stops_identity_matching() -> None:
    """Snapshot semantics: the list-identity tier captures THE list at
    snapshot time. Replacing the message's contents list defeats tier 1;
    tier 2 (shared content objects) still binds the same message."""
    source = _tool_result("c1", "payload")
    anchor = _ExclusionAnchor.from_message(source)
    assert anchor.contents_ref is not None

    source.contents = list(source.contents)

    assert anchor.contents_ref() is not source.contents
    assert anchor.find_in([source], set()) == 0  # content_refs tier


def test_exclusion_anchor_shared_list_twin_keeps_identity_match_after_owner_dies() -> None:
    """A shared-list twin keeps the list object alive, so the list-identity
    tier keeps matching it — exactly today's captured-id semantics."""
    source = _tool_result("c1", "payload")
    twin = _tool_result("c1", "payload")
    twin.contents = source.contents  # share THE list object
    anchor = _ExclusionAnchor.from_message(source)

    del source
    gc.collect()

    assert anchor.contents_ref is not None
    assert anchor.contents_ref() is twin.contents
    assert anchor.find_in([twin], set()) == 0


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_last_included_tokens_tracked():
    """Strategy tracks its own included_token_count for calibration."""
    messages = _build_single_turn(4, result_size=1000)

    strategy = _make_strategy(max_context_tokens=1_000_000)
    assert strategy.last_included_tokens == 0

    await strategy(messages)
    assert strategy.last_included_tokens > 0

    expected = _estimate_tokens(messages)
    assert strategy.last_included_tokens == expected


@pytest.mark.asyncio
async def test_calibrate_learns_ratio():
    """calibrate() learns overhead on first call, then ratio on subsequent calls."""
    messages = _build_single_turn(4, result_size=4000)

    strategy = _make_strategy(max_context_tokens=1_000_000)
    assert strategy.calibration_ratio == 1.0

    await strategy(messages)
    our_estimate = strategy.last_included_tokens
    assert our_estimate > 0

    # First calibrate: learns overhead, skips ratio (circular)
    overhead = 5000
    strategy.calibrate(our_estimate + overhead)
    assert strategy.system_overhead_tokens == overhead
    assert strategy.calibration_ratio == 1.0  # unchanged on first call

    # Second calibrate: ratio = api / (local + overhead)
    # API reports 10% more than our estimate + overhead → ratio ≈ 1.1
    api_tokens = int((our_estimate + overhead) * 1.1)
    strategy.calibrate(api_tokens)
    expected = api_tokens / (our_estimate + overhead)
    assert abs(strategy.calibration_ratio - expected) < 0.01


@pytest.mark.asyncio
async def test_restored_old_estimator_session_is_migrated_before_admission():
    """AF 1.12 (#7124) migration: artifacts persisted under the old
    (``ensure_ascii=True``) estimator are both discarded on restore — the
    prepare pass recomputes unversioned token counts and the v1 calibration
    record is rejected — so output-cap admission never mixes stale artifacts
    with new-estimator semantics before the first provider response.
    """
    # A restored CJK conversation carrying escape-era inflated counts with no
    # estimator-version stamp (exactly what pre-migration sessions persist).
    messages = [_user("中文内容" * 100), _assistant_text("日本語のかな" * 100)]
    annotate_message_groups(messages)
    for message in messages:
        annotation = message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY]
        annotation[chrys_compaction.GROUP_TOKEN_COUNT_KEY] = 45_000
        assert chrys_compaction.GROUP_TOKEN_ESTIMATOR_VERSION_KEY not in annotation

    strategy = _make_strategy(max_context_tokens=100_000)
    metadata = SessionRuntimeMetadata(
        context_calibration={
            "v": 1,
            "system_overhead_tokens": 5_000,
            "calibration_ratio": 0.4,
            "model_profile_fingerprint": "fp-model",
            "agent_profile_fingerprint": "fp-agent",
        }
    )
    # Fingerprints match, but the v1 record must be rejected wholesale.
    assert not metadata.restore_context_calibration(
        strategy,
        model_profile_fingerprint="fp-model",
        agent_profile_fingerprint="fp-agent",
    )
    assert strategy.calibration_ratio == 1.0
    assert strategy.system_overhead_tokens == 0

    # The real prepare pass, before any provider response arrives.
    await strategy(messages)

    recomputed = strategy.last_included_tokens
    assert recomputed == included_token_count(messages)
    assert recomputed < 5_000  # stale 90k total discarded, not reused
    for message in messages:
        annotation = message.additional_properties[chrys_compaction.GROUP_ANNOTATION_KEY]
        assert annotation[chrys_compaction.GROUP_TOKEN_ESTIMATOR_VERSION_KEY] == (
            chrys_compaction.TOKEN_ESTIMATOR_VERSION
        )

    # Admission with migrated state: recomputed input at ratio 1.0 leaves
    # ample room, so the cap survives untouched. Stale counts (room = 10k) or
    # the stale 0.4 ratio over stale counts (room = 62k) would clamp it.
    admitted = _clamp_output_cap_for_context(
        {"max_tokens": 90_000},
        strategy=strategy,
        request_overhead_tokens=0,
        client_kwargs={},
    )
    assert admitted["max_tokens"] == 90_000


@pytest.mark.asyncio
async def test_calibrate_handles_hot_estimator():
    """When our estimate exceeds API count, ratio goes below 1.0."""
    messages = _build_single_turn(4, result_size=4000)

    strategy = _make_strategy(max_context_tokens=1_000_000)
    await strategy(messages)
    local = strategy.last_included_tokens

    # First call: learn overhead
    overhead = 5000
    strategy.calibrate(local + overhead)

    # Second call: API reports less than (local + overhead) → ratio < 1.0
    api_tokens = int((local + overhead) * 0.9)
    strategy.calibrate(api_tokens)
    expected = api_tokens / (local + overhead)
    assert abs(strategy.calibration_ratio - expected) < 0.01


@pytest.mark.asyncio
async def test_calibrate_first_call_learns_overhead_only():
    """First calibrate() learns overhead and skips ratio (circular).

    This prevents extreme calibration ratios from the first API call where
    overhead = api_input - local would make ratio = 1.0 trivially.
    """
    # Small conversation: 2 tool groups with tiny results
    messages = _build_single_turn(2, result_size=100)

    strategy = _make_strategy(max_context_tokens=1_000_000)
    assert strategy.calibration_ratio == 1.0

    await strategy(messages)
    our_estimate = strategy.last_included_tokens
    assert our_estimate > 0

    # First calibrate learns overhead, keeps ratio at 1.0
    strategy.calibrate(50_000)
    assert strategy.system_overhead_tokens == 50_000 - our_estimate
    assert strategy.calibration_ratio == 1.0, "First call should skip ratio calibration"


@pytest.mark.asyncio
async def test_calibration_ratio_capped():
    """calibrate() caps the ratio to prevent overestimation from anomalies."""
    messages = _build_single_turn(4, result_size=4000)

    strategy = _make_strategy(max_context_tokens=1_000_000)
    await strategy(messages)
    local = strategy.last_included_tokens
    assert local > 0

    # First call: learn overhead
    overhead = 5000
    strategy.calibrate(local + overhead)

    # Second call: extreme API spike — ratio should be capped at 1.5
    strategy.calibrate((local + overhead) * 10)
    assert strategy.calibration_ratio == 1.5, "Ratio should be capped at _MAX_CALIBRATION_RATIO"

    # 1.2x is within the cap — should be stored as-is
    api = int((local + overhead) * 1.2)
    strategy.calibrate(api)
    expected = api / (local + overhead)
    assert abs(strategy.calibration_ratio - expected) < 0.01


@pytest.mark.asyncio
async def test_calibration_stable_across_approval_resubmission():
    """Approval re-submission doesn't inflate calibration when __call__ re-runs.

    During approval loops, the executor restores history and re-submits with
    approval messages appended.  The strategy's ``__call__`` runs again on
    the full message set (including approval messages), updating
    ``_last_included_tokens`` to match what the API sees.  The calibration
    ratio therefore stays stable.
    """
    messages = _build_multi_turn(3, groups_per_turn=5, result_size=5000)

    strategy = _make_strategy(max_context_tokens=1_000_000)
    await strategy(messages)
    local = strategy.last_included_tokens
    assert local >= 10_000, f"Need substantial local tokens for test, got {local}"

    # First calibrate: learn overhead
    overhead = 5000
    strategy.calibrate(local + overhead)
    assert strategy.system_overhead_tokens == overhead

    # Second calibrate: learn ratio with slight tokenizer drift (1.05x)
    api_tokens = int((local + overhead) * 1.05)
    strategy.calibrate(api_tokens)
    normal_ratio = strategy.calibration_ratio

    # Simulate re-submission: add approval messages, re-run __call__
    approval_msgs = []
    for i in range(3):
        approval_msgs.extend(_build_tool_group(f"approval_{i}", "tool_0", "approved"))
    messages.extend(approval_msgs)

    # __call__ re-runs with ALL messages (including approval), updating
    # _last_included_tokens to include the approval message tokens.
    await strategy(messages)
    local_with_approval = strategy.last_included_tokens
    assert local_with_approval > local

    # Calibrate with the API count that also includes approval tokens.
    api_with_approval = int((local_with_approval + overhead) * 1.05)
    strategy.calibrate(api_with_approval)
    resubmit_ratio = strategy.calibration_ratio

    # Ratio stays stable because both API and local include the same messages.
    assert abs(resubmit_ratio - normal_ratio) < 0.05, (
        f"Re-submission ratio ({resubmit_ratio:.2f}) should ≈ normal ({normal_ratio:.2f})"
    )


@pytest.mark.asyncio
async def test_calibration_accurate_for_tool_heavy_sessions():
    """Tool-heavy sessions calibrate correctly using _last_included_tokens.

    The calibration uses ``_last_included_tokens`` (set by ``__call__`` via
    the framework's ``included_token_count``) as the local baseline.  Since
    ``_usage_pct`` multiplies the same ``included_token_count`` by the ratio,
    the estimated API usage tracks the real API token count regardless of
    content type (text vs function_call vs function_result).
    """
    # Build a tool-heavy conversation: multiple read_file + write_file calls
    # with large file content in function_results.  Very little plain text.
    file_body = "x" * 8000  # ~8k chars per file read
    messages: list[Message] = [_user("Read and update the config files")]
    for i in range(6):
        cid_r = f"read_{i}"
        messages.append(_assistant_tool_call(cid_r, "read_file", {"path": f"/src/file_{i}.py"}))
        messages.append(_tool_result(cid_r, file_body))
        cid_w = f"write_{i}"
        messages.append(_assistant_tool_call(cid_w, "write_file", {"path": f"/src/file_{i}.py", "content": file_body}))
        messages.append(_tool_result(cid_w, "File written successfully."))
    messages.append(_assistant_text("Done updating files."))

    strategy = _make_strategy(max_context_tokens=1_000_000)
    await strategy(messages)
    local = strategy.last_included_tokens
    assert local > 10_000, f"Need substantial tokens for tool-heavy test, got {local}"

    # First calibrate: learn overhead
    system_overhead = 5000
    strategy.calibrate(local + system_overhead)
    assert strategy.system_overhead_tokens == system_overhead

    # Second calibrate: ratio should be ~1.0 (no tokenizer drift)
    strategy.calibrate(local + system_overhead)
    assert abs(strategy.calibration_ratio - 1.0) < 0.01, (
        f"Calibration ratio ({strategy.calibration_ratio:.3f}) should be ~1.0 with no drift"
    )

    # _usage_pct should estimate API tokens accurately
    estimated_pct = strategy._usage_pct(local)
    real_pct = (local + system_overhead) / 1_000_000
    assert abs(estimated_pct - real_pct) < 0.001, f"_usage_pct ({estimated_pct:.4f}) should match real ({real_pct:.4f})"


# ---------------------------------------------------------------------------
# Phase 2: Old turn tool-call removal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase2_removes_old_turn_groups():
    """Phase 2 removes old turn tool-call groups entirely when Phase 1 can't reach target."""
    # 3 turns, small tool results so summaries barely save tokens
    messages = _build_multi_turn(3, groups_per_turn=2, result_size=500)
    total = _estimate_tokens(messages)

    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,  # impossibly low — forces all phases
        on_compaction=_async_appender(received),
    )
    changed = await strategy(messages)
    assert changed

    phases_fired = {r.phase for r in received}
    # Phase 1 should fire (summarise old turns)
    assert "phase1" in phases_fired
    # Phase 2 should fire (remove old turn groups entirely)
    assert "phase2" in phases_fired

    p2 = [r for r in received if r.phase == "phase2"]
    assert p2[0].compacted_groups > 0
    assert p2[0].turn_numbers  # should report which turns were affected


@pytest.mark.asyncio
async def test_phase2_stops_at_target():
    """Phase 2 stops removing old turn groups once usage drops to target."""
    # 4 old turns + 1 current, moderate size
    messages = _build_multi_turn(5, groups_per_turn=2, result_size=2000)
    total = _estimate_tokens(messages)

    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.50,  # reachable after removing a few old turn groups
        on_compaction=_async_appender(received),
    )
    changed = await strategy(messages)
    assert changed

    # If phase 2 fired, it should not have removed ALL old turn groups
    p2 = [r for r in received if r.phase == "phase2"]
    if p2:
        # Not all 4 old turns removed — stopped at target
        assert p2[0].compacted_groups < 4 * 2  # 4 turns * 2 groups


@pytest.mark.asyncio
async def test_phase2_old_turn_user_agent_msgs_survive():
    """Phase 2 removes tool groups but user/agent text messages remain visible.

    Assertions target only *previous* turns — Phase 4 may additionally drop
    the current turn's inline assistant text along with its tool calls.
    """
    messages = _build_multi_turn(3, groups_per_turn=2, result_size=500)
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
    )
    await strategy(messages)

    last_user_idx = max(i for i, m in enumerate(messages) if m.role == "user")

    # User messages survive unconditionally.
    for msg in messages:
        if msg.role == "user":
            assert not msg.additional_properties.get(EXCLUDED_KEY, False), "user messages must survive Phase 2"

    # Previous-turn assistant text must survive Phase 2 removal.
    for idx, msg in enumerate(messages[:last_user_idx]):
        if msg.role == "assistant" and any(c.type == "text" for c in msg.contents):
            if msg.text and msg.text.startswith("[Tool call:"):
                continue
            assert not msg.additional_properties.get(EXCLUDED_KEY, False), (
                f"prior-turn agent text at index {idx} must survive Phase 2"
            )


# ---------------------------------------------------------------------------
# Phase 4: Current turn tool-call removal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase4_removes_current_turn_groups():
    """Phase 4 removes current turn tool-call groups."""
    messages = _build_single_turn(10, result_size=500)
    total = _estimate_tokens(messages)

    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,  # impossibly low — forces Phase 4
        on_compaction=_async_appender(received),
    )
    changed = await strategy(messages)
    assert changed

    phases_fired = {r.phase for r in received}
    assert "phase4" in phases_fired

    p4 = [r for r in received if r.phase == "phase4"]
    assert p4[0].compacted_groups > 0


@pytest.mark.asyncio
async def test_phase4_invokes_last_words_generator():
    """Phase 4 calls the bound LastWordsGenerator with user request + dropped messages."""
    messages = _build_single_turn(4, result_size=1000)
    messages[0] = _user("start <system-reminder>literal</system-reminder>")
    total = _estimate_tokens(messages)

    generator = _StubLastWordsGenerator(text="[stub progress note]")
    reminder = _StubReminderMiddleware()
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
    )
    changed = await strategy(messages)
    assert changed

    assert len(generator.calls) == 1
    call = generator.calls[0]
    assert _scoped_user_texts(call)[0] == "start <system-reminder>literal</system-reminder>"
    assert call["previous_last_words"] is None  # first invocation in the turn
    assert any(any(c.type == "function_call" for c in message.contents) for message in _scoped_messages(call))
    # LAST_WORDS must be stashed on the middleware
    assert reminder.get_last_words() == "[stub progress note]"
    # The round committed (spill + note + exclusions), so the post-commit
    # committed signal fires exactly once.
    assert generator.committed_publishes == 1


@pytest.mark.asyncio
async def test_phase4_refreshes_outgoing_last_words_reminder():
    """After stashing a new note, Phase 4 asks the middleware to refresh the
    current call's outgoing user message — the enrichment ran before the note
    existed, so without the refresh this very request would ship the dropped
    history with a stale note (or none)."""
    messages = _build_single_turn(4, result_size=1000)
    total = _estimate_tokens(messages)

    generator = _StubLastWordsGenerator(text="[stub progress note]")
    reminder = _StubReminderMiddleware()
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
    )
    changed = await strategy(messages)
    assert changed

    assert len(reminder.refresh_calls) == 1
    assert reminder.refresh_calls[0] is messages


@pytest.mark.asyncio
async def test_phase4_outgoing_user_message_carries_fresh_note_with_real_middleware():
    """End-to-end through the real ``SystemReminderMiddleware``: after the
    Phase 4 pass, the outgoing message list itself contains the new note as a
    ``[LAST_WORDS]`` reminder block on the last user message."""
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware

    messages = _build_single_turn(4, result_size=1000)
    total = _estimate_tokens(messages)

    generator = _StubLastWordsGenerator(text="[stub progress note]")
    middleware = SystemReminderMiddleware()
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=middleware,
    )
    changed = await strategy(messages)
    assert changed

    last_user = next(m for m in reversed(messages) if m.role == "user")
    note_blocks = [
        c.text
        for c in last_user.contents
        if c.type == "text" and (c.text or "").startswith("<system-reminder>\n[LAST_WORDS] ")
    ]
    assert len(note_blocks) == 1
    assert "[stub progress note]" in note_blocks[0]


@pytest.mark.asyncio
async def test_phase4_refreshed_note_updates_token_annotations():
    """Regression: the refreshed user message shares its annotation dict with
    the pre-refresh object (exclusion flags must write through), so without a
    recompute its token_count still describes the note-less contents —
    ``tokens_after`` and every later trigger check would undercount by the
    note's size for the rest of the turn, because the incremental annotate
    skips already-counted messages."""
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware

    note = "progress so far: " + "read another module and recorded its collaborators. " * 60
    note_tokens = _tokenizer.count_tokens(note)
    generator = _StubLastWordsGenerator(text=note)
    middleware = SystemReminderMiddleware()

    messages = _build_single_turn(4, result_size=1000)
    original_user = messages[0]
    total = _estimate_tokens(messages)

    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=middleware,
        on_compaction=_async_appender(received),
    )
    changed = await strategy(messages)
    assert changed

    last_user = next(m for m in reversed(messages) if m.role == "user")
    assert last_user is not original_user
    # The recorded count must reflect the refreshed contents, note included —
    # with the stale annotation it would stay at the note-less handful of tokens.
    assert chrys_compaction._token_count(last_user) == chrys_compaction._message_token_estimate(last_user, _tokenizer)
    assert included_token_count(messages) > note_tokens
    p4 = next(r for r in received if r.phase == "phase4")
    assert p4.tokens_after == included_token_count(messages)
    # The next call's incremental annotate must keep the fresh count rather
    # than resurrect the stale one.
    annotate_token_counts(messages, tokenizer=_tokenizer)
    assert included_token_count(messages) == p4.tokens_after
    # Accepted write-through: the dict is shared with the pre-refresh
    # (history) object whose contents lack the note, so the shared count
    # overstates that object — the safe direction (compaction fires earlier).
    assert original_user.additional_properties is last_user.additional_properties


@pytest.mark.asyncio
async def test_phase4_regenerates_with_previous_last_words():
    """Subsequent Phase 4 rounds feed the previous note back into the generator."""
    generator = _StubLastWordsGenerator(text="[note v2]")
    reminder = _StubReminderMiddleware()
    reminder.set_last_words("[note v1]")  # simulate prior Phase 4 in this turn

    messages = _build_single_turn(4, result_size=1000)
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
    )
    await strategy(messages)

    assert generator.calls[0]["previous_last_words"] == "[note v1]"
    assert reminder.get_last_words() == "[note v2]"


@pytest.mark.asyncio
async def test_phase4_passes_only_current_scoped_timeline_and_completer():
    """Phase 4 forwards the client completer with no previous-turn messages."""
    from chrys.kernel import CompactionCallContext

    class _SentinelCompleter:
        async def complete_last_words(self, base_messages, instruction, *, max_output_tokens, on_usage=None):  # type: ignore[no-untyped-def]
            raise AssertionError("stub generator must not invoke the completer")

    messages = _build_multi_turn(2, groups_per_turn=3, result_size=1000)
    total = _estimate_tokens(messages)

    generator = _StubLastWordsGenerator(text="[note]")
    completer = _SentinelCompleter()
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,  # impossibly low — forces phases 1..4
        last_words_generator=generator,
    )

    changed = await strategy(
        messages,
        CompactionCallContext(last_words_completer=completer, request_overhead_tokens=321),
    )
    assert changed

    call = generator.calls[0]
    assert call["completer"] is completer
    assert call["request_overhead_tokens"] == 321
    assert call["system_overhead_tokens"] == strategy.system_overhead_tokens
    assert _scoped_user_texts(call) == ["Turn 2 request"]
    assert all("Turn 1" not in (message.text or "") for message in _scoped_messages(call))


@pytest.mark.asyncio
async def test_strategy_without_context_passes_no_completer():
    """Calling the strategy without a context (legacy call shape) still works
    and hands the generator no completer."""
    messages = _build_single_turn(4, result_size=1000)
    total = _estimate_tokens(messages)

    generator = _StubLastWordsGenerator(text="[note]")
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
    )

    changed = await strategy(messages)
    assert changed
    assert generator.calls[0]["completer"] is None
    assert generator.calls[0]["scoped_groups"]


@pytest.mark.asyncio
async def test_phase4_does_not_drop_when_generator_yields_nothing():
    """If LAST_WORDS generator returns empty and no prior note exists, Phase 4 does NOT drop groups."""

    class _EmptyGenerator:
        async def generate(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return ""

    reminder = _StubReminderMiddleware()
    messages = _build_single_turn(4, result_size=1000)
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=_EmptyGenerator(),
        reminder_middleware=reminder,
    )
    changed = await strategy(messages)
    # No note produced and none pre-existing → nothing excluded by Phase 4.
    excluded_by_phase4 = [m for m in messages if m.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop"]
    assert not excluded_by_phase4
    assert not changed or all(m.additional_properties.get(EXCLUDE_REASON_KEY) != "current_turn_drop" for m in messages)


# ---------------------------------------------------------------------------
# Full tool compaction cascade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_tool_phase_cascade():
    """Old-tool phases and Phase 4 fire in order when target is impossibly low."""
    # 3 old turns + 1 current turn, many groups
    messages = _build_multi_turn(4, groups_per_turn=4, result_size=500)
    total = _estimate_tokens(messages)

    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,  # impossibly low — forces all available phases
        on_compaction=_async_appender(received),
    )
    changed = await strategy(messages)
    assert changed

    phases_fired = [r.phase for r in received]
    # Old-tool phases and current-turn Phase 4 should have fired in order.
    assert "phase1" in phases_fired
    assert "phase2" in phases_fired
    assert "phase4" in phases_fired

    # Verify ordering: phase1 before phase2 before phase4
    phase_order = [p for p in phases_fired if p.startswith("phase")]
    for i in range(len(phase_order) - 1):
        assert phase_order[i] <= phase_order[i + 1], f"Phases fired out of order: {phases_fired}"


@pytest.mark.asyncio
async def test_remove_group_excludes_summary_and_originals():
    """_remove_group excludes both original messages and any Phase 1 summary."""
    messages = _build_multi_turn(2, groups_per_turn=2, result_size=2000)
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
    )
    await strategy(messages)

    # After all phases: old turn tool results should be excluded
    # AND any summaries from Phase 1 for those groups should also be excluded
    old_turn_excluded = 0
    summary_excluded = 0
    for msg in messages:
        if msg.role == "user":
            continue
        # Old turn function_call/result should be excluded
        for content in msg.contents:
            if (
                content.type in ("function_call", "function_result")
                and content.call_id
                and content.call_id.startswith("t0_")
            ):
                assert msg.additional_properties.get(EXCLUDED_KEY, False), (
                    f"Old turn tool message should be excluded: {content.call_id}"
                )
                old_turn_excluded += 1
        # Summary messages for old turn groups should also be excluded by Phase 2
        if msg.message_id and msg.message_id.startswith("tool_summary_"):
            gid = msg.message_id.removeprefix("tool_summary_")
            if gid in strategy._removed_group_ids:
                assert msg.additional_properties.get(EXCLUDED_KEY, False), (
                    f"Summary for removed group should be excluded: {msg.message_id}"
                )
                summary_excluded += 1

    assert old_turn_excluded > 0, "Expected old turn tool messages to be excluded"
    assert summary_excluded > 0, "Expected summary messages for removed groups to be excluded"


@pytest.mark.asyncio
async def test_removed_groups_stay_excluded_across_iterations():
    """Groups removed by Phase 2/3 remain excluded in subsequent strategy calls."""
    messages = _build_multi_turn(3, groups_per_turn=2, result_size=1000)
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
    )

    # First call: compaction fires, groups removed
    changed = await strategy(messages)
    assert changed
    assert len(strategy._removed_group_ids) > 0

    removed_before = set(strategy._removed_group_ids)

    # Simulate tool-loop: framework does list(messages) shallow copy
    messages_copy = list(messages)

    # Second call on the copy: removed groups should stay excluded
    await strategy(messages_copy)

    # _removed_group_ids should not shrink
    assert removed_before.issubset(strategy._removed_group_ids)

    # Messages from removed groups should still be excluded
    for msg in messages_copy:
        from chrys.service.context.compaction import _group_id as gid_fn

        gid = gid_fn(msg)
        if gid and gid in removed_before:
            assert msg.additional_properties.get(EXCLUDED_KEY, False), (
                f"Removed group {gid} should stay excluded across iterations"
            )


# ---------------------------------------------------------------------------
# Reminder debug dump
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reminder_debug_dump_written_on_compaction(tmp_path):
    """A changed compaction dumps the injected LAST_WORDS reminder, owner-only."""
    messages = _build_multi_turn(3, groups_per_turn=2, result_size=2000)
    total = _estimate_tokens(messages)
    reminder = _StubReminderMiddleware()
    reminder.set_last_words("[dump-me progress note]")
    debug_dir = tmp_path / "compactions" / "debug"
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.50,
        reminder_middleware=reminder,
        debug_log_dir=debug_dir,
    )
    changed = await strategy(messages)
    assert changed

    dumps = sorted(debug_dir.glob("reminder_*.log"))
    assert len(dumps) == 1
    text = dumps[0].read_text(encoding="utf-8")
    assert text.startswith("<system-reminder>\n[LAST_WORDS]")
    assert "[dump-me progress note]" in text
    if os.name == "posix":
        assert (dumps[0].stat().st_mode & 0o777) == 0o600


@pytest.mark.asyncio
async def test_reminder_debug_dump_skipped_without_debug_dir_or_content(tmp_path):
    """No debug dir or nothing to inject -> no dump and no error."""
    messages = _build_multi_turn(3, groups_per_turn=2, result_size=2000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.50,
    )
    assert await strategy(messages)

    # Empty reminder state renders None -> nothing written even with a dir.
    messages = _build_multi_turn(3, groups_per_turn=2, result_size=2000)
    total = _estimate_tokens(messages)
    debug_dir = tmp_path / "compactions" / "debug"
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.50,
        debug_log_dir=debug_dir,
    )
    assert await strategy(messages)
    # render returned None before any filesystem work — the dir is never created
    assert not debug_dir.exists()


# ---------------------------------------------------------------------------
# Multi-turn compaction integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multi_turn_progressive_compaction():
    """Progressive: each call compacts oldest turns first as context grows."""
    # 4 turns with moderate content
    messages = _build_multi_turn(4, groups_per_turn=2, result_size=2000)
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.90,
        target_pct=0.50,
    )
    changed = await strategy(messages)
    assert changed

    # Verify summaries exist (from old turns being compacted)
    summaries = [m for m in messages if m.text and m.text.startswith("[Tool call:")]
    assert len(summaries) > 0

    total_after = included_token_count(messages)
    assert total_after < total


@pytest.mark.asyncio
async def test_ratio_affects_trigger():
    """High calibration ratio makes the trigger fire earlier."""
    messages = _build_single_turn(4, result_size=1000)
    total = _estimate_tokens(messages)

    # Without calibration (ratio=1.0): messages fill ~50% of context → no trigger
    strategy = _make_strategy(max_context_tokens=total * 2, trigger_pct=0.85, target_pct=0.30)
    changed = await strategy(messages)
    assert not changed

    # Ratio of 2.0 pushes effective usage to ~100% → triggers compaction
    strategy._calibration_ratio = 2.0
    changed = await strategy(messages)
    assert changed


@pytest.mark.asyncio
async def test_request_overhead_floor_triggers_fresh_strategy_earlier() -> None:
    from chrys.kernel import CompactionCallContext

    messages = _build_multi_turn(2, groups_per_turn=2, result_size=500)
    total = _estimate_tokens(messages)
    without_floor = _make_strategy(max_context_tokens=total * 2, trigger_pct=0.75, target_pct=0.50)
    with_floor = _make_strategy(max_context_tokens=total * 2, trigger_pct=0.75, target_pct=0.50)

    assert not await without_floor(messages)
    assert await with_floor(messages, CompactionCallContext(request_overhead_tokens=total))


@pytest.mark.asyncio
async def test_legacy_tool_definition_tokens_still_raise_the_overhead_floor() -> None:
    """External constructors may populate only the retained legacy field.

    The strategy folds it with ``max()``; reading the new field exclusively
    would size admission as though tools cost zero for such callers.
    """
    from chrys.kernel import CompactionCallContext

    messages = _build_multi_turn(2, groups_per_turn=2, result_size=500)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(max_context_tokens=total * 2, trigger_pct=0.75, target_pct=0.50)

    assert await strategy(messages, CompactionCallContext(tool_definition_tokens=total))


def test_compaction_call_context_keeps_legacy_tool_definition_field() -> None:
    from chrys.kernel import CompactionCallContext

    context = CompactionCallContext(tool_definition_tokens=123)

    assert context.tool_definition_tokens == 123
    assert context.request_overhead_tokens == 0


def test_strategy_satisfies_admission_protocol() -> None:
    """The runtime protocol gates the outbound clamp: if the real strategy
    stops conforming, the clamp silently no-ops in production while the
    double-based kernel tests keep passing."""
    from chrys.kernel import CompactionAdmissionState

    assert isinstance(_make_strategy(max_context_tokens=100), CompactionAdmissionState)


@pytest.mark.parametrize(
    ("overhead", "ratio"),
    [
        (-1, 1.0),
        (100, 1.0),
        (0, 0.0),
        (0, 1.500_001),
        (0, float("nan")),
        (0, float("inf")),
        (True, 1.0),
        (0, False),
    ],
)
def test_restore_calibration_rejects_invalid_values(overhead: object, ratio: object) -> None:
    strategy = _make_strategy(max_context_tokens=100)

    assert not strategy.restore_calibration(overhead, ratio)
    assert not strategy.calibration_initialized


def test_restore_calibration_hydrates_valid_values() -> None:
    strategy = _make_strategy(max_context_tokens=100)

    assert strategy.restore_calibration(20, 1.25)
    assert strategy.calibration_initialized
    assert strategy.system_overhead_tokens == 20
    assert strategy.calibration_ratio == 1.25


@pytest.mark.asyncio
async def test_middleware_calibrates_strategy():
    """UsageTrackingMiddleware calls calibrate() on the strategy."""
    from unittest.mock import MagicMock

    from chrys.kernel import ChatResponse
    from chrys.service.context.middleware.usage import UsageTrackingMiddleware

    strategy = _make_strategy(max_context_tokens=2_000_000)
    messages = _build_single_turn(4, result_size=4000)
    await strategy(messages)
    our_estimate = strategy.last_included_tokens

    mw = UsageTrackingMiddleware(
        max_context_tokens=2_000_000,
        compaction_strategy=strategy,
    )

    # First call: middleware calibrates — learns overhead
    overhead = 5000
    api_tokens = our_estimate + overhead
    context = MagicMock()
    context.messages = []
    context.stream = False
    mock_result = MagicMock(spec=ChatResponse)
    mock_result.usage_details = {
        "input_token_count": api_tokens,
        "output_token_count": 5_000,
        "total_token_count": api_tokens + 5_000,
    }
    context.result = mock_result

    async def call_next():
        pass

    await mw.process(context, call_next)

    # First call learns overhead, ratio stays 1.0
    assert strategy.system_overhead_tokens == overhead
    assert strategy.calibration_ratio == 1.0

    # Second call: learns ratio
    mock_result.usage_details = {
        "input_token_count": int((our_estimate + overhead) * 1.1),
        "output_token_count": 5_000,
        "total_token_count": int((our_estimate + overhead) * 1.1) + 5_000,
    }
    await mw.process(context, call_next)
    assert abs(strategy.calibration_ratio - 1.1) < 0.01


# ---------------------------------------------------------------------------
# Sub-agent (single-turn) compaction — phase 4 only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_turn_sub_agent_only_fires_phase4():
    """Sub-agents are single-turn: only phase 4 should fire (current turn removal).

    Simulates the Explore sub-agent scenario: one user message followed by
    many tool calls.  With only one turn, ``previous_turns`` is empty so
    phases 1-2 are no-ops.  Phase 4 drops every current-turn tool-call
    group once LAST_WORDS has been produced.
    """
    # Single turn: user + 10 tool groups + assistant (simulates Explore sub-agent)
    messages = _build_single_turn(10, result_size=2000)
    total = _estimate_tokens(messages)

    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.85,
        target_pct=0.50,
        on_compaction=_async_appender(received),
    )
    changed = await strategy(messages)
    assert changed

    # Only phase 4 callbacks — never phase 1/2
    assert all(info.phase == "phase4" for info in received), (
        f"Expected only phase4, got phases: {[info.phase for info in received]}"
    )
    assert len(received) == 1
    assert received[0].compacted_groups > 0
    # Phase 4 reports the resolver's current-span absolute number (§4.1).
    assert received[0].turn_numbers == [1]


@pytest.mark.asyncio
async def test_single_turn_sub_agent_no_compaction_under_threshold():
    """Sub-agent with low usage doesn't trigger compaction at all."""
    messages = _build_single_turn(3, result_size=500)

    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=1_000_000,  # way over — usage << 85%
        trigger_pct=0.85,
        target_pct=0.50,
        on_compaction=_async_appender(received),
    )
    changed = await strategy(messages)
    assert not changed
    assert len(received) == 0


# ---------------------------------------------------------------------------
# Tool-loop shared-object contamination regression test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_excluded_flags_reset_prevents_tool_loop_message_loss():
    """Regression: stale _excluded flags on shared Message objects must not
    cause silent loss of NEW messages when the strategy sees a copied list.

    Chrys mutates the active tool-loop list in place, but still defends
    copied/replayed lists where Message objects are shared
    and ``project_included_messages`` filters ``_excluded=True``.

    This test verifies that newly added tool results (batch 2) are NOT
    excluded, while legitimately compacted originals remain excluded and
    their summaries are present.
    """
    # ── Build a multi-turn conversation that will trigger compaction ──
    # Turn 1: old turn with large tool results
    history: list[Message] = [
        _user("Turn 1 question"),
        *_build_tool_group("t1_c0", "search", "x" * 3000),
        *_build_tool_group("t1_c1", "read_file", "x" * 3000),
        _assistant_text("Turn 1 answer"),
    ]
    # Turn 2 (current): tool loop in progress
    history.append(_user("Turn 2 question"))
    # Batch 1: glob + grep
    batch1 = [
        *_build_tool_group("t2_glob", "glob", "x" * 2000),
        *_build_tool_group("t2_grep", "grep", "x" * 2000),
    ]
    history.extend(batch1)

    total = _estimate_tokens(history)

    # Same strategy instance used across both iterations (matching real framework)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.50,
    )

    # ── Iteration 1: the loop shallow-copies, compacts, projects ──
    shallow_copy_1 = list(history)  # list copy, shared Message objects
    changed = await strategy(shallow_copy_1)
    assert changed, "Strategy should trigger on iteration 1"

    projected_1 = project_included_messages(shallow_copy_1)
    assert len(projected_1) < len(shallow_copy_1), "Some messages should be excluded"

    # Verify some original Message objects now have _excluded=True (shared)
    excluded_in_original = [m for m in history if m.additional_properties.get(EXCLUDED_KEY, False)]
    assert len(excluded_in_original) > 0, (
        "Shared Message objects should have _excluded=True after strategy runs on the copy"
    )

    # ── Simulate tool loop: add new batch (edit_file) to the accumulator ──
    batch2 = _build_tool_group("t2_edit", "edit_file", "ok")
    history.extend(batch2)

    # ── Iteration 2: same strategy, higher max_context so usage is below trigger ──
    strategy.max_context_tokens = 1_000_000
    shallow_copy_2 = list(history)
    changed = await strategy(shallow_copy_2)
    assert not changed, "Strategy should NOT trigger on iteration 2"

    projected_2 = project_included_messages(shallow_copy_2)

    # ── KEY ASSERTIONS ──
    # 1. Newly added batch2 messages must NOT be excluded
    for m in batch2:
        assert not m.additional_properties.get(EXCLUDED_KEY, False), (
            "New tool results from batch 2 should not be excluded"
        )

    # 2. Batch 2 messages should appear in the projected output
    batch2_ids = {id(m) for m in batch2}
    projected_ids = {id(m) for m in projected_2}
    assert batch2_ids <= projected_ids, "Newly added batch 2 messages must survive projection"


@pytest.mark.asyncio
async def test_no_double_compaction_across_tool_loop_iterations():
    """Regression: compaction must not re-compact already-compacted groups
    when a later compaction pass receives a copied list.

    Chrys keeps summaries on the active loop list; this keeps the defensive
    copied/replayed-list path covered. The strategy must cache
    and re-inject summaries so that:
    1. Already-compacted groups keep ``_excluded=True`` and are skipped
    2. The callback fires only once per group (no duplicate events)
    3. ``project_included_messages`` returns summaries (not bare originals)
    """
    # Build a multi-turn conversation that triggers compaction
    # Turn 1: old turn with large tool results
    history: list[Message] = [
        _user("Turn 1 question"),
        *_build_tool_group("t1_c0", "search", "x" * 3000),
        *_build_tool_group("t1_c1", "read_file", "x" * 3000),
        _assistant_text("Turn 1 answer"),
    ]
    # Turn 2 (current): tool loop in progress
    history.append(_user("Turn 2 question"))
    batch1 = [
        *_build_tool_group("t2_glob", "glob", "x" * 2000),
        *_build_tool_group("t2_grep", "grep", "x" * 2000),
    ]
    history.extend(batch1)

    total = _estimate_tokens(history)

    callback_events: list[CompactionInfo] = []

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.50,
        on_compaction=_async_appender(callback_events),
    )

    # ── Iteration 1: framework shallow-copies, strategy compacts ──
    copy_1 = list(history)
    assert await strategy(copy_1), "Strategy should compact on iteration 1"
    project_included_messages(copy_1)

    # Should have fired callback(s) for phase 1
    assert len(callback_events) > 0, "Callback should fire on iteration 1"
    events_after_iter1 = len(callback_events)

    # ── Simulate tool loop: LLM responds, adds a new tool call ──
    batch2 = _build_tool_group("t2_edit", "edit_file", "ok")
    history.extend(batch2)

    # ── Iteration 2: same strategy, new copy — should NOT re-compact ──
    copy_2 = list(history)
    await strategy(copy_2)
    projected_2 = project_included_messages(copy_2)

    # The key assertions:
    # 1. No additional callback events from re-compacting the same groups
    assert len(callback_events) == events_after_iter1, (
        f"Expected {events_after_iter1} callback events but got {len(callback_events)}. "
        f"Strategy fired duplicate compaction events on iteration 2."
    )

    # 2. Projected messages should include summaries (not bare originals)
    projected_texts = [m.text or "" for m in projected_2]
    has_summary = any("[Tool call:" in t for t in projected_texts)
    assert has_summary, (
        "Projected messages should contain summary text from compaction, but no summaries found — re-injection failed."
    )

    # 3. No original excluded messages should leak through
    excluded_in_projected = [m for m in projected_2 if m.additional_properties.get(EXCLUDED_KEY, False)]
    assert len(excluded_in_projected) == 0, "No excluded messages should appear in projected output"


@pytest.mark.asyncio
async def test_after_run_callback_not_suppressed_by_stale_summary_cache():
    """Regression: after_run compaction callback must fire when no intra-run
    compaction happened, even if _summary_cache has entries from a prior run.

    Scenario:
      - Run 1: compaction triggers during tool loop → _summary_cache populated,
        callback fires correctly.
      - Run 2: usage grows but only crosses the threshold in after_run
        (not during the tool loop).  The after_run compaction callback must
        still fire because no intra-run compaction occurred in THIS run.

    Before the fix, ``already_compacted = bool(_summary_cache)`` was always
    True after the first-ever compaction, permanently suppressing after_run
    callbacks and hiding Compact:Phase* events from the debug panel.
    """
    from unittest.mock import MagicMock

    from chrys.service.context.providers.history import CompressibleHistoryProvider

    tokenizer = MixedLanguageTokenizer()

    # ── Setup: strategy + callback tracking ──
    callback_events: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=500,
        trigger_pct=0.80,
        target_pct=0.50,
        on_compaction=_async_appender(callback_events),
        tokenizer=tokenizer,
    )

    provider = CompressibleHistoryProvider(compaction_strategy=strategy)

    # Minimal mocks for lifecycle hooks
    mock_agent = MagicMock()
    mock_session = MagicMock()
    mock_session.state = {}
    mock_context = MagicMock()
    mock_context.session_id = "test-session"

    # ── Run 1: build history that triggers compaction ──
    state_run1: dict = {
        "messages": _build_multi_turn(3, groups_per_turn=2, result_size=300),
        "compressed_msgs": [],
        "turn_counter": 3,
    }

    # Simulate before_run
    await provider.before_run(agent=mock_agent, session=mock_session, context=mock_context, state=state_run1)

    # Simulate intra-run compaction (framework calls strategy during tool loop)
    changed = await strategy(list(state_run1["messages"]))
    assert changed, "Run 1: strategy should trigger compaction"
    # Phase 4 populates _removed_group_ids (and Phase 1 may populate _summary_cache)
    compaction_state_run1 = len(strategy._summary_cache) + len(strategy._removed_group_ids)
    assert compaction_state_run1 > 0, "Run 1: compaction state should be populated"
    events_after_run1_intra = len(callback_events)
    assert events_after_run1_intra > 0, "Run 1: intra-run callback should have fired"

    # Simulate after_run — callback should be suppressed (intra-run already fired)
    await provider.after_run(agent=mock_agent, session=mock_session, context=mock_context, state=state_run1)
    events_after_run1_after = len(callback_events)
    # after_run may or may not compact further, but if it does the callback is
    # correctly suppressed because intra-run already handled it
    assert events_after_run1_after == events_after_run1_intra, (
        "Run 1: after_run should not produce duplicate callback events"
    )

    # ── Run 2: add more messages, but DON'T trigger intra-run compaction ──
    new_turn_messages = [
        _user("Run 2 question"),
        *_build_tool_group("r2_c0", "search", "y" * 300),
        *_build_tool_group("r2_c1", "read_file", "y" * 300),
        _assistant_text("Run 2 response"),
    ]
    state_run2 = state_run1  # same state dict, as engine reuses it
    state_run2["messages"].extend(new_turn_messages)

    # Simulate before_run for run 2 — this snapshots _summary_cache size
    await provider.before_run(agent=mock_agent, session=mock_session, context=mock_context, state=state_run2)

    # Track compaction state before after_run
    state_before = len(strategy._summary_cache) + len(strategy._removed_group_ids)

    # Now simulate after_run — compaction should trigger AND callback must fire
    # because no intra-run compaction happened in run 2.
    events_before_run2_after = len(callback_events)
    await provider.after_run(agent=mock_agent, session=mock_session, context=mock_context, state=state_run2)

    # The key assertion: if after_run compacted anything, the callback must
    # have fired.  We check by seeing if compaction state grew.
    state_after = len(strategy._summary_cache) + len(strategy._removed_group_ids)
    compaction_happened_in_after_run = state_after > state_before

    if compaction_happened_in_after_run:
        assert len(callback_events) > events_before_run2_after, (
            f"Run 2 after_run compacted but callback was suppressed. "
            f"This means Compact:Phase* events are lost. "
            f"Compaction state was {state_before} before, {state_after} after."
        )


# ---------------------------------------------------------------------------
# Session restore — fresh strategy must preserve prior compaction flags
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_restore_preserves_phase1_compaction():
    """A fresh strategy must not clear _excluded flags set by prior Phase 1 compaction.

    Simulates session restore: run strategy #1 to compact, then create
    a fresh strategy #2 (empty caches) and run it on the same messages.
    The flags from strategy #1 should be preserved.
    """
    messages = _build_multi_turn(4, groups_per_turn=3, result_size=3000)
    total = _estimate_tokens(messages)

    strategy1 = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.90,
        target_pct=0.50,
    )
    changed = await strategy1(messages)
    assert changed, "Strategy #1 should have compacted"

    excluded_after_s1 = {id(m) for m in messages if m.additional_properties.get(EXCLUDED_KEY, False)}
    tokens_after_s1 = included_token_count(messages)
    assert excluded_after_s1, "Strategy #1 should have excluded some messages"

    # Create a fresh strategy (simulating session restore — empty caches)
    strategy2 = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.90,
        target_pct=0.50,
    )
    assert not strategy2._summary_cache
    assert not strategy2._removed_group_ids

    events: list[CompactionInfo] = []
    strategy2._on_compaction = _async_appender(events)
    changed2 = await strategy2(messages)

    excluded_after_s2 = {id(m) for m in messages if m.additional_properties.get(EXCLUDED_KEY, False)}
    tokens_after_s2 = included_token_count(messages)

    # The fresh strategy must not un-exclude previously compacted messages
    assert excluded_after_s1 <= excluded_after_s2, "Fresh strategy cleared flags that were set by prior compaction"
    assert tokens_after_s2 <= tokens_after_s1 * 1.05, (
        f"Token count inflated after fresh strategy: {tokens_after_s2:,} vs {tokens_after_s1:,}"
    )
    # No redundant compaction callback should fire
    assert not events, f"Fresh strategy fired {len(events)} compaction events (expected 0)"
    assert not changed2


@pytest.mark.asyncio
async def test_session_restore_preserves_phase2_removal():
    """A fresh strategy must not clear _excluded flags set by Phase 2 removal."""
    messages = _build_multi_turn(4, groups_per_turn=3, result_size=3000)
    total = _estimate_tokens(messages)

    # Force Phase 2 by using a very low target that Phase 1 alone can't reach
    strategy1 = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.90,
        target_pct=0.10,
    )
    changed = await strategy1(messages)
    assert changed

    # Verify Phase 2 actually removed groups (check for budget_tool_removal reason)
    removals = [m for m in messages if m.additional_properties.get(EXCLUDE_REASON_KEY) == "budget_tool_removal"]
    assert removals, "Phase 2 should have removed some tool groups"

    excluded_after_s1 = {id(m) for m in messages if m.additional_properties.get(EXCLUDED_KEY, False)}
    tokens_after_s1 = included_token_count(messages)

    # Fresh strategy (session restore)
    strategy2 = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.90,
        target_pct=0.10,
    )
    events: list[CompactionInfo] = []
    strategy2._on_compaction = _async_appender(events)
    await strategy2(messages)

    excluded_after_s2 = {id(m) for m in messages if m.additional_properties.get(EXCLUDED_KEY, False)}
    tokens_after_s2 = included_token_count(messages)

    assert excluded_after_s1 <= excluded_after_s2
    assert tokens_after_s2 <= tokens_after_s1 * 1.05, (
        f"Token count inflated: {tokens_after_s2:,} vs {tokens_after_s1:,}"
    )
    assert not events


@pytest.mark.asyncio
async def test_fresh_strategy_clears_non_compaction_excluded():
    """Flags without compaction _exclude_reason must still be cleared by a fresh strategy."""
    messages = _build_multi_turn(2, groups_per_turn=2, result_size=500)

    # Manually set _excluded on some messages WITHOUT a compaction reason
    messages[1].additional_properties[EXCLUDED_KEY] = True
    messages[2].additional_properties[EXCLUDED_KEY] = True

    strategy = _make_strategy(max_context_tokens=1_000_000)
    await strategy(messages)

    # Those manually-excluded messages should have been cleared
    assert not messages[1].additional_properties.get(EXCLUDED_KEY, False)
    assert not messages[2].additional_properties.get(EXCLUDED_KEY, False)


@pytest.mark.asyncio
async def test_session_restore_token_count_not_inflated():
    """Reproduces the reported bug: after session restore, included_token_count
    was inflated from ~88K (API) to ~157K because the fresh strategy cleared
    all _excluded flags, un-hiding previously compacted tool results.

    After the fix, the fresh strategy preserves compaction flags, so the
    included_token_count should remain close to the post-compaction value.
    """
    # Build a large session: 6 turns with big tool results (simulating read_file)
    messages = _build_multi_turn(6, groups_per_turn=4, result_size=5000)
    total = _estimate_tokens(messages)

    # Run strategy #1 — should compact old turns
    strategy1 = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.85,
        target_pct=0.50,
    )
    changed = await strategy1(messages)
    assert changed
    tokens_after_compaction = included_token_count(messages)

    # Calculate what the inflated count WOULD have been before the fix:
    # clearing all flags means all messages are visible
    all_visible_tokens = 0
    for m in messages:
        ann = m.additional_properties.get("_group", {})
        tc = ann.get("token_count")
        if isinstance(tc, int):
            all_visible_tokens += tc

    assert all_visible_tokens > tokens_after_compaction * 1.3, (
        "Test setup: all-visible should be significantly larger than compacted"
    )

    # Create fresh strategy (session restore) and run on same messages
    strategy2 = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.85,
        target_pct=0.50,
    )
    await strategy2(messages)
    tokens_after_restore = included_token_count(messages)

    # Key assertion: token count should NOT inflate back to the all-visible level
    assert tokens_after_restore <= tokens_after_compaction * 1.05, (
        f"Token count inflated after session restore: "
        f"{tokens_after_restore:,} (expected ~{tokens_after_compaction:,}, "
        f"all-visible would be {all_visible_tokens:,})"
    )


@pytest.mark.asyncio
async def test_session_restore_with_if_branch_preserves_old_flags():
    """When the current session HAS done new compaction (if branch), old
    compaction flags from a prior session must still be preserved."""
    # Build messages, compact with strategy #1
    messages = _build_multi_turn(4, groups_per_turn=3, result_size=3000)
    total_before = _estimate_tokens(messages)

    strategy1 = _make_strategy(
        max_context_tokens=total_before + 100,
        trigger_pct=0.90,
        target_pct=0.50,
    )
    await strategy1(messages)
    excluded_after_s1 = {id(m) for m in messages if m.additional_properties.get(EXCLUDED_KEY, False)}

    # Add a new turn with lots of content (simulating new user interaction)
    new_turn = [
        _user("New big request"),
        *_build_tool_group("new_c0", "search", "z" * 5000),
        *_build_tool_group("new_c1", "read_file", "z" * 5000),
        *_build_tool_group("new_c2", "write_file", "z" * 5000),
        _assistant_text("New big response"),
    ]
    messages.extend(new_turn)
    total_with_new = _estimate_tokens(messages)

    # Create strategy #2 (fresh — simulating session restore) and compact
    strategy2 = _make_strategy(
        max_context_tokens=total_with_new + 100,
        trigger_pct=0.85,
        target_pct=0.50,
    )
    await strategy2(messages)

    # Old excluded messages must still be excluded
    for m in messages:
        if id(m) in excluded_after_s1:
            assert m.additional_properties.get(EXCLUDED_KEY, False), "Old compaction flag was cleared by strategy #2"


# ---------------------------------------------------------------------------
# Phase 4 additional edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_phase4_preserves_user_message() -> None:
    """The turn opener (user message) must survive Phase 4 drop-all.

    Phase 4 collects all current-turn groups and drops them, but the
    user-message group is explicitly filtered out in compaction.py.
    """
    messages = _build_single_turn(5, result_size=1500)
    user_msg = messages[0]
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
    )
    await strategy(messages)

    assert not user_msg.additional_properties.get(EXCLUDED_KEY, False), (
        "User message (turn opener) must never be excluded by Phase 4"
    )


@pytest.mark.asyncio
async def test_phase4_propagates_generator_exception() -> None:
    """If the generator raises, the exception must propagate out of the
    strategy so the caller can fail without dropping current-turn work.

    LAST_WORDS is LLM-only — there is no programmatic fallback.  Generator
    failures (transport error, empty response → ``LastWordsGenerationError``)
    are retried by the generator against the same compaction input before
    it raises.  Messages must not be mutated before the raise.
    """

    class _ExplodingGenerator:
        async def generate(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    reminder = _StubReminderMiddleware()
    messages = _build_single_turn(4, result_size=1000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=_ExplodingGenerator(),
        reminder_middleware=reminder,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await strategy(messages)

    # No note set, and no group excluded — the strategy raised before
    # applying Phase 4 exclusions, so retry sees a clean state.
    assert reminder.get_last_words() is None
    assert not any(m.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop" for m in messages)
    breaker = reminder.get_drop_round_breaker()
    assert breaker.attempts == 1
    assert breaker.consecutive_no_progress == 1
    assert breaker.tail_override is True


@pytest.mark.asyncio
async def test_phase4_does_not_drop_new_delta_when_regeneration_returns_empty() -> None:
    """A pre-existing note cannot authorize dropping a newly generated delta."""

    class _EmptyGenerator:
        async def generate(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            return ""

    reminder = _StubReminderMiddleware()
    reminder.set_last_words("[prior note]")
    messages = _build_single_turn(4, result_size=1000)
    total = _estimate_tokens(messages)

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=_EmptyGenerator(),
        reminder_middleware=reminder,
    )
    changed = await strategy(messages)
    assert not changed
    assert reminder.get_last_words() == "[prior note]"
    assert not any(m.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop" for m in messages)


@pytest.mark.asyncio
async def test_phase4_unexpected_spill_io_awaits_true_durability_then_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reminder = _StubReminderMiddleware()
    durability_calls = 0

    async def persist() -> bool:
        nonlocal durability_calls
        await asyncio.sleep(0)
        durability_calls += 1
        return True

    monkeypatch.setattr(
        "chrys.service.context.compaction.current_turn_drop.write_spill_batch",
        lambda *_args, **_kwargs: _failed_spill_result(),
    )
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    generator = _StubLastWordsGenerator()
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
    )
    strategy.set_recovery_persistence_callback(persist)

    assert await strategy(messages)

    assert durability_calls == 1
    assert reminder.get_last_words() is not None
    assert reminder.get_last_words_manifest()[0]["no_record_reason"] == "I/O failure"
    assert any(message.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop" for message in messages)
    # Recovery persistence made the round durable — the committed signal fires.
    assert generator.committed_publishes == 1


@pytest.mark.asyncio
async def test_phase4_commit_marks_quota_evicted_manifest_records_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reminder = _StubReminderMiddleware()
    evicted_path = "compactions/dropped/turn001/001_old_00000001.md"
    reminder.append_manifest(
        [
            ManifestEntry(
                record_id="old-record",
                group_id="old-group",
                record_dir="compactions/dropped/turn001",
                relative_path=evicted_path,
                turn=1,
                round=1,
                sequence=1,
                tool="old",
                display_argument="",
                outcome="ok",
                size_chars=10,
            )
        ]
    )
    spill_result = SpillBatchResult(
        entries=(
            SpillManifestItem(
                record_id="new-record",
                group_id="new-group",
                record_dir="compactions/dropped/turn001",
                relative_path="compactions/dropped/turn001/001_new_00000002.md",
                turn=1,
                round=1,
                sequence=1,
                tool="new",
                display_argument="",
                outcome="ok",
                size_chars=10,
            ),
        ),
        evicted_relative_paths=frozenset({evicted_path}),
    )
    monkeypatch.setattr(
        "chrys.service.context.compaction.current_turn_drop.write_spill_batch",
        lambda *_args, **_kwargs: spill_result,
    )
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        reminder_middleware=reminder,
    )

    assert await strategy(messages)

    manifest = reminder.get_last_words_manifest()
    assert manifest[0]["relative_path"] == evicted_path
    assert manifest[0]["available"] is False
    assert manifest[1]["record_id"] == "new-record"


@pytest.mark.asyncio
@pytest.mark.parametrize("durability", ["false", "exception", "absent"])
async def test_phase4_unexpected_spill_io_vetoes_without_true_durability(
    durability: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reminder = _StubReminderMiddleware()
    monkeypatch.setattr(
        "chrys.service.context.compaction.current_turn_drop.write_spill_batch",
        lambda *_args, **_kwargs: _failed_spill_result(),
    )
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    generator = _StubLastWordsGenerator()
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
    )
    if durability == "false":

        async def persist_false() -> bool:
            return False

        strategy.set_recovery_persistence_callback(persist_false)
    elif durability == "exception":

        async def persist_error() -> bool:
            raise OSError("sidecar failed")

        strategy.set_recovery_persistence_callback(persist_error)

    changed = await strategy(messages)

    assert not changed
    assert reminder.get_last_words() is None
    assert reminder.get_last_words_manifest() == []
    assert not any(message.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop" for message in messages)
    breaker = reminder.get_drop_round_breaker()
    assert (breaker.attempts, breaker.consecutive_no_progress, breaker.tail_override) == (1, 1, True)
    # An abandoned round must never claim the committed signal.
    assert generator.committed_publishes == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellation_point", ["generate", "persist_recovery"])
async def test_phase4_precommit_cancellation_retains_persisted_accounting_without_no_progress(
    cancellation_point: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ChargingGenerator(_StubLastWordsGenerator):
        async def generate(self, *args, spend_side_call_tokens=None, **kwargs):  # type: ignore[no-untyped-def]
            assert spend_side_call_tokens is not None
            assert spend_side_call_tokens(123)
            if cancellation_point == "generate":
                raise asyncio.CancelledError
            return await super().generate(*args, spend_side_call_tokens=spend_side_call_tokens, **kwargs)

    reminder = _StubReminderMiddleware()
    pressure_events: list[str] = []
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    generator = _ChargingGenerator(text="[note]")
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
        spill_root=tmp_path,
        spill_quota=SpillQuota(),
        spill_session_id="precommit-cancel",
        on_context_pressure=lambda reason, _breaker, _budget: pressure_events.append(reason),
    )

    if cancellation_point == "persist_recovery":
        real_write_spill_batch = compaction_mod.write_spill_batch

        def spill_then_report_projection_failure(*args, **kwargs):  # type: ignore[no-untyped-def]
            result = real_write_spill_batch(*args, **kwargs)
            return SpillBatchResult(
                entries=result.entries,
                unexpected_io_failure=True,
                evicted_relative_paths=result.evicted_relative_paths,
            )

        async def cancel_recovery_persistence() -> bool:
            raise asyncio.CancelledError

        monkeypatch.setattr(
            "chrys.service.context.compaction.current_turn_drop.write_spill_batch",
            spill_then_report_projection_failure,
        )
        strategy.set_recovery_persistence_callback(cancel_recovery_persistence)

    with pytest.raises(asyncio.CancelledError):
        await strategy(messages)

    breaker = reminder.get_drop_round_breaker()
    assert breaker.attempts == 1
    assert breaker.side_call_tokens == 123
    assert breaker.consecutive_no_progress == 0
    assert breaker.tail_override is False
    assert breaker.disabled is False
    assert pressure_events == []
    assert reminder.get_last_words() is None
    assert reminder.get_last_words_manifest() == []
    assert not any(message.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop" for message in messages)
    if cancellation_point == "persist_recovery":
        assert catalog_live_records(tmp_path)


@pytest.mark.asyncio
async def test_phase4_cancel_during_committed_publish_leaves_commit_finalized() -> None:
    """Cancellation delivered by the committed publish cannot split the commit.

    The publish is deliberately the first await after the synchronous commit
    block — by then the exclusion anchors and token bookkeeping must already
    be in place, or an interrupt-resume that persisted no anchors would
    retain both the original tool history and the LAST_WORDS note.
    """

    class _CancelOnCommittedGenerator(_StubLastWordsGenerator):
        async def publish_committed(self) -> None:
            raise asyncio.CancelledError

    generator = _CancelOnCommittedGenerator(text="[note]")
    reminder = _StubReminderMiddleware()
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    compaction_infos: list[CompactionInfo] = []

    async def record_compaction(info: CompactionInfo) -> None:
        compaction_infos.append(info)

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
        on_compaction=record_compaction,
    )

    with pytest.raises(asyncio.CancelledError):
        await strategy(messages)

    excluded = [m for m in messages if m.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop"]
    assert excluded
    assert reminder.get_last_words() == "[note]"
    assert len(strategy._excluded_anchors) == len(excluded)
    assert strategy._last_included_tokens == included_token_count(messages)
    # The on_compaction notification (ToolCompacted for the main agent) must
    # survive the interrupt on a detached task — the rounds it reports are
    # already durably committed.
    await wait_for(
        lambda: bool(compaction_infos),
        timeout=ENGINE_TURN_TIMEOUT,
        description="phase-4 compaction info",
    )
    assert compaction_infos[0].phase == "phase4"
    assert compaction_infos[0].compacted_groups > 0
    assert compaction_infos[0].last_words_generated is True
    assert compaction_infos[0].tokens_after == strategy._last_included_tokens
    assert not strategy._detached_deliveries


class _BlocksThePhaseAck(FakeSink):
    """Real sink shape: the line lands, then its acknowledgement keeps waiting."""

    def __init__(self) -> None:
        super().__init__()
        self.phase_ack_reached = asyncio.Event()
        self.release_phase_ack = asyncio.Event()

    async def emit(
        self, draft: EventDraft, *, payload_factory: Callable[[int], Mapping[str, Any]] | None = None
    ) -> EmitResult:
        result = await super().emit(draft, payload_factory=payload_factory)
        if draft.event_type == TrajectoryEventType.COMPACTION_PHASE_FINISHED:
            self.phase_ack_reached.set()
            await self.release_phase_ack.wait()
        return result


@pytest.mark.asyncio
async def test_phase4_records_its_phase_before_the_pass_can_close_the_run() -> None:
    """Cancelling the pass closes the run from the unwind, while the delivery
    it scheduled runs on undisturbed behind its shield. The phase's lines —
    including the segments continuing its arrays — must all be on the log by
    then, or a sub-event lands behind the terminal of its own parent."""
    generator = _StubLastWordsGenerator(text="[note]")
    reminder = _StubReminderMiddleware()
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocking_on_compaction(info: CompactionInfo) -> None:
        entered.set()
        await release.wait()

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
        on_compaction=blocking_on_compaction,
    )
    sink = _BlocksThePhaseAck()

    with trajectory_scope(make_context(sink)):
        run = asyncio.create_task(strategy(messages))
        # Whichever of the two the delivery reaches first: awaiting the ack of
        # the phase it writes, or the observer callback after it.
        await wait_for(
            lambda: sink.phase_ack_reached.is_set() or entered.is_set(),
            timeout=ENGINE_TURN_TIMEOUT,
            description="phase-4 delivery in flight",
        )
        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run
        sink.release_phase_ack.set()
        release.set()
        await wait_for(
            lambda: not strategy._detached_deliveries,
            timeout=ENGINE_TURN_TIMEOUT,
            description="detached compaction delivery drain",
        )

    closed = sink.event_types.index(TrajectoryEventType.COMPACTION_FINISHED)
    assert sink.event_types.index(TrajectoryEventType.COMPACTION_PHASE_FINISHED) < closed
    assert [index for index, name in enumerate(sink.event_types) if name == SEGMENT_EVENT_TYPE]
    assert all(index < closed for index, name in enumerate(sink.event_types) if name == SEGMENT_EVENT_TYPE)


@pytest.mark.asyncio
async def test_phase4_cancel_during_post_commit_notification_still_delivers() -> None:
    """Cancellation while the end-of-pass on_compaction await is in flight
    must not lose ToolCompacted.

    The commit is already durable by then, so the delivery runs as an
    anchored task behind a shield: cancellation still propagates to the
    caller, but the callback finishes detached.  Without the shield the
    cancellation would kill the only delivery of an already-committed
    round.
    """
    generator = _StubLastWordsGenerator(text="[note]")
    reminder = _StubReminderMiddleware()
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)

    entered = asyncio.Event()
    release = asyncio.Event()
    compaction_infos: list[CompactionInfo] = []

    async def blocking_on_compaction(info: CompactionInfo) -> None:
        entered.set()
        await release.wait()
        compaction_infos.append(info)

    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
        on_compaction=blocking_on_compaction,
    )

    run = asyncio.create_task(strategy(messages))
    await entered.wait()
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    # The delivery survived the cancellation on its detached anchor and
    # completes once the (EventBus-shaped) handler unblocks.
    assert not compaction_infos
    assert strategy._detached_deliveries
    release.set()
    await wait_for(
        lambda: bool(compaction_infos),
        timeout=ENGINE_TURN_TIMEOUT,
        description="detached phase-4 compaction info",
    )
    assert compaction_infos[0].phase == "phase4"
    assert compaction_infos[0].compacted_groups > 0
    assert compaction_infos[0].last_words_generated is True
    await wait_for(
        lambda: not strategy._detached_deliveries,
        timeout=ENGINE_TURN_TIMEOUT,
        description="detached compaction delivery drain",
    )


@pytest.mark.asyncio
async def test_stale_current_turn_drop_anchors_do_not_exclude_next_turn_twin() -> None:
    """Exclusion anchors are single-use — consumed by the after_run persist.

    Under-trigger turns return before ``_snapshot_excluded_anchors``, so
    without consumption the previous trigger-passing pass's anchors stay
    resident and ``after_run`` replays them every turn.  Pass 3 structural
    matching scans newest-first, so a later turn that emits a structural
    twin of a dropped message (repeated identical tool call or byte-equal
    assistant text) would bind the NEW live message and silently exclude
    it — with no code path ever clearing a ``current_turn_drop`` flag.
    """
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
    )

    changed = await strategy(messages)

    assert changed
    dropped = [m for m in messages if m.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop"]
    assert dropped
    assert strategy._excluded_anchors

    # after_run of the triggering turn.  Streaming finalizers rebuild the
    # current-turn response messages before storing them, so state holds
    # structural twins: distinct objects without the wire-minted
    # message_id — exactly the pass-3 binding path.
    state_messages = [
        _user("start"),
        *_build_tool_group("call_0", "tool_0", "x" * 1_000),
        *_build_tool_group("call_1", "tool_1", "x" * 1_000),
        _assistant_text("done"),
    ]
    strategy.persist_exclusions_to_state(state_messages)

    first_pass_excluded = [
        m for m in state_messages if m.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop"
    ]
    assert len(first_pass_excluded) == len(dropped)
    # Consumed: one snapshot pass, one persist pass.
    assert strategy._excluded_anchors == []

    # Next turn stays under trigger (no compaction, no re-snapshot) and
    # happens to repeat the same tool call and closing text.  Its
    # after_run persist must not bind the resident anchors onto these
    # live messages.
    next_turn = [
        _user("again"),
        *_build_tool_group("call_0", "tool_0", "x" * 1_000),
        _assistant_text("done"),
    ]
    state_messages.extend(next_turn)
    strategy.persist_exclusions_to_state(state_messages)

    for live in next_turn:
        assert live.additional_properties.get(EXCLUDED_KEY, False) is False


@pytest.mark.asyncio
async def test_retry_rollback_snapshot_discards_attempt_anchors() -> None:
    """Retry rollback restores anchor state captured at attempt entry.

    Phase 4 can commit mid-attempt and record current-turn-drop anchors; a
    transient-error rollback then restores pre-attempt history, so the
    dropped messages never reach state.  Replaying those anchors at
    after_run would let the newest-first structural fallback bind the
    successful retry's identical calls/text and silently exclude live
    messages — the retry loop therefore restores the anchor list together
    with history.
    """
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
    )

    pre_attempt = strategy.snapshot_retry_state()
    changed = await strategy(messages)
    assert changed
    assert strategy._excluded_anchors

    strategy.restore_retry_state(pre_attempt)
    assert strategy._excluded_anchors == []

    # The successful retry emits the same tool calls and closing text;
    # after_run persists against state holding those live twins — they
    # must stay included.
    state_messages = [
        _user("start"),
        *_build_tool_group("call_0", "tool_0", "x" * 1_000),
        *_build_tool_group("call_1", "tool_1", "x" * 1_000),
        _assistant_text("done"),
    ]
    strategy.persist_exclusions_to_state(state_messages)
    for live in state_messages:
        assert live.additional_properties.get(EXCLUDED_KEY, False) is False


def test_retry_state_restore_keeps_pre_attempt_anchors() -> None:
    """Anchors predating the attempt are restored, not wiped.

    Anchors surviving an interrupted prior run (after_run never fired) are
    still awaiting their one persist pass; a retry rollback inside the
    next run must put them back exactly as captured.
    """
    strategy = _make_strategy()
    prior = _assistant_text("from prior run")
    prior.additional_properties[EXCLUDED_KEY] = True
    prior.additional_properties[EXCLUDE_REASON_KEY] = "current_turn_drop"
    strategy._snapshot_excluded_anchors([prior])
    pre_attempt = strategy.snapshot_retry_state()
    assert pre_attempt

    # Mid-attempt Phase 4 replaces the list before the attempt fails.
    strategy._excluded_anchors = []
    strategy.restore_retry_state(pre_attempt)
    assert tuple(strategy._excluded_anchors) == pre_attempt.anchors


@pytest.mark.asyncio
async def test_retry_state_restores_phase4_content_but_keeps_breaker_monotonic(tmp_path: Path) -> None:
    """Phase-4 note context rolls back; real per-turn safety spend does not."""
    todo = ["[TODO] baseline"]
    spill_quota = SpillQuota()
    baseline_path = "compactions/dropped/turn001/baseline.md"
    spill_quota.initialize(1, available_relative_paths=[baseline_path])
    reminder = SystemReminderMiddleware(
        session_root=tmp_path,
        spill_quota=spill_quota,
        todo_state_provider=lambda: todo[0],
    )
    reminder.prepare_turn()
    reminder.set_last_words("[baseline note]")
    reminder.append_manifest(
        [
            ManifestEntry(
                record_id="baseline-record",
                group_id="baseline-group",
                record_dir="compactions/dropped/turn001",
                relative_path=baseline_path,
                turn=1,
                round=1,
                sequence=1,
                tool="baseline_tool",
                display_argument="baseline",
                outcome="completed",
                size_chars=8,
                assistant_text=False,
                available=True,
            )
        ]
    )
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=_StubLastWordsGenerator(text="[attempt note]"),
        reminder_middleware=reminder,
        spill_root=tmp_path,
        spill_quota=spill_quota,
        spill_session_id="phase4-retry",
    )
    retry_snapshot = strategy.snapshot_retry_state()
    todo[0] = "[TODO] changed during attempt"

    assert await strategy(messages)
    attempt_breaker = reminder.get_drop_round_breaker()
    assert attempt_breaker.attempts == 1
    assert reminder.get_last_words() == "[attempt note]"
    assert len(reminder.get_last_words_manifest()) > 1
    assert reminder.claim_context_pressure_notification()
    # The provider-side catalog can evict a record while this attempt is in
    # flight; retry restore must revalidate the snapshotted row via SpillQuota.
    spill_quota.initialize(0, available_relative_paths=[])

    strategy.restore_retry_state(retry_snapshot)

    assert reminder.get_last_words() == "[baseline note]"
    restored_manifest = reminder.get_last_words_manifest()
    assert [row["record_id"] for row in restored_manifest] == ["baseline-record"]
    assert restored_manifest[0]["available"] is False
    rendered = reminder.render_last_words_reminder_text()
    assert rendered is not None
    assert "[TODO] baseline" in rendered
    assert "changed during attempt" not in rendered
    assert reminder.get_drop_round_breaker() == attempt_breaker
    assert not reminder.claim_context_pressure_notification()


@pytest.mark.asyncio
async def test_phase4_cancel_after_catalog_flush_leaves_catalog_visible_and_reminder_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reminder = _StubReminderMiddleware()
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        reminder_middleware=reminder,
        spill_root=tmp_path,
        spill_quota=SpillQuota(),
        spill_session_id="cancel-session",
    )

    async def flush_then_cancel(function, /, *args, **kwargs):  # type: ignore[no-untyped-def]
        function(*args, **kwargs)
        raise asyncio.CancelledError

    monkeypatch.setattr(compaction_mod.asyncio, "to_thread", flush_then_cancel)

    with pytest.raises(asyncio.CancelledError):
        await strategy(messages)

    assert catalog_live_records(tmp_path)
    assert reminder.get_last_words() is None
    assert reminder.get_last_words_manifest() == []
    assert not any(message.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop" for message in messages)


@pytest.mark.asyncio
async def test_phase4_archives_superseded_note_record_alongside_group_records(tmp_path: Path) -> None:
    """A later round's merge archives the pre-merge note verbatim on disk."""
    generator = _StubLastWordsGenerator(text="[note v2]")
    reminder = _StubReminderMiddleware()
    reminder.set_last_words("[note v1]")
    messages = _build_single_turn(4, result_size=1000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
        spill_root=tmp_path,
        spill_quota=SpillQuota(),
        spill_session_id="note-session",
    )

    assert await strategy(messages)

    manifest = reminder.get_last_words_manifest()
    # The stub middleware stores rows without validating; every row must also
    # survive the REAL middleware's from_state, or it would silently vanish
    # from the rendered reminder (regression: empty note group_id).
    for row in manifest:
        assert ManifestEntry.from_state(row) is not None, row
    note_rows = [row for row in manifest if row["tool"] == "last_words"]
    assert len(note_rows) == 1
    note_row = note_rows[0]
    assert note_row["outcome"] == "merged"
    assert note_row["sequence"] == max(row["sequence"] for row in manifest)
    record = (tmp_path / note_row["relative_path"]).read_text(encoding="utf-8")
    assert record.startswith("# Superseded LAST_WORDS note\n")
    assert "[note v1]" in record
    kinds = {record.record_id: record.kind for record in catalog_live_records(tmp_path)}
    assert kinds.pop(note_row["record_id"]) == "note"
    assert set(kinds.values()) == {"group"}


@pytest.mark.asyncio
async def test_phase4_first_drop_without_previous_note_writes_no_note_record(tmp_path: Path) -> None:
    reminder = _StubReminderMiddleware()
    messages = _build_single_turn(4, result_size=1000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        reminder_middleware=reminder,
        spill_root=tmp_path,
        spill_quota=SpillQuota(),
        spill_session_id="note-session",
    )

    assert await strategy(messages)

    assert reminder.get_last_words() is not None
    assert all(row["tool"] != "last_words" for row in reminder.get_last_words_manifest())
    assert all(record.kind == "group" for record in catalog_live_records(tmp_path))


@pytest.mark.asyncio
async def test_phase4_cancellation_drains_real_spill_worker_before_propagating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = Event()
    release = Event()
    finished = Event()

    def blocked_spill(*_args, **_kwargs) -> SpillBatchResult:  # type: ignore[no-untyped-def]
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release spill worker")
        finished.set()
        return SpillBatchResult(entries=())

    monkeypatch.setattr("chrys.service.context.compaction.current_turn_drop.write_spill_batch", blocked_spill)
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        reminder_middleware=_StubReminderMiddleware(),
    )
    task = asyncio.create_task(strategy(messages))
    started_seen = False
    completed_while_blocked = False
    cancelled = False
    try:
        started_seen = await wait_until(started.is_set, timeout=1)
        task.cancel()
        completed_while_blocked = await wait_until(task.done, timeout=0.2, interval=0.01)
    finally:
        release.set()
        if not task.done() and not task.cancelling():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            cancelled = True

    assert started_seen
    assert not completed_while_blocked
    assert finished.is_set()
    assert cancelled


def test_phase4_commit_through_exclusion_contains_no_awaits() -> None:
    source = textwrap.dedent(inspect.getsource(CurrentTurnDropRound.run))
    tree = ast.parse(source)
    lines = source.splitlines()
    start = next(index for index, line in enumerate(lines, start=1) if "Steps 4-5: synchronous commit" in line)
    # The synchronous window ends at the deliberately-post-commit
    # ``publish_committed`` await — the first await allowed after the
    # note-set/manifest/exclusion/usage-sample block completes.
    end = next(index for index, line in enumerate(lines, start=1) if index > start and "publish_committed" in line)

    awaits = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Await) and start < node.lineno < end]

    assert awaits == []


@pytest.mark.asyncio
async def test_phase4_same_turn_retry_observes_failed_attempt_count() -> None:
    class FailOnceGenerator:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first failure")
            return "[recovered note]"

        async def publish_committed(self) -> None:
            return None

    generator = FailOnceGenerator()
    reminder = _StubReminderMiddleware()
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
    )

    with pytest.raises(RuntimeError, match="first failure"):
        await strategy(messages)
    assert reminder.get_drop_round_breaker().attempts == 1

    assert await strategy(messages)
    assert reminder.get_drop_round_breaker().attempts == 2
    assert reminder.get_drop_round_breaker().consecutive_no_progress == 0


@pytest.mark.asyncio
async def test_phase4_side_call_budget_trips_before_attempt_and_records_failed_attempt() -> None:
    class BudgetGenerator:
        async def generate(self, *_args, spend_side_call_tokens=None, **_kwargs):  # type: ignore[no-untyped-def]
            assert spend_side_call_tokens is not None
            if not spend_side_call_tokens(100):
                raise LastWordsSpendBudgetExceeded("budget")
            raise AssertionError("budget-equal attempt must not run")

    reminder = _StubReminderMiddleware()
    events: list[str] = []
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=BudgetGenerator(),
        reminder_middleware=reminder,
        phase4_side_call_token_budget=100,
        on_context_pressure=lambda reason, _breaker, _budget: events.append(reason),
    )

    assert not await strategy(messages)

    breaker = reminder.get_drop_round_breaker()
    assert breaker.attempts == 1
    assert breaker.side_call_tokens == 100
    assert breaker.consecutive_no_progress == 1
    assert breaker.tail_override is True
    assert breaker.disabled is True
    assert events == ["side_call_budget"]


@pytest.mark.asyncio
async def test_phase4_second_no_progress_disables_and_publishes_pressure_event() -> None:
    generator = _StubLastWordsGenerator(text="")
    reminder = _StubReminderMiddleware()
    events: list[tuple[str, object, int]] = []
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
        on_context_pressure=lambda reason, breaker, budget: events.append((reason, breaker, budget)),
    )

    assert not await strategy(messages)
    first = reminder.get_drop_round_breaker()
    assert first.tail_override is True and first.disabled is False
    assert not await strategy(messages)
    second = reminder.get_drop_round_breaker()
    assert second.consecutive_no_progress == 2
    assert second.disabled is True
    assert events and events[-1][0] == "generation_failure"

    assert not await strategy(messages)
    assert len(generator.calls) == 2
    assert reminder.get_drop_round_breaker().attempts == 2
    assert len(events) == 1


@pytest.mark.asyncio
async def test_phase4_restored_disabled_breaker_publishes_pressure_once() -> None:
    generator = _StubLastWordsGenerator()
    reminder = _StubReminderMiddleware()
    from chrys.service.agent_middleware.system_reminder import DropRoundBreakerState

    reminder.set_drop_round_breaker(DropRoundBreakerState(attempts=2, consecutive_no_progress=2, disabled=True))
    events: list[str] = []
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
        on_context_pressure=lambda reason, _breaker, _budget: events.append(reason),
    )

    assert not await strategy(messages)
    assert not await strategy(messages)

    assert generator.calls == []
    assert events == ["disabled"]


@pytest.mark.asyncio
async def test_phase4_success_freeing_less_than_five_points_sets_tail_override() -> None:
    reminder = _StubReminderMiddleware()
    messages = _build_single_turn(1, result_size=5)
    messages[0] = _user("large opener " + "context " * 20_000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        reminder_middleware=reminder,
    )

    assert await strategy(messages)

    breaker = reminder.get_drop_round_breaker()
    assert breaker.attempts == 1
    assert breaker.consecutive_no_progress == 1
    assert breaker.tail_override is True
    assert breaker.disabled is False


@pytest.mark.asyncio
async def test_phase4_entry_round_limit_disables_before_generator_call() -> None:
    generator = _StubLastWordsGenerator()
    reminder = _StubReminderMiddleware()
    from chrys.service.agent_middleware.system_reminder import DropRoundBreakerState
    from chrys.service.context.compaction import _MAX_DROP_ROUNDS_PER_TURN

    reminder.set_drop_round_breaker(DropRoundBreakerState(attempts=_MAX_DROP_ROUNDS_PER_TURN, side_call_tokens=10))
    events: list[str] = []
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
        on_context_pressure=lambda reason, _breaker, _budget: events.append(reason),
    )

    assert not await strategy(messages)

    assert generator.calls == []
    assert reminder.get_drop_round_breaker().disabled is True
    assert events == ["round_limit"]
    # The first trip surfaces one terminal failure card with the reason.
    assert generator.breaker_trips == [f"{_MAX_DROP_ROUNDS_PER_TURN} attempts limit exceeded for current turn"]

    # Later triggers re-enter through "disabled" and stay silent.
    assert not await strategy(messages)
    assert len(generator.breaker_trips) == 1


@pytest.mark.asyncio
async def test_phase4_unlimited_budget_never_refuses_spend_or_entry() -> None:
    generator = _StubLastWordsGenerator(text="[note]")
    reminder = _StubReminderMiddleware()
    from chrys.service.agent_middleware.system_reminder import DropRoundBreakerState

    reminder.set_drop_round_breaker(DropRoundBreakerState(side_call_tokens=10_000_000))
    messages = _build_single_turn(2, result_size=1_000)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
    )

    assert await strategy(messages)

    assert len(generator.calls) == 1
    # Spend still accumulates for observability but never refuses.
    assert strategy._spend_side_call_tokens(5_000) is True
    assert reminder.get_drop_round_breaker().side_call_tokens == 10_005_000
    assert generator.breaker_trips == []


@pytest.mark.asyncio
async def test_phase4_includes_tool_call_and_result_in_dropped_messages() -> None:
    """Dropped_messages must contain both function_call and function_result contents."""
    generator = _StubLastWordsGenerator(text="[note]")
    reminder = _StubReminderMiddleware()
    messages = _build_single_turn(3, result_size=1500)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=reminder,
    )
    await strategy(messages)

    assert generator.calls
    scoped_messages = _scoped_messages(generator.calls[0])
    has_call = any(any(c.type == "function_call" for c in message.contents) for message in scoped_messages)
    has_result = any(any(c.type == "function_result" for c in message.contents) for message in scoped_messages)
    assert has_call, "Dropped messages should contain the assistant tool-call message"
    assert has_result, "Dropped messages should contain the tool-result message"


@pytest.mark.asyncio
async def test_phase4_noop_when_collaborators_unbound() -> None:
    """When the strategy has no reminder middleware and no generator,
    Phase 4 must not silently drop current-turn work.

    This is the scenario the sub-agent path fell into before the fix —
    without collaborators ``have_note`` is False and the ``if have_note:``
    guard must prevent any exclusions.
    """
    messages = _build_single_turn(5, result_size=1500)
    total = _estimate_tokens(messages)
    # Build a strategy WITHOUT wiring collaborators (explicitly None).
    strategy = UnifiedContextStrategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
    )
    # No set_reminder_middleware / set_last_words_generator calls.
    changed = await strategy(messages)
    # Either unchanged, or only phases 1/2 fired.  Phase 4 must not have
    # produced any current_turn_drop exclusions.
    assert not any(m.additional_properties.get(EXCLUDE_REASON_KEY) == "current_turn_drop" for m in messages), (
        "Phase 4 dropped groups without a LAST_WORDS note — silent amputation"
    )
    _ = changed  # outcome is fine either way; what matters is no silent drop


@pytest.mark.asyncio
async def test_phase4_reports_last_words_generated_flag() -> None:
    """CompactionInfo.last_words_generated=True when a note was produced."""
    received: list[CompactionInfo] = []
    messages = _build_single_turn(4, result_size=1500)
    total = _estimate_tokens(messages)
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        on_compaction=_async_appender(received),
    )
    await strategy(messages)
    p4 = [info for info in received if info.phase == "phase4"]
    assert p4, "Phase 4 should have fired"
    assert p4[0].last_words_generated is True


# ---------------------------------------------------------------------------
# Turn-boundary unification integration pins (§7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_p1_p2_skip_injected_current_turn_groups():
    """Behavior-change pin: a mid-turn injection no longer splits the current
    task, so its pre-injection tool groups are protected from P1/P2 mechanical
    truncation and handed to Phase 4 instead — while real previous turns are
    still compacted."""
    messages = [
        _user("Turn 1 request"),
        *_build_tool_group("t1_c0", "search", "x" * 3000),
        *_build_tool_group("t1_c1", "read_file", "x" * 3000),
        _assistant_text("Turn 1 answer"),
        _user("Turn 2 request"),
        *_build_tool_group("t2_pre", "grep", "x" * 3000),
        _assistant_text("working"),
        _injected("mid-turn constraint"),
        *_build_tool_group("t2_post", "glob", "x" * 3000),
    ]
    total = _estimate_tokens(messages)

    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.85,
        target_pct=0.01,  # forces every phase to run
        on_compaction=_async_appender(received),
    )
    changed = await strategy(messages)
    assert changed

    p12_numbers = {n for r in received if r.phase in ("phase1", "phase2") for n in r.turn_numbers}
    assert p12_numbers == {1}, f"P1/P2 must only touch turn 1, reported {p12_numbers}"

    # The pre-injection current-turn group survives P1/P2 and is dropped by
    # Phase 4 (current_turn_drop), never truncated as a fake previous turn.
    for call_id in ("t2_pre", "t2_post"):
        for msg in messages:
            if _has_call_id(msg, call_id):
                assert msg.additional_properties.get(EXCLUDE_REASON_KEY) == _REASON_CURRENT_TURN_DROP, (
                    f"{call_id} should be Phase-4 dropped, got {msg.additional_properties.get(EXCLUDE_REASON_KEY)!r}"
                )

    injected_msg = next(m for m in messages if m.additional_properties.get(HistoryMarkerKind.INJECTED_KEY))
    assert not injected_msg.additional_properties.get(EXCLUDED_KEY, False)

    p4 = [r for r in received if r.phase == "phase4"]
    assert p4 and p4[0].turn_numbers == [2]


@pytest.mark.asyncio
async def test_provider_prefix_user_never_treated_as_previous_turn():
    """§3.2 rule 1 integration: a provider-contributed user-role prefix (the
    context_management prompt shape) opens no span, so P1/P2 never compact
    "turns" cut from provider context and the prefix itself is untouched."""
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    state["messages"].append(_user("Turn 1 request"))
    state["messages"].extend(_build_tool_group("t1_c0", "search", "x" * 3000))
    state["messages"].append(_assistant_text("Turn 1 answer"))
    CompressibleHistoryProvider.insert_marker(state, 1)
    state["messages"].append(_user("Turn 2 request"))
    t2_group = _build_tool_group("t2_c0", "grep", "x" * 3000)
    state["messages"].extend(t2_group)

    prefix_user = _user("Workspace context digest " + "p " * 1500)
    wire = [prefix_user, *_wire_view(state)]
    total = _estimate_tokens(wire)

    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.85,
        target_pct=0.01,
        on_compaction=_async_appender(received),
    )
    strategy.bind_state(state)
    changed = await strategy(wire)
    assert changed

    p12_numbers = {n for r in received if r.phase in ("phase1", "phase2") for n in r.turn_numbers}
    assert p12_numbers == {1}
    assert not prefix_user.additional_properties.get(EXCLUDED_KEY, False)
    for msg in t2_group:
        assert msg.additional_properties.get(EXCLUDE_REASON_KEY) == _REASON_CURRENT_TURN_DROP
    p4 = [r for r in received if r.phase == "phase4"]
    assert p4 and p4[0].turn_numbers == [2]


@pytest.mark.asyncio
async def test_p1_re_resolves_spans_after_summary_insertions():
    """§4.1 one-coherent-snapshot pin: Phase 1's summary insertions shift
    later turns' indices, so each turn iteration must re-resolve.  Turn 1
    holds three groups (three insertions); turn 2's last group starts within
    three messages of its span end — a stale pre-P1 span would exclude it."""
    messages: list[Message] = [_user("Turn 1 request")]
    for g in range(3):
        messages.extend(_build_tool_group(f"t0_c{g}", f"tool_a{g}", "x" * 3000))
    messages.append(_assistant_text("Turn 1 answer"))
    messages.append(_user("Turn 2 request"))
    messages.extend(_build_tool_group("t1_c0", "tool_b0", "x" * 3000))
    messages.extend(_build_tool_group("t1_last", "tool_b_last", "x" * 3000))
    messages.append(_assistant_text("Turn 2 answer"))
    # Current turn: text only, keeps Phase 4 out of the event stream.
    messages.append(_user("Turn 3 request"))
    messages.append(_assistant_text("Turn 3 answer"))
    total = _estimate_tokens(messages)

    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.85,
        target_pct=0.01,
        on_compaction=_async_appender(received),
    )
    changed = await strategy(messages)
    assert changed

    p1 = [r for r in received if r.phase == "phase1"]
    assert p1
    assert p1[0].compacted_groups == 5, f"stale spans skipped a shifted group: {p1[0].tool_names}"
    assert "tool_b_last" in p1[0].tool_names
    assert p1[0].turn_numbers == [1, 2]


def test_remove_group_never_inserts():
    """P2's phase-entry resolution is sufficient only because _remove_group
    never inserts messages — pin that contract."""
    messages = _build_single_turn(2, result_size=500)
    _estimate_tokens(messages)  # annotate groups
    strategy = _make_strategy()
    grouped = _group_messages_by_id(messages)
    kinds = _group_kind_map(messages)
    target_gid = next(g for g in _ordered_group_ids(messages) if kinds.get(g) == "tool_call")
    before_len = len(messages)

    result = strategy._remove_group(messages, target_gid, grouped[target_gid])

    assert result
    assert len(messages) == before_len


@pytest.mark.parametrize("result_text", ["", "x"], ids=["empty-result", "tiny-result"])
def test_compact_group_never_raises_included_tokens_for_partially_excluded_group(result_text: str) -> None:
    messages = _build_tool_group("c1", "read_file", result_text)
    set_excluded(messages[0], excluded=True, reason="cross_turn_compression")
    before = _estimate_tokens(messages)
    grouped = _group_messages_by_id(messages)
    group_id = _tool_group_id_of(messages, "c1")
    strategy = _make_strategy()

    result = strategy._compact_group(messages, group_id, grouped[group_id])
    assert included_token_count(messages) == before
    assert result is None


def test_compact_group_skips_group_already_removed_by_phase2() -> None:
    call = _assistant_tool_call("c1", "read_file", args={"path": "secret.txt"})
    narration = _assistant_text("checking secret payload")
    result = _tool_result("c1", "secret payload")
    messages = [call, narration, result]
    _estimate_tokens(messages)
    grouped = _group_messages_by_id(messages)
    group_id = _tool_group_id_of(messages, "c1")
    strategy = _make_strategy()

    assert strategy._remove_group(messages, group_id, grouped[group_id])
    assert strategy._compact_group(messages, group_id, grouped[group_id]) is None

    summary_id = f"tool_summary_{group_id}"
    assert summary_id not in strategy._summary_cache
    assert not any(
        message.message_id == summary_id and not message.additional_properties.get(EXCLUDED_KEY) for message in messages
    )


@pytest.mark.asyncio
async def test_phase4_injections_and_nudge_survive_note_rides_last_user():
    """P4 integration: injections and the nudge survive the drop, tool work on
    BOTH sides of the injection is dropped along with inline agent text, and
    the [LAST_WORDS] reminder rides the last user message (the nudge)."""
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware

    messages = [
        _user("start the big task"),
        *_build_tool_group("pre_c0", "search", "x" * 2000),
        _assistant_text("interim findings"),
        _injected("also check the docs"),
        *_build_tool_group("post_c0", "read_file", "x" * 2000),
        _nudge(),
    ]
    total = _estimate_tokens(messages)

    generator = _StubLastWordsGenerator(text="[stub progress note]")
    middleware = SystemReminderMiddleware()
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.90,
        target_pct=0.01,
        last_words_generator=generator,
        reminder_middleware=middleware,
    )
    changed = await strategy(messages)
    assert changed

    for call_id in ("pre_c0", "post_c0"):
        for msg in messages:
            if _has_call_id(msg, call_id):
                assert msg.additional_properties.get(EXCLUDE_REASON_KEY) == _REASON_CURRENT_TURN_DROP
    interim = next(m for m in messages if (m.text or "") == "interim findings")
    assert interim.additional_properties.get(EXCLUDED_KEY, False)

    projected = project_included_messages(messages)
    user_msgs = [m for m in projected if m.role == "user"]
    assert len(user_msgs) == 3  # opener + injection + nudge all survive
    assert user_msgs[1].additional_properties.get(HistoryMarkerKind.INJECTED_KEY)
    assert user_msgs[2].additional_properties.get(HistoryMarkerKind.CONTINUATION_KEY)

    note_blocks = [
        c.text
        for c in user_msgs[2].contents
        if c.type == "text" and (c.text or "").startswith("<system-reminder>\n[LAST_WORDS] ")
    ]
    assert len(note_blocks) == 1 and "[stub progress note]" in note_blocks[0]
    assert not any(
        (c.text or "").startswith("<system-reminder>\n[LAST_WORDS] ")
        for m in user_msgs[:2]
        for c in m.contents
        if c.type == "text"
    )

    # The generator receives the ordered current-turn user groups.
    call = generator.calls[0]
    assert _scoped_user_texts(call) == ["start the big task", "also check the docs", "continue"]
    assert call["has_continuation_nudges"] is True
    assert call["degraded_opener"] is False


@pytest.mark.asyncio
async def test_phase4_reports_last_healthy_plus_one_on_unstripped_crash_cluster():
    """§4.1 divergence pin: an unstripped crash cluster leaves turn_counter
    one high; the P4 event still reports last-healthy + 1 and P1/P2 never
    truncate the crashed turn's work (crash-cluster twin, §3.2 rule 1)."""
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    state["messages"].append(_user("Turn 1 request " + "x " * 1500))
    state["messages"].append(_assistant_text("Turn 1 answer " + "y " * 1500))
    CompressibleHistoryProvider.insert_marker(state, 1)
    opener2 = _user("Turn 2 request")
    state["messages"].append(opener2)
    work2 = _build_tool_group("t2_c0", "search", "z" * 4000)
    state["messages"].extend(work2)
    state["messages"].append(_status_marker(HistoryMarkerKind.INTERRUPTED))
    CompressibleHistoryProvider.insert_marker(state, 2)  # turn_counter now 2
    guidance = _injected("pick it back up with flag X")
    state["messages"].append(guidance)

    wire = _wire_view(state)
    total = _estimate_tokens(wire)

    received: list[CompactionInfo] = []
    compressed_events: list = []

    async def _on_compress(info) -> None:
        compressed_events.append(info)

    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.85,
        target_pct=0.01,
        on_compaction=_async_appender(received),
        on_compress=_on_compress,
    )
    strategy.bind_state(state)
    changed = await strategy(wire)
    assert changed

    # Turn 1 is text-only: P1/P2 fire no events, and the crashed turn's work
    # is never bucketed as a previous turn.
    assert not [r for r in received if r.phase in ("phase1", "phase2")]
    # P3 folds only the completed turn 1.
    assert [info.turn_range for info in compressed_events] == [(1, 1)]
    # The crashed turn's work is Phase-4 dropped, never folded/truncated.
    for msg in work2:
        assert msg.additional_properties.get(EXCLUDE_REASON_KEY) == _REASON_CURRENT_TURN_DROP
    assert not opener2.additional_properties.get(EXCLUDED_KEY, False)
    assert not guidance.additional_properties.get(EXCLUDED_KEY, False)

    p4 = [r for r in received if r.phase == "phase4"]
    assert p4
    # turn_counter+1 would say 3; the resolver reports last-healthy + 1.
    assert p4[0].turn_numbers == [2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status_kind",
    [HistoryMarkerKind.INTERRUPTED, HistoryMarkerKind.AWAITING_SUB_AGENTS],
)
async def test_p3_clamp_skips_interior_marker_protects_current_work(status_kind: str):
    """§4.1 P3 clamp: a crash-leftover interior marker inside the current real
    turn is never folded — current-task work survives while the real previous
    turn still folds.  Health is STATUS_MARKERS-driven, so the paused
    awaiting-sub-agents cluster behaves identically."""
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    state["messages"].append(_user("Turn 1 request " + "x " * 1500))
    state["messages"].append(_assistant_text("Turn 1 answer " + "y " * 1500))
    CompressibleHistoryProvider.insert_marker(state, 1)
    opener2 = _user("Turn 2 request")
    state["messages"].append(opener2)
    work2 = _build_tool_group("t2_c0", "search", "z" * 3000)
    state["messages"].extend(work2)
    state["messages"].append(_status_marker(status_kind))
    CompressibleHistoryProvider.insert_marker(state, 2)

    wire = _wire_view(state)
    total = _estimate_tokens(wire)

    compressed_events: list = []

    async def _on_compress(info) -> None:
        compressed_events.append(info)

    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.85,
        target_pct=0.01,  # unreachable: forces P3 to attempt every marker
        on_compress=_on_compress,
    )
    strategy.bind_state(state)
    changed = await strategy(wire)
    assert changed

    # Only turn 1 folded; the interior marker's candidate was clamped.
    assert [info.turn_range for info in compressed_events] == [(1, 1)]
    for msg in work2:
        assert msg.additional_properties.get(EXCLUDE_REASON_KEY) == _REASON_CURRENT_TURN_DROP, (
            "current-task work must reach Phase 4, never P3 folding"
        )
    # The interior marker survives unconsumed in provider state.
    markers = CompressibleHistoryProvider.list_compressed(state)["markers"]
    assert [m["marker_id"] for m in markers] == ["turn_2"]


@pytest.mark.asyncio
async def test_p3_fresh_prompt_completed_turn_remains_foldable():
    """§4.1: a fresh prompt's opener is not in provider state, so the clamp
    protects nothing and the normally-completed turn stays foldable — no
    capacity regression."""
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    state["messages"].append(_user("Turn 1 request " + "x " * 2000))
    state["messages"].append(_assistant_text("Turn 1 answer " + "y " * 2000))
    CompressibleHistoryProvider.insert_marker(state, 1)

    fresh_opener = _user("Turn 2 request")  # in-flight input, unsaved
    wire = [*_wire_view(state), fresh_opener]
    total = _estimate_tokens(wire)

    compressed_events: list = []

    async def _on_compress(info) -> None:
        compressed_events.append(info)

    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.85,
        target_pct=0.50,
        on_compress=_on_compress,
    )
    strategy.bind_state(state)
    changed = await strategy(wire)
    assert changed

    assert [info.turn_range for info in compressed_events] == [(1, 1)]
    assert not fresh_opener.additional_properties.get(EXCLUDED_KEY, False)
    assert strategy._usage_pct(included_token_count(wire)) <= strategy.target_pct


@pytest.mark.asyncio
async def test_p3_stale_index_trap_re_resolves_after_fold():
    """§4.1 STALE-INDEX trap: each fold shrinks state["messages"], so the
    clamp boundary must be re-resolved per candidate pass — a boundary cached
    before the folds would let the interior marker's candidate through."""
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    state["messages"].append(_user("Turn 1 request " + "x " * 1500))
    state["messages"].append(_assistant_text("Turn 1 answer " + "y " * 1500))
    CompressibleHistoryProvider.insert_marker(state, 1)
    state["messages"].append(_user("Turn 2 request " + "x " * 1500))
    state["messages"].append(_assistant_text("Turn 2 answer " + "y " * 1500))
    CompressibleHistoryProvider.insert_marker(state, 2)
    opener3 = _user("Turn 3 request")
    state["messages"].append(opener3)
    work3 = _assistant_text("Turn 3 findings " + "z " * 1500)  # text-only: P4 stays out
    state["messages"].append(work3)
    state["messages"].append(_status_marker(HistoryMarkerKind.INTERRUPTED))
    CompressibleHistoryProvider.insert_marker(state, 3)

    wire = _wire_view(state)
    total = _estimate_tokens(wire)

    compressed_events: list = []

    async def _on_compress(info) -> None:
        compressed_events.append(info)

    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.85,
        target_pct=0.01,  # unreachable: P3 folds everything it may
        on_compress=_on_compress,
    )
    strategy.bind_state(state)
    changed = await strategy(wire)
    assert changed

    # Both completed turns folded — in order — and NOTHING covers turn 3.
    assert [info.turn_range for info in compressed_events] == [(1, 1), (2, 2)]
    assert not opener3.additional_properties.get(EXCLUDED_KEY, False)
    assert not work3.additional_properties.get(EXCLUDED_KEY, False)
    markers = CompressibleHistoryProvider.list_compressed(state)["markers"]
    assert [m["marker_id"] for m in markers] == ["turn_3"]


@pytest.mark.asyncio
async def test_phase4_degraded_all_flagged_drops_work_quotes_synthetic_opener():
    """§3.2 rule 4 integration: the marker-less all-flagged resume shape spans
    the whole wire — pre-nudge tool work IS collected by P4's drop and the
    generator's user_request comes from the synthetic opener (the nudge)."""
    work = _build_tool_group("resume_c0", "search", "x" * 3000)
    state: dict = {"messages": [*work], "compressed_msgs": [], "turn_counter": 0}
    nudge = _nudge()
    wire = [*work, nudge]
    total = _estimate_tokens(wire)

    generator = _StubLastWordsGenerator(text="[note]")
    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.85,
        target_pct=0.01,
        last_words_generator=generator,
        on_compaction=_async_appender(received),
    )
    strategy.bind_state(state)
    changed = await strategy(wire)
    assert changed

    for msg in work:
        assert msg.additional_properties.get(EXCLUDE_REASON_KEY) == _REASON_CURRENT_TURN_DROP
    assert not nudge.additional_properties.get(EXCLUDED_KEY, False)
    assert generator.calls and _scoped_user_texts(generator.calls[0])[0] == DEGRADED_SCOPED_PREAMBLE
    assert generator.calls[0]["degraded_opener"] is True
    assert generator.calls[0]["has_continuation_nudges"] is True
    p4 = [r for r in received if r.phase == "phase4"]
    assert p4 and p4[0].turn_numbers == [1]


@pytest.mark.asyncio
async def test_phase4_degraded_mixed_shape_drops_only_region_work():
    """MIXED shape (§3.2): degradation is per marker region — the completed
    turn stays a previous turn (P1/P2 truncate it), P4 drops only region work,
    and the injected guidance is quoted as the synthetic opener."""
    state: dict = {"messages": [], "compressed_msgs": [], "turn_counter": 0}
    state["messages"].append(_user("Turn 1 request"))
    t1_group = _build_tool_group("t1_c0", "search", "x" * 3000)
    state["messages"].extend(t1_group)
    state["messages"].append(_assistant_text("Turn 1 answer"))
    CompressibleHistoryProvider.insert_marker(state, 1)
    guidance = _injected("resume with flag X")
    state["messages"].append(guidance)
    t2_group = _build_tool_group("t2_c0", "grep", "y" * 3000)
    state["messages"].extend(t2_group)

    wire = _wire_view(state)
    total = _estimate_tokens(wire)

    generator = _StubLastWordsGenerator(text="[note]")
    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 100,
        trigger_pct=0.85,
        target_pct=0.01,
        last_words_generator=generator,
        on_compaction=_async_appender(received),
    )
    strategy.bind_state(state)
    changed = await strategy(wire)
    assert changed

    p12_numbers = {n for r in received if r.phase in ("phase1", "phase2") for n in r.turn_numbers}
    assert p12_numbers == {1}
    for msg in t1_group:
        assert msg.additional_properties.get(EXCLUDED_KEY, False)
        assert msg.additional_properties.get(EXCLUDE_REASON_KEY) != _REASON_CURRENT_TURN_DROP
    for msg in t2_group:
        assert msg.additional_properties.get(EXCLUDE_REASON_KEY) == _REASON_CURRENT_TURN_DROP
    assert not guidance.additional_properties.get(EXCLUDED_KEY, False)
    assert generator.calls and _scoped_user_texts(generator.calls[0]) == [
        DEGRADED_SCOPED_PREAMBLE,
        "resume with flag X",
    ]
    assert generator.calls[0]["degraded_opener"] is True
    p4 = [r for r in received if r.phase == "phase4"]
    assert p4 and p4[0].turn_numbers == [2]


@pytest.mark.asyncio
async def test_phase4_leading_summary_survives_structural_skip():
    """§4.3: a degraded span can start at wire index 0 and reach a prior
    compressed-context summary — the structural-group skip keeps it."""
    summary = _assistant_text("[Compressed context: ctx_prior]")
    summary.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.SUMMARY
    work = _build_tool_group("c0", "search", "x" * 3000)
    state: dict = {
        "messages": [summary, *work],
        "compressed_msgs": [CompressedBlock(compressed_context_id="ctx_prior", turn_range=(1, 2))],
        "turn_counter": 2,
    }
    nudge = _nudge()
    wire = [summary, *work, nudge]
    total = _estimate_tokens(wire)

    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.85,
        target_pct=0.01,
        on_compaction=_async_appender(received),
    )
    strategy.bind_state(state)
    changed = await strategy(wire)
    assert changed

    assert not summary.additional_properties.get(EXCLUDED_KEY, False)
    for msg in work:
        assert msg.additional_properties.get(EXCLUDE_REASON_KEY) == _REASON_CURRENT_TURN_DROP
    # Numbering rides the folded seed: max folded turn_range[1] + 1.
    p4 = [r for r in received if r.phase == "phase4"]
    assert p4 and p4[0].turn_numbers == [3]


@pytest.mark.asyncio
async def test_no_phase_fires_on_summary_plus_nudge_window():
    """A [summary, flagged-nudge] window holds no tool work: no phase fires."""
    summary = _assistant_text("[Compressed context: ctx_prior] " + "s " * 2000)
    summary.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.SUMMARY
    state: dict = {
        "messages": [summary],
        "compressed_msgs": [CompressedBlock(compressed_context_id="ctx_prior", turn_range=(1, 2))],
        "turn_counter": 2,
    }
    nudge = _nudge()
    wire = [summary, nudge]
    total = _estimate_tokens(wire)

    received: list[CompactionInfo] = []
    precompact: list[PreCompactInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.50,
        target_pct=0.10,
        on_compaction=_async_appender(received),
        on_pre_compact=_async_appender(precompact),
    )
    strategy.bind_state(state)
    changed = await strategy(wire)

    assert not changed
    assert not received
    assert not precompact
    assert not summary.additional_properties.get(EXCLUDED_KEY, False)
    assert not nudge.additional_properties.get(EXCLUDED_KEY, False)


@pytest.mark.asyncio
async def test_no_phase_fires_without_user_messages():
    """§3.2 rule 5: a history that never held user input resolves to no spans
    — early return, no phase fires."""
    wire = [_assistant_text("boot log " + "a " * 2000), _assistant_text("more " + "b " * 2000)]
    total = _estimate_tokens(wire)

    received: list[CompactionInfo] = []
    precompact: list[PreCompactInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.50,
        target_pct=0.10,
        on_compaction=_async_appender(received),
        on_pre_compact=_async_appender(precompact),
    )
    changed = await strategy(wire)

    assert not changed
    assert not received
    assert not precompact


@pytest.mark.asyncio
async def test_phase4_degraded_provider_prefix_untouched():
    """History-segment bound + system-kind skip (§3.2 rule 1, §4.3): in a
    degraded run with memory/context providers enabled, the provider prefix
    is never collected nor excluded, and its user-role prompt never opens a
    turn that would hand the resume work to P1/P2."""
    work = _build_tool_group("c0", "search", "x" * 3000)
    state: dict = {"messages": [*work], "compressed_msgs": [], "turn_counter": 0}
    system_msg = Message(role="system", contents=[Content.from_text("You are chrys. " + "s " * 500)])
    context_prompt = _user("Workspace context digest " + "c " * 500)
    nudge = _nudge()
    wire = [system_msg, context_prompt, *work, nudge]
    total = _estimate_tokens(wire)

    generator = _StubLastWordsGenerator(text="[note]")
    received: list[CompactionInfo] = []
    strategy = _make_strategy(
        max_context_tokens=total + 50,
        trigger_pct=0.85,
        target_pct=0.01,
        last_words_generator=generator,
        on_compaction=_async_appender(received),
    )
    strategy.bind_state(state)
    changed = await strategy(wire)
    assert changed

    assert not system_msg.additional_properties.get(EXCLUDED_KEY, False)
    assert not context_prompt.additional_properties.get(EXCLUDED_KEY, False)
    for msg in work:
        assert msg.additional_properties.get(EXCLUDE_REASON_KEY) == _REASON_CURRENT_TURN_DROP
    assert not [r for r in received if r.phase in ("phase1", "phase2")]
    assert generator.calls and _scoped_user_texts(generator.calls[0])[0] == DEGRADED_SCOPED_PREAMBLE


# ---------------------------------------------------------------------------
# Exchange unit boundaries (Phases 1/2 compact or drop whole annotated units)
# ---------------------------------------------------------------------------


def _tool_group_id_of(messages: list[Message], call_id: str) -> str:
    grouped = _group_messages_by_id(messages)
    kinds = _group_kind_map(messages)
    return next(
        gid
        for gid in _ordered_group_ids(messages)
        if kinds.get(gid) == "tool_call"
        and any(
            getattr(content, "call_id", None) == call_id for message in grouped[gid] for content in message.contents
        )
    )


def _sibling_call_run() -> list[Message]:
    return [
        Message(role="assistant", contents=[Content.from_function_call("call_a", "tool_a", arguments={})]),
        Message(role="assistant", contents=[Content.from_function_call("call_b", "tool_b", arguments={})]),
        Message(
            role="tool",
            contents=[
                Content.from_function_result("call_a", result="alpha outcome"),
                Content.from_function_result("call_b", result="beta outcome"),
            ],
        ),
    ]


class TestExchangeUnitBoundaries:
    """A run of call-carrying assistant siblings answered by one shared
    result block is ONE annotated unit, and user-role or marker messages are
    never members of the unit whose call they answer."""

    def test_phase1_summarizes_sibling_call_run_as_one_unit(self) -> None:
        messages = _sibling_call_run()
        annotate_message_groups(messages)

        kinds = _group_kind_map(messages)
        tool_gids = [gid for gid in _ordered_group_ids(messages) if kinds.get(gid) == "tool_call"]
        assert len(tool_gids) == 1
        grouped = _group_messages_by_id(messages)
        assert grouped[tool_gids[0]] == messages[:3]

        strategy = _make_strategy()
        assert strategy._compact_group(messages, tool_gids[0], grouped[tool_gids[0]])
        summary = next(m for m in messages if m.text.startswith("[Tool call:"))
        for fragment in ("tool_a", "alpha outcome", "tool_b", "beta outcome"):
            assert fragment in summary.text

    def test_phase2_removes_sibling_call_run_as_one_unit(self) -> None:
        messages = _sibling_call_run()
        annotate_message_groups(messages)
        target_gid = _tool_group_id_of(messages, "call_a")
        grouped = _group_messages_by_id(messages)

        strategy = _make_strategy()
        assert strategy._remove_group(messages, target_gid, grouped[target_gid])
        assert all(m.additional_properties.get(EXCLUDED_KEY, False) for m in messages)

    def test_phase1_leaves_user_role_result_out_of_the_summarized_unit(self) -> None:
        call_message = Message(
            role="assistant", contents=[Content.from_function_call("call_a", "tool_a", arguments={})]
        )
        user_result = Message(role="user", contents=[Content.from_function_result("call_a", result="alpha outcome")])
        messages = [call_message, user_result]
        annotate_message_groups(messages)
        target_gid = _tool_group_id_of(messages, "call_a")
        grouped = _group_messages_by_id(messages)

        strategy = _make_strategy()
        assert strategy._compact_group(messages, target_gid, grouped[target_gid]) is None
        assert not call_message.additional_properties.get(EXCLUDED_KEY, False)
        assert not user_result.additional_properties.get(EXCLUDED_KEY, False)

    def test_phase1_leaves_marker_carried_result_out_of_the_summarized_unit(self) -> None:
        call_message = Message(
            role="assistant", contents=[Content.from_function_call("call_a", "tool_a", arguments={})]
        )
        marker_result = Message(
            role="assistant", contents=[Content.from_function_result("call_a", result="alpha outcome")]
        )
        marker_result.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
        messages = [call_message, marker_result]
        annotate_message_groups(messages)
        target_gid = _tool_group_id_of(messages, "call_a")
        grouped = _group_messages_by_id(messages)

        strategy = _make_strategy()
        assert strategy._compact_group(messages, target_gid, grouped[target_gid]) is None
        assert not call_message.additional_properties.get(EXCLUDED_KEY, False)
        assert not marker_result.additional_properties.get(EXCLUDED_KEY, False)

    def test_phase1_keeps_image_narration_on_the_wire(self) -> None:
        """A fused narration member carrying content the text summary cannot
        represent (an image) survives compaction: the summary replaces only
        the members it can represent, and the kept member is neither excluded
        nor recorded as summarized."""
        from chrys.service.context.compaction import SUMMARIZED_BY_SUMMARY_ID_KEY

        call_message = Message(
            role="assistant", contents=[Content.from_function_call("call_a", "tool_a", arguments={})]
        )
        image_narration = Message(
            role="assistant",
            contents=[Content.from_uri("data:image/png;base64,AAAA", media_type="image/png")],
        )
        image_narration.message_id = "image-narration"
        result_message = Message(role="tool", contents=[Content.from_function_result("call_a", result="alpha outcome")])
        messages = [call_message, image_narration, result_message]
        annotate_message_groups(messages)
        target_gid = _tool_group_id_of(messages, "call_a")
        grouped = _group_messages_by_id(messages)
        assert image_narration in grouped[target_gid]

        strategy = _make_strategy()
        assert strategy._compact_group(messages, target_gid, grouped[target_gid])

        assert call_message.additional_properties.get(EXCLUDED_KEY, False)
        assert result_message.additional_properties.get(EXCLUDED_KEY, False)
        assert not image_narration.additional_properties.get(EXCLUDED_KEY, False)
        assert SUMMARIZED_BY_SUMMARY_ID_KEY not in image_narration.additional_properties
        summary = next(m for m in messages if m.text and m.text.startswith("[Tool call:"))
        original_ids = summary.additional_properties.get("_summary_of_message_ids") or []
        assert "image-narration" not in original_ids

    def test_phase1_keeps_mixed_text_and_image_narration_with_its_member(self) -> None:
        """Text riding an unsummarizable narration member stays on the wire
        with the member instead of being duplicated into the summary."""
        call_message = Message(
            role="assistant", contents=[Content.from_function_call("call_a", "tool_a", arguments={})]
        )
        narration = Message(
            role="assistant",
            contents=[
                Content.from_text("look at this chart"),
                Content.from_uri("data:image/png;base64,AAAA", media_type="image/png"),
            ],
        )
        result_message = Message(role="tool", contents=[Content.from_function_result("call_a", result="alpha outcome")])
        messages = [call_message, narration, result_message]
        annotate_message_groups(messages)
        target_gid = _tool_group_id_of(messages, "call_a")
        grouped = _group_messages_by_id(messages)

        strategy = _make_strategy()
        assert strategy._compact_group(messages, target_gid, grouped[target_gid])

        assert not narration.additional_properties.get(EXCLUDED_KEY, False)
        summary = next(m for m in messages if m.text and m.text.startswith("[Tool call:"))
        assert "look at this chart" not in summary.text

    def test_phase2_keeps_fused_narration_on_the_wire(self) -> None:
        """Phase 2 removes args and results; fused narration members (text or
        image alike) are neither and stay on the wire, as they did when the
        exchange grouped them apart from the calls."""
        call_message = Message(
            role="assistant", contents=[Content.from_function_call("call_a", "tool_a", arguments={})]
        )
        text_narration = Message(role="assistant", contents=[Content.from_text("checking the chart next")])
        image_narration = Message(
            role="assistant",
            contents=[Content.from_uri("data:image/png;base64,AAAA", media_type="image/png")],
        )
        result_message = Message(role="tool", contents=[Content.from_function_result("call_a", result="alpha outcome")])
        messages = [call_message, text_narration, image_narration, result_message]
        annotate_message_groups(messages)
        target_gid = _tool_group_id_of(messages, "call_a")
        grouped = _group_messages_by_id(messages)

        strategy = _make_strategy()
        assert strategy._remove_group(messages, target_gid, grouped[target_gid])

        assert call_message.additional_properties.get(EXCLUDED_KEY, False)
        assert result_message.additional_properties.get(EXCLUDED_KEY, False)
        assert not text_narration.additional_properties.get(EXCLUDED_KEY, False)
        assert not image_narration.additional_properties.get(EXCLUDED_KEY, False)

    def test_phase1_keeps_image_narration_in_a_reasoning_exchange(self) -> None:
        """Retention must survive projection in reasoning-bearing exchanges:
        the kept image is outside the atomic set (carriers + reasoning
        riders), so partial-group degrading never drags it back out."""
        call_message = Message(
            role="assistant",
            contents=[
                Content.from_text_reasoning(text="thinking"),
                Content.from_function_call("call_a", "tool_a", arguments={}),
            ],
        )
        image_narration = Message(
            role="assistant",
            contents=[Content.from_uri("data:image/png;base64,AAAA", media_type="image/png")],
        )
        result_message = Message(role="tool", contents=[Content.from_function_result("call_a", result="alpha outcome")])
        messages = [call_message, image_narration, result_message]
        annotate_message_groups(messages)
        target_gid = _tool_group_id_of(messages, "call_a")
        grouped = _group_messages_by_id(messages)

        strategy = _make_strategy()
        assert strategy._compact_group(messages, target_gid, grouped[target_gid])
        projected = project_included_messages(messages)

        assert image_narration in projected
        assert not image_narration.additional_properties.get(EXCLUDED_KEY, False)
        assert call_message not in projected

    def test_phase1_reasoning_riding_narration_stays_group_bound(self) -> None:
        """A narration member that itself carries reasoning content belongs to
        the atomic set and is excluded with its group — an orphaned reasoning
        item must never outlive its exchange."""
        call_message = Message(
            role="assistant", contents=[Content.from_function_call("call_a", "tool_a", arguments={})]
        )
        narration = Message(
            role="assistant",
            contents=[
                Content.from_text_reasoning(text="thinking about the chart"),
                Content.from_uri("data:image/png;base64,AAAA", media_type="image/png"),
            ],
        )
        result_message = Message(role="tool", contents=[Content.from_function_result("call_a", result="alpha outcome")])
        messages = [call_message, narration, result_message]
        annotate_message_groups(messages)
        target_gid = _tool_group_id_of(messages, "call_a")
        grouped = _group_messages_by_id(messages)

        strategy = _make_strategy()
        assert strategy._compact_group(messages, target_gid, grouped[target_gid])

        assert narration.additional_properties.get(EXCLUDED_KEY, False)

    def test_phase2_reasoning_rider_excluded_with_its_group(self) -> None:
        """Phase 2's narration skip never spares reasoning-bearing members."""
        call_message = Message(
            role="assistant", contents=[Content.from_function_call("call_a", "tool_a", arguments={})]
        )
        reasoning_rider = Message(role="assistant", contents=[Content.from_text_reasoning(text="thinking")])
        result_message = Message(role="tool", contents=[Content.from_function_result("call_a", result="alpha outcome")])
        messages = [call_message, reasoning_rider, result_message]
        annotate_message_groups(messages)
        target_gid = _tool_group_id_of(messages, "call_a")
        grouped = _group_messages_by_id(messages)

        strategy = _make_strategy()
        assert strategy._remove_group(messages, target_gid, grouped[target_gid])

        assert reasoning_rider.additional_properties.get(EXCLUDED_KEY, False)
