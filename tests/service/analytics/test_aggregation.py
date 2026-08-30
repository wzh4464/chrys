# Copyright (c) 2026 Chrys. All rights reserved.

"""Counter-examples for the frozen P0 timing and KPI formulas."""

from __future__ import annotations

import json
from array import array
from dataclasses import replace
from pathlib import Path

import pytest

from chrys.foundation.trajectory.envelope import SYSTEM_ACTOR, Actor, Link, LinkRelation, SegmentedField, measurement
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.metadata import ANALYTICS_ITEM_ID_KEY
from chrys.service.analytics import (
    FLOW_TERMINAL_INDEX,
    ActionClass,
    AnalysisAvailability,
    Metric,
    Precision,
    SessionSpan,
    TimelineDiagnosticCode,
    TimelineOperation,
    TimelineOperationDetail,
    TokenUsage,
    TrajectoryAnalyzer,
    TurnAttemptRef,
    TurnFlow,
    UsageBucket,
    WallBucket,
    analyze_trajectory,
)
from chrys.service.analytics import _critical_path as critical_path_module
from chrys.service.analytics import _facts as facts_module
from chrys.service.analytics import _session_projection as session_projection_module
from chrys.service.analytics import _turns as turns_module
from chrys.service.analytics import aggregation as aggregation_module
from chrys.service.analytics._critical_path import _longest_interval_path
from chrys.service.analytics._facts import _RevisionEntry
from chrys.service.analytics._turns import _replay_delta
from chrys.service.analytics.aggregation import _sum_metrics
from chrys.service.analytics.classification import evidence_key
from tests.service.analytics._events import EventLog

_NS = 1_000_000_000


@pytest.mark.parametrize(
    ("reason", "diagnostic_code"),
    [
        ("missing code", None),
        (None, TimelineDiagnosticCode.MISSING_START),
    ],
)
def test_timeline_operation_requires_reason_and_diagnostic_code_together(
    reason: str | None,
    diagnostic_code: TimelineDiagnosticCode | None,
) -> None:
    with pytest.raises(ValueError, match="reason and diagnostic code must be set together"):
        TimelineOperationDetail(
            reason=reason,
            diagnostic_code=diagnostic_code,
        )


def _caused_by(operation_id: str) -> tuple[Link, ...]:
    return (Link(relation=LinkRelation.CAUSED_BY, target_operation_id=operation_id),)


def _append_resolved_rollback(
    log: EventLog,
    superseded_from_sequence: int,
    monotonic_ns: int,
    *,
    target_turn_id: str | None = None,
) -> None:
    old_branch = "3" * 32
    new_branch = "6" * 32
    payload: dict[str, object] = {
        "old_branch_id": old_branch,
        "new_branch_id": new_branch,
        "superseded_from_sequence": superseded_from_sequence,
        "superseded_to_sequence": log.next_sequence - 1,
    }
    if target_turn_id is not None:
        payload["target_turn_id"] = target_turn_id
    log.add(EventType.SESSION_ROLLBACK, monotonic_ns, turn_id=None, branch_id=new_branch, payload=payload)
    log.add(
        EventType.BRANCH_SUPERSEDED,
        monotonic_ns,
        turn_id=None,
        branch_id=new_branch,
        payload={"branch_id": old_branch, "superseded_by": new_branch},
    )


def test_metric_sum_preserves_estimated_precision() -> None:
    total = _sum_metrics([Metric(2, Precision.EXACT), Metric(3, Precision.ESTIMATED)])

    assert (total.value, total.precision) == (5, Precision.ESTIMATED)


def test_all_corrupt_input_degrades_every_exported_session_metric_group(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    analysis = analyze_trajectory(path)

    assert analysis.turns == ()
    assert analysis.overview is not None
    assert analysis.overview.elapsed_ns == Metric(
        0,
        Precision.UNRESOLVED,
        "session trajectory integrity is unresolved: corrupt lines",
    )
    assert analysis.token_usage is not None
    assert analysis.token_usage.buckets[UsageBucket.INPUT].precision is Precision.UNRESOLVED
    assert analysis.token_usage.buckets[UsageBucket.OUTPUT].precision is Precision.UNRESOLVED
    assert all(
        analysis.token_usage.buckets[bucket].precision is Precision.MISSING
        for bucket in (UsageBucket.REASONING, UsageBucket.CACHE_READ, UsageBucket.CACHE_CREATION)
    )
    assert analysis.validation is not None
    assert analysis.validation.tool_count.precision is Precision.UNRESOLVED
    assert analysis.validation.time_to_first_edit_ns.precision is Precision.MISSING
    assert analysis.change_verification is not None
    assert analysis.change_verification.files_touched.precision is Precision.UNRESOLVED
    assert analysis.change_verification.files_touched.reason == (
        "summed per-turn summary counts; usable session.json file detail is unavailable to fold repeat touches; "
        "session trajectory integrity is unresolved: corrupt lines"
    )


def test_corruption_after_a_turn_degrades_only_session_scope_metrics(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)
    path.write_bytes(path.read_bytes() + b"{corrupt after the turn}\n")

    analysis = analyze_trajectory(path)

    assert analysis.turns[0].elapsed_ns.precision is Precision.EXACT
    assert analysis.overview is not None
    assert analysis.overview.elapsed_ns.precision is Precision.UNRESOLVED
    assert analysis.overview.elapsed_ns.reason == "session trajectory integrity is unresolved: corrupt lines"


def test_retry_attempts_fold_into_one_logical_turn(tmp_path: Path) -> None:
    first_turn_id = "4" * 32
    retry_turn_id = "5" * 32
    second_retry_turn_id = "6" * 32
    log = EventLog()
    log.coverage()
    log.add(EventType.TURN_STARTED, 0, turn_id=first_turn_id, payload={"turn_number": 1, "is_retry": False})
    log.add(
        EventType.TURN_FINISHED,
        2 * _NS,
        turn_id=first_turn_id,
        payload={"end_reason": "interrupted", "duration_ms": 2_000},
    )
    log.settled(3 * _NS, turn_id=first_turn_id, drained_scopes=[])
    log.add(
        EventType.TURN_STARTED,
        100 * _NS,
        turn_id=retry_turn_id,
        payload={"turn_number": 1, "is_retry": True},
    )
    log.add(
        EventType.TURN_FINISHED,
        102 * _NS,
        turn_id=retry_turn_id,
        payload={"end_reason": "interrupted", "duration_ms": 2_000},
    )
    log.settled(103 * _NS, turn_id=retry_turn_id, drained_scopes=[])
    log.add(
        EventType.TURN_STARTED,
        200 * _NS,
        turn_id=second_retry_turn_id,
        payload={"turn_number": 1, "is_retry": True},
    )
    log.add(
        EventType.TURN_FINISHED,
        204 * _NS,
        turn_id=second_retry_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 4_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len(analysis.turns) == 1
    turn = analysis.turns[0]
    assert turn.turn_id == first_turn_id
    assert turn.turn_number == 1
    assert tuple(attempt.turn_id for attempt in turn.attempts) == (
        first_turn_id,
        retry_turn_id,
        second_retry_turn_id,
    )
    assert analysis.turn(retry_turn_id) is turn
    assert analysis.turn(second_retry_turn_id) is turn
    assert turn.elapsed_ns == Metric(10 * _NS, Precision.EXACT)
    assert turn.axis_end_ns - turn.axis_start_ns == 10 * _NS
    assert [attempt.is_retry for attempt in turn.attempts] == [False, True, True]
    assert analysis.change_verification is not None
    assert "turns' numbers cannot join" not in (analysis.change_verification.files_touched.reason or "")


def test_retry_timeline_diagnostic_keeps_physical_attempt_turn_id(tmp_path: Path) -> None:
    first_turn_id = "4" * 32
    retry_turn_id = "5" * 32
    hook_operation_id = "a" * 32
    log = EventLog()
    log.coverage()
    log.add(EventType.TURN_STARTED, 0, turn_id=first_turn_id, payload={"turn_number": 1, "is_retry": False})
    log.add(
        EventType.TURN_FINISHED,
        2 * _NS,
        turn_id=first_turn_id,
        payload={"end_reason": "interrupted", "duration_ms": 2_000},
    )
    log.settled(3 * _NS, turn_id=first_turn_id, drained_scopes=[])
    log.add(
        EventType.TURN_STARTED,
        100 * _NS,
        turn_id=retry_turn_id,
        payload={"turn_number": 1, "is_retry": True},
    )
    log.add(
        EventType.HOOK_OPERATION_STARTED,
        101 * _NS,
        turn_id=retry_turn_id,
        operation_id=hook_operation_id,
        payload={
            "hook_key": "retry-cleanup",
            "hook_event": "after_turn",
            "execution_mode": "async",
            "drain_scope": "turn",
        },
    )
    log.add(
        EventType.TURN_FINISHED,
        102 * _NS,
        turn_id=retry_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 2_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len(analysis.turns) == 1
    assert analysis.turns[0].turn_id == first_turn_id
    diagnostic = next(
        item for item in analysis.diagnostics.timeline_operations if item.operation_id == hook_operation_id
    )
    assert diagnostic.turn_id == retry_turn_id
    assert diagnostic.turn_number == 1


def test_same_number_fresh_turns_are_never_folded(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    for index, turn_id in enumerate(("4" * 32, "5" * 32)):
        log.add(
            EventType.TURN_STARTED,
            index * 2 * _NS,
            turn_id=turn_id,
            payload={"turn_number": 1, "is_retry": False},
        )
        log.add(
            EventType.TURN_FINISHED,
            (index * 2 + 1) * _NS,
            turn_id=turn_id,
            payload={"end_reason": "cancelled", "duration_ms": 1_000},
        )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len(analysis.turns) == 2
    assert [turn.turn_number for turn in analysis.turns] == [1, 1]


def test_orphan_retry_is_normalized_as_one_logical_turn(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    log.add(EventType.TURN_STARTED, 0, payload={"turn_number": 1, "is_retry": True})
    log.add(
        EventType.TURN_FINISHED,
        _NS,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert len(turn.attempts) == 1
    assert turn.attempts[0].turn_id == turn.turn_id
    assert turn.attempts[0].is_retry is True


def test_normalized_single_retry_keeps_physical_identity_for_refolding(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    log.add(EventType.TURN_STARTED, 10 * _NS, payload={"turn_number": 1, "is_retry": True})
    log.add(
        EventType.TURN_FINISHED,
        11 * _NS,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)
    cached_retry = analyze_trajectory(path).turns[0]
    first_turn_id = "5" * 32
    predecessor = replace(
        cached_retry,
        turn_id=first_turn_id,
        start_sequence=0,
        axis_start_ns=0,
        axis_end_ns=_NS,
        attempts=(
            replace(
                cached_retry.attempts[0],
                turn_id=first_turn_id,
                is_retry=False,
                physical_axis_start_ns=0,
                physical_axis_end_ns=_NS,
                logical_axis_start_ns=0,
            ),
        ),
    )

    refolded = aggregation_module._fold_retry_turns([predecessor, cached_retry])

    assert len(refolded) == 1
    assert [attempt.is_retry for attempt in refolded[0].attempts] == [False, True]


def test_noncontiguous_retry_number_is_not_folded_or_joinable(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    for index, (turn_id, turn_number, is_retry) in enumerate(
        (("4" * 32, 1, False), ("5" * 32, 2, False), ("6" * 32, 1, True))
    ):
        log.add(
            EventType.TURN_STARTED,
            index * 2 * _NS,
            turn_id=turn_id,
            payload={"turn_number": turn_number, "is_retry": is_retry},
        )
        log.add(
            EventType.TURN_FINISHED,
            (index * 2 + 1) * _NS,
            turn_id=turn_id,
            payload={"end_reason": "cancelled", "duration_ms": 1_000},
        )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len(analysis.turns) == 3
    assert [turn.turn_number for turn in analysis.turns] == [1, 2, 1]
    assert analysis.change_verification is not None
    assert "turns' numbers cannot join" in (analysis.change_verification.files_touched.reason or "")


def test_duplicate_consistent_starts_keep_identity_and_fold_with_retry(tmp_path: Path) -> None:
    first_turn_id = "4" * 32
    retry_turn_id = "5" * 32
    log = EventLog()
    log.coverage()
    for monotonic_ns in (0, _NS):
        log.add(
            EventType.TURN_STARTED,
            monotonic_ns,
            turn_id=first_turn_id,
            payload={"turn_number": 1, "is_retry": False},
        )
    log.add(
        EventType.TURN_FINISHED,
        2 * _NS,
        turn_id=first_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 2_000},
    )
    log.add(
        EventType.TURN_STARTED,
        10 * _NS,
        turn_id=retry_turn_id,
        payload={"turn_number": 1, "is_retry": True},
    )
    log.add(
        EventType.TURN_FINISHED,
        11 * _NS,
        turn_id=retry_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len(analysis.turns) == 1
    turn = analysis.turns[0]
    assert turn.turn_number == 1
    assert [attempt.turn_id for attempt in turn.attempts] == [first_turn_id, retry_turn_id]
    assert "turn lifecycle is not uniquely opened" in turn.diagnostics


def test_duplicate_retry_starts_restore_retry_identity_and_fold_with_predecessor(tmp_path: Path) -> None:
    first_turn_id = "4" * 32
    retry_turn_id = "5" * 32
    log = EventLog()
    log.coverage()
    log.add(
        EventType.TURN_STARTED,
        0,
        turn_id=first_turn_id,
        payload={"turn_number": 1, "is_retry": False},
    )
    log.add(
        EventType.TURN_FINISHED,
        _NS,
        turn_id=first_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    for monotonic_ns in (10 * _NS, 11 * _NS):
        log.add(
            EventType.TURN_STARTED,
            monotonic_ns,
            turn_id=retry_turn_id,
            payload={"turn_number": 1, "is_retry": True},
        )
    log.add(
        EventType.TURN_FINISHED,
        12 * _NS,
        turn_id=retry_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len(analysis.turns) == 1
    turn = analysis.turns[0]
    assert [attempt.turn_id for attempt in turn.attempts] == [first_turn_id, retry_turn_id]
    assert [attempt.is_retry for attempt in turn.attempts] == [False, True]
    assert "turn lifecycle is not uniquely opened" in turn.diagnostics


def test_fold_sums_attempt_tokens_actions_and_critical_contributions(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    log.turn(0, 5 * _NS)
    path = tmp_path / "base.jsonl"
    log.write(path)
    base = analyze_trajectory(path).turns[0]
    optional_missing = Metric(None, Precision.MISSING, "not reported")

    def physical_attempt(
        turn_id: str,
        *,
        axis_start_ns: int,
        is_retry: bool,
        input_tokens: int,
        action_count: int,
        contribution_ns: int,
        elapsed_precision: Precision,
    ):
        axis_end_ns = axis_start_ns + 5 * _NS
        return replace(
            base,
            turn_id=turn_id,
            start_sequence=1 if not is_retry else 10,
            elapsed_ns=Metric(5 * _NS, elapsed_precision),
            axis_start_ns=axis_start_ns,
            axis_end_ns=axis_end_ns,
            attempts=(
                TurnAttemptRef(
                    turn_id=turn_id,
                    runtime_id=base.runtime_id,
                    is_retry=is_retry,
                    physical_axis_start_ns=axis_start_ns,
                    physical_axis_end_ns=axis_end_ns,
                    logical_axis_start_ns=axis_start_ns,
                    operation_start_index=0,
                    operation_end_index=len(base.operations),
                    slice_start_index=0,
                    slice_end_index=len(base.slices),
                ),
            ),
            action_counts={ActionClass.SEARCH: Metric(action_count, Precision.EXACT)},
            critical_tool_contributions_ns={"shared-operation": contribution_ns},
            server_critical_contributions_ns={"shared-server": contribution_ns + 1},
            token_usage=TokenUsage(
                buckets={
                    UsageBucket.INPUT: Metric(input_tokens, Precision.EXACT),
                    UsageBucket.OUTPUT: Metric(1, Precision.EXACT),
                    UsageBucket.REASONING: optional_missing,
                    UsageBucket.CACHE_READ: optional_missing,
                    UsageBucket.CACHE_CREATION: optional_missing,
                }
            ),
        )

    folded = aggregation_module._fold_retry_turns(
        [
            physical_attempt(
                "4" * 32,
                axis_start_ns=0,
                is_retry=False,
                input_tokens=10,
                action_count=2,
                contribution_ns=3,
                elapsed_precision=Precision.EXACT,
            ),
            physical_attempt(
                "5" * 32,
                axis_start_ns=100 * _NS,
                is_retry=True,
                input_tokens=20,
                action_count=4,
                contribution_ns=7,
                elapsed_precision=Precision.ESTIMATED,
            ),
        ]
    )[0]

    assert folded.elapsed_ns == Metric(
        10 * _NS,
        Precision.ESTIMATED,
        "one or more attempts of this turn are not exact",
    )
    assert folded.action_counts[ActionClass.SEARCH] == Metric(6, Precision.EXACT)
    assert folded.critical_tool_contributions_ns == {"shared-operation": 10}
    assert folded.server_critical_contributions_ns == {"shared-server": 12}
    assert folded.token_usage is not None
    assert folded.token_usage.buckets[UsageBucket.INPUT] == Metric(30, Precision.EXACT)
    assert folded.token_usage.buckets[UsageBucket.OUTPUT] == Metric(2, Precision.EXACT)


def test_fold_diagnoses_a_clamped_attempt_axis(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    log.turn(0, 5 * _NS)
    path = tmp_path / "base.jsonl"
    log.write(path)
    base = analyze_trajectory(path).turns[0]
    malformed_ref = replace(
        base.attempts[0],
        physical_axis_start_ns=10 * _NS,
        physical_axis_end_ns=5 * _NS,
        logical_axis_start_ns=10 * _NS,
    )
    first = replace(
        base,
        axis_start_ns=10 * _NS,
        axis_end_ns=5 * _NS,
        attempts=(malformed_ref,),
    )
    retry_ref = replace(
        base.attempts[0],
        turn_id="5" * 32,
        is_retry=True,
        physical_axis_start_ns=20 * _NS,
        physical_axis_end_ns=25 * _NS,
        logical_axis_start_ns=20 * _NS,
    )
    retry = replace(
        base,
        turn_id=retry_ref.turn_id,
        start_sequence=10,
        axis_start_ns=20 * _NS,
        axis_end_ns=25 * _NS,
        attempts=(retry_ref,),
    )

    folded = aggregation_module._fold_retry_turns([first, retry])[0]

    assert "folded attempt axis end precedes its start" in folded.diagnostics
    assert [attempt.logical_axis_start_ns for attempt in folded.attempts] == [10 * _NS, 10 * _NS]


def test_fold_drops_nonfinal_attempt_response_terminal_edges(tmp_path: Path) -> None:
    log = EventLog()
    log.coverage()
    log.turn(0, 5 * _NS)
    path = tmp_path / "base.jsonl"
    log.write(path)
    base = analyze_trajectory(path).turns[0]

    def physical_attempt(turn_id: str, start_ns: int, *, is_retry: bool):
        end_ns = start_ns + 5 * _NS
        operation = TimelineOperation(
            operation_id=turn_id,
            family="model.exchange",
            depth=0,
            start_ns=start_ns,
            end_ns=end_ns,
            precision=Precision.EXACT,
        )
        flow = TurnFlow(
            turn_id=turn_id,
            root_index=0,
            has_terminal=True,
            parent_pairs=b"",
            causal_pairs=array("I", (0, FLOW_TERMINAL_INDEX)).tobytes(),
            acyclic=True,
        )
        return replace(
            base,
            turn_id=turn_id,
            start_sequence=1 if not is_retry else 10,
            axis_start_ns=start_ns,
            axis_end_ns=end_ns,
            operations=(operation,),
            flow=flow,
            attempts=(
                replace(
                    base.attempts[0],
                    turn_id=turn_id,
                    is_retry=is_retry,
                    physical_axis_start_ns=start_ns,
                    physical_axis_end_ns=end_ns,
                    logical_axis_start_ns=start_ns,
                    operation_end_index=1,
                ),
            ),
        )

    folded = aggregation_module._fold_retry_turns(
        [
            physical_attempt("4" * 32, 0, is_retry=False),
            physical_attempt("5" * 32, 100 * _NS, is_retry=True),
        ]
    )[0]

    assert folded.flow is not None
    assert folded.flow.causal_edges() == ((1, FLOW_TERMINAL_INDEX),)


def test_empty_log_degrades_exact_zero_session_metrics(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b"")

    analysis = analyze_trajectory(path)

    assert analysis.availability is AnalysisAvailability.AVAILABLE
    assert analysis.diagnostics.line_count == 0
    assert analysis.diagnostics.integrity_unresolved is True
    assert analysis.overview is not None
    assert analysis.overview.elapsed_ns == Metric(
        0,
        Precision.UNRESOLVED,
        "session trajectory integrity is unresolved: empty log",
    )


def test_torn_only_log_is_not_mislabeled_as_empty(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    tail = b'{"incomplete"'
    path.write_bytes(tail)

    analysis = analyze_trajectory(path)

    assert analysis.diagnostics.line_count == 0
    assert analysis.diagnostics.byte_count == 0
    assert analysis.diagnostics.torn_tail_bytes == len(tail)
    assert analysis.overview is not None
    assert analysis.overview.elapsed_ns.reason == "session trajectory integrity is unresolved: torn tail"


def test_integrity_damage_caps_usage_panels_and_nested_insights(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span(
        "tool.operation",
        "a" * 32,
        _NS,
        2 * _NS,
        start_payload={
            "tool_name": "load_skill",
            "tool_kind": "skill",
            "tool_context": {"skill_name": "slides", "skill_revision": "rev-a"},
        },
        finish_payload={"outcome": "success"},
    )
    log.span(
        "tool.operation",
        "b" * 32,
        3 * _NS,
        4 * _NS,
        start_payload={
            "tool_name": "run_skill_script",
            "tool_kind": "skill",
            "tool_context": {
                "skill_name": "slides",
                "skill_revision": "rev-a",
                "script_name": "scripts/render.py",
            },
        },
        finish_payload={"outcome": "success"},
    )
    log.span(
        "tool.operation",
        "c" * 32,
        5 * _NS,
        6 * _NS,
        start_payload={
            "tool_name": "figma_render",
            "tool_kind": "mcp",
            "tool_context": {"server_name": "figma", "remote_name": "render"},
        },
        finish_payload={"outcome": "success"},
    )
    log.add("turn.finished", 7 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)
    path.write_bytes(path.read_bytes() + b"{corrupt after the turn}\n")

    analysis = analyze_trajectory(path)

    reason = "session trajectory integrity is unresolved: corrupt lines"
    assert analysis.skill_usage is not None
    assert (analysis.skill_usage.total, analysis.skill_usage.precision, analysis.skill_usage.reason) == (
        2,
        Precision.UNRESOLVED,
        reason,
    )
    assert analysis.mcp_usage is not None
    assert (analysis.mcp_usage.total, analysis.mcp_usage.precision, analysis.mcp_usage.reason) == (
        1,
        Precision.UNRESOLVED,
        reason,
    )
    assert analysis.insights is not None
    assert analysis.insights.tools.precision is Precision.UNRESOLVED
    assert analysis.insights.mcp.precision is Precision.UNRESOLVED
    assert analysis.insights.skills.precision is Precision.UNRESOLVED
    assert analysis.insights.context_carrying_precision is Precision.UNRESOLVED
    assert {
        analysis.insights.tools.reason,
        analysis.insights.mcp.reason,
        analysis.insights.skills.reason,
        analysis.insights.context_carrying_reason,
    } == {reason}
    assert analysis.insights.tools.rows[0].duration_share.precision is Precision.UNRESOLVED
    assert analysis.insights.mcp.rows[0].duration_share.precision is Precision.UNRESOLVED
    assert analysis.insights.skills.rows[0].first_action_median_ns.precision is Precision.UNRESOLVED


@pytest.mark.parametrize("damage", ["gap", "unsupported", "torn"])
def test_non_turn_integrity_damage_degrades_session_scope_metrics(tmp_path, damage: str) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    if damage == "gap":
        log.add(EventType.GAP, _NS, turn_id=None, payload={"first_sequence": 100, "last_sequence": 100})
    elif damage == "unsupported":
        log.add("future.event", _NS, turn_id=None)
    path = tmp_path / "events.jsonl"
    log.write(path)
    if damage == "torn":
        path.write_bytes(path.read_bytes() + b'{"incomplete"')

    analysis = analyze_trajectory(path)

    assert analysis.turns[0].elapsed_ns.precision is Precision.EXACT
    assert analysis.overview is not None
    assert analysis.overview.elapsed_ns.precision is Precision.UNRESOLVED


def test_two_parallel_tasks_have_cp_equal_elapsed_and_parallelism_two(tmp_path) -> None:
    """CP/elapsed=1 does not imply an absence of parallel work."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "1" * 32,
        0,
        0,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "2" * 32},
    )
    log.span("model.exchange", "2" * 32, 0, 0, parent_operation_id="1" * 32)
    item_ids = (("7" * 32, "9" * 32), ("8" * 32, "0" * 32))
    for prefix, (call_item_id, result_item_id) in zip(("c", "d"), item_ids, strict=True):
        tool_id = prefix * 32
        preamble_id = ("e" if prefix == "c" else "f") * 32
        log.span(
            "preparation",
            preamble_id,
            0,
            0,
            parent_operation_id="2" * 32,
            start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": tool_id},
        )
        log.span(
            "tool.operation",
            tool_id,
            0,
            10 * _NS,
            parent_operation_id="2" * 32,
            start_payload={
                "tool_name": prefix,
                "tool_kind": "filesystem.read",
                "batch_index": 0,
                "parent_model_operation_id": "2" * 32,
                "call_item_id": call_item_id,
            },
            finish_payload={"result_item_id": result_item_id},
            links=_caused_by(preamble_id),
        )
    revision = log.add(
        "context.revision.recorded",
        10 * _NS,
        operation_id="5" * 32,
        parent_operation_id="4" * 32,
        payload={"revision_id": "5" * 32, "is_checkpoint": True, "item_count": 4, "unidentified_item_count": 0},
        segmented_fields=(SegmentedField(field_pointer="/payload/refs", segment_group_id="6" * 32, segment_count=1),),
    )
    log.add(
        "event.segment",
        10 * _NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": "6" * 32,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": item_id, "occurrence": 0, "position": index, "action": "add"}
                for index, item_id in enumerate(item_id for pair in item_ids for item_id in pair)
            ],
        },
    )
    log.span(
        "model.cycle",
        "3" * 32,
        10 * _NS,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "4" * 32},
    )
    log.span(
        "model.exchange",
        "4" * 32,
        10 * _NS,
        10 * _NS,
        parent_operation_id="3" * 32,
        start_payload={"context_revision_id": "5" * 32},
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.value == pytest.approx(10 * _NS)
    assert turn.response_cp_ns.value == pytest.approx(10 * _NS)
    assert turn.parallelism.value == pytest.approx(2.0)
    assert turn.overlap_gain_ns.value == 10 * _NS


def test_wall_partition_is_additive_but_utilization_is_not(tmp_path) -> None:
    """Concurrent model and hook work produces 100% wall time but 200% utilization."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * _NS, links=_caused_by("a" * 32))
    log.span(
        "hook.operation",
        "c" * 32,
        0,
        10 * _NS,
        start_payload={"hook_event": "session_start", "execution_mode": "async", "scope": "session"},
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert sum(metric.value for metric in turn.wall_time_ns.values()) == 10 * _NS
    assert turn.wall_time_ns[WallBucket.MODEL].value == 10 * _NS
    assert turn.wall_time_ns[WallBucket.TOOLS].value == 0
    assert turn.utilization[WallBucket.MODEL].value == pytest.approx(1.0)
    assert turn.utilization[WallBucket.TOOLS].value == pytest.approx(1.0)


def test_approval_only_turn_has_wait_wall_time_but_zero_work(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(
        "approval.requested",
        0,
        operation_id="a" * 32,
        payload={"approval_request_id": "a" * 32},
    )
    log.add(
        "approval.resolved",
        10 * _NS,
        operation_id="a" * 32,
        payload={"approval_request_id": "a" * 32, "outcome": "approved", "wait_ms": 10_000},
        measurements={"/payload/wait_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.exclusive_work_ns.value == 0
    assert turn.parallelism.value == 0
    assert turn.wall_time_ns[WallBucket.WAIT].value == 10 * _NS


def test_finalizer_tail_stays_idle_in_wall_partition(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * _NS, links=_caused_by("a" * 32))
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(20 * _NS)
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.elapsed_ns.value == 20 * _NS
    assert turn.wall_time_ns[WallBucket.MODEL].value == 10 * _NS
    assert turn.wall_time_ns[WallBucket.IDLE].value == 10 * _NS
    assert turn.wall_time_ns[WallBucket.TOOLS].value == 0
    assert turn.wall_time_ns[WallBucket.WAIT].value == 0


@pytest.mark.parametrize("missing", ["turn", "tool"])
def test_missing_required_caused_by_edge_makes_both_cp_families_unresolved(tmp_path, missing: str) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, _NS, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, _NS, 10 * _NS, links=() if missing == "turn" else _caused_by("a" * 32))
    log.span(
        "preparation",
        "c" * 32,
        2 * _NS,
        3 * _NS,
        parent_operation_id="b" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "d" * 32},
    )
    log.span(
        "tool.operation",
        "d" * 32,
        3 * _NS,
        4 * _NS,
        parent_operation_id="b" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "parent_model_operation_id": "b" * 32,
        },
        links=() if missing == "tool" else _caused_by("c" * 32),
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert turn.exclusive_work_ns.precision is Precision.EXACT


@pytest.mark.parametrize(
    ("outcome", "has_result", "expected"),
    [
        # Invalid-argument and unknown-tool closes still own a result item the
        # next exchange consumes; a filtered close never gets one.
        ("invalid_arguments", True, Precision.EXACT),
        ("unknown_tool", True, Precision.EXACT),
        ("filtered", False, Precision.EXACT),
        ("errored", True, Precision.UNRESOLVED),
    ],
)
def test_never_dispatched_tool_needs_no_preamble_pairing(
    tmp_path, outcome: str, has_result: bool, expected: Precision
) -> None:
    """The kernel closes undispatched calls without a ``tool_preamble``; only those outcomes are exempt."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 6 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        6 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "f" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, _NS, parent_operation_id="c" * 32)
    log.span(
        "preparation",
        "1" * 32,
        _NS,
        2 * _NS,
        parent_operation_id="d" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "2" * 32},
    )
    log.span(
        "tool.operation",
        "2" * 32,
        2 * _NS,
        4 * _NS,
        parent_operation_id="d" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "parent_model_operation_id": "d" * 32,
            "call_item_id": "5" * 32,
        },
        finish_payload={"result_item_id": "6" * 32},
        links=_caused_by("1" * 32),
    )
    log.span(
        "tool.operation",
        "3" * 32,
        _NS,
        _NS,
        parent_operation_id="d" * 32,
        start_payload={
            "tool_name": "zsh",
            "tool_kind": "shell",
            "parent_model_operation_id": "d" * 32,
            "call_item_id": "7" * 32,
        },
        finish_payload={"outcome": outcome, "error_kind": outcome}
        | ({"result_item_id": "8" * 32} if has_result else {}),
    )
    member_ids = ("5" * 32, "6" * 32, "7" * 32, *(("8" * 32,) if has_result else ()))
    revision = log.add(
        "context.revision.recorded",
        4 * _NS,
        operation_id="9" * 32,
        parent_operation_id="f" * 32,
        payload={
            "revision_id": "9" * 32,
            "is_checkpoint": True,
            "item_count": len(member_ids),
            "unidentified_item_count": 0,
        },
        segmented_fields=(SegmentedField(field_pointer="/payload/refs", segment_group_id="0" * 32, segment_count=1),),
    )
    log.add(
        "event.segment",
        4 * _NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": "0" * 32,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": item_id, "occurrence": 0, "position": index, "action": "add"}
                for index, item_id in enumerate(member_ids)
            ],
        },
    )
    log.span(
        "model.exchange",
        "f" * 32,
        4 * _NS,
        6 * _NS,
        parent_operation_id="c" * 32,
        start_payload={"context_revision_id": "9" * 32},
    )
    log.add("turn.finished", 6 * _NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(6 * _NS)
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is expected
    assert turn.response_cp_ns.precision is expected
    pairing_diagnostics = [diagnostic for diagnostic in turn.diagnostics if "tool_preamble" in diagnostic]
    if expected is Precision.EXACT:
        assert pairing_diagnostics == []
        assert not any("tool result fan-in" in diagnostic for diagnostic in turn.diagnostics)
        assert turn.compute_cp_ns.value == 6 * _NS
    else:
        assert "tool_preamble to tool pairing is not one-to-one" in pairing_diagnostics
        assert "tool operation lacks caused_by link to its tool_preamble" in pairing_diagnostics


def test_session_span_keeps_first_turn_start_last_turn_finish_and_runtime_count(tmp_path) -> None:
    """Wall-clock anchors stay the producer's strings; runtimes count once per start."""
    later_runtime = "7" * 32
    log = EventLog()
    log.coverage()
    log.add(EventType.RUNTIME_STARTED, 0, turn_id=None)
    log.add("turn.started", 0, payload={"turn_number": 1}, occurred_at="2026-08-01T10:00:00.000000Z")
    log.add(
        "turn.finished",
        _NS,
        payload={"end_reason": "cancelled", "duration_ms": 0},
        occurred_at="2026-08-01T10:00:01.000000Z",
    )
    log.add(EventType.RUNTIME_STARTED, 2 * _NS, turn_id=None, runtime_id=later_runtime)
    log.add(
        "turn.started",
        2 * _NS,
        turn_id="9" * 32,
        payload={"turn_number": 2},
        runtime_id=later_runtime,
        occurred_at="2026-08-02T10:00:00.000000Z",
    )
    log.add(
        "turn.finished",
        3 * _NS,
        turn_id="9" * 32,
        payload={"end_reason": "cancelled", "duration_ms": 0},
        runtime_id=later_runtime,
        occurred_at="2026-08-02T10:00:05.000000Z",
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    first = analyzer.load(path).session_span

    assert first == SessionSpan(
        first_turn_started_at="2026-08-01T10:00:00.000000Z",
        last_turn_finished_at="2026-08-02T10:00:05.000000Z",
        runtime_count=2,
    )

    log.add("turn.started", 4 * _NS, turn_id="8" * 32, payload={"turn_number": 3}, occurred_at="2026-08-03T00:00:00Z")
    log.add(
        "turn.finished",
        5 * _NS,
        turn_id="8" * 32,
        payload={"end_reason": "cancelled", "duration_ms": 0},
        occurred_at="2026-08-03T00:00:09Z",
    )
    log.write(path)

    refreshed = analyzer.refresh().session_span

    assert refreshed == SessionSpan(
        first_turn_started_at="2026-08-01T10:00:00.000000Z",
        last_turn_finished_at="2026-08-03T00:00:09Z",
        runtime_count=2,
    )


def test_refresh_surfaces_in_flight_work_added_to_an_already_cached_turn(tmp_path) -> None:
    """Adding a projected node re-resolves the turn even without a terminal event.

    One appended event per refresh: a single dirty flag re-resolves the whole
    turn, so batching the events would let any one working path mask the rest.
    """
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    assert analyzer.load(path).turns[0].operations == ()

    def visible() -> set[tuple[str, str]]:
        return {(operation.family, operation.operation_id) for operation in analyzer.refresh().turns[0].operations}

    log.add("model.exchange.started", 1 * _NS, operation_id="a" * 32)
    log.write(path)
    assert ("model.exchange", "a" * 32) in visible()

    log.add("approval.requested", 2 * _NS, operation_id="b" * 32, payload={"approval_request_id": "b" * 32})
    log.write(path)
    assert ("approval", "b" * 32) in visible()

    log.add(
        "retry.scheduled",
        3 * _NS,
        operation_id=None,
        payload={
            "retry_mode": "run",
            "previous_operation_id": "a" * 32,
            "next_operation_id": "c" * 32,
            "delay_ms": 1000,
        },
    )
    log.write(path)
    assert ("retry", "c" * 32) in visible()

    log.add(
        "compaction.phase.finished",
        4 * _NS,
        operation_id="d" * 32,
        payload={
            "compaction_run_id": "e" * 32,
            "phase": "summaries",
            "groups_compacted": 1,
            "duration_ms": 5,
            "tokens_before": 100,
            "tokens_after": 50,
            "last_words_generated": False,
        },
    )
    log.write(path)
    refreshed = visible()
    assert any(family == "compaction.phase" for family, _ in refreshed)

    fresh = {
        (operation.family, operation.operation_id) for operation in TrajectoryAnalyzer().load(path).turns[0].operations
    }
    assert refreshed == fresh


def test_refresh_revalidates_counter_axis_only_for_dirty_folded_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_turn_id = "4" * 32
    retry_turn_id = "5" * 32
    log = EventLog()
    log.coverage()
    log.add(
        EventType.TURN_STARTED,
        0,
        turn_id=first_turn_id,
        payload={"turn_number": 1, "is_retry": False},
    )
    log.add("model.exchange.started", 0, turn_id=first_turn_id, operation_id="a" * 32)
    log.add(
        "model.exchange.finished",
        10 * _NS,
        turn_id=first_turn_id,
        operation_id="a" * 32,
        payload={
            "outcome": "success",
            "duration_ms": 10_000,
            "usage": {"normalized": {"input_total": 5, "output_total": 5}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add(
        EventType.TURN_FINISHED,
        _NS,
        turn_id=first_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    log.add(
        EventType.TURN_STARTED,
        20 * _NS,
        turn_id=retry_turn_id,
        payload={"turn_number": 1, "is_retry": True},
    )
    log.add(
        EventType.TURN_FINISHED,
        21 * _NS,
        turn_id=retry_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    loaded = analyzer.load(path).turns[0]
    assert len(loaded.attempts) == 2
    assert "usage sample lies outside its owning attempt axis" in loaded.diagnostics
    calls: list[tuple[str, bool]] = []
    real_counter_axis_diagnostics = turns_module._counter_axis_diagnostics

    def tracked_counter_axis_diagnostics(*args, **kwargs):
        calls.append((args[1], kwargs["refresh"]))
        return real_counter_axis_diagnostics(*args, **kwargs)

    def unexpected_full_projection(*_args, **_kwargs):
        raise AssertionError("live refresh rebuilt the full counter sample projection")

    monkeypatch.setattr(turns_module, "_counter_axis_diagnostics", tracked_counter_axis_diagnostics)
    monkeypatch.setattr(turns_module, "_usage_samples_by_turn", unexpected_full_projection)
    monkeypatch.setattr(turns_module, "_context_samples_by_turn", unexpected_full_projection)
    revision_id = "6" * 32
    log.add(
        EventType.CONTEXT_REVISION_RECORDED,
        22 * _NS,
        turn_id=retry_turn_id,
        operation_id=revision_id,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 1},
    )
    log.write(path)

    refreshed = analyzer.refresh()

    assert sorted(calls) == [(first_turn_id, False), (retry_turn_id, True)]
    assert "usage sample lies outside its owning attempt axis" in refreshed.turns[0].diagnostics
    assert "context sample lies outside its owning attempt axis" in refreshed.turns[0].diagnostics
    with pytest.raises(AssertionError, match="live refresh rebuilt the full counter sample projection"):
        analyzer.counter_samples()


def test_refresh_duplicate_revision_invalidates_every_affected_turn(tmp_path: Path) -> None:
    first_turn_id = "4" * 32
    second_turn_id = "5" * 32
    revision_id = "6" * 32
    log = EventLog()
    log.coverage()
    log.add(EventType.TURN_STARTED, 0, turn_id=first_turn_id, payload={"turn_number": 1})
    log.add(
        EventType.CONTEXT_REVISION_RECORDED,
        10 * _NS,
        turn_id=first_turn_id,
        operation_id=revision_id,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 1},
    )
    log.add(
        EventType.TURN_FINISHED,
        5 * _NS,
        turn_id=first_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 5_000},
    )
    log.add(EventType.TURN_STARTED, 20 * _NS, turn_id=second_turn_id, payload={"turn_number": 2})
    log.add(
        EventType.TURN_FINISHED,
        21 * _NS,
        turn_id=second_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 1_000},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    initial = analyzer.load(path).turn(first_turn_id)
    assert initial is not None
    assert "context sample lies outside its owning attempt axis" in initial.diagnostics

    log.add(
        EventType.CONTEXT_REVISION_RECORDED,
        22 * _NS,
        turn_id=second_turn_id,
        operation_id="7" * 32,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 1},
    )
    log.write(path)

    refreshed = analyzer.refresh().turn(first_turn_id)
    cold = TrajectoryAnalyzer().load(path).turn(first_turn_id)

    assert refreshed is not None
    assert cold is not None
    assert refreshed == cold
    assert "context sample lies outside its owning attempt axis" not in refreshed.diagnostics


def test_rollback_reverses_superseded_usage_contributions_before_overview(tmp_path) -> None:
    """A later rollback removes earlier contributions instead of requiring inverse events."""
    old_turn_id = "4" * 32
    live_turn_id = "5" * 32
    old_branch_id = "3" * 32
    new_branch_id = "6" * 32
    log = EventLog()
    log.add("turn.started", 0, turn_id=old_turn_id, payload={"turn_number": 1})
    log.add("model.exchange.started", 0, turn_id=old_turn_id, operation_id="a" * 32)
    log.add(
        "model.exchange.finished",
        _NS,
        turn_id=old_turn_id,
        operation_id="a" * 32,
        payload={
            "outcome": "success",
            "duration_ms": 1000,
            "usage": {"normalized": {"input_total": 100, "output_total": 20}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add("turn.finished", _NS, turn_id=old_turn_id, payload={"end_reason": "cancelled", "duration_ms": 0})
    # A cold recorder resumes while rollback still owns the recovered branch.
    # These physical infrastructure lines fall inside the logical range that
    # rollback supersedes, but they remain the coverage/runtime of the live
    # turn opened on the successor branch.
    log.add(
        "trajectory.coverage.started",
        2 * _NS,
        turn_id=None,
        payload={"coverage_reason": "runtime_resumed"},
    )
    log.add("trajectory.runtime.started", 2 * _NS, turn_id=None)
    log.add(
        "trajectory.runtime.recovered",
        2 * _NS,
        turn_id=None,
        payload={"truncated_bytes": 0, "resumed_from_sequence": 4},
    )
    log.add(
        "session.rollback",
        2 * _NS,
        turn_id=None,
        branch_id=new_branch_id,
        payload={
            "old_branch_id": old_branch_id,
            "new_branch_id": new_branch_id,
            "superseded_from_sequence": 1,
            "superseded_to_sequence": 7,
        },
    )
    log.add(
        "branch.superseded",
        2 * _NS,
        turn_id=None,
        branch_id=new_branch_id,
        payload={"branch_id": old_branch_id, "superseded_by": new_branch_id},
    )
    log.add(
        "turn.started",
        3 * _NS,
        turn_id=live_turn_id,
        branch_id=new_branch_id,
        payload={"turn_number": 1},
    )
    log.add(
        "model.exchange.started",
        3 * _NS,
        turn_id=live_turn_id,
        operation_id="b" * 32,
        branch_id=new_branch_id,
    )
    log.add(
        "model.exchange.finished",
        4 * _NS,
        turn_id=live_turn_id,
        operation_id="b" * 32,
        branch_id=new_branch_id,
        payload={
            "outcome": "success",
            "duration_ms": 1000,
            "usage": {"normalized": {"input_total": 25, "output_total": 5}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add(
        "turn.finished",
        4 * _NS,
        turn_id=live_turn_id,
        branch_id=new_branch_id,
        payload={"end_reason": "cancelled", "duration_ms": 0},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert [turn.turn_id for turn in analysis.turns] == [live_turn_id]
    assert analysis.overview is not None
    assert analysis.overview.usage_tokens.value == 30
    assert analysis.diagnostics.rollback_projection_unresolved is False
    assert all(metric.precision is Precision.EXACT for metric in analysis.turns[0].wall_time_ns.values())
    assert analysis.turns[0].usage_tokens.precision is Precision.EXACT


def test_unmatched_branch_supersession_keeps_rollback_projection_unresolved(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add(
        "branch.superseded",
        1,
        turn_id=None,
        payload={"branch_id": "3" * 32, "superseded_by": "6" * 32},
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    reason = "session trajectory integrity is unresolved: unresolved rollback projection"
    assert analysis.diagnostics.rollback_projection_unresolved is True
    assert analysis.overview is not None
    assert analysis.overview.elapsed_ns == Metric(0, Precision.UNRESOLVED, reason)
    assert analysis.validation is not None
    assert analysis.validation.tool_count == Metric(0, Precision.UNRESOLVED, reason)


def test_one_slot_rollback_range_removes_that_exact_sequence(tmp_path) -> None:
    new_branch_id = "6" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(
        "session.rollback",
        1,
        turn_id=None,
        branch_id=new_branch_id,
        payload={
            "old_branch_id": "3" * 32,
            "new_branch_id": new_branch_id,
            "superseded_from_sequence": 2,
            "superseded_to_sequence": 2,
        },
    )
    path = tmp_path / "events.jsonl"
    log.write(path)

    assert analyze_trajectory(path).turns == ()


def test_serial_model_cycle_fans_in_from_prior_tool_subtree(tmp_path) -> None:
    call_item_id = "9" * 32
    result_item_id = "a" * 32
    revision_id = "6" * 32
    segment_group_id = "5" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 20 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 2 * _NS, parent_operation_id="c" * 32)
    log.span(
        "preparation",
        "e" * 32,
        2 * _NS,
        2 * _NS,
        parent_operation_id="c" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "f" * 32},
    )
    log.span(
        "tool.operation",
        "f" * 32,
        2 * _NS,
        10 * _NS,
        parent_operation_id="c" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "batch_index": 0,
            "parent_model_operation_id": "d" * 32,
            "call_item_id": call_item_id,
        },
        finish_payload={"result_item_id": result_item_id},
        links=_caused_by("e" * 32),
    )
    revision = log.add(
        "context.revision.recorded",
        10 * _NS,
        operation_id=revision_id,
        parent_operation_id="8" * 32,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 2, "unidentified_item_count": 0},
        segmented_fields=(
            SegmentedField(
                field_pointer="/payload/refs",
                segment_group_id=segment_group_id,
                segment_count=1,
            ),
        ),
    )
    log.add(
        "event.segment",
        10 * _NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": segment_group_id,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": call_item_id, "occurrence": 0, "position": 0, "action": "add"},
                {"item_id": result_item_id, "occurrence": 0, "position": 1, "action": "add"},
            ],
        },
    )
    log.span(
        "model.cycle",
        "7" * 32,
        10 * _NS,
        20 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "8" * 32},
    )
    log.span(
        "model.exchange",
        "8" * 32,
        10 * _NS,
        20 * _NS,
        parent_operation_id="7" * 32,
        start_payload={"context_revision_id": revision_id},
    )
    log.add("turn.finished", 20 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.compute_cp_ns.value == 20 * _NS


def test_continuation_poll_wait_connects_neighboring_exchanges_serially(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 4 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        4 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "e" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, _NS, parent_operation_id="c" * 32)
    log.span(
        "model.exchange",
        "e" * 32,
        3 * _NS,
        4 * _NS,
        parent_operation_id="c" * 32,
        start_payload={"continuation_mode": "poll"},
    )
    log.add("turn.finished", 4 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "continuation-poll.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.value == 2 * _NS
    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.value == 4 * _NS
    assert turn.response_cp_ns.precision is Precision.EXACT


def test_invalid_duplicate_revision_segments_cannot_prove_tool_fan_in(tmp_path) -> None:
    call_item_id = "1" * 32
    result_item_id = "2" * 32
    revision_id = "3" * 32
    segment_group_id = "4" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 20 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 2 * _NS, parent_operation_id="c" * 32)
    log.span(
        "preparation",
        "e" * 32,
        2 * _NS,
        2 * _NS,
        parent_operation_id="d" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "f" * 32},
    )
    log.span(
        "tool.operation",
        "f" * 32,
        2 * _NS,
        10 * _NS,
        parent_operation_id="d" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "parent_model_operation_id": "d" * 32,
            "call_item_id": call_item_id,
        },
        finish_payload={"result_item_id": result_item_id},
        links=_caused_by("e" * 32),
    )
    revision = log.add(
        "context.revision.recorded",
        10 * _NS,
        operation_id=revision_id,
        parent_operation_id="9" * 32,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 2, "unidentified_item_count": 0},
        segmented_fields=(
            SegmentedField(
                field_pointer="/payload/refs",
                segment_group_id=segment_group_id,
                segment_count=2,
            ),
        ),
    )
    for item_id in (call_item_id, result_item_id):
        log.add(
            "event.segment",
            10 * _NS,
            operation_id=None,
            payload={
                "parent_event_id": revision.event_id,
                "field_pointer": "/payload/refs",
                "segment_group_id": segment_group_id,
                "segment_index": 0,
                "segment_count": 2,
                "encoding": "array_slice",
                "entries": [{"item_id": item_id, "occurrence": 0, "position": 0, "action": "add"}],
            },
        )
    log.span(
        "model.cycle",
        "8" * 32,
        10 * _NS,
        20 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "9" * 32},
    )
    log.span(
        "model.exchange",
        "9" * 32,
        10 * _NS,
        20 * _NS,
        parent_operation_id="8" * 32,
        start_payload={"context_revision_id": revision_id},
    )
    log.add("turn.finished", 20 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "invalid-revision-segments.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert any("invalid segmented membership" in diagnostic for diagnostic in turn.diagnostics)


@pytest.mark.parametrize(
    ("revision_payload", "expected_diagnostic", "fingerprint_key"),
    [
        (
            {"item_count": 1, "unidentified_item_count": 0},
            "item_count does not match replayed membership",
            None,
        ),
        (
            {"item_count": 2, "unidentified_item_count": 0, "membership_hash": "0" * 64},
            "membership_hash does not match replay",
            b"k" * 32,
        ),
    ],
)
def test_invalid_frozen_revision_membership_cannot_forge_tool_fan_in(
    tmp_path,
    revision_payload: dict[str, object],
    expected_diagnostic: str,
    fingerprint_key: bytes | None,
) -> None:
    path = _write_fan_in_membership_probe(tmp_path, revision_payload)

    turn = analyze_trajectory(path, fingerprint_key=fingerprint_key).turns[0]

    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert any(expected_diagnostic in diagnostic for diagnostic in turn.diagnostics)


def test_installed_fingerprint_key_falls_back_to_the_recorder_config_directory(tmp_path, monkeypatch) -> None:
    """A custom session root stores sessions away from the config directory
    that holds the recorder's key; the reader must still find it there."""
    import chrys.foundation.platform as platform_module
    from chrys.foundation.trajectory.keys import load_or_create_fingerprint_key

    events = tmp_path / "custom-root" / "sessions" / "abc123" / "trajectory" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.touch()
    config_dir = tmp_path / "config"
    recorder_key = load_or_create_fingerprint_key(config_dir)

    real_platform = platform_module.get_platform()

    class _RecorderPlatform:
        def __getattr__(self, name: str) -> object:
            return getattr(real_platform, name)

    installed = _RecorderPlatform()
    installed.__dict__["config_dir"] = config_dir
    monkeypatch.setattr(platform_module, "get_platform", lambda: installed)

    assert aggregation_module._read_installed_fingerprint_key(events) == recorder_key

    # A key beside the session root (the default layout, or a tree copied
    # along with its key) still outranks the local installation's key.
    beside_key = load_or_create_fingerprint_key(tmp_path / "custom-root")
    assert aggregation_module._read_installed_fingerprint_key(events) == beside_key


@pytest.mark.parametrize(
    ("parent_operation_id", "duplicate_revision", "expected_diagnostic"),
    [
        ("7" * 32, False, "does not belong to its claiming exchange"),
        ("9" * 32, True, "is defined more than once"),
    ],
)
def test_corrupt_revision_identity_or_exchange_ownership_cannot_prove_fan_in(
    tmp_path,
    parent_operation_id: str,
    duplicate_revision: bool,
    expected_diagnostic: str,
) -> None:
    path = _write_fan_in_membership_probe(
        tmp_path,
        {"item_count": 2, "unidentified_item_count": 0},
        parent_operation_id=parent_operation_id,
        duplicate_revision=duplicate_revision,
    )

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert any(expected_diagnostic in diagnostic for diagnostic in turn.diagnostics)


def test_event_carrier_item_id_proves_chat_style_tool_fan_in_without_session_store(tmp_path) -> None:
    carrier_item_id = "7" * 32
    path = _write_fan_in_membership_probe(
        tmp_path,
        {"item_count": 2, "unidentified_item_count": 0},
        result_carrier_item_id=carrier_item_id,
        result_membership_item_id=carrier_item_id,
    )

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.precision is Precision.EXACT


@pytest.mark.parametrize("with_session_store", [True, False])
def test_legacy_tool_fan_in_uses_only_a_reachable_session_carrier_mapping(
    tmp_path,
    with_session_store: bool,
) -> None:
    carrier_item_id = "7" * 32
    result_item_id = "2" * 32
    trajectory_dir = tmp_path / "sessions" / "fixture" / "trajectory"
    trajectory_dir.mkdir(parents=True)
    path = _write_fan_in_membership_probe(
        trajectory_dir,
        {"item_count": 2, "unidentified_item_count": 0},
        result_membership_item_id=carrier_item_id,
        file_name="events.jsonl",
    )
    if with_session_store:
        session = {
            "state": {
                "messages": [
                    {
                        "role": "tool",
                        "additional_properties": {ANALYTICS_ITEM_ID_KEY: carrier_item_id},
                        "contents": [
                            {
                                "type": "function_result",
                                "additional_properties": {ANALYTICS_ITEM_ID_KEY: result_item_id},
                            }
                        ],
                    }
                ]
            }
        }
        (trajectory_dir.parent / "session.json").write_text(json.dumps(session), encoding="utf-8")

    turn = analyze_trajectory(path).turns[0]

    if with_session_store:
        assert turn.compute_cp_ns.precision is Precision.EXACT
        assert turn.response_cp_ns.precision is Precision.EXACT
    else:
        assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
        assert turn.response_cp_ns.precision is Precision.UNRESOLVED
        assert "carrier mapping unavailable" in turn.diagnostics


def test_conflicting_session_carrier_mappings_are_not_treated_as_proof(tmp_path) -> None:
    content_id = "2" * 32
    trajectory_dir = tmp_path / "sessions" / "fixture" / "trajectory"
    trajectory_dir.mkdir(parents=True)
    path = trajectory_dir / "events.jsonl"
    messages = [
        {
            "role": "tool",
            "additional_properties": {ANALYTICS_ITEM_ID_KEY: carrier_id},
            "contents": [
                {
                    "type": "function_result",
                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: content_id},
                }
            ],
        }
        for carrier_id in ("7" * 32, "8" * 32)
    ]
    (trajectory_dir.parent / "session.json").write_text(
        json.dumps({"state": {"messages": messages}}),
        encoding="utf-8",
    )

    projection = session_projection_module._read_session_projection(path)

    assert content_id not in projection.carriers


def test_conflicting_session_token_counts_are_not_used_for_context_load(tmp_path) -> None:
    item_id = "7" * 32
    content_id = "2" * 32
    trajectory_dir = tmp_path / "sessions" / "fixture" / "trajectory"
    trajectory_dir.mkdir(parents=True)
    path = trajectory_dir / "events.jsonl"
    messages = [
        {
            "role": "assistant",
            "additional_properties": {ANALYTICS_ITEM_ID_KEY: item_id, "_group": {"token_count": token_count}},
            "contents": [
                {
                    "type": "function_call",
                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: content_id},
                    "arguments": json.dumps(
                        {"command": f"command-{token_count}", "skill_name": f"skill-{token_count}"}
                    ),
                }
            ],
        }
        for token_count in (10, 20)
    ]
    (trajectory_dir.parent / "session.json").write_text(
        json.dumps({"state": {"messages": messages}}),
        encoding="utf-8",
    )

    projection = session_projection_module._read_session_projection(path)

    assert item_id not in projection.item_tokens
    assert content_id not in projection.commands
    assert content_id not in projection.skill_names


def test_invalid_utf8_session_document_degrades_to_an_unavailable_projection(tmp_path) -> None:
    trajectory_dir = tmp_path / "sessions" / "fixture" / "trajectory"
    trajectory_dir.mkdir(parents=True)
    path = trajectory_dir / "events.jsonl"
    # json.load decodes the raw bytes itself, so invalid UTF-8 must degrade
    # exactly like invalid JSON instead of escaping as UnicodeDecodeError.
    (trajectory_dir.parent / "session.json").write_bytes(b'{"state": {"messages": ["\xff"]}}')

    projection = session_projection_module._read_session_projection(path)

    assert projection.available is False


def test_nested_past_decoder_recursion_session_document_degrades_to_an_unavailable_projection(tmp_path) -> None:
    trajectory_dir = tmp_path / "sessions" / "fixture" / "trajectory"
    trajectory_dir.mkdir(parents=True)
    path = trajectory_dir / "events.jsonl"
    # Valid JSON nested past the decoder's recursion budget raises
    # RecursionError rather than JSONDecodeError; it must degrade the same way.
    (trajectory_dir.parent / "session.json").write_bytes(b"[" * 100_000 + b"]" * 100_000)

    projection = session_projection_module._read_session_projection(path)

    assert projection.available is False


def test_oversized_integer_session_document_degrades_to_an_unavailable_projection(tmp_path) -> None:
    trajectory_dir = tmp_path / "sessions" / "fixture" / "trajectory"
    trajectory_dir.mkdir(parents=True)
    path = trajectory_dir / "events.jsonl"
    # An integer token past the digit limit raises a bare ValueError rather
    # than JSONDecodeError; it must degrade the same way.
    (trajectory_dir.parent / "session.json").write_bytes(b'{"state": {"n": ' + b"1" * 5000 + b"}}")

    projection = session_projection_module._read_session_projection(path)

    assert projection.available is False


def test_live_tail_reprojects_clean_legacy_turn_when_session_carrier_mapping_changes(tmp_path) -> None:
    carrier_item_id = "7" * 32
    result_item_id = "2" * 32
    trajectory_dir = tmp_path / "sessions" / "fixture" / "trajectory"
    trajectory_dir.mkdir(parents=True)
    path = _write_fan_in_membership_probe(
        trajectory_dir,
        {"item_count": 2, "unidentified_item_count": 0},
        result_membership_item_id=carrier_item_id,
        file_name="events.jsonl",
    )
    analyzer = TrajectoryAnalyzer()
    initial = analyzer.load(path)
    assert initial.turns[0].compute_cp_ns.precision is Precision.UNRESOLVED

    session = {
        "state": {
            "messages": [
                {
                    "role": "tool",
                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: carrier_item_id},
                    "contents": [
                        {
                            "type": "function_result",
                            "additional_properties": {ANALYTICS_ITEM_ID_KEY: result_item_id},
                        }
                    ],
                }
            ]
        }
    }
    session_path = trajectory_dir.parent / "session.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    _append_profile_switched_event(path, monotonic_ns=21 * _NS)

    mapped = analyzer.refresh()

    assert mapped.turns[0].compute_cp_ns.precision is Precision.EXACT
    assert mapped.turns[0].response_cp_ns.precision is Precision.EXACT

    session_path.unlink()
    _append_profile_switched_event(path, monotonic_ns=22 * _NS)

    unavailable = analyzer.refresh()

    assert unavailable.turns[0].compute_cp_ns.precision is Precision.UNRESOLVED
    assert unavailable.turns[0].response_cp_ns.precision is Precision.UNRESOLVED
    assert "carrier mapping unavailable" in unavailable.turns[0].diagnostics


def _append_profile_switched_event(path, *, monotonic_ns: int) -> None:
    appended = EventLog()
    appended.add("profile.switched", monotonic_ns, turn_id=None, payload={"profile": "Code"})
    append_path = path.with_name("append.jsonl")
    appended.write(append_path, start_sequence=len(path.read_bytes().splitlines()) + 1)
    path.write_bytes(path.read_bytes() + append_path.read_bytes())


def test_real_shape_acceptance_keeps_side_calls_out_and_accepts_carrier_and_approval_shapes(tmp_path) -> None:
    side_actor = Actor(kind="side_call", role="approval_judge", actor_id="0" * 32)
    side_revision_id = "1" * 32
    side_exchange_id = "2" * 32
    side_segment_id = "3" * 32
    call_item_id = "4" * 32
    result_item_id = "5" * 32
    carrier_item_id = "6" * 32
    consuming_revision_id = "7" * 32
    consuming_segment_id = "8" * 32
    approval_id = "9" * 32
    tool_id = "f" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 20 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 2 * _NS, parent_operation_id="c" * 32)
    log.add(
        "approval.requested",
        1_800_000_000,
        operation_id=approval_id,
        parent_operation_id="d" * 32,
        payload={"approval_request_id": approval_id, "target_tool_operation_id": tool_id},
    )
    side_revision = log.add(
        "context.revision.recorded",
        2_100_000_000,
        operation_id=side_revision_id,
        parent_operation_id=side_exchange_id,
        payload={
            "revision_id": side_revision_id,
            "is_checkpoint": True,
            "item_count": 0,
            "unidentified_item_count": 2,
        },
        segmented_fields=(
            SegmentedField(field_pointer="/payload/refs", segment_group_id=side_segment_id, segment_count=1),
        ),
        actor=side_actor,
    )
    log.add(
        "event.segment",
        2_100_000_000,
        operation_id=None,
        payload={
            "parent_event_id": side_revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": side_segment_id,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [],
        },
        actor=side_actor,
    )
    log.span(
        "model.exchange",
        side_exchange_id,
        2_200_000_000,
        2_700_000_000,
        parent_operation_id="d" * 32,
        start_payload={"context_revision_id": side_revision_id},
        actor=side_actor,
    )
    log.add(
        "approval.resolved",
        3 * _NS,
        operation_id=approval_id,
        parent_operation_id="d" * 32,
        payload={
            "approval_request_id": approval_id,
            "target_tool_operation_id": tool_id,
            "outcome": "approved",
            "wait_ms": 2500,
        },
        measurements={"/payload/wait_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.span(
        "preparation",
        "e" * 32,
        3 * _NS,
        3 * _NS,
        parent_operation_id="d" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": tool_id},
    )
    log.span(
        "tool.operation",
        tool_id,
        3 * _NS,
        10 * _NS,
        parent_operation_id="d" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "parent_model_operation_id": "d" * 32,
            "call_item_id": call_item_id,
        },
        finish_payload={
            "result_item_id": result_item_id,
            "result_carrier_item_id": carrier_item_id,
        },
        links=_caused_by("e" * 32),
    )
    revision = log.add(
        "context.revision.recorded",
        10 * _NS,
        operation_id=consuming_revision_id,
        parent_operation_id="0" * 32,
        payload={
            "revision_id": consuming_revision_id,
            "is_checkpoint": True,
            "item_count": 2,
            "unidentified_item_count": 0,
        },
        segmented_fields=(
            SegmentedField(
                field_pointer="/payload/refs",
                segment_group_id=consuming_segment_id,
                segment_count=1,
            ),
        ),
    )
    log.add(
        "event.segment",
        10 * _NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": consuming_segment_id,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": call_item_id, "occurrence": 0, "position": 0, "action": "add"},
                {"item_id": carrier_item_id, "occurrence": 0, "position": 1, "action": "add"},
            ],
        },
    )
    log.span(
        "model.cycle",
        "9" * 32,
        10 * _NS,
        20 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "0" * 32},
    )
    log.span(
        "model.exchange",
        "0" * 32,
        10 * _NS,
        20 * _NS,
        parent_operation_id="9" * 32,
        start_payload={"context_revision_id": consuming_revision_id},
    )
    log.add("turn.finished", 20 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "real-shapes.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)
    turn = analysis.turns[0]

    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.precision is Precision.EXACT
    assert side_revision_id not in (turn.compute_cp_ns.reason or "")
    assert side_revision_id not in (turn.response_cp_ns.reason or "")
    assert len([operation for operation in turn.operations if operation.family == "model.exchange"]) == 2
    assert analysis.diagnostics.span_duration_mismatch_count == 0
    assert analysis.diagnostics.containment_violation_count == 0
    assert side_revision_id in analysis.diagnostics.side_call_empty_shell_revisions


def test_delta_membership_replay_requires_exact_occurrence_and_position() -> None:
    parent = (("1" * 32, 0), ("2" * 32, 0))
    valid = (
        _RevisionEntry("2" * 32, 0, 1, "remove"),
        _RevisionEntry("3" * 32, 0, 1, "add"),
    )
    wrong_occurrence = (
        _RevisionEntry("2" * 32, 1, 1, "remove"),
        _RevisionEntry("3" * 32, 0, 1, "add"),
    )

    assert _replay_delta(parent, valid) == (("1" * 32, 0), ("3" * 32, 0))
    assert _replay_delta(parent, wrong_occurrence) is None


def test_invalid_context_membership_is_diagnostic_without_poisoning_pure_timing(tmp_path) -> None:
    revision_id = "1" * 32
    exchange_id = "2" * 32
    group_id = "3" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": exchange_id},
    )
    revision = log.add(
        "context.revision.recorded",
        0,
        operation_id=revision_id,
        parent_operation_id=exchange_id,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 1, "unidentified_item_count": 0},
        segmented_fields=(SegmentedField(field_pointer="/payload/refs", segment_group_id=group_id, segment_count=1),),
    )
    log.add(
        "event.segment",
        0,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": group_id,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": "4" * 32, "occurrence": 0, "position": 0, "action": "add"},
                {"item_id": "5" * 32, "occurrence": 0, "position": 1, "action": "add"},
            ],
        },
    )
    log.span(
        "model.exchange",
        exchange_id,
        0,
        _NS,
        parent_operation_id="c" * 32,
        start_payload={"context_revision_id": revision_id},
    )
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "invalid-context-only.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.precision is Precision.EXACT
    assert any("item_count does not match replayed membership" in item for item in turn.diagnostics)


def test_retry_run_chain_uses_previous_run_pointer_without_time_adjacency(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * _NS, links=_caused_by("a" * 32))
    log.add(
        "retry.scheduled",
        10 * _NS,
        operation_id=None,
        payload={
            "retry_mode": "run",
            "previous_operation_id": "b" * 32,
            "next_operation_id": "c" * 32,
            "delay_ms": 1000,
        },
    )
    log.add(
        "retry.started",
        11 * _NS,
        operation_id="c" * 32,
        payload={
            "retry_mode": "run",
            "previous_operation_id": "b" * 32,
            "next_operation_id": "c" * 32,
        },
    )
    log.span(
        "model.run",
        "c" * 32,
        11 * _NS,
        21 * _NS,
        start_payload={"previous_run_operation_id": "b" * 32},
    )
    log.span(
        "model.cycle",
        "d" * 32,
        21 * _NS,
        21 * _NS,
        parent_operation_id="c" * 32,
        finish_payload={"final_exchange_operation_id": "e" * 32},
    )
    log.span("model.exchange", "e" * 32, 21 * _NS, 21 * _NS, parent_operation_id="d" * 32)
    log.add("turn.finished", 21 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)
    assert len(analysis.turns) == 1
    turn = analysis.turns[0]

    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.compute_cp_ns.value == 20 * _NS
    assert turn.response_cp_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.value == 21 * _NS


def test_response_cp_without_typed_final_exchange_is_unresolved(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * _NS, links=_caused_by("a" * 32))
    log.span("model.cycle", "c" * 32, 0, 10 * _NS, parent_operation_id="b" * 32)
    log.span("model.exchange", "d" * 32, 0, 10 * _NS, parent_operation_id="c" * 32)
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.EXACT
    assert turn.compute_cp_ns.value == 10 * _NS
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.value is None


@pytest.mark.parametrize("typed_fork", [False, True])
def test_response_cp_requires_typed_root_to_fence_path_and_deduplicates_fork_overlap(
    tmp_path,
    typed_fork: bool,
) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 10 * _NS, parent_operation_id="c" * 32)
    log.span(
        "hook.operation",
        "e" * 32,
        8 * _NS,
        13 * _NS,
        start_payload={
            "hook_event": "after_turn",
            "execution_mode": "async",
            "scope": "turn",
            **({"target_operation_id": "d" * 32} if typed_fork else {}),
        },
        links=_caused_by("d" * 32) if typed_fork else (),
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(13 * _NS, waited_hook_ids=["e" * 32])
    path = tmp_path / "events.jsonl"
    log.write(path)

    response_cp = analyze_trajectory(path).turns[0].response_cp_ns

    if typed_fork:
        assert response_cp.precision is Precision.EXACT
        assert response_cp.value == 13 * _NS
    else:
        assert response_cp.precision is Precision.UNRESOLVED
        assert response_cp.value is None


def test_usage_total_is_missing_when_one_required_normalized_bucket_is_absent(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("model.exchange.started", 0, operation_id="a" * 32)
    log.add(
        "model.exchange.finished",
        _NS,
        operation_id="a" * 32,
        payload={"outcome": "success", "duration_ms": 1000, "usage": {"normalized": {"input_total": 12}}},
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    usage = analyze_trajectory(path).turns[0].usage_tokens

    assert usage.value is None
    assert usage.precision is Precision.MISSING


def test_negative_normalized_usage_is_missing_instead_of_exact(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    _exchange_usage(log, "a" * 32, 0, _NS, {"input_total": -1, "output_total": 20})
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    usage = analyze_trajectory(path).turns[0].usage_tokens

    assert usage.value is None
    assert usage.precision is Precision.MISSING


def test_usage_is_missing_when_a_main_exchange_has_no_terminal_event(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("model.exchange.started", 0, operation_id="a" * 32)
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    usage = analyze_trajectory(path).turns[0].usage_tokens

    assert usage.value is None
    assert usage.precision is Precision.MISSING


def test_sub_agent_subgraph_nests_under_its_tool_and_counts_once(tmp_path) -> None:
    """An in-process sub-agent is the tool's displaced subgraph, not a side call."""
    sub_actor = Actor(kind="agent", role="sub_agent", actor_id="0" * 32)
    call_item_id = "9" * 32
    result_item_id = "a" * 32
    revision_id = "6" * 32
    segment_group_id = "5" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 20 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    _exchange_usage(log, "d" * 32, 0, 2 * _NS, {"input_total": 10, "output_total": 1}, parent_operation_id="c" * 32)
    log.span(
        "preparation",
        "e" * 32,
        2 * _NS,
        2 * _NS,
        parent_operation_id="c" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "f" * 32},
    )
    log.span(
        "tool.operation",
        "f" * 32,
        2 * _NS,
        10 * _NS,
        parent_operation_id="c" * 32,
        start_payload={
            "tool_name": "explore_agent",
            "tool_kind": "sub_agent",
            "batch_index": 0,
            "parent_model_operation_id": "d" * 32,
            "call_item_id": call_item_id,
        },
        finish_payload={"result_item_id": result_item_id},
        links=_caused_by("e" * 32),
    )
    log.span(
        "sub_agent",
        "1" * 32,
        2 * _NS,
        9 * _NS,
        parent_operation_id="f" * 32,
        start_payload={"invocation_id": "deadbeef1234", "parent_tool_operation_id": "f" * 32},
    )
    log.span(
        "model.cycle",
        "2" * 32,
        3 * _NS,
        8 * _NS,
        parent_operation_id="1" * 32,
        actor=sub_actor,
        finish_payload={"final_exchange_operation_id": "3" * 32},
    )
    _exchange_usage(
        log,
        "3" * 32,
        3 * _NS,
        8 * _NS,
        {"input_total": 100, "output_total": 20},
        parent_operation_id="2" * 32,
        actor=sub_actor,
    )
    revision = log.add(
        "context.revision.recorded",
        10 * _NS,
        operation_id=revision_id,
        parent_operation_id="8" * 32,
        payload={"revision_id": revision_id, "is_checkpoint": True, "item_count": 2, "unidentified_item_count": 0},
        segmented_fields=(
            SegmentedField(field_pointer="/payload/refs", segment_group_id=segment_group_id, segment_count=1),
        ),
    )
    log.add(
        "event.segment",
        10 * _NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": segment_group_id,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": call_item_id, "occurrence": 0, "position": 0, "action": "add"},
                {"item_id": result_item_id, "occurrence": 0, "position": 1, "action": "add"},
            ],
        },
    )
    log.span(
        "model.cycle",
        "7" * 32,
        10 * _NS,
        20 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "8" * 32},
    )
    _exchange_usage(
        log,
        "8" * 32,
        10 * _NS,
        20 * _NS,
        {"input_total": 30, "output_total": 3},
        parent_operation_id="7" * 32,
        start_payload={"context_revision_id": revision_id},
    )
    log.add("turn.finished", 20 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    analysis = analyzer.load(path)
    turn = analysis.turns[0]

    assert turn.diagnostics == ()
    # The sub-agent's exchange is real usage of the turn.
    assert (turn.usage_tokens.value, turn.usage_tokens.precision) == (164, Precision.EXACT)
    # The consumer waited for the whole tool, sub-agent included.
    assert (turn.compute_cp_ns.value, turn.compute_cp_ns.precision) == (20 * _NS, Precision.EXACT)
    # Displacement keeps the nested time counted exactly once.
    assert (turn.exclusive_work_ns.value, turn.exclusive_work_ns.precision) == (20 * _NS, Precision.EXACT)
    assert turn.wall_time_ns[WallBucket.MODEL].value == 17 * _NS
    assert turn.wall_time_ns[WallBucket.TOOLS].value == 3 * _NS
    operation_ids = {operation.operation_id for operation in turn.operations}
    assert {"1" * 32, "2" * 32, "3" * 32} <= operation_ids
    assert analyzer.counter_samples().usage_by_turn[turn.turn_id][-1].input_tokens == 30


def test_cross_runtime_turn_terminal_degrades_wall_and_derived_metrics(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(
        "turn.finished",
        5 * _NS,
        payload={"end_reason": "cancelled", "duration_ms": 0},
        runtime_id="9" * 32,
    )
    path = tmp_path / "cross-runtime-turn.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.elapsed_ns.precision is Precision.UNRESOLVED
    assert all(metric.precision is Precision.UNRESOLVED for metric in turn.wall_time_ns.values())
    assert turn.exclusive_work_ns.precision is Precision.UNRESOLVED
    assert turn.parallelism.precision is Precision.UNRESOLVED
    assert turn.overlap_gain_ns.precision is Precision.UNRESOLVED
    assert "turn interval has invalid monotonic endpoints" in turn.diagnostics


def test_main_role_lifecycle_endpoints_from_different_actors_are_not_paired_exactly(tmp_path) -> None:
    other_main_actor = Actor(kind="agent", role="main", actor_id="f" * 32)
    operation_id = "a" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("model.exchange.started", 0, operation_id=operation_id)
    log.add(
        "model.exchange.finished",
        _NS,
        operation_id=operation_id,
        actor=other_main_actor,
        payload={
            "outcome": "success",
            "duration_ms": 1000,
            "usage": {"normalized": {"input_total": 25, "output_total": 5}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "actor-mismatch.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.usage_tokens == Metric(
        30,
        Precision.UNRESOLVED,
        "usage requires a uniquely closed exchange and exact turn lifecycle",
    )
    operation = next(item for item in turn.operations if item.operation_id == operation_id)
    assert operation.precision is Precision.UNRESOLVED


@pytest.mark.parametrize(
    ("runtime_id", "branch_id"),
    [("9" * 32, "3" * 32), ("1" * 32, "9" * 32)],
)
def test_foreign_runtime_or_branch_subtree_never_enters_turn_arithmetic(
    tmp_path,
    runtime_id: str,
    branch_id: str,
) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span(
        "preparation",
        "a" * 32,
        0,
        0,
        start_payload={"scope": "turn_preamble", "phase": "dispatch"},
        runtime_id=runtime_id,
        branch_id=branch_id,
    )
    log.span(
        "model.run",
        "b" * 32,
        0,
        10 * _NS,
        links=_caused_by("a" * 32),
        runtime_id=runtime_id,
        branch_id=branch_id,
    )
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
        runtime_id=runtime_id,
        branch_id=branch_id,
    )
    log.span(
        "model.exchange",
        "d" * 32,
        0,
        10 * _NS,
        parent_operation_id="c" * 32,
        runtime_id=runtime_id,
        branch_id=branch_id,
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "foreign-subtree.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.elapsed_ns.precision is Precision.UNRESOLVED
    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert turn.exclusive_work_ns.precision is Precision.UNRESOLVED
    assert turn.parallelism.precision is Precision.UNRESOLVED
    assert turn.overlap_gain_ns.precision is Precision.UNRESOLVED
    assert all(metric.precision is Precision.UNRESOLVED for metric in turn.wall_time_ns.values())
    assert all(metric.precision is Precision.UNRESOLVED for metric in turn.utilization.values())
    assert not any(item.operation_id is not None for item in turn.slices)
    assert any("owning turn runtime and branch" in diagnostic for diagnostic in turn.diagnostics)


def test_lifecycle_terminal_must_follow_its_start_in_sequence_order(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.add("model.run.started", 0, operation_id="b" * 32, links=_caused_by("a" * 32))
    log.add("model.cycle.started", 0, operation_id="c" * 32, parent_operation_id="b" * 32)
    log.add(
        "model.exchange.finished",
        10 * _NS,
        operation_id="d" * 32,
        parent_operation_id="c" * 32,
        payload={"outcome": "success", "duration_ms": 10_000},
        measurements={"/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    log.add("model.exchange.started", 0, operation_id="d" * 32, parent_operation_id="c" * 32)
    log.add(
        "model.cycle.finished",
        10 * _NS,
        operation_id="c" * 32,
        parent_operation_id="b" * 32,
        payload={"outcome": "success", "duration_ms": 10_000, "final_exchange_operation_id": "d" * 32},
        measurements={"/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    log.add(
        "model.run.finished",
        10 * _NS,
        operation_id="b" * 32,
        payload={"outcome": "success", "duration_ms": 10_000},
        measurements={"/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "reversed-exchange.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert any("model.exchange interval has invalid monotonic endpoints" in item for item in turn.diagnostics)


def test_orphan_usage_terminal_is_unresolved_even_with_complete_buckets(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(
        "model.exchange.finished",
        _NS,
        operation_id="a" * 32,
        payload={
            "outcome": "success",
            "duration_ms": 1000,
            "usage": {"normalized": {"input_total": 100, "output_total": 20}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "orphan-usage.jsonl"
    log.write(path)

    usage = analyze_trajectory(path).turns[0].usage_tokens

    assert usage.value == 120
    assert usage.precision is Precision.UNRESOLVED
    assert usage.reason == "usage requires a uniquely closed exchange and exact turn lifecycle"


def test_response_fence_rejects_duplicate_hook_membership_ids(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 10 * _NS, parent_operation_id="c" * 32)
    for hook_id, end_ns in (("e" * 32, 5 * _NS), ("f" * 32, 15 * _NS)):
        log.span(
            "hook.operation",
            hook_id,
            4 * _NS,
            end_ns,
            start_payload={
                "hook_event": "after_turn",
                "execution_mode": "async",
                "scope": "turn",
                "target_operation_id": "d" * 32,
            },
            links=_caused_by("d" * 32),
        )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(16 * _NS, waited_hook_ids=["e" * 32, "e" * 32])
    path = tmp_path / "duplicate-fence-members.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert any("hook membership is incomplete" in item for item in turn.diagnostics)


def test_response_fence_accepts_empty_drained_scopes_without_hooks(tmp_path) -> None:
    """No hook manager means there was no turn scope to drain, not a partial fence."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 10 * _NS, parent_operation_id="c" * 32)
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(10 * _NS, drained_scopes=[])
    path = tmp_path / "no-hooks-fence.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.elapsed_ns.precision is Precision.EXACT
    assert turn.response_cp_ns.precision is Precision.EXACT
    assert not any("drained_scopes" in item for item in turn.diagnostics)


def test_corruption_between_response_fence_and_segment_degrades_turn_integrity(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 10 * _NS, parent_operation_id="c" * 32)
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(11 * _NS)
    path = tmp_path / "corrupt-fence-evidence.jsonl"
    log.write(path)
    lines = path.read_bytes().splitlines(keepends=True)
    lines.insert(-1, b"{corrupt between fence and segment}\n")
    path.write_bytes(b"".join(lines))

    turn = analyze_trajectory(path).turns[0]

    assert turn.elapsed_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert any("corrupt trajectory line intersects the turn" in item for item in turn.diagnostics)


def test_turn_after_closed_coverage_and_runtime_is_unresolved(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("trajectory.coverage.ended", 0, turn_id=None, payload={"last_sequence": 1})
    log.add("trajectory.runtime.finished", 0, turn_id=None, payload={"reason": "session_close"})
    log.add("turn.started", _NS, payload={"turn_number": 1})
    log.add("turn.finished", 2 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "closed-runtime-turn.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.elapsed_ns.precision is Precision.UNRESOLVED
    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert any("outside its trajectory coverage window" in item for item in turn.diagnostics)
    assert any("after trajectory.runtime.finished" in item for item in turn.diagnostics)


def test_usage_is_unresolved_when_a_corrupt_line_intersects_the_turn(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("model.exchange.started", 0, operation_id="a" * 32)
    log.add(
        "model.exchange.finished",
        _NS,
        operation_id="a" * 32,
        payload={
            "outcome": "success",
            "duration_ms": 1000,
            "usage": {"normalized": {"input_total": 100, "output_total": 20}},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            "/payload/usage/normalized/input_total": {"source": "provider", "adapter_version": 1},
            "/payload/usage/normalized/output_total": {"source": "provider", "adapter_version": 1},
        },
    )
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "corrupt-usage-region.jsonl"
    log.write(path)
    lines = path.read_bytes().splitlines(keepends=True)
    lines.insert(3, b"{complete but corrupt}\n")
    path.write_bytes(b"".join(lines))

    analysis = analyze_trajectory(path)
    usage = analysis.turns[0].usage_tokens

    assert usage.value == 120
    assert usage.precision is Precision.UNRESOLVED
    assert usage.reason == "corrupt trajectory line intersects the turn"
    assert analysis.diagnostics.corrupt_lines[0].after_sequence == 3
    assert analysis.diagnostics.corrupt_lines[0].line_number == 4


def test_closed_turn_before_later_prefix_violation_remains_exact(tmp_path) -> None:
    first = EventLog()
    first.coverage()
    first.turn(0, _NS)
    path = tmp_path / "regional-prefix.jsonl"
    first.write(path)
    second_turn_id = "5" * 32
    second = EventLog()
    second.add("turn.started", 2 * _NS, turn_id=second_turn_id, payload={"turn_number": 2})
    second.add(
        "turn.finished",
        3 * _NS,
        turn_id=second_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 0},
    )
    append_path = tmp_path / "regional-prefix-append.jsonl"
    second.write(append_path, start_sequence=5)
    path.write_bytes(path.read_bytes() + append_path.read_bytes())

    analysis = analyze_trajectory(path)

    assert analysis.turns[0].elapsed_ns.precision is Precision.EXACT
    assert analysis.turns[1].elapsed_ns.precision is Precision.UNRESOLVED
    assert analysis.diagnostics.accounted_prefix_violations == (
        "sequence 4 is missing and not covered by an earlier gap",
    )
    assert analysis.diagnostics.accounted_prefix_violation_details[0].first_sequence == 4
    assert analysis.diagnostics.accounted_prefix_violation_details[0].last_sequence == 4


def test_closed_producing_exchange_is_a_normal_tool_containment_shape(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 3 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        2 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 2 * _NS, parent_operation_id="c" * 32)
    log.span(
        "preparation",
        "e" * 32,
        2 * _NS,
        2 * _NS,
        parent_operation_id="d" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "f" * 32},
    )
    log.span(
        "tool.operation",
        "f" * 32,
        2 * _NS,
        3 * _NS,
        parent_operation_id="d" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": "filesystem.read",
            "parent_model_operation_id": "d" * 32,
            "call_item_id": "1" * 32,
        },
        finish_payload={"result_item_id": "2" * 32},
        links=_caused_by("e" * 32),
    )
    log.add("turn.finished", 3 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "closed-producing-exchange.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert analysis.diagnostics.containment_violation_count == 0


def test_critical_path_prunes_interval_unions_dominated_at_the_same_node() -> None:
    intervals = {
        "root": [(0, 1)],
        "short": [(1, 2)],
        "long": [(1, 3)],
        "terminal": [],
    }
    edges = {"root": {"short", "long"}, "short": {"terminal"}, "long": {"terminal"}}

    result = _longest_interval_path(intervals, edges, root_id="root", terminal_id="terminal")

    assert result.acyclic is True
    assert result.bounded is True
    assert result.value == 3


def test_sixteen_layer_diamond_critical_path_stays_bounded() -> None:
    intervals, edges, terminal = _layered_diamond(16)

    result = _longest_interval_path(intervals, edges, root_id="root", terminal_id=terminal)

    assert result.acyclic is True
    if result.bounded:
        assert result.value is not None
    else:
        assert result.value is None


def test_disjoint_critical_path_is_certified_without_candidate_enumeration(monkeypatch) -> None:
    """A max-sum path whose intervals never overlap bounds every interval union."""
    monkeypatch.setattr(critical_path_module, "_MAX_CRITICAL_PATH_CANDIDATES", 0)
    intervals = {"root": [(0, 1)], "left": [(1, 3)], "right": [(1, 2)], "terminal": [(3, 4)]}
    edges = {"root": {"left", "right"}, "left": {"terminal"}, "right": {"terminal"}}

    result = _longest_interval_path(intervals, edges, root_id="root", terminal_id="terminal")

    assert (result.value, result.acyclic, result.bounded) == (4, True, True)


def test_completion_edges_carry_displaced_descendants_but_forks_do_not() -> None:
    intervals = {
        "exchange": [(0, 1)],
        "tool": [(1, 2)],
        "approval": [(2, 5)],
        "hook": [(1, 2)],
        "consumer": [(5, 6)],
    }
    edges = {"exchange": {"tool"}, "tool": {"approval", "hook", "consumer"}, "hook": set(), "approval": set()}
    parents = {"tool": "exchange", "approval": "tool"}

    through_consumer = _longest_interval_path(
        intervals,
        edges,
        parents=parents,
        fork_edges=frozenset({("tool", "hook")}),
        root_id="exchange",
        terminal_id="consumer",
    )
    through_fork = _longest_interval_path(
        intervals,
        edges,
        parents=parents,
        fork_edges=frozenset({("tool", "hook")}),
        root_id="exchange",
        terminal_id="hook",
    )

    assert (through_consumer.value, through_consumer.bounded) == (6, True)
    assert (through_fork.value, through_fork.bounded) == (2, True)


def test_critical_path_candidate_cap_degrades_metrics_with_diagnostic(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(critical_path_module, "_MAX_CRITICAL_PATH_CANDIDATES", 0)
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 10 * _NS, parent_operation_id="c" * 32)
    # The waited hook overlaps its fork origin, so no disjoint witness exists
    # and the bounded enumeration must run.
    log.span(
        "hook.operation",
        "e" * 32,
        8 * _NS,
        13 * _NS,
        start_payload={
            "hook_event": "after_turn",
            "execution_mode": "async",
            "scope": "turn",
            "target_operation_id": "d" * 32,
        },
        links=_caused_by("d" * 32),
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(13 * _NS, waited_hook_ids=["e" * 32])
    path = tmp_path / "critical-path-cap.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.compute_cp_ns.precision is Precision.UNRESOLVED
    assert turn.response_cp_ns.precision is Precision.UNRESOLVED
    assert "compute critical-path candidate cap exceeded" in turn.diagnostics
    assert "response critical-path candidate cap exceeded" in turn.diagnostics


def _layered_diamond(
    layer_count: int,
) -> tuple[dict[str, list[tuple[int, int]]], dict[str, set[str]], str]:
    intervals = {"root": [(0, 1)]}
    edges: dict[str, set[str]] = {}
    previous = "root"
    for layer in range(layer_count):
        left = f"left-{layer}"
        right = f"right-{layer}"
        join = f"join-{layer}"
        base = 1 + layer * 2
        intervals[left] = [(base, base + 1)]
        intervals[right] = [(base + 1, base + 2)]
        intervals[join] = []
        edges[previous] = {left, right}
        edges[left] = {join}
        edges[right] = {join}
        previous = join
    return intervals, edges, previous


def _write_fan_in_membership_probe(
    tmp_path,
    revision_payload: dict[str, object],
    *,
    parent_operation_id: str = "9" * 32,
    duplicate_revision: bool = False,
    result_carrier_item_id: str | None = None,
    result_membership_item_id: str | None = None,
    file_name: str = "membership-fan-in.jsonl",
    tool_kind: str = "filesystem.read",
    tool_context: dict[str, str] | None = None,
):
    call_item_id = "1" * 32
    result_item_id = "2" * 32
    revision_id = "3" * 32
    group_id = "4" * 32
    consuming_exchange_id = "9" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 20 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "c" * 32,
        0,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "d" * 32},
    )
    log.span("model.exchange", "d" * 32, 0, 2 * _NS, parent_operation_id="c" * 32)
    log.span(
        "preparation",
        "e" * 32,
        2 * _NS,
        2 * _NS,
        parent_operation_id="d" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "f" * 32},
    )
    log.span(
        "tool.operation",
        "f" * 32,
        2 * _NS,
        10 * _NS,
        parent_operation_id="d" * 32,
        start_payload={
            "tool_name": "read",
            "tool_kind": tool_kind,
            "parent_model_operation_id": "d" * 32,
            "call_item_id": call_item_id,
            **({"tool_context": tool_context} if tool_context is not None else {}),
        },
        finish_payload={
            "result_item_id": result_item_id,
            **({"result_carrier_item_id": result_carrier_item_id} if result_carrier_item_id is not None else {}),
        },
        links=_caused_by("e" * 32),
    )
    revision = log.add(
        "context.revision.recorded",
        10 * _NS,
        operation_id=revision_id,
        parent_operation_id=parent_operation_id,
        payload={"revision_id": revision_id, "is_checkpoint": True, **revision_payload},
        segmented_fields=(SegmentedField(field_pointer="/payload/refs", segment_group_id=group_id, segment_count=1),),
    )
    log.add(
        "event.segment",
        10 * _NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": group_id,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": call_item_id, "occurrence": 0, "position": 0, "action": "add"},
                {
                    "item_id": result_membership_item_id or result_item_id,
                    "occurrence": 0,
                    "position": 1,
                    "action": "add",
                },
            ],
        },
    )
    if duplicate_revision:
        log.add(
            "context.revision.recorded",
            10 * _NS,
            operation_id=revision_id,
            parent_operation_id=parent_operation_id,
            payload={"revision_id": revision_id, "is_checkpoint": True, **revision_payload},
        )
    log.span(
        "model.cycle",
        "8" * 32,
        10 * _NS,
        20 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": consuming_exchange_id},
    )
    log.span(
        "model.exchange",
        consuming_exchange_id,
        10 * _NS,
        20 * _NS,
        parent_operation_id="8" * 32,
        start_payload={"context_revision_id": revision_id},
    )
    log.add("turn.finished", 20 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / file_name
    log.write(path)
    return path


def test_duration_mismatch_threshold_is_diagnostic_only(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("wait.started", 0, operation_id="a" * 32, payload={"category": "user_input"})
    log.add(
        "wait.finished",
        10 * _NS,
        operation_id="a" * 32,
        payload={"outcome": "completed", "duration_ms": 1},
        measurements={"/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert analysis.diagnostics.span_duration_mismatch_count == 1
    assert analysis.diagnostics.span_duration_mismatches[0].family == "wait"
    assert analysis.diagnostics.span_duration_mismatches[0].operation_id == "a" * 32
    assert analysis.turns[0].wall_time_ns[WallBucket.WAIT].value == 10 * _NS


def test_containment_diagnostics_retain_family_and_operation_callsite(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("model.run", "a" * 32, 2 * _NS, 8 * _NS)
    log.span(
        "wait",
        "b" * 32,
        _NS,
        9 * _NS,
        parent_operation_id="a" * 32,
        start_payload={"category": "user_input"},
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "containment-detail.jsonl"
    log.write(path)

    diagnostic = analyze_trajectory(path).diagnostics.containment_violations[0]

    assert diagnostic.family == "wait"
    assert diagnostic.operation_id == "b" * 32
    assert diagnostic.parent_family == "model.run"
    assert diagnostic.parent_operation_id == "a" * 32


def _exchange_usage(
    log: EventLog,
    operation_id: str,
    start_ns: int,
    end_ns: int,
    normalized: dict[str, int],
    *,
    provider_buckets: tuple[str, ...] | None = None,
    parent_operation_id: str | None = None,
    start_payload: dict[str, object] | None = None,
    actor: Actor = SYSTEM_ACTOR,
) -> None:
    buckets = tuple(normalized) if provider_buckets is None else provider_buckets
    log.add(
        "model.exchange.started",
        start_ns,
        operation_id=operation_id,
        parent_operation_id=parent_operation_id,
        payload=start_payload,
        actor=actor,
    )
    log.add(
        "model.exchange.finished",
        end_ns,
        operation_id=operation_id,
        parent_operation_id=parent_operation_id,
        payload={
            "outcome": "success",
            "duration_ms": max(1, (end_ns - start_ns) // 1_000_000),
            "usage": {"normalized": normalized},
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            **{
                f"/payload/usage/normalized/{bucket}": {"source": "provider", "adapter_version": 1}
                for bucket in buckets
            },
        },
        actor=actor,
    )


def test_token_usage_splits_buckets_and_marks_unreported_optional_buckets(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    _exchange_usage(
        log,
        "a" * 32,
        0,
        _NS,
        {"input_total": 100, "output_total": 20, "reasoning": 7, "cache_read": 60},
    )
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)
    token_usage = analysis.token_usage

    assert token_usage is not None
    buckets = token_usage.buckets
    assert (buckets[UsageBucket.INPUT].value, buckets[UsageBucket.INPUT].precision) == (100, Precision.EXACT)
    assert (buckets[UsageBucket.OUTPUT].value, buckets[UsageBucket.OUTPUT].precision) == (20, Precision.EXACT)
    assert (buckets[UsageBucket.REASONING].value, buckets[UsageBucket.REASONING].precision) == (7, Precision.EXACT)
    assert (buckets[UsageBucket.CACHE_READ].value, buckets[UsageBucket.CACHE_READ].precision) == (60, Precision.EXACT)
    assert buckets[UsageBucket.CACHE_CREATION].value is None
    assert buckets[UsageBucket.CACHE_CREATION].precision is Precision.MISSING
    assert analysis.turns[0].token_usage is not None
    assert analysis.turns[0].token_usage.buckets == buckets


def test_token_usage_partial_optional_reporting_is_estimated_and_bad_provenance_unresolved(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    _exchange_usage(
        log,
        "a" * 32,
        0,
        _NS,
        {"input_total": 100, "output_total": 20, "reasoning": 7, "cache_creation": 9},
        provider_buckets=("input_total", "output_total", "reasoning"),
    )
    _exchange_usage(
        log,
        "b" * 32,
        _NS,
        2 * _NS,
        {"input_total": 50, "output_total": 10, "cache_creation": 4},
        provider_buckets=("input_total", "output_total", "cache_creation"),
    )
    log.add("turn.finished", 2 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    token_usage = analyze_trajectory(path).token_usage

    assert token_usage is not None
    buckets = token_usage.buckets
    assert (buckets[UsageBucket.INPUT].value, buckets[UsageBucket.INPUT].precision) == (150, Precision.EXACT)
    assert (buckets[UsageBucket.REASONING].value, buckets[UsageBucket.REASONING].precision) == (
        7,
        Precision.ESTIMATED,
    )
    assert buckets[UsageBucket.CACHE_CREATION].value == 13
    assert buckets[UsageBucket.CACHE_CREATION].precision is Precision.UNRESOLVED
    assert buckets[UsageBucket.CACHE_CREATION].reason == "one or more selected turns are unresolved"


def test_reasoning_tokens_above_output_are_unresolved(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    _exchange_usage(log, "a" * 32, 0, _NS, {"input_total": 100, "output_total": 20, "reasoning": 21})
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    reasoning = analyze_trajectory(path).token_usage

    assert reasoning is not None
    assert reasoning.buckets[UsageBucket.REASONING].value == 21
    assert reasoning.buckets[UsageBucket.REASONING].precision is Precision.UNRESOLVED


def test_tool_usage_panels_group_skill_and_mcp_actions(tmp_path) -> None:
    skill_call = "7" * 32
    path = tmp_path / ".chrys" / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    for operation, second, tool_name, tool_kind, call_item_id in (
        ("a", 1, "load_skill", "skill", skill_call),
        ("b", 2, "run_skill_script", "skill", "8" * 32),
        ("c", 3, "figma.render", "mcp", "9" * 32),
        ("d", 4, "figma.render", "mcp", "0" * 32),
        ("e", 5, "jira.search", "mcp", "1" * 32),
    ):
        log.span(
            "tool.operation",
            operation * 32,
            second * _NS,
            (second + 1) * _NS,
            start_payload={
                "tool_name": tool_name,
                "tool_kind": tool_kind,
                "call_item_id": call_item_id,
                "argument_fingerprint": operation,
            },
            finish_payload={"outcome": "success"},
        )
    log.add("turn.finished", 7 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)
    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [
                        {
                            "contents": [
                                {
                                    "type": "function_call",
                                    "arguments": json.dumps({"skill_name": "review-deck"}),
                                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: skill_call},
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_trajectory(path)

    assert analysis.skill_usage is not None
    assert analysis.skill_usage.total == 2
    assert [(row.name, row.count) for row in analysis.skill_usage.rows] == [("review-deck", 1)]
    assert analysis.skill_usage.unattributed == 1
    assert analysis.skill_usage.precision is Precision.EXACT
    assert analysis.mcp_usage is not None
    assert analysis.mcp_usage.total == 3
    assert [(row.name, row.count) for row in analysis.mcp_usage.rows] == [("figma.render", 2), ("jira.search", 1)]
    assert analysis.mcp_usage.unattributed == 0


def test_insights_retain_structured_mcp_payload_wait_and_skill_metrics(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    mcp_id = "a" * 32
    log.add(
        "approval.requested",
        0,
        operation_id="b" * 32,
        payload={"approval_request_id": "b" * 32, "target_tool_operation_id": mcp_id},
    )
    log.add(
        "approval.resolved",
        _NS,
        operation_id="b" * 32,
        payload={"approval_request_id": "b" * 32, "target_tool_operation_id": mcp_id, "wait_ms": 1000},
        measurements={"/payload/wait_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add(
        "wait.started",
        0,
        operation_id="c" * 32,
        payload={"category": "mcp_connect", "server_name": "figma", "target_operation_id": mcp_id},
    )
    log.add(
        "wait.finished",
        _NS,
        operation_id="c" * 32,
        payload={
            "category": "mcp_connect",
            "server_name": "figma",
            "target_operation_id": mcp_id,
            "duration_ms": 1000,
        },
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add(
        "tool.operation.started",
        _NS,
        operation_id=mcp_id,
        payload={
            "tool_name": "figma_render",
            "tool_kind": "mcp",
            "tool_context": {"server_name": "figma", "remote_name": "render"},
        },
    )
    log.add(
        "tool.payload.observed",
        2 * _NS,
        operation_id=mcp_id,
        payload={
            "model_visible_bytes": 4096,
            "local_token_estimate": 100,
            "original_bytes": 8000,
            "truncated": True,
            "artifact_id": "artifact-1",
        },
    )
    log.add(
        "tool.operation.finished",
        3 * _NS,
        operation_id=mcp_id,
        payload={"outcome": "success", "duration_ms": 2000},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    load_id = "d" * 32
    log.add(
        "tool.operation.started",
        4 * _NS,
        operation_id=load_id,
        payload={
            "tool_name": "load_skill",
            "tool_kind": "skill",
            "tool_context": {"skill_name": "slides", "skill_revision": "rev-a"},
        },
    )
    log.add(
        "tool.payload.observed",
        5 * _NS,
        operation_id=load_id,
        payload={"model_visible_bytes": 1000, "local_token_estimate": 250, "truncated": False},
    )
    log.add(
        "tool.operation.finished",
        5 * _NS,
        operation_id=load_id,
        payload={"outcome": "success", "duration_ms": 1000},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    script_id = "e" * 32
    log.add(
        "tool.operation.started",
        6 * _NS,
        operation_id=script_id,
        payload={
            "tool_name": "run_skill_script",
            "tool_kind": "skill",
            "tool_context": {
                "skill_name": "slides",
                "skill_revision": "rev-b",
                "script_name": "scripts/render.py",
            },
        },
    )
    log.add(
        "tool.operation.finished",
        7 * _NS,
        operation_id=script_id,
        payload={"outcome": "failed", "duration_ms": 1000, "exit_code": 7},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add("turn.finished", 8 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)
    assert analysis.insights is not None
    mcp = analysis.insights.mcp.rows[0]
    assert (mcp.server_name, mcp.calls) == ("figma", 1)
    assert mcp.remotes[0].remote_name == "render"
    assert (mcp.result_bytes.value, mcp.result_bytes.precision) == (4096, Precision.EXACT)
    assert (mcp.result_tokens.value, mcp.result_tokens.precision) == (100, Precision.ESTIMATED)
    assert (mcp.truncated_count.value, mcp.spill_count.value) == (1, 1)
    assert (mcp.connection_wait_count.value, mcp.connection_wait_ns.value) == (1, _NS)
    assert mcp.approval_blocking_share.value == pytest.approx(1 / 2)
    skill = analysis.insights.skills.rows[0]
    assert (skill.skill_name, skill.load_count, skill.script_count, skill.turn_count) == ("slides", 1, 1, 1)
    assert (skill.first_action_median_ns.value, skill.first_action_median_ns.precision) == (_NS, Precision.EXACT)
    assert (skill.injected_tokens.value, skill.injected_tokens.precision) == (250, Precision.ESTIMATED)
    assert skill.revisions == ("rev-a", "rev-b")
    assert [(row.name, row.count) for row in skill.script_exit_codes] == [("7", 1)]


def test_cut_approval_and_mcp_wait_with_active_owner_remain_unresolved_samples(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    retained_turn_id = "4" * 32
    removed_turn_id = "5" * 32
    tool_id = "a" * 32
    approval_id = "b" * 32
    wait_id = "c" * 32
    log.add("turn.started", 0, turn_id=retained_turn_id, payload={"turn_number": 1})
    log.add(
        "tool.operation.started",
        _NS,
        turn_id=retained_turn_id,
        operation_id=tool_id,
        payload={"tool_name": "figma_render", "tool_kind": "mcp"},
    )
    log.add(
        "approval.requested",
        2 * _NS,
        turn_id=retained_turn_id,
        operation_id=approval_id,
        payload={"approval_request_id": approval_id, "target_tool_operation_id": tool_id},
    )
    log.add(
        "wait.started",
        3 * _NS,
        turn_id=retained_turn_id,
        operation_id=wait_id,
        payload={"category": "mcp_connect", "server_name": "figma", "target_operation_id": tool_id},
    )
    log.add(
        "turn.finished",
        4 * _NS,
        turn_id=retained_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 4000},
    )
    superseded_from = log.next_sequence
    log.add("turn.started", 5 * _NS, turn_id=removed_turn_id, payload={"turn_number": 2})
    log.add(
        "approval.resolved",
        6 * _NS,
        turn_id=retained_turn_id,
        operation_id=approval_id,
        payload={"approval_request_id": approval_id, "target_tool_operation_id": tool_id, "wait_ms": 4000},
        measurements={"/payload/wait_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add(
        "wait.finished",
        7 * _NS,
        turn_id=retained_turn_id,
        operation_id=wait_id,
        payload={
            "category": "mcp_connect",
            "server_name": "figma",
            "target_operation_id": tool_id,
            "duration_ms": 4000,
        },
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add(
        "turn.finished",
        8 * _NS,
        turn_id=removed_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 3000},
    )
    _append_resolved_rollback(log, superseded_from, 9 * _NS, target_turn_id=retained_turn_id)
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    analyzer.load(path)
    intermediate = analyzer._intermediate
    assert intermediate is not None
    inactive_ranges = facts_module._closed_sequence_union(intermediate.rollback_ranges)

    assert aggregation_module._approval_durations_by_tool(
        intermediate,
        inactive_ranges,
        cancel_event=None,
    ) == {tool_id: (None,)}
    waits, unattributed = aggregation_module._mcp_connection_waits(
        intermediate,
        inactive_ranges,
        cancel_event=None,
    )
    assert waits == {"figma": (None,)}
    assert unattributed == 0


def test_cut_approval_and_mcp_wait_with_exclusively_inactive_owner_are_dropped(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    removed_turn_id = "5" * 32
    tool_id = "a" * 32
    approval_id = "b" * 32
    wait_id = "c" * 32
    superseded_from = log.next_sequence
    log.add("turn.started", 0, turn_id=removed_turn_id, payload={"turn_number": 1})
    log.add(
        "tool.operation.started",
        _NS,
        turn_id=removed_turn_id,
        operation_id=tool_id,
        payload={"tool_name": "figma_render", "tool_kind": "mcp"},
    )
    log.add(
        "approval.requested",
        2 * _NS,
        turn_id=removed_turn_id,
        operation_id=approval_id,
        payload={"approval_request_id": approval_id, "target_tool_operation_id": tool_id},
    )
    log.add(
        "wait.started",
        3 * _NS,
        turn_id=removed_turn_id,
        operation_id=wait_id,
        payload={"category": "mcp_connect", "server_name": "figma", "target_operation_id": tool_id},
    )
    log.add(
        "turn.finished",
        4 * _NS,
        turn_id=removed_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 4000},
    )
    _append_resolved_rollback(log, superseded_from, 5 * _NS)
    new_branch = "6" * 32
    log.add(
        "approval.resolved",
        6 * _NS,
        turn_id=removed_turn_id,
        operation_id=approval_id,
        branch_id=new_branch,
        payload={"approval_request_id": approval_id, "target_tool_operation_id": tool_id, "wait_ms": 4000},
        measurements={"/payload/wait_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add(
        "wait.finished",
        7 * _NS,
        turn_id=removed_turn_id,
        operation_id=wait_id,
        branch_id=new_branch,
        payload={
            "category": "mcp_connect",
            "server_name": "figma",
            "target_operation_id": tool_id,
            "duration_ms": 4000,
        },
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    analyzer.load(path)
    intermediate = analyzer._intermediate
    assert intermediate is not None
    inactive_ranges = facts_module._closed_sequence_union(intermediate.rollback_ranges)

    assert (
        aggregation_module._approval_durations_by_tool(
            intermediate,
            inactive_ranges,
            cancel_event=None,
        )
        == {}
    )
    waits, unattributed = aggregation_module._mcp_connection_waits(
        intermediate,
        inactive_ranges,
        cancel_event=None,
    )
    assert waits == {}
    assert unattributed == 0


def test_skill_insights_count_retry_attempts_as_one_logical_turn(tmp_path: Path) -> None:
    first_turn_id = "4" * 32
    retry_turn_id = "5" * 32
    log = EventLog()
    log.coverage()
    for index, (turn_id, is_retry) in enumerate(((first_turn_id, False), (retry_turn_id, True))):
        start_ns = index * 10 * _NS
        log.add(
            EventType.TURN_STARTED,
            start_ns,
            turn_id=turn_id,
            payload={"turn_number": 1, "is_retry": is_retry},
        )
        log.span(
            "tool.operation",
            str(index + 6) * 32,
            start_ns,
            start_ns + _NS,
            turn_id=turn_id,
            start_payload={
                "tool_name": "load_skill",
                "tool_kind": "skill",
                "tool_context": {"skill_name": "slides", "skill_revision": "rev-a"},
            },
        )
        log.add(
            EventType.TURN_FINISHED,
            start_ns + 2 * _NS,
            turn_id=turn_id,
            payload={"end_reason": "cancelled", "duration_ms": 2_000},
        )
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert len(analysis.turns) == 1
    assert len(analysis.turns[0].attempts) == 2
    assert analysis.insights is not None
    skill = analysis.insights.skills.rows[0]
    assert (skill.load_count, skill.turn_count) == (2, 1)


def test_negative_tool_payload_counts_are_not_reported_as_exact(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    operation_id = "a" * 32
    log.add(
        "tool.operation.started",
        0,
        operation_id=operation_id,
        payload={
            "tool_name": "figma_render",
            "tool_kind": "mcp",
            "tool_context": {"server_name": "figma", "remote_name": "render"},
        },
    )
    log.add(
        "tool.payload.observed",
        _NS,
        operation_id=operation_id,
        payload={"model_visible_bytes": -1, "local_token_estimate": -2, "original_bytes": -3},
    )
    log.add(
        "tool.operation.finished",
        2 * _NS,
        operation_id=operation_id,
        payload={"outcome": "success", "duration_ms": 2000},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add("turn.finished", 2 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert analysis.insights is not None
    mcp = analysis.insights.mcp.rows[0]
    assert (mcp.result_bytes.value, mcp.result_bytes.precision) == (None, Precision.MISSING)
    assert (mcp.result_tokens.value, mcp.result_tokens.precision) == (None, Precision.MISSING)


def test_skill_action_starting_before_the_load_terminal_is_not_an_exact_latency(tmp_path) -> None:
    """A related action that began before the load's terminal landed cannot
    prove it used the loaded skill, even when the boundary timestamps meet."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    load_id = "d" * 32
    script_id = "e" * 32
    log.add(
        "tool.operation.started",
        _NS,
        operation_id=load_id,
        payload={
            "tool_name": "load_skill",
            "tool_kind": "skill",
            "tool_context": {"skill_name": "slides", "skill_revision": "rev-a"},
        },
    )
    log.add(
        "tool.operation.started",
        2 * _NS,
        operation_id=script_id,
        payload={
            "tool_name": "run_skill_script",
            "tool_kind": "skill",
            "tool_context": {"skill_name": "slides", "skill_revision": "rev-a", "script_name": "scripts/render.py"},
        },
    )
    log.add(
        "tool.operation.finished",
        2 * _NS,
        operation_id=load_id,
        payload={"outcome": "success", "duration_ms": 1000},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add(
        "tool.operation.finished",
        3 * _NS,
        operation_id=script_id,
        payload={"outcome": "success", "duration_ms": 1000},
        measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
    )
    log.add("turn.finished", 4 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert analysis.insights is not None
    skill = analysis.insights.skills.rows[0]
    assert skill.first_action_median_ns == Metric(
        None, Precision.UNRESOLVED, "load-to-action latency endpoints are unresolved"
    )


def test_malformed_tool_payload_truncated_flag_is_not_reported_as_exact(tmp_path) -> None:
    """A malformed boolean remains unknown on an otherwise valid log line."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    for index, payload in enumerate(
        (
            {"model_visible_bytes": 10, "truncated": True, "artifact_id": "artifact-1"},
            {"model_visible_bytes": 10, "truncated": "true", "artifact_id": "artifact-2"},
        )
    ):
        operation_id = f"{index + 10:032x}"
        log.add(
            "tool.operation.started",
            index * 2 * _NS,
            operation_id=operation_id,
            payload={
                "tool_name": "figma_render",
                "tool_kind": "mcp",
                "tool_context": {"server_name": "figma", "remote_name": "render"},
            },
        )
        log.add("tool.payload.observed", (index * 2 + 1) * _NS, operation_id=operation_id, payload=payload)
        log.add(
            "tool.operation.finished",
            (index * 2 + 2) * _NS,
            operation_id=operation_id,
            payload={"outcome": "success", "duration_ms": 2000},
            measurements={"/payload/duration_ms": measurement("monotonic_clock", method_version=1)},
        )
    log.add("turn.finished", 4 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)

    assert analysis.diagnostics is not None
    assert analysis.diagnostics.corrupt_line_count == 0
    assert analysis.diagnostics.accounted_prefix_violations == ()
    assert analysis.insights is not None
    mcp = analysis.insights.mcp.rows[0]
    assert mcp.truncated_count == Metric(
        1,
        Precision.ESTIMATED,
        "one or more tool payload observations are missing",
    )
    assert mcp.spill_count == Metric(2, Precision.EXACT)


def test_spilled_unknown_flag_remains_defensive_unknown_evidence() -> None:
    """Envelope validation makes this state unreachable from persisted logs."""
    payload = facts_module._ToolPayloadExtras(
        sequence=1,
        scope=facts_module._EventScope(
            runtime_id="runtime",
            branch_id="branch",
            coverage_id="coverage",
            actor_id=None,
            turn_id=None,
        ),
        model_visible_bytes=None,
        local_token_estimate=None,
        original_bytes=None,
        flags=facts_module._TOOL_PAYLOAD_SPILLED_UNKNOWN_BIT,
    )

    assert payload.spilled is None


def test_missing_or_unknown_tool_identity_uses_only_frozen_fallback_buckets(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span(
        "tool.operation",
        "a" * 32,
        0,
        _NS,
        start_payload={"tool_name": "figma_render", "tool_kind": "mcp"},
    )
    log.span(
        "tool.operation",
        "b" * 32,
        _NS,
        2 * _NS,
        start_payload={"tool_name": "load_skill", "tool_kind": "future.kind"},
    )
    log.add("turn.finished", 2 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)
    assert analysis.insights is not None
    assert analysis.insights.mcp.total == 1
    assert analysis.insights.mcp.unattributed == 1
    assert analysis.insights.mcp.rows == ()
    assert analysis.insights.mcp.precision is Precision.EXACT
    assert analysis.insights.skills.total == 0
    assert analysis.insights.tools.unclassified == 1
    assert any(
        row.tool_kind == "unclassified" and row.tool_name == "load_skill" for row in analysis.insights.tools.rows
    )
    assert analysis.turns[0].action_projection_precision is Precision.EXACT


def test_tool_action_start_beyond_the_turn_terminal_degrades_action_projection(tmp_path) -> None:
    """The scope check bounds actions from below; the terminal bounds them from above."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("tool.operation", "a" * 32, 0, _NS, start_payload={"tool_name": "zsh", "tool_kind": "shell"})
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.span("tool.operation", "b" * 32, 2 * _NS, 3 * _NS, start_payload={"tool_name": "zsh", "tool_kind": "shell"})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.action_projection_precision is not Precision.EXACT
    assert turn.action_projection_reason is not None
    assert "tool action endpoint lies beyond the turn terminal" in turn.action_projection_reason


def test_tool_action_between_turn_finish_and_response_fence_degrades_action_projection(tmp_path) -> None:
    """The bound is ``turn.finished`` itself, not the later response fence."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("tool.operation", "a" * 32, 0, _NS, start_payload={"tool_name": "zsh", "tool_kind": "shell"})
    log.add("turn.finished", 2 * _NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.span("tool.operation", "b" * 32, 3 * _NS, 4 * _NS, start_payload={"tool_name": "zsh", "tool_kind": "shell"})
    log.settled(5 * _NS)
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)
    turn = analysis.turns[0]

    assert turn.action_projection_precision is not Precision.EXACT
    assert turn.action_projection_reason is not None
    assert "tool action endpoint lies beyond the turn terminal" in turn.action_projection_reason
    stray = next(action for action in analysis.actions if action.operation_id == "b" * 32)
    assert stray.outcome is None
    assert stray.outcome_precision is Precision.UNRESOLVED


def test_tool_terminal_beyond_the_turn_finish_degrades_action_projection(tmp_path) -> None:
    """An outcome read from a finish beyond the turn is as unfounded as a stray start."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(
        "tool.operation.started",
        _NS,
        operation_id="a" * 32,
        payload={"tool_name": "zsh", "tool_kind": "shell"},
    )
    log.add("turn.finished", 2 * _NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.add(
        "tool.operation.finished",
        3 * _NS,
        operation_id="a" * 32,
        payload={"outcome": "success", "duration_ms": 2000},
    )
    log.settled(4 * _NS)
    path = tmp_path / "events.jsonl"
    log.write(path)

    analysis = analyze_trajectory(path)
    turn = analysis.turns[0]

    assert turn.action_projection_precision is not Precision.EXACT
    assert turn.action_projection_reason is not None
    assert "tool action endpoint lies beyond the turn terminal" in turn.action_projection_reason
    (action,) = analysis.actions
    assert action.outcome is None
    assert action.outcome_precision is Precision.UNRESOLVED


def test_tool_action_within_a_completed_turn_stays_exact_with_a_response_fence(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("tool.operation", "a" * 32, 0, _NS, start_payload={"tool_name": "zsh", "tool_kind": "shell"})
    log.add("turn.finished", 2 * _NS, payload={"end_reason": "completed", "duration_ms": 0})
    log.settled(3 * _NS)
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]

    assert turn.action_projection_precision is Precision.EXACT
    assert turn.action_projection_reason is None


def test_refresh_surfaces_a_duplicate_turn_start_on_an_already_cached_turn(tmp_path) -> None:
    """A duplicate lifecycle start must evict the cached analysis of its turn."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    assert analyzer.load(path).turns[0].turn_number == 1

    log.add("turn.started", 2 * _NS, payload={"turn_number": 1})
    log.write(path)

    refreshed = analyzer.refresh().turns[0]
    fresh = TrajectoryAnalyzer().load(path).turns[0]

    assert refreshed.turn_number == 1
    assert refreshed.elapsed_ns.precision is Precision.UNRESOLVED
    assert refreshed.elapsed_ns == fresh.elapsed_ns


def test_server_cp_group_is_exact_only_for_exact_response_turns(tmp_path) -> None:
    exact_path = _write_fan_in_membership_probe(
        tmp_path,
        {"item_count": 2, "unidentified_item_count": 0},
        file_name="exact-server-cp.jsonl",
        tool_kind="mcp",
        tool_context={"server_name": "figma", "remote_name": "read"},
    )
    exact = analyze_trajectory(exact_path)
    assert exact.insights is not None
    assert exact.turns[0].response_cp_ns.precision is Precision.EXACT
    assert exact.insights.mcp.rows[0].critical_path_exclusive_ns.precision is Precision.EXACT
    assert exact.insights.mcp.rows[0].critical_path_exclusive_ns.value == 8 * _NS

    unresolved_path = _write_fan_in_membership_probe(
        tmp_path,
        {"item_count": 1, "unidentified_item_count": 0},
        file_name="unresolved-server-cp.jsonl",
        tool_kind="mcp",
        tool_context={"server_name": "figma", "remote_name": "read"},
    )
    unresolved = analyze_trajectory(unresolved_path)
    assert unresolved.insights is not None
    assert unresolved.turns[0].response_cp_ns.precision is Precision.UNRESOLVED
    cp_metric = unresolved.insights.mcp.rows[0].critical_path_exclusive_ns
    assert cp_metric.value is None
    assert cp_metric.precision is Precision.UNRESOLVED


def test_context_carrying_load_is_surfaced_without_changing_finding_identity(tmp_path) -> None:
    item_id = "7" * 32
    revision_id = "8" * 32
    segment_id = "9" * 32
    exchange_id = "a" * 32
    side_revision_id = "b" * 32
    side_segment_id = "c" * 32
    side_exchange_id = "d" * 32
    side_actor = Actor(kind="side_call", role="title_gen", actor_id="0" * 32)
    path = tmp_path / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    revision = log.add(
        "context.revision.recorded",
        _NS,
        operation_id=revision_id,
        parent_operation_id=exchange_id,
        payload={
            "revision_id": revision_id,
            "is_checkpoint": True,
            "item_count": 1,
            "untokenized_item_count": 0,
            "unidentified_item_count": 0,
        },
        segmented_fields=(SegmentedField(field_pointer="/payload/refs", segment_group_id=segment_id, segment_count=1),),
    )
    log.add(
        "event.segment",
        _NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": segment_id,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [{"item_id": item_id, "occurrence": 0, "position": 0, "action": "add"}],
        },
    )
    log.span(
        "model.exchange",
        exchange_id,
        2 * _NS,
        3 * _NS,
        start_payload={"context_revision_id": revision_id},
    )
    side_revision = log.add(
        "context.revision.recorded",
        4 * _NS,
        operation_id=side_revision_id,
        parent_operation_id=side_exchange_id,
        actor=side_actor,
        payload={
            "revision_id": side_revision_id,
            "is_checkpoint": True,
            "item_count": 1,
            "untokenized_item_count": 0,
            "unidentified_item_count": 0,
        },
        segmented_fields=(
            SegmentedField(field_pointer="/payload/refs", segment_group_id=side_segment_id, segment_count=1),
        ),
    )
    log.add(
        "event.segment",
        4 * _NS,
        operation_id=None,
        actor=side_actor,
        payload={
            "parent_event_id": side_revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": side_segment_id,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [{"item_id": item_id, "occurrence": 0, "position": 0, "action": "add"}],
        },
    )
    log.span(
        "model.exchange",
        side_exchange_id,
        5 * _NS,
        6 * _NS,
        actor=side_actor,
        start_payload={"context_revision_id": side_revision_id},
    )
    log.add("turn.finished", 6 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)
    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [
                        {
                            "role": "assistant",
                            "additional_properties": {
                                ANALYTICS_ITEM_ID_KEY: item_id,
                                "_group": {"token_count": 10},
                            },
                            "contents": [
                                {"type": "text_reasoning", "text": "…"},
                                {"type": "function_call", "call_id": "call-1", "name": "zsh", "arguments": "{}"},
                                {"type": "function_call", "call_id": "call-2", "name": "zsh", "arguments": "{}"},
                            ],
                        },
                        {
                            "role": "tool",
                            "additional_properties": {
                                ANALYTICS_ITEM_ID_KEY: "6" * 32,
                                "_group": {"token_count": 4},
                            },
                            "contents": [{"type": "function_result", "call_id": "call-1", "result": "ok"}],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_trajectory(path)
    assert analysis.insights is not None
    carrying = analysis.insights.context_carrying_load
    assert len(carrying) == 1
    assert (carrying[0].item_id, carrying[0].load, carrying[0].turn_number) == (item_id, 10, 1)
    assert (carrying[0].token_count, carrying[0].carry_count, carrying[0].origin_turn_number) == (10, 1, 1)
    assert (carrying[0].role, carrying[0].tool_names) == ("assistant", ("zsh", "zsh"))
    projection = session_projection_module._read_session_projection(path)
    assert projection.item_roles["6" * 32] == "tool"
    assert projection.item_tool_names["6" * 32] == ("zsh",)
    finding = next(row for row in analysis.findings if row.rule_id == "context-carrying-load")
    assert finding.evidence_key == evidence_key("context-carrying-load", 1, (f"item:{item_id}",))


def _twin_projection_path(tmp_path):
    path = tmp_path / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    return path


def test_twin_messages_in_block_and_live_list_keep_single_tool_names_and_roles(tmp_path) -> None:
    """The session store may hold one message both in a compressed block and
    in the live list; the projection must not double its tool names."""
    path = _twin_projection_path(tmp_path)
    call_message = {
        "role": "assistant",
        "additional_properties": {ANALYTICS_ITEM_ID_KEY: "7" * 32},
        "contents": [
            {"type": "function_call", "call_id": "call-1", "name": "zsh", "arguments": "{}"},
            {"type": "function_call", "call_id": "call-2", "name": "zsh", "arguments": "{}"},
        ],
    }
    result_message = {
        "role": "tool",
        "additional_properties": {ANALYTICS_ITEM_ID_KEY: "6" * 32},
        "contents": [{"type": "function_result", "call_id": "call-1", "result": "ok"}],
    }
    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [call_message, result_message],
                    "compressed_msgs": [
                        {
                            "compressed_context_id": "ctx-1",
                            "summary_text": "summary",
                            "turn_range": [1, 1],
                            "messages": [call_message, result_message],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    projection = session_projection_module._read_session_projection(path)

    # Genuine duplicate calls inside one message survive; the twin copies of
    # that message across transcripts collapse to one.
    assert projection.item_tool_names["7" * 32] == ("zsh", "zsh")
    assert projection.item_tool_names["6" * 32] == ("zsh",)
    assert projection.item_roles["7" * 32] == "assistant"
    assert projection.item_roles["6" * 32] == "tool"


def test_conflicting_carrier_tool_names_or_roles_across_transcripts_are_dropped(tmp_path) -> None:
    path = _twin_projection_path(tmp_path)

    def message(carrier: str, *, role: str, name: str | None) -> dict:
        contents = [] if name is None else [{"type": "function_call", "call_id": "c", "name": name, "arguments": "{}"}]
        return {"role": role, "additional_properties": {ANALYTICS_ITEM_ID_KEY: carrier}, "contents": contents}

    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [
                        message("7" * 32, role="assistant", name="python"),
                        message("6" * 32, role="tool", name="zsh"),
                        message("5" * 32, role="assistant", name=None),
                    ],
                    "compressed_msgs": [
                        {
                            "compressed_context_id": "ctx-1",
                            "summary_text": "summary",
                            "turn_range": [1, 1],
                            "messages": [
                                message("7" * 32, role="assistant", name="zsh"),
                                message("6" * 32, role="assistant", name="zsh"),
                                message("5" * 32, role="assistant", name="zsh"),
                            ],
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    projection = session_projection_module._read_session_projection(path)

    # Same carrier with different names, roles, or call counts across
    # transcripts is not the same message; every conflicting join drops.
    assert "7" * 32 not in projection.item_tool_names
    assert "5" * 32 not in projection.item_tool_names
    assert "6" * 32 not in projection.item_roles
    assert projection.item_tool_names["6" * 32] == ("zsh",)
    assert projection.item_roles["7" * 32] == "assistant"


def _operation_index(turn, family: str, operation_id: str) -> int:
    for index, operation in enumerate(turn.operations):
        if operation.family == family and operation.operation_id == operation_id:
            return index
    raise AssertionError(f"{family}:{operation_id} has no operations row")


def test_turn_flow_types_parent_and_causal_edges_against_operations(tmp_path) -> None:
    """Flow edges are index pairs into operations, split by displacing-vs-pointer proof."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span("preparation", "a" * 32, 0, 0, start_payload={"scope": "turn_preamble", "phase": "dispatch"})
    log.span("model.run", "b" * 32, 0, 10 * _NS, links=_caused_by("a" * 32))
    log.span(
        "model.cycle",
        "1" * 32,
        0,
        0,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "2" * 32},
    )
    log.span("model.exchange", "2" * 32, 0, 0, parent_operation_id="1" * 32)
    item_ids = (("7" * 32, "9" * 32), ("8" * 32, "0" * 32))
    for prefix, (call_item_id, result_item_id) in zip(("c", "d"), item_ids, strict=True):
        tool_id = prefix * 32
        preamble_id = ("e" if prefix == "c" else "f") * 32
        log.span(
            "preparation",
            preamble_id,
            0,
            0,
            parent_operation_id="2" * 32,
            start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": tool_id},
        )
        log.span(
            "tool.operation",
            tool_id,
            0,
            10 * _NS,
            parent_operation_id="2" * 32,
            start_payload={
                "tool_name": prefix,
                "tool_kind": "filesystem.read",
                "batch_index": 0,
                "parent_model_operation_id": "2" * 32,
                "call_item_id": call_item_id,
            },
            finish_payload={"result_item_id": result_item_id},
            links=_caused_by(preamble_id),
        )
    revision = log.add(
        "context.revision.recorded",
        10 * _NS,
        operation_id="5" * 32,
        parent_operation_id="4" * 32,
        payload={"revision_id": "5" * 32, "is_checkpoint": True, "item_count": 4, "unidentified_item_count": 0},
        segmented_fields=(SegmentedField(field_pointer="/payload/refs", segment_group_id="6" * 32, segment_count=1),),
    )
    log.add(
        "event.segment",
        10 * _NS,
        operation_id=None,
        payload={
            "parent_event_id": revision.event_id,
            "field_pointer": "/payload/refs",
            "segment_group_id": "6" * 32,
            "segment_index": 0,
            "segment_count": 1,
            "encoding": "array_slice",
            "entries": [
                {"item_id": item_id, "occurrence": 0, "position": index, "action": "add"}
                for index, item_id in enumerate(item_id for pair in item_ids for item_id in pair)
            ],
        },
    )
    log.span(
        "model.cycle",
        "3" * 32,
        10 * _NS,
        10 * _NS,
        parent_operation_id="b" * 32,
        finish_payload={"final_exchange_operation_id": "4" * 32},
    )
    log.span(
        "model.exchange",
        "4" * 32,
        10 * _NS,
        10 * _NS,
        parent_operation_id="3" * 32,
        start_payload={"context_revision_id": "5" * 32},
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]
    flow = turn.flow

    assert flow is not None
    assert flow.acyclic
    assert flow.has_terminal
    assert flow.root_index == _operation_index(turn, "preparation", "a" * 32)
    parent_edges = set(flow.parent_edges())
    causal_edges = set(flow.causal_edges())
    run_to_cycle = (
        _operation_index(turn, "model.run", "b" * 32),
        _operation_index(turn, "model.cycle", "1" * 32),
    )
    exchange_to_tool = (
        _operation_index(turn, "model.exchange", "2" * 32),
        _operation_index(turn, "tool.operation", "c" * 32),
    )
    preamble_to_tool = (
        _operation_index(turn, "preparation", "e" * 32),
        _operation_index(turn, "tool.operation", "c" * 32),
    )
    preamble_to_run = (
        _operation_index(turn, "preparation", "a" * 32),
        _operation_index(turn, "model.run", "b" * 32),
    )
    assert run_to_cycle in parent_edges
    assert exchange_to_tool in parent_edges
    assert preamble_to_tool in causal_edges
    assert preamble_to_tool not in parent_edges
    assert preamble_to_run in causal_edges
    assert any(target == FLOW_TERMINAL_INDEX for _, target in causal_edges)
    assert all(target != FLOW_TERMINAL_INDEX for _, target in parent_edges)


def test_turn_flow_never_fabricates_edges_from_sequence_adjacency(tmp_path) -> None:
    """Back-to-back operations without declared pointers stay unconnected in the flow."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span(
        "tool.operation",
        "a" * 32,
        0,
        _NS,
        start_payload={"tool_name": "first", "tool_kind": "filesystem.read"},
    )
    log.span(
        "tool.operation",
        "b" * 32,
        _NS,
        2 * _NS,
        start_payload={"tool_name": "second", "tool_kind": "filesystem.read"},
    )
    log.add("turn.finished", 2 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    turn = analyze_trajectory(path).turns[0]
    flow = turn.flow

    assert flow is not None
    assert flow.parent_edges() == ()
    assert flow.causal_edges() == ()
    assert flow.root_index is None
    assert not flow.has_terminal


def test_counter_samples_come_from_the_analyzer_on_demand(tmp_path) -> None:
    """Timestamped counter samples join exchange finishes without per-turn retention."""
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    _exchange_usage(
        log,
        "a" * 32,
        0,
        _NS,
        {"input_total": 100, "output_total": 20, "reasoning": 7, "cache_read": 60},
    )
    log.add(
        "context.revision.recorded",
        _NS,
        operation_id="5" * 32,
        payload={"revision_id": "5" * 32, "is_checkpoint": True, "item_count": 3, "unidentified_item_count": 0},
    )
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    analysis = analyzer.load(path)
    turn_id = analysis.turns[0].turn_id
    samples = analyzer.counter_samples()

    usage = samples.usage_by_turn[turn_id]
    assert len(usage) == 1
    assert usage[0].end_ns == _NS
    assert (usage[0].input_tokens, usage[0].output_tokens) == (100, 20)
    assert (usage[0].reasoning_tokens, usage[0].cache_read_tokens, usage[0].cache_creation_tokens) == (7, 60, None)
    context = samples.context_by_turn[turn_id]
    assert [(sample.ns, sample.item_count) for sample in context] == [(_NS, 3)]


def test_counter_samples_omit_negative_context_item_counts(tmp_path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(
        "context.revision.recorded",
        _NS,
        operation_id="5" * 32,
        payload={"revision_id": "5" * 32, "is_checkpoint": True, "item_count": -1, "unidentified_item_count": 0},
    )
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    path = tmp_path / "events.jsonl"
    log.write(path)

    analyzer = TrajectoryAnalyzer()
    analysis = analyzer.load(path)

    assert analyzer.counter_samples().context_by_turn.get(analysis.turns[0].turn_id, ()) == ()


def test_session_projection_cache_reuses_the_unchanged_document(tmp_path) -> None:
    """The live dashboard refreshes twice a second; an unchanged session
    document must come back as the same parsed object, and a rewritten one
    must be reparsed."""
    path = tmp_path / ".chrys" / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    document = path.parents[1] / "session.json"
    document.write_text(json.dumps({"state": {"messages": []}}), encoding="utf-8")

    cache = session_projection_module._SessionProjectionCache()
    first = cache.lazy(path)()
    assert cache.lazy(path)() is first
    assert first.available and not first.mutation_detail_available

    document.write_text(
        json.dumps({"state": {"messages": [], "chrys_mutations": {"turns": []}}}),
        encoding="utf-8",
    )
    refreshed = cache.lazy(path)()
    assert refreshed is not first
    assert refreshed.mutation_detail_available
