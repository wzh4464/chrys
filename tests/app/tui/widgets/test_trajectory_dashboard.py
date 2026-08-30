# Copyright (c) 2026 Chrys. All rights reserved.

"""Frozen v5.51 trajectory-dashboard structure and display tests."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import NoReturn

import pytest
from rich.cells import cell_len
from rich.console import Console
from rich.style import Style
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.color import Color
from textual.geometry import Region, Size
from textual.pilot import Pilot
from textual.widgets import Tab, Tabs

from chrys.app.tui.support.gc_freeze import DetachedLruCache, GcFreezeBlockReason
from chrys.app.tui.widgets.chat.session_json import SessionJsonPanel
from chrys.app.tui.widgets.loading import ChrysLoadingIndicator
from chrys.app.tui.widgets.trajectory import DashboardTab, ResponsiveTier, TrajectoryDashboard
from chrys.app.tui.widgets.trajectory import panel as trajectory_panel
from chrys.app.tui.widgets.trajectory.chartkit import (
    bordered_section,
    coverage_bar,
    section_interior_width,
    time_ruler,
    unresolved_bar,
    waterfall_lanes,
)
from chrys.app.tui.widgets.trajectory.panel import (
    TrajectoryTextView,
    _align_edges,
    _cache_hit_metric,
    _fit_path_middle,
    _has_diagnostic_content,
    _input_share_metric,
)
from chrys.foundation.trajectory.envelope import Link, LinkRelation, SegmentedField
from chrys.foundation.trajectory.event_types import EventType
from chrys.foundation.trajectory.metadata import ANALYTICS_ITEM_ID_KEY
from chrys.service.analytics import (
    AnalysisAvailability,
    ChangeVerification,
    ChangeVerificationRow,
    ChangeVerificationState,
    Metric,
    Precision,
    SubmissionLatencyBucket,
    TimelineDiagnosticCode,
    TokenUsage,
    TrajectoryAnalysis,
    TrajectoryAnalyzer,
    TrajectoryDiagnostics,
    TrajectoryScanCancelled,
    UsageBucket,
)
from tests.service.analytics._events import EventLog
from tests.support.waiting import wait_for

_NS = 1_000_000_000


def test_coverage_bar_keeps_missing_distinct_from_unresolved() -> None:
    bar = coverage_bar(25, 25, 25, 25, width=8)

    assert bar.plain == "██▒▒░░··"


@pytest.mark.parametrize(
    ("identity", "hook_id", "expected"),
    [
        ("before_tool_call", "hook_id_1", "before_tool (hook_id_1)"),
        ("after_tool_call", "hook_abcdefghijklmnop", "after_tool (hook_abcdefgh...)"),
        ("after_turn", None, "after_turn"),
    ],
)
def test_hook_identity_shortens_tool_event_names_and_caps_only_the_hook_id(
    identity: str,
    hook_id: str | None,
    expected: str,
) -> None:
    assert trajectory_panel._identity_with_hook_id(identity, hook_id) == expected


def test_operation_reason_messages_cover_every_timeline_diagnostic_code() -> None:
    assert set(trajectory_panel._OPERATION_REASON_MESSAGES) == set(TimelineDiagnosticCode)


def test_grouped_grid_lines_keeps_each_semantic_group_in_one_column() -> None:
    groups = [[(Text(f"g{group}r{row}"), Text(str(row))) for row in range(3)] for group in range(4)]

    lines = trajectory_panel._grouped_grid_lines(groups, width=120, columns=4)

    assert len(lines) == 3
    for row, line in enumerate(lines):
        assert all(f"g{group}r{row}" in line.plain for group in range(4))


@pytest.mark.parametrize(
    "diagnostics",
    [
        TrajectoryDiagnostics(span_duration_mismatch_count=1),
        TrajectoryDiagnostics(containment_violation_count=1),
        TrajectoryDiagnostics(malformed_hook_execution_mode_count=1),
        TrajectoryDiagnostics(side_call_empty_shell_revisions=("a" * 32,)),
        TrajectoryDiagnostics(unidentified_membership_revision_count=1),
    ],
)
def test_diagnostic_content_gate_does_not_depend_on_overview_precision(
    diagnostics: TrajectoryDiagnostics,
) -> None:
    assert _has_diagnostic_content(diagnostics) is True
    assert _has_diagnostic_content(TrajectoryDiagnostics()) is False


@pytest.mark.parametrize("width", [1, 2, 4, 11])
def test_unknown_timeline_primitives_never_exceed_requested_width(width: int) -> None:
    assert cell_len(unresolved_bar(width).plain) == width
    assert cell_len(time_ruler(_NS, width=width).plain) == width


def test_time_ruler_spreads_tick_labels_across_the_axis() -> None:
    ruler = time_ruler(12 * _NS, width=60).plain

    assert cell_len(ruler) == 60
    assert ruler.startswith("0s")
    assert ruler.endswith("12s")
    assert "4.8s" in ruler and "7.2s" in ruler


def test_time_ruler_tick_units_follow_the_axis_magnitude() -> None:
    sub_second = time_ruler(800_000_000, width=60).plain
    minutes = time_ruler(638 * _NS, width=90).plain
    hours = time_ruler(3 * 3600 * _NS + 1800 * _NS, width=90).plain

    assert sub_second.startswith("0ms") and sub_second.endswith("800ms")
    assert minutes.startswith("0s") and minutes.endswith("10m38s")
    assert "1m46s" in minutes and "3m33s" in minutes
    assert hours.startswith("0s") and hours.endswith("3h30m")
    assert "35m00s" in hours and "1h10m" in hours


def test_derived_token_ratios_preserve_missing_and_inconsistent_evidence() -> None:
    estimated_usage = TokenUsage(
        {
            UsageBucket.INPUT: Metric(40, Precision.ESTIMATED),
            UsageBucket.CACHE_READ: Metric(20, Precision.MISSING),
        }
    )
    exact_session = TokenUsage({UsageBucket.INPUT: Metric(100, Precision.EXACT)})
    zero_session = TokenUsage({UsageBucket.INPUT: Metric(0, Precision.EXACT)})
    inconsistent_usage = TokenUsage(
        {
            UsageBucket.INPUT: Metric(40, Precision.EXACT),
            UsageBucket.CACHE_READ: Metric(50, Precision.EXACT),
        }
    )

    assert _input_share_metric(estimated_usage, exact_session).precision is Precision.ESTIMATED
    assert _input_share_metric(estimated_usage, zero_session).precision is Precision.MISSING
    assert _cache_hit_metric(estimated_usage).precision is Precision.MISSING
    assert _cache_hit_metric(inconsistent_usage).precision is Precision.UNRESOLVED


def test_bordered_section_is_cell_width_safe_for_cjk_titles() -> None:
    lines = bordered_section("发现", [Text("一行内容")], width=18, console=Console())

    assert lines[0].plain.startswith("┌─ 发现 ")
    assert all(cell_len(line.plain) == 18 for line in lines)


def test_bordered_section_pads_one_cell_inside_each_vertical_border() -> None:
    lines = bordered_section("T", [Text("x" * 20)], width=12, console=Console())

    assert section_interior_width(12) == 8
    assert lines[1].plain == "│ xxxxxxxx │"
    assert lines[2].plain == "│ xxxxxxxx │"
    assert lines[-1].plain == "└──────────┘"
    assert all(cell_len(line.plain) == 12 for line in lines)


def test_waterfall_lanes_paint_each_cell_in_exactly_one_lane() -> None:
    # One turn, 10 cells: the model covers everything except a tool window that
    # owns most of cells 4-5; model slivers around it must not repaint them.
    turn = (
        1000,
        {
            "model": [(0, 420), (421, 430), (580, 1000)],
            "tool": [(430, 580)],
        },
    )
    lanes = waterfall_lanes([turn], width=10, lanes=[("tool", "▬", ""), ("model", "█", "")])

    assert lanes["model"].plain == "████  ████"
    assert lanes["tool"].plain == "    ▬▬    "
    assert all(
        (model_cell == " ") or (tool_cell == " ")
        for model_cell, tool_cell in zip(lanes["model"].plain, lanes["tool"].plain, strict=True)
    )


def test_waterfall_lanes_keep_turn_separators_and_empty_canvas() -> None:
    lanes = waterfall_lanes(
        [(100, {"model": [(0, 100)]}), (100, {"model": [(0, 100)]})],
        width=9,
        lanes=[("model", "█", "")],
    )

    assert lanes["model"].plain == "████┊████"
    assert waterfall_lanes([], width=4, lanes=[("model", "█", "")])["model"].plain == "    "


def test_two_edge_alignment_is_cell_width_safe_for_cjk_labels() -> None:
    line = _align_edges(Text("标签"), Text("值 [精确]"), 20)

    assert cell_len(line.plain) == 20
    assert line.plain.startswith("标签")
    assert line.plain.endswith("值 [精确]")


@pytest.mark.parametrize(
    "path",
    [
        "/Users/jackil/Repos/deeply/nested/report.final.json",
        "C:\\Users\\jackil\\Repos\\deeply\\nested\\report.final.json",
    ],
)
def test_fit_path_middle_preserves_the_full_filename_and_extension(path: str) -> None:
    fitted = _fit_path_middle(path, 30)
    extension_only_fallback = _fit_path_middle(path, 10)

    assert cell_len(fitted) <= 30
    assert fitted.endswith(("/report.final.json", "\\report.final.json"))
    assert fitted.startswith(("/", "C:"))
    assert "…" in fitted
    assert cell_len(extension_only_fallback) <= 10
    assert extension_only_fallback.startswith("…")
    assert extension_only_fallback.endswith(".json")


class _DashboardApp(App[None]):
    def compose(self) -> ComposeResult:
        yield TrajectoryDashboard()


class _StyledDashboardApp(_DashboardApp):
    CSS = "TrajectoryTextView { background: #123456; color: #abcdef; }"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cache_hit", "semantic_name", "fallback"),
    [
        (29, "error", "red"),
        (30, "warning", "yellow"),
        (60, "warning", "yellow"),
        (61, "success", "green"),
    ],
)
async def test_cache_hit_meter_style_uses_threshold_theme_colors(
    cache_hit: int,
    semantic_name: str,
    fallback: str,
) -> None:
    async with _DashboardApp().run_test() as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)

        assert dashboard._cache_hit_style(cache_hit) == dashboard._semantic_style(semantic_name, fallback)


@pytest.mark.asyncio
async def test_change_verification_path_display_copy_is_surrogate_safe() -> None:
    exact_zero = Metric(0, Precision.EXACT)
    raw_path = "changed-\udcff.py"
    analysis = TrajectoryAnalysis(
        availability=AnalysisAvailability.AVAILABLE,
        path=Path("events.jsonl"),
        generation=1,
        change_verification=ChangeVerification(
            detail_available=True,
            detection_truncated=False,
            files_touched=Metric(1, Precision.EXACT),
            created=exact_zero,
            modified=Metric(1, Precision.EXACT),
            deleted=exact_zero,
            net_zero=exact_zero,
            rows=(
                ChangeVerificationRow(
                    path=raw_path,
                    state=ChangeVerificationState.VERIFIED,
                    last_change_turn=1,
                    precision=Precision.EXACT,
                ),
            ),
        ),
    )

    async with _DashboardApp().run_test() as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        lines = dashboard._change_verification_lines(analysis, width=80)

    assert any("changed-\\udcff.py" in line.plain for line in lines)


@pytest.mark.asyncio
async def test_change_verification_rows_and_counts_carry_precision_badges() -> None:
    """Each row shows its own precision and the counts line shows the worst
    of the five count precisions, so a degraded change section cannot pass
    for exact measurements."""
    exact_zero = Metric(0, Precision.EXACT)
    analysis = TrajectoryAnalysis(
        availability=AnalysisAvailability.AVAILABLE,
        path=Path("events.jsonl"),
        generation=1,
        change_verification=ChangeVerification(
            detail_available=True,
            detection_truncated=False,
            files_touched=Metric(2, Precision.EXACT),
            created=exact_zero,
            modified=Metric(2, Precision.ESTIMATED, "counts include window-inferred or peer-contested mutations"),
            deleted=exact_zero,
            net_zero=exact_zero,
            rows=(
                ChangeVerificationRow(
                    path="proven.py",
                    state=ChangeVerificationState.VERIFIED,
                    last_change_turn=1,
                    precision=Precision.EXACT,
                ),
                ChangeVerificationRow(
                    path="unprovable.py",
                    state=ChangeVerificationState.NET_ZERO,
                    last_change_turn=1,
                    precision=Precision.UNRESOLVED,
                ),
            ),
        ),
    )

    async with _DashboardApp().run_test() as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        lines = dashboard._change_verification_lines(analysis, width=80)
        narrow = dashboard._change_verification_lines(analysis, width=22)

    assert lines[0].plain.endswith("~")
    assert next(line.plain for line in lines if "proven.py" in line.plain).endswith("✓")
    assert next(line.plain for line in lines if "unprovable.py" in line.plain).endswith("✗")
    # A compressed column truncates the counts, never the badge.
    assert "…" in narrow[0].plain
    assert narrow[0].plain.endswith("~")


@pytest.mark.asyncio
async def test_change_verification_middle_crops_paths_before_the_filename() -> None:
    exact_zero = Metric(0, Precision.EXACT)
    analysis = TrajectoryAnalysis(
        availability=AnalysisAvailability.AVAILABLE,
        path=Path("events.jsonl"),
        generation=1,
        change_verification=ChangeVerification(
            detail_available=True,
            detection_truncated=False,
            files_touched=Metric(1, Precision.EXACT),
            created=exact_zero,
            modified=Metric(1, Precision.EXACT),
            deleted=exact_zero,
            net_zero=exact_zero,
            rows=(
                ChangeVerificationRow(
                    path="/Users/jackil/Repos/deeply/nested/report.final.json",
                    state=ChangeVerificationState.UNVERIFIED,
                    last_change_turn=1,
                    precision=Precision.EXACT,
                ),
            ),
        ),
    )

    async with _DashboardApp().run_test() as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        lines = dashboard._change_verification_lines(analysis, width=40)

    row = next(line.plain for line in lines if "report.final.json" in line.plain)
    assert row.startswith("/Users/")
    assert "…/report.final.json" in row
    assert row.endswith("✓")
    assert cell_len(row) == 40


@pytest.mark.parametrize(
    ("precision", "badge"),
    [(Precision.EXACT, "✓"), (Precision.UNRESOLVED, "✗")],
)
@pytest.mark.asyncio
async def test_insights_section_titles_render_their_panel_precision(
    tmp_path: Path,
    precision: Precision,
    badge: str,
) -> None:
    path = tmp_path / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    _write_p2_operations(path)
    analysis = TrajectoryAnalyzer().load(path)
    assert analysis.insights is not None
    reason = "sentinel unresolved panel" if precision is Precision.UNRESOLVED else None
    insights = replace(
        analysis.insights,
        tools=replace(analysis.insights.tools, precision=precision, reason=reason),
        mcp=replace(analysis.insights.mcp, precision=precision, reason=reason),
        skills=replace(analysis.insights.skills, precision=precision, reason=reason),
        context_carrying_precision=precision,
        context_carrying_reason=reason,
    )

    async with _DashboardApp().run_test(size=(220, 100)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("abcd1234", path)
        await _wait_loaded(dashboard, pilot)
        page = "\n".join(line.plain for line in dashboard._insights_lines(replace(analysis, insights=insights)))

    for title in ("Skills", "MCP servers", "Tool activity", "Context re-send cost · top 5"):
        assert f"{title} {badge}" in page


@pytest.mark.parametrize(
    ("precision", "badge"),
    [(Precision.EXACT, "✓"), (Precision.UNRESOLVED, "✗")],
)
@pytest.mark.asyncio
async def test_submission_aggregate_renders_derived_precision_for_the_same_duration(
    tmp_path: Path,
    precision: Precision,
    badge: str,
) -> None:
    path = tmp_path / ".chrys" / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    _write_p1_operations(path)
    analysis = TrajectoryAnalyzer().load(path)
    assert analysis.submission_latency is not None
    stats = next(
        bucket for bucket in analysis.submission_latency.buckets if bucket.bucket is SubmissionLatencyBucket.BECAME_TURN
    )
    reason = "sentinel unresolved aggregate" if precision is Precision.UNRESOLVED else None
    four_seconds = Metric(4 * _NS, precision, reason)
    submission = replace(
        analysis.submission_latency,
        buckets=(replace(stats, p50_ns=four_seconds, p90_ns=four_seconds, max_ns=four_seconds),),
    )

    async with _DashboardApp().run_test() as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        lines = dashboard._submission_latency_lines(replace(analysis, submission_latency=submission), width=80)

    aggregate = next(line.plain for line in lines if "started a new turn" in line.plain)
    assert aggregate.count("4.00 s") == 3
    assert aggregate.endswith(badge)


def _write_operations(path: Path, *, second_turn: bool = False, diagnostics: bool = False) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span(
        "preparation",
        "a" * 32,
        0,
        _NS,
        start_payload={"scope": "turn_preamble", "phase": "turn_dispatch"},
    )
    log.span(
        "model.run",
        "b" * 32,
        _NS,
        10 * _NS,
        links=(Link(relation=LinkRelation.CAUSED_BY, target_operation_id="a" * 32),),
    )
    log.span("model.cycle", "c" * 32, _NS, 10 * _NS, parent_operation_id="b" * 32)
    log.span(
        "model.exchange",
        "d" * 32,
        _NS,
        9 * _NS,
        parent_operation_id="c" * 32,
    )
    log.span(
        "preparation",
        "e" * 32,
        2 * _NS,
        3 * _NS,
        parent_operation_id="d" * 32,
        start_payload={"scope": "tool_preamble", "phase": "dispatch", "target_operation_id": "f" * 32},
    )
    log.span(
        "tool.operation",
        "f" * 32,
        3 * _NS,
        7 * _NS,
        parent_operation_id="d" * 32,
        start_payload={
            "tool_name": "Bash",
            "tool_kind": "shell",
            "argument_fingerprint": "0123456789abcdef",
            "parent_model_operation_id": "d" * 32,
        },
        links=(Link(relation=LinkRelation.CAUSED_BY, target_operation_id="e" * 32),),
    )
    log.span(
        "wait",
        "1" * 32,
        4 * _NS,
        5 * _NS,
        parent_operation_id="f" * 32,
        start_payload={"category": "approval"},
        finish_payload={"duration_ms": 900} if diagnostics else None,
    )
    log.span(
        "hook.operation",
        "2" * 32,
        6 * _NS,
        7 * _NS,
        parent_operation_id="f" * 32,
        start_payload={
            "hook_event": "after_tool_call",
            "hook_key": "register-session-to-git-after-turn",
            "execution_mode": "blocking",
            "scope": "turn",
        },
    )
    log.span(
        "sub_agent",
        "3" * 32,
        7 * _NS,
        9 * _NS,
        parent_operation_id="f" * 32,
        start_payload={"agent_profile": "Explore"},
    )
    log.add(
        "wait.started",
        8 * _NS,
        operation_id="4" * 32,
        parent_operation_id="d" * 32,
        payload={"category": "user_input"},
    )
    if diagnostics:
        log.span(
            "wait",
            "6" * 32,
            10 * _NS,
            11 * _NS,
            parent_operation_id="f" * 32,
            start_payload={"category": "new_wait_shape"},
        )
        log.add("trajectory.checkpoint", 11 * _NS, payload={"reason_code": "test"})
        log.add("profile.switched", 11 * _NS, payload={"kind": "agent"})
    log.add("turn.finished", 12 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    if second_turn:
        log.add("turn.started", 13 * _NS, turn_id="9" * 32, payload={"turn_number": 2})
        log.add(
            "turn.finished",
            14 * _NS,
            turn_id="9" * 32,
            payload={"end_reason": "cancelled", "duration_ms": 0},
        )
    log.write(path)
    if diagnostics:
        lines = path.read_bytes().splitlines()
        checkpoint_index = next(
            index for index, line in enumerate(lines) if json.loads(line)["event_type"] == "trajectory.checkpoint"
        )
        del lines[checkpoint_index]
        unsupported_index = next(
            index for index, line in enumerate(lines) if json.loads(line)["event_type"] == "profile.switched"
        )
        unsupported = json.loads(lines[unsupported_index])
        unsupported["event_type"] = "future.event"
        lines[unsupported_index] = json.dumps(unsupported, separators=(",", ":")).encode()
        lines.insert(unsupported_index, b"{corrupt trajectory test line}")
        path.write_bytes(b"\n".join(lines) + b"\n")


def _write_p1_operations(path: Path) -> None:
    turn_one = "4" * 32
    turn_two = "5" * 32
    verify_call = "7" * 32
    log = EventLog()
    log.coverage()
    log.span(
        "preparation",
        "a" * 32,
        0,
        _NS,
        turn_id=None,
        start_payload={"scope": "pre_turn", "phase": "admission"},
        finish_payload={"scope": "pre_turn", "outcome": "fresh_turn"},
    )
    log.add(
        "turn.started",
        2 * _NS,
        turn_id=turn_one,
        payload={"turn_number": 1, "preparation_scope_operation_id": "a" * 32},
    )
    _tool(log, "1" * 32, turn_one, 3, "search", "search", "search-1")
    _tool(log, "2" * 32, turn_one, 5, "read_file", "filesystem.read", "repeated", outcome="errored")
    _tool(log, "3" * 32, turn_one, 7, "write_file", "filesystem.write", "edit-1")
    log.add(
        "turn.finished",
        9 * _NS,
        turn_id=turn_one,
        payload={"end_reason": "cancelled", "duration_ms": 0},
    )
    log.span(
        "preparation",
        "b" * 32,
        10 * _NS,
        11 * _NS,
        turn_id=None,
        start_payload={"scope": "pre_turn", "phase": "admission"},
        finish_payload={"scope": "pre_turn", "outcome": "fresh_turn"},
    )
    log.add(
        "turn.started",
        12 * _NS,
        turn_id=turn_two,
        payload={"turn_number": 2, "preparation_scope_operation_id": "b" * 32},
    )
    _tool(log, "6" * 32, turn_two, 13, "read_file", "filesystem.read", "repeated", outcome="errored")
    _tool(log, "8" * 32, turn_two, 15, "read_file", "filesystem.read", "repeated")
    _tool(log, "9" * 32, turn_two, 17, "Bash", "shell", "verify", call_item_id=verify_call)
    _tool(log, "c" * 32, turn_two, 19, "write_file", "filesystem.write", "edit-2")
    log.span(
        "preparation",
        "d" * 32,
        21 * _NS,
        22 * _NS,
        turn_id=None,
        start_payload={"scope": "pre_turn", "phase": "admission"},
        finish_payload={"scope": "pre_turn", "outcome": "injected", "target_turn_id": turn_two},
    )
    log.span(
        "preparation",
        "e" * 32,
        23 * _NS,
        24 * _NS,
        turn_id=None,
        start_payload={"scope": "pre_turn", "phase": "admission"},
        finish_payload={"scope": "pre_turn", "outcome": "rejected"},
    )
    log.add(
        "turn.finished",
        25 * _NS,
        turn_id=turn_two,
        payload={"end_reason": "cancelled", "duration_ms": 0},
    )
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
                                    "arguments": json.dumps({"command": "pytest -q"}),
                                    "additional_properties": {ANALYTICS_ITEM_ID_KEY: verify_call},
                                }
                            ]
                        }
                    ],
                    "chrys_mutations": {
                        "turns": [
                            {
                                "turn_id": 1,
                                "detection_truncated": False,
                                "mutations": [
                                    {"path": "verified[bold].py", "before_hash": "a" * 64, "after_hash": "b" * 64}
                                ],
                            },
                            {
                                "turn_id": 2,
                                "detection_truncated": False,
                                "mutations": [
                                    {"path": "after_verify.py", "before_hash": "c" * 64, "after_hash": "d" * 64},
                                    {"path": "net_zero.py", "before_hash": "5a" * 32, "after_hash": "5a" * 32},
                                ],
                            },
                        ]
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def _write_p2_operations(path: Path) -> None:
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add("model.exchange.started", 0, operation_id="1" * 32)
    log.add(
        "model.exchange.finished",
        _NS,
        operation_id="1" * 32,
        payload={
            "outcome": "success",
            "duration_ms": 1000,
            "usage": {
                "normalized": {
                    "input_total": 1000,
                    "output_total": 200,
                    "reasoning": 50,
                    "cache_read": 750,
                    "cache_creation": 25,
                }
            },
        },
        measurements={
            "/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1},
            **{
                f"/payload/usage/normalized/{bucket}": {"source": "provider", "adapter_version": 1}
                for bucket in ("input_total", "output_total", "reasoning", "cache_read", "cache_creation")
            },
        },
    )
    mcp_id = "2" * 32
    log.add(
        "wait.started",
        _NS,
        operation_id="3" * 32,
        payload={"category": "mcp_connect", "server_name": "figma", "target_operation_id": mcp_id},
    )
    log.add(
        "wait.finished",
        2 * _NS,
        operation_id="3" * 32,
        payload={
            "category": "mcp_connect",
            "server_name": "figma",
            "target_operation_id": mcp_id,
            "duration_ms": 1000,
        },
        measurements={"/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    log.add(
        "tool.operation.started",
        2 * _NS,
        operation_id=mcp_id,
        payload={
            "tool_name": "figma_render",
            "tool_kind": "mcp",
            "tool_context": {"server_name": "figma", "remote_name": "render"},
        },
    )
    log.add(
        "tool.payload.observed",
        3 * _NS,
        operation_id=mcp_id,
        payload={
            "model_visible_bytes": 4096,
            "local_token_estimate": 100,
            "truncated": True,
            "artifact_id": "artifact-1",
        },
    )
    log.add(
        "tool.operation.finished",
        4 * _NS,
        operation_id=mcp_id,
        payload={"outcome": "success", "duration_ms": 2000},
        measurements={"/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    load_id = "4" * 32
    log.add(
        "tool.operation.started",
        5 * _NS,
        operation_id=load_id,
        payload={
            "tool_name": "load_skill",
            "tool_kind": "skill",
            "tool_context": {"skill_name": "slides", "skill_revision": "rev-a"},
        },
    )
    log.add(
        "tool.payload.observed",
        6 * _NS,
        operation_id=load_id,
        payload={"model_visible_bytes": 1000, "local_token_estimate": 250, "truncated": False},
    )
    log.add(
        "tool.operation.finished",
        6 * _NS,
        operation_id=load_id,
        payload={"outcome": "success", "duration_ms": 1000},
        measurements={"/payload/duration_ms": {"source": "monotonic_clock", "method_version": 1}},
    )
    log.span(
        "tool.operation",
        "5" * 32,
        7 * _NS,
        8 * _NS,
        start_payload={
            "tool_name": "run_skill_script",
            "tool_kind": "skill",
            "tool_context": {
                "skill_name": "slides",
                "skill_revision": "rev-b",
                "script_name": "scripts/render.py",
            },
        },
        finish_payload={"outcome": "failed", "exit_code": 7},
    )
    log.span(
        "tool.operation",
        "6" * 32,
        8 * _NS,
        9 * _NS,
        start_payload={"tool_name": "load_skill", "tool_kind": "future.kind"},
    )
    log.add("turn.finished", 10 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)
    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [],
                    "chrys_mutations": {
                        "turns": [
                            {
                                "turn_id": 1,
                                "detection_truncated": False,
                                "mutations": [
                                    {"path": "src/widget.py", "before_hash": "a" * 64, "after_hash": "b" * 64},
                                    {"path": "tests/test_widget.py", "before_hash": None, "after_hash": "c" * 64},
                                ],
                            }
                        ]
                    },
                }
            }
        ),
        encoding="utf-8",
    )


def _tool(
    log: EventLog,
    operation_id: str,
    turn_id: str,
    second: int,
    tool_name: str,
    tool_kind: str,
    fingerprint: str,
    *,
    outcome: str = "success",
    call_item_id: str | None = None,
) -> None:
    log.span(
        "tool.operation",
        operation_id,
        second * _NS,
        (second + 1) * _NS,
        turn_id=turn_id,
        start_payload={
            "tool_name": tool_name,
            "tool_kind": tool_kind,
            "call_item_id": call_item_id or operation_id,
            "argument_fingerprint": fingerprint,
        },
        finish_payload={"outcome": outcome},
    )


async def _wait_loaded(dashboard: TrajectoryDashboard, pilot: Pilot[None]) -> None:
    await wait_for(
        lambda: dashboard._analysis is not None,
        timeout=5,
        pilot=pilot,
        description="trajectory dashboard analysis",
    )


def _box_column_contents(lines: list[Text]) -> list[str]:
    """Per-box interior text with all whitespace removed, wrap- and column-safe.

    Side-by-side boxes interleave on each visual row; splitting on the border
    glyph and accumulating by column keeps every box's prose contiguous even
    when long lines fold inside a half-width box.
    """
    columns: dict[int, list[str]] = {}
    for line in lines:
        parts = line.plain.split("│")
        for index in range(1, len(parts) - 1, 2):
            columns.setdefault(index, []).append(parts[index])
    return ["".join("".join(parts).split()) for parts in columns.values()]


def _in_any_box(lines: list[Text], needle: str) -> bool:
    squashed = "".join(needle.split())
    return any(squashed in column for column in _box_column_contents(lines))


@pytest.mark.asyncio
async def test_dashboard_has_four_clickable_tabs_without_compare_and_placeholders(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_operations(path)

    async with _DashboardApp().run_test(size=(150, 32)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)

        assert [tab.id for tab in dashboard.query_one("#trajectory-tabs", Tabs).query(Tab)] == [
            DashboardTab.OVERVIEW,
            DashboardTab.TIMELINE,
            DashboardTab.INSIGHTS,
            DashboardTab.SESSION_DATA,
        ]
        assert all(not tab.disabled for tab in dashboard.query(Tab))
        assert all(not isinstance(tab.label, str) for tab in dashboard.query(Tab))
        assert list(dashboard.query("#insights-sub-tabs")) == []
        assert list(dashboard.query("#compare")) == []
        assert list(dashboard.query("#flow-mode-tabs")) == []
        assert dashboard.query_one("#trajectory-tabs", Tabs).size.height == 2
        assert dashboard.styles.background == Color.parse(pilot.app.theme_variables["background"])
        assert dashboard.border_subtitle is not None
        assert dashboard.border_subtitle.endswith(" · session")
        assert "exact" in dashboard.border_subtitle and "unresolved" in dashboard.border_subtitle
        # Legend labels share the subtitle's colour with the session id; only glyphs are coloured.
        legend = Text.from_markup(dashboard.border_subtitle)
        glyph_start = legend.plain.index("✓")
        label_start = legend.plain.index("exact")
        assert any(span.start <= glyph_start < span.end for span in legend.spans)
        assert not any(span.start <= label_start < span.end for span in legend.spans)

        await pilot.click("#insights")
        insights = "\n".join(line.plain for line in dashboard.query_one(TrajectoryTextView)._lines)
        assert "Findings" in insights
        assert "Diagnostics" in insights


@pytest.mark.asyncio
async def test_insights_renders_all_p2_sections_on_one_page(tmp_path: Path) -> None:
    path = tmp_path / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    _write_p2_operations(path)

    async with _DashboardApp().run_test(size=(220, 100)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("abcd1234", path)
        await _wait_loaded(dashboard, pilot)
        analysis = dashboard._analysis
        view = dashboard.query_one(TrajectoryTextView)

        await pilot.click("#insights")
        await wait_for(
            lambda: "Tokens per turn" in "\n".join(line.plain for line in view._lines),
            timeout=10,
            pilot=pilot,
            description="combined insights page render",
        )
        # Captured after the insights render settles: the live-refresh timer only
        # runs on the overview/timeline tabs, so any later bump is an accidental
        # analysis load scheduled by the insights render path.
        load_generation = dashboard._load_generation
        page = "\n".join(line.plain for line in view._lines)
        assert "Findings" in page and "Diagnostics" in page
        assert "Tool activity" in page
        assert _in_any_box(view._lines, "mcp · figma_render")
        assert _in_any_box(view._lines, "Unclassified: 1")
        assert "MCP servers" in page and "figma" in page and "render" in page
        assert _in_any_box(view._lines, "return volume") and _in_any_box(view._lines, "connection wait")
        assert "Skills" in page and "slides" in page and "scripts/render.py" in page
        assert _in_any_box(view._lines, "Skill changed during the session")
        assert "Tokens per turn" in page
        assert "cache creation" in page and "cache hit" in page
        assert "Context re-send cost · top 5" in page
        assert "FILE CHANGES" not in page
        assert "─ TOKENS ─" not in page
        assert (
            page.index("Skills")
            < page.index("MCP servers")
            < page.index("Tool activity")
            < page.index("Context re-send cost · top 5")
            < page.index("Tokens per turn")
            < page.index("Findings")
            < page.index("Diagnostics")
        )
        # The tall diagnostics wall exceeds the pair height gap, so this pair
        # falls back to stacked full-width boxes instead of a padded column.
        assert not any("Findings" in line.plain and "Diagnostics" in line.plain for line in view._lines)
        assert any("Skills" in line.plain and "MCP servers" in line.plain for line in view._lines)
        assert any(
            "Tool activity" in line.plain and "Context re-send cost · top 5" in line.plain for line in view._lines
        )
        assert "input share [" in page
        assert any("input share [" in line.plain and "cache hit [" in line.plain for line in view._lines)
        assert any("█" in line.plain and "%]" in line.plain for line in view._lines)

        assert dashboard._analysis is analysis
        assert dashboard._load_generation == load_generation
        await wait_for(
            lambda: view.max_scroll_x == 0,
            timeout=10,
            pilot=pilot,
            description="combined insights horizontal settle",
        )
        assert max(cell_len(line.plain) for line in view._lines) <= view.scrollable_content_region.width


@pytest.mark.asyncio
async def test_insights_keeps_unbalanced_pairs_side_by_side_despite_height_gap(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    turn = "4" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, turn_id=turn, payload={"turn_number": 1})
    for index in range(6):
        _tool(log, str(index) * 32, turn, index, f"tool{index}", "filesystem.read", f"{index}" * 16)
    log.add("turn.finished", 7 * _NS, turn_id=turn, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)

    async with _DashboardApp().run_test(size=(220, 100)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        view = dashboard.query_one(TrajectoryTextView)

        await pilot.click("#insights")
        await wait_for(
            lambda: "Tool activity" in "\n".join(line.plain for line in view._lines),
            timeout=10,
            pilot=pilot,
            description="insights render",
        )
        lines = [line.plain for line in view._lines]
        # Six tools make the tools box far taller than the empty context-cost
        # box, yet the pair stays on one row so the tools box does not stretch
        # full-width; the two integration summaries share a row the same way.
        assert any("Tool activity" in line and "Context re-send cost · top 5" in line for line in lines)
        assert any("Skills" in line and "MCP servers" in line for line in lines)
        assert _in_any_box(view._lines, "filesystem.read · tool5")
        assert _in_any_box(view._lines, "This session has no MCP calls.")


@pytest.mark.asyncio
async def test_insights_describes_context_re_send_rows_by_message_kind(tmp_path: Path) -> None:
    item_id = "7" * 32
    revision_id = "8" * 32
    segment_id = "9" * 32
    exchange_id = "a" * 32
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
    log.add("turn.finished", 3 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)
    path.parents[1].joinpath("session.json").write_text(
        json.dumps(
            {
                "state": {
                    "messages": [
                        {
                            "role": "assistant",
                            "additional_properties": {ANALYTICS_ITEM_ID_KEY: item_id, "_group": {"token_count": 1234}},
                            "contents": [
                                {"type": "function_call", "call_id": "c1", "name": "zsh", "arguments": "{}"},
                                {"type": "function_call", "call_id": "c2", "name": "zsh", "arguments": "{}"},
                                {"type": "function_call", "call_id": "c3", "name": "read_file", "arguments": "{}"},
                            ],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    async with _DashboardApp().run_test(size=(220, 100)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("abcd1234", path)
        await _wait_loaded(dashboard, pilot)
        view = dashboard.query_one(TrajectoryTextView)

        await pilot.click("#insights")
        await wait_for(
            lambda: "Context re-send cost · top 5" in "\n".join(line.plain for line in view._lines),
            timeout=10,
            pilot=pilot,
            description="insights render",
        )
        assert _in_any_box(view._lines, "tokens × model requests that re-sent the item")  # noqa: RUF001
        head_index = next(
            index
            for index, line in enumerate(view._lines)
            if "assistant message (zsh ×2, read_file) · since turn 1" in line.plain  # noqa: RUF001
        )
        # Two lines per item: the total cost rides the head line in compact
        # units, the cost formula and relative bar follow on the next line.
        assert view._lines[head_index].plain.rstrip(" │").endswith("1.2k")
        detail = view._lines[head_index + 1].plain
        assert "1.2k tok × 1 re-sends" in detail  # noqa: RUF001
        assert "▬" in detail
        assert not any(item_id[:12] in line.plain for line in view._lines)


@pytest.mark.asyncio
async def test_session_data_is_only_json_content_and_obeys_lifecycle_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / ".chrys" / "sessions" / "session" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    _write_operations(path)
    calls: list[tuple[str, bool]] = []

    async with _DashboardApp().run_test(size=(120, 30)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        session_json = dashboard.query_one(SessionJsonPanel)
        original_hide = session_json.hide_session_json

        def load_session(_session_id: str) -> None:
            calls.append(("load", session_json.display))

        def hide_session_json() -> None:
            calls.append(("hide", session_json.display))
            original_hide()

        monkeypatch.setattr(session_json, "load_session", load_session)
        monkeypatch.setattr(session_json, "hide_session_json", hide_session_json)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)

        await pilot.click("#session-data")
        await pilot.pause()
        assert dashboard.active_tab is DashboardTab.SESSION_DATA
        assert dashboard.display is True
        assert session_json.display is True
        assert dashboard.query_one(TrajectoryTextView).display is False
        assert calls[-1] == ("load", True)
        assert dashboard.border_subtitle == str(path.parents[1] / "session.json")
        assert session_json.styles.border.top[0] == ""

        await pilot.click("#overview")
        await pilot.pause()
        assert session_json.display is False
        assert calls[-1][0] == "hide"
        assert dashboard.border_subtitle is not None
        assert dashboard.border_subtitle.endswith(" · session")

        await pilot.click("#session-data")
        dashboard.hide_dashboard()
        assert session_json.display is False
        assert calls[-1][0] == "hide"


@pytest.mark.asyncio
async def test_tab_activation_moves_focus_into_the_visible_content_view(tmp_path: Path) -> None:
    """Page keys must scroll the activated tab's document without a click inside it."""
    path = tmp_path / ".chrys" / "sessions" / "session" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    _write_operations(path)

    async with _DashboardApp().run_test(size=(120, 30)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)

        await pilot.click("#session-data")
        await pilot.pause()
        assert pilot.app.focused is dashboard.query_one(SessionJsonPanel)

        await pilot.click("#overview")
        await pilot.pause()
        assert pilot.app.focused is dashboard.query_one(TrajectoryTextView)


@pytest.mark.asyncio
async def test_session_data_status_renders_like_the_dashboard_empty_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    log.write(path)

    def label_segment(strip, needle: str):
        return next(segment for segment in strip._segments if needle in segment.text)

    async with _DashboardApp().run_test(size=(120, 30)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        session_json = dashboard.query_one(SessionJsonPanel)
        view = dashboard.query_one(TrajectoryTextView)
        monkeypatch.setattr(session_json, "_resolve_session_path", lambda _session_id: None)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        await wait_for(
            lambda: any("No completed turns" in line.plain for line in view._lines),
            timeout=5,
            pilot=pilot,
            description="empty state label render",
        )
        empty_line = next(index for index, line in enumerate(view._lines) if "No completed turns" in line.plain)
        empty_strip = view.render_line(empty_line)
        empty_label = label_segment(empty_strip, "No completed turns")
        empty_hatch = next(segment for segment in empty_strip._segments if segment.text.startswith("╲"))
        empty_filler = next(segment for segment in view.render_line(0)._segments if segment.text.startswith("╲"))

        await pilot.click("#session-data")
        await wait_for(lambda: session_json.display and bool(session_json._status), timeout=5, pilot=pilot)
        await pilot.pause()
        width = session_json.scrollable_content_region.width
        status_strip = session_json.render_line(session_json.scrollable_content_region.height // 2)
        plain = "".join(segment.text for segment in status_strip._segments)
        status_label = label_segment(status_strip, "No session file found.")
        status_hatch = next(segment for segment in status_strip._segments if segment.text.startswith("╲"))
        # A label-free row is a span-less Text unless the helper adds one;
        # Text.render() would then drop the hatch colour and paint the row in
        # the widget's bright foreground (the defect seen in the real app).
        status_filler = next(
            segment for segment in session_json.render_line(0)._segments if segment.text.startswith("╲")
        )
        json_foreground = session_json.visual_style.rich_style.color

    # Same shape and styling as the trajectory empty state: a padded, centered
    # label between hatch runs, rendered through the same Rich helpers.
    assert status_label.text == " No session file found. "
    assert plain.startswith("╲") and plain.endswith("╲")
    assert cell_len(plain) == width
    left = plain.index(status_label.text)
    right = width - left - cell_len(status_label.text)
    assert abs(left - right) <= 1
    assert status_label.style is not None and empty_label.style is not None
    assert status_label.style.color == empty_label.style.color
    assert status_label.style.bold == empty_label.style.bold
    assert status_hatch.style is not None and empty_hatch.style is not None
    assert status_hatch.style.color == empty_hatch.style.color
    assert empty_filler.style is not None and status_filler.style is not None
    assert status_filler.style.color == empty_hatch.style.color
    assert empty_filler.style.color == empty_hatch.style.color
    assert status_filler.style.color != json_foreground
    # $text-muted is composite on every theme ("auto 60%"); a plain color
    # parse yields an uncolored label, so the muted effect only exists when
    # the stylesheet resolves it to a concrete blend.
    assert status_label.style.color is not None
    assert status_label.style.color != json_foreground


@pytest.mark.asyncio
async def test_session_data_load_overlays_the_shared_loading_indicator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.jsonl"
    _write_operations(path)
    session_file = tmp_path / "session.json"
    session_file.write_text(json.dumps({"value": 1}), encoding="utf-8")
    started = Event()
    release = Event()

    def slow_highlight(json_text: str, dark: bool, gutter_color: str | None):
        started.set()
        assert release.wait(timeout=10)
        return SessionJsonPanel._highlight(json_text, dark, gutter_color)

    async with _DashboardApp().run_test(size=(120, 30)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        session_json = dashboard.query_one(SessionJsonPanel)
        text_view = dashboard.query_one(TrajectoryTextView)
        loading_state = dashboard.query_one("#trajectory-loading-state")
        monkeypatch.setattr(session_json, "_resolve_session_path", lambda _session_id: session_file)
        monkeypatch.setattr(session_json, "_highlight", slow_highlight)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        assert loading_state.display is False

        await pilot.click("#session-data")
        await wait_for(
            lambda: started.is_set() and loading_state.display is True,
            timeout=5,
            pilot=pilot,
            description="session data load indicator",
        )
        # The viewer stays displayed so its worker can commit; the shared
        # indicator floats over it on its own layer, leaving the tab strip.
        assert session_json.is_loading is True
        assert session_json.display is True
        assert text_view.display is False
        assert dashboard.query_one(ChrysLoadingIndicator).display is True
        tabs = dashboard.query_one("#trajectory-tabs", Tabs)
        await wait_for(
            lambda: (
                loading_state.region == session_json.region
                and loading_state.region.y == tabs.region.y + tabs.region.height
            ),
            timeout=5,
            pilot=pilot,
            description="loading overlay covers the session viewer",
        )

        release.set()
        await wait_for(
            lambda: not session_json.is_loading and loading_state.display is False,
            timeout=5,
            pilot=pilot,
            description="session data load settles",
        )
        assert session_json.display is True
        assert session_json._plain_lines
        assert text_view.display is False

        # Leaving the tab releases the viewer and never leaves the overlay up.
        await pilot.click("#overview")
        await pilot.pause()
        assert session_json.display is False
        assert loading_state.display is False
        assert text_view.display is True


@pytest.mark.asyncio
async def test_overview_renders_kpi_waterfall_coverage_and_structured_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_operations(path, diagnostics=True)

    async with _DashboardApp().run_test(size=(220, 36)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        text = "\n".join(line.plain for line in dashboard.query_one(TrajectoryTextView)._lines)

        assert "Time & usage" in text
        assert "Session info ✗" in text
        assert "Where time went" in text
        assert "Parallelism & busy" in text
        assert "KEY METRICS" not in text
        assert "total time" in text
        assert "model" in text and "tools" in text and "wait" in text and "idle" in text
        assert "Busy share (model and tools independent; >100% = parallel work)" in text
        assert "data confidence" in text
        assert "Per-turn time breakdown" in text
        overview_lines = [line.plain for line in dashboard.query_one(TrajectoryTextView)._lines]
        idle_row = next(index for index, line in enumerate(overview_lines) if line.startswith("│ idle"))
        # The waterfall closes with a time ruler on the lanes' cumulative scale.
        assert overview_lines[idle_row + 1].startswith("│ time")
        assert "0s" in overview_lines[idle_row + 1]
        assert "Token usage" in text
        assert "Skill usage" in text
        assert "MCP usage" in text
        assert "Action breakdown" in text
        assert "Failure recovery" in text
        assert "Change verification" in text
        assert "Submission wait (submit → work starts)" in text
        assert "Diagnostics" not in text
        view = dashboard.query_one(TrajectoryTextView)
        await wait_for(
            lambda: view.max_scroll_x == 0,
            timeout=5,
            pilot=pilot,
            description="overview horizontal settle",
        )
        assert max(cell_len(line.plain) for line in view._lines) <= view.scrollable_content_region.width
        assert all(
            any(label in line.plain and any(symbol in line.plain for symbol in "✓~−✗") for line in view._lines)  # noqa: RUF001
            for label in ("model", "tools", "wait", "idle")
        )
        assert "✓ exact" not in text
        assert dashboard.border_subtitle is not None
        assert "exact" in dashboard.border_subtitle and "unresolved" in dashboard.border_subtitle
        assert dashboard.border_subtitle.endswith(" · session")
        assert "cache hit" in text
        assert "cache creation" not in text
        assert "█" in text and "░" in text

        await pilot.click("#insights")
        insight_lines = dashboard.query_one(TrajectoryTextView)._lines
        insights = "\n".join(line.plain for line in insight_lines)
        assert "Diagnostics" in insights
        assert _in_any_box(insight_lines, "Corrupt line")
        assert _in_any_box(insight_lines, "Unsupported line")
        assert _in_any_box(insight_lines, "metrics in the affected range degrade to unresolved")
        assert _in_any_box(insight_lines, "after seq")
        assert _in_any_box(insight_lines, "Accounted-prefix seq")
        assert _in_any_box(insight_lines, "response linkage lacks final_exchange_operation_id")
        # Recorded-duration drift collapses into one explanatory summary line
        # instead of a per-span wall of notes.
        assert _in_any_box(insight_lines, "1 span's recorded duration drifts from its lifecycle interval")
        assert _in_any_box(insight_lines, "(wait; up to 100 ms)")
        assert not _in_any_box(insight_lines, "@11111111")
        assert _in_any_box(insight_lines, "Containment wait @66666666")


@pytest.mark.asyncio
async def test_overview_session_info_shows_folder_sizes_and_wall_clock_for_store_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capability_checks: list[None] = []

    def can_open_folder() -> bool:
        capability_checks.append(None)
        return True

    monkeypatch.setattr(trajectory_panel, "can_open_in_file_manager", can_open_folder)
    session_dir = tmp_path / "sessions" / "0123456789ab"
    trajectory_dir = session_dir / "trajectory"
    trajectory_dir.mkdir(parents=True)
    path = trajectory_dir / "events.jsonl"
    _write_operations(path)
    (session_dir / "session.json").write_bytes(b"x" * 2048)
    (session_dir / "mutations").mkdir()
    (session_dir / "mutations" / "0001.diff").write_bytes(b"y" * 3072)

    async with _DashboardApp().run_test(size=(150, 40)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        assert capability_checks == [None]
        lines = dashboard.query_one(TrajectoryTextView)._lines
        text = "\n".join(line.plain for line in lines)

        assert "Session info ✓" in text
        assert "0123456789ab" in text  # the path keeps its tail when cropped
        assert "session.json" in text and "2.0 KB" in text
        assert "3.0 KB" in text  # mutations subtree
        assert "3 files" in text
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", text)  # local wall clock
        open_line = next(line for line in lines if "Open folder" in line.plain)
        assert open_line.plain.index("Copy path") < open_line.plain.index("Open folder")
        metas = [span.style.meta for span in open_line.spans if isinstance(span.style, Style)]
        assert {"@click": "copy_session_path"} in metas
        assert {"@click": "open_session_folder"} in metas

        copied: list[str] = []
        notices: list[str] = []
        monkeypatch.setattr(trajectory_panel, "copy_text_to_clipboards", lambda _app, value: copied.append(value))
        monkeypatch.setattr(dashboard, "notify", lambda message, **kwargs: notices.append(message))
        dashboard.query_one(TrajectoryTextView).action_copy_session_path()
        assert copied == [str(session_dir)]
        assert notices == ["Path copied"]

        opened: list[Path] = []
        monkeypatch.setattr(trajectory_panel, "open_in_file_manager", opened.append)
        dashboard.open_session_folder()
        assert opened == [session_dir]

        monkeypatch.setattr(
            trajectory_panel,
            "open_in_file_manager",
            lambda folder: (_ for _ in ()).throw(FileNotFoundError("xdg-open")),
        )
        dashboard.open_session_folder()
        assert len(notices) == 2 and "xdg-open" in notices[1]

        monkeypatch.setattr(trajectory_panel, "can_open_in_file_manager", lambda: False)
        dashboard.open_session_folder()
        assert len(notices) == 3 and "current environment" in notices[2]


@pytest.mark.asyncio
async def test_overview_session_info_hides_open_folder_over_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trajectory_panel, "can_open_in_file_manager", lambda: False)
    session_dir = tmp_path / "sessions" / "0123456789ab"
    trajectory_dir = session_dir / "trajectory"
    trajectory_dir.mkdir(parents=True)
    path = trajectory_dir / "events.jsonl"
    _write_operations(path)

    async with _DashboardApp().run_test(size=(150, 40)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        lines = dashboard.query_one(TrajectoryTextView)._lines

        assert any("0123456789ab" in line.plain for line in lines)
        assert any("Copy path" in line.plain for line in lines)
        assert all("Open folder" not in line.plain for line in lines)
        assert any(
            span.style.meta.get("@click") == "copy_session_path"
            for line in lines
            for span in line.spans
            if isinstance(span.style, Style)
        )
        assert all(
            span.style.meta.get("@click") != "open_session_folder"
            for line in lines
            for span in line.spans
            if isinstance(span.style, Style)
        )


@pytest.mark.asyncio
async def test_overview_session_info_degrades_to_placeholders_for_a_loose_events_file(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_operations(path)

    async with _DashboardApp().run_test(size=(150, 40)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        lines = dashboard.query_one(TrajectoryTextView)._lines
        text = "\n".join(line.plain for line in lines)

        assert "Session info" in text
        assert dashboard._session_storage is None
        on_disk_line = next(line.plain for line in lines if "on disk" in line.plain)
        assert "—" in on_disk_line
        # The folder line still names the log's parent so "Open folder" stays honest.
        assert tmp_path.name in text


@pytest.mark.asyncio
async def test_p1_overview_renders_findings_bottom_panes_and_non_additive_submission_latency(tmp_path: Path) -> None:
    path = tmp_path / ".chrys" / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    _write_p1_operations(path)

    async with _DashboardApp().run_test(size=(240, 400)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("abcd1234", path)
        await _wait_loaded(dashboard, pilot)
        text_view = dashboard.query_one(TrajectoryTextView)
        text = "\n".join(line.plain for line in text_view._lines)

        assert "Findings" not in text
        assert "Unverified change" not in text
        assert "Action breakdown" in text
        for label, value in (("search", "1"), ("read", "3"), ("edit", "2"), ("verify", "1")):
            row = next(line.plain for line in text_view._lines if line.plain.startswith(f"│ {label}"))
            assert value in row
        assert "Failure recovery" in text
        assert any("tool failures" in line.plain and "2/7" in line.plain for line in text_view._lines)
        assert any("repeated identical failures" in line.plain and "1" in line.plain for line in text_view._lines)
        assert "Change verification" in text
        assert any("verified[bold].py" in line.plain and "verified" in line.plain for line in text_view._lines)
        assert any("after_verify.py" in line.plain and "after verify" in line.plain for line in text_view._lines)
        assert any("net_zero.py" in line.plain and "cancelled out" in line.plain for line in text_view._lines)
        assert "Token usage" in text
        assert "No skills were used." in text
        assert "No MCP tools were called." in text
        assert "Submission wait (submit → work starts)" in text
        assert "How long each message waited" in text
        assert any("started a new turn" in line.plain and "2 samples" in line.plain for line in text_view._lines)
        assert any(
            "injected into an ongoing turn" in line.plain and "1 sample" in line.plain for line in text_view._lines
        )
        assert any("never became a turn" in line.plain and "1 sample" in line.plain for line in text_view._lines)
        assert "median" in text and "p90" in text and "slowest" in text
        assert any(line.plain.startswith("│ Turn 1") for line in text_view._lines)
        assert "Σ" not in text
        assert text_view.max_scroll_x == 0
        assert max(cell_len(line.plain) for line in text_view._lines) <= text_view.scrollable_content_region.width

        await pilot.click("#insights")
        insights_view = dashboard.query_one(TrajectoryTextView)
        insights = "\n".join(line.plain for line in insights_view._lines)
        assert "Findings" in insights
        assert "Unverified change" in insights
        assert "Repeated tool fingerprint" in insights
        assert "Changes cancelled out" in insights
        assert " · deterministic" not in insights
        assert insights.index("Findings") < insights.index("Diagnostics")
        # The ignore interaction no longer exists, so no ignored-count footer.
        assert "ignored finding" not in insights
        assert "Data-integrity notes" in insights


@pytest.mark.asyncio
async def test_overview_section_rows_align_both_edges_without_horizontal_overflow(tmp_path: Path) -> None:
    path = tmp_path / ".chrys" / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    _write_p1_operations(path)

    # Narrow tier with the box padding still leaving the latency labels intact.
    async with _DashboardApp().run_test(size=(74, 400)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("abcd1234", path)
        await _wait_loaded(dashboard, pilot)
        view = dashboard.query_one(TrajectoryTextView)

        elapsed = next(line.plain for line in view._lines if line.plain.startswith("│ total time"))
        search = next(line.plain for line in view._lines if line.plain.startswith("│ search"))
        failures = next(line.plain for line in view._lines if line.plain.startswith("│ tool failures"))
        change = next(line.plain for line in view._lines if line.plain.startswith("│ verified[bold].py"))
        stats = next(line.plain for line in view._lines if line.plain.startswith("│ started"))
        sample = next(line.plain for line in view._lines if line.plain.startswith("│ Turn 1"))

        assert elapsed.endswith("✗ │")
        assert search.endswith("✓ │")
        assert failures.endswith("✓ │")
        assert change.endswith("verified ~ │")
        assert stats.endswith("slowest 2.00 s ✓ │")
        assert sample.endswith("✓ │")
        assert view.max_scroll_x == 0
        assert max(cell_len(line.plain) for line in view._lines) <= view.scrollable_content_region.width


@pytest.mark.asyncio
async def test_overview_findings_are_display_only_and_arrow_keys_scroll(tmp_path: Path) -> None:
    path = tmp_path / ".chrys" / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    _write_p1_operations(path)

    async with _DashboardApp().run_test(size=(200, 24)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("abcd1234", path)
        await _wait_loaded(dashboard, pilot)
        await pilot.click("#insights")
        view = dashboard.query_one(TrajectoryTextView)
        text = "\n".join(line.plain for line in view._lines)
        view.focus()
        await pilot.press("down")
        await pilot.pause()

        assert "Unverified change" in text
        assert "↑/↓ select" not in text
        assert dashboard.active_tab is DashboardTab.INSIGHTS
        assert view.scroll_offset.y > 0


@pytest.mark.asyncio
async def test_live_refresh_preserves_scroll_and_tab_switch_resets_it(tmp_path: Path) -> None:
    path = tmp_path / ".chrys" / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    _write_p1_operations(path)

    async with _DashboardApp().run_test(size=(200, 24)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("abcd1234", path)
        await _wait_loaded(dashboard, pilot)
        view = dashboard.query_one(TrajectoryTextView)
        view.scroll_to(y=3, animate=False, force=True, immediate=True)
        await pilot.pause()
        assert view.scroll_offset.y == 3

        analysis = dashboard._analysis
        assert analysis is not None
        # A live session appends events and republishes the same view under a
        # new analysis generation; the reader's scroll position must survive.
        dashboard._analysis = replace(analysis, generation=analysis.generation + 1)
        dashboard._render_active_view()
        await pilot.pause()
        assert view.scroll_offset.y == 3

        await pilot.click("#insights")
        await pilot.pause()
        assert dashboard.active_tab is DashboardTab.INSIGHTS
        assert view.scroll_offset.y == 0


@pytest.mark.asyncio
async def test_stale_text_view_region_cannot_widen_the_render_past_the_dashboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".chrys" / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    _write_p1_operations(path)

    async with _DashboardApp().run_test(size=(74, 24)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("abcd1234", path)
        await _wait_loaded(dashboard, pilot)
        view = dashboard.query_one(TrajectoryTextView)
        # The load schedules one more render for after the text view's own
        # layout settles; the first frame the user keeps must fit the view.
        await wait_for(
            lambda: (
                bool(view._lines)
                and view.max_scroll_x == 0
                and max(cell_len(line.plain) for line in view._lines) <= view.scrollable_content_region.width
            ),
            timeout=5,
            pilot=pilot,
            description="trajectory dashboard settled first render",
        )
        available = dashboard._available_width()
        assert 0 < available <= 74
        # The display flip that ends a load publishes lines before the text
        # view's region reflects the new layout; a stale, wider region must
        # not leak oversized lines into the commit.
        stale = Region(0, 0, available + 126, view.scrollable_content_region.height)
        monkeypatch.setattr(TrajectoryTextView, "scrollable_content_region", property(lambda self: stale))
        assert dashboard._content_width() <= available
        dashboard._render_active_view()
        assert max(cell_len(line.plain) for line in view._lines) <= available


def test_log_appends_do_not_bypass_the_storage_scan_throttle(tmp_path: Path) -> None:
    path = tmp_path / ".chrys" / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    log = EventLog()
    log.coverage()
    log.turn(1, 2)
    log.write(path)
    analyzer = TrajectoryAnalyzer()
    previous = analyzer.load(path)
    append = EventLog()
    append.add("turn.started", 10 * _NS, turn_id="5" * 32, payload={"turn_number": 2})
    append.add("turn.finished", 12 * _NS, turn_id="5" * 32, payload={"end_reason": "cancelled", "duration_ms": 0})
    append.write(tmp_path / "append.jsonl", start_sequence=4)
    with path.open("ab") as handle:
        handle.write((tmp_path / "append.jsonl").read_bytes())

    analysis, storage = trajectory_panel._refresh_with_storage(analyzer, collect_storage=False, cancel_event=Event())

    # The append produced a fresh analysis, yet the directory walk stays on
    # the caller's coarse clock; the panel keeps its previous figures.
    assert analysis is not previous
    assert storage is None

    analysis, storage = trajectory_panel._refresh_with_storage(analyzer, collect_storage=True, cancel_event=Event())

    assert storage is not None


@pytest.mark.asyncio
async def test_verify_command_change_has_distinct_presentation_identity_and_reaggregates(tmp_path: Path) -> None:
    path = tmp_path / ".chrys" / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    _write_p1_operations(path)

    async with _DashboardApp().run_test(size=(200, 70)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("abcd1234", path)
        await _wait_loaded(dashboard, pilot)
        before = dashboard._analysis
        assert before is not None
        assert before.validation is not None
        assert before.validation.funnel.verify.value == 1

        view = dashboard.query_one(TrajectoryTextView)

        def presentation_key() -> tuple[object, ...]:
            analysis = dashboard._analysis
            region = view.scrollable_content_region
            return (
                DashboardTab.OVERVIEW,
                False,
                None,
                None,
                analysis.generation if analysis is not None else -1,
                dashboard._available_width(),
                dashboard._available_height(),
                region.width,
                region.height,
                view.show_vertical_scrollbar,
                view.show_horizontal_scrollbar,
                "cargo test",
                dashboard._presentation_revision,
            )

        dashboard.set_verify_commands("cargo test")
        # The reload hides the text view behind the loading indicator, so the
        # first build after the analysis swap may run before the view's region
        # settles (uncached by design); wait for the settled presentation.
        await wait_for(
            lambda: (
                dashboard._analysis is not None
                and dashboard._analysis is not before
                and dashboard._presentation_cache.get(presentation_key()) is not None
            ),
            timeout=5,
            pilot=pilot,
            description="trajectory verify-command reprojection",
        )

        after = dashboard._analysis
        assert after is not None
        assert after.validation is not None
        assert after.validation.funnel.verify.value == 0
        assert view.scrollable_content_region.height > 0
        assert dashboard._presentation_cache.get(presentation_key()) is not None


@pytest.mark.asyncio
async def test_timeline_renders_operations_hierarchy_identity_ruler_and_unresolved_bar(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_operations(path, second_turn=True)

    async with _DashboardApp().run_test(size=(150, 32)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        analysis = dashboard._analysis
        assert analysis is not None
        # Both turns live in the events file, so a live refresh landing at any
        # point reproduces the same turn ids and cannot invalidate navigation.
        first_turn, second_turn = analysis.turns
        dashboard.select_turn(second_turn.turn_id)
        dashboard.query_one(TrajectoryTextView).focus()
        await pilot.pause()
        assert dashboard._selected_turn_id == second_turn.turn_id

        turn_tabs = dashboard.query_one("#timeline-turn-tabs", Tabs)
        assert turn_tabs.display is True
        assert [tab.id for tab in turn_tabs.query(Tab)] == ["turn-0", "turn-1"]
        assert [tab.label.plain for tab in turn_tabs.query(Tab)] == ["Turn 1", "Turn 2"]
        assert turn_tabs.active == "turn-1"

        # Arrow keys scroll the timeline; they no longer switch turns.
        await pilot.press("up")
        await pilot.pause()
        assert dashboard._selected_turn_id == second_turn.turn_id

        turn_tabs.active = "turn-0"
        await wait_for(
            lambda: dashboard._selected_turn_id == first_turn.turn_id,
            timeout=5,
            pilot=pilot,
            description="turn tab navigation",
        )
        text = "\n".join(line.plain for line in dashboard.query_one(TrajectoryTextView)._lines)

        assert "time" in text
        assert "↑/↓" not in text
        assert re.search(r"Prepare\s+preparation", text)
        assert "Bash (#01234567)" in text
        hook_operation = next(operation for operation in first_turn.operations if operation.family == "hook.operation")
        assert dashboard._operation_identity(hook_operation) == "after_tool (register-sess...)"
        assert "after_tool (register-sess...)" in text
        assert "approval" in text
        assert "Explore" in text
        assert "│ Bash" in text
        assert "│ │ approval" in text
        assert "?···" in text
        assert "wait lifecycle has no terminal endpoint" not in text
        wait_diagnostic = next(
            item for item in analysis.diagnostics.timeline_operations if item.identity == "user_input"
        )
        assert wait_diagnostic.code is TimelineDiagnosticCode.MISSING_TERMINAL
        diagnostic_text = "\n".join(line.plain for line in dashboard._diagnostic_lines(analysis))
        assert "Turn 1 · user_input @44444444: lifecycle has no terminal endpoint" in diagnostic_text
        assert "idle" not in text.lower()

        view = dashboard.query_one(TrajectoryTextView)

        def bash_line() -> str:
            return next(line.plain for line in view._lines if "Bash (#01234567)" in line.plain)

        def operation_lines() -> list[str]:
            rows = view._lines[2 : 2 + len(first_turn.operations)]
            assert len(rows) == len(first_turn.operations) > 0
            return [line.plain for line in rows]

        def canvas_width() -> int:
            # 71 cells is the narrowest layout whose columns all fit; below it
            # the timeline draws on a fixed 92-cell canvas and scrolls
            # horizontally instead of squeezing the bars away.
            content_width = view.scrollable_content_region.width
            return content_width if content_width >= 71 else max(content_width, 92)

        for width in (150, 90, 71, 70, 50):
            await pilot.resize_terminal(width, 32)
            await wait_for(
                lambda width=width: dashboard.outer_size.width == width and cell_len(bash_line()) == canvas_width(),
                timeout=5,
                pilot=pilot,
                description="timeline canvas resize",
            )
            bash = bash_line()
            assert bash.endswith("4.00 s")
            assert all(cell_len(line) == canvas_width() for line in operation_lines())
            if canvas_width() > view.scrollable_content_region.width:
                await wait_for(
                    lambda: view.max_scroll_x > 0,
                    timeout=5,
                    pilot=pilot,
                    description="timeline horizontal scroll",
                )
            else:
                await wait_for(
                    lambda: view.max_scroll_x == 0,
                    timeout=5,
                    pilot=pilot,
                    description="timeline no horizontal overflow",
                )


@pytest.mark.asyncio
async def test_timeline_model_run_bar_uses_accent_without_recoloring_model_category(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_operations(path)

    async with _DashboardApp().run_test(size=(150, 32)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        analysis = dashboard._analysis
        assert analysis is not None
        dashboard.select_turn(analysis.turns[0].turn_id)

        lines = dashboard.query_one(TrajectoryTextView)._lines
        run_line = next(line for line in lines if re.match(r"Model\s+run\s", line.plain))
        cycle_line = next(line for line in lines if re.match(r"Model\s+\u2502 cycle\s", line.plain))
        console = Console()

        run_label_style = run_line.get_style_at_offset(console, 0)
        run_bar_style = run_line.get_style_at_offset(console, run_line.plain.index("▮"))
        cycle_label_style = cycle_line.get_style_at_offset(console, 0)
        cycle_bar_style = cycle_line.get_style_at_offset(console, cycle_line.plain.index("▮"))

        assert run_label_style.color == cycle_label_style.color
        assert cycle_bar_style.color == cycle_label_style.color
        assert run_bar_style.color == dashboard._semantic_style("accent", "magenta", bold=True).color
        assert run_bar_style.color != run_label_style.color


@pytest.mark.asyncio
async def test_interrupted_retry_renders_one_logical_turn_tab(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
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
        2 * _NS,
        turn_id=first_turn_id,
        payload={"end_reason": "interrupted", "duration_ms": 2_000},
    )
    log.settled(3 * _NS, drained_scopes=[])
    log.add(
        EventType.TURN_STARTED,
        100 * _NS,
        turn_id=retry_turn_id,
        payload={"turn_number": 1, "is_retry": True},
    )
    log.add(
        EventType.TURN_FINISHED,
        103 * _NS,
        turn_id=retry_turn_id,
        payload={"end_reason": "cancelled", "duration_ms": 3_000},
    )
    log.write(path)

    async with _DashboardApp().run_test(size=(120, 32)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)

        analysis = dashboard._analysis
        assert analysis is not None
        assert len(analysis.turns) == 1
        assert analysis.turn(retry_turn_id) is analysis.turns[0]
        dashboard.select_turn(retry_turn_id)
        assert dashboard._selected_turn_id == analysis.turns[0].turn_id
        # Keep the calls adjacent: an intervening layout pass changes the
        # presentation key and lets the Timeline builders normalize the alias,
        # masking a missing select_turn boundary normalization.
        dashboard.select_turn(retry_turn_id)
        assert dashboard._selected_turn_id == analysis.turns[0].turn_id
        await pilot.pause()
        await wait_for(
            lambda: (
                [tab.label.plain for tab in dashboard.query_one("#timeline-turn-tabs", Tabs).query(Tab)] == ["Turn 1"]
            ),
            timeout=5,
            pilot=pilot,
            description="logical turn tab replacement",
        )
        turn_tabs = dashboard.query_one("#timeline-turn-tabs", Tabs)
        assert [tab.label.plain for tab in turn_tabs.query(Tab)] == ["Turn 1"]


@pytest.mark.asyncio
async def test_timeline_operation_selection_has_distinct_presentation_identity(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_operations(path)

    async with _DashboardApp().run_test(size=(150, 32)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        analysis = dashboard._analysis
        assert analysis is not None
        turn = analysis.turns[0]
        view = dashboard.query_one(TrajectoryTextView)
        console = Console()

        dashboard.select_turn(turn.turn_id, "f" * 32)
        bash = next(line for line in view._lines if "Bash (#01234567)" in line.plain)
        approval = next(line for line in view._lines if "approval" in line.plain)
        assert bash.get_style_at_offset(console, 0).reverse is True
        assert approval.get_style_at_offset(console, 0).reverse is not True

        dashboard.select_turn(turn.turn_id, "1" * 32)
        bash = next(line for line in view._lines if "Bash (#01234567)" in line.plain)
        approval = next(line for line in view._lines if "approval" in line.plain)
        assert bash.get_style_at_offset(console, 0).reverse is not True
        assert approval.get_style_at_offset(console, 0).reverse is True


@pytest.mark.asyncio
async def test_vertical_scrollbar_drag_moves_virtualized_content() -> None:
    async with _DashboardApp().run_test(size=(90, 24)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.display = True
        view = dashboard.query_one(TrajectoryTextView)
        view.display = True
        view.set_lines([Text(f"operation {index}") for index in range(200)])
        await pilot.pause()
        assert view.max_scroll_y > 0

        x = view.size.width - 1
        await pilot._post_mouse_events([events.MouseDown], view, offset=(x, 1), button=1)
        await pilot._post_mouse_events([events.MouseMove], view, offset=(x, view.size.height - 2), button=1)
        await pilot._post_mouse_events([events.MouseUp], view, offset=(x, view.size.height - 2), button=1)

        assert view.scroll_offset.y > 0


@pytest.mark.asyncio
async def test_vertical_scroll_repaints_from_the_absolute_content_line() -> None:
    async with _DashboardApp().run_test(size=(90, 24)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.display = True
        view = dashboard.query_one(TrajectoryTextView)
        view.display = True
        view.set_lines([Text(f"operation {index}") for index in range(200)])
        await pilot.pause()

        before = view.render_line(0).text.rstrip()
        view.scroll_to(y=12, animate=False, force=True, immediate=True)
        await pilot.pause()
        after = view.render_line(0).text.rstrip()

        assert before == "operation 0"
        assert after == "operation 12"
        assert before != after
        assert (12, 0, view.scrollable_content_region.width) in view._strips


@pytest.mark.asyncio
async def test_render_line_composes_text_base_style_over_widget_background() -> None:
    async with _StyledDashboardApp().run_test(size=(90, 24)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.display = True
        view = dashboard.query_one(TrajectoryTextView)
        view.display = True
        view.set_lines([Text("base style", style="#ff0000")])
        await pilot.pause()

        content = next(segment for segment in view.render_line(0) if segment.text.startswith("base style"))

        assert content.style is not None
        assert content.style.color is not None
        assert content.style.color.triplet is not None
        assert content.style.color.triplet.hex == "#ff0000"
        assert content.style.bgcolor is not None
        assert content.style.bgcolor.triplet is not None
        assert content.style.bgcolor.triplet.hex == "#123456"


@pytest.mark.asyncio
async def test_render_line_composes_widget_line_spans_zebra_and_selection_styles() -> None:
    async with _StyledDashboardApp().run_test(size=(90, 24)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.display = True
        view = dashboard.query_one(TrajectoryTextView)
        view.display = True
        line = Text("base span", style="#ff0000")
        line.stylize("#00ff00 bold", 5, 9)
        line.stylize("on #654321", 0, len(line))
        line.stylize("reverse", 0, len(line))
        view.set_lines([line])
        await pilot.pause()

        segments = tuple(segment for segment in view.render_line(0) if segment.text.strip())
        base = next(segment for segment in segments if "base" in segment.text)
        span = next(segment for segment in segments if "span" in segment.text)

        assert base.style is not None
        assert base.style.color is not None
        assert base.style.color.triplet is not None
        assert base.style.color.triplet.hex == "#ff0000"
        assert base.style.bgcolor is not None
        assert base.style.bgcolor.triplet is not None
        assert base.style.bgcolor.triplet.hex == "#654321"
        assert base.style.reverse is True
        assert span.style is not None
        assert span.style.color is not None
        assert span.style.color.triplet is not None
        assert span.style.color.triplet.hex == "#00ff00"
        assert span.style.bgcolor is not None
        assert span.style.bgcolor.triplet is not None
        assert span.style.bgcolor.triplet.hex == "#654321"
        assert span.style.bold is True
        assert span.style.reverse is True


@pytest.mark.asyncio
async def test_theme_and_locale_refresh_invalidate_both_dashboard_cache_levels(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_operations(path)

    async with _DashboardApp().run_test(size=(100, 30)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        view = dashboard.query_one(TrajectoryTextView)
        presentation_marker = (
            DashboardTab.OVERVIEW,
            None,
            None,
            -999,
            1,
            1,
            "theme-marker",
            -999,
        )
        strip_marker = (-999, -999, -999)
        dashboard._presentation_cache[presentation_marker] = (Text("stale"),)
        view._strips[strip_marker] = view.render_line(0)

        pilot.app.theme = "textual-light"
        await pilot.pause()

        assert presentation_marker not in dashboard._presentation_cache
        assert strip_marker not in view._strips

        dashboard._presentation_cache[presentation_marker] = (Text("stale"),)
        view._strips[strip_marker] = view.render_line(0)
        dashboard.refresh_localization()

        assert presentation_marker not in dashboard._presentation_cache
        assert strip_marker not in view._strips


@pytest.mark.asyncio
async def test_resize_reflows_presentation_without_reaggregation(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_operations(path)

    async with _DashboardApp().run_test(size=(100, 30)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        assert dashboard._live_timer is not None
        dashboard._live_timer.pause()
        analysis = dashboard._analysis
        generation = dashboard._load_generation
        old_key = dashboard._presentation_key
        old_width = max(cell_len(line.plain) for line in dashboard.query_one(TrajectoryTextView)._lines)

        await pilot.resize_terminal(150, 30)
        # The resize reaches the dashboard as its own widget event after the
        # app-level relayout; poll for the reflow rather than assume one pump.
        await wait_for(
            lambda: dashboard._presentation_key != old_key,
            timeout=5,
            pilot=pilot,
            description="presentation reflow after resize",
        )
        new_width = max(cell_len(line.plain) for line in dashboard.query_one(TrajectoryTextView)._lines)

        assert dashboard._analysis is analysis
        assert dashboard._load_generation == generation
        assert new_width != old_width


@pytest.mark.asyncio
async def test_breakpoints_produce_wide_mid_narrow_and_floor_shapes(tmp_path: Path) -> None:
    path = tmp_path / ".chrys" / "sessions" / "abcd1234" / "trajectory" / "events.jsonl"
    path.parent.mkdir(parents=True)
    _write_p1_operations(path)

    async with _DashboardApp().run_test(size=(180, 400)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        assert dashboard._analysis is not None

        rendered: dict[ResponsiveTier, str] = {}
        for tier, width in (
            (ResponsiveTier.WIDE, 140),
            (ResponsiveTier.MID, 90),
            (ResponsiveTier.NARROW, 70),
            (ResponsiveTier.FLOOR, 50),
        ):
            await pilot.resize_terminal(width, 400)
            dashboard._available_size = Size(width, 400)
            dashboard._clear_presentation_cache()
            dashboard._render_active_view()
            assert dashboard.responsive_tier is tier
            view = dashboard.query_one(TrajectoryTextView)
            rendered[tier] = "\n".join(line.plain for line in view._lines)
            assert view.max_scroll_x == 0
            assert max(cell_len(line.plain) for line in view._lines) <= view.scrollable_content_region.width
            if tier in {ResponsiveTier.WIDE, ResponsiveTier.NARROW}:
                section_titles = (
                    "Time & usage",
                    "Where time went",
                    "Parallelism & busy",
                    "Per-turn time breakdown",
                    "Token usage",
                    "Skill usage",
                    "MCP usage",
                    "Action breakdown",
                    "Failure recovery",
                    "Change verification",
                    "Submission wait (submit → work starts)",
                )
                for title in section_titles:
                    assert any(line.plain.startswith("┌") and title in line.plain for line in view._lines)

        assert "Per-turn time breakdown" in rendered[ResponsiveTier.WIDE]
        assert "Token usage" in rendered[ResponsiveTier.WIDE]
        assert "Action breakdown" in rendered[ResponsiveTier.WIDE]
        assert "Failure recovery" in rendered[ResponsiveTier.WIDE]
        assert "Change verification" in rendered[ResponsiveTier.WIDE]
        assert "Per-turn time breakdown" in rendered[ResponsiveTier.MID]
        assert "Token usage" in rendered[ResponsiveTier.MID]
        assert "Action breakdown" in rendered[ResponsiveTier.MID]
        assert "Change verification" in rendered[ResponsiveTier.MID]
        assert "Per-turn time breakdown" in rendered[ResponsiveTier.NARROW]
        assert "Action breakdown" in rendered[ResponsiveTier.NARROW]
        assert all(label in rendered[ResponsiveTier.NARROW] for label in ("search", "read", "edit", "verify"))
        assert "Per-turn time breakdown" not in rendered[ResponsiveTier.FLOOR]
        assert "Terminal too narrow" in rendered[ResponsiveTier.FLOOR]
        assert "Findings" not in rendered[ResponsiveTier.FLOOR]
        assert "cache read" not in rendered[ResponsiveTier.FLOOR]
        assert "Action breakdown" not in rendered[ResponsiveTier.FLOOR]
        assert "Failure recovery" not in rendered[ResponsiveTier.FLOOR]
        assert "Change verification" not in rendered[ResponsiveTier.FLOOR]
        assert "Submission wait" not in rendered[ResponsiveTier.FLOOR]


@pytest.mark.asyncio
async def test_missing_trajectory_renders_hatched_empty_state_on_every_data_tab(tmp_path: Path) -> None:
    async with _DashboardApp().run_test(size=(100, 30)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", tmp_path / "missing.jsonl")
        await _wait_loaded(dashboard, pilot)
        view = dashboard.query_one(TrajectoryTextView)

        for tab in (DashboardTab.OVERVIEW, DashboardTab.TIMELINE, DashboardTab.INSIGHTS):
            dashboard.query_one("#trajectory-tabs", Tabs).active = tab
            await pilot.pause()
            text = "\n".join(line.plain for line in view._lines)
            assert "No trajectory data is available for this session." in text
            assert "╲" in text
            assert len(view._lines) > 1
            assert "P2" not in text and "P3" not in text
            assert "legacy" not in text.lower()
        assert dashboard.query_one("#timeline-turn-tabs", Tabs).display is False


@pytest.mark.asyncio
async def test_session_without_completed_turns_renders_hatched_empty_state(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    log.write(path)

    async with _DashboardApp().run_test(size=(100, 30)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        view = dashboard.query_one(TrajectoryTextView)

        for tab in (DashboardTab.OVERVIEW, DashboardTab.TIMELINE, DashboardTab.INSIGHTS):
            dashboard.query_one("#trajectory-tabs", Tabs).active = tab
            await pilot.pause()
            text = "\n".join(line.plain for line in view._lines)
            assert "No completed turns are available." in text
            assert "╲" in text
            assert len(view._lines) > 1
            assert "total time" not in text and "Findings" not in text


@pytest.mark.asyncio
async def test_session_with_only_corrupt_lines_exposes_diagnostics_without_turns(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    async with _DashboardApp().run_test(size=(120, 40)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        view = dashboard.query_one(TrajectoryTextView)

        assert "No completed turns are available." in "\n".join(line.plain for line in view._lines)
        await pilot.click("#timeline")
        await pilot.pause()
        assert "No completed turns are available." in "\n".join(line.plain for line in view._lines)
        await pilot.click("#insights")
        await wait_for(
            lambda: "Diagnostics" in "\n".join(line.plain for line in view._lines),
            timeout=5,
            pilot=pilot,
            description="zero-turn diagnostics render",
        )
        text = "\n".join(line.plain for line in view._lines)
        assert "Corrupt line" in text
        assert "No completed turns are available." not in text


@pytest.mark.asyncio
async def test_short_content_hatch_fills_viewport_and_overflowing_content_does_not(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_operations(path)

    async with _DashboardApp().run_test(size=(220, 100)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        view = dashboard.query_one(TrajectoryTextView)
        await wait_for(
            lambda: not view.show_vertical_scrollbar and len(view._lines) == view.scrollable_content_region.height,
            timeout=5,
            pilot=pilot,
            description="hatch fill settle",
        )
        assert set(view._lines[-1].plain) == {"╲"}

        await pilot.resize_terminal(220, 24)
        await wait_for(
            lambda: view.show_vertical_scrollbar,
            timeout=5,
            pilot=pilot,
            description="overflow scrollbar settle",
        )
        assert "╲" not in view._lines[-1].plain


@pytest.mark.asyncio
async def test_hatch_fill_built_during_resize_transition_does_not_poison_the_cache(tmp_path: Path) -> None:
    """The dashboard resizes before its text view reflows; a fill built for the
    old region must not be served for the settled one."""
    path = tmp_path / "events.jsonl"
    _write_operations(path)

    async with _DashboardApp().run_test(size=(220, 100)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        view = dashboard.query_one(TrajectoryTextView)
        await wait_for(
            lambda: not view.show_vertical_scrollbar and len(view._lines) == view.scrollable_content_region.height,
            timeout=5,
            pilot=pilot,
            description="tall fill settle",
        )
        await pilot.resize_terminal(220, 24)
        await wait_for(
            lambda: view.show_vertical_scrollbar,
            timeout=5,
            pilot=pilot,
            description="shrink settle",
        )

        # Replay a grow to a height this session has never rendered: the
        # dashboard's resize handler renders while the text view still has the
        # shrunken region, then the terminal actually grows and only the
        # view's own settle re-render can repair the fill. The dashboard spans
        # the whole test terminal, so its resize event carries the raw size.
        grown_size = Size(220, 90)
        dashboard.on_resize(events.Resize(grown_size, grown_size))
        await pilot.resize_terminal(220, 90)
        await wait_for(
            lambda: not view.show_vertical_scrollbar and len(view._lines) == view.scrollable_content_region.height,
            timeout=5,
            pilot=pilot,
            description="regrown fill settle",
        )
        assert dashboard._available_size == dashboard.size
        assert dashboard._available_size != grown_size
        assert set(view._lines[-1].plain) == {"╲"}


@pytest.mark.asyncio
async def test_dashboard_participates_in_gc_freeze_and_releases_hidden_cache(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_operations(path)

    async with _DashboardApp().run_test(size=(100, 30)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        text_view = dashboard.query_one(TrajectoryTextView)

        assert dashboard.gc_freeze_block_reason() is GcFreezeBlockReason.TRAJECTORY_DASHBOARD_VISIBLE
        dashboard.active_tab = DashboardTab.SESSION_DATA
        assert dashboard.gc_freeze_block_reason() is None
        dashboard.prepare_for_gc_freeze()
        assert isinstance(text_view._strips, DetachedLruCache)
        assert isinstance(dashboard._presentation_cache, DetachedLruCache)
        dashboard.after_gc_freeze()
        dashboard.active_tab = DashboardTab.OVERVIEW
        dashboard.hide_dashboard()
        assert dashboard.gc_freeze_block_reason() is None
        dashboard.prepare_for_gc_freeze()
        assert isinstance(text_view._strips, DetachedLruCache)
        assert isinstance(dashboard._presentation_cache, DetachedLruCache)
        dashboard.after_gc_freeze()
        assert not isinstance(text_view._strips, DetachedLruCache)
        assert dashboard._analyzer is None
        assert dashboard._analysis is None


@pytest.mark.asyncio
async def test_tab_switch_pauses_incremental_scan_and_hide_cancels_and_releases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = Event()
    stopped = Event()

    def slow_load(
        _analyzer: TrajectoryAnalyzer,
        _path: Path,
        *,
        cancel_event: Event | None = None,
    ) -> NoReturn:
        assert cancel_event is not None
        started.set()
        if not cancel_event.wait(timeout=5):
            raise AssertionError("dashboard scan was not cancelled")
        stopped.set()
        raise TrajectoryScanCancelled

    monkeypatch.setattr(TrajectoryAnalyzer, "load", slow_load)

    async with _DashboardApp().run_test(size=(100, 30)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", tmp_path / "events.jsonl")
        await wait_for(started.is_set, timeout=5, pilot=pilot, description="trajectory scan start")
        cancel_event = dashboard._scan_cancel_event
        assert cancel_event is not None
        assert dashboard._live_timer is not None
        assert dashboard._live_timer._active.is_set()

        dashboard.query_one("#trajectory-tabs").active = DashboardTab.INSIGHTS
        await pilot.pause()
        assert not cancel_event.is_set()
        assert not dashboard._live_timer._active.is_set()

        dashboard.hide_dashboard()
        await wait_for(stopped.is_set, timeout=5, pilot=pilot, description="trajectory scan cancellation")
        assert cancel_event.is_set()
        assert dashboard._analyzer is None
        assert dashboard.query_one(TrajectoryTextView)._lines == []


@pytest.mark.asyncio
async def test_long_load_shows_loading_indicator_instead_of_empty_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.jsonl"
    _write_operations(path)
    started = Event()
    release = Event()
    original_load = TrajectoryAnalyzer.load

    def slow_load(analyzer: TrajectoryAnalyzer, load_path: Path, *, cancel_event: Event | None = None) -> object:
        started.set()
        if not release.wait(timeout=5):
            raise AssertionError("dashboard scan was never released")
        return original_load(analyzer, load_path, cancel_event=cancel_event)

    monkeypatch.setattr(TrajectoryAnalyzer, "load", slow_load)

    async with _DashboardApp().run_test(size=(100, 30)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await wait_for(started.is_set, timeout=5, pilot=pilot, description="trajectory scan start")
        await pilot.pause()
        loading_state = dashboard.query_one("#trajectory-loading-state")
        view = dashboard.query_one(TrajectoryTextView)
        assert loading_state.display is True
        assert dashboard.query_one(ChrysLoadingIndicator).display is True
        assert view.display is False

        # Re-renders provoked while the scan is still running (tab switches,
        # resizes) must not surface the "no data" empty state.
        dashboard.query_one("#trajectory-tabs", Tabs).active = DashboardTab.INSIGHTS
        await pilot.pause()
        await pilot.resize_terminal(110, 32)
        await pilot.pause()
        assert view.display is False
        assert "No trajectory data" not in "\n".join(line.plain for line in view._lines)

        release.set()
        await _wait_loaded(dashboard, pilot)
        await pilot.pause()
        assert loading_state.display is False
        assert view.display is True
        assert "Diagnostics" in "\n".join(line.plain for line in view._lines)


@pytest.mark.asyncio
async def test_localized_tab_labels_are_rich_text_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TrajectoryDashboard, "_render_message", lambda self, reference: "label [literal")

    async with _DashboardApp().run_test(size=(100, 30)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        await pilot.pause()

        assert all(tab.label.plain == "label [literal" for tab in dashboard.query(Tab))


@pytest.mark.asyncio
async def test_space_toggles_timeline_between_time_axis_and_dependency_graph(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    _write_operations(path)

    async with _DashboardApp().run_test(size=(150, 40)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        view = dashboard.query_one(TrajectoryTextView)

        await pilot.click("#timeline")
        await wait_for(
            lambda: "Space: dependency graph" in "\n".join(line.plain for line in view._lines),
            timeout=10,
            pilot=pilot,
            description="timeline render",
        )
        timeline = "\n".join(line.plain for line in view._lines)
        assert "Turn 1" in timeline
        assert "Dependency graph" not in timeline
        assert dashboard.query_one("#timeline-turn-tabs", Tabs).display

        await pilot.press("space")
        await wait_for(
            lambda: "Dependency graph · turn 1" in "\n".join(line.plain for line in view._lines),
            timeout=10,
            pilot=pilot,
            description="dependency graph render",
        )
        graph = "\n".join(line.plain for line in view._lines)
        assert "⇠" in graph
        assert "Space: timeline" in graph
        assert "after_tool (register-sess...)" in graph
        assert dashboard.query_one("#timeline-turn-tabs", Tabs).display

        await pilot.press("space")
        await wait_for(
            lambda: "Space: dependency graph" in "\n".join(line.plain for line in view._lines),
            timeout=10,
            pilot=pilot,
            description="timeline restored after second space",
        )
        assert "Dependency graph" not in "\n".join(line.plain for line in view._lines)


@pytest.mark.asyncio
async def test_timeline_preserves_separator_after_full_width_category(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    approval_id = "a" * 32
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.add(
        EventType.APPROVAL_REQUESTED,
        0,
        payload={"approval_request_id": approval_id},
    )
    log.add(
        EventType.APPROVAL_RESOLVED,
        _NS,
        payload={"approval_request_id": approval_id},
    )
    log.add("turn.finished", _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)

    async with _DashboardApp().run_test(size=(150, 32)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        view = dashboard.query_one(TrajectoryTextView)

        await pilot.click("#timeline")
        await wait_for(
            lambda: "Space: dependency graph" in "\n".join(line.plain for line in view._lines),
            timeout=10,
            pilot=pilot,
            description="full-width timeline category render",
        )
        timeline_approval = next(line.plain for line in view._lines if line.plain.startswith("Approval"))
        assert timeline_approval.startswith("Approval approval")

        await pilot.press("space")
        await wait_for(
            lambda: "Dependency graph" in "\n".join(line.plain for line in view._lines),
            timeout=10,
            pilot=pilot,
            description="full-width dependency category render",
        )
        graph_approval = next(line.plain for line in view._lines if line.plain.startswith("Approval"))
        assert graph_approval.startswith("Approval ┄ approval")


@pytest.mark.asyncio
async def test_dependency_graph_marks_adjacent_only_operations_instead_of_fabricating_edges(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog()
    log.coverage()
    log.add("turn.started", 0, payload={"turn_number": 1})
    log.span(
        "tool.operation",
        "a" * 32,
        0,
        _NS,
        start_payload={"tool_name": "first", "tool_kind": "filesystem.read", "argument_fingerprint": "1" * 16},
    )
    log.span(
        "tool.operation",
        "b" * 32,
        _NS,
        2 * _NS,
        start_payload={"tool_name": "second", "tool_kind": "filesystem.read", "argument_fingerprint": "2" * 16},
    )
    log.add("turn.finished", 2 * _NS, payload={"end_reason": "cancelled", "duration_ms": 0})
    log.write(path)

    async with _DashboardApp().run_test(size=(150, 40)) as pilot:
        dashboard = pilot.app.query_one(TrajectoryDashboard)
        dashboard.show_session("session", path)
        await _wait_loaded(dashboard, pilot)
        view = dashboard.query_one(TrajectoryTextView)

        await pilot.click("#timeline")
        await pilot.press("space")
        await wait_for(
            lambda: "Dependency graph" in "\n".join(line.plain for line in view._lines),
            timeout=10,
            pilot=pilot,
            description="dependency graph render for adjacency fixture",
        )
        lines = [line.plain for line in view._lines]
        body = "\n".join(line for line in lines if "adjacent only" not in line)
        assert "⇠" not in body
        assert "└→" not in body
        first_line = next(line for line in lines if "first (#" in line)
        second_line = next(line for line in lines if "second (#" in line)
        assert "┄" in first_line
        assert "┄" in second_line
