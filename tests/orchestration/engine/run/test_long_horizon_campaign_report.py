# Copyright (c) 2026 Chrys. All rights reserved.

"""The delegation pass reads the campaign's outcome from the tool result."""

from __future__ import annotations

import pytest

from chrys.orchestration.engine.run.long_horizon import parse_campaign_report

_SUMMARY = """PACT Campaign result
status: completed
campaign_id: chrys-pact-73a39b8b6f7547ac8a1438df5ac53f06
revision: 1
next_action: none
artifacts: .pact/runtime/campaigns/chrys-pact-73a39b8b6f7547ac8a1438df5ac53f06"""


def test_the_servers_summary_text_is_a_report() -> None:
    assert parse_campaign_report(_SUMMARY) == {
        "status": "completed",
        "campaign_id": "chrys-pact-73a39b8b6f7547ac8a1438df5ac53f06",
        "artifact": ".pact/runtime/campaigns/chrys-pact-73a39b8b6f7547ac8a1438df5ac53f06",
    }


def test_a_blocked_campaign_reports_blocked() -> None:
    text = _SUMMARY.replace("status: completed", "status: blocked").replace(
        "next_action: none", "next_action: manager_protocol_error"
    )

    report = parse_campaign_report(text)

    assert report is not None
    assert report["status"] == "blocked"


def test_prose_around_the_block_is_ignored() -> None:
    text = "The campaign finished.\n\n" + _SUMMARY + "\n\nInspect the artifacts for details."

    report = parse_campaign_report(text)

    assert report is not None
    assert report["status"] == "completed"


def test_a_json_report_is_still_accepted() -> None:
    assert parse_campaign_report('{"status": "completed", "campaign_id": "c1", "artifact": "a"}') == {
        "status": "completed",
        "campaign_id": "c1",
        "artifact": "a",
    }


@pytest.mark.parametrize(
    "text",
    ["", "Error: The ACP agent refused the request.", "PACT Campaign result\nrevision: 1", "status: completed"],
)
def test_anything_without_a_reported_status_is_not_a_report(text: str) -> None:
    assert parse_campaign_report(text) is None
