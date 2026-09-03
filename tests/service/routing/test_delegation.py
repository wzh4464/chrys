# Copyright (c) 2026 Chrys. All rights reserved.

"""Rendering a code search's candidates as bounded, untrusted evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chrys.service.routing.delegation import (
    MAX_HINT_ITEMS,
    PactRunRequest,
    augment_delta_with_locations,
    build_delegation_reminder,
    build_task_brief,
    localization_hints,
    materialize_pact_request,
)


def _location(index: int, *, reason: str = "touches the token exchange") -> dict[str, object]:
    return {
        "file": f"src/auth/module{index}.py",
        "symbol": f"handler_{index}",
        "start_line": index,
        "end_line": index + 10,
        "role": "primary" if index == 1 else "propagation",
        "reason": reason,
    }


# --------------------------------------------------------------------------
# hints
# --------------------------------------------------------------------------


def test_hints_render_one_line_per_location() -> None:
    rendered = localization_hints([_location(1), _location(2)])

    assert rendered.count("\n") == 1
    assert "src/auth/module1.py:1-11" in rendered
    assert "handler_1" in rendered
    assert "[primary]" in rendered


def test_hints_are_capped_by_item_count() -> None:
    rendered = localization_hints([_location(index) for index in range(1, 30)])

    assert len(rendered.splitlines()) == MAX_HINT_ITEMS


def test_hints_are_capped_by_characters_without_truncating_a_row() -> None:
    """A half-written path is worse than a missing one."""
    rendered = localization_hints([_location(index, reason="x" * 400) for index in range(1, 10)], max_chars=300)

    assert len(rendered) <= 300
    for line in rendered.splitlines():
        assert line.startswith("- src/auth/module")


def test_no_locations_renders_nothing() -> None:
    assert localization_hints([]) == ""


def test_a_location_without_a_file_is_skipped() -> None:
    assert localization_hints([{"symbol": "orphan"}]) == ""


def test_a_reversed_line_span_is_dropped_rather_than_rendered() -> None:
    rendered = localization_hints([{"file": "a.py", "start_line": 90, "end_line": 10}])

    assert "90" not in rendered


# --------------------------------------------------------------------------
# repair reminder
# --------------------------------------------------------------------------


def test_the_delta_comes_first_and_is_never_edited() -> None:
    """The delta is the repair's authority; the search is only an aid to it."""
    delta = "Original clarification delta.\nSecond line."

    rendered = augment_delta_with_locations(delta, [_location(1)])

    assert rendered.startswith(delta)
    assert "untrusted" in rendered.lower()


def test_no_locations_leaves_the_delta_alone() -> None:
    assert augment_delta_with_locations("delta", []) == "delta"


def test_the_appended_table_is_bounded() -> None:
    rendered = augment_delta_with_locations(
        "delta", [_location(index, reason="y" * 400) for index in range(1, 20)], max_chars=500
    )

    assert len(rendered) <= len("delta") + 500 + 200


# --------------------------------------------------------------------------
# task brief
# --------------------------------------------------------------------------


def test_the_brief_keeps_the_original_requirement_as_authority() -> None:
    brief = build_task_brief(
        original_requirement="Add OAuth login",
        clarified_requirement_md="## Clarified\nUse the provider abstraction.",
        locations=[_location(1)],
        baseline="p1",
    )

    assert "Original requirement (authority)" in brief
    assert "Add OAuth login" in brief
    assert "Use the provider abstraction." in brief
    assert "src/auth/module1.py" in brief
    assert "p1" in brief


def test_the_brief_says_what_is_missing_rather_than_omitting_the_section() -> None:
    """A role needs to know a stage degraded, not to silently see less."""
    brief = build_task_brief(
        original_requirement="Add OAuth login",
        clarified_requirement_md=None,
        locations=[],
        baseline="p0",
        warnings=["code localization exceeded 120 seconds"],
    )

    assert "(clarification produced none)" in brief
    assert "(no candidate locations)" in brief
    assert "Degraded stages" in brief
    assert "exceeded 120 seconds" in brief


def test_the_brief_ends_with_exactly_one_newline() -> None:
    brief = build_task_brief(original_requirement="r", clarified_requirement_md="c", locations=[], baseline="none")

    assert brief.endswith("\n")
    assert not brief.endswith("\n\n")


# --------------------------------------------------------------------------
# run request
# --------------------------------------------------------------------------


def test_the_run_request_matches_the_pact_launch_contract() -> None:
    request = PactRunRequest(
        request_id="r1",
        contract_path=".pact-io/chrys-pact/r1/goal-contract.json",
        plan_path=".pact-io/chrys-pact/r1/initial-plan.json",
    )

    assert json.loads(request.to_json()) == {
        "schema": "chrys-pact/run-request/v1",
        "contract_path": ".pact-io/chrys-pact/r1/goal-contract.json",
        "plan_path": ".pact-io/chrys-pact/r1/initial-plan.json",
    }
    assert list(json.loads(request.to_json())) == ["schema", "contract_path", "plan_path"]


def test_materializing_copies_the_pair_into_the_workspace(tmp_path: Path) -> None:
    """The campaign runs in the workspace, so its inputs have to live there."""
    source = tmp_path / "artifacts" / "06-pact-input"
    source.mkdir(parents=True)
    (source / "goal-contract.json").write_text('{"goal": "x"}', encoding="utf-8")
    (source / "initial-plan.json").write_text('{"plan": []}', encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    request = materialize_pact_request(workspace, source, "req-1")

    assert (workspace / request.contract_path).read_text(encoding="utf-8") == '{"goal": "x"}'
    assert (workspace / request.plan_path).read_text(encoding="utf-8") == '{"plan": []}'
    assert not Path(request.contract_path).is_absolute()
    assert request.contract_path.startswith(".pact-io/chrys-pact/req-1/")


def test_materializing_is_repeatable(tmp_path: Path) -> None:
    source = tmp_path / "06-pact-input"
    source.mkdir()
    (source / "goal-contract.json").write_text("{}", encoding="utf-8")
    (source / "initial-plan.json").write_text("{}", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    first = materialize_pact_request(workspace, source, "req-1")
    second = materialize_pact_request(workspace, source, "req-1")

    assert first == second


def test_a_missing_input_pair_raises_rather_than_writing_half(tmp_path: Path) -> None:
    source = tmp_path / "06-pact-input"
    source.mkdir()
    (source / "goal-contract.json").write_text("{}", encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(OSError):
        materialize_pact_request(workspace, source, "req-1")


# --------------------------------------------------------------------------
# delegation reminder
# --------------------------------------------------------------------------


def test_the_reminder_carries_the_exact_json_and_the_rules(tmp_path: Path) -> None:
    request = PactRunRequest("r1", "a/goal-contract.json", "a/initial-plan.json")

    reminder = build_delegation_reminder(
        brief_path=tmp_path / "brief.md",
        brief_summary="Add OAuth login across api and web.",
        baseline="p1",
        request=request,
        pact_tool="chrys_pact",
    )

    assert request.to_json() in reminder
    assert "chrys_pact" in reminder
    assert "brief.md" in reminder
    assert "verbatim" in reminder
    assert "p1" in reminder


def test_the_reminder_is_bounded(tmp_path: Path) -> None:
    reminder = build_delegation_reminder(
        brief_path=tmp_path / "brief.md",
        brief_summary="x" * 20000,
        baseline="p1",
        request=PactRunRequest("r1", "a.json", "b.json"),
        pact_tool="chrys_pact",
        max_chars=800,
    )

    assert len(reminder) == 800
