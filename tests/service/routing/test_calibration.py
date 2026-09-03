# Copyright (c) 2026 Chrys. All rights reserved.

"""The heuristic classifier's calibration gate.

Precision is weighted far above recall on purpose: promoting a turn that did
not need it costs a whole governed campaign, while missing one costs an
ordinary turn the user can escalate with ``/longrun``.
"""

from __future__ import annotations

import json
from pathlib import Path

from chrys.service.routing.classifier import RouteBand, band_for, extract_prompt_signals, prompt_score

_FIXTURE = Path(__file__).parent / "fixtures" / "calibration.jsonl"
_GATE = Path(__file__).parent / "gate.json"
_LONG_BANDS = {RouteBand.LEAN_LONG_HORIZON, RouteBand.STRONG_LONG_HORIZON}


def _rows() -> list[dict[str, str]]:
    return [json.loads(line) for line in _FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


def _predict(prompt: str) -> RouteBand:
    return band_for(prompt_score(extract_prompt_signals(prompt))[0])


def test_the_sample_is_bilingual_and_balanced() -> None:
    rows = _rows()

    assert len(rows) == 60
    chinese = [row for row in rows if any("一" <= character <= "鿿" for character in row["prompt"])]
    assert 25 <= len(chinese) <= 35, "the sample must not become mostly one language"
    assert {row["expected"] for row in rows} <= {"standard", "lean_long_horizon", "strong_long_horizon"}
    assert all(row["rationale"] for row in rows), "every row states why it is labelled that way"


def test_calibration_gate() -> None:
    rows = _rows()
    gate = json.loads(_GATE.read_text(encoding="utf-8"))
    predicted = {row["prompt"]: _predict(row["prompt"]) for row in rows}

    true_positive = sum(1 for row in rows if row["expected"] != "standard" and predicted[row["prompt"]] in _LONG_BANDS)
    false_positive = sum(1 for row in rows if row["expected"] == "standard" and predicted[row["prompt"]] in _LONG_BANDS)
    # An uncertain verdict is not a miss: it is exactly the case the LLM
    # tiebreaker exists for, so it does not count against recall.
    false_negative = sum(
        1
        for row in rows
        if row["expected"] != "standard"
        and predicted[row["prompt"]] not in _LONG_BANDS
        and predicted[row["prompt"]] is not RouteBand.UNCERTAIN
    )
    strong_false_positive = sum(
        1
        for row in rows
        if row["expected"] != "strong_long_horizon" and predicted[row["prompt"]] is RouteBand.STRONG_LONG_HORIZON
    )

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0

    assert precision >= gate["precision_min"], f"precision {precision:.2f}: {_report(rows, predicted)}"
    assert recall >= gate["recall_min"], f"recall {recall:.2f}: {_report(rows, predicted)}"
    assert strong_false_positive <= gate["strong_false_positives_max"], _report(rows, predicted)


def test_no_acknowledgement_is_ever_promoted() -> None:
    """The cheapest possible mistake to make, and the most annoying."""
    rows = [row for row in _rows() if row["rationale"] == "acknowledgement"]

    assert rows
    assert all(_predict(row["prompt"]) is RouteBand.STRONG_STANDARD for row in rows)


def test_every_fully_specified_request_reaches_the_strong_band() -> None:
    rows = [row for row in _rows() if row["expected"] == "strong_long_horizon"]

    assert len(rows) == 10
    assert all(_predict(row["prompt"]) is RouteBand.STRONG_LONG_HORIZON for row in rows)


def _report(rows: list[dict[str, str]], predicted: dict[str, RouteBand]) -> str:
    lines = [
        f"  {row['expected']:>20s} -> {predicted[row['prompt']].value:<20s} {row['prompt'][:60]}"
        for row in rows
        if (row["expected"] == "standard") != (predicted[row["prompt"]] not in _LONG_BANDS)
    ]
    return "\n" + "\n".join(lines) if lines else "(no disagreements)"
