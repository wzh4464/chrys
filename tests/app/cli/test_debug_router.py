# Copyright (c) 2026 Chrys. All rights reserved.

"""``chrys debug router`` — dry-run classification with no agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chrys.app.cli import debug_router


@pytest.fixture(autouse=True)
def prepared_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the bootstrap: it mutates process-global stdio state."""
    from types import SimpleNamespace

    from chrys.foundation.config.settings import Settings

    monkeypatch.setattr(
        debug_router,
        "bootstrap_runtime",
        lambda **_kwargs: SimpleNamespace(settings=Settings()),
    )


def test_a_trivial_prompt_is_strong_standard_and_fires_no_model(capsys) -> None:
    code = debug_router.main(["--json", "fix typo"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["band"] == "strong_standard"
    assert payload["tiebreaker"]["would_fire"] is False


_SPECIFIED = (
    "Migrate the entire persistence layer. 1) add the pool 2) port every repository "
    "3) backfill. Acceptance criteria: all tests pass. Touch src/db/pool.py, "
    "src/models/user.py and src/migrations/run.py."
)


def test_a_fully_specified_request_is_strong_long_horizon(capsys) -> None:
    code = debug_router.main(["--json", _SPECIFIED])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["band"] == "strong_long_horizon"
    assert payload["signals"]["archetype"] == "mutating_broad"
    assert payload["reason"]


def test_an_ambiguous_request_says_the_tiebreaker_would_fire(capsys) -> None:
    code = debug_router.main(["--json", "refactor the entire auth system"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["band"] == "uncertain"
    assert payload["tiebreaker"]["would_fire"] is True


def test_the_dry_run_never_calls_a_model_without_full(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    called: list[object] = []
    monkeypatch.setattr(debug_router, "_run_tiebreaker", lambda *a, **k: called.append(a) or {})

    debug_router.main(["--json", "refactor the entire auth system"])

    assert called == []


def test_full_reports_an_unavailable_model_instead_of_raising(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        debug_router,
        "_run_tiebreaker",
        lambda *_a, **_k: {"failure": "unavailable", "detail": "no model profile resolved"},
    )

    code = debug_router.main(["--json", "--full", "refactor the entire auth system"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["tiebreaker"]["failure"] == "unavailable"


def test_readiness_is_probed_in_the_named_workspace(tmp_path: Path, capsys) -> None:
    (tmp_path / "tests").mkdir()

    debug_router.main(["--json", "-C", str(tmp_path), "fix typo"])

    payload = json.loads(capsys.readouterr().out)
    assert payload["readiness"]["has_tests"] is True
    assert payload["readiness"]["verify_command_configured"] is False
    assert payload["readiness"]["pact_ready"] is False


def test_a_prompt_can_come_from_a_file(tmp_path: Path, capsys) -> None:
    task = tmp_path / "task.md"
    task.write_text("what does this do?", encoding="utf-8")

    debug_router.main(["--json", "--task", str(task)])

    assert json.loads(capsys.readouterr().out)["signals"]["question_like"] is True


def test_both_a_prompt_and_a_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        debug_router.main(["hello", "--task", str(tmp_path / "task.md")])


def test_neither_a_prompt_nor_a_file_is_rejected() -> None:
    with pytest.raises(SystemExit):
        debug_router.main([])


def test_human_output_names_the_band_and_the_reason(capsys) -> None:
    debug_router.main(["refactor the entire auth system"])

    out = capsys.readouterr().out
    assert "band" in out
    assert "uncertain" in out
    assert "tiebreaker" in out
