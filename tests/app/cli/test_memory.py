# Copyright (c) 2026 Chrys. All rights reserved.

"""``chrys memory doctor | sweep | init``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chrys.app.cli import memory as memory_cli
from chrys.kernel import Content, Message
from chrys.service.memory.writeback import WATERMARK_KEY, WritebackOutcome
from chrys.service.state.serializers import serialize_state
from chrys.service.state.store import SESSION_FILE_NAME


def _tool_turn(index: int) -> list[Message]:
    return [
        Message("user", [Content.from_text(f"Fix failure {index}")]),
        Message("assistant", [Content.from_function_call(f"shell-{index}", "shell", arguments={"command": "ls"})]),
        Message("tool", [Content.from_function_result(f"shell-{index}", result=f"done {index}")]),
        Message("assistant", [Content.from_text(f"Fixed {index}.")]),
    ]


def _write_session(sessions_dir: Path, session_id: str, *, turns: int, watermark: int) -> Path:
    directory = sessions_dir / session_id
    directory.mkdir(parents=True, exist_ok=True)
    messages: list[Message] = []
    for index in range(1, turns + 1):
        messages.extend(_tool_turn(index))
    state = serialize_state({"messages": messages, "compressed_msgs": [], "turn_counter": turns})
    state[WATERMARK_KEY] = watermark
    path = directory / SESSION_FILE_NAME
    path.write_text(json.dumps({"state": state}), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def prepare_runtime_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub the environment bootstrap: it mutates process-global stdio state."""
    calls: list[str] = []

    def prepare() -> None:
        calls.append("prepare")

    monkeypatch.setattr(memory_cli, "_prepare_runtime", prepare)
    return calls


def test_every_subcommand_bootstraps_the_runtime_first(
    sessions_root: Path,
    prepare_runtime_calls: list[str],
) -> None:
    memory_cli.main(["sweep", "--dry-run", "--idle-seconds", "0"])

    assert prepare_runtime_calls == ["prepare"]


@pytest.fixture
def sessions_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "root"
    (root / "sessions").mkdir(parents=True)
    monkeypatch.setenv("CHRYS_SESSION_ROOT_DIR", str(root))
    return root / "sessions"


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------


def test_sweep_dry_run_lists_only_sessions_that_are_behind(
    sessions_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_session(sessions_root, "behind1", turns=3, watermark=1)
    _write_session(sessions_root, "behind2", turns=2, watermark=0)
    _write_session(sessions_root, "current", turns=2, watermark=2)

    code = memory_cli.main(["sweep", "--dry-run", "--idle-seconds", "0"])

    out = capsys.readouterr().out
    assert code == 0
    assert "behind1" in out
    assert "behind2" in out
    assert "current" not in out


def test_sweep_dry_run_writes_nothing(sessions_root: Path) -> None:
    path = _write_session(sessions_root, "behind1", turns=3, watermark=1)
    before = path.read_text(encoding="utf-8")

    memory_cli.main(["sweep", "--dry-run", "--idle-seconds", "0"])

    assert path.read_text(encoding="utf-8") == before


def test_sweep_deposits_and_advances_the_persisted_watermark(
    sessions_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_session(sessions_root, "behind1", turns=3, watermark=1)
    seen: list[int] = []

    def _deposit(_session_file: Path, **kwargs: Any) -> WritebackOutcome:
        seen.append(kwargs["watermark"])
        return WritebackOutcome(deposited=(2, 3), failed=None, watermark=3)

    monkeypatch.setattr(memory_cli, "deposit_pending_turns", _deposit)

    code = memory_cli.main(["sweep", "--idle-seconds", "0"])

    assert code == 0
    assert seen == [1]
    state = json.loads(path.read_text(encoding="utf-8"))["state"]
    assert state[WATERMARK_KEY] == 3


def test_sweep_skips_sessions_touched_more_recently_than_the_idle_window(
    sessions_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_session(sessions_root, "fresh", turns=3, watermark=0)

    code = memory_cli.main(["sweep", "--dry-run", "--idle-seconds", "3600"])

    assert code == 0
    assert "fresh" not in capsys.readouterr().out


def test_sweep_reports_a_failed_turn_without_advancing_past_it(
    sessions_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_session(sessions_root, "behind1", turns=3, watermark=0)
    monkeypatch.setattr(
        memory_cli,
        "deposit_pending_turns",
        lambda _f, **_k: WritebackOutcome(deposited=(1,), failed=2, watermark=1),
    )

    code = memory_cli.main(["sweep", "--idle-seconds", "0"])

    assert code == 1
    assert "turn 2" in capsys.readouterr().out
    assert json.loads(path.read_text(encoding="utf-8"))["state"][WATERMARK_KEY] == 1


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def test_doctor_reports_every_missing_prerequisite(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in (
        "CONTEXTGRAPH_NEO4J_URI",
        "CONTEXTGRAPH_NEO4J_PASSWORD",
        "CONTEXTGRAPH_REPO",
        "CONTEXTGRAPH_EMBEDDING_API_KEY",
        "OPENAI_API_KEY",
        "NEO4J_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)

    code = memory_cli.main(["doctor"])

    out = capsys.readouterr().out
    assert code == 2
    assert "CONTEXTGRAPH_NEO4J_URI" in out


def test_doctor_json_output_is_machine_readable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("CONTEXTGRAPH_NEO4J_URI", raising=False)

    code = memory_cli.main(["doctor", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["ok"] is False
    assert any(check["ok"] is False for check in payload["checks"])


# --------------------------------------------------------------------------
# init
# --------------------------------------------------------------------------


def test_init_refuses_a_missing_dump_without_starting_anything(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    started: list[object] = []
    monkeypatch.setattr(memory_cli.subprocess, "run", lambda *a, **k: started.append(a))

    code = memory_cli.main(["init", "--import", str(tmp_path / "absent.dump")])

    assert code == 2
    assert started == []
    assert "absent.dump" in capsys.readouterr().err


def test_init_requires_a_contextgraph_checkout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("CONTEXTGRAPH_REPO", raising=False)
    started: list[object] = []
    monkeypatch.setattr(memory_cli.subprocess, "run", lambda *a, **k: started.append(a))

    code = memory_cli.main(["init"])

    assert code == 2
    assert started == []
    assert "CONTEXTGRAPH_REPO" in capsys.readouterr().err


def test_unknown_subcommand_is_rejected() -> None:
    with pytest.raises(SystemExit):
        memory_cli.main(["nonsense"])
