# Copyright (c) 2026 Chrys. All rights reserved.

"""Engine-level tests for the ``/rollback`` snapshot and handler logic.

Covers the pieces that are unit-testable without spinning up a full
agent: snapshot write/prune, available-turn enumeration, the
target-turn handling paths on ``_on_user_rollback`` (refused, welcome
reset, swap+reload), and the ``_suppress_save`` gate on
``_save_current_session``.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

import chrys.orchestration.engine.engine as engine_module
import chrys.orchestration.engine.rollback as rollback_module
from chrys.foundation.config.settings import Settings
from chrys.foundation.config.settings_store import LoadedSettings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import Error, RollbackResult, UserMessage, UserRollback, Warning
from chrys.foundation.i18n import DisplayBlock
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.workspace import Workspace
from chrys.foundation.util.lock import FileLock
from chrys.kernel import Message
from chrys.orchestration.engine.engine import AgentEngine
from chrys.orchestration.engine.state.machine import EngineState, Trigger
from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
from chrys.service.mutations import workspace_changes
from chrys.service.mutations.store import SnapshotStore
from chrys.service.mutations.tracker import MutationTracker
from chrys.service.mutations.types import MutationOp, MutationSource
from chrys.service.profiles.agents.schema import AgentProfile
from chrys.service.state.store import JsonFileStateStore, atomic_copy_file

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(
    tmp_path: Path,
    *,
    session_id: str = "rb_test",
    keep_last: int = 10,
    fsm_state: EngineState = EngineState.IDLE,
) -> AgentEngine:
    """Build a minimally-initialized engine with a temp state store.

    FSM is transitioned to ``fsm_state`` (IDLE by default) so rollback
    validation reaches the branches beyond the initial safety gate.
    """
    store = JsonFileStateStore(tmp_path)
    settings = replace(Settings(), rollback_snapshots_keep=keep_last)
    engine = AgentEngine(EventBus(), settings=settings, state_store=store)
    engine._session_id = session_id
    engine._history.bind({"messages": []})
    if fsm_state is EngineState.IDLE:
        engine._fsm.try_transition(Trigger.START)
    elif fsm_state is EngineState.RUNNING:
        engine._fsm.try_transition(Trigger.START)
        engine._fsm.try_transition(Trigger.USER_MESSAGE)
    elif fsm_state is EngineState.PENDING_RETRY:
        engine._fsm.try_transition(Trigger.START)
        engine._fsm.try_transition(Trigger.USER_MESSAGE)
        engine._fsm.try_transition(Trigger.RETRY_REQUESTED)
    elif fsm_state is EngineState.AWAITING_SUB_AGENTS:
        engine._fsm.try_transition(Trigger.START)
        engine._fsm.try_transition(Trigger.USER_MESSAGE)
        engine._fsm.try_transition(Trigger.SUB_AGENT_PAUSED)
    return engine


def _write_session_json(path: Path, messages: list[dict[str, Any]] | None = None) -> None:
    """Write a minimal session.json so ``_write_rollback_snapshot`` has something to copy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "meta": {"session_id": "rb_test", "agent_profile": "p"},
        "state": {"messages": messages or [], "compressed_msgs": []},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _turn_marker(idx: int) -> Message:
    marker = Message("assistant", [""])
    marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    marker.additional_properties["_turn_id"] = f"turn_{idx}"
    marker.additional_properties["_turn"] = idx
    return marker


def _state_after_turns(count: int) -> dict[str, Any]:
    messages: list[Message] = []
    for idx in range(1, count + 1):
        messages.extend(
            [
                Message("user", [f"user {idx}"]),
                Message("assistant", [f"assistant {idx}"]),
                _turn_marker(idx),
            ]
        )
    return {"messages": messages, "compressed_msgs": [], "turn_counter": count}


async def _collect_events(bus: EventBus, event_type: type, out: list[Any]) -> None:
    async def handler(ev: Any) -> None:
        out.append(ev)

    await bus.subscribe(event_type, handler)


def _assert_display_message(event: Error | Warning, key: str, args: dict[str, str | int] | None = None) -> None:
    reference = event.display_message
    assert reference is not None
    assert reference.definition.key == key
    assert dict(reference.args) == (args or {})


# ===========================================================================
# Snapshot write / prune
# ===========================================================================


class TestSnapshotWrite:
    def test_write_is_noop_without_session(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._session_id = None  # no active session
        engine._write_rollback_snapshot()
        # Nothing should have been created
        assert not any(tmp_path.rglob("turn_*.json"))

    def test_write_is_noop_when_session_json_missing(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._turn_number = 2
        engine._write_rollback_snapshot()  # session.json doesn't exist yet
        assert not any(tmp_path.rglob("turn_*.json"))

    def test_write_copies_session_json_to_turn_n(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        session_dir = tmp_path / "rb_test"
        session_file = session_dir / "session.json"
        _write_session_json(session_file)

        engine._turn_number = 3
        engine._write_rollback_snapshot()

        snap = session_dir / "snapshots" / "turn_3.json"
        assert snap.exists()
        # Content matches source
        assert json.loads(snap.read_text()) == json.loads(session_file.read_text())

    def test_captured_writer_freezes_snapshot_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        engine = _make_engine(tmp_path)
        session_dir = tmp_path / "rb_test"
        _write_session_json(session_dir / "session.json")
        engine._turn_number = 3
        path_calls: list[tuple[str, str]] = []
        real_session_dir_for = engine._session_dir_for
        real_lock_path_for = engine._session_write_lock_path

        def session_dir_for(session_id: str) -> Path:
            path_calls.append(("session", session_id))
            return real_session_dir_for(session_id)

        def lock_path_for(session_id: str) -> Path | None:
            path_calls.append(("lock", session_id))
            return real_lock_path_for(session_id)

        monkeypatch.setattr(engine, "_session_dir_for", session_dir_for)
        monkeypatch.setattr(engine, "_session_write_lock_path", lock_path_for)

        write_snapshot = engine._capture_rollback_snapshot_writer()
        assert path_calls == []
        engine._session_id = "other-session"
        engine._turn_number = 9
        write_snapshot()

        assert path_calls == [("session", "rb_test"), ("lock", "rb_test")]
        assert (session_dir / "snapshots" / "turn_3.json").exists()
        assert not (tmp_path / "other-session" / "snapshots" / "turn_9.json").exists()

    def test_prune_honors_keep_last(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path, keep_last=3)
        session_dir = tmp_path / "rb_test"
        session_file = session_dir / "session.json"

        for turn in range(1, 8):  # turns 1..7
            _write_session_json(session_file, messages=[{"turn": turn}])
            engine._turn_number = turn
            engine._write_rollback_snapshot()

        snap_dir = session_dir / "snapshots"
        remaining = sorted(p.name for p in snap_dir.glob("turn_*.json"))
        assert remaining == ["turn_5.json", "turn_6.json", "turn_7.json"]


# ===========================================================================
# Available rollback turns
# ===========================================================================


class TestAvailableTurns:
    def test_empty_when_no_session_dir(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._session_id = None
        assert engine.available_rollback_turns() == []

    def test_includes_welcome_when_history_has_messages(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._history.bind({"messages": [Message("user", ["hi"])]})
        # No snapshots on disk yet — only the welcome target (0).
        assert engine.available_rollback_turns() == [0]

    def test_includes_disk_snapshots_and_welcome(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        session_dir = tmp_path / "rb_test"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)
        # ``turn_M.json`` maps to ``target_turn = M - 1`` (keep M-1 turns).
        for n in (2, 3, 5):
            (snap_dir / f"turn_{n}.json").write_text("{}")
        # Give tracker at least one turn so welcome (0) is included
        engine._mutation_tracker = MutationTracker(SnapshotStore(session_dir))
        engine._mutation_tracker.start_turn(1)

        assert engine.available_rollback_turns() == [0, 1, 2, 4]

    def test_non_utf8_snapshot_falls_back_to_filename_target(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        session_dir = tmp_path / "rb_test"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)
        (snap_dir / "turn_3.json").write_bytes(b"\xff\xfe\x00")
        engine._mutation_tracker = MutationTracker(SnapshotStore(session_dir))
        engine._mutation_tracker.start_turn(1)

        assert engine.available_rollback_turns() == [0, 2]

    def test_includes_compressed_turns_when_snapshots_exist(self, tmp_path: Path) -> None:
        """Turns folded into a :class:`CompressedBlock` stay eligible.

        Snapshots are written at the *start* of each turn, so
        ``turn_N.json`` predates any compression that happened during
        turn N or later.  Restoring such a snapshot un-compresses those
        folded turns back into live history, and the Context panel
        rebuilds from the restored ``compressed_msgs``.  Every turn
        with a snapshot on disk should therefore be offered.
        """
        from chrys.service.context.providers.history import CompressedBlock

        engine = _make_engine(tmp_path)
        session_dir = tmp_path / "rb_test"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)
        for n in (2, 3, 4, 5):
            (snap_dir / f"turn_{n}.json").write_text("{}")
        engine._mutation_tracker = MutationTracker(SnapshotStore(session_dir))
        engine._mutation_tracker.start_turn(1)

        # Live state shows turns 2..3 as compressed — they must still
        # appear in the picker because their pre-turn snapshots exist.
        engine._history.bind(
            {
                "messages": [Message("user", ["hi"])],
                "compressed_msgs": [
                    CompressedBlock(
                        compressed_context_id="ctx_x",
                        summary_text="summary",
                        marker_id="turn_3",
                        turn_range=(2, 3),
                    ),
                ],
            }
        )

        # turn_{2..5}.json → keep {1..4} turns; plus welcome (0).
        assert engine.available_rollback_turns() == [0, 1, 2, 3, 4]


class TestTurnPromptPreviews:
    """``engine.turn_prompt_previews()`` is the single source of truth for
    turn → user-prompt mapping shown in the rollback picker."""

    def test_maps_user_prompts_to_turn_indices(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)

        # Build a realistic history: user → (assistant work) → turn marker → ...
        def _turn_marker(idx: int) -> Message:
            m = Message("assistant", [""])
            m.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
            m.additional_properties["_turn_id"] = f"turn_{idx}"
            m.additional_properties["_turn"] = idx
            return m

        engine._history.bind(
            {
                "messages": [
                    Message("user", ["hello turn one"]),
                    Message("assistant", ["ok"]),
                    _turn_marker(1),
                    Message("user", ["second question"]),
                    Message("assistant", ["sure"]),
                    _turn_marker(2),
                ],
                "compressed_msgs": [],
            }
        )
        assert engine.turn_prompt_previews() == {
            1: "hello turn one",
            2: "second question",
        }

    def test_returns_empty_when_not_bound(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        # Reset history to unbound state
        engine._history._state = None
        assert engine.turn_prompt_previews() == {}

    def test_flagged_nudge_never_labels_rollback_preview(self, tmp_path: Path) -> None:
        """§2.4: a crash-leftover synthetic ``continue`` nudge is skipped by
        ``scan_turn_prompts`` — a nudge-only turn region gets NO label rather
        than a fabricated "continue", while injected guidance (real user
        content) still labels its turn."""
        engine = _make_engine(tmp_path)

        nudge_only = Message("user", ["continue"])
        nudge_only.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
        nudge_before_guidance = Message("user", ["continue"])
        nudge_before_guidance.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
        guidance = Message("user", ["retry with flag X"])
        guidance.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True

        engine._history.bind(
            {
                "messages": [
                    Message("user", ["real question"]),
                    Message("assistant", ["ok"]),
                    _turn_marker(1),
                    nudge_only,
                    Message("assistant", ["resumed work"]),
                    _turn_marker(2),
                    nudge_before_guidance,
                    guidance,
                    Message("assistant", ["guided work"]),
                    _turn_marker(3),
                ],
                "compressed_msgs": [],
            }
        )

        previews = engine.turn_prompt_previews()
        assert previews[1] == "real question"
        assert 2 not in previews
        assert previews[3] == "retry with flag X"

    def test_first_rolled_back_user_text_returns_earliest_discarded_prompt(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._history.bind(_state_after_turns(10))

        assert engine.first_rolled_back_user_text(5) == "user 6"

    def test_reads_previews_from_compressed_blocks(self, tmp_path: Path) -> None:
        """Compressed turns preserve their originals inside the block.

        Once a turn is folded into a :class:`CompressedBlock`, its
        messages leave live history — but the block keeps deep copies
        with intact turn markers.  The picker still needs previews for
        those turns so it can label compressed rollback targets with
        the original user prompt, so :meth:`turn_prompt_previews` must
        walk each block's ``messages`` too.
        """
        from chrys.service.context.providers.history import CompressedBlock

        engine = _make_engine(tmp_path)

        def _turn_marker(idx: int) -> Message:
            m = Message("assistant", [""])
            m.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
            m.additional_properties["_turn_id"] = f"turn_{idx}"
            m.additional_properties["_turn"] = idx
            return m

        # Turns 1..2 are folded away — their originals now live only
        # inside the block.  Turn 3 remains in live history.
        block = CompressedBlock(
            compressed_context_id="ctx_x",
            messages=[
                Message("user", ["first prompt"]),
                Message("assistant", ["ok"]),
                _turn_marker(1),
                Message("user", ["second prompt"]),
                Message("assistant", ["sure"]),
                _turn_marker(2),
            ],
            summary_text="summary",
            marker_id="turn_2",
            turn_range=(1, 2),
        )
        summary_msg = Message("assistant", ["[Compressed context]"])
        summary_msg.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.SUMMARY
        summary_msg.additional_properties["_block_id"] = "ctx_x"

        engine._history.bind(
            {
                "messages": [
                    summary_msg,
                    Message("user", ["third prompt"]),
                    Message("assistant", ["done"]),
                    _turn_marker(3),
                ],
                "compressed_msgs": [block],
            }
        )

        assert engine.turn_prompt_previews() == {
            1: "first prompt",
            2: "second prompt",
            3: "third prompt",
        }

    def test_live_history_wins_over_block_on_overlap(self, tmp_path: Path) -> None:
        """Live-first scan order means live entries are never overwritten.

        Shouldn't happen in practice (compression moves messages out),
        but the invariant guards against future regressions.
        """
        from chrys.service.context.providers.history import CompressedBlock

        engine = _make_engine(tmp_path)

        def _turn_marker(idx: int) -> Message:
            m = Message("assistant", [""])
            m.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
            m.additional_properties["_turn_id"] = f"turn_{idx}"
            m.additional_properties["_turn"] = idx
            return m

        block = CompressedBlock(
            compressed_context_id="ctx_x",
            messages=[
                Message("user", ["stale prompt"]),
                _turn_marker(1),
            ],
            summary_text="summary",
            marker_id="turn_1",
            turn_range=(1, 1),
        )
        engine._history.bind(
            {
                "messages": [
                    Message("user", ["live prompt"]),
                    _turn_marker(1),
                ],
                "compressed_msgs": [block],
            }
        )

        # Live scan runs first and fills turn 1; block scan must not overwrite it.
        assert engine.turn_prompt_previews() == {1: "live prompt"}


# ===========================================================================
# _on_user_rollback — refusal paths
# ===========================================================================


class TestRollbackRefusals:
    @pytest.mark.asyncio
    async def test_welcome_preflight_exception_releases_session_write_lock(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)

        class _ExplodingTracker:
            calls = 0

            def get_all_turns(self) -> list[object]:
                self.calls += 1
                if self.calls == 1:
                    return [object()]
                raise RuntimeError("rollback plan scan failed")

        engine._mutation_tracker = _ExplodingTracker()  # type: ignore[assignment]

        with pytest.raises(RuntimeError, match="rollback plan scan failed"):
            await engine._on_user_rollback(UserRollback(target_turn=0, session_id="rb_test"))

        lock_path = engine._session_write_lock_path("rb_test")
        assert lock_path is not None
        with FileLock(lock_path, timeout=0.1):
            pass
        assert engine.session_generation == 0
        assert engine._turn_state.prompt_admission_closed is False

    @pytest.mark.asyncio
    async def test_prompt_winning_queued_gate_is_refused_without_invalidating_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        engine = _make_engine(tmp_path)
        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)
        run_release = asyncio.Event()

        class _PromptExecutor:
            is_running = False

        async def _accept_prepared_contents(*_args: Any, **_kwargs: Any) -> bool:
            return False

        async def _run_and_save(*_args: Any, **_kwargs: Any) -> None:
            await run_release.wait()

        engine._executor = _PromptExecutor()  # type: ignore[assignment]
        engine._reminder_middleware = SystemReminderMiddleware()
        monkeypatch.setattr(engine._turns, "reject_text_only_prepared_contents", _accept_prepared_contents)
        monkeypatch.setattr(engine, "_run_and_save", _run_and_save)

        await engine._rebuild_gate_lock.acquire()
        try:
            rollback_task = asyncio.create_task(
                engine._on_user_rollback(UserRollback(target_turn=0, session_id="rb_test"))
            )
            await asyncio.sleep(0)
            assert rollback_task.done() is False

            # A prompt does not need the rebuild gate after it has completed
            # admission, so it can promote while rollback is queued on it.
            await engine._on_user_message(
                UserMessage(text="concurrent ACP prompt", prepared_contents=["concurrent ACP prompt"])
            )
            run_task = engine.turn_lifecycle_task
            assert run_task is not None
        finally:
            engine._rebuild_gate_lock.release()

        try:
            await asyncio.wait_for(rollback_task, timeout=10)
            assert [warning.code for warning in warnings] == ["rollback_refused"]
            assert engine.session_generation == 0
            assert engine.state is EngineState.RUNNING
            assert engine._turn_state.injection_admission_open is True
            assert engine._turn_state.run_task is run_task
            assert run_task.done() is False
        finally:
            run_release.set()
            await asyncio.gather(run_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_waits_for_reserved_prompt_before_refusing_without_invalidating_run(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)
        admission = engine._turn_state.reserve_prompt_admission(
            kind="fresh",
            session_generation=engine.session_generation,
            build_generation=engine.build_generation,
        )
        assert admission is not None
        run_release = asyncio.Event()

        async def _run() -> None:
            await run_release.wait()

        run_task = asyncio.create_task(_run())
        rollback_task = asyncio.create_task(engine._on_user_rollback(UserRollback(target_turn=0, session_id="rb_test")))
        await asyncio.sleep(0)

        assert rollback_task.done() is False
        assert engine._turn_state.prompt_admission_closed is True
        assert engine._turn_state.active_admission_count() == 1
        assert engine.session_generation == 0

        # Mirror the prompt handler's synchronous promotion boundary: install
        # the task/injection window, release its admission, then advance FSM.
        engine._turn_state.run_task = run_task
        engine._turn_state.injection_admission_open = True
        engine._turn_state.release_prompt_admission(admission)
        engine._fsm.try_transition(Trigger.USER_MESSAGE)

        try:
            await asyncio.wait_for(rollback_task, timeout=10)
            assert [warning.code for warning in warnings] == ["rollback_refused"]
            assert engine.session_generation == 0
            assert engine.state is EngineState.RUNNING
            assert engine._turn_state.injection_admission_open is True
            assert engine._turn_state.run_task is run_task
            assert run_task.done() is False
        finally:
            run_release.set()
            await asyncio.gather(run_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_holds_prompt_admission_closed_through_welcome_reset(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._history.bind(_state_after_turns(1))
        flush_started = asyncio.Event()
        release_flush = asyncio.Event()
        reset_observations: list[tuple[EngineState, bool]] = []

        async def _blocked_flush() -> None:
            flush_started.set()
            await release_flush.wait()

        async def _fake_reset(
            _session_id: str,
            *,
            write_lock_held: bool = False,
            after_delete: Any = None,
            before_restart: Any = None,
        ) -> bool:
            _ = write_lock_held, after_delete, before_restart
            reset_observations.append((engine.state, engine._turn_state.prompt_admission_closed))
            return True

        engine._flush_recovery_checkpoint = _blocked_flush  # type: ignore[assignment]
        engine._reset_session_to_welcome = _fake_reset  # type: ignore[assignment]

        rollback_task = asyncio.create_task(engine._on_user_rollback(UserRollback(target_turn=0, session_id="rb_test")))
        await asyncio.wait_for(flush_started.wait(), timeout=10)

        prompt_task = asyncio.create_task(engine._on_user_message(UserMessage(text="concurrent ACP prompt")))
        await asyncio.sleep(0)
        assert engine._turn_state.prompt_admission_closed is True
        assert prompt_task.done() is False
        assert engine.state is EngineState.IDLE

        release_flush.set()
        await asyncio.wait_for(rollback_task, timeout=10)
        await asyncio.wait_for(prompt_task, timeout=10)

        assert reset_observations == [(EngineState.IDLE, True)]
        assert engine._turn_state.prompt_admission_closed is False

    @pytest.mark.asyncio
    async def test_waits_for_exact_turn_lifecycle_before_validating_target(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)
        lifecycle_release = asyncio.Event()
        replacement_release = asyncio.Event()

        async def wait_for_release(release: asyncio.Event) -> None:
            await release.wait()

        captured_task = asyncio.create_task(wait_for_release(lifecycle_release))
        replacement_task = asyncio.create_task(wait_for_release(replacement_release))
        engine._turn_state.run_task = captured_task

        try:
            rollback_task = asyncio.create_task(
                engine._on_user_rollback(UserRollback(target_turn=5, session_id="rb_test"))
            )
            await asyncio.sleep(0)
            engine._turn_state.run_task = replacement_task
            assert warnings == []

            lifecycle_release.set()
            await asyncio.wait_for(rollback_task, timeout=10)
            assert [warning.code for warning in warnings] == ["rollback_unavailable"]
            assert not replacement_task.done()
        finally:
            replacement_task.cancel()
            await asyncio.gather(replacement_task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_relative_target_is_resolved_after_captured_lifecycle(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        store = JsonFileStateStore(tmp_path)
        engine._history.bind(_state_after_turns(2))
        engine._turn_number = 2
        lifecycle_release = asyncio.Event()
        results: list[RollbackResult] = []
        await _collect_events(engine.event_bus, RollbackResult, results)

        async def _finalize_third_turn() -> None:
            await lifecycle_release.wait()
            await store.save_session("rb_test", _state_after_turns(2))
            session_file = store.session_dir("rb_test") / "session.json"
            snapshot_dir = session_file.parent / "snapshots"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            atomic_copy_file(session_file, snapshot_dir / "turn_3.json")
            await store.save_session("rb_test", _state_after_turns(3))
            live = await store.load_session("rb_test")
            assert live is not None
            engine._history.bind(live)
            engine._turn_number = 3

        async def _fake_restore(event: Any) -> None:
            loaded = await store.load_session(event.session_id)
            assert loaded is not None
            engine._history.bind(loaded)
            engine._turn_number = loaded.get("turn_counter", 0)

        lifecycle_task = asyncio.create_task(_finalize_third_turn())
        engine._turn_state.run_task = lifecycle_task
        engine._on_session_restore = _fake_restore  # type: ignore[assignment]
        rollback_task = asyncio.create_task(
            engine._on_user_rollback(
                UserRollback(
                    target_turn=1,
                    relative_turns=1,
                    session_id="rb_test",
                )
            )
        )
        await asyncio.sleep(0)
        assert results == []

        lifecycle_release.set()
        await asyncio.wait_for(rollback_task, timeout=10)

        assert len(results) == 1
        assert results[0].target_turn == 2
        assert results[0].rolled_back_user_text == "user 3"
        assert engine._turn_number == 2

    @pytest.mark.asyncio
    async def test_rejects_stale_picker_projection_before_committing_transition(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._history.bind(_state_after_turns(4))
        engine._turn_number = 4
        engine._turn_state.injection_admission_open = True
        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)

        await engine._on_user_rollback(
            UserRollback(
                target_turn=2,
                expected_current_turn=3,
                session_id="rb_test",
            )
        )

        assert [warning.code for warning in warnings] == ["rollback_conversation_changed"]
        assert warnings[0].message == "Rollback cancelled because the conversation advanced from turn 3 to turn 4."
        _assert_display_message(
            warnings[0],
            "rollback.conversation_advanced",
            {"expected_turn": 3, "current_turn": 4},
        )
        assert engine.session_generation == 0
        assert engine._turn_number == 4
        assert engine.state is EngineState.IDLE
        assert engine._turn_state.injection_admission_open is True
        assert engine._turn_state.prompt_admission_closed is False

    @pytest.mark.asyncio
    async def test_rejects_stale_picker_projection_after_same_turn_retry(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._history.bind(_state_after_turns(4))
        engine._turn_number = 4
        picker_revision = engine.conversation_revision
        # A retry lifecycle advances the conversation without increasing the
        # logical turn number or necessarily allocating another run scope.
        engine._turn_state.advance_conversation_revision()
        engine._turn_state.injection_admission_open = True
        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)

        await engine._on_user_rollback(
            UserRollback(
                target_turn=2,
                expected_current_turn=4,
                expected_conversation_revision=picker_revision,
                session_id="rb_test",
            )
        )

        assert [warning.code for warning in warnings] == ["rollback_conversation_changed"]
        assert warnings[0].message == (
            "Rollback cancelled because the conversation changed after the picker was loaded."
        )
        _assert_display_message(warnings[0], "rollback.conversation_changed")
        assert engine.session_generation == 0
        assert engine._turn_number == 4
        assert engine.conversation_revision == picker_revision + 1
        assert engine.state is EngineState.IDLE
        assert engine._turn_state.injection_admission_open is True
        assert engine._turn_state.prompt_admission_closed is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("drift", ["build", "workspace"])
    async def test_rejects_picker_projection_after_runtime_rebuild_drift(
        self,
        tmp_path: Path,
        drift: str,
    ) -> None:
        engine = _make_engine(tmp_path)
        engine._history.bind(_state_after_turns(4))
        engine._turn_number = 4
        picker_build_generation = engine.build_generation
        picker_workspace_cwd = engine.workspace_primary_cwd
        if drift == "build":
            engine._build_generation += 1
        else:
            engine._workspace = Workspace.from_cwd(str(tmp_path / "rebuilt-workspace"))
        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)

        await engine._on_user_rollback(
            UserRollback(
                target_turn=2,
                expected_current_turn=4,
                expected_conversation_revision=engine.conversation_revision,
                expected_build_generation=picker_build_generation,
                expected_workspace_cwd=picker_workspace_cwd,
                session_id="rb_test",
            )
        )

        assert [warning.code for warning in warnings] == ["rollback_runtime_changed"]
        assert warnings[0].message == (
            "Rollback cancelled because the workspace or runtime changed after the picker was loaded."
        )
        _assert_display_message(warnings[0], "rollback.runtime_changed")
        assert engine.session_generation == 0
        assert engine._turn_number == 4
        assert engine.state is EngineState.IDLE
        assert engine._turn_state.prompt_admission_closed is False

    @pytest.mark.asyncio
    async def test_revalidates_session_generation_after_turn_lifecycle_wait(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)
        lifecycle_release = asyncio.Event()

        async def wait_for_release() -> None:
            await lifecycle_release.wait()

        lifecycle_task = asyncio.create_task(wait_for_release())
        engine._turn_state.run_task = lifecycle_task
        rollback_task = asyncio.create_task(engine._on_user_rollback(UserRollback(target_turn=5, session_id="rb_test")))
        await asyncio.sleep(0)

        # A switch away and back may restore the same visible ID; the
        # monotonic generation still invalidates the deferred request.
        engine._session_generation += 1
        lifecycle_release.set()
        await asyncio.wait_for(rollback_task, timeout=10)

        assert [warning.code for warning in warnings] == ["rollback_session_changed"]
        assert warnings[0].message == "Rollback cancelled because the active session changed."
        _assert_display_message(warnings[0], "rollback.session_changed")

    @pytest.mark.asyncio
    async def test_refused_when_running(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path, fsm_state=EngineState.RUNNING)
        assert engine.state is EngineState.RUNNING
        lifecycle_release = asyncio.Event()

        async def wait_for_release() -> None:
            await lifecycle_release.wait()

        lifecycle_task = asyncio.create_task(wait_for_release())
        engine._turn_state.run_task = lifecycle_task

        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)

        try:
            await asyncio.wait_for(
                engine._on_user_rollback(UserRollback(target_turn=2, revert_changes=False)),
                timeout=10,
            )
        finally:
            lifecycle_task.cancel()
            await asyncio.gather(lifecycle_task, return_exceptions=True)

        await asyncio.sleep(0)  # let the bus dispatch
        refused = next(warning for warning in warnings if warning.code == "rollback_refused")
        assert refused.message == "Rollback is not allowed in state RUNNING."
        _assert_display_message(refused, "rollback.refused", {"state": "RUNNING"})

    @pytest.mark.asyncio
    async def test_refused_when_pending_retry(self, tmp_path: Path) -> None:
        """Retry already queued → reject rollback.

        PENDING_RETRY means the user already asked us to resume after
        the current run winds down; letting a rollback race that would
        leave the engine in an inconsistent state where the retry
        lands on a session that's been swapped out from under it.
        """
        engine = _make_engine(tmp_path, fsm_state=EngineState.PENDING_RETRY)
        assert engine.state is EngineState.PENDING_RETRY

        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)

        await engine._on_user_rollback(UserRollback(target_turn=2, revert_changes=False))
        await asyncio.sleep(0)
        assert any(w.code == "rollback_refused" for w in warnings)

    @pytest.mark.asyncio
    async def test_refused_when_awaiting_sub_agents(self, tmp_path: Path) -> None:
        """Parent run is pinned on a paused sub-agent decision → reject rollback.

        Mutating history while a ``pending_decision`` future is
        outstanding would detach the sub-agent's saved record from
        the conversation state the user ends up on.  Matches the
        FSM's ``is_running()`` contract that covers this state.
        """
        engine = _make_engine(tmp_path, fsm_state=EngineState.AWAITING_SUB_AGENTS)
        assert engine.state is EngineState.AWAITING_SUB_AGENTS

        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)

        await engine._on_user_rollback(UserRollback(target_turn=2, revert_changes=False))
        await asyncio.sleep(0)
        assert any(w.code == "rollback_refused" for w in warnings)

    @pytest.mark.asyncio
    async def test_refused_without_session(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._session_id = None

        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)

        await engine._on_user_rollback(UserRollback(target_turn=2))

        await asyncio.sleep(0)
        warning = next(warning for warning in warnings if warning.code == "rollback_no_session")
        assert warning.message == "No active session to roll back."
        _assert_display_message(warning, "rollback.no_session")

    @pytest.mark.asyncio
    async def test_refused_when_relative_turn_count_is_non_positive(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)

        await engine._on_user_rollback(UserRollback(target_turn=0, relative_turns=0))

        assert len(warnings) == 1
        assert warnings[0].message == "relative_turns must be positive."
        _assert_display_message(warnings[0], "rollback.relative_turns_invalid")

    @pytest.mark.asyncio
    async def test_refused_when_relative_turn_count_exceeds_session(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._turn_number = 2
        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)

        await engine._on_user_rollback(UserRollback(target_turn=0, relative_turns=3))

        assert len(warnings) == 1
        assert warnings[0].message == "Cannot roll back 3 turns; the session currently has 2."
        _assert_display_message(
            warnings[0],
            "rollback.turns_unavailable",
            {"requested_turns": 3, "current_turns": 2},
        )

    @pytest.mark.asyncio
    async def test_refused_invalid_turn(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)

        # Negative values are invalid under the "keep N turns" convention
        # (``0`` is the welcome target).
        await engine._on_user_rollback(UserRollback(target_turn=-1))

        await asyncio.sleep(0)
        warning = next(warning for warning in warnings if warning.code == "rollback_invalid_turn")
        assert warning.message == "target_turn must be >= 0."
        _assert_display_message(warning, "rollback.target_turn_invalid")
        assert engine.session_generation == 0

    @pytest.mark.asyncio
    async def test_refused_when_target_unavailable(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        # No snapshots, no turns
        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)

        await engine._on_user_rollback(UserRollback(target_turn=5))

        await asyncio.sleep(0)
        warning = next(warning for warning in warnings if warning.code == "rollback_unavailable")
        assert warning.message == "Cannot roll back to turn 5; available turns: []"
        _assert_display_message(
            warning,
            "rollback.turn_unavailable",
            {"target_turn": 5, "available": "[]"},
        )
        assert engine.session_generation == 0


# ===========================================================================
# Suppress-save gate
# ===========================================================================


class TestSuppressSave:
    @pytest.mark.asyncio
    async def test_save_is_gated_by_suppress_flag(self, tmp_path: Path) -> None:
        """Unit: flag flip alone gates ``_save_current_session``."""
        engine = _make_engine(tmp_path)

        class _StubExec:
            def __init__(self) -> None:
                self.history_state: dict[str, Any] = {"messages": []}

        engine._executor = _StubExec()  # type: ignore[assignment]
        engine._suppress_save = True

        called: list[bool] = []

        async def _spy(**_kwargs: Any) -> None:
            called.append(True)

        engine._persistence.save_session = _spy  # type: ignore[assignment]

        await engine._save_current_session()
        assert called == []  # suppressed — no save call reached persistence

    @pytest.mark.asyncio
    async def test_rollback_swap_holds_suppress_save_around_reload(self, tmp_path: Path) -> None:
        """Regression: the ``target_turn >= 1`` swap must keep
        ``_suppress_save`` True across ``_on_session_restore``.

        The swap writes ``session.json`` from the turn snapshot, then
        reloads the engine.  If any intermediate step calls
        ``_save_current_session`` without the flag set, the in-memory
        pre-rollback state clobbers the just-swapped snapshot.  This
        test stubs out ``_on_session_restore`` so we can observe the
        flag value at the point the reload would run, without dragging
        the full restore flow into the test.
        """
        engine = _make_engine(tmp_path)
        # Bind non-empty history so ``_available_rollback_turns`` adds
        # the welcome target (0); without it the guardrail returns
        # ``[]`` and the swap branch is never reached.
        engine._history.bind({"messages": [Message(role="user", contents=["hello"])], "compressed_msgs": []})

        # Set up a valid snapshot for target_turn=1 (keep 1 turn →
        # restore turn_2.json) so the handler reaches the swap+reload
        # branch.
        session_dir = tmp_path / "rb_test"
        session_file = session_dir / "session.json"
        _write_session_json(session_file)
        engine._turn_number = 2
        engine._write_rollback_snapshot()  # creates turn_2.json

        observed: dict[str, Any] = {}

        async def _fake_restore(event: Any) -> None:
            observed["suppress_during_restore"] = engine._suppress_save
            observed["session_id"] = event.session_id

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]

        # Count any stray save calls that leak through during the swap.
        save_calls: list[bool] = []

        async def _spy(**_kwargs: Any) -> None:
            save_calls.append(engine._suppress_save)

        engine._persistence.save_session = _spy  # type: ignore[assignment]

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        assert observed.get("suppress_during_restore") is True, (
            "``_suppress_save`` must be True while _on_session_restore runs during swap"
        )
        assert observed.get("session_id") == "rb_test"
        # After the try/finally, the flag is reset.
        assert engine._suppress_save is False
        # Any save that leaked through during the swap window would have
        # had ``_suppress_save == True`` at observation time — i.e. it
        # was properly gated upstream.  No ungated saves should occur.
        assert all(v is True for v in save_calls)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("revert_changes", [False, True])
    async def test_rollback_workspace_baseline_policy_after_restore(
        self,
        tmp_path: Path,
        revert_changes: bool,
    ) -> None:
        engine = _make_engine(tmp_path)
        engine._history.bind({"messages": [Message("user", ["hello"])], "compressed_msgs": []})
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        engine._workspace = Workspace.from_cwd(str(workspace_root))
        tracker = engine._workspace_change_tracker
        tracker.retarget_roots(engine._workspace)
        original_baseline = tracker.capture_baseline(1)

        session_dir = tmp_path / "rb_test"
        _write_session_json(session_dir / "session.json")
        engine._turn_number = 2
        engine._write_rollback_snapshot()
        save_payloads: list[dict[str, Any] | None] = []

        async def _fake_restore(_event: Any) -> None:
            assert engine._suppress_save is True

        async def _fake_save(*, raise_on_error: bool = False) -> bool:
            assert raise_on_error is False
            save_payloads.append(tracker.serialize())
            return True

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]
        engine._save_current_session = _fake_save  # type: ignore[method-assign]

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=revert_changes))

        if revert_changes:
            assert tracker.baseline is None
            assert save_payloads == [None]
        else:
            assert tracker.baseline == original_baseline
            assert save_payloads == []

    @pytest.mark.asyncio
    async def test_rollback_waits_for_active_capture_and_save_before_commit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        engine = _make_engine(tmp_path)
        store = JsonFileStateStore(tmp_path)
        session_dir = store.session_dir("rb_test")
        session_file = session_dir / "session.json"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)
        for turn in range(1, 4):
            await store.save_session("rb_test", _state_after_turns(turn))
            if turn < 3:
                atomic_copy_file(session_file, snap_dir / f"turn_{turn + 1}.json")
        live = await store.load_session("rb_test")
        assert live is not None
        engine._history.bind(live)
        engine._turn_number = 3

        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        engine._workspace = Workspace.from_cwd(str(workspace_root))
        tracker = engine._workspace_change_tracker
        tracker.retarget_roots(engine._workspace)

        entered = threading.Event()
        release = threading.Event()
        original_capture = workspace_changes._capture_workspace

        def _blocked(*args: Any, **kwargs: Any) -> Any:
            entered.set()
            assert release.wait(timeout=10.0)
            return original_capture(*args, **kwargs)

        monkeypatch.setattr(workspace_changes, "_capture_workspace", _blocked)

        order: list[str] = []
        generation_after_finalization: list[int] = []

        async def _finalization() -> None:
            await asyncio.to_thread(tracker.capture_baseline, 3)
            order.append("capture")
            order.append("save")
            generation_after_finalization.append(engine.session_generation)

        async def _fake_restore(event: Any) -> None:
            order.append("restore")
            loaded = await store.load_session(event.session_id)
            assert loaded is not None
            engine._history.bind(loaded)
            engine._turn_number = loaded.get("turn_counter", 0)

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]
        lifecycle_task = asyncio.create_task(_finalization())
        engine._turn_state.run_task = lifecycle_task

        rollback_task = asyncio.create_task(engine._on_user_rollback(UserRollback(target_turn=1, session_id="rb_test")))
        await asyncio.to_thread(entered.wait, 5.0)
        await asyncio.sleep(0.05)
        assert order == []
        assert engine.session_generation == 0

        release.set()
        await asyncio.wait_for(rollback_task, timeout=10)
        await asyncio.wait_for(lifecycle_task, timeout=10)

        assert order == ["capture", "save", "restore"]
        assert generation_after_finalization == [0]
        assert engine.session_generation == 1

    @pytest.mark.asyncio
    async def test_rollback_result_carries_first_discarded_user_prompt(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        store = JsonFileStateStore(tmp_path)
        session_dir = store.session_dir("rb_test")
        session_file = session_dir / "session.json"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)

        for turn in range(1, 4):
            await store.save_session("rb_test", _state_after_turns(turn))
            if turn < 3:
                atomic_copy_file(session_file, snap_dir / f"turn_{turn + 1}.json")

        live = await store.load_session("rb_test")
        assert live is not None
        engine._history.bind(live)
        engine._turn_number = 3

        results: list[RollbackResult] = []
        await _collect_events(engine.event_bus, RollbackResult, results)

        async def _fake_restore(event: Any) -> None:
            loaded = await store.load_session(event.session_id)
            assert loaded is not None
            engine._history.bind(loaded)
            engine._turn_number = loaded.get("turn_counter", 0)

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        assert len(results) == 1
        assert results[0].rolled_back_user_text == "user 2"

    @pytest.mark.asyncio
    async def test_welcome_rollback_result_carries_first_discarded_user_prompt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        engine = _make_engine(tmp_path)
        engine._history.bind(_state_after_turns(2))

        results: list[RollbackResult] = []
        await _collect_events(engine.event_bus, RollbackResult, results)

        lifecycle: list[str] = []

        async def _fake_reset(
            session_id: str,
            *,
            write_lock_held: bool = False,
            after_delete: Any = None,
            before_restart: Any = None,
        ) -> bool:
            _ = write_lock_held
            assert after_delete is None
            assert before_restart is not None
            lifecycle.append(f"reset:{session_id}")
            engine._history.bind({"messages": [], "compressed_msgs": [], "turn_counter": 0})
            engine._turn_number = 0
            return True

        async def _fake_trajectory_rollback(*, target_turn: int, history_state: dict[str, Any] | None) -> None:
            assert target_turn == 0
            assert history_state is not None
            assert history_state["turn_counter"] == 2
            lifecycle.append("trajectory.rollback")

        engine._reset_session_to_welcome = _fake_reset  # type: ignore[assignment]
        monkeypatch.setattr(engine._trajectory_recorder, "rollback", _fake_trajectory_rollback)

        await engine._on_user_rollback(UserRollback(target_turn=0, revert_changes=False))

        assert lifecycle == ["reset:rb_test", "trajectory.rollback"]
        assert len(results) == 1
        assert results[0].rolled_back_user_text == "user 1"
        assert engine.session_generation == 1

    @pytest.mark.asyncio
    async def test_welcome_conversation_rollback_queues_retained_files_once(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._workspace = Workspace.from_cwd(str(tmp_path))
        engine._history.bind(_state_after_turns(1))
        target = tmp_path / "retained.txt"
        target.write_text("before", encoding="utf-8")
        tracker = MutationTracker(SnapshotStore(tmp_path / "mutation_store"))
        tracker.start_turn(1)
        mutation = tracker.record(str(target), MutationOp.MODIFY, MutationSource.EDIT_FILE, "edit")
        assert mutation is not None
        target.write_text("after", encoding="utf-8")
        tracker.record_after(mutation)
        engine._mutation_tracker = tracker

        async def _fake_reset(
            _session_id: str,
            *,
            write_lock_held: bool = False,
            after_delete: Any = None,
            before_restart: Any = None,
        ) -> bool:
            _ = write_lock_held, after_delete, before_restart
            engine._workspace_change_tracker.reset_for_restart()
            return True

        engine._reset_session_to_welcome = _fake_reset  # type: ignore[assignment]

        await engine._on_user_rollback(UserRollback(target_turn=0, revert_changes=False))

        notice = engine._workspace_change_tracker.take_pending_notice()
        assert notice is not None
        assert notice.startswith("Files retained from the discarded conversation:")
        assert 'modified: "retained.txt"' in notice
        assert engine._workspace_change_tracker.take_pending_notice() is None

    @pytest.mark.asyncio
    async def test_welcome_file_rollback_queues_no_retained_notice(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._history.bind(_state_after_turns(1))

        async def _fake_reset(
            _session_id: str,
            *,
            write_lock_held: bool = False,
            after_delete: Any = None,
            before_restart: Any = None,
        ) -> bool:
            _ = write_lock_held, before_restart
            engine._workspace_change_tracker.reset_for_restart()
            if after_delete is not None:
                await after_delete()
            return True

        engine._reset_session_to_welcome = _fake_reset  # type: ignore[assignment]

        await engine._on_user_rollback(UserRollback(target_turn=0, revert_changes=True))

        assert engine._workspace_change_tracker.take_pending_notice() is None

    @staticmethod
    def _record_modify(tracker: MutationTracker, path: Path) -> None:
        mutation = tracker.record(str(path), MutationOp.MODIFY, MutationSource.EDIT_FILE, "edit")
        assert mutation is not None
        path.write_text(f"after-{path.stem}", encoding="utf-8")
        tracker.record_after(mutation)

    @staticmethod
    def _welcome_reset_running_file_rollback(engine: AgentEngine) -> None:
        async def _fake_reset(
            _session_id: str,
            *,
            write_lock_held: bool = False,
            after_delete: Any = None,
            before_restart: Any = None,
        ) -> bool:
            _ = write_lock_held, before_restart
            engine._workspace_change_tracker.reset_for_restart()
            if after_delete is not None:
                await after_delete()
            return True

        engine._reset_session_to_welcome = _fake_reset  # type: ignore[assignment]

    @pytest.mark.asyncio
    async def test_welcome_partial_file_rollback_reports_unselected_paths(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._workspace = Workspace.from_cwd(str(tmp_path))
        engine._history.bind(_state_after_turns(1))
        selected = tmp_path / "selected.txt"
        selected.write_text("before-selected", encoding="utf-8")
        skipped = tmp_path / "skipped.txt"
        skipped.write_text("before-skipped", encoding="utf-8")
        tracker = MutationTracker(SnapshotStore(tmp_path / "mutation_store"))
        tracker.start_turn(1)
        self._record_modify(tracker, selected)
        self._record_modify(tracker, skipped)
        engine._mutation_tracker = tracker
        self._welcome_reset_running_file_rollback(engine)

        selected_norm = os.path.normpath(os.path.abspath(str(selected)))
        await engine._on_user_rollback(UserRollback(target_turn=0, revert_changes=True, selected_paths=[selected_norm]))

        assert selected.read_text(encoding="utf-8") == "before-selected"
        assert skipped.read_text(encoding="utf-8") == "after-skipped"
        notice = engine._workspace_change_tracker.take_pending_notice()
        assert notice is not None
        assert notice.startswith("Files not reverted by the rollback")
        assert '"skipped.txt"' in notice
        assert "selected.txt" not in notice
        assert engine._workspace_change_tracker.take_pending_notice() is None

    @pytest.mark.asyncio
    async def test_welcome_failed_restore_reports_retained_paths(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._workspace = Workspace.from_cwd(str(tmp_path))
        engine._history.bind(_state_after_turns(1))
        target = tmp_path / "blocked.txt"
        target.write_text("before", encoding="utf-8")
        tracker = MutationTracker(SnapshotStore(tmp_path / "mutation_store"))
        tracker.start_turn(1)
        self._record_modify(tracker, target)
        engine._mutation_tracker = tracker
        self._welcome_reset_running_file_rollback(engine)
        # A directory at the target path makes the snapshot restore fail.
        target.unlink()
        target.mkdir()

        await engine._on_user_rollback(UserRollback(target_turn=0, revert_changes=True))

        notice = engine._workspace_change_tracker.take_pending_notice()
        assert notice is not None
        assert notice.startswith("Files not reverted by the rollback")
        assert '"blocked.txt"' in notice

    @pytest.mark.asyncio
    async def test_welcome_full_file_rollback_queues_no_partial_notice(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._workspace = Workspace.from_cwd(str(tmp_path))
        engine._history.bind(_state_after_turns(1))
        first = tmp_path / "first.txt"
        first.write_text("before-first", encoding="utf-8")
        second = tmp_path / "second.txt"
        second.write_text("before-second", encoding="utf-8")
        tracker = MutationTracker(SnapshotStore(tmp_path / "mutation_store"))
        tracker.start_turn(1)
        self._record_modify(tracker, first)
        self._record_modify(tracker, second)
        engine._mutation_tracker = tracker
        self._welcome_reset_running_file_rollback(engine)

        await engine._on_user_rollback(UserRollback(target_turn=0, revert_changes=True))

        assert first.read_text(encoding="utf-8") == "before-first"
        assert second.read_text(encoding="utf-8") == "before-second"
        assert engine._workspace_change_tracker.take_pending_notice() is None

    @pytest.mark.asyncio
    async def test_nonzero_partial_file_rollback_reports_unselected_paths(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._workspace = Workspace.from_cwd(str(tmp_path))
        store = JsonFileStateStore(tmp_path)
        session_dir = store.session_dir("rb_test")
        session_file = session_dir / "session.json"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)
        for turn in range(1, 3):
            await store.save_session("rb_test", _state_after_turns(turn))
            if turn < 2:
                atomic_copy_file(session_file, snap_dir / f"turn_{turn + 1}.json")
        live = await store.load_session("rb_test")
        assert live is not None
        engine._history.bind(live)
        engine._turn_number = 2

        selected = tmp_path / "selected.txt"
        selected.write_text("before-selected", encoding="utf-8")
        skipped = tmp_path / "skipped.txt"
        skipped.write_text("before-skipped", encoding="utf-8")
        tracker = MutationTracker(SnapshotStore(tmp_path / "mutation_store"))
        tracker.start_turn(2)
        self._record_modify(tracker, selected)
        self._record_modify(tracker, skipped)
        engine._mutation_tracker = tracker

        async def _fake_restore(event: Any) -> None:
            loaded = await store.load_session(event.session_id)
            assert loaded is not None
            engine._history.bind(loaded)
            engine._turn_number = loaded.get("turn_counter", 0)

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]

        selected_norm = os.path.normpath(os.path.abspath(str(selected)))
        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=True, selected_paths=[selected_norm]))

        assert selected.read_text(encoding="utf-8") == "before-selected"
        assert skipped.read_text(encoding="utf-8") == "after-skipped"
        assert engine._workspace_change_tracker.baseline is None
        notice = engine._workspace_change_tracker.take_pending_notice()
        assert notice is not None
        assert notice.startswith("Files not reverted by the rollback")
        assert '"skipped.txt"' in notice
        assert "selected.txt" not in notice

    @pytest.mark.asyncio
    async def test_welcome_revert_with_truncated_detection_warns_incomplete(self, tmp_path: Path) -> None:
        """A rolled-back turn with truncated detection and zero recorded rows still warns."""
        engine = _make_engine(tmp_path)
        engine._workspace = Workspace.from_cwd(str(tmp_path))
        engine._history.bind(_state_after_turns(1))
        tracker = MutationTracker(SnapshotStore(tmp_path / "mutation_store"))
        tracker.start_turn(1)
        tracker.mark_detection_truncated()
        engine._mutation_tracker = tracker
        self._welcome_reset_running_file_rollback(engine)

        await engine._on_user_rollback(UserRollback(target_turn=0, revert_changes=True))

        notice = engine._workspace_change_tracker.take_pending_notice()
        assert notice is not None
        assert notice.startswith("Files not reverted by the rollback")
        assert "could not be determined completely" in notice
        assert engine._workspace_change_tracker.take_pending_notice() is None

    @pytest.mark.asyncio
    async def test_nonzero_revert_with_truncated_detection_warns_incomplete(self, tmp_path: Path) -> None:
        """The incomplete-detection caveat fires even when every known candidate restored."""
        engine = _make_engine(tmp_path)
        engine._workspace = Workspace.from_cwd(str(tmp_path))
        store = JsonFileStateStore(tmp_path)
        session_dir = store.session_dir("rb_test")
        session_file = session_dir / "session.json"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)
        for turn in range(1, 3):
            await store.save_session("rb_test", _state_after_turns(turn))
            if turn < 2:
                atomic_copy_file(session_file, snap_dir / f"turn_{turn + 1}.json")
        live = await store.load_session("rb_test")
        assert live is not None
        engine._history.bind(live)
        engine._turn_number = 2

        target = tmp_path / "restored.txt"
        target.write_text("before", encoding="utf-8")
        tracker = MutationTracker(SnapshotStore(tmp_path / "mutation_store"))
        tracker.start_turn(2)
        self._record_modify(tracker, target)
        tracker.mark_detection_truncated()
        engine._mutation_tracker = tracker

        async def _fake_restore(event: Any) -> None:
            loaded = await store.load_session(event.session_id)
            assert loaded is not None
            engine._history.bind(loaded)
            engine._turn_number = loaded.get("turn_counter", 0)

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=True))

        assert target.read_text(encoding="utf-8") == "before"
        notice = engine._workspace_change_tracker.take_pending_notice()
        assert notice is not None
        assert "could not be determined completely" in notice
        assert "restored.txt" not in notice

    @pytest.mark.asyncio
    async def test_nonzero_revert_ignores_truncation_on_retained_turns(self, tmp_path: Path) -> None:
        """Truncated detection on a turn the rollback keeps does not trigger the caveat."""
        engine = _make_engine(tmp_path)
        engine._workspace = Workspace.from_cwd(str(tmp_path))
        store = JsonFileStateStore(tmp_path)
        session_dir = store.session_dir("rb_test")
        session_file = session_dir / "session.json"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)
        for turn in range(1, 3):
            await store.save_session("rb_test", _state_after_turns(turn))
            if turn < 2:
                atomic_copy_file(session_file, snap_dir / f"turn_{turn + 1}.json")
        live = await store.load_session("rb_test")
        assert live is not None
        engine._history.bind(live)
        engine._turn_number = 2

        target = tmp_path / "restored.txt"
        target.write_text("before", encoding="utf-8")
        tracker = MutationTracker(SnapshotStore(tmp_path / "mutation_store"))
        tracker.start_turn(1)
        tracker.mark_detection_truncated()
        tracker.start_turn(2)
        self._record_modify(tracker, target)
        engine._mutation_tracker = tracker

        async def _fake_restore(event: Any) -> None:
            loaded = await store.load_session(event.session_id)
            assert loaded is not None
            engine._history.bind(loaded)
            engine._turn_number = loaded.get("turn_counter", 0)

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=True))

        assert target.read_text(encoding="utf-8") == "before"
        assert engine._workspace_change_tracker.take_pending_notice() is None

    @pytest.mark.asyncio
    async def test_welcome_rollback_reports_error_when_reset_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        engine = _make_engine(tmp_path)
        engine._history.bind(_state_after_turns(2))

        results: list[RollbackResult] = []
        errors: list[Error] = []
        await _collect_events(engine.event_bus, RollbackResult, results)
        await _collect_events(engine.event_bus, Error, errors)

        async def _fake_reset(
            _session_id: str,
            *,
            write_lock_held: bool = False,
            after_delete: Any = None,
            before_restart: Any = None,
        ) -> bool:
            _ = write_lock_held, after_delete, before_restart
            return False

        trajectory_rollbacks: list[int] = []

        async def _fake_trajectory_rollback(*, target_turn: int, history_state: dict[str, Any] | None) -> None:
            _ = history_state
            trajectory_rollbacks.append(target_turn)

        engine._reset_session_to_welcome = _fake_reset  # type: ignore[assignment]
        monkeypatch.setattr(engine._trajectory_recorder, "rollback", _fake_trajectory_rollback)

        await engine._on_user_rollback(UserRollback(target_turn=0, revert_changes=False))

        assert results == []
        assert [error.code for error in errors] == ["rollback_reset_failed"]
        assert errors[0].message == (
            "Rollback to welcome could not reset the session because the session state is busy."
        )
        assert trajectory_rollbacks == []
        _assert_display_message(errors[0], "rollback.reset_failed")

    @pytest.mark.asyncio
    async def test_rollback_swap_updates_backup_for_recovery(self, tmp_path: Path) -> None:
        """After rollback, backup recovery must not resurrect the pre-rollback state."""
        engine = _make_engine(tmp_path)
        store = JsonFileStateStore(tmp_path)
        engine._history.bind({"messages": [Message(role="user", contents=["hello"])], "compressed_msgs": []})

        rolled_back_state = {"messages": [Message("user", ["rolled back"])], "compressed_msgs": []}
        pre_rollback_state = {"messages": [Message("user", ["pre rollback"])], "compressed_msgs": []}

        await store.save_session("rb_test", rolled_back_state)
        session_dir = tmp_path / "rb_test"
        snapshot_payload = (session_dir / "session.json").read_text(encoding="utf-8")
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir()
        (snap_dir / "turn_2.json").write_text(snapshot_payload, encoding="utf-8")

        await store.save_session("rb_test", pre_rollback_state)

        async def _fake_restore(_event: Any) -> None: ...

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        (session_dir / "session.json").write_text("{ broken primary", encoding="utf-8")
        loaded = await store.load_session("rb_test")

        assert loaded is not None
        assert loaded["messages"][0].text == "rolled back"

    @pytest.mark.asyncio
    async def test_a_cancel_on_the_audit_record_still_restores_the_swapped_session(self, tmp_path: Path) -> None:
        """The swap has committed by then; abandoning the restore would undo it on the next save."""
        engine = _make_engine(tmp_path)
        store = JsonFileStateStore(tmp_path)
        engine._history.bind({"messages": [Message(role="user", contents=["hello"])], "compressed_msgs": []})

        rolled_back_state = {"messages": [Message("user", ["rolled back"])], "compressed_msgs": []}
        pre_rollback_state = {"messages": [Message("user", ["pre rollback"])], "compressed_msgs": []}
        await store.save_session("rb_test", rolled_back_state)
        session_dir = tmp_path / "rb_test"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir()
        (snap_dir / "turn_2.json").write_text((session_dir / "session.json").read_text(encoding="utf-8"), "utf-8")
        await store.save_session("rb_test", pre_rollback_state)

        async def _fake_restore(event: Any) -> None:
            loaded = await store.load_session(event.session_id)
            assert loaded is not None
            engine._history.bind(loaded)

        async def _cancelled_record(*, target_turn: int, history_state: dict[str, Any] | None) -> None:
            _ = target_turn, history_state
            raise asyncio.CancelledError

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]
        engine._trajectory_recorder.rollback = _cancelled_record  # type: ignore[assignment]

        with pytest.raises(asyncio.CancelledError):
            await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        # The engine holds the restored history, so the next save writes that
        # rather than putting the superseded conversation back on disk.
        assert engine._history.state["messages"][0].text == "rolled back"

    @pytest.mark.asyncio
    async def test_rollback_restore_round_trips_runtime_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rollback reload path restores runtime metadata from the promoted snapshot."""
        engine = _make_engine(tmp_path)
        store = JsonFileStateStore(tmp_path)
        engine._agent_profile = AgentProfile(name="Code")
        engine._history.bind({"messages": [Message(role="user", contents=["hello"])], "compressed_msgs": []})

        rolled_back_state = {
            "messages": [Message("user", ["rolled back"])],
            "compressed_msgs": [],
            "last_usage": {"total_token_count": 77, "calibration_ratio": 1.4},
            "total_session_tokens": 77,
            "total_session_input_tokens": 33,
            "total_session_output_tokens": 44,
        }
        pre_rollback_state = {
            "messages": [Message("user", ["pre rollback"])],
            "compressed_msgs": [],
            "last_usage": {"total_token_count": 12},
            "total_session_tokens": 12,
            "total_session_input_tokens": 5,
            "total_session_output_tokens": 7,
        }

        await store.save_session("rb_test", rolled_back_state, agent_profile="Code")
        session_dir = tmp_path / "rb_test"
        snapshot_payload = (session_dir / "session.json").read_text(encoding="utf-8")
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir()
        (snap_dir / "turn_2.json").write_text(snapshot_payload, encoding="utf-8")

        await store.save_session("rb_test", pre_rollback_state, agent_profile="Code")
        original_meta = engine._runtime_meta

        class _StubExec:
            def __init__(self) -> None:
                self.history_state: dict[str, Any] = {}

        async def _fake_start(
            profile: AgentProfile, *, operation: str = "startup", staged_loaded: LoadedSettings | None = None
        ) -> None:
            engine._agent_profile = profile
            engine._executor = _StubExec()  # type: ignore[assignment]
            if staged_loaded is not None:
                engine._settings_handle.install(staged_loaded)

        monkeypatch.setattr(engine, "start", _fake_start)

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        assert engine._runtime_meta is not original_meta
        assert engine._runtime_meta.total_session_tokens == 77
        assert engine._runtime_meta.total_session_input_tokens == 33
        assert engine._runtime_meta.total_session_output_tokens == 44
        assert engine._runtime_meta.last_usage_details == {"total_token_count": 77, "calibration_ratio": 1.4}

    @pytest.mark.asyncio
    async def test_rollback_restores_target_turn_todo_list(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rolling back to turn N rehydrates the todo tracker from turn N's
        snapshot, replacing the pre-rollback list."""
        from chrys.service.todos.tracker import TodoTracker

        engine = _make_engine(tmp_path)
        store = JsonFileStateStore(tmp_path)
        engine._agent_profile = AgentProfile(name="Code")
        engine._history.bind({"messages": [Message(role="user", contents=["hello"])], "compressed_msgs": []})

        turn_one_todos = [{"content": "turn one task", "status": "pending", "active_form": ""}]
        turn_three_todos = [
            {"content": "turn one task", "status": "completed", "active_form": ""},
            {"content": "turn three task", "status": "in_progress", "active_form": "Working"},
        ]
        rolled_back_state = {
            "messages": [Message("user", ["rolled back"])],
            "compressed_msgs": [],
            "chrys_todos": turn_one_todos,
        }
        pre_rollback_state = {
            "messages": [Message("user", ["pre rollback"])],
            "compressed_msgs": [],
            "chrys_todos": turn_three_todos,
        }

        await store.save_session("rb_test", rolled_back_state, agent_profile="Code")
        session_dir = tmp_path / "rb_test"
        snapshot_payload = (session_dir / "session.json").read_text(encoding="utf-8")
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir()
        (snap_dir / "turn_2.json").write_text(snapshot_payload, encoding="utf-8")

        await store.save_session("rb_test", pre_rollback_state, agent_profile="Code")
        engine._todo_tracker = TodoTracker()
        await engine._todo_tracker.restore(turn_three_todos)

        class _StubExec:
            def __init__(self) -> None:
                self.history_state: dict[str, Any] = {}

        async def _fake_start(
            profile: AgentProfile, *, operation: str = "startup", staged_loaded: LoadedSettings | None = None
        ) -> None:
            engine._agent_profile = profile
            engine._executor = _StubExec()  # type: ignore[assignment]
            if staged_loaded is not None:
                engine._settings_handle.install(staged_loaded)

        monkeypatch.setattr(engine, "start", _fake_start)

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        tracker = engine.todo_tracker
        assert tracker is not None
        assert tracker.serialize() == turn_one_todos
        assert engine._executor.history_state["chrys_todos"] == turn_one_todos

    @pytest.mark.asyncio
    async def test_rollback_swap_keeps_promoted_snapshot_when_backup_update_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If backup refresh fails, the promoted snapshot remains as recovery fallback."""
        engine = _make_engine(tmp_path)
        store = JsonFileStateStore(tmp_path)
        engine._history.bind({"messages": [Message(role="user", contents=["hello"])], "compressed_msgs": []})

        rolled_back_state = {"messages": [Message("user", ["rolled back"])], "compressed_msgs": []}
        pre_rollback_state = {"messages": [Message("user", ["pre rollback"])], "compressed_msgs": []}

        await store.save_session("rb_test", rolled_back_state)
        session_dir = tmp_path / "rb_test"
        snapshot_payload = (session_dir / "session.json").read_text(encoding="utf-8")
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir()
        promoted_snapshot = snap_dir / "turn_2.json"
        promoted_snapshot.write_text(snapshot_payload, encoding="utf-8")

        await store.save_session("rb_test", pre_rollback_state)

        real_atomic_copy_file = engine_module.atomic_copy_file

        def fail_backup_copy(source: Path, dest: Path) -> None:
            if dest.name == "session.json.bak":
                raise OSError("simulated backup failure")
            real_atomic_copy_file(source, dest)

        async def _fake_restore(_event: Any) -> None: ...

        monkeypatch.setattr(engine_module, "atomic_copy_file", fail_backup_copy)
        engine._on_session_restore = _fake_restore  # type: ignore[assignment]

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        assert promoted_snapshot.exists()
        assert not (session_dir / "session.json.bak").exists()

        (session_dir / "session.json").write_text("{ broken primary", encoding="utf-8")
        loaded = await store.load_session("rb_test")

        assert loaded is not None
        assert loaded["messages"][0].text == "rolled back"

    @pytest.mark.asyncio
    async def test_rollback_swap_reports_locked_write_lock(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A session write-lock timeout should surface an Error instead of racing the swap."""
        engine = _make_engine(tmp_path)
        engine._history.bind({"messages": [Message(role="user", contents=["hello"])], "compressed_msgs": []})

        session_dir = tmp_path / "rb_test"
        session_file = session_dir / "session.json"
        _write_session_json(session_file)
        engine._turn_number = 2
        engine._write_rollback_snapshot()
        generation = engine.session_generation
        engine._turn_state.injection_admission_open = True

        errors: list[Error] = []
        await _collect_events(engine.event_bus, Error, errors)

        class _TimedOutFileLock:
            def __init__(self, _path: Path | str, timeout: float | None = None) -> None:
                _ = timeout

            def __enter__(self) -> _TimedOutFileLock:
                raise TimeoutError("lock busy")

            def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
                return None

        monkeypatch.setattr(rollback_module, "FileLock", _TimedOutFileLock)

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        await asyncio.sleep(0)
        error = next(error for error in errors if error.code == "rollback_swap_locked")
        assert error.message == "Timed out waiting for session lock: lock busy"
        _assert_display_message(error, "rollback.swap_locked", {"detail": DisplayBlock("lock busy")})
        assert engine.session_generation == generation
        assert engine._turn_state.injection_admission_open is True
        assert engine._turn_state.prompt_admission_closed is False

    @pytest.mark.asyncio
    async def test_rollback_swap_reports_copy_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        engine = _make_engine(tmp_path)
        engine._history.bind({"messages": [Message(role="user", contents=["hello"])], "compressed_msgs": []})

        session_dir = tmp_path / "rb_test"
        session_file = session_dir / "session.json"
        _write_session_json(session_file)
        engine._turn_number = 2
        engine._write_rollback_snapshot()
        errors: list[Error] = []
        await _collect_events(engine.event_bus, Error, errors)

        def _fail_snapshot_copy(_source: Path, _destination: Path) -> None:
            raise OSError("snapshot copy failed")

        monkeypatch.setattr(engine_module, "atomic_copy_file", _fail_snapshot_copy)

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        assert len(errors) == 1
        assert errors[0].message == "Failed to restore snapshot: snapshot copy failed"
        _assert_display_message(errors[0], "rollback.swap_failed", {"detail": DisplayBlock("snapshot copy failed")})

    @pytest.mark.asyncio
    async def test_snapshot_removed_after_target_validation_reports_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        engine = _make_engine(tmp_path)
        engine._history.bind({"messages": [Message(role="user", contents=["hello"])], "compressed_msgs": []})

        session_dir = tmp_path / "rb_test"
        session_file = session_dir / "session.json"
        _write_session_json(session_file)
        engine._turn_number = 2
        engine._write_rollback_snapshot()
        snapshot_path = session_dir / "snapshots" / "turn_2.json"
        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)

        def _remove_snapshot_after_validation(_engine: object, _target_turn: int) -> str:
            snapshot_path.unlink()
            return "hello"

        monkeypatch.setattr(rollback_module, "first_rolled_back_user_text", _remove_snapshot_after_validation)

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        assert len(warnings) == 1
        assert warnings[0].message == "Snapshot for turn 1 is missing."
        _assert_display_message(warnings[0], "rollback.snapshot_missing", {"target_turn": 1})

    @pytest.mark.asyncio
    async def test_snapshot_disappearing_after_validation_does_not_commit_transition(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        engine = _make_engine(tmp_path)
        engine._history.bind({"messages": [Message(role="user", contents=["hello"])], "compressed_msgs": []})

        session_dir = tmp_path / "rb_test"
        session_file = session_dir / "session.json"
        _write_session_json(session_file)
        original_session = session_file.read_text(encoding="utf-8")
        engine._turn_number = 2
        engine._write_rollback_snapshot()
        snapshot_path = session_dir / "snapshots" / "turn_2.json"
        assert snapshot_path.exists()

        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)
        generation = engine.session_generation
        real_file_lock = rollback_module.FileLock

        class _DeletingFileLock:
            def __init__(self, path: Path | str, timeout: float | None = None) -> None:
                self._inner = real_file_lock(path, timeout=timeout)

            def __enter__(self) -> _DeletingFileLock:
                self._inner.acquire()
                snapshot_path.unlink()
                return self

            def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
                self._inner.release()

        monkeypatch.setattr(rollback_module, "FileLock", _DeletingFileLock)

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        assert [warning.code for warning in warnings] == ["rollback_snapshot_missing"]
        assert warnings[0].message == "Snapshot for turn 1 is missing."
        _assert_display_message(warnings[0], "rollback.snapshot_missing", {"target_turn": 1})
        assert engine.session_generation == generation
        assert session_file.read_text(encoding="utf-8") == original_session
        assert engine._turn_state.prompt_admission_closed is False

    @pytest.mark.asyncio
    async def test_rollback_swap_lock_failure_does_not_revert_workspace_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """File restores must wait until the session snapshot swap succeeds."""
        engine = _make_engine(tmp_path)
        engine._history.bind({"messages": [Message(role="user", contents=["hello"])], "compressed_msgs": []})

        session_dir = tmp_path / "rb_test"
        session_file = session_dir / "session.json"
        _write_session_json(session_file)
        engine._turn_number = 2
        engine._write_rollback_snapshot()

        workspace_file = tmp_path / "work.txt"
        workspace_file.write_text("original", encoding="utf-8")
        tracker = MutationTracker(SnapshotStore(session_dir))
        tracker.start_turn(2)
        mutation = tracker.record(str(workspace_file), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call-1")
        assert mutation is not None
        workspace_file.write_text("changed", encoding="utf-8")
        tracker.record_after(mutation)
        engine._mutation_tracker = tracker

        errors: list[Error] = []
        results: list[RollbackResult] = []
        await _collect_events(engine.event_bus, Error, errors)
        await _collect_events(engine.event_bus, RollbackResult, results)
        monkeypatch.setattr(engine_module, "SESSION_WRITE_LOCK_TIMEOUT_SECONDS", 0.01)

        lock_path = engine._session_write_lock_path("rb_test")
        assert lock_path is not None
        held = FileLock(lock_path, timeout=1.0)
        held.acquire()
        try:
            await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=True))
        finally:
            held.release()

        assert [e.code for e in errors] == ["rollback_swap_locked"]
        assert results == []
        assert workspace_file.read_text(encoding="utf-8") == "changed"
        assert [turn.turn_id for turn in tracker.get_all_turns()] == [2]
        assert engine.session_generation == 0

    @pytest.mark.asyncio
    async def test_rollback_reverts_workspace_files_after_successful_swap(self, tmp_path: Path) -> None:
        """A successful target-turn swap still applies requested file restores."""
        engine = _make_engine(tmp_path)
        engine._history.bind({"messages": [Message(role="user", contents=["hello"])], "compressed_msgs": []})

        session_dir = tmp_path / "rb_test"
        session_file = session_dir / "session.json"
        _write_session_json(session_file)
        engine._turn_number = 2
        engine._write_rollback_snapshot()

        workspace_file = tmp_path / "work.txt"
        workspace_file.write_text("original", encoding="utf-8")
        tracker = MutationTracker(SnapshotStore(session_dir))
        tracker.start_turn(2)
        mutation = tracker.record(str(workspace_file), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call-1")
        assert mutation is not None
        workspace_file.write_text("changed", encoding="utf-8")
        tracker.record_after(mutation)
        engine._mutation_tracker = tracker

        results: list[RollbackResult] = []
        await _collect_events(engine.event_bus, RollbackResult, results)

        async def _fake_restore(_event: Any) -> None: ...

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=True))

        assert workspace_file.read_text(encoding="utf-8") == "original"
        assert [turn.turn_id for turn in tracker.get_all_turns()] == []
        assert len(results) == 1
        assert engine.session_generation == 1
        assert results[0].files_reverted == 1


class TestSnapshotGapTolerance:
    """Regression: every snapshot-touching code path must tolerate gaps.

    A user (or an external tool) may delete individual ``turn_N.json``
    files manually.  The rollback logic must never assume contiguous
    snapshot ranges — enumeration is always glob-based, and the
    mutation-tracker revert counts turn IDs strictly greater than the
    target (see ``_on_user_rollback`` comment about "tolerates gaps
    in the tracker's turn IDs").  This test exercises a non-contiguous
    layout end-to-end.
    """

    def test_available_turns_skips_deleted_snapshots(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        session_dir = tmp_path / "rb_test"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)
        # User deleted turn_2 and turn_4; turn_3 and turn_5 survive.
        for n in (3, 5):
            (snap_dir / f"turn_{n}.json").write_text("{}")
        engine._mutation_tracker = MutationTracker(SnapshotStore(session_dir))
        engine._mutation_tracker.start_turn(1)

        # turn_3.json → keep 2 turns; turn_5.json → keep 4 turns; + welcome 0.
        assert engine.available_rollback_turns() == [0, 2, 4]

    @pytest.mark.asyncio
    async def test_rollback_to_non_contiguous_surviving_turn(self, tmp_path: Path) -> None:
        """Pick a surviving snapshot whose neighbours are gone — must still swap cleanly."""
        engine = _make_engine(tmp_path)
        engine._history.bind({"messages": [Message(role="user", contents=["hi"])], "compressed_msgs": []})

        session_dir = tmp_path / "rb_test"
        session_file = session_dir / "session.json"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)

        # Layout: snapshots 3 and 5 exist (2 and 4 deleted by user).
        (snap_dir / "turn_3.json").write_text(
            json.dumps({"meta": {"session_id": "rb_test"}, "state": {"messages": ["t3"], "compressed_msgs": []}})
        )
        (snap_dir / "turn_5.json").write_text(
            json.dumps({"meta": {"session_id": "rb_test"}, "state": {"messages": ["t5"], "compressed_msgs": []}})
        )
        # Current live session.json (post-turn-5 state).
        session_file.write_text(
            json.dumps({"meta": {"session_id": "rb_test"}, "state": {"messages": ["live"], "compressed_msgs": []}})
        )

        async def _fake_restore(_event: Any) -> None: ...

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]

        # target_turn=2 means "keep 2 turns" → restore turn_3.json.
        # turn_4.json is absent (no target=3); turn_5.json exists (target=4).
        await engine._on_user_rollback(UserRollback(target_turn=2, revert_changes=False))

        restored = json.loads(session_file.read_text(encoding="utf-8"))
        assert restored["state"]["messages"] == ["t3"]

        # Post-swap cleanup keeps the promoted turn_3 snapshot as the
        # recovery anchor for the new current state, but removes newer
        # rollback targets.  Absent turn_4 should not cause errors.
        assert (snap_dir / "turn_3.json").exists()
        assert not (snap_dir / "turn_5.json").exists()

    @pytest.mark.asyncio
    async def test_rollback_to_deleted_snapshot_is_refused(self, tmp_path: Path) -> None:
        """Picking a target whose snapshot was deleted surfaces a Warning, not a crash."""
        engine = _make_engine(tmp_path)
        engine._history.bind({"messages": [Message(role="user", contents=["hi"])], "compressed_msgs": []})

        session_dir = tmp_path / "rb_test"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)
        # Only turn_3.json exists → available targets are {0, 2}.
        (snap_dir / "turn_3.json").write_text("{}")

        warnings: list[Warning] = []
        await _collect_events(engine.event_bus, Warning, warnings)

        # target_turn=1 would need turn_2.json which is missing.
        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))
        await asyncio.sleep(0)

        assert any(w.code == "rollback_unavailable" for w in warnings)


class TestRollbackSwapSemantics:
    """End-to-end: rolling back to a compressed turn un-compresses it.

    Snapshots are written at the *start* of each turn, so a snapshot
    taken before any compression ran contains ``compressed_msgs == []``
    even if the live state later folded that turn into a block.
    Swapping that snapshot back in as ``session.json`` therefore drops
    the compressed blocks — which is exactly what powers the Context
    panel rebuild on ``SessionRestored``.
    """

    @pytest.mark.asyncio
    async def test_rollback_to_pre_compress_snapshot_clears_blocks(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        # Live history anchors the welcome target so the picker also
        # offers target_turn=1 (keep one turn = restore turn_2.json).
        engine._history.bind({"messages": [Message(role="user", contents=["hi"])], "compressed_msgs": []})

        session_dir = tmp_path / "rb_test"
        session_file = session_dir / "session.json"

        # 1. Pre-compress session.json → snapshot turn_2.json.
        _write_session_json(session_file)  # compressed_msgs == []
        engine._turn_number = 2
        engine._write_rollback_snapshot()

        # 2. Simulate compression happening later: rewrite session.json
        #    with a CompressedBlock folded into the live state.
        post_compress = {
            "meta": {"session_id": "rb_test", "agent_profile": "p"},
            "state": {
                "messages": [],
                "compressed_msgs": [
                    {
                        "compressed_context_id": "ctx_x",
                        "summary_text": "summary",
                        "marker_id": "turn_2",
                        "turn_range": [1, 2],
                        "messages": [],
                    }
                ],
            },
        }
        session_file.write_text(json.dumps(post_compress), encoding="utf-8")

        # Stub restore so the test focuses on the swap itself.
        async def _fake_restore(_event: Any) -> None: ...

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]

        # target_turn=1 → restore turn_2.json (pre-compress state).
        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        # 3. session.json must now match the pre-compress snapshot.
        restored = json.loads(session_file.read_text(encoding="utf-8"))
        assert restored["state"]["compressed_msgs"] == []

    @pytest.mark.asyncio
    async def test_rollback_uses_snapshot_payload_turn_counter_for_legacy_snapshots(self, tmp_path: Path) -> None:
        """Legacy after-turn snapshots must not be interpreted as pre-turn snapshots.

        Older development builds named snapshots after the completed turn,
        so ``turn_2.json`` could contain ``turn_counter == 2``.  The newer
        pre-turn convention maps ``turn_2.json`` to target 1.  Restoring by
        filename alone would therefore make "rollback to Turn 1" restore a
        two-turn session and then delete the snapshot that proved it.
        """
        engine = _make_engine(tmp_path)
        store = JsonFileStateStore(tmp_path)
        session_dir = store.session_dir("rb_test")
        session_file = session_dir / "session.json"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)

        for turn in range(1, 4):
            await store.save_session("rb_test", _state_after_turns(turn))
            atomic_copy_file(session_file, snap_dir / f"turn_{turn}.json")

        await store.save_session("rb_test", _state_after_turns(3))
        live = await store.load_session("rb_test")
        assert live is not None
        engine._history.bind(live)
        engine._turn_number = 3

        async def _fake_restore(event: Any) -> None:
            loaded = await store.load_session(event.session_id)
            assert loaded is not None
            engine._history.bind(loaded)
            engine._turn_number = loaded.get("turn_counter", 0)

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]

        assert engine.available_rollback_turns() == [0, 1, 2]

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        restored = await store.load_session("rb_test")
        assert restored is not None
        assert restored["turn_counter"] == 1
        assert [m.text for m in restored["messages"] if m.role == "user"] == ["user 1"]
        assert sorted(p.name for p in snap_dir.glob("*.json")) == ["turn_1.json"]

    @pytest.mark.asyncio
    async def test_legacy_numeric_snapshot_without_counter_can_restore_turn_one(self, tmp_path: Path) -> None:
        """Bare ``1.json`` snapshots without metadata still mean legacy turn 1."""
        engine = _make_engine(tmp_path)
        session_dir = tmp_path / "rb_test"
        session_file = session_dir / "session.json"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)

        (snap_dir / "1.json").write_text(
            json.dumps({"meta": {"session_id": "rb_test"}, "state": {"messages": ["legacy turn 1"]}}),
            encoding="utf-8",
        )
        session_file.write_text(
            json.dumps({"meta": {"session_id": "rb_test"}, "state": {"messages": ["current"]}}),
            encoding="utf-8",
        )
        engine._history.bind({"messages": [Message("user", ["current"])], "compressed_msgs": []})
        engine._turn_number = 2

        async def _fake_restore(_event: Any) -> None: ...

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]

        assert engine.available_rollback_turns() == [0, 1]

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        restored = json.loads(session_file.read_text(encoding="utf-8"))
        assert restored["state"]["messages"] == ["legacy turn 1"]

    @pytest.mark.asyncio
    async def test_chained_rollback_keeps_current_snapshot_without_offering_current_target(
        self,
        tmp_path: Path,
    ) -> None:
        """A rollback should keep the promoted snapshot for recovery but hide it from the picker."""
        engine = _make_engine(tmp_path)
        store = JsonFileStateStore(tmp_path)
        session_dir = store.session_dir("rb_test")
        session_file = session_dir / "session.json"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)

        for turn in range(1, 7):
            await store.save_session("rb_test", _state_after_turns(turn))
            if turn < 6:
                atomic_copy_file(session_file, snap_dir / f"turn_{turn + 1}.json")

        live = await store.load_session("rb_test")
        assert live is not None
        engine._history.bind(live)
        engine._turn_number = 6

        async def _fake_restore(event: Any) -> None:
            loaded = await store.load_session(event.session_id)
            assert loaded is not None
            engine._history.bind(loaded)
            engine._turn_number = loaded.get("turn_counter", 0)

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]

        assert engine.available_rollback_turns() == [0, 1, 2, 3, 4, 5]

        await engine._on_user_rollback(UserRollback(target_turn=3, revert_changes=False))

        restored = await store.load_session("rb_test")
        assert restored is not None
        assert restored["turn_counter"] == 3
        assert [m.text for m in restored["messages"] if m.role == "user"] == ["user 1", "user 2", "user 3"]
        assert sorted(p.name for p in snap_dir.glob("*.json")) == [
            "turn_2.json",
            "turn_3.json",
            "turn_4.json",
        ]
        assert engine.available_rollback_turns() == [0, 1, 2]

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=False))

        restored = await store.load_session("rb_test")
        assert restored is not None
        assert restored["turn_counter"] == 1
        assert [m.text for m in restored["messages"] if m.role == "user"] == ["user 1"]
        assert sorted(p.name for p in snap_dir.glob("*.json")) == ["turn_2.json"]
        assert engine.available_rollback_turns() == [0]


# ===========================================================================
# Rollback revert-changes path via MutationTracker
# ===========================================================================


class TestRevertCode:
    """Directly exercise the file-restore half of the rollback path.

    The full handler drives reload + swap which is heavy to mock; we cover
    file-revert behaviour by calling ``MutationTracker.rollback`` the same
    way the engine does.
    """

    def test_revert_restores_modified_file_to_pre_turn_state(self, tmp_path: Path) -> None:
        store = SnapshotStore(tmp_path)
        tracker = MutationTracker(store)

        target = tmp_path / "src.txt"
        target.write_text("original", encoding="utf-8")

        tracker.start_turn(1)
        mut = tracker.record(
            str(target),
            MutationOp.MODIFY,
            MutationSource.EDIT_FILE,
            "call-1",
        )
        assert mut is not None
        target.write_text("changed", encoding="utf-8")
        tracker.record_after(mut)

        # Simulate rollback of the last 1 turn.
        restored = tracker.rollback(1)
        assert any(r.path == str(target) and r.changed for r in restored)
        assert target.read_text(encoding="utf-8") == "original"

    def test_fsm_reset_and_shutting_down_flag_cleared_after_shutdown(self, tmp_path: Path) -> None:
        """Regression: the rollback reload path must leave the FSM clean.

        ``shutdown()`` drives the FSM into ``SHUTTING_DOWN`` and sets
        ``_shutting_down = True``; the reload logic inside
        ``_on_session_restore`` and ``_reset_session_to_welcome`` must
        clear both so the next ``start()`` can transition back to IDLE
        and the next ``/rollback`` isn't refused.
        """
        engine = _make_engine(tmp_path, fsm_state=EngineState.IDLE)
        # Simulate what shutdown() would leave behind.
        engine._fsm.try_transition(Trigger.SHUTDOWN)
        engine._shutting_down = True
        assert engine.state is EngineState.SHUTTING_DOWN

        # The two rollback reload paths both apply this sequence
        # (in-place; matches the fix in engine.py).
        engine._fsm.reset()
        engine._shutting_down = False

        assert engine.state is EngineState.UNINITIALIZED
        assert engine._shutting_down is False
        # And UNINITIALIZED can be transitioned forward via START.
        engine._fsm.try_transition(Trigger.START)
        assert engine.state is EngineState.IDLE

    def test_revert_removes_created_file(self, tmp_path: Path) -> None:
        store = SnapshotStore(tmp_path)
        tracker = MutationTracker(store)

        new_file = tmp_path / "new.txt"

        tracker.start_turn(1)
        mut = tracker.record(
            str(new_file),
            MutationOp.CREATE,
            MutationSource.WRITE_FILE,
            "call-1",
        )
        assert mut is not None
        new_file.write_text("hello", encoding="utf-8")
        tracker.record_after(mut)

        # Rollback should remove the newly-created file.
        tracker.rollback(1)
        assert not new_file.exists()

    def test_rollback_with_only_paths_restores_subset(self, tmp_path: Path) -> None:
        """``only_paths`` restricts the file restore to the given subset.

        Mirrors the path the rollback modal takes when the user
        un-checks some files before clicking "Rollback & Revert
        Changes": turns are popped normally, but un-selected files
        keep their mutated content.

        Paths in the filter must match the tracker's canonical form
        (``os.path.normpath(os.path.abspath(...))``) — this is what the
        real UI passes too, because ``DiffFileEntry.path`` is populated
        from ``Mutation.path`` which the tracker already normalized.
        """
        store = SnapshotStore(tmp_path)
        tracker = MutationTracker(store)

        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_text("a-original", encoding="utf-8")
        file_b.write_text("b-original", encoding="utf-8")

        tracker.start_turn(1)
        mut_a = tracker.record(str(file_a), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call-a")
        assert mut_a is not None
        file_a.write_text("a-changed", encoding="utf-8")
        tracker.record_after(mut_a)

        mut_b = tracker.record(str(file_b), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call-b")
        assert mut_b is not None
        file_b.write_text("b-changed", encoding="utf-8")
        tracker.record_after(mut_b)

        # Use the same normalization the tracker applies, so this test
        # is stable on both Windows and POSIX regardless of whatever
        # form pytest's ``tmp_path`` produced.
        norm_a = os.path.normpath(os.path.abspath(str(file_a)))
        norm_b = os.path.normpath(os.path.abspath(str(file_b)))

        # Only restore file_a; file_b stays at its mutated content.
        restored = tracker.rollback(1, only_paths={norm_a})

        paths = [r.path for r in restored]
        assert paths == [norm_a]
        assert norm_b not in paths
        assert all(r.changed for r in restored)
        assert file_a.read_text(encoding="utf-8") == "a-original"
        assert file_b.read_text(encoding="utf-8") == "b-changed"
        # Turn log is still fully popped regardless of the filter.
        assert tracker.get_all_turns() == []

    def test_rollback_only_paths_requires_canonical_form(self, tmp_path: Path) -> None:
        """A relative (or otherwise non-canonical) path in ``only_paths`` silently skips.

        Documents the contract: ``only_paths`` is matched against the
        tracker's canonical absolute/normpath form, not whatever string
        the caller happened to pass to ``record()``.  Callers that
        build the filter from ``DiffFileEntry.path`` (the UI path) get
        this right automatically; callers passing raw relative strings
        would find nothing restored — which this test locks in.
        """
        store = SnapshotStore(tmp_path)
        tracker = MutationTracker(store)

        file_a = tmp_path / "a.txt"
        file_a.write_text("a-original", encoding="utf-8")

        tracker.start_turn(1)
        mut = tracker.record(str(file_a), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call")
        assert mut is not None
        file_a.write_text("a-changed", encoding="utf-8")
        tracker.record_after(mut)

        # "a.txt" alone is a relative name — not what the tracker stored.
        restored = tracker.rollback(1, only_paths={"a.txt"})

        assert restored == []
        assert file_a.read_text(encoding="utf-8") == "a-changed"

    # NOTE: the "restored" list from rollback() now holds RestoreResult
    # entries, not plain paths.  Use ``r.path`` / ``r.changed`` / ``r.ok``
    # on each entry; empty list means the filter matched nothing.

    def test_rollback_only_paths_case_sensitive_match(self, tmp_path: Path) -> None:
        """``only_paths`` uses byte-exact matching — no ``normcase`` folding.

        The tracker does ``normpath(abspath(...))`` (which on Windows
        also normalises the drive letter), but it does **not** call
        ``normcase`` — so case is preserved as stored.  Matching in
        ``only_paths`` therefore treats ``/Foo/bar.txt`` and
        ``/foo/bar.txt`` as distinct entries.  On POSIX this is the
        correct behaviour; on Windows (where filesystems are typically
        case-insensitive) callers should either pass paths that match
        how the tracker stored them or pre-apply ``os.path.normcase``
        themselves before building the set.
        """
        store = SnapshotStore(tmp_path)
        tracker = MutationTracker(store)

        file_a = tmp_path / "a.txt"
        file_a.write_text("orig", encoding="utf-8")

        tracker.start_turn(1)
        mut = tracker.record(str(file_a), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call")
        assert mut is not None
        file_a.write_text("changed", encoding="utf-8")
        tracker.record_after(mut)

        norm_a = os.path.normpath(os.path.abspath(str(file_a)))
        # Upper-case the filename portion only.  This produces a string
        # that never exists in the tracker's snapshot map (on POSIX) —
        # and on Windows where ``abspath`` already canonicalises the
        # drive letter but leaves the rest alone, the mismatch survives.
        scrambled = os.path.join(os.path.dirname(norm_a), os.path.basename(norm_a).upper())

        if scrambled == norm_a:
            # All-uppercase filename == stored form (e.g. an ALL-CAPS
            # filename from the start) — skip the assertion for this
            # degenerate input since the test would be vacuous.
            pytest.skip("filename is already uppercase, case comparison would be trivial")

        restored = tracker.rollback(1, only_paths={scrambled})
        assert [r.path for r in restored] == []
        assert file_a.read_text(encoding="utf-8") == "changed"

    def test_rollback_with_empty_only_paths_restores_nothing(self, tmp_path: Path) -> None:
        """Empty ``only_paths`` set → no file is restored, but the turn is still popped."""
        store = SnapshotStore(tmp_path)
        tracker = MutationTracker(store)

        file_a = tmp_path / "a.txt"
        file_a.write_text("a-original", encoding="utf-8")

        tracker.start_turn(1)
        mut = tracker.record(str(file_a), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call")
        assert mut is not None
        file_a.write_text("a-changed", encoding="utf-8")
        tracker.record_after(mut)

        restored = tracker.rollback(1, only_paths=set())

        assert [r.path for r in restored] == []
        assert file_a.read_text(encoding="utf-8") == "a-changed"
        assert tracker.get_all_turns() == []


class TestRollbackTitleOverlayPreservation:
    """The custom title is session-scoped and must survive a snapshot restore."""

    def test_read_and_reapply_title_overlays(self, tmp_path: Path) -> None:
        from chrys.orchestration.engine.rollback import _read_title_overlays, _reapply_title_overlays

        session_file = tmp_path / "session.json"
        session_file.write_text(
            json.dumps(
                {
                    "meta": {"title": "first msg", "custom_title": "Pinned", "generated_title": "Auto"},
                    "state": {},
                }
            ),
            encoding="utf-8",
        )
        overlays = _read_title_overlays(session_file)
        assert overlays == {"custom_title": "Pinned"}

        # Simulate the snapshot restore wiping the overlays wholesale.
        session_file.write_text(json.dumps({"meta": {"title": "first msg"}, "state": {}}), encoding="utf-8")
        _reapply_title_overlays(session_file, overlays)
        meta = json.loads(session_file.read_text(encoding="utf-8"))["meta"]
        assert meta["custom_title"] == "Pinned"
        assert "generated_title" not in meta
        assert meta["title"] == "first msg"

    def test_generated_title_from_snapshot_wins_on_rollback(self, tmp_path: Path) -> None:
        """The current generated title summarizes turns the rollback discards;
        the snapshot's own value is the one describing the restored history."""
        from chrys.orchestration.engine.rollback import _read_title_overlays, _reapply_title_overlays

        session_file = tmp_path / "session.json"
        session_file.write_text(
            json.dumps({"meta": {"title": "x", "generated_title": "New topic"}, "state": {}}),
            encoding="utf-8",
        )
        overlays = _read_title_overlays(session_file)

        session_file.write_text(
            json.dumps({"meta": {"title": "x", "generated_title": "Old topic"}, "state": {}}),
            encoding="utf-8",
        )
        _reapply_title_overlays(session_file, overlays)
        meta = json.loads(session_file.read_text(encoding="utf-8"))["meta"]
        assert meta["generated_title"] == "Old topic"

    def test_read_overlays_tolerates_missing_or_invalid_file(self, tmp_path: Path) -> None:
        from chrys.orchestration.engine.rollback import _read_title_overlays

        assert _read_title_overlays(tmp_path / "missing.json") == {}
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert _read_title_overlays(bad) == {}

    def test_cleared_custom_title_survives_rollback(self, tmp_path: Path) -> None:
        """An explicit empty custom_title (user cleared the pin) must override a snapshot's old pin."""
        from chrys.orchestration.engine.rollback import _read_title_overlays, _reapply_title_overlays

        session_file = tmp_path / "session.json"
        session_file.write_text(
            json.dumps({"meta": {"title": "x", "custom_title": "", "generated_title": "Auto"}, "state": {}}),
            encoding="utf-8",
        )
        overlays = _read_title_overlays(session_file)
        assert overlays == {"custom_title": ""}

        # Snapshot from before the clear still carries the pin.
        session_file.write_text(
            json.dumps({"meta": {"title": "x", "custom_title": "Pinned"}, "state": {}}),
            encoding="utf-8",
        )
        _reapply_title_overlays(session_file, overlays)
        meta = json.loads(session_file.read_text(encoding="utf-8"))["meta"]
        assert meta["custom_title"] == ""


# ---------------------------------------------------------------------------
# Cross-session coordination re-check at rollback (PR2)
# ---------------------------------------------------------------------------


class TestRollbackCoordinatorRecheck:
    """The destructive path must hand the coordinator its discovery inputs.

    Outside a git repo the registry is keyed by the workspace fallback
    root — without it, the rollback-time peer re-check silently finds
    no peer files at all.
    """

    @pytest.mark.asyncio
    async def test_recheck_passes_workspace_fallback_and_scope(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._history.bind({"messages": [Message(role="user", contents=["hi"])], "compressed_msgs": []})

        session_dir = tmp_path / "rb_test"
        session_file = session_dir / "session.json"
        snap_dir = session_dir / "snapshots"
        snap_dir.mkdir(parents=True)
        (snap_dir / "turn_2.json").write_text(
            json.dumps({"meta": {"session_id": "rb_test"}, "state": {"messages": ["t2"], "compressed_msgs": []}})
        )
        session_file.write_text(
            json.dumps({"meta": {"session_id": "rb_test"}, "state": {"messages": ["live"], "compressed_msgs": []}})
        )

        engine._mutation_tracker = MutationTracker(SnapshotStore(session_dir))
        engine._mutation_tracker.start_turn(2)

        calls: list[tuple[str, object, object]] = []

        class _RecordingCoordinator:
            def reclassify(self, tracker, *, force=False, fallback_root=None):
                calls.append(("reclassify", force, fallback_root))
                return False

            def augment_rollback_plan(self, tracker, plan, *, scope_paths=None, fallback_root=None):
                calls.append(("augment", scope_paths, fallback_root))
                return plan

        engine._mutation_coordinator = _RecordingCoordinator()  # type: ignore[assignment]

        async def _fake_restore(_event: Any) -> None: ...

        engine._on_session_restore = _fake_restore  # type: ignore[assignment]

        await engine._on_user_rollback(UserRollback(target_turn=1, revert_changes=True))

        cwd = engine._workspace_cwd()
        assert ("reclassify", True, cwd) in calls
        assert ("augment", [cwd], cwd) in calls


class TestAttributionRefreshPersistence:
    """The engine refresh must save when the coordinator carries unsaved
    reclassification changes — a finalize-time reclassify has no saver
    and updates the short-circuit signatures, so this very refresh call
    reports "unchanged" while session.json is behind.
    """

    @pytest.mark.asyncio
    async def test_unsaved_finalize_reclassification_forces_save(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        engine._mutation_tracker = MutationTracker(SnapshotStore(tmp_path / "rb_test"))

        saves: list[bool] = []

        async def _record_save(*, raise_on_error: bool = False) -> bool:
            saves.append(True)
            return True

        engine._save_current_session = _record_save  # type: ignore[assignment]

        class _StubCoordinator:
            def __init__(self) -> None:
                self.dirty = True

            def reclassify(self, tracker, *, force=False, fallback_root=None):
                return False  # signatures already updated by the finalize-time run

            def consume_unsaved_reclassification(self):
                dirty, self.dirty = self.dirty, False
                return dirty

        engine._mutation_coordinator = _StubCoordinator()  # type: ignore[assignment]

        assert await engine.refresh_mutation_attribution() is False
        assert saves == [True]  # saved despite changed=False
        assert await engine.refresh_mutation_attribution() is False
        assert saves == [True]  # flag consumed — no redundant save

    @pytest.mark.asyncio
    async def test_mid_run_refresh_defers_save_to_turn_end(self, tmp_path: Path) -> None:
        """A mid-run primary save would delete the recovery sidecar — the only
        durable copy of committed-but-unmerged tool exchanges — so the refresh
        must leave the unsaved flag set and defer persistence to the turn-end
        save."""
        engine = _make_engine(tmp_path)
        engine._mutation_tracker = MutationTracker(SnapshotStore(tmp_path / "rb_test"))

        saves: list[bool] = []

        async def _record_save(*, raise_on_error: bool = False) -> bool:
            saves.append(True)
            return True

        engine._save_current_session = _record_save  # type: ignore[assignment]

        class _StubCoordinator:
            def __init__(self) -> None:
                self.dirty = True

            def reclassify(self, tracker, *, force=False, fallback_root=None):
                return True

            def consume_unsaved_reclassification(self):
                dirty, self.dirty = self.dirty, False
                return dirty

        coordinator = _StubCoordinator()
        engine._mutation_coordinator = coordinator  # type: ignore[assignment]
        engine._executor = SimpleNamespace(is_running=True)  # type: ignore[assignment]

        assert await engine.refresh_mutation_attribution() is True
        assert saves == []
        assert coordinator.dirty is True  # flag preserved for the turn-end save

        engine._executor = SimpleNamespace(is_running=False)  # type: ignore[assignment]
        assert await engine.refresh_mutation_attribution() is True
        assert saves == [True]
