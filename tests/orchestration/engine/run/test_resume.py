# Copyright (c) 2026 Chrys. All rights reserved.

"""Unit tests for resume/retry logic in Executor and Engine.

Tests:
1. Executor.resume() with completed work continues with empty input
2. Executor.resume() without completed work re-sends original user message
3. Legacy continuation-message cleanup remains read-compatible
4. Integration: retry after interrupt preserves completed tools
5. Integration: retry after error with no progress re-sends original message
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from chrys.kernel import Message
from chrys.orchestration.engine.trajectory import TrajectoryRecorder
from tests.support.waiting import ENGINE_TURN_TIMEOUT, await_run_task_chain, wait_for, wait_until

if TYPE_CHECKING:
    from pathlib import Path

    from chrys.service.session.history import SessionHistoryManager

import chrys.orchestration.engine.build.builder as builder_module
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    AgentMessage,
    AgentRuntimeDetails,
    ContextCompressed,
    Error,
    RetryAttempt,
    RuntimeModelDetails,
    SessionReady,
    SessionRestore,
    SessionRestored,
    SessionSaved,
    UsageUpdate,
    UserInterrupt,
    UserMessage,
    UserRetry,
)
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.orchestration.engine.engine import AgentEngine
from chrys.orchestration.engine.executor import Executor
from chrys.orchestration.engine.run.coordinator import TurnCoordinator
from chrys.orchestration.engine.run.runner import TurnRunner
from chrys.orchestration.engine.run.turn_state import TurnRuntimeState
from chrys.orchestration.engine.state.machine import EngineState, EngineStateMachine, Trigger
from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
from chrys.service.context.compaction import UnifiedContextStrategy
from chrys.service.context.providers.history import (
    PRE_OUTPUT_HISTORY_LEN_STATE_KEY,
    CompressibleHistoryProvider,
)
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker
from chrys.service.profiles.agents.schema import AgentProfile, ApprovalConfig, CompactionConfig, ToolsConfig
from chrys.service.state.store import JsonFileStateStore
from tests.support.pipeline_helpers import make_mock_settings_and_registry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROFILE = AgentProfile(
    name="Code",
    display_name="Code Agent",
    instructions="You are a coding assistant.",
    tools=ToolsConfig(builtins=[]),
    approval=ApprovalConfig(default="auto"),
    compaction=CompactionConfig(enabled=False),
)


async def retry_and_save(
    host: object,
    additional_text: str = "",
    created_at: object = None,
    **kwargs: object,
) -> None:
    await TurnRunner(host).run_retry(additional_text, created_at=created_at, **kwargs)  # type: ignore[arg-type]


async def post_run(host: object) -> None:
    await TurnRunner(host).finalize_current_run()  # type: ignore[arg-type]


async def on_user_retry(host: object, event: UserRetry) -> None:
    await TurnCoordinator(host).on_user_retry(event)  # type: ignore[arg-type]


def _make_registry():
    from chrys.service.profiles.agents.registry import AgentProfileRegistry

    registry = AgentProfileRegistry()
    registry.register(_PROFILE)
    return registry


async def _collect(events: list, event: object) -> None:
    events.append(event)


def _filter(events: list, cls: type) -> list:
    return [e for e in events if isinstance(e, cls)]


def _history_messages(engine: AgentEngine) -> list:
    state = engine._executor.history_state
    return state.get("messages", [])


async def _wait_for_call_count(
    client: MockChatClient,
    count: int,
    timeout: float = ENGINE_TURN_TIMEOUT,
) -> None:
    """Poll until the mock client has begun serving its ``count``-th LLM call.

    ``call_count`` bumps synchronously when a call reaches the raw client, so
    this marks "the run is mid-stream".  A fixed sleep instead flakes on slow
    runners (Windows CI): the interrupt can land before the run reaches the
    LLM, become a no-op, and the run completes with no interrupted marker.
    """
    await wait_for(
        lambda: client.call_count >= count,
        timeout=timeout,
        description=f"mock client call count >= {count}",
    )


async def _final_agent_messages_after_run(
    engine: AgentEngine,
    events: list,
    timeout: float = ENGINE_TURN_TIMEOUT,
) -> list:
    """Return final messages after a non-strict boundary drain.

    An internally cancelled run is reported as a drain outcome rather than
    raised; these resume tests diagnose it through their event assertions.
    """
    await await_run_task_chain(engine, timeout=timeout)
    return [e for e in events if isinstance(e, AgentMessage) and e.is_final]


# ---------------------------------------------------------------------------
# Unit tests: _remove_continuation_message
# ---------------------------------------------------------------------------


class TestRemoveContinuationMessage:
    """Test SessionHistoryManager.remove_continuation_message in isolation."""

    def _setup_history(self, messages: list[Message]) -> SessionHistoryManager:
        """Create a SessionHistoryManager bound to a state with given messages."""
        from chrys.service.session.history import SessionHistoryManager

        history = SessionHistoryManager()
        history.bind({"messages": messages})
        return history

    def test_removes_continue_after_work(self):
        """'continue' user message after assistant/tool messages is removed."""
        msgs = [
            Message("user", ["original prompt"]),
            Message("assistant", ["tool calls"]),
            Message("tool", ["results"]),
            Message("user", ["continue"]),
            Message("assistant", ["continuation response"]),
        ]
        history = self._setup_history(msgs)
        history.remove_continuation_message()

        assert len(msgs) == 4
        roles = [m.role for m in msgs]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert msgs[0].text == "original prompt"

    def test_does_not_remove_first_user_message(self):
        """'continue' as the first user message is not removed (no preceding work)."""
        msgs = [
            Message("user", ["continue"]),
            Message("assistant", ["response"]),
        ]
        history = self._setup_history(msgs)
        history.remove_continuation_message()

        assert len(msgs) == 2  # unchanged

    def test_does_not_remove_non_continue_message(self):
        """User messages that aren't 'continue' are not removed."""
        msgs = [
            Message("user", ["original"]),
            Message("assistant", ["tools"]),
            Message("tool", ["results"]),
            Message("user", ["something else"]),
        ]
        history = self._setup_history(msgs)
        history.remove_continuation_message()

        assert len(msgs) == 4  # unchanged

    def test_removes_only_first_continue(self):
        """Only the first 'continue' after work is removed."""
        msgs = [
            Message("user", ["original"]),
            Message("assistant", ["batch 1"]),
            Message("tool", ["result 1"]),
            Message("user", ["continue"]),  # ← this one removed
            Message("assistant", ["batch 2"]),
            Message("tool", ["result 2"]),
            Message("user", ["continue"]),  # ← this one stays
        ]
        history = self._setup_history(msgs)
        history.remove_continuation_message()

        assert len(msgs) == 6
        user_msgs = [m for m in msgs if m.role == "user"]
        assert len(user_msgs) == 2
        assert user_msgs[0].text == "original"
        assert user_msgs[1].text == "continue"

    def test_no_messages_no_crash(self):
        """Empty message list doesn't crash."""
        msgs: list[Message] = []
        history = self._setup_history(msgs)
        history.remove_continuation_message()
        assert msgs == []

    def test_no_executor_no_crash(self):
        """Unbound history manager doesn't crash."""
        from chrys.service.session.history import SessionHistoryManager

        history = SessionHistoryManager()
        history.remove_continuation_message()  # no-op when unbound


# ---------------------------------------------------------------------------
# Unit tests: resume() continuation vs retry mode
# ---------------------------------------------------------------------------


class TestResumeMode:
    """Test that resume() picks the right strategy based on conversation state."""

    def _make_session_state(self, messages: list[Message]) -> dict:
        return {"chrys_history": {"messages": messages}}

    @pytest.mark.asyncio
    async def test_resume_with_completed_work_uses_empty_input(self):
        """Completed work resumes from history without a synthetic user message."""
        from unittest.mock import AsyncMock, MagicMock

        executor = MagicMock()
        session = MagicMock()
        session.state = self._make_session_state(
            [
                Message("user", ["original prompt"]),
                Message("assistant", ["tool calls"]),
                Message("tool", ["results"]),
            ]
        )
        executor._session = session
        executor._execute = AsyncMock()
        executor._running = False
        executor._was_interrupted = False
        executor._run_failed = False
        executor._last_error = ""
        from chrys.orchestration.engine.executor import Executor

        # Call resume directly, binding to our mock
        await Executor.resume(executor)

        executor._execute.assert_awaited_once_with([])

    @pytest.mark.asyncio
    async def test_legacy_nudge_is_ignored_as_anchor_and_left_persisted(self):
        """A legacy read-compatible nudge neither becomes input nor gets scrubbed."""
        from unittest.mock import AsyncMock, MagicMock

        legacy_nudge = Message("user", ["continue"])
        legacy_nudge.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
        messages = [
            Message("user", ["original prompt"]),
            Message("assistant", ["completed work"]),
            legacy_nudge,
        ]
        executor = MagicMock()
        session = MagicMock()
        session.state = self._make_session_state(messages)
        executor._session = session
        executor._execute = AsyncMock()
        executor._run_failed = False
        executor._was_interrupted = False
        executor._last_error = ""

        from chrys.orchestration.engine.executor import Executor

        await Executor.resume(executor)

        executor._execute.assert_awaited_once_with([])
        assert messages[-1] is legacy_nudge

    @pytest.mark.asyncio
    async def test_resume_without_completed_work_resends_original(self):
        """When no work after user message, resume pops and re-sends original text."""
        from unittest.mock import AsyncMock, MagicMock

        executor = MagicMock()
        session = MagicMock()
        msgs = [Message("user", ["original prompt"])]
        session.state = self._make_session_state(msgs)
        executor._session = session
        executor._execute = AsyncMock()
        executor._running = False
        executor._was_interrupted = False
        executor._run_failed = False
        executor._last_error = ""
        executor._pending_continuation_token = None
        from chrys.orchestration.engine.executor import Executor

        await Executor.resume(executor)

        executor._execute.assert_called_once()
        call_arg = executor._execute.call_args[0][0]
        assert len(call_arg) == 1
        assert call_arg[0].role == "user"
        assert call_arg[0].text == "original prompt"
        # User message should have been popped
        assert len(msgs) == 0

    @pytest.mark.asyncio
    async def test_resume_with_additional_text_sends_note_not_continue(self):
        """When additional_text is provided, the user's note replaces 'continue'.

        The original user message is **not** popped; the note becomes a
        follow-up user turn within the same logical turn.
        """
        from unittest.mock import AsyncMock, MagicMock

        executor = MagicMock()
        session = MagicMock()
        msgs = [
            Message("user", ["original prompt"]),
            Message("assistant", ["tool calls"]),
            Message("tool", ["results"]),
        ]
        session.state = self._make_session_state(msgs)
        executor._session = session
        executor._execute = AsyncMock()
        executor._running = False
        executor._was_interrupted = False
        executor._run_failed = False
        executor._last_error = ""
        from chrys.orchestration.engine.executor import Executor

        await Executor.resume(executor, additional_text="also check env vars")

        executor._execute.assert_called_once()
        call_arg = executor._execute.call_args[0][0]
        assert len(call_arg) == 1
        assert call_arg[0].role == "user"
        assert call_arg[0].text == "also check env vars"
        # Original messages are untouched — no pop, no mutation.
        assert len(msgs) == 3
        assert msgs[0].text == "original prompt"

    @pytest.mark.asyncio
    async def test_resume_with_additional_text_no_work_after_keeps_orphan(self):
        """Additional text + no-work-after: keep orphan, send note only.

        Without additional_text, the no-work-after branch pops the
        orphan and re-sends it.  With additional_text present we don't
        pop — the original prompt stays and the user's note is sent as
        the retry prompt.
        """
        from unittest.mock import AsyncMock, MagicMock

        executor = MagicMock()
        session = MagicMock()
        msgs = [Message("user", ["original prompt"])]
        session.state = self._make_session_state(msgs)
        executor._session = session
        executor._execute = AsyncMock()
        executor._running = False
        executor._was_interrupted = False
        executor._run_failed = False
        executor._last_error = ""
        from chrys.orchestration.engine.executor import Executor

        await Executor.resume(executor, additional_text="with this extra context")

        executor._execute.assert_called_once()
        call_arg = executor._execute.call_args[0][0]
        assert call_arg[0].text == "with this extra context"
        # Orphan user message must be preserved.
        assert len(msgs) == 1
        assert msgs[0].text == "original prompt"

    @pytest.mark.asyncio
    async def test_resume_with_additional_text_preserves_on_failure(self):
        """If the LLM call fails, the note is appended to history as a fallback."""
        from unittest.mock import AsyncMock, MagicMock

        executor = MagicMock()
        session = MagicMock()
        msgs = [
            Message("user", ["original prompt"]),
            Message("assistant", ["tool calls"]),
            Message("tool", ["results"]),
        ]
        session.state = self._make_session_state(msgs)
        executor._session = session

        # Simulate run failure — the framework didn't persist the note.
        async def _fake_execute(_input):
            executor._run_failed = True

        executor._execute = AsyncMock(side_effect=_fake_execute)
        executor._running = False
        executor._was_interrupted = False
        executor._run_failed = False
        executor._last_error = ""
        from chrys.orchestration.engine.executor import Executor

        await Executor.resume(executor, additional_text="my note")

        # Fallback appended the note so a subsequent retry can see it.
        assert any(m.role == "user" and (m.text or "") == "my note" for m in msgs)


class TestResumeMidTurnStamps:
    """Creation-time mid-turn flags and pop-time recovery registration (§2.5)."""

    def _make_executor(self, msgs: list[Message], execute=None):
        from unittest.mock import AsyncMock, MagicMock

        executor = MagicMock()
        session = MagicMock()
        session.state = {"chrys_history": {"messages": msgs}}
        executor._session = session
        executor._execute = AsyncMock(side_effect=execute)
        executor._running = False
        executor._was_interrupted = False
        executor._run_failed = False
        executor._last_error = ""
        executor.recovery_input_recorder = None
        executor._pending_continuation_token = None
        return executor

    @pytest.mark.asyncio
    async def test_empty_input_resume_keeps_pending_continuation_token(self):
        """The bare-continue branch resumes the announced background response."""
        from chrys.orchestration.engine.executor import Executor

        executor = self._make_executor([Message("user", ["prompt"]), Message("assistant", ["work"])])
        token = {"response_id": "bg-pending"}
        executor._pending_continuation_token = token

        await Executor.resume(executor)

        executor._execute.assert_awaited_once_with([])
        assert executor._pending_continuation_token == token

    @pytest.mark.asyncio
    async def test_note_resume_drops_pending_continuation_token(self):
        """A note needs a fresh create — a retrieve would silently ignore it."""
        from chrys.orchestration.engine.executor import Executor

        noted = self._make_executor([Message("user", ["prompt"]), Message("assistant", ["work"])])
        noted._pending_continuation_token = {"response_id": "bg-a"}
        await Executor.resume(noted, additional_text="also check env vars")
        assert noted._pending_continuation_token is None

    @pytest.mark.asyncio
    async def test_pending_token_routes_bare_resume_to_empty_input_without_landed_work(self):
        """A slow background response can announce its token before any output
        lands — bare retry must retrieve it, not replay the opener into a
        duplicate create."""
        from chrys.orchestration.engine.executor import Executor

        executor = self._make_executor([Message("user", ["prompt"])])
        token = {"response_id": "bg-in-flight"}
        executor._pending_continuation_token = token

        await Executor.resume(executor)

        executor._execute.assert_awaited_once_with([])
        assert executor._pending_continuation_token == token
        # The opener stays in history — it was not popped for replay.
        assert [m.text for m in executor._session.state["chrys_history"]["messages"]] == ["prompt"]

    @pytest.mark.asyncio
    async def test_fresh_user_message_drops_pending_continuation_token(self):
        """A new prompt abandons the failed turn's in-flight response."""
        from chrys.orchestration.engine.executor import Executor

        executor = self._make_executor([])
        executor._pending_continuation_token = {"response_id": "bg-stale"}

        await Executor.run(executor, ["new question"])

        assert executor._pending_continuation_token is None
        executor._execute.assert_awaited_once()
        assert executor._execute.call_args[0][0][0].text == "new question"

    @staticmethod
    def _flagged(text: str, key: str) -> Message:
        msg = Message("user", [text])
        msg.additional_properties[key] = True
        return msg

    @pytest.mark.asyncio
    async def test_bare_resume_after_work_uses_empty_input(self):
        from chrys.orchestration.engine.executor import Executor

        executor = self._make_executor(
            [Message("user", ["prompt"]), Message("assistant", ["work"]), Message("tool", ["results"])]
        )
        await Executor.resume(executor)

        executor._execute.assert_awaited_once_with([])

    @pytest.mark.asyncio
    async def test_no_user_message_resume_is_defensive_noop(self):
        from chrys.orchestration.engine.executor import Executor

        executor = self._make_executor([Message("assistant", ["work"])])
        await Executor.resume(executor)

        executor._execute.assert_not_awaited()
        assert executor._run_failed is True
        assert executor._last_error == "Cannot resume without a real user message in history."

    @pytest.mark.asyncio
    async def test_additional_text_is_flagged_injected(self):
        from chrys.orchestration.engine.executor import Executor

        executor = self._make_executor([Message("user", ["prompt"]), Message("assistant", ["work"])])
        await Executor.resume(executor, additional_text="also check env vars")

        sent = executor._execute.call_args[0][0][0]
        assert sent.text == "also check env vars"
        assert sent.additional_properties[HistoryMarkerKind.INJECTED_KEY] is True
        assert HistoryMarkerKind.CONTINUATION_KEY not in sent.additional_properties

    @pytest.mark.asyncio
    async def test_failed_guidance_fallback_appends_flagged_even_when_opener_text_matches(self):
        """Kind-aware dedup: guidance worded identically to the opener still lands, flagged."""
        from chrys.orchestration.engine.executor import Executor

        msgs = [Message("user", ["do the thing"]), Message("assistant", ["work"])]
        executor = self._make_executor(msgs)

        async def _fake_execute(_input):
            executor._run_failed = True

        executor._execute.side_effect = _fake_execute

        await Executor.resume(executor, additional_text="do the thing")

        flagged = [m for m in msgs if m.additional_properties.get(HistoryMarkerKind.INJECTED_KEY)]
        assert [m.text for m in flagged] == ["do the thing"]
        # The opener stays unflagged — never backfilled.
        assert HistoryMarkerKind.INJECTED_KEY not in msgs[0].additional_properties

    @pytest.mark.asyncio
    async def test_opener_replay_unflagged_and_registered_as_opener_kind(self):
        from chrys.orchestration.engine.executor import Executor

        msgs = [Message("user", ["original prompt"])]
        executor = self._make_executor(msgs)
        registered: list[tuple] = []
        executor.recovery_input_recorder = lambda text, contents, created_at, kind="opener": registered.append(
            (text, contents, created_at, kind)
        )

        await Executor.resume(executor)

        sent = executor._execute.call_args[0][0][0]
        assert sent.text == "original prompt"
        assert HistoryMarkerKind.INJECTED_KEY not in sent.additional_properties
        assert HistoryMarkerKind.CONTINUATION_KEY not in sent.additional_properties
        # The pop destroyed the only durable copy — registration happens at pop time.
        assert registered == [("original prompt", None, None, "opener")]

    @pytest.mark.asyncio
    async def test_popped_injection_replay_preserves_flag_and_registers_injected_kind(self):
        from chrys.orchestration.engine.executor import Executor

        msgs = [self._flagged("crashed injection", HistoryMarkerKind.INJECTED_KEY)]
        executor = self._make_executor(msgs)
        registered: list[tuple] = []
        executor.recovery_input_recorder = lambda text, contents, created_at, kind="opener": registered.append(
            (text, contents, created_at, kind)
        )

        await Executor.resume(executor)

        sent = executor._execute.call_args[0][0][0]
        assert sent.text == "crashed injection"
        # Re-sending a popped injection unflagged would launder it into an opener.
        assert sent.additional_properties[HistoryMarkerKind.INJECTED_KEY] is True
        assert registered == [("crashed injection", None, None, "injected")]

    @pytest.mark.asyncio
    async def test_failed_opener_replay_reappend_not_suppressed_by_same_text_injection(self):
        """A same-text flagged injection in the region must not swallow the popped opener."""
        from chrys.orchestration.engine.executor import Executor

        msgs = [Message("user", ["what time is it?"])]
        executor = self._make_executor(msgs)

        async def _fake_execute(_input):
            # The failed replay run persisted only a same-text INJECTION copy.
            msgs.append(self._flagged("what time is it?", HistoryMarkerKind.INJECTED_KEY))
            executor._run_failed = True

        executor._execute.side_effect = _fake_execute

        await Executor.resume(executor)

        openers = [
            m for m in msgs if m.role == "user" and not m.additional_properties.get(HistoryMarkerKind.INJECTED_KEY)
        ]
        assert [m.text for m in openers] == ["what time is it?"]

    @pytest.mark.asyncio
    async def test_failed_injection_replay_reappends_flagged_and_dedups_against_flagged_copy(self):
        """Popped-injection replay follows the anchor's kind in both dedup directions."""
        from chrys.orchestration.engine.executor import Executor

        # Direction 1: persisted flagged copy exists → no duplicate append.
        msgs = [self._flagged("my note", HistoryMarkerKind.INJECTED_KEY)]
        executor = self._make_executor(msgs)

        async def _fail_with_copy(_input):
            msgs.append(self._flagged("my note", HistoryMarkerKind.INJECTED_KEY))
            executor._run_failed = True

        executor._execute.side_effect = _fail_with_copy
        await Executor.resume(executor)
        assert [m.text for m in msgs] == ["my note"]
        assert msgs[0].additional_properties[HistoryMarkerKind.INJECTED_KEY] is True

        # Direction 2: only a same-text unflagged opener exists → re-append FLAGGED.
        msgs2 = [self._flagged("my note", HistoryMarkerKind.INJECTED_KEY)]
        executor2 = self._make_executor(msgs2)

        async def _fail_with_opener(_input):
            msgs2.append(Message("user", ["my note"]))
            executor2._run_failed = True

        executor2._execute.side_effect = _fail_with_opener
        await Executor.resume(executor2)
        flagged = [m for m in msgs2 if m.additional_properties.get(HistoryMarkerKind.INJECTED_KEY)]
        assert [m.text for m in flagged] == ["my note"]
        # The unflagged opener copy is untouched.
        assert sum(1 for m in msgs2 if m.role == "user") == 2

    # -- replay-branch blind spot (§2.5): a checkpoint built AFTER the pop --

    @staticmethod
    def _make_turn_state():
        """Real TurnRuntimeState wired the way the runner leaves it: cleared."""
        from chrys.orchestration.engine.run.turn_state import TurnRuntimeState

        turn_state = TurnRuntimeState()
        turn_state.clear_current_input()
        return turn_state

    @staticmethod
    def _crash_checkpoint(msgs: list[Message], turn_state) -> dict | None:
        """Build a recovery checkpoint from post-pop live state.

        Simulates a hard crash mid-replay-run: nothing was captured and
        nothing persisted — the registered current input is the popped
        anchor's only durable copy.  Mirrors the engine's call
        (``engine.py`` ``_save_recovery_checkpoint``): text/contents/
        created_at/kind all come from the turn state.
        """
        from chrys.kernel import LoopRecorder
        from chrys.service.session.checkpoint import build_recovery_state
        from chrys.service.session.runtime_metadata import SessionRuntimeMetadata

        current = turn_state.current_input
        return build_recovery_state(
            {"messages": msgs, "compressed_msgs": [], "turn_counter": 0},
            LoopRecorder(),
            mutation_tracker=None,
            runtime_meta=SessionRuntimeMetadata(),
            user_text=current.text,
            user_contents=current.contents,
            user_created_at=current.created_at,
            user_kind=current.kind,
        )

    @pytest.mark.asyncio
    async def test_crash_after_opener_pop_recovers_unflagged_opener(self):
        """The pop destroys the opener's only durable copy — recovery re-creates it."""
        from chrys.foundation.models.turns import opens_turn
        from chrys.orchestration.engine.executor import Executor

        msgs = [Message("user", ["original prompt"])]
        executor = self._make_executor(msgs)
        turn_state = self._make_turn_state()
        executor.recovery_input_recorder = turn_state.set_current_input

        # _execute is a no-op: the run hard-crashed before persisting anything.
        await Executor.resume(executor)

        recovered = self._crash_checkpoint(msgs, turn_state)
        assert recovered is not None
        users = [m for m in recovered["messages"] if m.role == "user"]
        assert [m.text for m in users] == ["original prompt"]
        assert opens_turn(users[0])
        assert HistoryMarkerKind.INJECTED_KEY not in users[0].additional_properties

    @pytest.mark.asyncio
    async def test_crash_after_injection_pop_recovers_flagged_injection(self):
        """A popped injection comes back ``_injected`` — never laundered into an opener."""
        from chrys.foundation.models.turns import is_mid_turn_user_message
        from chrys.orchestration.engine.executor import Executor

        msgs = [self._flagged("crashed injection", HistoryMarkerKind.INJECTED_KEY)]
        executor = self._make_executor(msgs)
        turn_state = self._make_turn_state()
        executor.recovery_input_recorder = turn_state.set_current_input

        await Executor.resume(executor)

        recovered = self._crash_checkpoint(msgs, turn_state)
        assert recovered is not None
        users = [m for m in recovered["messages"] if m.role == "user"]
        assert [m.text for m in users] == ["crashed injection"]
        assert users[0].additional_properties[HistoryMarkerKind.INJECTED_KEY] is True
        assert is_mid_turn_user_message(users[0])

        # When the run instead persists the input before the checkpoint (a
        # normal finalization), kind-aware dedup yields no duplicate.
        msgs2 = [self._flagged("crashed injection", HistoryMarkerKind.INJECTED_KEY)]
        executor2 = self._make_executor(msgs2)
        turn_state2 = self._make_turn_state()
        executor2.recovery_input_recorder = turn_state2.set_current_input

        async def _persist_input(_input):
            msgs2.append(self._flagged("crashed injection", HistoryMarkerKind.INJECTED_KEY))

        executor2._execute.side_effect = _persist_input
        await Executor.resume(executor2)

        recovered2 = self._crash_checkpoint(msgs2, turn_state2)
        assert recovered2 is not None
        users2 = [m for m in recovered2["messages"] if m.role == "user"]
        assert [m.text for m in users2] == ["crashed injection"]

    @pytest.mark.asyncio
    async def test_crash_during_empty_input_resume_synthesizes_no_user_message(self):
        """Empty-input resume keeps recovery input clear."""
        from chrys.orchestration.engine.executor import Executor

        # Completed work after the opener resumes with empty input.
        msgs = [Message("user", ["prompt"]), Message("assistant", ["work"])]
        executor = self._make_executor(msgs)
        turn_state = self._make_turn_state()
        executor.recovery_input_recorder = turn_state.set_current_input

        await Executor.resume(executor)

        assert turn_state.current_input.text == ""
        recovered = self._crash_checkpoint(msgs, turn_state)
        assert recovered is not None
        assert [m.text for m in recovered["messages"] if m.role == "user"] == ["prompt"]

        # The admission guard makes this unreachable; Executor remains defensive.
        msgs2 = [Message("assistant", ["work"])]
        executor2 = self._make_executor(msgs2)
        turn_state2 = self._make_turn_state()
        executor2.recovery_input_recorder = turn_state2.set_current_input

        await Executor.resume(executor2)

        executor2._execute.assert_not_awaited()
        assert executor2._run_failed is True
        assert turn_state2.current_input.text == ""
        recovered2 = self._crash_checkpoint(msgs2, turn_state2)
        assert recovered2 is not None
        assert [m for m in recovered2["messages"] if m.role == "user"] == []


class TestRetryLifecycleApprovalContext:
    """retry_and_save restores approval middleware context after pre-run reset."""

    class _Executor:
        def __init__(self) -> None:
            self.user_messages: list[str] = []
            self.resume_texts: list[str] = []
            self.run_failed = False
            self.was_interrupted = False
            self.last_error = None
            self.service_session_id = ""
            self.on_resume = None
            self.history_state: dict[str, object] = {}
            self.trajectory_context = None
            self.opening_item_ids: list[str | None] = []
            self.reset_counter_calls: list[bool] = []

        def set_opening_item_id(self, item_id: str | None) -> None:
            self.opening_item_ids.append(item_id)

        def set_user_message(self, text: str) -> None:
            self.user_messages.append(text)

        def set_user_messages(self, messages: list[str]) -> None:
            self.user_messages.extend(messages)

        def reset_counters(self, *, reset_batch_id: bool) -> None:
            self.reset_counter_calls.append(reset_batch_id)

        async def resume(self, additional_text: str = "", **_kwargs: object) -> None:
            self.resume_texts.append(additional_text)
            if self.on_resume is not None:
                self.on_resume()

        def drain_approval_decisions(self) -> list[object]:
            return []

        def drain_batch_records(self) -> list[object]:
            return []

    class _History:
        def __init__(self, messages: list[Message]) -> None:
            self.messages = messages
            self.tags: list[tuple[str, str]] = []

        def tag_last_user_message(self, key: str, value: str) -> None:
            self.tags.append((key, value))

        def merge_loop_messages(self, _loop_recorder: object, *, insert_index: int | None = None) -> None:
            return None

        def persist_approval_decisions(self, *_args: object, **_kwargs: object) -> None:
            return None

        def trim_to_last_complete_tool_results(self) -> None:
            return None

        def remove_trailing_agent_text(self) -> None:
            return None

        def persist_batch_ids(self, _batch_records: list[object]) -> dict[int, object]:
            return {}

        def persist_intermediate_texts(self, _texts: dict[int, str], _batch_anchors: dict[int, object]) -> None:
            return None

        def persist_consumed_injections(self, _injections: list[object]) -> None:
            return None

        def backfill_missing_created_at(self, *, start_index: int) -> None:
            return None

        def remove_awaiting_sub_agents_marker(self) -> None:
            return None

        def insert_interrupted_marker(self, *_args: object, **_kwargs: object) -> None:
            return None

        def remove_all_status_markers(self) -> None:
            return None

        def insert_turn_marker(self) -> None:
            return None

    class _RuntimeMeta:
        def __init__(self) -> None:
            self.last_usage_details = {"input_token_count": 7}

    class _Reminder:
        def __init__(self) -> None:
            self.prepare_calls: list[dict[str, object]] = []
            self.consumed_switch_to: str | None = None

        def prepare_turn(self, **kwargs: object) -> None:
            self.prepare_calls.append(kwargs)
            self.consumed_switch_to = None

        def take_undelivered_file_change(self) -> str | None:
            return None

    class _Injection:
        def drain_pending(self) -> list[object]:
            return []

    class _Host:
        def __init__(self, messages: list[Message]) -> None:
            self._turn_state = TurnRuntimeState()
            self._bus = EventBus()
            self._session_id = None
            self._executor = TestRetryLifecycleApprovalContext._Executor()
            self._history = TestRetryLifecycleApprovalContext._History(messages)
            self._runtime_meta = TestRetryLifecycleApprovalContext._RuntimeMeta()
            self._reminder_middleware = TestRetryLifecycleApprovalContext._Reminder()
            self._hook_manager = None
            self._outbox_recovery_task = None
            self._agent_profile = None
            self._turn_number = 0
            self._fsm = EngineStateMachine()
            self._consumed_injections: list[object] = []
            self._intermediate_texts: dict[int, str] = {}
            self._loop_recorder = None
            self._mutation_tracker = None
            self._agent_profile_fingerprint = ""
            self._model_profile_fingerprint = ""
            self._persistence = SimpleNamespace(checkpoint_for=lambda _session_id: None, state_store=None)
            self._trajectory_recorder = TrajectoryRecorder()
            self._workspace_change_tracker = WorkspaceChangeTracker()
            self._settings = Settings(workspace_change_notice=False)
            self._mutation_coordinator = None
            self._injection = TestRetryLifecycleApprovalContext._Injection()
            self._shutting_down = False
            self._paused_sub_agents: set[str] = set()
            self._workspace = None
            self.session_generation = 0
            self.build_generation = 0
            self.post_run_calls = 0

        async def _save_current_session(self) -> bool:
            self.post_run_calls += 1
            return True

        def _on_successful_turn(self) -> None:
            return None

        async def _retry_and_save(self, *_args: object, **_kwargs: object) -> None:
            return None

    @pytest.mark.asyncio
    async def test_retry_without_note_uses_latest_user_message(self) -> None:
        host = self._Host(
            [
                Message("user", ["original request"]),
                Message("assistant", ["tool call"]),
                Message("tool", ["result"]),
            ]
        )

        await retry_and_save(host)

        assert host._executor.user_messages == ["original request"]
        assert host._executor.resume_texts == [""]
        assert host._reminder_middleware.prepare_calls == [
            {"usage": {"input_token_count": 7}, "preserve_last_words": True, "preserve_turn_reminders": True}
        ]
        assert host._executor.reset_counter_calls == [False]
        assert host._turn_number == 0
        assert host.post_run_calls == 1

    @pytest.mark.asyncio
    async def test_retry_with_note_uses_note_for_approval_context(self) -> None:
        host = self._Host([Message("user", ["original request"])])

        await retry_and_save(host, additional_text=" also inspect env ")

        assert host._executor.user_messages == ["original request", "also inspect env"]
        assert host._executor.resume_texts == [" also inspect env "]
        assert host._reminder_middleware.prepare_calls == [
            {"usage": {"input_token_count": 7}, "preserve_last_words": True, "preserve_turn_reminders": True}
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("is_retry", "agent_turn", "overlap_turn"),
        [(False, 7, 7), (True, None, 8)],
    )
    async def test_workspace_notice_identity_uses_live_turn_number(
        self,
        is_retry: bool,
        agent_turn: int | None,
        overlap_turn: int,
    ) -> None:
        host = self._Host([Message("user", ["request"])])
        host._turn_number = 8
        host._executor.history_state["turn_counter"] = 999
        host._settings = Settings(workspace_change_notice=True)
        host._recovered_from_sidecar = False
        calls: list[dict[str, object]] = []

        def _compute(**kwargs: object) -> None:
            calls.append(kwargs)

        host._workspace_change_tracker.compute_turn_notice = _compute  # type: ignore[method-assign]

        await TurnRunner(host)._compute_workspace_notice(is_retry=is_retry)

        assert len(calls) == 1
        assert calls[0]["turn_id"] == 8
        assert calls[0]["agent_turn_id"] == agent_turn
        assert calls[0]["overlap_turn_id"] == overlap_turn

    @pytest.mark.asyncio
    async def test_workspace_notice_timeout_leaves_persistent_caveat(self) -> None:
        """The timeout branch must queue the degraded warning — bare
        cancellation would let finalization's fresh baseline absorb the
        lost comparison silently."""
        host = self._Host([Message("user", ["request"])])
        host._turn_number = 3
        host._settings = Settings(workspace_change_notice=True)
        host._recovered_from_sidecar = False

        def _compute(**_kwargs: object) -> None:
            raise TimeoutError

        host._workspace_change_tracker.compute_turn_notice = _compute  # type: ignore[method-assign]

        await TurnRunner(host)._compute_workspace_notice(is_retry=False)

        notice = host._workspace_change_tracker.take_pending_notice()
        assert notice is not None
        assert "Workspace change detection could not be completed" in notice

    @pytest.mark.asyncio
    async def test_retry_computes_notice_before_prepare_turn(self) -> None:
        host = self._Host([Message("user", ["request"])])
        host._turn_number = 3
        host._settings = Settings(workspace_change_notice=True)
        host._recovered_from_sidecar = False
        order: list[str] = []

        def _compute(**_kwargs: object) -> None:
            order.append("compute")

        original_prepare = host._reminder_middleware.prepare_turn

        def _prepare(**kwargs: object) -> None:
            order.append("prepare")
            original_prepare(**kwargs)

        host._workspace_change_tracker.compute_turn_notice = _compute  # type: ignore[method-assign]
        host._reminder_middleware.prepare_turn = _prepare  # type: ignore[method-assign]

        await retry_and_save(host)

        assert order[:2] == ["compute", "prepare"]

    @pytest.mark.asyncio
    async def test_retry_pre_executor_interrupt_requeues_drained_notice(self) -> None:
        host = self._Host([Message("user", ["request"])])
        tracker = WorkspaceChangeTracker()
        tracker.queue_safety_notice("safety")
        host._workspace_change_tracker = tracker
        host._settings = Settings(workspace_change_notice=True)
        host._recovered_from_sidecar = False
        host._reminder_middleware = SystemReminderMiddleware(file_change_provider=tracker.take_pending_notice)

        def _record_pre_run_interrupt() -> None:
            host._executor.was_interrupted = True

        host._executor.record_pre_run_interrupt = _record_pre_run_interrupt  # type: ignore[attr-defined]
        current = asyncio.current_task()
        assert current is not None
        assert host._turn_state.request_pre_executor_interrupt(current) is True

        await retry_and_save(host)

        tracker.requeue_notice("new boundary")
        assert tracker.take_pending_notice() == "safety\n\nnew boundary"
        assert tracker.take_pending_notice() is None

    @pytest.mark.asyncio
    async def test_approval_context_skips_nudges_keeps_injections(self) -> None:
        """§2.2: the current-turn approval context excludes synthetic
        ``continue`` nudges (orchestration placeholders, not user input) while
        keeping user-authored injections."""
        marker = Message("assistant", [""])
        marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
        injected = Message("user", ["also check the docs"])
        injected.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
        nudge = Message("user", ["continue"])
        nudge.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
        host = self._Host(
            [
                Message("user", ["old request"]),
                Message("assistant", ["done"]),
                marker,
                Message("user", ["current request"]),
                Message("assistant", ["working"]),
                injected,
                nudge,
            ]
        )

        await retry_and_save(host)

        assert host._executor.user_messages == ["current request", "also check the docs"]

    @pytest.mark.asyncio
    async def test_synthetic_only_region_falls_back_to_previous_real_opener(self) -> None:
        """§2.2: when the current region holds only a flagged nudge, the
        fallback hands the judge the previous REAL opener, never "continue"."""
        marker = Message("assistant", [""])
        marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
        nudge = Message("user", ["continue"])
        nudge.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
        host = self._Host(
            [
                Message("user", ["real question"]),
                Message("assistant", ["answer"]),
                marker,
                nudge,
                Message("assistant", ["resumed work"]),
            ]
        )

        await retry_and_save(host)

        assert host._executor.user_messages == ["real question"]

    @pytest.mark.asyncio
    async def test_all_synthetic_history_yields_empty_approval_context(self) -> None:
        """§2.2: a history holding only flagged nudges (no real user input)
        yields an empty approval context — no fabricated "continue"."""
        nudge = Message("user", ["continue"])
        nudge.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
        host = self._Host(
            [
                Message("assistant", ["crash-leftover work"]),
                nudge,
            ]
        )

        await retry_and_save(host)

        assert host._executor.user_messages == []

    @pytest.mark.asyncio
    async def test_retry_tags_consumed_profile_switch(self) -> None:
        host = self._Host([Message("user", ["after switch"])])

        def _consume_switch() -> None:
            host._reminder_middleware.consumed_switch_to = "Explore Agent"

        host._executor.on_resume = _consume_switch

        await retry_and_save(host)

        assert host._history.tags == [(HistoryMarkerKind.PROFILE_SWITCH_TO_KEY, "Explore Agent")]


@pytest.mark.asyncio
async def test_pending_retry_dispatch_strips_trailing_markers_before_task() -> None:
    """Interrupt→resume marker hygiene (§7): the pending-retry dispatch path
    (``RetryCoordinator.start_pending_retry_if_due``) strips trailing markers
    BEFORE creating the retry task, so the resumed task's history carries no
    interior turn marker.  The immediate-dispatch paths are covered end-to-end
    by ``test_retry_turn_number.py`` (exactly one ``_turn=1`` marker after
    interrupt→retry)."""
    from chrys.orchestration.engine.run.retry import RetryCoordinator
    from chrys.orchestration.engine.run.turn_state import PendingRetry

    order: list[str] = []

    class _History:
        def remove_trailing_markers(self) -> None:
            order.append("remove_trailing_markers")

    class _Host:
        def __init__(self) -> None:
            self._turn_state = TurnRuntimeState()
            self._history = _History()
            self._fsm = EngineStateMachine()
            self._fsm.try_transition(Trigger.START)
            self._fsm.try_transition(Trigger.USER_MESSAGE)  # RUNNING
            self._reminder_middleware = None
            self._shutting_down = False
            self.session_generation = 0
            self.build_generation = 0

        async def _retry_and_save(self, *_args: object, **_kwargs: object) -> None:
            order.append("retry_task")

    host = _Host()
    host._turn_state.pending_retry = PendingRetry(owner_admission_id=1)

    RetryCoordinator(host).start_pending_retry_if_due()

    task = host._turn_state.run_task
    assert task is not None
    await task
    assert order == ["remove_trailing_markers", "retry_task"]


@pytest.mark.asyncio
async def test_later_retry_after_interrupted_finalization_preserves_last_words_and_turn_reminders() -> None:
    class _Executor:
        def __init__(self) -> None:
            self.run_failed = False
            self.was_interrupted = True
            self.last_error = None
            self.service_session_id = "service-session"
            self.is_running = False
            self.history_state: dict[str, object] = {}
            self.trajectory_context = None
            self.opening_item_ids: list[str | None] = []
            self.user_messages: list[str] = []
            self.resume_texts: list[str] = []
            self.observed_last_words: str | None = None
            self.observed_turn_reminders: list[str] = []
            self.observed_last_words_reminders: list[str] = []

        def set_opening_item_id(self, item_id: str | None) -> None:
            self.opening_item_ids.append(item_id)

        def reset_counters(self, *, reset_batch_id: bool) -> None:
            assert reset_batch_id is False

        def drain_approval_decisions(self) -> list[object]:
            return []

        def drain_batch_records(self) -> list[object]:
            return []

        def set_user_message(self, text: str) -> None:
            self.user_messages.append(text)

        def set_user_messages(self, messages: list[str]) -> None:
            self.user_messages.extend(messages)

        async def resume(self, additional_text: str = "", **_kwargs: object) -> None:
            self.resume_texts.append(additional_text)
            self.observed_last_words = host._reminder_middleware.get_last_words()
            self.observed_turn_reminders = host._reminder_middleware._build_reminders()
            self.observed_last_words_reminders = host._reminder_middleware._build_last_words_reminders()

    class _History:
        def __init__(self) -> None:
            self.messages = [Message("user", ["original request"])]

        def merge_loop_messages(self, _loop_recorder: object, *, insert_index: int | None = None) -> None:
            return None

        def persist_approval_decisions(self, *_args: object, **_kwargs: object) -> None:
            return None

        def trim_to_last_complete_tool_results(self) -> None:
            return None

        def remove_trailing_agent_text(self) -> None:
            return None

        def persist_batch_ids(self, _batch_records: list[object]) -> dict[int, object]:
            return {}

        def persist_intermediate_texts(self, _texts: dict[int, str], _batch_anchors: dict[int, object]) -> None:
            return None

        def persist_consumed_injections(self, _injections: list[object]) -> None:
            return None

        def backfill_missing_created_at(self, *, start_index: int) -> None:
            return None

        def remove_awaiting_sub_agents_marker(self) -> None:
            return None

        def insert_interrupted_marker(self, *_args: object, **_kwargs: object) -> None:
            return None

        def remove_all_status_markers(self) -> None:
            return None

        def insert_turn_marker(self) -> None:
            return None

        def remove_trailing_markers(self) -> None:
            return None

        def tag_last_user_message(self, _key: str, _value: str) -> None:
            return None

    class _Injection:
        def drain_pending(self) -> list[object]:
            return []

    class _RuntimeMeta:
        def __init__(self) -> None:
            self.last_usage_details = {"input_token_count": 7}

    class _Host:
        def __init__(self) -> None:
            self._agent_loading = False
            self._bus = EventBus()
            self._session_id = "s1"
            self._turn_state = TurnRuntimeState()
            self._reminder_middleware = SystemReminderMiddleware()
            self._executor = _Executor()
            self._history = _History()
            self._runtime_meta = _RuntimeMeta()
            self._fsm = EngineStateMachine()
            self._fsm.transition(Trigger.START)
            self._fsm.transition(Trigger.USER_MESSAGE)
            self._consumed_injections: list[object] = []
            self._intermediate_texts: dict[int, str] = {}
            self._turn_number = 1
            self._loop_recorder = None
            self._mutation_tracker = None
            self._agent_profile_fingerprint = ""
            self._model_profile_fingerprint = ""
            self._persistence = SimpleNamespace(checkpoint_for=lambda _session_id: None, state_store=None)
            self._trajectory_recorder = TrajectoryRecorder()
            self._workspace_change_tracker = WorkspaceChangeTracker()
            self._settings = Settings(workspace_change_notice=False)
            self._mutation_coordinator = None
            self._injection = _Injection()
            self._shutting_down = False
            self._paused_sub_agents: set[str] = set()
            self._hook_manager = None
            self._agent_profile = None
            self._workspace = None
            self._runtime_details = AgentRuntimeDetails(model=RuntimeModelDetails(vision=True))
            self._skills_provider = None
            self.session_generation = 0
            self.build_generation = 0

        async def _wait_for_agent_load_idle(self) -> None:
            return None

        async def _save_current_session(self) -> bool:
            return True

        def _on_successful_turn(self) -> None:
            return None

        async def _retry_and_save(
            self,
            additional_text: str = "",
            created_at: object = None,
            **kwargs: object,
        ) -> None:
            await retry_and_save(self, additional_text, created_at, **kwargs)

    host = _Host()
    reminder_scope = host._reminder_middleware.create_current_run_scope()
    run_scope = host._turn_state.begin_current_run_scope(
        owner_admission_id=1,
        session_generation=host.session_generation,
        build_generation=host.build_generation,
        reminder_scope=reminder_scope,
    )
    host._turn_state.open_injection_admission(run_scope)
    host._reminder_middleware.prepare_turn(reminder_scope=reminder_scope)
    host._reminder_middleware.queue_hook_reminders(["stable turn reminder"])
    host._reminder_middleware.set_last_words("[progress before interrupt]")

    finalization_task = asyncio.create_task(post_run(host))
    host._turn_state.run_task = finalization_task
    await finalization_task

    assert host._turn_state.current_run_scope == run_scope
    assert host._reminder_middleware.get_last_words() == "[progress before interrupt]"

    await on_user_retry(host, UserRetry(text="retry note"))
    assert host._turn_state.run_task is not None
    await host._turn_state.run_task

    assert host._executor.resume_texts == ["retry note"]
    assert host._executor.observed_last_words == "[progress before interrupt]"
    assert "stable turn reminder" in host._executor.observed_turn_reminders
    assert any("[progress before interrupt]" in reminder for reminder in host._executor.observed_last_words_reminders)
    assert host._turn_state.current_input.text == ""
    assert host._turn_state.current_input.contents is None
    assert host._turn_state.current_input.created_at is None


# ---------------------------------------------------------------------------
# Integration tests: retry after interrupt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_after_interrupt_preserves_tools(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """Interrupt mid-run, then retry → completed tools preserved, no duplicate user message.

    Flow:
    1. User sends message, agent streams slowly
    2. User interrupts
    3. User retries → agent continues from completed work
    4. Verify: no duplicate user messages, completed work preserved
    """
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, AgentMessage, Error, UsageUpdate, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    settings, model_registry = make_mock_settings_and_registry(stream=True)
    state_store = JsonFileStateStore(tmp_path)
    registry = _make_registry()

    mock_clients: list[MockChatClient] = []

    def _mock_create_client(s=None, **kw):
        client = MockChatClient(
            responses=[
                # Run 1: slow stream, will be interrupted
                MockResponse(text="A" * 20, chunk_size=2, chunk_delay=0.02),
                # Run 2 (retry): fast response
                MockResponse(text="Continued successfully."),
            ]
        )
        mock_clients.append(client)
        return client

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=registry,
        model_registry=model_registry,
        state_store=state_store,
    )
    await engine.start(_PROFILE)

    # Send message and interrupt mid-stream (poll until the run reaches the
    # LLM so the interrupt lands during the stream, not before it starts).
    await bus.publish(UserMessage(text="Do something"))
    await _wait_for_call_count(mock_clients[0], 1)
    await bus.publish(UserInterrupt())
    await await_run_task_chain(engine)

    # Verify interrupted state
    msgs = _history_messages(engine)
    user_msgs = [m for m in msgs if getattr(m, "role", "") == "user"]
    assert len(user_msgs) >= 1, "User message should be preserved after interrupt"

    # Now retry
    events.clear()
    await bus.publish(UserRetry())

    # After retry, should have a final response
    finals = await _final_agent_messages_after_run(engine, events)
    assert len(finals) == 1, f"Expected 1 final message after retry, got {len(finals)}"

    # Verify no "continue" user message in final state
    msgs = _history_messages(engine)
    continue_msgs = [m for m in msgs if getattr(m, "role", "") == "user" and (m.text or "").strip() == "continue"]
    assert len(continue_msgs) == 0, "Continuation 'continue' message should be cleaned up"

    # Only one user message (the original)
    user_msgs = [m for m in msgs if getattr(m, "role", "") == "user"]
    original_msgs = [m for m in user_msgs if (m.text or "").strip() == "Do something"]
    assert len(original_msgs) == 1, f"Expected exactly 1 original user message, got {len(original_msgs)}"

    # No stale interrupted markers after successful retry
    interrupted = [
        m
        for m in msgs
        if (getattr(m, "additional_properties", None) or {}).get(HistoryMarkerKind.KEY) == HistoryMarkerKind.INTERRUPTED
    ]
    assert len(interrupted) == 0, "Interrupted markers should be cleaned up after successful retry"


@pytest.mark.asyncio
async def test_retry_with_note_after_interrupt_is_mid_turn(
    tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch
):
    """User types a note and clicks Continue after an interrupt.

    The note becomes a real user message in history (not stripped like
    ``"continue"``), no ``"continue"`` placeholder appears in history,
    and the turn_counter is unchanged — the note is mid-turn.
    """
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, AgentMessage, Error, UsageUpdate, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    settings, model_registry = make_mock_settings_and_registry(stream=True)
    state_store = JsonFileStateStore(tmp_path)
    registry = _make_registry()

    mock_clients: list[MockChatClient] = []

    def _mock_create_client(s=None, **kw):
        client = MockChatClient(
            responses=[
                # Run 1: slow stream, will be interrupted
                MockResponse(text="A" * 20, chunk_size=2, chunk_delay=0.02),
                # Run 2 (retry with note): fast response
                MockResponse(text="Picked up your note."),
            ]
        )
        mock_clients.append(client)
        return client

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=registry,
        model_registry=model_registry,
        state_store=state_store,
    )
    await engine.start(_PROFILE)

    await bus.publish(UserMessage(text="Do something"))
    await _wait_for_call_count(mock_clients[0], 1)
    await bus.publish(UserInterrupt())
    await await_run_task_chain(engine)

    turn_counter_before = engine._executor.history_state.get("turn_counter", 0)

    events.clear()
    await bus.publish(UserRetry(text="also check env vars"))

    finals = await _final_agent_messages_after_run(engine, events)
    assert len(finals) == 1, f"Expected 1 final after retry-with-note, got {len(finals)}"

    msgs = _history_messages(engine)

    # No bare "continue" placeholder was persisted.
    continue_msgs = [m for m in msgs if getattr(m, "role", "") == "user" and (m.text or "").strip() == "continue"]
    assert len(continue_msgs) == 0, "Mid-turn note must not leave a 'continue' placeholder behind"

    # The user's note is a real user message.
    note_msgs = [
        m for m in msgs if getattr(m, "role", "") == "user" and (m.text or "").strip() == "also check env vars"
    ]
    assert len(note_msgs) == 1, f"User note missing from history: {[m.text for m in msgs if m.role == 'user']}"

    # Both the original prompt and the note survive — the note does
    # not replace the original.
    original_msgs = [m for m in msgs if getattr(m, "role", "") == "user" and (m.text or "").strip() == "Do something"]
    assert len(original_msgs) == 1

    # Mid-turn semantics: turn_counter unchanged across the retry.
    turn_counter_after = engine._executor.history_state.get("turn_counter", 0)
    assert turn_counter_after == turn_counter_before, (
        f"Retry-with-note must be mid-turn: turn_counter {turn_counter_before} → {turn_counter_after}"
    )

    # No stale interrupted markers.
    interrupted = [
        m
        for m in msgs
        if (getattr(m, "additional_properties", None) or {}).get(HistoryMarkerKind.KEY) == HistoryMarkerKind.INTERRUPTED
    ]
    assert len(interrupted) == 0


@pytest.mark.parametrize(
    ("marker_source", "expected_state"),
    [("user", EngineState.INTERRUPTED), ("error", EngineState.FAILED)],
)
@pytest.mark.asyncio
async def test_retry_after_restored_interrupted_session_starts_model_call(
    tmp_path: Path,
    marker_source: str,
    expected_state: EngineState,
    agent_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restored terminal history should rebuild FSM state so Continue starts a retry."""
    state_store = JsonFileStateStore(tmp_path)
    session_id = f"restore-interrupted-session-{marker_source}"
    user = Message("user", ["What tools do you have?"])
    interrupted = Message("assistant", ["Execution interrupted"])
    interrupted.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.INTERRUPTED
    interrupted.additional_properties["_interrupted_by"] = marker_source
    turn = Message("assistant", [""])
    turn.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    turn.additional_properties["_turn"] = 1
    await state_store.save_session(
        session_id,
        {"messages": [user, interrupted, turn], "turn_counter": 1},
        agent_profile="Code",
        agent_display_name="Code Agent",
        primary_cwd=str(tmp_path),
        agent_profile_history=["Code"],
    )

    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, SessionRestored, AgentMessage, Error, UsageUpdate, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    settings, model_registry = make_mock_settings_and_registry(stream=True)
    mock_clients: list[MockChatClient] = []

    def _mock_create_client(s=None, **kw):
        client = MockChatClient(responses=[MockResponse(text="Restored retry completed.")])
        mock_clients.append(client)
        return client

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=_make_registry(),
        model_registry=model_registry,
        state_store=state_store,
    )
    await engine.start(_PROFILE)

    await bus.publish(SessionRestore(session_id=session_id))
    assert await wait_until(lambda: len(_filter(events, SessionRestored)) >= 1), (
        "session restore did not complete in time"
    )
    assert engine.state is expected_state

    events.clear()
    await bus.publish(UserRetry())

    finals = await _final_agent_messages_after_run(engine, events)
    assert len(finals) == 1
    assert mock_clients[-1].call_count == 1

    msgs = _history_messages(engine)
    interrupted_markers = [
        m for m in msgs if m.additional_properties.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.INTERRUPTED
    ]
    assert interrupted_markers == []
    user_msgs = [m for m in msgs if m.role == "user"]
    assert [(m.text or "").strip() for m in user_msgs] == ["What tools do you have?"]
    assert engine._executor.history_state.get("turn_counter", 0) == 1


@pytest.mark.parametrize("has_trailing_turn_marker", [False, True])
@pytest.mark.asyncio
async def test_retry_after_restored_awaiting_sub_agents_marker_starts_model_call(
    tmp_path: Path,
    has_trailing_turn_marker: bool,
    agent_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restored awaiting-sub-agents markers should become retryable terminal history."""
    state_store = JsonFileStateStore(tmp_path)
    session_id = f"restore-awaiting-sub-agents-session-{has_trailing_turn_marker}"
    user = Message("user", ["Run the child agent"])
    awaiting = Message("assistant", ["Awaiting 1 sub-agent(s)"])
    awaiting.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.AWAITING_SUB_AGENTS
    awaiting.additional_properties["_invocation_ids"] = ["inv-1"]
    messages = [user, awaiting]
    turn_counter = 0
    if has_trailing_turn_marker:
        turn = Message("assistant", [""])
        turn.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
        turn.additional_properties["_turn"] = 1
        messages.append(turn)
        turn_counter = 1
    await state_store.save_session(
        session_id,
        {"messages": messages, "turn_counter": turn_counter},
        agent_profile="Code",
        agent_display_name="Code Agent",
        primary_cwd=str(tmp_path),
        agent_profile_history=["Code"],
    )

    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, SessionRestored, AgentMessage, Error, UsageUpdate, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    settings, model_registry = make_mock_settings_and_registry(stream=True)
    mock_clients: list[MockChatClient] = []

    def _mock_create_client(s=None, **kw):
        client = MockChatClient(responses=[MockResponse(text="Restored awaiting retry completed.")])
        mock_clients.append(client)
        return client

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=_make_registry(),
        model_registry=model_registry,
        state_store=state_store,
    )
    await engine.start(_PROFILE)

    await bus.publish(SessionRestore(session_id=session_id))
    assert await wait_until(lambda: len(_filter(events, SessionRestored)) >= 1), (
        "session restore did not complete in time"
    )
    assert engine.state is EngineState.FAILED
    repaired_raw = await state_store.load_session_raw(session_id)
    assert repaired_raw is not None
    repaired_kinds = [message.get("additional_properties", {}).get(HistoryMarkerKind.KEY) for message in repaired_raw]
    assert HistoryMarkerKind.AWAITING_SUB_AGENTS not in repaired_kinds
    assert HistoryMarkerKind.INTERRUPTED in repaired_kinds

    events.clear()
    await bus.publish(UserRetry())

    finals = await _final_agent_messages_after_run(engine, events)
    assert len(finals) == 1
    assert mock_clients[-1].call_count == 1

    msgs = _history_messages(engine)
    status_markers = [
        m for m in msgs if m.additional_properties.get(HistoryMarkerKind.KEY) in HistoryMarkerKind.STATUS_MARKERS
    ]
    assert status_markers == []
    user_msgs = [m for m in msgs if m.role == "user"]
    assert [(m.text or "").strip() for m in user_msgs] == ["Run the child agent"]
    assert engine._executor.history_state.get("turn_counter", 0) == 1


@pytest.mark.asyncio
async def test_retry_after_error_no_progress(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """Error before any tools execute → retry re-sends original message.

    Flow:
    1. User sends message, LLM call fails immediately
    2. User retries → same message re-sent
    3. Second attempt succeeds
    """
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, AgentMessage, Error, UsageUpdate, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    settings = Settings()
    state_store = JsonFileStateStore(tmp_path)
    registry = _make_registry()

    call_count = 0

    class FailThenSucceedClient(MockChatClient):
        def _inner_get_response(self, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated non-retryable error")
            return super()._inner_get_response(**kwargs)

    def _mock_create_client(s=None, **kw):
        return FailThenSucceedClient(
            responses=[
                MockResponse(text="Success after retry"),
                MockResponse(text="Success after retry"),  # extra for safety
            ]
        )

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    engine = agent_engine(bus, settings=settings, agent_registry=registry, state_store=state_store)
    await engine.start(_PROFILE)

    # Send message — will fail
    await bus.publish(UserMessage(text="Do work"))
    await wait_for(
        lambda: len(_filter(events, Error)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="resume failure error event",
    )

    errors = _filter(events, Error)
    assert len(errors) >= 1, "Should have received an error"

    # Retry
    events.clear()
    await bus.publish(UserRetry())

    # Should succeed this time
    finals = await _final_agent_messages_after_run(engine, events)
    assert len(finals) == 1

    # Only one user message in state
    msgs = _history_messages(engine)
    user_msgs = [m for m in msgs if getattr(m, "role", "") == "user"]
    assert len(user_msgs) == 1
    assert (user_msgs[0].text or "").strip() == "Do work"


@pytest.mark.asyncio
async def test_carried_compression_records_post_fold_metadata_floor() -> None:
    """A pre-retry fold replaces the stale pre_run history boundary."""
    state: dict = {
        "messages": [Message("user", ["first"]), Message("assistant", ["done"])],
        "compressed_msgs": [],
        "turn_counter": 0,
    }
    marker_id = CompressibleHistoryProvider.insert_marker(state, 1)
    pre_fold_len = len(state["messages"])
    strategy = UnifiedContextStrategy(compaction_enabled=False)
    strategy.bind_state(state)
    strategy.queue_compression(marker_id, "durable summary")
    executor = cast("Executor", SimpleNamespace(_compaction_strategy=strategy, history_state=state))

    await Executor._flush_carried_compressions(executor)

    assert len(state["messages"]) < pre_fold_len
    assert state[PRE_OUTPUT_HISTORY_LEN_STATE_KEY] == len(state["messages"])


@pytest.mark.asyncio
async def test_retry_persists_carried_compression_before_transient_attempt_rollback(
    tmp_path: Path,
    agent_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fold queued by the failed pass must be durable across retry rollback.

    This models the live failure sequence precisely: a terminal error leaves
    a validated ``compress_context`` request pending; the user retries; the
    first stored-mode provider attempt applies the fold and then fails
    transiently; the automatic retry succeeds.  The live fold event and the
    saved session must describe the same compression.
    """
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, AgentMessage, ContextCompressed, Error, RetryAttempt, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    settings, model_registry = make_mock_settings_and_registry(stream=False)
    state_store = JsonFileStateStore(tmp_path)
    registry = _make_registry()
    raw_call_count = 0

    class RetryRollbackClient(MockChatClient):
        def _inner_get_response(self, **kwargs):
            nonlocal raw_call_count
            raw_call_count += 1
            if raw_call_count == 3:
                raise RuntimeError("terminal failure before retry")
            if raw_call_count == 4:
                raise ConnectionError("transient retry failure")
            return super()._inner_get_response(**kwargs)

    client = RetryRollbackClient(
        responses=[
            MockResponse(text="turn one"),
            MockResponse(text="turn two"),
            MockResponse(text="retry recovered"),
        ]
    )
    monkeypatch.setattr(builder_module, "create_client", lambda s=None, **kw: client)
    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=registry,
        model_registry=model_registry,
        state_store=state_store,
    )
    await engine.start(_PROFILE)
    assert engine._executor is not None
    engine._executor._chat_options = {"store": True}
    engine._executor._BACKOFF_SCHEDULE = (0,)
    engine._executor._max_retries_override = 1

    await bus.publish(UserMessage(text="first"))
    await await_run_task_chain(engine)
    await bus.publish(UserMessage(text="second"))
    await await_run_task_chain(engine)
    await bus.publish(UserMessage(text="will fail"))
    await await_run_task_chain(engine)
    assert _filter(events, Error)

    strategy = engine._executor.compaction_strategy
    assert strategy is not None
    context_id, _ = strategy.queue_compression("turn_1", "durable retry summary")

    events.clear()
    await bus.publish(UserRetry())
    await await_run_task_chain(engine)

    compressed_events = _filter(events, ContextCompressed)
    assert [event.compressed_context_id for event in compressed_events] == [context_id]
    assert len(_filter(events, RetryAttempt)) == 1

    session_id = engine._session_id
    assert session_id is not None
    loaded = await state_store.load_session(session_id)
    assert loaded is not None
    assert [block.compressed_context_id for block in loaded["compressed_msgs"]] == [context_id]
    assert any(message.additional_properties.get("_block_id") == context_id for message in loaded["messages"])


@pytest.mark.asyncio
async def test_stored_retry_preserves_published_auto_phase3_fold(
    tmp_path: Path,
    agent_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request-time auto fold is replayed silently after whole-run rollback."""
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, AgentMessage, ContextCompressed, Error, RetryAttempt, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    profile = AgentProfile(
        name="Code",
        display_name="Code Agent",
        instructions="You are a coding assistant.",
        tools=ToolsConfig(builtins=[]),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=True),
    )
    from chrys.service.profiles.agents.registry import AgentProfileRegistry

    registry = AgentProfileRegistry()
    registry.register(profile)
    settings, model_registry = make_mock_settings_and_registry(stream=False)
    state_store = JsonFileStateStore(tmp_path)
    raw_call_count = 0

    class AutoFoldRetryClient(MockChatClient):
        def _inner_get_response(self, **kwargs):
            nonlocal raw_call_count
            raw_call_count += 1
            if raw_call_count == 2:
                raise ConnectionError("transient failure after auto fold")
            return super()._inner_get_response(**kwargs)

    client = AutoFoldRetryClient(
        responses=[
            MockResponse(text="completed first turn"),
            MockResponse(text="recovered second turn"),
        ]
    )
    monkeypatch.setattr(builder_module, "create_client", lambda s=None, **kw: client)
    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=registry,
        model_registry=model_registry,
        state_store=state_store,
    )
    await engine.start(profile)
    assert engine._executor is not None
    engine._executor._chat_options = {"store": True}
    engine._executor._BACKOFF_SCHEDULE = (0,)
    engine._executor._max_retries_override = 1

    await bus.publish(UserMessage(text="first turn"))
    await await_run_task_chain(engine)

    strategy = engine._executor.compaction_strategy
    assert strategy is not None
    # The first turn completed under normal thresholds. Force Phase 3 only
    # for the next request's first wire build, where the transient lands.
    strategy.trigger_pct = 0.00001
    strategy.target_pct = 0.000005
    events.clear()

    second_input = "second turn"
    await bus.publish(UserMessage(text=second_input))
    await await_run_task_chain(engine)

    compressed_events = _filter(events, ContextCompressed)
    assert len(compressed_events) == 1
    assert compressed_events[0].source == "auto"
    assert len(_filter(events, RetryAttempt)) == 1
    context_id = compressed_events[0].compressed_context_id

    session_id = engine._session_id
    assert session_id is not None
    loaded = await state_store.load_session(session_id)
    assert loaded is not None
    assert [block.compressed_context_id for block in loaded["compressed_msgs"]] == [context_id]
    assert any(message.additional_properties.get("_block_id") == context_id for message in loaded["messages"])
    assert [message.text for message in loaded["messages"] if message.role == "user"].count(second_input) == 1


class TestResumeMultipleNotes:
    """Calling resume(additional_text=...) repeatedly must not corrupt state.

    Each call should append a fresh user-note message and never re-send
    a stale note from a prior retry.  Exercises the unit surface that
    the multi-interrupt integration scenario exercises end-to-end.
    """

    def _make_session_state(self, messages: list[Message]) -> dict:
        return {"chrys_history": {"messages": messages}}

    @pytest.mark.asyncio
    async def test_two_sequential_retries_with_notes(self):
        from unittest.mock import AsyncMock, MagicMock

        executor = MagicMock()
        session = MagicMock()
        msgs = [
            Message("user", ["original"]),
            Message("assistant", ["tool calls"]),
            Message("tool", ["results"]),
        ]
        session.state = self._make_session_state(msgs)
        executor._session = session
        executor._running = False
        executor._was_interrupted = False
        executor._run_failed = False
        executor._last_error = ""
        sent_inputs: list[list[Message]] = []

        async def _fake_execute(input_msgs):
            sent_inputs.append(input_msgs)
            # Simulate the framework persisting the user input +
            # producing a response before the next retry starts.
            msgs.extend(input_msgs)
            msgs.append(Message("assistant", ["response"]))

        executor._execute = AsyncMock(side_effect=_fake_execute)

        from chrys.orchestration.engine.executor import Executor

        await Executor.resume(executor, additional_text="note one")
        await Executor.resume(executor, additional_text="note two")

        # Each call must have sent its own note — not the other one.
        assert [i[0].text for i in sent_inputs] == ["note one", "note two"]
        # No "continue" placeholder was ever sent.
        assert all(i[0].text != "continue" for i in sent_inputs)
        # Both notes live in state.
        texts = [m.text for m in msgs if m.role == "user"]
        assert texts == ["original", "note one", "note two"]


@pytest.mark.asyncio
async def test_multiple_interrupt_retry_notes_stay_in_single_turn(
    tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch
):
    """User interrupts twice within the same turn, sending a note each time.

    End state must contain the original prompt + both notes as distinct
    user messages inside a single turn (``turn_counter == 1``), with no
    ``"continue"`` placeholder ever persisted.
    """
    events: list[object] = []
    bus = EventBus()
    for cls in [SessionReady, AgentMessage, Error, UsageUpdate, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    settings, model_registry = make_mock_settings_and_registry(stream=True)
    state_store = JsonFileStateStore(tmp_path)
    registry = _make_registry()

    mock_clients: list[MockChatClient] = []

    def _mock_create_client(s=None, **kw):
        client = MockChatClient(
            responses=[
                # Run 1 (original): slow stream, gets interrupted.
                MockResponse(text="A" * 40, chunk_size=2, chunk_delay=0.03),
                # Run 2 (retry with note 1): also slow, also interrupted.
                MockResponse(text="B" * 40, chunk_size=2, chunk_delay=0.03),
                # Run 3 (retry with note 2): fast, completes.
                MockResponse(text="Finished after both notes."),
            ]
        )
        mock_clients.append(client)
        return client

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    engine = agent_engine(
        bus,
        settings=settings,
        agent_registry=registry,
        model_registry=model_registry,
        state_store=state_store,
    )
    await engine.start(_PROFILE)

    # Run 1: original prompt, interrupt mid-stream.
    await bus.publish(UserMessage(text="original prompt"))
    await _wait_for_call_count(mock_clients[0], 1)
    await bus.publish(UserInterrupt())
    await await_run_task_chain(engine)

    # Run 2: retry with note 1, interrupt mid-stream again.
    await bus.publish(UserRetry(text="note one"))
    await _wait_for_call_count(mock_clients[0], 2)
    await bus.publish(UserInterrupt())
    await await_run_task_chain(engine)

    # Run 3: retry with note 2, let it complete.
    events.clear()
    await bus.publish(UserRetry(text="note two"))

    finals = await _final_agent_messages_after_run(engine, events)
    assert len(finals) == 1, f"Expected 1 final after the second note-retry, got {len(finals)}"

    msgs = _history_messages(engine)
    user_texts = [(m.text or "").strip() for m in msgs if getattr(m, "role", "") == "user"]

    # All three distinct user messages survived.
    assert "original prompt" in user_texts
    assert "note one" in user_texts
    assert "note two" in user_texts
    # Order: original → note one → note two.
    order = [t for t in user_texts if t in {"original prompt", "note one", "note two"}]
    assert order == ["original prompt", "note one", "note two"], f"Unexpected ordering: {order}"

    # No "continue" placeholder ever landed in history.
    assert "continue" not in user_texts, "Placeholder 'continue' must not leak into persisted history"

    # Still a single turn — two retries didn't bump turn_counter.
    assert engine._executor.history_state.get("turn_counter", 0) == 1, "Two mid-turn retries must stay inside turn 1"

    # No stale interrupted markers after the final successful retry.
    interrupted = [
        m
        for m in msgs
        if (getattr(m, "additional_properties", None) or {}).get(HistoryMarkerKind.KEY) == HistoryMarkerKind.INTERRUPTED
    ]
    assert len(interrupted) == 0
