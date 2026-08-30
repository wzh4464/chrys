# Copyright (c) 2026 Chrys. All rights reserved.

"""Unit tests for LoopRecorder, _merge_loop_messages, and _remove_trailing_agent_text.

Verifies that:
- LoopRecorder correctly captures tool loop messages
- _merge_loop_messages inserts recovered messages at the correct position
- _remove_trailing_agent_text drops leaked final responses on interrupt
- Normal completion (no lost messages) is a no-op
- Edge cases: single iteration, no tool calls, empty state
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.recovery import RecoveryPersistOutcome
from chrys.foundation.tool_invocation_order import TOOL_INVOCATION_ORDER_KEY
from chrys.foundation.tool_result_metadata import TOOL_RESULT_METADATA_KEY
from chrys.foundation.trajectory.metadata import ANALYTICS_ITEM_ID_KEY
from chrys.kernel import Agent, ChatResponse, Content, LoopRecorder, Message, tool
from chrys.service.llm.mock import MockChatClient, MockResponse


@tool
def loop_capture_test_tool() -> str:
    return "tool result"


def _stamped_call(ordinal: int, *, call_id: str | None = None) -> Content:
    call = Content.from_function_call(call_id if call_id is not None else f"call-{ordinal}", "echo", arguments={})
    call.additional_properties[TOOL_INVOCATION_ORDER_KEY] = ordinal
    return call


# ---------------------------------------------------------------------------
# LoopRecorder unit tests
# ---------------------------------------------------------------------------


class TestLoopRecorder:
    """Test the loop-fed LoopRecorder (#405 capture semantics)."""

    def test_initial_state(self):
        capture = LoopRecorder()
        assert capture.loop_messages is None

    def test_single_iteration_no_loop_messages(self):
        """Single LLM call (no tool loop): loop_messages is None."""
        capture = LoopRecorder()
        # Simulate first (and only) LLM call with 5 initial messages
        capture._initial_count = 5
        capture._captured = [Message("system", ["sys"]), Message("user", ["hi"])]  # Only initial
        assert capture.loop_messages is None

    def test_multi_iteration_captures_delta(self):
        """Multiple iterations: loop_messages returns the delta."""
        capture = LoopRecorder()

        # Simulate: initial messages = [sys, history, user] (count=3)
        sys_msg = Message("system", ["instructions"])
        hist_msg = Message("assistant", ["previous response"])
        user_msg = Message("user", ["do something"])
        initial = [sys_msg, hist_msg, user_msg]

        capture._initial_count = len(initial)

        # After iteration 0: prepped_messages extended with assistant+tool
        iter0_assistant = Message("assistant", ["tool call 0"])
        iter0_tool = Message("tool", ["result 0"])

        # After iteration 1: prepped_messages extended again
        iter1_assistant = Message("assistant", ["tool call 1"])
        iter1_tool = Message("tool", ["result 1"])

        # Captured at iteration 2 (before 3rd LLM call)
        capture._captured = [*initial, iter0_assistant, iter0_tool, iter1_assistant, iter1_tool]

        loop_msgs = capture.loop_messages
        assert loop_msgs is not None
        assert len(loop_msgs) == 4
        assert loop_msgs[0] is iter0_assistant
        assert loop_msgs[1] is iter0_tool
        assert loop_msgs[2] is iter1_assistant
        assert loop_msgs[3] is iter1_tool

    def test_service_side_loop_captures_replaced_continuation_messages(self):
        """Responses-style service continuation can replace the growing local prepped list."""
        capture = LoopRecorder(capture_service_loop_messages=True)
        assistant_call = Message("assistant", [Content.from_function_call("call-1", "sleep", arguments={})])
        tool_result = Message("tool", [Content.from_function_result("call-1", result="slept")])

        capture._initial_count = 1
        capture._captured = [tool_result]
        capture._service_loop_messages = [assistant_call, tool_result]

        assert capture.loop_messages == [assistant_call, tool_result]

    @pytest.mark.asyncio
    async def test_conversation_id_alone_does_not_enable_service_capture(self):
        """Non-Responses providers may populate conversation_id; that should not switch capture mode."""
        capture = LoopRecorder()
        assistant_call = Message("assistant", [Content.from_function_call("call-1", "sleep", arguments={})])

        await capture.record_pre_call([Message("user", ["hi"])])
        capture.record_response(ChatResponse(messages=[assistant_call], conversation_id="provider-conversation"))

        assert capture.loop_messages is None
        assert capture._service_loop_messages == []

    @pytest.mark.asyncio
    async def test_checkpoint_callback_is_suppressed_for_unchanged_prefix(self):
        calls = 0

        async def on_checkpoint() -> None:
            nonlocal calls
            calls += 1

        capture = LoopRecorder(on_checkpoint=on_checkpoint)
        messages = [Message("user", ["hi"])]

        await capture.record_pre_call(messages)
        await capture.record_pre_call(messages)

        assert calls == 1

    @pytest.mark.asyncio
    async def test_checkpoint_callback_runs_when_prefix_content_changes(self):
        calls = 0

        async def on_checkpoint() -> None:
            nonlocal calls
            calls += 1

        capture = LoopRecorder(on_checkpoint=on_checkpoint)

        await capture.record_pre_call([Message("user", ["hi"])])
        await capture.record_pre_call([Message("user", ["different"])])

        assert calls == 2

    @pytest.mark.asyncio
    async def test_checkpoint_suppression_resets_between_runs(self):
        calls = 0

        async def on_checkpoint() -> None:
            nonlocal calls
            calls += 1

        capture = LoopRecorder(on_checkpoint=on_checkpoint)
        messages = [Message("user", ["hi"])]

        await capture.record_pre_call(messages)
        capture.reset()
        await capture.record_pre_call(messages)

        assert calls == 2

    @pytest.mark.asyncio
    async def test_service_capture_uses_explicit_mode_not_conversation_id(self):
        capture = LoopRecorder(capture_service_loop_messages=True)
        assistant_call = Message("assistant", [Content.from_function_call("call-1", "sleep", arguments={})])

        await capture.record_pre_call([Message("user", ["hi"])])
        capture.record_response(ChatResponse(messages=[assistant_call]))

        assert capture.loop_messages == [assistant_call]

    @pytest.mark.asyncio
    async def test_streaming_service_capture_records_compact_continuation_messages(self):
        """Streaming Responses continuations finalize through ResponseStream result hooks."""
        capture = LoopRecorder(capture_service_loop_messages=True)
        client = MockChatClient(
            responses=[
                MockResponse(
                    tool_calls=[("loop_capture_test_tool", "call-1", {})],
                    conversation_id="resp_tool_call",
                    chunk_delay=0,
                ),
                MockResponse(text="done", conversation_id="resp_done", chunk_delay=0),
            ]
        )

        async with Agent(
            client=client,
            name="capture-test",
            instructions="Use the tool, then finish.",
            tools=[loop_capture_test_tool],
        ) as agent:
            stream = agent.run(
                "hi",
                stream=True,
                options={"store": True},
                client_kwargs={"loop_recorder": capture},
            )
            async for _update in stream:
                pass
            final = await stream.get_final_response()

        assert final.text == "done"
        assert [msg.role for msg in client.call_history[1][0]] == ["tool"]
        loop_messages = capture.loop_messages
        assert loop_messages is not None
        assert [msg.role for msg in loop_messages] == ["assistant", "tool"]
        assert loop_messages[0].contents[0].call_id == "call-1"
        assert loop_messages[1].contents[0].call_id == "call-1"

    def test_reset_clears_state(self):
        capture = LoopRecorder()
        capture._initial_count = 5
        capture._captured = [Message("system", ["x"])] * 10
        capture._service_loop_messages = [Message("assistant", ["service"])]
        assert capture.loop_messages is not None

        capture.reset()
        assert capture.loop_messages is None
        assert capture._initial_count is None
        assert capture._captured is None
        assert capture._service_loop_messages == []

    def test_pending_journal_projects_only_answered_ordinals_in_slot_order(self) -> None:
        capture = LoopRecorder()
        calls = [_stamped_call(index) for index in range(5)]
        assistant = Message("assistant", ["working", *calls])
        commits = capture.stage_exchange([assistant], calls, result_carrier_item_id="a" * 32)

        for index in (0, 2, 4):
            commits[index].commit_final(Content.from_function_result(calls[index].call_id, result=f"result-{index}"))

        projected = capture.loop_messages
        assert projected is not None
        assert [message.role for message in projected] == ["assistant", "tool"]
        assert projected[0].text == "working"
        assert [content for content in projected[0].contents if content.type == "function_call"] == [
            calls[0],
            calls[2],
            calls[4],
        ]
        assert [content.result for content in projected[1].contents] == ["result-0", "result-2", "result-4"]
        assert projected[1].additional_properties[ANALYTICS_ITEM_ID_KEY] == "a" * 32
        assert capture.committed_count == 3

    def test_sealed_journal_reuses_canonical_result_content_without_duplicates(self) -> None:
        capture = LoopRecorder()
        calls = [_stamped_call(0), _stamped_call(1)]
        assistant = Message("assistant", calls)
        commits = capture.stage_exchange([assistant], calls, result_carrier_item_id="a" * 32)
        results = [
            Content.from_function_result(calls[0].call_id, result="first"),
            Content.from_function_result(calls[1].call_id, result="second"),
        ]
        for commit, result in zip(commits, results, strict=True):
            commit.commit_final(result)
        canonical_tool = Message("tool", results)
        capture.seal_exchange(canonical_tool)
        capture._initial_count = 0
        capture._captured = [assistant, canonical_tool]

        projected = capture.loop_messages
        assert projected == [assistant, canonical_tool]
        assert projected[1] is canonical_tool
        assert projected[1].contents == results
        assert capture.committed_count == 2

    def test_falsy_id_slots_are_associated_by_ordinal_not_position(self) -> None:
        capture = LoopRecorder(capture_service_loop_messages=True)
        first = _stamped_call(0, call_id="")
        second = _stamped_call(1, call_id="")
        assistant = Message("assistant", [first, second])
        capture.record_response(ChatResponse(messages=[assistant]))
        commits = capture.stage_exchange([assistant], [first, second], result_carrier_item_id="a" * 32)

        commits[1].commit_final(Content.from_function_result("", result="second-only"))

        projected = capture.loop_messages
        assert projected is not None
        assert projected[0].contents == [second]
        assert [content.result for content in projected[1].contents] == ["second-only"]

    def test_snapshot_restore_preserves_commits_and_reset_starts_a_new_pass(self) -> None:
        capture = LoopRecorder()
        snapshot = capture.snapshot()
        unanswered = _stamped_call(0)
        capture.stage_exchange([Message("assistant", [unanswered])], [unanswered], result_carrier_item_id="a" * 32)

        capture.restore(snapshot)
        assert capture.loop_messages is None
        assert capture.committed_count == 0

        answered = _stamped_call(1)
        commit = capture.stage_exchange(
            [Message("assistant", [answered])], [answered], result_carrier_item_id="a" * 32
        )[0]
        result = Content.from_function_result(answered.call_id, result="durable")
        commit.commit_final(result)
        capture.restore(snapshot)

        assert capture.committed_count == 1
        assert capture.loop_messages is not None
        capture.seal_exchange(Message("tool", [result]))
        capture.restore(snapshot)
        assert capture.committed_count == 1

        capture.reset()
        assert capture.committed_count == 0
        assert capture.loop_messages is None

    def test_interrupted_fills_are_not_commits_and_restore_allows_redispatch(self) -> None:
        capture = LoopRecorder()
        snapshot = capture.snapshot()
        call = _stamped_call(0)
        commit = capture.stage_exchange([Message("assistant", [call])], [call], result_carrier_item_id="a" * 32)[0]
        commit.commit_interrupted(call, None)

        assert capture.committed_count == 0
        projected = capture.loop_messages
        assert projected is not None
        marker = projected[-1].contents[0]
        assert marker.additional_properties[TOOL_RESULT_METADATA_KEY]["interrupted"] is True

        capture.restore(snapshot)
        assert capture.loop_messages is None
        assert capture.committed_count == 0

    @pytest.mark.asyncio
    async def test_barrier_retries_non_persisted_outcomes_once_then_degrades(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        outcomes = [RecoveryPersistOutcome.FAILED, RecoveryPersistOutcome.NOTHING_TO_PERSIST]
        barrier_calls = 0
        checkpoint_calls = 0

        async def barrier() -> RecoveryPersistOutcome:
            nonlocal barrier_calls
            outcome = outcomes[barrier_calls]
            barrier_calls += 1
            return outcome

        async def checkpoint() -> None:
            nonlocal checkpoint_calls
            checkpoint_calls += 1

        capture = LoopRecorder(on_pre_wire_barrier=barrier, on_result_checkpoint=checkpoint)
        call = _stamped_call(0)
        commit = capture.stage_exchange([Message("assistant", [call])], [call], result_carrier_item_id="a" * 32)[0]
        commit.commit_final(Content.from_function_result(call.call_id, result="done"))
        await asyncio.sleep(0)

        with caplog.at_level(logging.ERROR, logger="chrys.kernel.loop"):
            await capture.record_pre_call([Message("user", ["continue"])])
            await capture.record_pre_call([Message("user", ["continue again"])])

        assert barrier_calls == 2
        assert checkpoint_calls >= 1
        assert (
            caplog.messages.count(
                "Recovery sidecar persistence failed twice after committed tool work; "
                "continuing with the in-memory journal."
            )
            == 1
        )

    @pytest.mark.asyncio
    async def test_unconfigured_barrier_becomes_a_silent_permanent_noop(self) -> None:
        calls = 0

        async def barrier() -> RecoveryPersistOutcome:
            nonlocal calls
            calls += 1
            return RecoveryPersistOutcome.UNCONFIGURED

        capture = LoopRecorder(on_pre_wire_barrier=barrier)
        call = _stamped_call(0)
        commit = capture.stage_exchange([Message("assistant", [call])], [call], result_carrier_item_id="a" * 32)[0]
        commit.commit_final(Content.from_function_result(call.call_id, result="done"))

        await capture.record_pre_call([Message("user", ["one"])])
        await capture.record_pre_call([Message("user", ["two"])])

        assert calls == 1

    @pytest.mark.asyncio
    async def test_barrier_fault_keeps_narration_and_approval_metadata_aligned_to_answered_slot(self) -> None:
        async def barrier() -> RecoveryPersistOutcome:
            return RecoveryPersistOutcome.FAILED

        capture = LoopRecorder(on_pre_wire_barrier=barrier)
        first = _stamped_call(0)
        second = _stamped_call(1)
        assistant = Message("assistant", ["I need approval before the write.", first, second])
        commits = capture.stage_exchange([assistant], [first, second], result_carrier_item_id="a" * 32)
        approved_result = Content.from_function_result(second.call_id, result="approved result")
        approved_result.additional_properties[TOOL_RESULT_METADATA_KEY] = {
            "approval": "user_approved",
        }
        commits[1].commit_final(approved_result)

        await capture.record_pre_call([Message("user", ["continue"])])

        projected = capture.loop_messages
        assert projected is not None
        assert projected[0].text == "I need approval before the write."
        assert [content.call_id for content in projected[0].contents if content.type == "function_call"] == [
            second.call_id
        ]
        assert projected[1].contents == [approved_result]
        assert projected[1].contents[0].additional_properties[TOOL_RESULT_METADATA_KEY] == {
            "approval": "user_approved",
        }

    def test_stale_initial_count_causes_duplication(self):
        """Without reset between approval iterations, stale _initial_count
        causes the capture to include messages already stored by the framework,
        leading to duplicates in session state.
        """
        capture = LoopRecorder()

        # --- Run 1: LLM generates read_file, then write_file (needs approval) ---
        # Before LLM call 1: initial_count set
        run1_context_before_call1 = [
            Message("system", ["sys"]),
            Message("user", ["do work"]),
        ]
        capture._initial_count = len(run1_context_before_call1)  # 2

        # Before LLM call 2: captured includes read_file from call 1
        read_call = Message("assistant", ["read_file call"])
        read_result = Message("tool", ["read_file result"])
        run1_context_before_call2 = [*run1_context_before_call1, read_call, read_result]
        capture._captured = list(run1_context_before_call2)

        # Loop messages from run 1 correctly captures read_file pair
        assert capture.loop_messages == [read_call, read_result]

        # --- Without reset, start run 2 (re-submission with approval) ---
        # Run 2's context is different — includes write_file approval in input
        write_call = Message("assistant", ["write_file call"])
        write_approval = Message("user", ["approved"])
        write_result = Message("tool", ["write_file result"])
        # After write_file executes, LLM returns read_file_2
        read_call_2 = Message("assistant", ["read_file_2 call"])
        read_result_2 = Message("tool", ["read_file_2 result"])

        # Before LLM call 2 of run 2 (captured includes run 2's context)
        run2_context = [
            Message("system", ["sys"]),
            Message("user", ["do work"]),
            write_call,
            write_approval,
            write_result,
            read_call_2,
            read_result_2,
        ]
        # _initial_count is still 2 from run 1!
        capture._captured = list(run2_context)

        # BUG: loop_messages includes write_file messages that the framework
        # already stored via after_run — causes duplication
        stale_loop_msgs = capture.loop_messages
        assert stale_loop_msgs is not None
        assert len(stale_loop_msgs) == 5  # Too many! Includes write_file + approval + result
        assert write_call in stale_loop_msgs  # These would be duplicates

    def test_reset_between_approval_iterations_prevents_duplication(self):
        """With reset after each capture, the next run starts fresh and only
        captures its own intermediate messages.
        """
        capture = LoopRecorder()
        pending: list[Message] = []

        # --- Run 1: captures read_file ---
        run1_initial = [Message("system", ["sys"]), Message("user", ["do work"])]
        capture._initial_count = len(run1_initial)
        read_call = Message("assistant", ["read_file call"])
        read_result = Message("tool", ["read_file result"])
        capture._captured = [*run1_initial, read_call, read_result]

        loop_msgs = capture.loop_messages
        assert loop_msgs == [read_call, read_result]
        pending.extend(loop_msgs)

        # Reset after capture (the fix)
        capture.reset()
        assert capture.loop_messages is None

        # --- Run 2: fresh start, captures read_file_2 only ---
        write_call = Message("assistant", ["write_file call"])
        write_approval = Message("user", ["approved"])
        write_result = Message("tool", ["write_file result"])
        read_call_2 = Message("assistant", ["read_file_2 call"])
        read_result_2 = Message("tool", ["read_file_2 result"])

        run2_initial = [
            Message("system", ["sys"]),
            Message("user", ["do work"]),
            write_call,
            write_approval,
            write_result,
        ]
        capture._initial_count = len(run2_initial)  # Fresh count for run 2
        capture._captured = [*run2_initial, read_call_2, read_result_2]

        loop_msgs_2 = capture.loop_messages
        assert loop_msgs_2 == [read_call_2, read_result_2]  # Only run 2's intermediates
        assert write_call not in loop_msgs_2  # No duplicates
        pending.extend(loop_msgs_2)

        # Final pending has both runs' intermediates, no duplicates
        assert len(pending) == 4
        assert pending == [read_call, read_result, read_call_2, read_result_2]


# ---------------------------------------------------------------------------
# _merge_loop_messages logic tests (via engine helper)
# ---------------------------------------------------------------------------


def _make_engine_state(messages: list[Message]) -> dict:
    """Build a minimal chrys_history state dict."""
    return {"messages": messages}


class TestMergeLoopMessages:
    """Test the merge logic that inserts recovered loop messages into state."""

    def test_merge_inserts_after_user_message(self):
        """Loop messages are inserted between user message and last-iteration messages."""
        # State after after_run: [old_history, user_input, last_assistant, last_tool]
        old_assistant = Message("assistant", ["old response"])
        user_msg = Message("user", ["do work"])
        last_assistant = Message("assistant", ["interrupted tool call"])
        last_tool = Message("tool", [""])  # empty result from interrupt

        messages = [old_assistant, user_msg, last_assistant, last_tool]

        # Loop messages from completed iterations
        iter0_assistant = Message("assistant", ["tool call 0"])
        iter0_tool = Message("tool", ["result 0"])
        iter1_assistant = Message("assistant", ["tool call 1"])
        iter1_tool = Message("tool", ["result 1"])
        loop_msgs = [iter0_assistant, iter0_tool, iter1_assistant, iter1_tool]

        # Simulate _merge_loop_messages logic
        user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if getattr(messages[i], "role", "") == "user":
                user_idx = i
                break

        insert_at = user_idx + 1
        for j, lm in enumerate(loop_msgs):
            messages.insert(insert_at + j, lm)

        # Verify order: old_assistant, user_msg, iter0_*, iter1_*, last_*
        assert messages[0] is old_assistant
        assert messages[1] is user_msg
        assert messages[2] is iter0_assistant
        assert messages[3] is iter0_tool
        assert messages[4] is iter1_assistant
        assert messages[5] is iter1_tool
        assert messages[6] is last_assistant
        assert messages[7] is last_tool

    def test_skip_when_already_present(self):
        """Normal completion: loop messages already in state, merge is no-op."""
        # In normal completion, _prepend_fcc_messages puts all messages in response
        iter0_assistant = Message("assistant", ["tool call 0"])
        iter0_tool = Message("tool", ["result 0"])
        user_msg = Message("user", ["do work"])
        final_text = Message("assistant", ["done!"])

        messages = [user_msg, iter0_assistant, iter0_tool, final_text]

        # Check identity — loop messages are already present
        loop_msgs = [iter0_assistant, iter0_tool]
        first_loop = loop_msgs[0]

        already_present = any(m is first_loop for m in messages)
        assert already_present, "Identity check should detect already-merged messages"

    def test_no_user_message_is_noop(self):
        """If there's no user message in state, merge does nothing."""
        messages = [Message("assistant", ["some text"])]
        loop_msgs = [Message("assistant", ["tool"]), Message("tool", ["result"])]

        user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if getattr(messages[i], "role", "") == "user":
                user_idx = i
                break

        original_len = len(messages)
        if user_idx >= 0:
            insert_at = user_idx + 1
            for j, lm in enumerate(loop_msgs):
                messages.insert(insert_at + j, lm)

        assert len(messages) == original_len  # No insertion

    def test_insert_index_places_pass_after_retained_earlier_work(self):
        """The pass boundary beats the user anchor when earlier-pass work follows it.

        Empty-retry repair: the synthetic continuation user message was
        removed before the merge, so the last user message sits BEFORE the
        earlier pass's retained call/result pair. Anchoring to it alone would
        insert this pass's messages ahead of the earlier pass, inverting the
        transcript.
        """
        from chrys.service.session.history import SessionHistoryManager

        user_msg = Message("user", ["run it"])
        pass1_call = Message("assistant", ["invalid call"])
        pass1_result = Message("tool", ["Error: invalid arguments"])
        messages = [user_msg, pass1_call, pass1_result]

        retry_call = Message("assistant", ["valid call"])
        retry_result = Message("tool", ["done"])
        capture = LoopRecorder()
        capture._initial_count = 0
        capture._captured = [retry_call, retry_result]

        history = SessionHistoryManager()
        history.bind({"messages": messages})
        history.merge_loop_messages(capture, insert_index=3)

        assert messages == [user_msg, pass1_call, pass1_result, retry_call, retry_result]

    def test_insert_index_defers_to_later_user_anchor(self):
        """A user message after the boundary (mid-run injection) still anchors the merge."""
        from chrys.service.session.history import SessionHistoryManager

        user_msg = Message("user", ["run it"])
        injection = Message("user", ["also do this"])
        messages = [user_msg, injection]

        loop_call = Message("assistant", ["call"])
        loop_result = Message("tool", ["result"])
        capture = LoopRecorder()
        capture._initial_count = 0
        capture._captured = [loop_call, loop_result]

        history = SessionHistoryManager()
        history.bind({"messages": messages})
        history.merge_loop_messages(capture, insert_index=1)

        assert messages == [user_msg, injection, loop_call, loop_result]


# ---------------------------------------------------------------------------
# _remove_trailing_agent_text logic tests
# ---------------------------------------------------------------------------


def _remove_trailing_agent_text(messages: list[Message]) -> bool:
    """Replicate the engine's _remove_trailing_agent_text logic for testing.

    Returns True if a message was removed.
    """
    if not messages:
        return False
    last = messages[-1]
    if getattr(last, "role", "") != "assistant":
        return False
    props = getattr(last, "additional_properties", None) or {}
    if props.get(HistoryMarkerKind.KEY):
        return False
    contents = getattr(last, "contents", [])
    has_function_call = any(getattr(c, "type", "") == "function_call" for c in contents)
    if has_function_call:
        return False
    messages.pop()
    return True


class TestRemoveTrailingAgentText:
    """Test _remove_trailing_agent_text logic."""

    def test_removes_text_only_assistant(self):
        """Trailing text-only assistant message is removed."""
        user_msg = Message("user", ["do something"])
        tool_assistant = Message("assistant", [Content("function_call", name="echo", call_id="c1")])
        tool_result = Message("tool", [Content("function_result", call_id="c1", result="done")])
        final_text = Message("assistant", ["Here is the final answer."])

        messages = [user_msg, tool_assistant, tool_result, final_text]
        removed = _remove_trailing_agent_text(messages)

        assert removed
        assert len(messages) == 3
        assert messages[-1] is tool_result

    def test_preserves_assistant_with_function_calls(self):
        """Assistant message with function_calls is not removed."""
        user_msg = Message("user", ["do something"])
        tool_assistant = Message("assistant", [Content("function_call", name="echo", call_id="c1")])

        messages = [user_msg, tool_assistant]
        removed = _remove_trailing_agent_text(messages)

        assert not removed
        assert len(messages) == 2
        assert messages[-1] is tool_assistant

    def test_preserves_tool_message(self):
        """Trailing tool message is not removed (role != assistant)."""
        user_msg = Message("user", ["do something"])
        tool_result = Message("tool", [Content("function_result", call_id="c1", result="done")])

        messages = [user_msg, tool_result]
        removed = _remove_trailing_agent_text(messages)

        assert not removed
        assert len(messages) == 2

    def test_preserves_chrys_marker(self):
        """Chrys internal markers are not removed."""
        user_msg = Message("user", ["do something"])
        marker = Message("assistant", ["Execution interrupted"])
        marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.INTERRUPTED

        messages = [user_msg, marker]
        removed = _remove_trailing_agent_text(messages)

        assert not removed
        assert len(messages) == 2

    def test_empty_messages_noop(self):
        """Empty message list is a no-op."""
        messages: list[Message] = []
        removed = _remove_trailing_agent_text(messages)

        assert not removed
        assert len(messages) == 0

    def test_removes_when_no_tools_at_all(self):
        """If LLM returned text-only on first call (no tools), still removes."""
        user_msg = Message("user", ["explain something"])
        text_response = Message("assistant", ["Here is my explanation."])

        messages = [user_msg, text_response]
        removed = _remove_trailing_agent_text(messages)

        assert removed
        assert len(messages) == 1
        assert messages[0] is user_msg
