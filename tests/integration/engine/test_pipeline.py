# Copyright (c) 2026 Chrys. All rights reserved.

"""Pipeline integration tests — end-to-end from LLM client through engine to session state.

Tests cover:
- S1: Multi-turn sequential tool calls
- S2: Parallel tool calls (single LLM response)
- S3: Intermediate text alongside tool calls
- S4: Approval auto-approve with rollback
- S5: Approval user-approve / user-reject
- S6: Interrupt + resume
- S7: Three-way consistency (events vs session vs replay)
- S8: Approval preserves intermediate tool calls
- S9: Multi-approval iterations (no duplicate messages)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

from chrys.foundation.events.types import ApprovalRequest, ToolCallResult, ToolCallStart, UserMessage
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.tool_kinds import KIND_SKILL
from chrys.service.llm.mock import MockResponse
from chrys.service.profiles.agents.schema import SkillConfig, SkillsConfig
from chrys.service.skills.constants import RUN_SKILL_SCRIPT_TOOL_NAME
from tests.support.pipeline_helpers import (
    create_test_engine,
    extract_approval_requests,
    extract_final_messages,
    extract_intermediate_messages,
    extract_session_approvals,
    extract_session_batch_ids,
    extract_session_call_ids,
    extract_session_intermediate_texts,
    extract_session_tool_names,
    extract_tool_results,
    extract_tool_starts,
    wait_for_event,
    wait_for_idle,
)
from tests.support.waiting import wait_for

if TYPE_CHECKING:
    from tests.support.pipeline_helpers import PipelineTestContext


async def _wait_for_loop_snapshot(ctx: PipelineTestContext, min_iterations: int, *, timeout: float = 5.0) -> None:
    """Wait until the LoopRecorder has snapshotted ``min_iterations`` completed iterations.

    Replaces the old fixed ``asyncio.sleep`` settle that gave the framework time
    to enter the next tool-loop iteration so ``LoopRecorder.record_pre_call``
    captures the completed iteration(s) into recoverable state.  Each completed
    iteration contributes an assistant (function_call) + tool (function_result)
    message pair to ``loop_messages``, so ``>= 2 * min_iterations`` snapshotted
    messages proves the iterations are recoverable before we interrupt.
    """

    def _snapshotted() -> bool:
        recorder = ctx.engine._loop_recorder
        if recorder is None:
            return False
        loop_msgs = recorder.loop_messages
        return loop_msgs is not None and len(loop_msgs) >= 2 * min_iterations

    await wait_for(_snapshotted, timeout=timeout, description="workspace snapshot")


# ---------------------------------------------------------------------------
# S1: Multi-turn sequential tool calls
# ---------------------------------------------------------------------------


class TestMultiTurnSequential:
    """Two turns, each with sequential tool calls from different LLM responses."""

    @pytest.fixture
    async def ctx(self, tmp_path):
        responses = [
            # Turn 1: echo → concat → final text
            MockResponse(tool_calls=[("echo", "c1", {"message": "hello"})]),
            MockResponse(tool_calls=[("concat", "c2", {"a": "x", "b": "y"})]),
            MockResponse(text="Turn 1 complete"),
            # Turn 2: echo → final text
            MockResponse(tool_calls=[("echo", "c3", {"message": "world"})]),
            MockResponse(text="Turn 2 complete"),
        ]
        ctx = await create_test_engine(responses, tmp_path)
        yield ctx
        await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_events(self, ctx):
        """Events: correct ToolCallStart/Result pairs and final messages."""
        await ctx.send_message("Do turn 1")
        await ctx.send_message("Do turn 2")

        starts = extract_tool_starts(ctx.events)
        results = extract_tool_results(ctx.events)
        finals = extract_final_messages(ctx.events)

        assert len(starts) == 3
        assert len(results) == 3
        assert starts[0]["tool_name"] == "echo"
        assert starts[1]["tool_name"] == "concat"
        assert starts[2]["tool_name"] == "echo"

        assert len(finals) == 2
        assert finals[0] == "Turn 1 complete"
        assert finals[1] == "Turn 2 complete"

    @pytest.mark.asyncio
    async def test_session_batch_ids(self, ctx):
        """Session: sequential tool calls get distinct batch_ids per LLM response."""
        await ctx.send_message("Do turn 1")
        await ctx.send_message("Do turn 2")

        raw = await ctx.get_session_messages()
        tool_names = extract_session_tool_names(raw)
        batch_ids = extract_session_batch_ids(raw)

        assert tool_names == ["echo", "concat", "echo"]
        assert len(batch_ids) == 3
        assert all(b is not None for b in batch_ids)
        # echo and concat are from different LLM responses → distinct batch_ids
        assert batch_ids[0] != batch_ids[1]


class TestSlowTurnBoundary:
    """A turn outlasting wait_for_idle's historical 5s ceiling stays one turn.

    wait_for_idle used to silently suppress its 5s timeout: on an overloaded
    runner send_message returned mid-turn, so the test's NEXT message was
    consumed as an injection into turn 1 instead of starting turn 2, and
    turn 2's tool call vanished from the persisted session (the recurring
    Windows CI failure).  The wait is now generous and fails loudly instead;
    this test pins that by keeping turn 1 active beyond the old ceiling.
    """

    @pytest.mark.asyncio
    async def test_turn_slower_than_legacy_wait_ceiling_keeps_turn_boundary(self, tmp_path):
        from chrys.kernel import FunctionTool

        async def _stalled_echo(message: str) -> str:
            # Beyond the old 5s wait_for_idle ceiling, well under the new 30s.
            await asyncio.sleep(6.0)
            return f"echo: {message}"

        async def _echo(message: str) -> str:
            return f"echo: {message}"

        stalled_echo = FunctionTool(func=_stalled_echo, name="stalled_echo", description="Echo after a stall")
        echo = FunctionTool(func=_echo, name="echo", description="Echo")

        responses = [
            # Turn 1: one tool call that stalls past the legacy ceiling
            MockResponse(tool_calls=[("stalled_echo", "sl1", {"message": "slow"})]),
            MockResponse(text="Turn 1 complete"),
            # Turn 2: a fast tool call that must land in a NEW turn
            MockResponse(tool_calls=[("echo", "sl2", {"message": "fast"})]),
            MockResponse(text="Turn 2 complete"),
        ]
        ctx = await create_test_engine(responses, tmp_path, tools=[stalled_echo, echo])
        try:
            await ctx.send_message("Do turn 1")
            await ctx.send_message("Do turn 2")

            # Two distinct final messages prove message 2 started its own turn
            # rather than being injected into the still-running turn 1.
            finals = extract_final_messages(ctx.events)
            assert finals == ["Turn 1 complete", "Turn 2 complete"]

            raw = await ctx.get_session_messages()
            assert extract_session_tool_names(raw) == ["stalled_echo", "echo"]
        finally:
            await ctx.cleanup()


# ---------------------------------------------------------------------------
# S2: Parallel tool calls (single LLM response)
# ---------------------------------------------------------------------------


class TestParallelToolCalls:
    """One LLM response returns multiple tool_calls simultaneously."""

    @pytest.fixture
    async def ctx(self, tmp_path):
        responses = [
            MockResponse(
                tool_calls=[
                    ("echo", "p1", {"message": "a"}),
                    ("concat", "p2", {"a": "b", "b": "c"}),
                    ("uppercase", "p3", {"text": "hello"}),
                ]
            ),
            MockResponse(text="All done"),
        ]
        ctx = await create_test_engine(responses, tmp_path)
        yield ctx
        await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_events(self, ctx):
        """Events: 3 starts + 3 results from single LLM response."""
        await ctx.send_message("Run three tools")

        starts = extract_tool_starts(ctx.events)
        results = extract_tool_results(ctx.events)

        assert len(starts) == 3
        assert len(results) == 3
        tool_names = {s["tool_name"] for s in starts}
        assert tool_names == {"echo", "concat", "uppercase"}

    @pytest.mark.asyncio
    async def test_session_batch_ids(self, ctx):
        """Session: all parallel tools share the same batch_id."""
        await ctx.send_message("Run three tools")

        raw = await ctx.get_session_messages()
        batch_ids = extract_session_batch_ids(raw)

        # All 3 tool calls are in a single assistant message with one batch_id
        assert len(batch_ids) == 1
        assert batch_ids[0] is not None


# ---------------------------------------------------------------------------
# S3: Intermediate text alongside tool calls
# ---------------------------------------------------------------------------


class TestIntermediateText:
    """LLM returns text alongside tool calls — mock client now fires callbacks."""

    @pytest.mark.asyncio
    async def test_intermediate_text_callback(self, tmp_path):
        """Mock client fires intermediate text callback for text+tools response."""
        responses = [
            MockResponse(text="Let me search...", tool_calls=[("echo", "it1", {"message": "query"})]),
            MockResponse(text="Found the answer!"),
        ]
        ctx = await create_test_engine(responses, tmp_path)

        try:
            await ctx.send_message("Search for something")

            # Intermediate text should be captured via the callback
            intermediates = extract_intermediate_messages(ctx.events)
            assert len(intermediates) >= 1

            # Final message should arrive
            finals = extract_final_messages(ctx.events)
            assert len(finals) == 1
            assert finals[0] == "Found the answer!"
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_no_duplicate_intermediate_text(self, tmp_path):
        """When text is already in message contents, _intermediate_text is NOT set."""
        responses = [
            MockResponse(text="Thinking...", tool_calls=[("echo", "it2", {"message": "q"})]),
            MockResponse(text="Done"),
        ]
        ctx = await create_test_engine(responses, tmp_path)

        try:
            await ctx.send_message("Think and act")

            raw = await ctx.get_session_messages()
            # The text is in the message contents (text + fn_calls),
            # so _intermediate_text should NOT be set (would cause duplication)
            itexts = extract_session_intermediate_texts(raw)
            assert len(itexts) == 0

            # But the text should be in the message contents
            for msg in raw:
                if msg.get("role") == "assistant":
                    contents = msg.get("contents", [])
                    has_fc = any(isinstance(c, dict) and c.get("type") == "function_call" for c in contents)
                    text_parts = [
                        c.get("text", "") for c in contents if isinstance(c, dict) and c.get("type") == "text"
                    ]
                    if has_fc and text_parts:
                        assert "Thinking" in text_parts[0]
                        break
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_multi_turn_intermediate_text_isolation(self, tmp_path):
        """Intermediate texts from turn 2 must NOT be assigned to turn 1 messages.

        Regression test: _persist_intermediate_texts used to iterate from the
        start of the message list, so turn 2's texts would be assigned to
        unmatched assistant messages from turn 1.
        """
        responses = [
            # Turn 1: tool-only response (no intermediate text)
            MockResponse(tool_calls=[("echo", "mt1", {"message": "a"})]),
            MockResponse(text="Turn 1 done"),
            # Turn 2: text + tools (intermediate text)
            MockResponse(text="Planning...", tool_calls=[("echo", "mt2", {"message": "b"})]),
            MockResponse(text="Turn 2 done"),
        ]
        ctx = await create_test_engine(responses, tmp_path)

        try:
            # Turn 1
            await ctx.send_message("First turn")
            raw1 = await ctx.get_session_messages()
            itexts1 = extract_session_intermediate_texts(raw1)
            # Turn 1 had no intermediate text
            assert len(itexts1) == 0, f"Turn 1 should have no intermediate texts, got {itexts1}"

            # Turn 2
            await ctx.send_message("Second turn")
            raw2 = await ctx.get_session_messages()
            # Crucially, turn 1's messages must NOT have _intermediate_text
            for msg in raw2:
                if msg.get("role") != "assistant":
                    continue
                extra = msg.get("additional_properties", {}) or {}
                batch_id = extra.get("_batch_id")
                itext = extra.get("_intermediate_text")
                # Turn 1 messages (batch_id from turn 1) must not have itext
                # from turn 2.  If itext is set, it should match the batch's
                # own response, not a later turn's text.
                if itext and batch_id is not None:
                    # Verify the text makes sense for this batch
                    assert itext != "Planning...", (
                        f"Turn 2 intermediate text leaked to turn 1 message (batch_id={batch_id})"
                    )
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_parallel_tools_intermediate_text_not_duplicated(self, tmp_path):
        """When parallel tool calls are split into separate messages, only the
        first message of each batch should get intermediate text.

        Regression test: _persist_intermediate_texts used to assign one text
        per assistant message (not per batch), so text from batch 2 would leak
        to extra messages from batch 1's parallel tool calls.
        """
        responses = [
            # Batch 1: text + 3 parallel tools
            MockResponse(
                text="Step 1: searching...",
                tool_calls=[
                    ("echo", "p1", {"message": "a"}),
                    ("concat", "p2", {"a": "b", "b": "c"}),
                    ("uppercase", "p3", {"text": "d"}),
                ],
            ),
            # Batch 2: text + 1 tool
            MockResponse(text="Step 2: reading...", tool_calls=[("echo", "p4", {"message": "e"})]),
            MockResponse(text="Done"),
        ]
        ctx = await create_test_engine(responses, tmp_path)

        try:
            await ctx.send_message("Do work")
            raw = await ctx.get_session_messages()

            # At most 1 intermediate text should be set (for batch 1's first msg).
            # Batch 2's text should NOT leak to batch 1's other messages.
            # (Batch 2 might have 0 or 1 depending on whether the provider
            # preserves text in contents.)
            for msg in raw:
                if msg.get("role") != "assistant":
                    continue
                extra = msg.get("additional_properties", {}) or {}
                itext = extra.get("_intermediate_text")
                if itext:
                    assert itext != "Step 2: reading...", (
                        f"Batch 2 text leaked to batch 1 message (batch_id={extra.get('_batch_id')})"
                    )
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_batch_boundary_without_text(self, tmp_path):
        """Tool-only responses still fire empty callback → batch_id increments."""
        responses = [
            MockResponse(tool_calls=[("echo", "b1", {"message": "first"})]),
            MockResponse(tool_calls=[("echo", "b2", {"message": "second"})]),
            MockResponse(text="Done"),
        ]
        ctx = await create_test_engine(responses, tmp_path)

        try:
            await ctx.send_message("Run two tools")

            raw = await ctx.get_session_messages()
            batch_ids = extract_session_batch_ids(raw)

            assert len(batch_ids) == 2
            assert all(b is not None for b in batch_ids)
            # Two separate LLM responses → distinct batch_ids
            assert batch_ids[0] != batch_ids[1]
        finally:
            await ctx.cleanup()


# ---------------------------------------------------------------------------
# S4: Approval auto-approve with rollback
# ---------------------------------------------------------------------------


class TestAutoApproval:
    """With default="auto", all tools execute without approval — no ApprovalRequest events."""

    @pytest.fixture
    async def ctx(self, tmp_path):
        responses = [
            # Both guarded and normal tools execute inline (no approval needed)
            MockResponse(
                tool_calls=[
                    ("echo", "a1", {"message": "safe"}),
                    ("guarded_echo", "a2", {"message": "also safe with auto"}),
                ]
            ),
            MockResponse(text="Tools executed without approval"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_default="auto")
        yield ctx
        await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_auto_approval_events(self, ctx):
        """Events: no ApprovalRequest published when policy is auto."""
        await ctx.send_message("Run tools")

        approval_reqs = extract_approval_requests(ctx.events)
        assert len(approval_reqs) == 0

        # Both tools should execute
        results = extract_tool_results(ctx.events)
        assert len(results) == 2

        finals = extract_final_messages(ctx.events)
        assert len(finals) == 1

    @pytest.mark.asyncio
    async def test_session_no_approval_tags(self, ctx):
        """Session: no _approval tags when all tools are auto (no approval needed)."""
        await ctx.send_message("Run tools")

        raw = await ctx.get_session_messages()
        approvals = extract_session_approvals(raw)

        # No approval tags — tools just execute normally
        tagged = [a for a in approvals if a is not None]
        assert len(tagged) == 0


class TestRuntimeSkillApproval:
    """The real build/provider/kernel chain preserves live skill provenance."""

    @pytest.mark.asyncio
    async def test_built_agent_gates_runtime_skill_script_under_auto_default(self, tmp_path) -> None:
        responses = [
            MockResponse(
                tool_calls=[
                    (
                        RUN_SKILL_SCRIPT_TOOL_NAME,
                        "skill-call",
                        {"skill_name": "review", "script_name": "scripts/review.py"},
                    )
                ]
            ),
            MockResponse(text="Rejected script handled"),
        ]
        skills = SkillsConfig(
            inline=[SkillConfig(name="review", description="Review files", instructions="Review carefully.")],
            auto_load_user_agents_skills=False,
            auto_load_cwd_agents_skills=False,
        )
        ctx = await create_test_engine(responses, tmp_path, approval_default="auto", skills=skills)
        try:
            await ctx.bus.publish(UserMessage(text="Run the review skill"))

            approval_events = await wait_for_event(ctx.events, ApprovalRequest, timeout=20.0)
            request = approval_events[0]
            assert request.tool_name == RUN_SKILL_SCRIPT_TOOL_NAME
            assert request.tool_kind == KIND_SKILL

            await ctx.reject(request.request_id, reason="untrusted skill")
            await wait_for_idle(ctx)

            results = extract_tool_results(ctx.events)
            assert len(results) == 1
            assert results[0]["tool_name"] == RUN_SKILL_SCRIPT_TOOL_NAME
            assert "rejected by user" in results[0]["result"]
        finally:
            await ctx.cleanup()


class TestPendingApprovalDetection:
    """wait_for_idle's blocked-on-approval discriminator trusts the request.

    The production policy decides whether approval is needed (name overrides,
    kind overrides like "shell", qualified "kind.tool" keys, sensitive-access
    triggers) — a published ApprovalRequest already encodes that decision, so
    the helper must not re-derive the mode from the profile config: a
    kind-based override would resolve to "auto" by tool name and wrongly wait
    the full timeout on a run that only the test can unblock.
    """

    def _ctx_with_events(self, events):
        from tests.support.pipeline_helpers import PipelineTestContext

        return PipelineTestContext(
            engine=None, bus=None, events=list(events), mock_client=None, store=None, session_id=""
        )

    def test_unresponded_request_pends_regardless_of_override_shape(self):
        from tests.support.pipeline_helpers import _pending_user_approval

        # tool_name deliberately matches no override key — as with a
        # kind-based override ("shell") or a sensitive-access trigger.
        ctx = self._ctx_with_events([ApprovalRequest(request_id="r1", tool_name="run_command", tool_kind="shell")])
        assert _pending_user_approval(ctx) is True

    def test_judging_request_defers_to_the_verdict(self):
        from chrys.foundation.events.types import ApprovalReviewed
        from tests.support.pipeline_helpers import _pending_user_approval

        request = ApprovalRequest(request_id="r1", tool_name="run_command", tool_kind="shell", judging=True)

        # No verdict yet: the judge may still auto-approve (resolving the
        # middleware future without any ApprovalResponse), so keep waiting.
        assert _pending_user_approval(self._ctx_with_events([request])) is False

        # Approved verdict: the judge fulfilled the future — run progresses.
        approved = self._ctx_with_events([request, ApprovalReviewed(request_id="r1", approved=True)])
        assert _pending_user_approval(approved) is False

        # Flagged verdict (incl. judge errors): _run_judge leaves the future
        # unresolved for the user, so the request pends on the test again.
        flagged = self._ctx_with_events([request, ApprovalReviewed(request_id="r1", approved=False)])
        assert _pending_user_approval(flagged) is True

        # A verdict for a different request does not re-block this one.
        other = self._ctx_with_events([request, ApprovalReviewed(request_id="r9", approved=False)])
        assert _pending_user_approval(other) is False

    def test_responded_and_interrupted_requests_do_not_pend(self):
        from chrys.foundation.events.types import ApprovalResponse, UserInterrupt
        from tests.support.pipeline_helpers import _pending_user_approval

        responded = self._ctx_with_events(
            [
                ApprovalRequest(request_id="r1", tool_name="guarded_echo"),
                ApprovalResponse(request_id="r1", approved=True),
            ]
        )
        assert _pending_user_approval(responded) is False

        interrupted = self._ctx_with_events(
            [ApprovalRequest(request_id="r2", tool_name="guarded_echo"), UserInterrupt()]
        )
        assert _pending_user_approval(interrupted) is False

        # A request published after the interrupt (e.g. the resumed run) pends.
        resumed = self._ctx_with_events([UserInterrupt(), ApprovalRequest(request_id="r3", tool_name="guarded_echo")])
        assert _pending_user_approval(resumed) is True


# ---------------------------------------------------------------------------
# S5: User approval — approve and reject
# ---------------------------------------------------------------------------


class TestUserApproval:
    """guarded_echo with override require — user must approve/reject via ApprovalMiddleware."""

    @pytest.mark.asyncio
    async def test_user_approve(self, tmp_path):
        """User approves the tool → tool executes."""
        responses = [
            # LLM calls guarded_echo → ApprovalMiddleware pauses for approval
            MockResponse(tool_calls=[("guarded_echo", "u1", {"message": "please approve"})]),
            # After approval + tool execution, LLM responds with text
            MockResponse(text="Approved and done"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_overrides={"guarded_echo": "require"})

        try:
            await ctx.bus.publish(UserMessage(text="Run guarded tool"))

            approval_events = await wait_for_event(ctx.events, ApprovalRequest, timeout=20.0)
            assert len(approval_events) >= 1
            request_id = approval_events[0].request_id

            # Approve
            await ctx.approve(request_id)
            await wait_for_idle(ctx)

            finals = extract_final_messages(ctx.events)
            assert len(finals) >= 1

            # Session should show user_approved
            raw = await ctx.get_session_messages()
            approvals = extract_session_approvals(raw)
            user_approved = [a for a in approvals if a is not None and a.get("status") == "user_approved"]
            assert len(user_approved) >= 1
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_user_reject(self, tmp_path):
        """User rejects the tool → LLM gets rejection error, responds with text."""
        responses = [
            MockResponse(tool_calls=[("guarded_echo", "r1", {"message": "reject me"})]),
            # After rejection (error result), LLM responds with text
            MockResponse(text="OK, I won't run that tool."),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_overrides={"guarded_echo": "require"})

        try:
            await ctx.bus.publish(UserMessage(text="Run guarded tool"))

            approval_events = await wait_for_event(ctx.events, ApprovalRequest, timeout=20.0)
            request_id = approval_events[0].request_id

            # Reject with a reason so the model can steer its next response.
            rejection_reason = "Use a safer command"
            await ctx.reject(request_id, reason=rejection_reason)
            await wait_for_idle(ctx)

            finals = extract_final_messages(ctx.events)
            assert len(finals) >= 1

            # Rejected tool should still produce ToolCallStart/Result events
            # (middleware order: tool_events wraps approval)
            starts = extract_tool_starts(ctx.events)
            assert any(s["tool_name"] == "guarded_echo" for s in starts)
            results = extract_tool_results(ctx.events)
            rejected_results = [r for r in results if "rejected" in r["result"].lower()]
            assert len(rejected_results) >= 1
            assert any(rejection_reason in r["result"] for r in rejected_results)

            # Session should show user_rejected and preserve the reason.
            raw = await ctx.get_session_messages()
            approvals = extract_session_approvals(raw)
            user_rejected = [a for a in approvals if a is not None and a.get("status") == "user_rejected"]
            assert len(user_rejected) >= 1
            assert any(a.get("reason") == rejection_reason for a in user_rejected)
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_reused_provider_call_id_still_requires_each_approval(self, tmp_path) -> None:
        """Provider call IDs are not approval identities; every invocation gates independently."""
        responses = [
            MockResponse(tool_calls=[("guarded_echo", "reused-id", {"message": "first"})]),
            MockResponse(tool_calls=[("guarded_echo", "reused-id", {"message": "second"})]),
            MockResponse(text="Both calls approved"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_overrides={"guarded_echo": "require"})
        try:
            await ctx.bus.publish(UserMessage(text="Run both guarded calls"))

            first_events = await wait_for_event(ctx.events, ApprovalRequest, timeout=20.0)
            await ctx.approve(first_events[0].request_id)

            both_events = await wait_for_event(ctx.events, ApprovalRequest, timeout=20.0, min_count=2)
            assert both_events[0].request_id != both_events[1].request_id
            await ctx.approve(both_events[1].request_id)
            await wait_for_idle(ctx)

            guarded_results = [
                result for result in extract_tool_results(ctx.events) if result["tool_name"] == "guarded_echo"
            ]
            assert len(guarded_results) == 2
        finally:
            await ctx.cleanup()


# ---------------------------------------------------------------------------
# S5b: Interrupt during approval wait
# ---------------------------------------------------------------------------


class TestInterruptDuringApproval:
    """Interrupt while ApprovalMiddleware is waiting for user response."""

    @pytest.mark.asyncio
    async def test_interrupt_during_approval_preserves_prior_work(self, tmp_path):
        """Interrupt while waiting for approval: prior tools preserved, no crash."""
        responses = [
            # LLM calls echo (auto) then guarded_echo (requires approval)
            MockResponse(tool_calls=[("echo", "e1", {"message": "before"})]),
            MockResponse(tool_calls=[("guarded_echo", "g1", {"message": "needs approval"})]),
            # Resume responses
            MockResponse(tool_calls=[("echo", "e2", {"message": "resumed"})]),
            MockResponse(text="Resumed OK"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_overrides={"guarded_echo": "require"})

        try:
            await ctx.bus.publish(UserMessage(text="Run tools"))

            # Wait for the approval request to appear
            approval_events = await wait_for_event(ctx.events, ApprovalRequest, timeout=20.0)
            assert len(approval_events) >= 1

            # Interrupt instead of approving/rejecting
            await ctx.send_interrupt()
            await wait_for_idle(ctx)

            # Verify prior echo tool is preserved in session
            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            assert "echo" in tool_names

            # Resume after interrupt
            ctx.events.clear()
            await ctx.send_retry()

            finals = extract_final_messages(ctx.events)
            assert len(finals) >= 1
        finally:
            await ctx.cleanup()


# ---------------------------------------------------------------------------
# S6: Interrupt + resume
# ---------------------------------------------------------------------------


class TestInterruptAndResume:
    """Interrupt after first tool completes, then resume."""

    @pytest.mark.asyncio
    async def test_interrupt_preserves_completed_work(self, tmp_path):
        """Interrupt mid-run: completed tools preserved, engine can resume."""
        from chrys.kernel import FunctionTool

        async def _slow_echo(message: str) -> str:
            await asyncio.sleep(0.2)
            return f"echo: {message}"

        slow_echo = FunctionTool(func=_slow_echo, name="echo", description="Slow echo")

        # 3 sequential tool calls — we'll interrupt after the 1st
        responses = [
            MockResponse(tool_calls=[("echo", "i1", {"message": "first"})]),
            MockResponse(tool_calls=[("echo", "i2", {"message": "second"})]),
            MockResponse(tool_calls=[("echo", "i3", {"message": "third"})]),
            MockResponse(text="All three done"),
            # Resume responses: continue from where we left off
            MockResponse(tool_calls=[("echo", "i4", {"message": "resumed"})]),
            MockResponse(text="Resumed and done"),
        ]
        ctx = await create_test_engine(responses, tmp_path, tools=[slow_echo])

        try:
            await ctx.bus.publish(UserMessage(text="Run three tools"))

            # Wait for the second tool to START — this proves the framework
            # has committed the first tool's result to message history.
            # (ToolCallResult fires from middleware BEFORE the framework
            # appends the result to messages, so waiting on ToolCallResult
            # alone is racy on macOS CI.)
            await wait_for_event(ctx.events, ToolCallStart, timeout=20.0, min_count=2)

            # Interrupt
            await ctx.send_interrupt()
            await wait_for_idle(ctx)

            # Verify completed work preserved in session
            raw_after_interrupt = await ctx.get_session_messages()
            tool_names_after = extract_session_tool_names(raw_after_interrupt)
            assert len(tool_names_after) >= 1
            assert "echo" in tool_names_after

            # Verify batch_ids exist on completed work
            batch_ids_after = extract_session_batch_ids(raw_after_interrupt)
            completed_bids = [b for b in batch_ids_after if b is not None]
            assert len(completed_bids) >= 1
            pre_resume_max = max(completed_bids)

            # Resume
            ctx.events.clear()
            await ctx.send_retry()

            # Verify resumed work completes
            finals = extract_final_messages(ctx.events)
            assert len(finals) >= 1

            # Verify batch_ids don't collide after resume
            raw_after_resume = await ctx.get_session_messages()
            all_batch_ids = extract_session_batch_ids(raw_after_resume)
            non_none_bids = [b for b in all_batch_ids if b is not None]
            post_resume_bids = non_none_bids[len(completed_bids) :]
            if post_resume_bids:
                assert all(b > pre_resume_max for b in post_resume_bids), (
                    f"Resume batch_ids {post_resume_bids} should be > pre-interrupt max {pre_resume_max}"
                )
        finally:
            await ctx.cleanup()


# ---------------------------------------------------------------------------
# S6b: Interrupt preserves completed iterations in multi-iteration tool loop
# ---------------------------------------------------------------------------


class TestInterruptMultiIteration:
    """When interrupt fires mid-tool-loop, completed iterations must survive."""

    @pytest.mark.asyncio
    async def test_completed_iterations_preserved_on_interrupt(self, tmp_path):
        """Multi-iteration tool loop: interrupt after 2 iterations, 1st preserved."""
        import asyncio

        from chrys.kernel import FunctionTool

        # Async tool that sleeps long enough for the interrupt to land
        # between iterations.  The ToolCallResult event fires when the
        # middleware sees the result, but the framework may not have
        # committed it to recoverable state (LoopRecorder) until
        # the *next* LLM call starts.  The sleep in the tool + the
        # settle delay after the event give the framework enough time
        # to enter the next iteration and snapshot the messages.
        async def _slow_echo(message: str) -> str:
            await asyncio.sleep(0.1)
            return f"echo: {message}"

        slow_echo = FunctionTool(func=_slow_echo, name="slow_echo", description="Slow echo")

        # 3 sequential iterations + final text.
        # Interrupt should fire after iteration 0 completes.
        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            MockResponse(tool_calls=[("slow_echo", "c2", {"message": "second"})]),
            MockResponse(tool_calls=[("slow_echo", "c3", {"message": "third"})]),
            MockResponse(text="All done"),
            # Resume responses
            MockResponse(tool_calls=[("slow_echo", "c4", {"message": "resumed"})]),
            MockResponse(text="Resume done"),
        ]
        ctx = await create_test_engine(responses, tmp_path, tools=[slow_echo])

        try:
            # Start the run
            await ctx.bus.publish(UserMessage(text="Run three tools"))

            # Wait for first tool result, then for the framework to enter the
            # next tool loop iteration so LoopRecorder snapshots the
            # completed iteration's messages into recoverable state.  Polling
            # the recorder's snapshot (instead of a fixed sleep) avoids racing
            # the background checkpoint on slow CI.
            await wait_for_event(ctx.events, ToolCallResult, timeout=20.0, min_count=1)
            await _wait_for_loop_snapshot(ctx, min_iterations=1)

            # Interrupt — will fire before the next tool executes
            await ctx.send_interrupt()
            await wait_for_idle(ctx)

            # Session must contain the completed tool call from iteration 0
            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            assert "slow_echo" in tool_names, f"Completed tool call lost on interrupt. Session tools: {tool_names}"

            # Verify at least 1 function_result has actual content (not None)
            has_real_result = False
            for msg in raw:
                for c in msg.get("contents", []):
                    if isinstance(c, dict) and c.get("type") == "function_result":
                        result = c.get("result", "")
                        if result and "echo:" in result:
                            has_real_result = True
            assert has_real_result, "No completed tool results found in session"

            # Resume and verify it works
            ctx.events.clear()
            await ctx.send_retry()
            finals = extract_final_messages(ctx.events)
            assert len(finals) >= 1

        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_multiple_completed_iterations_preserved(self, tmp_path):
        """Interrupt after 2+ iterations: all completed iterations preserved."""
        import asyncio

        from chrys.kernel import FunctionTool

        async def _slow_echo(message: str) -> str:
            await asyncio.sleep(0.1)
            return f"echo: {message}"

        slow_echo = FunctionTool(func=_slow_echo, name="slow_echo", description="Slow echo")

        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            MockResponse(tool_calls=[("slow_echo", "c2", {"message": "second"})]),
            MockResponse(tool_calls=[("slow_echo", "c3", {"message": "third"})]),
            MockResponse(tool_calls=[("slow_echo", "c4", {"message": "fourth"})]),
            MockResponse(text="All done"),
            # Resume
            MockResponse(text="Resume done"),
        ]
        ctx = await create_test_engine(responses, tmp_path, tools=[slow_echo])

        try:
            await ctx.bus.publish(UserMessage(text="Run four tools"))

            # Wait for 2 tool results, then for the LoopRecorder to snapshot
            # both completed iterations into recoverable state.  Polling the
            # recorder's snapshot replaces a fixed settle that raced on CI.
            await wait_for_event(ctx.events, ToolCallResult, timeout=20.0, min_count=2)
            await _wait_for_loop_snapshot(ctx, min_iterations=2)

            await ctx.send_interrupt()
            await wait_for_idle(ctx)

            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)

            # At least the first 2 completed iterations should be preserved
            assert len(tool_names) >= 2, f"Expected >= 2 completed tools, got {len(tool_names)}: {tool_names}"

            # Verify results are real (not None from interrupt)
            real_results = []
            for msg in raw:
                for c in msg.get("contents", []):
                    if isinstance(c, dict) and c.get("type") == "function_result":
                        result = c.get("result", "")
                        if result and "echo:" in result:
                            real_results.append(result)
            assert len(real_results) >= 2, f"Expected >= 2 real results, got {len(real_results)}: {real_results}"

        finally:
            await ctx.cleanup()


# ---------------------------------------------------------------------------
# S6c: Error during tool loop + retry preserves completed work
# ---------------------------------------------------------------------------


class TestErrorAndRetry:
    """Error during multi-iteration tool loop: completed work survives, retry sees it."""

    @pytest.mark.asyncio
    async def test_error_preserves_completed_iterations(self, tmp_path):
        """Network error during LLM call: completed iterations preserved in session."""
        import asyncio

        from chrys.kernel import FunctionTool

        async def _slow_echo(message: str) -> str:
            await asyncio.sleep(0)
            return f"echo: {message}"

        slow_echo = FunctionTool(func=_slow_echo, name="slow_echo", description="Slow echo")

        # 2 iterations succeed, then error on 3rd LLM call.
        # Retry responses follow.
        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            MockResponse(tool_calls=[("slow_echo", "c2", {"message": "second"})]),
            # 3rd LLM call will throw (patched below)
            # After retry:
            MockResponse(tool_calls=[("slow_echo", "c3", {"message": "third"})]),
            MockResponse(text="All done after retry"),
        ]
        ctx = await create_test_engine(responses, tmp_path, tools=[slow_echo])

        try:
            # Patch mock client to throw on the 3rd LLM call
            original_inner = ctx.mock_client._inner_get_response
            call_count = [0]

            def _error_on_third(*, messages, stream, options, **kwargs):
                call_count[0] += 1
                if call_count[0] == 3:

                    async def _throw():
                        raise RuntimeError("Simulated non-retryable error")

                    return _throw()
                return original_inner(messages=messages, stream=stream, options=options, **kwargs)

            ctx.mock_client._inner_get_response = _error_on_third

            # Run — should fail on 3rd LLM call after 2 iterations complete
            await ctx.send_message("Run three tools")

            # Engine should have set run_failed
            assert ctx.engine._executor.run_failed, "Expected run_failed after non-retryable error"

            # Session must have both completed iterations
            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            assert len(tool_names) >= 2, f"Expected >= 2 tools preserved after error, got {tool_names}"

            # Verify results are real (not empty)
            real_results = []
            for msg in raw:
                for c in msg.get("contents", []):
                    if isinstance(c, dict) and c.get("type") == "function_result":
                        result = c.get("result", "")
                        if result and "echo:" in result:
                            real_results.append(result)
            assert len(real_results) >= 2, f"Expected >= 2 real results, got {real_results}"

        finally:
            ctx.mock_client._inner_get_response = original_inner
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_retry_after_error_sends_preserved_work_to_llm(self, tmp_path):
        """On retry after error, LLM receives previously completed tool results."""
        import asyncio

        from chrys.kernel import FunctionTool

        async def _slow_echo(message: str) -> str:
            await asyncio.sleep(0)
            return f"echo: {message}"

        slow_echo = FunctionTool(func=_slow_echo, name="slow_echo", description="Slow echo")

        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            MockResponse(tool_calls=[("slow_echo", "c2", {"message": "second"})]),
            # 3rd call errors
            # Retry:
            MockResponse(tool_calls=[("slow_echo", "c3", {"message": "third"})]),
            MockResponse(text="Retry complete"),
        ]
        ctx = await create_test_engine(responses, tmp_path, tools=[slow_echo])

        try:
            original_inner = ctx.mock_client._inner_get_response
            call_count = [0]

            def _error_on_third(*, messages, stream, options, **kwargs):
                call_count[0] += 1
                if call_count[0] == 3:

                    async def _throw():
                        raise RuntimeError("Simulated non-retryable error")

                    return _throw()
                return original_inner(messages=messages, stream=stream, options=options, **kwargs)

            ctx.mock_client._inner_get_response = _error_on_third

            await ctx.send_message("Run tools")
            assert ctx.engine._executor.run_failed

            # Restore normal client for retry
            ctx.mock_client._inner_get_response = original_inner

            # Retry
            ctx.events.clear()
            await ctx.send_retry()

            # Verify retry completes
            finals = extract_final_messages(ctx.events)
            assert len(finals) >= 1, "Retry should produce a final message"

            # Verify the LLM received preserved work on retry.
            # The FIRST LLM call of the retry should contain function_result
            # contents from the completed iterations.
            retry_first_call_msgs = ctx.mock_client.call_history[-2][0]  # -2 = first retry LLM call
            preserved_results = []
            for msg in retry_first_call_msgs:
                for c in getattr(msg, "contents", []):
                    if getattr(c, "type", "") == "function_result":
                        result = getattr(c, "result", "") or ""
                        if "echo:" in result:
                            preserved_results.append(result)
            assert len(preserved_results) >= 2, (
                f"LLM should see >= 2 preserved results on retry, got {preserved_results}"
            )

        finally:
            ctx.mock_client._inner_get_response = original_inner
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_retry_after_error_completes_full_session(self, tmp_path):
        """Full round-trip: error → retry → final session has all work."""
        import asyncio

        from chrys.kernel import FunctionTool

        async def _slow_echo(message: str) -> str:
            await asyncio.sleep(0)
            return f"echo: {message}"

        slow_echo = FunctionTool(func=_slow_echo, name="slow_echo", description="Slow echo")

        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            MockResponse(tool_calls=[("slow_echo", "c2", {"message": "second"})]),
            # 3rd call errors
            # Retry:
            MockResponse(tool_calls=[("slow_echo", "c3", {"message": "third"})]),
            MockResponse(text="All done"),
        ]
        ctx = await create_test_engine(responses, tmp_path, tools=[slow_echo])

        try:
            original_inner = ctx.mock_client._inner_get_response
            call_count = [0]

            def _error_on_third(*, messages, stream, options, **kwargs):
                call_count[0] += 1
                if call_count[0] == 3:

                    async def _throw():
                        raise RuntimeError("Simulated non-retryable error")

                    return _throw()
                return original_inner(messages=messages, stream=stream, options=options, **kwargs)

            ctx.mock_client._inner_get_response = _error_on_third
            await ctx.send_message("Run tools")
            assert ctx.engine._executor.run_failed

            ctx.mock_client._inner_get_response = original_inner
            ctx.events.clear()
            await ctx.send_retry()

            # Final session should have original user message + all tool calls + final text
            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            # Should have at least the 2 from before error + 1 from retry = 3
            assert len(tool_names) >= 3, f"Expected >= 3 tools in final session, got {tool_names}"

            # Verify final text is present
            has_final = any(
                msg.get("role") == "assistant"
                and any(
                    isinstance(c, dict) and c.get("type") == "text" and "All done" in (c.get("text") or "")
                    for c in msg.get("contents", [])
                )
                for msg in raw
            )
            assert has_final, "Final text 'All done' not found in session"

        finally:
            ctx.mock_client._inner_get_response = original_inner
            await ctx.cleanup()


# ---------------------------------------------------------------------------
# S6d: Interrupt during final LLM response drops leaked text
# ---------------------------------------------------------------------------


class TestInterruptDuringFinalResponse:
    """Interrupt during the LLM's final text response (no tool calls).

    The InterruptMiddleware only fires at tool call boundaries.  When the LLM
    is generating its final text-only response, the interrupt flag is set but
    the response completes and after_run stores it.  The engine must detect
    and remove this leaked response from session state.
    """

    @pytest.mark.asyncio
    async def test_interrupt_during_final_response_drops_text(self, tmp_path):
        """Interrupt during final LLM call: text response not in session."""
        import asyncio

        from chrys.kernel import FunctionTool

        async def _slow_echo(message: str) -> str:
            await asyncio.sleep(0)
            return f"echo: {message}"

        slow_echo = FunctionTool(func=_slow_echo, name="slow_echo", description="Slow echo")

        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            # 2nd LLM call returns text-only (final response).
            # Interrupt will be set during this call (see monkey-patch below).
            MockResponse(text="This is the final answer that should be dropped"),
            # Resume responses
            MockResponse(text="Resumed answer"),
        ]
        ctx = await create_test_engine(responses, tmp_path, tools=[slow_echo])

        try:
            # Monkey-patch mock client to set interrupt flag during the 2nd
            # LLM call, simulating the user pressing Stop while the LLM is
            # generating its final text response.
            original_inner = ctx.mock_client._inner_get_response
            call_count = [0]
            interrupt_mw = ctx.engine._executor._interrupt

            def _interrupt_on_second(*, messages, stream, options, **kwargs):
                call_count[0] += 1
                if call_count[0] == 2:
                    # Set interrupt flag BEFORE returning the response.
                    # This simulates: LLM call in flight, user presses Stop,
                    # response arrives anyway.
                    interrupt_mw.set_interrupted()
                return original_inner(messages=messages, stream=stream, options=options, **kwargs)

            ctx.mock_client._inner_get_response = _interrupt_on_second

            await ctx.send_message("Run one tool then respond")

            # Engine should have detected the interrupt
            assert ctx.engine._executor.was_interrupted, (
                "Expected was_interrupted after interrupt during final response"
            )

            # Session must NOT contain the leaked final text
            raw = await ctx.get_session_messages()
            for msg in raw:
                for c in msg.get("contents", []):
                    if isinstance(c, dict) and c.get("type") == "text":
                        text = c.get("text", "")
                        assert "final answer that should be dropped" not in text, (
                            f"Leaked final response found in session: {text}"
                        )

            # Session should still have the completed tool call
            tool_names = extract_session_tool_names(raw)
            assert "slow_echo" in tool_names, f"Completed tool call should be preserved: {tool_names}"

        finally:
            ctx.mock_client._inner_get_response = original_inner
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_interrupt_before_any_tools_drops_text(self, tmp_path):
        """Interrupt during first LLM call (text-only, no tools): session has only user msg."""
        responses = [
            # LLM returns text-only on first call (no tools needed).
            # Interrupt set during this call.
            MockResponse(text="Text response that should be dropped"),
            # Resume
            MockResponse(text="Resumed"),
        ]
        ctx = await create_test_engine(responses, tmp_path)

        try:
            original_inner = ctx.mock_client._inner_get_response
            call_count = [0]
            interrupt_mw = ctx.engine._executor._interrupt

            def _interrupt_on_first(*, messages, stream, options, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    interrupt_mw.set_interrupted()
                return original_inner(messages=messages, stream=stream, options=options, **kwargs)

            ctx.mock_client._inner_get_response = _interrupt_on_first

            await ctx.send_message("Just explain something")

            assert ctx.engine._executor.was_interrupted

            raw = await ctx.get_session_messages()
            # Should have user message + interrupted marker + turn marker, but NO assistant text
            for msg in raw:
                role = msg.get("role", "")
                if role == "assistant":
                    extra = msg.get("additional_properties", {})
                    chrys_kind = extra.get(HistoryMarkerKind.KEY, "")
                    # Only chrys markers (interrupted, turn_marker) allowed
                    assert chrys_kind in (HistoryMarkerKind.INTERRUPTED, HistoryMarkerKind.TURN), (
                        f"Unexpected assistant message in session: contents={msg.get('contents')}"
                    )

        finally:
            ctx.mock_client._inner_get_response = original_inner
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_normal_completion_preserves_final_text(self, tmp_path):
        """Regression: normal completion (no interrupt) keeps the final text."""
        import asyncio

        from chrys.kernel import FunctionTool

        async def _slow_echo(message: str) -> str:
            await asyncio.sleep(0)
            return f"echo: {message}"

        slow_echo = FunctionTool(func=_slow_echo, name="slow_echo", description="Slow echo")

        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            MockResponse(text="Final answer preserved"),
        ]
        ctx = await create_test_engine(responses, tmp_path, tools=[slow_echo])

        try:
            await ctx.send_message("Run tool and respond")

            # Should NOT be interrupted
            assert not ctx.engine._executor.was_interrupted

            # Final text should be in session
            raw = await ctx.get_session_messages()
            has_final_text = any(
                msg.get("role") == "assistant"
                and any(
                    isinstance(c, dict)
                    and c.get("type") == "text"
                    and "Final answer preserved" in (c.get("text") or "")
                    for c in msg.get("contents", [])
                )
                for msg in raw
            )
            assert has_final_text, "Final text should be preserved on normal completion"

        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_resume_after_interrupt_during_final_response(self, tmp_path):
        """Interrupt during final LLM call, then resume: session ends with new response."""
        import asyncio

        from chrys.kernel import FunctionTool

        async def _slow_echo(message: str) -> str:
            await asyncio.sleep(0)
            return f"echo: {message}"

        slow_echo = FunctionTool(func=_slow_echo, name="slow_echo", description="Slow echo")

        responses = [
            MockResponse(tool_calls=[("slow_echo", "c1", {"message": "first"})]),
            # Final response — interrupt fires during this LLM call
            MockResponse(text="Leaked answer"),
            # Resume responses
            MockResponse(text="Correct resumed answer"),
        ]
        ctx = await create_test_engine(responses, tmp_path, tools=[slow_echo])

        try:
            original_inner = ctx.mock_client._inner_get_response
            call_count = [0]
            interrupt_mw = ctx.engine._executor._interrupt

            def _interrupt_on_second(*, messages, stream, options, **kwargs):
                call_count[0] += 1
                if call_count[0] == 2:
                    interrupt_mw.set_interrupted()
                return original_inner(messages=messages, stream=stream, options=options, **kwargs)

            ctx.mock_client._inner_get_response = _interrupt_on_second
            await ctx.send_message("Run tool then respond")

            assert ctx.engine._executor.was_interrupted

            # Resume
            ctx.mock_client._inner_get_response = original_inner
            ctx.events.clear()
            await ctx.send_retry()

            # Verify the resumed response is in session
            raw = await ctx.get_session_messages()
            has_resumed = any(
                msg.get("role") == "assistant"
                and any(
                    isinstance(c, dict)
                    and c.get("type") == "text"
                    and "Correct resumed answer" in (c.get("text") or "")
                    for c in msg.get("contents", [])
                )
                for msg in raw
            )
            assert has_resumed, "Resumed answer should be in session"

            # Leaked answer should NOT be present
            has_leaked = any(
                msg.get("role") == "assistant"
                and any(
                    isinstance(c, dict) and c.get("type") == "text" and "Leaked answer" in (c.get("text") or "")
                    for c in msg.get("contents", [])
                )
                for msg in raw
            )
            assert not has_leaked, "Leaked answer from interrupted run should not be in session"

        finally:
            ctx.mock_client._inner_get_response = original_inner
            await ctx.cleanup()


# ---------------------------------------------------------------------------
# S6e: Resume appends to existing state (never drops content)
# ---------------------------------------------------------------------------


class TestResumeAppendsToState:
    """On resume, completed work is preserved and the LLM continues.

    Session history is append-only: only chrys markers (interrupted,
    turn_marker) are removed.  Completed tool calls, user messages, and
    assistant messages are never deleted.
    """

    @pytest.mark.asyncio
    async def test_resume_preserves_completed_tools(self, tmp_path):
        """Interrupt during final LLM call: completed tool preserved on resume."""
        import asyncio

        from chrys.kernel import FunctionTool

        async def _fast_tool(message: str) -> str:
            await asyncio.sleep(0)
            return f"result: {message}"

        fast_tool = FunctionTool(func=_fast_tool, name="fast_tool", description="Fast tool")

        responses = [
            MockResponse(tool_calls=[("fast_tool", "c1", {"message": "hello"})]),
            # Final response — interrupt fires here
            MockResponse(text="Interrupted final"),
            # Resume: LLM sees completed tool result, generates new response
            MockResponse(text="Resumed with context"),
        ]
        ctx = await create_test_engine(responses, tmp_path, tools=[fast_tool])

        try:
            original_inner = ctx.mock_client._inner_get_response
            call_count = [0]
            interrupt_mw = ctx.engine._executor._interrupt

            def _interrupt_on_second(*, messages, stream, options, **kwargs):
                call_count[0] += 1
                if call_count[0] == 2:
                    interrupt_mw.set_interrupted()
                return original_inner(messages=messages, stream=stream, options=options, **kwargs)

            ctx.mock_client._inner_get_response = _interrupt_on_second
            await ctx.send_message("Do something")

            assert ctx.engine._executor.was_interrupted

            # Completed tool should be in session
            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            assert "fast_tool" in tool_names, "Completed tool should be preserved"

            # Resume — LLM continues from completed work
            ctx.mock_client._inner_get_response = original_inner
            ctx.events.clear()
            await ctx.send_retry()

            finals = extract_final_messages(ctx.events)
            assert any("Resumed" in t for t in finals)

            # Session still has the completed tool AND the new response
            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            assert "fast_tool" in tool_names, "Completed tool preserved after resume"

        finally:
            ctx.mock_client._inner_get_response = original_inner
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_resume_text_only_resends_user_message(self, tmp_path):
        """Interrupt during text-only response: resume re-sends user message."""
        responses = [
            MockResponse(text="Will be interrupted"),
            MockResponse(text="Fresh response"),
        ]
        ctx = await create_test_engine(responses, tmp_path)

        try:
            original_inner = ctx.mock_client._inner_get_response
            interrupt_mw = ctx.engine._executor._interrupt

            def _interrupt_on_first(*, messages, stream, options, **kwargs):
                interrupt_mw.set_interrupted()
                return original_inner(messages=messages, stream=stream, options=options, **kwargs)

            ctx.mock_client._inner_get_response = _interrupt_on_first
            await ctx.send_message("Say something")

            assert ctx.engine._executor.was_interrupted

            ctx.mock_client._inner_get_response = original_inner
            ctx.events.clear()
            await ctx.send_retry()

            finals = extract_final_messages(ctx.events)
            assert any("Fresh" in t for t in finals), "Should get fresh response on resume"

        finally:
            ctx.mock_client._inner_get_response = original_inner
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_new_message_after_interrupt_keeps_context(self, tmp_path):
        """New message after interrupt preserves all previous context."""
        import asyncio

        from chrys.kernel import FunctionTool

        async def _echo(message: str) -> str:
            await asyncio.sleep(0)
            return f"echo: {message}"

        echo_tool = FunctionTool(func=_echo, name="echo", description="Echo")

        responses = [
            MockResponse(tool_calls=[("echo", "c1", {"message": "first"})]),
            # Interrupt fires during this response
            MockResponse(text="Interrupted"),
            # New message gets this response
            MockResponse(text="Got your follow-up"),
        ]
        ctx = await create_test_engine(responses, tmp_path, tools=[echo_tool])

        try:
            original_inner = ctx.mock_client._inner_get_response
            call_count = [0]
            interrupt_mw = ctx.engine._executor._interrupt

            def _interrupt_on_second(*, messages, stream, options, **kwargs):
                call_count[0] += 1
                if call_count[0] == 2:
                    interrupt_mw.set_interrupted()
                return original_inner(messages=messages, stream=stream, options=options, **kwargs)

            ctx.mock_client._inner_get_response = _interrupt_on_second
            await ctx.send_message("Do something")

            assert ctx.engine._executor.was_interrupted

            # Send a NEW message (not resume)
            ctx.mock_client._inner_get_response = original_inner
            ctx.events.clear()
            await ctx.send_message("Continue please")

            # Session should have: original user msg + completed tool + new user msg + response
            raw = await ctx.get_session_messages()
            user_msgs = [
                m
                for m in raw
                if m.get("role") == "user" and not m.get("additional_properties", {}).get(HistoryMarkerKind.KEY)
            ]
            assert len(user_msgs) >= 2, f"Both user messages should be preserved, got {len(user_msgs)}"

            # Completed tool from first run should still be there
            tool_names = extract_session_tool_names(raw)
            assert "echo" in tool_names, "Completed tool from interrupted run preserved"

        finally:
            ctx.mock_client._inner_get_response = original_inner
            await ctx.cleanup()


# ---------------------------------------------------------------------------
# S7: Three-way consistency (events ↔ session ↔ replay)
# ---------------------------------------------------------------------------


class TestThreeWayConsistency:
    """Verify events, session state, and replay all agree on tool call structure."""

    @pytest.mark.asyncio
    async def test_sequential_consistency(self, tmp_path):
        """Sequential tool calls: events and session match, distinct batch_ids."""
        responses = [
            MockResponse(tool_calls=[("echo", "s1", {"message": "a"})]),
            MockResponse(tool_calls=[("concat", "s2", {"a": "b", "b": "c"})]),
            MockResponse(text="Sequential done"),
        ]
        ctx = await create_test_engine(responses, tmp_path)

        try:
            await ctx.send_message("Run sequential tools")

            # 1. Event tool names
            event_tools = [s["tool_name"] for s in extract_tool_starts(ctx.events)]

            # 2. Session tool names
            raw = await ctx.get_session_messages()
            session_tools = extract_session_tool_names(raw)

            # Events and session match
            assert event_tools == ["echo", "concat"]
            assert session_tools == ["echo", "concat"]

            # Distinct batch_ids for sequential tools
            batch_ids = extract_session_batch_ids(raw)
            assert all(b is not None for b in batch_ids)
            assert batch_ids[0] != batch_ids[1]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_parallel_consistency(self, tmp_path):
        """Parallel tool calls: events and session match, shared batch_id."""
        responses = [
            MockResponse(
                tool_calls=[
                    ("echo", "p1", {"message": "x"}),
                    ("uppercase", "p2", {"text": "y"}),
                ]
            ),
            MockResponse(text="Parallel done"),
        ]
        ctx = await create_test_engine(responses, tmp_path)

        try:
            await ctx.send_message("Run parallel tools")

            event_tools = [s["tool_name"] for s in extract_tool_starts(ctx.events)]
            raw = await ctx.get_session_messages()
            session_tools = extract_session_tool_names(raw)

            assert set(event_tools) == {"echo", "uppercase"}
            assert set(session_tools) == {"echo", "uppercase"}

            # Parallel tools share one batch_id
            batch_ids = extract_session_batch_ids(raw)
            assert len(batch_ids) == 1
            assert batch_ids[0] is not None
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_replay_groups_match_batch_ids(self, tmp_path):
        """Replay produces one merged tool group when no intermediate text separates batches."""
        responses = [
            MockResponse(tool_calls=[("echo", "r1", {"message": "one"})]),
            MockResponse(
                tool_calls=[
                    ("concat", "r2", {"a": "a", "b": "b"}),
                    ("uppercase", "r3", {"text": "test"}),
                ]
            ),
            MockResponse(text="Done with all tools"),
        ]
        ctx = await create_test_engine(responses, tmp_path)

        try:
            await ctx.send_message("Run mixed tools")

            raw = await ctx.get_session_messages()

            # Verify session structure: all 3 tools present
            session_tools = extract_session_tool_names(raw)
            assert session_tools == ["echo", "concat", "uppercase"]

            # echo has batch_id=1, concat+uppercase have batch_id=2
            batch_ids = extract_session_batch_ids(raw)
            assert len(batch_ids) == 2
            assert batch_ids[0] != batch_ids[1]

            # Run replay through ChatPanel
            replay_result = await _run_replay(raw)

            # No intermediate text separates the two batches, so replay
            # merges them into one 3-tool group.
            assert replay_result["user_count"] == 1
            assert replay_result["tool_group_count"] == 1
            assert replay_result["tool_group_sizes"] == [3]
            assert replay_result["agent_message_count"] >= 1

        finally:
            await ctx.cleanup()


# ---------------------------------------------------------------------------
# Replay helper (minimal Textual app)
# ---------------------------------------------------------------------------


async def _run_replay(raw_messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Run replay_history in a headless Textual app and inspect widget tree.

    Returns:
        Dict with: user_count, tool_group_count, tool_group_sizes,
        agent_message_count.
    """
    from textual.app import App, ComposeResult

    from chrys.app.tui.theme import TuiVariableDefaultsMixin
    from chrys.app.tui.widgets.chat.panel import ChatPanel

    class ReplayApp(TuiVariableDefaultsMixin, App):
        def compose(self) -> ComposeResult:
            yield ChatPanel()

    async with ReplayApp().run_test() as pilot:
        cp = pilot.app.query_one(ChatPanel)
        await cp.replay_history(raw_messages)
        await pilot.pause()

        from chrys.app.tui.widgets.chat.messages import AgentMessage as AgentMessageWidget
        from chrys.app.tui.widgets.chat.messages import UserMessage as UserMessageWidget
        from chrys.app.tui.widgets.chat.tool_call import ToolGroup

        user_msgs = cp.query(UserMessageWidget)
        tool_groups = cp.query(ToolGroup)
        agent_msgs = cp.query(AgentMessageWidget)

        tool_group_sizes = []
        for tg in tool_groups:
            tool_group_sizes.append(len(tg._tool_records))

        return {
            "user_count": len(user_msgs),
            "tool_group_count": len(tool_groups),
            "tool_group_sizes": tool_group_sizes,
            "agent_message_count": len(agent_msgs),
        }


# ---------------------------------------------------------------------------
# S8: Approval preserves intermediate tool calls
# ---------------------------------------------------------------------------


class TestApprovalPreservesIntermediateTools:
    """When a non-guarded tool (echo) runs before a guarded tool (guarded_echo),
    the approval re-submission loop must preserve the intermediate echo call
    in the session state with correct ordering:
    user → echo_call → echo_result → guarded_call → guarded_result → final_text.

    With inline approval via ``ApprovalMiddleware``, intermediate tools are
    naturally preserved — no snapshot/restore or loop-message merging needed.
    These tests verify the correct ordering and uniqueness of tool calls in
    sessions with both approved and non-approved tools.
    """

    @pytest.mark.asyncio
    async def test_intermediate_tool_preserved_auto(self, tmp_path):
        """Auto (no approval): echo + guarded_echo both execute and appear in session."""
        responses = [
            MockResponse(tool_calls=[("echo", "e1", {"message": "intermediate"})]),
            MockResponse(tool_calls=[("guarded_echo", "g1", {"message": "guarded"})]),
            MockResponse(text="All done"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_default="auto")
        try:
            await ctx.send_message("Run both tools")

            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            assert "echo" in tool_names, f"echo lost! Tools: {tool_names}"
            assert "guarded_echo" in tool_names
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_intermediate_tool_preserved_user_approval(self, tmp_path):
        """User-approve: echo runs before guarded_echo — both in session."""
        responses = [
            # LLM call 1: echo (no approval needed, executes immediately)
            MockResponse(tool_calls=[("echo", "e1", {"message": "intermediate"})]),
            # LLM call 2: guarded_echo (middleware pauses for approval)
            MockResponse(tool_calls=[("guarded_echo", "g1", {"message": "guarded"})]),
            # LLM call 3: final text
            MockResponse(text="User approved and done"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_overrides={"guarded_echo": "require"})
        try:
            await ctx.bus.publish(UserMessage(text="Run both tools"))

            approval_events = await wait_for_event(ctx.events, ApprovalRequest, timeout=20.0)
            await ctx.approve(approval_events[0].request_id)
            await wait_for_idle(ctx)

            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            assert "echo" in tool_names, f"echo lost! Tools: {tool_names}"
            assert "guarded_echo" in tool_names

            # Verify ordering: user → echo → guarded_echo
            echo_idx = guarded_idx = user_idx = None
            for i, m in enumerate(raw):
                if m.get("role") == "user" and user_idx is None:
                    user_idx = i
                for c in m.get("contents", []):
                    if isinstance(c, dict) and c.get("type") == "function_call":
                        if c.get("name") == "echo" and echo_idx is None:
                            echo_idx = i
                        elif c.get("name") == "guarded_echo" and guarded_idx is None:
                            guarded_idx = i
            assert user_idx is not None and echo_idx is not None and guarded_idx is not None
            assert user_idx < echo_idx < guarded_idx
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_multiple_intermediate_tools_preserved(self, tmp_path):
        """Multiple intermediate tools (echo, concat) before guarded — all preserved."""
        responses = [
            MockResponse(tool_calls=[("echo", "e1", {"message": "first"})]),
            MockResponse(tool_calls=[("concat", "c1", {"a": "x", "b": "y"})]),
            MockResponse(tool_calls=[("guarded_echo", "g1", {"message": "guarded"})]),
            MockResponse(text="All three executed"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_default="auto")
        try:
            await ctx.send_message("Run all three tools")

            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            assert "echo" in tool_names
            assert "concat" in tool_names
            assert "guarded_echo" in tool_names
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_no_duplicate_tools(self, tmp_path):
        """Each tool appears exactly once in session — no duplicates."""
        responses = [
            MockResponse(tool_calls=[("echo", "e1", {"message": "one"})]),
            MockResponse(tool_calls=[("concat", "c1", {"a": "x", "b": "y"})]),
            MockResponse(tool_calls=[("guarded_echo", "g1", {"message": "guarded"})]),
            MockResponse(text="Done"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_default="auto")
        try:
            await ctx.send_message("Run tools")

            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            assert tool_names.count("echo") == 1, f"echo duplicated! Tools: {tool_names}"
            assert tool_names.count("concat") == 1, f"concat duplicated! Tools: {tool_names}"
            assert tool_names.count("guarded_echo") == 1, f"guarded_echo duplicated! Tools: {tool_names}"
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_no_approval_normal_completion(self, tmp_path):
        """Without approval tools, normal completion produces clean session."""
        responses = [
            MockResponse(tool_calls=[("echo", "e1", {"message": "first"})]),
            MockResponse(tool_calls=[("concat", "c1", {"a": "a", "b": "b"})]),
            MockResponse(text="Normal completion"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_default="auto")
        try:
            await ctx.send_message("Run normal tools")

            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            assert tool_names == ["echo", "concat"], f"Unexpected tools: {tool_names}"

            non_marker = [
                m["role"]
                for m in raw
                if m.get("additional_properties", {}).get(HistoryMarkerKind.KEY) != HistoryMarkerKind.TURN
            ]
            assert non_marker == ["user", "assistant", "tool", "assistant", "tool", "assistant"]
        finally:
            await ctx.cleanup()


# ---------------------------------------------------------------------------
# S9: Multi-approval iterations (no duplicate messages)
# ---------------------------------------------------------------------------


class TestMultipleApprovalTools:
    """Multiple approval-needing tools in a single turn execute inline via
    ``ApprovalMiddleware`` — each is individually approved within the
    Chrys tool loop. No snapshot/restore, no loop-message capture
    between iterations.

    Real-world pattern: read_file → write_file(approval) → read_file →
    write_file(approval) → text.
    """

    @pytest.mark.asyncio
    async def test_two_guarded_tools_auto(self, tmp_path):
        """Auto: echo → guarded → echo → guarded → text. All execute inline."""
        responses = [
            MockResponse(tool_calls=[("echo", "e1", {"message": "read1"})]),
            MockResponse(tool_calls=[("guarded_echo", "g1", {"message": "write1"})]),
            MockResponse(tool_calls=[("echo", "e2", {"message": "read2"})]),
            MockResponse(tool_calls=[("guarded_echo", "g2", {"message": "write2"})]),
            MockResponse(text="Both writes done"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_default="auto")
        try:
            await ctx.send_message("Read and write two files")

            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            call_ids = extract_session_call_ids(raw)

            assert tool_names.count("echo") == 2
            assert tool_names.count("guarded_echo") == 2
            assert len(call_ids) == len(set(call_ids)), f"Duplicate call_ids: {call_ids}"
            assert tool_names == ["echo", "guarded_echo", "echo", "guarded_echo"]
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_two_guarded_tools_user_approve(self, tmp_path):
        """User-approve: echo → guarded → echo → guarded → text.
        Two user approvals — each pauses the tool loop inline."""
        responses = [
            MockResponse(tool_calls=[("echo", "e1", {"message": "read1"})]),
            MockResponse(tool_calls=[("guarded_echo", "g1", {"message": "write1"})]),
            MockResponse(tool_calls=[("echo", "e2", {"message": "read2"})]),
            MockResponse(tool_calls=[("guarded_echo", "g2", {"message": "write2"})]),
            MockResponse(text="Both writes done"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_overrides={"guarded_echo": "require"})
        try:
            await ctx.bus.publish(UserMessage(text="Read and write two files"))

            # First approval
            events1 = await wait_for_event(ctx.events, ApprovalRequest, timeout=20.0)
            await ctx.approve(events1[0].request_id)

            # Second approval
            events2 = await wait_for_event(ctx.events, ApprovalRequest, timeout=20.0, min_count=2)
            await ctx.approve(events2[1].request_id)
            await wait_for_idle(ctx)

            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            call_ids = extract_session_call_ids(raw)

            assert tool_names.count("echo") == 2
            assert tool_names.count("guarded_echo") == 2
            assert len(call_ids) == len(set(call_ids)), f"Duplicate call_ids: {call_ids}"
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_three_guarded_tools(self, tmp_path):
        """Three consecutive guarded tools — stress test."""
        responses = [
            MockResponse(tool_calls=[("echo", "e1", {"message": "r1"})]),
            MockResponse(tool_calls=[("guarded_echo", "g1", {"message": "w1"})]),
            MockResponse(tool_calls=[("concat", "c1", {"a": "x", "b": "y"})]),
            MockResponse(tool_calls=[("guarded_echo", "g2", {"message": "w2"})]),
            MockResponse(tool_calls=[("echo", "e2", {"message": "r2"})]),
            MockResponse(tool_calls=[("guarded_echo", "g3", {"message": "w3"})]),
            MockResponse(text="All three writes done"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_default="auto")
        try:
            await ctx.send_message("Three rounds of read-write")

            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            call_ids = extract_session_call_ids(raw)

            assert tool_names.count("echo") == 2
            assert tool_names.count("concat") == 1
            assert tool_names.count("guarded_echo") == 3
            assert len(call_ids) == len(set(call_ids)), f"Duplicate call_ids: {call_ids}"
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_multi_approval_second_turn(self, tmp_path):
        """Multiple guarded tools in the second turn — verifies turn boundary handling."""
        responses = [
            # Turn 1: simple, no approval
            MockResponse(tool_calls=[("echo", "t1e1", {"message": "turn1"})]),
            MockResponse(text="Turn 1 done"),
            # Turn 2: echo → guarded → echo → guarded → text
            MockResponse(tool_calls=[("echo", "t2e1", {"message": "read1"})]),
            MockResponse(tool_calls=[("guarded_echo", "t2g1", {"message": "write1"})]),
            MockResponse(tool_calls=[("echo", "t2e2", {"message": "read2"})]),
            MockResponse(tool_calls=[("guarded_echo", "t2g2", {"message": "write2"})]),
            MockResponse(text="Turn 2 done"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_default="auto")
        try:
            await ctx.send_message("Turn 1")
            await ctx.send_message("Turn 2 with guarded tools")

            raw = await ctx.get_session_messages()
            call_ids = extract_session_call_ids(raw)

            assert len(call_ids) == len(set(call_ids)), f"Duplicate call_ids: {call_ids}"
            tool_names = extract_session_tool_names(raw)
            assert tool_names.count("echo") == 3  # 1 from turn1, 2 from turn2
            assert tool_names.count("guarded_echo") == 2
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_multi_approval_with_compaction(self, tmp_path):
        """Multiple guarded tools with compaction active — no duplicates."""
        from chrys.service.profiles.agents.schema import CompactionConfig

        big_result_text = "x" * 500

        from typing import Annotated

        from chrys.kernel import FunctionTool

        def _big_echo(message: Annotated[str, "Message"]) -> str:
            return f"big: {big_result_text}"

        big_echo = FunctionTool(func=_big_echo, name="big_echo", description="Big echo")
        guarded = FunctionTool(func=_big_echo, name="guarded_big", description="Guarded big")

        responses = [
            MockResponse(tool_calls=[("big_echo", "t1a", {"message": "fill1"})]),
            MockResponse(tool_calls=[("big_echo", "t1b", {"message": "fill2"})]),
            MockResponse(tool_calls=[("big_echo", "t1c", {"message": "fill3"})]),
            MockResponse(text="Turn 1 done"),
            MockResponse(tool_calls=[("big_echo", "t2e1", {"message": "read1"})]),
            MockResponse(tool_calls=[("guarded_big", "t2g1", {"message": "write1"})]),
            MockResponse(tool_calls=[("big_echo", "t2e2", {"message": "read2"})]),
            MockResponse(tool_calls=[("guarded_big", "t2g2", {"message": "write2"})]),
            MockResponse(text="Turn 2 done"),
        ]

        ctx = await create_test_engine(
            responses,
            tmp_path,
            approval_default="auto",
            tools=[big_echo, guarded],
            compaction=CompactionConfig(),
            max_context_tokens=2000,
        )
        try:
            await ctx.send_message("Fill context")
            await ctx.send_message("Multi-approval under compaction")

            raw = await ctx.get_session_messages()
            call_ids = extract_session_call_ids(raw)
            # No duplicate call ids regardless of whether compaction excluded
            # messages — duplicates would still surface as repeated ids.
            assert len(call_ids) == len(set(call_ids)), f"Duplicate call_ids: {call_ids}"
            # Phase 4 drop-all may have excluded tool_call messages from the
            # visible session view; surviving names are a subset of what was
            # attempted, and no unexpected names should appear.
            tool_names = extract_session_tool_names(raw)
            assert set(tool_names).issubset({"big_echo", "guarded_big"})
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_multi_approval_streaming(self, tmp_path):
        """Multiple guarded tools with streaming — same correctness guarantees."""
        responses = [
            MockResponse(tool_calls=[("echo", "e1", {"message": "read1"})]),
            MockResponse(tool_calls=[("guarded_echo", "g1", {"message": "write1"})]),
            MockResponse(tool_calls=[("echo", "e2", {"message": "read2"})]),
            MockResponse(tool_calls=[("guarded_echo", "g2", {"message": "write2"})]),
            MockResponse(text="Streaming done"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_default="auto", stream=True)
        try:
            await ctx.send_message("Stream with guarded tools")

            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            call_ids = extract_session_call_ids(raw)

            assert tool_names.count("echo") == 2
            assert tool_names.count("guarded_echo") == 2
            assert len(call_ids) == len(set(call_ids)), f"Duplicate call_ids: {call_ids}"
        finally:
            await ctx.cleanup()


# ---------------------------------------------------------------------------
# S10: Compression + approval rollback
# ---------------------------------------------------------------------------


class TestCompressionWithApproval:
    """Compression (compress_context) followed by a guarded tool in the same
    turn.  With inline approval via ``ApprovalMiddleware``, there is no
    snapshot/restore — the compression is permanent and the guarded tool
    executes (or is rejected) inline.

    Verifies state consistency: blocks match summaries, no stale flags.
    """

    @pytest.mark.asyncio
    async def test_compression_then_guarded_tool(self, tmp_path):
        """compress_context in LLM response 1, guarded_echo in response 2 — both succeed."""
        responses = [
            MockResponse(text="R1"),
            MockResponse(text="R2"),
            MockResponse(text="R3"),
            MockResponse(
                tool_calls=[
                    ("compress_context", "cc1", {"marker_id": "turn_2", "summary": "Turns 1-2 summary"}),
                ]
            ),
            MockResponse(
                tool_calls=[
                    ("guarded_echo", "g1", {"message": "after compress"}),
                ]
            ),
            MockResponse(text="Done after compress + guarded"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_default="auto")
        try:
            for i in range(3):
                await ctx.send_message(f"turn {i + 1}")

            await ctx.send_message("compress and guard")

            state = ctx.engine._executor.history_state
            compressed = state.get("compressed_msgs", [])
            messages = state.get("messages", [])

            from chrys.service.context.providers.history import _is_compressed_summary

            summaries = [m for m in messages if _is_compressed_summary(m)]
            assert len(compressed) == len(summaries), (
                f"Inconsistent state: {len(compressed)} blocks but {len(summaries)} summaries"
            )

            raw = await ctx.get_session_messages()
            tool_names = extract_session_tool_names(raw)
            assert "guarded_echo" in tool_names

            finals = extract_final_messages(ctx.events)
            assert any("Done" in f for f in finals)
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_compression_then_guarded_tool_streaming(self, tmp_path):
        """Same scenario in streaming mode."""
        responses = [
            MockResponse(text="R1"),
            MockResponse(text="R2"),
            MockResponse(text="R3"),
            MockResponse(
                tool_calls=[
                    ("compress_context", "cc1", {"marker_id": "turn_2", "summary": "Turns 1-2 summary"}),
                ]
            ),
            MockResponse(
                tool_calls=[
                    ("guarded_echo", "g1", {"message": "after compress"}),
                ]
            ),
            MockResponse(text="Streaming done"),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_default="auto", stream=True)
        try:
            for i in range(3):
                await ctx.send_message(f"turn {i + 1}")

            await ctx.send_message("compress and guard")

            state = ctx.engine._executor.history_state
            compressed = state.get("compressed_msgs", [])
            messages = state.get("messages", [])

            from chrys.service.context.providers.history import _is_compressed_summary

            summaries = [m for m in messages if _is_compressed_summary(m)]
            assert len(compressed) == len(summaries)
        finally:
            await ctx.cleanup()

    @pytest.mark.asyncio
    async def test_compression_then_guarded_tool_rejected(self, tmp_path):
        """compress_context processed, then guarded_echo rejected.

        The compression is permanent (no rollback needed with inline approval).
        The rejection returns an error to the LLM which responds with text.
        """
        responses = [
            MockResponse(text="R1"),
            MockResponse(text="R2"),
            MockResponse(text="R3"),
            MockResponse(
                tool_calls=[
                    ("compress_context", "cc1", {"marker_id": "turn_2", "summary": "Turns 1-2 summary"}),
                ]
            ),
            MockResponse(
                tool_calls=[
                    ("guarded_echo", "g1", {"message": "will be rejected"}),
                ]
            ),
            MockResponse(text="OK, skipping that tool."),
        ]
        ctx = await create_test_engine(responses, tmp_path, approval_overrides={"guarded_echo": "require"})
        try:
            for i in range(3):
                await ctx.send_message(f"turn {i + 1}")

            await ctx.bus.publish(UserMessage(text="compress and guard"))

            approval_events = await wait_for_event(ctx.events, ApprovalRequest, timeout=20.0)
            await ctx.reject(approval_events[0].request_id)
            await wait_for_idle(ctx)

            state = ctx.engine._executor.history_state
            compressed = state.get("compressed_msgs", [])
            messages = state.get("messages", [])

            from chrys.service.context.providers.history import _is_compressed_summary

            summaries = [m for m in messages if _is_compressed_summary(m)]
            assert len(compressed) == len(summaries), (
                f"Inconsistent after rejection: {len(compressed)} blocks but {len(summaries)} summaries"
            )

            raw = await ctx.get_session_messages()
            approvals = extract_session_approvals(raw)
            rejected = [a for a in approvals if a and a.get("status") == "user_rejected"]
            assert len(rejected) >= 1

            finals = extract_final_messages(ctx.events)
            assert any("skipping" in f.lower() or "OK" in f for f in finals)
        finally:
            await ctx.cleanup()
