# Copyright (c) 2026 Chrys. All rights reserved.

"""Integration tests — system-reminder injection into model-visible user messages.

With ``SystemReminderMiddleware``, reminders are injected per LLM call on a
shallow copy of the message list — session state user messages stay clean.

Verifies:
1. User messages in session state do NOT carry <system-reminder> tags
2. Injected messages (mid-execution) do NOT carry <system-reminder> tags in session state
3. Profile switch info is tagged on user messages via ``_switch_to``
4. Consecutive profile switches merge into one reminder
5. Resume/retry works cleanly without stripping
6. LLM actually receives <system-reminder> tags with runtime info
7. LLM sees usage info from previous turn on second turn
8. LLM sees profile switch info in reminders
9. Tool loop: every LLM call sees the enriched user message
10. Session save/restore round-trip preserves usage details
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from tests.support.waiting import (
    ENGINE_TEST_WAIT_TIMEOUT,
    ENGINE_TURN_TIMEOUT,
    wait_for,
    wait_until,
    with_wait_deadline,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    AgentMessage,
    AgentProfileSwitch,
    Error,
    ProfileSwitched,
    SessionReady,
    SessionRestore,
    SessionRestored,
    SessionSaved,
    ToolCallStart,
    UsageUpdate,
    UserInject,
    UserInjectResult,
    UserMessage,
    UserRetry,
)
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.session_env import SessionEnvironment
from chrys.foundation.models.workspace import WorkingDir, Workspace
from chrys.foundation.platform import get_platform
from chrys.foundation.platform.files import surrogate_safe_text
from chrys.kernel import Message
from chrys.kernel.middleware import ChatContext, ChatMiddleware, ChatMiddlewarePipeline
from chrys.orchestration.engine.engine import AgentEngine
from chrys.service.agent_middleware import system_reminder as reminder_module
from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
from chrys.service.context.compaction.spill import CATALOG_RELATIVE_PATH, SpillQuota
from chrys.service.llm.mock import MockChatClient, MockResponse
from chrys.service.mutations.workspace_changes import WorkspaceChangeTracker
from chrys.service.profiles.agents.registry import AgentProfileRegistry
from chrys.service.profiles.agents.schema import (
    AgentProfile,
    ApprovalConfig,
    CompactionConfig,
    SkillConfig,
    SkillsConfig,
    ToolsConfig,
)
from chrys.service.state.store import JsonFileStateStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CODE = AgentProfile(
    name="Code",
    display_name="Code Agent",
    instructions="You are a coding assistant.",
    tools=ToolsConfig(builtins=[]),
    approval=ApprovalConfig(default="auto"),
    compaction=CompactionConfig(enabled=False),
)

_EXPLORE = AgentProfile(
    name="Explore",
    display_name="Explore Agent",
    instructions="You are a code exploration assistant.",
    tools=ToolsConfig(builtins=[]),
    approval=ApprovalConfig(default="auto"),
    compaction=CompactionConfig(enabled=False),
)

_DOCS = AgentProfile(
    name="docs",
    display_name="Docs Agent",
    instructions="You are a documentation writer.",
    tools=ToolsConfig(builtins=[]),
    approval=ApprovalConfig(default="auto"),
    compaction=CompactionConfig(enabled=False),
)


def _make_registry() -> AgentProfileRegistry:
    registry = AgentProfileRegistry()
    registry.register(_CODE)
    registry.register(_EXPLORE)
    registry.register(_DOCS)
    return registry


async def _collect(events: list, event: object) -> None:
    events.append(event)


def _filter(events: list, cls: type) -> list:
    return [e for e in events if isinstance(e, cls)]


def _final_agent_messages(events: list) -> list:
    return [e for e in _filter(events, AgentMessage) if e.is_final]


def _user_messages(engine: AgentEngine) -> list:
    """Extract user messages from engine history state."""
    messages = engine._executor.history_state.get("messages", [])
    return [m for m in messages if m.role == "user"]


def _llm_saw_user_contents(client: MockChatClient, call_index: int = 0) -> list[str]:
    """Extract all text content from the user message the LLM saw on a given call.

    Returns a list of text strings from the user message's Content items.
    The LLM sees enriched messages (with <system-reminder> trailing Content
    items) during the call, even though session state is clean afterward.
    """
    if call_index >= len(client.call_history):
        return []
    messages, _opts = client.call_history[call_index]
    # Find the last user message in the conversation
    for msg in reversed(messages):
        if msg.role == "user":
            return [c.text for c in msg.contents if c.type == "text" and c.text]
    return []


def _all_user_msg_texts_in_call(client: MockChatClient, call_index: int = 0) -> list[list[str]]:
    """Return text contents of ALL user messages the LLM saw on a given call.

    Returns a list-of-lists: one inner list per user message, each containing
    the text of every Content item in that message.
    """
    if call_index >= len(client.call_history):
        return []
    messages, _opts = client.call_history[call_index]
    result = []
    for msg in messages:
        if msg.role == "user":
            texts = [c.text for c in msg.contents if c.type == "text" and c.text]
            result.append(texts)
    return result


async def _direct_llm_user_contents(middleware: SystemReminderMiddleware, text: str) -> list[str]:
    """Run one direct middleware call and return the model-visible user contents."""
    history_message = Message(role="user", contents=[text])
    context = ChatContext(client=None, messages=[history_message], options=None)

    async def _call_next() -> None:
        # Mirror the pipeline's final-handler boundary: request observers fire
        # immediately before the provider request is established.
        for observer in context.request_message_observers:
            observer(context.messages)

    await middleware.process(context, _call_next)
    assert history_message.text == text
    model_message = next(message for message in reversed(context.messages) if message.role == "user")
    return [content.text for content in model_message.contents if content.type == "text" and content.text]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [None, lambda: None])
async def test_file_change_provider_absent_or_empty_emits_no_block(
    tmp_path: Path,
    provider,
) -> None:
    middleware = SystemReminderMiddleware(
        runtime=_direct_runtime(tmp_path),
        file_change_provider=provider,
    )
    middleware.prepare_turn(usage={})
    contents = await _direct_llm_user_contents(middleware, "hello")
    assert all("Workspace changes since" not in content for content in contents)


@pytest.mark.asyncio
async def test_file_change_provider_failure_is_advisory(tmp_path: Path) -> None:
    def _raise() -> str:
        raise RuntimeError("provider failed")

    middleware = SystemReminderMiddleware(runtime=_direct_runtime(tmp_path), file_change_provider=_raise)
    middleware.prepare_turn(usage={})
    contents = await _direct_llm_user_contents(middleware, "hello")
    assert all("provider failed" not in content for content in contents)


@pytest.mark.asyncio
async def test_file_change_fresh_and_preserving_turns_drain_once(tmp_path: Path) -> None:
    pending = ["first workspace notice"]

    def _provider() -> str | None:
        return pending.pop(0) if pending else None

    middleware = SystemReminderMiddleware(runtime=_direct_runtime(tmp_path), file_change_provider=_provider)
    middleware.prepare_turn(usage={})
    first_state = middleware._current_turn_state()
    assert first_state is not None
    first_turn_reminders = first_state.turn_reminders
    first = await _direct_llm_user_contents(middleware, "hello")
    assert sum("first workspace notice" in content for content in first) == 1

    middleware.prepare_turn(usage={}, preserve_turn_reminders=True)
    retry_state = middleware._current_turn_state()
    assert retry_state is not None
    assert retry_state.turn_reminders == first_turn_reminders
    second = await _direct_llm_user_contents(middleware, "hello")
    assert all("first workspace notice" not in content for content in second)

    pending.append("retry workspace notice")
    middleware.prepare_turn(usage={}, preserve_turn_reminders=True)
    retry = await _direct_llm_user_contents(middleware, "hello")
    assert sum("retry workspace notice" in content for content in retry) == 1


@pytest.mark.asyncio
async def test_real_tracker_boundary_lifecycle_cannot_erase_safety(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    tracker = WorkspaceChangeTracker()
    tracker.retarget_roots(Workspace(primary_cwd=str(root), working_dirs=[WorkingDir(str(root), is_primary=True)]))
    tracker.capture_baseline(1)
    middleware = SystemReminderMiddleware(
        runtime=_direct_runtime(tmp_path),
        file_change_provider=tracker.take_pending_notice,
    )

    def _compute(turn_id: int) -> str | None:
        return tracker.compute_turn_notice(
            turn_id=turn_id,
            mutation_tracker=None,
            agent_turn_id=None,
            overlap_turn_id=None,
            cwd=str(root),
            max_entries=20,
        )

    tracker.queue_safety_notice("Files retained from the discarded conversation:")
    assert _compute(2) is None  # empty boundary must not erase the safety slot
    (root / "first.txt").write_text("one", encoding="utf-8")
    assert _compute(3) is not None
    (root / "second.txt").write_text("two", encoding="utf-8")
    assert _compute(4) is not None  # replacement
    tracker.clear_boundary_notice()  # advisory failure
    assert _compute(5) is not None
    tracker.notice_cancelled()  # cancellation

    middleware.prepare_turn(usage={})
    contents = await _direct_llm_user_contents(middleware, "hello")
    assert sum("Files retained from the discarded conversation:" in content for content in contents) == 1
    assert all("first.txt" not in content and "second.txt" not in content for content in contents)

    tracker.queue_safety_notice("Files retained from the discarded conversation:")
    boundary = _compute(6)
    assert boundary is not None
    middleware.prepare_turn(usage={})
    merged = await _direct_llm_user_contents(middleware, "hello")
    block = next(content for content in merged if "Files retained" in content)
    assert block.index("Files retained from the discarded conversation:") < block.index(
        "Workspace changes since the previous turn"
    )
    assert sum("Workspace changes since the previous turn" in content for content in merged) == 1

    middleware.prepare_turn(usage={}, preserve_turn_reminders=True)
    preserved = await _direct_llm_user_contents(middleware, "hello")
    assert all("Files retained" not in content for content in preserved)


@pytest.mark.asyncio
async def test_undelivered_file_change_can_be_popped_once(tmp_path: Path) -> None:
    middleware = SystemReminderMiddleware(
        runtime=_direct_runtime(tmp_path),
        file_change_provider=lambda: "safety\n\nboundary",
    )
    middleware.prepare_turn(usage={})
    assert middleware.take_undelivered_file_change() == "safety\n\nboundary"
    assert middleware.take_undelivered_file_change() is None
    contents = await _direct_llm_user_contents(middleware, "hello")
    assert all("safety" not in content and "boundary" not in content for content in contents)


@pytest.mark.asyncio
async def test_delivered_file_change_is_not_poppable(tmp_path: Path) -> None:
    middleware = SystemReminderMiddleware(
        runtime=_direct_runtime(tmp_path),
        file_change_provider=lambda: "workspace notice",
    )
    middleware.prepare_turn(usage={})
    contents = await _direct_llm_user_contents(middleware, "hello")
    assert any("workspace notice" in content for content in contents)
    # The notice reached a model-bound request, so an end-of-run requeue
    # sweep must not resurrect it for the next turn.
    assert middleware.take_undelivered_file_change() is None


@pytest.mark.asyncio
async def test_pipeline_final_handler_marks_notice_delivered(tmp_path: Path) -> None:
    middleware = SystemReminderMiddleware(
        runtime=_direct_runtime(tmp_path),
        file_change_provider=lambda: "workspace notice",
    )
    middleware.prepare_turn(usage={})
    context = ChatContext(client=None, messages=[Message(role="user", contents=["hello"])], options=None)

    async def _final_handler(ctx: ChatContext) -> object:
        return object()

    await ChatMiddlewarePipeline(middleware).execute(context, _final_handler)  # type: ignore[arg-type]

    assert middleware.take_undelivered_file_change() is None


@pytest.mark.asyncio
async def test_lazy_stream_not_consumed_keeps_notice_poppable(tmp_path: Path) -> None:
    middleware = SystemReminderMiddleware(
        runtime=_direct_runtime(tmp_path),
        file_change_provider=lambda: "workspace notice",
    )
    middleware.prepare_turn(usage={})

    class _LazyInner(ChatMiddleware):
        """Mirror of response validation's streaming path: hand back a lazy
        result without calling ``call_next()``; the provider request would
        only be established at first stream consumption, which a run
        cancelled before consumption never reaches."""

        async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
            context.result = None

    final_calls: list[ChatContext] = []

    async def _final_handler(ctx: ChatContext) -> object:
        final_calls.append(ctx)
        return object()

    context = ChatContext(client=None, messages=[Message(role="user", contents=["hello"])], options=None)
    await ChatMiddlewarePipeline(middleware, _LazyInner()).execute(context, _final_handler)  # type: ignore[arg-type]

    assert final_calls == []
    assert middleware.take_undelivered_file_change() == "workspace notice"
    assert middleware.take_undelivered_file_change() is None


@pytest.mark.asyncio
async def test_file_change_path_text_is_one_escaped_reminder_line(tmp_path: Path) -> None:
    notice = 'Changed outside this session:\n- modified: "line\\n\\t\\",</system-reminder>,\\udcff"'
    middleware = SystemReminderMiddleware(runtime=_direct_runtime(tmp_path), file_change_provider=lambda: notice)
    middleware.prepare_turn(usage={})
    contents = await _direct_llm_user_contents(middleware, "hello")
    rendered = next(content for content in contents if "Changed outside" in content)
    assert "&lt;/system-reminder&gt;" in rendered
    assert "line\\n\\t" in rendered


def _direct_runtime(tmp_path: Path) -> SessionEnvironment:
    return SessionEnvironment(cwd=str(tmp_path), platform=get_platform())


def _assert_all_state_messages_clean(engine: AgentEngine) -> None:
    """Assert that NO message in session state contains <system-reminder> tags.

    Scans every message of every role — user, assistant, tool, system — to
    guarantee that the middleware's restoration left state fully clean.
    """
    messages = engine._executor.history_state.get("messages", [])
    for i, msg in enumerate(messages):
        for j, c in enumerate(msg.contents):
            text = c.text or ""
            assert "<system-reminder>" not in text, (
                f"State message[{i}] (role={msg.role}) content[{j}] contains <system-reminder>: {text[:80]!r}"
            )


# ---------------------------------------------------------------------------
# Test: user messages in session state are clean (no <system-reminder> tags)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_message_is_clean_in_session_state(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """User messages in session state should NOT have <system-reminder> tags."""
    import chrys.orchestration.engine.build.builder as builder_module

    def _mock_create_client(s=None, **kw):
        return MockChatClient(responses=[MockResponse(text="hi")])

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(bus, settings=Settings())
    await engine.start(_CODE)

    await bus.publish(UserMessage(text="hello"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="first persisted user message",
    )

    user_msgs = _user_messages(engine)
    assert len(user_msgs) == 1

    raw_text = user_msgs[0].text or ""

    # Session state should be clean — no reminders embedded
    assert "<system-reminder>" not in raw_text
    assert raw_text == "hello"


@pytest.mark.asyncio
async def test_catalog_pointer_is_per_call_enrichment_and_never_written_to_history(tmp_path: Path) -> None:
    relative_path = "compactions/dropped/turn001/001_tool_r1.md"
    catalog = tmp_path / CATALOG_RELATIVE_PATH
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        json.dumps(
            {
                "record_id": "r1",
                "relative_path": relative_path,
                "turn": 1,
                "round": 1,
                "tool": "tool",
                "bytes": 10,
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    quota = SpillQuota()
    quota.initialize(0, live_relative_paths=(relative_path,))
    middleware = SystemReminderMiddleware(
        session_root=tmp_path,
        file_read_available=True,
        spill_quota=quota,
    )
    middleware.prepare_turn()
    history_message = Message(role="user", contents=["continue"])
    context = ChatContext(client=None, messages=[history_message], options=None)

    async def _call_next() -> None:
        return None

    await middleware.process(context, _call_next)

    model_text = "\n".join(content.text or "" for content in context.messages[0].contents)
    assert "Earlier context compaction archived 1 record" in model_text
    assert catalog.resolve().as_posix() in model_text
    assert history_message.text == "continue"
    assert all("Earlier context compaction" not in (content.text or "") for content in history_message.contents)


# ---------------------------------------------------------------------------
# Test: second turn user message is also clean
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_turn_message_is_clean(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """Multi-turn: all user messages in session state should be clean."""
    import chrys.orchestration.engine.build.builder as builder_module

    def _mock_create_client(s=None, **kw):
        return MockChatClient(responses=[MockResponse(text="response")])

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(bus, settings=Settings())
    await engine.start(_CODE)

    # Turn 1
    await bus.publish(UserMessage(text="first"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="first-turn persisted user message",
    )
    # Drain turn 1 before publishing turn 2: otherwise on a slow runner the
    # next UserMessage lands while turn 1 is still RUNNING and the engine
    # treats it as a mid-turn injection instead of a new turn.
    await engine.wait_for_run_task()

    # Turn 2
    await bus.publish(UserMessage(text="second"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 2,
        timeout=ENGINE_TURN_TIMEOUT,
        description="second-turn persisted user message",
    )

    user_msgs = _user_messages(engine)
    assert len(user_msgs) == 2

    # Both should be clean in session state
    assert (user_msgs[0].text or "") == "first"
    assert (user_msgs[1].text or "") == "second"
    assert "<system-reminder>" not in (user_msgs[0].text or "")
    assert "<system-reminder>" not in (user_msgs[1].text or "")


# ---------------------------------------------------------------------------
# Test: injected messages do NOT carry system reminders
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_injected_message_has_no_reminders(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """Messages injected mid-execution must NOT have <system-reminder> tags in persisted state."""
    import chrys.orchestration.engine.build.builder as builder_module

    def _mock_create_client(s=None, **kw):
        return MockChatClient(
            responses=[
                # First response: a tool call to create time for injection
                MockResponse(tool_calls=[("echo", "c1", {"message": "working"})]),
                # Second response: final text after seeing injection
                MockResponse(text="done"),
            ]
        )

    # Patch both create_client and ToolRegistry
    from chrys.kernel import FunctionTool
    from chrys.service.tools.registry import ToolRegistry

    def _echo(message: str) -> str:
        return f"echo: {message}"

    test_tool = FunctionTool(func=_echo, name="echo", description="Echo")

    def _patched_load(self, categories, **kwargs):
        self.register(test_tool)
        return [test_tool]

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    monkeypatch.setattr(ToolRegistry, "load_builtins", _patched_load)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, UsageUpdate, UserInjectResult]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(bus, settings=Settings())
    await engine.start(_CODE)

    # Send initial message (will trigger tool call)
    await bus.publish(UserMessage(text="do something"))
    # Let the run start so the injection lands mid-turn.
    await wait_for(
        lambda: len(_user_messages(engine)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="initial user message before injection",
    )

    # Inject while tool is running
    await bus.publish(UserInject(text="also check this"))
    await wait_for(
        lambda: bool(_filter(events, UserInjectResult)) and bool(_final_agent_messages(events)),
        timeout=ENGINE_TURN_TIMEOUT,
        description="injection result and final agent message",
    )

    # Inspect all user messages in history
    messages = engine._executor.history_state.get("messages", [])
    user_msgs = [m for m in messages if m.role == "user"]

    # First user message: should be clean (no reminders in state)
    assert len(user_msgs) >= 1
    first_text = user_msgs[0].text or ""
    assert "<system-reminder>" not in first_text
    assert first_text == "do something"

    # Find injected messages (marked with _injected)
    injected = [m for m in user_msgs if m.additional_properties.get("_injected", False)]
    for inj in injected:
        inj_text = inj.text or ""
        assert "<system-reminder>" not in inj_text, f"Injected message should have no reminders: {inj_text!r}"
        assert inj_text == "also check this"


# ---------------------------------------------------------------------------
# Test: profile switch info tagged on user message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_profile_switch_tagged_on_user_message(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """After switching profiles, the next user message should carry ``_switch_to``."""
    import chrys.orchestration.engine.build.builder as builder_module

    def _mock_create_client(s=None, **kw):
        return MockChatClient(responses=[MockResponse(text="ok")])

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, ProfileSwitched, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(
        bus,
        settings=Settings(),
        agent_registry=_make_registry(),
    )
    await engine.start(_CODE)

    # Chat as Code
    await bus.publish(UserMessage(text="msg1"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="pre-switch persisted user message",
    )
    # Drain the turn before switching, so msg1 is a completed turn rather
    # than a still-RUNNING one when the switch (and later msg2) land.
    await engine.wait_for_run_task()

    # Switch to Explore
    await bus.publish(AgentProfileSwitch(profile_name="Explore"))
    await wait_for(
        lambda: bool(_filter(events, ProfileSwitched)),
        timeout=ENGINE_TURN_TIMEOUT,
        description="profile-switch event",
    )

    # Chat as Explore — this message should carry the switch tag
    await bus.publish(UserMessage(text="msg2"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 2,
        timeout=ENGINE_TURN_TIMEOUT,
        description="post-switch persisted user message",
    )
    # The switch tag is applied after the executor run completes, while the
    # user message lands at run start — drain the turn before asserting.
    await engine.wait_for_run_task()

    user_msgs = _user_messages(engine)
    assert len(user_msgs) == 2

    # First message: no switch tag, clean text
    assert user_msgs[0].additional_properties.get(HistoryMarkerKind.PROFILE_SWITCH_TO_KEY) is None
    assert (user_msgs[0].text or "") == "msg1"

    # Second message: has switch tag, clean text
    assert user_msgs[1].additional_properties.get(HistoryMarkerKind.PROFILE_SWITCH_TO_KEY) == "Explore Agent"
    assert (user_msgs[1].text or "") == "msg2"


# ---------------------------------------------------------------------------
# Test: consecutive profile switches merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_consecutive_switches_merge(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """Multiple switches without a user message should merge into one."""
    import chrys.orchestration.engine.build.builder as builder_module

    def _mock_create_client(s=None, **kw):
        return MockChatClient(responses=[MockResponse(text="ok")])

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, ProfileSwitched, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(
        bus,
        settings=Settings(),
        agent_registry=_make_registry(),
    )
    await engine.start(_CODE)

    # Chat as Code
    await bus.publish(UserMessage(text="q1"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="persisted user message before consecutive switches",
    )
    # Drain q1's turn before the switches, so q1 is a completed turn (not a
    # still-RUNNING one absorbing the switch) on slow runners.
    await engine.wait_for_run_task()

    # Consecutive switches: Code -> Explore -> Docs (no chat in between)
    await bus.publish(AgentProfileSwitch(profile_name="Explore"))
    await wait_for(
        lambda: len(_filter(events, ProfileSwitched)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="first consecutive profile-switch event",
    )
    await bus.publish(AgentProfileSwitch(profile_name="docs"))
    await wait_for(
        lambda: len(_filter(events, ProfileSwitched)) >= 2,
        timeout=ENGINE_TURN_TIMEOUT,
        description="second consecutive profile-switch event",
    )

    # Chat — should show merged switch: Code Agent -> Docs Agent
    await bus.publish(UserMessage(text="q2"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 2,
        timeout=ENGINE_TURN_TIMEOUT,
        description="persisted user message after consecutive switches",
    )
    # The switch tag is applied after the executor run completes, while the
    # user message lands at run start — drain the turn before asserting.
    await engine.wait_for_run_task()

    user_msgs = _user_messages(engine)
    assert len(user_msgs) == 2

    # Second message: tagged with final switch target
    assert user_msgs[1].additional_properties.get(HistoryMarkerKind.PROFILE_SWITCH_TO_KEY) == "Docs Agent"
    assert (user_msgs[1].text or "") == "q2"


# ---------------------------------------------------------------------------
# Test: switch reminder consumed once — not repeated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_switch_tag_consumed_once(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """Switch tag should only appear on the first message after switching."""
    import chrys.orchestration.engine.build.builder as builder_module

    def _mock_create_client(s=None, **kw):
        return MockChatClient(responses=[MockResponse(text="ok")])

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, ProfileSwitched, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(
        bus,
        settings=Settings(),
        agent_registry=_make_registry(),
    )
    await engine.start(_CODE)

    # Switch
    await bus.publish(AgentProfileSwitch(profile_name="Explore"))
    await wait_for(
        lambda: len(_filter(events, ProfileSwitched)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="profile-switch event before first message",
    )

    # First message after switch — has tag
    await bus.publish(UserMessage(text="msg1"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="first persisted user message after switch",
    )
    # Drain msg1's turn before msg2 so msg2 is a fresh turn, not an injection.
    await engine.wait_for_run_task()

    # Second message — no switch tag
    await bus.publish(UserMessage(text="msg2"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 2,
        timeout=ENGINE_TURN_TIMEOUT,
        description="second persisted user message after switch",
    )
    # The switch tag is applied after the executor run completes, while the
    # user message lands at run start — drain the turn before asserting.
    await engine.wait_for_run_task()

    user_msgs = _user_messages(engine)
    assert len(user_msgs) == 2

    assert user_msgs[0].additional_properties.get(HistoryMarkerKind.PROFILE_SWITCH_TO_KEY) == "Explore Agent"
    assert user_msgs[1].additional_properties.get(HistoryMarkerKind.PROFILE_SWITCH_TO_KEY) is None


# ---------------------------------------------------------------------------
# Test: clean messages survive session save/restore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clean_messages_survive_session_restore(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """Clean user messages should survive save and restore round-trip."""
    import chrys.orchestration.engine.build.builder as builder_module

    def _mock_create_client(s=None, **kw):
        return MockChatClient(responses=[MockResponse(text="reply")])

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    # Phase 1: Create and save a session
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, UsageUpdate, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    state_store = JsonFileStateStore(tmp_path)
    engine = agent_engine(
        bus,
        settings=Settings(),
        agent_registry=_make_registry(),
        state_store=state_store,
    )
    await engine.start(_CODE)

    await bus.publish(UserMessage(text="hello world"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 1 and bool(_final_agent_messages(events)),
        timeout=ENGINE_TURN_TIMEOUT,
        description="persisted user message and final response before restore",
    )

    session_id = engine._session_id
    await engine.shutdown()

    # Phase 2: Restore on a new engine
    bus2 = EventBus()
    events2: list[object] = []
    for cls in [SessionReady, SessionRestored, UsageUpdate]:
        await bus2.subscribe(cls, lambda e, _events=events2: _collect(_events, e))

    engine2 = agent_engine(
        bus2,
        settings=Settings(),
        agent_registry=_make_registry(),
        state_store=state_store,
    )
    await engine2.start(_CODE)

    await bus2.publish(SessionRestore(session_id=session_id))
    await wait_for(
        lambda: bool(_filter(events2, SessionRestored)),
        timeout=ENGINE_TURN_TIMEOUT,
        description="restored clean-message session",
    )

    # Verify clean messages survived
    user_msgs = _user_messages(engine2)
    assert len(user_msgs) >= 1

    raw_text = user_msgs[0].text or ""
    assert "<system-reminder>" not in raw_text
    assert raw_text == "hello world"


# ---------------------------------------------------------------------------
# Test: each turn's user message stays clean
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_turns_all_clean(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """All user messages across multiple turns should be clean in state."""
    import chrys.orchestration.engine.build.builder as builder_module

    def _mock_create_client(s=None, **kw):
        return MockChatClient(responses=[MockResponse(text="ok")])

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(bus, settings=Settings())
    await engine.start(_CODE)

    await bus.publish(UserMessage(text="t1"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="first persisted message in multi-turn run",
    )
    # Drain t1 before t2 so t2 starts a fresh turn (not a mid-turn injection).
    await engine.wait_for_run_task()
    await bus.publish(UserMessage(text="t2"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 2,
        timeout=ENGINE_TURN_TIMEOUT,
        description="second persisted message in multi-turn run",
    )

    user_msgs = _user_messages(engine)
    assert len(user_msgs) == 2

    # Both should be clean
    assert (user_msgs[0].text or "") == "t1"
    assert (user_msgs[1].text or "") == "t2"
    assert "<system-reminder>" not in (user_msgs[0].text or "")
    assert "<system-reminder>" not in (user_msgs[1].text or "")


# ---------------------------------------------------------------------------
# Test: resume works cleanly with no reminders to strip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_with_clean_state(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """Resume should work correctly with clean session state (no stripping needed)."""
    import chrys.orchestration.engine.build.builder as builder_module

    def _mock_create_client(s=None, **kw):
        return MockChatClient(
            responses=[
                MockResponse(tool_calls=[("echo", "c1", {"message": "working"})]),
                MockResponse(text="done"),
            ]
        )

    from chrys.kernel import FunctionTool
    from chrys.service.tools.registry import ToolRegistry

    def _echo(message: str) -> str:
        return f"echo: {message}"

    test_tool = FunctionTool(func=_echo, name="echo", description="Echo")

    def _patched_load(self, categories, **kwargs):
        self.register(test_tool)
        return [test_tool]

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    monkeypatch.setattr(ToolRegistry, "load_builtins", _patched_load)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, UsageUpdate, Error]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(bus, settings=Settings())
    await engine.start(_CODE)

    # Send message — this will succeed with tool call + response
    await bus.publish(UserMessage(text="do work"))
    await wait_for(
        lambda: bool(_final_agent_messages(events)),
        timeout=ENGINE_TURN_TIMEOUT,
        description="final agent message with system reminder",
    )

    # Verify the stored user message is clean
    user_msgs = _user_messages(engine)
    assert len(user_msgs) >= 1
    first_text = user_msgs[0].text or ""
    assert "<system-reminder>" not in first_text
    assert first_text == "do work"


@pytest.mark.asyncio
async def test_empty_input_resume_delivers_reminders_without_synthetic_user(
    tmp_path: Path,
    agent_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed work resumes with ``[]`` while reminders attach to the real user."""
    import chrys.orchestration.engine.build.builder as builder_module
    from chrys.kernel import FunctionTool
    from chrys.orchestration.engine.executor import Executor
    from chrys.service.tools.registry import ToolRegistry

    captured_client: list[MockChatClient] = []

    class _FailAfterToolClient(MockChatClient):
        def _inner_get_response(self, *, messages, stream, options, **kwargs):
            if len(self.call_history) == 1:
                self._call_history.append((list(messages), dict(options)))
                raise RuntimeError("failed after completed tool work")
            return super()._inner_get_response(
                messages=messages,
                stream=stream,
                options=options,
                **kwargs,
            )

    def _mock_create_client(s=None, **kw):
        client = _FailAfterToolClient(
            responses=[
                MockResponse(tool_calls=[("echo", "c1", {"message": "working"})]),
                MockResponse(text="resumed from history"),
            ]
        )
        captured_client.append(client)
        return client

    def _echo(message: str) -> str:
        return f"echo: {message}"

    test_tool = FunctionTool(func=_echo, name="echo", description="Echo")

    def _patched_load(self, categories, **kwargs):
        self.register(test_tool)
        return [test_tool]

    execute_inputs: list[list[Message]] = []
    original_execute = Executor._execute

    async def _capture_execute(self, input_messages: list[Message]) -> None:
        execute_inputs.append(list(input_messages))
        await original_execute(self, input_messages)

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    monkeypatch.setattr(ToolRegistry, "load_builtins", _patched_load)
    monkeypatch.setattr(Executor, "_execute", _capture_execute)

    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, UsageUpdate, Error]:
        await bus.subscribe(cls, lambda event, _events=events: _collect(_events, event))

    engine = agent_engine(bus, settings=Settings())
    await engine.start(_CODE)

    await bus.publish(UserMessage(text="do work"))
    assert await wait_until(lambda: bool(_filter(events, Error)))
    await engine.wait_for_run_task()

    events.clear()
    await bus.publish(UserRetry())
    assert await wait_until(lambda: bool(_final_agent_messages(events)))
    await engine.wait_for_run_task()

    assert execute_inputs[-1] == []
    client = captured_client[-1]
    retry_wire_messages = client.call_history[2][0]
    retry_users = [message for message in retry_wire_messages if message.role == "user"]
    assert len(retry_users) == 1
    retry_user_texts = [content.text for content in retry_users[0].contents if content.type == "text"]
    assert retry_user_texts[0] == "do work"
    assert any(text and text.startswith("<system-reminder>") for text in retry_user_texts[1:])

    history_messages = engine._executor.history_state["messages"]
    all_messages = [*history_messages, *(message for call, _options in client.call_history for message in call)]
    assert all(not message.additional_properties.get(HistoryMarkerKind.CONTINUATION_KEY) for message in all_messages)
    assert all(message.role != "user" or (message.text or "").strip() != "continue" for message in all_messages)


# ===========================================================================
# End-to-end tests: verify the LLM actually receives <system-reminder> tags
# ===========================================================================


@pytest.mark.asyncio
async def test_llm_receives_runtime_reminder(tmp_path: Path):
    """The LLM should see a <system-reminder> with runtime info (cwd, shell, time)."""
    middleware = SystemReminderMiddleware(runtime=_direct_runtime(tmp_path))
    middleware.prepare_turn(usage={})

    user_contents = await _direct_llm_user_contents(middleware, "hello")
    assert len(user_contents) >= 2, f"Expected user text + reminder, got: {user_contents}"
    assert user_contents[0] == "hello"

    reminders = [text for text in user_contents if text.startswith("<system-reminder>")]
    assert reminders, f"Expected reminder tag, got: {user_contents}"
    assert any("Working directory" in text or "working directory" in text.lower() for text in reminders)


@pytest.mark.asyncio
async def test_runtime_reminder_renders_safe_copy_of_complete_dynamic_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_cwd = "/work/pro\udcffject"
    raw_extra = "/root/\udcfe"
    raw_label = "ext\udcfdra"
    raw_shell_name = "ba\udcfcsh"
    raw_shell_path = "/bin/ba\udcfbsh"
    raw_extra_shell_path = "/bin/z\udcfash"
    raw_python_path = "/runtime/bin/py\udcf9thon"
    platform = get_platform()
    runtime = SessionEnvironment(
        cwd=raw_cwd,
        platform=replace(
            platform,
            shell=replace(platform.shell, name=raw_shell_name, path=raw_shell_path),
            extra_shells=(replace(platform.shell, name="zsh", path=raw_extra_shell_path),),
        ),
        working_dirs=(WorkingDir(path=raw_extra, label=raw_label),),
    )
    monkeypatch.setattr(
        reminder_module,
        "_format_python_execution_paths_hint",
        lambda: [f"    - your runtime Python: {raw_python_path}"],
    )
    middleware = SystemReminderMiddleware(runtime=runtime, shell_tool_enabled=True)
    middleware.prepare_turn(usage={})

    user_contents = await _direct_llm_user_contents(middleware, "hello")
    payload = "\n".join(user_contents)

    assert runtime.cwd == raw_cwd
    assert runtime.working_dirs[0].path == raw_extra
    raw_values = [
        raw_cwd,
        raw_extra,
        raw_label,
        raw_shell_name,
        raw_shell_path,
        raw_extra_shell_path,
        raw_python_path,
    ]
    assert all(surrogate_safe_text(value) in payload for value in raw_values)
    assert all(value not in payload for value in raw_values)
    payload.encode("utf-8")


@pytest.mark.asyncio
async def test_llm_receives_skill_reference_reminder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent_engine,
):
    """A leading /skill token should queue a skill-reference system reminder."""
    import chrys.orchestration.engine.build.builder as builder_module

    fake_platform = type("P", (), {"config_dir": tmp_path / "chrys-config"})()
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)
    monkeypatch.setattr("chrys.service.skills.adapter.user_agents_dir", lambda: tmp_path / "agents-root")

    captured_client: list[MockChatClient] = []

    def _mock_create_client(s=None, **kw):
        client = MockChatClient(responses=[MockResponse(text="hi")])
        captured_client.append(client)
        return client

    profile = AgentProfile(
        name="Code",
        display_name="Code Agent",
        instructions="You are a coding assistant.",
        tools=ToolsConfig(builtins=[]),
        skills=SkillsConfig(
            inline=[
                SkillConfig(
                    name="review",
                    description="Review code and identify issues",
                    instructions="Review carefully.",
                )
            ],
            auto_load_user_agents_skills=False,
            auto_load_cwd_agents_skills=False,
        ),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
    )

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(bus, settings=Settings())
    await engine.start(profile)

    await bus.publish(UserMessage(text="/review focus on auth"))
    await wait_for(
        lambda: bool(captured_client) and captured_client[-1].call_count >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="model call with live runtime reminder",
    )

    client = captured_client[-1]
    user_contents = _llm_saw_user_contents(client, 0)
    instructions = str(client.call_history[0][1].get("instructions", ""))
    assert "<available_skills>" not in instructions
    reminders = [text for text in user_contents if text.startswith("<system-reminder>")]

    assert any("<available_skills>" in text for text in reminders)
    assert any("<name>review</name>" in text for text in reminders)
    assert any("[Skill Reference] User explicitly invoked a skill: review" in text for text in reminders)
    assert any('skill_name="review"' in text for text in reminders)
    assert user_contents[0] == "/review focus on auth"

    user_msgs = _user_messages(engine)
    assert len(user_msgs) == 1
    assert (user_msgs[0].text or "") == "/review focus on auth"
    assert "<system-reminder>" not in (user_msgs[0].text or "")


@pytest.mark.asyncio
async def test_llm_receives_skill_reference_reminder_for_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent_engine,
):
    """A mid-turn /skill injection should update active-turn reminders."""
    import chrys.orchestration.engine.build.builder as builder_module
    from chrys.kernel import FunctionTool
    from chrys.service.tools.registry import ToolRegistry

    fake_platform = type("P", (), {"config_dir": tmp_path / "chrys-config"})()
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)
    monkeypatch.setattr("chrys.service.skills.adapter.user_agents_dir", lambda: tmp_path / "agents-root")

    captured_client: list[MockChatClient] = []

    def _mock_create_client(s=None, **kw):
        client = MockChatClient(
            responses=[
                MockResponse(tool_calls=[("echo", "c1", {"message": "working"})]),
                MockResponse(text="done"),
            ]
        )
        captured_client.append(client)
        return client

    async def _echo(message: str) -> str:
        await asyncio.sleep(0.25)
        return f"echo: {message}"

    test_tool = FunctionTool(func=_echo, name="echo", description="Echo")

    def _patched_load(self, categories, **kwargs):
        self.register(test_tool)
        return [test_tool]

    profile = AgentProfile(
        name="Code",
        display_name="Code Agent",
        instructions="You are a coding assistant.",
        tools=ToolsConfig(builtins=[]),
        skills=SkillsConfig(
            inline=[
                SkillConfig(
                    name="review",
                    description="Review code and identify issues",
                    instructions="Review carefully.",
                )
            ],
            auto_load_user_agents_skills=False,
            auto_load_cwd_agents_skills=False,
        ),
        approval=ApprovalConfig(default="auto"),
        compaction=CompactionConfig(enabled=False),
    )

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    monkeypatch.setattr(ToolRegistry, "load_builtins", _patched_load)
    bus = EventBus()
    events: list[object] = []
    tool_started = asyncio.Event()
    final_seen = asyncio.Event()

    async def _on_tool_start(event: ToolCallStart) -> None:
        events.append(event)
        tool_started.set()

    async def _on_agent_message(event: AgentMessage) -> None:
        events.append(event)
        if event.is_final and not event.is_intermediate:
            final_seen.set()

    for cls in [SessionReady, UsageUpdate, UserInjectResult]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))
    await bus.subscribe(ToolCallStart, _on_tool_start)
    await bus.subscribe(AgentMessage, _on_agent_message)

    engine = agent_engine(bus, settings=Settings())
    await engine.start(profile)

    await bus.publish(UserMessage(text="do something"))
    await asyncio.wait_for(tool_started.wait(), timeout=3.0)

    await bus.publish(UserInject(text="/review focus on auth"))
    await asyncio.wait_for(final_seen.wait(), timeout=3.0)

    client = captured_client[-1]
    assert client.call_count >= 2

    all_user_msgs = _all_user_msg_texts_in_call(client, 1)
    flattened = [text for user_texts in all_user_msgs for text in user_texts]
    assert any("[Skill Reference] User explicitly invoked a skill: review" in text for text in flattened)
    assert any('skill_name="review"' in text for text in flattened)

    assert "/review focus on auth" in flattened

    messages = engine._executor.history_state.get("messages", [])
    injected = [m for m in messages if m.role == "user" and m.additional_properties.get("_injected", False)]
    assert any((m.text or "") == "/review focus on auth" for m in injected)
    for message in injected:
        assert "<system-reminder>" not in (message.text or "")


@pytest.mark.asyncio
async def test_llm_receives_usage_reminder_on_second_turn(tmp_path: Path):
    """On the second turn, the LLM should see usage info from the first turn."""
    middleware = SystemReminderMiddleware(runtime=_direct_runtime(tmp_path))
    middleware.prepare_turn(usage={})
    turn1_contents = await _direct_llm_user_contents(middleware, "first")
    turn1_reminders = [text for text in turn1_contents if text.startswith("<system-reminder>")]
    assert not any("[Context Usage]" in text for text in turn1_reminders), (
        f"First turn has no prior usage and must not carry a usage reminder: {turn1_reminders}"
    )

    middleware.prepare_turn(usage={"total_token_count": 12_500})
    user_contents = await _direct_llm_user_contents(middleware, "second")
    reminder_texts = [text for text in user_contents if text.startswith("<system-reminder>")]

    usage_reminders = [text for text in reminder_texts if "[Context Usage]" in text]
    assert len(usage_reminders) == 1, f"Expected exactly one usage reminder, got: {reminder_texts}"
    assert "12,500/200,000" in usage_reminders[0]

    assert any("Working directory" in text or "working directory" in text.lower() for text in reminder_texts), (
        f"Expected runtime reminder, got: {reminder_texts}"
    )
    assert user_contents[0] == "second"


@pytest.mark.asyncio
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_llm_receives_profile_switch_reminder(agent_engine, monkeypatch: pytest.MonkeyPatch):
    """After a real AgentProfileSwitch, the LLM sees the switch reminder.

    Kept engine-driven on purpose: it pins the wiring that AgentEngine
    forwards profile switches AND the current tool names into
    SystemReminderMiddleware. Seeding the middleware directly would still
    pass with that wiring broken.
    """
    import chrys.orchestration.engine.build.builder as builder_module
    from chrys.service.tools.registry import ToolRegistry

    captured_clients: list[MockChatClient] = []

    def _mock_create_client(s=None, **kw):
        client = MockChatClient(responses=[MockResponse(text="ok")])
        captured_clients.append(client)
        return client

    from chrys.kernel import FunctionTool

    def _echo(message: str) -> str:
        return f"echo: {message}"

    test_tool = FunctionTool(func=_echo, name="echo", description="Echo")

    def _patched_load(self, categories, **kwargs):
        self.register(test_tool)
        return [test_tool]

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    monkeypatch.setattr(ToolRegistry, "load_builtins", _patched_load)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, ProfileSwitched, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(bus, settings=Settings(), agent_registry=_make_registry())
    await engine.start(_CODE)

    # Chat as Code; drain msg1's turn before switching so it is complete.
    await bus.publish(UserMessage(text="msg1"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="pre-switch persisted user message for provider refresh",
    )
    await engine.wait_for_run_task()

    # Switch to Explore (creates a new client)
    await bus.publish(AgentProfileSwitch(profile_name="Explore"))
    await wait_for(
        lambda: bool(_filter(events, ProfileSwitched)),
        timeout=ENGINE_TURN_TIMEOUT,
        description="profile-switch event for provider refresh",
    )

    # Chat as Explore; wait for the post-switch client to be invoked.
    await bus.publish(UserMessage(text="msg2"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 2 and bool(captured_clients) and captured_clients[-1].call_count >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="post-switch persisted message and model call",
    )
    # call_count increments when the model call STARTS; the profile-switch
    # marker is applied to session state only after executor.run() returns.
    # Drain the turn before inspecting state.
    await engine.wait_for_run_task()

    # The post-switch client is the last one
    post_switch_client = captured_clients[-1]
    assert post_switch_client.call_count >= 1

    user_contents = _llm_saw_user_contents(post_switch_client, 0)
    reminder_texts = [t for t in user_contents if t.startswith("<system-reminder>")]

    # Should contain a switch reminder mentioning both profiles
    switch_reminders = [r for r in reminder_texts if "switched" in r.lower() or "profile" in r.lower()]
    assert len(switch_reminders) >= 1, f"Expected switch reminder, got reminders: {reminder_texts}"

    switch_text = switch_reminders[0]
    assert "Code Agent" in switch_text, f"Expected 'Code Agent' in switch reminder: {switch_text}"
    assert "Explore Agent" in switch_text, f"Expected 'Explore Agent' in switch reminder: {switch_text}"

    # The engine forwarded the post-switch tool names into the middleware.
    assert "Your currently available tools are:" in switch_text, (
        f"Expected the current tool list in switch reminder: {switch_text}"
    )
    assert "echo" in switch_text

    # User text is still first and clean.
    assert user_contents[0] == "msg2"

    # State is clean
    user_msgs = _user_messages(engine)
    assert (user_msgs[1].text or "") == "msg2"
    assert user_msgs[1].additional_properties.get(HistoryMarkerKind.PROFILE_SWITCH_TO_KEY) == "Explore Agent"


@pytest.mark.asyncio
async def test_tool_loop_all_calls_see_enriched_message(agent_engine, monkeypatch: pytest.MonkeyPatch):
    """During a real tool loop, every LLM call should see the <system-reminder> tags.

    Kept as a full-engine test on purpose: it pins the kernel↔ChatMiddleware
    contract that the middleware is re-invoked before EVERY model call of the
    loop — including calls whose context ends with trailing assistant/tool
    messages. Direct middleware invocations cannot detect the kernel skipping
    enrichment after the first call.
    """
    import chrys.orchestration.engine.build.builder as builder_module
    from chrys.service.tools.registry import ToolRegistry

    captured_client: list[MockChatClient] = []

    def _mock_create_client(s=None, **kw):
        client = MockChatClient(
            responses=[
                # LLM call 1: tool call
                MockResponse(tool_calls=[("echo", "c1", {"message": "step1"})]),
                # LLM call 2: another tool call
                MockResponse(tool_calls=[("echo", "c2", {"message": "step2"})]),
                # LLM call 3: final text response
                MockResponse(text="all done"),
            ]
        )
        captured_client.append(client)
        return client

    from chrys.kernel import FunctionTool

    def _echo(message: str) -> str:
        return f"echo: {message}"

    test_tool = FunctionTool(func=_echo, name="echo", description="Echo")

    def _patched_load(self, categories, **kwargs):
        self.register(test_tool)
        return [test_tool]

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    monkeypatch.setattr(ToolRegistry, "load_builtins", _patched_load)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(bus, settings=Settings())
    await engine.start(_CODE)

    await bus.publish(UserMessage(text="do multi-step work"))
    await wait_for(
        lambda: bool(captured_client) and captured_client[-1].call_count >= 3,
        timeout=ENGINE_TURN_TIMEOUT,
        description="three model calls in tool loop",
    )

    client = captured_client[-1]
    # Tool call → tool call → final response = 3 model calls
    assert client.call_count == 3, f"Expected 3 LLM calls, got {client.call_count}"

    # ALL three calls must see the user message with <system-reminder> tags,
    # even though calls 2 and 3 carry trailing assistant/tool messages.
    for i in range(3):
        user_contents = _llm_saw_user_contents(client, i)
        reminder_texts = [t for t in user_contents if t.startswith("<system-reminder>")]
        assert len(reminder_texts) >= 1, f"LLM call {i}: expected at least 1 reminder, got {len(reminder_texts)}"
        assert any("do multi-step work" in t for t in user_contents), (
            f"LLM call {i}: user text missing from: {user_contents}"
        )

    # After all calls, session state should be clean
    user_msgs = _user_messages(engine)
    assert len(user_msgs) >= 1
    assert (user_msgs[0].text or "") == "do multi-step work"


@pytest.mark.asyncio
async def test_tool_loop_uses_stable_turn_reminders_after_profile_switch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Runtime and profile-switch reminders should be stable for a full tool loop."""
    runtime_calls = 0

    def _runtime_hint(_self: SystemReminderMiddleware) -> str:
        nonlocal runtime_calls
        runtime_calls += 1
        return f"[Runtime Environment]\n  Runtime marker: {runtime_calls}"

    monkeypatch.setattr(SystemReminderMiddleware, "_format_runtime_hint", _runtime_hint)
    middleware = SystemReminderMiddleware(
        runtime=_direct_runtime(tmp_path),
        tool_names=["echo", "compress_context", "load_skill", "read_skill_resource", "run_skill_script"],
    )
    middleware.set_profile_switch("Code Agent", "Explore Agent")
    middleware.prepare_turn(usage={})

    first_reminders: list[str] | None = None
    for _ in range(3):
        user_contents = await _direct_llm_user_contents(middleware, "do switched multi-step work")
        reminder_texts = [text for text in user_contents if text.startswith("<system-reminder>")]
        runtime_reminders = [text for text in reminder_texts if "Runtime marker" in text]
        switch_reminders = [text for text in reminder_texts if "switched" in text.lower()]

        assert len(runtime_reminders) == 1
        assert "Runtime marker: 1" in runtime_reminders[0]
        assert len(switch_reminders) == 1
        assert "Code Agent" in switch_reminders[0]
        assert "Explore Agent" in switch_reminders[0]
        assert "Your currently available tools are:" in switch_reminders[0]
        assert "echo" in switch_reminders[0]
        assert "compress_context" in switch_reminders[0]
        assert "load_skill" in switch_reminders[0]
        assert "read_skill_resource" in switch_reminders[0]
        assert "run_skill_script" in switch_reminders[0]
        if first_reminders is None:
            first_reminders = reminder_texts
        else:
            assert reminder_texts == first_reminders

    assert runtime_calls == 1


@pytest.mark.asyncio
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_session_restore_preserves_usage_details(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """Usage details should survive session save/restore and be used in next turn's reminders."""
    import chrys.orchestration.engine.build.builder as builder_module

    captured_clients: list[MockChatClient] = []

    def _mock_create_client(s=None, **kw):
        client = MockChatClient(responses=[MockResponse(text="reply")])
        captured_clients.append(client)
        return client

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    # Phase 1: Create session and send a message (generates usage)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, UsageUpdate, SessionSaved]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    state_store = JsonFileStateStore(tmp_path)
    engine = agent_engine(
        bus,
        settings=Settings(),
        agent_registry=_make_registry(),
        state_store=state_store,
    )
    await engine.start(_CODE)

    await bus.publish(UserMessage(text="initial"))
    await wait_for(
        lambda: engine._runtime_meta.last_usage_details is not None,
        timeout=ENGINE_TURN_TIMEOUT,
        description="persisted runtime usage details",
    )

    session_id = engine._session_id
    # Check that usage_details are set
    assert engine._runtime_meta.last_usage_details is not None

    await engine.shutdown()

    # Phase 2: Restore session on new engine
    bus2 = EventBus()
    events2: list[object] = []
    for cls in [SessionReady, SessionRestored, UsageUpdate]:
        await bus2.subscribe(cls, lambda e, _events=events2: _collect(_events, e))

    engine2 = agent_engine(
        bus2,
        settings=Settings(),
        agent_registry=_make_registry(),
        state_store=state_store,
    )
    await engine2.start(_CODE)

    await bus2.publish(SessionRestore(session_id=session_id))
    await wait_for(
        lambda: isinstance(engine2._runtime_meta.last_usage_details, dict),
        timeout=ENGINE_TURN_TIMEOUT,
        description="restored runtime usage details",
    )

    # Verify usage details were restored
    assert isinstance(engine2._runtime_meta.last_usage_details, dict)

    # Phase 3: Send a new message — should see runtime reminder
    await bus2.publish(UserMessage(text="after restore"))
    await wait_for(
        lambda: captured_clients[-1].call_count >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="post-restore model call with runtime reminder",
    )
    # The reminder is stripped from the stored user message only after the
    # run ends, so the state assertion below needs the turn to be over.
    await engine2.wait_for_run_task()

    # The post-restore client should have received reminders
    post_restore_client = captured_clients[-1]
    assert post_restore_client.call_count >= 1

    user_contents = _llm_saw_user_contents(post_restore_client, post_restore_client.call_count - 1)
    reminder_texts = [t for t in user_contents if t.startswith("<system-reminder>")]

    # Should have at least runtime reminder
    assert len(reminder_texts) >= 1, f"Expected reminders after restore, got: {user_contents}"

    # User message in state should be clean
    user_msgs = _user_messages(engine2)
    last_msg = user_msgs[-1]
    assert (last_msg.text or "") == "after restore"


@pytest.mark.asyncio
async def test_consecutive_switch_llm_sees_merged_reminder(tmp_path: Path):
    """Consecutive switches (A→B→C) should show A→C in the LLM's reminder, not intermediate steps."""
    middleware = SystemReminderMiddleware(runtime=_direct_runtime(tmp_path))
    middleware.set_profile_switch("Code Agent", "Explore Agent")
    middleware.update_profile_switch_to("Docs Agent")
    middleware.prepare_turn(usage={})

    user_contents = await _direct_llm_user_contents(middleware, "q2")
    reminder_texts = [text for text in user_contents if text.startswith("<system-reminder>")]
    switch_reminders = [text for text in reminder_texts if "switched" in text.lower()]
    assert len(switch_reminders) == 1, f"Expected exactly 1 switch reminder, got: {switch_reminders}"

    switch_text = switch_reminders[0]
    assert "Code Agent" in switch_text, f"Expected original 'from' profile: {switch_text}"
    assert "Docs Agent" in switch_text, f"Expected final 'to' profile: {switch_text}"
    assert "Explore Agent" not in switch_text, f"Intermediate profile should not appear in merged switch: {switch_text}"


@pytest.mark.asyncio
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_consecutive_switch_back_to_origin_no_reminder(
    tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch
):
    """Consecutive A→B→A should NOT inject a switch reminder — no net change."""
    import chrys.orchestration.engine.build.builder as builder_module

    captured_clients: list[MockChatClient] = []

    def _mock_create_client(s=None, **kw):
        client = MockChatClient(responses=[MockResponse(text="ok")])
        captured_clients.append(client)
        return client

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, ProfileSwitched, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(
        bus,
        settings=Settings(),
        agent_registry=_make_registry(),
    )
    await engine.start(_CODE)

    # Chat as Code first
    await bus.publish(UserMessage(text="q1"))
    await wait_for(
        lambda: len(_final_agent_messages(events)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="first final agent message before round-trip switch",
    )

    # Consecutive round-trip: Code → Explore → Code
    await bus.publish(AgentProfileSwitch(profile_name="Explore"))
    await wait_for(
        lambda: len(_filter(events, ProfileSwitched)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="outbound round-trip profile-switch event",
    )
    await bus.publish(AgentProfileSwitch(profile_name="Code"))
    await wait_for(
        lambda: len(_filter(events, ProfileSwitched)) >= 2,
        timeout=ENGINE_TURN_TIMEOUT,
        description="returning round-trip profile-switch event",
    )

    # Chat — LLM should NOT see any switch reminder (back to origin)
    await bus.publish(UserMessage(text="q2"))
    await wait_for(
        lambda: len(_final_agent_messages(events)) >= 2,
        timeout=ENGINE_TURN_TIMEOUT,
        description="second final agent message after round-trip switch",
    )

    final_client = captured_clients[-1]
    assert final_client.call_count >= 1

    user_contents = _llm_saw_user_contents(final_client, 0)
    reminder_texts = [t for t in user_contents if t.startswith("<system-reminder>")]
    switch_reminders = [r for r in reminder_texts if "switched" in r.lower()]
    assert len(switch_reminders) == 0, f"Should be no switch reminder for round-trip, got: {switch_reminders}"


@pytest.mark.asyncio
async def test_switch_chat_switch_back_llm_sees_both_reminders(tmp_path: Path):
    """A→B (chat) B→A: LLM should see BOTH switch reminders on the respective turns."""
    middleware = SystemReminderMiddleware(runtime=_direct_runtime(tmp_path))

    middleware.set_profile_switch("Code Agent", "Explore Agent")
    middleware.prepare_turn(usage={})
    user_contents_q2 = await _direct_llm_user_contents(middleware, "q2")
    reminder_q2 = [text for text in user_contents_q2 if "switched" in text.lower()]
    assert len(reminder_q2) == 1, f"Expected 1 switch reminder on q2, got: {reminder_q2}"
    assert "Code Agent" in reminder_q2[0]
    assert "Explore Agent" in reminder_q2[0]

    middleware.set_profile_switch("Explore Agent", "Code Agent")
    middleware.prepare_turn(usage={})
    user_contents_q3 = await _direct_llm_user_contents(middleware, "q3")
    reminder_q3 = [text for text in user_contents_q3 if "switched" in text.lower()]
    assert len(reminder_q3) == 1, f"Expected 1 switch reminder on q3, got: {reminder_q3}"
    assert "Explore Agent" in reminder_q3[0]
    assert "Code Agent" in reminder_q3[0]


@pytest.mark.asyncio
async def test_no_switch_reminder_without_profile_change(tmp_path: Path):
    """Normal messages (no profile switch) should not contain switch reminders."""
    middleware = SystemReminderMiddleware(runtime=_direct_runtime(tmp_path))
    middleware.prepare_turn(usage={})

    user_contents = await _direct_llm_user_contents(middleware, "hello")
    reminder_texts = [text for text in user_contents if text.startswith("<system-reminder>")]
    switch_reminders = [text for text in reminder_texts if "switched" in text.lower()]
    assert len(switch_reminders) == 0, f"Unexpected switch reminders: {switch_reminders}"


# ===========================================================================
# Injection + state cleanliness: verify injected messages are not polluted
# and that ALL state messages are clean after every run
# ===========================================================================


@pytest.mark.asyncio
async def test_injected_message_receives_trailing_reminders_in_llm_view(
    tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch
):
    """The LLM should see injected user text first, followed by trailing reminders.

    InjectionMiddleware (ChatMiddleware) appends clean ``Message("user", [text])``
    inside the tool loop.  SystemReminderMiddleware is the next ChatMiddleware,
    so it defangs all user messages and attaches turn reminders to the last
    user message, which is the injected message for that model call.
    """
    import chrys.orchestration.engine.build.builder as builder_module
    from chrys.service.tools.registry import ToolRegistry

    captured_client: list[MockChatClient] = []

    def _mock_create_client(s=None, **kw):
        client = MockChatClient(
            responses=[
                # LLM call 1: tool call (creates time for injection)
                MockResponse(tool_calls=[("echo", "c1", {"message": "working"})]),
                # LLM call 2: final text (after seeing injection)
                MockResponse(text="done"),
            ]
        )
        captured_client.append(client)
        return client

    from chrys.kernel import FunctionTool

    async def _echo(message: str) -> str:
        await asyncio.sleep(0.25)
        return f"echo: {message}"

    test_tool = FunctionTool(func=_echo, name="echo", description="Echo")

    def _patched_load(self, categories, **kwargs):
        self.register(test_tool)
        return [test_tool]

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    monkeypatch.setattr(ToolRegistry, "load_builtins", _patched_load)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, UsageUpdate, UserInjectResult]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(bus, settings=Settings())
    await engine.start(_CODE)

    # Send initial message
    await bus.publish(UserMessage(text="do something"))
    # Let the run reach the tool loop so the injection lands mid-turn.
    await wait_for(
        lambda: bool(captured_client) and captured_client[-1].call_count >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="initial model call before injected reminder",
    )

    # Inject while tool is running
    await bus.publish(UserInject(text="also check this"))
    await wait_for(
        lambda: bool(captured_client) and captured_client[-1].call_count >= 2,
        timeout=ENGINE_TURN_TIMEOUT,
        description="second model call after injected reminder",
    )

    client = captured_client[-1]

    # LLM call 2 (after tool result + injection): check all user messages.
    assert client.call_count >= 2
    all_user_msgs = _all_user_msg_texts_in_call(client, 1)
    assert len(all_user_msgs) >= 2

    original_user_texts = all_user_msgs[0]
    injected_user_texts = all_user_msgs[-1]
    assert original_user_texts == ["do something"]
    assert injected_user_texts[0] == "also check this"
    assert any(text.startswith("<system-reminder>") for text in injected_user_texts[1:])

    # After run: ALL messages in state are clean
    _assert_all_state_messages_clean(engine)


@pytest.mark.asyncio
async def test_all_state_messages_clean_after_simple_run(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """After a simple single-turn run, every message in state should be clean."""
    import chrys.orchestration.engine.build.builder as builder_module

    def _mock_create_client(s=None, **kw):
        return MockChatClient(responses=[MockResponse(text="response")])

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(bus, settings=Settings())
    await engine.start(_CODE)

    await bus.publish(UserMessage(text="hello"))
    await wait_for(
        lambda: bool(_final_agent_messages(events)),
        timeout=ENGINE_TURN_TIMEOUT,
        description="final agent message for simple clean-state run",
    )

    _assert_all_state_messages_clean(engine)


@pytest.mark.asyncio
async def test_all_state_messages_clean_after_tool_loop(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """After a multi-step tool loop, every message in state should be clean."""
    import chrys.orchestration.engine.build.builder as builder_module
    from chrys.service.tools.registry import ToolRegistry

    def _mock_create_client(s=None, **kw):
        return MockChatClient(
            responses=[
                MockResponse(tool_calls=[("echo", "c1", {"message": "step1"})]),
                MockResponse(tool_calls=[("echo", "c2", {"message": "step2"})]),
                MockResponse(text="all done"),
            ]
        )

    from chrys.kernel import FunctionTool

    def _echo(message: str) -> str:
        return f"echo: {message}"

    test_tool = FunctionTool(func=_echo, name="echo", description="Echo")

    def _patched_load(self, categories, **kwargs):
        self.register(test_tool)
        return [test_tool]

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    monkeypatch.setattr(ToolRegistry, "load_builtins", _patched_load)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(bus, settings=Settings())
    await engine.start(_CODE)

    await bus.publish(UserMessage(text="multi-step"))
    await wait_for(
        lambda: bool(_final_agent_messages(events)),
        timeout=ENGINE_TURN_TIMEOUT,
        description="final agent message for tool-loop clean-state run",
    )

    # Every message in state must be clean — user, assistant (tool calls),
    # tool (results), and final assistant response
    _assert_all_state_messages_clean(engine)


@pytest.mark.asyncio
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_all_state_messages_clean_after_multi_turn(tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch):
    """After multiple turns, every message in state should be clean."""
    import chrys.orchestration.engine.build.builder as builder_module

    def _mock_create_client(s=None, **kw):
        return MockChatClient(responses=[MockResponse(text="ok")])

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(bus, settings=Settings())
    await engine.start(_CODE)

    await bus.publish(UserMessage(text="turn1"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="first clean-state turn user message",
    )
    # Drain each turn before the next publish so every UserMessage starts a
    # fresh turn rather than being injected into a still-RUNNING one.
    await engine.wait_for_run_task()
    await bus.publish(UserMessage(text="turn2"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 2,
        timeout=ENGINE_TURN_TIMEOUT,
        description="second clean-state turn user message",
    )
    await engine.wait_for_run_task()
    await bus.publish(UserMessage(text="turn3"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 3,
        timeout=ENGINE_TURN_TIMEOUT,
        description="third clean-state turn user message",
    )

    _assert_all_state_messages_clean(engine)


@pytest.mark.asyncio
@with_wait_deadline(ENGINE_TEST_WAIT_TIMEOUT)
async def test_all_state_messages_clean_after_profile_switch(
    tmp_path: Path, agent_engine, monkeypatch: pytest.MonkeyPatch
):
    """After a profile switch + message, every message in state should be clean."""
    import chrys.orchestration.engine.build.builder as builder_module

    def _mock_create_client(s=None, **kw):
        return MockChatClient(responses=[MockResponse(text="ok")])

    monkeypatch.setattr(builder_module, "create_client", _mock_create_client)
    bus = EventBus()
    events: list[object] = []
    for cls in [SessionReady, AgentMessage, ProfileSwitched, UsageUpdate]:
        await bus.subscribe(cls, lambda e, _events=events: _collect(_events, e))

    engine = agent_engine(
        bus,
        settings=Settings(),
        agent_registry=_make_registry(),
    )
    await engine.start(_CODE)

    await bus.publish(UserMessage(text="before switch"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="pre-switch clean-state user message",
    )
    # Drain the pre-switch turn so it completes before the switch lands.
    await engine.wait_for_run_task()

    await bus.publish(AgentProfileSwitch(profile_name="Explore"))
    await wait_for(
        lambda: len(_filter(events, ProfileSwitched)) >= 1,
        timeout=ENGINE_TURN_TIMEOUT,
        description="clean-state profile-switch event",
    )

    await bus.publish(UserMessage(text="after switch"))
    await wait_for(
        lambda: len(_user_messages(engine)) >= 2,
        timeout=ENGINE_TURN_TIMEOUT,
        description="post-switch clean-state user message",
    )

    # All messages clean — including the one that carried a switch reminder
    _assert_all_state_messages_clean(engine)


# ===========================================================================
# MCP server instructions reminder (<mcp_instructions>)
# ===========================================================================


@pytest.mark.asyncio
async def test_llm_receives_mcp_instructions_reminder(tmp_path: Path):
    """A configured mcp_instructions block reaches the LLM as a <system-reminder>."""
    block = '<mcp_instructions>\n  <server name="github">\n    Use the GitHub API.\n  </server>\n</mcp_instructions>'
    middleware = SystemReminderMiddleware(runtime=_direct_runtime(tmp_path), mcp_instructions_provider=lambda: block)
    middleware.prepare_turn(usage={})

    user_contents = await _direct_llm_user_contents(middleware, "hello")
    assert user_contents[0] == "hello"
    reminders = [text for text in user_contents if text.startswith("<system-reminder>")]
    assert any(block in text for text in reminders), f"Expected <mcp_instructions> reminder, got: {user_contents}"


@pytest.mark.asyncio
async def test_no_mcp_instructions_reminder_when_unset(tmp_path: Path):
    """Without mcp_instructions no <mcp_instructions> block is injected."""
    middleware = SystemReminderMiddleware(runtime=_direct_runtime(tmp_path))
    middleware.prepare_turn(usage={})

    user_contents = await _direct_llm_user_contents(middleware, "hello")
    assert not any("<mcp_instructions>" in text for text in user_contents)


@pytest.mark.asyncio
async def test_mcp_instructions_follow_skill_catalog(tmp_path: Path):
    """MCP instructions render after the skill catalog in the reminder sequence."""
    catalog = "<available_skills>\n  <name>review</name>\n</available_skills>"
    block = '<mcp_instructions>\n  <server name="fs">\n    Read-only access.\n  </server>\n</mcp_instructions>'
    middleware = SystemReminderMiddleware(
        runtime=_direct_runtime(tmp_path),
        skill_catalog_provider=lambda: catalog,
        mcp_instructions_provider=lambda: block,
    )
    middleware.prepare_turn(usage={})

    user_contents = await _direct_llm_user_contents(middleware, "hello")
    catalog_idx = next(i for i, text in enumerate(user_contents) if catalog in text)
    mcp_idx = next(i for i, text in enumerate(user_contents) if block in text)
    assert catalog_idx < mcp_idx


@pytest.mark.asyncio
async def test_mcp_instructions_refresh_per_turn_and_preserve_on_retry(tmp_path: Path):
    """The provider is re-snapshotted each fresh turn but kept byte-identical on retry."""
    blocks = iter(
        [
            '<mcp_instructions>\n  <server name="a">\n    First.\n  </server>\n</mcp_instructions>',
            '<mcp_instructions>\n  <server name="a">\n    Second.\n  </server>\n</mcp_instructions>',
        ]
    )
    current: list[str] = []

    def _provider() -> str | None:
        current.append(next(blocks, current[-1] if current else ""))
        return current[-1]

    middleware = SystemReminderMiddleware(runtime=_direct_runtime(tmp_path), mcp_instructions_provider=_provider)

    middleware.prepare_turn(usage={})
    first_contents = await _direct_llm_user_contents(middleware, "one")
    assert any("First." in text for text in first_contents)

    # Retry of the same turn must keep the snapshot byte-identical.
    middleware.prepare_turn(usage={}, preserve_turn_reminders=True)
    retry_contents = await _direct_llm_user_contents(middleware, "one")
    assert any("First." in text for text in retry_contents)
    assert not any("Second." in text for text in retry_contents)

    # A fresh turn picks up the provider's current value.
    middleware.prepare_turn(usage={})
    second_contents = await _direct_llm_user_contents(middleware, "two")
    assert any("Second." in text for text in second_contents)
    assert not any("First." in text for text in second_contents)
