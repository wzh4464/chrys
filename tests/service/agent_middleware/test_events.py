# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for core middleware helpers and classes."""

from __future__ import annotations

import asyncio
import sys
import textwrap
from datetime import date
from types import SimpleNamespace

import pytest

from chrys.foundation.tool_call_context import (
    TOOL_CALL_CONTEXT_METADATA_KEY,
    set_tool_context,
    set_tool_context_builder,
)
from chrys.foundation.tool_execution_stamp import (
    EFFECTIVE_ARGS_MAX_CHARS,
    EXECUTION_STAMP_KEY,
    build_execution_stamp,
)
from chrys.foundation.tool_kinds import KIND_SHELL
from chrys.foundation.tool_result_metadata import (
    PROCESS_EXIT_CODE_METADATA_KEY,
    SHELL_EXIT_CODE_METADATA_KEY,
    TOOL_ERROR_CODE_METADATA_KEY,
    TOOL_ERROR_KIND_METADATA_KEY,
    TOOL_ERROR_MESSAGE_METADATA_KEY,
    TOOL_ERROR_RETRYABLE_METADATA_KEY,
    TOOL_FAILED_METADATA_KEY,
    TOOL_RESULT_METADATA_KEY,
)
from chrys.foundation.trajectory.context import trajectory_scope
from chrys.foundation.trajectory.event_types import EventType, ToolOutcome
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.trajectory.metadata import OPERATION_ID_KEY
from chrys.foundation.trajectory_timing import TRAJECTORY_TIMING_KEY
from chrys.kernel._result_ceiling import apply_result_ceiling
from chrys.service.agent_middleware import (
    IntermediateTextBuffer,
    InterruptMiddleware,
    SubAgentEventMiddleware,
    SubAgentStatsMiddleware,
    ToolEventMiddleware,
    _read_file_snapshot,
    _read_shell_snapshots,
    extract_result_images,
    extract_result_text,
)
from chrys.service.agent_middleware._metadata_keys import (
    _APPROVAL_REJECTED_KEY,
    _REJECTION_MESSAGE_KEY,
    _REJECTION_SOURCE_KEY,
    _TOOL_INVOCATION_ORDER_KEY,
)
from chrys.service.agent_middleware.events import sub_agent_events as sub_agent_events_module
from chrys.service.agent_middleware.events import tool_events as tool_events_module
from chrys.service.agent_middleware.events.hook_dispatch import get_tool_invocation_order
from chrys.service.agent_middleware.events.result_persistence import persistable_result_metadata
from chrys.service.mutations.types import (
    FileHashDiff,
    FileMutation,
    FileMutationTextSnapshot,
    MutationOp,
    MutationSource,
)
from chrys.service.session.sub_agent_logs import SubAgentLogStats
from tests.service.trajectory._fakes import CancelAckSink, FakeSink, make_context
from tests.support.waiting import wait_until

# ──────────────── extract_result_text ──────────────────────────────────


def test_persistable_result_metadata_preserves_explicit_falsey_error_fields() -> None:
    metadata = {
        PROCESS_EXIT_CODE_METADATA_KEY: 0,
        TOOL_ERROR_RETRYABLE_METADATA_KEY: False,
        TOOL_ERROR_CODE_METADATA_KEY: 0,
        TOOL_FAILED_METADATA_KEY: False,
        "unrelated": False,
    }

    assert persistable_result_metadata(metadata) == {
        PROCESS_EXIT_CODE_METADATA_KEY: 0,
        TOOL_ERROR_RETRYABLE_METADATA_KEY: False,
        TOOL_ERROR_CODE_METADATA_KEY: 0,
        TOOL_FAILED_METADATA_KEY: False,
    }


def test_extract_result_none() -> None:
    assert extract_result_text(None) == ""


def test_extract_result_string() -> None:
    assert extract_result_text("hello") == "hello"


def test_extract_result_list_with_text() -> None:
    items = [SimpleNamespace(text="line1", result=None), SimpleNamespace(text="line2", result=None)]
    assert extract_result_text(items) == "line1\nline2"


def test_extract_result_list_with_result_attr() -> None:
    # Mirrors ``Content.from_function_result(..., result="output")``, which
    # leaves ``text`` unset (None) and populates ``result`` with the joined
    # text from the items list.
    items = [SimpleNamespace(text=None, result="output")]
    assert extract_result_text(items) == "output"


def test_extract_result_list_of_strings() -> None:
    assert extract_result_text(["a", "b"]) == "a\nb"


def test_extract_result_list_empty() -> None:
    # Items with no recognizable text-bearing attribute and no other
    # extractable shape contribute nothing; the joined result is empty.
    items = [SimpleNamespace(text="", result="")]
    assert extract_result_text(items) == ""


def test_extract_result_list_with_output_attr() -> None:
    # ``output`` is the text-bearing attr on Content(type="mcp_server_tool_result").
    items = [SimpleNamespace(text=None, result=None, output="payload")]
    assert extract_result_text(items) == "payload"


def test_extract_result_list_uri_content() -> None:
    # Image / blob / resource Content items (text=None) get a placeholder
    # without leaking base64 data into event reprs or ACP raw_output.
    items = [SimpleNamespace(text=None, result=None, uri="data:image/png;base64,AAA", media_type="image/png")]
    assert extract_result_text(items) == "[image/png image]"


def test_extract_result_images_from_direct_content_list() -> None:
    from chrys.kernel import Content

    image = Content.from_uri("data:image/png;base64,AAA", media_type="image/png")
    audio = Content.from_uri("data:audio/wav;base64,AAA", media_type="audio/wav")

    assert extract_result_images([Content.from_text("caption"), image, audio]) == [image]


def test_extract_result_images_from_function_result_items() -> None:
    from chrys.kernel import Content

    image = Content.from_uri("data:image/png;base64,AAA", media_type="image/png")
    result = Content.from_function_result("call_1", result=[Content.from_text("caption"), image])

    assert extract_result_images(result) == [image]


def test_extract_result_images_ignores_unknown_uri_without_media_type() -> None:
    from chrys.kernel import Content

    unknown = Content.from_uri("https://example.com/blob")

    assert extract_result_images([unknown]) == []


# Same shapes again, but constructed via real Chrys ``Content`` factories —
# ``Content`` always has structural fields like ``type`` and
# ``additional_properties``, which would otherwise sneak through the
# JSON-dump fallback.  These tests pin down the contract for the actual
# objects produced by the MCP adapter.


def test_extract_result_real_content_text() -> None:
    from chrys.kernel import Content

    items = [Content.from_text("hello")]
    assert extract_result_text(items) == "hello"


def test_extract_result_real_content_text_empty() -> None:
    """An empty-string text item must render as "", not as ``{"type": "text", ...}``."""
    from chrys.kernel import Content

    items = [Content.from_text("")]
    assert extract_result_text(items) == ""


def test_extract_result_real_content_function_result_none() -> None:
    """``Content.from_function_result(..., result=None)`` builds a single empty-text item."""
    from chrys.kernel import Content

    items = [Content.from_function_result(call_id="call_1", result=None)]
    assert extract_result_text(items) == ""


def test_extract_result_real_content_data_uri() -> None:
    """Image data items render as a short placeholder, never base64 or a class repr."""
    from chrys.kernel import Content

    items = [Content.from_data(data=b"\x89PNG", media_type="image/png")]
    out = extract_result_text(items)
    assert out == "[image/png image]"
    assert " object at " not in out


def test_extract_result_other_type() -> None:
    assert extract_result_text(42) == "42"


# ────── Non-string text-bearing attributes (don't render as obj repr) ──────


def test_extract_result_mcp_server_tool_result_list_output() -> None:
    """Anthropic-managed MCP servers surface tool results as
    ``Content.from_mcp_server_tool_result(output=list[Content])``.  The
    ``output`` slot is typed ``Any``, so a naive ``str(output)`` renders
    as a list of object reprs.  We must
    recurse into the inner Contents so the actual text reaches the LLM.
    """
    from chrys.kernel import Content

    inner = [Content.from_text("actual response"), Content.from_text("line 2")]
    items = [Content("mcp_server_tool_result", call_id="c1", output=inner)]
    out = extract_result_text(items)
    assert "actual response" in out
    assert "line 2" in out
    assert " object at " not in out, f"raw object repr leaked: {out!r}"


def test_extract_result_function_result_list_result() -> None:
    """Same pattern for ``Content`` with ``result=list[Content]`` — recurse
    rather than stringifying the list.
    """
    from chrys.kernel import Content

    items = [Content("function_result", call_id="c1", result=[Content.from_text("inner-text")])]
    out = extract_result_text(items)
    assert out == "inner-text"
    assert " object at " not in out


def test_extract_result_error_message_dict_renders_as_json() -> None:
    """Structured error payloads (``message=dict``) must render as JSON
    so the LLM sees ``{"code": 500, ...}`` instead of Python's
    ``{'code': 500, ...}`` repr (subtly different — single vs double
    quotes — and not parseable as JSON downstream).
    """
    from chrys.kernel import Content

    items = [Content("error", message={"code": 500, "body": "Server Error"})]
    out = extract_result_text(items)
    assert out == '{"code": 500, "body": "Server Error"}'


def test_extract_result_mcp_output_dict_renders_as_json() -> None:
    """MCP servers may return structured JSON in ``output`` directly."""
    from chrys.kernel import Content

    items = [Content("mcp_server_tool_result", call_id="c1", output={"result": "ok", "status": 200})]
    out = extract_result_text(items)
    assert out == '{"result": "ok", "status": 200}'


# ──────────── Unrecognized Content fallback (treated as error) ──────────


def test_extract_result_unrecognized_content_returns_error_prefix(caplog: pytest.LogCaptureFixture) -> None:
    """A Content with no text/result/output/message/uri (e.g. ``from_function_call``)
    is an unexpected shape in a tool result list — render it as an
    ``"Error: …"`` string and log a WARNING so the LLM and TUI both
    see it as a failure rather than a silent JSON dump.
    """
    import logging

    from chrys.kernel import Content

    items = [Content.from_function_call(call_id="c1", name="some_tool", arguments={"k": "v"})]

    with caplog.at_level(logging.WARNING, logger="chrys.foundation.io.result_content"):
        out = extract_result_text(items)

    assert out.startswith("Error: "), f"expected error prefix, got {out!r}"
    assert "function_call" in out, "type hint should be present in the error string"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Unrecognized tool result Content" in r.getMessage() for r in warnings), (
        "expected a WARNING log for the unrecognized content shape"
    )


def test_extract_result_mixed_recognized_and_unrecognized_promotes_to_error() -> None:
    """If any item triggers the unrecognized fallback, the joined result
    must surface as an error so the TUI styling and the chrys
    ``Error: …`` convention apply to the whole string, not just the
    middle of it.
    """
    from chrys.kernel import Content

    items = [
        Content.from_text("normal output"),
        Content.from_function_call(call_id="c1", name="oops"),
    ]
    out = extract_result_text(items)
    assert out.startswith("Error: "), f"expected error prefix, got {out!r}"
    assert "normal output" in out, "recognized item content should still appear in the joined result"


def test_extract_result_debug_log_truncates_large_strings(caplog: pytest.LogCaptureFixture) -> None:
    """The DEBUG dump must summarize, not embed, large string fields.

    Otherwise an MCP server returning a big base64 ``data:`` URI would
    leak its full payload into log files.
    """
    import logging

    from chrys.kernel import Content

    big = "data:image/png;base64," + ("A" * 10_000)
    items = [Content.from_data(data=b"\x89PNG", media_type="image/png")]
    items[0].uri = big  # type: ignore[attr-defined]

    with caplog.at_level(logging.DEBUG, logger="chrys.foundation.io.result_content"):
        extract_result_text(items)

    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert big not in log_text, "Full URI must not appear in DEBUG log"
    assert "<str len=" in log_text, "Long string fields should be summarized as <str len=…>"


def test_extract_result_debug_log_truncates_plain_string_items(caplog: pytest.LogCaptureFixture) -> None:
    """Plain ``str`` items in the result list must also be summarized in DEBUG.

    ``extract_result_text`` accepts ``list[str]`` directly, so a 10 KB
    string in the list would otherwise leak into the log via the
    non-``__dict__`` branch of the shape summarizer.
    """
    import logging

    big = "X" * 10_000
    with caplog.at_level(logging.DEBUG, logger="chrys.foundation.io.result_content"):
        extract_result_text([big])

    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert big not in log_text, "Full plain-string item must not appear in DEBUG log"
    assert "<str len=" in log_text, "Long plain-string items should be summarized as <str len=…>"


def test_extract_result_coroutine_returns_empty() -> None:
    """Dangling coroutine objects (from cancelled tool calls) should not leak their repr."""

    async def _dummy() -> str:
        return "never"

    coro = _dummy()
    result = extract_result_text(coro)
    assert result == ""
    # Coroutine should be closed to prevent ResourceWarning
    # (calling close() on an already-closed coroutine is a no-op)
    coro.close()


async def test_extract_result_awaitable_returns_empty() -> None:
    """Any awaitable (not just coroutines) should be handled safely."""
    import asyncio

    fut = asyncio.get_running_loop().create_future()
    result = extract_result_text(fut)
    assert result == ""
    fut.cancel()  # Clean up


# ──────────────── IntermediateTextBuffer ────────────────────────────────


def test_buffer_starts_empty() -> None:
    buf = IntermediateTextBuffer()
    assert buf.drain() == []
    assert buf.batch_id == 0


def test_buffer_store_and_drain() -> None:
    buf = IntermediateTextBuffer()
    buf.store("hello")
    buf.store("world")
    assert buf.drain() == ["hello", "world"]
    # Second drain is empty
    assert buf.drain() == []


def test_buffer_batch_id_increments_on_store() -> None:
    buf = IntermediateTextBuffer()
    assert buf.batch_id == 0
    buf.store("text")
    assert buf.batch_id == 1
    buf.store("more")
    assert buf.batch_id == 2


def test_buffer_new_batch_increments_without_text() -> None:
    buf = IntermediateTextBuffer()
    buf.new_batch()
    assert buf.batch_id == 1
    buf.new_batch()
    assert buf.batch_id == 2
    # No text was stored
    assert buf.drain() == []


# ──────────────── InterruptMiddleware ───────────────────────────────────


async def test_interrupt_not_set_passes_through() -> None:
    mw = InterruptMiddleware()
    called = False

    async def call_next() -> None:
        nonlocal called
        called = True

    # Minimal FunctionInvocationContext mock
    ctx = SimpleNamespace()
    await mw.process(ctx, call_next)
    assert called


async def test_interrupt_set_raises() -> None:
    mw = InterruptMiddleware()
    mw.set_interrupted()

    async def call_next() -> None:
        pass

    ctx = SimpleNamespace()
    with pytest.raises(Exception, match="interrupted"):
        await mw.process(ctx, call_next)


async def test_interrupt_reset_clears_flag() -> None:
    mw = InterruptMiddleware()
    mw.set_interrupted()
    mw.reset()

    called = False

    async def call_next() -> None:
        nonlocal called
        called = True

    ctx = SimpleNamespace()
    await mw.process(ctx, call_next)
    assert called


# ──────────────── ToolEventMiddleware — CancelledError handling ────────


async def test_tool_event_middleware_cancelled_no_result_event() -> None:
    """CancelledError during tool execution must not publish ToolCallResult.

    When the agent task is cancelled by user interrupt, ToolEventMiddleware
    should skip ToolCallResult publishing so the TUI's cancel_running_tools()
    is the sole authority on widget state.
    """
    import asyncio

    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ToolCallResult, ToolCallStart

    bus = EventBus()
    mw = ToolEventMiddleware(bus, session_id="test")

    # Collect published events
    events: list = []

    async def _on_start(e: ToolCallStart) -> None:
        events.append(("start", e))

    async def _on_result(e: ToolCallResult) -> None:
        events.append(("result", e))

    await bus.subscribe(ToolCallStart, _on_start)
    await bus.subscribe(ToolCallResult, _on_result)

    async def call_next_cancelled() -> None:
        raise asyncio.CancelledError()

    ctx = SimpleNamespace(
        function=SimpleNamespace(name="test_tool", chrys_kind=None),
        arguments={"arg": "val"},
        result=None,
        metadata=None,
    )

    with pytest.raises(asyncio.CancelledError):
        await mw.process(ctx, call_next_cancelled)

    # ToolCallStart should be published, but ToolCallResult should NOT
    start_events = [e for e in events if e[0] == "start"]
    result_events = [e for e in events if e[0] == "result"]
    assert len(start_events) == 1
    assert len(result_events) == 0
    timing = ctx.metadata[TRAJECTORY_TIMING_KEY]
    assert timing["started_at"] <= timing["finished_at"]
    assert timing["duration_ms"] >= 0


async def test_tool_event_middleware_sets_invocation_order_before_awaited_start_handler() -> None:
    """Parallel tool calls keep function-call order even if the first start publish blocks."""
    import asyncio

    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ToolCallStart

    bus = EventBus()
    mw = ToolEventMiddleware(bus, session_id="test")
    first_start_seen = asyncio.Event()
    release_first_start = asyncio.Event()

    async def _on_start(event: ToolCallStart) -> None:
        if event.tool_name == "first":
            first_start_seen.set()
            await release_first_start.wait()

    await bus.subscribe(ToolCallStart, _on_start)

    async def _next() -> None:
        return None

    first_ctx = SimpleNamespace(
        function=SimpleNamespace(name="first", chrys_kind=None),
        arguments={},
        result=None,
        metadata={},
    )
    second_ctx = SimpleNamespace(
        function=SimpleNamespace(name="second", chrys_kind=None),
        arguments={},
        result=None,
        metadata={},
    )

    first_task = asyncio.create_task(mw.process(first_ctx, _next))
    await first_start_seen.wait()
    second_task = asyncio.create_task(mw.process(second_ctx, _next))
    await asyncio.sleep(0)

    assert get_tool_invocation_order(first_ctx) == 0
    assert get_tool_invocation_order(second_ctx) == 1

    release_first_start.set()
    await asyncio.gather(first_task, second_task)


async def test_intermediate_text_delivered_before_any_parallel_tool_start() -> None:
    """The batch's intermediate text must beat every sibling's ToolCallStart.

    Parallel tool calls run ``process`` concurrently; whichever sibling
    drains the buffer suspends inside the AgentMessage publish while bus
    handlers run. Without cross-sibling serialization the other siblings'
    starts overtake the text, land mid-batch on the TUI, and split the
    tool group — orphaning already-rendered sub-agent cards.
    """
    import asyncio

    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import AgentMessage, ToolCallStart

    bus = EventBus()
    buf = IntermediateTextBuffer()
    buf.store("I'll launch two explore agents.")
    mw = ToolEventMiddleware(bus, session_id="test", intermediate_buffer=buf)

    delivered: list[str] = []
    text_publish_blocked = asyncio.Event()
    release_text_publish = asyncio.Event()

    async def _on_text(event: AgentMessage) -> None:
        text_publish_blocked.set()
        await asyncio.wait_for(release_text_publish.wait(), timeout=5)
        delivered.append("text")

    async def _on_start(event: ToolCallStart) -> None:
        delivered.append(f"start:{event.tool_name}")

    await bus.subscribe(AgentMessage, _on_text)
    await bus.subscribe(ToolCallStart, _on_start)

    async def _next() -> None:
        return None

    first_ctx = SimpleNamespace(
        function=SimpleNamespace(name="first", chrys_kind=None),
        arguments={},
        result=None,
        metadata={},
    )
    second_ctx = SimpleNamespace(
        function=SimpleNamespace(name="second", chrys_kind=None),
        arguments={},
        result=None,
        metadata={},
    )

    first_task = asyncio.create_task(mw.process(first_ctx, _next))
    await asyncio.wait_for(text_publish_blocked.wait(), timeout=5)
    second_task = asyncio.create_task(mw.process(second_ctx, _next))
    # Bounded negative assertion: while the batch text is still in flight
    # nothing may be delivered. Pre-fix, the sibling's start overtakes the
    # suspended publish as soon as it is scheduled; polling with real sleeps
    # guarantees it gets those chances before the window closes.
    assert not await wait_until(lambda: delivered, timeout=0.3)
    release_text_publish.set()
    await asyncio.gather(first_task, second_task)

    assert delivered[0] == "text"
    assert sorted(delivered[1:]) == ["start:first", "start:second"]


async def test_tool_event_middleware_uses_approval_modified_args_for_start_event() -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ToolCallStart
    from chrys.service.agent_middleware import _APPROVAL_MODIFIED_ARGS_KEY

    bus = EventBus()
    mw = ToolEventMiddleware(bus, session_id="test")
    starts: list[ToolCallStart] = []

    async def _on_start(event: ToolCallStart) -> None:
        starts.append(event)

    await bus.subscribe(ToolCallStart, _on_start)

    async def _next() -> None:
        return None

    ctx = SimpleNamespace(
        function=SimpleNamespace(name="explore_agent", chrys_kind="sub_agent"),
        arguments={"prompt": "old"},
        result=None,
        metadata={_APPROVAL_MODIFIED_ARGS_KEY: {"prompt": "new"}},
    )

    await mw.process(ctx, _next)

    assert len(starts) == 1
    assert starts[0].args == {"prompt": "new"}


async def test_tool_event_middleware_sub_agent_metadata_uses_provider_and_event_ids() -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ToolCallResult
    from chrys.foundation.util.sub_agent_context import sub_agent_parent_result_metadata
    from chrys.service.tools.result_metadata import record_tool_success

    bus = EventBus()
    mw = ToolEventMiddleware(bus, session_id="test")
    results: list[ToolCallResult] = []

    async def _on_result(event: ToolCallResult) -> None:
        results.append(event)

    await bus.subscribe(ToolCallResult, _on_result)

    async def _next() -> None:
        holder = sub_agent_parent_result_metadata.get()
        assert holder is not None
        assert holder.parent_provider_call_id == "provider-call"
        assert holder.parent_event_call_id
        assert holder.parent_event_call_id != holder.parent_provider_call_id
        holder.sub_agent_invocation_id = "a1b2c3d4e5f6"
        holder.sub_agent_log_file = "Explore_a1b2c3d4e5f6.json"
        record_tool_success()
        ctx.result = "done"

    ctx = SimpleNamespace(
        function=SimpleNamespace(name="Explore", chrys_kind="sub_agent"),
        arguments={"prompt": "inspect"},
        result=None,
        metadata={"call_id": "provider-call"},
    )

    await mw.process(ctx, _next)

    assert len(results) == 1
    assert results[0].call_id != "provider-call"
    assert results[0].metadata[TOOL_FAILED_METADATA_KEY] is False
    assert results[0].metadata["sub_agent_invocation_id"] == "a1b2c3d4e5f6"
    assert results[0].metadata["sub_agent_log_file"] == "Explore_a1b2c3d4e5f6.json"
    assert ctx.metadata[TOOL_RESULT_METADATA_KEY] == {
        TOOL_FAILED_METADATA_KEY: False,
        "sub_agent_invocation_id": "a1b2c3d4e5f6",
        "sub_agent_log_file": "Explore_a1b2c3d4e5f6.json",
    }


@pytest.mark.parametrize("exit_code", [0, 42])
async def test_tool_event_middleware_publishes_and_persists_shell_exit_metadata(exit_code: int) -> None:
    """Shell backend metadata should survive ToolCallResult and replay persistence."""
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ToolCallResult
    from chrys.service.tools.builtins.shell import shell_result_metadata

    bus = EventBus()
    mw = ToolEventMiddleware(bus, session_id="test")
    results: list[ToolCallResult] = []

    async def _on_result(event: ToolCallResult) -> None:
        results.append(event)

    await bus.subscribe(ToolCallResult, _on_result)

    async def _next() -> None:
        metadata = shell_result_metadata.get()
        assert metadata is not None
        metadata[SHELL_EXIT_CODE_METADATA_KEY] = exit_code
        ctx.result = f"boom\n[exit_code: {exit_code}]"

    ctx = SimpleNamespace(
        function=SimpleNamespace(name="bash", chrys_kind=KIND_SHELL),
        arguments={"command": "sleep 999"},
        result=None,
        metadata={"call_id": "provider-shell-call", _TOOL_INVOCATION_ORDER_KEY: 3},
    )

    await mw.process(ctx, _next)

    assert results[0].metadata[SHELL_EXIT_CODE_METADATA_KEY] == exit_code
    # The upstream-seeded invocation ordinal is honored, and the persistable
    # subset rides the invocation context for the kernel fold.
    assert ctx.metadata[_TOOL_INVOCATION_ORDER_KEY] == 3
    assert ctx.metadata[TOOL_RESULT_METADATA_KEY] == {SHELL_EXIT_CODE_METADATA_KEY: exit_code}


async def test_tool_event_middleware_publishes_and_persists_generic_failure_metadata() -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ToolCallResult
    from chrys.service.tools.result_metadata import tool_error

    bus = EventBus()
    mw = ToolEventMiddleware(bus, session_id="test")
    results: list[ToolCallResult] = []

    async def _on_result(event: ToolCallResult) -> None:
        results.append(event)

    await bus.subscribe(ToolCallResult, _on_result)

    async def _next() -> None:
        ctx.result = tool_error("validation", "bad input")

    ctx = SimpleNamespace(
        function=SimpleNamespace(name="read_file", chrys_kind="filesystem.read"),
        arguments={"path": "missing.txt"},
        result=None,
        metadata={},
    )

    await mw.process(ctx, _next)

    assert results[0].result == "Error: bad input"
    assert results[0].metadata[TOOL_FAILED_METADATA_KEY] is True
    assert results[0].metadata[TOOL_ERROR_KIND_METADATA_KEY] == "validation"
    assert results[0].metadata[TOOL_ERROR_MESSAGE_METADATA_KEY] == "bad input"
    carried = ctx.metadata[TOOL_RESULT_METADATA_KEY]
    assert carried[TOOL_FAILED_METADATA_KEY] is True
    assert carried[TOOL_ERROR_KIND_METADATA_KEY] == "validation"


async def test_tool_event_middleware_hook_denial_is_not_public_approval_rejection() -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ToolCallResult
    from chrys.service.hooks.events import HookEvent
    from chrys.service.hooks.schema import HookDecision

    class FakeHookManager:
        def __init__(self) -> None:
            self.after_payloads: list[dict[str, object]] = []

        def has_hooks_for(self, event: HookEvent) -> bool:
            return event in {HookEvent.BEFORE_TOOL_CALL, HookEvent.AFTER_TOOL_CALL}

        async def fire(self, event: HookEvent, payload: dict[str, object], **_kwargs: object) -> HookDecision:
            if event == HookEvent.BEFORE_TOOL_CALL:
                return HookDecision(blocked=True, block_reason="blocked by policy")
            self.after_payloads.append(payload)
            return HookDecision()

    bus = EventBus()
    results: list[ToolCallResult] = []

    async def _on_result(event: ToolCallResult) -> None:
        results.append(event)

    await bus.subscribe(ToolCallResult, _on_result)
    manager = FakeHookManager()
    mw = ToolEventMiddleware(bus, session_id="test", hook_manager=manager, profile_name="Code")
    ctx = SimpleNamespace(
        function=SimpleNamespace(name="write_file", chrys_kind="filesystem.write"),
        arguments={"path": "a.txt", "content": "x"},
        result=None,
        metadata={},
    )

    async def _next() -> None:
        raise AssertionError("blocked hook should prevent the tool body")

    await mw.process(ctx, _next)

    assert results[0].result == "Error: blocked by policy"
    assert results[0].metadata[TOOL_FAILED_METADATA_KEY] is True
    assert results[0].metadata[TOOL_ERROR_KIND_METADATA_KEY] == "hook_denied"
    assert results[0].metadata[TOOL_ERROR_MESSAGE_METADATA_KEY] == "blocked by policy"
    assert "approval" not in results[0].metadata
    assert manager.after_payloads[0]["result"] == {
        "text": "Error: blocked by policy",
        "duration_ms": 0,
        "error": False,
        "failed": True,
        "approval_rejected": True,
        "rejection_source": "hook",
        "hook_denied": True,
    }


async def test_tool_event_middleware_user_rejection_public_metadata_is_persisted() -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ToolCallResult

    bus = EventBus()
    results: list[ToolCallResult] = []

    async def _on_result(event: ToolCallResult) -> None:
        results.append(event)

    await bus.subscribe(ToolCallResult, _on_result)
    mw = ToolEventMiddleware(bus, session_id="test")
    ctx = SimpleNamespace(
        function=SimpleNamespace(name="write_file", chrys_kind="filesystem.write"),
        arguments={"path": "a.txt", "content": "x"},
        result=None,
        metadata={
            "call_id": "provider-write-call",
            _APPROVAL_REJECTED_KEY: True,
            _REJECTION_SOURCE_KEY: "user",
            _REJECTION_MESSAGE_KEY: "Denied by operator",
            _TOOL_INVOCATION_ORDER_KEY: 7,
        },
    )

    async def _next() -> None:
        ctx.result = "Error: Denied by operator"

    await mw.process(ctx, _next)

    assert results[0].metadata[TOOL_FAILED_METADATA_KEY] is True
    assert results[0].metadata["approval"] == "user_rejected"
    assert results[0].metadata[TOOL_ERROR_KIND_METADATA_KEY] == "approval_rejected"
    assert results[0].metadata[TOOL_ERROR_MESSAGE_METADATA_KEY] == "Denied by operator"
    assert ctx.metadata[_TOOL_INVOCATION_ORDER_KEY] == 7
    assert ctx.metadata[TOOL_RESULT_METADATA_KEY] == {
        TOOL_FAILED_METADATA_KEY: True,
        "approval": "user_rejected",
        TOOL_ERROR_KIND_METADATA_KEY: "approval_rejected",
        TOOL_ERROR_MESSAGE_METADATA_KEY: "Denied by operator",
    }


async def test_sub_agent_stats_middleware_counts_completed_tools_without_event_bus() -> None:
    stats = SubAgentLogStats()
    mw = SubAgentStatsMiddleware(stats)
    ctx = SimpleNamespace(function=SimpleNamespace(name="read_file"), arguments={}, result=None, metadata={})

    async def _ok() -> None:
        ctx.result = "done"

    await mw.process(ctx, _ok)

    assert stats.tool_call_count == 1

    async def _cancel() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await mw.process(ctx, _cancel)
    assert stats.tool_call_count == 1


async def test_tool_event_middleware_exception_publishes_result() -> None:
    """Regular exceptions should still publish ToolCallResult with error text."""
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ToolCallResult, ToolCallStart

    bus = EventBus()
    mw = ToolEventMiddleware(bus, session_id="test")

    events: list = []

    async def _on_start(e: ToolCallStart) -> None:
        events.append(("start", e))

    async def _on_result(e: ToolCallResult) -> None:
        events.append(("result", e))

    await bus.subscribe(ToolCallStart, _on_start)
    await bus.subscribe(ToolCallResult, _on_result)

    async def call_next_error() -> None:
        raise ValueError("tool failed")

    ctx = SimpleNamespace(
        function=SimpleNamespace(name="test_tool", chrys_kind=None),
        arguments={},
        result=None,
        metadata=None,
    )

    with pytest.raises(ValueError, match="tool failed"):
        await mw.process(ctx, call_next_error)

    result_events = [e for e in events if e[0] == "result"]
    assert len(result_events) == 1
    assert "tool failed" in result_events[0][1].result
    assert result_events[0][1].metadata["errored"] is True


async def test_tool_event_middleware_empty_exception_message_publishes_readable_error() -> None:
    """Transport exceptions with blank ``str(exc)`` still need visible tool output."""
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ToolCallResult

    class ReadTimeout(Exception):
        pass

    bus = EventBus()
    mw = ToolEventMiddleware(bus, session_id="test")
    results: list[ToolCallResult] = []

    async def _on_result(e: ToolCallResult) -> None:
        results.append(e)

    await bus.subscribe(ToolCallResult, _on_result)

    async def call_next_error() -> None:
        raise ReadTimeout(TimeoutError())

    ctx = SimpleNamespace(
        function=SimpleNamespace(name="test_tool", chrys_kind=None),
        arguments={},
        result=None,
        metadata=None,
    )

    with pytest.raises(ReadTimeout):
        await mw.process(ctx, call_next_error)

    assert len(results) == 1
    assert results[0].result == "Error: Read timed out (ReadTimeout)"


async def test_tool_event_middleware_hooks_modify_args_before_start_event(tmp_path) -> None:
    """Hook-modified args should be what UI events and the tool body see."""
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ToolCallStart
    from chrys.service.hooks.events import HookEvent
    from chrys.service.hooks.manager import HookManager
    from chrys.service.hooks.schema import HookConfig, HookExecution, HookMatch, HookRun, HooksFile

    script = textwrap.dedent(
        """
        import json, os
        with open(os.environ["CHRYS_HOOK_RESULT"], "w") as f:
            json.dump({"action": "modify", "args_override": {"path": "/tmp/safer"}}, f)
        """
    )
    manager = HookManager(
        file=HooksFile(
            hooks=[
                HookConfig(
                    id="rewrite",
                    event=HookEvent.BEFORE_TOOL_CALL,
                    run=HookRun(type="command", argv=[sys.executable, "-c", script]),
                    execution=HookExecution(mode="blocking"),
                    match=HookMatch(tool_name="write_file"),
                )
            ]
        ),
        hooks_dir=tmp_path / "hooks",
    )
    bus = EventBus()
    starts: list[ToolCallStart] = []

    async def _on_start(event: ToolCallStart) -> None:
        starts.append(event)

    await bus.subscribe(ToolCallStart, _on_start)
    mw = ToolEventMiddleware(bus, session_id="s1", hook_manager=manager, profile_name="Code")
    ctx = SimpleNamespace(
        function=SimpleNamespace(name="write_file", chrys_kind="filesystem.write"),
        arguments={"path": "/tmp/original", "content": "x"},
        metadata={},
        result=None,
    )

    async def call_next() -> None:
        assert ctx.arguments["path"] == "/tmp/safer"
        ctx.result = "ok"

    await mw.process(ctx, call_next)

    assert ctx.arguments["path"] == "/tmp/safer"
    assert ctx.arguments["content"] == "x"
    assert starts[0].args["path"] == "/tmp/safer"
    assert starts[0].args["content"] == "x"


async def test_tool_event_middleware_file_tool_missing_path_does_not_keyerror(tmp_path) -> None:
    """Hook rewrites can remove a usable lock path; middleware should still emit events."""
    from chrys.foundation.events.bus import EventBus
    from chrys.service.hooks.events import HookEvent
    from chrys.service.hooks.manager import HookManager
    from chrys.service.hooks.schema import HookConfig, HookExecution, HookMatch, HookRun, HooksFile

    script = textwrap.dedent(
        """
        import json, os
        with open(os.environ["CHRYS_HOOK_RESULT"], "w") as f:
            json.dump({"action": "modify", "args_override": {"path": None}}, f)
        """
    )
    manager = HookManager(
        file=HooksFile(
            hooks=[
                HookConfig(
                    id="remove-path",
                    event=HookEvent.BEFORE_TOOL_CALL,
                    run=HookRun(type="command", argv=[sys.executable, "-c", script]),
                    execution=HookExecution(mode="blocking"),
                    match=HookMatch(tool_name="write_file"),
                )
            ]
        ),
        hooks_dir=tmp_path / "hooks",
    )
    mw = ToolEventMiddleware(EventBus(), session_id="s1", hook_manager=manager, profile_name="Code")
    ctx = SimpleNamespace(
        function=SimpleNamespace(name="write_file", chrys_kind="filesystem.write"),
        arguments={"path": "/tmp/original", "content": "x"},
        metadata={},
        result=None,
    )

    async def call_next() -> None:
        ctx.result = "ok"

    await mw.process(ctx, call_next)
    assert ctx.arguments["path"] is None


async def test_tool_event_middleware_after_hook_extra_context_appends_to_text_result(tmp_path) -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import ToolCallResult
    from chrys.service.hooks.events import HookEvent
    from chrys.service.hooks.manager import HookManager
    from chrys.service.hooks.schema import HookConfig, HookExecution, HookRun, HooksFile

    script = textwrap.dedent(
        """
        import json, os
        with open(os.environ["CHRYS_HOOK_RESULT"], "w") as f:
            json.dump({"extra_context": "hook note"}, f)
        """
    )
    manager = HookManager(
        file=HooksFile(
            hooks=[
                HookConfig(
                    id="note",
                    event=HookEvent.AFTER_TOOL_CALL,
                    run=HookRun(type="command", argv=[sys.executable, "-c", script]),
                    execution=HookExecution(mode="blocking"),
                )
            ]
        ),
        hooks_dir=tmp_path / "hooks",
    )
    bus = EventBus()
    results: list[ToolCallResult] = []

    async def _on_result(event: ToolCallResult) -> None:
        results.append(event)

    await bus.subscribe(ToolCallResult, _on_result)
    mw = ToolEventMiddleware(bus, session_id="s1", hook_manager=manager, profile_name="Code")
    ctx = SimpleNamespace(
        function=SimpleNamespace(name="read_file", chrys_kind="filesystem.read"),
        arguments={"path": "/tmp/x"},
        metadata={},
        result=None,
    )

    async def call_next() -> None:
        ctx.result = "base result"

    await mw.process(ctx, call_next)

    assert ctx.result == "base result\n\nhook note"
    assert results[0].result == "base result\n\nhook note"


@pytest.mark.asyncio
async def test_failed_tool_call_merges_after_and_error_hook_decisions() -> None:
    from chrys.service.agent_middleware.events.hook_dispatch import fire_after_tool_hooks
    from chrys.service.hooks.events import HookEvent
    from chrys.service.hooks.schema import HookDecision

    class _DualHookManager:
        def __init__(self) -> None:
            self.events: list[HookEvent] = []

        def has_hooks_for(self, event: HookEvent) -> bool:
            return event in {HookEvent.AFTER_TOOL_CALL, HookEvent.TOOL_ERROR}

        async def fire(self, event: HookEvent, _payload: dict[str, object], **_kwargs: object) -> HookDecision:
            self.events.append(event)
            if event is HookEvent.AFTER_TOOL_CALL:
                return HookDecision(
                    blocked=True,
                    block_reason="after blocked",
                    args_override={"shared": "after", "after": 1},
                    system_reminders=["after reminder"],
                    extra_context=["after context"],
                )
            return HookDecision(
                blocked=False,
                block_reason="error did not block",
                args_override={"shared": "error", "error": 2},
                system_reminders=["error reminder"],
                extra_context=["error context"],
            )

    manager = _DualHookManager()
    decision = await fire_after_tool_hooks(
        manager=manager,  # type: ignore[arg-type]
        session_id="s1",
        profile_name="Code",
        tool_name="read_file",
        kind="filesystem.read",
        call_id="call-1",
        args={"path": "/tmp/x"},
        result_text="Error: failed",
        duration_ms=12,
        errored=True,
        failed=True,
        approval_rejected=False,
        rejection_source="",
        workspace_cwd="/tmp",
    )

    assert manager.events == [HookEvent.AFTER_TOOL_CALL, HookEvent.TOOL_ERROR]
    assert decision.blocked is True
    assert decision.block_reason == "after blocked"
    assert decision.args_override == {"shared": "error", "after": 1, "error": 2}
    assert decision.system_reminders == ["after reminder", "error reminder"]
    assert decision.extra_context == ["after context", "error context"]


def test_append_extra_context_updates_content_list_result() -> None:
    from chrys.kernel import Content
    from chrys.service.agent_middleware.events.hook_dispatch import append_extra_context_to_result

    result, result_text = append_extra_context_to_result(
        [Content.from_text("base result")],
        "base result",
        ["hook note"],
    )

    assert result_text == "base result\n\nhook note"
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[1].text == "\nhook note"
    assert extract_result_text(result) == result_text


# ──────────────── _read_file_snapshot ─────────────────────────────────


def _make_mutation(
    path: str = "/tmp/f.py",
    op: MutationOp = MutationOp.MODIFY,
    before_hash: str | None = "aaa",
    after_hash: str | None = "bbb",
    source: MutationSource = MutationSource.SHELL,
) -> FileMutation:
    return FileMutation(
        path=path,
        operation=op,
        source=source,
        tool_call_id="call-1",
        timestamp=0.0,
        before_hash=before_hash,
        after_hash=after_hash,
    )


def _make_tracker(blobs: dict[str, bytes]) -> SimpleNamespace:
    """Return a minimal tracker-like object with a fake store."""
    store = SimpleNamespace(read_blob=blobs.get)
    return SimpleNamespace(store=store)


def test_read_file_snapshot_basic() -> None:
    tracker = _make_tracker({"aaa": b"before content", "bbb": b"after content"})
    mutation = _make_mutation()
    result = _read_file_snapshot(tracker, mutation)
    assert result == ("before content", "after content")


def test_read_file_snapshot_no_before_hash() -> None:
    tracker = _make_tracker({"bbb": b"after"})
    mutation = _make_mutation(before_hash=None)
    result = _read_file_snapshot(tracker, mutation)
    assert result == ("", "after")


def test_read_file_snapshot_no_after_hash() -> None:
    tracker = _make_tracker({"aaa": b"before"})
    mutation = _make_mutation(after_hash=None)
    result = _read_file_snapshot(tracker, mutation)
    assert result == ("before", "")


def test_read_file_snapshot_both_none() -> None:
    tracker = _make_tracker({})
    mutation = _make_mutation(before_hash=None, after_hash=None)
    result = _read_file_snapshot(tracker, mutation)
    assert result == ("", "")


def test_read_file_snapshot_blob_missing_returns_empty() -> None:
    tracker = _make_tracker({})  # no blobs at all
    mutation = _make_mutation(before_hash="missing", after_hash="also_missing")
    result = _read_file_snapshot(tracker, mutation)
    assert result == ("", "")


@pytest.mark.asyncio
async def test_finalize_mutation_tracking_reports_file_operation(tmp_path) -> None:
    from chrys.service.mutations.pipeline import MutationContext, finalize_mutation_tracking
    from chrys.service.mutations.store import SnapshotStore
    from chrys.service.mutations.tracker import MutationTracker

    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")
    tracker = MutationTracker(SnapshotStore(tmp_path))
    tracker.start_turn(1)
    mutation = tracker.record(str(file_path), MutationOp.MODIFY, MutationSource.EDIT_FILE, "call-1")
    assert mutation is not None

    file_path.write_text("new content", encoding="utf-8")
    result = await finalize_mutation_tracking(tracker, MutationContext(file_mutation=mutation), "call-1")

    assert result.file_snapshot == ("", "new content")
    assert result.file_operation == "modify"
    assert result.file_bytes_changed is True
    assert result.file_hashes == FileHashDiff(before=mutation.before_hash, after=mutation.after_hash)


# ──────────────── _read_shell_snapshots ───────────────────────────────


def test_read_shell_snapshots_single_mutation() -> None:
    tracker = _make_tracker({"aaa": b"old", "bbb": b"new"})
    mutations = [_make_mutation(path="/tmp/x.py")]
    result = _read_shell_snapshots(tracker, mutations)
    assert result == {
        "/tmp/x.py": FileMutationTextSnapshot(
            before_text="old",
            after_text="new",
            operation="modify",
            bytes_changed=True,
            source="shell",
            before_hash="aaa",
            after_hash="bbb",
            provenance="assumed",
        )
    }


def test_read_shell_snapshots_collapse_multi_mutations() -> None:
    """Multiple mutations to the same path should keep the first before_text."""
    tracker = _make_tracker({"h1": b"original", "h2": b"middle", "h3": b"final"})
    mutations = [
        _make_mutation(path="/tmp/x.py", before_hash="h1", after_hash="h2"),
        _make_mutation(path="/tmp/x.py", before_hash="h2", after_hash="h3"),
    ]
    result = _read_shell_snapshots(tracker, mutations)
    assert result["/tmp/x.py"] == FileMutationTextSnapshot(
        before_text="original",
        after_text="final",
        operation="modify",
        bytes_changed=True,
        source="shell",
        before_hash="h1",
        after_hash="h3",
        provenance="assumed",
    )


def test_read_shell_snapshots_preserves_first_op() -> None:
    """Collapsed mutations should keep the operation from the first mutation."""
    tracker = _make_tracker({"h1": b"", "h2": b"v1", "h3": b"v2"})
    mutations = [
        _make_mutation(path="/tmp/x.py", op=MutationOp.CREATE, before_hash="h1", after_hash="h2"),
        _make_mutation(path="/tmp/x.py", op=MutationOp.MODIFY, before_hash="h2", after_hash="h3"),
    ]
    result = _read_shell_snapshots(tracker, mutations)
    assert result["/tmp/x.py"].operation == "create"


def test_read_shell_snapshots_collapse_preserves_implicit_source() -> None:
    """A collapsed path should stay marked implicit if any mutation was implicit."""
    tracker = _make_tracker({"h1": b"original", "h2": b"middle", "h3": b"final"})
    mutations = [
        _make_mutation(path="/tmp/x.py", before_hash="h1", after_hash="h2", source=MutationSource.SHELL),
        _make_mutation(path="/tmp/x.py", before_hash="h2", after_hash="h3", source=MutationSource.IMPLICIT),
    ]
    result = _read_shell_snapshots(tracker, mutations)
    assert result["/tmp/x.py"].source == "implicit"


def test_read_shell_snapshots_multiple_paths() -> None:
    tracker = _make_tracker({"a1": b"A-old", "a2": b"A-new", "b1": b"B-old", "b2": b"B-new"})
    mutations = [
        _make_mutation(path="/tmp/a.py", before_hash="a1", after_hash="a2"),
        _make_mutation(path="/tmp/b.py", before_hash="b1", after_hash="b2"),
    ]
    result = _read_shell_snapshots(tracker, mutations)
    assert len(result) == 2
    assert result["/tmp/a.py"] == FileMutationTextSnapshot(
        before_text="A-old",
        after_text="A-new",
        operation="modify",
        bytes_changed=True,
        source="shell",
        before_hash="a1",
        after_hash="a2",
        provenance="assumed",
    )
    assert result["/tmp/b.py"] == FileMutationTextSnapshot(
        before_text="B-old",
        after_text="B-new",
        operation="modify",
        bytes_changed=True,
        source="shell",
        before_hash="b1",
        after_hash="b2",
        provenance="assumed",
    )


def test_read_shell_snapshots_empty_list() -> None:
    tracker = _make_tracker({})
    result = _read_shell_snapshots(tracker, [])
    assert result == {}


# ──────────────── File lock serialization ───────────────────────────────


@pytest.mark.asyncio
async def test_file_lock_serializes_concurrent_edits(tmp_path) -> None:
    """Concurrent edit_file calls to the same file are serialized by file lock.

    Without the lock, asyncio.gather dispatches both record() calls before
    either record_after() runs, breaking the _last_known_hash chain so
    mutation 2 gets the wrong before_hash.  The lock ensures sequential
    execution: record+edit+record_after for edit 1 completes before edit 2
    starts.
    """
    import asyncio

    from chrys.service.mutations.store import SnapshotStore
    from chrys.service.mutations.tracker import MutationTracker
    from chrys.service.mutations.types import MutationOp, MutationSource

    f = tmp_path / "test.py"
    f.write_text("original", encoding="utf-8")

    tracker = MutationTracker(SnapshotStore(tmp_path))
    tracker.start_turn(1)

    async def simulate_edit(content: str, call_id: str) -> None:
        """Simulate an edit_file tool call under the file lock."""
        async with tracker.get_file_lock(str(f)):
            m = tracker.record(str(f), MutationOp.MODIFY, MutationSource.EDIT_FILE, call_id)
            # Simulate async tool execution with a yield point
            await asyncio.sleep(0)
            f.write_text(content, encoding="utf-8")
            tracker.record_after(m)

    # Run two edits concurrently (as asyncio.gather would in the framework)
    await asyncio.gather(
        simulate_edit("v2", "c1"),
        simulate_edit("v3", "c2"),
    )

    turn = tracker.current_turn
    assert len(turn.mutations) == 2
    m1, m2 = turn.mutations

    # Key assertion: the incremental before_hash chain is correct
    # m1.after == m2.before (not m1.before == m2.before which is the bug)
    assert m1.after_hash == m2.before_hash, "Lock should serialize mutations so m2.before_hash reflects m1's result"


@pytest.mark.asyncio
async def test_tool_event_middleware_serializes_implicit_windows(tmp_path) -> None:
    """Concurrent shell calls should not overlap their before/after mutation windows."""
    import asyncio

    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.tool_kinds import KIND_SHELL
    from chrys.service.mutations.store import SnapshotStore
    from chrys.service.mutations.tracker import MutationTracker

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    tracker = MutationTracker(SnapshotStore(session_dir))
    tracker.start_turn(1)
    mw = ToolEventMiddleware(
        EventBus(),
        session_id="s1",
        mutation_tracker=tracker,
        workspace_cwd=str(tmp_path),
        serialize_implicit_windows=True,
    )

    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()
    active = 0
    max_active = 0

    async def first_next() -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        first_entered.set()
        await release_first.wait()
        active -= 1

    async def second_next() -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        second_entered.set()
        active -= 1

    first_ctx = SimpleNamespace(
        function=SimpleNamespace(name="bash", chrys_kind=KIND_SHELL),
        arguments={"command": "true"},
        result=None,
        metadata={},
    )
    second_ctx = SimpleNamespace(
        function=SimpleNamespace(name="bash", chrys_kind=KIND_SHELL),
        arguments={"command": "true"},
        result=None,
        metadata={},
    )

    first_task = asyncio.create_task(mw.process(first_ctx, first_next))
    await first_entered.wait()
    second_task = asyncio.create_task(mw.process(second_ctx, second_next))
    await asyncio.sleep(0.05)

    assert not second_entered.is_set()
    release_first.set()
    await asyncio.gather(first_task, second_task)
    assert second_entered.is_set()
    assert max_active == 1


@pytest.mark.asyncio
async def test_tool_event_middleware_runs_implicit_windows_in_parallel_by_default(tmp_path) -> None:
    """Shell calls keep the historical parallel behavior unless serialization is enabled."""
    import asyncio

    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.tool_kinds import KIND_SHELL
    from chrys.service.mutations.store import SnapshotStore
    from chrys.service.mutations.tracker import MutationTracker

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    tracker = MutationTracker(SnapshotStore(session_dir))
    tracker.start_turn(1)
    mw = ToolEventMiddleware(EventBus(), session_id="s1", mutation_tracker=tracker, workspace_cwd=str(tmp_path))

    first_entered = asyncio.Event()
    second_entered = asyncio.Event()
    release_both = asyncio.Event()
    active = 0
    max_active = 0

    async def first_next() -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        first_entered.set()
        await release_both.wait()
        active -= 1

    async def second_next() -> None:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        second_entered.set()
        await release_both.wait()
        active -= 1

    first_ctx = SimpleNamespace(
        function=SimpleNamespace(name="bash", chrys_kind=KIND_SHELL),
        arguments={"command": "true"},
        result=None,
        metadata={},
    )
    second_ctx = SimpleNamespace(
        function=SimpleNamespace(name="bash", chrys_kind=KIND_SHELL),
        arguments={"command": "true"},
        result=None,
        metadata={},
    )

    first_task = asyncio.create_task(mw.process(first_ctx, first_next))
    await first_entered.wait()
    second_task = asyncio.create_task(mw.process(second_ctx, second_next))
    await asyncio.wait_for(second_entered.wait(), timeout=0.5)

    release_both.set()
    await asyncio.gather(first_task, second_task)
    assert max_active == 2


# ──────────────── provenance context carriage on the invocation context ────


def _provenance_function(name: str = "load_skill") -> SimpleNamespace:
    from chrys.foundation.tool_call_context import set_tool_context_builder

    fn = SimpleNamespace(name=name, chrys_kind="skill")
    set_tool_context_builder(fn, lambda args: {"skill_name": str(args.get("skill_name", "")).lower()})
    return fn


async def test_tool_event_middleware_carries_context_built_from_final_args() -> None:
    """The builder must see approval-modified args, not the stale captured ones."""
    from chrys.foundation.events.bus import EventBus
    from chrys.service.agent_middleware import _APPROVAL_MODIFIED_ARGS_KEY

    mw = ToolEventMiddleware(EventBus(), session_id="test")
    ctx = SimpleNamespace(
        function=_provenance_function(),
        arguments={"skill_name": "STALE"},
        result=None,
        metadata={_APPROVAL_MODIFIED_ARGS_KEY: {"skill_name": "MODIFIED"}},
    )

    async def _next() -> None:
        ctx.result = "ok"

    await mw.process(ctx, _next)

    assert ctx.metadata[TOOL_CALL_CONTEXT_METADATA_KEY] == {"skill_name": "modified"}


async def test_tool_event_middleware_rejected_call_still_carries_context() -> None:
    """Rejected calls never execute, but the model still sees the call — provenance must persist."""
    from chrys.foundation.events.bus import EventBus

    mw = ToolEventMiddleware(EventBus(), session_id="test")
    ctx = SimpleNamespace(
        function=_provenance_function(),
        arguments={"skill_name": "PDF"},
        result="Rejected by user.",
        metadata={_APPROVAL_REJECTED_KEY: True, _REJECTION_SOURCE_KEY: "user"},
    )

    async def _blocked_next() -> None:
        return None

    await mw.process(ctx, _blocked_next)

    assert ctx.metadata[TOOL_CALL_CONTEXT_METADATA_KEY] == {"skill_name": "pdf"}


async def test_sub_agent_event_middleware_resolves_context_from_modified_args() -> None:
    """The shared final_tool_args helper fixes the sub-agent stale-args path too."""
    from chrys.foundation.events.bus import EventBus
    from chrys.service.agent_middleware import _APPROVAL_MODIFIED_ARGS_KEY
    from chrys.service.agent_middleware.events.sub_agent_events import SubAgentEventMiddleware

    mw = SubAgentEventMiddleware(EventBus(), agent_name="Explore", invocation_id="inv123")
    ctx = SimpleNamespace(
        function=_provenance_function(),
        arguments={"skill_name": "STALE"},
        result=None,
        metadata={_APPROVAL_MODIFIED_ARGS_KEY: {"skill_name": "MODIFIED"}},
    )

    async def _next() -> None:
        ctx.result = "ok"

    await mw.process(ctx, _next)
    await mw.flush_progress()

    assert ctx.metadata[TOOL_CALL_CONTEXT_METADATA_KEY] == {"skill_name": "modified"}


# ──────────────── ToolEventMiddleware — batch records (§2.1.1) ─────────


def _batch_record_ctx(provider_call_id: str, tool_name: str = "echo") -> SimpleNamespace:
    return SimpleNamespace(
        function=SimpleNamespace(name=tool_name, chrys_kind=None),
        arguments={"message": "hi"},
        result=None,
        metadata={"call_id": provider_call_id},
    )


async def test_tool_event_middleware_mainline_batch_record_has_non_empty_provider_id() -> None:
    """POSITIVE producer pin: a normal middleware-dispatched call always records
    a non-empty provider_call_id — the persist-side drop-without-provider-id
    guard must never fire on the mainline path.
    """
    from chrys.foundation.events.bus import EventBus

    buffer = IntermediateTextBuffer()
    buffer.new_batch()
    mw = ToolEventMiddleware(EventBus(), session_id="test", intermediate_buffer=buffer)
    ctx = _batch_record_ctx("provider-batch-call")

    async def _next() -> None:
        ctx.result = "ok"

    await mw.process(ctx, _next)

    records = mw.drain_batch_records()
    assert len(records) == 1
    assert records[0].provider_call_id == "provider-batch-call"
    assert records[0].provider_call_id  # never empty on the mainline path
    assert records[0].tool_name == "echo"
    assert records[0].batch_id == 1
    # drain clears
    assert mw.drain_batch_records() == []


async def test_tool_event_middleware_records_batch_record_pre_execution() -> None:
    """A cancelled (interrupted) call still gets its record — recording happens
    before the tool runs, so no result/metadata is required.
    """
    import asyncio

    from chrys.foundation.events.bus import EventBus

    buffer = IntermediateTextBuffer()
    buffer.new_batch()
    mw = ToolEventMiddleware(EventBus(), session_id="test", intermediate_buffer=buffer)
    ctx = _batch_record_ctx("provider-cancelled-call")

    async def _next() -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await mw.process(ctx, _next)

    records = mw.drain_batch_records()
    assert len(records) == 1
    assert records[0].provider_call_id == "provider-cancelled-call"
    assert records[0].batch_id == 1


async def test_tool_event_middleware_metadata_less_call_still_records_batch_record() -> None:
    """A call with no persistable result metadata keeps its batch record —
    records must never be derived from the tool-result metadata stream.
    """
    from chrys.foundation.events.bus import EventBus

    buffer = IntermediateTextBuffer()
    buffer.new_batch()
    mw = ToolEventMiddleware(EventBus(), session_id="test", intermediate_buffer=buffer)
    ctx = _batch_record_ctx("provider-plain-call")

    async def _next() -> None:
        return None  # result stays None; nothing persistable

    await mw.process(ctx, _next)

    assert TOOL_RESULT_METADATA_KEY not in ctx.metadata
    records = mw.drain_batch_records()
    assert len(records) == 1
    assert records[0].provider_call_id == "provider-plain-call"


async def test_clear_batch_records_drops_rolled_back_attempt_records() -> None:
    """Retry rollback (restore_history_snapshot) clears records so a stale
    provider id can never stamp post-retry history.
    """
    from chrys.foundation.events.bus import EventBus

    buffer = IntermediateTextBuffer()
    buffer.new_batch()
    mw = ToolEventMiddleware(EventBus(), session_id="test", intermediate_buffer=buffer)
    ctx = _batch_record_ctx("provider-rolled-back-call")

    async def _next() -> None:
        ctx.result = "ok"

    await mw.process(ctx, _next)
    mw.clear_batch_records()

    assert mw.drain_batch_records() == []


def _execution_stamp_middleware(kind: str, *, tool_result_ceiling_tokens: int | None = None):
    from chrys.foundation.events.bus import EventBus

    if kind == "main":
        return ToolEventMiddleware(
            EventBus(), session_id="stamp-test", tool_result_ceiling_tokens=tool_result_ceiling_tokens
        )
    return SubAgentEventMiddleware(
        EventBus(),
        agent_name="Explore",
        invocation_id="inv-stamp",
        tool_result_ceiling_tokens=tool_result_ceiling_tokens,
    )


@pytest.mark.parametrize("kind", ["main", "sub_agent"])
async def test_event_middlewares_stamp_success(kind: str) -> None:
    middleware = _execution_stamp_middleware(kind)
    context = SimpleNamespace(
        function=SimpleNamespace(name="echo", chrys_kind=None),
        arguments={"b": 2, "a": 1},
        result=None,
        metadata={},
    )

    async def _next() -> None:
        context.result = "done"

    await middleware.process(context, _next)

    stamp = context.metadata[EXECUTION_STAMP_KEY]
    assert stamp["effective_args"] == '{"a":1,"b":2}'
    assert stamp["outcome"] == "ok"
    assert "error_kind" not in stamp
    timing = context.metadata[TRAJECTORY_TIMING_KEY]
    assert timing["started_at"] <= timing["finished_at"]
    assert isinstance(timing["duration_ms"], int)
    assert timing["duration_ms"] >= 0


@pytest.mark.parametrize("kind", ["main", "sub_agent"])
async def test_event_middlewares_stamp_rejection_as_instant(kind: str) -> None:
    middleware = _execution_stamp_middleware(kind)
    context = SimpleNamespace(
        function=SimpleNamespace(name="write_file", chrys_kind="filesystem.write"),
        arguments={"path": "a.txt", "content": "x"},
        result=None,
        metadata={_APPROVAL_REJECTED_KEY: True},
    )

    async def _next() -> None:
        context.result = "Error: rejected"

    await middleware.process(context, _next)

    timing = context.metadata[TRAJECTORY_TIMING_KEY]
    assert timing["started_at"] == timing["finished_at"]
    assert timing["duration_ms"] == 0


@pytest.mark.parametrize("kind", ["main", "sub_agent"])
async def test_event_middlewares_record_the_wait_a_rejection_ended(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The card shows no duration for a call that never ran; the log keeps the wait."""
    module = tool_events_module if kind == "main" else sub_agent_events_module
    # Two reads: the start of the call itself and the finish. Replace the
    # module's own ``time`` name,
    # never the stdlib module: the event loop reads ``time.monotonic`` too, and
    # a fixed clock stalls it.
    clock = iter([100.0, 112.5])
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    middleware = _execution_stamp_middleware(kind)
    context = SimpleNamespace(
        function=SimpleNamespace(name="write_file", chrys_kind="filesystem.write"),
        arguments={"path": "a.txt", "content": "x"},
        result=None,
        metadata={_APPROVAL_REJECTED_KEY: True},
    )

    async def _next() -> None:
        context.result = "Error: rejected"

    sink = FakeSink()
    with trajectory_scope(make_context(sink)):
        await middleware.process(context, _next)

    assert context.metadata[TRAJECTORY_TIMING_KEY]["duration_ms"] == 0
    finished = sink.only(EventType.TOOL_OPERATION_FINISHED)
    assert finished.payload["outcome"] == ToolOutcome.REJECTED
    assert finished.payload["duration_ms"] == 12500


@pytest.mark.parametrize("kind", ["main", "sub_agent"])
async def test_event_middlewares_close_an_operation_abandoned_before_its_start_marker(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The kernel counts the call dispatched before this middleware can open the
    operation, and the preprocessing in between — hooks, the mutation lock, the
    start event on the bus — is cancellable. An interrupt there must still leave
    the minted operation accounted for."""
    module = tool_events_module if kind == "main" else sub_agent_events_module

    async def _cancelled(**_kwargs: object) -> bool:
        raise asyncio.CancelledError

    monkeypatch.setattr(module, "apply_before_tool_hooks", _cancelled)
    middleware = _execution_stamp_middleware(kind)
    operation_id = new_analytics_id()
    context = SimpleNamespace(
        function=SimpleNamespace(name="write_file", chrys_kind="filesystem.write"),
        arguments={"path": "a.txt", "content": "x"},
        result=None,
        metadata={OPERATION_ID_KEY: operation_id},
    )

    async def _next() -> None:
        context.result = "ok"

    sink = FakeSink()
    with trajectory_scope(make_context(sink)), pytest.raises(asyncio.CancelledError):
        await middleware.process(context, _next)

    tool_started = sink.only(EventType.TOOL_OPERATION_STARTED)
    assert tool_started.operation_id == operation_id
    preparation = sink.only(EventType.PREPARATION_STARTED)
    preparation_finished = sink.only(EventType.PREPARATION_FINISHED)
    assert preparation.payload["scope"] == "tool_preamble"
    assert preparation.payload["target_operation_id"] == operation_id
    finished = sink.only(EventType.TOOL_OPERATION_FINISHED)
    assert finished.operation_id == operation_id
    assert finished.payload["outcome"] == ToolOutcome.INTERRUPTED
    assert finished.payload["abandoned"] is True
    assert finished.payload["duration_ms"] == 0
    assert tool_started.links[0].target_operation_id == preparation.operation_id
    assert preparation.monotonic_ns <= preparation_finished.monotonic_ns <= tool_started.monotonic_ns
    assert tool_started.monotonic_ns <= finished.monotonic_ns
    sink.assert_operations_settled()


@pytest.mark.parametrize("kind", ["main", "sub_agent"])
async def test_parallel_tool_preambles_pair_with_their_own_tool_operations(kind: str) -> None:
    """A concurrent batch keeps every preamble link attached to its own tool."""
    middleware = _execution_stamp_middleware(kind)
    operation_ids = [new_analytics_id() for _ in range(4)]
    entered = 0
    all_entered = asyncio.Event()

    async def invoke(operation_id: str) -> None:
        context = SimpleNamespace(
            function=SimpleNamespace(name=f"tool_{operation_id[:4]}", chrys_kind=None),
            arguments={},
            result=None,
            metadata={OPERATION_ID_KEY: operation_id},
        )

        async def _next() -> None:
            nonlocal entered
            entered += 1
            if entered == len(operation_ids):
                all_entered.set()
            await all_entered.wait()
            context.result = "ok"

        await middleware.process(context, _next)

    sink = FakeSink()
    with trajectory_scope(make_context(sink)):
        await asyncio.gather(*(invoke(operation_id) for operation_id in operation_ids))

    preambles = {
        str(draft.payload["target_operation_id"]): draft.operation_id
        for draft in sink.of_type(EventType.PREPARATION_STARTED)
        if draft.payload["scope"] == "tool_preamble"
    }
    tool_starts = {draft.operation_id: draft for draft in sink.of_type(EventType.TOOL_OPERATION_STARTED)}
    assert set(preambles) == set(tool_starts) == set(operation_ids)
    for operation_id, tool_start in tool_starts.items():
        caused_by = [link.target_operation_id for link in tool_start.links if link.relation == "caused_by"]
        assert caused_by == [preambles[operation_id]]
    sink.assert_operations_settled()


@pytest.mark.parametrize("kind", ["main", "sub_agent"])
async def test_event_middlewares_close_an_operation_that_failed_before_its_start_marker(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same window, ordinary exception: the operation still owes a terminal."""
    module = tool_events_module if kind == "main" else sub_agent_events_module

    async def _raises(**_kwargs: object) -> bool:
        raise RuntimeError("hook dispatch blew up")

    monkeypatch.setattr(module, "apply_before_tool_hooks", _raises)
    middleware = _execution_stamp_middleware(kind)
    operation_id = new_analytics_id()
    context = SimpleNamespace(
        function=SimpleNamespace(name="write_file", chrys_kind="filesystem.write"),
        arguments={"path": "a.txt", "content": "x"},
        result=None,
        metadata={OPERATION_ID_KEY: operation_id},
    )

    async def _next() -> None:
        context.result = "ok"

    sink = FakeSink()
    with trajectory_scope(make_context(sink)), pytest.raises(RuntimeError):
        await middleware.process(context, _next)

    assert sink.only(EventType.TOOL_OPERATION_STARTED).operation_id == operation_id
    finished = sink.only(EventType.TOOL_OPERATION_FINISHED)
    assert finished.payload["outcome"] == ToolOutcome.ERRORED
    assert finished.payload["abandoned"] is True
    sink.assert_operations_settled()


@pytest.mark.parametrize("kind", ["main", "sub_agent"])
async def test_event_middlewares_close_an_operation_interrupted_in_its_start_marker(kind: str) -> None:
    """The start marker awaits its write ack, and the line lands even when that
    wait is cancelled — so the operation it opened has to be closed here."""
    middleware = _execution_stamp_middleware(kind)
    context = SimpleNamespace(
        function=SimpleNamespace(name="write_file", chrys_kind="filesystem.write"),
        arguments={"path": "a.txt", "content": "x"},
        result=None,
        metadata={},
    )

    async def _next() -> None:
        context.result = "ok"

    # Preparation start/finish land first; cancellation hits tool start's ack.
    sink = CancelAckSink(at=3)
    with trajectory_scope(make_context(sink)), pytest.raises(asyncio.CancelledError):
        await middleware.process(context, _next)

    assert sink.only(EventType.TOOL_OPERATION_STARTED)
    assert sink.only(EventType.TOOL_OPERATION_FINISHED).payload["outcome"] == ToolOutcome.INTERRUPTED


@pytest.mark.parametrize("kind", ["main", "sub_agent"])
async def test_event_middlewares_close_an_operation_interrupted_after_the_call_ran(kind: str) -> None:
    """The tool already ran; an interrupt while closing it out must not leave
    the operation open."""
    middleware = _execution_stamp_middleware(kind)
    context = SimpleNamespace(
        function=SimpleNamespace(name="write_file", chrys_kind="filesystem.write"),
        arguments={"path": "a.txt", "content": "x"},
        result=None,
        metadata={},
    )

    async def _next() -> None:
        context.result = "ok"

    # Preparation start/finish precede tool start; 4 is the observed result.
    sink = CancelAckSink(at=4)
    with trajectory_scope(make_context(sink)), pytest.raises(asyncio.CancelledError):
        await middleware.process(context, _next)

    assert sink.only(EventType.TOOL_PAYLOAD_OBSERVED)
    assert sink.only(EventType.TOOL_OPERATION_FINISHED).payload["outcome"] == ToolOutcome.INTERRUPTED


@pytest.mark.parametrize("kind", ["main", "sub_agent"])
async def test_event_middlewares_record_the_result_the_ceiling_leaves(kind: str) -> None:
    """The kernel bounds the result after this middleware returns — the same
    ceiling a hook's appended context escapes — so a payload measured before
    that describes bytes the model was never handed."""
    middleware = _execution_stamp_middleware(kind, tool_result_ceiling_tokens=100)
    context = SimpleNamespace(
        function=SimpleNamespace(name="read_file", chrys_kind="filesystem.read"),
        arguments={"path": "a.txt"},
        result=None,
        metadata={},
    )
    oversized = "word " * 4000

    async def _next() -> None:
        context.result = oversized

    sink = FakeSink()
    with trajectory_scope(make_context(sink)):
        await middleware.process(context, _next)

    payload = sink.only(EventType.TOOL_PAYLOAD_OBSERVED).payload
    assert payload["model_visible_bytes"] == len(apply_result_ceiling(oversized, 100).encode())
    assert payload["model_visible_bytes"] < len(oversized.encode())
    assert payload["truncated"] is True
    assert payload["original_bytes"] == len(oversized.encode())


@pytest.mark.parametrize("kind", ["main", "sub_agent"])
async def test_event_middlewares_record_which_server_or_skill_served_the_call(kind: str) -> None:
    """``search_issues`` and ``load_skill`` say nothing on their own about who answered."""
    middleware = _execution_stamp_middleware(kind)
    mcp_tool = SimpleNamespace(name="search_issues", chrys_kind="mcp")
    set_tool_context(mcp_tool, {"server_name": "github", "remote_name": "search-issues"})
    skill_tool = SimpleNamespace(name="load_skill", chrys_kind="skill")
    set_tool_context_builder(skill_tool, lambda args: {"skill_name": args["skill_name"]})

    async def _next() -> None:
        context.result = "ok"

    sink = FakeSink()
    with trajectory_scope(make_context(sink)):
        for function, arguments in ((mcp_tool, {"query": "open"}), (skill_tool, {"skill_name": "pdf-forms"})):
            context = SimpleNamespace(function=function, arguments=arguments, result=None, metadata={})
            await middleware.process(context, _next)

    served_by = [event.payload["tool_context"] for event in sink.of_type(EventType.TOOL_OPERATION_STARTED)]
    assert served_by == [{"server_name": "github", "remote_name": "search-issues"}, {"skill_name": "pdf-forms"}]


@pytest.mark.parametrize("kind", ["main", "sub_agent"])
async def test_event_middlewares_stamp_structured_error(kind: str) -> None:
    from chrys.service.tools.result_metadata import tool_error

    middleware = _execution_stamp_middleware(kind)
    context = SimpleNamespace(
        function=SimpleNamespace(name="read_file", chrys_kind="filesystem.read"),
        arguments={"path": "missing.txt"},
        result=None,
        metadata={},
    )

    async def _next() -> None:
        context.result = tool_error("not_found", "missing")

    await middleware.process(context, _next)

    stamp = context.metadata[EXECUTION_STAMP_KEY]
    assert stamp["outcome"] == "error"
    assert stamp["error_kind"] == "not_found"


@pytest.mark.parametrize("kind", ["main", "sub_agent"])
async def test_event_middlewares_stamp_raised_error(kind: str) -> None:
    middleware = _execution_stamp_middleware(kind)
    context = SimpleNamespace(
        function=SimpleNamespace(name="explode", chrys_kind=None),
        arguments={"value": 7},
        result=None,
        metadata={},
    )

    async def _next() -> None:
        raise LookupError("boom")

    with pytest.raises(LookupError, match="boom"):
        await middleware.process(context, _next)

    stamp = context.metadata[EXECUTION_STAMP_KEY]
    assert stamp["outcome"] == "error"
    assert stamp["error_kind"] == "LookupError"


async def test_execution_stamp_digest_is_stable_and_tracks_hook_rewrite() -> None:
    from chrys.foundation.events.bus import EventBus
    from chrys.service.hooks.events import HookEvent
    from chrys.service.hooks.schema import HookDecision

    async def _run(arguments: dict[str, int], hook_manager=None) -> dict[str, str]:
        middleware = ToolEventMiddleware(EventBus(), session_id="stamp-test", hook_manager=hook_manager)
        context = SimpleNamespace(
            function=SimpleNamespace(name="echo", chrys_kind=None),
            arguments=arguments,
            result=None,
            metadata={},
        )

        async def _next() -> None:
            context.result = "done"

        await middleware.process(context, _next)
        return context.metadata[EXECUTION_STAMP_KEY]

    class _RewriteHookManager:
        def has_hooks_for(self, event: HookEvent) -> bool:
            return event == HookEvent.BEFORE_TOOL_CALL

        async def fire(self, _event: HookEvent, _payload: dict[str, object], **_kwargs: object) -> HookDecision:
            return HookDecision(args_override={"a": 9})

    first = await _run({"a": 1, "b": 2})
    reordered = await _run({"b": 2, "a": 1})
    rewritten = await _run({"a": 1, "b": 2}, _RewriteHookManager())

    assert first["effective_args_digest"] == reordered["effective_args_digest"]
    assert rewritten["effective_args"] == '{"a":9,"b":2}'
    assert rewritten["effective_args_digest"] != first["effective_args_digest"]


def test_execution_stamp_caps_effective_arguments_by_middle_truncation() -> None:
    stamp = build_execution_stamp({"payload": "x" * 4_096}, outcome="ok")

    assert len(stamp["effective_args"]) == EFFECTIVE_ARGS_MAX_CHARS
    assert "…" in stamp["effective_args"]


@pytest.mark.parametrize("kind", ["main", "sub_agent"])
async def test_event_middlewares_stamp_typed_mapping_keys_without_failing_tool(kind: str) -> None:
    middleware = _execution_stamp_middleware(kind)
    context = SimpleNamespace(
        function=SimpleNamespace(name="dated", chrys_kind=None),
        arguments={"values": {date(2026, 7, 18): "ok"}},
        result=None,
        metadata={},
    )

    async def _next() -> None:
        context.result = "done"

    await middleware.process(context, _next)

    assert context.result == "done"
    stamp = context.metadata[EXECUTION_STAMP_KEY]
    assert stamp["effective_args"] == '{"values":{"2026-07-18":"ok"}}'
    assert stamp["outcome"] == "ok"


@pytest.mark.parametrize("kind", ["main", "sub_agent"])
async def test_event_middlewares_never_fail_tool_when_stamp_serialization_fails(kind: str) -> None:
    class _Unserializable:
        def __str__(self) -> str:
            raise RuntimeError("cannot stringify")

    middleware = _execution_stamp_middleware(kind)
    context = SimpleNamespace(
        function=SimpleNamespace(name="opaque", chrys_kind=None),
        arguments={"value": _Unserializable()},
        result=None,
        metadata={},
    )

    async def _next() -> None:
        context.result = "done"

    await middleware.process(context, _next)

    assert context.result == "done"
    assert EXECUTION_STAMP_KEY not in context.metadata
