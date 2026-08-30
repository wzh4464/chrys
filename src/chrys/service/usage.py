# Copyright (c) 2026 Chrys. All rights reserved.

"""Engine-internal usage and compaction event publishing."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from chrys.foundation.events.types import ContextCompressed, ToolCompacted, UsageUpdate
from chrys.service.profiles.models.schema import DEFAULT_MAX_CONTEXT_TOKENS

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.service.context.compaction import CompactionInfo, CompressInfo
    from chrys.service.profiles.agents.schema import AgentProfile
    from chrys.service.profiles.models.schema import ModelProfile
    from chrys.service.session.runtime_metadata import SessionRuntimeMetadata


class UsageHost(Protocol):
    """Engine state needed by usage and compaction publishing."""

    _agent_profile: AgentProfile | None
    _active_profile: ModelProfile | None
    _session_id: str | None
    _runtime_meta: SessionRuntimeMetadata
    _agent_profile_fingerprint: str
    _model_profile_fingerprint: str
    _usage_tasks: set[asyncio.Task[None]]
    _usage_publish_tail: asyncio.Task[None] | None

    @property
    def event_bus(self) -> EventBus: ...


def make_usage_event(host: UsageHost, *, session_id: str | None = None) -> UsageUpdate:
    """Build a UsageUpdate event — pct always computed from the active profile."""
    max_ctx = host._active_profile.max_context_tokens if host._active_profile else DEFAULT_MAX_CONTEXT_TOKENS
    last_usage = host._runtime_meta.last_usage_details
    input_tok = last_usage.get("input_token_count", 0)
    output_tok = last_usage.get("output_token_count", 0)
    total = last_usage.get("total_token_count", 0)
    pct = round(total / max_ctx * 100, 1) if max_ctx else 0.0
    event_session_id = session_id or host._session_id
    return UsageUpdate(
        agent_profile=host._agent_profile.name if host._agent_profile else "",
        usage_source_id=event_session_id or "",
        input_tokens=input_tok,
        output_tokens=output_tok,
        total_tokens=total,
        pct=pct,
        max_context_tokens=max_ctx,
        total_session_tokens=host._runtime_meta.total_session_tokens,
        total_session_input_tokens=host._runtime_meta.total_session_input_tokens,
        total_session_output_tokens=host._runtime_meta.total_session_output_tokens,
        total_session_cache_hit_tokens=host._runtime_meta.total_session_cache_hit_tokens,
        cache_hit_tokens=last_usage.get("cache_hit_tokens"),
        local_tokens=last_usage.get("local_tokens", 0),
        calibration_ratio=last_usage.get("calibration_ratio", 1.0),
        system_overhead_tokens=last_usage.get("system_overhead_tokens", 0),
        session_id=event_session_id,
    )


def publish_usage(
    host: UsageHost,
    total_tokens: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
    local_tokens: int = 0,
    calibration_ratio: float = 1.0,
    system_overhead_tokens: int = 0,
    cache_hit_tokens: int | None = None,
    calibration_initialized: bool = False,
    use_local_context_estimate: bool = False,
) -> None:
    """Callback from UsageTrackingMiddleware — publish UsageUpdate event."""
    host._runtime_meta.record_usage(
        total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        local_tokens=local_tokens,
        calibration_ratio=calibration_ratio,
        system_overhead_tokens=system_overhead_tokens,
        cache_hit_tokens=cache_hit_tokens,
        use_local_context_estimate=use_local_context_estimate,
    )
    if calibration_initialized:
        host._runtime_meta.record_context_calibration(
            system_overhead_tokens=system_overhead_tokens,
            calibration_ratio=calibration_ratio,
            model_profile_fingerprint=host._model_profile_fingerprint,
            agent_profile_fingerprint=host._agent_profile_fingerprint,
        )
    _track_usage_publish(host, make_usage_event(host))


def accumulate_sub_agent_usage(
    host: UsageHost,
    total_tokens: int,
    input_tokens: int = 0,
    output_tokens: int = 0,
    local_tokens: int = 0,
    calibration_ratio: float = 1.0,
    system_overhead_tokens: int = 0,
    cache_hit_tokens: int | None = None,
    max_context_tokens: int | None = None,
    agent_profile: str = "",
    usage_source_id: str = "",
    authoritative_total: int | None = None,
    use_local_context_estimate: bool = False,
) -> None:
    """Callback from sub-agent UsageTrackingMiddleware — accumulate and publish live usage.

    ``local_tokens``, ``calibration_ratio``, ``system_overhead_tokens``, and
    ``max_context_tokens`` come from the sub-agent's own ContextManager/model.
    They are copied only onto the live UsageUpdate.  Parent-agent
    ``last_usage_details`` stays parent-owned; only session totals accumulate.
    """
    host._runtime_meta.accumulate_sub_agent_usage(
        total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit_tokens=cache_hit_tokens,
        authoritative_total=authoritative_total,
    )
    # Sub-agent runs on its own model; falling back to the parent's
    # ``host._active_profile.max_context_tokens`` would compute pct against the
    # wrong window (e.g. a small-window Explore reported as ~30% full when it
    # is actually ~94%).  Use the sub-agent's value, or a neutral default.
    max_ctx = DEFAULT_MAX_CONTEXT_TOKENS if max_context_tokens is None else max_context_tokens
    window_input_tokens = local_tokens if use_local_context_estimate else input_tokens
    window_total_tokens = window_input_tokens + output_tokens if use_local_context_estimate else total_tokens
    pct = round(window_total_tokens / max_ctx * 100, 1) if max_ctx else 0.0
    _track_usage_publish(
        host,
        UsageUpdate(
            agent_profile=agent_profile,
            usage_source_id=usage_source_id,
            input_tokens=window_input_tokens,
            output_tokens=output_tokens,
            total_tokens=window_total_tokens,
            pct=pct,
            max_context_tokens=max_ctx,
            total_session_tokens=host._runtime_meta.total_session_tokens,
            total_session_input_tokens=host._runtime_meta.total_session_input_tokens,
            total_session_output_tokens=host._runtime_meta.total_session_output_tokens,
            total_session_cache_hit_tokens=host._runtime_meta.total_session_cache_hit_tokens,
            cache_hit_tokens=cache_hit_tokens,
            local_tokens=local_tokens,
            calibration_ratio=calibration_ratio,
            system_overhead_tokens=system_overhead_tokens,
            session_id=host._session_id,
        ),
    )


def accumulate_side_call_usage(host: UsageHost, usage_details: Mapping[str, Any]) -> None:
    """Callback from Phase-4 LAST_WORDS generators — fold side-call spend into session totals.

    Side-call responses are consumed inside the client and never reach the
    middleware chain, so their provider-reported usage is fed here directly
    — every attempt, including responses later rejected by validation or
    the tool-call guard.  Accumulate-only: the parent's ``last_usage_details``
    (context meter) stays untouched; one UsageUpdate then refreshes the
    session totals with the main window numbers unchanged.
    """
    from chrys.service.context.middleware.usage import extract_cache_hit_tokens

    input_tokens = int(usage_details.get("input_token_count") or 0)
    output_tokens = int(usage_details.get("output_token_count") or 0)
    total_tokens = int(usage_details.get("total_token_count") or 0)
    if not (input_tokens or output_tokens or total_tokens):
        return
    # Response usage_details carry the kernel-normalized cache field; raw
    # provider keys (what extract_cache_hit_tokens reads) only as a fallback.
    cache_hit_tokens = usage_details.get("cache_read_input_token_count")
    if cache_hit_tokens is None:
        cache_hit_tokens = extract_cache_hit_tokens(dict(usage_details))
    host._runtime_meta.accumulate_out_of_band_usage(
        total_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit_tokens=int(cache_hit_tokens) if cache_hit_tokens is not None else None,
    )
    _track_usage_publish(host, make_usage_event(host))


def _track_usage_publish(host: UsageHost, event: UsageUpdate) -> None:
    """Keep usage publish tasks alive and deliver them in creation order."""
    previous = host._usage_publish_tail

    async def _publish_after_previous() -> None:
        if previous is not None:
            # ``gather(return_exceptions=True)`` waits for ``previous`` without
            # inheriting its failure or cancellation, so a cancelled earlier
            # publish cannot cascade-cancel every queued UsageUpdate behind it.
            # ``contextlib.suppress(Exception)`` would not catch CancelledError
            # (BaseException since 3.8), so the chain would otherwise unravel
            # silently on shutdown.
            await asyncio.gather(previous, return_exceptions=True)
        await host.event_bus.publish(event)

    task = asyncio.create_task(_publish_after_previous())
    host._usage_publish_tail = task
    host._usage_tasks.add(task)
    task.add_done_callback(host._usage_tasks.discard)
    task.add_done_callback(lambda done: _clear_usage_tail(host, done))
    task.add_done_callback(_observe_task_exception)


def enqueue_usage_event(host: UsageHost, event: UsageUpdate) -> None:
    """Schedule a pre-built UsageUpdate through the ordered publish chain.

    Use this when a caller has already constructed the event (e.g. session
    restore, soft-restart) and needs the publish to respect the same in-order
    delivery guarantee that ``publish_usage`` / ``accumulate_sub_agent_usage``
    provide.  Direct ``bus.publish`` from those paths can overtake a still-
    pending sub-agent UsageUpdate already queued in the chain.
    """
    _track_usage_publish(host, event)


def _clear_usage_tail(host: UsageHost, task: asyncio.Task[None]) -> None:
    if host._usage_publish_tail is task:
        host._usage_publish_tail = None


def _observe_task_exception(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        return
    with contextlib.suppress(Exception):
        task.exception()


async def publish_compress(host: UsageHost, info: CompressInfo) -> None:
    """Callback from ContextManagementProvider or force-compress — publish ContextCompressed event."""
    await host.event_bus.publish(
        ContextCompressed(
            compressed_context_id=info.compressed_context_id,
            summary=info.summary,
            freed_messages=info.freed_messages,
            turn_range=info.turn_range,
            source=info.source,
            session_id=host._session_id,
        )
    )


async def publish_compaction(host: UsageHost, info: CompactionInfo) -> None:
    """Callback from UnifiedContextStrategy — publish ToolCompacted event."""
    await host.event_bus.publish(
        ToolCompacted(
            compacted_groups=info.compacted_groups,
            phase=info.phase,
            turn_numbers=info.turn_numbers,
            compacted_tool_names=info.tool_names,
            tokens_before=info.tokens_before,
            tokens_after=info.tokens_after,
            last_words_generated=info.last_words_generated,
            session_id=host._session_id,
        )
    )
