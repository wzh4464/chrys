# Copyright (c) 2026 Chrys. All rights reserved.

"""Engine-internal sub-agent pause coordination."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from chrys.orchestration.engine.state.machine import Trigger

if TYPE_CHECKING:
    from chrys.foundation.events.types import (
        SubAgentAborted,
        SubAgentAbortRequested,
        SubAgentCascadeAborted,
        SubAgentPaused,
        SubAgentResumed,
        SubAgentRetryRequested,
    )
    from chrys.orchestration.engine.state.machine import EngineStateMachine
    from chrys.orchestration.engine.trajectory import TrajectoryRecorder
    from chrys.orchestration.sub_agents.tools import SubAgentTools
    from chrys.service.session.history import SessionHistoryManager


logger = logging.getLogger(__name__)


class SubAgentCoordinationHost(Protocol):
    """Engine state needed by sub-agent pause coordination."""

    _sub_agent_tools: SubAgentTools | None
    _paused_sub_agents: set[str]
    _fsm: EngineStateMachine
    _history: SessionHistoryManager
    _trajectory_recorder: TrajectoryRecorder


async def on_sub_agent_retry(host: SubAgentCoordinationHost, event: SubAgentRetryRequested) -> None:
    """Route a user's per-card Retry click to the owning controller."""
    if host._sub_agent_tools is None:
        return
    host._sub_agent_tools.request_retry(event.invocation_id)


async def on_sub_agent_abort(host: SubAgentCoordinationHost, event: SubAgentAbortRequested) -> None:
    """Route a user's per-card Abort click to the owning controller."""
    if host._sub_agent_tools is None:
        return
    host._sub_agent_tools.request_abort(event.invocation_id)


async def on_sub_agent_paused(host: SubAgentCoordinationHost, event: SubAgentPaused) -> None:
    """Track a newly paused sub-agent and drive FSM / marker.

    Idempotent on the paused set — if the same id arrives twice
    (controller re-publishes after retry exhaustion, for example)
    the FSM transition only fires on the 0→1 edge.

    Defensive FSM guard: if the parent run has already terminated
    (e.g. the bus is dispatching a stale pause event that was queued
    before the parent's task was cancelled and ``_post_run`` ran),
    drop the event silently rather than re-inserting a marker on top
    of an already-terminal ``interrupted``/``error`` marker.  The
    parent invariant ("parent run ends only after all sub-agents
    resolve") should make this unreachable, but defense in depth
    keeps history well-formed if that invariant ever breaks.
    """
    if not host._fsm.is_running():
        logger.debug(
            "Ignoring SubAgentPaused for %s: FSM=%s (parent run already terminated)",
            event.invocation_id,
            host._fsm.state.name,
        )
        return
    first = not host._paused_sub_agents
    host._paused_sub_agents.add(event.invocation_id)
    if first:
        host._fsm.try_transition(Trigger.SUB_AGENT_PAUSED)
        await host._trajectory_recorder.turn_suspended()
    if host._history.is_bound:
        host._history.upsert_awaiting_sub_agents_marker(sorted(host._paused_sub_agents))


async def on_sub_agent_unpaused(
    host: SubAgentCoordinationHost,
    event: SubAgentResumed | SubAgentAborted | SubAgentCascadeAborted,
) -> None:
    """Common handler — retry/abort/cascade all remove the invocation from the paused set.

    FSM transitions only on the N→0 edge (last paused sub-agent
    resolved).  Marker is updated or stripped accordingly.
    """
    if event.invocation_id not in host._paused_sub_agents:
        return
    host._paused_sub_agents.discard(event.invocation_id)
    if not host._paused_sub_agents:
        host._fsm.try_transition(Trigger.SUB_AGENT_RESOLVED)
        await host._trajectory_recorder.turn_resumed()
        if host._history.is_bound:
            host._history.remove_awaiting_sub_agents_marker()
    elif host._history.is_bound:
        host._history.upsert_awaiting_sub_agents_marker(sorted(host._paused_sub_agents))
    # Defensive invariant: FSM ``AWAITING_SUB_AGENTS`` must mirror a
    # non-empty paused set.  A desync here would mean either a stale
    # pause event leaked past the ``_fsm.is_running()`` guard in
    # ``_on_sub_agent_paused`` or a ``try_transition`` silently no-op'd.
    # Log instead of raising so a surprise in production doesn't crash
    # the engine mid-run — the log points straight at the bug.
    if bool(host._paused_sub_agents) != host._fsm.is_awaiting_sub_agents():
        logger.error(
            "FSM/paused-set desync: paused=%s FSM=%s",
            sorted(host._paused_sub_agents),
            host._fsm.state.name,
        )


def clear_parent_paused_state(host: SubAgentCoordinationHost) -> None:
    """Clear engine-side paused sub-agent state after a parent run ends."""
    host._paused_sub_agents.clear()
    host._history.remove_awaiting_sub_agents_marker()
