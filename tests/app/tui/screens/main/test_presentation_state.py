# Copyright (c) 2026 Chrys. All rights reserved.

"""Attempt-scoped provisional presentation state tests."""

from chrys.app.tui.screens.main.state import TurnRenderGate
from chrys.foundation.events.types import AgentMessage, ProvisionalPresentation


def test_render_gate_commits_or_drops_deferred_provisional_messages() -> None:
    gate = TurnRenderGate()
    gate.begin()
    accepted = AgentMessage(
        text="accepted",
        is_intermediate=True,
        presentation=ProvisionalPresentation("attempt-1", "segment-1"),
    )
    dropped = AgentMessage(
        text="dropped",
        is_intermediate=True,
        presentation=ProvisionalPresentation("attempt-1", "segment-2"),
    )
    rejected = AgentMessage(
        text="rejected",
        is_intermediate=True,
        presentation=ProvisionalPresentation("attempt-2", "segment-3"),
    )
    gate.defer(accepted)
    gate.defer(dropped)
    gate.defer(rejected)

    gate.accept_presentation_attempt("attempt-1", ("segment-1",))
    gate.reject_presentation_attempt("attempt-2")

    assert gate.consume_deferred() == [accepted]
    assert accepted.presentation is None
