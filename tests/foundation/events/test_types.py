# Copyright (c) 2026 Chrys. All rights reserved.

"""Event contract tests."""

from chrys.foundation.events import types
from chrys.foundation.events.types import AgentLoadFailed, AgentLoadProgress, Error, RetryAttempt, Warning
from chrys.foundation.i18n import MessageRef, msg

_BOUND_EVENT_MESSAGE = msg("test_types.bound_event_message", fallback="Bound {value}")


def test_events_with_display_messages_default_to_none() -> None:
    assert Warning().display_message is None
    assert Error().display_message is None
    assert RetryAttempt().display_message is None
    assert AgentLoadFailed().display_message is None


def test_events_accept_bound_display_messages() -> None:
    display_message = _BOUND_EVENT_MESSAGE.bind(value="detail")

    warning = Warning(display_message=display_message)
    error = Error(display_message=display_message)
    retry = RetryAttempt(display_message=display_message)
    load_failed = AgentLoadFailed(display_message=display_message)

    assert isinstance(warning.display_message, MessageRef)
    assert warning.display_message is display_message
    assert isinstance(error.display_message, MessageRef)
    assert error.display_message is display_message
    assert isinstance(retry.display_message, MessageRef)
    assert retry.display_message is display_message
    assert isinstance(load_failed.display_message, MessageRef)
    assert load_failed.display_message is display_message


def test_agent_load_progress_semantic_fields_default_to_empty_strings() -> None:
    progress = AgentLoadProgress()

    assert progress.status == ""
    assert progress.subject == ""
    assert progress.detail == ""


def test_warning_and_error_keep_legacy_fields_with_keyword_construction() -> None:
    warning = Warning(code="warning_code", message="warning message")
    error = Error(code="error_code", message="error message", recoverable=False)

    assert (warning.code, warning.message) == ("warning_code", "warning message")
    assert (error.code, error.message, error.recoverable) == ("error_code", "error message", False)


def test_warning_and_error_remain_constructible_without_arguments() -> None:
    assert isinstance(Warning(), Warning)
    assert isinstance(Error(), Error)


def test_routing_events_default_to_a_standard_no_op() -> None:
    """A default-constructed override clears rather than routes."""
    override = types.RouteOverride()

    assert override.track == ""
    assert override.one_shot is True
    assert override.reroute is False
    assert override.plan_localization is None


def test_turn_routed_carries_the_whole_decision() -> None:
    routed = types.TurnRouted(
        turn=3,
        track="long_horizon",
        band="strong_long_horizon",
        reason="scope=entire",
        confidence=0.9,
        source="heuristic",
        inherited=False,
        prompt_score=0.9,
        plan_localization=True,
        plan_clarification=True,
        plan_pact=True,
        pact_ready=True,
        tiebreaker_failure="",
        switched_to="LongHorizon",
        can_downgrade=True,
    )

    assert routed.track == "long_horizon"
    assert routed.switched_to == "LongHorizon"
    assert routed.can_downgrade is True


def test_turn_routed_defaults_are_a_standard_turn() -> None:
    routed = types.TurnRouted()

    assert routed.track == ""
    assert routed.plan_pact is False
    assert routed.can_downgrade is False
    assert routed.switched_to == ""


def test_long_horizon_phase_defaults_are_non_terminal() -> None:
    phase = types.LongHorizonPhaseChanged(phase="localizing")

    assert phase.terminal is False
    assert phase.detail == ""


def test_memory_writeback_completed_reports_a_held_watermark() -> None:
    completed = types.MemoryWritebackCompleted(reason="idle", deposited=1, failed_turn=2, watermark=1)

    assert completed.failed_turn == 2
    assert completed.watermark == 1
