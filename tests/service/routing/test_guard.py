# Copyright (c) 2026 Chrys. All rights reserved.

"""Rate limit and circuit breaker around the routing tiebreaker."""

from __future__ import annotations

from chrys.service.routing.guard import TiebreakerGuard


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_a_fresh_guard_allows() -> None:
    assert TiebreakerGuard().allow() == (True, "")


def test_successes_are_capped_per_session() -> None:
    guard = TiebreakerGuard(max_calls=2)

    assert guard.allow() == (True, "")
    guard.record_success()
    assert guard.allow() == (True, "")
    guard.record_success()

    assert guard.allow() == (False, "rate_limited")
    assert guard.calls == 2


def test_consecutive_failures_open_the_breaker() -> None:
    clock = FakeClock()
    guard = TiebreakerGuard(max_calls=20, trip_after=2, cooldown_seconds=60, clock=clock)

    guard.record_failure()
    assert guard.allow() == (True, "")
    guard.record_failure()

    assert guard.allow() == (False, "circuit_open")


def test_a_success_resets_the_failure_run() -> None:
    guard = TiebreakerGuard(max_calls=20, trip_after=2)

    guard.record_failure()
    guard.record_success()
    guard.record_failure()

    assert guard.allow() == (True, "")


def test_the_breaker_half_opens_after_the_cooldown() -> None:
    clock = FakeClock()
    guard = TiebreakerGuard(max_calls=20, trip_after=2, cooldown_seconds=60, clock=clock)
    guard.record_failure()
    guard.record_failure()
    assert guard.allow() == (False, "circuit_open")

    clock.now += 61

    # Exactly one probe is let through.
    assert guard.allow() == (True, "")


def test_a_failed_probe_reopens_the_breaker_for_another_cooldown() -> None:
    clock = FakeClock()
    guard = TiebreakerGuard(max_calls=20, trip_after=2, cooldown_seconds=60, clock=clock)
    guard.record_failure()
    guard.record_failure()
    clock.now += 61
    assert guard.allow() == (True, "")

    guard.record_failure()

    assert guard.allow() == (False, "circuit_open")
    clock.now += 61
    assert guard.allow() == (True, "")


def test_a_successful_probe_closes_the_breaker() -> None:
    clock = FakeClock()
    guard = TiebreakerGuard(max_calls=20, trip_after=2, cooldown_seconds=60, clock=clock)
    guard.record_failure()
    guard.record_failure()
    clock.now += 61
    assert guard.allow() == (True, "")

    guard.record_success()

    assert guard.allow() == (True, "")
    assert guard.allow() == (True, "")


def test_the_rate_limit_counts_failures_too() -> None:
    """A failing model must not be retried without bound just because it fails."""
    guard = TiebreakerGuard(max_calls=2, trip_after=10)

    guard.record_failure()
    guard.record_failure()

    assert guard.allow() == (False, "rate_limited")


def test_the_rate_limit_is_reported_before_the_breaker() -> None:
    clock = FakeClock()
    guard = TiebreakerGuard(max_calls=2, trip_after=2, cooldown_seconds=60, clock=clock)

    guard.record_failure()
    guard.record_failure()

    # Both conditions hold; the caller is told the one that will not clear.
    assert guard.allow() == (False, "rate_limited")
