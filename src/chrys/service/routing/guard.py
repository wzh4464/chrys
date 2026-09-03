# Copyright (c) 2026 Chrys. All rights reserved.

"""Session-scoped budget for the routing tiebreaker.

The router may call a model, and a router that calls a model on every turn is
a router nobody keeps enabled. Two independent bounds apply: a per-session call
cap so routing overhead stays proportional to the session, and a circuit
breaker so an unreachable or misconfigured model costs one round of latency
rather than one per turn.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class TiebreakerGuard:
    """Decide whether the router may spend one model call right now."""

    def __init__(
        self,
        *,
        max_calls: int = 20,
        trip_after: int = 5,
        cooldown_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_calls = max_calls
        self._trip_after = trip_after
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._calls = 0
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def calls(self) -> int:
        """How many attempts this session has made, successful or not."""
        return self._calls

    def allow(self) -> tuple[bool, str]:
        """Return whether a call may proceed, and why not when it may not.

        The rate limit is reported ahead of the breaker because it is the
        condition that will not clear: a breaker reopens after its cooldown,
        a spent session budget does not.
        """
        if self._calls >= self._max_calls:
            return False, "rate_limited"
        if self._opened_at is not None and self._clock() - self._opened_at < self._cooldown:
            return False, "circuit_open"
        return True, ""

    def record_success(self) -> None:
        """Count a completed call and close the breaker."""
        self._calls += 1
        self._consecutive_failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        """Count a failed call; a run of them opens the breaker.

        Failures count against the rate limit too — a model that is failing
        must not be retried more often than one that is working.
        """
        self._calls += 1
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._trip_after:
            self._opened_at = self._clock()
            # The half-open probe is one call, not a fresh run: another failure
            # re-opens immediately rather than needing `trip_after` more.
            self._consecutive_failures = self._trip_after
