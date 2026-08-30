# Copyright (c) 2026 Chrys. All rights reserved.

"""Process-wide telemetry gate shared by foundation setup and kernel spans."""

from __future__ import annotations


class TelemetryGate:
    """Process-wide switch for the chrys telemetry layers.

    Replaces the framework ``OBSERVABILITY_SETTINGS`` singleton
    (``observability.py:666-811``) for the two mirrored layers: a plain pair
    of booleans, default **off**, no env reads, no sticky-disable defenses.
    """

    def __init__(self) -> None:
        self.enabled: bool = False
        self.sensitive_data: bool = False

    @property
    def sensitive_enabled(self) -> bool:
        """Mirror of ``SENSITIVE_DATA_ENABLED`` (``:790-795``): both flags."""
        return self.enabled and self.sensitive_data


TELEMETRY_GATE = TelemetryGate()


def configure_telemetry(*, enabled: bool, sensitive_data: bool = False) -> None:
    """Flip the telemetry gate (``setup_otel`` closes it at entry, reopens on success)."""
    TELEMETRY_GATE.enabled = enabled
    TELEMETRY_GATE.sensitive_data = sensitive_data
