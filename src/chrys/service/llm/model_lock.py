# Copyright (c) 2026 Chrys. All rights reserved.

"""Fail-closed runtime lock for outbound model clients.

The regular profile resolver decides which model Chrys should use. This
module provides a narrower deployment policy: when ``CHRYS_MODEL_LOCK`` is
present, every client created by Chrys must match one exact wire identity.
That final check covers main agents, sub-agents, judges, summarizers, and
other side calls without duplicating policy across their resolvers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Final

from chrys.service.profiles.models.schema import ModelProfile

MODEL_LOCK_ENV: Final = "CHRYS_MODEL_LOCK"
_MODEL_LOCK_FIELDS: Final = frozenset({"provider", "api_style", "base_url", "model_id"})


class ModelLockError(RuntimeError):
    """Raised before client construction when the runtime model lock is invalid or violated."""


@dataclass(frozen=True, slots=True)
class ModelWireIdentity:
    """The provider-visible fields that identify one permitted model route."""

    provider: str
    api_style: str
    base_url: str
    model_id: str


def _normalized_base_url(value: str) -> str:
    return value.strip().rstrip("/")


def _identity_from_mapping(value: Any) -> ModelWireIdentity:
    if not isinstance(value, dict):
        raise ModelLockError(f"{MODEL_LOCK_ENV} must be a JSON object")

    keys = frozenset(value)
    if keys != _MODEL_LOCK_FIELDS:
        missing = sorted(_MODEL_LOCK_FIELDS - keys)
        unknown = sorted(keys - _MODEL_LOCK_FIELDS)
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise ModelLockError(f"Invalid {MODEL_LOCK_ENV} ({'; '.join(details)})")

    fields: dict[str, str] = {}
    for name in sorted(_MODEL_LOCK_FIELDS):
        field = value[name]
        if not isinstance(field, str) or not field.strip():
            raise ModelLockError(f"{MODEL_LOCK_ENV}.{name} must be a non-empty string")
        fields[name] = field.strip()
    fields["base_url"] = _normalized_base_url(fields["base_url"])
    return ModelWireIdentity(**fields)


def configured_model_lock() -> ModelWireIdentity | None:
    """Parse the process model lock, or return ``None`` when it is absent.

    Presence is intentional: an explicitly empty or malformed value fails
    closed instead of silently disabling the guard.
    """
    raw = os.environ.get(MODEL_LOCK_ENV)
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelLockError(f"{MODEL_LOCK_ENV} must contain valid JSON: {exc.msg}") from exc
    return _identity_from_mapping(value)


def enforce_model_lock(profile: ModelProfile, *, effective_base_url: str) -> None:
    """Reject *profile* unless it matches the configured wire identity."""
    expected = configured_model_lock()
    if expected is None:
        return

    actual = ModelWireIdentity(
        provider=profile.provider.strip(),
        api_style=profile.api_style.strip(),
        base_url=_normalized_base_url(effective_base_url),
        model_id=profile.model_id.strip(),
    )
    if actual == expected:
        return

    raise ModelLockError(
        f"Model profile {profile.name!r} is blocked by {MODEL_LOCK_ENV}: expected {expected!r}, got {actual!r}"
    )
