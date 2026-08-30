# Copyright (c) 2026 Chrys. All rights reserved.

"""Direct tests for JSON-safe serialization fallbacks."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from chrys.kernel._serialization import make_json_safe


def test_make_json_safe_serializes_dataclass_via_asdict() -> None:
    @dataclass
    class _Record:
        label: str
        count: int

    assert make_json_safe(_Record(label="ready", count=2)) == {"label": "ready", "count": 2}


class _ModelDumpObject:
    def model_dump(self) -> dict[str, str]:
        return {"source": "model_dump"}


class _ToDictObject:
    def to_dict(self) -> dict[str, str]:
        return {"source": "to_dict"}


class _DictObject:
    def dict(self) -> dict[str, str]:
        return {"source": "dict"}


@pytest.mark.parametrize(
    ("value", "source"),
    [
        (_ModelDumpObject(), "model_dump"),
        (_ToDictObject(), "to_dict"),
        (_DictObject(), "dict"),
    ],
)
def test_make_json_safe_uses_supported_conversion_method(value: object, source: str) -> None:
    assert make_json_safe(value) == {"source": source}


def test_make_json_safe_broken_conversion_methods_fall_through_silently() -> None:
    class _BrokenMethods:
        def __init__(self) -> None:
            self.value = "fallback"

        def model_dump(self) -> dict[str, object]:
            raise TypeError("broken model_dump")

        def to_dict(self) -> dict[str, object]:
            raise TypeError("broken to_dict")

        def dict(self) -> dict[str, object]:
            raise TypeError("broken dict")

    assert make_json_safe(_BrokenMethods()) == {"value": "fallback"}


def test_make_json_safe_serializes_frozenset_deterministically() -> None:
    assert make_json_safe(frozenset({"beta", "alpha"})) == ["alpha", "beta"]


def test_make_json_safe_depth_cap_terminates_self_referencing_mapping() -> None:
    value: dict[str, object] = {}
    value["self"] = value

    safe = make_json_safe(value)

    assert isinstance(safe, dict)
    json.dumps(safe)


def test_make_json_safe_uses_vars_as_final_object_fallback() -> None:
    class _PlainObject:
        def __init__(self) -> None:
            self.label = "plain"
            self.count = 3

    assert make_json_safe(_PlainObject()) == {"label": "plain", "count": 3}
