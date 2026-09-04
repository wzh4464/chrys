# Copyright (c) 2026 Chrys. All rights reserved.

"""A structured reply that explains before it answers still yields its object."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from chrys.kernel._types import _embedded_json_object, _parse_structured_response_value


class _Decision(BaseModel):
    action: str
    reviews: list[str] = []


@pytest.mark.parametrize(
    "text",
    [
        '{"action": "select", "reviews": ["p1-g1"]}',
        'All six candidates have merit; here is the decision.\n\n{"action": "select", "reviews": ["p1-g1"]}',
        'Reasoning first.\n```json\n{"action": "select", "reviews": ["p1-g1"]}\n```\nDone.',
        'Note: braces {like these} are not JSON. {"action": "select", "reviews": ["p1-g1"]}',
    ],
)
def test_the_object_is_found_wherever_the_model_put_it(text: str) -> None:
    assert _parse_structured_response_value(text, _Decision) == _Decision(action="select", reviews=["p1-g1"])


def test_a_reply_with_no_object_still_fails_validation() -> None:
    with pytest.raises(ValueError):
        _parse_structured_response_value("All six candidates have merit.", _Decision)
    assert _embedded_json_object("nothing here") is None


def test_a_mapping_format_gets_the_same_fallback() -> None:
    assert _parse_structured_response_value('Sure: {"a": 1}', {"type": "object"}) == {"a": 1}
