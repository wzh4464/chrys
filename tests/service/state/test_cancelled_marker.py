# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for interrupted marker handling in state store and message counting."""

from __future__ import annotations

from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.kernel import Message
from chrys.service.state.store import _is_visible_message


def _make_interrupted_marker() -> Message:
    m = Message("assistant", ["Execution interrupted"])
    m.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.INTERRUPTED
    m.additional_properties["_interrupted_by"] = "user"
    return m


def _make_turn_marker() -> Message:
    m = Message("assistant", [""])
    m.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    return m


def test_interrupted_marker_not_visible():
    """Interrupted markers should not be counted as visible messages."""
    m = _make_interrupted_marker()
    assert not _is_visible_message(m)


def test_turn_marker_not_visible():
    """Turn markers should not be counted as visible messages."""
    m = _make_turn_marker()
    assert not _is_visible_message(m)


def test_user_message_visible():
    """User messages should be counted as visible."""
    m = Message("user", ["hello"])
    assert _is_visible_message(m)


def test_assistant_with_text_visible():
    """Assistant messages with non-empty text should be counted as visible."""
    m = Message("assistant", ["I can help with that."])
    assert _is_visible_message(m)


def test_assistant_empty_text_not_visible():
    """Assistant messages with empty text should not be counted as visible."""
    m = Message("assistant", [""])
    assert not _is_visible_message(m)
