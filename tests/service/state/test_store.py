# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for state persistence — serializers and JsonFileStateStore."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import sys
import threading
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest

import chrys.foundation.config.settings as settings_module
import chrys.service.state.session_mru as session_mru_module
import chrys.service.state.store as store_module
import chrys.service.trajectory.tombstone as tombstone_module
from chrys.foundation.config.settings import (
    SESSION_ROOT_DIR_ENV_VAR,
    resolve_session_root_dir,
    resolve_sessions_dir,
)
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.foundation.text.mentions import format_file_mention
from chrys.foundation.util.lock import FileLock
from chrys.foundation.util.session_ids import session_short_id
from chrys.kernel import Content, Message
from chrys.service.agent_middleware.system_reminder import CATALOG_POINTER_RECORD_COUNT_STATE_KEY
from chrys.service.context.providers.history import CompressedBlock
from chrys.service.session.message_metadata import MESSAGE_CREATED_AT_KEY, stamp_message_created_at
from chrys.service.state.serializers import (
    deserialize_compressed_block,
    deserialize_message,
    deserialize_state,
    serialize_compressed_block,
    serialize_message,
    serialize_state,
)
from chrys.service.state.session_mru import SESSION_MRU_FILE_NAME, SessionMruEntry, SessionMruIndex, coerce_utc
from chrys.service.state.store import (
    RAW_HTTP_LOG_FILE_NAME,
    SESSION_RECOVERY_FILE_NAME,
    JsonFileStateStore,
    SessionForkError,
    _dir_size,
    _earliest_history_created_at,
    _extract_title,
    _is_visible_message,
    parse_snapshot_turn,
)
from chrys.service.trajectory.state import TRAJECTORY_STATE_KEY
from chrys.service.trajectory.tombstone import DeleteOutcome, DeleteResult

# --- Serializer tests ---


def test_parse_snapshot_turn_accepts_prefixed_and_legacy_numeric_names(tmp_path: Path) -> None:
    assert parse_snapshot_turn(tmp_path / "turn_12.json") == 12
    assert parse_snapshot_turn(tmp_path / "12.json") == 12
    assert parse_snapshot_turn(tmp_path / "turn_nope.json") == -1


def test_serialize_message_roundtrip() -> None:
    msg = Message("user", ["hello", "world"])
    msg.additional_properties["_key"] = "val"
    data = serialize_message(msg)
    restored = deserialize_message(data)
    assert restored.role == "user"
    assert [str(c) for c in restored.contents] == ["hello", "world"]
    assert restored.additional_properties["_key"] == "val"


def test_serialize_compressed_block_roundtrip() -> None:
    block = CompressedBlock(
        compressed_context_id="ctx_abc12345",
        messages=[Message("user", ["msg1"]), Message("assistant", ["msg2"])],
        summary_text="Did some work",
        marker_id="turn_2",
        turn_range=(1, 2),
        created_at="2026-03-17T00:00:00+00:00",
    )
    data = serialize_compressed_block(block)
    restored = deserialize_compressed_block(data)
    assert restored.compressed_context_id == "ctx_abc12345"
    assert restored.summary_text == "Did some work"
    assert restored.marker_id == "turn_2"
    assert restored.turn_range == (1, 2)
    assert restored.created_at == "2026-03-17T00:00:00+00:00"
    assert len(restored.messages) == 2


def test_serialize_state_roundtrip() -> None:
    state = {
        "messages": [Message("user", ["hi"]), Message("assistant", ["hello"])],
        "compressed_msgs": [
            CompressedBlock(
                compressed_context_id="ctx_001",
                messages=[Message("user", ["old"])],
                summary_text="summary",
                marker_id="turn_1",
                turn_range=(1, 1),
                created_at="2026-03-17T00:00:00+00:00",
            )
        ],
        "turn_counter": 3,
    }
    data = serialize_state(state)
    restored = deserialize_state(data)
    assert len(restored["messages"]) == 2
    assert len(restored["compressed_msgs"]) == 1
    assert restored["compressed_msgs"][0].compressed_context_id == "ctx_001"
    assert restored["turn_counter"] == 3


def test_earliest_history_created_at_supports_live_and_serialized_compressed_blocks() -> None:
    legacy = Message("user", ["legacy unstamped"])
    compressed = Message("user", ["old"])
    live = Message("user", ["new"])
    stamp_message_created_at(compressed, "2026-01-01T00:00:00+00:00")
    stamp_message_created_at(live, "2026-02-01T00:00:00+00:00")
    state = {
        "messages": [live],
        "compressed_msgs": [
            CompressedBlock(
                compressed_context_id="ctx_legacy",
                messages=[legacy],
                summary_text="legacy summary",
            ),
            CompressedBlock(
                compressed_context_id="ctx_001",
                messages=[compressed],
                summary_text="summary",
            ),
        ],
    }

    expected = datetime(2026, 1, 1, tzinfo=UTC)
    assert _earliest_history_created_at(state) == expected
    assert _earliest_history_created_at(serialize_state(state)) == expected


def test_serialize_state_roundtrip_with_chrys_mutations() -> None:
    """chrys_mutations data survives serialize/deserialize round-trip."""
    mutations_data = {
        "turns": [
            {
                "turn_id": 1,
                "mutations": [
                    {
                        "path": "/tmp/test.py",
                        "operation": "modify",
                        "source": "edit_file",
                        "tool_call_id": "call_001",
                        "timestamp": 1234567890.0,
                        "before_hash": "abc123",
                        "after_hash": "def456",
                    }
                ],
            }
        ],
        "snapshots": {
            "/tmp/test.py::1": {
                "path": "/tmp/test.py",
                "turn_id": 1,
                "existed": True,
                "content_hash": "abc123",
                "size": 42,
            }
        },
    }
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
        "chrys_mutations": mutations_data,
    }
    data = serialize_state(state)
    assert "chrys_mutations" in data

    restored = deserialize_state(data)
    assert "chrys_mutations" in restored
    assert restored["chrys_mutations"]["turns"][0]["turn_id"] == 1
    assert restored["chrys_mutations"]["turns"][0]["mutations"][0]["path"] == "/tmp/test.py"
    assert restored["chrys_mutations"]["snapshots"]["/tmp/test.py::1"]["content_hash"] == "abc123"


def test_serialize_state_round_trips_context_calibration() -> None:
    """The calibration record is allowlisted and copied (not aliased) both ways."""
    record = {
        "v": 2,
        "system_overhead_tokens": 7,
        "calibration_ratio": 1.2,
        "model_profile_fingerprint": "mfp",
        "agent_profile_fingerprint": "afp",
    }
    state = {"messages": [], "compressed_msgs": [], "context_calibration": record}

    data = serialize_state(state)
    assert data["context_calibration"] == record
    record["v"] = 999
    assert data["context_calibration"]["v"] == 2

    restored = deserialize_state(json.loads(json.dumps(data)))
    assert restored["context_calibration"]["calibration_ratio"] == 1.2


@pytest.mark.parametrize("malformed", ["not-a-dict", 42, ["v", 1], True])
@pytest.mark.parametrize("key", ["context_calibration", "last_usage"])
def test_serialize_state_drops_malformed_dict_valued_keys(key: str, malformed: object) -> None:
    """A corrupted dict-valued key degrades to "absent" instead of crashing the load."""
    state = {"messages": [], "compressed_msgs": [], key: malformed}
    data = serialize_state(state)
    assert key not in data

    restored = deserialize_state({"messages": [], "compressed_msgs": [], key: malformed})
    assert key not in restored


def test_serialize_state_without_chrys_mutations() -> None:
    """State without chrys_mutations serializes cleanly (no key added)."""
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
    }
    data = serialize_state(state)
    assert "chrys_mutations" not in data
    restored = deserialize_state(data)
    assert "chrys_mutations" not in restored


def test_serialize_state_roundtrip_with_last_words() -> None:
    """The Phase 4 LAST_WORDS note survives serialize/deserialize round-trip."""
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
        "last_words": "[LAST_WORDS] progress note for the interrupted turn",
    }
    data = serialize_state(state)
    assert data["last_words"] == "[LAST_WORDS] progress note for the interrupted turn"
    restored = deserialize_state(data)
    assert restored["last_words"] == "[LAST_WORDS] progress note for the interrupted turn"


def test_serialize_state_roundtrip_with_last_words_manifest_and_breaker() -> None:
    manifest = [{"record_id": "r1", "relative_path": "compactions/dropped/turn001/a.md"}]
    breaker = {
        "version": 1,
        "attempts": 3,
        "consecutive_no_progress": 1,
        "tail_override": True,
        "disabled": False,
        "side_call_tokens": 900,
    }
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
        "last_words_manifest": manifest,
        "last_words_breaker": breaker,
        CATALOG_POINTER_RECORD_COUNT_STATE_KEY: 0,
    }

    restored = deserialize_state(serialize_state(state))

    assert restored["last_words_manifest"] == manifest
    assert restored["last_words_breaker"] == breaker
    assert restored[CATALOG_POINTER_RECORD_COUNT_STATE_KEY] == 0


def test_serialize_state_without_last_words() -> None:
    """State without a LAST_WORDS note serializes cleanly (no key added)."""
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
    }
    data = serialize_state(state)
    assert "last_words" not in data
    restored = deserialize_state(data)
    assert "last_words" not in restored


def test_serialize_state_roundtrip_with_chrys_todos() -> None:
    """The session todo list survives serialize/deserialize round-trip."""
    todos = [
        {"content": "Read the plan", "status": "completed", "active_form": "Reading the plan"},
        {"content": "Implement", "status": "in_progress", "active_form": "Implementing"},
        {"content": "Test", "status": "pending", "active_form": ""},
    ]
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
        "chrys_todos": todos,
    }
    data = serialize_state(state)
    assert data["chrys_todos"] == todos
    restored = deserialize_state(data)
    assert restored["chrys_todos"] == todos


def test_serialize_state_drops_empty_chrys_todos() -> None:
    """An empty todo list is dropped (empty ≡ absent), and no key is invented."""
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
        "chrys_todos": [],
    }
    data = serialize_state(state)
    assert "chrys_todos" not in data
    restored = deserialize_state(data)
    assert "chrys_todos" not in restored

    without_key = serialize_state({"messages": [], "compressed_msgs": []})
    assert "chrys_todos" not in without_key


def test_serialize_state_optional_keys_preserve_truthy_only_behavior() -> None:
    """Optional metadata keys are copied from one declarative key table."""
    state = {
        "messages": [Message("user", ["hi"])],
        "compressed_msgs": [],
        "last_usage": {},
        "agent_profile_switches": [],
        "chrys_mutations": {},
        "total_session_tokens": 0,
        "total_session_input_tokens": 0,
        "total_session_output_tokens": 0,
    }

    data = serialize_state(state)

    assert "last_usage" not in data
    assert "agent_profile_switches" not in data
    assert "chrys_mutations" not in data
    assert "total_session_tokens" not in data
    assert "total_session_input_tokens" not in data
    assert "total_session_output_tokens" not in data

    clean_baseline = {
        "version": 1,
        "turn_id": 4,
        "roots": {},
        "degraded": {},
        "root_limit_omitted": 0,
    }
    state["chrys_workspace_baseline"] = clean_baseline
    data = serialize_state(state)
    restored = deserialize_state(data)
    assert restored["chrys_workspace_baseline"] == clean_baseline

    state.pop("chrys_workspace_baseline")
    assert "chrys_workspace_baseline" not in serialize_state(state)

    state.update(
        {
            "last_usage": {"total_token_count": 11, "calibration_ratio": 1.2},
            "agent_profile_switches": [{"from": "Code", "to": "Explore"}],
            "chrys_mutations": {"turns": []},
            "total_session_tokens": 11,
            "total_session_input_tokens": 5,
            "total_session_output_tokens": 6,
        }
    )
    data = serialize_state(state)
    restored = deserialize_state(data)

    assert restored["last_usage"] == {"total_token_count": 11, "calibration_ratio": 1.2}
    assert restored["agent_profile_switches"] == [{"from": "Code", "to": "Explore"}]
    assert restored["chrys_mutations"] == {"turns": []}
    assert restored["total_session_tokens"] == 11
    assert restored["total_session_input_tokens"] == 5
    assert restored["total_session_output_tokens"] == 6


# --- JsonFileStateStore tests ---


def _patch_platform_config(monkeypatch: pytest.MonkeyPatch, config_dir: Path) -> None:
    fake_platform = type("P", (), {"config_dir": config_dir})()
    monkeypatch.setattr("chrys.foundation.platform.get_platform", lambda: fake_platform)


def test_resolve_session_root_dir_defaults_to_config_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    monkeypatch.delenv(SESSION_ROOT_DIR_ENV_VAR, raising=False)
    _patch_platform_config(monkeypatch, config_dir)

    root = resolve_session_root_dir()
    sessions_dir = resolve_sessions_dir()

    assert root == config_dir
    assert sessions_dir == config_dir / "sessions"
    assert sessions_dir.is_dir()
    assert JsonFileStateStore().session_dir("sess1").parent == sessions_dir


def test_resolve_sessions_dir_appends_sessions_to_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    custom_root = tmp_path / "custom-root"
    monkeypatch.setenv(SESSION_ROOT_DIR_ENV_VAR, str(custom_root))
    _patch_platform_config(monkeypatch, config_dir)

    root = resolve_session_root_dir()
    sessions_dir = resolve_sessions_dir()

    assert root == custom_root
    assert sessions_dir == custom_root / "sessions"
    assert sessions_dir.is_dir()
    assert JsonFileStateStore().session_dir("sess1").parent == sessions_dir


def test_resolve_session_root_dir_falls_back_when_env_points_to_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    not_a_directory = tmp_path / "not-a-directory"
    not_a_directory.write_text("", encoding="utf-8")
    monkeypatch.setenv(SESSION_ROOT_DIR_ENV_VAR, str(not_a_directory))
    _patch_platform_config(monkeypatch, config_dir)

    root = resolve_session_root_dir()
    sessions_dir = resolve_sessions_dir()

    assert root == config_dir
    assert sessions_dir == config_dir / "sessions"
    assert sessions_dir.is_dir()
    assert JsonFileStateStore().session_dir("sess1").parent == sessions_dir


def test_resolve_sessions_dir_falls_back_when_env_sessions_path_is_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    custom_root = tmp_path / "custom-root"
    custom_root.mkdir()
    (custom_root / "sessions").write_text("", encoding="utf-8")
    monkeypatch.setenv(SESSION_ROOT_DIR_ENV_VAR, str(custom_root))
    _patch_platform_config(monkeypatch, config_dir)

    sessions_dir = resolve_sessions_dir()

    assert sessions_dir == config_dir / "sessions"
    assert sessions_dir.is_dir()
    assert JsonFileStateStore().session_dir("sess1").parent == sessions_dir


def test_resolve_sessions_dir_caches_successful_write_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_module._RESOLVED_SESSIONS_DIR_CACHE.clear()
    custom_root = tmp_path / "custom-root"
    monkeypatch.setenv(SESSION_ROOT_DIR_ENV_VAR, str(custom_root))
    calls = 0
    real_temporary_file = settings_module.tempfile.TemporaryFile

    def counting_temporary_file(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return real_temporary_file(*args, **kwargs)

    monkeypatch.setattr(settings_module.tempfile, "TemporaryFile", counting_temporary_file)

    first = resolve_sessions_dir()
    second = resolve_sessions_dir()

    assert first == custom_root / "sessions"
    assert second == first
    assert calls == 1


@pytest.mark.asyncio
async def test_save_and_load_session(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    state = {
        "messages": [Message("user", ["hello"])],
        "compressed_msgs": [],
    }
    await store.save_session("sess1", state, agent_profile="code")
    loaded = await store.load_session("sess1")

    assert loaded is not None
    assert len(loaded["messages"]) == 1
    assert loaded["messages"][0].role == "user"


@pytest.mark.asyncio
async def test_load_session_accepts_legacy_serialized_history(tmp_path: Path) -> None:
    """Legacy-compatible type ids deserialize into Chrys-owned Message/Content."""
    store = JsonFileStateStore(tmp_path)
    legacy_messages = [
        Message("user", [Content.from_text("legacy prompt")]),
        Message(
            "assistant",
            [
                Content.from_text("legacy tool call"),
                Content.from_function_call("call_legacy", "read_file", arguments={"path": "a.py"}),
            ],
        ),
        Message(
            "tool",
            [Content.from_function_result("call_legacy", result="contents")],
        ),
    ]
    envelope = {
        "meta": {"session_id": "framework-legacy"},
        "state": {
            "messages": [message.to_dict() for message in legacy_messages],
            "compressed_msgs": [],
            "turn_counter": 1,
        },
    }
    session_file = store.session_dir("framework-legacy") / "session.json"
    session_file.parent.mkdir(parents=True, exist_ok=True)
    session_file.write_text(json.dumps(envelope), encoding="utf-8")

    loaded = await store.load_session("framework-legacy")

    assert loaded is not None
    restored_messages = loaded["messages"]
    assert [type(message) for message in restored_messages] == [Message, Message, Message]
    assert [message.to_dict() for message in restored_messages] == [message.to_dict() for message in legacy_messages]
    assert all(type(content) is Content for message in restored_messages for content in message.contents)


@pytest.mark.asyncio
async def test_load_nonexistent_session(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    result = await store.load_session("nope")
    assert result is None


@pytest.mark.asyncio
async def test_list_sessions(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["a"])], "compressed_msgs": []}, agent_profile="code")
    await store.save_session("s2", {"messages": [], "compressed_msgs": []}, agent_profile="task")

    sessions = await store.list_sessions()
    assert len(sessions) == 2
    ids = {s.session_id for s in sessions}
    assert ids == {"s1", "s2"}


@pytest.mark.asyncio
async def test_a_session_whose_delete_could_not_finish_is_not_listed_again(tmp_path: Path) -> None:
    """A delete that left remains behind (an open trajectory file) still means deleted."""
    from chrys.service.trajectory.tombstone import INTENT_SUFFIX, tombstones_dir

    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["a"])], "compressed_msgs": []}, agent_profile="code")
    await store.save_session("s2", {"messages": [Message("user", ["b"])], "compressed_msgs": []}, agent_profile="code")
    doomed = store.session_dir("s1")

    graveyard = tombstones_dir(tmp_path)
    graveyard.mkdir(parents=True, exist_ok=True)
    (graveyard / f"{doomed.name}{INTENT_SUFFIX}").write_text(doomed.name, encoding="utf-8")

    listed = {meta.session_id for meta in await store.list_sessions()}
    assert listed == {"s2"}


@pytest.mark.asyncio
async def test_list_sessions_turn_count_from_state_counter(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "turny",
        {"messages": [Message("user", ["a"])], "compressed_msgs": [], "turn_counter": 7},
    )

    sessions = await store.list_sessions()

    assert [s.turn_count for s in sessions] == [7]


@pytest.mark.asyncio
async def test_list_sessions_total_tokens_from_state(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "tokeny",
        {"messages": [Message("user", ["a"])], "compressed_msgs": [], "total_session_tokens": 123_456},
    )
    await store.save_session("pre-usage", {"messages": [Message("user", ["b"])], "compressed_msgs": []})

    sessions = await store.list_sessions()

    by_id = {s.session_id: s.total_tokens for s in sessions}
    assert by_id == {"tokeny": 123_456, "pre-usage": 0}


@pytest.mark.asyncio
async def test_list_sessions_turn_count_falls_back_to_turn_markers(tmp_path: Path) -> None:
    """Sessions saved before ``turn_counter`` existed count their markers."""
    from chrys.foundation.models.history_markers import HistoryMarkerKind

    store = JsonFileStateStore(tmp_path)
    marker_one = Message("user", ["turn 1"])
    marker_one.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    marker_two = Message("user", ["turn 2"])
    marker_two.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    await store.save_session(
        "legacy-turns",
        {"messages": [Message("user", ["a"]), marker_one, marker_two], "compressed_msgs": []},
    )
    # Strip the counter the serializer stamped, mimicking a pre-counter file.
    session_file = store.session_dir("legacy-turns") / "session.json"
    envelope = json.loads(session_file.read_text(encoding="utf-8"))
    del envelope["state"]["turn_counter"]
    session_file.write_text(json.dumps(envelope), encoding="utf-8")

    sessions = await store.list_sessions()

    assert [s.turn_count for s in sessions] == [2]


@pytest.mark.asyncio
async def test_list_sessions_turn_count_fallback_covers_compacted_turns(tmp_path: Path) -> None:
    """Pre-counter sessions whose markers were compacted away use turn_range."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("compacted", {"messages": [Message("user", ["tail"])], "compressed_msgs": []})
    session_file = store.session_dir("compacted") / "session.json"
    envelope = json.loads(session_file.read_text(encoding="utf-8"))
    del envelope["state"]["turn_counter"]
    envelope["state"]["compressed_msgs"] = [
        {
            "compressed_context_id": "c1",
            "messages": [],
            "summary_text": "…",
            "marker_id": "turn_9",
            "turn_range": [1, 9],
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    session_file.write_text(json.dumps(envelope), encoding="utf-8")

    sessions = await store.list_sessions()

    assert [s.turn_count for s in sessions] == [9]


@pytest.mark.asyncio
async def test_list_sessions_turn_count_ignores_restarted_counter(tmp_path: Path) -> None:
    """A counter restarted below the compacted evidence must not undercount."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("restarted", {"messages": [Message("user", ["tail"])], "compressed_msgs": []})
    session_file = store.session_dir("restarted") / "session.json"
    envelope = json.loads(session_file.read_text(encoding="utf-8"))
    envelope["state"]["turn_counter"] = 2
    envelope["state"]["compressed_msgs"] = [
        {
            "compressed_context_id": "c1",
            "messages": [],
            "summary_text": "…",
            "marker_id": "turn_9",
            "turn_range": [1, 9],
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    session_file.write_text(json.dumps(envelope), encoding="utf-8")

    sessions = await store.list_sessions()

    assert [s.turn_count for s in sessions] == [9]


@pytest.mark.asyncio
async def test_list_sessions_user_prompt_search_text_covers_every_turn(tmp_path: Path) -> None:
    """User text from all turns lands in meta; synthetic history does not."""
    from chrys.foundation.models.history_markers import HistoryMarkerKind

    store = JsonFileStateStore(tmp_path)
    marker = Message("user", ["turn 1"])
    marker.additional_properties[HistoryMarkerKind.KEY] = HistoryMarkerKind.TURN
    nudge = Message("user", ["continue"])
    nudge.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
    await store.save_session(
        "prompty",
        {
            "messages": [
                Message("user", ["first ask about kernels"]),
                Message("assistant", ["assistant reply text"]),
                marker,
                nudge,
                Message("user", ["second ask\nabout   scrollbars"]),
            ],
            "compressed_msgs": [],
        },
    )

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert "first ask about kernels" in text
    # Internal whitespace collapses so a space-separated query matches a
    # phrase that wrapped across lines in the original prompt.
    assert "second ask about scrollbars" in text
    assert "assistant reply" not in text
    assert "turn 1" not in text
    assert "continue" not in text


@pytest.mark.asyncio
async def test_list_sessions_user_prompt_search_text_keeps_head_and_tail(tmp_path: Path) -> None:
    """A pasted-log prompt keeps both edges: the dump's start AND the ask
    typed after it (middle elided, seam joined by a newline)."""
    store = JsonFileStateStore(tmp_path)
    prompt = "HEAD-ERROR-DUMP " + "x" * 3000 + " TAIL-ACTUAL-ASK"
    await store.save_session("pasty", {"messages": [Message("user", [prompt])], "compressed_msgs": []})

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert "HEAD-ERROR-DUMP" in text
    assert "TAIL-ACTUAL-ASK" in text
    assert len(text) <= 2001  # 1000 head + newline seam + 1000 tail
    # The elided middle must not fabricate a match across the seam.
    assert "x" * 2001 not in text


@pytest.mark.asyncio
async def test_list_sessions_user_prompt_search_text_total_cap_keeps_both_ends(tmp_path: Path) -> None:
    """Over-budget sessions keep their earliest AND latest prompts — a big
    early paste must not evict every later turn from the search index."""
    store = JsonFileStateStore(tmp_path)
    messages = [Message("user", [f"turn-{index}-marker " + "y" * 900]) for index in range(20)]
    await store.save_session("chatty", {"messages": messages, "compressed_msgs": []})

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert len(text) <= 8000
    assert "turn-0-marker" in text
    assert "turn-19-marker" in text


@pytest.mark.asyncio
async def test_list_sessions_user_prompt_search_text_exact_budget_keeps_all_turns(tmp_path: Path) -> None:
    """An index whose joined length is EXACTLY the total budget fits —
    separators only exist between excerpts, so charging one for the first
    excerpt overcounted by one and evicted a middle prompt from a
    perfectly fitting session (review-caught)."""
    store = JsonFileStateStore(tmp_path)
    # 9 excerpts of 888 chars join to 9*888 + 8 = 8000 chars exactly.
    messages = [Message("user", [f"turn-{index}-marker" + "y" * 875]) for index in range(9)]
    await store.save_session("snug", {"messages": messages, "compressed_msgs": []})

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert len(text) == 8000
    for index in range(9):
        assert f"turn-{index}-marker" in text


@pytest.mark.asyncio
async def test_list_sessions_user_prompt_search_text_repeated_final_prompt_survives_cap(tmp_path: Path) -> None:
    """A final-turn prompt that repeats a middle-turn text must stay
    searchable when the session overruns the total budget.  Dedup runs
    INSIDE the front/back selection — collapsing repeats onto their first
    copy up front would pin the text to the elided middle and evict it
    entirely (review-caught)."""
    store = JsonFileStateStore(tmp_path)
    repeat = "repeated ask about flux capacitors"
    messages = [Message("user", [f"turn-{index}-marker " + "y" * 900]) for index in range(10)]
    messages.append(Message("user", [repeat]))
    messages.extend(Message("user", [f"turn-{index}-marker " + "y" * 900]) for index in range(10, 20))
    messages.append(Message("user", [repeat]))
    await store.save_session("echoing", {"messages": messages, "compressed_msgs": []})

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert len(text) <= 8000
    # The latest copy sits in the back window even though the first copy
    # was elided with the middle — and it is still indexed only once.
    assert text.count(repeat) == 1
    assert "turn-0-marker" in text
    assert "turn-19-marker" in text


@pytest.mark.asyncio
async def test_list_sessions_user_prompt_search_text_front_repeat_skips_free(tmp_path: Path) -> None:
    """A final-turn repeat of a text already kept in the front window must
    skip for free during the back walk — neither breaking it nor spending
    its budget — so the latest unique prompt still gets indexed."""
    store = JsonFileStateStore(tmp_path)
    repeat = "opening ask typed again at the end"
    messages = [Message("user", [repeat])]
    messages.extend(Message("user", [f"turn-{index}-marker " + "y" * 900]) for index in range(20))
    messages.append(Message("user", [repeat]))
    await store.save_session("bookended", {"messages": messages, "compressed_msgs": []})

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert len(text) <= 8000
    assert text.count(repeat) == 1
    assert "turn-19-marker" in text


@pytest.mark.asyncio
async def test_list_sessions_user_prompt_search_text_reads_legacy_string_contents(tmp_path: Path) -> None:
    """Legacy sessions serialize contents as plain strings — those prompts
    must be searchable too (the title fallback only covers the first one).
    Mirrors replay's rule: a "Content(type=" repr blob is not text."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("legacy", {"messages": [], "compressed_msgs": []})
    session_file = store.session_dir("legacy") / "session.json"
    envelope = json.loads(session_file.read_text(encoding="utf-8"))
    envelope["state"]["messages"] = [
        {
            "type": "message",
            "role": "user",
            "contents": ["first ask about endianness"],
            "message_id": "m1",
            "additional_properties": {},
        },
        {
            "type": "message",
            "role": "user",
            "contents": ["later legacy\nask   about semaphores", "Content(type=whatever repr)"],
            "message_id": "m2",
            "additional_properties": {},
        },
    ]
    session_file.write_text(json.dumps(envelope), encoding="utf-8")

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert "first ask about endianness" in text
    assert "later legacy ask about semaphores" in text
    assert "Content(type=" not in text


def test_collapsed_prompt_excerpt_matches_naive_collapse() -> None:
    """The fixed-block collapse scan must be output-identical to capping
    a full ``" ".join(text.split())`` — including when a 4096-char block
    boundary falls mid-word or mid-whitespace-run, and when
    whitespace-dominated input makes each block yield almost nothing."""
    edge = 1000  # scan blocks are a fixed 4096 chars, independent of edge

    def naive(text: str) -> str:
        collapsed = " ".join(text.split())
        if len(collapsed) <= 2 * edge:
            return collapsed
        return f"{collapsed[:edge]}\n{collapsed[-edge:]}"

    cases = [
        "",
        "hi there",
        "word " * 2000,  # dense short words: one block already fills the cap
        "x" * 5000,  # single mega-word longer than both caps, glued across blocks
        "a" * 3990 + "b" * 30 + " " + "tail words " * 200,  # word straddles the first block boundary
        ("w " * 1995) + "\t\n  " + ("v " * 1995),  # whitespace run straddles a block boundary
        " " * 6000 + "short tail after huge indent",  # all-whitespace leading blocks yield nothing
        "lead words " + " " * 6000 + "z" * 3000,  # whitespace-dominated middle
        "w" * 1500 + " " * 5000 + "z" * 1500,  # tail-side blocks open on whitespace
        "p" * 999 + " " + "q" * 1000,  # collapsed length exactly 2000: uncapped
        "p" * 1000 + " " + "q" * 1000,  # collapsed length 2001: capped
        "word  " * 640,  # raw 3840 shorter than one block yet collapsed 3199 > cap:
        # the tail scan must clamp its first block to the string start, not
        # wrap through a negative index into a tiny suffix
        # (real-session-caught against the earlier growing-window version)
        "　　混合  空白\tacross\nlines　" * 400,  # unicode whitespace
        # ~2MB whitespace-dominated with period 509 (does not divide the
        # 4096 scan block): boundaries drift through mid-run, mid-word and
        # word-edge phases across thousands of stitches.
        (" " * 507 + "wx") * 4000,
        # Words straddling exact block boundaries in an under-cap text:
        # the head path must glue the split word, not space it apart.
        " " * 4090 + "straddleword" + " " * 4090 + "tail",
    ]
    for text in cases:
        assert JsonFileStateStore._collapsed_prompt_excerpt(text, edge) == naive(text), repr(text[:60])


def test_message_prompt_excerpt_is_memory_bounded() -> None:
    """A multi-megabyte pasted log must not be word-split whole —
    ``str.split()`` transiently costs ~14x the text size; one meta scan
    parses sessions serially in its to_thread worker, but independent
    callers can overlap and multiply the peak.  Fixed-block collapse keeps
    the peak near the excerpt caps for dense AND whitespace-dominated
    input (a growing window slice would copy O(input) raw text on the
    latter before yielding enough collapsed chars)."""
    import tracemalloc

    cases = [
        ("err " * 500_000, "err", 1_000_000),  # 2MB dense short words: split() peaks ~28MB
        ((" " * 499 + "z") * 4000, "z", 400_000),  # 2MB, 0.2% density: doubling windows peaked ~512KB
    ]
    for text, needle, bound in cases:
        message = {
            "type": "message",
            "role": "user",
            "contents": [{"type": "text", "text": text}],
            "additional_properties": {},
        }
        tracemalloc.start()
        excerpt = JsonFileStateStore._message_prompt_excerpt(message)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert excerpt is not None
        assert needle in excerpt
        assert len(excerpt) <= 2001
        assert peak < bound, f"peak {peak} for needle {needle!r}"


@pytest.mark.asyncio
async def test_list_sessions_user_prompt_search_text_includes_compacted_turns(tmp_path: Path) -> None:
    """Prompts folded into compressed_msgs blocks stay searchable, and a
    message present in both a block and the live list counts once."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("compacted-prompts", {"messages": [], "compressed_msgs": []})
    session_file = store.session_dir("compacted-prompts") / "session.json"
    envelope = json.loads(session_file.read_text(encoding="utf-8"))

    def user_msg(message_id: str, text: str) -> dict:
        return {
            "type": "message",
            "role": "user",
            "contents": [{"type": "text", "text": text}],
            "message_id": message_id,
            "additional_properties": {},
        }

    envelope["state"]["messages"] = [user_msg("m2", "live twin popcount ask")]
    envelope["state"]["compressed_msgs"] = [
        {
            "compressed_context_id": "c1",
            "messages": [
                user_msg("m1", "folded ask about quaternions"),
                user_msg("m2", "live twin popcount ask"),
            ],
            "summary_text": "…",
            "marker_id": "turn_1",
            "turn_range": [1, 2],
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    session_file.write_text(json.dumps(envelope), encoding="utf-8")

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert "quaternions" in text
    assert text.count("popcount") == 1


@pytest.mark.asyncio
async def test_list_sessions_user_prompt_search_text_id_reuse_keeps_both_prompts(tmp_path: Path) -> None:
    """message_id restarts across runs, so a folded and a live prompt can
    share an id while being DIFFERENT messages — both must stay searchable
    (dedup is by indexed text, never by id).  A synthetic marker squatting
    on a live prompt's id must not suppress it either."""
    from chrys.foundation.models.history_markers import HistoryMarkerKind

    store = JsonFileStateStore(tmp_path)
    await store.save_session("id-reuse", {"messages": [], "compressed_msgs": []})
    session_file = store.session_dir("id-reuse") / "session.json"
    envelope = json.loads(session_file.read_text(encoding="utf-8"))

    def user_msg(message_id: str, text: str, *, kind: str = "") -> dict:
        props: dict = {HistoryMarkerKind.KEY: kind} if kind else {}
        return {
            "type": "message",
            "role": "user",
            "contents": [{"type": "text", "text": text}],
            "message_id": message_id,
            "additional_properties": props,
        }

    envelope["state"]["messages"] = [
        user_msg("msg_1", "later ask about heisenbugs"),
        user_msg("msg_2", "final ask about ringbuffers"),
    ]
    envelope["state"]["compressed_msgs"] = [
        {
            "compressed_context_id": "c1",
            "messages": [
                user_msg("msg_1", "early ask about monoids"),
                user_msg("msg_2", "turn 2", kind=HistoryMarkerKind.TURN),
            ],
            "summary_text": "…",
            "marker_id": "turn_1",
            "turn_range": [1, 2],
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    ]
    session_file.write_text(json.dumps(envelope), encoding="utf-8")

    sessions = await store.list_sessions()

    text = sessions[0].user_prompt_search_text
    assert "monoids" in text
    assert "heisenbugs" in text
    assert "ringbuffers" in text
    assert "turn 2" not in text


@pytest.mark.asyncio
async def test_legacy_flat_files_with_duplicate_ids_list_once(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    envelope = {
        "meta": {
            "session_id": "dupe",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "message_count": 0,
        },
        "state": {"messages": [], "compressed_msgs": []},
    }
    (tmp_path / "aaa.json").write_text(json.dumps(envelope), encoding="utf-8")
    (tmp_path / "bbb.json").write_text(json.dumps(envelope), encoding="utf-8")

    sessions = await store.list_sessions()

    assert [s.session_id for s in sessions] == ["dupe"]


@pytest.mark.asyncio
async def test_list_sessions_serves_cached_meta_until_file_changes(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("cached", {"messages": [Message("user", ["a"])], "compressed_msgs": []})

    first = await store.list_sessions()
    second = await store.list_sessions()

    # Unchanged on disk — the exact cached object is reused, not re-parsed.
    assert second[0] is first[0]

    await store.save_session(
        "cached",
        {"messages": [Message("user", ["a"]), Message("user", ["b"])], "compressed_msgs": []},
    )
    third = await store.list_sessions()

    assert third[0] is not first[0]
    assert third[0].message_count == 2


@pytest.mark.asyncio
async def test_list_sessions_cache_evicts_deleted_sessions(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("keep", {"messages": [], "compressed_msgs": []})
    await store.save_session("drop", {"messages": [], "compressed_msgs": []})
    await store.list_sessions()

    await store.delete_session("drop")
    sessions = await store.list_sessions()

    assert [s.session_id for s in sessions] == ["keep"]
    assert set(store._meta_cache) == {"keep"}


@pytest.mark.asyncio
async def test_stream_session_metas_yields_batches_matching_list(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    for index in range(5):
        await store.save_session(f"s{index}", {"messages": [], "compressed_msgs": []})

    batches = [batch async for batch in store.stream_session_metas(batch_size=2)]

    assert [len(batch) for batch in batches] == [2, 2, 1]
    streamed_ids = [meta.session_id for batch in batches for meta in batch]
    listed_ids = [meta.session_id for meta in await store.list_sessions()]
    assert streamed_ids == listed_ids


@pytest.mark.asyncio
async def test_stream_session_metas_includes_legacy_flat_files(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("modern", {"messages": [], "compressed_msgs": []})
    (tmp_path / "oldstyle.json").write_text(
        json.dumps(
            {
                "meta": {
                    "session_id": "oldstyle",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "message_count": 0,
                },
                "state": {"messages": [], "compressed_msgs": []},
            }
        ),
        encoding="utf-8",
    )

    streamed_ids = [meta.session_id async for batch in store.stream_session_metas() for meta in batch]

    assert sorted(streamed_ids) == ["modern", "oldstyle"]


@pytest.mark.asyncio
async def test_list_sessions_ignores_lock_directory(tmp_path: Path) -> None:
    """Root-level .locks metadata is not a session directory."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("real", {"messages": [Message("user", ["a"])], "compressed_msgs": []})

    lock_dir = tmp_path / ".locks"
    lock_dir.mkdir(exist_ok=True)
    (lock_dir / "session.json").write_text(
        json.dumps(
            {
                "meta": {
                    "session_id": "lock-metadata",
                    "agent_profile": "not-a-session",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "message_count": 0,
                },
                "state": {"messages": [], "compressed_msgs": []},
            }
        ),
        encoding="utf-8",
    )

    sessions = await store.list_sessions()

    assert [s.session_id for s in sessions] == ["real"]


@pytest.mark.asyncio
async def test_list_sessions_skips_malformed_timestamp_metadata(tmp_path: Path) -> None:
    """One bad timestamp must not abort folder or legacy session listing."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("good", {"messages": [], "compressed_msgs": []})

    bad_folder = tmp_path / "bad-folder"
    bad_folder.mkdir()
    bad_envelope = {
        "meta": {
            "session_id": "bad-folder",
            "created_at": "not-a-timestamp",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "message_count": 0,
        },
        "state": {"messages": [], "compressed_msgs": []},
    }
    (bad_folder / "session.json").write_text(json.dumps(bad_envelope), encoding="utf-8")

    bad_legacy = {
        "meta": {
            "session_id": "bad-legacy",
            "created_at": 123,
            "updated_at": "2026-01-01T00:00:00+00:00",
            "message_count": 0,
        },
        "state": {"messages": [], "compressed_msgs": []},
    }
    (tmp_path / "bad-legacy.json").write_text(json.dumps(bad_legacy), encoding="utf-8")

    sessions = await store.list_sessions()
    streamed_ids = [meta.session_id async for batch in store.stream_session_metas() for meta in batch]

    assert [s.session_id for s in sessions] == ["good"]
    assert streamed_ids == ["good"]


@pytest.mark.asyncio
async def test_delete_session(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("del_me", {"messages": [], "compressed_msgs": []})
    session_dir = tmp_path / "del_me"
    snap_dir = session_dir / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "turn_2.json").write_text("{}", encoding="utf-8")
    store.active_owner_path("del_me").write_text("{}", encoding="utf-8")

    await store.delete_session("del_me")

    assert await store.load_session("del_me") is None
    assert not session_dir.exists()
    assert not store.active_owner_path("del_me").exists()


@pytest.mark.asyncio
async def test_delete_session_surfaces_an_unowned_surviving_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("still_here", {"messages": [], "compressed_msgs": []})
    store._pending_custom_titles["still_here"] = "Keep me"

    def fail_without_owner(*_args: object, **_kwargs: object) -> DeleteResult:
        return DeleteResult(DeleteOutcome.INTENT_FAILED)

    monkeypatch.setattr(tombstone_module, "delete_session_directory", fail_without_owner)

    with pytest.raises(OSError, match="could not be deleted or scheduled"):
        await store.delete_session("still_here")

    assert await store.load_session("still_here") is not None
    assert await store.load_latest_session_id() == "still_here"
    assert store._pending_custom_titles["still_here"] == "Keep me"


@pytest.mark.asyncio
async def test_delete_session_refuses_active_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store-level delete must not remove a session open in another process."""
    monkeypatch.setattr(store_module, "SESSION_ACTIVE_LOCK_TIMEOUT_SECONDS", 0.05)
    store = JsonFileStateStore(tmp_path)
    await store.save_session("busy", {"messages": [], "compressed_msgs": []})

    held = FileLock(store.active_lock_path("busy"), timeout=1.0)
    held.acquire()
    try:
        with pytest.raises(TimeoutError):
            await store.delete_session("busy")
    finally:
        held.release()

    assert (tmp_path / "busy").is_dir()
    assert await store.load_session("busy") is not None


@pytest.mark.asyncio
async def test_delete_session_allow_active_for_current_owner(tmp_path: Path) -> None:
    """Engine current-session deletion already owns the active lock."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("current", {"messages": [], "compressed_msgs": []})

    held = FileLock(store.active_lock_path("current"), timeout=1.0)
    held.acquire()
    try:
        await store.delete_session("current", allow_active=True)
    finally:
        held.release()

    assert not (tmp_path / "current").exists()


@pytest.mark.asyncio
async def test_fork_copies_document_artifacts_and_resolves_handles_in_fork(tmp_path: Path) -> None:
    from PIL import Image

    from chrys.foundation.models.session_env import SessionEnvironment
    from chrys.foundation.platform.files import secure_open_owner_only_binary
    from chrys.service.tools.builtins.doc_converter import _write_unique_markdown
    from chrys.service.tools.builtins.doc_converter.artifacts import DocumentImageSink
    from chrys.service.tools.builtins.filesystem import FilesystemTools
    from chrys.service.tools.session_artifacts import (
        resolve_document_image_artifact_handle,
        resolve_document_markdown_artifact_handle,
    )

    store = JsonFileStateStore(tmp_path)
    parent_id = "parent-document-image"
    await store.save_session(parent_id, {"messages": [], "compressed_msgs": []})
    parent_dir = store.session_dir(parent_id)
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(image_bytes, format="PNG")
    sink = DocumentImageSink(parent_dir / "doc_converter", source_stem="report")
    assert sink.try_reserve_occurrence()
    occurrence = sink.save_image(image_bytes.getvalue(), location="Page 1", ordinal=1, source_name="page.png")
    assert occurrence is not None
    sink.commit_occurrences((occurrence,))
    handle = occurrence.reference
    resolved_parent_image = resolve_document_image_artifact_handle(handle, parent_dir)
    assert resolved_parent_image is not None
    parent_image = Path(resolved_parent_image)
    parent_absolute_path = str(parent_image)
    markdown_artifact = _write_unique_markdown(str(parent_dir / "doc_converter"), "report", "# Report")
    markdown_handle = markdown_artifact.handle
    resolved_parent_markdown = resolve_document_markdown_artifact_handle(markdown_handle, parent_dir)
    assert resolved_parent_markdown is not None
    parent_markdown = Path(resolved_parent_markdown)

    fork_id = store.fork_session(parent_id)
    fork_dir = store.session_dir(fork_id)
    resolved_fork_image = resolve_document_image_artifact_handle(handle, fork_dir)
    assert resolved_fork_image is not None
    fork_image = Path(resolved_fork_image)
    resolved_fork_markdown = resolve_document_markdown_artifact_handle(markdown_handle, fork_dir)
    assert resolved_fork_markdown is not None
    fork_markdown = Path(resolved_fork_markdown)

    assert fork_image.read_bytes() == parent_image.read_bytes()
    with secure_open_owner_only_binary(fork_image) as copied_image:
        assert copied_image.read() == image_bytes.getvalue()
    reused_sink = DocumentImageSink(fork_dir / "doc_converter", source_stem="reused")
    assert reused_sink.try_reserve_occurrence()
    reused = reused_sink.save_image(
        image_bytes.getvalue(),
        location="Page 1",
        ordinal=1,
        source_name="reused.png",
    )
    assert reused is not None
    assert reused.reference == handle
    reused_sink.commit_occurrences((reused,))
    assert fork_markdown.read_text(encoding="utf-8") == "# Report"
    runtime = SessionEnvironment.capture()
    filesystem = FilesystemTools(runtime, session_dir=fork_dir)
    viewed = filesystem.view_image(handle)
    assert viewed[0].media_type == "image/png"
    assert viewed[0].additional_properties["source_path"] == str(fork_image)
    assert "1|# Report" in filesystem.read_file(markdown_handle)

    parent_image.unlink()
    parent_markdown.unlink()
    assert not Path(parent_absolute_path).exists()
    assert filesystem.view_image(handle)[0].media_type == "image/png"
    assert "1|# Report" in filesystem.read_file(markdown_handle)


@pytest.mark.asyncio
async def test_fork_session_copies_and_rewrites_session_identity(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    parent_id = "parent-session-id"
    parent_dir = store.session_dir(parent_id)
    workspace_file = tmp_path / "workspace" / "file.py"
    workspace_file.parent.mkdir()
    workspace_file.write_text("print('shared workspace')", encoding="utf-8")
    clipboard_dir = parent_dir / "attachments" / "clipboard"
    clipboard_dir.mkdir(parents=True)
    # A double-quote exercises mention quote-escaping but is an illegal path
    # character on Windows; only include it where the filesystem allows it.
    # (The escape round-trip itself is covered platform-independently by the
    # string-only mention tests in tests/orchestration/engine/run/test_attachments.py.)
    clipboard_image = clipboard_dir / ("screen one.png" if sys.platform == "win32" else 'screen "one".png')
    clipboard_image.write_bytes(b"png")
    parent_mention = format_file_mention(clipboard_image)
    second_clipboard_image = clipboard_dir / "screen two.png"
    second_clipboard_image.write_bytes(b"png2")
    second_parent_mention = format_file_mention(second_clipboard_image)
    outside_image = tmp_path / "outside.png"
    outside_image.write_bytes(b"outside")
    outside_mention = format_file_mention(outside_image)
    data_bytes = b"\x00\x01"
    state = {
        "messages": [
            Message(
                "user",
                [
                    f"look at {parent_mention} then {second_parent_mention}; leave {outside_mention}",
                    Content.from_data(data=data_bytes, media_type="image/png"),
                ],
            ),
            Message("assistant", [f"the original path was {parent_mention}"]),
        ],
        "compressed_msgs": [
            CompressedBlock(
                compressed_context_id="ctx_clipboard",
                messages=[
                    Message("user", [f"compressed {parent_mention} and {second_parent_mention}"]),
                    Message("assistant", [f"compressed assistant keeps {parent_mention}"]),
                ],
                summary_text="compressed clipboard mentions",
                marker_id="turn_1",
                turn_range=(1, 1),
                created_at="2026-03-17T00:00:00+00:00",
            )
        ],
        "chrys_mutations": {
            "turns": [
                {
                    "turn_id": 1,
                    "mutations": [
                        {
                            "path": str(workspace_file),
                            "operation": "modify",
                            "source": "edit_file",
                            "tool_call_id": "call_1",
                            "timestamp": 1.0,
                            "before_hash": "old",
                            "after_hash": "new",
                        }
                    ],
                }
            ],
        },
    }
    await store.save_session(
        parent_id,
        state,
        agent_profile="Code",
        primary_cwd=str(workspace_file.parent),
        working_dirs=[str(workspace_file.parent)],
        service_session_id="provider-session",
    )
    (parent_dir / "mutations").mkdir()
    (parent_dir / "mutations" / "marker.txt").write_text("mutation data", encoding="utf-8")
    (parent_dir / SESSION_RECOVERY_FILE_NAME).write_text("{}", encoding="utf-8")
    (parent_dir / RAW_HTTP_LOG_FILE_NAME).write_text(
        json.dumps({"session_id": parent_id}) + "\n",
        encoding="utf-8",
    )
    snapshots = parent_dir / "snapshots"
    snapshots.mkdir()
    (snapshots / "turn_1.json").write_bytes((parent_dir / "session.json").read_bytes())
    (snapshots / "not_a_turn.json").write_text(
        json.dumps({"meta": {"session_id": parent_id}}),
        encoding="utf-8",
    )
    sub_agents = parent_dir / "sub_agents"
    (sub_agents / "pending").mkdir(parents=True)
    (sub_agents / "legacy.json").write_text(json.dumps({"invocation_id": "legacy"}), encoding="utf-8")
    (sub_agents / "pending" / "active.json").write_text(json.dumps({"invocation_id": "active"}), encoding="utf-8")
    (sub_agents / "pending" / "active.json.tmp").write_text("partial", encoding="utf-8")
    (sub_agents / "sessions").mkdir(parents=True)
    (sub_agents / "sessions" / ".Explore_a1b2c3d4e5f6.json.tmp").write_text("partial", encoding="utf-8")
    atomic_write_owner_only_text(
        sub_agents / "sessions" / "Explore_a1b2c3d4e5f6.json",
        json.dumps(
            {
                "meta": {
                    "record_type": "sub_agent_session",
                    "parent_session_id": parent_id,
                    "invocation_id": "a1b2c3d4e5f6",
                    "status": "completed",
                },
                "state": {
                    "messages": [
                        serialize_message(Message("user", [f"sub-agent saw {parent_mention}"])),
                    ]
                },
            }
        ),
    )
    parent_before = {
        path.relative_to(parent_dir): path.read_bytes() for path in parent_dir.rglob("*") if path.is_file()
    }

    fork_id = store.fork_session(parent_id)

    fork_dir = store.session_dir(fork_id)
    assert fork_id != parent_id
    assert fork_dir == tmp_path / session_short_id(fork_id)
    assert fork_dir.is_dir()
    parent_after = {path.relative_to(parent_dir): path.read_bytes() for path in parent_dir.rglob("*") if path.is_file()}
    assert parent_after == parent_before

    parent_envelope = json.loads((parent_dir / "session.json").read_text(encoding="utf-8"))
    fork_envelope = json.loads((fork_dir / "session.json").read_text(encoding="utf-8"))
    parent_meta = parent_envelope["meta"]
    fork_meta = fork_envelope["meta"]
    assert fork_meta["session_id"] == fork_id
    assert fork_meta["parent_session_id"] == parent_id
    assert fork_meta["created_at"] == parent_meta["created_at"]
    assert datetime.fromisoformat(fork_meta["updated_at"]) >= datetime.fromisoformat(parent_meta["updated_at"])
    assert fork_meta["service_session_id"] == ""
    assert fork_meta["primary_cwd"] == str(workspace_file.parent)
    assert fork_meta["working_dirs"] == [str(workspace_file.parent)]

    fork_backup = json.loads((fork_dir / "session.json.bak").read_text(encoding="utf-8"))
    fork_snapshot = json.loads((fork_dir / "snapshots" / "turn_1.json").read_text(encoding="utf-8"))
    assert fork_backup["meta"]["session_id"] == fork_id
    assert fork_snapshot["meta"]["session_id"] == fork_id
    assert fork_snapshot["meta"]["parent_session_id"] == parent_id
    assert (
        json.loads((fork_dir / "snapshots" / "not_a_turn.json").read_text(encoding="utf-8"))["meta"]["session_id"]
        == parent_id
    )
    assert not (fork_dir / SESSION_RECOVERY_FILE_NAME).exists()
    assert not (fork_dir / RAW_HTTP_LOG_FILE_NAME).exists()
    assert not (fork_dir / "sub_agents" / "legacy.json").exists()
    assert not (fork_dir / "sub_agents" / "pending").exists()
    assert not list((fork_dir / "sub_agents").rglob("*.tmp"))
    fork_sub_agent_log = json.loads(
        (fork_dir / "sub_agents" / "sessions" / "Explore_a1b2c3d4e5f6.json").read_text(encoding="utf-8")
    )
    assert fork_sub_agent_log["meta"]["parent_session_id"] == fork_id
    fork_sub_agent_text = fork_sub_agent_log["state"]["messages"][0]["contents"][0]["text"]
    assert format_file_mention(fork_dir / "attachments" / "clipboard" / clipboard_image.name) in fork_sub_agent_text
    assert parent_mention not in fork_sub_agent_text
    assert (fork_dir / "mutations" / "marker.txt").read_text(encoding="utf-8") == "mutation data"
    assert (fork_dir / "attachments" / "clipboard" / clipboard_image.name).read_bytes() == b"png"
    assert (fork_dir / "attachments" / "clipboard" / second_clipboard_image.name).read_bytes() == b"png2"

    fork_clipboard_image = fork_dir / "attachments" / "clipboard" / clipboard_image.name
    fork_second_clipboard_image = fork_dir / "attachments" / "clipboard" / second_clipboard_image.name
    user_text = fork_envelope["state"]["messages"][0]["contents"][0]["text"]
    assistant_text = fork_envelope["state"]["messages"][1]["contents"][0]["text"]
    assert format_file_mention(fork_clipboard_image) in user_text
    assert format_file_mention(fork_second_clipboard_image) in user_text
    assert outside_mention in user_text
    assert parent_mention not in user_text
    assert second_parent_mention not in user_text
    assert str(parent_dir) not in user_text
    assert parent_mention in assistant_text
    compressed_user_text = fork_envelope["state"]["compressed_msgs"][0]["messages"][0]["contents"][0]["text"]
    compressed_assistant_text = fork_envelope["state"]["compressed_msgs"][0]["messages"][1]["contents"][0]["text"]
    assert format_file_mention(fork_clipboard_image) in compressed_user_text
    assert format_file_mention(fork_second_clipboard_image) in compressed_user_text
    assert parent_mention not in compressed_user_text
    assert second_parent_mention not in compressed_user_text
    assert parent_mention in compressed_assistant_text
    assert (
        fork_envelope["state"]["messages"][0]["contents"][1]["uri"]
        == parent_envelope["state"]["messages"][0]["contents"][1]["uri"]
    )
    assert fork_envelope["state"]["chrys_mutations"]["turns"][0]["mutations"][0]["path"] == str(workspace_file)


@pytest.mark.asyncio
async def test_fork_drops_the_turn_registry_that_names_the_parents_log(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    registry = {"turns": {"1": {"turn_id": "a" * 32, "started_sequence": 42}}}
    await store.save_session(
        "parent",
        {"messages": [Message("user", ["hi"])], "compressed_msgs": [], TRAJECTORY_STATE_KEY: dict(registry)},
    )
    parent_dir = store.session_dir("parent")
    snapshots = parent_dir / "snapshots"
    snapshots.mkdir()
    (snapshots / "turn_1.json").write_bytes((parent_dir / "session.json").read_bytes())

    fork_id = store.fork_session("parent")

    fork_dir = store.session_dir(fork_id)
    for name in ("session.json", "snapshots/turn_1.json"):
        envelope = json.loads((fork_dir / name).read_text(encoding="utf-8"))
        # The fork numbers its own log from one, so a sequence copied from the
        # parent would make a rollback here name a range that never existed.
        assert TRAJECTORY_STATE_KEY not in envelope["state"], name
    parent_envelope = json.loads((parent_dir / "session.json").read_text(encoding="utf-8"))
    assert parent_envelope["state"][TRAJECTORY_STATE_KEY] == registry


@pytest.mark.asyncio
async def test_fork_session_retries_short_id_collisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("parent", {"messages": [Message("user", ["hi"])], "compressed_msgs": []})
    first = UUID("11111111-1111-4111-8111-111111111111")
    second = UUID("22222222-2222-4222-8222-222222222222")
    temp = UUID("33333333-3333-4333-8333-333333333333")
    store.session_dir(str(first)).mkdir()
    values = iter([first, second, temp])
    monkeypatch.setattr(store_module, "uuid4", lambda: next(values))

    fork_id = store.fork_session("parent")

    assert fork_id == str(second)
    assert store.session_dir(fork_id).is_dir()


@pytest.mark.asyncio
async def test_fork_session_skips_invalid_and_malformed_auxiliary_envelopes(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    parent_id = "parent"
    await store.save_session(parent_id, {"messages": [Message("user", ["hi"])], "compressed_msgs": []})
    parent_dir = store.session_dir(parent_id)
    (parent_dir / "session.json.bak").write_text(
        json.dumps({"state": {"messages": [], "compressed_msgs": []}}),
        encoding="utf-8",
    )
    snapshots = parent_dir / "snapshots"
    snapshots.mkdir()
    (snapshots / "turn_1.json").write_text("{ broken snapshot", encoding="utf-8")
    (snapshots / "turn_2.json").write_bytes((parent_dir / "session.json").read_bytes())
    (snapshots / "turn_3.json").write_text(
        json.dumps({"state": {"messages": [], "compressed_msgs": []}}),
        encoding="utf-8",
    )

    fork_id = store.fork_session(parent_id)

    fork_dir = store.session_dir(fork_id)
    fork_primary = json.loads((fork_dir / "session.json").read_text(encoding="utf-8"))
    fork_backup = json.loads((fork_dir / "session.json.bak").read_text(encoding="utf-8"))
    fork_snapshot = json.loads((fork_dir / "snapshots" / "turn_2.json").read_text(encoding="utf-8"))
    assert fork_primary["meta"]["session_id"] == fork_id
    assert fork_backup["meta"]["session_id"] == fork_id
    assert fork_backup["state"]["messages"][0]["contents"][0]["text"] == "hi"
    assert not (fork_dir / "snapshots" / "turn_1.json").exists()
    assert not (fork_dir / "snapshots" / "turn_3.json").exists()
    assert fork_snapshot["meta"]["session_id"] == fork_id
    assert fork_snapshot["meta"]["parent_session_id"] == parent_id


@pytest.mark.asyncio
async def test_fork_session_retries_destination_recheck_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("parent", {"messages": [Message("user", ["hi"])], "compressed_msgs": []})
    first = UUID("11111111-1111-4111-8111-111111111111")
    second = UUID("22222222-2222-4222-8222-222222222222")
    temp = UUID("33333333-3333-4333-8333-333333333333")
    values = iter([first, second, temp])
    monkeypatch.setattr(store_module, "uuid4", lambda: next(values))
    real_assert_destination_available = store._assert_fork_destination_available
    collided = False

    def flaky_assert_destination_available(session_id: str, *, include_write_lock: bool = True) -> None:
        nonlocal collided
        if session_id == str(first) and not include_write_lock and not collided:
            collided = True
            store.session_dir(session_id).mkdir()
            raise SessionForkError("Fork destination collision at simulated target")
        real_assert_destination_available(session_id, include_write_lock=include_write_lock)

    monkeypatch.setattr(store, "_assert_fork_destination_available", flaky_assert_destination_available)

    fork_id = store.fork_session("parent")

    assert collided is True
    assert fork_id == str(second)
    assert store.session_dir(fork_id).is_dir()


@pytest.mark.asyncio
async def test_fork_session_stops_after_collision_retry_cap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("parent", {"messages": [Message("user", ["hi"])], "compressed_msgs": []})
    colliding = UUID("11111111-1111-4111-8111-111111111111")
    store.session_dir(str(colliding)).mkdir()
    monkeypatch.setattr(store_module, "uuid4", lambda: colliding)

    with pytest.raises(SessionForkError):
        store.fork_session("parent")


@pytest.mark.asyncio
async def test_fork_session_failure_cleans_temp_dir_and_preserves_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("parent", {"messages": [Message("user", ["hi"])], "compressed_msgs": []})
    parent_dir = store.session_dir("parent")
    parent_before = {
        path.relative_to(parent_dir): path.read_bytes() for path in parent_dir.rglob("*") if path.is_file()
    }

    def fail_rewrite(*args: object, **kwargs: object) -> None:
        raise RuntimeError("rewrite failed")

    monkeypatch.setattr(JsonFileStateStore, "_rewrite_fork_envelope_file", fail_rewrite)

    with pytest.raises(SessionForkError, match="Failed to fork session"):
        store.fork_session("parent")

    parent_after = {path.relative_to(parent_dir): path.read_bytes() for path in parent_dir.rglob("*") if path.is_file()}
    assert parent_after == parent_before
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".") and path.name != ".locks"]


@pytest.mark.asyncio
async def test_save_updates_metadata(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [], "compressed_msgs": []}, agent_profile="code")
    sessions = await store.list_sessions()
    assert sessions[0].agent_profile == "code"
    assert sessions[0].message_count == 0

    # Save again with more messages
    await store.save_session("s1", {"messages": [Message("user", ["hi"])], "compressed_msgs": []})
    sessions = await store.list_sessions()
    assert sessions[0].message_count == 1
    # created_at should be preserved
    assert sessions[0].agent_profile == "code"


@pytest.mark.asyncio
async def test_save_inherits_metadata_from_valid_backup_hidden_by_malformed_primary(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    state = {"messages": [Message("user", ["old"])], "compressed_msgs": []}
    await store.save_session("s1", state, agent_profile="code")
    session_dir = store.session_dir("s1")
    primary_file = session_dir / "session.json"
    backup_file = session_dir / "session.json.bak"
    primary = json.loads(primary_file.read_text(encoding="utf-8"))
    backup = json.loads(backup_file.read_text(encoding="utf-8"))
    inherited_created_at = "2020-01-01T00:00:00+00:00"
    backup["meta"]["created_at"] = inherited_created_at
    backup["meta"]["updated_at"] = "2020-01-02T00:00:00+00:00"
    backup_file.write_text(json.dumps(backup), encoding="utf-8")
    primary["meta"] = "malformed"
    primary_file.write_text(json.dumps(primary), encoding="utf-8")

    await store.save_session(
        "s1",
        {"messages": [Message("user", ["old"]), Message("assistant", ["new"])], "compressed_msgs": []},
    )

    saved = json.loads(primary_file.read_text(encoding="utf-8"))
    assert saved["meta"]["created_at"] == inherited_created_at
    assert saved["meta"]["agent_profile"] == "code"


@pytest.mark.asyncio
async def test_recovery_only_created_at_is_stable_and_repairs_drift_from_history(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    old_message = Message("user", ["old message"])
    stamp_message_created_at(old_message, "2020-01-01T00:00:00+00:00")
    state = {"messages": [old_message], "compressed_msgs": []}
    recovery_file = store.session_dir("s1") / SESSION_RECOVERY_FILE_NAME

    await asyncio.to_thread(store.save_recovery_session, "s1", state)
    first = json.loads(recovery_file.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(first["meta"]["created_at"]) == datetime(2020, 1, 1, tzinfo=UTC)
    assert first["meta"]["created_at"] != first["meta"]["updated_at"]

    drifted = datetime.now(UTC).isoformat()
    first["meta"]["created_at"] = drifted
    first["meta"]["updated_at"] = drifted
    recovery_file.write_text(json.dumps(first), encoding="utf-8")

    await asyncio.to_thread(store.save_recovery_session, "s1", state)
    second = json.loads(recovery_file.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(second["meta"]["created_at"]) == datetime(2020, 1, 1, tzinfo=UTC)

    await store.save_session("s1", state)
    primary = json.loads((store.session_dir("s1") / "session.json").read_text(encoding="utf-8"))
    assert datetime.fromisoformat(primary["meta"]["created_at"]) == datetime(2020, 1, 1, tzinfo=UTC)
    assert primary["state"]["messages"][0]["additional_properties"][MESSAGE_CREATED_AT_KEY].startswith("2020-01-01")


@pytest.mark.asyncio
async def test_missing_created_at_repairs_only_that_field(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    message = Message("user", ["old message"])
    stamp_message_created_at(message, "2020-01-01T00:00:00+00:00")
    state = {"messages": [message], "compressed_msgs": []}
    await store.save_session(
        "s1",
        state,
        agent_profile="code",
        model_provider="openai",
        primary_cwd="/work",
    )
    await store.update_session_titles("s1", custom_title="Pinned")
    session_dir = store.session_dir("s1")
    for path in (session_dir / "session.json", session_dir / "session.json.bak"):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["meta"].pop("created_at")
        path.write_text(json.dumps(envelope), encoding="utf-8")

    await store.save_session("s1", state)

    saved = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))["meta"]
    assert datetime.fromisoformat(saved["created_at"]) == datetime(2020, 1, 1, tzinfo=UTC)
    assert saved["custom_title"] == "Pinned"
    assert saved["agent_profile"] == "code"
    assert saved["model_provider"] == "openai"
    assert saved["primary_cwd"] == "/work"


@pytest.mark.asyncio
async def test_clock_rollback_clamps_updated_at_without_discarding_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SkewedDateTime(datetime):
        current = datetime(2026, 1, 1, 12, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return cls.current.replace(tzinfo=None)
            return cls.current.astimezone(tz)

    monkeypatch.setattr(store_module, "datetime", SkewedDateTime)
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "s1",
        {"messages": [Message("user", ["one"])], "compressed_msgs": []},
        agent_profile="code",
        model_provider="openai",
        primary_cwd="/work",
    )
    await store.update_session_titles("s1", custom_title="Pinned")

    SkewedDateTime.current = datetime(2026, 1, 1, 11, tzinfo=UTC)
    await store.save_session(
        "s1",
        {"messages": [Message("user", ["one"]), Message("assistant", ["two"])], "compressed_msgs": []},
    )
    skewed = json.loads((store.session_dir("s1") / "session.json").read_text(encoding="utf-8"))["meta"]
    assert datetime.fromisoformat(skewed["created_at"]) == datetime(2026, 1, 1, 12, tzinfo=UTC)
    assert datetime.fromisoformat(skewed["updated_at"]) == datetime(2026, 1, 1, 12, tzinfo=UTC)

    SkewedDateTime.current = datetime(2026, 1, 1, 13, tzinfo=UTC)
    await store.save_session(
        "s1",
        {
            "messages": [
                Message("user", ["one"]),
                Message("assistant", ["two"]),
                Message("user", ["three"]),
            ],
            "compressed_msgs": [],
        },
    )
    saved = json.loads((store.session_dir("s1") / "session.json").read_text(encoding="utf-8"))["meta"]
    assert saved["custom_title"] == "Pinned"
    assert saved["agent_profile"] == "code"
    assert saved["model_provider"] == "openai"
    assert saved["primary_cwd"] == "/work"


@pytest.mark.asyncio
async def test_recovery_timestamp_stays_strictly_newer_when_clock_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SkewedDateTime(datetime):
        current = datetime(2026, 1, 1, 12, tzinfo=UTC)

        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return cls.current.replace(tzinfo=None)
            return cls.current.astimezone(tz)

    monkeypatch.setattr(store_module, "datetime", SkewedDateTime)
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "s1",
        {"messages": [Message("user", ["primary one"])], "compressed_msgs": []},
    )
    SkewedDateTime.current = datetime(2026, 1, 1, 15, tzinfo=UTC)
    await store.save_session(
        "s1",
        {
            "messages": [Message("user", ["primary one"]), Message("assistant", ["primary two"])],
            "compressed_msgs": [],
        },
    )

    SkewedDateTime.current = datetime(2026, 1, 1, 14, tzinfo=UTC)
    recovery_state = {"messages": [Message("user", ["recovery wins"])], "compressed_msgs": []}
    await asyncio.to_thread(store.save_recovery_session, "s1", recovery_state)

    session_dir = store.session_dir("s1")
    primary = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    recovery_file = session_dir / SESSION_RECOVERY_FILE_NAME
    recovery = json.loads(recovery_file.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(recovery["meta"]["updated_at"]) > datetime.fromisoformat(
        primary["meta"]["updated_at"]
    )

    loaded = await store.load_session("s1", prefer_recovery=True)
    assert loaded is not None
    assert loaded["messages"][0].text == "recovery wins"
    assert recovery_file.exists()


@pytest.mark.asyncio
async def test_healthy_primary_checkpoint_does_not_parse_existing_recovery_meta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    state = {"messages": [Message("user", ["hello"])], "compressed_msgs": []}
    await store.save_session("s1", state)
    await asyncio.to_thread(store.save_recovery_session, "s1", state)

    def fail_recovery_read(_session_id: str) -> dict[str, object]:
        raise AssertionError("healthy checkpoint parsed the recovery sidecar")

    monkeypatch.setattr(store, "_read_recovery_meta_unlocked", fail_recovery_read)
    await asyncio.to_thread(store.save_recovery_session, "s1", state)


@pytest.mark.asyncio
async def test_naive_created_at_clamp_and_recovery_arbitration_are_utc_normalized(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    primary_state = {"messages": [Message("user", ["primary"])], "compressed_msgs": []}
    await store.save_session("s1", primary_state)
    session_dir = store.session_dir("s1")
    primary_file = session_dir / "session.json"
    backup_file = session_dir / "session.json.bak"
    for path in (primary_file, backup_file):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        envelope["meta"]["created_at"] = "2030-01-01T12:00:00"
        envelope["meta"]["updated_at"] = "2029-01-01T00:00:00+00:00"
        path.write_text(json.dumps(envelope), encoding="utf-8")

    await store.save_session("s1", primary_state)
    primary = json.loads(primary_file.read_text(encoding="utf-8"))
    assert primary["meta"]["created_at"] == "2030-01-01T12:00:00"
    assert primary["meta"]["updated_at"] == "2030-01-01T12:00:00+00:00"

    recovery_state = {"messages": [Message("user", ["recovery"])], "compressed_msgs": []}
    await asyncio.to_thread(store.save_recovery_session, "s1", recovery_state)
    recovery_file = session_dir / SESSION_RECOVERY_FILE_NAME
    recovery = json.loads(recovery_file.read_text(encoding="utf-8"))
    recovery["meta"]["updated_at"] = "2031-01-01T00:00:00"
    recovery_file.write_text(json.dumps(recovery), encoding="utf-8")

    loaded = await store.load_session("s1", prefer_recovery=True)
    assert loaded is not None
    assert loaded["messages"][0].text == "recovery"


# --- mid-turn user-message guards (§2.4) ---


def test_is_visible_message_excludes_continuation_counts_injected() -> None:
    """A synthetic ``continue`` nudge is a orchestration placeholder, not user
    content; an injected mid-turn message IS user input."""
    nudge = Message("user", ["continue"])
    nudge.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
    injected = Message("user", ["mid-turn guidance"])
    injected.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True

    assert not _is_visible_message(nudge)
    assert _is_visible_message(injected)
    assert _is_visible_message(Message("user", ["real question"]))


def test_extract_title_skips_leading_continuation_nudge() -> None:
    """The no-prior-user resume shape must not title the session "continue"."""
    nudge = Message("user", ["continue"])
    nudge.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
    real = Message("user", ["fix the flaky scroll test"])

    assert _extract_title([nudge, Message("assistant", ["resuming"]), real]) == "fix the flaky scroll test"
    assert _extract_title([nudge]) == ""


def test_extract_title_accepts_injected_user_message() -> None:
    injected = Message("user", ["mid-turn guidance"])
    injected.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True

    assert _extract_title([injected]) == "mid-turn guidance"


@pytest.mark.asyncio
async def test_message_count_stable_across_continuation_nudge(tmp_path: Path) -> None:
    """A crash-leftover flagged nudge must not inflate message_count — and,
    through it, must not advance updated_at on the next save."""
    store = JsonFileStateStore(tmp_path)
    user = Message("user", ["do the thing"])
    answer = Message("assistant", ["done"])
    await store.save_session("s1", {"messages": [user, answer], "compressed_msgs": []})
    first = (await store.list_sessions())[0]
    assert first.message_count == 2

    nudge = Message("user", ["continue"])
    nudge.additional_properties[HistoryMarkerKind.CONTINUATION_KEY] = True
    await store.save_session("s1", {"messages": [user, answer, nudge], "compressed_msgs": []})
    second = (await store.list_sessions())[0]
    assert second.message_count == 2
    assert second.updated_at == first.updated_at

    injected = Message("user", ["also check the docs"])
    injected.additional_properties[HistoryMarkerKind.INJECTED_KEY] = True
    await store.save_session("s1", {"messages": [user, answer, nudge, injected], "compressed_msgs": []})
    third = (await store.list_sessions())[0]
    assert third.message_count == 3


# --- _dir_size tests ---


def test_dir_size_empty(tmp_path: Path) -> None:
    assert _dir_size(tmp_path) == 0


def test_dir_size_nested(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("abc", encoding="utf-8")
    (tmp_path / "top.txt").write_text("xy", encoding="utf-8")
    assert _dir_size(tmp_path) == 5  # 3 + 2


def test_dir_size_nonexistent(tmp_path: Path) -> None:
    assert _dir_size(tmp_path / "nope") == 0


# --- list_sessions size_bytes ---


@pytest.mark.asyncio
async def test_list_sessions_includes_size_bytes(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["data"])], "compressed_msgs": []})

    sessions = await store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].size_bytes > 0  # session.json has content


# --- crash-safe session.json writes --------------------------------------


@pytest.mark.asyncio
async def test_save_preserves_existing_session_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash during the primary replace keeps the prior session readable."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["old"])], "compressed_msgs": []})

    session_file = tmp_path / "s1" / "session.json"
    before = session_file.read_text(encoding="utf-8")
    real_replace = store_module.os.replace

    def fail_primary_replace(src: object, dst: object) -> None:
        if Path(dst).name == "session.json":
            raise RuntimeError("simulated crash during primary replace")
        real_replace(src, dst)

    monkeypatch.setattr(store_module.os, "replace", fail_primary_replace)

    with pytest.raises(RuntimeError, match="simulated crash"):
        await store.save_session("s1", {"messages": [Message("user", ["new"])], "compressed_msgs": []})

    assert session_file.read_text(encoding="utf-8") == before
    loaded = await store.load_session("s1")
    assert loaded is not None
    assert loaded["messages"][0].text == "old"
    assert not list((tmp_path / "s1").glob("session.json.*.tmp"))


@pytest.mark.asyncio
async def test_load_recovers_corrupt_primary_from_backup(tmp_path: Path) -> None:
    """If primary JSON is corrupt, the backup is used and the primary is healed."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["latest"])], "compressed_msgs": []})

    session_file = tmp_path / "s1" / "session.json"
    session_file.write_text("{ not valid json", encoding="utf-8")

    loaded = await store.load_session("s1")

    assert loaded is not None
    assert loaded["messages"][0].text == "latest"
    healed = json.loads(session_file.read_text(encoding="utf-8"))
    assert healed["meta"]["session_id"] == "s1"


@pytest.mark.asyncio
async def test_load_recovers_missing_primary_from_backup(tmp_path: Path) -> None:
    """If session.json disappears, the backup is enough to restore and heal it."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["backup"])], "compressed_msgs": []})

    session_file = tmp_path / "s1" / "session.json"
    session_file.unlink()

    loaded = await store.load_session("s1")

    assert loaded is not None
    assert loaded["messages"][0].text == "backup"
    assert json.loads(session_file.read_text(encoding="utf-8"))["meta"]["session_id"] == "s1"


@pytest.mark.asyncio
async def test_save_succeeds_when_backup_update_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The backup is best-effort; a .bak write failure must not fail the primary save."""
    store = JsonFileStateStore(tmp_path)
    real_atomic_write_text = store_module._atomic_write_text

    def fail_backup_write(path: Path, payload: str, *, encoding: str = "utf-8") -> bytes:
        if Path(path).name == "session.json.bak":
            raise OSError("simulated backup failure")
        return real_atomic_write_text(path, payload, encoding=encoding)

    monkeypatch.setattr(store_module, "_atomic_write_text", fail_backup_write)

    await store.save_session("s1", {"messages": [Message("user", ["primary"])], "compressed_msgs": []})

    loaded = await store.load_session("s1")
    assert loaded is not None
    assert loaded["messages"][0].text == "primary"
    assert (tmp_path / "s1" / "session.json").exists()


@pytest.mark.asyncio
async def test_recovery_sidecar_wins_when_explicitly_allowed_without_healing_primary(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "s1",
        {"messages": [Message("user", ["primary"])], "compressed_msgs": []},
        service_session_id="provider-session",
    )
    await asyncio.to_thread(
        store.save_recovery_session,
        "s1",
        {"messages": [Message("user", ["recovered"])], "compressed_msgs": []},
    )

    recovery_file = tmp_path / "s1" / SESSION_RECOVERY_FILE_NAME
    recovery = json.loads(recovery_file.read_text(encoding="utf-8"))
    recovery["meta"]["updated_at"] = "2999-01-01T00:00:00+00:00"
    recovery_file.write_text(json.dumps(recovery), encoding="utf-8")

    primary = await store.load_session("s1")
    loaded = await store.load_session("s1", prefer_recovery=True)
    raw = await store.load_session_raw("s1", prefer_recovery=True)
    meta = await store.load_session_meta("s1", prefer_recovery=True)

    assert primary is not None
    assert primary["messages"][0].text == "primary"
    assert loaded is not None
    assert raw is not None
    assert meta is not None
    assert await store.recovery_session_wins("s1") is True
    assert loaded["messages"][0].text == "recovered"
    assert raw[0]["contents"][0]["text"] == "recovered"
    assert meta.service_session_id == ""
    primary = json.loads((tmp_path / "s1" / "session.json").read_text(encoding="utf-8"))
    assert primary["state"]["messages"][0]["contents"][0]["text"] == "primary"


@pytest.mark.asyncio
async def test_save_recovery_session_neutralizes_surrogate_workspace_metadata(tmp_path: Path) -> None:
    """A surrogateescaped workspace path (undecodable byte in cwd/args) must not
    crash the recovery sidecar write. The recovery path funnels through the total
    text sink (atomic_write_text), not a strict bytes encode, so the file stays
    strict-UTF-8 and the crash-recovery checkpoint is not silently lost."""
    store = JsonFileStateStore(tmp_path)
    await asyncio.to_thread(
        store.save_recovery_session,
        "s1",
        {"messages": [Message("user", ["recovered"])], "compressed_msgs": []},
        primary_cwd="/work/pro\udcffject",
        working_dirs=["/root/\udcfe"],
        title="draft \udcfd",
    )
    recovery_file = tmp_path / "s1" / SESSION_RECOVERY_FILE_NAME
    # Discriminator: a strict read of the on-disk bytes must succeed (no lone
    # surrogate in the file); the pre-fix strict bytes-encode raised here.
    recovery = json.loads(recovery_file.read_text(encoding="utf-8"))
    assert recovery["state"]["messages"][0]["contents"][0]["text"] == "recovered"

    meta = await store.load_session_meta("s1", prefer_recovery=True)

    assert meta is not None
    assert meta.primary_cwd == "/work/pro\udcffject"
    assert meta.working_dirs == ["/root/\udcfe"]


@pytest.mark.asyncio
async def test_load_session_meta_treats_null_workspace_paths_as_empty(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [], "compressed_msgs": []})
    session_file = tmp_path / "s1" / "session.json"
    envelope = json.loads(session_file.read_text(encoding="utf-8"))
    envelope["meta"]["primary_cwd"] = None
    envelope["meta"]["working_dirs"] = None
    session_file.write_text(json.dumps(envelope), encoding="utf-8")

    meta = await store.load_session_meta("s1")

    assert meta is not None
    assert meta.primary_cwd == ""
    assert meta.working_dirs == []


@pytest.mark.asyncio
async def test_recovery_title_mirror_neutralizes_surrogate_title(tmp_path: Path) -> None:
    """The title-mirror into a live recovery sidecar shares the same sink; a
    surrogate-bearing title must not crash (or silently skip) the mirror write."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["primary"])], "compressed_msgs": []})
    await asyncio.to_thread(
        store.save_recovery_session,
        "s1",
        {"messages": [Message("user", ["recovered"])], "compressed_msgs": []},
    )
    await store.update_session_titles("s1", custom_title="my \udcff title")
    recovery_file = tmp_path / "s1" / SESSION_RECOVERY_FILE_NAME
    # Discriminator: file strict-UTF-8 readable AND the patch actually landed. The
    # pre-fix strict encode raised inside the (OSError-only) mirror guard, so the
    # mirror was skipped and custom_title stayed unset.
    recovery = json.loads(recovery_file.read_text(encoding="utf-8"))
    assert (recovery["meta"].get("custom_title") or "").startswith("my ")


@pytest.mark.asyncio
async def test_stale_recovery_sidecar_is_ignored_and_deleted(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["primary"])], "compressed_msgs": []})
    await asyncio.to_thread(
        store.save_recovery_session,
        "s1",
        {"messages": [Message("user", ["stale"])], "compressed_msgs": []},
    )

    recovery_file = tmp_path / "s1" / SESSION_RECOVERY_FILE_NAME
    recovery = json.loads(recovery_file.read_text(encoding="utf-8"))
    recovery["meta"]["updated_at"] = "2000-01-01T00:00:00+00:00"
    recovery_file.write_text(json.dumps(recovery), encoding="utf-8")

    loaded = await store.load_session("s1", prefer_recovery=True)

    assert loaded is not None
    assert loaded["messages"][0].text == "primary"
    assert not recovery_file.exists()


@pytest.mark.asyncio
async def test_recovery_sidecar_with_invalid_timestamp_is_ignored_when_primary_exists(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["primary"])], "compressed_msgs": []})
    await asyncio.to_thread(
        store.save_recovery_session,
        "s1",
        {"messages": [Message("user", ["invalid recovery"])], "compressed_msgs": []},
    )

    recovery_file = tmp_path / "s1" / SESSION_RECOVERY_FILE_NAME
    recovery = json.loads(recovery_file.read_text(encoding="utf-8"))
    recovery["meta"]["updated_at"] = "not-a-timestamp"
    recovery_file.write_text(json.dumps(recovery), encoding="utf-8")

    loaded = await store.load_session("s1", prefer_recovery=True)

    assert loaded is not None
    assert loaded["messages"][0].text == "primary"
    assert not recovery_file.exists()


@pytest.mark.asyncio
async def test_primary_save_deletes_recovery_sidecar_after_success(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await asyncio.to_thread(
        store.save_recovery_session,
        "s1",
        {"messages": [Message("user", ["recover"])], "compressed_msgs": []},
    )

    await store.save_session("s1", {"messages": [Message("user", ["primary"])], "compressed_msgs": []})

    assert not (tmp_path / "s1" / SESSION_RECOVERY_FILE_NAME).exists()


@pytest.mark.asyncio
async def test_primary_save_retiring_sidecar_advances_updated_at_before_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["old primary"])], "compressed_msgs": []})
    primary_file = tmp_path / "s1" / "session.json"
    primary = json.loads(primary_file.read_text(encoding="utf-8"))
    primary["meta"]["updated_at"] = "2026-01-01T00:00:00+00:00"
    primary_file.write_text(json.dumps(primary), encoding="utf-8")
    await asyncio.to_thread(
        store.save_recovery_session,
        "s1",
        {"messages": [Message("user", ["recovery"])], "compressed_msgs": []},
    )
    recovery_file = tmp_path / "s1" / SESSION_RECOVERY_FILE_NAME
    recovery = json.loads(recovery_file.read_text(encoding="utf-8"))

    # Simulate a crash after the primary write but before structural sidecar deletion.
    monkeypatch.setattr(store, "_delete_recovery_session_unlocked", lambda _session_id: None)
    await store.save_session("s1", {"messages": [Message("user", ["clean primary"])], "compressed_msgs": []})

    primary_after = json.loads(primary_file.read_text(encoding="utf-8"))
    assert datetime.fromisoformat(primary_after["meta"]["updated_at"]) >= datetime.fromisoformat(
        recovery["meta"]["updated_at"]
    )
    loaded = await store.load_session("s1", prefer_recovery=True)
    assert loaded is not None
    assert loaded["messages"][0].text == "clean primary"


@pytest.mark.asyncio
async def test_primary_save_failure_leaves_recovery_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = JsonFileStateStore(tmp_path)
    await asyncio.to_thread(
        store.save_recovery_session,
        "s1",
        {"messages": [Message("user", ["recover"])], "compressed_msgs": []},
    )
    real_atomic_write_text = store_module._atomic_write_text

    def fail_primary_write(path: Path, payload: str, *, encoding: str = "utf-8") -> None:
        if Path(path).name == "session.json":
            raise OSError("simulated primary failure")
        real_atomic_write_text(path, payload, encoding=encoding)

    monkeypatch.setattr(store_module, "_atomic_write_text", fail_primary_write)

    with pytest.raises(OSError, match="simulated primary failure"):
        await store.save_session("s1", {"messages": [Message("user", ["primary"])], "compressed_msgs": []})

    assert (tmp_path / "s1" / SESSION_RECOVERY_FILE_NAME).exists()


@pytest.mark.asyncio
async def test_list_sessions_includes_recovery_only_sidecar(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await asyncio.to_thread(
        store.save_recovery_session,
        "s1",
        {"messages": [Message("user", ["recover"])], "compressed_msgs": []},
        agent_profile="code",
    )

    sessions = await store.list_sessions()

    assert len(sessions) == 1
    assert sessions[0].session_id == "s1"
    assert sessions[0].agent_profile == "code"


@pytest.mark.asyncio
async def test_list_sessions_ignores_recovery_sidecar_when_active_lock_is_held(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "s1",
        {"messages": [Message("user", ["primary"])], "compressed_msgs": []},
        agent_profile="primary-agent",
    )
    await asyncio.to_thread(
        store.save_recovery_session,
        "s1",
        {"messages": [Message("user", ["recover"])], "compressed_msgs": []},
        agent_profile="recovery-agent",
    )
    recovery_file = tmp_path / "s1" / SESSION_RECOVERY_FILE_NAME
    recovery = json.loads(recovery_file.read_text(encoding="utf-8"))
    recovery["meta"]["updated_at"] = "2999-01-01T00:00:00+00:00"
    recovery_file.write_text(json.dumps(recovery), encoding="utf-8")

    active_lock = FileLock(store.active_lock_path("s1"), timeout=1.0)
    active_lock.acquire()
    try:
        sessions = await store.list_sessions()
    finally:
        active_lock.release()

    assert len(sessions) == 1
    assert sessions[0].agent_profile == "primary-agent"
    assert recovery_file.exists()


@pytest.mark.asyncio
async def test_direct_session_readers_do_not_prefer_recovery_by_default_under_active_lock(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "s1",
        {"messages": [Message("user", ["primary"])], "compressed_msgs": []},
        agent_profile="primary-agent",
    )
    await asyncio.to_thread(
        store.save_recovery_session,
        "s1",
        {"messages": [Message("user", ["recover"])], "compressed_msgs": []},
        agent_profile="recovery-agent",
    )
    recovery_file = tmp_path / "s1" / SESSION_RECOVERY_FILE_NAME
    recovery = json.loads(recovery_file.read_text(encoding="utf-8"))
    recovery["meta"]["updated_at"] = "2999-01-01T00:00:00+00:00"
    recovery_file.write_text(json.dumps(recovery), encoding="utf-8")

    active_lock = FileLock(store.active_lock_path("s1"), timeout=1.0)
    active_lock.acquire()
    try:
        loaded = await store.load_session("s1")
        raw = await store.load_session_raw("s1")
        meta = await store.load_session_meta("s1")
    finally:
        active_lock.release()

    assert loaded is not None
    assert raw is not None
    assert meta is not None
    assert loaded["messages"][0].text == "primary"
    assert raw[0]["contents"][0]["text"] == "primary"
    assert meta.agent_profile == "primary-agent"
    assert recovery_file.exists()


@pytest.mark.asyncio
async def test_list_sessions_skips_recovery_only_sidecar_when_active_lock_is_held(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await asyncio.to_thread(
        store.save_recovery_session,
        "s1",
        {"messages": [Message("user", ["recover"])], "compressed_msgs": []},
        agent_profile="code",
    )
    active_lock = FileLock(store.active_lock_path("s1"), timeout=1.0)
    active_lock.acquire()
    try:
        sessions = await store.list_sessions()
    finally:
        active_lock.release()

    assert sessions == []


@pytest.mark.asyncio
async def test_load_recovers_corrupt_primary_and_backup_from_latest_snapshot(tmp_path: Path) -> None:
    """Rollback snapshots are the last-resort restore point if both live files are bad."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["snapshot"])], "compressed_msgs": []})

    session_dir = tmp_path / "s1"
    snap_dir = session_dir / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "turn_4.json").write_text((session_dir / "session.json").read_text(encoding="utf-8"), encoding="utf-8")
    (session_dir / "session.json").write_text("{ broken primary", encoding="utf-8")
    (session_dir / "session.json.bak").write_text("{ broken backup", encoding="utf-8")

    loaded = await store.load_session("s1")

    assert loaded is not None
    assert loaded["messages"][0].text == "snapshot"
    assert json.loads((session_dir / "session.json").read_text(encoding="utf-8"))["meta"]["session_id"] == "s1"


@pytest.mark.asyncio
async def test_snapshot_recovery_uses_newest_turn(tmp_path: Path) -> None:
    """When live files are corrupt, recovery picks the highest-numbered valid snapshot."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["older"])], "compressed_msgs": []})
    session_dir = tmp_path / "s1"
    older = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))

    await store.save_session("s1", {"messages": [Message("user", ["newer"])], "compressed_msgs": []})
    newer = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))

    snap_dir = session_dir / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "turn_2.json").write_text(json.dumps(older), encoding="utf-8")
    (snap_dir / "turn_9.json").write_text(json.dumps(newer), encoding="utf-8")
    (session_dir / "session.json").write_text("{ broken primary", encoding="utf-8")
    (session_dir / "session.json.bak").write_text("{ broken backup", encoding="utf-8")

    loaded = await store.load_session("s1")

    assert loaded is not None
    assert loaded["messages"][0].text == "newer"


@pytest.mark.asyncio
async def test_snapshot_recovery_works_when_primary_and_backup_are_missing(tmp_path: Path) -> None:
    """Rollback snapshots remain usable even if both live session files are gone."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["snapshot only"])], "compressed_msgs": []})

    session_dir = tmp_path / "s1"
    snap_dir = session_dir / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "turn_2.json").write_text(
        (session_dir / "session.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (session_dir / "session.json").unlink()
    (session_dir / "session.json.bak").unlink()

    loaded = await store.load_session("s1")

    assert loaded is not None
    assert loaded["messages"][0].text == "snapshot only"
    assert json.loads((session_dir / "session.json").read_text(encoding="utf-8"))["meta"]["session_id"] == "s1"


@pytest.mark.asyncio
async def test_list_sessions_recovers_snapshot_only_session(tmp_path: Path) -> None:
    """Session listing should rediscover sessions whose only valid file is a rollback snapshot."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["snapshot only"])], "compressed_msgs": []})

    session_dir = tmp_path / "s1"
    snap_dir = session_dir / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "turn_2.json").write_text(
        (session_dir / "session.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (session_dir / "session.json").unlink()
    (session_dir / "session.json.bak").unlink()

    sessions = await store.list_sessions()

    assert [s.session_id for s in sessions] == ["s1"]
    assert (session_dir / "session.json").exists()


@pytest.mark.asyncio
async def test_save_does_not_seed_metadata_from_rollback_snapshot(tmp_path: Path) -> None:
    """Save metadata inheritance must use live files only, not stale turn snapshots."""
    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [Message("user", ["live"])], "compressed_msgs": []})

    session_dir = tmp_path / "s1"
    stale_snapshot = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    stale_snapshot["meta"]["agent_profile"] = "stale-profile"
    stale_snapshot["meta"]["agent_profile_history"] = ["stale"]
    stale_snapshot["meta"]["parent_session_id"] = "stale-parent"
    snap_dir = session_dir / "snapshots"
    snap_dir.mkdir()
    (snap_dir / "turn_9.json").write_text(json.dumps(stale_snapshot), encoding="utf-8")

    (session_dir / "session.json").write_text("{ broken primary", encoding="utf-8")
    (session_dir / "session.json.bak").write_text("{ broken backup", encoding="utf-8")

    await store.save_session("s1", {"messages": [Message("user", ["new"])], "compressed_msgs": []})

    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))["meta"]
    assert meta["agent_profile"] == ""
    assert meta["agent_profile_history"] == []
    assert meta["parent_session_id"] == ""


@pytest.mark.asyncio
async def test_save_times_out_when_write_lock_is_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second writer should fail cleanly instead of racing session.json."""
    store = JsonFileStateStore(tmp_path)
    monkeypatch.setattr(store_module, "SESSION_WRITE_LOCK_TIMEOUT_SECONDS", 0.01)

    lock_path = store_module.session_write_lock_path(tmp_path, "locked")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    held = FileLock(lock_path, timeout=1.0)
    held.acquire()
    try:
        with pytest.raises(TimeoutError):
            await store.save_session("locked", {"messages": [], "compressed_msgs": []})
    finally:
        held.release()


# --- meta provenance (schema_version + app_version) --------------------


@pytest.mark.asyncio
async def test_save_stamps_schema_and_app_version(tmp_path: Path) -> None:
    """New saves write ``schema_version`` and ``app_version`` into meta
    so future readers can gate migration on version rather than sniffing
    field presence."""
    import json as _json

    from chrys import __version__ as app_version
    from chrys.service.state.store import SESSION_SCHEMA_VERSION

    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [], "compressed_msgs": []}, agent_profile="code")

    raw = _json.loads((tmp_path / "s1" / "session.json").read_text(encoding="utf-8"))
    assert raw["meta"]["schema_version"] == SESSION_SCHEMA_VERSION
    assert raw["meta"]["app_version"] == app_version


@pytest.mark.asyncio
async def test_save_stamps_platform_os_and_arch(tmp_path: Path) -> None:
    """``os_name`` + ``arch`` come from ``common.platform`` so session
    files carry provenance of the runtime that wrote them."""
    import json as _json

    from chrys.foundation.platform import get_platform

    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [], "compressed_msgs": []})

    plat = get_platform()
    meta = _json.loads((tmp_path / "s1" / "session.json").read_text(encoding="utf-8"))["meta"]
    assert meta["os_name"] == plat.os_name
    assert meta["arch"] == plat.arch


@pytest.mark.asyncio
async def test_save_stamps_model_fields_only_when_supplied(tmp_path: Path) -> None:
    """Model provenance is limited to non-sensitive model/session fields;
    nothing else from the caller's model profile should leak into the
    file."""
    import json as _json

    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "s1",
        {"messages": [], "compressed_msgs": []},
        model_provider="anthropic",
        model_api_style="chat_completions",
        model_id="claude-sonnet-4-6",
        model_base_url="https://api.anthropic.com",
        model_profile_fingerprint="model-fp",
        agent_profile_fingerprint="agent-fp",
        service_session_id="resp_123",
    )

    meta = _json.loads((tmp_path / "s1" / "session.json").read_text(encoding="utf-8"))["meta"]
    assert meta["model_provider"] == "anthropic"
    assert meta["model_api_style"] == "chat_completions"
    assert meta["model_id"] == "claude-sonnet-4-6"
    assert meta["model_base_url"] == "https://api.anthropic.com"
    assert meta["model_profile_fingerprint"] == "model-fp"
    assert meta["agent_profile_fingerprint"] == "agent-fp"
    assert meta["service_session_id"] == "resp_123"
    # Explicitly assert the environment fields are NOT in meta — the
    # store's surface deliberately won't accept them, so disk output
    # stays provider/id/base_url only.
    for forbidden in ("api_key", "http_headers", "chat_options", "http_read_timeout"):
        assert forbidden not in meta


@pytest.mark.asyncio
async def test_model_profile_id_round_trips_and_explicit_empty_overwrites(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    state = {"messages": [], "compressed_msgs": []}

    await store.save_session("s1", state, model_profile_id="saved-profile")

    saved = await store.load_session_meta("s1")
    assert saved is not None
    assert saved.model_profile_id == "saved-profile"

    await store.save_session("s1", state, model_profile_id="")

    cleared = await store.load_session_meta("s1")
    assert cleared is not None
    assert cleared.model_profile_id == ""
    raw = json.loads((tmp_path / "s1" / "session.json").read_text(encoding="utf-8"))
    assert raw["meta"]["model_profile_id"] == ""


@pytest.mark.asyncio
async def test_agent_profile_id_round_trips_and_explicit_empty_overwrites(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    state = {"messages": [], "compressed_msgs": []}

    await store.save_session("s1", state, agent_profile_id="saved-agent")

    saved = await store.load_session_meta("s1")
    assert saved is not None
    assert saved.agent_profile_id == "saved-agent"

    await store.save_session("s1", state, agent_profile_id="")

    cleared = await store.load_session_meta("s1")
    assert cleared is not None
    assert cleared.agent_profile_id == ""
    raw = json.loads((tmp_path / "s1" / "session.json").read_text(encoding="utf-8"))
    assert raw["meta"]["agent_profile_id"] == ""


@pytest.mark.asyncio
async def test_recovery_session_model_profile_id_round_trips(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await asyncio.to_thread(
        store.save_recovery_session,
        "recovering",
        {"messages": [], "compressed_msgs": []},
        model_profile_id="recovery-profile",
    )

    meta = await store.load_recovery_session_meta("recovering")

    assert meta is not None
    assert meta.model_profile_id == "recovery-profile"


@pytest.mark.asyncio
async def test_recovery_session_agent_profile_id_round_trips(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await asyncio.to_thread(
        store.save_recovery_session,
        "recovering",
        {"messages": [], "compressed_msgs": []},
        agent_profile_id="recovery-agent",
    )

    meta = await store.load_recovery_session_meta("recovering")

    assert meta is not None
    assert meta.agent_profile_id == "recovery-agent"


@pytest.mark.asyncio
async def test_save_preserves_model_fields_when_next_save_omits_them(tmp_path: Path) -> None:
    """If a later save call omits model_* (e.g. a profile swap where
    the caller lost the reference), the prior values are preserved
    rather than blanked."""
    import json as _json

    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "s1",
        {"messages": [], "compressed_msgs": []},
        model_provider="openai",
        model_api_style="responses",
        model_id="gpt-4o",
        model_base_url="https://api.openai.com/v1",
        service_session_id="resp_123",
    )
    # Second save with no model fields — must NOT overwrite the stamp.
    await store.save_session("s1", {"messages": [], "compressed_msgs": []})

    meta = _json.loads((tmp_path / "s1" / "session.json").read_text(encoding="utf-8"))["meta"]
    assert meta["model_provider"] == "openai"
    assert meta["model_api_style"] == "responses"
    assert meta["model_id"] == "gpt-4o"
    assert meta["model_base_url"] == "https://api.openai.com/v1"
    assert meta["service_session_id"] == "resp_123"


@pytest.mark.asyncio
async def test_save_can_clear_service_session_id(tmp_path: Path) -> None:
    """A later non-Responses save must clear stale provider-side session ids."""
    import json as _json

    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "s1",
        {"messages": [], "compressed_msgs": []},
        model_provider="openai",
        model_api_style="responses",
        model_id="gpt-5",
        service_session_id="resp_123",
    )
    await store.save_session(
        "s1",
        {"messages": [], "compressed_msgs": []},
        model_provider="openai",
        model_api_style="chat_completions",
        model_id="gpt-5",
        service_session_id="",
    )

    meta = _json.loads((tmp_path / "s1" / "session.json").read_text(encoding="utf-8"))["meta"]
    assert meta["model_api_style"] == "chat_completions"
    assert meta["service_session_id"] == ""


@pytest.mark.asyncio
async def test_save_can_clear_model_api_style(tmp_path: Path) -> None:
    """A later non-OpenAI save must clear stale Responses API-style metadata."""
    import json as _json

    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "s1",
        {"messages": [], "compressed_msgs": []},
        model_provider="openai",
        model_api_style="responses",
        model_id="gpt-5",
        service_session_id="resp_123",
    )
    await store.save_session(
        "s1",
        {"messages": [], "compressed_msgs": []},
        model_provider="anthropic",
        model_api_style="",
        model_id="claude-sonnet-4-6",
        service_session_id="",
    )

    meta = _json.loads((tmp_path / "s1" / "session.json").read_text(encoding="utf-8"))["meta"]
    assert meta["model_provider"] == "anthropic"
    assert meta["model_api_style"] == ""
    assert meta["model_id"] == "claude-sonnet-4-6"
    assert meta["service_session_id"] == ""


@pytest.mark.asyncio
async def test_list_sessions_surfaces_version_fields(tmp_path: Path) -> None:
    """``SessionMeta`` round-trips ``schema_version`` and ``app_version``
    from disk."""
    from chrys import __version__ as app_version
    from chrys.service.state.store import SESSION_SCHEMA_VERSION

    store = JsonFileStateStore(tmp_path)
    await store.save_session("s1", {"messages": [], "compressed_msgs": []})

    sessions = await store.list_sessions()
    assert sessions[0].schema_version == SESSION_SCHEMA_VERSION
    assert sessions[0].app_version == app_version


@pytest.mark.asyncio
async def test_list_sessions_surfaces_platform_and_model(tmp_path: Path) -> None:
    """``SessionMeta`` also round-trips the platform + model fields."""
    from chrys.foundation.platform import get_platform

    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "s1",
        {"messages": [], "compressed_msgs": []},
        model_provider="anthropic",
        model_api_style="chat_completions",
        model_id="claude-sonnet-4-6",
        model_base_url="https://api.anthropic.com",
        service_session_id="resp_123",
    )
    sessions = await store.list_sessions()

    plat = get_platform()
    assert sessions[0].os_name == plat.os_name
    assert sessions[0].arch == plat.arch
    assert sessions[0].model_provider == "anthropic"
    assert sessions[0].model_api_style == "chat_completions"
    assert sessions[0].model_id == "claude-sonnet-4-6"
    assert sessions[0].model_base_url == "https://api.anthropic.com"
    assert sessions[0].service_session_id == "resp_123"


@pytest.mark.asyncio
async def test_list_sessions_legacy_files_default_to_zero(tmp_path: Path) -> None:
    """Sessions written before the version fields existed read back with
    ``schema_version=0`` and empty ``app_version`` — those defaults let
    readers treat them as pre-versioned without special-casing key
    presence."""
    import json as _json

    sess_dir = tmp_path / "legacy"
    sess_dir.mkdir()
    (sess_dir / "session.json").write_text(
        _json.dumps(
            {
                "meta": {
                    "session_id": "legacy",
                    "agent_profile": "code",
                    "display_name": "",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "message_count": 0,
                },
                "state": {"messages": [], "compressed_msgs": []},
            }
        ),
        encoding="utf-8",
    )

    store = JsonFileStateStore(tmp_path)
    sessions = await store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].schema_version == 0
    # Legacy file used the v1 key name ``display_name`` — the reader
    # must still surface it via the new ``agent_display_name`` attr.
    assert sessions[0].agent_display_name == ""
    assert sessions[0].app_version == ""
    # Platform + model fields also default to empty on legacy files.
    assert sessions[0].os_name == ""
    assert sessions[0].arch == ""
    assert sessions[0].model_provider == ""
    assert sessions[0].model_api_style == ""
    assert sessions[0].model_id == ""
    assert sessions[0].model_profile_id == ""
    assert sessions[0].model_base_url == ""
    assert sessions[0].service_session_id == ""


@pytest.mark.asyncio
async def test_list_sessions_reads_legacy_key_names(tmp_path: Path) -> None:
    """A pre-versioned file on disk uses the old key names
    (``display_name`` / ``profile_history``).  Readers must surface
    them via the current attributes so TUI code never has to branch
    on whether the file predates the rename."""
    import json as _json

    sess_dir = tmp_path / "legacy"
    sess_dir.mkdir()
    (sess_dir / "session.json").write_text(
        _json.dumps(
            {
                "meta": {
                    # No ``schema_version`` — this file was written by
                    # pre-versioned code and uses the old key names.
                    "session_id": "legacy",
                    "agent_profile": "code",
                    "display_name": "Code Agent",
                    "profile_history": ["Code Agent", "Explore Agent"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "message_count": 3,
                },
                "state": {"messages": [], "compressed_msgs": []},
            }
        ),
        encoding="utf-8",
    )

    store = JsonFileStateStore(tmp_path)
    sessions = await store.list_sessions()

    assert len(sessions) == 1
    assert sessions[0].schema_version == 0  # pre-versioned
    # Rename is invisible to readers — old keys surface on new attrs.
    assert sessions[0].agent_display_name == "Code Agent"
    assert sessions[0].agent_profile_history == ["Code Agent", "Explore Agent"]


@pytest.mark.asyncio
async def test_list_sessions_surfaces_agent_profile_fingerprint(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "s1",
        {"messages": [], "compressed_msgs": []},
        agent_profile_fingerprint="agent-fp",
    )

    sessions = await store.list_sessions()

    assert sessions[0].agent_profile_fingerprint == "agent-fp"


@pytest.mark.asyncio
async def test_list_sessions_surfaces_model_profile_fingerprint(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await store.save_session(
        "s1",
        {"messages": [], "compressed_msgs": []},
        model_profile_fingerprint="model-fp",
    )

    sessions = await store.list_sessions()

    assert sessions[0].model_profile_fingerprint == "model-fp"


@pytest.mark.asyncio
async def test_resave_migrates_legacy_keys_without_blanking(tmp_path: Path) -> None:
    """A pre-versioned file re-saved by the current writer should keep
    its display/history values — the writer falls back to the old keys
    on ``existing_meta`` lookups so the second save doesn't clobber
    them just because the caller didn't explicitly repeat them."""
    import json as _json

    from chrys.service.state.store import SESSION_SCHEMA_VERSION

    sess_dir = tmp_path / "legacy"
    sess_dir.mkdir()
    (sess_dir / "session.json").write_text(
        _json.dumps(
            {
                "meta": {
                    # No ``schema_version`` — pre-versioned shape.
                    "session_id": "legacy",
                    "agent_profile": "code",
                    "display_name": "Code Agent",
                    "profile_history": ["Code Agent", "Explore Agent"],
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "message_count": 0,
                },
                "state": {"messages": [], "compressed_msgs": []},
            }
        ),
        encoding="utf-8",
    )

    store = JsonFileStateStore(tmp_path)
    # Re-save omitting both renamed kwargs — writer must pick them up
    # off the existing legacy meta rather than writing empty strings.
    await store.save_session("legacy", {"messages": [], "compressed_msgs": []})

    meta = _json.loads((tmp_path / "legacy" / "session.json").read_text(encoding="utf-8"))["meta"]
    # File now carries the current schema version — new keys present
    # with preserved values, old keys gone.
    assert meta["schema_version"] == SESSION_SCHEMA_VERSION
    assert meta["agent_display_name"] == "Code Agent"
    assert meta["agent_profile_history"] == ["Code Agent", "Explore Agent"]
    assert "display_name" not in meta
    assert "profile_history" not in meta


# --- Session MRU index / load_latest_session_id ---


def _mru_ids(root: Path) -> list[str]:
    snapshot = SessionMruIndex(root).load()
    assert snapshot is not None
    return [entry.session_id for entry in snapshot.sessions]


def _mru_raw(root: Path) -> dict:
    return json.loads((root / SESSION_MRU_FILE_NAME).read_text(encoding="utf-8"))


async def _save(store: JsonFileStateStore, session_id: str, *texts: str) -> None:
    await store.save_session(
        session_id,
        {"messages": [Message("user", [text]) for text in texts], "compressed_msgs": []},
    )


def _count_scans(monkeypatch: pytest.MonkeyPatch, store: JsonFileStateStore) -> list[int]:
    scans = [0]
    original = store._list_sessions_sync

    def counting() -> list:
        scans[0] += 1
        return original()

    monkeypatch.setattr(store, "_list_sessions_sync", counting)
    return scans


@pytest.mark.asyncio
async def test_save_records_mru_and_unchanged_save_keeps_order(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await _save(store, "older", "a")
    await _save(store, "newer", "a")
    assert _mru_ids(tmp_path) == ["newer", "older"]
    assert _mru_raw(tmp_path)["complete"] is False

    # Same visible message count -> updated_at preserved -> no promotion.
    await _save(store, "older", "a")
    assert _mru_ids(tmp_path) == ["newer", "older"]

    await _save(store, "older", "a", "b")
    assert _mru_ids(tmp_path) == ["older", "newer"]
    assert await store.load_latest_session_id() == "older"


@pytest.mark.asyncio
async def test_latest_session_prefers_valid_index_without_scanning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonFileStateStore(tmp_path)
    await _save(store, "s1", "a")
    await _save(store, "s2", "a")
    scans = _count_scans(monkeypatch, store)

    # Incomplete index -> exactly one backfill scan, then complete.
    assert await store.load_latest_session_id() == "s2"
    assert scans[0] == 1
    assert _mru_raw(tmp_path)["complete"] is True

    assert await store.load_latest_session_id() == "s2"
    assert scans[0] == 1


@pytest.mark.asyncio
async def test_latest_session_rebuilds_missing_and_corrupt_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonFileStateStore(tmp_path)
    await _save(store, "s1", "a")
    await _save(store, "s2", "a")
    scans = _count_scans(monkeypatch, store)

    (tmp_path / SESSION_MRU_FILE_NAME).unlink()
    assert await store.load_latest_session_id() == "s2"
    assert scans[0] == 1
    assert _mru_ids(tmp_path) == ["s2", "s1"]

    (tmp_path / SESSION_MRU_FILE_NAME).write_text("{corrupt", encoding="utf-8")
    assert await store.load_latest_session_id() == "s2"
    assert scans[0] == 2
    assert _mru_raw(tmp_path)["complete"] is True

    assert await store.load_latest_session_id() == "s2"
    assert scans[0] == 2


@pytest.mark.asyncio
async def test_latest_session_returns_none_without_sessions(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    assert await store.load_latest_session_id() is None
    assert _mru_raw(tmp_path)["complete"] is True
    assert _mru_ids(tmp_path) == []


@pytest.mark.asyncio
async def test_recovery_only_session_becomes_latest(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await _save(store, "saved", "a")
    await asyncio.to_thread(
        store.save_recovery_session,
        "crashed",
        {"messages": [Message("user", ["in-flight"])], "compressed_msgs": []},
    )
    assert _mru_ids(tmp_path)[0] == "crashed"
    assert await store.load_latest_session_id() == "crashed"
    listed = sorted(await store.list_sessions(), key=lambda m: m.updated_at, reverse=True)
    assert listed[0].session_id == "crashed"


@pytest.mark.asyncio
async def test_latest_session_survives_corrupt_primary_via_backup(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await _save(store, "old", "a")
    await _save(store, "top", "a")
    assert await store.load_latest_session_id() == "top"
    (tmp_path / "top" / "session.json").write_text("{corrupt", encoding="utf-8")
    assert await store.load_latest_session_id() == "top"


@pytest.mark.asyncio
async def test_fork_enters_index_first_and_delete_removes(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    parent = str(uuid4())
    await _save(store, parent, "a")
    await _save(store, "other", "a")
    fork_id = await asyncio.to_thread(store.fork_session, parent)
    assert _mru_ids(tmp_path)[0] == fork_id
    assert await store.load_latest_session_id() == fork_id

    await store.delete_session(fork_id)
    assert fork_id not in _mru_ids(tmp_path)
    assert await store.load_latest_session_id() == "other"


@pytest.mark.asyncio
async def test_mru_records_before_the_envelope_commits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Save / recovery / fork index first and commit second.

    A crash between the two can only leave the index *ahead* of disk — which
    verification re-ranks in memory — never behind it with a stale
    ``complete`` index that would hide the committed session forever.
    """
    store = JsonFileStateStore(tmp_path)
    parent = str(uuid4())
    await _save(store, parent, "a")
    original_record = store._mru.record
    # Observations only — an assertion raised inside ``record`` would be
    # swallowed by ``_note_mru``'s best-effort boundary, so assert afterwards.
    observed: list[tuple[str, bool, str | None, str]] = []
    probing = False

    def probe(session_id: str, updated_at: datetime) -> None:
        nonlocal probing
        if probing:  # the racing lookup's own write-back of a verified stamp
            original_record(session_id, updated_at)
            return
        probing = True
        on_disk = store._meta_for_session_dir(store._session_dir(session_id))
        # Nothing committed yet (fork) or the pre-save envelope (save/recovery).
        uncommitted = on_disk is None or coerce_utc(on_disk.updated_at) < coerce_utc(updated_at)
        original_record(session_id, updated_at)
        # A lookup racing the commit sees the index ahead of disk: it must
        # neither pick the uncommitted session nor prune it as a ghost.
        observed.append((session_id, uncommitted, store._load_latest_session_id_sync(), _mru_ids(tmp_path)[0]))
        probing = False

    monkeypatch.setattr(store._mru, "record", probe)
    await _save(store, parent, "a", "b")
    await asyncio.to_thread(
        store.save_recovery_session,
        parent,
        {"messages": [Message("user", ["x"])], "compressed_msgs": []},
    )
    fork_id = await asyncio.to_thread(store.fork_session, parent)
    assert observed == [
        (parent, True, parent, parent),
        (parent, True, parent, parent),
        (fork_id, True, parent, fork_id),
    ]
    assert _mru_ids(tmp_path)[0] == fork_id
    assert await store.load_latest_session_id() == fork_id


@pytest.mark.asyncio
async def test_ghost_prune_rechecks_absence_under_the_write_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork that commits between lookup's first absence check and its lock probe is kept."""
    store = JsonFileStateStore(tmp_path)
    await _save(store, "parent", "a")
    fork_id = str(uuid4())
    # Mid-commit fork: recorded, write lock held, directory not renamed yet.
    store._mru.record(fork_id, datetime.now(UTC) + timedelta(seconds=1))
    writer = FileLock(store._write_lock_path(fork_id))
    writer.acquire()
    original_legacy_path = store._legacy_path
    committed = False

    def legacy_path(session_id: str) -> Path:
        nonlocal committed
        path = original_legacy_path(session_id)
        if session_id == fork_id and not committed:
            committed = True
            # Lookup has just seen the directory absent; the fork now lands
            # and releases before lookup can probe the write lock.
            writer.release()
            store._save_session_sync(fork_id, {"messages": [Message("user", ["a"])], "compressed_msgs": []})
        return path

    monkeypatch.setattr(store, "_legacy_path", legacy_path)
    removed: list[str] = []
    monkeypatch.setattr(store._mru, "remove", removed.append)
    # The verify pass skips the in-flight entry, but must not prune it; the
    # closing sweep then sees the fork's freshly committed folder.
    assert store._load_latest_session_id_sync() == fork_id
    assert committed
    assert removed == []
    assert fork_id in _mru_ids(tmp_path)
    monkeypatch.setattr(store, "_legacy_path", original_legacy_path)
    assert await store.load_latest_session_id() == fork_id


@pytest.mark.asyncio
async def test_stuck_index_lock_neither_stalls_saves_nor_hides_the_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Index writers give up after the (shrunk) timeout; the sweep still finds the commit."""
    monkeypatch.setattr(session_mru_module, "SESSION_MRU_LOCK_TIMEOUT_SECONDS", 0.05)
    store = JsonFileStateStore(tmp_path)
    await _save(store, "a", "a")
    assert await store.load_latest_session_id() == "a"  # complete index
    holder = FileLock(store._mru.lock_path)
    holder.acquire()
    done = threading.Event()

    def save_b() -> None:
        store._save_session_sync("b", {"messages": [Message("user", ["b"])], "compressed_msgs": []})
        done.set()

    worker = threading.Thread(target=save_b)
    worker.start()
    try:
        assert done.wait(2.0)  # bounded: does not wait for the holder
        assert store._session_file("b").exists()
        assert store._mru.load() is None  # read side times out and falls back too
    finally:
        holder.release()
    worker.join(timeout=5.0)
    assert _mru_ids(tmp_path) == ["a"]  # the record was skipped, index untouched
    scans = _count_scans(monkeypatch, store)
    assert await store.load_latest_session_id() == "b"  # via the modified-folder sweep
    assert scans[0] == 0
    assert _mru_ids(tmp_path)[0] == "b"  # ...which also repairs the index


@pytest.mark.asyncio
async def test_lookup_stays_bounded_when_the_index_lock_is_stuck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_mru_module, "SESSION_MRU_LOCK_TIMEOUT_SECONDS", 0.05)
    store = JsonFileStateStore(tmp_path)
    await _save(store, "a", "a")
    await _save(store, "b", "a")
    holder = FileLock(store._mru.lock_path)
    holder.acquire()
    try:
        result: list[str | None] = []
        worker = threading.Thread(target=lambda: result.append(store._load_latest_session_id_sync()))
        worker.start()
        worker.join(timeout=5.0)
        assert not worker.is_alive()  # read timeout + scan + rebuild timeout, then returns
        assert result == ["b"]
    finally:
        holder.release()


@pytest.mark.asyncio
async def test_sweep_finds_sessions_written_behind_the_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An older chrys or a copied folder never records; a lookup still sees it."""
    store = JsonFileStateStore(tmp_path)
    await _save(store, "old", "a")
    assert await store.load_latest_session_id() == "old"
    # A foreign writer drops a newer session folder straight onto disk.
    copied = str(uuid4())
    shutil.copytree(store.session_dir("old"), store.session_dir(copied))
    session_file = store._session_file(copied)
    data = json.loads(session_file.read_text(encoding="utf-8"))
    data["meta"]["session_id"] = copied
    data["meta"]["updated_at"] = datetime.now(UTC).isoformat()
    session_file.write_text(json.dumps(data), encoding="utf-8")
    scans = _count_scans(monkeypatch, store)
    assert await store.load_latest_session_id() == copied
    assert scans[0] == 0
    assert _mru_ids(tmp_path)[0] == copied  # repaired for next time
    # And a legacy flat file dropped in the same way (ranked via the listing).
    legacy_id = str(uuid4())
    stamp = datetime.now(UTC).isoformat()
    store._legacy_path(legacy_id).write_text(
        json.dumps({"meta": {"session_id": legacy_id, "created_at": stamp, "updated_at": stamp}, "state": {}}),
        encoding="utf-8",
    )
    assert await store.load_latest_session_id() == legacy_id


@pytest.mark.asyncio
async def test_sweep_applies_the_listing_precedence_to_duplicate_legacy_ids(tmp_path: Path) -> None:
    """A later-sorted legacy copy of an id the listing suppresses must not win the lookup."""
    store = JsonFileStateStore(tmp_path)
    await _save(store, "winner", "a")
    winner_meta = await store.load_session_meta("winner")
    assert winner_meta is not None
    older = (winner_meta.updated_at - timedelta(minutes=5)).isoformat()
    newer = (winner_meta.updated_at + timedelta(minutes=5)).isoformat()
    for name, stamp in (("dupe.json", older), ("zzz.json", newer)):  # same embedded id; the first sorted wins
        (tmp_path / name).write_text(
            json.dumps({"meta": {"session_id": "dupe", "created_at": stamp, "updated_at": stamp}, "state": {}}),
            encoding="utf-8",
        )
    metas = await store.list_sessions()
    assert max(metas, key=lambda m: m.updated_at).session_id == "winner"
    store._mru.invalidate()
    assert await store.load_latest_session_id() == "winner"  # backfill + sweep
    assert await store.load_latest_session_id() == "winner"  # index path + sweep
    # A legacy copy of a folder session's id is suppressed by the folder too.
    (tmp_path / "copy.json").write_text(
        json.dumps({"meta": {"session_id": "winner", "created_at": newer, "updated_at": newer}, "state": {}}),
        encoding="utf-8",
    )
    metas = await store.list_sessions()
    assert [m.session_id for m in metas if m.session_id == "winner"] == ["winner"]
    assert await store.load_latest_session_id() == "winner"
    snapshot = store._mru.load()
    assert snapshot is not None
    assert snapshot.sessions[0] == SessionMruEntry("winner", coerce_utc(winner_meta.updated_at))  # copy not recorded


@pytest.mark.asyncio
async def test_commit_after_failed_record_and_invalidate_is_still_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonFileStateStore(tmp_path)
    await _save(store, "b", "a")
    assert await store.load_latest_session_id() == "b"

    broken = True
    real_record, real_invalidate = store._mru.record, store._mru.invalidate

    def record(*args: object, **kwargs: object) -> None:
        if broken:
            raise OSError("root not writable")
        real_record(*args, **kwargs)

    def invalidate() -> None:
        if broken:
            raise OSError("root not writable")
        real_invalidate()

    monkeypatch.setattr(store._mru, "record", record)
    monkeypatch.setattr(store._mru, "invalidate", invalidate)
    await _save(store, "a", "a")  # commits behind the old complete index
    assert _mru_ids(tmp_path) == ["b"]
    broken = False
    metas = await store.list_sessions()
    assert max(metas, key=lambda m: m.updated_at).session_id == "a"
    assert await store.load_latest_session_id() == "a"


@pytest.mark.asyncio
async def test_recovery_only_session_hidden_during_backfill_surfaces_after_owner_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonFileStateStore(tmp_path)
    await _save(store, "saved", "a")
    owner = FileLock(store.active_lock_path("live"))
    owner.acquire()
    try:
        await asyncio.to_thread(
            store.save_recovery_session,
            "live",
            {"messages": [Message("user", ["in-flight"])], "compressed_msgs": []},
        )
        store._mru.invalidate()  # backfill from scratch while the owner is alive
        assert await store.load_latest_session_id() == "saved"  # hidden, like list_sessions()
        assert _mru_ids(tmp_path) == ["saved"]
    finally:
        owner.release()
    metas = await store.list_sessions()
    assert max(metas, key=lambda m: m.updated_at).session_id == "live"
    scans = _count_scans(monkeypatch, store)
    assert await store.load_latest_session_id() == "live"  # via the sweep, no rescan
    assert scans[0] == 0
    assert _mru_ids(tmp_path)[0] == "live"


@pytest.mark.asyncio
async def test_malformed_index_entry_triggers_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonFileStateStore(tmp_path)
    await _save(store, "old", "a")
    await _save(store, "new", "a")
    raw = _mru_raw(tmp_path)
    assert raw["sessions"][0]["session_id"] == "new"
    raw["sessions"][0]["last_updated_at"] = "garbage"
    (tmp_path / SESSION_MRU_FILE_NAME).write_text(json.dumps(raw), encoding="utf-8")
    scans = _count_scans(monkeypatch, store)
    assert await store.load_latest_session_id() == "new"
    assert scans[0] == 1
    assert _mru_ids(tmp_path) == ["new", "old"]


@pytest.mark.asyncio
async def test_legacy_migration_records_before_rename_so_backfill_cannot_miss_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legacy file migrated between the folder and flat-file enumerations stays indexed."""
    store = JsonFileStateStore(tmp_path)
    await _save(store, "older", "a")
    newer = str(uuid4())
    stamp = (datetime.now(UTC) + timedelta(minutes=1)).isoformat()
    envelope = {
        "meta": {"session_id": newer, "created_at": stamp, "updated_at": stamp, "message_count": 0},
        "state": {"messages": [], "compressed_msgs": []},
    }
    store._legacy_path(newer).write_text(json.dumps(envelope), encoding="utf-8")
    store._mru.invalidate()
    original_candidates = store._session_dir_candidates

    def candidates_then_migrate() -> list[Path]:
        found = original_candidates()
        store._migrate_if_needed(newer)  # another window opens the legacy session right now
        return found

    monkeypatch.setattr(store, "_session_dir_candidates", candidates_then_migrate)
    assert await store.load_latest_session_id() == newer
    monkeypatch.setattr(store, "_session_dir_candidates", original_candidates)
    snapshot = store._mru.load()
    assert snapshot is not None and snapshot.complete is True
    assert [e.session_id for e in snapshot.sessions] == [newer, "older"]
    assert await store.load_latest_session_id() == newer
    assert (await store.list_sessions())[0].session_id == newer


@pytest.mark.asyncio
async def test_rebuild_keeps_horizon_raised_by_a_record_trimmed_during_the_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A concurrent record that only survives in the horizon must not be erased by the rebuild."""
    store = JsonFileStateStore(tmp_path)
    store._mru = SessionMruIndex(tmp_path, max_entries=2)
    for name in ("a", "y", "x"):
        await _save(store, name, "a")
    y_meta = await store.load_session_meta("y")
    assert y_meta is not None
    # Index retains stale-high x/y; on disk both rolled back below a, so
    # the next lookup has to rescan.
    _shift_updated_at(store, "a", -timedelta(days=2))
    _shift_updated_at(store, "x", -timedelta(days=1))
    _shift_updated_at(store, "y", -timedelta(days=1))
    # Legacy session s: newer than everything on disk, older than the stale
    # retained stamps — its record gets trimmed at once and only lifts the horizon.
    s_id = str(uuid4())
    s_stamp = (y_meta.updated_at - timedelta(minutes=1)).isoformat()
    envelope = {
        "meta": {"session_id": s_id, "created_at": s_stamp, "updated_at": s_stamp, "message_count": 0},
        "state": {"messages": [], "compressed_msgs": []},
    }
    store._legacy_path(s_id).write_text(json.dumps(envelope), encoding="utf-8")
    original_candidates = store._session_dir_candidates

    def candidates_then_migrate() -> list[Path]:
        found = original_candidates()
        store._migrate_if_needed(s_id)
        return found

    monkeypatch.setattr(store, "_session_dir_candidates", candidates_then_migrate)
    # The rescan races the migration: the scan misses s, but the rebuild's
    # merged ranking (not just the kept top-2) still surfaces its record.
    assert await store.load_latest_session_id() == s_id
    monkeypatch.setattr(store, "_session_dir_candidates", original_candidates)
    snapshot = store._mru.load()
    assert snapshot is not None and s_id not in [e.session_id for e in snapshot.sessions]
    assert snapshot.horizon is not None and snapshot.horizon >= coerce_utc(datetime.fromisoformat(s_stamp))
    metas = await store.list_sessions()
    assert max(metas, key=lambda m: m.updated_at).session_id == s_id
    assert await store.load_latest_session_id() == s_id


@pytest.mark.asyncio
async def test_delete_removes_index_entry_before_a_same_id_recreate_can_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A save recreating a just-deleted id must not lose its record to the delayed removal."""
    store = JsonFileStateStore(tmp_path)
    await _save(store, "other", "a")
    await _save(store, "reborn", "a")
    original_remove = store._mru.remove
    removing = threading.Event()
    release = threading.Event()
    saver_at_lock = threading.Event()
    reborn_lock_path = store._write_lock_path("reborn")

    class SignallingLock(FileLock):
        def acquire(self) -> None:
            if self._path == reborn_lock_path and threading.current_thread().name == "saver":
                saver_at_lock.set()
            super().acquire()

    monkeypatch.setattr(store_module, "FileLock", SignallingLock)

    def slow_remove(session_id: str) -> None:
        removing.set()
        assert release.wait(5.0)
        original_remove(session_id)

    store._mru.remove = slow_remove  # type: ignore[method-assign]
    deleter = threading.Thread(target=lambda: store._delete_session_sync("reborn"))
    deleter.start()
    assert removing.wait(5.0)
    saved = threading.Event()

    def recreate() -> None:
        store._save_session_sync("reborn", {"messages": [Message("user", ["again"])], "compressed_msgs": []})
        saved.set()

    saver = threading.Thread(target=recreate, name="saver")
    saver.start()
    assert saver_at_lock.wait(5.0)  # the recreate is at the write lock...
    assert not saved.wait(0.3)  # ...and blocked there until the removal is done
    release.set()
    deleter.join(timeout=5.0)
    saver.join(timeout=5.0)
    assert saved.is_set()
    assert _mru_ids(tmp_path)[0] == "reborn"
    assert await store.load_latest_session_id() == "reborn"
    metas = await store.list_sessions()
    assert max(metas, key=lambda m: m.updated_at).session_id == "reborn"


@pytest.mark.asyncio
async def test_latest_session_returns_post_merge_latest_after_backfill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A save that lands during the backfill scan wins this lookup, not just the next."""
    store = JsonFileStateStore(tmp_path)
    await _save(store, "s1", "a")
    await _save(store, "s2", "a")
    store._mru.invalidate()
    original = store._list_sessions_sync

    def scan_then_concurrent_save() -> list:
        listed = original()
        store._save_session_sync("s3", {"messages": [Message("user", ["late"])], "compressed_msgs": []})
        return listed

    monkeypatch.setattr(store, "_list_sessions_sync", scan_then_concurrent_save)
    assert await store.load_latest_session_id() == "s3"
    snapshot = SessionMruIndex(tmp_path).load()
    assert snapshot is not None and snapshot.complete is True
    assert [e.session_id for e in snapshot.sessions] == ["s3", "s2", "s1"]


@pytest.mark.asyncio
async def test_backfill_ranks_every_merged_entry_not_just_the_leader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An in-flight record above a committed concurrent save must not hide that save."""
    store = JsonFileStateStore(tmp_path)
    await _save(store, "old", "a")
    store._mru.invalidate()
    original = store._list_sessions_sync
    inflight_lock = FileLock(store._write_lock_path("inflight"))

    def scan_then_concurrent_writers() -> list:
        listed = original()
        # Writer A: recorded (under its write lock) but not yet committed.
        inflight_lock.acquire()
        store._mru.record("inflight", datetime.now(UTC) + timedelta(seconds=30))
        # Writer B: recorded and committed, newer than "old".
        store._save_session_sync("committed", {"messages": [Message("user", ["b"])], "compressed_msgs": []})
        return listed

    monkeypatch.setattr(store, "_list_sessions_sync", scan_then_concurrent_writers)
    try:
        assert await store.load_latest_session_id() == "committed"
    finally:
        inflight_lock.release()
    metas = await store.list_sessions()
    assert max(metas, key=lambda m: m.updated_at).session_id == "committed"


@pytest.mark.asyncio
async def test_backfill_accepts_a_merged_leader_that_advanced_before_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonFileStateStore(tmp_path)
    await _save(store, "old", "a")
    store._mru.invalidate()
    original_scan = store._list_sessions_sync
    original_verify = store._mru_verify

    def scan_then_concurrent_save() -> list:
        listed = original_scan()
        store._save_session_sync("late", {"messages": [Message("user", ["1"])], "compressed_msgs": []})
        return listed

    def verify_after_another_save(session_id: str) -> datetime | None:
        if session_id == "late":
            monkeypatch.setattr(store, "_mru_verify", original_verify)
            store._save_session_sync("late", {"messages": [Message("user", ["1", "2"])], "compressed_msgs": []})
        return original_verify(session_id)

    monkeypatch.setattr(store, "_list_sessions_sync", scan_then_concurrent_save)
    monkeypatch.setattr(store, "_mru_verify", verify_after_another_save)
    assert await store.load_latest_session_id() == "late"


@pytest.mark.asyncio
async def test_backfill_of_an_empty_root_still_returns_a_concurrently_committed_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonFileStateStore(tmp_path)
    original = store._list_sessions_sync

    def empty_scan_then_concurrent_save() -> list:
        listed = original()
        assert listed == []
        store._save_session_sync("late", {"messages": [Message("user", ["x"])], "compressed_msgs": []})
        return listed

    monkeypatch.setattr(store, "_list_sessions_sync", empty_scan_then_concurrent_save)
    assert await store.load_latest_session_id() == "late"


@pytest.mark.asyncio
async def test_lookup_tolerates_offset_stamps_at_the_datetime_bounds(tmp_path: Path) -> None:
    """Stamps whose UTC conversion under/overflows must not raise out of ``/resume``."""
    store = JsonFileStateStore(tmp_path)
    await _save(store, "only", "a")
    for stamp in ("0001-01-01T00:00:00+23:59", "9999-12-31T23:59:59-23:59"):
        for path in (store._session_file("only"), store._backup_file("only")):
            data = json.loads(path.read_text(encoding="utf-8"))
            data["meta"]["updated_at"] = stamp
            path.write_text(json.dumps(data), encoding="utf-8")
        assert [m.session_id for m in await store.list_sessions()] == ["only"]
        store._mru.invalidate()
        assert await store.load_latest_session_id() == "only"  # backfill
        assert await store.load_latest_session_id() == "only"  # index path


@pytest.mark.asyncio
async def test_backfill_does_not_resurrect_a_session_deleted_after_the_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonFileStateStore(tmp_path)
    await _save(store, "older", "a")
    await _save(store, "newer", "a")
    store._mru.invalidate()
    original = store._list_sessions_sync

    def scan_then_concurrent_delete() -> list:
        listed = original()
        assert {m.session_id for m in listed} == {"older", "newer"}
        store._delete_session_sync("newer")
        return listed

    monkeypatch.setattr(store, "_list_sessions_sync", scan_then_concurrent_delete)
    assert await store.load_latest_session_id() == "older"
    assert "newer" not in _mru_ids(tmp_path)  # the stale scan input was rebuilt in, then pruned as a ghost


@pytest.mark.asyncio
async def test_backfill_of_an_empty_root_sees_a_first_save_that_commits_after_failing_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonFileStateStore(tmp_path)
    writer = FileLock(store._write_lock_path("first"))
    writer.acquire()  # in-flight first save: recorded, not yet committed
    store._mru.record("first", datetime.now(UTC))
    original_verify = store._mru_verify
    committed = False

    def verify_then_commit(session_id: str) -> datetime | None:
        nonlocal committed
        actual = original_verify(session_id)  # absent, and not prunable: the writer holds the lock
        if session_id == "first" and not committed:
            committed = True
            writer.release()
            store._save_session_sync("first", {"messages": [Message("user", ["x"])], "compressed_msgs": []})
        return actual

    monkeypatch.setattr(store, "_mru_verify", verify_then_commit)
    try:
        assert await store.load_latest_session_id() == "first"
    finally:
        if not committed:
            writer.release()
    assert committed


@pytest.mark.asyncio
async def test_lookup_verifies_a_legacy_session_under_a_non_canonical_filename(tmp_path: Path) -> None:
    """The listing attributes ``*.json`` by embedded id; verification must not prune such a session."""
    store = JsonFileStateStore(tmp_path)
    stamp = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    path = tmp_path / "arbitrary-name.json"
    path.write_text(
        json.dumps({"meta": {"session_id": "embedded-id", "created_at": stamp, "updated_at": stamp}, "state": {}}),
        encoding="utf-8",
    )
    aged = (datetime.now(UTC) - timedelta(hours=1)).timestamp()
    os.utime(path, (aged, aged))  # outside the sweep's mtime window
    assert [m.session_id for m in await store.list_sessions()] == ["embedded-id"]
    assert await store.load_latest_session_id() == "embedded-id"  # backfill
    assert await store.load_latest_session_id() == "embedded-id"  # index path: verify, don't prune
    assert _mru_ids(tmp_path) == ["embedded-id"]
    # A corrupt canonical copy must not mask it either (the listing skips unreadable files).
    store._legacy_path("embedded-id").write_text("{not json", encoding="utf-8")
    assert [m.session_id for m in await store.list_sessions()] == ["embedded-id"]
    assert await store.load_latest_session_id() == "embedded-id"
    assert await store.load_latest_session_id() == "embedded-id"


@pytest.mark.asyncio
async def test_ghost_prune_lock_errors_do_not_fail_the_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonFileStateStore(tmp_path)
    await _save(store, "older", "a")
    await _save(store, "newer", "a")
    assert await store.load_latest_session_id() == "newer"
    shutil.rmtree(store.session_dir("newer"))  # ghost entry stays in the index
    ghost_lock = store._write_lock_path("newer")

    class DeniedLock(FileLock):
        def acquire(self) -> None:
            if self._path == ghost_lock:
                raise PermissionError("simulated denied per-session lock")
            super().acquire()

    monkeypatch.setattr(store_module, "FileLock", DeniedLock)
    assert await store.load_latest_session_id() == "older"
    assert "newer" in _mru_ids(tmp_path)  # left for a later prune


@pytest.mark.asyncio
async def test_pre_commit_crash_folder_is_pruned_but_a_hidden_sidecar_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded session whose folder holds nothing restorable is a ghost; one hidden by an active lock is not."""
    store = JsonFileStateStore(tmp_path)
    await _save(store, "a", "a")
    assert await store.load_latest_session_id() == "a"
    # Save crashed after mkdir + record, before the envelope commit.
    store._mru.record("ghost", datetime.now(UTC) + timedelta(seconds=1))
    store.session_dir("ghost").mkdir()
    # Recovery-only session whose owner is alive: hidden, but restorable later.
    owner = FileLock(store.active_lock_path("live"))
    owner.acquire()
    try:
        await asyncio.to_thread(
            store.save_recovery_session, "live", {"messages": [Message("user", ["x"])], "compressed_msgs": []}
        )
        scans = _count_scans(monkeypatch, store)
        assert await store.load_latest_session_id() == "a"
        assert scans[0] == 0
        assert "ghost" not in _mru_ids(tmp_path)  # pruned for good
        assert "live" in _mru_ids(tmp_path)  # merely skipped this time
        assert await store.load_latest_session_id() == "a"
        assert scans[0] == 0
    finally:
        owner.release()
    assert await store.load_latest_session_id() == "live"


def _shift_updated_at(store: JsonFileStateStore, session_id: str, delta: timedelta) -> None:
    path = store._session_file(session_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    stamp = datetime.fromisoformat(data["meta"]["updated_at"]) + delta
    data["meta"]["updated_at"] = stamp.isoformat()
    path.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.asyncio
async def test_latest_session_rescans_when_downgraded_winner_falls_below_horizon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-ranking only the indexed entries is unsafe once they drop below the trim horizon."""
    store = JsonFileStateStore(tmp_path)
    store._mru = SessionMruIndex(tmp_path, max_entries=2)
    for name in ("s0", "s1", "s2"):
        await _save(store, name, "a")
    assert await store.load_latest_session_id() == "s2"
    snapshot = store._mru.load()
    assert snapshot is not None
    assert [e.session_id for e in snapshot.sessions] == ["s2", "s1"]
    s0_meta = await store.load_session_meta("s0")
    assert s0_meta is not None and snapshot.horizon == coerce_utc(s0_meta.updated_at)
    list_sync = store._list_sessions_sync
    scans = _count_scans(monkeypatch, store)

    # s2 rolled back below s0, but s1 still tops the horizon: in-memory
    # re-rank, no scan.
    _shift_updated_at(store, "s2", -timedelta(days=1))
    assert await store.load_latest_session_id() == "s1"
    assert scans[0] == 0

    # Every indexed session now sits below the unindexed s0: only a full
    # scan can find it, and it must agree with list_sessions().
    _shift_updated_at(store, "s1", -timedelta(days=2))
    metas = await asyncio.to_thread(list_sync)
    assert max(metas, key=lambda m: m.updated_at).session_id == "s0"
    assert await store.load_latest_session_id() == "s0"
    assert scans[0] == 1
    # The index never writes a downgrade, so the stale entries survive the
    # rebuild and each lookup keeps rescanning — correct, just slow — until
    # a save re-records one of them above the horizon.
    assert _mru_ids(tmp_path) == ["s2", "s1"]
    assert await store.load_latest_session_id() == "s0"
    assert scans[0] == 2
    await _save(store, "s1", "a", "b")
    assert await store.load_latest_session_id() == "s1"
    assert scans[0] == 2


@pytest.mark.asyncio
async def test_latest_session_skips_ghost_top_entry_and_repairs_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonFileStateStore(tmp_path)
    await _save(store, "s1", "a")
    await _save(store, "s2", "a")
    assert await store.load_latest_session_id() == "s2"
    scans = _count_scans(monkeypatch, store)

    shutil.rmtree(tmp_path / "s2")  # deleted out of band
    assert await store.load_latest_session_id() == "s1"
    assert scans[0] == 0
    assert _mru_ids(tmp_path) == ["s1"]


@pytest.mark.asyncio
async def test_latest_session_reranks_when_index_timestamp_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonFileStateStore(tmp_path)
    await _save(store, "s1", "a")
    await _save(store, "s2", "a")
    assert await store.load_latest_session_id() == "s2"
    scans = _count_scans(monkeypatch, store)

    # Index claims s1 is far in the future, but disk says otherwise: the
    # verified (older) value re-ranks in memory only.
    SessionMruIndex(tmp_path).record("s1", datetime(2999, 1, 1, tzinfo=UTC))
    assert _mru_ids(tmp_path) == ["s1", "s2"]
    assert await store.load_latest_session_id() == "s2"
    assert scans[0] == 0
    assert _mru_ids(tmp_path) == ["s1", "s2"]

    # Index lagging behind disk (older stamp than the envelope) is written
    # back once verified.
    await _save(store, "s1", "a", "b")  # promotes s1 on disk (and in the index)
    s1_meta = await store.load_session_meta("s1")
    s2_meta = await store.load_session_meta("s2")
    assert s1_meta is not None and s2_meta is not None
    index = SessionMruIndex(tmp_path)
    index.remove("s1")
    index.record("s1", s2_meta.updated_at + timedelta(microseconds=1))
    assert _mru_ids(tmp_path) == ["s1", "s2"]
    assert await store.load_latest_session_id() == "s1"
    assert scans[0] == 0
    repaired = index.load()
    assert repaired is not None
    assert repaired.sessions[0] == SessionMruEntry("s1", coerce_utc(s1_meta.updated_at))


@pytest.mark.asyncio
async def test_latest_session_rescans_when_complete_index_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = JsonFileStateStore(tmp_path)
    for i in range(3):
        await _save(store, f"s{i}", "a")
    assert await store.load_latest_session_id() == "s2"
    index = SessionMruIndex(tmp_path)
    for i in range(3):
        index.remove(f"s{i}")
    assert _mru_ids(tmp_path) == []
    scans = _count_scans(monkeypatch, store)

    assert await store.load_latest_session_id() == "s2"
    assert scans[0] == 1
    assert _mru_ids(tmp_path) == ["s2", "s1", "s0"]


@pytest.mark.asyncio
async def test_index_file_is_not_listed_as_legacy_session(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path)
    await _save(store, "s1", "a")
    assert (tmp_path / SESSION_MRU_FILE_NAME).exists()
    metas = await store.list_sessions()
    assert [m.session_id for m in metas] == ["s1"]
    streamed = [m.session_id async for batch in store.stream_session_metas() for m in batch]
    assert streamed == ["s1"]


@pytest.mark.asyncio
async def test_mru_failure_never_fails_session_operations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = JsonFileStateStore(tmp_path)
    parent = str(uuid4())
    await _save(store, parent, "a")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("index disk full")

    monkeypatch.setattr(store._mru, "record", boom)
    monkeypatch.setattr(store._mru, "remove", boom)

    await _save(store, parent, "a", "b")
    await asyncio.to_thread(
        store.save_recovery_session,
        parent,
        {"messages": [Message("user", ["x"])], "compressed_msgs": []},
    )
    fork_id = await asyncio.to_thread(store.fork_session, parent)
    await store.delete_session(fork_id)
    assert not (tmp_path / SESSION_MRU_FILE_NAME).exists()  # invalidated
    assert await store.load_session(parent) is not None
    assert not store.session_dir(fork_id).exists()


@pytest.mark.asyncio
async def test_latest_session_matches_list_sessions_under_active_lock(tmp_path: Path) -> None:
    """A live owner keeps the primary authoritative; the index must agree."""
    store = JsonFileStateStore(tmp_path)
    await _save(store, "held", "a")
    await _save(store, "free", "a")
    await asyncio.to_thread(
        store.save_recovery_session,
        "held",
        {"messages": [Message("user", ["in-flight"])], "compressed_msgs": []},
    )
    assert _mru_ids(tmp_path)[0] == "held"

    assert await store.load_latest_session_id() == "held"  # backfilled, complete

    lock = FileLock(store.active_lock_path("held"), timeout=1.0)
    lock.acquire()
    try:
        listed = sorted(await store.list_sessions(), key=lambda m: m.updated_at, reverse=True)
        assert listed[0].session_id == "free"
        assert await store.load_latest_session_id() == "free"
    finally:
        lock.release()
    # The owner "crashed": the sidecar wins again and the index still knows
    # it (the lock-time downgrade was never persisted).
    assert _mru_ids(tmp_path)[0] == "held"
    assert await store.load_latest_session_id() == "held"


@pytest.mark.asyncio
async def test_sweep_survives_extreme_winner_stamps(tmp_path: Path) -> None:
    """A datetime.min/max ``updated_at`` must not turn the sweep's cutoff into an OverflowError."""
    store = JsonFileStateStore(tmp_path)
    await _save(store, "only", "a")
    for stamp in ("0001-01-01T00:00:00+00:00", "9999-12-31T23:59:59.999999+00:00"):
        for path in (store._session_file("only"), store._backup_file("only")):
            data = json.loads(path.read_text(encoding="utf-8"))
            data["meta"]["updated_at"] = stamp
            path.write_text(json.dumps(data), encoding="utf-8")
        store._mru.invalidate()
        assert await store.load_latest_session_id() == "only"  # backfill
        assert await store.load_latest_session_id() == "only"  # index path + sweep


@pytest.mark.asyncio
async def test_a_checkpoint_digests_the_bytes_that_reached_the_file(tmp_path: Path) -> None:
    """A surrogateescaped path is escaped on the way to disk; the digest must
    cover what landed, not a second encoding of the same string."""
    store = JsonFileStateStore(tmp_path)
    checkpoint = await store.save_session(
        "s1",
        {"messages": [Message("user", ["hi"])], "compressed_msgs": []},
        primary_cwd="/work/pro\udcffject",
    )
    on_disk = (tmp_path / "s1" / "session.json").read_bytes()
    assert checkpoint.content_hash == hashlib.sha256(on_disk).hexdigest()
