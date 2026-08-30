# Copyright (c) 2026 Chrys. All rights reserved.

"""StateStore — protocol and implementations for session persistence."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, runtime_checkable
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

from chrys.foundation.config.settings import resolve_sessions_dir
from chrys.foundation.models.history_markers import HistoryMarkerKind
from chrys.foundation.models.turns import is_continuation_message
from chrys.foundation.platform.files import _atomic_write_bytes as _common_atomic_write_bytes
from chrys.foundation.platform.files import _fsync_dir as _common_fsync_dir
from chrys.foundation.platform.files import (
    atomic_write_owner_only_bytes,
    atomic_write_owner_only_text,
)
from chrys.foundation.platform.files import atomic_write_text as _common_atomic_write_text
from chrys.foundation.text.mentions import format_file_mention, iter_mention_tokens
from chrys.foundation.trajectory.ids import new_analytics_id
from chrys.foundation.util.lock import FileLock
from chrys.foundation.util.session_ids import session_short_id as _session_short_id
from chrys.foundation.util.time import parse_created_at
from chrys.kernel import Message
from chrys.service.context.providers.history import TURN_INDEX_KEY, CompressedBlock
from chrys.service.session.message_metadata import MESSAGE_CREATED_AT_KEY
from chrys.service.session.runtime_metadata import TOTAL_SESSION_TOKENS_KEY
from chrys.service.session.sub_agent_logs import (
    MAX_AUDIT_SCAN_TOTAL_BYTES,
    MAX_SCANNED_AUDIT_FILES,
    MAX_SUB_AGENT_AUDIT_BYTES,
    collect_sub_agent_log_file_references,
    is_safe_sub_agent_log_basename,
    read_owner_verified_bounded,
    scan_bounded_json_files,
    validate_acp_state_file_references,
)
from chrys.service.state.serializers import deserialize_state, serialize_state
from chrys.service.state.session_mru import (
    SESSION_MRU_FILE_NAME,
    SessionMruEntry,
    SessionMruIndex,
    coerce_utc,
    sort_entries,
)
from chrys.service.tools.session_artifacts import reharden_document_image_artifacts
from chrys.service.trajectory.state import TRAJECTORY_STATE_KEY


def _is_string_keyed_dict(value: object) -> TypeGuard[dict[str, Any]]:
    """Narrow serialized state objects, whose JSON keys are strings."""
    return isinstance(value, dict)


logger = logging.getLogger(__name__)

# ``meta.schema_version`` is the on-disk layout version of
# ``session.json``.  Bump whenever the meta shape or serialized state
# shape changes in a way older readers can't cope with.  Pre-versioned
# files (written before this field existed) read back with
# ``schema_version == 0`` via ``meta.get("schema_version", 0)`` and
# used the keys ``display_name`` / ``profile_history``; readers still
# accept those names and surface them via the v1 attributes.
SESSION_SCHEMA_VERSION = 1

SESSION_WRITE_LOCK_TIMEOUT_SECONDS = 10.0
SESSION_ACTIVE_LOCK_TIMEOUT_SECONDS = 5.0
# Folder-mtime slack for the post-lookup MRU sweep: covers coarse filesystem
# timestamps and the gap between an envelope's ``updated_at`` and its write.
_MRU_SWEEP_SLACK = timedelta(seconds=2)
SESSION_FILE_NAME = "session.json"
SESSION_BACKUP_FILE_NAME = "session.json.bak"
SESSION_RECOVERY_FILE_NAME = "session.recovery.json"
SESSION_FORK_MAX_ID_ATTEMPTS = 10
RAW_HTTP_LOG_FILE_NAME = "llm_raw_http.jsonl"
SESSION_CHECKPOINT_ID_KEY = "session_checkpoint_id"


@dataclass(frozen=True)
class SessionCheckpoint:
    """Identity of one persisted ``session.json`` revision.

    ``session_checkpoint_id`` is minted per save and written at the top level
    of the envelope; ``content_hash`` digests the exact bytes written, so a
    derived trajectory summary can name the revision it was computed from.
    """

    session_checkpoint_id: str
    content_hash: str


class SessionNotFoundError(FileNotFoundError):
    """Raised when a requested session does not exist."""


class SessionForkError(RuntimeError):
    """Raised when a session cannot be forked safely."""


def session_write_lock_path(sessions_dir: Path, session_id: str) -> Path:
    """Return the root-level write lock path for *session_id*."""
    return sessions_dir / ".locks" / f"{_session_short_id(session_id)}.write.lock"


def session_active_lock_path(sessions_dir: Path, session_id: str) -> Path:
    """Return the root-level active-session lock path for *session_id*."""
    return sessions_dir / ".locks" / f"{_session_short_id(session_id)}.active.lock"


def session_active_owner_path(sessions_dir: Path, session_id: str) -> Path:
    """Return the owner metadata path paired with the active-session lock."""
    return sessions_dir / ".locks" / f"{_session_short_id(session_id)}.active.json"


def parse_snapshot_turn(path: Path) -> int:
    """Parse rollback snapshot names into ``N``; return -1 on malformed names.

    Current snapshots are named ``turn_{N}.json``.  Older development
    builds used bare numeric names (``{N}.json``), so readers accept both
    forms while writers continue producing the prefixed shape.
    """
    try:
        stem = path.stem
        raw = stem.split("_", 1)[1] if stem.startswith("turn_") else stem
        return int(raw)
    except IndexError, ValueError:
        return -1


def _ensure_lock_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_copy_file(source: Path, dest: Path) -> None:
    """Copy *source* to *dest* without ever exposing a partial destination."""
    _common_atomic_write_bytes(dest, source.read_bytes())


def _atomic_write_text(path: Path, payload: str, *, encoding: str = "utf-8") -> bytes:
    return _common_atomic_write_text(path, payload, encoding=encoding)


def session_checkpoint_of(envelope: dict[str, Any], written: bytes) -> SessionCheckpoint:
    """The checkpoint identity of *envelope* as the bytes *written* to disk.

    The digest covers what the atomic writer produced rather than a second
    encoding of the same string: the two disagree on a lone surrogate, and a
    hash of bytes that never landed would never match the file it names.
    """
    checkpoint_id = envelope.get(SESSION_CHECKPOINT_ID_KEY)
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        checkpoint_id = ""
    digest = hashlib.sha256(written).hexdigest()
    return SessionCheckpoint(session_checkpoint_id=checkpoint_id, content_hash=digest)


def session_dir_has_artifacts(session_dir: Path) -> bool:
    """Whether *session_dir* holds anything a restore could start from."""
    return (
        (session_dir / SESSION_FILE_NAME).exists()
        or (session_dir / SESSION_BACKUP_FILE_NAME).exists()
        or (session_dir / SESSION_RECOVERY_FILE_NAME).exists()
        or any(parse_snapshot_turn(snap) >= 1 for snap in (session_dir / "snapshots").glob("*.json"))
    )


def session_dir_candidates(sessions_dir: Path) -> list[Path]:
    """Session folders under *sessions_dir* that plausibly hold a restorable session.

    Dot-prefixed entries (``.locks``, temp/partial dirs) are never sessions.
    """
    # New format: {session_id}/session.json
    return sorted(
        p
        for p in sessions_dir.iterdir()
        if p.is_dir() and p.name != ".locks" and not p.name.startswith(".") and session_dir_has_artifacts(p)
    )


def legacy_session_files(sessions_dir: Path) -> list[Path]:
    """Root-level ``*.json`` legacy flat-file sessions (never the MRU index)."""
    return sorted(p for p in sessions_dir.glob("*.json") if p.name != SESSION_MRU_FILE_NAME)


def make_junction_dropping_ignore() -> Callable[[str, list[str]], set[str]]:
    """Return a ``shutil.copytree`` ignore callback that drops NT junctions at EVERY level.

    ``copytree(symlinks=True)`` preserves POSIX symlinks verbatim but FOLLOWS
    NT directory junctions (``os.path.islink()`` is False for them), so a
    junction planted anywhere the copy reaches would be materialized as a
    real directory or, if it points at an ancestor, recurse to disk
    exhaustion. chrys never creates junctions, so excluding every one is
    loss-free; ``os.path.isjunction`` is always False on POSIX and for real
    dirs.
    """

    def _ignore(path: str, names: list[str]) -> set[str]:
        return {name for name in names if os.path.isjunction(os.path.join(path, name))}

    return _ignore


@dataclass
class SessionMeta:
    """Metadata about a saved session."""

    session_id: str
    agent_profile: str
    agent_display_name: str
    created_at: datetime
    updated_at: datetime
    message_count: int
    #: Total conversation turns (``state["turn_counter"]``); falls back to
    #: counting persisted turn markers for sessions saved before the
    #: counter existed.
    turn_count: int = 0
    #: Cumulative token usage (``state["total_session_tokens"]``); 0 for
    #: sessions saved before usage totals were persisted.
    total_tokens: int = 0
    primary_cwd: str = ""
    working_dirs: list[str] = field(default_factory=list)
    title: str = ""
    # Title overlays written only via ``update_session_titles``:
    # ``custom_title`` is user-set and permanently wins over any automatic
    # title; ``generated_title`` is the latest LLM-generated summary.
    custom_title: str = ""
    generated_title: str = ""
    agent_profile_id: str = ""
    agent_profile_history: list[str] = field(default_factory=list)
    agent_profile_fingerprint: str = ""
    size_bytes: int = 0
    app_version: str = ""
    schema_version: int = 0
    os_name: str = ""
    arch: str = ""
    model_provider: str = ""
    model_api_style: str = ""
    model_id: str = ""
    model_profile_id: str = ""
    model_base_url: str = ""
    model_profile_fingerprint: str = ""
    service_session_id: str = ""
    parent_session_id: str = ""
    #: Search EXCERPTS of the user-typed prompts from every turn — NOT the
    #: complete text: capped head+tail per message and a bounded session
    #: total, because this string lives in the store's in-memory meta
    #: cache, which must stay small (never envelope-sized).  Extracted from
    #: the envelope the meta scan already parses (zero extra IO); consumed
    #: by the sessions-browser second-tier search.
    user_prompt_search_text: str = ""

    @property
    def display_title(self) -> str:
        """User-facing session title: custom wins, then generated, then first-message."""
        return self.custom_title or self.generated_title or self.title


def _is_visible_message(m: Message) -> bool:
    """Return True for user messages and non-marker assistant messages with text.

    Synthetic ``continue`` nudges do not count: a crash-leftover nudge is a
    orchestration placeholder, not user content, and must not inflate
    ``meta.message_count`` (or, through it, ``updated_at`` preservation and
    the buddy total).
    """
    chrys_kind = m.additional_properties.get(HistoryMarkerKind.KEY)
    if chrys_kind in HistoryMarkerKind.SESSION_COUNT_EXCLUDED:
        return False
    if m.role == "user":
        return not is_continuation_message(m)
    if m.role == "assistant":
        return any(c.type == "text" and (c.text or "").strip() for c in m.contents)
    return False


def _parse_session_timestamp(value: object) -> datetime | None:
    """Return one persisted timestamp as an aware UTC datetime, if valid."""
    parsed = parse_created_at(value)
    return coerce_utc(parsed) if parsed is not None else None


def _message_created_at(message: object) -> datetime | None:
    """Read a Chrys message timestamp from either live or serialized state."""
    if isinstance(message, Message):
        properties = message.additional_properties
    elif isinstance(message, dict):
        raw_properties = message.get("additional_properties")
        properties = raw_properties if isinstance(raw_properties, dict) else {}
    else:
        return None
    return _parse_session_timestamp(properties.get(MESSAGE_CREATED_AT_KEY))


def _first_message_created_at(messages: object) -> datetime | None:
    """Return the first valid message timestamp in an ordered message list."""
    if not isinstance(messages, list):
        return None
    for message in messages:
        created_at = _message_created_at(message)
        if created_at is not None:
            return created_at
    return None


def _earliest_history_created_at(state: dict[str, Any]) -> datetime | None:
    """Return the oldest available timestamp from ordered live or serialized history.

    Compaction preserves chronological order. Scan blocks oldest-first until
    one carries a timestamp, then fall back to the live tail; this handles a
    legacy unstamped block followed by blocks created after timestamping was
    introduced. Both runtime ``CompressedBlock`` values and serialized dict
    mirrors are accepted.
    """
    compressed_blocks = state.get("compressed_msgs")
    if isinstance(compressed_blocks, list):
        for block in compressed_blocks:
            if isinstance(block, CompressedBlock):
                compressed_messages: object = block.messages
            elif isinstance(block, dict):
                compressed_messages = block.get("messages")
            else:
                continue
            created_at = _first_message_created_at(compressed_messages)
            if created_at is not None:
                return created_at
    return _first_message_created_at(state.get("messages"))


_TITLE_MAX_LEN = 200

_PROMPT_SEARCH_EDGE_CHARS = 1000
"""Head AND tail characters kept per user message for prompt search."""

_PROMPT_SEARCH_TOTAL_CHARS = 8000
"""Per-session cap on ``SessionMeta.user_prompt_search_text`` (meta-cache bound)."""

_PROMPT_SEARCH_SCAN_BLOCK = 4096
"""Fixed raw-text block size for the bounded collapse scan — the transient
memory unit of ``_collapsed_prompt_excerpt`` (never a growing slice)."""


def _extract_title(messages: list[Message]) -> str:
    """Extract title from the first user message text, truncated.

    Skips synthetic ``continue`` nudges: in the no-prior-user resume shape
    the first user message IS a nudge and the session would be titled
    "continue".
    """
    for m in messages:
        if m.role != "user" or is_continuation_message(m):
            continue
        for c in m.contents:
            if c.type == "text":
                text = c.text or ""
                text = text.strip().replace("\n", " ")
                if text:
                    if len(text) > _TITLE_MAX_LEN:
                        return text[:_TITLE_MAX_LEN] + "..."
                    return text
    return ""


@runtime_checkable
class StateStore(Protocol):
    """Protocol for session state persistence."""

    def session_dir(self, session_id: str) -> Path: ...
    async def save_session(
        self,
        session_id: str,
        state: dict[str, Any],
        *,
        agent_profile: str = "",
        agent_display_name: str = "",
        agent_profile_id: str = "",
        agent_profile_fingerprint: str = "",
        primary_cwd: str = "",
        working_dirs: list[str] | None = None,
        title: str = "",
        agent_profile_history: list[str] | None = None,
        model_provider: str = "",
        model_api_style: str | None = None,
        model_id: str = "",
        model_profile_id: str = "",
        model_base_url: str = "",
        model_profile_fingerprint: str | None = None,
        service_session_id: str | None = None,
        parent_session_id: str = "",
    ) -> SessionCheckpoint | None: ...
    def save_recovery_session(
        self,
        session_id: str,
        state: dict[str, Any],
        *,
        agent_profile: str = "",
        agent_display_name: str = "",
        agent_profile_id: str = "",
        agent_profile_fingerprint: str = "",
        primary_cwd: str = "",
        working_dirs: list[str] | None = None,
        title: str = "",
        agent_profile_history: list[str] | None = None,
        model_provider: str = "",
        model_api_style: str | None = None,
        model_id: str = "",
        model_profile_id: str = "",
        model_base_url: str = "",
        model_profile_fingerprint: str | None = None,
        parent_session_id: str = "",
    ) -> None: ...
    async def load_recovery_session(self, session_id: str) -> dict[str, Any] | None: ...
    async def load_recovery_session_meta(self, session_id: str) -> SessionMeta | None: ...
    def delete_recovery_session(self, session_id: str) -> None: ...
    async def recovery_session_wins(self, session_id: str) -> bool: ...
    async def load_session(self, session_id: str, *, prefer_recovery: bool = False) -> dict[str, Any] | None: ...
    async def load_session_raw(
        self,
        session_id: str,
        *,
        prefer_recovery: bool = False,
    ) -> list[dict[str, Any]] | None: ...
    async def load_session_meta(self, session_id: str, *, prefer_recovery: bool = False) -> SessionMeta | None: ...
    async def list_sessions(self) -> list[SessionMeta]: ...
    async def load_latest_session_id(self) -> str | None: ...
    def stream_session_metas(self, *, batch_size: int = 32) -> AsyncIterator[list[SessionMeta]]: ...
    async def update_session_titles(
        self,
        session_id: str,
        *,
        custom_title: str | None = None,
        generated_title: str | None = None,
    ) -> SessionMeta | None: ...
    def fork_session(self, parent_session_id: str) -> str: ...
    async def delete_session(self, session_id: str, *, allow_active: bool = False) -> None: ...


def _dir_size(path: Path) -> int:
    """Total size in bytes of all files under *path* (non-recursive-safe)."""
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except OSError:
        pass
    return total


def _newer_of(current: SessionMruEntry | None, candidate: SessionMruEntry) -> SessionMruEntry:
    return candidate if current is None else sort_entries([current, candidate])[0]


class JsonFileStateStore:
    """JSON file-based state store.

    Stores sessions as ``{short_session_id}/session.json`` folders with a
    sidecar backup and root-level locks for cross-process coordination.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self._dir = resolve_sessions_dir() if directory is None else Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Custom titles set before a session's first save.  A never-saved
        # session must not be materialized on disk just for its title (it
        # would list as a ghost entry and may still be discarded), so the
        # title waits here until the first full save merges it.
        self._pending_custom_titles: dict[str, str] = {}
        # Listing cache: cache-id -> (file-signature key, SessionMeta).
        # Holds ONLY the small ``SessionMeta`` dataclasses — the multi-MB
        # envelope payloads are parsed transiently and dropped — so memory
        # stays bounded even with thousands of sessions.  Accessed from
        # ``asyncio.to_thread`` workers, so all reads/writes/evictions take
        # ``_meta_cache_lock``; a lost check-then-set race costs at most one
        # redundant re-parse.
        self._meta_cache: dict[str, tuple[tuple[object, ...], SessionMeta]] = {}
        self._meta_cache_lock = threading.Lock()
        # Derived per-root MRU index (``session_mru.json``) so ``/resume``
        # can find the newest session without parsing every envelope.
        # Session files stay authoritative; every index update is
        # best-effort via ``_note_mru`` and never fails a session operation.
        # Writers record *before* committing the envelope (inside the
        # session write lock) so a crash can only leave the index ahead of
        # disk — a state the lookup's verification tolerates — never behind.
        # Whatever still bypasses the index (an older chrys, a copied
        # folder, a record that failed) is caught by the lookup's sweep of
        # folders modified after the indexed winner.
        self._mru = SessionMruIndex(self._dir)
        self._sweep_trajectory_tombstones()

    def _sweep_trajectory_tombstones(self) -> None:
        """Finish logical deletes whose writer has since released its lease."""
        from chrys.service.trajectory.tombstone import sweep_tombstones

        try:
            removed = sweep_tombstones(self._dir)
        except Exception:
            logger.debug("Trajectory tombstone sweep failed under %s", self._dir, exc_info=True)
            return
        if removed:
            logger.info("Removed %d tombstoned session director%s", removed, "y" if removed == 1 else "ies")

    def session_dir(self, session_id: str) -> Path:
        """Return the folder for a session.

        Delegates folder-name derivation to :func:`chrys.foundation.util.session_ids.session_short_id`
        so callers that don't hold a store instance (engine fallback,
        agent_builder fallback) produce bit-identical paths.
        """
        return self._dir / _session_short_id(session_id)

    # Keep private alias for internal callers.
    _session_dir = session_dir

    def _session_file(self, session_id: str) -> Path:
        """Return the session.json path inside the session folder."""
        return self._session_dir(session_id) / SESSION_FILE_NAME

    def _backup_file(self, session_id: str) -> Path:
        """Return the session.json recovery backup path."""
        return self._session_dir(session_id) / SESSION_BACKUP_FILE_NAME

    def _recovery_file(self, session_id: str) -> Path:
        """Return the crash-recovery sidecar path."""
        return self._session_dir(session_id) / SESSION_RECOVERY_FILE_NAME

    def _write_lock_path(self, session_id: str) -> Path:
        """Return the write lock path for this store/session."""
        path = session_write_lock_path(self._dir, session_id)
        _ensure_lock_parent(path)
        return path

    def active_lock_path(self, session_id: str) -> Path:
        """Return the long-lived active-session lock path."""
        path = session_active_lock_path(self._dir, session_id)
        _ensure_lock_parent(path)
        return path

    def active_owner_path(self, session_id: str) -> Path:
        """Return the active-session owner metadata path."""
        path = session_active_owner_path(self._dir, session_id)
        _ensure_lock_parent(path)
        return path

    def _legacy_path(self, session_id: str) -> Path:
        """Return the old flat-file path for migration fallback."""
        return self._dir / f"{_session_short_id(session_id)}.json"

    def _legacy_session_files(self) -> list[Path]:
        """Root-level ``*.json`` legacy flat-file sessions (never the MRU index)."""
        return legacy_session_files(self._dir)

    def _note_mru[T](self, update: Callable[[], T]) -> T | None:
        """Apply a best-effort MRU index update; never fails the session operation.

        On an I/O error the index is invalidated so the next ``/resume``
        rebuilds it from a full scan.  A lock timeout means a stuck peer:
        invalidating would only wait on the same lock again inside the
        session write lock, so the update is just dropped — the lookup's
        modified-folder sweep still finds the committed session.  ``None``
        is returned in place of the update's result either way.
        """
        try:
            return update()
        except TimeoutError:
            logger.warning("Session MRU index %s is locked; skipping update", self._mru.path)
            return None
        except Exception:
            logger.warning("Failed to update session MRU index %s", self._mru.path, exc_info=True)
            with contextlib.suppress(Exception):
                self._mru.invalidate()
            return None

    def _record_mru(self, session_id: str, updated_at: datetime | None) -> None:
        """Index *session_id* at *updated_at*; call this before the envelope commit."""
        if updated_at is None:
            return
        self._note_mru(lambda: self._mru.record(session_id, updated_at))

    def _migrate_if_needed_unlocked(self, session_id: str) -> None:
        """Move a legacy flat-file session into folder format on first access."""
        legacy = self._legacy_path(session_id)
        session_file = self._session_file(session_id)
        if legacy.exists() and not session_file.exists():
            session_file.parent.mkdir(parents=True, exist_ok=True)
            # Record before the rename like every other commit: a backfill
            # scan racing this migration can miss the file in both its
            # folder and its legacy enumeration, and only the index entry
            # keeps the session from vanishing behind a complete index.
            self._record_mru(session_id, self._envelope_updated_at(self._read_json_file(legacy)))
            legacy.rename(session_file)

    def _migrate_if_needed(self, session_id: str) -> None:
        """Move a legacy flat-file session into folder format on first access."""
        if not self._legacy_path(session_id).exists():
            return
        with FileLock(self._write_lock_path(session_id), timeout=SESSION_WRITE_LOCK_TIMEOUT_SECONDS):
            self._migrate_if_needed_unlocked(session_id)

    def _resolve_session_file(self, session_id: str) -> Path | None:
        """Return the session.json path, migrating from legacy if needed."""
        self._migrate_if_needed(session_id)
        path = self._session_file(session_id)
        return path if path.exists() else None

    @staticmethod
    def _read_json_file(path: Path) -> dict[str, Any] | None:
        """Read a JSON object from *path*, returning ``None`` on invalid input."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in session file %s", path)
            return None
        except OSError:
            return None
        return raw if isinstance(raw, dict) else None

    def _snapshot_recovery_files(self, session_id: str) -> list[Path]:
        """Return rollback snapshots newest-first for last-ditch session recovery."""
        snap_dir = self._session_dir(session_id) / "snapshots"
        if not snap_dir.is_dir():
            return []

        return sorted(
            (p for p in snap_dir.glob("*.json") if parse_snapshot_turn(p) >= 1),
            key=parse_snapshot_turn,
            reverse=True,
        )

    def _read_session_envelope(
        self,
        session_id: str,
        *,
        heal: bool = True,
        migrate: bool = True,
        include_snapshots: bool = True,
    ) -> dict[str, Any] | None:
        """Read a session envelope, falling back to backup/snapshots if needed."""
        if migrate:
            path = self._resolve_session_file(session_id)
        else:
            session_file = self._session_file(session_id)
            path = session_file if session_file.exists() else None
        backup = self._backup_file(session_id)

        candidates: list[Path] = []
        if path is not None:
            candidates.append(path)
        if backup.exists():
            candidates.append(backup)
        if include_snapshots:
            candidates.extend(self._snapshot_recovery_files(session_id))
        if not candidates:
            return None

        for candidate in candidates:
            envelope = self._read_json_file(candidate)
            if envelope is None:
                continue
            if heal and candidate != path:
                self._heal_session_file_from_candidate(session_id, self._session_file(session_id), envelope, candidate)
            return envelope
        return None

    def _heal_session_file_from_candidate(
        self,
        session_id: str,
        path: Path,
        envelope: dict[str, Any],
        candidate: Path,
    ) -> None:
        """Best-effort repair of a corrupt primary session file."""
        try:
            payload = json.dumps(envelope, indent=2, ensure_ascii=False)
            with FileLock(self._write_lock_path(session_id), timeout=SESSION_WRITE_LOCK_TIMEOUT_SECONDS):
                current = self._read_json_file(path)
                if current is None:
                    _atomic_write_text(path, payload)
                    _atomic_write_text(self._backup_file(session_id), payload)
                    logger.warning("Recovered corrupt session %s from %s", session_id, candidate)
        except Exception:
            logger.warning("Failed to heal session %s from %s", session_id, candidate, exc_info=True)

    @staticmethod
    def _inheritable_meta(
        session_id: str,
        envelope: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Return metadata when the live envelope can be attributed to the session."""
        if envelope is None:
            return None
        meta = envelope.get("meta")
        if not isinstance(meta, dict) or meta.get("session_id") != session_id:
            return None
        return meta

    @classmethod
    def _title_patch_meta(
        cls,
        session_id: str,
        envelope: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Return attributable metadata whose required timestamps are parseable."""
        meta = cls._inheritable_meta(session_id, envelope)
        if meta is None:
            return None
        created_at = _parse_session_timestamp(meta.get("created_at"))
        updated_at = _parse_session_timestamp(meta.get("updated_at"))
        if created_at is None or updated_at is None:
            return None
        return meta

    def _read_inherited_live_meta_unlocked(
        self,
        session_id: str,
    ) -> tuple[dict[str, Any], str | None, bool]:
        """Return live metadata, creation time, and whether primary supplied the meta.

        This is deliberately separate from ``_read_session_envelope``: save
        inheritance must continue past a parseable but malformed primary to a
        healthy backup, while a timestamp defect repairs only that field. The
        immutable creation time may come from the backup when the selected
        primary meta lacks it. Rollback snapshots are never consulted.
        """
        inherited_meta: dict[str, Any] | None = None
        inherited_from_primary = False
        created_at: str | None = None
        primary_file = self._session_file(session_id)
        for path in (primary_file, self._backup_file(session_id)):
            envelope = self._read_json_file(path)
            meta = self._inheritable_meta(session_id, envelope)
            if meta is None:
                continue
            if inherited_meta is None:
                inherited_meta = meta
                inherited_from_primary = path == primary_file
            value = meta.get("created_at")
            if created_at is None and isinstance(value, str) and _parse_session_timestamp(value) is not None:
                created_at = value
            if inherited_meta is meta and created_at is not None:
                break
        return inherited_meta or {}, created_at, inherited_from_primary

    def _read_recovery_meta_unlocked(self, session_id: str) -> dict[str, Any]:
        """Return current crash-recovery sidecar metadata, best-effort."""
        envelope = self._read_json_file(self._recovery_file(session_id))
        return self._inheritable_meta(session_id, envelope) or {}

    def _effective_title_unlocked(
        self,
        session_id: str,
        existing_meta: dict[str, Any],
        recovery_meta: dict[str, Any],
        *,
        title_key: str,
        prefer_recovery: bool,
    ) -> str:
        """Resolve one title without making healthy saves parse the sidecar.

        Pending custom titles are explicit user actions made after the live
        metadata was read, so they always win until a primary write consumes
        them. Recovery titles outrank live titles only when the primary was
        rejected and the live metadata came from backup (or did not exist).
        """
        if title_key == "custom_title" and session_id in self._pending_custom_titles:
            return self._pending_custom_titles[session_id]

        existing_value = existing_meta.get(title_key)
        recovery_value = recovery_meta.get(title_key)
        candidates = (recovery_value, existing_value) if prefer_recovery else (existing_value, recovery_value)
        return next((value for value in candidates if isinstance(value, str)), "")

    @staticmethod
    def _resolve_created_at(
        live_created_at: str | None,
        recovery_meta: dict[str, Any],
        state: dict[str, Any],
        *,
        now: str,
    ) -> str:
        """Resolve the immutable session creation timestamp for a full save."""
        if live_created_at is not None:
            return live_created_at

        recovery_created_at = _parse_session_timestamp(recovery_meta.get("created_at"))
        history_created_at = _earliest_history_created_at(state)
        fallback_candidates = [
            timestamp for timestamp in (recovery_created_at, history_created_at) if timestamp is not None
        ]
        if fallback_candidates:
            return min(fallback_candidates).isoformat()
        return now

    def _build_session_envelope(
        self,
        session_id: str,
        state: dict[str, Any],
        *,
        agent_profile: str = "",
        agent_display_name: str = "",
        agent_profile_id: str = "",
        agent_profile_fingerprint: str = "",
        primary_cwd: str = "",
        working_dirs: list[str] | None = None,
        title: str = "",
        agent_profile_history: list[str] | None = None,
        model_provider: str = "",
        model_api_style: str | None = None,
        model_id: str = "",
        model_profile_id: str = "",
        model_base_url: str = "",
        model_profile_fingerprint: str | None = None,
        service_session_id: str | None = None,
        parent_session_id: str = "",
        force_updated_at: bool = False,
    ) -> dict[str, Any]:
        """Build the persisted session envelope shared by primary and recovery saves."""
        now = datetime.now(UTC).isoformat()
        existing_meta, live_created_at, inherited_from_primary = self._read_inherited_live_meta_unlocked(session_id)

        # Healthy primary metadata already owns both title fields and the
        # immutable creation time, so the common checkpoint path must not
        # parse a potentially multi-megabyte recovery envelope. Consult it
        # only for recovery/legacy fallback data. A pending custom title is
        # handled by the resolver and always wins until a primary save.
        needs_recovery_meta = (
            not inherited_from_primary
            or live_created_at is None
            or "custom_title" not in existing_meta
            or "generated_title" not in existing_meta
        )
        recovery_meta = self._read_recovery_meta_unlocked(session_id) if needs_recovery_meta else {}
        carry_custom_title = self._effective_title_unlocked(
            session_id,
            existing_meta,
            recovery_meta,
            title_key="custom_title",
            prefer_recovery=not inherited_from_primary,
        )
        carry_generated_title = self._effective_title_unlocked(
            session_id,
            existing_meta,
            recovery_meta,
            title_key="generated_title",
            prefer_recovery=not inherited_from_primary,
        )

        messages = state.get("messages", [])
        new_count = sum(1 for m in messages if _is_visible_message(m))
        old_count = existing_meta.get("message_count", 0)
        created_at = self._resolve_created_at(
            live_created_at,
            recovery_meta,
            state,
            now=now,
        )
        # Normal saves preserve updated_at when visible count is unchanged.
        # Recovery saves always advance it so recency can arbitrate sidecars.
        existing_updated_at = existing_meta.get("updated_at")
        parsed_existing_updated_at = _parse_session_timestamp(existing_updated_at)
        preserve_updated_at = (
            not force_updated_at
            and new_count == old_count
            and isinstance(existing_updated_at, str)
            and parsed_existing_updated_at is not None
        )
        updated_at = existing_updated_at if preserve_updated_at else now
        parsed_created_at = _parse_session_timestamp(created_at)
        parsed_updated_at = _parse_session_timestamp(updated_at)
        updated_at_floor = parsed_created_at
        if force_updated_at and parsed_existing_updated_at is not None:
            # Recovery arbitration requires a strict ``>`` and stale-sidecar
            # GC deletes on ``<=``. The 1µs advance is therefore intentional:
            # do not simplify this to max(now, parsed_existing_updated_at),
            # because equality would discard the new recovery checkpoint.
            recovery_floor = parsed_existing_updated_at + timedelta(microseconds=1)
            updated_at_floor = max(updated_at_floor, recovery_floor) if updated_at_floor is not None else recovery_floor
        if updated_at_floor is not None and parsed_updated_at is not None and parsed_updated_at < updated_at_floor:
            updated_at = updated_at_floor.isoformat()
        # Import lazily to avoid a circular import at module load:
        # ``chrys/__init__.py`` resolves the installed package version,
        # which would otherwise be pulled in at state-store import time.
        from chrys import __version__ as _app_version
        from chrys.foundation.platform import get_platform

        # Platform snapshot is derived here (not taken from the caller)
        # so every save stamps the *actual* runtime the file was written
        # on — the cached ``get_platform()`` is cheap and consistent
        # across saves within a process.
        plat = get_platform()

        return {
            "meta": {
                "schema_version": SESSION_SCHEMA_VERSION,
                "app_version": _app_version,
                "os_name": plat.os_name,
                "arch": plat.arch,
                # Model provenance — preserve existing values when the
                # caller doesn't supply new ones so a late save after a
                # profile swap doesn't blank out the original stamp.
                "model_provider": model_provider or existing_meta.get("model_provider", ""),
                "model_api_style": (
                    model_api_style if model_api_style is not None else existing_meta.get("model_api_style", "")
                ),
                "model_id": model_id or existing_meta.get("model_id", ""),
                "model_profile_id": model_profile_id,
                "model_base_url": model_base_url or existing_meta.get("model_base_url", ""),
                "model_profile_fingerprint": (
                    model_profile_fingerprint
                    if model_profile_fingerprint is not None
                    else existing_meta.get("model_profile_fingerprint", "")
                ),
                "service_session_id": (
                    service_session_id
                    if service_session_id is not None
                    else existing_meta.get("service_session_id", "")
                ),
                "session_id": session_id,
                "agent_profile": agent_profile or existing_meta.get("agent_profile", ""),
                "agent_display_name": (
                    agent_display_name
                    or existing_meta.get("agent_display_name")
                    or existing_meta.get("display_name", "")
                ),
                "agent_profile_id": agent_profile_id,
                "agent_profile_fingerprint": (
                    agent_profile_fingerprint or existing_meta.get("agent_profile_fingerprint", "")
                ),
                "created_at": created_at,
                "updated_at": updated_at,
                "message_count": new_count,
                "primary_cwd": primary_cwd or existing_meta.get("primary_cwd", ""),
                "working_dirs": working_dirs if working_dirs is not None else existing_meta.get("working_dirs", []),
                "title": title or _extract_title(messages) or existing_meta.get("title", ""),
                # Title overlays are written only via ``update_session_titles``;
                # ordinary saves carry forward whatever is on disk (an
                # explicit empty string means the user cleared it).  See the
                # sidecar/pending fallback computed above.
                "custom_title": carry_custom_title,
                "generated_title": carry_generated_title,
                "agent_profile_history": (
                    agent_profile_history
                    if agent_profile_history is not None
                    else existing_meta.get("agent_profile_history", existing_meta.get("profile_history", []))
                ),
                "parent_session_id": parent_session_id or existing_meta.get("parent_session_id", ""),
            },
            "state": serialize_state(state),
            # Minted per save: a derived trajectory summary names the revision
            # it was computed from through this id plus the content hash.
            SESSION_CHECKPOINT_ID_KEY: new_analytics_id(),
        }

    def _delete_recovery_session_unlocked(self, session_id: str) -> None:
        """Best-effort deletion of the recovery sidecar."""
        with contextlib.suppress(OSError):
            self._recovery_file(session_id).unlink()

    def _delete_recovery_session_if_not_newer(
        self,
        session_id: str,
        *,
        primary_updated_at: datetime | None,
    ) -> None:
        """Best-effort stale sidecar GC, rechecking under the write lock."""
        try:
            with FileLock(self._write_lock_path(session_id), timeout=0.0):
                current = self._read_json_file(self._recovery_file(session_id))
                current_updated_at = self._envelope_updated_at(current)
                if (
                    current is None
                    or current_updated_at is None
                    or (primary_updated_at is not None and current_updated_at <= primary_updated_at)
                ):
                    self._delete_recovery_session_unlocked(session_id)
        except TimeoutError:
            logger.debug("Timed out deleting stale recovery sidecar for %s", session_id, exc_info=True)
        except OSError:
            logger.debug("Failed to delete stale recovery sidecar for %s", session_id, exc_info=True)

    def _write_session_envelope_unlocked(self, session_id: str, data: dict[str, Any]) -> SessionCheckpoint:
        """Write primary + backup session envelopes atomically.

        The primary write is authoritative: failures propagate so callers know
        the save did not complete.  The backup is best-effort and can lag by one
        successful write, but is itself updated atomically when it succeeds.
        Returns the checkpoint identity of the bytes written.
        """
        payload = json.dumps(data, indent=2, ensure_ascii=False)
        written = _atomic_write_text(self._session_file(session_id), payload)
        self._delete_recovery_session_unlocked(session_id)
        try:
            _atomic_write_text(self._backup_file(session_id), payload)
        except OSError:
            logger.warning("Failed to update session backup for %s", session_id, exc_info=True)
        return session_checkpoint_of(data, written)

    def _save_session_sync(
        self,
        session_id: str,
        state: dict[str, Any],
        *,
        agent_profile: str = "",
        agent_display_name: str = "",
        agent_profile_id: str = "",
        agent_profile_fingerprint: str = "",
        primary_cwd: str = "",
        working_dirs: list[str] | None = None,
        title: str = "",
        agent_profile_history: list[str] | None = None,
        model_provider: str = "",
        model_api_style: str | None = None,
        model_id: str = "",
        model_profile_id: str = "",
        model_base_url: str = "",
        model_profile_fingerprint: str | None = None,
        service_session_id: str | None = None,
        parent_session_id: str = "",
    ) -> SessionCheckpoint:
        """Sync implementation of session save (runs in a thread)."""
        path = self._session_file(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(self._write_lock_path(session_id), timeout=SESSION_WRITE_LOCK_TIMEOUT_SECONDS):
            self._migrate_if_needed_unlocked(session_id)
            data = self._build_session_envelope(
                session_id,
                state,
                agent_profile=agent_profile,
                agent_display_name=agent_display_name,
                agent_profile_id=agent_profile_id,
                agent_profile_fingerprint=agent_profile_fingerprint,
                primary_cwd=primary_cwd,
                working_dirs=working_dirs,
                title=title,
                agent_profile_history=agent_profile_history,
                model_provider=model_provider,
                model_api_style=model_api_style,
                model_id=model_id,
                model_profile_id=model_profile_id,
                model_base_url=model_base_url,
                model_profile_fingerprint=model_profile_fingerprint,
                service_session_id=service_session_id,
                parent_session_id=parent_session_id,
                force_updated_at=self._recovery_file(session_id).exists(),
            )

            # Index first, commit second: a save that kept ``updated_at``
            # (unchanged visible count) is a no-op for the index.
            self._record_mru(session_id, self._envelope_updated_at(data))
            checkpoint = self._write_session_envelope_unlocked(session_id, data)
            # The primary envelope now owns the title; a stale pending entry
            # must not resurrect it after a later clear.
            self._pending_custom_titles.pop(session_id, None)
            return checkpoint

    async def save_session(
        self,
        session_id: str,
        state: dict[str, Any],
        *,
        agent_profile: str = "",
        agent_display_name: str = "",
        agent_profile_id: str = "",
        agent_profile_fingerprint: str = "",
        primary_cwd: str = "",
        working_dirs: list[str] | None = None,
        title: str = "",
        agent_profile_history: list[str] | None = None,
        model_provider: str = "",
        model_api_style: str | None = None,
        model_id: str = "",
        model_profile_id: str = "",
        model_base_url: str = "",
        model_profile_fingerprint: str | None = None,
        service_session_id: str | None = None,
        parent_session_id: str = "",
    ) -> SessionCheckpoint:
        """Save session state to a JSON file inside the session folder."""
        return await asyncio.to_thread(
            self._save_session_sync,
            session_id,
            state,
            agent_profile=agent_profile,
            agent_display_name=agent_display_name,
            agent_profile_id=agent_profile_id,
            agent_profile_fingerprint=agent_profile_fingerprint,
            primary_cwd=primary_cwd,
            working_dirs=working_dirs,
            title=title,
            agent_profile_history=agent_profile_history,
            model_provider=model_provider,
            model_api_style=model_api_style,
            model_id=model_id,
            model_profile_id=model_profile_id,
            model_base_url=model_base_url,
            model_profile_fingerprint=model_profile_fingerprint,
            service_session_id=service_session_id,
            parent_session_id=parent_session_id,
        )

    def save_recovery_session(
        self,
        session_id: str,
        state: dict[str, Any],
        *,
        agent_profile: str = "",
        agent_display_name: str = "",
        agent_profile_id: str = "",
        agent_profile_fingerprint: str = "",
        primary_cwd: str = "",
        working_dirs: list[str] | None = None,
        title: str = "",
        agent_profile_history: list[str] | None = None,
        model_provider: str = "",
        model_api_style: str | None = None,
        model_id: str = "",
        model_profile_id: str = "",
        model_base_url: str = "",
        model_profile_fingerprint: str | None = None,
        parent_session_id: str = "",
    ) -> None:
        """Synchronously write a crash-recovery sidecar for an in-flight turn."""
        recovery_file = self._recovery_file(session_id)
        recovery_file.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(self._write_lock_path(session_id), timeout=SESSION_WRITE_LOCK_TIMEOUT_SECONDS):
            self._migrate_if_needed_unlocked(session_id)
            data = self._build_session_envelope(
                session_id,
                state,
                agent_profile=agent_profile,
                agent_display_name=agent_display_name,
                agent_profile_id=agent_profile_id,
                agent_profile_fingerprint=agent_profile_fingerprint,
                primary_cwd=primary_cwd,
                working_dirs=working_dirs,
                title=title,
                agent_profile_history=agent_profile_history,
                model_provider=model_provider,
                model_api_style=model_api_style,
                model_id=model_id,
                model_profile_id=model_profile_id,
                model_base_url=model_base_url,
                model_profile_fingerprint=model_profile_fingerprint,
                service_session_id="",
                parent_session_id=parent_session_id,
                force_updated_at=True,
            )
            payload = json.dumps(data, indent=2, ensure_ascii=False)
            self._record_mru(session_id, self._envelope_updated_at(data))
            _common_atomic_write_text(recovery_file, payload)

    @staticmethod
    def _envelope_updated_at(envelope: dict[str, Any] | None) -> datetime | None:
        """Return the envelope updated_at timestamp when it is parseable."""
        if envelope is None:
            return None
        meta = envelope.get("meta", {})
        if not isinstance(meta, dict):
            return None
        return _parse_session_timestamp(meta.get("updated_at"))

    def _resolve_effective_envelope_with_source(
        self,
        session_id: str,
        *,
        prefer_recovery: bool = False,
    ) -> tuple[dict[str, Any] | None, str]:
        """Return the effective envelope and source without healing recovery into primary."""
        self._migrate_if_needed(session_id)
        recovery_path = self._recovery_file(session_id)
        if prefer_recovery and recovery_path.exists():
            recovery = self._read_json_file(recovery_path)
            if recovery is not None:
                primary_path = self._session_file(session_id)
                primary = self._read_json_file(primary_path) if primary_path.exists() else None
                recovery_updated_at = self._envelope_updated_at(recovery)
                primary_updated_at = self._envelope_updated_at(primary)
                if primary is None:
                    return recovery, "recovery"
                if recovery_updated_at is not None and (
                    primary_updated_at is None or recovery_updated_at > primary_updated_at
                ):
                    return recovery, "recovery"
                self._delete_recovery_session_if_not_newer(session_id, primary_updated_at=primary_updated_at)
            else:
                self._delete_recovery_session_if_not_newer(session_id, primary_updated_at=None)
        return self._read_session_envelope(session_id), "primary"

    def _resolve_effective_envelope(
        self,
        session_id: str,
        *,
        prefer_recovery: bool = False,
    ) -> dict[str, Any] | None:
        """Return the effective envelope, optionally preferring a fresh recovery sidecar."""
        envelope, _source = self._resolve_effective_envelope_with_source(
            session_id,
            prefer_recovery=prefer_recovery,
        )
        return envelope

    def _recovery_session_wins_sync(self, session_id: str) -> bool:
        """Return whether the crash-recovery sidecar is the winning effective source."""
        _envelope, source = self._resolve_effective_envelope_with_source(session_id, prefer_recovery=True)
        return source == "recovery"

    async def recovery_session_wins(self, session_id: str) -> bool:
        """Return whether the crash-recovery sidecar would win when recovery is allowed."""
        return await asyncio.to_thread(self._recovery_session_wins_sync, session_id)

    def _active_lock_is_held(self, session_id: str) -> bool:
        """Return True when another live owner, or this process, holds the active-session lock."""
        lock = FileLock(self.active_lock_path(session_id), timeout=0.0)
        try:
            lock.acquire()
        except TimeoutError:
            return True
        else:
            lock.release()
            return False

    def _load_recovery_session_sync(self, session_id: str) -> dict[str, Any] | None:
        """Sync implementation of direct recovery sidecar load."""
        try:
            raw = self._read_json_file(self._recovery_file(session_id))
            if raw is None:
                return None
            return deserialize_state(raw.get("state", {}))
        except KeyError:
            return None

    async def load_recovery_session(self, session_id: str) -> dict[str, Any] | None:
        """Load only the recovery sidecar state, if present."""
        return await asyncio.to_thread(self._load_recovery_session_sync, session_id)

    def _load_recovery_session_meta_sync(self, session_id: str) -> SessionMeta | None:
        """Sync implementation of direct recovery sidecar metadata load."""
        try:
            raw = self._read_json_file(self._recovery_file(session_id))
            if raw is None or "meta" not in raw:
                return None
            return self._session_meta_from_envelope(raw, size_bytes=_dir_size(self._session_dir(session_id)))
        except KeyError, OSError:
            return None

    async def load_recovery_session_meta(self, session_id: str) -> SessionMeta | None:
        """Load only the recovery sidecar metadata, if present."""
        return await asyncio.to_thread(self._load_recovery_session_meta_sync, session_id)

    def delete_recovery_session(self, session_id: str) -> None:
        """Delete the recovery sidecar, if present."""
        with FileLock(self._write_lock_path(session_id), timeout=SESSION_WRITE_LOCK_TIMEOUT_SECONDS):
            self._delete_recovery_session_unlocked(session_id)

    def _load_session_sync(self, session_id: str, *, prefer_recovery: bool = False) -> dict[str, Any] | None:
        """Sync implementation of session load (runs in a thread)."""
        try:
            raw = self._resolve_effective_envelope(session_id, prefer_recovery=prefer_recovery)
            if raw is None:
                return None
            return deserialize_state(raw.get("state", {}))
        except KeyError:
            return None

    async def load_session(self, session_id: str, *, prefer_recovery: bool = False) -> dict[str, Any] | None:
        """Load session state from the session folder."""
        return await asyncio.to_thread(self._load_session_sync, session_id, prefer_recovery=prefer_recovery)

    def _load_session_raw_sync(
        self,
        session_id: str,
        *,
        prefer_recovery: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Sync implementation of raw session load (runs in a thread)."""
        try:
            raw = self._resolve_effective_envelope(session_id, prefer_recovery=prefer_recovery)
            if raw is None:
                return None
            state = raw.get("state", {})
            messages = state.get("messages", [])
            # Normalize: ensure each message has role + contents as dicts.
            # Skip messages marked as excluded by intra-run compaction.
            # Keep compaction summaries — they replace excluded tool calls
            # with a text description and should be visible during replay
            # so the user can see what was compacted.
            result = []
            for msg in messages:
                ap = msg.get("additional_properties", {})
                if ap.get("_excluded", False):
                    continue
                contents = []
                for c in msg.get("contents", []):
                    if isinstance(c, str):
                        # Legacy plain-string format
                        if c.startswith("Content(type="):
                            contents.append({"type": "legacy", "raw": c})
                        else:
                            contents.append({"type": "text", "text": c})
                    elif isinstance(c, dict):
                        contents.append(c)
                entry: dict[str, Any] = {"role": msg.get("role", ""), "contents": contents}
                if msg.get("additional_properties"):
                    entry["additional_properties"] = msg["additional_properties"]
                result.append(entry)
            return result
        except KeyError:
            return None

    async def load_session_raw(
        self,
        session_id: str,
        *,
        prefer_recovery: bool = False,
    ) -> list[dict[str, Any]] | None:
        """Load raw serialized messages for replay.

        Returns:
            A list of message dicts or ``None`` if the session does not
            exist.  Intermediate text (agent text returned alongside tool
            calls) is embedded per-message in
            ``additional_properties["_intermediate_text"]``.
        """
        return await asyncio.to_thread(self._load_session_raw_sync, session_id, prefer_recovery=prefer_recovery)

    @staticmethod
    def _envelope_turn_count(envelope: dict[str, Any]) -> int:
        """Total turns recorded in a serialized envelope's state payload.

        Takes the max of ``state["turn_counter"]`` (authoritative — kept
        aligned with history turn markers by ``SessionHistoryManager``) and
        the evidence surviving on disk: the highest ``_turn`` index stamped
        on live turn markers (with the marker count as a floor) and the
        ``turn_range`` ends of compacted blocks, whose original markers were
        folded away.  The max guards pre-counter sessions later re-saved
        with a restarted counter.
        """
        state = envelope.get("state")
        if not isinstance(state, dict):
            return 0
        counter = state.get("turn_counter", 0)
        best = counter if isinstance(counter, int) and counter > 0 else 0
        messages = state.get("messages", [])
        if isinstance(messages, list):
            marker_count = 0
            for message in messages:
                if not isinstance(message, dict):
                    continue
                props = message.get("additional_properties")
                if not (isinstance(props, dict) and props.get(HistoryMarkerKind.KEY) == HistoryMarkerKind.TURN):
                    continue
                marker_count += 1
                turn_index = props.get(TURN_INDEX_KEY)
                if isinstance(turn_index, int):
                    best = max(best, turn_index)
            best = max(best, marker_count)
        blocks = state.get("compressed_msgs", [])
        if isinstance(blocks, list):
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                turn_range = block.get("turn_range", ())
                if isinstance(turn_range, list | tuple) and len(turn_range) == 2 and isinstance(turn_range[1], int):
                    best = max(best, turn_range[1])
        return best

    @staticmethod
    def _collapsed_prompt_excerpt(text: str, edge: int) -> str:
        """Whitespace-collapsed ``text``, capped to head+tail ``edge`` chars.

        Output-equivalent to capping ``" ".join(text.split())`` in
        O(edge + block) transient memory instead of O(input):
        ``str.split()`` on a multi-megabyte paste of short words costs
        ~14x its size, and even a growing window slice copies O(input)
        raw text when whitespace-dominated input keeps the collapsed
        yield low.  The scan therefore walks FIXED-size blocks by index
        from each end, collapsing block by block and stitching at block
        boundaries: a boundary splitting a word glues the pieces, a
        boundary touching whitespace on either side becomes the single
        space the full collapse would have produced there, and an
        all-whitespace block contributes nothing now but leaves a
        whitespace boundary char for the next stitch to see.  The head
        accumulator is an exact PREFIX of the full collapse and the tail
        accumulator an exact SUFFIX, so each side may stop as soon as it
        covers its capped region.
        """
        block = _PROMPT_SEARCH_SCAN_BLOCK
        limit = 2 * edge
        head = ""
        pos = 0
        while pos < len(text) and len(head) <= limit:
            piece = " ".join(text[pos : pos + block].split())
            if piece:
                if not head:
                    head = piece
                elif text[pos - 1].isspace() or text[pos].isspace():
                    head = f"{head} {piece}"
                else:
                    head += piece
            pos += block
        if len(head) <= limit:
            # The loop only exits with head <= limit once the whole text
            # is consumed, so head IS the full collapse: no cap needed.
            return head
        tail = ""
        pos = len(text)
        while pos > 0 and len(tail) < edge:
            start = max(0, pos - block)
            piece = " ".join(text[start:pos].split())
            if piece:
                if not tail:
                    tail = piece
                elif text[pos - 1].isspace() or text[pos].isspace():
                    tail = f"{piece} {tail}"
                else:
                    tail = piece + tail
            pos = start
        return f"{head[:edge]}\n{tail[-edge:]}"

    @staticmethod
    def _message_prompt_excerpt(message: dict[str, Any]) -> str | None:
        """Capped search excerpt of one user-typed prompt, else ``None``.

        ``None`` means the message is not a user-typed prompt: wrong role,
        synthetic history (turn/status markers, ``continue`` nudges), or no
        text content.  Legacy sessions store ``contents`` as plain strings;
        those count as text under the same rule replay's
        ``_load_session_raw_sync`` uses (a ``Content(type=`` repr blob is
        not text).  Capping each block to its collapsed head+tail before
        joining yields the same final excerpt as capping the full joined
        collapse — the message-level windows only ever read a block's
        first/last ``edge`` chars, which block-level capping preserves.
        """
        if message.get("role") != "user":
            return None
        props = message.get("additional_properties")
        if isinstance(props, dict) and (
            props.get(HistoryMarkerKind.KEY) or props.get(HistoryMarkerKind.CONTINUATION_KEY)
        ):
            return None
        contents = message.get("contents")
        if not isinstance(contents, list):
            return None
        chunks: list[str] = []
        for content in contents:
            if isinstance(content, dict) and content.get("type") == "text":
                text = content.get("text")
            elif isinstance(content, str) and not content.startswith("Content(type="):
                text = content
            else:
                continue
            if isinstance(text, str) and text and not text.isspace():
                chunks.append(JsonFileStateStore._collapsed_prompt_excerpt(text, _PROMPT_SEARCH_EDGE_CHARS))
        if not chunks:
            return None
        joined = "\n".join(chunks)
        if len(joined) > 2 * _PROMPT_SEARCH_EDGE_CHARS:
            joined = f"{joined[:_PROMPT_SEARCH_EDGE_CHARS]}\n{joined[-_PROMPT_SEARCH_EDGE_CHARS:]}"
        return joined

    @staticmethod
    def _envelope_user_prompt_search_text(envelope: dict[str, Any]) -> str:
        """Capped search excerpts of the user-typed prompts in an envelope.

        Walks the messages of the envelope the caller already parsed for
        :meth:`_envelope_turn_count` — compacted originals folded into
        ``compressed_msgs`` blocks first (they are the older turns), then
        the live list — keeping ``role == "user"`` text while skipping
        synthetic history (turn/status markers, ``continue`` nudges).
        Legacy sessions store ``contents`` as plain strings; those count as
        text under the same rule replay's ``_load_session_raw_sync`` uses
        (a ``Content(type=`` repr blob is not text).
        Compaction can leave a message in both places; such twins dedup by
        their capped TEXT, never by ``message_id`` — ids restart across
        runs, so two different prompts can share one (observed in real
        sessions), while an identical indexed string is worthless twice by
        construction.  Whitespace runs inside each text block collapse to single
        spaces so a space-separated query matches a phrase that wrapped
        across lines.  Long prompts keep their head AND tail
        (``_PROMPT_SEARCH_EDGE_CHARS`` each) — users often paste an error
        dump first and type the actual ask at the end.  When the session
        total overruns ``_PROMPT_SEARCH_TOTAL_CHARS`` the same rule applies
        across turns: the earliest and latest prompts survive (half the
        budget each) and the middle is elided.  Dedup then happens INSIDE
        that front/back selection, not before it: repeats of an
        already-kept text skip for free, so a text stays searchable
        whenever ANY of its copies lands in a kept window.  Collapsing
        repeats up front would pin a text to its first copy — and when that
        copy sits in the elided middle, a final-turn repeat of it would
        vanish from the index (review-caught).  Message joins and every
        elision seam use a newline, so no phantom match can straddle them
        (the single-line search box cannot type one).  The scan is
        memory-bounded end to end: block collapse walks fixed-size raw
        blocks (:meth:`_collapsed_prompt_excerpt`), the accumulation
        pass stops retaining as soon as the budget overruns, and the
        over-budget pass re-derives excerpts per message while keeping
        only the selected windows — so a huge paste or a very long session
        allocates O(budget) transient text.  The one whole-session
        allocation left is the flat list of message REFERENCES below,
        pointers into the already-parsed envelope.
        """
        state = envelope.get("state")
        if not isinstance(state, dict):
            return ""
        message_dicts: list[dict[str, Any]] = []
        blocks = state.get("compressed_msgs", [])
        if isinstance(blocks, list):
            for block in blocks:
                folded = block.get("messages", []) if isinstance(block, dict) else []
                if isinstance(folded, list):
                    message_dicts.extend(message for message in folded if isinstance(message, dict))
        messages = state.get("messages", [])
        if isinstance(messages, list):
            message_dicts.extend(message for message in messages if isinstance(message, dict))

        excerpt = JsonFileStateStore._message_prompt_excerpt
        deduped: list[str] = []
        seen_texts: set[str] = set()
        total = 0
        over_budget = False
        for message in message_dicts:
            part = excerpt(message)
            # Twin dedup keys on the indexed text, NEVER message_id: msg_N
            # ids are positional and collide across compaction archives
            # (same id, different prompt — observed in real sessions).
            if part is None or part in seen_texts:
                continue
            # Separator newlines only exist BETWEEN excerpts — charging one
            # for the first excerpt too would flag an index whose joined
            # length is exactly the budget as over it and needlessly evict
            # middle prompts (review-caught).  total tracks the exact
            # length of "\n".join(deduped).
            total += len(part) + 1 if deduped else len(part)
            seen_texts.add(part)
            deduped.append(part)
            if total > _PROMPT_SEARCH_TOTAL_CHARS:
                over_budget = True
                break
        if not over_budget:
            # Every unique text fits, so collapsing repeats onto their first
            # copy loses nothing a substring search could ever see.
            return "\n".join(deduped)
        # Over budget: keep whole prompts from the front and the back until
        # each half fills.  The walks re-derive excerpts in RAW message
        # order — repeats of an already-kept text skip for free (no budget,
        # no break) rather than being pre-collapsed onto a first copy that
        # the elided middle may swallow: a text survives if ANY copy lands
        # in a window.  A capped excerpt (≤ 2·edge+1 chars) always fits an
        # empty half, so the earliest and latest prompts always stay
        # searchable.  Unlike the accumulation pass above, each half DOES
        # charge a separator for its first excerpt: those two spare chars
        # are what keep the final front+back join within the total budget.
        half = _PROMPT_SEARCH_TOTAL_CHARS // 2
        selected: set[str] = set()
        front: list[str] = []
        used = 0
        boundary = len(message_dicts)
        for index, message in enumerate(message_dicts):
            part = excerpt(message)
            if part is None or part in selected:
                continue
            if used + len(part) + 1 > half:
                boundary = index
                break
            front.append(part)
            selected.add(part)
            used += len(part) + 1
        back: list[str] = []
        used = 0
        # The message that overflowed the front half is still eligible for
        # the back half, hence boundary itself is included in the walk.
        for index in range(len(message_dicts) - 1, boundary - 1, -1):
            part = excerpt(message_dicts[index])
            if part is None or part in selected:
                continue
            if used + len(part) + 1 > half:
                break
            back.append(part)
            selected.add(part)
            used += len(part) + 1
        return "\n".join([*front, *reversed(back)])

    @staticmethod
    def _envelope_total_tokens(envelope: dict[str, Any]) -> int:
        """Cumulative session token usage recorded in a serialized envelope."""
        state = envelope.get("state")
        if not isinstance(state, dict):
            return 0
        total = state.get(TOTAL_SESSION_TOKENS_KEY, 0)
        return total if isinstance(total, int) and total > 0 else 0

    @staticmethod
    def _session_meta_from_envelope(envelope: dict[str, Any], *, size_bytes: int) -> SessionMeta:
        """Build a ``SessionMeta`` from a parsed session envelope."""
        meta = envelope["meta"]
        return SessionMeta(
            session_id=meta["session_id"],
            agent_profile=meta.get("agent_profile", ""),
            # Prefer v2 keys; fall back to v1 so pre-rename sessions
            # (``display_name`` / ``profile_history``) still load intact.
            agent_display_name=meta.get("agent_display_name", meta.get("display_name", "")),
            created_at=datetime.fromisoformat(meta["created_at"]),
            updated_at=datetime.fromisoformat(meta["updated_at"]),
            message_count=meta.get("message_count", 0),
            turn_count=JsonFileStateStore._envelope_turn_count(envelope),
            total_tokens=JsonFileStateStore._envelope_total_tokens(envelope),
            primary_cwd=meta.get("primary_cwd") or "",
            working_dirs=meta.get("working_dirs") or [],
            title=meta.get("title", ""),
            custom_title=meta.get("custom_title", ""),
            generated_title=meta.get("generated_title", ""),
            agent_profile_id=meta.get("agent_profile_id", ""),
            agent_profile_history=meta.get("agent_profile_history", meta.get("profile_history", [])),
            agent_profile_fingerprint=meta.get("agent_profile_fingerprint", ""),
            size_bytes=size_bytes,
            app_version=meta.get("app_version", ""),
            schema_version=meta.get("schema_version", 0),
            os_name=meta.get("os_name", ""),
            arch=meta.get("arch", ""),
            model_provider=meta.get("model_provider", ""),
            model_api_style=meta.get("model_api_style", ""),
            model_id=meta.get("model_id", ""),
            model_profile_id=meta.get("model_profile_id", ""),
            model_base_url=meta.get("model_base_url", ""),
            model_profile_fingerprint=meta.get("model_profile_fingerprint", ""),
            service_session_id=meta.get("service_session_id", ""),
            parent_session_id=meta.get("parent_session_id", ""),
            user_prompt_search_text=JsonFileStateStore._envelope_user_prompt_search_text(envelope),
        )

    def _load_session_meta_sync(self, session_id: str, *, prefer_recovery: bool = False) -> SessionMeta | None:
        """Sync implementation of single-session meta load (runs in a thread)."""
        try:
            raw = self._resolve_effective_envelope(session_id, prefer_recovery=prefer_recovery)
            if raw is None or "meta" not in raw:
                return None
            return self._session_meta_from_envelope(raw, size_bytes=_dir_size(self._session_dir(session_id)))
        except KeyError, OSError:
            return None

    async def load_session_meta(self, session_id: str, *, prefer_recovery: bool = False) -> SessionMeta | None:
        """Load metadata for one specific session by id.

        Unlike :py:meth:`list_sessions`, this resolves the session file
        directly via :py:meth:`pathlib.Path.exists`, avoiding the Windows
        ``iterdir`` directory cache that can briefly hide a just-written
        session folder.
        """
        return await asyncio.to_thread(self._load_session_meta_sync, session_id, prefer_recovery=prefer_recovery)

    def _patch_recovery_titles_unlocked(
        self,
        session_id: str,
        *,
        custom_title: str | None,
        generated_title: str | None,
    ) -> None:
        """Mirror title patches into a live crash-recovery sidecar, best-effort."""
        recovery_file = self._recovery_file(session_id)
        if not recovery_file.exists():
            return
        try:
            envelope = self._read_json_file(recovery_file)
            if envelope is None:
                return
            meta = self._inheritable_meta(session_id, envelope)
            if meta is None:
                return
            if custom_title is not None:
                meta["custom_title"] = custom_title
            if generated_title is not None:
                meta["generated_title"] = generated_title
            payload = json.dumps(envelope, indent=2, ensure_ascii=False)
            _common_atomic_write_text(recovery_file, payload)
        except OSError:
            logger.debug("Failed to mirror titles into recovery sidecar for %s", session_id, exc_info=True)

    def _update_session_titles_sync(
        self,
        session_id: str,
        *,
        custom_title: str | None = None,
        generated_title: str | None = None,
    ) -> SessionMeta | None:
        """Sync implementation of the meta-only title patch (runs in a thread)."""
        session_file = self._session_file(session_id)
        with FileLock(self._write_lock_path(session_id), timeout=SESSION_WRITE_LOCK_TIMEOUT_SECONDS):
            self._migrate_if_needed_unlocked(session_id)
            envelope = self._read_session_envelope(
                session_id,
                heal=False,
                migrate=False,
                include_snapshots=False,
            )
            meta = self._title_patch_meta(session_id, envelope)
            if envelope is None or meta is None:
                # Never materialize or rewrite a session file just for a title
                # patch when no semantically valid live envelope is available.
                # Auto-generated titles are simply dropped (the next turn
                # regenerates one); a custom title is held in memory until
                # a later full save can merge it — or discarded with the
                # session if it never saves. A recovery-only session
                # additionally gets the patch mirrored into its sidecar so
                # the edit survives the process (the envelope builder reads
                # it back on the next save).
                if custom_title is not None:
                    has_persisted_target = any(
                        path.exists()
                        for path in (
                            session_file,
                            self._backup_file(session_id),
                            self._recovery_file(session_id),
                        )
                    )
                    if custom_title or has_persisted_target:
                        self._pending_custom_titles[session_id] = custom_title
                    else:
                        self._pending_custom_titles.pop(session_id, None)
                    self._patch_recovery_titles_unlocked(
                        session_id,
                        custom_title=custom_title,
                        generated_title=None,
                    )
                return None
            session_file.parent.mkdir(parents=True, exist_ok=True)
            effective_custom = self._effective_title_unlocked(
                session_id,
                meta,
                {},
                title_key="custom_title",
                prefer_recovery=False,
            )
            effective_generated = self._effective_title_unlocked(
                session_id,
                meta,
                {},
                title_key="generated_title",
                prefer_recovery=False,
            )
            if custom_title is None and generated_title is not None and effective_custom:
                # A custom title permanently disables auto-generated updates.
                return None
            if custom_title is None and generated_title is None:
                return self._meta_from_patched_envelope(session_id, envelope)

            final_custom = effective_custom
            if custom_title is not None and custom_title != effective_custom:
                final_custom = custom_title
            final_generated = effective_generated
            if generated_title is not None and generated_title != effective_generated:
                final_generated = generated_title

            custom_needs_write = meta.get("custom_title") != final_custom
            generated_needs_write = meta.get("generated_title") != final_generated
            if not custom_needs_write and not generated_needs_write:
                # The durable envelope already holds the effective winners;
                # any pending custom candidate is therefore stale or synced.
                self._pending_custom_titles.pop(session_id, None)
                return self._meta_from_patched_envelope(session_id, envelope)
            if custom_needs_write:
                meta["custom_title"] = final_custom
            if generated_needs_write:
                meta["generated_title"] = final_generated
            payload = json.dumps(envelope, indent=2, ensure_ascii=False)
            _atomic_write_text(session_file, payload)
            try:
                _atomic_write_text(self._backup_file(session_id), payload)
            except OSError:
                logger.warning("Failed to update session backup for %s", session_id, exc_info=True)
            self._patch_recovery_titles_unlocked(
                session_id,
                custom_title=final_custom if custom_needs_write else None,
                generated_title=final_generated if generated_needs_write else None,
            )
            # The primary now durably contains the effective custom winner,
            # so no in-memory candidate may override a later full save.
            self._pending_custom_titles.pop(session_id, None)
        return self._meta_from_patched_envelope(session_id, envelope)

    def _meta_from_patched_envelope(self, session_id: str, envelope: dict[str, Any]) -> SessionMeta | None:
        """Best-effort SessionMeta for a title patch's return value.

        The patch itself has already succeeded (or been skipped as a no-op)
        by the time this runs, so malformed meta must degrade to ``None``
        rather than escape as a false failure.  ``TypeError`` covers
        ``datetime.fromisoformat`` on non-string timestamps.
        """
        try:
            return self._session_meta_from_envelope(envelope, size_bytes=_dir_size(self._session_dir(session_id)))
        except KeyError, ValueError, TypeError:
            return None

    async def update_session_titles(
        self,
        session_id: str,
        *,
        custom_title: str | None = None,
        generated_title: str | None = None,
    ) -> SessionMeta | None:
        """Patch the session's title overlay fields without touching state.

        ``None`` leaves a field unchanged; empty string clears it.  A
        ``generated_title``-only patch is refused (returns ``None``) when a
        custom title is already set.  This never creates a session file: for
        a session that has not been saved yet a custom title is held in
        memory (and ``None`` is returned) until the first full save merges
        it, while a generated title is dropped.  Returns the updated
        ``SessionMeta`` on success.
        """
        return await asyncio.to_thread(
            self._update_session_titles_sync,
            session_id,
            custom_title=custom_title,
            generated_title=generated_title,
        )

    def _session_dir_candidates(self) -> list[Path]:
        """Session folders that plausibly hold a restorable session."""
        from chrys.service.trajectory.tombstone import pending_delete_intents

        candidates = session_dir_candidates(self._dir)
        # A folder a delete could not finish removing (its trajectory file was
        # still open) is already gone as far as the user is concerned; listing
        # its remains would resurrect a deleted session.
        pending = pending_delete_intents(self._dir)
        if not pending:
            return candidates
        return [candidate for candidate in candidates if candidate.name not in pending]

    @staticmethod
    def _has_session_artifacts(session_dir: Path) -> bool:
        """Whether *session_dir* holds anything a restore could start from."""
        return session_dir_has_artifacts(session_dir)

    @staticmethod
    def _file_signature(path: Path) -> tuple[int, int]:
        """(mtime_ns, size) change-signature for *path*; ``(-1, -1)`` if absent."""
        try:
            stat = path.stat()
        except OSError:
            return (-1, -1)
        return (stat.st_mtime_ns, stat.st_size)

    def _meta_cache_key(self, session_dir: Path, *, lock_held: bool, dir_size: int) -> tuple[object, ...]:
        """Invalidation key covering every input that shapes a listed meta.

        The active-lock state participates because it decides whether a
        recovery sidecar is eligible to win envelope resolution.  The total
        directory size participates so ``size_bytes`` (and snapshot-healed
        sessions with no primary envelope) can't stay stale when sidecar
        artifacts (snapshots, mutations, logs) change without touching
        ``session.json``.
        """
        return (
            self._file_signature(session_dir / SESSION_FILE_NAME),
            self._file_signature(session_dir / SESSION_BACKUP_FILE_NAME),
            self._file_signature(session_dir / SESSION_RECOVERY_FILE_NAME),
            lock_held,
            dir_size,
        )

    def _meta_for_session_dir(self, session_dir: Path) -> SessionMeta | None:
        """Load one folder-format session's meta, via the listing cache."""
        short_id = session_dir.name
        try:
            lock_held = self._active_lock_is_held(short_id)
            cache_key = self._meta_cache_key(session_dir, lock_held=lock_held, dir_size=_dir_size(session_dir))
            with self._meta_cache_lock:
                cached = self._meta_cache.get(short_id)
            if cached is not None and cached[0] == cache_key:
                return cached[1]
            raw = self._resolve_effective_envelope(short_id, prefer_recovery=not lock_held)
            if raw is None:
                return None
            # Recompute size and key after parsing: envelope resolution may
            # GC a stale recovery sidecar or heal the primary, and caching
            # the pre-parse key would serve a meta whose inputs already
            # moved.
            size_bytes = _dir_size(session_dir)
            meta = self._session_meta_from_envelope(raw, size_bytes=size_bytes)
            post_key = self._meta_cache_key(session_dir, lock_held=lock_held, dir_size=size_bytes)
            with self._meta_cache_lock:
                self._meta_cache[short_id] = (post_key, meta)
            return meta
        except KeyError, ValueError, TypeError, OSError:
            return None

    def _legacy_session_metas(self, seen_session_ids: set[str]) -> list[SessionMeta]:
        """Metas for legacy flat-file sessions not yet migrated to folders.

        Duplicate ``meta.session_id`` values — across flat files, or against
        the folder-format ids in *seen_session_ids* — keep only the first
        occurrence, matching the pre-cache listing behavior.
        """
        sessions: list[SessionMeta] = []
        seen = set(seen_session_ids)
        for path in self._legacy_session_files():
            meta = self._legacy_meta_for_file(path)
            if meta is None or meta.session_id in seen:
                continue
            seen.add(meta.session_id)
            sessions.append(meta)
        return sessions

    def _legacy_meta_for_file(self, path: Path) -> SessionMeta | None:
        """Load one legacy flat-file session's meta, via the listing cache."""
        try:
            cache_id = f"legacy:{path.name}"
            cache_key: tuple[object, ...] = (self._file_signature(path),)
            with self._meta_cache_lock:
                cached = self._meta_cache.get(cache_id)
            if cached is not None and cached[0] == cache_key:
                return cached[1]
            raw = json.loads(path.read_text(encoding="utf-8"))
            meta = self._session_meta_from_envelope(raw, size_bytes=path.stat().st_size)
            with self._meta_cache_lock:
                self._meta_cache[cache_id] = (cache_key, meta)
            return meta
        except json.JSONDecodeError, KeyError, ValueError, TypeError, OSError:
            return None

    def _evict_stale_meta_cache_sync(self) -> None:
        """Drop cache entries whose backing session no longer exists on disk."""
        try:
            live = {p.name for p in self._dir.iterdir() if p.is_dir()}
            live |= {f"legacy:{p.name}" for p in self._legacy_session_files()}
        except OSError:
            return
        with self._meta_cache_lock:
            for key in [k for k in self._meta_cache if k not in live]:
                del self._meta_cache[key]

    def _list_sessions_sync(self) -> list[SessionMeta]:
        """Sync implementation of session listing (runs in a thread)."""
        sessions: list[SessionMeta] = []
        for session_dir in self._session_dir_candidates():
            meta = self._meta_for_session_dir(session_dir)
            if meta is not None:
                sessions.append(meta)
        sessions.extend(self._legacy_session_metas({s.session_id for s in sessions}))
        self._evict_stale_meta_cache_sync()
        return sessions

    async def list_sessions(self) -> list[SessionMeta]:
        """List all saved sessions with metadata."""
        return await asyncio.to_thread(self._list_sessions_sync)

    # ------------------------------------------------------------------ #
    # Latest-session lookup via the MRU index
    # ------------------------------------------------------------------ #

    def _rescan_latest_session_id(self) -> str | None:
        """Full listing scan; rebuilds the MRU index and returns the newest id.

        The rebuild keeps index entries newer than the scan (a save whose
        pre-commit record beat our read of its envelope, or a stale entry the
        scan already contradicts).  The merged ranking — with the scan's top
        as the floor — goes through the same verification as an indexed
        lookup: a matching or newer envelope means the save has landed and
        wins; an unavailable (or, racing a delete, vanished) or older one is
        skipped/re-ranked and the next candidate stands.  The modified-folder
        sweep then covers whatever committed during the scan without leaving
        a rankable record behind, including a first session that landed
        after its in-flight record failed verification.
        """
        scan_started = datetime.now(UTC)
        sessions = self._list_sessions_sync()
        entries = sort_entries(SessionMruEntry(m.session_id, coerce_utc(m.updated_at)) for m in sessions)
        scanned = entries[0] if entries else None
        merged = self._note_mru(lambda: self._mru.rebuild(entries))
        candidates = list(merged) if merged else entries
        if scanned is not None and all(entry.session_id != scanned.session_id for entry in candidates):
            candidates.append(scanned)  # ranked below every record: keep it as the floor
        winner = self._verify_ranked(sort_entries(candidates))
        since = scan_started if winner is None else min(winner.last_updated_at, scan_started)
        best = self._newest_modified_after(winner, since=since)
        return best.session_id if best is not None else None

    def _verify_ranked(self, entries: list[SessionMruEntry]) -> SessionMruEntry | None:
        """Verify *entries* (sorted newest first) against disk; return the winner.

        The top entry is checked against its envelope: an unavailable session
        is dropped, a differing on-disk stamp re-ranks it in memory (only a
        newer stamp is written back — an older one is either transient (a
        live owner keeping the primary ahead of a sidecar, a save between its
        record and its commit) or a rollback that the session's next save
        overwrites, and persisting it would break the crash-recovery case
        where the sidecar must win) and the loop repeats until the top has
        been verified.  ``None`` when every entry is unavailable.
        """
        verified: set[str] = set()
        while entries:
            top = entries[0]
            if top.session_id in verified:
                return top
            actual = self._mru_verify(top.session_id)
            if actual is None:
                entries.pop(0)
                continue
            verified.add(top.session_id)
            if actual != top.last_updated_at:
                entries[0] = SessionMruEntry(top.session_id, actual)
                self._record_mru(top.session_id, actual)
                entries = sort_entries(entries)
        return None

    def _mru_verify(self, session_id: str) -> datetime | None:
        """Return the on-disk ``updated_at`` for an indexed session, or ``None``.

        Uses ``_meta_for_session_dir`` so active-lock, recovery-sidecar and
        backup/snapshot fallback rules match ``list_sessions()`` exactly.  A
        session with nothing restorable on disk — no folder, or a folder a
        pre-commit crash left without envelope/backup/sidecar/snapshots — is
        dropped from the index (see ``_prune_ghost_entry``); one that has
        artifacts but cannot be read right now is only skipped in memory.
        """
        session_dir = self._session_dir(session_id)
        if not self._has_session_artifacts(session_dir) and not self._legacy_path(session_id).exists():
            legacy = self._legacy_meta_for_id(session_id)
            if legacy is not None:  # a legacy flat file under a non-canonical name
                return coerce_utc(legacy.updated_at)
            self._prune_ghost_entry(session_id)
            return None
        self._migrate_if_needed(session_id)
        meta = self._meta_for_session_dir(session_dir)
        if meta is None:  # unreadable folder: the listing may still attribute a legacy file to the id
            meta = self._legacy_meta_for_id(session_id)
        return None if meta is None else coerce_utc(meta.updated_at)

    def _legacy_meta_for_id(self, session_id: str) -> SessionMeta | None:
        """The legacy flat file the listing would attribute to *session_id*, if any.

        Legacy files are discovered by ``*.json``, not by name, so a session
        may live under a file that is not ``<id>.json``; the listing takes
        the first sorted readable file embedding the id, and so does this.
        """
        for path in self._legacy_session_files():
            meta = self._legacy_meta_for_file(path)
            if meta is not None and meta.session_id == session_id:
                return meta
        return None

    def _prune_ghost_entry(self, session_id: str) -> None:
        """Drop an indexed session with nothing restorable — decided under its write lock.

        Saves and forks record their id before the commit that makes the
        envelope (or the directory) appear, so "nothing on disk" alone is
        ambiguous: a busy write lock means a writer is committing it right
        now, and the absence must be re-checked while holding the lock so a
        commit that lands between the first look and this one is not
        mistaken for a ghost.  What survives is a real ghost: a folder a
        pre-commit crash left empty, or a session deleted behind the index.
        """
        try:
            lock = FileLock(self._write_lock_path(session_id), timeout=0.0)
            lock.acquire()
        except TimeoutError:
            return
        except OSError:  # unwritable/odd lock leaf: pruning is optional, the lookup is not
            logger.debug("Skipping ghost prune of %s: write lock unavailable", session_id, exc_info=True)
            return
        try:
            if (
                not self._has_session_artifacts(self._session_dir(session_id))
                and not self._legacy_path(session_id).exists()
                and self._legacy_meta_for_id(session_id) is None
            ):
                self._note_mru(lambda: self._mru.remove(session_id))
        finally:
            lock.release()

    def _newest_modified_after(self, winner: SessionMruEntry | None, *, since: datetime) -> SessionMruEntry | None:
        """Safety net for writers that bypass the index: re-rank *winner* against
        every session whose folder (or legacy file) was modified after *since*
        (the winner's stamp, or the start of a scan that found no winner).

        A session that outranks the winner was necessarily written after the
        winner's ``updated_at``, and writing it touches its folder's mtime —
        so an older chrys, a copied folder, a save whose record failed, or a
        recovery-only session that was hidden behind its owner's active lock
        during the backfill scan all surface here at the cost of one stat per
        folder plus a parse for the (normally zero) recently touched ones.
        Folders are ranked one by one; a recently touched legacy flat file
        defers to the full listing so duplicate-id precedence matches it.
        """
        try:
            cutoff_ns = int((since - _MRU_SWEEP_SLACK).timestamp() * 1_000_000_000)
        except OverflowError, OSError, ValueError:
            cutoff_ns = -1  # absurd stamp (datetime.min/max): parse everything rather than guess
        winner_dir = self._session_dir(winner.session_id) if winner is not None else None
        best = winner
        try:
            candidates = self._session_dir_candidates()
        except OSError:
            return best
        for session_dir in candidates:
            if session_dir == winner_dir or self._file_signature(session_dir)[0] <= cutoff_ns:
                continue
            meta = self._meta_for_session_dir(session_dir)
            if meta is not None:
                best = _newer_of(best, SessionMruEntry(meta.session_id, coerce_utc(meta.updated_at)))
        if any(self._file_signature(legacy)[0] > cutoff_ns for legacy in self._legacy_session_files()):
            # A freshly touched legacy flat file: which copy of a session id
            # counts (folder over legacy, first sorted legacy otherwise) is the
            # listing's rule, so let the listing rank it — legacy roots are
            # transitional and the meta cache makes the repeat cheap.
            for meta in self._list_sessions_sync():
                best = _newer_of(best, SessionMruEntry(meta.session_id, coerce_utc(meta.updated_at)))
        if best is not None and best != winner:
            self._record_mru(best.session_id, best.last_updated_at)
        return best

    def _load_latest_session_id_sync(self) -> str | None:
        """Sync implementation of the MRU-backed latest-session lookup."""
        snapshot = self._mru.load()
        if snapshot is None or not snapshot.complete:
            return self._rescan_latest_session_id()
        winner = self._verify_ranked(list(snapshot.sessions))
        if winner is None:
            # Every indexed entry is gone (or the index is empty): older
            # sessions may still exist below the index horizon, so scan once
            # and rebuild.
            return self._rescan_latest_session_id()
        if snapshot.horizon is not None and winner.last_updated_at < snapshot.horizon:
            # A downgrade dropped the winner below sessions that were trimmed
            # out of the index; only a full scan can rank those.
            return self._rescan_latest_session_id()
        best = self._newest_modified_after(winner, since=winner.last_updated_at)
        return best.session_id if best is not None else None

    async def load_latest_session_id(self) -> str | None:
        """Return the id of the most recently updated restorable session.

        Backed by the ``session_mru.json`` index; a valid index costs one
        small read, verification of the newest session's envelope and a
        stat sweep of the session folders (parsing only those modified after
        the winner), while a missing/corrupt/incomplete index falls back to
        a full listing scan that also rebuilds the index.
        """
        return await asyncio.to_thread(self._load_latest_session_id_sync)

    async def stream_session_metas(self, *, batch_size: int = 32) -> AsyncIterator[list[SessionMeta]]:
        """Yield session metas in batches without blocking the event loop.

        Folders are parsed ``batch_size`` at a time inside worker threads and
        each chunk is yielded as soon as it is ready, so a UI can render
        progressively instead of waiting for a full scan of a potentially
        huge sessions directory.  Unchanged sessions are served from the
        in-process meta cache; the multi-MB envelope payloads themselves are
        never retained.
        """
        seen_ids: set[str] = set()
        session_dirs = await asyncio.to_thread(self._session_dir_candidates)
        for chunk in batched(session_dirs, batch_size, strict=False):

            def _parse(chunk: tuple[Path, ...] = chunk) -> list[SessionMeta]:
                return [meta for d in chunk if (meta := self._meta_for_session_dir(d)) is not None]

            metas = [m for m in await asyncio.to_thread(_parse) if m.session_id not in seen_ids]
            seen_ids.update(m.session_id for m in metas)
            if metas:
                yield metas
        legacy = await asyncio.to_thread(self._legacy_session_metas, seen_ids)
        if legacy:
            yield legacy
        await asyncio.to_thread(self._evict_stale_meta_cache_sync)

    def _delete_session_sync(self, session_id: str, *, allow_active: bool = False) -> None:
        """Sync implementation of session delete (runs in a thread)."""
        session_dir = self._session_dir(session_id)
        legacy = self._legacy_path(session_id)
        if not session_dir.exists() and not legacy.exists():
            return

        active_lock: FileLock | None = None
        if not allow_active:
            active_lock = FileLock(self.active_lock_path(session_id), timeout=SESSION_ACTIVE_LOCK_TIMEOUT_SECONDS)
            active_lock.acquire()
        try:
            with FileLock(self._write_lock_path(session_id), timeout=SESSION_WRITE_LOCK_TIMEOUT_SECONDS):
                if session_dir.is_dir():
                    # A live trajectory writer (a stuck worker here, a writer
                    # in another process) turns this into a logical delete:
                    # the directory moves to a tombstone now and is removed
                    # once the writer's lease is free.
                    from chrys.service.trajectory.tombstone import DeleteOutcome, delete_session_directory

                    delete_result = delete_session_directory(session_dir, sessions_root=self._dir)
                    if delete_result.outcome == DeleteOutcome.INTENT_FAILED and session_dir.exists():
                        # The rename failed, the durable delete intent failed,
                        # and best-effort removal left something behind. No
                        # sweeper owns that directory, so reporting success
                        # would let it reappear after its MRU entry was removed.
                        raise OSError(f"Session directory could not be deleted or scheduled: {session_dir}")
                # Also clean up legacy flat file if it exists
                if legacy.exists():
                    legacy.unlink()
                with contextlib.suppress(OSError):
                    self.active_owner_path(session_id).unlink()
                # Still under the write lock: a same-id save recreating the
                # session records before it can commit, so it cannot slip in
                # between the deletion and this removal and lose its entry.
                self._note_mru(lambda: self._mru.remove(session_id))
                self._pending_custom_titles.pop(session_id, None)
        finally:
            if active_lock is not None:
                active_lock.release()

    async def delete_session(self, session_id: str, *, allow_active: bool = False) -> None:
        """Delete a saved session (folder or legacy file)."""
        await asyncio.to_thread(self._delete_session_sync, session_id, allow_active=allow_active)

    def fork_session(self, parent_session_id: str) -> str:
        """Create an independent copy of *parent_session_id* with a new canonical id."""
        try:
            return self._fork_session_sync(parent_session_id)
        except SessionNotFoundError:
            raise
        except SessionForkError:
            raise
        except Exception as exc:
            raise SessionForkError(f"Failed to fork session {parent_session_id}") from exc

    def _fork_session_sync(self, parent_session_id: str) -> str:
        """Sync implementation of session fork."""
        with FileLock(self._write_lock_path(parent_session_id), timeout=SESSION_WRITE_LOCK_TIMEOUT_SECONDS):
            self._migrate_if_needed_unlocked(parent_session_id)
            parent_dir = self._session_dir(parent_session_id)
            if not parent_dir.is_dir() or not self._session_file(parent_session_id).exists():
                raise SessionNotFoundError(f"Session '{parent_session_id}' not found")

            last_collision: SessionForkError | None = None
            for _attempt in range(SESSION_FORK_MAX_ID_ATTEMPTS):
                new_session_id = self._new_fork_session_id()
                dest_dir = self._session_dir(new_session_id)
                with FileLock(self._write_lock_path(new_session_id), timeout=SESSION_WRITE_LOCK_TIMEOUT_SECONDS):
                    try:
                        self._assert_fork_destination_available(new_session_id, include_write_lock=False)
                    except SessionForkError as exc:
                        last_collision = exc
                        continue
                    tmp_dir = self._new_fork_temp_dir(new_session_id)
                    try:
                        shutil.copytree(
                            parent_dir, tmp_dir, symlinks=True, ignore=self._make_fork_copy_ignore(parent_dir)
                        )
                        reharden_document_image_artifacts(parent_dir, tmp_dir)
                        self._sanitize_fork_sub_agent_layout(tmp_dir)
                        copied_sub_agent_artifacts = self._secure_copy_sub_agent_artifacts(parent_dir, tmp_dir)
                        fork_updated_at = self._prepare_fork_directory(
                            tmp_dir,
                            dest_dir=dest_dir,
                            parent_session_id=parent_session_id,
                            new_session_id=new_session_id,
                            copied_sub_agent_artifacts=copied_sub_agent_artifacts,
                        )
                        # Index before the rename commits the fork.
                        self._record_mru(new_session_id, fork_updated_at)
                        os.replace(tmp_dir, dest_dir)
                        _common_fsync_dir(self._dir)
                        tmp_dir = None
                        return new_session_id
                    finally:
                        if tmp_dir is not None:
                            self._remove_fork_temp_dir(tmp_dir)
            raise SessionForkError("Could not allocate a collision-free fork session id") from last_collision

    def _new_fork_session_id(self) -> str:
        """Return a collision-free canonical session id for a fork."""
        for _attempt in range(SESSION_FORK_MAX_ID_ATTEMPTS):
            candidate = str(uuid4())
            if self._fork_destination_available(candidate):
                return candidate
        raise SessionForkError("Could not allocate a collision-free fork session id")

    def _fork_destination_available(self, session_id: str) -> bool:
        try:
            self._assert_fork_destination_available(session_id)
        except SessionForkError:
            return False
        return True

    def _assert_fork_destination_available(self, session_id: str, *, include_write_lock: bool = True) -> None:
        paths = [
            self._session_dir(session_id),
            self._legacy_path(session_id),
            session_active_lock_path(self._dir, session_id),
            session_active_owner_path(self._dir, session_id),
        ]
        if include_write_lock:
            paths.append(session_write_lock_path(self._dir, session_id))
        for path in paths:
            if path.exists():
                raise SessionForkError(f"Fork destination collision at {path}")

    def _new_fork_temp_dir(self, session_id: str) -> Path:
        for _attempt in range(SESSION_FORK_MAX_ID_ATTEMPTS):
            path = self._dir / f".{_session_short_id(session_id)}.fork.{uuid4().hex}.tmp"
            if not path.exists():
                return path
        raise SessionForkError("Could not allocate a temporary fork directory")

    @staticmethod
    def _remove_fork_temp_dir(path: Path) -> None:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            return
        except OSError:
            logger.warning("Failed to remove temporary fork directory %s", path, exc_info=True)

    def _prepare_fork_directory(
        self,
        tmp_dir: Path,
        *,
        dest_dir: Path,
        parent_session_id: str,
        new_session_id: str,
        copied_sub_agent_artifacts: list[str],
    ) -> datetime:
        """Delete non-copyable files and rewrite fork-local session identity.

        Returns the ``updated_at`` stamped into the fork's envelope so the
        caller can index the new session with the value actually on disk.
        """
        with contextlib.suppress(FileNotFoundError):
            (tmp_dir / SESSION_RECOVERY_FILE_NAME).unlink()
        with contextlib.suppress(FileNotFoundError):
            (tmp_dir / RAW_HTTP_LOG_FILE_NAME).unlink()
        self._prepare_fork_sub_agent_artifacts(tmp_dir)

        fork_updated_at = datetime.now(UTC)
        updated_at = fork_updated_at.isoformat()
        parent_clipboard_root = (self._session_dir(parent_session_id) / "attachments" / "clipboard").resolve(
            strict=False
        )
        fork_clipboard_root = (dest_dir / "attachments" / "clipboard").resolve(strict=False)
        primary = tmp_dir / SESSION_FILE_NAME
        self._rewrite_fork_envelope_file(
            primary,
            parent_session_id=parent_session_id,
            new_session_id=new_session_id,
            updated_at=updated_at,
            parent_clipboard_root=parent_clipboard_root,
            fork_clipboard_root=fork_clipboard_root,
        )
        backup = tmp_dir / SESSION_BACKUP_FILE_NAME
        if backup.exists():
            rewritten = self._try_rewrite_auxiliary_fork_envelope_file(
                backup,
                parent_session_id=parent_session_id,
                new_session_id=new_session_id,
                updated_at=updated_at,
                parent_clipboard_root=parent_clipboard_root,
                fork_clipboard_root=fork_clipboard_root,
            )
            if not rewritten:
                atomic_copy_file(primary, backup)

        from chrys.service.mutations.snapshot_files import rollback_snapshot_paths

        for snapshot in rollback_snapshot_paths(tmp_dir):
            rewritten = self._try_rewrite_auxiliary_fork_envelope_file(
                snapshot,
                parent_session_id=parent_session_id,
                new_session_id=new_session_id,
                updated_at=updated_at,
                parent_clipboard_root=parent_clipboard_root,
                fork_clipboard_root=fork_clipboard_root,
            )
            if not rewritten:
                snapshot.unlink()

        self._rewrite_fork_sub_agent_logs(
            tmp_dir,
            new_session_id=new_session_id,
            parent_clipboard_root=parent_clipboard_root,
            fork_clipboard_root=fork_clipboard_root,
            artifact_names=copied_sub_agent_artifacts,
        )
        return fork_updated_at

    @staticmethod
    def _make_fork_copy_ignore(parent_root: Path) -> Callable[[str, list[str]], set[str]]:
        # Exclude the SESSION-ROOT sub_agents subtree from the generic tree
        # copy. copytree(symlinks=True) copies symlinks verbatim but still
        # FOLLOWS NT directory junctions (os.path.islink() is False for them),
        # so a same-uid untrusted ACP process could plant a junction here that
        # gets materialized as a real directory and evades every post-copy link
        # check — a path back to the parent recurses to disk exhaustion. The
        # only artifacts that carry forward (sessions/*.json + *.stderr.log)
        # are re-created afterward by _secure_copy_sub_agent_artifacts, whose
        # owner-verified writes validate a link-free parent chain.
        parent_key = os.path.normcase(os.path.abspath(parent_root))
        drop_junctions = make_junction_dropping_ignore()

        def _ignore(path: str, names: list[str]) -> set[str]:
            # Drop NT directory junctions at EVERY level, not just the root —
            # a junction planted anywhere the copy reaches, including a nested
            # ``compactions/sub_agents/<tool>/<invocation>/…`` spill dir the
            # same-uid untrusted ACP process can write, would be materialized
            # through plain CopyFile2 (bypassing owner-only recreation).
            excluded = drop_junctions(path, names)
            # Match the session ROOT only for the by-name sub_agents exclusion:
            # nested directories that merely share the name — kernel compaction
            # spill lives under ``compactions/sub_agents/<tool>/<invocation>/…``
            # — must copy through untouched, or the fork loses dropped-turn
            # records while keeping audit/catalog references to them.
            if os.path.normcase(os.path.abspath(path)) == parent_key:
                # Case-insensitive on the name: NTFS is case-insensitive, so a
                # planted ``SUB_AGENTS`` dir would slip past an exact-case check
                # yet copytree would still copy it. Exclude every case variant
                # at the root; sessions/ is rebuilt afterward regardless.
                excluded.update(name for name in names if name.casefold() == "sub_agents")
                # The trajectory log is per-session execution history: the
                # fork opens its own (see service.trajectory.fork), and the
                # parent's writer may hold this one open right now.
                excluded.update(name for name in names if name.casefold() == "trajectory")
            return excluded

        return _ignore

    @staticmethod
    def _sanitize_fork_sub_agent_layout(tmp_dir: Path) -> None:
        """Drop symlinked sub_agents levels from the fork copy before touching them.

        The parent session's sub_agents tree is writable by the same-uid
        UNTRUSTED ACP process, and ``copytree(symlinks=True)`` preserves a
        planted directory symlink verbatim (its children never reach the
        ignore callback). Every later cleanup/rewrite in the fork copy would
        then resolve THROUGH the link and rmtree/unlink data outside the
        session. Links are removed, never followed.
        """
        sub_agents = tmp_dir / "sub_agents"
        for candidate in (sub_agents, sub_agents / "sessions", sub_agents / "pending"):
            try:
                if candidate.is_symlink() or candidate.is_junction():
                    candidate.unlink()
                    logger.warning("Removed linked sub-agent layout entry from fork copy: %s", candidate.name)
            except OSError:
                logger.warning("Failed to sanitize fork sub-agent layout entry: %s", candidate.name, exc_info=True)

    @classmethod
    def _secure_copy_sub_agent_artifacts(cls, parent_dir: Path, tmp_dir: Path) -> list[str]:
        """Rebuild the fork's sub_agents/sessions dir; return the copied names.

        The returned basenames are the EXACT set the rewrite pass must process
        (see _rewrite_fork_sub_agent_logs): the reference-driven pass below can
        copy audits past the anti-flood scan cap, so the rewrite must iterate
        this concrete set rather than re-scanning under the same cap (which would
        leave a referenced-but-uncapped audit copied yet never id-rewritten).
        """
        source_root = parent_dir / "sub_agents"
        source_dir = source_root / "sessions"
        # Same trust boundary as the fork-copy sanitizer: never traverse a
        # linked layout the ACP process may have planted in the live session.
        if source_root.is_symlink() or source_root.is_junction() or source_dir.is_symlink() or source_dir.is_junction():
            logger.warning("Skipping fork copy of linked sub-agent layout: %s", source_root.name)
            return []
        if not source_dir.is_dir():
            return []
        destination_dir = tmp_dir / "sub_agents" / "sessions"
        copied: set[str] = set()

        def _copy_artifact(name: str) -> None:
            try:
                # Owner-verified, not owner-only: parent sessions written by
                # releases predating the owner-only writers must still fork.
                # Bounded: a same-uid untrusted ACP process could plant an
                # over-sized/sparse artifact; over-cap files are skipped.
                payload = read_owner_verified_bounded(source_dir / name, max_bytes=MAX_SUB_AGENT_AUDIT_BYTES)
                atomic_write_owner_only_bytes(destination_dir / name, payload)
            except OSError, ValueError:
                logger.warning("Omitting insecure fork sub-agent artifact: %s", name, exc_info=True)
            copied.add(name)

        # The parent audit dir is same-uid ACP-writable: bound BOTH enumeration
        # (Path.iterdir is eager on 3.14 → a planted flood OOMs the fork during
        # listing) and cumulative read (4096 x 128 MiB would be a huge copy) with
        # the shared scandir scanner. This best-effort pass carries UNREFERENCED
        # artifacts (e.g. reconcile orphans) forward up to the caps; the
        # referenced pass below guarantees the ones the forked history links.
        sources, truncated = scan_bounded_json_files(
            source_dir,
            max_files=MAX_SCANNED_AUDIT_FILES,
            max_total_bytes=MAX_AUDIT_SCAN_TOTAL_BYTES,
            suffixes=(".json", ".stderr.log"),
        )
        if truncated:
            logger.warning("Fork sub-agent artifact scan truncated; unreferenced artifacts beyond the cap are omitted")
        for source in sources:
            _copy_artifact(source.name)

        # Guarantee every audit the forked history references is carried forward
        # even when the bounded scan truncated before reaching it. The reference
        # set comes from the already-copied forked history (session.json), so it
        # is bounded by history size — NOT by the attacker-writable file count —
        # and a legitimately large session can never fork with a dangling
        # sub_agent_log_file link. The generic copytree already placed
        # session.json in tmp_dir; the later fork envelope rewrite leaves these
        # references untouched. A corrupt/over-huge session.json simply yields no
        # referenced set (the bounded scan above still ran).
        try:
            envelope = cls._read_json_file(tmp_dir / SESSION_FILE_NAME)
        except ValueError, RecursionError:
            envelope = None
        if not isinstance(envelope, dict):
            return sorted(copied)
        for name in sorted(collect_sub_agent_log_file_references(envelope)):
            if not is_safe_sub_agent_log_basename(name):
                continue
            if name not in copied:
                _copy_artifact(name)
            # Carry the audit's stderr sibling too (deterministic name); a
            # missing one is normal (the sub-agent may not have written stderr),
            # so only attempt a copy when it is present to avoid log noise.
            stderr_name = f"{name}.stderr.log"
            if stderr_name not in copied and (source_dir / stderr_name).is_file():
                _copy_artifact(stderr_name)
        return sorted(copied)

    @staticmethod
    def _prepare_fork_sub_agent_artifacts(tmp_dir: Path) -> None:
        sub_agents = tmp_dir / "sub_agents"
        # Defense in depth behind _sanitize_fork_sub_agent_layout: if the
        # link removal failed, refusing to traverse beats deleting through it.
        if sub_agents.is_symlink() or sub_agents.is_junction() or not sub_agents.is_dir():
            return
        shutil.rmtree(sub_agents / "pending", ignore_errors=True)
        for legacy_pending in sub_agents.glob("*.json"):
            with contextlib.suppress(OSError):
                legacy_pending.unlink()
        for temp_file in sub_agents.rglob("*.tmp"):
            with contextlib.suppress(OSError):
                temp_file.unlink()

    @classmethod
    def _rewrite_fork_sub_agent_logs(
        cls,
        tmp_dir: Path,
        *,
        new_session_id: str,
        parent_clipboard_root: Path,
        fork_clipboard_root: Path,
        artifact_names: list[str] | None = None,
    ) -> None:
        sessions_dir = tmp_dir / "sub_agents" / "sessions"
        # Defense in depth behind _sanitize_fork_sub_agent_layout: rewritten
        # envelopes must never be written through a linked layout.
        if (
            (tmp_dir / "sub_agents").is_symlink()
            or (tmp_dir / "sub_agents").is_junction()
            or sessions_dir.is_symlink()
            or sessions_dir.is_junction()
            or not sessions_dir.is_dir()
        ):
            return
        if artifact_names is not None:
            # Rewrite EXACTLY the set _secure_copy_sub_agent_artifacts wrote. Its
            # reference-driven pass can copy audits past the anti-flood scan cap,
            # so re-scanning here under the same cap would leave those referenced
            # audits copied yet never id-rewritten (stale parent_session_id). The
            # rebuilt dir is chrys-owned inside the unguessable fork tmp, so this
            # concrete list — bounded by the copier — is the authoritative set.
            rewrite_paths = sorted(sessions_dir / name for name in artifact_names if name.endswith(".json"))
        else:
            # Direct callers (no prior copy) fall back to the bounded scanner so
            # no fork audit scan relies on an eager glob over an ACP-influenced
            # tree.
            rewrite_paths, _truncated = scan_bounded_json_files(
                sessions_dir,
                max_files=MAX_SCANNED_AUDIT_FILES,
                max_total_bytes=MAX_AUDIT_SCAN_TOTAL_BYTES,
            )
        for path in rewrite_paths:
            try:
                envelope = json.loads(
                    read_owner_verified_bounded(path, max_bytes=MAX_SUB_AGENT_AUDIT_BYTES).decode("utf-8")
                )
            except OSError, UnicodeError, json.JSONDecodeError, ValueError, RecursionError:
                # RecursionError (not a ValueError): json.loads recurses per
                # nesting level, so a deeply nested but byte-bounded artifact
                # blows the stack. Skip that one file rather than abort the fork.
                logger.warning("Skipping malformed fork sub-agent audit log: %s", path, exc_info=True)
                continue
            if not isinstance(envelope, dict):
                logger.warning("Leaving opaque fork sub-agent artifact untouched: %s", path)
                continue
            meta = envelope.get("meta")
            if not isinstance(meta, dict):
                logger.warning("Leaving metadata-free fork sub-agent artifact untouched: %s", path)
                continue
            runner = meta.get("runner", "kernel")
            if runner not in {"kernel", "acp"}:
                continue
            if runner == "acp":
                acp_state = envelope.get("acp_state")
                try:
                    if not isinstance(acp_state, dict):
                        raise ValueError("ACP sub-agent log is missing ACP state")
                    validate_acp_state_file_references(acp_state, parent_session_dir=tmp_dir)
                except ValueError:
                    logger.warning("Omitting unsafe ACP fork sub-agent audit log: %s", path, exc_info=True)
                    path.unlink()
                    continue
            meta["parent_session_id"] = new_session_id
            state = envelope.get("state")
            if runner == "kernel" and isinstance(state, dict):
                cls._rewrite_clipboard_mentions_in_state(
                    state,
                    parent_clipboard_root=parent_clipboard_root,
                    fork_clipboard_root=fork_clipboard_root,
                )
            try:
                serialized = json.dumps(envelope, indent=2, ensure_ascii=False, allow_nan=False)
            except ValueError:
                # A legacy artifact written with the default allow_nan=True can
                # carry NaN/Infinity that json.loads accepts but the strict
                # re-encode rejects. Skip this one file (leave the original,
                # parent-scoped copy) rather than aborting the entire fork.
                logger.warning("Leaving non-encodable fork sub-agent artifact untouched: %s", path, exc_info=True)
                continue
            # A lone surrogate a same-uid child may have planted in the audit is
            # neutralized at the sink (atomic_write_owner_only_text encodes with
            # backslashreplace), so this write cannot abort the fork.
            atomic_write_owner_only_text(path, serialized)

    def _rewrite_fork_envelope_file(
        self,
        path: Path,
        *,
        parent_session_id: str,
        new_session_id: str,
        updated_at: str,
        parent_clipboard_root: Path,
        fork_clipboard_root: Path,
    ) -> None:
        envelope = self._read_json_file(path)
        if envelope is None:
            raise SessionForkError(f"Cannot rewrite invalid session envelope: {path}")
        self._rewrite_fork_envelope(
            envelope,
            parent_session_id=parent_session_id,
            new_session_id=new_session_id,
            updated_at=updated_at,
            parent_clipboard_root=parent_clipboard_root,
            fork_clipboard_root=fork_clipboard_root,
        )
        _atomic_write_text(path, json.dumps(envelope, indent=2, ensure_ascii=False))

    def _try_rewrite_auxiliary_fork_envelope_file(
        self,
        path: Path,
        *,
        parent_session_id: str,
        new_session_id: str,
        updated_at: str,
        parent_clipboard_root: Path,
        fork_clipboard_root: Path,
    ) -> bool:
        envelope = self._read_json_file(path)
        if envelope is None:
            logger.warning("Skipping invalid fork auxiliary session envelope: %s", path)
            return False
        try:
            self._rewrite_fork_envelope(
                envelope,
                parent_session_id=parent_session_id,
                new_session_id=new_session_id,
                updated_at=updated_at,
                parent_clipboard_root=parent_clipboard_root,
                fork_clipboard_root=fork_clipboard_root,
            )
        except SessionForkError:
            logger.warning("Skipping malformed fork auxiliary session envelope: %s", path, exc_info=True)
            return False
        _atomic_write_text(path, json.dumps(envelope, indent=2, ensure_ascii=False))
        return True

    @classmethod
    def _rewrite_fork_envelope(
        cls,
        envelope: dict[str, Any],
        *,
        parent_session_id: str,
        new_session_id: str,
        updated_at: str,
        parent_clipboard_root: Path,
        fork_clipboard_root: Path,
    ) -> None:
        meta = envelope.get("meta")
        if not isinstance(meta, dict):
            raise SessionForkError("Session envelope is missing metadata")
        meta["session_id"] = new_session_id
        meta["parent_session_id"] = parent_session_id
        meta["updated_at"] = updated_at
        meta["service_session_id"] = ""
        # The fork is a new on-disk revision: derived summaries must not be
        # able to claim they were computed from it via the parent's id.
        envelope[SESSION_CHECKPOINT_ID_KEY] = new_analytics_id()
        # A user-pinned custom_title carries over verbatim (losing the rename
        # reads as data loss).  The carried pin keeps auto-generated refreshes
        # disabled on the fork, exactly as on the parent, until renamed.

        state = envelope.get("state")
        if isinstance(state, dict):
            # The fork records into a log of its own, numbered from one: the
            # turn registry's sequences point into the parent's log and would
            # make a rollback here name a range that never existed.
            state.pop(TRAJECTORY_STATE_KEY, None)
            cls._rewrite_clipboard_mentions_in_state(
                state,
                parent_clipboard_root=parent_clipboard_root,
                fork_clipboard_root=fork_clipboard_root,
            )

    @classmethod
    def _rewrite_clipboard_mentions_in_state(
        cls,
        state: dict[str, Any],
        *,
        parent_clipboard_root: Path,
        fork_clipboard_root: Path,
    ) -> None:
        messages = state.get("messages")
        if isinstance(messages, list):
            cls._rewrite_clipboard_mentions_in_messages(
                messages,
                parent_clipboard_root=parent_clipboard_root,
                fork_clipboard_root=fork_clipboard_root,
            )
        compressed_blocks = state.get("compressed_msgs")
        if not isinstance(compressed_blocks, list):
            return
        for block in compressed_blocks:
            if not isinstance(block, dict):
                continue
            block_messages = block.get("messages")
            if isinstance(block_messages, list):
                cls._rewrite_clipboard_mentions_in_messages(
                    block_messages,
                    parent_clipboard_root=parent_clipboard_root,
                    fork_clipboard_root=fork_clipboard_root,
                )

    @classmethod
    def _rewrite_clipboard_mentions_in_messages(
        cls,
        messages: list[Any],
        *,
        parent_clipboard_root: Path,
        fork_clipboard_root: Path,
    ) -> None:
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            contents = message.get("contents")
            if not isinstance(contents, list):
                continue
            for index, content in enumerate(contents):
                if isinstance(content, str):
                    contents[index] = cls._rewrite_clipboard_mentions_in_text(
                        content,
                        parent_clipboard_root=parent_clipboard_root,
                        fork_clipboard_root=fork_clipboard_root,
                    )
                    continue
                if not _is_string_keyed_dict(content) or content.get("type") != "text":
                    continue
                text = content.get("text")
                if isinstance(text, str):
                    content["text"] = cls._rewrite_clipboard_mentions_in_text(
                        text,
                        parent_clipboard_root=parent_clipboard_root,
                        fork_clipboard_root=fork_clipboard_root,
                    )

    @staticmethod
    def _rewrite_clipboard_mentions_in_text(
        text: str,
        *,
        parent_clipboard_root: Path,
        fork_clipboard_root: Path,
    ) -> str:
        tokens = iter_mention_tokens(text)
        if not tokens:
            return text

        pieces: list[str] = []
        last = 0
        changed = False
        for token in tokens:
            # Clipboard mentions written by the TUI are absolute paths. Relative
            # hand-authored mentions resolve against the process cwd here and
            # normally will not match the session-local clipboard directory.
            resolved = JsonFileStateStore._resolve_mention_path(token.value)
            if resolved is None:
                continue
            try:
                relative = resolved.relative_to(parent_clipboard_root)
            except ValueError:
                continue
            pieces.append(text[last : token.start])
            pieces.append(format_file_mention(fork_clipboard_root / relative))
            last = token.end
            changed = True
        if not changed:
            return text
        pieces.append(text[last:])
        return "".join(pieces)

    @staticmethod
    def _resolve_mention_path(value: str) -> Path | None:
        try:
            return Path(value).expanduser().resolve(strict=False)
        except OSError:
            return None
        except RuntimeError:
            return None
        except ValueError:
            return None
