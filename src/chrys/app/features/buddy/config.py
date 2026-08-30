# Copyright (c) 2026 Chrys. All rights reserved.

"""Configuration persistence for Buddy companion.

Two chrys instances share a single ``buddy.json`` in the buddy platform config directory.
All read-modify-write paths go through :func:`mutate_buddy_config`, which holds
an exclusive :class:`FileLock` on a sidecar ``buddy.lock`` for the entire
critical section and uses an atomic ``os.replace`` write so a crash mid-write
cannot corrupt the file. Pure reads (``load_buddy_config`` / ``is_muted`` /
``has_hatched``) stay lock-free — ``os.replace`` is atomic, so readers see
either the old or new file, never a torn one.

On-disk format (v1)
-------------------
The file is wrapped in an obfuscated envelope so casual hand-edits don't
silently corrupt buddy state::

    {
      "v": 1,
      "data": "<base64-encoded nonce+ciphertext>",
      "sig": "<hex HMAC-SHA256 of data>"
    }

This is **obfuscation, not security**. The encryption and signing keys are
derived deterministically from ``platform.node()`` + ``getpass.getuser()``
and a constant baked into source. Anyone with access to the source and the
file can reproduce both keys and forge a valid envelope. The goal is to
deter casual hand-edits and detect accidental corruption — not to protect
against a determined user editing their own buddy data.

A sidecar ``buddy.json.bak`` holds the previous write. When the main file
is corrupted or tampered with (sig mismatch, bad base64, decrypt failure)
the config is transparently restored from the backup. Both files must be
destroyed for data to be permanently lost.

Legacy plaintext files (no envelope) are accepted on read for backward
compatibility and rewritten to v1 on the next write.
"""

from __future__ import annotations

import base64
import contextlib
import getpass
import hashlib
import hmac
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from chrys.foundation.platform import get_platform
from chrys.foundation.util.lock import FileLock

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from chrys.app.features.buddy.types import StoredCompanion

CONFIG_DIR = Path("extras") / "buddy"
CONFIG_FILE = "buddy.json"
LOCK_FILE = "buddy.lock"

# Bound the wait so a wedged peer process cannot deadlock the TUI forever.
_LOCK_TIMEOUT_SECONDS = 10.0

# HMAC-SHA256 produces 32-byte digests — natural block size for the keystream.
_BLOCK_SIZE = 32


# ── key derivation ──────────────────────────────────────────────────────────


def _derive_encryption_key() -> bytes:
    """Deterministic encryption key from machine identity."""
    raw = f"{platform.node()}::{getpass.getuser()}::buddy-encrypt-2026"
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _derive_signing_key() -> bytes:
    """Deterministic signing key from machine identity (separate from encryption)."""
    raw = f"{platform.node()}::{getpass.getuser()}::buddy-integrity-2026"
    return hashlib.sha256(raw.encode("utf-8")).digest()


# ── obfuscation (HMAC-SHA256 in counter mode) ───────────────────────────────
# Not real encryption — see module docstring. Stream cipher built from stdlib
# HMAC-SHA256 in counter mode so we don't pull in pyca/cryptography.


def _encrypt(plaintext: bytes) -> bytes:
    """Obfuscate *plaintext* with a random nonce prepended.

    Uses HMAC-SHA256 in counter mode so the only dependency is stdlib.
    """
    key = _derive_encryption_key()
    nonce = os.urandom(_BLOCK_SIZE)
    result = bytearray(nonce)
    for i in range(0, len(plaintext), _BLOCK_SIZE):
        block = plaintext[i : i + _BLOCK_SIZE]
        counter = (i // _BLOCK_SIZE).to_bytes(8, "big")
        keystream = hmac.new(key, nonce + counter, hashlib.sha256).digest()
        result.extend(b ^ k for b, k in zip(block, keystream, strict=False))
    return bytes(result)


def _decrypt(data: bytes) -> bytes | None:
    """De-obfuscate *data* (nonce + ciphertext). Returns ``None`` on failure."""
    if len(data) < _BLOCK_SIZE:
        return None
    try:
        key = _derive_encryption_key()
        nonce = data[:_BLOCK_SIZE]
        ciphertext = data[_BLOCK_SIZE:]
        plaintext = bytearray()
        for i in range(0, len(ciphertext), _BLOCK_SIZE):
            block = ciphertext[i : i + _BLOCK_SIZE]
            counter = (i // _BLOCK_SIZE).to_bytes(8, "big")
            keystream = hmac.new(key, nonce + counter, hashlib.sha256).digest()
            plaintext.extend(b ^ k for b, k in zip(block, keystream, strict=False))
        return bytes(plaintext)
    except ValueError, TypeError:
        return None


# ── signing ─────────────────────────────────────────────────────────────────


def _compute_hmac(payload: str) -> str:
    """HMAC-SHA256 hex digest for *payload*."""
    key = _derive_signing_key()
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _verify_hmac(payload: str, signature: str) -> bool:
    """Constant-time comparison of HMAC for *payload*."""
    return hmac.compare_digest(_compute_hmac(payload), signature)


# ── pack / unpack ───────────────────────────────────────────────────────────


def _pack(plain_config: Mapping[str, Any]) -> str:
    """Obfuscate *plain_config* and wrap it in the signed envelope."""
    plaintext = json.dumps(plain_config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ciphertext = _encrypt(plaintext)
    data_b64 = base64.b64encode(ciphertext).decode("ascii")
    sig = _compute_hmac(data_b64)
    return json.dumps({"v": 1, "data": data_b64, "sig": sig}, indent=2)


def _unpack(raw: str) -> dict[str, Any] | None:
    """Parse the signed envelope, verify, and decrypt.  Returns ``None`` on any failure."""
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError, ValueError:
        return None
    if not isinstance(envelope, dict):
        return None

    # v1 envelope: {v, data, sig}
    if envelope.get("v") == 1 and "data" in envelope and "sig" in envelope:
        data_b64 = str(envelope["data"])
        sig = str(envelope["sig"])
        if not _verify_hmac(data_b64, sig):
            return None
        try:
            ciphertext = base64.b64decode(data_b64, validate=True)
        except ValueError:
            # binascii.Error (raised on non-base64 input) is a ValueError subclass.
            return None
        plaintext = _decrypt(ciphertext)
        if plaintext is None:
            return None
        try:
            return json.loads(plaintext.decode("utf-8"))
        except json.JSONDecodeError, UnicodeDecodeError:
            return None

    # Legacy plaintext format — accept for backward compatibility.
    # The next write will upgrade to v1 automatically.
    if "name" in envelope and "personality" in envelope and "hatchedAt" in envelope:
        return envelope

    return None


# ── file paths ──────────────────────────────────────────────────────────────


def _config_path() -> Path:
    """Get path to buddy config file."""
    return get_platform().config_dir / CONFIG_DIR / CONFIG_FILE


def _backup_path() -> Path:
    """Sidecar backup used for automatic recovery."""
    return _config_path().with_name(CONFIG_FILE + ".bak")


def _lock_path() -> Path:
    """Get path to buddy lock file (sidecar of buddy.json).

    Derived from ``_config_path()`` so tests that patch only the config path
    automatically redirect the lock file too.
    """
    return _config_path().with_name(LOCK_FILE)


def _ensure_dir(path: Path) -> None:
    """Create parent directory of ``path`` if missing (idempotent, race-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ── decode helpers ──────────────────────────────────────────────────────────


def _decode_full(raw: str) -> dict[str, Any] | None:
    """Parse the on-disk record (Soul + optional ``muted``).

    Returns ``None`` when the file is missing, corrupt, or tampered with.
    """
    data = _unpack(raw)
    if data is None:
        return None
    if "name" not in data or "personality" not in data or "hatchedAt" not in data:
        return None
    try:
        full: dict[str, Any] = {
            "name": str(data["name"]),
            "personality": str(data["personality"]),
            "hatchedAt": int(data["hatchedAt"]),
            "petCount": int(data.get("petCount", 0)),
            "evolutionCount": int(data.get("evolutionCount", 0)),
            "totalMessageCount": int(data.get("totalMessageCount", 0)),
        }
    except TypeError, ValueError:
        return None
    if "muted" in data:
        full["muted"] = bool(data["muted"])
    return full


def _to_soul(full: Mapping[str, Any]) -> StoredCompanion:
    """Project a full record down to the Soul-only view (no ``muted``)."""
    return {
        "name": full["name"],
        "personality": full["personality"],
        "hatchedAt": int(full["hatchedAt"]),
        "petCount": int(full.get("petCount", 0)),
        "evolutionCount": int(full.get("evolutionCount", 0)),
        "totalMessageCount": int(full.get("totalMessageCount", 0)),
    }


def _decode_soul(raw: str) -> StoredCompanion | None:
    """Parse the Soul-only view of the config (no muted flag)."""
    full = _decode_full(raw)
    return _to_soul(full) if full is not None else None


# ── read paths (lock-free) ──────────────────────────────────────────────────


def _read_file_with_fallback(path: Path) -> str | None:
    """Read *path*; if unreadable or decode would fail, try *path*.bak.

    Returns the raw file text or ``None`` when both files are unavailable.
    The lock-free read path deliberately does **not** rewrite the primary
    even when it's corrupt — that would race with any concurrent writer
    and break the "pure reads stay lock-free" invariant. Healing happens
    on the next ``mutate_buddy_config`` / ``save_buddy_config`` call,
    which holds the lock and refreshes both primary and backup atomically.
    """
    # Try primary path first.
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        raw = None

    if raw is not None and _decode_full(raw) is not None:
        return raw

    # Primary missing or corrupt — fall back to backup.
    try:
        raw = _backup_path().read_text(encoding="utf-8")
    except OSError:
        return None
    return raw if _decode_full(raw) is not None else None


def _read_full_unlocked(path: Path) -> dict[str, Any] | None:
    """Read the full config (with ``muted``). Caller may already hold the lock."""
    raw = _read_file_with_fallback(path)
    return _decode_full(raw) if raw is not None else None


def _read_soul_unlocked(path: Path) -> StoredCompanion | None:
    """Read the Soul-only view. Caller may already hold the lock."""
    raw = _read_file_with_fallback(path)
    return _decode_soul(raw) if raw is not None else None


# ── write paths ─────────────────────────────────────────────────────────────


def _atomic_write(path: Path, payload: str) -> None:
    """Write *payload* to *path* atomically, then sync to backup.

    Writes a temp file in the same directory (so ``os.replace`` is atomic on
    POSIX and Windows), fsyncs, then renames into place.  After the primary
    succeeds the backup file is updated so it always lags by at most one
    successful write.
    """
    _ensure_dir(path)

    # Write primary.
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise

    # Sync backup (best-effort — failure here does not affect primary).
    bak = _backup_path()
    try:
        fd2, tmp2_name = tempfile.mkstemp(prefix=bak.name + ".", suffix=".tmp", dir=bak.parent)
        tmp2_path = Path(tmp2_name)
        try:
            with os.fdopen(fd2, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp2_path, bak)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                tmp2_path.unlink()
            raise
    except OSError:
        pass


def _serialize(config: Mapping[str, Any]) -> str:
    """Render the full config record (Soul + muted) to the encrypted envelope."""
    data = {
        "name": config["name"],
        "personality": config["personality"],
        "hatchedAt": config["hatchedAt"],
        "petCount": config.get("petCount", 0),
        "evolutionCount": int(config.get("evolutionCount", 0)),
        "totalMessageCount": int(config.get("totalMessageCount", 0)),
        "muted": bool(config.get("muted", False)),
    }
    return _pack(data)


# ── public API ──────────────────────────────────────────────────────────────


def load_buddy_config() -> StoredCompanion | None:
    """Load the Soul view of the buddy config (no ``muted`` field).

    Lock-free read — ``os.replace`` keeps writes atomic, so this returns the
    pre- or post-write state but never a partial one.  If the primary file is
    corrupted the backup is tried transparently.

    Returns:
        StoredCompanion dict if exists, None if not hatched yet.
    """
    return _read_soul_unlocked(_config_path())


def save_buddy_config(config: StoredCompanion) -> None:
    """Overwrite the on-disk config with *config*.

    Held under the cross-process lock and written atomically. Preserves the
    on-disk ``muted`` flag if not explicitly set in *config*.  Prefer
    :func:`mutate_buddy_config` for read-modify-write sequences.
    """
    path = _config_path()
    _ensure_dir(path)
    with FileLock(_lock_path(), timeout=_LOCK_TIMEOUT_SECONDS):
        existing = _read_full_unlocked(path) or {}
        merged = {**existing, **config}
        _atomic_write(path, _serialize(merged))


@overload
def mutate_buddy_config(
    fn: Callable[[StoredCompanion | None], StoredCompanion | None],
) -> StoredCompanion | None: ...


@overload
def mutate_buddy_config(
    fn: Callable[[dict[str, Any] | None], dict[str, Any] | None],
) -> StoredCompanion | None: ...


def mutate_buddy_config(
    fn: Callable[[Any], Any],
) -> StoredCompanion | None:
    """Atomically read-modify-write the buddy config.

    *fn* receives the current full record (Soul + ``muted``) as a plain dict
    or ``None`` if not hatched, and returns the new full record — or ``None`` to
    leave the file untouched.  The whole sequence runs under an exclusive
    cross-process lock and the write is atomic.

    Returns the post-mutation Soul view (or ``None`` if *fn* opted out and the
    file was empty).
    """
    path = _config_path()
    _ensure_dir(path)
    with FileLock(_lock_path(), timeout=_LOCK_TIMEOUT_SECONDS):
        current = _read_full_unlocked(path)
        updated = fn(current)
        if updated is None:
            return _to_soul(current) if current is not None else None
        _atomic_write(path, _serialize(updated))
        return _to_soul(updated)


def update_buddy_config(**updates) -> StoredCompanion | None:
    """Update specific fields in buddy config.

    Args:
        **updates: Keyword arguments to update (e.g., name="Fluffy", muted=True, petCount=10)

    Returns:
        Updated config if exists, None if not hatched yet.
    """
    # Derive from StoredCompanion so adding a field there updates this set.
    # Local import keeps the module top free of runtime type imports.
    from chrys.app.features.buddy.types import StoredCompanion

    allowed = set(StoredCompanion.__annotations__) | {"muted"}
    sanitized = {k: v for k, v in updates.items() if k in allowed}

    def _apply(current: dict[str, Any] | None) -> dict[str, Any] | None:
        if current is None:
            return None
        merged = dict(current)
        merged.update(sanitized)
        return merged

    return mutate_buddy_config(_apply)


def is_muted() -> bool:
    """Check if buddy notifications are muted."""
    full = _read_full_unlocked(_config_path())
    if full is None:
        return False
    return bool(full.get("muted", False))


def set_muted(muted: bool) -> None:
    """Enable or disable buddy notifications."""

    def _apply(current: dict[str, Any] | None) -> dict[str, Any] | None:
        if current is None:
            return None
        merged = dict(current)
        merged["muted"] = muted
        return merged

    mutate_buddy_config(_apply)


def increment_pet_count() -> StoredCompanion | None:
    """Atomically increment ``petCount`` under the cross-process lock.

    Returns the post-mutation Soul view, or ``None`` if no companion is
    hatched.
    """

    def _bump(current: dict[str, Any] | None) -> dict[str, Any] | None:
        if current is None:
            return None
        merged = dict(current)
        merged["petCount"] = int(merged.get("petCount", 0)) + 1
        return merged

    return mutate_buddy_config(_bump)


def increment_message_count() -> StoredCompanion | None:
    """Atomically increment ``totalMessageCount`` under the cross-process lock.

    Returns the post-mutation Soul view, or ``None`` if no companion is
    hatched.
    """

    def _bump(current: dict[str, Any] | None) -> dict[str, Any] | None:
        if current is None:
            return None
        merged = dict(current)
        merged["totalMessageCount"] = int(merged.get("totalMessageCount", 0)) + 1
        return merged

    return mutate_buddy_config(_bump)


def delete_buddy_config() -> None:
    """Delete buddy configuration and backup (reset pet)."""
    path = _config_path()
    _ensure_dir(path)
    with FileLock(_lock_path(), timeout=_LOCK_TIMEOUT_SECONDS):
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        with contextlib.suppress(FileNotFoundError):
            _backup_path().unlink()


def has_hatched() -> bool:
    """Check if user has hatched a buddy."""
    return load_buddy_config() is not None
