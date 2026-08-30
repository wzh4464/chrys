# Copyright (c) 2026 Chrys. All rights reserved.

"""Concurrency safety tests for buddy config persistence.

Two chrys instances share one persisted ``buddy.json``. These tests
exercise the locking + atomic-write path to make sure concurrent petting,
hatching, and config updates do not lose data or corrupt the file.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from chrys.app.features.buddy import config as buddy_config


@pytest.fixture
def buddy_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect buddy config + lock files into ``tmp_path``."""
    config_path = tmp_path / "buddy.json"
    lock_path = tmp_path / "buddy.lock"
    monkeypatch.setattr(buddy_config, "_config_path", lambda: config_path)
    monkeypatch.setattr(buddy_config, "_lock_path", lambda: lock_path)
    return tmp_path


def _seed_companion(petCount: int = 0) -> None:
    """Write a baseline hatched companion to disk."""
    buddy_config.save_buddy_config(
        {
            "name": "Test",
            "personality": "tester",
            "hatchedAt": 1_700_000_000_000,
            "petCount": petCount,
        }
    )


# ── basic semantics ────────────────────────────────────────────────────────


def test_round_trip_preserves_fields(buddy_dir: Path) -> None:
    _seed_companion(petCount=3)
    loaded = buddy_config.load_buddy_config()
    assert loaded is not None
    assert loaded["name"] == "Test"
    assert loaded["petCount"] == 3


def test_mutate_returns_none_when_unhatched(buddy_dir: Path) -> None:
    result = buddy_config.mutate_buddy_config(lambda current: current)
    assert result is None
    assert not (buddy_dir / "buddy.json").exists()


def test_mutate_noop_leaves_file_untouched(buddy_dir: Path) -> None:
    _seed_companion(petCount=5)
    mtime_before = (buddy_dir / "buddy.json").stat().st_mtime_ns
    time.sleep(0.01)
    result = buddy_config.mutate_buddy_config(lambda _current: None)
    assert result is not None
    assert result["petCount"] == 5
    assert (buddy_dir / "buddy.json").stat().st_mtime_ns == mtime_before


# ── threaded hammer ────────────────────────────────────────────────────────


def test_threaded_pet_count_no_lost_updates(buddy_dir: Path) -> None:
    """N threads each increment petCount M times; final must equal N*M."""
    _seed_companion(petCount=0)

    n_threads = 8
    increments_per_thread = 25

    def _bump_once(current):
        if current is None:
            return None
        merged = dict(current)
        merged["petCount"] = int(merged.get("petCount", 0)) + 1
        return merged

    barrier = threading.Barrier(n_threads)

    def worker() -> None:
        barrier.wait()
        for _ in range(increments_per_thread):
            buddy_config.mutate_buddy_config(_bump_once)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = buddy_config.load_buddy_config()
    assert final is not None
    assert final["petCount"] == n_threads * increments_per_thread


# ── multi-process hammer ───────────────────────────────────────────────────

_WORKER = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    config_path = Path(sys.argv[1])
    lock_path = Path(sys.argv[2])
    increments = int(sys.argv[3])

    from chrys.app.features.buddy import config as bc

    bc._config_path = lambda: config_path
    bc._lock_path = lambda: lock_path

    def _bump(current):
        if current is None:
            return None
        merged = dict(current)
        merged["petCount"] = int(merged.get("petCount", 0)) + 1
        return merged

    for _ in range(increments):
        bc.mutate_buddy_config(_bump)
    """
)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Cross-process subprocess lock contention is genuinely fragile under "
    "8-worker xdist on slow Windows runners (subprocess startup + msvcrt.locking "
    "latency exceeds the 10s lock timeout). The threaded test exercises the same "
    "FileLock path within a process; OS-level cross-process behaviour on Windows "
    "is unit-tested via the lock primitives directly elsewhere.",
)
def test_multiprocess_pet_count_no_lost_updates(buddy_dir: Path) -> None:
    """Real OS-level lock, not just a within-process threading.Lock."""
    _seed_companion(petCount=0)

    n_procs = 4
    increments_per_proc = 10
    config_path = buddy_dir / "buddy.json"
    lock_path = buddy_dir / "buddy.lock"

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WORKER, str(config_path), str(lock_path), str(increments_per_proc)],
        )
        for _ in range(n_procs)
    ]
    for p in procs:
        p.wait(timeout=30)
        assert p.returncode == 0, f"worker failed with code {p.returncode}"

    final = buddy_config.load_buddy_config()
    assert final is not None
    assert final["petCount"] == n_procs * increments_per_proc


# ── atomic write resilience ────────────────────────────────────────────────


def test_writes_are_atomic(buddy_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the write step crashes, the existing config is preserved."""
    _seed_companion(petCount=7)

    real_replace = buddy_config.os.replace

    def boom(_src, _dst) -> None:
        raise RuntimeError("simulated crash mid-replace")

    monkeypatch.setattr(buddy_config.os, "replace", boom)

    with pytest.raises(RuntimeError):
        buddy_config.update_buddy_config(petCount=999)

    # Restore so subsequent loads work.
    monkeypatch.setattr(buddy_config.os, "replace", real_replace)

    final = buddy_config.load_buddy_config()
    assert final is not None
    assert final["petCount"] == 7  # unchanged

    # No leftover temp files in the directory.
    leftovers = [p for p in buddy_dir.iterdir() if p.name.startswith("buddy.json.") and p.suffix == ".tmp"]
    assert leftovers == []


def test_corrupt_json_returns_none(buddy_dir: Path) -> None:
    config_path = buddy_dir / "buddy.json"
    config_path.write_text("{ not valid json", encoding="utf-8")
    assert buddy_config.load_buddy_config() is None


# ── hatch idempotency ──────────────────────────────────────────────────────


def test_hatch_race_keeps_first_companion(buddy_dir: Path) -> None:
    """Concurrent hatch calls must not overwrite each other."""
    from chrys.app.features.buddy.companion import hatch_companion

    n_threads = 6
    barrier = threading.Barrier(n_threads)
    results = []

    def worker(idx: int) -> None:
        barrier.wait()
        results.append(hatch_companion(name=f"Pet{idx}", personality=f"persona{idx}"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stored = buddy_config.load_buddy_config()
    assert stored is not None
    # All callers see the same canonical name on disk.
    assert all(r.name == stored["name"] for r in results)
    # The on-disk record matches one of the proposed names (whichever won the race).
    assert stored["name"].startswith("Pet")


# ── muted flag preservation under concurrent updates ───────────────────────


def test_concurrent_mute_and_pet_preserves_both(buddy_dir: Path) -> None:
    """petCount increments and mute toggles must not clobber each other."""
    _seed_companion(petCount=0)

    def _bump(current):
        if current is None:
            return None
        merged = dict(current)
        merged["petCount"] = int(merged.get("petCount", 0)) + 1
        return merged

    def pet_worker() -> None:
        for _ in range(50):
            buddy_config.mutate_buddy_config(_bump)

    def mute_worker() -> None:
        for i in range(50):
            buddy_config.set_muted(i % 2 == 0)

    threads = [threading.Thread(target=pet_worker), threading.Thread(target=mute_worker)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    final = buddy_config.load_buddy_config()
    assert final is not None
    assert final["petCount"] == 50
    # name / personality preserved.
    assert final["name"] == "Test"
    assert final["personality"] == "tester"
    # muted flag must survive, whichever value won last.
    assert isinstance(buddy_config.is_muted(), bool)
