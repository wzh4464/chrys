# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for Buddy configuration persistence."""

import json
from unittest.mock import patch

from chrys.app.features.buddy.config import load_buddy_config, save_buddy_config, update_buddy_config


def test_config_save_load(tmp_path):
    with patch("chrys.app.features.buddy.config._config_path", return_value=tmp_path / "buddy.json"):
        # 1. Initial load should be None
        assert load_buddy_config() is None

        # 2. Save a config
        config = {
            "name": "Neko",
            "personality": "Lazy",
            "hatchedAt": 12345,
            "petCount": 100,
            "evolutionCount": 1,
            "totalMessageCount": 500,
        }
        save_buddy_config(config)

        # 3. Load it back
        loaded = load_buddy_config()
        assert loaded["name"] == "Neko"
        assert loaded["petCount"] == 100
        assert loaded["evolutionCount"] == 1
        assert loaded["totalMessageCount"] == 500
        assert "muted" not in loaded  # Soul only

        # 4. Update it
        update_buddy_config(petCount=101, name="Super Neko", evolutionCount=2)
        updated = load_buddy_config()
        assert updated["petCount"] == 101
        assert updated["name"] == "Super Neko"
        assert updated["evolutionCount"] == 2


def test_config_migration_petCount(tmp_path):
    # Old-format configs (plain JSON, missing petCount) are still accepted.
    path = tmp_path / "buddy.json"
    old_data = {"name": "Oldie", "personality": "Old", "hatchedAt": 555}
    path.write_text(json.dumps(old_data))

    with patch("chrys.app.features.buddy.config._config_path", return_value=path):
        loaded = load_buddy_config()
        assert loaded["name"] == "Oldie"
        assert loaded["petCount"] == 0  # Migrated
        assert loaded["totalMessageCount"] == 0  # Migrated


def test_config_migration_evolutionCount(tmp_path):
    # Old-format configs (missing evolutionCount) get default 0.
    path = tmp_path / "buddy.json"
    old_data = {"name": "Oldie", "personality": "Old", "hatchedAt": 555, "petCount": 50}
    path.write_text(json.dumps(old_data))

    with patch("chrys.app.features.buddy.config._config_path", return_value=path):
        loaded = load_buddy_config()
        assert loaded["evolutionCount"] == 0  # Migrated


def test_increment_message_count(tmp_path):
    from chrys.app.features.buddy.config import increment_message_count

    config = {
        "name": "Test",
        "personality": "P",
        "hatchedAt": 0,
        "petCount": 0,
        "evolutionCount": 0,
        "totalMessageCount": 10,
    }
    with patch("chrys.app.features.buddy.config._config_path", return_value=tmp_path / "buddy.json"):
        save_buddy_config(config)
        increment_message_count()
        loaded = load_buddy_config()
        assert loaded["totalMessageCount"] == 11


def test_increment_message_count_not_hatched(tmp_path):
    from chrys.app.features.buddy.config import increment_message_count

    with patch("chrys.app.features.buddy.config._config_path", return_value=tmp_path / "buddy.json"):
        result = increment_message_count()
        assert result is None


# ── encryption envelope ────────────────────────────────────────────────────


def test_file_is_encrypted_envelope(tmp_path):
    """On-disk format is the v1 envelope, not plaintext fields."""
    path = tmp_path / "buddy.json"
    with patch("chrys.app.features.buddy.config._config_path", return_value=path):
        save_buddy_config(
            {
                "name": "Test",
                "personality": "P",
                "hatchedAt": 0,
                "petCount": 5,
                "evolutionCount": 0,
                "totalMessageCount": 100,
            }
        )
        raw = path.read_text("utf-8")
        envelope = json.loads(raw)
        assert envelope["v"] == 1
        assert "data" in envelope
        assert "sig" in envelope
        # Plaintext fields must NOT be visible.
        assert "petCount" not in envelope
        assert "name" not in envelope


def test_tampered_data_rejected_and_restored_from_backup(tmp_path):
    """Corrupt the primary — reads transparently fall back to .bak; primary heals on next write."""
    from chrys.app.features.buddy.config import load_buddy_config, save_buddy_config

    path = tmp_path / "buddy.json"
    bak = tmp_path / "buddy.json.bak"
    with patch("chrys.app.features.buddy.config._config_path", return_value=path):
        save_buddy_config(
            {
                "name": "Test",
                "personality": "P",
                "hatchedAt": 0,
                "petCount": 5,
                "evolutionCount": 0,
                "totalMessageCount": 100,
            }
        )
        # Both files should exist and be valid.
        assert path.exists()
        assert bak.exists()
        garbage = "garbage"
        # Corrupt the primary file.
        path.write_text(garbage, "utf-8")

        # Reads transparently fall back to backup.
        loaded2 = load_buddy_config()
        assert loaded2 is not None
        assert loaded2["petCount"] == 5

        # Lock-free reads must NOT rewrite the primary — that would race with
        # concurrent writers. Primary stays corrupt until the next write.
        assert path.read_text("utf-8") == garbage

        # Next write heals the primary as a side effect of _atomic_write.
        update_buddy_config(petCount=6)
        loaded3 = load_buddy_config()
        assert loaded3 is not None
        assert loaded3["petCount"] == 6
        # Primary is now valid (envelope-formatted, not garbage).
        envelope = json.loads(path.read_text("utf-8"))
        assert envelope["v"] == 1


def test_tampered_signature_rejected_and_falls_back_to_backup(tmp_path):
    """Flip a byte in the signature — sig-mismatch path triggers backup fallback."""
    from chrys.app.features.buddy.config import load_buddy_config, save_buddy_config

    path = tmp_path / "buddy.json"
    with patch("chrys.app.features.buddy.config._config_path", return_value=path):
        save_buddy_config(
            {
                "name": "Test",
                "personality": "P",
                "hatchedAt": 0,
                "petCount": 5,
                "evolutionCount": 0,
                "totalMessageCount": 100,
            }
        )

        # Tamper: flip the first character of the signature.
        envelope = json.loads(path.read_text("utf-8"))
        original_sig = envelope["sig"]
        flipped = "f" if original_sig[0] != "f" else "0"
        envelope["sig"] = flipped + original_sig[1:]
        path.write_text(json.dumps(envelope), "utf-8")

        # _verify_hmac fails → primary rejected → backup wins.
        loaded = load_buddy_config()
        assert loaded is not None
        assert loaded["petCount"] == 5


def test_both_files_corrupted_returns_none(tmp_path):
    """When both primary and backup are destroyed, return None."""
    from chrys.app.features.buddy.config import load_buddy_config, save_buddy_config

    path = tmp_path / "buddy.json"
    bak = tmp_path / "buddy.json.bak"
    with patch("chrys.app.features.buddy.config._config_path", return_value=path):
        save_buddy_config(
            {
                "name": "Test",
                "personality": "P",
                "hatchedAt": 0,
                "petCount": 5,
                "evolutionCount": 0,
                "totalMessageCount": 100,
            }
        )
        path.write_text("garbage", "utf-8")
        bak.write_text("also garbage", "utf-8")

        assert load_buddy_config() is None


def test_legacy_format_auto_upgraded_on_write(tmp_path):
    """Old-format file is read correctly, then upgraded to v1 on next write."""
    path = tmp_path / "buddy.json"
    old_data = {"name": "Legacy", "personality": "Old", "hatchedAt": 0, "petCount": 10}
    path.write_text(json.dumps(old_data), "utf-8")

    with patch("chrys.app.features.buddy.config._config_path", return_value=path):
        # Read succeeds (legacy compat).
        loaded = load_buddy_config()
        assert loaded is not None
        assert loaded["petCount"] == 10

        # Write triggers upgrade.
        update_buddy_config(petCount=11)
        raw = path.read_text("utf-8")
        envelope = json.loads(raw)
        assert envelope["v"] == 1
        assert "petCount" not in envelope  # encrypted

        # Still loads correctly.
        loaded2 = load_buddy_config()
        assert loaded2 is not None
        assert loaded2["petCount"] == 11
