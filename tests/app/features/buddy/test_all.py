# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for Buddy companion message counting."""

import json
from unittest.mock import patch

from chrys.app.features.buddy.companion import get_total_messages
from chrys.foundation.config.settings import SESSION_ROOT_DIR_ENV_VAR


def test_total_messages_counting(tmp_path, monkeypatch):
    # Setup mock session directory
    monkeypatch.delenv(SESSION_ROOT_DIR_ENV_VAR, raising=False)
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # 1. Session in folder format
    s1 = sessions_dir / "session1"
    s1.mkdir()
    (s1 / "session.json").write_text(json.dumps({"meta": {"message_count": 15}}))

    # 2. Another session in folder format
    s2 = sessions_dir / "session2"
    s2.mkdir()
    (s2 / "session.json").write_text(json.dumps({"meta": {"message_count": 25}}))

    # 3. Legacy session in flat file format
    (sessions_dir / "session3.json").write_text(json.dumps({"meta": {"message_count": 10}}))

    # 4. Corrupt session (should be ignored)
    (sessions_dir / "corrupt.json").write_text("{broken")

    with patch("chrys.foundation.platform.get_platform") as mock_platform:
        mock_platform.return_value.config_dir = tmp_path

        count = get_total_messages()
        # Total = 15 + 25 + 10 = 50
        assert count == 50


def test_total_messages_counting_uses_session_root_env(tmp_path, monkeypatch):
    custom_root = tmp_path / "custom-root"
    session_dir = custom_root / "sessions" / "session1"
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text(json.dumps({"meta": {"message_count": 7}}))
    monkeypatch.setenv(SESSION_ROOT_DIR_ENV_VAR, str(custom_root))

    assert get_total_messages() == 7
