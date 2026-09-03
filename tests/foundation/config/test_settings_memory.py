# Copyright (c) 2026 Chrys. All rights reserved.

"""Settings that govern the ContextGraph memory MCP and its idle writeback."""

from __future__ import annotations

from chrys.foundation.config.settings import Settings
from chrys.foundation.config.settings_store import load_settings
from chrys.foundation.config.spec import specs_by_field


def test_memory_settings_defaults() -> None:
    loaded = load_settings(env={})

    assert loaded.settings.memory_mcp_enabled is True
    assert loaded.settings.memory_writeback_idle_seconds == 3600
    assert loaded.settings.memory_writeback_on_session_end is True


def test_memory_settings_env_override() -> None:
    loaded = load_settings(
        env={
            "CHRYS_MEMORY_MCP": "0",
            "CHRYS_MEMORY_WRITEBACK_IDLE_SECONDS": "60",
            "CHRYS_MEMORY_WRITEBACK_ON_END": "0",
        }
    )

    assert loaded.settings.memory_mcp_enabled is False
    assert loaded.settings.memory_writeback_idle_seconds == 60
    assert loaded.settings.memory_writeback_on_session_end is False


def test_zero_idle_seconds_is_accepted_as_the_disable_switch() -> None:
    loaded = load_settings(env={"CHRYS_MEMORY_WRITEBACK_IDLE_SECONDS": "0"})

    assert loaded.settings.memory_writeback_idle_seconds == 0


def test_negative_idle_seconds_falls_back_to_the_default() -> None:
    loaded = load_settings(env={"CHRYS_MEMORY_WRITEBACK_IDLE_SECONDS": "-1"})

    assert loaded.settings.memory_writeback_idle_seconds == 3600


def test_memory_keys_are_project_settable_but_not_the_mcp_switch() -> None:
    specs = specs_by_field(Settings)

    assert specs["memory_mcp_enabled"].key == "memory.mcp.enabled"
    assert specs["memory_writeback_idle_seconds"].key == "memory.writeback.idle_seconds"
    assert specs["memory_writeback_on_session_end"].key == "memory.writeback.on_session_end"
    assert specs["memory_mcp_enabled"].group == "memory"
