# Copyright (c) 2026 Chrys. All rights reserved.

"""Global routing settings."""

from __future__ import annotations

import pytest

from chrys.foundation.config.settings import Settings
from chrys.foundation.config.settings_store import load_settings
from chrys.foundation.config.spec import ProjectMerge, specs_by_field


def test_routing_defaults() -> None:
    loaded = load_settings(env={})

    assert loaded.settings.routing_mode == "auto"
    assert loaded.settings.routing_tiebreaker_model_profile == ""


@pytest.mark.parametrize("mode", ["off", "auto", "always"])
def test_routing_mode_env_override(mode: str) -> None:
    loaded = load_settings(env={"CHRYS_ROUTING_MODE": mode})

    assert loaded.settings.routing_mode == mode


def test_an_invalid_routing_mode_falls_back_to_the_default() -> None:
    loaded = load_settings(env={"CHRYS_ROUTING_MODE": "sometimes"})

    assert loaded.settings.routing_mode == "auto"


def test_tiebreaker_model_profile_env_override() -> None:
    loaded = load_settings(env={"CHRYS_ROUTING_TIEBREAKER_MODEL_PROFILE": "cheap-model"})

    assert loaded.settings.routing_tiebreaker_model_profile == "cheap-model"


def test_neither_routing_key_is_project_settable() -> None:
    """`always` commits the machine to a campaign per turn, and the tiebreaker
    names a model: a repository must be able to set neither."""
    specs = specs_by_field(Settings)

    assert specs["routing_mode"].key == "routing.mode"
    assert specs["routing_mode"].project_merge is ProjectMerge.DENY
    assert specs["routing_tiebreaker_model_profile"].key == "routing.tiebreaker_model_profile"
    assert specs["routing_tiebreaker_model_profile"].project_merge is ProjectMerge.DENY
