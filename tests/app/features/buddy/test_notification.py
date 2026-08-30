# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for Buddy notification display text."""

from chrys.app.features.buddy.notification import _buddy_call_options, get_buddy_info
from chrys.app.features.buddy.types import STATS, Companion, Eye, Hat, Rarity, Species
from chrys.service.profiles.models.registry import ModelProfileRegistry
from chrys.service.profiles.models.schema import ModelProfile


def _make_companion(evolution_count=0):
    return Companion(
        rarity=Rarity.N,
        species=Species.DUCK,
        eye=Eye.DOT,
        hat=Hat.NONE,
        shiny=False,
        stats=dict.fromkeys(STATS, 10),
        name="Test",
        personality="P",
        hatchedAt=0,
        level=1,
        is_evolved=evolution_count > 0,
        petCount=0,
        evolution_count=evolution_count,
        total_message_count=0,
    )


def test_buddy_info_omits_zero_evolution_count():
    info = get_buddy_info(_make_companion(evolution_count=0))

    assert "Evolutions:" not in info


def test_buddy_info_shows_positive_evolution_count():
    info = get_buddy_info(_make_companion(evolution_count=2))

    assert "   Evolutions: 2x" in info


# -----------------------------------------------------------------------------
# Pet-call LLM options — CHRYS_PET_MODEL output-cap handling
# -----------------------------------------------------------------------------


def _registry_with(*profiles):
    registry = ModelProfileRegistry()
    for profile in profiles:
        registry.register(profile)
    return registry


def _main_profile(**overrides):
    defaults = {"id": "main", "name": "Main", "model_id": "big-model", "max_output_tokens": 64000}
    defaults.update(overrides)
    return ModelProfile(**defaults)


def test_pet_call_options_same_model_keeps_main_profile_cap():
    options = _buddy_call_options(_main_profile(), "big-model", _registry_with())

    assert options == {"model": "big-model", "max_tokens": 64000}


def test_pet_call_options_swapped_model_uses_pet_profile_cap():
    registry = _registry_with(ModelProfile(id="pet", name="Pet", model_id="small-model", max_output_tokens=8000))

    options = _buddy_call_options(_main_profile(), "small-model", registry)

    assert options == {"model": "small-model", "max_tokens": 8000}


def test_pet_call_options_swapped_model_without_profile_sends_no_cap():
    options = _buddy_call_options(_main_profile(), "small-model", _registry_with())

    assert options == {"model": "small-model"}


def test_pet_call_options_swapped_model_ambiguous_caps_sends_no_cap():
    registry = _registry_with(
        ModelProfile(id="pet-a", name="Pet A", model_id="small-model", max_output_tokens=8000),
        ModelProfile(id="pet-b", name="Pet B", model_id="small-model", max_output_tokens=16000),
    )

    options = _buddy_call_options(_main_profile(), "small-model", registry)

    assert options == {"model": "small-model"}


def test_pet_call_options_swapped_model_strips_native_cap_aliases():
    profile = _main_profile(chat_options='{"max_output_tokens": 32000, "temperature": 0.5}')

    options = _buddy_call_options(profile, "small-model", _registry_with())

    assert options == {"model": "small-model", "temperature": 0.5}


def test_pet_call_options_swapped_model_keeps_other_chat_options():
    profile = _main_profile(chat_options='{"temperature": 0.5}')
    registry = _registry_with(ModelProfile(id="pet", name="Pet", model_id="small-model", max_output_tokens=8000))

    options = _buddy_call_options(profile, "small-model", registry)

    assert options == {"model": "small-model", "temperature": 0.5, "max_tokens": 8000}
