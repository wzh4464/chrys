# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for Buddy companion generation and multi-stage evolution logic."""

from unittest.mock import patch

from chrys.app.features.buddy.types import (
    MAX_LEVEL,
    STATS,
    CompanionBones,
    Eye,
    Hat,
    Rarity,
    Species,
)


def _make_bones(rarity=Rarity.N, species=Species.DUCK):
    return CompanionBones(rarity, species, Eye.DOT, Hat.NONE, False, dict.fromkeys(STATS, 10))


def _make_soul(name="Test", pet_count=0, evolution_count=0, total_message_count=0):
    return {
        "name": name,
        "personality": "P",
        "hatchedAt": 0,
        "petCount": pet_count,
        "evolutionCount": evolution_count,
        "totalMessageCount": total_message_count,
    }


# ── level within first tier (no evolution) ─────────────────────────────────


def test_level_within_first_tier():
    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.N)
    soul = _make_soul(pet_count=45, total_message_count=50)

    with (
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        # raw_level = 1 + 50//10 + 45//20 = 1 + 5 + 2 = 8
        assert comp.level == 8
        assert not comp.is_evolved
        assert comp.rarity == Rarity.N
        assert comp.evolution_count == 0


# ── first evolution (N → R) ────────────────────────────────────────────────


def test_first_evolution_N_to_R():
    """At raw_level=100 the companion evolves: N→R, level resets to 1."""
    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.N)
    # raw_level = 1 + 990//10 + 0//20 = 1 + 99 + 0 = 100
    # evolution_count = (100 - 1) // 99 = 1
    # level = 100 - 1*99 = 1
    soul = _make_soul(pet_count=0, evolution_count=0, total_message_count=990)

    with (
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.config.update_buddy_config"),  # suppress auto-write
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        assert comp.evolution_count == 1
        assert comp.rarity == Rarity.R
        assert comp.level == 1
        assert comp.is_evolved
        assert "✨" in comp.name


def test_level_99_is_max_before_first_evolution():
    """At raw_level=99 the companion is at max tier level but not evolved yet."""
    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.N)
    # raw_level = 1 + 980//10 + 0//20 = 1 + 98 + 0 = 99
    # evolution_count = (99 - 1) // 99 = 0
    soul = _make_soul(pet_count=0, evolution_count=0, total_message_count=980)

    with (
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        assert comp.level == 99
        assert not comp.is_evolved
        assert comp.rarity == Rarity.N
        assert comp.evolution_count == 0


# ── second evolution (R → SR) ──────────────────────────────────────────────


def test_second_evolution_R_to_SR():
    """At raw_level=199 the companion evolves again: R→SR, level resets to 1."""
    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.N)
    # raw_level = 1 + 0 + 3960//20 = 1 + 198 = 199
    # evolution_count = (199 - 1) // 99 = 2
    # level = 199 - 2*99 = 1
    soul = _make_soul(pet_count=3960, evolution_count=0)

    with (
        patch("chrys.app.features.buddy.companion.get_total_messages", return_value=0),
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.config.update_buddy_config"),
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        assert comp.evolution_count == 2
        assert comp.rarity == Rarity.SR
        assert comp.level == 1
        assert comp.is_evolved


def test_level_99_again_before_second_evolution():
    """At raw_level=198 the R-rarity companion is at level 99."""
    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.N)
    # raw_level = 1 + 0 + 3940//20 = 1 + 197 = 198
    # evolution_count = (198 - 1) // 99 = 1
    # level = 198 - 1*99 = 99
    soul = _make_soul(pet_count=3940, evolution_count=1)

    with (
        patch("chrys.app.features.buddy.companion.get_total_messages", return_value=0),
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        assert comp.level == 99
        assert comp.rarity == Rarity.R
        assert comp.evolution_count == 1


# ── third evolution (SR → SSR) ─────────────────────────────────────────────


def test_third_evolution_SR_to_SSR():
    """At raw_level=298 the companion evolves to SSR, level resets to 1."""
    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.N)
    # raw_level = 1 + 0 + 5940//20 = 1 + 297 = 298
    # evolution_count = (298 - 1) // 99 = 3
    # level = 298 - 3*99 = 1
    soul = _make_soul(pet_count=5940, evolution_count=0)

    with (
        patch("chrys.app.features.buddy.companion.get_total_messages", return_value=0),
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.config.update_buddy_config"),
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        assert comp.evolution_count == 3
        assert comp.rarity == Rarity.SSR
        assert comp.level == 1
        assert comp.is_evolved
        assert "🌟" in comp.name


# ── SSR max level ──────────────────────────────────────────────────────────


def test_ssr_reaches_level_100():
    """SSR is the only rarity that can reach level 100."""
    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.N)
    # raw_level = 1 + 0 + 7920//20 = 1 + 396 = 397
    # evolution_count = min(3, (397 - 1) // 99) = min(3, 4) = 3
    # level = 397 - 3*99 = 100
    soul = _make_soul(pet_count=7920, evolution_count=3)

    with (
        patch("chrys.app.features.buddy.companion.get_total_messages", return_value=0),
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        assert comp.rarity == Rarity.SSR
        assert comp.level == 100
        assert comp.evolution_count == 3


def test_ssr_cannot_exceed_level_100():
    """Level caps at 100 even with excessive XP."""
    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.N)
    # Massive pet count — level still capped at 100
    soul = _make_soul(pet_count=20000, evolution_count=3)

    with (
        patch("chrys.app.features.buddy.companion.get_total_messages", return_value=0),
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        assert comp.level == MAX_LEVEL
        assert comp.rarity == Rarity.SSR


# ── born-SSR companions ────────────────────────────────────────────────────


def test_born_ssr_no_evolution():
    """An SSR-born companion has evolution_count=0 and just levels normally."""
    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.SSR, Species.DRAGON)
    # raw_level = 1 + 0 + 1980//20 = 100
    # evolution_count = min(0, (100-1)//99) = 0 (SSR can't evolve)
    # level = min(100, 100) = 100
    soul = _make_soul(pet_count=1980, evolution_count=0)

    with (
        patch("chrys.app.features.buddy.companion.get_total_messages", return_value=0),
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        assert comp.rarity == Rarity.SSR
        assert comp.level == MAX_LEVEL
        assert not comp.is_evolved
        assert comp.evolution_count == 0
        # Born SSR — no evolution prefix
        assert "🌟" not in comp.name
        assert "✨" not in comp.name


# ── R-born companions (fewer evolutions available) ──────────────────────────


def test_R_born_evolves_to_SR_then_SSR():
    """An R-born companion can evolve twice (R→SR→SSR)."""
    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.R, Species.OWL)
    # raw_level = 1 + 0 + 3960//20 = 199
    # evolution_count = min(2, (199-1)//99) = min(2, 2) = 2
    # R → SR → SSR
    soul = _make_soul(pet_count=3960, evolution_count=0)

    with (
        patch("chrys.app.features.buddy.companion.get_total_messages", return_value=0),
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.config.update_buddy_config"),
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        assert comp.evolution_count == 2
        assert comp.rarity == Rarity.SSR
        assert comp.is_evolved


# ── auto-persist on detection ──────────────────────────────────────────────


def test_auto_persist_evolution_count():
    """When computed > stored, get_companion() writes back to config."""
    from unittest.mock import MagicMock

    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.N)
    # raw_level=100 → evolution_count should be 1
    # stored says 0 → should trigger auto-write
    soul = _make_soul(pet_count=0, evolution_count=0, total_message_count=990)

    mock_update = MagicMock()

    with (
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.config.update_buddy_config", mock_update),
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        assert comp.evolution_count == 1
        mock_update.assert_called_once_with(evolutionCount=1)


def test_no_auto_persist_when_counts_match():
    """When computed == stored, no write happens."""
    from unittest.mock import MagicMock

    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.N)
    soul = _make_soul(pet_count=0, evolution_count=1, total_message_count=990)

    mock_update = MagicMock()

    with (
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.config.update_buddy_config", mock_update),
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        assert comp.evolution_count == 1
        mock_update.assert_not_called()


# ── backward compatibility — stored evolutionCount missing ──────────────────


def test_missing_evolution_count_defaults_to_zero():
    """Old configs without evolutionCount or totalMessageCount field default to 0."""
    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.N)
    # Soul without evolutionCount or totalMessageCount keys
    soul = {"name": "Test", "personality": "P", "hatchedAt": 0, "petCount": 50}

    with (
        patch("chrys.app.features.buddy.companion.get_total_messages", return_value=0),
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        # totalMessageCount defaults to 0, petCount=50 → raw_level = 1 + 0 + 2 = 3
        assert comp.evolution_count == 0
        assert not comp.is_evolved
        assert comp.level == 3


# ── migration: totalMessageCount seeded from sessions on first load ─────────


def test_migration_totalMessageCount_from_sessions():
    """When totalMessageCount is missing/0 but sessions exist, migrate from disk."""
    from unittest.mock import MagicMock

    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.N)
    # Soul with totalMessageCount=0 but sessions on disk have 500 messages
    soul = _make_soul(pet_count=0, evolution_count=0, total_message_count=0)

    mock_update = MagicMock()

    with (
        patch("chrys.app.features.buddy.companion.get_total_messages", return_value=500),
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.config.update_buddy_config", mock_update),
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        # raw_level = 1 + 500//10 + 0//20 = 51
        assert comp.level == 51
        assert comp.total_message_count == 500
        # Should persist the migrated count
        mock_update.assert_called_once_with(totalMessageCount=500)


def test_migration_skipped_when_totalMessageCount_already_set():
    """When totalMessageCount is already set, no migration happens."""
    from unittest.mock import MagicMock

    from chrys.app.features.buddy.companion import get_companion

    bones = _make_bones(Rarity.N)
    soul = _make_soul(pet_count=0, evolution_count=0, total_message_count=300)

    mock_update = MagicMock()

    with (
        patch("chrys.app.features.buddy.config.load_buddy_config", return_value=soul),
        patch("chrys.app.features.buddy.config.update_buddy_config", mock_update),
        patch("chrys.app.features.buddy.companion.get_bones", return_value=bones),
    ):
        comp = get_companion()
        # raw_level = 1 + 300//10 + 0//20 = 31
        assert comp.level == 31
        assert comp.total_message_count == 300
        # get_total_messages is NOT called, and update_buddy_config is not
        # called for migration (it may be called for evolutionCount which is 0==0)
        for call in mock_update.call_args_list:
            assert "totalMessageCount" not in call.kwargs
