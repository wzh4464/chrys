# Copyright (c) 2026 Chrys. All rights reserved.

"""Deterministic companion generation using seeded PRNG."""

from __future__ import annotations

import contextlib
import functools
import getpass
import hashlib
import platform
from typing import TYPE_CHECKING

from chrys.app.features.buddy.types import (
    HATS,
    LEVELS_PER_EVOLUTION,
    MAX_LEVEL,
    RARITIES,
    RARITY_FLOOR,
    RARITY_WEIGHTS,
    STATS,
    Companion,
    CompanionBones,
    Eye,
    Hat,
    Rarity,
    Species,
    Stat,
    StoredCompanion,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# -----------------------------------------------------------------------------
# Salt and cache
# -----------------------------------------------------------------------------

SALT = "friend-2026-401"
"""Salt for userId hashing — change to invalidate all existing pets."""

_roll_cache: dict[str, Roll] = {}
"""Cache roll results by userId to avoid recomputing during animation frames."""


# -----------------------------------------------------------------------------
# PRNG — Mulberry32
# -----------------------------------------------------------------------------


def mulberry32(seed: int) -> Callable[[], float]:
    """Mulberry32 PRNG — fast, deterministic, 2^32 period.

    Returns a zero-argument function that yields floats in [0, 1).
    """
    state = seed & 0xFFFFFFFF

    def next_float() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = (t ^ (t >> 15)) * (t + 1) & 0xFFFFFFFF
        t = (t ^ (t >> 7)) * (t + 61) & 0xFFFFFFFF
        return (t ^ (t >> 14)) / 0xFFFFFFFF

    return next_float


def hash_string(s: str) -> int:
    """Hash a string to a 32-bit integer using SHA256."""
    h = hashlib.sha256(s.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


# -----------------------------------------------------------------------------
# Roll data structure
# -----------------------------------------------------------------------------


@functools.total_ordering
class Roll:
    """All randomized values for a single user."""

    __slots__ = ("eye", "hat", "rarity", "shiny", "species", "stats")

    def __init__(
        self,
        rarity: Rarity,
        species: Species,
        eye: Eye,
        hat: Hat,
        shiny: bool,
        stats: dict[Stat, int],
    ) -> None:
        self.rarity = rarity
        self.species = species
        self.eye = eye
        self.hat = hat
        self.shiny = shiny
        self.stats = stats

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Roll):
            return NotImplemented
        return (
            self.rarity == other.rarity
            and self.species == other.species
            and self.eye == other.eye
            and self.hat == other.hat
            and self.shiny == other.shiny
            and self.stats == other.stats
        )

    def __lt__(self, other: Roll) -> bool:
        if self.rarity != other.rarity:
            return self.rarity.value < other.rarity.value
        return self.species.value < other.species.value

    def __hash__(self) -> int:
        return hash((self.rarity, self.species))


# -----------------------------------------------------------------------------
# Core rolling logic
# -----------------------------------------------------------------------------


def weighted_choice(rng: Callable[[], float], weights: dict[Rarity, int]) -> Rarity:
    """Select a rarity based on weights using weighted random."""
    total = sum(weights.values())
    r = rng() * total
    cumulative = 0
    for rarity, weight in weights.items():
        cumulative += weight
        if r < cumulative:
            return rarity
    return RARITIES[-1]  # fallback to SSR


def roll_stats(rng: Callable[[], float], rarity: Rarity) -> dict[Stat, int]:
    """Generate 5 stats with one peak, one dump, rest scattered.

    Peak stat gets floor + 50 + [0-30]
    Dump stat gets floor - 10 + [0-15]
    Others get floor + [0-40]
    """
    floor = RARITY_FLOOR[rarity]
    stats: dict[Stat, int] = {}

    # Pick peak and dump (ensure different)
    stat_list = list(STATS)
    peak = stat_list[int(rng() * len(stat_list))]
    dump_choices = [s for s in stat_list if s != peak]
    dump = dump_choices[int(rng() * len(dump_choices))]

    for stat in STATS:
        if stat == peak:
            stats[stat] = min(100, floor + 50 + int(rng() * 31))
        elif stat == dump:
            stats[stat] = max(1, floor - 10 + int(rng() * 16))
        else:
            stats[stat] = floor + int(rng() * 41)

    return stats


def roll_from(rng: Callable[[], float]) -> Roll:
    """Generate all randomized values from a PRNG."""
    from chrys.app.features.buddy.types import SPECIES_BY_RARITY

    rarity = weighted_choice(rng, RARITY_WEIGHTS)

    # Pick species from the specific rarity pool
    pool = SPECIES_BY_RARITY[rarity]
    species = pool[int(rng() * len(pool))]

    eye = Eye.DOT  # default
    if rng() < 0.3:  # 30% chance for special eyes
        eye_choices = [e for e in Eye if e != Eye.DOT]
        eye = eye_choices[int(rng() * len(eye_choices))]
    hat = Hat.NONE  # default
    if rng() < 0.25:  # 25% chance for hat
        hat = HATS[int(rng() * len(HATS))]
    shiny = rng() < 0.01  # 1% shiny chance
    stats = roll_stats(rng, rarity)

    return Roll(rarity, species, eye, hat, shiny, stats)


def roll(userId: str) -> Roll:
    """Generate deterministic roll for a userId with caching."""
    if userId in _roll_cache:
        return _roll_cache[userId]

    key = userId + SALT
    seed = hash_string(key)
    rng = mulberry32(seed)
    result = roll_from(rng)

    _roll_cache[userId] = result
    return result


# -----------------------------------------------------------------------------
# User ID generation
# -----------------------------------------------------------------------------


def user_id() -> str:
    """Generate a stable user identifier from machine + username.

    Uses hostname + username, hashed for privacy and fixed length.
    """
    raw = f"{platform.node()}::{getpass.getuser()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# -----------------------------------------------------------------------------
# Companion retrieval
# -----------------------------------------------------------------------------


def get_bones() -> CompanionBones:
    """Get deterministic bones for current user."""
    uid = user_id()
    r = roll(uid)
    return CompanionBones(
        rarity=r.rarity,
        species=r.species,
        eye=r.eye,
        hat=r.hat,
        shiny=r.shiny,
        stats=r.stats,
    )


def get_total_messages() -> int:
    """Calculate total dialogue count (user messages) across all sessions."""

    from chrys.foundation.config.settings import resolve_sessions_dir

    sessions_dir = resolve_sessions_dir(create=False)
    if not sessions_dir.exists():
        return 0

    import json

    total = 0
    # Folder format: {session_id}/session.json
    for path in sessions_dir.glob("*/session.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                total += data.get("meta", {}).get("message_count", 0)
        except json.JSONDecodeError, OSError, KeyError:
            continue

    # Legacy flat file format: {session_id}.json
    for path in sessions_dir.glob("*.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                total += data.get("meta", {}).get("message_count", 0)
        except json.JSONDecodeError, OSError, KeyError:
            continue

    return total


def get_companion() -> Companion | None:
    """Get full companion by combining bones with stored soul.

    Returns None if no soul has been stored (user hasn't hatched yet).

    Evolution is multi-stage: each time the companion accumulates 99 raw
    levels (``1 + messages//10 + petCount//20``), it evolves to the next
    rarity tier and the display level resets to 1. SSR is the final tier
    and can reach level 100; all lower tiers cap at 99.
    """
    from chrys.app.features.buddy.config import load_buddy_config, update_buddy_config

    stored = load_buddy_config()
    if stored is None:
        return None

    bones = get_bones()
    total_messages = stored.get("totalMessageCount", 0)
    pet_count = stored.get("petCount", 0)

    # Migration: if totalMessageCount is missing (legacy config) but sessions
    # exist on disk, compute from sessions and persist so level survives
    # future session deletions.
    if total_messages == 0:
        from_disk = get_total_messages()
        if from_disk > 0:
            total_messages = from_disk
            with contextlib.suppress(TimeoutError):
                update_buddy_config(totalMessageCount=total_messages)

    # Raw cumulative level from all-time XP
    raw_level = 1 + (total_messages // 10) + (pet_count // 20)

    # How many evolutions have happened based on raw XP
    base_rarity_idx = RARITIES.index(bones.rarity)
    max_evolutions = len(RARITIES) - 1 - base_rarity_idx
    evolution_count = min(max_evolutions, (raw_level - 1) // LEVELS_PER_EVOLUTION)

    # Current rarity after evolution
    rarity = RARITIES[base_rarity_idx + evolution_count]
    is_evolved = evolution_count > 0

    # Display level within the current evolution tier
    level_in_tier = raw_level - evolution_count * LEVELS_PER_EVOLUTION
    level = min(MAX_LEVEL, level_in_tier) if rarity == Rarity.SSR else min(LEVELS_PER_EVOLUTION, level_in_tier)

    # Auto-persist evolution count so it survives across versions / future
    # formula changes. Write is best-effort — failure leaves the file stale
    # but the derived value stays correct for this call.
    stored_evo = stored.get("evolutionCount", 0)
    if evolution_count > stored_evo:
        with contextlib.suppress(TimeoutError):
            update_buddy_config(evolutionCount=evolution_count)

    # Recalculate stats if evolved (deterministic seed from name + species)
    stats = bones.stats
    if is_evolved:
        seed = hash_string(stored["name"] + bones.species.value + "evolve")
        rng = mulberry32(seed)
        stats = roll_stats(rng, rarity)

    # Visual name prefix
    display_name = stored["name"]
    if is_evolved:
        display_name = f"🌟 {display_name}" if rarity == Rarity.SSR else f"✨ {display_name}"

    return Companion(
        rarity=rarity,
        species=bones.species,
        eye=bones.eye,
        hat=bones.hat,
        shiny=bones.shiny or (is_evolved and rarity == Rarity.SSR),
        stats=stats,
        name=display_name,
        personality=stored["personality"],
        hatchedAt=stored["hatchedAt"],
        level=level,
        is_evolved=is_evolved,
        petCount=pet_count,
        evolution_count=evolution_count,
        total_message_count=total_messages,
    )


def hatch_companion(name: str, personality: str) -> Companion:
    """Hatch a new companion for the current user.

    Idempotent across processes: if another instance already hatched a
    companion, this returns the existing one rather than overwriting it.
    """
    import time

    from chrys.app.features.buddy.config import mutate_buddy_config

    existing_messages = get_total_messages()

    new_record: StoredCompanion = {
        "name": name,
        "personality": personality,
        "hatchedAt": int(time.time() * 1000),
        "petCount": 0,
        "evolutionCount": 0,
        "totalMessageCount": existing_messages,
    }

    def _hatch_or_keep(current: StoredCompanion | None) -> StoredCompanion | None:
        # Returning ``None`` from mutate_buddy_config means "leave file alone";
        # the existing record stays intact.
        if current is not None:
            return None
        return new_record

    # ``mutate_buddy_config`` returns the post-mutation soul — the existing
    # record if another process already hatched, or our new one if we won.
    soul = mutate_buddy_config(_hatch_or_keep)
    assert soul is not None  # _hatch_or_keep always yields a populated record

    bones = get_bones()
    total_messages = soul.get("totalMessageCount", existing_messages)
    level = min(LEVELS_PER_EVOLUTION, 1 + (total_messages // 10))
    return Companion.from_parts(bones, soul, level=level, evolution_count=0, total_message_count=total_messages)


# -----------------------------------------------------------------------------
# Cache invalidation
# -----------------------------------------------------------------------------


def clear_roll_cache() -> None:
    """Clear the roll cache (e.g., after SALT change)."""
    _roll_cache.clear()
