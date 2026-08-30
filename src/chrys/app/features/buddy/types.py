# Copyright (c) 2026 Chrys. All rights reserved.

"""Type definitions for the Buddy companion system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypedDict

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class Rarity(StrEnum):
    """Pet rarity tiers: N, R, SR, SSR."""

    N = "N"
    R = "R"
    SR = "SR"
    SSR = "SSR"


class Species(StrEnum):
    """28 different pet species."""

    DUCK = "duck"
    GOOSE = "goose"
    BLOB = "blob"
    CAT = "cat"
    DRAGON = "dragon"
    OCTOPUS = "octopus"
    OWL = "owl"
    PENGUIN = "penguin"
    TURTLE = "turtle"
    SNAIL = "snail"
    GHOST = "ghost"
    AXOLOTL = "axolotl"
    CAPYBARA = "capybara"
    CACTUS = "cactus"
    ROBOT = "robot"
    RABBIT = "rabbit"
    MUSHROOM = "mushroom"
    CHONK = "chonk"
    FOX = "fox"
    FROG = "frog"
    BEE = "bee"
    CRAB = "crab"
    BAT = "bat"
    SHARK = "shark"
    LLAMA = "llama"
    PANDA = "panda"
    SNAKE = "snake"
    ALIEN = "alien"


class Eye(StrEnum):
    """6 eye styles."""

    DOT = "dot"  # ·
    STAR = "star"  # ✦
    X = "x"  # ×
    BULLSEYE = "bullseye"  # ◉
    AT = "at"  # @
    DEGREE = "degree"  # °


class Hat(StrEnum):
    """8 hat decorations."""

    NONE = "none"
    CROWN = "crown"
    TOPHAT = "tophat"
    PROPELLER = "propeller"
    HALO = "halo"
    WIZARD = "wizard"
    BEANIE = "beanie"
    TINYDUCK = "tinyduck"


class Stat(StrEnum):
    """5 pet attributes."""

    DEBUGGING = "debugging"
    PATIENCE = "patience"
    CHAOS = "chaos"
    WISDOM = "wisdom"
    SNARK = "snark"


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

RARITIES = [
    Rarity.N,
    Rarity.R,
    Rarity.SR,
    Rarity.SSR,
]

RARITY_WEIGHTS: dict[Rarity, int] = {
    Rarity.N: 60,
    Rarity.R: 25,
    Rarity.SR: 10,
    Rarity.SSR: 5,
}

RARITY_FLOOR: dict[Rarity, int] = {
    Rarity.N: 5,
    Rarity.R: 20,
    Rarity.SR: 40,
    Rarity.SSR: 65,
}

RARITY_COLORS: dict[Rarity, str] = {
    Rarity.N: "#808080",  # Gray
    Rarity.R: "#00ff00",  # Green
    Rarity.SR: "#bf00ff",  # Purple
    Rarity.SSR: "#ffa500",  # Orange/Gold
}

# Species assignments by rarity
SPECIES_BY_RARITY: dict[Rarity, list[Species]] = {
    Rarity.N: [
        Species.DUCK,
        Species.GOOSE,
        Species.BLOB,
        Species.FROG,
        Species.BEE,
        Species.MUSHROOM,
        Species.CACTUS,
        Species.RABBIT,
    ],
    Rarity.R: [
        Species.SNAIL,
        Species.TURTLE,
        Species.OCTOPUS,
        Species.OWL,
        Species.PENGUIN,
        Species.SNAKE,
        Species.AXOLOTL,
    ],
    Rarity.SR: [
        Species.GHOST,
        Species.ROBOT,
        Species.CHONK,
        Species.SHARK,
        Species.LLAMA,
        Species.FOX,
        Species.CRAB,
        Species.BAT,
    ],
    Rarity.SSR: [
        Species.DRAGON,
        Species.PANDA,
        Species.CAT,
        Species.CAPYBARA,
        Species.ALIEN,
    ],
}

SPECIES = list(Species)
EYES = [e for e in Eye if e != Eye.DOT]  # dot is default, others are special
HATS = [h for h in Hat if h != Hat.NONE]
STATS = list(Stat)

SPRITE_HEIGHT = 5
SPRITE_WIDTH = 12

LEVELS_PER_EVOLUTION = 99
"""Number of raw levels consumed per evolution tier (level 1 → 99)."""

MAX_LEVEL = 100
"""Absolute maximum display level, only reachable at SSR rarity."""

DEFAULT_IDLE_SEQUENCE = [0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 2, 0, 0, 0]
"""Fallback 15-step idle animation: 0=rest, 1/2=jiggle, -1=blink on frame 0"""

IDLE_SEQUENCES: dict[Species, list[int]] = {
    # === Birds ===
    Species.DUCK: [0, 0, 1, 0, 0, 2, 0, 0, -1, 0, 0, 1, 0, 0, 0],
    Species.GOOSE: [0, 0, 1, 0, 0, 2, 0, -1, 0, 0, 1, 0, 0, 0, 0],
    Species.OWL: [0, 0, 0, 1, 0, 0, 2, 0, -1, 0, 0, 1, 0, 0, 0],
    Species.PENGUIN: [0, 0, 1, 0, 2, 0, -1, 0, 1, 0, 0, 2, 0, 0, 0],
    # === Aquatic ===
    Species.SHARK: [0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, -1, 1, 0],
    Species.OCTOPUS: [0, 1, 0, 2, 0, 1, 0, -1, 0, 2, 0, 1, 0, 0, 0],
    Species.CRAB: [0, 1, 0, 2, 0, -1, 0, 1, 0, 2, 0, 0, 0, 0, 0],
    Species.AXOLOTL: [0, 0, 1, 0, 2, 0, -1, 0, 1, 0, 0, 2, 0, 0, 0],
    Species.TURTLE: [0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 2, 0, 0, 0, 0],
    Species.FROG: [0, 0, 1, 0, 2, 0, -1, 0, 0, 3, 0, 0, 1, 0, 0],
    # === Mammals ===
    Species.CAT: [0, 0, 0, 1, 0, -1, 0, 0, 2, 0, 0, -1, 0, 0, 0],
    Species.CHONK: [0, 0, 0, 1, 0, -1, 0, 0, 2, 0, 0, 0, 0, -1, 0],
    Species.FOX: [0, 0, 1, 0, 0, 2, 0, -1, 0, 1, 0, 0, 2, 0, 0],
    Species.RABBIT: [0, 0, 1, 0, 2, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0],
    Species.CAPYBARA: [0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 2, 0, 0],
    Species.PANDA: [0, 0, 1, 0, 0, -1, 0, 0, 2, 0, 0, 0, 0, -1, 0],
    Species.LLAMA: [0, 0, 0, 1, 0, 0, -1, 0, 0, 2, 0, 0, 0, 0, 0],
    # === Fantasy ===
    Species.DRAGON: [0, 1, 0, 2, 0, 1, 0, -1, 0, 3, 0, 1, 0, 0, 0],
    Species.GHOST: [0, 0, 1, 0, 0, 2, 0, -1, 0, 1, 0, 0, 2, 0, 0],
    Species.ALIEN: [0, 1, 0, 2, 0, -1, 0, 0, 1, 0, 0, 2, 0, 0, 0],
    # === Insects/Arthropods ===
    Species.BEE: [0, 1, 0, 1, 2, 1, 0, 1, -1, 1, 0, 1, 2, 1, 0],
    Species.SNAIL: [0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 2],
    # === Plants/Fungi ===
    Species.CACTUS: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 2],
    Species.MUSHROOM: [0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, -1, 0, 0, 0],
    # === Other ===
    Species.ROBOT: [0, 0, 1, 0, 2, 0, -1, 0, 0, 1, 0, 0, 2, 0, 0],
    Species.SNAKE: [0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 3, 1, 0, -1, 0],
    Species.BAT: [0, 0, 1, 0, 2, 0, -1, 0, 0, 1, 0, 2, 0, 0, 0],
    Species.BLOB: [0, 0, 1, 0, 0, 2, 0, 0, -1, 0, 1, 0, 0, 2, 0],
}
"""Per-species idle animation sequences. Positive = frame index, -1 = blink on default frame."""

BUBBLE_SHOW_TICKS = 20  # ~10 seconds at 500ms/tick
FADE_WINDOW_TICKS = 6  # last 3 seconds fade

TICK_MS = 500
PET_BURST_MS = 2500


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CompanionBones:
    """Deterministically generated pet attributes (not persisted)."""

    rarity: Rarity
    species: Species
    eye: Eye
    hat: Hat
    shiny: bool
    stats: dict[Stat, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure stats dict uses Stat enum keys
        if self.stats and not isinstance(next(iter(self.stats.keys())), Stat):
            object.__setattr__(
                self,
                "stats",
                {Stat(k) if isinstance(k, str) else k: v for k, v in self.stats.items()},
            )


@dataclass
class CompanionSoul:
    """Persisted pet attributes (name, personality)."""

    name: str
    personality: str


class StoredCompanion(TypedDict):
    """JSON-serializable stored companion (Soul only)."""

    name: str
    personality: str
    hatchedAt: int
    petCount: int
    evolutionCount: int
    totalMessageCount: int


@dataclass
class Companion:
    """Full companion object (bones + soul + metadata)."""

    rarity: Rarity
    species: Species
    eye: Eye
    hat: Hat
    shiny: bool
    stats: dict[Stat, int]
    name: str
    personality: str
    hatchedAt: int
    level: int = 1
    is_evolved: bool = False
    petCount: int = 0
    evolution_count: int = 0
    total_message_count: int = 0

    @classmethod
    def from_parts(
        cls,
        bones: CompanionBones,
        soul: StoredCompanion,
        level: int = 1,
        is_evolved: bool = False,
        evolution_count: int = 0,
        total_message_count: int = 0,
    ) -> Companion:
        """Combine bones and stored soul into full Companion."""
        return cls(
            rarity=bones.rarity,
            species=bones.species,
            eye=bones.eye,
            hat=bones.hat,
            shiny=bones.shiny,
            stats=bones.stats,
            name=soul["name"],
            personality=soul["personality"],
            hatchedAt=soul["hatchedAt"],
            level=level,
            is_evolved=is_evolved,
            petCount=soul.get("petCount", 0),
            evolution_count=evolution_count,
            total_message_count=total_message_count,
        )

    def to_stored(self) -> StoredCompanion:
        """Extract Soul portion for persistence."""
        return StoredCompanion(
            name=self.name,
            personality=self.personality,
            hatchedAt=self.hatchedAt,
            petCount=self.petCount,
            evolutionCount=self.evolution_count,
            totalMessageCount=self.total_message_count,
        )
