# Copyright (c) 2026 Chrys. All rights reserved.

"""ASCII sprite definitions and rendering for Buddy companions.

Each species has 3-4 animation frames with a species-specific mouth
(no universal >_ template). Row 0 is the hat slot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from chrys.app.features.buddy.types import (
    DEFAULT_IDLE_SEQUENCE,
    IDLE_SEQUENCES,
    Eye,
    Hat,
    Species,
)

if TYPE_CHECKING:
    from chrys.app.features.buddy.types import Companion, CompanionBones


def _normalize_body_widths(bodies: dict[Species, list[list[str]]]) -> dict[Species, list[list[str]]]:
    """Pad each species' frames to a stable row width."""
    normalized: dict[Species, list[list[str]]] = {}
    for species, frames in bodies.items():
        width = max(len(row) for frame in frames for row in frame)
        normalized[species] = [[row.ljust(width) for row in frame] for frame in frames]
    return normalized


# -----------------------------------------------------------------------------
# Sprite data — 28 species with species-specific mouths
# Each frame is 5 rows. Row 0 is hat slot (must be blank on all non-action frames).
# Mouth characters are baked into sprites (not replaced at render time).
# -----------------------------------------------------------------------------

BODIES: dict[Species, list[list[str]]] = _normalize_body_widths(
    {
        # === Birds ===
        # Duck — flat bill (__) + bob/quack
        Species.DUCK: [
            ["    __      ", "  <(··)___  ", "   ( __ )   ", "    `--'    ", "            "],
            ["    __      ", "  <(··)___  ", "   ( __ )~  ", "    `--'    ", "            "],
            ["    __      ", "  <(··°)__  ", "   ( __ )   ", "    `--'    ", "            "],
        ],
        # Goose — long curved neck + open-beak honk (distinct from Duck)
        Species.GOOSE: [
            ["     __     ", "  __/··)___ ", " (  >.°  )  ", "  `-----'   ", "            "],
            ["     __     ", "  __/··)___ ", " (  >.°  )~ ", "  `-----'   ", "            "],
            ["     __     ", "  __/··°)   ", " (  >.°° )  ", "  `-----'   ", "            "],
        ],
        # === Aquatic ===
        # Blob — slime with occasional 'o' mouth, shape-shifting
        Species.BLOB: [
            ["            ", "   .----.   ", "  / ·· \\    ", "  |  o |    ", "   '----'   "],
            ["            ", "  .------.  ", " /  ··  \\   ", " |   o  |   ", "  '------'  "],
            ["            ", "   .----.   ", "  / ·· \\    ", "  | o  |    ", "   '---'    "],
        ],
        # Cat — ω mouth + lick-paw / tail-flick
        Species.CAT: [
            ["   /\\_/\\    ", "  ( ·· )    ", "  /  ω \\    ", " /|    |\\   ", "  '----'    "],
            ["   /\\_/\\    ", "  ( ·· )↑   ", "  /  ω \\    ", " /|    |\\   ", "  '----'    "],
            ["   /\\_/\\    ", "  ( ·· )    ", "  /  ω \\~   ", " /|    |\\   ", "  '----'    "],
        ],
        # Dragon — fire-breathing ﹀~~ mouth, 4 frames with breath cycle
        Species.DRAGON: [
            ["   (\\_/)    ", "  ( ·· )    ", "  /( ﹀ )\\━━", "    /|\\     ", "   /_|_\\    "],
            ["   (\\_/)~   ", "  ( ·· )    ", "  /( ﹀ )\\━━", "    /|\\     ", "   /_|_\\    "],
            ["   (\\_/)    ", "  ( ·· )    ", "  /( ﹀~~)\\ ", "    /|\\     ", "   /_|_\\    "],
            ["   (\\_/)    ", "  ( ·· )    ", "  /( ﹀~~~)\\", "    /|\\     ", "   /_|_\\    "],
        ],
        # Octopus — siphon ~ among tentacles
        Species.OCTOPUS: [
            ["   .---.    ", "  ( ·· )    ", " (  ~ )~~~  ", "  `-----´   ", "            "],
            ["   .---.    ", "  ( ·· )    ", " (  ~ )~~ ~ ", "  `-----´   ", "            "],
            ["   .---.    ", "  ( ·· )    ", " (  ~ )~ ~~ ", "  `-----´   ", "            "],
        ],
        # Owl — ear tufts + facial disc + v beak + head rotation
        Species.OWL: [
            ["   /^-^\\    ", "  / ·· \\    ", "  |  v |    ", "  \\___/    ", "   / \\      "],
            ["   /^-^\\    ", "  / ··~\\    ", "  |  v |    ", "  \\___/    ", "   / \\      "],
            ["   /^-^\\    ", "  /··  \\~   ", "  |  v |    ", "  \\___/    ", "   / \\      "],
        ],
        # Penguin — > beak + tuxedo body + waddle
        Species.PENGUIN: [
            ["    ___     ", "   /··\\     ", "   |> |     ", "   |__|     ", "   /__\\     "],
            ["    ___     ", "   /··\\~    ", "   |> |     ", "   |__|     ", "   /__\\     "],
            ["    ___     ", "  ~/··\\     ", "   |> |     ", "   |__|     ", "   /__\\     "],
        ],
        # Turtle — v mouth + shell retreat
        Species.TURTLE: [
            ["   _____    ", "  /··  \\    ", " |  v  |    ", "  \\___/    ", "            "],
            ["   _____    ", "  /··  \\    ", " |  v  |    ", "  \\___/    ", "     .      "],
            ["   _____    ", "  /··  \\    ", " |  v  |~   ", "  \\___/    ", "            "],
        ],
        # Snail — spiral shell + v mouth + eye-stalk extension
        Species.SNAIL: [
            ["     __     ", "   @/··\\    ", "   | v |    ", "    \\_/     ", "            "],
            ["     __     ", "   @/··\\~   ", "   | v |    ", "    \\_/     ", "            "],
            ["     __     ", "   @/··\\    ", "   | v |~   ", "    \\_/     ", "            "],
        ],
        # Ghost — O surprised mouth + ≈≈≈ ethereal wavy bottom
        Species.GHOST: [
            ["   .---.    ", "  / ·· \\    ", "  |  O |    ", "   ≈≈≈≈     ", "            "],
            ["   .---.    ", "  / ·· \\≈   ", "  |  O |    ", "   ≈≈≈≈     ", "            "],
            ["   .-≈-.    ", "  / ·· \\    ", "  |  O |    ", "   ≈≈≈≈     ", "            "],
        ],
        # Axolotl — ≈≈≈ external gills + v smile
        Species.AXOLOTL: [
            ["  ≈≈≈  ≈≈≈ ", "   (··)    ", "  <  v  >~~", "    ---     ", "            "],
            ["  ≈≈ ≈≈ ≈≈ ", "   (··)    ", "  <  v  >~~", "    ---     ", "            "],
            ["  ≈≈≈  ≈≈≈ ", "   (··)    ", "  <  v  >~ ", "    ---     ", "            "],
        ],
        # === Mammals ===
        # Capybara — flat _ mouth, ultra-chill (barely moves)
        Species.CAPYBARA: [
            ["            ", "   ___      ", "  |··|__    ", "  | _|  )   ", "   -- --    "],
            ["            ", "   ___      ", "  |··|__    ", "  | _|  )   ", "   -- .-    "],
            ["            ", "   ___      ", "  |··|__    ", "  | _|  )~  ", "   -- --    "],
        ],
        # Cactus — no mouth, plant with bloom
        Species.CACTUS: [
            ["     |      ", "   .-+-.    ", "   |··|     ", "   |  |     ", "   '---'    "],
            ["     |*     ", "   .-+-.    ", "   |··|     ", "   |  |     ", "   '---'    "],
            ["     |      ", "  .--+--.   ", "  | ·· |    ", "  |    |    ", "  '----'    "],
        ],
        # Robot — [=] LED eyes + □ speaker mouth
        Species.ROBOT: [
            ["   .---.    ", "   |[=]|    ", "   | □ |    ", "   '---'    ", "    --      "],
            ["   .---.    ", "   |[=]|*   ", "   | □ |    ", "   '---'    ", "    --      "],
            ["   .---.    ", "   |[=]|    ", "   | □ |◷   ", "   '---'    ", "    --      "],
        ],
        # Rabbit — Y bunny nose + ear wiggle
        Species.RABBIT: [
            ["   |\\ /|    ", "   \\··/     ", "  (  Y )    ", "  /|   |\\   ", "   U   U    "],
            ["   \\| /|    ", "   \\··/     ", "  (  Y )    ", "  /|   |\\   ", "   U   U    "],
            ["   |\\ |/    ", "   \\··/     ", "  (  Y )~   ", "  /|   |\\   ", "   U   U    "],
        ],
        # Mushroom — o tiny mouth + cap bounce + spore
        Species.MUSHROOM: [
            ["   .---.    ", "  /·· ·\\    ", "  \\__o_/    ", "    | |     ", "    '-'     "],
            ["   .---.    ", "  /·· ·\\    ", "  \\__o_/    ", "    | |     ", "    '-' ·   "],
            ["   .-~-.    ", "  /·· ·\\    ", "  \\__o_/    ", "    | |     ", "    '-' ··  "],
        ],
        # Chonk — ω fat cat mouth (wider than normal cat)
        Species.CHONK: [
            ["   /\\_/\\    ", "  ( ··· )   ", " (  ω   )  ", "  \\_____/   ", "            "],
            ["   /\\_/\\    ", "  ( ··· )   ", " (  ω   )  ", "  \\_____/~  ", "            "],
            ["   /\\_/\\    ", "  ( ··· )   ", " (  ω   )~ ", "  \\_____/   ", "            "],
        ],
        # Fox — ﹀ sly smile + tail sweep
        Species.FOX: [
            ["   /^-^\\    ", "  ( ·· )    ", "  / ﹀ \\__~ ", " /|___|    ", "  /_/ \\_   "],
            ["   /^-^\\    ", "  ( ·· )    ", "  / ﹀ \\_~  ", " /|___|    ", "  /_/ \\_   "],
            ["   /^-^\\    ", "  ( ·· )    ", "  / ﹀ \\___~", " /|___|    ", "  /_/ \\_   "],
        ],
        # Frog — ﹀ wide mouth + throat sac + jump (4 frames)
        Species.FROG: [
            ["  (··)(··)  ", " (   ﹀   ) ", "  (____)   ", "   `--´     ", "            "],
            ["  (··)(··)  ", " (  ﹀oo  ) ", "  (____)   ", "   `--´     ", "            "],
            ["  (··)(··)  ", " (   ﹀   ) ", "  (____)   ", "   `--´     ", "            "],
            ["  (··)(··)  ", " (   ﹀   ) ", "  (____)   ", "   `--´     ", "    ··      "],
        ],
        # Bee — ~ proboscis + ║║║║ stripes + wing buzz
        Species.BEE: [
            ["    \\ /     ", "   (··)     ", "  <( ~ )=== ", "   (___)    ", "    / \\     "],
            ["     X      ", "   (··)     ", "  <( ~ )=== ", "   (___)    ", "    / \\     "],
            ["    / \\~    ", "   (··)     ", "  <( ~ )=== ", "   (___)    ", "    / \\     "],
        ],
        # Crab — ﹀ small mouth + claw snap
        Species.CRAB: [
            [" (V) (··) (V)", "  \\__ ﹀ __/ ", "     \\__/    ", "    /    \\   ", "             "],
            [" (v) (··) (V)", "  \\__ ﹀ __/ ", "     \\__/    ", "    /    \\   ", "             "],
            [" (V) (··) (v)", "  \\__ ﹀ __/ ", "     \\__/~   ", "    /    \\   ", "             "],
        ],
        # Bat — v tiny mouth + wing spread (hanging pose)
        Species.BAT: [
            ["   /--^--\\  ", "  / (··) \\  ", "  \\ (v ) /  ", "   \\    /   ", "    \\  /    "],
            ["   /--^--\\  ", "  / (··) \\~ ", "  \\ (v ) /  ", "   \\    /   ", "    \\  /    "],
            ["   /----\\   ", "  / (··) \\  ", "  \\ (v ) /  ", "   \\    /   ", "    \\  /    "],
        ],
        # Shark — ﹏ jagged teeth + continuous swim
        Species.SHARK: [
            ["      ^     ", "  ____|___  ", " <··  ﹏  ) ", "  ¯¯¯¯¯¯¯¯  ", "            "],
            ["      ^     ", "  ____|___  ", " <··  ﹏  )~", "  ¯¯¯¯¯¯¯¯  ", "            "],
            ["      ^     ", "  ____|___  ", " <··  ﹏﹏ ) ", "  ¯¯¯¯¯¯¯¯  ", "            "],
        ],
        # Llama — v smile + long neck
        Species.LLAMA: [
            ["  (··)      ", "  ( v )     ", "   ||___    ", "   |    |   ", "   |__|_|   "],
            ["  (··)      ", "  ( v )     ", "   ||___ ~  ", "   |    |   ", "   |__|_|   "],
            ["  (··)      ", "  ( ~ )     ", "   ||___    ", "   |    |   ", "   |__|_|   "],
        ],
        # Panda — round ears + eye patches + ﹀ gentle smile
        Species.PANDA: [
            ["   ○     ○  ", "  (·)___(·) ", "  (  ··   ) ", "  (  ﹀   ) ", "   ¯¯¯¯¯¯   "],
            ["   ○     ○  ", "  (·)___(·) ", "  (  ··   ) ", "  (  ﹀   )~", "   ¯¯¯¯¯¯   "],
            ["   ○     ○  ", "  (·)___(·) ", "  (  ··   ) ", "  (  ﹀   ) ", "   ¯¯¯¯¯¯   "],
        ],
        # Snake — V forked tongue + slither (4 frames)
        Species.SNAKE: [
            ["    (··)    ", "     V \\   ", "  \\____/    ", "  /         ", "  ¯¯        "],
            ["    (··)    ", "     V \\~  ", "  \\____/    ", "  /         ", "  ¯¯        "],
            ["    (··)    ", "     V \\   ", "  \\____/~   ", "  /         ", "  ¯¯        "],
            ["    (··)    ", "     VV\\   ", "  \\____/    ", "  /         ", "  ¯¯        "],
        ],
        # Alien — oversized dome head + ﹀ mysterious smile + antenna
        Species.ALIEN: [
            ["    .---.    ", "   / ··  \\   ", "  │  ﹀   │  ", "   \\     /   ", "    '---'    "],
            ["    .-*-.    ", "   / ··  \\   ", "  │  ﹀   │  ", "   \\     /   ", "    '---'    "],
            ["    .---.    ", "   / ··  \\   ", "  │  ﹀   │  ", "   \\     /   ", "    '---'    "],
        ],
    }
)

# -----------------------------------------------------------------------------
# Hat decorations (rendered on row 0, only if row 0 is blank)
# -----------------------------------------------------------------------------

HAT_LINES: dict[Hat, str] = {
    Hat.NONE: "            ",
    Hat.CROWN: "   \\^^^/    ",
    Hat.TOPHAT: "   [___]    ",
    Hat.PROPELLER: "    -+-     ",
    Hat.HALO: "    )))     ",
    Hat.WIZARD: "   /~~~\\    ",
    Hat.BEANIE: "   (___)    ",
    Hat.TINYDUCK: "   <(·)     ",
}

# -----------------------------------------------------------------------------
# Eye characters
# -----------------------------------------------------------------------------

EYE_CHARS: dict[Eye, str] = {
    Eye.DOT: "·",
    Eye.STAR: "✦",
    Eye.X: "×",
    Eye.BULLSEYE: "◉",
    Eye.AT: "@",
    Eye.DEGREE: "°",
}

# -----------------------------------------------------------------------------
# Face rendering for narrow mode
# -----------------------------------------------------------------------------

FACE_TEMPLATES: dict[Species, str] = {
    Species.DUCK: "({eye}·{eye})",
    Species.GOOSE: "({eye}°{eye})",
    Species.BLOB: "[{eye}o{eye}]",
    Species.CAT: "({eye}ω{eye})",
    Species.DRAGON: "({eye}﹀{eye})",
    Species.OCTOPUS: "({eye}~{eye})",
    Species.OWL: "^({eye}v{eye})^",
    Species.PENGUIN: "│{eye}>{eye}│",
    Species.TURTLE: "│{eye}v{eye}│",
    Species.SNAIL: "@({eye}v{eye})",
    Species.GHOST: "≈{eye}O{eye}≈",
    Species.AXOLOTL: "≈({eye}v{eye})≈",
    Species.CAPYBARA: "│{eye}_{eye}│",
    Species.CACTUS: "│{eye} {eye}│",
    Species.ROBOT: "[{eye}□{eye}]",
    Species.RABBIT: "│\\{eye}Y{eye}/",
    Species.MUSHROOM: "│{eye}o{eye}│",
    Species.CHONK: "({eye}ω{eye})",
    Species.FOX: "({eye}﹀{eye})",
    Species.FROG: "({eye}﹀{eye})",
    Species.BEE: "({eye}~{eye})",
    Species.CRAB: "V({eye}﹀{eye})V",
    Species.BAT: "/({eye}v{eye})\\",
    Species.SHARK: "^({eye}﹏{eye})",
    Species.LLAMA: "({eye}v{eye})",
    Species.PANDA: "○({eye}﹀{eye})○",
    Species.SNAKE: "({eye}V{eye})",
    Species.ALIEN: "({eye}﹀{eye})",
}

# -----------------------------------------------------------------------------
# Rendering functions
# -----------------------------------------------------------------------------


def get_idle_sequence(species: Species) -> list[int]:
    """Get the per-species idle animation sequence.

    Args:
        species: The companion species.

    Returns:
        Animation step list (positive = frame index, -1 = blink).
        Falls back to DEFAULT_IDLE_SEQUENCE if species has no custom sequence.
    """
    return IDLE_SEQUENCES.get(species, DEFAULT_IDLE_SEQUENCE)


def get_frame_count(species: Species) -> int:
    """Get the number of animation frames available for a species."""
    return len(BODIES[species])


def render_sprite(
    bones: CompanionBones | Companion,
    frame: int = 0,
    blink: bool = False,
) -> list[str]:
    """Render a complete sprite with hat and eye applied.

    Args:
        bones: Companion bones with species, eye, hat
        frame: Animation frame index
        blink: If True, replace eyes with '-' for blink effect

    Returns:
        List of 5 strings representing the sprite rows
    """
    species = bones.species
    hat = bones.hat
    eye = bones.eye

    if species not in BODIES:
        species = Species.DUCK  # fallback

    body_frames = BODIES[species]
    frame_idx = frame % len(body_frames)
    lines = list(body_frames[frame_idx])

    # Apply hat on row 0 if hat is not none and row 0 is blank
    if hat != Hat.NONE and not lines[0].strip():
        hat_line = HAT_LINES.get(hat, HAT_LINES[Hat.NONE])
        lines[0] = hat_line.ljust(len(lines[0]))

    # Apply eyes
    eye_char = EYE_CHARS.get(eye, "·")
    if blink:
        eye_char = "-"

    # Replace eye placeholders in the body
    for i, line in enumerate(lines):
        if "··" in line:
            lines[i] = line.replace("··", f"{eye_char}{eye_char}")
        elif "[=]" in line:
            # Robot LED eye bracket: [=] -> [{eye_char}]
            lines[i] = line.replace("[=]", f"[{eye_char}]")

    return lines


def render_face(companion: Companion) -> str:
    """Render a one-line face for narrow terminal mode."""
    species = companion.species
    eye = companion.eye
    template = FACE_TEMPLATES.get(species, "({eye}·{eye})")
    eye_char = EYE_CHARS.get(eye, "·")
    return template.format(eye=eye_char)


def get_idle_frame(species: Species, tick: int) -> tuple[int, bool]:
    """Get animation frame and blink state for a species from tick count.

    Uses the per-species IDLE_SEQUENCES. Falls back to DEFAULT_IDLE_SEQUENCE.

    Args:
        species: The companion species.
        tick: Animation tick counter.

    Returns:
        Tuple of (frame_index, should_blink)
    """
    seq = get_idle_sequence(species)
    seq_idx = tick % len(seq)
    step = seq[seq_idx]

    if step == -1:
        return 0, True  # blink on rest frame
    if step > 0:
        return step, False  # animation frame
    return 0, False  # rest frame
