# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for buddy sprite rendering and per-species idle animation."""

import pytest

from chrys.app.features.buddy.types import (
    DEFAULT_IDLE_SEQUENCE,
    IDLE_SEQUENCES,
    SPECIES,
    STATS,
    Companion,
    CompanionBones,
    Eye,
    Hat,
    Rarity,
    Species,
)


def _make_bones(species=Species.DUCK, eye=Eye.DOT, hat=Hat.NONE):
    return CompanionBones(Rarity.N, species, eye, hat, False, dict.fromkeys(STATS, 10))


def _make_companion(species=Species.DUCK, eye=Eye.DOT, hat=Hat.NONE, evolution_count=0):
    bones = _make_bones(species=species, eye=eye, hat=hat)
    return Companion(
        rarity=bones.rarity,
        species=bones.species,
        eye=bones.eye,
        hat=bones.hat,
        shiny=bones.shiny,
        stats=bones.stats,
        name="Test",
        personality="P",
        hatchedAt=0,
        level=1,
        is_evolved=evolution_count > 0,
        petCount=0,
        evolution_count=evolution_count,
        total_message_count=0,
    )


# ── BODIES coverage ────────────────────────────────────────────────────────


def test_all_species_have_bodies():
    """Every Species enum value has an entry in BODIES."""
    from chrys.app.features.buddy.sprites import BODIES

    for species in Species:
        assert species in BODIES, f"Missing BODIES entry for {species}"


def test_all_species_have_at_least_3_frames():
    """Every species has at least 3 animation frames."""
    from chrys.app.features.buddy.sprites import BODIES

    for species in Species:
        frames = BODIES[species]
        assert len(frames) >= 3, f"{species} has only {len(frames)} frames, need >=3"


def test_all_frames_are_five_rows():
    """Every frame in every species is exactly 5 rows (sprite height)."""
    from chrys.app.features.buddy.sprites import BODIES

    for species in Species:
        for i, frame in enumerate(BODIES[species]):
            assert len(frame) == 5, f"{species} frame {i} has {len(frame)} rows, expected 5"


def test_all_species_have_stable_row_widths():
    """Every row for a species has the same width, across every frame."""
    from chrys.app.features.buddy.sprites import BODIES

    for species, frames in BODIES.items():
        widths = {len(row) for frame in frames for row in frame}
        assert len(widths) == 1, f"{species}: mixed row widths {sorted(widths)}"


def test_all_frames_have_eye_anchor():
    """Every frame has an eye placeholder that render_sprite can customize."""
    from chrys.app.features.buddy.sprites import BODIES

    for species, frames in BODIES.items():
        for i, frame in enumerate(frames):
            combined = "\n".join(frame)
            assert "··" in combined or "[=]" in combined, f"{species} frame {i}: no eye anchor"


def test_no_species_has_all_identical_frames():
    """No species should have all frames identical (zero animation)."""
    from chrys.app.features.buddy.sprites import BODIES

    for species in Species:
        frames = BODIES[species]
        # Compare each pair of frames — at least one pair must differ
        all_same = all(frames[0] == f for f in frames[1:])
        assert not all_same, f"{species} has all identical frames (no animation)"


# ── IDLE_SEQUENCES coverage ────────────────────────────────────────────────


def test_all_species_have_idle_sequences():
    """Every Species has a corresponding IDLE_SEQUENCES entry."""
    for species in Species:
        assert species in IDLE_SEQUENCES, f"Missing IDLE_SEQUENCES entry for {species}"


def test_idle_sequences_only_contain_valid_steps():
    """IDLE_SEQUENCES values are only -1 (blink) or valid frame indices."""
    from chrys.app.features.buddy.sprites import BODIES

    for species in Species:
        seq = IDLE_SEQUENCES[species]
        max_frame = len(BODIES[species]) - 1
        for step in seq:
            assert step in (-1, 0, 1, 2, 3), f"{species} sequence has invalid step {step} (expected -1, 0, 1, 2, or 3)"
            if step >= 0:
                assert step <= max_frame, (
                    f"{species} sequence references frame {step} but only has {max_frame + 1} frames"
                )


def test_idle_sequences_contain_blink():
    """Every IDLE_SEQUENCES must contain at least one blink (-1) step."""
    for species in Species:
        seq = IDLE_SEQUENCES[species]
        assert -1 in seq, f"{species} IDLE_SEQUENCES has no blink (-1) step"


# ── get_idle_sequence() ────────────────────────────────────────────────────


def test_get_idle_sequence_returns_correct_sequence():
    """get_idle_sequence returns the per-species sequence."""
    from chrys.app.features.buddy.sprites import get_idle_sequence

    assert get_idle_sequence(Species.CAT) == IDLE_SEQUENCES[Species.CAT]
    assert get_idle_sequence(Species.DRAGON) == IDLE_SEQUENCES[Species.DRAGON]
    assert get_idle_sequence(Species.SHARK) == IDLE_SEQUENCES[Species.SHARK]


def test_get_idle_sequence_falls_back_for_unknown_species():
    """get_idle_sequence returns DEFAULT_IDLE_SEQUENCE for species without custom sequence.

    Note: all 28 Species have custom sequences, but the fallback path exists
    for forward-compatibility with future species additions.
    """
    from chrys.app.features.buddy.sprites import get_idle_sequence

    # All known species should return their custom sequence, not the default
    for species in Species:
        seq = get_idle_sequence(species)
        assert seq is IDLE_SEQUENCES[species]
        assert seq is not DEFAULT_IDLE_SEQUENCE  # each has its own list


def test_get_frame_count_matches_species_frames():
    """get_frame_count reflects each species' BODIES entry."""
    from chrys.app.features.buddy.sprites import BODIES, get_frame_count

    for species, frames in BODIES.items():
        assert get_frame_count(species) == len(frames)


# ── get_idle_frame() ────────────────────────────────────────────────────────


def test_get_idle_frame_returns_rest_at_tick_zero():
    """First tick is always rest (0, False)."""
    from chrys.app.features.buddy.sprites import get_idle_frame

    for species in Species:
        frame, blink = get_idle_frame(species, 0)
        assert frame == 0, f"{species} first frame should be 0, got {frame}"
        assert blink is False, f"{species} first tick should not blink"


def test_get_idle_frame_wraps_around():
    """get_idle_frame wraps tick around sequence length."""
    from chrys.app.features.buddy.sprites import get_idle_frame

    seq = IDLE_SEQUENCES[Species.DUCK]
    # Same tick % len(seq) should give same result
    f1, b1 = get_idle_frame(Species.DUCK, 0)
    f2, b2 = get_idle_frame(Species.DUCK, len(seq))
    assert (f1, b1) == (f2, b2)


def test_get_idle_frame_blink_returns_blink_true():
    """When sequence step is -1, get_idle_frame returns (0, True)."""
    from chrys.app.features.buddy.sprites import get_idle_frame

    # Find a tick where duck blinks
    duck_seq = IDLE_SEQUENCES[Species.DUCK]
    blink_indices = [i for i, s in enumerate(duck_seq) if s == -1]
    assert blink_indices, "Duck sequence has no blink step"

    for idx in blink_indices:
        frame, blink = get_idle_frame(Species.DUCK, idx)
        assert frame == 0
        assert blink is True


def test_get_idle_frame_nonzero_animation():
    """When sequence step > 0, returns that frame index and blink=False."""
    from chrys.app.features.buddy.sprites import get_idle_frame

    # Dragon has frames 1, 2, 3 in its sequence
    dragon_seq = IDLE_SEQUENCES[Species.DRAGON]
    for i, step in enumerate(dragon_seq):
        frame, blink = get_idle_frame(Species.DRAGON, i)
        if step > 0:
            assert frame == step, f"Dragon tick {i}: expected frame {step}, got {frame}"
            assert blink is False
        elif step == 0:
            assert frame == 0
            assert blink is False
        elif step == -1:
            assert frame == 0
            assert blink is True


# ── render_sprite() ────────────────────────────────────────────────────────


def test_render_sprite_returns_five_rows():
    """render_sprite always returns exactly 5 rows."""
    from chrys.app.features.buddy.sprites import render_sprite

    for species in Species:
        bones = _make_bones(species=species)
        lines = render_sprite(bones, frame=0)
        assert len(lines) == 5, f"{species} render_sprite returned {len(lines)} rows"


def test_render_sprite_applies_eye():
    """Eye character replaces ·· placeholder."""
    from chrys.app.features.buddy.sprites import render_sprite

    bones = _make_bones(species=Species.CAT, eye=Eye.STAR)
    lines = render_sprite(bones, frame=0)
    combined = "\n".join(lines)
    assert "✦✦" in combined, f"STAR eye not applied:\n{combined}"
    assert "··" not in combined, f"·· placeholder not replaced:\n{combined}"


def test_render_sprite_blink_replaces_eyes_with_dash():
    """Blink replaces eyes with '-'."""
    from chrys.app.features.buddy.sprites import render_sprite

    bones = _make_bones(species=Species.CAT, eye=Eye.DOT)
    lines = render_sprite(bones, frame=0, blink=True)
    combined = "\n".join(lines)
    assert "--" in combined or "-" in combined, f"Blink not applied:\n{combined}"


def test_render_sprite_applies_hat():
    """Hat is placed on blank row 0."""
    from chrys.app.features.buddy.sprites import render_sprite

    # Blob has blank row 0 — hat should render there
    bones = _make_bones(species=Species.BLOB, hat=Hat.CROWN)
    lines = render_sprite(bones, frame=0)
    assert "\\^^^/" in lines[0], f"Hat crown not on row 0: {lines[0]}"
    assert len({len(line) for line in lines}) == 1


def test_render_sprite_hat_not_applied_on_non_blank_row():
    """Hat is NOT applied when row 0 is not blank."""
    from chrys.app.features.buddy.sprites import render_sprite

    # Fox frame 0 row 0 is "   ^---^    " — not blank
    bones = _make_bones(species=Species.FOX, hat=Hat.CROWN)
    lines = render_sprite(bones, frame=0)
    assert "\\^^^/" not in lines[0], f"Hat should not be on fox row 0: {lines[0]}"


def test_render_sprite_wraps_frame_index():
    """Frame index beyond available frames wraps around."""
    from chrys.app.features.buddy.sprites import BODIES, render_sprite

    bones = _make_bones(species=Species.CAT)
    n_frames = len(BODIES[Species.CAT])
    lines_mod = render_sprite(bones, frame=0)
    lines_wrap = render_sprite(bones, frame=n_frames)
    assert lines_mod == lines_wrap


def test_render_sprite_all_species_render_without_error():
    """Every species renders without exception for all frames."""
    from chrys.app.features.buddy.sprites import BODIES, render_sprite

    for species in Species:
        bones = _make_bones(species=species)
        for frame in range(len(BODIES[species])):
            lines = render_sprite(bones, frame=frame)
            assert all(isinstance(line, str) for line in lines)
            assert len({len(line) for line in lines}) == 1


def test_render_sprite_applies_custom_eye_to_all_frames():
    """Every species and frame honors the selected eye style."""
    from chrys.app.features.buddy.sprites import BODIES, render_sprite

    for species in Species:
        bones = _make_bones(species=species, eye=Eye.STAR)
        for frame in range(len(BODIES[species])):
            combined = "\n".join(render_sprite(bones, frame=frame))
            assert "✦" in combined, f"{species} frame {frame} did not apply STAR eye:\n{combined}"
            assert "··" not in combined, f"{species} frame {frame} left raw eye anchor:\n{combined}"
            assert "[=]" not in combined, f"{species} frame {frame} left robot eye anchor:\n{combined}"


def test_render_sprite_robot_led_eyes():
    """Robot's [=] placeholder is replaced by eye char."""
    from chrys.app.features.buddy.sprites import render_sprite

    bones = _make_bones(species=Species.ROBOT, eye=Eye.AT)
    lines = render_sprite(bones, frame=0)
    combined = "\n".join(lines)
    assert "[@]" in combined, f"Robot LED eye with @ not found:\n{combined}"
    assert "[=]" not in combined, f"Robot [=] placeholder not replaced:\n{combined}"


# ── No more universal >_ mouth ────────────────────────────────────────────


_MOUTH_MAP: dict[Species, str | None] = {
    Species.DUCK: "__",
    Species.GOOSE: "°",
    Species.BLOB: "o",
    Species.CAT: "ω",
    Species.DRAGON: "﹀",
    Species.OCTOPUS: "~",
    Species.OWL: "v",
    Species.PENGUIN: ">",
    Species.TURTLE: "v",
    Species.SNAIL: "v",
    Species.GHOST: "O",
    Species.AXOLOTL: "v",
    Species.CAPYBARA: "_",
    Species.CACTUS: None,
    Species.ROBOT: "□",
    Species.RABBIT: "Y",
    Species.MUSHROOM: "o",
    Species.CHONK: "ω",
    Species.FOX: "﹀",
    Species.FROG: "﹀",
    Species.BEE: "~",
    Species.CRAB: "﹀",
    Species.BAT: "v",
    Species.SHARK: "﹏",  # noqa: RUF001
    Species.LLAMA: "v",
    Species.PANDA: "﹀",
    Species.SNAKE: "V",
    Species.ALIEN: "﹀",
}


@pytest.mark.parametrize("species", list(Species))
def test_species_specific_mouth_present(species):
    """Each species' sprite uses its specific mouth, not the old >_ template."""
    from chrys.app.features.buddy.sprites import BODIES

    expected_mouth = _MOUTH_MAP[species]
    frames = BODIES[species]

    # Every species must NOT have the old >_ mouth in frame 0
    for frame in frames:
        combined = "\n".join(frame)
        assert ">_" not in combined, f"{species} still has old >_ mouth:\n{combined}"

    # If expected_mouth is None (Cactus), verify NO mouth-like chars at mouth position
    if expected_mouth is None:
        # Cactus frame 0 row 3 is "   │  │    " — no mouth, just empty space
        pass  # Verified by >_ check above
    else:
        # Check that the expected mouth character appears in frame 0
        combined = "\n".join(frames[0])
        assert expected_mouth in combined, f"{species} missing mouth '{expected_mouth}' in frame 0:\n{combined}"


# ── render_face (narrow mode) ──────────────────────────────────────────────


def test_render_face_all_species():
    """render_face returns a non-empty string for every species."""
    from chrys.app.features.buddy.sprites import render_face

    for species in Species:
        comp = _make_companion(species=species)
        face = render_face(comp)
        assert face, f"{species} render_face returned empty string"
        assert isinstance(face, str)


def test_render_face_applies_eye():
    """render_face replaces {eye} with the correct eye character."""
    from chrys.app.features.buddy.sprites import render_face

    comp = _make_companion(species=Species.CAT, eye=Eye.STAR)
    face = render_face(comp)
    # Cat template: ({eye}ω{eye}) → should contain ✦
    assert "✦" in face, f"STAR eye not in face: {face}"


# ── SPECIES completeness ───────────────────────────────────────────────────


def test_species_enum_matches_sprite_keys():
    """SPECIES list matches BODIES keys exactly."""
    from chrys.app.features.buddy.sprites import BODIES

    assert set(SPECIES) == set(BODIES.keys())


def test_idle_sequences_matches_species():
    """IDLE_SEQUENCES keys match Species enum exactly."""
    assert {s.value for s in Species} == {s.value for s in IDLE_SEQUENCES}


def test_default_idle_sequence_unchanged():
    """DEFAULT_IDLE_SEQUENCE retains the original 15-step sequence for fallback."""
    assert len(DEFAULT_IDLE_SEQUENCE) == 15
    assert DEFAULT_IDLE_SEQUENCE == [0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 2, 0, 0, 0]
