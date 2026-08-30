# Copyright (c) 2026 Chrys. All rights reserved.

"""Playground: inspect Buddy species sprites and animation frames.

Usage:
    uv run python playground/buddy.py

Keybindings:
    Space       — play / pause animation
    n / p       — next / previous animation tick
    x           — toggle pet-mode animation
    b           — toggle blink override
    e           — cycle eye style
    h           — cycle hat
    r           — reset all controls
    q           — quit
"""

from __future__ import annotations

from itertools import batched

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.timer import Timer
from textual.widgets import Button, Footer, Header, Static

from chrys.app.features.buddy.sprites import EYE_CHARS, HAT_LINES, get_frame_count, get_idle_frame, render_sprite
from chrys.app.features.buddy.types import (
    HATS,
    RARITY_COLORS,
    SPECIES,
    SPECIES_BY_RARITY,
    STATS,
    CompanionBones,
    Eye,
    Hat,
    Rarity,
    Species,
)

COLUMNS = 4
TICK_SECONDS = 0.35
EYES = list(Eye)
HAT_OPTIONS = [Hat.NONE, *HATS]


RARITY_BY_SPECIES = {
    species: rarity for rarity, species_group in SPECIES_BY_RARITY.items() for species in species_group
}


def _bones_for(species: Species, eye: Eye, hat: Hat) -> CompanionBones:
    rarity = RARITY_BY_SPECIES.get(species, Rarity.N)
    return CompanionBones(rarity, species, eye, hat, False, dict.fromkeys(STATS, 10))


class SpeciesTile(Static):
    """One rendered Buddy species."""

    def __init__(self, species: Species) -> None:
        super().__init__("", classes="species-tile")
        self.species = species

    def render_species(self, tick: int, eye: Eye, hat: Hat, blink_override: bool, pet_mode: bool) -> None:
        frame_count = get_frame_count(self.species)
        if pet_mode:
            frame = tick % frame_count
            blink = False
        else:
            frame, should_blink = get_idle_frame(self.species, tick)
            blink = blink_override or should_blink
        bones = _bones_for(self.species, eye, hat)
        rows = render_sprite(bones, frame=frame, blink=blink)
        rarity = RARITY_BY_SPECIES.get(self.species, Rarity.N)
        color = RARITY_COLORS.get(rarity, "#ffffff")

        text = Text()
        text.append(f"{self.species.value}  ", style="bold")
        text.append(f"{rarity.value}", style=color)
        text.append(f"  frame {frame + 1}/{frame_count}")
        if blink:
            text.append("  blink", style="italic")
        if pet_mode:
            text.append("  pet", style="italic")
        text.append("\n")
        text.append("\n".join(rows), style=color)
        self.update(text)


class BuddyPlayground(App):
    """Buddy sprite debugging playground."""

    BINDINGS = [
        Binding("space", "toggle_play", "Play/Pause"),
        Binding("n", "next_tick", "Next"),
        Binding("p", "previous_tick", "Previous"),
        Binding("x", "toggle_pet", "Pet Mode"),
        Binding("b", "toggle_blink", "Blink"),
        Binding("e", "next_eye", "Eye"),
        Binding("h", "next_hat", "Hat"),
        Binding("r", "reset", "Reset"),
        Binding("q", "quit", "Quit"),
    ]

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }

    #controls {
        dock: top;
        height: 3;
        padding: 0 1;
        background: $panel;
        align: left middle;
    }

    #status {
        width: 1fr;
        content-align: right middle;
        padding: 0 1;
        color: $text-muted;
    }

    #species-scroll {
        height: 1fr;
    }

    .species-row {
        height: auto;
    }

    .species-tile {
        width: 1fr;
        min-height: 10;
        margin: 0 1 1 0;
        padding: 1;
        border: solid $surface;
        background: $surface 20%;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.tick = 0
        self.playing = True
        self.pet_mode = False
        self.blink_override = False
        self.eye_index = 0
        self.hat_index = 0
        self._timer: Timer | None = None

    @property
    def eye(self) -> Eye:
        return EYES[self.eye_index]

    @property
    def hat(self) -> Hat:
        return HAT_OPTIONS[self.hat_index]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="controls"):
            yield Button("Play / Pause", id="play-pause")
            yield Button("Prev", id="previous")
            yield Button("Next", id="next")
            yield Button("Pet", id="pet")
            yield Button("Blink", id="blink")
            yield Button("Eye", id="eye")
            yield Button("Hat", id="hat")
            yield Button("Reset", id="reset")
            yield Static("", id="status")
        with VerticalScroll(id="species-scroll"):
            for row in batched(SPECIES, COLUMNS, strict=False):
                with Horizontal(classes="species-row"):
                    for species in row:
                        yield SpeciesTile(species)
        yield Footer()

    def on_mount(self) -> None:
        self._timer = self.set_interval(TICK_SECONDS, self._advance)
        self._refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "play-pause":
                self.action_toggle_play()
            case "previous":
                self.action_previous_tick()
            case "next":
                self.action_next_tick()
            case "pet":
                self.action_toggle_pet()
            case "blink":
                self.action_toggle_blink()
            case "eye":
                self.action_next_eye()
            case "hat":
                self.action_next_hat()
            case "reset":
                self.action_reset()

    def action_toggle_play(self) -> None:
        self.playing = not self.playing
        if self._timer is not None:
            if self.playing:
                self._timer.resume()
            else:
                self._timer.pause()
        self._refresh()

    def action_next_tick(self) -> None:
        self.tick += 1
        self._refresh()

    def action_previous_tick(self) -> None:
        self.tick = max(0, self.tick - 1)
        self._refresh()

    def action_toggle_blink(self) -> None:
        self.blink_override = not self.blink_override
        self._refresh()

    def action_toggle_pet(self) -> None:
        self.pet_mode = not self.pet_mode
        self._refresh()

    def action_next_eye(self) -> None:
        self.eye_index = (self.eye_index + 1) % len(EYES)
        self._refresh()

    def action_next_hat(self) -> None:
        self.hat_index = (self.hat_index + 1) % len(HAT_OPTIONS)
        self._refresh()

    def action_reset(self) -> None:
        self.tick = 0
        self.pet_mode = False
        self.blink_override = False
        self.eye_index = 0
        self.hat_index = 0
        self._refresh()

    def _advance(self) -> None:
        self.tick += 1
        self._refresh()

    def _refresh(self) -> None:
        for tile in self.query(SpeciesTile):
            tile.render_species(self.tick, self.eye, self.hat, self.blink_override, self.pet_mode)

        eye_label = EYE_CHARS[self.eye]
        hat_label = "none" if self.hat == Hat.NONE else HAT_LINES[self.hat].strip()
        state = "playing" if self.playing else "paused"
        blink = "forced blink" if self.blink_override else "sequence blink"
        mode = "pet mode" if self.pet_mode else "idle mode"
        self.query_one("#status", Static).update(
            f"tick {self.tick} | {state} | {mode} | eye {self.eye.value} {eye_label} | "
            f"hat {self.hat.value} {hat_label} | {blink}"
        )


if __name__ == "__main__":
    BuddyPlayground().run()
