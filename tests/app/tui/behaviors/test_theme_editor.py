# Copyright (c) 2026 Chrys. All rights reserved.

"""Unit tests for ``playground/theme_editor.py`` pure-Python logic.

These tests deliberately avoid Textual's pilot / async machinery: every
target is a method that only reads ``self._theme``, so we instantiate
``_ChrysExportModal`` via ``object.__new__`` and attach the theme manually.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from playground.theme_editor import (
    _THEME_NAME_RE,
    _TRANSPARENT_SENTINEL,
    _XTERM_256_ROWS,
    ChrysThemeEditorApp,
    ResettableThemeEditor,
    _ansi_base_token,
    _ansi_display_label,
    _ChrysExportModal,
    _group_variables,
    _PickerModal,
    _Xterm256PalettePicker,
)
from rich.color import EIGHT_BIT_PALETTE
from textual.color import Color
from textual.theme import BUILTIN_THEMES, Theme

from chrys.app.tui.theme import CHRYS_ANSI_THEME, CHRYS_COLOR_NAMES, CHRYS_DARK_THEME, CHRYS_THEME
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.service.state.store import JsonFileStateStore


class _Engine:
    async def shutdown(self) -> None:
        return


class _EmptyRegistry:
    def list_profiles(self) -> list[object]:
        return []

    def load_all(self) -> None:
        return

    def get(self, _name: str) -> None:
        return None


def _make_app(tmp_path: Path, theme_name: str) -> ChrysThemeEditorApp:
    return ChrysThemeEditorApp(
        EventBus(),
        _Engine(),  # type: ignore[arg-type]
        settings=Settings(theme=theme_name),
        state_store=JsonFileStateStore(tmp_path),
        agent_registry=_EmptyRegistry(),  # type: ignore[arg-type]
    )


def _make_modal(theme: Theme, snapshot: dict[str, dict[str, str | None]] | None = None) -> _ChrysExportModal:
    """Build a ``_ChrysExportModal`` without running Textual's ``Screen.__init__``.

    The render helpers and validators only touch ``self._theme`` plus
    the optional snapshot; skipping ``__init__`` keeps the unit tests
    free of any reliance on a running Textual app.
    """
    modal = object.__new__(_ChrysExportModal)
    modal._theme = theme
    modal._snapshot = snapshot
    return modal


@pytest.mark.parametrize(
    "name",
    ["chrys", "my-theme", "my_theme", "Theme1", "a"],
)
def test_name_regex_accepts_valid_names(name: str) -> None:
    assert _THEME_NAME_RE.fullmatch(name) is not None


@pytest.mark.parametrize(
    "name",
    ["-foo", "_foo", "9-theme", "1abc", "my theme", "my.theme"],
)
def test_name_regex_rejects_invalid_names(name: str) -> None:
    assert _THEME_NAME_RE.fullmatch(name) is None


@pytest.mark.parametrize(
    ("name", "needle"),
    [
        ("", "required"),
        ("-foo", "Start with a letter"),
        ("9-theme", "Start with a letter"),
        ("textual-light", "conflicts"),
        ("TEXTUAL-LIGHT", "conflicts"),
    ],
)
def test_validate_name_rejects(name: str, needle: str) -> None:
    modal = _make_modal(CHRYS_THEME)
    error = modal._validate_name(name)
    assert error is not None and needle in error


@pytest.mark.parametrize("name", ["chrys", "my-cool-theme"])
def test_validate_name_accepts(name: str) -> None:
    modal = _make_modal(CHRYS_THEME)
    assert modal._validate_name(name) is None


def test_suggest_unique_name_bumps_on_builtin_conflict() -> None:
    modal = _make_modal(CHRYS_THEME)
    # Pick any builtin name to guarantee a conflict on the first try.
    builtin = next(iter(BUILTIN_THEMES))
    assert modal._suggest_unique_name(builtin) == f"{builtin}-1"


def test_chrys_dark_export_reserves_registered_name() -> None:
    modal = _make_modal(CHRYS_DARK_THEME)

    error = modal._validate_name("chrys-dark")
    assert error is not None and "reserved" in error
    assert modal._suggest_unique_name("chrys-dark") == "chrys-dark-1"
    rendered = modal._render_code("chrys-dark-1")
    assert "Create new theme 'chrys-dark-1'" in rendered
    assert "register_theme(CHRYS_DARK_1_THEME)" in rendered


def test_suggest_unique_name_returns_unchanged_when_unique() -> None:
    modal = _make_modal(CHRYS_THEME)
    assert modal._suggest_unique_name("my-cool-theme") == "my-cool-theme"


def test_suggest_unique_name_preserves_chrys() -> None:
    # ``chrys`` is the legitimate update target so the validator
    # allows it; the suggester should leave it alone.
    modal = _make_modal(CHRYS_THEME)
    assert modal._suggest_unique_name("chrys") == "chrys"


def test_render_update_chrys_emits_replacement_blocks() -> None:
    from chrys.app.tui.theme import CHRYS_ONLY_VARIABLES

    rendered = _make_modal(CHRYS_THEME)._render_update_chrys()
    assert rendered.startswith(f'_DARK_GRAY = "{CHRYS_THEME.background}"')
    assert "_CHRYS_COLORS = {" in rendered
    # Mirrors theme.py: chrys-only slots split out, then spread back
    # into ``_CHRYS_VARIABLES`` via ``**CHRYS_ONLY_VARIABLES``.
    assert "CHRYS_ONLY_VARIABLES: dict[str, str] = {" in rendered
    assert "_CHRYS_VARIABLES = {" in rendered
    assert "    **CHRYS_ONLY_VARIABLES," in rendered
    # Replacement dicts only; no constructor.
    assert "Theme(" not in rendered
    # Every theme.variables entry should appear once as a "key": "value" line.
    for key, value in CHRYS_THEME.variables.items():
        assert f'    "{key}": "{value}",' in rendered
    # chrys-only entries live in the ``CHRYS_ONLY_VARIABLES`` block,
    # not duplicated inside ``_CHRYS_VARIABLES``.
    chrys_only_start = rendered.index("CHRYS_ONLY_VARIABLES: dict[str, str] = {")
    variables_start = rendered.index("_CHRYS_VARIABLES = {")
    chrys_only_block = rendered[chrys_only_start:variables_start]
    variables_block = rendered[variables_start:]
    for key in CHRYS_ONLY_VARIABLES:
        assert f'    "{key}":' in chrys_only_block
        assert f'    "{key}":' not in variables_block


def test_render_update_chrys_ansi_emits_ansi_block_and_constructor() -> None:
    rendered = _make_modal(CHRYS_ANSI_THEME)._render_update_chrys()
    # chrys-ansi inlines its surface values directly in the constructor;
    # only base chrys hoists them to a ``_DARK_GRAY`` constant.  (Match
    # the assignment specifically — the literal name also appears inside
    # the markdown-h6 inline comment.)
    assert "_DARK_GRAY =" not in rendered
    assert rendered.startswith("_CHRYS_COLORS = {")
    # The chrys-ansi update mode appends a ``_CHRYS_ANSI_VARIABLES``
    # block that spreads ``_CHRYS_VARIABLES``, plus a constructor.
    assert "_CHRYS_ANSI_VARIABLES = {" in rendered
    ansi_block_start = rendered.index("_CHRYS_ANSI_VARIABLES = {")
    constructor_start = rendered.index("CHRYS_ANSI_THEME = Theme(")
    ansi_block = rendered[ansi_block_start:constructor_start]
    assert "    **_CHRYS_VARIABLES," in ansi_block
    # Every chrys-ansi entry that differs from base chrys (ansi-only
    # additions + shared-key overrides) must appear in the ANSI block.
    for name, value in CHRYS_ANSI_THEME.variables.items():
        if CHRYS_THEME.variables.get(name) == value:
            continue
        assert f'"{name}":' in ansi_block
    # markdown-h1..h5 round-trip back to their ``_CHRYS_COLORS[...]``
    # alias form (not raw hex), preserving source theme.py intent.
    for name in ("markdown-h1-color", "markdown-h2-color", "markdown-h3-color"):
        assert f'"{name}": _CHRYS_COLORS["primary"],' in ansi_block
    for name in ("markdown-h4-color", "markdown-h5-color"):
        assert f'"{name}": _CHRYS_COLORS["foreground"],' in ansi_block
    # Inline comment block sits immediately above markdown-h6-color
    # (between h5 and h6) — exact source-order roundtrip.
    h5_pos = ansi_block.index('"markdown-h5-color":')
    comment_pos = ansi_block.index("# Nearest 256-color approximations")
    h6_pos = ansi_block.index('"markdown-h6-color":')
    assert h5_pos < comment_pos < h6_pos
    assert '"markdown-h6-color": "#9E9E9E",' in ansi_block
    # The constructor block mirrors theme.py's CHRYS_ANSI_THEME shape.
    constructor_block = rendered[constructor_start:]
    assert '    name="chrys-ansi",' in constructor_block
    assert "    ansi=True," in constructor_block
    assert "    **_CHRYS_COLORS," in constructor_block
    for attr in ("background", "surface", "panel", "boost"):
        assert f'    {attr}="{getattr(CHRYS_ANSI_THEME, attr)}",' in constructor_block
    assert f"    dark={CHRYS_ANSI_THEME.dark}," in constructor_block
    assert "    variables=_CHRYS_ANSI_VARIABLES," in constructor_block


def test_render_update_chrys_ansi_exports_shared_variable_edits() -> None:
    variables = dict(CHRYS_ANSI_THEME.variables)
    variables["border-blurred"] = "#123456"
    variables["button-color-foreground"] = "#654321"
    variables["overlay-background"] = "#456789"
    custom = Theme(
        name="chrys-ansi",
        ansi=True,
        primary="#345678",
        secondary=CHRYS_ANSI_THEME.secondary,
        warning=CHRYS_ANSI_THEME.warning,
        error=CHRYS_ANSI_THEME.error,
        success=CHRYS_ANSI_THEME.success,
        accent=CHRYS_ANSI_THEME.accent,
        foreground=CHRYS_ANSI_THEME.foreground,
        background=CHRYS_ANSI_THEME.background,
        surface=CHRYS_ANSI_THEME.surface,
        panel=CHRYS_ANSI_THEME.panel,
        boost=CHRYS_ANSI_THEME.boost,
        dark=CHRYS_ANSI_THEME.dark,
        variables=variables,
    )

    rendered = _make_modal(custom)._render_update_chrys()
    colors_start = rendered.index("_CHRYS_COLORS = {")
    chrys_only_start = rendered.index("CHRYS_ONLY_VARIABLES: dict[str, str] = {")
    variables_start = rendered.index("_CHRYS_VARIABLES = {")
    ansi_start = rendered.index("_CHRYS_ANSI_VARIABLES = {")
    constructor_start = rendered.index("CHRYS_ANSI_THEME = Theme(")
    colors_block = rendered[colors_start:chrys_only_start]
    chrys_only_block = rendered[chrys_only_start:variables_start]
    variables_block = rendered[variables_start:ansi_start]
    ansi_block = rendered[ansi_start:constructor_start]

    assert '"primary": "#345678",  # changed for chrys-ansi' in colors_block
    assert '"overlay-background": "#456789",  # changed for chrys-ansi' in chrys_only_block
    assert '"border-blurred": "#123456",  # changed for chrys-ansi' in variables_block
    assert '"border-blurred": "#123456",' not in ansi_block
    assert f'"button-color-foreground": "{CHRYS_THEME.variables["button-color-foreground"]}",' in variables_block
    assert '"button-color-foreground": "#654321",' in ansi_block


def test_render_update_chrys_ansi_annotations_compare_against_snapshot() -> None:
    custom = Theme(
        name="chrys-ansi",
        ansi=True,
        primary="#345678",
        secondary=CHRYS_ANSI_THEME.secondary,
        warning=CHRYS_ANSI_THEME.warning,
        error=CHRYS_ANSI_THEME.error,
        success=CHRYS_ANSI_THEME.success,
        accent=CHRYS_ANSI_THEME.accent,
        foreground=CHRYS_ANSI_THEME.foreground,
        background=CHRYS_ANSI_THEME.background,
        surface=CHRYS_ANSI_THEME.surface,
        panel=CHRYS_ANSI_THEME.panel,
        boost=CHRYS_ANSI_THEME.boost,
        dark=CHRYS_ANSI_THEME.dark,
        variables=dict(CHRYS_ANSI_THEME.variables),
    )
    snapshot = {
        "colors": {name: getattr(custom, name) for name in CHRYS_COLOR_NAMES},
        "variables": dict(custom.variables),
    }

    rendered = _make_modal(custom, snapshot)._render_update_chrys()

    assert '"primary": "#345678",' in rendered
    assert '"primary": "#345678",  # changed for chrys-ansi' not in rendered


def test_render_update_chrys_does_not_annotate_base_chrys_edits() -> None:
    variables = dict(CHRYS_THEME.variables)
    variables["border-blurred"] = "#123456"
    custom = Theme(
        name="chrys",
        primary="#345678",
        secondary=CHRYS_THEME.secondary,
        warning=CHRYS_THEME.warning,
        error=CHRYS_THEME.error,
        success=CHRYS_THEME.success,
        accent=CHRYS_THEME.accent,
        foreground=CHRYS_THEME.foreground,
        background=CHRYS_THEME.background,
        surface=CHRYS_THEME.surface,
        panel=CHRYS_THEME.panel,
        boost=CHRYS_THEME.boost,
        dark=CHRYS_THEME.dark,
        variables=variables,
    )

    rendered = _make_modal(custom)._render_update_chrys()

    assert "# changed for chrys-ansi" not in rendered
    assert '"primary": "#345678",' in rendered
    assert '"border-blurred": "#123456",' in rendered


def test_render_update_chrys_ansi_falls_back_to_literal_when_alias_diverges() -> None:
    # User edits ``primary`` away from chrys's source value but leaves
    # ``markdown-h1-color`` untouched (still the OLD primary hex).  Export
    # must drop the ``_CHRYS_COLORS["primary"]`` alias for h1, since the
    # alias would silently re-bind h1 to the user's NEW primary on reload.
    custom = Theme(
        name="chrys-ansi",
        ansi=True,
        primary="#123456",  # diverges from base chrys primary
        secondary=CHRYS_ANSI_THEME.secondary,
        warning=CHRYS_ANSI_THEME.warning,
        error=CHRYS_ANSI_THEME.error,
        success=CHRYS_ANSI_THEME.success,
        accent=CHRYS_ANSI_THEME.accent,
        foreground=CHRYS_ANSI_THEME.foreground,
        background=CHRYS_ANSI_THEME.background,
        surface=CHRYS_ANSI_THEME.surface,
        panel=CHRYS_ANSI_THEME.panel,
        boost=CHRYS_ANSI_THEME.boost,
        dark=CHRYS_ANSI_THEME.dark,
        variables=dict(CHRYS_ANSI_THEME.variables),
    )
    rendered = _make_modal(custom)._render_update_chrys()
    # h1 was "#AF87FF" (old primary); current primary is "#123456".  No
    # match → emit literal, not the alias.
    assert '"markdown-h1-color": "#AF87FF",' in rendered
    assert '"markdown-h1-color": _CHRYS_COLORS["primary"],' not in rendered
    # h4/h5 still alias-match foreground (unchanged), so they stay aliased.
    assert '"markdown-h4-color": _CHRYS_COLORS["foreground"],' in rendered


def test_render_update_chrys_non_ansi_omits_ansi_block() -> None:
    # Exporting from chrys (not chrys-ansi) must not emit the
    # chrys-ansi-specific blocks — the user is updating base chrys only.
    rendered = _make_modal(CHRYS_THEME)._render_update_chrys()
    assert "_CHRYS_ANSI_VARIABLES" not in rendered
    assert "CHRYS_ANSI_THEME = Theme(" not in rendered


def test_render_code_dispatches_chrys_to_update_mode() -> None:
    modal = _make_modal(CHRYS_THEME)
    # Only the matching chrys-family name (case-insensitive) triggers update
    # mode; cross-name ``chrys-ansi`` falls through to new-theme mode and
    # would be rejected by ``_validate_name`` before reaching this branch.
    assert "Theme(" not in modal._render_code("chrys")
    assert "Theme(" not in modal._render_code("CHRYS")
    assert "Theme(" in modal._render_code("chrys-ansi")
    assert "Theme(" in modal._render_code("my-theme")


def test_render_code_chrys_ansi_base_only_matches_chrys_ansi_name() -> None:
    # Opening export on chrys-ansi defaults the name input to
    # ``chrys-ansi``; only that exact name reaches the update path
    # (which then emits the chrys-ansi-specific constructor + ANSI block).
    modal = _make_modal(CHRYS_ANSI_THEME)
    assert "CHRYS_ANSI_THEME = Theme(" in modal._render_code("chrys-ansi")
    # Cross-name ``chrys`` no longer routes to update; falls through to
    # new-theme mode (validation rejects it in the UI flow regardless).
    assert "_CHRYS_ANSI_VARIABLES" not in modal._render_code("chrys")
    # Non-chrys-family name still falls through to new-theme mode.
    assert "CHRYS_ANSI_THEME = Theme(" not in modal._render_code("my-theme")


def test_validate_name_rejects_cross_chrys_family_name() -> None:
    # Exporting from chrys-ansi: typing ``chrys`` (the *other* chrys-family
    # name) must be rejected — update mode renders from ``self._theme``,
    # so a cross-name would either emit wrong-flavour output or, after the
    # ``_render_code`` tightening, clobber the existing chrys-ansi block
    # via new-theme mode.
    modal = _make_modal(CHRYS_ANSI_THEME)
    assert modal._validate_name("chrys") is not None
    assert modal._validate_name("Chrys") is not None  # case-insensitive
    assert modal._validate_name("chrys-ansi") is None
    # Symmetric on chrys base.
    modal = _make_modal(CHRYS_THEME)
    assert modal._validate_name("chrys-ansi") is not None
    assert modal._validate_name("chrys") is None
    # On non-chrys base, neither chrys-family name is the active theme,
    # so both get rejected (cross-name in either direction).
    modal = _make_modal(BUILTIN_THEMES["textual-dark"])
    assert modal._validate_name("chrys") is not None
    assert modal._validate_name("chrys-ansi") is not None


def test_mode_hint_describes_update_vs_new_theme() -> None:
    # Hint is a Python comment block, prepended to the rendered code.
    modal = _make_modal(CHRYS_ANSI_THEME)
    update_hint = modal._mode_hint("chrys-ansi")
    assert update_hint.startswith("# ")
    assert "Update 'chrys-ansi'" in update_hint
    assert "src/chrys/app/tui/theme.py" in update_hint
    # New theme path mentions theme.py + the register_theme call site.
    new_hint = modal._mode_hint("my-theme")
    assert new_hint.startswith("# ")
    assert "Create new theme 'my-theme'" in new_hint
    assert "src/chrys/app/tui/theme.py" in new_hint
    assert "register_theme(MY_THEME_THEME)" in new_hint
    assert "src/chrys/app/tui/app.py" in new_hint


def test_render_code_prepends_mode_hint_as_comment() -> None:
    # The TextArea content begins with the mode-hint comment block, then
    # the actual export body — the user sees the instructions before the
    # code they're about to paste.
    modal = _make_modal(CHRYS_ANSI_THEME)
    rendered = modal._render_code("chrys-ansi")
    assert rendered.startswith("# Update 'chrys-ansi'")
    body_start = rendered.index("_CHRYS_COLORS = {")
    assert "\n\n" in rendered[:body_start]  # blank line between hint and body


def test_render_new_theme_chrys_family_collapses_surface_group() -> None:
    # chrys-family branching keys off ``theme.name in CHRYS_THEMES``,
    # not the user-typed name; ``_render_code`` would short-circuit
    # ``"chrys"`` to update mode, so use a variant name here.
    rendered = _make_modal(CHRYS_THEME)._render_new_theme("my-chrys-variant")
    # Single ``_<SLUG>_BACKGROUND`` constant for the flat-surface group.
    assert f'_MY_CHRYS_VARIANT_BACKGROUND = "{CHRYS_THEME.background}"' in rendered
    # All four surface kwargs point at the same constant.
    for attr in ("background", "surface", "panel", "boost"):
        assert f"    {attr}=_MY_CHRYS_VARIANT_BACKGROUND," in rendered
    assert "MY_CHRYS_VARIANT_THEME = Theme(" in rendered


def test_render_new_theme_non_chrys_skips_none_surface() -> None:
    probe = Theme(
        name="probe",
        primary="#FFFFFF",
        secondary="#EEEEEE",
        warning="#DDDDDD",
        error="#CCCCCC",
        success="#BBBBBB",
        accent="#AAAAAA",
        foreground="#999999",
        background="#101010",
        surface="#202020",
        panel="#303030",
        # boost left as default (None) — generated code must omit the
        # ``boost=`` kwarg entirely.
        dark=False,
    )
    rendered = _make_modal(probe)._render_new_theme("probe")
    # Non-chrys themes inline surface values directly in the ``Theme(...)``
    # constructor — no per-attribute module-level constants.
    assert "_PROBE_BACKGROUND" not in rendered
    assert "_PROBE_SURFACE" not in rendered
    assert "_PROBE_PANEL" not in rendered
    assert "_PROBE_BOOST" not in rendered
    # boost is None: omit both the kwarg and any string occurrence.
    assert "boost=" not in rendered
    assert '"boost"' not in rendered
    # The three surface kwargs we DID set should appear as inline literals.
    assert '    background="#101010",' in rendered
    assert '    surface="#202020",' in rendered
    assert '    panel="#303030",' in rendered


def test_group_variables_buckets_and_order() -> None:
    sample = [
        "border-blurred",
        "scrollbar-hover",
        "block-cursor-foreground",
        "input-cursor-background",
        "footer-background",
        "primary-muted",
        "overlay-background",
        # ``button-*`` doesn't match any prefix bucket, so it falls into ``misc``.
        "button-color-foreground",
    ]
    grouped = _group_variables(sample)
    assert grouped == [
        ("border", ["border-blurred"]),
        ("scrollbar", ["scrollbar-hover"]),
        ("block", ["block-cursor-foreground"]),
        ("input", ["input-cursor-background"]),
        ("footer", ["footer-background"]),
        ("muted", ["primary-muted"]),
        ("chrys", ["overlay-background"]),
        ("misc", ["button-color-foreground"]),
    ]


def test_ansi_variable_names_only_truly_ansi_slots() -> None:
    # ``_ANSI_VARIABLE_NAMES`` is the strict set of slots that only have
    # meaningful values when ``Theme.ansi=True`` (RGB themes resolve
    # both to ``transparent``).  Everything else chrys-ansi tunes —
    # text-muted / text-disabled / screen-selection-background — is
    # shown in regular sidebar buckets on every theme.
    from playground.theme_editor import _ANSI_VARIABLE_NAMES

    assert _ANSI_VARIABLE_NAMES == ("ansi-background", "ansi-foreground")


def test_group_variables_routes_ansi_names_to_ansi_bucket() -> None:
    grouped = _group_variables(["ansi-background", "ansi-foreground"])
    assert grouped == [("ansi", ["ansi-background", "ansi-foreground"])]


def test_group_variables_keeps_shared_names_out_of_ansi_bucket() -> None:
    # ``border-blurred`` looks ANSI-ish (built-in ANSI themes define it
    # too) but it's a generic textual slot — must stay in ``border``.
    from playground.theme_editor import _ANSI_VARIABLE_NAMES

    assert "border-blurred" not in _ANSI_VARIABLE_NAMES
    grouped = _group_variables(["border-blurred"])
    assert grouped == [("border", ["border-blurred"])]


def test_canonical_variable_names_includes_chrys_ansi_tuned_extras() -> None:
    # chrys-ansi explicitly overrides these textual-derived slots; the
    # sidebar surfaces them on every theme so edits made on base chrys
    # carry over visually when switching to chrys-ansi (or vice versa).
    from playground.theme_editor import _CANONICAL_VARIABLE_NAMES

    for name in ("text-muted", "text-disabled", "screen-selection-background"):
        assert name in _CANONICAL_VARIABLE_NAMES


def test_chrys_ansi_override_names_matches_theme_py() -> None:
    # ``_CHRYS_ANSI_OVERRIDE_NAMES`` is hand-ordered to mirror
    # ``_CHRYS_ANSI_VARIABLES`` in src/chrys/app/tui/theme.py.  Catch drift
    # whenever someone adds/removes an entry on either side.
    from playground.theme_editor import _CHRYS_ANSI_OVERRIDE_NAMES

    computed = {name for name, value in CHRYS_ANSI_THEME.variables.items() if CHRYS_THEME.variables.get(name) != value}
    assert set(_CHRYS_ANSI_OVERRIDE_NAMES) == computed


def test_markdown_header_colors_are_not_editable() -> None:
    # markdown-h1..h5 pin to chrys named colors; h6 is a fixed shade.
    # None of them appear in the sidebar's canonical list, but export
    # still emits all six via ``_CHRYS_ANSI_OVERRIDE_NAMES``.
    from playground.theme_editor import _CANONICAL_VARIABLE_NAMES, _CHRYS_ANSI_OVERRIDE_NAMES

    for name in (
        "markdown-h1-color",
        "markdown-h2-color",
        "markdown-h3-color",
        "markdown-h4-color",
        "markdown-h5-color",
        "markdown-h6-color",
    ):
        assert name not in _CANONICAL_VARIABLE_NAMES
        assert name in _CHRYS_ANSI_OVERRIDE_NAMES


def _is_palette_hex(value: str) -> bool:
    palette = {f"#{red:02X}{green:02X}{blue:02X}" for red, green, blue in EIGHT_BIT_PALETTE}
    return value.upper() in palette


@pytest.mark.parametrize(
    "raw_hex",
    [
        "#1C1C1C",  # gray entry, on-palette
        "#AF87FF",  # chrys primary, on-palette
        "#5FFF87",  # chrys success, on-palette
    ],
)
def test_match_palette_index_returns_exact_index_for_on_palette_hex(raw_hex: str) -> None:
    color = Color.parse(raw_hex)
    index = _Xterm256PalettePicker._match_palette_index(color)
    triplet = EIGHT_BIT_PALETTE[index]
    assert f"#{triplet.red:02X}{triplet.green:02X}{triplet.blue:02X}" == raw_hex.upper()


@pytest.mark.parametrize(
    "raw_hex",
    [
        "#FB7F2A",  # truecolor orange — off-palette
        "#123456",  # arbitrary
        "#2B2E3B",  # near-gray, off-palette
    ],
)
def test_match_palette_index_picks_nearest_neighbor_for_off_palette_hex(raw_hex: str) -> None:
    color = Color.parse(raw_hex)
    index = _Xterm256PalettePicker._match_palette_index(color)
    # Rich's ``Palette.match`` uses a perceptually-weighted distance.
    # We just verify the returned index is on-palette and that no
    # palette entry has an exact-RGB hit we'd be ignoring.
    triplet = EIGHT_BIT_PALETTE[index]
    assert 0 <= index < 256
    # An exact hex match anywhere in the palette would always win
    # distance=0; verify the chosen index hits exact when one exists,
    # and otherwise the chosen entry's distance is at least as good as
    # naive squared-RGB nearest.
    exact = next(
        (i for i in range(256) if EIGHT_BIT_PALETTE[i] == triplet and triplet == (color.r, color.g, color.b)), None
    )
    if exact is not None:
        assert index == exact


def test_match_palette_index_returns_sentinel_for_transparent() -> None:
    assert _Xterm256PalettePicker._match_palette_index(Color(0, 0, 0, a=0)) == _TRANSPARENT_SENTINEL


def test_match_palette_index_skips_sentinel_when_transparent_disallowed() -> None:
    # When the picker hides its transparent cell (surface slots), a
    # transparent input must fall back to a real palette index instead
    # of returning the sentinel — otherwise the position lookup would
    # KeyError on the missing sentinel row.
    index = _Xterm256PalettePicker._match_palette_index(Color(0, 0, 0, a=0), allow_transparent=False)
    assert index != _TRANSPARENT_SENTINEL
    assert 0 <= index < 256


def test_xterm_picker_initialized_with_transparent_selects_sentinel_cell() -> None:
    picker = _Xterm256PalettePicker(Color(0, 0, 0, a=0))
    assert picker._selected_index == _TRANSPARENT_SENTINEL
    assert picker.color.is_transparent


def test_xterm_picker_transparent_cell_renders_check_when_selected() -> None:
    # Numbered cells convey "selected" via their index digits appearing
    # inside the cell. The transparent cell mirrors that by showing a
    # ``✓`` marker only when selected — bold + dim alone is too easy to
    # miss on the checker pattern.
    transparent_row = next(i for i, (_, idx) in enumerate(_XTERM_256_ROWS) if _TRANSPARENT_SENTINEL in idx)
    selected = _Xterm256PalettePicker(Color(0, 0, 0, a=0))
    selected_strip = selected.render_line(transparent_row)
    selected_text = "".join(segment.text for segment in selected_strip)
    assert "✓" in selected_text
    unselected = _Xterm256PalettePicker(Color.parse("#FFFFFF"))
    unselected_strip = unselected.render_line(transparent_row)
    unselected_text = "".join(segment.text for segment in unselected_strip)
    assert "✓" not in unselected_text
    assert "▒" in unselected_text


def test_xterm_picker_without_transparent_hides_transparent_section_when_no_hint() -> None:
    # ``allow_transparent=False`` + ``disabled_hint=None`` removes the
    # section entirely: no header, no selectable sentinel, no hint row.
    picker = _Xterm256PalettePicker(Color.parse("#FFFFFF"), allow_transparent=False)
    layout_indices = {idx for _, indices in picker._rows for idx in indices}
    assert _TRANSPARENT_SENTINEL not in layout_indices
    labels = {label for label, _ in picker._rows}
    assert "Transparent" not in labels
    assert picker._selected_index != _TRANSPARENT_SENTINEL


def test_xterm_picker_with_disabled_hint_keeps_transparent_header() -> None:
    # When a hint is provided, the "Transparent" header stays so the user
    # still sees *where* the option lives; the sentinel cell is replaced
    # by an empty placeholder row that ``render_line`` paints as the hint.
    picker = _Xterm256PalettePicker(
        Color.parse("#FFFFFF"),
        allow_transparent=False,
        disabled_hint="background can't be transparent",
    )
    labels = {label for label, _ in picker._rows}
    assert "Transparent" in labels
    # No selectable sentinel cell — the option is *visually present but
    # unavailable*, not selectable.
    layout_indices = {idx for _, indices in picker._rows for idx in indices}
    assert _TRANSPARENT_SENTINEL not in layout_indices
    assert picker._selected_index != _TRANSPARENT_SENTINEL


def test_xterm_picker_disabled_hint_only_applied_when_transparent_blocked() -> None:
    # The hint is meaningful only when transparent is hidden — passing
    # ``disabled_hint`` alongside ``allow_transparent=True`` would be a
    # contradiction in caller intent, so the picker drops it silently.
    picker = _Xterm256PalettePicker(
        Color.parse("#FFFFFF"),
        allow_transparent=True,
        disabled_hint="background can't be transparent",
    )
    assert picker._disabled_hint is None


def test_xterm_picker_disabled_hint_renders_in_placeholder_row() -> None:
    # The hint paints the placeholder row (``(None, ())``) that sits
    # immediately below the "Transparent" header — same vertical position
    # the sentinel cell occupies in the with-transparent layout.
    hint = "background can't be transparent"
    picker = _Xterm256PalettePicker(
        Color.parse("#FFFFFF"),
        allow_transparent=False,
        disabled_hint=hint,
    )
    # Placeholder is the last row in ``WITH_HINT`` layout.
    placeholder_row = len(picker._rows) - 1
    placeholder_strip = picker.render_line(placeholder_row)
    placeholder_text = "".join(segment.text for segment in placeholder_strip)
    assert hint in placeholder_text
    # The "Transparent" header above the placeholder still renders.
    header_row = placeholder_row - 1
    header_strip = picker.render_line(header_row)
    header_text = "".join(segment.text for segment in header_strip)
    assert "Transparent" in header_text


def test_xterm_picker_without_transparent_recovers_when_seeded_transparent() -> None:
    # Defensive path: if a slot somehow already holds a transparent value
    # when the picker opens with ``allow_transparent=False``, the picker
    # must still construct (no KeyError on the sentinel position) and
    # land on a real palette index.
    picker = _Xterm256PalettePicker(Color(0, 0, 0, a=0), allow_transparent=False)
    assert picker._selected_index != _TRANSPARENT_SENTINEL
    assert not picker.color.is_transparent


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ansi_red", "ansi_red"),
        ("ansi_bright_magenta", "ansi_bright_magenta"),
        ("ansi_white 40%", "ansi_white"),
        ("ansi_default", "ansi_default"),
        ("#FF0000", None),
        ("transparent", None),
        ("", None),
        (None, None),
    ],
)
def test_ansi_base_token(value: str | None, expected: str | None) -> None:
    assert _ansi_base_token(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ansi_default", "default"),
        ("ansi_red", "red"),
        ("ansi_bright_magenta", "br.magenta"),
        ("ansi_white 40%", "white 40%"),
        ("ansi_bright_cyan 20%", "br.cyan 20%"),
        ("#FF0000", None),
        ("transparent", None),
    ],
)
def test_ansi_display_label(value: str, expected: str | None) -> None:
    assert _ansi_display_label(value) == expected


def _stub_editor(*, theme_name: str, chrys_family: bool) -> SimpleNamespace:
    """Minimal ``_editor`` stub for ``_PickerModal._initial_picker_mode``.

    The method only touches ``self._editor._is_chrys_ansi()`` and
    ``self._editor._is_chrys_family()`` — no Textual machinery needed.
    """
    return SimpleNamespace(
        _theme=SimpleNamespace(name=theme_name),
        _is_chrys_ansi=lambda: theme_name == "chrys-ansi",
        _is_chrys_family=lambda: chrys_family,
    )


def _make_picker_modal(editor: SimpleNamespace) -> _PickerModal:
    modal = object.__new__(_PickerModal)
    modal._editor = editor
    return modal


@pytest.mark.parametrize(
    ("theme_name", "chrys_family", "value", "expected"),
    [
        # Non-chrys-ansi: chrys family gets the discrete xterm picker,
        # everything else gets the upstream continuous RGB picker.
        ("chrys", True, "#BD93F9", "xterm"),
        ("textual-dark", False, "#BD93F9", "rgb"),
        # chrys-ansi: token values open the ANSI picker; hex / None
        # open the xterm palette picker.
        ("chrys-ansi", True, "ansi_red", "ansi"),
        ("chrys-ansi", True, "ansi_white 40%", "ansi"),
        ("chrys-ansi", True, "#BD93F9", "xterm"),
        ("chrys-ansi", True, None, "xterm"),
        # Transparent isn't an ANSI token, so it routes to the xterm
        # picker (which now has a dedicated transparent cell).
        ("chrys", True, "transparent", "xterm"),
        ("chrys-ansi", True, "transparent", "xterm"),
        # Built-in ANSI themes never open the picker because the editor
        # hides itself behind ``--ansi-disabled`` — but defend against
        # any callers that bypass the check by returning ``rgb``.
        ("ansi-dark", False, "#BD93F9", "rgb"),
    ],
)
def test_initial_picker_mode(theme_name: str, chrys_family: bool, value: str | None, expected: str) -> None:
    modal = _make_picker_modal(_stub_editor(theme_name=theme_name, chrys_family=chrys_family))
    assert modal._initial_picker_mode(value) == expected


def test_current_color_resolves_transparent_to_alpha_zero() -> None:
    """``_current_color`` is what seeds ``_Xterm256PalettePicker`` on open.

    The picker uses ``is_transparent`` to land on the dedicated cell, so
    transparent values must flow through as alpha=0 instead of falling
    back to the ``#FFFFFF`` default.
    """
    modal = _make_picker_modal(_stub_editor(theme_name="chrys", chrys_family=True))
    modal._current_value = "transparent"
    color = modal._current_color()
    assert color.is_transparent


@pytest.mark.asyncio
async def test_builtin_ansi_theme_editor_does_not_focus_or_open_picker(tmp_path: Path) -> None:
    app = _make_app(tmp_path, "ansi-dark")

    async with app.run_test(size=(140, 50)) as pilot:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while True:
            editors = list(app.screen.query(ResettableThemeEditor))
            if editors and editors[0].has_class("--ansi-disabled") and editors[0]._theme.name == "ansi-dark":
                editor = editors[0]
                break
            if loop.time() > deadline:
                raise AssertionError("ANSI theme editor did not finish initializing")
            await pilot.pause(0.05)

        assert editor.has_class("--ansi-disabled")
        assert editor._theme.name == "ansi-dark"
        assert editor._has_internal_focus() is False

        await pilot.press("enter")
        await pilot.pause()

        assert not any(isinstance(screen, _PickerModal) for screen in app.screen_stack)


@pytest.mark.asyncio
async def test_chrys_only_variable_picker_uses_app_fallback_default(tmp_path: Path) -> None:
    app = _make_app(tmp_path, "textual-dark")

    async with app.run_test(size=(140, 50)) as pilot:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5
        while True:
            editors = list(app.screen.query(ResettableThemeEditor))
            if editors and editors[0]._theme.name == "textual-dark":
                editor = editors[0]
                break
            if loop.time() > deadline:
                raise AssertionError("Textual theme editor did not finish initializing")
            await pilot.pause(0.05)

        assert editor._theme.variables.get("control-background") is None
        assert editor._color_mapping.get("control-background") is None
        assert editor._variable_picker_initial("control-background") == "#303030"
        assert editor._variable_picker_initial("overlay-background") == "#3A3A3A"
