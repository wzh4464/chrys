# Copyright (c) 2026 Chrys. All rights reserved.

"""Chrys TUI theme definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

from textual.theme import Theme

if TYPE_CHECKING:
    from textual.app import App

    _TuiVariableDefaultsBase = App[Any]
else:
    _TuiVariableDefaultsBase = object

# Themes that use native ANSI color passthrough for transparent backgrounds.
# Keep each of these themes' ``Theme.ansi`` flags in sync with this set; ChrysApp
# uses the set only for CSS classes and widget-level theme decisions.
ANSI_THEMES = {"ansi-dark", "ansi-light", "chrys-ansi"}

# Chrys themes supported by the palette editor's specialized update/export path.
CHRYS_THEMES = {"chrys", "chrys-ansi"}
CHRYS_DARK_THEME_NAME = "chrys-dark"
# All themes that share the ``-chrys`` CSS styling family.
CHRYS_FAMILY_THEMES = {*CHRYS_THEMES, CHRYS_DARK_THEME_NAME}


# Shared Dracula-inspired palette used by both chrys themes.
class _ChrysColors(TypedDict):
    """Keyword-compatible color subset accepted by ``Theme``."""

    primary: str
    secondary: str
    warning: str
    error: str
    success: str
    accent: str
    foreground: str


_CHRYS_COLORS: _ChrysColors = {
    "primary": "#AF87FF",
    "secondary": "#5F5FAF",
    "warning": "#FFAF5F",
    "error": "#FF5F5F",
    "success": "#5FFF87",
    "accent": "#FF87D7",
    "foreground": "#EEEEEE",
}
CHRYS_COLOR_NAMES = tuple(_CHRYS_COLORS)

# Chrys invents these slots — textual themes don't generate them.
# Exposed as a single dict because:
# - ``ChrysApp.get_theme_variable_defaults`` registers them as App-level
#   fallbacks so non-chrys theme switches can parse the ``$<chrys-only>``
#   refs in chrys.tcss without crashing.
# - playground/theme_editor.py uses ``in`` to bucket chrys-only rows.
CHRYS_ONLY_VARIABLES: dict[str, str] = {
    "control-background": "#303030",
    "overlay-background": "#3A3A3A",
    "markdown-block-background": "#303030",
}

# CSS variables that must remain resolvable with every Textual theme.
TUI_VARIABLE_DEFAULTS: dict[str, str] = {
    **CHRYS_ONLY_VARIABLES,
    "border-opacity": "80%",
}

_CHRYS_DARK_BORDER_COLOR = "#5F5F5F"
_CHRYS_DARK_TOOL_GROUP_TITLE_COLOR = "#949494"
_BORDER_COLOR_VARIABLE_SOURCES: dict[str, str] = {
    "tui-border-accent": "accent",
    "tui-border-block-hover-background": "block-hover-background",
    "tui-border-control-background": "control-background",
    "tui-border-error": "error",
    "tui-border-foreground": "foreground",
    "tui-border-primary": "primary",
    "tui-border-primary-darken-2": "primary-darken-2",
    "tui-border-primary-darken-3": "primary-darken-3",
    "tui-border-secondary": "secondary",
    "tui-border-success": "success",
    "tui-border-warning": "warning",
    "tui-border-warning-darken-1": "warning-darken-1",
}
_BORDER_COLOR_LITERAL_DEFAULTS: dict[str, str] = {
    "tui-border-neutral-100": "rgb(100, 100, 100)",
    "tui-border-neutral-128": "#808080",
    "tui-border-neutral-160": "rgb(160, 160, 160)",
    "tui-border-neutral-gray": "gray",
}
_SEMANTIC_BORDER_COLOR_VARIABLE_SOURCES: dict[str, str] = {
    "tui-border-agent-message": "success",
    "tui-border-status-error": "error",
    "tui-border-status-warning": "warning",
    "tui-border-user-message": "accent",
}
_BORDER_TITLE_COLOR_VARIABLE_SOURCES: dict[str, str] = {
    "tui-border-title-accent": "accent",
    "tui-border-title-primary": "primary",
    "tui-border-title-warning": "warning",
}


def with_tui_css_variables(theme_name: str, variables: dict[str, str]) -> dict[str, str]:
    """Add theme-aware border colors to a resolved Textual variable map."""
    resolved = dict(variables)
    border_override = _CHRYS_DARK_BORDER_COLOR if theme_name == CHRYS_DARK_THEME_NAME else None
    for alias, source in _BORDER_COLOR_VARIABLE_SOURCES.items():
        resolved[alias] = border_override or resolved[source]
    for alias, color in _BORDER_COLOR_LITERAL_DEFAULTS.items():
        resolved[alias] = border_override or color
    # Selected transcript chrome remains semantic even when chrys-dark makes
    # ordinary interface borders neutral gray.
    for alias, source in _SEMANTIC_BORDER_COLOR_VARIABLE_SOURCES.items():
        resolved[alias] = resolved[source]
    for alias, source in _BORDER_TITLE_COLOR_VARIABLE_SOURCES.items():
        resolved[alias] = resolved[source]
    resolved["tui-tool-group-title"] = (
        _CHRYS_DARK_TOOL_GROUP_TITLE_COLOR if theme_name == CHRYS_DARK_THEME_NAME else resolved["warning"]
    )
    return resolved


class TuiVariableDefaultsMixin(_TuiVariableDefaultsBase):
    """Supply Chrys-specific CSS variables to lightweight Textual hosts.

    Border alpha applies to RGB-backed themes. Textual deliberately preserves
    native ANSI palette tokens as opaque, so ``ansi-dark`` and ``ansi-light``
    ignore the percentage while ``chrys-ansi`` (RGB palette, ANSI output mode)
    honors it.
    """

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return TUI_VARIABLE_DEFAULTS

    def get_css_variables(self) -> dict[str, str]:
        variables = with_tui_css_variables(self.theme, super().get_css_variables())
        self.theme_variables = variables
        return variables


# Chrys overrides textual semantic variables where useful, plus the
# chrys-only slots spread in below.
_CHRYS_VARIABLES = {
    "button-color-foreground": "#303030",
    "border": "#AF87FF",
    "border-blurred": "#5F5FAF",
    "scrollbar": "#5F5FAF",
    "scrollbar-hover": "#AF87FF",
    "scrollbar-active": "#FF87D7",
    "scrollbar-background": "#1C1C1C",
    "scrollbar-background-hover": "#1C1C1C",
    "scrollbar-background-active": "#1C1C1C",
    "primary-muted": "#875FAF",
    "error-muted": "#D75F5F",
    "warning-muted": "#D7875F",
    "success-muted": "#5FD75F",
    "block-cursor-foreground": "#303030",
    "block-cursor-background": "#8787D7",
    "block-cursor-blurred-background": "#8787D7",
    "block-hover-background": "#4E4E4E",
    "footer-background": "#262626",
    "footer-foreground": "#EEEEEE",
    "footer-key-foreground": "#AF87FF",
    "footer-key-background": "transparent",
    "footer-description-foreground": "#D0D0D0",
    "footer-description-background": "transparent",
    "footer-item-background": "transparent",
    **CHRYS_ONLY_VARIABLES,
}

_CHRYS_DARK_VARIABLES = {
    **_CHRYS_VARIABLES,
    "scrollbar": "#585858",
    "scrollbar-hover": "#767676",
    "scrollbar-active": "#808080",
}

_CHRYS_ANSI_VARIABLES = {
    **_CHRYS_VARIABLES,
    "ansi-background": "ansi_black",
    "ansi-foreground": "ansi_white",
    "button-color-foreground": "ansi_black",
    "foreground-muted": "ansi_bright_black",
    "text-muted": "ansi_white 40%",
    "text-disabled": "ansi_bright_black",
    "markdown-h1-color": _CHRYS_COLORS["primary"],
    "markdown-h2-color": _CHRYS_COLORS["primary"],
    "markdown-h3-color": _CHRYS_COLORS["primary"],
    "markdown-h4-color": _CHRYS_COLORS["foreground"],
    "markdown-h5-color": _CHRYS_COLORS["foreground"],
    # Nearest 256-color approximations of Chrys alpha colors over _DARK_GRAY.
    # Keep these covered by tests/app/tui/behaviors/test_app.py so chrys-ansi remains close
    # to chrys without forcing the truecolor theme onto flat approximations.
    "markdown-h6-color": "#9E9E9E",
    "input-selection-background": "#5F5F87",
    "screen-selection-background": "#5F5F87",
}

_DARK_GRAY = "#1C1C1C"

CHRYS_THEME = Theme(
    name="chrys",
    **_CHRYS_COLORS,
    background=_DARK_GRAY,
    surface=_DARK_GRAY,
    panel=_DARK_GRAY,
    boost=_DARK_GRAY,
    dark=True,
    variables=_CHRYS_VARIABLES,
)

CHRYS_DARK_THEME = Theme(
    name=CHRYS_DARK_THEME_NAME,
    **_CHRYS_COLORS,
    background=_DARK_GRAY,
    surface=_DARK_GRAY,
    panel=_DARK_GRAY,
    boost=_DARK_GRAY,
    dark=True,
    variables=_CHRYS_DARK_VARIABLES,
)

CHRYS_ANSI_THEME = Theme(
    name="chrys-ansi",
    ansi=True,
    **_CHRYS_COLORS,
    background="ansi_default",
    surface="ansi_default",
    panel="ansi_default",
    boost="ansi_default",
    dark=True,
    variables=_CHRYS_ANSI_VARIABLES,
)
