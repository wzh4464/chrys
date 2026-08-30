# Copyright (c) 2026 Chrys. All rights reserved.

"""Patch: use empty style for block border outer when parent bg is terminal default.

Problem
-------
The ``block`` border type uses half-block characters (▄/▀) whose "other
half" is rendered with the parent widget's background.  When the parent
background is ``ansi_default``, Textual emits ``Style(bgcolor=default)``
which Rich renders as ``\\x1b[49m`` (SGR default background).  Some
terminals render this as opaque black instead of true transparent for
half-block characters, creating visible black lines in block borders.

Solution
--------
When ``base_background`` is ``ansi_default`` (ansi == -1) or fully
transparent (a == 0), return an empty ``Style()`` for the outer style.
An empty style has no bgcolor at all — Rich emits no background escape
code, letting the terminal render its native background in those cells.
"""

from chrys.foundation.patches.patcher import FilePatch, register

_OLD = """\
    def get_inner_outer(
        cls, base_background: Color, background: Color
    ) -> tuple[Style, Style]:
        \"\"\"Get inner and outer background colors.\"\"\"
        return (
            Style(background=base_background + background),
            Style(background=base_background),
        )"""

_NEW = """\
    def get_inner_outer(
        cls, base_background: Color, background: Color
    ) -> tuple[Style, Style]:
        \"\"\"Get inner and outer background colors.\"\"\"
        # --- Chrys patch: transparent parent → empty outer style ---
        # When base_background is ansi_default or transparent, emit no
        # bgcolor so the terminal renders its native background.
        is_default = getattr(base_background, "ansi", None) == -1
        outer = Style() if (is_default or base_background.a == 0) else Style(background=base_background)
        return (
            Style(background=base_background + background),
            outer,
        )"""

register(
    FilePatch(
        package="textual",
        module_file="_styles_cache.py",
        old_fragment=_OLD,
        new_fragment=_NEW,
        description="Use empty style for block border outer on transparent terminals",
    ),
)
