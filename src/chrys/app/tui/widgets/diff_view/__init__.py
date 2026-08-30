# Copyright (c) 2026 Chrys. All rights reserved.

"""DiffView widget package for Textual.

Provides a high-performance diff renderer supporting both split and unified formats,
with syntax highlighting, character-level diff highlighting, and virtualized scrolling.

Adapted from toad (https://github.com/willmcgugan/toad) — original author: Will McGugan, License: MIT.
"""

from chrys.app.tui.widgets.diff_view.annotations import AnnotationsColumn
from chrys.app.tui.widgets.diff_view.code import CodeColumn
from chrys.app.tui.widgets.diff_view.protocols import SupportsPrepare
from chrys.app.tui.widgets.diff_view.widget import DiffView

__all__ = [
    "AnnotationsColumn",
    "CodeColumn",
    "DiffView",
    "SupportsPrepare",
]
