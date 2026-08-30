# Copyright (c) 2026 Chrys. All rights reserved.

"""Shared semantic messages for clipboard toast titles."""

from chrys.foundation.i18n import msg

COPY_TITLE = msg("tui.copy.title.action", fallback="Copy")
COPIED_TITLE = msg("tui.copy.title.success", fallback="Copied")

__all__ = ["COPIED_TITLE", "COPY_TITLE"]
