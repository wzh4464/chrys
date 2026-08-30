# Copyright (c) 2026 Chrys. All rights reserved.

"""Buddy lifecycle callbacks for application entrypoints."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def on_successful_turn() -> None:
    """Persist Buddy progress after a successful Chrys turn."""
    try:
        from chrys.app.features.buddy.config import has_hatched, increment_message_count

        if has_hatched():
            increment_message_count()
    except Exception:
        logger.debug("Failed to persist buddy message count", exc_info=True)
