# Copyright (c) 2026 Chrys. All rights reserved.

"""Desktop notification support for the TUI."""

from chrys.app.tui.notifications.service import NotificationService
from chrys.app.tui.notifications.settings import NotificationEvent, NotificationSettings

__all__ = [
    "NotificationEvent",
    "NotificationService",
    "NotificationSettings",
]
