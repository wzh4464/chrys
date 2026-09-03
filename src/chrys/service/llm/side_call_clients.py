# Copyright (c) 2026 Chrys. All rights reserved.

"""One model client per profile, shared by a session's side calls.

A long-horizon turn makes several model calls that are not the main agent's:
the routing tiebreaker, the localization search, the clarification proposals.
Each creating its own client would open its own connection pool and its own
provider session for what is usually the same model profile, so they share one
here and it closes with the session.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from chrys.service.llm.clients import create_client
from chrys.service.profiles.models.schema import ModelProfile

logger = logging.getLogger(__name__)


class SideCallClientCache:
    """Session-scoped client reuse, keyed by model profile id."""

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}

    def get(
        self,
        profile: ModelProfile,
        *,
        session_id: str | None = None,
        parent_session_id: str | None = None,
        session_dir: Path | None = None,
    ) -> Any:
        """Return the shared client for *profile*, creating it on first use."""
        existing = self._clients.get(profile.id)
        if existing is not None:
            return existing
        client = create_client(
            profile,
            session_id=session_id,
            parent_session_id=parent_session_id,
            session_dir=session_dir,
            use_route_session_context=True,
        )
        self._clients[profile.id] = client
        return client

    async def close(self) -> None:
        """Close every client this session opened.

        Failures are logged, never raised: this runs during shutdown, where one
        unreachable provider must not stop the rest from being released.
        """
        clients, self._clients = list(self._clients.values()), {}
        for client in clients:
            close = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close is None:
                continue
            try:
                result = close()
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.debug("side-call client close failed", exc_info=True)
