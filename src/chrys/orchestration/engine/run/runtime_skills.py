# Copyright (c) 2026 Chrys. All rights reserved.

"""Runtime skill refresh helpers for main-agent turns."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from chrys.foundation.events.types import AgentRuntimeUpdated, Warning

if TYPE_CHECKING:
    from chrys.foundation.events.bus import EventBus
    from chrys.foundation.events.types import AgentRuntimeDetails, RuntimeSkillDetails
    from chrys.service.agent_middleware.system_reminder import SystemReminderMiddleware
    from chrys.service.skills.model import SkillProviderWarning
    from chrys.service.skills.provider import ChrysSkillsProvider, StagedSkillRefresh


logger = logging.getLogger(__name__)


class RuntimeSkillHost(Protocol):
    """Engine host contract needed for runtime skill refresh."""

    _bus: EventBus
    _session_id: str | None
    _skills_provider: ChrysSkillsProvider | None
    _tool_names: list[str]
    _skill_names: list[str]
    _memory_files: list[str]
    _runtime_details: AgentRuntimeDetails
    _reminder_middleware: SystemReminderMiddleware | None


@dataclass(frozen=True)
class StagedRuntimeSkillRefresh:
    """Runtime skill refresh staged before an active-injection owner is revalidated."""

    provider: ChrysSkillsProvider
    context: StagedSkillRefresh
    warnings: list[SkillProviderWarning]
    skill_names: list[str]
    skill_sources: dict[str, list[str]]
    skill_details: list[RuntimeSkillDetails]
    skill_catalog: str | None


class RuntimeSkillRefresher:
    """Refresh runtime skills through the engine host."""

    def __init__(self, host: RuntimeSkillHost) -> None:
        self._host = host

    async def refresh(self, *, update_active_turn: bool = False) -> None:
        """Refresh runtime skills and optionally update the active reminder snapshot."""
        host = self._host
        provider = host._skills_provider
        if provider is None:
            return
        try:
            warnings = await provider.refresh_context()
        except Exception:
            logger.exception("Failed to refresh runtime skills")
            return
        for warning in warnings:
            await host._bus.publish(Warning(code=warning.code, message=warning.message, session_id=host._session_id))
        host._skill_names = provider.skill_names()
        host._runtime_details.skill_sources = provider.skill_sources()
        host._runtime_details.skill_details = provider.skill_details()
        await self._publish_runtime_update(session_id=host._session_id)
        if update_active_turn and host._reminder_middleware is not None:
            host._reminder_middleware.update_skill_catalog_for_active_turn()

    async def stage_refresh(self) -> StagedRuntimeSkillRefresh | None:
        """Discover runtime skills without mutating live provider or host runtime state."""
        host = self._host
        provider = host._skills_provider
        if provider is None:
            return None
        try:
            staged = await provider.stage_context_refresh()
        except Exception:
            logger.exception("Failed to stage runtime skill refresh")
            return None
        return StagedRuntimeSkillRefresh(
            provider=provider,
            context=staged,
            warnings=list(staged.warnings),
            skill_names=staged.skill_names(),
            skill_sources=staged.skill_sources(),
            skill_details=staged.skill_details(),
            skill_catalog=staged.render_catalog_reminder(),
        )

    def commit_staged_refresh(
        self,
        staged: StagedRuntimeSkillRefresh | None,
    ) -> list[SkillProviderWarning]:
        """Commit a staged runtime skill refresh synchronously to the captured owner."""
        if staged is None:
            return []
        warnings = staged.provider.commit_context_refresh(staged.context)
        host = self._host
        host._skill_names = list(staged.skill_names)
        host._runtime_details.skill_sources = {source: list(names) for source, names in staged.skill_sources.items()}
        host._runtime_details.skill_details = list(staged.skill_details)
        return warnings

    async def publish_committed_refresh(
        self,
        staged: StagedRuntimeSkillRefresh | None,
        *,
        warnings: list[SkillProviderWarning],
        session_id: str | None,
    ) -> None:
        """Publish warning/runtime update events for a committed staged refresh."""
        if staged is None:
            return
        host = self._host
        for warning in warnings:
            await host._bus.publish(Warning(code=warning.code, message=warning.message, session_id=session_id))
        await self._publish_runtime_update(session_id=session_id)

    async def _publish_runtime_update(self, *, session_id: str | None) -> None:
        """Publish the current runtime catalog."""
        host = self._host
        await host._bus.publish(
            AgentRuntimeUpdated(
                session_id=session_id,
                model_profile_id=host._runtime_details.model.profile_id,
                max_context_tokens=host._runtime_details.model.max_context_tokens,
                tool_names=list(host._tool_names),
                skill_names=list(host._skill_names),
                memory_files=list(host._memory_files),
                runtime_details=copy.deepcopy(host._runtime_details),
            )
        )
