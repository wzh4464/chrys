# Copyright (c) 2026 Chrys. All rights reserved.

"""Fresh, read-only Chrys agents used by proposal and selector passes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from chrys.foundation.models.session_env import SessionEnvironment
from chrys.kernel import CONVERSATION_HANDLE_KEYS, Agent
from chrys.service.llm.clients import create_client
from chrys.service.llm.route_sessions import derive_llm_route_session_id
from chrys.service.profiles.models.options import effective_chat_options
from chrys.service.profiles.models.schema import ModelProfile
from chrys.service.requirement_clarification.prompts import (
    proposal_instructions,
    selector_instructions,
)
from chrys.service.requirement_clarification.snapshot import WorkspaceSnapshot
from chrys.service.requirement_clarification.types import ClarificationProposal, ClarificationSelection
from chrys.service.tools.registry import ToolRegistry

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class ClarificationModelRunner(Protocol):
    """Model boundary consumed by :class:`ClarificationService`."""

    async def propose(self, prompt: str, *, sample_index: int) -> tuple[ClarificationProposal, dict[str, object]]: ...

    async def select(self, prompt: str) -> tuple[ClarificationSelection, dict[str, object]]: ...


def _stateless_options[ResponseT: BaseModel](
    profile: ModelProfile,
    response_format: type[ResponseT],
) -> dict[str, Any]:
    options = dict(effective_chat_options(profile) or {})
    options["response_format"] = response_format
    if "store" in options or profile.provider == "openai":
        options["store"] = False
    for key in CONVERSATION_HANDLE_KEYS:
        options.pop(key, None)
    options.pop("continuation_token", None)
    options.pop("background", None)
    extra_body = options.get("extra_body")
    if isinstance(extra_body, Mapping):
        clean_extra = {
            key: value
            for key, value in extra_body.items()
            if key not in {*CONVERSATION_HANDLE_KEYS, "continuation_token", "background"}
        }
        if "store" in clean_extra:
            clean_extra["store"] = False
        options["extra_body"] = clean_extra
    return options


class ChrysClarificationModel:
    """Run each clarification pass as a fresh, tool-bounded Chrys agent."""

    def __init__(
        self,
        *,
        profile: ModelProfile,
        snapshot: WorkspaceSnapshot,
        session_id: str | None,
        session_dir,
        report_usage: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        self._profile = profile
        self._snapshot = snapshot
        self._session_id = session_id
        self._session_dir = session_dir
        self._report_usage = report_usage

    async def propose(self, prompt: str, *, sample_index: int) -> tuple[ClarificationProposal, dict[str, object]]:
        return await self._run(
            prompt,
            response_format=ClarificationProposal,
            instructions=proposal_instructions(),
            route_kind="requirement-clarification-proposal",
            route_part=str(sample_index),
        )

    async def select(self, prompt: str) -> tuple[ClarificationSelection, dict[str, object]]:
        return await self._run(
            prompt,
            response_format=ClarificationSelection,
            instructions=selector_instructions(),
            route_kind="requirement-clarification-selector",
            route_part="1",
        )

    async def _run(
        self,
        prompt: str,
        *,
        response_format: type[ResponseT],
        instructions: str,
        route_kind: str,
        route_part: str,
    ) -> tuple[ResponseT, dict[str, object]]:
        route_session_id = derive_llm_route_session_id(
            self._session_id,
            route_kind=route_kind,
            route_parts=(self._snapshot.snapshot_id, route_part),
            model_profile=self._profile,
        )
        client = create_client(
            self._profile,
            session_id=route_session_id,
            parent_session_id=self._session_id,
            session_dir=self._session_dir,
        )
        runtime = SessionEnvironment.capture(
            session_id=route_session_id,
            workspace=self._snapshot.clarification_workspace(),
        )
        registry = ToolRegistry(vision_enabled=False)
        tools = registry.load_builtins(
            ["filesystem.read", "search"],
            runtime=runtime,
            session_id=route_session_id,
            session_dir=self._session_dir,
        )
        roots = "\n".join(
            f"- {root.view_root} ({'primary' if root.is_primary else 'additional'})" for root in self._snapshot.roots
        )
        agent = Agent(
            client=client,
            name="ChrysRequirementClarifier",
            instructions=(
                instructions
                + "\n\nFrozen workspace roots:\n"
                + roots
                + "\nUse only read_file, view_image, grep, and glob. Never address a path outside these roots."
            ),
            tools=tools,
        )
        await agent.__aenter__()
        try:
            response = await agent.run(
                prompt,
                stream=False,
                options=_stateless_options(self._profile, response_format),
            )
            value = response.value
            if not isinstance(value, response_format):
                raise ValueError(f"clarification side call returned no {response_format.__name__}")
            usage = dict(response.usage_details or {})
            if usage and self._report_usage is not None:
                self._report_usage(usage)
            return value, usage
        finally:
            await agent.__aexit__(None, None, None)
