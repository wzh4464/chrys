# Copyright (c) 2026 Chrys. All rights reserved.

"""Minimal one-session ACP shell for the Chrys-PACT Campaign agent."""

from __future__ import annotations

import json
import logging
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from acp import PROTOCOL_VERSION
from acp import schema as acp_schema
from acp.exceptions import RequestError
from acp.helpers import ContentBlock, start_tool_call, update_agent_message_text
from acp.interfaces import Client

from chrys import __version__
from chrys.app.acp.bridge import SessionUpdate
from chrys.pact.campaign import CampaignCancelled, CampaignCoordinator, CampaignTerminal, UpdateSender

if TYPE_CHECKING:
    from chrys.foundation.config.settings_store import LoadedSettings

logger = logging.getLogger(__name__)

_RUN_REQUEST_SCHEMA = "chrys-pact/run-request/v1"
_RUN_REQUEST_KEYS = frozenset(("schema", "contract_path", "plan_path"))
# How long a campaign gets to settle after its client has gone. Long enough for
# an ordinary teardown, far short of the 1800 s a verify command may take -- and
# nothing is left to receive that result anyway.
_SHUTDOWN_GRACE_SECONDS = 30.0


class LaunchContractError(ValueError):
    """The Primary Chrys launch envelope is not the strict v1 contract."""


@dataclass(frozen=True, slots=True)
class LaunchRequest:
    """Validated absolute inputs for one PACT Campaign."""

    contract_file: Path
    plan_file: Path


class Coordinator(Protocol):
    """Injectable Campaign lifecycle used by the ACP shell."""

    async def run(
        self,
        *,
        workspace: Path,
        contract_file: Path,
        plan_file: Path,
        send_update: UpdateSender,
    ) -> CampaignTerminal: ...

    async def cancel(self) -> None: ...

    async def wait_closed(self, timeout: float | None = None) -> bool: ...


class CoordinatorFactory(Protocol):
    """Build the single coordinator after a session claims the process."""

    def __call__(self) -> Coordinator: ...


@dataclass(slots=True)
class _Session:
    session_id: str
    workspace: Path
    coordinator: Coordinator
    closed: bool = False


class ChrysPactServer:
    """Expose exactly one PACT Campaign as an external ACP agent."""

    def __init__(self, coordinator_factory: CoordinatorFactory) -> None:
        self._coordinator_factory = coordinator_factory
        self._client: Client | None = None
        self._session: _Session | None = None
        self._session_claimed = False
        self._prompt_started = False

    def on_connect(self, conn: Client) -> None:
        """Capture the ACP client used for standard session updates."""
        self._client = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: acp_schema.ClientCapabilities | None = None,
        client_info: acp_schema.Implementation | None = None,
        **kwargs: Any,
    ) -> acp_schema.InitializeResponse:
        """Advertise only text prompts and session close."""
        _ = client_capabilities, client_info, kwargs
        return acp_schema.InitializeResponse(
            protocolVersion=min(protocol_version, PROTOCOL_VERSION),
            agentInfo=acp_schema.Implementation(name="chrys-pact", title="Chrys PACT", version=__version__),
            agentCapabilities=acp_schema.AgentCapabilities(
                auth=None,
                loadSession=False,
                mcpCapabilities=None,
                promptCapabilities=acp_schema.PromptCapabilities(
                    embeddedContext=False,
                    image=False,
                    audio=False,
                ),
                sessionCapabilities=acp_schema.SessionCapabilities(close=acp_schema.SessionCloseCapabilities()),
            ),
        )

    async def new_session(
        self,
        cwd: str | None = None,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> acp_schema.NewSessionResponse:
        """Claim this process for one workspace and one Campaign."""
        _ = kwargs
        if self._session_claimed:
            raise RequestError.invalid_params({"details": "chrys-pact accepts exactly one ACP session per process."})
        if additional_directories:
            raise RequestError.invalid_params({"details": "additional directories are not supported by chrys-pact v1."})
        if mcp_servers:
            raise RequestError.invalid_params({"details": "client-supplied MCP servers are not supported."})
        try:
            workspace = _resolve_workspace(cwd)
        except LaunchContractError as exc:
            raise RequestError.invalid_params({"details": str(exc)}) from exc

        self._session_claimed = True
        session_id = f"chrys-pact-session-{uuid.uuid4().hex}"
        self._session = _Session(
            session_id=session_id,
            workspace=workspace,
            coordinator=self._coordinator_factory(),
        )
        return acp_schema.NewSessionResponse(sessionId=session_id)

    async def prompt(
        self,
        prompt: list[ContentBlock],
        session_id: str,
        message_id: str | None = None,
        **kwargs: Any,
    ) -> acp_schema.PromptResponse:
        """Validate and run the process's sole launch prompt."""
        _ = kwargs
        session = self._active_session(session_id)
        if self._prompt_started:
            raise RequestError.invalid_params({"details": "chrys-pact accepts exactly one prompt per process."})
        self._prompt_started = True
        if self._client is None:
            raise RequestError.internal_error({"details": "ACP client connection is not ready."})

        try:
            launch = parse_launch_request(prompt, workspace=session.workspace)
        except LaunchContractError as exc:
            await self._send_update(session_id, update_agent_message_text(f"PACT launch refused: {exc}"))
            return acp_schema.PromptResponse(stopReason="refusal", userMessageId=message_id)

        try:
            terminal = await session.coordinator.run(
                workspace=session.workspace,
                contract_file=launch.contract_file,
                plan_file=launch.plan_file,
                send_update=lambda update: self._send_update(session_id, update),
            )
        except CampaignCancelled as exc:
            await self._send_update(session_id, update_agent_message_text(str(exc)))
            return acp_schema.PromptResponse(stopReason="cancelled", userMessageId=message_id)
        except Exception as exc:
            detail = (str(exc) or type(exc).__name__)[:1_000]
            await self._send_update(session_id, update_agent_message_text(f"PACT Campaign failed: {detail}"))
            return acp_schema.PromptResponse(stopReason="refusal", userMessageId=message_id)

        # ToolCallStart seals the stock Primary translator's previous prose
        # segment. ToolCallProgress would merge, causing last_segment mode to
        # append the last role response to the canonical Campaign summary.
        summary = terminal.summary_text()
        await self._send_update(
            session_id,
            start_tool_call(
                f"{terminal.campaign_id}/result",
                "PACT Campaign result",
                kind="think",
                status="completed" if terminal.completed else "failed",
                raw_output=summary,
            ),
        )
        await self._send_update(session_id, update_agent_message_text(summary))
        return acp_schema.PromptResponse(
            # A canonical blocked/active result is a normal ACP turn result, not a
            # protocol refusal. The final summary carries the authoritative status.
            stopReason="end_turn",
            userMessageId=message_id,
        )

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """Best-effort cancel the active role turn without inventing Work State."""
        _ = kwargs
        session = self._session
        if session is None or session.session_id != session_id or session.closed:
            return
        await session.coordinator.cancel()

    async def close_session(self, session_id: str, **kwargs: Any) -> acp_schema.CloseSessionResponse:
        """Close the sole session and interrupt any active role turn."""
        _ = kwargs
        session = self._active_session(session_id)
        session.closed = True
        await session.coordinator.cancel()
        return acp_schema.CloseSessionResponse()

    async def shutdown(self) -> None:
        """Release the active invocation when the ACP transport stops."""
        session = self._session
        if session is not None and not session.closed:
            session.closed = True
            await session.coordinator.cancel()
        if session is not None and not await session.coordinator.wait_closed(_SHUTDOWN_GRACE_SECONDS):
            # The control-plane thread is a daemon, so returning here lets the
            # process exit rather than outliving its client by a verify timeout.
            logger.warning(
                "PACT control plane did not settle within %.0fs of transport shutdown", _SHUTDOWN_GRACE_SECONDS
            )

    async def _send_update(self, session_id: str, update: SessionUpdate) -> None:
        client = self._client
        if client is None:
            raise RuntimeError("ACP client connection is not ready.")
        await client.session_update(session_id=session_id, update=update)

    def _active_session(self, session_id: str) -> _Session:
        session = self._session
        if session is None or session.session_id != session_id or session.closed:
            raise RequestError.invalid_params({"details": f"ACP session is not active: {session_id}"})
        return session


def parse_launch_request(prompt: list[ContentBlock], *, workspace: Path) -> LaunchRequest:
    """Parse one strict text-only run-request/v1 envelope and contain both files."""
    if len(prompt) != 1 or not isinstance(prompt[0], acp_schema.TextContentBlock):
        raise LaunchContractError("prompt must contain exactly one text block.")
    try:
        payload = json.loads(prompt[0].text)
    except json.JSONDecodeError as exc:
        raise LaunchContractError("prompt text must be one valid JSON object.") from exc
    if not isinstance(payload, dict):
        raise LaunchContractError("run request must be a JSON object.")
    keys = frozenset(payload)
    if keys != _RUN_REQUEST_KEYS:
        missing = sorted(_RUN_REQUEST_KEYS - keys)
        unknown = sorted(keys - _RUN_REQUEST_KEYS)
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown fields: {', '.join(unknown)}")
        raise LaunchContractError("; ".join(details))
    if payload["schema"] != _RUN_REQUEST_SCHEMA:
        raise LaunchContractError(f"schema must be {_RUN_REQUEST_SCHEMA!r}.")
    contract_path = _request_path(payload["contract_path"], label="contract_path", expected_name="goal-contract.json")
    plan_path = _request_path(payload["plan_path"], label="plan_path", expected_name="initial-plan.json")
    if contract_path.parent != plan_path.parent:
        raise LaunchContractError("contract_path and plan_path must use the same request directory.")
    return LaunchRequest(
        contract_file=_resolve_input_file(workspace, contract_path, label="contract_path"),
        plan_file=_resolve_input_file(workspace, plan_path, label="plan_path"),
    )


def _request_path(value: object, *, label: str, expected_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise LaunchContractError(f"{label} must be a non-empty string.")
    relative = Path(value)
    if relative.is_absolute():
        raise LaunchContractError(f"{label} must be workspace-relative.")
    parts = relative.parts
    if len(parts) != 4 or parts[:2] != (".pact-io", "chrys-pact") or parts[2] in {"", ".", ".."}:
        raise LaunchContractError(f"{label} must be under .pact-io/chrys-pact/<request-id>/.")
    if parts[3] != expected_name:
        raise LaunchContractError(f"{label} must end with {expected_name}.")
    return relative


def _resolve_input_file(workspace: Path, relative: Path, *, label: str) -> Path:
    root = workspace.resolve(strict=True)
    try:
        resolved = (root / relative).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LaunchContractError(f"{label} does not identify an existing file.") from exc
    if not resolved.is_relative_to(root):
        raise LaunchContractError(f"{label} resolves outside the session workspace.")
    try:
        mode = resolved.stat().st_mode
    except OSError as exc:
        raise LaunchContractError(f"{label} could not be inspected.") from exc
    if not stat.S_ISREG(mode):
        raise LaunchContractError(f"{label} must identify a regular file.")
    return resolved


def _resolve_workspace(cwd: str | None) -> Path:
    if not cwd:
        raise LaunchContractError("session/new must provide cwd.")
    try:
        workspace = Path(cwd).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise LaunchContractError(f"workspace does not exist: {cwd}") from exc
    if not workspace.is_dir():
        raise LaunchContractError(f"workspace is not a directory: {workspace}")
    return workspace


def default_server(
    *,
    profile_name: str,
    loaded_settings: LoadedSettings,
    verify_command: str | None,
    allow_unverified: bool,
    max_rounds: int = 3,
) -> ChrysPactServer:
    """Build the production server while keeping tests dependency-injected."""
    return ChrysPactServer(
        lambda: CampaignCoordinator(
            profile_name=profile_name,
            loaded_settings=loaded_settings,
            verify_command=verify_command,
            allow_unverified=allow_unverified,
            max_rounds=max_rounds,
        )
    )
