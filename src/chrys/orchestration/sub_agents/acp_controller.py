# Copyright (c) 2026 Chrys. All rights reserved.

"""ACP sub-agent orchestration, update translation, and human interaction."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hashlib
import json
import logging
import math
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from acp.exceptions import RequestError
from acp.schema import (
    AgentMessageChunk,
    AudioContentBlock,
    ContentToolCallContent,
    EmbeddedResourceContentBlock,
    FileEditToolCallContent,
    ImageContentBlock,
    PermissionOption,
    ResourceContentBlock,
    SessionNotification,
    TerminalToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
    ToolCallUpdate,
    UsageUpdate,
)

from chrys.foundation.events.types import (
    ApprovalCancelled,
    ApprovalRequest,
    ApprovalResponse,
    AskUserResponse,
    AskUserTimedOut,
    QuestionToUser,
    SubAgentAborted,
    SubAgentCascadeAborted,
    SubAgentPaused,
    SubAgentProgress,
    SubAgentResumed,
    SubAgentRetryAttempt,
    SubAgentToolCallResult,
    SubAgentToolCallStart,
)
from chrys.foundation.platform.files import atomic_write_owner_only_text
from chrys.foundation.text.images import MAX_IMAGE_BYTES
from chrys.foundation.tool_result_metadata import TOOL_FAILED_METADATA_KEY
from chrys.foundation.trajectory.context import TrajectoryContext
from chrys.foundation.trajectory.event_types import RetryMode, RetryReason, WaitCategory
from chrys.kernel import Content
from chrys.orchestration.sub_agents.controller import SubAgentFailureReason, SubAgentStatus
from chrys.service.acp_client import AcpAgentClient
from chrys.service.acp_client.errors import (
    AcpAuthRequiredError,
    AcpConfigError,
    AcpConnectError,
    AcpIdleTimeoutError,
    AcpRefusalError,
    AcpSpawnError,
    AcpTransportError,
)
from chrys.service.acp_client.spec import AcpAgentSpec, AcpPromptUsage, PermissionDecision
from chrys.service.approval.arbitration import ApprovalDecisionArbiter, ApprovalJudgeInput
from chrys.service.approval.correlation import OneShotCorrelation
from chrys.service.approval.policy import ApprovalMode
from chrys.service.tools.result_metadata import record_tool_success, tool_error
from chrys.service.trajectory.approvals import ApprovalDecider, ApprovalTrace
from chrys.service.trajectory.retries import RetryBackoffTrace
from chrys.service.trajectory.waits import WaitOutcome, WaitTrace

if TYPE_CHECKING:
    from collections.abc import Awaitable, Coroutine

    from chrys.foundation.events.bus import EventBus
    from chrys.service.approval.judge import ApprovalJudge
    from chrys.service.approval.turn_context import TurnContextReader

logger = logging.getLogger(__name__)

_MAX_CALLS = 5_000
_MAX_FIELD_CHARS = 2_000
_MAX_VALUE_DEPTH = 16
_MAX_COLLECTION_ITEMS = 256
_AUDIT_RING_ITEMS = 500
_AUDIT_RING_BYTES = 256 * 1024
_BACKOFF = (3, 7, 15, 30, 60)
_ALLOW_KINDS = ("allow_once", "allow_always")
_REJECT_KINDS = ("reject_once", "reject_always")
_ASK_META_COLLISIONS = frozenset({"options", "sessionId", "session_id", "requestId", "question", "callerName"})
_MAX_BASE64_IMAGE_CHARS = ((MAX_IMAGE_BYTES + 2) // 3) * 4
# A maximum-size accepted image expands to 4 MiB in base64. Keep one
# additional MiB for its data-URI prefix and the surrounding bounded record.
_MAX_ASSEMBLED_BYTES = _MAX_BASE64_IMAGE_CHARS + 1024 * 1024


def display_tool_kind(kind: str | None) -> str:
    """Map ACP presentation kinds to Chrys display/judge kinds."""
    if kind == "execute":
        return "shell"
    if kind == "read":
        return "filesystem.read"
    if kind in {"edit", "delete", "move"}:
        return "filesystem.write"
    if kind == "search":
        return "search"
    return ""


def _preview(value: Any, *, limit: int = _MAX_FIELD_CHARS) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            text = str(value)
    if len(text) <= limit:
        return text
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"{text[: limit - 28]}… [sha256:{digest}]"


def _bounded_value(value: Any, *, depth: int = 0) -> Any:
    """Return a JSON-safe bounded value before it enters events or persistence."""
    if depth >= _MAX_VALUE_DEPTH:
        return "[depth limit]"
    if value is None or type(value) is bool:
        return value
    if type(value) is int:
        rendered = str(value)
        return value if len(rendered) <= _MAX_FIELD_CHARS else _preview(rendered)
    if type(value) is float:
        return value if math.isfinite(value) else _preview(value)
    if isinstance(value, str):
        return _preview(value)
    if hasattr(value, "model_dump"):
        return _bounded_value(value.model_dump(mode="json", by_alias=True), depth=depth)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_COLLECTION_ITEMS:
                result["…"] = "[item limit]"
                break
            result[_preview(key, limit=200)] = _bounded_value(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [_bounded_value(item, depth=depth + 1) for item in value[:_MAX_COLLECTION_ITEMS]]
    return _preview(value)


def canonical_raw_input(value: Any) -> dict[str, Any]:
    bounded = _bounded_value(value)
    if isinstance(bounded, dict):
        return bounded
    return {"input": _preview(bounded)}


def _raw_args_dict(value: Any) -> dict[str, Any]:
    """Dict-shaped raw arguments for approval/judge inputs — never truncated.

    The judge and the approval dialog must see the arguments the remote will
    execute (§9.2): a preview cap here would let a destructive suffix hide
    beyond the truncation point and be approved unseen. Size is already
    bounded upstream — permission frames ride the client's per-payload caps
    before SDK routing — so the untruncated form is safe to publish.
    canonical_raw_input stays for retained records and audit appends (§7.4).
    """
    if isinstance(value, Mapping):
        return dict(value)
    return {"input": value}


def _content_block_text(block: Any) -> str:
    if isinstance(block, TextContentBlock):
        return block.text
    if isinstance(block, ResourceContentBlock):
        return f"[resource: {block.uri}]"
    if isinstance(block, EmbeddedResourceContentBlock):
        resource = block.resource
        uri = getattr(resource, "uri", "")
        return f"[resource: {uri}]" if uri else "[resource]"
    if isinstance(block, ImageContentBlock | AudioContentBlock):
        return ""
    return _preview(block)


def _structured_tool_content(content: Sequence[Any]) -> tuple[list[Content], list[dict[str, Any]]]:
    """Extract images and resource descriptors before bounded text projection."""
    image_contents: list[Content] = []
    artifacts: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, ContentToolCallContent):
            continue
        block = item.content
        if isinstance(block, ImageContentBlock):
            media_type = block.mime_type.strip().lower() or "image/png"
            if block.data:
                if len(block.data) > _MAX_BASE64_IMAGE_CHARS:
                    raise AcpTransportError("The ACP agent returned an image that exceeds the supported size limit.")
                try:
                    data = base64.b64decode(block.data, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise AcpTransportError("The ACP agent returned invalid base64 image content.") from exc
                if not data:
                    raise AcpTransportError("The ACP agent returned empty image content.")
                if len(data) > MAX_IMAGE_BYTES:
                    raise AcpTransportError("The ACP agent returned an image that exceeds the supported size limit.")
                image_contents.append(Content.from_data(data=data, media_type=media_type))
            elif block.uri:
                image_contents.append(Content.from_uri(uri=block.uri, media_type=media_type))
            continue
        if isinstance(block, ResourceContentBlock):
            descriptor: dict[str, Any] = {"name": block.name, "path": block.uri}
            if block.mime_type:
                descriptor["mime"] = block.mime_type
            if block.size is not None:
                descriptor["size"] = block.size
            artifacts.append(descriptor)
            continue
        if isinstance(block, EmbeddedResourceContentBlock):
            resource = block.resource
            uri = resource.uri
            descriptor = {"name": uri.rstrip("/").rsplit("/", 1)[-1] or uri, "path": uri}
            if resource.mime_type:
                descriptor["mime"] = resource.mime_type
            artifacts.append(descriptor)
    return image_contents, artifacts


def _tool_content_text(content: Sequence[Any] | None) -> str:
    rendered: list[str] = []
    for item in content or ():
        if isinstance(item, ContentToolCallContent):
            text = _content_block_text(item.content)
        elif isinstance(item, FileEditToolCallContent):
            text = f"edited {item.path}"
        elif isinstance(item, TerminalToolCallContent):
            text = f"[terminal {item.terminal_id}]"
        else:
            text = _preview(item)
        if text:
            rendered.append(text)
    return _preview("\n".join(rendered))


@dataclass(slots=True)
class _CallRecord:
    local_call_id: str
    title: str = "tool"
    kind: str = ""
    raw_input: dict[str, Any] = field(default_factory=dict)
    raw_output: Any = None
    content: list[Any] = field(default_factory=list)
    image_contents: list[Content] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    locations: list[Any] = field(default_factory=list)
    status: str = ""
    started_at: float = field(default_factory=time.monotonic)
    start_published: bool = False
    terminal_published: bool = False
    last_start_signature: str = ""
    provider_hosted: bool = False
    hosted_family: str = ""
    provider: str = ""
    provider_item_type: str = ""
    provider_call_id: str = ""
    provider_status: str = ""


def _explicit_hosted_extension(update: ToolCallStart | ToolCallProgress) -> dict[str, Any]:
    """Read hosted presentation metadata only when the remote explicitly sends it."""
    metadata = update.field_meta or {}
    chrys_meta = metadata.get("chrys")
    if not isinstance(chrys_meta, Mapping):
        return {}
    family = chrys_meta.get("hosted_family", chrys_meta.get("hostedFamily"))
    provider_hosted = chrys_meta.get("provider_hosted", chrys_meta.get("providerHosted")) is True
    if not provider_hosted and not isinstance(family, str):
        return {}

    extension: dict[str, Any] = {"provider_hosted": provider_hosted}
    if isinstance(family, str):
        extension["hosted_family"] = family

    def _copy_text_value(target_key: str, snake_key: str, camel_key: str) -> None:
        value = chrys_meta.get(snake_key, chrys_meta.get(camel_key))
        if isinstance(value, str):
            extension[target_key] = value

    _copy_text_value("provider", "provider", "provider")
    _copy_text_value("provider_item_type", "provider_item_type", "providerItemType")
    _copy_text_value("provider_call_id", "provider_call_id", "providerCallId")
    _copy_text_value("provider_status", "provider_status", "providerStatus")
    return extension


class AcpUpdateTranslator:
    """Merge ACP partial updates into bounded Chrys events and final text."""

    def __init__(
        self,
        *,
        event_bus: EventBus | None,
        session_id: str | None,
        agent_name: str,
        invocation_id: str,
        attempt: int,
        result_mode: Literal["last_segment", "transcript"] = "last_segment",
        completed_before: int = 0,
        spend_before: int = 0,
        unreported_before: int = 0,
    ) -> None:
        self._bus = event_bus
        self._session_id = session_id
        self.agent_name = agent_name
        self.invocation_id = invocation_id
        self.attempt = attempt
        self.result_mode = result_mode
        self.calls: dict[str, _CallRecord] = {}
        self._completed_before = completed_before
        self._completed_here = 0
        self._spend = spend_before
        self._unreported = unreported_before
        self.latest_context_used = 0
        self.latest_context_size: int | None = None
        self._segments: list[str] = []
        self._open_segment = ""
        self._message_id: str | None = None
        self._retained_bytes = 0
        self._retained_image_bytes_by_call: dict[str, int] = {}
        self._audit: deque[dict[str, Any]] = deque()
        # Include the JSON list delimiters so the persisted ring itself,
        # rather than only the sum of its entries, stays within the cap.
        self._audit_bytes = 2

    @property
    def completed_count(self) -> int:
        return self._completed_before + self._completed_here

    @property
    def translated_updates(self) -> list[dict[str, Any]]:
        return list(self._audit)

    async def put(self, seq: int, notification: SessionNotification) -> None:
        await self.on_update(seq, notification)

    async def on_update(self, seq: int, notification: SessionNotification) -> None:
        update = notification.update
        self._append_audit(seq, update)
        if isinstance(update, ToolCallStart | ToolCallProgress):
            # Only a ToolCallStart seals the open segment (§7.5): progress
            # merges tool state and must not split an assistant message that
            # keeps streaming around a running call — last_segment would
            # otherwise return just the suffix.
            if isinstance(update, ToolCallStart):
                self._seal_segment()
            await self._merge_tool(update)
        elif isinstance(update, AgentMessageChunk):
            self._consume_message(update)
        elif isinstance(update, UsageUpdate):
            # PR-1 validates raw exact-int usage before SDK coercion. Keep this
            # defensive check for direct translator tests and custom sinks.
            if type(update.used) is int and type(update.size) is int and update.used >= 0 and update.size >= 0:
                self.latest_context_used = update.used
                self.latest_context_size = update.size
                await self.publish_progress()

    def _append_audit(self, seq: int, update: Any) -> None:
        self._append_audit_item({"seq": seq, "update": _bounded_value(update.model_dump(mode="json", by_alias=True))})

    def record_permission_request(self, *, title: str, kind: str, raw_input: Any, outcome: str) -> None:
        """Append a permission request to the audit ring.

        ACP permits ``session/request_permission`` before any ``ToolCallStart``;
        without this entry a standalone request's rawInput would reach no
        durable artifact (judge logging is deliberately ``log_dir=None``, §9.2
        — the owner-only envelope is the place remote rawInput is audited).
        """
        self._append_audit_item(
            {
                "seq": None,
                "update": {
                    "sessionUpdate": "permission_request",
                    "title": _preview(title),
                    "kind": _preview(kind),
                    "rawInput": _bounded_value(raw_input),
                    "outcome": outcome,
                },
            }
        )

    def _append_audit_item(self, item: dict[str, Any]) -> None:
        size = len(json.dumps(item, ensure_ascii=False, separators=(",", ":"), default=str).encode())
        if size > _AUDIT_RING_BYTES:
            item = {"seq": item.get("seq"), "update": _preview(item)}
            size = len(json.dumps(item, ensure_ascii=False).encode())
        added = size + (1 if self._audit else 0)
        while self._audit and (len(self._audit) >= _AUDIT_RING_ITEMS or self._audit_bytes + added > _AUDIT_RING_BYTES):
            removed = self._audit.popleft()
            removed_size = len(json.dumps(removed, ensure_ascii=False, separators=(",", ":")).encode())
            self._audit_bytes -= removed_size + (1 if self._audit else 0)
            added = size + (1 if self._audit else 0)
        if self._audit_bytes + added <= _AUDIT_RING_BYTES:
            self._audit.append(item)
            self._audit_bytes += added

    def _charge(self, value: Any) -> None:
        self._retained_bytes += len(json.dumps(value, ensure_ascii=False, default=str).encode())
        if self._retained_bytes > _MAX_ASSEMBLED_BYTES:
            raise AcpTransportError("The ACP agent exceeded the translated update budget.")

    def _charge_tool_update(self, call_id: str, value: Any, image_uris: Sequence[str]) -> None:
        update_bytes = len(json.dumps(value, ensure_ascii=False, default=str).encode())
        image_bytes = len(json.dumps(image_uris, ensure_ascii=False).encode()) if image_uris else 0
        previous_image_bytes = self._retained_image_bytes_by_call.get(call_id, 0)
        retained_bytes = self._retained_bytes - previous_image_bytes + image_bytes + update_bytes
        if retained_bytes > _MAX_ASSEMBLED_BYTES:
            raise AcpTransportError("The ACP agent exceeded the translated update budget.")
        self._retained_bytes = retained_bytes
        if image_bytes:
            self._retained_image_bytes_by_call[call_id] = image_bytes
        else:
            self._retained_image_bytes_by_call.pop(call_id, None)

    async def _merge_tool(self, update: ToolCallStart | ToolCallProgress) -> None:
        local_id = f"a{self.attempt}:{update.tool_call_id}"
        record = self.calls.get(local_id)
        if record is None:
            if len(self.calls) >= _MAX_CALLS:
                raise AcpTransportError("The ACP agent exceeded the tool-call update budget.")
            record = _CallRecord(local_call_id=local_id)
            self.calls[local_id] = record

        title = getattr(update, "title", None)
        kind = getattr(update, "kind", None)
        raw_input = getattr(update, "raw_input", None)
        raw_output = getattr(update, "raw_output", None)
        content = getattr(update, "content", None)
        locations = getattr(update, "locations", None)
        status = getattr(update, "status", None)
        if title is not None:
            record.title = _preview(title) or "tool"
        if kind is not None:
            record.kind = display_tool_kind(kind)
        if raw_input is not None:
            record.raw_input = canonical_raw_input(raw_input)
        if raw_output is not None:
            record.raw_output = _bounded_value(raw_output)
        if content is not None:
            record.image_contents, record.artifacts = _structured_tool_content(content)
            record.content = list(_bounded_value(content) or [])
        if locations is not None:
            record.locations = list(_bounded_value(locations) or [])
        if status is not None:
            record.status = status
        hosted_extension = _explicit_hosted_extension(update)
        if hosted_extension:
            record.provider_hosted = bool(hosted_extension["provider_hosted"])
            if "hosted_family" in hosted_extension:
                record.hosted_family = str(hosted_extension["hosted_family"])
            if "provider" in hosted_extension:
                record.provider = str(hosted_extension["provider"])
            if "provider_item_type" in hosted_extension:
                record.provider_item_type = str(hosted_extension["provider_item_type"])
            if "provider_call_id" in hosted_extension:
                record.provider_call_id = str(hosted_extension["provider_call_id"])
            if "provider_status" in hosted_extension:
                record.provider_status = str(hosted_extension["provider_status"])
        self._charge_tool_update(
            record.local_call_id,
            {
                "title": record.title,
                "kind": record.kind,
                "input": record.raw_input,
                "output": record.raw_output,
                "content": record.content,
                "artifacts": record.artifacts,
                "locations": record.locations,
            },
            [image.uri or "" for image in record.image_contents],
        )

        await self._publish_start(record)
        if record.status in {"completed", "failed"}:
            await self._publish_terminal(record)

    async def _publish_start(self, record: _CallRecord, *, ensure: bool = False) -> None:
        signature = json.dumps(
            [record.title, record.kind, record.raw_input],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if not ensure and (
            record.terminal_published or (record.start_published and signature == record.last_start_signature)
        ):
            return
        record.start_published = True
        record.last_start_signature = signature
        if self._bus is not None:
            await self._bus.publish(
                SubAgentToolCallStart(
                    agent_name=self.agent_name,
                    invocation_id=self.invocation_id,
                    tool_name=record.title,
                    tool_kind=record.kind,
                    args=dict(record.raw_input),
                    call_id=record.local_call_id,
                    provider_hosted=record.provider_hosted,
                    hosted_family=record.hosted_family,
                    provider=record.provider,
                    provider_item_type=record.provider_item_type,
                    provider_call_id=record.provider_call_id,
                    provider_status=record.provider_status,
                    session_id=self._session_id,
                )
            )

    async def _publish_terminal(self, record: _CallRecord, *, interrupted: bool = False) -> None:
        if record.terminal_published:
            return
        # Claim the terminal synchronously, before any await. The live update
        # consumer and a cancel-path flush_interrupted() run as separate tasks;
        # if the guard above and the claim below straddled an await (the
        # _publish_start / bus.publish points), both could pass the guard and
        # double-deliver a SubAgentToolCallResult (and double-count). Claiming
        # up front serializes them — the loser returns at the guard. The
        # ensure=True start below intentionally bypasses this same flag.
        record.terminal_published = True
        self._completed_here += 1
        try:
            await self._publish_start(record, ensure=True)
            if interrupted:
                result = "(interrupted)"
                failed = True
            else:
                result = _tool_content_text_from_bounded(record.content)
                if not result:
                    result = _preview(record.raw_output)
                failed = record.status == "failed"
            if self._bus is not None:
                await self._bus.publish(
                    SubAgentToolCallResult(
                        agent_name=self.agent_name,
                        invocation_id=self.invocation_id,
                        tool_name=record.title,
                        call_id=record.local_call_id,
                        result=result,
                        image_contents=list(record.image_contents),
                        duration_ms=max(0, int((time.monotonic() - record.started_at) * 1_000)),
                        metadata={TOOL_FAILED_METADATA_KEY: True} if failed else {},
                        artifacts=[dict(artifact) for artifact in record.artifacts],
                        provider_hosted=record.provider_hosted,
                        hosted_family=record.hosted_family,
                        provider=record.provider,
                        provider_item_type=record.provider_item_type,
                        provider_call_id=record.provider_call_id,
                        provider_status=record.provider_status,
                        session_id=self._session_id,
                    )
                )
        except asyncio.CancelledError:
            # A cancellation mid-publish must not eat the terminal: release the
            # claim so the cancel path's flush_interrupted() re-delivers (a
            # duplicate for early subscribers beats a permanently dangling card).
            record.terminal_published = False
            self._completed_here -= 1
            raise
        await self.publish_progress()

    async def flush_interrupted(self) -> None:
        for record in list(self.calls.values()):
            if not record.terminal_published:
                await self._publish_terminal(record, interrupted=True)

    def _consume_message(self, update: AgentMessageChunk) -> None:
        if update.message_id is not None and self._message_id is not None and update.message_id != self._message_id:
            self._seal_segment()
        if update.message_id is not None:
            self._message_id = update.message_id
        text = _content_block_text(update.content)
        if text:
            self._charge(text)
            self._open_segment += text

    def _seal_segment(self) -> None:
        if self._open_segment.strip():
            self._segments.append(self._open_segment)
        self._open_segment = ""

    def result_text(self) -> str:
        self._seal_segment()
        segments = [segment for segment in self._segments if segment.strip()]
        if not segments:
            return ""
        if self.result_mode == "transcript":
            return "\n\n".join(segments)
        return segments[-1]

    def partial_text(self) -> str:
        segments = [*self._segments]
        if self._open_segment.strip():
            segments.append(self._open_segment)
        return "\n\n".join(segment for segment in segments if segment.strip())

    async def finalize_usage(self, *, spend: int, unreported_attempts: int) -> None:
        self._spend = spend
        self._unreported = unreported_attempts
        await self.publish_progress()

    async def publish_progress(self) -> None:
        if self._bus is not None:
            await self._bus.publish(
                SubAgentProgress(
                    agent_name=self.agent_name,
                    invocation_id=self.invocation_id,
                    tool_call_count=self.completed_count,
                    total_tokens=self.latest_context_used,
                    total_usage_tokens=self._spend,
                    usage_unreported_attempts=self._unreported,
                    session_id=self._session_id,
                )
            )


def _tool_content_text_from_bounded(content: Sequence[Any]) -> str:
    rendered: list[str] = []
    for item in content:
        if not isinstance(item, Mapping):
            rendered.append(_preview(item))
            continue
        item_type = item.get("type")
        if item_type == "content":
            block = item.get("content")
            if isinstance(block, Mapping):
                if block.get("type") == "text":
                    rendered.append(str(block.get("text", "")))
                elif block.get("type") == "resource_link":
                    rendered.append(f"[resource: {block.get('uri', '')}]")
                elif block.get("type") == "resource":
                    rendered.append("[resource]")
        elif item_type == "diff":
            rendered.append(f"edited {item.get('path', '')}")
        elif item_type == "terminal":
            rendered.append(f"[terminal {item.get('terminalId', item.get('terminal_id', ''))}]")
    return _preview("\n".join(part for part in rendered if part))


class AcpPermissionBroker:
    """Translate remote permissions and ask-user requests into Chrys events."""

    def __init__(
        self,
        *,
        event_bus: EventBus,
        session_id: str | None,
        caller_name: str,
        mode_getter: Callable[[], ApprovalMode],
        turn_context: TurnContextReader,
        workspace_roots: Sequence[str],
        workspace_cwd: str,
        approval_judge: ApprovalJudge | None,
        ask_user_timeout_seconds: float | None,
        allow_user_interaction: bool = True,
        trajectory_context: TrajectoryContext | None = None,
        trajectory_boundary_operation_id: str | None = None,
    ) -> None:
        self._bus = event_bus
        self._session_id = session_id
        self._caller_name = caller_name
        self._mode_getter = mode_getter
        self._turn_context = turn_context
        self._workspace_roots = list(workspace_roots)
        self._workspace_cwd = workspace_cwd
        self._judge = approval_judge
        self._ask_timeout = ask_user_timeout_seconds
        self._allow_user_interaction = allow_user_interaction
        self._trajectory_context = trajectory_context
        self._trajectory_boundary_operation_id = trajectory_boundary_operation_id
        self._translator: AcpUpdateTranslator | None = None
        self._arbiter = ApprovalDecisionArbiter(event_bus, session_id=session_id)
        self._permission_waits: dict[str, asyncio.Future[ApprovalResponse]] = {}
        self._ask_waits: dict[str, asyncio.Future[AskUserResponse]] = {}
        # Latched by cascade_abort() synchronously, ahead of the backgrounded
        # cancel_pending_waits(). Any in-flight or freshly-arriving permission
        # decision consults it so an already-settled "allow" cannot slip
        # through after the global interrupt.
        self._aborted = False

    def mark_aborted(self) -> None:
        """Latch the abort flag synchronously (called from cascade_abort)."""
        self._aborted = True

    def set_translator(self, translator: AcpUpdateTranslator) -> None:
        self._translator = translator

    async def on_update(self, seq: int, notification: SessionNotification) -> None:
        translator = self._translator
        if translator is not None:
            await translator.on_update(seq, notification)

    async def on_permission_request(
        self,
        tool_call: ToolCallUpdate,
        options: Sequence[PermissionOption],
    ) -> PermissionDecision:
        # Every permission request lands in the owner-only audit ring with its
        # outcome — including ones the remote sends before any ToolCallStart,
        # which would otherwise reach no durable artifact.
        decision = PermissionDecision.cancelled()
        try:
            decision = await self._decide_permission(tool_call, options)
            return decision
        finally:
            self._record_permission_audit(tool_call, decision)

    def _record_permission_audit(self, tool_call: ToolCallUpdate, decision: PermissionDecision) -> None:
        translator = self._translator
        if translator is None:
            return
        outcome = {"allow": "allowed", "deny": "denied"}.get(decision.action, "cancelled")
        translator.record_permission_request(
            title=str(tool_call.title or "tool"),
            kind=display_tool_kind(tool_call.kind),
            raw_input=tool_call.raw_input,
            outcome=outcome,
        )

    async def _decide_permission(
        self,
        tool_call: ToolCallUpdate,
        options: Sequence[PermissionOption],
    ) -> PermissionDecision:
        # A cascade abort already fired: never open a fresh permission — even a
        # BYPASS-mode auto-allow — after the global interrupt.
        if self._aborted:
            return PermissionDecision.cancelled()
        allow = _select_option(options, _ALLOW_KINDS)
        reject = _select_option(options, _REJECT_KINDS)
        # Polarity is checked before every mode, including BYPASS.
        if allow is None:
            return PermissionDecision.deny(reject.option_id) if reject is not None else PermissionDecision.cancelled()
        mode = self._mode_getter()
        if mode == ApprovalMode.BYPASS:
            return PermissionDecision.allow(allow.option_id)

        approval_trace = ApprovalTrace.open_for_operation(
            context=self._trajectory_context,
            target_operation_id=self._trajectory_boundary_operation_id,
        )
        async with OneShotCorrelation(self._bus, ApprovalResponse) as correlation:
            request_id = correlation.request_id
            future = correlation.future
            self._permission_waits[request_id] = future
            judging = mode == ApprovalMode.AUTO and self._judge is not None
            judge_title, judge_kind, raw_args = _permission_judge_fields(tool_call)
            presentation_title = _preview(tool_call.title or "tool", limit=_MAX_FIELD_CHARS - 4)
            published_args = _raw_args_dict(tool_call.raw_input)
            request_publish_started = False
            try:
                if judging:
                    await self._arbiter.ensure_subscription()
                if approval_trace is not None:
                    await approval_trace.requested(
                        tool_name=f"acp:{presentation_title}",
                        approval_mode=mode.value,
                        approval_level="require",
                    )
                request_publish_started = True
                await self._bus.publish(
                    ApprovalRequest(
                        request_id=request_id,
                        caller_name=self._caller_name,
                        tool_name=f"acp:{presentation_title}",
                        tool_kind="",
                        # judge_kind folds in the chrys _meta enhancement (nested
                        # chrys declares kind precisely); the "remote" fallback
                        # keeps this non-empty for every bridged request so the
                        # dialog renders the friendly header even when the remote
                        # claims no recognizable kind.
                        presentation_kind=judge_kind or "remote",
                        args=published_args,
                        intent_summary=f"Allow remote tool: {presentation_title}",
                        user_message=self._turn_context.user_message,
                        workspace_roots=list(self._workspace_roots),
                        workspace_cwd=self._workspace_cwd,
                        judging=judging,
                        session_id=self._session_id,
                    )
                )
            except asyncio.CancelledError:
                self._permission_waits.pop(request_id, None)
                if approval_trace is not None:
                    approval_trace.interrupted_soon(reason_code="cancelled")
                if request_publish_started:
                    await self._publish_initial_dialog_cleanup(
                        ApprovalCancelled(request_id=request_id, session_id=self._session_id)
                    )
                return PermissionDecision.cancelled()
            except BaseException:
                self._permission_waits.pop(request_id, None)
                if approval_trace is not None:
                    approval_trace.interrupted_soon(reason_code="failed")
                raise
            judge_task: asyncio.Task[None] | None = None
            try:
                if judging:
                    judge_task = asyncio.create_task(
                        self._arbiter.judge(
                            request_id=request_id,
                            judge=self._judge,
                            judge_input=ApprovalJudgeInput(
                                tool_name=judge_title,
                                tool_kind=judge_kind,
                                args=raw_args,
                                user_message=self._turn_context.user_message,
                                user_messages=self._turn_context.user_messages,
                                workspace_roots=list(self._workspace_roots),
                            ),
                            decision_future=future,
                            approved_value=ApprovalResponse(
                                request_id=request_id,
                                approved=True,
                                session_id=self._session_id,
                            ),
                            log_dir=None,
                        ),
                    )
            except BaseException:
                self._permission_waits.pop(request_id, None)
                if approval_trace is not None:
                    approval_trace.interrupted_soon(reason_code="failed")
                raise
            try:
                response = await future
            except asyncio.CancelledError:
                should_notify = self._permission_waits.pop(request_id, None) is not None
                if approval_trace is not None:
                    approval_trace.interrupted_soon(reason_code="cancelled")
                if should_notify:
                    await self._bus.publish(ApprovalCancelled(request_id=request_id, session_id=self._session_id))
                return PermissionDecision.cancelled()
            finally:
                self._permission_waits.pop(request_id, None)
                if judge_task is not None and not judge_task.done():
                    judge_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await judge_task

        # A cascade abort that landed while this permission was in flight must
        # never resolve to allow, even when the ApprovalResponse future was
        # already settled (or its resolving continuation raced ahead of the
        # backgrounded cancel_pending_waits): cascade_abort calls
        # broker.mark_aborted() synchronously before yielding, so by the time
        # `await future` returns the flag is visible. Honoring the approval
        # here would let the remote run a tool after the global interrupt.
        if self._aborted:
            if approval_trace is not None:
                approval_trace.interrupted_soon(reason_code="cascade_abort")
            return PermissionDecision.cancelled()
        if response.approved:
            if approval_trace is not None:
                await approval_trace.resolved(
                    approved=True,
                    decider=ApprovalDecider.USER if correlation.resolved_by_event else ApprovalDecider.JUDGE,
                    reason_code="approved",
                    arguments_modified=False,
                )
            return PermissionDecision.allow(allow.option_id)
        if approval_trace is not None:
            await approval_trace.resolved(
                approved=False,
                decider=ApprovalDecider.USER if correlation.resolved_by_event else ApprovalDecider.JUDGE,
                reason_code="rejected",
                arguments_modified=False,
            )
        return PermissionDecision.deny(reject.option_id) if reject is not None else PermissionDecision.cancelled()

    async def on_ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method != "chrys/request_input":
            raise RequestError.method_not_found(method)
        required = {"sessionId", "requestId", "question", "options"}
        allowed = required | {"callerName", "_meta"}
        if not required.issubset(params) or set(params).difference(allowed):
            raise RequestError.invalid_params({"method": method})
        if (
            type(params["sessionId"]) is not str
            or type(params["requestId"]) is not str
            or type(params["question"]) is not str
            or type(params["options"]) is not list
            or len(params["options"]) > 128
            or any(type(option) is not str for option in params["options"])
            or ("callerName" in params and params["callerName"] is not None and type(params["callerName"]) is not str)
            or (
                "_meta" in params
                and (type(params["_meta"]) is not dict or bool(_ASK_META_COLLISIONS.intersection(params["_meta"])))
            )
        ):
            raise RequestError.invalid_params({"method": method})
        if not self._allow_user_interaction:
            # Empty text is the protocol-compatible refusal already used for
            # aborted and expired requests.  Do not publish QuestionToUser:
            # noninteractive owners have no response channel to service it.
            return {"text": ""}
        question = _preview(params["question"], limit=_MAX_FIELD_CHARS)
        options = [_preview(option, limit=_MAX_FIELD_CHARS) for option in params["options"][:128]]
        caller = params.get("callerName")
        caller_name = _preview(caller, limit=_MAX_FIELD_CHARS) if isinstance(caller, str) else self._caller_name

        # A cascade abort already fired: do not open a fresh question dialog
        # (symmetric with _decide_permission). Return the same empty answer the
        # timeout/cancel paths use so the remote cannot proceed post-interrupt.
        if self._aborted:
            return {"text": ""}
        wait = WaitTrace.open(
            WaitCategory.USER_INPUT,
            target_operation_id=self._trajectory_boundary_operation_id,
            context=self._trajectory_context,
        )
        async with OneShotCorrelation(self._bus, AskUserResponse) as correlation:
            self._ask_waits[correlation.request_id] = correlation.future
            question_publish_started = False
            try:
                if wait is not None:
                    await wait.started()
                question_publish_started = True
                await self._bus.publish(
                    QuestionToUser(
                        question=question,
                        options=options,
                        request_id=correlation.request_id,
                        caller_name=caller_name or self._caller_name,
                        session_id=self._session_id,
                    )
                )
            except asyncio.CancelledError:
                self._ask_waits.pop(correlation.request_id, None)
                if wait is not None:
                    wait.finished_soon(outcome=WaitOutcome.CANCELLED)
                if question_publish_started:
                    await self._publish_initial_dialog_cleanup(
                        AskUserTimedOut(request_id=correlation.request_id, session_id=self._session_id)
                    )
                return {"text": ""}
            except BaseException:
                self._ask_waits.pop(correlation.request_id, None)
                if wait is not None:
                    wait.finished_soon(outcome=WaitOutcome.FAILED)
                raise
            try:
                if self._ask_timeout is None:
                    response = await correlation.future
                else:
                    async with asyncio.timeout(self._ask_timeout):
                        response = await correlation.future
            except TimeoutError:
                if wait is not None:
                    wait.finished_soon(outcome=WaitOutcome.TIMED_OUT)
                await self._bus.publish(AskUserTimedOut(request_id=correlation.request_id, session_id=self._session_id))
                return {"text": ""}
            except asyncio.CancelledError:
                should_notify = self._ask_waits.pop(correlation.request_id, None) is not None
                if wait is not None:
                    wait.finished_soon(outcome=WaitOutcome.CANCELLED)
                if should_notify:
                    await self._bus.publish(
                        AskUserTimedOut(
                            request_id=correlation.request_id,
                            session_id=self._session_id,
                        )
                    )
                return {"text": ""}
            finally:
                self._ask_waits.pop(correlation.request_id, None)
        # An abort that latched while the answer future was settling (its
        # continuation racing ahead of cancel_pending_waits) must not hand the
        # remote a real answer after the global interrupt.
        if self._aborted:
            if wait is not None:
                await wait.finished(outcome=WaitOutcome.CANCELLED)
            return {"text": ""}
        if wait is not None:
            await wait.finished()
        return {"text": response.text}

    async def _publish_initial_dialog_cleanup(self, event: ApprovalCancelled | AskUserTimedOut) -> None:
        """Finish cleanup even when cancellation interrupted the opening publish."""
        task = asyncio.create_task(self._bus.publish(event))
        with contextlib.suppress(asyncio.CancelledError):
            await _drain_task_cancellation_safe(task)

    async def cancel_pending_waits(self) -> None:
        permission_waits = list(self._permission_waits.items())
        ask_waits = list(self._ask_waits.items())
        for request_id, future in permission_waits:
            if self._permission_waits.pop(request_id, None) is None:
                continue
            if not future.done():
                future.cancel()
            await self._bus.publish(ApprovalCancelled(request_id=request_id, session_id=self._session_id))
        for request_id, future in ask_waits:
            if self._ask_waits.pop(request_id, None) is None:
                continue
            if not future.done():
                future.cancel()
            await self._bus.publish(AskUserTimedOut(request_id=request_id, session_id=self._session_id))

    async def close(self) -> None:
        await self.cancel_pending_waits()
        await self._arbiter.close()


def _select_option(
    options: Sequence[PermissionOption],
    preferred_kinds: Sequence[str],
) -> PermissionOption | None:
    for kind in preferred_kinds:
        for option in options:
            if option.kind == kind:
                return option
    return None


def _permission_judge_fields(tool_call: ToolCallUpdate) -> tuple[str, str, dict[str, Any]]:
    title = _preview(tool_call.title or "tool", limit=_MAX_FIELD_CHARS)
    kind = display_tool_kind(tool_call.kind)
    metadata = tool_call.field_meta or {}
    chrys_meta = metadata.get("chrys")
    if isinstance(chrys_meta, Mapping):
        remote_name = chrys_meta.get("tool_name", chrys_meta.get("toolName"))
        remote_kind = chrys_meta.get("tool_kind", chrys_meta.get("toolKind"))
        if isinstance(remote_name, str) and remote_name:
            title = _preview(remote_name, limit=_MAX_FIELD_CHARS)
        if isinstance(remote_kind, str):
            kind = _preview(remote_kind, limit=_MAX_FIELD_CHARS)
    return title, kind, _raw_args_dict(tool_call.raw_input)


async def _drain_task_cancellation_safe(task: asyncio.Future[Any]) -> None:
    """Await *task* to completion, absorbing external cancels of the awaiter.

    ``asyncio.shield`` only stops an external cancel from cancelling the SHIELDED
    task — the awaiter still receives ``CancelledError``, so a single
    ``await asyncio.shield(task)`` abandons *task* the instant a SECOND interrupt
    (or shutdown) lands on the awaiter. Re-await in a loop until the task is
    genuinely done, then re-raise ``CancelledError`` if any external cancel was
    absorbed. A cancel raised inside the task's own body still surfaces via
    ``task.result()``.

    This is the single shield-loop the ACP teardown/flush/cascade-publish paths
    all funnel through, so no future call site can reintroduce the naive
    single-shield hazard.
    """
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    task.result()
    if cancelled:
        raise asyncio.CancelledError


class AcpSubAgentController:
    """Drive one ACP process/session per attempt with stateless connect retries."""

    def __init__(
        self,
        *,
        invocation_id: str,
        tool_name: str,
        agent_name: str,
        prompt: str,
        spec_factory: Callable[[int], AcpAgentSpec],
        broker: AcpPermissionBroker,
        event_bus: EventBus | None,
        session_id: str | None = None,
        result_mode: Literal["last_segment", "transcript"] = "last_segment",
        usage_callback: Callable[..., None] | None = None,
        attempt_callback: Callable[[int, Any, AcpUpdateTranslator], Awaitable[None]] | None = None,
        translator_callback: Callable[[AcpUpdateTranslator], Awaitable[None]] | None = None,
        pause_callback: Callable[[], Awaitable[None]] | None = None,
        max_connect_retries: int = 5,
        backoff_schedule: tuple[int, ...] = _BACKOFF,
        persist_dir: Path | None = None,
        parent_provider_call_id: str = "",
        parent_event_call_id: str = "",
        sub_agent_log_file: str = "",
        pending_record_finalizer: Callable[[Path | None], None] | None = None,
        parent_interrupted_result_commit: Callable[[], None] | None = None,
        cancellation_finalizer: Callable[[], Awaitable[None]] | None = None,
        trajectory_context: TrajectoryContext | None = None,
        trajectory_boundary_operation_id: str | None = None,
    ) -> None:
        self._invocation_id = invocation_id
        self._tool_name = tool_name
        self._agent_name = agent_name
        self._prompt = prompt
        self._spec_factory = spec_factory
        self._broker = broker
        self._bus = event_bus
        self._session_id = session_id
        self._result_mode = result_mode
        self._usage_callback = usage_callback
        self._attempt_callback = attempt_callback
        self._translator_callback = translator_callback
        self._pause_callback = pause_callback
        self._max_connect_retries = max_connect_retries
        self._backoff = backoff_schedule
        self._status = SubAgentStatus.IDLE
        self._pending_decision: asyncio.Future[str] | None = None
        self._cascade_requested = False
        self._cascade_published = False
        self._cascade_event = asyncio.Event()
        self._active_client: AcpAgentClient | None = None
        self._cancel_watchdog: asyncio.Task[None] | None = None
        self._cascade_publish_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._attempt = 0
        self._retry_attempts_total = 0
        self._completed_calls = 0
        self._spend = 0
        self._usage_unreported = 0
        self._last_error = ""
        self._diagnostic_path: str | None = None
        # High-water mark, not a set: attempts are strictly monotonic and
        # manual retries are unlimited, so retained per-attempt state must
        # stay O(1) for the controller's lifetime.
        self._last_accounted_attempt = 0
        self._latest_context_tokens = 0
        self._stop_reason = ""
        self._persist_dir = persist_dir
        self._parent_provider_call_id = parent_provider_call_id
        self._parent_event_call_id = parent_event_call_id
        self._sub_agent_log_file = sub_agent_log_file
        self._pending_record_finalizer = pending_record_finalizer
        self._parent_interrupted_result_commit = parent_interrupted_result_commit
        self._cancellation_finalizer = cancellation_finalizer
        self._cancellation_finalized = False
        self._cancellation_finalize_lock = asyncio.Lock()
        self._last_spec: AcpAgentSpec | None = None
        self._trajectory_context = trajectory_context
        self._trajectory_boundary_operation_id = trajectory_boundary_operation_id

    @property
    def status(self) -> SubAgentStatus:
        return self._status

    @property
    def total_usage_tokens(self) -> int:
        return self._spend

    @property
    def usage_unreported_attempts(self) -> int:
        return self._usage_unreported

    @property
    def latest_context_tokens(self) -> int:
        return self._latest_context_tokens

    @property
    def completed_tool_calls(self) -> int:
        return self._completed_calls

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    @property
    def last_error(self) -> str:
        return self._last_error

    @property
    def last_spec(self) -> AcpAgentSpec | None:
        return self._last_spec

    def request_retry(self) -> bool:
        return self._resolve_decision("retry")

    def request_abort(self) -> bool:
        return self._resolve_decision("abort")

    async def cascade_abort(self) -> None:
        self._cascade_requested = True
        self._commit_parent_interrupted_result()
        self._cascade_event.set()
        self._resolve_decision("cascade_abort")
        # Latch the broker abort flag synchronously, BEFORE backgrounding the
        # cancel: a permission future already settled to "allow" and queued
        # ahead of cancel_pending_waits must still resolve to cancelled.
        self._broker.mark_aborted()
        self._start_background(self._broker.cancel_pending_waits())
        if self._cascade_publish_task is None:
            self._cascade_publish_task = self._start_background(self._publish_cascade_aborted_once())
        client = self._active_client
        if client is not None:
            self._start_background(self._cancel_client(client))
            if self._cancel_watchdog is None or self._cancel_watchdog.done():
                self._cancel_watchdog = asyncio.create_task(self._force_close_after_grace(client))

    async def finalize_cancellation(self) -> None:
        """Finish controller-owned cancellation work exactly once."""
        self._commit_parent_interrupted_result()
        async with self._cancellation_finalize_lock:
            if self._cancellation_finalized:
                return
            if self._cascade_requested:
                publish_task = self._cascade_publish_task
                if publish_task is None:
                    await self._publish_cascade_aborted_once()
                else:
                    await _drain_task_cancellation_safe(publish_task)
            elif self._status not in {SubAgentStatus.COMPLETED, SubAgentStatus.ABORTED}:
                self._status = SubAgentStatus.ABORTED
            pending = tuple(self._background_tasks)
            if pending:
                await _drain_task_cancellation_safe(asyncio.gather(*pending, return_exceptions=True))
            if self._cancellation_finalizer is not None:
                await self._cancellation_finalizer()
            self._cancellation_finalized = True

    def _commit_parent_interrupted_result(self) -> None:
        callback = self._parent_interrupted_result_commit
        if callback is None:
            return
        self._parent_interrupted_result_commit = None
        callback()

    def _start_background(self, awaitable: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        task = asyncio.create_task(awaitable)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _cancel_client(self, client: AcpAgentClient) -> None:
        with contextlib.suppress(Exception):
            await client.cancel()

    async def _force_close_after_grace(self, client: AcpAgentClient) -> None:
        await asyncio.sleep(5)
        await asyncio.shield(client.force_close())

    @staticmethod
    async def _force_close_cancellation_safe(client: AcpAgentClient) -> None:
        """Finish teardown before propagating any cancellation."""
        await _drain_task_cancellation_safe(asyncio.create_task(client.force_close()))

    @staticmethod
    async def _flush_interrupted_cancellation_safe(translator: AcpUpdateTranslator) -> None:
        """Run the terminal-flush sweep to completion before propagating cancel.

        The post-teardown sweep is the last guarantee that every started tool
        call gets a terminal result. A bare ``await`` would let a second
        interrupt landing mid-sweep tear it, re-stranding a call — so shield the
        sweep exactly like ``_force_close_cancellation_safe`` shields teardown.
        """
        await _drain_task_cancellation_safe(asyncio.create_task(translator.flush_interrupted()))

    async def _wait_backoff(self, delay: int, trace: RetryBackoffTrace | None = None) -> None:
        sleep_task = asyncio.create_task(asyncio.sleep(delay))
        cascade_task = asyncio.create_task(self._cascade_event.wait())
        tasks = {sleep_task, cascade_task}
        try:
            try:
                done, _pending = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                raise
        finally:
            pending = {task for task in tasks if not task.done()}
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        if cascade_task in done and self._cascade_requested:
            raise asyncio.CancelledError
        sleep_task.result()
        if trace is not None:
            await trace.started()

    def _resolve_decision(self, value: str) -> bool:
        future = self._pending_decision
        if future is None or future.done():
            return False
        future.set_result(value)
        return True

    async def run(self) -> str:
        try:
            while True:
                result = await self._run_fresh_execution()
                if result is not None:
                    # A cascade_abort can land while a terminal handler awaits
                    # its flush_interrupted() or the finally force-close (both
                    # added in r8) — windows where _run_fresh_execution computed
                    # a COMPLETED result but SubAgentCascadeAborted was already
                    # published. Re-check here so the cascade wins instead of a
                    # success/tool_error contradicting the published event.
                    if self._cascade_requested:
                        raise asyncio.CancelledError
                    return result
                decision = await self._pause_and_wait()
                if self._cascade_requested:
                    # A cascade_abort may have set the flag AFTER a user
                    # Retry/Abort already resolved _pending_decision — the
                    # one-shot _resolve_decision("cascade_abort") then no-ops on
                    # the settled future, so `decision` is still "retry"/"abort".
                    # The cascade terminal event is already published (background
                    # task), so it must win here: acting on the stale decision
                    # would publish a contradictory SubAgentResumed/SubAgentAborted
                    # and return normally instead of cascade-cancelling.
                    raise asyncio.CancelledError
                if decision == "retry":
                    if self._bus is not None:
                        await self._bus.publish(
                            SubAgentResumed(
                                agent_name=self._agent_name,
                                invocation_id=self._invocation_id,
                                session_id=self._session_id,
                            )
                        )
                    if self._cascade_requested:
                        # Same window as the abort branch below: a cascade_abort
                        # can set the flag DURING the SubAgentResumed publish
                        # await (the 1023 recheck already passed). It must win
                        # now — otherwise we announce a resume and run a full
                        # extra attempt that contradicts the already-scheduled
                        # SubAgentCascadeAborted before _run_fresh_execution's
                        # own entry check finally cancels it.
                        raise asyncio.CancelledError
                    continue
                if decision == "abort":
                    self._status = SubAgentStatus.ABORTED
                    result = tool_error(
                        "sub_agent_aborted",
                        f"sub-agent '{self._tool_name}' aborted by user after failure — {self._last_error}",
                        details={"tool_name": self._tool_name, "invocation_id": self._invocation_id},
                    )
                    if self._bus is not None:
                        await self._bus.publish(
                            SubAgentAborted(
                                agent_name=self._agent_name,
                                invocation_id=self._invocation_id,
                                last_error=self._last_error,
                                session_id=self._session_id,
                            )
                        )
                    if self._cascade_requested:
                        # A cascade_abort can land while SubAgentAborted is
                        # publishing (the 1023 recheck already passed). The
                        # cascade terminal is already scheduled, so it must win:
                        # returning the abort tool_error here would contradict the
                        # published SubAgentCascadeAborted and skip the
                        # CASCADE_ABORTED status the except handler sets.
                        raise asyncio.CancelledError
                    return result
                raise asyncio.CancelledError
        except asyncio.CancelledError:
            self._status = SubAgentStatus.CASCADE_ABORTED if self._cascade_requested else self._status
            if self._cascade_requested:
                publish_task = self._cascade_publish_task
                if publish_task is None:
                    await self._publish_cascade_aborted_once()
                else:
                    # A single ``await asyncio.shield(publish_task)`` is torn by a
                    # SECOND cancel of this handler (shutdown / repeated interrupt),
                    # abandoning the SubAgentCascadeAborted publish mid-flight and
                    # letting invocation cleanup cancel the orphaned task. Drain it
                    # to completion first, then re-raise.
                    await _drain_task_cancellation_safe(publish_task)
            raise
        finally:
            watchdog = self._cancel_watchdog
            if watchdog is not None and not watchdog.done():
                watchdog.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watchdog
            if self._status != SubAgentStatus.PAUSED and self._pending_record_finalizer is not None:
                self._pending_record_finalizer(self._persist_path())

    async def _publish_cascade_aborted_once(self) -> None:
        if self._cascade_published:
            return
        self._cascade_published = True
        self._status = SubAgentStatus.CASCADE_ABORTED
        if self._bus is not None:
            await self._bus.publish(
                SubAgentCascadeAborted(
                    agent_name=self._agent_name,
                    invocation_id=self._invocation_id,
                    session_id=self._session_id,
                )
            )

    async def _run_fresh_execution(self) -> str | None:
        self._status = SubAgentStatus.RUNNING
        connect_failure = ""
        for connect_index in range(self._max_connect_retries + 1):
            if self._cascade_requested:
                raise asyncio.CancelledError
            self._attempt += 1
            translator = AcpUpdateTranslator(
                event_bus=self._bus,
                session_id=self._session_id,
                agent_name=self._agent_name,
                invocation_id=self._invocation_id,
                attempt=self._attempt,
                result_mode=self._result_mode,
                completed_before=self._completed_calls,
                spend_before=self._spend,
                unreported_before=self._usage_unreported,
            )
            self._broker.set_translator(translator)
            if self._translator_callback is not None:
                # Adopt the translator into the durable audit trail BEFORE the
                # transport exists: ACP permits a permission request the moment
                # session/new is announced (inside open_session), and a failure
                # in that window must not lose the recorded decision.
                await self._translator_callback(translator)
            try:
                spec = self._spec_factory(self._attempt)
            except (AcpConfigError, ValueError, TypeError) as exc:
                self._status = SubAgentStatus.COMPLETED
                detail = exc.detail if isinstance(exc, AcpConfigError) else _preview(exc, limit=500)
                return tool_error("sub_agent_acp_config", detail)
            self._last_spec = spec
            self._diagnostic_path = str(spec.stderr_log_path)
            client = AcpAgentClient(spec, self._broker, update_sink=translator, wait_controller=self._broker)
            self._active_client = client
            prompt_started = False
            usage: AcpPromptUsage | None = None
            retry_delay: int | None = None
            try:
                await client.connect()
                handshake = await client.open_session()
                if self._attempt_callback is not None:
                    await self._attempt_callback(self._attempt, handshake, translator)
                prompt_started = True
                outcome = await client.prompt(self._prompt)
                self._stop_reason = outcome.stop_reason
                usage = outcome.usage
                await self._account_usage(self._attempt, usage, translator)
                await translator.flush_interrupted()
                self._completed_calls = translator.completed_count
                if self._cascade_requested:
                    # A cascade abort must win even when the remote finishes
                    # the prompt (end_turn/refusal/truncation) before our
                    # cancel reaches it: the parent interrupt is already
                    # tearing the turn down, and the cascade terminal event
                    # may already be published — returning a success here
                    # would contradict both.
                    raise asyncio.CancelledError
                if outcome.stop_reason == "end_turn":
                    text = translator.result_text()
                    if not text:
                        self._status = SubAgentStatus.COMPLETED
                        return tool_error(
                            "sub_agent_empty_output",
                            f"sub-agent '{self._tool_name}' returned no output",
                        )
                    self._status = SubAgentStatus.COMPLETED
                    record_tool_success()
                    return text
                if outcome.stop_reason == "cancelled":
                    self._last_error = "The ACP agent cancelled the prompt unexpectedly."
                    return None
                if outcome.stop_reason == "refusal":
                    self._status = SubAgentStatus.COMPLETED
                    return tool_error("sub_agent_refusal", "The ACP agent refused the request.")
                partial = translator.partial_text()
                message = f"The ACP agent stopped with {outcome.stop_reason}."
                if partial:
                    message = f"{message} Partial output: {partial}"
                self._status = SubAgentStatus.COMPLETED
                return tool_error("sub_agent_truncated", message)
            except AcpConnectError as exc:
                connect_failure = exc.detail
                # The client, not a local flag, is the authority on retryability:
                # open_session deliberately classifies failures that settled
                # before session/new was sent as stateless AcpConnectError, and
                # the retry policy (§6.2) covers the whole spawn/initialize
                # window — pausing there would demand user intervention for a
                # provably side-effect-free respawn.
                if client.stateful_phase_started:
                    self._last_error = connect_failure
                    await translator.flush_interrupted()
                    self._completed_calls = translator.completed_count
                    return None
                if connect_index >= self._max_connect_retries:
                    self._status = SubAgentStatus.COMPLETED
                    return tool_error("sub_agent_acp_setup", connect_failure)
                retry_delay = self._backoff[min(connect_index, len(self._backoff) - 1)]
                self._retry_attempts_total += 1
                if self._bus is not None:
                    await self._bus.publish(
                        SubAgentRetryAttempt(
                            agent_name=self._agent_name,
                            invocation_id=self._invocation_id,
                            message=connect_failure,
                            attempt=connect_index + 1,
                            max_attempts=self._max_connect_retries,
                            delay_seconds=retry_delay,
                            session_id=self._session_id,
                        )
                    )
            except AcpSpawnError as exc:
                self._status = SubAgentStatus.COMPLETED
                return tool_error("sub_agent_acp_spawn", exc.detail)
            except AcpConfigError as exc:
                # Terminal-flush invariant: any exit that had a translator must
                # finalize started-but-unresolved tool calls, or a tool update
                # the remote streamed before rejecting leaves a SubAgentToolCallStart
                # with no result and stale running ownership. flush_interrupted is
                # a no-op when nothing started (the pre-prompt config/auth cases).
                await translator.flush_interrupted()
                self._completed_calls = translator.completed_count
                self._status = SubAgentStatus.COMPLETED
                return tool_error("sub_agent_acp_config", exc.detail)
            except AcpAuthRequiredError as exc:
                await translator.flush_interrupted()
                self._completed_calls = translator.completed_count
                methods = ", ".join(exc.method_names)
                suffix = f" Advertised methods: {methods}." if methods else ""
                self._status = SubAgentStatus.COMPLETED
                return tool_error(
                    "sub_agent_acp_auth",
                    f"{exc.detail}{suffix} Authenticate via the agent's own CLI.",
                )
            except AcpRefusalError as exc:
                await translator.flush_interrupted()
                self._completed_calls = translator.completed_count
                self._status = SubAgentStatus.COMPLETED
                return tool_error("sub_agent_refusal", exc.detail)
            except AcpIdleTimeoutError as exc:
                usage = exc.usage
                self._last_error = exc.detail
                if prompt_started:
                    await self._account_usage(self._attempt, usage, translator)
                await translator.flush_interrupted()
                self._completed_calls = translator.completed_count
                if self._cascade_requested:
                    raise asyncio.CancelledError from exc
                return None
            except AcpTransportError as exc:
                usage = exc.usage
                self._last_error = exc.detail
                if prompt_started:
                    await self._account_usage(self._attempt, usage, translator)
                await translator.flush_interrupted()
                self._completed_calls = translator.completed_count
                if self._cascade_requested:
                    raise asyncio.CancelledError from exc
                return None
            except asyncio.CancelledError:
                if prompt_started:
                    await self._account_usage(self._attempt, usage, translator)
                # Flush unconditionally, like every sibling handler: a tool call
                # the remote streamed after session/new but before the prompt
                # (prompt_started is still False) must not be left unresolved.
                await translator.flush_interrupted()
                raise
            finally:
                self._active_client = None
                try:
                    await self._force_close_cancellation_safe(client)
                finally:
                    # Terminal-flush invariant, enforced AFTER teardown. The
                    # update consumer stays alive through the whole force-close
                    # ladder (the client cancels it only at the very end) and
                    # ACP updates are valid from session/new onward — before the
                    # prompt and during close. A tool-call start streamed in
                    # either window would otherwise leave a SubAgentToolCallStart
                    # with no result and stale running ownership. flush_interrupted
                    # is idempotent and a no-op when nothing is unresolved, so the
                    # per-handler flushes above still drive result/usage timing
                    # while this guarantees the invariant on every exit path. The
                    # sweep is cancellation-safe so a late interrupt can't tear
                    # it; the count sync (mirroring every sibling handler) then
                    # reflects anything the sweep terminalized into
                    # completed_tool_calls, which the audit log persists.
                    try:
                        await self._flush_interrupted_cancellation_safe(translator)
                    finally:
                        self._completed_calls = translator.completed_count
            if retry_delay is not None:
                retry_trace = RetryBackoffTrace.open(
                    context=self._trajectory_context,
                    parent_operation_id=self._trajectory_boundary_operation_id,
                    retry_mode=RetryMode.CONNECTION,
                )
                if retry_trace is not None:
                    await retry_trace.scheduled(reason_code=RetryReason.CONNECTION, delay_seconds=retry_delay)
                await self._wait_backoff(retry_delay, retry_trace)
                continue
        return tool_error("sub_agent_acp_setup", connect_failure)

    async def _account_usage(
        self,
        attempt: int,
        usage: AcpPromptUsage | None,
        translator: AcpUpdateTranslator,
    ) -> None:
        if attempt <= self._last_accounted_attempt:
            return
        self._last_accounted_attempt = attempt
        self._latest_context_tokens = translator.latest_context_used
        if usage is None:
            self._usage_unreported += 1
        else:
            self._spend += usage.total_tokens
            if self._usage_callback is not None:
                self._usage_callback(
                    (
                        translator.latest_context_used
                        if translator.latest_context_size is not None
                        else usage.total_tokens
                    ),
                    usage.input_tokens,
                    usage.output_tokens,
                    0,
                    1.0,
                    0,
                    usage.cached_read_tokens,
                    translator.latest_context_size,
                    self._agent_name,
                    f"{self._invocation_id}:a{attempt}",
                    usage.total_tokens,
                )
        await translator.finalize_usage(
            spend=self._spend,
            unreported_attempts=self._usage_unreported,
        )

    async def _pause_and_wait(self) -> str:
        self._status = SubAgentStatus.PAUSED
        self._write_pending_record()
        if self._pause_callback is not None:
            # Kernel parity (controller.py pause path): stamp the audit log
            # "paused" so a crash while paused never leaves it "running"
            # forever — restore's orphan repair skips pending/backlinked
            # invocations and would otherwise never touch it again.
            try:
                await self._pause_callback()
            except Exception:
                logger.warning(
                    "Failed to record paused status for ACP sub-agent %s",
                    self._invocation_id,
                    exc_info=True,
                )
        self._pending_decision = asyncio.get_running_loop().create_future()
        if self._cascade_requested and not self._pending_decision.done():
            # A cascade abort that arrived during the pause-callback await above
            # ran its one-shot _resolve_decision while _pending_decision was
            # still None — a no-op. The abort fires exactly once, so honor it
            # now; otherwise run() blocks forever on a future nothing resolves.
            # The done() guard below then also suppresses the pause announcement
            # for an invocation that is really being torn down.
            self._pending_decision.set_result("cascade_abort")
        if self._bus is not None and not self._pending_decision.done():
            await self._bus.publish(
                SubAgentPaused(
                    agent_name=self._agent_name,
                    invocation_id=self._invocation_id,
                    tool_name=self._tool_name,
                    reason=SubAgentFailureReason.ACP_TRANSPORT,
                    last_error=self._last_error,
                    retry_attempts=self._retry_attempts_total,
                    diagnostic_path=self._diagnostic_path,
                    session_id=self._session_id,
                )
            )
        try:
            return await self._pending_decision
        finally:
            self._pending_decision = None

    def _persist_path(self) -> Path | None:
        if self._persist_dir is None:
            return None
        return self._persist_dir / f"{self._invocation_id}.json"

    def _write_pending_record(self) -> None:
        path = self._persist_path()
        if path is None:
            return
        now = datetime.now(UTC).isoformat()
        payload = {
            "schema_version": 1,
            "record_type": "sub_agent_pending",
            "runner": "acp",
            "invocation_id": self._invocation_id,
            "tool_name": self._tool_name,
            "agent_name": self._agent_name,
            "prompt_preview": _preview(self._prompt),
            "session_id": self._session_id or "",
            "parent_call_id": self._parent_provider_call_id,
            "parent_provider_call_id": self._parent_provider_call_id,
            "parent_event_call_id": self._parent_event_call_id,
            "sub_agent_log_file": self._sub_agent_log_file,
            "created_at": now,
            "paused_at": now,
            "failure_reason": SubAgentFailureReason.ACP_TRANSPORT,
            "last_error": self._last_error,
            "retry_attempts_total": self._retry_attempts_total,
        }
        try:
            atomic_write_owner_only_text(
                path,
                json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            )
        except OSError as exc:
            # Kernel parity: a failed pause-record write must degrade the
            # restore experience, not crash the pause flow (and an OSError
            # here may embed a filesystem path the model must never see).
            logger.warning("Failed to persist paused ACP sub-agent %s: %s", self._invocation_id, exc)
