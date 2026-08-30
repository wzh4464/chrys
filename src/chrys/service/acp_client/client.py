# Copyright (c) 2026 Chrys. All rights reserved.

"""Hardened, owned ACP client connection for one external sub-agent attempt."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import contextvars
import json
import logging
import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from acp import PROTOCOL_VERSION, Client
from acp.client.connection import ClientSideConnection
from acp.connection import StreamDirection, StreamEvent
from acp.exceptions import RequestError
from acp.schema import (
    AgentCapabilities,
    AllowedOutcome,
    ClientCapabilities,
    CreateTerminalResponse,
    DeniedOutcome,
    EnvVariable,
    FileSystemCapabilities,
    KillTerminalResponse,
    NewSessionResponse,
    PermissionOption,
    PromptResponse,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    SessionConfigOptionBoolean,
    SessionConfigOptionSelect,
    SessionNotification,
    TerminalOutputResponse,
    TextContentBlock,
    ToolCallUpdate,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)
from acp.task import (
    InMemoryMessageQueue,
    MessageQueue,
    MessageStateStore,
    NotificationRunner,
    RequestRunner,
    RpcTaskKind,
    TaskSupervisor,
)
from acp.task.state import IncomingMessage
from pydantic import ValidationError

from chrys.foundation.text.images import MAX_IMAGE_BYTES

from .errors import (
    AcpAuthRequiredError,
    AcpClientError,
    AcpConfigError,
    AcpConnectError,
    AcpIdleTimeoutError,
    AcpOperation,
    AcpSpawnError,
    AcpTransportError,
    classify_acp_error,
    classify_protocol_frame_error,
    is_best_effort_option_rejection,
)
from .spawn import ACP_STDIO_LIMIT_BYTES, AcpSpawnResult, spawn_acp_process, validate_agent_spec
from .spec import AcpAgentSpec, AcpHandshakeInfo, AcpPromptOutcome, AcpPromptUsage, PermissionDecision

logger = logging.getLogger(__name__)

_MAX_ATTEMPT_FRAMES = 100_000
_MAX_ATTEMPT_BYTES = 256 * 1024 * 1024
_MAX_REQUEST_FRAMES = 8_192
_MAX_UPDATE_ITEMS = 4_096
_MAX_UPDATE_BYTES = 64 * 1024 * 1024
_MAX_PAYLOAD_BYTES = 1024 * 1024
_MAX_PAYLOAD_DEPTH = 32
_MAX_COLLECTION_ITEMS = 4_096
_MAX_STRING_CHARS = 256 * 1024
_MAX_BASE64_IMAGE_CHARS = ((MAX_IMAGE_BYTES + 2) // 3) * 4
_MAX_REQUEST_ID_CHARS = 4_096
_MAX_METHOD_CHARS = 1_024
_MAX_PERMISSION_OPTIONS = 128
_MAX_HUMAN_WAITS = 8
_MAX_USAGE_VALUE = (1 << 63) - 1
_REQUEST_CONCURRENCY = 32
_CLOSE_RPC_TIMEOUT_SECONDS = 2.0
_CLOSE_STAGE_TIMEOUT_SECONDS = 2.0
_REQUEST_DRAIN_TIMEOUT_SECONDS = 1.0
_FINAL_DRAIN_TIMEOUT_SECONDS = 30.0

_PROMPT_METHOD = "session/prompt"
_INITIALIZE_METHOD = "initialize"
_NEW_SESSION_METHOD = "session/new"
_UPDATE_METHOD = "session/update"
_PERMISSION_METHOD = "session/request_permission"
_ASK_USER_METHOD = "_chrys/request_input"
_HUMAN_METHODS = frozenset({_PERMISSION_METHOD, _ASK_USER_METHOD})
# The selected outcome carries only an option id, so the id's kind is the
# sole proof that the wire outcome matches the callback's decision.
_DECISION_OPTION_KINDS: dict[str, frozenset[str]] = {
    "allow": frozenset({"allow_once", "allow_always"}),
    "deny": frozenset({"reject_once", "reject_always"}),
}
_META_COLLISION_KEYS = frozenset(
    {
        "options",
        "sessionId",
        "session_id",
        "toolCall",
        "tool_call",
        "kwargs",
        "method",
        "params",
    }
)
_CURRENT_INBOUND_ID: contextvars.ContextVar[str | int | None] = contextvars.ContextVar(
    "chrys_acp_inbound_request_id",
    default=None,
)


class AcpClientCallbacks(Protocol):
    """Orchestration-owned callbacks; the service layer publishes no events."""

    async def on_update(self, seq: int, notification: SessionNotification) -> None: ...

    async def on_permission_request(
        self,
        tool_call: ToolCallUpdate,
        options: Sequence[PermissionOption],
    ) -> PermissionDecision: ...

    async def on_ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...


class AcpUpdateSink(Protocol):
    """Small observer seam consumed by the PR-2 translator."""

    async def put(self, seq: int, notification: SessionNotification) -> None: ...


class AcpClientWaitController(Protocol):
    """PR-2 broker seam for resolving human waits during transport teardown."""

    async def cancel_pending_waits(self) -> None: ...


class _CallbackUpdateSink:
    def __init__(self, callbacks: AcpClientCallbacks) -> None:
        self._callbacks = callbacks

    async def put(self, seq: int, notification: SessionNotification) -> None:
        await self._callbacks.on_update(seq, notification)


def validate_json_scalar_tree(value: Any) -> None:
    """Validate every JSON key, value, and sequence element recursively."""
    if value is None or type(value) in {bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite.")
        return
    if type(value) is str:
        _validate_surrogates(value)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("JSON object keys must be strings.")
            _validate_surrogates(key)
            validate_json_scalar_tree(item)
        return
    if type(value) in {list, tuple}:
        for item in value:
            validate_json_scalar_tree(item)
        return
    raise ValueError(f"Unsupported JSON value type: {type(value).__name__}")


def _validate_surrogates(value: str) -> None:
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 >= len(value) or not 0xDC00 <= ord(value[index + 1]) <= 0xDFFF:
                raise ValueError("JSON strings cannot contain unpaired surrogates.")
            index += 2
            continue
        if 0xDC00 <= codepoint <= 0xDFFF:
            raise ValueError("JSON strings cannot contain unpaired surrogates.")
        index += 1


def encode_protocol_json(value: Any) -> bytes:
    """Encode a protocol-bound value after the shared scalar validation pass."""
    validate_json_scalar_tree(value)
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def parse_protocol_json(raw: bytes) -> Any:
    """Parse one JSON value while rejecting JavaScript non-finite constants."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON numeric constant: {value}")

    parsed = json.loads(raw, parse_constant=reject_constant)
    validate_json_scalar_tree(parsed)
    return parsed


_ENVELOPE_MEMBERS = frozenset({"jsonrpc", "id", "method", "params", "result", "error"})


def validate_json_rpc_envelope(message: Any) -> Literal["request", "notification", "response"]:
    """Validate the strict JSON-RPC envelope accepted by the SDK-facing reader."""
    if type(message) is not dict:
        raise ValueError("JSON-RPC frame root must be an object.")
    # The per-kind caps run on params/result only, and the SDK re-parses the
    # complete frame; an unknown top-level member would ride around both.
    if not _ENVELOPE_MEMBERS.issuperset(message):
        raise ValueError("JSON-RPC frame contains unknown envelope members.")
    if message.get("jsonrpc") != "2.0" or type(message.get("jsonrpc")) is not str:
        raise ValueError("JSON-RPC frame must declare version 2.0.")

    has_id = "id" in message
    has_method = "method" in message
    has_result = "result" in message
    has_error = "error" in message

    if has_id and type(message["id"]) not in {str, int}:
        raise ValueError("JSON-RPC ids must be exact strings or integers.")
    # Ids and methods are retained (inbound map keys, outgoing records) and
    # echoed into response frames, so they carry their own §7.4-style caps.
    if has_id and type(message["id"]) is str and len(message["id"]) > _MAX_REQUEST_ID_CHARS:
        raise ValueError("JSON-RPC id exceeds the retained-field cap.")
    if has_id and type(message["id"]) is int and not -(1 << 63) <= message["id"] < 1 << 63:
        raise ValueError("JSON-RPC id exceeds the interoperable integer range.")
    if has_method and type(message["method"]) is not str:
        raise ValueError("JSON-RPC methods must be strings.")
    if has_method and len(message["method"]) > _MAX_METHOD_CHARS:
        raise ValueError("JSON-RPC method exceeds the retained-field cap.")
    if "params" in message:
        if not has_method:
            raise ValueError("JSON-RPC responses cannot contain params.")
        if type(message["params"]) is not dict:
            raise ValueError("JSON-RPC params must be an object.")

    if has_method:
        if has_result or has_error:
            raise ValueError("JSON-RPC requests cannot contain result or error.")
        return "request" if has_id else "notification"

    if not has_id or has_result == has_error:
        raise ValueError("JSON-RPC responses require an id and exactly one of result or error.")
    if has_error:
        error = message["error"]
        if type(error) is not dict:
            raise ValueError("JSON-RPC error must be an object.")
        if type(error.get("code")) is not int or type(error.get("message")) is not str:
            raise ValueError("JSON-RPC error requires an exact integer code and string message.")
    return "response"


def _measure_payload(value: Any, *, depth: int = 0) -> tuple[int, int]:
    if depth > _MAX_PAYLOAD_DEPTH:
        raise ValueError("ACP payload nesting is too deep.")
    if type(value) is str:
        if len(value) > _MAX_STRING_CHARS:
            raise ValueError("ACP payload string is too large.")
        return len(value.encode("utf-8", errors="surrogatepass")), 1
    if type(value) is dict:
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise ValueError("ACP payload object has too many fields.")
        total = 0
        items = 1
        for key, item in value.items():
            if len(key) > _MAX_STRING_CHARS:
                raise ValueError("ACP payload string is too large.")
            total += len(key.encode("utf-8", errors="surrogatepass"))
            child_bytes, child_items = _measure_payload(item, depth=depth + 1)
            total += child_bytes
            items += child_items
        return total, items
    if type(value) in {list, tuple}:
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise ValueError("ACP payload sequence has too many items.")
        total = 0
        items = 1
        for item in value:
            child_bytes, child_items = _measure_payload(item, depth=depth + 1)
            total += child_bytes
            items += child_items
        return total, items
    return 16, 1


def _validate_payload_caps(value: Any) -> None:
    _total_bytes, total_items = _measure_payload(value)
    # Per-collection checks alone admit 4096-wide children at every level;
    # the aggregate bound is what actually caps the retained object count.
    if total_items > _MAX_COLLECTION_ITEMS:
        raise ValueError("ACP payload has too many items in total.")
    if len(encode_protocol_json(value)) > _MAX_PAYLOAD_BYTES:
        raise ValueError("ACP retained payload exceeds the byte limit.")


def _validate_update_image_data(data: str) -> None:
    """Validate one image payload before exempting its encoding overhead."""
    if not data:
        raise ValueError("ACP image payload is empty.")
    if len(data) > _MAX_BASE64_IMAGE_CHARS:
        raise ValueError("ACP image payload exceeds the supported size limit.")
    try:
        decoded = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("ACP image payload is not valid base64.") from exc
    if not decoded:
        raise ValueError("ACP image payload is empty.")
    if len(decoded) > MAX_IMAGE_BYTES:
        raise ValueError("ACP image payload exceeds the supported size limit.")


def _payload_without_tool_image_data(value: Any) -> Any:
    """Replace validated tool-image encodings for retained-cap accounting."""
    if type(value) is not dict:
        return value
    update = value.get("update")
    if type(update) is not dict or update.get("sessionUpdate") not in {"tool_call", "tool_call_update"}:
        return value
    content = update.get("content")
    if type(content) is not list:
        return value

    bounded_content: list[Any] | None = None
    for index, item in enumerate(content):
        if type(item) is not dict or item.get("type") != "content":
            continue
        block = item.get("content")
        if type(block) is not dict or block.get("type") != "image" or type(block.get("mimeType")) is not str:
            continue
        data = block.get("data")
        if type(data) is not str:
            continue
        _validate_update_image_data(data)
        if bounded_content is None:
            bounded_content = list(content)
        bounded_content[index] = {**item, "content": {**block, "data": ""}}

    if bounded_content is None:
        return value
    return {**value, "update": {**update, "content": bounded_content}}


def _validate_update_payload_caps(value: Any) -> None:
    """Apply retained caps while excluding validated tool-image encoding overhead."""
    _validate_payload_caps(_payload_without_tool_image_data(value))


@dataclass(slots=True)
class _OutgoingRecord:
    request_id: int
    method: str
    future: asyncio.Future[Any]
    write_committed: bool = False
    response_observed: bool = False
    caps_violated: bool = False
    raw_response: dict[str, Any] | None = None
    usage: AcpPromptUsage | None = None


class _TrackingStateStore(MessageStateStore):
    """SDK state store with outgoing metadata and no inbound retention."""

    def __init__(self) -> None:
        self._outgoing: dict[int, _OutgoingRecord] = {}
        self._completed: dict[int, _OutgoingRecord] = {}
        self._prompt_id: int | None = None
        self._prompt_registered = asyncio.Event()

    @property
    def prompt_id(self) -> int | None:
        return self._prompt_id

    async def wait_for_prompt_id(self) -> int:
        await self._prompt_registered.wait()
        if self._prompt_id is None:
            raise RuntimeError("Prompt registration event had no request id.")
        return self._prompt_id

    def reset_prompt(self) -> None:
        self._prompt_id = None
        self._prompt_registered.clear()

    def register_outgoing(self, request_id: int, method: str) -> asyncio.Future[Any]:
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        future.add_done_callback(_consume_future_exception)
        record = _OutgoingRecord(request_id=request_id, method=method, future=future)
        self._outgoing[request_id] = record
        if method == _PROMPT_METHOD:
            self._prompt_id = request_id
            self._prompt_registered.set()
        return future

    def mark_write_committed(self, request_id: int) -> None:
        record = self._outgoing.get(request_id)
        if record is not None:
            record.write_committed = True

    def observe_response(self, request_id: str | int, message: dict[str, Any]) -> _OutgoingRecord | None:
        if type(request_id) is not int:
            return None
        record = self._outgoing.get(request_id)
        if record is None:
            return None
        if not record.write_committed:
            raise ValueError("ACP response arrived before the request write was committed.")
        if record.response_observed or record.caps_violated:
            # First-response-wins: the pump observes every frame, and two
            # coalesced responses for one id both arrive before SDK routing
            # pops the record. The SDK resolves only the first, so a later
            # duplicate must not overwrite the retained response or usage,
            # re-run the caps, or resurrect a caps-violated record.
            return None
        # Retained record fields ride the same §7.4 caps as permission and
        # ask-user payloads: without this, a handful of oversized responses
        # could park most of the attempt byte budget in ``_completed`` forever.
        # The flag is load-bearing beyond the raise: the SDK receive loop still
        # routes this same frame into resolve/reject afterwards (the observer
        # cannot veto it), and those paths must not store the oversized payload
        # in the completed record's future.
        try:
            _validate_payload_caps(message)
        except ValueError as exc:
            record.caps_violated = True
            # §6.2: validly reported usage must still account even when the
            # response itself is rejected — extract it before raising so every
            # resulting AcpTransportError carries it.
            if record.method == _PROMPT_METHOD:
                record.usage = _extract_raw_usage(message)
            raise _AcpProtocolLimitError(
                "ACP response frame exceeds the retained-payload caps.", cause=exc, usage=record.usage
            ) from exc
        record.response_observed = True
        record.raw_response = message
        if record.method == _PROMPT_METHOD:
            record.usage = _extract_raw_usage(message)
        return record

    def record_for(self, request_id: int) -> _OutgoingRecord | None:
        return self._outgoing.get(request_id) or self._completed.get(request_id)

    def outstanding_record(self, request_id: str | int) -> _OutgoingRecord | None:
        """The in-flight record for a response id, or None once routed.

        The observer keys seal/barrier off this instead of ``record_for`` so
        a duplicate response frame arriving after SDK routing stays inert.
        """
        if type(request_id) is not int:
            return None
        return self._outgoing.get(request_id)

    def latest_raw_result(self, method: str) -> dict[str, Any] | None:
        records = [*self._outgoing.values(), *self._completed.values()]
        for record in reversed(records):
            if record.method != method or record.raw_response is None:
                continue
            result = record.raw_response.get("result")
            return result if type(result) is dict else None
        return None

    def resolve_outgoing(self, request_id: int, result: Any) -> None:
        record = self._outgoing.pop(request_id, None)
        if record is not None:
            self._completed[request_id] = record
            if not record.future.done():
                if record.caps_violated:
                    record.future.set_exception(
                        AcpTransportError("ACP response frame exceeds the retained-payload caps.", usage=record.usage)
                    )
                else:
                    record.future.set_result(result)

    def reject_outgoing(self, request_id: int, error: Any) -> None:
        record = self._outgoing.pop(request_id, None)
        if record is not None:
            self._completed[request_id] = record
            if not record.future.done():
                if record.caps_violated:
                    record.future.set_exception(
                        AcpTransportError("ACP response frame exceeds the retained-payload caps.", usage=record.usage)
                    )
                else:
                    record.future.set_exception(error)

    def reject_all_outgoing(self, error: Any) -> None:
        for request_id, record in tuple(self._outgoing.items()):
            self._completed[request_id] = record
            if not record.future.done():
                record.future.set_exception(error)
        self._outgoing.clear()

    def begin_incoming(self, method: str, params: Any) -> IncomingMessage:
        return IncomingMessage(method=method, params=None)

    def complete_incoming(self, record: IncomingMessage, result: Any) -> None:
        return

    def fail_incoming(self, record: IncomingMessage, error: Any) -> None:
        return


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    """Mark background rejection exceptions retrieved without changing await semantics."""
    if not future.cancelled():
        future.exception()


@dataclass(slots=True)
class _PendingSend:
    payload: dict[str, Any]
    encoded: bytes
    future: asyncio.Future[None]


class _RetainedSender:
    """SDK sender replacement with strict encoding and total pending-send failure."""

    def __init__(
        self,
        writer: asyncio.StreamWriter,
        supervisor: TaskSupervisor,
        *,
        store: _TrackingStateStore,
        before_response_send: Callable[[str | int], None],
        on_failure: Callable[[BaseException], None],
    ) -> None:
        self._writer = writer
        self._store = store
        self._before_response_send = before_response_send
        self._on_failure = on_failure
        self._queue: asyncio.Queue[_PendingSend | None] = asyncio.Queue()
        self._closing = False
        self._active: _PendingSend | None = None
        self._task = supervisor.create(self._run(), name="chrys.acp.sender", on_error=self._task_failed)

    async def send(self, payload: dict[str, Any]) -> None:
        is_response = "method" not in payload and ("result" in payload or "error" in payload)
        if self._task.done():
            raise ConnectionError("ACP sender has stopped.")
        if self._closing and not is_response:
            raise ConnectionError("ACP sender is closing.")
        try:
            encoded = encode_protocol_json(payload) + b"\n"
        except ValueError as exc:
            failure = AcpConfigError("A local ACP payload is not valid JSON.", cause=exc)
            request_id = payload.get("id")
            if "method" in payload and type(request_id) is int:
                self._store.reject_outgoing(request_id, failure)
            self._on_failure(failure)
            raise failure from exc
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._queue.put(_PendingSend(payload=payload, encoded=encoded, future=future))
        await future

    def begin_close(self) -> None:
        self._closing = True

    async def close(self) -> None:
        if not self._closing:
            self._closing = True
        if self._task.done():
            self._reject_pending(ConnectionError("ACP sender stopped."))
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            return
        await self._queue.put(None)
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def emergency_stop(self) -> None:
        self._closing = True
        self._task.cancel()
        # This runs only after the normal close already wedged, so the sender
        # cannot be trusted to honor that single cancellation; a direct await
        # would hand the close ladder to a task that may never finish, and
        # queued sends must be rejected even when it never does.
        await _reap_task(self._task)
        self._reject_pending(ConnectionError("ACP sender stopped."))
        # The item _run already popped is unreachable from the queue; if the
        # task outlived both reap cancellations, its caller must still be
        # released here (both sides guard with future.done()).
        active = self._active
        if active is not None and not active.future.done():
            active.future.set_exception(ConnectionError("ACP sender stopped."))

    async def _run(self) -> None:
        failure: BaseException | None = None
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    return
                if item.future.cancelled() and "method" in item.payload:
                    # Normal close appends its sentinel behind queued items,
                    # so a request whose caller already gave up (caller
                    # cancellation or a terminal wake) would still flush to
                    # the agent and execute remotely. Responses stay owed to
                    # the peer regardless of local cancellation, so only
                    # method-bearing frames are dropped.
                    continue
                self._active = item
                try:
                    payload = item.payload
                    if "method" in payload and "id" in payload and type(payload["id"]) is int:
                        self._store.mark_write_committed(payload["id"])
                    elif "method" not in payload and "id" in payload:
                        self._before_response_send(payload["id"])
                    self._writer.write(item.encoded)
                    await self._writer.drain()
                except BaseException as exc:
                    failure = exc
                    if not item.future.done():
                        item.future.set_exception(exc)
                    raise
                else:
                    if not item.future.done():
                        item.future.set_result(None)
                    self._active = None
        except asyncio.CancelledError:
            raise
        finally:
            self._reject_pending(failure or ConnectionError("ACP sender stopped."))

    def _reject_pending(self, error: BaseException) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if item is not None and not item.future.done():
                item.future.set_exception(error)

    def _task_failed(self, _task: asyncio.Task[Any], error: BaseException) -> None:
        self._store.reject_all_outgoing(ConnectionError("ACP sender failed."))
        self._on_failure(error)


class _BackpressureDispatcher:
    """Request-semaphore dispatcher that sinks every notification without tasks."""

    def __init__(
        self,
        *,
        queue: MessageQueue,
        supervisor: TaskSupervisor,
        store: MessageStateStore,
        request_runner: RequestRunner,
        notification_runner: NotificationRunner,
        on_failure: Callable[[BaseException], None],
    ) -> None:
        self._queue = queue
        self._supervisor = supervisor
        self._store = store
        self._request_runner = request_runner
        self._on_failure = on_failure
        self._semaphore = asyncio.Semaphore(_REQUEST_CONCURRENCY)
        self._task: asyncio.Task[None] | None = None
        self._runners: set[asyncio.Task[Any]] = set()
        self._admitting = True

    def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("ACP dispatcher already started.")
        self._task = self._supervisor.create(
            self._run(),
            name="chrys.acp.dispatcher",
            on_error=self._dispatcher_failed,
        )

    async def _run(self) -> None:
        async for task in self._queue:
            try:
                if task.kind is not RpcTaskKind.REQUEST or not self._admitting:
                    continue
                await self._semaphore.acquire()
                if not self._admitting:
                    self._semaphore.release()
                    continue
                self._dispatch_request(task.message)
            finally:
                self._queue.task_done()

    def _dispatch_request(self, message: dict[str, Any]) -> None:
        record = self._store.begin_incoming(message["method"], message.get("params"))

        async def runner() -> None:
            token = _CURRENT_INBOUND_ID.set(message["id"])
            try:
                result = await self._request_runner(message)
            except Exception as exc:
                self._store.fail_incoming(record, exc)
                raise
            else:
                self._store.complete_incoming(record, result)
            finally:
                _CURRENT_INBOUND_ID.reset(token)
                self._semaphore.release()

        task = self._supervisor.create(runner(), name="chrys.acp.request")
        self._runners.add(task)
        task.add_done_callback(self._runners.discard)

    def close_admission(self) -> None:
        self._admitting = False
        if self._task is not None:
            self._task.cancel()

    async def drain_runners(self) -> None:
        if not self._runners:
            return
        done, pending = await asyncio.wait(tuple(self._runners), timeout=_REQUEST_DRAIN_TIMEOUT_SECONDS)
        for task in pending:
            task.cancel()
        cancelled_done: set[asyncio.Task[Any]] = set()
        if pending:
            cancelled_done, _still_pending = await asyncio.wait(
                pending,
                timeout=_REQUEST_DRAIN_TIMEOUT_SECONDS,
            )
        for task in (*done, *cancelled_done):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def stop(self) -> None:
        self.close_admission()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    async def cancel_runners(self) -> None:
        """Cancel in-flight request callbacks without closing admission.

        Admission stays open so requests arriving after the prompt seal are
        still dispatched and politely auto-denied by the callbacks; the close
        ladder remains the sole owner of dispatcher teardown. Supervisor error
        hooks skip cancelled tasks, so this never feeds the failure path.
        """
        for task in tuple(self._runners):
            task.cancel()
        await self.drain_runners()

    async def emergency_stop(self) -> None:
        self.close_admission()
        await self.cancel_runners()

    def _dispatcher_failed(self, _task: asyncio.Task[Any], error: BaseException) -> None:
        self._on_failure(error)


@dataclass(frozen=True, slots=True)
class _PromptBarrier:
    request_id: int


class _AcpDisconnectError(AcpTransportError):
    """EOF after the exact prompt response; drain the barrier before closing."""


class _AcpProtocolLimitError(AcpTransportError):
    """Deterministic cap or budget violation; never retryable pre-stateful."""


class _ObserverBuffer:
    """Bounded update lane plus an unmetered control-sentinel lane."""

    def __init__(self) -> None:
        self._items: deque[tuple[dict[str, Any] | _PromptBarrier, int]] = deque()
        self._data_items = 0
        self._data_bytes = 0
        self._ready = asyncio.Event()
        self._closed = False

    def put_update(self, params: dict[str, Any]) -> None:
        # Updates are retained (buffered) and then published, so they ride the
        # same §7.4 caps as every other retained frame. Validated tool-image
        # encodings are the sole exception because base64 expands the shared
        # 3 MiB image limit beyond the generic string and payload ceilings.
        try:
            _validate_update_payload_caps(params)
        except ValueError as exc:
            raise AcpTransportError("An ACP update exceeds the retained-payload caps.", cause=exc) from exc
        size = len(encode_protocol_json(params))
        if self._data_items >= _MAX_UPDATE_ITEMS or self._data_bytes + size > _MAX_UPDATE_BYTES:
            raise AcpTransportError("The ACP update buffer exceeded its safety budget.")
        self._items.append((params, size))
        self._data_items += 1
        self._data_bytes += size
        self._ready.set()

    def put_barrier(self, request_id: int) -> None:
        self._items.append((_PromptBarrier(request_id), 0))
        self._ready.set()

    async def get(self) -> dict[str, Any] | _PromptBarrier:
        while not self._items:
            if self._closed:
                raise EOFError
            self._ready.clear()
            await self._ready.wait()
        item, size = self._items.popleft()
        if size:
            self._data_items -= 1
            self._data_bytes -= size
        return item

    def close(self) -> None:
        self._closed = True
        self._ready.set()


@dataclass(slots=True)
class _InboundRequest:
    method: str
    human_wait: bool


class _FrameObserver:
    """Synchronous SDK observer retaining exact wire order without raising."""

    def __init__(
        self,
        *,
        store: _TrackingStateStore,
        updates: _ObserverBuffer,
        active_session: Callable[[], str | None],
        fail: Callable[[BaseException], None],
        activity: Callable[[], None],
    ) -> None:
        self._store = store
        self._updates = updates
        self._active_session = active_session
        self._fail = fail
        self._activity = activity
        self._inbound: dict[str | int, _InboundRequest] = {}
        self._human_waits = 0
        self._sealed = False

    @property
    def human_wait_count(self) -> int:
        return self._human_waits

    @property
    def prompt_sealed(self) -> bool:
        """Whether the exact prompt response has been observed on the wire."""
        return self._sealed

    def human_wait_admitted(self, request_id: str | int | None) -> bool:
        """Return whether this request holds one of the capped human-wait slots."""
        if request_id is None:
            return False
        request = self._inbound.get(request_id)
        return request is not None and request.human_wait

    def __call__(self, event: StreamEvent) -> None:
        try:
            if event.direction is StreamDirection.INCOMING:
                self._observe_incoming(event.message)
            else:
                self._observe_outgoing(event.message)
        except BaseException as exc:
            failure = (
                exc
                if isinstance(exc, AcpTransportError)
                else AcpTransportError(
                    "The ACP frame observer failed.",
                    cause=exc,
                )
            )
            self._fail(failure)

    def _observe_incoming(self, message: dict[str, Any]) -> None:
        if "method" in message and "id" in message:
            request_id = message["id"]
            if request_id in self._inbound:
                raise ValueError("Duplicate outstanding inbound JSON-RPC id.")
            method = message["method"]
            active_session = self._is_active_session_frame(message)
            # Registration is capped here, at arrival: an uncapped flag from a
            # request the dispatcher never admits would suspend idle expiry
            # forever. Unflagged requests are auto-rejected by the callbacks.
            # A sealed attempt admits no new waits: the turn is already over.
            human_wait = (
                method in _HUMAN_METHODS
                and active_session
                and self._human_waits < _MAX_HUMAN_WAITS
                and not self._sealed
            )
            self._inbound[request_id] = _InboundRequest(method=method, human_wait=human_wait)
            if human_wait:
                self._human_waits += 1
            if active_session:
                self._activity()
            return

        if "method" in message:
            if message["method"] == _UPDATE_METHOD:
                if self._sealed:
                    # Wire order is queue order, so anything observed after
                    # the exact prompt response would land behind the barrier
                    # and leak into the sink after prompt() returned.
                    logger.debug("Dropping an ACP session/update observed after the prompt response")
                    return
                if not self._is_active_session_frame(message):
                    # Foreign traffic is inert by contract: it must not
                    # consume the active attempt's update budget, and a
                    # malformed or oversized foreign frame must not be able
                    # to fail the attempt from here or from the consumer.
                    logger.debug("Dropping an ACP session/update for a foreign session")
                    return
                self._activity()
                self._updates.put_update(message.get("params", {}))
            elif self._is_active_session_frame(message):
                self._activity()
            return

        request_id = message["id"]
        # The pump already ran ``observe_response`` in frame order (a response
        # coalesced with EOF must be tracked before the EOF classification);
        # this branch only orders seal/barrier against the queued updates.
        record = self._store.outstanding_record(request_id)
        if record is None or not record.response_observed:
            return
        self._activity()
        if record.method == _PROMPT_METHOD:
            if record.request_id == self._store.prompt_id:
                self._seal_prompt()
            self._updates.put_barrier(record.request_id)

    def _seal_prompt(self) -> None:
        """Mark the attempt terminal at the exact prompt response.

        Sealing happens synchronously in the pump's frame order, so every
        later frame — updates, new human requests — sees it, and the idle
        watchdog can never blame the agent for local post-response drains.
        Outstanding human-wait slots are released here: their callbacks are
        auto-denied (not yet started) or cancelled (in flight) by prompt()'s
        completion path and must not suspend idle expiry meanwhile.
        """
        self._sealed = True
        for request in self._inbound.values():
            request.human_wait = False
        self._human_waits = 0

    def _observe_outgoing(self, message: dict[str, Any]) -> None:
        # Inbound ids are retired at the response-write seam
        # (before_response_send). This observer runs only after the SDK
        # finishes its send, and the peer may legally reuse an id the moment
        # it reads the response — popping here could retire the reused id's
        # fresh registration.
        return

    def before_response_send(self, request_id: str | int) -> None:
        # The write seam is the id's end of life: writer.write follows this
        # call synchronously, and every response flows through the retained
        # sender, so retirement here is total and the peer may reuse the id
        # as soon as it reads the response.
        request = self._inbound.pop(request_id, None)
        if request is not None and request.human_wait:
            request.human_wait = False
            self._human_waits -= 1
            self._activity()

    def clear_human_wait(self, request_id: str | int | None) -> None:
        if request_id is None:
            return
        request = self._inbound.get(request_id)
        if request is not None and request.human_wait:
            request.human_wait = False
            self._human_waits -= 1
            self._activity()

    def clear(self) -> None:
        self._inbound.clear()
        self._human_waits = 0
        self._activity()

    def _is_active_session_frame(self, message: dict[str, Any]) -> bool:
        return _is_active_session_payload(message, self._active_session())


def _is_active_session_payload(message: dict[str, Any], active: str | None) -> bool:
    if active is None:
        return False
    params = message.get("params")
    return type(params) is dict and type(params.get("sessionId")) is str and params["sessionId"] == active


class _PauseTransport:
    """StreamReader pause/resume hook that backpressures the raw framing pump."""

    def __init__(self) -> None:
        self._reading = asyncio.Event()
        self._reading.set()

    async def wait_until_reading(self) -> None:
        await self._reading.wait()

    def pause_reading(self) -> None:
        self._reading.clear()

    def resume_reading(self) -> None:
        self._reading.set()


class _FramingPump:
    """Validate and meter raw stdout before forwarding frames to the SDK."""

    def __init__(
        self,
        raw_reader: asyncio.StreamReader,
        sdk_reader: asyncio.StreamReader,
        pause_transport: _PauseTransport,
        *,
        stateful: Callable[[], bool],
        closing: Callable[[], bool],
        preflight: Callable[[dict[str, Any]], dict[str, Any]],
        on_response: Callable[[dict[str, Any]], None],
        active_session: Callable[[], str | None],
        fail: Callable[[BaseException], None],
    ) -> None:
        self._raw_reader = raw_reader
        self._sdk_reader = sdk_reader
        self._pause_transport = pause_transport
        self._stateful = stateful
        self._closing = closing
        self._preflight = preflight
        self._on_response = on_response
        self._active_session = active_session
        self._fail = fail
        self._frames = 0
        self._bytes = 0
        self._request_frames = 0
        self._task: asyncio.Task[None] | None = None

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def bytes(self) -> int:
        return self._bytes

    def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="chrys.acp.framing")

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        if task is asyncio.current_task():
            return
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._task = None
        # The SDK receive loop blocks on this reader; without EOF it would
        # outlive a close whose SDK-side shutdown never runs (wedged sender).
        self._sdk_reader.feed_eof()

    async def _run(self) -> None:
        try:
            while True:
                await self._pause_transport.wait_until_reading()
                try:
                    line = await self._raw_reader.readline()
                except ValueError as exc:
                    raise _AcpProtocolLimitError("The ACP agent emitted an oversized stdout frame.", cause=exc) from exc
                if not line:
                    if not self._closing():
                        failure = (
                            _AcpDisconnectError("The ACP process closed its protocol stream.")
                            if self._stateful()
                            else classify_protocol_frame_error(
                                EOFError("ACP stdout closed."),
                                stateful=False,
                                complete=False,
                            )
                        )
                        self._fail(failure)
                    self._sdk_reader.feed_eof()
                    return
                self._frames += 1
                self._bytes += len(line)
                if self._frames > _MAX_ATTEMPT_FRAMES or self._bytes > _MAX_ATTEMPT_BYTES:
                    raise _AcpProtocolLimitError("The ACP inbound frame budget was exhausted.")

                complete = line.endswith(b"\n")
                raw = line[:-1]
                if raw.endswith(b"\r"):
                    raw = raw[:-1]
                if not complete:
                    raise classify_protocol_frame_error(
                        EOFError("ACP frame ended without a newline."),
                        stateful=self._stateful(),
                        complete=False,
                    )
                try:
                    message = parse_protocol_json(raw)
                    kind = validate_json_rpc_envelope(message)
                except BaseException as exc:
                    raise classify_protocol_frame_error(
                        exc,
                        stateful=self._stateful(),
                        complete=True,
                    ) from exc
                if kind == "request":
                    self._request_frames += 1
                    if self._request_frames > _MAX_REQUEST_FRAMES:
                        raise _AcpProtocolLimitError("The ACP inbound request-frame budget was exhausted.")
                    message = self._preflight(message)
                    raw = encode_protocol_json(message)
                elif kind == "notification":
                    if message.get("method") == _UPDATE_METHOD and not _is_active_session_payload(
                        message, self._active_session()
                    ):
                        # Foreign traffic is inert by contract even when
                        # oversized: it must be dropped before the caps can
                        # turn it transport-fatal. A flood still burns the
                        # attempt frame budget above, so it stays bounded.
                        logger.debug("Dropping an ACP session/update for a foreign session at the pump")
                        continue
                    # The SDK re-parses every forwarded notification and the
                    # dispatcher caps only session/update, so ext/unknown
                    # methods must ride the retained-payload caps here.
                    # Notifications owe no response, so a violation is
                    # transport-fatal (matching put_update) rather than
                    # substituted like request params.
                    try:
                        params = message.get("params")
                        if message.get("method") == _UPDATE_METHOD:
                            _validate_update_payload_caps(params)
                        else:
                            _validate_payload_caps(params)
                    except ValueError as exc:
                        raise _AcpProtocolLimitError(
                            "An ACP notification exceeds the retained-payload caps.",
                            cause=exc,
                        ) from exc
                elif kind == "response":
                    # The pump is the single serial point ahead of SDK routing;
                    # announcing here is what keeps preflight/observer session
                    # checks race-free for frames written back-to-back with the
                    # session/new response.
                    self._on_response(message)
                self._sdk_reader.feed_data(raw + b"\n")
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._fail(exc)
            self._sdk_reader.feed_eof()


class AcpAgentClient:
    """One-process ACP client with strict protocol and lifecycle ownership."""

    def __init__(
        self,
        spec: AcpAgentSpec,
        callbacks: AcpClientCallbacks,
        *,
        update_sink: AcpUpdateSink | None = None,
        wait_controller: AcpClientWaitController | None = None,
    ) -> None:
        self._spec = spec
        self._callbacks = callbacks
        self._update_sink = update_sink or _CallbackUpdateSink(callbacks)
        self._wait_controller = wait_controller
        self._spawn: AcpSpawnResult | None = None
        self._conn: ClientSideConnection | None = None
        self._store: _TrackingStateStore | None = None
        self._sender: _RetainedSender | None = None
        self._dispatcher: _BackpressureDispatcher | None = None
        self._supervisor: TaskSupervisor | None = None
        self._observer: _FrameObserver | None = None
        self._updates: _ObserverBuffer | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        self._pump: _FramingPump | None = None
        self._terminal_failure: asyncio.Future[BaseException] | None = None
        self._completion_future: asyncio.Future[None] | None = None
        self._handshake_deadline: float | None = None
        self._stateful = False
        self._closing = False
        self._session_id: str | None = None
        self._announced_session_id: str | None = None
        self._session_response_claimed = False
        self._agent_info: Any | None = None
        self._auth_methods: tuple[Any, ...] = ()
        self._capabilities = AgentCapabilities()
        self._connect_started = False
        self._prompt_started = False
        self._cancel_requested = False
        self._cancel_task: asyncio.Task[None] | None = None
        self._force_close_task: asyncio.Task[None] | None = None
        self._failure_close_task: asyncio.Task[None] | None = None
        self._activity_event: asyncio.Event | None = None
        self._last_activity = 0.0

    @property
    def stderr_tail(self) -> str:
        return self._spawn.stderr_tail if self._spawn is not None else ""

    @property
    def stateful_phase_started(self) -> bool:
        return self._stateful

    async def connect(self) -> None:
        """Spawn and initialize; roll back every acquired resource on failure."""
        if self._conn is not None:
            raise RuntimeError("ACP client is already connected.")
        if self._closing:
            raise RuntimeError("ACP client is already closed.")
        # Claimed synchronously before the first await: _conn is only
        # installed after spawning, so two concurrent calls would otherwise
        # both spawn and the second installation would strand the first
        # child beyond force_close's reach.
        if self._connect_started:
            raise RuntimeError("An ACP connection attempt has already been started.")
        self._connect_started = True
        validate_agent_spec(self._spec)
        loop = asyncio.get_running_loop()
        self._handshake_deadline = loop.time() + self._spec.handshake_timeout_seconds
        self._terminal_failure = loop.create_future()
        self._activity_event = asyncio.Event()
        self._last_activity = loop.time()
        try:
            async with asyncio.timeout_at(self._handshake_deadline):
                spawn = await spawn_acp_process(self._spec)
                if self._closing:
                    # A force_close that completed during the spawn await saw
                    # nothing to release, and its cached task never runs
                    # again; this late acquisition must be disposed here.
                    await _dispose_spawn(spawn)
                    raise AcpConnectError("The ACP client was closed.")
                # No suspension between the check above and installation, so a
                # force_close starting any later sees the full component set.
                self._spawn = spawn
                self._install_connection()
                initialize = await self._require_conn().initialize(
                    protocol_version=PROTOCOL_VERSION,
                    client_capabilities=ClientCapabilities(
                        fs=FileSystemCapabilities(read_text_file=False, write_text_file=False),
                        terminal=False,
                    ),
                    client_info=self._spec.client_info,
                )
                raw_result = self._require_store().latest_raw_result(_INITIALIZE_METHOD)
                raw_version = raw_result.get("protocolVersion") if raw_result is not None else None
                if type(raw_version) is not int or raw_version != PROTOCOL_VERSION:
                    raise AcpConfigError("The ACP agent returned an unsupported protocol version.")
                self._agent_info = initialize.agent_info
                self._auth_methods = tuple(initialize.auth_methods or ())
                self._capabilities = initialize.agent_capabilities or AgentCapabilities()
                terminal = self._require_terminal_failure()
                if terminal.done():
                    raise terminal.result()
        except asyncio.CancelledError:
            await self.force_close()
            raise
        except BaseException as exc:
            classified, failure = self._classified_connect_failure(exc)
            await self.force_close()
            raise classified from failure

    def _classified_connect_failure(self, exc: BaseException) -> tuple[AcpClientError, BaseException]:
        """Pick the classified failure and root cause for a failed connect.

        The terminal future usually holds the most specific failure (the pump
        sees protocol violations the SDK surfaces only as generic closes), but
        a deterministic local classification must not be displaced by it — an
        agent closing stdout right after a bad initialize response would
        otherwise flip from config-rejected to retryable.
        """
        local = classify_acp_error(exc, operation=AcpOperation.INITIALIZE, stateful=False)
        # AcpSpawnError is in the preserved set for the close-vs-spawn race: a
        # force_close landing while process creation is pending settles the
        # terminal with the retryable close classification, which must not
        # displace a deterministic launch failure such as ENOENT.
        if isinstance(local, AcpConfigError | AcpAuthRequiredError | AcpSpawnError):
            return local, exc
        terminal = self._terminal_failure
        if terminal is not None and terminal.done():
            root = terminal.result()
            return classify_acp_error(root, operation=AcpOperation.INITIALIZE, stateful=False), root
        return local, exc

    async def open_session(self) -> AcpHandshakeInfo:
        """Create exactly one fresh remote session and apply configured options."""
        # Single-use: a second session/new would race the pump announcement and
        # every session-id comparison site against the first session's id.
        if self._stateful:
            raise RuntimeError("An ACP session has already been opened for this client.")
        # A background failure that settled after connect returned must fail
        # here: the child may still be running with the connection already
        # known dead, and enqueueing session/new would create an orphaned
        # remote session no later frame can reach. Nothing stateful has been
        # sent yet, so the connect-phase classification (retryable) applies.
        terminal = self._require_terminal_failure()
        if terminal.done():
            root = terminal.result()
            raise classify_acp_error(root, operation=AcpOperation.NEW_SESSION, stateful=False) from root
        self._stateful = True
        conn = self._require_conn()
        try:
            async with asyncio.timeout_at(self._require_handshake_deadline()):
                additional = list(dict.fromkeys(self._spec.additional_directories))
                supported = (
                    self._capabilities.session_capabilities is not None
                    and self._capabilities.session_capabilities.additional_directories is not None
                )
                if additional and not supported:
                    logger.warning("ACP agent does not advertise additionalDirectories; sending cwd only")
                    additional = []
                response = await conn.new_session(
                    cwd=self._spec.cwd,
                    additional_directories=additional or None,
                    mcp_servers=[],
                )
                # The SDK schema accepts any str; an empty id would send ""
                # outbound while _active_session_id() reads it as no-session,
                # rejecting every callback on a "successfully" opened client.
                # The id is retained and echoed on every session frame, so it
                # rides the same cap as retained JSON-RPC ids.
                session_id = response.session_id
                if type(session_id) is not str or not 0 < len(session_id) <= _MAX_REQUEST_ID_CHARS:
                    raise AcpConfigError("The ACP agent returned an invalid session id.")
                self._session_id = session_id
                final_options = await self._apply_session_options(response)
                terminal = self._require_terminal_failure()
                if terminal.done():
                    raise terminal.result()
        except asyncio.CancelledError:
            await self.force_close()
            raise
        except BaseException as exc:
            raise classify_acp_error(exc, operation=AcpOperation.NEW_SESSION, stateful=True) from exc

        if self._cancel_requested:
            await self.cancel()
        return AcpHandshakeInfo(
            session_id=self._session_id,
            agent_info=self._agent_info,
            auth_methods=self._auth_methods,
            capabilities=self._capabilities,
            modes=response.modes,
            models=response.models,
            config_options=final_options,
        )

    async def _apply_session_options(
        self, response: Any
    ) -> tuple[SessionConfigOptionSelect | SessionConfigOptionBoolean, ...]:
        conn = self._require_conn()
        session_id = self._require_session_id()
        if self._spec.session_mode:
            advertised = response.modes is not None and self._spec.session_mode in {
                mode.id for mode in response.modes.available_modes
            }
            if not advertised:
                if self._spec.best_effort_options:
                    logger.warning("Configured ACP session mode is not advertised; continuing best-effort")
                else:
                    raise AcpConfigError("The configured ACP session mode is not advertised.")
            else:
                try:
                    await conn.set_session_mode(mode_id=self._spec.session_mode, session_id=session_id)
                except RequestError as exc:
                    if self._spec.best_effort_options and is_best_effort_option_rejection(exc):
                        logger.warning("ACP agent deterministically rejected session mode; continuing best-effort")
                    else:
                        raise classify_acp_error(exc, operation=AcpOperation.SET_MODE, stateful=True) from exc
                except Exception as exc:
                    raise classify_acp_error(exc, operation=AcpOperation.SET_MODE, stateful=True) from exc
                else:
                    # The session/new snapshot predates this accepted switch;
                    # patch it so the handshake reports the applied state
                    # (config options already get this via the returned sets).
                    response.modes.current_mode_id = self._spec.session_mode

        if self._spec.model_id:
            advertised = response.models is not None and self._spec.model_id in {
                model.model_id for model in response.models.available_models
            }
            if not advertised:
                logger.warning("Configured ACP model is not advertised; continuing with the agent default")
            else:
                try:
                    await conn.set_session_model(model_id=self._spec.model_id, session_id=session_id)
                except RequestError as exc:
                    if is_best_effort_option_rejection(exc):
                        logger.warning("ACP agent rejected the configured model; continuing with the agent default")
                    else:
                        raise classify_acp_error(exc, operation=AcpOperation.SET_MODEL, stateful=True) from exc
                except Exception as exc:
                    raise classify_acp_error(exc, operation=AcpOperation.SET_MODEL, stateful=True) from exc
                else:
                    # Same snapshot-staleness rule as the mode above.
                    response.models.current_model_id = self._spec.model_id

        # Config options may depend on one another, and every set response
        # carries the complete replacement set, so each option validates
        # against the latest returned state rather than the session/new list.
        current_options: tuple[SessionConfigOptionSelect | SessionConfigOptionBoolean, ...] = tuple(
            response.config_options or ()
        )
        for option_id, value in self._spec.config_options.items():
            if option_id not in {option.id for option in current_options}:
                if self._spec.best_effort_options:
                    logger.warning("Configured ACP option %r is not advertised; continuing best-effort", option_id)
                    continue
                raise AcpConfigError("A configured ACP session option is not advertised.")
            try:
                result = await conn.set_config_option(
                    config_id=option_id,
                    session_id=session_id,
                    value=value,
                )
            except RequestError as exc:
                if self._spec.best_effort_options and is_best_effort_option_rejection(exc):
                    logger.warning("ACP agent rejected option %r; continuing best-effort", option_id)
                    continue
                raise classify_acp_error(exc, operation=AcpOperation.SET_CONFIG_OPTION, stateful=True) from exc
            except Exception as exc:
                raise classify_acp_error(exc, operation=AcpOperation.SET_CONFIG_OPTION, stateful=True) from exc
            if result is not None and result.config_options is not None:
                current_options = tuple(result.config_options)
        return current_options

    async def prompt(self, text: str) -> AcpPromptOutcome:
        """Prompt once and return only after the exact response barrier drains."""
        if type(text) is not str:
            raise AcpConfigError("ACP prompt text must be a string.")
        try:
            validate_json_scalar_tree(text)
        except ValueError as exc:
            raise AcpConfigError("ACP prompt text is not valid Unicode.", cause=exc) from exc
        # One prompt per session, claimed synchronously: a second call would
        # replace the shared completion future and reset the store's prompt
        # registration (cross-wiring barriers and ids under concurrency), and
        # ACP usage is session-cumulative, so a sequential reuse would report
        # cumulative totals as per-attempt usage.
        if self._prompt_started:
            raise RuntimeError("An ACP prompt has already been started for this client.")
        self._prompt_started = True
        conn = self._require_conn()
        store = self._require_store()
        terminal = self._require_terminal_failure()
        loop = asyncio.get_running_loop()
        self._completion_future = loop.create_future()
        self._completion_future.add_done_callback(_consume_future_exception)
        store.reset_prompt()
        self._touch_activity()
        # An already-settled terminal failure must fail before create_task:
        # conn.prompt registers and enqueues session/prompt at its first
        # yield, and cancelling the caller never removes the queued sender
        # item — the remote could execute a prompt after the client failed.
        if terminal.done():
            await self._fail_prompt_without_response(None, terminal.result())
        sdk_prompt_task = asyncio.create_task(
            conn.prompt(
                prompt=[TextContentBlock(type="text", text=text)],
                session_id=self._require_session_id(),
            ),
            name="chrys.acp.prompt",
        )
        registration_task = asyncio.create_task(store.wait_for_prompt_id(), name="chrys.acp.prompt-registration")
        watchdog = self._start_idle_watchdog()
        try:
            registered, _ = await asyncio.wait(
                {registration_task, terminal},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if terminal in registered and registration_task not in registered:
                registration_task.cancel()
                await _reap_task(registration_task)
                await self._fail_prompt_without_response(sdk_prompt_task, terminal.result())
            prompt_id = await registration_task

            done, _ = await asyncio.wait(
                {sdk_prompt_task, terminal},
                return_when=asyncio.FIRST_COMPLETED,
            )
            record = store.record_for(prompt_id)
            response_observed = record is not None and record.response_observed
            if terminal in done and not response_observed:
                await self._fail_prompt_without_response(sdk_prompt_task, terminal.result())

            usage = record.usage if record is not None else None
            response = None
            response_error: BaseException | None = None
            try:
                response = await sdk_prompt_task
            except asyncio.CancelledError as exc:
                # This await suspends when the response was observed on the
                # wire while the SDK task is still pending (e.g. a settled
                # terminal woke the wait above). A caller cancellation lands
                # here and must reach the outer handler, which owns the
                # force-close-and-reraise; only an independently cancelled
                # SDK task stays a classified transport failure.
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                response_error = exc
            except BaseException as exc:
                response_error = exc

            # The final drain must stay bounded even with the idle watchdog
            # disabled (idle_timeout_seconds=0): a stuck update sink would
            # otherwise hold prompt() open forever after the response arrived.
            drain_error: BaseException | None = None
            completion = self._require_completion_future()
            drained, _ = await asyncio.wait({completion}, timeout=_FINAL_DRAIN_TIMEOUT_SECONDS)
            if completion in drained:
                try:
                    completion.result()
                except BaseException as exc:
                    drain_error = exc
            else:
                drain_error = TimeoutError("ACP update processing did not drain after the prompt response.")

            if drain_error is not None:
                error = AcpTransportError(
                    "ACP updates failed before prompt completion.", cause=drain_error, usage=usage
                )
                await self.force_close()
                raise error from drain_error
            # The response sealed the attempt — success AND error frames both
            # end the turn: outstanding inbound callbacks must not outlive it.
            await self._finish_sealed_prompt()
            if terminal.done():
                root = terminal.result()
                if not isinstance(root, _AcpDisconnectError):
                    # Response-wins is reserved for EOF after the exact
                    # response; any other settled terminal failure (malformed
                    # frame, oversized notification, idle expiry that raced
                    # the response) is a protocol failure the caller must
                    # see — it beats even an in-band error response — with
                    # the observed usage preserved.
                    classified = classify_acp_error(root, operation=AcpOperation.PROMPT, stateful=True)
                    if isinstance(classified, AcpTransportError):
                        classified.usage = usage
                    await self.force_close()
                    raise classified from root
            if response_error is not None:
                classified = classify_acp_error(
                    response_error,
                    operation=AcpOperation.PROMPT,
                    stateful=True,
                )
                if isinstance(classified, AcpTransportError):
                    classified.usage = usage
                    await self.force_close()
                elif terminal.done():
                    # Only the deferred EOF-after-response settles here (the
                    # arbitration above raised every other root). The exact
                    # error response still wins the classification, but the
                    # known-dead transport must be closed before raising:
                    # the deferred disconnect spawned no failure-close.
                    await self.force_close()
                raise classified from response_error
            if terminal.done():
                # Only the deferred EOF-after-response reaches here: the
                # response wins, and this close releases the dead transport.
                await self.force_close()
            # A missing response is possible only on the error path raised immediately above.
            prompt_response = cast(PromptResponse, response)
            return AcpPromptOutcome(
                stop_reason=prompt_response.stop_reason,
                usage=usage,
                user_message_id=prompt_response.user_message_id,
            )
        except asyncio.CancelledError:
            store.reject_all_outgoing(ConnectionError("ACP prompt was cancelled."))
            sdk_prompt_task.cancel()
            await _reap_task(sdk_prompt_task)
            completion = self._completion_future
            if completion is not None and not completion.done():
                completion.cancel()
            await self.force_close()
            raise
        finally:
            if watchdog is not None:
                watchdog.cancel()
                await _reap_task(watchdog)
            if not registration_task.done():
                registration_task.cancel()
                await _reap_task(registration_task)

    async def _finish_sealed_prompt(self) -> None:
        """Release inbound waits once the exact prompt response is observed.

        In-flight human callbacks are cancelled and PR-2 waits torn down here
        on the success path; every failure path reaches the same releases via
        the force-close ladder instead.
        """
        if self._wait_controller is not None:
            # Run the PR-2 seam as a task: an inline await could swallow an
            # outer cancellation inside the seam and stall the caller. Unlike
            # the close ladder's _reap_task, this bounded wait must PRESERVE a
            # caller cancellation: prompt()'s CancelledError handler owns the
            # force-close-and-reraise, and suppressing here would hand a
            # successful outcome to a cancelled caller.
            waits_task = asyncio.create_task(
                self._wait_controller.cancel_pending_waits(),
                name="chrys.acp.cancel-waits",
            )
            waits_task.add_done_callback(_consume_future_exception)
            try:
                _done, pending = await asyncio.wait({waits_task}, timeout=_REQUEST_DRAIN_TIMEOUT_SECONDS)
            except asyncio.CancelledError:
                waits_task.cancel()
                raise
            for stuck in pending:
                stuck.cancel()
        if self._dispatcher is not None:
            await self._dispatcher.cancel_runners()

    async def _fail_prompt_without_response(
        self,
        sdk_prompt_task: asyncio.Task[Any] | None,
        failure: BaseException,
    ) -> None:
        self._require_store().reject_all_outgoing(ConnectionError("ACP prompt failed."))
        if sdk_prompt_task is not None:
            sdk_prompt_task.cancel()
            await _reap_task(sdk_prompt_task)
        completion = self._completion_future
        if completion is not None:
            if not completion.done():
                completion.set_exception(failure)
            with contextlib.suppress(BaseException):
                await completion
        await self.force_close()
        if isinstance(failure, _AcpDisconnectError):
            raise AcpTransportError("The ACP process closed before returning a prompt response.", cause=failure)
        if isinstance(failure, AcpClientError):
            raise failure
        raise AcpTransportError("The ACP transport failed before a prompt response.", cause=failure) from failure

    async def cancel(self) -> None:
        """Fire a session-cancel notification and return without waiting for the peer."""
        self._cancel_requested = True
        if self._conn is None or self._session_id is None or self._closing:
            return
        if self._cancel_task is not None and not self._cancel_task.done():
            return

        async def send_cancel() -> None:
            with contextlib.suppress(Exception):
                await self._require_conn().cancel(session_id=self._require_session_id())

        self._cancel_task = asyncio.create_task(send_cancel(), name="chrys.acp.cancel")

    async def aclose(self) -> None:
        """Attempt a bounded session close, then always execute the force-close ladder."""
        try:
            if self._conn is not None and self._session_id is not None and not self._closing:
                try:
                    async with asyncio.timeout(_CLOSE_RPC_TIMEOUT_SECONDS):
                        await self._conn.close_session(session_id=self._session_id)
                except RequestError as exc:
                    if not (type(exc.code) is int and exc.code == -32601):
                        logger.debug("ACP session/close failed", exc_info=True)
                except Exception:
                    logger.debug("ACP session/close failed", exc_info=True)
        finally:
            await self.force_close()

    async def force_close(self) -> None:
        """Idempotently close SDK tasks and terminate the retained process tree."""
        if self._force_close_task is None:
            self._force_close_task = asyncio.create_task(self._force_close_impl(), name="chrys.acp.force-close")
        task = self._force_close_task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.shield(task)
            raise

    async def _force_close_impl(self) -> None:
        self._closing = True
        close_failure: AcpClientError = (
            AcpTransportError("The ACP client was closed.")
            if self._stateful
            else AcpConnectError("The ACP client was closed.")
        )
        terminal = self._terminal_failure
        if terminal is not None and not terminal.done():
            terminal.set_result(close_failure)
        completion = self._completion_future
        if completion is not None and not completion.done():
            completion.set_exception(close_failure)
        pump = self._pump
        if pump is not None:
            await pump.stop()
        dispatcher = self._dispatcher
        sender = self._sender
        if dispatcher is not None:
            dispatcher.close_admission()
        if self._observer is not None:
            self._observer.clear()
        if self._wait_controller is not None:
            # Run the PR-2 seam as a task: an inline await could swallow the
            # timeout's single cancellation and stall the ladder forever.
            waits_task = asyncio.create_task(
                self._wait_controller.cancel_pending_waits(),
                name="chrys.acp.cancel-waits",
            )
            await _reap_task(waits_task, timeout_seconds=_REQUEST_DRAIN_TIMEOUT_SECONDS)
        if dispatcher is not None:
            await dispatcher.drain_runners()
        if self._cancel_task is not None:
            self._cancel_task.cancel()
            await _reap_task(self._cancel_task)
        if sender is not None:
            sender.begin_close()

        close_failed = False
        if self._conn is not None:
            close_task = asyncio.create_task(self._conn.close(), name="chrys.acp.sdk-close")
            # asyncio.wait, never `asyncio.timeout: await close_task`: the
            # timeout's single cancellation would forward into close_task,
            # where conn.close()'s own suppress(CancelledError) awaits (our
            # dispatcher.stop, the SDK supervisor.shutdown) can consume it
            # while a wedged sender keeps close_task pending forever.
            close_failed = True
            with contextlib.suppress(asyncio.CancelledError, Exception):
                done, _ = await asyncio.wait({close_task}, timeout=_CLOSE_STAGE_TIMEOUT_SECONDS)
                if close_task in done:
                    close_failed = close_task.cancelled() or close_task.exception() is not None
            if close_failed:
                close_task.cancel()
                await _reap_task(close_task)
        if close_failed:
            if self._store is not None:
                self._store.reject_all_outgoing(ConnectionError("ACP connection close failed."))
            if sender is not None:
                await sender.emergency_stop()
            if dispatcher is not None:
                await dispatcher.emergency_stop()
            # The SDK only shuts its supervisor down after its sender close
            # succeeds; a wedged close would otherwise leak the receive task.
            # Run shutdown as a task: awaited inline, the SDK's own
            # suppress(CancelledError) would swallow the timeout's single
            # cancellation and the next stuck runner would hang the ladder.
            supervisor = self._supervisor
            if supervisor is not None:
                shutdown_task = asyncio.create_task(
                    supervisor.shutdown(),
                    name="chrys.acp.supervisor-shutdown",
                )
                await _reap_task(shutdown_task)

        spawn = self._spawn
        try:
            if spawn is not None:
                exited = False
                try:
                    spawn.process.close_stdin()
                    exited = await _wait_process_bounded(spawn, _CLOSE_STAGE_TIMEOUT_SECONDS)
                    if not exited:
                        spawn.process.terminate_tree()
                        exited = await _wait_process_bounded(spawn, _CLOSE_STAGE_TIMEOUT_SECONDS)
                except OSError:
                    # A failed OS-level wait must not abort the ladder: this
                    # close task is cached, so an escape here would leave every
                    # later force_close re-raising with resources still held.
                    pass
                finally:
                    # Descendants may survive stdin EOF or SIGTERM even when
                    # the direct child exits; the owned process group/Job stays
                    # the authority, so always finish with a kill sweep.
                    spawn.process.kill_tree()
                if not exited:
                    with contextlib.suppress(OSError):
                        await _wait_process_bounded(spawn, _CLOSE_STAGE_TIMEOUT_SECONDS)
        finally:
            if self._updates is not None:
                self._updates.close()
            if self._consumer_task is not None:
                self._consumer_task.cancel()
                await _reap_task(self._consumer_task)
            if spawn is not None:
                await spawn.stderr.stop()
                spawn.process.close_transports()

    def _install_connection(self) -> None:
        spawn = self._spawn
        if spawn is None:
            raise RuntimeError("ACP process has not been spawned.")
        sdk_reader = asyncio.StreamReader(limit=ACP_STDIO_LIMIT_BYTES)
        pause_transport = _PauseTransport()
        # StreamReader uses only pause/resume hooks here, which _PauseTransport implements.
        sdk_reader.set_transport(cast(asyncio.BaseTransport, pause_transport))
        self._store = _TrackingStateStore()
        self._updates = _ObserverBuffer()
        self._observer = _FrameObserver(
            store=self._store,
            updates=self._updates,
            active_session=self._active_session_id,
            fail=self._fail,
            activity=self._touch_activity,
        )
        self._pump = _FramingPump(
            spawn.process.stdout,
            sdk_reader,
            pause_transport,
            stateful=lambda: self._stateful,
            closing=lambda: self._closing,
            preflight=self._preflight_request,
            on_response=self._observe_response_at_pump,
            active_session=self._active_session_id,
            fail=self._fail,
        )

        def sender_factory(writer: asyncio.StreamWriter, supervisor: TaskSupervisor) -> _RetainedSender:
            self._supervisor = supervisor
            self._sender = _RetainedSender(
                writer,
                supervisor,
                store=self._require_store(),
                before_response_send=self._require_observer().before_response_send,
                on_failure=self._fail,
            )
            return self._sender

        def dispatcher_factory(
            queue: MessageQueue,
            supervisor: TaskSupervisor,
            state: MessageStateStore,
            request_runner: RequestRunner,
            notification_runner: NotificationRunner,
        ) -> _BackpressureDispatcher:
            self._dispatcher = _BackpressureDispatcher(
                queue=queue,
                supervisor=supervisor,
                store=state,
                request_runner=request_runner,
                notification_runner=notification_runner,
                on_failure=self._fail,
            )
            return self._dispatcher

        # AcpAgentClient implements the SDK Client protocol; decorators obscure structural matching from ty.
        self._conn = ClientSideConnection(
            cast(Client, self),
            spawn.process.stdin,
            sdk_reader,
            queue=InMemoryMessageQueue(maxsize=256),
            state_store=self._store,
            sender_factory=sender_factory,
            dispatcher_factory=dispatcher_factory,
            observers=[self._observer],
            receive_timeout=None,
        )
        self._pump.start()
        self._consumer_task = asyncio.create_task(self._consume_updates(), name="chrys.acp.updates")

    async def _consume_updates(self) -> None:
        sequence = 0
        try:
            while True:
                item = await self._require_updates().get()
                if isinstance(item, _PromptBarrier):
                    completion = self._completion_future
                    if (
                        completion is not None
                        and not completion.done()
                        and self._store is not None
                        and self._store.prompt_id == item.request_id
                    ):
                        completion.set_result(None)
                    continue
                # Residual guard behind the observer's intake filter — a raw
                # peek, so a foreign frame is dropped before its shape can
                # fail validation and terminate the attempt.
                if item.get("sessionId") != self._active_session_id():
                    logger.warning("Dropping ACP session/update for a foreign session")
                    continue
                if not _valid_raw_usage_update(item):
                    continue
                notification = SessionNotification.model_validate(item)
                sequence += 1
                await self._update_sink.put(sequence, notification)
        except asyncio.CancelledError:
            raise
        except EOFError:
            return
        except BaseException as exc:
            failure = AcpTransportError("ACP update validation or callback failed.", cause=exc)
            completion = self._completion_future
            if completion is not None and not completion.done():
                completion.set_exception(failure)
            self._fail(failure)

    def _fail(self, failure: BaseException) -> None:
        if not self._stateful and isinstance(failure, AcpTransportError):
            if isinstance(failure, _AcpProtocolLimitError):
                # Cap and budget violations replay identically on every
                # respawn (same executable, same greeting), so the connect
                # window must surface them as deterministic incompatibility,
                # not as a retryable connect failure.
                failure = AcpConfigError(failure.detail, cause=failure.cause or failure)
            else:
                failure = AcpConnectError(failure.detail, cause=failure.cause or failure)
        terminal = self._terminal_failure
        if terminal is not None and not terminal.done():
            terminal.set_result(failure)
        completion = self._completion_future
        response_observed = False
        if self._store is not None and self._store.prompt_id is not None:
            record = self._store.record_for(self._store.prompt_id)
            response_observed = record is not None and record.response_observed
        defer_disconnect = isinstance(failure, _AcpDisconnectError) and response_observed
        if completion is not None and not completion.done() and not defer_disconnect:
            completion.set_exception(failure)
        if isinstance(failure, AcpTransportError | AcpConnectError) and not self._closing and not defer_disconnect:
            self._failure_close_task = asyncio.create_task(self.force_close(), name="chrys.acp.failure-close")

    def _active_session_id(self) -> str | None:
        """The authoritative session id, or the pump-announced one before assignment."""
        return self._session_id or self._announced_session_id

    def _observe_response_at_pump(self, message: dict[str, Any]) -> None:
        """Track a matching response synchronously, in pump frame order.

        A response coalesced with EOF in one transport read is processed by
        the pump without yielding, so the SDK observer would mark
        ``response_observed`` only after the EOF classification already ran —
        misclassifying a completed prompt, or a deterministic error response,
        as a transport failure. Observation therefore happens here; sealing
        and the update barrier stay in the SDK observer, which is the only
        point ordered against the queued updates.
        """
        store = self._store
        request_id = message.get("id")
        if store is not None and type(request_id) is int:
            try:
                store.observe_response(request_id, message)
            except AcpTransportError:
                raise
            except BaseException as exc:
                raise AcpTransportError(
                    "The ACP frame observer failed.",
                    cause=exc,
                ) from exc
        self._sniff_session_response(message)

    def _sniff_session_response(self, message: dict[str, Any]) -> None:
        """Capture the announced session id at the pump, ahead of SDK routing.

        ``_session_id`` is assigned only when ``open_session`` resumes after
        the SDK resolves ``session/new``; an agent may write a session-bound
        request back-to-back with that response, and the pump would preflight
        it before the assignment lands. The pump orders frames serially, so
        sniffing the response stream here closes the race for every session
        check downstream (preflight, observer, callbacks, update consumer).
        """
        if self._session_id is not None or self._announced_session_id is not None:
            return
        store = self._store
        request_id = message.get("id")
        if store is None or type(request_id) is not int:
            return
        record = store.record_for(request_id)
        if record is None or record.method != _NEW_SESSION_METHOD or not record.write_committed:
            return
        # First-response claim: the SDK ignores duplicate response ids, but the
        # pump sees every frame before routing — after a failed session/new, a
        # duplicate success frame with the same id must not announce.
        if self._session_response_claimed:
            return
        self._session_response_claimed = True
        result = message.get("result")
        if type(result) is not dict:
            return
        # The announced id is retained and compared on every subsequent frame
        # BEFORE the store's caps run SDK-side, so an uncapped result must
        # never announce; downstream validation still fails the open attempt.
        try:
            _validate_payload_caps(result)
        except ValueError:
            return
        # Caps bound only size and shape; open_session accepts this result
        # solely when it validates as NewSessionResponse, so announcement
        # must clear the same schema gate before exposing the id.
        try:
            NewSessionResponse.model_validate(result)
        except ValidationError:
            return
        session_id = result.get("sessionId")
        if type(session_id) is str and 0 < len(session_id) <= _MAX_REQUEST_ID_CHARS:
            self._announced_session_id = session_id

    def _touch_activity(self) -> None:
        with contextlib.suppress(RuntimeError):
            self._last_activity = asyncio.get_running_loop().time()
        if self._activity_event is not None:
            self._activity_event.set()

    def _start_idle_watchdog(self) -> asyncio.Task[None] | None:
        if self._spec.idle_timeout_seconds == 0:
            return None
        return asyncio.create_task(self._idle_watchdog(), name="chrys.acp.idle-watchdog")

    async def _idle_watchdog(self) -> None:
        timeout_seconds = self._spec.idle_timeout_seconds
        event = self._activity_event
        if event is None:
            return
        while True:
            event.clear()
            # The exact prompt response ends the agent's idle responsibility;
            # draining already-received updates through a slow local sink is
            # governed by the final-drain bound, never blamed on the agent.
            if self._prompt_response_sealed():
                return
            observer = self._observer
            if observer is not None and observer.human_wait_count:
                await event.wait()
                continue
            remaining = timeout_seconds - (asyncio.get_running_loop().time() - self._last_activity)
            if remaining <= 0:
                self._fail(AcpIdleTimeoutError("The ACP agent stopped producing protocol activity."))
                return
            try:
                async with asyncio.timeout(remaining):
                    await event.wait()
            except TimeoutError:
                # The seal and the expiry timer can land on the same wake;
                # a sealed attempt must never fail as agent-idle.
                if self._prompt_response_sealed():
                    return
                self._fail(AcpIdleTimeoutError("The ACP agent stopped producing protocol activity."))
                return

    def _prompt_response_sealed(self) -> bool:
        observer = self._observer
        if observer is not None and observer.prompt_sealed:
            return True
        # The pump marks the prompt record strictly before the SDK observer
        # runs the seal; a response already observed on the wire must never
        # lose to idle expiry inside that scheduling window.
        store = self._store
        if store is None or store.prompt_id is None:
            return False
        record = store.record_for(store.prompt_id)
        return record is not None and record.response_observed

    def _preflight_request(self, message: dict[str, Any]) -> dict[str, Any]:
        try:
            params = message.get("params")
            if type(params) is not dict:
                raise ValueError("ACP request params must be an object.")
            _validate_payload_caps(params)
            active = self._active_session_id()
            if active is None or type(params.get("sessionId")) is not str or params["sessionId"] != active:
                raise ValueError("ACP request is not bound to the active session.")
            meta = params.get("_meta")
            if meta is not None and (type(meta) is not dict or _META_COLLISION_KEYS.intersection(meta)):
                raise ValueError("ACP request metadata collides with callback fields.")
            method = message["method"]
            if method == _PERMISSION_METHOD:
                _preflight_permission(params)
            elif method == _ASK_USER_METHOD:
                _preflight_ask_user(params)
        except ValueError:
            replacement = dict(message)
            replacement["params"] = {}
            return replacement
        return message

    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        if session_id != self._active_session_id():
            raise RequestError.invalid_params({"details": "Foreign ACP session."})
        request_id = _CURRENT_INBOUND_ID.get()
        observer = self._require_observer()
        if not observer.human_wait_admitted(request_id):
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        options_by_id: dict[str, PermissionOption] = {}
        for option in options:
            if option.option_id in options_by_id:
                # The wire outcome names only an option id, so duplicate ids
                # hand the agent the choice of what a selection meant — no
                # response can be transmitted faithfully.
                observer.clear_human_wait(request_id)
                raise RequestError.invalid_params({"details": "Duplicate ACP permission option ids."})
            options_by_id[option.option_id] = option
        try:
            try:
                decision = await self._callbacks.on_permission_request(tool_call, tuple(options))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._fail(AcpTransportError("The ACP permission callback failed.", cause=exc))
                raise
        finally:
            observer.clear_human_wait(request_id)
        if decision.action == "cancelled" or decision.option_id is None:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        selected = options_by_id.get(decision.option_id)
        if selected is None:
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        if selected.kind not in _DECISION_OPTION_KINDS.get(decision.action, frozenset()):
            # An id of the wrong kind must never transmute the decision —
            # deny stays deny even when the callback names an allow option.
            logger.warning("An ACP permission decision does not match its selected option kind; cancelling")
            return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        return RequestPermissionResponse(
            outcome=AllowedOutcome(outcome="selected", option_id=decision.option_id),
        )

    async def session_update(self, **kwargs: Any) -> None:
        return

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = _CURRENT_INBOUND_ID.get()
        try:
            if method != "chrys/request_input":
                raise RequestError.method_not_found(method)
            active = self._active_session_id()
            if active is None:
                raise RequestError.invalid_params({"details": "Foreign ACP session."})
            _validate_ext_params(method, params, active)
            if not self._require_observer().human_wait_admitted(request_id):
                return {"text": ""}
            try:
                return await self._callbacks.on_ext_method(method, params)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._fail(AcpTransportError("The ACP request-input callback failed.", cause=exc))
                raise
        finally:
            if method == "chrys/request_input":
                self._require_observer().clear_human_wait(request_id)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return

    async def write_text_file(
        self,
        content: str,
        path: str,
        session_id: str,
        **kwargs: Any,
    ) -> WriteTextFileResponse | None:
        raise RequestError.method_not_found("fs/write_text_file")

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **kwargs: Any,
    ) -> ReadTextFileResponse:
        raise RequestError.method_not_found("fs/read_text_file")

    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        raise RequestError.method_not_found("terminal/create")

    async def terminal_output(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> TerminalOutputResponse:
        raise RequestError.method_not_found("terminal/output")

    async def release_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> ReleaseTerminalResponse | None:
        raise RequestError.method_not_found("terminal/release")

    async def wait_for_terminal_exit(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> WaitForTerminalExitResponse:
        raise RequestError.method_not_found("terminal/wait_for_exit")

    async def kill_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> KillTerminalResponse | None:
        raise RequestError.method_not_found("terminal/kill")

    def on_connect(self, conn: Any) -> None:
        return

    def _require_conn(self) -> ClientSideConnection:
        if self._conn is None:
            raise RuntimeError("ACP client is not connected.")
        return self._conn

    def _require_store(self) -> _TrackingStateStore:
        if self._store is None:
            raise RuntimeError("ACP state store is not installed.")
        return self._store

    def _require_observer(self) -> _FrameObserver:
        if self._observer is None:
            raise RuntimeError("ACP frame observer is not installed.")
        return self._observer

    def _require_updates(self) -> _ObserverBuffer:
        if self._updates is None:
            raise RuntimeError("ACP update buffer is not installed.")
        return self._updates

    def _require_terminal_failure(self) -> asyncio.Future[BaseException]:
        if self._terminal_failure is None:
            raise RuntimeError("ACP terminal failure future is not installed.")
        return self._terminal_failure

    def _require_completion_future(self) -> asyncio.Future[None]:
        if self._completion_future is None:
            raise RuntimeError("ACP completion future is not installed.")
        return self._completion_future

    def _require_session_id(self) -> str:
        if self._session_id is None:
            raise RuntimeError("ACP session is not open.")
        return self._session_id

    def _require_handshake_deadline(self) -> float:
        if self._handshake_deadline is None:
            raise RuntimeError("ACP handshake has not started.")
        return self._handshake_deadline


def _preflight_permission(params: dict[str, Any]) -> None:
    options = params.get("options")
    tool_call = params.get("toolCall")
    if type(options) is not list or len(options) > _MAX_PERMISSION_OPTIONS or type(tool_call) is not dict:
        raise ValueError("Malformed ACP permission request.")
    for option in options:
        if type(option) is not dict:
            raise ValueError("Malformed ACP permission option.")
        if type(option.get("optionId")) is not str or type(option.get("name")) is not str:
            raise ValueError("Malformed ACP permission option.")
    for field in ("toolCallId",):
        if type(tool_call.get(field)) is not str:
            raise ValueError("Malformed ACP permission tool call.")
    if "title" in tool_call and tool_call["title"] is not None and type(tool_call["title"]) is not str:
        raise ValueError("Malformed ACP permission title.")


def _preflight_ask_user(params: dict[str, Any]) -> None:
    if set(params).difference({"sessionId", "requestId", "question", "options", "callerName", "_meta"}):
        raise ValueError("Malformed ACP ask-user fields.")
    if type(params.get("requestId")) is not str or type(params.get("question")) is not str:
        raise ValueError("Malformed ACP ask-user request.")
    if (
        type(params.get("options")) is not list
        or len(params["options"]) > _MAX_PERMISSION_OPTIONS
        or any(type(option) is not str for option in params["options"])
    ):
        raise ValueError("Malformed ACP ask-user options.")
    caller = params.get("callerName")
    if caller is not None and type(caller) is not str:
        raise ValueError("Malformed ACP ask-user caller name.")
    meta = params.get("_meta")
    if meta is not None and (type(meta) is not dict or _META_COLLISION_KEYS.intersection(meta)):
        raise ValueError("Malformed ACP ask-user metadata.")


def _validate_ext_params(method: str, params: dict[str, Any], session_id: str) -> None:
    if type(method) is not str or type(params) is not dict:
        raise RequestError.invalid_params({"details": "Malformed ACP extension request."})
    try:
        validate_json_scalar_tree(params)
        _validate_payload_caps(params)
        if type(params.get("sessionId")) is not str or params["sessionId"] != session_id:
            raise ValueError
        if method == "chrys/request_input":
            _preflight_ask_user(params)
    except ValueError as exc:
        raise RequestError.invalid_params({"details": "Malformed ACP extension request."}) from exc


def _extract_raw_usage(message: dict[str, Any]) -> AcpPromptUsage | None:
    result = message.get("result")
    usage = result.get("usage") if type(result) is dict else None
    if usage is None:
        error = message.get("error")
        data = error.get("data") if type(error) is dict else None
        usage = data.get("usage") if type(data) is dict else None
    if type(usage) is not dict:
        return None
    required = ("inputTokens", "outputTokens", "totalTokens")
    optional = ("cachedReadTokens", "cachedWriteTokens", "thoughtTokens")
    if any(not _valid_usage_int(usage.get(field)) for field in required):
        return None
    if any(field in usage and usage[field] is not None and not _valid_usage_int(usage[field]) for field in optional):
        return None
    return AcpPromptUsage(
        input_tokens=usage["inputTokens"],
        output_tokens=usage["outputTokens"],
        total_tokens=usage["totalTokens"],
        cached_read_tokens=usage.get("cachedReadTokens"),
        cached_write_tokens=usage.get("cachedWriteTokens"),
        thought_tokens=usage.get("thoughtTokens"),
    )


def _valid_usage_int(value: Any) -> bool:
    return type(value) is int and 0 <= value <= _MAX_USAGE_VALUE


def _valid_raw_usage_update(params: dict[str, Any]) -> bool:
    update = params.get("update")
    if type(update) is not dict or update.get("sessionUpdate") != "usage_update":
        return True
    return _valid_usage_int(update.get("used")) and _valid_usage_int(update.get("size"))


async def _reap_task(
    task: asyncio.Task[Any],
    *,
    timeout_seconds: float = _CLOSE_STAGE_TIMEOUT_SECONDS,
) -> None:
    """Bounded task reap that cannot be held hostage by swallowed cancellation.

    ``await task`` would forward an outer cancellation into the awaited task
    (``Task.cancel`` cancels ``_fut_waiter``); a task that swallows that single
    CancelledError would strand this reaper after ``asyncio.timeout`` spent its
    one shot. ``asyncio.wait`` never cancels the watched task and returns at
    the deadline unconditionally, so the close ladder below stays reachable.
    """
    task.add_done_callback(_consume_future_exception)
    pending: set[asyncio.Task[Any]] = {task}
    with contextlib.suppress(asyncio.CancelledError, Exception):
        _, pending = await asyncio.wait(pending, timeout=timeout_seconds)
    for stuck in pending:
        stuck.cancel()


async def _dispose_spawn(spawn: AcpSpawnResult) -> None:
    """Release a spawn acquired after an already-completed force_close ran."""
    spawn.process.kill_tree()
    try:
        await _wait_process_bounded(spawn, _CLOSE_STAGE_TIMEOUT_SECONDS)
    finally:
        await spawn.stderr.stop()
        spawn.process.close_transports()


async def _wait_process_bounded(spawn: AcpSpawnResult, timeout_seconds: float) -> bool:
    if spawn.process.returncode is not None:
        return True
    try:
        async with asyncio.timeout(timeout_seconds):
            await spawn.process.wait()
    except TimeoutError:
        return False
    return True
