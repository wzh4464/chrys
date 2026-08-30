# Copyright (c) 2026 Chrys. All rights reserved.

"""MCP adapter — wraps MCP server tools as Chrys FunctionTools.

Uses Chrys-owned MCPStdioTool and MCPStreamableHTTPTool implementations to
connect to MCP servers and expose their tools as FunctionTool instances.

Failure model
-------------
``connect()`` raises :class:`MCPConnectionError` on failure; ``connect_all()``
catches ordinary per-server connection failures so one unavailable server
cannot block the agent build. Invalid or colliding tool names are configuration
errors and fail closed through :class:`MCPToolConfigurationError`; otherwise
the provider may reject the request or Chrys's last-wins tool map could execute
the wrong function. Ordinary failures are published to the ``EventBus`` as
``Warning`` events (``code=
"mcp.connect_failed"``) and retained on :attr:`MCPAdapter.failures`.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import inspect
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from html import escape as xml_escape
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol, TextIO, cast

from chrys.foundation.errors import clean_error_message
from chrys.foundation.i18n.formatting import sanitize_legacy_block, sanitize_legacy_scalar
from chrys.foundation.models.turns import current_turn_start, turn_slices
from chrys.foundation.platform.files import surrogate_safe_text
from chrys.foundation.platform.runtime_paths import reorder_path_demoting_runtime, strip_python_runtime_overrides
from chrys.foundation.text.tool_output import process_carriage_returns, strip_ansi
from chrys.foundation.tool_call_context import TOOL_CALL_CONTEXT_METADATA_KEY, get_tool_context, set_tool_context
from chrys.foundation.tool_kinds import KIND_MCP, set_tool_kind
from chrys.foundation.util.env_templates import resolve_env_template_mapping
from chrys.foundation.util.httpx_helpers import BYPASS_PROXY_MOUNTS
from chrys.kernel.exceptions import ToolExecutionException
from chrys.kernel.exchanges import (
    EmptyIdPolicy,
    LiveAccessor,
    NoneIdPolicy,
    PairingPolicy,
    iter_exchanges,
    pair_results,
)
from chrys.kernel.middleware import FunctionInvocationContext
from chrys.kernel.sessions import ContextProvider
from chrys.kernel.tools import FunctionTool, normalize_tools
from chrys.service.mcp.cache import MCPConnectionCache, MCPConnectionLease, clone_mcp_function_tool
from chrys.service.mcp.owned import (
    _MCP_NORMALIZED_NAME_KEY,
    _MCP_REMOTE_NAME_KEY,
    MCPStdioTool,
    MCPStreamableHTTPTool,
    _build_prefixed_mcp_name,
    _mcp_call_headers,
    _mcp_config_candidate_names,
    _url_origin,
)
from chrys.service.mcp.result_limits import (
    DEFAULT_MCP_TOOL_RESULT_MAX_TOKENS as DEFAULT_MCP_TOOL_RESULT_MAX_TOKENS,
)
from chrys.service.mcp.result_limits import (
    resolve_mcp_result_cap,
)
from chrys.service.mcp.validation import (
    MAX_MCP_EXPOSED_TOOL_NAME_LENGTH,
    MCP_PROGRESSIVE_CONTROL_TOOL_NAMES,
    validate_mcp_exposed_tool_name,
    validate_mcp_tool_loading_policy,
    validate_mcp_tool_name_prefix,
)

if TYPE_CHECKING:
    from pathlib import Path

    from chrys.foundation.events.bus import EventBus
    from chrys.kernel import AgentSession, Message, SessionContext
    from chrys.service.mcp.owned import MCPTool
    from chrys.service.profiles.agents.schema import MCPServerConfig

logger = logging.getLogger(__name__)

# Hard timeouts that bound how long a misbehaving MCP server can hold up the
# UI / agent build.  The MCP SDK's per-request timeout
# (``read_timeout_seconds``) defaults to ``None`` (wait forever), and a
# server that emits welcome banner text instead of a valid ``initialize``
# response will leave the client blocked indefinitely on
# ``response_stream_reader.receive()``.  These floors are injected into
# the MCP tool's ``request_timeout`` when the user didn't set one, so the
# SDK's own timeout fires from *inside* the framework's lifecycle-owner
# task.  We intentionally do **not** wrap ``__aenter__`` in
# ``asyncio.wait_for`` from the outside: the lifecycle owner is a
# separate ``asyncio.Task`` (see ``MCPTool._run_lifecycle_owner``) which
# ``wait_for`` cannot cancel safely (anyio cancel scopes are scoped per
# task and cross-task cancellation raises "Attempted to exit cancel scope
# in a different task").  ``MCPServerConfig.request_timeout`` (when set)
# always takes precedence — these are only the defaults applied when no
# per-server timeout is configured.
DEFAULT_TEST_TIMEOUT_SECONDS = 30
DEFAULT_CONNECT_TIMEOUT_SECONDS = 30

# Cap how many non-JSON banner lines we keep per stdio server so a chatty
# logger can't unboundedly grow our diagnostic state.
MAX_BANNER_LINES_CAPTURED = 10

# Cap each captured banner line's length so a single very long log line
# (e.g. a stack trace serialized to one row) can't blow up an error
# message or notification.
MAX_BANNER_LINE_CHARS = 200

# Keep only a bounded tail of stderr from each stdio child.  Capturing a tail
# (rather than a prefix) preserves the final exception and exit context in the
# common traceback case; 16 KiB is enough that launcher output such as
# ``uv``/``npm`` — one ``error:`` line followed by screens of hints — still
# keeps the ``error:`` line, while a noisy server cannot grow Chrys memory
# without bound.
MAX_STDIO_STDERR_BYTES_CAPTURED = 16 * 1024

# The full tail lives on ``MCPConnectionError.stderr_tail`` (structured
# consumers and tests read it there); the error *message* — which is what the
# warning toast, the MCP Test dialog and runtime details currently show —
# previews only the last few lines, each capped like banner lines.
MAX_STDIO_STDERR_PREVIEW_LINES = 10

# Display cap for filesystem-derived paths (executable, cwd) in error messages.
MAX_STDIO_PATH_DISPLAY_CHARS = 512

# Cap per-server InitializeResult.instructions rendered into the
# <mcp_instructions> reminder.  The value is server-controlled and repeated
# on every LLM call, so it must stay bounded no matter what the server
# sends.  The budget is charged against the *rendered* (escaped + indented)
# text — capping the raw input instead would let escape expansion multiply
# the cap (e.g. every '"' becomes '&quot;').  4k rendered characters fits
# real-world server instructions comfortably; the aggregate is bounded by
# the number of user-configured servers.
MCP_INSTRUCTIONS_CHAR_LIMIT = 4_000

# A trailing entity fragment left behind when a rendered line is cut at the
# budget boundary (e.g. '&quo').  Complete entities end in ';' and don't match.
_SEVERED_ENTITY_TAIL_RE = re.compile(r"&[#0-9A-Za-z]{0,9}$")
_MCP_PROGRESSIVE_CONTROL_MARKER = "_chrys_mcp_progressive_control"
(
    _MCP_PROGRESSIVE_LIST_TOOL_NAME,
    _MCP_PROGRESSIVE_LOAD_TOOL_NAME,
    _MCP_PROGRESSIVE_UNLOAD_TOOL_NAME,
) = MCP_PROGRESSIVE_CONTROL_TOOL_NAMES
_MAX_PROGRESSIVE_CONTROL_PREFIX_LENGTH = (
    MAX_MCP_EXPOSED_TOOL_NAME_LENGTH - 1 - max(len(name) for name in MCP_PROGRESSIVE_CONTROL_TOOL_NAMES)
)
_CONTROL_PREFIX_HASH_LENGTH = 10
_STDIO_ENV_OVERRIDE: ContextVar[dict[str, str] | None] = ContextVar("mcp_stdio_env_override", default=None)
_STDIO_CWD_OVERRIDE: ContextVar[str | None] = ContextVar("mcp_stdio_cwd_override", default=None)
_HTTP_HEADERS_OVERRIDE: ContextVar[dict[str, str] | None] = ContextVar("mcp_http_headers_override", default=None)
_STDIO_ENV_IS_COMPLETE: ContextVar[bool] = ContextVar("mcp_stdio_env_is_complete", default=False)


def _clean_stdio_stderr(text: str) -> str:
    """Normalize untrusted subprocess stderr for logs and UI diagnostics."""
    normalized = process_carriage_returns(strip_ansi(text))
    return sanitize_legacy_block(normalized).strip()


@dataclasses.dataclass(slots=True)
class _StdioProcessDiagnostics:
    """Bounded, mutable diagnostics populated by one stdio transport run."""

    _stderr_tail: bytearray = dataclasses.field(default_factory=bytearray)
    stderr_bytes_seen: int = 0
    exit_code: int | None = None
    resolved_executable: str = ""
    effective_cwd: str = ""

    def reset(self) -> None:
        """Reset state before a new child process is spawned."""
        self._stderr_tail.clear()
        self.stderr_bytes_seen = 0
        self.exit_code = None
        self.resolved_executable = ""
        self.effective_cwd = ""

    def record_spawn(self, *, resolved_executable: str, cwd: object) -> None:
        """Record what is about to be spawned and where (``None`` cwd = inherited)."""
        self.resolved_executable = resolved_executable
        if cwd is None or cwd == "":
            try:
                self.effective_cwd = os.getcwd()
            except OSError:
                self.effective_cwd = ""
        else:
            self.effective_cwd = str(cwd)

    def append_stderr(self, chunk: bytes) -> None:
        """Append bytes while retaining only the configured tail."""
        if not chunk:
            return
        self.stderr_bytes_seen += len(chunk)
        self._stderr_tail.extend(chunk)
        overflow = len(self._stderr_tail) - MAX_STDIO_STDERR_BYTES_CAPTURED
        if overflow > 0:
            del self._stderr_tail[:overflow]

    def record_exit_code(self, wait_result: object, process: Any) -> None:
        """Record a return code from an external anyio/MCP process wrapper."""
        exit_code = wait_result if type(wait_result) is int else getattr(process, "returncode", None)
        if type(exit_code) is not int:
            popen = getattr(process, "popen", None)
            exit_code = getattr(popen, "returncode", None) if popen is not None else None
        if type(exit_code) is int:
            self.exit_code = exit_code

    @property
    def stderr_tail(self) -> str:
        """Return a display-safe decoded copy of the retained stderr tail."""
        return _clean_stdio_stderr(bytes(self._stderr_tail).decode("utf-8", errors="replace"))

    @property
    def stderr_dropped_bytes(self) -> int:
        """Number of earlier stderr bytes omitted from the retained tail."""
        return max(0, self.stderr_bytes_seen - len(self._stderr_tail))


@dataclasses.dataclass(frozen=True, slots=True)
class _StdioFailureDiagnostics:
    """Immutable snapshot attached to an MCP connection failure."""

    stderr_tail: str = ""
    stderr_dropped_bytes: int = 0
    process_exit_code: int | None = None
    resolved_executable: str = ""
    effective_cwd: str = ""


def _display_path(value: str) -> str:
    """Make a raw filesystem path safe for a single-line message.

    Paths come from ``os.fsdecode`` and config, so they may carry lone
    surrogates (not UTF-8 encodable) or control characters (which could forge
    extra diagnostic lines).  The raw value stays on the exception attribute.
    """
    text = sanitize_legacy_scalar(surrogate_safe_text(value))
    if len(text) > MAX_STDIO_PATH_DISPLAY_CHARS:
        text = text[: MAX_STDIO_PATH_DISPLAY_CHARS - 1] + "…"
    return text


def _resolve_spawn_executable(command: str, env: Mapping[str, str]) -> str:
    """Best-effort path of what ``Popen`` will exec for *command* under *env*.

    Mirrors POSIX ``subprocess``: a command with a path component is used as
    is; otherwise the search uses the CHILD environment's ``PATH``
    (``os.get_exec_path(env)``), which is what decides "which ``uv``" when the
    TUI and a headless run carry different runtime paths.  Falls back to the
    unresolved command when nothing is found — the spawn itself is unchanged.
    """
    if os.path.dirname(command):
        return command
    try:
        found = shutil.which(command, path=os.pathsep.join(os.get_exec_path(env)))
    except OSError, TypeError, ValueError:
        return command
    return found or command


def _stderr_preview(stderr_tail: str, *, dropped_bytes: int) -> str:
    """Render the last few stderr lines for an error message / warning toast.

    The full tail stays on the exception; the message keeps at most
    :data:`MAX_STDIO_STDERR_PREVIEW_LINES` lines, each capped like banner
    lines, so a traceback cannot turn a toast into a wall of text.
    """
    lines = stderr_tail.splitlines()
    omitted_lines = max(0, len(lines) - MAX_STDIO_STDERR_PREVIEW_LINES)
    shown = [
        (line[:MAX_BANNER_LINE_CHARS] + "…") if len(line) > MAX_BANNER_LINE_CHARS else line
        for line in lines[-MAX_STDIO_STDERR_PREVIEW_LINES:]
    ]
    omitted: list[str] = []
    if omitted_lines:
        omitted.append(f"{omitted_lines} earlier lines")
    if dropped_bytes:
        omitted.append(f"{dropped_bytes} earlier bytes")
    suffix = f" ({' and '.join(omitted)} omitted)" if omitted else ""
    body = "\n  ".join(shown)
    return f"Server stderr tail{suffix}:\n  {body}"


def _mirror_no_proxy_aliases(env: dict[str, str], extra_keys: set[str]) -> None:
    """Keep ``NO_PROXY`` and ``no_proxy`` aligned for stdio child processes.

    User environments commonly set either canonical uppercase ``NO_PROXY``
    or lowercase ``no_proxy``.  Stdio MCP servers are a different boundary:
    the child command may be implemented by a runtime that only checks the
    lowercase proxy convention.  Mirror the value into the missing casing
    just before spawning, without mutating the parent process environment.

    Per-server MCP env values should win over inherited parent values.  When
    the user supplies only one casing in the per-server env, mirror that
    supplied value over the inherited alias too.  When the user explicitly
    supplies both casings, preserve both values unchanged.

    This deliberately handles only the proxy-bypass list, not proxy URL
    variables such as ``HTTP_PROXY`` / ``http_proxy``: those have different
    security and precedence behavior across runtimes, including the
    historical CGI ``HTTP_PROXY`` hazard.
    """
    upper = "NO_PROXY"
    lower = "no_proxy"

    if upper in extra_keys and lower not in extra_keys:
        env[lower] = env[upper]
        return
    if lower in extra_keys and upper not in extra_keys:
        env[upper] = env[lower]
        return
    if upper in env and lower not in env:
        env[lower] = env[upper]
        return
    if lower in env and upper not in env:
        env[upper] = env[lower]


def _inherited_stdio_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build the env passed to a spawned stdio MCP server subprocess.

    Unlike the MCP SDK's ``get_default_environment()`` — which inherits a
    strict ~6-var allowlist (``PATH`` / ``HOME`` / ``USER`` / ...) — chrys
    forwards nearly all of the parent ``os.environ`` so subprocesses see the
    user's proxy settings (``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``NO_PROXY``),
    TLS bundle paths (``SSL_CERT_FILE`` / ``SSL_CERT_DIR`` /
    ``REQUESTS_CA_BUNDLE``), locale (``LANG`` / ``LC_ALL``), language
    runtimes' caches, and other vars the user expects when running the same
    command from their shell.  Without this, a tool like ``uvx`` that reaches
    PyPI through a corporate proxy hangs forever, because the SDK default
    strips ``HTTPS_PROXY`` and the child never reaches the network — surfaced
    as an ``initialize`` timeout.

    Bash function exports (values starting with ``"()"``) are dropped, the
    same Shellshock-era filter the SDK applies.  Inherited ``PYTHONHOME`` and
    ``PYTHONPATH`` are also stripped so CPython-based server launchers are not
    forced to use the parent's stdlib/search path.  Optional ``extra``
    overrides — typically ``MCPServerConfig.env`` — are merged on top so
    per-server config wins over the sanitized inherited environment.  For
    packaged PyApp runs, Chrys runtime executable directories are moved behind
    user/system PATH entries so commands like ``uvx`` resolve the user's
    install first and fall back to the embedded runtime only if needed.
    """
    env: dict[str, str] = {k: v for k, v in os.environ.items() if not v.startswith("()")}
    strip_python_runtime_overrides(env)
    extra_keys = set(extra or ())
    if extra:
        env.update(extra)
    _mirror_no_proxy_aliases(env, extra_keys)
    reorder_path_demoting_runtime(env)
    return env


class MCPConnectionError(RuntimeError):
    """Raised when an MCP server fails to connect, initialize, or list tools.

    ``banner_lines`` carries any non-JSON output the stdio child wrote to
    its stdout before initialization (captured by ``tolerant_stdio_client``).
    Surfaced in the error string so the user can see at a glance that a
    server is emitting a banner instead of speaking JSON-RPC.

    Stdio process failures may also carry a bounded ``stderr_tail``,
    ``process_exit_code``, ``resolved_executable`` and ``effective_cwd``.
    Stderr is captured separately from stdout so server diagnostics can never
    corrupt the JSON-RPC protocol stream; the message previews only the last
    :data:`MAX_STDIO_STDERR_PREVIEW_LINES` lines of it.
    """

    def __init__(
        self,
        server_name: str,
        transport: str,
        cause: BaseException | None = None,
        banner_lines: list[str] | None = None,
        *,
        failure_summary: str = "failed to connect",
        stderr_tail: str = "",
        stderr_dropped_bytes: int = 0,
        process_exit_code: int | None = None,
        resolved_executable: str = "",
        effective_cwd: str = "",
    ) -> None:
        self.server_name = server_name
        self.transport = transport
        self.cause = cause
        # Truncate each line so a single very long banner row (e.g. a
        # serialized stack trace) can't blow up the error message.
        self.banner_lines: list[str] = (
            [
                (line[:MAX_BANNER_LINE_CHARS] + "…") if len(line) > MAX_BANNER_LINE_CHARS else line
                for line in banner_lines
            ]
            if banner_lines
            else []
        )
        self.stderr_tail = _clean_stdio_stderr(stderr_tail)[-MAX_STDIO_STDERR_BYTES_CAPTURED:]
        self.stderr_dropped_bytes = max(0, stderr_dropped_bytes)
        self.process_exit_code = process_exit_code
        self.resolved_executable = resolved_executable
        self.effective_cwd = effective_cwd
        detail = f": {clean_error_message(cause)}" if cause is not None else ""
        message = f"MCP server {server_name!r} ({transport}) {failure_summary}{detail}"
        context_lines = []
        if self.resolved_executable:
            context_lines.append(f"Executable: {_display_path(self.resolved_executable)}")
        if self.effective_cwd:
            context_lines.append(f"Working directory: {_display_path(self.effective_cwd)}")
        if context_lines:
            message = f"{message}\n\n" + "\n".join(context_lines)
        if self.process_exit_code == 0:
            message = f"{message}\n\nServer process exited normally (code 0) before completing the MCP handshake."
        elif self.process_exit_code is not None:
            message = f"{message}\n\nServer process exit code: {self.process_exit_code}."
        if self.stderr_tail:
            preview = _stderr_preview(self.stderr_tail, dropped_bytes=self.stderr_dropped_bytes)
            message = f"{message}\n\n{preview}"
        if self.banner_lines:
            preview = "\n  ".join(self.banner_lines)
            message = f"{message}\n\nServer emitted non-JSON output before initialization:\n  {preview}"
        super().__init__(message)


class MCPToolConfigurationError(MCPConnectionError):
    """Base class for deterministic MCP tool-namespace configuration errors."""


class MCPToolNameCollisionError(MCPToolConfigurationError):
    """Raised when a permitted MCP tool namespace is not globally unique."""

    def __init__(
        self,
        server_name: str,
        transport: str,
        *,
        conflicting_names: Collection[str],
        conflict_with: str,
        guidance: str,
    ) -> None:
        self.conflicting_names = tuple(sorted(set(conflicting_names)))
        self.conflict_with = conflict_with
        names = ", ".join(self.conflicting_names)
        cause = ValueError(f"permitted tool name collision with {conflict_with}: {names}. {guidance}")
        super().__init__(
            server_name,
            transport,
            cause,
            failure_summary="has invalid tool configuration",
        )


class MCPToolNameValidationError(MCPToolConfigurationError):
    """Raised when an exposed MCP tool name is not accepted by model providers."""

    def __init__(
        self,
        server_name: str,
        transport: str,
        *,
        violations: Collection[str],
    ) -> None:
        self.violations = tuple(sorted(set(violations)))
        cause = ValueError("provider-incompatible MCP tool name: " + "; ".join(self.violations))
        super().__init__(
            server_name,
            transport,
            cause,
            failure_summary="has invalid tool configuration",
        )


@dataclasses.dataclass(frozen=True)
class MCPTestReport:
    """What a successful one-shot connection test learned about a server.

    Every string field is server-controlled: renderers must sanitize each
    one against markup/structure injection before display. ``capabilities``
    holds dotted feature names the server advertised (``tools``,
    ``tools.listChanged``, …) flattened from ``ServerCapabilities``.
    ``initial_tool_names`` is the progressive-disclosure starting surface,
    or ``None`` when progressive disclosure is off.
    """

    server_name: str | None
    server_title: str | None
    server_version: str | None
    protocol_version: str | None
    capabilities: tuple[str, ...]
    instructions: str | None
    tools: tuple[tuple[str, str], ...]
    prompts: tuple[tuple[str, str], ...]
    initial_tool_names: tuple[str, ...] | None


# Standard MCP server capability groups (the spec's named feature sets).
# The report renderer always shows these; the flattener records them ahead
# of extras so the entry cap can never make a declared one look
# unadvertised.
STANDARD_CAPABILITY_GROUPS = ("completions", "logging", "prompts", "resources", "tools")

# Bounds for flattening server-controlled capability trees: a hostile
# server could otherwise inflate the report with unbounded extra keys or
# deep nesting (``ServerCapabilities`` accepts extras).
_MAX_CAPABILITY_ENTRIES = 128
_MAX_CAPABILITY_DEPTH = 4


def _top_capability_order(key: object) -> tuple[bool, str]:
    return (str(key) not in STANDARD_CAPABILITY_GROUPS, str(key))


def _flatten_server_capabilities(capabilities: Any) -> tuple[str, ...]:
    """Flatten advertised ``ServerCapabilities`` into sorted dotted feature names.

    Walks nested capability objects (e.g. ``tasks.requests.tools``):
    ``False`` leaves are omitted, any other advertised value marks presence.
    Traversal is breadth-first with standard groups first at the top level,
    so the entry cap only ever drops the deepest/extra branches — a declared
    standard capability is always recorded.
    """
    if capabilities is None:
        return ()
    try:
        dumped = capabilities.model_dump(exclude_none=True)
    except Exception:
        return ()
    if not isinstance(dumped, dict):
        return ()
    flattened: list[str] = []
    queue: deque[tuple[str, dict[Any, Any], int]] = deque([("", dumped, 1)])
    while queue:
        prefix, mapping, depth = queue.popleft()
        for key in sorted(mapping, key=_top_capability_order if depth == 1 else str):
            if len(flattened) >= _MAX_CAPABILITY_ENTRIES:
                return tuple(sorted(flattened))
            value = mapping[key]
            if value is False or value is None:
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.append(path)
            if isinstance(value, dict) and depth < _MAX_CAPABILITY_DEPTH:
                queue.append((path, value, depth + 1))
    return tuple(sorted(flattened))


class _NoPrePagePingMixin:
    """Disable the owned engine's pre-pagination ``send_ping`` health check.

    Upstream ``MCPTool._ensure_connected`` sends ``ping`` before every
    ``list_tools`` / ``list_prompts`` page and treats *any* failure (including
    ``method_not_found`` from servers that don't implement the optional ``ping``
    utility) as a dead session — triggering ``connect(reset=True)``, which
    closes the transport (DELETE on Streamable HTTP) and re-handshakes (POST
    initialize + background GET). With a non-implementing server this recurses
    into a tight POST → GET → DELETE loop until ``request_timeout`` fires.

    The check is also redundant: ``_ensure_connected`` only runs immediately
    after ``session.initialize()`` succeeds, when the session is freshly
    handshaken. Real transport loss surfaces from the actual ``list_*`` call as
    ``ClosedResourceError`` and propagates up to ``__aenter__`` cleanly.
    Runtime tool calls (``call_tool`` / ``get_prompt``) have their own
    reconnect-on-``ClosedResourceError`` paths that don't go through
    ``_ensure_connected``.
    """

    async def _ensure_connected(self) -> None:
        return


def _content_has_meaningful_payload(content: list[Any]) -> bool:
    """Return True if ``content`` carries any non-empty/non-text payload.

    Used to decide whether to fall back to ``CallToolResult.structuredContent``.
    A list that is empty, or that contains only whitespace-only ``TextContent``
    items, is considered to have no meaningful payload — those are the cases
    where a server is relying on the structured field to deliver its result.
    Any non-text item (image, audio, resource, link, etc.) counts as
    meaningful regardless of whether ``structuredContent`` is also set, since
    the framework parser already renders those usefully.
    """
    from mcp import types

    for item in content:
        if isinstance(item, types.TextContent):
            if (item.text or "").strip():
                return True
        else:
            return True
    return False


if TYPE_CHECKING:

    class _StructuredContentFallbackBase(Protocol):
        def _parse_tool_result_from_mcp(self, mcp_type: Any) -> Any: ...

else:
    _StructuredContentFallbackBase = object


class _StructuredContentFallbackMixin(_StructuredContentFallbackBase):
    """Surface ``CallToolResult.structuredContent`` when ``content`` is empty.

    Per the MCP spec, a server may return its payload in
    ``structuredContent`` (a JSON object) and leave ``content`` empty or
    populated with only an empty text fallback for clients that don't yet
    consume the structured field.  Upstream
    ``MCPTool._parse_tool_result_from_mcp`` walks ``content`` exclusively
    and never inspects ``structuredContent`` — so chrys, which threads
    rendered text back to the LLM, ends up with an empty tool result.

    This mixin overrides the parser to detect the "no meaningful content"
    case and return a single ``Content.from_text`` carrying a JSON dump
    of ``structuredContent`` instead.  When ``content`` already has a
    meaningful payload, the framework parser is used unchanged — a
    server that emits both a populated text fallback *and*
    ``structuredContent`` keeps its existing rendering (the structured
    field is a duplicate in that case).
    """

    def _parse_tool_result_from_mcp(self, mcp_type: Any) -> Any:
        from chrys.kernel import Content

        structured = getattr(mcp_type, "structuredContent", None)
        content = list(getattr(mcp_type, "content", None) or [])
        if structured is not None and not _content_has_meaningful_payload(content):
            try:
                payload = json.dumps(structured, default=str, ensure_ascii=False)
            except TypeError, ValueError:
                payload = str(structured)
            return [Content.from_text(payload)]
        return super()._parse_tool_result_from_mcp(mcp_type)  # type: ignore[misc]


@asynccontextmanager
async def _chrys_streamable_http_client(
    url: str,
    *,
    http_client: Any = None,
    terminate_on_close: bool = True,
    request_timeout: float | None = None,
) -> Any:
    """Streamable HTTP client that wakes pending requests on POST failures.

    Upstream ``StreamableHTTPTransport`` logs exceptions raised by POST
    request tasks, but the pending ``ClientSession.send_request`` can remain
    blocked waiting for a JSON-RPC response.  Convert request-scoped HTTP
    failures into JSON-RPC errors so initialization/list-tools fail through
    the normal MCP error path instead of hanging the agent build.

    Compatibility note: this is intentionally a narrow patch over MCP SDK
    1.28.1's ``StreamableHTTPTransport._handle_post_request`` private hook.
    If those private symbols move, fall back to the stock SDK client so a
    dependency update degrades to upstream behavior instead of breaking all
    HTTP MCP connections.
    """
    try:
        import anyio
        from mcp.client.streamable_http import StreamableHTTPTransport
        from mcp.shared._httpx_utils import create_mcp_http_client
        from mcp.shared.message import SessionMessage
        from mcp.types import CONNECTION_CLOSED, ErrorData, JSONRPCError, JSONRPCMessage, JSONRPCRequest
    except ImportError:
        logger.warning(
            "HTTP MCP POST-failure wakeup patch is disabled because MCP SDK internals changed.",
            exc_info=True,
        )
        from mcp.client.streamable_http import streamable_http_client

        async with streamable_http_client(
            url,
            http_client=http_client,
            terminate_on_close=terminate_on_close,
        ) as transport:
            yield transport
        return

    if not hasattr(StreamableHTTPTransport, "_handle_post_request"):
        logger.warning("HTTP MCP POST-failure wakeup patch is disabled because MCP SDK internals changed.")
        from mcp.client.streamable_http import streamable_http_client

        async with streamable_http_client(
            url,
            http_client=http_client,
            terminate_on_close=terminate_on_close,
        ) as transport:
            yield transport
        return

    class _ChrysStreamableHTTPTransport(StreamableHTTPTransport):
        def __init__(self, url: str, *, request_timeout: float | None = None) -> None:
            super().__init__(url)
            self._request_timeout = request_timeout

        @property
        def _post_timeout(self) -> float | None:
            if self._request_timeout is None:
                return None
            margin = min(0.25, self._request_timeout / 4)
            return max(0.001, self._request_timeout - margin)

        async def _handle_post_request(self, ctx: Any) -> None:
            from httpx import Timeout, TimeoutException, codes
            from mcp.client.streamable_http import CONTENT_TYPE, JSON, SSE

            headers = self._prepare_headers()
            message = ctx.session_message.message
            is_initialization = self._is_initialization_request(message)
            stream_kwargs: dict[str, Any] = {
                "json": message.model_dump(by_alias=True, mode="json", exclude_none=True),
                "headers": headers,
            }
            post_timeout = self._post_timeout
            if post_timeout is not None:
                stream_kwargs["timeout"] = Timeout(post_timeout)

            try:
                async with ctx.client.stream("POST", self.url, **stream_kwargs) as response:
                    if response.status_code == 202:
                        return

                    if response.status_code == 404:
                        if isinstance(message.root, JSONRPCRequest):
                            await self._send_session_terminated_error(ctx.read_stream_writer, message.root.id)
                        return

                    response.raise_for_status()
                    if is_initialization:
                        self._maybe_extract_session_id_from_response(response)

                    if isinstance(message.root, JSONRPCRequest):
                        content_type = response.headers.get(CONTENT_TYPE, "").lower()
                        if content_type.startswith(JSON):
                            await self._handle_json_response(response, ctx.read_stream_writer, is_initialization)
                        elif content_type.startswith(SSE):
                            await self._handle_sse_response(response, ctx, is_initialization)
                        else:
                            await self._handle_unexpected_content_type(content_type, ctx.read_stream_writer)
            except Exception as exc:
                root = getattr(message, "root", None)
                if isinstance(root, JSONRPCRequest):
                    detail = clean_error_message(exc)
                    code = codes.REQUEST_TIMEOUT if isinstance(exc, TimeoutException) else CONNECTION_CLOSED
                    error = JSONRPCError(
                        jsonrpc="2.0",
                        id=root.id,
                        error=ErrorData(
                            code=code,
                            message=f"HTTP MCP request failed: {detail}",
                        ),
                    )
                    await ctx.read_stream_writer.send(SessionMessage(message=JSONRPCMessage(error)))
                    return
                raise

    read_stream_writer, read_stream = anyio.create_memory_object_stream[Any](0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream[Any](0)

    client_provided = http_client is not None
    client = http_client or create_mcp_http_client()
    transport = _ChrysStreamableHTTPTransport(url, request_timeout=request_timeout)

    async with anyio.create_task_group() as tg:
        try:
            async with contextlib.AsyncExitStack() as stack:
                if not client_provided:
                    await stack.enter_async_context(client)

                def start_get_stream() -> None:
                    tg.start_soon(transport.handle_get_stream, client, read_stream_writer)

                tg.start_soon(
                    transport.post_writer,
                    client,
                    write_stream_reader,
                    read_stream_writer,
                    write_stream,
                    start_get_stream,
                    tg,
                )

                try:
                    yield read_stream, write_stream, transport.get_session_id
                finally:
                    if transport.session_id and terminate_on_close:
                        await transport.terminate_session(client)
                    tg.cancel_scope.cancel()
        finally:
            await read_stream_writer.aclose()
            await write_stream.aclose()


class _HTTPMCPTool(_NoPrePagePingMixin, _StructuredContentFallbackMixin, MCPStreamableHTTPTool):
    """MCPStreamableHTTPTool with TLS/proxy options and static headers.

    Three profile knobs:

    * ``verify_ssl=False`` — disables TLS cert validation. Insecure;
      local/self-signed dev only.
    * ``bypass_proxy=True`` — disables environment-derived proxy transports
      for this server.
    * ``headers`` — static per-server headers (typically auth tokens)
      attached by a same-origin request hook, so they reach **every**
      request including ``initialize`` and ``list_tools`` without leaking
      across cross-origin redirects. Bypasses the framework's
      ``header_provider``, which only fires inside ``call_tool()``.

    When any knob is non-default we pre-build the ``httpx.AsyncClient``
    and hand it to the parent via ``self._httpx_client``.

    When all knobs are default and no ``http_client`` is supplied, we
    skip the pre-build entirely so upstream defaults stay in force.
    """

    def __init__(
        self,
        *args: Any,
        verify_ssl: bool = True,
        bypass_proxy: bool = False,
        headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._verify_ssl: bool = verify_ssl
        self._bypass_proxy: bool = bypass_proxy
        self._static_headers: dict[str, str] = dict(headers) if headers else {}
        self._owned_httpx_client: Any = None
        super().__init__(*args, **kwargs)

    async def __aenter__(self) -> Any:
        if self._needs_prebuild() and self._httpx_client is None:
            client = self._build_httpx_client()
            self._owned_httpx_client = client
            self._httpx_client = client
        try:
            return await super().__aenter__()
        except BaseException:
            await self._close_owned_httpx_client()
            raise

    def get_mcp_client(self) -> Any:
        """Build Streamable HTTP transport with request-failure wakeups."""
        # Preserve the owned parent dynamic-header behavior while swapping
        # only the MCP streamable HTTP transport for _chrys_streamable_http_client.
        # _url_origin guards against replaying per-server and per-call secrets
        # across a cross-origin redirect.
        from httpx import AsyncClient, Timeout

        from chrys.service.mcp.owned import MCP_DEFAULT_SSE_READ_TIMEOUT, MCP_DEFAULT_TIMEOUT

        http_client = self._httpx_client
        needs_header_hook = bool(self._static_headers) or self._header_provider is not None
        if http_client is None:
            if self._needs_prebuild():
                http_client = self._build_httpx_client()
                self._owned_httpx_client = http_client
            elif needs_header_hook:
                http_client = AsyncClient(
                    follow_redirects=True,
                    timeout=Timeout(MCP_DEFAULT_TIMEOUT, read=MCP_DEFAULT_SSE_READ_TIMEOUT),
                )
            self._httpx_client = http_client

        if http_client is not None and needs_header_hook:
            self._ensure_header_hook(http_client)

        return _chrys_streamable_http_client(
            url=self.url,
            http_client=http_client,
            terminate_on_close=self.terminate_on_close if self.terminate_on_close is not None else True,
            request_timeout=float(self.request_timeout) if self.request_timeout is not None else None,
        )

    def _ensure_header_hook(self, http_client: Any) -> None:
        if self._inject_headers_hook is not None:
            return

        from httpx import URL

        target_origin = _url_origin(URL(self.url))

        async def _inject_headers(request: Any) -> None:
            dynamic_headers = _mcp_call_headers.get({})
            if _url_origin(request.url) != target_origin:
                for key in self._static_headers:
                    request.headers.pop(key, None)
                for key in dynamic_headers:
                    request.headers.pop(key, None)
                return
            for key, value in self._static_headers.items():
                request.headers[key] = value
            if self._header_provider is None:
                return
            for key, value in dynamic_headers.items():
                request.headers[key] = value

        self._inject_headers_hook = _inject_headers
        http_client.event_hooks["request"].append(self._inject_headers_hook)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            await super().__aexit__(exc_type, exc_value, traceback)
        finally:
            await self._close_owned_httpx_client()

    def _needs_prebuild(self) -> bool:
        return (
            not self._verify_ssl or self._bypass_proxy or bool(self._static_headers) or self.request_timeout is not None
        )

    def _build_httpx_client(self) -> Any:
        """Construct an ``httpx.AsyncClient`` mirroring MCP defaults.

        We construct ``httpx.AsyncClient`` directly (rather than via
        ``mcp.shared._httpx_utils.create_mcp_http_client``) because we
        need to thread ``verify=`` through, which the factory does not
        accept.  Timeout constants are pulled from the same module so we
        stay aligned with upstream defaults; if that
        private module changes shape, we fall back to the historical
        literals (30.0s connect / 300.0s SSE read) instead of crashing.

        When ``request_timeout`` is set we narrow connect/write/pool to
        it so a dead HTTP server fails inside the user's bound instead
        of waiting out the 30s httpx default — Linux returns ECONNREFUSED
        instantly so this only bites on Windows, but the unbounded
        connect violates the user's intent on every platform.  SSE read
        stays at ``MCP_DEFAULT_SSE_READ_TIMEOUT`` so long-lived streams
        aren't capped by the per-request bound.
        """
        from httpx import AsyncClient, Timeout

        try:
            from mcp.shared._httpx_utils import MCP_DEFAULT_SSE_READ_TIMEOUT, MCP_DEFAULT_TIMEOUT
        except ImportError:
            logger.warning(
                "MCP default timeout constants are unavailable (mcp.shared._httpx_utils changed); "
                "using literal fallbacks (30.0s connect / 300.0s SSE read).",
                exc_info=True,
            )
            MCP_DEFAULT_TIMEOUT = 30.0
            MCP_DEFAULT_SSE_READ_TIMEOUT = 300.0

        connect_timeout = float(self.request_timeout) if self.request_timeout is not None else MCP_DEFAULT_TIMEOUT
        kwargs: dict[str, Any] = {
            "follow_redirects": True,
            "timeout": Timeout(connect_timeout, read=MCP_DEFAULT_SSE_READ_TIMEOUT),
            "verify": self._verify_ssl,
        }
        if self._bypass_proxy:
            kwargs["mounts"] = dict(BYPASS_PROXY_MOUNTS)
        return AsyncClient(**kwargs)

    async def _close_owned_httpx_client(self) -> None:
        client = self._owned_httpx_client
        if client is None:
            return
        self._owned_httpx_client = None
        # Drop the framework's reference too so a second __aenter__ rebuilds
        # under fresh env (and doesn't reuse a closed client).
        if self._httpx_client is client:
            self._httpx_client = None
        # Header injection is attached once per client.  Clear the sentinel so
        # the hook re-attaches to the next owned client on re-enter.
        self._inject_headers_hook = None
        with contextlib.suppress(Exception):
            await client.aclose()


@asynccontextmanager
async def tolerant_stdio_client(
    server: Any,
    errlog: TextIO = sys.stderr,
    *,
    dropped_banner_lines: list[str] | None = None,
    process_diagnostics: _StdioProcessDiagnostics | None = None,
    inherit_env: bool = True,
) -> AsyncIterator[tuple[Any, Any]]:
    """``mcp.client.stdio.stdio_client`` variant that ignores banner lines.

    Many real-world stdio MCP servers (vendor CLI wrappers, several
    ``npx`` packages, locally-built tools) violate the spec by emitting
    human-readable startup chatter on stdout before they begin speaking
    JSON-RPC.
    Upstream ``stdio_client`` treats every line as a JSON-RPC message and
    pushes a parse exception onto the read stream when it fails — but the
    base session quietly drops those exceptions, so the pending
    ``initialize()`` waits forever.  The Test button hangs and the agent
    build hangs.

    This wrapper classifies each stdout line into three buckets:

    * **Empty/whitespace-only** — skipped silently (no log, not captured).
    * **Not JSON at all** (``json.loads`` fails) — treated as banner
      chatter.  Dropped.  The first ``MAX_BANNER_LINES_CAPTURED`` banner
      lines are appended to ``dropped_banner_lines`` (when supplied) so
      callers can include them in error messages, and the first one is
      logged at WARNING.  This catches plaintext banners like
      ``"FooBarServer v1.0 (build 42)"`` and bracketed log lines like
      ``"[INFO] starting"``.
    * **Valid JSON but fails JSON-RPC validation** — kept as the
      original "push exception onto read stream" behavior.  Note this
      doesn't surface immediately: ``BaseSession._receive_loop`` routes
      stream exceptions through ``_handle_incoming`` which the default
      handler swallows.  The pending ``initialize()`` then waits until
      ``read_timeout_seconds`` (set via ``request_timeout``) fires.  Our
      ``MCPAdapter`` injects a default ``request_timeout`` floor when
      none is configured, so this case is bounded — but it's "fails
      eventually under the timeout floor", not "fails fast".

    Process lifecycle and shutdown mirror upstream ``stdio_client``.  In
    addition to line handling and the inherited environment, stderr is piped
    into a bounded diagnostic tail and the subprocess return code is recorded
    during shutdown.  Stderr is always drained concurrently so a verbose child
    cannot block on a full pipe.

    This reader is built on private ``mcp.client.stdio`` primitives
    (``_create_platform_compatible_process`` / ``_get_executable_command``
    / ``_terminate_process_tree`` / ``PROCESS_TERMINATION_TIMEOUT``) plus the
    SDK's ``SessionMessage`` wrapper.  If a dependency update removes any of
    them, we degrade to the stock ``stdio_client`` — losing banner tolerance
    and bounded process diagnostics — instead of breaking every stdio MCP
    connection.  The sanitized inherited environment is still forwarded in
    that path by copying it onto the server params.
    """
    import anyio
    import anyio.lowlevel
    from anyio import to_thread
    from anyio.streams.text import TextReceiveStream
    from mcp import types

    diagnostics = process_diagnostics or _StdioProcessDiagnostics()
    diagnostics.reset()

    try:
        from mcp.client.stdio import (
            PROCESS_TERMINATION_TIMEOUT,
            _create_platform_compatible_process,
            _get_executable_command,
            _terminate_process_tree,
        )
        from mcp.shared.message import SessionMessage
    except ImportError:
        logger.warning(
            "Tolerant stdio MCP reader is disabled because MCP SDK stdio internals changed; "
            "falling back to the stock stdio client (stdout banner and process diagnostics lost).",
            exc_info=True,
        )
        from mcp.client.stdio import stdio_client

        stock_server = (
            server.model_copy(update={"env": _inherited_stdio_environment(server.env)}) if inherit_env else server
        )
        async with stdio_client(stock_server, errlog=errlog) as streams:
            yield streams
        return

    captured = dropped_banner_lines if dropped_banner_lines is not None else []
    warned_once = False

    read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
    write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

    try:
        command = _get_executable_command(server.command)
        env = _inherited_stdio_environment(server.env) if inherit_env else dict(server.env or {})
        diagnostics.record_spawn(resolved_executable=_resolve_spawn_executable(command, env), cwd=server.cwd)
        # The SDK types ``errlog`` as ``TextIO`` but hands it straight to
        # ``anyio.open_process`` / ``subprocess.Popen(stderr=...)`` on every
        # platform path, so the ``PIPE`` sentinel is accepted at runtime.
        process = await _create_platform_compatible_process(
            command=command,
            args=server.args,
            env=env,
            errlog=cast("TextIO", subprocess.PIPE),
            cwd=server.cwd,
        )
    except OSError:
        await read_stream.aclose()
        await write_stream.aclose()
        await read_stream_writer.aclose()
        await write_stream_reader.aclose()
        raise

    async def stdout_reader() -> None:
        nonlocal warned_once
        assert process.stdout, "Opened process is missing stdout"
        try:
            async with read_stream_writer:
                buffer = ""
                async for chunk in TextReceiveStream(
                    process.stdout,
                    encoding=server.encoding,
                    errors=server.encoding_error_handler,
                ):
                    lines = (buffer + chunk).split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        stripped = line.lstrip()
                        if not stripped:
                            # Empty/whitespace-only: silently ignore.
                            continue

                        try:
                            parsed = json.loads(line)
                        except json.JSONDecodeError:
                            # Not JSON at all — banner chatter.  Capture a
                            # few for diagnostics, then drop.
                            if len(captured) < MAX_BANNER_LINES_CAPTURED:
                                captured.append(line.rstrip("\r"))
                            if not warned_once:
                                warned_once = True
                                logger.warning(
                                    "MCP stdio server emitted non-JSON line on stdout (dropped): %r",
                                    line[:200],
                                )
                            else:
                                logger.debug(
                                    "MCP stdio server emitted non-JSON line on stdout (dropped): %r", line[:200]
                                )
                            continue

                        try:
                            message = types.JSONRPCMessage.model_validate(parsed)
                        except Exception as exc:
                            # Valid JSON but failed JSON-RPC validation.
                            # Push onto the read stream where the SDK's
                            # ``_handle_incoming`` will eventually surface
                            # it (in practice — once the pending request
                            # times out under our injected
                            # ``request_timeout`` floor).
                            logger.exception("Failed to parse JSONRPC message from server")
                            await read_stream_writer.send(exc)
                            continue

                        await read_stream_writer.send(SessionMessage(message))
        except anyio.ClosedResourceError, anyio.BrokenResourceError:
            # ClosedResourceError: our writer was closed (normal shutdown).
            # BrokenResourceError: the SDK closed the receive end before we
            # finished draining stdout — happens during failed-connect
            # cleanup, where letting this escape would mask the real
            # initialize() exception with an ExceptionGroup.
            await anyio.lowlevel.checkpoint()

    async def stdin_writer() -> None:
        assert process.stdin, "Opened process is missing stdin"
        try:
            async with write_stream_reader:
                async for session_message in write_stream_reader:
                    json_text = session_message.message.model_dump_json(by_alias=True, exclude_none=True)
                    await process.stdin.send(
                        (json_text + "\n").encode(
                            encoding=server.encoding,
                            errors=server.encoding_error_handler,
                        )
                    )
        except anyio.ClosedResourceError, anyio.BrokenResourceError:
            await anyio.lowlevel.checkpoint()

    stderr_drained = anyio.Event()

    async def stderr_reader() -> None:
        """Drain and retain a bounded tail from anyio or Windows fallback stderr."""
        try:
            stderr = getattr(process, "stderr", None)
            if stderr is None:
                return

            receive = getattr(stderr, "receive", None)
            if receive is not None:
                try:
                    while True:
                        chunk = await receive()
                        if chunk:
                            diagnostics.append_stderr(bytes(chunk))
                except anyio.EndOfStream, anyio.ClosedResourceError, anyio.BrokenResourceError, OSError, ValueError:
                    return

            # The SDK's Windows ``FallbackProcess`` exposes the raw Popen
            # stderr file.  Prefer ``read1`` (returns whatever is available)
            # over ``read`` (blocks until the size is filled or EOF) so a
            # hanging server still shows what it wrote so far.
            read = getattr(stderr, "read1", None) or getattr(stderr, "read", None)
            if read is None:
                return
            try:
                while chunk := await to_thread.run_sync(read, 64 * 1024):
                    if inspect.isawaitable(chunk):
                        # Not a sync file after all; never loop on an awaitable.
                        # Only real coroutine objects need (and have) close().
                        if inspect.iscoroutine(chunk):
                            chunk.close()
                        return
                    diagnostics.append_stderr(chunk if isinstance(chunk, bytes) else str(chunk).encode())
            except OSError, ValueError:
                return
        finally:
            stderr_drained.set()

    async with (
        anyio.create_task_group() as tg,
        process,
    ):
        tg.start_soon(stdout_reader)
        tg.start_soon(stdin_writer)
        tg.start_soon(stderr_reader)
        try:
            yield read_stream, write_stream
        finally:
            if process.stdin:
                with contextlib.suppress(Exception):
                    await process.stdin.aclose()
            wait_result: object = None
            try:
                try:
                    with anyio.fail_after(PROCESS_TERMINATION_TIMEOUT):
                        wait_result = await process.wait()
                except TimeoutError:
                    await _terminate_process_tree(process)
                    try:
                        with anyio.fail_after(PROCESS_TERMINATION_TIMEOUT):
                            wait_result = await process.wait()
                    except TimeoutError, ProcessLookupError:
                        pass
                except ProcessLookupError:
                    pass
            finally:
                diagnostics.record_exit_code(wait_result, process)
                # wait() only proves the child exited; unread bytes may still
                # be buffered in its stderr pipe.  Let the reader drain before
                # process.__aexit__ closes that stream.  A descendant can keep
                # the pipe open after the direct child exits, so this wait must
                # remain bounded.
                with anyio.move_on_after(PROCESS_TERMINATION_TIMEOUT):
                    await stderr_drained.wait()
            await read_stream.aclose()
            await write_stream.aclose()
            await read_stream_writer.aclose()
            await write_stream_reader.aclose()


class _SafeStdioTool(_NoPrePagePingMixin, _StructuredContentFallbackMixin, MCPStdioTool):
    """MCPStdioTool that hardens the stdio transport for chrys.

    Two behaviors layered on top of upstream ``MCPStdioTool``:

    1. **Bounded stderr capture.**  Upstream ``stdio_client`` defaults
       ``errlog=sys.stderr``, inheriting the parent's stderr fd into the
       spawned child.  Inside the Chrys TUI (especially on Python 3.14),
       ``sys.stderr.fileno()`` may raise or return a fd unusable for
       subprocess inheritance, causing ``anyio.open_process`` to fail
       outright.  Even when the fd is valid, child stderr bytes corrupt
       the Chrys TUI frame.  The tolerant transport instead pipes and drains
       stderr into a bounded tail, records the exit code, and uses devnull only
       for the compatibility fallback to the stock SDK client.

    2. **Tolerant stdout reader.**  Many stdio MCP servers (vendor CLI
       wrappers, some ``npx`` packages) emit human-readable banner text on stdout before
       speaking JSON-RPC.  Upstream's reader pushes a parse exception
       onto the read stream — which the base session silently drops —
       leaving the pending ``initialize()`` blocked forever.  We swap in
       :func:`tolerant_stdio_client` which ignores banner lines and
       captures them on ``dropped_banner_lines`` so the adapter can
       surface them in error messages.
    """

    def __init__(self, *args: Any, env_is_complete: bool = False, **kwargs: Any) -> None:
        self._errlog_file: io.TextIOWrapper | None = None
        self.dropped_banner_lines: list[str] = []
        self._process_diagnostics = _StdioProcessDiagnostics()
        self._env_is_complete = env_is_complete
        super().__init__(*args, **kwargs)

    def get_mcp_client(self) -> Any:
        """Build a tolerant stdio transport with bounded process diagnostics."""
        args: dict[str, Any] = {
            "command": self.command,
            "args": self.args,
            "env": self.env,
        }
        if self.encoding:
            args["encoding"] = self.encoding
        if self._client_kwargs:
            args.update(self._client_kwargs)
        try:
            from mcp.client.stdio import StdioServerParameters
        except ModuleNotFoundError as ex:
            raise ModuleNotFoundError("`mcp` is required to use `MCPStdioTool`. Please install `mcp`.") from ex

        return tolerant_stdio_client(
            server=StdioServerParameters(**args),
            errlog=self._devnull_errlog(),
            dropped_banner_lines=self.dropped_banner_lines,
            process_diagnostics=self._process_diagnostics,
            inherit_env=not self._env_is_complete,
        )

    def _devnull_errlog(self) -> io.TextIOWrapper:
        """Lazily open the stock-client fallback's safe stderr target."""
        if self._errlog_file is None or self._errlog_file.closed:
            self._errlog_file = open(os.devnull, "w")  # noqa: SIM115
        return self._errlog_file

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Clean up the stock-client fallback's devnull handle on exit."""
        try:
            await super().__aexit__(exc_type, exc_value, traceback)
        finally:
            if self._errlog_file is not None and not self._errlog_file.closed:
                self._errlog_file.close()
                self._errlog_file = None


def _validate_config(config: MCPServerConfig) -> None:
    """Reject malformed configs before we hand them to the MCP framework."""
    if not config.name:
        raise ValueError("MCP server config requires a non-empty 'name'")
    if not isinstance(config.use_progressive_disclosure, bool):
        raise ValueError(f"MCP server {config.name!r}: 'use_progressive_disclosure' must be a boolean")
    if not isinstance(config.expose_instructions, bool):
        raise ValueError(f"MCP server {config.name!r}: 'expose_instructions' must be a boolean")
    cap = config.max_tool_result_tokens
    if not (cap is None or (type(cap) is int and (cap == 0 or cap >= 100))):
        raise ValueError(
            f"MCP server {config.name!r}: 'max_tool_result_tokens' must be null, 0, or an integer of at least 100"
        )
    if config.allowed_tools is not None and (
        not isinstance(config.allowed_tools, list) or not all(isinstance(name, str) for name in config.allowed_tools)
    ):
        raise ValueError(f"MCP server {config.name!r}: 'allowed_tools' must be a list of strings or null")
    if not isinstance(config.always_load, list) or not all(isinstance(name, str) for name in config.always_load):
        raise ValueError(f"MCP server {config.name!r}: 'always_load' must be a list of strings")
    if policy_errors := validate_mcp_tool_loading_policy(
        allowed_tools=config.allowed_tools,
        use_progressive_disclosure=config.use_progressive_disclosure,
        always_load=config.always_load,
    ):
        raise ValueError(f"MCP server {config.name!r}: {' '.join(policy_errors)}")
    prefix_error = validate_mcp_tool_name_prefix(
        config.tool_name_prefix,
        generated_suffixes=MCP_PROGRESSIVE_CONTROL_TOOL_NAMES if config.use_progressive_disclosure else (),
    )
    if prefix_error is not None:
        raise MCPToolNameValidationError(
            config.name,
            config.transport,
            violations={prefix_error},
        )
    if config.transport == "stdio":
        if not config.command:
            raise ValueError(f"MCP server {config.name!r} (stdio) requires 'command'")
    elif config.transport == "http":
        if not config.url:
            raise ValueError(f"MCP server {config.name!r} (http) requires 'url'")
    else:
        raise ValueError(f"Unknown MCP transport: {config.transport!r}")


def _build_common_kwargs(config: MCPServerConfig) -> dict[str, Any]:
    """Kwargs shared by every MCPTool subclass."""
    kwargs: dict[str, Any] = {
        "name": config.name,
        "load_prompts": config.load_prompts,
    }
    if config.description:
        kwargs["description"] = config.description
    if config.tool_name_prefix:
        kwargs["tool_name_prefix"] = config.tool_name_prefix
    if config.allowed_tools is not None:
        kwargs["allowed_tools"] = config.allowed_tools
    if config.request_timeout is not None:
        kwargs["request_timeout"] = config.request_timeout
    return kwargs


def _create_mcp_tool(config: MCPServerConfig) -> MCPTool:
    """Create the appropriate MCPTool subclass from a profile config."""
    _validate_config(config)
    common = _build_common_kwargs(config)

    if config.transport == "stdio":
        env = _STDIO_ENV_OVERRIDE.get()
        env_is_complete = _STDIO_ENV_IS_COMPLETE.get()
        if env is None:
            env = resolve_env_template_mapping(config.env, location=f"MCP server {config.name!r} env")
            env_is_complete = False
        kwargs: dict[str, Any] = {
            **common,
            "command": config.command,
            "args": config.args or None,
            "env": env if env_is_complete else env or None,
            "encoding": config.encoding,
            "env_is_complete": env_is_complete,
        }
        cwd = _STDIO_CWD_OVERRIDE.get()
        if cwd is not None:
            kwargs["cwd"] = cwd
        return _SafeStdioTool(**kwargs)

    # transport == "http" (validated above).  Static headers are attached by
    # ``_HTTPMCPTool``'s same-origin request hook rather than wired through the
    # framework's ``header_provider`` — header_provider only fires during
    # ``call_tool``, which leaves ``initialize`` / ``list_tools``
    # unauthenticated.
    headers = _HTTP_HEADERS_OVERRIDE.get()
    if headers is None:
        headers = (
            resolve_env_template_mapping(config.headers, location=f"MCP server {config.name!r} headers")
            if config.resolve_header_templates
            else dict(config.headers)
        )
    return _HTTPMCPTool(
        **common,
        url=config.url,
        terminate_on_close=config.terminate_on_close,
        verify_ssl=config.verify_ssl,
        bypass_proxy=config.bypass_proxy,
        headers=headers or None,
    )


def _create_tool_with_timeout_floor(
    config: MCPServerConfig,
    default_timeout: int,
    *,
    resolved_stdio_env: dict[str, str] | None = None,
    stdio_cwd: str | None = None,
    resolved_http_headers: dict[str, str] | None = None,
) -> Any:
    """Build the MCP tool, injecting the timeout floor when none is configured.

    We bound a hung connect/initialize by setting the MCP tool's
    ``request_timeout`` (used as ``ClientSession.read_timeout_seconds``)
    — the SDK's own ``anyio.fail_after`` then fires from inside the
    framework's lifecycle-owner task, where ``_safe_close_exit_stack``
    runs cleanup safely.  A user-configured ``request_timeout`` always
    wins; we only inject when ``config.request_timeout is None``.

    Why not ``asyncio.wait_for(__aenter__, ...)`` from outside?  The
    lifecycle owner runs in a separate task started by
    ``MCPTool._run_lifecycle_owner`` and uses anyio cancel scopes
    which are pinned to the owning task.  Cross-task cancellation
    raises ``RuntimeError("Attempted to exit cancel scope in a
    different task than it was entered in")`` and leaves the inner
    coroutine still running — defeating both the timeout and cleanup.
    """
    effective_config = (
        dataclasses.replace(config, request_timeout=default_timeout) if config.request_timeout is None else config
    )
    env_token = _STDIO_ENV_OVERRIDE.set(resolved_stdio_env)
    cwd_token = _STDIO_CWD_OVERRIDE.set(stdio_cwd)
    headers_token = _HTTP_HEADERS_OVERRIDE.set(resolved_http_headers)
    complete_token = _STDIO_ENV_IS_COMPLETE.set(resolved_stdio_env is not None)
    try:
        return _create_mcp_tool(effective_config)
    finally:
        _STDIO_ENV_IS_COMPLETE.reset(complete_token)
        _HTTP_HEADERS_OVERRIDE.reset(headers_token)
        _STDIO_CWD_OVERRIDE.reset(cwd_token)
        _STDIO_ENV_OVERRIDE.reset(env_token)


def _banner_lines_from(tool: Any) -> list[str]:
    """Extract banner diagnostics from a stdio tool, empty list otherwise."""
    return list(getattr(tool, "dropped_banner_lines", []) or [])


def _stdio_failure_diagnostics_from(tool: Any) -> _StdioFailureDiagnostics:
    """Snapshot bounded process diagnostics from Chrys's stdio tool."""
    if not isinstance(tool, _SafeStdioTool):
        return _StdioFailureDiagnostics()
    diagnostics = tool._process_diagnostics
    return _StdioFailureDiagnostics(
        stderr_tail=diagnostics.stderr_tail,
        stderr_dropped_bytes=diagnostics.stderr_dropped_bytes,
        process_exit_code=diagnostics.exit_code,
        resolved_executable=diagnostics.resolved_executable,
        effective_cwd=diagnostics.effective_cwd,
    )


def _is_timeout_cause(exc: BaseException) -> bool:
    """True if the exception chain contains an MCP/anyio read-timeout signal.

    The MCP SDK's ``ClientSession.send_request`` translates an
    ``anyio.fail_after`` timeout into ``McpError(ErrorData(code=408, ...))``.
    For other paths (e.g. cancelled transport) we also accept plain
    ``TimeoutError`` / ``asyncio.TimeoutError``.
    """
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, (TimeoutError, asyncio.TimeoutError)):
            return True
        data = getattr(cur, "error", None)
        if getattr(data, "code", None) == 408:
            return True
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return False


class _ProgressiveMCPExposure:
    """Per-adapter progressive view over one shared MCP connection catalog.

    The cached :class:`MCPTool` remains the source of truth for the complete
    allowed catalog.  This wrapper owns only stable control tools and initial
    ``always_load`` clones.  Loaded state is derived from the invocation's live
    tool list, so one adapter/run cannot leak exposure state into another
    consumer that shares the same cached MCP session.
    """

    def __init__(
        self,
        config: MCPServerConfig,
        mcp_tool: MCPTool,
        validate_namespace: Callable[[MCPServerConfig, list[FunctionTool], set[str]], None] | None = None,
        *,
        result_cap: int | None = None,
        spill_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._mcp_tool = mcp_tool
        self._validate_namespace = validate_namespace
        self._result_cap = result_cap
        self._spill_dir = spill_dir
        self._always_load_names = set(config.always_load)
        self._control_prefix = self._build_control_prefix(config)
        self._control_tools = self._create_control_tools()
        self._control_names = {tool.name for tool in self._control_tools}

    @staticmethod
    def _build_control_prefix(config: MCPServerConfig) -> str:
        """Return a stable, provider-safe namespace for progressive controls."""
        if config.tool_name_prefix:
            return config.tool_name_prefix
        # Encode every non-ASCII-alphanumeric character instead of lossy MCP
        # normalization. Distinct raw server names such as ``a/b`` and ``a-b``
        # must not collapse onto the same control-tool namespace.
        server_name = "".join(char if char.isascii() and char.isalnum() else f"-{ord(char):x}-" for char in config.name)
        prefix = f"mcp_{server_name}"
        if len(prefix) <= _MAX_PROGRESSIVE_CONTROL_PREFIX_LENGTH:
            return prefix

        # The longest generated suffix is ``list_mcp_tools``. Keep room for
        # it while retaining a stable digest of the complete raw server name,
        # so long names with the same visible stem cannot share a namespace.
        digest = hashlib.sha256(config.name.encode("utf-8")).hexdigest()[:_CONTROL_PREFIX_HASH_LENGTH]
        stem_length = _MAX_PROGRESSIVE_CONTROL_PREFIX_LENGTH - len(digest) - 1
        stem = prefix[:stem_length].rstrip("_-")
        return f"{stem}-{digest}"

    @property
    def initial_tools(self) -> list[FunctionTool]:
        """Return controls plus fresh clones of configured always-loaded tools."""
        tools = list(self._control_tools)
        tools.extend(self.clone_source(source) for source in self.always_load_sources)
        return tools

    def clone_source(self, source: FunctionTool) -> FunctionTool:
        """Clone a catalog tool with this acquisition's result policy."""
        return clone_mcp_function_tool(source, result_cap=self._result_cap, spill_dir=self._spill_dir)

    @property
    def current_surface_tools(self) -> list[FunctionTool]:
        """Return the complete required surface for a fresh progressive run."""
        return self.initial_tools

    @property
    def control_names(self) -> set[str]:
        """Return generated control names for namespace validation."""
        return set(self._control_names)

    @property
    def always_load_sources(self) -> list[FunctionTool]:
        """Return current catalog sources selected by ``always_load``."""
        return [
            source
            for source in self._catalog()
            if self._matches_configured_name(source, self._always_load_names) and source.name not in self._control_names
        ]

    @property
    def unmatched_always_load_names(self) -> list[str]:
        """Return configured initial names not present in the current catalog."""
        matched = {
            configured_name
            for configured_name in self._always_load_names
            if any(self._matches_configured_name(source, {configured_name}) for source in self._catalog())
        }
        return [name for name in self._config.always_load if name not in matched]

    def _catalog(self) -> list[FunctionTool]:
        """Read the current allowed catalog, including list-changed additions."""
        catalog = [tool for tool in self._mcp_tool.functions if isinstance(tool, FunctionTool)]
        if self._validate_namespace is not None:
            self._validate_namespace(self._config, catalog, self._control_names)
        return catalog

    @property
    def catalog_tool_names(self) -> list[str]:
        """Return the complete allowed remote catalog for diagnostics."""
        return [tool.name for tool in self._catalog()]

    @staticmethod
    def _matches_configured_name(function: FunctionTool, names: set[str]) -> bool:
        """Match config names with the MCP raw/lossless-local safety rules."""
        if not names:
            return False
        additional = function.additional_properties or {}
        normalized_name = additional.get(_MCP_NORMALIZED_NAME_KEY)
        remote_name = additional.get(_MCP_REMOTE_NAME_KEY)
        if not isinstance(normalized_name, str) or not isinstance(remote_name, str):
            return False
        candidates = _mcp_config_candidate_names(
            local_name=function.name,
            normalized_name=normalized_name,
            remote_name=remote_name,
        )
        return any(name in names for name in candidates)

    def _resolve(self, requested_name: str) -> FunctionTool:
        """Resolve one listed, allowed remote tool without unsafe alias matching."""
        catalog = [tool for tool in self._catalog() if tool.name not in self._control_names]
        matches = [tool for tool in catalog if self._matches_configured_name(tool, {requested_name})]
        matches.extend(tool for tool in catalog if tool.name == requested_name and tool not in matches)
        if not matches:
            available = ", ".join(tool.name for tool in catalog) or "none"
            raise ToolExecutionException(
                f"MCP tool '{requested_name}' is not available from server "
                f"'{self._config.name}'. Available tools: {available}."
            )
        if len(matches) > 1:
            raise ToolExecutionException(
                f"MCP tool name '{requested_name}' is ambiguous on server '{self._config.name}'."
            )
        return matches[0]

    def resume_source(self, local_name: str) -> FunctionTool | None:
        """Return a hidden catalog source for one persisted local call name."""
        match = next(
            (
                source
                for source in self._catalog()
                if source.name == local_name
                and source.name not in self._control_names
                and not self._matches_configured_name(source, self._always_load_names)
            ),
            None,
        )
        return match

    @staticmethod
    def _requested_names(tool: str | list[str]) -> list[str]:
        """Normalize a control request into a validated list of names."""
        if isinstance(tool, str):
            return [tool]
        if not isinstance(tool, list) or not all(isinstance(name, str) for name in tool):
            raise ToolExecutionException("MCP tool request must be a string or a list of strings.")
        return tool

    def _is_owned_catalog_tool(self, tool: FunctionTool) -> bool:
        """Return whether a live function belongs to this server's remote catalog."""
        provenance = get_tool_context(tool)
        return bool(
            provenance
            and provenance.get("server_name") == self._config.name
            and isinstance(provenance.get("remote_name"), str)
        )

    def _same_catalog_identity(self, left: FunctionTool, right: FunctionTool) -> bool:
        """Compare MCP ownership without relying on clone identity or local name alone."""
        left_context = get_tool_context(left)
        right_context = get_tool_context(right)
        return bool(
            left_context
            and right_context
            and left_context.get("server_name") == right_context.get("server_name") == self._config.name
            and isinstance(left_context.get("remote_name"), str)
            and left_context.get("remote_name") == right_context.get("remote_name")
        )

    def _same_surface_identity(self, left: FunctionTool, right: FunctionTool) -> bool:
        """Compare either a catalog clone or one stable progressive control."""
        if left is right or self._same_catalog_identity(left, right):
            return True
        left_context = get_tool_context(left)
        right_context = get_tool_context(right)
        return bool(
            left_context
            and right_context
            and left_context.get("server_name") == right_context.get("server_name") == self._config.name
            and isinstance(left_context.get("progressive_control"), str)
            and left_context.get("progressive_control") == right_context.get("progressive_control")
        )

    def _live_tool_names(self, ctx: FunctionInvocationContext) -> set[str]:
        """Return live catalog names owned by this server, excluding foreign collisions."""
        return {tool.name for tool in self._live_catalog_tools(ctx)}

    def _live_catalog_tools(self, ctx: FunctionInvocationContext) -> list[FunctionTool]:
        """Return exact live catalog instances owned by this server."""
        if ctx.tools is None:
            return []
        return [tool for tool in ctx.tools if isinstance(tool, FunctionTool) and self._is_owned_catalog_tool(tool)]

    def _create_control_tools(self) -> list[FunctionTool]:
        list_name = _build_prefixed_mcp_name(_MCP_PROGRESSIVE_LIST_TOOL_NAME, self._control_prefix)
        load_name = _build_prefixed_mcp_name(_MCP_PROGRESSIVE_LOAD_TOOL_NAME, self._control_prefix)
        unload_name = _build_prefixed_mcp_name(_MCP_PROGRESSIVE_UNLOAD_TOOL_NAME, self._control_prefix)

        async def _list(ctx: FunctionInvocationContext) -> list[dict[str, Any]]:
            loaded_names = self._live_tool_names(ctx)
            result: list[dict[str, Any]] = []
            for source in self._catalog():
                if source.name in {list_name, load_name, unload_name}:
                    continue
                additional = source.additional_properties or {}
                remote_name = additional.get(_MCP_REMOTE_NAME_KEY)
                result.append(
                    {
                        "name": source.name,
                        "remote_name": remote_name if isinstance(remote_name, str) else source.name,
                        "description": source.description,
                        "loaded": source.name in loaded_names,
                        "always_loaded": self._matches_configured_name(source, self._always_load_names),
                    }
                )
            return result

        async def _load(ctx: FunctionInvocationContext, tool: str | list[str]) -> str:
            if ctx.tools is None:
                raise ToolExecutionException("MCP tools can only be loaded inside an active agent tool loop.")
            messages: list[str] = []
            loaded_names = self._live_tool_names(ctx)
            pending: list[FunctionTool] = []
            pending_names: set[str] = set()
            for requested_name in self._requested_names(tool):
                try:
                    source = self._resolve(requested_name)
                except ToolExecutionException as exc:
                    messages.append(f"Error: {exc}")
                    continue
                if source.name in loaded_names:
                    messages.append(f"MCP tool '{source.name}' is already available.")
                    continue
                if source.name in pending_names:
                    messages.append(f"MCP tool '{source.name}' is already queued to load.")
                    continue
                pending.append(self.clone_source(source))
                pending_names.add(source.name)
                messages.append(f"Loaded MCP tool '{source.name}'. It is available on the next model iteration.")
            if pending:
                try:
                    ctx.add_tools(pending)
                except ValueError as exc:
                    raise ToolExecutionException(str(exc), inner_exception=exc) from exc
            return "\n".join(messages) if messages else "No MCP tools requested."

        async def _unload(ctx: FunctionInvocationContext, tool: str | list[str]) -> str:
            if ctx.tools is None:
                raise ToolExecutionException("MCP tools can only be unloaded inside an active agent tool loop.")
            messages: list[str] = []
            live_tools_by_name: dict[str, list[FunctionTool]] = {}
            for live_tool in self._live_catalog_tools(ctx):
                live_tools_by_name.setdefault(live_tool.name, []).append(live_tool)
            queued_names: set[str] = set()
            tools_to_remove: list[FunctionTool] = []
            for requested_name in self._requested_names(tool):
                if requested_name in {list_name, load_name, unload_name}:
                    messages.append(f"MCP control tool '{requested_name}' cannot be unloaded.")
                    continue
                try:
                    source = self._resolve(requested_name)
                except ToolExecutionException as exc:
                    messages.append(f"Error: {exc}")
                    continue
                if self._matches_configured_name(source, self._always_load_names):
                    messages.append(f"MCP tool '{source.name}' is configured in always_load and cannot be unloaded.")
                    continue
                if source.name not in live_tools_by_name:
                    messages.append(f"MCP tool '{source.name}' is not currently loaded.")
                    continue
                if source.name in queued_names:
                    messages.append(f"MCP tool '{source.name}' is already queued to unload.")
                    continue
                queued_names.add(source.name)
                tools_to_remove.extend(live_tools_by_name[source.name])
                messages.append(f"Unloaded MCP tool '{source.name}'. It will be removed on the next model iteration.")
            if tools_to_remove:
                ctx.remove_tools(tools_to_remove)
            return "\n".join(messages) if messages else "No MCP tools requested."

        input_model = {
            "type": "object",
            "properties": {
                "tool": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                    "description": "The MCP tool name, or MCP tool names, to update.",
                }
            },
            "required": ["tool"],
        }
        tools = [
            FunctionTool(
                func=_list,
                name=list_name,
                description=f"List MCP tools available from server '{self._config.name}'.",
                input_model={"type": "object", "properties": {}},
                additional_properties={_MCP_PROGRESSIVE_CONTROL_MARKER: True},
            ),
            FunctionTool(
                func=_load,
                name=load_name,
                description=f"Load MCP tools from server '{self._config.name}' for the current run.",
                input_model=input_model,
                additional_properties={_MCP_PROGRESSIVE_CONTROL_MARKER: True},
            ),
            FunctionTool(
                func=_unload,
                name=unload_name,
                description=f"Unload MCP tools from server '{self._config.name}' for the current run.",
                input_model=input_model,
                additional_properties={_MCP_PROGRESSIVE_CONTROL_MARKER: True},
            ),
        ]
        for tool in tools:
            set_tool_kind(tool, KIND_MCP)
            set_tool_context(tool, {"server_name": self._config.name, "progressive_control": tool.name})
        return tools


# Resume re-exposes tools only for calls still lacking any answer in their
# exchange. Informational calls never need resuming. Id-less calls (None or
# empty ids; non-string ids read through their string form) can never be
# answered by any result, so they always surface as unresolved.
_RESUME_PAIRING_POLICY = PairingPolicy(
    call_types=frozenset({"function_call"}),
    include_informational_calls=False,
    result_types=frozenset({"function_result"}),
    none_id=NoneIdPolicy.UNPAIRABLE_OCCURRENCE,
    empty_id=EmptyIdPolicy.UNPAIRABLE_OCCURRENCE,
    malformed_id="stringify",
)


class ProgressiveMCPResumeProvider(ContextProvider):
    """Refresh per-run progressive tools and resume unresolved current-turn calls.

    Ordinary load/unload state remains run-local. Before each run, newly
    advertised ``always_load`` tools are added to the invocation. The narrower
    retry/restore safety case also re-exposes exact hidden functions for
    unresolved actionable calls in the current turn.
    """

    def __init__(self, adapter: MCPAdapter) -> None:
        super().__init__("progressive_mcp_resume")
        self._adapter = adapter

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        default_options = agent.default_options or {}
        visible_tools = normalize_tools(default_options.get("tools"))
        visible_tools.extend(normalize_tools(context.tools))
        candidates = self._adapter._progressive_surface_tools()

        messages = context.get_messages(include_input=True)
        current_messages = messages[current_turn_start(messages) :]
        slices = turn_slices(current_messages)
        if slices:
            current_messages = current_messages[slices[-1][0] : slices[-1][1]]
        unresolved_calls = self._unresolved_actionable_calls(current_messages)
        if unresolved_calls:
            candidates.extend(self._adapter._resume_tools(unresolved_calls))

        tools = self._adapter._context_tools_without_collisions(visible_tools, candidates)
        if tools:
            context.extend_tools(self.source_id, tools)

    @staticmethod
    def _unresolved_actionable_calls(messages: list[Message]) -> list[tuple[str, str | None]]:
        """Return unpaired call identity, preserving stamped MCP server provenance."""
        accessor = LiveAccessor()
        unresolved: list[tuple[str, str | None]] = []
        for exchange in iter_exchanges(messages, accessor):
            pairing = pair_results(messages, exchange, accessor, _RESUME_PAIRING_POLICY)
            dangling = [
                call
                for assignments in pairing.truthy_assignments.values()
                for call, result in assignments
                if result is None
            ]
            dangling.extend(pairing.unpairable_calls)
            dangling.sort(key=lambda occurrence: (occurrence.message_index, occurrence.content_index))
            for occurrence in dangling:
                call = messages[occurrence.message_index].contents[occurrence.content_index]
                if isinstance(call.name, str) and call.name:
                    context_value = call.additional_properties.get(TOOL_CALL_CONTEXT_METADATA_KEY)
                    server_name = context_value.get("server_name") if isinstance(context_value, dict) else None
                    unresolved.append((call.name, server_name if isinstance(server_name, str) else None))
        return list(dict.fromkeys(unresolved))


class MCPAdapter:
    """Manages MCP server connections and exposes their tools.

    Supports stdio and HTTP (Streamable HTTP/SSE) transports as configured
    in the agent profile YAML.

    An ordinary per-server connection failure does not abort the agent build.
    Deterministic tool-name collisions do abort it before any ambiguous tool
    surface can reach the model. Ordinary failures are:

    * published on the optional ``EventBus`` as ``Warning`` events
      (``code="mcp.connect_failed"``) so the TUI can display them, and
    * retained on :attr:`failures` for inspection by diagnostic UIs.
    """

    def __init__(
        self,
        bus: EventBus | None = None,
        session_id: str | None = None,
        *,
        cache: MCPConnectionCache | None = None,
        stdio_cwd: str | None = None,
        session_dir: Path | None = None,
        reserved_tool_names: Collection[str] = (),
    ) -> None:
        self._cache = cache or MCPConnectionCache()
        self._owns_cache = cache is None
        self._leases: dict[str, MCPConnectionLease] = {}
        # Compatibility/debug view of this adapter's acquired underlying
        # MCPTool objects.  Ownership still flows through ``_leases``.
        self._servers: dict[str, MCPTool] = {}
        self._functions_by_server: dict[str, list[FunctionTool]] = {}
        # Per-server ``expose_instructions`` config, captured at registration.
        # Adapter-level (not on the shared MCPTool): profiles sharing a cached
        # connection may disagree on exposure.
        self._instructions_exposure: dict[str, bool] = {}
        self._progressive_exposures: dict[str, _ProgressiveMCPExposure] = {}
        self._tool_namespace_by_server: dict[str, set[str]] = {}
        self._connecting: dict[str, asyncio.Task[list[Any]]] = {}
        self._failures: dict[str, MCPConnectionError] = {}
        self._tool_names_by_server: dict[str, list[str]] = {}
        self._bus = bus
        self._session_id = session_id
        self._stdio_cwd = stdio_cwd
        self._session_dir = session_dir
        self._reserved_tool_names = frozenset(reserved_tool_names)
        # Serializes the registration critical section shared by ``connect``
        # and ``disconnect``/``disconnect_all``.  Without this, a ``disconnect_all``
        # that snapshots ``_servers`` while a concurrent ``connect`` is mid-
        # ``__aenter__`` would miss the about-to-be-registered server, leaving
        # an orphaned subprocess.  Tool ``__aexit__`` calls are intentionally
        # performed outside the lock so parallel disconnect is preserved.
        self._lock = asyncio.Lock()

    def set_reserved_tool_names(self, reserved_tool_names: Collection[str]) -> None:
        """Replace names reserved from MCP namespace validation.

        This is primarily used by configuration diagnostics whose surrounding
        agent draft can change while the test panel remains mounted. Active
        production adapters keep a stable namespace for their acquired leases.
        """
        if self._leases or self._connecting:
            raise RuntimeError("Cannot replace reserved MCP tool names while connections are active.")
        self._reserved_tool_names = frozenset(reserved_tool_names)

    @staticmethod
    def _create_tool_with_timeout_floor(config: MCPServerConfig, default_timeout: int) -> Any:
        """Build the MCP tool, injecting the timeout floor when none is configured."""
        return _create_tool_with_timeout_floor(config, default_timeout)

    @staticmethod
    def _banner_lines_from(tool: Any) -> list[str]:
        """Extract banner diagnostics from a stdio tool, empty list otherwise."""
        return _banner_lines_from(tool)

    @staticmethod
    def _stdio_failure_diagnostics_from(tool: Any) -> _StdioFailureDiagnostics:
        """Snapshot bounded stdio subprocess diagnostics, empty otherwise."""
        return _stdio_failure_diagnostics_from(tool)

    @staticmethod
    def _is_timeout_cause(exc: BaseException) -> bool:
        """True if the exception chain contains an MCP/anyio read-timeout signal."""
        return _is_timeout_cause(exc)

    def create_resume_provider(self) -> ProgressiveMCPResumeProvider | None:
        """Return the per-run refresh/resume provider when progressive servers exist."""
        if not self._progressive_exposures:
            return None
        return ProgressiveMCPResumeProvider(self)

    def get_server_instructions_map(self) -> dict[str, str]:
        """Return {server_name: instructions} for connected servers with instructions.

        Servers whose config disabled ``expose_instructions`` are excluded.
        """
        result: dict[str, str] = {}
        for name, server in self._servers.items():
            if not self._instructions_exposure.get(name, True):
                continue
            instructions = getattr(server, "_server_instructions", None)
            if instructions:
                result[name] = instructions
        for name, lease in self._leases.items():
            if name not in result and self._instructions_exposure.get(name, True):
                instructions = getattr(lease.mcp_tool, "_server_instructions", None)
                if instructions:
                    result[name] = instructions
        return result

    def render_instructions_reminder(self) -> str | None:
        """Return an ``<mcp_instructions>`` block with all server instructions, or None.

        Servers render in name order: ``connect_all`` registers concurrently,
        so insertion order varies run to run and would leak that
        nondeterminism into the prompt text.  Per-server text is capped at
        ``MCP_INSTRUCTIONS_CHAR_LIMIT``: instructions are server-controlled
        and injected into every LLM call, so an unbounded value from one
        misbehaving server could push every request past the context limit.
        """
        instructions_map = self.get_server_instructions_map()
        if not instructions_map:
            return None
        lines = ["<mcp_instructions>"]
        for name in sorted(instructions_map):
            escaped_name = xml_escape(name, quote=True)
            lines.append(f'  <server name="{escaped_name}">')
            budget = MCP_INSTRUCTIONS_CHAR_LIMIT
            truncated = False
            for line in instructions_map[name].strip().splitlines():
                rendered = f"    {xml_escape(line)}".rstrip()
                # +1 charges the joining newline so a blank-line flood
                # (zero-length rendered lines) cannot bypass the budget.
                cost = len(rendered) + 1
                if cost > budget:
                    if budget > 1:
                        lines.append(_SEVERED_ENTITY_TAIL_RE.sub("", rendered[: budget - 1]))
                    truncated = True
                    break
                lines.append(rendered)
                budget -= cost
            if truncated:
                lines.append(f"    [instructions truncated: exceeded {MCP_INSTRUCTIONS_CHAR_LIMIT} characters]")
            lines.append("  </server>")
        lines.append("</mcp_instructions>")
        return "\n".join(lines)

    def _progressive_surface_tools(self) -> list[FunctionTool]:
        """Return current controls and always-loaded tools for a fresh run."""
        return [tool for exposure in self._progressive_exposures.values() for tool in exposure.current_surface_tools]

    def _validate_server_namespace(
        self,
        config: MCPServerConfig,
        catalog: list[FunctionTool],
        control_names: set[str],
    ) -> None:
        """Reject invalid or colliding names before tools reach the model."""
        violations: list[str] = []
        for tool in catalog:
            error = validate_mcp_exposed_tool_name(tool.name)
            if error is None:
                continue
            additional = tool.additional_properties or {}
            remote_name = additional.get(_MCP_REMOTE_NAME_KEY)
            origin = f" Remote catalog name: {remote_name!r}." if isinstance(remote_name, str) else ""
            violations.append(f"{error}{origin}")
        violations.extend(
            f"{error} Generated progressive-disclosure control."
            for control_name in sorted(control_names)
            if (error := validate_mcp_exposed_tool_name(control_name)) is not None
        )
        if violations:
            raise MCPToolNameValidationError(
                config.name,
                config.transport,
                violations=violations,
            )

        catalog_names = [tool.name for tool in catalog]
        duplicate_catalog_names = {name for name, count in Counter(catalog_names).items() if count > 1}
        if duplicate_catalog_names:
            raise MCPToolNameCollisionError(
                config.name,
                config.transport,
                conflicting_names=duplicate_catalog_names,
                conflict_with=f"another permitted tool from MCP server {config.name!r}",
                guidance=(
                    "Rename one of the remote entries, exclude the shared name from the Permitted Tool Set, "
                    "or disable 'Expose server prompts' when the duplicate is a prompt."
                ),
            )

        catalog_name_set = set(catalog_names)
        if control_collisions := catalog_name_set & control_names:
            raise MCPToolNameCollisionError(
                config.name,
                config.transport,
                conflicting_names=control_collisions,
                conflict_with="a generated progressive-disclosure control",
                guidance=(
                    "Exclude the remote tool, choose a non-conflicting Tool Name Prefix, "
                    "or disable progressive loading."
                ),
            )

        namespace = catalog_name_set | control_names
        if chrys_collisions := namespace & self._reserved_tool_names:
            raise MCPToolNameCollisionError(
                config.name,
                config.transport,
                conflicting_names=chrys_collisions,
                conflict_with="a Chrys tool",
                guidance="Exclude the remote tool from the Permitted Tool Set or configure a Tool Name Prefix.",
            )

        for other_server, other_namespace in self._tool_namespace_by_server.items():
            if other_server == config.name:
                continue
            if cross_server_collisions := namespace & other_namespace:
                raise MCPToolNameCollisionError(
                    config.name,
                    config.transport,
                    conflicting_names=cross_server_collisions,
                    conflict_with=f"MCP server {other_server!r}",
                    guidance="Configure distinct Tool Name Prefix values or exclude the duplicate tool.",
                )

        if config.name in self._leases:
            self._tool_namespace_by_server[config.name] = namespace

    def _current_functions_for_server(
        self,
        name: str,
        lease: MCPConnectionLease,
    ) -> list[FunctionTool]:
        """Return a fresh progressive surface or the ordinary cached functions."""
        exposure = self._progressive_exposures.get(name)
        if exposure is not None:
            return exposure.initial_tools
        return list(self._functions_by_server.get(name, lease.functions))

    def _context_tools_without_collisions(
        self,
        visible_tools: list[Any],
        candidates: list[FunctionTool],
    ) -> list[FunctionTool]:
        """Return missing candidates, rejecting foreign same-name tools loudly."""
        visible_by_name: dict[str, list[Any]] = {}
        for tool in visible_tools:
            if isinstance(tool, FunctionTool):
                visible_by_name.setdefault(tool.name, []).append(tool)
                continue
            if isinstance(tool, Mapping):
                function = tool.get("function")
                if isinstance(function, Mapping) and isinstance(function.get("name"), str):
                    visible_by_name.setdefault(function["name"], []).append(tool)
        additions: list[FunctionTool] = []
        for candidate in candidates:
            existing = visible_by_name.get(candidate.name, [])
            if not existing:
                additions.append(candidate)
                visible_by_name[candidate.name] = [candidate]
                continue

            provenance = get_tool_context(candidate)
            server_name = provenance.get("server_name") if provenance is not None else None
            exposure = self._progressive_exposures.get(server_name) if isinstance(server_name, str) else None
            if (
                len(existing) == 1
                and exposure is not None
                and isinstance(existing[0], FunctionTool)
                and exposure._same_surface_identity(existing[0], candidate)
            ):
                continue
            config = exposure._config if exposure is not None else None
            raise MCPToolNameCollisionError(
                config.name if config is not None else server_name or "unknown",
                config.transport if config is not None else "unknown",
                conflicting_names={candidate.name},
                conflict_with="another visible tool contributed to this run",
                guidance="Configure a distinct Tool Name Prefix or remove the conflicting tool.",
            )
        return additions

    def _resume_tools(self, calls: list[tuple[str, str | None]]) -> list[FunctionTool]:
        """Clone uniquely resolved hidden MCP tools referenced by pending calls."""
        tools: list[FunctionTool] = []
        added_names: set[str] = set()
        for name, server_name in calls:
            if name in added_names:
                continue
            if server_name is not None:
                exposure = self._progressive_exposures.get(server_name)
                source = exposure.resume_source(name) if exposure is not None else None
                matches = [source] if source is not None else []
            else:
                matches = [
                    source
                    for exposure in self._progressive_exposures.values()
                    if (source := exposure.resume_source(name)) is not None
                ]
            if len(matches) == 1:
                match_context = get_tool_context(matches[0])
                matched_server = (
                    server_name
                    if server_name is not None
                    else match_context.get("server_name")
                    if match_context is not None
                    else None
                )
                if (
                    isinstance(matched_server, str)
                    and (exposure := self._progressive_exposures.get(matched_server)) is not None
                ):
                    tools.append(exposure.clone_source(matches[0]))
                else:
                    tools.append(clone_mcp_function_tool(matches[0]))
                added_names.add(name)
            elif len(matches) > 1:
                logger.warning(
                    "Cannot resume ambiguous progressive MCP tool %r; configure distinct tool_name_prefix values.",
                    name,
                )
        return tools

    async def connect(self, config: MCPServerConfig) -> list[Any]:
        """Connect to an MCP server and return its tools as FunctionTools.

        Idempotent: calling with a previously-connected name returns the
        cached tool list.  Bounded by a hard timeout floor
        (``DEFAULT_CONNECT_TIMEOUT_SECONDS``) injected into the SDK's
        ``request_timeout`` when the user didn't configure one — so a
        hung server cannot block the agent build.

        Raises:
            MCPConnectionError: If the server fails to connect, initialize,
                or exceeds the connect timeout.  ``asyncio.CancelledError``
                is propagated unchanged after the partially-opened transport
                is closed.
        """
        # Validate before cache lookup as well as connection creation: a cache
        # hit must not let malformed wrapper-only progressive fields bypass the
        # checks performed by ``_create_mcp_tool`` on a cold connection.
        _validate_config(config)
        async with self._lock:
            lease = self._leases.get(config.name)
            if lease is not None:
                return self._current_functions_for_server(config.name, lease)
            task = self._connecting.get(config.name)
            if task is None:
                task = asyncio.create_task(self._acquire_new(config))
                self._connecting[config.name] = task

                def _discard_finished(done: asyncio.Task[list[Any]], name: str = config.name) -> None:
                    if self._connecting.get(name) is done:
                        self._connecting.pop(name, None)

                task.add_done_callback(_discard_finished)

        try:
            return await task
        finally:
            if task.done():
                async with self._lock:
                    if self._connecting.get(config.name) is task:
                        self._connecting.pop(config.name, None)

    async def _connect_new(self, config: MCPServerConfig) -> list[Any]:
        """Open or lease a new MCP server connection and register it on success."""
        current_task = asyncio.current_task()
        inserted = False
        if current_task is not None:
            async with self._lock:
                if config.name not in self._connecting:
                    self._connecting[config.name] = current_task
                    inserted = True
        try:
            return await self._acquire_new(config)
        finally:
            if inserted:
                async with self._lock:
                    if self._connecting.get(config.name) is current_task:
                        self._connecting.pop(config.name, None)

    async def _acquire_new(self, config: MCPServerConfig) -> list[Any]:
        """Acquire a cache lease and register it on this adapter."""
        lease: MCPConnectionLease | None = None
        duplicate_tools: list[Any] | None = None
        release_duplicate = False
        prune_private_cache = False
        registered = False
        exposure: _ProgressiveMCPExposure | None = None
        delivered_tools: list[FunctionTool]
        catalog: list[FunctionTool]
        control_names: set[str] = set()
        namespace: set[str]

        try:
            lease = await self._cache.acquire(config, stdio_cwd=self._stdio_cwd)
        except MCPConnectionError as err:
            self._failures[config.name] = err
            raise
        assert lease is not None

        try:
            catalog = [tool for tool in lease.mcp_tool.functions if isinstance(tool, FunctionTool)]
            result_cap = resolve_mcp_result_cap(config.max_tool_result_tokens)
            if config.use_progressive_disclosure:
                exposure = _ProgressiveMCPExposure(
                    config,
                    lease.mcp_tool,
                    self._validate_server_namespace,
                    result_cap=result_cap,
                    spill_dir=self._session_dir,
                )
                control_names = exposure.control_names
                delivered_tools = exposure.initial_tools
            else:
                self._validate_server_namespace(config, catalog, control_names)
                delivered_tools = [
                    clone_mcp_function_tool(source, result_cap=result_cap, spill_dir=self._session_dir)
                    for source in catalog
                ]
            namespace = {tool.name for tool in catalog} | control_names
        except Exception:
            await self._release_unregistered_lease(config.name, lease)
            raise

        try:
            async with self._lock:
                existing = self._leases.get(config.name)
                if existing is not None:
                    duplicate_tools = self._current_functions_for_server(config.name, existing)
                    release_duplicate = True
                elif (existing_server := self._servers.get(config.name)) is not None:
                    # ``existing_server.functions`` are the engine-domain
                    # framework-native instances — clone-deliver like every
                    # other outward path (N5 invariant: tools leaving the MCP
                    # domain are chrys FunctionTool clones).
                    result_cap = resolve_mcp_result_cap(config.max_tool_result_tokens)
                    duplicate_tools = [
                        clone_mcp_function_tool(t, result_cap=result_cap, spill_dir=self._session_dir)
                        for t in existing_server.functions
                    ]
                    release_duplicate = True
                    prune_private_cache = self._owns_cache
                else:
                    current_task = asyncio.current_task()
                    if self._connecting.get(config.name) is not current_task:
                        raise asyncio.CancelledError
                    # Re-check under the registration lock so parallel server
                    # connects cannot both pass against an empty namespace.
                    self._validate_server_namespace(config, catalog, control_names)
                    self._leases[config.name] = lease
                    self._servers[config.name] = lease.mcp_tool
                    self._functions_by_server[config.name] = delivered_tools
                    self._instructions_exposure[config.name] = config.expose_instructions
                    self._tool_namespace_by_server[config.name] = namespace
                    if exposure is not None:
                        self._progressive_exposures[config.name] = exposure
                    registered = True
        except BaseException:
            if lease is not None and not registered:
                await self._release_unregistered_lease(config.name, lease)
            raise

        if release_duplicate:
            await self._release_unregistered_lease(config.name, lease)
            if prune_private_cache:
                await self._cache.prune_idle(max_idle_seconds=0)
            assert duplicate_tools is not None
            return duplicate_tools

        self._failures.pop(config.name, None)
        if exposure is not None and (missing := exposure.unmatched_always_load_names):
            missing_names = ", ".join(missing)
            await self._publish_warning(
                code="mcp.always_load_missing",
                message=(
                    f"MCP server '{config.name}' did not advertise configured initially visible "
                    f"tool(s): {missing_names}. They will be picked up automatically if advertised later."
                ),
            )
        logger.debug(
            "Connected to MCP server '%s' (%s): %d tools loaded",
            config.name,
            config.transport,
            len(lease.functions),
        )
        return list(delivered_tools)

    async def _release_unregistered_lease(self, name: str, lease: MCPConnectionLease) -> None:
        try:
            await self._finish_cleanup_on_cancel(self._cache.release(lease))
        except Exception:
            logger.exception("Error releasing duplicate MCP lease for '%s'", name)
            raise

    async def _finish_cleanup_on_cancel(self, awaitable: Awaitable[Any]) -> None:
        """Let cleanup finish before propagating caller cancellation."""
        cleanup = asyncio.ensure_future(awaitable)
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await cleanup
            raise

    async def test_connection(self, config: MCPServerConfig) -> MCPTestReport:
        """Attempt a one-shot connection test for an MCP server.

        Establishes the server connection, snapshots what it advertised
        (identity, capabilities, instructions, tool/prompt catalog) into an
        :class:`MCPTestReport`, and immediately closes it.  Bounded by
        ``DEFAULT_TEST_TIMEOUT_SECONDS`` injected into the SDK's
        ``request_timeout`` when none is configured — so the Test button
        can never hang indefinitely.

        Raises:
            MCPConnectionError: If the server fails to connect, initialize,
                or exceeds the test timeout.  Carries any captured stdout
                banner lines, bounded stderr tail, and child exit code so the
                UI can show why a misbehaving server failed.
        """
        mcp_tool: Any | None = None
        try:
            try:
                mcp_tool = self._create_tool_with_timeout_floor(config, DEFAULT_TEST_TIMEOUT_SECONDS)
                await mcp_tool.__aenter__()
                catalog = [tool for tool in mcp_tool.functions if isinstance(tool, FunctionTool)]
                initial_tool_names: tuple[str, ...] | None = None
                if config.use_progressive_disclosure:
                    exposure = _ProgressiveMCPExposure(config, mcp_tool, self._validate_server_namespace)
                    # Materializing the surface validates both catalog names
                    # and generated controls through the production path.
                    initial_tool_names = tuple(tool.name for tool in exposure.initial_tools)
                else:
                    self._validate_server_namespace(config, catalog, set())
                return self._build_test_report(mcp_tool, catalog, initial_tool_names)
            except MCPConnectionError:
                raise
            except Exception as exc:
                diagnostics = self._stdio_failure_diagnostics_from(mcp_tool)
                cause: BaseException = exc
                if self._is_timeout_cause(exc):
                    timeout = getattr(mcp_tool, "request_timeout", DEFAULT_TEST_TIMEOUT_SECONDS)
                    cause = TimeoutError(
                        f"connection did not complete within {timeout}s (server may not be speaking MCP)"
                    )
                raise MCPConnectionError(
                    config.name,
                    config.transport,
                    cause,
                    banner_lines=self._banner_lines_from(mcp_tool),
                    stderr_tail=diagnostics.stderr_tail,
                    stderr_dropped_bytes=diagnostics.stderr_dropped_bytes,
                    process_exit_code=diagnostics.process_exit_code,
                    resolved_executable=diagnostics.resolved_executable,
                    effective_cwd=diagnostics.effective_cwd,
                ) from exc
        finally:
            if mcp_tool is not None:
                with contextlib.suppress(Exception):
                    await mcp_tool.__aexit__(None, None, None)

    @staticmethod
    def _build_test_report(
        mcp_tool: Any,
        catalog: list[FunctionTool],
        initial_tool_names: tuple[str, ...] | None,
    ) -> MCPTestReport:
        """Snapshot server-advertised state before the test connection closes."""
        server_info = getattr(mcp_tool, "_server_info", None)
        prompt_remote_names = getattr(mcp_tool, "_loaded_prompt_remote_names", None) or set()
        tools: list[tuple[str, str]] = []
        prompts: list[tuple[str, str]] = []
        for tool in catalog:
            additional = tool.additional_properties or {}
            remote_name = additional.get(_MCP_REMOTE_NAME_KEY)
            entry = (tool.name, tool.description or "")
            if isinstance(remote_name, str) and remote_name in prompt_remote_names:
                prompts.append(entry)
            else:
                tools.append(entry)
        return MCPTestReport(
            server_name=getattr(server_info, "name", None),
            server_title=getattr(server_info, "title", None),
            server_version=getattr(server_info, "version", None),
            protocol_version=getattr(mcp_tool, "_protocol_version", None),
            capabilities=_flatten_server_capabilities(getattr(mcp_tool, "_server_capabilities", None)),
            instructions=getattr(mcp_tool, "_server_instructions", None),
            tools=tuple(tools),
            prompts=tuple(prompts),
            initial_tool_names=initial_tool_names,
        )

    async def connect_all(
        self,
        configs: list[MCPServerConfig],
        *,
        progress: Callable[[MCPServerConfig, str, int, int, int], Awaitable[None]] | None = None,
    ) -> list[Any]:
        """Connect to multiple enabled MCP servers and return all tools.

        Per-server failures do not abort the sequence — they are recorded on
        :attr:`failures` and published on the event bus as ``Warning`` events
        so the TUI can display them to the user.

        Connections run concurrently, so progress callbacks arrive in
        completion order rather than profile declaration order.  Callers
        should treat ``current`` / ``failed`` / ``total`` as aggregate
        counters and key any per-server UI by server name.

        Any ordinary ``Exception`` is treated as a per-server failure and
        dropped from the returned tool list.  This includes the documented
        ``MCPConnectionError`` path and also defensive handling of
        unexpected errors (e.g. ``ValueError`` from a malformed config
        caught by ``_validate_config``, framework-level errors,
        ``ExceptionGroup`` leaks from anyio task groups) — none of which
        should be allowed to abort the agent build and leave the engine
        half-initialised. ``MCPToolConfigurationError`` is the deliberate
        exception: deterministic invalid/colliding tool names must abort the
        build before a provider rejects the request or a last-wins tool map
        selects the wrong function.
        ``BaseException`` subclasses (``CancelledError``,
        ``KeyboardInterrupt``, ``SystemExit``, and bare ``BaseExceptionGroup``)
        propagate so structural cancellation / shutdown signals are never
        swallowed.
        """
        enabled_configs: list[MCPServerConfig] = []
        for config in configs:
            if not config.enabled:
                logger.debug("Skipping disabled MCP server '%s'", config.name)
                continue
            enabled_configs.append(config)
        self._tool_names_by_server = {}

        total = len(enabled_configs)
        connected = 0
        failed = 0

        async def _connect_one(config: MCPServerConfig) -> list[Any]:
            nonlocal connected, failed
            if progress is not None:
                await progress(config, "starting", connected, total, failed)
            try:
                tools = await self.connect(config)
            except asyncio.CancelledError:
                raise
            except MCPToolConfigurationError:
                raise
            except MCPConnectionError as exc:
                await self._publish_failure(exc)
                failed += 1
                if progress is not None:
                    await progress(config, "failed", connected, total, failed)
                return []
            except Exception as exc:
                err = MCPConnectionError(config.name, config.transport, exc)
                self._failures[config.name] = err
                await self._publish_failure(err)
                failed += 1
                if progress is not None:
                    await progress(config, "failed", connected, total, failed)
                return []
            connected += 1
            exposure = self._progressive_exposures.get(config.name)
            self._tool_names_by_server[config.name] = (
                exposure.catalog_tool_names
                if exposure is not None
                else [getattr(tool, "name", str(tool)) for tool in tools]
            )
            if progress is not None:
                await progress(config, "connected", connected, total, failed)
            return tools

        results = await asyncio.gather(*(_connect_one(config) for config in enabled_configs))
        self._tool_names_by_server = {
            config.name: self._tool_names_by_server[config.name]
            for config in enabled_configs
            if config.name in self._tool_names_by_server
        }
        return [tool for tools in results for tool in tools]

    async def disconnect(self, name: str) -> None:
        """Disconnect from an MCP server.

        The lock is held only for the pop so that a concurrent ``connect``
        cannot stamp the server back into ``_servers`` after we snapshotted
        its absence.  ``__aexit__`` runs outside the lock.
        """
        async with self._lock:
            task = self._connecting.pop(name, None)
            lease = self._leases.pop(name, None)
            server = self._servers.pop(name, None)
            self._functions_by_server.pop(name, None)
            self._instructions_exposure.pop(name, None)
            self._progressive_exposures.pop(name, None)
            self._tool_namespace_by_server.pop(name, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if lease is None:
            if server is not None:
                try:
                    await server.__aexit__(None, None, None)
                except Exception:
                    logger.exception("Error disconnecting MCP server '%s'", name)
            return
        try:
            await self._cache.release(lease)
            if self._owns_cache:
                await self._cache.prune_idle(max_idle_seconds=0)
        except Exception:
            logger.exception("Error disconnecting MCP server '%s'", name)

    async def disconnect_all(self) -> None:
        """Disconnect from all connected MCP servers in parallel.

        The snapshot+clear runs under ``_lock`` so connected servers are
        disconnected and in-flight connects are cancelled before they can
        register a server.  The per-tool ``__aexit__`` calls run outside
        the lock so they proceed concurrently and a slow shutdown does not
        block new connects.
        """
        async with self._lock:
            items = list(self._leases.items())
            leased_server_names = set(self._leases)
            fallback_items = [
                (name, server) for name, server in self._servers.items() if name not in leased_server_names
            ]
            connecting = list(self._connecting.values())
            self._leases.clear()
            self._connecting.clear()
            self._servers.clear()
            self._functions_by_server.clear()
            self._instructions_exposure.clear()
            self._progressive_exposures.clear()
            self._tool_namespace_by_server.clear()
        for task in connecting:
            task.cancel()
        if connecting:
            await self._finish_cleanup_on_cancel(asyncio.gather(*connecting, return_exceptions=True))

        async def _release_one(name: str, lease: MCPConnectionLease) -> None:
            try:
                await self._cache.release(lease)
            except Exception:
                logger.exception("Error disconnecting MCP server '%s'", name)

        if items:
            await self._finish_cleanup_on_cancel(
                asyncio.gather(*(_release_one(n, lease) for n, lease in items), return_exceptions=True)
            )
        if fallback_items:

            async def _exit_one(name: str, tool: MCPTool) -> None:
                try:
                    await tool.__aexit__(None, None, None)
                except Exception:
                    logger.exception("Error disconnecting MCP server '%s'", name)

            await self._finish_cleanup_on_cancel(
                asyncio.gather(*(_exit_one(n, t) for n, t in fallback_items), return_exceptions=True)
            )
        if self._owns_cache:
            try:
                await self._finish_cleanup_on_cancel(self._cache.close_all())
            finally:
                self._cache = MCPConnectionCache()

    async def _publish_warning(self, *, code: str, message: str) -> None:
        """Log and best-effort publish one user-visible non-fatal warning."""
        logger.warning("%s", message)
        if self._bus is None:
            return
        from chrys.foundation.events.types import Warning as WarningEvent

        with contextlib.suppress(Exception):
            await self._bus.publish(
                WarningEvent(
                    code=code,
                    message=message,
                    session_id=self._session_id,
                )
            )

    async def _publish_failure(self, err: MCPConnectionError) -> None:
        """Log and, if a bus is configured, publish a user-visible Warning."""
        logger.error(
            "MCP server '%s' (%s) failed to connect: %s",
            err.server_name,
            err.transport,
            err.cause,
            exc_info=err.cause,
        )
        if self._bus is None:
            return
        from chrys.foundation.events.types import Warning as WarningEvent

        with contextlib.suppress(Exception):
            await self._bus.publish(
                WarningEvent(
                    code="mcp.connect_failed",
                    message=str(err),
                    session_id=self._session_id,
                )
            )

    @property
    def server_names(self) -> list[str]:
        """Names of currently connected MCP servers."""
        return list(dict.fromkeys([*self._leases.keys(), *self._servers.keys()]))

    @property
    def tool_names_by_server(self) -> dict[str, list[str]]:
        """Snapshot of server-name -> loaded tool names from the last connect_all call."""
        return {name: list(tools) for name, tools in self._tool_names_by_server.items()}

    @property
    def failures(self) -> dict[str, MCPConnectionError]:
        """Snapshot of server-name → last connect failure."""
        return dict(self._failures)
