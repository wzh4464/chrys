# Copyright (c) 2026 Chrys. All rights reserved.

"""Chrys-owned OTel telemetry layers.

:class:`ChatTelemetryLayer` is mixed into the instrumented chat clients and
:class:`AgentTelemetryLayer` is the base of :class:`~.agent.Agent`. The
executable acceptance baseline is ``tests/kernel/test_telemetry.py``.

Deliberate divergences from the framework (everything else mirrors line by
line):

- **Gate**: both layers gate on the foundation-owned :data:`TELEMETRY_GATE`
  (default **off**, flipped by ``chrys.foundation.observability.setup.setup_otel`` via
  :func:`~chrys.foundation.observability.gate.configure_telemetry`) instead of the framework's default-on
  ``OBSERVABILITY_SETTINGS`` singleton with its sticky-disable machinery.
  No env reads here — ``CHRYS_OTEL*`` parsing belongs to chrys settings.
- **ContextVar lifecycle fix** (framework bug chrys used to carry): the
  framework sets the INNER dedup ContextVars in the *synchronous* ``run()``
  call and resets them inside the ``_run()`` coroutine — when a caller wraps
  the returned coroutine in ``asyncio.create_task`` (sub-agent controller
  non-streaming attempt), the reset fires in a different context and raises
  ``ValueError: token was created in a different Context``. Here the
  non-streaming branch sets *and* resets the vars inside ``_run()`` (one
  context by construction); the streaming branch keeps the framework shape
  (set in ``run()``, reset in the stream finalizer) but closes the span
  *before* the resets and downgrades a cross-context reset to a debug log.
- **Private reads replaced**: the stream-error probe uses
  ``getattr(stream, "_stream_error", None)`` (drift pin in the test module)
  instead of a bare attribute read; the ``tools`` span attribute uses the
  module-owned :func:`_serialize_tool_definitions` canonicalization pipeline
  (``observability.py`` in upstream commit ``b2549337f``) and the data-URI
  helper is owned locally.
- **Vendored** :class:`OtelAttr` subset (only the members this module emits),
  ``ROLE_EVENT_MAP``/``FINISH_REASON_MAP``, both histogram bucket tables and
  :class:`MessageListTimestampFilter` — byte-identical values, pinned against
  the framework originals by the E-group drift tests.
- **Scope/provider naming**: tracer/meter instrumentation scope is ``"chrys"``
  with the installed chrys version; the agent-span ``gen_ai.provider.name``
  default is the ``AGENT_PROVIDER_NAME`` ClassVar of the host class —
  ``"chrys"`` for the kernel Agent since P5.5. Chat-span provider names stay
  ``"openai"``/``"anthropic"``.
- **Function metric schema migration**: Step 5 intentionally renames the
  exported per-tool duration metric and function-name attribute from the
  pre-decoupling function metric schema to ``chrys.function.*``. Dashboards
  and alerts that consume per-tool metrics must update those metric selectors.

Per-tool span (N5): :func:`get_function_span` /
:func:`get_function_span_attributes` are ported (``:1984-2022``) together
with the function duration histogram (``_tools.py:197-220``) and the
framework-kwargs exclusion set (``_tools.py:715-724``) — the framework
emitted these *inside* ``FunctionTool.invoke``; with invoke chrys-owned and
telemetry-free, the loop's single invoke call site emits them instead,
gated on :data:`TELEMETRY_GATE` (same span name, attributes and metric,
chrys scope).

Framework paths deliberately not ported (dead in chrys, with reasons):

- ``EmbeddingTelemetryLayer`` (``:1632-1691``): chrys has no embedding
  clients.
- ``ObservabilitySettings`` + sticky enable/disable machinery, console
  exporters, VS Code extension port, ``.env``-file loading,
  ``enable_sensitive_telemetry`` (``:657-1150``): replaced by the plain gate
  here plus the chrys-owned provider setup.
- typing overloads on ``get_response``/``run``: typing-only sugar; runtime
  signatures are identical.

HARD RULE: this package imports only the stdlib, intra-package modules, the
``opentelemetry`` API package, and downward ``chrys.foundation.*`` modules —
never sibling or upward ``chrys.{kernel,service,orchestration,app}`` modules.
The chrys version for the tracer scope is therefore resolved via
``importlib.metadata``, not a root ``chrys`` import.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib.metadata
import json
import logging
import weakref
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from enum import Enum
from time import perf_counter, time_ns
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Protocol, TypeGuard, cast, overload

from opentelemetry import context as otel_context
from opentelemetry import metrics, trace
from pydantic import BaseModel

from chrys.foundation.observability.gate import TELEMETRY_GATE

from ._types import (
    AgentResponse,
    ChatResponse,
    ResponseStream,
    add_usage_details,
    merge_chat_options,
    normalize_messages,
)
from .client import _PreparedRequestObserverClient
from .tools import FunctionTool, normalize_tools

if TYPE_CHECKING:
    from ._types import (
        AgentResponseUpdate,
        AgentRunInputs,
        ChatOptions,
        ChatResponseUpdate,
        Content,
        FinishReason,
        FinishReasonLiteral,
        Message,
        ToolTypes,
        UsageDetails,
    )
    from .middleware import ChatMiddleware, FunctionMiddleware
    from .sessions import AgentSession

    class _ChatTelemetryBase(Protocol):
        def get_response(
            self,
            messages: Sequence[Message],
            *,
            stream: bool = False,
            options: Mapping[str, Any] | None = None,
            compaction_strategy: Any = None,
            tokenizer: Any = None,
            function_invocation_kwargs: Mapping[str, Any] | None = None,
            client_kwargs: Mapping[str, Any] | None = None,
            request_message_observer: Callable[[Sequence[Message]], None] | None = None,
        ) -> Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]: ...

    class _AgentTelemetryBase(Protocol):
        @overload
        def run(
            self,
            messages: AgentRunInputs | None = None,
            *,
            stream: Literal[False] = False,
            session: AgentSession | None = None,
            middleware: ChatMiddleware
            | FunctionMiddleware
            | Sequence[ChatMiddleware | FunctionMiddleware]
            | None = None,
            tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None = None,
            options: Mapping[str, Any] | None = None,
            compaction_strategy: Any = None,
            tokenizer: Any = None,
            function_invocation_kwargs: Mapping[str, Any] | None = None,
            client_kwargs: Mapping[str, Any] | None = None,
        ) -> Awaitable[AgentResponse[Any]]: ...

        @overload
        def run(
            self,
            messages: AgentRunInputs | None = None,
            *,
            stream: Literal[True],
            session: AgentSession | None = None,
            middleware: ChatMiddleware
            | FunctionMiddleware
            | Sequence[ChatMiddleware | FunctionMiddleware]
            | None = None,
            tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None = None,
            options: Mapping[str, Any] | None = None,
            compaction_strategy: Any = None,
            tokenizer: Any = None,
            function_invocation_kwargs: Mapping[str, Any] | None = None,
            client_kwargs: Mapping[str, Any] | None = None,
        ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

        @overload
        def run(
            self,
            messages: AgentRunInputs | None = None,
            *,
            stream: bool,
            session: AgentSession | None = None,
            middleware: ChatMiddleware
            | FunctionMiddleware
            | Sequence[ChatMiddleware | FunctionMiddleware]
            | None = None,
            tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None = None,
            options: Mapping[str, Any] | None = None,
            compaction_strategy: Any = None,
            tokenizer: Any = None,
            function_invocation_kwargs: Mapping[str, Any] | None = None,
            client_kwargs: Mapping[str, Any] | None = None,
        ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

else:
    _ChatTelemetryBase = object
    _AgentTelemetryBase = object


def _is_agent_response_stream(
    value: object,
) -> TypeGuard[ResponseStream[AgentResponseUpdate, AgentResponse[Any]]]:
    """Narrow the streaming branch of an agent-run result."""
    return isinstance(value, ResponseStream)


__all__ = [
    "FUNCTION_SPAN_EXCLUDED_KWARGS",
    "AgentTelemetryLayer",
    "ChatTelemetryLayer",
    "OtelAttr",
    "capture_exception",
    "create_mcp_client_span",
    "get_function_duration_histogram",
    "get_function_span",
    "get_function_span_attributes",
    "get_meter",
    "get_tracer",
    "set_mcp_span_error",
]

logger = logging.getLogger(__name__)

_SCOPE_NAME: Final[str] = "chrys"
try:
    _scope_version = importlib.metadata.version("chrys")
except importlib.metadata.PackageNotFoundError:  # pragma: no cover - editable installs always have metadata
    _scope_version = "0.0.0"
_SCOPE_VERSION: Final[str] = _scope_version


# Mirror of ``observability.py:104-113``: dedup channel between the agent
# layer and the inner chat layer so usage/response_id are not double-counted
# on the agent span.
INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS: Final[contextvars.ContextVar[set[str] | None]] = contextvars.ContextVar(
    "chrys_inner_response_telemetry_captured_fields", default=None
)
INNER_RESPONSE_ID_CAPTURED_FIELD: Final[str] = "response_id"
INNER_USAGE_CAPTURED_FIELD: Final[str] = "usage"
INNER_ACCUMULATED_USAGE: Final[contextvars.ContextVar[UsageDetails | None]] = contextvars.ContextVar(
    "chrys_inner_accumulated_usage", default=None
)

# Mirror of ``observability.py:116-147`` (E-group equality pins).
TOKEN_USAGE_BUCKET_BOUNDARIES: Final[tuple[float, ...]] = (
    1,
    4,
    16,
    64,
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    4194304,
    16777216,
    67108864,
)
OPERATION_DURATION_BUCKET_BOUNDARIES: Final[tuple[float, ...]] = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
)


class OtelAttr(str, Enum):  # noqa: UP042 - mirrors the framework class shape (str+Enum with value-returning __str__/__repr__), not StrEnum
    """Vendored subset of the framework ``OtelAttr`` (``observability.py:176-315``).

    Only the members this module emits; values byte-identical (E-group drift
    pin iterates this enum against the framework one).
    """

    OPERATION = "gen_ai.operation.name"
    PROVIDER_NAME = "gen_ai.provider.name"
    ERROR_TYPE = "error.type"
    PORT = "server.port"
    ADDRESS = "server.address"
    # Request attributes
    SEED = "gen_ai.request.seed"
    ENCODING_FORMATS = "gen_ai.request.encoding_formats"
    FREQUENCY_PENALTY = "gen_ai.request.frequency_penalty"
    PRESENCE_PENALTY = "gen_ai.request.presence_penalty"
    STOP_SEQUENCES = "gen_ai.request.stop_sequences"
    TOP_K = "gen_ai.request.top_k"
    CHOICE_COUNT = "gen_ai.request.choice.count"
    REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    REQUEST_TOP_P = "gen_ai.request.top_p"
    REQUEST_MODEL = "gen_ai.request.model"
    # Response attributes
    FINISH_REASONS = "gen_ai.response.finish_reasons"
    RESPONSE_ID = "gen_ai.response.id"
    RESPONSE_MODEL = "gen_ai.response.model"
    # Usage attributes
    INPUT_TOKENS = "gen_ai.usage.input_tokens"
    OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    CACHE_CREATION_INPUT_TOKENS = "gen_ai.usage.cache_creation.input_tokens"
    CACHE_READ_INPUT_TOKENS = "gen_ai.usage.cache_read.input_tokens"
    REASONING_OUTPUT_TOKENS = "gen_ai.usage.reasoning.output_tokens"
    # Tool attributes
    TOOL_DEFINITIONS = "gen_ai.tool.definitions"
    TOOL_CALL_ID = "gen_ai.tool.call.id"
    TOOL_DESCRIPTION = "gen_ai.tool.description"
    TOOL_NAME = "gen_ai.tool.name"
    TOOL_TYPE = "gen_ai.tool.type"
    TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
    TOOL_RESULT = "gen_ai.tool.call.result"
    # Agent attributes
    AGENT_ID = "gen_ai.agent.id"
    AGENT_NAME = "gen_ai.agent.name"
    AGENT_DESCRIPTION = "gen_ai.agent.description"
    CONVERSATION_ID = "gen_ai.conversation.id"
    INPUT_MESSAGES = "gen_ai.input.messages"
    OUTPUT_MESSAGES = "gen_ai.output.messages"
    SYSTEM_INSTRUCTIONS = "gen_ai.system_instructions"
    SYSTEM = "gen_ai.system"
    SERVICE_NAME = "service.name"
    SERVICE_VERSION = "service.version"
    # Client metric attributes
    T_UNIT = "tokens"
    T_TYPE = "gen_ai.token.type"
    T_TYPE_INPUT = "input"
    T_TYPE_OUTPUT = "output"
    DURATION_UNIT = "s"
    LLM_OPERATION_DURATION = "gen_ai.client.operation.duration"
    LLM_TOKEN_USAGE = "gen_ai.client.token.usage"
    # Activity events
    EVENT_NAME = "event.name"
    SYSTEM_MESSAGE = "gen_ai.system.message"
    USER_MESSAGE = "gen_ai.user.message"
    ASSISTANT_MESSAGE = "gen_ai.assistant.message"
    TOOL_MESSAGE = "gen_ai.tool.message"
    CHOICE = "gen_ai.choice"
    # Operation names
    CHAT_COMPLETION_OPERATION = "chat"
    AGENT_INVOKE_OPERATION = "invoke_agent"
    TOOL_EXECUTION_OPERATION = "execute_tool"
    # MCP attributes
    MCP_METHOD_NAME = "mcp.method.name"
    MCP_PROTOCOL_VERSION = "mcp.protocol.version"
    MCP_SESSION_ID = "mcp.session.id"
    PROMPT_NAME = "gen_ai.prompt.name"
    NETWORK_TRANSPORT = "network.transport"
    NETWORK_PROTOCOL_NAME = "network.protocol.name"
    # Measurement attributes. Step 5 deliberately migrates these exported
    # metric names to the Chrys-owned ``chrys.function.*`` schema.
    MEASUREMENT_FUNCTION_TAG_NAME = "chrys.function.name"
    MEASUREMENT_FUNCTION_INVOCATION_DURATION = "chrys.function.invocation.duration"

    def __repr__(self) -> str:
        """Return the string representation of the enum member."""
        return self.value

    def __str__(self) -> str:
        """Return the string representation of the enum member."""
        return self.value


# Mirror of ``observability.py:318-329``.
ROLE_EVENT_MAP = {
    "system": OtelAttr.SYSTEM_MESSAGE,
    "user": OtelAttr.USER_MESSAGE,
    "assistant": OtelAttr.ASSISTANT_MESSAGE,
    "tool": OtelAttr.TOOL_MESSAGE,
}
FINISH_REASON_MAP = {
    "stop": "stop",
    "content_filter": "content_filter",
    "tool_calls": "tool_call",
    "length": "length",
}
USAGE_DETAIL_TO_OTEL_ATTR: Final[tuple[tuple[str, OtelAttr], ...]] = (
    ("input_token_count", OtelAttr.INPUT_TOKENS),
    ("output_token_count", OtelAttr.OUTPUT_TOKENS),
    ("anthropic.cache_creation_input_tokens", OtelAttr.CACHE_CREATION_INPUT_TOKENS),
    ("anthropic.cache_read_input_tokens", OtelAttr.CACHE_READ_INPUT_TOKENS),
    ("deepseek.prompt_cache_hit_tokens", OtelAttr.CACHE_READ_INPUT_TOKENS),
    ("openai.cached_input_tokens", OtelAttr.CACHE_READ_INPUT_TOKENS),
    ("prompt/cached_tokens", OtelAttr.CACHE_READ_INPUT_TOKENS),
    ("cache_creation_input_token_count", OtelAttr.CACHE_CREATION_INPUT_TOKENS),
    ("cache_read_input_token_count", OtelAttr.CACHE_READ_INPUT_TOKENS),
    ("reasoning_output_token_count", OtelAttr.REASONING_OUTPUT_TOKENS),
    ("openai.reasoning_tokens", OtelAttr.REASONING_OUTPUT_TOKENS),
    ("completion/reasoning_tokens", OtelAttr.REASONING_OUTPUT_TOKENS),
)


class MessageListTimestampFilter(logging.Filter):
    """Mirror of ``observability.py:160-170``: bump per-message log timestamps.

    Chat-history events are emitted within nanoseconds of each other and many
    backends truncate timestamp resolution; incrementing by 1 microsecond per
    message index preserves a restorable order.
    """

    INDEX_KEY: ClassVar[str] = "chat_message_index"

    def filter(self, record: logging.LogRecord) -> bool:
        """Increment the timestamp of INFO logs by 1 microsecond."""
        if hasattr(record, self.INDEX_KEY):
            idx = getattr(record, self.INDEX_KEY)
            record.created += idx * 1e-6
        return True


logger.addFilter(MessageListTimestampFilter())


def get_tracer(
    instrumenting_module_name: str = _SCOPE_NAME,
    instrumenting_library_version: str = _SCOPE_VERSION,
    schema_url: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> trace.Tracer:
    """Mirror of ``observability.py:933-981`` with the chrys scope defaults."""
    return trace.get_tracer(
        instrumenting_module_name=instrumenting_module_name,
        instrumenting_library_version=instrumenting_library_version,
        schema_url=schema_url,
        attributes=attributes,
    )


def get_meter(
    name: str = _SCOPE_NAME,
    version: str = _SCOPE_VERSION,
    schema_url: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> metrics.Meter:
    """Mirror of ``observability.py:984-1033`` with the chrys scope defaults."""
    try:
        return metrics.get_meter(name=name, version=version, schema_url=schema_url, attributes=attributes)
    except TypeError:
        # Older OpenTelemetry releases do not support the attributes parameter.
        return metrics.get_meter(name=name, version=version, schema_url=schema_url)


def _get_duration_histogram() -> metrics.Histogram:
    """Mirror of ``observability.py:1328-1334``."""
    return get_meter().create_histogram(
        name=OtelAttr.LLM_OPERATION_DURATION,
        unit=OtelAttr.DURATION_UNIT,
        description="Captures the duration of operations of function-invoking chat clients",
        explicit_bucket_boundaries_advisory=OPERATION_DURATION_BUCKET_BOUNDARIES,
    )


def _get_token_usage_histogram() -> metrics.Histogram:
    """Mirror of ``observability.py:1337-1343``."""
    return get_meter().create_histogram(
        name=OtelAttr.LLM_TOKEN_USAGE,
        unit=OtelAttr.T_UNIT,
        description="Captures the token usage of chat clients",
        explicit_bucket_boundaries_advisory=TOKEN_USAGE_BUCKET_BOUNDARIES,
    )


def get_function_duration_histogram() -> metrics.Histogram:
    """Mirror of ``_tools.py:197-220`` (the live branch).

    The framework built this per ``FunctionTool`` instance at ctor time
    (NoOp when its settings were disabled at that moment); chrys builds it
    per loop layer through the proxy meter — same name/unit/buckets, and the
    loop only records when :data:`TELEMETRY_GATE` is on.
    """
    return get_meter().create_histogram(
        name=OtelAttr.MEASUREMENT_FUNCTION_INVOCATION_DURATION,
        unit=OtelAttr.DURATION_UNIT,
        description="Measures the duration of a function's execution",
        explicit_bucket_boundaries_advisory=OPERATION_DURATION_BUCKET_BOUNDARIES,
    )


# Mirror of ``_tools.py:715-729``: framework kwargs that are not JSON
# serializable, excluded from the sensitive ``TOOL_ARGUMENTS`` capture.
FUNCTION_SPAN_EXCLUDED_KWARGS: Final[frozenset[str]] = frozenset(
    {
        "chat_options",
        "tools",
        "tool_choice",
        "session",
        "conversation_id",
        "options",
        "response_format",
    }
)


def get_function_span_attributes(function: FunctionTool, tool_call_id: str | None = None) -> dict[str, str]:
    """Mirror of ``observability.py:1984-2002``: base attributes for a tool span."""
    attributes: dict[str, str] = {
        OtelAttr.OPERATION: OtelAttr.TOOL_EXECUTION_OPERATION,
        OtelAttr.TOOL_NAME: function.name,
        OtelAttr.TOOL_CALL_ID: tool_call_id or "unknown",
        OtelAttr.TOOL_TYPE: "function",
    }
    if function.description:
        attributes[OtelAttr.TOOL_DESCRIPTION] = function.description
    return attributes


def get_function_span(attributes: dict[str, str]) -> contextlib.AbstractContextManager[trace.Span]:
    """Mirror of ``observability.py:2005-2022``: start the ``execute_tool {name}`` span."""
    return get_tracer().start_as_current_span(
        name=f"{attributes[OtelAttr.OPERATION]} {attributes[OtelAttr.TOOL_NAME]}",
        attributes=attributes,
        set_status_on_exception=False,
        end_on_exit=True,
        record_exception=False,
    )


@contextlib.contextmanager
def create_mcp_client_span(
    method_name: str,
    target: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span, Any, Any]:
    """Create a Chrys-gated MCP client span."""
    span_name = f"{method_name} {target}" if target else method_name
    attrs: dict[str, Any] = {OtelAttr.MCP_METHOD_NAME: method_name}
    if attributes:
        attrs.update(attributes)
    tracer = get_tracer() if TELEMETRY_GATE.enabled else trace.NoOpTracer()
    span = tracer.start_span(span_name, kind=trace.SpanKind.CLIENT, attributes=attrs)
    with trace.use_span(
        span=span,
        end_on_exit=True,
        record_exception=True,
        set_status_on_exception=True,
    ) as current_span:
        yield current_span


def set_mcp_span_error(span: trace.Span, error_type: str, description: str | None = None) -> None:
    """Set MCP span error status and ``error.type``."""
    span.set_attribute(OtelAttr.ERROR_TYPE, error_type)
    span.set_status(trace.StatusCode.ERROR, description=description)


def _serialize_tool_definitions(tools: Any) -> str | None:
    """Serialize canonical OTel tool definitions without raising into a run.

    Mirrors the per-tool pipeline added to upstream ``observability.py`` by
    ``b2549337f``: normalize once, then independently convert, canonicalize,
    narrow, and serialize each tool so one malformed declaration cannot drop
    the valid definitions around it.
    """
    try:
        normalized_tools = normalize_tools(tools)
    except Exception:
        logger.warning(
            "Failed to normalize tool definitions for telemetry; skipping attribute.",
            exc_info=True,
        )
        return None
    if not normalized_tools:
        return None

    fragments: list[str] = []
    for tool_item in normalized_tools:
        try:
            definition = _build_tool_otel_definition(tool_item)
            if definition is None:
                continue
            fragments.append(json.dumps(definition, ensure_ascii=False))
        except Exception:
            logger.warning(
                "Failed to build or serialize telemetry definition for tool %r (%s); skipping tool.",
                _tool_name_for_log(tool_item),
                type(tool_item).__name__,
                exc_info=True,
            )
            continue
    return f"[{','.join(fragments)}]" if fragments else None


def _tool_name_for_log(tool_item: Any) -> str:
    """Return a best-effort tool name without risking the warning path."""
    if isinstance(tool_item, FunctionTool):
        return tool_item.name
    if isinstance(tool_item, Mapping):
        try:
            nested = tool_item.get("function") if tool_item.get("type") == "function" else None
            source = nested if isinstance(nested, Mapping) else tool_item
            for key in ("name", "server_label", "type"):
                value = source.get(key)
                if isinstance(value, str) and value:
                    return value
        except Exception:
            pass
    return type(tool_item).__name__


def _build_tool_otel_definition(tool_item: Any) -> dict[str, Any] | None:
    """Convert one tool declaration and canonicalize its OTel field set."""
    raw: Any
    if isinstance(tool_item, FunctionTool):
        raw = tool_item.to_json_schema_spec()
    else:
        model_dump = getattr(tool_item, "model_dump", None)
        if callable(model_dump):
            raw = model_dump(exclude_none=True)
        else:
            to_dict = getattr(tool_item, "to_dict", None)
            if callable(to_dict):
                raw = to_dict()
            elif isinstance(tool_item, Mapping):
                raw = tool_item
            else:
                logger.warning(
                    "Can't parse tool to OpenTelemetry tool definition: %s.",
                    type(tool_item).__name__,
                )
                return None

    if not isinstance(raw, Mapping):
        logger.warning(
            "Can't parse tool to OpenTelemetry tool definition: %s conversion returned %s.",
            type(tool_item).__name__,
            type(raw).__name__,
        )
        return None
    return _otel_definition_from_mapping(raw)


def _otel_definition_from_mapping(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    """Flatten provider tool shapes and retain only OTel-safe fields."""
    nested_function = raw.get("function") if raw.get("type") == "function" else None
    if isinstance(nested_function, Mapping):
        name = nested_function.get("name")
        if not isinstance(name, str) or not name:
            logger.warning("Can't parse tool to OpenTelemetry tool definition: missing 'name'.")
            return None
        return _otel_tool_definition(
            "function",
            name,
            description=nested_function.get("description"),
            parameters=nested_function.get("parameters"),
        )

    type_value = raw.get("type")
    if not isinstance(type_value, str) or not type_value:
        logger.warning("Can't parse tool to OpenTelemetry tool definition: missing 'type'.")
        return None

    name_value = next(
        (value for key in ("name", "server_label") if isinstance((value := raw.get(key)), str) and value),
        type_value,
    )
    return _otel_tool_definition(
        type_value,
        name_value,
        description=raw.get("description") or raw.get("server_description"),
        parameters=raw.get("parameters") or raw.get("input_schema"),
    )


def _otel_tool_definition(
    type_value: str,
    name_value: str,
    *,
    description: Any = None,
    parameters: Any = None,
) -> dict[str, Any]:
    """Emit only ``type``, ``name``, and non-empty descriptive fields."""
    definition: dict[str, Any] = {"type": type_value, "name": name_value}
    if description:
        definition["description"] = description
    if parameters:
        definition["parameters"] = parameters
    return definition


def _get_data_bytes_as_str(content: Content) -> str | None:
    """Return the base64 payload for data URI content.

    Returns the base64 payload of a ``data:`` URI; the framework raises its
    ``ContentError`` on a non-base64 data URI — mirrored as ``ValueError``
    (the framework class lives in the private ``exceptions`` tree and the
    condition is unreachable for framework-built ``Content``).
    """
    if content.type not in ("data", "uri"):
        return None
    uri = content.uri
    if not uri:
        return None
    if not uri.startswith("data:"):
        return None
    if ";base64," not in uri:
        raise ValueError("Data URI must use base64 encoding")
    _, data = uri.split(";base64,", 1)
    return data


@contextlib.contextmanager
def _activate_span(span: trace.Span) -> Generator[None]:
    """Mirror of ``observability.py:2081-2098``: per-pull span attach/detach."""
    token = otel_context.attach(trace.set_span_in_context(span))
    try:
        yield
    finally:
        otel_context.detach(token)


@contextlib.contextmanager
def _get_span(
    attributes: dict[str, Any],
    span_name_attribute: str,
) -> Generator[trace.Span, Any, Any]:
    """Mirror of ``observability.py:2101-2120``. ``attributes`` must contain the name key."""
    operation = attributes.get(OtelAttr.OPERATION, "operation")
    span_name = attributes.get(span_name_attribute, "unknown")
    span = get_tracer().start_span(f"{operation} {span_name}")
    span.set_attributes(attributes)
    with trace.use_span(
        span=span,
        end_on_exit=True,
        record_exception=False,
        set_status_on_exception=False,
    ) as current_span:
        yield current_span


def _start_streaming_span(attributes: dict[str, Any], span_name_attribute: str) -> trace.Span:
    """Mirror of ``observability.py:2123-2143``: non-current span for streaming.

    The caller owns ending it (cleanup hooks) and activating it around each
    pull (:func:`_activate_span` via ``with_pull_context_manager``) —
    attaching at creation would die in cross-context cleanup.
    """
    operation = attributes.get(OtelAttr.OPERATION, "operation")
    span_name = attributes.get(span_name_attribute, "unknown")
    span = get_tracer().start_span(f"{operation} {span_name}")
    span.set_attributes(attributes)
    return span


def _get_instructions_from_options(options: Any) -> str | list[str] | None:
    """Mirror of ``observability.py:2146-2157``."""
    if options is None:
        return None
    if isinstance(options, Mapping):
        instructions = cast("Mapping[str, Any]", options).get("instructions")
        if isinstance(instructions, str):
            return instructions
        if isinstance(instructions, list) and all(isinstance(item, str) for item in instructions):
            return instructions
        return None
    return None


# Mirror of ``observability.py:2167-2207``. Each entry:
# source_keys -> (otel_attribute_key, transform_func, check_options_first, default_value).
# The ``tools`` transform performs guarded, per-tool canonicalization so span
# construction cannot expose provider-only fields or raise into a live request.
OTEL_ATTR_MAP: dict[str | tuple[str, ...], tuple[str, Callable[[Any], Any] | None, bool, Any]] = {
    "choice_count": (OtelAttr.CHOICE_COUNT, None, False, 1),
    "operation_name": (OtelAttr.OPERATION, None, False, None),
    "system_name": (OtelAttr.SYSTEM, None, False, None),
    "provider_name": (OtelAttr.PROVIDER_NAME, None, False, None),
    "service_url": (OtelAttr.ADDRESS, None, False, None),
    "conversation_id": (OtelAttr.CONVERSATION_ID, None, True, None),
    "seed": (OtelAttr.SEED, None, True, None),
    "frequency_penalty": (OtelAttr.FREQUENCY_PENALTY, None, True, None),
    "max_tokens": (OtelAttr.REQUEST_MAX_TOKENS, None, True, None),
    "stop": (OtelAttr.STOP_SEQUENCES, None, True, None),
    "temperature": (OtelAttr.REQUEST_TEMPERATURE, None, True, None),
    "top_p": (OtelAttr.REQUEST_TOP_P, None, True, None),
    "presence_penalty": (OtelAttr.PRESENCE_PENALTY, None, True, None),
    "top_k": (OtelAttr.TOP_K, None, True, None),
    "encoding_formats": (
        OtelAttr.ENCODING_FORMATS,
        lambda v: json.dumps(v if isinstance(v, list) else [v]),
        True,
        None,
    ),
    "agent_id": (OtelAttr.AGENT_ID, None, False, None),
    "agent_name": (OtelAttr.AGENT_NAME, None, False, None),
    "agent_description": (OtelAttr.AGENT_DESCRIPTION, None, False, None),
    "model": (OtelAttr.REQUEST_MODEL, None, True, None),
    # Tools with validation - returns None if no valid tools
    "tools": (
        OtelAttr.TOOL_DEFINITIONS,
        _serialize_tool_definitions,
        True,
        None,
    ),
    # Error type extraction
    "error": (OtelAttr.ERROR_TYPE, lambda e: type(e).__name__, False, None),
    # thread_id overrides conversation_id - processed after conversation_id due to dict ordering
    "thread_id": (OtelAttr.CONVERSATION_ID, None, False, None),
}


def _get_span_attributes(**kwargs: Any) -> dict[str, Any]:
    """Mirror of ``observability.py:2210-2239``."""
    attributes: dict[str, Any] = {}
    options = kwargs.get("all_options", kwargs.get("options"))
    options_mapping = cast("Mapping[str, Any]", options) if isinstance(options, Mapping) else None

    for source_keys, (otel_key, transform_func, check_options, default_value) in OTEL_ATTR_MAP.items():
        keys = (source_keys,) if isinstance(source_keys, str) else source_keys

        value = None
        for key in keys:
            if check_options and options_mapping is not None:
                value = options_mapping.get(key)
            if value is None:
                value = kwargs.get(key)
            if value is not None:
                break

        if value is None and default_value is not None:
            value = default_value

        if value is not None:
            result = transform_func(value) if transform_func else value
            # Allow transform_func to return None to skip attribute
            if result is not None:
                attributes[otel_key] = result

    return attributes


def capture_exception(span: trace.Span, exception: BaseException, timestamp: int | None = None) -> None:
    """Mirror of ``observability.py:2242-2246``."""
    span.set_attribute(OtelAttr.ERROR_TYPE, type(exception).__name__)
    span.record_exception(exception=exception, timestamp=timestamp)
    span.set_status(status=trace.StatusCode.ERROR, description=repr(exception))


def _capture_system_instructions(span: trace.Span, system_instructions: str | list[str] | None) -> None:
    """Capture system instructions on a span."""
    if not system_instructions:
        return
    otel_sys_instructions = [
        {"type": "text", "content": instruction} for instruction in _normalize_instructions(system_instructions)
    ]
    span.set_attribute(OtelAttr.SYSTEM_INSTRUCTIONS, json.dumps(otel_sys_instructions, ensure_ascii=False))


def _capture_current_agent_system_instructions(
    agent_span: trace.Span,
    chat_span: trace.Span,
    system_instructions: str | list[str] | None,
) -> None:
    """Capture final chat instructions on the current agent span when the chat span belongs to it."""
    if not system_instructions or not agent_span.is_recording():
        return

    agent_attributes_obj = getattr(agent_span, "attributes", None)
    if not isinstance(agent_attributes_obj, Mapping):
        return
    agent_attributes = cast("Mapping[str, Any]", agent_attributes_obj)
    if agent_attributes.get(OtelAttr.OPERATION.value) != OtelAttr.AGENT_INVOKE_OPERATION:
        return

    if not _instructions_preserve_existing_agent_instructions(agent_attributes, system_instructions):
        return

    chat_parent = getattr(chat_span, "parent", None)
    agent_context = agent_span.get_span_context()
    if (
        chat_parent is None
        or chat_parent.span_id != agent_context.span_id
        or chat_parent.trace_id != agent_context.trace_id
    ):
        return

    _capture_system_instructions(agent_span, system_instructions)


def _normalize_instructions(system_instructions: str | list[str]) -> list[str]:
    """Normalize system instructions to telemetry text items."""
    return system_instructions if isinstance(system_instructions, list) else [system_instructions]


def _instructions_preserve_existing_agent_instructions(
    agent_attributes: Mapping[str, Any],
    system_instructions: str | list[str],
) -> bool:
    """Return True when chat instructions preserve the agent span's existing instructions."""
    existing = agent_attributes.get(OtelAttr.SYSTEM_INSTRUCTIONS)
    if not isinstance(existing, str):
        return True

    try:
        existing_items_obj = json.loads(existing)
    except json.JSONDecodeError:
        return False

    if not isinstance(existing_items_obj, list):
        return False
    existing_items = cast("list[object]", existing_items_obj)

    existing_contents: list[str] = []
    for item in existing_items:
        if not isinstance(item, Mapping):
            continue
        content = cast("Mapping[str, Any]", item).get("content")
        if isinstance(content, str):
            existing_contents.append(content)

    existing_text = "\n".join(existing_contents)
    new_text = "\n".join(_normalize_instructions(system_instructions))
    return new_text == existing_text or new_text.startswith(f"{existing_text}\n")


def _capture_messages(
    span: trace.Span,
    provider_name: str,
    messages: AgentRunInputs,
    system_instructions: str | list[str] | None = None,
    output: bool = False,
    finish_reason: FinishReasonLiteral | FinishReason | None = None,
) -> None:
    """Mirror of ``observability.py:2249-2284``: sensitive message events + span attrs."""
    normalized_messages = normalize_messages(messages)
    otel_messages: list[dict[str, Any]] = []
    for index, message in enumerate(normalized_messages):
        # Reuse the otel message representation for logging instead of calling to_dict()
        # to avoid expensive Pydantic serialization overhead
        otel_message = _to_otel_message(message)
        logger.info(
            otel_message,
            extra={
                OtelAttr.EVENT_NAME: OtelAttr.CHOICE if output else ROLE_EVENT_MAP.get(message.role),
                OtelAttr.PROVIDER_NAME: provider_name,
                MessageListTimestampFilter.INDEX_KEY: index,
            },
        )
        otel_messages.append(otel_message)
    # Total in both directions: an empty message list has no tail to annotate, and a
    # provider-specific reason no map entry covers is emitted verbatim rather than
    # raising past the caller's exception handler and mislabelling the span as failed.
    if finish_reason and otel_messages:
        otel_messages[-1]["finish_reason"] = FINISH_REASON_MAP.get(finish_reason, finish_reason)
    span.set_attribute(
        OtelAttr.OUTPUT_MESSAGES if output else OtelAttr.INPUT_MESSAGES, json.dumps(otel_messages, ensure_ascii=False)
    )
    _capture_system_instructions(span, system_instructions)


def _to_otel_message(message: Message) -> dict[str, Any]:
    """Mirror of ``observability.py:2287-2289``."""
    return {"role": message.role, "parts": [_to_otel_part(content) for content in message.contents]}


def _to_otel_part(content: Content) -> dict[str, Any] | None:
    """Mirror of ``observability.py:2292-2327``."""
    match content.type:
        case "text":
            return {"type": "text", "content": content.text}
        case "text_reasoning":
            return {"type": "reasoning", "content": content.text}
        case "uri":
            return {
                "type": "uri",
                "uri": content.uri,
                "mime_type": content.media_type,
                "modality": content.media_type.split("/")[0] if content.media_type else None,
            }
        case "data":
            return {
                "type": "blob",
                "content": _get_data_bytes_as_str(content),
                "mime_type": content.media_type,
                "modality": content.media_type.split("/")[0] if content.media_type else None,
            }
        case "function_call":
            return {"type": "tool_call", "id": content.call_id, "name": content.name, "arguments": content.arguments}
        case "function_result":
            return {
                "type": "tool_call_response",
                "id": content.call_id,
                "response": content.result if content.result is not None else "",
            }
        case _:
            # GenericPart in otel output messages json spec.
            # just required type, and arbitrary other fields.
            return content.to_dict(exclude_none=True)
    return None


def _mark_inner_response_telemetry_captured(response: ChatResponse | AgentResponse) -> None:
    """Mirror of ``observability.py:2330-2343``."""
    captured_fields = INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.get()
    if captured_fields is None:
        return
    if response.response_id:
        captured_fields.add(INNER_RESPONSE_ID_CAPTURED_FIELD)
    if response.usage_details:
        captured_fields.add(INNER_USAGE_CAPTURED_FIELD)
        accumulated = INNER_ACCUMULATED_USAGE.get()
        if accumulated is not None:
            INNER_ACCUMULATED_USAGE.set(add_usage_details(accumulated, response.usage_details))


def _apply_accumulated_usage(attributes: dict[str, Any], captured_fields: set[str]) -> None:
    """Mirror of ``observability.py:2346-2358``."""
    if INNER_USAGE_CAPTURED_FIELD not in captured_fields:
        return
    accumulated = INNER_ACCUMULATED_USAGE.get()
    if not accumulated:
        return
    _apply_usage_attributes(attributes, accumulated)


def _apply_usage_attributes(attributes: dict[str, Any], usage: Mapping[str, Any]) -> None:
    """Apply known usage details as standard OTel GenAI attributes."""
    for usage_key, otel_attr in USAGE_DETAIL_TO_OTEL_ATTR:
        value = usage.get(usage_key)
        if value is None or isinstance(value, bool) or not isinstance(value, int):
            continue
        attributes.setdefault(otel_attr, value)


def _get_response_attributes(
    attributes: dict[str, Any],
    response: ChatResponse | AgentResponse,
    *,
    capture_response_id: bool = True,
    capture_usage: bool = True,
) -> dict[str, Any]:
    """Mirror of ``observability.py:2361-2387``."""
    if capture_response_id and response.response_id:
        attributes[OtelAttr.RESPONSE_ID] = response.response_id
    finish_reason = response.finish_reason
    if not finish_reason:
        finish_reason = (
            getattr(response.raw_representation, "finish_reason", None) if response.raw_representation else None
        )
    if isinstance(finish_reason, str) and finish_reason:
        attributes[OtelAttr.FINISH_REASONS] = json.dumps([finish_reason])
    if model := getattr(response, "model", None):
        attributes[OtelAttr.RESPONSE_MODEL] = model
    if capture_usage and (usage := response.usage_details):
        _apply_usage_attributes(attributes, usage)
    return attributes


# Mirror of ``observability.py:2390-2397``.
GEN_AI_METRIC_ATTRIBUTES = (
    OtelAttr.OPERATION,
    OtelAttr.PROVIDER_NAME,
    OtelAttr.REQUEST_MODEL,
    OtelAttr.RESPONSE_MODEL,
    OtelAttr.ADDRESS,
    OtelAttr.PORT,
)


def _capture_response(
    span: trace.Span,
    attributes: dict[str, Any],
    operation_duration_histogram: metrics.Histogram | None = None,
    token_usage_histogram: metrics.Histogram | None = None,
    duration: float | None = None,
) -> None:
    """Mirror of ``observability.py:2400-2417``."""
    span.set_attributes(attributes)
    attrs: dict[str, Any] = {k: v for k, v in attributes.items() if k in GEN_AI_METRIC_ATTRIBUTES}
    if token_usage_histogram and (input_tokens := attributes.get(OtelAttr.INPUT_TOKENS)) is not None:
        token_usage_histogram.record(input_tokens, attributes={**attrs, OtelAttr.T_TYPE: OtelAttr.T_TYPE_INPUT})
    if token_usage_histogram and (output_tokens := attributes.get(OtelAttr.OUTPUT_TOKENS)) is not None:
        token_usage_histogram.record(output_tokens, {**attrs, OtelAttr.T_TYPE: OtelAttr.T_TYPE_OUTPUT})
    if operation_duration_histogram and duration is not None:
        if OtelAttr.ERROR_TYPE in attributes:
            attrs[OtelAttr.ERROR_TYPE] = attributes[OtelAttr.ERROR_TYPE]
        operation_duration_histogram.record(duration, attributes=attrs)


def _stream_error_of(stream: Any) -> Exception | None:
    """Probe ``ResponseStream._stream_error`` without a bare private read.

    The framework finalizers read the private attribute directly
    (``observability.py:1519,:1791``); the attribute's existence and
    None-default are pinned by the E-group drift tests.
    """
    return getattr(stream, "_stream_error", None)


def _stream_abandoned(stream: Any) -> bool:
    """Return whether the stream was closed before producing a final response.

    An abandoned stream's partial updates must never be finalized — a
    ``get_final_response()`` here would run result hooks (``after_run``
    context providers included) on a response the run never produced.
    """
    return getattr(stream, "_abandoned", False)


class ChatTelemetryLayer(_ChatTelemetryBase):
    """Layer that wraps chat client get_response with OpenTelemetry tracing.

    Mirror of the framework ``ChatTelemetryLayer`` (``observability.py:1354-1621``)
    gated on :data:`TELEMETRY_GATE`; cooperative mixin over
    ``super().get_response()`` exactly like the original (must never define
    ``_inner_get_response`` — the ``_IntermediateTextMixin`` chain bypasses
    this class below ``get_response``).
    """

    def __init__(self, *args: Any, otel_provider_name: str | None = None, **kwargs: Any) -> None:
        """Initialize telemetry attributes and histograms (``:1357-1362``: super first)."""
        super().__init__(*args, **kwargs)
        self.token_usage_histogram = _get_token_usage_histogram()
        self.duration_histogram = _get_duration_histogram()
        self.otel_provider_name = otel_provider_name or getattr(self, "OTEL_PROVIDER_NAME", "unknown")

    @staticmethod
    def _backfill_request_model(span: trace.Span, attributes: dict[str, Any]) -> None:
        """Mirror of ``:1364-1378``: backfill REQUEST_MODEL + span name from the response."""
        response_model = attributes.get(OtelAttr.RESPONSE_MODEL)
        if response_model and attributes.get(OtelAttr.REQUEST_MODEL, "unknown") == "unknown":
            attributes[OtelAttr.REQUEST_MODEL] = response_model
            operation = attributes.get(OtelAttr.OPERATION, "operation")
            span.update_name(f"{operation} {response_model}")

    @overload
    def get_response[ResponseModelT: BaseModel](
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = False,
        options: ChatOptions[ResponseModelT],
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        request_message_observer: Callable[[Sequence[Message]], None] | None = None,
    ) -> Awaitable[ChatResponse[ResponseModelT]]: ...

    @overload
    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = False,
        options: Mapping[str, Any] | None = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        request_message_observer: Callable[[Sequence[Message]], None] | None = None,
    ) -> Awaitable[ChatResponse[Any]]: ...

    @overload
    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        options: Mapping[str, Any] | None = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        request_message_observer: Callable[[Sequence[Message]], None] | None = None,
    ) -> ResponseStream[ChatResponseUpdate, ChatResponse[Any]]: ...

    @overload
    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool,
        options: Mapping[str, Any] | None = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        request_message_observer: Callable[[Sequence[Message]], None] | None = None,
    ) -> Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]: ...

    def get_response(  # ty: ignore[invalid-method-override]  # Overloads refine the cooperative mixin protocol.
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        options: Mapping[str, Any] | None = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
        request_message_observer: Callable[[Sequence[Message]], None] | None = None,
    ) -> Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]:
        """Trace chat responses with OpenTelemetry spans and metrics (``:1419-1621``)."""
        super_get_response = super().get_response  # type: ignore[misc]
        merged_client_kwargs = dict(client_kwargs) if client_kwargs is not None else {}
        prepared_observer_kwargs = (
            {"request_message_observer": request_message_observer}
            if request_message_observer is not None and isinstance(self, _PreparedRequestObserverClient)
            else {}
        )

        if not TELEMETRY_GATE.enabled:
            return super_get_response(  # type: ignore[no-any-return]
                messages=messages,
                stream=stream,
                options=options,
                compaction_strategy=compaction_strategy,
                tokenizer=tokenizer,
                function_invocation_kwargs=function_invocation_kwargs,
                client_kwargs=merged_client_kwargs,
                **prepared_observer_kwargs,
            )

        opts: Mapping[str, Any] = options or {}
        provider_name = str(self.otel_provider_name)
        model = merged_client_kwargs.get("model") or opts.get("model") or getattr(self, "model", None) or "unknown"
        service_url_func = getattr(self, "service_url", None)
        service_url = str(service_url_func() if callable(service_url_func) else "unknown")
        attributes = _get_span_attributes(
            operation_name=OtelAttr.CHAT_COMPLETION_OPERATION,
            provider_name=provider_name,
            model=model,
            service_url=service_url,
            **merged_client_kwargs,
        )

        if stream:
            agent_span = trace.get_current_span()
            span = _start_streaming_span(attributes, OtelAttr.REQUEST_MODEL)

            if TELEMETRY_GATE.sensitive_enabled and messages and span.is_recording():
                system_instructions = _get_instructions_from_options(opts)
                _capture_current_agent_system_instructions(agent_span, span, system_instructions)
                _capture_messages(
                    span=span,
                    provider_name=provider_name,
                    messages=messages,
                    system_instructions=system_instructions,
                )

            span_state = {"closed": False}
            duration_state: dict[str, float] = {}
            start_time = perf_counter()

            def _close_span() -> None:
                if span_state["closed"]:
                    return
                span_state["closed"] = True
                span.end()

            def _record_duration() -> None:
                duration_state["duration"] = perf_counter() - start_time

            try:
                with _activate_span(span):
                    result_stream = cast(
                        "ResponseStream[ChatResponseUpdate, ChatResponse[Any]]",
                        super_get_response(
                            messages=messages,
                            stream=True,
                            options=opts,
                            compaction_strategy=compaction_strategy,
                            tokenizer=tokenizer,
                            function_invocation_kwargs=function_invocation_kwargs,
                            client_kwargs=merged_client_kwargs,
                            **prepared_observer_kwargs,
                        ),
                    )
            except Exception as exception:
                capture_exception(span=span, exception=exception, timestamp=time_ns())
                _close_span()
                raise

            async def _finalize_stream() -> None:
                try:
                    stream_error = _stream_error_of(result_stream)
                    if stream_error is not None:
                        # Stream errored; skip get_final_response() to avoid firing
                        # result hooks such as after_run context providers on error
                        # paths. Capture the error on the span before returning.
                        capture_exception(span=span, exception=stream_error, timestamp=time_ns())
                        return
                    if _stream_abandoned(result_stream):
                        # Closed before completion (cancel/stall abandonment):
                        # the partial updates are not a response — close the
                        # span without finalizing them.
                        return
                    response: ChatResponse[Any] = await result_stream.get_final_response()
                    duration = duration_state.get("duration")
                    response_attributes = _get_response_attributes(attributes, response)
                    self._backfill_request_model(span, response_attributes)
                    _capture_response(
                        span=span,
                        attributes=response_attributes,
                        token_usage_histogram=self.token_usage_histogram,
                        operation_duration_histogram=self.duration_histogram,
                        duration=duration,
                    )
                    _mark_inner_response_telemetry_captured(response)
                    if (
                        TELEMETRY_GATE.sensitive_enabled
                        and isinstance(response, ChatResponse)
                        and response.messages
                        and span.is_recording()
                    ):
                        _capture_messages(
                            span=span,
                            provider_name=provider_name,
                            messages=response.messages,
                            finish_reason=response.finish_reason,  # type: ignore[arg-type]
                            output=True,
                        )
                except Exception as exception:
                    capture_exception(span=span, exception=exception, timestamp=time_ns())
                finally:
                    _close_span()

            # The pull context manager attaches the span around each underlying iterator pull so
            # that child spans created during the pull (e.g. HTTP requests, inner tool execution)
            # are parented under this chat span. Attach and detach happen in the same async
            # context as the pull, avoiding cross-context cleanup issues. The weakref finalizer
            # ensures the span is closed even if the stream is garbage collected without being
            # consumed.
            wrapped_stream: ResponseStream[ChatResponseUpdate, ChatResponse[Any]] = (
                result_stream.with_cleanup_hook(_record_duration)
                .with_cleanup_hook(_finalize_stream)
                .with_pull_context_manager(lambda: _activate_span(span))
            )
            weakref.finalize(wrapped_stream, _close_span)
            return wrapped_stream

        async def _get_response() -> ChatResponse:
            agent_span = trace.get_current_span()
            with _get_span(attributes=attributes, span_name_attribute=OtelAttr.REQUEST_MODEL) as span:
                if TELEMETRY_GATE.sensitive_enabled and messages and span.is_recording():
                    system_instructions = _get_instructions_from_options(opts)
                    _capture_current_agent_system_instructions(agent_span, span, system_instructions)
                    _capture_messages(
                        span=span,
                        provider_name=provider_name,
                        messages=messages,
                        system_instructions=system_instructions,
                    )
                start_time_stamp = perf_counter()
                try:
                    response = cast(
                        "ChatResponse[Any]",
                        await super_get_response(
                            messages=messages,
                            stream=False,
                            options=opts,
                            compaction_strategy=compaction_strategy,
                            tokenizer=tokenizer,
                            function_invocation_kwargs=function_invocation_kwargs,
                            client_kwargs=merged_client_kwargs,
                            **prepared_observer_kwargs,
                        ),
                    )
                except Exception as exception:
                    capture_exception(span=span, exception=exception, timestamp=time_ns())
                    raise
                duration = perf_counter() - start_time_stamp
                response_attributes = _get_response_attributes(attributes, response)
                self._backfill_request_model(span, response_attributes)
                _capture_response(
                    span=span,
                    attributes=response_attributes,
                    token_usage_histogram=self.token_usage_histogram,
                    operation_duration_histogram=self.duration_histogram,
                    duration=duration,
                )
                _mark_inner_response_telemetry_captured(response)
                if TELEMETRY_GATE.sensitive_enabled and response.messages and span.is_recording():
                    finish_reason = cast(
                        "FinishReason | None",
                        response.finish_reason if response.finish_reason in FINISH_REASON_MAP else None,
                    )
                    _capture_messages(
                        span=span,
                        provider_name=provider_name,
                        messages=response.messages,
                        finish_reason=finish_reason,
                        output=True,
                    )
                return response  # type: ignore[return-value,no-any-return]

        return _get_response()


class AgentTelemetryLayer(_AgentTelemetryBase):
    """Layer that wraps agent run with OpenTelemetry tracing.

    Mirror of the framework ``AgentTelemetryLayer`` (``observability.py:1694-1978``)
    gated on :data:`TELEMETRY_GATE`, with the ContextVar lifecycle fix
    described in the module docstring: the non-streaming branch confines the
    INNER dedup vars to the ``_run()`` coroutine so the returned coroutine can
    be driven from any task (``asyncio.create_task`` included); the streaming
    branch hardens the finalizer against cross-context resets.
    """

    def __init__(
        self,
        *args: Any,
        otel_agent_provider_name: str | None = None,
        otel_provider_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize telemetry attributes and histograms (``:1697-1710``: provider name before super)."""
        self.otel_provider_name = (
            otel_agent_provider_name or otel_provider_name or getattr(self, "AGENT_PROVIDER_NAME", "unknown")
        )
        super().__init__(*args, **kwargs)
        self.token_usage_histogram = _get_token_usage_histogram()
        self.duration_histogram = _get_duration_histogram()

    def _trace_agent_invocation(
        self,
        *,
        messages: AgentRunInputs | None,
        session: AgentSession | None,
        merged_options: Mapping[str, Any],
        client_kwargs: Mapping[str, Any] | None,
        stream: bool,
        execute: Callable[[], Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]],
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        """Trace an agent invocation while delegating execution to ``execute`` (``:1712-1881``)."""
        if not TELEMETRY_GATE.enabled:
            return execute()

        provider_name = str(self.otel_provider_name)
        merged_client_kwargs = dict(client_kwargs) if client_kwargs is not None else {}
        attributes = _get_span_attributes(
            operation_name=OtelAttr.AGENT_INVOKE_OPERATION,
            provider_name=provider_name,
            agent_id=getattr(self, "id", "unknown"),
            agent_name=getattr(self, "name", None) or getattr(self, "id", "unknown"),
            agent_description=getattr(self, "description", None),
            thread_id=session.service_session_id if session else None,
            all_options=dict(merged_options),
            **merged_client_kwargs,
        )

        if stream:
            # Streaming keeps the framework ContextVar shape (set in this sync
            # call, reset in the finalizer): the inner chat layer reads the
            # vars during pulls, which happen in the consumer's context — in
            # every chrys streaming caller the stream is created and consumed
            # in the same task, so set/reset stay context-consistent.
            inner_response_telemetry_captured_fields: set[str] = set()
            captured_fields_token = INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.set(
                inner_response_telemetry_captured_fields
            )
            accumulated_usage_token = INNER_ACCUMULATED_USAGE.set({})

            span = _start_streaming_span(attributes, OtelAttr.AGENT_NAME)

            if TELEMETRY_GATE.sensitive_enabled and messages and span.is_recording():
                _capture_messages(
                    span=span,
                    provider_name=provider_name,
                    messages=messages,
                    system_instructions=_get_instructions_from_options(dict(merged_options)),
                )

            span_state = {"closed": False}
            duration_state: dict[str, float] = {}
            start_time = perf_counter()

            def _close_span() -> None:
                if span_state["closed"]:
                    return
                span_state["closed"] = True
                span.end()

            def _record_duration() -> None:
                duration_state["duration"] = perf_counter() - start_time

            def _reset_inner_vars() -> None:
                # The finalizer may run in a different context than the sync
                # set above (only when a caller hands the un-consumed stream
                # across tasks); a cross-context reset must not break span
                # finalization, so it degrades to a debug log. The vars are
                # task-scoped copies — a leaked set in a dead context is inert.
                try:
                    INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.reset(captured_fields_token)
                    INNER_ACCUMULATED_USAGE.reset(accumulated_usage_token)
                except ValueError:
                    logger.debug("Skipped cross-context reset of inner telemetry ContextVars")

            try:
                with _activate_span(span):
                    run_result: object = execute()
                if _is_agent_response_stream(run_result):
                    result_stream = run_result
                elif isinstance(run_result, Awaitable):
                    result_stream = ResponseStream.from_awaitable(
                        cast(
                            "Awaitable[ResponseStream[AgentResponseUpdate, AgentResponse[Any]]]",
                            run_result,
                        )
                    )
                else:
                    raise RuntimeError("Streaming telemetry requires a ResponseStream result.")
            except Exception as exception:
                capture_exception(span=span, exception=exception, timestamp=time_ns())
                _reset_inner_vars()
                _close_span()
                raise

            async def _finalize_stream() -> None:
                try:
                    stream_error = _stream_error_of(result_stream)
                    if stream_error is not None:
                        # Stream errored; skip get_final_response() to avoid firing
                        # result hooks such as after_run context providers on error
                        # paths. Capture the error on the span before returning.
                        capture_exception(span=span, exception=stream_error, timestamp=time_ns())
                        return
                    if _stream_abandoned(result_stream):
                        # Closed before completion (cancel/stall abandonment):
                        # the partial updates are not a response — close the
                        # span without finalizing them.
                        return
                    response: AgentResponse[Any] = await result_stream.get_final_response()
                    duration = duration_state.get("duration")
                    response_attributes = _get_response_attributes(
                        attributes,
                        response,
                        capture_response_id=INNER_RESPONSE_ID_CAPTURED_FIELD
                        not in inner_response_telemetry_captured_fields,
                        capture_usage=INNER_USAGE_CAPTURED_FIELD not in inner_response_telemetry_captured_fields,
                    )
                    _apply_accumulated_usage(response_attributes, inner_response_telemetry_captured_fields)
                    _capture_response(span=span, attributes=response_attributes, duration=duration)
                    if (
                        TELEMETRY_GATE.sensitive_enabled
                        and isinstance(response, AgentResponse)
                        and response.messages
                        and span.is_recording()
                    ):
                        _capture_messages(
                            span=span,
                            provider_name=provider_name,
                            messages=response.messages,
                            output=True,
                        )
                except Exception as exception:
                    capture_exception(span=span, exception=exception, timestamp=time_ns())
                finally:
                    # Close-first (divergence from ``:1822-1825``): a
                    # cross-context reset must never leave the span open.
                    _close_span()
                    _reset_inner_vars()

            # The pull context manager attaches the span around each underlying iterator pull so
            # that child spans created during the pull (e.g. inner chat completion spans from the
            # underlying ChatTelemetryLayer) are parented under this agent invoke span. Attach and
            # detach happen in the same async context as the pull, avoiding cross-context cleanup
            # issues. The weakref finalizer ensures the span is closed even if the stream is
            # garbage collected without being consumed.
            wrapped_stream: ResponseStream[AgentResponseUpdate, AgentResponse[Any]] = (
                result_stream.with_cleanup_hook(_record_duration)
                .with_cleanup_hook(_finalize_stream)
                .with_pull_context_manager(lambda: _activate_span(span))
            )
            weakref.finalize(wrapped_stream, _close_span)
            return wrapped_stream

        async def _run() -> AgentResponse[Any]:
            # ContextVar fix (framework sets these in the sync ``run()`` call,
            # ``:1742-1746``): set AND reset inside this coroutine so both run
            # in the awaiting task's context no matter which task that is —
            # the inner chat layer executes within ``await execute()`` below
            # and therefore sees the vars regardless.
            inner_response_telemetry_captured_fields: set[str] = set()
            captured_fields_token = INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.set(
                inner_response_telemetry_captured_fields
            )
            accumulated_usage_token = INNER_ACCUMULATED_USAGE.set({})
            try:
                with _get_span(attributes=attributes, span_name_attribute=OtelAttr.AGENT_NAME) as span:
                    if TELEMETRY_GATE.sensitive_enabled and messages and span.is_recording():
                        _capture_messages(
                            span=span,
                            provider_name=provider_name,
                            messages=messages,
                            system_instructions=_get_instructions_from_options(dict(merged_options)),
                        )
                    start_time_stamp = perf_counter()
                    try:
                        response = await cast("Awaitable[AgentResponse[Any]]", execute())
                    except Exception as exception:
                        capture_exception(span=span, exception=exception, timestamp=time_ns())
                        raise
                    duration = perf_counter() - start_time_stamp
                    if response:
                        response_attributes = _get_response_attributes(
                            attributes,
                            response,
                            capture_response_id=INNER_RESPONSE_ID_CAPTURED_FIELD
                            not in inner_response_telemetry_captured_fields,
                            capture_usage=INNER_USAGE_CAPTURED_FIELD not in inner_response_telemetry_captured_fields,
                        )
                        _apply_accumulated_usage(response_attributes, inner_response_telemetry_captured_fields)
                        _capture_response(span=span, attributes=response_attributes, duration=duration)
                        if TELEMETRY_GATE.sensitive_enabled and response.messages and span.is_recording():
                            _capture_messages(
                                span=span,
                                provider_name=provider_name,
                                messages=response.messages,
                                output=True,
                            )
                    return response
            finally:
                INNER_RESPONSE_TELEMETRY_CAPTURED_FIELDS.reset(captured_fields_token)
                INNER_ACCUMULATED_USAGE.reset(accumulated_usage_token)

        return _run()

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: Literal[False] = False,
        session: AgentSession | None = None,
        middleware: ChatMiddleware | FunctionMiddleware | Sequence[ChatMiddleware | FunctionMiddleware] | None = None,
        tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> Awaitable[AgentResponse[Any]]: ...

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: Literal[True],
        session: AgentSession | None = None,
        middleware: ChatMiddleware | FunctionMiddleware | Sequence[ChatMiddleware | FunctionMiddleware] | None = None,
        tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    @overload
    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool,
        session: AgentSession | None = None,
        middleware: ChatMiddleware | FunctionMiddleware | Sequence[ChatMiddleware | FunctionMiddleware] | None = None,
        tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]: ...

    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        middleware: ChatMiddleware | FunctionMiddleware | Sequence[ChatMiddleware | FunctionMiddleware] | None = None,
        tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        """Trace agent runs with OpenTelemetry spans and metrics (``:1931-1978``)."""
        super_run = cast(
            "Callable[..., Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]]",
            super().run,  # type: ignore[misc]
        )
        super_run_kwargs: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
            "session": session,
            "tools": tools,
            "options": options,
            "compaction_strategy": compaction_strategy,
            "tokenizer": tokenizer,
            "function_invocation_kwargs": function_invocation_kwargs,
            "client_kwargs": client_kwargs,
        }
        if middleware is not None:
            super_run_kwargs["middleware"] = middleware

        default_options = dict(getattr(self, "default_options", {}))
        merged_client_kwargs = dict(client_kwargs) if client_kwargs is not None else {}
        merged_options: dict[str, Any] = merge_chat_options(
            default_options, dict(options) if options is not None else {}
        )
        return self._trace_agent_invocation(
            messages=messages,
            session=session,
            merged_options=merged_options,
            client_kwargs=merged_client_kwargs,
            stream=stream,
            execute=lambda: super_run(**super_run_kwargs),
        )
