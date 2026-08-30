# Copyright (c) 2026 Chrys. All rights reserved.

"""Chrys-owned agent orchestrator.

Own agent implementation: :class:`_AgentCore` drives history + provider
contributions + instructions -> ``client.get_response()`` ->
``AgentResponse`` aggregation -> before/after_run, and :class:`Agent` stacks
the chrys-owned
:class:`~.instrumentation.AgentTelemetryLayer` on top exactly like the framework
production class minus the agent-middleware pipeline. The executable
acceptance baseline is ``tests/kernel/test_agent.py``.

The client stack underneath is the P5.2 shape and is consumed unchanged::

    Agent = AgentTelemetryLayer over _AgentCore          (this module)
    client = ToolLoopLayer(ChatMiddlewareLayer(instrumented Raw*))
    session/providers = chrys ``kernel.sessions`` (P5.4)

Framework paths deliberately not ported (dead in chrys, with reasons):

- per-service-call history persistence (``:377,:802-824,:1215-1219,:1324-1340``
  and the ``_suppress_response_id`` stream hook ``:1101-1104``): chrys never
  sets ``require_per_service_call_history_persistence``, so
  ``suppress_response_id`` is constantly False and the whole branch family
  collapses.
- MCPTool separation/expansion and the mcp exit-stack entries
  (``:746-750,:1262-1281``): chrys hands the agent pre-expanded flat
  ``FunctionTool`` lists (the mcp adapter exposes ``lease.functions``), never
  ``MCPTool`` instances.
- the agent middleware pipeline (``AgentMiddlewareLayer``/``AgentContext``):
  chrys has zero agent-bucket middleware; ctor/run middleware are
  function/chat only and are forwarded to the client stack (see
  :meth:`Agent.run`).
- ``as_tool``/``as_mcp_server``, the ctor ``default_options`` parameter, and
  the not-function-invoking-client warning (``:722-725``): unused by chrys
  (the client is always the chrys ``ToolLoopLayer``).

Telemetry: the agent-level OTel span layer is the chrys-owned
:class:`~.instrumentation.AgentTelemetryLayer` (P5.5), which also fixes the
framework's INNER ContextVar lifecycle so the returned non-streaming
coroutine may be driven from any task. ``AGENT_PROVIDER_NAME`` is ``"chrys"``
since P5.5 (user-approved rebrand of the agent-span
``gen_ai.provider.name``); chat-span provider names are unchanged.

HARD RULE: kernel modules may import only the stdlib, intra-package modules,
allowed third-party packages, and downward ``chrys.foundation.*`` modules.
Sibling or upward ``chrys.*`` imports remain forbidden.
The streaming/non-streaming
finalizers write ``SessionContext._response`` directly — since P5.4 that is an
intra-package private write into ``kernel.sessions`` (``response`` stays a
read-only property for providers), mirroring how the framework finalized via
the private field (``_agents.py:1079,:1411``).
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack
from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, TypedDict, overload
from uuid import uuid4

from ._types import (
    AgentResponse,
    ChatResponseUpdate,
    ResponseStream,
    _build_agent_response_from_chat_response,
    map_chat_to_agent_update,
    normalize_messages,
)
from .client import (
    CONVERSATION_HANDLE_KEYS,
    collect_conversation_handles,
    continuation_token_response_id,
    conversation_handle_value,
    resolve_storage_mode_and_handles,
    strip_invalidated_conversation_handles,
)
from .exceptions import AgentInvalidResponseException
from .instrumentation import AgentTelemetryLayer
from .middleware import split_middleware
from .sessions import (
    AgentSession,
    HistoryProvider,
    InMemoryHistoryProvider,
    ServiceFallbackHistoryProvider,
    SessionContext,
    is_local_history_conversation_id,
)

# The chrys normalize_tools (N5): delegates to the framework surface and
# upcasts framework-base instances — every tool entering the run pipeline
# through the agent is guaranteed to be the chrys FunctionTool subclass.
from .tools import normalize_tools

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from ._types import (
        AgentResponseUpdate,
        AgentRunInputs,
        ChatResponse,
        Message,
        ToolTypes,
    )
    from .middleware import ChatMiddleware, FunctionMiddleware
    from .sessions import ContextProvider


def _get_tool_name(tool: Any) -> str | None:
    """Return the name of a function-tool declaration or tool-like object."""
    if isinstance(tool, Mapping):
        func = tool.get("function", None)
        if func and isinstance(func, Mapping):
            name = func.get("name")
            return name if isinstance(name, str) else None
        return None
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) else None


def _tool_identity(tool: Any) -> Any:
    """Return duplicate identity, preserving residual framework source identity after cloning."""
    return getattr(tool, "_foreign_invocation_state_source", tool)


def _append_unique_tools(
    existing_tools: list[Any],
    new_tools: Sequence[Any],
    *,
    duplicate_error_message: str | None = None,
) -> list[Any]:
    """Append tools while preserving the run-pipeline duplicate-name contract.

    Appends in place and returns the same list. Nameless tools always append;
    the same object (or same cloned residual framework source) seen twice is
    skipped; a *different* object under an existing name raises ``ValueError``.
    """
    seen_by_name: dict[str, Any] = {}
    for tool_item in existing_tools:
        if tool_name := _get_tool_name(tool_item):
            seen_by_name[tool_name] = tool_item

    for tool_item in new_tools:
        tool_name = _get_tool_name(tool_item)
        if tool_name is None:
            existing_tools.append(tool_item)
            continue

        existing_tool = seen_by_name.get(tool_name)
        if existing_tool is None:
            seen_by_name[tool_name] = tool_item
            existing_tools.append(tool_item)
            continue

        if existing_tool is tool_item or _tool_identity(existing_tool) is _tool_identity(tool_item):
            continue

        message = duplicate_error_message or "Tool names must be unique."
        raise ValueError(f"Duplicate tool name '{tool_name}'. {message}")

    return existing_tools


def _merge_options(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge base and override chat options.

    Returns a new dict; ``None`` override values are skipped, ``tools`` are
    unioned via :func:`_append_unique_tools`, ``logit_bias``/``metadata`` are
    dict-merged, ``instructions`` are newline-concatenated, everything else is
    replaced. ``None`` is treated as "unset" throughout: the merged result is
    stripped of any remaining ``None`` values in a final pass so unset options
    are never forwarded (e.g. an unset ``store`` is left
    for the service to default).
    """
    result = dict(base)

    for key, value in override.items():
        if value is None:
            continue
        if key == "tools" and (result.get("tools") or value):
            base_tools = normalize_tools(result.get("tools"))
            override_tools = normalize_tools(value)
            result["tools"] = _append_unique_tools(
                list(base_tools),
                override_tools,
                duplicate_error_message="Tool names must be unique.",
            )
        elif key == "logit_bias" and result.get("logit_bias"):
            result["logit_bias"] = {**result["logit_bias"], **value}
        elif key == "metadata" and result.get("metadata"):
            result["metadata"] = {**result["metadata"], **value}
        elif key == "instructions" and result.get("instructions"):
            result["instructions"] = f"{result['instructions']}\n{value}"
        else:
            result[key] = value
    return {key: value for key, value in result.items() if value is not None}


class _RunContext(TypedDict):
    """Prepared per-run state (mirror of ``_agents.py:170-181`` minus ``suppress_response_id``)."""

    session: AgentSession | None
    session_context: SessionContext
    input_messages: Sequence[Message]
    session_messages: Sequence[Message]
    agent_name: str
    chat_options: MutableMapping[str, Any]
    compaction_strategy: Any
    tokenizer: Any
    client_kwargs: Mapping[str, Any]
    function_invocation_kwargs: Mapping[str, Any]


class _AgentCore:
    """Own mirror of the ``RawAgent`` orchestration trunk (``_agents.py:599-1632``).

    Constructor folds the relevant ``BaseAgent`` semantics in directly
    (``:379-409``) and ends with a bare cooperative ``super().__init__()`` so
    the class composes as the MRO terminus under the chrys-owned
    :class:`~.instrumentation.AgentTelemetryLayer`.

    The ``run()`` surface and internal phase split mirror the framework
    method-for-method so the streaming chain shape is preserved bit-for-bit:
    ``ResponseStream.from_awaitable`` around ``map(transform, finalizer)``
    around the tool-loop stream, with ``_propagate_conversation_id`` as a
    transform hook and ``_post_hook`` as a result hook (``:1112-1122``).
    """

    # Read by the AgentTelemetryLayer constructor via getattr BEFORE this
    # class's __init__ runs (instrumentation.py mirror of the framework pre-super
    # read, ``observability.py:1705-1708``) — must stay a ClassVar. The value
    # is the P5.5 agent-span ``gen_ai.provider.name`` rebrand ("chrys"
    # instead of the retired framework provider name.
    AGENT_PROVIDER_NAME: ClassVar[str] = "chrys"

    def __init__(
        self,
        client: Any,
        instructions: str | None = None,
        *,
        id: str | None = None,  # shadows the builtin on purpose: framework ctor parity
        name: str | None = None,
        description: str | None = None,
        tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None = None,
        context_providers: Sequence[ContextProvider] | None = None,
        middleware: ChatMiddleware | FunctionMiddleware | Sequence[ChatMiddleware | FunctionMiddleware] | None = None,
        additional_properties: MutableMapping[str, Any] | None = None,
    ) -> None:
        """Mirror of ``RawAgent.__init__`` (``:666-780``) minus dead parameters.

        Dropped vs the framework: ``default_options`` (chrys never passes it),
        ``require_per_service_call_history_persistence`` (constantly False),
        and ctor-level ``compaction_strategy``/``tokenizer`` (chrys always
        passes them per run; the attributes are still set to ``None`` because
        ``_prepare_run_context`` falls back to ``x or self.x``,
        ``:1371-1372``).
        """
        # BaseAgent field semantics (:400-409): generated id, listified
        # providers, middleware stored as given (None stays None — the run
        # method re-reads it every call for dynamic-change support).
        self.id = id if id is not None else str(uuid4())
        self.name = name
        self.description = description
        self.context_providers: list[ContextProvider] = list(context_providers or [])
        self.middleware: ChatMiddleware | FunctionMiddleware | Sequence[ChatMiddleware | FunctionMiddleware] | None = (
            middleware
        )
        self.additional_properties: MutableMapping[str, Any] = additional_properties or {}

        self.client = client
        self.compaction_strategy: Any = None
        self.tokenizer: Any = None

        # default_options mirror (:752-778), simplified to the keys this ctor
        # can produce: instructions, the "auto" tool_choice default, the
        # normalized flat tool list (no MCPTool split — chrys tools are always
        # pre-expanded FunctionTools), and the client-derived model (resolved
        # through the layered clients' __getattr__ delegation). Kept as a
        # plain dict so AgentTelemetryLayer can read
        # ``getattr(self, "default_options", {})`` for span attributes.
        initial_tools = _append_unique_tools([], normalize_tools(tools))
        default_options: dict[str, Any] = {
            "instructions": instructions,
            "tool_choice": "auto",
            "tools": initial_tools,
            "model": getattr(client, "model", None),
        }
        self.default_options: dict[str, Any] = {k: v for k, v in default_options.items() if v is not None}
        self._async_exit_stack = AsyncExitStack()
        self._update_agent_name_and_description()
        super().__init__()

    def _update_agent_name_and_description(self) -> None:
        """Mirror of ``:843-852``: optional delegation hook on the client.

        No client in the current chrys stack implements it (the getattr
        resolves through the ToolLoopLayer/ChatMiddlewareLayer delegation to
        the raw client), so this is a no-op today — kept because the framework
        Agent called it on the same stack.
        """
        update_fn = getattr(self.client, "_update_agent_name_and_description", None)
        if callable(update_fn):
            update_fn(self.name, self.description)

    async def __aenter__(self) -> Self:
        """Mirror of ``:782-797`` minus the mcp_tools chain.

        A no-op for the current chrys client stack (no layer implements the
        async-CM protocol), but the enter-if-CM check is kept so a future
        client shape cleans up through the same stack.
        """
        if isinstance(self.client, AbstractAsyncContextManager):
            await self._async_exit_stack.enter_async_context(self.client)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Mirror of ``:826-841``."""
        await self._async_exit_stack.aclose()

    def create_session(self, *, session_id: str | None = None) -> AgentSession:
        """Mirror of ``BaseAgent.create_session`` (``:411-432``)."""
        return AgentSession(session_id=session_id)

    def get_session(self, service_session_id: str, *, session_id: str | None = None) -> AgentSession:
        """Mirror of ``BaseAgent.get_session`` (``:434-449``)."""
        return AgentSession(session_id=session_id, service_session_id=service_session_id)

    def _get_agent_name(self) -> str:
        """Mirror of ``:1626-1632``."""
        return self.name or "UnnamedAgent"

    def run(
        self,
        messages: AgentRunInputs | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None = None,
        options: Mapping[str, Any] | None = None,
        compaction_strategy: Any = None,
        tokenizer: Any = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> Awaitable[AgentResponse[Any]] | ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        """Mirror of ``RawAgent.run`` (``:899-977``).

        ``stream=False`` returns an un-awaited coroutine; ``stream=True``
        returns ``ResponseStream.from_awaitable(...)`` synchronously — nothing
        (including ``_prepare_run_context`` provider calls) executes at call
        time in either mode.
        """

        async def _prepare() -> _RunContext:
            return await self._prepare_run_context(
                messages=messages,
                session=session,
                tools=tools,
                options=options,
                compaction_strategy=compaction_strategy,
                tokenizer=tokenizer,
                function_invocation_kwargs=function_invocation_kwargs,
                client_kwargs=client_kwargs,
            )

        if not stream:

            async def _run_non_streaming() -> AgentResponse[Any]:
                ctx = await _prepare()
                response = await self._call_chat_client(ctx, stream=False)
                return await self._parse_non_streaming_response(ctx, response)

            return _run_non_streaming()

        async def _run_streaming() -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
            ctx = await _prepare()
            stream_response = self._call_chat_client(ctx, stream=True)
            return self._parse_streaming_response(ctx, stream_response)

        return ResponseStream.from_awaitable(_run_streaming())

    def _call_chat_client(self, context: _RunContext, *, stream: bool) -> Any:
        """Mirror of ``:995-1021`` (single body; the framework's two branches differ only in overload typing)."""
        return self.client.get_response(
            messages=context["session_messages"],
            stream=stream,
            options=context["chat_options"],
            compaction_strategy=context["compaction_strategy"],
            tokenizer=context["tokenizer"],
            function_invocation_kwargs=context["function_invocation_kwargs"],
            client_kwargs=context["client_kwargs"],
        )

    async def _prepare_run_context(
        self,
        *,
        messages: AgentRunInputs | None,
        session: AgentSession | None,
        tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None,
        options: Mapping[str, Any] | None,
        compaction_strategy: Any,
        tokenizer: Any,
        function_invocation_kwargs: Mapping[str, Any] | None,
        client_kwargs: Mapping[str, Any] | None,
    ) -> _RunContext:
        """Mirror of ``:1160-1375`` minus the dead paths listed in the module docstring."""
        opts = dict(options) if options else {}
        existing_additional_args: dict[str, Any] = opts.pop("additional_function_arguments", None) or {}

        # Get tools from options or named parameter (named param takes precedence)
        tools_ = tools if tools is not None else opts.pop("tools", None)

        input_messages = normalize_messages(messages)

        # Combine agent-level defaults with runtime options up front so the
        # store/conversation_id decisions below read from a single place
        # (:1184); _merge_options applies the same precedence used for the
        # actual client call (runtime wins; unset/None falls back to the agent
        # default).
        effective_options = _merge_options(self.default_options, opts)

        # `store` in runtime or agent options takes precedence over the
        # client's default storage behavior (the STORES_BY_DEFAULT getattr
        # resolves through the layered clients' __getattr__ delegation). An
        # explicit store=False forces local history injection even when the
        # client stores server-side by default; store=None/unset falls back to
        # STORES_BY_DEFAULT: under the merge's None-strip an explicit None
        # means "unset", not "falsy False".
        # The upstream per-service-call consumers of this hint stay unported.
        try:
            stores_by_default = bool(self.client.STORES_BY_DEFAULT)
        except AttributeError:
            stores_by_default = False
        try:
            force_stateless = bool(self.client.FORCES_STATELESS)
        except AttributeError:
            force_stateless = False
        storage_resolution = resolve_storage_mode_and_handles(
            effective_options,
            stores_by_default=stores_by_default,
            client_kwargs=client_kwargs,
            force_stateless=force_stateless,
        )
        service_stores_history = storage_resolution.service_side
        # One normalized, capability- and invalidation-aware reflection view
        # of the effective request. Forced-stateless clients see the
        # resolver's handle-free copy here so local history remains eligible;
        # the provider wire view is rebuilt later from the original options.
        # handles the tool loop withheld are removed outright, because the
        # service transcript behind them still holds stripped or
        # never-answered calls — re-sending one would combine a poisoned
        # remote transcript with the local fallback replay. The same view
        # feeds the eager-shadow decision here and the history providers
        # below, so no consumer can mistake a withheld handle for a live one.
        reflected_options = storage_resolution.options_without_handles if force_stateless else effective_options
        if session is not None and session.invalidated_service_session_ids:
            reflected_options = strip_invalidated_conversation_handles(
                reflected_options, session.invalidated_service_session_ids
            )
        # Resolve the request-level continuation handles from that view
        # (:1199; a live session id still takes precedence below). Every
        # CONVERSATION_HANDLE_KEYS spelling counts, top-level and inside
        # extra_body — each one reaches the wire as service-side
        # continuation state, so any of them means the service owns history.
        request_handles = collect_conversation_handles(reflected_options)
        # Auto-inject local history when no loading HistoryProvider owns it.
        # Non-history and persist-only providers must not suppress prior turns.
        # NOTE: appends permanently to self.context_providers — agent state,
        # not run state (:1200-1209).
        if (
            session is not None
            and not any(
                provider.load_messages for provider in self.context_providers if isinstance(provider, HistoryProvider)
            )
            and not request_handles
        ):
            if service_stores_history:
                # Service-side storage keeps the transcript only behind the
                # service handle, and the exhaustion tail may have to discard
                # that handle. Shadow every turn locally from the start —
                # waiting until invalidation would permanently lose the turns
                # that succeeded through service storage before it. The
                # handle-gated fallback persists through the ordinary
                # after_run pass but loads only once the handle is gone, so
                # a live service conversation never sees duplicated history.
                self.context_providers.append(ServiceFallbackHistoryProvider())
            elif not session.service_session_id:
                self.context_providers.append(InMemoryHistoryProvider())

        active_session = session
        if active_session is None and self.context_providers:
            active_session = AgentSession()

        session_context, chat_options = await self._prepare_session_and_messages(
            session=active_session,
            input_messages=input_messages,
            # Providers reflect on the effective request, not the raw run
            # arguments: agent-default continuation handles must be visible
            # to history-provider load gates exactly like run-level ones,
            # and an invalidated handle must not look live to them — a
            # provider that trusts it would skip local replay while the
            # wire choke point strips the handle, sending the request with
            # neither remote nor local history. Hand over the sanitized
            # merged snapshot.
            options=reflected_options,
        )
        default_additional_args = chat_options.pop("additional_function_arguments", None)
        if isinstance(default_additional_args, Mapping):
            existing_additional_args = {
                **dict(default_additional_args),
                **existing_additional_args,
            }

        agent_name = self._get_agent_name()
        base_tools = normalize_tools(chat_options.pop("tools", None))

        # Resolve final tool list (configured + provider + runtime tools). The
        # framework's MCPTool branches (:1262-1281) are not ported: runtime
        # tools are flat FunctionTools in chrys, appended with the same
        # uniqueness rules.
        final_tools = list(base_tools)
        _append_unique_tools(final_tools, normalize_tools(tools_))

        effective_function_invocation_kwargs = (
            dict(function_invocation_kwargs) if function_invocation_kwargs is not None else {}
        )
        additional_function_arguments = {**effective_function_invocation_kwargs, **existing_additional_args}

        model = opts.pop("model", None)

        # Build options dict from run() options merged with provided options.
        # ORDER MATTERS for conversation_id: when a session exists, the key is
        # NOT popped from opts above, so an options-level conversation_id
        # overrides the session-derived value via the trailing **opts spread
        # (:1292-1294,:1310) — keep this line-for-line.
        run_opts: dict[str, Any] = {
            "conversation_id": active_session.service_session_id
            if active_session
            else opts.pop("conversation_id", None),
            "allow_multiple_tool_calls": opts.pop("allow_multiple_tool_calls", None),
            "frequency_penalty": opts.pop("frequency_penalty", None),
            "logit_bias": opts.pop("logit_bias", None),
            "max_tokens": opts.pop("max_tokens", None),
            "metadata": opts.pop("metadata", None),
            "presence_penalty": opts.pop("presence_penalty", None),
            "response_format": opts.pop("response_format", None),
            "seed": opts.pop("seed", None),
            "stop": opts.pop("stop", None),
            "store": opts.pop("store", None),
            "temperature": opts.pop("temperature", None),
            "tool_choice": opts.pop("tool_choice", None),
            "tools": final_tools or None,
            "top_p": opts.pop("top_p", None),
            "user": opts.pop("user", None),
            **opts,  # Remaining options are provider-specific
        }
        if model is not None:
            run_opts["model"] = model
        # _merge_options strips unset (None) options, so e.g. an unset `store`
        # is not forwarded and the service decides its own default (the
        # explicit pre-filter folded into the merge's final pass).
        co = _merge_options(chat_options, run_opts)
        # Single wire choke point for invalidated handles: run options, agent
        # defaults, and provider chat_options all land in the merged view, so
        # a caller repeating a handle the tool loop withheld is suppressed
        # here regardless of which layer supplied it — under every
        # CONVERSATION_HANDLE_KEYS spelling, top-level or nested in
        # extra_body (provider SDKs merge extra_body over the named
        # parameters, so a nested spelling reaches the wire all the same).
        # Suppression is removal-only: poisoned keys are dropped, never
        # rewritten, except that conversation_id falls back to the session's
        # own live handle when one exists — the fallback provider already
        # skipped its replay on seeing that handle, so bare removal would
        # send neither a handle nor local history.
        if active_session is not None and active_session.invalidated_service_session_ids:
            invalidated = active_session.invalidated_service_session_ids
            for key in CONVERSATION_HANDLE_KEYS:
                handle = conversation_handle_value(co.get(key))
                if handle is None or handle not in invalidated:
                    continue
                if key == "conversation_id":
                    live_handle = active_session.service_session_id
                    if (
                        isinstance(live_handle, str)
                        and live_handle
                        and not is_local_history_conversation_id(live_handle)
                        and live_handle not in invalidated
                    ):
                        co[key] = live_handle
                        continue
                co.pop(key)
            token_response_id = continuation_token_response_id(co.get("continuation_token"))
            if token_response_id is not None and token_response_id in invalidated:
                # The token would short-circuit the request into retrieving
                # the poisoned response outright — resurfacing the very
                # stripped calls the invalidation withheld, with the request
                # messages ignored. Removal-only: tokens are
                # response-specific, no fall-back applies.
                co.pop("continuation_token")
            extra_body = co.get("extra_body")
            if isinstance(extra_body, Mapping):
                poisoned_keys = [
                    key
                    for key in CONVERSATION_HANDLE_KEYS
                    if (nested := conversation_handle_value(extra_body.get(key))) is not None and nested in invalidated
                ]
                if poisoned_keys:
                    # Copy-on-write: the mapping may be a caller-owned object
                    # shared beyond this run.
                    clean_extra_body = dict(extra_body)
                    for key in poisoned_keys:
                        del clean_extra_body[key]
                    co["extra_body"] = clean_extra_body

        # Build session_messages from session context: context messages + input messages
        session_messages: list[Message] = session_context.get_messages(include_input=True)

        effective_client_kwargs = dict(client_kwargs) if client_kwargs is not None else {}
        if active_session is not None:
            effective_client_kwargs["session"] = active_session
        # Provider-contributed middleware (:1341-1361), bucketed through the
        # kernel split_middleware instead of the framework categorizer: chrys
        # providers contribute none today, non-kernel middleware raises a loud
        # TypeError, and the P5.4 session layer plugs in here.
        provider_middleware = session_context.get_middleware()
        if provider_middleware:
            provider_split = split_middleware(provider_middleware)
            provider_function_chat_middleware: list[Any] = [*provider_split.function, *provider_split.chat]
            if provider_function_chat_middleware:
                existing_middleware = effective_client_kwargs.get("middleware")
                if isinstance(existing_middleware, Sequence) and not isinstance(existing_middleware, (str, bytes)):
                    effective_client_kwargs["middleware"] = [
                        *existing_middleware,
                        *provider_function_chat_middleware,
                    ]
                elif existing_middleware is not None:
                    effective_client_kwargs["middleware"] = [
                        existing_middleware,
                        *provider_function_chat_middleware,
                    ]
                else:
                    effective_client_kwargs["middleware"] = provider_function_chat_middleware

        return {
            "session": active_session,
            "session_context": session_context,
            "input_messages": input_messages,
            "session_messages": session_messages,
            "agent_name": agent_name,
            "chat_options": co,
            "compaction_strategy": compaction_strategy or self.compaction_strategy,
            "tokenizer": tokenizer or self.tokenizer,
            "client_kwargs": effective_client_kwargs,
            "function_invocation_kwargs": additional_function_arguments,
        }

    async def _prepare_session_and_messages(
        self,
        *,
        session: AgentSession | None,
        input_messages: list[Message] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[SessionContext, dict[str, Any]]:
        """Mirror of ``:1419-1501`` minus the per-service-call skip branch."""
        # Shallow-copy the tools list, deepcopy every other default (:1441-1450).
        chat_options_tools: list[Any] | None = None
        if self.default_options:
            chat_options: dict[str, Any] = {}
            for key, value in self.default_options.items():
                if key == "tools":
                    chat_options_tools = list(value) if value else []
                    chat_options[key] = chat_options_tools
                else:
                    chat_options[key] = deepcopy(value)
        else:
            chat_options = {}

        provider_session = session
        if provider_session is None and self.context_providers:
            provider_session = AgentSession()

        session_context = SessionContext(
            session_id=provider_session.session_id if provider_session else None,
            service_session_id=provider_session.service_session_id if provider_session else None,
            input_messages=input_messages or [],
            options=options or {},
        )

        # Run before_run providers in forward order; HistoryProviders that opt
        # out of message loading are skipped (:1472-1484).
        for provider in self.context_providers:
            if isinstance(provider, HistoryProvider) and not provider.load_messages:
                continue
            if provider_session is None:
                raise RuntimeError("Provider session must be available when context providers are configured.")
            await provider.before_run(
                agent=self,  # type: ignore[arg-type]
                session=provider_session,
                context=session_context,
                state=provider_session.state.setdefault(provider.source_id, {}),
            )

        # Merge provider-contributed tools into chat_options (:1486-1491).
        if session_context.tools:
            if chat_options_tools is not None:
                chat_options_tools.extend(session_context.tools)
            else:
                chat_options["tools"] = list(session_context.tools)

        # Merge provider-contributed instructions into chat_options (:1493-1499).
        if session_context.instructions:
            combined_instructions = "\n".join(session_context.instructions)
            if "instructions" in chat_options:
                chat_options["instructions"] = f"{chat_options['instructions']}\n{combined_instructions}"
            else:
                chat_options["instructions"] = combined_instructions

        return session_context, chat_options

    async def _parse_non_streaming_response(
        self,
        context: _RunContext,
        response: ChatResponse,
    ) -> AgentResponse[Any]:
        """Mirror of ``:1023-1049``; ``suppress_response_id`` is gone so ``response_id`` passes through."""
        if not response:
            raise AgentInvalidResponseException("Chat client did not return a response.")

        for message in response.messages:
            if message.author_name is None:
                message.author_name = context["agent_name"]

        session = context["session"]
        if response._chrys_service_state_invalidated:
            # The tool loop withheld this run's continuation handles and
            # already cleared the session's service handle in place; the
            # discarded service transcript was the only history store, so
            # install the local fallback before the after-providers pass
            # persists this run's messages through it.
            self._install_local_history_fallback(session)
        if (
            session
            and response.conversation_id
            and not is_local_history_conversation_id(response.conversation_id)
            and session.service_session_id != response.conversation_id
        ):
            session.service_session_id = response.conversation_id

        agent_response = _build_agent_response_from_chat_response(
            response,
            response_format=context["chat_options"].get("response_format"),
        )
        session_context = context["session_context"]
        # Providers observe the exact object returned to the caller. Response-
        # level fields are shared run results and should be treated as read-only
        # unless a provider intentionally coordinates a mutation.
        session_context._response = agent_response
        await self._run_after_providers(session=session, context=session_context)
        return agent_response

    def _parse_streaming_response(
        self,
        context: _RunContext,
        stream_response: ResponseStream[ChatResponseUpdate, ChatResponse],
    ) -> ResponseStream[AgentResponseUpdate, AgentResponse[Any]]:
        """Mirror of ``:1051-1122`` minus the ``_suppress_response_id`` hook.

        Chain shape is load-bearing (#404 stream-final-state class of bugs):
        the returned object is ``loop-stream.map(...)`` with
        ``_propagate_conversation_id`` as a transform hook and ``_post_hook``
        as a result hook; the caller's ``run()`` wraps it in
        ``from_awaitable``. The transform's ``agent_name=self.name`` (may be
        None) vs the post-hook's backfill with ``_get_agent_name()`` (never
        None) is a deliberate framework asymmetry — keep both sources.
        """

        async def _post_hook(response: AgentResponse[Any]) -> None:
            # Update session with conversation_id derived from streaming raw
            # updates. Using response_id here can break function-call
            # continuation for APIs where response IDs are not valid
            # conversation handles (:1058-1083).
            conversation_id = self._extract_conversation_id_from_streaming_response(response)

            for message in response.messages:
                if message.author_name is None:
                    message.author_name = context["agent_name"]

            session = context["session"]
            if response._chrys_service_state_invalidated:
                # The tool loop withheld this run's continuation handles (its
                # exhaustion strip left the service-side transcript holding
                # unanswered calls); the raw updates still carry the stale id
                # and the eager transform already wrote it mid-stream, so
                # clear instead of restoring. Install the local fallback
                # first: the after-providers pass below persists this run's
                # messages through it, so the next run replays them instead
                # of starting from the bare new user message.
                self._install_local_history_fallback(session)
                if session is not None:
                    # The raw-scan id recovered above is the withheld handle;
                    # record it so a caller-supplied conversation_id repeating
                    # it gets suppressed instead of re-sent with the replay.
                    for handle in (conversation_id, session.service_session_id):
                        if isinstance(handle, str) and handle and not is_local_history_conversation_id(handle):
                            session.invalidated_service_session_ids.add(handle)
                    session.service_session_id = None
            elif (
                session
                and conversation_id
                and not is_local_history_conversation_id(conversation_id)
                and session.service_session_id != conversation_id
            ):
                session.service_session_id = conversation_id

            session_context = context["session_context"]
            # Intra-package private write: ``SessionContext.response`` stays a
            # read-only property for providers; the Agent finalizes through
            # the backing field like the framework did (:1079).
            session_context._response = response
            await self._run_after_providers(session=session, context=session_context)

        def _propagate_conversation_id(update: AgentResponseUpdate) -> AgentResponseUpdate:
            """Eagerly propagate conversation_id to the session as updates arrive (:1085-1099)."""
            session = context["session"]
            if session is None:
                return update
            raw = update.raw_representation
            if not isinstance(raw, ChatResponseUpdate):
                return update
            conversation_id = raw.conversation_id
            if (
                isinstance(conversation_id, str)
                and conversation_id
                and not is_local_history_conversation_id(conversation_id)
                # Exhaustion-tail updates defer to the tail's own verdict
                # (post-hook restore or invalidation): mirroring them eagerly
                # would survive a mid-tail abandonment as a stale pointer to a
                # service transcript holding stripped, never-answered calls.
                and not raw.__dict__.get("_chrys_exhaustion_tail_update", False)
                and session.service_session_id != conversation_id
            ):
                session.service_session_id = conversation_id
            return update

        async def _finalizer(updates: Sequence[AgentResponseUpdate]) -> AgentResponse[Any]:
            # Inlines the framework's _finalize_response_updates indirection
            # (:1106-1134) — no chrys caller needs the separate method.  The
            # mapped updates span every tool-loop iteration, so copy the
            # inner loop's normalized aggregate and final-call usage instead
            # of re-summing raw cumulative usage snapshots here.
            response = AgentResponse.from_updates(
                updates,
                output_format_type=context["chat_options"].get("response_format"),
            )
            chat_response = await stream_response.get_final_response()
            response.usage_details = chat_response.usage_details
            response.latest_usage_details = chat_response.latest_usage_details
            # Chrys divergence (single reconstruction authority): reuse the
            # tool loop's assembled Message objects instead of the local
            # ``from_updates`` re-merge, which would rebuild multi-fragment
            # function calls as new Content objects and drop kernel-stamped
            # call provenance and folded result metadata. Element identity is
            # preserved; the fresh list container means an after-provider
            # mutating this response's list cannot corrupt the loop's.
            response.messages = list(chat_response.messages)
            if chat_response._chrys_service_state_invalidated:
                # ``AgentResponse.from_updates`` restored ``response_id`` from
                # the mapped updates even though the loop's finalizer withheld
                # it; mirror the invalidation so the post-hook and the caller
                # see no stale continuation handle.
                response.response_id = None
                response._chrys_service_state_invalidated = True
            return response

        stream = stream_response.map(
            transform=partial(
                map_chat_to_agent_update,
                agent_name=self.name,
            ),
            finalizer=_finalizer,
        )
        return stream.with_transform_hook(_propagate_conversation_id).with_result_hook(_post_hook)

    @staticmethod
    def _extract_conversation_id_from_streaming_response(
        response: AgentResponse[Any],
    ) -> str | None:
        """Mirror of ``:1136-1158``."""
        raw = response.raw_representation
        if raw is None:
            return None

        raw_items: list[Any] = list(raw) if isinstance(raw, list) else [raw]
        for item in reversed(raw_items):
            if isinstance(item, Mapping):
                value = item.get("conversation_id")
                if isinstance(value, str) and value:
                    return value
                continue

            value = getattr(item, "conversation_id", None)
            if isinstance(value, str) and value:
                return value

        return None

    def _install_local_history_fallback(self, session: AgentSession | None) -> None:
        """Give an invalidated service-stored session a local history source.

        Safety net behind the ``_prepare_run_context`` shadow install: any
        service-stored run normally carries the handle-gated fallback from
        run start, but if an invalidation arrives on a run whose preparation
        did not select it, installing before the after-providers pass still
        lets the standard ``after_run`` persist this run's input/output for
        replay once no live handle exists. Sessions that already have a
        loading HistoryProvider (including a previously installed fallback)
        keep their existing history owner.
        """
        if session is None:
            return
        if any(provider.load_messages for provider in self.context_providers if isinstance(provider, HistoryProvider)):
            return
        self.context_providers.append(ServiceFallbackHistoryProvider())

    async def _run_after_providers(
        self,
        *,
        session: AgentSession | None,
        context: SessionContext,
    ) -> None:
        """Mirror of ``BaseAgent._run_after_providers`` (``:451-483``) minus the per-service-call skip."""
        provider_session = session
        if provider_session is None and self.context_providers:
            provider_session = AgentSession()

        for provider in reversed(self.context_providers):
            if provider_session is None:
                raise RuntimeError("Provider session must be available when context providers are configured.")
            await provider.after_run(
                agent=self,  # type: ignore[arg-type]
                session=provider_session,
                context=context,
                state=provider_session.state.setdefault(provider.source_id, {}),
            )


class Agent(AgentTelemetryLayer, _AgentCore):
    """Chrys production agent: chrys-owned OTel span layer over the own core.

    Layering mirror of the framework production class (``Agent =
    AgentMiddlewareLayer -> AgentTelemetryLayer -> RawAgent``,
    ``_agents.py:1635-1763``) with the agent-middleware pipeline deleted:
    chrys has zero agent-bucket middleware, so this ``run()`` keeps only the
    bucket-and-forward half of ``AgentMiddlewareLayer.run``
    (``_middleware.py:1336-1385``), re-expressed through the kernel
    :func:`~chrys.kernel.middleware.split_middleware`.

    Telemetry: the chrys :class:`~.instrumentation.AgentTelemetryLayer` confines the
    non-streaming INNER ContextVars to the returned coroutine (P5.5 fix), so
    the awaitable may be created in one task and driven in another — both the
    executor's create-task-then-call pattern and the sub-agent controller's
    ``create_task(run(...))`` are safe under ``CHRYS_OTEL=1``. Streaming keeps
    the framework shape (set in the sync call, reset in the stream finalizer)
    with the finalizer hardened against cross-context resets.
    """

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
        """Bucket ctor+run middleware into ``client_kwargs`` and delegate.

        Mirror of the function/chat half of ``AgentMiddlewareLayer.run``
        (``_middleware.py:1351-1385``): ``self.middleware`` is re-read on
        every call (dynamic-change support), combined order is
        ``[ctor-function, ctor-chat, run-function, run-chat]``, and a
        non-empty combination *overwrites* ``client_kwargs["middleware"]``
        (:1414-1416). The ``middleware`` kwarg itself is NOT forwarded —
        ``AgentTelemetryLayer.run`` would pass it through to ``_AgentCore.run``
        (``observability.py:1963-1964``), which mirrors ``RawAgent`` and takes
        no such parameter.
        """
        ctor_split = split_middleware(self.middleware)
        run_split = split_middleware(middleware)
        combined_function_chat_middleware: list[Any] = [
            *ctor_split.function,
            *ctor_split.chat,
            *run_split.function,
            *run_split.chat,
        ]
        effective_client_kwargs = dict(client_kwargs) if client_kwargs is not None else {}
        if combined_function_chat_middleware:
            effective_client_kwargs["middleware"] = combined_function_chat_middleware
        effective_function_invocation_kwargs = (
            dict(function_invocation_kwargs) if function_invocation_kwargs is not None else {}
        )
        return super().run(
            messages,
            stream=stream,
            session=session,
            tools=tools,
            options=options,
            compaction_strategy=compaction_strategy,
            tokenizer=tokenizer,
            function_invocation_kwargs=effective_function_invocation_kwargs,
            client_kwargs=effective_client_kwargs,
        )
