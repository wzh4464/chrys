# Copyright (c) 2026 Chrys. All rights reserved.

"""Chrys-owned middleware foundation.

Own middleware primitives: the ``ChatMiddleware``/``FunctionMiddleware`` ABCs,
``ChatContext``/``FunctionInvocationContext``, ``MiddlewareTermination``, the
two execution pipelines, and a composition-style ``ChatMiddlewareLayer``.

Since P5.2 production executes these classes inside the chrys-owned
``ToolLoopLayer`` (``loop.py``); since P5.3 the chrys-owned ``Agent``
(``agent.py``) routes ctor/run middleware through :func:`split_middleware`,
so the module is fully de-bridged: ``MiddlewareTermination`` is a plain
``Exception`` (the only catcher is the own loop) and the two ABCs are plain
``ABC`` (no framework ``categorize_middleware`` consumes chrys middleware
anywhere anymore). The context classes are pure-owned and mirror the
framework constructors field-for-field so the run-contract parity tests stay
assertable against upstream.

Deliberately not ported: ``AgentMiddleware``/``AgentContext`` (chrys has no
agent middleware; P5.3 defines what the own Agent needs), the middleware
function decorators, and ``_determine_middleware_type`` signature sniffing
(chrys middleware is class-based only; plain callables are rejected loudly).

HARD RULE: kernel modules may import only the stdlib, intra-package modules,
allowed third-party packages, and downward ``chrys.foundation.*`` modules.
Sibling or upward ``chrys.*`` imports remain forbidden:
``chrys.service.agent_middleware`` imports this module, so any reverse import is
a cycle.
"""

from __future__ import annotations

import contextlib
import inspect
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeIs, cast

# ResponseStream is a keeper (`_types` stays framework by plan). The two
# middleware-ABC bridge imports were removed at P5.3: with the chrys Agent
# routing middleware via split_middleware, no framework
# ``categorize_middleware`` ever sees a chrys middleware instance again.
# (``MiddlewareTermination`` was de-bridged earlier, at P5.2.)
from ._types import ResponseStream
from .client import _PreparedRequestObserverClient

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from pydantic import BaseModel

    from ._types import ChatResponse, ChatResponseUpdate, Message, ToolTypes
    from .sessions import AgentSession
    from .tools import FunctionTool


def _is_chat_response_stream(
    value: object,
) -> TypeIs[ResponseStream[ChatResponseUpdate, ChatResponse]]:
    """Narrow a stream held in the chat-context result union."""
    return isinstance(value, ResponseStream)


class MiddlewareTermination(Exception):
    """Control-flow exception that terminates middleware execution early.

    ``result`` is consumed by whichever layer catches the termination: the chat
    pipeline suppresses the exception and returns ``context.result``; the
    function pipeline lets it propagate so the tool loop can stop and read
    ``exc.result`` (framework consumption pattern: ``_tools.py:1610-1623``,
    mirrored by ``loop.py``).
    """

    result: Any

    def __init__(self, message: str = "Middleware terminated execution.", *, result: Any = None) -> None:
        super().__init__(message)
        self.result = result


class FunctionInvocationContext:
    """Context for one tool invocation through the function middleware pipeline.

    Field-for-field mirror of the framework constructor
    (``_middleware.py:256-285``): ``metadata`` and ``kwargs`` are stored as
    fresh dict copies of the given mappings; ``tools`` stores the live list it
    is given (aliased on purpose — mutations through :meth:`add_tools` and
    :meth:`remove_tools` are how progressive tool exposure works in the loop).

    Chrys middleware consumes: ``function.name``, the out-of-band chrys kind
    (``function.chrys_kind`` via ``chrys.foundation.tool_kinds.get_tool_kind``),
    ``arguments``, ``metadata``, ``result``.
    """

    def __init__(
        self,
        function: FunctionTool,
        arguments: BaseModel | Mapping[str, Any],
        session: AgentSession | None = None,
        metadata: Mapping[str, Any] | None = None,
        result: Any = None,
        kwargs: Mapping[str, Any] | None = None,
        tools: list[ToolTypes] | None = None,
    ) -> None:
        self.function = function
        self.arguments = arguments
        self.session = session
        self.metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        self.result = result
        self.kwargs: dict[str, Any] = dict(kwargs) if kwargs is not None else {}
        self.tools = tools
        # Chrys extension (not part of the framework mirror): how many calls in
        # the current gather-batch target the same tool. The tool loop stamps
        # the real per-name count before execution; the default keeps direct /
        # test invocations behaving as singletons. Middleware enforcing
        # singleton semantics (whole-list replacement tools) rejects when > 1.
        self.same_tool_calls_in_batch: int = 1

    def add_tools(
        self,
        tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]],
    ) -> None:
        """Add tools to the current run for the next model iteration.

        The complete input batch is normalized and duplicate-checked against a
        copy before the aliased live list is mutated. Re-adding the same object
        is a no-op; a different object with the same name raises ``ValueError``.

        Raises:
            RuntimeError: If this invocation is not bound to a live tool loop.
            ValueError: If the batch introduces a conflicting duplicate name.
        """
        from .tools import _append_unique_tools, normalize_tools

        if self.tools is None:
            raise RuntimeError(
                "Cannot add tools: this FunctionInvocationContext is not bound to a live "
                "agent run. add_tools is only available for functions invoked within an "
                "agent's function-calling loop."
            )
        merged = _append_unique_tools(list(self.tools), normalize_tools(tools))
        self.tools[:] = merged

    def remove_tools(
        self,
        tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | str | Sequence[str],
    ) -> None:
        """Remove tools from the current run for the next model iteration.

        String inputs remove every tool with that name. Tool objects and
        original callables remove their exact normalized live instances, so an
        unrelated tool that shares a name is preserved. Missing and nameless
        tools are ignored.

        Raises:
            RuntimeError: If this invocation is not bound to a live tool loop.
        """
        from .tools import FunctionTool, _get_tool_name, normalize_tools

        if self.tools is None:
            raise RuntimeError(
                "Cannot remove tools: this FunctionInvocationContext is not bound to a live "
                "agent run. remove_tools is only available for functions invoked within an "
                "agent's function-calling loop."
            )

        raw_items: list[Any]
        if isinstance(tools, str):
            raw_items = [tools]
        elif isinstance(tools, Sequence) and not isinstance(tools, (bytes, bytearray)):
            raw_items = list(tools)
        else:
            raw_items = [tools]

        names_to_remove: set[str] = set()
        instances_to_remove: set[int] = set()
        callables_to_remove: set[int] = set()
        for tool_item in raw_items:
            if isinstance(tool_item, str):
                names_to_remove.add(tool_item)
                continue
            if isinstance(tool_item, FunctionTool):
                instances_to_remove.add(id(tool_item))
                continue
            if callable(tool_item):
                callables_to_remove.add(id(tool_item))
                continue
            for normalized_tool in normalize_tools(tool_item):
                instances_to_remove.add(id(normalized_tool))

        if names_to_remove or instances_to_remove or callables_to_remove:
            self.tools[:] = [
                tool_item
                for tool_item in self.tools
                if id(tool_item) not in instances_to_remove
                and _get_tool_name(tool_item) not in names_to_remove
                and not (
                    isinstance(tool_item, FunctionTool)
                    and tool_item.func is not None
                    and id(tool_item.func) in callables_to_remove
                )
            ]


class ChatContext:
    """Context for one model call through the chat middleware pipeline.

    Field-for-field mirror of the framework constructor
    (``_middleware.py:422-466``); ``client`` is typed ``Any`` because chrys
    middleware never reads it and tests pass placeholders.

    ``messages`` copy contract — "调用前可变换的副本" (a mutable-before-call
    copy): ``ChatMiddlewareLayer`` builds this context with
    ``messages=list(messages)`` — a fresh list per call that shares the
    caller's Message objects. Middleware may replace elements in place or
    rebind the attribute to a new list; the post-pipeline value of
    ``context.messages`` is what reaches the inner client, and mutations never
    leak into the caller's list or into later calls. The copy is the *layer's*
    contract — this constructor stores exactly what it is given.
    """

    def __init__(
        self,
        client: Any,
        messages: Sequence[Message],
        options: Mapping[str, Any] | None,
        stream: bool = False,
        metadata: Mapping[str, Any] | None = None,
        result: ChatResponse | ResponseStream[ChatResponseUpdate, ChatResponse] | None = None,
        kwargs: Mapping[str, Any] | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        stream_transform_hooks: Sequence[
            Callable[[ChatResponseUpdate], ChatResponseUpdate | Awaitable[ChatResponseUpdate]]
        ]
        | None = None,
        stream_result_hooks: Sequence[Callable[[ChatResponse], ChatResponse | Awaitable[ChatResponse]]] | None = None,
        stream_cleanup_hooks: Sequence[Callable[[], Awaitable[None] | None]] | None = None,
        stream_update_filters: Sequence[Callable[[ChatResponseUpdate], ChatResponseUpdate]] | None = None,
        request_message_observers: Sequence[Callable[[Sequence[Message]], None]] | None = None,
    ) -> None:
        self.client = client
        self.messages = messages
        self.options = options
        self.stream = stream
        self.metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        self.result = result
        self.kwargs: dict[str, Any] = dict(kwargs) if kwargs is not None else {}
        self.function_invocation_kwargs: dict[str, Any] = (
            dict(function_invocation_kwargs) if function_invocation_kwargs is not None else {}
        )
        self.stream_transform_hooks = list(stream_transform_hooks or [])
        self.stream_result_hooks = list(stream_result_hooks or [])
        self.stream_cleanup_hooks = list(stream_cleanup_hooks or [])
        # Unlike the stream_* hooks above (attached to the FINAL result stream
        # after the chain), update filters attach to the stream the final
        # handler resolves — beneath every middleware — so a middleware that
        # drains the inner stream and re-streams it through a proxy (response
        # validation) still consumes filtered updates, and a retry's fresh
        # inner stream is filtered again automatically. Middleware may append
        # here before calling ``call_next()``.
        self.stream_update_filters = list(stream_update_filters or [])
        # Request observers run at the final-handler boundary after every
        # middleware has transformed ``messages`` and immediately before the
        # inner client sees them. They are chrys-owned request-path plumbing,
        # not middleware hooks, and therefore are not forwarded on the wire.
        self.request_message_observers = list(request_message_observers or [])


class ChatMiddleware(ABC):
    """Intercepts one model call.

    Contract (all data flows through the context; ``call_next`` takes no
    arguments and returns nothing):

    - ``await call_next()`` runs the rest of the chain plus the inner client
      and populates ``context.result``; post-processing after it sees the
      actual result.
    - Setting ``context.result`` and *not* calling ``call_next()`` overrides
      execution (the inner client is never reached).
    - Raising :class:`MiddlewareTermination` stops the chain; the chat
      pipeline suppresses it and returns whatever ``context.result`` holds.
    """

    @abstractmethod
    async def process(self, context: ChatContext, call_next: Callable[[], Awaitable[None]]) -> None:
        """Process one chat request; mutate ``context`` rather than returning."""


class FunctionMiddleware(ABC):
    """Intercepts one tool invocation.

    Same contract shape as :class:`ChatMiddleware`, with one difference:
    :class:`MiddlewareTermination` raised here *propagates* out of the function
    pipeline so the tool loop can terminate and read ``exc.result``
    (``loop.py``, mirroring ``_tools.py:1609-1623``).
    """

    @abstractmethod
    async def process(self, context: FunctionInvocationContext, call_next: Callable[[], Awaitable[None]]) -> None:
        """Process one tool invocation; mutate ``context`` rather than returning."""


class _EmptyAsyncIterator:
    """Async iterator yielding nothing (own copy of ``_middleware.py:53-69``)."""

    def __aiter__(self) -> _EmptyAsyncIterator:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration


class ChatMiddlewarePipeline:
    """Executes chat middleware as a nested chain; first in list = outermost.

    Mirror of ``_middleware.py:1007-1096``. Only :class:`ChatMiddleware`
    instances are accepted — plain callables raise ``TypeError`` (no decorator
    or signature-sniffing support by design).
    """

    def __init__(self, *middleware: ChatMiddleware) -> None:
        for item in middleware:
            if not isinstance(item, ChatMiddleware):
                raise TypeError(
                    f"chat middleware must subclass chrys ChatMiddleware (plain callables are not supported), got {item!r}"
                )
        self._middleware: list[ChatMiddleware] = list(middleware)

    @property
    def has_middlewares(self) -> bool:
        return bool(self._middleware)

    async def execute(
        self,
        context: ChatContext,
        final_handler: Callable[
            [ChatContext], Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]
        ],
    ) -> ChatResponse | ResponseStream[ChatResponseUpdate, ChatResponse] | None:
        def observe_request_messages() -> None:
            # Run immediately before every final-handler invocation. A
            # middleware may call ``call_next`` repeatedly (for example a
            # validation retry), and each resulting wire request must be
            # observed independently.
            for observer in context.request_message_observers:
                observer(context.messages)

        def attach_update_filters() -> None:
            # At the final-handler boundary — beneath every middleware — so
            # semantic stream proxies (a middleware draining the inner stream
            # and replaying it through its own generator) consume filtered
            # updates, and every retry's fresh inner stream is covered.
            result = context.result
            if context.stream and context.stream_update_filters and _is_chat_response_stream(result):
                for update_filter in context.stream_update_filters:
                    result.with_update_filter(update_filter)

        if not self._middleware:
            # Fast path parity (:1056-1067): resolve, store, validate stream shape.
            observe_request_messages()
            raw = final_handler(context)
            resolved = (await raw) if inspect.isawaitable(raw) else raw
            context.result = resolved
            if context.stream and not isinstance(resolved, ResponseStream):
                raise ValueError("Streaming agent middleware requires a ResponseStream result.")
            attach_update_filters()
            return resolved

        def create_next_handler(index: int) -> Callable[[], Awaitable[None]]:
            if index >= len(self._middleware):

                async def final_wrapper() -> None:
                    observe_request_messages()
                    raw = final_handler(context)
                    context.result = (await raw) if inspect.isawaitable(raw) else raw
                    attach_update_filters()

                return final_wrapper

            async def current_handler() -> None:
                await self._middleware[index].process(context, create_next_handler(index + 1))

            return current_handler

        # Termination is suppressed here (parity with :1086-1087): the chain stops
        # and whatever context.result holds is the outcome.
        with contextlib.suppress(MiddlewareTermination):
            await create_next_handler(0)()

        # Parity with :1089-1096, including the truthiness (not `is not None`) check.
        result = context.result
        if result and _is_chat_response_stream(result):
            for transform_hook in context.stream_transform_hooks:
                result.with_transform_hook(transform_hook)
            for result_hook in context.stream_result_hooks:
                result.with_result_hook(result_hook)
            for cleanup_hook in context.stream_cleanup_hooks:
                result.with_cleanup_hook(cleanup_hook)
        return context.result


class FunctionMiddlewarePipeline:
    """Executes function middleware as a nested chain; first in list = outermost.

    Mirror of ``_middleware.py:934-1004``. Unlike the chat pipeline,
    :class:`MiddlewareTermination` is NOT suppressed — it propagates to the
    caller (the tool loop) to signal loop termination.
    """

    def __init__(self, *middleware: FunctionMiddleware) -> None:
        for item in middleware:
            if not isinstance(item, FunctionMiddleware):
                raise TypeError(
                    f"function middleware must subclass chrys FunctionMiddleware (plain callables are not supported), got {item!r}"
                )
        self._middleware: list[FunctionMiddleware] = list(middleware)

    @property
    def has_middlewares(self) -> bool:
        return bool(self._middleware)

    async def execute(
        self,
        context: FunctionInvocationContext,
        final_handler: Callable[[FunctionInvocationContext], Awaitable[Any]],
    ) -> Any:
        if not self._middleware:
            # Fast path parity (:981-982): returns the raw result and deliberately
            # does NOT populate context.result — a framework asymmetry we keep.
            return await final_handler(context)

        def create_next_handler(index: int) -> Callable[[], Awaitable[None]]:
            if index >= len(self._middleware):

                async def final_wrapper() -> None:
                    context.result = final_handler(context)
                    if inspect.isawaitable(context.result):
                        context.result = await context.result

                return final_wrapper

            async def current_handler() -> None:
                await self._middleware[index].process(context, create_next_handler(index + 1))

            return current_handler

        # No suppression (parity with :1001-1002): MiddlewareTermination propagates.
        await create_next_handler(0)()
        return context.result


class MiddlewareSplit(NamedTuple):
    """Result of :func:`split_middleware`."""

    chat: list[ChatMiddleware]
    function: list[FunctionMiddleware]


def _as_middleware_list(
    source: ChatMiddleware | FunctionMiddleware | Sequence[ChatMiddleware | FunctionMiddleware] | None,
) -> list[ChatMiddleware | FunctionMiddleware]:
    """Normalize one source, treating only ``None`` as empty."""
    if source is None:
        return []
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        return list(cast("Sequence[ChatMiddleware | FunctionMiddleware]", source))
    return [cast("ChatMiddleware | FunctionMiddleware", source)]


def split_middleware(
    *sources: ChatMiddleware | FunctionMiddleware | Sequence[ChatMiddleware | FunctionMiddleware] | None,
) -> MiddlewareSplit:
    """Flatten and bucket middleware sources into chat vs function lists.

    Own replacement for the framework's ``categorize_middleware``
    (``_middleware.py:1517-1560``) minus the agent bucket and the callable
    signature sniffing: sources may be ``None`` (skipped), single instances, or
    sequences (flattened); anything that is not a chrys middleware instance
    raises ``TypeError``. Relative order is preserved within each bucket;
    FunctionMiddleware is checked first (framework precedence).
    """
    chat: list[ChatMiddleware] = []
    function: list[FunctionMiddleware] = []
    flattened: list[Any] = []
    for source in sources:
        flattened.extend(_as_middleware_list(source))
    for item in flattened:
        if isinstance(item, FunctionMiddleware):
            function.append(item)
        elif isinstance(item, ChatMiddleware):
            chat.append(item)
        else:
            raise TypeError(
                f"middleware must be a chrys ChatMiddleware or FunctionMiddleware instance (plain callables are not supported), got {item!r}"
            )
    return MiddlewareSplit(chat=chat, function=function)


class SupportsGetResponse(Protocol):
    """Minimal inner-client contract :class:`ChatMiddlewareLayer` wraps.

    - ``stream=False``: returns an awaitable resolving to a ``ChatResponse``.
    - ``stream=True``: MUST return a ``ResponseStream`` synchronously, not a
      coroutine — the layer never awaits at call time (the framework pins the
      same shape at ``_clients.py:423``; see run-contract §3).
    - Extra keyword arguments must be accepted and forwarded as appropriate.
    """

    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = ...,
        options: Mapping[str, Any] | None = ...,
        function_invocation_kwargs: Mapping[str, Any] | None = ...,
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]: ...


class ChatMiddlewareLayer:
    """Composition-style chat middleware layer over an inner client.

    Mirror of the framework mixin layer (``_middleware.py:1108-1255``) with two
    deliberate deltas: composition instead of cooperative-MRO mixing, and
    per-call middleware as an explicit ``middleware=`` kwarg instead of the
    framework's ``client_kwargs["middleware"]`` smuggling (equivalent — in both
    designs the key never lands in ``context.kwargs``). The
    ``compaction_strategy``/``tokenizer`` plumbing is not ported; that belongs
    to the loop/agent layers (P5.2/P5.3). Transparent extras still flow
    ``**kwargs -> context.kwargs -> inner client``.

    ``context.messages`` copy contract: when middleware is present, the context
    gets a fresh ``list(messages)`` per call sharing the caller's Message
    objects, and ``_middleware_handler`` re-reads ``context.messages`` after
    the pipeline ran — element swaps and whole-list rebinds reach the wire and
    never leak across calls. The no-middleware fast path forwards the caller's
    sequence without copying (framework parity, :1194-1203).
    """

    def __init__(self, inner: SupportsGetResponse, *, middleware: Sequence[ChatMiddleware] | None = None) -> None:
        self.inner = inner
        self.chat_middleware: list[ChatMiddleware] = list(middleware) if middleware else []

    def __getattr__(self, name: str) -> Any:
        # Delegate unknown attributes to the inner client so agent-facing client
        # surface (model, STORES_BY_DEFAULT, ...) resolves through the stack.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)

    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        options: Mapping[str, Any] | None = None,
        middleware: Sequence[ChatMiddleware] | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        stream_update_filter: Callable[[ChatResponseUpdate], ChatResponseUpdate] | None = None,
        request_message_observer: Callable[[Sequence[Message]], None] | None = None,
        **kwargs: Any,
    ) -> Awaitable[ChatResponse | None] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        # Constructor middleware first = outermost (parity with :1125). Pipelines
        # are stateless, so rebuilding per call is behaviorally identical to the
        # framework's tuple-equality cache (:1121-1132); add caching only if
        # profiling ever demands it.
        #
        # ``stream_update_filter`` and ``request_message_observer`` are chrys
        # extensions consumed here (like ``middleware=``, neither lands in
        # ``context.kwargs``). The filter reaches the stream the FINAL handler
        # resolves, beneath every middleware — the only delivery path that
        # crosses semantic stream proxies (a middleware draining the inner
        # stream and replaying it), which ``ResponseStream.with_update_filter``
        # push-down cannot. The observer sees the corresponding post-middleware
        # request immediately before the final handler runs.
        pipeline = ChatMiddlewarePipeline(*self.chat_middleware, *(middleware or []))
        if not pipeline.has_middlewares:
            if request_message_observer is not None:
                request_message_observer(messages)
            prepared_observer_kwargs = (
                {"request_message_observer": request_message_observer}
                if request_message_observer is not None and isinstance(self.inner, _PreparedRequestObserverClient)
                else {}
            )
            result = self.inner.get_response(
                messages,
                stream=stream,
                options=options,
                function_invocation_kwargs=function_invocation_kwargs,
                **prepared_observer_kwargs,
                **kwargs,
            )
            if stream and stream_update_filter is not None and _is_chat_response_stream(result):
                result.with_update_filter(stream_update_filter)
            return result

        context = ChatContext(
            client=self,
            messages=list(messages),  # the fresh-copy contract point (:1207)
            options=options,
            stream=stream,
            kwargs=dict(kwargs),
            function_invocation_kwargs=function_invocation_kwargs,
            stream_update_filters=[stream_update_filter] if stream_update_filter is not None else None,
            request_message_observers=[request_message_observer] if request_message_observer is not None else None,
        )

        async def _execute() -> Any:
            return await pipeline.execute(context=context, final_handler=self._middleware_handler)

        if not stream:
            return _execute()  # un-awaited coroutine; nothing runs at call time

        async def _execute_stream() -> ResponseStream[ChatResponseUpdate, ChatResponse]:
            result = await _execute()
            if result is None:
                # Middleware terminated without setting a result (:1225-1226).
                return ResponseStream(_EmptyAsyncIterator())
            if _is_chat_response_stream(result):
                return result
            raise ValueError("Expected ResponseStream for streaming, got ChatResponse")

        # from_awaitable stores the coroutine and resolves it on first consumption
        # (_types.py:3054-3079) — construction runs zero middleware.
        return ResponseStream.from_awaitable(_execute_stream())

    def _middleware_handler(
        self, context: ChatContext
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        # Re-reads the context AFTER the pipeline ran — this is what makes element
        # swaps and whole-list rebinds of context.messages reach the wire
        # (parity with :1240-1255, including `options or {}`).
        prepared_observer_kwargs: dict[str, Any] = {}
        if context.request_message_observers and isinstance(self.inner, _PreparedRequestObserverClient):

            def observe_prepared_request(messages: Sequence[Message]) -> None:
                for observer in context.request_message_observers:
                    observer(messages)

            prepared_observer_kwargs["request_message_observer"] = observe_prepared_request
        return self.inner.get_response(
            context.messages,
            stream=context.stream,
            options=context.options or {},
            function_invocation_kwargs=context.function_invocation_kwargs,
            **prepared_observer_kwargs,
            **context.kwargs,
        )
