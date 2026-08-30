# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for MCP adapter — unit tests without actual MCP servers."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import sys
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest

import chrys.service.mcp.owned as owned_mcp
from chrys.foundation.events.types import Warning as WarningEvent
from chrys.foundation.platform import runtime_paths
from chrys.foundation.tool_kinds import get_tool_kind
from chrys.foundation.util.env_templates import EnvVarResolutionError
from chrys.kernel import ChatResponse, Message
from chrys.kernel import FunctionTool as ChrysFunctionTool
from chrys.kernel.middleware import FunctionInvocationContext
from chrys.service.mcp.adapter import (
    _SEVERED_ENTITY_TAIL_RE,
    DEFAULT_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_TEST_TIMEOUT_SECONDS,
    MAX_STDIO_PATH_DISPLAY_CHARS,
    MAX_STDIO_STDERR_BYTES_CAPTURED,
    MAX_STDIO_STDERR_PREVIEW_LINES,
    MCP_INSTRUCTIONS_CHAR_LIMIT,
    MCPAdapter,
    MCPConnectionError,
    MCPToolNameCollisionError,
    MCPToolNameValidationError,
    _chrys_streamable_http_client,
    _create_mcp_tool,
    _display_path,
    _HTTPMCPTool,
    _inherited_stdio_environment,
    _resolve_spawn_executable,
    _SafeStdioTool,
    _StdioProcessDiagnostics,
    _validate_config,
    tolerant_stdio_client,
)
from chrys.service.mcp.cache import MCPConnectionLease
from chrys.service.mcp.owned import (
    _MCP_FRAMEWORK_DENYLIST,
    _MCP_NORMALIZED_NAME_KEY,
    _MCP_REMOTE_NAME_KEY,
    MCPStdioTool,
    MCPStreamableHTTPTool,
    MCPTool,
    _mcp_call_headers,
)
from chrys.service.profiles.agents.schema import MCPServerConfig


async def _remote_tool(ctx, value: str = "") -> str:
    return value or "ok"


def _function_tool(name: str = "remote") -> ChrysFunctionTool:
    return ChrysFunctionTool(
        func=_remote_tool,
        name=name,
        description="Remote tool",
        input_model={
            "type": "object",
            "properties": {"value": {"type": "string"}},
        },
    )


def _mcp_remote_tool(
    name: str,
    *,
    meta: dict[str, Any] | None = None,
    input_schema: dict[str, Any] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description="Remote tool",
        inputSchema=input_schema if input_schema is not None else {"type": "object", "properties": {}},
        meta=meta,
        execution=None,
    )


@pytest.mark.parametrize("value", [-1, 1, 99, True, False, "100", 1.5])
def test_adapter_rejects_invalid_max_tool_result_tokens(value: Any) -> None:
    config = MCPServerConfig(name="srv", transport="stdio", command="python")
    config.max_tool_result_tokens = value

    with pytest.raises(ValueError, match="max_tool_result_tokens"):
        _validate_config(config)


@pytest.mark.parametrize("value", [None, 0, 100, 1234])
def test_adapter_accepts_valid_max_tool_result_tokens(value: int | None) -> None:
    config = MCPServerConfig(
        name="srv",
        transport="stdio",
        command="python",
        max_tool_result_tokens=value,
    )

    _validate_config(config)


def _mcp_remote_prompt(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, description="Remote prompt", arguments=[])


async def _load_fake_remote_tools(tool: MCPTool, *remote_tools: SimpleNamespace) -> None:
    tool.session = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=list(remote_tools), nextCursor=None))
    )
    tool._ensure_connected = AsyncMock()  # type: ignore[method-assign]
    await tool.load_tools()


async def _load_calling_remote_tools(tool: MCPTool, *remote_tools: SimpleNamespace) -> SimpleNamespace:
    from mcp import types

    session = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=list(remote_tools), nextCursor=None)),
        call_tool=AsyncMock(
            return_value=types.CallToolResult(
                content=[types.TextContent(type="text", text="ok")],
                isError=False,
            )
        ),
    )
    tool.session = session
    tool._ensure_connected = AsyncMock()  # type: ignore[method-assign]
    await tool.load_tools()
    return session


def _sampling_params(max_tokens: int = 9999) -> Any:
    from mcp import types

    return types.CreateMessageRequestParams(
        messages=[
            types.SamplingMessage(
                role="user",
                content=types.TextContent(type="text", text="hi"),
            )
        ],
        maxTokens=max_tokens,
    )


def _tool_list_changed_notification() -> Any:
    from mcp import types

    return types.ServerNotification(root=types.ToolListChangedNotification())


class _SamplingClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def get_response(self, messages: Any, options: Any = None) -> ChatResponse:
        self.calls.append({"messages": messages, "options": dict(options or {})})
        return ChatResponse(messages=[Message(role="assistant", contents=["sampled"])], model="m-test")


def test_adapter_init() -> None:
    adapter = MCPAdapter()
    assert adapter.server_names == []
    assert adapter.failures == {}


async def test_adapter_disconnect_nonexistent() -> None:
    """Disconnecting a non-connected server should not raise."""
    adapter = MCPAdapter()
    await adapter.disconnect("nonexistent")
    assert adapter.server_names == []


# ---------------------------------------------------------------------------
# _create_mcp_tool
# ---------------------------------------------------------------------------


def test_create_stdio_tool() -> None:
    config = MCPServerConfig(name="s", transport="stdio", command="python", args=["-m", "srv"], env={"K": "V"})
    tool = _create_mcp_tool(config)
    assert isinstance(tool, _SafeStdioTool)
    assert isinstance(tool, MCPStdioTool)
    assert tool.name == "s"
    assert tool.command == "python"


def test_create_stdio_tool_resolves_env_templates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_MCP_SECRET", "resolved-secret")
    config = MCPServerConfig(
        name="s",
        transport="stdio",
        command="python",
        env={"TOKEN": "{{CHRYS_MCP_SECRET}}", "AUTH": "Bearer {{CHRYS_MCP_SECRET}}"},
    )

    tool = _create_mcp_tool(config)

    assert tool.env == {"TOKEN": "resolved-secret", "AUTH": "Bearer resolved-secret"}


def test_create_stdio_tool_forwards_allowed_tools() -> None:
    config = MCPServerConfig(name="s", transport="stdio", command="python", allowed_tools=["ping"], request_timeout=7)
    tool = _create_mcp_tool(config)
    assert tool.allowed_tools == ["ping"]
    assert tool.request_timeout == 7


def test_create_stdio_tool_forwards_empty_allowed_tools() -> None:
    config = MCPServerConfig(name="s", transport="stdio", command="python", allowed_tools=[])
    tool = _create_mcp_tool(config)
    tool._functions = [_function_tool("one"), _function_tool("two")]

    assert tool.allowed_tools == []
    assert tool.functions == []


def test_mcp_tool_empty_allowed_tools_exposes_no_functions() -> None:
    tool = MCPTool(name="m", allowed_tools=[])
    tool._functions = [_function_tool("one"), _function_tool("two")]

    assert tool.functions == []


def test_mcp_allowed_tools_does_not_match_lossy_normalized_alias() -> None:
    exposed = _function_tool("delete-everything")
    exposed.additional_properties = {
        _MCP_REMOTE_NAME_KEY: "delete/everything",
        _MCP_NORMALIZED_NAME_KEY: "delete-everything",
    }
    tool = MCPTool(name="m", allowed_tools=["delete-everything"])
    tool._functions = [exposed]

    assert tool.functions == []


def test_mcp_allowed_tools_accepts_local_name_when_remote_is_already_normalized() -> None:
    exposed = _function_tool("srv_echo")
    exposed.additional_properties = {
        _MCP_REMOTE_NAME_KEY: "echo",
        _MCP_NORMALIZED_NAME_KEY: "echo",
    }
    tool = MCPTool(name="m", allowed_tools=["srv_echo"])
    tool._functions = [exposed]

    assert tool.functions == [exposed]


class _RecordingMCPTool(MCPTool):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="m", **kwargs)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, tool_name: str, **kwargs: Any) -> str:
        self.calls.append((tool_name, dict(kwargs)))
        return "ok"


@pytest.mark.asyncio
async def test_mcp_generated_tool_closure_strips_model_supplied_meta() -> None:
    tool = _RecordingMCPTool()
    await _load_fake_remote_tools(tool, _mcp_remote_tool("remote"))
    func = tool._functions[0]

    await func.invoke(arguments={"_meta": {"forged": "bad"}}, skip_parsing=True)

    assert tool.calls == [("remote", {})]


@pytest.mark.asyncio
async def test_mcp_generated_tool_closure_preserves_trusted_runtime_meta() -> None:
    tool = _RecordingMCPTool()
    await _load_fake_remote_tools(tool, _mcp_remote_tool("remote"))
    func = tool._functions[0]
    context = FunctionInvocationContext(
        function=func,
        arguments={},
        kwargs={"_meta": {"trusted": "ok"}},
    )

    await func.invoke(arguments={"_meta": {"forged": "bad"}}, context=context, skip_parsing=True)

    assert tool.calls == [("remote", {"_meta": {"trusted": "ok"}})]


@pytest.mark.asyncio
async def test_mcp_generated_tool_separates_declared_arguments_from_runtime_kwargs() -> None:
    tool = _RecordingMCPTool()
    await _load_fake_remote_tools(
        tool,
        _mcp_remote_tool(
            "remote",
            input_schema={
                "type": "object",
                "properties": {"session": {"type": "string"}},
            },
        ),
    )
    func = tool._functions[0]
    context = FunctionInvocationContext(
        function=func,
        arguments={"session": "sr-design"},
        kwargs={"session": object(), "future_runtime_key": object()},
    )

    await func.invoke(arguments={"session": "sr-design"}, context=context, skip_parsing=True)

    assert tool.calls == [("remote", {"session": "sr-design"})]


@pytest.mark.asyncio
async def test_mcp_generated_tool_does_not_substitute_runtime_value_for_omitted_optional_argument() -> None:
    tool = _RecordingMCPTool()
    await _load_fake_remote_tools(
        tool,
        _mcp_remote_tool(
            "remote",
            input_schema={
                "type": "object",
                "properties": {"session": {"type": "string"}},
            },
        ),
    )
    func = tool._functions[0]
    context = FunctionInvocationContext(
        function=func,
        arguments={},
        kwargs={"session": object()},
    )

    await func.invoke(arguments={}, context=context, skip_parsing=True)

    assert tool.calls == [("remote", {})]


@pytest.mark.asyncio
async def test_mcp_generated_tool_forwards_only_explicit_runtime_extras() -> None:
    tool = _RecordingMCPTool(additional_tool_argument_names={"remote": ["tenant_id"]})
    await _load_fake_remote_tools(tool, _mcp_remote_tool("remote"))
    func = tool._functions[0]
    context = FunctionInvocationContext(
        function=func,
        arguments={},
        kwargs={"tenant_id": "trusted-tenant", "internal": object()},
    )

    await func.invoke(arguments={}, context=context, skip_parsing=True)

    assert tool.calls == [("remote", {"tenant_id": "trusted-tenant"})]


@pytest.mark.asyncio
async def test_mcp_generated_tool_cannot_override_bound_remote_name() -> None:
    tool = MCPTool(name="m", allowed_tools=["safe"])
    session = await _load_calling_remote_tools(
        tool,
        _mcp_remote_tool(
            "safe",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
        ),
        _mcp_remote_tool("danger"),
    )
    assert [func.name for func in tool.functions] == ["safe"]

    await tool.functions[0].invoke(
        arguments={"value": "ok", "_remote_tool_name": "danger"},
        skip_parsing=True,
    )

    call = session.call_tool.await_args
    assert call.args == ("safe",)
    assert call.kwargs["arguments"] == {"value": "ok"}


@pytest.mark.asyncio
async def test_mcp_declared_remote_name_argument_is_data_not_dispatch_control() -> None:
    tool = MCPTool(name="m", allowed_tools=["safe"])
    session = await _load_calling_remote_tools(
        tool,
        _mcp_remote_tool(
            "safe",
            input_schema={
                "type": "object",
                "properties": {"_remote_tool_name": {"type": "string"}},
            },
        ),
        _mcp_remote_tool("danger"),
    )

    await tool.functions[0].invoke(
        arguments={"_remote_tool_name": "danger"},
        skip_parsing=True,
    )

    call = session.call_tool.await_args
    assert call.args == ("safe",)
    assert call.kwargs["arguments"] == {"_remote_tool_name": "danger"}


@pytest.mark.asyncio
async def test_mcp_declared_ctx_argument_survives_context_injection() -> None:
    tool = MCPTool(name="m")
    session = await _load_calling_remote_tools(
        tool,
        _mcp_remote_tool(
            "safe",
            input_schema={
                "type": "object",
                "properties": {"ctx": {"type": "string"}},
                "required": ["ctx"],
            },
        ),
    )

    await tool.functions[0].invoke(
        arguments={"ctx": "business-value"},
        skip_parsing=True,
    )

    call = session.call_tool.await_args
    assert call.args == ("safe",)
    assert call.kwargs["arguments"] == {"ctx": "business-value"}


@pytest.mark.asyncio
async def test_mcp_trusted_runtime_extra_overrides_same_named_model_argument() -> None:
    tool = MCPTool(
        name="m",
        additional_tool_argument_names={"safe": ["tenant_id"]},
    )
    session = await _load_calling_remote_tools(
        tool,
        _mcp_remote_tool(
            "safe",
            input_schema={
                "type": "object",
                "properties": {"tenant_id": {"type": "string"}},
            },
        ),
    )
    func = tool.functions[0]
    context = FunctionInvocationContext(
        function=func,
        arguments={"tenant_id": "model-tenant"},
        kwargs={"tenant_id": "trusted-tenant"},
    )

    await func.invoke(
        arguments={"tenant_id": "model-tenant"},
        context=context,
        skip_parsing=True,
    )

    call = session.call_tool.await_args
    assert call.args == ("safe",)
    assert call.kwargs["arguments"] == {"tenant_id": "trusted-tenant"}


@pytest.mark.parametrize("argument_name", sorted(_MCP_FRAMEWORK_DENYLIST - {"_meta"}))
def test_mcp_declared_framework_named_argument_is_forwarded(argument_name: str) -> None:
    tool = MCPTool(name="m")
    tool._tool_param_names_by_name = {"remote": {argument_name}}

    filtered, _meta = tool._prepare_call_kwargs("remote", {argument_name: "declared-value"})

    assert filtered == {argument_name: "declared-value"}


@pytest.mark.asyncio
async def test_mcp_declared_collision_reaches_client_session_without_model_meta() -> None:
    from mcp import types

    session = SimpleNamespace(
        list_tools=AsyncMock(
            return_value=SimpleNamespace(
                tools=[
                    _mcp_remote_tool(
                        "remote",
                        input_schema={
                            "type": "object",
                            "properties": {"session": {"type": "string"}},
                        },
                    )
                ],
                nextCursor=None,
            )
        ),
        call_tool=AsyncMock(
            return_value=types.CallToolResult(
                content=[types.TextContent(type="text", text="ok")],
                isError=False,
            )
        ),
    )
    tool = MCPTool(name="m", session=session)
    tool._ensure_connected = AsyncMock()  # type: ignore[method-assign]
    await tool.load_tools()
    func = tool._functions[0]
    context = FunctionInvocationContext(
        function=func,
        arguments={"session": "sr-design", "_meta": {"forged": "bad"}},
        kwargs={"session": object()},
    )

    await func.invoke(
        arguments={"session": "sr-design", "_meta": {"forged": "bad"}},
        context=context,
        skip_parsing=True,
    )

    call = session.call_tool.await_args
    assert call.args == ("remote",)
    assert call.kwargs["arguments"] == {"session": "sr-design"}
    assert call.kwargs["meta"] is None or "forged" not in call.kwargs["meta"]


@pytest.mark.asyncio
async def test_mcp_header_provider_sees_model_arguments_and_explicit_runtime_extras_only() -> None:
    from mcp import types

    provider_inputs: list[dict[str, Any]] = []

    def provide_headers(arguments: dict[str, Any]) -> dict[str, str]:
        provider_inputs.append(dict(arguments))
        return {}

    session = SimpleNamespace(
        list_tools=AsyncMock(
            return_value=SimpleNamespace(
                tools=[
                    _mcp_remote_tool(
                        "remote",
                        input_schema={
                            "type": "object",
                            "properties": {"value": {"type": "string"}},
                        },
                    )
                ],
                nextCursor=None,
            )
        ),
        call_tool=AsyncMock(
            return_value=types.CallToolResult(
                content=[types.TextContent(type="text", text="ok")],
                isError=False,
            )
        ),
    )
    tool = MCPStreamableHTTPTool(
        name="m",
        url="https://mcp.example/mcp",
        session=session,
        header_provider=provide_headers,
        additional_tool_argument_names={"remote": ["tenant_id"]},
    )
    tool._ensure_connected = AsyncMock()  # type: ignore[method-assign]
    await tool.load_tools()
    func = tool._functions[0]
    context = FunctionInvocationContext(
        function=func,
        arguments={"value": "model-value", "_meta": {"forged": "bad"}},
        kwargs={"tenant_id": "trusted-tenant", "session": object(), "internal": object()},
    )

    await func.invoke(
        arguments={"value": "model-value", "_meta": {"forged": "bad"}},
        context=context,
        skip_parsing=True,
    )

    assert provider_inputs == [{"tenant_id": "trusted-tenant", "value": "model-value"}]
    call = session.call_tool.await_args
    assert call.kwargs["arguments"] == {"tenant_id": "trusted-tenant", "value": "model-value"}


def test_mcp_request_meta_precedence_is_tool_meta_over_otel_over_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    def inject(carrier: dict[str, str]) -> None:
        carrier.update({"traceparent": "otel", "baggage": "otel"})

    monkeypatch.setattr(owned_mcp.propagate, "inject", inject)
    tool = MCPTool(name="m")
    tool._tool_param_names_by_name = {"remote": {"value"}}
    tool._tool_call_meta_by_name = {"remote": {"traceparent": "tool", "tool": "meta"}}

    filtered, meta = tool._prepare_call_kwargs(
        "remote",
        {
            "value": "ok",
            "_meta": {"traceparent": "caller", "caller": "meta"},
        },
    )

    assert filtered == {"value": "ok"}
    assert meta == {
        "traceparent": "tool",
        "caller": "meta",
        "baggage": "otel",
        "tool": "meta",
    }


@pytest.mark.asyncio
async def test_mcp_sampling_denies_by_default() -> None:
    from mcp import types

    client = _SamplingClient()
    tool = MCPTool(name="m", client=client)

    result = await tool.sampling_callback(SimpleNamespace(), _sampling_params())

    assert isinstance(result, types.ErrorData)
    assert result.code == types.INVALID_REQUEST
    assert "disabled by default" in result.message
    assert client.calls == []


@pytest.mark.asyncio
async def test_mcp_sampling_approval_callback_can_allow_and_clamp_tokens() -> None:
    from mcp import types

    client = _SamplingClient()
    tool = MCPTool(
        name="m",
        client=client,
        sampling_approval_callback=lambda params: True,
        sampling_max_tokens=10,
    )

    result = await tool.sampling_callback(SimpleNamespace(), _sampling_params(max_tokens=99))

    assert isinstance(result, types.CreateMessageResult)
    assert result.model == "m-test"
    assert client.calls[0]["options"]["max_tokens"] == 10


@pytest.mark.asyncio
async def test_mcp_sampling_async_approve_can_leave_tokens_uncapped_or_under_cap() -> None:
    from mcp import types

    async def approve(_params: Any) -> bool:
        return True

    uncapped_client = _SamplingClient()
    uncapped_tool = MCPTool(
        name="uncapped",
        client=uncapped_client,
        sampling_approval_callback=approve,
        sampling_max_tokens=None,
    )
    uncapped = await uncapped_tool.sampling_callback(SimpleNamespace(), _sampling_params(max_tokens=99))

    assert isinstance(uncapped, types.CreateMessageResult)
    assert uncapped_client.calls[0]["options"]["max_tokens"] == 99

    under_cap_client = _SamplingClient()
    under_cap_tool = MCPTool(
        name="under-cap",
        client=under_cap_client,
        sampling_approval_callback=approve,
        sampling_max_tokens=100,
    )
    under_cap = await under_cap_tool.sampling_callback(SimpleNamespace(), _sampling_params(max_tokens=99))

    assert isinstance(under_cap, types.CreateMessageResult)
    assert under_cap_client.calls[0]["options"]["max_tokens"] == 99


@pytest.mark.asyncio
async def test_mcp_sampling_async_deny_and_callback_error_do_not_call_client() -> None:
    from mcp import types

    async def deny(_params: Any) -> bool:
        return False

    denied_client = _SamplingClient()
    denied_tool = MCPTool(name="denied", client=denied_client, sampling_approval_callback=deny)
    denied = await denied_tool.sampling_callback(SimpleNamespace(), _sampling_params())
    assert isinstance(denied, types.ErrorData)
    assert denied.code == types.INVALID_REQUEST
    assert denied_client.calls == []

    def fail(_params: Any) -> bool:
        raise RuntimeError("nope")

    failed_client = _SamplingClient()
    failed_tool = MCPTool(name="failed", client=failed_client, sampling_approval_callback=fail)
    failed = await failed_tool.sampling_callback(SimpleNamespace(), _sampling_params())
    assert isinstance(failed, types.ErrorData)
    assert failed.code == types.INVALID_REQUEST
    assert failed_client.calls == []


@pytest.mark.asyncio
async def test_mcp_sampling_rate_limit_resets_with_session_state() -> None:
    from mcp import types

    client = _SamplingClient()
    tool = MCPTool(
        name="m",
        client=client,
        sampling_approval_callback=lambda params: True,
        sampling_max_requests=1,
    )

    first = await tool.sampling_callback(SimpleNamespace(), _sampling_params())
    second = await tool.sampling_callback(SimpleNamespace(), _sampling_params())
    tool._reset_session_state()
    third = await tool.sampling_callback(SimpleNamespace(), _sampling_params())

    assert isinstance(first, types.CreateMessageResult)
    assert isinstance(second, types.ErrorData)
    assert second.code == types.INVALID_REQUEST
    assert isinstance(third, types.CreateMessageResult)
    assert len(client.calls) == 2


def test_create_stdio_tool_forwards_description_prefix_and_load_prompts() -> None:
    config = MCPServerConfig(
        name="s",
        transport="stdio",
        command="python",
        description="Local server",
        tool_name_prefix="local",
        load_prompts=False,
    )
    tool = _create_mcp_tool(config)

    assert tool.description == "Local server"
    assert tool.tool_name_prefix == "local"
    assert tool.load_prompts_flag is False


@pytest.mark.parametrize(
    ("prefix", "progressive", "message"),
    [
        ("bad prefix", False, "tool name prefix"),
        ("github.v1", False, "underscores, and hyphens"),
        ("a" * 50, True, "invalid generated control.*maximum is 64"),
    ],
)
def test_create_mcp_tool_rejects_invalid_tool_name_prefix(
    prefix: str,
    progressive: bool,
    message: str,
) -> None:
    config = MCPServerConfig(
        name="s",
        transport="stdio",
        command="python",
        tool_name_prefix=prefix,
        use_progressive_disclosure=progressive,
    )

    with pytest.raises(MCPToolNameValidationError, match=message):
        _create_mcp_tool(config)


def test_create_mcp_tool_accepts_longest_progressive_control_prefix_boundary() -> None:
    prefix = "a" * 49
    config = MCPServerConfig(
        name="s",
        transport="stdio",
        command="python",
        tool_name_prefix=prefix,
        use_progressive_disclosure=True,
    )

    tool = _create_mcp_tool(config)

    assert tool.tool_name_prefix == prefix


def test_create_http_tool() -> None:
    config = MCPServerConfig(
        name="h",
        transport="http",
        url="http://localhost:8080/mcp",
        headers={"Authorization": "Bearer token"},
        terminate_on_close=True,
        allowed_tools=["echo"],
        request_timeout=15,
    )
    tool = _create_mcp_tool(config)
    assert isinstance(tool, MCPStreamableHTTPTool)
    assert isinstance(tool, _HTTPMCPTool)
    assert tool.name == "h"
    assert tool.url == "http://localhost:8080/mcp"
    assert tool.allowed_tools == ["echo"]
    assert tool.request_timeout == 15
    # Static headers use ``_HTTPMCPTool``'s same-origin request hook, not
    # header_provider, which would skip them on initialize / list_tools.
    assert tool._static_headers == {"Authorization": "Bearer token"}
    assert tool._header_provider is None


def test_create_http_tool_resolves_header_env_templates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHRYS_MCP_HTTP_TOKEN", "token-value")
    config = MCPServerConfig(
        name="h",
        transport="http",
        url="http://localhost:8080/mcp",
        headers={"Authorization": "Bearer {{CHRYS_MCP_HTTP_TOKEN}}"},
    )

    tool = _create_mcp_tool(config)

    assert tool._static_headers == {"Authorization": "Bearer token-value"}


def test_create_mcp_tool_rejects_missing_env_template(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHRYS_MCP_MISSING", raising=False)
    config = MCPServerConfig(
        name="s",
        transport="stdio",
        command="python",
        env={"TOKEN": "{{CHRYS_MCP_MISSING}}"},
    )

    with pytest.raises(EnvVarResolutionError) as info:
        _create_mcp_tool(config)

    message = str(info.value)
    assert "CHRYS_MCP_MISSING" in message
    assert "MCP server 's' env['TOKEN']" in message


def test_create_http_tool_without_headers_has_empty_static_headers() -> None:
    config = MCPServerConfig(name="h", transport="http", url="http://localhost:8080/mcp")
    tool = _create_mcp_tool(config)
    assert tool._header_provider is None
    assert tool._static_headers == {}


def test_create_http_tool_ignores_env_overrides() -> None:
    config = MCPServerConfig(
        name="h",
        transport="http",
        url="http://localhost:8080/mcp",
        env={"NO_PROXY": "*", "SSL_CERT_FILE": "/tmp/ca.pem"},
    )
    tool = _create_mcp_tool(config)
    assert isinstance(tool, _HTTPMCPTool)
    assert tool._needs_prebuild() is False


def test_create_http_tool_carries_bypass_proxy() -> None:
    config = MCPServerConfig(
        name="h",
        transport="http",
        url="http://localhost:8080/mcp",
        bypass_proxy=True,
    )
    tool = _create_mcp_tool(config)
    assert isinstance(tool, _HTTPMCPTool)
    assert tool._bypass_proxy is True


def test_create_unknown_transport() -> None:
    config = MCPServerConfig(name="x", transport="grpc")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown MCP transport"):
        _create_mcp_tool(config)


def test_create_stdio_without_command_rejected() -> None:
    config = MCPServerConfig(name="s", transport="stdio")
    with pytest.raises(ValueError, match="requires 'command'"):
        _create_mcp_tool(config)


def test_create_http_without_url_rejected() -> None:
    config = MCPServerConfig(name="h", transport="http")
    with pytest.raises(ValueError, match="requires 'url'"):
        _create_mcp_tool(config)


def test_create_without_name_rejected() -> None:
    config = MCPServerConfig(name="", transport="stdio", command="python")
    with pytest.raises(ValueError, match="non-empty 'name'"):
        _create_mcp_tool(config)


# ---------------------------------------------------------------------------
# _HTTPMCPTool — transport options and custom httpx ctor path
# ---------------------------------------------------------------------------


def test_streamable_http_private_post_hook_contract() -> None:
    """Fail loudly if MCP SDK private HTTP hook changes under our patch.

    ``_chrys_streamable_http_client`` subclasses ``StreamableHTTPTransport``
    and overrides ``_handle_post_request`` to wake pending MCP requests when
    an HTTP POST task fails.  That is a private SDK hook, so dependency updates
    must surface here instead of silently disabling the workaround.
    """
    import inspect

    import mcp.client.streamable_http as streamable_http
    from mcp.client.streamable_http import RequestContext, StreamableHTTPTransport

    hook = getattr(StreamableHTTPTransport, "_handle_post_request", None)
    assert hook is not None, "MCP SDK removed StreamableHTTPTransport._handle_post_request"
    assert inspect.iscoroutinefunction(hook), "MCP SDK changed _handle_post_request away from async"

    signature = inspect.signature(hook)
    assert list(signature.parameters) == ["self", "ctx"]
    assert signature.parameters["ctx"].annotation is RequestContext
    assert signature.return_annotation is None

    context_fields = set(getattr(RequestContext, "__annotations__", {}))
    assert {"client", "session_message", "read_stream_writer"} <= context_fields

    assert streamable_http.CONTENT_TYPE == "content-type"
    assert streamable_http.JSON == "application/json"
    assert streamable_http.SSE == "text/event-stream"

    helper_names = [
        "_prepare_headers",
        "_is_initialization_request",
        "_maybe_extract_session_id_from_response",
        "_handle_json_response",
        "_handle_sse_response",
        "_handle_unexpected_content_type",
        "_send_session_terminated_error",
    ]
    for name in helper_names:
        assert hasattr(StreamableHTTPTransport, name), f"MCP SDK removed StreamableHTTPTransport.{name}"


async def test_streamable_http_post_failure_wakes_pending_request() -> None:
    """A failing HTTP POST task must wake the pending request with JSON-RPC error."""
    from mcp.client.streamable_http import StreamableHTTPTransport
    from mcp.shared.message import SessionMessage
    from mcp.types import CONNECTION_CLOSED, JSONRPCError, JSONRPCMessage, JSONRPCRequest

    client = MagicMock()
    terminated: list[tuple[str | None, object]] = []

    @contextlib.asynccontextmanager
    async def fail_stream(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("post exploded")
        yield

    async def post_writer(
        self: Any,
        client_arg: object,
        write_stream_reader: object,
        read_stream_writer: object,
        write_stream: object,
        start_get_stream: object,
        tg: object,
    ) -> None:
        self.session_id = "sess-1"
        start_get_stream()
        request = JSONRPCRequest(jsonrpc="2.0", id=7, method="tools/list")
        ctx = SimpleNamespace(
            client=client_arg,
            session_message=SessionMessage(message=JSONRPCMessage(request)),
            read_stream_writer=read_stream_writer,
        )
        await self._handle_post_request(ctx)

    async def handle_get_stream(self: object, client_arg: object, read_stream_writer: object) -> None:
        return None

    async def terminate_session(self: Any, client_arg: object) -> None:
        terminated.append((self.session_id, client_arg))

    with (
        patch.object(client, "stream", side_effect=fail_stream),
        patch.object(StreamableHTTPTransport, "post_writer", new=post_writer),
        patch.object(StreamableHTTPTransport, "handle_get_stream", new=handle_get_stream),
        patch.object(StreamableHTTPTransport, "terminate_session", new=terminate_session),
    ):
        async with _chrys_streamable_http_client(
            "http://mcp.example/mcp",
            http_client=client,
            terminate_on_close=True,
        ) as (read_stream, _write_stream, get_session_id):
            received = await asyncio.wait_for(read_stream.receive(), timeout=20.0)
            assert get_session_id() == "sess-1"

    error = received.message.root
    assert isinstance(error, JSONRPCError)
    assert error.id == 7
    assert error.error.code == CONNECTION_CLOSED
    assert "post exploded" in error.error.message
    assert terminated == [("sess-1", client)]


async def test_streamable_http_post_failure_uses_readable_empty_exception_label() -> None:
    """The synthetic JSON-RPC error should not inherit blank transport messages."""
    from mcp.client.streamable_http import StreamableHTTPTransport
    from mcp.shared.message import SessionMessage
    from mcp.types import CONNECTION_CLOSED, JSONRPCError, JSONRPCMessage, JSONRPCRequest

    class ReadError(Exception):
        pass

    client = MagicMock()

    @contextlib.asynccontextmanager
    async def fail_stream(*_args: object, **_kwargs: object) -> object:
        raise ReadError(TimeoutError())
        yield

    async def post_writer(
        self: Any,
        client_arg: object,
        write_stream_reader: object,
        read_stream_writer: object,
        write_stream: object,
        start_get_stream: object,
        tg: object,
    ) -> None:
        request = JSONRPCRequest(jsonrpc="2.0", id=8, method="tools/list")
        ctx = SimpleNamespace(
            client=client_arg,
            session_message=SessionMessage(message=JSONRPCMessage(request)),
            read_stream_writer=read_stream_writer,
        )
        await self._handle_post_request(ctx)

    async def handle_get_stream(self: object, client_arg: object, read_stream_writer: object) -> None:
        return None

    with (
        patch.object(client, "stream", side_effect=fail_stream),
        patch.object(StreamableHTTPTransport, "post_writer", new=post_writer),
        patch.object(StreamableHTTPTransport, "handle_get_stream", new=handle_get_stream),
    ):
        async with _chrys_streamable_http_client(
            "http://mcp.example/mcp",
            http_client=client,
            terminate_on_close=False,
        ) as (read_stream, _write_stream, _get_session_id):
            received = await asyncio.wait_for(read_stream.receive(), timeout=20.0)

    error = received.message.root
    assert isinstance(error, JSONRPCError)
    assert error.id == 8
    assert error.error.code == CONNECTION_CLOSED
    assert error.error.message == "HTTP MCP request failed: Read failed (ReadError)"


async def test_streamable_http_post_failure_for_non_request_reraises() -> None:
    """Only request messages get synthetic JSON-RPC errors; other POST failures still raise."""
    from mcp.client.streamable_http import StreamableHTTPTransport
    from mcp.shared.message import SessionMessage
    from mcp.types import JSONRPCMessage, JSONRPCNotification

    raised = asyncio.Event()
    seen: list[str] = []

    client = MagicMock()

    @contextlib.asynccontextmanager
    async def fail_stream(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("notify failed")
        yield

    async def post_writer(
        self: Any,
        client_arg: object,
        write_stream_reader: object,
        read_stream_writer: object,
        write_stream: object,
        start_get_stream: object,
        tg: object,
    ) -> None:
        notification = JSONRPCNotification(jsonrpc="2.0", method="notifications/initialized")
        ctx = SimpleNamespace(
            client=client_arg,
            session_message=SessionMessage(message=JSONRPCMessage(notification)),
            read_stream_writer=read_stream_writer,
        )
        try:
            await self._handle_post_request(ctx)
        except RuntimeError as exc:
            seen.append(str(exc))
            raised.set()

    with (
        patch.object(client, "stream", side_effect=fail_stream),
        patch.object(StreamableHTTPTransport, "post_writer", new=post_writer),
    ):
        async with _chrys_streamable_http_client("http://mcp.example/mcp", http_client=client) as _transport:
            await asyncio.wait_for(raised.wait(), timeout=20.0)

    assert seen == ["notify failed"]


async def test_streamable_http_client_falls_back_when_private_hook_is_missing() -> None:
    """If the SDK private hook disappears, use the stock SDK client instead."""
    import mcp.client.streamable_http as streamable_http

    client = MagicMock()
    calls: list[tuple[str, object, bool]] = []

    class _TransportWithoutPrivateHook:
        pass

    @asynccontextmanager
    async def fallback_client(url: str, *, http_client: object = None, terminate_on_close: bool = True) -> Any:
        calls.append((url, http_client, terminate_on_close))
        yield "fallback-transport"

    with (
        patch.object(streamable_http, "StreamableHTTPTransport", _TransportWithoutPrivateHook),
        patch.object(streamable_http, "streamable_http_client", fallback_client),
    ):
        async with _chrys_streamable_http_client(
            "http://mcp.example/mcp",
            http_client=client,
            terminate_on_close=False,
        ) as transport:
            assert transport == "fallback-transport"

    assert calls == [("http://mcp.example/mcp", client, False)]


async def test_streamable_http_client_falls_back_when_session_message_moves() -> None:
    """If the SDK message wrapper moves, use the stock SDK client instead."""
    import builtins

    import mcp.client.streamable_http as streamable_http

    client = MagicMock()
    calls: list[tuple[str, object, bool]] = []

    @asynccontextmanager
    async def fallback_client(url: str, *, http_client: object = None, terminate_on_close: bool = True) -> Any:
        calls.append((url, http_client, terminate_on_close))
        yield "fallback-transport"

    with (
        patch.object(builtins, "__import__", side_effect=_session_message_import_blocker()),
        patch.object(streamable_http, "streamable_http_client", fallback_client),
    ):
        async with _chrys_streamable_http_client(
            "http://mcp.example/mcp",
            http_client=client,
            terminate_on_close=False,
        ) as transport:
            assert transport == "fallback-transport"

    assert calls == [("http://mcp.example/mcp", client, False)]


async def test_owned_streamable_http_dynamic_headers_are_stripped_from_actual_cross_origin_redirect() -> None:
    """The base owned HTTP MCP hook must strip inherited dynamic headers on redirects."""
    import httpx

    seen: list[tuple[str, dict[str, str]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), dict(request.headers)))
        if request.url.host == "origin.test":
            return httpx.Response(302, headers={"Location": "https://foreign.test/mcp"})
        return httpx.Response(200)

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    )
    tool = MCPStreamableHTTPTool(
        name="h",
        url="https://origin.test/mcp",
        header_provider=lambda _args: {"unused": "provider"},
        http_client=client,
    )

    with patch("chrys.service.mcp.owned.streamable_http_client", return_value=object()):
        tool.get_mcp_client()

    token = _mcp_call_headers.set({"X-Dynamic-Secret": "dynamic"})
    try:
        await client.get("https://origin.test/mcp")
    finally:
        _mcp_call_headers.reset(token)
        await client.aclose()

    assert seen[0][1]["x-dynamic-secret"] == "dynamic"
    assert "x-dynamic-secret" not in seen[1][1]


class TestHTTPMCPToolTransport:
    """``_HTTPMCPTool`` pre-builds an ``httpx.AsyncClient`` when HTTP transport
    options require constructor-time settings.

    The pre-build path threads ``verify=verify_ssl`` into the constructor
    because httpx has no env-var equivalent for disabling cert verification.
    """

    HTTPX_CTOR = "httpx.AsyncClient"

    @staticmethod
    def _new_tool(
        *,
        verify_ssl: bool = True,
        bypass_proxy: bool = False,
        headers: dict[str, str] | None = None,
        request_timeout: float | None = None,
    ) -> _HTTPMCPTool:
        return _HTTPMCPTool(
            name="h",
            url="http://localhost:8080/mcp",
            verify_ssl=verify_ssl,
            bypass_proxy=bypass_proxy,
            headers=headers,
            request_timeout=request_timeout,
        )

    @staticmethod
    def _mock_client() -> MagicMock:
        return MagicMock(name="httpx-client", aclose=AsyncMock())

    async def test_ctor_failure_propagates(self) -> None:
        """If ``httpx.AsyncClient`` raises, the failure is propagated."""
        tool = self._new_tool(verify_ssl=False)

        with (
            patch(self.HTTPX_CTOR, side_effect=RuntimeError("build failed")),
            pytest.raises(RuntimeError, match="build failed"),
        ):
            await tool.__aenter__()

    async def test_super_aenter_failure_closes_owned_client(self) -> None:
        tool = self._new_tool(verify_ssl=False)

        client = self._mock_client()
        with (
            patch(self.HTTPX_CTOR, return_value=client),
            patch.object(MCPStreamableHTTPTool, "__aenter__", new=AsyncMock(side_effect=RuntimeError("connect fail"))),
            pytest.raises(RuntimeError, match="connect fail"),
        ):
            await tool.__aenter__()

        client.aclose.assert_awaited_once()
        assert tool._owned_httpx_client is None
        assert tool._httpx_client is None

    async def test_aexit_closes_owned_client(self) -> None:
        tool = self._new_tool(verify_ssl=False)
        client = self._mock_client()

        with (
            patch(self.HTTPX_CTOR, return_value=client),
            patch.object(MCPStreamableHTTPTool, "__aenter__", new=AsyncMock(return_value=tool)),
        ):
            await tool.__aenter__()

        with patch.object(MCPStreamableHTTPTool, "__aexit__", new=AsyncMock()):
            await tool.__aexit__(None, None, None)

        client.aclose.assert_awaited_once()
        assert tool._owned_httpx_client is None

    async def test_default_verify_and_no_env_skips_prebuild(self) -> None:
        """Default verify=True with no env → no pre-build, upstream defaults preserved."""
        tool = self._new_tool()
        with (
            patch(self.HTTPX_CTOR) as ctor,
            patch.object(MCPStreamableHTTPTool, "__aenter__", new=AsyncMock(return_value=tool)),
        ):
            await tool.__aenter__()

        ctor.assert_not_called()
        assert tool._httpx_client is None
        assert tool._owned_httpx_client is None

    async def test_verify_ssl_false_triggers_prebuild_with_verify_false(self) -> None:
        """``verify_ssl=False`` alone (no env) builds an httpx client with verify=False."""
        client = self._mock_client()
        tool = self._new_tool(verify_ssl=False)

        with (
            patch(self.HTTPX_CTOR, return_value=client) as ctor,
            patch.object(MCPStreamableHTTPTool, "__aenter__", new=AsyncMock(return_value=tool)),
        ):
            await tool.__aenter__()

        ctor.assert_called_once()
        assert ctor.call_args.kwargs["verify"] is False
        assert ctor.call_args.kwargs["follow_redirects"] is True
        assert tool._httpx_client is client
        assert tool._owned_httpx_client is client

    async def test_bypass_proxy_triggers_prebuild_with_no_proxy_mounts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``bypass_proxy=True`` disables env-derived proxy transports without env patching."""
        client = self._mock_client()
        key = "NO_PROXY"
        monkeypatch.setenv(key, "original.example")
        seen: list[str | None] = []
        tool = self._new_tool(bypass_proxy=True)

        def _ctor(*_a: object, **_kw: object) -> MagicMock:
            seen.append(os.environ.get(key))
            return client

        with (
            patch(self.HTTPX_CTOR, side_effect=_ctor) as ctor,
            patch.object(MCPStreamableHTTPTool, "__aenter__", new=AsyncMock(return_value=tool)),
        ):
            await tool.__aenter__()

        ctor.assert_called_once()
        assert seen == ["original.example"]
        assert ctor.call_args.kwargs["verify"] is True
        assert ctor.call_args.kwargs["mounts"] == {
            "http://": None,
            "https://": None,
            "all://": None,
        }
        assert tool._httpx_client is client
        assert tool._owned_httpx_client is client

    async def test_bypass_proxy_reaches_origin_when_proxy_env_is_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clear_proxy_env,
        local_http_server,
    ) -> None:
        """The MCP-owned httpx client bypasses env proxies for real requests."""
        import httpx

        clear_proxy_env()

        async with (
            local_http_server(b"origin") as origin,
            local_http_server(b"proxy") as proxy,
        ):
            monkeypatch.setenv("HTTP_PROXY", proxy.url)

            control = httpx.AsyncClient(timeout=1)
            try:
                proxied = await control.get(f"{origin.url}/mcp")
            finally:
                await control.aclose()

            client = self._new_tool(bypass_proxy=True)._build_httpx_client()
            try:
                bypassed = await client.get(f"{origin.url}/mcp")
            finally:
                await client.aclose()

        assert proxied.text == "proxy"
        assert bypassed.text == "origin"
        assert proxy.hits and proxy.hits[0].startswith(f"GET {origin.url}/mcp ")
        assert origin.hits == ["GET /mcp HTTP/1.1"]

    async def test_verify_ssl_false_accepts_self_signed_https(
        self,
        monkeypatch: pytest.MonkeyPatch,
        clear_proxy_env,
        local_http_server,
        self_signed_server_ssl_context,
    ) -> None:
        """The MCP-owned httpx client threads verify=False into actual TLS handshakes."""
        import httpx

        clear_proxy_env()
        monkeypatch.setenv("NO_PROXY", "*")

        async with local_http_server(
            b"self-signed",
            scheme="https",
            ssl_context=self_signed_server_ssl_context,
        ) as server:
            verifying = self._new_tool(request_timeout=1)._build_httpx_client()
            try:
                with pytest.raises(httpx.ConnectError):
                    await verifying.get(f"{server.url}/mcp")
            finally:
                await verifying.aclose()

            client = self._new_tool(verify_ssl=False)._build_httpx_client()
            try:
                response = await client.get(f"{server.url}/mcp")
            finally:
                await client.aclose()

        assert response.text == "self-signed"
        assert server.hits == ["GET /mcp HTTP/1.1"]

    async def test_static_headers_trigger_prebuild_without_client_default_headers(self) -> None:
        """``headers`` alone (no env, default verify) must force a pre-build
        but must not pass ``headers=...`` into ``httpx.AsyncClient`` because
        httpx propagates client-level headers across cross-origin redirects.
        """
        client = self._mock_client()
        tool = self._new_tool(headers={"Authorization": "Bearer token", "X-Custom": "v"})

        with (
            patch(self.HTTPX_CTOR, return_value=client) as ctor,
            patch.object(MCPStreamableHTTPTool, "__aenter__", new=AsyncMock(return_value=tool)),
        ):
            await tool.__aenter__()

        ctor.assert_called_once()
        assert "headers" not in ctor.call_args.kwargs
        assert ctor.call_args.kwargs["verify"] is True
        assert tool._httpx_client is client
        assert tool._owned_httpx_client is client

    async def test_no_headers_omits_headers_kwarg(self) -> None:
        """When headers are empty, ``headers=`` must not be passed to
        ``httpx.AsyncClient`` — leaving httpx's own defaults in force."""
        client = self._mock_client()
        tool = self._new_tool(verify_ssl=False)  # force prebuild without headers

        with (
            patch(self.HTTPX_CTOR, return_value=client) as ctor,
            patch.object(MCPStreamableHTTPTool, "__aenter__", new=AsyncMock(return_value=tool)),
        ):
            await tool.__aenter__()

        assert "headers" not in ctor.call_args.kwargs

    async def test_request_timeout_triggers_prebuild_and_bounds_httpx_handshake(self) -> None:
        """A user request timeout should bound httpx connect/write/pool too."""
        from mcp.shared._httpx_utils import MCP_DEFAULT_SSE_READ_TIMEOUT

        client = self._mock_client()
        tool = self._new_tool(request_timeout=1)

        with (
            patch(self.HTTPX_CTOR, return_value=client) as ctor,
            patch.object(MCPStreamableHTTPTool, "__aenter__", new=AsyncMock(return_value=tool)),
        ):
            await tool.__aenter__()

        timeout = ctor.call_args.kwargs["timeout"]
        assert timeout.connect == 1.0
        assert timeout.write == 1.0
        assert timeout.pool == 1.0
        assert timeout.read == MCP_DEFAULT_SSE_READ_TIMEOUT

    async def test_close_clears_inject_headers_sentinel(self) -> None:
        """Closing an owned client must let the next client reattach headers."""
        tool = self._new_tool(verify_ssl=False)
        client = self._mock_client()

        with (
            patch(self.HTTPX_CTOR, return_value=client),
            patch.object(MCPStreamableHTTPTool, "__aenter__", new=AsyncMock(return_value=tool)),
        ):
            await tool.__aenter__()

        async def _hook(_request: object) -> None: ...

        # Simulate request-hook installation during get_mcp_client.
        tool._inject_headers_hook = _hook

        with patch.object(MCPStreamableHTTPTool, "__aexit__", new=AsyncMock()):
            await tool.__aexit__(None, None, None)

        # Sentinel cleared so a hook installs on the next client.
        assert tool._inject_headers_hook is None
        assert tool._owned_httpx_client is None
        assert tool._httpx_client is None

    async def test_get_mcp_client_attaches_dynamic_header_hook_once(self) -> None:
        """Dynamic header providers need the parent request hook on our custom client."""
        from httpx import Request

        sentinel = object()
        client = MagicMock()
        client.event_hooks = {"request": []}
        tool = _HTTPMCPTool(
            name="h",
            url="http://localhost:8080/mcp",
            header_provider=lambda _args: {"unused": "provider"},
        )

        with (
            patch(self.HTTPX_CTOR, return_value=client) as ctor,
            patch("chrys.service.mcp.adapter._chrys_streamable_http_client", return_value=sentinel) as stream_client,
        ):
            first = tool.get_mcp_client()
            second = tool.get_mcp_client()

        assert first is sentinel
        assert second is sentinel
        ctor.assert_called_once()
        assert tool._httpx_client is client
        assert len(client.event_hooks["request"]) == 1
        stream_client.assert_called_with(
            url="http://localhost:8080/mcp",
            http_client=client,
            terminate_on_close=True,
            request_timeout=None,
        )

        token = _mcp_call_headers.set({"Authorization": "Bearer dyn", "X-Trace": "abc"})
        try:
            request = Request("POST", "http://localhost:8080/mcp")
            await client.event_hooks["request"][0](request)
        finally:
            _mcp_call_headers.reset(token)

        assert request.headers["Authorization"] == "Bearer dyn"
        assert request.headers["X-Trace"] == "abc"

    async def test_get_mcp_client_dynamic_headers_are_same_origin_only(self) -> None:
        """Dynamic MCP call headers must not leak onto cross-origin redirects."""
        from httpx import Request

        client = MagicMock()
        client.event_hooks = {"request": []}
        tool = _HTTPMCPTool(
            name="h",
            url="http://localhost:8080/mcp",
            header_provider=lambda _args: {"unused": "provider"},
        )

        with (
            patch(self.HTTPX_CTOR, return_value=client),
            patch("chrys.service.mcp.adapter._chrys_streamable_http_client", return_value=object()),
        ):
            tool.get_mcp_client()

        token = _mcp_call_headers.set({"Authorization": "Bearer dyn", "X-Trace": "abc"})
        try:
            same_origin = Request("POST", "http://localhost:8080/mcp")
            cross_origin = Request("POST", "http://localhost:9090/mcp")
            await client.event_hooks["request"][0](same_origin)
            await client.event_hooks["request"][0](cross_origin)
        finally:
            _mcp_call_headers.reset(token)

        assert same_origin.headers["Authorization"] == "Bearer dyn"
        assert same_origin.headers["X-Trace"] == "abc"
        assert "Authorization" not in cross_origin.headers
        assert "X-Trace" not in cross_origin.headers

    async def test_get_mcp_client_static_headers_are_same_origin_only(self) -> None:
        """Static per-server headers must not leak onto cross-origin redirects."""
        from httpx import Request

        client = MagicMock()
        client.event_hooks = {"request": []}
        tool = _HTTPMCPTool(
            name="h",
            url="http://localhost:8080/mcp",
            headers={"Authorization": "Bearer static", "X-Server": "srv"},
            http_client=client,
        )

        with (
            patch(self.HTTPX_CTOR) as ctor,
            patch("chrys.service.mcp.adapter._chrys_streamable_http_client", return_value=object()),
        ):
            tool.get_mcp_client()

        ctor.assert_not_called()
        assert len(client.event_hooks["request"]) == 1

        same_origin = Request("POST", "http://localhost:8080/mcp")
        cross_origin = Request("POST", "http://localhost:9090/mcp")
        await client.event_hooks["request"][0](same_origin)
        await client.event_hooks["request"][0](cross_origin)

        assert same_origin.headers["Authorization"] == "Bearer static"
        assert same_origin.headers["X-Server"] == "srv"
        assert "Authorization" not in cross_origin.headers
        assert "X-Server" not in cross_origin.headers

    async def test_static_headers_are_stripped_from_actual_cross_origin_redirect(self) -> None:
        """httpx may inherit custom request headers across redirects; strip configured keys."""
        import httpx

        seen: list[tuple[str, dict[str, str]]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append((str(request.url), dict(request.headers)))
            if request.url.host == "origin.test":
                return httpx.Response(302, headers={"Location": "https://foreign.test/mcp"})
            return httpx.Response(200)

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
        )
        tool = _HTTPMCPTool(
            name="h",
            url="https://origin.test/mcp",
            headers={"X-Secret": "static", "X-Server": "srv"},
            http_client=client,
        )
        tool._ensure_header_hook(client)

        try:
            await client.get("https://origin.test/mcp")
        finally:
            await client.aclose()

        assert seen[0][1]["x-secret"] == "static"
        assert seen[0][1]["x-server"] == "srv"
        assert "x-secret" not in seen[1][1]
        assert "x-server" not in seen[1][1]

    async def test_get_mcp_client_dynamic_headers_override_static_headers(self) -> None:
        """Per-call dynamic headers should retain the old override behavior."""
        from httpx import Request

        client = MagicMock()
        client.event_hooks = {"request": []}
        tool = _HTTPMCPTool(
            name="h",
            url="http://localhost:8080/mcp",
            headers={"Authorization": "Bearer static", "X-Server": "srv"},
            header_provider=lambda _args: {"unused": "provider"},
            http_client=client,
        )

        with patch("chrys.service.mcp.adapter._chrys_streamable_http_client", return_value=object()):
            tool.get_mcp_client()

        token = _mcp_call_headers.set({"Authorization": "Bearer dynamic", "X-Trace": "abc"})
        try:
            request = Request("POST", "http://localhost:8080/mcp")
            await client.event_hooks["request"][0](request)
        finally:
            _mcp_call_headers.reset(token)

        assert request.headers["Authorization"] == "Bearer dynamic"
        assert request.headers["X-Server"] == "srv"
        assert request.headers["X-Trace"] == "abc"

    async def test_dynamic_headers_are_stripped_from_actual_cross_origin_redirect(self) -> None:
        """Dynamic header keys also need redirect-time stripping, not only guarded injection."""
        import httpx

        seen: list[tuple[str, dict[str, str]]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen.append((str(request.url), dict(request.headers)))
            if request.url.host == "origin.test":
                return httpx.Response(302, headers={"Location": "https://foreign.test/mcp"})
            return httpx.Response(200)

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
        )
        tool = _HTTPMCPTool(
            name="h",
            url="https://origin.test/mcp",
            header_provider=lambda _args: {"unused": "provider"},
            http_client=client,
        )
        tool._ensure_header_hook(client)

        token = _mcp_call_headers.set({"X-Dynamic-Secret": "dynamic"})
        try:
            await client.get("https://origin.test/mcp")
        finally:
            _mcp_call_headers.reset(token)
            await client.aclose()

        assert seen[0][1]["x-dynamic-secret"] == "dynamic"
        assert "x-dynamic-secret" not in seen[1][1]

    def test_get_mcp_client_uses_custom_transport_without_header_provider(self) -> None:
        """No header provider: pass the current client through without creating a new one."""
        existing = MagicMock()
        sentinel = object()
        tool = _HTTPMCPTool(
            name="h",
            url="http://localhost:8080/mcp",
            http_client=existing,
            terminate_on_close=False,
        )

        with (
            patch(self.HTTPX_CTOR) as ctor,
            patch("chrys.service.mcp.adapter._chrys_streamable_http_client", return_value=sentinel) as stream_client,
        ):
            result = tool.get_mcp_client()

        assert result is sentinel
        ctor.assert_not_called()
        stream_client.assert_called_once_with(
            url="http://localhost:8080/mcp",
            http_client=existing,
            terminate_on_close=False,
            request_timeout=None,
        )

    def test_build_httpx_client_falls_back_to_literal_timeouts_when_mcp_utils_change(self) -> None:
        """If ``mcp.shared._httpx_utils`` loses its default constants, fall back
        to the historical literals (30.0s connect / 300.0s SSE read) instead of
        crashing every prebuilt HTTP client."""
        import builtins

        real_import = builtins.__import__

        def guarded_import(
            name: str,
            globals_: dict[str, Any] | None = None,
            locals_: dict[str, Any] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            if name == "mcp.shared._httpx_utils" and "MCP_DEFAULT_TIMEOUT" in fromlist:
                raise ImportError("httpx utils moved")
            return real_import(name, globals_, locals_, fromlist, level)

        captured: dict[str, Any] = {}

        class _CapturingClient:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        tool = self._new_tool(verify_ssl=False)
        with (
            patch.object(builtins, "__import__", side_effect=guarded_import),
            patch(self.HTTPX_CTOR, _CapturingClient),
        ):
            tool._build_httpx_client()

        timeout = captured["timeout"]
        assert timeout.connect == 30.0
        assert timeout.read == 300.0

    async def test_caller_provided_http_client_is_not_overridden(self) -> None:
        """A caller-supplied ``http_client`` wins over TLS and proxy settings."""
        existing = MagicMock(aclose=AsyncMock())
        tool = _HTTPMCPTool(
            name="h",
            url="http://localhost:8080/mcp",
            verify_ssl=False,
            bypass_proxy=True,
            http_client=existing,
        )

        with (
            patch(self.HTTPX_CTOR) as ctor,
            patch.object(MCPStreamableHTTPTool, "__aenter__", new=AsyncMock(return_value=tool)),
        ):
            await tool.__aenter__()

        ctor.assert_not_called()
        assert tool._httpx_client is existing
        assert tool._owned_httpx_client is None


# ---------------------------------------------------------------------------
# connect / connect_all failure surfacing
# ---------------------------------------------------------------------------


class _FakeTool:
    """Minimal async-ctx-manager MCPTool stand-in for patching _create_mcp_tool."""

    def __init__(self, *, enter_error: Exception | None = None, functions: list | None = None) -> None:
        self._enter_error = enter_error
        self.functions = functions or []
        self.__aenter__ = AsyncMock(side_effect=enter_error) if enter_error else AsyncMock(return_value=self)
        self.__aexit__ = AsyncMock(return_value=None)


async def test_owned_catalog_normalizes_periods_to_provider_safe_names() -> None:
    owned_tool = MCPTool(name="s")
    await _load_fake_remote_tools(owned_tool, _mcp_remote_tool("github.v1.search"))

    assert [
        (function.name, function.additional_properties[_MCP_REMOTE_NAME_KEY]) for function in owned_tool.functions
    ] == [("github-v1-search", "github.v1.search")]


@pytest.mark.parametrize("connection_path", ["test", "agent"])
async def test_prefix_and_remote_name_combination_over_64_characters_fails_early(connection_path: str) -> None:
    prefix = "a" * 50
    remote_name = "b" * 14
    owned_tool = MCPTool(name="s", tool_name_prefix=prefix)
    await _load_fake_remote_tools(owned_tool, _mcp_remote_tool(remote_name))
    assert owned_tool.functions[0].name == f"{prefix}_{remote_name}"

    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python", tool_name_prefix=prefix)
    fake = _FakeTool(functions=owned_tool.functions)
    with (
        patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=fake),
        pytest.raises(MCPToolNameValidationError, match=r"65 characters.*maximum is 64"),
    ):
        if connection_path == "test":
            await adapter.test_connection(config)
        else:
            await adapter.connect(config)

    await adapter.disconnect_all()


@pytest.mark.parametrize("connection_path", ["test", "agent"])
async def test_owned_catalog_normalized_collision_fails_test_and_agent_connection(connection_path: str) -> None:
    owned_tool = MCPTool(name="s")
    await _load_fake_remote_tools(owned_tool, _mcp_remote_tool("a/b"), _mcp_remote_tool("a-b"))
    exposed = [
        (function.name, function.additional_properties[_MCP_REMOTE_NAME_KEY]) for function in owned_tool.functions
    ]
    assert exposed == [("a-b", "a/b"), ("a-b", "a-b")]

    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python")
    fake = _FakeTool(functions=owned_tool.functions)
    with (
        patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=fake),
        pytest.raises(MCPToolNameCollisionError, match=r"invalid tool configuration.*a-b"),
    ):
        if connection_path == "test":
            await adapter.test_connection(config)
        else:
            await adapter.connect(config)

    await adapter.disconnect_all()


@pytest.mark.parametrize("remote_order", [("a/b", "a-b"), ("a-b", "a/b")])
async def test_owned_catalog_filters_allowlist_before_collision_validation(remote_order: tuple[str, str]) -> None:
    owned_tool = MCPTool(name="s", allowed_tools=["a/b"])
    await _load_fake_remote_tools(owned_tool, *(_mcp_remote_tool(name) for name in remote_order))

    assert [
        (function.name, function.additional_properties[_MCP_REMOTE_NAME_KEY]) for function in owned_tool.functions
    ] == [("a-b", "a/b")]


async def test_owned_catalog_tool_and_prompt_collision_fails_connection() -> None:
    owned_tool = MCPTool(name="s")
    owned_tool.session = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=[_mcp_remote_tool("shared")], nextCursor=None)),
        list_prompts=AsyncMock(return_value=SimpleNamespace(prompts=[_mcp_remote_prompt("shared")], nextCursor=None)),
    )
    owned_tool._ensure_connected = AsyncMock()  # type: ignore[method-assign]
    await owned_tool.load_tools()
    await owned_tool.load_prompts()
    assert [function.name for function in owned_tool.functions] == ["shared", "shared"]

    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python")
    fake = _FakeTool(functions=owned_tool.functions)
    with (
        patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=fake),
        pytest.raises(MCPToolNameCollisionError, match=r"invalid tool configuration.*shared") as exc_info,
    ):
        await adapter.connect(config)

    assert "disable 'Expose server prompts'" in str(exc_info.value)
    await adapter.disconnect_all()


async def test_owned_catalog_reload_deduplicates_only_the_same_remote_declaration() -> None:
    owned_tool = MCPTool(name="s")
    owned_tool.session = SimpleNamespace(
        list_tools=AsyncMock(return_value=SimpleNamespace(tools=[_mcp_remote_tool("tool")], nextCursor=None)),
        list_prompts=AsyncMock(return_value=SimpleNamespace(prompts=[_mcp_remote_prompt("prompt")], nextCursor=None)),
    )
    owned_tool._ensure_connected = AsyncMock()  # type: ignore[method-assign]

    await owned_tool.load_tools()
    await owned_tool.load_prompts()
    await owned_tool.load_tools()
    await owned_tool.load_prompts()

    assert [function.name for function in owned_tool.functions] == ["tool", "prompt"]


async def test_owned_catalog_notification_reloads_without_duplicate_remote_tools() -> None:
    owned_tool = MCPTool(name="s")
    list_tools = AsyncMock(return_value=SimpleNamespace(tools=[_mcp_remote_tool("tool")], nextCursor=None))
    owned_tool.session = SimpleNamespace(list_tools=list_tools)
    owned_tool._ensure_connected = AsyncMock()  # type: ignore[method-assign]
    await owned_tool.load_tools()

    await owned_tool.message_handler(_tool_list_changed_notification())
    tasks = list(owned_tool._pending_reload_tasks)
    assert len(tasks) == 1
    await asyncio.gather(*tasks)

    assert [function.name for function in owned_tool.functions] == ["tool"]
    assert list_tools.await_count == 2


async def test_owned_catalog_notifications_coalesce_by_cancelling_first_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned_tool = MCPTool(name="s")
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    second_release = asyncio.Event()
    calls = 0
    cancelled = 0

    async def slow_load_tools() -> None:
        nonlocal calls, cancelled
        calls += 1
        if calls == 1:
            first_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled += 1
                raise
        second_started.set()
        await second_release.wait()

    monkeypatch.setattr(owned_tool, "load_tools", slow_load_tools)
    await owned_tool.message_handler(_tool_list_changed_notification())
    first_task = next(iter(owned_tool._pending_reload_tasks))
    await asyncio.wait_for(first_started.wait(), timeout=5)

    await owned_tool.message_handler(_tool_list_changed_notification())
    second_task = next(task for task in owned_tool._pending_reload_tasks if task is not first_task)
    await asyncio.wait_for(second_started.wait(), timeout=5)
    await asyncio.gather(first_task, return_exceptions=True)

    assert cancelled == 1
    assert calls == 2
    second_release.set()
    await second_task


async def test_owned_catalog_close_cancels_pending_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    owned_tool = MCPTool(name="s")
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def pending_load_tools() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr(owned_tool, "load_tools", pending_load_tools)
    await owned_tool.message_handler(_tool_list_changed_notification())
    await asyncio.wait_for(started.wait(), timeout=5)

    await owned_tool.close()

    assert cancelled.is_set()
    assert owned_tool._pending_reload_tasks == set()


async def test_owned_catalog_reload_exception_is_logged_not_propagated(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    owned_tool = MCPTool(name="s")

    async def failing_load_tools() -> None:
        raise RuntimeError("catalog failed")

    monkeypatch.setattr(owned_tool, "load_tools", failing_load_tools)
    with caplog.at_level(logging.WARNING, logger=owned_mcp.__name__):
        await owned_tool.message_handler(_tool_list_changed_notification())
        tasks = list(owned_tool._pending_reload_tasks)
        assert len(tasks) == 1
        await asyncio.gather(*tasks)

    assert "Background MCP reload failed" in caplog.text
    assert "catalog failed" in caplog.text


async def test_connect_success_stamps_kind_and_caches() -> None:
    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python")
    fn = _function_tool()
    fake = _FakeTool(functions=[fn])

    with patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=fake):
        first = await adapter.connect(config)
        second = await adapter.connect(config)  # cache hit, no second aenter

    assert first == second
    assert first[0] is not fn
    assert first[0].name == "remote"
    assert get_tool_kind(first[0]) == "mcp"
    assert adapter.server_names == ["s"]
    fake.__aenter__.assert_awaited_once()


async def test_connect_raises_mcp_connection_error_and_records_failure() -> None:
    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python")
    fake = _FakeTool(enter_error=RuntimeError("boom"))

    with (
        patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=fake),
        pytest.raises(MCPConnectionError) as info,
    ):
        await adapter.connect(config)

    err = info.value
    assert err.server_name == "s"
    assert err.transport == "stdio"
    assert isinstance(err.__cause__, RuntimeError)
    assert "s" in adapter.failures
    assert adapter.server_names == []
    fake.__aexit__.assert_awaited_once()  # partial transport was closed


async def test_connect_reraises_cancelled_error() -> None:
    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python")
    fake = _FakeTool(enter_error=asyncio.CancelledError())

    with (
        patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=fake),
        pytest.raises(asyncio.CancelledError),
    ):
        await adapter.connect(config)

    assert adapter.server_names == []
    assert adapter.failures == {}  # cancellation is not a "failure"
    fake.__aexit__.assert_awaited_once()


async def test_owned_mcp_close_runs_on_lifecycle_owner_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open/close must happen in the same task for anyio cancel scopes."""
    enter_task: asyncio.Task[object] | None = None
    exit_task: asyncio.Task[object] | None = None

    @asynccontextmanager
    async def transport() -> Any:
        nonlocal enter_task, exit_task
        enter_task = asyncio.current_task()
        yield object(), object()
        exit_task = asyncio.current_task()

    class _FakeClientSession:
        _request_id = 1

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> _FakeClientSession:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def initialize(self) -> SimpleNamespace:
            return SimpleNamespace(protocolVersion="2024-11-05", capabilities=None)

    class _TaskRecordingTool(MCPTool):
        def get_mcp_client(self) -> Any:
            return transport()

    monkeypatch.setattr("mcp.client.session.ClientSession", _FakeClientSession)

    tool = _TaskRecordingTool(name="task-recorder")
    await tool.connect()
    assert enter_task is not None
    assert enter_task is not asyncio.current_task()

    close_task = asyncio.create_task(tool.close())
    await close_task

    assert exit_task is enter_task


async def test_owned_mcp_initialize_failure_does_not_commit_closed_session(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def transport() -> Any:
        yield object(), object()

    class _FakeClientSession:
        _request_id = 0

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.closed = False

        async def __aenter__(self) -> _FakeClientSession:
            return self

        async def __aexit__(self, *_args: object) -> None:
            self.closed = True

        async def initialize(self) -> SimpleNamespace:
            raise RuntimeError("init failed")

    class _InitFailTool(MCPTool):
        def get_mcp_client(self) -> Any:
            return transport()

    monkeypatch.setattr("mcp.client.session.ClientSession", _FakeClientSession)

    tool = _InitFailTool(name="init-fail")
    with pytest.raises(Exception, match="init failed"):
        await tool.connect()

    assert tool.session is None
    assert tool.is_connected is False


async def test_load_retries_reconnect_without_recursive_configured_load() -> None:
    """ClosedResourceError during list pagination must not deadlock on the load lock."""
    from mcp import types

    tool = _HTTPMCPTool(name="h", url="http://localhost/mcp")
    reconnect_calls = 0

    async def reconnect_without_loading() -> None:
        nonlocal reconnect_calls
        reconnect_calls += 1
        assert tool._function_load_lock.locked()

    tool._ensure_connected = AsyncMock()
    tool._reconnect_without_loading = AsyncMock(side_effect=reconnect_without_loading)
    tool.session = SimpleNamespace(
        list_tools=AsyncMock(side_effect=[anyio.ClosedResourceError(), types.ListToolsResult(tools=[])]),
        list_prompts=AsyncMock(side_effect=[anyio.ClosedResourceError(), types.ListPromptsResult(prompts=[])]),
    )

    await asyncio.wait_for(tool.load_tools(), timeout=20.0)
    await asyncio.wait_for(tool.load_prompts(), timeout=20.0)

    assert reconnect_calls == 2
    assert tool._reconnect_without_loading.await_count == 2


async def test_load_reconnect_failure_is_wrapped() -> None:
    tool = _HTTPMCPTool(name="h", url="http://localhost/mcp")
    tool._ensure_connected = AsyncMock()
    tool._reconnect_without_loading = AsyncMock(side_effect=RuntimeError("reconnect failed"))
    tool.session = SimpleNamespace(list_tools=AsyncMock(side_effect=anyio.ClosedResourceError()))

    with pytest.raises(Exception) as info:
        await tool.load_tools()

    assert type(info.value).__name__ == "ToolExecutionException"
    assert "Failed to reconnect to MCP server." in str(info.value)


async def test_get_prompt_reconnect_failure_is_wrapped() -> None:
    tool = _HTTPMCPTool(name="h", url="http://localhost/mcp")
    tool.connect = AsyncMock(side_effect=RuntimeError("reconnect failed"))
    tool.session = SimpleNamespace(get_prompt=AsyncMock(side_effect=anyio.ClosedResourceError()))

    with pytest.raises(Exception) as info:
        await tool.get_prompt("p")

    assert type(info.value).__name__ == "ToolExecutionException"
    assert "Failed to reconnect to MCP server." in str(info.value)
    tool.connect.assert_awaited_once_with(reset=True)


async def test_connect_all_skips_disabled_servers() -> None:
    adapter = MCPAdapter()
    enabled = MCPServerConfig(name="enabled", transport="stdio", command="python")
    disabled = MCPServerConfig(name="disabled", transport="stdio", command="python", enabled=False)

    with patch.object(adapter, "connect", new=AsyncMock(return_value=["tool-enabled"])) as connect_mock:
        tools = await adapter.connect_all([enabled, disabled])

    assert tools == ["tool-enabled"]
    connect_mock.assert_awaited_once_with(enabled)


async def test_connect_all_publishes_warning_on_failure_and_continues() -> None:
    bus = MagicMock()
    bus.publish = AsyncMock()
    adapter = MCPAdapter(bus=bus, session_id="sess-1")

    good = MCPServerConfig(name="good", transport="stdio", command="python")
    bad = MCPServerConfig(name="bad", transport="stdio", command="python")

    async def fake_connect(cfg: MCPServerConfig) -> list:
        if cfg.name == "bad":
            raise MCPConnectionError(cfg.name, cfg.transport, RuntimeError("spawn fail"))
        return [f"tool-{cfg.name}"]

    with patch.object(adapter, "connect", new=AsyncMock(side_effect=fake_connect)):
        tools = await adapter.connect_all([bad, good])

    # Bad server does not stop the good one
    assert tools == ["tool-good"]
    assert adapter.tool_names_by_server == {"good": ["tool-good"]}
    # Warning event went to the bus with the expected shape
    bus.publish.assert_awaited_once()
    (event,), _ = bus.publish.await_args
    assert isinstance(event, WarningEvent)
    assert event.code == "mcp.connect_failed"
    assert "bad" in event.message
    assert event.session_id == "sess-1"


async def test_connect_all_publishes_warning_for_missing_env_template(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHRYS_MCP_MISSING", raising=False)
    bus = MagicMock()
    bus.publish = AsyncMock()
    adapter = MCPAdapter(bus=bus, session_id="sess-1")
    bad = MCPServerConfig(
        name="bad",
        transport="stdio",
        command="python",
        env={"TOKEN": "{{CHRYS_MCP_MISSING}}"},
    )

    tools = await adapter.connect_all([bad])

    assert tools == []
    bus.publish.assert_awaited_once()
    (event,), _ = bus.publish.await_args
    assert isinstance(event, WarningEvent)
    assert event.code == "mcp.connect_failed"
    assert "CHRYS_MCP_MISSING" in event.message
    assert "MCP server 'bad' env['TOKEN']" in event.message


async def test_connect_all_without_bus_still_continues() -> None:
    adapter = MCPAdapter()  # bus=None
    bad = MCPServerConfig(name="bad", transport="stdio", command="python")

    async def fake_connect(cfg: MCPServerConfig) -> list:
        raise MCPConnectionError(cfg.name, cfg.transport, RuntimeError("x"))

    with patch.object(adapter, "connect", new=AsyncMock(side_effect=fake_connect)):
        tools = await adapter.connect_all([bad])

    assert tools == []


async def test_connect_all_wraps_unexpected_exception_and_continues() -> None:
    """Non-``MCPConnectionError`` failures must not abort the build either.

    If ``connect()`` raises e.g. ``ValueError`` (malformed config caught by
    ``_validate_config``) or any unexpected MCP error, the engine must
    still come up with the remaining good MCP tools rather than failing
    startup and leaving the user with "Engine not started".
    """
    bus = MagicMock()
    bus.publish = AsyncMock()
    adapter = MCPAdapter(bus=bus, session_id="sess-1")

    bad = MCPServerConfig(name="bad", transport="stdio", command="python")
    good = MCPServerConfig(name="good", transport="stdio", command="python")

    async def fake_connect(cfg: MCPServerConfig) -> list:
        if cfg.name == "bad":
            raise ValueError("malformed config")
        return [f"tool-{cfg.name}"]

    with patch.object(adapter, "connect", new=AsyncMock(side_effect=fake_connect)):
        tools = await adapter.connect_all([bad, good])

    assert tools == ["tool-good"]
    assert adapter.tool_names_by_server == {"good": ["tool-good"]}
    assert "bad" in adapter.failures
    assert isinstance(adapter.failures["bad"].cause, ValueError)
    bus.publish.assert_awaited_once()
    (event,), _ = bus.publish.await_args
    assert isinstance(event, WarningEvent)
    assert event.code == "mcp.connect_failed"
    assert "bad" in event.message


async def test_connect_all_reports_progress_for_unexpected_exception() -> None:
    adapter = MCPAdapter()
    bad = MCPServerConfig(name="bad", transport="stdio", command="python")
    progress_events: list[tuple[str, str, int, int, int]] = []

    async def fake_connect(cfg: MCPServerConfig) -> list:
        raise ValueError("malformed")

    async def progress(config: MCPServerConfig, state: str, current: int, total: int, failed: int) -> None:
        progress_events.append((config.name, state, current, total, failed))

    with patch.object(adapter, "connect", new=AsyncMock(side_effect=fake_connect)):
        tools = await adapter.connect_all([bad], progress=progress)

    assert tools == []
    assert progress_events == [("bad", "starting", 0, 1, 0), ("bad", "failed", 0, 1, 1)]


async def test_connect_all_starts_enabled_servers_in_parallel() -> None:
    adapter = MCPAdapter()
    gate = asyncio.Event()
    entered = {"a": asyncio.Event(), "b": asyncio.Event()}
    calls: list[str] = []
    progress_events: list[tuple[str, str, int, int, int]] = []
    fn_a = _function_tool("a_remote")
    fn_b = _function_tool("b_remote")

    class _SlowTool:
        def __init__(self, name: str, fn: ChrysFunctionTool) -> None:
            self.name = name
            self.functions = [fn]
            self.request_timeout = DEFAULT_CONNECT_TIMEOUT_SECONDS
            self.dropped_banner_lines: list[str] = []

        async def __aenter__(self) -> _SlowTool:
            calls.append(f"{self.name}:enter")
            entered[self.name].set()
            await gate.wait()
            calls.append(f"{self.name}:ready")
            return self

        async def __aexit__(self, *args: object) -> None:
            calls.append(f"{self.name}:exit")

    tools = {"a": _SlowTool("a", fn_a), "b": _SlowTool("b", fn_b)}

    def fake_factory(cfg: MCPServerConfig) -> _SlowTool:
        return tools[cfg.name]

    config_a = MCPServerConfig(name="a", transport="stdio", command="python")
    config_b = MCPServerConfig(name="b", transport="stdio", command="python")

    async def progress(config: MCPServerConfig, state: str, current: int, total: int, failed: int) -> None:
        progress_events.append((config.name, state, current, total, failed))

    with patch("chrys.service.mcp.adapter._create_mcp_tool", side_effect=fake_factory):
        task = asyncio.create_task(adapter.connect_all([config_a, config_b], progress=progress))
        await asyncio.wait_for(
            asyncio.gather(entered["a"].wait(), entered["b"].wait()),
            timeout=20.0,
        )
        assert sorted(calls) == ["a:enter", "b:enter"]
        gate.set()
        result = await task

    assert [tool.name for tool in result] == ["a_remote", "b_remote"]
    assert all(get_tool_kind(tool) == "mcp" for tool in result)
    assert all(tool.kind is None for tool in result)
    starting = [event for event in progress_events if event[1] == "starting"]
    connected = [event for event in progress_events if event[1] == "connected"]
    assert sorted((current, total, failed) for _, _, current, total, failed in starting) == [(0, 2, 0), (0, 2, 0)]
    assert sorted(current for _, _, current, _, _ in connected) == [1, 2]
    await adapter.disconnect_all()


async def test_connect_all_progress_tracks_connected_and_failed_counts() -> None:
    adapter = MCPAdapter()
    bad = MCPServerConfig(name="bad", transport="stdio", command="python")
    good = MCPServerConfig(name="good", transport="stdio", command="python")
    progress_events: list[tuple[str, str, int, int, int]] = []

    async def fake_connect(config: MCPServerConfig) -> list[str]:
        if config.name == "bad":
            raise MCPConnectionError(config.name, config.transport, RuntimeError("boom"))
        return [f"tool-{config.name}"]

    async def progress(config: MCPServerConfig, state: str, current: int, total: int, failed: int) -> None:
        progress_events.append((config.name, state, current, total, failed))

    with patch.object(adapter, "connect", new=AsyncMock(side_effect=fake_connect)):
        tools = await adapter.connect_all([bad, good], progress=progress)

    assert tools == ["tool-good"]
    assert ("bad", "failed", 0, 2, 1) in progress_events
    assert ("good", "connected", 1, 2, 1) in progress_events


async def test_http_connect_to_stopped_server_fails_without_hanging() -> None:
    """Connect to a closed local port must surface as a failure, not a hang.

    The wait_for bound is intentionally generous: Windows' IOCP backend does
    not abort an in-flight connect as crisply as Linux on cancellation, so
    closed-port flows can take a couple of seconds longer than the configured
    ``request_timeout`` to fully unwind.  Linux/macOS fail in well under a
    second; the bound just keeps Windows from flaking.  The semantic check is
    "doesn't hang for many seconds" — exact timing is covered by
    ``test_http_connect_to_silent_server_fails_without_hanging``.
    """
    adapter = MCPAdapter()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

        config = MCPServerConfig(
            name="dead-http",
            transport="http",
            url=f"http://127.0.0.1:{port}/mcp",
            request_timeout=1,
        )

        # Keep the port reserved but deliberately not listening: connection
        # attempts are refused while no concurrent worker can claim it.
        tools = await asyncio.wait_for(adapter.connect_all([config]), timeout=10)

    assert tools == []
    assert "dead-http" in adapter.failures


async def test_http_connect_to_silent_server_fails_without_hanging() -> None:
    """Silent server must surface as a normalized timeout within ``request_timeout``.

    Unlike the closed-port case (which depends on OS connect-refused timing),
    this path goes through our explicit ``Timeout(post_timeout)`` on the
    streamed POST, so the timing is fully under our control.  We assert
    elapsed time stays under a small ceiling so a regression where the POST
    timeout silently widens (e.g. losing the per-request timeout override
    and falling back to the SSE read default) fails the test instead of
    passing under a generous ``wait_for`` bound.
    """
    adapter = MCPAdapter()
    stop = asyncio.Event()
    writers: list[asyncio.StreamWriter] = []

    async def handle_connection(_reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        writers.append(writer)
        await stop.wait()

    server = await asyncio.start_server(handle_connection, "127.0.0.1", 0)
    try:
        port = server.sockets[0].getsockname()[1]
        config = MCPServerConfig(
            name="silent-http",
            transport="http",
            url=f"http://127.0.0.1:{port}/mcp",
            bypass_proxy=True,
            request_timeout=1,
        )

        start = time.monotonic()
        tools = await asyncio.wait_for(adapter.connect_all([config]), timeout=10)
        elapsed = time.monotonic() - start

        assert tools == []
        assert "silent-http" in adapter.failures
        err = adapter.failures["silent-http"]
        assert isinstance(err.cause, TimeoutError)
        assert "connection did not complete within 1s" in str(err.cause)
        # request_timeout=1 + cleanup overhead; even on slow Windows CI this
        # should land well under 5s when the explicit POST timeout fires.
        assert elapsed < 5.0, f"silent-server connect took {elapsed:.2f}s, expected < 5s"
    finally:
        stop.set()
        server.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(server.wait_closed(), timeout=0.5)
        for writer in writers:
            writer.close()
        for writer in writers:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(writer.wait_closed(), timeout=0.5)


async def test_disconnect_all_cancels_in_flight_connect_and_closes_partial_tool() -> None:
    adapter = MCPAdapter()
    entered = asyncio.Event()
    exited = asyncio.Event()

    class _HangingTool:
        def __init__(self) -> None:
            self.functions: list = []
            self.request_timeout = DEFAULT_CONNECT_TIMEOUT_SECONDS
            self.dropped_banner_lines: list[str] = []

        async def __aenter__(self) -> _HangingTool:
            entered.set()
            await asyncio.Event().wait()
            return self

        async def __aexit__(self, *args: object) -> None:
            exited.set()

    config = MCPServerConfig(name="slow", transport="stdio", command="python")

    with patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=_HangingTool()):
        task = asyncio.create_task(adapter.connect_all([config]))
        await asyncio.wait_for(entered.wait(), timeout=20.0)
        await asyncio.wait_for(adapter.disconnect_all(), timeout=5.0)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2.0)
    assert exited.is_set()
    assert adapter.server_names == []


async def test_connect_all_propagates_cancelled_error() -> None:
    """``CancelledError`` is structural and must propagate out of ``connect_all``."""
    adapter = MCPAdapter()
    bad = MCPServerConfig(name="bad", transport="stdio", command="python")

    async def fake_connect(cfg: MCPServerConfig) -> list:
        raise asyncio.CancelledError

    with (
        patch.object(adapter, "connect", new=AsyncMock(side_effect=fake_connect)),
        pytest.raises(asyncio.CancelledError),
    ):
        await adapter.connect_all([bad])


async def test_connect_new_releases_duplicate_lease_if_server_already_registered() -> None:
    """If registration was won elsewhere, the duplicate cache lease must release.

    The duplicate branch reads ``existing_server.functions`` — engine-domain
    native instances — so it must clone-deliver (N5 invariant) instead of
    returning the originals.
    """
    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python")
    existing_fn = _function_tool("existing")
    existing = _FakeTool(functions=[existing_fn])
    duplicate_exit = AsyncMock(return_value=None)

    class _DuplicateTool:
        def __init__(self) -> None:
            self.functions = [_function_tool("duplicate")]
            self.request_timeout = DEFAULT_CONNECT_TIMEOUT_SECONDS
            self.dropped_banner_lines: list[str] = []

        async def __aenter__(self) -> _DuplicateTool:
            adapter._servers["s"] = existing  # type: ignore[assignment]
            return self

        async def __aexit__(self, *args: object) -> None:
            await duplicate_exit(*args)

    with patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=_DuplicateTool()):
        result = await adapter._connect_new(config)

    assert len(result) == 1
    delivered = result[0]
    assert delivered is not existing_fn
    assert isinstance(delivered, ChrysFunctionTool)
    assert delivered.name == "existing"
    duplicate_exit.assert_awaited_once_with(None, None, None)
    assert adapter.server_names == ["s"]


async def test_registration_revalidation_failure_releases_unregistered_lease() -> None:
    """A future await before registration must not turn a collision into a leaked lease."""
    config = MCPServerConfig(name="s", transport="stdio", command="python")
    function = _function_tool("remote")
    fake = _FakeTool(functions=[function])
    lease = MCPConnectionLease(
        key="key",
        server_name="s",
        config=config,
        functions=[function],
        mcp_tool=fake,  # type: ignore[arg-type]
    )
    cache = MagicMock()
    cache.acquire = AsyncMock(return_value=lease)
    cache.release = AsyncMock()
    adapter = MCPAdapter(cache=cache)
    collision = MCPToolNameCollisionError(
        "s",
        "stdio",
        conflicting_names={"remote"},
        conflict_with="another tool",
        guidance="Rename it.",
    )

    with (
        patch.object(adapter, "_validate_server_namespace", side_effect=[None, collision]),
        pytest.raises(MCPToolNameCollisionError, match="remote"),
    ):
        await adapter.connect(config)

    cache.release.assert_awaited_once_with(lease)
    assert adapter.server_names == []


async def test_disconnect_all_finishes_private_cache_close_when_cancelled() -> None:
    """Private adapters should not leave cached tools running after cancellation."""
    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python")
    close_started = asyncio.Event()
    close_continue = asyncio.Event()

    class _SlowExitTool:
        def __init__(self) -> None:
            self.functions = [_function_tool()]
            self.request_timeout = DEFAULT_CONNECT_TIMEOUT_SECONDS
            self.dropped_banner_lines: list[str] = []
            self.exit_count = 0

        async def __aenter__(self) -> _SlowExitTool:
            return self

        async def __aexit__(self, *args: object) -> None:
            close_started.set()
            await close_continue.wait()
            self.exit_count += 1

    fake = _SlowExitTool()

    with patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=fake):
        await adapter.connect(config)

    task = asyncio.create_task(adapter.disconnect_all())
    await asyncio.wait_for(close_started.wait(), timeout=20.0)
    task.cancel()
    close_continue.set()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert fake.exit_count == 1
    assert adapter.server_names == []
    assert adapter._cache.closed is False


# ---------------------------------------------------------------------------
# test_connection
# ---------------------------------------------------------------------------


async def test_test_connection_enters_and_exits_tool() -> None:
    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python")
    fake = _FakeTool()

    with patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=fake):
        await adapter.test_connection(config)

    fake.__aenter__.assert_awaited_once()
    fake.__aexit__.assert_awaited_once_with(None, None, None)


async def test_test_connection_returns_server_report() -> None:
    """A successful test snapshots identity, capabilities, and the catalog."""
    from mcp import types as mcp_types

    owned_tool = MCPTool(name="s")
    await _load_fake_remote_tools(owned_tool, _mcp_remote_tool("echo"), _mcp_remote_tool("greet"))
    # Mark "greet" as prompt-derived so the report splits it out of the tools list.
    owned_tool._loaded_prompt_remote_names = {"greet"}

    fake = _FakeTool(functions=owned_tool.functions)
    fake._server_info = mcp_types.Implementation(name="everything", title="Everything Server", version="1.0.0")
    fake._protocol_version = "2025-06-18"
    fake._server_capabilities = mcp_types.ServerCapabilities(
        tools=mcp_types.ToolsCapability(listChanged=True),
        prompts=mcp_types.PromptsCapability(),
    )
    fake._server_instructions = "Use the echo tool."
    fake._loaded_prompt_remote_names = owned_tool._loaded_prompt_remote_names

    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python")
    with patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=fake):
        report = await adapter.test_connection(config)

    assert report.server_name == "everything"
    assert report.server_title == "Everything Server"
    assert report.server_version == "1.0.0"
    assert report.protocol_version == "2025-06-18"
    assert report.capabilities == ("prompts", "tools", "tools.listChanged")
    assert report.instructions == "Use the echo tool."
    assert report.tools == (("echo", "Remote tool"),)
    assert report.prompts == (("greet", "Remote tool"),)
    assert report.initial_tool_names is None


def test_flatten_server_capabilities_walks_nested_feature_trees() -> None:
    """Nested capability objects (tasks, extras) flatten to dotted presence rows."""
    from mcp import types as mcp_types

    from chrys.service.mcp.adapter import _flatten_server_capabilities

    caps = mcp_types.ServerCapabilities(
        tools=mcp_types.ToolsCapability(listChanged=True),
        tasks={"list": {}, "cancel": {}, "requests": {"tools": {}}},
    )

    assert _flatten_server_capabilities(caps) == (
        "tasks",
        "tasks.cancel",
        "tasks.list",
        "tasks.requests",
        "tasks.requests.tools",
        "tools",
        "tools.listChanged",
    )


def test_flatten_server_capabilities_cap_never_starves_standard_groups() -> None:
    """A bloated extras branch must not make a declared standard capability
    look unadvertised: the entry cap only drops the deepest/extra leaves."""
    from mcp import types as mcp_types

    from chrys.service.mcp.adapter import _MAX_CAPABILITY_ENTRIES, _flatten_server_capabilities

    caps = mcp_types.ServerCapabilities(
        experimental={f"e{i:03d}": {} for i in range(_MAX_CAPABILITY_ENTRIES + 50)},
        tools=mcp_types.ToolsCapability(listChanged=True),
    )

    flattened = _flatten_server_capabilities(caps)

    assert len(flattened) == _MAX_CAPABILITY_ENTRIES
    assert "tools" in flattened


async def test_test_connection_reports_progressive_initial_surface() -> None:
    """Progressive configs report the initial visible surface by name."""
    owned_tool = MCPTool(name="s")
    await _load_fake_remote_tools(owned_tool, _mcp_remote_tool("echo"), _mcp_remote_tool("search"))
    fake = _FakeTool(functions=owned_tool.functions)

    adapter = MCPAdapter()
    config = MCPServerConfig(
        name="s",
        transport="stdio",
        command="python",
        use_progressive_disclosure=True,
        always_load=["search"],
    )
    with patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=fake):
        report = await adapter.test_connection(config)

    assert report.initial_tool_names is not None
    assert "search" in report.initial_tool_names
    assert "echo" not in report.initial_tool_names
    # Server-scoped control tools are part of the initial surface.
    assert "mcp_s_list_mcp_tools" in report.initial_tool_names


async def test_test_connection_wraps_aenter_error_in_mcp_connection_error() -> None:
    """``test_connection`` wraps any non-cancellation failure as ``MCPConnectionError``."""
    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python", request_timeout=5)
    fake = _FakeTool(enter_error=RuntimeError("boom"))

    with (
        patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=fake),
        pytest.raises(MCPConnectionError) as info,
    ):
        await adapter.test_connection(config)

    assert isinstance(info.value.cause, RuntimeError)
    assert "boom" in str(info.value.cause)
    fake.__aenter__.assert_awaited_once()
    fake.__aexit__.assert_awaited_once_with(None, None, None)


async def test_test_connection_wraps_missing_env_template(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHRYS_MCP_TEST_MISSING", raising=False)
    adapter = MCPAdapter()
    config = MCPServerConfig(
        name="s",
        transport="stdio",
        command="python",
        env={"TOKEN": "{{CHRYS_MCP_TEST_MISSING}}"},
    )

    with pytest.raises(MCPConnectionError) as info:
        await adapter.test_connection(config)

    message = str(info.value)
    assert "CHRYS_MCP_TEST_MISSING" in message
    assert "MCP server 's' env['TOKEN']" in message
    assert isinstance(info.value.cause, EnvVarResolutionError)


async def test_test_connection_reraises_mcp_connection_error() -> None:
    """A pre-wrapped MCPConnectionError is preserved exactly."""
    adapter = MCPAdapter()
    config = MCPServerConfig(name="s", transport="stdio", command="python")
    original = MCPConnectionError("s", "stdio", RuntimeError("wrapped"))
    fake = _FakeTool(enter_error=original)

    with (
        patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=fake),
        pytest.raises(MCPConnectionError) as info,
    ):
        await adapter.test_connection(config)

    assert info.value is original
    fake.__aexit__.assert_awaited_once_with(None, None, None)


async def test_test_connection_timeout_cause_is_wrapped() -> None:
    """One-shot tests use the same user-facing timeout cause as connect()."""
    adapter = MCPAdapter()
    config = MCPServerConfig(name="hung", transport="stdio", command="python", request_timeout=4)

    class _TimeoutTool:
        def __init__(self) -> None:
            self.functions: list = []
            self.request_timeout = 4
            self.dropped_banner_lines: list[str] = []

        async def __aenter__(self) -> object:
            raise TimeoutError("read timed out")

        async def __aexit__(self, *args: object) -> None:
            return None

    with (
        patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=_TimeoutTool()),
        pytest.raises(MCPConnectionError) as info,
    ):
        await adapter.test_connection(config)

    assert isinstance(info.value.cause, TimeoutError)
    assert "did not complete within 4s" in str(info.value.cause)


# ---------------------------------------------------------------------------
# disconnect_all — runs in parallel
# ---------------------------------------------------------------------------


async def test_disconnect_all_runs_in_parallel() -> None:
    adapter = MCPAdapter()
    gate = asyncio.Event()
    # Each __aexit__ signals its own "entered" barrier before awaiting the
    # shared gate.  Waiting on both barriers (rather than spinning the loop
    # with ``sleep(0)``) proves the two exits are running concurrently
    # without depending on scheduler step count.
    entered: dict[str, asyncio.Event] = {"a": asyncio.Event(), "b": asyncio.Event()}
    calls: list[str] = []

    class _SlowTool:
        def __init__(self, name: str) -> None:
            self.name = name

        async def __aexit__(self, *args: object) -> None:
            calls.append(f"{self.name}:enter")
            entered[self.name].set()
            await gate.wait()
            calls.append(f"{self.name}:exit")

    adapter._servers = {"a": _SlowTool("a"), "b": _SlowTool("b")}  # type: ignore[dict-item]

    task = asyncio.create_task(adapter.disconnect_all())
    # Both __aexit__ coroutines must be in flight before either is released.
    await asyncio.wait_for(
        asyncio.gather(entered["a"].wait(), entered["b"].wait()),
        timeout=20.0,
    )
    assert sorted(calls) == ["a:enter", "b:enter"]

    gate.set()
    await task
    assert sorted(calls) == ["a:enter", "a:exit", "b:enter", "b:exit"]
    assert adapter.server_names == []


async def test_disconnect_all_tolerates_individual_failures() -> None:
    adapter = MCPAdapter()

    class _BadTool:
        async def __aexit__(self, *args: object) -> None:
            raise RuntimeError("bad")

    class _GoodTool:
        def __init__(self) -> None:
            self.exited = False

        async def __aexit__(self, *args: object) -> None:
            self.exited = True

    good = _GoodTool()
    adapter._servers = {"bad": _BadTool(), "good": good}  # type: ignore[dict-item]

    await adapter.disconnect_all()
    assert good.exited is True
    assert adapter.server_names == []


async def test_disconnect_cancels_in_flight_connect() -> None:
    adapter = MCPAdapter()
    task = asyncio.create_task(asyncio.Event().wait())
    adapter._connecting["slow"] = task

    await adapter.disconnect("slow")

    assert task.cancelled()
    assert adapter.server_names == []


async def test_disconnect_tolerates_server_exit_failure() -> None:
    adapter = MCPAdapter()

    class _BadTool:
        async def __aexit__(self, *args: object) -> None:
            raise RuntimeError("exit failed")

    adapter._servers["bad"] = _BadTool()  # type: ignore[assignment]

    await adapter.disconnect("bad")

    assert adapter.server_names == []


# ---------------------------------------------------------------------------
# _SafeStdioTool — captures stderr without inheriting the TUI's descriptor
# ---------------------------------------------------------------------------


class TestSafeStdioToolErrlog:
    """Verify ``_SafeStdioTool`` keeps a safe stock-client fallback errlog.

    Upstream's default of ``errlog=sys.stderr`` has two failure modes that
    an earlier ``_safe_errlog`` probe could not cover reliably:

    1. Under the Chrys TUI on Python 3.14+, ``sys.stderr.fileno()`` may
       return an fd that *passes* the probe but is not usable for subprocess
       inheritance — ``anyio.open_process`` then raises
       ``OSError(9, "Bad file descriptor")`` and the server never starts.
    2. Even when the fd IS valid, inheriting it corrupts the Chrys TUI frame.

    The custom transport captures stderr through a pipe.  The stock-client
    compatibility fallback still needs a fresh ``os.devnull`` handle so an
    SDK-private API change cannot reintroduce the TUI descriptor issue.
    """

    def test_devnull_errlog_returns_open_file(self) -> None:
        """``_devnull_errlog`` returns a live, writable handle to devnull."""
        tool = _SafeStdioTool(name="t", command="python")
        result = tool._devnull_errlog()

        assert hasattr(result, "write")
        assert hasattr(result, "fileno")
        assert not result.closed
        # Must be a real OS fd so subprocess can inherit it.
        assert result.fileno() >= 0
        assert result is not sys.stderr
        result.close()

    def test_devnull_errlog_ignores_stderr_state(self) -> None:
        """A broken ``sys.stderr`` must not affect the result — we never probe it."""
        tool = _SafeStdioTool(name="t", command="python")

        class _BrokenStderr:
            def fileno(self) -> int:
                raise OSError(9, "Bad file descriptor")

        with patch.object(sys, "stderr", _BrokenStderr()):
            result = tool._devnull_errlog()

        assert result is not sys.stderr
        assert not result.closed
        assert result.fileno() >= 0
        result.close()

    def test_reuses_devnull_handle(self) -> None:
        """Multiple calls return the same cached open handle."""
        tool = _SafeStdioTool(name="t", command="python")

        first = tool._devnull_errlog()
        second = tool._devnull_errlog()

        assert first is second
        assert not first.closed
        first.close()

    def test_reopens_if_previous_handle_closed(self) -> None:
        """If the cached devnull handle was closed, a new one is opened."""
        tool = _SafeStdioTool(name="t", command="python")

        first = tool._devnull_errlog()
        first.close()
        second = tool._devnull_errlog()

        assert second is not first
        assert not second.closed
        second.close()

    def test_get_mcp_client_passes_devnull_errlog(self) -> None:
        """``get_mcp_client`` passes the cached devnull handle to the tolerant client."""
        tool = _SafeStdioTool(name="t", command="python")

        with patch("chrys.service.mcp.adapter.tolerant_stdio_client") as mock_client:
            mock_client.return_value = AsyncMock()
            tool.get_mcp_client()
            mock_client.assert_called_once()
            _, kwargs = mock_client.call_args
            assert "errlog" in kwargs
            # The errlog is the cached handle, not sys.stderr.
            assert kwargs["errlog"] is tool._errlog_file
            assert kwargs["errlog"] is not sys.stderr
            # And the dropped-banner buffer travels along.
            assert kwargs["dropped_banner_lines"] is tool.dropped_banner_lines
            assert kwargs["process_diagnostics"] is tool._process_diagnostics
        # Clean up the lazily-opened handle so pytest doesn't warn.
        if tool._errlog_file is not None:
            tool._errlog_file.close()

    def test_get_mcp_client_forwards_encoding_and_client_kwargs(self) -> None:
        """Encoding plus framework client kwargs must reach StdioServerParameters."""
        tool = _SafeStdioTool(name="t", command="python", encoding="utf-16", cwd="/tmp/chrys-mcp")

        with patch("chrys.service.mcp.adapter.tolerant_stdio_client") as mock_client:
            mock_client.return_value = AsyncMock()
            tool.get_mcp_client()

        _, kwargs = mock_client.call_args
        server = kwargs["server"]
        assert server.encoding == "utf-16"
        assert server.cwd == "/tmp/chrys-mcp"
        if tool._errlog_file is not None:
            tool._errlog_file.close()

    async def test_aexit_closes_devnull_file(self) -> None:
        """``__aexit__`` cleans up the devnull file handle."""
        tool = _SafeStdioTool(name="t", command="python")
        tool._errlog_file = open(os.devnull, "w")  # noqa: SIM115, ASYNC230

        with patch.object(MCPStdioTool, "__aexit__", new=AsyncMock()):
            await tool.__aexit__(None, None, None)

        assert tool._errlog_file is None

    async def test_aexit_without_errlog_file(self) -> None:
        """``__aexit__`` succeeds when no devnull file was opened."""
        tool = _SafeStdioTool(name="t", command="python")

        with patch.object(MCPStdioTool, "__aexit__", new=AsyncMock()):
            await tool.__aexit__(None, None, None)

        assert tool._errlog_file is None


# ---------------------------------------------------------------------------
# MCPConnectionError — banner diagnostics in the error message
# ---------------------------------------------------------------------------


class TestMCPConnectionErrorMessage:
    """Verify bounded stdio process diagnostics surface in the error string."""

    def test_no_banner_lines_keeps_message_terse(self) -> None:
        err = MCPConnectionError("srv", "stdio", RuntimeError("boom"))
        assert err.banner_lines == []
        assert "boom" in str(err)
        assert "non-JSON" not in str(err)

    def test_empty_cause_message_uses_exception_label(self) -> None:
        class ReadTimeout(Exception):
            pass

        err = MCPConnectionError("srv", "http", ReadTimeout(TimeoutError()))
        assert "Read timed out (ReadTimeout)" in str(err)

    def test_banner_lines_appear_in_message(self) -> None:
        err = MCPConnectionError(
            "srv",
            "stdio",
            TimeoutError("timed out"),
            banner_lines=["FooBarServer v1.0", "Initializing..."],
        )
        text = str(err)
        assert "non-JSON output before initialization" in text
        assert "FooBarServer v1.0" in text
        assert "Initializing..." in text

    def test_banner_lines_default_empty_and_independent(self) -> None:
        # Mutating the input list after construction should not affect the error.
        lines = ["a"]
        err = MCPConnectionError("srv", "stdio", RuntimeError("x"), banner_lines=lines)
        lines.append("b")
        assert err.banner_lines == ["a"]

    def test_stderr_and_process_exit_code_appear_in_message(self) -> None:
        err = MCPConnectionError(
            "srv",
            "stdio",
            RuntimeError("Connection closed"),
            stderr_tail="Traceback...\nModuleNotFoundError: missing_pkg",
            stderr_dropped_bytes=128,
            process_exit_code=23,
        )

        text = str(err)
        assert err.process_exit_code == 23
        assert err.stderr_dropped_bytes == 128
        assert "Server process exit code: 23" in text
        assert "128 earlier bytes omitted" in text
        assert "ModuleNotFoundError: missing_pkg" in text

    def test_stderr_is_sanitized_and_bounded(self) -> None:
        err = MCPConnectionError(
            "srv",
            "stdio",
            stderr_tail="\x1b[31m" + ("x" * (MAX_STDIO_STDERR_BYTES_CAPTURED + 10)) + "\x1b[0m",
        )

        assert "\x1b" not in err.stderr_tail
        assert len(err.stderr_tail) == MAX_STDIO_STDERR_BYTES_CAPTURED

    def test_message_previews_only_the_last_stderr_lines(self) -> None:
        """The message (which feeds a toast) shows a few lines; the attribute keeps the whole tail."""
        lines = [f"line {i}" for i in range(MAX_STDIO_STDERR_PREVIEW_LINES + 5)]
        err = MCPConnectionError("srv", "stdio", stderr_tail="\n".join(lines))

        text = str(err)
        assert err.stderr_tail == "\n".join(lines)
        assert "line 0" not in text
        assert f"line {MAX_STDIO_STDERR_PREVIEW_LINES + 4}" in text
        assert "5 earlier lines omitted" in text
        assert text.count("\n  line ") == MAX_STDIO_STDERR_PREVIEW_LINES

    def test_message_caps_each_previewed_stderr_line(self) -> None:
        err = MCPConnectionError("srv", "stdio", stderr_tail="y" * 500)

        assert "y" * 500 not in str(err)
        assert ("y" * 200 + "…") in str(err)
        assert err.stderr_tail == "y" * 500

    def test_exit_code_zero_is_explained_as_early_normal_exit(self) -> None:
        err = MCPConnectionError("srv", "stdio", RuntimeError("Connection closed"), process_exit_code=0)

        assert err.process_exit_code == 0
        assert "exited normally (code 0) before completing the MCP handshake" in str(err)
        assert "exit code: 0" not in str(err)

    def test_executable_and_working_directory_appear_in_message(self) -> None:
        err = MCPConnectionError(
            "srv",
            "stdio",
            RuntimeError("Connection closed"),
            resolved_executable="/opt/homebrew/bin/uv",
            effective_cwd="/workspace/foo",
        )

        text = str(err)
        assert err.resolved_executable == "/opt/homebrew/bin/uv"
        assert err.effective_cwd == "/workspace/foo"
        assert "Executable: /opt/homebrew/bin/uv" in text
        assert "Working directory: /workspace/foo" in text

    def test_context_paths_are_display_safe_but_attributes_stay_raw(self) -> None:
        # Lone surrogate (os.fsdecode of undecodable bytes) + control chars that
        # could forge extra diagnostic lines + an oversized path.
        raw_exe = "/opt/bin/\udcff-uv\nExecutable: /forged"
        raw_cwd = "/w/" + "x" * (MAX_STDIO_PATH_DISPLAY_CHARS + 20)
        err = MCPConnectionError(
            "srv", "stdio", RuntimeError("Connection closed"), resolved_executable=raw_exe, effective_cwd=raw_cwd
        )

        text = str(err)
        text.encode("utf-8")  # message must be UTF-8 encodable end to end
        assert err.resolved_executable == raw_exe
        assert err.effective_cwd == raw_cwd
        assert "\udcff" not in text
        assert "\nExecutable: /forged" not in text  # embedded newline cannot forge a second line
        assert sum(line.startswith("Executable:") for line in text.splitlines()) == 1
        assert raw_cwd not in text
        assert "Working directory: /w/" + "x" * (MAX_STDIO_PATH_DISPLAY_CHARS - 4) + "…" in text

    def test_context_lines_are_omitted_when_unknown(self) -> None:
        text = str(MCPConnectionError("srv", "http", RuntimeError("boom")))

        assert "Executable:" not in text
        assert "Working directory:" not in text


# ---------------------------------------------------------------------------
# MCPAdapter timeouts — connect / test_connection (request_timeout injection)
# ---------------------------------------------------------------------------
#
# These verify the adapter injects a default timeout floor into the SDK's
# own ``request_timeout`` (= ``ClientSession.read_timeout_seconds``) when
# the user didn't configure one.  The SDK's ``anyio.fail_after`` then fires
# from inside the framework's lifecycle-owner task, which is the only safe
# place to cancel — see ``_create_tool_with_timeout_floor``'s docstring.
#
# We don't try to drive a real subprocess hang here (that lives in the
# integration tests).  These unit tests verify the *wiring*: the right
# timeout value reaches ``_create_mcp_tool``, and a synthetic
# timeout-style failure from ``__aenter__`` becomes a
# ``MCPConnectionError`` whose ``cause`` is a ``TimeoutError``.


def _capture_create_tool_calls() -> tuple[list[MCPServerConfig], Any]:
    """Spy on ``_create_mcp_tool`` while still returning a usable fake."""
    seen_configs: list[MCPServerConfig] = []

    def _record(cfg: MCPServerConfig) -> _FakeTool:
        seen_configs.append(cfg)
        return _FakeTool()

    return seen_configs, _record


async def test_connect_injects_default_timeout_when_user_did_not_set_one() -> None:
    """``connect`` patches the config so the SDK's request_timeout = the floor."""
    adapter = MCPAdapter()
    config = MCPServerConfig(name="srv", transport="stdio", command="python")
    seen, factory = _capture_create_tool_calls()

    with patch("chrys.service.mcp.adapter._create_mcp_tool", side_effect=factory):
        await adapter.connect(config)

    assert len(seen) == 1
    assert seen[0].request_timeout == DEFAULT_CONNECT_TIMEOUT_SECONDS


async def test_test_connection_injects_default_timeout_when_user_did_not_set_one() -> None:
    """``test_connection`` injects its default floor when unset."""
    adapter = MCPAdapter()
    config = MCPServerConfig(name="srv", transport="stdio", command="python")
    seen, factory = _capture_create_tool_calls()

    with patch("chrys.service.mcp.adapter._create_mcp_tool", side_effect=factory):
        await adapter.test_connection(config)

    assert len(seen) == 1
    assert seen[0].request_timeout == DEFAULT_TEST_TIMEOUT_SECONDS


async def test_connect_preserves_user_configured_request_timeout() -> None:
    """A user-set ``request_timeout`` is never overwritten by the floor."""
    adapter = MCPAdapter()
    config = MCPServerConfig(name="srv", transport="stdio", command="python", request_timeout=5)
    seen, factory = _capture_create_tool_calls()

    with patch("chrys.service.mcp.adapter._create_mcp_tool", side_effect=factory):
        await adapter.connect(config)

    assert len(seen) == 1
    assert seen[0].request_timeout == 5


async def test_connect_timeout_cause_is_recognized_and_wrapped() -> None:
    """A SDK-style ``TimeoutError`` from ``__aenter__`` becomes a ``cause=TimeoutError`` wrap."""
    adapter = MCPAdapter()
    config = MCPServerConfig(name="hung", transport="stdio", command="python", request_timeout=2)

    class _TimeoutFromSdkTool:
        def __init__(self) -> None:
            self.functions: list = []
            self.request_timeout = 2
            self.dropped_banner_lines = ["welcome"]

        async def __aenter__(self) -> object:
            raise TimeoutError("read_timeout fired inside session.initialize()")

        async def __aexit__(self, *args: object) -> None:
            return None

    with (
        patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=_TimeoutFromSdkTool()),
        pytest.raises(MCPConnectionError) as info,
    ):
        await adapter.connect(config)

    err = info.value
    assert err.server_name == "hung"
    assert isinstance(err.cause, TimeoutError)
    assert "did not complete within 2s" in str(err.cause)
    assert err.banner_lines == ["welcome"]
    assert "hung" in adapter.failures


async def test_connect_mcp_error_408_recognized_as_timeout() -> None:
    """The MCP SDK signals read-timeout via ``McpError(ErrorData(code=408))``.

    The adapter must recognise that and wrap as ``cause=TimeoutError``.
    """
    adapter = MCPAdapter()
    config = MCPServerConfig(name="hung", transport="stdio", command="python", request_timeout=3)

    class _ErrData:
        code = 408
        message = "Timed out"

    class _McpError(Exception):
        def __init__(self) -> None:
            self.error = _ErrData()
            super().__init__("Timed out")

    class _ToolRaisingMcpError:
        def __init__(self) -> None:
            self.functions: list = []
            self.request_timeout = 3
            self.dropped_banner_lines: list[str] = []

        async def __aenter__(self) -> object:
            raise _McpError()

        async def __aexit__(self, *args: object) -> None:
            return None

    with (
        patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=_ToolRaisingMcpError()),
        pytest.raises(MCPConnectionError) as info,
    ):
        await adapter.connect(config)

    assert isinstance(info.value.cause, TimeoutError)


async def test_connect_arbitrary_failure_keeps_original_cause() -> None:
    """Non-timeout failures keep the original exception as ``cause`` (not a TimeoutError)."""
    adapter = MCPAdapter()
    config = MCPServerConfig(name="srv", transport="stdio", command="python", request_timeout=5)

    class _BoomTool:
        def __init__(self) -> None:
            self.functions: list = []
            self.request_timeout = 5
            self.dropped_banner_lines = ["partial banner"]

        async def __aenter__(self) -> object:
            raise RuntimeError("spawn failed")

        async def __aexit__(self, *args: object) -> None:
            return None

    with (
        patch("chrys.service.mcp.adapter._create_mcp_tool", return_value=_BoomTool()),
        pytest.raises(MCPConnectionError) as info,
    ):
        await adapter.connect(config)

    assert isinstance(info.value.cause, RuntimeError)
    assert info.value.banner_lines == ["partial banner"]


async def test_connect_all_skips_failed_server_continues_with_others() -> None:
    """A failing server is recorded; the next one still connects."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    adapter = MCPAdapter(bus=bus, session_id="sess")

    bad = MCPServerConfig(name="bad", transport="stdio", command="python")
    good = MCPServerConfig(name="good", transport="stdio", command="python")

    good_fn = _function_tool("good_remote")

    class _BadTool:
        def __init__(self) -> None:
            self.functions: list = []
            self.request_timeout = DEFAULT_CONNECT_TIMEOUT_SECONDS
            self.dropped_banner_lines: list[str] = []

        async def __aenter__(self) -> object:
            raise TimeoutError("simulated SDK timeout")

        async def __aexit__(self, *args: object) -> None:
            return None

    class _GoodTool:
        def __init__(self) -> None:
            self.functions = [good_fn]
            self.request_timeout = DEFAULT_CONNECT_TIMEOUT_SECONDS

        async def __aenter__(self) -> _GoodTool:
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

    def fake_factory(cfg: MCPServerConfig) -> object:
        return _BadTool() if cfg.name == "bad" else _GoodTool()

    with patch("chrys.service.mcp.adapter._create_mcp_tool", side_effect=fake_factory):
        tools = await adapter.connect_all([bad, good])

    assert [tool.name for tool in tools] == ["good_remote"]
    assert tools[0] is not good_fn
    assert "good" in adapter.server_names
    assert "bad" in adapter.failures
    bus.publish.assert_awaited_once()  # warning for the bad server


# ---------------------------------------------------------------------------
# _inherited_stdio_environment — sanitized parent env passthrough for stdio MCP servers
# ---------------------------------------------------------------------------


class TestInheritedStdioEnvironment:
    """Stdio MCP subprocesses must inherit the user's sanitized env by default.

    The MCP SDK's ``get_default_environment()`` is a strict ~6-var
    allowlist that strips ``HTTPS_PROXY`` / ``NO_PROXY`` / ``SSL_CERT_FILE``
    / etc. — fine for an untrusted-server sandbox, wrong for a developer
    tool where ``uvx some-pkg`` is expected to use the same proxy that
    works in the user's shell.  Chrys forwards nearly all of ``os.environ``
    (minus bash function exports and Python runtime path overrides) and
    merges per-server overrides on top.
    """

    def test_inherits_proxy_and_tls_vars(self) -> None:
        """Vars stripped by the SDK allowlist must reach the child env."""
        with patch.dict(
            os.environ,
            {
                "HTTPS_PROXY": "http://proxy.corp.example:8080",
                "NO_PROXY": "localhost,.corp.example",
                "SSL_CERT_FILE": "/etc/ssl/corp.pem",
            },
            clear=False,
        ):
            env = _inherited_stdio_environment()

        assert env["HTTPS_PROXY"] == "http://proxy.corp.example:8080"
        assert env["NO_PROXY"] == "localhost,.corp.example"
        assert env["SSL_CERT_FILE"] == "/etc/ssl/corp.pem"

    def test_mirrors_parent_uppercase_no_proxy_to_lowercase(self) -> None:
        """Stdio children get lowercase ``no_proxy`` even when parent stores ``NO_PROXY``."""
        with patch.dict(os.environ, {"NO_PROXY": "localhost,.corp.example"}, clear=True):
            env = _inherited_stdio_environment()

        assert env["NO_PROXY"] == "localhost,.corp.example"
        assert env["no_proxy"] == "localhost,.corp.example"

    def test_mirrors_parent_lowercase_no_proxy_to_uppercase(self) -> None:
        """Stdio children get uppercase ``NO_PROXY`` even when parent stores ``no_proxy``."""
        with patch.dict(os.environ, {"no_proxy": "localhost,.corp.example"}, clear=True):
            env = _inherited_stdio_environment()

        assert env["NO_PROXY"] == "localhost,.corp.example"
        assert env["no_proxy"] == "localhost,.corp.example"

    def test_extra_overrides_inherited(self) -> None:
        """Per-server ``env`` wins on conflict with the parent process."""
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://parent:1"}, clear=False):
            env = _inherited_stdio_environment({"HTTPS_PROXY": "http://override:2"})

        assert env["HTTPS_PROXY"] == "http://override:2"

    def test_extra_uppercase_no_proxy_overrides_inherited_lowercase_alias(self) -> None:
        """A per-server ``NO_PROXY`` override also replaces inherited lowercase alias."""
        with patch.dict(os.environ, {"no_proxy": "stale-parent.example"}, clear=True):
            env = _inherited_stdio_environment({"NO_PROXY": "localhost"})

        assert env["NO_PROXY"] == "localhost"
        assert env["no_proxy"] == "localhost"

    def test_extra_lowercase_no_proxy_overrides_inherited_uppercase_alias(self) -> None:
        """A per-server ``no_proxy`` override also replaces inherited uppercase alias."""
        with patch.dict(os.environ, {"NO_PROXY": "stale-parent.example"}, clear=True):
            env = _inherited_stdio_environment({"no_proxy": "localhost"})

        assert env["NO_PROXY"] == "localhost"
        assert env["no_proxy"] == "localhost"

    def test_extra_adds_new_vars(self) -> None:
        """Extra entries not in the parent env appear in the result."""
        env = _inherited_stdio_environment({"CHRYS_TEST_NEW_VAR": "value"})
        assert env["CHRYS_TEST_NEW_VAR"] == "value"

    def test_strips_inherited_python_runtime_overrides(self) -> None:
        """Parent Python runtime path overrides must not leak into stdio MCP children."""
        with patch.dict(
            os.environ,
            {
                "PYTHONHOME": "/bad/home",
                "PYTHONPATH": "/bad/path",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
            },
            clear=True,
        ):
            env = _inherited_stdio_environment()

        env_keys = {key.upper() for key in env}
        assert "PYTHONHOME" not in env_keys
        assert "PYTHONPATH" not in env_keys
        assert env["PYTHONUTF8"] == "1"
        assert env["PYTHONIOENCODING"] == "utf-8"

    def test_extra_can_restore_python_runtime_overrides(self) -> None:
        """Per-server env is an explicit opt-in and wins over inherited cleanup."""
        with patch.dict(
            os.environ,
            {
                "PYTHONHOME": "/bad/home",
                "PYTHONPATH": "/bad/path",
            },
            clear=True,
        ):
            env = _inherited_stdio_environment({"PYTHONHOME": "/explicit/home", "PYTHONPATH": "/explicit/path"})

        assert env["PYTHONHOME"] == "/explicit/home"
        assert env["PYTHONPATH"] == "/explicit/path"

    def test_no_extra_returns_pure_parent(self) -> None:
        """``extra=None`` returns the inherited env unchanged (apart from filter)."""
        with patch.dict(os.environ, {"CHRYS_TEST_PASSTHROUGH": "ok"}, clear=False):
            env = _inherited_stdio_environment()

        assert env["CHRYS_TEST_PASSTHROUGH"] == "ok"

    def test_demotes_pyapp_runtime_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stdio MCP commands should resolve system executables before PyApp runtime shims."""
        runtime_bin = tmp_path / "runtime" / "bin"
        system_bin = tmp_path / "system" / "bin"
        monkeypatch.setattr(runtime_paths.sys, "executable", str(runtime_bin / "python"))
        monkeypatch.setattr(runtime_paths.sys, "prefix", str(tmp_path / "runtime"))
        monkeypatch.setattr(runtime_paths.sys, "exec_prefix", str(tmp_path / "runtime"))
        monkeypatch.setattr(
            runtime_paths.sysconfig,
            "get_path",
            lambda name: str(runtime_bin) if name == "scripts" else "",
        )
        with patch.dict(
            os.environ,
            {
                "PYAPP": "1",
                "PATH": os.pathsep.join([str(runtime_bin), str(system_bin)]),
            },
            clear=True,
        ):
            env = _inherited_stdio_environment()

        assert env["PATH"].split(os.pathsep) == [str(system_bin), str(runtime_bin)]

    def test_bash_function_exports_filtered(self) -> None:
        """Shellshock-style ``() {`` function exports must not be forwarded."""
        with patch.dict(
            os.environ,
            {
                "BASH_FUNC_x%%": "() {  echo hi\n}",
                "NORMAL_VAR": "kept",
            },
            clear=False,
        ):
            env = _inherited_stdio_environment()

        assert "BASH_FUNC_x%%" not in env
        assert env["NORMAL_VAR"] == "kept"

    def test_returned_dict_is_independent(self) -> None:
        """Mutating the result must not leak back into ``os.environ``."""
        env = _inherited_stdio_environment()
        env["CHRYS_TEST_LEAK_GUARD"] = "should_not_leak"
        assert "CHRYS_TEST_LEAK_GUARD" not in os.environ


# ---------------------------------------------------------------------------
# tolerant_stdio_client — process fakes
# ---------------------------------------------------------------------------


class _FakeByteReceiveStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def receive(self) -> bytes:
        if self._chunks:
            return self._chunks.pop(0)
        raise anyio.EndOfStream


class _FakeStdin:
    def __init__(self) -> None:
        self.sent: list[bytes] = []
        self.closed = False
        self.wrote = asyncio.Event()

    async def send(self, data: bytes) -> None:
        self.sent.append(data)
        self.wrote.set()

    async def aclose(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(
        self,
        stdout_chunks: list[bytes],
        *,
        stderr_chunks: list[bytes] | None = None,
        exit_code: int = 0,
    ) -> None:
        self.stdout = _FakeByteReceiveStream(stdout_chunks)
        self.stderr = _FakeByteReceiveStream(stderr_chunks or [])
        self.stdin = _FakeStdin()
        self.returncode = exit_code
        self.waited = False

    async def __aenter__(self) -> _FakeProcess:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def wait(self) -> int:
        self.waited = True
        return self.returncode


def _stdio_server_params() -> Any:
    from mcp.client.stdio import StdioServerParameters

    return StdioServerParameters(command="python", args=[], env=None)


@contextmanager
def _patched_stdio_spawn(process: object | None = None, *, error: BaseException | None = None) -> Iterator[None]:
    import mcp.client.stdio as stdio

    spawn = AsyncMock(side_effect=error) if error is not None else AsyncMock(return_value=process)
    with (
        patch.object(stdio, "_get_executable_command", return_value="/usr/bin/python"),
        patch.object(stdio, "_create_platform_compatible_process", new=spawn),
    ):
        yield


# ---------------------------------------------------------------------------
# tolerant_stdio_client — async transport behavior
# ---------------------------------------------------------------------------


async def test_tolerant_stdio_client_drops_banners_and_delivers_jsonrpc() -> None:
    """Exercise the real async stdout reader, not just the mirrored classifier."""
    process = _FakeProcess(
        [
            b"Plain startup banner\r\nSecond banner\n   \n",
            b'{"jsonrpc":"2.0","id":1,"method":"ping"}\n',
        ]
    )
    captured: list[str] = []

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params(), dropped_banner_lines=captured) as (
            read_stream,
            _write_stream,
        ):
            received = await asyncio.wait_for(read_stream.receive(), timeout=20.0)

    assert captured == ["Plain startup banner", "Second banner"]
    assert received.message.root.method == "ping"
    assert received.message.root.id == 1
    assert process.stdin.closed is True
    assert process.waited is True


async def test_tolerant_stdio_client_captures_bounded_stderr_tail_and_exit_code() -> None:
    """Stderr stays off the protocol stream while preserving actionable failure context."""
    stderr = b"x" * (MAX_STDIO_STDERR_BYTES_CAPTURED + 64) + b"\nModuleNotFoundError: missing_pkg\n"
    process = _FakeProcess([], stderr_chunks=[stderr], exit_code=23)
    diagnostics = _StdioProcessDiagnostics()

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params(), process_diagnostics=diagnostics):
            pass

    assert diagnostics.exit_code == 23
    assert diagnostics.stderr_dropped_bytes > 0
    assert len(diagnostics._stderr_tail) == MAX_STDIO_STDERR_BYTES_CAPTURED
    assert diagnostics.stderr_tail.endswith("ModuleNotFoundError: missing_pkg")
    # Spawn context is recorded up front so it survives an early exit.
    assert diagnostics.resolved_executable == "/usr/bin/python"
    assert diagnostics.effective_cwd == os.getcwd()  # no server cwd => inherited


def test_display_path_keeps_ordinary_paths_verbatim() -> None:
    assert _display_path("/opt/homebrew/bin/uv") == "/opt/homebrew/bin/uv"
    assert _display_path("C:\\Tools\\uv.exe") == "C:\\Tools\\uv.exe"
    assert _display_path("/w/ünïcode/路径") == "/w/ünïcode/路径"


class TestResolveSpawnExecutable:
    def test_command_with_path_component_is_used_as_is(self, tmp_path: Path) -> None:
        assert _resolve_spawn_executable("./bin/uv", {"PATH": str(tmp_path)}) == "./bin/uv"
        assert _resolve_spawn_executable(str(tmp_path / "uv"), {}) == str(tmp_path / "uv")

    def test_unresolvable_command_falls_back_to_raw_name(self, tmp_path: Path) -> None:
        assert _resolve_spawn_executable("definitely-not-a-real-cmd", {"PATH": str(tmp_path)}) == (
            "definitely-not-a-real-cmd"
        )

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable-bit PATH lookup")
    def test_bare_command_resolves_against_child_env_path_not_parent(self, tmp_path: Path, monkeypatch: Any) -> None:
        child_bin = tmp_path / "child-bin"
        child_bin.mkdir()
        exe = child_bin / "uv"
        exe.write_text("#!/bin/sh\n")
        exe.chmod(0o755)
        parent_bin = tmp_path / "parent-bin"
        parent_bin.mkdir()
        (parent_bin / "uv").write_text("#!/bin/sh\n")
        (parent_bin / "uv").chmod(0o755)
        monkeypatch.setenv("PATH", str(parent_bin))  # parent PATH must NOT win

        assert _resolve_spawn_executable("uv", {"PATH": str(child_bin)}) == str(exe)

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable-bit PATH lookup")
    def test_non_executable_candidate_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "uv").write_text("not executable\n")
        (tmp_path / "uv").chmod(0o644)

        assert _resolve_spawn_executable("uv", {"PATH": str(tmp_path)}) == "uv"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX executable-bit PATH lookup")
async def test_tolerant_stdio_client_resolves_bare_command_against_child_path(tmp_path: Path) -> None:
    """The diagnostics show *which* ``uv`` the child would exec, not the bare name."""
    import mcp.client.stdio as stdio
    from mcp.client.stdio import StdioServerParameters

    exe = tmp_path / "uv"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    process = _FakeProcess([])
    diagnostics = _StdioProcessDiagnostics()
    params = StdioServerParameters(command="uv", args=[], env={"PATH": str(tmp_path)})
    spawn = AsyncMock(return_value=process)

    with (
        patch.object(stdio, "_get_executable_command", side_effect=lambda cmd: cmd),
        patch.object(stdio, "_create_platform_compatible_process", new=spawn),
    ):
        async with tolerant_stdio_client(params, process_diagnostics=diagnostics, inherit_env=False):
            pass

    assert diagnostics.resolved_executable == str(exe)
    # The spawn itself still receives the configured command untouched.
    assert spawn.await_args is not None
    assert spawn.await_args.kwargs["command"] == "uv"


async def test_tolerant_stdio_client_records_configured_cwd(tmp_path: Path) -> None:
    from mcp.client.stdio import StdioServerParameters

    process = _FakeProcess([])
    diagnostics = _StdioProcessDiagnostics()
    params = StdioServerParameters(command="python", args=[], env=None, cwd=tmp_path)

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(params, process_diagnostics=diagnostics):
            pass

    assert diagnostics.effective_cwd == str(tmp_path)


async def test_tolerant_stdio_client_drains_delayed_stderr_before_process_stream_closes() -> None:
    """A final pipe chunk must be consumed before process.__aexit__ closes stderr."""

    class _DelayedStderr:
        def __init__(self) -> None:
            self.released = asyncio.Event()
            self.closed = False
            self.delivered = False

        async def receive(self) -> bytes:
            if self.delivered:
                raise anyio.EndOfStream
            await self.released.wait()
            await asyncio.sleep(0)
            if self.closed:
                raise anyio.ClosedResourceError
            self.delivered = True
            return b"RuntimeError: final buffered traceback line\n"

    class _DelayedStderrProcess(_FakeProcess):
        def __init__(self) -> None:
            super().__init__([])
            self.stderr = _DelayedStderr()

        async def wait(self) -> int:
            self.waited = True
            self.stderr.released.set()
            return self.returncode

        async def __aexit__(self, *args: object) -> None:
            self.stderr.closed = True

    process = _DelayedStderrProcess()
    diagnostics = _StdioProcessDiagnostics()

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params(), process_diagnostics=diagnostics):
            pass

    assert diagnostics.stderr_tail == "RuntimeError: final buffered traceback line"


async def test_tolerant_stdio_client_reads_windows_fallback_shaped_sync_stderr() -> None:
    """The MCP SDK's Windows FallbackProcess exposes a synchronous stderr file."""

    class _SyncStderr:
        def __init__(self) -> None:
            self.chunks = [b"fallback stderr line\n", b""]
            self.read1_calls = 0

        def read1(self, _size: int) -> bytes:
            # ``read1`` returns what is available instead of blocking until
            # ``size`` bytes arrive; the reader must prefer it when present.
            self.read1_calls += 1
            return self.chunks.pop(0)

        def read(self, _size: int) -> bytes:
            raise AssertionError("read1 must be preferred over read")

    process = _FakeProcess([], exit_code=5)
    stderr = _SyncStderr()
    process.stderr = stderr  # type: ignore[assignment]
    diagnostics = _StdioProcessDiagnostics()

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params(), process_diagnostics=diagnostics):
            pass

    assert diagnostics.stderr_tail == "fallback stderr line"
    assert diagnostics.exit_code == 5
    assert stderr.read1_calls == 2


async def test_tolerant_stdio_client_reads_windows_fallback_without_read1() -> None:
    class _ReadOnlyStderr:
        def __init__(self) -> None:
            self.chunks = [b"plain read line\n", b""]

        def read(self, _size: int) -> bytes:
            return self.chunks.pop(0)

    process = _FakeProcess([])
    process.stderr = _ReadOnlyStderr()  # type: ignore[assignment]
    diagnostics = _StdioProcessDiagnostics()

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params(), process_diagnostics=diagnostics):
            pass

    assert diagnostics.stderr_tail == "plain read line"


async def test_tolerant_stdio_client_fallback_reader_tolerates_non_coroutine_awaitable() -> None:
    """A ``read`` that returns a Future-like awaitable (no ``close``) must not crash the reader."""

    class _Awaitable:
        # Awaitable protocol only: no ``close`` (unlike coroutine objects).
        def __await__(self) -> Any:
            yield from ()
            return b"never read synchronously\n"

    class _FutureStderr:
        def read(self, _size: int) -> Any:
            return _Awaitable()

    process = _FakeProcess([])
    process.stderr = _FutureStderr()  # type: ignore[assignment]
    diagnostics = _StdioProcessDiagnostics()

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params(), process_diagnostics=diagnostics):
            pass

    assert diagnostics.stderr_tail == ""


async def test_tolerant_stdio_client_fallback_reader_closes_stray_coroutine() -> None:
    class _CoroutineStderr:
        def __init__(self) -> None:
            self.coroutines: list[Any] = []

        def read(self, _size: int) -> Any:
            async def _reader() -> bytes:
                return b"async read\n"

            coro = _reader()
            self.coroutines.append(coro)
            return coro

    process = _FakeProcess([])
    stderr = _CoroutineStderr()
    process.stderr = stderr  # type: ignore[assignment]
    diagnostics = _StdioProcessDiagnostics()

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params(), process_diagnostics=diagnostics):
            pass

    assert diagnostics.stderr_tail == ""
    assert len(stderr.coroutines) == 1
    assert stderr.coroutines[0].cr_frame is None  # closed, so no "never awaited" warning


async def test_tolerant_stdio_client_pushes_invalid_jsonrpc_exception() -> None:
    """Valid JSON that is not JSON-RPC stays a protocol error on the read stream."""
    process = _FakeProcess([b'{"hello":"world"}\n'])

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params()) as (read_stream, _write_stream):
            received = await asyncio.wait_for(read_stream.receive(), timeout=20.0)

    assert isinstance(received, Exception)


async def test_tolerant_stdio_client_serializes_stdin_messages() -> None:
    """Outbound SessionMessages should be JSON-lines encoded to process stdin."""
    import json

    from mcp.shared.message import SessionMessage
    from mcp.types import JSONRPCMessage, JSONRPCRequest

    process = _FakeProcess([])
    request = JSONRPCRequest(jsonrpc="2.0", id=9, method="tools/list")

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params()) as (_read_stream, write_stream):
            await write_stream.send(SessionMessage(message=JSONRPCMessage(request)))
            await asyncio.wait_for(process.stdin.wrote.wait(), timeout=20.0)

    assert len(process.stdin.sent) == 1
    assert process.stdin.sent[0].endswith(b"\n")
    assert json.loads(process.stdin.sent[0]) == {"jsonrpc": "2.0", "id": 9, "method": "tools/list"}


async def test_tolerant_stdio_client_closes_streams_when_process_spawn_fails() -> None:
    """Spawn OSError should propagate after the local memory streams are closed."""
    with (
        _patched_stdio_spawn(error=OSError("spawn failed")),
        pytest.raises(OSError, match="spawn failed"),
    ):
        async with tolerant_stdio_client(_stdio_server_params()):
            pass


async def test_tolerant_stdio_client_terminates_process_tree_when_wait_times_out() -> None:
    import mcp.client.stdio as stdio

    class _TimeoutWaitProcess(_FakeProcess):
        def __init__(self) -> None:
            super().__init__([])
            self.wait_attempts = 0

        async def wait(self) -> int:
            self.wait_attempts += 1
            if self.wait_attempts == 1:
                raise TimeoutError
            self.returncode = -9
            return self.returncode

    process = _TimeoutWaitProcess()
    terminate = AsyncMock()
    diagnostics = _StdioProcessDiagnostics()

    with _patched_stdio_spawn(process), patch.object(stdio, "_terminate_process_tree", new=terminate):
        async with tolerant_stdio_client(_stdio_server_params(), process_diagnostics=diagnostics):
            pass

    terminate.assert_awaited_once_with(process)
    assert process.wait_attempts == 2
    assert diagnostics.exit_code == -9


async def test_tolerant_stdio_client_ignores_process_lookup_during_shutdown() -> None:

    class _GoneProcess(_FakeProcess):
        async def wait(self) -> None:
            raise ProcessLookupError

    process = _GoneProcess([])

    with _patched_stdio_spawn(process):
        async with tolerant_stdio_client(_stdio_server_params()):
            pass

    assert process.stdin.closed is True


def _stdio_private_import_blocker() -> Any:
    """An ``__import__`` replacement simulating the SDK dropping the private
    stdio primitives ``tolerant_stdio_client`` imports — forcing the stock-client
    fallback — while leaving every other import working."""
    import builtins

    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "mcp.client.stdio" and "_create_platform_compatible_process" in fromlist:
            raise ImportError("stdio internals moved")
        return real_import(name, globals_, locals_, fromlist, level)

    return guarded_import


def _session_message_import_blocker() -> Any:
    """An ``__import__`` replacement simulating the SDK moving ``SessionMessage``."""
    import builtins

    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "mcp.shared.message" and "SessionMessage" in fromlist:
            raise ImportError("session message moved")
        return real_import(name, globals_, locals_, fromlist, level)

    return guarded_import


async def test_tolerant_stdio_falls_back_to_stock_client_when_privates_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the SDK stdio privates vanish, delegate to the stock ``stdio_client``.

    Banner tolerance is lost, but every stdio MCP connection keeps working and
    the sanitized inherited environment is still forwarded (re-injected onto
    the server params via ``model_copy``), not reduced to the SDK's ~6-var allowlist.
    """
    import builtins

    import mcp.client.stdio as stdio
    from mcp.client.stdio import StdioServerParameters

    monkeypatch.setenv("CHRYS_FALLBACK_PROBE", "present")
    received: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_stock(server: Any, errlog: Any = sys.stderr) -> Any:
        received["server"] = server
        received["errlog"] = errlog
        yield ("stock-read", "stock-write")

    server = StdioServerParameters(command="python", args=["-m", "srv"], env={"CHRYS_TEST_ONLY": "1"})

    with (
        patch.object(builtins, "__import__", side_effect=_stdio_private_import_blocker()),
        patch.object(stdio, "stdio_client", fake_stock),
    ):
        async with tolerant_stdio_client(server, inherit_env=True) as streams:
            assert streams == ("stock-read", "stock-write")

    forwarded = received["server"]
    assert forwarded.env == _inherited_stdio_environment({"CHRYS_TEST_ONLY": "1"})
    assert forwarded.env["CHRYS_TEST_ONLY"] == "1"
    assert forwarded.env["CHRYS_FALLBACK_PROBE"] == "present"


async def test_tolerant_stdio_falls_back_to_stock_client_when_session_message_moves() -> None:
    """``SessionMessage`` is part of the patched reader; if it moves, use stock stdio."""
    import builtins

    import mcp.client.stdio as stdio
    from mcp.client.stdio import StdioServerParameters

    received: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_stock(server: Any, errlog: Any = sys.stderr) -> Any:
        received["server"] = server
        yield ("stock-read", "stock-write")

    server = StdioServerParameters(command="python", env={"ONLY": "this"})

    with (
        patch.object(builtins, "__import__", side_effect=_session_message_import_blocker()),
        patch.object(stdio, "stdio_client", fake_stock),
    ):
        async with tolerant_stdio_client(server, inherit_env=False) as streams:
            assert streams == ("stock-read", "stock-write")

    assert received["server"] is server


async def test_tolerant_stdio_stock_fallback_passes_server_unchanged_when_not_inheriting() -> None:
    """With ``inherit_env=False`` (cache path; env already complete) the stock
    client receives the original server params unchanged."""
    import builtins

    import mcp.client.stdio as stdio
    from mcp.client.stdio import StdioServerParameters

    received: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_stock(server: Any, errlog: Any = sys.stderr) -> Any:
        received["server"] = server
        yield ("r", "w")

    server = StdioServerParameters(command="python", env={"ONLY": "this"})

    with (
        patch.object(builtins, "__import__", side_effect=_stdio_private_import_blocker()),
        patch.object(stdio, "stdio_client", fake_stock),
    ):
        async with tolerant_stdio_client(server, inherit_env=False) as streams:
            assert streams == ("r", "w")

    assert received["server"] is server
    assert received["server"].env == {"ONLY": "this"}


# ---------------------------------------------------------------------------
# tolerant_stdio_client — line classification
# ---------------------------------------------------------------------------


class TestTolerantStdioClassification:
    """Pure unit-test of the three-bucket classifier in ``tolerant_stdio_client``.

    We don't spawn a real subprocess here — the line-classification logic is
    self-contained and deterministic, and we want fast/portable tests.  Real
    subprocess behavior is covered by the integration tests in
    ``test_mcp_integration.py``.
    """

    @staticmethod
    def _classify(line: str) -> str:
        """Mirror the bucket logic in ``tolerant_stdio_client.stdout_reader``.

        Returns one of: 'empty', 'banner', 'json-invalid', 'json'.  This
        tracks the production three-bucket scheme exactly; if the
        production code ever drifts, this mirror has to be updated and
        the tests below will tell us how.
        """
        import json as _json

        from mcp import types

        if not line.lstrip():
            return "empty"
        try:
            parsed = _json.loads(line)
        except _json.JSONDecodeError:
            return "banner"
        try:
            types.JSONRPCMessage.model_validate(parsed)
        except Exception:
            return "json-invalid"
        return "json"

    def test_empty_line_is_empty(self) -> None:
        assert self._classify("") == "empty"
        assert self._classify("   ") == "empty"
        assert self._classify("\t  \r") == "empty"

    def test_plaintext_banner_is_banner(self) -> None:
        assert self._classify("FooBarServer v1.0 (build 42)") == "banner"
        assert self._classify("Initializing backend...") == "banner"

    def test_bracketed_log_line_is_banner(self) -> None:
        # ``[INFO] starting`` is not valid JSON, so it's a banner — not a
        # protocol error.  This is the difference between our classifier
        # and a naive ``startswith('{', '[')`` check.
        assert self._classify("[INFO] starting") == "banner"

    def test_json_rpc_message_is_json(self) -> None:
        line = '{"jsonrpc":"2.0","id":1,"method":"ping"}'
        assert self._classify(line) == "json"

    def test_valid_json_but_not_jsonrpc_is_invalid(self) -> None:
        # Parses as JSON object but doesn't match the JSON-RPC schema.
        assert self._classify('{"hello": "world"}') == "json-invalid"

    def test_malformed_object_is_banner(self) -> None:
        # Doesn't even parse as JSON, so it's banner not protocol error.
        assert self._classify('{"jsonrpc": "2.0", bad}') == "banner"


# ---------------------------------------------------------------------------
# _NoPrePagePingMixin — pre-page ``send_ping`` health check is disabled
# ---------------------------------------------------------------------------


class TestNoPrePagePing:
    """Both subclasses must override ``_ensure_connected`` to a no-op.

    Locks the workaround in place: if a future MCP transport change
    renames the method or the mixin's MRO position changes, this test fails
    instead of silently re-introducing the POST → GET → DELETE reconnect
    storm against servers that don't implement the optional ``ping`` utility.
    """

    async def test_http_tool_skips_ping(self) -> None:
        tool = _HTTPMCPTool(name="h", url="http://localhost/mcp")
        tool.session = MagicMock()
        tool.session.send_ping = AsyncMock(side_effect=RuntimeError("ping should not be called"))

        await tool._ensure_connected()

        tool.session.send_ping.assert_not_called()

    async def test_stdio_tool_skips_ping(self) -> None:
        tool = _SafeStdioTool(name="s", command="python", args=["-m", "srv"])
        tool.session = MagicMock()
        tool.session.send_ping = AsyncMock(side_effect=RuntimeError("ping should not be called"))

        await tool._ensure_connected()

        tool.session.send_ping.assert_not_called()


class TestStructuredContentFallback:
    """``_StructuredContentFallbackMixin`` surfaces ``CallToolResult.structuredContent``
    when ``content`` carries no meaningful payload.

    The MCP spec lets servers deliver their result via the ``structuredContent``
    JSON object, optionally with an empty ``TextContent`` placeholder in
    ``content`` for clients that don't yet read structured output.  Upstream
    ``MCPTool._parse_tool_result_from_mcp`` only walks ``content`` and would
    drop the structured payload, so chrys's tool result would render empty.
    """

    @staticmethod
    def _text(text: str) -> Any:
        from mcp import types

        return types.TextContent(type="text", text=text)

    @staticmethod
    def _result(content: list[Any], structured: dict[str, Any] | None, *, is_error: bool = False) -> Any:
        from mcp import types

        return types.CallToolResult(content=content, structuredContent=structured, isError=is_error)

    def _parse(self, tool: Any, result: Any) -> list[Any]:
        return tool._parse_tool_result_from_mcp(result)

    def test_empty_content_with_structured_returns_json_dump(self) -> None:
        tool = _HTTPMCPTool(name="h", url="http://localhost/mcp")
        out = self._parse(tool, self._result([], {"items": [1, 2], "ok": True}))
        assert len(out) == 1
        assert out[0].text == '{"items": [1, 2], "ok": true}'

    def test_empty_text_block_with_structured_returns_json_dump(self) -> None:
        tool = _SafeStdioTool(name="s", command="python")
        out = self._parse(tool, self._result([self._text("")], {"k": "v"}))
        assert len(out) == 1
        assert out[0].text == '{"k": "v"}'

    def test_whitespace_text_block_with_structured_returns_json_dump(self) -> None:
        tool = _HTTPMCPTool(name="h", url="http://localhost/mcp")
        out = self._parse(tool, self._result([self._text("   \n  ")], {"k": "v"}))
        assert len(out) == 1
        assert out[0].text == '{"k": "v"}'

    def test_meaningful_text_with_structured_uses_framework_parser(self) -> None:
        """A populated text fallback wins — structuredContent is treated as a duplicate."""
        tool = _HTTPMCPTool(name="h", url="http://localhost/mcp")
        out = self._parse(tool, self._result([self._text("hello")], {"k": "v"}))
        assert len(out) == 1
        assert out[0].text == "hello"

    def test_empty_content_without_structured_uses_framework_parser(self) -> None:
        """No structured payload → framework's ``"null"`` placeholder is preserved."""
        tool = _HTTPMCPTool(name="h", url="http://localhost/mcp")
        out = self._parse(tool, self._result([], None))
        assert len(out) == 1
        assert out[0].text == "null"

    def test_non_text_content_with_structured_uses_framework_parser(self) -> None:
        """Image/audio/etc. items count as meaningful — don't override their rendering."""
        from mcp import types

        tool = _HTTPMCPTool(name="h", url="http://localhost/mcp")
        image = types.ImageContent(type="image", data="aGVsbG8=", mimeType="image/png")
        out = self._parse(tool, self._result([image], {"k": "v"}))
        assert len(out) == 1
        assert getattr(out[0], "media_type", None) == "image/png"

    def test_cyclic_structured_falls_back_to_str(self) -> None:
        """Cyclic ``structuredContent`` defeats ``json.dumps`` — the ``str()`` fallback must catch it."""
        cyclic: dict[str, Any] = {"k": "v"}
        cyclic["self"] = cyclic

        tool = _HTTPMCPTool(name="h", url="http://localhost/mcp")
        out = self._parse(tool, self._result([], cyclic))
        assert len(out) == 1
        assert isinstance(out[0].text, str)
        assert out[0].text  # non-empty


# ---------------------------------------------------------------------------
# MCPAdapter server instructions
# ---------------------------------------------------------------------------


def test_get_server_instructions_map_empty() -> None:
    """Empty when no servers have instructions."""
    adapter = MCPAdapter()
    assert adapter.get_server_instructions_map() == {}


def test_get_server_instructions_map_from_servers() -> None:
    """Aggregates from _servers when instructions are set."""
    adapter = MCPAdapter()
    tool_a = MagicMock()
    tool_a._server_instructions = "Instructions for A."
    tool_b = MagicMock()
    tool_b._server_instructions = "Instructions for B."
    adapter._servers["srv_a"] = tool_a
    adapter._servers["srv_b"] = tool_b

    result = adapter.get_server_instructions_map()
    assert result == {"srv_a": "Instructions for A.", "srv_b": "Instructions for B."}


def test_get_server_instructions_map_skips_none() -> None:
    """Servers with None instructions are excluded."""
    adapter = MCPAdapter()
    tool = MagicMock()
    tool._server_instructions = None
    adapter._servers["srv"] = tool

    assert adapter.get_server_instructions_map() == {}


def test_get_server_instructions_map_from_leases() -> None:
    """Aggregates from _leases when _servers lacks the entry."""
    adapter = MCPAdapter()
    lease = MagicMock()
    lease.mcp_tool._server_instructions = "From lease."
    adapter._leases["leased_srv"] = lease

    result = adapter.get_server_instructions_map()
    assert result == {"leased_srv": "From lease."}


def test_get_server_instructions_map_servers_win_over_leases() -> None:
    """_servers takes priority over _leases for the same server name."""
    adapter = MCPAdapter()
    tool = MagicMock()
    tool._server_instructions = "From servers."
    adapter._servers["srv"] = tool
    lease = MagicMock()
    lease.mcp_tool._server_instructions = "From leases."
    adapter._leases["srv"] = lease

    # _servers is iterated first, _leases only fills missing names
    assert adapter.get_server_instructions_map() == {"srv": "From servers."}


def test_render_instructions_reminder_empty() -> None:
    """None when no servers have instructions."""
    adapter = MCPAdapter()
    assert adapter.render_instructions_reminder() is None


def test_render_instructions_reminder_single_server() -> None:
    """Single server rendered in XML format."""
    adapter = MCPAdapter()
    tool = MagicMock()
    tool._server_instructions = "Use this server for file operations."
    adapter._servers["filesystem"] = tool

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert result.startswith("<mcp_instructions>")
    assert result.endswith("</mcp_instructions>")
    assert '<server name="filesystem">' in result
    assert "  </server>" in result
    assert "    Use this server for file operations." in result


def test_render_instructions_reminder_multi_server() -> None:
    """Multiple servers each get their own <server> element."""
    adapter = MCPAdapter()
    tool_a = MagicMock()
    tool_a._server_instructions = "GitHub API access."
    tool_b = MagicMock()
    tool_b._server_instructions = "Weather data provider."
    adapter._servers["github"] = tool_a
    adapter._servers["weather"] = tool_b

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert '<server name="github">' in result
    assert "    GitHub API access." in result
    assert '<server name="weather">' in result
    assert "    Weather data provider." in result
    # Both servers present, ordered by iteration
    assert result.index("github") < result.index("weather")


def test_render_instructions_reminder_multiline_instructions() -> None:
    """Multi-line instructions are indented per line."""
    adapter = MCPAdapter()
    tool = MagicMock()
    tool._server_instructions = "Line one.\nLine two.\nLine three."
    adapter._servers["srv"] = tool

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert "    Line one." in result
    assert "    Line two." in result
    assert "    Line three." in result
    # Check ordering: each line indented at the same level
    idx_one = result.index("    Line one.")
    idx_two = result.index("    Line two.")
    idx_three = result.index("    Line three.")
    assert idx_one < idx_two < idx_three


def test_render_instructions_reminder_xml_escape_name() -> None:
    """Server name with XML special characters is escaped."""
    adapter = MCPAdapter()
    tool = MagicMock()
    tool._server_instructions = "Test."
    adapter._servers['test"&<>'] = tool

    result = adapter.render_instructions_reminder()
    assert result is not None
    # The quote needs to be escaped for the attribute value
    assert 'name="test&quot;&amp;&lt;&gt;"' in result


def test_render_instructions_reminder_xml_escape_content() -> None:
    """Instruction text with XML special characters is escaped."""
    adapter = MCPAdapter()
    tool = MagicMock()
    tool._server_instructions = 'Use <tag> & "quotes".'
    adapter._servers["srv"] = tool

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert "    Use &lt;tag&gt; &amp; &quot;quotes&quot;." in result


# ---------------------------------------------------------------------------
# MCPTool._server_instructions reset
# ---------------------------------------------------------------------------


def test_server_instructions_reset_on_session_state_reset() -> None:
    """_server_instructions is cleared after _reset_session_state."""
    tool = MCPTool(name="test")
    tool._server_instructions = "Some instructions"
    assert tool._server_instructions == "Some instructions"

    tool._reset_session_state()
    assert tool._server_instructions is None


def test_render_instructions_reminder_sorted_by_server_name() -> None:
    """Rendering is name-ordered even when registration order differs."""
    adapter = MCPAdapter()
    tool_z = MagicMock()
    tool_z._server_instructions = "Z instructions."
    tool_a = MagicMock()
    tool_a._server_instructions = "A instructions."
    adapter._servers["zeta"] = tool_z
    adapter._servers["alpha"] = tool_a

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert result.index('<server name="alpha">') < result.index('<server name="zeta">')


def test_render_instructions_reminder_crlf_and_blank_lines() -> None:
    """CRLF newlines are normalized and blank lines carry no trailing spaces."""
    adapter = MCPAdapter()
    tool = MagicMock()
    tool._server_instructions = "First.\r\n\r\nSecond."
    adapter._servers["srv"] = tool

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert "\r" not in result
    lines = result.split("\n")
    assert "    First." in lines
    assert "    Second." in lines
    # The blank instruction line renders empty, not as indent whitespace.
    assert "" in lines[lines.index("    First.") + 1 : lines.index("    Second.")]
    assert not any(line != line.rstrip() for line in lines)


def test_render_instructions_reminder_caps_oversized_instructions() -> None:
    """Server-controlled instructions are truncated at MCP_INSTRUCTIONS_CHAR_LIMIT."""
    adapter = MCPAdapter()
    tool = MagicMock()
    tool._server_instructions = "x" * (MCP_INSTRUCTIONS_CHAR_LIMIT + 500)
    adapter._servers["srv"] = tool

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert len(result) < MCP_INSTRUCTIONS_CHAR_LIMIT + 500
    assert f"[instructions truncated: exceeded {MCP_INSTRUCTIONS_CHAR_LIMIT} characters]" in result


def test_render_instructions_reminder_cap_counts_escaped_output() -> None:
    """The budget is charged post-escaping: entity expansion cannot multiply the cap."""
    adapter = MCPAdapter()
    tool = MagicMock()
    # Each '"' renders as '&quot;' (6 chars); a raw-input cap would admit ~6x the
    # budget.  The leading 'a' shifts the cut point mid-entity so the
    # severed-fragment cleanup path is exercised too.
    tool._server_instructions = "a" + '"' * (MCP_INSTRUCTIONS_CHAR_LIMIT + 1)
    adapter._servers["srv"] = tool

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert len(result) < MCP_INSTRUCTIONS_CHAR_LIMIT + 200
    assert f"[instructions truncated: exceeded {MCP_INSTRUCTIONS_CHAR_LIMIT} characters]" in result
    # The cut must not leave a severed entity fragment at the truncation point.
    kept = result.split("\n")[2]
    assert not kept.endswith("&") and not _SEVERED_ENTITY_TAIL_RE.search(kept)


def test_render_instructions_reminder_multiline_budget_keeps_leading_lines() -> None:
    """Truncation keeps complete leading lines and cuts within the overflowing one."""
    adapter = MCPAdapter()
    tool = MagicMock()
    tool._server_instructions = "short intro line\n" + "y" * (MCP_INSTRUCTIONS_CHAR_LIMIT * 2)
    adapter._servers["srv"] = tool

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert "    short intro line" in result
    assert f"[instructions truncated: exceeded {MCP_INSTRUCTIONS_CHAR_LIMIT} characters]" in result
    assert len(result) < MCP_INSTRUCTIONS_CHAR_LIMIT + 200


def test_render_instructions_reminder_blank_line_flood_is_bounded() -> None:
    """Thousands of blank lines cannot bypass the budget via free newline separators."""
    adapter = MCPAdapter()
    tool = MagicMock()
    tool._server_instructions = "x\n" + "\n" * (MCP_INSTRUCTIONS_CHAR_LIMIT * 3) + "x"
    adapter._servers["srv"] = tool

    result = adapter.render_instructions_reminder()
    assert result is not None
    assert len(result) < MCP_INSTRUCTIONS_CHAR_LIMIT + 200
    assert f"[instructions truncated: exceeded {MCP_INSTRUCTIONS_CHAR_LIMIT} characters]" in result


def test_get_server_instructions_map_respects_expose_instructions_flag() -> None:
    """Servers registered with expose_instructions=False are excluded from the map."""
    adapter = MCPAdapter()
    tool_shown = MagicMock()
    tool_shown._server_instructions = "Visible."
    tool_hidden = MagicMock()
    tool_hidden._server_instructions = "Hidden."
    adapter._servers["shown"] = tool_shown
    adapter._servers["hidden"] = tool_hidden
    adapter._instructions_exposure["shown"] = True
    adapter._instructions_exposure["hidden"] = False

    assert adapter.get_server_instructions_map() == {"shown": "Visible."}
    reminder = adapter.render_instructions_reminder()
    assert reminder is not None
    assert "Visible." in reminder
    assert "Hidden." not in reminder


def test_get_server_instructions_map_defaults_to_exposed_without_registration() -> None:
    """A server with no exposure record (e.g. hand-registered in tests) stays visible."""
    adapter = MCPAdapter()
    tool = MagicMock()
    tool._server_instructions = "Default-visible."
    adapter._servers["srv"] = tool

    assert adapter.get_server_instructions_map() == {"srv": "Default-visible."}


def test_expose_instructions_flag_excludes_lease_fallback() -> None:
    """The _leases fallback path honors the exposure flag too."""
    adapter = MCPAdapter()
    lease = MagicMock()
    lease.mcp_tool._server_instructions = "From lease."
    adapter._leases["leased"] = lease
    adapter._instructions_exposure["leased"] = False

    assert adapter.get_server_instructions_map() == {}


async def test_connect_registers_instructions_exposure_and_disconnect_clears_it() -> None:
    """The production connect path records the flag; disconnect paths drop it."""
    adapter = MCPAdapter()
    hidden_cfg = MCPServerConfig(name="hidden", transport="stdio", command="python", expose_instructions=False)
    shown_cfg = MCPServerConfig(name="shown", transport="stdio", command="python")
    fake_hidden = _FakeTool(functions=[_function_tool("remote_hidden")])
    fake_hidden._server_instructions = "Hidden."
    fake_shown = _FakeTool(functions=[_function_tool("remote_shown")])
    fake_shown._server_instructions = "Shown."

    with patch("chrys.service.mcp.adapter._create_mcp_tool", side_effect=[fake_hidden, fake_shown]):
        await adapter.connect(hidden_cfg)
        await adapter.connect(shown_cfg)

    assert adapter._instructions_exposure == {"hidden": False, "shown": True}
    assert adapter.get_server_instructions_map() == {"shown": "Shown."}

    await adapter.disconnect("hidden")
    assert adapter._instructions_exposure == {"shown": True}

    await adapter.disconnect_all()
    assert adapter._instructions_exposure == {}
