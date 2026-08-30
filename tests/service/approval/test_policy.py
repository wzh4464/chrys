# Copyright (c) 2026 Chrys. All rights reserved.

"""Tests for ApprovalPolicy and ApprovalMiddleware read-only auto-approval."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import ApprovalRequest
from chrys.foundation.tool_kinds import KIND_CONTEXT, KIND_SKILL
from chrys.service.approval.policy import ApprovalMode, ApprovalPolicy
from chrys.service.profiles.agents.schema import ApprovalConfig
from chrys.service.skills.constants import RUN_SKILL_SCRIPT_TOOL_NAME
from tests.support.waiting import wait_until


def _mock_tool(name: str, kind: str | None = None) -> MagicMock:
    """Create a mock FunctionTool with name and kind."""
    t = MagicMock()
    t.name = name
    t.chrys_kind = kind
    return t


def test_default_auto_does_not_require() -> None:
    policy = ApprovalPolicy(ApprovalConfig(default="auto"))
    assert not policy.should_require_approval("any_tool")


def test_default_require() -> None:
    policy = ApprovalPolicy(ApprovalConfig(default="require"))
    assert policy.should_require_approval("any_tool")


def test_default_config_skips_todo_kind() -> None:
    """The schema-default overrides carry ``todo: skip`` — even under a
    require-everything default, todo_write never prompts (it only mutates
    engine-side todo state; the visible list update IS the review surface)."""
    tools = [
        _mock_tool("todo_write", kind="todo"),
        _mock_tool("other_tool"),
    ]
    policy = ApprovalPolicy(ApprovalConfig(default="require"), tools)
    assert not policy.should_require_approval("todo_write")
    assert policy.should_require_approval("other_tool")


def test_bare_tool_name_override() -> None:
    tools = [
        _mock_tool("run_skill_script"),
        _mock_tool("load_skill"),
    ]
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={"run_skill_script": "require"}),
        tools=tools,
    )
    assert policy.should_require_approval("run_skill_script")
    assert not policy.should_require_approval("load_skill")


def test_kind_only_override() -> None:
    """A kind-only key (e.g. ``shell``) applies to ALL tools with that kind."""
    tools = [
        _mock_tool("zsh", kind="shell"),
        _mock_tool("bash", kind="shell"),
        _mock_tool("pwsh", kind="shell"),
        _mock_tool("powershell", kind="shell"),
        _mock_tool("read_file", kind="filesystem.read"),
    ]
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={"shell": "require"}),
        tools=tools,
    )
    assert policy.should_require_approval("zsh")
    assert policy.should_require_approval("bash")
    assert policy.should_require_approval("pwsh")
    assert policy.should_require_approval("powershell")
    assert not policy.should_require_approval("read_file")


def test_kind_dot_name_override() -> None:
    """``kind.tool_name`` applies only to the exact tool, not all tools of that kind."""
    tools = [
        _mock_tool("write_file", kind="filesystem.write"),
        _mock_tool("edit_file", kind="filesystem.write"),
        _mock_tool("read_file", kind="filesystem.read"),
    ]
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={"filesystem.write.write_file": "require"}),
        tools=tools,
    )
    assert policy.should_require_approval("write_file")
    # Same kind but different name — must NOT be affected
    assert not policy.should_require_approval("edit_file")
    assert not policy.should_require_approval("read_file")


def test_mixed_overrides() -> None:
    """Kind-only and kind.name overrides work together."""
    tools = [
        _mock_tool("zsh", kind="shell"),
        _mock_tool("bash", kind="shell"),
        _mock_tool("write_file", kind="filesystem.write"),
        _mock_tool("edit_file", kind="filesystem.write"),
        _mock_tool("read_file", kind="filesystem.read"),
    ]
    policy = ApprovalPolicy(
        ApprovalConfig(
            default="auto",
            overrides={
                "shell": "require",
                "filesystem.write": "require",
            },
        ),
        tools=tools,
    )
    assert policy.should_require_approval("zsh")
    assert policy.should_require_approval("bash")
    assert policy.should_require_approval("write_file")
    assert policy.should_require_approval("edit_file")
    assert not policy.should_require_approval("read_file")


def test_user_override_when_allowed() -> None:
    policy = ApprovalPolicy(ApprovalConfig(default="require", user_can_override=True))
    assert policy.should_require_approval("any_tool")

    ok = policy.set_user_override("skip")
    assert ok
    assert not policy.should_require_approval("any_tool")


def test_user_override_when_not_allowed() -> None:
    policy = ApprovalPolicy(ApprovalConfig(default="require", user_can_override=False))
    ok = policy.set_user_override("skip")
    assert not ok
    # Policy unchanged
    assert policy.should_require_approval("any_tool")


def test_clear_user_override() -> None:
    policy = ApprovalPolicy(ApprovalConfig(default="require", user_can_override=True))
    policy.set_user_override("skip")
    assert not policy.should_require_approval("any_tool")

    policy.set_user_override(None)
    assert policy.should_require_approval("any_tool")


def test_sub_agent_with_subset_tools() -> None:
    """Parent overrides for missing tools should not bleed to other tools of the same kind.

    Reproduces the bug where a sub-agent with only ``read_file`` (kind=filesystem.read)
    inherited ``filesystem.write: require`` and incorrectly required approval
    for ``read_file``.
    """
    tools = [
        _mock_tool("read_file", kind="filesystem.read"),
        _mock_tool("zsh", kind="shell"),
    ]
    policy = ApprovalPolicy(
        ApprovalConfig(
            default="auto",
            overrides={
                "shell": "require",
                "filesystem.write": "require",
            },
        ),
        tools=tools,
    )
    # kind-only "shell" matches the shell tool
    assert policy.should_require_approval("zsh")
    # kind-only "filesystem.write" must NOT apply to read_file (different kind)
    assert not policy.should_require_approval("read_file")


# ──────────────── ApprovalMiddleware read-only auto-approval ────────────


def _make_middleware_context(tool_name: str, tool_kind: str, args: dict) -> MagicMock:
    """Create a mock FunctionInvocationContext for middleware tests."""
    ctx = MagicMock()
    ctx.function = SimpleNamespace(name=tool_name, chrys_kind=tool_kind)
    ctx.arguments = args
    ctx.metadata = {}
    ctx.result = None
    return ctx


@pytest.mark.asyncio
async def test_middleware_auto_approves_readonly_shell() -> None:
    """ApprovalMiddleware should auto-approve safe read-only shell commands."""
    from chrys.service.agent_middleware import ApprovalMiddleware

    tools = [_mock_tool("zsh", kind="shell")]
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={"shell": "require"}),
        tools=tools,
    )
    bus = EventBus()
    mw = ApprovalMiddleware(approval_policy=policy, event_bus=bus)

    ctx = _make_middleware_context("zsh", "shell", {"command": "ls -la | grep foo", "reason": "list files"})
    call_next_called = False

    async def mock_call_next() -> None:
        nonlocal call_next_called
        call_next_called = True

    await mw.process(ctx, mock_call_next)

    assert call_next_called, "read-only command should auto-approve and call next"


@pytest.mark.asyncio
async def test_middleware_auto_approves_powershell_null_redirect() -> None:
    """PowerShell null redirection should stay on the read-only auto-approval path."""
    from chrys.service.agent_middleware import ApprovalMiddleware

    tools = [_mock_tool("pwsh", kind="shell")]
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={"shell": "require"}),
        tools=tools,
    )
    bus = EventBus()
    mw = ApprovalMiddleware(approval_policy=policy, event_bus=bus)
    requests: list[ApprovalRequest] = []

    async def _collect(event: ApprovalRequest) -> None:
        requests.append(event)

    await bus.subscribe(ApprovalRequest, _collect)

    ctx = _make_middleware_context("pwsh", "shell", {"command": "Get-ChildItem 2> $null"})
    call_next_called = False

    async def mock_call_next() -> None:
        nonlocal call_next_called
        call_next_called = True

    await mw.process(ctx, mock_call_next)

    assert call_next_called
    assert not await wait_until(lambda: bool(requests), timeout=0.2, interval=0.01)


@pytest.mark.asyncio
async def test_middleware_requires_approval_for_unsafe_shell() -> None:
    """Unsafe shell commands should still publish ApprovalRequest."""
    from chrys.service.agent_middleware import ApprovalMiddleware

    tools = [_mock_tool("zsh", kind="shell")]
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={"shell": "require"}),
        tools=tools,
    )
    bus = EventBus()
    mw = ApprovalMiddleware(approval_policy=policy, event_bus=bus, workspace_cwd="/workspace")

    ctx = _make_middleware_context("zsh", "shell", {"command": "git push origin main", "reason": "push code"})

    # Collect approval requests
    requests: list[ApprovalRequest] = []

    async def _collect(event: ApprovalRequest) -> None:
        requests.append(event)

    await bus.subscribe(ApprovalRequest, _collect)

    # Run middleware in background — it will block waiting for ApprovalResponse
    async def _noop() -> None:
        pass

    task = asyncio.create_task(mw.process(ctx, _noop))

    # Wait for the middleware to publish the request before asserting.
    assert await wait_until(lambda: len(requests) >= 1), "unsafe command should trigger approval request"

    assert len(requests) == 1, "unsafe command should trigger approval request"
    assert requests[0].tool_name == "zsh"
    assert requests[0].workspace_cwd == "/workspace"

    # Clean up — cancel the blocked task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_middleware_auto_approves_non_shell_unaffected() -> None:
    """Non-shell tools should not be affected by the read-only bypass."""
    from chrys.service.agent_middleware import ApprovalMiddleware

    tools = [_mock_tool("write_file", kind="filesystem.write")]
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={"filesystem.write": "require"}),
        tools=tools,
    )
    bus = EventBus()
    mw = ApprovalMiddleware(approval_policy=policy, event_bus=bus)

    ctx = _make_middleware_context("write_file", "filesystem.write", {"path": "/tmp/test.txt", "content": "hello"})

    requests: list[ApprovalRequest] = []

    async def _collect(event: ApprovalRequest) -> None:
        requests.append(event)

    await bus.subscribe(ApprovalRequest, _collect)

    async def _noop() -> None:
        pass

    task = asyncio.create_task(mw.process(ctx, _noop))
    assert await wait_until(lambda: len(requests) >= 1), "non-shell tool should still require approval"

    assert len(requests) == 1, "non-shell tool should still require approval"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ──────────── File write auto-approval (workspace + git) ────────────


@pytest.fixture
def git_workspace(tmp_path: Path, git_repo_factory: Callable[[Path], Path]) -> str:
    """Create a temp directory with a git repo for testing."""
    git_repo_factory(tmp_path)
    return str(tmp_path)


@pytest.fixture
def non_git_workspace(tmp_path: Path) -> str:
    """Create a temp directory WITHOUT git for testing."""
    d = tmp_path / "no_git"
    d.mkdir()
    return str(d)


@pytest.mark.asyncio
async def test_middleware_auto_approves_file_write_in_git_workspace(git_workspace) -> None:
    """write_file inside a workspace git repo should auto-approve."""
    from chrys.service.agent_middleware import ApprovalMiddleware

    tools = [_mock_tool("write_file", kind="filesystem.write")]
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={"filesystem.write": "require"}),
        tools=tools,
    )
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=policy,
        event_bus=bus,
        workspace_roots=[git_workspace],
    )

    target = os.path.join(git_workspace, "new_file.txt")
    ctx = _make_middleware_context("write_file", "filesystem.write", {"path": target, "content": "hello"})
    call_next_called = False

    async def mock_call_next() -> None:
        nonlocal call_next_called
        call_next_called = True

    await mw.process(ctx, mock_call_next)
    assert call_next_called, "write inside git workspace should auto-approve"


@pytest.mark.asyncio
async def test_middleware_auto_approves_edit_file_in_git_workspace(git_workspace) -> None:
    """edit_file inside a workspace git repo should auto-approve."""
    from chrys.service.agent_middleware import ApprovalMiddleware

    tools = [_mock_tool("edit_file", kind="filesystem.write")]
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={"filesystem.write": "require"}),
        tools=tools,
    )
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=policy,
        event_bus=bus,
        workspace_roots=[git_workspace],
    )

    target = os.path.join(git_workspace, "src", "main.py")
    ctx = _make_middleware_context(
        "edit_file", "filesystem.write", {"path": target, "old_string": "a", "new_string": "b"}
    )
    call_next_called = False

    async def mock_call_next() -> None:
        nonlocal call_next_called
        call_next_called = True

    await mw.process(ctx, mock_call_next)
    assert call_next_called, "edit inside git workspace should auto-approve"


@pytest.mark.asyncio
async def test_middleware_requires_approval_outside_workspace(git_workspace) -> None:
    """File write OUTSIDE workspace roots should still require approval."""
    from chrys.service.agent_middleware import ApprovalMiddleware

    tools = [_mock_tool("write_file", kind="filesystem.write")]
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={"filesystem.write": "require"}),
        tools=tools,
    )
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=policy,
        event_bus=bus,
        workspace_roots=[git_workspace],
    )

    # Target outside workspace
    ctx = _make_middleware_context("write_file", "filesystem.write", {"path": "/etc/passwd", "content": "bad"})

    requests: list[ApprovalRequest] = []

    async def _collect(event: ApprovalRequest) -> None:
        requests.append(event)

    await bus.subscribe(ApprovalRequest, _collect)

    async def _noop() -> None:
        pass

    task = asyncio.create_task(mw.process(ctx, _noop))
    assert await wait_until(lambda: len(requests) >= 1), "write outside workspace should require approval"

    assert len(requests) == 1, "write outside workspace should require approval"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_middleware_requires_approval_no_git(non_git_workspace) -> None:
    """File write inside workspace but WITHOUT git should require approval."""
    from chrys.service.agent_middleware import ApprovalMiddleware

    tools = [_mock_tool("write_file", kind="filesystem.write")]
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={"filesystem.write": "require"}),
        tools=tools,
    )
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=policy,
        event_bus=bus,
        workspace_roots=[non_git_workspace],
    )

    target = os.path.join(non_git_workspace, "file.txt")
    ctx = _make_middleware_context("write_file", "filesystem.write", {"path": target, "content": "hello"})

    requests: list[ApprovalRequest] = []

    async def _collect(event: ApprovalRequest) -> None:
        requests.append(event)

    await bus.subscribe(ApprovalRequest, _collect)

    async def _noop() -> None:
        pass

    task = asyncio.create_task(mw.process(ctx, _noop))
    assert await wait_until(lambda: len(requests) >= 1), "write without git should require approval"

    assert len(requests) == 1, "write without git should require approval"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_middleware_no_workspace_roots_requires_approval() -> None:
    """Without workspace_roots configured, file writes should require approval."""
    from chrys.service.agent_middleware import ApprovalMiddleware

    tools = [_mock_tool("write_file", kind="filesystem.write")]
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={"filesystem.write": "require"}),
        tools=tools,
    )
    bus = EventBus()
    mw = ApprovalMiddleware(approval_policy=policy, event_bus=bus)

    ctx = _make_middleware_context("write_file", "filesystem.write", {"path": "/tmp/test.txt", "content": "hello"})

    requests: list[ApprovalRequest] = []

    async def _collect(event: ApprovalRequest) -> None:
        requests.append(event)

    await bus.subscribe(ApprovalRequest, _collect)

    async def _noop() -> None:
        pass

    task = asyncio.create_task(mw.process(ctx, _noop))
    assert await wait_until(lambda: len(requests) >= 1), "no workspace roots means approval required"

    assert len(requests) == 1, "no workspace roots means approval required"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.fixture
def workspace_with_secondary_git(
    tmp_path: Path,
    git_repo_factory: Callable[[Path], Path],
) -> tuple[str, str]:
    """Create primary (no git) + secondary (git) workspace dirs."""
    primary = tmp_path / "primary"
    primary.mkdir()
    secondary = tmp_path / "secondary"
    git_repo_factory(secondary)
    return str(primary), str(secondary)


@pytest.mark.asyncio
async def test_middleware_auto_approves_working_dir(workspace_with_secondary_git) -> None:
    """File write inside a secondary working_dir (not primary_cwd) should auto-approve."""
    from chrys.service.agent_middleware import ApprovalMiddleware

    primary, secondary = workspace_with_secondary_git

    tools = [_mock_tool("write_file", kind="filesystem.write")]
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={"filesystem.write": "require"}),
        tools=tools,
    )
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=policy,
        event_bus=bus,
        workspace_roots=[primary, secondary],
    )

    target = os.path.join(secondary, "test.py")
    ctx = _make_middleware_context("write_file", "filesystem.write", {"path": target, "content": "print(1)"})
    call_next_called = False

    async def mock_call_next() -> None:
        nonlocal call_next_called
        call_next_called = True

    await mw.process(ctx, mock_call_next)
    assert call_next_called, "write in secondary working dir with git should auto-approve"


# ── ApprovalMode.from_string ────────────────────────────────────────


def test_approval_mode_from_string_valid() -> None:
    assert ApprovalMode.from_string("manual") is ApprovalMode.MANUAL
    assert ApprovalMode.from_string("auto") is ApprovalMode.AUTO
    assert ApprovalMode.from_string("bypass") is ApprovalMode.BYPASS


def test_approval_mode_from_string_normalizes_case_and_whitespace() -> None:
    assert ApprovalMode.from_string("  AUTO  ") is ApprovalMode.AUTO
    assert ApprovalMode.from_string("Manual") is ApprovalMode.MANUAL


def test_approval_mode_from_string_falls_back_on_invalid() -> None:
    assert ApprovalMode.from_string("") is ApprovalMode.MANUAL
    assert ApprovalMode.from_string(None) is ApprovalMode.MANUAL
    assert ApprovalMode.from_string("nonsense") is ApprovalMode.MANUAL


def test_approval_mode_from_string_custom_default() -> None:
    assert ApprovalMode.from_string("", default=ApprovalMode.AUTO) is ApprovalMode.AUTO
    assert ApprovalMode.from_string("garbage", default=ApprovalMode.BYPASS) is ApprovalMode.BYPASS


# ──────────────── provenance kinds in the override namespace ────────────────


def test_skill_kind_only_override_applies_to_skill_tools() -> None:
    """`skill: require` targets the whole skill tool family via KIND_SKILL."""
    tools = [
        _mock_tool("load_skill", kind=KIND_SKILL),
        _mock_tool("run_skill_script", kind=KIND_SKILL),
        _mock_tool("read_file", kind="filesystem.read"),
    ]
    policy = ApprovalPolicy(ApprovalConfig(default="auto", overrides={"skill": "require"}), tools=tools)
    assert policy.should_require_approval("load_skill")
    assert policy.should_require_approval("run_skill_script")
    assert not policy.should_require_approval("read_file")


def test_context_kind_only_override_applies_to_context_tools() -> None:
    tools = [
        _mock_tool("compress_context", kind=KIND_CONTEXT),
        _mock_tool("list_compressed_contexts", kind=KIND_CONTEXT),
        _mock_tool("zsh", kind="shell"),
    ]
    policy = ApprovalPolicy(ApprovalConfig(default="auto", overrides={"context": "require"}), tools=tools)
    assert policy.should_require_approval("compress_context")
    assert policy.should_require_approval("list_compressed_contexts")
    assert not policy.should_require_approval("zsh")


@pytest.mark.parametrize(
    ("tool_name", "tool_kind", "override_key"),
    [
        ("load_skill", KIND_SKILL, "skill"),
        ("compress_context", KIND_CONTEXT, "context"),
    ],
)
def test_kind_override_applies_to_runtime_added_tool(
    tool_name: str,
    tool_kind: str,
    override_key: str,
) -> None:
    """Live provenance enforces kind rules even when construction saw no tool."""
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={override_key: "require"}),
        tools=[],
    )

    assert policy.should_require_approval(tool_name, tool_kind)
    assert not policy.should_require_approval("unrelated_tool", "mcp")


@pytest.mark.parametrize(
    "overrides",
    [
        {"skill": "auto", "load_skill": "skip", "skill.load_skill": "require"},
        {"skill.load_skill": "require", "load_skill": "skip", "skill": "auto"},
    ],
)
def test_kind_qualified_tool_override_wins_independent_of_mapping_order(overrides: dict[str, str]) -> None:
    policy = ApprovalPolicy(ApprovalConfig(default="skip", overrides=overrides), tools=[])

    assert policy.should_require_approval("load_skill", KIND_SKILL)


@pytest.mark.parametrize(
    "overrides",
    [
        {"skill": "require", "load_skill": "skip"},
        {"load_skill": "skip", "skill": "require"},
    ],
)
def test_bare_tool_override_wins_over_kind_independent_of_mapping_order(overrides: dict[str, str]) -> None:
    policy = ApprovalPolicy(ApprovalConfig(default="require", overrides=overrides), tools=[])

    assert not policy.should_require_approval("load_skill", KIND_SKILL)


@pytest.mark.parametrize("default", ["auto", "skip"])
def test_run_skill_script_has_secure_approval_floor(default: str) -> None:
    policy = ApprovalPolicy(ApprovalConfig(default=default, overrides={}), tools=[])

    assert policy.should_require_approval("run_skill_script", KIND_SKILL)
    assert policy.should_require_approval("run_skill_script")


@pytest.mark.parametrize(
    "override_key",
    [RUN_SKILL_SCRIPT_TOOL_NAME, f"{KIND_SKILL}.{RUN_SKILL_SCRIPT_TOOL_NAME}"],
)
@pytest.mark.parametrize("rule", ["auto", "skip"])
def test_explicit_tool_override_can_opt_out_of_run_skill_script_floor(
    override_key: str,
    rule: str,
) -> None:
    policy = ApprovalPolicy(
        ApprovalConfig(default="require", overrides={override_key: rule}),
        tools=[],
    )

    assert not policy.should_require_approval("run_skill_script", KIND_SKILL)


@pytest.mark.parametrize("rule", ["auto", "skip"])
def test_kind_wide_skill_override_intentionally_opts_out_of_script_floor(rule: str) -> None:
    """A kind rule is explicit authority over every skill tool, including execution."""
    policy = ApprovalPolicy(
        ApprovalConfig(default="require", overrides={KIND_SKILL: rule}),
        tools=[],
    )

    assert not policy.should_require_approval(RUN_SKILL_SCRIPT_TOOL_NAME, KIND_SKILL)
    assert not policy.should_require_approval("load_skill", KIND_SKILL)


def test_invalid_matching_override_fails_closed(caplog: pytest.LogCaptureFixture) -> None:
    config = ApprovalConfig(default="auto", overrides={})
    config.overrides[RUN_SKILL_SCRIPT_TOOL_NAME] = cast(Any, True)

    with caplog.at_level("ERROR", logger="chrys.service.approval.policy"):
        policy = ApprovalPolicy(config, tools=[])

    assert policy.should_require_approval(RUN_SKILL_SCRIPT_TOOL_NAME, KIND_SKILL)
    assert any("Invalid approval rule" in record.message for record in caplog.records)


def test_invalid_default_fails_closed() -> None:
    config = ApprovalConfig(default=cast(Any, "maybe"), overrides={})
    policy = ApprovalPolicy(config, tools=[])

    assert policy.should_require_approval("ordinary_tool", "mcp")


def test_allowed_user_override_precedes_run_skill_script_floor() -> None:
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={}, user_can_override=True),
        tools=[],
    )
    assert policy.set_user_override("skip")

    assert not policy.should_require_approval("run_skill_script", KIND_SKILL)


def test_invalid_user_override_fails_closed() -> None:
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={}, user_can_override=True),
        tools=[],
    )

    assert policy.set_user_override(cast(Any, True))
    assert policy.should_require_approval(RUN_SKILL_SCRIPT_TOOL_NAME, KIND_SKILL)


def test_reserved_kind_key_shadowing_tool_name_warns_when_kind_in_toolset(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tool literally named `skill` next to KIND_SKILL tools: key applies to the kind, warns."""
    tools = [
        _mock_tool("load_skill", kind=KIND_SKILL),
        _mock_tool("skill", kind="mcp"),
    ]
    with caplog.at_level("WARNING", logger="chrys.service.approval.policy"):
        policy = ApprovalPolicy(ApprovalConfig(default="auto", overrides={"skill": "require"}), tools=tools)
    assert policy.should_require_approval("load_skill"), "the kind semantics must still win"
    assert not policy.should_require_approval("skill"), "the literally-named tool is NOT targeted"
    assert any("'mcp.skill'" in r.message for r in caplog.records), "the warning must carry the migration form"


def test_reserved_kind_key_shadowing_tool_name_warns_when_kind_absent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A reserved kind can target future tools, never the same-named MCP tool."""
    tools = [_mock_tool("context", kind="mcp")]
    with caplog.at_level("WARNING", logger="chrys.service.approval.policy"):
        policy = ApprovalPolicy(ApprovalConfig(default="auto", overrides={"context": "require"}), tools=tools)
    assert not policy.should_require_approval("context")
    assert any("runtime-added tools" in r.message and "'mcp.context'" in r.message for r in caplog.records)


def test_kind_dot_name_form_targets_kind_named_tool_exactly() -> None:
    """The migration path: `mcp.context` reaches a tool literally named `context`."""
    tools = [
        _mock_tool("context", kind="mcp"),
        _mock_tool("other_remote", kind="mcp"),
    ]
    policy = ApprovalPolicy(ApprovalConfig(default="auto", overrides={"mcp.context": "require"}), tools=tools)
    assert policy.should_require_approval("context")
    assert not policy.should_require_approval("other_remote")


def test_no_shadow_warning_when_no_tool_bears_the_kind_name(caplog: pytest.LogCaptureFixture) -> None:
    tools = [_mock_tool("load_skill", kind=KIND_SKILL)]
    with caplog.at_level("WARNING", logger="chrys.service.approval.policy"):
        ApprovalPolicy(ApprovalConfig(default="auto", overrides={"skill": "require"}), tools=tools)
    assert not caplog.records
