# Copyright (c) 2026 Chrys. All rights reserved.

"""Additional coverage for ``ApprovalMiddleware`` branches not covered by the
existing test suite: BYPASS mode, AUTO + judge (approve / flag / cancel /
error), args parsing, rejection path, and the small helper methods
(``set_user_message``, ``set_approval_mode``, ``drain_decisions``, ``reset``)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from chrys.foundation.events.bus import EventBus
from chrys.foundation.events.types import (
    ApprovalAutoFulfillBlocked,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalReviewed,
    ToolCallArgsUpdated,
)
from chrys.foundation.tool_kinds import KIND_CONTEXT, KIND_SKILL, KIND_SUB_AGENT
from chrys.foundation.trajectory.context import trajectory_scope
from chrys.foundation.trajectory.event_types import EventType
from chrys.service.agent_middleware.control.approval import (
    ApprovalMiddleware,
    _is_workspace_git_path,
)
from chrys.service.agent_middleware.events.hook_dispatch import set_call_id, set_tool_invocation_order
from chrys.service.approval.judge import JudgeVerdict
from chrys.service.approval.policy import ApprovalMode, ApprovalPolicy
from chrys.service.profiles.agents.schema import ApprovalConfig
from chrys.service.trajectory.approvals import ApprovalDecider, ApprovalDecision
from tests.service.trajectory._fakes import FakeSink, make_context
from tests.support.waiting import ENGINE_TURN_TIMEOUT, wait_for, wait_until


def _mock_tool(name: str, kind: str | None = None) -> MagicMock:
    t = MagicMock()
    t.name = name
    t.chrys_kind = kind
    return t


def _ctx(tool_name: str, tool_kind: str, args) -> MagicMock:
    """Make a mock FunctionInvocationContext. *args* may be dict, str, or other."""
    c = MagicMock()
    c.function = SimpleNamespace(name=tool_name, chrys_kind=tool_kind)
    c.arguments = args
    c.metadata = {}
    c.result = None
    return c


def _require_all_policy(kind: str = "shell", tool_names: tuple[str, ...] = ("write_file",)) -> ApprovalPolicy:
    """Policy that requires approval for all *tool_names* with *kind*."""
    tools = [_mock_tool(n, kind=kind) for n in tool_names]
    return ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={kind: "require"}),
        tools=tools,
    )


# ───────────────────────── helper functions ───────────────────────────


def test_is_workspace_git_path_outside_workspace(tmp_path) -> None:
    """Path outside any resolved root → False."""
    inside = tmp_path / "inside"
    inside.mkdir()
    (inside / ".git").mkdir()
    other = tmp_path / "other" / "file.txt"
    other.parent.mkdir()
    other.write_text("x")

    import os

    roots = [os.path.realpath(str(inside))]
    assert _is_workspace_git_path(str(other), roots) is False


def test_is_workspace_git_path_inside_but_no_git(tmp_path) -> None:
    """Inside a root but no .git anywhere up the tree → False."""
    import os

    (tmp_path / "sub").mkdir()
    target = tmp_path / "sub" / "x.txt"
    target.write_text("x")
    roots = [os.path.realpath(str(tmp_path))]
    assert _is_workspace_git_path(str(target), roots) is False


def test_is_workspace_git_path_realpath_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If ``os.path.realpath`` raises (e.g. OS-level error), return False."""
    import os

    def _boom(_: str) -> str:
        raise OSError("simulated realpath failure")

    monkeypatch.setattr(os.path, "realpath", _boom)
    assert _is_workspace_git_path("/any/path", ["/roots"]) is False


def test_is_workspace_git_path_git_in_ancestor(tmp_path) -> None:
    """``.git`` in an ancestor directory → True (walks up from file's dir)."""
    import os

    (tmp_path / ".git").mkdir()
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    target = nested / "file.txt"
    target.write_text("x")
    roots = [os.path.realpath(str(tmp_path))]
    assert _is_workspace_git_path(str(target), roots) is True


@pytest.mark.parametrize("target", [".git", ".git/config", ".git/hooks/pre-commit", ".git/hooks/post-checkout"])
def test_is_workspace_git_path_rejects_git_internals(tmp_path, target: str) -> None:
    """Git internals are not recoverable tracked workspace writes."""
    import os

    (tmp_path / ".git" / "hooks").mkdir(parents=True)
    roots = [os.path.realpath(str(tmp_path))]
    assert _is_workspace_git_path(str(tmp_path / target), roots) is False


def test_is_workspace_git_path_rejects_symlinked_git_internals(tmp_path) -> None:
    """The git-internals guard compares real paths so .git symlinks cannot bypass it."""
    import os

    git_dir = tmp_path / "actual-git-dir"
    git_dir.mkdir()
    (tmp_path / ".git").symlink_to(git_dir, target_is_directory=True)
    roots = [os.path.realpath(str(tmp_path))]
    assert _is_workspace_git_path(str(tmp_path / ".git" / "config"), roots) is False


def test_is_workspace_git_path_git_file_in_ancestor(tmp_path) -> None:
    """``.git`` file in an ancestor directory → True for worktrees/submodules."""
    import os

    (tmp_path / ".git").write_text("gitdir: ../.git/modules/example\n")
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    target = nested / "file.txt"
    target.write_text("x")
    roots = [os.path.realpath(str(tmp_path))]
    assert _is_workspace_git_path(str(target), roots) is True


def test_is_workspace_git_path_rejects_gitdir_pointer_target(tmp_path) -> None:
    """Real gitdir targets referenced by .git pointer files are metadata internals."""
    import os

    git_dir = tmp_path / "actual-git-dir"
    git_dir.mkdir()
    (tmp_path / ".git").write_text("gitdir: actual-git-dir\n")
    roots = [os.path.realpath(str(tmp_path))]
    assert _is_workspace_git_path(str(git_dir / "config"), roots) is False


@pytest.mark.asyncio
async def test_workspace_git_auto_approval_resolves_relative_path_from_workspace_cwd(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relative file paths should be checked against the middleware workspace cwd."""
    workspace = tmp_path / "workspace"
    other = tmp_path / "other"
    workspace.mkdir()
    other.mkdir()
    (workspace / ".git").mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "file.txt").write_text("x")
    monkeypatch.chdir(other)

    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write"),
        event_bus=bus,
        workspace_roots=[str(workspace)],
        workspace_cwd=str(workspace),
    )

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)
    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx("write_file", "filesystem.write", {"path": "src/file.txt"}), _next)

    assert called is True
    assert not await wait_until(lambda: bool(requests), timeout=0.2, interval=0.01)
    assert mw.drain_decisions() == [{"request_id": "", "tool_name": "write_file", "status": "auto_approved"}]


# ──────────────────── property / mutator helpers ──────────────────────


def test_set_user_message_and_property_roundtrip() -> None:
    mw = ApprovalMiddleware(approval_policy=_require_all_policy(), event_bus=EventBus())
    assert mw._user_message == ""
    assert mw._user_messages == []
    mw.set_user_message("hello world")
    assert mw._user_message == "hello world"
    assert mw._user_messages == ["hello world"]


def test_set_user_messages_stores_current_turn_context_and_latest() -> None:
    mw = ApprovalMiddleware(approval_policy=_require_all_policy(), event_bus=EventBus())
    mw.set_user_messages(["repo overview", "how many lines?"])
    assert mw._user_messages == ["repo overview", "how many lines?"]
    assert mw._user_message == "how many lines?"


def test_set_approval_mode_and_property() -> None:
    mw = ApprovalMiddleware(approval_policy=_require_all_policy(), event_bus=EventBus())
    assert mw.approval_mode == ApprovalMode.MANUAL
    mw.set_approval_mode(ApprovalMode.BYPASS)
    assert mw.approval_mode == ApprovalMode.BYPASS


def test_drain_decisions_returns_copy_and_clears() -> None:
    mw = ApprovalMiddleware(approval_policy=_require_all_policy(), event_bus=EventBus())
    mw._decisions.append({"request_id": "a", "tool_name": "t", "status": "user_approved"})
    mw._decisions.append({"request_id": "b", "tool_name": "t", "status": "user_rejected"})

    out = mw.drain_decisions()
    assert len(out) == 2
    assert mw._decisions == []  # cleared
    # Mutating the returned list must not affect internal state
    out.append({"x": "y"})
    assert mw._decisions == []


def test_reset_clears_decisions_and_user_message() -> None:
    mw = ApprovalMiddleware(approval_policy=_require_all_policy(), event_bus=EventBus())
    mw._decisions.append({"request_id": "a", "tool_name": "t", "status": "user_approved"})
    mw.set_user_message("hi")

    mw.reset()
    assert mw._decisions == []
    assert mw._user_message == ""
    assert mw._user_messages == []


def test_retry_snapshot_drops_failed_decision_and_keeps_completed_baseline() -> None:
    middleware = ApprovalMiddleware(approval_policy=_require_all_policy(), event_bus=EventBus())
    baseline = {"request_id": "baseline", "tool_name": "write_file", "status": "user_approved"}
    failed = {"request_id": "failed", "tool_name": "write_file", "status": "user_rejected"}
    successful = {"request_id": "successful", "tool_name": "write_file", "status": "user_approved"}
    middleware._decisions.append(baseline)

    snapshot = middleware.snapshot_retry_state()
    middleware._decisions.append(failed)
    middleware.restore_retry_state(snapshot)
    middleware._decisions.append(successful)

    assert middleware.drain_decisions() == [baseline, successful]


# ───────────────────────── auto path (no approval) ─────────────────────


@pytest.mark.asyncio
async def test_should_not_require_calls_next_immediately() -> None:
    """Policy says no approval needed → call_next runs, no event published."""
    policy = ApprovalPolicy(ApprovalConfig(default="auto"))
    bus = EventBus()
    mw = ApprovalMiddleware(approval_policy=policy, event_bus=bus)

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx("any_tool", "", {}), _next)

    assert called
    assert not await wait_until(lambda: bool(requests), timeout=0.2, interval=0.01)


@pytest.mark.asyncio
async def test_dev_mode_requires_sub_agent_and_applies_modified_prompt() -> None:
    """Developer mode gates sub-agent handoff even when policy default is auto."""
    policy = ApprovalPolicy(ApprovalConfig(default="auto"))
    bus = EventBus()
    mw = ApprovalMiddleware(approval_policy=policy, event_bus=bus, dev_mode=True)
    ctx = _ctx("explore_agent", KIND_SUB_AGENT, {"prompt": "inspect src"})
    set_call_id(ctx, "call-123")

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    task = asyncio.create_task(mw.process(ctx, _next))
    assert await wait_until(lambda: len(requests) >= 1)

    assert not called
    assert len(requests) == 1
    assert requests[0].call_id == "call-123"
    assert requests[0].tool_kind == KIND_SUB_AGENT
    assert requests[0].judging is False
    # No fabricated intent: an "Execute {tool_name}" placeholder would win
    # over the ACP server's informative title fallbacks downstream.
    assert requests[0].intent_summary == ""

    await bus.publish(
        ApprovalResponse(
            request_id=requests[0].request_id,
            approved=True,
            modified_args={"prompt": "inspect src and tests"},
        )
    )
    await task

    assert called
    assert ctx.arguments == {"prompt": "inspect src and tests"}


@pytest.mark.asyncio
async def test_dev_mode_sub_agent_review_runs_in_auto_mode() -> None:
    """Developer-mode sub-agent handoff review runs in AUTO approval mode."""
    policy = ApprovalPolicy(ApprovalConfig(default="auto"))
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=policy,
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        dev_mode=True,
    )
    ctx = _ctx("explore_agent", KIND_SUB_AGENT, {"prompt": "inspect src"})

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    task = asyncio.create_task(mw.process(ctx, _next))
    assert await wait_until(lambda: len(requests) >= 1)

    assert not called
    assert len(requests) == 1
    assert requests[0].tool_name == "explore_agent"
    assert requests[0].judging is False

    await bus.publish(ApprovalResponse(request_id=requests[0].request_id, approved=True))
    await task

    assert called


@pytest.mark.asyncio
async def test_dev_mode_sub_agent_review_skips_bypass_mode() -> None:
    """BYPASS mode skips developer-mode sub-agent handoff review."""
    policy = ApprovalPolicy(ApprovalConfig(default="auto"))
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=policy,
        event_bus=bus,
        approval_mode=ApprovalMode.BYPASS,
        dev_mode=True,
    )
    ctx = _ctx("explore_agent", KIND_SUB_AGENT, {"prompt": "inspect src"})

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    task = asyncio.create_task(mw.process(ctx, _next))
    # BYPASS short-circuits: the task completes on its own, so awaiting it is
    # the barrier — no fixed sleep needed before asserting.
    await task

    assert called
    assert requests == []
    assert ctx.arguments == {"prompt": "inspect src"}


@pytest.mark.asyncio
async def test_dev_mode_modified_prompt_re_dispatches_blocking_hook() -> None:
    """Hook gating on ``args.prompt`` sees the user's edited prompt, not the original.

    Without re-dispatch, a hook that approves the original could let the edited
    prompt execute unchecked — the P1 bug this guards against.
    """
    from typing import Any

    from chrys.service.hooks.events import HookEvent
    from chrys.service.hooks.schema import HookDecision

    class _BlockingHookManager:
        def __init__(self, block_substring: str) -> None:
            self._block = block_substring
            self.payloads: list[dict[str, Any]] = []

        def has_hooks_for(self, event: HookEvent) -> bool:
            return event == HookEvent.BEFORE_TOOL_CALL

        async def fire(self, _event: HookEvent, payload: dict[str, Any], **_kwargs: object) -> HookDecision:
            self.payloads.append(payload)
            prompt = payload.get("tool", {}).get("args", {}).get("prompt", "")
            if self._block in prompt:
                return HookDecision(blocked=True, block_reason=f"forbidden token: {self._block}")
            return HookDecision()

    hook_mgr = _BlockingHookManager(block_substring="rm -rf")
    policy = ApprovalPolicy(ApprovalConfig(default="auto"))
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=policy,
        event_bus=bus,
        dev_mode=True,
        hook_manager=hook_mgr,
        profile_name="Code",
    )
    ctx = _ctx("explore_agent", KIND_SUB_AGENT, {"prompt": "inspect src"})
    set_call_id(ctx, "call-77")

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)
    updates: list[ToolCallArgsUpdated] = []

    async def _collect_update(ev: ToolCallArgsUpdated) -> None:
        updates.append(ev)

    await bus.subscribe(ToolCallArgsUpdated, _collect_update)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    task = asyncio.create_task(mw.process(ctx, _next))
    # Wait for the request to be pending before responding to it.
    assert await wait_until(lambda: len(requests) >= 1)

    # User edits the prompt to something the hook would have blocked at dispatch time.
    await bus.publish(
        ApprovalResponse(
            request_id=requests[0].request_id,
            approved=True,
            modified_args={"prompt": "rm -rf /"},
        )
    )
    await task

    assert called is False
    assert len(hook_mgr.payloads) == 1
    assert hook_mgr.payloads[0]["tool"]["args"] == {"prompt": "rm -rf /"}
    assert len(updates) == 1
    assert updates[0].call_id == "call-77"
    assert updates[0].args == {"prompt": "rm -rf /"}
    assert ctx.metadata.get("_approval_rejected") is True
    assert "forbidden token" in (ctx.result or "")


@pytest.mark.asyncio
async def test_re_dispatched_hooks_target_the_tool_operation_they_can_still_block() -> None:
    """The second dispatch gates the same call as the first, so it belongs to the same operation.

    Without the target, a hook that blocks the edited arguments is recorded
    against the exchange instead of the tool call it just stopped.
    """
    from typing import Any

    from chrys.foundation.trajectory.metadata import OPERATION_ID_KEY
    from chrys.service.hooks.events import HookEvent
    from chrys.service.hooks.schema import HookDecision

    class _RecordingHookManager:
        def __init__(self) -> None:
            self.targets: list[str | None] = []

        def has_hooks_for(self, event: HookEvent) -> bool:
            return event == HookEvent.BEFORE_TOOL_CALL

        async def fire(
            self, _event: HookEvent, _payload: dict[str, Any], *, target_operation_id: str | None = None
        ) -> HookDecision:
            self.targets.append(target_operation_id)
            return HookDecision()

    hook_mgr = _RecordingHookManager()
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=ApprovalPolicy(ApprovalConfig(default="auto")),
        event_bus=bus,
        dev_mode=True,
        hook_manager=hook_mgr,
        profile_name="Code",
    )
    ctx = _ctx("explore_agent", KIND_SUB_AGENT, {"prompt": "inspect src"})
    set_call_id(ctx, "call-88")
    ctx.metadata[OPERATION_ID_KEY] = "op-88"

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    async def _next() -> None:
        return None

    task = asyncio.create_task(mw.process(ctx, _next))
    assert await wait_until(lambda: len(requests) >= 1)
    await bus.publish(
        ApprovalResponse(
            request_id=requests[0].request_id,
            approved=True,
            modified_args={"prompt": "inspect tests"},
        )
    )
    await task

    assert hook_mgr.targets == ["op-88"]


@pytest.mark.asyncio
async def test_dev_mode_modified_prompt_re_dispatched_hook_can_rewrite() -> None:
    """A non-blocking hook can still rewrite edited args before execution."""
    from typing import Any

    from chrys.service.hooks.events import HookEvent
    from chrys.service.hooks.schema import HookDecision

    class _RewriteHookManager:
        def has_hooks_for(self, event: HookEvent) -> bool:
            return event == HookEvent.BEFORE_TOOL_CALL

        async def fire(self, _event: HookEvent, _payload: dict[str, Any], **_kwargs: object) -> HookDecision:
            return HookDecision(args_override={"prompt": "inspect src --safe"})

    policy = ApprovalPolicy(ApprovalConfig(default="auto"))
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=policy,
        event_bus=bus,
        dev_mode=True,
        hook_manager=_RewriteHookManager(),
        profile_name="Code",
    )
    ctx = _ctx("explore_agent", KIND_SUB_AGENT, {"prompt": "inspect src"})
    set_call_id(ctx, "call-88")

    requests: list[ApprovalRequest] = []
    updates: list[ToolCallArgsUpdated] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    async def _collect_update(ev: ToolCallArgsUpdated) -> None:
        updates.append(ev)

    await bus.subscribe(ToolCallArgsUpdated, _collect_update)

    async def _next() -> None:
        return None

    task = asyncio.create_task(mw.process(ctx, _next))
    # Wait for the request to be pending before responding to it.
    assert await wait_until(lambda: len(requests) >= 1)

    await bus.publish(
        ApprovalResponse(
            request_id=requests[0].request_id,
            approved=True,
            modified_args={"prompt": "inspect src and tests"},
        )
    )
    await task

    assert ctx.arguments == {"prompt": "inspect src --safe"}
    assert ctx.metadata.get("_approval_modified_args") == {"prompt": "inspect src --safe"}
    assert len(updates) == 1
    assert updates[0].call_id == "call-88"
    assert updates[0].args == {"prompt": "inspect src --safe"}


# ───────────────────────── BYPASS mode ─────────────────────────────────


@pytest.mark.asyncio
async def test_bypass_mode_short_circuits_and_records_decision() -> None:
    """BYPASS mode: no event published, call_next runs, decision recorded."""
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write"),
        event_bus=bus,
        approval_mode=ApprovalMode.BYPASS,
    )

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx("write_file", "filesystem.write", {"path": "/tmp/x"}), _next)

    assert called
    assert not await wait_until(lambda: bool(requests), timeout=0.2, interval=0.01)
    decisions = mw.drain_decisions()
    assert decisions == [{"request_id": "", "tool_name": "write_file", "status": "bypass_approved"}]


@pytest.mark.asyncio
async def test_safe_shell_auto_approval_records_alignment_placeholder() -> None:
    """Safe shell auto-approval should still keep decision order aligned."""
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("shell", tool_names=("zsh",)),
        event_bus=bus,
    )

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx("zsh", "shell", {"command": "ls"}), _next)

    assert called
    assert not await wait_until(lambda: bool(requests), timeout=0.2, interval=0.01)
    assert mw.drain_decisions() == [{"request_id": "", "tool_name": "zsh", "status": "auto_approved"}]


@pytest.mark.asyncio
async def test_workspace_git_file_auto_approval_records_alignment_placeholder(tmp_path) -> None:
    """Workspace-git file auto-approval should also keep decision order aligned."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    target = root / "file.txt"

    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write"),
        event_bus=bus,
        workspace_roots=[str(root)],
    )

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx("write_file", "filesystem.write", {"path": str(target)}), _next)

    assert called
    assert not await wait_until(lambda: bool(requests), timeout=0.2, interval=0.01)
    assert mw.drain_decisions() == [{"request_id": "", "tool_name": "write_file", "status": "auto_approved"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("target", [".git/config", ".git/hooks/pre-commit", ".git/hooks/post-checkout"])
async def test_workspace_git_internals_require_approval_in_manual_mode(tmp_path, target: str) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git" / "hooks").mkdir(parents=True)

    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write"),
        event_bus=bus,
        approval_mode=ApprovalMode.MANUAL,
        workspace_roots=[str(root)],
        workspace_cwd=str(root),
    )

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    task = asyncio.create_task(mw.process(_ctx("write_file", "filesystem.write", {"path": target}), _next))
    assert await wait_until(lambda: len(requests) >= 1)

    assert called is False
    assert requests[0].args == {"path": target}

    await bus.publish(ApprovalResponse(request_id=requests[0].request_id, approved=False, reason="review required"))
    await task

    assert called is False
    assert mw.drain_decisions() == [
        {
            "request_id": requests[0].request_id,
            "tool_name": "write_file",
            "status": "user_rejected",
            "reason": "review required",
        }
    ]


@pytest.mark.asyncio
async def test_session_archive_read_auto_approved_even_under_require_policy(tmp_path) -> None:
    """Reads of session compaction archives never show the approval dialog."""
    archive_root = tmp_path / "compactions"
    record = archive_root / "dropped" / "turn001" / "001_read_file_aaaaaaaa.md"
    record.parent.mkdir(parents=True)
    record.write_text("archived", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("not archived", encoding="utf-8")

    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.read", tool_names=("read_file",)),
        event_bus=bus,
        approval_mode=ApprovalMode.MANUAL,
        session_archive_read_roots=[archive_root],
    )

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx("read_file", "filesystem.read", {"path": str(record)}), _next)

    assert called
    assert not await wait_until(lambda: bool(requests), timeout=0.2, interval=0.01)
    assert mw.drain_decisions() == [{"request_id": "", "tool_name": "read_file", "status": "auto_approved"}]

    # Control: the same policy still gates reads outside the archive root.
    called = False
    task = asyncio.create_task(mw.process(_ctx("read_file", "filesystem.read", {"path": str(outside)}), _next))
    assert await wait_until(lambda: len(requests) >= 1)
    assert called is False
    await bus.publish(ApprovalResponse(request_id=requests[0].request_id, approved=True))
    await task
    assert called


@pytest.mark.asyncio
async def test_session_archive_read_overrides_sensitive_path_classifier(tmp_path) -> None:
    """A sensitive-looking archived component (e.g. a sub-agent tool named
    ``credentials``) is whitelisted inside the archive root instead of gated."""
    archive_root = tmp_path / "compactions"
    record = archive_root / "sub_agents" / "credentials" / "inv1" / "dropped" / "turn001" / "001_fetch_bbbbbbbb.md"
    record.parent.mkdir(parents=True)
    record.write_text("archived", encoding="utf-8")

    bus = EventBus()
    mw = ApprovalMiddleware(
        # Default policy: reads gate only through the sensitive classifier.
        approval_policy=ApprovalPolicy(ApprovalConfig(default="auto")),
        event_bus=bus,
        approval_mode=ApprovalMode.MANUAL,
        session_archive_read_roots=[archive_root],
    )

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx("read_file", "filesystem.read", {"path": str(record)}), _next)

    assert called
    assert not await wait_until(lambda: bool(requests), timeout=0.2, interval=0.01)
    assert mw.drain_decisions() == [{"request_id": "", "tool_name": "read_file", "status": "auto_approved"}]


@pytest.mark.asyncio
async def test_session_archive_read_symlink_escape_is_not_auto_approved(tmp_path) -> None:
    """A link planted inside the archive cannot smuggle an outside read."""
    import os

    archive_root = tmp_path / "compactions"
    (archive_root / "dropped").mkdir(parents=True)
    secret = tmp_path / "secret.md"
    secret.write_text("outside", encoding="utf-8")
    link = archive_root / "dropped" / "link.md"
    os.symlink(secret, link)

    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.read", tool_names=("read_file",)),
        event_bus=bus,
        approval_mode=ApprovalMode.MANUAL,
        session_archive_read_roots=[archive_root],
    )

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    task = asyncio.create_task(mw.process(_ctx("read_file", "filesystem.read", {"path": str(link)}), _next))
    assert await wait_until(lambda: len(requests) >= 1)
    assert called is False
    await bus.publish(ApprovalResponse(request_id=requests[0].request_id, approved=False, reason="escape"))
    await task
    assert called is False


@pytest.mark.asyncio
async def test_parallel_same_name_manual_decision_keeps_invocation_order() -> None:
    """A later safe auto-approval must not leapfrog an earlier pending request."""
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("shell", tool_names=("zsh",)),
        event_bus=bus,
    )

    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    unsafe_called = False
    safe_called = False

    async def _unsafe_next() -> None:
        nonlocal unsafe_called
        unsafe_called = True

    async def _safe_next() -> None:
        nonlocal safe_called
        safe_called = True

    unsafe_ctx = _ctx("zsh", "shell", {"command": "rm file"})
    safe_ctx = _ctx("zsh", "shell", {"command": "ls"})
    set_call_id(unsafe_ctx, "unsafe-call")
    set_call_id(safe_ctx, "safe-call")
    set_tool_invocation_order(unsafe_ctx, 0)
    set_tool_invocation_order(safe_ctx, 1)

    unsafe_task = asyncio.create_task(mw.process(unsafe_ctx, _unsafe_next))
    assert await wait_until(lambda: bool(requests))
    assert len(requests) == 1

    await mw.process(safe_ctx, _safe_next)
    assert safe_called

    await bus.publish(ApprovalResponse(request_id=requests[0].request_id, approved=False, reason="too broad"))
    await unsafe_task

    assert not unsafe_called
    assert mw.drain_decisions() == [
        {
            "request_id": requests[0].request_id,
            "tool_name": "zsh",
            "status": "user_rejected",
            "call_id": "unsafe-call",
            "tool_order": "0",
            "reason": "too broad",
        },
        {
            "request_id": "",
            "tool_name": "zsh",
            "status": "auto_approved",
            "call_id": "safe-call",
            "tool_order": "1",
        },
    ]


# ───────────────────────── args parsing branches ───────────────────────


@pytest.mark.asyncio
async def test_args_json_string_is_parsed() -> None:
    """When ``context.arguments`` is a JSON string, it should be decoded."""
    bus = EventBus()
    mw = ApprovalMiddleware(approval_policy=_require_all_policy("filesystem.write"), event_bus=bus)

    seen: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        seen.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    ctx = _ctx("write_file", "filesystem.write", '{"path": "/tmp/a.txt", "content": "x"}')

    async def _noop() -> None:
        pass

    task = asyncio.create_task(mw.process(ctx, _noop))
    assert await wait_until(lambda: len(seen) >= 1)

    assert len(seen) == 1
    assert seen[0].args == {"path": "/tmp/a.txt", "content": "x"}

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_args_invalid_json_string_becomes_empty_dict() -> None:
    """Malformed JSON string is swallowed; args default to ``{}``."""
    bus = EventBus()
    mw = ApprovalMiddleware(approval_policy=_require_all_policy("filesystem.write"), event_bus=bus)

    seen: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        seen.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    ctx = _ctx("write_file", "filesystem.write", "not a json {{{")

    async def _noop() -> None:
        pass

    task = asyncio.create_task(mw.process(ctx, _noop))
    assert await wait_until(lambda: len(seen) >= 1)

    assert len(seen) == 1
    assert seen[0].args == {}

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_args_none_becomes_empty_dict() -> None:
    """Non-dict, non-str arguments (e.g. None) fall through to ``{}``."""
    bus = EventBus()
    mw = ApprovalMiddleware(approval_policy=_require_all_policy("filesystem.write"), event_bus=bus)

    seen: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        seen.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    ctx = _ctx("write_file", "filesystem.write", None)

    async def _noop() -> None:
        pass

    task = asyncio.create_task(mw.process(ctx, _noop))
    assert await wait_until(lambda: len(seen) >= 1)

    assert len(seen) == 1
    assert seen[0].args == {}

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ───────────────────────── approval flow ───────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "tool_kind", "override_key"),
    [
        ("load_skill", KIND_SKILL, "skill"),
        ("compress_context", KIND_CONTEXT, "context"),
    ],
)
async def test_runtime_added_tool_kind_rule_is_enforced_by_middleware(
    tool_name: str,
    tool_kind: str,
    override_key: str,
) -> None:
    """The middleware passes live provenance for dynamically injected tools."""
    bus = EventBus()
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={override_key: "require"}),
        tools=[],
    )
    mw = ApprovalMiddleware(approval_policy=policy, event_bus=bus)
    requests: list[ApprovalRequest] = []

    async def _respond(ev: ApprovalRequest) -> None:
        requests.append(ev)
        asyncio.create_task(  # noqa: RUF006
            bus.publish(ApprovalResponse(request_id=ev.request_id, approved=True, reason="approved"))
        )

    await bus.subscribe(ApprovalRequest, _respond)
    next_called = False

    async def _next() -> None:
        nonlocal next_called
        next_called = True

    await mw.process(_ctx(tool_name, tool_kind, {}), _next)

    assert next_called
    assert len(requests) == 1
    assert requests[0].tool_kind == tool_kind


@pytest.mark.asyncio
async def test_run_skill_script_floor_rejects_before_tool_process_can_start() -> None:
    """An auto-default custom profile still gates the skill subprocess call."""
    bus = EventBus()
    policy = ApprovalPolicy(ApprovalConfig(default="auto", overrides={}), tools=[])
    mw = ApprovalMiddleware(approval_policy=policy, event_bus=bus)
    requests: list[ApprovalRequest] = []

    async def _respond(ev: ApprovalRequest) -> None:
        requests.append(ev)
        asyncio.create_task(  # noqa: RUF006
            bus.publish(ApprovalResponse(request_id=ev.request_id, approved=False, reason="untrusted skill"))
        )

    await bus.subscribe(ApprovalRequest, _respond)
    tool_process_started = False

    async def _start_tool_process() -> None:
        nonlocal tool_process_started
        tool_process_started = True

    ctx = _ctx("run_skill_script", KIND_SKILL, {"skill_name": "example", "script_path": "scripts/run.py"})
    await mw.process(ctx, _start_tool_process)

    assert not tool_process_started
    assert len(requests) == 1
    assert requests[0].tool_kind == KIND_SKILL
    assert ctx.result == "Error: Tool execution was rejected by user.\nUser reason: untrusted skill"


@pytest.mark.asyncio
async def test_user_approval_runs_next_and_records_decision() -> None:
    """Approved response → call_next runs, ``user_approved`` recorded."""
    bus = EventBus()
    mw = ApprovalMiddleware(approval_policy=_require_all_policy("filesystem.write"), event_bus=bus)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    ctx = _ctx("write_file", "filesystem.write", {"path": "/tmp/x"})

    async def _respond(rid: str) -> None:
        await bus.publish(ApprovalResponse(request_id=rid, approved=True, reason="looks fine"))

    async def _responder(ev: ApprovalRequest) -> None:
        asyncio.create_task(_respond(ev.request_id))  # noqa: RUF006

    await bus.subscribe(ApprovalRequest, _responder)
    await mw.process(ctx, _next)

    assert called
    decisions = mw.drain_decisions()
    assert len(decisions) == 1
    assert decisions[0]["status"] == "user_approved"
    assert decisions[0]["reason"] == "looks fine"


@pytest.mark.asyncio
async def test_user_rejection_returns_error_and_sets_flag() -> None:
    """Rejected response → error string set, call_next NOT run, metadata flagged."""
    from chrys.service.agent_middleware._metadata_keys import (
        _APPROVAL_REJECTED_KEY,
        _REJECTION_MESSAGE_KEY,
        _REJECTION_SOURCE_KEY,
    )

    bus = EventBus()
    mw = ApprovalMiddleware(approval_policy=_require_all_policy("filesystem.write"), event_bus=bus)

    next_called = False

    async def _next() -> None:
        nonlocal next_called
        next_called = True

    ctx = _ctx("write_file", "filesystem.write", {"path": "/tmp/x"})

    async def _respond(rid: str) -> None:
        await bus.publish(ApprovalResponse(request_id=rid, approved=False, reason=""))

    async def _responder(ev: ApprovalRequest) -> None:
        asyncio.create_task(_respond(ev.request_id))  # noqa: RUF006

    await bus.subscribe(ApprovalRequest, _responder)
    await mw.process(ctx, _next)

    assert not next_called
    assert ctx.result == "Error: Tool execution was rejected by user."
    assert ctx.metadata.get(_APPROVAL_REJECTED_KEY) is True
    assert ctx.metadata.get(_REJECTION_SOURCE_KEY) == "user"
    assert ctx.metadata.get(_REJECTION_MESSAGE_KEY) == "Tool execution was rejected by user."

    decisions = mw.drain_decisions()
    assert decisions[0]["status"] == "user_rejected"
    # No reason in decision when empty
    assert "reason" not in decisions[0]


@pytest.mark.asyncio
async def test_user_rejection_includes_reason_in_error_result() -> None:
    """Rejected response with reason → tool result tells the model why it was rejected."""
    from chrys.service.agent_middleware._metadata_keys import (
        _APPROVAL_REJECTED_KEY,
        _REJECTION_MESSAGE_KEY,
        _REJECTION_SOURCE_KEY,
    )

    bus = EventBus()
    mw = ApprovalMiddleware(approval_policy=_require_all_policy("filesystem.write"), event_bus=bus)
    ctx = _ctx("write_file", "filesystem.write", {"path": "/tmp/x"})

    async def _respond(rid: str) -> None:
        await bus.publish(ApprovalResponse(request_id=rid, approved=False, reason="  Use a safer path  "))

    async def _responder(ev: ApprovalRequest) -> None:
        asyncio.create_task(_respond(ev.request_id))  # noqa: RUF006

    await bus.subscribe(ApprovalRequest, _responder)
    await mw.process(ctx, MagicMock())

    assert ctx.result == "Error: Tool execution was rejected by user.\nUser reason: Use a safer path"
    assert ctx.metadata.get(_APPROVAL_REJECTED_KEY) is True
    assert ctx.metadata.get(_REJECTION_SOURCE_KEY) == "user"
    assert ctx.metadata.get(_REJECTION_MESSAGE_KEY) == (
        "Tool execution was rejected by user. User reason: Use a safer path"
    )

    decisions = mw.drain_decisions()
    assert decisions[0]["status"] == "user_rejected"
    assert decisions[0]["reason"] == "Use a safer path"


# ───────────────────────── AUTO + judge ────────────────────────────────


class _FakeJudge:
    """Stand-in for ``ApprovalJudge``: returns a pre-built verdict (or raises)."""

    def __init__(
        self,
        verdict: JudgeVerdict | None = None,
        exc: BaseException | None = None,
        delay: float = 0.0,
    ) -> None:
        self.verdict = verdict
        self.exc = exc
        self.delay = delay
        self.called_with: dict | None = None

    async def evaluate(self, **kwargs) -> JudgeVerdict:
        self.called_with = kwargs
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.exc is not None:
            raise self.exc
        assert self.verdict is not None
        return self.verdict


@pytest.mark.asyncio
async def test_auto_judge_approved_auto_fulfils(tmp_path) -> None:
    """AUTO + judge returns approved → future is fulfilled automatically; ApprovalReviewed published; call_next runs."""
    bus = EventBus()
    judge = _FakeJudge(verdict=JudgeVerdict(approved=True, reason="safe read"))
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write"),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
        approval_log_dir=tmp_path,
        session_id="sess-1",
    )

    reviews: list[ApprovalReviewed] = []
    requests: list[ApprovalRequest] = []

    async def _rev(ev: ApprovalReviewed) -> None:
        reviews.append(ev)

    async def _req(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalReviewed, _rev)
    await bus.subscribe(ApprovalRequest, _req)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    ctx = _ctx("write_file", "filesystem.write", {"path": "/tmp/x"})
    mw.set_user_message("please write")

    await mw.process(ctx, _next)

    assert called
    assert len(requests) == 1
    assert requests[0].judging is True  # dialog shows "evaluating"
    assert len(reviews) == 1
    assert reviews[0].approved is True
    assert reviews[0].reason == "safe read"

    # Judge was called with user message + args
    assert judge.called_with is not None
    assert judge.called_with["user_message"] == "please write"
    assert judge.called_with["user_messages"] == ["please write"]
    assert judge.called_with["tool_name"] == "write_file"
    assert judge.called_with["log_dir"] == tmp_path


@pytest.mark.parametrize(
    "command",
    [
        "cat .env",
        "printenv",
        "env -0",
        "env --null",
        "env -u SECRET",
        "env VAR=value printenv",
        "env VAR=value env",
        'env FOO="bar baz" printenv',
        "env FOO='bar baz' env",
        r"env FOO=bar\ baz printenv",
        "cat ~/.aws/credentials",
        "echo $OPENAI_API_KEY",
        'printf "%s\n" "$GITHUB_TOKEN"',
        "echo ${AWS_SECRET_ACCESS_KEY}",
        "echo \"'$OPENAI_API_KEY'\"",
    ],
)
@pytest.mark.asyncio
async def test_auto_judge_reviews_sensitive_readonly_shell_commands(command: str, tmp_path) -> None:
    """Sensitive read-only-looking shell commands must not bypass the judge."""
    bus = EventBus()
    judge = _FakeJudge(verdict=JudgeVerdict(approved=True, reason="reviewed"))
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("shell", tool_names=("zsh",)),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
        approval_log_dir=tmp_path,
    )

    requests: list[ApprovalRequest] = []

    async def _req(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _req)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx("zsh", "shell", {"command": command}), _next)

    assert called
    assert len(requests) == 1
    assert requests[0].judging is True
    assert judge.called_with is not None
    assert judge.called_with["args"]["command"] == command
    assert mw.drain_decisions()[0]["status"] == "user_approved"


@pytest.mark.asyncio
async def test_auto_judge_reviews_cmd_sensitive_environment_reads(tmp_path) -> None:
    """cmd-style %VAR% secret reads must not use shell read-only auto approval."""
    bus = EventBus()
    judge = _FakeJudge(verdict=JudgeVerdict(approved=True, reason="reviewed"))
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("shell", tool_names=("cmd",)),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
        approval_log_dir=tmp_path,
    )

    requests: list[ApprovalRequest] = []

    async def _req(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _req)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx("cmd", "shell", {"command": "echo %OPENAI_API_KEY%"}), _next)

    assert called
    assert len(requests) == 1
    assert requests[0].judging is True
    assert judge.called_with is not None
    assert judge.called_with["args"]["command"] == "echo %OPENAI_API_KEY%"
    assert mw.drain_decisions()[0]["status"] == "user_approved"


@pytest.mark.parametrize(
    "command",
    [
        "env FOO=bar ls",
        "env NODE_ENV=production npm --version",
        'env FOO="bar baz" ls',
        r"env FOO=bar\ baz ls",
    ],
)
@pytest.mark.asyncio
async def test_env_assignment_prefix_shell_child_programs_go_to_approval(command: str, tmp_path) -> None:
    """env NAME=value <program> can execute arbitrary code, so it is never auto-approved."""
    bus = EventBus()
    judge = _FakeJudge(verdict=JudgeVerdict(approved=True, reason="reviewed"))
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("shell", tool_names=("zsh",)),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
        approval_log_dir=tmp_path,
    )

    requests: list[ApprovalRequest] = []

    async def _req(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _req)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx("zsh", "shell", {"command": command}), _next)

    assert called
    assert len(requests) == 1
    assert requests[0].judging is True
    assert judge.called_with is not None
    assert judge.called_with["args"]["command"] == command
    assert mw.drain_decisions()[0]["status"] == "user_approved"


@pytest.mark.asyncio
async def test_quoted_sensitive_env_reference_keeps_readonly_auto_approval(tmp_path) -> None:
    """Single-quoted env references are inert text, not secret expansion."""
    bus = EventBus()
    judge = _FakeJudge(verdict=JudgeVerdict(approved=True, reason="reviewed"))
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("shell", tool_names=("zsh",)),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
        approval_log_dir=tmp_path,
    )

    requests: list[ApprovalRequest] = []

    async def _req(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _req)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx("zsh", "shell", {"command": "echo '$OPENAI_API_KEY'"}), _next)

    assert called
    assert requests == []
    assert judge.called_with is None
    assert mw.drain_decisions()[0]["status"] == "auto_approved"


@pytest.mark.parametrize(
    ("tool_name", "tool_kind", "args"),
    [
        ("zsh", "shell", {"command": "cat .env"}),
        ("zsh", "shell", {"command": "echo $OPENAI_API_KEY"}),
        ("zsh", "shell", {"command": "echo $(printenv)"}),
        ("zsh", "shell", {"command": "echo `printenv`"}),
        ("zsh", "shell", {"command": "cat <(printenv)"}),
        ("zsh", "shell", {"command": "bash -lc 'echo $OPENAI_API_KEY'"}),
        ("zsh", "shell", {"command": "echo ok\nprintenv"}),
        ("zsh", "shell", {"command": "echo ok\nbash -lc 'echo $OPENAI_API_KEY'"}),
        ("zsh", "shell", {"command": "bash --norc -c 'printenv'"}),
        ("zsh", "shell", {"command": "bash --rcfile x -c 'printenv'"}),
        ("zsh", "shell", {"command": "bash --rcfile=x -c 'printenv'"}),
        ("zsh", "shell", {"command": 'command sh -c "printenv"'}),
        ("zsh", "shell", {"command": 'sudo sh -c "printenv"'}),
        ("zsh", "shell", {"command": 'echo $(command sh -c "printenv")'}),
        ("pwsh", "shell", {"command": "Write-Output ok\nGet-ChildItem Env:"}),
        ("pwsh", "shell", {"command": "Write-Output ${Env:OPENAI_API_KEY}"}),
        ("write_file", "filesystem.write", {"path": ".env", "content": "TOKEN=x"}),
    ],
)
@pytest.mark.asyncio
async def test_auto_judge_reviews_sensitive_shell_and_write_even_when_policy_is_auto(
    tool_name: str,
    tool_kind: str,
    args: dict[str, str],
    tmp_path,
) -> None:
    """Sensitive shell/write targets are hard approval overrides, not policy-auto calls."""
    bus = EventBus()
    judge = _FakeJudge(verdict=JudgeVerdict(approved=True, reason="reviewed"))
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto", overrides={}),
        tools=[
            _mock_tool("zsh", kind="shell"),
            _mock_tool("pwsh", kind="shell"),
            _mock_tool("write_file", kind="filesystem.write"),
        ],
    )
    mw = ApprovalMiddleware(
        approval_policy=policy,
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
        approval_log_dir=tmp_path,
    )

    requests: list[ApprovalRequest] = []

    async def _req(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _req)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx(tool_name, tool_kind, args), _next)

    assert called
    assert len(requests) == 1
    assert requests[0].judging is True
    assert judge.called_with is not None
    assert judge.called_with["tool_name"] == tool_name
    assert judge.called_with["args"] == args
    assert mw.drain_decisions()[0]["status"] == "user_approved"


@pytest.mark.parametrize(
    "command",
    [
        "Get-ChildItem Env:",
        "Get-Item Env:OPENAI_API_KEY",
        "Get-Content Env:OPENAI_API_KEY",
        "dir Env:",
        "Get-ChildItem Env:\\",
        r"Get-Item Env:\OPENAI_API_KEY",
        r"Get-Content Env:\OPENAI_API_KEY",
        "dir Env:\\",
        r"cat Env:\OPENAI_API_KEY",
        "Write-Output $Env:OPENAI_API_KEY",
        "Write-Output ${Env:OPENAI_API_KEY}",
        "Write-Output ok\nGet-ChildItem Env:",
    ],
)
@pytest.mark.asyncio
async def test_auto_judge_reviews_powershell_environment_reads(command: str, tmp_path) -> None:
    """PowerShell Env: provider reads can expose secrets and must be judged."""
    bus = EventBus()
    judge = _FakeJudge(verdict=JudgeVerdict(approved=True, reason="reviewed"))
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("shell", tool_names=("pwsh",)),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
        approval_log_dir=tmp_path,
    )

    requests: list[ApprovalRequest] = []

    async def _req(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _req)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx("pwsh", "shell", {"command": command}), _next)

    assert called
    assert len(requests) == 1
    assert requests[0].judging is True
    assert judge.called_with is not None
    assert judge.called_with["args"]["command"] == command
    assert mw.drain_decisions()[0]["status"] == "user_approved"


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("read_file", {"path": ".env"}),
        ("read_file", {"path": ".envrc"}),
        ("view_image", {"path": ".aws/credentials"}),
        ("read_file", {"path": "aws_credentials"}),
        ("read_file", {"path": "credentials.yaml"}),
        ("read_file", {"path": ".config/gcloud/application_default_credentials.json"}),
        ("read_file", {"path": "~/.git-credentials"}),
        ("read_file", {"path": "~/Library/Keychains/login.keychain-db"}),
        ("read_file", {"path": "~/.kube/config"}),
        ("read_file", {"path": "~/.config/gh/hosts.yml"}),
        ("read_file", {"path": "~/.azure/accessTokens.json"}),
        ("read_file", {"path": "~/.azure/msal_token_cache.bin"}),
    ],
)
@pytest.mark.asyncio
async def test_auto_judge_reviews_sensitive_filesystem_reads_even_when_policy_is_auto(
    tool_name: str,
    args: dict[str, str],
    tmp_path,
) -> None:
    """Sensitive read targets must not silently pass through policy default=auto."""
    bus = EventBus()
    judge = _FakeJudge(verdict=JudgeVerdict(approved=True, reason="reviewed"))
    policy = ApprovalPolicy(
        ApprovalConfig(default="auto"),
        tools=[_mock_tool("read_file", kind="filesystem.read"), _mock_tool("view_image", kind="filesystem.read")],
    )
    mw = ApprovalMiddleware(
        approval_policy=policy,
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
        approval_log_dir=tmp_path,
    )

    requests: list[ApprovalRequest] = []

    async def _req(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _req)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx(tool_name, "filesystem.read", args), _next)

    assert called
    assert len(requests) == 1
    assert requests[0].judging is True
    assert judge.called_with is not None
    assert judge.called_with["tool_name"] == tool_name
    assert judge.called_with["args"] == args
    assert mw.drain_decisions()[0]["status"] == "user_approved"


@pytest.mark.parametrize(
    ("tool_name", "args"),
    [
        ("write_file", {"path": ".env", "content": "TOKEN=x"}),
        ("write_file", {"path": ".envrc", "content": "export TOKEN=x"}),
        ("edit_file", {"path": ".npmrc", "old_string": "a", "new_string": "b"}),
        ("write_file", {"path": ".aws/credentials", "content": "secret"}),
        ("write_file", {"path": "aws_credentials", "content": "secret"}),
        ("edit_file", {"path": "credentials.yml", "old_string": "a", "new_string": "b"}),
        (
            "write_file",
            {"path": ".config/gcloud/application_default_credentials.json", "content": "secret"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_auto_judge_reviews_sensitive_filesystem_writes_in_git_workspace(
    tool_name: str,
    args: dict[str, str],
    tmp_path,
) -> None:
    """Sensitive write/edit targets must not use workspace-git auto-approval."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    if args["path"].startswith(".aws/"):
        (root / ".aws").mkdir()

    bus = EventBus()
    judge = _FakeJudge(verdict=JudgeVerdict(approved=True, reason="reviewed"))
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write", tool_names=("write_file", "edit_file")),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
        workspace_roots=[str(root)],
        workspace_cwd=str(root),
    )

    requests: list[ApprovalRequest] = []

    async def _req(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _req)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx(tool_name, "filesystem.write", args), _next)

    assert called
    assert len(requests) == 1
    assert requests[0].judging is True
    assert judge.called_with is not None
    assert judge.called_with["tool_name"] == tool_name
    assert judge.called_with["args"] == args
    assert mw.drain_decisions()[0]["status"] == "user_approved"


@pytest.mark.asyncio
async def test_auto_judge_receives_all_current_turn_user_messages(tmp_path) -> None:
    """AUTO judge sees prior current-turn user messages plus the latest one."""
    bus = EventBus()
    judge = _FakeJudge(verdict=JudgeVerdict(approved=True, reason="safe read"))
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write"),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
        approval_log_dir=tmp_path,
    )

    async def _next() -> None:
        return None

    mw.set_user_messages(["这个代码仓是做什么的?", "当前代码仓有多少行代码?"])
    await mw.process(_ctx("write_file", "filesystem.write", {"path": "/tmp/x"}), _next)

    assert judge.called_with is not None
    assert judge.called_with["user_message"] == "当前代码仓有多少行代码?"
    assert judge.called_with["user_messages"] == ["这个代码仓是做什么的?", "当前代码仓有多少行代码?"]


@pytest.mark.asyncio
async def test_auto_judge_approved_blocked_by_frontend_waits_for_user_response() -> None:
    """If the TUI has a user decision in flight, judge approval must not win the race."""
    bus = EventBus()
    judge = _FakeJudge(verdict=JudgeVerdict(approved=True, reason="safe enough"))
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write"),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
    )

    reviews: list[ApprovalReviewed] = []

    async def _rev(ev: ApprovalReviewed) -> None:
        reviews.append(ev)
        await bus.publish(ApprovalAutoFulfillBlocked(request_id=ev.request_id))
        await bus.publish(ApprovalResponse(request_id=ev.request_id, approved=False, reason="User declined"))

    await bus.subscribe(ApprovalReviewed, _rev)

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    ctx = _ctx("write_file", "filesystem.write", {"path": "/tmp/x"})
    await mw.process(ctx, _next)

    assert len(reviews) == 1
    assert not called
    assert ctx.result == "Error: Tool execution was rejected by user.\nUser reason: User declined"
    assert mw.drain_decisions() == [
        {
            "request_id": reviews[0].request_id,
            "tool_name": "write_file",
            "status": "user_rejected",
            "reason": "User declined",
        }
    ]


@pytest.mark.asyncio
async def test_auto_fulfill_block_subscription_close_unsubscribes() -> None:
    """ApprovalMiddleware.close should release its EventBus handler."""
    bus = EventBus()
    judge = _FakeJudge(verdict=JudgeVerdict(approved=True, reason="safe enough"))
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write"),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
    )

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    await mw.process(_ctx("write_file", "filesystem.write", {"path": "/tmp/x"}), _next)
    assert called
    assert mw._on_auto_fulfill_blocked in bus._handlers[ApprovalAutoFulfillBlocked]

    await mw.close()

    assert mw._on_auto_fulfill_blocked not in bus._handlers[ApprovalAutoFulfillBlocked]
    assert mw._approval_arbiter._subscribed is False


@pytest.mark.asyncio
async def test_auto_fulfill_block_subscription_close_is_idempotent() -> None:
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write"),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=_FakeJudge(verdict=JudgeVerdict(approved=True, reason="safe enough")),
    )

    await mw._ensure_auto_fulfill_block_subscription()
    await mw.close()
    await mw.close()

    assert mw._on_auto_fulfill_blocked not in bus._handlers[ApprovalAutoFulfillBlocked]


@pytest.mark.asyncio
async def test_auto_judge_flagged_waits_for_user(tmp_path) -> None:
    """AUTO + judge flags → future is NOT auto-fulfilled; user still decides."""
    bus = EventBus()
    judge = _FakeJudge(verdict=JudgeVerdict(approved=False, reason="rm -rf detected"))
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write"),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
    )

    reviews: list[ApprovalReviewed] = []

    async def _rev(ev: ApprovalReviewed) -> None:
        reviews.append(ev)

    await bus.subscribe(ApprovalReviewed, _rev)

    # User approves manually after the judge has flagged. Use a background
    # task so the on_request handler returns immediately — otherwise it
    # blocks publish() and prevents the judge task from being created.
    async def _respond_later(rid: str) -> None:
        # Respond only after the judge has flagged (review published), so the
        # manual override deterministically follows the judge verdict.
        await wait_for(
            lambda: len(reviews) >= 1,
            timeout=ENGINE_TURN_TIMEOUT,
            description="approval review before response",
        )
        await bus.publish(ApprovalResponse(request_id=rid, approved=True, reason="user override"))

    called = False

    async def _next() -> None:
        nonlocal called
        called = True

    ctx = _ctx("write_file", "filesystem.write", {"path": "/tmp/x"})
    async with asyncio.TaskGroup() as responders:

        async def _on_request(ev: ApprovalRequest) -> None:
            responders.create_task(_respond_later(ev.request_id))

        await bus.subscribe(ApprovalRequest, _on_request)
        await mw.process(ctx, _next)

    assert called
    assert len(reviews) == 1
    assert reviews[0].approved is False
    assert reviews[0].reason == "rm -rf detected"


@pytest.mark.asyncio
async def test_auto_judge_exception_flags_as_failure() -> None:
    """If the judge raises, publish a flagged ApprovalReviewed with the error reason."""
    bus = EventBus()
    judge = _FakeJudge(exc=RuntimeError("boom"))
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write"),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
    )

    reviews: list[ApprovalReviewed] = []

    async def _rev(ev: ApprovalReviewed) -> None:
        reviews.append(ev)

    await bus.subscribe(ApprovalReviewed, _rev)

    async def _respond_later(rid: str) -> None:
        # Respond only after the judge has flagged the failure (review
        # published), so the manual response follows the judge verdict.
        await wait_for(
            lambda: len(reviews) >= 1,
            timeout=ENGINE_TURN_TIMEOUT,
            description="rewritten approval review before response",
        )
        await bus.publish(ApprovalResponse(request_id=rid, approved=True, reason=""))

    async def _noop() -> None:
        pass

    async with asyncio.TaskGroup() as responders:

        async def _on_request(ev: ApprovalRequest) -> None:
            responders.create_task(_respond_later(ev.request_id))

        await bus.subscribe(ApprovalRequest, _on_request)
        await mw.process(_ctx("write_file", "filesystem.write", {"path": "/x"}), _noop)

    assert len(reviews) == 1
    assert reviews[0].approved is False
    assert "boom" in reviews[0].reason


@pytest.mark.asyncio
async def test_auto_judge_cancelled_when_user_decides_first() -> None:
    """User responds before judge finishes → judge task is cancelled, no ApprovalReviewed published."""
    bus = EventBus()
    # Slow judge relative to the immediate user response.
    judge = _FakeJudge(verdict=JudgeVerdict(approved=True, reason="eventual"), delay=0.01)
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write"),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
    )

    reviews: list[ApprovalReviewed] = []

    async def _rev(ev: ApprovalReviewed) -> None:
        reviews.append(ev)

    await bus.subscribe(ApprovalReviewed, _rev)

    async def _respond_fast(rid: str) -> None:
        # User decides almost immediately (0 delay — runs as soon as control yields)
        await bus.publish(ApprovalResponse(request_id=rid, approved=True, reason="I trust it"))

    async def _on_request(ev: ApprovalRequest) -> None:
        asyncio.create_task(_respond_fast(ev.request_id))  # noqa: RUF006

    await bus.subscribe(ApprovalRequest, _on_request)

    async def _noop() -> None:
        pass

    await mw.process(_ctx("write_file", "filesystem.write", {"path": "/x"}), _noop)

    # Judge was cancelled mid-sleep → no review ever published
    assert reviews == []


@pytest.mark.asyncio
async def test_auto_mode_without_judge_falls_back_to_manual() -> None:
    """AUTO mode but no judge configured → behaves like MANUAL (no judging, no review)."""
    bus = EventBus()
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write"),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=None,
    )

    requests: list[ApprovalRequest] = []
    reviews: list[ApprovalReviewed] = []

    async def _req(ev: ApprovalRequest) -> None:
        requests.append(ev)

    async def _rev(ev: ApprovalReviewed) -> None:
        reviews.append(ev)

    await bus.subscribe(ApprovalRequest, _req)
    await bus.subscribe(ApprovalReviewed, _rev)

    async def _respond(rid: str) -> None:
        await bus.publish(ApprovalResponse(request_id=rid, approved=True, reason=""))

    async def _on_req(ev: ApprovalRequest) -> None:
        asyncio.create_task(_respond(ev.request_id))  # noqa: RUF006

    await bus.subscribe(ApprovalRequest, _on_req)

    async def _noop() -> None:
        pass

    await mw.process(_ctx("write_file", "filesystem.write", {"path": "/x"}), _noop)

    assert len(requests) == 1
    assert requests[0].judging is False  # no judge → dialog shows normal state
    assert reviews == []  # no auto-review ever published


class _CancelOnRequestedSink(FakeSink):
    """A sink whose request marker is interrupted, as an Esc on its write ack would be."""

    def __init__(self, judging: asyncio.Event) -> None:
        super().__init__()
        self._judging = judging

    async def emit(self, draft, *, payload_factory=None):  # type: ignore[no-untyped-def]
        if draft.event_type == EventType.APPROVAL_REQUESTED:
            await self._judging.wait()
            raise asyncio.CancelledError
        return await super().emit(draft, payload_factory=payload_factory)


class _WatchedJudge:
    """A judge that reports when it was reached and whether it was cancelled."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = False

    async def evaluate(self, **kwargs) -> JudgeVerdict:
        self.entered.set()
        try:
            await asyncio.sleep(ENGINE_TURN_TIMEOUT)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return JudgeVerdict(approved=True, reason="never reached")


@pytest.mark.asyncio
async def test_an_interrupted_request_marker_still_reaps_the_handler_and_the_judge(tmp_path) -> None:
    """The marker sits between the subscription and the block that reaps it."""
    bus = EventBus()
    judge = _WatchedJudge()
    mw = ApprovalMiddleware(
        approval_policy=_require_all_policy("filesystem.write"),
        event_bus=bus,
        approval_mode=ApprovalMode.AUTO,
        approval_judge=judge,
        approval_log_dir=tmp_path,
    )
    subscribed_before = len(bus._handlers[ApprovalResponse])

    async def _next() -> None:
        raise AssertionError("the tool must not run")

    with (
        trajectory_scope(make_context(_CancelOnRequestedSink(judge.entered))),
        pytest.raises(asyncio.CancelledError),
    ):
        await mw.process(_ctx("write_file", "filesystem.write", {"path": "/tmp/x"}), _next)

    # Nothing of this request outlives it: no handler left on the bus, no
    # judge left calling the model for a dialog that is already gone.
    assert len(bus._handlers[ApprovalResponse]) == subscribed_before
    assert judge.cancelled is True


@pytest.mark.asyncio
async def test_interrupting_a_pending_approval_records_its_resolution() -> None:
    """An abandoned request still resolves — a reader can never close it otherwise."""
    bus = EventBus()
    mw = ApprovalMiddleware(approval_policy=_require_all_policy(), event_bus=bus)
    requests: list[ApprovalRequest] = []

    async def _collect(ev: ApprovalRequest) -> None:
        requests.append(ev)

    await bus.subscribe(ApprovalRequest, _collect)

    async def _next() -> None:
        raise AssertionError("the tool must not run")

    sink = FakeSink()
    with trajectory_scope(make_context(sink)):
        task = asyncio.create_task(mw.process(_ctx("write_file", "shell", {"path": "x"}), _next))
        assert await wait_until(lambda: len(requests) >= 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    resolved = sink.only(EventType.APPROVAL_RESOLVED)
    assert resolved.payload["decision"] == ApprovalDecision.INTERRUPTED
    assert resolved.payload["decider"] == ApprovalDecider.NONE
    assert (
        resolved.payload["approval_request_id"]
        == sink.only(EventType.APPROVAL_REQUESTED).payload["approval_request_id"]
    )
    assert resolved.payload["wait_ms"] >= 0
