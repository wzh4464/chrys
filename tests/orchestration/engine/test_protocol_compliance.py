# Copyright (c) 2026 Chrys. All rights reserved.

"""Smoke tests that ``AgentEngine`` satisfies every ``*Host`` Protocol contract.

The handler-extraction refactors moved most engine logic into companion
modules (``engine_controls``, ``run/lifecycle``, ``run/sub_agents``,
``usage``, ``agent_lifecycle``, ``session/lifecycle``, ``rollback/controller``,
and the ``run/`` turn-phase modules) each of which declares a ``*Host``
``Protocol`` listing the engine state it needs.  The engine itself is passed
directly as every host (the forwarding adapter that used to sit between the
turn coordinator and the engine was removed).
Because Protocols are structural and these are not
``@runtime_checkable``, an engine attribute rename or a deleted method
would silently fail until the corresponding handler runs in production.

Three guards keep the contract chain honest, statically:

1. every Protocol member exists on ``AgentEngine`` (declared ⊆ engine);
2. every ``host.<attr>`` access in a run/ module is declared by that
   module's Protocol (accessed ⊆ declared);
3. on every host-forwarding edge (a service passing its host into a
   narrower service) the caller Protocol is a superset of the callee
   Protocol, and every forwarding call site maps to a registered edge.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import chrys.orchestration.engine.run as run_package
from chrys.foundation.config.settings import Settings
from chrys.foundation.events.bus import EventBus
from chrys.orchestration.engine.build.construction import AgentLifecycleHost
from chrys.orchestration.engine.engine import AgentEngine
from chrys.orchestration.engine.rollback import RollbackHost
from chrys.orchestration.engine.run.active_injection import ActiveInjectionHost
from chrys.orchestration.engine.run.coordinator import TurnCoordinatorHost
from chrys.orchestration.engine.run.finalizer import TurnFinalizerHost
from chrys.orchestration.engine.run.prompt_content import PromptContentHost
from chrys.orchestration.engine.run.retry import RetryHost
from chrys.orchestration.engine.run.runner import TurnRunnerHost
from chrys.orchestration.engine.run.runtime_skills import RuntimeSkillHost
from chrys.orchestration.engine.run.sub_agent_coordination import SubAgentCoordinationHost
from chrys.orchestration.engine.run.turn_hooks import TurnHookHost
from chrys.orchestration.engine.state.controls import EngineControlHost
from chrys.service.session.lifecycle import SessionLifecycleHost
from chrys.service.usage import UsageHost

_PROTOCOLS = [
    AgentLifecycleHost,
    RollbackHost,
    EngineControlHost,
    SessionLifecycleHost,
    SubAgentCoordinationHost,
    UsageHost,
    TurnCoordinatorHost,
    TurnRunnerHost,
    RetryHost,
    ActiveInjectionHost,
    TurnFinalizerHost,
    PromptContentHost,
    TurnHookHost,
    RuntimeSkillHost,
]

_RUN_DIR = Path(run_package.__file__).parent

# run/ module → the Host Protocol its ``host`` objects are typed against.
_MODULE_HOSTS: dict[str, type] = {
    "coordinator.py": TurnCoordinatorHost,
    "runner.py": TurnRunnerHost,
    "retry.py": RetryHost,
    "active_injection.py": ActiveInjectionHost,
    "finalizer.py": TurnFinalizerHost,
    "prompt_content.py": PromptContentHost,
    "turn_hooks.py": TurnHookHost,
    "runtime_skills.py": RuntimeSkillHost,
    "sub_agent_coordination.py": SubAgentCoordinationHost,
}

# Callable name → the Host Protocol its host parameter is typed against.
# Used to discover host-forwarding call sites (``Service(host)`` /
# ``function(self._host, ...)``) in run/ module source.
_SERVICE_HOSTS: dict[str, type] = {
    "TurnCoordinator": TurnCoordinatorHost,
    "TurnRunner": TurnRunnerHost,
    "RetryCoordinator": RetryHost,
    "ActiveTurnInjector": ActiveInjectionHost,
    "TurnFinalizer": TurnFinalizerHost,
    "_expire_current_run_scope": TurnFinalizerHost,
    "_post_compaction_history_start_index": TurnFinalizerHost,
    "_clear_post_compaction_history_start_index": TurnFinalizerHost,
    "PromptContentPreparer": PromptContentHost,
    "ImageHistoryPolicy": PromptContentHost,
    "prompt_content_preparer": PromptContentHost,
    "PromptSubmitGate": TurnHookHost,
    "TurnHookDispatcher": TurnHookHost,
    "_workspace_cwd": TurnHookHost,
    "RuntimeSkillRefresher": RuntimeSkillHost,
    "on_sub_agent_retry": SubAgentCoordinationHost,
    "on_sub_agent_abort": SubAgentCoordinationHost,
    "on_sub_agent_paused": SubAgentCoordinationHost,
    "clear_parent_paused_state": SubAgentCoordinationHost,
}

# Registered host-forwarding edges: caller Protocol → callee Protocols it
# forwards its host into.  Every edge discovered in source must be listed
# here; edges may also be listed without a directly-scannable call site
# (e.g. ``TurnRunner``'s default prompt-content factory is the
# ``PromptContentPreparer`` class itself, invoked through an attribute).
_FORWARDING_EDGES: dict[type, frozenset[type]] = {
    TurnCoordinatorHost: frozenset(
        {
            TurnRunnerHost,
            RetryHost,
            ActiveInjectionHost,
            TurnHookHost,
            PromptContentHost,
            TurnFinalizerHost,
        }
    ),
    TurnRunnerHost: frozenset(
        {
            TurnFinalizerHost,
            TurnHookHost,
            RuntimeSkillHost,
            RetryHost,
            PromptContentHost,
        }
    ),
    RetryHost: frozenset({ActiveInjectionHost, TurnHookHost, PromptContentHost, RuntimeSkillHost}),
    ActiveInjectionHost: frozenset({TurnHookHost, PromptContentHost, RuntimeSkillHost}),
    TurnFinalizerHost: frozenset({TurnHookHost, SubAgentCoordinationHost}),
}

_EDGES = [
    (caller, callee)
    for caller, callees in _FORWARDING_EDGES.items()
    for callee in sorted(callees, key=lambda c: c.__name__)
]


def _required_names(proto: type) -> list[str]:
    """Collect every name a Protocol expects on its host, including inherited.

    Walks the MRO (chrys-defined classes only, skipping ``Protocol`` /
    ``Generic`` / ``object`` machinery) and picks up:
    - attribute annotations declared at class scope
    - regular and async methods defined inside the Protocol body
    - properties declared with ``@property``
    """
    names: set[str] = set()
    for klass in proto.__mro__:
        if not klass.__module__.startswith("chrys."):
            continue
        names.update(getattr(klass, "__annotations__", {}))
        for name, value in klass.__dict__.items():
            if name.startswith("__"):
                continue
            if callable(value) or isinstance(value, property):
                names.add(name)
    return sorted(names)


def _is_host_reference(node: ast.expr) -> bool:
    """Return whether *node* is the conventional host reference in run/ code."""
    if isinstance(node, ast.Name) and node.id == "host":
        return True
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "_host"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _host_attribute_accesses(source: str) -> set[str]:
    """Collect first-level attribute names accessed on ``host``/``self._host``."""
    accessed: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and _is_host_reference(node.value):
            accessed.add(node.attr)
    return accessed


def _host_forwarding_callees(source: str) -> set[str]:
    """Collect mapped callable names invoked with ``host``/``self._host`` as an argument."""
    callees: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name not in _SERVICE_HOSTS:
            continue
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
        if any(_is_host_reference(argument) for argument in arguments):
            callees.add(name)
    return callees


def _make_engine() -> AgentEngine:
    return AgentEngine(EventBus(), settings=Settings())


@pytest.mark.parametrize("protocol", _PROTOCOLS, ids=lambda p: p.__name__)
def test_agent_engine_carries_every_protocol_member(protocol: type) -> None:
    """Each ``*Host`` Protocol's required members must exist on ``AgentEngine``."""
    engine = _make_engine()
    missing = [name for name in _required_names(protocol) if not hasattr(engine, name)]
    assert missing == [], (
        f"{protocol.__name__} requires members the engine does not expose: {missing}. "
        "An attr or method was likely renamed or deleted without updating the Protocol."
    )


@pytest.mark.parametrize("filename", sorted(_MODULE_HOSTS), ids=lambda f: f.removesuffix(".py"))
def test_run_module_host_accesses_are_declared(filename: str) -> None:
    """Every ``host.<attr>`` access in a run/ module is declared by its Protocol.

    This replaces the deleted live-port's only real safety property: with the
    engine passed directly as host, an access to an undeclared engine member
    would silently work at runtime and rot the Protocol into an understatement
    of the real contract.  Fix failures by declaring the member on the
    module's Host Protocol, not by rewriting the access.
    """
    protocol = _MODULE_HOSTS[filename]
    source = (_RUN_DIR / filename).read_text(encoding="utf-8")
    undeclared = sorted(_host_attribute_accesses(source) - set(_required_names(protocol)))
    assert undeclared == [], (
        f"{filename} accesses host members not declared on {protocol.__name__}: {undeclared}. "
        "Declare them on the Protocol so the engine-compliance guard covers them."
    )


@pytest.mark.parametrize(("caller", "callee"), _EDGES, ids=lambda p: p.__name__)
def test_forwarding_edge_caller_covers_callee(caller: type, callee: type) -> None:
    """On every host-forwarding edge the caller Protocol must cover the callee's.

    Run services pass their own host object into narrower services
    (e.g. ``TurnRunner`` hands ``self._host`` to ``RuntimeSkillRefresher``),
    so a member only the callee uses must still be part of the caller's
    contract — otherwise deleting it from the caller Protocol would pass the
    access guard while breaking the callee at runtime.
    """
    missing = sorted(set(_required_names(callee)) - set(_required_names(caller)))
    assert missing == [], f"{caller.__name__} forwards its host into {callee.__name__} but does not declare: {missing}."


def test_every_discovered_forwarding_call_site_has_a_registered_edge() -> None:
    """Every ``Service(host)`` call site in run/ maps to a registered forwarding edge.

    Keeps ``_FORWARDING_EDGES`` from drifting: adding a new host-forwarding
    call without registering the edge (and thus without superset coverage)
    fails here.  Self-edges (a module invoking helpers bound to its own
    Protocol) are trivially covered and skipped.
    """
    unregistered: list[str] = []
    for filename, caller in _MODULE_HOSTS.items():
        source = (_RUN_DIR / filename).read_text(encoding="utf-8")
        for callee_name in sorted(_host_forwarding_callees(source)):
            callee = _SERVICE_HOSTS[callee_name]
            if callee is caller:
                continue
            if callee not in _FORWARDING_EDGES.get(caller, frozenset()):
                unregistered.append(f"{filename}: {caller.__name__} -> {callee_name} ({callee.__name__})")
    assert unregistered == [], (
        f"Host-forwarding call sites without a registered edge in _FORWARDING_EDGES: {unregistered}"
    )


def test_no_protocol_is_empty() -> None:
    """Sanity check: protocol-name set is non-empty for every host.

    Catches the failure mode where an empty or malformed Protocol would
    silently pass the membership test above.
    """
    for protocol in _PROTOCOLS:
        names = _required_names(protocol)
        assert names, f"{protocol.__name__} declares no members — refactor probably removed them"


def test_runtime_meta_is_protocol_required_by_relevant_hosts() -> None:
    """The post-refactor metadata bundle must be on every Host that touches it.

    If someone re-introduces ``_total_session_tokens`` etc. as separate engine
    attrs, the corresponding Protocol won't declare them and this regression
    will be invisible.  Locking the contract here makes that drift loud.
    ``TurnRunnerHost`` declares ``_runtime_meta`` for executor passes, and
    ``TurnCoordinatorHost`` inherits it through ``TurnRunnerHost``.
    """
    expected = {
        "UsageHost",
        "AgentLifecycleHost",
        "SessionLifecycleHost",
        "TurnRunnerHost",
        "TurnCoordinatorHost",
    }
    actual = {p.__name__ for p in _PROTOCOLS if "_runtime_meta" in _required_names(p)}
    assert actual == expected, (
        f"_runtime_meta presence diverged from expected hosts. "
        f"Expected exactly {sorted(expected)}, got {sorted(actual)}."
    )


def test_no_protocol_still_declares_legacy_token_attrs() -> None:
    """Legacy attrs replaced by ``SessionRuntimeMetadata`` must be gone from every Host."""
    legacy = {
        "_total_session_tokens",
        "_total_session_input_tokens",
        "_total_session_output_tokens",
        "_last_usage_details",
    }
    leaks: dict[str, set[str]] = {}
    for protocol in _PROTOCOLS:
        names = set(_required_names(protocol))
        intersection = legacy & names
        if intersection:
            leaks[protocol.__name__] = intersection
    assert not leaks, f"Protocols still declare legacy token/usage attrs that were folded into _runtime_meta: {leaks}"


def test_no_protocol_still_declares_turn_state_compat_attrs() -> None:
    """Engine compat projections of ``TurnRuntimeState`` must stay gone from every Host.

    The engine-side compatibility properties were deleted during the turn-state
    migration; run/ services read ``host._turn_state.*`` directly. Re-declaring one of
    these names on a Protocol would resurrect the dual-storage drift the
    migration removed (fakes holding the attr separately from turn state).
    """
    compat = {
        "_run_task",
        "_pending_retry_text",
        "_pending_retry_created_at",
        "_run_history_start_index",
        "_current_turn_text",
        "_current_turn_contents",
        "_current_turn_created_at",
    }
    leaks: dict[str, set[str]] = {}
    for protocol in _PROTOCOLS:
        names = set(_required_names(protocol))
        intersection = compat & names
        if intersection:
            leaks[protocol.__name__] = intersection
    assert not leaks, f"Protocols re-declare deleted turn-state compatibility attrs: {leaks}"
