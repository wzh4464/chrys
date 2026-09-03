# Copyright (c) 2026 Chrys. All rights reserved.

"""Architecture layering checks for first-party imports."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, get_type_hints

import pytest

from chrys.service.agent_middleware.system_reminder import (
    CurrentRunReminderScope,
    CurrentRunReminderTarget,
    SystemReminderMiddleware,
)
from tests.support.ci import CI_LINUX_ONLY
from tests.support.paths import REPO_ROOT, SRC_ROOT

# Platform-independent source analysis: the Linux CI job covers it.
pytestmark = CI_LINUX_ONLY

ROOT = REPO_ROOT
SRC = SRC_ROOT / "chrys"

FOUNDATION = "foundation"
KERNEL = "kernel"
SERVICE = "service"
ORCHESTRATION = "orchestration"
APP = "app"
PACT = "pact"
ROOT_INIT = "__root__"

TIER_ORDER = {
    FOUNDATION: 0,
    KERNEL: 1,
    SERVICE: 2,
    ORCHESTRATION: 3,
    APP: 4,
    PACT: 4,
}

ROOT_METADATA_EXPORTS = {"__version__"}

_KERNEL_PRIVATE_PROMOTION_REASON = (
    "preview shaping / provider-specific key shared with service tier; promote to a public kernel module"
)
_KERNEL_PRIVATE_IMPORT_ALLOWLIST = {
    (
        Path("src/chrys/service/agent_middleware/events/tool_events.py"),
        "chrys.kernel._result_ceiling",
    ): _KERNEL_PRIVATE_PROMOTION_REASON,
    (
        Path("src/chrys/service/agent_middleware/events/sub_agent_events.py"),
        "chrys.kernel._result_ceiling",
    ): _KERNEL_PRIVATE_PROMOTION_REASON,
    (
        Path("src/chrys/service/llm/openai_chat_completion.py"),
        "chrys.kernel._types",
    ): _KERNEL_PRIVATE_PROMOTION_REASON,
    (
        Path("src/chrys/service/llm/anthropic_chat.py"),
        "chrys.kernel._types",
    ): _KERNEL_PRIVATE_PROMOTION_REASON,
    (
        Path("src/chrys/service/tools/result_metadata.py"),
        "chrys.kernel._serialization",
    ): _KERNEL_PRIVATE_PROMOTION_REASON,
    (
        Path("src/chrys/service/mcp/owned.py"),
        "chrys.kernel._tool_expansion",
    ): _KERNEL_PRIVATE_PROMOTION_REASON,
}


@dataclass(frozen=True)
class ImportEdge:
    source_path: Path
    source_top: str
    target: str
    target_top: str
    line: int
    is_relative: bool = False
    is_root_alias: bool = False


@cache
def _is_submodule(module: str, name: str) -> bool:
    """Report whether ``name`` names a real submodule file of ``module``."""
    if not module.startswith("chrys"):
        return False
    package = SRC.parent / Path(module.replace(".", "/"))
    return (package / f"{name}.py").is_file() or (package / name / "__init__.py").is_file()


class FirstPartyImportCollector(ast.NodeVisitor):
    """Collect first-party static and dynamic imports, excluding TYPE_CHECKING blocks."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.source_top = _source_top(path)
        self.edges: list[ImportEdge] = []

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_guard(node.test):
            for stmt in node.orelse:
                self.visit(stmt)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add_edge(alias.name, node.lineno)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level != 0:
            for target in _resolve_relative_import(self.path, node):
                self._add_edge(target, node.lineno, is_relative=True)
                # ``from ...kernel import _types`` resolves to the package
                # alone, exactly as the absolute form does. The bare ``from
                # ... import x`` spelling is already per-alias above, so only
                # the named-module spelling needs expanding here.
                if node.module is not None:
                    self._add_submodule_edges(target, node, is_relative=True)
            return
        if node.module is None:
            return
        if node.module == "chrys":
            for alias in node.names:
                if alias.name in ROOT_METADATA_EXPORTS:
                    continue
                self._add_edge(f"chrys.{alias.name}", node.lineno, is_root_alias=True)
            return
        self._add_edge(node.module, node.lineno)
        self._add_submodule_edges(node.module, node)

    def _add_submodule_edges(self, module: str, node: ast.ImportFrom, *, is_relative: bool = False) -> None:
        """Add an edge per alias that names a real submodule of *module*.

        ``from chrys.kernel import _types`` imports a module, not a member, yet
        the statement alone only names the package. Without resolving the alias
        against the tree, every module-level rule below sees the public package
        and the private submodule slips through.
        """
        for alias in node.names:
            if _is_submodule(module, alias.name):
                self._add_edge(f"{module}.{alias.name}", node.lineno, is_relative=is_relative)

    def visit_Call(self, node: ast.Call) -> None:
        if _call_name(node.func) in {"__import__", "import_module"} and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                self._add_edge(first_arg.value, node.lineno)
        self.generic_visit(node)

    def _add_edge(
        self,
        target: str,
        line: int,
        *,
        is_relative: bool = False,
        is_root_alias: bool = False,
    ) -> None:
        target_top = _target_top(target)
        if target_top is None:
            return
        self.edges.append(
            ImportEdge(
                source_path=self.path,
                source_top=self.source_top,
                target=target,
                target_top=target_top,
                line=line,
                is_relative=is_relative,
                is_root_alias=is_root_alias,
            )
        )


def _is_kernel_private_edge(path: Path, edge: ImportEdge) -> bool:
    return not path.is_relative_to(SRC / KERNEL) and edge.target.startswith("chrys.kernel._")


def _kernel_private_import_problem(path: Path, source: Path, edge: ImportEdge) -> str | None:
    """Return the violation text for an unexempted kernel-private import."""
    if not _is_kernel_private_edge(path, edge):
        return None
    if (source, edge.target) in _KERNEL_PRIVATE_IMPORT_ALLOWLIST:
        return None
    return (
        f"{source}:{edge.line}: kernel-private-modules-are-kernel-only forbids import of "
        f"{edge.target!r} outside src/chrys/kernel; violates AGENTS.md:54 (underscore-prefixed "
        "kernel modules are private helpers). Fix: import the public chrys.kernel facade, or promote "
        "the shared API to a public kernel module"
    )


def test_first_party_imports_follow_layer_dag() -> None:
    violations: list[str] = []
    observed_private_import_exemptions: set[tuple[Path, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        if _is_root_init(path):
            violations.extend(_root_init_violations(path))
            continue

        source_top = _source_top(path)
        if _tier_for_top(source_top) is None:
            violations.append(f"{path.relative_to(ROOT)}:1: unregistered first-party top-level package {source_top!r}")
            continue

        tree = _parse(path)
        collector = FirstPartyImportCollector(path)
        collector.visit(tree)
        for edge in collector.edges:
            source = edge.source_path.relative_to(ROOT)
            problem = _kernel_private_import_problem(path, source, edge)
            if problem is not None:
                violations.append(problem)
            elif _is_kernel_private_edge(path, edge):
                observed_private_import_exemptions.add((source, edge.target))
            if _is_violation(edge):
                violations.append(_format_violation(edge))

    for source, target in sorted(_KERNEL_PRIVATE_IMPORT_ALLOWLIST.keys() - observed_private_import_exemptions):
        reason = _KERNEL_PRIVATE_IMPORT_ALLOWLIST[(source, target)]
        violations.append(
            f"{source}:1: kernel private-import allowlist entry for {target!r} has no real import edge "
            f"(reason: {reason}); violates AGENTS.md:129 (\"A guard that can't go red is worse than none — it is "
            'believed."). Fix: remove the stale allowlist entry or update it to the exact live source/target pair'
        )

    assert violations == []


def test_relative_cross_tier_imports_are_checked() -> None:
    path = SRC / "service" / "llm" / "foo.py"
    tree = ast.parse("from ...orchestration.engine import engine\n", filename=str(path))
    collector = FirstPartyImportCollector(path)

    collector.visit(tree)

    # ``engine`` is itself a module, so the alias expands to a second edge.
    assert collector.edges == [
        ImportEdge(
            source_path=path,
            source_top=SERVICE,
            target="chrys.orchestration.engine",
            target_top=ORCHESTRATION,
            line=1,
            is_relative=True,
        ),
        ImportEdge(
            source_path=path,
            source_top=SERVICE,
            target="chrys.orchestration.engine.engine",
            target_top=ORCHESTRATION,
            line=1,
            is_relative=True,
        ),
    ]
    assert all(_is_violation(edge) for edge in collector.edges)


@pytest.mark.parametrize(
    "statement",
    [
        "from chrys.kernel import _types, Message",
        "from ...kernel import _types, Message",
    ],
)
def test_package_form_private_kernel_import_is_seen(statement: str) -> None:
    """A private submodule must not hide behind the package name, either spelling."""
    path = SRC / "service" / "llm" / "foo.py"
    tree = ast.parse(f"{statement}\n", filename=str(path))
    collector = FirstPartyImportCollector(path)

    collector.visit(tree)

    targets = [edge.target for edge in collector.edges]
    assert "chrys.kernel._types" in targets
    # ``Message`` is a re-exported member, not a module, so it stays unresolved.
    assert "chrys.kernel.Message" not in targets

    private_edge = next(edge for edge in collector.edges if edge.target == "chrys.kernel._types")
    problem = _kernel_private_import_problem(path, Path("src/chrys/service/llm/foo.py"), private_edge)
    assert problem is not None
    assert "chrys.kernel._types" in problem
    assert "AGENTS.md:54" in problem


def test_relative_same_tier_imports_are_allowed() -> None:
    path = SRC / "service" / "llm" / "foo.py"
    tree = ast.parse("from .clients import create_client\n", filename=str(path))
    collector = FirstPartyImportCollector(path)

    collector.visit(tree)

    assert collector.edges == [
        ImportEdge(
            source_path=path,
            source_top=SERVICE,
            target="chrys.service.llm.clients",
            target_top=SERVICE,
            line=1,
            is_relative=True,
        )
    ]
    assert not _is_violation(collector.edges[0])


def test_root_version_import_is_allowed() -> None:
    path = SRC / "app" / "cli" / "foo.py"
    tree = ast.parse("from chrys import __version__\n", filename=str(path))
    collector = FirstPartyImportCollector(path)

    collector.visit(tree)

    assert collector.edges == []


def test_root_non_tier_alias_import_is_rejected() -> None:
    path = SRC / "app" / "cli" / "foo.py"
    tree = ast.parse("from chrys import not_a_tier\n", filename=str(path))
    collector = FirstPartyImportCollector(path)

    collector.visit(tree)

    assert collector.edges == [
        ImportEdge(
            source_path=path,
            source_top=APP,
            target="chrys.not_a_tier",
            target_top="not_a_tier",
            line=1,
            is_root_alias=True,
        )
    ]
    assert _is_violation(collector.edges[0])


def test_root_tier_alias_import_is_classified() -> None:
    path = SRC / "app" / "cli" / "foo.py"
    tree = ast.parse("from chrys import kernel\n", filename=str(path))
    collector = FirstPartyImportCollector(path)

    collector.visit(tree)

    assert collector.edges == [
        ImportEdge(
            source_path=path,
            source_top=APP,
            target="chrys.kernel",
            target_top=KERNEL,
            line=1,
            is_root_alias=True,
        )
    ]
    assert not _is_violation(collector.edges[0])


def test_bare_root_imports_are_rejected_by_default() -> None:
    path = SRC / "app" / "cli" / "foo.py"
    sources = [
        "import chrys\n",
        "__import__('chrys')\n",
        "from importlib import import_module\nimport_module('chrys')\n",
    ]

    for source in sources:
        tree = ast.parse(source, filename=str(path))
        collector = FirstPartyImportCollector(path)
        collector.visit(tree)

        assert len(collector.edges) == 1
        assert collector.edges[0].target == "chrys"
        assert _is_violation(collector.edges[0])


def test_dynamic_subpackage_import_is_classified() -> None:
    path = SRC / "app" / "cli" / "foo.py"
    tree = ast.parse("from importlib import import_module\nimport_module('chrys.app.tui')\n", filename=str(path))
    collector = FirstPartyImportCollector(path)

    collector.visit(tree)

    assert collector.edges == [
        ImportEdge(
            source_path=path,
            source_top=APP,
            target="chrys.app.tui",
            target_top=APP,
            line=2,
        )
    ]
    assert not _is_violation(collector.edges[0])


def test_system_reminder_scoped_current_run_api_stays_service_owned_and_typed() -> None:
    """Scoped reminder tokens must stay service-owned and explicitly typed."""
    path = SRC / "service" / "agent_middleware" / "system_reminder.py"
    tree = _parse(path)
    orchestration_imports = [
        target
        for target in _import_targets_including_type_checking(path, tree)
        if target == "chrys.orchestration" or target.startswith("chrys.orchestration.")
    ]
    assert orchestration_imports == []

    assert get_type_hints(SystemReminderMiddleware.create_current_run_scope)["return"] is CurrentRunReminderScope

    capture_hints = get_type_hints(SystemReminderMiddleware.capture_current_run_target)
    assert capture_hints["reminder_scope"] is CurrentRunReminderScope
    assert capture_hints["return"] == CurrentRunReminderTarget | None

    queue_hints = get_type_hints(SystemReminderMiddleware.queue_hook_reminders_for_current_run)
    assert queue_hints["target"] is CurrentRunReminderTarget
    assert queue_hints["return"] is bool

    catalog_hints = get_type_hints(SystemReminderMiddleware.update_skill_catalog_for_current_run)
    assert catalog_hints["target"] is CurrentRunReminderTarget
    assert catalog_hints["return"] is bool

    set_catalog_hints = get_type_hints(SystemReminderMiddleware.set_skill_catalog_for_current_run)
    assert set_catalog_hints["target"] is CurrentRunReminderTarget
    assert set_catalog_hints["skill_catalog"] == str | None
    assert set_catalog_hints["return"] is bool

    valid_hints = get_type_hints(SystemReminderMiddleware.is_current_run_target_valid)
    assert valid_hints["target"] is CurrentRunReminderTarget
    assert valid_hints["return"] is bool

    expire_hints = get_type_hints(SystemReminderMiddleware.expire_current_run_scope)
    assert expire_hints["reminder_scope"] is CurrentRunReminderScope
    assert expire_hints["return"] is type(None)

    prepare_hints = get_type_hints(SystemReminderMiddleware.prepare_turn)
    assert prepare_hints["reminder_scope"] == CurrentRunReminderScope | None
    assert get_type_hints(CurrentRunReminderTarget)["scope"] is CurrentRunReminderScope

    scoped_api_names = {
        "create_current_run_scope",
        "capture_current_run_target",
        "queue_hook_reminders_for_current_run",
        "update_skill_catalog_for_current_run",
        "set_skill_catalog_for_current_run",
        "is_current_run_target_valid",
        "expire_current_run_scope",
        "prepare_turn",
    }
    for name in scoped_api_names:
        signature = inspect.signature(getattr(SystemReminderMiddleware, name))
        hints = get_type_hints(getattr(SystemReminderMiddleware, name))
        for parameter in signature.parameters.values():
            if parameter.name == "self":
                continue
            annotation = hints.get(parameter.name, parameter.annotation)
            assert annotation is not Any, f"{name}.{parameter.name} degraded to Any"
            assert annotation is not object, f"{name}.{parameter.name} degraded to object"
        return_annotation = hints.get("return", signature.return_annotation)
        assert return_annotation is not Any, f"{name} return degraded to Any"
        assert return_annotation is not object, f"{name} return degraded to object"


def _import_targets_including_type_checking(path: Path, tree: ast.AST) -> list[str]:
    """Collect static/dynamic first-party import targets without skipping TYPE_CHECKING."""
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                targets.extend(_resolve_relative_import(path, node))
            elif node.module == "chrys":
                targets.extend(f"chrys.{alias.name}" for alias in node.names)
            elif node.module is not None:
                targets.append(node.module)
        elif isinstance(node, ast.Call) and _call_name(node.func) in {"__import__", "import_module"} and node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                targets.append(first_arg.value)
    return [target for target in targets if target == "chrys" or target.startswith("chrys.")]


def _root_init_violations(path: Path) -> list[str]:
    tree = _parse(path)
    collector = FirstPartyImportCollector(path)
    collector.visit(tree)
    return [_format_violation(edge) for edge in collector.edges]


@cache
def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def teardown_module() -> None:
    """Release cached source trees after this worker finishes the module."""
    _parse.cache_clear()


def _is_violation(edge: ImportEdge) -> bool:
    source_tier = _tier_for_top(edge.source_top)
    target_tier = _tier_for_top(edge.target_top)
    if edge.is_root_alias and edge.target_top not in TIER_ORDER:
        return True
    if source_tier is None or target_tier is None:
        return True
    if source_tier == KERNEL and target_tier == KERNEL and not edge.is_relative:
        return True
    return TIER_ORDER[target_tier] > TIER_ORDER[source_tier]


def _format_violation(edge: ImportEdge) -> str:
    source = edge.source_path.relative_to(ROOT)
    source_tier = _tier_for_top(edge.source_top) or "unknown"
    target_tier = _tier_for_top(edge.target_top) or "unknown"
    root_alias = " root alias" if edge.is_root_alias else ""
    return (
        f"{source}:{edge.line}: {edge.source_top} ({source_tier}) "
        f"must not import{root_alias} {edge.target!r} ({edge.target_top}, {target_tier})"
    )


def _source_top(path: Path) -> str:
    rel = path.relative_to(SRC)
    first = rel.parts[0]
    if first == "__init__.py":
        return ROOT_INIT
    if first.endswith(".py"):
        return first.removesuffix(".py")
    return first


def _source_package(path: Path) -> str:
    rel = path.relative_to(SRC)
    package_parts = rel.parts[:-1]
    return ".".join(("chrys", *package_parts))


def _resolve_relative_import(path: Path, node: ast.ImportFrom) -> list[str]:
    package_parts = _source_package(path).split(".")
    if node.level > len(package_parts):
        return []
    base_parts = package_parts[: len(package_parts) - node.level + 1]
    base = ".".join(base_parts)
    if node.module is not None:
        return [f"{base}.{node.module}" if base else node.module]
    return [base if alias.name == "*" else f"{base}.{alias.name}" for alias in node.names]


def _target_top(name: str) -> str | None:
    if name == "chrys":
        return ROOT_INIT
    if not name.startswith("chrys."):
        return None
    parts = name.split(".")
    return parts[1] if len(parts) > 1 else ROOT_INIT


def _tier_for_top(top: str) -> str | None:
    return top if top in TIER_ORDER else None


def _is_root_init(path: Path) -> bool:
    return path == SRC / "__init__.py"


def _is_type_checking_guard(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "TYPE_CHECKING") or (
        isinstance(node, ast.Attribute)
        and node.attr == "TYPE_CHECKING"
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
    )


def _call_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""
